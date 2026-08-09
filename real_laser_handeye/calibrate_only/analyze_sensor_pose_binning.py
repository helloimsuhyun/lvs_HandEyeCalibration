"""Analyze two captured sensor-pose distributions relative to their planes.

VERSION: selected-calibration-server-v5-minus-z-tilt

The report includes U/V/d pose coverage, -Z-to-normal tilt/azimuth/sensor-roll binning,
non-empty four-dimensional joint bins, and the fitted-plane normal expressed
in each sensor frame. Numerical self-checks verify rotation reconstruction,
plane-normal transforms, signed distances, diagnostic counts, and bin totals.

Example
-------
PYTHONPATH=. python3 \
  real_laser_handeye/calibrate_only/analyze_sensor_pose_binning.py

The default arguments compare ``runs/real/backup/dataset`` with
``runs/real/dataset`` and use the corresponding calibrated hand-eye transform
and final fitted-plane diagnostics for each dataset. The physical sensor origin
is modeled 80 mm along sensor -Z from the measurement-coordinate origin.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import io
import json
import math
import threading
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any

import numpy as np


BACKUP_COLOR = "#4C78A8"
CURRENT_COLOR = "#F58518"
REJECTED_COLOR = "#D62728"
WORLD_SCAN_MAX_POINTS = 50_000


@dataclass(frozen=True)
class WorldScanSpec:
    scan_id: str
    label: str
    path: Path


DEFAULT_WORLD_SCANS = (
    WorldScanSpec("block", "Block", Path("runs/real/block.npz")),
    WorldScanSpec("coin", "Coin", Path("runs/real/coin.npz")),
    WorldScanSpec("lenovo", "Lenovo", Path("runs/real/lenovo.npz")),
    WorldScanSpec(
        "manual_world_scan",
        "Manual world scan",
        Path("runs/real/manual_world_scan.npz"),
    ),
    WorldScanSpec(
        "two_point_stop_and_scan",
        "Two-point stop-and-scan",
        Path("runs/real/two_point_stop_and_scan.npz"),
    ),
)


@dataclass(frozen=True)
class DatasetSpec:
    label: str
    dataset_dir: Path
    transform_path: Path
    diagnostics_path: Path
    color: str
    sensor_origin_offset_z_mm: float = -80.0


@dataclass
class PoseDistribution:
    spec: DatasetSpec
    capture_names: list[str]
    scan_ids: np.ndarray
    accepted: np.ndarray
    profiles_sensor_xyz_mm: list[np.ndarray]
    profile_source_keys: list[str]
    origins_plane_mm: np.ndarray
    measurement_origins_base_mm: np.ndarray
    physical_origins_base_mm: np.ndarray
    rotations_base_sensor: np.ndarray
    rotations_plane_sensor: np.ndarray
    plane_normals_sensor: np.ndarray
    normal_tilt_deg: np.ndarray
    normal_azimuth_sensor_deg: np.ndarray
    tilt_deg: np.ndarray
    azimuth_deg: np.ndarray
    roll_deg: np.ndarray
    plane_centroid_base_mm: np.ndarray
    plane_normal_base: np.ndarray
    plane_basis_base: np.ndarray
    calibration_scan_count: int | None
    calibration_rejected_count: int | None


@dataclass(frozen=True)
class BinningConfig:
    tilt_edges_deg: np.ndarray
    azimuth_edges_deg: np.ndarray
    distance_edges_mm: np.ndarray
    roll_edges_deg: np.ndarray
    accepted_only: bool = True


@dataclass
class CalibrationContext:
    spec: DatasetSpec
    initial_transform: np.ndarray
    full_transform: np.ndarray
    full_diagnostics: dict[str, Any]
    scans_by_id: dict[int, Any]
    all_scan_ids: set[int]
    ransac_rejected_ids: set[int]
    lock: Any
    cache: dict[tuple[int, ...], dict[str, Any]]


def _wrap_degrees(values: np.ndarray) -> np.ndarray:
    """Wrap angles to [-180, 180)."""
    return (np.asarray(values, dtype=float) + 180.0) % 360.0 - 180.0


def _euler_tilt_azimuth_roll(
    rotations_plane_sensor: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decompose orientation as Rz(azimuth) @ Ry(euler_tilt) @ Rz(roll).

    ``euler_tilt`` is only the internal Z-Y-Z decomposition angle measured
    from the plane normal (+W) to sensor +Z.  The reported pose ``tilt`` is
    computed separately as the angle between the oriented plane normal in
    the sensor frame and sensor -Z.
    """
    rotations = np.asarray(rotations_plane_sensor, dtype=float)
    sensor_z = rotations[:, :, 2]
    sensor_x = rotations[:, :, 0]

    tilt_rad = np.arccos(np.clip(sensor_z[:, 2], -1.0, 1.0))
    azimuth_rad = np.arctan2(sensor_z[:, 1], sensor_z[:, 0])

    cos_a = np.cos(azimuth_rad)
    sin_a = np.sin(azimuth_rad)
    cos_t = np.cos(tilt_rad)
    sin_t = np.sin(tilt_rad)
    zero_roll_x = np.column_stack(
        [cos_a * cos_t, sin_a * cos_t, -sin_t]
    )
    zero_roll_y = np.column_stack([-sin_a, cos_a, np.zeros_like(cos_a)])
    roll_rad = np.arctan2(
        np.einsum("ij,ij->i", sensor_x, zero_roll_y),
        np.einsum("ij,ij->i", sensor_x, zero_roll_x),
    )

    return (
        np.degrees(tilt_rad),
        _wrap_degrees(np.degrees(azimuth_rad)),
        _wrap_degrees(np.degrees(roll_rad)),
    )


def _validate_transform(value: np.ndarray, name: str) -> np.ndarray:
    transform = np.asarray(value, dtype=float)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError(f"{name} has an invalid homogeneous last row")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError(f"{name} rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
        raise ValueError(f"{name} rotation determinant must be +1")
    return transform.copy()


def _load_transform(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"hand-eye transform not found: {path}")
    try:
        value = np.loadtxt(path, delimiter=",")
    except ValueError:
        value = np.loadtxt(path)
    return _validate_transform(value, str(path))


def _world_scan_capture_indices(payload: Any) -> list[int]:
    suffix = "_points_sensor"
    indices: list[int] = []
    for key in payload.files:
        if not key.startswith("capture_") or not key.endswith(suffix):
            continue
        index_text = key[len("capture_") : -len(suffix)]
        try:
            indices.append(int(index_text))
        except ValueError:
            continue
    return sorted(indices)


def reproject_world_scan_for_browser(
    spec: WorldScanSpec,
    transform_tcp_sensor: np.ndarray,
    *,
    max_points: int,
) -> dict[str, Any]:
    """Reproject a saved scan and return a bounded browser visualization payload."""
    transform_tcp_sensor = _validate_transform(
        transform_tcp_sensor, "selected T_tcp_sensor"
    )
    if max_points <= 0:
        raise ValueError("world-scan max points must be positive")
    if not spec.path.is_file():
        raise FileNotFoundError(f"world scan not found: {spec.path}")

    world_groups: list[np.ndarray] = []
    with np.load(spec.path, allow_pickle=False) as payload:
        if "T_base_tcp" not in payload:
            raise KeyError(f"{spec.path} does not contain T_base_tcp")
        capture_indices = _world_scan_capture_indices(payload)
        if not capture_indices:
            raise KeyError(
                f"{spec.path} does not contain capture_*_points_sensor arrays"
            )
        transforms_base_tcp = np.asarray(payload["T_base_tcp"], dtype=float)
        if transforms_base_tcp.shape != (len(capture_indices), 4, 4):
            raise ValueError(
                f"{spec.path}: T_base_tcp shape {transforms_base_tcp.shape} "
                f"does not match {len(capture_indices)} captures"
            )

        for pose_index, capture_index in enumerate(capture_indices):
            points_sensor = np.asarray(
                payload[f"capture_{capture_index:04d}_points_sensor"],
                dtype=float,
            )
            if points_sensor.ndim != 2 or points_sensor.shape[1] != 3:
                raise ValueError(
                    f"{spec.path}: capture {capture_index} has invalid point "
                    f"shape {points_sensor.shape}"
                )
            finite = np.all(np.isfinite(points_sensor), axis=1)
            points_sensor = points_sensor[finite]
            if not len(points_sensor):
                continue
            transform_base_sensor = (
                transforms_base_tcp[pose_index] @ transform_tcp_sensor
            )
            points_world = (
                points_sensor @ transform_base_sensor[:3, :3].T
                + transform_base_sensor[:3, 3]
            )
            world_groups.append(points_world)

    if not world_groups:
        raise RuntimeError(f"{spec.path}: no finite sensor points")
    points_world = np.vstack(world_groups)
    total_count = int(len(points_world))
    if total_count > max_points:
        sample_indices = np.linspace(
            0, total_count - 1, num=max_points, dtype=np.int64
        )
        displayed = points_world[sample_indices]
    else:
        displayed = points_world

    world_min = np.min(points_world, axis=0)
    world_max = np.max(points_world, axis=0)
    padding = np.maximum(2.0, 0.04 * np.maximum(world_max - world_min, 1.0))
    ranges = np.column_stack((world_min - padding, world_max + padding))
    displayed = np.round(displayed, 4)
    result: dict[str, Any] = {
        "success": True,
        "scan": {
            "id": spec.scan_id,
            "label": spec.label,
            "file": spec.path.name,
            "path": str(spec.path),
            "capture_count": len(capture_indices),
            "point_count": total_count,
            "displayed_point_count": int(len(displayed)),
        },
        "points_world": {
            "x": displayed[:, 0].tolist(),
            "y": displayed[:, 1].tolist(),
            "z": displayed[:, 2].tolist(),
        },
        "ranges_world": np.round(ranges, 4).tolist(),
        "transform_convention": (
            "points_world = T_base_tcp @ T_tcp_sensor @ points_sensor"
        ),
    }
    if spec.scan_id == "two_point_stop_and_scan":
        result["sphere_evaluation"] = evaluate_sphere_world_groups(world_groups)
    return result


def evaluate_sphere_world_groups(
    world_groups: list[np.ndarray],
) -> dict[str, Any]:
    """Run fit_saved_scan_sphere defaults with the requested z filtering."""
    from real_laser_handeye.laser_scan_demo import fit_saved_scan_sphere as sphere

    clean_groups = [
        np.ascontiguousarray(group[np.all(np.isfinite(group), axis=1)], dtype=float)
        for group in world_groups
    ]
    clean_groups = [group for group in clean_groups if len(group)]
    names = [f"capture_{index:04d}" for index in range(len(clean_groups))]
    original = sphere._make_loaded(clean_groups, names)
    floor_detection = sphere.detect_floor_z_min(
        original,
        known_radius_mm=5.0,
        profile_quantile=0.10,
        clearance_mm=None,
    )

    cropped_groups: list[np.ndarray] = []
    cropped_names: list[str] = []
    for name, group in zip(original.group_names, original.groups):
        cropped = group[group[:, 2] >= floor_detection.z_min_mm]
        if len(cropped):
            cropped_groups.append(cropped)
            cropped_names.append(name)
    cropped = sphere._make_loaded(cropped_groups, cropped_names)
    z_clean, group_stats, z_min_removed = sphere.remove_profile_z_min(
        cropped,
        margin_mm=0.0,
        min_points_per_group=4,
    )
    precleaned, profile_spike_removed = sphere.remove_profile_spikes(
        z_clean,
        absolute_threshold_mm=0.5,
        sigma=6.0,
        max_run_length=3,
        min_points_per_group=4,
        group_stats=group_stats,
    )
    free_result = sphere.fit_free_radius_sphere(
        precleaned.merged_points,
        expected_radius_mm=5.0,
        ransac_iterations=3000,
        ransac_threshold_mm=0.4,
        candidate_radius_tolerance_mm=2.0,
        max_iterations=100,
        tolerance_mm=1e-6,
        huber_delta_mm=0.2,
        final_inlier_threshold_mm=0.5,
        min_inliers=30,
        seed=1701,
    )
    fixed_result = sphere.fit_fixed_radius_sphere(
        precleaned.merged_points,
        radius_mm=5.0,
        initial_center=free_result.center,
        max_iterations=100,
        tolerance_mm=1e-6,
        huber_delta_mm=0.2,
        final_inlier_threshold_mm=0.5,
        min_inliers=30,
    )
    filtered = sphere.apply_merged_mask_to_groups(
        precleaned,
        fixed_result.inlier_mask,
        min_points_per_group=4,
        min_profile_inlier_ratio=0.20,
        group_stats=group_stats,
    )
    free_payload = sphere.fit_payload(free_result)
    fixed_payload = sphere.fit_payload(fixed_result)
    return {
        "title": "5 mm sphere-fit evaluation",
        "settings": {
            "z_min_mm": floor_detection.z_min_mm,
            "z_min_margin_mm": 0.0,
            "point_size": 2.0,
            "known_radius_mm": 5.0,
            "ransac_threshold_mm": 0.4,
            "inlier_threshold_mm": 0.5,
            "auto_floor_z_min": True,
            "floor_z_mm": floor_detection.floor_z_mm,
            "floor_clearance_mm": floor_detection.clearance_mm,
            "floor_profile_quantile": floor_detection.profile_quantile,
            "floor_robust_sigma_mm": floor_detection.robust_sigma_mm,
            "floor_inlier_profiles": floor_detection.inlier_profile_count,
            "floor_candidate_profiles": floor_detection.candidate_profile_count,
        },
        "preprocessing": {
            "input_points": int(len(original.merged_points)),
            "after_crop_points": int(len(cropped.merged_points)),
            "z_min_removed_points": int(z_min_removed),
            "profile_spike_removed_points": int(profile_spike_removed),
            "sphere_outlier_removed_points": int(
                len(precleaned.merged_points) - len(filtered.merged_points)
            ),
            "final_points": int(len(filtered.merged_points)),
            "input_profile_groups": int(len(original.groups)),
            "final_profile_groups": int(len(filtered.groups)),
        },
        "free_radius_fit": free_payload,
        "fixed_radius_fit": fixed_payload,
        "comparison": {
            "radius_error_mm": float(free_result.radius_mm - 5.0),
            "diameter_error_mm": float(2.0 * (free_result.radius_mm - 5.0)),
            "center_difference_mm": float(
                np.linalg.norm(free_result.center - fixed_result.center)
            ),
        },
    }


def _capture_scan_id(path: Path, fallback: int) -> int:
    try:
        return int(path.stem.rsplit("_", 1)[-1])
    except ValueError:
        return fallback


_PROFILE_KEY_PRIORITY = (
    "points_sensor",
    "profile_points_sensor",
    "points_sensor_mm",
    "profile_sensor",
    "profile_points",
    "mean_profile_sensor",
    "mean_profile",
    "latest_profile",
    "points",
)


def _profile_candidate_score(key: str) -> tuple[int, int]:
    """Return a deterministic preference score for a profile-point key."""
    lower = key.lower()
    exact_rank = len(_PROFILE_KEY_PRIORITY)
    for index, candidate in enumerate(_PROFILE_KEY_PRIORITY):
        if lower == candidate or lower.endswith("_" + candidate):
            exact_rank = index
            break
    semantic_penalty = 0
    if "sensor" not in lower:
        semantic_penalty += 4
    if "profile" not in lower:
        semantic_penalty += 2
    if "point" not in lower:
        semantic_penalty += 2
    if any(token in lower for token in ("base", "tcp", "transform", "normal")):
        semantic_penalty += 20
    return exact_rank, semantic_penalty


def _extract_profile_sensor_xyz(
    capture: Any,
    path: Path,
) -> tuple[np.ndarray, str]:
    """Extract one sensor-frame profile as an ``N x 3`` [X, Y, Z] array.

    The loader first prefers conventional sensor/profile point keys, then
    inspects other numeric arrays whose names contain ``point`` or ``profile``.
    Three-dimensional points preserve X, Y and Z. Two-dimensional points are
    interpreted as [X, Z] and embedded in the laser plane with Y=0. Arrays
    with fewer than five points are ignored so transforms and short metadata
    arrays are not misidentified.
    """
    candidates: list[tuple[tuple[int, int], str, np.ndarray]] = []
    for key in capture.files:
        lower = key.lower()
        if not (
            key in _PROFILE_KEY_PRIORITY
            or any(lower.endswith("_" + item) for item in _PROFILE_KEY_PRIORITY)
            or "point" in lower
            or "profile" in lower
        ):
            continue
        try:
            array = np.asarray(capture[key])
        except Exception:
            continue
        if not np.issubdtype(array.dtype, np.number) or array.ndim != 2:
            continue
        if array.shape[0] in (2, 3, 4) and array.shape[1] > array.shape[0]:
            array = array.T
        if array.shape[0] < 5 or array.shape[1] not in (2, 3, 4):
            continue
        array = np.asarray(array, dtype=float)
        if array.shape[1] == 2:
            xyz = np.column_stack(
                [array[:, 0], np.zeros(array.shape[0]), array[:, 1]]
            )
        else:
            xyz = array[:, :3]
        finite = np.all(np.isfinite(xyz), axis=1)
        xyz = xyz[finite]
        if len(xyz) < 5:
            continue
        candidates.append((_profile_candidate_score(key), key, xyz))

    if not candidates:
        return np.empty((0, 3), dtype=float), ""
    candidates.sort(key=lambda item: (item[0], item[1]))
    _, key, xyz = candidates[0]
    return xyz.copy(), key


def _downsample_profile(profile: np.ndarray, max_points: int) -> np.ndarray:
    profile = np.asarray(profile, dtype=float)
    if max_points <= 0 or len(profile) <= max_points:
        return profile
    indices = np.linspace(0, len(profile) - 1, max_points).round().astype(int)
    return profile[np.unique(indices)]


def _plane_frame(
    normal_base: np.ndarray,
    centroid_base: np.ndarray,
    sensor_origins_base: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a repeatable plane basis with columns [u, v, normal]."""
    normal = np.asarray(normal_base, dtype=float).reshape(3)
    normal /= np.linalg.norm(normal)

    # The fitted plane normal has arbitrary sign. Point it toward the average
    # sensor reference origin so signed distance is comparable between runs.
    mean_sensor_offset = np.mean(sensor_origins_base, axis=0) - centroid_base
    if float(mean_sensor_offset @ normal) < 0.0:
        normal = -normal

    # Project base +X onto the plane to define a repeatable in-plane U axis.
    # Base +Y is only needed for the degenerate case where normal ~= base +X.
    reference = np.array([1.0, 0.0, 0.0])
    u_axis = reference - normal * float(reference @ normal)
    if np.linalg.norm(u_axis) < 1e-8:
        reference = np.array([0.0, 1.0, 0.0])
        u_axis = reference - normal * float(reference @ normal)
    u_axis /= np.linalg.norm(u_axis)
    if float(u_axis @ reference) < 0.0:
        u_axis = -u_axis

    v_axis = np.cross(normal, u_axis)
    v_axis /= np.linalg.norm(v_axis)
    return np.column_stack([u_axis, v_axis, normal]), normal


def load_pose_distribution(spec: DatasetSpec) -> PoseDistribution:
    capture_paths = list(spec.dataset_dir.glob("capture_*.npz"))
    capture_paths.sort(key=lambda path: (_capture_scan_id(path, 10**12), path.name))
    if not capture_paths:
        raise RuntimeError(f"no capture_*.npz files found in {spec.dataset_dir}")

    transform = _load_transform(spec.transform_path)
    diagnostics = json.loads(spec.diagnostics_path.read_text(encoding="utf-8"))
    plane = diagnostics.get("final_point_to_plane")
    if not isinstance(plane, dict):
        raise KeyError(
            f"{spec.diagnostics_path} has no final_point_to_plane diagnostics"
        )

    normal = np.asarray(plane["plane_normal_base"], dtype=float).reshape(3)
    centroid = np.asarray(plane["plane_centroid_base_mm"], dtype=float).reshape(3)
    rejected_ids = {int(value) for value in diagnostics.get("rejected_scan_ids", [])}

    capture_names: list[str] = []
    scan_ids: list[int] = []
    origins: list[np.ndarray] = []
    rotations: list[np.ndarray] = []
    profiles_sensor_xyz_mm: list[np.ndarray] = []
    profile_source_keys: list[str] = []
    for fallback_id, path in enumerate(capture_paths, start=1):
        with np.load(path, allow_pickle=False) as capture:
            if "T_base_tcp" not in capture:
                raise KeyError(f"{path} does not contain T_base_tcp")
            T_base_tcp = _validate_transform(capture["T_base_tcp"], str(path))
            profile_xyz, profile_key = _extract_profile_sensor_xyz(capture, path)
        T_base_sensor = T_base_tcp @ transform
        capture_names.append(path.name)
        scan_ids.append(_capture_scan_id(path, fallback_id))
        origins.append(T_base_sensor[:3, 3].copy())
        rotations.append(T_base_sensor[:3, :3].copy())
        profiles_sensor_xyz_mm.append(profile_xyz)
        profile_source_keys.append(profile_key)

    measurement_origins_base = np.asarray(origins, dtype=float)
    rotations_base_sensor = np.asarray(rotations, dtype=float)
    origins_base = (
        measurement_origins_base
        + float(spec.sensor_origin_offset_z_mm) * rotations_base_sensor[:, :, 2]
    )
    basis, oriented_normal = _plane_frame(normal, centroid, origins_base)
    origins_plane = (origins_base - centroid) @ basis
    rotations_plane_sensor = np.einsum(
        "ij,njk->nik", basis.T, rotations_base_sensor
    )

    _euler_tilt_deg, azimuth_deg, roll_deg = _euler_tilt_azimuth_roll(
        rotations_plane_sensor
    )

    # Oriented fitted-plane normal expressed in each sensor frame.  Since
    # rotations_plane_sensor = R_plane_base @ R_base_sensor and the plane
    # normal is plane-frame +Z, this is R_plane_sensor.T @ [0, 0, 1].
    plane_normals_sensor = rotations_plane_sensor[:, 2, :].copy()
    plane_normals_sensor /= np.linalg.norm(
        plane_normals_sensor, axis=1, keepdims=True
    )
    # Analysis tilt definition used throughout this report:
    #   tilt = angle(n_plane expressed in sensor frame, sensor -Z).
    # Since sensor -Z = [0, 0, -1], cos(tilt) = -n_s,z.
    normal_tilt_deg = np.degrees(
        np.arccos(np.clip(-plane_normals_sensor[:, 2], -1.0, 1.0))
    )
    tilt_deg = normal_tilt_deg.copy()
    normal_azimuth_sensor_deg = _wrap_degrees(
        np.degrees(
            np.arctan2(
                plane_normals_sensor[:, 1],
                plane_normals_sensor[:, 0],
            )
        )
    )

    ids = np.asarray(scan_ids, dtype=int)
    if len(np.unique(ids)) != len(ids):
        raise ValueError(f"duplicate scan IDs found in {spec.dataset_dir}")
    accepted = np.asarray([int(value) not in rejected_ids for value in ids])
    return PoseDistribution(
        spec=spec,
        capture_names=capture_names,
        scan_ids=ids,
        accepted=accepted,
        profiles_sensor_xyz_mm=profiles_sensor_xyz_mm,
        profile_source_keys=profile_source_keys,
        origins_plane_mm=origins_plane,
        measurement_origins_base_mm=measurement_origins_base,
        physical_origins_base_mm=origins_base,
        rotations_base_sensor=rotations_base_sensor,
        rotations_plane_sensor=rotations_plane_sensor,
        plane_normals_sensor=plane_normals_sensor,
        normal_tilt_deg=normal_tilt_deg,
        normal_azimuth_sensor_deg=normal_azimuth_sensor_deg,
        tilt_deg=tilt_deg,
        azimuth_deg=azimuth_deg,
        roll_deg=roll_deg,
        plane_centroid_base_mm=centroid,
        plane_normal_base=oriented_normal,
        plane_basis_base=basis,
        calibration_scan_count=diagnostics.get("accepted_scan_count"),
        calibration_rejected_count=diagnostics.get("rejected_scan_count"),
    )


def _convex_hull_area(points: np.ndarray) -> float:
    """Return the 2-D convex-hull area without requiring SciPy."""
    unique = sorted({(float(x), float(y)) for x, y in np.asarray(points)})
    if len(unique) < 3:
        return 0.0

    def cross(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    hull = lower[:-1] + upper[:-1]
    return 0.5 * abs(
        sum(
            hull[index][0] * hull[(index + 1) % len(hull)][1]
            - hull[(index + 1) % len(hull)][0] * hull[index][1]
            for index in range(len(hull))
        )
    )


def _circular_statistics(values_deg: np.ndarray) -> dict[str, float]:
    """Return circular mean/std and the smallest arc containing all angles."""
    values = _wrap_degrees(values_deg)
    radians = np.radians(values)
    mean_sin = float(np.mean(np.sin(radians)))
    mean_cos = float(np.mean(np.cos(radians)))
    resultant = float(np.hypot(mean_sin, mean_cos))
    mean_deg = float(np.degrees(np.arctan2(mean_sin, mean_cos)))
    std_deg = float(
        np.degrees(np.sqrt(max(-2.0 * np.log(max(resultant, 1e-15)), 0.0)))
    )

    around_circle = np.sort(values % 360.0)
    gaps = np.diff(np.r_[around_circle, around_circle[0] + 360.0])
    span_deg = float(360.0 - np.max(gaps)) if len(values) > 1 else 0.0
    return {
        "circular_mean_deg": mean_deg,
        "circular_std_deg": std_deg,
        "circular_span_deg": span_deg,
    }


def summarize(distribution: PoseDistribution) -> dict[str, Any]:
    positions = distribution.origins_plane_mm
    ranges = np.ptp(positions, axis=0)
    stds = np.std(positions, axis=0)
    tilt = distribution.tilt_deg
    azimuth_stats = _circular_statistics(distribution.azimuth_deg)
    roll_stats = _circular_statistics(distribution.roll_deg)
    return {
        "label": distribution.spec.label,
        "capture_count": int(len(positions)),
        "accepted_pose_count": int(np.sum(distribution.accepted)),
        "rejected_pose_count": int(np.sum(~distribution.accepted)),
        "profile_available_count": int(
            sum(len(profile) > 0 for profile in distribution.profiles_sensor_xyz_mm)
        ),
        "profile_missing_count": int(
            sum(len(profile) == 0 for profile in distribution.profiles_sensor_xyz_mm)
        ),
        "profile_point_count_total": int(
            sum(len(profile) for profile in distribution.profiles_sensor_xyz_mm)
        ),
        "plane_u_range_mm": float(ranges[0]),
        "plane_v_range_mm": float(ranges[1]),
        "plane_d_range_mm": float(ranges[2]),
        "plane_u_std_mm": float(stds[0]),
        "plane_v_std_mm": float(stds[1]),
        "plane_d_mean_mm": float(np.mean(positions[:, 2])),
        "plane_d_std_mm": float(stds[2]),
        "plane_d_range_mm": float(ranges[2]),
        "plane_d_min_mm": float(np.min(positions[:, 2])),
        "plane_d_max_mm": float(np.max(positions[:, 2])),
        "in_plane_convex_hull_area_mm2": _convex_hull_area(positions[:, :2]),
        "tilt_mean_deg": float(np.mean(tilt)),
        "tilt_std_deg": float(np.std(tilt)),
        "tilt_min_deg": float(np.min(tilt)),
        "tilt_max_deg": float(np.max(tilt)),
        "azimuth_circular_mean_deg": azimuth_stats["circular_mean_deg"],
        "azimuth_circular_std_deg": azimuth_stats["circular_std_deg"],
        "azimuth_circular_span_deg": azimuth_stats["circular_span_deg"],
        "roll_circular_mean_deg": roll_stats["circular_mean_deg"],
        "roll_circular_std_deg": roll_stats["circular_std_deg"],
        "roll_circular_span_deg": roll_stats["circular_span_deg"],
        "plane_centroid_base_mm": distribution.plane_centroid_base_mm.tolist(),
        "plane_normal_base": distribution.plane_normal_base.tolist(),
        "dataset_dir": str(distribution.spec.dataset_dir),
        "transform_path": str(distribution.spec.transform_path),
        "diagnostics_path": str(distribution.spec.diagnostics_path),
        "sensor_origin_offset_z_mm": float(
            distribution.spec.sensor_origin_offset_z_mm
        ),
    }


def _hover_text(distribution: PoseDistribution) -> list[str]:
    result: list[str] = []
    for index, name in enumerate(distribution.capture_names):
        u, v, d = distribution.origins_plane_mm[index]
        status = "accepted" if distribution.accepted[index] else "rejected"
        result.append(
            f"{escape(name)}<br>calibration: {status}<br>"
            f"U: {u:.3f} mm<br>V: {v:.3f} mm<br>"
            f"d: {d:.3f} mm<br>"
            f"tilt: {distribution.tilt_deg[index]:.3f} deg<br>"
            f"azimuth: {distribution.azimuth_deg[index]:.3f} deg<br>"
            f"roll: {distribution.roll_deg[index]:.3f} deg"
        )
    return result


def _angular_histogram(values_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    edges = np.linspace(-180.0, 180.0, 25)
    counts, _ = np.histogram(_wrap_degrees(values_deg), bins=edges)
    return 0.5 * (edges[:-1] + edges[1:]), counts


def make_dataset_figure(distribution: PoseDistribution):
    """Build one complete U/V/d/tilt/azimuth/roll analysis dashboard."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    figure = make_subplots(
        rows=2,
        cols=3,
        specs=[
            [{"type": "xy", "colspan": 2}, None, {"type": "xy"}],
            [{"type": "xy"}, {"type": "polar"}, {"type": "polar"}],
        ],
        subplot_titles=(
            "Board position: U / V (color = d)",
            "Normal distance d",
            "Tilt ∠(nˢ, −Zₛ)",
            "Azimuth",
            "Roll",
        ),
        horizontal_spacing=0.1,
        vertical_spacing=0.16,
    )
    hover = np.asarray(_hover_text(distribution), dtype=object)
    d_values = distribution.origins_plane_mm[:, 2]
    d_limit = max(float(np.max(np.abs(d_values))), 1.0)
    for accepted, symbol, label in (
        (True, "circle", "accepted"),
        (False, "x", "rejected"),
    ):
        mask = distribution.accepted == accepted
        if not np.any(mask):
            continue
        marker: dict[str, Any] = {
            "size": 9,
            "symbol": symbol,
            "opacity": 0.85,
        }
        if accepted:
            marker.update(
                {
                    "color": d_values[mask],
                    "colorscale": "RdBu_r",
                    "cmin": -d_limit,
                    "cmax": d_limit,
                    "colorbar": {"title": "d [mm]", "x": 0.63, "len": 0.42},
                }
            )
        else:
            marker["color"] = REJECTED_COLOR
        figure.add_trace(
            go.Scatter(
                x=distribution.origins_plane_mm[mask, 0],
                y=distribution.origins_plane_mm[mask, 1],
                mode="markers",
                marker=marker,
                text=hover[mask],
                hovertemplate="%{text}<extra></extra>",
                name=label,
            ),
            row=1,
            col=1,
        )

    figure.add_trace(
        go.Histogram(
            x=d_values,
            nbinsx=18,
            marker_color=distribution.spec.color,
            opacity=0.82,
            name="d",
            showlegend=False,
            hovertemplate="d: %{x:.2f} mm<br>count: %{y}<extra></extra>",
        ),
        row=1,
        col=3,
    )
    figure.add_trace(
        go.Histogram(
            x=distribution.tilt_deg,
            nbinsx=18,
            marker_color=distribution.spec.color,
            opacity=0.82,
            name="tilt",
            showlegend=False,
            hovertemplate="tilt: %{x:.2f}°<br>count: %{y}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    for column, values, label in (
        (2, distribution.azimuth_deg, "azimuth"),
        (3, distribution.roll_deg, "roll"),
    ):
        centers, counts = _angular_histogram(values)
        figure.add_trace(
            go.Barpolar(
                theta=centers,
                r=counts,
                width=np.full_like(centers, 15.0),
                marker_color=distribution.spec.color,
                opacity=0.78,
                name=label,
                showlegend=False,
                hovertemplate=(
                    f"{label}: %{{theta:.1f}}° bin<br>count: %{{r}}<extra></extra>"
                ),
            ),
            row=2,
            col=column,
        )

    figure.update_xaxes(title_text="U [mm]", row=1, col=1)
    figure.update_yaxes(
        title_text="V [mm]", scaleanchor="x", scaleratio=1, row=1, col=1
    )
    figure.update_xaxes(title_text="d [mm]", row=1, col=3)
    figure.update_yaxes(title_text="Count", row=1, col=3)
    figure.update_xaxes(title_text="Tilt ∠(nˢ, −Zₛ) [deg]", row=2, col=1)
    figure.update_yaxes(title_text="Count", row=2, col=1)
    polar_axis = {
        "radialaxis": {"title": "Count", "rangemode": "tozero"},
        "angularaxis": {
            "direction": "counterclockwise",
            "rotation": 0,
            "thetaunit": "degrees",
        },
    }
    figure.update_layout(
        title=(
            f"{distribution.spec.label} · board-relative sensor pose analysis · "
            f"N={len(distribution.scan_ids)}"
        ),
        polar=polar_axis,
        polar2=polar_axis,
        height=840,
        margin={"l": 60, "r": 40, "b": 60, "t": 90},
        legend={"orientation": "h", "y": -0.08},
        bargap=0.08,
    )
    return figure


def _line_segments(
    origins: np.ndarray,
    directions: np.ndarray,
    length: float,
) -> tuple[list[float | None], list[float | None], list[float | None]]:
    coordinates: list[list[float | None]] = [[], [], []]
    for origin, direction in zip(origins, directions):
        end = origin + length * direction
        for axis in range(3):
            coordinates[axis].extend(
                [float(origin[axis]), float(end[axis]), None]
            )
    return coordinates[0], coordinates[1], coordinates[2]


def make_sensor_frame_figure(
    distributions: list[PoseDistribution], axis_length_mm: float = 9.0
):
    """Visualize every sensor coordinate frame relative to its fitted board."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    figure = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=[
            f"{item.spec.label}: fitted board + sensor frames"
            for item in distributions
        ],
        horizontal_spacing=0.02,
    )
    all_positions = np.vstack(
        [distribution.origins_plane_mm for distribution in distributions]
    )
    axis_ranges: list[list[float]] = []
    for axis in range(3):
        low = float(np.min(all_positions[:, axis]))
        high = float(np.max(all_positions[:, axis]))
        if axis == 2:
            low = min(low, 0.0)
            high = max(high, 0.0)
        padding = max(0.08 * (high - low), 6.0)
        axis_ranges.append([low - padding, high + padding])

    u_grid, v_grid = np.meshgrid(
        np.linspace(axis_ranges[0][0], axis_ranges[0][1], 2),
        np.linspace(axis_ranges[1][0], axis_ranges[1][1], 2),
    )
    for column, distribution in enumerate(distributions, start=1):
        figure.add_trace(
            go.Surface(
                x=u_grid,
                y=v_grid,
                z=np.zeros_like(u_grid),
                colorscale=[[0.0, "#C8C8C8"], [1.0, "#C8C8C8"]],
                opacity=0.25,
                showscale=False,
                hoverinfo="skip",
                name="fitted board plane",
                showlegend=column == 1,
                legendgroup="board",
            ),
            row=1,
            col=column,
        )
        hover = np.asarray(_hover_text(distribution), dtype=object)
        for accepted, symbol, color, label in (
            (
                True,
                "circle",
                distribution.spec.color,
                "accepted physical origin",
            ),
            (False, "x", REJECTED_COLOR, "rejected physical origin"),
        ):
            mask = distribution.accepted == accepted
            if not np.any(mask):
                continue
            figure.add_trace(
                go.Scatter3d(
                    x=distribution.origins_plane_mm[mask, 0],
                    y=distribution.origins_plane_mm[mask, 1],
                    z=distribution.origins_plane_mm[mask, 2],
                    mode="markers",
                    marker={"size": 4, "symbol": symbol, "color": color},
                    text=hover[mask],
                    hovertemplate="%{text}<extra></extra>",
                    name=f"{distribution.spec.label} · {label}",
                    legendgroup=f"{distribution.spec.label}-{accepted}",
                    showlegend=True,
                ),
                row=1,
                col=column,
            )

        for axis, color, label in (
            (0, "#D62728", "sensor +X"),
            (1, "#2CA02C", "sensor +Y"),
            (2, "#1F77B4", "sensor +Z"),
        ):
            xyz = _line_segments(
                distribution.origins_plane_mm,
                distribution.rotations_plane_sensor[:, :, axis],
                axis_length_mm,
            )
            figure.add_trace(
                go.Scatter3d(
                    x=xyz[0],
                    y=xyz[1],
                    z=xyz[2],
                    mode="lines",
                    line={"color": color, "width": 3},
                    hoverinfo="skip",
                    name=label,
                    legendgroup=f"sensor-axis-{axis}",
                    showlegend=column == 1,
                ),
                row=1,
                col=column,
            )

        board_axis_length = 0.18 * min(
            axis_ranges[0][1] - axis_ranges[0][0],
            axis_ranges[1][1] - axis_ranges[1][0],
        )
        for endpoint, color, label in (
            ((board_axis_length, 0.0, 0.0), "#8B0000", "board +U"),
            ((0.0, board_axis_length, 0.0), "#006400", "board +V"),
            ((0.0, 0.0, board_axis_length), "#222222", "board normal +d"),
        ):
            figure.add_trace(
                go.Scatter3d(
                    x=[0.0, endpoint[0]],
                    y=[0.0, endpoint[1]],
                    z=[0.0, endpoint[2]],
                    mode="lines+text",
                    line={"color": color, "width": 8},
                    text=[None, label.rsplit(" ", 1)[-1]],
                    textposition="top center",
                    hoverinfo="skip",
                    name=label,
                    legendgroup=label,
                    showlegend=column == 1,
                ),
                row=1,
                col=column,
            )

    scene = {
        "xaxis": {"title": "Board U [mm]", "range": axis_ranges[0]},
        "yaxis": {"title": "Board V [mm]", "range": axis_ranges[1]},
        "zaxis": {"title": "Board normal d [mm]", "range": axis_ranges[2]},
        "aspectmode": "data",
        "camera": {"eye": {"x": 1.4, "y": 1.4, "z": 1.1}},
    }
    figure.update_layout(
        title=(
            "Physical sensor coordinate frames relative to the fitted board plane"
        ),
        scene=scene,
        scene2=scene,
        height=760,
        margin={"l": 0, "r": 0, "b": 0, "t": 90},
        legend={"orientation": "h", "y": -0.06},
    )
    return figure


def make_comparison_figure(distributions: list[PoseDistribution]):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    figure = make_subplots(
        rows=2,
        cols=2,
        specs=[
            [{"type": "xy"}, {"type": "polar"}],
            [{"type": "domain"}, {"type": "polar"}],
        ],
        subplot_titles=(
            "Position: U / V",
            "Orientation: tilt / azimuth",
            "All pose parameters",
            "Orientation: roll distribution",
        ),
        horizontal_spacing=0.1,
        vertical_spacing=0.12,
    )

    for distribution in distributions:
        hover = _hover_text(distribution)
        for accepted, symbol in ((True, "circle"), (False, "x")):
            mask = distribution.accepted == accepted
            if not np.any(mask):
                continue
            color = distribution.spec.color if accepted else REJECTED_COLOR
            figure.add_trace(
                go.Scatter(
                    x=distribution.origins_plane_mm[mask, 0],
                    y=distribution.origins_plane_mm[mask, 1],
                    mode="markers",
                    marker={"size": 9, "color": color, "symbol": symbol, "opacity": 0.8},
                    text=np.asarray(hover, dtype=object)[mask],
                    hovertemplate="%{text}<extra></extra>",
                    name=(
                        distribution.spec.label
                        if accepted
                        else f"{distribution.spec.label} rejected"
                    ),
                    legendgroup=f"{distribution.spec.label}-{accepted}",
                ),
                row=1,
                col=1,
            )
            figure.add_trace(
                go.Scatterpolar(
                    theta=distribution.azimuth_deg[mask],
                    r=distribution.tilt_deg[mask],
                    mode="markers",
                    marker={
                        "size": 8,
                        "color": color,
                        "symbol": symbol,
                        "opacity": 0.8,
                    },
                    text=np.asarray(hover, dtype=object)[mask],
                    hovertemplate="%{text}<extra></extra>",
                    name=(
                        distribution.spec.label
                        if accepted
                        else f"{distribution.spec.label} rejected"
                    ),
                    legendgroup=f"{distribution.spec.label}-{accepted}",
                    showlegend=False,
                ),
                row=1,
                col=2,
            )

        bin_edges = np.linspace(-180.0, 180.0, 25)
        counts, _ = np.histogram(distribution.roll_deg, bins=bin_edges)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        figure.add_trace(
            go.Barpolar(
                theta=bin_centers,
                r=counts,
                width=np.diff(bin_edges),
                marker_color=distribution.spec.color,
                opacity=0.58,
                name=distribution.spec.label,
                legendgroup=distribution.spec.label,
                showlegend=False,
                hovertemplate=(
                    f"{distribution.spec.label}<br>"
                    "roll bin center: %{theta:.1f} deg<br>count: %{r}<extra></extra>"
                ),
            ),
            row=2,
            col=2,
        )

    combined = distributions[0].origins_plane_mm[:, :2]
    u_values = [combined[:, 0]]
    v_values = [combined[:, 1]]
    d_values = [distributions[0].origins_plane_mm[:, 2]]
    tilt_values = [distributions[0].tilt_deg]
    azimuth_values = [distributions[0].azimuth_deg]
    roll_values = [distributions[0].roll_deg]
    color_values = [np.zeros(len(distributions[0].scan_ids))]
    for index, distribution in enumerate(distributions[1:], start=1):
        u_values.append(distribution.origins_plane_mm[:, 0])
        v_values.append(distribution.origins_plane_mm[:, 1])
        d_values.append(distribution.origins_plane_mm[:, 2])
        tilt_values.append(distribution.tilt_deg)
        azimuth_values.append(distribution.azimuth_deg)
        roll_values.append(distribution.roll_deg)
        color_values.append(np.full(len(distribution.scan_ids), index))

    colorscale: list[list[float | str]] = []
    denominator = max(len(distributions) - 1, 1)
    for index, distribution in enumerate(distributions):
        center = index / denominator
        epsilon = 1e-6
        colorscale.extend(
            [
                [max(center - epsilon, 0.0), distribution.spec.color],
                [min(center + epsilon, 1.0), distribution.spec.color],
            ]
        )
    figure.add_trace(
        go.Parcoords(
            line={
                "color": np.concatenate(color_values),
                "colorscale": colorscale,
                "showscale": False,
            },
            dimensions=[
                {"label": "U [mm]", "values": np.concatenate(u_values)},
                {"label": "V [mm]", "values": np.concatenate(v_values)},
                {"label": "d [mm]", "values": np.concatenate(d_values)},
                {"label": "Tilt ∠(nˢ, −Zₛ) [deg]", "values": np.concatenate(tilt_values)},
                {
                    "label": "Azimuth [deg]",
                    "values": np.concatenate(azimuth_values),
                    "range": [-180.0, 180.0],
                },
                {
                    "label": "Roll [deg]",
                    "values": np.concatenate(roll_values),
                    "range": [-180.0, 180.0],
                },
            ],
            labelfont={"size": 12},
            tickfont={"size": 10},
        ),
        row=2,
        col=1,
    )

    figure.update_xaxes(title_text="Plane U [mm]", row=1, col=1)
    figure.update_yaxes(
        title_text="Plane V [mm]", scaleanchor="x", scaleratio=1, row=1, col=1
    )
    figure.update_layout(
        title="Plane-normalized sensor poses: (U, V, d, tilt, azimuth, roll)",
        height=1050,
        margin={"l": 55, "r": 35, "b": 70, "t": 90},
        legend={"orientation": "h", "y": -0.07},
        polar={
            "radialaxis": {"title": "∠(nˢ, −Zₛ) [deg]", "rangemode": "tozero"},
            "angularaxis": {
                "direction": "counterclockwise",
                "rotation": 0,
                "thetaunit": "degrees",
            },
        },
        polar2={
            "radialaxis": {"title": "Count", "rangemode": "tozero"},
            "angularaxis": {
                "direction": "counterclockwise",
                "rotation": 0,
                "thetaunit": "degrees",
            },
            "barmode": "overlay",
        },
    )
    return figure


def _summary_table(summaries: list[dict[str, Any]]) -> str:
    rows = (
        ("Captured poses", "capture_count", "{:d}"),
        ("Calibration accepted", "accepted_pose_count", "{:d}"),
        ("Calibration rejected", "rejected_pose_count", "{:d}"),
        (
            "Physical-origin signed Z offset [mm]",
            "sensor_origin_offset_z_mm",
            "{:.1f}",
        ),
        ("U range [mm]", "plane_u_range_mm", "{:.2f}"),
        ("V range [mm]", "plane_v_range_mm", "{:.2f}"),
        ("d mean ± std [mm]", None, None),
        ("d range [mm]", None, None),
        ("U std [mm]", "plane_u_std_mm", "{:.2f}"),
        ("V std [mm]", "plane_v_std_mm", "{:.2f}"),
        ("In-plane convex-hull area [mm²]", "in_plane_convex_hull_area_mm2", "{:.1f}"),
        ("Tilt mean ± std [deg]", None, None),
        ("Tilt range [deg]", None, None),
        ("Azimuth circular mean ± std [deg]", None, None),
        ("Azimuth circular span [deg]", None, None),
        ("Roll circular mean ± std [deg]", None, None),
        ("Roll circular span [deg]", None, None),
    )
    header = "".join(f"<th>{escape(item['label'])}</th>" for item in summaries)
    body: list[str] = []
    for label, key, formatter in rows:
        values: list[str] = []
        for item in summaries:
            if key is not None and formatter is not None:
                values.append(formatter.format(item[key]))
            elif label.startswith("d mean"):
                values.append(
                    f"{item['plane_d_mean_mm']:.2f} ± "
                    f"{item['plane_d_std_mm']:.2f}"
                )
            elif label.startswith("d range"):
                values.append(
                    f"{item['plane_d_min_mm']:.2f} – "
                    f"{item['plane_d_max_mm']:.2f}"
                )
            elif label.startswith("Tilt mean"):
                values.append(
                    f"{item['tilt_mean_deg']:.2f} ± "
                    f"{item['tilt_std_deg']:.2f}"
                )
            elif label.startswith("Tilt range"):
                values.append(
                    f"{item['tilt_min_deg']:.2f} – "
                    f"{item['tilt_max_deg']:.2f}"
                )
            elif label.startswith("Azimuth circular mean"):
                values.append(
                    f"{item['azimuth_circular_mean_deg']:.2f} ± "
                    f"{item['azimuth_circular_std_deg']:.2f}"
                )
            elif label.startswith("Azimuth circular span"):
                values.append(
                    f"{item['azimuth_circular_span_deg']:.2f}"
                )
            elif label.startswith("Roll circular mean"):
                values.append(
                    f"{item['roll_circular_mean_deg']:.2f} ± "
                    f"{item['roll_circular_std_deg']:.2f}"
                )
            else:
                values.append(f"{item['roll_circular_span_deg']:.2f}")
        cells = "".join(f"<td>{value}</td>" for value in values)
        body.append(f"<tr><th>{escape(label)}</th>{cells}</tr>")
    return (
        "<table><thead><tr><th>Metric</th>"
        + header
        + "</tr></thead><tbody>"
        + "".join(body)
        + "</tbody></table>"
    )


def _dataset_analysis_html(summary: dict[str, Any]) -> str:
    acceptance = 100.0 * summary["accepted_pose_count"] / summary["capture_count"]
    azimuth_span = summary["azimuth_circular_span_deg"]
    roll_span = summary["roll_circular_span_deg"]

    def coverage_description(span: float) -> str:
        if span >= 300.0:
            return "거의 전 방향을 포함"
        if span >= 180.0:
            return "넓은 방향 영역을 포함"
        if span >= 90.0:
            return "한쪽 방향군에 비교적 집중"
        return "좁은 방향군에 집중"

    return f"""
    <div class="analysis-grid">
      <div class="analysis-card">
        <h3>Capture 품질</h3>
        <p>{summary['capture_count']}개 중 {summary['accepted_pose_count']}개가
        calibration에 사용되어 수용률은 <strong>{acceptance:.1f}%</strong>입니다.</p>
      </div>
      <div class="analysis-card">
        <h3>U / V 위치</h3>
        <p>U 범위 {summary['plane_u_range_mm']:.2f} mm, V 범위
        {summary['plane_v_range_mm']:.2f} mm이며, 평면 내 convex-hull 면적은
        <strong>{summary['in_plane_convex_hull_area_mm2']:.1f} mm²</strong>입니다.</p>
      </div>
      <div class="analysis-card">
        <h3>d: 실제 원점의 보드 법선거리</h3>
        <p>평균 {summary['plane_d_mean_mm']:.2f} mm, 표준편차
        {summary['plane_d_std_mm']:.2f} mm, 범위
        {summary['plane_d_min_mm']:.2f}–{summary['plane_d_max_mm']:.2f} mm입니다.</p>
      </div>
      <div class="analysis-card">
        <h3>Tilt = ∠(nˢ, −Zₛ)</h3>
        <p>센서 좌표계에서 표현한 평면 법선과 sensor −Z축 사이의 각도입니다.
        평균 {summary['tilt_mean_deg']:.2f}°, 표준편차
        {summary['tilt_std_deg']:.2f}°, 범위
        {summary['tilt_min_deg']:.2f}–{summary['tilt_max_deg']:.2f}°입니다.</p>
      </div>
      <div class="analysis-card">
        <h3>Azimuth</h3>
        <p>원형 평균 {summary['azimuth_circular_mean_deg']:.2f}°, 원형 표준편차
        {summary['azimuth_circular_std_deg']:.2f}°, coverage span
        <strong>{azimuth_span:.2f}°</strong>로 {coverage_description(azimuth_span)}합니다.</p>
      </div>
      <div class="analysis-card">
        <h3>Roll</h3>
        <p>원형 평균 {summary['roll_circular_mean_deg']:.2f}°, 원형 표준편차
        {summary['roll_circular_std_deg']:.2f}°, coverage span
        <strong>{roll_span:.2f}°</strong>로 {coverage_description(roll_span)}합니다.</p>
      </div>
    </div>
    """


def _comparison_analysis_html(summaries: list[dict[str, Any]]) -> str:
    backup, current = summaries

    def relative_change(key: str) -> float:
        baseline = float(backup[key])
        return 100.0 * (float(current[key]) / baseline - 1.0)

    def changed_by(key: str, increased: str, decreased: str) -> str:
        change = relative_change(key)
        direction = increased if change >= 0.0 else decreased
        return f"{abs(change):.1f}% {direction}"

    return f"""
    <div class="analysis-grid">
      <div class="analysis-card">
        <h3>U 위치 비교</h3>
        <p>Current의 U 범위는 Backup보다
        <strong>{changed_by('plane_u_range_mm', '넓고', '좁고')}</strong>,
        U 표준편차는 {changed_by('plane_u_std_mm', '큽니다', '작습니다')}.</p>
      </div>
      <div class="analysis-card">
        <h3>V 위치 비교</h3>
        <p>Current의 V 범위는 Backup보다
        <strong>{changed_by('plane_v_range_mm', '넓고', '좁고')}</strong>, 평면 내
        convex-hull 면적은
        {changed_by('in_plane_convex_hull_area_mm2', '큽니다', '작습니다')}.</p>
      </div>
      <div class="analysis-card">
        <h3>d 비교</h3>
        <p>Current의 d 표준편차는 Backup보다
        <strong>{changed_by('plane_d_std_mm', '크고', '작고')}</strong>, 전체
        d 범위도 {changed_by('plane_d_range_mm', '넓습니다', '좁습니다')}.</p>
      </div>
      <div class="analysis-card">
        <h3>Tilt 비교</h3>
        <p>Current 평균 tilt는 Backup보다
        <strong>{current['tilt_mean_deg'] - backup['tilt_mean_deg']:.2f}° 큽니다.</strong>
        반면 표준편차는 {backup['tilt_std_deg'] - current['tilt_std_deg']:.2f}°
        작아, Current는 더 기울었지만 tilt 다양성은 낮습니다.</p>
      </div>
      <div class="analysis-card">
        <h3>Azimuth 비교</h3>
        <p>Coverage span은 Backup {backup['azimuth_circular_span_deg']:.2f}°,
        Current {current['azimuth_circular_span_deg']:.2f}°입니다. Backup이
        <strong>{backup['azimuth_circular_span_deg'] - current['azimuth_circular_span_deg']:.2f}°
        더 넓은</strong> azimuth 방향을 포함합니다.</p>
      </div>
      <div class="analysis-card">
        <h3>Roll 비교</h3>
        <p>Coverage span은 Backup {backup['roll_circular_span_deg']:.2f}°,
        Current {current['roll_circular_span_deg']:.2f}°입니다. Backup이
        <strong>{backup['roll_circular_span_deg'] - current['roll_circular_span_deg']:.2f}°
        더 넓어</strong> roll 다양성이 큽니다.</p>
      </div>
    </div>
    """


def save_html(
    distributions: list[PoseDistribution],
    summaries: list[dict[str, Any]],
    output: Path,
) -> None:
    dataset_sections: list[str] = []
    plotly_included = False
    for distribution, summary in zip(distributions, summaries):
        dataset_figure = make_dataset_figure(distribution)
        figure_html = dataset_figure.to_html(
            full_html=False,
            include_plotlyjs=True if not plotly_included else False,
            config={"responsive": True},
        )
        plotly_included = True
        section_id = distribution.spec.label.lower().replace(" ", "-")
        dataset_sections.append(
            f"""
            <section id="{escape(section_id)}">
              <h2>{escape(distribution.spec.label)} dataset analysis</h2>
              {_dataset_analysis_html(summary)}
              <div class="plot">{figure_html}</div>
            </section>
            """
        )

    comparison_figure = make_comparison_figure(distributions)
    comparison_html = comparison_figure.to_html(
        full_html=False, include_plotlyjs=False, config={"responsive": True}
    )
    sensor_frame_figure = make_sensor_frame_figure(distributions)
    sensor_frame_html = sensor_frame_figure.to_html(
        full_html=False, include_plotlyjs=False, config={"responsive": True}
    )
    table = _summary_table(summaries)
    document = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Sensor pose distribution comparison</title>
  <style>
    body {{ font-family: system-ui, sans-serif; color: #222; margin: 24px; }}
    h1 {{ margin-bottom: 6px; }}
    h2 {{ margin-top: 42px; border-bottom: 2px solid #e7e7e7; padding-bottom: 8px; }}
    .note {{ color: #555; max-width: 1100px; line-height: 1.5; }}
    table {{ border-collapse: collapse; margin: 20px 0 30px; min-width: 760px; }}
    th, td {{ border: 1px solid #ddd; padding: 8px 11px; text-align: right; }}
    thead th {{ background: #f1f3f5; }}
    tbody th {{ text-align: left; background: #fafafa; }}
    .plot {{ width: 100%; overflow-x: auto; }}
    .analysis-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 12px;
      margin: 18px 0;
    }}
    .analysis-card {{
      border: 1px solid #ddd;
      border-radius: 8px;
      padding: 13px 16px;
      background: #fafafa;
    }}
    .analysis-card h3 {{ margin: 0 0 7px; font-size: 1rem; }}
    .analysis-card p {{ margin: 0; line-height: 1.5; color: #444; }}
    nav a {{ margin-right: 16px; }}
    code {{ background: #f4f4f4; padding: 2px 4px; }}
  </style>
</head>
<body>
  <h1>Backup vs current sensor pose distribution</h1>
  <nav>
    <a href="#backup">Backup</a>
    <a href="#current">Current</a>
    <a href="#comparison">Comparison</a>
    <a href="#sensor-frames">Sensor frames</a>
  </nav>
  <p class="note">
    Every physical sensor origin is computed in World/Base coordinates so
    runs with differently placed target boards can be compared directly. U is
    projected base +X, V completes the board plane, and d is signed distance
    along the board normal. The physical origin is the measurement-coordinate
    origin translated <strong>{summaries[0]['sensor_origin_offset_z_mm']:g} mm
    along signed sensor Z</strong> (negative means -Z). Orientation uses
    the decomposition
    <code>R_plane_sensor = Rz(azimuth) Ry(tilt) Rz(roll)</code>: tilt is the
    reported tilt is the angle between the oriented plane normal and sensor
    -Z. Azimuth and roll remain the orientation-decomposition quantities used
    previously.
    Azimuth and roll are wrapped to [-180°, 180°). Red crosses are captures
    rejected by the calibration profile-RANSAC stage.
  </p>
  {table}
  {''.join(dataset_sections)}
  <section id="comparison">
    <h2>Backup / Current comparison</h2>
    {_comparison_analysis_html(summaries)}
    <p class="note">
      Parallel coordinates show each capture across all six parameters.
      Azimuth and roll use circular values, so values near -180° and +180°
      are adjacent directions.
    </p>
    <div class="plot">{comparison_html}</div>
  </section>
  <section id="sensor-frames">
    <h2>Board plane / physical sensor coordinate frames</h2>
    <p class="note">
      The gray surface is the fitted board plane (d=0). Thick U, V and d axes
      define the board frame. Physical sensor origins use the
      {summaries[0]['sensor_origin_offset_z_mm']:g} mm signed sensor-Z offset; each
      frame uses X=red, Y=green and Z=blue. Hover over an origin to inspect U,
      V, d, tilt, azimuth and roll.
    </p>
    <div class="plot">{sensor_frame_html}</div>
  </section>
</body>
</html>
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")


def save_pose_csv(distributions: list[PoseDistribution], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "dataset",
                "capture",
                "scan_id",
                "calibration_accepted",
                "profile_source_key",
                "profile_point_count",
                "plane_u_mm",
                "plane_v_mm",
                "plane_d_mm",
                "tilt_angle_normal_to_sensor_minus_z_deg",
                "azimuth_deg",
                "roll_deg",
            ]
        )
        for distribution in distributions:
            for index, name in enumerate(distribution.capture_names):
                u, v, d = distribution.origins_plane_mm[index]
                writer.writerow(
                    [
                        distribution.spec.label,
                        name,
                        int(distribution.scan_ids[index]),
                        bool(distribution.accepted[index]),
                        distribution.profile_source_keys[index],
                        int(len(distribution.profiles_sensor_xyz_mm[index])),
                        float(u),
                        float(v),
                        float(d),
                        float(distribution.tilt_deg[index]),
                        float(distribution.azimuth_deg[index]),
                        float(distribution.roll_deg[index]),
                    ]
                )


def _parse_edge_list(text: str | None, name: str) -> np.ndarray | None:
    if text is None:
        return None
    values = np.asarray(
        [float(item.strip()) for item in text.split(",") if item.strip()],
        dtype=float,
    )
    if values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain at least two finite values")
    if not np.all(np.diff(values) > 0.0):
        raise ValueError(f"{name} must be strictly increasing")
    return values


def _uniform_edges(low: float, high: float, width: float) -> np.ndarray:
    if not np.isfinite(width) or width <= 0.0:
        raise ValueError("bin width must be a positive finite number")
    if high <= low:
        raise ValueError("bin edge high value must be greater than low value")
    count = int(np.ceil((high - low) / width))
    edges = low + width * np.arange(count + 1, dtype=float)
    edges[-1] = high
    if edges[-2] >= high:
        edges = edges[:-1]
        edges[-1] = high
    return edges


def _automatic_distance_edges(values: np.ndarray, width_mm: float) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("no finite distance values are available for binning")
    low = float(np.floor(np.min(values) / width_mm) * width_mm)
    high = float(np.ceil(np.max(values) / width_mm) * width_mm)
    if np.isclose(low, high):
        low -= 0.5 * width_mm
        high += 0.5 * width_mm
    edges = np.arange(low, high + 0.5 * width_mm, width_mm, dtype=float)
    if edges[-1] < np.max(values):
        edges = np.r_[edges, edges[-1] + width_mm]
    return edges


def build_binning_config(
    args: argparse.Namespace,
    distributions: list[PoseDistribution],
) -> BinningConfig:
    tilt_edges = _parse_edge_list(args.tilt_bin_edges_deg, "tilt edges")
    if tilt_edges is None:
        tilt_edges = _uniform_edges(0.0, 180.0, args.tilt_bin_width_deg)
    if tilt_edges[0] > 0.0 or tilt_edges[-1] < 180.0:
        raise ValueError("tilt edges must cover the full [0, 180] degree range")

    azimuth_edges = _uniform_edges(
        -180.0, 180.0, args.azimuth_bin_width_deg
    )
    roll_edges = _uniform_edges(-180.0, 180.0, args.roll_bin_width_deg)

    distance_edges = _parse_edge_list(
        args.distance_bin_edges_mm, "distance edges"
    )
    # Build common d edges from every captured pose so rejected poses can still
    # be written to the per-pose CSV without creating out-of-range assignments.
    # The accepted-only option controls counts/plots, not geometric coverage.
    distance_values = [
        distribution.origins_plane_mm[:, 2] for distribution in distributions
    ]
    combined_distance = np.concatenate(distance_values)
    if distance_edges is None:
        distance_edges = _automatic_distance_edges(
            combined_distance, args.distance_bin_width_mm
        )
    if (
        np.min(combined_distance) < distance_edges[0] - 1e-9
        or np.max(combined_distance) > distance_edges[-1] + 1e-9
    ):
        raise ValueError(
            "distance bin edges do not cover all selected distance values: "
            f"data=[{np.min(combined_distance):.6g}, "
            f"{np.max(combined_distance):.6g}], "
            f"edges=[{distance_edges[0]:.6g}, {distance_edges[-1]:.6g}]"
        )

    return BinningConfig(
        tilt_edges_deg=np.asarray(tilt_edges, dtype=float),
        azimuth_edges_deg=np.asarray(azimuth_edges, dtype=float),
        distance_edges_mm=np.asarray(distance_edges, dtype=float),
        roll_edges_deg=np.asarray(roll_edges, dtype=float),
        accepted_only=not args.bin_include_rejected,
    )


def _assign_bin_indices(
    values: np.ndarray,
    edges: np.ndarray,
    *,
    circular: bool = False,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    edges = np.asarray(edges, dtype=float)
    if circular:
        values = _wrap_degrees(values)
    # Numerical rotations often produce values such as 19.999999999999993
    # for an intended 20-degree boundary. Snap only machine-level deviations
    # to the configured edge so boundary assignment follows the stated bins.
    values = values.copy()
    for edge in edges:
        values[np.isclose(values, edge, rtol=0.0, atol=1e-9)] = edge
    indices = np.searchsorted(edges, values, side="right") - 1
    indices[np.isclose(values, edges[-1], atol=1e-10)] = len(edges) - 2
    invalid = (
        ~np.isfinite(values)
        | (indices < 0)
        | (indices >= len(edges) - 1)
    )
    if np.any(invalid):
        bad_values = values[invalid][:8]
        raise ValueError(
            "some values fall outside the configured bin edges: "
            f"{bad_values.tolist()}"
        )
    return indices.astype(int)


def _bin_labels(edges: np.ndarray, unit: str) -> list[str]:
    labels: list[str] = []
    for index, (low, high) in enumerate(zip(edges[:-1], edges[1:])):
        closing = "]" if index == len(edges) - 2 else ")"
        labels.append(f"[{low:g}, {high:g}{closing} {unit}")
    return labels


def assign_pose_bins(
    distribution: PoseDistribution,
    config: BinningConfig,
) -> dict[str, np.ndarray]:
    return {
        "tilt": _assign_bin_indices(
            distribution.tilt_deg, config.tilt_edges_deg
        ),
        "azimuth": _assign_bin_indices(
            distribution.azimuth_deg,
            config.azimuth_edges_deg,
            circular=True,
        ),
        "distance": _assign_bin_indices(
            distribution.origins_plane_mm[:, 2], config.distance_edges_mm
        ),
        "roll": _assign_bin_indices(
            distribution.roll_deg, config.roll_edges_deg, circular=True
        ),
    }


def _bin_count(
    indices: np.ndarray,
    bin_count: int,
    mask: np.ndarray,
) -> np.ndarray:
    return np.bincount(indices[mask], minlength=bin_count).astype(int)


def marginal_bin_rows(
    distribution: PoseDistribution,
    config: BinningConfig,
    assignments: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    parameters = (
        ("tilt", config.tilt_edges_deg, "deg"),
        ("azimuth", config.azimuth_edges_deg, "deg"),
        ("distance", config.distance_edges_mm, "mm"),
        ("roll", config.roll_edges_deg, "deg"),
    )
    rows: list[dict[str, Any]] = []
    accepted = distribution.accepted
    for name, edges, unit in parameters:
        labels = _bin_labels(edges, unit)
        accepted_counts = _bin_count(
            assignments[name], len(edges) - 1, accepted
        )
        rejected_counts = _bin_count(
            assignments[name], len(edges) - 1, ~accepted
        )
        all_counts = accepted_counts + rejected_counts
        accepted_total = int(np.sum(accepted_counts))
        for index, label in enumerate(labels):
            rows.append(
                {
                    "dataset": distribution.spec.label,
                    "parameter": name,
                    "bin_index": index,
                    "bin_label": label,
                    "lower": float(edges[index]),
                    "upper": float(edges[index + 1]),
                    "unit": unit,
                    "accepted_count": int(accepted_counts[index]),
                    "rejected_count": int(rejected_counts[index]),
                    "all_count": int(all_counts[index]),
                    "accepted_fraction": (
                        float(accepted_counts[index] / accepted_total)
                        if accepted_total > 0
                        else 0.0
                    ),
                }
            )
    return rows


def joint_bin_rows(
    distribution: PoseDistribution,
    config: BinningConfig,
    assignments: dict[str, np.ndarray],
) -> list[dict[str, Any]]:
    labels = {
        "tilt": _bin_labels(config.tilt_edges_deg, "deg"),
        "azimuth": _bin_labels(config.azimuth_edges_deg, "deg"),
        "distance": _bin_labels(config.distance_edges_mm, "mm"),
        "roll": _bin_labels(config.roll_edges_deg, "deg"),
    }
    groups: dict[tuple[int, int, int, int], dict[str, list[int]]] = {}
    for index, scan_id in enumerate(distribution.scan_ids):
        key = (
            int(assignments["tilt"][index]),
            int(assignments["azimuth"][index]),
            int(assignments["distance"][index]),
            int(assignments["roll"][index]),
        )
        bucket = groups.setdefault(key, {"accepted": [], "rejected": []})
        status = "accepted" if distribution.accepted[index] else "rejected"
        bucket[status].append(int(scan_id))

    rows: list[dict[str, Any]] = []
    for key, bucket in groups.items():
        tilt_index, azimuth_index, distance_index, roll_index = key
        accepted_ids = sorted(bucket["accepted"])
        rejected_ids = sorted(bucket["rejected"])
        rows.append(
            {
                "dataset": distribution.spec.label,
                "tilt_bin_index": tilt_index,
                "azimuth_bin_index": azimuth_index,
                "distance_bin_index": distance_index,
                "roll_bin_index": roll_index,
                "tilt_bin": labels["tilt"][tilt_index],
                "azimuth_bin": labels["azimuth"][azimuth_index],
                "distance_bin": labels["distance"][distance_index],
                "roll_bin": labels["roll"][roll_index],
                "accepted_count": len(accepted_ids),
                "rejected_count": len(rejected_ids),
                "all_count": len(accepted_ids) + len(rejected_ids),
                "accepted_scan_ids": accepted_ids,
                "rejected_scan_ids": rejected_ids,
            }
        )
    rows.sort(
        key=lambda row: (
            -int(row["accepted_count"]),
            -int(row["all_count"]),
            int(row["tilt_bin_index"]),
            int(row["azimuth_bin_index"]),
            int(row["distance_bin_index"]),
            int(row["roll_bin_index"]),
        )
    )
    return rows


def _rotation_from_tilt_azimuth_roll(
    tilt_deg: np.ndarray,
    azimuth_deg: np.ndarray,
    roll_deg: np.ndarray,
) -> np.ndarray:
    tilt = np.radians(np.asarray(tilt_deg, dtype=float))
    azimuth = np.radians(np.asarray(azimuth_deg, dtype=float))
    roll = np.radians(np.asarray(roll_deg, dtype=float))
    result = np.empty((len(tilt), 3, 3), dtype=float)
    for index, (t, a, r) in enumerate(zip(tilt, azimuth, roll)):
        ca, sa = np.cos(a), np.sin(a)
        ct, st = np.cos(t), np.sin(t)
        cr, sr = np.cos(r), np.sin(r)
        rz_a = np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]])
        ry_t = np.array([[ct, 0.0, st], [0.0, 1.0, 0.0], [-st, 0.0, ct]])
        rz_r = np.array([[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]])
        result[index] = rz_a @ ry_t @ rz_r
    return result


def validate_pose_distribution(
    distribution: PoseDistribution,
    config: BinningConfig,
    assignments: dict[str, np.ndarray],
) -> dict[str, Any]:
    rotations = distribution.rotations_plane_sensor
    identity = np.eye(3)
    orthogonality_errors = np.linalg.norm(
        np.einsum("nji,njk->nik", rotations, rotations) - identity,
        axis=(1, 2),
    )
    determinant_errors = np.abs(np.linalg.det(rotations) - 1.0)

    basis = distribution.plane_basis_base
    basis_orthogonality_error = float(
        np.linalg.norm(basis.T @ basis - identity)
    )
    basis_determinant_error = float(abs(np.linalg.det(basis) - 1.0))
    basis_normal_error = float(
        np.linalg.norm(basis[:, 2] - distribution.plane_normal_base)
    )

    direct_normals_sensor = np.einsum(
        "nji,j->ni",
        distribution.rotations_base_sensor,
        distribution.plane_normal_base,
    )
    normal_transform_error = np.linalg.norm(
        direct_normals_sensor - distribution.plane_normals_sensor, axis=1
    )
    normal_norm_error = np.abs(
        np.linalg.norm(distribution.plane_normals_sensor, axis=1) - 1.0
    )

    direct_distance = (
        distribution.physical_origins_base_mm
        - distribution.plane_centroid_base_mm
    ) @ distribution.plane_normal_base
    distance_transform_error = np.abs(
        direct_distance - distribution.origins_plane_mm[:, 2]
    )

    tilt_normal_consistency_error = np.abs(
        distribution.tilt_deg - distribution.normal_tilt_deg
    )
    # The reported tilt is angle(n_s, -Z_s), whereas the internal Z-Y-Z
    # decomposition angle is angle(n_s, +Z_s).  The two sum to 180 degrees.
    euler_tilt_deg = 180.0 - distribution.tilt_deg
    reconstructed = _rotation_from_tilt_azimuth_roll(
        euler_tilt_deg,
        distribution.azimuth_deg,
        distribution.roll_deg,
    )
    delta = np.einsum("nji,njk->nik", reconstructed, rotations)
    cosine = np.clip((np.trace(delta, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    reconstruction_angle_error_deg = np.degrees(np.arccos(cosine))

    selected_mask = (
        distribution.accepted
        if config.accepted_only
        else np.ones(len(distribution.scan_ids), dtype=bool)
    )
    selected_count = int(np.sum(selected_mask))
    bin_count_checks: dict[str, dict[str, Any]] = {}
    edges_by_name = {
        "tilt": config.tilt_edges_deg,
        "azimuth": config.azimuth_edges_deg,
        "distance": config.distance_edges_mm,
        "roll": config.roll_edges_deg,
    }
    for name, edges in edges_by_name.items():
        counts = _bin_count(assignments[name], len(edges) - 1, selected_mask)
        bin_count_checks[name] = {
            "sum": int(np.sum(counts)),
            "expected": selected_count,
            "matches": bool(int(np.sum(counts)) == selected_count),
            "occupied_bins": int(np.sum(counts > 0)),
            "total_bins": int(len(counts)),
            "max_bin_count": int(np.max(counts)) if counts.size else 0,
        }

    accepted_count_matches = (
        distribution.calibration_scan_count is None
        or int(distribution.calibration_scan_count)
        == int(np.sum(distribution.accepted))
    )
    rejected_count_matches = (
        distribution.calibration_rejected_count is None
        or int(distribution.calibration_rejected_count)
        == int(np.sum(~distribution.accepted))
    )

    singular_mask = (
        (distribution.tilt_deg < 1.0)
        | (distribution.tilt_deg > 179.0)
    )
    warnings: list[str] = []
    if np.any(singular_mask):
        warnings.append(
            f"{int(np.sum(singular_mask))} poses have tilt within 1 degree of "
            "the ZYZ singularity; azimuth and roll are individually unstable there."
        )
    if not accepted_count_matches or not rejected_count_matches:
        warnings.append(
            "accepted/rejected counts derived from scan IDs do not match the "
            "diagnostics JSON counts. Check scan-ID indexing and filename parsing."
        )
    warnings.append(
        "The signed sensor-origin Z offset is a user-supplied physical model and cannot "
        "be verified from the capture transforms alone."
    )
    warnings.append(
        "The fitted plane and hand-eye transform are calibration-derived references, "
        "not external ground truth; plots are internally consistent but not an "
        "absolute accuracy validation."
    )

    numeric_pass = bool(
        np.max(orthogonality_errors) < 1e-8
        and np.max(determinant_errors) < 1e-8
        and basis_orthogonality_error < 1e-8
        and basis_determinant_error < 1e-8
        and basis_normal_error < 1e-8
        and np.max(normal_transform_error) < 1e-8
        and np.max(normal_norm_error) < 1e-8
        and np.max(distance_transform_error) < 1e-8
        and np.max(tilt_normal_consistency_error) < 1e-8
        and np.max(reconstruction_angle_error_deg) < 1e-4
        and all(item["matches"] for item in bin_count_checks.values())
    )
    status = "pass" if numeric_pass and accepted_count_matches and rejected_count_matches else "warning"
    return {
        "dataset": distribution.spec.label,
        "status": status,
        "pose_count": int(len(distribution.scan_ids)),
        "selected_for_bins": selected_count,
        "max_rotation_orthogonality_error_fro": float(np.max(orthogonality_errors)),
        "max_rotation_determinant_error": float(np.max(determinant_errors)),
        "plane_basis_orthogonality_error_fro": basis_orthogonality_error,
        "plane_basis_determinant_error": basis_determinant_error,
        "plane_basis_normal_error": basis_normal_error,
        "max_normal_transform_error": float(np.max(normal_transform_error)),
        "max_normal_unit_length_error": float(np.max(normal_norm_error)),
        "max_distance_transform_error_mm": float(np.max(distance_transform_error)),
        "max_tilt_normal_consistency_error_deg": float(
            np.max(tilt_normal_consistency_error)
        ),
        "max_rotation_reconstruction_error_deg": float(
            np.max(reconstruction_angle_error_deg)
        ),
        "diagnostics_accepted_count_matches": bool(accepted_count_matches),
        "diagnostics_rejected_count_matches": bool(rejected_count_matches),
        "zyz_singularity_pose_count": int(np.sum(singular_mask)),
        "bin_count_checks": bin_count_checks,
        "warnings": warnings,
    }


def make_binning_figure(
    distribution: PoseDistribution,
    config: BinningConfig,
    assignments: dict[str, np.ndarray],
):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    mask = (
        distribution.accepted
        if config.accepted_only
        else np.ones(len(distribution.scan_ids), dtype=bool)
    )
    tilt_labels = _bin_labels(config.tilt_edges_deg, "deg")
    azimuth_labels = _bin_labels(config.azimuth_edges_deg, "deg")
    distance_labels = _bin_labels(config.distance_edges_mm, "mm")
    roll_labels = _bin_labels(config.roll_edges_deg, "deg")

    tilt_counts = _bin_count(
        assignments["tilt"], len(tilt_labels), mask
    )
    azimuth_counts = _bin_count(
        assignments["azimuth"], len(azimuth_labels), mask
    )
    distance_counts = _bin_count(
        assignments["distance"], len(distance_labels), mask
    )
    roll_counts = _bin_count(
        assignments["roll"], len(roll_labels), mask
    )

    tilt_azimuth = np.zeros(
        (len(tilt_labels), len(azimuth_labels)), dtype=int
    )
    distance_tilt = np.zeros(
        (len(distance_labels), len(tilt_labels)), dtype=int
    )
    for index in np.flatnonzero(mask):
        tilt_azimuth[
            assignments["tilt"][index], assignments["azimuth"][index]
        ] += 1
        distance_tilt[
            assignments["distance"][index], assignments["tilt"][index]
        ] += 1

    figure = make_subplots(
        rows=3,
        cols=2,
        specs=[
            [{"type": "xy"}, {"type": "polar"}],
            [{"type": "xy"}, {"type": "polar"}],
            [{"type": "xy"}, {"type": "xy"}],
        ],
        subplot_titles=(
            "Tilt ∠(nˢ, −Zₛ) bins",
            "Azimuth bins",
            "Distance d bins",
            "Sensor roll bins",
            "Tilt ∠(nˢ, −Zₛ) × azimuth occupancy",
            "Distance d × tilt ∠(nˢ, −Zₛ) occupancy",
        ),
        vertical_spacing=0.12,
        horizontal_spacing=0.12,
    )
    figure.add_trace(
        go.Bar(
            x=tilt_labels,
            y=tilt_counts,
            marker_color=distribution.spec.color,
            name="tilt",
            hovertemplate="%{x}<br>count=%{y}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    az_centers = 0.5 * (
        config.azimuth_edges_deg[:-1] + config.azimuth_edges_deg[1:]
    )
    figure.add_trace(
        go.Barpolar(
            theta=az_centers,
            r=azimuth_counts,
            width=np.diff(config.azimuth_edges_deg),
            marker_color=distribution.spec.color,
            name="azimuth",
            hovertemplate="center=%{theta:.1f}°<br>count=%{r}<extra></extra>",
        ),
        row=1,
        col=2,
    )
    figure.add_trace(
        go.Bar(
            x=distance_labels,
            y=distance_counts,
            marker_color=distribution.spec.color,
            name="distance",
            hovertemplate="%{x}<br>count=%{y}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    roll_centers = 0.5 * (
        config.roll_edges_deg[:-1] + config.roll_edges_deg[1:]
    )
    figure.add_trace(
        go.Barpolar(
            theta=roll_centers,
            r=roll_counts,
            width=np.diff(config.roll_edges_deg),
            marker_color=distribution.spec.color,
            name="roll",
            hovertemplate="center=%{theta:.1f}°<br>count=%{r}<extra></extra>",
        ),
        row=2,
        col=2,
    )
    figure.add_trace(
        go.Heatmap(
            x=azimuth_labels,
            y=tilt_labels,
            z=tilt_azimuth,
            colorscale="Blues",
            colorbar={"title": "Count", "x": 0.45, "len": 0.28, "y": 0.14},
            hovertemplate="azimuth=%{x}<br>tilt=%{y}<br>count=%{z}<extra></extra>",
        ),
        row=3,
        col=1,
    )
    figure.add_trace(
        go.Heatmap(
            x=tilt_labels,
            y=distance_labels,
            z=distance_tilt,
            colorscale="Oranges",
            colorbar={"title": "Count", "x": 1.02, "len": 0.28, "y": 0.14},
            hovertemplate="tilt=%{x}<br>d=%{y}<br>count=%{z}<extra></extra>",
        ),
        row=3,
        col=2,
    )
    figure.update_xaxes(tickangle=-35, row=1, col=1)
    figure.update_xaxes(tickangle=-35, row=2, col=1)
    figure.update_xaxes(tickangle=-35, row=3, col=1)
    figure.update_xaxes(tickangle=-35, row=3, col=2)
    figure.update_yaxes(title_text="Count", row=1, col=1)
    figure.update_yaxes(title_text="Count", row=2, col=1)
    polar_layout = {
        "radialaxis": {"title": "Count", "rangemode": "tozero"},
        "angularaxis": {
            "direction": "counterclockwise",
            "rotation": 0,
            "thetaunit": "degrees",
        },
    }
    figure.update_layout(
        title=(
            f"{distribution.spec.label} · fixed-range pose binning · "
            f"N={int(np.sum(mask))}"
        ),
        polar=polar_layout,
        polar2=polar_layout,
        height=1200,
        margin={"l": 70, "r": 70, "b": 130, "t": 90},
        showlegend=False,
    )
    return figure


def make_plane_normal_sensor_figure(distribution: PoseDistribution):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    figure = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "polar"}]],
        subplot_titles=(
            "Plane normal expressed in sensor coordinates",
            "Sensor-frame normal: tilt / azimuth",
        ),
        horizontal_spacing=0.06,
    )

    phi = np.linspace(0.0, 2.0 * np.pi, 50)
    theta = np.linspace(0.0, np.pi, 25)
    sphere_x = np.outer(np.sin(theta), np.cos(phi))
    sphere_y = np.outer(np.sin(theta), np.sin(phi))
    sphere_z = np.outer(np.cos(theta), np.ones_like(phi))
    figure.add_trace(
        go.Surface(
            x=sphere_x,
            y=sphere_y,
            z=sphere_z,
            opacity=0.08,
            colorscale=[[0.0, "#B0B0B0"], [1.0, "#B0B0B0"]],
            showscale=False,
            hoverinfo="skip",
            name="unit sphere",
        ),
        row=1,
        col=1,
    )

    hover = np.asarray(_hover_text(distribution), dtype=object)
    for accepted, symbol, label, color in (
        (True, "circle", "accepted", distribution.spec.color),
        (False, "x", "rejected", REJECTED_COLOR),
    ):
        mask = distribution.accepted == accepted
        if not np.any(mask):
            continue
        normals = distribution.plane_normals_sensor[mask]
        normal_hover = []
        for source, normal, tilt, azimuth in zip(
            hover[mask],
            normals,
            distribution.normal_tilt_deg[mask],
            distribution.normal_azimuth_sensor_deg[mask],
        ):
            normal_hover.append(
                f"{source}<br>n_s=[{normal[0]:.5f}, {normal[1]:.5f}, {normal[2]:.5f}]"
                f"<br>tilt from -Z_s={tilt:.3f}°"
                f"<br>normal azimuth in sensor={azimuth:.3f}°"
            )
        figure.add_trace(
            go.Scatter3d(
                x=normals[:, 0],
                y=normals[:, 1],
                z=normals[:, 2],
                mode="markers",
                marker={"size": 5, "symbol": symbol, "color": color},
                text=normal_hover,
                hovertemplate="%{text}<extra></extra>",
                name=label,
            ),
            row=1,
            col=1,
        )
        figure.add_trace(
            go.Scatterpolar(
                theta=distribution.normal_azimuth_sensor_deg[mask],
                r=distribution.normal_tilt_deg[mask],
                mode="markers",
                marker={"size": 8, "symbol": symbol, "color": color},
                text=normal_hover,
                hovertemplate="%{text}<extra></extra>",
                name=label,
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    axis_length = 1.15
    for axis, label in ((0, "+X_s"), (1, "+Y_s"), (2, "+Z_s")):
        end = [0.0, 0.0, 0.0]
        end[axis] = axis_length
        figure.add_trace(
            go.Scatter3d(
                x=[0.0, end[0]],
                y=[0.0, end[1]],
                z=[0.0, end[2]],
                mode="lines+text",
                line={"width": 6},
                text=[None, label],
                hoverinfo="skip",
                showlegend=False,
            ),
            row=1,
            col=1,
        )

    figure.update_layout(
        title=f"{distribution.spec.label} · fitted-plane normal diversity in sensor frame",
        scene={
            "xaxis": {"title": "n_x^S", "range": [-1.1, 1.1]},
            "yaxis": {"title": "n_y^S", "range": [-1.1, 1.1]},
            "zaxis": {"title": "n_z^S", "range": [-1.1, 1.1]},
            "aspectmode": "cube",
            "camera": {"eye": {"x": 1.45, "y": 1.45, "z": 1.15}},
        },
        polar={
            "radialaxis": {
                "title": "Tilt from -Z_s [deg]",
                "range": [0.0, 180.0],
            },
            "angularaxis": {
                "direction": "counterclockwise",
                "rotation": 0,
                "thetaunit": "degrees",
            },
        },
        height=720,
        margin={"l": 20, "r": 20, "b": 30, "t": 90},
        legend={"orientation": "h", "y": -0.06},
    )
    return figure


def _html_id(value: str) -> str:
    result = "".join(
        character.lower() if character.isalnum() else "-"
        for character in str(value)
    ).strip("-")
    while "--" in result:
        result = result.replace("--", "-")
    return result or "dataset"


def _profile_explorer_payload(
    distribution: PoseDistribution,
    assignments: dict[str, np.ndarray],
    *,
    max_points: int,
    accepted_only: bool,
) -> dict[str, Any]:
    """Build the browser payload for cumulative world-frame profile viewing.

    Profile points are transformed with the measurement-coordinate pose because
    the captured profile coordinates are expressed in that frame.  The displayed
    sensor marker and sensor XYZ axes are instead placed at the physical sensor
    origin, offset by ``sensor_origin_offset_z_mm`` along the sensor Z axis.
    """
    scans: list[dict[str, Any]] = []
    all_profile_world: list[np.ndarray] = []
    final_plane_profile_world: list[np.ndarray] = []
    for index, scan_id in enumerate(distribution.scan_ids):
        profile = _downsample_profile(
            distribution.profiles_sensor_xyz_mm[index], max_points
        )
        measurement_origin_world = distribution.measurement_origins_base_mm[index]
        physical_origin_world = distribution.physical_origins_base_mm[index]
        rotation_world_sensor = distribution.rotations_base_sensor[index]
        if len(profile):
            profile_world = (
                profile @ rotation_world_sensor.T
                + measurement_origin_world.reshape(1, 3)
            )
        else:
            profile_world = np.empty((0, 3), dtype=float)
        if len(profile_world):
            all_profile_world.append(profile_world)
            if not accepted_only or bool(distribution.accepted[index]):
                final_plane_profile_world.append(profile_world)
        scans.append(
            {
                "scanId": int(scan_id),
                "capture": distribution.capture_names[index],
                "accepted": bool(distribution.accepted[index]),
                "profileKey": distribution.profile_source_keys[index],
                "profileSensor": {
                    "x": np.round(profile[:, 0], 6).tolist() if len(profile) else [],
                    "y": np.round(profile[:, 1], 6).tolist() if len(profile) else [],
                    "z": np.round(profile[:, 2], 6).tolist() if len(profile) else [],
                },
                "profileWorld": {
                    "x": np.round(profile_world[:, 0], 6).tolist() if len(profile_world) else [],
                    "y": np.round(profile_world[:, 1], 6).tolist() if len(profile_world) else [],
                    "z": np.round(profile_world[:, 2], 6).tolist() if len(profile_world) else [],
                },
                "measurementOriginWorld": np.round(
                    measurement_origin_world, 6
                ).tolist(),
                "physicalSensorOriginWorld": np.round(
                    physical_origin_world, 6
                ).tolist(),
                "sensorRotationWorld": np.round(
                    rotation_world_sensor, 9
                ).tolist(),
                "bins": {
                    name: int(values[index])
                    for name, values in assignments.items()
                },
                "pose": {
                    "tilt": float(distribution.tilt_deg[index]),
                    "azimuth": float(distribution.azimuth_deg[index]),
                    "distance": float(distribution.origins_plane_mm[index, 2]),
                    "roll": float(distribution.roll_deg[index]),
                },
            }
        )

    physical_origins = np.asarray(
        distribution.physical_origins_base_mm, dtype=float
    ).reshape(-1, 3)
    plane_reference_parts = final_plane_profile_world or all_profile_world
    if plane_reference_parts:
        plane_reference_points = np.vstack(plane_reference_parts)
    else:
        plane_reference_points = physical_origins

    plane_centroid = np.asarray(
        distribution.plane_centroid_base_mm, dtype=float
    )
    plane_basis = np.asarray(distribution.plane_basis_base, dtype=float)
    plane_u_axis = plane_basis[:, 0]
    plane_v_axis = plane_basis[:, 1]
    plane_deltas = plane_reference_points - plane_centroid.reshape(1, 3)
    plane_u = plane_deltas @ plane_u_axis
    plane_v = plane_deltas @ plane_v_axis
    u_min, u_max = float(np.min(plane_u)), float(np.max(plane_u))
    v_min, v_max = float(np.min(plane_v)), float(np.max(plane_v))
    plane_pad = max(10.0, 0.08 * max(u_max - u_min, v_max - v_min, 1.0))
    plane_corners = np.asarray(
        [
            plane_centroid + (u_min - plane_pad) * plane_u_axis
            + (v_min - plane_pad) * plane_v_axis,
            plane_centroid + (u_max + plane_pad) * plane_u_axis
            + (v_min - plane_pad) * plane_v_axis,
            plane_centroid + (u_max + plane_pad) * plane_u_axis
            + (v_max + plane_pad) * plane_v_axis,
            plane_centroid + (u_min - plane_pad) * plane_u_axis
            + (v_max + plane_pad) * plane_v_axis,
        ],
        dtype=float,
    )

    scene_parts = list(all_profile_world)
    if len(physical_origins):
        scene_parts.append(physical_origins)
    scene_parts.append(plane_corners)
    scene_reference = np.vstack(scene_parts)
    raw_spans = np.ptp(scene_reference, axis=0)
    axis_length_mm = max(8.0, min(40.0, float(np.max(raw_spans)) * 0.08))
    rotations = np.asarray(distribution.rotations_base_sensor, dtype=float)
    if len(physical_origins) and len(rotations):
        axis_endpoints = (
            physical_origins[:, None, :]
            + axis_length_mm * np.transpose(rotations, (0, 2, 1))
        ).reshape(-1, 3)
        scene_reference = np.vstack((scene_reference, axis_endpoints))
    scene_min = np.min(scene_reference, axis=0)
    scene_max = np.max(scene_reference, axis=0)
    scene_padding = np.maximum(5.0, 0.05 * np.maximum(scene_max - scene_min, 1.0))
    scene_ranges = np.column_stack(
        (scene_min - scene_padding, scene_max + scene_padding)
    )

    return {
        "label": distribution.spec.label,
        "color": distribution.spec.color,
        "acceptedOnly": bool(accepted_only),
        "sensorOriginOffsetZMm": float(
            distribution.spec.sensor_origin_offset_z_mm
        ),
        "plane": {
            "centroidWorld": np.round(
                distribution.plane_centroid_base_mm, 6
            ).tolist(),
            "normalWorld": np.round(
                distribution.plane_normal_base, 9
            ).tolist(),
            "basisWorld": np.round(
                distribution.plane_basis_base, 9
            ).tolist(),
        },
        "fixedScene": {
            "planeCornersWorld": np.round(plane_corners, 6).tolist(),
            "rangesWorld": np.round(scene_ranges, 6).tolist(),
            "axisLengthMm": float(axis_length_mm),
            "cameraEye": {"x": 1.45, "y": 1.45, "z": 1.15},
        },
        "scans": scans,
    }


def _validation_html(validation: dict[str, Any]) -> str:
    status = escape(str(validation["status"]))
    checks = validation["bin_count_checks"]
    bin_rows = "".join(
        f"<tr><th>{escape(name)}</th><td>{item['sum']}</td>"
        f"<td>{item['expected']}</td><td>{item['occupied_bins']} / "
        f"{item['total_bins']}</td><td>{item['max_bin_count']}</td></tr>"
        for name, item in checks.items()
    )
    warnings = "".join(
        f"<li>{escape(item)}</li>" for item in validation["warnings"]
    )
    return f"""
    <details class="validation-details {status}">
      <summary>
        <span>Numerical validation</span>
        <span class="status-pill {status}">{status.upper()}</span>
      </summary>
      <div class="validation-content">
        <div class="validation-metrics">
          <div><span>Rotation reconstruction</span><strong>
            {validation['max_rotation_reconstruction_error_deg']:.3e}°</strong></div>
          <div><span>Plane-normal transform</span><strong>
            {validation['max_normal_transform_error']:.3e}</strong></div>
          <div><span>Signed-distance transform</span><strong>
            {validation['max_distance_transform_error_mm']:.3e} mm</strong></div>
          <div><span>Tilt/normal consistency</span><strong>
            {validation['max_tilt_normal_consistency_error_deg']:.3e}°</strong></div>
        </div>
        <div class="table-scroll compact-table">
          <table>
            <thead><tr><th>Parameter</th><th>Assigned</th><th>Expected</th>
            <th>Occupied</th><th>Largest bin</th></tr></thead>
            <tbody>{bin_rows}</tbody>
          </table>
        </div>
        <div class="limit-note">
          <strong>Interpretation limits</strong><ul>{warnings}</ul>
        </div>
      </div>
    </details>
    """


def _marginal_table_html(
    rows: list[dict[str, Any]], dataset_label: str
) -> str:
    selected = [row for row in rows if row["dataset"] == dataset_label]
    dataset_id = _html_id(dataset_label)
    body_parts: list[str] = []
    for row in selected:
        is_empty = int(row["all_count"]) == 0
        empty_class = " empty" if is_empty else ""
        disabled = " disabled" if is_empty else ""
        aria = escape(
            f'Select {row["parameter"]} bin {row["bin_label"]}'
        )
        body_parts.append(
            f'<tr class="marginal-row{empty_class}" '
            f'data-dataset="{dataset_id}" '
            f'data-parameter="{escape(str(row["parameter"]))}" '
            f'data-bin-index="{int(row["bin_index"])}" '
            f'data-bin-label="{escape(str(row["bin_label"]))}">'
            f'<td class="select-cell"><input type="checkbox" '
            f'class="bin-select" aria-label="{aria}"{disabled}></td>'
            f'<td><span class="parameter-badge">{escape(str(row["parameter"]))}</span></td>'
            f'<td>{escape(str(row["bin_label"]))}</td>'
            f'<td>{row["accepted_count"]}</td>'
            f'<td>{row["rejected_count"]}</td>'
            f'<td>{100.0 * row["accepted_fraction"]:.1f}%</td>'
            '</tr>'
        )
    body = "".join(body_parts)
    return (
        '<div class="table-scroll marginal-scroll">'
        '<table class="interactive-table"><thead><tr>'
        '<th></th><th>Parameter</th><th>Bin</th><th>Accepted</th><th>Rejected</th>'
        '<th>Accepted share</th>'
        f'</tr></thead><tbody>{body}</tbody></table></div>'
    )


def _joint_table_html(
    rows: list[dict[str, Any]], dataset_label: str, limit: int = 80
) -> str:
    selected = [
        row
        for row in rows
        if row["dataset"] == dataset_label and row["all_count"] > 0
    ][:limit]
    body = "".join(
        "<tr>"
        f"<td>{escape(str(row['tilt_bin']))}</td>"
        f"<td>{escape(str(row['azimuth_bin']))}</td>"
        f"<td>{escape(str(row['distance_bin']))}</td>"
        f"<td>{escape(str(row['roll_bin']))}</td>"
        f"<td>{row['accepted_count']}</td>"
        f"<td>{escape(', '.join(map(str, row['accepted_scan_ids'])))}</td>"
        "</tr>"
        for row in selected
    )
    return (
        '<div class="table-scroll joint-scroll"><table><thead><tr>'
        '<th>Tilt bin</th><th>Azimuth bin</th><th>d bin</th><th>Roll bin</th>'
        '<th>Accepted</th><th>Accepted scan IDs</th>'
        f'</tr></thead><tbody>{body}</tbody></table></div>'
    )


def _rotation_angle_deg(rotation: np.ndarray) -> float:
    rotation = np.asarray(rotation, dtype=float).reshape(3, 3)
    skew_vee = np.array(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ],
        dtype=float,
    )
    sine = 0.5 * float(np.linalg.norm(skew_vee))
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    return float(np.degrees(np.arctan2(sine, cosine)))


def _percent_change(value: float, baseline: float) -> float | None:
    if not np.isfinite(value) or not np.isfinite(baseline) or abs(baseline) < 1e-15:
        return None
    return float(100.0 * (value - baseline) / abs(baseline))


def _rms_dict_values(value: Any) -> float | None:
    if not isinstance(value, dict) or not value:
        return None
    values = np.asarray(list(value.values()), dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return None
    return float(np.sqrt(np.mean(values**2)))


def _finite_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def prepare_calibration_contexts(
    distributions: list[PoseDistribution],
) -> dict[str, CalibrationContext]:
    """Load full RANSAC-filtered scans once for localhost calibration requests."""
    from real_laser_handeye.calibrate_only import with_ransac_calibrate_only as calibration

    contexts: dict[str, CalibrationContext] = {}
    for distribution in distributions:
        spec = distribution.spec
        diagnostics = json.loads(spec.diagnostics_path.read_text(encoding="utf-8"))
        profile = diagnostics.get("profile_ransac", {})
        with contextlib.redirect_stdout(io.StringIO()):
            scans, rows = calibration.load_scans(
                spec.dataset_dir,
                use_profile_ransac=bool(profile.get("enabled", True)),
                ransac_threshold_mm=float(profile.get("threshold_mm", 0.15)),
                ransac_max_iterations=int(profile.get("max_iterations", 1000)),
                ransac_min_inliers=int(profile.get("min_inliers", 20)),
                ransac_min_inlier_ratio=float(
                    profile.get("min_inlier_ratio", 0.65)
                ),
                ransac_seed=int(profile.get("seed", 1701)),
                ransac_refine_iterations=int(
                    profile.get("refine_iterations", 3)
                ),
                ransac_reject_policy="skip",
            )
        scans_by_id = {int(scan.scan_id): scan for scan in scans}
        all_scan_ids = {int(row["scan_id"]) for row in rows}
        rejected_ids = {
            int(row["scan_id"])
            for row in rows
            if row.get("status") == "rejected"
        }
        expected_accepted = int(diagnostics.get("accepted_scan_count", len(scans)))
        if len(scans_by_id) != expected_accepted:
            raise RuntimeError(
                f"{spec.label}: RANSAC reload accepted {len(scans_by_id)} scans, "
                f"but full diagnostics report {expected_accepted}"
            )

        initial_path = Path(
            diagnostics.get(
                "initial_transform", "real_laser_handeye/initial_T_tcp_sensor.json"
            )
        )
        with contextlib.redirect_stdout(io.StringIO()):
            initial = calibration.load_transform(initial_path)
        contexts[_html_id(spec.label)] = CalibrationContext(
            spec=spec,
            initial_transform=np.asarray(initial, dtype=float),
            full_transform=_load_transform(spec.transform_path),
            full_diagnostics=diagnostics,
            scans_by_id=scans_by_id,
            all_scan_ids=all_scan_ids,
            ransac_rejected_ids=rejected_ids,
            lock=threading.Lock(),
            cache={},
        )
    return contexts


def run_selected_calibration(
    context: CalibrationContext,
    requested_scan_ids: list[int],
) -> dict[str, Any]:
    """Calibrate one selected scan subset and compare it with the full result."""
    from real_laser_handeye.calibrate_only import with_ransac_calibrate_only as calibration

    requested = tuple(sorted({int(value) for value in requested_scan_ids}))
    if not requested:
        raise ValueError("no scans were selected")
    unknown = sorted(set(requested) - context.all_scan_ids)
    if unknown:
        raise ValueError(f"unknown scan IDs: {unknown}")

    with context.lock:
        cached = context.cache.get(requested)
        if cached is not None:
            return cached

        used_ids = [value for value in requested if value in context.scans_by_id]
        rejected_ids = [
            value for value in requested if value in context.ransac_rejected_ids
        ]
        if len(used_ids) < 4:
            raise ValueError(
                "at least 4 RANSAC-accepted scans are required; "
                f"selected={len(requested)}, usable={len(used_ids)}, "
                f"rejected={len(rejected_ids)}"
            )

        selected_scans = [context.scans_by_id[value] for value in used_ids]
        full_scans = list(context.scans_by_id.values())
        diagnostics = context.full_diagnostics
        with contextlib.redirect_stdout(io.StringIO()):
            result = calibration.calibrate_single_plane(
                selected_scans,
                T_init=context.initial_transform,
                max_iter=4000,
                tol=1e-7,
                plane_offset_mode=str(diagnostics.get("plane_offset_mode", "joint")),
                max_translation_offset_condition=1e6,
            )

        selected_transform = np.asarray(result.T_ef_s, dtype=float)
        selected_rms = float(result.plane_rms_history[-1])
        full_rms = float(diagnostics["final_plane_rms_mm"])
        selected_accepted = bool(
            result.converged
            and np.isfinite(selected_rms)
            and selected_rms
            <= float(diagnostics.get("max_final_plane_rms_mm", 2.0))
        )
        full_eval_residuals, _, _, _ = calibration._fit_profiled_plane_residuals(
            full_scans,
            selected_transform,
        )
        selected_on_full_rms = float(
            np.sqrt(np.mean(np.asarray(full_eval_residuals, dtype=float) ** 2))
        )

        uncertainty_warning = None
        try:
            selected_uncertainty = calibration.compute_handeye_observability_uncertainty(
                scans=selected_scans,
                T_final=selected_transform,
                characteristic_length_mm=100.0,
                finite_difference_step_mm=1e-3,
                svd_relative_tolerance=1e-10,
            )
        except Exception as exc:
            selected_uncertainty = None
            uncertainty_warning = f"selected uncertainty analysis failed: {exc}"

        full_transform = context.full_transform
        translation_delta = selected_transform[:3, 3] - full_transform[:3, 3]
        translation_delta_norm = float(np.linalg.norm(translation_delta))
        full_translation_norm = float(np.linalg.norm(full_transform[:3, 3]))
        rotation_delta_deg = _rotation_angle_deg(
            full_transform[:3, :3].T @ selected_transform[:3, :3]
        )
        full_rotation_magnitude_deg = _rotation_angle_deg(full_transform[:3, :3])

        full_uncertainty = diagnostics.get("observability_uncertainty") or {}
        selected_condition = (
            None
            if selected_uncertainty is None
            else _finite_or_none(selected_uncertainty.get("condition_number"))
        )
        full_condition = _finite_or_none(full_uncertainty.get("condition_number"))
        selected_rotation_std = (
            None
            if selected_uncertainty is None
            else _rms_dict_values(selected_uncertainty.get("std_rotation_deg"))
        )
        full_rotation_std = _rms_dict_values(
            full_uncertainty.get("std_rotation_deg")
        )
        selected_translation_std = (
            None
            if selected_uncertainty is None
            else _rms_dict_values(selected_uncertainty.get("std_translation_mm"))
        )
        full_translation_std = _rms_dict_values(
            full_uncertainty.get("std_translation_mm")
        )

        def metric(
            full_value: float | None,
            selected_value: float | None,
            *,
            unit: str,
            lower_is_better: bool,
        ) -> dict[str, Any]:
            percent = (
                None
                if full_value is None or selected_value is None
                else _percent_change(selected_value, full_value)
            )
            return {
                "full": full_value,
                "selected": selected_value,
                "unit": unit,
                "percent_change": percent,
                "lower_is_better": lower_is_better,
            }

        response: dict[str, Any] = {
            "success": True,
            "dataset": context.spec.label,
            "requested_scan_ids": list(requested),
            "used_scan_ids": used_ids,
            "ransac_rejected_scan_ids": rejected_ids,
            "counts": {
                "requested": len(requested),
                "selected_usable": len(used_ids),
                "full_usable": len(full_scans),
                "scan_reduction_percent": float(
                    100.0 * (1.0 - len(used_ids) / len(full_scans))
                ),
            },
            "solver": {
                "converged": bool(result.converged),
                "accepted": selected_accepted,
                "iterations": int(result.iterations),
                "selected_training_rms_mm": selected_rms,
            },
            "transform_difference": {
                "translation_delta_xyz_mm": translation_delta.tolist(),
                "translation_delta_norm_mm": translation_delta_norm,
                "translation_percent_of_full_magnitude": (
                    None
                    if full_translation_norm < 1e-15
                    else float(100.0 * translation_delta_norm / full_translation_norm)
                ),
                "rotation_delta_deg": rotation_delta_deg,
                "rotation_percent_of_full_magnitude": (
                    None
                    if full_rotation_magnitude_deg < 1e-15
                    else float(
                        100.0 * rotation_delta_deg / full_rotation_magnitude_deg
                    )
                ),
            },
            "metrics": {
                "full_evaluation_plane_rms": metric(
                    full_rms,
                    selected_on_full_rms,
                    unit="mm",
                    lower_is_better=True,
                ),
                "condition_number": metric(
                    full_condition,
                    selected_condition,
                    unit="",
                    lower_is_better=True,
                ),
                "rotation_std_rms": metric(
                    full_rotation_std,
                    selected_rotation_std,
                    unit="deg",
                    lower_is_better=True,
                ),
                "translation_std_rms": metric(
                    full_translation_std,
                    selected_translation_std,
                    unit="mm",
                    lower_is_better=True,
                ),
            },
            "selected_transform": selected_transform.tolist(),
            "full_transform": full_transform.tolist(),
            "notes": [
                (
                    "The selected-training RMS uses only selected scans; the main "
                    "RMS percentage evaluates the selected transform on the full "
                    "RANSAC-accepted dataset for a like-for-like comparison."
                )
            ],
        }
        if uncertainty_warning:
            response["notes"].append(uncertainty_warning)
        if not selected_accepted:
            response["notes"].append(
                "The selected calibration did not satisfy the full-run convergence/RMS "
                "acceptance rule; treat its transform differences as diagnostic only."
            )
        context.cache[requested] = response
        return response


def save_binning_html(
    distributions: list[PoseDistribution],
    config: BinningConfig,
    assignments_by_label: dict[str, dict[str, np.ndarray]],
    marginal_rows_all: list[dict[str, Any]],
    joint_rows_all: list[dict[str, Any]],
    validations: list[dict[str, Any]],
    output: Path,
    *,
    profile_max_points: int = 500,
) -> None:
    sections: list[str] = []
    plotly_included = False
    explorer_data: dict[str, Any] = {}
    for distribution, validation in zip(distributions, validations):
        dataset_id = _html_id(distribution.spec.label)
        assignments = assignments_by_label[distribution.spec.label]
        explorer_data[dataset_id] = _profile_explorer_payload(
            distribution,
            assignments,
            max_points=profile_max_points,
            accepted_only=config.accepted_only,
        )
        binning_figure = make_binning_figure(distribution, config, assignments)
        binning_html = binning_figure.to_html(
            full_html=False,
            include_plotlyjs=True if not plotly_included else False,
            config={"responsive": True, "displaylogo": False},
        )
        plotly_included = True
        normal_figure = make_plane_normal_sensor_figure(distribution)
        normal_html = normal_figure.to_html(
            full_html=False,
            include_plotlyjs=False,
            config={"responsive": True, "displaylogo": False},
        )
        profile_available = sum(
            len(profile) > 0 for profile in distribution.profiles_sensor_xyz_mm
        )
        profile_missing = len(distribution.scan_ids) - profile_available
        selected_count = (
            int(np.sum(distribution.accepted))
            if config.accepted_only
            else len(distribution.scan_ids)
        )
        occupied_joint = sum(
            1
            for row in joint_rows_all
            if row["dataset"] == distribution.spec.label
            and row["all_count"] > 0
        )
        sections.append(
            f"""
            <section class="dataset-section" id="dataset-{dataset_id}">
              <div class="dataset-heading">
                <div>
                  <p class="eyebrow">Dataset</p>
                  <h2>{escape(distribution.spec.label)}</h2>
                </div>
                <a class="top-link" href="#top">Back to top ↑</a>
              </div>
              <div class="metric-grid">
                <div class="metric"><span>Captured scans</span><strong>{len(distribution.scan_ids)}</strong></div>
                <div class="metric"><span>Binning scans</span><strong>{selected_count}</strong></div>
                <div class="metric"><span>Profiles found</span><strong>{profile_available}</strong><small>{profile_missing} missing</small></div>
                <div class="metric"><span>Occupied 4-D bins</span><strong>{occupied_joint}</strong></div>
              </div>
              {_validation_html(validation)}

              <div class="content-block">
                <div class="block-heading">
                  <div><span class="step">01</span><h3>Pose coverage</h3></div>
                  <p>Tilt ∠(nˢ, −Zₛ), azimuth, distance and roll occupancy.</p>
                </div>
                <div class="plot-shell">{binning_html}</div>
              </div>

              <div class="content-block">
                <div class="block-heading">
                  <div><span class="step">02</span><h3>Marginal bins and cumulative world-frame profiles</h3></div>
                  <p>Check multiple rows to accumulate their scans in one World/Base-coordinate 3-D scene.</p>
                </div>
                <div class="explorer-grid">
                  <div class="table-panel">
                    {_marginal_table_html(marginal_rows_all, distribution.spec.label)}
                  </div>
                  <div class="profile-panel">
                    <div class="profile-toolbar">
                      <div>
                        <p class="eyebrow">Selected marginal bins</p>
                        <h4 id="{dataset_id}-profile-title">No bins selected</h4>
                      </div>
                      <div class="profile-options">
                        <label class="checkbox-label">
                          <input type="checkbox" id="{dataset_id}-include-rejected">
                          Include rejected
                        </label>
                        <label class="checkbox-label">
                          <input type="checkbox" id="{dataset_id}-show-axes" checked>
                          Show sensor XYZ axes
                        </label>
                        <button type="button" class="clear-selection" id="{dataset_id}-clear-bins">
                          Clear selected bins
                        </button>
                      </div>
                    </div>
                    <p id="{dataset_id}-profile-meta" class="profile-meta">
                      Scene coordinates are World/Base X,Y,Z. Profiles use their captured measurement-coordinate pose; the displayed marker and XYZ axes use the physical sensor origin at -80 mm along sensor Z.
                    </p>
                    <div id="{dataset_id}-selected-bin-list" class="selected-bin-list">No bins selected.</div>
                    <div id="{dataset_id}-profile-plot" class="profile-plot"></div>
                    <div id="{dataset_id}-scan-list" class="scan-list"></div>
                    <div class="calibration-panel">
                      <div class="calibration-heading">
                        <div>
                          <p class="eyebrow">Selected-scan calibration</p>
                          <h4>Compare with full calibration</h4>
                        </div>
                        <button type="button" class="calibrate-selection"
                          id="{dataset_id}-calibrate-selection" disabled>
                          Select at least 4 usable scans
                        </button>
                      </div>
                      <p class="calibration-help">
                        Runs the existing single-plane calibration with the selected,
                        RANSAC-accepted scans. This requires the localhost server mode:
                        <code>PYTHONPATH=. python3 real_laser_handeye/calibrate_only/analyze_sensor_pose_binning.py --serve</code>
                      </p>
                      <div id="{dataset_id}-calibration-status" class="calibration-status">
                        Select bins to enable calibration.
                      </div>
                      <div id="{dataset_id}-calibration-result" class="calibration-result"></div>
                    </div>
                  </div>
                </div>
              </div>

              <div class="content-block">
                <div class="block-heading">
                  <div><span class="step">03</span><h3>Plane-normal diversity</h3></div>
                  <p>The fitted plane normal expressed in each sensor coordinate frame.</p>
                </div>
                <div class="plot-shell">{normal_html}</div>
              </div>

              <details class="joint-details content-block">
                <summary><span><span class="step">04</span>Non-empty 4-D bins</span><small>Open table</small></summary>
                <p class="section-note">Sorted by accepted count. The complete table is also written to CSV.</p>
                {_joint_table_html(joint_rows_all, distribution.spec.label)}
              </details>
            </section>
            """
        )

    edges_text = {
        "tilt": config.tilt_edges_deg.tolist(),
        "azimuth": config.azimuth_edges_deg.tolist(),
        "distance": config.distance_edges_mm.tolist(),
        "roll": config.roll_edges_deg.tolist(),
    }
    navigation = "".join(
        f'<a href="#dataset-{_html_id(item.spec.label)}">{escape(item.spec.label)}</a>'
        for item in distributions
    )
    explorer_json = json.dumps(explorer_data, ensure_ascii=False, separators=(",", ":"))
    document = f"""<!doctype html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Sensor pose distribution explorer</title>
  <style>
    :root {{
      --bg: #f4f6f8;
      --panel: #ffffff;
      --text: #1f2933;
      --muted: #667085;
      --border: #dfe3e8;
      --soft: #f8fafc;
      --accent: #315f9f;
      --accent-soft: #eaf1fb;
      --warning: #a66b00;
      --success: #287a45;
    }}
    * {{ box-sizing: border-box; }}
    html {{ scroll-behavior: smooth; }}
    body {{ margin: 0; background: var(--bg); color: var(--text); font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    .page {{ width: min(1500px, calc(100% - 32px)); margin: 0 auto; padding: 28px 0 64px; }}
    .page-header {{ display: flex; justify-content: space-between; align-items: flex-start; gap: 28px; padding: 26px 30px; background: var(--panel); border: 1px solid var(--border); border-radius: 16px; }}
    .page-header h1 {{ margin: 2px 0 8px; font-size: clamp(1.55rem, 3vw, 2.25rem); }}
    .page-header p {{ max-width: 920px; margin: 0; color: var(--muted); line-height: 1.55; }}
    .eyebrow {{ margin: 0 0 4px; color: var(--accent); text-transform: uppercase; letter-spacing: .1em; font-size: .72rem; font-weight: 700; }}
    .dataset-nav {{ display: flex; gap: 8px; flex-wrap: wrap; }}
    .dataset-nav a, .top-link {{ color: var(--accent); text-decoration: none; border: 1px solid var(--border); background: var(--soft); padding: 8px 11px; border-radius: 9px; font-size: .86rem; white-space: nowrap; }}
    .method-details {{ margin-top: 14px; background: var(--panel); border: 1px solid var(--border); border-radius: 12px; padding: 0 18px; }}
    .method-details summary {{ cursor: pointer; padding: 14px 0; font-weight: 650; }}
    .method-details p {{ color: var(--muted); line-height: 1.55; }}
    code {{ background: #eef1f4; padding: 2px 5px; border-radius: 4px; overflow-wrap: anywhere; }}
    .dataset-section {{ margin-top: 24px; background: var(--panel); border: 1px solid var(--border); border-radius: 16px; padding: 28px; }}
    .dataset-heading {{ display: flex; justify-content: space-between; align-items: center; gap: 20px; }}
    .dataset-heading h2 {{ margin: 0; font-size: 1.7rem; }}
    .metric-grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; margin: 20px 0; }}
    .metric {{ background: var(--soft); border: 1px solid var(--border); border-radius: 11px; padding: 13px 15px; display: grid; gap: 2px; }}
    .metric span {{ color: var(--muted); font-size: .8rem; }}
    .metric strong {{ font-size: 1.35rem; font-weight: 700; }}
    .metric small {{ color: var(--muted); }}
    .validation-details {{ border: 1px solid var(--border); border-left: 4px solid var(--success); border-radius: 11px; background: var(--soft); margin-bottom: 24px; }}
    .validation-details.warning {{ border-left-color: var(--warning); }}
    .validation-details summary {{ cursor: pointer; display: flex; justify-content: space-between; align-items: center; padding: 13px 15px; font-weight: 650; }}
    .status-pill {{ font-size: .72rem; letter-spacing: .06em; border-radius: 999px; padding: 4px 8px; background: #e7f5ec; color: var(--success); }}
    .status-pill.warning {{ background: #fff3d6; color: var(--warning); }}
    .validation-content {{ padding: 0 15px 15px; }}
    .validation-metrics {{ display: grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap: 8px; margin-bottom: 12px; }}
    .validation-metrics div {{ background: var(--panel); border: 1px solid var(--border); border-radius: 8px; padding: 10px; display: grid; gap: 4px; }}
    .validation-metrics span {{ color: var(--muted); font-size: .76rem; }}
    .validation-metrics strong {{ font-size: .9rem; }}
    .limit-note {{ color: var(--muted); font-size: .84rem; line-height: 1.45; }}
    .limit-note ul {{ margin-bottom: 0; }}
    .content-block {{ border-top: 1px solid var(--border); padding-top: 24px; margin-top: 26px; }}
    .block-heading {{ display: flex; justify-content: space-between; align-items: flex-end; gap: 24px; margin-bottom: 12px; }}
    .block-heading > div {{ display: flex; align-items: center; gap: 9px; }}
    .block-heading h3 {{ margin: 0; font-size: 1.15rem; }}
    .block-heading p, .section-note {{ margin: 0; color: var(--muted); font-size: .86rem; }}
    .step {{ display: inline-grid; place-items: center; min-width: 30px; height: 24px; border-radius: 7px; background: var(--accent-soft); color: var(--accent); font-size: .72rem; font-weight: 750; }}
    .plot-shell {{ width: 100%; min-width: 0; overflow: hidden; border: 1px solid var(--border); border-radius: 11px; background: #fff; }}
    .explorer-grid {{ display: grid; grid-template-columns: minmax(430px, .88fr) minmax(520px, 1.12fr); gap: 16px; align-items: start; }}
    .table-panel, .profile-panel {{ border: 1px solid var(--border); border-radius: 11px; background: var(--panel); min-width: 0; }}
    .profile-panel {{ padding: 15px; position: sticky; top: 12px; }}
    .profile-toolbar {{ display: flex; justify-content: space-between; align-items: flex-start; gap: 14px; }}
    .profile-toolbar h4 {{ margin: 0; font-size: 1rem; }}
    .profile-options {{ display: flex; flex-direction: column; gap: 7px; align-items: flex-start; }}
    .clear-selection {{ border: 1px solid var(--border); background: var(--soft); color: var(--text); border-radius: 7px; padding: 6px 9px; font-size: .78rem; cursor: pointer; }}
    .clear-selection:hover {{ background: var(--accent-soft); }}
    .checkbox-label {{ display: flex; align-items: center; gap: 6px; color: var(--muted); font-size: .82rem; white-space: nowrap; }}
    .profile-meta {{ color: var(--muted); font-size: .83rem; line-height: 1.4; min-height: 2.3em; }}
    .selected-bin-list {{ color: var(--muted); font-size: .78rem; line-height: 1.45; padding: 8px 10px; margin: 8px 0 4px; border: 1px solid var(--border); border-radius: 8px; background: var(--soft); max-height: 74px; overflow: auto; }}
    .profile-plot {{ width: 100%; height: 620px; }}
    .scan-list {{ color: var(--muted); font-size: .78rem; line-height: 1.5; max-height: 72px; overflow: auto; border-top: 1px solid var(--border); padding-top: 8px; }}
    .calibration-panel {{ margin-top: 14px; padding: 14px; border: 1px solid var(--border); border-radius: 10px; background: var(--soft); }}
    .calibration-heading {{ display: flex; justify-content: space-between; align-items: center; gap: 14px; }}
    .calibration-heading h4 {{ margin: 0; }}
    .calibrate-selection {{ border: 0; border-radius: 8px; padding: 9px 12px; color: #fff; background: var(--accent); font-weight: 700; cursor: pointer; }}
    .calibrate-selection:disabled {{ cursor: not-allowed; opacity: .48; }}
    .calibration-help, .calibration-status {{ color: var(--muted); font-size: .8rem; line-height: 1.45; }}
    .calibration-status.running {{ color: var(--accent); font-weight: 650; }}
    .calibration-status.error {{ color: #b42318; font-weight: 650; }}
    .calibration-status.warning {{ color: var(--warning); font-weight: 650; }}
    .calibration-status.success {{ color: var(--success); font-weight: 650; }}
    .calibration-result:empty {{ display: none; }}
    .calibration-result {{ margin-top: 10px; }}
    .calibration-result-grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 8px; }}
    .calibration-result-card {{ padding: 10px; border: 1px solid var(--border); border-radius: 8px; background: #fff; display: grid; gap: 3px; }}
    .calibration-result-card span {{ color: var(--muted); font-size: .74rem; }}
    .calibration-result-card strong {{ font-size: 1rem; }}
    .change-better {{ color: var(--success); }}
    .change-worse {{ color: #b42318; }}
    .calibration-matrix {{ overflow: auto; margin-top: 9px; padding: 9px; border-radius: 7px; background: #111827; color: #f8fafc; font-size: .72rem; line-height: 1.45; }}
    .calibration-notes {{ color: var(--muted); font-size: .76rem; line-height: 1.45; }}
    .world-scan-viewer {{ margin-top: 14px; padding: 12px; border: 1px solid var(--border); border-radius: 10px; background: #fff; min-width: 0; }}
    .world-scan-heading {{ display: flex; align-items: flex-start; justify-content: space-between; gap: 12px; }}
    .world-scan-heading h5 {{ margin: 0; font-size: .96rem; }}
    .world-scan-heading p {{ margin: 4px 0 0; color: var(--muted); font-size: .76rem; line-height: 1.4; }}
    .world-scan-tabs {{ display: flex; flex-wrap: wrap; gap: 6px; margin: 10px 0 8px; }}
    .world-scan-tab, .world-scan-fullscreen {{ border: 1px solid var(--border); background: var(--soft); color: var(--text); border-radius: 7px; padding: 7px 9px; font-size: .76rem; cursor: pointer; }}
    .world-scan-tab:hover, .world-scan-fullscreen:hover {{ background: var(--accent-soft); }}
    .world-scan-tab.active {{ border-color: var(--accent); background: var(--accent); color: #fff; }}
    .world-scan-tab:disabled {{ cursor: wait; opacity: .62; }}
    .world-scan-status {{ min-height: 1.4em; color: var(--muted); font-size: .76rem; line-height: 1.4; }}
    .world-scan-status.error {{ color: #b42318; font-weight: 650; }}
    .world-scan-plot {{ width: 100%; height: 620px; min-height: 420px; }}
    .world-scan-viewer:fullscreen {{ width: 100vw; height: 100vh; margin: 0; border: 0; border-radius: 0; padding: 18px; display: flex; flex-direction: column; background: #fff; }}
    .world-scan-viewer:fullscreen .world-scan-plot {{ flex: 1 1 auto; height: auto; min-height: 0; }}
    .world-scan-viewer:fullscreen .world-scan-heading h5 {{ font-size: 1.15rem; }}
    .sphere-evaluation {{ margin-top: 10px; padding-top: 12px; border-top: 1px solid var(--border); }}
    .sphere-evaluation:empty {{ display: none; }}
    .sphere-evaluation h6 {{ margin: 0 0 4px; font-size: .9rem; }}
    .sphere-evaluation-summary {{ margin: 0 0 9px; color: var(--muted); font-size: .75rem; line-height: 1.45; }}
    .sphere-evaluation-grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 7px; }}
    .sphere-evaluation-card {{ padding: 9px; border: 1px solid var(--border); border-radius: 8px; background: var(--soft); display: grid; gap: 2px; }}
    .sphere-evaluation-card span {{ color: var(--muted); font-size: .7rem; }}
    .sphere-evaluation-card strong {{ font-size: .9rem; }}
    .sphere-evaluation details {{ margin-top: 9px; color: var(--muted); font-size: .74rem; }}
    .sphere-evaluation pre {{ overflow: auto; padding: 8px; border-radius: 7px; background: #111827; color: #f8fafc; line-height: 1.45; }}
    .table-scroll {{ overflow: auto; max-width: 100%; }}
    .marginal-scroll {{ max-height: 590px; }}
    .joint-scroll {{ max-height: 520px; margin-top: 12px; }}
    table {{ width: 100%; border-collapse: separate; border-spacing: 0; font-size: .83rem; }}
    th, td {{ padding: 9px 10px; text-align: right; border-bottom: 1px solid var(--border); white-space: nowrap; }}
    th {{ position: sticky; top: 0; z-index: 1; background: #f3f5f7; color: #48515c; font-size: .77rem; }}
    th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
    tbody tr:last-child td {{ border-bottom: 0; }}
    .interactive-table tbody tr {{ cursor: pointer; transition: background .12s ease; }}
    .select-cell {{ text-align: center !important; width: 34px; }}
    .bin-select {{ cursor: pointer; }}
    .interactive-table tbody tr:hover {{ background: #f4f8fd; }}
    .interactive-table tbody tr.selected {{ background: var(--accent-soft); box-shadow: inset 3px 0 0 var(--accent); }}
    .interactive-table tbody tr.empty {{ opacity: .46; }}
    .parameter-badge {{ display: inline-block; padding: 3px 7px; border-radius: 999px; background: #edf0f3; font-size: .72rem; text-transform: capitalize; }}
    .compact-table table {{ width: auto; min-width: 560px; }}
    .joint-details summary {{ cursor: pointer; display: flex; align-items: center; justify-content: space-between; gap: 15px; font-weight: 700; }}
    .joint-details summary > span {{ display: flex; align-items: center; gap: 9px; }}
    .joint-details summary small {{ color: var(--muted); font-weight: 500; }}
    @media (max-width: 1050px) {{
      .explorer-grid {{ grid-template-columns: 1fr; }}
      .profile-panel {{ position: static; }}
      .metric-grid, .validation-metrics {{ grid-template-columns: repeat(2, minmax(0,1fr)); }}
    }}
    @media (max-width: 680px) {{
      .page {{ width: min(100% - 16px, 1500px); padding-top: 8px; }}
      .page-header, .dataset-section {{ padding: 18px; border-radius: 12px; }}
      .page-header, .block-heading {{ align-items: flex-start; flex-direction: column; }}
      .metric-grid, .validation-metrics {{ grid-template-columns: 1fr 1fr; }}
      .sphere-evaluation-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
      .explorer-grid {{ display: block; }}
      .profile-panel {{ margin-top: 12px; }}
      .profile-plot {{ height: 500px; }}
      .world-scan-plot {{ height: 500px; }}
      .calibration-heading {{ align-items: flex-start; flex-direction: column; }}
      .calibration-result-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <main class="page" id="top">
    <header class="page-header">
      <div>
        <p class="eyebrow">Calibration geometry report</p>
        <h1>Sensor pose distribution explorer</h1>
        <p>
          Review pose occupancy first, then check one or more marginal-bin rows to accumulate
          the corresponding 3-D sensor poses and measured profile points in World/Base coordinates. Plane-normal
          diversity and the detailed four-dimensional bins follow afterward. In localhost
          server mode, the selected scans can be recalibrated and compared with the full result.
        </p>
      </div>
      <nav class="dataset-nav">{navigation}</nav>
    </header>
    <details class="method-details">
      <summary>Definitions and bin edges</summary>
      <p>
        Binning uses {'accepted scans only' if config.accepted_only else 'all captured scans'}.
        Parameters follow <code>R_plane_sensor = Rz(azimuth) Ry(tilt) Rz(roll)</code>.
        Each profile is read in sensor coordinates and transformed into World/Base coordinates.
        Bins within the same parameter are combined by union (OR), while different
        parameters are combined by intersection (AND). Duplicate scans are drawn once,
        and sensor axes originate at the physical sensor origin using the configured
        signed sensor-Z offset.
      </p>
      <p><strong>Edges:</strong> <code>{escape(json.dumps(edges_text, ensure_ascii=False))}</code></p>
    </details>
    {''.join(sections)}
  </main>
  <script>
    const PROFILE_DATA = {explorer_json};
    const WORLD_SCAN_VIEWER_STATE = new Map();

    function vectorEnd(origin, rotation, axis, length) {{
      return [
        origin[0] + rotation[0][axis] * length,
        origin[1] + rotation[1][axis] * length,
        origin[2] + rotation[2][axis] * length
      ];
    }}

    function selectedBins(datasetId) {{
      return Array.from(
        document.querySelectorAll(`.marginal-row[data-dataset="${{datasetId}}"]`)
      ).filter(row => row.querySelector('.bin-select')?.checked).map(row => ({{
        parameter: row.dataset.parameter,
        binIndex: Number(row.dataset.binIndex),
        binLabel: row.dataset.binLabel,
        row
      }}));
    }}

    function scanMatchesBins(scan, bins) {{
      const binsByParameter = new Map();
      bins.forEach(bin => {{
        if (!binsByParameter.has(bin.parameter)) {{
          binsByParameter.set(bin.parameter, []);
        }}
        binsByParameter.get(bin.parameter).push(bin);
      }});
      return Array.from(binsByParameter.values()).every(parameterBins =>
        parameterBins.some(bin =>
          scan.bins[bin.parameter] === bin.binIndex
        )
      );
    }}

    function selectionExpression(bins) {{
      const binsByParameter = new Map();
      bins.forEach(bin => {{
        if (!binsByParameter.has(bin.parameter)) {{
          binsByParameter.set(bin.parameter, []);
        }}
        binsByParameter.get(bin.parameter).push(bin.binLabel);
      }});
      return Array.from(binsByParameter.entries()).map(([parameter, labels]) =>
        `(${{labels.map(label => `${{parameter}} ${{label}}`).join(' OR ')}})`
      ).join(' AND ');
    }}

    function currentSelectedScans(datasetId) {{
      const dataset = PROFILE_DATA[datasetId];
      if (!dataset) return [];
      const bins = selectedBins(datasetId);
      if (!bins.length) return [];
      const includeRejected = document.getElementById(
        `${{datasetId}}-include-rejected`
      ).checked;
      return dataset.scans.filter(scan =>
        scanMatchesBins(scan, bins) && (includeRejected || scan.accepted)
      );
    }}

    function planeTrace(dataset) {{
      const corners = dataset.fixedScene.planeCornersWorld;
      return {{
        type: 'mesh3d',
        x: corners.map(p => p[0]),
        y: corners.map(p => p[1]),
        z: corners.map(p => p[2]),
        i: [0, 0], j: [1, 2], k: [2, 3],
        color: '#9aa4b2', opacity: 0.12,
        hoverinfo: 'skip', showlegend: false,
        name: 'Fixed full-data fitted plane'
      }};
    }}

    function renderProfiles(datasetId) {{
      const dataset = PROFILE_DATA[datasetId];
      const meta = document.getElementById(`${{datasetId}}-profile-meta`);
      const title = document.getElementById(`${{datasetId}}-profile-title`);
      const binList = document.getElementById(`${{datasetId}}-selected-bin-list`);
      const scanList = document.getElementById(`${{datasetId}}-scan-list`);
      if (!dataset) {{
        if (meta) meta.textContent = 'Dataset payload is missing.';
        return;
      }}

      const bins = selectedBins(datasetId);
      const includeRejected = Boolean(
        document.getElementById(`${{datasetId}}-include-rejected`)?.checked
      );
      const showAxes = Boolean(
        document.getElementById(`${{datasetId}}-show-axes`)?.checked
      );
      const selected = currentSelectedScans(datasetId);
      const available = selected.filter(scan => scan.profileWorld.x.length > 0);
      const missing = selected.length - available.length;
      const acceptedCount = selected.filter(scan => scan.accepted).length;
      const rejectedCount = selected.length - acceptedCount;

      // Update the selection UI before touching Plotly. A 3-D/WebGL failure must
      // never prevent a checked bin from appearing in the selected-bin summary.
      if (title) {{
        title.textContent = bins.length
          ? `${{bins.length}} bins selected · ${{selected.length}} unique scans`
          : 'No bins selected';
      }}
      if (binList) {{
        binList.textContent = bins.length
          ? 'Selection rule: ' + selectionExpression(bins)
          : 'No bins selected.';
      }}
      if (meta) {{
        meta.textContent =
          `${{selected.length}} unique scans · ${{acceptedCount}} accepted · ` +
          `${{rejectedCount}} rejected · ${{available.length}} profiles drawn` +
          (missing ? ` · ${{missing}} missing profile arrays` : '') +
          ` · fixed full-data plane/view · scene frame: World/Base XYZ` +
          ` · physical-origin offset: ${{dataset.sensorOriginOffsetZMm.toFixed(1)}} mm along sensor Z`;
      }}
      if (scanList) {{
        scanList.textContent = selected.length
          ? 'Scan IDs: ' + selected.map(scan => scan.scanId + (scan.accepted ? '' : ' (R)')).join(', ')
          : 'No matching scan IDs.';
      }}
      const calibrationButton = document.getElementById(
        `${{datasetId}}-calibrate-selection`
      );
      const calibrationStatus = document.getElementById(
        `${{datasetId}}-calibration-status`
      );
      const calibrationResult = document.getElementById(
        `${{datasetId}}-calibration-result`
      );
      const usableCount = selected.filter(scan => scan.accepted).length;
      const selectionSignature = selected.map(scan => scan.scanId).sort((a, b) => a-b).join(',');
      if (calibrationButton && calibrationStatus && calibrationResult) {{
        if (calibrationButton.dataset.selectionSignature !== selectionSignature) {{
          calibrationButton.dataset.selectionSignature = selectionSignature;
          clearWorldScanViewer(datasetId);
          calibrationResult.innerHTML = '';
          calibrationStatus.className = 'calibration-status';
          calibrationStatus.textContent = usableCount >= 4
            ? `${{usableCount}} usable selected scans are ready.`
            : 'Select bins containing at least 4 RANSAC-accepted scans.';
        }}
        calibrationButton.disabled = usableCount < 4;
        calibrationButton.textContent = usableCount >= 4
          ? `Calibrate ${{usableCount}} usable scans`
          : 'Select at least 4 usable scans';
      }}

      if (!window.Plotly) {{
        if (meta) meta.textContent += ' · Plotly unavailable: selection is active, but the 3-D scene cannot be rendered.';
        return;
      }}

      const traces = [];

      const plane = planeTrace(dataset);
      if (plane) traces.push(plane);

      available.forEach(scan => {{
        const n = scan.profileWorld.x.length;
        const customdata = Array.from({{length: n}}, (_, index) => [
          scan.profileSensor.x[index],
          scan.profileSensor.y[index],
          scan.profileSensor.z[index]
        ]);
        traces.push({{
          x: scan.profileWorld.x,
          y: scan.profileWorld.y,
          z: scan.profileWorld.z,
          customdata,
          type: 'scatter3d',
          mode: 'lines+markers',
          name: `scan ${{scan.scanId}} profile`,
          opacity: scan.accepted ? 0.62 : 0.9,
          line: {{
            color: scan.accepted ? dataset.color : '#D62728',
            width: scan.accepted ? 4 : 5,
            dash: scan.accepted ? 'solid' : 'dash'
          }},
          marker: {{
            color: scan.accepted ? dataset.color : '#D62728',
            size: 1.7,
            opacity: scan.accepted ? 0.55 : 0.85
          }},
          hovertemplate:
            `scan ${{scan.scanId}} · ${{scan.accepted ? 'accepted' : 'rejected'}}` +
            `<br>tilt ${{scan.pose.tilt.toFixed(2)}}° · azimuth ${{scan.pose.azimuth.toFixed(2)}}°` +
            `<br>d ${{scan.pose.distance.toFixed(2)}} mm · roll ${{scan.pose.roll.toFixed(2)}}°` +
            `<br>sensor point [X,Y,Z] = [%{{customdata[0]:.3f}}, %{{customdata[1]:.3f}}, %{{customdata[2]:.3f}}] mm` +
            `<br>World/Base point [X,Y,Z] = [%{{x:.3f}}, %{{y:.3f}}, %{{z:.3f}}] mm` +
            `<br>source: ${{scan.profileKey || 'not found'}}<extra></extra>`,
          showlegend: false
        }});
      }});

      if (showAxes && selected.length) {{
        const axisLength = dataset.fixedScene.axisLengthMm;
        const axisInfo = [
          {{axis: 0, label: 'Sensor X', color: '#d62728'}},
          {{axis: 1, label: 'Sensor Y', color: '#2ca02c'}},
          {{axis: 2, label: 'Sensor Z', color: '#1f77b4'}}
        ];
        selected.forEach((scan, scanIndex) => {{
          const physical = scan.physicalSensorOriginWorld;
          traces.push({{
            type: 'scatter3d', mode: 'markers',
            x: [physical[0]], y: [physical[1]], z: [physical[2]],
            marker: {{
              size: 5,
              symbol: 'diamond',
              color: scan.accepted ? '#111827' : '#D62728'
            }},
            text: [`physical sensor origin · scan ${{scan.scanId}}`],
            customdata: [[dataset.sensorOriginOffsetZMm]],
            hovertemplate:
              '%{{text}}<br>World/Base [X,Y,Z] = [%{{x:.3f}}, %{{y:.3f}}, %{{z:.3f}}] mm' +
              '<br>signed sensor-Z offset = %{{customdata[0]:.1f}} mm<extra></extra>',
            showlegend: false
          }});
          axisInfo.forEach(info => {{
            const end = vectorEnd(
              physical,
              scan.sensorRotationWorld,
              info.axis,
              axisLength
            );
            traces.push({{
              type: 'scatter3d', mode: 'lines',
              x: [physical[0], end[0]],
              y: [physical[1], end[1]],
              z: [physical[2], end[2]],
              line: {{color: info.color, width: 5}},
              name: info.label,
              legendgroup: info.label,
              showlegend: scanIndex === 0,
              hovertemplate: `scan ${{scan.scanId}} · ${{info.label}}<extra></extra>`
            }});
          }});
        }});
      }}

      const annotationText = !bins.length
        ? 'Check one or more marginal-bin rows to add profiles and sensor frames to the fixed scene.'
        : (!selected.length
            ? 'No scans match the selected bins and status filter.'
            : (!available.length
                ? 'No usable profile-point array was found for the selected bins.'
                : ''));
      const annotations = annotationText ? [{{
        text: annotationText,
        x: 0.5, y: 0.5, xref: 'paper', yref: 'paper', showarrow: false,
        font: {{color: '#667085', size: 14}},
        bgcolor: 'rgba(255,255,255,0.86)', bordercolor: '#dfe3e8', borderpad: 6
      }}] : [];

      try {{
        const fixedRanges = dataset.fixedScene.rangesWorld;
        const fixedRevision = `${{datasetId}}-fixed-full-scene-v1`;
        const plotElement = document.getElementById(`${{datasetId}}-profile-plot`);
        const currentCamera = plotElement?._fullLayout?.scene?.camera
          ? JSON.parse(JSON.stringify(plotElement._fullLayout.scene.camera))
          : {{eye: dataset.fixedScene.cameraEye}};
        const plotResult = Plotly.react(plotElement, traces, {{
          margin: {{l: 0, r: 0, t: 12, b: 0}},
          paper_bgcolor: '#ffffff',
          plot_bgcolor: '#ffffff',
          scene: {{
            xaxis: {{title: {{text: 'World/Base X [mm]'}}, range: fixedRanges[0], autorange: false, gridcolor: '#e8ebef', zerolinecolor: '#c7cdd4'}},
            yaxis: {{title: {{text: 'World/Base Y [mm]'}}, range: fixedRanges[1], autorange: false, gridcolor: '#e8ebef', zerolinecolor: '#c7cdd4'}},
            zaxis: {{title: {{text: 'World/Base Z [mm]'}}, range: fixedRanges[2], autorange: false, gridcolor: '#e8ebef', zerolinecolor: '#c7cdd4'}},
            aspectmode: 'data',
            camera: currentCamera,
            uirevision: fixedRevision
          }},
          legend: {{orientation: 'h', y: 1.02, x: 0}},
          hovermode: 'closest',
          annotations,
          uirevision: fixedRevision
        }}, {{responsive: true, displaylogo: false, scrollZoom: true}});
        if (plotResult && typeof plotResult.catch === 'function') {{
          plotResult.catch(error => {{
            console.error('3-D profile render failed', error);
            if (meta) meta.textContent += ' · 3-D rendering failed: ' + error.message;
          }});
        }}
      }} catch (error) {{
        console.error('3-D profile render failed', error);
        if (meta) meta.textContent += ' · 3-D rendering failed: ' + error.message;
      }}
    }}

    function escapeHtml(value) {{
      return String(value).replace(/[&<>"']/g, character => ({{
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;'
      }})[character]);
    }}

    function formatNumber(value, digits=4) {{
      return Number.isFinite(value) ? Number(value).toFixed(digits) : 'n/a';
    }}

    function formatPercent(value) {{
      if (!Number.isFinite(value)) return 'n/a';
      const sign = value > 0 ? '+' : '';
      return `${{sign}}${{Number(value).toFixed(2)}}%`;
    }}

    function comparisonMetricCard(label, metric) {{
      if (!metric) return '';
      const percent = metric.percent_change;
      let changeClass = '';
      if (Number.isFinite(percent) && metric.lower_is_better) {{
        changeClass = percent <= 0 ? 'change-better' : 'change-worse';
      }}
      const unit = metric.unit ? ` ${{escapeHtml(metric.unit)}}` : '';
      return `
        <div class="calibration-result-card">
          <span>${{escapeHtml(label)}} · full → selected</span>
          <strong>${{formatNumber(metric.full)}} → ${{formatNumber(metric.selected)}}${{unit}}</strong>
          <small class="${{changeClass}}">${{formatPercent(percent)}}</small>
        </div>`;
    }}

    function clearWorldScanViewer(datasetId) {{
      const oldPlot = document.getElementById(`${{datasetId}}-world-scan-plot`);
      if (oldPlot && window.Plotly && oldPlot.classList.contains('js-plotly-plot')) {{
        Plotly.purge(oldPlot);
      }}
      WORLD_SCAN_VIEWER_STATE.delete(datasetId);
    }}

    function sphereWireframeTrace(center, radius, color, name) {{
      const x = [], y = [], z = [];
      const samples = 64;
      [-60, -30, 0, 30, 60].forEach(latitudeDeg => {{
        const latitude = latitudeDeg * Math.PI / 180;
        for (let index = 0; index <= samples; index += 1) {{
          const angle = 2 * Math.PI * index / samples;
          x.push(center[0] + radius * Math.cos(latitude) * Math.cos(angle));
          y.push(center[1] + radius * Math.cos(latitude) * Math.sin(angle));
          z.push(center[2] + radius * Math.sin(latitude));
        }}
        x.push(null); y.push(null); z.push(null);
      }});
      for (let longitudeIndex = 0; longitudeIndex < 10; longitudeIndex += 1) {{
        const longitude = 2 * Math.PI * longitudeIndex / 10;
        for (let index = 0; index <= samples; index += 1) {{
          const angle = -Math.PI / 2 + Math.PI * index / samples;
          x.push(center[0] + radius * Math.cos(angle) * Math.cos(longitude));
          y.push(center[1] + radius * Math.cos(angle) * Math.sin(longitude));
          z.push(center[2] + radius * Math.sin(angle));
        }}
        x.push(null); y.push(null); z.push(null);
      }}
      return {{
        type: 'scatter3d', mode: 'lines', x, y, z, name,
        line: {{color, width: 4}}, hoverinfo: 'name', opacity: 0.92
      }};
    }}

    function renderSphereEvaluation(datasetId, evaluation) {{
      const target = document.getElementById(`${{datasetId}}-sphere-evaluation`);
      if (!target) return;
      if (!evaluation) {{
        target.innerHTML = '';
        return;
      }}
      const free = evaluation.free_radius_fit;
      const fixed = evaluation.fixed_radius_fit;
      const comparison = evaluation.comparison;
      const preprocessing = evaluation.preprocessing;
      const freeMetrics = free.inlier_metrics;
      const fixedMetrics = fixed.inlier_metrics;
      const freeCenter = free.center_base_mm.map(value => formatNumber(value, 6)).join(', ');
      const fixedCenter = fixed.center_base_mm.map(value => formatNumber(value, 6)).join(', ');
      target.innerHTML = `
        <h6>Sphere-fit evaluation · selected hand-eye</h6>
        <p class="sphere-evaluation-summary">
          Automatic floor detection: floor Z ${{formatNumber(evaluation.settings.floor_z_mm, 4)}} mm,
          effective z-min ${{formatNumber(evaluation.settings.z_min_mm, 4)}} mm
          (+${{formatNumber(evaluation.settings.floor_clearance_mm, 3)}} mm clearance).
          Same remaining settings as fit_saved_scan_sphere.py:
          per-profile z-min margin ${{evaluation.settings.z_min_margin_mm}} mm,
          known radius ${{evaluation.settings.known_radius_mm}} mm, point size ${{evaluation.settings.point_size}}.
        </p>
        <div class="sphere-evaluation-grid">
          <div class="sphere-evaluation-card"><span>Detected floor Z</span><strong>${{formatNumber(evaluation.settings.floor_z_mm, 6)}} mm</strong></div>
          <div class="sphere-evaluation-card"><span>Automatic z-min</span><strong>${{formatNumber(evaluation.settings.z_min_mm, 6)}} mm</strong></div>
          <div class="sphere-evaluation-card"><span>Free-fit radius</span><strong>${{formatNumber(free.radius_mm, 6)}} mm</strong></div>
          <div class="sphere-evaluation-card"><span>Radius error vs 5 mm</span><strong>${{formatNumber(comparison.radius_error_mm, 6)}} mm</strong></div>
          <div class="sphere-evaluation-card"><span>Diameter error</span><strong>${{formatNumber(comparison.diameter_error_mm, 6)}} mm</strong></div>
          <div class="sphere-evaluation-card"><span>Free-fit inlier RMS</span><strong>${{formatNumber(freeMetrics.rms_radial_error_mm, 6)}} mm</strong></div>
          <div class="sphere-evaluation-card"><span>Free-fit P95 abs</span><strong>${{formatNumber(freeMetrics.p95_abs_radial_error_mm, 6)}} mm</strong></div>
          <div class="sphere-evaluation-card"><span>Fixed 5 mm inlier RMS</span><strong>${{formatNumber(fixedMetrics.rms_radial_error_mm, 6)}} mm</strong></div>
          <div class="sphere-evaluation-card"><span>Fixed 5 mm P95 abs</span><strong>${{formatNumber(fixedMetrics.p95_abs_radial_error_mm, 6)}} mm</strong></div>
          <div class="sphere-evaluation-card"><span>Fixed-fit inliers</span><strong>${{fixed.inlier_count.toLocaleString()}} / ${{fixed.total_point_count.toLocaleString()}} (${{formatNumber(100 * fixed.inlier_ratio, 1)}}%)</strong></div>
          <div class="sphere-evaluation-card"><span>Center difference</span><strong>${{formatNumber(comparison.center_difference_mm, 6)}} mm</strong></div>
          <div class="sphere-evaluation-card"><span>Crop points</span><strong>${{preprocessing.after_crop_points.toLocaleString()}} / ${{preprocessing.input_points.toLocaleString()}}</strong></div>
          <div class="sphere-evaluation-card"><span>z-min / spike removed</span><strong>${{preprocessing.z_min_removed_points.toLocaleString()}} / ${{preprocessing.profile_spike_removed_points.toLocaleString()}}</strong></div>
          <div class="sphere-evaluation-card"><span>Final sphere points</span><strong>${{preprocessing.final_points.toLocaleString()}}</strong></div>
        </div>
        <details>
          <summary>Centers and complete fit status</summary>
          <pre>free center  [${{escapeHtml(freeCenter)}}] mm
fixed center [${{escapeHtml(fixedCenter)}}] mm
free:  ${{free.converged ? 'converged' : 'not converged'}}, ${{free.iterations}} iterations
fixed: ${{fixed.converged ? 'converged' : 'not converged'}}, ${{fixed.iterations}} iterations
profiles: ${{preprocessing.final_profile_groups}} / ${{preprocessing.input_profile_groups}} retained</pre>
        </details>
      `;
    }}

    async function renderWorldScanPayload(datasetId, payload) {{
      const plot = document.getElementById(`${{datasetId}}-world-scan-plot`);
      const status = document.getElementById(`${{datasetId}}-world-scan-status`);
      if (!plot || !status) return;
      const points = payload.points_world;
      const ranges = payload.ranges_world;
      const existingCamera = plot?._fullLayout?.scene?.camera
        ? JSON.parse(JSON.stringify(plot._fullLayout.scene.camera))
        : {{eye: {{x: 1.45, y: 1.45, z: 1.15}}}};
      const trace = {{
        type: 'scatter3d',
        mode: 'markers',
        x: points.x,
        y: points.y,
        z: points.z,
        name: payload.scan.label,
        marker: {{
          size: payload.sphere_evaluation?.settings?.point_size || 2.1,
          opacity: 0.84,
          color: points.z,
          colorscale: 'Turbo',
          showscale: true,
          colorbar: {{title: {{text: 'World Z [mm]'}}, thickness: 13}}
        }},
        hovertemplate:
          'World X=%{{x:.3f}} mm<br>World Y=%{{y:.3f}} mm' +
          '<br>World Z=%{{z:.3f}} mm<extra></extra>'
      }};
      const traces = [trace];
      if (payload.sphere_evaluation) {{
        const evaluation = payload.sphere_evaluation;
        traces.push(
          sphereWireframeTrace(
            evaluation.fixed_radius_fit.center_base_mm,
            evaluation.settings.known_radius_mm,
            '#2ca02c',
            'Known 5 mm sphere (fixed center)'
          ),
          sphereWireframeTrace(
            evaluation.free_radius_fit.center_base_mm,
            evaluation.free_radius_fit.radius_mm,
            '#ff7f0e',
            'Free-radius fitted sphere'
          )
        );
      }}
      await Plotly.react(plot, traces, {{
        margin: {{l: 0, r: 20, t: 10, b: 0}},
        paper_bgcolor: '#ffffff',
        plot_bgcolor: '#ffffff',
        scene: {{
          xaxis: {{title: {{text: 'World/Base X [mm]'}}, range: ranges[0], autorange: false}},
          yaxis: {{title: {{text: 'World/Base Y [mm]'}}, range: ranges[1], autorange: false}},
          zaxis: {{title: {{text: 'World/Base Z [mm]'}}, range: ranges[2], autorange: false}},
          aspectmode: 'data',
          camera: existingCamera,
          uirevision: `${{datasetId}}-world-scan-camera-v1`
        }},
        hovermode: 'closest',
        legend: {{orientation: 'h', y: 1.02, x: 0}},
        uirevision: `${{datasetId}}-world-scan-camera-v1`
      }}, {{responsive: true, displaylogo: false, scrollZoom: true}});
      renderSphereEvaluation(datasetId, payload.sphere_evaluation || null);
      status.className = 'world-scan-status';
      status.textContent =
        `${{payload.scan.file}} · ${{payload.scan.capture_count}} captures · ` +
        `${{payload.scan.displayed_point_count.toLocaleString()}} / ` +
        `${{payload.scan.point_count.toLocaleString()}} points shown · World/Base frame`;
    }}

    async function loadWorldScan(datasetId, scanId) {{
      const state = WORLD_SCAN_VIEWER_STATE.get(datasetId);
      const status = document.getElementById(`${{datasetId}}-world-scan-status`);
      if (!state || !status) return;
      const requestId = ++state.requestId;
      document.querySelectorAll(
        `[data-world-scan-dataset="${{datasetId}}"]`
      ).forEach(button => {{
        button.classList.toggle('active', button.dataset.worldScanId === scanId);
      }});
      status.className = 'world-scan-status';
      status.textContent = `Loading and reprojecting ${{scanId}} with selected hand-eye…`;
      try {{
        let payload = state.cache.get(scanId);
        if (!payload) {{
          const response = await fetch('/api/world-scan', {{
            method: 'POST',
            headers: {{'Content-Type': 'application/json'}},
            body: JSON.stringify({{
              scan_id: scanId,
              T_tcp_sensor: state.transform
            }})
          }});
          payload = await response.json();
          if (!response.ok || !payload.success) {{
            throw new Error(payload.error || `HTTP ${{response.status}}`);
          }}
          state.cache.set(scanId, payload);
        }}
        if (requestId !== state.requestId) return;
        await renderWorldScanPayload(datasetId, payload);
      }} catch (error) {{
        if (requestId !== state.requestId) return;
        console.error('World scan rendering failed', error);
        status.className = 'world-scan-status error';
        status.textContent = 'World scan failed: ' + error.message;
      }}
    }}

    async function toggleWorldScanFullscreen(datasetId) {{
      const viewer = document.getElementById(`${{datasetId}}-world-scan-viewer`);
      const status = document.getElementById(`${{datasetId}}-world-scan-status`);
      if (!viewer) return;
      try {{
        if (document.fullscreenElement === viewer) {{
          await document.exitFullscreen();
        }} else if (viewer.requestFullscreen) {{
          await viewer.requestFullscreen();
        }} else {{
          throw new Error('Fullscreen API is not supported by this browser.');
        }}
      }} catch (error) {{
        if (status) {{
          status.className = 'world-scan-status error';
          status.textContent = 'Fullscreen failed: ' + error.message;
        }}
      }}
    }}

    function initialiseWorldScanViewer(datasetId, result) {{
      const scans = Array.isArray(result.world_scans) ? result.world_scans : [];
      if (!scans.length) return;
      WORLD_SCAN_VIEWER_STATE.set(datasetId, {{
        transform: result.selected_transform,
        cache: new Map(),
        requestId: 0
      }});
      document.querySelectorAll(
        `[data-world-scan-dataset="${{datasetId}}"]`
      ).forEach(button => {{
        button.addEventListener('click', () =>
          loadWorldScan(datasetId, button.dataset.worldScanId)
        );
      }});
      const fullscreen = document.getElementById(
        `${{datasetId}}-world-scan-fullscreen`
      );
      fullscreen?.addEventListener(
        'click', () => toggleWorldScanFullscreen(datasetId)
      );
      loadWorldScan(datasetId, scans[0].id);
    }}

    function renderCalibrationResult(datasetId, result) {{
      const target = document.getElementById(`${{datasetId}}-calibration-result`);
      const status = document.getElementById(`${{datasetId}}-calibration-status`);
      const difference = result.transform_difference;
      const counts = result.counts;
      const metrics = result.metrics;
      const matrix = result.selected_transform.map(row =>
        row.map(value => Number(value).toFixed(9)).join('  ')
      ).join('\\n');
      const notes = (result.notes || []).map(note =>
        `<li>${{escapeHtml(note)}}</li>`
      ).join('');
      const worldScans = Array.isArray(result.world_scans)
        ? result.world_scans
        : [];
      const worldScanTabs = worldScans.map(scan => `
        <button type="button" class="world-scan-tab"
          data-world-scan-dataset="${{datasetId}}"
          data-world-scan-id="${{escapeHtml(scan.id)}}">
          ${{escapeHtml(scan.label)}}
        </button>
      `).join('');
      clearWorldScanViewer(datasetId);
      target.innerHTML = `
        <div class="calibration-result-grid">
          <div class="calibration-result-card">
            <span>Usable scans · selected / full</span>
            <strong>${{counts.selected_usable}} / ${{counts.full_usable}}</strong>
            <small>${{formatNumber(counts.scan_reduction_percent, 2)}}% reduction</small>
          </div>
          <div class="calibration-result-card">
            <span>Translation difference / full magnitude</span>
            <strong>${{formatNumber(difference.translation_delta_norm_mm)}} mm</strong>
            <small>${{formatPercent(difference.translation_percent_of_full_magnitude)}}</small>
          </div>
          <div class="calibration-result-card">
            <span>Rotation difference / full magnitude</span>
            <strong>${{formatNumber(difference.rotation_delta_deg)}}°</strong>
            <small>${{formatPercent(difference.rotation_percent_of_full_magnitude)}}</small>
          </div>
          ${{comparisonMetricCard('Full-data plane RMS', metrics.full_evaluation_plane_rms)}}
          ${{comparisonMetricCard('Condition number', metrics.condition_number)}}
          ${{comparisonMetricCard('Rotation σ RMS', metrics.rotation_std_rms)}}
          ${{comparisonMetricCard('Translation σ RMS', metrics.translation_std_rms)}}
          <div class="calibration-result-card">
            <span>Selected training plane RMS</span>
            <strong>${{formatNumber(result.solver.selected_training_rms_mm)}} mm</strong>
            <small>${{result.solver.converged ? 'converged' : 'not converged'}} · ${{result.solver.iterations}} iterations</small>
          </div>
        </div>
        <details>
          <summary>Selected T_tcp_sensor</summary>
          <pre class="calibration-matrix">${{escapeHtml(matrix)}}</pre>
        </details>
        ${{notes ? `<ul class="calibration-notes">${{notes}}</ul>` : ''}}
        ${{worldScans.length ? `
          <section class="world-scan-viewer" id="${{datasetId}}-world-scan-viewer">
            <div class="world-scan-heading">
              <div>
                <h5>Selected hand-eye · world-scan viewer</h5>
                <p>Each NPZ is reprojected as T_base_tcp @ selected T_tcp_sensor @ points_sensor.</p>
              </div>
              <button type="button" class="world-scan-fullscreen"
                id="${{datasetId}}-world-scan-fullscreen">Full screen</button>
            </div>
            <div class="world-scan-tabs">${{worldScanTabs}}</div>
            <div class="world-scan-status" id="${{datasetId}}-world-scan-status">
              Select a scan to load its World/Base point cloud.
            </div>
            <div class="world-scan-plot" id="${{datasetId}}-world-scan-plot"></div>
            <div class="sphere-evaluation" id="${{datasetId}}-sphere-evaluation"></div>
          </section>
        ` : ''}}
      `;
      status.className = result.solver.accepted
        ? 'calibration-status success'
        : 'calibration-status warning';
      status.textContent = result.solver.accepted
        ? `Calibration accepted: ${{counts.selected_usable}} selected usable scans compared with ${{counts.full_usable}} full scans.`
        : `Calibration completed but did not pass convergence/RMS acceptance; differences are diagnostic only.`;
      initialiseWorldScanViewer(datasetId, result);
    }}

    async function runSelectedCalibration(datasetId) {{
      const button = document.getElementById(`${{datasetId}}-calibrate-selection`);
      const status = document.getElementById(`${{datasetId}}-calibration-status`);
      const target = document.getElementById(`${{datasetId}}-calibration-result`);
      const scans = currentSelectedScans(datasetId);
      const usable = scans.filter(scan => scan.accepted);
      if (usable.length < 4) return;
      if (window.location.protocol === 'file:') {{
        status.className = 'calibration-status error';
        status.textContent =
          'Calibration needs localhost mode. Run this script with --serve and open the printed http:// URL.';
        return;
      }}

      button.disabled = true;
      status.className = 'calibration-status running';
      status.textContent =
        `Calibrating ${{usable.length}} scans. This may take several seconds…`;
      target.innerHTML = '';
      try {{
        const response = await fetch('/api/calibrate', {{
          method: 'POST',
          headers: {{'Content-Type': 'application/json'}},
          body: JSON.stringify({{
            dataset_id: datasetId,
            scan_ids: scans.map(scan => scan.scanId)
          }})
        }});
        const payload = await response.json();
        if (!response.ok || !payload.success) {{
          throw new Error(payload.error || `HTTP ${{response.status}}`);
        }}
        renderCalibrationResult(datasetId, payload);
      }} catch (error) {{
        console.error('Selected calibration failed', error);
        status.className = 'calibration-status error';
        status.textContent = 'Calibration failed: ' + error.message;
      }} finally {{
        button.disabled = currentSelectedScans(datasetId).filter(
          scan => scan.accepted
        ).length < 4;
      }}
    }}

    function updateRowState(row) {{
      const checkbox = row.querySelector('.bin-select');
      row.classList.toggle('selected', Boolean(checkbox?.checked));
    }}

    function initialiseProfileExplorer() {{
      document.addEventListener('fullscreenchange', () => {{
        document.querySelectorAll('.world-scan-viewer').forEach(viewer => {{
          const button = viewer.querySelector('.world-scan-fullscreen');
          if (button) {{
            button.textContent = document.fullscreenElement === viewer
              ? 'Exit full screen'
              : 'Full screen';
          }}
          const plot = viewer.querySelector('.world-scan-plot');
          if (plot && window.Plotly && plot.classList.contains('js-plotly-plot')) {{
            requestAnimationFrame(() => Plotly.Plots.resize(plot));
          }}
        }});
      }});
      document.addEventListener('click', event => {{
        const row = event.target.closest('.marginal-row');
        if (!row || event.target.matches('input')) return;
        const checkbox = row.querySelector('.bin-select');
        if (!checkbox || checkbox.disabled) return;
        checkbox.checked = !checkbox.checked;
        updateRowState(row);
        renderProfiles(row.dataset.dataset);
      }});

      document.addEventListener('change', event => {{
        if (!event.target.matches('.bin-select')) return;
        const row = event.target.closest('.marginal-row');
        if (!row) return;
        updateRowState(row);
        renderProfiles(row.dataset.dataset);
      }});

      Object.keys(PROFILE_DATA).forEach(datasetId => {{
        const includeRejected = document.getElementById(`${{datasetId}}-include-rejected`);
        const showAxes = document.getElementById(`${{datasetId}}-show-axes`);
        const clear = document.getElementById(`${{datasetId}}-clear-bins`);
        const calibrate = document.getElementById(
          `${{datasetId}}-calibrate-selection`
        );
        includeRejected.checked = !PROFILE_DATA[datasetId].acceptedOnly;
        [includeRejected, showAxes].forEach(control =>
          control.addEventListener('change', () => renderProfiles(datasetId))
        );
        clear.addEventListener('click', () => {{
          document.querySelectorAll(`.marginal-row[data-dataset="${{datasetId}}"] .bin-select`).forEach(box => {{
            box.checked = false;
            const row = box.closest('.marginal-row');
            if (row) updateRowState(row);
          }});
          renderProfiles(datasetId);
        }});
        calibrate.addEventListener(
          'click', () => runSelectedCalibration(datasetId)
        );
        document.querySelectorAll(`.marginal-row[data-dataset="${{datasetId}}"]`).forEach(updateRowState);
        renderProfiles(datasetId);
      }});
      window.PROFILE_EXPLORER_READY = true;
    }}

    if (document.readyState === 'loading') {{
      document.addEventListener('DOMContentLoaded', initialiseProfileExplorer, {{once: true}});
    }} else {{
      initialiseProfileExplorer();
    }}
  </script>
</body>
</html>
"""
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")


def serve_binning_report(
    report_path: Path,
    distributions: list[PoseDistribution],
    *,
    host: str,
    port: int,
    open_browser: bool,
    world_scan_max_points: int,
) -> None:
    """Serve the report and selected-calibration API on localhost."""
    import webbrowser
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from urllib.parse import urlparse

    report_bytes = report_path.read_bytes()
    contexts = prepare_calibration_contexts(distributions)
    project_root = Path(__file__).resolve().parents[2]
    world_scan_specs = {
        spec.scan_id: WorldScanSpec(
            spec.scan_id,
            spec.label,
            spec.path if spec.path.is_absolute() else project_root / spec.path,
        )
        for spec in DEFAULT_WORLD_SCANS
        if (spec.path if spec.path.is_absolute() else project_root / spec.path).is_file()
    }
    world_scan_descriptors = [
        {
            "id": spec.scan_id,
            "label": spec.label,
            "file": spec.path.name,
        }
        for spec in world_scan_specs.values()
    ]

    class Handler(BaseHTTPRequestHandler):
        server_version = "SensorPoseCalibration/1.0"

        def _send_json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(
                payload,
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            path = urlparse(self.path).path
            if path in {"/", "/index.html"}:
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(report_bytes)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(report_bytes)
                return
            if path == "/api/health":
                self._send_json(
                    200,
                    {
                        "success": True,
                        "datasets": sorted(contexts),
                        "world_scans": world_scan_descriptors,
                    },
                )
                return
            if path == "/favicon.ico":
                self.send_response(204)
                self.end_headers()
                return
            self._send_json(404, {"success": False, "error": "not found"})

        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            request_path = urlparse(self.path).path
            if request_path not in {"/api/calibrate", "/api/world-scan"}:
                self._send_json(404, {"success": False, "error": "not found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > 1_000_000:
                    raise ValueError("invalid request size")
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                if request_path == "/api/calibrate":
                    dataset_id = str(payload.get("dataset_id", ""))
                    scan_ids = payload.get("scan_ids")
                    if dataset_id not in contexts:
                        raise ValueError(f"unknown dataset: {dataset_id!r}")
                    if not isinstance(scan_ids, list) or not all(
                        isinstance(value, int) and not isinstance(value, bool)
                        for value in scan_ids
                    ):
                        raise ValueError("scan_ids must be an integer list")
                    result = dict(
                        run_selected_calibration(contexts[dataset_id], scan_ids)
                    )
                    result["world_scans"] = world_scan_descriptors
                else:
                    scan_id = str(payload.get("scan_id", ""))
                    if scan_id not in world_scan_specs:
                        raise ValueError(f"unknown world scan: {scan_id!r}")
                    transform = _validate_transform(
                        np.asarray(payload.get("T_tcp_sensor"), dtype=float),
                        "T_tcp_sensor",
                    )
                    result = reproject_world_scan_for_browser(
                        world_scan_specs[scan_id],
                        transform,
                        max_points=world_scan_max_points,
                    )
            except (ValueError, KeyError, FileNotFoundError) as exc:
                self._send_json(
                    400,
                    {"success": False, "error": str(exc)},
                )
                return
            except Exception as exc:
                self._send_json(
                    500,
                    {
                        "success": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    },
                )
                return
            self._send_json(200, result)

        def log_message(self, fmt: str, *args: Any) -> None:
            print(f"[calibration server] {self.address_string()} - {fmt % args}")

    server = ThreadingHTTPServer((host, port), Handler)
    actual_host, actual_port = server.server_address[:2]
    browser_host = "127.0.0.1" if actual_host in {"0.0.0.0", "::"} else actual_host
    url = f"http://{browser_host}:{actual_port}/"
    print(f"selected-calibration server: {url}")
    print(
        "world-scan viewer files: "
        + (", ".join(world_scan_specs) if world_scan_specs else "none")
    )
    print("Press Ctrl+C to stop.")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        print("\nselected-calibration server stopped")
    finally:
        server.server_close()


def save_binning_csvs(
    distributions: list[PoseDistribution],
    config: BinningConfig,
    assignments_by_label: dict[str, dict[str, np.ndarray]],
    marginal_rows_all: list[dict[str, Any]],
    joint_rows_all: list[dict[str, Any]],
    output_base: Path,
) -> dict[str, Path]:
    pose_path = output_base.with_name(output_base.stem + "_binned_poses.csv")
    marginal_path = output_base.with_name(output_base.stem + "_marginal_bins.csv")
    joint_path = output_base.with_name(output_base.stem + "_joint_bins.csv")
    pose_path.parent.mkdir(parents=True, exist_ok=True)

    labels = {
        "tilt": _bin_labels(config.tilt_edges_deg, "deg"),
        "azimuth": _bin_labels(config.azimuth_edges_deg, "deg"),
        "distance": _bin_labels(config.distance_edges_mm, "mm"),
        "roll": _bin_labels(config.roll_edges_deg, "deg"),
    }
    with pose_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "dataset",
                "capture",
                "scan_id",
                "calibration_accepted",
                "profile_source_key",
                "profile_point_count",
                "plane_u_mm",
                "plane_v_mm",
                "plane_d_mm",
                "tilt_angle_normal_to_sensor_minus_z_deg",
                "azimuth_deg",
                "roll_deg",
                "plane_normal_sensor_x",
                "plane_normal_sensor_y",
                "plane_normal_sensor_z",
                "normal_tilt_from_sensor_minus_z_deg",
                "normal_azimuth_sensor_deg",
                "tilt_bin_index",
                "tilt_bin",
                "azimuth_bin_index",
                "azimuth_bin",
                "distance_bin_index",
                "distance_bin",
                "roll_bin_index",
                "roll_bin",
            ]
        )
        for distribution in distributions:
            assignments = assignments_by_label[distribution.spec.label]
            for index, name in enumerate(distribution.capture_names):
                u, v, d = distribution.origins_plane_mm[index]
                normal = distribution.plane_normals_sensor[index]
                writer.writerow(
                    [
                        distribution.spec.label,
                        name,
                        int(distribution.scan_ids[index]),
                        bool(distribution.accepted[index]),
                        distribution.profile_source_keys[index],
                        int(len(distribution.profiles_sensor_xyz_mm[index])),
                        float(u),
                        float(v),
                        float(d),
                        float(distribution.tilt_deg[index]),
                        float(distribution.azimuth_deg[index]),
                        float(distribution.roll_deg[index]),
                        float(normal[0]),
                        float(normal[1]),
                        float(normal[2]),
                        float(distribution.normal_tilt_deg[index]),
                        float(distribution.normal_azimuth_sensor_deg[index]),
                        int(assignments["tilt"][index]),
                        labels["tilt"][assignments["tilt"][index]],
                        int(assignments["azimuth"][index]),
                        labels["azimuth"][assignments["azimuth"][index]],
                        int(assignments["distance"][index]),
                        labels["distance"][assignments["distance"][index]],
                        int(assignments["roll"][index]),
                        labels["roll"][assignments["roll"][index]],
                    ]
                )

    marginal_fields = [
        "dataset",
        "parameter",
        "bin_index",
        "bin_label",
        "lower",
        "upper",
        "unit",
        "accepted_count",
        "rejected_count",
        "all_count",
        "accepted_fraction",
    ]
    with marginal_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=marginal_fields)
        writer.writeheader()
        writer.writerows(marginal_rows_all)

    joint_fields = [
        "dataset",
        "tilt_bin_index",
        "azimuth_bin_index",
        "distance_bin_index",
        "roll_bin_index",
        "tilt_bin",
        "azimuth_bin",
        "distance_bin",
        "roll_bin",
        "accepted_count",
        "rejected_count",
        "all_count",
        "accepted_scan_ids",
        "rejected_scan_ids",
    ]
    with joint_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=joint_fields)
        writer.writeheader()
        for row in joint_rows_all:
            serialized = dict(row)
            serialized["accepted_scan_ids"] = " ".join(
                map(str, row["accepted_scan_ids"])
            )
            serialized["rejected_scan_ids"] = " ".join(
                map(str, row["rejected_scan_ids"])
            )
            writer.writerow(serialized)

    return {
        "poses": pose_path,
        "marginal": marginal_path,
        "joint": joint_path,
    }

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backup-dataset", type=Path, default=Path("runs/real/backup/dataset"))
    parser.add_argument("--current-dataset", type=Path, default=Path("runs/real/dataset"))
    parser.add_argument(
        "--backup-transform",
        type=Path,
        default=Path(
            "runs/real/backup/real_initial/"
            "T_tcp_sensor_calibrate_initial_value.csv"
        ),
    )
    parser.add_argument(
        "--current-transform",
        type=Path,
        default=Path(
            "runs/real/real_initial/T_tcp_sensor_calibrate_initial_value.csv"
        ),
    )
    parser.add_argument(
        "--backup-diagnostics",
        type=Path,
        default=Path(
            "runs/real/backup/real_initial/"
            "T_tcp_sensor_calibrate_initial_value.diagnostics.json"
        ),
    )
    parser.add_argument(
        "--current-diagnostics",
        type=Path,
        default=Path(
            "runs/real/real_initial/"
            "T_tcp_sensor_calibrate_initial_value.diagnostics.json"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "runs/real/sensor_pose_distribution_comparison/"
            "backup_vs_current_sensor_pose_distribution.html"
        ),
    )
    parser.add_argument(
        "--sensor-origin-offset-z-mm",
        type=float,
        default=-80.0,
        help=(
            "Signed physical sensor-origin displacement along sensor Z from the "
            "measurement-coordinate origin (default: -80 mm, i.e. 80 mm along -Z)"
        ),
    )
    parser.add_argument(
        "--tilt-bin-width-deg",
        type=float,
        default=10.0,
        help="Bin width for tilt=angle(n_sensor, -Z_sensor) when explicit edges are omitted",
    )
    parser.add_argument(
        "--tilt-bin-edges-deg",
        type=str,
        default=None,
        help="Comma-separated tilt edges, e.g. 0,10,20,30,45,60,90,180",
    )
    parser.add_argument(
        "--azimuth-bin-width-deg",
        type=float,
        default=30.0,
        help="Azimuth bin width over [-180, 180)",
    )
    parser.add_argument(
        "--distance-bin-width-mm",
        type=float,
        default=25.0,
        help="Automatic common distance-bin width",
    )
    parser.add_argument(
        "--distance-bin-edges-mm",
        type=str,
        default=None,
        help="Comma-separated signed-distance edges in mm",
    )
    parser.add_argument(
        "--roll-bin-width-deg",
        type=float,
        default=30.0,
        help="Sensor-roll bin width over [-180, 180)",
    )
    parser.add_argument(
        "--bin-include-rejected",
        action="store_true",
        help="Use rejected captures as well as accepted captures in bin graphs",
    )
    parser.add_argument(
        "--profile-max-points",
        type=int,
        default=500,
        help=(
            "Maximum number of 3-D sensor-profile points embedded per capture in "
            "the interactive HTML explorer (default: 500; <=0 keeps all points)"
        ),
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help=(
            "Serve the binning report on localhost and enable calibration of "
            "the scans selected in the browser"
        ),
    )
    parser.add_argument(
        "--serve-host",
        type=str,
        default="127.0.0.1",
        help="Host for --serve (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--serve-port",
        type=int,
        default=8765,
        help="Port for --serve; use 0 to choose an available port (default: 8765)",
    )
    parser.add_argument(
        "--world-scan-max-points",
        type=int,
        default=WORLD_SCAN_MAX_POINTS,
        help=(
            "Maximum points returned for each selected-hand-eye world-scan "
            f"viewer request (default: {WORLD_SCAN_MAX_POINTS})"
        ),
    )
    parser.add_argument(
        "--no-open-browser",
        action="store_true",
        help="Do not open the localhost report automatically with --serve",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not np.isfinite(args.sensor_origin_offset_z_mm):
        raise ValueError("--sensor-origin-offset-z-mm must be finite")
    if args.profile_max_points == 0:
        raise ValueError("--profile-max-points must be positive or negative to keep all")
    if not (0 <= args.serve_port <= 65535):
        raise ValueError("--serve-port must be in [0, 65535]")
    if args.world_scan_max_points <= 0:
        raise ValueError("--world-scan-max-points must be positive")

    specs = [
        DatasetSpec(
            "Backup",
            args.backup_dataset,
            args.backup_transform,
            args.backup_diagnostics,
            BACKUP_COLOR,
            args.sensor_origin_offset_z_mm,
        ),
        DatasetSpec(
            "Current",
            args.current_dataset,
            args.current_transform,
            args.current_diagnostics,
            CURRENT_COLOR,
            args.sensor_origin_offset_z_mm,
        ),
    ]
    distributions = [load_pose_distribution(spec) for spec in specs]
    summaries = [summarize(item) for item in distributions]

    save_html(distributions, summaries, args.output)
    summary_path = args.output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps({"datasets": summaries}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    pose_csv_path = args.output.with_suffix(".poses.csv")
    save_pose_csv(distributions, pose_csv_path)

    config = build_binning_config(args, distributions)
    assignments_by_label = {
        distribution.spec.label: assign_pose_bins(distribution, config)
        for distribution in distributions
    }
    marginal_rows_all: list[dict[str, Any]] = []
    joint_rows_all: list[dict[str, Any]] = []
    validations: list[dict[str, Any]] = []
    for distribution in distributions:
        assignments = assignments_by_label[distribution.spec.label]
        marginal_rows_all.extend(
            marginal_bin_rows(distribution, config, assignments)
        )
        joint_rows_all.extend(
            joint_bin_rows(distribution, config, assignments)
        )
        validations.append(
            validate_pose_distribution(distribution, config, assignments)
        )

    binning_html_path = args.output.with_name(
        args.output.stem + "_binning.html"
    )
    save_binning_html(
        distributions,
        config,
        assignments_by_label,
        marginal_rows_all,
        joint_rows_all,
        validations,
        binning_html_path,
        profile_max_points=args.profile_max_points,
    )
    csv_paths = save_binning_csvs(
        distributions,
        config,
        assignments_by_label,
        marginal_rows_all,
        joint_rows_all,
        args.output,
    )
    validation_path = args.output.with_name(
        args.output.stem + "_validation.json"
    )
    validation_path.write_text(
        json.dumps(
            {
                "binning": {
                    "accepted_only": config.accepted_only,
                    "tilt_edges_deg": config.tilt_edges_deg.tolist(),
                    "azimuth_edges_deg": config.azimuth_edges_deg.tolist(),
                    "distance_edges_mm": config.distance_edges_mm.tolist(),
                    "roll_edges_deg": config.roll_edges_deg.tolist(),
                },
                "datasets": validations,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"saved HTML      : {args.output}")
    print(f"saved summary   : {summary_path}")
    print(f"saved poses     : {pose_csv_path}")
    print(f"saved binning   : {binning_html_path}")
    print(f"saved validation: {validation_path}")
    for label, path in csv_paths.items():
        print(f"saved {label:8s}: {path}")
    for summary in summaries:
        print(
            f"{summary['label']}: N={summary['capture_count']}, "
            f"accepted={summary['accepted_pose_count']}, "
            f"U/V/d range="
            f"{summary['plane_u_range_mm']:.2f}/"
            f"{summary['plane_v_range_mm']:.2f}/"
            f"{summary['plane_d_range_mm']:.2f} mm, "
            f"tilt={summary['tilt_mean_deg']:.2f}±"
            f"{summary['tilt_std_deg']:.2f} deg, "
            f"azimuth span={summary['azimuth_circular_span_deg']:.2f} deg, "
            f"roll span={summary['roll_circular_span_deg']:.2f} deg"
        )
    if args.serve:
        serve_binning_report(
            binning_html_path,
            distributions,
            host=args.serve_host,
            port=args.serve_port,
            open_browser=not args.no_open_browser,
            world_scan_max_points=args.world_scan_max_points,
        )


if __name__ == "__main__":
    main()
