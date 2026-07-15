from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from robust_laser_handeye.laser_handeye.patterns import scan_parameter_grid
from robust_laser_handeye.laser_handeye.se3 import inv_T
from robust_laser_handeye.laser_handeye.simulation import sensor_pose_from_target_line

from .make_sensor_pose import directional_distance_to_boundary, read_bounds
from .hardware import xyz_rpy_deg_from_transform
from .safety import SafetyConfig, validate_segment, validate_transform


PLAN_SCHEMA_VERSION = 1


def safety_snapshot(safety: SafetyConfig) -> dict[str, Any]:
    return {
        "workspace_mm": {
            "min": safety.workspace.minimum_mm.tolist(),
            "max": safety.workspace.maximum_mm.tolist(),
        },
        "no_go_boxes_mm": [
            {
                "name": box.name,
                "min": box.minimum_mm.tolist(),
                "max": box.maximum_mm.tolist(),
            }
            for box in safety.no_go_boxes
        ],
        "safe_transit_T_base_tcp": (
            None
            if safety.safe_transit_T_base_tcp is None
            else safety.safe_transit_T_base_tcp.tolist()
        ),
        "approach_clearance_mm": float(safety.approach_clearance_mm),
        "max_linear_step_mm": float(safety.max_linear_step_mm),
        "max_angular_step_deg": float(safety.max_angular_step_deg),
        "linear_speed_mm_s": float(safety.linear_speed_mm_s),
        "angular_speed_deg_s": float(safety.angular_speed_deg_s),
        "motion_timeout_s": float(safety.motion_timeout_s),
        "settle_time_s": float(safety.settle_time_s),
        "readback_position_tolerance_mm": float(
            safety.readback_position_tolerance_mm
        ),
        "readback_rotation_tolerance_deg": float(
            safety.readback_rotation_tolerance_deg
        ),
        "stationarity_position_tolerance_mm": float(
            safety.stationarity_position_tolerance_mm
        ),
        "stationarity_rotation_tolerance_deg": float(
            safety.stationarity_rotation_tolerance_deg
        ),
        "min_sensor_plane_clearance_mm": float(
            safety.min_sensor_plane_clearance_mm
        ),
        "initial_handeye_uncertainty_mm": float(
            safety.initial_handeye_uncertainty_mm
        ),
        "initial_handeye_uncertainty_deg": float(
            safety.initial_handeye_uncertainty_deg
        ),
        "require_controller_collision_check": bool(
            safety.require_controller_collision_check
        ),
    }


def plan_identity(plan: dict[str, Any]) -> str:
    canonical = {key: value for key, value in plan.items() if key != "plan_id"}
    encoded = json.dumps(
        canonical, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_plan_identity(plan: dict[str, Any]) -> str:
    stored = plan.get("plan_id")
    computed = plan_identity(plan)
    if stored is None:
        raise ValueError("motion plan has no plan_id; regenerate it before execution")
    if str(stored) != computed:
        raise ValueError("motion plan content does not match its plan_id")
    return computed


def validate_plan_runtime_safety(
    plan: dict[str, Any], safety: SafetyConfig
) -> None:
    planned = plan.get("safety_snapshot")
    current = safety_snapshot(safety)
    if planned != current:
        raise RuntimeError(
            "runtime safety config differs from the reviewed motion plan; "
            "regenerate and review the plan before live execution"
        )


def load_transform(path: str | Path) -> np.ndarray:
    path = Path(path)
    for delimiter in (",", None):
        try:
            T = np.loadtxt(path, delimiter=delimiter)
        except ValueError:
            continue
        if T.shape == (4, 4):
            return validate_transform(T, str(path))
    raise ValueError(f"{path}: expected a 4x4 transform")


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def save_json(path: str | Path, value: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
    temporary.replace(path)


def save_plan_csv(path: str | Path, plan: dict[str, Any]) -> None:
    """Export target/approach poses for human review and controller simulation."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "scan_id",
        "pose_type",
        "line_id",
        "reference_pose",
        "d_mm",
        "theta_deg",
        "beta_deg",
        "x_mm",
        "y_mm",
        "z_mm",
        "rx_xyz_deg",
        "ry_xyz_deg",
        "rz_xyz_deg",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for entry in plan["entries"]:
            for pose_type, key in (
                ("approach", "T_base_tcp_approach"),
                ("target", "T_base_tcp"),
            ):
                values = xyz_rpy_deg_from_transform(np.asarray(entry[key], dtype=float))
                writer.writerow(
                    {
                        "scan_id": entry["scan_id"],
                        "pose_type": pose_type,
                        "line_id": entry["line_id"],
                        "reference_pose": entry["reference_pose"],
                        "d_mm": entry["d_mm"],
                        "theta_deg": entry["theta_deg"],
                        "beta_deg": entry["beta_deg"],
                        "x_mm": values[0],
                        "y_mm": values[1],
                        "z_mm": values[2],
                        "rx_xyz_deg": values[3],
                        "ry_xyz_deg": values[4],
                        "rz_xyz_deg": values[5],
                    }
                )


def _parameter_grid(
    heights_mm: Iterable[float], theta_deg: Iterable[float], beta_deg: Iterable[float]
) -> list[dict[str, float]]:
    return scan_parameter_grid(
        heights_mm=tuple(float(value) for value in heights_mm),
        projection_deg=tuple(float(value) for value in theta_deg),
        tilt_deg=tuple(float(value) for value in beta_deg),
    )


def _circular_rays(
    bounds: dict[str, float],
    *,
    center_uv: np.ndarray | None,
    radius_mm: float | None,
    radius_scale: float,
    line_count: int = 9,
    start_angle_deg: float = 0.0,
) -> tuple[np.ndarray, float, list[tuple[np.ndarray, np.ndarray]]]:
    if center_uv is None:
        center = np.array(
            [
                0.5 * (bounds["u_min"] + bounds["u_max"]),
                0.5 * (bounds["v_min"] + bounds["v_max"]),
            ]
        )
    else:
        center = np.asarray(center_uv, dtype=float).reshape(2)
    if not (
        bounds["u_min"] <= center[0] <= bounds["u_max"]
        and bounds["v_min"] <= center[1] <= bounds["v_max"]
    ):
        raise ValueError("circular-pattern center is outside the safe plane bounds")
    if not 0.0 < float(radius_scale) <= 1.0:
        raise ValueError("radius_scale must be in (0, 1]")

    directions = []
    distances = []
    for line_id in range(int(line_count)):
        angle = np.radians(float(start_angle_deg) + 360.0 * line_id / line_count)
        direction = np.array([np.cos(angle), np.sin(angle)])
        directions.append(direction)
        distances.append(directional_distance_to_boundary(center, direction, bounds))
    maximum = float(min(distances))
    radius = float(radius_scale) * maximum if radius_mm is None else float(radius_mm)
    if radius <= 0.0 or radius > maximum + 1e-9:
        raise ValueError(f"pattern radius {radius:.3f} mm exceeds safe maximum {maximum:.3f} mm")
    return center, radius, [(center.copy(), center + radius * d) for d in directions]


def translation_offset_observability(entries: list[dict[str, Any]], normal: np.ndarray) -> dict[str, Any]:
    normal = np.asarray(normal, dtype=float).reshape(3)
    normal /= np.linalg.norm(normal)
    rows = []
    for entry in entries:
        T = np.asarray(entry["T_base_tcp"], dtype=float).reshape(4, 4)
        rows.append(np.concatenate([normal @ T[:3, :3], [-1.0]]))
    A = np.asarray(rows, dtype=float)
    scale = np.linalg.norm(A, axis=0)
    if len(A) == 0 or np.any(scale <= 0.0):
        rank, condition, singular_values = 0, float("inf"), []
    else:
        singular = np.linalg.svd(A / scale, compute_uv=False)
        rank = int(np.sum(singular > 1e-10 * singular[0]))
        condition = float(singular[0] / singular[-1]) if singular[-1] > 0 else float("inf")
        singular_values = singular.tolist()
    return {
        "rank": rank,
        "required_rank": 4,
        "column_normalized_condition": condition,
        "singular_values": singular_values,
        "observable": bool(rank == 4 and np.isfinite(condition)),
    }


def _make_entry(
    *,
    scan_id: int,
    line_id: int,
    parameter_id: int,
    parameter: dict[str, float],
    line: tuple[np.ndarray, np.ndarray],
    T_base_plane: np.ndarray,
    T_tcp_sensor_init: np.ndarray,
    branch_sign: float,
    pose_geometry: str,
    reference_pose: bool,
    approach_clearance_mm: float,
) -> dict[str, Any]:
    p0, p1 = line
    T_base_sensor = sensor_pose_from_target_line(
        plane_R=T_base_plane[:3, :3],
        plane_t=T_base_plane[:3, 3],
        line_p0=p0,
        line_p1=p1,
        d_mm=parameter["d_mm"],
        theta_deg=parameter["theta_deg"],
        beta_deg=parameter["beta_deg"],
        branch_sign=branch_sign,
        pose_geometry=pose_geometry,
    )
    T_base_tcp = T_base_sensor @ inv_T(T_tcp_sensor_init)
    T_base_sensor_approach = T_base_sensor.copy()
    T_base_sensor_approach[:3, 3] += (
        float(approach_clearance_mm) * T_base_plane[:3, 2]
    )
    T_base_tcp_approach = T_base_sensor_approach @ inv_T(T_tcp_sensor_init)
    return {
        "scan_id": int(scan_id),
        "line_id": int(line_id),
        "parameter_id": int(parameter_id),
        "reference_pose": bool(reference_pose),
        "branch_sign": float(branch_sign),
        "pose_geometry": pose_geometry,
        **{key: float(value) for key, value in parameter.items()},
        "line_p0_uv_mm": p0.tolist(),
        "line_p1_uv_mm": p1.tolist(),
        "T_base_sensor_nominal": T_base_sensor.tolist(),
        "T_base_tcp": T_base_tcp.tolist(),
        "T_base_tcp_approach": T_base_tcp_approach.tolist(),
    }


def build_single_plane_plan(
    *,
    plane_boundary: dict[str, Any],
    T_tcp_sensor_init: np.ndarray,
    safety: SafetyConfig,
    heights_mm: Iterable[float] = (60.0, 90.0, 120.0),
    theta_deg: Iterable[float] = (30.0,),
    beta_deg: Iterable[float] = (60.0, 90.0, 120.0),
    reference_scan_count: int = 24,
    reference_theta_deg: Iterable[float] = (60.0,),
    reference_heights_mm: Iterable[float] = (60.0, 90.0, 120.0),
    reference_beta_deg: Iterable[float] = (60.0, 90.0, 120.0),
    pattern_radius_mm: float | None = None,
    pattern_radius_scale: float = 0.8,
    pattern_center_uv_mm: np.ndarray | None = None,
    pose_geometry: str = "paper_incidence",
    allow_unobservable: bool = False,
) -> dict[str, Any]:
    T_tcp_sensor_init = validate_transform(T_tcp_sensor_init, "T_tcp_sensor_init")
    T_base_plane = validate_transform(
        np.asarray(plane_boundary["plane_frame"]["T_base_plane"], dtype=float),
        "T_base_plane",
    )
    bounds = read_bounds(plane_boundary)
    center, radius, lines = _circular_rays(
        bounds,
        center_uv=pattern_center_uv_mm,
        radius_mm=pattern_radius_mm,
        radius_scale=pattern_radius_scale,
    )
    main_parameters = _parameter_grid(heights_mm, theta_deg, beta_deg)
    reference_parameters = _parameter_grid(
        reference_heights_mm, reference_theta_deg, reference_beta_deg
    )
    if int(reference_scan_count) < 0:
        raise ValueError("reference_scan_count must be non-negative")

    entries: list[dict[str, Any]] = []
    for line_id, line in enumerate(lines):
        branch = 1.0 if line_id % 2 == 0 else -1.0
        for parameter_id, parameter in enumerate(main_parameters):
            entries.append(
                _make_entry(
                    scan_id=len(entries),
                    line_id=line_id,
                    parameter_id=parameter_id,
                    parameter=parameter,
                    line=line,
                    T_base_plane=T_base_plane,
                    T_tcp_sensor_init=T_tcp_sensor_init,
                    branch_sign=branch,
                    pose_geometry=pose_geometry,
                    reference_pose=False,
                    approach_clearance_mm=safety.approach_clearance_mm,
                )
            )
    for reference_id in range(int(reference_scan_count)):
        line_id = reference_id % len(lines)
        cycle = reference_id // len(lines)
        parameter_id = (4 * line_id + cycle) % len(reference_parameters)
        entries.append(
            _make_entry(
                scan_id=len(entries),
                line_id=line_id,
                parameter_id=parameter_id,
                parameter=reference_parameters[parameter_id],
                line=lines[line_id],
                T_base_plane=T_base_plane,
                T_tcp_sensor_init=T_tcp_sensor_init,
                branch_sign=1.0 if reference_id % 2 == 0 else -1.0,
                pose_geometry=pose_geometry,
                reference_pose=True,
                approach_clearance_mm=safety.approach_clearance_mm,
            )
        )

    plane_normal = T_base_plane[:3, 2]
    plane_origin = T_base_plane[:3, 3]
    mount_radius = float(np.linalg.norm(T_tcp_sensor_init[:3, 3]))
    uncertainty = float(safety.initial_handeye_uncertainty_mm) + 2.0 * mount_radius * np.sin(
        0.5 * np.radians(float(safety.initial_handeye_uncertainty_deg))
    )
    for entry in entries:
        T_sensor = np.asarray(entry["T_base_sensor_nominal"], dtype=float)
        signed_clearance = float(plane_normal @ (T_sensor[:3, 3] - plane_origin))
        entry["nominal_sensor_plane_clearance_mm"] = signed_clearance
        entry["conservative_sensor_plane_clearance_mm"] = signed_clearance - uncertainty
        if signed_clearance - uncertainty < safety.min_sensor_plane_clearance_mm:
            raise ValueError(
                f"scan {entry['scan_id']} conservative sensor-plane clearance "
                f"{signed_clearance - uncertainty:.3f} mm is below "
                f"{safety.min_sensor_plane_clearance_mm:.3f} mm"
            )
        target = np.asarray(entry["T_base_tcp"], dtype=float)
        approach = np.asarray(entry["T_base_tcp_approach"], dtype=float)
        safety.assert_pose_safe(target, name=f"scan {entry['scan_id']} target")
        safety.assert_pose_safe(approach, name=f"scan {entry['scan_id']} approach")
        validate_segment(approach, target, safety, name=f"scan {entry['scan_id']} approach")
        if safety.safe_transit_T_base_tcp is not None:
            validate_segment(
                safety.safe_transit_T_base_tcp,
                approach,
                safety,
                name=f"scan {entry['scan_id']} transit",
            )

    observability = translation_offset_observability(entries, plane_normal)
    if not observability["observable"] and not allow_unobservable:
        raise ValueError(
            "planned poses cannot separate hand-eye translation from the unknown "
            "plane offset (rank < 4). Add a second theta, e.g. 24 theta=60 "
            "reference scans, or pass allow_unobservable only for diagnosis."
        )
    boundary_quality = plane_boundary.get("quality_gate")
    boundary_provenance = plane_boundary.get("bootstrap_provenance")
    if isinstance(boundary_quality, dict):
        observed_bounds = plane_boundary["observed_bounds_uv_mm"]
        bootstrap_quality = {
            "accepted": boundary_quality.get("accepted") is True,
            "plane_rms_mm": float(plane_boundary["plane"]["rms_error_mm"]),
            "observed_u_span_mm": (
                float(observed_bounds["u_max"]) - float(observed_bounds["u_min"])
            ),
            "observed_v_span_mm": (
                float(observed_bounds["v_max"]) - float(observed_bounds["v_min"])
            ),
            "sensor_plane_signed_distances_mm": [
                float(value)
                for value in boundary_quality.get(
                    "sensor_plane_signed_distances_mm", []
                )
            ],
        }
    else:
        bootstrap_quality = None
    if isinstance(boundary_provenance, dict):
        bootstrap_provenance = {
            "capture_count": int(boundary_provenance.get("capture_count", 0)),
            "T_tcp_sensor_init": np.asarray(
                boundary_provenance.get("T_tcp_sensor_init"), dtype=float
            ).reshape(4, 4).tolist(),
            "transform_convention": str(
                boundary_provenance.get(
                    "transform_convention",
                    "T_tcp_sensor maps sensor coordinates into TCP",
                )
            ),
        }
    else:
        bootstrap_provenance = None
    plan = {
        "schema_version": PLAN_SCHEMA_VERSION,
        "units": {"translation": "mm", "angle": "deg"},
        "transform_convention": "T_A_B maps frame-B coordinates into frame A",
        "tcp_role": "robot TCP is the calibration end-effector frame",
        "pose_geometry": pose_geometry,
        "T_tcp_sensor_init": T_tcp_sensor_init.tolist(),
        "T_base_plane": T_base_plane.tolist(),
        "plane_bounds_uv_mm": bounds,
        "pattern_center_uv_mm": center.tolist(),
        "pattern_radius_mm": radius,
        "main_scan_count": len(lines) * len(main_parameters),
        "reference_scan_count": int(reference_scan_count),
        "observability": observability,
        "initial_handeye_position_uncertainty_bound_mm": uncertainty,
        "bootstrap_quality": bootstrap_quality,
        "bootstrap_provenance": bootstrap_provenance,
        "safety_snapshot": safety_snapshot(safety),
        "entries": entries,
    }
    plan["plan_id"] = plan_identity(plan)
    return plan
