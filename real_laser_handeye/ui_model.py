from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from robust_laser_handeye.laser_handeye.geometry import fit_plane_pca
from robust_laser_handeye.laser_handeye.se3 import transform_points

from .hardware import ProfileSample, xyz_rpy_deg_from_transform
from .initial_point import load_profile
from .planning import load_json, load_transform, plan_identity


def _readonly(array: np.ndarray) -> np.ndarray:
    result = np.asarray(array, dtype=float).copy()
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class PlaneEstimate:
    available: bool
    status: str
    normal_base: np.ndarray
    center_base_mm: np.ndarray
    offset_mm: float
    rms_mm: float
    point_count: int
    distinct_line_count: int


@dataclass(frozen=True)
class DashboardSnapshot:
    stage: str
    plan_id: str
    total_count: int
    completed_ids: tuple[int, ...]
    next_scan_id: int | None
    next_line_id: int | None
    next_reference_pose: bool
    next_d_mm: float | None
    next_theta_deg: float | None
    next_beta_deg: float | None
    next_T_base_tcp: np.ndarray
    next_T_base_tcp_approach: np.ndarray
    next_T_base_sensor_nominal: np.ndarray
    next_tcp_xyzrpy_deg: np.ndarray
    live_sequence: int
    live_profile_s: np.ndarray
    live_error: str | None
    previous_scan_id: int | None
    previous_profile_s: np.ndarray
    accumulated_points_base: np.ndarray
    plane_estimate: PlaneEstimate
    bootstrap_plane_corners_base: np.ndarray
    planned_tcp_positions_base: np.ndarray
    planned_approach_positions_base: np.ndarray
    planned_sensor_positions_base: np.ndarray
    completed_mask: np.ndarray
    workspace_bounds_base: np.ndarray
    no_go_bounds_base: tuple[np.ndarray, ...]
    safe_transit_T_base_tcp: np.ndarray
    observability_rank: int
    observability_condition: float
    handeye_source: str
    data_errors: tuple[str, ...]

    @property
    def completed_count(self) -> int:
        return len(self.completed_ids)


def _empty_transform() -> np.ndarray:
    return _readonly(np.full((4, 4), np.nan))


def _boundary_corners(plan: dict[str, Any]) -> np.ndarray:
    bounds = plan["plane_bounds_uv_mm"]
    uv = np.array(
        [
            [bounds["u_min"], bounds["v_min"]],
            [bounds["u_max"], bounds["v_min"]],
            [bounds["u_max"], bounds["v_max"]],
            [bounds["u_min"], bounds["v_max"]],
            [bounds["u_min"], bounds["v_min"]],
        ],
        dtype=float,
    )
    T = np.asarray(plan["T_base_plane"], dtype=float)
    local = np.column_stack([uv, np.zeros(len(uv))])
    return transform_points(T, local)


def _downsample(points: np.ndarray, maximum: int) -> np.ndarray:
    if len(points) <= maximum:
        return points
    indices = np.linspace(0, len(points) - 1, int(maximum), dtype=int)
    return points[indices]


def build_dashboard_snapshot(
    *,
    plan: dict[str, Any],
    dataset_dir: str | Path,
    live_sequence: int = 0,
    live_sample: ProfileSample | None = None,
    live_error: str | None = None,
    T_tcp_sensor: np.ndarray | None = None,
    handeye_source: str = "initial",
    stage: str = "PLAN_READY",
    max_accumulated_points: int = 12000,
) -> DashboardSnapshot:
    dataset_dir = Path(dataset_dir)
    errors: list[str] = []
    stored_id = str(plan.get("plan_id", ""))
    computed_id = plan_identity(plan)
    if not stored_id or stored_id != computed_id:
        errors.append("motion plan identity is missing or invalid")
    entries = list(plan["entries"])
    entry_by_id = {int(entry["scan_id"]): entry for entry in entries}
    manifest_path = dataset_dir / "manifest.json"
    records: list[dict[str, Any]] = []
    if manifest_path.exists():
        try:
            manifest = load_json(manifest_path)
            if manifest.get("plan_id") != stored_id:
                errors.append("dataset manifest belongs to another motion plan")
                records = []
            else:
                expected = [int(entry["scan_id"]) for entry in entries]
                if manifest.get("expected_scan_ids") != expected:
                    errors.append("dataset expected scan IDs differ from this motion plan")
                    records = []
                else:
                    records = sorted(
                        list(manifest.get("scans", [])),
                        key=lambda item: int(item["scan_id"]),
                    )
        except Exception as exc:
            errors.append(f"manifest: {type(exc).__name__}: {exc}")

    completed_ids: list[int] = []
    point_sets: list[np.ndarray] = []
    captured_line_ids: set[int] = set()
    previous_scan_id: int | None = None
    previous_profile = np.empty((0, 3), dtype=float)
    handeye = np.asarray(
        plan["T_tcp_sensor_init"] if T_tcp_sensor is None else T_tcp_sensor,
        dtype=float,
    ).reshape(4, 4)
    for record in records:
        scan_id = int(record["scan_id"])
        if scan_id not in entry_by_id:
            errors.append(f"scan {scan_id}: not present in motion plan")
            continue
        try:
            pose = load_transform(dataset_dir / record["pose_file"])
            points_s = load_profile(dataset_dir / record["profile_file"])
            points_base = transform_points(pose @ handeye, points_s)
        except Exception as exc:
            errors.append(f"scan {scan_id}: {type(exc).__name__}: {exc}")
            continue
        completed_ids.append(scan_id)
        captured_line_ids.add(int(entry_by_id[scan_id]["line_id"]))
        point_sets.append(points_base)
        previous_scan_id = scan_id
        previous_profile = points_s

    accumulated_full = (
        np.vstack(point_sets) if point_sets else np.empty((0, 3), dtype=float)
    )
    accumulated = _downsample(accumulated_full, int(max_accumulated_points))
    bootstrap_T = np.asarray(plan["T_base_plane"], dtype=float)
    if len(accumulated_full) < 3 or len(captured_line_ids) < 2:
        estimate = PlaneEstimate(
            available=False,
            status="서로 다른 circular line의 캡처가 2개 이상 필요합니다",
            normal_base=_readonly(np.full(3, np.nan)),
            center_base_mm=_readonly(np.full(3, np.nan)),
            offset_mm=float("nan"),
            rms_mm=float("nan"),
            point_count=len(accumulated_full),
            distinct_line_count=len(captured_line_ids),
        )
    else:
        centered = accumulated_full - np.mean(accumulated_full, axis=0)
        singular = np.linalg.svd(centered, compute_uv=False)
        if len(singular) < 2 or singular[1] <= 1e-7 * singular[0]:
            estimate = PlaneEstimate(
                available=False,
                status="누적 점의 2D span이 부족합니다",
                normal_base=_readonly(np.full(3, np.nan)),
                center_base_mm=_readonly(np.full(3, np.nan)),
                offset_mm=float("nan"),
                rms_mm=float("nan"),
                point_count=len(accumulated_full),
                distinct_line_count=len(captured_line_ids),
            )
        else:
            normal, offset, center, rms = fit_plane_pca(accumulated_full)
            if float(normal @ bootstrap_T[:3, 2]) < 0.0:
                normal, offset = -normal, -offset
            estimate = PlaneEstimate(
                available=True,
                status=f"{handeye_source} hand-eye 기반 provisional plane",
                normal_base=_readonly(normal),
                center_base_mm=_readonly(center),
                offset_mm=float(offset),
                rms_mm=float(rms),
                point_count=len(accumulated_full),
                distinct_line_count=len(captured_line_ids),
            )

    completed_set = set(completed_ids)
    next_entry = next(
        (entry for entry in entries if int(entry["scan_id"]) not in completed_set),
        None,
    )
    if next_entry is None:
        next_id = next_line = None
        next_ref = False
        next_d = next_theta = next_beta = None
        next_target = next_approach = next_sensor = _empty_transform()
        next_xyzrpy = _readonly(np.full(6, np.nan))
    else:
        next_id = int(next_entry["scan_id"])
        next_line = int(next_entry["line_id"])
        next_ref = bool(next_entry["reference_pose"])
        next_d = float(next_entry["d_mm"])
        next_theta = float(next_entry["theta_deg"])
        next_beta = float(next_entry["beta_deg"])
        next_target_raw = np.asarray(next_entry["T_base_tcp"], dtype=float).copy()
        next_approach_raw = np.asarray(
            next_entry["T_base_tcp_approach"], dtype=float
        ).copy()
        next_sensor_raw = np.asarray(
            next_entry["T_base_sensor_nominal"], dtype=float
        ).copy()
        next_xyzrpy = _readonly(xyz_rpy_deg_from_transform(next_target_raw))
        next_target = _readonly(next_target_raw)
        next_approach = _readonly(next_approach_raw)
        next_sensor = _readonly(next_sensor_raw)

    planned_tcp = np.array(
        [np.asarray(entry["T_base_tcp"], dtype=float)[:3, 3] for entry in entries]
    )
    planned_approach = np.array(
        [
            np.asarray(entry["T_base_tcp_approach"], dtype=float)[:3, 3]
            for entry in entries
        ]
    )
    planned_sensor = np.array(
        [
            np.asarray(entry["T_base_sensor_nominal"], dtype=float)[:3, 3]
            for entry in entries
        ]
    )
    completed_mask = np.array(
        [int(entry["scan_id"]) in completed_set for entry in entries], dtype=bool
    )
    completed_mask.setflags(write=False)
    live_points = (
        np.empty((0, 3), dtype=float)
        if live_sample is None
        else np.asarray(live_sample.points_s, dtype=float)
    )
    observability = plan.get("observability", {})
    safety_snapshot = dict(plan.get("safety_snapshot", {}))
    workspace = safety_snapshot.get("workspace_mm")
    if workspace is None:
        workspace_bounds = np.full((2, 3), np.nan)
    else:
        workspace_bounds = np.vstack(
            [
                np.asarray(workspace["min"], dtype=float).reshape(3),
                np.asarray(workspace["max"], dtype=float).reshape(3),
            ]
        )
    no_go_bounds = tuple(
        _readonly(
            np.vstack(
                [
                    np.asarray(box["min"], dtype=float).reshape(3),
                    np.asarray(box["max"], dtype=float).reshape(3),
                ]
            )
        )
        for box in safety_snapshot.get("no_go_boxes_mm", [])
    )
    transit_raw = safety_snapshot.get("safe_transit_T_base_tcp")
    safe_transit = (
        _empty_transform()
        if transit_raw is None
        else _readonly(np.asarray(transit_raw, dtype=float).reshape(4, 4))
    )
    return DashboardSnapshot(
        stage=str(stage),
        plan_id=stored_id,
        total_count=len(entries),
        completed_ids=tuple(sorted(completed_set)),
        next_scan_id=next_id,
        next_line_id=next_line,
        next_reference_pose=next_ref,
        next_d_mm=next_d,
        next_theta_deg=next_theta,
        next_beta_deg=next_beta,
        next_T_base_tcp=next_target,
        next_T_base_tcp_approach=next_approach,
        next_T_base_sensor_nominal=next_sensor,
        next_tcp_xyzrpy_deg=next_xyzrpy,
        live_sequence=int(live_sequence),
        live_profile_s=_readonly(live_points),
        live_error=live_error,
        previous_scan_id=previous_scan_id,
        previous_profile_s=_readonly(previous_profile),
        accumulated_points_base=_readonly(accumulated),
        plane_estimate=estimate,
        bootstrap_plane_corners_base=_readonly(_boundary_corners(plan)),
        planned_tcp_positions_base=_readonly(planned_tcp),
        planned_approach_positions_base=_readonly(planned_approach),
        planned_sensor_positions_base=_readonly(planned_sensor),
        completed_mask=completed_mask,
        workspace_bounds_base=_readonly(workspace_bounds),
        no_go_bounds_base=no_go_bounds,
        safe_transit_T_base_tcp=safe_transit,
        observability_rank=int(observability.get("rank", 0)),
        observability_condition=float(
            observability.get("column_normalized_condition", float("inf"))
        ),
        handeye_source=str(handeye_source),
        data_errors=tuple(errors),
    )
