from __future__ import annotations

from pathlib import Path
import time
from typing import Any, Callable

import numpy as np

from robust_laser_handeye.laser_handeye.se3 import inv_T
from robust_laser_handeye.laser_handeye.simulation import sensor_pose_from_target_line

from .compat.planning import save_json
from .compat.hardware import LaserInterface
from .compat.safety import SafetyConfig
from .compat.workflow import (
    CaptureConfig,
    capture_bootstrap_once,
    finalize_bootstrap_plane,
    move_validated_segment,
)

from .adapters import MujocoRB5Robot


def _bootstrap_lines_uv(
    plane_size_mm: np.ndarray,
    line_half_length_mm: float | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return four lines whose measured segments span both plane axes."""

    half_min = 0.5 * float(np.min(np.asarray(plane_size_mm, dtype=float)))
    line_extent = (
        0.80 * half_min
        if line_half_length_mm is None
        else min(float(line_half_length_mm), 0.85 * half_min)
    )
    if line_extent <= 0.0:
        raise ValueError("bootstrap line half length must be positive")
    center_offset = min(1.30 * line_extent, 0.70 * half_min)
    return [
        (
            np.array([-line_extent, -center_offset]),
            np.array([line_extent, -center_offset]),
        ),
        (
            np.array([-line_extent, center_offset]),
            np.array([line_extent, center_offset]),
        ),
        (
            np.array([-center_offset, -line_extent]),
            np.array([-center_offset, line_extent]),
        ),
        (
            np.array([center_offset, -line_extent]),
            np.array([center_offset, line_extent]),
        ),
    ]


def _make_bootstrap_pose(
    robot: MujocoRB5Robot,
    T_tcp_sensor_init: np.ndarray,
    line_uv: tuple[np.ndarray, np.ndarray],
    *,
    d_mm: float,
    theta_deg: float,
    beta_deg: float,
    branch_sign: float,
    approach_clearance_mm: float,
) -> dict[str, Any]:
    plane = robot.T_base_plane
    T_base_sensor = sensor_pose_from_target_line(
        plane_R=plane[:3, :3],
        plane_t=plane[:3, 3],
        line_p0=line_uv[0],
        line_p1=line_uv[1],
        d_mm=float(d_mm),
        theta_deg=float(theta_deg),
        beta_deg=float(beta_deg),
        branch_sign=float(branch_sign),
        pose_geometry="paper_incidence",
    )
    T_base_sensor_approach = T_base_sensor.copy()
    T_base_sensor_approach[:3, 3] += (
        float(approach_clearance_mm) * plane[:3, 2]
    )
    return {
        "line_p0_uv_mm": line_uv[0].tolist(),
        "line_p1_uv_mm": line_uv[1].tolist(),
        "d_mm": float(d_mm),
        "theta_deg": float(theta_deg),
        "beta_deg": float(beta_deg),
        "branch_sign": float(branch_sign),
        "T_base_tcp": (T_base_sensor @ inv_T(T_tcp_sensor_init)).tolist(),
        "T_base_tcp_approach": (
            T_base_sensor_approach @ inv_T(T_tcp_sensor_init)
        ).tolist(),
    }


def _execute_bootstrap_route(
    robot: MujocoRB5Robot,
    entry: dict[str, Any],
    safety: SafetyConfig,
    *,
    label: str,
    capture_at_target: Callable[[], dict[str, Any]] | None = None,
) -> dict[str, Any] | None:
    transit = safety.safe_transit_T_base_tcp
    if transit is None:
        raise RuntimeError("automatic bootstrap requires a safe transit pose")
    approach = np.asarray(entry["T_base_tcp_approach"], dtype=float)
    target = np.asarray(entry["T_base_tcp"], dtype=float)
    move_validated_segment(robot, transit, safety, label=f"{label} safe transit")
    move_validated_segment(robot, approach, safety, label=f"{label} approach")
    move_validated_segment(robot, target, safety, label=f"{label} target")
    record = None if capture_at_target is None else capture_at_target()
    move_validated_segment(robot, approach, safety, label=f"{label} retreat")
    move_validated_segment(robot, transit, safety, label=f"{label} safe return")
    return record


def _select_safe_bootstrap_entry(
    robot: MujocoRB5Robot,
    T_tcp_sensor_init: np.ndarray,
    line_uv: tuple[np.ndarray, np.ndarray],
    safety: SafetyConfig,
    *,
    view_index: int,
    preferred_d_mm: float,
    preferred_theta_deg: float,
    preferred_beta_deg: float,
) -> dict[str, Any]:
    failures: list[str] = []
    theta_candidates = tuple(
        dict.fromkeys([float(preferred_theta_deg), 20.0, 30.0, 45.0])
    )
    d_candidates = tuple(
        dict.fromkeys([float(preferred_d_mm), 90.0, 110.0, 130.0])
    )
    beta_candidates = tuple(
        dict.fromkeys([float(preferred_beta_deg), 90.0])
    )
    for theta_deg in theta_candidates:
        for d_mm in d_candidates:
            for beta_deg in beta_candidates:
                for branch_sign in (1.0, -1.0):
                    entry = _make_bootstrap_pose(
                        robot,
                        T_tcp_sensor_init,
                        line_uv,
                        d_mm=d_mm,
                        theta_deg=theta_deg,
                        beta_deg=beta_deg,
                        branch_sign=branch_sign,
                        approach_clearance_mm=safety.approach_clearance_mm,
                    )
                    robot.reset()
                    try:
                        _execute_bootstrap_route(
                            robot,
                            entry,
                            safety,
                            label=f"bootstrap preflight {view_index}",
                        )
                        robot.reset()
                        return entry
                    except Exception as exc:
                        failures.append(robot.last_path_error or str(exc))
    robot.reset()
    detail = failures[-1] if failures else "no candidate was generated"
    raise RuntimeError(
        f"no collision-free RB5 pose for bootstrap view {view_index}: {detail}"
    )


def _plane_estimation_errors(
    estimated_boundary: dict[str, Any], T_base_plane_true: np.ndarray
) -> dict[str, float]:
    estimated = np.asarray(
        estimated_boundary["plane_frame"]["T_base_plane"], dtype=float
    )
    normal_angle = float(
        np.degrees(
            np.arccos(
                np.clip(
                    float(estimated[:3, 2] @ T_base_plane_true[:3, 2]),
                    -1.0,
                    1.0,
                )
            )
        )
    )
    estimated_normal = estimated[:3, 2]
    estimated_center = estimated[:3, 3]
    return {
        "normal_error_deg": normal_angle,
        "signed_distance_at_true_origin_mm": float(
            estimated_normal @ (T_base_plane_true[:3, 3] - estimated_center)
        ),
        "frame_origin_error_mm": float(
            np.linalg.norm(estimated_center - T_base_plane_true[:3, 3])
        ),
    }


def capture_automatic_mujoco_bootstrap(
    *,
    robot: MujocoRB5Robot,
    laser: LaserInterface,
    T_tcp_sensor_init: np.ndarray,
    output_dir: str | Path,
    safety: SafetyConfig,
    capture: CaptureConfig,
    margin_mm: float = 20.0,
    line_half_length_mm: float | None = None,
    bootstrap_d_mm: float = 120.0,
    bootstrap_theta_deg: float = 30.0,
    bootstrap_beta_deg: float = 90.0,
    on_progress: Callable[[int, int], None] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Capture four simulated views and estimate the plane without using GT points.

    MuJoCo ground truth is used only to propose the four automatic view poses.
    The saved boundary itself is reconstructed exclusively from noisy Keyence
    profiles, TCP readback, and ``T_tcp_sensor_init`` through the real workflow.
    """

    if not isinstance(robot, MujocoRB5Robot):
        raise TypeError("automatic MuJoCo bootstrap requires a MuJoCo RB5 robot")
    if not isinstance(laser, LaserInterface):
        raise TypeError("automatic MuJoCo bootstrap requires a laser/profile source")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    T_tcp_sensor_init = np.asarray(T_tcp_sensor_init, dtype=float).reshape(4, 4)
    lines = _bootstrap_lines_uv(
        robot.scene_config.plane_size_mm,
        line_half_length_mm=line_half_length_mm,
    )
    started = time.time()
    entries = [
        _select_safe_bootstrap_entry(
            robot,
            T_tcp_sensor_init,
            line,
            safety,
            view_index=index,
            preferred_d_mm=bootstrap_d_mm,
            preferred_theta_deg=bootstrap_theta_deg,
            preferred_beta_deg=bootstrap_beta_deg,
        )
        for index, line in enumerate(lines, start=1)
    ]
    plan_record = {
        "schema_version": 1,
        "simulation_only": True,
        "gt_usage": (
            "GT plane is used only to propose bootstrap robot views; plane_boundary "
            "is fitted from captured profiles and the initial hand-eye"
        ),
        "entries": [
            {"index": index, **entry}
            for index, entry in enumerate(entries, start=1)
        ],
    }
    save_json(output_dir / "bootstrap_motion_plan.json", plan_record)

    for index, entry in enumerate(entries, start=1):
        robot.reset()

        def capture_view() -> dict[str, Any]:
            return capture_bootstrap_once(
                robot=robot,
                profile_source=laser,
                output_dir=output_dir,
                index=index,
                safety=safety,
                capture=capture,
            )

        _execute_bootstrap_route(
            robot,
            entry,
            safety,
            label=f"bootstrap capture {index}",
            capture_at_target=capture_view,
        )
        if on_progress is not None:
            on_progress(index, len(entries))

    boundary = finalize_bootstrap_plane(
        output_dir=output_dir,
        T_tcp_sensor_init=T_tcp_sensor_init,
        margin_mm=margin_mm,
        max_plane_rms_mm=capture.max_bootstrap_plane_rms_mm,
        min_span_mm=capture.min_bootstrap_span_mm,
        min_sensor_distance_mm=capture.min_bootstrap_sensor_plane_distance_mm,
    )
    errors = _plane_estimation_errors(boundary, robot.T_base_plane)
    report = {
        "schema_version": 1,
        "status": "passed",
        "capture_count": 4,
        "plane_rms_mm": float(boundary["plane"]["rms_error_mm"]),
        "observed_bounds_uv_mm": boundary["observed_bounds_uv_mm"],
        "safe_bounds_uv_mm": boundary["safe_bounds_uv_mm"],
        "T_tcp_sensor_init": T_tcp_sensor_init.tolist(),
        "requested_bootstrap_parameters": {
            "line_half_length_mm": line_half_length_mm,
            "d_mm": float(bootstrap_d_mm),
            "theta_deg": float(bootstrap_theta_deg),
            "beta_deg": float(bootstrap_beta_deg),
        },
        "simulation_gt_diagnostics": errors,
        "elapsed_s": float(time.time() - started),
    }
    save_json(output_dir / "bootstrap_report.json", report)
    robot.reset()
    return boundary, report
