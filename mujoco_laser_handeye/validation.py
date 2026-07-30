from __future__ import annotations

import copy
from pathlib import Path
import time
from typing import Any, Callable

import numpy as np

from .compat.planning import (
    plan_identity,
    save_json,
    translation_offset_observability,
)
from .compat.safety import SafetyConfig
from .compat.workflow import move_validated_segment

from .adapters import MujocoRB5Robot


def filter_collision_free_plan(
    robot: MujocoRB5Robot,
    plan: dict[str, Any],
    safety: SafetyConfig,
    *,
    min_safe_scans: int = 20,
    require_all_lines: bool = True,
    min_safe_lines: int = 4,
    report_path: str | Path | None = None,
    on_progress: Callable[[int, int, bool], None] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Keep only scans whose complete transit/capture route passes in MuJoCo.

    Filtering is accepted only if the remaining dual-theta poses retain rank-4
    translation/plane-offset observability. Scan IDs are deliberately preserved
    so every dataset record remains traceable to the original theoretical plan.
    """

    if min_safe_scans <= 0:
        raise ValueError("min_safe_scans must be positive")
    if min_safe_lines <= 0:
        raise ValueError("min_safe_lines must be positive")
    transit = safety.safe_transit_T_base_tcp
    if transit is None:
        raise RuntimeError("collision filtering requires a safe transit pose")
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    total = len(plan["entries"])
    started = time.time()
    route_cache: dict[bytes, tuple[bool, str | None]] = {}
    unique_route_checks = 0
    for index, entry in enumerate(plan["entries"], start=1):
        robot.reset()
        scan_id = int(entry["scan_id"])
        approach = np.asarray(entry["T_base_tcp_approach"], dtype=float)
        target = np.asarray(entry["T_base_tcp"], dtype=float)
        route_key = np.round(
            np.stack([transit, approach, target]), decimals=8
        ).tobytes()
        cached = route_cache.get(route_key)
        if cached is not None:
            passed, reason = cached
        else:
            unique_route_checks += 1
            try:
                move_validated_segment(
                    robot, transit, safety, label=f"filter scan {scan_id} safe transit"
                )
                move_validated_segment(
                    robot, approach, safety, label=f"filter scan {scan_id} approach"
                )
                move_validated_segment(
                    robot, target, safety, label=f"filter scan {scan_id} target"
                )
                move_validated_segment(
                    robot, approach, safety, label=f"filter scan {scan_id} retreat"
                )
                move_validated_segment(
                    robot, transit, safety, label=f"filter scan {scan_id} safe return"
                )
                passed, reason = True, None
            except Exception as exc:
                passed = False
                reason = robot.last_path_error or str(exc)
            route_cache[route_key] = (passed, reason)
        if passed:
            accepted.append(copy.deepcopy(entry))
        else:
            rejected.append({"scan_id": scan_id, "reason": reason})
        if on_progress is not None:
            on_progress(index, total, passed)

    normal = np.asarray(plan["T_base_plane"], dtype=float)[:3, 2]
    observability = translation_offset_observability(accepted, normal)
    line_ids = sorted({int(entry["line_id"]) for entry in accepted})
    theta_values = sorted({float(entry["theta_deg"]) for entry in accepted})
    failures = []
    if len(accepted) < min_safe_scans:
        failures.append(f"only {len(accepted)} safe scans remain; need {min_safe_scans}")
    original_lines = sorted({int(entry["line_id"]) for entry in plan["entries"]})
    if require_all_lines and line_ids != original_lines:
        failures.append(f"safe scans cover lines {line_ids}, expected {original_lines}")
    if len(line_ids) < int(min_safe_lines):
        failures.append(
            f"safe scans cover only {len(line_ids)} circular lines; "
            f"need at least {int(min_safe_lines)}"
        )
    if len(theta_values) < 2:
        failures.append("safe scans do not retain two distinct theta values")
    if not observability["observable"]:
        failures.append("safe scans are not rank-4 observable")

    report = {
        "schema_version": 1,
        "simulator": "MuJoCo",
        "source_plan_id": plan.get("plan_id"),
        "status": "passed" if not failures else "failed",
        "total_scans": total,
        "accepted_scan_count": len(accepted),
        "rejected_scan_count": len(rejected),
        "unique_route_checks": unique_route_checks,
        "reused_route_checks": total - unique_route_checks,
        "accepted_scan_ids": [int(entry["scan_id"]) for entry in accepted],
        "rejected_scans": rejected,
        "retained_line_ids": line_ids,
        "minimum_safe_line_count": int(min_safe_lines),
        "all_lines_required": bool(require_all_lines),
        "retained_theta_deg": theta_values,
        "observability": observability,
        "failures": failures,
        "elapsed_s": float(time.time() - started),
    }
    if report_path is not None:
        save_json(report_path, report)
    robot.reset()
    if failures:
        raise RuntimeError("MuJoCo plan filter failed: " + "; ".join(failures))

    filtered = copy.deepcopy(plan)
    filtered["entries"] = accepted
    filtered["main_scan_count"] = sum(
        not bool(entry["reference_pose"]) for entry in accepted
    )
    filtered["reference_scan_count"] = sum(
        bool(entry["reference_pose"]) for entry in accepted
    )
    filtered["observability"] = observability
    filtered["mujoco_collision_filter"] = {
        "source_plan_id": plan.get("plan_id"),
        "accepted_scan_ids": report["accepted_scan_ids"],
        "rejected_scan_count": len(rejected),
        "retained_line_ids": line_ids,
        "retained_theta_deg": theta_values,
    }
    filtered["plan_id"] = plan_identity(filtered)
    report["filtered_plan_id"] = filtered["plan_id"]
    if report_path is not None:
        save_json(report_path, report)
    return filtered, report


def preflight_motion_plan(
    robot: MujocoRB5Robot,
    plan: dict[str, Any],
    safety: SafetyConfig,
    *,
    report_path: str | Path | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Execute the reviewed route in MuJoCo without taking laser captures."""

    started = time.time()
    report: dict[str, Any] = {
        "schema_version": 1,
        "simulator": "MuJoCo",
        "robot": "Rainbow Robotics RB5-850E",
        "plan_id": plan.get("plan_id"),
        "status": "running",
        "checked_scans": 0,
        "total_scans": len(plan["entries"]),
        "failure": None,
    }
    transit = safety.safe_transit_T_base_tcp
    if transit is None:
        raise RuntimeError("preflight requires safety.safe_transit_T_base_tcp")
    robot.reset()
    checked_routes: set[bytes] = set()
    try:
        move_validated_segment(robot, transit, safety, label="preflight safe transit")
        for index, entry in enumerate(plan["entries"], start=1):
            approach = np.asarray(entry["T_base_tcp_approach"], dtype=float)
            target = np.asarray(entry["T_base_tcp"], dtype=float)
            scan_id = int(entry["scan_id"])
            route_key = np.round(
                np.stack([transit, approach, target]), decimals=8
            ).tobytes()
            if route_key not in checked_routes:
                move_validated_segment(
                    robot, approach, safety, label=f"preflight scan {scan_id} approach"
                )
                move_validated_segment(
                    robot, target, safety, label=f"preflight scan {scan_id} target"
                )
                move_validated_segment(
                    robot, approach, safety, label=f"preflight scan {scan_id} retreat"
                )
                move_validated_segment(
                    robot, transit, safety, label=f"preflight scan {scan_id} safe return"
                )
                checked_routes.add(route_key)
            report["checked_scans"] = index
            if on_progress is not None:
                on_progress(index, len(plan["entries"]))
        report["status"] = "passed"
    except Exception as exc:
        report["status"] = "failed"
        report["failure"] = str(exc)
        if robot.last_path_error:
            report["controller_path_error"] = robot.last_path_error
        raise
    finally:
        report["elapsed_s"] = float(time.time() - started)
        report["unique_route_checks"] = len(checked_routes)
        report["reused_route_checks"] = report["checked_scans"] - len(checked_routes)
        report["motion_waypoint_count"] = len(robot.motion_log)
        if robot.motion_log:
            joints = np.asarray(
                [item["joint_positions_deg"] for item in robot.motion_log], dtype=float
            )
            report["joint_min_deg"] = np.min(joints, axis=0).tolist()
            report["joint_max_deg"] = np.max(joints, axis=0).tolist()
            report["max_ik_position_error_mm"] = float(
                max(item["position_error_mm"] for item in robot.motion_log)
            )
            report["max_ik_rotation_error_deg"] = float(
                max(item["rotation_error_deg"] for item in robot.motion_log)
            )
        if report_path is not None:
            save_json(report_path, report)
        robot.reset()
    return report
