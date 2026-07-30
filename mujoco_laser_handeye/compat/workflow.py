from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import threading
import time
from typing import Any, Callable

import numpy as np
from scipy.spatial.transform import Rotation

from robust_laser_handeye.laser_handeye.calibration import (
    calibrate_single_plane,
    reconstruct_points_base,
)
from robust_laser_handeye.laser_handeye.data import LaserScan
from robust_laser_handeye.laser_handeye.geometry import fit_plane_pca
from robust_laser_handeye.laser_handeye.se3 import transform_points

from .hardware import LaserInterface, ProfileSample, RobotInterface, load_adapter
from .initial_point import estimate_plane_and_boundary, load_profile
from .planning import (
    load_json,
    load_transform,
    save_json,
    translation_offset_observability,
    validate_plan_identity,
    validate_plan_runtime_safety,
)
from .safety import SafetyConfig, rotation_distance_deg, validate_segment, validate_transform


class MotionCancelled(RuntimeError):
    pass


@dataclass
class CaptureConfig:
    timeout_s: float = 3.0
    profiles_per_pose: int = 1
    min_points: int = 80
    max_profile_age_ms: float = 500.0
    max_abs_sensor_y_mm: float = 0.1
    max_initial_plane_rms_mm: float = 15.0
    max_final_plane_rms_mm: float = 2.0
    max_bootstrap_plane_rms_mm: float = 5.0
    min_bootstrap_span_mm: float = 50.0
    min_bootstrap_sensor_plane_distance_mm: float = 10.0
    return_to_safe_between_scans: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "CaptureConfig":
        kwargs = {key: data[key] for key in cls.__dataclass_fields__ if key in data}
        result = cls(**kwargs)
        if result.timeout_s <= 0 or result.profiles_per_pose <= 0 or result.min_points <= 0:
            raise ValueError("capture timeout, profiles_per_pose and min_points must be positive")
        if result.max_profile_age_ms <= 0 or result.max_abs_sensor_y_mm < 0:
            raise ValueError("invalid capture age/y limits")
        if result.max_initial_plane_rms_mm <= 0:
            raise ValueError("max_initial_plane_rms_mm must be positive")
        if result.max_final_plane_rms_mm <= 0:
            raise ValueError("max_final_plane_rms_mm must be positive")
        if result.max_bootstrap_plane_rms_mm <= 0:
            raise ValueError("max_bootstrap_plane_rms_mm must be positive")
        if result.min_bootstrap_span_mm <= 0:
            raise ValueError("min_bootstrap_span_mm must be positive")
        if result.min_bootstrap_sensor_plane_distance_mm < 0:
            raise ValueError(
                "min_bootstrap_sensor_plane_distance_mm must be non-negative"
            )
        return result


def load_runtime_config(path: str | Path) -> tuple[dict[str, Any], SafetyConfig, CaptureConfig]:
    config = load_json(path)
    safety = SafetyConfig.from_dict(dict(config.get("safety", {})))
    capture = CaptureConfig.from_dict(dict(config.get("capture", {})))
    return config, safety, capture


def _validate_bootstrap_quality_metrics(
    quality: Any, capture: CaptureConfig, *, missing_message: str
) -> None:
    if not isinstance(quality, dict) or quality.get("accepted") is not True:
        raise RuntimeError(missing_message)
    try:
        rms_mm = float(quality["plane_rms_mm"])
        u_span = float(quality["observed_u_span_mm"])
        v_span = float(quality["observed_v_span_mm"])
        sensor_distances = np.asarray(
            quality["sensor_plane_signed_distances_mm"], dtype=float
        ).reshape(-1)
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"bootstrap quality metadata is incomplete: {exc}") from exc
    values = np.concatenate([[rms_mm, u_span, v_span], sensor_distances])
    if len(sensor_distances) != 4 or not np.all(np.isfinite(values)):
        raise RuntimeError(
            "bootstrap quality metadata must contain four finite sensor distances"
        )
    failures = []
    if rms_mm < 0.0 or rms_mm > capture.max_bootstrap_plane_rms_mm:
        failures.append(
            f"RMS {rms_mm:.3f}>{capture.max_bootstrap_plane_rms_mm:.3f} mm"
        )
    if min(u_span, v_span) < capture.min_bootstrap_span_mm:
        failures.append(
            f"span u={u_span:.1f}, v={v_span:.1f}<{capture.min_bootstrap_span_mm:.1f} mm"
        )
    nearest = float(np.min(sensor_distances))
    if nearest < capture.min_bootstrap_sensor_plane_distance_mm:
        failures.append(
            "nearest sensor-plane distance "
            f"{nearest:.1f}<{capture.min_bootstrap_sensor_plane_distance_mm:.1f} mm"
        )
    if failures:
        raise RuntimeError(
            "bootstrap plane no longer satisfies the current capture limits: "
            + "; ".join(failures)
            + ". Recapture/finalize the four views and regenerate the plan"
        )


def validate_bootstrap_boundary_quality(
    boundary: dict[str, Any],
    capture: CaptureConfig,
    T_tcp_sensor_init: np.ndarray,
) -> None:
    """Require a boundary produced by the guarded four-view bootstrap fit."""
    expected_handeye = validate_transform(
        T_tcp_sensor_init, "plan T_tcp_sensor_init"
    )
    provenance = boundary.get("bootstrap_provenance")
    try:
        bootstrap_handeye = validate_transform(
            np.asarray(provenance["T_tcp_sensor_init"], dtype=float),
            "bootstrap provenance T_tcp_sensor_init",
        )
        capture_count = int(provenance["capture_count"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "plane boundary has no valid bootstrap hand-eye provenance; "
            "re-run bootstrap/finalize before planning"
        ) from exc
    if capture_count != 4:
        raise RuntimeError("bootstrap provenance must contain exactly four captures")
    if not np.allclose(
        bootstrap_handeye, expected_handeye, atol=1e-9, rtol=0.0
    ):
        raise RuntimeError(
            "plan T_tcp_sensor_init differs from the seed used to estimate the "
            "bootstrap plane/boundary; re-finalize bootstrap with this hand-eye"
        )
    quality = boundary.get("quality_gate")
    if isinstance(quality, dict):
        try:
            observed = boundary["observed_bounds_uv_mm"]
            safe = boundary["safe_bounds_uv_mm"]
            observed_values = np.array(
                [
                    observed["u_min"],
                    observed["u_max"],
                    observed["v_min"],
                    observed["v_max"],
                ],
                dtype=float,
            )
            safe_values = np.array(
                [safe["u_min"], safe["u_max"], safe["v_min"], safe["v_max"]],
                dtype=float,
            )
            if not np.all(np.isfinite(np.concatenate([observed_values, safe_values]))):
                raise ValueError("plane bounds must be finite")
            if not (
                observed_values[0] <= safe_values[0] < safe_values[1] <= observed_values[1]
                and observed_values[2] <= safe_values[2] < safe_values[3] <= observed_values[3]
            ):
                raise ValueError(
                    "safe UV bounds must be a non-empty subset of observed bounds"
                )
            quality = {
                "accepted": quality.get("accepted") is True,
                "plane_rms_mm": float(boundary["plane"]["rms_error_mm"]),
                "observed_u_span_mm": (
                    float(observed["u_max"]) - float(observed["u_min"])
                ),
                "observed_v_span_mm": (
                    float(observed["v_max"]) - float(observed["v_min"])
                ),
                "sensor_plane_signed_distances_mm": quality.get(
                    "sensor_plane_signed_distances_mm"
                ),
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                f"bootstrap quality metadata is incomplete: {exc}"
            ) from exc
    _validate_bootstrap_quality_metrics(
        quality,
        capture,
        missing_message=(
            "plane boundary has no accepted bootstrap quality gate; use the current "
            "bootstrap/finalize workflow before generating a live motion plan"
        ),
    )


def validate_plan_bootstrap_quality(
    plan: dict[str, Any], capture: CaptureConfig
) -> None:
    """Recheck the immutable plan's bootstrap evidence at live execution."""
    _validate_bootstrap_quality_metrics(
        plan.get("bootstrap_quality"),
        capture,
        missing_message=(
            "motion plan has no accepted bootstrap quality evidence; regenerate it "
            "from a boundary produced by the current bootstrap/finalize workflow"
        ),
    )
    provenance = plan.get("bootstrap_provenance")
    try:
        bootstrap_handeye = validate_transform(
            np.asarray(provenance["T_tcp_sensor_init"], dtype=float),
            "plan bootstrap provenance T_tcp_sensor_init",
        )
        capture_count = int(provenance["capture_count"])
        plan_handeye = validate_transform(
            np.asarray(plan["T_tcp_sensor_init"], dtype=float),
            "plan T_tcp_sensor_init",
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            "motion plan has no valid bootstrap hand-eye provenance; regenerate it"
        ) from exc
    if capture_count != 4 or not np.allclose(
        bootstrap_handeye, plan_handeye, atol=1e-9, rtol=0.0
    ):
        raise RuntimeError(
            "motion plan bootstrap provenance does not match its initial hand-eye; "
            "re-finalize bootstrap and regenerate the plan"
        )


def make_robot(config: dict[str, Any]) -> RobotInterface:
    section = dict(config.get("robot", {}))
    robot = load_adapter(str(section["adapter"]), dict(section.get("kwargs", {})))
    if not isinstance(robot, RobotInterface):
        raise TypeError("configured robot adapter must inherit RobotInterface")
    return robot


def make_laser(
    config: dict[str, Any], *, robot: RobotInterface | None = None
) -> LaserInterface:
    section = dict(config.get("laser", {}))
    kwargs = dict(section.get("kwargs", {}))
    if bool(section.get("inject_robot", False)):
        if robot is None:
            raise ValueError("laser.inject_robot=true requires a robot instance")
        kwargs["robot"] = robot
    laser = load_adapter(str(section["adapter"]), kwargs)
    if not isinstance(laser, LaserInterface):
        raise TypeError("configured laser adapter must inherit LaserInterface")
    return laser


def _pose_error(command: np.ndarray, readback: np.ndarray) -> tuple[float, float]:
    position = float(np.linalg.norm(command[:3, 3] - readback[:3, 3]))
    rotation = rotation_distance_deg(command[:3, :3], readback[:3, :3])
    return position, rotation


def _assert_pose_reached(
    command: np.ndarray, readback: np.ndarray, safety: SafetyConfig, *, label: str
) -> None:
    position, rotation = _pose_error(command, readback)
    if position > safety.readback_position_tolerance_mm:
        raise RuntimeError(f"{label}: TCP position error {position:.3f} mm exceeds tolerance")
    if rotation > safety.readback_rotation_tolerance_deg:
        raise RuntimeError(f"{label}: TCP rotation error {rotation:.3f} deg exceeds tolerance")


def move_validated_segment(
    robot: RobotInterface,
    end: np.ndarray,
    safety: SafetyConfig,
    *,
    label: str,
    cancel_event: threading.Event | None = None,
    on_event: Callable[[dict[str, Any]], None] | None = None,
) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise MotionCancelled(f"{label}: motion cancelled before start")
    start = validate_transform(robot.current_T_base_tcp(), f"{label} current pose")
    waypoints = validate_segment(start, end, safety, name=label)
    controller_result = robot.controller_path_is_safe([start, *waypoints])
    if controller_result is False or (
        safety.require_controller_collision_check and controller_result is not True
    ):
        raise RuntimeError(f"{label}: robot controller rejected the path")
    for index, waypoint in enumerate(waypoints):
        if cancel_event is not None and cancel_event.is_set():
            raise MotionCancelled(f"{label}: motion cancelled at waypoint {index}")
        if on_event is not None:
            on_event(
                {
                    "stage": "MOVING",
                    "label": label,
                    "waypoint_index": index,
                    "waypoint_count": len(waypoints),
                    "T_base_tcp_command": waypoint.copy(),
                }
            )
        robot.move_tcp(
            waypoint,
            linear_speed_mm_s=safety.linear_speed_mm_s,
            angular_speed_deg_s=safety.angular_speed_deg_s,
            timeout_s=safety.motion_timeout_s,
        )
        readback = validate_transform(
            robot.current_T_base_tcp(), f"{label} waypoint {index} readback"
        )
        _assert_pose_reached(waypoint, readback, safety, label=f"{label} waypoint {index}")
    if on_event is not None:
        on_event({"stage": "MOTION_COMPLETE", "label": label})


def _clean_profile(sample: ProfileSample, capture: CaptureConfig) -> np.ndarray:
    points = np.asarray(sample.points_s, dtype=float)
    points = points[np.all(np.isfinite(points), axis=1)]
    points = points[np.abs(points[:, 1]) <= capture.max_abs_sensor_y_mm]
    if len(points) < capture.min_points:
        raise RuntimeError(
            f"laser profile has {len(points)} valid points; need at least {capture.min_points}"
        )
    age_ms = (time.time_ns() - int(sample.timestamp_ns)) / 1e6
    if age_ms < -10.0 or age_ms > capture.max_profile_age_ms:
        raise RuntimeError(f"laser profile age {age_ms:.1f} ms is outside the allowed window")
    return points


def capture_stationary_profile(
    robot: RobotInterface,
    laser: LaserInterface,
    safety: SafetyConfig,
    capture: CaptureConfig,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if safety.settle_time_s > 0:
        time.sleep(safety.settle_time_s)
    before = validate_transform(robot.current_T_base_tcp(), "pre-capture TCP")
    profiles = []
    timestamps = []
    for _ in range(capture.profiles_per_pose):
        sample = laser.capture_profile(timeout_s=capture.timeout_s)
        profiles.append(_clean_profile(sample, capture))
        timestamps.append(int(sample.timestamp_ns))
    after = validate_transform(robot.current_T_base_tcp(), "post-capture TCP")
    position_delta, rotation_delta = _pose_error(before, after)
    if position_delta > safety.stationarity_position_tolerance_mm:
        raise RuntimeError(f"robot moved {position_delta:.3f} mm during profile capture")
    if rotation_delta > safety.stationarity_rotation_tolerance_deg:
        raise RuntimeError(f"robot rotated {rotation_delta:.3f} deg during profile capture")
    # Multiple profiles are retained as repeated observations.  This is robust
    # to sensors whose valid X indices differ between batches.
    points = np.vstack(profiles)
    return after, points, {
        "profile_timestamp_ns_min": min(timestamps),
        "profile_timestamp_ns_max": max(timestamps),
        "capture_time_ns": time.time_ns(),
        "stationarity_position_delta_mm": position_delta,
        "stationarity_rotation_delta_deg": rotation_delta,
    }


def _initial_plane_rms(
    points_s: np.ndarray,
    T_base_tcp: np.ndarray,
    T_tcp_sensor_init: np.ndarray,
    T_base_plane: np.ndarray,
) -> float:
    points_base = transform_points(T_base_tcp @ T_tcp_sensor_init, points_s)
    normal = T_base_plane[:3, 2]
    distances = (points_base - T_base_plane[:3, 3]) @ normal
    return float(np.sqrt(np.mean(np.square(distances))))


def _save_scan(
    output_dir: Path,
    entry: dict[str, Any],
    T_base_tcp: np.ndarray,
    points_s: np.ndarray,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    scan_id = int(entry["scan_id"])
    stem = f"scan_{scan_id:04d}"
    profile_name = f"{stem}_profile.csv"
    pose_name = f"{stem}_T_base_tcp.csv"
    profile_path = output_dir / profile_name
    pose_path = output_dir / pose_name
    profile_temporary = profile_path.with_suffix(profile_path.suffix + ".tmp")
    pose_temporary = pose_path.with_suffix(pose_path.suffix + ".tmp")
    np.savetxt(
        profile_temporary,
        points_s,
        delimiter=",",
        header="x_s_mm,y_s_mm,z_s_mm",
        comments="",
    )
    np.savetxt(pose_temporary, T_base_tcp, delimiter=",")
    profile_temporary.replace(profile_path)
    pose_temporary.replace(pose_path)
    return {
        "scan_id": scan_id,
        "profile_file": profile_name,
        "pose_file": pose_name,
        "point_count": int(len(points_s)),
        "plan": {
            key: entry[key]
            for key in (
                "line_id",
                "parameter_id",
                "reference_pose",
                "branch_sign",
                "d_mm",
                "theta_deg",
                "beta_deg",
            )
        },
        **metadata,
    }


class ScanCaptureSession:
    """Resumable one-scan-at-a-time collection state machine for CLI and UI."""

    def __init__(
        self,
        *,
        robot: RobotInterface,
        profile_source: LaserInterface,
        plan: dict[str, Any],
        output_dir: str | Path,
        safety: SafetyConfig,
        capture: CaptureConfig,
        cancel_event: threading.Event | None = None,
        on_event: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        if int(plan.get("schema_version", -1)) != 1:
            raise ValueError("unsupported or missing plan schema_version")
        self.plan_id = validate_plan_identity(plan)
        validate_plan_runtime_safety(plan, safety)
        if safety.live_enabled:
            validate_plan_bootstrap_quality(plan, capture)
        self.robot = robot
        self.profile_source = profile_source
        self.plan = plan
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.safety = safety
        self.capture = capture
        self.cancel_event = cancel_event or threading.Event()
        self.on_event = on_event
        self._operation_lock = threading.Lock()
        self._stop_lock = threading.Lock()
        self._stop_issued = threading.Event()
        self._initialized = False
        self.T_tcp_sensor_init = np.asarray(plan["T_tcp_sensor_init"], dtype=float)
        self.T_base_plane = np.asarray(plan["T_base_plane"], dtype=float)
        self.transit = safety.safe_transit_T_base_tcp
        if self.transit is None:
            raise RuntimeError("collection requires a configured safe transit pose")
        if safety.live_enabled and not robot.supports_independent_stop():
            raise RuntimeError(
                "live collection requires a robot adapter whose stop() can interrupt "
                "a concurrently blocking move_tcp() call"
            )
        self.manifest_path = self.output_dir / "manifest.json"
        expected_ids = [int(entry["scan_id"]) for entry in plan["entries"]]
        if len(expected_ids) != len(set(expected_ids)):
            raise ValueError("motion plan contains duplicate scan IDs")
        if self.manifest_path.exists():
            self.manifest = load_json(self.manifest_path)
            if self.manifest.get("plan_id") != self.plan_id:
                raise RuntimeError(
                    "dataset manifest belongs to a different motion plan; "
                    "choose another dataset directory or restore the reviewed plan"
                )
            if self.manifest.get("expected_scan_ids") != expected_ids:
                raise RuntimeError("dataset expected scan IDs do not match the motion plan")
        else:
            self.manifest = {
                "schema_version": 2,
                "plan_id": self.plan_id,
                "expected_scan_ids": expected_ids,
                "plan_observability": plan["observability"],
                "T_tcp_sensor_init": plan["T_tcp_sensor_init"],
                "T_base_plane_initial": plan["T_base_plane"],
                "scans": [],
            }
        completed = [int(item["scan_id"]) for item in self.manifest.get("scans", [])]
        if len(completed) != len(set(completed)) or any(
            scan_id not in expected_ids for scan_id in completed
        ):
            raise RuntimeError("dataset manifest contains invalid or duplicate scan IDs")

    @property
    def completed_ids(self) -> set[int]:
        return {int(item["scan_id"]) for item in self.manifest.get("scans", [])}

    @property
    def completed_count(self) -> int:
        return len(self.completed_ids)

    @property
    def total_count(self) -> int:
        return len(self.plan["entries"])

    def next_entry(self) -> dict[str, Any] | None:
        completed = self.completed_ids
        return next(
            (
                entry
                for entry in self.plan["entries"]
                if int(entry["scan_id"]) not in completed
            ),
            None,
        )

    def _emit(self, stage: str, **payload: Any) -> None:
        if self.on_event is not None:
            self.on_event({"stage": stage, **payload})

    def _move(self, end: np.ndarray, *, label: str) -> None:
        move_validated_segment(
            self.robot,
            end,
            self.safety,
            label=label,
            cancel_event=self.cancel_event,
            on_event=self.on_event,
        )

    def _issue_stop_once(self) -> None:
        if self._stop_issued.is_set():
            return
        with self._stop_lock:
            if self._stop_issued.is_set():
                return
            # Mark before entering the vendor call so a concurrent UI/worker
            # fault cannot invoke a non-reentrant stop API a second time.
            self._stop_issued.set()
            self.robot.stop()

    def capture_next(self) -> dict[str, Any] | None:
        if not self._operation_lock.acquire(blocking=False):
            raise RuntimeError("another scan operation is already running")
        try:
            if self.cancel_event.is_set():
                raise MotionCancelled("collection is cancelled")
            entry = self.next_entry()
            if entry is None:
                self._emit("COMPLETE", completed=self.completed_count, total=self.total_count)
                return None
            scan_id = int(entry["scan_id"])
            self._emit(
                "SCAN_START",
                scan_id=scan_id,
                completed=self.completed_count,
                total=self.total_count,
            )
            try:
                if not self._initialized:
                    self._move(self.transit, label="initial safe-transit move")
                    self._initialized = True
                approach = np.asarray(entry["T_base_tcp_approach"], dtype=float)
                target = np.asarray(entry["T_base_tcp"], dtype=float)
                self._move(approach, label=f"scan {scan_id} approach")
                self._move(target, label=f"scan {scan_id} target")
                if self.cancel_event.is_set():
                    raise MotionCancelled(
                        f"scan {scan_id}: cancelled before stationary capture"
                    )
                self._emit("SETTLING", scan_id=scan_id)
                T_readback, points_s, timestamps = capture_stationary_profile(
                    self.robot, self.profile_source, self.safety, self.capture
                )
                if self.cancel_event.is_set():
                    raise MotionCancelled(
                        f"scan {scan_id}: cancelled during stationary capture"
                    )
                self._emit("CAPTURING", scan_id=scan_id, point_count=len(points_s))
                _assert_pose_reached(
                    target, T_readback, self.safety, label=f"scan {scan_id} capture"
                )
                plane_rms = _initial_plane_rms(
                    points_s,
                    T_readback,
                    self.T_tcp_sensor_init,
                    self.T_base_plane,
                )
                if plane_rms > self.capture.max_initial_plane_rms_mm:
                    raise RuntimeError(
                        f"scan {scan_id}: reconstructed profile is {plane_rms:.3f} mm RMS "
                        "from the bootstrap plane; check the plane, TCP convention, "
                        "hand-eye seed, or laser units"
                    )
                record = _save_scan(
                    self.output_dir,
                    entry,
                    T_readback,
                    points_s,
                    {"initial_plane_rms_mm": plane_rms, **timestamps},
                )
                self.manifest["scans"].append(record)
                save_json(self.manifest_path, self.manifest)
                self._move(approach, label=f"scan {scan_id} retreat")
                if (
                    self.capture.return_to_safe_between_scans
                    or self.completed_count == self.total_count
                ):
                    self._move(
                        self.transit, label=f"scan {scan_id} safe return"
                    )
                self._emit(
                    "SCAN_COMPLETE",
                    scan_id=scan_id,
                    completed=self.completed_count,
                    total=self.total_count,
                    point_count=len(points_s),
                    initial_plane_rms_mm=plane_rms,
                )
                return record
            except BaseException as exc:
                self.cancel_event.set()
                stop_error = None
                try:
                    self._issue_stop_once()
                except Exception as stop_exc:
                    stop_error = stop_exc
                message = str(exc)
                if stop_error is not None:
                    message += f"; controlled stop failed: {stop_error}"
                self._emit("FAULT", scan_id=scan_id, message=message)
                raise
        finally:
            self._operation_lock.release()

    def finalize(self) -> None:
        if self.cancel_event.is_set():
            return
        self._move(self.transit, label="final safe-transit move")
        self._emit("SAFE_IDLE", completed=self.completed_count, total=self.total_count)

    def request_stop(self) -> None:
        self.cancel_event.set()
        self._issue_stop_once()
        self._emit("STOPPING")


def collect_plan(
    *,
    robot: RobotInterface,
    laser: LaserInterface,
    plan: dict[str, Any],
    output_dir: str | Path,
    safety: SafetyConfig,
    capture: CaptureConfig,
) -> dict[str, Any]:
    session = ScanCaptureSession(
        robot=robot,
        profile_source=laser,
        plan=plan,
        output_dir=output_dir,
        safety=safety,
        capture=capture,
    )
    while True:
        record = session.capture_next()
        if record is None:
            break
        print(
            f"[capture {session.completed_count}/{session.total_count}] "
            f"points={record['point_count']} "
            f"initial-plane-rms={record['initial_plane_rms_mm']:.3f} mm"
        )
    session.finalize()
    return session.manifest


def capture_bootstrap_plane(
    *,
    robot: RobotInterface,
    laser: LaserInterface,
    T_tcp_sensor_init: np.ndarray,
    output_dir: str | Path,
    safety: SafetyConfig,
    capture: CaptureConfig,
    margin_mm: float,
    prompt: Callable[[str], str] = input,
) -> dict[str, Any]:
    """Capture four manually jogged, stationary profiles and estimate the plane.

    Automatic motion is intentionally not used before the plane location is
    known.  The operator should jog four well-separated views around the usable
    target region with the teach pendant, then press Enter for each capture.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for index in range(4):
        answer = prompt(
            f"[{index + 1}/4] Jog to a stationary, collision-checked plane view; "
            "press Enter to capture or type q to abort: "
        )
        if answer.strip().lower() == "q":
            raise KeyboardInterrupt("bootstrap capture aborted")
        capture_bootstrap_once(
            robot=robot,
            profile_source=laser,
            output_dir=output_dir,
            index=index + 1,
            safety=safety,
            capture=capture,
        )
    return finalize_bootstrap_plane(
        output_dir=output_dir,
        T_tcp_sensor_init=T_tcp_sensor_init,
        margin_mm=margin_mm,
        max_plane_rms_mm=capture.max_bootstrap_plane_rms_mm,
        min_span_mm=capture.min_bootstrap_span_mm,
        min_sensor_distance_mm=capture.min_bootstrap_sensor_plane_distance_mm,
    )


def capture_bootstrap_once(
    *,
    robot: RobotInterface,
    profile_source: LaserInterface,
    output_dir: str | Path,
    index: int,
    safety: SafetyConfig,
    capture: CaptureConfig,
) -> dict[str, Any]:
    """Capture one manually jogged bootstrap view without commanding motion."""
    index = int(index)
    if not 1 <= index <= 4:
        raise ValueError("bootstrap index must be in [1, 4]")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pose, points, metadata = capture_stationary_profile(
        robot, profile_source, safety, capture
    )
    safety.assert_pose_safe(pose, name=f"bootstrap pose {index}")
    pose_path = output_dir / f"bootstrap_pose_{index}.csv"
    profile_path = output_dir / f"bootstrap_profile_{index}.csv"
    pose_tmp = pose_path.with_suffix(".csv.tmp")
    profile_tmp = profile_path.with_suffix(".csv.tmp")
    np.savetxt(pose_tmp, pose, delimiter=",")
    np.savetxt(
        profile_tmp,
        points,
        delimiter=",",
        header="x_s_mm,y_s_mm,z_s_mm",
        comments="",
    )
    pose_tmp.replace(pose_path)
    profile_tmp.replace(profile_path)
    record = {
        "index": index,
        "pose_file": pose_path.name,
        "profile_file": profile_path.name,
        "point_count": int(len(points)),
        **metadata,
    }
    manifest_path = output_dir / "bootstrap_manifest.json"
    manifest = load_json(manifest_path) if manifest_path.exists() else {"captures": []}
    manifest["captures"] = [
        item for item in manifest.get("captures", []) if int(item["index"]) != index
    ] + [record]
    manifest["captures"].sort(key=lambda item: int(item["index"]))
    save_json(manifest_path, manifest)
    return record


def finalize_bootstrap_plane(
    *,
    output_dir: str | Path,
    T_tcp_sensor_init: np.ndarray,
    margin_mm: float,
    max_plane_rms_mm: float = 5.0,
    min_span_mm: float = 50.0,
    min_sensor_distance_mm: float = 10.0,
) -> dict[str, Any]:
    margin_mm = float(margin_mm)
    if not np.isfinite(margin_mm) or margin_mm < 0.0:
        raise ValueError("margin_mm must be finite and non-negative")
    if max_plane_rms_mm <= 0:
        raise ValueError("max_plane_rms_mm must be positive")
    if min_span_mm <= 0:
        raise ValueError("min_span_mm must be positive")
    if min_sensor_distance_mm < 0:
        raise ValueError("min_sensor_distance_mm must be non-negative")
    output_dir = Path(output_dir)
    T_tcp_sensor_init = validate_transform(
        T_tcp_sensor_init, "bootstrap T_tcp_sensor_init"
    )
    poses = []
    profiles = []
    for index in range(1, 5):
        pose_path = output_dir / f"bootstrap_pose_{index}.csv"
        profile_path = output_dir / f"bootstrap_profile_{index}.csv"
        if not pose_path.exists() or not profile_path.exists():
            raise RuntimeError(f"bootstrap capture {index}/4 is missing")
        poses.append(load_transform(pose_path))
        profiles.append(load_profile(profile_path))
    boundary = estimate_plane_and_boundary(
        T_ef_s_init=T_tcp_sensor_init,
        T_base_ef_list=poses,
        profile_list=profiles,
        margin_mm=margin_mm,
    )
    rms_mm = float(boundary["plane"]["rms_error_mm"])
    if not np.isfinite(rms_mm) or rms_mm > float(max_plane_rms_mm):
        raise RuntimeError(
            f"bootstrap plane RMS {rms_mm:.3f} mm exceeds "
            f"{float(max_plane_rms_mm):.3f} mm; recapture four stationary, "
            "well-distributed views"
        )
    observed = boundary["observed_bounds_uv_mm"]
    u_span = float(observed["u_max"]) - float(observed["u_min"])
    v_span = float(observed["v_max"]) - float(observed["v_min"])
    if min(u_span, v_span) < float(min_span_mm):
        raise RuntimeError(
            f"bootstrap observed span is only u={u_span:.1f}, v={v_span:.1f} mm; "
            f"both must be at least {float(min_span_mm):.1f} mm. "
            "Spread the four views across the usable target region"
        )
    normal = np.asarray(boundary["plane"]["normal_base"], dtype=float)
    center = np.asarray(boundary["plane"]["center_base_mm"], dtype=float)
    sensor_origins = np.asarray(
        [(pose @ T_tcp_sensor_init)[:3, 3] for pose in poses], dtype=float
    )
    signed_distances = (sensor_origins - center) @ normal
    nearest_distance = float(np.min(signed_distances))
    if nearest_distance < float(min_sensor_distance_mm):
        raise RuntimeError(
            "bootstrap plane orientation/clearance is inconsistent: nearest "
            f"sensor origin is {nearest_distance:.1f} mm on the normal side; "
            f"need at least {float(min_sensor_distance_mm):.1f} mm"
        )
    boundary["quality_gate"] = {
        "accepted": True,
        "max_plane_rms_mm": float(max_plane_rms_mm),
        "min_span_mm": float(min_span_mm),
        "min_sensor_distance_mm": float(min_sensor_distance_mm),
        "observed_u_span_mm": u_span,
        "observed_v_span_mm": v_span,
        "sensor_plane_signed_distances_mm": signed_distances.tolist(),
    }
    boundary["bootstrap_provenance"] = {
        "capture_count": 4,
        "T_tcp_sensor_init": T_tcp_sensor_init.tolist(),
        "transform_convention": "T_tcp_sensor maps sensor coordinates into TCP",
    }
    save_json(output_dir / "plane_boundary.json", boundary)
    return boundary


def load_dataset(dataset_dir: str | Path) -> tuple[dict[str, Any], list[LaserScan]]:
    dataset_dir = Path(dataset_dir)
    manifest = load_json(dataset_dir / "manifest.json")
    scans = []
    for record in sorted(manifest["scans"], key=lambda item: int(item["scan_id"])):
        pose = load_transform(dataset_dir / record["pose_file"])
        try:
            points = np.loadtxt(dataset_dir / record["profile_file"], delimiter=",", skiprows=1)
        except ValueError:
            points = np.loadtxt(dataset_dir / record["profile_file"], delimiter=",")
        points = np.asarray(points, dtype=float).reshape(-1, 3)
        scans.append(
            LaserScan(
                T_base_ef=pose,
                points_s=points,
                plane_id=0,
                scan_id=int(record["scan_id"]),
                meta=dict(record.get("plan", {})),
            )
        )
    if not scans:
        raise ValueError("dataset contains no scans")
    return manifest, scans


def calibrate_dataset(
    *,
    dataset_dir: str | Path,
    T_tcp_sensor_init: np.ndarray,
    output_transform: str | Path,
    max_iter: int = 30,
    tol: float = 1e-9,
    max_translation_offset_condition: float = 1e6,
    linear_multistart: bool = True,
    linear_multistart_threshold_mm: float = 1.0,
    linear_multistart_angle_deg: float = 30.0,
    max_final_plane_rms_mm: float = 2.0,
    allow_partial: bool = False,
    expected_plan_id: str | None = None,
) -> dict[str, Any]:
    manifest, scans = load_dataset(dataset_dir)
    if expected_plan_id is not None and manifest.get("plan_id") != expected_plan_id:
        raise RuntimeError(
            "dataset manifest does not belong to the motion plan currently selected"
        )
    expected_ids = manifest.get("expected_scan_ids")
    captured_ids = [int(scan.scan_id) for scan in scans if scan.scan_id is not None]
    if not allow_partial:
        if expected_ids is None:
            raise RuntimeError(
                "dataset manifest has no reviewed-plan identity/expected scan list; "
                "recollect it or explicitly allow partial legacy calibration"
            )
        if set(captured_ids) != {int(value) for value in expected_ids}:
            raise RuntimeError(
                f"dataset is incomplete ({len(set(captured_ids))}/{len(expected_ids)} scans); "
                "finish the reviewed plan before calibration"
            )
    T_tcp_sensor_init = validate_transform(T_tcp_sensor_init, "T_tcp_sensor_init")
    manifest_init = manifest.get("T_tcp_sensor_init")
    if manifest_init is not None and not np.allclose(
        T_tcp_sensor_init,
        np.asarray(manifest_init, dtype=float).reshape(4, 4),
        atol=1e-9,
        rtol=0.0,
    ):
        raise RuntimeError(
            "calibration initial hand-eye differs from the transform embedded in "
            "the reviewed capture plan/dataset"
        )
    initial_points_base = reconstruct_points_base(scans, T_tcp_sensor_init)
    normal, _offset, _centroid, initial_rms = fit_plane_pca(initial_points_base)
    pose_entries = [{"T_base_tcp": scan.T_base_ef.tolist()} for scan in scans]
    observability = translation_offset_observability(pose_entries, normal)
    if not observability["observable"]:
        raise np.linalg.LinAlgError(
            "actual captured TCP poses are rank deficient for translation/plane offset; "
            "do not calibrate theta=30-only unknown-plane data"
        )
    if linear_multistart_threshold_mm <= 0.0:
        raise ValueError("linear_multistart_threshold_mm must be positive")
    if linear_multistart_angle_deg <= 0.0:
        raise ValueError("linear_multistart_angle_deg must be positive")
    if max_final_plane_rms_mm <= 0.0:
        raise ValueError("max_final_plane_rms_mm must be positive")

    result = calibrate_single_plane(
        scans,
        T_init=T_tcp_sensor_init,
        max_iter=int(max_iter),
        tol=float(tol),
        plane_offset_mode="joint",
        max_translation_offset_condition=float(max_translation_offset_condition),
    )
    candidates = [result]

    def candidate_plane_rms(candidate) -> float:
        points = reconstruct_points_base(scans, candidate.T_ef_s)
        return float(fit_plane_pca(points)[3])

    candidate_rms = [candidate_plane_rms(result)]
    selected = 0
    multistart_used = bool(
        linear_multistart
        and candidate_rms[0] > float(linear_multistart_threshold_mm)
    )
    if multistart_used:
        initial_euler = Rotation.from_matrix(T_tcp_sensor_init[:3, :3]).as_euler(
            "xyz", degrees=True
        )
        for axis in range(3):
            for sign in (-1.0, 1.0):
                euler = initial_euler.copy()
                euler[axis] += sign * float(linear_multistart_angle_deg)
                retry_init = T_tcp_sensor_init.copy()
                retry_init[:3, :3] = Rotation.from_euler(
                    "xyz", euler, degrees=True
                ).as_matrix()
                try:
                    candidate = calibrate_single_plane(
                        scans,
                        T_init=retry_init,
                        max_iter=int(max_iter),
                        tol=float(tol),
                        plane_offset_mode="joint",
                        max_translation_offset_condition=float(
                            max_translation_offset_condition
                        ),
                    )
                except np.linalg.LinAlgError:
                    continue
                candidates.append(candidate)
                candidate_rms.append(candidate_plane_rms(candidate))
        eligible = [
            index for index, candidate in enumerate(candidates) if candidate.converged
        ]
        if not eligible:
            eligible = list(range(len(candidates)))
        selected = min(eligible, key=lambda index: candidate_rms[index])
        result = candidates[selected]
    output_transform = Path(output_transform)
    output_transform.parent.mkdir(parents=True, exist_ok=True)
    final_plane_rms_mm = float(candidate_rms[selected])
    accepted = bool(
        result.converged and final_plane_rms_mm <= float(max_final_plane_rms_mm)
    )
    diagnostics = {
        "dataset_dir": str(Path(dataset_dir)),
        "scan_count": len(scans),
        "complete_dataset_required": not bool(allow_partial),
        "point_count": int(sum(scan.num_points for scan in scans)),
        "solver": "single-plane joint linear plane-offset iteration",
        "nonlinear_refinement": False,
        "linear_multistart_enabled": bool(linear_multistart),
        "linear_multistart_used": multistart_used,
        "linear_start_count": len(candidates),
        "linear_selected_start_index": int(selected),
        "linear_candidate_plane_rms_mm": [float(value) for value in candidate_rms],
        "linear_multistart_threshold_mm": float(linear_multistart_threshold_mm),
        "linear_multistart_angle_deg": float(linear_multistart_angle_deg),
        "converged": bool(result.converged),
        "accepted": accepted,
        "iterations": int(result.iterations),
        "initial_plane_rms_mm": float(initial_rms),
        "final_plane_rms_mm": final_plane_rms_mm,
        "max_final_plane_rms_mm": float(max_final_plane_rms_mm),
        "observability": observability,
        "rank_history": [int(value) for value in result.rank_history],
        "condition_history": [float(value) for value in result.cond_history],
        "delta_history": [float(value) for value in result.delta_history],
        "plane_offsets_mm": [float(value) for value in result.plane_offsets],
        "T_tcp_sensor": result.T_ef_s.tolist(),
        "source_plan_observability": manifest.get("plan_observability"),
    }
    save_json(output_transform.with_suffix(".diagnostics.json"), diagnostics)
    if not accepted:
        reason = (
            "linear calibration did not converge"
            if not result.converged
            else (
                f"final plane RMS {final_plane_rms_mm:.3f} mm exceeds the "
                f"acceptance limit {float(max_final_plane_rms_mm):.3f} mm"
            )
        )
        raise RuntimeError(
            reason
            + "; diagnostics were saved but the calibrated transform was not "
            "written or activated"
        )
    temporary_transform = output_transform.with_suffix(
        output_transform.suffix + ".tmp"
    )
    np.savetxt(temporary_transform, result.T_ef_s, delimiter=",")
    temporary_transform.replace(output_transform)
    return diagnostics
