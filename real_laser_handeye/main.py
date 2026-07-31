from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import time
import uuid

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets

from robust_laser_handeye.laser_handeye.calibration import calibrate_single_plane

try:
    from .calibrate_only.with_ransac_calibrate_only import (
        load_scans as load_scans_with_ransac,
        save_profile_ransac_diagnostics,
    )
except ImportError:  # Direct execution: python real_laser_handeye/main.py
    from calibrate_only.with_ransac_calibrate_only import (
        load_scans as load_scans_with_ransac,
        save_profile_ransac_diagnostics,
    )

try:
    from .util.make_plot import save_plane_rms_plot
except ImportError:  # Direct execution: python real_laser_handeye/main.py
    from util.make_plot import save_plane_rms_plot

try:
    from .laser_adapter import LaserAdapter
    from .robot_adapter import RobotAdapter
except ImportError:  # Direct execution: python real_laser_handeye/main.py
    from laser_adapter import LaserAdapter
    from real_laser_handeye.robot_adapter import RobotAdapter



class LiveMonitor:
    """Fast PyQtGraph visualization of the laser profile and TCP pose."""

    def __init__(self, history_length: int = 200) -> None:
        self.history_length = max(2, int(history_length))
        self.times: list[float] = []
        self.xyz_history: list[np.ndarray] = []
        self.start_time = time.monotonic()
        self.capture_requested = False
        self.calibration_requested = False
        self.quit_requested = False

        pg.setConfigOptions(antialias=False, background="k", foreground="w")
        self.app = pg.mkQApp("Laser hand-eye monitor")
        self.window = pg.GraphicsLayoutWidget(
            title="Laser profile and TCP monitor",
            show=True,
        )
        self.window.resize(1200, 520)

        self.profile_plot = self.window.addPlot(
            row=0,
            col=0,
            title="Waiting for laser profile...",
        )
        self.profile_plot.setLabel("bottom", "Sensor x", units="mm")
        self.profile_plot.setLabel("left", "Sensor z", units="mm")
        self.profile_plot.showGrid(x=True, y=True, alpha=0.3)
        self.profile_curve = self.profile_plot.plot(
            pen=pg.mkPen((0, 220, 255), width=1),
        )

        self.tcp_plot = self.window.addPlot(
            row=0,
            col=1,
            title="Waiting for TCP sample...",
        )
        self.tcp_plot.setLabel("bottom", "Elapsed time", units="s")
        self.tcp_plot.setLabel("left", "TCP position", units="mm")
        self.tcp_plot.showGrid(x=True, y=True, alpha=0.3)
        self.tcp_plot.addLegend()
        self.tcp_x_curve = self.tcp_plot.plot(
            pen=pg.mkPen((255, 90, 90), width=2), name="x"
        )
        self.tcp_y_curve = self.tcp_plot.plot(
            pen=pg.mkPen((90, 255, 120), width=2), name="y"
        )
        self.tcp_z_curve = self.tcp_plot.plot(
            pen=pg.mkPen((100, 160, 255), width=2), name="z"
        )

        controls = QtWidgets.QWidget()
        controls_layout = QtWidgets.QHBoxLayout(controls)
        self.capture_button = QtWidgets.QPushButton("Capture")
        self.calibrate_button = QtWidgets.QPushButton("Calibrate (RANSAC)")
        self.quit_button = QtWidgets.QPushButton("Quit")
        self.status_label = QtWidgets.QLabel("Ready")
        controls_layout.addWidget(self.capture_button)
        controls_layout.addWidget(self.calibrate_button)
        controls_layout.addWidget(self.quit_button)
        controls_layout.addWidget(self.status_label, 1)
        controls_proxy = QtWidgets.QGraphicsProxyWidget()
        controls_proxy.setWidget(controls)
        self.window.addItem(controls_proxy, row=1, col=0, colspan=2)

        self.capture_button.clicked.connect(self._request_capture)
        self.calibrate_button.clicked.connect(self._request_calibration)
        self.quit_button.clicked.connect(self._request_quit)
        self.process_events()

    def _request_capture(self) -> None:
        self.capture_requested = True

    def _request_calibration(self) -> None:
        self.calibration_requested = True

    def _request_quit(self) -> None:
        self.quit_requested = True

    def take_requests(self) -> tuple[bool, bool, bool]:
        requests = (
            self.capture_requested,
            self.calibration_requested,
            self.quit_requested,
        )
        self.capture_requested = False
        self.calibration_requested = False
        return requests

    def set_busy(self, busy: bool, status: str) -> None:
        self.capture_button.setEnabled(not busy)
        self.calibrate_button.setEnabled(not busy)
        self.status_label.setText(status)
        self.process_events()

    def show_rms(self, history: list[float]) -> None:
        """Show calibration RMS convergence in a separate PyQtGraph window."""
        values = np.asarray(history, dtype=float)
        self.rms_window = pg.plot(
            np.arange(len(values)),
            values,
            pen=pg.mkPen((0, 180, 255), width=2),
            symbol="o",
            symbolSize=5,
            title="Calibration plane RMS convergence",
        )
        self.rms_window.setLabel("bottom", "Iteration")
        self.rms_window.setLabel("left", "Plane RMS", units="mm")
        self.rms_window.showGrid(x=True, y=True, alpha=0.3)
        self.process_events()

    def process_events(self) -> None:
        """Keep the monitor window visible and responsive without blocking."""
        self.app.processEvents()

    @staticmethod
    def _rotation_to_rpy_deg(rotation: np.ndarray) -> np.ndarray:
        # The adapter constructs R as extrinsic XYZ: Rz @ Ry @ Rx.
        sy = float(np.hypot(rotation[0, 0], rotation[1, 0]))
        if sy > 1e-9:
            rx = math.atan2(rotation[2, 1], rotation[2, 2])
            ry = math.atan2(-rotation[2, 0], sy)
            rz = math.atan2(rotation[1, 0], rotation[0, 0])
        else:
            rx = math.atan2(-rotation[1, 2], rotation[1, 1])
            ry = math.atan2(-rotation[2, 0], sy)
            rz = 0.0
        return np.degrees([rx, ry, rz])

    def update(
        self,
        points: np.ndarray | None,
        T_base_tcp: np.ndarray,
        profile_status: str | None = None,
    ) -> None:
        profile = None if points is None else np.asarray(points, dtype=float)
        transform = validate_transform(T_base_tcp, "live TCP")

        elapsed = time.monotonic() - self.start_time
        xyz = transform[:3, 3].copy()
        rpy = self._rotation_to_rpy_deg(transform[:3, :3])

        self.times.append(elapsed)
        self.xyz_history.append(xyz)
        if len(self.times) > self.history_length:
            self.times = self.times[-self.history_length:]
            self.xyz_history = self.xyz_history[-self.history_length:]

        if profile is not None and len(profile):
            valid = np.all(np.isfinite(profile), axis=1)
            visible = profile[valid]
            if len(visible):
                self.profile_curve.setData(visible[:, 0], visible[:, 2])
            else:
                self.profile_curve.setData([], [])
        else:
            self.profile_curve.setData([], [])

        if profile is None:
            self.profile_plot.setTitle(
                f"{profile_status or 'No fresh laser profile'} "
                f"(t={elapsed:.1f} s)"
            )
        else:
            self.profile_plot.setTitle(
                f"Live laser profile ({len(profile)} points, t={elapsed:.1f} s)"
            )

        history = np.asarray(self.xyz_history, dtype=float)
        if len(history):
            timeline = np.asarray(self.times, dtype=float)
            self.tcp_x_curve.setData(timeline, history[:, 0])
            self.tcp_y_curve.setData(timeline, history[:, 1])
            self.tcp_z_curve.setData(timeline, history[:, 2])
        self.tcp_plot.setTitle(
            f"TCP position history (t={elapsed:.1f} s)\n"
            f"xyz=[{xyz[0]:.2f}, {xyz[1]:.2f}, {xyz[2]:.2f}] mm  "
            f"rpy=[{rpy[0]:.2f}, {rpy[1]:.2f}, {rpy[2]:.2f}] deg"
        )
        self.process_events()

    def close(self) -> None:
        self.window.close()
        self.process_events()


def read_live_sample(
    robot: RobotAdapter,
    laser: LaserAdapter,
    profile_stale_after_s: float,
) -> tuple[np.ndarray | None, np.ndarray, str | None]:
    """Always sample TCP; treat missing/bad laser data as display state."""
    transform = validate_transform(robot.read_T_base_tcp(), "live TCP")
    try:
        points = laser.read_latest_profile(max_age_s=profile_stale_after_s)
        if points is None:
            return None, transform, "No fresh laser profile"
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("laser adapter must return an (N, 3) array")
        return points, transform, None
    except Exception as exc:
        return None, transform, f"Laser profile error: {type(exc).__name__}"


def _format_api_result(value: object) -> str:
    return f"0x{int(value) & 0xFFFFFFFF:08X}"


def print_laser_diagnostics(
    diagnostics: dict[str, object],
    *,
    missing_for_s: float,
) -> None:
    """Print a compact, state-preserving Keyence diagnostic snapshot."""
    error_codes = diagnostics["error_codes"]
    error_text = (
        "query failed"
        if error_codes is None
        else ", ".join(f"0x{int(code):04X}" for code in error_codes) or "none"
    )
    attention = diagnostics["attention_status"]
    attention_text = (
        "query failed" if attention is None else f"0x{int(attention):04X}"
    )
    trigger_count = diagnostics["trigger_count"]
    encoder_count = diagnostics["encoder_count"]
    callback_age = diagnostics["callback_age_s"]
    profile_callback_age = diagnostics["profile_callback_age_s"]
    last_notify = diagnostics["last_notify"]

    print(
        f"\nKEYENCE DIAGNOSTICS: no valid profile for {missing_for_s:.1f} s"
    )
    print(
        "  API results: "
        f"GetError={_format_api_result(diagnostics['get_error_result'])}, "
        "GetAttentionStatus="
        f"{_format_api_result(diagnostics['get_attention_result'])}, "
        "GetTriggerAndPulseCount="
        f"{_format_api_result(diagnostics['get_count_result'])}"
    )
    print(f"  controller errors: {error_text}")
    print(f"  attention status: {attention_text}")
    print(
        "  trigger/encoder count: "
        f"{trigger_count if trigger_count is not None else 'unknown'} / "
        f"{encoder_count if encoder_count is not None else 'unknown'}"
    )
    print(
        "  callbacks: "
        f"total={diagnostics['callback_count']}, "
        f"accepted={diagnostics['accepted_callback_count']}, "
        f"dropped={diagnostics['dropped_callback_count']}, "
        "last_notify="
        f"{f'0x{int(last_notify):X}' if last_notify is not None else 'none'}"
    )
    print(
        "  callback age: "
        f"{f'{float(callback_age):.3f} s' if callback_age is not None else 'never'}, "
        "accepted profile age: "
        f"{f'{float(profile_callback_age):.3f} s' if profile_callback_age is not None else 'never'}, "
        f"latest buffer has data={diagnostics['latest_buffer_has_data']}"
    )


def validate_transform(value: np.ndarray, name: str) -> np.ndarray:
    transform = np.asarray(value, dtype=float)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    if not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-8):
        raise ValueError(f"{name} has an invalid homogeneous last row")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError(f"{name} rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
        raise ValueError(f"{name} rotation determinant must be +1")
    return transform.copy()


def rotation_distance_deg(first: np.ndarray, second: np.ndarray) -> float:
    relative = first.T @ second
    cosine = np.clip((float(np.trace(relative)) - 1.0) * 0.5, -1.0, 1.0)
    return math.degrees(math.acos(float(cosine)))


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def atomic_matrix(path: Path, value: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            np.savetxt(stream, value, delimiter=",", fmt="%.12g")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def load_transform(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            value = value["T_tcp_sensor"]
        return validate_transform(np.asarray(value, dtype=float), str(path))
    try:
        value = np.loadtxt(path, delimiter=",")
    except ValueError:
        value = np.loadtxt(path)
    return validate_transform(value, str(path))


def next_capture_path(dataset_dir: Path) -> Path:
    ids = []
    for path in dataset_dir.glob("capture_*.npz"):
        try:
            ids.append(int(path.stem.split("_")[-1]))
        except ValueError:
            continue
    return dataset_dir / f"capture_{max(ids, default=0) + 1:04d}.npz"


def capture_once(robot: RobotAdapter, laser: LaserAdapter, args) -> Path:
    before = validate_transform(robot.read_T_base_tcp(), "pre-capture TCP")
    points = np.asarray(
        laser.read_profile(timeout_s=args.timeout_s), dtype=float
    )
    profile_timestamp_ns = time.time_ns()
    after = validate_transform(robot.read_T_base_tcp(), "post-capture TCP")
    tcp_timestamp_ns = time.time_ns()

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("laser adapter must return an (N, 3) array")
    valid = np.all(np.isfinite(points), axis=1)
    valid &= np.abs(points[:, 1]) <= args.max_abs_sensor_y_mm
    points = points[valid]
    if len(points) < args.min_points:
        raise RuntimeError(
            f"profile has {len(points)} valid points; need {args.min_points}"
        )

    translation_delta = float(np.linalg.norm(after[:3, 3] - before[:3, 3]))
    rotation_delta = rotation_distance_deg(before[:3, :3], after[:3, :3])
    if translation_delta > args.max_stationarity_translation_mm:
        raise RuntimeError(
            f"robot moved {translation_delta:.3f} mm during capture"
        )
    if rotation_delta > args.max_stationarity_rotation_deg:
        raise RuntimeError(
            f"robot rotated {rotation_delta:.3f} deg during capture"
        )

    args.dataset_dir.mkdir(parents=True, exist_ok=True)
    output = next_capture_path(args.dataset_dir)
    temporary = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            np.savez_compressed(
                stream,
                T_base_tcp=after,
                points_s=points,
                tcp_timestamp_ns=np.int64(tcp_timestamp_ns),
                profile_timestamp_ns=np.int64(profile_timestamp_ns),
                captured_at=np.array(datetime.now(timezone.utc).isoformat()),
            )
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()

    # Mock adapters can move to their next synthetic pose only after the
    # pre/post TCP stationarity check and atomic capture save have completed.
    advance_pose = getattr(robot, "advance_pose", None)
    if callable(advance_pose):
        advance_pose()

    print(f"saved {output} ({len(points)} points)")
    return output


def calibrate(args, monitor: LiveMonitor | None = None) -> np.ndarray:
    if args.ransac_threshold_mm <= 0:
        raise ValueError("--ransac-threshold-mm must be positive")
    if args.ransac_max_iterations < 1:
        raise ValueError("--ransac-max-iterations must be at least 1")
    if args.ransac_min_inliers < 2:
        raise ValueError("--ransac-min-inliers must be at least 2")
    if not (0.0 < args.ransac_min_inlier_ratio <= 1.0):
        raise ValueError("--ransac-min-inlier-ratio must be in (0, 1]")
    if not (0.0 <= args.max_ransac_skip_ratio <= 1.0):
        raise ValueError("--max-ransac-skip-ratio must be in [0, 1]")

    scans, ransac_rows = load_scans_with_ransac(
        args.dataset_dir,
        use_profile_ransac=not args.disable_profile_ransac,
        ransac_threshold_mm=args.ransac_threshold_mm,
        ransac_max_iterations=args.ransac_max_iterations,
        ransac_min_inliers=args.ransac_min_inliers,
        ransac_min_inlier_ratio=args.ransac_min_inlier_ratio,
        ransac_seed=args.ransac_seed,
        ransac_refine_iterations=args.ransac_refine_iterations,
        ransac_reject_policy=args.ransac_reject_policy,
    )
    total_scan_count = len(ransac_rows)
    rejected_rows = [
        row for row in ransac_rows if row.get("status") == "rejected"
    ]
    rejected_count = len(rejected_rows)
    skip_ratio = rejected_count / total_scan_count if total_scan_count else 0.0
    ransac_path = args.output.with_suffix(".profile_ransac.csv")
    save_profile_ransac_diagnostics(ransac_path, ransac_rows)
    print(
        "RANSAC summary: "
        f"{len(scans)}/{total_scan_count} scans accepted, "
        f"{rejected_count} rejected ({skip_ratio:.1%})"
    )
    print(f"saved {ransac_path}")

    if len(scans) < args.min_scans:
        raise RuntimeError(
            f"need at least {args.min_scans} accepted captures; found {len(scans)}"
        )
    if not args.disable_profile_ransac and skip_ratio > args.max_ransac_skip_ratio:
        raise RuntimeError(
            "too many scans rejected by RANSAC: "
            f"{rejected_count}/{total_scan_count} ({skip_ratio:.1%}) > "
            f"{args.max_ransac_skip_ratio:.1%}"
        )
    initial = load_transform(args.initial_transform)
    result = calibrate_single_plane(
        scans,
        T_init=initial,
        max_iter=args.max_iter,
        tol=args.tol,
        plane_offset_mode="joint",
        max_translation_offset_condition=args.max_condition,
    )
    final_rms = float(result.plane_rms_history[-1])
    accepted = bool(
        result.converged
        and np.isfinite(final_rms)
        and final_rms <= args.max_final_plane_rms_mm
    )
    plane_rms_history = [
        float(value) for value in result.plane_rms_history
    ]
    plot_path = args.output.with_suffix(".plane_rms.png")
    save_plane_rms_plot(plane_rms_history, plot_path)
    if monitor is not None:
        monitor.show_rms(plane_rms_history)

    diagnostics_path = args.output.with_suffix(".diagnostics.json")
    atomic_json(
        diagnostics_path,
        {
            "scan_count": len(scans),
            "total_capture_count": total_scan_count,
            "ransac_rejected_count": rejected_count,
            "ransac_skip_ratio": skip_ratio,
            "profile_ransac_diagnostics": str(ransac_path),
            "point_count": int(sum(scan.num_points for scan in scans)),
            "converged": bool(result.converged),
            "accepted": accepted,
            "iterations": int(result.iterations),
            "initial_plane_rms_mm": plane_rms_history[0],
            "final_plane_rms_mm": final_rms,
            "plane_rms_history_mm": plane_rms_history,
            "plane_rms_plot": str(plot_path),
            "rank_history": [int(value) for value in result.rank_history],
            "condition_history": [
                float(value) if np.isfinite(value) else None
                for value in result.cond_history
            ],
            "T_tcp_sensor": result.T_ef_s.tolist(),
        },
    )
    print(f"saved {plot_path}")
    if not accepted:
        raise RuntimeError(
            f"calibration rejected: converged={result.converged}, "
            f"final RMS={final_rms:.6f} mm; see {diagnostics_path}"
        )
    atomic_matrix(args.output, result.T_ef_s)
    print(f"saved {args.output}")
    print(np.array2string(result.T_ef_s, precision=8, suppress_small=True))
    return result.T_ef_s


def connect_hardware(args) -> tuple[RobotAdapter, LaserAdapter]:
    robot = RobotAdapter(args.robot_host, args.robot_port)
    laser = LaserAdapter(
        ip=args.laser_ip,
        control_port=args.laser_control_port,
        high_speed_port=args.laser_high_speed_port,
        batch_profiles=args.batch_profiles,
        aggregate=args.aggregate,
    )
    robot.connect()
    try:
        laser.connect()
    except BaseException:
        robot.close()
        raise
    return robot, laser


def run_capture(args) -> None:
    robot, laser = connect_hardware(args)
    try:
        capture_once(robot, laser, args)
    finally:
        laser.close()
        robot.close()


def run_session(args) -> None:
    if args.profile_stale_after_s <= 0:
        raise ValueError("--profile-stale-after-s must be positive")
    if args.profile_diagnostic_after_s <= 0:
        raise ValueError("--profile-diagnostic-after-s must be positive")
    robot, laser = connect_hardware(args)
    monitor = LiveMonitor(history_length=args.monitor_history)

    print("Live monitor started")
    print("Use the Capture, Calibrate (RANSAC), and Quit buttons")

    next_update = 0.0
    missing_profile_since: float | None = time.monotonic()
    missing_profile_diagnosed = False
    running = True
    try:
        while running:
            now = time.monotonic()
            if now >= next_update:
                try:
                    points, transform, profile_status = read_live_sample(
                        robot,
                        laser,
                        profile_stale_after_s=args.profile_stale_after_s,
                    )
                    monitor.update(points, transform, profile_status)
                    observed_at = time.monotonic()
                    if points is not None:
                        missing_profile_since = None
                        missing_profile_diagnosed = False
                    else:
                        if missing_profile_since is None:
                            missing_profile_since = observed_at
                        missing_for_s = observed_at - missing_profile_since
                        if (
                            missing_for_s >= args.profile_diagnostic_after_s
                            and not missing_profile_diagnosed
                        ):
                            try:
                                diagnostics = laser.read_diagnostics()
                                print_laser_diagnostics(
                                    diagnostics,
                                    missing_for_s=missing_for_s,
                                )
                            except Exception as diagnostic_exc:
                                print(
                                    "\nKEYENCE DIAGNOSTIC ERROR: "
                                    f"{type(diagnostic_exc).__name__}: "
                                    f"{diagnostic_exc}"
                                )
                            missing_profile_diagnosed = True
                except Exception as exc:
                    print(f"\nMONITOR ERROR: {exc}")
                next_update = time.monotonic() + args.monitor_interval_s

            monitor.process_events()
            if not monitor.window.isVisible():
                running = False
                continue
            capture_requested, calibration_requested, quit_requested = (
                monitor.take_requests()
            )
            if quit_requested:
                running = False
                continue
            try:
                if capture_requested:
                    monitor.set_busy(True, "Capturing...")
                    capture_once(robot, laser, args)
                    monitor.set_busy(False, "Capture saved")
                elif calibration_requested:
                    monitor.set_busy(True, "Calibrating with RANSAC...")
                    calibrate(args, monitor=monitor)
                    monitor.set_busy(False, "Calibration complete")
            except Exception as exc:
                print(f"ERROR: {exc}")
                monitor.set_busy(False, f"ERROR: {exc}")
            time.sleep(0.02)
    finally:
        monitor.close()
        laser.close()
        robot.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal manual 2D-laser hand-eye capture/calibration"
    )
    parser.add_argument(
        "command", nargs="?", choices=("session", "capture", "calibrate"), default="session"
    )
    parser.add_argument("--dataset-dir", type=Path, default=Path("runs/real/dataset"))
    parser.add_argument(
        "--initial-transform",
        type=Path,
        default=Path("real_laser_handeye/initial_T_tcp_sensor.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runs/real/T_tcp_sensor_calibrated.csv"),
    )
    parser.add_argument("--robot-host", default="169.254.186.20")
    parser.add_argument("--robot-port", type=int)
    parser.add_argument("--laser-ip", default="169.254.186.182")
    parser.add_argument("--laser-control-port", type=int, default=24691)
    parser.add_argument("--laser-high-speed-port", type=int, default=24692)
    parser.add_argument("--batch-profiles", type=int, default=5)
    parser.add_argument("--aggregate", choices=("median", "latest"), default="median")
    parser.add_argument("--timeout-s", type=float, default=3.0)
    parser.add_argument("--min-points", type=int, default=50)
    parser.add_argument("--max-abs-sensor-y-mm", type=float, default=0.1)
    parser.add_argument("--max-stationarity-translation-mm", type=float, default=0.2)
    parser.add_argument("--max-stationarity-rotation-deg", type=float, default=0.2)
    parser.add_argument("--min-scans", type=int, default=4)
    parser.add_argument("--max-iter", type=int, default=3000)
    parser.add_argument("--tol", type=float, default=1e-9)
    parser.add_argument("--max-condition", type=float, default=1e6)
    parser.add_argument("--max-final-plane-rms-mm", type=float, default=2.0)
    parser.add_argument(
        "--disable-profile-ransac",
        action="store_true",
        help="Disable per-profile line RANSAC (enabled by default)",
    )
    parser.add_argument(
        "--ransac-reject-policy",
        choices=("skip", "error"),
        default="skip",
    )
    parser.add_argument("--ransac-threshold-mm", type=float, default=0.15)
    parser.add_argument("--ransac-max-iterations", type=int, default=1000)
    parser.add_argument("--ransac-min-inliers", type=int, default=20)
    parser.add_argument("--ransac-min-inlier-ratio", type=float, default=0.65)
    parser.add_argument("--max-ransac-skip-ratio", type=float, default=0.3)
    parser.add_argument("--ransac-seed", type=int, default=1701)
    parser.add_argument("--ransac-refine-iterations", type=int, default=3)
    parser.add_argument(
        "--monitor-interval-s",
        type=float,
        default=0.1,
        help="Seconds between live profile/TCP plot updates",
    )
    parser.add_argument(
        "--monitor-history",
        type=int,
        default=200,
        help="Number of TCP position samples retained in the live plot",
    )
    parser.add_argument(
        "--profile-stale-after-s",
        type=float,
        default=1.0,
        help="Clear the displayed profile when no callback arrives for this long",
    )
    parser.add_argument(
        "--profile-diagnostic-after-s",
        type=float,
        default=3.0,
        help="Query Keyence diagnostics after no valid profile for this long",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "calibrate":
        calibrate(args)
    elif args.command == "capture":
        run_capture(args)
    else:
        run_session(args)


if __name__ == "__main__":
    main()
