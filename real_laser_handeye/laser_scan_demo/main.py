from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
import importlib
import json
import math
import os
from pathlib import Path
import select
import sys
import time
from typing import Any
import uuid

import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtWidgets


DEFAULT_HANDEYE = Path("runs/real/real_initial/T_tcp_sensor_calibrated.csv")
DEFAULT_SAVE_PATH = Path("runs/real_initial/live_scan.npz")


def validate_transform(value: np.ndarray, name: str) -> np.ndarray:
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


def load_transform(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(
            f"calibrated hand-eye matrix not found: {path}. "
            "Run real_laser_handeye.main calibrate first or pass --handeye."
        )
    if path.suffix.lower() == ".json":
        value: Any = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict):
            value = value["T_tcp_sensor"]
        return validate_transform(np.asarray(value, dtype=float), str(path))
    try:
        value = np.loadtxt(path, delimiter=",")
    except ValueError:
        value = np.loadtxt(path)
    return validate_transform(value, str(path))


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    return points @ transform[:3, :3].T + transform[:3, 3]


def rotation_distance_deg(first: np.ndarray, second: np.ndarray) -> float:
    relative = first.T @ second
    cosine = np.clip((float(np.trace(relative)) - 1.0) * 0.5, -1.0, 1.0)
    return math.degrees(math.acos(float(cosine)))


def rotation_to_rpy_deg(rotation: np.ndarray) -> np.ndarray:
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


@dataclass
class ScanFrame:
    captured_at: float
    points_base: np.ndarray


class RollingScanBuffer:
    """Bounded in-memory world-frame point cloud."""

    def __init__(self, *, history_seconds: float, max_points: int) -> None:
        if history_seconds <= 0:
            raise ValueError("history_seconds must be positive")
        if max_points <= 0:
            raise ValueError("max_points must be positive")
        self.history_seconds = float(history_seconds)
        self.max_points = int(max_points)
        self.frames: deque[ScanFrame] = deque()
        self.point_count = 0
        self.scan_count = 0

    def add(self, captured_at: float, points_base: np.ndarray) -> None:
        points = np.asarray(points_base, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points_base must have shape (N, 3)")
        points = np.ascontiguousarray(points[np.all(np.isfinite(points), axis=1)])
        if len(points) == 0:
            return
        self.frames.append(ScanFrame(float(captured_at), points))
        self.point_count += len(points)
        self.scan_count += 1
        self.evict(float(captured_at))

    def evict(self, now: float) -> None:
        oldest_allowed = float(now) - self.history_seconds
        while self.frames and self.frames[0].captured_at < oldest_allowed:
            self.point_count -= len(self.frames.popleft().points_base)
        while self.frames and self.point_count > self.max_points:
            self.point_count -= len(self.frames.popleft().points_base)

    def points(self) -> np.ndarray:
        if not self.frames:
            return np.empty((0, 3), dtype=np.float32)
        return np.concatenate(
            [frame.points_base for frame in self.frames], axis=0
        )

    def clear(self) -> None:
        self.frames.clear()
        self.point_count = 0
        self.scan_count = 0


class WorldScanViewer:
    """PyQtGraph/OpenGL view of the world cloud and robot TCP."""

    def __init__(self, *, point_size_px: float) -> None:
        pg.setConfigOptions(antialias=False)
        self.app = pg.mkQApp("World laser scan")
        self.window = QtWidgets.QWidget()
        self.window.setWindowTitle("Laser scan in robot base/world frame")
        self.window.resize(1280, 800)
        layout = QtWidgets.QVBoxLayout(self.window)
        self.status = QtWidgets.QLabel("Connecting...")
        layout.addWidget(self.status)

        self.view = gl.GLViewWidget()
        self.view.setCameraPosition(distance=1200, elevation=25, azimuth=45)
        self.view.setBackgroundColor((12, 14, 18))
        layout.addWidget(self.view, stretch=1)

        grid = gl.GLGridItem()
        grid.setSize(x=2000, y=2000)
        grid.setSpacing(x=100, y=100)
        self.view.addItem(grid)
        self._add_world_axes(length_mm=250.0)

        self.cloud_item = gl.GLScatterPlotItem(
            pos=np.empty((0, 3), dtype=np.float32),
            color=(0.1, 0.75, 1.0, 0.65),
            size=float(point_size_px),
            pxMode=True,
        )
        self.current_profile_item = gl.GLScatterPlotItem(
            pos=np.empty((0, 3), dtype=np.float32),
            color=(1.0, 0.85, 0.15, 1.0),
            size=max(2.0, float(point_size_px) + 1.0),
            pxMode=True,
        )
        self.tcp_path_item = gl.GLLinePlotItem(
            pos=np.empty((0, 3), dtype=np.float32),
            color=(1.0, 0.25, 0.8, 0.85),
            width=2.0,
            antialias=False,
            mode="line_strip",
        )
        self.tcp_item = gl.GLScatterPlotItem(
            pos=np.empty((0, 3), dtype=np.float32),
            color=(1.0, 0.15, 0.15, 1.0),
            size=10.0,
            pxMode=True,
        )
        for item in (
            self.cloud_item,
            self.current_profile_item,
            self.tcp_path_item,
            self.tcp_item,
        ):
            self.view.addItem(item)

        self.window.show()
        self.process_events()

    def _add_world_axes(self, *, length_mm: float) -> None:
        origin = np.zeros(3, dtype=np.float32)
        axes = (
            (np.array([length_mm, 0.0, 0.0], dtype=np.float32), (1, 0, 0, 1)),
            (np.array([0.0, length_mm, 0.0], dtype=np.float32), (0, 1, 0, 1)),
            (np.array([0.0, 0.0, length_mm], dtype=np.float32), (0, 0.5, 1, 1)),
        )
        for endpoint, color in axes:
            self.view.addItem(
                gl.GLLinePlotItem(
                    pos=np.vstack([origin, endpoint]),
                    color=color,
                    width=3.0,
                    antialias=False,
                    mode="lines",
                )
            )

    def update(
        self,
        *,
        cloud: np.ndarray,
        current_profile: np.ndarray,
        tcp_path: np.ndarray,
        T_base_tcp: np.ndarray | None,
        message: str,
    ) -> None:
        self.cloud_item.setData(pos=np.asarray(cloud, dtype=np.float32))
        self.current_profile_item.setData(
            pos=np.asarray(current_profile, dtype=np.float32)
        )
        self.tcp_path_item.setData(pos=np.asarray(tcp_path, dtype=np.float32))
        if T_base_tcp is None:
            self.tcp_item.setData(pos=np.empty((0, 3), dtype=np.float32))
        else:
            self.tcp_item.setData(
                pos=np.asarray(T_base_tcp[:3, 3], dtype=np.float32).reshape(1, 3)
            )
        self.status.setText(message)
        self.process_events()

    def process_events(self) -> None:
        self.app.processEvents()

    def is_open(self) -> bool:
        return self.window.isVisible()

    def close(self) -> None:
        self.window.close()
        self.process_events()


def load_adapter(module_name: str, class_name: str) -> type:
    module = importlib.import_module(module_name)
    adapter_class = getattr(module, class_name, None)
    if adapter_class is None:
        raise ImportError(f"{module_name} has no class named {class_name}")
    return adapter_class


def should_store_scan(
    current: np.ndarray,
    previous: np.ndarray | None,
    *,
    elapsed_since_previous_s: float,
    min_translation_mm: float,
    min_rotation_deg: float,
    max_stationary_interval_s: float,
) -> bool:
    if previous is None:
        return True
    translation = float(np.linalg.norm(current[:3, 3] - previous[:3, 3]))
    rotation = rotation_distance_deg(previous[:3, :3], current[:3, :3])
    return bool(
        translation >= min_translation_mm
        or rotation >= min_rotation_deg
        or elapsed_since_previous_s >= max_stationary_interval_s
    )


def diagnostic_summary(diagnostics: dict[str, object]) -> str:
    errors = diagnostics.get("error_codes")
    if errors is None:
        error_text = "query failed"
    else:
        error_text = ",".join(f"0x{int(code):04X}" for code in errors) or "none"
    attention = diagnostics.get("attention_status")
    attention_text = (
        "query failed" if attention is None else f"0x{int(attention):04X}"
    )
    return (
        f"errors={error_text}, attention={attention_text}, "
        f"trigger={diagnostics.get('trigger_count')}, "
        f"callbacks={diagnostics.get('callback_count')}, "
        f"last_notify={diagnostics.get('last_notify')}"
    )


def save_snapshot(
    path: Path,
    *,
    cloud: np.ndarray,
    tcp_path: np.ndarray,
    T_tcp_sensor: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as stream:
            np.savez_compressed(
                stream,
                points_base=np.asarray(cloud, dtype=np.float32),
                tcp_positions_base=np.asarray(tcp_path, dtype=np.float32),
                T_tcp_sensor=np.asarray(T_tcp_sensor, dtype=float),
                saved_at=np.array(datetime.now(timezone.utc).isoformat()),
            )
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _advance_deadline(previous: float, interval_s: float, now: float) -> float:
    candidate = previous + interval_s
    return candidate if candidate > now else now + interval_s


def run(args: argparse.Namespace) -> None:
    if args.scan_rate_hz <= 0 or args.render_rate_hz <= 0:
        raise ValueError("scan and render rates must be positive")
    if args.batch_profiles <= 0 or args.profile_stride <= 0:
        raise ValueError("batch size and profile stride must be positive")
    if args.profile_stale_after_s <= 0 or args.diagnostic_after_s <= 0:
        raise ValueError("profile timeout values must be positive")
    if args.max_stationary_interval_s <= 0:
        raise ValueError("--max-stationary-interval-s must be positive")
    if args.min_tcp_translation_mm < 0 or args.min_tcp_rotation_deg < 0:
        raise ValueError("TCP movement thresholds cannot be negative")
    if args.max_tcp_path_points <= 0 or args.point_size_px <= 0:
        raise ValueError("TCP path length and point size must be positive")

    T_tcp_sensor = load_transform(args.handeye)
    RobotAdapter = load_adapter(args.robot_adapter_module, "RobotAdapter")
    LaserAdapter = load_adapter(args.laser_adapter_module, "LaserAdapter")
    robot = RobotAdapter(args.robot_host, args.robot_port)
    laser = LaserAdapter(
        ip=args.laser_ip,
        control_port=args.laser_control_port,
        high_speed_port=args.laser_high_speed_port,
        device_id=args.device_id,
        batch_profiles=args.batch_profiles,
        aggregate="latest",
    )

    robot.connect()
    try:
        laser.connect()
    except BaseException:
        robot.close()
        raise

    viewer: WorldScanViewer | None = None
    buffer = RollingScanBuffer(
        history_seconds=args.history_seconds,
        max_points=args.max_points,
    )
    tcp_path: deque[np.ndarray] = deque(maxlen=args.max_tcp_path_points)
    current_profile = np.empty((0, 3), dtype=np.float32)
    current_tcp: np.ndarray | None = None
    last_stored_tcp: np.ndarray | None = None
    last_stored_at = 0.0
    last_callback_id: int | None = None
    missing_profile_since: float | None = time.monotonic()
    missing_diagnosed = False
    latest_status = "Waiting for profile"
    started_at = time.monotonic()
    scan_interval_s = 1.0 / args.scan_rate_hz
    render_interval_s = 1.0 / args.render_rate_hz
    next_scan = 0.0
    next_render = 0.0

    print(f"hand-eye: {args.handeye}")
    print(
        f"scan={args.scan_rate_hz:g} Hz, render={args.render_rate_hz:g} Hz, "
        f"history={args.history_seconds:g} s, max_points={args.max_points}"
    )
    print("Type s + Enter: save, c + Enter: clear, q + Enter: quit")
    print("> ", end="", flush=True)

    try:
        viewer = WorldScanViewer(point_size_px=args.point_size_px)
        running = True
        terminal_input_open = True
        while running and viewer.is_open():
            now = time.monotonic()
            if now >= next_scan:
                try:
                    current_tcp = validate_transform(
                        robot.read_T_base_tcp(), "live T_base_tcp"
                    )
                    tcp_path.append(current_tcp[:3, 3].astype(np.float32))
                except Exception as exc:
                    current_tcp = None
                    latest_status = f"TCP error: {type(exc).__name__}: {exc}"

                profile_sample = None
                try:
                    profile_sample = laser.read_latest_profile_sample(
                        max_age_s=args.profile_stale_after_s
                    )
                    if profile_sample is None:
                        latest_status = "No fresh laser profile"
                except Exception as exc:
                    latest_status = (
                        f"Profile error: {type(exc).__name__}: {exc}"
                    )

                observed_at = time.monotonic()
                if profile_sample is None:
                    current_profile = np.empty((0, 3), dtype=np.float32)
                    if missing_profile_since is None:
                        missing_profile_since = observed_at
                    missing_for_s = observed_at - missing_profile_since
                    if (
                        missing_for_s >= args.diagnostic_after_s
                        and not missing_diagnosed
                    ):
                        try:
                            diagnostic = laser.read_diagnostics()
                            print(
                                f"\nKEYENCE DIAGNOSTICS after {missing_for_s:.1f} s: "
                                f"{diagnostic_summary(diagnostic)}"
                            )
                        except Exception as exc:
                            print(
                                "\nKEYENCE DIAGNOSTIC ERROR: "
                                f"{type(exc).__name__}: {exc}"
                            )
                        print("> ", end="", flush=True)
                        missing_diagnosed = True
                else:
                    callback_id, profile_received_at, points_sensor = profile_sample
                    missing_profile_since = None
                    missing_diagnosed = False
                    if callback_id != last_callback_id:
                        last_callback_id = callback_id
                        if current_tcp is not None:
                            elapsed_since_stored = (
                                math.inf
                                if last_stored_tcp is None
                                else observed_at - last_stored_at
                            )
                            if should_store_scan(
                                current_tcp,
                                last_stored_tcp,
                                elapsed_since_previous_s=elapsed_since_stored,
                                min_translation_mm=args.min_tcp_translation_mm,
                                min_rotation_deg=args.min_tcp_rotation_deg,
                                max_stationary_interval_s=(
                                    args.max_stationary_interval_s
                                ),
                            ):
                                points_sensor = np.asarray(
                                    points_sensor, dtype=float
                                )[:: args.profile_stride]
                                T_base_sensor = current_tcp @ T_tcp_sensor
                                current_profile = transform_points(
                                    T_base_sensor, points_sensor
                                ).astype(np.float32)
                                buffer.add(profile_received_at, current_profile)
                                last_stored_tcp = current_tcp.copy()
                                last_stored_at = observed_at
                                latest_status = "Scanning"
                            else:
                                latest_status = "Profile skipped: TCP nearly stationary"
                    elif current_tcp is not None:
                        latest_status = "Waiting for a new profile batch"

                buffer.evict(observed_at)
                next_scan = _advance_deadline(next_scan, scan_interval_s, observed_at)

            now = time.monotonic()
            if now >= next_render:
                path_array = (
                    np.asarray(tcp_path, dtype=np.float32).reshape(-1, 3)
                    if tcp_path
                    else np.empty((0, 3), dtype=np.float32)
                )
                tcp_text = "TCP unavailable"
                if current_tcp is not None:
                    xyz = current_tcp[:3, 3]
                    rpy = rotation_to_rpy_deg(current_tcp[:3, :3])
                    tcp_text = (
                        f"TCP xyz=[{xyz[0]:.1f}, {xyz[1]:.1f}, {xyz[2]:.1f}] mm, "
                        f"rpy=[{rpy[0]:.1f}, {rpy[1]:.1f}, {rpy[2]:.1f}] deg"
                    )
                message = (
                    f"{latest_status} | {tcp_text} | "
                    f"stored scans={buffer.scan_count}, points={buffer.point_count}, "
                    f"elapsed={now - started_at:.1f} s"
                )
                viewer.update(
                    cloud=buffer.points(),
                    current_profile=current_profile,
                    tcp_path=path_array,
                    T_base_tcp=current_tcp,
                    message=message,
                )
                next_render = _advance_deadline(
                    next_render, render_interval_s, time.monotonic()
                )

            readable = []
            if terminal_input_open:
                readable, _, _ = select.select([sys.stdin], [], [], 0.0)
            if readable:
                line = sys.stdin.readline()
                if line == "":
                    terminal_input_open = False
                    viewer.process_events()
                    time.sleep(0.005)
                    continue
                command = line.strip().lower()
                if command in ("q", "quit", "exit"):
                    running = False
                elif command in ("c", "clear"):
                    buffer.clear()
                    tcp_path.clear()
                    current_profile = np.empty((0, 3), dtype=np.float32)
                    last_stored_tcp = None
                    print("scan buffer cleared")
                elif command in ("s", "save"):
                    save_snapshot(
                        args.save_path,
                        cloud=buffer.points(),
                        tcp_path=np.asarray(tcp_path, dtype=np.float32).reshape(-1, 3),
                        T_tcp_sensor=T_tcp_sensor,
                    )
                    print(
                        f"saved {args.save_path} "
                        f"({buffer.point_count} world points)"
                    )
                elif command:
                    print("unknown command: s, c, q")
                if running:
                    print("> ", end="", flush=True)

            viewer.process_events()
            time.sleep(0.005)
    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        if args.save_on_exit and buffer.point_count:
            save_snapshot(
                args.save_path,
                cloud=buffer.points(),
                tcp_path=np.asarray(tcp_path, dtype=np.float32).reshape(-1, 3),
                T_tcp_sensor=T_tcp_sensor,
            )
            print(f"saved {args.save_path} ({buffer.point_count} world points)")
        if viewer is not None:
            viewer.close()
        laser.close()
        robot.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Live Keyence profiles transformed into robot base/world frame"
    )
    parser.add_argument("--handeye", type=Path, default=DEFAULT_HANDEYE)
    parser.add_argument("--robot-host", default="192.168.0.10")
    parser.add_argument("--robot-port", type=int)
    parser.add_argument("--laser-ip", default="192.168.1.1")
    parser.add_argument("--laser-control-port", type=int, default=24691)
    parser.add_argument("--laser-high-speed-port", type=int, default=24692)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--batch-profiles", type=int, default=20)
    parser.add_argument("--scan-rate-hz", type=float, default=10.0)
    parser.add_argument("--render-rate-hz", type=float, default=5.0)
    parser.add_argument("--profile-stride", type=int, default=4)
    parser.add_argument("--profile-stale-after-s", type=float, default=0.5)
    parser.add_argument("--diagnostic-after-s", type=float, default=3.0)
    parser.add_argument("--min-tcp-translation-mm", type=float, default=0.5)
    parser.add_argument("--min-tcp-rotation-deg", type=float, default=0.2)
    parser.add_argument("--max-stationary-interval-s", type=float, default=1.0)
    parser.add_argument("--history-seconds", type=float, default=30.0)
    parser.add_argument("--max-points", type=int, default=200_000)
    parser.add_argument("--max-tcp-path-points", type=int, default=2_000)
    parser.add_argument("--point-size-px", type=float, default=2.0)
    parser.add_argument("--save-path", type=Path, default=DEFAULT_SAVE_PATH)
    parser.add_argument("--save-on-exit", action="store_true")
    parser.add_argument(
        "--robot-adapter-module",
        default="real_laser_handeye.robot_adapter",
        help="Module containing RobotAdapter (default is the real RB adapter)",
    )
    parser.add_argument(
        "--laser-adapter-module",
        default="real_laser_handeye.laser_adapter",
        help="Module containing LaserAdapter",
    )
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
