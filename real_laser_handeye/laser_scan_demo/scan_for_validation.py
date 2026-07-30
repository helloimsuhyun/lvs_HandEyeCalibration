"""
PYTHONPATH=. python real_laser_handeye/laser_scan_demo/scan_for_validation.py \
  --handeye /home/choisuhyun/lvs_HandEyeCalibration/runs/real/real_initial/T_tcp_sensor_calibrate_initial_value.csv \
  --save-path runs/real/manual_world_scan.npz \
  --auto-save \
  --save-on-exit
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import importlib
import json
import math
import os
from pathlib import Path
import time
from typing import Any, Callable
import uuid
import warnings

import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtWidgets


DEFAULT_HANDEYE = Path("runs/real/real_initial/T_tcp_sensor_calibrated.csv")
DEFAULT_SAVE_PATH = Path("runs/real/real_initial/manual_world_scan.npz")


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
            "Run calibration first or pass --handeye."
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


def load_adapter(module_name: str, class_name: str) -> type:
    module = importlib.import_module(module_name)
    adapter_class = getattr(module, class_name, None)
    if adapter_class is None:
        raise ImportError(f"{module_name} has no class named {class_name}")
    return adapter_class


def utc_now_text() -> str:
    return datetime.now(timezone.utc).isoformat()


def wait_while_processing_events(
    duration_s: float,
    process_events: Callable[[], None],
) -> None:
    deadline = time.monotonic() + max(0.0, float(duration_s))
    while time.monotonic() < deadline:
        process_events()
        time.sleep(min(0.01, max(0.0, deadline - time.monotonic())))


@dataclass
class CaptureFrame:
    captured_at_utc: str
    profile_received_at: float
    callback_ids: np.ndarray
    points_sensor: np.ndarray
    points_base: np.ndarray
    T_base_tcp: np.ndarray
    T_base_sensor: np.ndarray
    requested_profile_count: int
    used_profile_count: int
    aggregate: str


class CaptureCollection:
    """Full-resolution manual captures retained for saving and reprocessing."""

    def __init__(self) -> None:
        self.frames: list[CaptureFrame] = []
        self._display_cache = np.empty((0, 3), dtype=np.float32)
        self._display_cache_limit: int | None = None
        self._display_cache_valid = True

    @property
    def capture_count(self) -> int:
        return len(self.frames)

    @property
    def point_count(self) -> int:
        return sum(len(frame.points_base) for frame in self.frames)

    def _invalidate_display_cache(self) -> None:
        self._display_cache_valid = False

    def add(self, frame: CaptureFrame) -> None:
        self.frames.append(frame)
        self._invalidate_display_cache()

    def undo(self) -> CaptureFrame | None:
        if not self.frames:
            return None
        removed = self.frames.pop()
        self._invalidate_display_cache()
        return removed

    def clear(self) -> None:
        self.frames.clear()
        self._invalidate_display_cache()

    def latest(self) -> CaptureFrame | None:
        return self.frames[-1] if self.frames else None

    def tcp_positions(self) -> np.ndarray:
        if not self.frames:
            return np.empty((0, 3), dtype=np.float32)
        return np.asarray(
            [frame.T_base_tcp[:3, 3] for frame in self.frames],
            dtype=np.float32,
        )

    def display_cloud(self, max_points: int) -> np.ndarray:
        if self._display_cache_valid and self._display_cache_limit == max_points:
            return self._display_cache

        if not self.frames:
            cloud = np.empty((0, 3), dtype=np.float32)
        else:
            cloud = np.concatenate(
                [frame.points_base for frame in self.frames], axis=0
            ).astype(np.float32, copy=False)
            if len(cloud) > max_points:
                stride = max(1, math.ceil(len(cloud) / max_points))
                cloud = np.ascontiguousarray(cloud[::stride])

        self._display_cache = cloud
        self._display_cache_limit = max_points
        self._display_cache_valid = True
        return self._display_cache


class WorldCaptureViewer:
    """Manual capture UI and world-frame OpenGL visualization."""

    def __init__(self, *, point_size_px: float, average_profiles: int) -> None:
        pg.setConfigOptions(antialias=False)
        self.app = pg.mkQApp("Manual world profile capture")
        self.window = QtWidgets.QWidget()
        self.window.setWindowTitle("Manual laser profile capture in base/world frame")
        self.window.resize(1280, 850)

        self.capture_requested = False
        self.undo_requested = False
        self.clear_requested = False
        self.save_requested = False
        self.quit_requested = False

        root = QtWidgets.QVBoxLayout(self.window)

        self.status = QtWidgets.QLabel("Connecting...")
        self.status.setWordWrap(True)
        root.addWidget(self.status)

        controls = QtWidgets.QHBoxLayout()
        self.capture_button = QtWidgets.QPushButton(
            f"Capture ({average_profiles} profiles)"
        )
        self.undo_button = QtWidgets.QPushButton("Undo last")
        self.clear_button = QtWidgets.QPushButton("Clear")
        self.save_button = QtWidgets.QPushButton("Save")
        self.quit_button = QtWidgets.QPushButton("Quit")

        self.capture_button.setShortcut("Space")
        self.save_button.setShortcut("Ctrl+S")
        self.quit_button.setShortcut("Ctrl+Q")

        self.capture_button.clicked.connect(self._request_capture)
        self.undo_button.clicked.connect(self._request_undo)
        self.clear_button.clicked.connect(self._request_clear)
        self.save_button.clicked.connect(self._request_save)
        self.quit_button.clicked.connect(self._request_quit)

        for button in (
            self.capture_button,
            self.undo_button,
            self.clear_button,
            self.save_button,
            self.quit_button,
        ):
            controls.addWidget(button)
        controls.addStretch(1)
        root.addLayout(controls)

        self.help_text = QtWidgets.QLabel(
            "Robot stop → Capture → fresh profiles are averaged → "
            "T_base_tcp @ T_tcp_sensor transforms them into the world frame."
        )
        self.help_text.setWordWrap(True)
        root.addWidget(self.help_text)

        self.view = gl.GLViewWidget()
        self.view.setCameraPosition(distance=1200, elevation=25, azimuth=45)
        self.view.setBackgroundColor((12, 14, 18))
        root.addWidget(self.view, stretch=1)

        grid = gl.GLGridItem()
        grid.setSize(x=2000, y=2000)
        grid.setSpacing(x=100, y=100)
        self.view.addItem(grid)
        self._add_world_axes(length_mm=250.0)

        self.cloud_item = gl.GLScatterPlotItem(
            pos=np.empty((0, 3), dtype=np.float32),
            color=(0.1, 0.75, 1.0, 0.55),
            size=float(point_size_px),
            pxMode=True,
        )
        self.latest_profile_item = gl.GLScatterPlotItem(
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

        self.sensor_axis_items = [
            gl.GLLinePlotItem(
                pos=np.empty((0, 3), dtype=np.float32),
                color=color,
                width=3.0,
                antialias=False,
                mode="lines",
            )
            for color in ((1, 0, 0, 1), (0, 1, 0, 1), (0, 0.5, 1, 1))
        ]

        for item in (
            self.cloud_item,
            self.latest_profile_item,
            self.tcp_path_item,
            self.tcp_item,
            *self.sensor_axis_items,
        ):
            self.view.addItem(item)

        self.window.show()
        self.process_events()

    def _request_capture(self) -> None:
        self.capture_requested = True

    def _request_undo(self) -> None:
        self.undo_requested = True

    def _request_clear(self) -> None:
        self.clear_requested = True

    def _request_save(self) -> None:
        self.save_requested = True

    def _request_quit(self) -> None:
        self.quit_requested = True

    def take_capture_request(self) -> bool:
        requested, self.capture_requested = self.capture_requested, False
        return requested

    def take_undo_request(self) -> bool:
        requested, self.undo_requested = self.undo_requested, False
        return requested

    def take_clear_request(self) -> bool:
        requested, self.clear_requested = self.clear_requested, False
        return requested

    def take_save_request(self) -> bool:
        requested, self.save_requested = self.save_requested, False
        return requested

    def take_quit_request(self) -> bool:
        requested, self.quit_requested = self.quit_requested, False
        return requested

    def set_busy(self, busy: bool) -> None:
        self.capture_button.setEnabled(not busy)
        self.undo_button.setEnabled(not busy)
        self.clear_button.setEnabled(not busy)
        self.save_button.setEnabled(not busy)
        self.process_events()

    def set_status(self, message: str) -> None:
        self.status.setText(message)
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

    def update_scene(
        self,
        *,
        cloud: np.ndarray,
        latest_profile: np.ndarray,
        tcp_path: np.ndarray,
        T_base_tcp: np.ndarray | None,
        T_base_sensor: np.ndarray | None,
        sensor_axis_length_mm: float,
    ) -> None:
        self.cloud_item.setData(pos=np.asarray(cloud, dtype=np.float32))
        self.latest_profile_item.setData(
            pos=np.asarray(latest_profile, dtype=np.float32)
        )
        self.tcp_path_item.setData(pos=np.asarray(tcp_path, dtype=np.float32))

        if T_base_tcp is None:
            self.tcp_item.setData(pos=np.empty((0, 3), dtype=np.float32))
        else:
            self.tcp_item.setData(
                pos=np.asarray(T_base_tcp[:3, 3], dtype=np.float32).reshape(1, 3)
            )

        if T_base_sensor is None:
            for item in self.sensor_axis_items:
                item.setData(pos=np.empty((0, 3), dtype=np.float32))
        else:
            origin = T_base_sensor[:3, 3]
            rotation = T_base_sensor[:3, :3]
            for axis_index, item in enumerate(self.sensor_axis_items):
                endpoint = origin + rotation[:, axis_index] * sensor_axis_length_mm
                item.setData(
                    pos=np.asarray([origin, endpoint], dtype=np.float32)
                )

        self.process_events()

    def process_events(self) -> None:
        self.app.processEvents()

    def is_open(self) -> bool:
        return self.window.isVisible()

    def close(self) -> None:
        self.window.close()
        self.process_events()


def aggregate_profiles(
    profiles: list[np.ndarray],
    method: str,
) -> tuple[np.ndarray, int]:
    if not profiles:
        raise ValueError("no profiles were collected")

    shape_counts: dict[tuple[int, int], int] = {}
    for profile in profiles:
        shape_counts[profile.shape] = shape_counts.get(profile.shape, 0) + 1
    target_shape = max(shape_counts, key=shape_counts.get)
    usable = [profile for profile in profiles if profile.shape == target_shape]
    if not usable:
        raise ValueError("collected profiles have no common shape")

    if method == "latest":
        result = np.asarray(usable[-1], dtype=float)
    else:
        stack = np.stack(usable, axis=0).astype(float, copy=False)
        stack[~np.isfinite(stack)] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            if method == "mean":
                result = np.nanmean(stack, axis=0)
            elif method == "median":
                result = np.nanmedian(stack, axis=0)
            else:
                raise ValueError(f"unsupported profile aggregate: {method}")

    valid = np.all(np.isfinite(result), axis=1)
    result = np.ascontiguousarray(result[valid], dtype=np.float64)
    if len(result) == 0:
        raise ValueError("the aggregated profile has no finite 3D points")
    return result, len(usable)


def read_stable_tcp_pose(
    robot: Any,
    *,
    stability_window_s: float,
    max_translation_mm: float,
    max_rotation_deg: float,
    process_events: Callable[[], None],
) -> np.ndarray:
    first = validate_transform(robot.read_T_base_tcp(), "T_base_tcp before capture")
    wait_while_processing_events(stability_window_s, process_events)
    second = validate_transform(robot.read_T_base_tcp(), "T_base_tcp at capture")

    translation = float(np.linalg.norm(second[:3, 3] - first[:3, 3]))
    rotation = rotation_distance_deg(first[:3, :3], second[:3, :3])
    if translation > max_translation_mm or rotation > max_rotation_deg:
        raise RuntimeError(
            "robot is not stationary: "
            f"TCP moved {translation:.4f} mm and {rotation:.4f} deg during "
            f"the {stability_window_s:.3f} s stability window"
        )
    return second


def collect_fresh_profiles(
    laser: Any,
    *,
    count: int,
    timeout_s: float,
    max_age_s: float,
    process_events: Callable[[], None],
    progress: Callable[[int, int], None],
) -> tuple[list[np.ndarray], np.ndarray, float]:
    baseline = laser.read_latest_profile_sample(max_age_s=max_age_s)
    last_callback_id = None if baseline is None else int(baseline[0])

    profiles: list[np.ndarray] = []
    callback_ids: list[int] = []
    latest_received_at = math.nan
    deadline = time.monotonic() + timeout_s

    while len(profiles) < count:
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"received {len(profiles)}/{count} fresh profiles within "
                f"{timeout_s:.1f} s"
            )

        sample = laser.read_latest_profile_sample(max_age_s=max_age_s)
        if sample is not None:
            callback_id, profile_received_at, points_sensor = sample
            callback_id = int(callback_id)
            if callback_id != last_callback_id:
                array = np.asarray(points_sensor, dtype=float)
                if array.ndim == 2 and array.shape[1] == 3 and len(array) > 0:
                    profiles.append(array.copy())
                    callback_ids.append(callback_id)
                    latest_received_at = float(profile_received_at)
                    progress(len(profiles), count)
                last_callback_id = callback_id

        process_events()
        time.sleep(0.002)

    return profiles, np.asarray(callback_ids, dtype=np.int64), latest_received_at


def capture_once(
    *,
    robot: Any,
    laser: Any,
    T_tcp_sensor: np.ndarray,
    args: argparse.Namespace,
    viewer: WorldCaptureViewer,
) -> CaptureFrame:
    viewer.set_status("Checking that the robot is stationary...")
    T_base_tcp = read_stable_tcp_pose(
        robot,
        stability_window_s=args.robot_stability_window_s,
        max_translation_mm=args.max_capture_translation_mm,
        max_rotation_deg=args.max_capture_rotation_deg,
        process_events=viewer.process_events,
    )

    def update_progress(received: int, requested: int) -> None:
        viewer.set_status(
            f"Capturing fresh profiles: {received}/{requested} "
            f"({args.capture_aggregate})"
        )

    profiles, callback_ids, profile_received_at = collect_fresh_profiles(
        laser,
        count=args.average_profiles,
        timeout_s=args.capture_timeout_s,
        max_age_s=args.profile_stale_after_s,
        process_events=viewer.process_events,
        progress=update_progress,
    )

    viewer.set_status("Checking that the robot remained stationary...")
    T_base_tcp_after = validate_transform(
        robot.read_T_base_tcp(), "T_base_tcp after capture"
    )
    translation = float(
        np.linalg.norm(T_base_tcp_after[:3, 3] - T_base_tcp[:3, 3])
    )
    rotation = rotation_distance_deg(
        T_base_tcp[:3, :3], T_base_tcp_after[:3, :3]
    )
    if (
        translation > args.max_capture_translation_mm
        or rotation > args.max_capture_rotation_deg
    ):
        raise RuntimeError(
            "capture rejected because the robot moved while profiles were acquired: "
            f"{translation:.4f} mm, {rotation:.4f} deg"
        )

    points_sensor, used_profile_count = aggregate_profiles(
        profiles, args.capture_aggregate
    )
    T_base_sensor = T_base_tcp @ T_tcp_sensor
    points_base = transform_points(T_base_sensor, points_sensor)

    return CaptureFrame(
        captured_at_utc=utc_now_text(),
        profile_received_at=profile_received_at,
        callback_ids=callback_ids,
        points_sensor=np.asarray(points_sensor, dtype=np.float32),
        points_base=np.asarray(points_base, dtype=np.float32),
        T_base_tcp=np.asarray(T_base_tcp, dtype=float),
        T_base_sensor=np.asarray(T_base_sensor, dtype=float),
        requested_profile_count=args.average_profiles,
        used_profile_count=used_profile_count,
        aggregate=args.capture_aggregate,
    )


def save_capture_collection(
    path: Path,
    *,
    collection: CaptureCollection,
    T_tcp_sensor: np.ndarray,
    handeye_path: Path,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")

    metadata = {
        "format": "manual_world_profile_capture",
        "format_version": 1,
        "saved_at_utc": utc_now_text(),
        "handeye_path": str(handeye_path),
        "capture_count": collection.capture_count,
        "point_count": collection.point_count,
        "average_profiles": args.average_profiles,
        "capture_aggregate": args.capture_aggregate,
        "coordinate_units": "mm",
        "transform_convention": (
            "points_base = T_base_tcp @ T_tcp_sensor @ points_sensor"
        ),
    }

    payload: dict[str, np.ndarray] = {
        "metadata_json": np.array(json.dumps(metadata, ensure_ascii=False)),
        "T_tcp_sensor": np.asarray(T_tcp_sensor, dtype=float),
        "capture_count": np.array(collection.capture_count, dtype=np.int64),
    }

    if collection.frames:
        payload["T_base_tcp"] = np.stack(
            [frame.T_base_tcp for frame in collection.frames], axis=0
        )
        payload["T_base_sensor"] = np.stack(
            [frame.T_base_sensor for frame in collection.frames], axis=0
        )
        payload["tcp_positions_base"] = collection.tcp_positions()
        payload["captured_at_utc"] = np.asarray(
            [frame.captured_at_utc for frame in collection.frames], dtype=str
        )
        payload["profile_received_at"] = np.asarray(
            [frame.profile_received_at for frame in collection.frames], dtype=float
        )
        payload["requested_profile_count"] = np.asarray(
            [frame.requested_profile_count for frame in collection.frames],
            dtype=np.int64,
        )
        payload["used_profile_count"] = np.asarray(
            [frame.used_profile_count for frame in collection.frames],
            dtype=np.int64,
        )
        payload["aggregate"] = np.asarray(
            [frame.aggregate for frame in collection.frames], dtype=str
        )
        payload["points_base_merged"] = np.concatenate(
            [frame.points_base for frame in collection.frames], axis=0
        ).astype(np.float32)

        for index, frame in enumerate(collection.frames):
            prefix = f"capture_{index:04d}"
            payload[f"{prefix}_points_sensor"] = frame.points_sensor
            payload[f"{prefix}_points_base"] = frame.points_base
            payload[f"{prefix}_callback_ids"] = frame.callback_ids

    try:
        with temporary.open("xb") as stream:
            np.savez_compressed(stream, **payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


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


def scene_arrays(
    collection: CaptureCollection,
    *,
    display_max_points: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    latest = collection.latest()
    return (
        collection.display_cloud(display_max_points),
        (
            np.empty((0, 3), dtype=np.float32)
            if latest is None
            else latest.points_base
        ),
        collection.tcp_positions(),
        None if latest is None else latest.T_base_tcp,
        None if latest is None else latest.T_base_sensor,
    )


def run(args: argparse.Namespace) -> None:
    if args.average_profiles <= 0:
        raise ValueError("--average-profiles must be positive")
    if args.capture_timeout_s <= 0 or args.profile_stale_after_s <= 0:
        raise ValueError("capture timeout values must be positive")
    if args.robot_stability_window_s < 0:
        raise ValueError("--robot-stability-window-s cannot be negative")
    if args.max_capture_translation_mm < 0 or args.max_capture_rotation_deg < 0:
        raise ValueError("capture motion thresholds cannot be negative")
    if args.status_rate_hz <= 0 or args.render_rate_hz <= 0:
        raise ValueError("status and render rates must be positive")
    if args.display_max_points <= 0 or args.point_size_px <= 0:
        raise ValueError("display limits and point size must be positive")
    if args.sensor_axis_length_mm <= 0:
        raise ValueError("--sensor-axis-length-mm must be positive")

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

    collection = CaptureCollection()
    viewer: WorldCaptureViewer | None = None
    latest_message = "Connected. Stop the robot and press Capture."
    latest_live_tcp: np.ndarray | None = None
    next_status_update = 0.0
    next_render = 0.0

    print(f"hand-eye: {args.handeye}")
    print(f"save path: {args.save_path}")
    print(
        f"capture={args.average_profiles} fresh profile(s), "
        f"aggregate={args.capture_aggregate}, batch_profiles={args.batch_profiles}"
    )

    try:
        viewer = WorldCaptureViewer(
            point_size_px=args.point_size_px,
            average_profiles=args.average_profiles,
        )
        running = True

        while running and viewer.is_open():
            viewer.process_events()

            if viewer.take_quit_request():
                running = False
                continue

            if viewer.take_capture_request():
                viewer.set_busy(True)
                try:
                    frame = capture_once(
                        robot=robot,
                        laser=laser,
                        T_tcp_sensor=T_tcp_sensor,
                        args=args,
                        viewer=viewer,
                    )
                    collection.add(frame)
                    latest_live_tcp = frame.T_base_tcp
                    latest_message = (
                        f"Capture {collection.capture_count} complete: "
                        f"{len(frame.points_base)} points from "
                        f"{frame.used_profile_count} profile(s)."
                    )
                    print(latest_message)
                    if args.auto_save:
                        save_capture_collection(
                            args.save_path,
                            collection=collection,
                            T_tcp_sensor=T_tcp_sensor,
                            handeye_path=args.handeye,
                            args=args,
                        )
                        latest_message += f" Auto-saved to {args.save_path}."
                except Exception as exc:
                    latest_message = f"Capture failed: {type(exc).__name__}: {exc}"
                    print(latest_message)
                    if isinstance(exc, TimeoutError):
                        try:
                            diagnostic = laser.read_diagnostics()
                            print(f"KEYENCE DIAGNOSTICS: {diagnostic_summary(diagnostic)}")
                        except Exception as diagnostic_exc:
                            print(
                                "KEYENCE DIAGNOSTIC ERROR: "
                                f"{type(diagnostic_exc).__name__}: {diagnostic_exc}"
                            )
                finally:
                    viewer.set_busy(False)

            if viewer.take_undo_request():
                removed = collection.undo()
                if removed is None:
                    latest_message = "Nothing to undo."
                else:
                    latest_message = (
                        f"Removed the last capture. "
                        f"Remaining captures={collection.capture_count}."
                    )

            if viewer.take_clear_request():
                collection.clear()
                latest_message = "All captures cleared."

            if viewer.take_save_request():
                try:
                    save_capture_collection(
                        args.save_path,
                        collection=collection,
                        T_tcp_sensor=T_tcp_sensor,
                        handeye_path=args.handeye,
                        args=args,
                    )
                    latest_message = (
                        f"Saved {collection.capture_count} capture(s), "
                        f"{collection.point_count} points to {args.save_path}."
                    )
                    print(latest_message)
                except Exception as exc:
                    latest_message = f"Save failed: {type(exc).__name__}: {exc}"
                    print(latest_message)

            now = time.monotonic()
            if now >= next_status_update:
                try:
                    latest_live_tcp = validate_transform(
                        robot.read_T_base_tcp(), "live T_base_tcp"
                    )
                    xyz = latest_live_tcp[:3, 3]
                    rpy = rotation_to_rpy_deg(latest_live_tcp[:3, :3])
                    tcp_text = (
                        f"TCP xyz=[{xyz[0]:.1f}, {xyz[1]:.1f}, {xyz[2]:.1f}] mm, "
                        f"rpy=[{rpy[0]:.1f}, {rpy[1]:.1f}, {rpy[2]:.1f}] deg"
                    )
                except Exception as exc:
                    tcp_text = f"TCP unavailable: {type(exc).__name__}: {exc}"

                viewer.set_status(
                    f"{latest_message} | {tcp_text} | "
                    f"captures={collection.capture_count}, "
                    f"stored points={collection.point_count}"
                )
                next_status_update = now + 1.0 / args.status_rate_hz

            now = time.monotonic()
            if now >= next_render:
                cloud, latest_profile, tcp_path, last_tcp, last_sensor = scene_arrays(
                    collection,
                    display_max_points=args.display_max_points,
                )
                viewer.update_scene(
                    cloud=cloud,
                    latest_profile=latest_profile,
                    tcp_path=tcp_path,
                    T_base_tcp=(
                        last_tcp if last_tcp is not None else latest_live_tcp
                    ),
                    T_base_sensor=last_sensor,
                    sensor_axis_length_mm=args.sensor_axis_length_mm,
                )
                next_render = now + 1.0 / args.render_rate_hz

            time.sleep(0.005)

    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        if args.save_on_exit and collection.capture_count:
            try:
                save_capture_collection(
                    args.save_path,
                    collection=collection,
                    T_tcp_sensor=T_tcp_sensor,
                    handeye_path=args.handeye,
                    args=args,
                )
                print(
                    f"saved on exit: {args.save_path} "
                    f"({collection.capture_count} captures, "
                    f"{collection.point_count} points)"
                )
            except Exception as exc:
                print(f"save-on-exit failed: {type(exc).__name__}: {exc}")
        if viewer is not None:
            viewer.close()
        laser.close()
        robot.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Manual stop-and-capture Keyence profiles transformed into the "
            "robot base/world frame"
        )
    )
    parser.add_argument("--handeye", type=Path, default=DEFAULT_HANDEYE)
    parser.add_argument("--robot-host", default="192.168.0.10")
    parser.add_argument("--robot-port", type=int)
    parser.add_argument("--laser-ip", default="192.168.1.1")
    parser.add_argument("--laser-control-port", type=int, default=24691)
    parser.add_argument("--laser-high-speed-port", type=int, default=24692)
    parser.add_argument("--device-id", type=int, default=0)

    parser.add_argument(
        "--batch-profiles",
        type=int,
        default=1,
        help=(
            "Profiles per laser callback. Keep 1 for manual precision capture "
            "when the adapter returns only the latest profile from each batch."
        ),
    )
    parser.add_argument("--average-profiles", type=int, default=10)
    parser.add_argument(
        "--capture-aggregate",
        choices=("mean", "median", "latest"),
        default="mean",
    )
    parser.add_argument("--capture-timeout-s", type=float, default=5.0)
    parser.add_argument("--profile-stale-after-s", type=float, default=0.5)

    parser.add_argument("--robot-stability-window-s", type=float, default=0.20)
    parser.add_argument("--max-capture-translation-mm", type=float, default=0.10)
    parser.add_argument("--max-capture-rotation-deg", type=float, default=0.05)

    parser.add_argument("--status-rate-hz", type=float, default=5.0)
    parser.add_argument("--render-rate-hz", type=float, default=10.0)
    parser.add_argument("--display-max-points", type=int, default=200_000)
    parser.add_argument("--point-size-px", type=float, default=2.0)
    parser.add_argument("--sensor-axis-length-mm", type=float, default=50.0)

    parser.add_argument("--save-path", type=Path, default=DEFAULT_SAVE_PATH)
    parser.add_argument("--auto-save", action="store_true")
    parser.add_argument("--save-on-exit", action="store_true")

    parser.add_argument(
        "--robot-adapter-module",
        default="real_laser_handeye.robot_adapter",
        help="Module containing RobotAdapter",
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