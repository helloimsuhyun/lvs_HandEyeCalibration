from __future__ import annotations

"""
Two-point, multi-step translation scanner for an RB robot + Keyence profile sensor.

One scan step
-------------
1. Manually jog the robot to the first position and click "Teach first".
   - The first position's TCP orientation is stored as the fixed scan orientation.
2. Manually jog the robot to the second position and click "Teach second".
   - Only the second position's XYZ is used.
3. Click "Start scan".
   - The robot is already near the second taught position.
   - It first aligns to the first position's orientation at the second XYZ.
   - The second-to-first segment is divided into equally spaced waypoints.
   - At each waypoint the robot stops, settles, acquires several fresh profiles,
     aggregates them, transforms the result into the robot base frame, and stores it.
4. Click "New step" and repeat. "Save and finish" saves all steps together.

Notes
-----
- The scan direction is deliberately SECOND -> FIRST.
- The scan motion holds the FIRST taught TCP orientation.
- A blocking RobotAdapter.move_l() is executed in a worker thread so the main
  thread can continue refreshing the live profile and UI while each waypoint move runs.
- No scan data are stored while the robot is moving. Data are stored only after
  arrival, settling, and a stationary-pose verification.
- This file does not perform IK, collision checking, singularity checking, or
  workspace-model validation. Start with a very low speed and use the teach
  pendant / emergency stop as the primary safety mechanism.

Example
-------
PYTHONPATH=. python real_laser_handeye/laser_scan_demo/two_point_step_scan.py \
  --handeye runs/real/real_initial/T_tcp_sensor_calibrated.csv \
  --save-path runs/real/two_point_steps_scan.npz \
  --robot-host 192.168.0.10 \
  --laser-ip 192.168.1.1 \
  --scan-speed-mm-s 20 \
  --scan-accel-mm-s2 20 \
  --waypoint-spacing-mm 1.0 \
  --profiles-per-waypoint 10 \
  --auto-save
"""

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
import importlib
import json
import math
import os
from pathlib import Path
import threading
import time
from typing import Any, Callable
import uuid
import warnings

import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from pyqtgraph.Qt import QtCore, QtWidgets


DEFAULT_HANDEYE = Path("runs/real/real_initial/T_tcp_sensor_calibrated.csv")
DEFAULT_SAVE_PATH = Path("runs/real/two_point_steps_scan.npz")


# ---------------------------------------------------------------------------
# Geometry and loading
# ---------------------------------------------------------------------------


def utc_now_text() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_transform(value: np.ndarray, name: str) -> np.ndarray:
    T = np.asarray(value, dtype=float)
    if T.shape != (4, 4) or not np.all(np.isfinite(T)):
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    if not np.allclose(T[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError(f"{name} has an invalid homogeneous last row")
    R = T[:3, :3]
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-5):
        raise ValueError(f"{name} rotation is not orthonormal")
    if not np.isclose(np.linalg.det(R), 1.0, atol=1e-5):
        raise ValueError(f"{name} rotation determinant must be +1")
    return T.copy()


def load_transform(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(
            f"hand-eye matrix not found: {path}. Pass --handeye."
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


def transform_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    return points @ T[:3, :3].T + T[:3, 3]


def rotation_distance_deg(R1: np.ndarray, R2: np.ndarray) -> float:
    relative = R1.T @ R2
    cosine = np.clip((float(np.trace(relative)) - 1.0) * 0.5, -1.0, 1.0)
    return math.degrees(math.acos(float(cosine)))


def rotation_to_rpy_deg(R: np.ndarray) -> np.ndarray:
    sy = float(np.hypot(R[0, 0], R[1, 0]))
    if sy > 1e-9:
        rx = math.atan2(R[2, 1], R[2, 2])
        ry = math.atan2(-R[2, 0], sy)
        rz = math.atan2(R[1, 0], R[0, 0])
    else:
        rx = math.atan2(-R[1, 2], R[1, 1])
        ry = math.atan2(-R[2, 0], sy)
        rz = 0.0
    return np.degrees([rx, ry, rz])


def load_adapter(module_name: str, class_name: str) -> type:
    module = importlib.import_module(module_name)
    adapter_class = getattr(module, class_name, None)
    if adapter_class is None:
        raise ImportError(f"{module_name} has no class named {class_name}")
    return adapter_class


def pose_text(pose: np.ndarray | None) -> str:
    if pose is None:
        return "not taught"
    values = np.asarray(pose, dtype=float).reshape(6)
    return (
        f"xyz=[{values[0]:.2f}, {values[1]:.2f}, {values[2]:.2f}] mm, "
        f"rpy=[{values[3]:.2f}, {values[4]:.2f}, {values[5]:.2f}] deg"
    )


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class MotionProfile:
    step_index: int
    profile_index_in_step: int
    waypoint_fraction: float
    captured_at_utc: str
    profile_received_at: float
    callback_id: int
    callback_ids: np.ndarray
    points_sensor: np.ndarray
    points_base: np.ndarray
    T_base_tcp: np.ndarray
    T_base_sensor: np.ndarray
    requested_profile_count: int
    used_profile_count: int
    aggregate: str


@dataclass
class ScanStep:
    step_index: int
    first_taught_pose_vec: np.ndarray
    second_taught_pose_vec: np.ndarray
    scan_from_pose_vec: np.ndarray
    scan_to_pose_vec: np.ndarray
    started_at_utc: str
    completed_at_utc: str | None = None
    profiles: list[MotionProfile] = field(default_factory=list)
    move_error: str | None = None

    @property
    def point_count(self) -> int:
        return sum(len(profile.points_base) for profile in self.profiles)


class ScanSession:
    def __init__(self) -> None:
        self.steps: list[ScanStep] = []
        self._display_cache = np.empty((0, 3), dtype=np.float32)
        self._display_cache_limit: int | None = None
        self._display_cache_valid = False

    @property
    def profile_count(self) -> int:
        return sum(len(step.profiles) for step in self.steps)

    @property
    def point_count(self) -> int:
        return sum(step.point_count for step in self.steps)

    def invalidate(self) -> None:
        self._display_cache_valid = False

    def add_step(self, step: ScanStep) -> None:
        self.steps.append(step)
        self.invalidate()

    def remove_last_step(self) -> ScanStep | None:
        if not self.steps:
            return None
        step = self.steps.pop()
        self.invalidate()
        return step

    def display_cloud(self, max_points: int) -> np.ndarray:
        if self._display_cache_valid and self._display_cache_limit == max_points:
            return self._display_cache

        groups = [
            profile.points_base
            for step in self.steps
            for profile in step.profiles
            if len(profile.points_base)
        ]
        if not groups:
            cloud = np.empty((0, 3), dtype=np.float32)
        else:
            cloud = np.concatenate(groups, axis=0).astype(np.float32, copy=False)
            if len(cloud) > max_points:
                stride = max(1, math.ceil(len(cloud) / max_points))
                cloud = np.ascontiguousarray(cloud[::stride])

        self._display_cache = cloud
        self._display_cache_limit = max_points
        self._display_cache_valid = True
        return cloud

    def latest_profile(self) -> np.ndarray:
        for step in reversed(self.steps):
            if step.profiles:
                return step.profiles[-1].points_base
        return np.empty((0, 3), dtype=np.float32)

    def tcp_path(self) -> np.ndarray:
        poses = [
            profile.T_base_tcp[:3, 3]
            for step in self.steps
            for profile in step.profiles
        ]
        if not poses:
            return np.empty((0, 3), dtype=np.float32)
        return np.asarray(poses, dtype=np.float32)


# ---------------------------------------------------------------------------
# GUI
# ---------------------------------------------------------------------------


class TwoPointScanViewer:
    def __init__(self, *, point_size_px: float, display_profile_stride: int) -> None:
        pg.setConfigOptions(antialias=False)
        self.app = pg.mkQApp("Two-point translation scanner")
        self.window = QtWidgets.QWidget()
        self.window.setWindowTitle("Two-point multi-step laser translation scanner")
        self.window.resize(1450, 900)

        self.display_profile_stride = max(1, int(display_profile_stride))

        self.teach_first_requested = False
        self.teach_second_requested = False
        self.start_scan_requested = False
        self.new_step_requested = False
        self.undo_step_requested = False
        self.save_requested = False
        self.finish_requested = False
        self.abort_requested = False
        self.quit_requested = False

        root = QtWidgets.QVBoxLayout(self.window)

        self.status = QtWidgets.QLabel("Connecting...")
        self.status.setWordWrap(True)
        root.addWidget(self.status)

        self.first_label = QtWidgets.QLabel("First: not taught")
        self.second_label = QtWidgets.QLabel("Second: not taught")
        self.direction_label = QtWidgets.QLabel(
            "Stop-and-scan: second -> first, fixed orientation: first pose"
        )
        self.first_label.setWordWrap(True)
        self.second_label.setWordWrap(True)
        root.addWidget(self.first_label)
        root.addWidget(self.second_label)
        root.addWidget(self.direction_label)

        controls = QtWidgets.QHBoxLayout()
        self.teach_first_button = QtWidgets.QPushButton("1. Teach first")
        self.teach_second_button = QtWidgets.QPushButton("2. Teach second")
        self.start_scan_button = QtWidgets.QPushButton("3. Start stop-and-scan: second -> first")
        self.new_step_button = QtWidgets.QPushButton("New step")
        self.undo_step_button = QtWidgets.QPushButton("Undo last step")
        self.save_button = QtWidgets.QPushButton("Save")
        self.finish_button = QtWidgets.QPushButton("Save and finish")
        self.abort_button = QtWidgets.QPushButton("Abort move")
        self.quit_button = QtWidgets.QPushButton("Save and quit")

        self.teach_first_button.setShortcut("1")
        self.teach_second_button.setShortcut("2")
        self.start_scan_button.setShortcut("Space")
        self.save_button.setShortcut("Ctrl+S")
        self.quit_button.setShortcut("Ctrl+Q")

        self.teach_first_button.clicked.connect(self._teach_first)
        self.teach_second_button.clicked.connect(self._teach_second)
        self.start_scan_button.clicked.connect(self._start_scan)
        self.new_step_button.clicked.connect(self._new_step)
        self.undo_step_button.clicked.connect(self._undo_step)
        self.save_button.clicked.connect(self._save)
        self.finish_button.clicked.connect(self._finish)
        self.abort_button.clicked.connect(self._abort)
        self.quit_button.clicked.connect(self._quit)

        for button in (
            self.teach_first_button,
            self.teach_second_button,
            self.start_scan_button,
            self.new_step_button,
            self.undo_step_button,
            self.save_button,
            self.finish_button,
            self.abort_button,
            self.quit_button,
        ):
            controls.addWidget(button)
        root.addLayout(controls)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Horizontal)
        root.addWidget(splitter, stretch=1)

        # Lightweight live 2D profile plot for teaching and motion.
        profile_panel = QtWidgets.QWidget()
        profile_layout = QtWidgets.QVBoxLayout(profile_panel)
        profile_title = QtWidgets.QLabel(
            "Live sensor profile (display-downsampled only; saved data stay full resolution)"
        )
        profile_layout.addWidget(profile_title)
        self.profile_plot = pg.PlotWidget()
        self.profile_plot.setBackground((12, 14, 18))
        self.profile_plot.showGrid(x=True, y=True, alpha=0.25)
        self.profile_plot.setLabel("bottom", "Sensor X", units="mm")
        self.profile_plot.setLabel("left", "Sensor Z", units="mm")
        self.profile_curve = self.profile_plot.plot(
            [], [], pen=None, symbol="o", symbolSize=2
        )
        profile_layout.addWidget(self.profile_plot, stretch=1)
        splitter.addWidget(profile_panel)

        # Accumulated 3D base-frame cloud.
        cloud_panel = QtWidgets.QWidget()
        cloud_layout = QtWidgets.QVBoxLayout(cloud_panel)
        cloud_layout.addWidget(QtWidgets.QLabel("Accumulated world/base-frame scan"))
        self.view = gl.GLViewWidget()
        self.view.setBackgroundColor((12, 14, 18))
        self.view.setCameraPosition(distance=1000, elevation=25, azimuth=45)
        cloud_layout.addWidget(self.view, stretch=1)
        splitter.addWidget(cloud_panel)
        splitter.setSizes([520, 900])

        grid = gl.GLGridItem()
        grid.setSize(x=2000, y=2000)
        grid.setSpacing(x=100, y=100)
        self.view.addItem(grid)
        self._add_world_axes(250.0)

        self.cloud_item = gl.GLScatterPlotItem(
            pos=np.empty((0, 3), dtype=np.float32),
            color=(0.1, 0.72, 1.0, 0.55),
            size=float(point_size_px),
            pxMode=True,
        )
        self.latest_profile_item = gl.GLScatterPlotItem(
            pos=np.empty((0, 3), dtype=np.float32),
            color=(1.0, 0.85, 0.1, 1.0),
            size=max(2.0, point_size_px + 1.0),
            pxMode=True,
        )
        self.tcp_path_item = gl.GLLinePlotItem(
            pos=np.empty((0, 3), dtype=np.float32),
            color=(1.0, 0.25, 0.8, 0.9),
            width=2.0,
            antialias=False,
            mode="line_strip",
        )
        self.current_tcp_item = gl.GLScatterPlotItem(
            pos=np.empty((0, 3), dtype=np.float32),
            color=(1.0, 0.1, 0.1, 1.0),
            size=10.0,
            pxMode=True,
        )
        self.first_pose_item = gl.GLScatterPlotItem(
            pos=np.empty((0, 3), dtype=np.float32),
            color=(0.1, 1.0, 0.2, 1.0),
            size=12.0,
            pxMode=True,
        )
        self.second_pose_item = gl.GLScatterPlotItem(
            pos=np.empty((0, 3), dtype=np.float32),
            color=(1.0, 0.55, 0.1, 1.0),
            size=12.0,
            pxMode=True,
        )
        self.planned_line_item = gl.GLLinePlotItem(
            pos=np.empty((0, 3), dtype=np.float32),
            color=(0.2, 1.0, 0.4, 0.85),
            width=3.0,
            antialias=False,
            mode="lines",
        )

        for item in (
            self.cloud_item,
            self.latest_profile_item,
            self.tcp_path_item,
            self.current_tcp_item,
            self.first_pose_item,
            self.second_pose_item,
            self.planned_line_item,
        ):
            self.view.addItem(item)

        self.window.show()
        self.process_events()

    def _add_world_axes(self, length_mm: float) -> None:
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

    def _teach_first(self) -> None:
        self.teach_first_requested = True

    def _teach_second(self) -> None:
        self.teach_second_requested = True

    def _start_scan(self) -> None:
        self.start_scan_requested = True

    def _new_step(self) -> None:
        self.new_step_requested = True

    def _undo_step(self) -> None:
        self.undo_step_requested = True

    def _save(self) -> None:
        self.save_requested = True

    def _finish(self) -> None:
        self.finish_requested = True

    def _abort(self) -> None:
        self.abort_requested = True

    def _quit(self) -> None:
        self.quit_requested = True

    def _take(self, name: str) -> bool:
        value = bool(getattr(self, name))
        setattr(self, name, False)
        return value

    def take_teach_first(self) -> bool:
        return self._take("teach_first_requested")

    def take_teach_second(self) -> bool:
        return self._take("teach_second_requested")

    def take_start_scan(self) -> bool:
        return self._take("start_scan_requested")

    def take_new_step(self) -> bool:
        return self._take("new_step_requested")

    def take_undo_step(self) -> bool:
        return self._take("undo_step_requested")

    def take_save(self) -> bool:
        return self._take("save_requested")

    def take_finish(self) -> bool:
        return self._take("finish_requested")

    def take_abort(self) -> bool:
        return self._take("abort_requested")

    def take_quit(self) -> bool:
        return self._take("quit_requested")

    def set_busy(self, busy: bool) -> None:
        self.teach_first_button.setEnabled(not busy)
        self.teach_second_button.setEnabled(not busy)
        self.start_scan_button.setEnabled(not busy)
        self.new_step_button.setEnabled(not busy)
        self.undo_step_button.setEnabled(not busy)
        self.save_button.setEnabled(not busy)
        self.finish_button.setEnabled(not busy)
        self.abort_button.setEnabled(busy)
        self.process_events()

    def set_teaching(self, first: np.ndarray | None, second: np.ndarray | None) -> None:
        self.first_label.setText(f"First: {pose_text(first)}")
        self.second_label.setText(f"Second: {pose_text(second)}")
        self.first_pose_item.setData(
            pos=(
                np.empty((0, 3), dtype=np.float32)
                if first is None
                else np.asarray(first[:3], dtype=np.float32).reshape(1, 3)
            )
        )
        self.second_pose_item.setData(
            pos=(
                np.empty((0, 3), dtype=np.float32)
                if second is None
                else np.asarray(second[:3], dtype=np.float32).reshape(1, 3)
            )
        )
        if first is not None and second is not None:
            # Arrowless line: visually represents scan direction second -> first.
            self.planned_line_item.setData(
                pos=np.asarray([second[:3], first[:3]], dtype=np.float32)
            )
        else:
            self.planned_line_item.setData(pos=np.empty((0, 3), dtype=np.float32))
        self.process_events()

    def set_status(self, message: str) -> None:
        self.status.setText(message)
        self.process_events()

    def update_live_profile(self, points_sensor: np.ndarray | None) -> None:
        if points_sensor is None:
            self.profile_curve.setData([], [])
            return
        points = np.asarray(points_sensor, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
            self.profile_curve.setData([], [])
            return
        draw = points[:: self.display_profile_stride]
        valid = np.all(np.isfinite(draw), axis=1)
        draw = draw[valid]
        self.profile_curve.setData(draw[:, 0], draw[:, 2])

    def update_cloud(
        self,
        *,
        cloud: np.ndarray,
        latest_profile: np.ndarray,
        tcp_path: np.ndarray,
        current_T_base_tcp: np.ndarray | None,
    ) -> None:
        self.cloud_item.setData(pos=np.asarray(cloud, dtype=np.float32))
        self.latest_profile_item.setData(
            pos=np.asarray(latest_profile, dtype=np.float32)
        )
        self.tcp_path_item.setData(pos=np.asarray(tcp_path, dtype=np.float32))
        if current_T_base_tcp is None:
            self.current_tcp_item.setData(pos=np.empty((0, 3), dtype=np.float32))
        else:
            self.current_tcp_item.setData(
                pos=np.asarray(current_T_base_tcp[:3, 3], dtype=np.float32).reshape(1, 3)
            )

    def process_events(self) -> None:
        self.app.processEvents()

    def is_open(self) -> bool:
        return self.window.isVisible()

    def close(self) -> None:
        self.window.close()
        self.process_events()


# ---------------------------------------------------------------------------
# Motion and capture
# ---------------------------------------------------------------------------


class MoveWorker:
    def __init__(self, target: Callable[[], np.ndarray]) -> None:
        self.result: np.ndarray | None = None
        self.error: BaseException | None = None
        self.done = threading.Event()

        def wrapped() -> None:
            try:
                self.result = target()
            except BaseException as exc:  # preserve robot errors for main thread
                self.error = exc
            finally:
                self.done.set()

        self.thread = threading.Thread(target=wrapped, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def is_alive(self) -> bool:
        return self.thread.is_alive()

    def join(self, timeout: float | None = None) -> None:
        self.thread.join(timeout)



def read_latest_live_profile(laser: Any, max_age_s: float) -> tuple[int, float, np.ndarray] | None:
    sample = laser.read_latest_profile_sample(max_age_s=max_age_s)
    if sample is None:
        return None
    callback_id, profile_received_at, points_sensor = sample
    points = np.asarray(points_sensor, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) == 0:
        return None
    points = points[np.all(np.isfinite(points), axis=1)]
    if len(points) == 0:
        return None
    return int(callback_id), float(profile_received_at), np.ascontiguousarray(points)



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
        raise ValueError("aggregated profile has no finite 3D points")
    return result, len(usable)


def collect_fresh_profiles(
    *,
    laser: Any,
    count: int,
    timeout_s: float,
    args: argparse.Namespace,
    viewer: TwoPointScanViewer,
    status_prefix: str,
) -> tuple[list[np.ndarray], np.ndarray, float]:
    baseline = read_latest_live_profile(laser, args.profile_stale_after_s)
    last_callback_id = None if baseline is None else baseline[0]

    profiles: list[np.ndarray] = []
    callback_ids: list[int] = []
    latest_received_at = math.nan
    deadline = time.monotonic() + timeout_s

    while len(profiles) < count and viewer.is_open():
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"received {len(profiles)}/{count} fresh profiles within "
                f"{timeout_s:.1f} s"
            )

        sample = read_latest_live_profile(laser, args.profile_stale_after_s)
        if sample is not None:
            callback_id, profile_received_at, points = sample
            if callback_id != last_callback_id:
                profiles.append(points.copy())
                callback_ids.append(callback_id)
                latest_received_at = profile_received_at
                last_callback_id = callback_id
                viewer.update_live_profile(points)

        viewer.set_status(
            f"{status_prefix} | fresh profiles={len(profiles)}/{count}"
        )
        viewer.process_events()
        time.sleep(0.002)

    return profiles, np.asarray(callback_ids, dtype=np.int64), latest_received_at


def verify_stationary_pose(
    *,
    robot: Any,
    duration_s: float,
    max_translation_mm: float,
    max_rotation_deg: float,
    viewer: TwoPointScanViewer,
    laser: Any,
    args: argparse.Namespace,
    status_prefix: str,
) -> np.ndarray:
    first = validate_transform(robot.read_T_base_tcp(), "T_base_tcp before stability check")
    wait_with_ui(
        duration_s,
        viewer=viewer,
        laser=laser,
        args=args,
        status_prefix=status_prefix,
    )
    second = validate_transform(robot.read_T_base_tcp(), "T_base_tcp after stability check")

    translation = float(np.linalg.norm(second[:3, 3] - first[:3, 3]))
    rotation = rotation_distance_deg(first[:3, :3], second[:3, :3])
    if translation > max_translation_mm or rotation > max_rotation_deg:
        raise RuntimeError(
            "robot is not stationary: "
            f"translation={translation:.4f} mm, rotation={rotation:.4f} deg"
        )
    return second


def build_stop_scan_waypoints(
    scan_from: np.ndarray,
    scan_to: np.ndarray,
    distance_mm: float,
    args: argparse.Namespace,
) -> list[np.ndarray]:
    if args.waypoint_count is not None:
        count = int(args.waypoint_count)
    else:
        count = max(2, int(math.ceil(distance_mm / args.waypoint_spacing_mm)) + 1)

    if count < 2:
        raise ValueError("waypoint count must be at least 2")
    if count > args.max_waypoints:
        raise ValueError(
            f"generated {count} waypoints, exceeding --max-waypoints={args.max_waypoints}. "
            "Increase spacing or explicitly reduce --waypoint-count."
        )

    waypoints: list[np.ndarray] = []
    for fraction in np.linspace(0.0, 1.0, count):
        pose = np.asarray(scan_from, dtype=float).copy()
        pose[:3] = (1.0 - fraction) * scan_from[:3] + fraction * scan_to[:3]
        pose[3:6] = scan_to[3:6]  # fixed first-taught orientation
        waypoints.append(pose)
    return waypoints


def move_to_waypoint_with_ui(
    *,
    robot: Any,
    laser: Any,
    target_pose: np.ndarray,
    waypoint_index: int,
    waypoint_count: int,
    args: argparse.Namespace,
    viewer: TwoPointScanViewer,
) -> np.ndarray:
    worker = MoveWorker(
        lambda: robot.move_l(
            target_pose,
            speed_mm_s=args.scan_speed_mm_s,
            accel_mm_s2=args.scan_accel_mm_s2,
            position_tolerance_mm=args.position_tolerance_mm,
            rotation_tolerance_deg=args.rotation_tolerance_deg,
            timeout_s=args.move_timeout_s,
            stable_count=args.arrival_stable_count,
            poll_interval_s=args.robot_poll_interval_s,
        )
    )
    worker.start()

    last_callback: int | None = None
    while worker.is_alive() and viewer.is_open():
        if viewer.take_abort():
            try:
                robot.stop()
            except Exception as exc:
                print(f"abort request failed: {type(exc).__name__}: {exc}")

        sample = read_latest_live_profile(laser, args.profile_stale_after_s)
        if sample is not None:
            callback_id, _, points = sample
            if callback_id != last_callback:
                viewer.update_live_profile(points)
                last_callback = callback_id

        viewer.set_status(
            f"Moving to waypoint {waypoint_index + 1}/{waypoint_count}. "
            "No scan data are saved while moving."
        )
        viewer.process_events()
        time.sleep(0.005)

    worker.join()
    if worker.error is not None:
        raise RuntimeError(
            f"waypoint {waypoint_index + 1} move failed: {worker.error}"
        )
    if worker.result is None:
        raise RuntimeError("waypoint move returned no TCP pose")
    return np.asarray(worker.result, dtype=float).reshape(6)


def wait_with_ui(
    duration_s: float,
    *,
    viewer: TwoPointScanViewer,
    laser: Any,
    args: argparse.Namespace,
    status_prefix: str,
) -> None:
    deadline = time.monotonic() + max(0.0, float(duration_s))
    last_callback: int | None = None
    while time.monotonic() < deadline and viewer.is_open():
        sample = read_latest_live_profile(laser, args.profile_stale_after_s)
        if sample is not None:
            callback_id, _, points = sample
            if callback_id != last_callback:
                viewer.update_live_profile(points)
                last_callback = callback_id
        remaining = max(0.0, deadline - time.monotonic())
        viewer.set_status(f"{status_prefix} | remaining={remaining:.2f} s")
        viewer.process_events()
        time.sleep(0.005)


def validate_step_geometry(
    first_pose: np.ndarray,
    second_pose: np.ndarray,
    args: argparse.Namespace,
) -> float:
    distance = float(np.linalg.norm(second_pose[:3] - first_pose[:3]))
    if distance < args.min_scan_distance_mm:
        raise ValueError(
            f"scan distance is too short: {distance:.3f} mm < "
            f"{args.min_scan_distance_mm:.3f} mm"
        )
    if distance > args.max_scan_distance_mm:
        raise ValueError(
            f"scan distance is too long: {distance:.3f} mm > "
            f"{args.max_scan_distance_mm:.3f} mm"
        )
    return distance


def ensure_near_second_position(
    robot: Any,
    second_pose: np.ndarray,
    tolerance_mm: float,
) -> None:
    current = np.asarray(robot.read_tcp_pose_vec_mm(), dtype=float).reshape(6)
    error = float(np.linalg.norm(current[:3] - second_pose[:3]))
    if error > tolerance_mm:
        raise RuntimeError(
            "robot is not near the taught second XYZ. "
            f"distance={error:.3f} mm, tolerance={tolerance_mm:.3f} mm"
        )


def align_orientation_at_second(
    *,
    robot: Any,
    first_pose: np.ndarray,
    second_pose: np.ndarray,
    args: argparse.Namespace,
    viewer: TwoPointScanViewer,
    laser: Any,
) -> np.ndarray:
    scan_from = np.r_[second_pose[:3], first_pose[3:6]].astype(float)
    current = np.asarray(robot.read_tcp_pose_vec_mm(), dtype=float).reshape(6)
    rotation_difference = np.linalg.norm(
        (current[3:6] - scan_from[3:6] + 180.0) % 360.0 - 180.0
    )
    position_difference = float(np.linalg.norm(current[:3] - scan_from[:3]))

    if (
        position_difference <= args.position_tolerance_mm
        and rotation_difference <= args.rotation_tolerance_deg
    ):
        return current

    viewer.set_status(
        "Aligning the first taught orientation at the second taught XYZ. "
        "No scan data are saved during alignment."
    )

    worker = MoveWorker(
        lambda: robot.move_l(
            scan_from,
            speed_mm_s=args.alignment_speed_mm_s,
            accel_mm_s2=args.alignment_accel_mm_s2,
            position_tolerance_mm=args.position_tolerance_mm,
            rotation_tolerance_deg=args.rotation_tolerance_deg,
            timeout_s=args.move_timeout_s,
            stable_count=args.arrival_stable_count,
            poll_interval_s=args.robot_poll_interval_s,
        )
    )
    worker.start()

    last_callback: int | None = None
    while worker.is_alive() and viewer.is_open():
        sample = read_latest_live_profile(laser, args.profile_stale_after_s)
        if sample is not None:
            callback_id, _, points = sample
            if callback_id != last_callback:
                viewer.update_live_profile(points)
                last_callback = callback_id
        viewer.process_events()
        time.sleep(0.005)

    worker.join()
    if worker.error is not None:
        raise RuntimeError(f"orientation alignment failed: {worker.error}")
    if worker.result is None:
        raise RuntimeError("orientation alignment returned no TCP pose")
    return worker.result


def scan_second_to_first(
    *,
    robot: Any,
    laser: Any,
    T_tcp_sensor: np.ndarray,
    first_pose: np.ndarray,
    second_pose: np.ndarray,
    step_index: int,
    args: argparse.Namespace,
    viewer: TwoPointScanViewer,
    session: ScanSession,
) -> ScanStep:
    distance = validate_step_geometry(first_pose, second_pose, args)
    ensure_near_second_position(
        robot,
        second_pose,
        tolerance_mm=args.second_position_start_tolerance_mm,
    )

    scan_from = np.r_[second_pose[:3], first_pose[3:6]].astype(float)
    scan_to = np.asarray(first_pose, dtype=float).reshape(6).copy()

    align_orientation_at_second(
        robot=robot,
        first_pose=first_pose,
        second_pose=second_pose,
        args=args,
        viewer=viewer,
        laser=laser,
    )

    waypoints = build_stop_scan_waypoints(scan_from, scan_to, distance, args)

    step = ScanStep(
        step_index=step_index,
        first_taught_pose_vec=first_pose.copy(),
        second_taught_pose_vec=second_pose.copy(),
        scan_from_pose_vec=scan_from.copy(),
        scan_to_pose_vec=scan_to.copy(),
        started_at_utc=utc_now_text(),
    )

    last_render = 0.0

    for waypoint_index, target_pose in enumerate(waypoints):
        if waypoint_index > 0:
            move_to_waypoint_with_ui(
                robot=robot,
                laser=laser,
                target_pose=target_pose,
                waypoint_index=waypoint_index,
                waypoint_count=len(waypoints),
                args=args,
                viewer=viewer,
            )

        # Adapter arrival verification is followed by an independent stability check.
        T_base_tcp = verify_stationary_pose(
            robot=robot,
            duration_s=args.settle_at_waypoint_s,
            max_translation_mm=args.max_capture_translation_mm,
            max_rotation_deg=args.max_capture_rotation_deg,
            viewer=viewer,
            laser=laser,
            args=args,
            status_prefix=(
                f"Settling at waypoint {waypoint_index + 1}/{len(waypoints)}"
            ),
        )

        profiles, callback_ids, profile_received_at = collect_fresh_profiles(
            laser=laser,
            count=args.profiles_per_waypoint,
            timeout_s=args.capture_timeout_s,
            args=args,
            viewer=viewer,
            status_prefix=(
                f"Capturing waypoint {waypoint_index + 1}/{len(waypoints)}"
            ),
        )

        points_sensor, used_profile_count = aggregate_profiles(
            profiles,
            args.capture_aggregate,
        )

        T_base_tcp_after = validate_transform(
            robot.read_T_base_tcp(),
            "T_base_tcp after waypoint capture",
        )
        capture_translation = float(
            np.linalg.norm(T_base_tcp_after[:3, 3] - T_base_tcp[:3, 3])
        )
        capture_rotation = rotation_distance_deg(
            T_base_tcp[:3, :3],
            T_base_tcp_after[:3, :3],
        )
        if (
            capture_translation > args.max_capture_translation_mm
            or capture_rotation > args.max_capture_rotation_deg
        ):
            raise RuntimeError(
                f"waypoint {waypoint_index + 1} capture rejected because TCP moved "
                f"{capture_translation:.4f} mm and {capture_rotation:.4f} deg"
            )

        T_base_sensor = T_base_tcp @ T_tcp_sensor
        points_base = transform_points(T_base_sensor, points_sensor)
        fraction = waypoint_index / max(1, len(waypoints) - 1)

        step.profiles.append(
            MotionProfile(
                step_index=step_index,
                profile_index_in_step=waypoint_index,
                waypoint_fraction=float(fraction),
                captured_at_utc=utc_now_text(),
                profile_received_at=profile_received_at,
                callback_id=int(callback_ids[-1]),
                callback_ids=callback_ids,
                points_sensor=np.asarray(points_sensor, dtype=np.float32),
                points_base=np.asarray(points_base, dtype=np.float32),
                T_base_tcp=np.asarray(T_base_tcp, dtype=float),
                T_base_sensor=np.asarray(T_base_sensor, dtype=float),
                requested_profile_count=args.profiles_per_waypoint,
                used_profile_count=used_profile_count,
                aggregate=args.capture_aggregate,
            )
        )

        now = time.monotonic()
        if now - last_render >= 1.0 / args.render_rate_hz or waypoint_index == len(waypoints) - 1:
            active_groups = [p.points_base for p in step.profiles]
            active_cloud = np.concatenate(active_groups, axis=0)
            prior_cloud = session.display_cloud(args.display_max_points)
            cloud = (
                np.concatenate([prior_cloud, active_cloud], axis=0)
                if len(prior_cloud)
                else active_cloud
            )
            if len(cloud) > args.display_max_points:
                stride = max(1, math.ceil(len(cloud) / args.display_max_points))
                cloud = cloud[::stride]

            active_tcp_path = np.asarray(
                [p.T_base_tcp[:3, 3] for p in step.profiles],
                dtype=np.float32,
            )
            prior_path = session.tcp_path()
            tcp_path = (
                np.concatenate([prior_path, active_tcp_path], axis=0)
                if len(prior_path)
                else active_tcp_path
            )

            viewer.update_cloud(
                cloud=np.asarray(cloud, dtype=np.float32),
                latest_profile=np.asarray(step.profiles[-1].points_base, dtype=np.float32),
                tcp_path=np.asarray(tcp_path, dtype=np.float32),
                current_T_base_tcp=T_base_tcp_after,
            )
            last_render = now

        viewer.set_status(
            f"Step {step_index}: captured waypoint {waypoint_index + 1}/{len(waypoints)} | "
            f"stored captures={len(step.profiles)} | points={step.point_count}"
        )
        viewer.process_events()

    step.completed_at_utc = utc_now_text()

    if len(step.profiles) < 2:
        raise RuntimeError("stop-and-scan step produced fewer than two waypoint captures")

    return step


# ---------------------------------------------------------------------------
# Saving
# ---------------------------------------------------------------------------


def save_session(
    path: Path,
    *,
    session: ScanSession,
    T_tcp_sensor: np.ndarray,
    handeye_path: Path,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")

    all_profiles = [
        profile
        for step in session.steps
        for profile in step.profiles
    ]

    metadata = {
        "format": "two_point_multi_step_stop_and_scan",
        "format_version": 2,
        "saved_at_utc": utc_now_text(),
        "handeye_path": str(handeye_path),
        "step_count": len(session.steps),
        "profile_count": len(all_profiles),
        "point_count": session.point_count,
        "scan_direction": "second_taught_position_to_first_taught_position",
        "scan_orientation": "first_taught_tcp_orientation",
        "coordinate_units": "mm",
        "transform_convention": (
            "points_base = T_base_tcp @ T_tcp_sensor @ points_sensor"
        ),
        "capture_mode": "move_stop_settle_aggregate_profiles",
        "waypoint_spacing_mm": args.waypoint_spacing_mm,
        "waypoint_count_override": args.waypoint_count,
        "profiles_per_waypoint": args.profiles_per_waypoint,
        "capture_aggregate": args.capture_aggregate,
        "scan_speed_mm_s": args.scan_speed_mm_s,
        "scan_accel_mm_s2": args.scan_accel_mm_s2,
    }

    payload: dict[str, np.ndarray] = {
        "metadata_json": np.array(json.dumps(metadata, ensure_ascii=False)),
        "T_tcp_sensor": np.asarray(T_tcp_sensor, dtype=float),
        "step_count": np.array(len(session.steps), dtype=np.int64),
        "capture_count": np.array(len(all_profiles), dtype=np.int64),
    }

    if session.steps:
        payload["step_first_taught_pose_vec"] = np.stack(
            [step.first_taught_pose_vec for step in session.steps], axis=0
        )
        payload["step_second_taught_pose_vec"] = np.stack(
            [step.second_taught_pose_vec for step in session.steps], axis=0
        )
        payload["step_scan_from_pose_vec"] = np.stack(
            [step.scan_from_pose_vec for step in session.steps], axis=0
        )
        payload["step_scan_to_pose_vec"] = np.stack(
            [step.scan_to_pose_vec for step in session.steps], axis=0
        )
        payload["step_started_at_utc"] = np.asarray(
            [step.started_at_utc for step in session.steps], dtype=str
        )
        payload["step_completed_at_utc"] = np.asarray(
            [step.completed_at_utc or "" for step in session.steps], dtype=str
        )
        payload["step_profile_count"] = np.asarray(
            [len(step.profiles) for step in session.steps], dtype=np.int64
        )

    if all_profiles:
        payload["T_base_tcp"] = np.stack(
            [profile.T_base_tcp for profile in all_profiles], axis=0
        )
        payload["T_base_sensor"] = np.stack(
            [profile.T_base_sensor for profile in all_profiles], axis=0
        )
        payload["step_index"] = np.asarray(
            [profile.step_index for profile in all_profiles], dtype=np.int64
        )
        payload["profile_index_in_step"] = np.asarray(
            [profile.profile_index_in_step for profile in all_profiles],
            dtype=np.int64,
        )
        payload["waypoint_fraction"] = np.asarray(
            [profile.waypoint_fraction for profile in all_profiles], dtype=float
        )
        payload["callback_ids"] = np.asarray(
            [profile.callback_id for profile in all_profiles], dtype=np.int64
        )
        payload["requested_profile_count"] = np.asarray(
            [profile.requested_profile_count for profile in all_profiles], dtype=np.int64
        )
        payload["used_profile_count"] = np.asarray(
            [profile.used_profile_count for profile in all_profiles], dtype=np.int64
        )
        payload["aggregate"] = np.asarray(
            [profile.aggregate for profile in all_profiles], dtype=str
        )
        payload["profile_received_at"] = np.asarray(
            [profile.profile_received_at for profile in all_profiles], dtype=float
        )
        payload["captured_at_utc"] = np.asarray(
            [profile.captured_at_utc for profile in all_profiles], dtype=str
        )
        payload["tcp_positions_base"] = np.asarray(
            [profile.T_base_tcp[:3, 3] for profile in all_profiles],
            dtype=np.float32,
        )
        payload["points_base_merged"] = np.concatenate(
            [profile.points_base for profile in all_profiles], axis=0
        ).astype(np.float32)

        # Compatibility with the existing sphere-fit loader:
        # every motion profile is exposed as capture_XXXX_points_base.
        for capture_index, profile in enumerate(all_profiles):
            prefix = f"capture_{capture_index:04d}"
            payload[f"{prefix}_points_sensor"] = profile.points_sensor
            payload[f"{prefix}_points_base"] = profile.points_base
            payload[f"{prefix}_callback_ids"] = profile.callback_ids
            payload[f"{prefix}_waypoint_fraction"] = np.array(
                profile.waypoint_fraction, dtype=float
            )
            payload[f"{prefix}_T_base_tcp"] = profile.T_base_tcp
            payload[f"{prefix}_step_index"] = np.array(
                profile.step_index, dtype=np.int64
            )

    try:
        with temporary.open("xb") as stream:
            np.savez_compressed(stream, **payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def connect_robot(robot: Any, args: argparse.Namespace) -> None:
    kwargs: dict[str, Any] = {}
    if args.operation_mode != "current":
        kwargs["operation_mode"] = args.operation_mode
    if args.speed_bar is not None:
        kwargs["speed_bar"] = args.speed_bar

    try:
        robot.connect(**kwargs)
    except TypeError:
        # Backward compatibility with a read-only/older adapter connect().
        if kwargs:
            raise TypeError(
                "RobotAdapter.connect() does not accept operation_mode/speed_bar. "
                "Use the updated adapter containing move_l()."
            )
        robot.connect()


def run(args: argparse.Namespace) -> None:
    if args.batch_profiles <= 0:
        raise ValueError("--batch-profiles must be positive")
    if args.scan_speed_mm_s <= 0 or args.scan_accel_mm_s2 <= 0:
        raise ValueError("scan speed and acceleration must be positive")
    if args.alignment_speed_mm_s <= 0 or args.alignment_accel_mm_s2 <= 0:
        raise ValueError("alignment speed and acceleration must be positive")
    if args.render_rate_hz <= 0 or args.live_profile_rate_hz <= 0:
        raise ValueError("render rates must be positive")
    if args.display_max_points <= 0 or args.point_size_px <= 0:
        raise ValueError("display limits must be positive")
    if args.waypoint_spacing_mm <= 0:
        raise ValueError("--waypoint-spacing-mm must be positive")
    if args.waypoint_count is not None and args.waypoint_count < 2:
        raise ValueError("--waypoint-count must be at least 2")
    if args.max_waypoints < 2:
        raise ValueError("--max-waypoints must be at least 2")
    if args.profiles_per_waypoint <= 0:
        raise ValueError("--profiles-per-waypoint must be positive")
    if args.capture_timeout_s <= 0:
        raise ValueError("--capture-timeout-s must be positive")
    if args.settle_at_waypoint_s < 0:
        raise ValueError("--settle-at-waypoint-s cannot be negative")

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

    connect_robot(robot, args)
    try:
        laser.connect()
    except BaseException:
        robot.close()
        raise

    viewer: TwoPointScanViewer | None = None
    session = ScanSession()
    first_pose: np.ndarray | None = None
    second_pose: np.ndarray | None = None
    latest_live_T: np.ndarray | None = None
    latest_live_profile: np.ndarray | None = None
    latest_callback_id: int | None = None
    latest_message = (
        "Jog the robot to the first position and click Teach first. "
        "The live profile is display-only until scanning starts."
    )

    next_live_profile = 0.0
    next_cloud_render = 0.0

    print(f"hand-eye: {args.handeye}")
    print(f"save path: {args.save_path}")
    print("Each step stop-and-scans SECOND -> FIRST with FIRST orientation fixed.")

    try:
        viewer = TwoPointScanViewer(
            point_size_px=args.point_size_px,
            display_profile_stride=args.display_profile_stride,
        )
        viewer.set_busy(False)
        running = True

        while running and viewer.is_open():
            viewer.process_events()

            if viewer.take_quit():
                try:
                    save_session(
                        args.save_path,
                        session=session,
                        T_tcp_sensor=T_tcp_sensor,
                        handeye_path=args.handeye,
                        args=args,
                    )
                    print(
                        f"saved on quit: {args.save_path} | "
                        f"steps={len(session.steps)}, profiles={session.profile_count}, "
                        f"points={session.point_count}"
                    )
                except Exception as exc:
                    print(f"save on quit failed: {type(exc).__name__}: {exc}")
                running = False
                continue

            if viewer.take_teach_first():
                try:
                    first_pose = np.asarray(
                        robot.read_tcp_pose_vec_mm(), dtype=float
                    ).reshape(6)
                    latest_message = (
                        "First position taught. Its orientation will be fixed "
                        "during this step's scan."
                    )
                    viewer.set_teaching(first_pose, second_pose)
                except Exception as exc:
                    latest_message = (
                        f"Teach first failed: {type(exc).__name__}: {exc}"
                    )

            if viewer.take_teach_second():
                try:
                    second_pose = np.asarray(
                        robot.read_tcp_pose_vec_mm(), dtype=float
                    ).reshape(6)
                    if first_pose is not None:
                        distance = validate_step_geometry(first_pose, second_pose, args)
                        latest_message = (
                            f"Second position taught. Planned scan is second -> first, "
                            f"translation={distance:.2f} mm. Second orientation is ignored."
                        )
                    else:
                        latest_message = (
                            "Second position taught, but first position is still missing."
                        )
                    viewer.set_teaching(first_pose, second_pose)
                except Exception as exc:
                    latest_message = (
                        f"Teach second failed: {type(exc).__name__}: {exc}"
                    )

            if viewer.take_new_step():
                first_pose = None
                second_pose = None
                viewer.set_teaching(first_pose, second_pose)
                latest_message = (
                    f"New step ready. Completed steps={len(session.steps)}. "
                    "Teach the next first position."
                )

            if viewer.take_undo_step():
                removed = session.remove_last_step()
                if removed is None:
                    latest_message = "No completed step to undo."
                else:
                    latest_message = (
                        f"Removed completed step {removed.step_index}. "
                        f"Remaining steps={len(session.steps)}."
                    )

            if viewer.take_save():
                try:
                    save_session(
                        args.save_path,
                        session=session,
                        T_tcp_sensor=T_tcp_sensor,
                        handeye_path=args.handeye,
                        args=args,
                    )
                    latest_message = (
                        f"Saved {len(session.steps)} steps, {session.profile_count} "
                        f"profiles, {session.point_count} points to {args.save_path}."
                    )
                    print(latest_message)
                except Exception as exc:
                    latest_message = f"Save failed: {type(exc).__name__}: {exc}"

            if viewer.take_finish():
                save_session(
                    args.save_path,
                    session=session,
                    T_tcp_sensor=T_tcp_sensor,
                    handeye_path=args.handeye,
                    args=args,
                )
                print(
                    f"Saved and finished: {args.save_path} | "
                    f"steps={len(session.steps)}, profiles={session.profile_count}, "
                    f"points={session.point_count}"
                )
                running = False
                continue

            if viewer.take_start_scan():
                if first_pose is None or second_pose is None:
                    latest_message = "Teach both first and second positions before scanning."
                else:
                    viewer.set_busy(True)
                    try:
                        step = scan_second_to_first(
                            robot=robot,
                            laser=laser,
                            T_tcp_sensor=T_tcp_sensor,
                            first_pose=first_pose,
                            second_pose=second_pose,
                            step_index=len(session.steps),
                            args=args,
                            viewer=viewer,
                            session=session,
                        )
                        session.add_step(step)
                        latest_message = (
                            f"Step {step.step_index} complete: "
                            f"waypoint captures={len(step.profiles)}, points={step.point_count}. "
                            "Robot is now at the first taught position. "
                            "Click New step to teach the next scan."
                        )
                        print(latest_message)
                        if args.auto_save:
                            save_session(
                                args.save_path,
                                session=session,
                                T_tcp_sensor=T_tcp_sensor,
                                handeye_path=args.handeye,
                                args=args,
                            )
                            latest_message += f" Auto-saved to {args.save_path}."
                    except Exception as exc:
                        latest_message = (
                            f"Scan step failed: {type(exc).__name__}: {exc}"
                        )
                        print(latest_message)
                    finally:
                        viewer.set_busy(False)

            now = time.monotonic()
            if now >= next_live_profile:
                try:
                    sample = read_latest_live_profile(
                        laser, args.profile_stale_after_s
                    )
                    if sample is not None:
                        callback_id, _, points = sample
                        if callback_id != latest_callback_id:
                            latest_callback_id = callback_id
                            latest_live_profile = points
                            viewer.update_live_profile(points)
                except Exception as exc:
                    latest_message = (
                        f"Live profile error: {type(exc).__name__}: {exc}"
                    )
                next_live_profile = now + 1.0 / args.live_profile_rate_hz

            if now >= next_cloud_render:
                try:
                    latest_live_T = validate_transform(
                        robot.read_T_base_tcp(), "live T_base_tcp"
                    )
                    xyz = latest_live_T[:3, 3]
                    rpy = rotation_to_rpy_deg(latest_live_T[:3, :3])
                    tcp_text = (
                        f"TCP xyz=[{xyz[0]:.1f}, {xyz[1]:.1f}, {xyz[2]:.1f}] mm, "
                        f"rpy=[{rpy[0]:.1f}, {rpy[1]:.1f}, {rpy[2]:.1f}] deg"
                    )
                except Exception as exc:
                    tcp_text = f"TCP unavailable: {type(exc).__name__}: {exc}"

                viewer.update_cloud(
                    cloud=session.display_cloud(args.display_max_points),
                    latest_profile=session.latest_profile(),
                    tcp_path=session.tcp_path(),
                    current_T_base_tcp=latest_live_T,
                )
                viewer.set_status(
                    f"{latest_message} | {tcp_text} | "
                    f"steps={len(session.steps)}, profiles={session.profile_count}, "
                    f"points={session.point_count}"
                )
                next_cloud_render = now + 1.0 / args.render_rate_hz

            time.sleep(0.003)

    except KeyboardInterrupt:
        print("\nInterrupted")
    finally:
        if args.save_on_exit and session.steps:
            try:
                save_session(
                    args.save_path,
                    session=session,
                    T_tcp_sensor=T_tcp_sensor,
                    handeye_path=args.handeye,
                    args=args,
                )
                print(
                    f"saved on exit: {args.save_path} | "
                    f"steps={len(session.steps)}, profiles={session.profile_count}, "
                    f"points={session.point_count}"
                )
            except Exception as exc:
                print(f"save-on-exit failed: {type(exc).__name__}: {exc}")
        if viewer is not None:
            viewer.close()
        laser.close()
        robot.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Teach two TCP positions per step, divide the second-to-first segment "
            "into waypoints, then move-stop-capture at each waypoint with the "
            "first taught orientation fixed."
        )
    )

    parser.add_argument("--handeye", type=Path, default=DEFAULT_HANDEYE)
    parser.add_argument("--save-path", type=Path, default=DEFAULT_SAVE_PATH)

    parser.add_argument("--robot-host", default="192.168.0.10")
    parser.add_argument("--robot-port", type=int)
    parser.add_argument("--operation-mode", choices=("current", "real", "simulation"), default="current")
    parser.add_argument("--speed-bar", type=float)

    parser.add_argument("--laser-ip", default="192.168.1.1")
    parser.add_argument("--laser-control-port", type=int, default=24691)
    parser.add_argument("--laser-high-speed-port", type=int, default=24692)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument(
        "--batch-profiles",
        type=int,
        default=1,
        help="Use 1 so every laser callback corresponds to one scan profile.",
    )
    parser.add_argument("--profile-stale-after-s", type=float, default=0.5)

    parser.add_argument("--scan-speed-mm-s", type=float, default=20.0)
    parser.add_argument("--scan-accel-mm-s2", type=float, default=20.0)
    parser.add_argument("--alignment-speed-mm-s", type=float, default=10.0)
    parser.add_argument("--alignment-accel-mm-s2", type=float, default=10.0)
    parser.add_argument(
        "--waypoint-spacing-mm",
        type=float,
        default=1.0,
        help="Nominal distance between stationary capture waypoints. Endpoints are included.",
    )
    parser.add_argument(
        "--waypoint-count",
        type=int,
        help="Optional fixed number of waypoints, overriding --waypoint-spacing-mm.",
    )
    parser.add_argument("--max-waypoints", type=int, default=500)
    parser.add_argument("--settle-at-waypoint-s", type=float, default=0.30)
    parser.add_argument("--profiles-per-waypoint", type=int, default=10)
    parser.add_argument(
        "--capture-aggregate",
        choices=("mean", "median", "latest"),
        default="mean",
    )
    parser.add_argument("--capture-timeout-s", type=float, default=5.0)
    parser.add_argument("--max-capture-translation-mm", type=float, default=0.10)
    parser.add_argument("--max-capture-rotation-deg", type=float, default=0.05)
    parser.add_argument("--move-timeout-s", type=float, default=60.0)
    parser.add_argument("--position-tolerance-mm", type=float, default=1.0)
    parser.add_argument("--rotation-tolerance-deg", type=float, default=1.0)
    parser.add_argument("--arrival-stable-count", type=int, default=5)
    parser.add_argument("--robot-poll-interval-s", type=float, default=0.05)

    parser.add_argument("--min-scan-distance-mm", type=float, default=2.0)
    parser.add_argument("--max-scan-distance-mm", type=float, default=500.0)
    parser.add_argument(
        "--second-position-start-tolerance-mm",
        type=float,
        default=5.0,
        help="Robot must still be near the taught second XYZ when Start scan is pressed.",
    )

    parser.add_argument("--live-profile-rate-hz", type=float, default=20.0)
    parser.add_argument("--display-profile-stride", type=int, default=4)
    parser.add_argument("--render-rate-hz", type=float, default=8.0)
    parser.add_argument("--display-max-points", type=int, default=250_000)
    parser.add_argument("--point-size-px", type=float, default=2.0)

    parser.add_argument("--auto-save", action="store_true")
    parser.add_argument("--save-on-exit", action="store_true")

    parser.add_argument(
        "--robot-adapter-module",
        default="real_laser_handeye.robot_adapter",
        help="Module containing the updated RobotAdapter with move_l().",
    )
    parser.add_argument(
        "--laser-adapter-module",
        default="real_laser_handeye.laser_adapter",
        help="Module containing LaserAdapter.",
    )

    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()