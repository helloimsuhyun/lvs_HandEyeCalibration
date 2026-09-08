"""

cd ~/lvs_HandEyeCalibration

conda activate laser_handeye

source /opt/ros/humble/setup.bash
source ~/lvs_HandEyeCalibration/moveit2_ws/install/setup.bash

export PYTHONPATH="$PWD:$PWD/mujoco_handeye_sim/src${PYTHONPATH:+:$PYTHONPATH}"

python -m real_laser_handeye.workflow_gui --mode sim
"""


from __future__ import annotations

import argparse
from collections import deque
from copy import deepcopy
import ipaddress
import json
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
from typing import Callable

import numpy as np
from scipy.spatial.transform import Rotation

try:
    import pyqtgraph as pg
    from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
except ImportError as exc:
    raise RuntimeError(
        "This GUI requires PyQtGraph + a Qt binding.\n"
        "Install with:\n"
        "  pip install pyqtgraph PyQt6"
    ) from exc

from matplotlib.backends.backend_qtagg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure

from .workflow import (
    ScanPlan,
    ScanPlanner,
    VisualizationSample,
    WorkflowConfig,
    WorkflowController,
    _load_transform,
)
from .generate_optimal_tcp_poses import generate_optimal_tcp_poses


# ---------------------------------------------------------------------------
# Qt compatibility helpers
# ---------------------------------------------------------------------------

def _qt_exec(app) -> int:
    execute = getattr(app, "exec", None)
    if callable(execute):
        return int(execute())
    return int(app.exec_())


def _qt_enum(owner, group: str, name: str):
    """Resolve Qt5 flat enums and Qt6 scoped enums with one code path."""
    scoped = getattr(owner, group, None)
    if scoped is not None and hasattr(scoped, name):
        return getattr(scoped, name)
    return getattr(owner, name)


def _header_stretch(header) -> None:
    resize_to_contents = _qt_enum(
        QtWidgets.QHeaderView, "ResizeMode", "ResizeToContents"
    )
    header.setSectionResizeMode(resize_to_contents)
    header.setStretchLastSection(True)


def _message(parent, title: str, text: str, *, error: bool = False) -> None:
    if error:
        QtWidgets.QMessageBox.critical(parent, title, text)
    else:
        QtWidgets.QMessageBox.information(parent, title, text)


# ---------------------------------------------------------------------------
# Reusable widgets
# ---------------------------------------------------------------------------

class StatusPill(QtWidgets.QLabel):
    """Small, readable state indicator used in the header."""

    def __init__(self, text: str, state: str = "neutral"):
        super().__init__(text)
        self.set_state(text, state)

    def set_state(self, text: str, state: str) -> None:
        self.setText(text)
        self.setProperty("state", state)
        self.style().unpolish(self)
        self.style().polish(self)


class MetricCard(QtWidgets.QFrame):
    def __init__(self, title: str, value: str = "—"):
        super().__init__()
        self.setObjectName("metricCard")
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(2)

        title_label = QtWidgets.QLabel(title)
        title_label.setObjectName("metricTitle")
        self.value_label = QtWidgets.QLabel(value)
        self.value_label.setObjectName("metricValue")

        layout.addWidget(title_label)
        layout.addWidget(self.value_label)

    def set_value(self, value: str) -> None:
        self.value_label.setText(value)


class StartupDialog(QtWidgets.QDialog):
    """Select the execution target before any robot or laser is opened."""

    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.setWindowTitle("Laser Hand-eye — Start")
        self.setModal(True)
        self.setMinimumWidth(430)

        form = QtWidgets.QFormLayout(self)
        form.setContentsMargins(22, 20, 22, 18)
        form.setSpacing(12)

        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItems(["sim", "real"])
        self.mode_combo.setCurrentText(args.mode or "sim")
        self.robot_combo = QtWidgets.QComboBox()
        self.robot_combo.addItems(["rb5", "ur5e"])
        self.robot_combo.setCurrentText(args.robot or "rb5")
        self.robot_ip_edit = QtWidgets.QLineEdit(args.robot_ip or "")
        self.laser_ip_edit = QtWidgets.QLineEdit(args.laser_ip or "")
        self.robot_ip_edit.setPlaceholderText("e.g. 169.254.186.20")
        self.laser_ip_edit.setPlaceholderText("e.g. 169.254.186.182")

        form.addRow("Mode", self.mode_combo)
        form.addRow("Robot", self.robot_combo)
        form.addRow("Robot IP", self.robot_ip_edit)
        form.addRow("Laser IP", self.laser_ip_edit)

        note = QtWidgets.QLabel(
            "Real mode connects to the selected robot's ROS 2 driver and the "
            "Keyence laser. IP fields are used only in real mode."
        )
        note.setWordWrap(True)
        form.addRow(note)

        buttons = QtWidgets.QDialogButtonBox(
            _qt_enum(QtWidgets.QDialogButtonBox, "StandardButton", "Ok")
            | _qt_enum(QtWidgets.QDialogButtonBox, "StandardButton", "Cancel")
        )
        buttons.accepted.connect(self._accept_checked)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)
        self.mode_combo.currentTextChanged.connect(self._update_mode)
        self._update_mode(self.mode_combo.currentText())

    def _update_mode(self, mode: str) -> None:
        enabled = mode == "real"
        self.robot_ip_edit.setEnabled(enabled)
        self.laser_ip_edit.setEnabled(enabled)

    def _accept_checked(self) -> None:
        if self.mode_combo.currentText() == "real":
            for label, edit in (
                ("Robot IP", self.robot_ip_edit),
                ("Laser IP", self.laser_ip_edit),
            ):
                try:
                    address = ipaddress.ip_address(edit.text().strip())
                    if address.version != 4:
                        raise ValueError("IPv4 required")
                except ValueError:
                    _message(self, "Invalid connection setting", f"{label} must be a valid IPv4 address.", error=True)
                    edit.setFocus()
                    return
        self.accept()

    def selections(self) -> tuple[str, str, str | None, str | None]:
        mode = self.mode_combo.currentText()
        robot_ip = self.robot_ip_edit.text().strip() or None
        laser_ip = self.laser_ip_edit.text().strip() or None
        return mode, self.robot_combo.currentText(), robot_ip, laser_ip


class LivePlots(QtWidgets.QWidget):
    """High-rate live plots: persistent PlotDataItems, never clear/rebuild."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.profile = pg.PlotWidget(title="Live laser profile")
        self.profile.setLabel("bottom", "Sensor X", units="mm")
        self.profile.setLabel("left", "Sensor Z", units="mm")
        self.profile.showGrid(x=True, y=True, alpha=0.22)
        self.profile_curve = self.profile.plot(
            pen=pg.mkPen("#38bdf8", width=2),
        )

        self.tcp = pg.PlotWidget(title="TCP position history")
        self.tcp.setLabel("bottom", "Elapsed time", units="s")
        self.tcp.setLabel("left", "Base position", units="mm")
        self.tcp.showGrid(x=True, y=True, alpha=0.22)
        self.tcp.addLegend(offset=(10, 10))
        self.tcp_x = self.tcp.plot(pen=pg.mkPen("#fb7185", width=2), name="X")
        self.tcp_y = self.tcp.plot(pen=pg.mkPen("#4ade80", width=2), name="Y")
        self.tcp_z = self.tcp.plot(pen=pg.mkPen("#60a5fa", width=2), name="Z")

        layout.addWidget(self.profile, 1)
        layout.addWidget(self.tcp, 1)

    def update_data(
        self,
        points: np.ndarray,
        timeline: np.ndarray,
        history: np.ndarray,
        xyz: np.ndarray,
        rpy: np.ndarray,
    ) -> None:
        if points.ndim == 2 and points.shape[1] == 3 and len(points):
            valid = points[np.all(np.isfinite(points), axis=1)]
            self.profile_curve.setData(valid[:, 0], valid[:, 2])
            self.profile.setTitle(f"Live laser profile  ·  {len(valid):,} points")
        else:
            self.profile_curve.setData([], [])
            self.profile.setTitle("Live laser profile  ·  no fresh profile")

        if len(history):
            self.tcp_x.setData(timeline, history[:, 0])
            self.tcp_y.setData(timeline, history[:, 1])
            self.tcp_z.setData(timeline, history[:, 2])
            if len(timeline) > 1:
                right = float(timeline[-1])
                self.tcp.setXRange(max(0.0, right - 30.0), right + 0.5, padding=0)

        self.tcp.setTitle(
            "TCP  "
            f"xyz=[{xyz[0]:.1f}, {xyz[1]:.1f}, {xyz[2]:.1f}] mm  ·  "
            f"rpy=[{rpy[0]:.1f}, {rpy[1]:.1f}, {rpy[2]:.1f}]°"
        )


class AspectImageLabel(QtWidgets.QLabel):
    """QLabel that keeps the latest RGB frame fitted without distortion."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._source_pixmap: QtGui.QPixmap | None = None
        self.setAlignment(_qt_enum(QtCore.Qt, "AlignmentFlag", "AlignCenter"))
        self.setMinimumSize(520, 420)
        self.setText("Connect to start the live robot model")

    def set_rgb_frame(self, frame: np.ndarray) -> None:
        rgb = np.ascontiguousarray(frame, dtype=np.uint8)
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError("rendered frame must have shape (H, W, 3)")
        height, width, _ = rgb.shape
        image = QtGui.QImage(
            rgb.data,
            width,
            height,
            int(rgb.strides[0]),
            _qt_enum(QtGui.QImage, "Format", "Format_RGB888"),
        ).copy()
        self._source_pixmap = QtGui.QPixmap.fromImage(image)
        self._fit_pixmap()

    def _fit_pixmap(self) -> None:
        if self._source_pixmap is None:
            return
        keep_aspect = _qt_enum(QtCore.Qt, "AspectRatioMode", "KeepAspectRatio")
        smooth = _qt_enum(QtCore.Qt, "TransformationMode", "SmoothTransformation")
        self.setPixmap(self._source_pixmap.scaled(self.size(), keep_aspect, smooth))

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._fit_pixmap()


class MountVerificationView(QtWidgets.QWidget):
    """Read-only measured-state plots beside a joint-driven MuJoCo twin."""

    HISTORY = 600

    def __init__(self, config: WorkflowConfig, parent=None):
        super().__init__(parent)
        self.config = config
        self.joint_names = config.joint_names
        self.times: deque[float] = deque(maxlen=self.HISTORY)
        self.joints: deque[np.ndarray] = deque(maxlen=self.HISTORY)
        self.tcp_xyz: deque[np.ndarray] = deque(maxlen=self.HISTORY)
        self.tcp_rpy: deque[np.ndarray] = deque(maxlen=self.HISTORY)
        self.simulation = None
        self.renderer = None
        self.scene_option = None
        self._base_qpos = None
        self._T_tcp_sensor_mm = _load_transform(config.path("handeye"))

        root = QtWidgets.QHBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(10)
        splitter = QtWidgets.QSplitter()
        root.addWidget(splitter)

        plots = QtWidgets.QWidget()
        plots_layout = QtWidgets.QVBoxLayout(plots)
        plots_layout.setContentsMargins(0, 0, 0, 0)
        plots_layout.setSpacing(7)

        colors = ("#fb7185", "#fbbf24", "#4ade80", "#22d3ee", "#60a5fa", "#c084fc")
        self.joint_plot = pg.PlotWidget(title="Measured joint position")
        self.joint_plot.setLabel("bottom", "Elapsed time", units="s")
        self.joint_plot.setLabel("left", "Joint angle", units="deg")
        self.joint_plot.showGrid(x=True, y=True, alpha=0.22)
        self.joint_plot.addLegend(offset=(8, 8))
        self.joint_curves = [
            self.joint_plot.plot(
                pen=pg.mkPen(colors[index], width=1.8),
                name=name,
            )
            for index, name in enumerate(self.joint_names)
        ]

        self.position_plot = pg.PlotWidget(title="Measured TCP position")
        self.position_plot.setLabel("bottom", "Elapsed time", units="s")
        self.position_plot.setLabel("left", "Base position", units="mm")
        self.position_plot.showGrid(x=True, y=True, alpha=0.22)
        self.position_plot.addLegend(offset=(8, 8))
        self.position_curves = [
            self.position_plot.plot(pen=pg.mkPen(color, width=1.8), name=axis)
            for color, axis in zip(colors[:3], ("X", "Y", "Z"))
        ]

        self.orientation_plot = pg.PlotWidget(title="Measured TCP orientation")
        self.orientation_plot.setLabel("bottom", "Elapsed time", units="s")
        self.orientation_plot.setLabel("left", "Base-frame RPY", units="deg")
        self.orientation_plot.showGrid(x=True, y=True, alpha=0.22)
        self.orientation_plot.addLegend(offset=(8, 8))
        self.orientation_curves = [
            self.orientation_plot.plot(pen=pg.mkPen(color, width=1.8), name=axis)
            for color, axis in zip(colors[3:], ("Roll", "Pitch", "Yaw"))
        ]
        plots_layout.addWidget(self.joint_plot, 2)
        plots_layout.addWidget(self.position_plot, 1)
        plots_layout.addWidget(self.orientation_plot, 1)

        model_panel = QtWidgets.QFrame()
        model_panel.setObjectName("sidePanel")
        model_layout = QtWidgets.QVBoxLayout(model_panel)
        model_layout.setContentsMargins(10, 10, 10, 10)
        self.model_status = QtWidgets.QLabel(
            "Read-only verification: move the real robot with its pendant/freedrive. "
            "This screen sends no motion commands."
        )
        self.model_status.setWordWrap(True)
        self.model_status.setObjectName("legendText")
        self.image = AspectImageLabel()
        model_layout.addWidget(self.model_status)
        model_layout.addWidget(self.image, 1)

        splitter.addWidget(plots)
        splitter.addWidget(model_panel)
        splitter.setSizes([690, 990])

    def _ensure_renderer(self) -> None:
        if self.renderer is not None:
            return
        import mujoco
        from handeye_mujoco import HandEyeSimulation

        self.simulation = HandEyeSimulation(self.config.path("mujoco_model"))
        self.simulation.reset_home()
        self._base_qpos = self.simulation.data.qpos.copy()
        self.renderer = mujoco.Renderer(
            self.simulation.model, height=480, width=640
        )
        self.scene_option = mujoco.MjvOption()
        mujoco.mjv_defaultOption(self.scene_option)
        self.scene_option.geomgroup[:] = 1
        self.scene_option.geomgroup[3] = 0
        if any(
            (mujoco.mj_id2name(
                self.simulation.model, mujoco.mjtObj.mjOBJ_GEOM, index
            ) or "").startswith("robot_visual_")
            for index in range(self.simulation.model.ngeom)
        ):
            self.scene_option.geomgroup[1] = 0

    def update_state(
        self,
        elapsed_s: float,
        joint_positions_deg: np.ndarray,
        T_base_tcp_mm: np.ndarray,
    ) -> None:
        joints = np.asarray(joint_positions_deg, dtype=float).reshape(-1)
        transform = np.asarray(T_base_tcp_mm, dtype=float).reshape(4, 4)
        if joints.shape != (len(self.joint_names),):
            raise ValueError("measured joint vector does not match configured joint names")

        rpy = Rotation.from_matrix(transform[:3, :3]).as_euler("xyz", degrees=True)
        self.times.append(float(elapsed_s))
        self.joints.append(joints.copy())
        self.tcp_xyz.append(transform[:3, 3].copy())
        self.tcp_rpy.append(rpy)
        timeline = np.asarray(self.times)
        joint_history = np.asarray(self.joints)
        xyz_history = np.asarray(self.tcp_xyz)
        rpy_history = np.asarray(self.tcp_rpy)
        for index, curve in enumerate(self.joint_curves):
            curve.setData(timeline, joint_history[:, index])
        for index, curve in enumerate(self.position_curves):
            curve.setData(timeline, xyz_history[:, index])
        for index, curve in enumerate(self.orientation_curves):
            curve.setData(timeline, rpy_history[:, index])
        if len(timeline) > 1:
            right = float(timeline[-1])
            for plot in (self.joint_plot, self.position_plot, self.orientation_plot):
                plot.setXRange(max(0.0, right - 30.0), right + 0.5, padding=0)

        self._ensure_renderer()
        import mujoco

        assert self.simulation is not None
        qpos = self.simulation.qpos_with_named_joints(
            dict(zip(self.joint_names, np.radians(joints))),
            base_qpos=self._base_qpos,
        )
        self.simulation.set_joint_positions(qpos)
        T_model_sensor = self.simulation.sensor_pose_world().copy()
        T_model_sensor[:3, 3] *= 1e3
        T_model_tcp = T_model_sensor @ np.linalg.inv(self._T_tcp_sensor_mm)
        position_error = float(
            np.linalg.norm(transform[:3, 3] - T_model_tcp[:3, 3])
        )
        rotation_error = float(np.degrees(Rotation.from_matrix(
            T_model_tcp[:3, :3].T @ transform[:3, :3]
        ).magnitude()))
        collision = self.simulation.collision_report()
        collision_text = "CLEAR" if collision.collision_free else "COLLISION"
        self.model_status.setText(
            "READ ONLY · pendant/freedrive로 실제 로봇을 천천히 움직이세요.  "
            f"Model↔measured TCP Δ: {position_error:.2f} mm / {rotation_error:.2f}°  ·  "
            f"MuJoCo: {collision_text}\n"
            f"Measured xyz={np.round(transform[:3, 3], 2).tolist()} mm  ·  "
            f"rpy={np.round(rpy, 2).tolist()}°"
        )
        assert self.renderer is not None and self.scene_option is not None
        self.renderer.update_scene(
            self.simulation.data,
            camera="overview",
            scene_option=self.scene_option,
        )
        self.image.set_rgb_frame(self.renderer.render())

    def shutdown(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


class WorldScanView(QtWidgets.QWidget):
    """Static 3D scan view.

    This deliberately uses Matplotlib instead of another OpenGL context.
    It is refreshed only after captures / stable-pose changes, so it does not
    participate in the high-rate live loop.  Each capture remains a separate
    polyline, which makes the physical laser scan lines visually explicit.
    """

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.base_title = title
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.figure.patch.set_facecolor("#0b1220")
        self.axis = self.figure.add_subplot(111, projection="3d")
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.setIconSize(QtCore.QSize(16, 16))
        layout.addWidget(self.canvas, 1)
        layout.addWidget(self.toolbar, 0)
        self._reset_axis(self.base_title)
        self.canvas.draw_idle()

    def _reset_axis(self, title: str) -> None:
        ax = self.axis
        ax.clear()
        ax.set_facecolor("#0b1220")
        ax.set_title(title, color="#e5edf6", pad=12)
        ax.set_xlabel("Base X [mm]", color="#cbd5e1")
        ax.set_ylabel("Base Y [mm]", color="#cbd5e1")
        ax.set_zlabel("Base Z [mm]", color="#cbd5e1")
        ax.tick_params(colors="#aab7c8", labelsize=8)
        ax.grid(True, alpha=0.18)
        try:
            ax.xaxis.pane.set_facecolor((0.05, 0.09, 0.15, 1.0))
            ax.yaxis.pane.set_facecolor((0.05, 0.09, 0.15, 1.0))
            ax.zaxis.pane.set_facecolor((0.05, 0.09, 0.15, 1.0))
        except Exception:
            pass

    @staticmethod
    def _finite(points: np.ndarray) -> np.ndarray:
        p = np.asarray(points, dtype=float).reshape(-1, 3)
        return p[np.all(np.isfinite(p), axis=1)]

    @staticmethod
    def _set_equal_limits(axis, groups: list[np.ndarray]) -> None:
        valid = [g for g in groups if len(g)]
        if not valid:
            return
        p = np.vstack(valid)
        lo = np.min(p, axis=0)
        hi = np.max(p, axis=0)
        center = 0.5 * (lo + hi)
        span = max(float(np.max(hi - lo)), 80.0)
        half = 0.58 * span
        axis.set_xlim(center[0] - half, center[0] + half)
        axis.set_ylim(center[1] - half, center[1] + half)
        axis.set_zlim(center[2] - half, center[2] + half)
        try:
            axis.set_box_aspect((1, 1, 1))
        except Exception:
            pass

    @staticmethod
    def _draw_frame(axis, T: np.ndarray, length: float, forward_axis: str | None) -> list[np.ndarray]:
        T = np.asarray(T, dtype=float).reshape(4, 4)
        o = T[:3, 3]
        R = T[:3, :3]
        colors = ("#ef4444", "#22c55e", "#3b82f6")
        groups: list[np.ndarray] = [o.reshape(1, 3)]
        for i, color in enumerate(colors):
            e = o + R[:, i] * length
            axis.plot([o[0], e[0]], [o[1], e[1]], [o[2], e[2]], color=color, linewidth=2.6)
            groups.append(e.reshape(1, 3))
        if forward_axis in {"+z", "-z"}:
            sign = 1.0 if forward_axis == "+z" else -1.0
            e = o + sign * R[:, 2] * length * 1.35
            axis.plot(
                [o[0], e[0]], [o[1], e[1]], [o[2], e[2]],
                color="#c084fc", linewidth=2.4, linestyle="--",
            )
            axis.text(*e, f" forward {forward_axis.upper()}", color="#d8b4fe", fontsize=8)
            groups.append(e.reshape(1, 3))
        axis.scatter(*o, s=35, color="#f0abfc", depthshade=False)
        axis.text(*o, "  S", color="#f0abfc", fontsize=8, weight="bold")
        return groups

    def set_scans(
        self,
        scans: list[tuple[str, np.ndarray]],
        *,
        stable_transform: np.ndarray | None = None,
        forward_axis: str | None = None,
        frame_length: float = 70.0,
    ) -> None:
        total_points = sum(len(points) for _name, points in scans)
        title = f"{self.base_title}  ·  {len(scans)} lines / {total_points:,} points"
        self._reset_axis(title)
        ax = self.axis
        all_groups: list[np.ndarray] = []

        # World/base coordinate frame at the origin.
        origin = np.zeros(3)
        world_len = max(45.0, frame_length * 0.75)
        for i, color in enumerate(("#ef4444", "#22c55e", "#3b82f6")):
            e = origin.copy(); e[i] = world_len
            ax.plot([0, e[0]], [0, e[1]], [0, e[2]], color=color, linewidth=2.0, alpha=0.75)
        ax.text(0, 0, 0, "  BASE", color="#cbd5e1", fontsize=8)
        all_groups.append(np.vstack([origin, [world_len, world_len, world_len]]))

        for idx, (name, raw) in enumerate(scans):
            p = self._finite(raw)
            if len(p) < 2:
                continue
            all_groups.append(p)
            latest = idx == len(scans) - 1
            # Keep the original Keyence profile ordering.  A scan is rendered
            # as one connected world-coordinate line, not as an anonymous cloud.
            color = "#f59e0b" if latest else None
            ax.plot(
                p[:, 0], p[:, 1], p[:, 2],
                linewidth=3.6 if latest else 2.0,
                alpha=1.0 if latest else 0.72,
                color=color,
                solid_capstyle="round",
                label="latest scan" if latest else None,
            )
            # Endpoints make orientation/extent immediately visible.
            ax.scatter(
                [p[0, 0], p[-1, 0]],
                [p[0, 1], p[-1, 1]],
                [p[0, 2], p[-1, 2]],
                s=24 if latest else 12,
                color="#fde68a" if latest else None,
                alpha=0.95 if latest else 0.55,
                depthshade=False,
            )
            midpoint = p[len(p) // 2]
            if len(scans) <= 24:
                short = Path(name).stem.replace("capture_", "L")
                ax.text(*midpoint, f" {short}", fontsize=7, color="#f8fafc" if latest else "#94a3b8")

        if stable_transform is not None:
            all_groups.extend(
                self._draw_frame(
                    ax,
                    stable_transform,
                    float(frame_length),
                    forward_axis,
                )
            )

        if scans:
            ax.legend(loc="upper left", fontsize=8)
        self._set_equal_limits(ax, all_groups)
        ax.view_init(elev=24.0, azim=-55.0)
        self.canvas.draw_idle()


class PoseGeometryWindow(QtWidgets.QWidget):
    """Static 12-pose geometry in the Qt Matplotlib backend.

    No PyQtGraph OpenGL context is created, so MuJoCo remains the only native
    OpenGL/GLFW renderer in the workflow.
    """

    def __init__(self, result: dict[str, object], parent=None):
        window_flag = _qt_enum(QtCore.Qt, "WindowType", "Window")
        super().__init__(parent, flags=window_flag)
        self.setWindowTitle("12 calibration poses — geometry only")
        self.resize(1400, 900)

        outer = QtWidgets.QHBoxLayout(self)
        outer.setContentsMargins(12, 12, 12, 12)
        outer.setSpacing(12)

        plot_holder = QtWidgets.QWidget()
        plot_layout = QtWidgets.QVBoxLayout(plot_holder)
        plot_layout.setContentsMargins(0, 0, 0, 0)
        self.figure = Figure(figsize=(10, 8), dpi=100)
        self.figure.patch.set_facecolor("#0b1220")
        self.axis = self.figure.add_subplot(111, projection="3d")
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, plot_holder)
        plot_layout.addWidget(self.canvas, 1)
        plot_layout.addWidget(self.toolbar, 0)
        outer.addWidget(plot_holder, 4)

        info = QtWidgets.QFrame()
        info.setObjectName("sidePanel")
        info.setMinimumWidth(300)
        info_layout = QtWidgets.QVBoxLayout(info)
        info_layout.setContentsMargins(16, 16, 16, 16)

        forward_axis = str(result["forward_axis"])
        radius_mm = float(result["radius_mm"])
        offset = np.asarray(result["T_physical_measurement"], dtype=float)[:3, 3]

        title = QtWidgets.QLabel("POSE GEOMETRY")
        title.setObjectName("panelTitle")
        info_layout.addWidget(title)
        description = QtWidgets.QLabel(
            f"12 desired calibration poses\\n\\n"
            f"Forward axis: {forward_axis.upper()}\\n"
            f"Radius: {radius_mm:.2f} mm\\n"
            f"^P t_S: {np.round(offset, 2).tolist()} mm\\n\\n"
            "Geometry only: no IK / collision / path planning."
        )
        description.setWordWrap(True)
        info_layout.addWidget(description)
        legend = QtWidgets.QLabel(
            "Yellow: physical P\\nMagenta: measurement S\\nCyan: board target\\n"
            "RGB: TCP axes\\nPurple: sensor forward\\nBlack/white: board boundary + normal"
        )
        legend.setObjectName("legendText")
        info_layout.addSpacing(14)
        info_layout.addWidget(legend)
        info_layout.addStretch(1)
        outer.addWidget(info, 1)

        self._draw(result)

    @staticmethod
    def _equal(axis, groups: list[np.ndarray]) -> None:
        valid = [np.asarray(g, dtype=float).reshape(-1, 3) for g in groups if np.asarray(g).size]
        if not valid:
            return
        p = np.vstack(valid)
        lo, hi = np.min(p, axis=0), np.max(p, axis=0)
        c = 0.5 * (lo + hi)
        span = max(float(np.max(hi - lo)), 80.0)
        h = 0.58 * span
        axis.set_xlim(c[0]-h, c[0]+h); axis.set_ylim(c[1]-h, c[1]+h); axis.set_zlim(c[2]-h, c[2]+h)
        try: axis.set_box_aspect((1, 1, 1))
        except Exception: pass

    def _frame(self, T: np.ndarray, length: float) -> list[np.ndarray]:
        T = np.asarray(T, dtype=float).reshape(4, 4)
        o, R = T[:3, 3], T[:3, :3]
        groups = [o.reshape(1, 3)]
        for i, color in enumerate(("#ef4444", "#22c55e", "#3b82f6")):
            e = o + R[:, i] * length
            self.axis.plot([o[0], e[0]], [o[1], e[1]], [o[2], e[2]], color=color, linewidth=1.6, alpha=0.8)
            groups.append(e.reshape(1, 3))
        return groups

    def _draw(self, result: dict[str, object]) -> None:
        ax = self.axis
        ax.clear(); ax.set_facecolor("#0b1220")
        ax.set_title("12 desired calibration poses", color="#e5edf6")
        ax.set_xlabel("Base X [mm]", color="#cbd5e1"); ax.set_ylabel("Base Y [mm]", color="#cbd5e1"); ax.set_zlabel("Base Z [mm]", color="#cbd5e1")
        ax.tick_params(colors="#aab7c8", labelsize=8); ax.grid(True, alpha=.18)

        poses = result["poses"]
        center = np.asarray(result["center"], dtype=float)
        normal = np.asarray(result["normal"], dtype=float)
        boundary = np.asarray(result["boundary_world"], dtype=float)
        radius = float(result["radius_mm"])
        forward_axis = str(result["forward_axis"])
        sign = -1.0 if forward_axis == "-z" else 1.0
        groups: list[np.ndarray] = [center.reshape(1, 3)]

        if len(boundary):
            closed = np.vstack([boundary, boundary[0]])
            ax.plot(closed[:,0], closed[:,1], closed[:,2], color="#e2e8f0", linewidth=2.0, alpha=.8)
            groups.append(boundary)
        n_end = center + normal * max(60.0, radius * .65)
        ax.scatter(*center, s=42, color="#f8fafc", depthshade=False)
        ax.plot([center[0], n_end[0]], [center[1], n_end[1]], [center[2], n_end[2]], color="#f8fafc", linewidth=2.5)
        groups.append(n_end.reshape(1,3))

        p_all, s_all, t_all = [], [], []
        for pose in poses:
            T_tcp = np.asarray(pose.T_base_tcp, dtype=float)
            T_p = np.asarray(pose.T_base_physical, dtype=float)
            T_s = np.asarray(pose.T_base_sensor, dtype=float)
            groups.extend(self._frame(T_tcp, 35.0))
            p, s = T_p[:3,3], T_s[:3,3]
            forward = sign * T_p[:3,2]
            target = p + forward * float(pose.distance_mm)
            p_all.append(p); s_all.append(s); t_all.append(target)
            ax.plot([p[0], s[0]], [p[1], s[1]], [p[2], s[2]], color="#64748b", linewidth=1.0)
            f_end = p + forward * 75.0
            ax.plot([p[0], f_end[0]], [p[1], f_end[1]], [p[2], f_end[2]], color="#c084fc", linewidth=2.0)
            ax.plot([p[0], target[0]], [p[1], target[1]], [p[2], target[2]], color="#c084fc", linewidth=.8, alpha=.35, linestyle="--")
            ax.text(*p, f" P{pose.scan_id}", fontsize=7, color="#fde68a")
            groups.extend([p.reshape(1,3), s.reshape(1,3), target.reshape(1,3), f_end.reshape(1,3)])

        if p_all: ax.scatter(*np.asarray(p_all).T, s=35, color="#eab308", depthshade=False)
        if s_all: ax.scatter(*np.asarray(s_all).T, s=28, color="#d946ef", depthshade=False)
        if t_all: ax.scatter(*np.asarray(t_all).T, s=24, color="#22d3ee", depthshade=False)
        self._equal(ax, groups)
        ax.view_init(elev=24.0, azim=-55.0)
        self.canvas.draw_idle()


# ---------------------------------------------------------------------------
# Main workflow UI
# ---------------------------------------------------------------------------

class WorkflowWindow(QtWidgets.QMainWindow):
    """Qt workflow UI: PyQtGraph for live 2D, static Matplotlib for 3D, MuJoCo isolated."""

    UI_POLL_MS = 50
    VISUAL_REFRESH_MS = 100
    MONITOR_SAMPLE_S = 0.06

    def __init__(self, controller: WorkflowController, *, mode: str, robot: str):
        super().__init__()
        self.controller = controller
        self.mode = mode
        self.robot_name = robot

        self.setWindowTitle(
            f"Laser Hand-eye Workflow — {mode.upper()} / {robot.upper()}"
        )
        self.resize(1720, 1020)
        self.setMinimumSize(1250, 780)

        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.busy = False
        self.monitor_started = False
        self.preview_ready = False
        self.preview_speed = 1.0
        self.preview_speed_combos: list[QtWidgets.QComboBox] = []
        self.alive = True
        self._last_monitor_warning_at = 0.0
        self.action_buttons: list[QtWidgets.QPushButton] = []
        self.extra_windows: list[QtWidgets.QWidget] = []
        self.controller.motion_progress_callback = (
            lambda payload: self.events.put(("motion_progress", payload))
        )
        self._motion_phase = "idle"
        self._motion_elapsed_base_s = 0.0
        self._motion_elapsed_received_at = time.monotonic()
        self._motion_eta_deadline: float | None = None

        self.tcp_times: deque[float] = deque(maxlen=500)
        self.tcp_xyz: deque[np.ndarray] = deque(maxlen=500)
        self.monitor_started_at = time.monotonic()

        # Latest-only visualization buffer: no GUI backlog.
        self._visual_lock = threading.Lock()
        self._latest_visual: VisualizationSample | None = None
        self._latest_visual_version = 0
        self._drawn_visual_version = -1

        self._build_ui()
        self._apply_style()

        initial = (
            f"SIM ready · session={getattr(controller, 'session_directory', '')}"
            if mode == "sim"
            else "REAL ready · real motion is disabled"
        )
        self.set_status(initial, "ready")
        self._refresh_counts()
        # Show any existing captures immediately on launch.
        try:
            self.refresh_initial_world()
            self.refresh_scan_world()
            self.refresh_stage4_results(load_existing=True)
        except Exception as exc:
            self.statusBar().showMessage(f"World-view load warning: {exc}")
        self._set_actions_enabled(True)

        self.event_timer = QtCore.QTimer(self)
        self.event_timer.timeout.connect(self.poll_events)
        self.event_timer.start(self.UI_POLL_MS)

        self.visual_timer = QtCore.QTimer(self)
        self.visual_timer.timeout.connect(self.flush_latest_visual)
        self.visual_timer.start(self.VISUAL_REFRESH_MS)

        self.motion_clock_timer = QtCore.QTimer(self)
        self.motion_clock_timer.timeout.connect(self._tick_motion_clock)
        self.motion_clock_timer.start(100)

    # ------------------------------------------------------------------
    # Visual structure / design
    # ------------------------------------------------------------------

    def _apply_style(self) -> None:
        pg.setConfigOptions(antialias=False, background="#0b1220", foreground="#d7e0ea")
        self.setStyleSheet(
            """
            QMainWindow, QWidget {
                background: #0f172a;
                color: #e5edf6;
                font-size: 13px;
            }
            QFrame#header, QFrame#card, QFrame#sidePanel, QFrame#metricCard {
                background: #111c31;
                border: 1px solid #26364d;
                border-radius: 10px;
            }
            QLabel#appTitle {
                font-size: 20px;
                font-weight: 700;
            }
            QLabel#subtitle, QLabel#metricTitle {
                color: #93a4b8;
            }
            QLabel#metricValue {
                font-size: 18px;
                font-weight: 700;
            }
            QLabel#panelTitle {
                font-size: 15px;
                font-weight: 700;
                color: #dce8f5;
            }
            QLabel#legendText {
                color: #afbdd0;
                line-height: 1.35em;
            }
            QLabel[state="ready"], QLabel[state="neutral"] {
                background: #26364d;
                border-radius: 9px;
                padding: 4px 10px;
            }
            QLabel[state="ok"] {
                background: #123c2e;
                color: #86efac;
                border-radius: 9px;
                padding: 4px 10px;
                font-weight: 700;
            }
            QLabel[state="warn"] {
                background: #4a3515;
                color: #fde68a;
                border-radius: 9px;
                padding: 4px 10px;
                font-weight: 700;
            }
            QLabel[state="error"] {
                background: #4a1f27;
                color: #fda4af;
                border-radius: 9px;
                padding: 4px 10px;
                font-weight: 700;
            }
            QPushButton {
                background: #1c2b42;
                border: 1px solid #334963;
                border-radius: 7px;
                padding: 8px 13px;
                min-height: 20px;
                font-weight: 600;
            }
            QPushButton:hover {
                background: #263b59;
            }
            QPushButton:pressed {
                background: #152338;
            }
            QPushButton:disabled {
                color: #64748b;
                background: #172033;
                border-color: #243147;
            }
            QPushButton#primary {
                background: #155e75;
                border-color: #1f8aa5;
            }
            QPushButton#primary:hover {
                background: #18748e;
            }
            QPushButton#previewButton {
                background: #164e63;
                border: 2px solid #22d3ee;
                color: #cffafe;
                font-size: 14px;
                font-weight: 800;
            }
            QPushButton#previewButton:hover {
                background: #155e75;
            }
            QPushButton#executeButton {
                background: #9a3412;
                border: 2px solid #fb923c;
                color: #fff7ed;
                font-size: 14px;
                font-weight: 900;
            }
            QPushButton#executeButton:hover {
                background: #c2410c;
            }
            QProgressBar {
                background: #0b1322;
                border: 1px solid #334963;
                border-radius: 7px;
                text-align: center;
                min-height: 20px;
                font-weight: 800;
            }
            QProgressBar::chunk {
                background: #f97316;
                border-radius: 6px;
            }
            QPushButton#danger {
                background: #7f1d1d;
                border-color: #b33a3a;
                color: white;
            }
            QTabWidget::pane {
                border: 1px solid #26364d;
                border-radius: 8px;
                background: #0f172a;
            }
            QTabBar::tab {
                background: #111c31;
                border: 1px solid #26364d;
                padding: 10px 18px;
                margin-right: 3px;
                border-top-left-radius: 7px;
                border-top-right-radius: 7px;
            }
            QTabBar::tab:selected {
                background: #1c2b42;
                color: #7dd3fc;
                border-bottom-color: #1c2b42;
            }
            QLineEdit, QComboBox, QPlainTextEdit {
                background: #0b1322;
                border: 1px solid #334963;
                border-radius: 6px;
                padding: 7px;
            }
            QTableWidget {
                background: #0b1322;
                alternate-background-color: #101b2c;
                border: 1px solid #26364d;
                border-radius: 7px;
                gridline-color: #243349;
                selection-background-color: #234b63;
            }
            QHeaderView::section {
                background: #17243a;
                border: none;
                border-right: 1px solid #26364d;
                padding: 7px;
                font-weight: 700;
            }
            QCheckBox {
                spacing: 7px;
            }
            QStatusBar {
                background: #0b1322;
                color: #b8c5d6;
                border-top: 1px solid #26364d;
            }
            """
        )

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        outer = QtWidgets.QVBoxLayout(central)
        outer.setContentsMargins(10, 10, 10, 8)
        outer.setSpacing(9)

        outer.addWidget(self._build_header())

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.currentChanged.connect(self._on_tab_changed)
        outer.addWidget(self.tabs, 1)

        self._build_mount_verification()
        self._build_stage1()
        self._build_stage2()
        self._build_stage3()
        self._build_stage4()

        self.statusBar().showMessage("Ready")

    def _build_header(self) -> QtWidgets.QFrame:
        header = QtWidgets.QFrame()
        header.setObjectName("header")
        layout = QtWidgets.QHBoxLayout(header)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(10)

        title_box = QtWidgets.QVBoxLayout()
        title_box.setSpacing(0)
        title = QtWidgets.QLabel("Laser Hand-eye Calibration")
        title.setObjectName("appTitle")
        subtitle = QtWidgets.QLabel("Teach → estimate → plan → execute → calibrate")
        subtitle.setObjectName("subtitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        layout.addLayout(title_box)

        self.mode_pill = StatusPill(self.mode.upper(), "neutral")
        self.robot_pill = StatusPill(self.robot_name.upper(), "neutral")
        self.connection_pill = StatusPill("DISCONNECTED", "warn")
        layout.addWidget(self.mode_pill)
        layout.addWidget(self.robot_pill)
        layout.addWidget(self.connection_pill)
        if self.mode == "real":
            self.robot_connection_pill = StatusPill("ROBOT OFFLINE", "warn")
            self.laser_connection_pill = StatusPill("LASER OFFLINE", "warn")
            layout.addWidget(self.robot_connection_pill)
            layout.addWidget(self.laser_connection_pill)
        layout.addStretch(1)

        if self.mode == "real":
            self.robot_connect_button = self._button(
                "Connect Robot", self.toggle_robot_connection
            )
            self.robot_connect_button.setObjectName("primary")
            layout.addWidget(self.robot_connect_button)
            self.laser_connect_button = self._button(
                "Connect Laser", self.toggle_laser_connection
            )
            self.laser_connect_button.setObjectName("primary")
            layout.addWidget(self.laser_connect_button)
        else:
            self.connect_button = self._button("Connect", self.connect)
            self.connect_button.setObjectName("primary")
            layout.addWidget(self.connect_button)

        self.motion_enabled = QtWidgets.QCheckBox(
            "Enable real motion" if self.mode == "real" else "Enable simulated motion"
        )
        self.motion_enabled.setChecked(self.mode == "sim")
        self.motion_enabled.toggled.connect(self._motion_toggled)
        layout.addWidget(self.motion_enabled)

        self.stop_button = QtWidgets.QPushButton("STOP")
        self.stop_button.setObjectName("danger")
        self.stop_button.clicked.connect(self.stop)
        layout.addWidget(self.stop_button)

        return header

    def _button(
        self,
        text: str,
        command: Callable[[], None],
        *,
        primary: bool = False,
    ) -> QtWidgets.QPushButton:
        button = QtWidgets.QPushButton(text)
        if primary:
            button.setObjectName("primary")
        button.clicked.connect(command)
        self.action_buttons.append(button)
        return button

    def _preview_speed_combo(self) -> QtWidgets.QComboBox:
        """Create a synchronized, visualization-only playback speed selector."""
        combo = QtWidgets.QComboBox()
        for speed in (1.0, 2.0, 5.0, 10.0):
            combo.addItem(f"x{speed:g}", speed)
        combo.setCurrentIndex(combo.findData(self.preview_speed))
        combo.setMaximumWidth(82)
        combo.setToolTip(
            "MuJoCo preview playback only. Robot command timing is unchanged."
        )
        combo.currentIndexChanged.connect(
            lambda _index, source=combo: self._set_preview_speed(
                float(source.currentData())
            )
        )
        self.preview_speed_combos.append(combo)
        return combo

    def _set_preview_speed(self, speed: float) -> None:
        if not np.isfinite(speed) or speed <= 0.0:
            return
        self.preview_speed = float(speed)
        for combo in self.preview_speed_combos:
            index = combo.findData(self.preview_speed)
            if index < 0 or combo.currentIndex() == index:
                continue
            blocked = combo.blockSignals(True)
            combo.setCurrentIndex(index)
            combo.blockSignals(blocked)

    def _stage_container(self, step: str, title: str, description: str):
        page = QtWidgets.QWidget()
        root = QtWidgets.QVBoxLayout(page)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(10)

        heading = QtWidgets.QFrame()
        heading.setObjectName("card")
        h = QtWidgets.QHBoxLayout(heading)
        h.setContentsMargins(14, 10, 14, 10)

        step_label = QtWidgets.QLabel(step)
        step_label.setObjectName("panelTitle")
        title_box = QtWidgets.QVBoxLayout()
        title_text = QtWidgets.QLabel(title)
        title_text.setObjectName("panelTitle")
        desc = QtWidgets.QLabel(description)
        desc.setObjectName("subtitle")
        desc.setWordWrap(True)
        title_box.addWidget(title_text)
        title_box.addWidget(desc)

        h.addWidget(step_label)
        h.addSpacing(8)
        h.addLayout(title_box, 1)
        root.addWidget(heading)
        return page, root

    def _build_mount_verification(self) -> None:
        page, root = self._stage_container(
            "00",
            "Verify the real mounting against the digital model",
            "After Connect, manually articulate the robot with its pendant or freedrive. "
            "Measured joints/TCP drive the plots and the MuJoCo model in real time.",
        )
        self.tabs.addTab(page, "0  Mount check")
        self.mount_verification = MountVerificationView(self.controller.config)
        root.addWidget(self.mount_verification, 1)

    # ------------------------------------------------------------------
    # Stage 1
    # ------------------------------------------------------------------

    def _build_stage1(self) -> None:
        page, root = self._stage_container(
            "01",
            "Teach the plane and stable pose",
            "Capture several initial laser lines, estimate the board plane, then save a stable robot pose.",
        )
        self.tabs.addTab(page, "1  Setup")

        actions = QtWidgets.QHBoxLayout()
        actions.addWidget(self._button("Capture initial line", self.capture_initial, primary=True))
        actions.addWidget(self._button("Estimate plane", self.estimate_plane))
        actions.addWidget(self._button("Save stable pose", self.record_safe))
        actions.addWidget(self._button("Preview 12 pose geometry", self.preview_12_pose_geometry))
        actions.addStretch(1)
        root.addLayout(actions)

        metrics = QtWidgets.QHBoxLayout()
        self.initial_count_card = MetricCard("Initial lines")
        self.radius_card = MetricCard("Recommended radius")
        self.stable_card = MetricCard("Stable pose")
        metrics.addWidget(self.initial_count_card)
        metrics.addWidget(self.radius_card)
        metrics.addWidget(self.stable_card)
        metrics.addStretch(1)
        root.addLayout(metrics)

        splitter = QtWidgets.QSplitter()
        self.stage1_live = LivePlots()
        self.stage1_world = WorldScanView("Initial scans in BASE/WORLD")
        splitter.addWidget(self.stage1_live)
        splitter.addWidget(self.stage1_world)
        splitter.setSizes([620, 980])
        root.addWidget(splitter, 1)

    # ------------------------------------------------------------------
    # Stage 2
    # ------------------------------------------------------------------

    def _build_stage2(self) -> None:
        page, root = self._stage_container(
            "02",
            "Generate poses and validate collision-free paths",
            "MoveIt determines feasibility. MuJoCo is visualization only.",
        )
        self.tabs.addTab(page, "2  Plan")

        controls = QtWidgets.QFrame()
        controls.setObjectName("card")
        c = QtWidgets.QHBoxLayout(controls)
        c.setContentsMargins(12, 10, 12, 10)

        c.addWidget(QtWidgets.QLabel("Scan radius [mm]"))
        configured = self.controller.config.values["planning"].get("radius_mm")
        self.radius_edit = QtWidgets.QLineEdit("" if configured is None else str(configured))
        self.radius_edit.setMaximumWidth(110)
        c.addWidget(self.radius_edit)

        c.addWidget(QtWidgets.QLabel("Route mode"))
        self.route_mode_combo = QtWidgets.QComboBox()
        self.route_mode_combo.addItem(
            "Circular greedy (existing)", "circular_greedy"
        )
        self.route_mode_combo.addItem(
            "Global order + alpha", "global_alpha_dp"
        )
        configured_mode = str(
            self.controller.config.values["planning"].get(
                "route_mode", "circular_greedy"
            )
        ).strip().lower()
        configured_index = self.route_mode_combo.findData(configured_mode)
        if configured_index >= 0:
            self.route_mode_combo.setCurrentIndex(configured_index)
        self.route_mode_combo.setMaximumWidth(220)
        c.addWidget(self.route_mode_combo)

        self.recommended_radius_label = QtWidgets.QLabel("Estimate the plane first")
        self.recommended_radius_label.setObjectName("subtitle")
        c.addWidget(self.recommended_radius_label)
        c.addStretch(1)

        backend = str(self.controller.config.values["planning"].get("backend", "moveit"))
        c.addWidget(
            self._button(
                f"Generate 12 poses + validate ({backend})",
                self.create_plan,
                primary=True,
            )
        )
        root.addWidget(controls)

        full_preview_card = QtWidgets.QFrame()
        full_preview_card.setObjectName("card")
        full_preview_layout = QtWidgets.QHBoxLayout(full_preview_card)
        full_preview_layout.setContentsMargins(12, 8, 12, 8)
        full_preview_title = QtWidgets.QLabel("FULL PLAN PREVIEW")
        full_preview_title.setObjectName("panelTitle")
        full_preview_layout.addWidget(full_preview_title)
        full_preview_note = QtWidgets.QLabel(
            "Play every validated pose path continuously in MuJoCo"
        )
        full_preview_note.setObjectName("subtitle")
        full_preview_layout.addWidget(full_preview_note)
        full_preview_layout.addStretch(1)
        full_preview_layout.addWidget(QtWidgets.QLabel("Preview speed"))
        full_preview_layout.addWidget(self._preview_speed_combo())
        self.preview_full_button = self._button(
            "▶  Preview full plan",
            self.preview_full_plan,
        )
        self.preview_full_button.setObjectName("previewButton")
        full_preview_layout.addWidget(self.preview_full_button)
        root.addWidget(full_preview_card)

        splitter = QtWidgets.QSplitter()
        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)

        columns = [
            "Scan",
            "Status",
            "Support",
            "u",
            "v",
            "Tilt",
            "Distance",
            "Branch",
            "Joint Cost",
            "Samples",
            "Reason",
        ]
        self.plan_table = QtWidgets.QTableWidget(0, len(columns))
        self.plan_table.setHorizontalHeaderLabels(columns)
        self.plan_table.setAlternatingRowColors(True)
        self.plan_table.setSelectionBehavior(
            _qt_enum(QtWidgets.QAbstractItemView, "SelectionBehavior", "SelectRows")
        )
        self.plan_table.setSelectionMode(
            _qt_enum(QtWidgets.QAbstractItemView, "SelectionMode", "SingleSelection")
        )
        self.plan_table.setEditTriggers(
            _qt_enum(QtWidgets.QAbstractItemView, "EditTrigger", "NoEditTriggers")
        )
        _header_stretch(self.plan_table.horizontalHeader())
        self.plan_table.itemSelectionChanged.connect(self.on_plan_selection)
        left_layout.addWidget(self.plan_table)

        right = QtWidgets.QFrame()
        right.setObjectName("sidePanel")
        r = QtWidgets.QVBoxLayout(right)
        r.setContentsMargins(18, 18, 18, 18)
        selected_title = QtWidgets.QLabel("SELECTED POSE")
        selected_title.setObjectName("panelTitle")
        r.addWidget(selected_title)

        self.selected_pose_text = QtWidgets.QPlainTextEdit()
        self.selected_pose_text.setReadOnly(True)
        self.selected_pose_text.setPlainText(
            "Generate a plan and select a SAFE row.\n\n"
            "The path is no longer rendered into Tk/Matplotlib frames. "
            "Use the native MuJoCo viewer for smooth 3D inspection."
        )
        self.selected_pose_text.setMinimumHeight(260)
        r.addWidget(self.selected_pose_text, 1)

        selected_preview_options = QtWidgets.QHBoxLayout()
        selected_preview_options.addWidget(QtWidgets.QLabel("Preview speed"))
        selected_preview_options.addWidget(self._preview_speed_combo())
        selected_preview_options.addStretch(1)
        r.addLayout(selected_preview_options)

        self.open_selected_button = self._button(
            "Open selected path in MuJoCo",
            self.open_selected_3d,
            primary=True,
        )
        r.addWidget(self.open_selected_button)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([1150, 400])
        root.addWidget(splitter, 1)

    # ------------------------------------------------------------------
    # Stage 3
    # ------------------------------------------------------------------

    def _build_stage3(self) -> None:
        page, root = self._stage_container(
            "03",
            "Execute calibration scans",
            "Review the full execution route on the left. Preview and execute only the next planned step on the right.",
        )
        self.tabs.addTab(page, "3  Execute")

        splitter = QtWidgets.QSplitter()
        splitter.setOrientation(
            _qt_enum(QtCore.Qt, "Orientation", "Horizontal")
        )

        # --------------------------------------------------------------
        # LEFT: full route + progress
        # --------------------------------------------------------------
        left = QtWidgets.QWidget()
        left_layout = QtWidgets.QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)

        route_card = QtWidgets.QFrame()
        route_card.setObjectName("card")
        route_layout = QtWidgets.QVBoxLayout(route_card)
        route_layout.setContentsMargins(12, 10, 12, 10)
        route_layout.setSpacing(7)

        route_header = QtWidgets.QHBoxLayout()
        route_title = QtWidgets.QLabel("EXECUTION ROUTE")
        route_title.setObjectName("panelTitle")
        route_header.addWidget(route_title)
        route_header.addStretch(1)
        self.route_summary_label = QtWidgets.QLabel("Generate a plan first")
        self.route_summary_label.setObjectName("subtitle")
        self.route_summary_label.setWordWrap(True)
        route_header.addWidget(self.route_summary_label, 1)
        route_layout.addLayout(route_header)

        route_columns = ["Step", "Target", "Route", "Status"]
        self.execution_table = QtWidgets.QTableWidget(0, len(route_columns))
        self.execution_table.setHorizontalHeaderLabels(route_columns)
        self.execution_table.setAlternatingRowColors(True)
        self.execution_table.setSelectionBehavior(
            _qt_enum(QtWidgets.QAbstractItemView, "SelectionBehavior", "SelectRows")
        )
        self.execution_table.setSelectionMode(
            _qt_enum(QtWidgets.QAbstractItemView, "SelectionMode", "SingleSelection")
        )
        self.execution_table.setEditTriggers(
            _qt_enum(QtWidgets.QAbstractItemView, "EditTrigger", "NoEditTriggers")
        )
        _header_stretch(self.execution_table.horizontalHeader())
        route_layout.addWidget(self.execution_table, 1)
        left_layout.addWidget(route_card, 1)

        metrics = QtWidgets.QHBoxLayout()
        self.next_pose_card = MetricCard("Next step", "None")
        self.scan_count_card = MetricCard("Calibration scans", "0")
        self.plan_progress_card = MetricCard("Plan progress", "No plan")
        metrics.addWidget(self.next_pose_card)
        metrics.addWidget(self.scan_count_card)
        metrics.addWidget(self.plan_progress_card)
        left_layout.addLayout(metrics)

        # --------------------------------------------------------------
        # RIGHT: only next-step controls + live feedback
        # --------------------------------------------------------------
        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)

        action_card = QtWidgets.QFrame()
        action_card.setObjectName("card")
        a = QtWidgets.QVBoxLayout(action_card)
        a.setContentsMargins(12, 12, 12, 12)
        a.setSpacing(9)

        self.go_start_button = self._button(
            "⌂  GO TO START / SAFE",
            self.go_to_start,
        )
        self.go_start_button.setMinimumHeight(44)
        a.addWidget(self.go_start_button)

        preview_options = QtWidgets.QHBoxLayout()
        preview_options.addWidget(QtWidgets.QLabel("Preview speed"))
        preview_options.addWidget(self._preview_speed_combo())
        preview_options.addStretch(1)
        preview_only = QtWidgets.QLabel("viewer only · robot timing unchanged")
        preview_only.setObjectName("legendText")
        preview_options.addWidget(preview_only)
        a.addLayout(preview_options)

        self.preview_full_execute_button = self._button(
            "👁  FULL PLAN VIEW",
            self.preview_full_plan,
        )
        self.preview_full_execute_button.setObjectName("previewButton")
        self.preview_full_execute_button.setMinimumHeight(44)
        a.addWidget(self.preview_full_execute_button)

        self.preview_next_button = self._button(
            "👁  NEXT VIEW",
            self.preview_next,
        )
        self.preview_next_button.setObjectName("previewButton")
        self.preview_next_button.setMinimumHeight(52)
        a.addWidget(self.preview_next_button)

        self.next_button = self._button(
            "🤖  EXECUTE",
            self.next_scan,
        )
        self.next_button.setObjectName("executeButton")
        self.next_button.setMinimumHeight(58)
        a.addWidget(self.next_button)

        right_layout.addWidget(action_card)

        motion_card = QtWidgets.QFrame()
        motion_card.setObjectName("card")
        motion_layout = QtWidgets.QVBoxLayout(motion_card)
        motion_layout.setContentsMargins(14, 10, 14, 10)
        motion_layout.setSpacing(6)
        motion_header = QtWidgets.QHBoxLayout()
        self.motion_state_label = QtWidgets.QLabel("● IDLE · waiting for preview")
        self.motion_state_label.setObjectName("panelTitle")
        self.motion_time_label = QtWidgets.QLabel("Elapsed —  ·  Remaining —")
        self.motion_time_label.setObjectName("subtitle")
        motion_header.addWidget(self.motion_state_label)
        motion_header.addStretch(1)
        motion_header.addWidget(self.motion_time_label)
        motion_layout.addLayout(motion_header)
        self.motion_progress = QtWidgets.QProgressBar()
        self.motion_progress.setRange(0, 100)
        self.motion_progress.setValue(0)
        self.motion_progress.setFormat("0%")
        motion_layout.addWidget(self.motion_progress)
        self.motion_detail_label = QtWidgets.QLabel(
            "NEXT VIEW shows the exact next planned trajectory. "
            "EXECUTE runs that step only."
        )
        self.motion_detail_label.setObjectName("legendText")
        self.motion_detail_label.setWordWrap(True)
        motion_layout.addWidget(self.motion_detail_label)
        right_layout.addWidget(motion_card)

        live_splitter = QtWidgets.QSplitter()
        live_splitter.setOrientation(
            _qt_enum(QtCore.Qt, "Orientation", "Vertical")
        )
        self.stage3_live = LivePlots()
        self.stage3_world = WorldScanView("Calibration scans in BASE/WORLD")
        live_splitter.addWidget(self.stage3_live)
        live_splitter.addWidget(self.stage3_world)
        live_splitter.setSizes([330, 420])
        right_layout.addWidget(live_splitter, 1)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setSizes([760, 900])

        root.addWidget(splitter, 1)

    # ------------------------------------------------------------------
    # Stage 4
    # ------------------------------------------------------------------

    def _build_stage4(self) -> None:
        page, root = self._stage_container(
            "04",
            "Extra scans and final calibration",
            "Add manually taught measurements if needed, then solve hand-eye calibration using the accumulated dataset.",
        )
        self.tabs.addTab(page, "4  Calibrate")

        actions = QtWidgets.QHBoxLayout()
        actions.addWidget(
            self._button(
                "Load manual dataset…",
                self.choose_calibration_dataset,
            )
        )
        actions.addWidget(
            self._button(
                "Capture additional manual scan",
                self.capture_additional,
            )
        )
        actions.addWidget(
            self._button(
                "Run hand-eye calibration",
                self.calibrate,
                primary=True,
            )
        )
        actions.addStretch(1)
        root.addLayout(actions)

        dataset_row = QtWidgets.QHBoxLayout()
        dataset_row.addWidget(QtWidgets.QLabel("Calibration dataset"))
        self.stage4_dataset_path = QtWidgets.QLineEdit()
        self.stage4_dataset_path.setReadOnly(True)
        dataset_row.addWidget(self.stage4_dataset_path, 1)
        root.addLayout(dataset_row)

        splitter = QtWidgets.QSplitter()
        self.stage4_world = WorldScanView("Accumulated calibration profiles")
        splitter.addWidget(self.stage4_world)

        result_panel = QtWidgets.QWidget()
        result_layout = QtWidgets.QVBoxLayout(result_panel)
        result_layout.setContentsMargins(4, 0, 0, 0)

        self.stage4_rms_plot = pg.PlotWidget(title="Plane RMS by calibration iteration")
        self.stage4_rms_plot.setLabel("bottom", "Iteration")
        self.stage4_rms_plot.setLabel("left", "Plane RMS", units="mm")
        self.stage4_rms_plot.showGrid(x=True, y=True, alpha=0.25)
        self.stage4_rms_curve = self.stage4_rms_plot.plot(
            pen=pg.mkPen("#22d3ee", width=2.5),
            symbol="o",
            symbolSize=6,
            symbolBrush="#f59e0b",
        )
        result_layout.addWidget(self.stage4_rms_plot, 1)

        matrices = QtWidgets.QHBoxLayout()
        initial_box = QtWidgets.QVBoxLayout()
        initial_box.addWidget(QtWidgets.QLabel("Initial T_tcp_sensor"))
        self.stage4_initial_matrix = QtWidgets.QPlainTextEdit()
        self.stage4_initial_matrix.setReadOnly(True)
        self.stage4_initial_matrix.setMaximumHeight(145)
        initial_box.addWidget(self.stage4_initial_matrix)
        matrices.addLayout(initial_box, 1)
        matrix_arrow = QtWidgets.QLabel("→")
        matrix_arrow.setAlignment(_qt_enum(QtCore.Qt, "AlignmentFlag", "AlignCenter"))
        matrices.addWidget(matrix_arrow)
        final_box = QtWidgets.QVBoxLayout()
        final_box.addWidget(QtWidgets.QLabel("Final T_tcp_sensor"))
        self.stage4_final_matrix = QtWidgets.QPlainTextEdit()
        self.stage4_final_matrix.setReadOnly(True)
        self.stage4_final_matrix.setMaximumHeight(145)
        final_box.addWidget(self.stage4_final_matrix)
        matrices.addLayout(final_box, 1)
        result_layout.addLayout(matrices)

        self.stage4_message = QtWidgets.QPlainTextEdit()
        self.stage4_message.setReadOnly(True)
        self.stage4_message.setMaximumHeight(105)
        self.stage4_message.setPlainText(
            "Select any manual capture folder containing capture_*.npz, or use the "
            "workflow scan dataset. Stages 2 and 3 are not required for manual data."
        )
        result_layout.addWidget(self.stage4_message)
        splitter.addWidget(result_panel)
        splitter.setSizes([900, 760])
        root.addWidget(splitter, 1)

        paths = QtWidgets.QFormLayout()
        self.stage4_accumulated_path = QtWidgets.QLineEdit()
        self.stage4_accumulated_path.setReadOnly(True)
        paths.addRow("Accumulated scan file", self.stage4_accumulated_path)
        self.stage4_output_path = QtWidgets.QLineEdit()
        self.stage4_output_path.setReadOnly(True)
        paths.addRow("Final matrix file", self.stage4_output_path)
        root.addLayout(paths)

    # ------------------------------------------------------------------
    # Background execution / events
    # ------------------------------------------------------------------

    def run_background(self, label, function, on_success=None, on_error=None) -> None:
        if self.busy:
            return
        self.busy = True
        self.set_status(label, "busy")
        self._set_actions_enabled(False)

        def target():
            try:
                result = function()
                self.events.put(("success", (label, result, on_success)))
            except BaseException as error:
                self.events.put(("error", (label, error, on_error)))

        threading.Thread(target=target, daemon=True).start()

    def poll_events(self) -> None:
        try:
            while True:
                kind, payload = self.events.get_nowait()

                if kind == "viewer_error":
                    self.set_status(f"MuJoCo viewer error: {payload}", "error")
                    _message(self, "MuJoCo viewer", str(payload), error=True)
                    continue

                if kind == "monitor_warning":
                    message = str(payload)
                    self.statusBar().showMessage(f"Live-state warning · {message}")
                    if self.tabs.currentIndex() == 0:
                        self.mount_verification.model_status.setText(
                            "Live robot state update failed.\n" + message
                        )
                    continue

                if kind == "motion_progress":
                    self._update_motion_progress(payload)
                    continue

                self.busy = False
                self._set_actions_enabled(True)

                if kind == "error":
                    label, error, callback = payload
                    if callback is not None:
                        callback(error)
                    self.set_status(f"FAILED · {label}: {error}", "error")
                    _message(self, "Workflow error", f"{label}\n\n{error}", error=True)
                else:
                    label, result, callback = payload
                    if callback is not None:
                        callback(result)
                    self.set_status(f"Complete · {label}", "ok")
        except queue.Empty:
            pass

    @staticmethod
    def _format_seconds(value) -> str:
        if value is None:
            return "—"
        seconds = max(0.0, float(value))
        if seconds < 60.0:
            return f"{seconds:.1f} s"
        minutes = int(seconds // 60)
        return f"{minutes}m {seconds - 60 * minutes:.0f}s"

    def _update_motion_progress(self, payload: object) -> None:
        if not isinstance(payload, dict):
            return
        phase = str(payload.get("phase", "moving"))
        label = str(payload.get("label", "Robot motion"))
        fraction = float(payload.get("fraction", 0.0) or 0.0)
        fraction = min(1.0, max(0.0, fraction))
        completed = int(payload.get("completed", 0) or 0)
        total = int(payload.get("total", 0) or 0)
        elapsed = payload.get("elapsed_s")
        remaining = payload.get("remaining_s")

        if phase == "moving":
            icon = "🤖"
            state = "MOVING"
            detail = (
                f"Full trajectory active · {completed}/{total} timed segments"
                if total > 0
                else "Full FollowJointTrajectory command active"
            )
        elif phase == "settling":
            icon = "⏳"
            state = "SETTLING"
            detail = "Robot reached the target · waiting for mechanical settling"
        elif phase == "capture":
            icon = "📡"
            state = "CAPTURING"
            detail = "Robot motion complete · acquiring the laser profile"
        else:
            icon = "●"
            state = phase.upper()
            detail = label

        self._motion_phase = phase
        self._motion_elapsed_base_s = 0.0 if elapsed is None else float(elapsed)
        self._motion_elapsed_received_at = time.monotonic()
        self._motion_eta_deadline = (
            None
            if remaining is None
            else time.monotonic() + max(0.0, float(remaining))
        )

        self.motion_state_label.setText(f"{icon} {state} · {label}")
        self.motion_detail_label.setText(detail)
        self.motion_progress.setValue(int(round(100.0 * fraction)))
        self.motion_progress.setFormat(f"{100.0 * fraction:.0f}%")
        self._tick_motion_clock()

    def _tick_motion_clock(self) -> None:
        if not hasattr(self, "motion_time_label"):
            return
        if self._motion_phase not in {"moving", "settling"}:
            return
        now = time.monotonic()
        elapsed = self._motion_elapsed_base_s + max(
            0.0, now - self._motion_elapsed_received_at
        )
        remaining = (
            None
            if self._motion_eta_deadline is None
            else max(0.0, self._motion_eta_deadline - now)
        )
        self.motion_time_label.setText(
            f"Elapsed {self._format_seconds(elapsed)}  ·  "
            f"Remaining ≈ {self._format_seconds(remaining)}"
        )

    def _motion_complete(self, text: str = "Sequence complete") -> None:
        self._motion_phase = "complete"
        self._motion_eta_deadline = None
        self.motion_state_label.setText(f"✅ COMPLETE · {text}")
        self.motion_progress.setValue(100)
        self.motion_progress.setFormat("100%")
        self.motion_time_label.setText("Elapsed —  ·  Remaining 0.0 s")
        self.motion_detail_label.setText("Robot command finished successfully.")

    def _motion_failed(self, error) -> None:
        self._motion_phase = "failed"
        self._motion_eta_deadline = None
        self.motion_state_label.setText("⚠ MOTION FAILED")
        self.motion_progress.setFormat("FAILED")
        self.motion_time_label.setText("Check the workflow error dialog")
        self.motion_detail_label.setText(str(error))

    def set_status(self, text: str, state: str = "ready") -> None:
        self.statusBar().showMessage(text)
        if state == "error":
            self.connection_pill.set_state("ERROR", "error")

    def _set_actions_enabled(self, enabled: bool) -> None:
        for button in self.action_buttons:
            button.setEnabled(enabled)
        self.motion_enabled.setEnabled(enabled)

        if enabled and not self.preview_ready:
            self.next_button.setEnabled(False)

        self.stop_button.setEnabled(True)

    # ------------------------------------------------------------------
    # Live monitor
    # ------------------------------------------------------------------

    def start_monitor(self) -> None:
        if self.monitor_started:
            return
        self.monitor_started = True
        self.monitor_started_at = time.monotonic()

        def monitor():
            while self.alive:
                if self.mode == "real" and self.controller.robot is None:
                    time.sleep(self.MONITOR_SAMPLE_S)
                    continue
                try:
                    sample = self.controller.read_visualization_sample()
                    if sample is not None:
                        with self._visual_lock:
                            self._latest_visual = sample
                            self._latest_visual_version += 1
                except BaseException as error:
                    now = time.monotonic()
                    if now - self._last_monitor_warning_at >= 2.0:
                        self._last_monitor_warning_at = now
                        self.events.put(("monitor_warning", str(error)))
                time.sleep(self.MONITOR_SAMPLE_S)

        threading.Thread(target=monitor, daemon=True).start()

    def flush_latest_visual(self) -> None:
        if not self.monitor_started:
            return

        with self._visual_lock:
            version = self._latest_visual_version
            sample = self._latest_visual

        if sample is None or version == self._drawn_visual_version:
            return

        self._drawn_visual_version = version
        self.update_live(sample)

    def update_live(self, sample: VisualizationSample) -> None:
        points = np.asarray(sample.points_s, dtype=float)
        T = np.asarray(sample.T_base_tcp, dtype=float)

        elapsed = time.monotonic() - self.monitor_started_at
        xyz = T[:3, 3].copy()
        rpy = Rotation.from_matrix(T[:3, :3]).as_euler("xyz", degrees=True)

        self.tcp_times.append(elapsed)
        self.tcp_xyz.append(xyz)
        timeline = np.asarray(self.tcp_times, dtype=float)
        history = np.asarray(self.tcp_xyz, dtype=float)

        # Only render the currently visible stage. Hidden tabs do zero plot work.
        current = self.tabs.currentIndex()
        if current == 0 and sample.joint_positions_deg is not None:
            try:
                self.mount_verification.update_state(
                    elapsed,
                    np.asarray(sample.joint_positions_deg, dtype=float),
                    T,
                )
            except BaseException as error:
                self.mount_verification.model_status.setText(
                    f"Live model update failed: {error}"
                )
        elif current == 1:
            self.stage1_live.update_data(points, timeline, history, xyz, rpy)
        elif current == 3:
            self.stage3_live.update_data(points, timeline, history, xyz, rpy)

    def _on_tab_changed(self, index: int) -> None:
        # Force one redraw with the newest sample when entering a live tab.
        if index in {0, 1, 3}:
            self._drawn_visual_version = -1

    # ------------------------------------------------------------------
    # Stage 1 actions
    # ------------------------------------------------------------------

    def connect(self) -> None:
        def connected(_):
            self.connection_pill.set_state("CONNECTED", "ok")
            if self.mode == "real":
                self._set_motion_checked(False)
                self.set_status(self.controller.connection_report, "ok")
            self.start_monitor()

        self.run_background("Connecting robot and laser", self.controller.connect, connected)

    def _refresh_connection_status(self) -> None:
        if self.mode != "real":
            return
        robot_connected = self.controller.robot is not None
        laser_connected = self.controller.laser is not None
        self.robot_connection_pill.set_state(
            "ROBOT CONNECTED" if robot_connected else "ROBOT OFFLINE",
            "ok" if robot_connected else "warn",
        )
        self.laser_connection_pill.set_state(
            "LASER CONNECTED" if laser_connected else "LASER OFFLINE",
            "ok" if laser_connected else "warn",
        )
        self.robot_connect_button.setText(
            "Disconnect Robot" if robot_connected else "Connect Robot"
        )
        self.laser_connect_button.setText(
            "Disconnect Laser" if laser_connected else "Connect Laser"
        )
        if robot_connected and laser_connected:
            self.connection_pill.set_state("CONNECTED", "ok")
        elif robot_connected:
            self.connection_pill.set_state("ROBOT ONLY", "warn")
        elif laser_connected:
            self.connection_pill.set_state("LASER ONLY", "warn")
        else:
            self.connection_pill.set_state("DISCONNECTED", "warn")

    def toggle_robot_connection(self) -> None:
        if self.controller.robot is not None:
            self.run_background(
                "Disconnecting robot",
                self.controller.disconnect_robot,
                lambda _: (
                    self._set_motion_checked(False),
                    self._refresh_connection_status(),
                ),
                lambda _: self._refresh_connection_status(),
            )
            return

        def connected(_):
            self._set_motion_checked(False)
            self._refresh_connection_status()
            self.set_status(self.controller.robot_connection_report, "ok")
            self.start_monitor()

        self.run_background(
            "Connecting robot",
            self.controller.connect_robot,
            connected,
            lambda _: self._refresh_connection_status(),
        )

    def toggle_laser_connection(self) -> None:
        if self.controller.laser is not None:
            self.run_background(
                "Disconnecting laser",
                self.controller.disconnect_laser,
                lambda _: self._refresh_connection_status(),
                lambda _: self._refresh_connection_status(),
            )
            return

        def connected(_):
            self._refresh_connection_status()
            self.set_status(self.controller.laser_connection_report, "ok")
            if self.controller.robot is not None:
                self.start_monitor()

        self.run_background(
            "Connecting laser",
            self.controller.connect_laser,
            connected,
            lambda _: self._refresh_connection_status(),
        )

    def capture_initial(self) -> None:
        self.run_background(
            "Capturing initial line",
            self.controller.capture_initial_line,
            lambda _: (self.refresh_initial_world(), self._refresh_counts()),
        )

    def estimate_plane(self) -> None:
        self.run_background(
            "Estimating plane from initial lines",
            self.controller.estimate_plane,
            lambda _: self.update_recommended_radius(),
        )

    def record_safe(self) -> None:
        self.preview_ready = False
        self.next_button.setEnabled(False)

        def success(_):
            self.stable_card.set_value("Saved")
            self.refresh_initial_world()

        self.run_background(
            "Saving current TCP/joints as stable pose",
            self.controller.record_safe_pose,
            success,
        )

    def update_recommended_radius(self) -> None:
        radius = float(self.controller.recommended_radius_mm())
        self.recommended_radius_label.setText(f"Recommended: {radius:.2f} mm")
        self.radius_card.set_value(f"{radius:.2f} mm")
        if not self.radius_edit.text().strip():
            self.radius_edit.setText(f"{radius:.2f}")

    # ------------------------------------------------------------------
    # 12-pose geometry
    # ------------------------------------------------------------------

    def preview_12_pose_geometry(self) -> None:
        try:
            text = self.radius_edit.text().strip()
            radius = float(text) if text else float(self.controller.recommended_radius_mm())
        except (ValueError, OSError, RuntimeError) as error:
            _message(
                self,
                "Cannot preview poses",
                "Estimate the plane first and enter a valid scan radius.\n\n"
                f"{error}",
                error=True,
            )
            return

        if not np.isfinite(radius) or radius <= 0.0:
            _message(self, "Invalid radius", "Scan radius must be positive.", error=True)
            return

        self.run_background(
            "Generating 12 pose geometry",
            lambda: self._build_12_pose_geometry(radius),
            self._show_12_pose_geometry,
        )

    def _build_12_pose_geometry(self, radius_mm: float) -> dict[str, object]:
        planner = ScanPlanner(self.controller.config)
        boundary_uv, T_world_plane, metadata = planner._load_plane_geometry()

        center = np.asarray(metadata["centroid_w_mm"], dtype=float).reshape(3)
        normal = np.asarray(metadata["normal_w"], dtype=float).reshape(3)
        normal /= np.linalg.norm(normal)

        config = self.controller.config
        planning = config.values["planning"]
        forward_axis = str(planning.get("sensor_forward_axis", "+z")).strip().lower()
        if forward_axis not in {"+z", "-z"}:
            raise ValueError(
                "planning.sensor_forward_axis must be '+z' or '-z'"
            )

        T_tcp_sensor = _load_transform(config.path("handeye"))
        T_physical_measurement = config.T_physical_measurement_mm

        poses = generate_optimal_tcp_poses(
            board_center_base_mm=center,
            board_normal_base=normal,
            T_tcp_sensor_init=T_tcp_sensor,
            radius_mm=float(radius_mm),
            tilt_min_deg=float(planning.get("tilt_min_deg", 5.0)),
            tilt_max_deg=float(planning.get("tilt_max_deg", 40.0)),
            distance_near_mm=float(planning.get("distance_near_mm", 60.0)),
            distance_far_mm=float(planning.get("distance_far_mm", 120.0)),
            sensor_forward_axis=forward_axis,
            T_physical_measurement=T_physical_measurement,
        )

        boundary_uv = np.asarray(boundary_uv, dtype=float).reshape(-1, 2)
        T_world_plane = np.asarray(T_world_plane, dtype=float).reshape(4, 4)
        local_boundary = np.column_stack(
            [boundary_uv, np.zeros(len(boundary_uv)), np.ones(len(boundary_uv))]
        )
        boundary_world = (T_world_plane @ local_boundary.T).T[:, :3]

        return {
            "poses": poses,
            "center": center,
            "normal": normal,
            "boundary_world": boundary_world,
            "forward_axis": forward_axis,
            "radius_mm": float(radius_mm),
            "T_physical_measurement": T_physical_measurement,
            "T_tcp_sensor": T_tcp_sensor,
        }

    def _show_12_pose_geometry(self, result: dict[str, object]) -> None:
        window = PoseGeometryWindow(result, self)
        window.setStyleSheet(self.styleSheet())
        window.show()
        self.extra_windows.append(window)

    # ------------------------------------------------------------------
    # Stage 2 planning
    # ------------------------------------------------------------------

    def create_plan(self) -> None:
        try:
            radius = float(self.radius_edit.text())
        except ValueError:
            _message(self, "Invalid radius", "Enter a positive radius in mm.", error=True)
            return

        if not np.isfinite(radius) or radius <= 0:
            _message(self, "Invalid radius", "Enter a positive radius in mm.", error=True)
            return

        route_mode = self.route_mode_combo.currentData()
        self.controller.config.values["planning"]["route_mode"] = str(
            route_mode
        )

        self.preview_ready = False
        self.next_button.setEnabled(False)

        self.run_background(
            "Generating 12 poses and validating paths",
            lambda: self.controller.create_plan(radius),
            self.show_plan,
        )

    def show_plan(self, plan: ScanPlan | None) -> None:
        if plan is None:
            return

        self.plan_table.setRowCount(0)

        for scan in plan.scans:
            p = scan.pose
            row = self.plan_table.rowCount()
            self.plan_table.insertRow(row)
            joint_cost_text = (
                "—"
                if scan.approach_joint_cost_deg is None
                else f"{scan.approach_joint_cost_deg:.1f}°"
            )
            values = [
                p.scan_id,
                scan.status,
                p.support_id,
                f"{p.target_u_mm:.1f}",
                f"{p.target_v_mm:.1f}",
                f"{p.tilt_deg:.1f}",
                f"{p.distance_mm:.1f}",
                scan.selected_branch or "—",
                joint_cost_text,
                scan.path_sample_count,
                scan.reason,
            ]
            for column, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(str(value))
                self.plan_table.setItem(row, column, item)

        safe = sum(scan.status == "SAFE" for scan in plan.scans)
        scanned = sum(scan.status == "SCANNED" for scan in plan.scans)

        self.recommended_radius_label.setText(
            f"Recommended: {plan.recommended_radius_mm:.2f} mm"
        )
        self.plan_progress_card.set_value(f"{scanned} scanned / {safe} ready")
        self.set_status(
            f"Plan ready · {plan.planner_backend}/{plan.route_mode} · "
            f"SAFE={safe}/12 · radius={plan.radius_mm:.2f} mm",
            "ok",
        )

        if plan.radius_warning:
            QtWidgets.QMessageBox.warning(self, "Radius outside observed hull", plan.radius_warning)

        self._refresh_execution_route()

    def _refresh_execution_route(self) -> None:
        plan = self.controller.plan
        self.execution_table.setRowCount(0)
        if plan is None or not getattr(plan, "execution_steps", None):
            self.route_summary_label.setText("No execution route")
            self.next_pose_card.set_value("None")
            self.next_button.setText("🤖  EXECUTE")
            self.next_button.setEnabled(False)
            if hasattr(self, "go_start_button"):
                self.go_start_button.setEnabled(False)
            return

        steps = plan.execution_steps
        current = int(getattr(plan, "execution_step_index", 0))
        route_tokens = ["START"]
        for step in steps:
            route_tokens.append(step.target_label)
        self.route_summary_label.setText("  →  ".join(route_tokens))

        for index, step in enumerate(steps):
            row = self.execution_table.rowCount()
            self.execution_table.insertRow(row)
            status = "NEXT" if index == current and step.status != "DONE" else step.status
            values = [
                step.step_id,
                step.target_label,
                step.route_label,
                status,
            ]
            for column, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(str(value))
                self.execution_table.setItem(row, column, item)
            if index == current:
                self.execution_table.selectRow(row)

        if hasattr(self, "go_start_button"):
            self.go_start_button.setEnabled(
                current == 0 and not self.busy
            )

        if current < len(steps):
            step = steps[current]
            self.next_pose_card.set_value(step.target_label)
            self.next_button.setText(f"🤖  EXECUTE · {step.target_label}")
            self.next_button.setEnabled(self.preview_ready and not self.busy)
        else:
            self.next_pose_card.set_value("Complete")
            self.next_button.setText("✅  COMPLETE")
            self.next_button.setEnabled(False)

    def selected_scan_id(self) -> int | None:
        rows = self.plan_table.selectionModel().selectedRows()
        if not rows:
            return None
        item = self.plan_table.item(rows[0].row(), 0)
        return None if item is None else int(item.text())

    def on_plan_selection(self) -> None:
        scan_id = self.selected_scan_id()
        if scan_id is None:
            return

        plan = self.controller.plan
        scan = next(
            (s for s in (plan.scans if plan else []) if s.pose.scan_id == scan_id),
            None,
        )
        if scan is None:
            return

        p = scan.pose
        joint_cost_text = (
            "—"
            if scan.approach_joint_cost_deg is None
            else f"{scan.approach_joint_cost_deg:.2f}°"
        )
        self.selected_pose_text.setPlainText(
            f"Scan {scan_id} · {scan.status}\n\n"
            f"Support: {p.support_id}\n"
            f"Target UV: ({p.target_u_mm:.1f}, {p.target_v_mm:.1f}) mm\n"
            f"Tilt: {p.tilt_deg:.1f}°\n"
            f"Distance: {p.distance_mm:.1f} mm\n"
            f"Branch: {scan.selected_branch or '—'}\n"
            f"Weighted joint cost: {joint_cost_text}\n"
            f"Preview samples: {scan.path_sample_count}\n\n"
            f"{scan.reason}"
        )
        self.open_selected_button.setEnabled(
            scan.preview_qpos_trajectory is not None
            and scan.preview_time_from_start_s is not None
            and not self.busy
        )

    def open_selected_3d(self) -> None:
        scan_id = self.selected_scan_id()
        if scan_id is None:
            _message(self, "Select pose", "Select a SAFE pose first.")
            return
        self.open_interactive_viewer(scan_id)

    # ------------------------------------------------------------------
    # Native MuJoCo viewer
    # ------------------------------------------------------------------

    def open_interactive_viewer(self, scan_id: int) -> None:
        """Launch MuJoCo in a completely separate Python process.

        Qt/PyQtGraph never shares an OpenGL/GLFW context with MuJoCo.  This is
        intentionally process-isolated because a native GL driver crash cannot
        be caught by Python exceptions.
        """
        plan = self.controller.plan
        if plan is None:
            _message(self, "No plan", "Generate a scan plan first.", error=True)
            return
        scan = next((s for s in plan.scans if s.pose.scan_id == int(scan_id)), None)
        if (
            scan is None
            or scan.preview_qpos_trajectory is None
            or scan.preview_time_from_start_s is None
        ):
            _message(
                self,
                "No preview",
                f"Scan {scan_id} has no timed MuJoCo preview trajectory.",
                error=True,
            )
            return

        self.open_trajectory_viewer(
            scan.preview_qpos_trajectory,
            scan.preview_time_from_start_s,
            filename=f"scan_{int(scan_id):04d}_timed_preview.npz",
            title=f"Scan {int(scan_id)} collision-checked path",
        )

    def open_trajectory_viewer(
        self,
        trajectory: np.ndarray,
        time_from_start_s: np.ndarray,
        *,
        filename: str,
        title: str,
    ) -> None:
        """Launch MuJoCo at the selected preview-only playback multiplier."""
        plan = self.controller.plan
        if plan is None:
            raise RuntimeError("generate a scan plan first")

        qpos = np.asarray(trajectory, dtype=float)
        times = np.asarray(time_from_start_s, dtype=float).reshape(-1)
        if qpos.ndim != 2 or len(qpos) == 0:
            raise ValueError("MuJoCo preview trajectory must be a non-empty 2D array")
        if times.shape != (len(qpos),):
            raise ValueError("MuJoCo preview timing does not match qpos trajectory")
        if not np.all(np.isfinite(qpos)) or not np.all(np.isfinite(times)):
            raise ValueError("MuJoCo preview data contains non-finite values")
        if len(times) > 1 and np.any(np.diff(times) <= 0.0):
            raise ValueError("MuJoCo preview timing must be strictly increasing")

        preview_dir = (
            self.controller.config.path("scan_plan").parent
            / "viewer_previews"
        )
        preview_dir.mkdir(parents=True, exist_ok=True)
        trajectory_path = preview_dir / filename
        np.savez_compressed(
            trajectory_path,
            qpos=qpos,
            time_s=times,
        )

        command = [
            sys.executable,
            "-m",
            "real_laser_handeye.workflow_mujoco_viewer",
            "--model", str(plan.board_model_path),
            "--trajectory", str(trajectory_path),
            "--title", title,
            "--playback-speed", f"{self.preview_speed:g}",
        ]
        try:
            subprocess.Popen(command, cwd=str(Path.cwd()))
            self.set_status(
                f"MuJoCo viewer launched · {title} · x{self.preview_speed:g} preview",
                "ok",
            )
        except BaseException as error:
            self.events.put(("viewer_error", error))

    def preview_full_plan(self) -> None:
        if self.controller.plan is None:
            _message(
                self,
                "No plan",
                "Generate a validated scan plan first.",
                error=True,
            )
            return

        def success(result):
            trajectory, time_from_start_s, step_count = result
            self.open_trajectory_viewer(
                trajectory,
                time_from_start_s,
                filename="full_execution_plan_timed_preview.npz",
                title=f"Full execution plan · {step_count} steps",
            )

        self.run_background(
            "Preparing full validated execution-plan preview",
            self.controller.preview_full_execution_plan,
            success,
        )

    def preview_next(self) -> None:
        self.preview_ready = False

        def success(result):
            step, trajectory, time_from_start_s = result
            self.preview_ready = True
            self.next_button.setEnabled(True)
            self.next_pose_card.set_value(step.target_label)
            self.next_button.setText(f"🤖  EXECUTE · {step.target_label}")
            self.open_trajectory_viewer(
                trajectory,
                time_from_start_s,
                filename=(
                    f"execution_step_{int(step.step_id):03d}_timed_preview.npz"
                ),
                title=f"Step {step.step_id}: {step.route_label}",
            )
            self._refresh_execution_route()

        self.run_background(
            "Preparing next validated execution step",
            self.controller.preview_next_execution_step,
            success,
        )

    # ------------------------------------------------------------------
    # Stage 3 / 4 actions
    # ------------------------------------------------------------------

    def _set_motion_checked(self, checked: bool) -> None:
        blocked = self.motion_enabled.blockSignals(True)
        self.motion_enabled.setChecked(checked)
        self.motion_enabled.blockSignals(blocked)

    def _motion_toggled(self, checked: bool) -> None:
        if self.mode != "real":
            return
        if not checked:
            self.controller.motion_verified = False
            return
        if self.controller.robot is None:
            self._set_motion_checked(False)
            _message(
                self,
                "Connect first",
                "Connect the robot before enabling real motion.",
                error=True,
            )
            return

        detail = (
            "UR5e: release the E-stop, power/brake-release the robot, and start the "
            "External Control program on the teach pendant.\n\n"
            if self.robot_name == "ur5e"
            else "RB5: release the E-stop, initialize the arm, and keep the work area clear.\n\n"
        )
        answer = QtWidgets.QMessageBox.question(
            self,
            "Verify real motion path",
            detail
            + "The app will send the CURRENT joint pose as a low-speed trajectory. "
            "No displacement is intended, but the robot will enter commanded motion mode. "
            "Keep a hand near the physical E-stop. Continue?",
            _qt_enum(QtWidgets.QMessageBox, "StandardButton", "Yes")
            | _qt_enum(QtWidgets.QMessageBox, "StandardButton", "No"),
            _qt_enum(QtWidgets.QMessageBox, "StandardButton", "No"),
        )
        if answer != _qt_enum(QtWidgets.QMessageBox, "StandardButton", "Yes"):
            self._set_motion_checked(False)
            return

        self.run_background(
            "Verifying real trajectory command path",
            self.controller.verify_motion_ready,
            lambda report: (
                self._set_motion_checked(True),
                _message(self, "Real motion enabled", report),
            ),
            lambda _error: self._set_motion_checked(False),
        )

    def require_motion(self) -> bool:
        if self.motion_enabled.isChecked() and (
            self.mode != "real" or self.controller.motion_verified
        ):
            return True
        QtWidgets.QMessageBox.warning(
            self,
            "Motion disabled",
            "Review the path first, then enable motion.",
        )
        return False

    def go_to_start(self) -> None:
        if not self.require_motion():
            return

        plan = self.controller.plan
        if plan is None:
            _message(
                self,
                "No plan",
                "Generate a validated plan first.",
                error=True,
            )
            return
        if int(getattr(plan, "execution_step_index", 0)) != 0:
            _message(
                self,
                "Sequence already started",
                "GO TO START is only available before the first execution step.",
            )
            return

        # A preview prepared from the old state is invalid after moving to START.
        self.preview_ready = False
        self.next_button.setEnabled(False)

        def success(report):
            self._motion_complete("START / SAFE reached")
            self.motion_detail_label.setText(str(report))
            self.set_status(str(report), "ok")
            self._refresh_execution_route()

        self.motion_state_label.setText("🤖 COMMAND SENT · current → START")
        self.motion_progress.setValue(0)
        self.motion_progress.setFormat("0%")
        self.motion_time_label.setText(
            "Elapsed 0.0 s  ·  Remaining calculating…"
        )
        self.motion_detail_label.setText(
            "Moving to the recorded START/safe state used by the plan."
        )
        self.run_background(
            "Go to plan START / safe",
            self.controller.go_to_plan_start,
            success,
            self._motion_failed,
        )

    def next_scan(self) -> None:
        if not self.require_motion():
            return
        if not self.preview_ready:
            _message(self, "Preview required", "Press NEXT VIEW before EXECUTE.")
            return

        plan = self.controller.plan
        if plan is None or not getattr(plan, "execution_steps", None):
            _message(self, "No route", "Generate a validated execution route first.", error=True)
            return
        index = int(plan.execution_step_index)
        if index >= len(plan.execution_steps):
            _message(self, "Complete", "The execution sequence is already complete.")
            return
        step = plan.execution_steps[index]
        self.preview_ready = False

        def success(_):
            self._motion_complete(f"{step.target_label} complete")
            self.show_plan(self.controller.plan)
            self.refresh_scan_world()
            self._refresh_counts()
            self._refresh_execution_route()

        self.motion_state_label.setText(
            f"🤖 COMMAND SENT · {step.route_label}"
        )
        self.motion_progress.setValue(0)
        self.motion_progress.setFormat("0%")
        self.motion_time_label.setText("Elapsed 0.0 s  ·  Remaining calculating…")
        self.motion_detail_label.setText(
            "Executing exactly the trajectory shown by NEXT VIEW."
        )
        self.run_background(
            f"Execute {step.target_label}",
            self.controller.execute_next_execution_step,
            success,
            self._motion_failed,
        )

    def capture_additional(self) -> None:
        self.run_background(
            "Capturing additional manually taught scan",
            self.controller.capture_additional_scan,
            lambda _: (
                self.refresh_scan_world(),
                self.refresh_stage4_results(load_existing=False),
                self._refresh_counts(),
            ),
        )

    def choose_calibration_dataset(self) -> None:
        current = self.controller.calibration_dataset_path()
        manual_default = Path(__file__).resolve().parents[1] / "runs" / "real" / "dataset"
        start = current if current.exists() else manual_default
        selected = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select manual capture dataset (capture_*.npz)",
            str(start),
        )
        if not selected:
            return
        try:
            path = self.controller.set_calibration_dataset_path(selected)
            self.refresh_stage4_results(load_existing=False)
            self._refresh_counts()
            self.set_status(f"Stage 4 dataset selected · {path}", "ok")
        except BaseException as error:
            _message(self, "Invalid calibration dataset", str(error), error=True)

    def calibrate(self) -> None:
        self.run_background(
            "Running hand-eye calibration",
            self.controller.calibrate,
            self.show_calibration_result,
            lambda _error: self.refresh_stage4_results(load_existing=True),
        )

    def show_calibration_result(self, transform) -> None:
        matrix = np.asarray(transform, dtype=float)
        self.refresh_stage4_results(final_transform=matrix, load_existing=True)
        self.stage4_message.appendPlainText(
            "\nCalibration completed successfully.\n"
            f"Accumulated scans: {self.controller.accumulated_scan_path()}\n"
            f"Final matrix: {self.controller.config.path('calibrated_transform')}"
        )

    def stop(self) -> None:
        self.preview_ready = False
        self.next_button.setEnabled(False)
        if self.mode == "real":
            self.controller.motion_verified = False
            self._set_motion_checked(False)
        robot = self.controller.robot
        if robot is None:
            self.set_status("STOP ignored · robot is not connected", "warn")
            return
        try:
            robot.stop()
            self.set_status("STOP command sent", "warn")
        except BaseException as error:
            _message(self, "STOP failed", str(error), error=True)

    # ------------------------------------------------------------------
    # Dataset 3D refresh
    # ------------------------------------------------------------------

    def _dataset_world_scans(
        self,
        dataset: Path,
        *,
        handeye: np.ndarray | None = None,
    ) -> list[tuple[str, np.ndarray]]:
        scans: list[tuple[str, np.ndarray]] = []
        if handeye is None:
            handeye = _load_transform(self.controller.config.path("handeye"))
        handeye = np.asarray(handeye, dtype=float).reshape(4, 4)
        for path in sorted(dataset.glob("capture_*.npz")):
            try:
                with np.load(path, allow_pickle=False) as data:
                    if "points_w" in data:
                        points = np.asarray(data["points_w"], dtype=float)
                    elif "points_s" in data and ("T_world_tcp" in data or "T_base_tcp" in data):
                        points_s = np.asarray(data["points_s"], dtype=float)
                        T_tcp = np.asarray(
                            data["T_world_tcp"] if "T_world_tcp" in data else data["T_base_tcp"],
                            dtype=float,
                        )
                        T_world_sensor = T_tcp @ handeye
                        points = points_s @ T_world_sensor[:3, :3].T + T_world_sensor[:3, 3]
                    else:
                        continue
                if points.ndim == 2 and points.shape[1] == 3:
                    points = points[np.all(np.isfinite(points), axis=1)]
                    if len(points):
                        scans.append((path.name, points))
            except (OSError, ValueError, KeyError):
                pass
        return scans

    def refresh_initial_world(self) -> None:
        scans = self._dataset_world_scans(
            self.controller.config.path("initial_dataset")
        )
        transform = self.controller.stable_sensor_pose_mm()
        forward = str(
            self.controller.config.values["planning"].get(
                "sensor_forward_axis", "+z"
            )
        ).strip().lower()
        self.stage1_world.set_scans(
            scans,
            stable_transform=transform,
            forward_axis=forward,
            frame_length=float(
                self.controller.config.values.get("visualization", {}).get(
                    "sensor_frame_axis_length_mm", 70.0
                )
            ),
        )

    def refresh_scan_world(self) -> None:
        scans = self._dataset_world_scans(
            self.controller.config.path("scan_dataset")
        )
        self.stage3_world.set_scans(scans)

    @staticmethod
    def _matrix_text(matrix: np.ndarray) -> str:
        return np.array2string(
            np.asarray(matrix, dtype=float).reshape(4, 4),
            precision=8,
            suppress_small=True,
        )

    def refresh_stage4_results(
        self,
        *,
        final_transform: np.ndarray | None = None,
        load_existing: bool = False,
    ) -> None:
        dataset = self.controller.calibration_dataset_path()
        initial = _load_transform(self.controller.config.path("handeye"))
        output = self.controller.config.path("calibrated_transform")
        diagnostics = self.controller.calibration_diagnostics_path()

        diagnostic_payload = None
        if load_existing and diagnostics.is_file():
            try:
                diagnostic_payload = json.loads(diagnostics.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                diagnostic_payload = None
        accepted_existing = bool(
            isinstance(diagnostic_payload, dict)
            and diagnostic_payload.get("accepted", False)
        )
        if (
            final_transform is None
            and load_existing
            and accepted_existing
            and output.is_file()
        ):
            try:
                final_transform = _load_transform(output)
            except (OSError, ValueError, KeyError):
                final_transform = None

        display_transform = initial if final_transform is None else final_transform
        scans = self._dataset_world_scans(dataset, handeye=display_transform)
        self.stage4_world.set_scans(scans)
        self.stage4_dataset_path.setText(str(dataset))
        self.stage4_initial_matrix.setPlainText(self._matrix_text(initial))
        self.stage4_final_matrix.setPlainText(
            "Not calculated" if final_transform is None else self._matrix_text(final_transform)
        )
        self.stage4_accumulated_path.setText(str(self.controller.accumulated_scan_path()))
        self.stage4_output_path.setText(str(output))

        history: list[float] = []
        if isinstance(diagnostic_payload, dict):
            try:
                history = [
                    float(value)
                    for value in diagnostic_payload.get("plane_rms_history_mm", [])
                    if value is not None and np.isfinite(float(value))
                ]
            except (ValueError, TypeError):
                history = []
        self.stage4_rms_curve.setData(
            np.arange(len(history), dtype=float),
            np.asarray(history, dtype=float),
        )
        self.stage4_rms_plot.setTitle(
            "Plane RMS by calibration iteration"
            if not history
            else f"Plane RMS convergence · final {history[-1]:.6f} mm"
        )

    def _refresh_counts(self) -> None:
        initial = len(
            list(self.controller.config.path("initial_dataset").glob("capture_*.npz"))
        )
        scans = len(list(self.controller.calibration_dataset_path().glob("capture_*.npz")))
        self.initial_count_card.set_value(str(initial))
        self.scan_count_card.set_value(str(scans))

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def closeEvent(self, event) -> None:
        self.alive = False
        try:
            self.mount_verification.shutdown()
            self.controller.close()
        finally:
            event.accept()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collision-aware laser hand-eye workflow (Qt/PyQtGraph)"
    )
    parser.add_argument("--mode", choices=("real", "sim"))
    parser.add_argument("--robot", choices=("ur5e", "rb5"))
    parser.add_argument("--robot-ip", "--robot_ip", dest="robot_ip")
    parser.add_argument("--laser-ip", "--laser_ip", dest="laser_ip")
    parser.add_argument("--config")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--safe-joints-deg", type=float, nargs="+")
    parser.add_argument("--radius-mm", type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.plan_only:
        mode = args.mode or "sim"
        robot = args.robot or "rb5"
        config = WorkflowConfig.load(_default_config_path(mode, robot, args.config))
        if args.safe_joints_deg is None or args.radius_mm is None:
            raise SystemExit(
                "--plan-only requires --safe-joints-deg and --radius-mm"
            )
        plan = ScanPlanner(config).plan(
            np.asarray(args.safe_joints_deg),
            radius_mm=args.radius_mm,
        )
        print(
            f"radius={plan.radius_mm:.3f}; "
            f"safe={sum(s.executable for s in plan.scans)}/{len(plan.scans)}"
        )
        return

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    app.setApplicationName("Laser Hand-eye Workflow")

    complete_cli = args.mode is not None and args.robot is not None
    if args.mode == "real":
        complete_cli = complete_cli and bool(args.robot_ip and args.laser_ip)
    if complete_cli:
        mode, robot = args.mode, args.robot
        robot_ip, laser_ip = args.robot_ip, args.laser_ip
        if mode == "real":
            for label, value in (("--robot-ip", robot_ip), ("--laser-ip", laser_ip)):
                try:
                    address = ipaddress.ip_address(str(value))
                    if address.version != 4:
                        raise ValueError("IPv4 required")
                except ValueError as error:
                    raise SystemExit(f"{label} must be a valid IPv4 address") from error
    else:
        dialog = StartupDialog(args)
        if _qt_exec(dialog) == 0:
            return
        mode, robot, robot_ip, laser_ip = dialog.selections()

    config = WorkflowConfig.load(_default_config_path(mode, robot, args.config))
    if mode == "real":
        values = deepcopy(config.values)
        values["equipment"]["robot_host"] = str(robot_ip)
        values["equipment"]["laser_ip"] = str(laser_ip)
        config = WorkflowConfig(config.source_path, values)

    if mode == "sim":
        from .sim_workflow import SimWorkflowController
        controller = SimWorkflowController(config)
    else:
        controller = WorkflowController(config)

    window = WorkflowWindow(controller, mode=mode, robot=robot)
    window.show()
    raise SystemExit(_qt_exec(app))


def _default_config_path(mode: str, robot: str, custom: str | None) -> Path:
    if custom:
        return Path(custom)
    names = {
        ("sim", "rb5"): "sim_workflow.yaml",
        ("sim", "ur5e"): "ur5e_sim_workflow.yaml",
        ("real", "rb5"): "rb5_ljv7080_workflow.yaml",
        ("real", "ur5e"): "ur5e_ljv7080_workflow.yaml",
    }
    return Path(__file__).resolve().parent / "configs" / names[(mode, robot)]


if __name__ == "__main__":
    main()
