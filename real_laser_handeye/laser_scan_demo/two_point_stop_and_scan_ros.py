from __future__ import annotations

"""ROS two-point stop-and-scan GUI for RB5 and UR5e.

ur
cd /home/choisuhyun/lvs_HandEyeCalibration

/home/choisuhyun/miniconda3/envs/laser_handeye/bin/python \
  -m real_laser_handeye.laser_scan_demo.two_point_stop_and_scan_ros \
  --robot ur5e \
  --robot-ip 169.254.186.10 \
  --laser-ip 169.254.186.182 \
  --handeye runs/real/ur5e/handeye_workflow/T_tcp_sensor_calibrated.csv \
  --auto-save \
  --save-on-exit

"""

import argparse
from dataclasses import dataclass
import importlib
import ipaddress
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
import yaml

from . import two_point_stop_and_scan as legacy
from ..ros2_driver_launcher import ROS2DriverLauncher


CONFIG_NAMES = {
    "rb5": "rb5_ljv7080_workflow.yaml",
    "ur5e": "ur5e_ljv7080_workflow.yaml",
}
DEFAULT_ROBOT_IP = "169.254.186.20"
DEFAULT_LASER_IP = "169.254.186.182"
ROS_SETUP_FILES = (
    Path("/opt/ros/humble/setup.bash"),
    Path("/home/choisuhyun/rbpodo_ros2_ws/install/setup.bash"),
)
ROS_BOOTSTRAP_MARKER = "REAL_LASER_HANDEYE_ROS_BOOTSTRAPPED"


def ensure_ros_environment() -> None:
    """Re-exec this module after sourcing the required ROS 2 environments.

    A child process cannot modify its parent shell. Re-executing the same
    absolute Python interpreter preserves the active conda environment while
    importing the ROS setup variables into this GUI process.
    """
    if os.environ.get(ROS_BOOTSTRAP_MARKER) == "1":
        return
    missing = [str(path) for path in ROS_SETUP_FILES if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Required ROS setup file is missing: " + ", ".join(missing)
        )

    project_root = Path(__file__).resolve().parents[2]
    mujoco_src = project_root / "mujoco_handeye_sim" / "src"
    script = """
set -e
source "$1"
source "$2"
export REAL_LASER_HANDEYE_ROS_BOOTSTRAPPED=1
export PYTHONPATH="$3:$4${PYTHONPATH:+:$PYTHONPATH}"
shift 4
exec "$@"
""".strip()
    argv = [
        "bash",
        "-lc",
        script,
        "laser_handeye_ros_bootstrap",
        *(str(path) for path in ROS_SETUP_FILES),
        str(project_root),
        str(mujoco_src),
        sys.executable,
        "-m",
        "real_laser_handeye.laser_scan_demo.two_point_stop_and_scan_ros",
        *sys.argv[1:],
    ]
    os.execv("/bin/bash", argv)


def _dialog_buttons(*names: str):
    scoped = getattr(QtWidgets.QDialogButtonBox, "StandardButton", None)
    owner = scoped if scoped is not None else QtWidgets.QDialogButtonBox
    result = None
    for name in names:
        value = getattr(owner, name)
        result = value if result is None else result | value
    return result


def _exec_dialog(dialog: QtWidgets.QDialog) -> int:
    execute = getattr(dialog, "exec", None)
    return int(execute() if callable(execute) else dialog.exec_())


@dataclass
class DemoConfig:
    source_path: Path
    values: dict[str, Any]

    @classmethod
    def load(cls, path: str | Path) -> "DemoConfig":
        source = Path(path).expanduser().resolve()
        values = cls._load_values(source, ())
        for section in ("paths", "equipment", "planning", "motion"):
            if not isinstance(values.get(section), dict):
                raise ValueError(f"workflow config requires a {section!r} mapping")
        return cls(source, values)

    @classmethod
    def _load_values(
        cls, source: Path, loading: tuple[Path, ...]
    ) -> dict[str, Any]:
        source = source.resolve()
        if source in loading:
            raise ValueError(f"cyclic workflow config inheritance at {source}")
        values = yaml.safe_load(source.read_text(encoding="utf-8"))
        if not isinstance(values, dict):
            raise ValueError(f"workflow config must contain a mapping: {source}")
        parent = values.pop("extends", None)
        if parent is None:
            return values
        parent_path = Path(str(parent))
        if not parent_path.is_absolute():
            parent_path = source.parent / parent_path
        base = cls._load_values(parent_path, (*loading, source))
        return cls._deep_merge(base, values)

    @classmethod
    def _deep_merge(cls, base: dict[str, Any], overlay: dict[str, Any]):
        result = dict(base)
        for key, value in overlay.items():
            if isinstance(result.get(key), dict) and isinstance(value, dict):
                result[key] = cls._deep_merge(result[key], value)
            else:
                result[key] = value
        return result

    def path(self, key: str) -> Path:
        value = Path(str(self.values["paths"][key])).expanduser()
        if not value.is_absolute():
            value = self.source_path.parent / value
        return value.resolve()

    def moveit_settings(self) -> dict[str, Any]:
        settings = dict(self.values["planning"].get("moveit", {}))
        if not settings:
            raise ValueError("planning.moveit configuration is required")
        for key in ("ros_setup", "workspace_setup"):
            raw = settings.get(key)
            if not raw:
                continue
            path = Path(str(raw)).expanduser()
            if not path.is_absolute():
                path = self.source_path.parent / path
            settings[key] = str(path.resolve())
        motion = self.values["motion"]
        settings["joint_tolerance_deg"] = float(
            motion.get("joint_tolerance_deg", 0.05)
        )
        settings["path_tolerance_deg"] = float(
            motion.get("path_tolerance_deg", 0.5)
        )
        return settings

    @property
    def joint_names(self) -> tuple[str, ...]:
        names = tuple(str(name) for name in self.values["equipment"]["joint_names"])
        if len(names) != 6 or len(set(names)) != 6:
            raise ValueError("equipment.joint_names must contain six unique names")
        return names


def default_config_path(robot: str) -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / CONFIG_NAMES[robot]


def load_object(specification: str):
    module_name, separator, name = str(specification).partition(":")
    if not separator:
        raise ValueError(f"expected module:object specification, got {specification!r}")
    return getattr(importlib.import_module(module_name), name)


class StartupDialog(QtWidgets.QDialog):
    def __init__(self, args: argparse.Namespace):
        super().__init__()
        self.args = args
        self.connected_robot = None
        self.connected_laser = None
        self.driver_launcher: ROS2DriverLauncher | None = None
        self._worker = None
        self._operation = ""
        self.embedded = False
        self.ready_requested = False
        self.setWindowTitle("ROS two-point stop-and-scan · Connections")
        self.setModal(True)
        self.setMinimumWidth(720)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(7)

        self.robot = QtWidgets.QComboBox()
        self.robot.addItems(["rb5", "ur5e"])
        self.robot.setCurrentText(args.robot or "rb5")
        self.robot_ip = QtWidgets.QLineEdit(args.robot_ip or DEFAULT_ROBOT_IP)
        self.laser_ip = QtWidgets.QLineEdit(args.laser_ip or DEFAULT_LASER_IP)
        self.robot_connect = QtWidgets.QPushButton("Connect robot")
        self.laser_connect = QtWidgets.QPushButton("Connect laser")
        self.robot_status = QtWidgets.QLabel("ROBOT OFFLINE")
        self.laser_status = QtWidgets.QLabel("LASER OFFLINE")
        self.robot_status.setStyleSheet("color: #b45309; font-weight: 700;")
        self.laser_status.setStyleSheet("color: #b45309; font-weight: 700;")
        self.handeye = QtWidgets.QLineEdit(
            "" if args.handeye is None else str(Path(args.handeye).expanduser())
        )
        self.save_path = QtWidgets.QLineEdit(
            "" if args.save_path is None else str(Path(args.save_path).expanduser())
        )
        self.scan_speed = QtWidgets.QDoubleSpinBox()
        self.scan_speed.setRange(0.1, 1000.0)
        self.scan_speed.setDecimals(2)
        self.scan_speed.setValue(float(args.scan_speed_mm_s))
        self.scan_accel = QtWidgets.QDoubleSpinBox()
        self.scan_accel.setRange(0.1, 5000.0)
        self.scan_accel.setDecimals(2)
        self.scan_accel.setValue(float(args.scan_accel_mm_s2))
        self.spacing = QtWidgets.QDoubleSpinBox()
        self.spacing.setRange(0.01, 500.0)
        self.spacing.setDecimals(3)
        self.spacing.setValue(float(args.waypoint_spacing_mm))
        self.profile_count = QtWidgets.QSpinBox()
        self.profile_count.setRange(1, 1000)
        self.profile_count.setValue(int(args.profiles_per_waypoint))
        self.auto_driver = QtWidgets.QCheckBox("Launch robot ROS 2 driver automatically")
        self.auto_driver.setChecked(not args.no_auto_launch_driver)

        device_row = QtWidgets.QGridLayout()
        device_row.setHorizontalSpacing(8)
        device_row.addWidget(QtWidgets.QLabel("ROBOT"), 0, 0)
        device_row.addWidget(self.robot, 0, 1)
        device_row.addWidget(self.robot_ip, 0, 2)
        device_row.addWidget(self.robot_connect, 0, 3)
        device_row.addWidget(self.robot_status, 0, 4)
        device_row.addWidget(QtWidgets.QLabel("LASER"), 1, 0)
        device_row.addWidget(self.laser_ip, 1, 2)
        device_row.addWidget(self.laser_connect, 1, 3)
        device_row.addWidget(self.laser_status, 1, 4)
        device_row.setColumnStretch(2, 1)
        layout.addLayout(device_row)

        self.settings_toggle = QtWidgets.QToolButton()
        self.settings_toggle.setText("▸ Scan / file settings")
        self.settings_toggle.setCheckable(True)
        self.settings_toggle.setToolButtonStyle(QtCore.Qt.ToolButtonStyle.ToolButtonTextOnly)
        layout.addWidget(self.settings_toggle, alignment=QtCore.Qt.AlignmentFlag.AlignLeft)

        self.settings_frame = QtWidgets.QFrame()
        settings = QtWidgets.QGridLayout(self.settings_frame)
        settings.setContentsMargins(0, 2, 0, 2)
        settings.addWidget(QtWidgets.QLabel("Hand-eye"), 0, 0)
        settings.addWidget(self.handeye, 0, 1, 1, 5)
        settings.addWidget(QtWidgets.QLabel("Save"), 1, 0)
        settings.addWidget(self.save_path, 1, 1, 1, 5)
        settings.addWidget(QtWidgets.QLabel("Speed"), 2, 0)
        settings.addWidget(self.scan_speed, 2, 1)
        settings.addWidget(QtWidgets.QLabel("Accel"), 2, 2)
        settings.addWidget(self.scan_accel, 2, 3)
        settings.addWidget(QtWidgets.QLabel("Spacing"), 2, 4)
        settings.addWidget(self.spacing, 2, 5)
        settings.addWidget(QtWidgets.QLabel("Profiles"), 2, 6)
        settings.addWidget(self.profile_count, 2, 7)
        settings.addWidget(self.auto_driver, 3, 0, 1, 8)
        self.settings_frame.setVisible(False)
        layout.addWidget(self.settings_frame)
        self.settings_toggle.toggled.connect(self._toggle_settings)

        note = QtWidgets.QLabel(
            "RB5 uses the rbpodo_ros2 MoveL action. UR5e uses MoveIt Cartesian "
            "planning and the scaled joint trajectory controller; its External "
            "Control program must be running before real motion is enabled. "
            "Real motion stays locked until the command path is verified in the "
            "main window."
        )
        note.setWordWrap(True)
        note.setVisible(False)

        self.operation_status = QtWidgets.QLabel(
            "Select a robot and connect the robot and laser independently."
        )
        self.operation_status.setWordWrap(True)
        layout.addWidget(self.operation_status)

        action_row = QtWidgets.QHBoxLayout()
        action_row.addStretch(1)
        self.cancel_button = QtWidgets.QPushButton("Cancel")
        self.continue_button = QtWidgets.QPushButton("Open scan GUI")
        self.continue_button.setEnabled(False)
        action_row.addWidget(self.cancel_button)
        action_row.addWidget(self.continue_button)
        layout.addLayout(action_row)

        self._custom_handeye = args.handeye is not None
        self._custom_save = args.save_path is not None
        self.robot.currentTextChanged.connect(self._update_robot_defaults)
        self.robot_connect.clicked.connect(self._toggle_robot)
        self.laser_connect.clicked.connect(self._toggle_laser)
        self.continue_button.clicked.connect(self._accept_checked)
        self.cancel_button.clicked.connect(self.reject)
        self._poll_timer = QtCore.QTimer(self)
        self._poll_timer.setInterval(50)
        self._poll_timer.timeout.connect(self._poll_worker)
        self._update_robot_defaults(self.robot.currentText())

    def _toggle_settings(self, visible: bool) -> None:
        self.settings_frame.setVisible(bool(visible))
        self.settings_toggle.setText(
            "▾ Scan / file settings" if visible else "▸ Scan / file settings"
        )

    def _config(self, robot: str) -> DemoConfig:
        path = self.args.config or default_config_path(robot)
        return DemoConfig.load(path)

    def _update_robot_defaults(self, robot: str) -> None:
        config = self._config(robot)
        requires_driver = bool(
            config.values["equipment"].get(
                "cartesian_requires_ros_driver", robot == "ur5e"
            )
        )
        self.auto_driver.setEnabled(requires_driver)
        self.auto_driver.setChecked(
            requires_driver and not self.args.no_auto_launch_driver
        )
        self.auto_driver.setToolTip("")
        if not self._custom_handeye:
            self.handeye.setText(str(config.path("handeye")))
        if not self._custom_save:
            self.save_path.setText(
                str(config.path("session_root") / "two_point_ros_stop_scan.npz")
            )

    @staticmethod
    def _validated_ipv4(text: str, label: str) -> str:
        value = text.strip()
        address = ipaddress.ip_address(value)
        if address.version != 4:
            raise ValueError(f"{label} must be IPv4")
        return value

    def _set_busy(self, busy: bool, message: str = "") -> None:
        self.robot_connect.setEnabled(not busy)
        self.laser_connect.setEnabled(not busy)
        self.continue_button.setEnabled(
            not busy
            and self.connected_robot is not None
            and self.connected_laser is not None
        )
        self.cancel_button.setEnabled(not busy)
        self.robot.setEnabled(not busy and self.connected_robot is None)
        self.robot_ip.setEnabled(not busy and self.connected_robot is None)
        self.laser_ip.setEnabled(not busy and self.connected_laser is None)
        self.auto_driver.setEnabled(not busy and self.connected_robot is None)
        if message:
            self.operation_status.setText(message)

    def _refresh_connection_state(self) -> None:
        robot_online = self.connected_robot is not None
        laser_online = self.connected_laser is not None
        self.robot_status.setText("ROBOT CONNECTED" if robot_online else "ROBOT OFFLINE")
        self.laser_status.setText("LASER CONNECTED" if laser_online else "LASER OFFLINE")
        self.robot_status.setStyleSheet(
            f"color: {'#15803d' if robot_online else '#b45309'}; font-weight: 700;"
        )
        self.laser_status.setStyleSheet(
            f"color: {'#15803d' if laser_online else '#b45309'}; font-weight: 700;"
        )
        self.robot_connect.setText("Disconnect robot" if robot_online else "Connect robot")
        self.laser_connect.setText("Disconnect laser" if laser_online else "Connect laser")
        self._set_busy(False)

    def _start_worker(self, operation: str, target, message: str) -> None:
        if self._worker is not None:
            return
        self._operation = operation
        self._worker = legacy.MoveWorker(target)
        self._set_busy(True, message)
        self._worker.start()
        self._poll_timer.start()

    def _poll_worker(self) -> None:
        worker = self._worker
        if worker is None or worker.is_alive():
            return
        self._poll_timer.stop()
        worker.join()
        operation = self._operation
        self._worker = None
        self._operation = ""
        if worker.error is not None:
            self._refresh_connection_state()
            message = f"{operation} failed: {type(worker.error).__name__}: {worker.error}"
            self.operation_status.setText(message)
            QtWidgets.QMessageBox.critical(self, "Connection failed", message)
            return

        if operation == "connect_robot":
            self.connected_robot, self.driver_launcher = worker.result
            self.operation_status.setText(
                f"{self.robot.currentText().upper()} robot connected. "
                + (
                    "Both devices are ready."
                    if self.connected_laser is not None
                    else "Connect the laser next."
                )
            )
        elif operation == "connect_laser":
            self.connected_laser, point_count = worker.result
            self.operation_status.setText(
                f"Laser connected; fresh profile received ({point_count} points). "
                + (
                    "Both devices are ready."
                    if self.connected_robot is not None
                    else "Connect the robot next."
                )
            )
        elif operation == "disconnect_robot":
            self.operation_status.setText("Robot disconnected; real motion is locked.")
        elif operation == "disconnect_laser":
            self.operation_status.setText("Laser disconnected.")
        self._refresh_connection_state()

    def _connect_robot_resources(self):
        robot_name = self.robot.currentText()
        robot_ip = self._validated_ipv4(self.robot_ip.text(), "Robot IP")
        config = self._config(robot_name)
        equipment = config.values["equipment"]
        launcher = ROS2DriverLauncher(config)
        if not self.auto_driver.isChecked():
            launcher.settings["auto_launch"] = False
        robot_instance = None
        try:
            launcher.start(robot_ip)
            adapter_spec = str(equipment["cartesian_robot_adapter"])
            robot_class = load_object(adapter_spec)
            robot_instance = robot_class(robot_ip, self.args.robot_port)
            connect_kwargs: dict[str, Any] = {}
            if bool(equipment.get("cartesian_uses_moveit", False)):
                moveit_settings = config.moveit_settings()
                if launcher.calibration_path is not None:
                    launch_arguments = dict(
                        moveit_settings.get("launch_arguments", {})
                    )
                    launch_arguments["kinematics_params_file"] = str(
                        launcher.calibration_path
                    )
                    moveit_settings["launch_arguments"] = launch_arguments
                connect_kwargs["moveit_settings"] = moveit_settings
            robot_instance.connect(
                startup_timeout_s=float(
                    equipment.get("driver", {}).get("startup_timeout_s", 20.0)
                ),
                **connect_kwargs,
            )
            launcher.raise_if_exited()
            robot_instance.read_state_snapshot()
            return robot_instance, launcher
        except BaseException as error:
            log_path = launcher.log_path
            if robot_instance is not None:
                robot_instance.close()
            launcher.close()
            if log_path is not None:
                raise RuntimeError(f"{error}\nROS driver log: {log_path}") from error
            raise

    def _connect_laser_resource(self):
        laser_ip = self._validated_ipv4(self.laser_ip.text(), "Laser IP")
        config = self._config(self.robot.currentText())
        equipment = config.values["equipment"]
        laser_class = load_object(str(equipment["laser_adapter"]))
        laser = laser_class(
            ip=laser_ip,
            control_port=int(self.args.laser_control_port),
            high_speed_port=int(self.args.laser_high_speed_port),
            device_id=int(self.args.device_id),
            batch_profiles=int(self.args.batch_profiles),
            aggregate="latest",
        )
        try:
            laser.connect()
            profile = np.asarray(
                laser.read_profile(timeout_s=float(self.args.capture_timeout_s)),
                dtype=float,
            )
            if profile.ndim != 2 or profile.shape[1] != 3 or len(profile) == 0:
                raise RuntimeError(f"invalid startup profile shape: {profile.shape}")
            return laser, len(profile)
        except BaseException:
            laser.close()
            raise

    def _toggle_robot(self) -> None:
        if self.connected_robot is None:
            self._start_worker(
                "connect_robot",
                self._connect_robot_resources,
                "Starting the selected ROS 2 driver and connecting the robot…",
            )
            return
        robot = self.connected_robot
        launcher = self.driver_launcher
        self.connected_robot = None
        self.driver_launcher = None

        def disconnect() -> None:
            try:
                robot.close()
            finally:
                if launcher is not None:
                    launcher.close()

        self._start_worker("disconnect_robot", disconnect, "Disconnecting robot…")

    def _toggle_laser(self) -> None:
        if self.connected_laser is None:
            self._start_worker(
                "connect_laser",
                self._connect_laser_resource,
                "Connecting the Keyence laser and waiting for a fresh profile…",
            )
            return
        laser = self.connected_laser
        self.connected_laser = None
        self._start_worker("disconnect_laser", laser.close, "Disconnecting laser…")

    def _accept_checked(self) -> None:
        if self.connected_robot is None or self.connected_laser is None:
            QtWidgets.QMessageBox.warning(
                self,
                "Connect both devices",
                "Connect the robot and laser before opening the scan GUI.",
            )
            return
        for label, edit in (("Robot IP", self.robot_ip), ("Laser IP", self.laser_ip)):
            try:
                address = ipaddress.ip_address(edit.text().strip())
                if address.version != 4:
                    raise ValueError("IPv4 required")
            except ValueError:
                QtWidgets.QMessageBox.critical(self, "Invalid address", f"{label} must be IPv4")
                edit.setFocus()
                return
        handeye = Path(self.handeye.text().strip()).expanduser()
        if not handeye.is_file():
            QtWidgets.QMessageBox.critical(
                self, "Missing hand-eye", f"Hand-eye file not found:\n{handeye}"
            )
            return
        if not self.save_path.text().strip():
            QtWidgets.QMessageBox.critical(self, "Missing save path", "Save path is required")
            return
        if self.embedded:
            self.ready_requested = True
            self.operation_status.setText("Preparing the connected scan workspace…")
        else:
            self.accept()

    def take_ready(self) -> bool:
        requested = self.ready_requested
        self.ready_requested = False
        return requested

    def take_connections(self):
        if self.connected_robot is None or self.connected_laser is None:
            raise RuntimeError("robot and laser must both be connected")
        resources = (
            self.connected_robot,
            self.connected_laser,
            self.driver_launcher,
        )
        self.connected_robot = None
        self.connected_laser = None
        self.driver_launcher = None
        return resources

    def reject(self) -> None:
        if self._worker is not None:
            return
        if self.connected_laser is not None:
            try:
                self.connected_laser.close()
            except Exception:
                pass
            self.connected_laser = None
        if self.connected_robot is not None:
            try:
                self.connected_robot.close()
            except Exception:
                pass
            self.connected_robot = None
        if self.driver_launcher is not None:
            self.driver_launcher.close()
            self.driver_launcher = None
        super().reject()

    def apply(self, args: argparse.Namespace) -> None:
        args.robot = self.robot.currentText()
        args.robot_ip = self.robot_ip.text().strip()
        args.laser_ip = self.laser_ip.text().strip()
        args.handeye = Path(self.handeye.text().strip()).expanduser().resolve()
        args.save_path = Path(self.save_path.text().strip()).expanduser().resolve()
        args.scan_speed_mm_s = float(self.scan_speed.value())
        args.scan_accel_mm_s2 = float(self.scan_accel.value())
        args.waypoint_spacing_mm = float(self.spacing.value())
        args.profiles_per_waypoint = int(self.profile_count.value())
        args.no_auto_launch_driver = not self.auto_driver.isChecked()


class ROS2TwoPointScanViewer(legacy.TwoPointScanViewer):
    def __init__(
        self,
        *,
        robot_name: str,
        point_size_px: float,
        display_profile_stride: int,
        connection_panel: StartupDialog | None = None,
    ):
        super().__init__(
            point_size_px=point_size_px,
            display_profile_stride=display_profile_stride,
        )
        self._workspace_ready = connection_panel is None
        self.connection_panel = connection_panel
        self.window.setWindowTitle(
            f"ROS two-point stop-and-scan — {robot_name.upper()}"
        )
        self.enable_motion_requested = False
        self.motion_verified = False
        self.enable_motion = QtWidgets.QPushButton("Enable real motion")
        self.enable_motion.setCheckable(True)
        self.enable_motion.setStyleSheet(
            "QPushButton:checked { background: #9a3412; color: white; font-weight: bold; }"
        )
        self.enable_motion.clicked.connect(self._motion_clicked)
        banner = QtWidgets.QHBoxLayout()
        self.robot_mode_label = QtWidgets.QLabel(
            f"MOVEIT + EXTERNAL CONTROL · {robot_name.upper()}"
        )
        self.robot_mode_label.setObjectName("muted")
        banner.addWidget(self.robot_mode_label)
        banner.addStretch(1)
        banner.addWidget(self.enable_motion)
        insert_index = 1
        if connection_panel is not None:
            connection_panel.embedded = True
            connection_panel.setParent(self.window)
            connection_panel.setWindowFlags(QtCore.Qt.WindowType.Widget)
            connection_panel.setObjectName("card")
            connection_panel.cancel_button.hide()
            connection_panel.continue_button.setText("연결 완료 · 스캔 준비")
            self.window.layout().insertWidget(insert_index, connection_panel)
            connection_panel.show()
            insert_index += 1
        self.window.layout().insertLayout(insert_index, banner)
        self.set_workspace_connected(self._workspace_ready)

    def set_workspace_connected(self, connected: bool) -> None:
        self._workspace_ready = bool(connected)
        for button in (
            self.teach_first_button,
            self.teach_second_button,
            self.start_scan_button,
            self.new_step_button,
            self.undo_step_button,
            self.save_button,
            self.finish_button,
            self.quit_button,
            self.enable_motion,
        ):
            button.setEnabled(self._workspace_ready)
        if not connected:
            self.set_status("상단에서 ROBOT과 LASER를 각각 CONNECT 하세요.")

    def set_robot_name(self, robot_name: str) -> None:
        self.window.setWindowTitle(
            f"ROS two-point stop-and-scan — {robot_name.upper()}"
        )
        self.robot_mode_label.setText(
            f"MOVEIT + EXTERNAL CONTROL · {robot_name.upper()}"
        )

    def _motion_clicked(self, checked: bool) -> None:
        if checked:
            self.enable_motion_requested = True
        else:
            self.motion_verified = False
            self.set_status("Real motion disabled.")

    def take_enable_motion(self) -> bool:
        value = self.enable_motion_requested
        self.enable_motion_requested = False
        return value

    def set_motion_verified(self, verified: bool, message: str) -> None:
        self.motion_verified = bool(verified)
        self.enable_motion.blockSignals(True)
        self.enable_motion.setChecked(bool(verified))
        self.enable_motion.setText(
            "Real motion enabled" if verified else "Enable real motion"
        )
        self.enable_motion.blockSignals(False)
        self.set_status(message)

    def take_start_scan(self) -> bool:
        requested = super().take_start_scan()
        if requested and not self.motion_verified:
            self.set_status(
                "Motion is locked. Press Enable real motion and wait for verification."
            )
            return False
        return requested

    def take_abort(self) -> bool:
        requested = super().take_abort()
        if requested:
            self.set_motion_verified(False, "Abort requested; real motion locked again.")
        return requested

    def set_busy(self, busy: bool) -> None:
        super().set_busy(busy)
        self.enable_motion.setEnabled(not busy and self._workspace_ready)
        if not self._workspace_ready:
            self.set_workspace_connected(False)


def verify_motion_with_ui(
    robot: Any,
    laser: Any,
    args: argparse.Namespace,
    viewer: ROS2TwoPointScanViewer,
) -> str:
    def verify_robot_motion_interface() -> str:
        verifier = getattr(robot, "verify_motion_ready", None)
        if callable(verifier):
            return str(verifier())
        target = robot.read_tcp_pose_vec_mm()
        arrived = robot.move_l(
            target,
            speed_mm_s=float(args.alignment_speed_mm_s),
            accel_mm_s2=float(args.alignment_accel_mm_s2),
            position_tolerance_mm=float(args.position_tolerance_mm),
            rotation_tolerance_deg=float(args.rotation_tolerance_deg),
            timeout_s=float(args.move_timeout_s),
            stable_count=int(args.arrival_stable_count),
            poll_interval_s=float(args.robot_poll_interval_s),
        )
        error_mm = float(np.linalg.norm(np.asarray(arrived)[:3] - target[:3]))
        return (
            f"Direct Cartesian motion verified ({args.motion_backend}); "
            f"zero-displacement position error={error_mm:.3f} mm."
        )

    worker = legacy.MoveWorker(verify_robot_motion_interface)
    worker.start()
    last_callback = None
    while worker.is_alive() and viewer.is_open():
        if viewer.take_abort():
            robot.stop()
        sample = legacy.read_latest_live_profile(laser, args.profile_stale_after_s)
        if sample is not None and sample[0] != last_callback:
            last_callback = sample[0]
            viewer.update_live_profile(sample[2])
        viewer.set_status("Verifying ROS Cartesian action and measured state...")
        viewer.process_events()
        time.sleep(0.005)
    if worker.is_alive() and not viewer.is_open():
        robot.stop()
    worker.join()
    if worker.error is not None:
        raise RuntimeError(str(worker.error))
    return str(worker.result)


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_profiles <= 0:
        raise ValueError("--batch-profiles must be positive")
    for name in (
        "scan_speed_mm_s",
        "scan_accel_mm_s2",
        "alignment_speed_mm_s",
        "alignment_accel_mm_s2",
        "waypoint_spacing_mm",
        "capture_timeout_s",
        "move_timeout_s",
        "position_tolerance_mm",
        "rotation_tolerance_deg",
        "robot_poll_interval_s",
        "live_profile_rate_hz",
        "render_rate_hz",
        "point_size_px",
    ):
        if float(getattr(args, name)) <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.waypoint_count is not None and args.waypoint_count < 2:
        raise ValueError("--waypoint-count must be at least 2")
    if args.max_waypoints < 2 or args.profiles_per_waypoint < 1:
        raise ValueError("waypoint/profile counts are invalid")
    if args.arrival_stable_count < 1:
        raise ValueError("--arrival-stable-count must be at least 1")
    if args.display_profile_stride < 1 or args.display_max_points < 1:
        raise ValueError("display stride/point limits must be positive")
    if args.settle_at_waypoint_s < 0.0:
        raise ValueError("--settle-at-waypoint-s cannot be negative")
    if args.min_scan_distance_mm <= 0.0:
        raise ValueError("--min-scan-distance-mm must be positive")
    if args.max_scan_distance_mm <= args.min_scan_distance_mm:
        raise ValueError("--max-scan-distance-mm must exceed the minimum")
    if args.second_position_start_tolerance_mm <= 0.0:
        raise ValueError("--second-position-start-tolerance-mm must be positive")
def run_session(
    *,
    robot: Any,
    laser: Any,
    T_tcp_sensor: np.ndarray,
    args: argparse.Namespace,
    viewer: ROS2TwoPointScanViewer | None = None,
) -> None:
    if viewer is None:
        viewer = ROS2TwoPointScanViewer(
            robot_name=args.robot,
            point_size_px=args.point_size_px,
            display_profile_stride=args.display_profile_stride,
        )
    viewer.set_robot_name(args.robot)
    viewer.set_workspace_connected(True)
    session = legacy.ScanSession()
    first_pose: np.ndarray | None = None
    second_pose: np.ndarray | None = None
    latest_live_T: np.ndarray | None = None
    latest_callback_id: int | None = None
    latest_message = (
        "Connected read-only. Teach two poses, then explicitly enable real motion."
    )
    next_live_profile = 0.0
    next_cloud_render = 0.0
    running = True

    try:
        while running and viewer.is_open():
            viewer.process_events()

            if viewer.take_quit():
                if session.steps:
                    legacy.save_session(
                        args.save_path,
                        session=session,
                        T_tcp_sensor=T_tcp_sensor,
                        handeye_path=args.handeye,
                        args=args,
                    )
                break

            if viewer.take_enable_motion():
                viewer.set_busy(True)
                viewer.set_motion_progress(None, "VERIFYING MOTION PATH")
                try:
                    report = verify_motion_with_ui(robot, laser, args, viewer)
                    viewer.set_motion_verified(True, report)
                    latest_message = report
                except Exception as error:
                    latest_message = f"Motion verification failed: {type(error).__name__}: {error}"
                    viewer.set_motion_verified(False, latest_message)
                finally:
                    viewer.set_busy(False)
                    viewer.reset_motion_progress("READY" if viewer.motion_verified else "LOCKED")

            if viewer.take_teach_first():
                try:
                    first_pose = robot.read_tcp_pose_vec_mm()
                    viewer.set_teaching(first_pose, second_pose)
                    latest_message = "First TCP pose taught; its orientation will be fixed."
                except Exception as error:
                    latest_message = f"Teach first failed: {type(error).__name__}: {error}"

            if viewer.take_teach_second():
                try:
                    second_pose = robot.read_tcp_pose_vec_mm()
                    if first_pose is not None:
                        distance = legacy.validate_step_geometry(first_pose, second_pose, args)
                        latest_message = (
                            "Second TCP pose taught; scan direction is second -> first, "
                            f"distance={distance:.2f} mm."
                        )
                    else:
                        latest_message = "Second pose taught; first pose is still missing."
                    viewer.set_teaching(first_pose, second_pose)
                except Exception as error:
                    latest_message = f"Teach second failed: {type(error).__name__}: {error}"

            if viewer.take_new_step():
                first_pose = None
                second_pose = None
                viewer.set_teaching(None, None)
                latest_message = f"New step ready; completed steps={len(session.steps)}."

            if viewer.take_undo_step():
                removed = session.remove_last_step()
                latest_message = (
                    "No completed step to undo."
                    if removed is None
                    else f"Removed step {removed.step_index}; remaining={len(session.steps)}."
                )

            if viewer.take_save():
                try:
                    legacy.save_session(
                        args.save_path,
                        session=session,
                        T_tcp_sensor=T_tcp_sensor,
                        handeye_path=args.handeye,
                        args=args,
                    )
                    latest_message = f"Saved {session.profile_count} captures to {args.save_path}."
                except Exception as error:
                    latest_message = f"Save failed: {type(error).__name__}: {error}"

            if viewer.take_finish():
                legacy.save_session(
                    args.save_path,
                    session=session,
                    T_tcp_sensor=T_tcp_sensor,
                    handeye_path=args.handeye,
                    args=args,
                )
                print(
                    f"Saved and finished: {args.save_path} | steps={len(session.steps)}, "
                    f"profiles={session.profile_count}, points={session.point_count}"
                )
                running = False
                continue

            if viewer.take_start_scan():
                if first_pose is None or second_pose is None:
                    latest_message = "Teach both first and second positions before scanning."
                else:
                    viewer.set_busy(True)
                    try:
                        step = legacy.scan_second_to_first(
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
                            f"Step {step.step_index} complete: captures={len(step.profiles)}, "
                            f"points={step.point_count}."
                        )
                        if args.auto_save:
                            legacy.save_session(
                                args.save_path,
                                session=session,
                                T_tcp_sensor=T_tcp_sensor,
                                handeye_path=args.handeye,
                                args=args,
                            )
                            latest_message += " Auto-saved."
                    except Exception as error:
                        latest_message = f"Scan failed: {type(error).__name__}: {error}"
                        viewer.reset_motion_progress("ERROR")
                        print(latest_message)
                    finally:
                        viewer.set_busy(False)

            now = time.monotonic()
            if now >= next_live_profile:
                try:
                    sample = legacy.read_latest_live_profile(
                        laser, args.profile_stale_after_s
                    )
                    if sample is not None and sample[0] != latest_callback_id:
                        latest_callback_id = sample[0]
                        viewer.update_live_profile(sample[2])
                except Exception as error:
                    latest_message = f"Live profile error: {type(error).__name__}: {error}"
                next_live_profile = now + 1.0 / args.live_profile_rate_hz

            if now >= next_cloud_render:
                try:
                    latest_live_T = legacy.validate_transform(
                        robot.read_T_base_tcp(), "live T_base_tcp"
                    )
                    xyz = latest_live_T[:3, 3]
                    rpy = legacy.rotation_to_rpy_deg(latest_live_T[:3, :3])
                    tcp_text = (
                        f"TCP xyz=[{xyz[0]:.1f}, {xyz[1]:.1f}, {xyz[2]:.1f}] mm, "
                        f"rpy=[{rpy[0]:.1f}, {rpy[1]:.1f}, {rpy[2]:.1f}] deg"
                    )
                except Exception as error:
                    latest_live_T = None
                    tcp_text = f"TCP unavailable: {type(error).__name__}: {error}"
                viewer.update_cloud(
                    cloud=session.display_cloud(args.display_max_points),
                    latest_profile=session.latest_profile(),
                    tcp_path=session.tcp_path(),
                    current_T_base_tcp=latest_live_T,
                )
                viewer.set_status(
                    f"{latest_message} | {tcp_text} | steps={len(session.steps)}, "
                    f"profiles={session.profile_count}, points={session.point_count}"
                )
                next_cloud_render = now + 1.0 / args.render_rate_hz

            time.sleep(0.003)
    finally:
        if args.save_on_exit and session.steps:
            legacy.save_session(
                args.save_path,
                session=session,
                T_tcp_sensor=T_tcp_sensor,
                handeye_path=args.handeye,
                args=args,
            )
        viewer.close()


def resolve_args(args: argparse.Namespace) -> tuple[argparse.Namespace, DemoConfig]:
    if not args.robot or not args.robot_ip or not args.laser_ip:
        raise ValueError("robot selection and both device IPs are required")
    config = DemoConfig.load(args.config or default_config_path(args.robot))
    args.handeye = (
        config.path("handeye")
        if args.handeye is None
        else Path(args.handeye).expanduser().resolve()
    )
    args.save_path = (
        config.path("session_root") / "two_point_ros_stop_scan.npz"
        if args.save_path is None
        else Path(args.save_path).expanduser().resolve()
    )
    for label, value in (("robot IP", args.robot_ip), ("laser IP", args.laser_ip)):
        address = ipaddress.ip_address(str(value))
        if address.version != 4:
            raise ValueError(f"{label} must be IPv4")

    motion = config.values["motion"]
    capture = config.values.get("capture", {})
    if args.position_tolerance_mm is None:
        args.position_tolerance_mm = float(
            motion.get("linear_position_tolerance_mm", 0.2)
        )
    if args.rotation_tolerance_deg is None:
        args.rotation_tolerance_deg = float(
            motion.get("linear_rotation_tolerance_deg", 0.1)
        )
    if args.max_capture_translation_mm is None:
        args.max_capture_translation_mm = float(
            capture.get("max_stationarity_translation_mm", 0.1)
        )
    if args.max_capture_rotation_deg is None:
        args.max_capture_rotation_deg = float(
            capture.get("max_stationarity_rotation_deg", 0.1)
        )
    equipment = config.values["equipment"]
    adapter_spec = str(equipment.get("cartesian_robot_adapter", ""))
    if not adapter_spec:
        raise ValueError("equipment.cartesian_robot_adapter is required")
    args.robot_adapter_module = adapter_spec.partition(":")[0]
    args.motion_backend = str(
        equipment.get("cartesian_motion_backend", "direct_cartesian")
    )
    return args, config


def run(args: argparse.Namespace) -> None:
    pg.mkQApp("ROS two-point stop-and-scan")
    connection = StartupDialog(args)
    viewer = ROS2TwoPointScanViewer(
        robot_name=args.robot or "rb5",
        point_size_px=args.point_size_px,
        display_profile_stride=args.display_profile_stride,
        connection_panel=connection,
    )
    robot = None
    laser = None
    launcher = None
    try:
        while viewer.is_open() and not connection.take_ready():
            viewer.process_events()
            time.sleep(0.01)
        if not viewer.is_open():
            connection.reject()
            return

        connection.apply(args)
        args, config = resolve_args(args)
        validate_args(args)
        T_tcp_sensor = legacy.load_transform(args.handeye)
        robot, laser, launcher = connection.take_connections()
        for widget in (
            connection.robot,
            connection.robot_ip,
            connection.laser_ip,
            connection.robot_connect,
            connection.laser_connect,
            connection.continue_button,
            connection.settings_toggle,
        ):
            widget.setEnabled(False)
        connection.operation_status.setText(
            "ROBOT + LASER CONNECTED · scanning controls are ready."
        )
        equipment = config.values["equipment"]
        adapter_spec = str(equipment["cartesian_robot_adapter"])
        print(f"robot={args.robot}, adapter={adapter_spec}")
        print(f"motion_backend={args.motion_backend}")
        print(f"hand-eye={args.handeye}")
        print(f"save={args.save_path}")
        run_session(
            robot=robot,
            laser=laser,
            T_tcp_sensor=T_tcp_sensor,
            args=args,
            viewer=viewer,
        )
    finally:
        if robot is None and laser is None:
            connection.reject()
        if laser is not None:
            laser.close()
        if robot is not None:
            robot.close()
        if launcher is not None:
            launcher.close()
        if viewer.is_open():
            viewer.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "ROS two-point stop-and-scan GUI for RB5 or UR5e"
        )
    )
    parser.add_argument("--robot", choices=("rb5", "ur5e"))
    parser.add_argument("--config", type=Path)
    parser.add_argument("--robot-ip", "--robot-host", dest="robot_ip")
    parser.add_argument("--robot-port", type=int)
    parser.add_argument("--laser-ip")
    parser.add_argument("--handeye", type=Path)
    parser.add_argument("--save-path", type=Path)
    parser.add_argument("--no-auto-launch-driver", action="store_true")

    parser.add_argument("--laser-control-port", type=int, default=24691)
    parser.add_argument("--laser-high-speed-port", type=int, default=24692)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--batch-profiles", type=int, default=1)
    parser.add_argument("--profile-stale-after-s", type=float, default=0.5)

    parser.add_argument("--scan-speed-mm-s", type=float, default=5.0)
    parser.add_argument("--scan-accel-mm-s2", type=float, default=5.0)
    parser.add_argument("--alignment-speed-mm-s", type=float, default=10.0)
    parser.add_argument("--alignment-accel-mm-s2", type=float, default=10.0)

    parser.add_argument("--waypoint-spacing-mm", type=float, default=1.0)
    parser.add_argument("--waypoint-count", type=int)
    parser.add_argument("--max-waypoints", type=int, default=500)
    parser.add_argument("--settle-at-waypoint-s", type=float, default=0.5)
    parser.add_argument("--profiles-per-waypoint", type=int, default=10)
    parser.add_argument(
        "--capture-aggregate", choices=("mean", "median", "latest"), default="mean"
    )
    parser.add_argument("--capture-timeout-s", type=float, default=5.0)
    parser.add_argument("--max-capture-translation-mm", type=float)
    parser.add_argument("--max-capture-rotation-deg", type=float)

    parser.add_argument("--move-timeout-s", type=float, default=60.0)
    parser.add_argument("--position-tolerance-mm", type=float)
    parser.add_argument("--rotation-tolerance-deg", type=float)
    parser.add_argument("--arrival-stable-count", type=int, default=5)
    parser.add_argument("--robot-poll-interval-s", type=float, default=0.05)
    parser.add_argument("--min-scan-distance-mm", type=float, default=2.0)
    parser.add_argument("--max-scan-distance-mm", type=float, default=500.0)
    parser.add_argument("--second-position-start-tolerance-mm", type=float, default=5.0)

    parser.add_argument("--live-profile-rate-hz", type=float, default=20.0)
    parser.add_argument("--display-profile-stride", type=int, default=4)
    parser.add_argument("--render-rate-hz", type=float, default=8.0)
    parser.add_argument("--display-max-points", type=int, default=250_000)
    parser.add_argument("--point-size-px", type=float, default=2.0)
    parser.add_argument("--auto-save", action="store_true")
    parser.add_argument("--save-on-exit", action="store_true")
    return parser.parse_args()


def main() -> None:
    ensure_ros_environment()
    run(parse_args())


if __name__ == "__main__":
    main()
