from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

import real_laser_handeye.ros2_driver_launcher as driver_launcher_module
import real_laser_handeye.workflow as workflow_module
from real_laser_handeye.ros2_driver_launcher import ROS2DriverLauncher
from real_laser_handeye.workflow import WorkflowController


def test_driver_command_quotes_ip_and_static_arguments(tmp_path: Path):
    setup = tmp_path / "setup.bash"
    setup.write_text("# test\n", encoding="utf-8")
    config = SimpleNamespace(
        source_path=tmp_path / "workflow.yaml",
        values={
            "equipment": {
                "driver": {
                    "auto_launch": True,
                    "ros_setup": str(setup),
                    "launch_package": "ur_robot_driver",
                    "launch_file": "ur_control.launch.py",
                }
            }
        },
    )
    launcher = ROS2DriverLauncher(config)
    command = launcher._command(
        "ur_robot_driver",
        "ur_control.launch.py",
        {"ur_type": "ur5e", "robot_ip": "192.168.0.2", "launch_rviz": False},
    )
    assert command[:2] == ["bash", "-lc"]
    assert "source" in command[2]
    assert "ur_type:=ur5e" in command[2]
    assert "robot_ip:=192.168.0.2" in command[2]
    assert "launch_rviz:=false" in command[2]


class _HoldRobot:
    trajectory_action = "/test/follow_joint_trajectory"

    def __init__(self):
        self.q = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        self.T = np.eye(4)
        self.T[:3, 3] = [100.0, 200.0, 300.0]
        self.command = None

    def read_joint_positions_deg(self):
        return self.q.copy()

    def read_T_base_tcp(self):
        return self.T.copy()

    def execute_joint_trajectory(
        self,
        positions_rad,
        time_from_start_s,
        **kwargs,
    ):
        self.command = (
            np.asarray(positions_rad, dtype=float),
            np.asarray(time_from_start_s, dtype=float),
            kwargs,
        )
        return self.q.copy()


def test_real_motion_is_locked_until_hold_trajectory_verifies():
    config = SimpleNamespace(
        values={
            "motion": {
                "joint_speed_deg_s": 5.0,
                "joint_accel_deg_s2": 10.0,
                "joint_tolerance_deg": 0.5,
                "timeout_s": 60.0,
            }
        }
    )
    controller = WorkflowController(config)
    controller.robot = _HoldRobot()
    controller.laser = object()

    try:
        controller._require_motion_verified()
    except RuntimeError as error:
        assert "interlock" in str(error)
    else:
        raise AssertionError("real motion must start locked")

    report = controller.verify_motion_ready()
    assert controller.motion_verified is True
    positions, times, kwargs = controller.robot.command
    np.testing.assert_allclose(positions[0], np.radians(controller.robot.q))
    np.testing.assert_allclose(positions[1], np.radians(controller.robot.q))
    np.testing.assert_allclose(times, [0.0, 1.0])
    assert kwargs["velocities_rad_s"].shape == (2, 6)
    assert kwargs["accelerations_rad_s2"].shape == (2, 6)
    assert "trajectory verified" in report


class _IndependentRobot:
    JOINT_NAMES = ("j1", "j2", "j3", "j4", "j5", "j6")
    BASE_FRAME = "base"

    def __init__(self, host, port):
        self.connected = False
        self.closed = False

    def connect(self, **kwargs):
        self.connected = True

    def read_state_snapshot(self):
        return np.zeros(6), np.eye(4)

    def stop(self):
        pass

    def close(self):
        self.closed = True


class _IndependentLaser:
    def __init__(self, **kwargs):
        self.connected = False
        self.closed = False

    def connect(self):
        self.connected = True

    def read_profile(self, **kwargs):
        return np.zeros((8, 3))

    def close(self):
        self.closed = True


class _IndependentLauncher:
    calibration_path = None
    log_path = None

    def __init__(self, config):
        self.closed = False

    def start(self, host):
        pass

    def raise_if_exited(self):
        pass

    def close(self):
        self.closed = True


def _independent_config():
    return SimpleNamespace(
        joint_names=_IndependentRobot.JOINT_NAMES,
        values={
            "equipment": {
                "robot_host": "192.0.2.10",
                "robot_port": 5000,
                "robot_adapter": "test:Robot",
                "base_frame": "base",
                "laser_ip": "192.0.2.20",
                "laser_adapter": "test:Laser",
                "driver": {},
            },
            "planning": {},
        },
    )


def test_robot_and_laser_can_connect_and_disconnect_independently(monkeypatch):
    monkeypatch.setattr(
        workflow_module,
        "_load_object",
        lambda path: _IndependentRobot if path.endswith("Robot") else _IndependentLaser,
    )
    monkeypatch.setattr(
        driver_launcher_module, "ROS2DriverLauncher", _IndependentLauncher
    )
    controller = WorkflowController(_independent_config())

    controller.connect_laser()
    assert controller.laser is not None
    assert controller.robot is None
    assert "laser" in controller.connection_report

    controller.connect_robot()
    assert controller.robot is not None
    assert controller.laser is not None
    assert "robot" in controller.connection_report

    laser = controller.laser
    controller.disconnect_robot()
    assert controller.robot is None
    assert controller.laser is laser
    assert controller.laser.closed is False

    controller.disconnect_laser()
    assert controller.laser is None
    assert laser.closed is True
