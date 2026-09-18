from __future__ import annotations

import threading
from types import SimpleNamespace

import numpy as np

from real_laser_handeye.moveit_planner import MoveItPath
from real_laser_handeye.robot_adapter_rb5_ros import RobotAdapter as RB5Adapter
from real_laser_handeye.robot_adapter_ur5e_ros import RobotAdapter as UR5eAdapter


def _successful_path(duration_s: float = 0.25) -> MoveItPath:
    zeros = np.zeros((2, 6), dtype=float)
    return MoveItPath(
        success=True,
        reason="ok",
        joint_names=UR5eAdapter.JOINT_NAMES,
        positions_rad=zeros.copy(),
        time_from_start_s=np.asarray([0.0, duration_s]),
        velocities_rad_s=np.ones_like(zeros),
        accelerations_rad_s2=2.0 * np.ones_like(zeros),
    )


def test_ur_cartesian_retiming_never_speeds_up_moveit_path() -> None:
    path = _successful_path(duration_s=0.25)

    times, velocities, accelerations = UR5eAdapter._retime_for_linear_limits(
        path,
        distance_mm=1.0,
        speed_mm_s=5.0,
        accel_mm_s2=5.0,
    )

    # A 1 mm triangular move at 5 mm/s² needs 2*sqrt(1/5) seconds.
    expected_duration = 2.0 * np.sqrt(1.0 / 5.0)
    assert np.isclose(times[-1], expected_duration)
    stretch = expected_duration / 0.25
    np.testing.assert_allclose(velocities, path.velocities_rad_s / stretch)
    np.testing.assert_allclose(
        accelerations,
        path.accelerations_rad_s2 / (stretch * stretch),
    )

    slow_path = _successful_path(duration_s=2.0)
    slow_times, _, _ = UR5eAdapter._retime_for_linear_limits(
        slow_path,
        distance_mm=1.0,
        speed_mm_s=5.0,
        accel_mm_s2=5.0,
    )
    np.testing.assert_allclose(slow_times, slow_path.time_from_start_s)


def test_ur_move_l_plans_with_moveit_and_executes_trajectory_controller() -> None:
    current = np.eye(4, dtype=float)
    target = np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    path = _successful_path()
    calls: dict[str, object] = {}

    class FakeMoveIt:
        def plan_cartesian_to_pose(self, names, start, target_mm, **kwargs):
            calls["plan"] = (tuple(names), np.asarray(start), target_mm, kwargs)
            return path

    adapter = object.__new__(UR5eAdapter)
    adapter._connected = True
    adapter._node = object()
    adapter._moveit = FakeMoveIt()
    adapter._joint_tolerance_deg = 0.05
    adapter._path_tolerance_deg = 0.5
    adapter.read_T_base_tcp = lambda: current.copy()
    adapter._planning_from_base = lambda: np.eye(4, dtype=float)
    adapter.read_joint_positions_deg = lambda: np.zeros(6, dtype=float)
    adapter.execute_joint_trajectory = lambda *args, **kwargs: calls.update(
        execute=(args, kwargs)
    )
    adapter.wait_until_tcp_reached = lambda reached, **_kwargs: np.asarray(reached)

    actual = adapter.move_l(
        target,
        speed_mm_s=5.0,
        accel_mm_s2=5.0,
        timeout_s=20.0,
    )

    np.testing.assert_allclose(actual, target)
    planned_target = calls["plan"][2]
    np.testing.assert_allclose(planned_target[:3, 3], [1.0, 0.0, 0.0])
    execute_args, execute_kwargs = calls["execute"]
    np.testing.assert_allclose(execute_args[0], path.positions_rad)
    assert execute_args[1][-1] >= path.time_from_start_s[-1]
    assert execute_kwargs["path_tolerance_deg"] == 0.5


def _configure_fake_action_adapter(adapter, client, wrapped) -> None:
    goal_handle = SimpleNamespace(
        accepted=True,
        get_result_async=lambda: "result_future",
    )
    adapter._connected = True
    adapter._node = object()
    adapter._goal_lock = threading.Lock()
    adapter._active_goal = None
    adapter._wait_future = lambda future, _timeout, _description: (
        goal_handle if future == "send_future" else wrapped
    )
    adapter.wait_until_tcp_reached = lambda target, **_kwargs: np.asarray(target)


class _FakeClient:
    def __init__(self) -> None:
        self.goal = None

    def send_goal_async(self, goal):
        self.goal = goal
        return "send_future"


def test_rb_move_l_uses_existing_adapter_ros_action_and_si_units() -> None:
    client = _FakeClient()
    adapter = object.__new__(RB5Adapter)
    adapter._move_l_client = client
    _configure_fake_action_adapter(
        adapter,
        client,
        SimpleNamespace(status=4, result=SimpleNamespace(success=True)),
    )
    target = np.asarray([100.0, -200.0, 300.0, 10.0, -20.0, 30.0])

    actual = adapter.move_l(
        target,
        speed_mm_s=5.0,
        accel_mm_s2=10.0,
        timeout_s=20.0,
    )

    np.testing.assert_allclose(actual, target)
    np.testing.assert_allclose(client.goal.point[:3], [0.1, -0.2, 0.3])
    np.testing.assert_allclose(
        client.goal.point[3:], np.deg2rad([10.0, -20.0, 30.0]), rtol=1e-6
    )
    assert np.isclose(client.goal.speed, 0.005)
    assert np.isclose(client.goal.acceleration, 0.01)


def test_existing_adapter_action_endpoints() -> None:
    assert RB5Adapter.MOVE_L_ACTION == "/rbpodo_hardware/move_l"
    assert UR5eAdapter.TRAJECTORY_ACTIONS[0] == (
        "/scaled_joint_trajectory_controller/follow_joint_trajectory"
    )
    assert UR5eAdapter.BASE_FRAME == "base_link"
    assert UR5eAdapter.TCP_FRAME == "tool0"
