from __future__ import annotations

import threading

import numpy as np
from action_msgs.msg import GoalStatus
from rclpy.action import ActionClient
from rbpodo_msgs.action import MoveL
from scipy.spatial.transform import Rotation

from .ros2_robot_adapter_base import ROS2JointTrajectoryRobotAdapter


class RobotAdapter(ROS2JointTrajectoryRobotAdapter):
    """RB5 trajectory control via ROS, with measured state via rbpodo.

    rbpodo_ros2 currently exports ``jnt_ref`` as ros2_control joint state.
    The direct data channel is therefore deliberately used for encoder joints
    and TCP verification, while motion remains a FollowJointTrajectory action.
    """

    JOINT_NAMES = (
        "base",
        "shoulder",
        "elbow",
        "wrist1",
        "wrist2",
        "wrist3",
    )
    BASE_FRAME = "link0"
    TCP_FRAME = "tcp"
    JOINT_STATE_TOPIC = "/joint_states"
    TRAJECTORY_ACTION = "/joint_trajectory_controller/follow_joint_trajectory"
    MOVE_L_ACTION = "/rbpodo_hardware/move_l"

    def __init__(self, host: str = "", port: int | None = None) -> None:
        super().__init__(host, port)
        self._rb_data = None
        self._rb_data_lock = threading.Lock()
        self._move_l_client: ActionClient | None = None

    def connect(self, **kwargs) -> None:
        super().connect(**kwargs)
        try:
            import rbpodo as rb

            self._rb_data = rb.CobotData(self.host)
            self._read_rb_state()
            assert self._node is not None
            self._move_l_client = ActionClient(
                self._node,
                MoveL,
                self.MOVE_L_ACTION,
            )
            if not self._move_l_client.wait_for_server(
                timeout_sec=float(kwargs.get("startup_timeout_s", 15.0))
            ):
                raise ConnectionError(
                    f"RB MoveL action server is unavailable: {self.MOVE_L_ACTION}"
                )
        except BaseException:
            self.close()
            raise

    def _read_rb_state(self):
        if self._rb_data is None:
            raise RuntimeError(
                "RB5 measured-state channel is unavailable. Install the rbpodo "
                "Python package and reconnect."
            )
        with self._rb_data_lock:
            state = self._rb_data.request_data()
        if state is None or not hasattr(state, "sdata"):
            raise RuntimeError("RB5 returned no valid measured state")
        return state.sdata

    @staticmethod
    def _joint_positions_from_state(
        state,
        *,
        prefer_reference: bool = False,
    ) -> np.ndarray:
        field = "jnt_ref" if prefer_reference else "jnt_ang"
        if not hasattr(state, field):
            raise RuntimeError(f"RB5 state is missing {field}")
        value = np.asarray(getattr(state, field), dtype=float).reshape(-1)[:6]
        if value.shape != (6,) or not np.all(np.isfinite(value)):
            raise RuntimeError(f"invalid RB5 {field}: {value}")
        # Direct rbpodo SystemState fields use degrees.
        return value.copy()

    @staticmethod
    def _T_base_tcp_from_state(state) -> np.ndarray:
        field = next(
            (name for name in ("tcp_pos", "tcp", "cur_pos") if hasattr(state, name)),
            None,
        )
        if field is None:
            raise RuntimeError("RB5 state has no measured TCP pose field")
        pose = np.asarray(getattr(state, field), dtype=float).reshape(-1)[:6]
        if pose.shape != (6,) or not np.all(np.isfinite(pose)):
            raise RuntimeError(f"invalid RB5 measured TCP pose: {pose}")
        transform = np.eye(4)
        # RB controller [rx, ry, rz] is Euler ZYX expressed as extrinsic xyz.
        transform[:3, :3] = Rotation.from_euler(
            "xyz", pose[3:], degrees=True
        ).as_matrix()
        transform[:3, 3] = pose[:3]
        return transform

    def read_joint_positions_deg(self, *, prefer_reference: bool = False) -> np.ndarray:
        return self._joint_positions_from_state(
            self._read_rb_state(),
            prefer_reference=prefer_reference,
        )

    def read_T_base_tcp(self) -> np.ndarray:
        return self._T_base_tcp_from_state(self._read_rb_state())

    def read_state_snapshot(self) -> tuple[np.ndarray, np.ndarray]:
        """Read RB5 encoder joints and TCP from the same rbpodo SystemState."""
        state = self._read_rb_state()
        joints = self._joint_positions_from_state(state, prefer_reference=False)
        transform = self._T_base_tcp_from_state(state)
        return joints, transform

    def verify_motion_ready(self) -> str:
        """Verify the RB ROS MoveL endpoint without commanding a tiny motion.

        rbpodo_ros2 rejects a zero-distance MoveL as "target too close", so the
        interlock checks action availability and fresh measured state instead.
        """
        self._require_connected()
        if self._move_l_client is None or not self._move_l_client.server_is_ready():
            raise RuntimeError(f"RB MoveL action is not ready: {self.MOVE_L_ACTION}")
        self.read_state_snapshot()
        return (
            f"RB ROS MoveL ready ({self.MOVE_L_ACTION}); measured state is live. "
            "Zero-distance command skipped because rbpodo_ros2 rejects targets "
            "that are too close."
        )

    def move_l(
        self,
        target_pose_vec_mm,
        *,
        speed_mm_s: float = 80.0,
        accel_mm_s2: float = 80.0,
        position_tolerance_mm: float = 1.0,
        rotation_tolerance_deg: float = 1.0,
        timeout_s: float = 30.0,
        stable_count: int = 5,
        poll_interval_s: float = 0.05,
    ) -> np.ndarray:
        self._require_connected()
        if self._move_l_client is None:
            raise RuntimeError("RB MoveL action client is unavailable")
        target = self._validate_tcp_pose_vec(target_pose_vec_mm)
        if speed_mm_s <= 0.0 or accel_mm_s2 <= 0.0 or timeout_s <= 0.0:
            raise ValueError("Cartesian speed, acceleration, and timeout must be positive")

        goal = MoveL.Goal()
        # rbpodo_ros2 MoveL uses SI units: xyz/speed/acceleration in metres,
        # orientation in radians. The controller converts them to rbpodo units.
        goal.point = np.concatenate(
            [target[:3] / 1000.0, np.deg2rad(target[3:])]
        ).astype(np.float32).tolist()
        goal.speed = float(speed_mm_s) / 1000.0
        goal.acceleration = float(accel_mm_s2) / 1000.0
        goal.time_for_waiting_start = min(10.0, max(1.0, float(timeout_s) / 4.0))

        send_future = self._move_l_client.send_goal_async(goal)
        goal_handle = self._wait_future(
            send_future,
            min(5.0, float(timeout_s)),
            "RB MoveL goal acceptance",
        )
        if goal_handle is None or not goal_handle.accepted:
            raise RuntimeError(f"RB MoveL goal rejected by {self.MOVE_L_ACTION}")

        with self._goal_lock:
            self._active_goal = goal_handle
        result_future = goal_handle.get_result_async()
        try:
            wrapped = self._wait_future(
                result_future,
                float(timeout_s),
                "RB MoveL execution",
            )
        except TimeoutError:
            try:
                cancel_future = goal_handle.cancel_goal_async()
                self._wait_future(
                    cancel_future, 2.0, "timed-out RB MoveL cancellation"
                )
            finally:
                raise
        finally:
            with self._goal_lock:
                self._active_goal = None

        if wrapped is None:
            raise RuntimeError("RB MoveL action returned no result")
        if int(wrapped.status) != int(GoalStatus.STATUS_SUCCEEDED):
            raise RuntimeError(
                f"RB MoveL action did not succeed: status={wrapped.status}"
            )
        if not bool(wrapped.result.success):
            raise RuntimeError("RB MoveL action reported failure")
        return self.wait_until_tcp_reached(
            target,
            position_tolerance_mm=float(position_tolerance_mm),
            rotation_tolerance_deg=float(rotation_tolerance_deg),
            timeout_s=max(2.0, min(15.0, float(timeout_s))),
            stable_count=int(stable_count),
            poll_interval_s=float(poll_interval_s),
        )

    def close(self) -> None:
        try:
            self.stop()
        except Exception:
            pass
        move_l_client = self._move_l_client
        self._move_l_client = None
        if move_l_client is not None:
            try:
                move_l_client.destroy()
            except Exception:
                pass
        self._rb_data = None
        super().close()
