from __future__ import annotations

import threading

import numpy as np
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

    def __init__(self, host: str = "", port: int | None = None) -> None:
        super().__init__(host, port)
        self._rb_data = None
        self._rb_data_lock = threading.Lock()

    def connect(self, **kwargs) -> None:
        super().connect(**kwargs)
        try:
            import rbpodo as rb

            self._rb_data = rb.CobotData(self.host)
            self._read_rb_state()
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

    def close(self) -> None:
        self._rb_data = None
        super().close()
