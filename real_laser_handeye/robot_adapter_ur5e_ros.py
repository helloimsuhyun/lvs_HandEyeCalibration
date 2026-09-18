from __future__ import annotations

"""Universal Robots UR5e adapter for MoveIt + ros2_control execution."""

import math
from typing import Any

import numpy as np
from rclpy.duration import Duration
from rclpy.time import Time
from scipy.spatial.transform import Rotation

from .moveit_planner import MoveItPath, MoveItServicePlanner
from .ros2_robot_adapter_base import ROS2JointTrajectoryRobotAdapter


class RobotAdapter(ROS2JointTrajectoryRobotAdapter):
    """UR5e trajectory adapter with collision-checked Cartesian MoveIt planning.

    The regular hand-eye workflow supplies complete MoveIt trajectories directly
    to :meth:`execute_joint_trajectory`. Tools that use the legacy ``move_l``
    compatibility API, such as the two-point stop-and-scan GUI, are routed
    through MoveIt's Cartesian-path service and the same ros2_control trajectory
    controller. No URScript program is sent by this adapter.
    """

    JOINT_NAMES = (
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    )
    BASE_FRAME = "base_link"
    TCP_FRAME = "tool0"
    JOINT_STATE_TOPIC = "/joint_states"
    TRAJECTORY_ACTIONS = (
        "/scaled_joint_trajectory_controller/follow_joint_trajectory",
        "/joint_trajectory_controller/follow_joint_trajectory",
    )

    def __init__(self, host: str = "", port: int | None = None) -> None:
        super().__init__(host, port)
        self._moveit: MoveItServicePlanner | None = None
        self._joint_tolerance_deg = 0.05
        self._path_tolerance_deg = 0.5

    def connect(
        self,
        *,
        startup_timeout_s: float = 15.0,
        moveit_settings: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        """Connect the trajectory controller and optionally start MoveIt.

        ``workflow_gui.py`` plans before execution and therefore does not need
        an adapter-owned planner. Cartesian compatibility clients pass the
        workflow's MoveIt settings here so ``move_l`` can use the exact same
        model, collision scene, and FollowJointTrajectory execution path.
        """
        super().connect(startup_timeout_s=startup_timeout_s, **kwargs)
        if moveit_settings is None:
            return

        try:
            settings = dict(moveit_settings)
            # The compatibility API receives base_link -> tool0 targets. The
            # sensor remains in the model and is collision checked even though
            # tool0, an intermediate group link, is the requested link.
            settings["tip_link"] = self.TCP_FRAME
            self._joint_tolerance_deg = float(
                settings.pop("joint_tolerance_deg", self._joint_tolerance_deg)
            )
            self._path_tolerance_deg = float(
                settings.pop("path_tolerance_deg", self._path_tolerance_deg)
            )
            self._moveit = MoveItServicePlanner(settings)
            self._moveit.__enter__()
        except BaseException:
            self.close()
            raise

    def _planning_from_base(self) -> np.ndarray:
        """Return ``planning_frame -> base_link`` in millimetres."""
        if self._moveit is None:
            raise RuntimeError(
                "MoveIt is not configured for Cartesian motion; reconnect the "
                "robot through the workflow configuration"
            )
        planning_frame = self._moveit.planning_frame
        if planning_frame == self.BASE_FRAME:
            return np.eye(4, dtype=float)
        if self._tf_buffer is None:
            raise RuntimeError("TF buffer is not initialized")
        stamped = self._tf_buffer.lookup_transform(
            planning_frame,
            self.BASE_FRAME,
            Time(),
            timeout=Duration(seconds=0.5),
        )
        t = stamped.transform.translation
        q = stamped.transform.rotation
        transform = np.eye(4, dtype=float)
        transform[:3, :3] = Rotation.from_quat(
            [q.x, q.y, q.z, q.w]
        ).as_matrix()
        transform[:3, 3] = 1000.0 * np.asarray([t.x, t.y, t.z], dtype=float)
        return transform

    @staticmethod
    def _minimum_linear_duration_s(
        distance_mm: float,
        speed_mm_s: float,
        accel_mm_s2: float,
    ) -> float:
        """Conservative triangular/trapezoidal duration for a TCP translation."""
        distance = float(distance_mm)
        speed = float(speed_mm_s)
        accel = float(accel_mm_s2)
        if distance < 0.0 or not np.isfinite(distance):
            raise ValueError("Cartesian distance must be finite and non-negative")
        if speed <= 0.0 or accel <= 0.0:
            raise ValueError("Cartesian speed and acceleration must be positive")
        if distance == 0.0:
            return 0.0
        accel_distance = speed * speed / accel
        if distance <= accel_distance:
            return 2.0 * math.sqrt(distance / accel)
        return 2.0 * speed / accel + (distance - accel_distance) / speed

    @classmethod
    def _retime_for_linear_limits(
        cls,
        path: MoveItPath,
        *,
        distance_mm: float,
        speed_mm_s: float,
        accel_mm_s2: float,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        """Slow a MoveIt trajectory when needed to honor the GUI linear limits."""
        times = np.asarray(path.time_from_start_s, dtype=float).reshape(-1)
        if len(times) < 2 or not np.all(np.isfinite(times)) or times[-1] <= 0.0:
            raise RuntimeError("MoveIt returned invalid Cartesian trajectory timing")

        minimum = cls._minimum_linear_duration_s(
            distance_mm,
            speed_mm_s,
            accel_mm_s2,
        )
        stretch = max(1.0, minimum / float(times[-1]))
        times = times * stretch

        def optional(values, divisor: float) -> np.ndarray | None:
            array = np.asarray(values, dtype=float)
            if array.size == 0:
                return None
            return array / divisor

        velocities = optional(path.velocities_rad_s, stretch)
        accelerations = optional(path.accelerations_rad_s2, stretch * stretch)
        return times, velocities, accelerations

    def verify_motion_ready(self) -> str:
        """Exercise the External Control trajectory path without displacement."""
        self._require_connected()
        if self._moveit is None:
            raise RuntimeError("MoveIt Cartesian planner is not connected")
        if self._action_client is None or not self._action_client.server_is_ready():
            raise RuntimeError(
                f"trajectory action is not ready: {self.trajectory_action}"
            )

        joints = np.radians(self.read_joint_positions_deg())
        hold = np.vstack([joints, joints])
        zeros = np.zeros_like(hold)
        self.execute_joint_trajectory(
            hold,
            np.asarray([0.0, 1.0], dtype=float),
            velocities_rad_s=zeros,
            accelerations_rad_s2=zeros,
            tolerance_deg=min(0.5, self._joint_tolerance_deg),
            path_tolerance_deg=self._path_tolerance_deg,
            timeout_s=15.0,
        )
        self.read_state_snapshot()
        return (
            "MoveIt + External Control trajectory verified "
            f"({self.trajectory_action}); measured joint/TF state is live."
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
        """Plan a straight collision-checked tool0 path and execute it once."""
        self._require_connected()
        if self._moveit is None:
            raise RuntimeError("MoveIt Cartesian planner is unavailable")
        target = self._validate_tcp_pose_vec(target_pose_vec_mm)
        if speed_mm_s <= 0.0 or accel_mm_s2 <= 0.0 or timeout_s <= 0.0:
            raise ValueError("Cartesian speed, acceleration, and timeout must be positive")

        target_base = self._tcp_pose_vec_to_transform(target)
        current_base = self.read_T_base_tcp()
        planning_from_base = self._planning_from_base()
        target_planning = planning_from_base @ target_base
        current_planning = planning_from_base @ current_base
        current_rad = np.radians(self.read_joint_positions_deg())

        path = self._moveit.plan_cartesian_to_pose(
            self.JOINT_NAMES,
            current_rad,
            target_planning,
            start_pose_mm=current_planning,
            debug_label="UR5e two-point Cartesian move",
        )
        if not path.success:
            raise RuntimeError(f"MoveIt Cartesian planning failed: {path.reason}")

        distance_mm = float(
            np.linalg.norm(target_base[:3, 3] - current_base[:3, 3])
        )
        times, velocities, accelerations = self._retime_for_linear_limits(
            path,
            distance_mm=distance_mm,
            speed_mm_s=float(speed_mm_s),
            accel_mm_s2=float(accel_mm_s2),
        )
        execution_timeout = max(float(timeout_s), float(times[-1]) * 2.0 + 5.0)
        self.execute_joint_trajectory(
            path.positions_rad,
            times,
            velocities_rad_s=velocities,
            accelerations_rad_s2=accelerations,
            tolerance_deg=self._joint_tolerance_deg,
            path_tolerance_deg=self._path_tolerance_deg,
            timeout_s=execution_timeout,
            stable_count=int(stable_count),
            poll_interval_s=float(poll_interval_s),
        )
        return self.wait_until_tcp_reached(
            target,
            position_tolerance_mm=float(position_tolerance_mm),
            rotation_tolerance_deg=float(rotation_tolerance_deg),
            timeout_s=max(2.0, min(15.0, float(timeout_s))),
            stable_count=int(stable_count),
            poll_interval_s=float(poll_interval_s),
        )

    def close(self) -> None:
        planner = self._moveit
        self._moveit = None
        if planner is not None:
            try:
                planner.close()
            except Exception:
                pass
        super().close()
