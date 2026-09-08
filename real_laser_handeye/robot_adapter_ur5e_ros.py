from __future__ import annotations

from .ros2_robot_adapter_base import ROS2JointTrajectoryRobotAdapter


class RobotAdapter(ROS2JointTrajectoryRobotAdapter):
    """Universal Robots UR5e using the official Universal Robots ROS 2 driver."""

    JOINT_NAMES = (
        "shoulder_pan_joint",
        "shoulder_lift_joint",
        "elbow_joint",
        "wrist_1_joint",
        "wrist_2_joint",
        "wrist_3_joint",
    )
    # Keep the measured TCP pose in the exact frame used by the project
    # MoveIt model.  The official driver's `base` frame differs from
    # REP-103 `base_link` by a fixed rotation.
    BASE_FRAME = "base_link"
    TCP_FRAME = "tool0"
    JOINT_STATE_TOPIC = "/joint_states"
    # The official driver normally starts the scaled controller, while some
    # deployments explicitly select the unscaled joint trajectory controller.
    TRAJECTORY_ACTIONS = (
        "/scaled_joint_trajectory_controller/follow_joint_trajectory",
        "/joint_trajectory_controller/follow_joint_trajectory",
    )
