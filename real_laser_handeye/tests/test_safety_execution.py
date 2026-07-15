import numpy as np
import pytest

from real_laser_handeye.hardware import MockRobot
from real_laser_handeye.safety import AxisAlignedBox, SafetyConfig, validate_segment
from real_laser_handeye.workflow import move_validated_segment


def transform_at(x, y=0.0, z=0.0):
    T = np.eye(4)
    T[:3, 3] = [x, y, z]
    return T


def test_thin_no_go_box_cannot_be_skipped_between_waypoints():
    safety = SafetyConfig(
        workspace=AxisAlignedBox(np.array([-20.0, -20.0, -20.0]), np.array([20.0, 20.0, 20.0])),
        no_go_boxes=[
            AxisAlignedBox(
                np.array([4.9, -1.0, -1.0]),
                np.array([5.1, 1.0, 1.0]),
                "thin fixture",
            )
        ],
        max_linear_step_mm=20.0,
    )
    with pytest.raises(ValueError, match="intersects no-go box"):
        validate_segment(transform_at(0), transform_at(10), safety, name="test")


class NoCollisionQueryRobot(MockRobot):
    def controller_path_is_safe(self, poses):
        del poses
        return None


def test_required_controller_collision_query_must_return_true():
    robot = NoCollisionQueryRobot(transform_at(0).tolist())
    safety = SafetyConfig(
        workspace=AxisAlignedBox(np.array([-20.0, -20.0, -20.0]), np.array([20.0, 20.0, 20.0])),
        require_controller_collision_check=True,
    )
    with pytest.raises(RuntimeError, match="controller rejected"):
        move_validated_segment(robot, transform_at(1), safety, label="test")
