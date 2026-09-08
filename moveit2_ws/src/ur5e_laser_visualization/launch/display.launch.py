from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.substitutions import Command
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    share = Path(get_package_share_directory("ur5e_laser_visualization"))
    xacro_file = share / "urdf" / "ur5e_ljv7080_preview.urdf.xacro"
    rviz_file = share / "rviz" / "ur5e_laser_preview.rviz"

    robot_description = ParameterValue(
        Command(["xacro ", str(xacro_file)]),
        value_type=str,
    )

    return LaunchDescription(
        [
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                output="screen",
                parameters=[{"robot_description": robot_description}],
            ),
            Node(
                package="joint_state_publisher_gui",
                executable="joint_state_publisher_gui",
                output="screen",
                parameters=[{"robot_description": robot_description}],
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                output="screen",
                arguments=["-d", str(rviz_file)],
            ),
        ]
    )
