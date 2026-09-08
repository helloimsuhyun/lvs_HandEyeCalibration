"""RB5 real-hardware bringup without a second MoveGroup or RViz instance.

This follows rbpodo_bringup's official launch, but activates the
JointTrajectoryController required by the workflow adapter instead of the
default JointGroupPositionController.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, Shutdown
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    robot_ip = LaunchConfiguration("robot_ip")
    model_id = LaunchConfiguration("model_id")
    model_path = PathJoinSubstitution(
        [FindPackageShare("rbpodo_description"), "robots", [model_id, ".urdf.xacro"]]
    )
    robot_description = Command(
        [
            FindExecutable(name="xacro"), " ", model_path,
            " cb_simulation:=false robot_ip:=", robot_ip,
            " use_fake_hardware:=false fake_sensor_commands:=false",
        ]
    )
    controllers = PathJoinSubstitution(
        [FindPackageShare("rbpodo_bringup"), "config", "controllers.yaml"]
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("robot_ip", description="RB control-box IP address"),
            DeclareLaunchArgument("model_id", default_value="rb5_850e"),
            Node(
                package="controller_manager",
                executable="ros2_control_node",
                parameters=[controllers],
                remappings=[
                    ("joint_states", "rbpodo/joint_states"),
                    ("~/robot_description", "/robot_description"),
                ],
                output="both",
                on_exit=Shutdown(),
            ),
            Node(
                package="robot_state_publisher",
                executable="robot_state_publisher",
                name="robot_state_publisher",
                parameters=[{"robot_description": robot_description}],
                output="both",
            ),
            Node(
                package="joint_state_publisher",
                executable="joint_state_publisher",
                name="joint_state_publisher",
                parameters=[{"source_list": ["rbpodo/joint_states"], "rate": 30}],
            ),
            Node(
                package="controller_manager",
                executable="spawner",
                arguments=["joint_state_broadcaster", "--controller-manager-timeout", "30"],
                output="screen",
            ),
            Node(
                package="controller_manager",
                executable="spawner",
                arguments=["joint_trajectory_controller", "--controller-manager-timeout", "30"],
                output="screen",
            ),
        ]
    )
