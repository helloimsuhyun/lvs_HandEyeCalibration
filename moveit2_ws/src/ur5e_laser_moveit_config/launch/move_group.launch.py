from moveit_configs_utils import MoveItConfigsBuilder
from moveit_configs_utils.launches import generate_move_group_launch
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    config = (
        MoveItConfigsBuilder(
            "ur5e_laser",
            package_name="ur5e_laser_moveit_config",
        )
        .robot_description(
            file_path="urdf/ur5e_laser.urdf.xacro",
            mappings={
                "kinematics_parameters_file": LaunchConfiguration(
                    "kinematics_params_file"
                )
            },
        )
        .robot_description_semantic(file_path="config/ur5e_laser.srdf")
        .robot_description_kinematics(file_path="config/kinematics.yaml")
        .joint_limits(file_path="config/joint_limits.yaml")
        .planning_pipelines(
            pipelines=["ompl", "pilz_industrial_motion_planner"],
            default_planning_pipeline="ompl",
        )
        .pilz_cartesian_limits(file_path="config/pilz_cartesian_limits.yaml")
        .to_moveit_configs()
    )

    move_group = generate_move_group_launch(config)
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "kinematics_params_file",
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare("ur_description"),
                        "config",
                        "ur5e",
                        "default_kinematics.yaml",
                    ]
                ),
                description="UR factory kinematics YAML used by the planning model",
            ),
            *move_group.entities,
        ]
    )
