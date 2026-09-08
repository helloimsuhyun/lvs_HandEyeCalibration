from moveit_configs_utils import MoveItConfigsBuilder
from moveit_configs_utils.launches import generate_move_group_launch


def generate_launch_description():
    config = (
        MoveItConfigsBuilder(
            "rb5_laser",
            package_name="rb5_laser_moveit_config",
        )
        .robot_description(
            file_path="config/rb5_laser.urdf"
        )
        .robot_description_semantic(
            file_path="config/rb5_laser.srdf"
        )
        .robot_description_kinematics(
            file_path="config/kinematics.yaml"
        )
        .joint_limits(
            file_path="config/joint_limits.yaml"
        )
        .planning_pipelines(
            pipelines=["ompl", "pilz_industrial_motion_planner"],
            default_planning_pipeline="ompl",
        )
        .pilz_cartesian_limits(
            file_path="config/pilz_cartesian_limits.yaml"
        )
        .to_moveit_configs()
    )

    return generate_move_group_launch(config)
