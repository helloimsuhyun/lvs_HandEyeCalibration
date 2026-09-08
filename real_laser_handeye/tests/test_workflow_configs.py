from __future__ import annotations

from pathlib import Path

import yaml

from real_laser_handeye.workflow import WorkflowConfig


CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_all_robot_mode_configs_have_consistent_interfaces():
    expected = {
        "sim_workflow.yaml": (
            ("base", "shoulder", "elbow", "wrist1", "wrist2", "wrist3"),
            "link0",
            "rb5_laser_moveit_config",
        ),
        "rb5_ljv7080_workflow.yaml": (
            ("base", "shoulder", "elbow", "wrist1", "wrist2", "wrist3"),
            "link0",
            "rb5_laser_moveit_config",
        ),
        "ur5e_sim_workflow.yaml": (
            (
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ),
            "world",
            "ur5e_laser_moveit_config",
        ),
        "ur5e_ljv7080_workflow.yaml": (
            (
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ),
            "world",
            "ur5e_laser_moveit_config",
        ),
    }

    for filename, (joints, base_frame, package) in expected.items():
        config = WorkflowConfig.load(CONFIGS / filename)
        moveit = config.values["planning"]["moveit"]
        assert config.joint_names == joints
        assert moveit["planning_frame"] == base_frame
        assert moveit["launch_package"] == package
        assert config.path("handeye").is_file()


def test_robot_mode_configs_keep_session_outputs_separate():
    filenames = (
        "sim_workflow.yaml",
        "rb5_ljv7080_workflow.yaml",
        "ur5e_sim_workflow.yaml",
        "ur5e_ljv7080_workflow.yaml",
    )
    configs = [WorkflowConfig.load(CONFIGS / filename) for filename in filenames]
    roots = [config.path("session_root") for config in configs]
    assert len(set(roots)) == len(roots)
    for config, root in zip(configs, roots):
        for key in (
            "initial_dataset",
            "estimate_dir",
            "scan_dataset",
            "planning_model",
            "scan_plan",
            "calibrated_transform",
        ):
            assert config.path(key).is_relative_to(root)


def test_real_configs_select_matching_ros_adapters():
    expected = {
        "rb5_ljv7080_workflow.yaml": "robot_adapter_rb5_ros",
        "ur5e_ljv7080_workflow.yaml": "robot_adapter_ur5e_ros",
    }
    for filename, module_name in expected.items():
        config = WorkflowConfig.load(CONFIGS / filename)
        assert module_name in config.values["equipment"]["robot_adapter"]
        assert config.values["equipment"]["laser_adapter"].endswith(":LaserAdapter")


def test_real_configs_auto_launch_matching_official_drivers():
    expected = {
        "rb5_ljv7080_workflow.yaml": (
            "rb5_laser_moveit_config",
            "real_driver.launch.py",
        ),
        "ur5e_ljv7080_workflow.yaml": (
            "ur_robot_driver",
            "ur_control.launch.py",
        ),
    }
    for filename, (package, launch_file) in expected.items():
        config = WorkflowConfig.load(CONFIGS / filename)
        driver = config.values["equipment"]["driver"]
        assert driver["auto_launch"] is True
        assert driver["launch_package"] == package
        assert driver["launch_file"] == launch_file
        assert driver["startup_timeout_s"] > 0

    ur = WorkflowConfig.load(CONFIGS / "ur5e_ljv7080_workflow.yaml")
    assert ur.values["equipment"]["driver"]["calibration"]["auto_extract"] is True


def test_sim_configs_inherit_matching_real_configs_with_sim_only_overrides():
    pairs = (
        ("sim_workflow.yaml", "rb5_ljv7080_workflow.yaml"),
        ("ur5e_sim_workflow.yaml", "ur5e_ljv7080_workflow.yaml"),
    )

    for sim_name, real_name in pairs:
        raw_sim = yaml.safe_load((CONFIGS / sim_name).read_text(encoding="utf-8"))
        raw_real = yaml.safe_load((CONFIGS / real_name).read_text(encoding="utf-8"))
        assert raw_sim["extends"] == real_name
        assert "extends" not in raw_real
        assert "simulation" not in raw_real

        sim = WorkflowConfig.load(CONFIGS / sim_name).values
        real = WorkflowConfig.load(CONFIGS / real_name).values
        for section in (
            "equipment",
            "sensor_frames",
            "planning",
            "visualization",
            "capture",
            "calibration",
        ):
            assert sim[section] == real[section]

        sim_motion = dict(sim["motion"])
        assert sim_motion.pop("simulated_delay_s") > 0.0
        assert sim_motion == real["motion"]
        assert sim["paths"]["handeye"] == real["paths"]["handeye"]
        assert sim["paths"]["mujoco_model"] == real["paths"]["mujoco_model"]
        assert sim["simulation"]["new_session_per_launch"] is True
