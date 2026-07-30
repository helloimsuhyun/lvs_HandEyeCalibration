from __future__ import annotations

from pathlib import Path

import numpy as np

from mujoco_laser_handeye.compat.planning import (
    build_single_plane_plan,
    load_json,
    load_transform,
)
from robust_laser_handeye.laser_handeye.se3 import transform_points
from mujoco_laser_handeye.compat.workflow import (
    load_runtime_config,
    make_laser,
    make_robot,
    move_validated_segment,
)

from mujoco_laser_handeye.bootstrap import capture_automatic_mujoco_bootstrap


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "mujoco_laser_handeye/config/rb5_keyence_single_plane.json"
BOUNDARY = ROOT / "mujoco_laser_handeye/config/plane_boundary.json"
HANDEYE = ROOT / "mujoco_laser_handeye/config/initial_T_tcp_sensor.csv"


def _plan_and_hardware():
    config, safety, capture = load_runtime_config(CONFIG)
    plan = build_single_plane_plan(
        plane_boundary=load_json(BOUNDARY),
        T_tcp_sensor_init=load_transform(HANDEYE),
        safety=safety,
        pose_geometry="paper_incidence",
    )
    robot = make_robot(config)
    laser = make_laser(config, robot=robot)
    return safety, capture, plan, robot, laser


def test_rb5_model_starts_at_reviewed_safe_pose_without_collision():
    safety, _capture, _plan, robot, _laser = _plan_and_hardware()
    assert robot.collision_contacts() == []
    assert np.allclose(
        robot.current_T_base_tcp(), safety.safe_transit_T_base_tcp, atol=1e-6
    )


def test_rb5_ik_round_trip_and_known_table_collision():
    _safety, _capture, _plan, robot, _laser = _plan_and_hardware()
    q_target = np.radians([5.0, 27.0, 35.0, -62.0, 88.0, 3.0])
    target = robot.forward_kinematics(q_target)
    q_solved = robot.solve_ik(target)
    reached = robot.forward_kinematics(q_solved)
    assert np.linalg.norm(reached[:3, 3] - target[:3, 3]) < 2.1
    colliding_q = np.radians([0.0, 60.0, -60.0, 0.0, 90.0, 0.0])
    assert any("work_table" in item for item in robot.collision_contacts(colliding_q))


def test_keyence_profile_is_on_true_plane_at_safe_scan_pose():
    safety, _capture, plan, robot, laser = _plan_and_hardware()
    robot.connect()
    laser.connect()
    try:
        entry = plan["entries"][0]
        move_validated_segment(
            robot,
            np.asarray(entry["T_base_tcp_approach"]),
            safety,
            label="test approach",
        )
        move_validated_segment(
            robot,
            np.asarray(entry["T_base_tcp"]),
            safety,
            label="test target",
        )
        sample = laser.capture_profile(timeout_s=1.0)
        points_base = transform_points(
            robot.current_T_base_tcp() @ robot.T_tcp_sensor_true,
            sample.points_s,
        )
        distances = (
            points_base - robot.T_base_plane[:3, 3]
        ) @ robot.T_base_plane[:3, 2]
        assert len(points_base) >= 100
        assert float(np.sqrt(np.mean(distances**2))) < 0.1
    finally:
        laser.close()
        robot.close()


def test_automatic_bootstrap_captures_four_views_and_builds_observable_plan(tmp_path):
    safety, capture, _plan, robot, laser = _plan_and_hardware()
    handeye = load_transform(HANDEYE)
    robot.connect()
    laser.connect()
    try:
        boundary, report = capture_automatic_mujoco_bootstrap(
            robot=robot,
            laser=laser,
            T_tcp_sensor_init=handeye,
            output_dir=tmp_path / "bootstrap",
            safety=safety,
            capture=capture,
            margin_mm=20.0,
        )
    finally:
        laser.close()
        robot.close()

    assert report["capture_count"] == 4
    assert report["plane_rms_mm"] < capture.max_bootstrap_plane_rms_mm
    assert report["simulation_gt_diagnostics"]["normal_error_deg"] < 0.2
    assert (tmp_path / "bootstrap/plane_boundary.json").is_file()
    assert (tmp_path / "bootstrap/bootstrap_motion_plan.json").is_file()
    assert len(load_json(tmp_path / "bootstrap/bootstrap_manifest.json")["captures"]) == 4

    observed = boundary["observed_bounds_uv_mm"]
    assert observed["u_max"] - observed["u_min"] > 200.0
    assert observed["v_max"] - observed["v_min"] > 200.0
    plan = build_single_plane_plan(
        plane_boundary=boundary,
        T_tcp_sensor_init=handeye,
        safety=safety,
        pose_geometry="paper_incidence",
    )
    assert len(plan["entries"]) == 105
    assert plan["observability"]["rank"] == 4
