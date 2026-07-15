from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from real_laser_handeye.hardware import MockPlanarLaser, MockRobot
from real_laser_handeye.planning import build_single_plane_plan
from real_laser_handeye.safety import AxisAlignedBox, SafetyConfig
from real_laser_handeye.workflow import CaptureConfig, calibrate_dataset, collect_plan


def scene():
    T_base_plane = np.eye(4)
    T_base_plane[:3, 3] = [50.0, -20.0, 400.0]
    boundary = {
        "plane_frame": {"T_base_plane": T_base_plane.tolist()},
        "safe_bounds_uv_mm": {
            "u_min": -150.0,
            "u_max": 150.0,
            "v_min": -150.0,
            "v_max": 150.0,
        },
    }
    T_true = np.eye(4)
    T_true[:3, :3] = Rotation.from_euler(
        "xyz", [4.0, -7.0, 12.0], degrees=True
    ).as_matrix()
    T_true[:3, 3] = [35.0, -12.0, 70.0]
    T_init = T_true.copy()
    T_init[:3, :3] = T_true[:3, :3] @ Rotation.from_euler(
        "xyz", [1.5, -1.0, 2.0], degrees=True
    ).as_matrix()
    T_init[:3, 3] += [3.0, -2.0, 5.0]
    transit = np.eye(4)
    transit[:3, 3] = [0.0, 0.0, 750.0]
    safety = SafetyConfig(
        workspace=AxisAlignedBox(
            np.array([-800.0, -800.0, -100.0]),
            np.array([800.0, 800.0, 1000.0]),
        ),
        safe_transit_T_base_tcp=transit,
        approach_clearance_mm=50.0,
        max_linear_step_mm=100.0,
        max_angular_step_deg=30.0,
        settle_time_s=0.0,
        readback_position_tolerance_mm=0.01,
        readback_rotation_tolerance_deg=0.01,
        stationarity_position_tolerance_mm=0.01,
        stationarity_rotation_tolerance_deg=0.01,
        min_sensor_plane_clearance_mm=10.0,
    )
    return T_base_plane, boundary, T_true, T_init, transit, safety


def test_fixed_theta_only_plan_is_rejected():
    _plane, boundary, _true, T_init, _transit, safety = scene()
    with pytest.raises(ValueError, match="cannot separate hand-eye translation"):
        build_single_plane_plan(
            plane_boundary=boundary,
            T_tcp_sensor_init=T_init,
            safety=safety,
            reference_scan_count=0,
        )


def test_mock_105_pose_collection_and_joint_calibration(tmp_path):
    T_plane, boundary, T_true, T_init, transit, safety = scene()
    plan = build_single_plane_plan(
        plane_boundary=boundary,
        T_tcp_sensor_init=T_init,
        safety=safety,
    )
    assert len(plan["entries"]) == 105
    assert plan["observability"]["rank"] == 4

    robot = MockRobot(transit.tolist())
    laser = MockPlanarLaser(
        robot,
        T_true,
        T_plane[:3, 2],
        float(T_plane[:3, 2] @ T_plane[:3, 3]),
        point_count=101,
        noise_std_mm=0.2,
        seed=3,
    )
    collect_plan(
        robot=robot,
        laser=laser,
        plan=plan,
        output_dir=tmp_path,
        safety=safety,
        capture=CaptureConfig(
            timeout_s=1.0,
            min_points=80,
            max_profile_age_ms=1000.0,
            max_initial_plane_rms_mm=20.0,
        ),
    )
    output = tmp_path / "T_tcp_sensor.csv"
    diagnostics = calibrate_dataset(
        dataset_dir=tmp_path,
        T_tcp_sensor_init=T_init,
        output_transform=output,
        max_iter=30,
    )
    estimate = np.loadtxt(output, delimiter=",")
    translation_error = np.linalg.norm(estimate[:3, 3] - T_true[:3, 3])
    rotation_error = np.degrees(
        Rotation.from_matrix(estimate[:3, :3] @ T_true[:3, :3].T).magnitude()
    )
    assert diagnostics["scan_count"] == 105
    assert diagnostics["nonlinear_refinement"] is False
    assert translation_error < 0.1
    assert rotation_error < 0.1

    rejected_output = tmp_path / "T_tcp_sensor_rejected.csv"
    with pytest.raises(RuntimeError, match="acceptance limit"):
        calibrate_dataset(
            dataset_dir=tmp_path,
            T_tcp_sensor_init=T_init,
            output_transform=rejected_output,
            max_iter=30,
            max_final_plane_rms_mm=1e-6,
        )
    assert not rejected_output.exists()
