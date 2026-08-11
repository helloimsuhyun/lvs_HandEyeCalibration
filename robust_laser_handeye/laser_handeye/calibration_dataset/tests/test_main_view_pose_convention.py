from __future__ import annotations

import math

import numpy as np

from main import generate_independent_random_plane_comparison as base
from main import generate_plane_uniform_comparison as uniform
from main import generate_single_uniform_vs_fisher as single


def _canonical_view_parameters(transform: np.ndarray) -> np.ndarray:
    rotation = transform[:3, :3]
    origin = transform[:3, 3]
    sensor_x = rotation[:, 0]
    sensor_z = rotation[:, 2]

    ray_distance = -float(origin[2]) / float(sensor_z[2])
    intersection = origin + ray_distance * sensor_z
    tilt = math.degrees(math.acos(np.clip(-sensor_z[2], -1.0, 1.0)))
    azimuth = (
        math.degrees(math.atan2(float(sensor_z[1]), float(sensor_z[0])))
        + 180.0
    ) % 360.0 - 180.0

    roll_reference = np.array([1.0, 0.0, 0.0])
    roll_reference -= float(roll_reference @ sensor_z) * sensor_z
    roll_reference /= np.linalg.norm(roll_reference)
    roll = math.degrees(
        math.atan2(
            float(sensor_z @ np.cross(roll_reference, sensor_x)),
            float(roll_reference @ sensor_x),
        )
    )
    return np.array(
        [
            intersection[0],
            intersection[1],
            abs(ray_distance),
            tilt,
            azimuth,
            roll,
        ]
    )


def test_main_generators_round_trip_canonical_view_parameters() -> None:
    frame = base.PlaneFrame(
        u=np.array([1.0, 0.0, 0.0]),
        v=np.array([0.0, 1.0, 0.0]),
        n=np.array([0.0, 0.0, 1.0]),
        l=0.0,
    )
    center = np.zeros(3)
    expected = np.array([12.0, -7.0, 100.0, 30.0, 40.0, 25.0])
    common = dict(
        target_u_mm=expected[0],
        target_v_mm=expected[1],
        center_depth_mm=expected[2],
        view_tilt_deg=expected[3],
        view_azimuth_deg=expected[4],
        sensor_roll_deg=expected[5],
        noise_seed=0,
        simulation_seed=0,
    )

    uniform_pose = uniform.UniformRelativePose(
        sample_id=0,
        block_id=0,
        index_in_block=0,
        **common,
    )
    single_pose = single.PlaneRelativeCandidate(candidate_id=0, **common)
    uniform_transform = uniform._make_sensor_pose_relative_to_plane(
        frame, center, uniform_pose
    )
    single_transform = single._make_sensor_pose_relative_to_plane(
        frame, center, single_pose
    )

    assert np.allclose(
        _canonical_view_parameters(uniform_transform), expected, atol=1e-12
    )
    assert np.allclose(
        _canonical_view_parameters(single_transform), expected, atol=1e-12
    )
    assert np.allclose(uniform_transform, single_transform, atol=1e-12)
