from __future__ import annotations

import numpy as np
import pytest

from laser_handeye.pose_design import (
    PlaneFrame,
    PlaneRelativePose,
    PlaneRelativePoseBounds,
    sensor_pose_from_plane_relative,
)


def _frame() -> PlaneFrame:
    normal = np.asarray([0.2, -0.3, 0.9327379053088815])
    normal /= np.linalg.norm(normal)
    u = np.asarray([1.0, 0.0, 0.0])
    u -= float(u @ normal) * normal
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    return PlaneFrame(u, v, normal, 420.0)


@pytest.mark.parametrize("tilt_deg", [10.0, 32.0, 55.0])
@pytest.mark.parametrize("view_azimuth_deg", [-170.0, -20.0, 95.0])
@pytest.mark.parametrize("normal_azimuth_sensor_deg", [-150.0, 0.0, 73.0, 179.0])
def test_plane_normal_sensor_coordinates_are_directly_parameterized(
    tilt_deg: float,
    view_azimuth_deg: float,
    normal_azimuth_sensor_deg: float,
) -> None:
    frame = _frame()
    center = 420.0 * frame.n
    pose = PlaneRelativePose(
        sample_id=0,
        target_u_mm=13.0,
        target_v_mm=-7.0,
        distance_mm=105.0,
        tilt_deg=tilt_deg,
        azimuth_deg=view_azimuth_deg,
        normal_azimuth_sensor_deg=normal_azimuth_sensor_deg,
    )
    transform = sensor_pose_from_plane_relative(frame, center, pose)
    rotation = transform[:3, :3]
    b = rotation.T @ frame.n
    tilt = np.deg2rad(tilt_deg)
    alpha = np.deg2rad(normal_azimuth_sensor_deg)
    expected_b = np.asarray(
        [
            np.sin(tilt) * np.cos(alpha),
            np.sin(tilt) * np.sin(alpha),
            -np.cos(tilt),
        ]
    )

    np.testing.assert_allclose(b, expected_b, atol=1e-12)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0, abs=1e-12)

    recovered = np.degrees(np.arctan2(b[1], b[0]))
    angular_error = (recovered - normal_azimuth_sensor_deg + 180.0) % 360.0 - 180.0
    assert angular_error == pytest.approx(0.0, abs=1e-10)

    target = center + pose.target_u_mm * frame.u + pose.target_v_mm * frame.v
    np.testing.assert_allclose(
        transform[:3, 3] + pose.distance_mm * rotation[:, 2],
        target,
        atol=1e-12,
    )


def test_view_azimuth_is_remaining_orientation_dof_at_fixed_b() -> None:
    frame = _frame()
    center = 420.0 * frame.n
    transforms = []
    for view_azimuth in (-140.0, -15.0, 80.0, 170.0):
        pose = PlaneRelativePose(
            0,
            0.0,
            0.0,
            100.0,
            35.0,
            view_azimuth,
            42.0,
        )
        transforms.append(sensor_pose_from_plane_relative(frame, center, pose))

    b = np.stack([transform[:3, :3].T @ frame.n for transform in transforms])
    np.testing.assert_allclose(b, np.repeat(b[:1], len(b), axis=0), atol=1e-12)
    assert not np.allclose(transforms[0][:3, :3], transforms[1][:3, :3])


def test_zero_tilt_is_rejected_because_normal_azimuth_is_undefined() -> None:
    with pytest.raises(ValueError, match="undefined at zero tilt"):
        PlaneRelativePoseBounds(tilt_deg=(0.0, 50.0))

    frame = _frame()
    pose = PlaneRelativePose(0, 0.0, 0.0, 100.0, 0.0, 0.0, 20.0)
    with pytest.raises(ValueError, match="undefined at zero tilt"):
        sensor_pose_from_plane_relative(frame, 420.0 * frame.n, pose)
