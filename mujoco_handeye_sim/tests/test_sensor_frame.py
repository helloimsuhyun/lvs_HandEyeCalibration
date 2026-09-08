import numpy as np

from handeye_mujoco.builder import _sensor_cad_transform


FRAME = {
    "frame": {
        "origin_in_cad": [-52.6428785, -97.6682626, 12.7832449],
        "x_axis_in_cad": [0.0, 0.0, 1.0],
        "y_axis_in_cad": [1.0, 0.0, 0.0],
        "z_axis_in_cad": [0.0, 1.0, 0.0],
    }
}


def transform_point(transform, point_mm):
    return transform[:3, :3] @ (np.asarray(point_mm) * 1e-3) + transform[:3, 3]


def test_lj_v7080_trapezoid_center_is_sensor_origin():
    transform = _sensor_cad_transform(FRAME, 1e-3)
    np.testing.assert_allclose(
        transform_point(transform, FRAME["frame"]["origin_in_cad"]),
        [0, 0, 0],
        atol=1e-12,
    )


def test_lj_v7080_near_and_far_planes_are_plus_minus_23_mm():
    transform = _sensor_cad_transform(FRAME, 1e-3)
    near = [-52.6428785, -74.6682626, 12.7832449]
    far = [-52.6428785, -120.6682626, 12.7832449]
    np.testing.assert_allclose(transform_point(transform, near), [0, 0, 0.023], atol=1e-9)
    np.testing.assert_allclose(transform_point(transform, far), [0, 0, -0.023], atol=1e-9)


def test_lj_v7080_reference_profile_width_is_32_mm():
    transform = _sensor_cad_transform(FRAME, 1e-3)
    left = [-52.6428785, -97.6682626, 12.7832449 - 16.0]
    right = [-52.6428785, -97.6682626, 12.7832449 + 16.0]
    np.testing.assert_allclose(transform_point(transform, left), [-0.016, 0, 0], atol=1e-9)
    np.testing.assert_allclose(transform_point(transform, right), [0.016, 0, 0], atol=1e-9)
