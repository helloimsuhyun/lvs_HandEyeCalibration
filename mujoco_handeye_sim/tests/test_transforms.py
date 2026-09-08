import numpy as np

from handeye_mujoco.transforms import matrix_to_quaternion, validate_transform, xyz_rpy_transform


def test_identity_quaternion():
    np.testing.assert_allclose(matrix_to_quaternion(np.eye(3)), [1, 0, 0, 0])


def test_urdf_transform_translation():
    transform = xyz_rpy_transform([1, 2, 3], [0, 0, 0])
    np.testing.assert_allclose(transform[:3, 3], [1, 2, 3])


def test_validate_transform_rejects_reflection():
    transform = np.eye(4)
    transform[0, 0] = -1
    try:
        validate_transform(transform)
    except ValueError:
        pass
    else:
        raise AssertionError("reflection must be rejected")
