import numpy as np

from real_laser_handeye.profile_conversion import keyence_raw_to_points


def test_keyence_conversion_median_and_invalid_values():
    raw = np.array(
        [
            [32768, 0, 32778],
            [32770, 0, 32780],
            [32772, 0, 32782],
        ],
        dtype=np.int32,
    )
    points = keyence_raw_to_points(
        raw_z=raw,
        x_count=3,
        profile_count=3,
        x_start_raw=-100000,
        x_pitch_raw=100000,
        z_unit_raw=100,
        aggregate="median",
    )
    assert points.shape == (2, 3)
    np.testing.assert_allclose(points[:, 0], [-1.0, 1.0])
    np.testing.assert_allclose(points[:, 1], 0.0)
    np.testing.assert_allclose(points[:, 2], [0.002, 0.012])
