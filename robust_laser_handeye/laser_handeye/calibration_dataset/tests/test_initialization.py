from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from laser_handeye.initialization import make_initial_guess
from laser_handeye.se3 import euler_xyz_deg


REFERENCE_ANGLES_DEG = np.array([23.0, -17.0, 41.0])
REFERENCE_TRANSLATION_MM = np.array([120.0, -30.0, 75.0])
REFERENCE_ROTATION = euler_xyz_deg(*REFERENCE_ANGLES_DEG)


def _errors(T_initial: np.ndarray) -> tuple[float, float]:
    translation_error_mm = float(
        np.linalg.norm(T_initial[:3, 3] - REFERENCE_TRANSLATION_MM)
    )
    rotation_error_deg = float(
        np.rad2deg(
            Rotation.from_matrix(
                REFERENCE_ROTATION.T @ T_initial[:3, :3]
            ).magnitude()
        )
    )
    return translation_error_mm, rotation_error_deg


def _sample(seed: int, translation_mm: float, rotation_deg: float) -> np.ndarray:
    return make_initial_guess(
        None,
        REFERENCE_TRANSLATION_MM,
        reference_rotation=REFERENCE_ROTATION,
        rng=np.random.default_rng(seed),
        mode="carlson",
        translation_range_mm=translation_mm,
        angle_range_deg=rotation_deg,
        rotation_perturbation="axis_angle",
        translation_perturbation="direction_norm",
    )


def test_isotropic_initial_error_respects_exact_norm_and_angle_bounds() -> None:
    for seed in range(256):
        translation_error_mm, rotation_error_deg = _errors(
            _sample(seed, translation_mm=100.0, rotation_deg=15.0)
        )
        assert 0.0 <= translation_error_mm <= 100.0 + 1e-10
        assert 0.0 <= rotation_error_deg <= 15.0 + 1e-10


def test_isotropic_initial_error_is_reproducible_and_scales_by_level() -> None:
    easy = _sample(1701, translation_mm=25.0, rotation_deg=5.0)
    hard = _sample(1701, translation_mm=200.0, rotation_deg=30.0)
    repeated = _sample(1701, translation_mm=25.0, rotation_deg=5.0)

    assert np.array_equal(easy, repeated)

    easy_translation, easy_rotation = _errors(easy)
    hard_translation, hard_rotation = _errors(hard)
    assert np.isclose(hard_translation, 8.0 * easy_translation)
    assert np.isclose(hard_rotation, 6.0 * easy_rotation)


def test_zero_isotropic_bounds_return_the_reference_transform() -> None:
    initial = _sample(1701, translation_mm=0.0, rotation_deg=0.0)
    translation_error_mm, rotation_error_deg = _errors(initial)
    assert np.isclose(translation_error_mm, 0.0)
    assert np.isclose(rotation_error_deg, 0.0)
