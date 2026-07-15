from __future__ import annotations

from typing import Literal

import numpy as np

from .se3 import euler_xyz_deg, make_T


InitialGuessMode = Literal["relative", "carlson"]


def make_initial_guess(
    reference_angles_deg: np.ndarray,
    reference_translation_mm: np.ndarray,
    rng: np.random.Generator | None = None,
    mode: InitialGuessMode = "relative",
    rel_offset: float = 0.10,
    translation_range_mm: float = 200.0,
    angle_range_deg: float = 30.0,
    min_angle_offset_deg: float = 0.0,
    min_translation_offset_mm: float = 0.0,
) -> np.ndarray:
    """Generate an initial hand-eye transform around a reference transform.

    ``relative`` scales each reference parameter by an independent uniform
    perturbation in ``[-rel_offset, rel_offset]``. Optional minimum absolute
    perturbations can be added for parameters whose reference value is near
    zero.

    ``carlson`` adds independent absolute perturbations to Euler angles and
    translation using the supplied ranges.
    """
    rng = np.random.default_rng() if rng is None else rng

    angles_deg = np.asarray(reference_angles_deg, dtype=float).reshape(3)
    translation_mm = np.asarray(reference_translation_mm, dtype=float).reshape(3)

    if not np.all(np.isfinite(angles_deg)):
        raise ValueError("reference_angles_deg must contain only finite values")
    if not np.all(np.isfinite(translation_mm)):
        raise ValueError("reference_translation_mm must contain only finite values")

    rel_offset = float(rel_offset)
    translation_range_mm = float(translation_range_mm)
    angle_range_deg = float(angle_range_deg)
    min_angle_offset_deg = float(min_angle_offset_deg)
    min_translation_offset_mm = float(min_translation_offset_mm)

    if rel_offset < 0.0:
        raise ValueError("rel_offset must be non-negative")
    if translation_range_mm < 0.0:
        raise ValueError("translation_range_mm must be non-negative")
    if angle_range_deg < 0.0:
        raise ValueError("angle_range_deg must be non-negative")
    if min_angle_offset_deg < 0.0:
        raise ValueError("min_angle_offset_deg must be non-negative")
    if min_translation_offset_mm < 0.0:
        raise ValueError("min_translation_offset_mm must be non-negative")

    if mode == "relative":
        angle_delta_deg = angles_deg * rng.uniform(
            -rel_offset,
            rel_offset,
            size=3,
        )
        translation_delta_mm = translation_mm * rng.uniform(
            -rel_offset,
            rel_offset,
            size=3,
        )

        if min_angle_offset_deg > 0.0:
            angle_delta_deg += rng.uniform(
                -min_angle_offset_deg,
                min_angle_offset_deg,
                size=3,
            )
        if min_translation_offset_mm > 0.0:
            translation_delta_mm += rng.uniform(
                -min_translation_offset_mm,
                min_translation_offset_mm,
                size=3,
            )

    elif mode == "carlson":
        angle_delta_deg = rng.uniform(
            -angle_range_deg,
            angle_range_deg,
            size=3,
        )
        translation_delta_mm = rng.uniform(
            -translation_range_mm,
            translation_range_mm,
            size=3,
        )

    else:
        raise ValueError("mode must be 'relative' or 'carlson'")

    initial_angles_deg = angles_deg + angle_delta_deg
    initial_translation_mm = translation_mm + translation_delta_mm

    return make_T(
        euler_xyz_deg(*initial_angles_deg),
        initial_translation_mm,
    )