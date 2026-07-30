from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.spatial.transform import Rotation

from .se3 import euler_xyz_deg, make_T


InitialGuessMode = Literal["relative", "carlson"]
RotationPerturbation = Literal["euler_xyz", "axis_angle"]
TranslationPerturbation = Literal["box_xyz", "direction_norm"]


def _sample_unit_direction(rng: np.random.Generator) -> np.ndarray:
    """Sample a direction uniformly on the unit sphere."""
    direction = rng.normal(size=3)
    direction_norm = float(np.linalg.norm(direction))
    while direction_norm == 0.0:
        direction = rng.normal(size=3)
        direction_norm = float(np.linalg.norm(direction))
    return direction / direction_norm


def _sample_axis_angle_rotation(
    rng: np.random.Generator,
    max_angle_deg: float,
) -> np.ndarray:
    """Sample an isotropic rotation with angle uniformly bounded in SO(3)."""
    axis = _sample_unit_direction(rng)
    angle_rad = np.deg2rad(rng.uniform(0.0, max_angle_deg))
    return Rotation.from_rotvec(axis * angle_rad).as_matrix()


def _sample_direction_norm_translation(
    rng: np.random.Generator,
    max_norm_mm: float,
) -> np.ndarray:
    """Sample an isotropic translation with uniformly distributed norm."""
    direction = _sample_unit_direction(rng)
    norm_mm = rng.uniform(0.0, max_norm_mm)
    return direction * norm_mm


def make_initial_guess(
    reference_angles_deg: np.ndarray | None,
    reference_translation_mm: np.ndarray,
    rng: np.random.Generator | None = None,
    mode: InitialGuessMode = "relative",
    rel_offset: float = 0.10,
    translation_range_mm: float = 200.0,
    angle_range_deg: float = 30.0,
    min_angle_offset_deg: float = 0.0,
    min_translation_offset_mm: float = 0.0,
    rotation_perturbation: RotationPerturbation = "euler_xyz",
    translation_perturbation: TranslationPerturbation = "box_xyz",
    reference_rotation: np.ndarray | None = None,
) -> np.ndarray:
    """Generate an initial hand-eye transform around a reference transform.

    ``relative`` scales each reference parameter by an independent uniform
    perturbation in ``[-rel_offset, rel_offset]``. Optional minimum absolute
    perturbations can be added for parameters whose reference value is near
    zero.

    ``reference_rotation`` may be supplied to avoid an Euler round trip.
    ``reference_angles_deg`` may then be ``None`` for the axis-angle mode,
    while the relative and Euler-perturbation modes still require Euler
    reference angles.

    ``carlson`` adds absolute rotation and translation perturbations using
    the supplied ranges. With ``rotation_perturbation="euler_xyz"``, each
    Euler angle is perturbed independently in ``[-angle_range_deg,
    angle_range_deg]``. With ``rotation_perturbation="axis_angle"``, an
    isotropic axis is sampled uniformly on the unit sphere and a non-negative
    angle is sampled uniformly in ``[0, angle_range_deg]``. The sampled
    rotation is right-composed with the reference rotation, so the requested
    angle range is an exact geodesic-error bound.

    With ``translation_perturbation="box_xyz"``, translation remains an
    independent per-axis box perturbation. With
    ``translation_perturbation="direction_norm"``, an isotropic direction is
    sampled uniformly on the unit sphere and the translation norm is sampled
    uniformly in ``[0, translation_range_mm]``.
    """
    rng = np.random.default_rng() if rng is None else rng

    angles_deg = (
        None
        if reference_angles_deg is None
        else np.asarray(reference_angles_deg, dtype=float).reshape(3)
    )
    translation_mm = np.asarray(
        reference_translation_mm,
        dtype=float,
    ).reshape(3)

    if angles_deg is not None and not np.all(np.isfinite(angles_deg)):
        raise ValueError("reference_angles_deg must contain only finite values")
    if not np.all(np.isfinite(translation_mm)):
        raise ValueError(
            "reference_translation_mm must contain only finite values"
        )

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

    if reference_rotation is None:
        if angles_deg is None:
            raise ValueError(
                "reference_angles_deg is required when reference_rotation "
                "is not supplied"
            )
        reference_rotation_matrix = euler_xyz_deg(*angles_deg)
    else:
        reference_rotation_matrix = np.asarray(
            reference_rotation,
            dtype=float,
        ).reshape(3, 3)
        if not np.all(np.isfinite(reference_rotation_matrix)):
            raise ValueError(
                "reference_rotation must contain only finite values"
            )
        if not np.allclose(
            reference_rotation_matrix.T @ reference_rotation_matrix,
            np.eye(3),
            atol=1e-8,
        ) or not np.isclose(
            np.linalg.det(reference_rotation_matrix),
            1.0,
            atol=1e-8,
        ):
            raise ValueError("reference_rotation must belong to SO(3)")

    if mode == "relative":
        if angles_deg is None:
            raise ValueError("relative mode requires reference_angles_deg")
        if rotation_perturbation != "euler_xyz":
            raise ValueError(
                "relative mode supports only rotation_perturbation='euler_xyz'"
            )
        if translation_perturbation != "box_xyz":
            raise ValueError(
                "relative mode supports only translation_perturbation='box_xyz'"
            )
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
        if rotation_perturbation == "euler_xyz":
            if angles_deg is None:
                raise ValueError(
                    "euler_xyz perturbation requires reference_angles_deg"
                )
            angle_delta_deg = rng.uniform(
                -angle_range_deg,
                angle_range_deg,
                size=3,
            )
            initial_rotation = euler_xyz_deg(*(angles_deg + angle_delta_deg))
        elif rotation_perturbation == "axis_angle":
            initial_rotation = (
                reference_rotation_matrix
                @ _sample_axis_angle_rotation(rng, angle_range_deg)
            )
        else:
            raise ValueError(
                "rotation_perturbation must be 'euler_xyz' or 'axis_angle'"
            )
        if translation_perturbation == "box_xyz":
            translation_delta_mm = rng.uniform(
                -translation_range_mm,
                translation_range_mm,
                size=3,
            )
        elif translation_perturbation == "direction_norm":
            translation_delta_mm = _sample_direction_norm_translation(
                rng,
                translation_range_mm,
            )
        else:
            raise ValueError(
                "translation_perturbation must be 'box_xyz' or "
                "'direction_norm'"
            )

    else:
        raise ValueError("mode must be 'relative' or 'carlson'")

    initial_translation_mm = translation_mm + translation_delta_mm

    if mode == "relative":
        initial_rotation = euler_xyz_deg(*(angles_deg + angle_delta_deg))

    return make_T(
        initial_rotation,
        initial_translation_mm,
    )
