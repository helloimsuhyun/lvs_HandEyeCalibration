from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from ..geometry import fit_plane_pca
from ..se3 import rot_error_deg, transform_points
from .models import Tan2025Dataset, Tan2025GroundTruth


def euler_zyx_deg(rotation: np.ndarray) -> np.ndarray:
    """Return paper angles [alpha(Z), beta(Y), gamma(X)] in degrees."""
    matrix = np.asarray(rotation, dtype=float).reshape(3, 3)
    beta = float(np.arcsin(np.clip(-matrix[2, 0], -1.0, 1.0)))
    cosine_beta = float(np.cos(beta))
    if abs(cosine_beta) > 1e-9:
        alpha = float(np.arctan2(matrix[1, 0], matrix[0, 0]))
        gamma = float(np.arctan2(matrix[2, 1], matrix[2, 2]))
    else:
        alpha = float(np.arctan2(-matrix[0, 1], matrix[1, 1]))
        gamma = 0.0
    return np.degrees([alpha, beta, gamma])


def wrapped_angle_difference_deg(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    difference = np.asarray(first, dtype=float) - np.asarray(second, dtype=float)
    return (difference + 180.0) % 360.0 - 180.0


@dataclass(frozen=True)
class Tan2025Metrics:
    plane_normal_error_deg: float
    rotation_geodesic_error_deg: float
    rotation_euler_l1_error_deg: float
    translation_error_mm: float
    reconstruction_mean_error_mm: float
    reconstruction_rms_error_mm: float
    self_fitted_mpde_mm: float
    estimated_euler_alpha_deg: float
    estimated_euler_beta_deg: float
    estimated_euler_gamma_deg: float

    def to_dict(self) -> dict[str, float]:
        return {key: float(value) for key, value in asdict(self).items()}


def evaluate_tan2025_estimate(
    dataset: Tan2025Dataset,
    truth: Tan2025GroundTruth,
    T_ef_s_estimate: np.ndarray,
    *,
    estimated_plane_normal_base: np.ndarray | None = None,
) -> Tan2025Metrics:
    estimate = np.asarray(T_ef_s_estimate, dtype=float).reshape(4, 4)
    true = truth.T_ef_s

    if estimated_plane_normal_base is None:
        normal_error_deg = float("nan")
    else:
        estimated_normal = np.asarray(
            estimated_plane_normal_base, dtype=float
        ).reshape(3)
        estimated_normal /= np.linalg.norm(estimated_normal)
        cosine = np.clip(
            abs(float(estimated_normal @ truth.plane_normal_base)),
            -1.0,
            1.0,
        )
        normal_error_deg = float(np.degrees(np.arccos(cosine)))

    estimated_euler = euler_zyx_deg(estimate[:3, :3])
    true_euler = euler_zyx_deg(true[:3, :3])
    euler_difference = wrapped_angle_difference_deg(estimated_euler, true_euler)

    reconstruction_errors: list[np.ndarray] = []
    reconstructed_points: list[np.ndarray] = []
    for scan in dataset.all_scans:
        points = scan.valid_points_s
        if not len(points):
            continue
        estimated_ef = transform_points(estimate, points)
        true_ef = transform_points(true, points)
        estimated_base = transform_points(scan.T_base_ef, estimated_ef)
        true_base = transform_points(scan.T_base_ef, true_ef)
        reconstruction_errors.append(
            np.linalg.norm(estimated_base - true_base, axis=1)
        )
        reconstructed_points.append(estimated_base)
    if not reconstruction_errors:
        raise ValueError("dataset contains no finite profile points")
    errors = np.concatenate(reconstruction_errors)
    all_reconstructed = np.vstack(reconstructed_points)
    plane_normal, plane_offset, _, _ = fit_plane_pca(all_reconstructed)
    plane_distances = np.abs(all_reconstructed @ plane_normal - plane_offset)

    return Tan2025Metrics(
        plane_normal_error_deg=normal_error_deg,
        rotation_geodesic_error_deg=rot_error_deg(
            estimate[:3, :3], true[:3, :3]
        ),
        rotation_euler_l1_error_deg=float(np.sum(np.abs(euler_difference))),
        translation_error_mm=float(np.linalg.norm(estimate[:3, 3] - true[:3, 3])),
        reconstruction_mean_error_mm=float(np.mean(errors)),
        reconstruction_rms_error_mm=float(np.sqrt(np.mean(np.square(errors)))),
        self_fitted_mpde_mm=float(np.mean(plane_distances)),
        estimated_euler_alpha_deg=float(estimated_euler[0]),
        estimated_euler_beta_deg=float(estimated_euler[1]),
        estimated_euler_gamma_deg=float(estimated_euler[2]),
    )
