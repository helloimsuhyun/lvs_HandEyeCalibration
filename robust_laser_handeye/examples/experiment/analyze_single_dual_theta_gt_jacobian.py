"""
# Search both the number of theta values K and their values.
# With candidates 0..60 deg and K=1..3, this evaluates
# C(7,1)+C(7,2)+C(7,3)=63 theta sets while keeping 81 scans.
PYTHONPATH=. python examples/experiment/analyze_single_dual_theta_gt_jacobian.py \
  --seed 7 \
  --profile-points 100 \
  --heights-mm 60 90 120 \
  --beta-deg 60 90 120 \
  --single-theta-deg 30 \
  --dual-theta-1-deg 30 \
  --dual-theta-2-deg 60 \
  --pose-geometry paper_incidence \
  --noise-std 0.0 \
  --plane-offset-mode fitted \
  --search-theta-sets \
  --theta-search-candidates-deg 0 10 20 30 40 50 60 \
  --theta-search-min-count 1 \
  --theta-search-max-count 3 \
  --theta-search-systems 20 \
  --theta-search-top-k 30 \
  --skip-iteration-trace \
  --output-dir results/theta_set_search_K1_K3

# Optional rigorous validation: evaluate every cyclic line assignment for each set.
# This is slower, especially for K=3.
# Add: --theta-search-all-cyclic-assignments
"""


from __future__ import annotations

"""
Compare single-theta and multi-theta observability at the same ground truth.

This script performs four analyses:

1. Joint local Jacobian
       J_joint = [J_HE, J_plane]
   for the 9 local parameters
       [dphi(3), dt_E(3), dn(2), dl(1)].

2. Plane-eliminated effective hand-eye Jacobian
       J_eff = (I - P_plane) J_HE
   which answers:
       "After the plane is refitted optimally, which hand-eye perturbations
        still change the residual?"

3. Focused sensor-Z / plane-offset block
       J_zl = [j_tz_sensor, j_l]
   which directly tests the suspected ambiguity.

4. Jacobian of the actual alternating one-step map
       T_{k+1} = F(T_k)
   around the ground truth. Its spectral radius describes local convergence.

The single- and multi-theta datasets use the same:
    - ground-truth hand-eye transform,
    - calibration plane,
    - d and beta values,
    - number of retained target scans.

Default comparison:
    single: theta = 30 deg on all nine lines
    multi:  theta = 30,60,30,60,30,60,30,60,30 deg

Optional search:
    jointly search the theta count K and the selected theta values while
    keeping one theta assignment per line and 81 retained scans.
"""

import argparse
import csv
from itertools import combinations
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

from laser_handeye.calibration import calibrate_planes
from laser_handeye.data import LaserScan
from laser_handeye.patterns import scan_parameter_grid
from laser_handeye.se3 import transform_points
from laser_handeye.simulation import (
    generate_circular_pattern_scans,
    sample_random_handeye,
)


STATE_LABELS = (
    "dphi_x",
    "dphi_y",
    "dphi_z",
    "dt_E_x",
    "dt_E_y",
    "dt_E_z",
    "dn_u",
    "dn_v",
    "dl",
)

HAND_EYE_LABELS = STATE_LABELS[:6]


@dataclass
class MatrixDiagnostics:
    matrix: np.ndarray
    rank: int
    singular_values: np.ndarray
    condition: float
    min_singular_value: float
    normalized_matrix: np.ndarray
    normalized_rank: int
    normalized_singular_values: np.ndarray
    normalized_condition: float
    normalized_min_singular_value: float
    column_norms: np.ndarray
    normalized_weakest_vector: np.ndarray


@dataclass
class DatasetAnalysis:
    name: str
    scans: list[LaserScan]
    joint: MatrixDiagnostics
    effective: MatrixDiagnostics
    focused_tz_offset: MatrixDiagnostics
    joint_null_vector: np.ndarray
    effective_weakest_vector_physical: np.ndarray
    effective_weakest_translation_sensor: np.ndarray
    effective_weakest_translation_angle_to_sensor_z_deg: float
    sensor_z_effective_gain: float
    sensor_x_effective_gain: float
    scan_coefficients: np.ndarray
    scan_thetas_deg: np.ndarray
    iteration_spectral_radius: float
    iteration_fixed_point_rotation_defect_rad: float
    iteration_fixed_point_translation_defect_mm: float
    iteration_eigenvalue_magnitudes: np.ndarray
    iteration_dominant_eigenvalue: complex
    iteration_slowest_state_direction: np.ndarray
    iteration_slowest_state_direction_is_complex: bool
    iteration_slowest_translation_direction_ef: np.ndarray
    iteration_slowest_translation_direction_sensor: np.ndarray
    iteration_slowest_translation_angle_to_sensor_axes_deg: np.ndarray


@dataclass
class IterationTrace:
    """Actual repeated-solver trajectory from a translation perturbation."""

    name: str
    iterations: np.ndarray
    translation_error_mm: np.ndarray
    slow_projection_signed_mm: np.ndarray
    slow_projection_abs_mm: np.ndarray
    slow_projection_ratio: np.ndarray
    slow_direction_sensor: np.ndarray
    expected_spectral_radius: float
    tail_median_ratio: float


def sample_random_plane_pose(
    rng: np.random.Generator,
    tilt_min_deg: float,
    tilt_max_deg: float,
    yaw_min_deg: float,
    yaw_max_deg: float,
    center_xy_range_mm: float,
    center_z_min_mm: float,
    center_z_max_mm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Generate one plane pose using the convention of the benchmark script."""

    if tilt_min_deg < 0.0 or tilt_max_deg < tilt_min_deg:
        raise ValueError("invalid plane tilt range")
    if center_z_max_mm < center_z_min_mm:
        raise ValueError("invalid plane center-z range")

    def signed_tilt() -> float:
        magnitude = float(rng.uniform(tilt_min_deg, tilt_max_deg))
        return magnitude if rng.random() >= 0.5 else -magnitude

    angles_deg = np.array(
        [
            signed_tilt(),
            signed_tilt(),
            rng.uniform(yaw_min_deg, yaw_max_deg),
        ],
        dtype=float,
    )
    plane_R = Rotation.from_euler("xyz", angles_deg, degrees=True).as_matrix()
    plane_t = np.array(
        [
            rng.uniform(-center_xy_range_mm, center_xy_range_mm),
            rng.uniform(-center_xy_range_mm, center_xy_range_mm),
            rng.uniform(center_z_min_mm, center_z_max_mm),
        ],
        dtype=float,
    )

    plane_n = np.asarray(plane_R[:, 2], dtype=float)
    plane_n /= np.linalg.norm(plane_n)
    plane_l = float(plane_n @ plane_t)

    if plane_l < 0.0:
        plane_n = -plane_n
        plane_l = -plane_l
        plane_R = plane_R.copy()
        plane_R[:, 0] = -plane_R[:, 0]
        plane_R[:, 2] = -plane_R[:, 2]

    return plane_R, plane_t, plane_n, plane_l


def make_scan_params(
    heights_mm: tuple[float, ...],
    theta_deg: tuple[float, ...],
    beta_deg: tuple[float, ...],
) -> list[dict]:
    return scan_parameter_grid(
        heights_mm=tuple(float(v) for v in heights_mm),
        projection_deg=tuple(float(v) for v in theta_deg),
        tilt_deg=tuple(float(v) for v in beta_deg),
    )


def theta_from_param(param: dict) -> float:
    for key in ("theta_deg", "projection_deg", "projection_angle_deg", "theta"):
        if key in param:
            return float(param[key])
    raise KeyError(f"theta is missing from scan parameter: {param}")


def generate_dataset(
    *,
    T_true: np.ndarray,
    plane_R: np.ndarray,
    plane_t: np.ndarray,
    x_values: np.ndarray,
    radius_mm: float,
    noise_std: float,
    check_reachability: bool,
    heights_mm: tuple[float, ...],
    beta_deg: tuple[float, ...],
    theta_pool_deg: tuple[float, ...],
    theta_by_line_deg: tuple[float, ...] | None,
    pose_geometry: str,
    rng: np.random.Generator,
) -> list[LaserScan]:
    """Generate a circular dataset and optionally retain one theta per line."""

    scan_params = make_scan_params(
        heights_mm=heights_mm,
        theta_deg=theta_pool_deg,
        beta_deg=beta_deg,
    )
    scans = generate_circular_pattern_scans(
        plane_R=plane_R,
        plane_t=plane_t,
        T_ef_s_true=T_true,
        radius_mm=radius_mm,
        x_values=x_values,
        noise_std=noise_std,
        rng=rng,
        scan_params=scan_params,
        check_reachability=check_reachability,
        plane_id=0,
        pose_geometry=pose_geometry,
    )

    parameter_thetas = np.asarray(
        [theta_from_param(param) for param in scan_params],
        dtype=float,
    )

    if theta_by_line_deg is None:
        for scan in scans:
            parameter_id = int(scan.meta["parameter_id"])
            scan.meta["assigned_theta_deg"] = float(
                parameter_thetas[parameter_id]
            )
        retained = scans
    else:
        if len(theta_by_line_deg) != 9:
            raise ValueError("theta_by_line_deg must contain exactly 9 values")

        retained = []
        count_by_line = np.zeros(9, dtype=int)
        for scan in scans:
            line_id = int(scan.meta.get("line_id", -1))
            parameter_id = int(scan.meta.get("parameter_id", -1))
            if not 0 <= line_id < 9:
                raise ValueError("scan.meta['line_id'] is invalid")
            if not 0 <= parameter_id < len(scan_params):
                raise ValueError("scan.meta['parameter_id'] is invalid")

            assigned_theta = float(theta_by_line_deg[line_id])
            if np.isclose(
                parameter_thetas[parameter_id],
                assigned_theta,
                atol=1e-9,
                rtol=0.0,
            ):
                scan.meta["assigned_theta_deg"] = assigned_theta
                retained.append(scan)
                count_by_line[line_id] += 1

        if np.any(count_by_line == 0):
            missing = np.flatnonzero(count_by_line == 0)
            raise RuntimeError(
                "theta filtering removed all scans from line(s): "
                + ", ".join(map(str, missing))
            )

    for scan_id, scan in enumerate(retained):
        scan.scan_id = scan_id

    if not retained:
        raise RuntimeError("no valid scans were generated")
    return retained


def plane_tangent_basis(plane_n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = np.asarray(plane_n, dtype=float).reshape(3)
    n /= np.linalg.norm(n)

    axis = np.eye(3)[int(np.argmin(np.abs(n)))]
    tangent_u = axis - float(axis @ n) * n
    tangent_u /= np.linalg.norm(tangent_u)
    tangent_v = np.cross(n, tangent_u)
    tangent_v /= np.linalg.norm(tangent_v)
    return tangent_u, tangent_v


def residuals(
    scans: list[LaserScan],
    T_ef_s: np.ndarray,
    plane_n: np.ndarray,
    plane_l: float,
) -> np.ndarray:
    n = np.asarray(plane_n, dtype=float).reshape(3)
    n /= np.linalg.norm(n)

    residual_sets: list[np.ndarray] = []
    for scan in scans:
        points_s = np.asarray(scan.valid_points_s, dtype=float)
        if points_s.size == 0:
            continue
        points_ef = transform_points(T_ef_s, points_s)
        points_base = transform_points(scan.T_base_ef, points_ef)
        residual_sets.append(points_base @ n - float(plane_l))

    if not residual_sets:
        raise ValueError("no valid residuals")
    return np.concatenate(residual_sets)


def build_joint_gt_jacobian(
    *,
    scans: list[LaserScan],
    T_true: np.ndarray,
    plane_n: np.ndarray,
    plane_l: float,
    rotation_step_rad: float,
    translation_step_mm: float,
    normal_step: float,
    offset_step_mm: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Numerically differentiate the complete 9-parameter residual at GT."""

    T_true = np.asarray(T_true, dtype=float).reshape(4, 4)
    n0 = np.asarray(plane_n, dtype=float).reshape(3)
    n0 /= np.linalg.norm(n0)
    l0 = float(plane_l)
    tangent_u, tangent_v = plane_tangent_basis(n0)

    steps = np.array(
        [
            rotation_step_rad,
            rotation_step_rad,
            rotation_step_rad,
            translation_step_mm,
            translation_step_mm,
            translation_step_mm,
            normal_step,
            normal_step,
            offset_step_mm,
        ],
        dtype=float,
    )

    def evaluate(delta: np.ndarray) -> np.ndarray:
        delta = np.asarray(delta, dtype=float).reshape(9)

        T = T_true.copy()
        T[:3, :3] = T_true[:3, :3] @ Rotation.from_rotvec(
            delta[:3]
        ).as_matrix()
        T[:3, 3] = T_true[:3, 3] + delta[3:6]

        n = n0 + delta[6] * tangent_u + delta[7] * tangent_v
        n /= np.linalg.norm(n)
        l = l0 + float(delta[8])
        return residuals(scans, T, n, l)

    residual0 = evaluate(np.zeros(9, dtype=float))
    J = np.empty((residual0.size, 9), dtype=float)

    for column, step in enumerate(steps):
        plus = np.zeros(9, dtype=float)
        minus = np.zeros(9, dtype=float)
        plus[column] = step
        minus[column] = -step
        J[:, column] = (evaluate(plus) - evaluate(minus)) / (2.0 * step)

    return J, residual0


def diagnose_matrix(
    matrix: np.ndarray,
    *,
    relative_rank_tol: float,
) -> MatrixDiagnostics:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or min(matrix.shape) == 0:
        raise ValueError("matrix must be non-empty and two-dimensional")

    _, singular_values, _ = np.linalg.svd(matrix, full_matrices=False)
    threshold = relative_rank_tol * singular_values[0]
    rank = int(np.sum(singular_values > threshold))
    min_sv = float(singular_values[-1])
    condition = (
        float(singular_values[0] / min_sv)
        if min_sv > np.finfo(float).eps
        else float("inf")
    )

    column_norms = np.linalg.norm(matrix, axis=0)
    safe_norms = np.where(
        column_norms > np.finfo(float).eps,
        column_norms,
        1.0,
    )
    normalized = matrix / safe_norms
    _, normalized_singular_values, normalized_vt = np.linalg.svd(
        normalized,
        full_matrices=False,
    )
    normalized_threshold = (
        relative_rank_tol * normalized_singular_values[0]
    )
    normalized_rank = int(
        np.sum(normalized_singular_values > normalized_threshold)
    )
    normalized_min_sv = float(normalized_singular_values[-1])
    normalized_condition = (
        float(normalized_singular_values[0] / normalized_min_sv)
        if normalized_min_sv > np.finfo(float).eps
        else float("inf")
    )

    weakest = np.asarray(normalized_vt[-1], dtype=float)
    max_abs = float(np.max(np.abs(weakest)))
    if max_abs > 0.0:
        weakest /= max_abs

    return MatrixDiagnostics(
        matrix=matrix,
        rank=rank,
        singular_values=singular_values,
        condition=condition,
        min_singular_value=min_sv,
        normalized_matrix=normalized,
        normalized_rank=normalized_rank,
        normalized_singular_values=normalized_singular_values,
        normalized_condition=normalized_condition,
        normalized_min_singular_value=normalized_min_sv,
        column_norms=column_norms,
        normalized_weakest_vector=weakest,
    )


def eliminate_plane_columns(
    J_handeye: np.ndarray,
    J_plane: np.ndarray,
    *,
    relative_rank_tol: float,
) -> tuple[np.ndarray, int]:
    """Return (I - projector_on_col(J_plane)) @ J_handeye."""

    U_plane, singular_values, _ = np.linalg.svd(
        J_plane,
        full_matrices=False,
    )
    if len(singular_values) == 0 or singular_values[0] <= 0.0:
        return J_handeye.copy(), 0

    rank_plane = int(
        np.sum(singular_values > relative_rank_tol * singular_values[0])
    )
    if rank_plane == 0:
        return J_handeye.copy(), 0

    basis = U_plane[:, :rank_plane]
    J_effective = J_handeye - basis @ (basis.T @ J_handeye)
    return J_effective, rank_plane


def sensor_axis_effective_gain(
    J_effective: np.ndarray,
    T_true: np.ndarray,
    sensor_axis_index: int,
) -> float:
    """RMS residual derivative for a 1 mm shift along one true sensor axis."""

    direction_ef = np.asarray(
        T_true[:3, sensor_axis_index],
        dtype=float,
    )
    direction_ef /= np.linalg.norm(direction_ef)

    state_direction = np.zeros(6, dtype=float)
    state_direction[3:6] = direction_ef
    response = J_effective @ state_direction
    return float(np.linalg.norm(response) / np.sqrt(len(response)))


def angle_to_unoriented_axis_deg(
    vector: np.ndarray,
    axis: np.ndarray,
) -> float:
    vector = np.asarray(vector, dtype=float).reshape(3)
    axis = np.asarray(axis, dtype=float).reshape(3)
    vector_norm = float(np.linalg.norm(vector))
    axis_norm = float(np.linalg.norm(axis))
    if vector_norm <= np.finfo(float).eps or axis_norm <= np.finfo(float).eps:
        return float("nan")

    cosine = abs(float(vector @ axis)) / (vector_norm * axis_norm)
    return float(np.degrees(np.arccos(np.clip(cosine, 0.0, 1.0))))


def scan_sensor_z_coefficients(
    scans: list[LaserScan],
    T_true: np.ndarray,
    plane_n: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute c_i = n^T R_base_ef_i R_ES e_z for every scan."""

    n = np.asarray(plane_n, dtype=float).reshape(3)
    n /= np.linalg.norm(n)
    sensor_z_ef = np.asarray(T_true[:3, 2], dtype=float)

    coefficients = []
    thetas = []
    for scan in scans:
        R_base_ef = np.asarray(scan.T_base_ef[:3, :3], dtype=float)
        coefficients.append(float(n @ R_base_ef @ sensor_z_ef))
        thetas.append(float(scan.meta["assigned_theta_deg"]))

    return (
        np.asarray(coefficients, dtype=float),
        np.asarray(thetas, dtype=float),
    )


def apply_local_se3_perturbation(
    T: np.ndarray,
    delta: np.ndarray,
) -> np.ndarray:
    T = np.asarray(T, dtype=float).reshape(4, 4)
    delta = np.asarray(delta, dtype=float).reshape(6)

    out = T.copy()
    out[:3, :3] = T[:3, :3] @ Rotation.from_rotvec(
        delta[:3]
    ).as_matrix()
    out[:3, 3] = T[:3, 3] + delta[3:6]
    return out


def local_se3_difference(
    T_reference: np.ndarray,
    T_other: np.ndarray,
) -> np.ndarray:
    T_reference = np.asarray(T_reference, dtype=float).reshape(4, 4)
    T_other = np.asarray(T_other, dtype=float).reshape(4, 4)

    relative_rotation = T_reference[:3, :3].T @ T_other[:3, :3]
    dphi = Rotation.from_matrix(relative_rotation).as_rotvec()
    dt = T_other[:3, 3] - T_reference[:3, 3]
    return np.r_[dphi, dt]


def one_alternating_iteration(
    scans: list[LaserScan],
    T_input: np.ndarray,
    plane_offset_mode: str,
) -> np.ndarray:
    result = calibrate_planes(
        {0: scans},
        T_init=np.asarray(T_input, dtype=float).reshape(4, 4),
        max_iter=1,
        tol=-1.0,
        plane_offset_mode=plane_offset_mode,
    )
    return np.asarray(result.T_ef_s, dtype=float).reshape(4, 4)


def _phase_normalize_eigenvector(vector: np.ndarray) -> np.ndarray:
    """Normalize a possibly complex eigenvector with a reproducible phase."""

    vector = np.asarray(vector, dtype=complex).reshape(-1)
    if vector.size == 0:
        return vector
    pivot = int(np.argmax(np.abs(vector)))
    if abs(vector[pivot]) > 0.0:
        vector = vector * np.exp(-1j * np.angle(vector[pivot]))
    scale = float(np.max(np.abs(vector)))
    if scale > 0.0:
        vector = vector / scale
    return vector


def _unit_vector_or_nan(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float).reshape(3)
    norm = float(np.linalg.norm(vector))
    if norm <= np.finfo(float).eps:
        return np.full(3, np.nan)
    return vector / norm


def iteration_map_analysis(
    *,
    scans: list[LaserScan],
    T_true: np.ndarray,
    plane_offset_mode: str,
    rotation_eps_rad: float,
    translation_eps_mm: float,
) -> tuple[
    float,
    np.ndarray,
    np.ndarray,
    complex,
    np.ndarray,
    bool,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Differentiate the actual map F around GT and extract its slowest mode."""

    F0 = one_alternating_iteration(
        scans,
        T_true,
        plane_offset_mode,
    )
    eps = np.array(
        [rotation_eps_rad] * 3 + [translation_eps_mm] * 3,
        dtype=float,
    )
    J_map = np.empty((6, 6), dtype=float)

    for column in range(6):
        delta = np.zeros(6, dtype=float)
        delta[column] = eps[column]

        F_plus = one_alternating_iteration(
            scans,
            apply_local_se3_perturbation(T_true, delta),
            plane_offset_mode,
        )
        F_minus = one_alternating_iteration(
            scans,
            apply_local_se3_perturbation(T_true, -delta),
            plane_offset_mode,
        )

        output_plus = local_se3_difference(F0, F_plus)
        output_minus = local_se3_difference(F0, F_minus)
        J_map[:, column] = (
            output_plus - output_minus
        ) / (2.0 * eps[column])

    eigenvalues, eigenvectors = np.linalg.eig(J_map)
    slow_index = int(np.argmax(np.abs(eigenvalues)))
    dominant_eigenvalue = complex(eigenvalues[slow_index])
    slow_direction_complex = _phase_normalize_eigenvector(
        eigenvectors[:, slow_index]
    )

    real_norm = float(np.linalg.norm(np.real(slow_direction_complex)))
    imag_norm = float(np.linalg.norm(np.imag(slow_direction_complex)))
    slow_direction_is_complex = bool(
        imag_norm > 1e-7 * max(real_norm, np.finfo(float).eps)
    )

    if slow_direction_is_complex:
        # A complex conjugate pair represents a two-dimensional oscillatory mode.
        # No unique real one-dimensional translation direction exists.
        slow_direction = np.real(slow_direction_complex)
        slow_translation_ef = np.full(3, np.nan)
        slow_translation_sensor = np.full(3, np.nan)
        slow_translation_angles = np.full(3, np.nan)
    else:
        slow_direction = np.real(slow_direction_complex)
        slow_translation_ef = _unit_vector_or_nan(slow_direction[3:6])
        if np.all(np.isfinite(slow_translation_ef)):
            slow_translation_sensor = _unit_vector_or_nan(
                T_true[:3, :3].T @ slow_translation_ef
            )
            slow_translation_angles = np.array(
                [
                    angle_to_unoriented_axis_deg(
                        slow_translation_sensor,
                        np.eye(3)[axis_index],
                    )
                    for axis_index in range(3)
                ],
                dtype=float,
            )
        else:
            slow_translation_sensor = np.full(3, np.nan)
            slow_translation_angles = np.full(3, np.nan)

    spectral_radius = float(abs(dominant_eigenvalue))
    fixed_point_defect = local_se3_difference(T_true, F0)
    return (
        spectral_radius,
        eigenvalues,
        fixed_point_defect,
        dominant_eigenvalue,
        slow_direction,
        slow_direction_is_complex,
        slow_translation_ef,
        slow_translation_sensor,
        slow_translation_angles,
    )


def analyze_dataset(
    *,
    name: str,
    scans: list[LaserScan],
    T_true: np.ndarray,
    plane_n: np.ndarray,
    plane_l: float,
    relative_rank_tol: float,
    rotation_step_rad: float,
    translation_step_mm: float,
    normal_step: float,
    offset_step_mm: float,
    analyze_iteration_map: bool,
    plane_offset_mode: str,
) -> DatasetAnalysis:
    J_joint, residual0 = build_joint_gt_jacobian(
        scans=scans,
        T_true=T_true,
        plane_n=plane_n,
        plane_l=plane_l,
        rotation_step_rad=rotation_step_rad,
        translation_step_mm=translation_step_mm,
        normal_step=normal_step,
        offset_step_mm=offset_step_mm,
    )

    residual_rms = float(np.sqrt(np.mean(residual0**2)))
    print(f"\n[{name}] scans={len(scans)}, GT residual RMS={residual_rms:.6g} mm")

    J_handeye = J_joint[:, :6]
    J_plane = J_joint[:, 6:9]
    J_effective, plane_rank = eliminate_plane_columns(
        J_handeye,
        J_plane,
        relative_rank_tol=relative_rank_tol,
    )

    sensor_z_ef = np.asarray(T_true[:3, 2], dtype=float)
    j_sensor_z = J_handeye[:, 3:6] @ sensor_z_ef
    j_offset = J_plane[:, 2]
    J_focused = np.column_stack([j_sensor_z, j_offset])

    joint = diagnose_matrix(
        J_joint,
        relative_rank_tol=relative_rank_tol,
    )
    effective = diagnose_matrix(
        J_effective,
        relative_rank_tol=relative_rank_tol,
    )
    focused = diagnose_matrix(
        J_focused,
        relative_rank_tol=relative_rank_tol,
    )

    # J_normalized v = 0 means J (v / column_norms) = 0.
    effective_weakest_physical = (
        effective.normalized_weakest_vector
        / np.where(
            effective.column_norms > np.finfo(float).eps,
            effective.column_norms,
            1.0,
        )
    )
    physical_scale = float(np.max(np.abs(effective_weakest_physical)))
    if physical_scale > 0.0:
        effective_weakest_physical /= physical_scale

    weak_translation_ef = effective_weakest_physical[3:6]
    weak_translation_sensor = T_true[:3, :3].T @ weak_translation_ef
    weak_angle = angle_to_unoriented_axis_deg(
        weak_translation_ef,
        sensor_z_ef,
    )

    coefficients, scan_thetas = scan_sensor_z_coefficients(
        scans,
        T_true,
        plane_n,
    )

    if analyze_iteration_map:
        try:
            (
                rho,
                eigenvalues,
                fixed_point_defect,
                dominant_eigenvalue,
                slow_direction,
                slow_direction_is_complex,
                slow_translation_ef,
                slow_translation_sensor,
                slow_translation_angles,
            ) = iteration_map_analysis(
                scans=scans,
                T_true=T_true,
                plane_offset_mode=plane_offset_mode,
                rotation_eps_rad=rotation_step_rad,
                translation_eps_mm=translation_step_mm,
            )
            eigenvalue_magnitudes = np.sort(np.abs(eigenvalues))[::-1]
            rotation_defect = float(np.linalg.norm(fixed_point_defect[:3]))
            translation_defect = float(np.linalg.norm(fixed_point_defect[3:]))
        except Exception as exc:
            print(
                f"  iteration-map analysis failed: "
                f"{type(exc).__name__}: {exc}"
            )
            rho = float("nan")
            eigenvalue_magnitudes = np.full(6, np.nan)
            rotation_defect = float("nan")
            translation_defect = float("nan")
            dominant_eigenvalue = complex(np.nan, np.nan)
            slow_direction = np.full(6, np.nan)
            slow_direction_is_complex = False
            slow_translation_ef = np.full(3, np.nan)
            slow_translation_sensor = np.full(3, np.nan)
            slow_translation_angles = np.full(3, np.nan)
    else:
        rho = float("nan")
        eigenvalue_magnitudes = np.full(6, np.nan)
        rotation_defect = float("nan")
        translation_defect = float("nan")
        dominant_eigenvalue = complex(np.nan, np.nan)
        slow_direction = np.full(6, np.nan)
        slow_direction_is_complex = False
        slow_translation_ef = np.full(3, np.nan)
        slow_translation_sensor = np.full(3, np.nan)
        slow_translation_angles = np.full(3, np.nan)

    joint_null = joint.normalized_weakest_vector.copy()

    analysis = DatasetAnalysis(
        name=name,
        scans=scans,
        joint=joint,
        effective=effective,
        focused_tz_offset=focused,
        joint_null_vector=joint_null,
        effective_weakest_vector_physical=effective_weakest_physical,
        effective_weakest_translation_sensor=weak_translation_sensor,
        effective_weakest_translation_angle_to_sensor_z_deg=weak_angle,
        sensor_z_effective_gain=sensor_axis_effective_gain(
            J_effective,
            T_true,
            sensor_axis_index=2,
        ),
        sensor_x_effective_gain=sensor_axis_effective_gain(
            J_effective,
            T_true,
            sensor_axis_index=0,
        ),
        scan_coefficients=coefficients,
        scan_thetas_deg=scan_thetas,
        iteration_spectral_radius=rho,
        iteration_fixed_point_rotation_defect_rad=rotation_defect,
        iteration_fixed_point_translation_defect_mm=translation_defect,
        iteration_eigenvalue_magnitudes=eigenvalue_magnitudes,
        iteration_dominant_eigenvalue=dominant_eigenvalue,
        iteration_slowest_state_direction=slow_direction,
        iteration_slowest_state_direction_is_complex=(
            slow_direction_is_complex
        ),
        iteration_slowest_translation_direction_ef=slow_translation_ef,
        iteration_slowest_translation_direction_sensor=(
            slow_translation_sensor
        ),
        iteration_slowest_translation_angle_to_sensor_axes_deg=(
            slow_translation_angles
        ),
    )

    print_analysis(analysis, plane_rank)
    return analysis


def format_vector(
    labels: tuple[str, ...],
    values: np.ndarray,
) -> str:
    terms = sorted(
        zip(labels, np.asarray(values, dtype=float)),
        key=lambda item: abs(item[1]),
        reverse=True,
    )
    return ", ".join(f"{label}={value:+.4f}" for label, value in terms)


def print_analysis(
    analysis: DatasetAnalysis,
    plane_rank: int,
) -> None:
    joint = analysis.joint
    effective = analysis.effective
    focused = analysis.focused_tz_offset

    print(
        "  joint J=[J_HE J_plane]: "
        f"rank_norm={joint.normalized_rank}/9, "
        f"sigma_min_norm={joint.normalized_min_singular_value:.6g}, "
        f"cond_norm={joint.normalized_condition:.6g}"
    )
    print(
        "  effective J_eff=(I-P_plane)J_HE: "
        f"plane_rank={plane_rank}/3, "
        f"rank_norm={effective.normalized_rank}/6, "
        f"sigma_min_norm={effective.normalized_min_singular_value:.6g}, "
        f"cond_norm={effective.normalized_condition:.6g}"
    )
    print(
        "  focused J_[tzS,l]: "
        f"rank={focused.rank}/2, "
        f"sigma_min={focused.min_singular_value:.6g}, "
        f"cond={focused.condition:.6g}"
    )
    print(
        "  focused singular values: "
        + np.array2string(focused.singular_values, precision=6)
    )
    print(
        "  joint weakest normalized state: "
        + format_vector(STATE_LABELS, analysis.joint_null_vector)
    )
    print(
        "  effective weakest physical direction: "
        + format_vector(
            HAND_EYE_LABELS,
            analysis.effective_weakest_vector_physical,
        )
    )
    print(
        "  effective weakest translation in true sensor frame "
        "[x,y,z]: "
        + np.array2string(
            analysis.effective_weakest_translation_sensor,
            precision=6,
        )
    )
    print(
        "  weakest translation angle to nearest ±sensor-Z: "
        f"{analysis.effective_weakest_translation_angle_to_sensor_z_deg:.6g} deg"
    )
    print(
        "  residual gain after plane refit [mm residual / mm shift]: "
        f"sensor-Z={analysis.sensor_z_effective_gain:.6g}, "
        f"sensor-X={analysis.sensor_x_effective_gain:.6g}"
    )

    for theta in np.unique(analysis.scan_thetas_deg):
        values = analysis.scan_coefficients[
            np.isclose(analysis.scan_thetas_deg, theta)
        ]
        print(
            f"  c_i for theta={theta:g} deg: "
            f"count={len(values)}, mean={np.mean(values):+.6g}, "
            f"std={np.std(values):.3e}, "
            f"range=[{np.min(values):+.6g}, {np.max(values):+.6g}]"
        )

    if np.isfinite(analysis.iteration_spectral_radius):
        print(
            "  actual one-step map T[k+1]=F(T[k]): "
            f"spectral_radius={analysis.iteration_spectral_radius:.6g}, "
            "GT fixed-point defect="
            f"{analysis.iteration_fixed_point_rotation_defect_rad:.3e} rad, "
            f"{analysis.iteration_fixed_point_translation_defect_mm:.3e} mm"
        )
        print(
            "  iteration-map |eigenvalues|: "
            + np.array2string(
                analysis.iteration_eigenvalue_magnitudes,
                precision=6,
            )
        )
        dominant = analysis.iteration_dominant_eigenvalue
        print(
            "  slowest iteration eigenvalue: "
            f"{dominant.real:+.9g}{dominant.imag:+.9g}j"
        )
        if analysis.iteration_slowest_state_direction_is_complex:
            print(
                "  slowest mode is a complex conjugate pair; "
                "there is no unique one-dimensional real direction."
            )
            print(
                "  phase-aligned real part "
                "[dphi(rad), dt(mm) coordinates]: "
                + np.array2string(
                    analysis.iteration_slowest_state_direction,
                    precision=6,
                )
            )
        else:
            print(
                "  slowest state direction "
                "[dphi_x,dphi_y,dphi_z,dt_E_x,dt_E_y,dt_E_z]: "
                + np.array2string(
                    analysis.iteration_slowest_state_direction,
                    precision=6,
                )
            )
            print(
                "    NOTE: rotation and translation entries use different "
                "units, so compare their directions separately."
            )
            print(
                "  slowest translation direction in end-effector frame "
                "[x,y,z] (unit): "
                + np.array2string(
                    analysis.iteration_slowest_translation_direction_ef,
                    precision=6,
                )
            )
            print(
                "  slowest translation direction in sensor frame "
                "[x,y,z] (unit): "
                + np.array2string(
                    analysis.iteration_slowest_translation_direction_sensor,
                    precision=6,
                )
            )
            angles = (
                analysis
                .iteration_slowest_translation_angle_to_sensor_axes_deg
            )
            print(
                "  angle to nearest ±sensor axes [X,Y,Z] [deg]: "
                + np.array2string(angles, precision=6)
            )


def save_summary_csv(
    analyses: list[DatasetAnalysis],
    out_path: Path,
) -> Path:
    fieldnames = [
        "dataset",
        "n_scans",
        "joint_rank_norm",
        "joint_sigma_min_norm",
        "joint_condition_norm",
        "effective_rank_norm",
        "effective_sigma_min_norm",
        "effective_condition_norm",
        "focused_tz_offset_rank",
        "focused_tz_offset_sigma_min",
        "focused_tz_offset_condition",
        "sensor_z_effective_gain",
        "sensor_x_effective_gain",
        "weak_translation_angle_to_sensor_z_deg",
        "iteration_spectral_radius",
        "iteration_dominant_eigenvalue_real",
        "iteration_dominant_eigenvalue_imag",
        "iteration_slowest_direction_is_complex",
        "iteration_slow_dphi_x",
        "iteration_slow_dphi_y",
        "iteration_slow_dphi_z",
        "iteration_slow_dt_E_x",
        "iteration_slow_dt_E_y",
        "iteration_slow_dt_E_z",
        "iteration_slow_translation_sensor_x",
        "iteration_slow_translation_sensor_y",
        "iteration_slow_translation_sensor_z",
        "iteration_slow_translation_angle_sensor_x_deg",
        "iteration_slow_translation_angle_sensor_y_deg",
        "iteration_slow_translation_angle_sensor_z_deg",
        "iteration_fixed_point_rotation_defect_rad",
        "iteration_fixed_point_translation_defect_mm",
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for analysis in analyses:
            writer.writerow(
                {
                    "dataset": analysis.name,
                    "n_scans": len(analysis.scans),
                    "joint_rank_norm": analysis.joint.normalized_rank,
                    "joint_sigma_min_norm": (
                        analysis.joint.normalized_min_singular_value
                    ),
                    "joint_condition_norm": (
                        analysis.joint.normalized_condition
                    ),
                    "effective_rank_norm": (
                        analysis.effective.normalized_rank
                    ),
                    "effective_sigma_min_norm": (
                        analysis.effective.normalized_min_singular_value
                    ),
                    "effective_condition_norm": (
                        analysis.effective.normalized_condition
                    ),
                    "focused_tz_offset_rank": (
                        analysis.focused_tz_offset.rank
                    ),
                    "focused_tz_offset_sigma_min": (
                        analysis.focused_tz_offset.min_singular_value
                    ),
                    "focused_tz_offset_condition": (
                        analysis.focused_tz_offset.condition
                    ),
                    "sensor_z_effective_gain": (
                        analysis.sensor_z_effective_gain
                    ),
                    "sensor_x_effective_gain": (
                        analysis.sensor_x_effective_gain
                    ),
                    "weak_translation_angle_to_sensor_z_deg": (
                        analysis
                        .effective_weakest_translation_angle_to_sensor_z_deg
                    ),
                    "iteration_spectral_radius": (
                        analysis.iteration_spectral_radius
                    ),
                    "iteration_dominant_eigenvalue_real": (
                        analysis.iteration_dominant_eigenvalue.real
                    ),
                    "iteration_dominant_eigenvalue_imag": (
                        analysis.iteration_dominant_eigenvalue.imag
                    ),
                    "iteration_slowest_direction_is_complex": (
                        analysis.iteration_slowest_state_direction_is_complex
                    ),
                    "iteration_slow_dphi_x": (
                        analysis.iteration_slowest_state_direction[0]
                    ),
                    "iteration_slow_dphi_y": (
                        analysis.iteration_slowest_state_direction[1]
                    ),
                    "iteration_slow_dphi_z": (
                        analysis.iteration_slowest_state_direction[2]
                    ),
                    "iteration_slow_dt_E_x": (
                        analysis.iteration_slowest_state_direction[3]
                    ),
                    "iteration_slow_dt_E_y": (
                        analysis.iteration_slowest_state_direction[4]
                    ),
                    "iteration_slow_dt_E_z": (
                        analysis.iteration_slowest_state_direction[5]
                    ),
                    "iteration_slow_translation_sensor_x": (
                        analysis
                        .iteration_slowest_translation_direction_sensor[0]
                    ),
                    "iteration_slow_translation_sensor_y": (
                        analysis
                        .iteration_slowest_translation_direction_sensor[1]
                    ),
                    "iteration_slow_translation_sensor_z": (
                        analysis
                        .iteration_slowest_translation_direction_sensor[2]
                    ),
                    "iteration_slow_translation_angle_sensor_x_deg": (
                        analysis
                        .iteration_slowest_translation_angle_to_sensor_axes_deg[0]
                    ),
                    "iteration_slow_translation_angle_sensor_y_deg": (
                        analysis
                        .iteration_slowest_translation_angle_to_sensor_axes_deg[1]
                    ),
                    "iteration_slow_translation_angle_sensor_z_deg": (
                        analysis
                        .iteration_slowest_translation_angle_to_sensor_axes_deg[2]
                    ),
                    "iteration_fixed_point_rotation_defect_rad": (
                        analysis.iteration_fixed_point_rotation_defect_rad
                    ),
                    "iteration_fixed_point_translation_defect_mm": (
                        analysis.iteration_fixed_point_translation_defect_mm
                    ),
                }
            )
    return out_path


def save_comparison_plot(
    analyses: list[DatasetAnalysis],
    out_path: Path,
) -> Path:
    """Save a compact three-panel observability comparison."""

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))

    for analysis in analyses:
        axes[0].semilogy(
            np.arange(1, len(analysis.joint.normalized_singular_values) + 1),
            analysis.joint.normalized_singular_values,
            marker="o",
            label=analysis.name,
        )
    axes[0].set_xlabel("singular-value index")
    axes[0].set_ylabel("normalized singular value")
    axes[0].set_title("Joint 9-parameter Jacobian")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend()

    for analysis in analyses:
        axes[1].semilogy(
            np.arange(
                1,
                len(analysis.effective.normalized_singular_values) + 1,
            ),
            analysis.effective.normalized_singular_values,
            marker="o",
            label=analysis.name,
        )
    axes[1].set_xlabel("singular-value index")
    axes[1].set_ylabel("normalized singular value")
    axes[1].set_title(r"Plane-eliminated $J_{\mathrm{eff}}$")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend()

    for analysis in analyses:
        axes[2].semilogy(
            [1, 2],
            analysis.focused_tz_offset.singular_values,
            marker="o",
            label=analysis.name,
        )
    axes[2].set_xticks([1, 2])
    axes[2].set_xlabel("singular-value index")
    axes[2].set_ylabel("raw singular value")
    axes[2].set_title(r"Focused $[j_{t_z^S}\;\;j_l]$")
    axes[2].grid(True, which="both", alpha=0.3)
    axes[2].legend()

    fig.suptitle(
        "Single-theta vs multi-theta observability at the same ground truth"
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_coefficient_plot(
    analyses: list[DatasetAnalysis],
    out_path: Path,
) -> Path:
    fig, axes = plt.subplots(
        len(analyses),
        1,
        figsize=(10.5, 3.6 * len(analyses)),
        squeeze=False,
    )

    for row, analysis in enumerate(analyses):
        axis = axes[row, 0]
        unique_thetas = np.unique(analysis.scan_thetas_deg)
        for theta in unique_thetas:
            mask = np.isclose(analysis.scan_thetas_deg, theta)
            scan_indices = np.flatnonzero(mask)
            axis.scatter(
                scan_indices,
                analysis.scan_coefficients[mask],
                label=rf"$\theta={theta:g}^\circ$",
            )
        axis.set_xlabel("retained scan index")
        axis.set_ylabel(r"$c_i=n^\top R_iR_{ES}e_z$")
        axis.set_title(analysis.name)
        axis.grid(True, alpha=0.3)
        axis.legend()

    fig.suptitle(
        "Sensor-Z translation coefficient: one level for single theta, "
        "two levels for dual theta"
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path





def run_actual_iteration_trace(
    *,
    name: str,
    scans: list[LaserScan],
    T_true: np.ndarray,
    plane_offset_mode: str,
    slow_direction_sensor: np.ndarray,
    expected_spectral_radius: float,
    iterations: int,
    initial_error_mm: float,
) -> IterationTrace:
    """Run the actual alternating solver repeatedly and track the slow mode.

    The initial estimate uses the ground-truth rotation and a pure translation
    perturbation along the measured slow translation direction.
    """

    if iterations < 1:
        raise ValueError("trace iterations must be at least 1")
    if initial_error_mm <= 0.0:
        raise ValueError("trace initial error must be positive")

    T_true = np.asarray(T_true, dtype=float).reshape(4, 4)
    slow_direction_sensor = _unit_vector_or_nan(slow_direction_sensor)
    if not np.all(np.isfinite(slow_direction_sensor)):
        raise ValueError(
            f"{name}: a finite real slow translation direction is required"
        )

    slow_direction_ef = (
        np.asarray(T_true[:3, :3], dtype=float) @ slow_direction_sensor
    )
    slow_direction_ef = _unit_vector_or_nan(slow_direction_ef)

    T_current = T_true.copy()
    T_current[:3, 3] += float(initial_error_mm) * slow_direction_ef

    count = int(iterations) + 1
    iteration_index = np.arange(count, dtype=int)
    translation_error = np.empty(count, dtype=float)
    projection_signed = np.empty(count, dtype=float)

    for iteration in range(count):
        delta_t_ef = T_current[:3, 3] - T_true[:3, 3]
        delta_t_sensor = T_true[:3, :3].T @ delta_t_ef

        translation_error[iteration] = float(np.linalg.norm(delta_t_sensor))
        projection_signed[iteration] = float(
            slow_direction_sensor @ delta_t_sensor
        )

        if iteration < iterations:
            T_current = one_alternating_iteration(
                scans,
                T_current,
                plane_offset_mode,
            )

    projection_abs = np.abs(projection_signed)
    ratio = np.full(count, np.nan, dtype=float)
    denominator = projection_abs[:-1]
    valid = denominator > 1e-12
    ratio[1:][valid] = projection_abs[1:][valid] / denominator[valid]

    finite_ratios = ratio[np.isfinite(ratio)]
    if finite_ratios.size:
        tail_count = min(50, finite_ratios.size)
        tail_median_ratio = float(np.median(finite_ratios[-tail_count:]))
    else:
        tail_median_ratio = float("nan")

    return IterationTrace(
        name=name,
        iterations=iteration_index,
        translation_error_mm=translation_error,
        slow_projection_signed_mm=projection_signed,
        slow_projection_abs_mm=projection_abs,
        slow_projection_ratio=ratio,
        slow_direction_sensor=slow_direction_sensor,
        expected_spectral_radius=float(expected_spectral_radius),
        tail_median_ratio=tail_median_ratio,
    )


def save_iteration_trace_csv(
    traces: list[IterationTrace],
    out_path: Path,
) -> Path:
    fieldnames = [
        "dataset",
        "iteration",
        "translation_error_mm",
        "slow_projection_signed_mm",
        "slow_projection_abs_mm",
        "slow_projection_ratio",
        "expected_spectral_radius",
        "slow_direction_sensor_x",
        "slow_direction_sensor_y",
        "slow_direction_sensor_z",
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for trace in traces:
            for index, iteration in enumerate(trace.iterations):
                writer.writerow(
                    {
                        "dataset": trace.name,
                        "iteration": int(iteration),
                        "translation_error_mm": (
                            trace.translation_error_mm[index]
                        ),
                        "slow_projection_signed_mm": (
                            trace.slow_projection_signed_mm[index]
                        ),
                        "slow_projection_abs_mm": (
                            trace.slow_projection_abs_mm[index]
                        ),
                        "slow_projection_ratio": (
                            trace.slow_projection_ratio[index]
                        ),
                        "expected_spectral_radius": (
                            trace.expected_spectral_radius
                        ),
                        "slow_direction_sensor_x": (
                            trace.slow_direction_sensor[0]
                        ),
                        "slow_direction_sensor_y": (
                            trace.slow_direction_sensor[1]
                        ),
                        "slow_direction_sensor_z": (
                            trace.slow_direction_sensor[2]
                        ),
                    }
                )
    return out_path


def save_iteration_translation_trace_plot(
    traces: list[IterationTrace],
    out_path: Path,
) -> Path:
    """Plot total translation error and its slow-mode component per iteration."""

    fig, axis = plt.subplots(figsize=(10.5, 6.0))
    for trace in traces:
        axis.semilogy(
            trace.iterations,
            np.maximum(trace.translation_error_mm, 1e-15),
            label=f"{trace.name}: total translation error",
        )
        axis.semilogy(
            trace.iterations,
            np.maximum(trace.slow_projection_abs_mm, 1e-15),
            linestyle="--",
            label=f"{trace.name}: |slow-mode projection|",
        )

    axis.set_xlabel("iteration")
    axis.set_ylabel("translation error [mm]")
    axis.set_title(
        "Actual repeated alternating iterations: translation error "
        "and slow-mode projection"
    )
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_iteration_slow_ratio_plot(
    traces: list[IterationTrace],
    out_path: Path,
) -> Path:
    """Plot |q[k+1]| / |q[k]| and the local spectral-radius prediction."""

    fig, axis = plt.subplots(figsize=(10.5, 5.5))
    for trace in traces:
        axis.plot(
            trace.iterations[1:],
            trace.slow_projection_ratio[1:],
            label=(
                f"{trace.name}: measured ratio "
                f"(tail median={trace.tail_median_ratio:.6f})"
            ),
        )
        if np.isfinite(trace.expected_spectral_radius):
            axis.axhline(
                trace.expected_spectral_radius,
                linestyle="--",
                label=(
                    f"{trace.name}: predicted rho="
                    f"{trace.expected_spectral_radius:.6f}"
                ),
            )

    axis.set_xlabel("iteration k")
    axis.set_ylabel(r"$|q_k| / |q_{k-1}|$")
    axis.set_title(
        "Measured slow-mode decay ratio versus iteration-map prediction"
    )
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def print_iteration_trace_summary(traces: list[IterationTrace]) -> None:
    print("\n" + "=" * 78)
    print("ACTUAL ITERATION TRACE FROM A SLOW-DIRECTION TRANSLATION ERROR")
    print("=" * 78)
    for trace in traces:
        print(f"[{trace.name}]")
        print(
            "  slow direction in sensor frame [x,y,z] = "
            + np.array2string(trace.slow_direction_sensor, precision=6)
        )
        print(
            f"  initial translation error = "
            f"{trace.translation_error_mm[0]:.6g} mm"
        )
        print(
            f"  final translation error = "
            f"{trace.translation_error_mm[-1]:.6g} mm"
        )
        print(
            f"  initial |slow projection| = "
            f"{trace.slow_projection_abs_mm[0]:.6g} mm"
        )
        print(
            f"  final |slow projection| = "
            f"{trace.slow_projection_abs_mm[-1]:.6g} mm"
        )
        print(
            f"  predicted local rho = "
            f"{trace.expected_spectral_radius:.9g}"
        )
        print(
            f"  measured tail median |q[k+1]|/|q[k]| = "
            f"{trace.tail_median_ratio:.9g}"
        )


def save_iteration_slow_mode_plot(
    analyses: list[DatasetAnalysis],
    out_path: Path,
) -> Path:
    """Plot the slowest-mode translation direction in the true sensor frame."""

    labels = ["sensor X", "sensor Y", "sensor Z"]
    x = np.arange(3, dtype=float)
    width = 0.8 / max(len(analyses), 1)

    fig, axis = plt.subplots(figsize=(9.5, 5.2))
    for index, analysis in enumerate(analyses):
        values = np.asarray(
            analysis.iteration_slowest_translation_direction_sensor,
            dtype=float,
        )
        offset = (index - 0.5 * (len(analyses) - 1)) * width
        axis.bar(
            x + offset,
            values,
            width=width,
            label=(
                f"{analysis.name}, "
                f"rho={analysis.iteration_spectral_radius:.6f}"
            ),
        )

    axis.axhline(0.0, linewidth=1.0)
    axis.set_xticks(x, labels)
    axis.set_ylabel("unit translation-direction component")
    axis.set_title(
        "Slowest local iteration mode: translation direction "
        "in the true sensor frame"
    )
    axis.grid(True, axis="y", alpha=0.3)
    axis.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def print_final_comparison(
    single: DatasetAnalysis,
    dual: DatasetAnalysis,
) -> None:
    print("\n" + "=" * 78)
    print("FINAL COMPARISON")
    print("=" * 78)
    print(
        "focused [sensor-Z translation, plane offset] rank: "
        f"single={single.focused_tz_offset.rank}/2, "
        f"dual={dual.focused_tz_offset.rank}/2"
    )
    print(
        "focused sigma_min: "
        f"single={single.focused_tz_offset.min_singular_value:.6g}, "
        f"dual={dual.focused_tz_offset.min_singular_value:.6g}"
    )
    print(
        "joint normalized rank: "
        f"single={single.joint.normalized_rank}/9, "
        f"dual={dual.joint.normalized_rank}/9"
    )
    print(
        "effective normalized sigma_min: "
        f"single={single.effective.normalized_min_singular_value:.6g}, "
        f"dual={dual.effective.normalized_min_singular_value:.6g}"
    )
    print(
        "sensor-Z residual gain after plane refit: "
        f"single={single.sensor_z_effective_gain:.6g}, "
        f"dual={dual.sensor_z_effective_gain:.6g}"
    )
    if (
        np.isfinite(single.iteration_spectral_radius)
        and np.isfinite(dual.iteration_spectral_radius)
    ):
        print(
            "actual iteration-map spectral radius: "
            f"single={single.iteration_spectral_radius:.6g}, "
            f"dual={dual.iteration_spectral_radius:.6g}"
        )

    if (
        single.focused_tz_offset.rank < 2
        and dual.focused_tz_offset.rank == 2
    ):
        print(
            "INTERPRETATION: the dual-theta data removes the local "
            "sensor-Z/plane-offset rank deficiency."
        )
    elif dual.focused_tz_offset.min_singular_value > (
        10.0 * single.focused_tz_offset.min_singular_value
    ):
        print(
            "INTERPRETATION: both blocks are numerically full rank under the "
            "selected tolerance, but dual theta strongly improves separation."
        )
    else:
        print(
            "INTERPRETATION: the expected improvement was not clearly observed. "
            "Check theta definition, pose geometry, line assignment, and "
            "reachability filtering."
        )



def _finite_or_inf(value: float) -> float:
    return float(value) if np.isfinite(value) else float("inf")


def _estimated_half_iterations(spectral_radius: float) -> float:
    """Iterations required for the dominant local mode to halve."""

    rho = float(spectral_radius)
    if not np.isfinite(rho) or rho >= 1.0:
        return float("inf")
    if rho <= 0.0:
        return 1.0
    return float(np.log(0.5) / np.log(rho))


def _aggregate_unoriented_slow_directions(
    rows: list[dict[str, float | int | bool | str]],
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Aggregate sign-ambiguous slow translation axes across evaluations."""

    directions: list[np.ndarray] = []
    for row in rows:
        if bool(row.get("slow_direction_is_complex", False)):
            continue
        direction = np.array(
            [
                float(row["slow_translation_sensor_x"]),
                float(row["slow_translation_sensor_y"]),
                float(row["slow_translation_sensor_z"]),
            ],
            dtype=float,
        )
        if not np.all(np.isfinite(direction)):
            continue
        norm = float(np.linalg.norm(direction))
        if norm <= np.finfo(float).eps:
            continue
        directions.append(direction / norm)

    if not directions:
        return (
            0,
            np.full(3, np.nan),
            np.empty((0, 3), dtype=float),
            np.empty(0, dtype=float),
            np.empty(0, dtype=float),
        )

    direction_matrix = np.vstack(directions)
    reference = direction_matrix[0]
    aligned = direction_matrix.copy()
    aligned[(aligned @ reference) < 0.0] *= -1.0

    mean_axis = np.sum(aligned, axis=0)
    mean_norm = float(np.linalg.norm(mean_axis))
    if mean_norm <= np.finfo(float).eps:
        mean_axis = reference.copy()
    else:
        mean_axis /= mean_norm

    # The axis is unoriented; fix only its displayed sign for readability.
    if mean_axis[2] > 0.0:
        mean_axis = -mean_axis

    aligned = direction_matrix.copy()
    aligned[(aligned @ mean_axis) < 0.0] *= -1.0
    cosine_to_mean = np.clip(np.abs(aligned @ mean_axis), 0.0, 1.0)
    deviation_deg = np.degrees(np.arccos(cosine_to_mean))
    angle_to_sensor_z_deg = np.degrees(
        np.arccos(np.clip(np.abs(aligned[:, 2]), 0.0, 1.0))
    )
    return (
        len(aligned),
        mean_axis,
        aligned,
        deviation_deg,
        angle_to_sensor_z_deg,
    )


def _make_search_systems(
    *,
    seed: int,
    count: int,
    plane_tilt_min_deg: float,
    plane_tilt_max_deg: float,
    plane_yaw_min_deg: float,
    plane_yaw_max_deg: float,
    plane_center_xy_range_mm: float,
    plane_center_z_min_mm: float,
    plane_center_z_max_mm: float,
) -> list[dict[str, object]]:
    """Generate GT hand-eye/plane systems shared by every theta set."""

    systems: list[dict[str, object]] = []
    for system_idx in range(count):
        gt_seed, plane_seed, scan_seed = np.random.SeedSequence(
            [int(seed), 917_431, system_idx]
        ).spawn(3)

        T_true, _, _ = sample_random_handeye(np.random.default_rng(gt_seed))
        plane_R, plane_t, plane_n, plane_l = sample_random_plane_pose(
            rng=np.random.default_rng(plane_seed),
            tilt_min_deg=plane_tilt_min_deg,
            tilt_max_deg=plane_tilt_max_deg,
            yaw_min_deg=plane_yaw_min_deg,
            yaw_max_deg=plane_yaw_max_deg,
            center_xy_range_mm=plane_center_xy_range_mm,
            center_z_min_mm=plane_center_z_min_mm,
            center_z_max_mm=plane_center_z_max_mm,
        )
        scan_seed_value = int(
            np.random.default_rng(scan_seed).integers(
                0,
                np.iinfo(np.uint32).max,
                dtype=np.uint32,
            )
        )
        systems.append(
            {
                "system_idx": system_idx,
                "T_true": T_true,
                "plane_R": plane_R,
                "plane_t": plane_t,
                "plane_n": plane_n,
                "plane_l": plane_l,
                "scan_seed": scan_seed_value,
            }
        )
    return systems


# =============================================================================
# General theta-set search: jointly search K and theta values
# =============================================================================


@dataclass
class ThetaSetSearchResult:
    """Aggregated metrics for one theta set with K unique values."""

    theta_values_deg: tuple[float, ...]
    assignments: tuple[tuple[float, ...], ...]
    requested_systems: int
    requested_evaluations: int
    generated_evaluations: int
    full_rank_evaluations: int
    iteration_map_evaluations: int
    full_rank_rate: float
    median_spectral_radius: float
    worst_spectral_radius: float
    estimated_half_iterations: float
    median_effective_sigma_min: float
    worst_effective_sigma_min: float
    median_focused_sigma_min: float
    worst_focused_sigma_min: float
    median_sensor_z_effective_gain: float
    max_fixed_point_rotation_defect_rad: float
    max_fixed_point_translation_defect_mm: float
    worst_rho_dominant_eigenvalue_real: float
    worst_rho_dominant_eigenvalue_imag: float
    worst_rho_slow_direction_is_complex: bool
    worst_rho_slow_translation_sensor_x: float
    worst_rho_slow_translation_sensor_y: float
    worst_rho_slow_translation_sensor_z: float
    worst_rho_slow_translation_angle_sensor_x_deg: float
    worst_rho_slow_translation_angle_sensor_y_deg: float
    worst_rho_slow_translation_angle_sensor_z_deg: float
    direction_evaluations: int
    mean_slow_translation_sensor_x: float
    mean_slow_translation_sensor_y: float
    mean_slow_translation_sensor_z: float
    median_direction_deviation_deg: float
    max_direction_deviation_deg: float
    median_angle_to_sensor_z_deg: float
    max_angle_to_sensor_z_deg: float
    failure_count: int
    failure_examples: str

    @property
    def theta_count(self) -> int:
        return len(self.theta_values_deg)

    @property
    def theta_label(self) -> str:
        return "/".join(f"{value:g}" for value in self.theta_values_deg)


def _theta_set_assignments(
    theta_values_deg: tuple[float, ...],
    *,
    all_cyclic_assignments: bool,
) -> tuple[tuple[float, ...], ...]:
    """Create balanced cyclic theta assignments for the nine target lines.

    K=1 gives 9:0...; K=2 gives a 5:4 split; K=3 gives 3:3:3.
    When all_cyclic_assignments is true, every cyclic starting offset is tested,
    removing the arbitrary choice of which theta is assigned to line 0.
    """

    values = tuple(float(value) for value in theta_values_deg)
    if not values:
        raise ValueError("theta_values_deg must not be empty")
    if len(values) > 9:
        raise ValueError("at most nine unique theta values can be assigned")

    offsets = range(len(values)) if all_cyclic_assignments else range(1)
    assignments: list[tuple[float, ...]] = []
    for offset in offsets:
        assignment = tuple(
            values[(line_id + offset) % len(values)]
            for line_id in range(9)
        )
        if assignment not in assignments:
            assignments.append(assignment)
    return tuple(assignments)


def _evaluate_theta_set_on_system(
    *,
    theta_values_deg: tuple[float, ...],
    assignment: tuple[float, ...],
    assignment_idx: int,
    system: dict[str, object],
    x_values: np.ndarray,
    radius_mm: float,
    noise_std: float,
    check_reachability: bool,
    heights_mm: tuple[float, ...],
    beta_deg: tuple[float, ...],
    pose_geometry: str,
    relative_rank_tol: float,
    rotation_step_rad: float,
    translation_step_mm: float,
    normal_step: float,
    offset_step_mm: float,
    plane_offset_mode: str,
) -> dict[str, float | int | bool | str]:
    """Evaluate one K-theta set, assignment, and random GT system."""

    scans = generate_dataset(
        T_true=np.asarray(system["T_true"], dtype=float),
        plane_R=np.asarray(system["plane_R"], dtype=float),
        plane_t=np.asarray(system["plane_t"], dtype=float),
        x_values=x_values,
        radius_mm=radius_mm,
        noise_std=noise_std,
        check_reachability=check_reachability,
        heights_mm=heights_mm,
        beta_deg=beta_deg,
        theta_pool_deg=theta_values_deg,
        theta_by_line_deg=assignment,
        pose_geometry=pose_geometry,
        rng=np.random.default_rng(int(system["scan_seed"])),
    )

    T_true = np.asarray(system["T_true"], dtype=float)
    plane_n = np.asarray(system["plane_n"], dtype=float)
    plane_l = float(system["plane_l"])

    J_joint, residual0 = build_joint_gt_jacobian(
        scans=scans,
        T_true=T_true,
        plane_n=plane_n,
        plane_l=plane_l,
        rotation_step_rad=rotation_step_rad,
        translation_step_mm=translation_step_mm,
        normal_step=normal_step,
        offset_step_mm=offset_step_mm,
    )
    J_handeye = J_joint[:, :6]
    J_plane = J_joint[:, 6:9]
    J_effective, _ = eliminate_plane_columns(
        J_handeye,
        J_plane,
        relative_rank_tol=relative_rank_tol,
    )

    sensor_z_ef = np.asarray(T_true[:3, 2], dtype=float)
    J_focused = np.column_stack(
        [
            J_handeye[:, 3:6] @ sensor_z_ef,
            J_plane[:, 2],
        ]
    )

    joint = diagnose_matrix(J_joint, relative_rank_tol=relative_rank_tol)
    effective = diagnose_matrix(
        J_effective,
        relative_rank_tol=relative_rank_tol,
    )
    focused = diagnose_matrix(
        J_focused,
        relative_rank_tol=relative_rank_tol,
    )

    full_rank = bool(
        joint.normalized_rank == 9
        and effective.normalized_rank == 6
        and focused.rank == 2
    )

    rho = float("nan")
    rotation_defect = float("nan")
    translation_defect = float("nan")
    dominant_eigenvalue = complex(np.nan, np.nan)
    slow_direction = np.full(6, np.nan)
    slow_direction_is_complex = False
    slow_translation_ef = np.full(3, np.nan)
    slow_translation_sensor = np.full(3, np.nan)
    slow_translation_angles = np.full(3, np.nan)

    # The iteration map is ranked only for locally observable configurations.
    # K=1 configurations therefore remain useful as rank-deficient baselines.
    if full_rank:
        (
            rho,
            _eigenvalues,
            fixed_point_defect,
            dominant_eigenvalue,
            slow_direction,
            slow_direction_is_complex,
            slow_translation_ef,
            slow_translation_sensor,
            slow_translation_angles,
        ) = iteration_map_analysis(
            scans=scans,
            T_true=T_true,
            plane_offset_mode=plane_offset_mode,
            rotation_eps_rad=rotation_step_rad,
            translation_eps_mm=translation_step_mm,
        )
        rotation_defect = float(np.linalg.norm(fixed_point_defect[:3]))
        translation_defect = float(np.linalg.norm(fixed_point_defect[3:]))

    return {
        "theta_count": len(theta_values_deg),
        "theta_set_deg": " ".join(f"{v:g}" for v in theta_values_deg),
        "assignment_idx": int(assignment_idx),
        "line_assignment_deg": " ".join(f"{v:g}" for v in assignment),
        "system_idx": int(system["system_idx"]),
        "n_scans": len(scans),
        "gt_residual_rms_mm": float(np.sqrt(np.mean(residual0**2))),
        "joint_rank": joint.normalized_rank,
        "effective_rank": effective.normalized_rank,
        "focused_rank": focused.rank,
        "full_rank": full_rank,
        "effective_sigma_min": effective.normalized_min_singular_value,
        "focused_sigma_min": focused.min_singular_value,
        "sensor_z_effective_gain": sensor_axis_effective_gain(
            J_effective,
            T_true,
            sensor_axis_index=2,
        ),
        "spectral_radius": rho,
        "dominant_eigenvalue_real": float(np.real(dominant_eigenvalue)),
        "dominant_eigenvalue_imag": float(np.imag(dominant_eigenvalue)),
        "slow_direction_is_complex": slow_direction_is_complex,
        "slow_dphi_x": float(slow_direction[0]),
        "slow_dphi_y": float(slow_direction[1]),
        "slow_dphi_z": float(slow_direction[2]),
        "slow_dt_E_x": float(slow_direction[3]),
        "slow_dt_E_y": float(slow_direction[4]),
        "slow_dt_E_z": float(slow_direction[5]),
        "slow_translation_ef_x": float(slow_translation_ef[0]),
        "slow_translation_ef_y": float(slow_translation_ef[1]),
        "slow_translation_ef_z": float(slow_translation_ef[2]),
        "slow_translation_sensor_x": float(slow_translation_sensor[0]),
        "slow_translation_sensor_y": float(slow_translation_sensor[1]),
        "slow_translation_sensor_z": float(slow_translation_sensor[2]),
        "slow_translation_angle_sensor_x_deg": float(
            slow_translation_angles[0]
        ),
        "slow_translation_angle_sensor_y_deg": float(
            slow_translation_angles[1]
        ),
        "slow_translation_angle_sensor_z_deg": float(
            slow_translation_angles[2]
        ),
        "fixed_point_rotation_defect_rad": rotation_defect,
        "fixed_point_translation_defect_mm": translation_defect,
    }


def _aggregate_theta_set_metrics(
    *,
    theta_values_deg: tuple[float, ...],
    assignments: tuple[tuple[float, ...], ...],
    requested_systems: int,
    metrics: list[dict[str, float | int | bool | str]],
    failures: list[str],
) -> ThetaSetSearchResult:
    full_rank_metrics = [row for row in metrics if bool(row["full_rank"])]
    iteration_metrics = [
        row
        for row in full_rank_metrics
        if np.isfinite(float(row["spectral_radius"]))
    ]

    def values(
        key: str,
        rows: list[dict[str, float | int | bool | str]],
    ) -> np.ndarray:
        return np.asarray([float(row[key]) for row in rows], dtype=float)

    rho_values = values("spectral_radius", iteration_metrics)
    effective_values = values("effective_sigma_min", full_rank_metrics)
    focused_values = values("focused_sigma_min", full_rank_metrics)
    gain_values = values("sensor_z_effective_gain", full_rank_metrics)
    rotation_defects = values(
        "fixed_point_rotation_defect_rad",
        iteration_metrics,
    )
    translation_defects = values(
        "fixed_point_translation_defect_mm",
        iteration_metrics,
    )

    (
        direction_evaluations,
        mean_slow_direction,
        _aligned_slow_directions,
        direction_deviation_deg,
        angle_to_sensor_z_deg,
    ) = _aggregate_unoriented_slow_directions(iteration_metrics)

    median_rho = (
        float(np.median(rho_values)) if rho_values.size else float("nan")
    )
    worst_rho = (
        float(np.max(rho_values)) if rho_values.size else float("nan")
    )
    worst_rho_metric = (
        max(iteration_metrics, key=lambda row: float(row["spectral_radius"]))
        if iteration_metrics
        else None
    )

    def worst_metric_value(key: str) -> float:
        if worst_rho_metric is None:
            return float("nan")
        return float(worst_rho_metric[key])

    requested_evaluations = int(requested_systems * len(assignments))
    return ThetaSetSearchResult(
        theta_values_deg=theta_values_deg,
        assignments=assignments,
        requested_systems=int(requested_systems),
        requested_evaluations=requested_evaluations,
        generated_evaluations=len(metrics),
        full_rank_evaluations=len(full_rank_metrics),
        iteration_map_evaluations=len(iteration_metrics),
        full_rank_rate=(
            float(len(full_rank_metrics) / requested_evaluations)
            if requested_evaluations > 0
            else 0.0
        ),
        median_spectral_radius=median_rho,
        worst_spectral_radius=worst_rho,
        estimated_half_iterations=_estimated_half_iterations(worst_rho),
        median_effective_sigma_min=(
            float(np.median(effective_values))
            if effective_values.size
            else float("nan")
        ),
        worst_effective_sigma_min=(
            float(np.min(effective_values))
            if effective_values.size
            else float("nan")
        ),
        median_focused_sigma_min=(
            float(np.median(focused_values))
            if focused_values.size
            else float("nan")
        ),
        worst_focused_sigma_min=(
            float(np.min(focused_values))
            if focused_values.size
            else float("nan")
        ),
        median_sensor_z_effective_gain=(
            float(np.median(gain_values))
            if gain_values.size
            else float("nan")
        ),
        max_fixed_point_rotation_defect_rad=(
            float(np.max(rotation_defects))
            if rotation_defects.size
            else float("nan")
        ),
        max_fixed_point_translation_defect_mm=(
            float(np.max(translation_defects))
            if translation_defects.size
            else float("nan")
        ),
        worst_rho_dominant_eigenvalue_real=worst_metric_value(
            "dominant_eigenvalue_real"
        ),
        worst_rho_dominant_eigenvalue_imag=worst_metric_value(
            "dominant_eigenvalue_imag"
        ),
        worst_rho_slow_direction_is_complex=(
            bool(worst_rho_metric["slow_direction_is_complex"])
            if worst_rho_metric is not None
            else False
        ),
        worst_rho_slow_translation_sensor_x=worst_metric_value(
            "slow_translation_sensor_x"
        ),
        worst_rho_slow_translation_sensor_y=worst_metric_value(
            "slow_translation_sensor_y"
        ),
        worst_rho_slow_translation_sensor_z=worst_metric_value(
            "slow_translation_sensor_z"
        ),
        worst_rho_slow_translation_angle_sensor_x_deg=worst_metric_value(
            "slow_translation_angle_sensor_x_deg"
        ),
        worst_rho_slow_translation_angle_sensor_y_deg=worst_metric_value(
            "slow_translation_angle_sensor_y_deg"
        ),
        worst_rho_slow_translation_angle_sensor_z_deg=worst_metric_value(
            "slow_translation_angle_sensor_z_deg"
        ),
        direction_evaluations=direction_evaluations,
        mean_slow_translation_sensor_x=float(mean_slow_direction[0]),
        mean_slow_translation_sensor_y=float(mean_slow_direction[1]),
        mean_slow_translation_sensor_z=float(mean_slow_direction[2]),
        median_direction_deviation_deg=(
            float(np.median(direction_deviation_deg))
            if direction_deviation_deg.size
            else float("nan")
        ),
        max_direction_deviation_deg=(
            float(np.max(direction_deviation_deg))
            if direction_deviation_deg.size
            else float("nan")
        ),
        median_angle_to_sensor_z_deg=(
            float(np.median(angle_to_sensor_z_deg))
            if angle_to_sensor_z_deg.size
            else float("nan")
        ),
        max_angle_to_sensor_z_deg=(
            float(np.max(angle_to_sensor_z_deg))
            if angle_to_sensor_z_deg.size
            else float("nan")
        ),
        failure_count=len(failures),
        failure_examples=" | ".join(failures[:3]),
    )


def _theta_set_search_sort_key(
    result: ThetaSetSearchResult,
) -> tuple[float, float, float, float, int]:
    """Rank by robustness, convergence, observability, then simplicity."""

    return (
        -float(result.full_rank_rate),
        _finite_or_inf(result.worst_spectral_radius),
        _finite_or_inf(result.median_spectral_radius),
        -(
            float(result.worst_effective_sigma_min)
            if np.isfinite(result.worst_effective_sigma_min)
            else -float("inf")
        ),
        int(result.theta_count),
    )


def _format_assignments(
    assignments: tuple[tuple[float, ...], ...],
) -> str:
    return " | ".join(
        " ".join(f"{value:g}" for value in assignment)
        for assignment in assignments
    )


def save_theta_set_search_csv(
    results: list[ThetaSetSearchResult],
    out_path: Path,
) -> Path:
    fieldnames = [
        "rank",
        "theta_count",
        "theta_set_deg",
        "assignment_count",
        "line_assignments_deg",
        "requested_systems",
        "requested_evaluations",
        "generated_evaluations",
        "full_rank_evaluations",
        "iteration_map_evaluations",
        "full_rank_rate",
        "median_spectral_radius",
        "worst_spectral_radius",
        "estimated_half_iterations_from_worst_rho",
        "median_effective_sigma_min",
        "worst_effective_sigma_min",
        "median_focused_sigma_min",
        "worst_focused_sigma_min",
        "median_sensor_z_effective_gain",
        "max_fixed_point_rotation_defect_rad",
        "max_fixed_point_translation_defect_mm",
        "worst_rho_dominant_eigenvalue_real",
        "worst_rho_dominant_eigenvalue_imag",
        "worst_rho_slow_direction_is_complex",
        "worst_rho_slow_translation_sensor_x",
        "worst_rho_slow_translation_sensor_y",
        "worst_rho_slow_translation_sensor_z",
        "worst_rho_slow_translation_angle_sensor_x_deg",
        "worst_rho_slow_translation_angle_sensor_y_deg",
        "worst_rho_slow_translation_angle_sensor_z_deg",
        "direction_evaluations",
        "mean_slow_translation_sensor_x",
        "mean_slow_translation_sensor_y",
        "mean_slow_translation_sensor_z",
        "median_direction_deviation_deg",
        "max_direction_deviation_deg",
        "median_angle_to_sensor_z_deg",
        "max_angle_to_sensor_z_deg",
        "failure_count",
        "failure_examples",
    ]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for rank, result in enumerate(results, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "theta_count": result.theta_count,
                    "theta_set_deg": " ".join(
                        f"{value:g}" for value in result.theta_values_deg
                    ),
                    "assignment_count": len(result.assignments),
                    "line_assignments_deg": _format_assignments(
                        result.assignments
                    ),
                    "requested_systems": result.requested_systems,
                    "requested_evaluations": result.requested_evaluations,
                    "generated_evaluations": result.generated_evaluations,
                    "full_rank_evaluations": result.full_rank_evaluations,
                    "iteration_map_evaluations": (
                        result.iteration_map_evaluations
                    ),
                    "full_rank_rate": result.full_rank_rate,
                    "median_spectral_radius": result.median_spectral_radius,
                    "worst_spectral_radius": result.worst_spectral_radius,
                    "estimated_half_iterations_from_worst_rho": (
                        result.estimated_half_iterations
                    ),
                    "median_effective_sigma_min": (
                        result.median_effective_sigma_min
                    ),
                    "worst_effective_sigma_min": (
                        result.worst_effective_sigma_min
                    ),
                    "median_focused_sigma_min": (
                        result.median_focused_sigma_min
                    ),
                    "worst_focused_sigma_min": (
                        result.worst_focused_sigma_min
                    ),
                    "median_sensor_z_effective_gain": (
                        result.median_sensor_z_effective_gain
                    ),
                    "max_fixed_point_rotation_defect_rad": (
                        result.max_fixed_point_rotation_defect_rad
                    ),
                    "max_fixed_point_translation_defect_mm": (
                        result.max_fixed_point_translation_defect_mm
                    ),
                    "worst_rho_dominant_eigenvalue_real": (
                        result.worst_rho_dominant_eigenvalue_real
                    ),
                    "worst_rho_dominant_eigenvalue_imag": (
                        result.worst_rho_dominant_eigenvalue_imag
                    ),
                    "worst_rho_slow_direction_is_complex": (
                        result.worst_rho_slow_direction_is_complex
                    ),
                    "worst_rho_slow_translation_sensor_x": (
                        result.worst_rho_slow_translation_sensor_x
                    ),
                    "worst_rho_slow_translation_sensor_y": (
                        result.worst_rho_slow_translation_sensor_y
                    ),
                    "worst_rho_slow_translation_sensor_z": (
                        result.worst_rho_slow_translation_sensor_z
                    ),
                    "worst_rho_slow_translation_angle_sensor_x_deg": (
                        result
                        .worst_rho_slow_translation_angle_sensor_x_deg
                    ),
                    "worst_rho_slow_translation_angle_sensor_y_deg": (
                        result
                        .worst_rho_slow_translation_angle_sensor_y_deg
                    ),
                    "worst_rho_slow_translation_angle_sensor_z_deg": (
                        result
                        .worst_rho_slow_translation_angle_sensor_z_deg
                    ),
                    "direction_evaluations": result.direction_evaluations,
                    "mean_slow_translation_sensor_x": (
                        result.mean_slow_translation_sensor_x
                    ),
                    "mean_slow_translation_sensor_y": (
                        result.mean_slow_translation_sensor_y
                    ),
                    "mean_slow_translation_sensor_z": (
                        result.mean_slow_translation_sensor_z
                    ),
                    "median_direction_deviation_deg": (
                        result.median_direction_deviation_deg
                    ),
                    "max_direction_deviation_deg": (
                        result.max_direction_deviation_deg
                    ),
                    "median_angle_to_sensor_z_deg": (
                        result.median_angle_to_sensor_z_deg
                    ),
                    "max_angle_to_sensor_z_deg": (
                        result.max_angle_to_sensor_z_deg
                    ),
                    "failure_count": result.failure_count,
                    "failure_examples": result.failure_examples,
                }
            )
    return out_path


def save_theta_set_system_metrics_csv(
    rows: list[dict[str, float | int | bool | str]],
    out_path: Path,
) -> Path:
    fieldnames = [
        "theta_count",
        "theta_set_deg",
        "assignment_idx",
        "line_assignment_deg",
        "system_idx",
        "n_scans",
        "full_rank",
        "joint_rank",
        "effective_rank",
        "focused_rank",
        "spectral_radius",
        "dominant_eigenvalue_real",
        "dominant_eigenvalue_imag",
        "slow_direction_is_complex",
        "slow_translation_sensor_x",
        "slow_translation_sensor_y",
        "slow_translation_sensor_z",
        "slow_translation_angle_sensor_x_deg",
        "slow_translation_angle_sensor_y_deg",
        "slow_translation_angle_sensor_z_deg",
        "effective_sigma_min",
        "focused_sigma_min",
        "sensor_z_effective_gain",
        "gt_residual_rms_mm",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return out_path


def _best_theta_set_per_count(
    results: list[ThetaSetSearchResult],
) -> list[ThetaSetSearchResult]:
    counts = sorted({result.theta_count for result in results})
    return [
        min(
            (result for result in results if result.theta_count == count),
            key=_theta_set_search_sort_key,
        )
        for count in counts
    ]


def save_theta_count_summary_csv(
    best_by_count: list[ThetaSetSearchResult],
    out_path: Path,
) -> Path:
    fieldnames = [
        "theta_count",
        "best_theta_set_deg",
        "full_rank_rate",
        "worst_spectral_radius",
        "median_spectral_radius",
        "worst_effective_sigma_min",
        "median_effective_sigma_min",
        "estimated_half_iterations_from_worst_rho",
        "worst_rho_improvement_from_previous_count",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        previous_rho = float("nan")
        for result in best_by_count:
            improvement = (
                previous_rho - result.worst_spectral_radius
                if np.isfinite(previous_rho)
                and np.isfinite(result.worst_spectral_radius)
                else float("nan")
            )
            writer.writerow(
                {
                    "theta_count": result.theta_count,
                    "best_theta_set_deg": " ".join(
                        f"{value:g}" for value in result.theta_values_deg
                    ),
                    "full_rank_rate": result.full_rank_rate,
                    "worst_spectral_radius": result.worst_spectral_radius,
                    "median_spectral_radius": result.median_spectral_radius,
                    "worst_effective_sigma_min": (
                        result.worst_effective_sigma_min
                    ),
                    "median_effective_sigma_min": (
                        result.median_effective_sigma_min
                    ),
                    "estimated_half_iterations_from_worst_rho": (
                        result.estimated_half_iterations
                    ),
                    "worst_rho_improvement_from_previous_count": improvement,
                }
            )
            if np.isfinite(result.worst_spectral_radius):
                previous_rho = result.worst_spectral_radius
    return out_path


def save_theta_count_summary_plot(
    best_by_count: list[ThetaSetSearchResult],
    out_path: Path,
) -> Path:
    counts = np.asarray([result.theta_count for result in best_by_count])
    worst_rho = np.asarray(
        [result.worst_spectral_radius for result in best_by_count],
        dtype=float,
    )
    worst_sigma = np.asarray(
        [result.worst_effective_sigma_min for result in best_by_count],
        dtype=float,
    )

    fig, axes = plt.subplots(2, 1, figsize=(9.0, 8.0), sharex=True)
    axes[0].plot(counts, worst_rho, marker="o")
    axes[0].set_ylabel(r"worst $\rho(M)$")
    axes[0].set_title("Best theta set for each number of unique theta values")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(counts, worst_sigma, marker="o")
    axes[1].set_xlabel(r"number of unique theta values $K$")
    axes[1].set_ylabel(r"worst $\sigma_{\min}(J_{\mathrm{eff}})$")
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xticks(counts)

    for axis_index, values in enumerate((worst_rho, worst_sigma)):
        for x_value, y_value, result in zip(
            counts,
            values,
            best_by_count,
        ):
            if np.isfinite(y_value):
                axes[axis_index].annotate(
                    "{" + result.theta_label + "}",
                    (x_value, y_value),
                    textcoords="offset points",
                    xytext=(6, 5),
                    fontsize=8,
                )

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_theta_set_pair_heatmap(
    *,
    results: list[ThetaSetSearchResult],
    candidates_deg: tuple[float, ...],
    out_path: Path,
) -> Path | None:
    """Save the familiar pair heatmap for the K=2 subset, when searched."""

    pair_results = [result for result in results if result.theta_count == 2]
    if not pair_results:
        return None

    candidates = np.asarray(candidates_deg, dtype=float)
    matrix = np.full((len(candidates), len(candidates)), np.nan, dtype=float)
    index = {
        round(float(theta), 12): position
        for position, theta in enumerate(candidates)
    }
    for result in pair_results:
        if (
            result.full_rank_rate < 1.0
            or not np.isfinite(result.worst_spectral_radius)
        ):
            continue
        theta_1, theta_2 = result.theta_values_deg
        row = index[round(theta_1, 12)]
        column = index[round(theta_2, 12)]
        matrix[row, column] = result.worst_spectral_radius
        matrix[column, row] = result.worst_spectral_radius

    masked = np.ma.masked_invalid(matrix)
    fig, axis = plt.subplots(figsize=(8.5, 7.2))
    image = axis.imshow(masked, origin="lower", aspect="equal")
    axis.set_xticks(np.arange(len(candidates)), [f"{v:g}" for v in candidates])
    axis.set_yticks(np.arange(len(candidates)), [f"{v:g}" for v in candidates])
    axis.set_xlabel(r"$\theta_2$ [deg]")
    axis.set_ylabel(r"$\theta_1$ [deg]")
    axis.set_title(
        "K=2 theta-set search: worst local spectral radius\n"
        "lower is faster; blank = not full rank in every evaluation"
    )
    for row in range(len(candidates)):
        for column in range(len(candidates)):
            value = matrix[row, column]
            if np.isfinite(value):
                axis.text(
                    column,
                    row,
                    f"{value:.3f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )
    colorbar = fig.colorbar(image, ax=axis)
    colorbar.set_label(r"worst $\rho(M)$")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _best_theta_set_direction_rows(
    *,
    best: ThetaSetSearchResult,
    rows: list[dict[str, float | int | bool | str]],
) -> list[dict[str, float | int | bool | str]]:
    target = np.asarray(best.theta_values_deg, dtype=float)
    selected = []
    for row in rows:
        row_values = np.fromstring(str(row["theta_set_deg"]), sep=" ")
        if (
            row_values.shape == target.shape
            and np.allclose(row_values, target)
            and bool(row["full_rank"])
            and np.isfinite(float(row["spectral_radius"]))
        ):
            selected.append(row)
    return selected


def save_best_theta_set_direction_consistency_csv(
    *,
    best: ThetaSetSearchResult,
    rows: list[dict[str, float | int | bool | str]],
    out_path: Path,
) -> Path:
    selected = _best_theta_set_direction_rows(best=best, rows=rows)
    (
        _count,
        mean_axis,
        aligned,
        deviations,
        angles_z,
    ) = _aggregate_unoriented_slow_directions(selected)

    valid_rows = [
        row
        for row in selected
        if not bool(row["slow_direction_is_complex"])
        and np.all(
            np.isfinite(
                [
                    float(row["slow_translation_sensor_x"]),
                    float(row["slow_translation_sensor_y"]),
                    float(row["slow_translation_sensor_z"]),
                ]
            )
        )
    ]
    fieldnames = [
        "assignment_idx",
        "system_idx",
        "spectral_radius",
        "raw_direction_sensor_x",
        "raw_direction_sensor_y",
        "raw_direction_sensor_z",
        "aligned_direction_sensor_x",
        "aligned_direction_sensor_y",
        "aligned_direction_sensor_z",
        "angle_to_mean_axis_deg",
        "angle_to_sensor_z_deg",
        "mean_axis_sensor_x",
        "mean_axis_sensor_y",
        "mean_axis_sensor_z",
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for index, row in enumerate(valid_rows):
            raw = np.array(
                [
                    float(row["slow_translation_sensor_x"]),
                    float(row["slow_translation_sensor_y"]),
                    float(row["slow_translation_sensor_z"]),
                ],
                dtype=float,
            )
            writer.writerow(
                {
                    "assignment_idx": int(row["assignment_idx"]),
                    "system_idx": int(row["system_idx"]),
                    "spectral_radius": float(row["spectral_radius"]),
                    "raw_direction_sensor_x": raw[0],
                    "raw_direction_sensor_y": raw[1],
                    "raw_direction_sensor_z": raw[2],
                    "aligned_direction_sensor_x": aligned[index, 0],
                    "aligned_direction_sensor_y": aligned[index, 1],
                    "aligned_direction_sensor_z": aligned[index, 2],
                    "angle_to_mean_axis_deg": deviations[index],
                    "angle_to_sensor_z_deg": angles_z[index],
                    "mean_axis_sensor_x": mean_axis[0],
                    "mean_axis_sensor_y": mean_axis[1],
                    "mean_axis_sensor_z": mean_axis[2],
                }
            )
    return out_path


def print_theta_set_search_results(
    results: list[ThetaSetSearchResult],
    *,
    top_k: int,
) -> None:
    print("\n" + "=" * 112)
    print(
        "THETA-SET SEARCH — search K and theta values; "
        "full rank first, then minimum worst spectral radius"
    )
    print("=" * 112)
    print(
        " rank | K | theta set          | full-rank evals | worst rho | "
        "median rho | worst sigma | half iters"
    )
    print("-" * 112)
    for rank, result in enumerate(results[:top_k], start=1):
        half_text = (
            f"{result.estimated_half_iterations:.1f}"
            if np.isfinite(result.estimated_half_iterations)
            else "inf"
        )
        print(
            f"{rank:5d} | {result.theta_count:1d} | "
            f"{result.theta_label:<18s} | "
            f"{result.full_rank_evaluations:4d}/"
            f"{result.requested_evaluations:<4d}       | "
            f"{result.worst_spectral_radius:9.6f} | "
            f"{result.median_spectral_radius:10.6f} | "
            f"{result.worst_effective_sigma_min:11.6g} | "
            f"{half_text:>10s}"
        )

    best_by_count = _best_theta_set_per_count(results)
    print("\nBEST SET FOR EACH K")
    previous_rho = float("nan")
    for result in best_by_count:
        improvement = (
            previous_rho - result.worst_spectral_radius
            if np.isfinite(previous_rho)
            and np.isfinite(result.worst_spectral_radius)
            else float("nan")
        )
        improvement_text = (
            f", delta worst-rho={improvement:+.6g}"
            if np.isfinite(improvement)
            else ""
        )
        print(
            f"  K={result.theta_count}: theta={{{result.theta_label}}} deg, "
            f"full-rank={result.full_rank_evaluations}/"
            f"{result.requested_evaluations}, "
            f"worst-rho={result.worst_spectral_radius:.6g}, "
            f"worst-sigma={result.worst_effective_sigma_min:.6g}"
            f"{improvement_text}"
        )
        if np.isfinite(result.worst_spectral_radius):
            previous_rho = result.worst_spectral_radius

    fully_observable = [
        result
        for result in best_by_count
        if result.full_rank_rate == 1.0
        and np.isfinite(result.worst_spectral_radius)
    ]
    if fully_observable:
        smallest = min(fully_observable, key=lambda result: result.theta_count)
        print(
            "\nSMALLEST FULLY OBSERVABLE K"
            f"\n  K={smallest.theta_count}, "
            f"theta={{{smallest.theta_label}}} deg"
        )

    if results:
        best = results[0]
        print("\nOVERALL PERFORMANCE-RANKED CANDIDATE")
        print(
            f"  K={best.theta_count}, theta={{{best.theta_label}}} deg"
        )
        print(f"  line assignments = {_format_assignments(best.assignments)}")
        print(
            f"  full-rank rate = {best.full_rank_evaluations}/"
            f"{best.requested_evaluations}"
        )
        print(f"  worst spectral radius = {best.worst_spectral_radius:.6g}")
        print(
            "  estimated iterations to halve the slowest local mode = "
            + (
                f"{best.estimated_half_iterations:.1f}"
                if np.isfinite(best.estimated_half_iterations)
                else "not contractive or not full rank"
            )
        )


def run_theta_set_search(
    *,
    args: argparse.Namespace,
    x_values: np.ndarray,
    heights_mm: tuple[float, ...],
    beta_deg: tuple[float, ...],
    min_count: int,
    max_count: int,
) -> list[ThetaSetSearchResult]:
    """Jointly search theta count K and selected theta values."""

    raw_candidates = tuple(
        dict.fromkeys(float(value) for value in args.theta_search_candidates_deg)
    )
    candidates = tuple(sorted(raw_candidates))
    if not candidates:
        raise ValueError("--theta-search-candidates-deg must not be empty")
    if min_count < 1:
        raise ValueError("--theta-search-min-count must be at least 1")
    if max_count < min_count:
        raise ValueError(
            "--theta-search-max-count must be greater than or equal to min"
        )
    if max_count > min(len(candidates), 9):
        raise ValueError(
            "--theta-search-max-count cannot exceed the number of candidates "
            "or the nine target lines"
        )
    if args.theta_search_systems < 1:
        raise ValueError("--theta-search-systems must be at least 1")
    if args.theta_search_top_k < 1:
        raise ValueError("--theta-search-top-k must be at least 1")
    if args.noise_std != 0.0:
        print(
            "WARNING: theta-set search uses nonzero noise. "
            "Use --noise-std 0.0 for structural GT analysis."
        )

    systems = _make_search_systems(
        seed=args.seed,
        count=args.theta_search_systems,
        plane_tilt_min_deg=args.plane_tilt_min_deg,
        plane_tilt_max_deg=args.plane_tilt_max_deg,
        plane_yaw_min_deg=args.plane_yaw_min_deg,
        plane_yaw_max_deg=args.plane_yaw_max_deg,
        plane_center_xy_range_mm=args.plane_center_xy_range_mm,
        plane_center_z_min_mm=args.plane_center_z_min_mm,
        plane_center_z_max_mm=args.plane_center_z_max_mm,
    )

    theta_sets = [
        tuple(float(value) for value in theta_set)
        for count in range(min_count, max_count + 1)
        for theta_set in combinations(candidates, count)
    ]
    results: list[ThetaSetSearchResult] = []
    system_metric_rows: list[dict[str, float | int | bool | str]] = []

    total_assignment_configs = sum(
        len(
            _theta_set_assignments(
                theta_set,
                all_cyclic_assignments=(
                    args.theta_search_all_cyclic_assignments
                ),
            )
        )
        for theta_set in theta_sets
    )
    print(
        "\n[theta-set search] "
        f"sets={len(theta_sets)}, assignment-configs={total_assignment_configs}, "
        f"systems/config={len(systems)}, K={min_count}..{max_count}, "
        f"candidates={candidates}, offset_mode={args.plane_offset_mode}, "
        "all-cyclic="
        f"{args.theta_search_all_cyclic_assignments}"
    )

    for set_index, theta_values in enumerate(theta_sets, start=1):
        assignments = _theta_set_assignments(
            theta_values,
            all_cyclic_assignments=args.theta_search_all_cyclic_assignments,
        )
        metrics: list[dict[str, float | int | bool | str]] = []
        failures: list[str] = []

        for assignment_idx, assignment in enumerate(assignments):
            for system in systems:
                try:
                    metric = _evaluate_theta_set_on_system(
                        theta_values_deg=theta_values,
                        assignment=assignment,
                        assignment_idx=assignment_idx,
                        system=system,
                        x_values=x_values,
                        radius_mm=args.radius_mm,
                        noise_std=args.noise_std,
                        check_reachability=args.check_reachability,
                        heights_mm=heights_mm,
                        beta_deg=beta_deg,
                        pose_geometry=args.pose_geometry,
                        relative_rank_tol=args.relative_rank_tol,
                        rotation_step_rad=args.rotation_step_rad,
                        translation_step_mm=args.translation_step_mm,
                        normal_step=args.normal_step,
                        offset_step_mm=args.offset_step_mm,
                        plane_offset_mode=args.plane_offset_mode,
                    )
                    metrics.append(metric)
                    system_metric_rows.append(metric.copy())
                except Exception as exc:
                    failures.append(
                        f"assignment {assignment_idx}, "
                        f"system {int(system['system_idx'])}: "
                        f"{type(exc).__name__}: {exc}"
                    )

        result = _aggregate_theta_set_metrics(
            theta_values_deg=theta_values,
            assignments=assignments,
            requested_systems=len(systems),
            metrics=metrics,
            failures=failures,
        )
        results.append(result)
        print(
            f"[theta-set search] {set_index:3d}/{len(theta_sets)} | "
            f"K={result.theta_count} theta={{{result.theta_label}}} | "
            f"full-rank={result.full_rank_evaluations}/"
            f"{result.requested_evaluations} | "
            f"worst-rho={result.worst_spectral_radius:.6g} | "
            f"failures={result.failure_count}"
        )

    results.sort(key=_theta_set_search_sort_key)
    print_theta_set_search_results(
        results,
        top_k=min(args.theta_search_top_k, len(results)),
    )

    search_dir = args.output_dir / "theta_set_search"
    csv_path = save_theta_set_search_csv(
        results,
        search_dir / "theta_set_search.csv",
    )
    system_csv_path = save_theta_set_system_metrics_csv(
        system_metric_rows,
        search_dir / "theta_set_system_metrics.csv",
    )
    best_by_count = _best_theta_set_per_count(results)
    count_csv_path = save_theta_count_summary_csv(
        best_by_count,
        search_dir / "theta_count_best_summary.csv",
    )
    count_plot_path = save_theta_count_summary_plot(
        best_by_count,
        search_dir / "theta_count_best_summary.png",
    )
    pair_heatmap_path = save_theta_set_pair_heatmap(
        results=results,
        candidates_deg=candidates,
        out_path=search_dir / "theta_pair_worst_rho_heatmap.png",
    )

    best_direction_csv_path: Path | None = None
    if results and results[0].direction_evaluations > 0:
        best_direction_csv_path = (
            save_best_theta_set_direction_consistency_csv(
                best=results[0],
                rows=system_metric_rows,
                out_path=(
                    search_dir / "best_theta_set_direction_by_evaluation.csv"
                ),
            )
        )

    print(f"saved theta-set search CSV: {csv_path}")
    print(f"saved per-evaluation metrics CSV: {system_csv_path}")
    print(f"saved best-by-K summary CSV: {count_csv_path}")
    print(f"saved best-by-K summary plot: {count_plot_path}")
    if pair_heatmap_path is not None:
        print(f"saved K=2 heatmap: {pair_heatmap_path}")
    if best_direction_csv_path is not None:
        print(f"saved best-set direction CSV: {best_direction_csv_path}")
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze GT observability and actual local iteration behavior for "
            "single-theta versus multi-theta single-plane calibration."
        )
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--profile-points", type=int, default=100)
    parser.add_argument("--profile-half-width", type=float, default=25.0)
    parser.add_argument("--radius-mm", type=float, default=100.0)
    parser.add_argument(
        "--heights-mm",
        type=float,
        nargs="+",
        default=[60.0, 90.0, 120.0],
    )
    parser.add_argument(
        "--beta-deg",
        type=float,
        nargs="+",
        default=[60.0, 90.0, 120.0],
    )
    parser.add_argument("--single-theta-deg", type=float, default=30.0)
    parser.add_argument("--dual-theta-1-deg", type=float, default=30.0)
    parser.add_argument("--dual-theta-2-deg", type=float, default=60.0)
    parser.add_argument(
        "--dual-theta-3-deg",
        type=float,
        default=None,
        help=(
            "Optional third theta value. When provided, the default line "
            "assignment cycles theta1, theta2, theta3 across the nine lines."
        ),
    )
    parser.add_argument(
        "--dual-theta-by-line-deg",
        type=float,
        nargs=9,
        default=None,
        metavar=(
            "L0", "L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8"
        ),
        help=(
            "Optional explicit theta assignment for lines 0..8. "
            "Values may use theta1, theta2, and optional theta3. "
            "Without this option, available theta values are cycled across lines."
        ),
    )
    parser.add_argument(
        "--pose-geometry",
        choices=["paper_incidence", "observable_dihedral"],
        default="paper_incidence",
    )
    parser.add_argument("--noise-std", type=float, default=0.0)
    parser.add_argument("--check-reachability", action="store_true")

    parser.add_argument("--plane-tilt-min-deg", type=float, default=1.0)
    parser.add_argument("--plane-tilt-max-deg", type=float, default=5.0)
    parser.add_argument("--plane-yaw-min-deg", type=float, default=-5.0)
    parser.add_argument("--plane-yaw-max-deg", type=float, default=5.0)
    parser.add_argument("--plane-center-xy-range-mm", type=float, default=100.0)
    parser.add_argument("--plane-center-z-min-mm", type=float, default=400.0)
    parser.add_argument("--plane-center-z-max-mm", type=float, default=550.0)

    parser.add_argument("--relative-rank-tol", type=float, default=1e-8)
    parser.add_argument("--rotation-step-rad", type=float, default=1e-6)
    parser.add_argument("--translation-step-mm", type=float, default=1e-4)
    parser.add_argument("--normal-step", type=float, default=1e-6)
    parser.add_argument("--offset-step-mm", type=float, default=1e-4)

    parser.add_argument(
        "--plane-offset-mode",
        choices=["fitted", "joint"],
        default="fitted",
        help="Mode used only for the actual one-step iteration-map analysis.",
    )
    parser.add_argument(
        "--skip-iteration-map",
        action="store_true",
        help="Skip the finite-difference Jacobian of the actual solver map.",
    )

    parser.add_argument(
        "--search-theta-pairs",
        action="store_true",
        help=(
            "Backward-compatible pair-only search. Equivalent to "
            "--search-theta-sets with min-count=max-count=2."
        ),
    )
    parser.add_argument(
        "--search-theta-sets",
        action="store_true",
        help=(
            "Jointly search the number K of unique theta values and their "
            "values using --theta-search-min-count and --theta-search-max-count."
        ),
    )
    parser.add_argument(
        "--theta-search-min-count",
        type=int,
        default=1,
        help="minimum number K of unique theta values to search",
    )
    parser.add_argument(
        "--theta-search-max-count",
        type=int,
        default=3,
        help="maximum number K of unique theta values to search",
    )
    parser.add_argument(
        "--theta-search-all-cyclic-assignments",
        action="store_true",
        help=(
            "Evaluate every cyclic starting offset of each theta set across "
            "the nine lines. More robust but K times slower than one cycle."
        ),
    )
    parser.add_argument(
        "--theta-search-candidates-deg",
        type=float,
        nargs="+",
        default=[30.0, 40.0, 50.0, 60.0, 70.0, 80.0],
        help="candidate theta values used to form all unordered K-theta sets",
    )
    parser.add_argument(
        "--theta-search-systems",
        type=int,
        default=3,
        help=(
            "number of shared random GT hand-eye/plane systems used for every "
            "theta set and assignment"
        ),
    )
    parser.add_argument(
        "--theta-search-top-k",
        type=int,
        default=10,
        help="number of highest-ranked theta sets printed",
    )

    parser.add_argument(
        "--trace-iterations",
        type=int,
        default=300,
        help=(
            "number of actual alternating iterations used to validate the "
            "measured slow translation mode"
        ),
    )
    parser.add_argument(
        "--trace-initial-error-mm",
        type=float,
        default=1.0,
        help=(
            "initial pure translation error along the slow direction for the "
            "actual iteration trace"
        ),
    )
    parser.add_argument(
        "--skip-iteration-trace",
        action="store_true",
        help="skip the repeated-solver slow-mode validation trace",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("single_dual_theta_observability"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.profile_points < 2:
        raise ValueError("--profile-points must be at least 2")
    if args.trace_iterations < 1:
        raise ValueError("--trace-iterations must be at least 1")
    if args.trace_initial_error_mm <= 0.0:
        raise ValueError("--trace-initial-error-mm must be positive")
    if args.noise_std != 0.0:
        print(
            "WARNING: structural GT observability is clearest with "
            "--noise-std 0.0."
        )

    multi_theta_values = [
        float(args.dual_theta_1_deg),
        float(args.dual_theta_2_deg),
    ]
    if args.dual_theta_3_deg is not None:
        multi_theta_values.append(float(args.dual_theta_3_deg))
    # Preserve order while removing duplicates.
    multi_theta_values = list(dict.fromkeys(multi_theta_values))
    if not 1 <= len(multi_theta_values) <= 3:
        raise ValueError("the comparison supports one to three unique theta values")

    if args.dual_theta_by_line_deg is None:
        dual_assignment = tuple(
            multi_theta_values[line_id % len(multi_theta_values)]
            for line_id in range(9)
        )
    else:
        dual_assignment = tuple(
            float(value) for value in args.dual_theta_by_line_deg
        )

    allowed_dual_values = {round(value, 12) for value in multi_theta_values}
    assignment_values = {
        round(float(value), 12) for value in dual_assignment
    }
    if not assignment_values.issubset(allowed_dual_values):
        raise ValueError(
            "--dual-theta-by-line-deg may use only values supplied by "
            "--dual-theta-1-deg, --dual-theta-2-deg, and "
            "--dual-theta-3-deg"
        )

    x_values = np.linspace(
        -args.profile_half_width,
        args.profile_half_width,
        args.profile_points,
    )
    heights_mm = tuple(float(value) for value in args.heights_mm)
    beta_deg = tuple(float(value) for value in args.beta_deg)

    # Separate RNG streams make the shared GT and plane explicit.
    seed_sequence = np.random.SeedSequence(args.seed)
    gt_seed, plane_seed, single_seed, dual_seed = seed_sequence.spawn(4)

    T_true, _, _ = sample_random_handeye(
        np.random.default_rng(gt_seed)
    )
    plane_R, plane_t, plane_n, plane_l = sample_random_plane_pose(
        rng=np.random.default_rng(plane_seed),
        tilt_min_deg=args.plane_tilt_min_deg,
        tilt_max_deg=args.plane_tilt_max_deg,
        yaw_min_deg=args.plane_yaw_min_deg,
        yaw_max_deg=args.plane_yaw_max_deg,
        center_xy_range_mm=args.plane_center_xy_range_mm,
        center_z_min_mm=args.plane_center_z_min_mm,
        center_z_max_mm=args.plane_center_z_max_mm,
    )

    single_scans = generate_dataset(
        T_true=T_true,
        plane_R=plane_R,
        plane_t=plane_t,
        x_values=x_values,
        radius_mm=args.radius_mm,
        noise_std=args.noise_std,
        check_reachability=args.check_reachability,
        heights_mm=heights_mm,
        beta_deg=beta_deg,
        theta_pool_deg=(float(args.single_theta_deg),),
        theta_by_line_deg=None,
        pose_geometry=args.pose_geometry,
        rng=np.random.default_rng(single_seed),
    )

    dual_theta_pool = tuple(multi_theta_values)
    dual_scans = generate_dataset(
        T_true=T_true,
        plane_R=plane_R,
        plane_t=plane_t,
        x_values=x_values,
        radius_mm=args.radius_mm,
        noise_std=args.noise_std,
        check_reachability=args.check_reachability,
        heights_mm=heights_mm,
        beta_deg=beta_deg,
        theta_pool_deg=dual_theta_pool,
        theta_by_line_deg=dual_assignment,
        pose_geometry=args.pose_geometry,
        rng=np.random.default_rng(dual_seed),
    )

    if len(single_scans) != len(dual_scans):
        print(
            "WARNING: retained scan counts differ: "
            f"single={len(single_scans)}, dual={len(dual_scans)}. "
            "This usually means reachability filtering changed the comparison."
        )

    print("Shared ground truth:")
    print(
        "  T_true translation [mm] = "
        + np.array2string(T_true[:3, 3], precision=6)
    )
    print(
        "  plane normal = "
        + np.array2string(plane_n, precision=6)
        + f", offset={plane_l:.6g} mm"
    )
    print(
        "  single theta assignment: "
        f"all lines = {args.single_theta_deg:g} deg"
    )
    print(
        "  multi-theta assignment: "
        + ", ".join(
            f"L{line_id}={theta:g}"
            for line_id, theta in enumerate(dual_assignment)
        )
        + " deg"
    )

    common_arguments = {
        "T_true": T_true,
        "plane_n": plane_n,
        "plane_l": plane_l,
        "relative_rank_tol": args.relative_rank_tol,
        "rotation_step_rad": args.rotation_step_rad,
        "translation_step_mm": args.translation_step_mm,
        "normal_step": args.normal_step,
        "offset_step_mm": args.offset_step_mm,
        "analyze_iteration_map": not args.skip_iteration_map,
        "plane_offset_mode": args.plane_offset_mode,
    }

    single = analyze_dataset(
        name=f"single theta={args.single_theta_deg:g} deg",
        scans=single_scans,
        **common_arguments,
    )
    theta_label = "/".join(f"{value:g}" for value in multi_theta_values)
    dual = analyze_dataset(
        name=f"multi theta={theta_label} deg",
        scans=dual_scans,
        **common_arguments,
    )

    print_final_comparison(single, dual)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = save_summary_csv(
        [single, dual],
        args.output_dir / "observability_summary.csv",
    )
    comparison_path = save_comparison_plot(
        [single, dual],
        args.output_dir / "observability_singular_values.png",
    )
    coefficient_path = save_coefficient_plot(
        [single, dual],
        args.output_dir / "sensor_z_coefficients.png",
    )
    slow_mode_path = save_iteration_slow_mode_plot(
        [single, dual],
        args.output_dir / "iteration_slow_mode_sensor_translation.png",
    )

    trace_csv_path: Path | None = None
    trace_plot_path: Path | None = None
    trace_ratio_path: Path | None = None
    if not args.skip_iteration_trace:
        traces: list[IterationTrace] = []
        for analysis in (single, dual):
            if (
                not analysis.iteration_slowest_state_direction_is_complex
                and np.all(
                    np.isfinite(
                        analysis
                        .iteration_slowest_translation_direction_sensor
                    )
                )
            ):
                traces.append(
                    run_actual_iteration_trace(
                        name=analysis.name,
                        scans=analysis.scans,
                        T_true=T_true,
                        plane_offset_mode=args.plane_offset_mode,
                        slow_direction_sensor=(
                            analysis
                            .iteration_slowest_translation_direction_sensor
                        ),
                        expected_spectral_radius=(
                            analysis.iteration_spectral_radius
                        ),
                        iterations=args.trace_iterations,
                        initial_error_mm=args.trace_initial_error_mm,
                    )
                )
            else:
                print(
                    f"WARNING: skipped iteration trace for {analysis.name}: "
                    "no unique finite real slow translation direction."
                )

        if traces:
            print_iteration_trace_summary(traces)
            trace_csv_path = save_iteration_trace_csv(
                traces,
                args.output_dir / "iteration_translation_error_trace.csv",
            )
            trace_plot_path = save_iteration_translation_trace_plot(
                traces,
                args.output_dir / "iteration_translation_error_trace.png",
            )
            trace_ratio_path = save_iteration_slow_ratio_plot(
                traces,
                args.output_dir / "iteration_slow_mode_ratio.png",
            )

    print(f"\\nsaved summary: {summary_path}")
    print(f"saved singular-value plot: {comparison_path}")
    print(f"saved coefficient plot: {coefficient_path}")
    print(f"saved slow-mode direction plot: {slow_mode_path}")
    if trace_csv_path is not None:
        print(f"saved actual iteration trace CSV: {trace_csv_path}")
    if trace_plot_path is not None:
        print(f"saved actual iteration trace plot: {trace_plot_path}")
    if trace_ratio_path is not None:
        print(f"saved slow-mode ratio plot: {trace_ratio_path}")

    if args.search_theta_sets or args.search_theta_pairs:
        # Preserve the old CLI behavior: --search-theta-pairs searches only K=2.
        if args.search_theta_pairs and not args.search_theta_sets:
            search_min_count = 2
            search_max_count = 2
        else:
            search_min_count = int(args.theta_search_min_count)
            search_max_count = int(args.theta_search_max_count)

        run_theta_set_search(
            args=args,
            x_values=x_values,
            heights_mm=heights_mm,
            beta_deg=beta_deg,
            min_count=search_min_count,
            max_count=search_max_count,
        )


if __name__ == "__main__":
    main()