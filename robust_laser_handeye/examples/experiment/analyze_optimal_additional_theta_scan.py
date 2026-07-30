from __future__ import annotations

"""
Search the optimal additional theta for single-plane laser hand-eye calibration.

Acquisition design
------------------
Base dataset:
    - theta = 30 deg
    - beta = {60, 90, 120} deg
    - d = {60, 90, 120} mm
    - all 9 circular-pattern lines
    - 9 lines x 3 beta x 3 d = 81 scans

Additional dataset for one candidate theta:
    - theta = candidate
    - beta = {60, 90, 120} deg
    - d = {60, 90, 120} mm
    - exactly 4 selected circular-pattern lines
    - 4 lines x 3 beta x 3 d = 36 scans

Combined dataset:
    - 81 + 36 = 117 scans

The script evaluates each candidate using the GT local joint Jacobian
    J_joint = [J_handeye, J_plane]
and also the plane-eliminated effective hand-eye information.

Ranking rule
------------
1. Highest full-rank rate across random GT systems.
2. Largest lower-tail minimum singular value (default: joint Jacobian).
3. Largest worst-case and median minimum singular values.

The implementation accumulates J^T J blocks rather than repeatedly stacking
large Jacobians. This is mathematically equivalent and makes exhaustive search
across all C(9,4)=126 four-line subsets practical.

Example: fixed four lines
-------------------------
PYTHONPATH=. python examples/experiment/analyze_optimal_additional_theta_scan.py \
  --seed 7 \
  --systems 20 \
  --profile-points 100 \
  --heights-mm 60 90 120 \
  --beta-deg 60 90 120 \
  --base-theta-deg 30 \
  --additional-theta-candidates-deg 0 10 20 30 40 50 60 \
  --additional-lines 0 2 4 7 \
  --pose-geometry paper_incidence \
  --noise-std 0.0 \
  --score-metric joint \
  --output-dir results/additional_theta_fixed_lines

Example: search theta and the best four-line subset together
-------------------------------------------------------------
PYTHONPATH=. python examples/experiment/analyze_optimal_additional_theta_scan.py \
  --seed 7 \
  --systems 20 \
  --profile-points 100 \
  --heights-mm 60 90 120 \
  --beta-deg 60 90 120 \
  --base-theta-deg 30 \
  --additional-theta-candidates-deg 0 10 20 30 40 50 60 \
  --search-line-subsets \
  --additional-line-count 4 \
  --pose-geometry paper_incidence \
  --noise-std 0.0 \
  --score-metric joint \
  --output-dir results/additional_theta_and_lines_search
"""

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation

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


@dataclass(frozen=True)
class SearchKey:
    theta_deg: float
    line_ids: tuple[int, ...]

    @property
    def line_label(self) -> str:
        return " ".join(str(value) for value in self.line_ids)


@dataclass
class GramDiagnostics:
    rank: int
    singular_values: np.ndarray
    min_singular_value: float
    condition: float
    normalized_rank: int
    normalized_singular_values: np.ndarray
    normalized_min_singular_value: float
    normalized_condition: float
    normalized_weakest_vector: np.ndarray
    column_norms: np.ndarray


@dataclass
class CandidateAggregate:
    key: SearchKey
    requested_systems: int
    valid_systems: int
    full_rank_systems: int
    full_rank_rate: float
    joint_sigma_min_p05: float
    joint_sigma_min_worst: float
    joint_sigma_min_median: float
    effective_sigma_min_p05: float
    effective_sigma_min_worst: float
    effective_sigma_min_median: float
    focused_sigma_min_p05: float
    focused_sigma_min_worst: float
    focused_sigma_min_median: float
    sensor_z_gain_p05: float
    sensor_z_gain_worst: float
    sensor_z_gain_median: float
    median_joint_condition: float
    median_effective_condition: float
    failure_count: int
    failure_examples: str


# -----------------------------------------------------------------------------
# Geometry and scan generation
# -----------------------------------------------------------------------------


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


def generate_single_theta_all_lines(
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
    theta_deg: float,
    pose_geometry: str,
    rng: np.random.Generator,
) -> list[LaserScan]:
    scan_params = make_scan_params(
        heights_mm=heights_mm,
        theta_deg=(float(theta_deg),),
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

    for scan_id, scan in enumerate(scans):
        scan.scan_id = scan_id
        scan.meta["assigned_theta_deg"] = float(theta_deg)

    if not scans:
        raise RuntimeError("no valid scans were generated")
    return scans


def group_scans_by_line(scans: Iterable[LaserScan]) -> dict[int, list[LaserScan]]:
    grouped: dict[int, list[LaserScan]] = defaultdict(list)
    for scan in scans:
        line_id = int(scan.meta.get("line_id", -1))
        if not 0 <= line_id < 9:
            raise ValueError("scan.meta['line_id'] must be in [0, 8]")
        grouped[line_id].append(scan)
    return dict(grouped)


# -----------------------------------------------------------------------------
# GT residual Jacobian
# -----------------------------------------------------------------------------


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
    """Numerically differentiate the 9-parameter residual at the GT state."""

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

    if np.any(steps <= 0.0):
        raise ValueError("all finite-difference steps must be positive")

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


def jacobian_gram(
    *,
    scans: list[LaserScan],
    T_true: np.ndarray,
    plane_n: np.ndarray,
    plane_l: float,
    rotation_step_rad: float,
    translation_step_mm: float,
    normal_step: float,
    offset_step_mm: float,
) -> tuple[np.ndarray, float, int]:
    J, residual0 = build_joint_gt_jacobian(
        scans=scans,
        T_true=T_true,
        plane_n=plane_n,
        plane_l=plane_l,
        rotation_step_rad=rotation_step_rad,
        translation_step_mm=translation_step_mm,
        normal_step=normal_step,
        offset_step_mm=offset_step_mm,
    )
    gram = J.T @ J
    residual_sse = float(residual0 @ residual0)
    return gram, residual_sse, int(residual0.size)


def jacobian_grams_by_line(
    *,
    scans: list[LaserScan],
    T_true: np.ndarray,
    plane_n: np.ndarray,
    plane_l: float,
    rotation_step_rad: float,
    translation_step_mm: float,
    normal_step: float,
    offset_step_mm: float,
) -> tuple[dict[int, np.ndarray], dict[int, int]]:
    """Build one Jacobian and accumulate exact J_line^T J_line blocks.

    The residual vector is concatenated scan by scan, so each scan's Jacobian
    rows can be assigned back to its circular-pattern line without repeating
    finite differences nine times.
    """

    J, residual0 = build_joint_gt_jacobian(
        scans=scans,
        T_true=T_true,
        plane_n=plane_n,
        plane_l=plane_l,
        rotation_step_rad=rotation_step_rad,
        translation_step_mm=translation_step_mm,
        normal_step=normal_step,
        offset_step_mm=offset_step_mm,
    )

    line_grams = {line_id: np.zeros((9, 9), dtype=float) for line_id in range(9)}
    line_counts = {line_id: 0 for line_id in range(9)}
    row_start = 0
    for scan in scans:
        points_s = np.asarray(scan.valid_points_s, dtype=float)
        point_count = int(len(points_s)) if points_s.ndim >= 1 else 0
        if point_count == 0:
            continue
        row_stop = row_start + point_count
        if row_stop > J.shape[0]:
            raise RuntimeError("scan-to-Jacobian row accounting exceeded J rows")

        line_id = int(scan.meta.get("line_id", -1))
        if not 0 <= line_id < 9:
            raise ValueError("scan.meta['line_id'] must be in [0, 8]")
        J_block = J[row_start:row_stop]
        line_grams[line_id] += J_block.T @ J_block
        line_counts[line_id] += point_count
        row_start = row_stop

    if row_start != J.shape[0] or row_start != residual0.size:
        raise RuntimeError(
            "scan-to-Jacobian row accounting mismatch: "
            f"used={row_start}, J_rows={J.shape[0]}, residuals={residual0.size}"
        )
    return line_grams, line_counts


# -----------------------------------------------------------------------------
# Diagnostics from J^T J
# -----------------------------------------------------------------------------


def _symmetric_psd_eigh(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(matrix, dtype=float)
    matrix = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)

    # Small negative values can appear from floating-point roundoff.
    scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
    if float(np.min(eigenvalues)) < -1e-9 * scale:
        raise ValueError(
            "matrix expected to be positive semidefinite but has a "
            f"negative eigenvalue {float(np.min(eigenvalues)):.6g}"
        )
    eigenvalues = np.maximum(eigenvalues, 0.0)
    order = np.argsort(eigenvalues)[::-1]
    return eigenvalues[order], eigenvectors[:, order]


def diagnose_gram(
    gram: np.ndarray,
    *,
    relative_rank_tol: float,
) -> GramDiagnostics:
    gram = np.asarray(gram, dtype=float)
    if gram.ndim != 2 or gram.shape[0] != gram.shape[1]:
        raise ValueError("gram must be square")

    eigenvalues, eigenvectors = _symmetric_psd_eigh(gram)
    singular_values = np.sqrt(eigenvalues)
    sigma_max = float(singular_values[0]) if singular_values.size else 0.0
    threshold = relative_rank_tol * sigma_max
    rank = int(np.sum(singular_values > threshold))
    sigma_min = float(singular_values[-1])
    condition = (
        float(sigma_max / sigma_min)
        if sigma_min > np.finfo(float).eps
        else float("inf")
    )

    column_norms = np.sqrt(np.maximum(np.diag(gram), 0.0))
    safe_norms = np.where(
        column_norms > np.finfo(float).eps,
        column_norms,
        1.0,
    )
    inverse_scale = np.diag(1.0 / safe_norms)
    normalized_gram = inverse_scale @ gram @ inverse_scale
    normalized_eigenvalues, normalized_eigenvectors = _symmetric_psd_eigh(
        normalized_gram
    )
    normalized_singular_values = np.sqrt(normalized_eigenvalues)
    normalized_sigma_max = float(normalized_singular_values[0])
    normalized_threshold = relative_rank_tol * normalized_sigma_max
    normalized_rank = int(
        np.sum(normalized_singular_values > normalized_threshold)
    )
    normalized_sigma_min = float(normalized_singular_values[-1])
    normalized_condition = (
        float(normalized_sigma_max / normalized_sigma_min)
        if normalized_sigma_min > np.finfo(float).eps
        else float("inf")
    )

    weakest = np.asarray(normalized_eigenvectors[:, -1], dtype=float)
    max_abs = float(np.max(np.abs(weakest)))
    if max_abs > 0.0:
        weakest = weakest / max_abs

    return GramDiagnostics(
        rank=rank,
        singular_values=singular_values,
        min_singular_value=sigma_min,
        condition=condition,
        normalized_rank=normalized_rank,
        normalized_singular_values=normalized_singular_values,
        normalized_min_singular_value=normalized_sigma_min,
        normalized_condition=normalized_condition,
        normalized_weakest_vector=weakest,
        column_norms=column_norms,
    )


def effective_handeye_gram(joint_gram: np.ndarray) -> np.ndarray:
    """Return J_HE^T (I-P_plane) J_HE using the Schur complement."""

    G = np.asarray(joint_gram, dtype=float).reshape(9, 9)
    G_xx = G[:6, :6]
    G_xp = G[:6, 6:9]
    G_pp = G[6:9, 6:9]
    G_eff = G_xx - G_xp @ np.linalg.pinv(G_pp) @ G_xp.T
    return 0.5 * (G_eff + G_eff.T)


def focused_tz_offset_gram(
    joint_gram: np.ndarray,
    T_true: np.ndarray,
) -> np.ndarray:
    """Gram matrix of [sensor-Z translation derivative, plane-offset derivative]."""

    G = np.asarray(joint_gram, dtype=float).reshape(9, 9)
    direction_ef = np.asarray(T_true[:3, 2], dtype=float)
    direction_ef /= np.linalg.norm(direction_ef)

    state_tz = np.zeros(9, dtype=float)
    state_tz[3:6] = direction_ef
    state_l = np.zeros(9, dtype=float)
    state_l[8] = 1.0

    B = np.column_stack([state_tz, state_l])
    return B.T @ G @ B


def sensor_z_effective_gain_from_gram(
    effective_gram: np.ndarray,
    T_true: np.ndarray,
    residual_count: int,
) -> float:
    direction_ef = np.asarray(T_true[:3, 2], dtype=float)
    direction_ef /= np.linalg.norm(direction_ef)
    state = np.zeros(6, dtype=float)
    state[3:6] = direction_ef
    squared_norm = float(state @ effective_gram @ state)
    return float(np.sqrt(max(squared_norm, 0.0) / max(residual_count, 1)))


# -----------------------------------------------------------------------------
# Search system generation
# -----------------------------------------------------------------------------


def make_search_systems(args: argparse.Namespace) -> list[dict[str, object]]:
    systems: list[dict[str, object]] = []
    for system_idx in range(args.systems):
        gt_seed, plane_seed, base_scan_seed, additional_scan_seed = (
            np.random.SeedSequence(
                [int(args.seed), 1_904_731, system_idx]
            ).spawn(4)
        )

        T_true, _, _ = sample_random_handeye(np.random.default_rng(gt_seed))
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

        def seed_value(seed_sequence: np.random.SeedSequence) -> int:
            return int(
                np.random.default_rng(seed_sequence).integers(
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
                "base_scan_seed": seed_value(base_scan_seed),
                "additional_scan_seed": seed_value(additional_scan_seed),
            }
        )
    return systems


def requested_line_counts(args: argparse.Namespace) -> tuple[int, ...]:
    """Return the unique additional-line counts requested by the user."""
    if args.additional_line_counts is not None:
        return tuple(sorted(set(int(v) for v in args.additional_line_counts)))
    return (int(args.additional_line_count),)


def candidate_line_subsets(args: argparse.Namespace) -> list[tuple[int, ...]]:
    counts = requested_line_counts(args)
    if args.search_line_subsets:
        return [
            subset
            for count in counts
            for subset in combinations(range(9), count)
        ]

    if len(counts) != 1:
        raise ValueError(
            "--additional-line-counts can only be used with "
            "--search-line-subsets"
        )
    line_ids = tuple(sorted(set(int(value) for value in args.additional_lines)))
    if len(line_ids) != counts[0]:
        raise ValueError(
            "--additional-lines must contain exactly "
            f"{counts[0]} unique line IDs"
        )
    if any(not 0 <= value < 9 for value in line_ids):
        raise ValueError("additional line IDs must be in [0, 8]")
    return [line_ids]


# -----------------------------------------------------------------------------
# Per-system evaluation and aggregation
# -----------------------------------------------------------------------------


def evaluate_combined_information(
    *,
    key: SearchKey,
    base_gram: np.ndarray,
    base_residual_count: int,
    line_grams: dict[int, np.ndarray],
    line_residual_counts: dict[int, int],
    T_true: np.ndarray,
    relative_rank_tol: float,
) -> dict[str, float | int | bool | str]:
    joint_gram = np.asarray(base_gram, dtype=float).copy()
    residual_count = int(base_residual_count)
    for line_id in key.line_ids:
        joint_gram += line_grams[line_id]
        residual_count += int(line_residual_counts[line_id])

    joint = diagnose_gram(joint_gram, relative_rank_tol=relative_rank_tol)
    effective_gram = effective_handeye_gram(joint_gram)
    effective = diagnose_gram(
        effective_gram,
        relative_rank_tol=relative_rank_tol,
    )
    focused = diagnose_gram(
        focused_tz_offset_gram(joint_gram, T_true),
        relative_rank_tol=relative_rank_tol,
    )

    full_rank = bool(
        joint.normalized_rank == 9
        and effective.normalized_rank == 6
        and focused.rank == 2
    )

    return {
        "theta_deg": float(key.theta_deg),
        "additional_lines": key.line_label,
        "n_base_scans": 81,
        "n_additional_scans": 36,
        "n_total_scans": 117,
        "joint_rank": joint.normalized_rank,
        "effective_rank": effective.normalized_rank,
        "focused_rank": focused.rank,
        "full_rank": full_rank,
        "joint_sigma_min_norm": joint.normalized_min_singular_value,
        "joint_condition_norm": joint.normalized_condition,
        "effective_sigma_min_norm": effective.normalized_min_singular_value,
        "effective_condition_norm": effective.normalized_condition,
        "focused_sigma_min": focused.min_singular_value,
        "sensor_z_effective_gain": sensor_z_effective_gain_from_gram(
            effective_gram,
            T_true,
            residual_count,
        ),
        "weak_dphi_x": float(joint.normalized_weakest_vector[0]),
        "weak_dphi_y": float(joint.normalized_weakest_vector[1]),
        "weak_dphi_z": float(joint.normalized_weakest_vector[2]),
        "weak_dt_E_x": float(joint.normalized_weakest_vector[3]),
        "weak_dt_E_y": float(joint.normalized_weakest_vector[4]),
        "weak_dt_E_z": float(joint.normalized_weakest_vector[5]),
        "weak_dn_u": float(joint.normalized_weakest_vector[6]),
        "weak_dn_v": float(joint.normalized_weakest_vector[7]),
        "weak_dl": float(joint.normalized_weakest_vector[8]),
        "joint_singular_values_norm": " ".join(
            f"{value:.12g}" for value in joint.normalized_singular_values
        ),
        "effective_singular_values_norm": " ".join(
            f"{value:.12g}" for value in effective.normalized_singular_values
        ),
    }


def finite_values(rows: list[dict[str, object]], key: str) -> np.ndarray:
    values = np.asarray([float(row[key]) for row in rows], dtype=float)
    return values[np.isfinite(values)]


def lower_percentile(values: np.ndarray, percentile: float) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.percentile(values, percentile))


def aggregate_candidate(
    *,
    key: SearchKey,
    requested_systems: int,
    rows: list[dict[str, object]],
    failures: list[str],
    lower_percentile_value: float,
) -> CandidateAggregate:
    full_rank_rows = [row for row in rows if bool(row["full_rank"])]

    # Sigma statistics are computed on all valid systems. Rank-deficient cases
    # naturally contribute values near zero and therefore penalize the score.
    joint = finite_values(rows, "joint_sigma_min_norm")
    effective = finite_values(rows, "effective_sigma_min_norm")
    focused = finite_values(rows, "focused_sigma_min")
    gain = finite_values(rows, "sensor_z_effective_gain")
    joint_condition = finite_values(rows, "joint_condition_norm")
    effective_condition = finite_values(rows, "effective_condition_norm")

    def worst(values: np.ndarray) -> float:
        return float(np.min(values)) if values.size else float("nan")

    def median(values: np.ndarray) -> float:
        return float(np.median(values)) if values.size else float("nan")

    return CandidateAggregate(
        key=key,
        requested_systems=requested_systems,
        valid_systems=len(rows),
        full_rank_systems=len(full_rank_rows),
        full_rank_rate=(
            float(len(full_rank_rows) / requested_systems)
            if requested_systems > 0
            else 0.0
        ),
        joint_sigma_min_p05=lower_percentile(
            joint, lower_percentile_value
        ),
        joint_sigma_min_worst=worst(joint),
        joint_sigma_min_median=median(joint),
        effective_sigma_min_p05=lower_percentile(
            effective, lower_percentile_value
        ),
        effective_sigma_min_worst=worst(effective),
        effective_sigma_min_median=median(effective),
        focused_sigma_min_p05=lower_percentile(
            focused, lower_percentile_value
        ),
        focused_sigma_min_worst=worst(focused),
        focused_sigma_min_median=median(focused),
        sensor_z_gain_p05=lower_percentile(gain, lower_percentile_value),
        sensor_z_gain_worst=worst(gain),
        sensor_z_gain_median=median(gain),
        median_joint_condition=median(joint_condition),
        median_effective_condition=median(effective_condition),
        failure_count=len(failures),
        failure_examples=" | ".join(failures[:3]),
    )


def safe_score(value: float, *, maximize: bool) -> float:
    if not np.isfinite(value):
        return -float("inf") if maximize else float("inf")
    return float(value)


def aggregate_sort_key(
    result: CandidateAggregate,
    *,
    score_metric: str,
) -> tuple[float, float, float, float, float, tuple[int, ...]]:
    if score_metric == "joint":
        p05 = result.joint_sigma_min_p05
        worst = result.joint_sigma_min_worst
        median = result.joint_sigma_min_median
        secondary = result.effective_sigma_min_p05
    elif score_metric == "effective":
        p05 = result.effective_sigma_min_p05
        worst = result.effective_sigma_min_worst
        median = result.effective_sigma_min_median
        secondary = result.joint_sigma_min_p05
    else:
        raise ValueError(f"unknown score metric: {score_metric}")

    return (
        -float(result.full_rank_rate),
        -safe_score(p05, maximize=True),
        -safe_score(worst, maximize=True),
        -safe_score(median, maximize=True),
        -safe_score(secondary, maximize=True),
        result.key.line_ids,
    )


# -----------------------------------------------------------------------------
# Output
# -----------------------------------------------------------------------------


def save_per_system_csv(rows: list[dict[str, object]], out_path: Path) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("no per-system rows to save")
    fieldnames = list(rows[0].keys())
    with out_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return out_path


def save_summary_csv(
    results: list[CandidateAggregate],
    out_path: Path,
) -> Path:
    fieldnames = [
        "rank",
        "theta_deg",
        "additional_lines",
        "requested_systems",
        "valid_systems",
        "full_rank_systems",
        "full_rank_rate",
        "joint_sigma_min_p05",
        "joint_sigma_min_worst",
        "joint_sigma_min_median",
        "effective_sigma_min_p05",
        "effective_sigma_min_worst",
        "effective_sigma_min_median",
        "focused_sigma_min_p05",
        "focused_sigma_min_worst",
        "focused_sigma_min_median",
        "sensor_z_gain_p05",
        "sensor_z_gain_worst",
        "sensor_z_gain_median",
        "median_joint_condition",
        "median_effective_condition",
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
                    "theta_deg": result.key.theta_deg,
                    "additional_lines": result.key.line_label,
                    "requested_systems": result.requested_systems,
                    "valid_systems": result.valid_systems,
                    "full_rank_systems": result.full_rank_systems,
                    "full_rank_rate": result.full_rank_rate,
                    "joint_sigma_min_p05": result.joint_sigma_min_p05,
                    "joint_sigma_min_worst": result.joint_sigma_min_worst,
                    "joint_sigma_min_median": result.joint_sigma_min_median,
                    "effective_sigma_min_p05": (
                        result.effective_sigma_min_p05
                    ),
                    "effective_sigma_min_worst": (
                        result.effective_sigma_min_worst
                    ),
                    "effective_sigma_min_median": (
                        result.effective_sigma_min_median
                    ),
                    "focused_sigma_min_p05": result.focused_sigma_min_p05,
                    "focused_sigma_min_worst": (
                        result.focused_sigma_min_worst
                    ),
                    "focused_sigma_min_median": (
                        result.focused_sigma_min_median
                    ),
                    "sensor_z_gain_p05": result.sensor_z_gain_p05,
                    "sensor_z_gain_worst": result.sensor_z_gain_worst,
                    "sensor_z_gain_median": result.sensor_z_gain_median,
                    "median_joint_condition": result.median_joint_condition,
                    "median_effective_condition": (
                        result.median_effective_condition
                    ),
                    "failure_count": result.failure_count,
                    "failure_examples": result.failure_examples,
                }
            )
    return out_path


def best_result_per_theta(
    results: list[CandidateAggregate],
    *,
    score_metric: str,
) -> list[CandidateAggregate]:
    grouped: dict[float, list[CandidateAggregate]] = defaultdict(list)
    for result in results:
        grouped[float(result.key.theta_deg)].append(result)
    return [
        sorted(
            grouped[theta],
            key=lambda item: aggregate_sort_key(
                item, score_metric=score_metric
            ),
        )[0]
        for theta in sorted(grouped)
    ]


def save_theta_summary_plot(
    results: list[CandidateAggregate],
    out_path: Path,
    *,
    score_metric: str,
    percentile: float,
) -> Path:
    best_by_theta = best_result_per_theta(results, score_metric=score_metric)
    theta = np.asarray([row.key.theta_deg for row in best_by_theta], dtype=float)

    if score_metric == "joint":
        lower = np.asarray(
            [row.joint_sigma_min_p05 for row in best_by_theta], dtype=float
        )
        median = np.asarray(
            [row.joint_sigma_min_median for row in best_by_theta], dtype=float
        )
        ylabel = "normalized joint minimum singular value"
    else:
        lower = np.asarray(
            [row.effective_sigma_min_p05 for row in best_by_theta], dtype=float
        )
        median = np.asarray(
            [row.effective_sigma_min_median for row in best_by_theta],
            dtype=float,
        )
        ylabel = "normalized effective minimum singular value"

    fig, axis = plt.subplots(figsize=(9.5, 5.5))
    axis.plot(theta, median, marker="o", label="median across GT systems")
    axis.plot(
        theta,
        lower,
        marker="s",
        linestyle="--",
        label=f"{percentile:g}th percentile across GT systems",
    )
    axis.set_xlabel("additional theta [deg]")
    axis.set_ylabel(ylabel)
    axis.set_title(
        "Best additional-scan design for each theta"
        if len({row.key.line_ids for row in results}) > 1
        else "Additional-theta search for the selected lines"
    )
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_best_candidate_weak_vector_csv(
    per_system_rows: list[dict[str, object]],
    best: CandidateAggregate,
    out_path: Path,
) -> Path:
    matching = [
        row
        for row in per_system_rows
        if np.isclose(float(row["theta_deg"]), best.key.theta_deg)
        and str(row["additional_lines"]) == best.key.line_label
    ]
    fieldnames = [
        "system_idx",
        "joint_rank",
        "joint_sigma_min_norm",
        *[f"weak_{label}" for label in STATE_LABELS],
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in matching:
            writer.writerow(
                {
                    "system_idx": row["system_idx"],
                    "joint_rank": row["joint_rank"],
                    "joint_sigma_min_norm": row["joint_sigma_min_norm"],
                    "weak_dphi_x": row["weak_dphi_x"],
                    "weak_dphi_y": row["weak_dphi_y"],
                    "weak_dphi_z": row["weak_dphi_z"],
                    "weak_dt_E_x": row["weak_dt_E_x"],
                    "weak_dt_E_y": row["weak_dt_E_y"],
                    "weak_dt_E_z": row["weak_dt_E_z"],
                    "weak_dn_u": row["weak_dn_u"],
                    "weak_dn_v": row["weak_dn_v"],
                    "weak_dl": row["weak_dl"],
                }
            )
    return out_path


def print_results(
    results: list[CandidateAggregate],
    *,
    score_metric: str,
    top_k: int,
    percentile: float,
    beta_count: int,
    height_count: int,
) -> None:
    print("\n" + "=" * 92)
    print("OPTIMAL ADDITIONAL-SCAN SEARCH")
    print("=" * 92)
    line_counts = sorted({len(result.key.line_ids) for result in results})
    base_scans = 9 * beta_count * height_count
    print(f"Base: all 9 lines, {base_scans} scans")
    print(
        "Additional line counts searched: "
        + ", ".join(str(value) for value in line_counts)
    )
    print(
        "Additional scans by count: "
        + ", ".join(
            f"{count} lines -> {count * beta_count * height_count} scans"
            for count in line_counts
        )
    )
    print(
        f"Ranking: full-rank rate -> {percentile:g}th-percentile "
        f"{score_metric} sigma_min -> worst -> median"
    )
    print()

    for rank, result in enumerate(results[:top_k], start=1):
        metric_p05 = (
            result.joint_sigma_min_p05
            if score_metric == "joint"
            else result.effective_sigma_min_p05
        )
        metric_worst = (
            result.joint_sigma_min_worst
            if score_metric == "joint"
            else result.effective_sigma_min_worst
        )
        print(
            f"#{rank:02d} theta={result.key.theta_deg:g} deg, "
            f"lines=[{result.key.line_label}] | "
            f"full-rank={result.full_rank_systems}/"
            f"{result.requested_systems} "
            f"({result.full_rank_rate:.3f}), "
            f"{score_metric} sigma_min "
            f"p{percentile:g}={metric_p05:.6g}, "
            f"worst={metric_worst:.6g}, "
            f"joint median={result.joint_sigma_min_median:.6g}, "
            f"effective median={result.effective_sigma_min_median:.6g}"
        )

    best = results[0]
    print("\nSELECTED DESIGN")
    print(
        f"  additional theta = {best.key.theta_deg:g} deg\n"
        f"  additional line IDs = [{best.key.line_label}]\n"
        f"  additional scans = {len(best.key.line_ids)} lines x "
        f"{beta_count} beta x {height_count} d = "
        f"{len(best.key.line_ids) * beta_count * height_count}\n"
        f"  total scans = "
        f"{9 * beta_count * height_count + len(best.key.line_ids) * beta_count * height_count}"
    )


def theta_is_feasible_for_all_betas(
    theta_deg: float,
    beta_deg: tuple[float, ...],
    pose_geometry: str,
    atol: float = 1e-10,
) -> bool:
    """Check whether one theta is valid for every requested beta.

    For paper_incidence geometry, the implementation requires
        |cos(theta)| <= |sin(beta)|
    for each beta. Otherwise some beta combinations are skipped by the scan
    generator, so one line no longer contains len(d) * len(beta) scans.
    """
    if pose_geometry != "paper_incidence":
        return True
    theta_rad = np.deg2rad(float(theta_deg))
    lhs = abs(float(np.cos(theta_rad)))
    return all(
        lhs <= abs(float(np.sin(np.deg2rad(float(beta))))) + atol
        for beta in beta_deg
    )


def explain_infeasible_theta(
    theta_deg: float,
    beta_deg: tuple[float, ...],
) -> str:
    theta_rad = np.deg2rad(float(theta_deg))
    lhs = abs(float(np.cos(theta_rad)))
    invalid = [
        float(beta)
        for beta in beta_deg
        if lhs > abs(float(np.sin(np.deg2rad(float(beta))))) + 1e-10
    ]
    return (
        f"theta={theta_deg:g} deg is incompatible with beta={invalid} "
        "under paper_incidence (requires |cos(theta)| <= |sin(beta)|)."
    )


# -----------------------------------------------------------------------------
# Main search
# -----------------------------------------------------------------------------


def run_search(args: argparse.Namespace) -> tuple[
    list[CandidateAggregate],
    list[dict[str, object]],
]:
    x_values = np.linspace(
        -args.profile_half_width,
        args.profile_half_width,
        args.profile_points,
    )
    heights_mm = tuple(float(value) for value in args.heights_mm)
    beta_deg = tuple(float(value) for value in args.beta_deg)
    theta_candidates = tuple(
        dict.fromkeys(float(value) for value in args.additional_theta_candidates_deg)
    )
    line_subsets = candidate_line_subsets(args)

    feasible_candidates = []
    for theta in theta_candidates:
        if theta_is_feasible_for_all_betas(theta, beta_deg, args.pose_geometry):
            feasible_candidates.append(theta)
        else:
            print("SKIP: " + explain_infeasible_theta(theta, beta_deg))
    theta_candidates = tuple(feasible_candidates)
    if not theta_candidates:
        raise ValueError(
            "No feasible additional-theta candidates remain for the requested "
            "beta values and pose geometry."
        )

    systems = make_search_systems(args)

    expected_parameter_count = len(heights_mm) * len(beta_deg)
    if expected_parameter_count != 9:
        raise ValueError(
            "This experiment expects exactly 3 d values and 3 beta values "
            f"(9 parameter combinations), but got {expected_parameter_count}."
        )

    rows_by_key: dict[SearchKey, list[dict[str, object]]] = defaultdict(list)
    failures_by_key: dict[SearchKey, list[str]] = defaultdict(list)
    all_rows: list[dict[str, object]] = []

    common_jacobian_args = {
        "rotation_step_rad": args.rotation_step_rad,
        "translation_step_mm": args.translation_step_mm,
        "normal_step": args.normal_step,
        "offset_step_mm": args.offset_step_mm,
    }

    for system_position, system in enumerate(systems, start=1):
        system_idx = int(system["system_idx"])
        print(
            f"[system {system_position}/{len(systems)}] "
            f"building base theta={args.base_theta_deg:g} deg"
        )

        T_true = np.asarray(system["T_true"], dtype=float)
        plane_R = np.asarray(system["plane_R"], dtype=float)
        plane_t = np.asarray(system["plane_t"], dtype=float)
        plane_n = np.asarray(system["plane_n"], dtype=float)
        plane_l = float(system["plane_l"])

        base_scans = generate_single_theta_all_lines(
            T_true=T_true,
            plane_R=plane_R,
            plane_t=plane_t,
            x_values=x_values,
            radius_mm=args.radius_mm,
            noise_std=args.noise_std,
            check_reachability=args.check_reachability,
            heights_mm=heights_mm,
            beta_deg=beta_deg,
            theta_deg=args.base_theta_deg,
            pose_geometry=args.pose_geometry,
            rng=np.random.default_rng(int(system["base_scan_seed"])),
        )
        if len(base_scans) != 81:
            raise RuntimeError(
                f"base dataset must contain 81 scans, got {len(base_scans)}"
            )

        base_gram, _base_sse, base_residual_count = jacobian_gram(
            scans=base_scans,
            T_true=T_true,
            plane_n=plane_n,
            plane_l=plane_l,
            **common_jacobian_args,
        )

        for theta_position, theta_deg in enumerate(theta_candidates, start=1):
            print(
                f"  candidate {theta_position}/{len(theta_candidates)}: "
                f"theta={theta_deg:g} deg"
            )
            try:
                additional_all_lines = generate_single_theta_all_lines(
                    T_true=T_true,
                    plane_R=plane_R,
                    plane_t=plane_t,
                    x_values=x_values,
                    radius_mm=args.radius_mm,
                    noise_std=args.noise_std,
                    check_reachability=args.check_reachability,
                    heights_mm=heights_mm,
                    beta_deg=beta_deg,
                    theta_deg=theta_deg,
                    pose_geometry=args.pose_geometry,
                    # Use the same seed for every candidate theta so that
                    # stochastic profile noise is paired across candidates.
                    rng=np.random.default_rng(
                        int(system["additional_scan_seed"])
                    ),
                )
                grouped = group_scans_by_line(additional_all_lines)

                for line_id in range(9):
                    scans_for_line = grouped.get(line_id, [])
                    if len(scans_for_line) != expected_parameter_count:
                        raise RuntimeError(
                            f"theta={theta_deg:g}, line={line_id}: expected "
                            f"{expected_parameter_count} additional scans "
                            f"({len(heights_mm)} d x {len(beta_deg)} beta), "
                            f"got {len(scans_for_line)}"
                        )

                line_grams, line_residual_counts = jacobian_grams_by_line(
                    scans=additional_all_lines,
                    T_true=T_true,
                    plane_n=plane_n,
                    plane_l=plane_l,
                    **common_jacobian_args,
                )

                for line_ids in line_subsets:
                    key = SearchKey(theta_deg=theta_deg, line_ids=line_ids)
                    row = evaluate_combined_information(
                        key=key,
                        base_gram=base_gram,
                        base_residual_count=base_residual_count,
                        line_grams=line_grams,
                        line_residual_counts=line_residual_counts,
                        T_true=T_true,
                        relative_rank_tol=args.relative_rank_tol,
                    )
                    row["system_idx"] = system_idx
                    rows_by_key[key].append(row)
                    all_rows.append(row)
            except Exception as exc:
                message = (
                    f"system={system_idx}, theta={theta_deg:g}: "
                    f"{type(exc).__name__}: {exc}"
                )
                print(f"    FAILED: {message}")
                for line_ids in line_subsets:
                    failures_by_key[
                        SearchKey(theta_deg=theta_deg, line_ids=line_ids)
                    ].append(message)

    aggregates: list[CandidateAggregate] = []
    for theta_deg in theta_candidates:
        for line_ids in line_subsets:
            key = SearchKey(theta_deg=theta_deg, line_ids=line_ids)
            aggregates.append(
                aggregate_candidate(
                    key=key,
                    requested_systems=args.systems,
                    rows=rows_by_key.get(key, []),
                    failures=failures_by_key.get(key, []),
                    lower_percentile_value=args.lower_percentile,
                )
            )

    aggregates.sort(
        key=lambda result: aggregate_sort_key(
            result,
            score_metric=args.score_metric,
        )
    )
    return aggregates, all_rows



def best_result_per_line_count(
    results: list[CandidateAggregate],
    *,
    score_metric: str,
) -> list[CandidateAggregate]:
    grouped: dict[int, list[CandidateAggregate]] = defaultdict(list)
    for result in results:
        grouped[len(result.key.line_ids)].append(result)
    return [
        sorted(
            grouped[count],
            key=lambda item: aggregate_sort_key(item, score_metric=score_metric),
        )[0]
        for count in sorted(grouped)
    ]


def save_line_count_summary_csv(
    results: list[CandidateAggregate],
    out_path: Path,
    *,
    score_metric: str,
    beta_count: int,
    height_count: int,
) -> Path:
    best_rows = best_result_per_line_count(results, score_metric=score_metric)
    fieldnames = [
        "additional_line_count",
        "additional_scan_count",
        "total_scan_count",
        "best_theta_deg",
        "best_line_ids",
        "full_rank_rate",
        "joint_sigma_min_p05",
        "joint_sigma_min_worst",
        "joint_sigma_min_median",
        "effective_sigma_min_p05",
        "effective_sigma_min_worst",
        "effective_sigma_min_median",
    ]
    base_scan_count = 9 * beta_count * height_count
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in best_rows:
            count = len(row.key.line_ids)
            additional = count * beta_count * height_count
            writer.writerow({
                "additional_line_count": count,
                "additional_scan_count": additional,
                "total_scan_count": base_scan_count + additional,
                "best_theta_deg": row.key.theta_deg,
                "best_line_ids": row.key.line_label,
                "full_rank_rate": row.full_rank_rate,
                "joint_sigma_min_p05": row.joint_sigma_min_p05,
                "joint_sigma_min_worst": row.joint_sigma_min_worst,
                "joint_sigma_min_median": row.joint_sigma_min_median,
                "effective_sigma_min_p05": row.effective_sigma_min_p05,
                "effective_sigma_min_worst": row.effective_sigma_min_worst,
                "effective_sigma_min_median": row.effective_sigma_min_median,
            })
    return out_path


def save_line_count_tradeoff_plot(
    results: list[CandidateAggregate],
    out_path: Path,
    *,
    score_metric: str,
    percentile: float,
) -> Path:
    best_rows = best_result_per_line_count(results, score_metric=score_metric)
    counts = np.asarray([len(row.key.line_ids) for row in best_rows], dtype=int)
    if score_metric == "joint":
        lower = np.asarray([row.joint_sigma_min_p05 for row in best_rows])
        median = np.asarray([row.joint_sigma_min_median for row in best_rows])
        ylabel = "normalized joint minimum singular value"
    else:
        lower = np.asarray([row.effective_sigma_min_p05 for row in best_rows])
        median = np.asarray([row.effective_sigma_min_median for row in best_rows])
        ylabel = "normalized effective minimum singular value"

    fig, axis = plt.subplots(figsize=(9.5, 5.5))
    axis.plot(counts, median, marker="o", label="median of best design")
    axis.plot(
        counts,
        lower,
        marker="s",
        linestyle="--",
        label=f"{percentile:g}th percentile of best design",
    )
    for count, row, value in zip(counts, best_rows, lower):
        axis.annotate(
            rf"$\theta={row.key.theta_deg:g}^\circ$",
            (count, value),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
        )
    axis.set_xticks(counts)
    axis.set_xlabel("number of additional circular-pattern lines")
    axis.set_ylabel(ylabel)
    axis.set_title("Observability gain versus additional acquisition cost")
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out_path


def print_best_per_line_count(
    results: list[CandidateAggregate],
    *,
    score_metric: str,
    beta_count: int,
    height_count: int,
) -> None:
    print("\n" + "=" * 92)
    print("BEST DESIGN FOR EACH ADDITIONAL-LINE COUNT")
    print("=" * 92)
    base_scan_count = 9 * beta_count * height_count
    for row in best_result_per_line_count(results, score_metric=score_metric):
        count = len(row.key.line_ids)
        additional = count * beta_count * height_count
        score = (
            row.joint_sigma_min_p05
            if score_metric == "joint"
            else row.effective_sigma_min_p05
        )
        print(
            f"{count} lines ({additional} additional, "
            f"{base_scan_count + additional} total): "
            f"theta={row.key.theta_deg:g} deg, "
            f"lines=[{row.key.line_label}], "
            f"full-rank={row.full_rank_rate:.3f}, "
            f"p5 sigma_min={score:.6g}"
        )

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Search an additional theta using 81 fixed base scans and 36 "
            "additional scans from a configurable number of circular-pattern lines."
        )
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--systems", type=int, default=20)
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
    parser.add_argument("--base-theta-deg", type=float, default=30.0)
    parser.add_argument(
        "--additional-theta-candidates-deg",
        type=float,
        nargs="+",
        default=[0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
    )
    parser.add_argument(
        "--additional-line-count",
        type=int,
        default=4,
        help=(
            "single number of circular-pattern lines used for additional "
            "scans; ignored when --additional-line-counts is provided"
        ),
    )
    parser.add_argument(
        "--additional-line-counts",
        type=int,
        nargs="+",
        default=None,
        help=(
            "search several additional-line counts in one run, e.g. "
            "--additional-line-counts 1 2 3 4 5. Requires "
            "--search-line-subsets."
        ),
    )
    parser.add_argument(
        "--additional-lines",
        type=int,
        nargs="+",
        default=[0, 2, 4, 7],
        help=(
            "fixed additional line IDs. Used when --search-line-subsets is "
            "not specified. Line IDs are 0..8."
        ),
    )
    parser.add_argument(
        "--search-line-subsets",
        action="store_true",
        help=(
            "Search every subset of size --additional-line-count together "
            "with theta. For example, 3 lines evaluates C(9,3)=84 subsets "
            "per theta."
        ),
    )
    parser.add_argument(
        "--score-metric",
        choices=["joint", "effective"],
        default="joint",
        help=(
            "Primary minimum-singular-value score. 'joint' evaluates all 9 "
            "hand-eye and plane parameters; 'effective' removes plane "
            "nuisance directions first."
        ),
    )
    parser.add_argument(
        "--lower-percentile",
        type=float,
        default=5.0,
        help=(
            "lower percentile across GT systems used as the robust score; "
            "5 means the 5th percentile"
        ),
    )
    parser.add_argument("--top-k", type=int, default=20)

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
        "--output-dir",
        type=Path,
        default=Path("results/optimal_additional_theta_scan"),
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.systems < 1:
        raise ValueError("--systems must be at least 1")
    if args.profile_points < 2:
        raise ValueError("--profile-points must be at least 2")
    counts = requested_line_counts(args)
    if any(not 1 <= count <= 9 for count in counts):
        raise ValueError(
            "every additional line count must be between 1 and 9"
        )
    if args.additional_line_counts is not None and not args.search_line_subsets:
        raise ValueError(
            "--additional-line-counts requires --search-line-subsets"
        )
    if not 0.0 <= args.lower_percentile <= 100.0:
        raise ValueError("--lower-percentile must be in [0, 100]")
    if args.top_k < 1:
        raise ValueError("--top-k must be at least 1")
    if args.noise_std != 0.0:
        print(
            "WARNING: structural GT observability is clearest with "
            "--noise-std 0.0."
        )


def main() -> None:
    args = parse_args()
    validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results, per_system_rows = run_search(args)
    if not results:
        raise RuntimeError("search produced no candidate results")

    print_results(
        results,
        score_metric=args.score_metric,
        top_k=args.top_k,
        percentile=args.lower_percentile,
        beta_count=len(args.beta_deg),
        height_count=len(args.heights_mm),
    )

    print_best_per_line_count(
        results,
        score_metric=args.score_metric,
        beta_count=len(args.beta_deg),
        height_count=len(args.heights_mm),
    )

    summary_path = save_summary_csv(
        results,
        args.output_dir / "additional_theta_search_summary.csv",
    )
    per_system_path = save_per_system_csv(
        per_system_rows,
        args.output_dir / "additional_theta_search_per_system.csv",
    )
    plot_path = save_theta_summary_plot(
        results,
        args.output_dir / "additional_theta_sigma_min.png",
        score_metric=args.score_metric,
        percentile=args.lower_percentile,
    )
    weak_vector_path = save_best_candidate_weak_vector_csv(
        per_system_rows,
        results[0],
        args.output_dir / "best_candidate_weakest_vectors.csv",
    )
    line_count_summary_path = save_line_count_summary_csv(
        results,
        args.output_dir / "best_design_per_line_count.csv",
        score_metric=args.score_metric,
        beta_count=len(args.beta_deg),
        height_count=len(args.heights_mm),
    )
    line_count_plot_path = save_line_count_tradeoff_plot(
        results,
        args.output_dir / "line_count_observability_tradeoff.png",
        score_metric=args.score_metric,
        percentile=args.lower_percentile,
    )

    print("\nSaved outputs:")
    print(f"  summary CSV: {summary_path}")
    print(f"  per-system CSV: {per_system_path}")
    print(f"  theta plot: {plot_path}")
    print(f"  best-candidate weakest vectors: {weak_vector_path}")
    print(f"  best design per line count: {line_count_summary_path}")
    print(f"  line-count tradeoff plot: {line_count_plot_path}")


if __name__ == "__main__":
    main()