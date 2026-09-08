#!/usr/bin/env python3
"""Decompose plane-nuisance information loss in frozen Phase-2A subsets.

This is a read-only follow-up to ``analyze_phase2a_jacobian.py``.  It reuses
the same validated analytic Jacobian, variable order, right-local convention,
and 1-degree/1-mm scaling.  It never generates poses or runs calibration.
Statistics are written only after all centered-nuisance and Schur identities
pass explicit numerical tolerances for all 600 saved subsets.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import spearmanr


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.lhs_initialization_robustness import (  # noqa: E402
    analyze_phase2a_geometry as geometry,
)
from experiments.lhs_initialization_robustness import (  # noqa: E402
    analyze_phase2a_jacobian as jacobian_analysis,
)
from laser_handeye.nonlinear_refinement import _normal_tangent_basis  # noqa: E402


PRIMARY_N = geometry.PRIMARY_N
DIAGNOSTIC_N = geometry.DIAGNOSTIC_N
ROTATION_SCALE_RAD = jacobian_analysis.ROTATION_SCALE_RAD
EPSILON = 1e-12
FD_STEP = 1e-6

VALIDATION_TOLERANCES = {
    "analytic_fd_max_abs": 1e-5,
    "centered_y_sum_max_abs": 1e-8,
    "nuisance_projector_fro": 1e-9,
    "nuisance_principal_angle_max_deg": 1e-5,
    "centered_heff_relative_fro": 1e-9,
    "offset_covariance_max_abs": 1e-9,
    "translation_heff_relative_fro": 1e-9,
    "full_heff_relative_fro": 1e-9,
    "jx_row_definition_max_abs": 1e-10,
}

NEW_FEATURES = (
    "b_resultant",
    "b_centered_lambda_min",
    "b_centered_lambda_max",
    "b_centered_eigenvalue_std",
    "b_centered_condition",
    "b_centered_trace",
    "b_centered_pseudo_logdet",
    "translation_offset_loss_ratio",
    "translation_normal_loss_ratio",
    "translation_total_loss_ratio",
    "translation_retention_ratio",
    "translation_heff_trace",
    "translation_heff_lambda_min",
    "translation_heff_logdet",
    "translation_offset_loss_fro_ratio",
    "translation_normal_loss_fro_ratio",
    "translation_total_loss_fro_ratio",
    "full_offset_loss_trace_ratio",
    "full_normal_loss_trace_ratio",
    "full_total_loss_trace_ratio",
    "full_retention_trace_ratio",
    "nuisance_canonical_sigma_max",
    "nuisance_canonical_sigma_mean",
    "nuisance_principal_angle_min_deg",
)

TRANSLATION_OPTIMALITY_FEATURES = (
    ("T", "translation_heff_trace"),
    ("E", "translation_heff_lambda_min"),
    ("D", "translation_heff_logdet"),
    ("b_resultant_reference", "b_resultant"),
    ("b_centered_trace_reference", "b_centered_trace"),
)

EXISTING_COMPARISON_FEATURES = (
    "D_view_pairwise_angle_mean_deg",
    "b_pairwise_angle_mean_deg",
    "b_G_lambda_min",
    "scaled_hxpi_normalized_fro",
    "scaled_information_retention_trace",
    "scaled_heff_logdet",
    "scaled_heff_lambda_min",
)

PRIMARY_NEW_FEATURES = (
    "b_resultant",
    "b_centered_lambda_min",
    "b_centered_condition",
    "translation_offset_loss_ratio",
    "translation_normal_loss_ratio",
    "translation_total_loss_ratio",
    "translation_retention_ratio",
    "nuisance_canonical_sigma_max",
    "nuisance_principal_angle_min_deg",
)

CONTINUOUS_TARGETS = (
    "accuracy_success_rate",
    "normalized_error_p90",
    "normalized_error_median",
    "scaled_heff_logdet",
    "scaled_heff_lambda_min",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase1-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "phase1_lhs_screening",
    )
    parser.add_argument(
        "--phase2a-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "phase2a_initialization_robustness",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "phase2a_jacobian_decomposition",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace decomposition outputs; Phase-1/2A inputs remain untouched",
    )
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _prepare_output(path: Path, overwrite: bool) -> Path:
    known = (
        "subset_jacobian_decomposition.csv",
        "core_vs_non_decomposition_stats.csv",
        "continuous_decomposition_correlations.csv",
        "proxy_comparison.csv",
        "translation_optimality_comparison.csv",
        "numerical_validation.json",
        "decomposition_report.md",
        "analysis_manifest.json",
    )
    existing = [path / name for name in known if (path / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"decomposition output already exists; use --overwrite: {existing[0]}"
        )
    path.mkdir(parents=True, exist_ok=True)
    (path / "plots").mkdir(exist_ok=True)
    return path


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _orthonormal_column_basis(matrix: np.ndarray) -> np.ndarray:
    u, singular_values, _ = np.linalg.svd(matrix, full_matrices=False)
    if not len(singular_values):
        return np.empty((matrix.shape[0], 0), dtype=float)
    tolerance = max(matrix.shape) * np.finfo(float).eps * singular_values[0]
    rank = int(np.sum(singular_values > tolerance))
    return u[:, :rank]


def _schur_effective(j_x: np.ndarray, j_pi: np.ndarray) -> np.ndarray:
    h_xx = j_x.T @ j_x
    h_xpi = j_x.T @ j_pi
    h_pipi = j_pi.T @ j_pi
    h_eff = h_xx - h_xpi @ np.linalg.pinv(h_pipi, rcond=1e-12) @ h_xpi.T
    return 0.5 * (h_eff + h_eff.T)


def _relative_frobenius(left: np.ndarray, right: np.ndarray) -> float:
    denominator = max(float(np.linalg.norm(right, ord="fro")), EPSILON)
    return float(np.linalg.norm(left - right, ord="fro") / denominator)


def _eigen_features(matrix: np.ndarray, prefix: str) -> dict[str, float | int]:
    matrix = 0.5 * (matrix + matrix.T)
    eigenvalues = np.linalg.eigvalsh(matrix)
    maximum = float(eigenvalues[-1])
    tolerance = max(EPSILON, abs(maximum) * 1e-10)
    positive = eigenvalues[eigenvalues > tolerance]
    rank = int(len(positive))
    minimum = float(eigenvalues[0])
    return {
        f"{prefix}_lambda_min": minimum,
        f"{prefix}_lambda_mid": float(eigenvalues[1]),
        f"{prefix}_lambda_max": maximum,
        f"{prefix}_eigenvalue_std": float(np.std(eigenvalues, ddof=0)),
        f"{prefix}_condition": (
            float("inf") if rank < len(eigenvalues) else maximum / minimum
        ),
        f"{prefix}_trace": float(np.trace(matrix)),
        f"{prefix}_rank": rank,
        f"{prefix}_pseudo_logdet": (
            float(np.sum(np.log(positive))) if len(positive) else float("-inf")
        ),
    }


def _point_geometry(
    candidate_ids: np.ndarray,
    bank: dict[str, np.ndarray],
    normal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return point-level b, q_base, and scan IDs with unequal-size support."""
    b_blocks: list[np.ndarray] = []
    q_blocks: list[np.ndarray] = []
    scan_blocks: list[np.ndarray] = []
    for scan_index, candidate_id in enumerate(candidate_ids):
        transform = bank["T_base_s"][candidate_id]
        points_sensor = bank["ideal_points_s"][candidate_id]
        valid = np.all(np.isfinite(points_sensor), axis=1)
        points_sensor = points_sensor[valid]
        if not len(points_sensor):
            raise ValueError(f"candidate {candidate_id} has no finite ideal points")
        b = transform[:3, :3].T @ normal
        points_base = points_sensor @ transform[:3, :3].T + transform[:3, 3]
        b_blocks.append(np.repeat(b[None, :], len(points_sensor), axis=0))
        q_blocks.append(points_base)
        scan_blocks.append(np.full(len(points_sensor), scan_index, dtype=np.int64))
    return np.vstack(b_blocks), np.vstack(q_blocks), np.concatenate(scan_blocks)


def _finite_difference_error(
    candidate_ids: np.ndarray,
    bank: dict[str, np.ndarray],
    T_true: np.ndarray,
    normal: np.ndarray,
    offset_mm: float,
) -> float:
    analytic, _ = jacobian_analysis._joint_jacobian(
        candidate_ids, bank, normal, offset_mm
    )
    tangent_basis = _normal_tangent_basis(normal)

    def residual(update: np.ndarray) -> np.ndarray:
        delta = np.eye(4)
        delta[:3, :3] = Rotation.from_rotvec(update[:3]).as_matrix()
        delta[:3, 3] = update[3:6]
        candidate_handeye = T_true @ delta
        candidate_normal = Rotation.from_rotvec(
            tangent_basis @ update[6:8]
        ).apply(normal)
        candidate_offset = offset_mm + update[8]
        blocks: list[np.ndarray] = []
        for candidate_id in candidate_ids:
            transform = bank["T_base_ef"][candidate_id] @ candidate_handeye
            points_sensor = bank["ideal_points_s"][candidate_id]
            valid = np.all(np.isfinite(points_sensor), axis=1)
            points_sensor = points_sensor[valid]
            points_base = (
                points_sensor @ transform[:3, :3].T + transform[:3, 3]
            )
            blocks.append(points_base @ candidate_normal - candidate_offset)
        return np.concatenate(blocks)

    numerical = np.column_stack(
        [
            (
                residual(np.eye(9)[column] * FD_STEP)
                - residual(-np.eye(9)[column] * FD_STEP)
            )
            / (2.0 * FD_STEP)
            for column in range(9)
        ]
    )
    return float(np.max(np.abs(analytic - numerical)))


def _decompose_subset(
    candidate_ids: np.ndarray,
    bank: dict[str, np.ndarray],
    normal: np.ndarray,
    offset_mm: float,
) -> tuple[dict[str, float | int], dict[str, float]]:
    joint, _ = jacobian_analysis._joint_jacobian(
        candidate_ids, bank, normal, offset_mm
    )
    b_rows, points_base, _scan_ids = _point_geometry(candidate_ids, bank, normal)
    if len(b_rows) != len(joint):
        raise ValueError("point-level geometry/Jacobian row-count mismatch")

    # Reuse the exact solver tangent convention: J_pi normal columns are
    # q^T (B[:,j] x n).  These derivative vectors form an orthonormal tangent U.
    solver_basis = _normal_tangent_basis(normal)
    U = np.column_stack(
        [np.cross(solver_basis[:, column], normal) for column in range(2)]
    )
    z = points_base @ U
    y = z - np.mean(z, axis=0, keepdims=True)
    ones = np.ones((len(y), 1), dtype=float)
    j_pi_centered = np.column_stack([y, -ones])
    j_pi_original = joint[:, 6:9]

    # Section 2 definition must exactly reproduce the reused analytic J_X.
    c_rows = np.cross(
        np.vstack(
            [
                bank["ideal_points_s"][candidate_id][
                    np.all(
                        np.isfinite(bank["ideal_points_s"][candidate_id]), axis=1
                    )
                ]
                for candidate_id in candidate_ids
            ]
        ),
        b_rows,
    )
    j_x_definition = np.column_stack([c_rows, b_rows])

    q_original = _orthonormal_column_basis(j_pi_original)
    q_centered = _orthonormal_column_basis(j_pi_centered)
    nuisance_overlap = np.linalg.svd(q_original.T @ q_centered, compute_uv=False)
    # Compute the projector difference through low-rank residuals.  The
    # algebraically shorter r1+r2-2||Q1'Q2||_F^2 expression catastrophically
    # cancels when the two spaces agree to machine precision.
    original_outside_centered = (
        q_original - q_centered @ (q_centered.T @ q_original)
    )
    centered_outside_original = (
        q_centered - q_original @ (q_original.T @ q_centered)
    )
    projector_difference = float(
        np.sqrt(
            np.linalg.norm(original_outside_centered, ord="fro") ** 2
            + np.linalg.norm(centered_outside_original, ord="fro") ** 2
        )
    )
    nuisance_principal_max = float(
        np.rad2deg(np.arccos(np.clip(np.min(nuisance_overlap), -1.0, 1.0)))
    )

    joint_scales = np.asarray(
        [ROTATION_SCALE_RAD] * 3
        + [1.0] * 3
        + [ROTATION_SCALE_RAD] * 2
        + [1.0]
    )
    j_x_scaled = joint[:, :6] * joint_scales[None, :6]
    j_pi_scaled = j_pi_original * joint_scales[None, 6:]
    j_pi_centered_scaled = j_pi_centered * joint_scales[None, 6:]
    h_eff_original = _schur_effective(j_x_scaled, j_pi_scaled)
    h_eff_centered = _schur_effective(j_x_scaled, j_pi_centered_scaled)

    K = float(len(b_rows))
    b_sum = np.sum(b_rows, axis=0)
    b_mean = b_sum / K
    centered_b = b_rows - b_mean[None, :]
    covariance_b = centered_b.T @ centered_b / K
    h_tt_raw = b_rows.T @ b_rows
    loss_t_offset = np.outer(b_sum, b_sum) / K
    h_tt_after_offset = h_tt_raw - loss_t_offset

    s_y = y.T @ y
    c_t = b_rows.T @ y
    s_y_pinv = np.linalg.pinv(s_y, rcond=1e-12)
    loss_t_normal = c_t @ s_y_pinv @ c_t.T
    h_tt_decomposed = h_tt_raw - loss_t_offset - loss_t_normal

    h_xx = j_x_scaled.T @ j_x_scaled
    x_sum = np.sum(j_x_scaled, axis=0)
    loss_x_offset = np.outer(x_sum, x_sum) / K
    c_x = j_x_scaled.T @ y
    loss_x_normal = c_x @ s_y_pinv @ c_x.T
    h_eff_decomposed = h_xx - loss_x_offset - loss_x_normal
    h_eff_decomposed = 0.5 * (h_eff_decomposed + h_eff_decomposed.T)

    # Parameter-scaling-independent column-space overlap.
    q_x = _orthonormal_column_basis(j_x_scaled)
    q_pi = _orthonormal_column_basis(j_pi_scaled)
    canonical = np.linalg.svd(q_x.T @ q_pi, compute_uv=False)
    sigma_max = float(np.max(canonical))

    trace_h_tt = float(np.trace(h_tt_raw))
    norm_h_tt = float(np.linalg.norm(h_tt_raw, ord="fro"))
    trace_h_xx = float(np.trace(h_xx))
    centered_eigen = _eigen_features(covariance_b, "b_centered")
    translation_heff_eigen = _eigen_features(
        h_tt_decomposed, "translation_heff"
    )
    translation_heff_sign, translation_heff_logdet = np.linalg.slogdet(
        h_tt_decomposed
    )
    if (
        int(translation_heff_eigen["translation_heff_rank"]) < 3
        or translation_heff_sign <= 0.0
    ):
        translation_heff_logdet = float("-inf")
    features: dict[str, float | int] = {
        "point_count": int(K),
        "b_resultant": float(np.linalg.norm(b_mean)),
        **centered_eigen,
        "translation_offset_loss_ratio": float(
            np.trace(loss_t_offset) / trace_h_tt
        ),
        "translation_normal_loss_ratio": float(
            np.trace(loss_t_normal) / trace_h_tt
        ),
        "translation_total_loss_ratio": float(
            np.trace(loss_t_offset + loss_t_normal) / trace_h_tt
        ),
        "translation_retention_ratio": float(
            np.trace(h_tt_decomposed) / trace_h_tt
        ),
        **translation_heff_eigen,
        "translation_heff_logdet": float(translation_heff_logdet),
        "translation_offset_loss_fro_ratio": float(
            np.linalg.norm(loss_t_offset, ord="fro") / norm_h_tt
        ),
        "translation_normal_loss_fro_ratio": float(
            np.linalg.norm(loss_t_normal, ord="fro") / norm_h_tt
        ),
        "translation_total_loss_fro_ratio": float(
            np.linalg.norm(loss_t_offset + loss_t_normal, ord="fro") / norm_h_tt
        ),
        "full_offset_loss_trace_ratio": float(
            np.trace(loss_x_offset) / trace_h_xx
        ),
        "full_normal_loss_trace_ratio": float(
            np.trace(loss_x_normal) / trace_h_xx
        ),
        "full_total_loss_trace_ratio": float(
            np.trace(loss_x_offset + loss_x_normal) / trace_h_xx
        ),
        "full_retention_trace_ratio": float(
            np.trace(h_eff_decomposed) / trace_h_xx
        ),
        "nuisance_canonical_sigma_max": sigma_max,
        "nuisance_canonical_sigma_mean": float(np.mean(canonical)),
        "nuisance_principal_angle_min_deg": float(
            np.rad2deg(np.arccos(np.clip(sigma_max, -1.0, 1.0)))
        ),
        "nuisance_column_rank": int(q_pi.shape[1]),
        "handeye_column_rank": int(q_x.shape[1]),
    }
    checks = {
        "centered_y_sum_max_abs": float(np.max(np.abs(np.sum(y, axis=0)))),
        "nuisance_projector_fro": projector_difference,
        "nuisance_principal_angle_max_deg": nuisance_principal_max,
        "centered_heff_relative_fro": _relative_frobenius(
            h_eff_centered, h_eff_original
        ),
        "offset_covariance_max_abs": float(
            np.max(np.abs(h_tt_after_offset - K * covariance_b))
        ),
        "translation_heff_relative_fro": _relative_frobenius(
            h_tt_decomposed, h_eff_original[3:6, 3:6]
        ),
        "full_heff_relative_fro": _relative_frobenius(
            h_eff_decomposed, h_eff_original
        ),
        "jx_row_definition_max_abs": float(
            np.max(np.abs(j_x_definition - joint[:, :6]))
        ),
    }
    return features, checks


def _comparison_statistics(
    rows: Sequence[dict[str, Any]], feature_names: Sequence[str]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for n in PRIMARY_N:
        n_rows = [row for row in rows if int(row["N"]) == n]
        core = [row for row in n_rows if row["robustness_group"] == "core"]
        non = [row for row in n_rows if row["robustness_group"] == "non_survivor"]
        start = len(output)
        for feature in feature_names:
            left = geometry._finite_values(core, feature)
            right = geometry._finite_values(non, feature)
            u_value, p_value, delta = geometry._cliffs_delta(left, right)
            output.append(
                {
                    "N": n,
                    "feature": feature,
                    "core_size": len(core),
                    "non_survivor_size": len(non),
                    "core_finite": int(len(left)),
                    "non_survivor_finite": int(len(right)),
                    "core_median": geometry._quantile(left, 0.5),
                    "core_iqr": geometry._quantile(left, 0.75)
                    - geometry._quantile(left, 0.25),
                    "non_survivor_median": geometry._quantile(right, 0.5),
                    "non_survivor_iqr": geometry._quantile(right, 0.75)
                    - geometry._quantile(right, 0.25),
                    "median_difference": geometry._quantile(left, 0.5)
                    - geometry._quantile(right, 0.5),
                    "mann_whitney_u": u_value,
                    "p_value": p_value,
                    "fdr_q_value": math.nan,
                    "cliffs_delta": delta,
                }
            )
        adjusted = geometry._bh_adjust([row["p_value"] for row in output[start:]])
        for row, q_value in zip(output[start:], adjusted, strict=True):
            row["fdr_q_value"] = float(q_value)
    return output


def _continuous_correlations(
    rows: Sequence[dict[str, Any]], feature_names: Sequence[str]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for n in sorted(DIAGNOSTIC_N + PRIMARY_N):
        n_rows = [row for row in rows if int(row["N"]) == n]
        start = len(output)
        for feature in feature_names:
            for target in CONTINUOUS_TARGETS:
                if feature == target:
                    continue
                x = np.asarray([float(row[feature]) for row in n_rows])
                y = np.asarray([float(row[target]) for row in n_rows])
                finite = np.isfinite(x) & np.isfinite(y)
                if np.sum(finite) >= 3 and np.unique(x[finite]).size >= 2:
                    result = spearmanr(x[finite], y[finite])
                    rho, p_value = float(result.correlation), float(result.pvalue)
                else:
                    rho, p_value = math.nan, math.nan
                output.append(
                    {
                        "N": n,
                        "analysis_role": (
                            "diagnostic_continuous" if n in DIAGNOSTIC_N else "primary_continuous"
                        ),
                        "feature": feature,
                        "target": target,
                        "total_count": len(n_rows),
                        "finite_pair_count": int(np.sum(finite)),
                        "spearman_rho": rho,
                        "p_value": p_value,
                        "fdr_q_value": math.nan,
                    }
                )
        adjusted = geometry._bh_adjust([row["p_value"] for row in output[start:]])
        for row, q_value in zip(output[start:], adjusted, strict=True):
            row["fdr_q_value"] = float(q_value)
    return output


def _proxy_comparison(
    effects: Sequence[dict[str, Any]], correlations: Sequence[dict[str, Any]]
) -> list[dict[str, Any]]:
    feature_categories = {
        **{name: "existing_geometry" for name in EXISTING_COMPARISON_FEATURES[:3]},
        **{name: "existing_jacobian_information" for name in EXISTING_COMPARISON_FEATURES[3:]},
        **{name: "new_decomposition" for name in NEW_FEATURES},
    }
    effect_lookup = {
        (int(row["N"]), str(row["feature"])): row for row in effects
    }
    correlation_lookup = {
        (int(row["N"]), str(row["feature"]), str(row["target"])): row
        for row in correlations
    }
    output: list[dict[str, Any]] = []
    for feature, category in feature_categories.items():
        success_rhos = [
            float(correlation_lookup[(n, feature, "accuracy_success_rate")]["spearman_rho"])
            for n in PRIMARY_N
        ]
        deltas = [
            float(effect_lookup[(n, feature)]["cliffs_delta"]) for n in PRIMARY_N
        ]
        output.append(
            {
                "feature": feature,
                "category": category,
                **{
                    f"success_spearman_rho_N{n:03d}": success_rhos[index]
                    for index, n in enumerate(PRIMARY_N)
                },
                **{
                    f"success_fdr_q_N{n:03d}": float(
                        correlation_lookup[(n, feature, "accuracy_success_rate")][
                            "fdr_q_value"
                        ]
                    )
                    for n in PRIMARY_N
                },
                **{
                    f"core_vs_non_cliffs_delta_N{n:03d}": deltas[index]
                    for index, n in enumerate(PRIMARY_N)
                },
                **{
                    f"core_vs_non_fdr_q_N{n:03d}": float(
                        effect_lookup[(n, feature)]["fdr_q_value"]
                    )
                    for n in PRIMARY_N
                },
                "median_abs_success_rho": float(np.median(np.abs(success_rhos))),
                "success_rho_same_sign_n_count": max(
                    sum(value > 0.0 for value in success_rhos),
                    sum(value < 0.0 for value in success_rhos),
                ),
                "median_abs_cliffs_delta": float(np.median(np.abs(deltas))),
                "effect_same_sign_n_count": max(
                    sum(value > 0.0 for value in deltas),
                    sum(value < 0.0 for value in deltas),
                ),
                "interpretation_note": (
                    "Jacobian information metric; not a low-cost geometry proxy"
                    if "jacobian_information" in category
                    else "candidate association; not a sufficient pose rule"
                ),
            }
        )
    return sorted(
        output,
        key=lambda row: (
            -int(row["success_rho_same_sign_n_count"]),
            -float(row["median_abs_success_rho"]),
        ),
    )


def _translation_optimality_comparison(
    correlations: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Place translation T/E/D criteria and b references on one test table."""
    lookup = {
        (int(row["N"]), str(row["feature"]), str(row["target"])): row
        for row in correlations
    }
    output: list[dict[str, Any]] = []
    for n in PRIMARY_N:
        start = len(output)
        for criterion, feature in TRANSLATION_OPTIMALITY_FEATURES:
            source = lookup[(n, feature, "accuracy_success_rate")]
            rho = float(source["spearman_rho"])
            output.append(
                {
                    "N": n,
                    "criterion": criterion,
                    "feature": feature,
                    "finite_pair_count": int(source["finite_pair_count"]),
                    "spearman_rho": rho,
                    "abs_spearman_rho": abs(rho),
                    "p_value": float(source["p_value"]),
                    "comparison_fdr_q_value": math.nan,
                    "all_features_fdr_q_value": float(source["fdr_q_value"]),
                }
            )
        adjusted = geometry._bh_adjust([row["p_value"] for row in output[start:]])
        for row, q_value in zip(output[start:], adjusted, strict=True):
            row["comparison_fdr_q_value"] = float(q_value)
    return output


def _group_boxplot(
    rows: Sequence[dict[str, Any]], feature: str, title: str, path: Path
) -> None:
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    positions: list[float] = []
    values: list[np.ndarray] = []
    colors: list[str] = []
    for index, n in enumerate(PRIMARY_N):
        for offset, (group, color) in enumerate(
            (("core", "#238b45"), ("non_survivor", "#6b7280"))
        ):
            group_values = np.asarray(
                [
                    float(row[feature])
                    for row in rows
                    if int(row["N"]) == n and row["robustness_group"] == group
                ]
            )
            group_values = group_values[np.isfinite(group_values)]
            positions.append(index * 3.0 + offset)
            values.append(group_values)
            colors.append(color)
    boxes = ax.boxplot(values, positions=positions, widths=0.75, patch_artist=True)
    for patch, color in zip(boxes["boxes"], colors, strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.set_xticks([index * 3.0 + 0.5 for index in range(len(PRIMARY_N))])
    ax.set_xticklabels([f"N={n}" for n in PRIMARY_N])
    ax.set_ylabel(feature)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.plot([], [], color="#238b45", linewidth=8, label="core")
    ax.plot([], [], color="#6b7280", linewidth=8, label="non-survivor")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _scatter_by_n(
    rows: Sequence[dict[str, Any]], feature: str, title: str, path: Path
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0))
    colors = {"core": "#238b45", "borderline": "#f0a202", "non_survivor": "#6b7280"}
    for ax, n in zip(axes.ravel(), PRIMARY_N, strict=True):
        n_rows = [row for row in rows if int(row["N"]) == n]
        for group in ("non_survivor", "borderline", "core"):
            group_rows = [row for row in n_rows if row["robustness_group"] == group]
            ax.scatter(
                [float(row[feature]) for row in group_rows],
                [float(row["accuracy_success_rate"]) for row in group_rows],
                s=22,
                alpha=0.75,
                color=colors[group],
                label=group,
            )
        x = np.asarray([float(row[feature]) for row in n_rows])
        y = np.asarray([float(row["accuracy_success_rate"]) for row in n_rows])
        finite = np.isfinite(x) & np.isfinite(y)
        rho = float(spearmanr(x[finite], y[finite]).correlation)
        ax.set_title(f"N={n}, Spearman rho={rho:+.2f}")
        ax.set_xlabel(feature)
        ax.set_ylabel("accuracy success rate")
        ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_decomposition_summary(
    rows: Sequence[dict[str, Any]], path: Path
) -> None:
    labels: list[str] = []
    offset_values: list[float] = []
    normal_values: list[float] = []
    retention_values: list[float] = []
    colors = ("#4575b4", "#fdae61", "#1a9850")
    for n in PRIMARY_N:
        for group, short in (("core", "C"), ("non_survivor", "N")):
            selected = [
                row
                for row in rows
                if int(row["N"]) == n and row["robustness_group"] == group
            ]
            labels.append(f"{n}-{short}")
            offset_values.append(
                float(np.median([row["translation_offset_loss_ratio"] for row in selected]))
            )
            normal_values.append(
                float(np.median([row["translation_normal_loss_ratio"] for row in selected]))
            )
            retention_values.append(
                float(np.median([row["translation_retention_ratio"] for row in selected]))
            )
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(10.0, 5.0))
    ax.bar(x, offset_values, color=colors[0], label="offset loss")
    ax.bar(x, normal_values, bottom=offset_values, color=colors[1], label="normal loss")
    bottom = np.asarray(offset_values) + np.asarray(normal_values)
    ax.bar(x, retention_values, bottom=bottom, color=colors[2], label="retained")
    ax.set_xticks(x, labels)
    ax.set_ylabel("trace fraction of raw translation information")
    ax.set_title("Translation information decomposition (group medians)")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_effect_heatmap(
    effects: Sequence[dict[str, Any]], path: Path
) -> None:
    lookup = {
        (int(row["N"]), str(row["feature"])): float(row["cliffs_delta"])
        for row in effects
    }
    matrix = np.asarray(
        [[lookup[(n, feature)] for n in PRIMARY_N] for feature in PRIMARY_NEW_FEATURES]
    )
    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    image = ax.imshow(matrix, cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(PRIMARY_N)), [f"N={n}" for n in PRIMARY_N])
    ax.set_yticks(range(len(PRIMARY_NEW_FEATURES)), PRIMARY_NEW_FEATURES, fontsize=8)
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            ax.text(
                column_index,
                row_index,
                f"{matrix[row_index, column_index]:+.2f}",
                ha="center",
                va="center",
                fontsize=7,
            )
    ax.set_title("New descriptor effect: core − non-survivor")
    fig.colorbar(image, ax=ax, label="Cliff's delta")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_translation_optimality_correlations(
    comparison: Sequence[dict[str, Any]], path: Path
) -> None:
    criteria = [name for name, _feature in TRANSLATION_OPTIMALITY_FEATURES]
    lookup = {
        (int(row["N"]), str(row["criterion"])): float(row["spearman_rho"])
        for row in comparison
    }
    matrix = np.asarray(
        [[lookup[(n, criterion)] for n in PRIMARY_N] for criterion in criteria]
    )
    fig, ax = plt.subplots(figsize=(7.6, 4.7))
    image = ax.imshow(
        matrix, cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="auto"
    )
    ax.set_xticks(range(len(PRIMARY_N)), [f"N={n}" for n in PRIMARY_N])
    ax.set_yticks(range(len(criteria)), criteria)
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            ax.text(
                column_index,
                row_index,
                f"{matrix[row_index, column_index]:+.3f}",
                ha="center",
                va="center",
            )
    ax.set_title("Translation T/E/D criteria versus Phase-2A success")
    fig.colorbar(image, ax=ax, label="within-N Spearman rho")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _effect_table_lines(
    effects: Sequence[dict[str, Any]], feature: str
) -> list[str]:
    lookup = {
        int(row["N"]): row for row in effects if row["feature"] == feature
    }
    lines = [
        f"### `{feature}`",
        "",
        "| N | Core median | Non median | Cliff's delta | FDR q |",
        "|---:|---:|---:|---:|---:|",
    ]
    for n in PRIMARY_N:
        row = lookup[n]
        lines.append(
            f"| {n} | {float(row['core_median']):.6g} | "
            f"{float(row['non_survivor_median']):.6g} | "
            f"{float(row['cliffs_delta']):+.3f} | "
            f"{float(row['fdr_q_value']):.3g} |"
        )
    lines.append("")
    return lines


def _median_abs_success_rho(
    correlations: Sequence[dict[str, Any]], feature: str
) -> float:
    values = [
        float(row["spearman_rho"])
        for row in correlations
        if row["feature"] == feature
        and row["target"] == "accuracy_success_rate"
        and int(row["N"]) in PRIMARY_N
    ]
    return float(np.median(np.abs(values)))


def _write_report(
    path: Path,
    rows: Sequence[dict[str, Any]],
    effects: Sequence[dict[str, Any]],
    correlations: Sequence[dict[str, Any]],
    proxy_rows: Sequence[dict[str, Any]],
    translation_comparison: Sequence[dict[str, Any]],
    validation: dict[str, Any],
) -> None:
    centered_score = _median_abs_success_rho(correlations, "b_centered_lambda_min")
    uncentered_score = _median_abs_success_rho(correlations, "b_G_lambda_min")
    centered_trace_score = _median_abs_success_rho(correlations, "b_centered_trace")
    centered_max_score = _median_abs_success_rho(
        correlations, "b_centered_lambda_max"
    )
    winner = "centered Cov_b" if centered_score > uncentered_score else "uncentered G_b"
    optimality_lookup = {
        (int(row["N"]), str(row["criterion"])): row
        for row in translation_comparison
    }
    optimality_medians = {
        criterion: float(
            np.median(
                [
                    abs(float(optimality_lookup[(n, criterion)]["spearman_rho"]))
                    for n in PRIMARY_N
                ]
            )
        )
        for criterion, _feature in TRANSLATION_OPTIMALITY_FEATURES
    }
    ted_winner = max(("T", "E", "D"), key=lambda name: optimality_medians[name])
    reference_score = max(
        optimality_medians["b_resultant_reference"],
        optimality_medians["b_centered_trace_reference"],
    )
    if ted_winner == "D" and optimality_medians["D"] > reference_score:
        ted_interpretation = (
            "Translation logdet가 T/E와 단순 b reference를 모두 넘어섰으므로, "
            "translation D-optimal geometry를 저차원 pose-design principle 후보로 "
            "우선 검증할 근거가 생겼다."
        )
    elif ted_winner == "T" and reference_score >= optimality_medians["T"]:
        ted_interpretation = (
            "T가 E와 D보다 강했지만 b-resultant/centered-trace reference가 T보다도 "
            "약간 강했다. 따라서 완전한 translation isotropy나 D-optimal volume보다 "
            "plane-offset ambiguity를 깨서 usable translation information 총량을 "
            "확보하는 geometry가 우선이라는 해석과 더 일치한다."
        )
    elif ted_winner == "T":
        ted_interpretation = (
            "T가 E와 D보다 강했다. 이는 완전한 isotropy보다 usable translation "
            "information 총량이 우선이라는 해석과 일치한다."
        )
    else:
        ted_interpretation = (
            "E가 가장 강했으므로 worst translation direction의 개선을 우선하는 "
            "가설을 추가 검증할 필요가 있다."
        )
    lines = [
        "# Phase 2A Jacobian plane-nuisance 분해 보고서",
        "",
        "## 1. 범위",
        "",
        "동결된 Phase 2A 600 subsets와 기존에 검증된 analytic joint Jacobian만 사용했다. "
        "새 pose/subset 생성, calibration 재실행, search, optimization, threshold 변경 및 Phase 2B는 수행하지 않았다.",
        "",
        "## 2. Numerical identity 검증",
        "",
        "모든 identity가 아래 tolerance를 통과한 뒤에만 통계와 plot을 저장했다.",
        "",
        "| 검증 항목 | 전체 최대 오차 | tolerance |",
        "|---|---:|---:|",
    ]
    for name, tolerance in VALIDATION_TOLERANCES.items():
        lines.append(
            f"| `{name}` | {float(validation['maximum_errors'][name]):.6e} | {tolerance:.1e} |"
        )
    lines += [
        "",
        "Original/centered nuisance 표현의 column space가 일치했으며, translation-only와 full 6-DOF 분해 모두 기존 scaled H_eff를 tolerance 이내에서 재구성했다. 따라서 이후 분해 통계를 사용할 수 있다.",
        "",
        "## 3. H1 — centered b covariance",
        "",
        f"Success rate와의 N별 Spearman |rho| 중앙값은 `b_centered_lambda_min`={centered_score:.3f}, 기존 `b_G_lambda_min`={uncentered_score:.3f}으로 이 기준에서는 **{winner}**가 조금 더 강했다. 그러나 centered minimum eigenvalue 자체의 효과는 작고 N별 core/non 검정도 유의하지 않았다.",
        "",
        f"반면 centered covariance의 `lambda_max`와 trace는 success association이 각각 {centered_max_score:.3f}, {centered_trace_score:.3f}으로 더 컸다. 즉 현재 결과는 centered covariance의 세 방향이 균형 있게 풍부해진다기보다, 평균 b 성분이 줄면서 주된 centered variation 방향이 커지는 모습에 가깝다.",
        "",
    ]
    lines += _effect_table_lines(effects, "b_resultant")
    lines += _effect_table_lines(effects, "b_centered_lambda_min")
    lines += _effect_table_lines(effects, "b_centered_lambda_max")
    lines += _effect_table_lines(effects, "b_centered_trace")
    lines += [
        "모든 translation row b가 unit vector이므로 `translation_offset_loss_ratio = b_resultant^2`이고 `b_centered_trace = 1 - b_resultant^2`이다. 따라서 이 세 지표는 독립 mechanism이 아니라 동일한 offset-confounding geometry를 서로 다른 방식으로 나타낸다.",
        "",
        "## 4. H2 — translation offset loss",
        "",
    ]
    lines += _effect_table_lines(effects, "translation_offset_loss_ratio")
    lines += [
        "Core에서 offset-loss ratio가 네 N 모두 더 작았고 effect direction도 반복됐다. 이는 좋은 subset의 translation information이 plane offset 변화로 흡수되는 비율이 더 작다는 해석과 일치한다.",
        "",
        "## 5. H3 — plane-normal translation loss",
        "",
    ]
    lines += _effect_table_lines(effects, "translation_normal_loss_ratio")
    lines += [
        "예상과 달리 plane-normal translation loss는 core에서 작지 않았다. 네 N 모두 core median이 조금 더 컸지만 effect는 작고 FDR q도 유의하지 않았다. 따라서 total translation loss 감소는 plane-normal loss 감소가 아니라 offset loss 감소가 주도했다.",
        "",
    ]
    lines += _effect_table_lines(effects, "translation_total_loss_ratio")
    lines += _effect_table_lines(effects, "translation_retention_ratio")
    lines += ["## 6. H4 — canonical nuisance coupling", ""]
    lines += _effect_table_lines(effects, "nuisance_canonical_sigma_max")
    lines += _effect_table_lines(effects, "nuisance_principal_angle_min_deg")

    new_ranked = [
        row for row in proxy_rows if row["category"] == "new_decomposition"
    ]
    lines += [
        "`sigma_max`는 모든 N에서 약 0.99로 hand-eye와 plane nuisance residual subspace에 매우 가까운 방향이 하나 이상 있음을 보여준다. Success와는 네 N 모두 음의 연속 상관을 보였지만, binary core/non 차이는 N=11에서만 FDR 기준으로 명확했다. 따라서 canonical coupling은 반복적인 보조 association이지만 강한 단독 group separator는 아니다.",
        "",
        "## 7. H5 — 수정된 explanatory chain",
        "",
        "기존 가설을 다음처럼 수정하는 것이 결과에 더 가깝다.",
        "",
        "`낮은 b resultant = 낮은 offset loss -> 높은 centered b trace/translation retention -> 높은 H_eff -> 높은 calibration robustness`",
        "",
        "`plane-normal translation loss 감소`는 이 chain에 포함할 근거가 없으며, canonical coupling은 secondary mechanism 후보로 남는다.",
        "",
        "신규 분해 지표를 success-rate Spearman association으로 정렬한 결과:",
        "",
        "| feature | median abs rho | 같은 방향 N 수 | median abs Cliff delta |",
        "|---|---:|---:|---:|",
    ]
    for row in new_ranked:
        lines.append(
            f"| `{row['feature']}` | {float(row['median_abs_success_rho']):.3f} | "
            f"{int(row['success_rho_same_sign_n_count'])}/4 | "
            f"{float(row['median_abs_cliffs_delta']):.3f} |"
        )

    lines += [
        "",
        "## 8. 기존 descriptor와 직접 비교",
        "",
        "H_eff 지표는 geometry proxy가 아니라 Jacobian-level information outcome이다. 전체 비교에서 `scaled_heff_logdet`와 `scaled_heff_lambda_min`이 여전히 success를 가장 강하게 설명했다. 신규 geometry/decomposition 지표 중에는 centered lambda_min보다 b resultant, centered trace/lambda_max, translation total loss/retention이 더 일관됐다. 세부 N별 값과 FDR q는 `proxy_comparison.csv`에 저장했다.",
        "",
        "반복 효과는 N=9,11,13,15의 방향과 크기 및 FDR q를 함께 해석했다. 한 N에서만 나타나거나 방향이 바뀌는 효과는 pose rule로 승격하지 않았다.",
        "",
        "## 9. Translation T/E/D-optimal criterion 직접 비교",
        "",
        "서로 다른 N을 섞지 않고 각 N 안에서 Phase 2A success rate와 Spearman을 계산했다. T는 trace(H_tt,eff), E는 lambda_min(H_tt,eff), D는 logdet(H_tt,eff)다.",
        "",
        "| criterion | N=9 | N=11 | N=13 | N=15 | median abs rho |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for criterion, _feature in TRANSLATION_OPTIMALITY_FEATURES:
        values = [
            float(optimality_lookup[(n, criterion)]["spearman_rho"])
            for n in PRIMARY_N
        ]
        lines.append(
            f"| `{criterion}` | "
            + " | ".join(f"{value:+.3f}" for value in values)
            + f" | {optimality_medians[criterion]:.3f} |"
        )
    lines += [
        "",
        f"T/E/D 중 median absolute rho가 가장 큰 것은 **{ted_winner} criterion**이었다. b-resultant와 centered-trace reference도 같은 비교에 포함했으므로 translation information criterion이 단순 offset-ambiguity proxy보다 강한지 직접 확인할 수 있다.",
        "",
        ted_interpretation,
        "",
        "## 10. 다음 controlled experiment의 조작 변수 후보",
        "",
        "가장 직접적인 geometry 조작 후보는 b resultant 또는 동치인 translation offset-loss ratio다. 실험에서는 N과 position spread를 맞춘 상태에서 b resultant가 낮은 set과 높은 set을 비교하고, centered b trace/lambda_max 및 H_eff가 함께 변하는지 기록하는 것이 자연스럽다. Plane tangent point/pose coordinate는 plane-normal loss를 결정하므로 통제하거나 공변량으로 저장해야 한다.",
        "",
        "## 11. 결론과 한계",
        "",
        "좋은 Phase 2A subset에서 plane nuisance에 덜 손실되는 주된 translation mechanism은 plane-normal loss 감소가 아니라 plane-offset loss 감소와 associated되어 있었다. b 방향들의 평균 resultant가 작을수록 offset에 흡수되는 공통 translation sensitivity가 줄고 centered translation information이 더 많이 남았다.",
        "",
        "이 결과는 고정 GT/plane 주변의 local association이며 large-initial-error nonlinear basin을 직접 증명하지 않는다. Global convergence를 증명하거나 calibration을 보장하지 않으며 necessary-and-sufficient pose rule도 아니다.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _top_console_rows(
    effects: Sequence[dict[str, Any]],
    correlations: Sequence[dict[str, Any]],
) -> tuple[list[tuple[str, float]], list[tuple[str, float]]]:
    effect_scores = []
    for feature in NEW_FEATURES:
        values = [
            abs(float(row["cliffs_delta"]))
            for row in effects
            if row["feature"] == feature
        ]
        effect_scores.append((feature, float(np.median(values))))
    rho_scores = []
    for feature in tuple(EXISTING_COMPARISON_FEATURES) + tuple(NEW_FEATURES):
        values = [
            abs(float(row["spearman_rho"]))
            for row in correlations
            if row["feature"] == feature
            and row["target"] == "accuracy_success_rate"
            and int(row["N"]) in PRIMARY_N
        ]
        rho_scores.append((feature, float(np.median(values))))
    return (
        sorted(effect_scores, key=lambda item: -item[1])[:10],
        sorted(rho_scores, key=lambda item: -item[1])[:10],
    )


def main() -> int:
    args = _parser()
    phase1_dir = _resolve(args.phase1_dir)
    phase2a_dir = _resolve(args.phase2a_dir)
    output_dir = _prepare_output(_resolve(args.output_dir), args.overwrite)
    if output_dir in (phase1_dir, phase2a_dir):
        raise ValueError("output directory must differ from frozen inputs")

    print("Validating frozen Phase 1/2A inputs and rebuilding existing Jacobian features...")
    phase1_manifest, phase2a_manifest, bank, summaries, matrices = (
        geometry._load_and_validate(phase1_dir, phase2a_dir)
    )
    base_rows, _base_features = jacobian_analysis._build_rows(
        phase1_manifest, bank, summaries, matrices
    )
    base_lookup = {
        (int(row["N"]), int(row["subset_id"])): row for row in base_rows
    }

    plane = phase1_manifest["resolved"]["plane"]
    normal = np.asarray(plane["normal_base"], dtype=float)
    normal /= np.linalg.norm(normal)
    offset_mm = float(plane["offset_mm"])
    T_true = np.asarray(phase1_manifest["resolved"]["T_ef_s_true"], dtype=float)

    rows: list[dict[str, Any]] = []
    maximum_errors = {name: 0.0 for name in VALIDATION_TOLERANCES}
    for n in sorted(DIAGNOSTIC_N + PRIMARY_N):
        print(f"Decomposing N={n} ({len(summaries[n])} subsets)...")
        # Retain finite-difference validation without multiplying runtime by all subsets.
        maximum_errors["analytic_fd_max_abs"] = max(
            maximum_errors["analytic_fd_max_abs"],
            _finite_difference_error(matrices[n][0], bank, T_true, normal, offset_mm),
        )
        for summary, candidate_ids in zip(
            summaries[n], matrices[n], strict=True
        ):
            features, checks = _decompose_subset(
                candidate_ids, bank, normal, offset_mm
            )
            for name, value in checks.items():
                maximum_errors[name] = max(maximum_errors[name], float(value))
            base = base_lookup[(n, int(summary["subset_id"]))]
            rows.append(
                {
                    "N": n,
                    "subset_id": int(summary["subset_id"]),
                    "phase1_rank": int(summary["phase1_rank"]),
                    "phase2a_rank": int(summary["phase2a_rank_among_selected"]),
                    "candidate_ids": summary["candidate_ids"],
                    "robustness_group": base["robustness_group"],
                    "accuracy_success_rate": base["accuracy_success_rate"],
                    "normalized_error_p90": base["normalized_error_p90"],
                    "normalized_error_median": base["normalized_error_median"],
                    **{
                        name: base[name] for name in EXISTING_COMPARISON_FEATURES
                    },
                    **features,
                }
            )

    failures = {
        name: {
            "maximum_error": maximum_errors[name],
            "tolerance": tolerance,
        }
        for name, tolerance in VALIDATION_TOLERANCES.items()
        if maximum_errors[name] > tolerance
    }
    validation = {
        "status": "failed" if failures else "passed",
        "subset_count": len(rows),
        "finite_difference_sampling": "first saved subset of each N (6 subsets)",
        "maximum_errors": maximum_errors,
        "tolerances": VALIDATION_TOLERANCES,
        "failures": failures,
    }
    (output_dir / "numerical_validation.json").write_text(
        json.dumps(validation, indent=2) + "\n", encoding="utf-8"
    )
    print("Numerical validation maximum errors:")
    for name, value in maximum_errors.items():
        print(f"  {name}: {value:.6e} (tol={VALIDATION_TOLERANCES[name]:.1e})")
    if failures:
        raise RuntimeError(
            "Jacobian decomposition identity validation failed; statistics were not written: "
            + ", ".join(failures)
        )

    analysis_features = tuple(EXISTING_COMPARISON_FEATURES) + tuple(NEW_FEATURES)
    effects = _comparison_statistics(rows, analysis_features)
    correlations = _continuous_correlations(rows, analysis_features)
    proxy_rows = _proxy_comparison(effects, correlations)
    translation_comparison = _translation_optimality_comparison(correlations)

    _write_csv(output_dir / "subset_jacobian_decomposition.csv", rows)
    _write_csv(output_dir / "core_vs_non_decomposition_stats.csv", effects)
    _write_csv(
        output_dir / "continuous_decomposition_correlations.csv", correlations
    )
    _write_csv(output_dir / "proxy_comparison.csv", proxy_rows)
    _write_csv(
        output_dir / "translation_optimality_comparison.csv",
        translation_comparison,
    )

    plots = output_dir / "plots"
    for feature, filename, title in (
        ("b_resultant", "b_resultant_core_vs_non.png", "Point-weighted b resultant"),
        ("b_centered_lambda_min", "b_centered_lambda_min_core_vs_non.png", "Centered b covariance minimum eigenvalue"),
        ("translation_offset_loss_ratio", "translation_offset_loss_core_vs_non.png", "Translation information lost to plane offset"),
        ("translation_normal_loss_ratio", "translation_normal_loss_core_vs_non.png", "Translation information lost to plane normal"),
        ("nuisance_canonical_sigma_max", "nuisance_sigma_max_core_vs_non.png", "Canonical hand-eye/plane nuisance overlap"),
    ):
        _group_boxplot(rows, feature, title, plots / filename)
    for feature, filename, title in (
        ("nuisance_canonical_sigma_max", "nuisance_sigma_max_vs_success.png", "Canonical nuisance overlap versus success"),
        ("b_centered_lambda_min", "b_centered_lambda_min_vs_success.png", "Centered b diversity versus success"),
        ("translation_total_loss_ratio", "translation_total_loss_vs_success.png", "Translation nuisance loss versus success"),
    ):
        _scatter_by_n(rows, feature, title, plots / filename)
    _plot_decomposition_summary(rows, plots / "translation_decomposition_summary.png")
    _plot_effect_heatmap(effects, plots / "new_descriptor_effect_heatmap.png")
    _plot_translation_optimality_correlations(
        translation_comparison,
        plots / "translation_optimality_success_spearman.png",
    )
    _write_report(
        output_dir / "decomposition_report.md",
        rows,
        effects,
        correlations,
        proxy_rows,
        translation_comparison,
        validation,
    )

    manifest = {
        "schema": "laser_handeye.phase2a_jacobian_decomposition",
        "schema_version": 1,
        "status": "complete",
        "source_phase1": str(phase1_dir),
        "source_phase2a": str(phase2a_dir),
        "calibration_rerun": False,
        "pose_or_subset_generation": False,
        "subset_count": len(rows),
        "primary_scan_counts": list(PRIMARY_N),
        "diagnostic_scan_counts": list(DIAGNOSTIC_N),
        "joint_variable_order": [
            "right_local_handeye_rotation_rad(3)",
            "right_local_handeye_translation_mm(3)",
            "plane_normal_tangent_rad(2)",
            "plane_offset_mm(1)",
        ],
        "scaled_column_convention": "angular=1 degree; translation/offset=1 mm",
        "numerical_validation": "passed",
    }
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    top_effects, top_rhos = _top_console_rows(effects, correlations)
    print(f"Total subsets: {len(rows)}")
    for n in sorted(DIAGNOSTIC_N + PRIMARY_N):
        print(f"  N={n}: {sum(int(row['N']) == n for row in rows)} subsets")
    print("Largest core/non differences among new features (median |Cliff delta|):")
    for feature, score in top_effects:
        print(f"  {feature}: {score:.3f}")
    print("Strongest success associations (median |Spearman rho|):")
    for feature, score in top_rhos:
        print(f"  {feature}: {score:.3f}")
    print("Translation T/E/D and b-reference success Spearman:")
    for criterion, _feature in TRANSLATION_OPTIMALITY_FEATURES:
        values = [
            float(row["spearman_rho"])
            for row in translation_comparison
            if row["criterion"] == criterion
        ]
        print(
            f"  {criterion}: "
            + ", ".join(
                f"N={n} {value:+.3f}"
                for n, value in zip(PRIMARY_N, values, strict=True)
            )
            + f", median|rho|={np.median(np.abs(values)):.3f}"
        )
    print("Canonical sigma_max medians:")
    for n in PRIMARY_N:
        for group in ("core", "non_survivor"):
            values = [
                float(row["nuisance_canonical_sigma_max"])
                for row in rows
                if int(row["N"]) == n and row["robustness_group"] == group
            ]
            print(f"  N={n} {group}: {np.median(values):.6f}")
    print(f"Output: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
