#!/usr/bin/env python3
"""Interpret frozen Phase-2A results using the joint calibration Jacobian.

No calibration, pose generation, subset selection, or relabeling is performed.
The joint Jacobian is evaluated at the saved Phase-1 ground truth and true
single plane using the saved noise-free candidate profiles.  Its columns match
the joint nonlinear solver's right-local parameterization:

    [hand-eye rotation(rad), hand-eye translation(mm),
     plane-normal tangent(rad), plane offset(mm)]

The primary effective hand-eye information uses the solver's fixed 1-degree /
1-mm column scaling before eliminating the plane nuisance block by Schur
complement.  Unscaled physical-unit results are retained as secondary fields.
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
from scipy.stats import spearmanr


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.lhs_initialization_robustness import (  # noqa: E402
    analyze_phase2a_geometry as geometry,
)
from laser_handeye.nonlinear_refinement import _normal_tangent_basis  # noqa: E402


ROTATION_SCALE_RAD = float(np.deg2rad(1.0))
PRIMARY_N = geometry.PRIMARY_N
DIAGNOSTIC_N = geometry.DIAGNOSTIC_N
PERFORMANCE_FIELDS = geometry.PERFORMANCE_FIELDS
EPSILON = 1e-12


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
        default=PROJECT_ROOT / "results" / "phase2a_jacobian_interpretation",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace this analysis output; Phase-1/2A source files remain untouched",
    )
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _prepare_output(path: Path, overwrite: bool) -> Path:
    names = (
        "subset_jacobian_features.csv",
        "group_statistics.csv",
        "core_vs_non_effects.csv",
        "continuous_correlations.csv",
        "proxy_chain_correlations.csv",
        "observations.md",
        "analysis_manifest.json",
    )
    existing = [path / name for name in names if (path / name).exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"analysis output already exists; use --overwrite: {existing[0]}"
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


def _pairwise_angles_deg(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=float)
    vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    cosine = np.clip(vectors @ vectors.T, -1.0, 1.0)
    return np.rad2deg(
        np.arccos(cosine[np.triu_indices(len(vectors), k=1)])
    )


def _spectral_metrics(matrix: np.ndarray, prefix: str) -> dict[str, float | int]:
    matrix = 0.5 * (np.asarray(matrix, dtype=float) + np.asarray(matrix, dtype=float).T)
    eigenvalues = np.linalg.eigvalsh(matrix)
    maximum = float(eigenvalues[-1])
    numerical_floor = max(EPSILON, abs(maximum) * 1e-10)
    rank = int(np.sum(eigenvalues > numerical_floor))
    minimum = float(eigenvalues[0])
    condition = float("inf") if minimum <= numerical_floor else maximum / minimum
    logdet = (
        float(np.sum(np.log(eigenvalues)))
        if np.all(eigenvalues > numerical_floor)
        else float("-inf")
    )
    output: dict[str, float | int] = {
        f"{prefix}_lambda_min": minimum,
        f"{prefix}_lambda_max": maximum,
        f"{prefix}_condition": condition,
        f"{prefix}_logdet": logdet,
        f"{prefix}_rank": rank,
    }
    for index, value in enumerate(eigenvalues):
        output[f"{prefix}_eigenvalue_{index + 1}"] = float(value)
    return output


def _effective_information(
    jacobian: np.ndarray, column_scales: np.ndarray, prefix: str
) -> dict[str, float | int]:
    scaled = np.asarray(jacobian, dtype=float) * column_scales[None, :]
    hessian = scaled.T @ scaled
    h_xx = hessian[:6, :6]
    h_xpi = hessian[:6, 6:]
    h_pipi = hessian[6:, 6:]
    h_eff = h_xx - h_xpi @ np.linalg.solve(h_pipi, h_xpi.T)
    h_eff = 0.5 * (h_eff + h_eff.T)

    singular_values = np.linalg.svd(scaled, compute_uv=False)
    joint_condition = (
        float("inf")
        if singular_values[-1] <= EPSILON
        else float(singular_values[0] / singular_values[-1])
    )
    denominator = math.sqrt(
        max(float(np.linalg.norm(h_xx, ord="fro")), EPSILON)
        * max(float(np.linalg.norm(h_pipi, ord="fro")), EPSILON)
    )
    trace_hxx = float(np.trace(h_xx))
    output: dict[str, float | int] = {
        f"{prefix}_joint_jacobian_rank": int(np.linalg.matrix_rank(scaled)),
        f"{prefix}_joint_jacobian_condition": joint_condition,
        f"{prefix}_nuisance_lambda_min": float(np.linalg.eigvalsh(h_pipi)[0]),
        f"{prefix}_nuisance_condition": float(np.linalg.cond(h_pipi)),
        f"{prefix}_hxpi_fro": float(np.linalg.norm(h_xpi, ord="fro")),
        f"{prefix}_hxpi_normalized_fro": float(
            np.linalg.norm(h_xpi, ord="fro") / denominator
        ),
        f"{prefix}_information_retention_trace": (
            float(np.trace(h_eff)) / trace_hxx if trace_hxx > EPSILON else math.nan
        ),
    }
    output.update(_spectral_metrics(h_eff, f"{prefix}_heff"))
    return output


def _joint_jacobian(
    candidate_ids: np.ndarray,
    bank: dict[str, np.ndarray],
    plane_normal: np.ndarray,
    plane_offset_mm: float,
) -> tuple[np.ndarray, float]:
    """Return the exact first derivative at GT for the solver parameterization."""
    normal = np.asarray(plane_normal, dtype=float).reshape(3)
    normal /= np.linalg.norm(normal)
    tangent_basis = _normal_tangent_basis(normal)
    # d Exp([B u]x)n / du at zero = (B[:,j] x n).
    normal_derivatives = np.column_stack(
        [np.cross(tangent_basis[:, j], normal) for j in range(2)]
    )

    blocks: list[np.ndarray] = []
    maximum_truth_residual = 0.0
    for candidate_id in np.asarray(candidate_ids, dtype=np.int64):
        rotation_base_sensor = bank["T_base_s"][candidate_id, :3, :3]
        translation_base_sensor = bank["T_base_s"][candidate_id, :3, 3]
        points_sensor = bank["ideal_points_s"][candidate_id]
        points_base = (
            points_sensor @ rotation_base_sensor.T
            + translation_base_sensor[None, :]
        )
        b = rotation_base_sensor.T @ normal
        block = np.empty((len(points_sensor), 9), dtype=float)
        block[:, :3] = np.cross(points_sensor, b[None, :])
        block[:, 3:6] = b[None, :]
        block[:, 6:8] = points_base @ normal_derivatives
        block[:, 8] = -1.0
        blocks.append(block)
        maximum_truth_residual = max(
            maximum_truth_residual,
            float(np.max(np.abs(points_base @ normal - plane_offset_mm))),
        )
    return np.vstack(blocks), maximum_truth_residual


def _subset_features(
    candidate_ids: np.ndarray,
    bank: dict[str, np.ndarray],
    plane_normal: np.ndarray,
    plane_offset_mm: float,
) -> dict[str, float | int]:
    transforms = bank["T_base_s"][candidate_ids]
    rotations = transforms[:, :3, :3]
    viewing_directions = rotations[:, :, 2]
    b_vectors = np.einsum("nji,j->ni", rotations, plane_normal)
    b_vectors /= np.linalg.norm(b_vectors, axis=1, keepdims=True)

    view_angles = _pairwise_angles_deg(viewing_directions)
    b_angles = _pairwise_angles_deg(b_vectors)
    g_b = b_vectors.T @ b_vectors / float(len(b_vectors))
    g_eigenvalues = np.linalg.eigvalsh(g_b)
    g_min = float(g_eigenvalues[0])
    g_max = float(g_eigenvalues[-1])

    # The canonical sensor +Z points toward the plane, so b_z=-cos(tilt).
    cos_theta = -b_vectors[:, 2]
    declared_cos = np.cos(
        np.deg2rad(bank["pose_parameters"][candidate_ids, 3])
    )
    cos_consistency = float(np.max(np.abs(cos_theta - declared_cos)))
    if cos_consistency > 1e-8:
        raise ValueError(
            "saved transform/tilt convention mismatch: "
            f"max |(-b_z)-cos(tilt)|={cos_consistency:.3e}"
        )

    jacobian, maximum_truth_residual = _joint_jacobian(
        candidate_ids, bank, plane_normal, plane_offset_mm
    )
    physical_scales = np.ones(9, dtype=float)
    solver_scales = np.asarray(
        [ROTATION_SCALE_RAD] * 3
        + [1.0] * 3
        + [ROTATION_SCALE_RAD] * 2
        + [1.0],
        dtype=float,
    )

    output: dict[str, float | int] = {
        "D_view_pairwise_angle_mean_deg": float(np.mean(view_angles)),
        "D_view_pairwise_angle_min_deg": float(np.min(view_angles)),
        "D_view_pairwise_angle_max_deg": float(np.max(view_angles)),
        "b_pairwise_angle_mean_deg": float(np.mean(b_angles)),
        "b_pairwise_angle_min_deg": float(np.min(b_angles)),
        "b_pairwise_angle_max_deg": float(np.max(b_angles)),
        "b_G_lambda_min": g_min,
        "b_G_lambda_mid": float(g_eigenvalues[1]),
        "b_G_lambda_max": g_max,
        "b_G_condition": float("inf") if g_min <= EPSILON else g_max / g_min,
        "b_G_eigenvalue_std": float(np.std(g_eigenvalues, ddof=0)),
        "b_resultant": float(np.linalg.norm(np.mean(b_vectors, axis=0))),
        "cos_theta_min": float(np.min(cos_theta)),
        "cos_theta_max": float(np.max(cos_theta)),
        "cos_theta_mean": float(np.mean(cos_theta)),
        "cos_theta_std": float(np.std(cos_theta, ddof=0)),
        "cos_theta_var": float(np.var(cos_theta, ddof=0)),
        "cos_theta_transform_parameter_max_abs_difference": cos_consistency,
        "gt_point_to_plane_max_abs_residual_mm": maximum_truth_residual,
        "joint_residual_count": int(jacobian.shape[0]),
        "joint_variable_count": int(jacobian.shape[1]),
    }
    output.update(_effective_information(jacobian, solver_scales, "scaled"))
    output.update(_effective_information(jacobian, physical_scales, "physical"))
    return output


def _build_rows(
    phase1_manifest: dict[str, Any],
    bank: dict[str, np.ndarray],
    summaries: dict[int, list[dict[str, str]]],
    candidate_matrices: dict[int, np.ndarray],
) -> tuple[list[dict[str, Any]], list[str]]:
    plane = phase1_manifest["resolved"]["plane"]
    normal = np.asarray(plane["normal_base"], dtype=float)
    normal /= np.linalg.norm(normal)
    offset = float(plane["offset_mm"])
    T_true = np.asarray(phase1_manifest["resolved"]["T_ef_s_true"], dtype=float)

    # The saved bank must be consistent with the GT used for linearization.
    reconstructed = np.einsum("nij,njk->nik", bank["T_base_ef"], T_true[None, :, :])
    bank_transform_error = float(np.max(np.abs(reconstructed - bank["T_base_s"])))
    if bank_transform_error > 1e-8:
        raise ValueError(
            "candidate bank transforms do not match the saved Phase-1 GT: "
            f"max abs difference={bank_transform_error:.3e}"
        )

    rows: list[dict[str, Any]] = []
    feature_names: list[str] | None = None
    for n in sorted(summaries):
        print(f"Computing frozen-GT Jacobians for N={n} ({len(summaries[n])} subsets)...")
        for summary, candidate_ids in zip(
            summaries[n], candidate_matrices[n], strict=True
        ):
            features = _subset_features(candidate_ids, bank, normal, offset)
            if feature_names is None:
                feature_names = list(features)
            elif list(features) != feature_names:
                raise RuntimeError("Jacobian feature order changed between subsets")
            rows.append(
                {
                    "N": n,
                    "subset_id": int(summary["subset_id"]),
                    "phase1_rank": int(summary["phase1_rank"]),
                    "phase2a_rank": int(summary["phase2a_rank_among_selected"]),
                    "candidate_ids": summary["candidate_ids"],
                    "robustness_group": geometry._label(summary),
                    "accuracy_success_rate": geometry._float(
                        summary, "accuracy_success_rate"
                    ),
                    "normalized_error_p90": geometry._float(
                        summary, "normalized_error_p90"
                    ),
                    "normalized_error_median": geometry._float(
                        summary, "normalized_error_median"
                    ),
                    **features,
                }
            )
    assert feature_names is not None
    return rows, feature_names


def _proxy_chain_correlations(
    rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    sources = (
        "D_view_pairwise_angle_mean_deg",
        "b_pairwise_angle_mean_deg",
        "b_G_lambda_min",
        "b_G_condition",
        "cos_theta_var",
        "scaled_heff_lambda_min",
        "scaled_heff_logdet",
        "scaled_heff_condition",
    )
    targets = (
        "b_pairwise_angle_mean_deg",
        "b_G_lambda_min",
        "b_G_condition",
        "cos_theta_var",
        "scaled_heff_lambda_min",
        "scaled_heff_logdet",
        "scaled_heff_condition",
        "scaled_hxpi_normalized_fro",
        "scaled_information_retention_trace",
        "accuracy_success_rate",
        "normalized_error_p90",
    )
    output: list[dict[str, Any]] = []
    for n in sorted({int(row["N"]) for row in rows}):
        n_rows = [row for row in rows if int(row["N"]) == n]
        start = len(output)
        for source in sources:
            for target in targets:
                if source == target:
                    continue
                x = np.asarray([float(row[source]) for row in n_rows])
                y = np.asarray([float(row[target]) for row in n_rows])
                finite = np.isfinite(x) & np.isfinite(y)
                if np.sum(finite) >= 3 and np.unique(x[finite]).size >= 2:
                    result = spearmanr(x[finite], y[finite])
                    rho, p = float(result.correlation), float(result.pvalue)
                else:
                    rho, p = math.nan, math.nan
                output.append(
                    {
                        "N": n,
                        "source_feature": source,
                        "target_feature": target,
                        "finite_pair_count": int(np.sum(finite)),
                        "spearman_rho": rho,
                        "p_value": p,
                        "fdr_q_value": math.nan,
                    }
                )
        q_values = geometry._bh_adjust([row["p_value"] for row in output[start:]])
        for row, q_value in zip(output[start:], q_values, strict=True):
            row["fdr_q_value"] = float(q_value)
    return output


def _plot_effect_heatmap(
    effects: Sequence[dict[str, Any]], path: Path
) -> None:
    requested = (
        "D_view_pairwise_angle_mean_deg",
        "b_pairwise_angle_mean_deg",
        "b_G_lambda_min",
        "b_G_condition",
        "cos_theta_var",
        "scaled_heff_lambda_min",
        "scaled_heff_logdet",
        "scaled_heff_condition",
        "scaled_hxpi_normalized_fro",
        "scaled_information_retention_trace",
    )
    lookup = {
        (int(row["N"]), str(row["feature"])): float(row["cliffs_delta"])
        for row in effects
    }
    matrix = np.asarray(
        [[lookup[(n, feature)] for n in PRIMARY_N] for feature in requested]
    )
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    image = ax.imshow(matrix, cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="auto")
    ax.set_xticks(range(len(PRIMARY_N)), [f"N={n}" for n in PRIMARY_N])
    ax.set_yticks(range(len(requested)), requested, fontsize=8)
    ax.set_title("Core survivor − non-survivor effect (Cliff's delta)")
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            ax.text(
                column_index,
                row_index,
                f"{matrix[row_index, column_index]:.2f}",
                ha="center",
                va="center",
                fontsize=7,
            )
    fig.colorbar(image, ax=ax, label="positive = larger in core")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_chain(
    rows: Sequence[dict[str, Any]], path: Path
) -> None:
    pairs = (
        ("D_view_pairwise_angle_mean_deg", "b_pairwise_angle_mean_deg"),
        ("D_view_pairwise_angle_mean_deg", "scaled_heff_lambda_min"),
        ("b_pairwise_angle_mean_deg", "scaled_heff_lambda_min"),
        ("scaled_heff_lambda_min", "accuracy_success_rate"),
    )
    fig, axes = plt.subplots(len(PRIMARY_N), len(pairs), figsize=(14.5, 11.0))
    colors = {"core": "#238b45", "borderline": "#f0a202", "non_survivor": "#6b7280"}
    for row_index, n in enumerate(PRIMARY_N):
        n_rows = [row for row in rows if int(row["N"]) == n]
        for column_index, (x_name, y_name) in enumerate(pairs):
            ax = axes[row_index, column_index]
            for group in ("non_survivor", "borderline", "core"):
                group_rows = [row for row in n_rows if row["robustness_group"] == group]
                if group_rows:
                    ax.scatter(
                        [row[x_name] for row in group_rows],
                        [row[y_name] for row in group_rows],
                        s=15,
                        alpha=0.70,
                        color=colors[group],
                        label=group if row_index == 0 and column_index == 0 else None,
                    )
            x = np.asarray([float(row[x_name]) for row in n_rows])
            y = np.asarray([float(row[y_name]) for row in n_rows])
            finite = np.isfinite(x) & np.isfinite(y)
            rho = float(spearmanr(x[finite], y[finite]).correlation)
            ax.text(0.03, 0.95, f"N={n}, rho={rho:.2f}", transform=ax.transAxes, va="top")
            if row_index == len(PRIMARY_N) - 1:
                ax.set_xlabel(x_name, fontsize=8)
            if column_index == 0:
                ax.set_ylabel(y_name, fontsize=8)
            elif row_index == 0:
                ax.set_title(f"{x_name}\n→ {y_name}", fontsize=8)
            ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Proposed proxy chain, evaluated within each N", y=1.00)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _find_correlation(
    rows: Sequence[dict[str, Any]], n: int, source: str, target: str
) -> dict[str, Any]:
    return next(
        row
        for row in rows
        if int(row["N"]) == n
        and row["source_feature"] == source
        and row["target_feature"] == target
    )


def _write_observations(
    path: Path,
    rows: Sequence[dict[str, Any]],
    effects: Sequence[dict[str, Any]],
    correlations: Sequence[dict[str, Any]],
    chain: Sequence[dict[str, Any]],
) -> None:
    effect_lookup = {
        (int(row["N"]), str(row["feature"])): row for row in effects
    }
    lines = [
        "# Phase 2A Jacobian interpretation",
        "",
        "## Scope and convention",
        "",
        "- No calibration was rerun and the saved Phase 2A labels were not changed.",
        "- Jacobians were evaluated at the frozen Phase 1 GT hand-eye and true plane, using saved noise-free profiles.",
        "- The primary `scaled_*` fields use 1 degree for angular columns and 1 mm for translation/offset columns, matching the joint nonlinear solver.",
        "- `physical_*` fields retain radian/mm columns and should not be compared as a unit-neutral condition score.",
        "- Primary survivor comparisons are within N for N=9,11,13,15; borderline subsets are excluded from core-vs-non tests.",
        "",
        "## Within-N proxy chain",
        "",
        "Spearman rho values below are descriptive associations, not causal effects.",
        "",
        "| N | D_view→b-angle | D_view→lambda_min(H_eff) | b-angle→lambda_min(H_eff) | lambda_min(H_eff)→success |",
        "|---:|---:|---:|---:|---:|",
    ]
    links = (
        ("D_view_pairwise_angle_mean_deg", "b_pairwise_angle_mean_deg"),
        ("D_view_pairwise_angle_mean_deg", "scaled_heff_lambda_min"),
        ("b_pairwise_angle_mean_deg", "scaled_heff_lambda_min"),
        ("scaled_heff_lambda_min", "accuracy_success_rate"),
    )
    for n in PRIMARY_N:
        values = [
            float(_find_correlation(chain, n, source, target)["spearman_rho"])
            for source, target in links
        ]
        lines.append(f"| {n} | " + " | ".join(f"{value:+.3f}" for value in values) + " |")

    lines += [
        "",
        "## Core survivor versus non-survivor",
        "",
        "Positive Cliff's delta means the feature is larger in core survivors.",
        "",
        "| feature | N=9 | N=11 | N=13 | N=15 |",
        "|---|---:|---:|---:|---:|",
    ]
    key_features = (
        "b_pairwise_angle_mean_deg",
        "b_G_lambda_min",
        "cos_theta_var",
        "scaled_heff_lambda_min",
        "scaled_heff_logdet",
        "scaled_heff_condition",
        "scaled_hxpi_normalized_fro",
    )
    for feature in key_features:
        cells = [float(effect_lookup[(n, feature)]["cliffs_delta"]) for n in PRIMARY_N]
        lines.append(f"| `{feature}` | " + " | ".join(f"{value:+.3f}" for value in cells) + " |")

    lines += [
        "",
        "## Reading guide",
        "",
        "- `b_*` measures diversity of the actual translation-Jacobian direction b_i=R_BS_i^T n.",
        "- `cos_theta_var` directly measures variation in the t_z/plane-offset coupling term under the saved canonical pose convention.",
        "- `scaled_heff_*` is the six-dimensional hand-eye information remaining after the plane nuisance variables are eliminated.",
        "- `scaled_hxpi_normalized_fro` and `scaled_information_retention_trace` help diagnose whether plane coupling, rather than b diversity alone, explains performance.",
        "- Multiple-testing-adjusted q-values and all finite/non-finite counts are in the CSV outputs. Do not infer a pose rule from one N or one metric alone.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = _parser()
    phase1_dir = _resolve(args.phase1_dir)
    phase2a_dir = _resolve(args.phase2a_dir)
    output_dir = _prepare_output(_resolve(args.output_dir), args.overwrite)
    if output_dir in (phase1_dir, phase2a_dir):
        raise ValueError("analysis output must differ from Phase-1/Phase-2A input")

    print("Validating frozen Phase 1 and Phase 2A source files...")
    phase1_manifest, phase2a_manifest, bank, summaries, matrices = (
        geometry._load_and_validate(phase1_dir, phase2a_dir)
    )
    rows, feature_names = _build_rows(
        phase1_manifest, bank, summaries, matrices
    )
    group_statistics = geometry._group_statistics(rows, feature_names)
    effects = geometry._effects_for_comparison(
        rows,
        feature_names,
        PRIMARY_N,
        "core",
        "non_survivor",
        "primary_within_N_core_vs_non_survivor",
    )
    continuous = geometry._continuous_correlations(rows, feature_names)
    chain = _proxy_chain_correlations(rows)

    _write_csv(output_dir / "subset_jacobian_features.csv", rows)
    _write_csv(output_dir / "group_statistics.csv", group_statistics)
    _write_csv(output_dir / "core_vs_non_effects.csv", effects)
    _write_csv(output_dir / "continuous_correlations.csv", continuous)
    _write_csv(output_dir / "proxy_chain_correlations.csv", chain)
    _plot_effect_heatmap(effects, output_dir / "plots" / "core_vs_non_effects.png")
    _plot_chain(rows, output_dir / "plots" / "proxy_chain_scatter.png")
    _write_observations(
        output_dir / "observations.md", rows, effects, continuous, chain
    )

    manifest = {
        "schema": "laser_handeye.phase2a_jacobian_interpretation",
        "schema_version": 1,
        "status": "complete",
        "source_phase1": str(phase1_dir),
        "source_phase2a": str(phase2a_dir),
        "source_phase1_schema": phase1_manifest["schema"],
        "source_phase2a_schema": phase2a_manifest["schema"],
        "calibration_rerun": False,
        "linearization": "Phase-1 GT hand-eye + true plane + noise-free profiles",
        "joint_variable_order": [
            "right_local_handeye_rotation_rad(3)",
            "right_local_handeye_translation_mm(3)",
            "plane_normal_tangent_rad(2)",
            "plane_offset_mm(1)",
        ],
        "primary_column_scales": [ROTATION_SCALE_RAD] * 3
        + [1.0] * 3
        + [ROTATION_SCALE_RAD] * 2
        + [1.0],
        "primary_scan_counts": list(PRIMARY_N),
        "diagnostic_scan_counts_saved_without_primary_group_test": list(DIAGNOSTIC_N),
        "subset_count": len(rows),
        "feature_count": len(feature_names),
        "label_thresholds_unchanged": {
            "core": {"success_rate_min": 0.90, "normalized_error_p90_max": 1.0},
            "borderline": {"success_rate_min": 0.80, "normalized_error_p90_max": 1.5},
        },
    }
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )

    print(f"Analyzed {len(rows)} frozen Phase-2A subsets; no calibration was run.")
    for n in sorted(summaries):
        n_rows = [row for row in rows if int(row["N"]) == n]
        counts = {
            group: sum(row["robustness_group"] == group for row in n_rows)
            for group in ("core", "borderline", "non_survivor")
        }
        print(
            f"N={n}: subsets={len(n_rows)}, core={counts['core']}, "
            f"borderline={counts['borderline']}, non={counts['non_survivor']}"
        )
    print(f"Output: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
