#!/usr/bin/env python3
"""Formal subset-level effect decomposition of the completed N=9 Stage 2."""

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


BASE_CONFOUNDERS = (
    "rotation_jacobian_logdet",
    "position_cov_trace_mm2",
    "position_span_mm",
    "d_std_mm",
    "uv_cov_trace_mm2",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"empty output: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def _zscore(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    mean = float(np.mean(values)); scale = float(np.std(values, ddof=0))
    if not np.isfinite(scale) or scale <= 1e-12:
        raise ValueError("cannot standardize constant predictor")
    return (values - mean) / scale, mean, scale


def _ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    design = np.column_stack([np.ones(len(X)), X])
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = y - design @ beta
    dof = max(1, len(y) - design.shape[1])
    sigma2 = float(residual @ residual / dof)
    covariance = sigma2 * np.linalg.pinv(design.T @ design)
    r2 = 1.0 - float(residual @ residual / np.sum((y - np.mean(y)) ** 2))
    return beta, r2, covariance


def _vif(X: np.ndarray, names: Sequence[str]) -> dict[str, float]:
    output: dict[str, float] = {}
    for index, name in enumerate(names):
        other = np.delete(X, index, axis=1)
        if other.shape[1] == 0:
            output[name] = 1.0; continue
        _beta, r2, _covariance = _ols(other, X[:, index])
        output[name] = float("inf") if r2 >= 1.0 - 1e-12 else 1.0 / (1.0 - r2)
    return output


def _select_adjusted_predictors(
    X_by_name: dict[str, np.ndarray], optional: Sequence[str]
) -> tuple[list[str], list[dict[str, Any]]]:
    selected = ["z_trace", "z_log_condition", *optional]
    audit: list[dict[str, Any]] = []
    while True:
        X = np.column_stack([X_by_name[name] for name in selected])
        values = _vif(X, selected)
        audit.extend({"iteration": len(audit), "predictor": name, "vif": value, "selected": True} for name, value in values.items())
        optional_selected = [name for name in selected if name not in ("z_trace", "z_log_condition")]
        if not optional_selected:
            break
        worst = max(optional_selected, key=lambda name: values[name])
        if values[worst] <= 10.0:
            break
        selected.remove(worst)
        audit.append({"iteration": len(audit), "predictor": worst, "vif": values[worst], "selected": False})
    return selected, audit


def _bootstrap_coefficients(
    X: np.ndarray, y: np.ndarray, repeats: int, rng: np.random.Generator
) -> np.ndarray:
    output = np.empty((repeats, X.shape[1] + 1), dtype=float)
    for repeat in range(repeats):
        indices = rng.integers(0, len(y), size=len(y))
        output[repeat] = _ols(X[indices], y[indices])[0]
    return output


def _model_rows(
    model: str,
    names: Sequence[str],
    beta: np.ndarray,
    standardized_beta: np.ndarray,
    bootstrap: np.ndarray,
    r2: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, name in enumerate(("intercept", *names)):
        values = bootstrap[:, index]
        rows.append({
            "model": model,
            "predictor": name,
            "coefficient": float(beta[index]),
            "bootstrap_ci_low": float(np.quantile(values, 0.025)),
            "bootstrap_ci_high": float(np.quantile(values, 0.975)),
            "standardized_outcome_coefficient": float(standardized_beta[index]),
            "r_squared": r2,
        })
    return rows


def _scatter(x: np.ndarray, y: np.ndarray, xlabel: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.8)); ax.scatter(x, y, s=18, alpha=0.65)
    result = spearmanr(x, y); ax.set_title(f"Spearman rho={float(result.correlation):+.3f}")
    ax.set_xlabel(xlabel); ax.set_ylabel("translation median error (mm)"); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def main() -> int:
    args = _parser(); config = json.loads(args.config.read_text(encoding="utf-8"))
    source = _resolve(config["existing_stage2_dir"])
    output = _resolve(args.output_dir or config["output_dir"]); output.mkdir(parents=True, exist_ok=True)
    plots = output / "plots"; plots.mkdir(exist_ok=True)
    path = output / "existing_stage2_effect_decomposition.csv"
    if path.exists(): raise FileExistsError(f"refusing to overwrite: {path}")
    raw_rows = _read_csv(source / "stage2_subset_summary.csv")
    rows = [{key: value for key, value in row.items()} for row in raw_rows]
    T = np.asarray([float(row["b_cov_trace"]) for row in rows])
    K = np.log(np.asarray([float(row["b_cov_condition"]) for row in rows]))
    raw_y = np.asarray([float(row["translation_error_median_mm"]) for row in rows])
    y = np.log(raw_y)
    zT, _, _ = _zscore(T); zK, _, _ = _zscore(K)
    y_standardized, _, _ = _zscore(y)
    predictors: dict[str, np.ndarray] = {"z_trace": zT, "z_log_condition": zK}
    optional_names: list[str] = []
    for field in BASE_CONFOUNDERS:
        values = np.asarray([float(row[field]) for row in rows])
        standardized, _, _ = _zscore(values)
        name = f"z_{field}"; predictors[name] = standardized; optional_names.append(name)

    repeats = int(config["bootstrap_repeats"]); rng = np.random.default_rng(int(config["seed"]))
    output_rows: list[dict[str, Any]] = []
    basic_names = ["z_trace", "z_log_condition"]
    basic_X = np.column_stack([predictors[name] for name in basic_names])
    beta, r2, _ = _ols(basic_X, y); standardized_beta, _, _ = _ols(basic_X, y_standardized)
    bootstrap = _bootstrap_coefficients(basic_X, y, repeats, rng)
    output_rows.extend(_model_rows("primary_log_error", basic_names, beta, standardized_beta, bootstrap, r2))
    magnitude_difference = np.abs(bootstrap[:, 1]) - np.abs(bootstrap[:, 2])

    raw_beta, raw_r2, _ = _ols(basic_X, raw_y)
    raw_standardized, _, _ = _zscore(raw_y); raw_standardized_beta, _, _ = _ols(basic_X, raw_standardized)
    raw_bootstrap = _bootstrap_coefficients(basic_X, raw_y, repeats, rng)
    output_rows.extend(_model_rows("sensitivity_raw_error", basic_names, raw_beta, raw_standardized_beta, raw_bootstrap, raw_r2))

    adjusted_names, vif_audit = _select_adjusted_predictors(predictors, optional_names)
    adjusted_X = np.column_stack([predictors[name] for name in adjusted_names])
    adjusted_beta, adjusted_r2, _ = _ols(adjusted_X, y); adjusted_standardized_beta, _, _ = _ols(adjusted_X, y_standardized)
    adjusted_bootstrap = _bootstrap_coefficients(adjusted_X, y, repeats, rng)
    output_rows.extend(_model_rows("confounder_adjusted_log_error", adjusted_names, adjusted_beta, adjusted_standardized_beta, adjusted_bootstrap, adjusted_r2))

    interaction_X = np.column_stack([zT, zK, zT * zK]); interaction_names = ["z_trace", "z_log_condition", "z_trace_x_z_log_condition"]
    interaction_beta, interaction_r2, _ = _ols(interaction_X, y); interaction_standardized_beta, _, _ = _ols(interaction_X, y_standardized)
    interaction_bootstrap = _bootstrap_coefficients(interaction_X, y, repeats, rng)
    output_rows.extend(_model_rows("exploratory_interaction", interaction_names, interaction_beta, interaction_standardized_beta, interaction_bootstrap, interaction_r2))
    _write_csv(path, output_rows); _write_csv(output / "existing_stage2_vif_audit.csv", vif_audit)

    lambda_max = np.asarray([float(row["b_cov_lambda_max"]) for row in rows]); lambda_mid = np.asarray([float(row["b_cov_lambda_mid"]) for row in rows]); lambda_min = np.asarray([float(row["b_cov_lambda_min"]) for row in rows])
    logdet = np.log(lambda_max) + np.log(lambda_mid) + np.log(lambda_min)
    spread_component = 3.0 * np.log(T)
    isotropy_component = np.log(lambda_max / T) + np.log(lambda_mid / T) + np.log(lambda_min / T)
    identity_error = float(np.max(np.abs(logdet - spread_component - isotropy_component)))
    correlation_rows: list[dict[str, Any]] = []
    for name, values in (("3_log_trace", spread_component), ("normalized_covariance_logdet", isotropy_component), ("covariance_logdet", logdet)):
        result = spearmanr(values, raw_y)
        correlation_rows.append({"component": name, "spearman_rho_with_translation_median_error": float(result.correlation), "p_value": float(result.pvalue), "logdet_identity_max_abs_error": identity_error})
    _write_csv(output / "existing_stage2_logdet_decomposition.csv", correlation_rows)

    correlation_matrix_names = basic_names + optional_names
    correlation_matrix = np.corrcoef(np.column_stack([predictors[name] for name in correlation_matrix_names]), rowvar=False)
    matrix_rows = [{"predictor": name, **{other: float(correlation_matrix[i, j]) for j, other in enumerate(correlation_matrix_names)}} for i, name in enumerate(correlation_matrix_names)]
    _write_csv(output / "existing_stage2_predictor_correlations.csv", matrix_rows)

    _scatter(T, raw_y, "trace(Cov(b))", plots / "existing_trace_vs_translation_error.png")
    _scatter(K, raw_y, "log condition(Cov(b))", plots / "existing_log_condition_vs_translation_error.png")
    _scatter(logdet, raw_y, "logdet(Cov(b))", plots / "existing_logdet_vs_translation_error.png")
    fig, ax = plt.subplots(figsize=(5.8, 4.6)); centers=[beta[1], beta[2]]; lower=[np.quantile(bootstrap[:,1],.025),np.quantile(bootstrap[:,2],.025)]; upper=[np.quantile(bootstrap[:,1],.975),np.quantile(bootstrap[:,2],.975)]
    ax.errorbar([0,1], centers, yerr=[np.asarray(centers)-lower, np.asarray(upper)-centers], fmt="o", capsize=5); ax.axhline(0,color="black",linewidth=.8); ax.set_xticks([0,1],["beta_T","beta_K"]); ax.set_ylabel("log-error coefficient per predictor SD"); ax.set_title("N=9 formal effect decomposition"); ax.grid(alpha=.25)
    fig.tight_layout(); fig.savefig(plots / "existing_standardized_beta_trace_vs_condition.png", dpi=180); plt.close(fig)

    report_lines = [
        "# Existing N=9 Stage 2 formal effect decomposition", "",
        f"Subset is the independent unit (n={len(rows)}); 90 runs per subset were summarized before modeling.", "",
        "## Primary model", "",
        f"- beta_T: {beta[1]:+.5f}, 95% bootstrap CI [{np.quantile(bootstrap[:,1],.025):+.5f}, {np.quantile(bootstrap[:,1],.975):+.5f}]",
        f"- beta_K: {beta[2]:+.5f}, 95% bootstrap CI [{np.quantile(bootstrap[:,2],.025):+.5f}, {np.quantile(bootstrap[:,2],.975):+.5f}]",
        f"- |beta_T|-|beta_K|: {abs(beta[1])-abs(beta[2]):+.5f}, bootstrap CI [{np.quantile(magnitude_difference,.025):+.5f}, {np.quantile(magnitude_difference,.975):+.5f}]", "",
        "Negative beta_T means larger trace is associated with lower translation error; positive beta_K means worse condition is associated with larger error.", "",
        "## Confounder-adjusted model", "",
        f"Retained predictors after VIF audit: {', '.join(adjusted_names)}", "",
        f"- adjusted beta_T: {adjusted_beta[adjusted_names.index('z_trace')+1]:+.5f}",
        f"- adjusted beta_K: {adjusted_beta[adjusted_names.index('z_log_condition')+1]:+.5f}", "",
        "## Exploratory interaction", "",
        f"beta_TxK: {interaction_beta[3]:+.5f}, 95% CI [{np.quantile(interaction_bootstrap[:,3],.025):+.5f}, {np.quantile(interaction_bootstrap[:,3],.975):+.5f}]", "",
        "## Logdet decomposition", "",
        f"Identity maximum error: {identity_error:.3e}", "",
    ]
    for row in correlation_rows:
        report_lines.append(f"- {row['component']}: Spearman rho={float(row['spearman_rho_with_translation_median_error']):+.3f}")
    report_lines += ["", "This is an observational decomposition within the selected matched population, not a causal or global-convergence claim.", ""]
    (output / "existing_stage2_effect_decomposition.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Existing Stage 2 decomposition complete: {output}")
    return 0


if __name__ == "__main__": raise SystemExit(main())
