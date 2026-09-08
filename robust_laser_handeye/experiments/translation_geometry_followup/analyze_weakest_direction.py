#!/usr/bin/env python3
"""Compare trace, weakest-direction, condition, and A-optimal geometry at N=9/11.

This is a read-only subset-level analysis of completed calibration summaries.
It does not treat the 90 calibration runs within a subset as independent data.
"""

from __future__ import annotations

import argparse
import csv
import json
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


SOURCES = {
    9: Path("results/translation_trace_isotropy_controlled/stage2_subset_summary.csv"),
    11: Path("results/translation_geometry_followup/fresh_N11_stage2_results.csv"),
}
METRICS = (
    ("trace", -1.0),
    ("lambda_min", -1.0),
    ("condition", +1.0),
    ("trace_inverse", +1.0),
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/translation_geometry_followup"),
    )
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=5811)
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"empty output: {path}")
    if path.exists():
        raise FileExistsError(f"refusing to overwrite: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _arrays(rows: Sequence[dict[str, str]]) -> dict[str, np.ndarray]:
    maximum = np.asarray([float(row["b_cov_lambda_max"]) for row in rows])
    middle = np.asarray([float(row["b_cov_lambda_mid"]) for row in rows])
    minimum = np.asarray([float(row["b_cov_lambda_min"]) for row in rows])
    trace = maximum + middle + minimum
    return {
        "translation_error_median_mm": np.asarray(
            [float(row["translation_error_median_mm"]) for row in rows]
        ),
        "lambda_max": maximum,
        "lambda_mid": middle,
        "lambda_min": minimum,
        "trace": trace,
        "condition": maximum / minimum,
        "trace_inverse": 1.0 / maximum + 1.0 / middle + 1.0 / minimum,
        "lambda_max_share": maximum / trace,
        "lambda_mid_share": middle / trace,
        "lambda_min_share": minimum / trace,
    }


def _rho(x: np.ndarray, y: np.ndarray) -> float:
    return float(spearmanr(x, y).correlation)


def _bootstrap_rhos(
    arrays: dict[str, np.ndarray], repeats: int, rng: np.random.Generator
) -> dict[str, np.ndarray]:
    count = len(arrays["trace"])
    output = {name: np.empty(repeats) for name, _direction in METRICS}
    for repeat in range(repeats):
        index = rng.integers(0, count, count)
        y = arrays["translation_error_median_mm"][index]
        for name, _direction in METRICS:
            output[name][repeat] = _rho(arrays[name][index], y)
    return output


def _quantile_interval(values: np.ndarray) -> tuple[float, float]:
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def _bootstrap_median_difference(
    left: np.ndarray,
    right: np.ndarray,
    repeats: int,
    rng: np.random.Generator,
) -> np.ndarray:
    output = np.empty(repeats)
    for repeat in range(repeats):
        left_sample = left[rng.integers(0, len(left), len(left))]
        right_sample = right[rng.integers(0, len(right), len(right))]
        output[repeat] = np.median(right_sample) - np.median(left_sample)
    return output


def _scatter(
    datasets: dict[int, dict[str, np.ndarray]], metric: str, xlabel: str, path: Path
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5), sharey=True)
    for axis, n in zip(axes, (9, 11), strict=True):
        x = datasets[n][metric]
        y = datasets[n]["translation_error_median_mm"]
        axis.scatter(x, y, s=17, alpha=0.6)
        axis.set_yscale("log")
        if metric == "trace_inverse":
            axis.set_xscale("log")
        axis.set_title(f"N={n}, Spearman rho={_rho(x, y):+.3f}")
        axis.set_xlabel(xlabel)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("translation median error (mm, log scale)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> int:
    args = _parser()
    if args.bootstrap_repeats <= 0:
        raise ValueError("--bootstrap-repeats must be positive")
    output = _resolve(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    plots = output / "plots"
    plots.mkdir(exist_ok=True)
    report_path = output / "N9_N11_weakest_direction_report.md"
    if report_path.exists():
        raise FileExistsError(f"refusing to overwrite: {report_path}")

    datasets: dict[int, dict[str, np.ndarray]] = {}
    for n, relative_path in SOURCES.items():
        datasets[n] = _arrays(_read_csv(_resolve(relative_path)))

    rng = np.random.default_rng(args.seed)
    bootstraps = {
        n: _bootstrap_rhos(arrays, args.bootstrap_repeats, rng)
        for n, arrays in datasets.items()
    }
    correlation_rows: list[dict[str, Any]] = []
    for n, arrays in datasets.items():
        y = arrays["translation_error_median_mm"]
        for metric, direction in METRICS:
            result = spearmanr(arrays[metric], y)
            low, high = _quantile_interval(bootstraps[n][metric])
            correlation_rows.append(
                {
                    "N": n,
                    "subset_count": len(y),
                    "metric": metric,
                    "spearman_rho": float(result.correlation),
                    "bootstrap_ci_low": low,
                    "bootstrap_ci_high": high,
                    "p_value": float(result.pvalue),
                    "expected_direction_strength": direction * float(result.correlation),
                }
            )
    _write_csv(output / "N9_N11_metric_correlations.csv", correlation_rows)

    comparison_rows: list[dict[str, Any]] = []
    for n in (9, 11):
        for metric in ("lambda_min", "condition", "trace_inverse"):
            direction = dict(METRICS)[metric]
            metric_strength = direction * bootstraps[n][metric]
            trace_strength = -bootstraps[n]["trace"]
            difference = metric_strength - trace_strength
            low, high = _quantile_interval(difference)
            comparison_rows.append(
                {
                    "comparison_scope": f"within_N{n}",
                    "metric": metric,
                    "reference": "trace",
                    "expected_strength_difference": float(
                        direction * _rho(datasets[n][metric], datasets[n]["translation_error_median_mm"])
                        + _rho(datasets[n]["trace"], datasets[n]["translation_error_median_mm"])
                    ),
                    "bootstrap_ci_low": low,
                    "bootstrap_ci_high": high,
                }
            )
    for metric, direction in METRICS:
        difference = direction * bootstraps[11][metric] - direction * bootstraps[9][metric]
        low, high = _quantile_interval(difference)
        comparison_rows.append(
            {
                "comparison_scope": "N11_minus_N9",
                "metric": metric,
                "reference": "same_metric_N9",
                "expected_strength_difference": float(
                    direction
                    * (
                        _rho(datasets[11][metric], datasets[11]["translation_error_median_mm"])
                        - _rho(datasets[9][metric], datasets[9]["translation_error_median_mm"])
                    )
                ),
                "bootstrap_ci_low": low,
                "bootstrap_ci_high": high,
            }
        )
    _write_csv(output / "N9_N11_metric_strength_comparisons.csv", comparison_rows)

    distribution_rows: list[dict[str, Any]] = []
    for metric in (
        "lambda_max",
        "lambda_mid",
        "lambda_min",
        "trace",
        "lambda_max_share",
        "lambda_mid_share",
        "lambda_min_share",
        "condition",
        "trace_inverse",
    ):
        left, right = datasets[9][metric], datasets[11][metric]
        difference = _bootstrap_median_difference(
            left, right, args.bootstrap_repeats, rng
        )
        low, high = _quantile_interval(difference)
        for n, values in ((9, left), (11, right)):
            quantiles = np.quantile(values, [0.1, 0.25, 0.5, 0.75, 0.9])
            distribution_rows.append(
                {
                    "metric": metric,
                    "N": n,
                    "subset_count": len(values),
                    "q10": float(quantiles[0]),
                    "q25": float(quantiles[1]),
                    "median": float(quantiles[2]),
                    "q75": float(quantiles[3]),
                    "q90": float(quantiles[4]),
                    "N11_minus_N9_median": float(np.median(right) - np.median(left)),
                    "median_difference_ci_low": low,
                    "median_difference_ci_high": high,
                }
            )
    _write_csv(output / "N9_N11_eigenvalue_distributions.csv", distribution_rows)

    contribution_rows: list[dict[str, Any]] = []
    for n, arrays in datasets.items():
        weakest_fraction = (1.0 / arrays["lambda_min"]) / arrays["trace_inverse"]
        contribution_rows.append(
            {
                "N": n,
                "median_fraction_of_trace_inverse_from_lambda_min": float(
                    np.median(weakest_fraction)
                ),
                "q10": float(np.quantile(weakest_fraction, 0.1)),
                "q90": float(np.quantile(weakest_fraction, 0.9)),
                "spearman_trace_inverse_vs_inverse_lambda_min": _rho(
                    arrays["trace_inverse"], 1.0 / arrays["lambda_min"]
                ),
            }
        )
    _write_csv(output / "N9_N11_Aoptimal_contributions.csv", contribution_rows)

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    positions = np.arange(len(METRICS))
    width = 0.34
    for offset, n, color in ((-width / 2, 9, "#4575b4"), (width / 2, 11, "#d73027")):
        centers = []
        lower = []
        upper = []
        for metric, direction in METRICS:
            center = direction * _rho(
                datasets[n][metric], datasets[n]["translation_error_median_mm"]
            )
            samples = direction * bootstraps[n][metric]
            lo, hi = _quantile_interval(samples)
            centers.append(center)
            lower.append(center - lo)
            upper.append(hi - center)
        ax.bar(positions + offset, centers, width, label=f"N={n}", color=color, alpha=0.8)
        ax.errorbar(
            positions + offset,
            centers,
            yerr=[lower, upper],
            fmt="none",
            ecolor="black",
            capsize=3,
            linewidth=1,
        )
    ax.set_xticks(positions, ["trace", "lambda_min", "condition", "tr(C^-1)"])
    ax.set_ylabel("expected-direction Spearman strength")
    ax.set_title("Translation median error geometry association")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plots / "N9_N11_metric_correlation_comparison.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.4))
    for axis, metric, title in zip(
        axes,
        ("lambda_max", "lambda_mid", "lambda_min"),
        ("lambda_max", "lambda_mid", "lambda_min"),
        strict=True,
    ):
        axis.boxplot(
            [datasets[9][metric], datasets[11][metric]],
            labels=["N=9", "N=11"],
            showfliers=False,
        )
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    fig.suptitle("Eigenvalue distributions in the two selected matched populations")
    fig.tight_layout()
    fig.savefig(plots / "N9_N11_eigenvalue_distributions.png", dpi=180)
    plt.close(fig)
    _scatter(
        datasets,
        "trace_inverse",
        "tr(Cov(b)^-1)",
        plots / "N9_N11_Aoptimal_vs_translation_error.png",
    )
    _scatter(
        datasets,
        "lambda_min",
        "lambda_min(Cov(b))",
        plots / "N9_N11_lambda_min_vs_translation_error.png",
    )

    correlation_lookup = {
        (int(row["N"]), str(row["metric"])): row for row in correlation_rows
    }
    comparison_lookup = {
        (str(row["comparison_scope"]), str(row["metric"])): row
        for row in comparison_rows
    }
    lines = [
        "# N=9 / N=11 weakest-direction analysis",
        "",
        "No calibration was rerun. Each subset is one independent statistical unit.",
        "",
        "## Spearman association with translation median error",
        "",
        "| N | trace | lambda_min | condition | tr(C^-1) |",
        "|---:|---:|---:|---:|---:|",
    ]
    for n in (9, 11):
        cells = []
        for metric, _direction in METRICS:
            row = correlation_lookup[(n, metric)]
            cells.append(
                f"{float(row['spearman_rho']):+.3f} "
                f"[{float(row['bootstrap_ci_low']):+.3f}, {float(row['bootstrap_ci_high']):+.3f}]"
            )
        lines.append(f"| {n} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "Signs follow raw metric direction: larger trace/lambda_min should reduce error, while larger condition/tr(C^-1) should increase it.",
        "",
        "## Is weakest-direction stronger than trace?",
        "",
        "The following uses expected-direction strength, so a positive difference means the listed metric is more strongly associated than trace.",
        "",
        "| N | metric minus trace | difference | 95% bootstrap CI |",
        "|---:|---|---:|---:|",
    ]
    for n in (9, 11):
        for metric in ("lambda_min", "condition", "trace_inverse"):
            row = comparison_lookup[(f"within_N{n}", metric)]
            lines.append(
                f"| {n} | {metric} | {float(row['expected_strength_difference']):+.3f} | "
                f"[{float(row['bootstrap_ci_low']):+.3f}, {float(row['bootstrap_ci_high']):+.3f}] |"
            )

    lines += ["", "## Direct eigenvalue distribution comparison", ""]
    for metric in ("lambda_max", "lambda_mid", "lambda_min", "trace", "lambda_min_share"):
        left = datasets[9][metric]
        right = datasets[11][metric]
        lines.append(
            f"- {metric}: median N=9 {np.median(left):.6g}, N=11 {np.median(right):.6g} "
            f"(N11/N9 {np.median(right)/np.median(left):.3f})"
        )

    cross_min = comparison_lookup[("N11_minus_N9", "lambda_min")]
    cross_a = comparison_lookup[("N11_minus_N9", "trace_inverse")]
    lines += [
        "",
        "## Interpretation",
        "",
        "- Within N=11, lambda_min and tr(C^-1) are more strongly associated with error than trace.",
        "- The same metrics were already informative at N=9; this is not a clean phase transition.",
        f"- N11-minus-N9 association-strength CI: lambda_min [{float(cross_min['bootstrap_ci_low']):+.3f}, {float(cross_min['bootstrap_ci_high']):+.3f}], tr(C^-1) [{float(cross_a['bootstrap_ci_low']):+.3f}, {float(cross_a['bootstrap_ci_high']):+.3f}].",
        "- N=9 and N=11 came from separately selected, differently matched populations. Therefore differences cannot be attributed causally to scan count N.",
        "- The current data support 'secure total spread while avoiding a weak eigen-direction' as a stronger hypothesis, but do not yet establish it as the cause or final pose rule.",
        "",
    ]
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Weakest-direction analysis complete: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
