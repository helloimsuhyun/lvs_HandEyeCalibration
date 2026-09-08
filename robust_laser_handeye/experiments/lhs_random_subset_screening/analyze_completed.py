#!/usr/bin/env python3
"""Create Phase-1 screening plots from an already completed result directory.

This script is intentionally independent of pandas so it also works in the
project's current NumPy environment.  It never runs calibration and never
modifies the source CSV/NPZ files.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "results" / "phase1_lhs_screening"
DEFAULT_OUTPUT_NAME = "analysis_plots"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot an existing Phase-1 LHS random-subset screening run."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Completed result directory (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Plot directory (default: <input-dir>/analysis_plots)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace PNG files already present in the output directory.",
    )
    return parser.parse_args()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _number(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return math.nan


def _truth(row: dict[str, str], key: str) -> bool:
    return row.get(key, "").strip().lower() in {"1", "true", "yes"}


def _finite(rows: Sequence[dict[str, str]], key: str) -> np.ndarray:
    values = np.asarray([_number(row, key) for row in rows], dtype=float)
    return values[np.isfinite(values)]


def _positive(values: np.ndarray) -> np.ndarray:
    return values[np.isfinite(values) & (values > 0.0)]


def _group_by_n(rows: Iterable[dict[str, str]]) -> dict[int, list[dict[str, str]]]:
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["N"])].append(row)
    return dict(sorted(grouped.items()))


def _save(figure: plt.Figure, path: Path, overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"plot already exists (use --overwrite): {path}")
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _log_hist(
    axis: plt.Axes,
    values: np.ndarray,
    threshold: float,
    xlabel: str,
) -> None:
    values = _positive(values)
    if values.size:
        low = min(float(values.min()), threshold)
        high = max(float(values.max()), threshold)
        if high > low:
            bins = np.geomspace(low, high, 41)
            axis.hist(values, bins=bins, color="tab:blue", alpha=0.8)
            axis.set_xscale("log")
        else:
            axis.hist(values, bins=5, color="tab:blue", alpha=0.8)
    else:
        axis.text(0.5, 0.5, "No finite positive values", ha="center", va="center")
    axis.axvline(threshold, color="tab:red", linestyle="--", label="threshold")
    axis.set(xlabel=xlabel, ylabel="Run count")
    axis.grid(alpha=0.2)
    axis.legend()


def _per_n_plots(
    output_dir: Path,
    runs_by_n: dict[int, list[dict[str, str]]],
    subsets_by_n: dict[int, list[dict[str, str]]],
    translation_threshold: float,
    rotation_threshold: float,
    minimum_pass_rate: float,
    overwrite: bool,
) -> list[Path]:
    paths: list[Path] = []
    for scan_count, runs in runs_by_n.items():
        directory = output_dir / "per_n" / f"N{scan_count:03d}"
        directory.mkdir(parents=True, exist_ok=True)
        subsets = subsets_by_n[scan_count]

        figure, axes = plt.subplots(2, 2, figsize=(11, 8))
        _log_hist(
            axes[0, 0],
            _finite(runs, "translation_error_mm"),
            translation_threshold,
            "Translation error [mm] (log scale)",
        )
        _log_hist(
            axes[0, 1],
            _finite(runs, "rotation_error_deg"),
            rotation_threshold,
            "Rotation error [deg] (log scale)",
        )
        _log_hist(
            axes[1, 0],
            _finite(runs, "normalized_error"),
            1.0,
            "Normalized error (log scale)",
        )
        pass_rates = _finite(subsets, "threshold_pass_rate")
        bins = np.linspace(-0.05, 1.05, 12)
        axes[1, 1].hist(pass_rates, bins=bins, color="tab:blue", alpha=0.8)
        axes[1, 1].axvline(
            minimum_pass_rate, color="tab:red", linestyle="--", label="promotion cutoff"
        )
        axes[1, 1].set(
            xlabel="Subset threshold pass rate", ylabel="Subset count", xlim=(-0.05, 1.05)
        )
        axes[1, 1].grid(alpha=0.2)
        axes[1, 1].legend()
        passed = sum(_truth(row, "passes_success_threshold") for row in runs)
        eligible = sum(_truth(row, "eligible_for_promotion") for row in subsets)
        figure.suptitle(
            f"N={scan_count}: {passed}/{len(runs)} runs pass; "
            f"{eligible}/{len(subsets)} subsets eligible"
        )
        figure.tight_layout()
        path = directory / "error_distributions.png"
        _save(figure, path, overwrite)
        paths.append(path)

        translation = _finite(runs, "translation_error_mm")
        rotation = _finite(runs, "rotation_error_deg")
        finite_mask = (
            np.isfinite(translation)
            & np.isfinite(rotation)
            & (translation > 0.0)
            & (rotation > 0.0)
        )
        # Both columns have one entry for every run in completed Phase-1 data.
        translation = translation[finite_mask]
        rotation = rotation[finite_mask]
        pass_mask = (
            (translation <= translation_threshold) & (rotation <= rotation_threshold)
        )
        figure, axis = plt.subplots(figsize=(7.5, 6.5))
        axis.scatter(
            translation[~pass_mask],
            rotation[~pass_mask],
            s=9,
            alpha=0.25,
            color="tab:gray",
            label="outside threshold",
            rasterized=True,
        )
        axis.scatter(
            translation[pass_mask],
            rotation[pass_mask],
            s=11,
            alpha=0.55,
            color="tab:green",
            label="inside threshold",
            rasterized=True,
        )
        axis.axvline(translation_threshold, color="tab:red", linestyle="--")
        axis.axhline(rotation_threshold, color="tab:red", linestyle="--")
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set(
            xlabel="Translation error [mm] (log scale)",
            ylabel="Rotation error [deg] (log scale)",
            title=f"N={scan_count}: calibration error pairs",
        )
        axis.grid(alpha=0.2, which="both")
        axis.legend()
        figure.tight_layout()
        path = directory / "translation_vs_rotation.png"
        _save(figure, path, overwrite)
        paths.append(path)
    return paths


def _quantile_band(
    axis: plt.Axes,
    scan_counts: Sequence[int],
    subsets_by_n: dict[int, list[dict[str, str]]],
    field: str,
    threshold: float,
    ylabel: str,
) -> None:
    quantiles = []
    for scan_count in scan_counts:
        values = _positive(_finite(subsets_by_n[scan_count], field))
        quantiles.append(np.quantile(values, [0.1, 0.5, 0.9]))
    quantiles_array = np.asarray(quantiles)
    axis.fill_between(
        scan_counts,
        quantiles_array[:, 0],
        quantiles_array[:, 2],
        alpha=0.2,
        color="tab:blue",
        label="subset 10th–90th percentile",
    )
    axis.plot(
        scan_counts,
        quantiles_array[:, 1],
        marker="o",
        color="tab:blue",
        label="subset median",
    )
    axis.axhline(threshold, color="tab:red", linestyle="--", label="success threshold")
    axis.set(xlabel="Scan count N", ylabel=ylabel, yscale="log", xticks=scan_counts)
    axis.grid(alpha=0.2, which="both")
    axis.legend(fontsize=8)


def _overview_plot(
    output_dir: Path,
    runs_by_n: dict[int, list[dict[str, str]]],
    subsets_by_n: dict[int, list[dict[str, str]]],
    translation_threshold: float,
    rotation_threshold: float,
    overwrite: bool,
) -> Path:
    scan_counts = list(subsets_by_n)
    run_pass_rates = [
        np.mean([_truth(row, "passes_success_threshold") for row in runs_by_n[n]])
        for n in scan_counts
    ]
    eligible_rates = [
        np.mean([_truth(row, "eligible_for_promotion") for row in subsets_by_n[n]])
        for n in scan_counts
    ]
    promoted_rates = [
        np.mean([_truth(row, "promoted") for row in subsets_by_n[n]])
        for n in scan_counts
    ]

    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes[0, 0].plot(scan_counts, run_pass_rates, marker="o", label="run success rate")
    axes[0, 0].plot(scan_counts, eligible_rates, marker="s", label="eligible subset rate")
    axes[0, 0].plot(scan_counts, promoted_rates, marker="^", label="promoted / all subsets")
    for n, rate in zip(scan_counts, run_pass_rates):
        axes[0, 0].annotate(f"{rate:.1%}", (n, rate), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=8)
    axes[0, 0].set(
        xlabel="Scan count N",
        ylabel="Fraction",
        ylim=(-0.03, 1.03),
        xticks=scan_counts,
        title="Accuracy classification and promotion",
    )
    axes[0, 0].grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)

    _quantile_band(
        axes[0, 1],
        scan_counts,
        subsets_by_n,
        "translation_error_median_mm",
        translation_threshold,
        "Subset median translation error [mm]",
    )
    _quantile_band(
        axes[1, 0],
        scan_counts,
        subsets_by_n,
        "rotation_error_median_deg",
        rotation_threshold,
        "Subset median rotation error [deg]",
    )
    _quantile_band(
        axes[1, 1],
        scan_counts,
        subsets_by_n,
        "normalized_error_median",
        1.0,
        "Subset median normalized error",
    )
    figure.suptitle("Phase 1 LHS random-subset screening overview")
    figure.tight_layout()
    path = output_dir / "phase1_overview.png"
    _save(figure, path, overwrite)
    return path


def _boxplot_panel(
    axis: plt.Axes,
    scan_counts: Sequence[int],
    subsets_by_n: dict[int, list[dict[str, str]]],
    field: str,
    ylabel: str,
    threshold: float | None,
    log_scale: bool,
) -> None:
    values = [_finite(subsets_by_n[n], field) for n in scan_counts]
    axis.boxplot(values, labels=[str(n) for n in scan_counts], showfliers=False)
    if threshold is not None:
        axis.axhline(threshold, color="tab:red", linestyle="--", label="threshold")
        axis.legend(fontsize=8)
    if log_scale:
        axis.set_yscale("log")
    axis.set(xlabel="Scan count N", ylabel=ylabel)
    axis.grid(alpha=0.2, which="both")


def _subset_distribution_plot(
    output_dir: Path,
    subsets_by_n: dict[int, list[dict[str, str]]],
    translation_threshold: float,
    rotation_threshold: float,
    minimum_pass_rate: float,
    overwrite: bool,
) -> Path:
    scan_counts = list(subsets_by_n)
    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    _boxplot_panel(
        axes[0, 0], scan_counts, subsets_by_n, "translation_error_median_mm",
        "Median translation error [mm]", translation_threshold, True
    )
    _boxplot_panel(
        axes[0, 1], scan_counts, subsets_by_n, "rotation_error_median_deg",
        "Median rotation error [deg]", rotation_threshold, True
    )
    _boxplot_panel(
        axes[1, 0], scan_counts, subsets_by_n, "normalized_error_median",
        "Median normalized error", 1.0, True
    )
    _boxplot_panel(
        axes[1, 1], scan_counts, subsets_by_n, "threshold_pass_rate",
        "Threshold pass rate", minimum_pass_rate, False
    )
    axes[1, 1].set_ylim(-0.05, 1.05)
    figure.suptitle("Distribution across 1,000 random subsets per N (outliers hidden)")
    figure.tight_layout()
    path = output_dir / "subset_metric_distributions.png"
    _save(figure, path, overwrite)
    return path


def _scatter_grid(
    output_dir: Path,
    runs_by_n: dict[int, list[dict[str, str]]],
    translation_threshold: float,
    rotation_threshold: float,
    overwrite: bool,
) -> Path:
    scan_counts = list(runs_by_n)
    figure, axes = plt.subplots(2, 3, figsize=(15, 9))
    for axis, scan_count in zip(axes.flat, scan_counts):
        runs = runs_by_n[scan_count]
        translation = np.asarray([_number(row, "translation_error_mm") for row in runs])
        rotation = np.asarray([_number(row, "rotation_error_deg") for row in runs])
        finite = (
            np.isfinite(translation) & np.isfinite(rotation)
            & (translation > 0.0) & (rotation > 0.0)
        )
        translation = translation[finite]
        rotation = rotation[finite]
        passed = (translation <= translation_threshold) & (rotation <= rotation_threshold)
        axis.scatter(
            translation[~passed], rotation[~passed], s=5, alpha=0.18,
            color="tab:gray", rasterized=True
        )
        axis.scatter(
            translation[passed], rotation[passed], s=6, alpha=0.45,
            color="tab:green", rasterized=True
        )
        axis.axvline(translation_threshold, color="tab:red", linestyle="--", linewidth=1)
        axis.axhline(rotation_threshold, color="tab:red", linestyle="--", linewidth=1)
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_title(f"N={scan_count} ({passed.mean():.1%} pass)")
        axis.grid(alpha=0.18, which="both")
    figure.supxlabel("Translation error [mm] (log scale)")
    figure.supylabel("Rotation error [deg] (log scale)")
    figure.suptitle("All 30,000 calibration runs")
    figure.tight_layout()
    path = output_dir / "translation_vs_rotation_by_n.png"
    _save(figure, path, overwrite)
    return path


def _rank_plot(
    output_dir: Path,
    subsets_by_n: dict[int, list[dict[str, str]]],
    overwrite: bool,
) -> Path:
    figure, axis = plt.subplots(figsize=(9, 6.5))
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(subsets_by_n)))
    promoted_label_used = False
    for color, (scan_count, rows) in zip(colors, subsets_by_n.items()):
        ranked = sorted(
            (
                (_number(row, "normalized_error_p90"), _truth(row, "promoted"))
                for row in rows
            ),
            key=lambda item: item[0],
        )
        ranked = [item for item in ranked if np.isfinite(item[0]) and item[0] > 0.0]
        values = np.asarray([item[0] for item in ranked], dtype=float)
        promoted = np.asarray([item[1] for item in ranked], dtype=bool)
        percentile = 100.0 * (np.arange(values.size) + 1) / values.size
        axis.plot(percentile, values, color=color, label=f"N={scan_count}")
        if promoted.any():
            axis.scatter(
                percentile[promoted],
                values[promoted],
                s=16,
                facecolors="none",
                edgecolors=color,
                linewidths=0.8,
                label="promoted subsets" if not promoted_label_used else None,
                zorder=3,
            )
            promoted_label_used = True
    axis.axhline(1.0, color="tab:red", linestyle="--", label="success threshold")
    axis.set(
        xlabel="Subset percentile after sorting by p90 normalized error [%]",
        ylabel="p90 normalized error (log scale)",
        yscale="log",
        title="Quality distribution of all random subsets",
    )
    axis.grid(alpha=0.2, which="both")
    axis.legend(ncol=2)
    figure.tight_layout()
    path = output_dir / "ranked_p90_normalized_error.png"
    _save(figure, path, overwrite)
    return path


def main() -> int:
    args = _parse_args()
    input_dir = args.input_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else input_dir / DEFAULT_OUTPUT_NAME
    )

    manifest_path = input_dir / "manifest.json"
    runs_path = input_dir / "calibration_runs.csv"
    subsets_path = input_dir / "subset_summary.csv"
    for path in (manifest_path, runs_path, subsets_path):
        if not path.is_file():
            raise FileNotFoundError(f"required completed-result file is missing: {path}")

    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest: dict[str, Any] = json.load(handle)
    if manifest.get("status") not in {None, "complete", "completed"}:
        raise RuntimeError(
            f"result manifest is not complete: status={manifest.get('status')!r}"
        )

    classification = manifest["config"]["classification"]
    translation_threshold = float(classification["translation_mm"])
    rotation_threshold = float(classification["rotation_deg"])
    minimum_pass_rate = float(classification["minimum_pass_rate_for_promotion"])

    print(f"Reading {runs_path}")
    runs = _read_csv(runs_path)
    print(f"Reading {subsets_path}")
    subsets = _read_csv(subsets_path)
    runs_by_n = _group_by_n(runs)
    subsets_by_n = _group_by_n(subsets)
    if set(runs_by_n) != set(subsets_by_n):
        raise ValueError("scan counts differ between calibration_runs.csv and subset_summary.csv")

    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    paths.extend(
        _per_n_plots(
            output_dir,
            runs_by_n,
            subsets_by_n,
            translation_threshold,
            rotation_threshold,
            minimum_pass_rate,
            args.overwrite,
        )
    )
    paths.append(
        _overview_plot(
            output_dir,
            runs_by_n,
            subsets_by_n,
            translation_threshold,
            rotation_threshold,
            args.overwrite,
        )
    )
    paths.append(
        _subset_distribution_plot(
            output_dir,
            subsets_by_n,
            translation_threshold,
            rotation_threshold,
            minimum_pass_rate,
            args.overwrite,
        )
    )
    paths.append(
        _scatter_grid(
            output_dir,
            runs_by_n,
            translation_threshold,
            rotation_threshold,
            args.overwrite,
        )
    )
    paths.append(_rank_plot(output_dir, subsets_by_n, args.overwrite))

    print(f"Created {len(paths)} plots in {output_dir}")
    for scan_count in sorted(runs_by_n):
        run_rows = runs_by_n[scan_count]
        subset_rows = subsets_by_n[scan_count]
        run_passes = sum(_truth(row, "passes_success_threshold") for row in run_rows)
        eligible = sum(_truth(row, "eligible_for_promotion") for row in subset_rows)
        promoted = sum(_truth(row, "promoted") for row in subset_rows)
        print(
            f"N={scan_count}: runs {run_passes}/{len(run_rows)} pass; "
            f"subsets {eligible}/{len(subset_rows)} eligible; {promoted} promoted"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
