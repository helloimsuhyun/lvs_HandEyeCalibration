#!/usr/bin/env python3
"""Compare fixed/moderate/wide target-position ranges."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


METHODS = (
    ("single_plane", "Single-plane"),
    ("three_plane", "Three-plane"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot calibration performance and translation-error outlier "
            "rates for several target U/V ranges."
        )
    )
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--levels", nargs="+", required=True)
    parser.add_argument(
        "--level-labels",
        nargs="+",
        default=None,
        help="Display labels in the same order as --levels.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--outlier-thresholds-mm",
        nargs="+",
        type=float,
        default=(1.0, 2.0),
    )
    parser.add_argument("--success-only", action="store_true")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def as_bool(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y", "t"}


def as_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def read_trials(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"trials.csv not found: {path}")

    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))

    if not rows:
        raise RuntimeError(f"No trial rows found: {path}")
    if "trial_index" not in rows[0]:
        raise KeyError(f"trial_index column is missing: {path}")

    seen: set[int] = set()
    for row in rows:
        index = int(row["trial_index"])
        if index in seen:
            raise RuntimeError(f"Duplicate trial_index={index}: {path}")
        seen.add(index)
    return rows


def finite_values(
    rows: Iterable[dict[str, str]],
    column: str,
    success_only: bool,
) -> np.ndarray:
    values = []
    for row in rows:
        if str(row.get("error_type", "")).strip():
            continue
        if success_only and not as_bool(row.get("success", False)):
            continue
        value = as_float(row.get(column))
        if math.isfinite(value):
            values.append(value)
    return np.asarray(values, dtype=float)


def summary(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {
            "mean": float("nan"),
            "median": float("nan"),
            "p95": float("nan"),
            "max": float("nan"),
        }
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_all(
    result_root: Path,
    levels: list[str],
) -> dict[str, dict[str, list[dict[str, str]]]]:
    data: dict[str, dict[str, list[dict[str, str]]]] = {}
    for level in levels:
        data[level] = {}
        for method, _ in METHODS:
            data[level][method] = read_trials(
                result_root / level / method / "trials.csv"
            )
    return data


def colors() -> list[str]:
    cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    return cycle[:2] if len(cycle) >= 2 else ["C0", "C1"]


def style_boxplot(boxplot: dict, color: str) -> None:
    for box in boxplot["boxes"]:
        box.set_facecolor(color)
        box.set_alpha(0.70)
    for median in boxplot["medians"]:
        median.set_linewidth(1.8)
    for flier in boxplot["fliers"]:
        flier.set_markersize(2.5)
        flier.set_alpha(0.35)


def grouped_boxplot(
    axis: plt.Axes,
    data: dict[str, dict[str, list[dict[str, str]]]],
    levels: list[str],
    labels: list[str],
    column: str,
    ylabel: str,
    success_only: bool,
    log_scale: bool = False,
) -> None:
    centers = np.arange(len(levels), dtype=float)
    palette = colors()
    for method_index, (method, _) in enumerate(METHODS):
        values = [
            finite_values(data[level][method], column, success_only)
            for level in levels
        ]
        values = [
            value if value.size else np.asarray([np.nan]) for value in values
        ]
        box = axis.boxplot(
            values,
            positions=centers + (-0.18, 0.18)[method_index],
            widths=0.30,
            patch_artist=True,
            showmeans=True,
            showfliers=True,
            manage_ticks=False,
            meanprops={
                "marker": "D",
                "markerfacecolor": "white",
                "markersize": 3.8,
            },
        )
        style_boxplot(box, palette[method_index])

    axis.set_xticks(centers)
    axis.set_xticklabels(labels)
    axis.set_xlabel("Target U/V range")
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
    axis.set_axisbelow(True)
    if log_scale:
        all_values = [
            finite_values(data[level][method], column, success_only)
            for level in levels
            for method, _ in METHODS
        ]
        positive = [value[value > 0.0] for value in all_values if value.size]
        if positive and sum(value.size for value in positive):
            axis.set_yscale("log")


def save_performance_plot(
    path: Path,
    data: dict[str, dict[str, list[dict[str, str]]]],
    levels: list[str],
    labels: list[str],
    success_only: bool,
    dpi: int,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13.0, 9.2))
    specs = (
        ("translation_error_mm", "Translation error [mm]", True),
        ("rotation_error_deg", "Rotation error [deg]", True),
        ("iterations", "Iterations", False),
        ("runtime_s", "Runtime [s]", False),
    )
    for axis, (column, ylabel, log_scale) in zip(axes.flat, specs):
        grouped_boxplot(
            axis,
            data,
            levels,
            labels,
            column,
            ylabel,
            success_only,
            log_scale,
        )
    for threshold in (1.0, 2.0):
        axes[0, 0].axhline(
            threshold,
            color="#555555",
            linestyle=(0, (4, 3)),
            linewidth=0.9,
        )
    palette = colors()
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=palette[index], alpha=0.70)
        for index in range(2)
    ]
    figure.legend(
        handles,
        [label for _, label in METHODS],
        loc="upper center",
        ncol=2,
        frameon=False,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def paired_rows(
    data: dict[str, dict[str, list[dict[str, str]]]],
    levels: list[str],
    success_only: bool,
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for level in levels:
        by_method = {
            method: {int(row["trial_index"]): row for row in data[level][method]}
            for method, _ in METHODS
        }
        common = sorted(
            set(by_method["single_plane"]) & set(by_method["three_plane"])
        )
        for index in common:
            single = by_method["single_plane"][index]
            three = by_method["three_plane"][index]
            if success_only and not (
                as_bool(single.get("success")) and as_bool(three.get("success"))
            ):
                continue
            single_t = as_float(single.get("translation_error_mm"))
            three_t = as_float(three.get("translation_error_mm"))
            single_r = as_float(single.get("rotation_error_deg"))
            three_r = as_float(three.get("rotation_error_deg"))
            if not all(map(math.isfinite, (single_t, three_t, single_r, three_r))):
                continue
            output.append(
                {
                    "target_level": level,
                    "trial_index": index,
                    "single_translation_error_mm": single_t,
                    "three_translation_error_mm": three_t,
                    "translation_advantage_mm": single_t - three_t,
                    "single_rotation_error_deg": single_r,
                    "three_rotation_error_deg": three_r,
                    "rotation_advantage_deg": single_r - three_r,
                }
            )
    return output


def save_paired_plot(
    path: Path,
    rows: list[dict[str, object]],
    levels: list[str],
    labels: list[str],
    dpi: int,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12.4, 5.2))
    for axis, column, ylabel in (
        (axes[0], "translation_advantage_mm", "Single − Three [mm]"),
        (axes[1], "rotation_advantage_deg", "Single − Three [deg]"),
    ):
        values = [
            np.asarray(
                [float(row[column]) for row in rows if row["target_level"] == level],
                dtype=float,
            )
            for level in levels
        ]
        shown = [value if value.size else np.asarray([np.nan]) for value in values]
        box = axis.boxplot(
            shown,
            patch_artist=True,
            showmeans=True,
            showfliers=True,
            meanprops={"marker": "D", "markerfacecolor": "white", "markersize": 4},
        )
        style_boxplot(box, "C2")
        axis.axhline(0.0, color="#444444", linestyle="--", linewidth=1.0)
        axis.set_xticklabels(
            [f"{label}\n(n={value.size})" for label, value in zip(labels, values)]
        )
        axis.set_xlabel("Target U/V range")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
        axis.set_axisbelow(True)
    figure.suptitle("Positive values mean lower error for three-plane calibration")
    figure.tight_layout()
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def target_effect_rows(
    data: dict[str, dict[str, list[dict[str, str]]]],
    levels: list[str],
    success_only: bool,
) -> list[dict[str, object]]:
    """Pair every non-baseline range with the first (fixed) range."""
    baseline_level = levels[0]
    output: list[dict[str, object]] = []
    for method, method_label in METHODS:
        baseline = {
            int(row["trial_index"]): row for row in data[baseline_level][method]
        }
        for level in levels[1:]:
            current = {
                int(row["trial_index"]): row for row in data[level][method]
            }
            for index in sorted(set(baseline) & set(current)):
                base_row = baseline[index]
                current_row = current[index]
                if success_only and not (
                    as_bool(base_row.get("success"))
                    and as_bool(current_row.get("success"))
                ):
                    continue
                base_t = as_float(base_row.get("translation_error_mm"))
                current_t = as_float(current_row.get("translation_error_mm"))
                base_r = as_float(base_row.get("rotation_error_deg"))
                current_r = as_float(current_row.get("rotation_error_deg"))
                if not all(map(math.isfinite, (base_t, current_t, base_r, current_r))):
                    continue
                output.append(
                    {
                        "baseline_level": baseline_level,
                        "target_level": level,
                        "method": method,
                        "method_label": method_label,
                        "trial_index": index,
                        "baseline_translation_error_mm": base_t,
                        "target_translation_error_mm": current_t,
                        "translation_advantage_vs_fixed_mm": base_t - current_t,
                        "baseline_rotation_error_deg": base_r,
                        "target_rotation_error_deg": current_r,
                        "rotation_advantage_vs_fixed_deg": base_r - current_r,
                    }
                )
    return output


def save_target_effect_plot(
    path: Path,
    rows: list[dict[str, object]],
    levels: list[str],
    labels: list[str],
    dpi: int,
) -> None:
    compared_levels = levels[1:]
    compared_labels = labels[1:]
    figure, axes = plt.subplots(1, 2, figsize=(12.4, 5.2))
    centers = np.arange(len(compared_levels), dtype=float)
    palette = colors()

    for axis, column, ylabel in (
        (
            axes[0],
            "translation_advantage_vs_fixed_mm",
            "Fixed − relaxed range [mm]",
        ),
        (
            axes[1],
            "rotation_advantage_vs_fixed_deg",
            "Fixed − relaxed range [deg]",
        ),
    ):
        for method_index, (method, _) in enumerate(METHODS):
            values = [
                np.asarray(
                    [
                        float(row[column])
                        for row in rows
                        if row["target_level"] == level
                        and row["method"] == method
                    ],
                    dtype=float,
                )
                for level in compared_levels
            ]
            shown = [
                value if value.size else np.asarray([np.nan]) for value in values
            ]
            box = axis.boxplot(
                shown,
                positions=centers + (-0.18, 0.18)[method_index],
                widths=0.30,
                patch_artist=True,
                showmeans=True,
                showfliers=True,
                manage_ticks=False,
                meanprops={
                    "marker": "D",
                    "markerfacecolor": "white",
                    "markersize": 4,
                },
            )
            style_boxplot(box, palette[method_index])
        axis.axhline(0.0, color="#444444", linestyle="--", linewidth=1.0)
        axis.set_xticks(centers)
        axis.set_xticklabels(compared_labels)
        axis.set_xlabel("Target U/V range")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
        axis.set_axisbelow(True)

    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=palette[index], alpha=0.70)
        for index in range(2)
    ]
    figure.legend(
        handles,
        [label for _, label in METHODS],
        loc="upper center",
        ncol=2,
        frameon=False,
    )
    figure.suptitle(
        "Positive values mean lower error than the fixed target condition",
        y=0.94,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def rate_rows(
    data: dict[str, dict[str, list[dict[str, str]]]],
    levels: list[str],
    thresholds: list[float],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    general: list[dict[str, object]] = []
    outliers: list[dict[str, object]] = []
    for level in levels:
        for method, method_label in METHODS:
            rows = data[level][method]
            total = len(rows)
            converged = sum(as_bool(row.get("converged")) for row in rows)
            successful = sum(as_bool(row.get("success")) for row in rows)
            general.append(
                {
                    "target_level": level,
                    "method": method,
                    "method_label": method_label,
                    "recorded_trials": total,
                    "converged_count": converged,
                    "convergence_rate": converged / total,
                    "success_count": successful,
                    "success_rate": successful / total,
                }
            )
            errors = np.asarray(
                [as_float(row.get("translation_error_mm")) for row in rows],
                dtype=float,
            )
            exceptions = np.asarray(
                [bool(str(row.get("error_type", "")).strip()) for row in rows],
                dtype=bool,
            )
            invalid = ~np.isfinite(errors) | exceptions
            for threshold in thresholds:
                exceeded = ~invalid & (errors > threshold)
                adjusted = invalid | exceeded
                finite_count = int((~invalid).sum())
                outliers.append(
                    {
                        "target_level": level,
                        "method": method,
                        "method_label": method_label,
                        "threshold_mm": threshold,
                        "recorded_trials": total,
                        "finite_error_trials": finite_count,
                        "threshold_exceedance_count": int(exceeded.sum()),
                        "invalid_error_count": int(invalid.sum()),
                        "outlier_count": int(adjusted.sum()),
                        "outlier_rate": float(adjusted.sum() / total),
                        "finite_only_exceedance_rate": (
                            float(exceeded.sum() / finite_count)
                            if finite_count
                            else float("nan")
                        ),
                    }
                )
    return general, outliers


def save_grouped_rate_plot(
    path: Path,
    rows: list[dict[str, object]],
    levels: list[str],
    labels: list[str],
    metrics: list[tuple[str, str, str]],
    dpi: int,
) -> None:
    figure, axes = plt.subplots(1, len(metrics), figsize=(6.2 * len(metrics), 5.2))
    axes_array = np.atleast_1d(axes)
    centers = np.arange(len(levels), dtype=float)
    width = 0.34
    palette = colors()
    for axis, (filter_column, filter_value, ylabel) in zip(axes_array, metrics):
        for method_index, (method, method_label) in enumerate(METHODS):
            values = []
            for level in levels:
                match = [
                    row
                    for row in rows
                    if row["target_level"] == level
                    and row["method"] == method
                    and (
                        filter_column != "threshold_mm"
                        or str(row.get(filter_column)) == filter_value
                    )
                ]
                if len(match) != 1:
                    raise RuntimeError(
                        f"Expected one rate row: level={level}, method={method}, "
                        f"{filter_column}={filter_value}"
                    )
                metric_column = (
                    "outlier_rate"
                    if filter_column == "threshold_mm"
                    else filter_column
                )
                values.append(100.0 * float(match[0][metric_column]))
            bars = axis.bar(
                centers + (-width / 2, width / 2)[method_index],
                values,
                width=width,
                color=palette[method_index],
                label=method_label,
            )
            for bar, value in zip(bars, values):
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 1.0,
                    f"{value:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
        axis.set_xticks(centers)
        axis.set_xticklabels(labels)
        axis.set_xlabel("Target U/V range")
        axis.set_ylabel(ylabel)
        axis.set_ylim(0.0, 108.0)
        axis.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
        axis.set_axisbelow(True)
    handles, legend_labels = axes_array[0].get_legend_handles_labels()
    figure.legend(
        handles,
        legend_labels,
        loc="upper center",
        ncol=2,
        frameon=False,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    figure.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def error_summary_rows(
    data: dict[str, dict[str, list[dict[str, str]]]],
    levels: list[str],
    success_only: bool,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for level in levels:
        for method, method_label in METHODS:
            trial_rows = data[level][method]
            translation = finite_values(
                trial_rows, "translation_error_mm", success_only
            )
            rotation = finite_values(trial_rows, "rotation_error_deg", success_only)
            iterations = finite_values(trial_rows, "iterations", success_only)
            runtime = finite_values(trial_rows, "runtime_s", success_only)
            trans_stats = summary(translation)
            rot_stats = summary(rotation)
            rows.append(
                {
                    "target_level": level,
                    "method": method,
                    "method_label": method_label,
                    "finite_error_trials": int(translation.size),
                    "translation_mean_mm": trans_stats["mean"],
                    "translation_median_mm": trans_stats["median"],
                    "translation_p95_mm": trans_stats["p95"],
                    "translation_max_mm": trans_stats["max"],
                    "rotation_mean_deg": rot_stats["mean"],
                    "rotation_median_deg": rot_stats["median"],
                    "rotation_p95_deg": rot_stats["p95"],
                    "rotation_max_deg": rot_stats["max"],
                    "iterations_median": summary(iterations)["median"],
                    "runtime_median_s": summary(runtime)["median"],
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    if len(args.levels) < 2:
        raise SystemExit(
            "--levels requires a fixed baseline and at least one comparison"
        )
    if args.level_labels is not None and len(args.level_labels) != len(args.levels):
        raise SystemExit("--level-labels must have the same length as --levels")
    if any(value <= 0.0 for value in args.outlier_thresholds_mm):
        raise SystemExit("--outlier-thresholds-mm values must be positive")

    result_root = args.result_root.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else result_root / "plots"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    labels = args.level_labels or args.levels
    thresholds = list(args.outlier_thresholds_mm)
    data = load_all(result_root, args.levels)

    paired = paired_rows(data, args.levels, args.success_only)
    target_effects = target_effect_rows(data, args.levels, args.success_only)
    general_rates, outlier_rates = rate_rows(data, args.levels, thresholds)
    errors = error_summary_rows(data, args.levels, args.success_only)

    performance_path = output_dir / "target_range_performance_boxplots.png"
    paired_path = output_dir / "target_range_paired_differences.png"
    target_effect_path = output_dir / "target_range_vs_fixed_differences.png"
    rates_path = output_dir / "target_range_success_convergence_rates.png"
    outlier_path = output_dir / "target_range_translation_outlier_rates.png"

    save_performance_plot(
        performance_path,
        data,
        args.levels,
        labels,
        args.success_only,
        args.dpi,
    )
    save_paired_plot(paired_path, paired, args.levels, labels, args.dpi)
    save_target_effect_plot(
        target_effect_path,
        target_effects,
        args.levels,
        labels,
        args.dpi,
    )
    save_grouped_rate_plot(
        rates_path,
        general_rates,
        args.levels,
        labels,
        [
            ("convergence_rate", "convergence_rate", "Convergence rate [%]"),
            ("success_rate", "success_rate", "Success rate [%]"),
        ],
        args.dpi,
    )
    save_grouped_rate_plot(
        outlier_path,
        outlier_rates,
        args.levels,
        labels,
        [
            ("threshold_mm", str(value), f"Translation error > {value:g} mm [%]")
            for value in thresholds
        ],
        args.dpi,
    )

    write_csv(output_dir / "target_range_error_summary.csv", errors)
    write_csv(output_dir / "target_range_paired_differences.csv", paired)
    write_csv(
        output_dir / "target_range_vs_fixed_differences.csv",
        target_effects,
    )
    write_csv(output_dir / "target_range_rates.csv", general_rates)
    write_csv(output_dir / "target_range_outlier_rates.csv", outlier_rates)

    print(f"Saved performance plot : {performance_path}")
    print(f"Saved paired plot      : {paired_path}")
    print(f"Saved vs-fixed plot    : {target_effect_path}")
    print(f"Saved rates plot       : {rates_path}")
    print(f"Saved outlier plot     : {outlier_path}")
    print(
        "Outlier definition    : translation error above threshold; "
        "failed/non-finite trials are included as outliers"
    )


if __name__ == "__main__":
    main()
