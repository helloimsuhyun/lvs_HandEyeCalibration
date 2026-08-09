#!/usr/bin/env python3
"""Plot median-error changes for the pose-parameter OAT ablation."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


METHODS = (
    ("single_plane", "Single-plane", "-", "o"),
    ("three_plane", "Three-plane", "--", "s"),
)

PARAMETERS = (
    ("Tilt", "tilt_mild", "tilt_strong", "#0072B2"),
    ("Azimuth", "azimuth_mild", "azimuth_strong", "#D55E00"),
    ("Roll", "roll_mild", "roll_strong", "#009E73"),
)

METRICS = (
    ("translation_error_mm", "Translation error change [%]", "(a)"),
    ("rotation_error_deg", "Rotation error change [%]", "(b)"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot median translation/rotation error changes relative to the "
            "moderate pose-diversity baseline."
        )
    )
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--csv-output", type=Path, default=None)
    parser.add_argument("--baseline", default="moderate")
    parser.add_argument(
        "--span-reductions-percent",
        nargs=2,
        type=float,
        default=(50.0, 200.0 / 3.0),
        metavar=("MILD", "STRONG"),
    )
    parser.add_argument("--success-only", action="store_true")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def parse_bool(value: object) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y", "t"}


def load_metric(
    path: Path,
    metric: str,
    success_only: bool,
) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"trials.csv not found: {path}")

    values: list[float] = []
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or metric not in reader.fieldnames:
            raise KeyError(f"{metric} column is missing: {path}")
        for row in reader:
            if str(row.get("error_type", "")).strip():
                continue
            if success_only and not parse_bool(row.get("success", False)):
                continue
            try:
                value = float(row[metric])
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                values.append(value)

    if not values:
        raise RuntimeError(f"No finite {metric} values found: {path}")
    return np.asarray(values, dtype=float)


def collect_rows(args: argparse.Namespace) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    result_root = args.result_root.expanduser().resolve()
    mild_reduction, strong_reduction = args.span_reductions_percent

    for method, method_label, _, _ in METHODS:
        for metric, _, _ in METRICS:
            baseline_values = load_metric(
                result_root / args.baseline / method / "trials.csv",
                metric,
                args.success_only,
            )
            baseline_median = float(np.median(baseline_values))
            if baseline_median <= 0.0:
                raise RuntimeError(
                    f"Baseline median must be positive: method={method}, metric={metric}"
                )

            for parameter, mild_level, strong_level, _ in PARAMETERS:
                for level, reduction in (
                    (args.baseline, 0.0),
                    (mild_level, mild_reduction),
                    (strong_level, strong_reduction),
                ):
                    if level == args.baseline:
                        condition_values = baseline_values
                    else:
                        condition_values = load_metric(
                            result_root / level / method / "trials.csv",
                            metric,
                            args.success_only,
                        )
                    condition_median = float(np.median(condition_values))
                    percent_change = (
                        100.0 * (condition_median - baseline_median) / baseline_median
                    )
                    rows.append(
                        {
                            "method": method,
                            "method_label": method_label,
                            "metric": metric,
                            "parameter": parameter,
                            "pose_level": level,
                            "span_reduction_percent": reduction,
                            "finite_trials": int(condition_values.size),
                            "baseline_median": baseline_median,
                            "condition_median": condition_median,
                            "median_error_change_percent": percent_change,
                        }
                    )
    return rows


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def matching_values(
    rows: list[dict[str, object]],
    *,
    method: str,
    metric: str,
    parameter: str,
) -> tuple[np.ndarray, np.ndarray]:
    selected = [
        row
        for row in rows
        if row["method"] == method
        and row["metric"] == metric
        and row["parameter"] == parameter
    ]
    selected.sort(key=lambda row: float(row["span_reduction_percent"]))
    return (
        np.asarray(
            [float(row["span_reduction_percent"]) for row in selected],
            dtype=float,
        ),
        np.asarray(
            [float(row["median_error_change_percent"]) for row in selected],
            dtype=float,
        ),
    )


def plot_rows(
    output: Path,
    rows: list[dict[str, object]],
    reductions: tuple[float, float],
    dpi: int,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13.2, 5.6))

    for axis, (metric, ylabel, panel_label) in zip(axes, METRICS):
        all_y_values: list[float] = []
        for parameter, _, _, color in PARAMETERS:
            for method, _, linestyle, marker in METHODS:
                x_values, y_values = matching_values(
                    rows,
                    method=method,
                    metric=metric,
                    parameter=parameter,
                )
                all_y_values.extend(y_values.tolist())
                axis.plot(
                    x_values,
                    y_values,
                    color=color,
                    linestyle=linestyle,
                    linewidth=2.0,
                    marker=marker,
                    markersize=6.0,
                    markerfacecolor="white",
                    markeredgewidth=1.6,
                )

        axis.axhline(0.0, color="#555555", linestyle=":", linewidth=1.2)
        axis.set_xticks([0.0, reductions[0], reductions[1]])
        axis.set_xticklabels(["0\nBaseline", "50", "66.7"])
        axis.set_xlabel("Pose span reduction from moderate baseline [%]")
        axis.set_ylabel(ylabel)
        axis.grid(axis="both", linestyle="--", linewidth=0.7, alpha=0.35)
        axis.set_axisbelow(True)
        axis.text(
            0.02,
            0.97,
            panel_label,
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=12,
            fontweight="bold",
        )

        minimum = min(all_y_values + [0.0])
        maximum = max(all_y_values + [0.0])
        padding = max(3.0, 0.10 * (maximum - minimum))
        axis.set_ylim(minimum - padding, maximum + padding)

    parameter_handles = [
        Line2D([0], [0], color=color, linewidth=2.5, label=parameter)
        for parameter, _, _, color in PARAMETERS
    ]
    method_handles = [
        Line2D(
            [0],
            [0],
            color="#333333",
            linestyle=linestyle,
            marker=marker,
            markerfacecolor="white",
            linewidth=2.0,
            label=method_label,
        )
        for _, method_label, linestyle, marker in METHODS
    ]
    figure.legend(
        parameter_handles + method_handles,
        [handle.get_label() for handle in parameter_handles + method_handles],
        loc="upper center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    result_root = args.result_root.expanduser().resolve()
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else result_root / "plots" / "pose_parameter_ablation_median_error_change.png"
    )
    csv_output = (
        args.csv_output.expanduser().resolve()
        if args.csv_output is not None
        else result_root / "plots" / "pose_parameter_ablation_median_error_change.csv"
    )

    rows = collect_rows(args)
    reductions = tuple(map(float, args.span_reductions_percent))
    plot_rows(output, rows, reductions, args.dpi)
    write_rows(csv_output, rows)

    print(f"Saved median-change plot: {output}")
    print(f"Saved median-change CSV : {csv_output}")


if __name__ == "__main__":
    main()
