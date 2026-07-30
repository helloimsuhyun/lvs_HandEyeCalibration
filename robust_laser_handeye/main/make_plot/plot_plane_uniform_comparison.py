#!/usr/bin/env python3
"""Plot single-uniform, three-random, and three-uniform calibration results."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np


METHODS = (
    ("single_uniform", "Single\nUniform", "#4C78A8"),
    ("three_random", "Three\nRandom", "#F58518"),
    ("three_uniform", "Three Uniform\nFull set/plane", "#54A24B"),
)
COMPARISONS = (
    (
        "single_uniform_vs_three_random",
        "single_uniform",
        "three_random",
        "Single uniform\nvs Three random",
    ),
    (
        "single_uniform_vs_three_uniform",
        "single_uniform",
        "three_uniform",
        "Single uniform\nvs Three uniform full set/plane",
    ),
)
THREE_LEVEL_METHODS = (
    ("single_uniform", "Single\nUniform", "#4C78A8"),
    ("three_hard", "Three\nRestricted", "#E45756"),
    ("three_moderate", "Three\nModerate", "#F58518"),
    ("three_easy", "Three\nWide", "#54A24B"),
)
THREE_LEVEL_COMPARISONS = (
    (
        "single_uniform_vs_three_hard",
        "single_uniform",
        "three_hard",
        "Single uniform\nvs Three restricted",
    ),
    (
        "single_uniform_vs_three_moderate",
        "single_uniform",
        "three_moderate",
        "Single uniform\nvs Three moderate",
    ),
    (
        "single_uniform_vs_three_easy",
        "single_uniform",
        "three_easy",
        "Single uniform\nvs Three wide",
    ),
)
SINGLE_UNIFORM_FISHER_METHODS = (
    ("single_uniform", "Single\nUniform maximin", "#4C78A8"),
    ("single_fisher", "Single\nActive Fisher", "#B279A2"),
)
SINGLE_UNIFORM_FISHER_COMPARISONS = (
    (
        "single_uniform_vs_single_fisher",
        "single_uniform",
        "single_fisher",
        "Single uniform\nvs Single Fisher",
    ),
)
DESIGN_TITLE = "Plane-relative uniform pose comparison"
NUMERIC_FIELDS = (
    "trial_index",
    "translation_error_mm",
    "rotation_error_deg",
    "iterations",
    "condition_last",
    "n_scans",
    "n_points",
)
Rows = list[dict[str, Any]]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument(
        "--design",
        choices=(
            "uniform_fullset",
            "three_pose_levels",
            "single_uniform_vs_fisher",
        ),
        default="uniform_fullset",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--success-only",
        action="store_true",
        help="Use successful trials only for error/iteration distributions.",
    )
    parser.add_argument(
        "--hide-outliers",
        action="store_true",
        help="Hide boxplot fliers and mean markers.",
    )
    parser.add_argument(
        "--log-errors",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--log-iterations",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args(argv)


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y", "t"}


def _load_trials(path: Path) -> Rows:
    if not path.is_file():
        raise FileNotFoundError(f"trials.csv not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise KeyError(f"trials.csv has no header: {path}")
        required = {
            "trial_index",
            "translation_error_mm",
            "rotation_error_deg",
        }
        missing = required.difference(reader.fieldnames)
        if missing:
            raise KeyError(f"{path} is missing columns: {sorted(missing)}")
        rows: Rows = []
        for raw in reader:
            trial_index = _as_float(raw.get("trial_index"))
            if not np.isfinite(trial_index):
                continue
            row: dict[str, Any] = dict(raw)
            row["trial_index"] = int(trial_index)
            for field in NUMERIC_FIELDS[1:]:
                row[field] = _as_float(row.get(field))
            rows.append(row)
    ids = [int(row["trial_index"]) for row in rows]
    if not rows or len(ids) != len(set(ids)):
        raise RuntimeError(f"empty or duplicate trial_index rows: {path}")
    return rows


def _error_rows(rows: Rows, success_only: bool) -> Rows:
    output = []
    for row in rows:
        if str(row.get("error_type", "")).strip():
            continue
        if not (
            np.isfinite(_as_float(row.get("translation_error_mm")))
            and np.isfinite(_as_float(row.get("rotation_error_deg")))
        ):
            continue
        if success_only and not _as_bool(row.get("success", False)):
            continue
        output.append(row)
    return output


def _values(rows: Rows, field: str, *, positive: bool = False) -> np.ndarray:
    values = np.asarray([_as_float(row.get(field)) for row in rows], dtype=float)
    values = values[np.isfinite(values)]
    return values[values > 0.0] if positive else values


def _style_boxplot(artists: dict[str, list[Any]], color: str) -> None:
    for box in artists["boxes"]:
        box.set_facecolor(color)
        box.set_alpha(0.72)
        box.set_edgecolor("#303030")
    for median in artists["medians"]:
        median.set_color("#111111")
        median.set_linewidth(1.8)
    for item in artists["whiskers"] + artists["caps"]:
        item.set_color("#555555")
    for flier in artists["fliers"]:
        flier.set(
            marker="o",
            markersize=3.0,
            markerfacecolor="none",
            markeredgecolor="#555555",
            markeredgewidth=0.7,
            alpha=0.45,
        )


def _boxplot(
    axis: plt.Axes,
    values: list[np.ndarray],
    labels: list[str],
    colors: list[str],
    hide_outliers: bool,
) -> None:
    artists = axis.boxplot(
        values,
        labels=labels,
        patch_artist=True,
        showmeans=not hide_outliers,
        showfliers=not hide_outliers,
        meanprops={
            "marker": "D",
            "markerfacecolor": "white",
            "markeredgecolor": "#111111",
            "markersize": 3.8,
        },
    )
    for index, box in enumerate(artists["boxes"]):
        box.set_facecolor(colors[index])
        box.set_alpha(0.72)
        box.set_edgecolor("#303030")
    for median in artists["medians"]:
        median.set_color("#111111")
        median.set_linewidth(1.8)
    for item in artists["whiskers"] + artists["caps"]:
        item.set_color("#555555")
    for flier in artists["fliers"]:
        flier.set(
            marker="o",
            markersize=3.0,
            markerfacecolor="none",
            markeredgecolor="#555555",
            markeredgewidth=0.7,
            alpha=0.45,
        )


def _save_final_errors(
    frames: dict[str, Rows],
    output: Path,
    args: argparse.Namespace,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11.6, 4.8))
    specs = (
        ("translation_error_mm", "Translation error [mm]", "(a)"),
        ("rotation_error_deg", "Rotation error [deg]", "(b)"),
    )
    labels = [label for _key, label, _color in METHODS]
    colors = [color for _key, _label, color in METHODS]
    for axis, (field, ylabel, panel) in zip(axes, specs, strict=True):
        values = [
            _values(
                _error_rows(frames[key], args.success_only),
                field,
                positive=args.log_errors,
            )
            for key, _label, _color in METHODS
        ]
        _boxplot(axis, values, labels, colors, args.hide_outliers)
        if args.log_errors:
            axis.set_yscale("log")
            ylabel += " (log scale)"
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", linestyle="--", alpha=0.3)
        axis.set_axisbelow(True)
        axis.text(
            0.01, 0.98, panel,
            transform=axis.transAxes,
            ha="left", va="top",
            fontweight="bold",
        )
    suffix = " (successful trials)" if args.success_only else ""
    figure.suptitle(f"{DESIGN_TITLE}{suffix}")
    figure.tight_layout()
    figure.savefig(output, dpi=args.dpi, bbox_inches="tight")
    plt.close(figure)


def _paired_rows(
    frames: dict[str, Rows],
    success_only: bool,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for comparison, left_key, right_key, _label in COMPARISONS:
        left = {
            int(row["trial_index"]): row
            for row in _error_rows(frames[left_key], success_only)
        }
        right = {
            int(row["trial_index"]): row
            for row in _error_rows(frames[right_key], success_only)
        }
        for trial_index in sorted(left.keys() & right.keys()):
            left_row = left[trial_index]
            right_row = right[trial_index]
            output.append(
                {
                    "comparison": comparison,
                    "trial_index": trial_index,
                    "left_method": left_key,
                    "right_method": right_key,
                    "translation_error_mm_left": _as_float(
                        left_row["translation_error_mm"]
                    ),
                    "translation_error_mm_right": _as_float(
                        right_row["translation_error_mm"]
                    ),
                    "rotation_error_deg_left": _as_float(
                        left_row["rotation_error_deg"]
                    ),
                    "rotation_error_deg_right": _as_float(
                        right_row["rotation_error_deg"]
                    ),
                    "translation_advantage_mm": (
                        _as_float(left_row["translation_error_mm"])
                        - _as_float(right_row["translation_error_mm"])
                    ),
                    "rotation_advantage_deg": (
                        _as_float(left_row["rotation_error_deg"])
                        - _as_float(right_row["rotation_error_deg"])
                    ),
                }
            )
    return output


def _save_paired(
    paired: list[dict[str, Any]],
    output: Path,
    args: argparse.Namespace,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11.6, 4.8))
    labels = [entry[3] for entry in COMPARISONS]
    method_colors = {key: color for key, _label, color in METHODS}
    colors = [
        method_colors[right_key]
        for _comparison, _left_key, right_key, _label in COMPARISONS
    ]
    specs = (
        (
            "translation_advantage_mm",
            "Paired advantage [mm]\nleft method − comparison method",
            "(a)",
        ),
        (
            "rotation_advantage_deg",
            "Paired advantage [deg]\nleft method − comparison method",
            "(b)",
        ),
    )
    for axis, (field, ylabel, panel) in zip(axes, specs, strict=True):
        values = [
            _values(
                [
                    row
                    for row in paired
                    if row["comparison"] == comparison
                ],
                field,
            )
            for comparison, *_unused in COMPARISONS
        ]
        _boxplot(axis, values, labels, colors, args.hide_outliers)
        if not args.hide_outliers:
            nonempty = [value for value in values if value.size]
            if nonempty:
                finite = np.concatenate(nonempty)
                nonzero = np.abs(finite[finite != 0.0])
            else:
                finite = np.asarray([], dtype=float)
                nonzero = np.asarray([], dtype=float)
            if finite.size and nonzero.size:
                axis.set_yscale(
                    "symlog",
                    linthresh=max(
                        float(np.percentile(nonzero, 75)),
                        np.finfo(float).tiny,
                    ),
                )
                limit = 1.15 * float(np.max(np.abs(finite)))
                axis.set_ylim(-limit, limit)
        axis.axhline(0.0, color="#202020", linestyle="--", linewidth=1.1)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", linestyle="--", alpha=0.3)
        axis.set_axisbelow(True)
        axis.text(
            0.01, 0.98, panel,
            transform=axis.transAxes,
            ha="left", va="top",
            fontweight="bold",
        )
    scale_note = (
        " (symmetric-log y-axis)"
        if not args.hide_outliers
        else ""
    )
    figure.suptitle(
        "Positive values mean the comparison method has lower error"
        + scale_note
    )
    figure.tight_layout()
    figure.savefig(output, dpi=args.dpi, bbox_inches="tight")
    plt.close(figure)


def _save_grouped_metric(
    frames: dict[str, Rows],
    output: Path,
    args: argparse.Namespace,
    *,
    field: str,
    ylabel: str,
    log_scale: bool,
) -> None:
    figure, axis = plt.subplots(figsize=(7.4, 4.8))
    values = [
        _values(
            _error_rows(frames[key], args.success_only),
            field,
            positive=log_scale,
        )
        for key, _label, _color in METHODS
    ]
    _boxplot(
        axis,
        values,
        [label for _key, label, _color in METHODS],
        [color for _key, _label, color in METHODS],
        args.hide_outliers,
    )
    if log_scale:
        axis.set_yscale("log")
        ylabel += " (log scale)"
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", linestyle="--", alpha=0.3)
    axis.set_axisbelow(True)
    figure.tight_layout()
    figure.savefig(output, dpi=args.dpi, bbox_inches="tight")
    plt.close(figure)


def _rate(rows: Rows, field: str) -> float:
    return float(np.mean([_as_bool(row.get(field, False)) for row in rows]))


def _save_rates(
    frames: dict[str, Rows],
    output: Path,
    dpi: int,
) -> None:
    x = np.arange(len(METHODS), dtype=float)
    width = 0.25
    figure, axis = plt.subplots(figsize=(8.2, 4.8))
    for index, (field, label, hatch) in enumerate(
        (
            ("converged", "Convergence", ""),
            ("success", "Success", "//"),
            ("outlier", "Outlier", "xx"),
        )
    ):
        values = [
            100.0 * _rate(frames[key], field)
            for key, _method_label, _color in METHODS
        ]
        axis.bar(
            x + (index - 1) * width,
            values,
            width,
            label=label,
            color=[color for _key, _label, color in METHODS],
            alpha=0.85 if index == 0 else 0.55,
            hatch=hatch,
        )
    axis.set_xticks(x, [label for _key, label, _color in METHODS])
    axis.set_ylim(0.0, 105.0)
    axis.set_ylabel("Rate [%]")
    axis.grid(axis="y", linestyle="--", alpha=0.3)
    axis.set_axisbelow(True)
    axis.legend(ncol=3, frameon=False)
    figure.tight_layout()
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def _save_data_volume(
    frames: dict[str, Rows],
    output: Path,
    dpi: int,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10.8, 4.6))
    labels = [label for _key, label, _color in METHODS]
    colors = [color for _key, _label, color in METHODS]
    for axis, field, ylabel in (
        (axes[0], "n_scans", "Scans per trial"),
        (axes[1], "n_points", "Profile points per trial"),
    ):
        values = [
            float(np.median(_values(frames[key], field)))
            for key, _label, _color in METHODS
        ]
        bars = axis.bar(labels, values, color=colors, alpha=0.8)
        axis.bar_label(bars, fmt="%.0f", padding=3)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", linestyle="--", alpha=0.3)
        axis.set_axisbelow(True)
    figure.suptitle("Data volume used by each calibration method")
    figure.tight_layout()
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _summaries(
    frames: dict[str, Rows],
    paired: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    def quartiles(values: np.ndarray) -> tuple[float, float, float]:
        if not values.size:
            nan = float("nan")
            return nan, nan, nan
        return (
            float(np.median(values)),
            float(np.percentile(values, 25)),
            float(np.percentile(values, 75)),
        )

    method_rows = []
    for key, label, _color in METHODS:
        errors = _error_rows(frames[key], success_only=False)
        row: dict[str, Any] = {
            "method": key,
            "label": label.replace("\n", " "),
            "trials": len(frames[key]),
            "convergence_rate": _rate(frames[key], "converged"),
            "success_rate": _rate(frames[key], "success"),
            "outlier_rate": _rate(frames[key], "outlier"),
            "scan_count_median": float(
                np.median(_values(frames[key], "n_scans"))
            ),
            "point_count_median": float(
                np.median(_values(frames[key], "n_points"))
            ),
        }
        for metric, field in (
            ("translation_mm", "translation_error_mm"),
            ("rotation_deg", "rotation_error_deg"),
        ):
            values = _values(errors, field)
            median, q25, q75 = quartiles(values)
            row[f"{metric}_median"] = median
            row[f"{metric}_q25"] = q25
            row[f"{metric}_q75"] = q75
        method_rows.append(row)

    paired_rows = []
    for comparison, *_unused in COMPARISONS:
        selected = [
            row for row in paired if row["comparison"] == comparison
        ]
        row = {"comparison": comparison, "paired_trials": len(selected)}
        for metric, field in (
            ("translation_mm", "translation_advantage_mm"),
            ("rotation_deg", "rotation_advantage_deg"),
        ):
            values = _values(selected, field)
            median, q25, q75 = quartiles(values)
            row[f"{metric}_advantage_median"] = median
            row[f"{metric}_advantage_q25"] = q25
            row[f"{metric}_advantage_q75"] = q75
            row[f"{metric}_three_better_fraction"] = (
                float(np.mean(values > 0.0))
                if values.size
                else float("nan")
            )
        paired_rows.append(row)
    return method_rows, paired_rows


def main(argv: Sequence[str] | None = None) -> int:
    global METHODS, COMPARISONS, DESIGN_TITLE
    args = parse_args(argv)
    output_prefix = "plane_uniform"
    if args.design == "three_pose_levels":
        METHODS = THREE_LEVEL_METHODS
        COMPARISONS = THREE_LEVEL_COMPARISONS
        DESIGN_TITLE = "Single uniform vs three-plane pose-diversity levels"
        output_prefix = "pose_diversity"
    elif args.design == "single_uniform_vs_fisher":
        METHODS = SINGLE_UNIFORM_FISHER_METHODS
        COMPARISONS = SINGLE_UNIFORM_FISHER_COMPARISONS
        DESIGN_TITLE = "Single-plane pose-selection policy comparison"
        output_prefix = "pose_diversity"
    result_root = args.result_root.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else result_root / "plots"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = {
        key: _load_trials(result_root / key / "trials.csv")
        for key, _label, _color in METHODS
    }
    paired = _paired_rows(frames, args.success_only)

    _save_final_errors(
        frames,
        output_dir / f"{output_prefix}_final_error_boxplots.png",
        args,
    )
    _save_paired(
        paired,
        output_dir / f"{output_prefix}_paired_differences.png",
        args,
    )
    _save_grouped_metric(
        frames,
        output_dir / f"{output_prefix}_iterations.png",
        args,
        field="iterations",
        ylabel="Iterations",
        log_scale=args.log_iterations,
    )
    _save_grouped_metric(
        frames,
        output_dir / f"{output_prefix}_condition_number.png",
        args,
        field="condition_last",
        ylabel="Final condition number",
        log_scale=True,
    )
    _save_rates(
        frames,
        output_dir / f"{output_prefix}_rates.png",
        args.dpi,
    )
    _save_data_volume(
        frames,
        output_dir / f"{output_prefix}_data_volume.png",
        args.dpi,
    )

    method_summary, paired_summary = _summaries(frames, paired)
    _write_csv(output_dir / f"{output_prefix}_paired_rows.csv", paired)
    _write_csv(
        output_dir / f"{output_prefix}_method_error_summary.csv",
        method_summary,
    )
    _write_csv(
        output_dir / f"{output_prefix}_paired_summary.csv",
        paired_summary,
    )
    print(f"Saved {DESIGN_TITLE.lower()} plots: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
