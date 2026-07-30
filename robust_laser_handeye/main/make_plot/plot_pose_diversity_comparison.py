#!/usr/bin/env python3
"""
PYTHONPATH=. python3 main/make_plot/plot_pose_diversity_comparison.py \
  --result-root results/fair_plane_global_pose_diversity
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHODS = (
    ("single_plane", "Single-plane"),
    ("three_plane", "Three-plane"),
)

DEFAULT_ORDER = ("restricted", "moderate", "wide")

DEFAULT_RESULT_ROOT = Path(
    "results/fair_plane_global_pose_diversity"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze pose-diversity comparison results.\n"
            "Expected structure:\n"
            "  RESULT_ROOT/restricted/single_plane/trials.csv\n"
            "  RESULT_ROOT/restricted/three_plane/trials.csv\n"
            "  RESULT_ROOT/moderate/...\n"
            "  RESULT_ROOT/wide/..."
        )
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=DEFAULT_RESULT_ROOT,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <result-root>/plots or plots_success_only.",
    )
    parser.add_argument(
        "--levels",
        nargs="+",
        default=None,
        help=(
            "Pose-diversity levels in plotting order. "
            "When omitted, complete level directories are auto-discovered "
            "with restricted/moderate/wide first. "
            "Example: --levels restricted moderate wide"
        ),
    )
    parser.add_argument(
        "--success-only",
        action="store_true",
        help=(
            "Use only successful trials for final-error, paired-error, "
            "iteration, and condition-number plots. Rate and paired-success "
            "statistics always use all recorded trials."
        ),
    )
    parser.add_argument(
        "--hide-outliers",
        action="store_true",
        help="Hide individual boxplot fliers.",
    )
    parser.add_argument(
        "--log-translation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use logarithmic scale for translation error "
            "(default: enabled; disable with --no-log-translation)."
        ),
    )
    parser.add_argument(
        "--log-rotation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use logarithmic scale for rotation error "
            "(default: enabled; disable with --no-log-rotation)."
        ),
    )
    parser.add_argument(
        "--log-iterations",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use logarithmic scale for iteration count "
            "(default: enabled; disable with --no-log-iterations)."
        ),
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def parse_bool_series(series: pd.Series) -> pd.Series:
    """Convert common CSV boolean representations to bool."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y", "t"})
    )


def _validate_unique_trial_index(frame: pd.DataFrame, path: Path) -> None:
    if frame["trial_index"].duplicated().any():
        duplicates = frame.loc[
            frame["trial_index"].duplicated(keep=False),
            "trial_index",
        ].tolist()
        raise RuntimeError(
            f"Duplicate trial_index values in {path}: {duplicates[:10]}"
        )


def load_trials_for_rates(path: Path) -> pd.DataFrame:
    """Load every recorded trial without dropping failed/error rows.

    Only rows with a valid trial_index are removed. In particular, rows with
    error_type, NaN final error, success=False, or converged=False remain in
    the denominator of rate calculations.
    """
    if not path.is_file():
        raise FileNotFoundError(f"trials.csv not found: {path}")

    frame = pd.read_csv(path)

    if "trial_index" not in frame.columns:
        raise KeyError(f"trial_index column is missing: {path}")

    frame["trial_index"] = pd.to_numeric(
        frame["trial_index"],
        errors="coerce",
    )
    frame = frame[np.isfinite(frame["trial_index"])].copy()
    frame["trial_index"] = frame["trial_index"].astype(int)

    if frame.empty:
        raise RuntimeError(f"No valid trial_index rows found: {path}")

    _validate_unique_trial_index(frame, path)
    return frame


def load_trials_for_errors(
    path: Path,
    success_only: bool = False,
) -> pd.DataFrame:
    """Load rows that contain finite final translation and rotation errors."""
    frame = load_trials_for_rates(path)

    required = {
        "translation_error_mm",
        "rotation_error_deg",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"{path} is missing columns: {sorted(missing)}")

    if "error_type" in frame.columns:
        no_exception = (
            frame["error_type"]
            .fillna("")
            .astype(str)
            .str.strip()
            .eq("")
        )
        frame = frame[no_exception].copy()

    if success_only:
        if "success" not in frame.columns:
            raise KeyError(
                f"--success-only requested, but success column is missing: {path}"
            )
        frame = frame[parse_bool_series(frame["success"])].copy()

    numeric_columns = (
        "translation_error_mm",
        "rotation_error_deg",
        "iterations",
        "condition_last",
        "rank_last",
        "init_translation_error_mm",
        "init_rotation_error_deg",
    )
    for column in numeric_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(
                frame[column],
                errors="coerce",
            )

    finite_error = (
        np.isfinite(frame["translation_error_mm"])
        & np.isfinite(frame["rotation_error_deg"])
    )
    frame = frame[finite_error].copy()

    # An empty frame is valid, especially for restricted + --success-only.
    return frame


def discover_levels(
    result_root: Path,
    requested_levels: list[str] | None,
) -> list[dict]:
    if not result_root.is_dir():
        raise FileNotFoundError(f"Result root not found: {result_root}")

    if requested_levels is None:
        complete_names = {
            directory.name
            for directory in result_root.iterdir()
            if directory.is_dir()
            and all(
                (directory / method / "trials.csv").is_file()
                for method, _ in METHODS
            )
        }
        requested_levels = [
            name for name in DEFAULT_ORDER if name in complete_names
        ]
        requested_levels.extend(
            sorted(complete_names.difference(requested_levels))
        )

    levels: list[dict] = []

    for level_name in requested_levels:
        level_dir = result_root / level_name

        if not level_dir.is_dir():
            print(f"warning: missing level directory; skipping: {level_dir}")
            continue

        missing_files = [
            level_dir / method / "trials.csv"
            for method, _ in METHODS
            if not (level_dir / method / "trials.csv").is_file()
        ]
        if missing_files:
            print(
                "warning: incomplete level; skipping: "
                f"{level_name}, missing={missing_files}"
            )
            continue

        levels.append(
            {
                "name": level_name,
                "display_label": level_name.capitalize(),
                "directory": level_dir,
            }
        )

    if not levels:
        raise RuntimeError(
            f"No complete pose-diversity levels found under {result_root}"
        )

    return levels


def collect_method_frames(
    levels: list[dict],
    success_only: bool,
) -> dict[str, list[pd.DataFrame]]:
    result = {method: [] for method, _ in METHODS}

    for level in levels:
        for method, _ in METHODS:
            result[method].append(
                load_trials_for_errors(
                    level["directory"] / method / "trials.csv",
                    success_only=success_only,
                )
            )

    return result


def _empty_paired_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "trial_index",
            "translation_error_mm_single",
            "rotation_error_deg_single",
            "translation_error_mm_three",
            "rotation_error_deg_three",
            "pose_level",
            "translation_advantage_mm",
            "rotation_advantage_deg",
        ]
    )


def pair_level_errors(
    level: dict,
    success_only: bool,
) -> pd.DataFrame:
    """Pair finite-error rows by trial_index.

    advantage = single-plane error - three-plane error

    Therefore:
      positive -> lower error for three-plane
      negative -> lower error for single-plane
    """
    single = load_trials_for_errors(
        level["directory"] / "single_plane" / "trials.csv",
        success_only=success_only,
    )
    three = load_trials_for_errors(
        level["directory"] / "three_plane" / "trials.csv",
        success_only=success_only,
    )

    base = [
        "trial_index",
        "translation_error_mm",
        "rotation_error_deg",
    ]
    optional = [
        "iterations",
        "condition_last",
        "rank_last",
        "init_translation_error_mm",
        "init_rotation_error_deg",
        "converged",
        "success",
        "outlier",
    ]

    single_columns = base + [
        column for column in optional if column in single.columns
    ]
    three_columns = base + [
        column for column in optional if column in three.columns
    ]

    paired = single[single_columns].merge(
        three[three_columns],
        on="trial_index",
        how="inner",
        validate="one_to_one",
        suffixes=("_single", "_three"),
    )

    if paired.empty:
        empty = _empty_paired_frame()
        empty["pose_level"] = pd.Series(dtype=str)
        return empty

    paired["pose_level"] = level["name"]
    paired["translation_advantage_mm"] = (
        paired["translation_error_mm_single"]
        - paired["translation_error_mm_three"]
    )
    paired["rotation_advantage_deg"] = (
        paired["rotation_error_deg_single"]
        - paired["rotation_error_deg_three"]
    )

    if {
        "iterations_single",
        "iterations_three",
    }.issubset(paired.columns):
        paired["iteration_advantage"] = (
            pd.to_numeric(
                paired["iterations_single"],
                errors="coerce",
            )
            - pd.to_numeric(
                paired["iterations_three"],
                errors="coerce",
            )
        )

    if {
        "condition_last_single",
        "condition_last_three",
    }.issubset(paired.columns):
        denominator = pd.to_numeric(
            paired["condition_last_three"],
            errors="coerce",
        )
        paired["condition_ratio_single_over_three"] = (
            pd.to_numeric(
                paired["condition_last_single"],
                errors="coerce",
            )
            / denominator.replace(0.0, np.nan)
        )

    for column in (
        "init_translation_error_mm",
        "init_rotation_error_deg",
    ):
        left = f"{column}_single"
        right = f"{column}_three"

        if left not in paired.columns or right not in paired.columns:
            continue

        mismatch = ~np.isclose(
            pd.to_numeric(paired[left], errors="coerce"),
            pd.to_numeric(paired[right], errors="coerce"),
            rtol=1e-10,
            atol=1e-10,
            equal_nan=True,
        )

        if bool(mismatch.any()):
            print(
                f"warning: {level['name']}: {int(mismatch.sum())} pairs "
                f"have different {column}"
            )

    return paired


def collect_paired_errors(
    levels: list[dict],
    success_only: bool,
) -> pd.DataFrame:
    frames = [
        pair_level_errors(level, success_only)
        for level in levels
    ]
    nonempty = [frame for frame in frames if not frame.empty]

    if not nonempty:
        return _empty_paired_frame()

    return pd.concat(nonempty, ignore_index=True)


def _finite_values(
    frame: pd.DataFrame,
    column: str,
) -> np.ndarray:
    if column not in frame.columns:
        return np.array([], dtype=float)

    values = pd.to_numeric(
        frame[column],
        errors="coerce",
    ).to_numpy(dtype=float)
    return values[np.isfinite(values)]


def _boxplot_values(values: np.ndarray) -> np.ndarray:
    """Return a harmless placeholder when a group has no observations."""
    if len(values) == 0:
        return np.array([np.nan], dtype=float)
    return values


def _default_colors(count: int) -> list[str]:
    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if not colors:
        return [f"C{index}" for index in range(count)]
    return [colors[index % len(colors)] for index in range(count)]


def style_boxplot(
    boxplot: dict,
    facecolor: str,
) -> None:
    for box in boxplot["boxes"]:
        box.set_facecolor(facecolor)
        box.set_alpha(0.70)
        box.set_linewidth(1.0)

    for median in boxplot["medians"]:
        median.set_linewidth(1.8)

    for whisker in boxplot["whiskers"]:
        whisker.set_linewidth(1.0)

    for cap in boxplot["caps"]:
        cap.set_linewidth(1.0)

    for flier in boxplot["fliers"]:
        flier.set_marker("o")
        flier.set_markersize(2.8)
        flier.set_alpha(0.35)
        flier.set_markerfacecolor("none")
        flier.set_markeredgecolor("#555555")
        flier.set_markeredgewidth(0.7)


def draw_grouped_boxplot(
    axis: plt.Axes,
    levels: list[dict],
    frames_by_method: dict[str, list[pd.DataFrame]],
    column: str,
    ylabel: str,
    panel_label: str,
    hide_outliers: bool,
    use_log: bool,
) -> None:
    centers = np.arange(len(levels), dtype=float)
    offset = 0.18
    width = 0.30
    colors = _default_colors(2)

    method_specs = (
        ("single_plane", -offset, colors[0]),
        ("three_plane", +offset, colors[1]),
    )

    for method, position_offset, color in method_specs:
        data = [
            _boxplot_values(_finite_values(frame, column))
            for frame in frames_by_method[method]
        ]

        box = axis.boxplot(
            data,
            positions=centers + position_offset,
            widths=width,
            patch_artist=True,
            showmeans=not hide_outliers,
            showfliers=not hide_outliers,
            manage_ticks=False,
            meanprops={
                "marker": "D",
                "markerfacecolor": "white",
                "markersize": 3.8,
            },
        )
        style_boxplot(box, color)

    axis.set_xticks(centers)
    axis.set_xticklabels(
        [level["display_label"] for level in levels]
    )
    axis.set_xlabel("Pose-diversity level")
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
    axis.set_axisbelow(True)

    if use_log:
        axis.set_yscale("log")

    axis.text(
        0.01,
        0.98,
        panel_label,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        fontweight="bold",
    )

def draw_paired_boxplot(
    axis: plt.Axes,
    levels: list[dict],
    paired: pd.DataFrame,
    column: str,
    ylabel: str,
    panel_label: str,
    hide_outliers: bool,
) -> None:
    data: list[np.ndarray] = []
    sample_counts: list[int] = []

    for level in levels:
        if paired.empty or column not in paired.columns:
            values = np.array([], dtype=float)
        else:
            values = pd.to_numeric(
                paired.loc[
                    paired["pose_level"].eq(level["name"]),
                    column,
                ],
                errors="coerce",
            ).to_numpy(dtype=float)
            values = values[np.isfinite(values)]

        sample_counts.append(len(values))
        data.append(_boxplot_values(values))

    positions = np.arange(len(levels), dtype=float)
    color = _default_colors(3)[2]

    box = axis.boxplot(
        data,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showmeans=not hide_outliers,
        showfliers=not hide_outliers,
        manage_ticks=False,
        meanprops={
            "marker": "D",
            "markerfacecolor": "white",
            "markersize": 3.8,
        },
    )
    style_boxplot(box, color)

    axis.axhline(0.0, linestyle="--", linewidth=1.1)
    axis.set_xticks(positions)
    axis.set_xticklabels(
        [
            f"{level['display_label']}\n(n={count})"
            for level, count in zip(levels, sample_counts)
        ]
    )
    axis.set_xlabel("Pose-diversity level")
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
    axis.set_axisbelow(True)

    finite_values = np.concatenate(
        [values for values in data if values.size > 0]
    )
    if not hide_outliers and finite_values.size > 0:
        nonzero_abs = np.abs(finite_values[finite_values != 0.0])
        if nonzero_abs.size > 0:
            linear_threshold = float(np.percentile(nonzero_abs, 75))
            linear_threshold = max(
                linear_threshold,
                np.finfo(float).tiny,
            )
            maximum_abs = float(np.max(np.abs(finite_values)))
            axis.set_yscale(
                "symlog",
                linthresh=linear_threshold,
            )
            axis.set_ylim(
                -1.15 * maximum_abs,
                1.15 * maximum_abs,
            )

    axis.text(
        0.01,
        0.98,
        panel_label,
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        fontweight="bold",
    )


def save_final_error_plot(
    output: Path,
    levels: list[dict],
    frames_by_method: dict[str, list[pd.DataFrame]],
    args: argparse.Namespace,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12.4, 5.2))

    draw_grouped_boxplot(
        axes[0],
        levels,
        frames_by_method,
        "translation_error_mm",
        "Translation error [mm]",
        "(a)",
        args.hide_outliers,
        args.log_translation,
    )
    draw_grouped_boxplot(
        axes[1],
        levels,
        frames_by_method,
        "rotation_error_deg",
        "Rotation error [deg]",
        "(b)",
        args.hide_outliers,
        args.log_rotation,
    )

    colors = _default_colors(2)
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=colors[0]),
        plt.Rectangle((0, 0), 1, 1, facecolor=colors[1]),
    ]
    figure.legend(
        legend_handles,
        ["Single-plane", "Three-plane"],
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    figure.savefig(output, dpi=args.dpi, bbox_inches="tight")
    plt.close(figure)


def save_paired_error_plot(
    output: Path,
    levels: list[dict],
    paired: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12.4, 5.2))

    draw_paired_boxplot(
        axes[0],
        levels,
        paired,
        "translation_advantage_mm",
        "Paired advantage [mm]\nSingle-plane − Three-plane",
        "(a)",
        args.hide_outliers,
    )
    draw_paired_boxplot(
        axes[1],
        levels,
        paired,
        "rotation_advantage_deg",
        "Paired advantage [deg]\nSingle-plane − Three-plane",
        "(b)",
        args.hide_outliers,
    )

    scale_note = (
        " (symmetric-log y-axis)"
        if not args.hide_outliers
        else ""
    )
    figure.text(
        0.5,
        0.995,
        (
            "Positive values indicate lower error for three-plane "
            f"calibration{scale_note}"
        ),
        ha="center",
        va="top",
        fontsize=10,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    figure.savefig(output, dpi=args.dpi, bbox_inches="tight")
    plt.close(figure)


def collect_rates(levels: list[dict]) -> pd.DataFrame:
    """Calculate rates with every recorded trial in the denominator."""
    rows: list[dict] = []

    for level in levels:
        for method, label in METHODS:
            frame = load_trials_for_rates(
                level["directory"] / method / "trials.csv"
            )

            converged = (
                parse_bool_series(frame["converged"])
                if "converged" in frame.columns
                else pd.Series(False, index=frame.index)
            )
            success = (
                parse_bool_series(frame["success"])
                if "success" in frame.columns
                else pd.Series(False, index=frame.index)
            )
            outlier = (
                parse_bool_series(frame["outlier"])
                if "outlier" in frame.columns
                else pd.Series(False, index=frame.index)
            )
            exception = (
                frame["error_type"]
                .fillna("")
                .astype(str)
                .str.strip()
                .ne("")
                if "error_type" in frame.columns
                else pd.Series(False, index=frame.index)
            )

            iterations = (
                pd.to_numeric(frame["iterations"], errors="coerce")
                if "iterations" in frame.columns
                else pd.Series(np.nan, index=frame.index)
            )
            condition_values = (
                pd.to_numeric(frame["condition_last"], errors="coerce")
                if "condition_last" in frame.columns
                else pd.Series(np.nan, index=frame.index)
            )

            finite_iterations = iterations[np.isfinite(iterations)]
            converged_iterations = iterations[
                converged & np.isfinite(iterations)
            ]
            finite_condition = condition_values[
                np.isfinite(condition_values)
            ]

            total = int(len(frame))
            converged_count = int(converged.sum())
            success_count = int(success.sum())
            outlier_count = int(outlier.sum())
            exception_count = int(exception.sum())

            rows.append(
                {
                    "pose_level": level["name"],
                    "method": method,
                    "method_label": label,
                    "recorded_trials": total,
                    "converged_count": converged_count,
                    "success_count": success_count,
                    "outlier_count": outlier_count,
                    "exception_count": exception_count,
                    "convergence_rate": converged_count / total,
                    "success_rate": success_count / total,
                    "outlier_rate": outlier_count / total,
                    "exception_rate": exception_count / total,
                    "iterations_median_all_finite": (
                        float(np.median(finite_iterations))
                        if len(finite_iterations)
                        else float("nan")
                    ),
                    "iterations_median_converged": (
                        float(np.median(converged_iterations))
                        if len(converged_iterations)
                        else float("nan")
                    ),
                    "condition_median_finite": (
                        float(np.median(finite_condition))
                        if len(finite_condition)
                        else float("nan")
                    ),
                }
            )

    return pd.DataFrame(rows)


def save_rate_plot(
    output: Path,
    levels: list[dict],
    rates: pd.DataFrame,
    dpi: int,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12.4, 5.2))
    centers = np.arange(len(levels), dtype=float)
    width = 0.34
    colors = _default_colors(2)

    for axis, metric, ylabel, panel in (
        (axes[0], "convergence_rate", "Convergence rate [%]", "(a)"),
        (axes[1], "success_rate", "Success rate [%]", "(b)"),
    ):
        for method_index, (method, label) in enumerate(METHODS):
            offset = (-width / 2, width / 2)[method_index]
            method_rows = rates[rates["method"].eq(method)].set_index(
                "pose_level"
            )
            values = [
                100.0 * float(method_rows.loc[level["name"], metric])
                for level in levels
            ]
            bars = axis.bar(
                centers + offset,
                values,
                width=width,
                label=label,
                color=colors[method_index],
            )

            for bar, value in zip(bars, values):
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    min(value + 1.5, 102.0),
                    f"{value:.0f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        axis.set_xticks(centers)
        axis.set_xticklabels(
            [level["display_label"] for level in levels]
        )
        axis.set_xlabel("Pose-diversity level")
        axis.set_ylabel(ylabel)
        axis.set_ylim(0.0, 108.0)
        axis.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
        axis.set_axisbelow(True)
        axis.text(
            0.01,
            0.98,
            panel,
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=11,
            fontweight="bold",
        )

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02),
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def _exact_mcnemar_pvalue(
    single_only: int,
    three_only: int,
) -> float:
    """Exact two-sided McNemar p-value for discordant paired outcomes."""
    discordant = int(single_only + three_only)

    if discordant == 0:
        return 1.0

    smaller = min(single_only, three_only)
    lower_tail = sum(
        math.comb(discordant, k)
        for k in range(smaller + 1)
    ) / (2.0 ** discordant)

    return float(min(1.0, 2.0 * lower_tail))


def collect_paired_success_outcomes(
    levels: list[dict],
) -> pd.DataFrame:
    """Count paired success categories using all recorded rows.

    Outcome counts use trial indices present in both method CSVs. Missing
    trial indices are reported separately rather than silently counted as
    failures.
    """
    rows: list[dict] = []

    for level in levels:
        single = load_trials_for_rates(
            level["directory"] / "single_plane" / "trials.csv"
        )
        three = load_trials_for_rates(
            level["directory"] / "three_plane" / "trials.csv"
        )

        single_success = (
            parse_bool_series(single["success"])
            if "success" in single.columns
            else pd.Series(False, index=single.index)
        )
        three_success = (
            parse_bool_series(three["success"])
            if "success" in three.columns
            else pd.Series(False, index=three.index)
        )

        single_table = pd.DataFrame(
            {
                "trial_index": single["trial_index"],
                "success_single": single_success.to_numpy(dtype=bool),
            }
        )
        three_table = pd.DataFrame(
            {
                "trial_index": three["trial_index"],
                "success_three": three_success.to_numpy(dtype=bool),
            }
        )

        paired = single_table.merge(
            three_table,
            on="trial_index",
            how="inner",
            validate="one_to_one",
        )

        single_indices = set(single_table["trial_index"].tolist())
        three_indices = set(three_table["trial_index"].tolist())

        s = paired["success_single"].astype(bool)
        t = paired["success_three"].astype(bool)

        both_success = int((s & t).sum())
        single_only = int((s & ~t).sum())
        three_only = int((~s & t).sum())
        neither = int((~s & ~t).sum())
        common_trials = int(len(paired))

        rows.append(
            {
                "pose_level": level["name"],
                "single_recorded_trials": int(len(single_table)),
                "three_recorded_trials": int(len(three_table)),
                "union_trials": int(
                    len(single_indices.union(three_indices))
                ),
                "common_trials": common_trials,
                "missing_from_single": int(
                    len(three_indices.difference(single_indices))
                ),
                "missing_from_three": int(
                    len(single_indices.difference(three_indices))
                ),
                "both_success": both_success,
                "single_only_success": single_only,
                "three_only_success": three_only,
                "neither_success": neither,
                "discordant_success_pairs": single_only + three_only,
                "net_three_success_advantage_count": (
                    three_only - single_only
                ),
                "both_success_fraction": (
                    both_success / common_trials
                    if common_trials
                    else float("nan")
                ),
                "single_only_fraction": (
                    single_only / common_trials
                    if common_trials
                    else float("nan")
                ),
                "three_only_fraction": (
                    three_only / common_trials
                    if common_trials
                    else float("nan")
                ),
                "neither_fraction": (
                    neither / common_trials
                    if common_trials
                    else float("nan")
                ),
                "mcnemar_exact_pvalue": _exact_mcnemar_pvalue(
                    single_only,
                    three_only,
                ),
            }
        )

    return pd.DataFrame(rows)


def save_paired_success_plot(
    output: Path,
    levels: list[dict],
    outcomes: pd.DataFrame,
    dpi: int,
) -> None:
    """Save a stacked composition plot of paired success outcomes."""
    figure, axis = plt.subplots(figsize=(8.8, 5.4))
    positions = np.arange(len(levels), dtype=float)
    colors = _default_colors(4)

    categories = (
        ("both_success", "Both succeed"),
        ("single_only_success", "Single only"),
        ("three_only_success", "Three only"),
        ("neither_success", "Neither"),
    )

    bottoms = np.zeros(len(levels), dtype=float)

    for category_index, (column, label) in enumerate(categories):
        values = []
        for level in levels:
            row = outcomes[
                outcomes["pose_level"].eq(level["name"])
            ].iloc[0]
            common = int(row["common_trials"])
            count = int(row[column])
            values.append(
                100.0 * count / common if common else 0.0
            )

        axis.bar(
            positions,
            values,
            bottom=bottoms,
            label=label,
            color=colors[category_index],
        )
        bottoms += np.asarray(values, dtype=float)

    axis.set_xticks(positions)
    axis.set_xticklabels(
        [level["display_label"] for level in levels]
    )
    axis.set_xlabel("Pose-diversity level")
    axis.set_ylabel("Paired outcome share [%]")
    axis.set_ylim(0.0, 100.0)
    axis.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
    axis.set_axisbelow(True)
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.13),
        ncol=4,
        frameon=False,
    )

    figure.tight_layout()
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def save_iterations_plot(
    output: Path,
    levels: list[dict],
    frames_by_method: dict[str, list[pd.DataFrame]],
    args: argparse.Namespace,
) -> None:
    if not any(
        "iterations" in frame.columns
        for method_frames in frames_by_method.values()
        for frame in method_frames
    ):
        print("warning: iterations column missing; skipping iterations plot.")
        return

    figure, axis = plt.subplots(figsize=(7.8, 5.2))
    centers = np.arange(len(levels), dtype=float)
    offset = 0.18
    width = 0.30
    colors = _default_colors(2)

    for method_index, (method, _) in enumerate(METHODS):
        position_offset = (-offset, +offset)[method_index]
        data = [
            _boxplot_values(_finite_values(frame, "iterations"))
            for frame in frames_by_method[method]
        ]

        box = axis.boxplot(
            data,
            positions=centers + position_offset,
            widths=width,
            patch_artist=True,
            showmeans=not args.hide_outliers,
            showfliers=not args.hide_outliers,
            manage_ticks=False,
            meanprops={
                "marker": "D",
                "markerfacecolor": "white",
                "markersize": 3.8,
            },
        )
        style_boxplot(box, colors[method_index])

    axis.set_xticks(centers)
    axis.set_xticklabels(
        [level["display_label"] for level in levels]
    )
    axis.set_xlabel("Pose-diversity level")
    axis.set_ylabel("Iterations")
    axis.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
    axis.set_axisbelow(True)

    if args.log_iterations:
        axis.set_yscale("log")

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=colors[0]),
        plt.Rectangle((0, 0), 1, 1, facecolor=colors[1]),
    ]
    axis.legend(
        legend_handles,
        ["Single-plane", "Three-plane"],
        frameon=False,
    )
    figure.tight_layout()
    figure.savefig(output, dpi=args.dpi, bbox_inches="tight")
    plt.close(figure)


def save_condition_plot(
    output: Path,
    levels: list[dict],
    frames_by_method: dict[str, list[pd.DataFrame]],
    args: argparse.Namespace,
) -> None:
    if not any(
        "condition_last" in frame.columns
        for method_frames in frames_by_method.values()
        for frame in method_frames
    ):
        print(
            "warning: condition_last column missing; "
            "skipping condition-number plot."
        )
        return

    figure, axis = plt.subplots(figsize=(7.8, 5.2))
    centers = np.arange(len(levels), dtype=float)
    offset = 0.18
    width = 0.30
    colors = _default_colors(2)

    for method_index, (method, _) in enumerate(METHODS):
        position_offset = (-offset, +offset)[method_index]
        data = [
            _boxplot_values(_finite_values(frame, "condition_last"))
            for frame in frames_by_method[method]
        ]

        box = axis.boxplot(
            data,
            positions=centers + position_offset,
            widths=width,
            patch_artist=True,
            showmeans=not args.hide_outliers,
            showfliers=not args.hide_outliers,
            manage_ticks=False,
            meanprops={
                "marker": "D",
                "markerfacecolor": "white",
                "markersize": 3.8,
            },
        )
        style_boxplot(box, colors[method_index])

    axis.set_xticks(centers)
    axis.set_xticklabels(
        [level["display_label"] for level in levels]
    )
    axis.set_xlabel("Pose-diversity level")
    axis.set_ylabel("Final condition number")
    axis.set_yscale("log")
    axis.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
    axis.set_axisbelow(True)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=colors[0]),
        plt.Rectangle((0, 0), 1, 1, facecolor=colors[1]),
    ]
    axis.legend(
        legend_handles,
        ["Single-plane", "Three-plane"],
        frameon=False,
    )
    figure.tight_layout()
    figure.savefig(output, dpi=args.dpi, bbox_inches="tight")
    plt.close(figure)


def _safe_summary_stats(values: np.ndarray) -> dict[str, float]:
    if len(values) == 0:
        return {
            "median": float("nan"),
            "mean": float("nan"),
            "q25": float("nan"),
            "q75": float("nan"),
            "positive_fraction": float("nan"),
        }

    return {
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "q25": float(np.percentile(values, 25)),
        "q75": float(np.percentile(values, 75)),
        "positive_fraction": float(np.mean(values > 0.0)),
    }


def paired_error_summary(
    paired: pd.DataFrame,
    level_order: Iterable[str],
) -> pd.DataFrame:
    rows: list[dict] = []

    for level_name in level_order:
        group = (
            paired[paired["pose_level"].eq(level_name)]
            if not paired.empty
            else paired
        )

        row: dict[str, object] = {
            "pose_level": level_name,
            "paired_error_trials": int(len(group)),
        }

        for metric, column in (
            ("translation", "translation_advantage_mm"),
            ("rotation", "rotation_advantage_deg"),
        ):
            values = _finite_values(group, column)
            stats = _safe_summary_stats(values)

            row[f"{metric}_advantage_median"] = stats["median"]
            row[f"{metric}_advantage_mean"] = stats["mean"]
            row[f"{metric}_advantage_q25"] = stats["q25"]
            row[f"{metric}_advantage_q75"] = stats["q75"]
            row[f"three_better_{metric}_fraction"] = stats[
                "positive_fraction"
            ]

        if "iteration_advantage" in group.columns:
            iteration_values = _finite_values(
                group,
                "iteration_advantage",
            )
            iteration_stats = _safe_summary_stats(iteration_values)
            row["iteration_advantage_median"] = iteration_stats[
                "median"
            ]
            row["three_faster_fraction"] = iteration_stats[
                "positive_fraction"
            ]
        else:
            row["iteration_advantage_median"] = float("nan")
            row["three_faster_fraction"] = float("nan")

        rows.append(row)

    return pd.DataFrame(rows)


def method_error_summary(
    levels: list[dict],
    frames_by_method: dict[str, list[pd.DataFrame]],
) -> pd.DataFrame:
    rows: list[dict] = []

    for level_index, level in enumerate(levels):
        for method, method_label in METHODS:
            frame = frames_by_method[method][level_index]

            translation = _finite_values(
                frame,
                "translation_error_mm",
            )
            rotation = _finite_values(
                frame,
                "rotation_error_deg",
            )

            translation_stats = _safe_summary_stats(translation)
            rotation_stats = _safe_summary_stats(rotation)

            rows.append(
                {
                    "pose_level": level["name"],
                    "method": method,
                    "method_label": method_label,
                    "finite_error_trials": int(len(frame)),
                    "translation_median_mm": translation_stats["median"],
                    "translation_mean_mm": translation_stats["mean"],
                    "translation_q25_mm": translation_stats["q25"],
                    "translation_q75_mm": translation_stats["q75"],
                    "rotation_median_deg": rotation_stats["median"],
                    "rotation_mean_deg": rotation_stats["mean"],
                    "rotation_q25_deg": rotation_stats["q25"],
                    "rotation_q75_deg": rotation_stats["q75"],
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()

    result_root = args.result_root.expanduser().resolve()

    if args.output_dir is not None:
        output_dir = args.output_dir.expanduser().resolve()
    else:
        output_name = (
            "plots_success_only"
            if args.success_only
            else "plots_all"
        )
        output_dir = result_root / output_name

    output_dir.mkdir(parents=True, exist_ok=True)

    levels = discover_levels(
        result_root=result_root,
        requested_levels=args.levels,
    )

    error_frames = collect_method_frames(
        levels=levels,
        success_only=args.success_only,
    )
    paired_errors = collect_paired_errors(
        levels=levels,
        success_only=args.success_only,
    )
    rates = collect_rates(levels)
    success_outcomes = collect_paired_success_outcomes(levels)

    final_error_path = (
        output_dir / "pose_diversity_final_error_boxplots.png"
    )
    paired_error_path = (
        output_dir / "pose_diversity_paired_differences.png"
    )
    rates_path = output_dir / "pose_diversity_rates.png"
    success_outcome_plot_path = (
        output_dir / "pose_diversity_paired_success_outcomes.png"
    )
    iterations_path = output_dir / "pose_diversity_iterations.png"
    condition_path = output_dir / "pose_diversity_condition_number.png"

    paired_rows_path = output_dir / "pose_diversity_paired_rows.csv"
    paired_summary_path = (
        output_dir / "pose_diversity_paired_summary.csv"
    )
    rate_summary_path = (
        output_dir / "pose_diversity_rates_summary.csv"
    )
    success_summary_path = (
        output_dir / "pose_diversity_paired_success_summary.csv"
    )
    method_error_summary_path = (
        output_dir / "pose_diversity_method_error_summary.csv"
    )

    save_final_error_plot(
        final_error_path,
        levels,
        error_frames,
        args,
    )
    save_paired_error_plot(
        paired_error_path,
        levels,
        paired_errors,
        args,
    )
    save_rate_plot(
        rates_path,
        levels,
        rates,
        args.dpi,
    )
    save_paired_success_plot(
        success_outcome_plot_path,
        levels,
        success_outcomes,
        args.dpi,
    )
    save_iterations_plot(
        iterations_path,
        levels,
        error_frames,
        args,
    )
    save_condition_plot(
        condition_path,
        levels,
        error_frames,
        args,
    )

    paired_errors.to_csv(paired_rows_path, index=False)
    paired_error_summary(
        paired_errors,
        [level["name"] for level in levels],
    ).to_csv(paired_summary_path, index=False)
    rates.to_csv(rate_summary_path, index=False)
    success_outcomes.to_csv(success_summary_path, index=False)
    method_error_summary(
        levels,
        error_frames,
    ).to_csv(method_error_summary_path, index=False)

    print("Pose-diversity levels:")
    for level in levels:
        print(f"  {level['name']}")

    print(
        "Error filtering       : "
        + ("success-only" if args.success_only else "all finite errors")
    )
    print(f"Saved final-error plot : {final_error_path}")
    print(f"Saved paired plot      : {paired_error_path}")
    print(f"Saved rate plot        : {rates_path}")
    print(f"Saved outcome plot     : {success_outcome_plot_path}")

    if iterations_path.is_file():
        print(f"Saved iterations plot  : {iterations_path}")

    if condition_path.is_file():
        print(f"Saved condition plot   : {condition_path}")

    print(f"Saved paired rows      : {paired_rows_path}")
    print(f"Saved paired summary   : {paired_summary_path}")
    print(f"Saved rate summary     : {rate_summary_path}")
    print(f"Saved success summary  : {success_summary_path}")
    print(f"Saved method summary   : {method_error_summary_path}")


if __name__ == "__main__":
    main()
