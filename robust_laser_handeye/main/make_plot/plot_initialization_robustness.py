#!/usr/bin/env python3
"""
PYTHONPATH=. python3 main/make_plot/plot_initialization_robustness.py \
  --result-root results/fair_plane_initialization_shared_global
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHODS = (
    ("single_plane", "Single-plane"),
    ("three_plane", "Three-plane"),
)

LEVEL_PATTERN = re.compile(
    r"(?P<label>.+)_t(?P<translation>[0-9]+(?:[p.][0-9]+)?)_r"
    r"(?P<rotation>[0-9]+(?:[p.][0-9]+)?)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot fixed-noise initialization-robustness experiments.\n"
            "Expected structure:\n"
            "  RESULT_ROOT/easy_t25_r5/single_plane/trials.csv\n"
            "  RESULT_ROOT/easy_t25_r5/three_plane/trials.csv"
        )
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=Path("results/fair_plane_initialization_shared_global"),
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=None,
        help=(
            "Condition directory names to include, in plotting order. "
            "When omitted, complete *_t*_r* directories are auto-discovered "
            "and ordered by severity."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <result-root>/plots",
    )
    parser.add_argument(
        "--success-only",
        action="store_true",
        help=(
            "For final-error and paired-error plots, use only rows whose "
            "success column is true. Rate plots always use all completed trials."
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
            "Use logarithmic scale for final translation error (default: on). "
            "Use --no-log-translation for a linear axis."
        ),
    )
    parser.add_argument(
        "--log-rotation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use logarithmic scale for final rotation error (default: on). "
            "Use --no-log-rotation for a linear axis."
        ),
    )
    parser.add_argument(
        "--log-iterations",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use logarithmic scale for iteration count (default: on). "
            "Use --no-log-iterations for a linear axis."
        ),
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def parse_number(text: str) -> float:
    return float(text.replace("p", "."))


def parse_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes", "y"})
    )


def load_trials_for_rates(path: Path) -> pd.DataFrame:
    """Load all recorded trials for unbiased rate denominators."""
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
        raise RuntimeError(f"No valid trial_index rows remain: {path}")

    if frame["trial_index"].duplicated().any():
        duplicate_values = frame.loc[
            frame["trial_index"].duplicated(keep=False),
            "trial_index",
        ].tolist()
        raise RuntimeError(
            f"Duplicate trial_index values in {path}: {duplicate_values[:10]}"
        )

    return frame


def load_trials(path: Path, success_only: bool = False) -> pd.DataFrame:
    """Load trials with finite final errors for error-based plots."""
    frame = load_trials_for_rates(path)

    required = {
        "translation_error_mm",
        "rotation_error_deg",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"{path} is missing columns: {sorted(missing)}")

    if "error_type" in frame.columns:
        frame = frame[
            frame["error_type"]
            .fillna("")
            .astype(str)
            .str.strip()
            .eq("")
        ].copy()

    if success_only:
        if "success" not in frame.columns:
            raise KeyError(
                f"--success-only requested, but success column is missing: {path}"
            )
        frame = frame[parse_bool_series(frame["success"])].copy()

    numeric_columns = [
        "trial_index",
        "translation_error_mm",
        "rotation_error_deg",
        "iterations",
        "init_translation_error_mm",
        "init_rotation_error_deg",
    ]
    for column in numeric_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame[
        np.isfinite(frame["trial_index"])
        & np.isfinite(frame["translation_error_mm"])
        & np.isfinite(frame["rotation_error_deg"])
    ].copy()
    return frame


def discover_conditions(
    result_root: Path,
    requested_conditions: list[str] | None = None,
) -> list[dict]:
    if not result_root.is_dir():
        raise FileNotFoundError(f"Result root not found: {result_root}")

    conditions: list[dict] = []

    for directory in result_root.iterdir():
        if not directory.is_dir():
            continue

        match = LEVEL_PATTERN.fullmatch(directory.name)
        if match is None:
            continue

        complete = all(
            (directory / method / "trials.csv").is_file()
            for method, _ in METHODS
        )
        if not complete:
            continue

        translation_range = parse_number(match.group("translation"))
        rotation_range = parse_number(match.group("rotation"))
        label = match.group("label")
        summary_path = directory / "single_plane" / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary_config = summary.get("config", {})
        translation_perturbation = summary_config.get(
            "init_translation_perturbation",
            "box_xyz",
        )
        rotation_perturbation = summary_config.get(
            "init_rotation_perturbation",
            "euler_xyz",
        )
        if (
            translation_perturbation == "direction_norm"
            and rotation_perturbation == "axis_angle"
        ):
            range_label = (
                f"≤{translation_range:g} mm, ≤{rotation_range:g}°"
            )
        else:
            range_label = (
                f"±{translation_range:g} mm, ±{rotation_range:g}°/axis"
            )

        conditions.append(
            {
                "name": directory.name,
                "label": label,
                "translation_range_mm": translation_range,
                "rotation_range_deg": rotation_range,
                "severity": translation_range + 10.0 * rotation_range,
                "directory": directory,
                "translation_perturbation": translation_perturbation,
                "rotation_perturbation": rotation_perturbation,
                "display_label": (
                    f"{label.capitalize()}\n"
                    f"{range_label}"
                ),
            }
        )

    conditions.sort(key=lambda item: item["severity"])

    if not conditions:
        raise RuntimeError(
            f"No complete *_t*_r* result folders found under {result_root}"
        )

    if requested_conditions is not None:
        by_name = {condition["name"]: condition for condition in conditions}
        missing = [
            name for name in requested_conditions if name not in by_name
        ]
        if missing:
            raise RuntimeError(
                f"Requested initialization conditions are incomplete or "
                f"missing under {result_root}: {missing}"
            )
        conditions = [by_name[name] for name in requested_conditions]

    return conditions


def collect_method_frames(
    conditions: list[dict],
    success_only: bool,
) -> dict[str, list[pd.DataFrame]]:
    result = {method: [] for method, _ in METHODS}

    for condition in conditions:
        for method, _ in METHODS:
            result[method].append(
                load_trials(
                    condition["directory"] / method / "trials.csv",
                    success_only=success_only,
                )
            )

    return result


def pair_condition(
    condition: dict,
    success_only: bool,
) -> pd.DataFrame:
    single = load_trials(
        condition["directory"] / "single_plane" / "trials.csv",
        success_only=success_only,
    )
    three = load_trials(
        condition["directory"] / "three_plane" / "trials.csv",
        success_only=success_only,
    )

    base = [
        "trial_index",
        "translation_error_mm",
        "rotation_error_deg",
    ]
    optional = [
        "init_translation_error_mm",
        "init_rotation_error_deg",
        "iterations",
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

    paired["condition"] = condition["name"]
    paired["level"] = condition["label"]
    paired["init_translation_range_mm"] = condition["translation_range_mm"]
    paired["init_rotation_range_deg"] = condition["rotation_range_deg"]

    paired["translation_advantage_mm"] = (
        paired["translation_error_mm_single"]
        - paired["translation_error_mm_three"]
    )
    paired["rotation_advantage_deg"] = (
        paired["rotation_error_deg_single"]
        - paired["rotation_error_deg_three"]
    )

    if (
        "iterations_single" in paired.columns
        and "iterations_three" in paired.columns
    ):
        paired["iteration_advantage"] = (
            pd.to_numeric(paired["iterations_single"], errors="coerce")
            - pd.to_numeric(paired["iterations_three"], errors="coerce")
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
                f"warning: {condition['name']}: {int(mismatch.sum())} pairs "
                f"have different {column}"
            )

    return paired


def collect_paired(
    conditions: list[dict],
    success_only: bool,
) -> pd.DataFrame:
    return pd.concat(
        [
            pair_condition(condition, success_only)
            for condition in conditions
        ],
        ignore_index=True,
    )


def style_boxplot(boxplot: dict, facecolor: str) -> None:
    for box in boxplot["boxes"]:
        box.set_facecolor(facecolor)
        box.set_alpha(0.78)
        box.set_edgecolor("#303030")
        box.set_linewidth(1.0)

    for median in boxplot["medians"]:
        median.set_color("#111111")
        median.set_linewidth(1.8)

    for whisker in boxplot["whiskers"]:
        whisker.set_color("#555555")
        whisker.set_linewidth(1.0)

    for cap in boxplot["caps"]:
        cap.set_color("#555555")
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
    conditions: list[dict],
    frames_by_method: dict[str, list[pd.DataFrame]],
    column: str,
    ylabel: str,
    panel_label: str,
    hide_outliers: bool,
    use_log: bool,
) -> None:
    centers = np.arange(len(conditions), dtype=float)
    offset = 0.18
    width = 0.30

    single_data = [
        frame[column].dropna().to_numpy(dtype=float)
        for frame in frames_by_method["single_plane"]
    ]
    three_data = [
        frame[column].dropna().to_numpy(dtype=float)
        for frame in frames_by_method["three_plane"]
    ]

    single = axis.boxplot(
        single_data,
        positions=centers - offset,
        widths=width,
        patch_artist=True,
        showmeans=not hide_outliers,
        showfliers=not hide_outliers,
        manage_ticks=False,
        meanprops={
            "marker": "D",
            "markerfacecolor": "white",
            "markeredgecolor": "#111111",
            "markersize": 3.8,
        },
    )
    three = axis.boxplot(
        three_data,
        positions=centers + offset,
        widths=width,
        patch_artist=True,
        showmeans=not hide_outliers,
        showfliers=not hide_outliers,
        manage_ticks=False,
        meanprops={
            "marker": "D",
            "markerfacecolor": "white",
            "markeredgecolor": "#111111",
            "markersize": 3.8,
        },
    )

    style_boxplot(single, "#7FA6D8")
    style_boxplot(three, "#E6A36A")

    axis.set_xticks(centers)
    axis.set_xticklabels(
        [condition["display_label"] for condition in conditions]
    )
    axis.set_xlabel("Initialization level")
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
    conditions: list[dict],
    paired: pd.DataFrame,
    column: str,
    ylabel: str,
    panel_label: str,
    hide_outliers: bool,
) -> None:
    data = []

    for condition in conditions:
        values = pd.to_numeric(
            paired.loc[paired["condition"].eq(condition["name"]), column],
            errors="coerce",
        ).to_numpy()
        data.append(values[np.isfinite(values)])

    positions = np.arange(len(conditions), dtype=float)

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
            "markeredgecolor": "#111111",
            "markersize": 3.8,
        },
    )
    style_boxplot(box, "#8DBF9B")

    axis.axhline(0.0, color="#202020", linestyle="--", linewidth=1.1)
    axis.set_xticks(positions)
    axis.set_xticklabels(
        [condition["display_label"] for condition in conditions]
    )
    axis.set_xlabel("Initialization level")
    axis.set_ylabel(ylabel)
    axis.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
    axis.set_axisbelow(True)

    finite_blocks = [
        np.asarray(values, dtype=float)[np.isfinite(values)]
        for values in data
        if len(values)
    ]
    if not hide_outliers and finite_blocks:
        finite_values = np.concatenate(finite_blocks)
        absolute_values = np.abs(finite_values)
        nonzero = absolute_values[absolute_values > 0.0]
        if len(nonzero):
            # Keep the central paired distribution linear around zero while
            # compressing catastrophic positive/negative local-minimum
            # failures. A symmetric range prevents one-sided failures from
            # pushing zero to the edge of the panel.
            linear_threshold = max(
                float(np.percentile(nonzero, 75)),
                np.finfo(float).eps,
            )
            symmetric_limit = 1.15 * float(np.max(nonzero))
            axis.set_yscale(
                "symlog",
                linthresh=linear_threshold,
                linscale=1.0,
            )
            axis.set_ylim(-symmetric_limit, symmetric_limit)

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
    conditions: list[dict],
    frames_by_method: dict[str, list[pd.DataFrame]],
    args: argparse.Namespace,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12.4, 5.2))

    draw_grouped_boxplot(
        axes[0],
        conditions,
        frames_by_method,
        "translation_error_mm",
        "Translation error [mm]",
        "(a)",
        args.hide_outliers,
        args.log_translation,
    )
    draw_grouped_boxplot(
        axes[1],
        conditions,
        frames_by_method,
        "rotation_error_deg",
        "Rotation error [deg]",
        "(b)",
        args.hide_outliers,
        args.log_rotation,
    )

    legend_handles = [
        plt.Rectangle(
            (0, 0), 1, 1,
            facecolor="#7FA6D8",
            edgecolor="#303030",
        ),
        plt.Rectangle(
            (0, 0), 1, 1,
            facecolor="#E6A36A",
            edgecolor="#303030",
        ),
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
    conditions: list[dict],
    paired: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12.4, 5.2))

    draw_paired_boxplot(
        axes[0],
        conditions,
        paired,
        "translation_advantage_mm",
        "Paired advantage [mm]\nSingle-plane − Three-plane",
        "(a)",
        args.hide_outliers,
    )
    draw_paired_boxplot(
        axes[1],
        conditions,
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


def condition_rates(
    conditions: list[dict],
) -> pd.DataFrame:
    rows = []

    for condition in conditions:
        for method, label in METHODS:
            frame = load_trials_for_rates(
                condition["directory"] / method / "trials.csv",
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

            iterations = (
                pd.to_numeric(frame["iterations"], errors="coerce")
                if "iterations" in frame.columns
                else pd.Series(np.nan, index=frame.index)
            )
            iterations = iterations[np.isfinite(iterations)]

            rows.append(
                {
                    "condition": condition["name"],
                    "level": condition["label"],
                    "display_label": condition["display_label"],
                    "method": method,
                    "method_label": label,
                    "trials": int(len(frame)),
                    "convergence_rate": float(converged.mean()),
                    "success_rate": float(success.mean()),
                    "outlier_rate": float(outlier.mean()),
                    "iterations_median": (
                        float(np.median(iterations))
                        if len(iterations)
                        else float("nan")
                    ),
                    "iterations_q25": (
                        float(np.percentile(iterations, 25))
                        if len(iterations)
                        else float("nan")
                    ),
                    "iterations_q75": (
                        float(np.percentile(iterations, 75))
                        if len(iterations)
                        else float("nan")
                    ),
                }
            )

    return pd.DataFrame(rows)


def save_rate_plot(
    output: Path,
    conditions: list[dict],
    rates: pd.DataFrame,
    dpi: int,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12.4, 5.2))
    centers = np.arange(len(conditions), dtype=float)
    width = 0.34

    for axis, metric, ylabel, panel in (
        (axes[0], "convergence_rate", "Convergence rate [%]", "(a)"),
        (axes[1], "success_rate", "Success rate [%]", "(b)"),
    ):
        for offset, (method, label) in zip((-width / 2, width / 2), METHODS):
            method_rows = (
                rates[rates["method"].eq(method)]
                .set_index("condition")
            )
            values = [
                100.0 * float(method_rows.loc[c["name"], metric])
                for c in conditions
            ]
            axis.bar(
                centers + offset,
                values,
                width=width,
                label=label,
            )

        axis.set_xticks(centers)
        axis.set_xticklabels(
            [condition["display_label"] for condition in conditions]
        )
        axis.set_xlabel("Initialization level")
        axis.set_ylabel(ylabel)
        axis.set_ylim(0.0, 105.0)
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


def save_iteration_plot(
    output: Path,
    conditions: list[dict],
    frames_by_method: dict[str, list[pd.DataFrame]],
    args: argparse.Namespace,
) -> None:
    if not all(
        "iterations" in frame.columns
        for method_frames in frames_by_method.values()
        for frame in method_frames
    ):
        print("warning: iterations column missing; skipping iteration plot.")
        return

    figure, axis = plt.subplots(figsize=(7.8, 5.2))
    centers = np.arange(len(conditions), dtype=float)
    offset = 0.18
    width = 0.30

    for method, position_offset, color in (
        ("single_plane", -offset, "#7FA6D8"),
        ("three_plane", offset, "#E6A36A"),
    ):
        data = []
        for frame in frames_by_method[method]:
            values = pd.to_numeric(frame["iterations"], errors="coerce").to_numpy()
            data.append(values[np.isfinite(values)])

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
                "markeredgecolor": "#111111",
                "markersize": 3.8,
            },
        )
        style_boxplot(box, color)

    axis.set_xticks(centers)
    axis.set_xticklabels(
        [condition["display_label"] for condition in conditions]
    )
    axis.set_xlabel("Initialization level")
    axis.set_ylabel("Iterations")
    axis.grid(axis="y", linestyle="--", linewidth=0.7, alpha=0.35)
    axis.set_axisbelow(True)

    if args.log_iterations:
        axis.set_yscale("log")

    legend_handles = [
        plt.Rectangle(
            (0, 0), 1, 1,
            facecolor="#7FA6D8",
            edgecolor="#303030",
        ),
        plt.Rectangle(
            (0, 0), 1, 1,
            facecolor="#E6A36A",
            edgecolor="#303030",
        ),
    ]
    axis.legend(
        legend_handles,
        ["Single-plane", "Three-plane"],
        frameon=False,
    )
    figure.tight_layout()
    figure.savefig(output, dpi=args.dpi, bbox_inches="tight")
    plt.close(figure)


def paired_summary(paired: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "condition",
        "level",
        "init_translation_range_mm",
        "init_rotation_range_deg",
        "paired_trials",
        "translation_advantage_median",
        "translation_advantage_mean",
        "translation_advantage_q25",
        "translation_advantage_q75",
        "three_better_translation_fraction",
        "rotation_advantage_median",
        "rotation_advantage_mean",
        "rotation_advantage_q25",
        "rotation_advantage_q75",
        "three_better_rotation_fraction",
    ]
    rows = []

    for condition, group in paired.groupby("condition", sort=False):
        row = {
            "condition": condition,
            "level": str(group["level"].iloc[0]),
            "init_translation_range_mm": float(
                group["init_translation_range_mm"].iloc[0]
            ),
            "init_rotation_range_deg": float(
                group["init_rotation_range_deg"].iloc[0]
            ),
            "paired_trials": int(len(group)),
        }

        for metric, column in (
            ("translation", "translation_advantage_mm"),
            ("rotation", "rotation_advantage_deg"),
        ):
            values = pd.to_numeric(group[column], errors="coerce").to_numpy()
            values = values[np.isfinite(values)]

            row[f"{metric}_advantage_median"] = float(np.median(values))
            row[f"{metric}_advantage_mean"] = float(np.mean(values))
            row[f"{metric}_advantage_q25"] = float(
                np.percentile(values, 25)
            )
            row[f"{metric}_advantage_q75"] = float(
                np.percentile(values, 75)
            )
            row[f"three_better_{metric}_fraction"] = float(
                np.mean(values > 0.0)
            )

        rows.append(row)

    return pd.DataFrame(rows, columns=columns)


def main() -> None:
    args = parse_args()

    result_root = args.result_root.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser()
        if args.output_dir is not None
        else result_root / "plots"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    conditions = discover_conditions(
        result_root,
        requested_conditions=args.conditions,
    )

    frames = collect_method_frames(
        conditions,
        success_only=args.success_only,
    )
    paired = collect_paired(
        conditions,
        success_only=args.success_only,
    )
    rates = condition_rates(conditions)

    final_error_path = output_dir / "initialization_final_error_boxplots.png"
    paired_error_path = output_dir / "initialization_paired_differences.png"
    rates_path = output_dir / "initialization_rates.png"
    iterations_path = output_dir / "initialization_iterations.png"
    paired_csv_path = output_dir / "initialization_paired_differences.csv"
    paired_summary_path = (
        output_dir / "initialization_paired_differences_summary.csv"
    )
    rates_csv_path = output_dir / "initialization_rates_summary.csv"

    save_final_error_plot(
        final_error_path,
        conditions,
        frames,
        args,
    )
    save_paired_error_plot(
        paired_error_path,
        conditions,
        paired,
        args,
    )
    save_rate_plot(
        rates_path,
        conditions,
        rates,
        args.dpi,
    )
    save_iteration_plot(
        iterations_path,
        conditions,
        frames,
        args,
    )

    paired.to_csv(paired_csv_path, index=False)
    paired_summary(paired).to_csv(paired_summary_path, index=False)
    rates.to_csv(rates_csv_path, index=False)

    print("Initialization conditions:")
    for condition in conditions:
        print(
            f"  {condition['name']}: "
            f"±{condition['translation_range_mm']:g} mm, "
            f"±{condition['rotation_range_deg']:g} deg"
        )

    print(f"Saved final-error plot : {final_error_path}")
    print(f"Saved paired plot      : {paired_error_path}")
    print(f"Saved rate plot        : {rates_path}")
    if iterations_path.is_file():
        print(f"Saved iteration plot   : {iterations_path}")
    print(f"Saved paired rows      : {paired_csv_path}")
    print(f"Saved paired summary   : {paired_summary_path}")
    print(f"Saved rate summary     : {rates_csv_path}")


if __name__ == "__main__":
    main()
