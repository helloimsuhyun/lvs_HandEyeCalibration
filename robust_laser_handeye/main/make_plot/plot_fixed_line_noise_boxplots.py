#!/usr/bin/env python3
"""
PYTHONPATH=. python3 main/make_plot/plot_fixed_line_noise_boxplots.py \
  --result-root results/fair_plane_noise_fixed_line_shared_global

"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHODS = (
    ("single_plane", "Single-plane"),
    ("three_plane", "Three-plane"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot grouped boxplots for fixed-line-count noise experiments.\n"
            "Expected structure:\n"
            "  result_root/noise_0p00/single_plane/trials.csv\n"
            "  result_root/noise_0p00/three_plane/trials.csv"
        )
    )

    parser.add_argument(
        "--result-root",
        type=Path,
        default=Path("results/fair_plane_noise_fixed_line_shared_global"),
    )
    parser.add_argument(
        "--noise-levels",
        nargs="+",
        type=float,
        default=None,
        help=(
            "Noise levels to include, in plotting order. "
            "When omitted, all complete noise_* directories are discovered."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Default: <result-root>/noise_boxplots.png",
    )
    parser.add_argument(
        "--success-only",
        action="store_true",
        help="Plot only rows whose success column is True.",
    )
    parser.add_argument(
        "--hide-outliers",
        action="store_true",
        help="Hide individual outlier markers.",
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
        "--dpi",
        type=int,
        default=300,
    )
    parser.add_argument(
        "--paired-output",
        type=Path,
        default=None,
        help="Default: <result-root>/noise_paired_differences.png",
    )
    parser.add_argument(
        "--paired-csv",
        type=Path,
        default=None,
        help="Default: <result-root>/noise_paired_differences.csv",
    )

    return parser.parse_args()


def parse_noise_directory(name: str) -> float:
    """
    Convert:
        noise_0p00 -> 0.00
        noise_0p15 -> 0.15
        noise_1p00 -> 1.00
    """
    match = re.fullmatch(r"noise_([0-9]+(?:[p.][0-9]+)?)", name)

    if match is None:
        raise ValueError(f"Invalid noise directory name: {name}")

    return float(match.group(1).replace("p", "."))


def parse_bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes"})
    )


def load_trials(
    csv_path: Path,
    success_only: bool,
) -> pd.DataFrame:
    if not csv_path.is_file():
        raise FileNotFoundError(f"Missing trials.csv: {csv_path}")

    frame = pd.read_csv(csv_path)

    required_columns = {
        "trial_index",
        "translation_error_mm",
        "rotation_error_deg",
    }

    missing = required_columns.difference(frame.columns)
    if missing:
        raise KeyError(
            f"{csv_path} is missing required columns: {sorted(missing)}"
        )

    # Remove rows that failed with an exception.
    if "error_type" in frame.columns:
        frame = frame[
            frame["error_type"]
            .fillna("")
            .astype(str)
            .str.strip()
            .eq("")
        ]

    # Optional filtering.
    if success_only:
        if "success" not in frame.columns:
            raise KeyError(
                f"--success-only requested, but no success column exists: "
                f"{csv_path}"
            )

        frame = frame[parse_bool_series(frame["success"])]

    numeric_columns = [
        "trial_index",
        "translation_error_mm",
        "rotation_error_deg",
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
    frame["trial_index"] = frame["trial_index"].astype(int)

    if frame["trial_index"].duplicated().any():
        duplicate_values = frame.loc[
            frame["trial_index"].duplicated(keep=False),
            "trial_index",
        ].tolist()
        raise RuntimeError(
            f"Duplicate trial_index values in {csv_path}: "
            f"{duplicate_values[:10]}"
        )

    return frame


def discover_noise_conditions(
    result_root: Path,
    requested_noise_levels: list[float] | None = None,
) -> list[tuple[float, Path]]:
    if not result_root.is_dir():
        raise FileNotFoundError(
            f"Result root does not exist: {result_root}"
        )

    conditions: list[tuple[float, Path]] = []

    for directory in result_root.iterdir():
        if not directory.is_dir():
            continue

        try:
            noise_std = parse_noise_directory(directory.name)
        except ValueError:
            continue

        complete = all(
            (directory / method_dir / "trials.csv").is_file()
            for method_dir, _ in METHODS
        )

        if complete:
            conditions.append((noise_std, directory))

    conditions.sort(key=lambda item: item[0])

    if not conditions:
        raise RuntimeError(
            f"No complete noise_* result folders found under: {result_root}"
        )

    if requested_noise_levels is not None:
        selected: list[tuple[float, Path]] = []
        for requested in requested_noise_levels:
            matches = [
                item
                for item in conditions
                if np.isclose(item[0], requested, rtol=0.0, atol=1e-12)
            ]
            if not matches:
                raise RuntimeError(
                    f"Requested noise level {requested:g} has no complete "
                    f"result folder under {result_root}"
                )
            if len(matches) > 1:
                raise RuntimeError(
                    f"Multiple result folders represent noise level "
                    f"{requested:g}: {[path.name for _, path in matches]}"
                )
            selected.append(matches[0])
        conditions = selected

    return conditions


def collect_metric(
    conditions: list[tuple[float, Path]],
    metric: str,
    success_only: bool,
) -> dict[str, list[np.ndarray]]:
    data = {
        method_dir: []
        for method_dir, _ in METHODS
    }

    for _, condition_dir in conditions:
        for method_dir, _ in METHODS:
            csv_path = (
                condition_dir
                / method_dir
                / "trials.csv"
            )

            frame = load_trials(
                csv_path,
                success_only=success_only,
            )

            values = frame[metric].to_numpy(dtype=float)
            data[method_dir].append(values)

    return data


def style_boxplot(
    boxplot: dict,
    face_color: str,
) -> None:
    for box in boxplot["boxes"]:
        box.set_facecolor(face_color)
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
    noise_levels: list[float],
    metric_data: dict[str, list[np.ndarray]],
    ylabel: str,
    panel_label: str,
    hide_outliers: bool,
    use_log_scale: bool,
) -> None:
    centers = np.arange(len(noise_levels), dtype=float)

    offset = 0.18
    width = 0.30

    single_box = axis.boxplot(
        metric_data["single_plane"],
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

    three_box = axis.boxplot(
        metric_data["three_plane"],
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

    style_boxplot(
        single_box,
        face_color="#7FA6D8",
    )

    style_boxplot(
        three_box,
        face_color="#E6A36A",
    )

    axis.set_xticks(centers)
    axis.set_xticklabels(
        [f"{value:g}" for value in noise_levels]
    )

    axis.set_xlabel(
        "Profile noise standard deviation [mm]"
    )
    axis.set_ylabel(ylabel)

    axis.grid(
        axis="y",
        linestyle="--",
        linewidth=0.7,
        alpha=0.35,
    )
    axis.set_axisbelow(True)

    if use_log_scale:
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



def pair_condition(
    condition_dir: Path,
    success_only: bool,
    noise_std: float,
) -> pd.DataFrame:
    """Pair rows by trial_index.

    advantage = single-plane error - three-plane error
    Positive values therefore indicate lower error for three-plane.
    """
    single = load_trials(
        condition_dir / "single_plane" / "trials.csv",
        success_only=success_only,
    )
    three = load_trials(
        condition_dir / "three_plane" / "trials.csv",
        success_only=success_only,
    )

    base = ["trial_index", "translation_error_mm", "rotation_error_deg"]
    optional = ["init_translation_error_mm", "init_rotation_error_deg"]

    paired = single[
        base + [name for name in optional if name in single.columns]
    ].merge(
        three[
            base + [name for name in optional if name in three.columns]
        ],
        on="trial_index",
        how="inner",
        validate="one_to_one",
        suffixes=("_single", "_three"),
    )

    paired["noise_std_mm"] = float(noise_std)
    paired["translation_advantage_mm"] = (
        paired["translation_error_mm_single"]
        - paired["translation_error_mm_three"]
    )
    paired["rotation_advantage_deg"] = (
        paired["rotation_error_deg_single"]
        - paired["rotation_error_deg_three"]
    )

    for name in optional:
        left = f"{name}_single"
        right = f"{name}_three"
        if left in paired.columns and right in paired.columns:
            mismatch = ~np.isclose(
                pd.to_numeric(paired[left], errors="coerce"),
                pd.to_numeric(paired[right], errors="coerce"),
                rtol=1e-10,
                atol=1e-10,
                equal_nan=True,
            )
            if bool(mismatch.any()):
                print(
                    f"warning: {int(mismatch.sum())} pairs have different "
                    f"{name} at noise={noise_std:g}"
                )

    return paired


def collect_paired_data(
    conditions: list[tuple[float, Path]],
    success_only: bool,
) -> pd.DataFrame:
    frames = [
        pair_condition(path, success_only, noise)
        for noise, path in conditions
    ]
    return pd.concat(frames, ignore_index=True)


def paired_summary(paired: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "noise_std_mm",
        "paired_trials",
        "translation_advantage_median",
        "translation_advantage_q25",
        "translation_advantage_q75",
        "three_better_translation_fraction",
        "rotation_advantage_median",
        "rotation_advantage_q25",
        "rotation_advantage_q75",
        "three_better_rotation_fraction",
    ]
    rows = []
    for noise, group in paired.groupby("noise_std_mm", sort=True):
        row = {
            "noise_std_mm": float(noise),
            "paired_trials": int(len(group)),
        }
        for name, column in (
            ("translation", "translation_advantage_mm"),
            ("rotation", "rotation_advantage_deg"),
        ):
            values = pd.to_numeric(group[column], errors="coerce").to_numpy()
            values = values[np.isfinite(values)]
            row[f"{name}_advantage_median"] = float(np.median(values))
            row[f"{name}_advantage_q25"] = float(np.percentile(values, 25))
            row[f"{name}_advantage_q75"] = float(np.percentile(values, 75))
            row[f"three_better_{name}_fraction"] = float(np.mean(values > 0.0))
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def draw_paired_boxplot(
    axis: plt.Axes,
    noise_levels: list[float],
    paired: pd.DataFrame,
    column: str,
    ylabel: str,
    panel_label: str,
    hide_outliers: bool,
) -> None:
    data = []
    for noise in noise_levels:
        values = pd.to_numeric(
            paired.loc[np.isclose(paired["noise_std_mm"], noise), column],
            errors="coerce",
        ).to_numpy()
        data.append(values[np.isfinite(values)])

    positions = np.arange(len(noise_levels), dtype=float)
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
    style_boxplot(box, face_color="#8DBF9B")
    axis.axhline(0.0, color="#202020", linestyle="--", linewidth=1.1)
    axis.set_xticks(positions)
    axis.set_xticklabels([f"{value:g}" for value in noise_levels])
    axis.set_xlabel("Profile noise standard deviation [mm]")
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
        0.01, 0.98, panel_label,
        transform=axis.transAxes,
        ha="left", va="top",
        fontsize=11, fontweight="bold",
    )


def save_paired_plot(
    paired: pd.DataFrame,
    noise_levels: list[float],
    output_path: Path,
    hide_outliers: bool,
    dpi: int,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))

    draw_paired_boxplot(
        axes[0],
        noise_levels,
        paired,
        "translation_advantage_mm",
        "Paired advantage [mm]\nSingle-plane − Three-plane",
        "(a)",
        hide_outliers,
    )
    draw_paired_boxplot(
        axes[1],
        noise_levels,
        paired,
        "rotation_advantage_deg",
        "Paired advantage [deg]\nSingle-plane − Three-plane",
        "(b)",
        hide_outliers,
    )

    scale_note = (
        " (symmetric-log y-axis)"
        if not hide_outliers
        else ""
    )
    figure.text(
        0.5, 0.995,
        (
            "Positive values indicate lower error for three-plane "
            f"calibration{scale_note}"
        ),
        ha="center", va="top", fontsize=10,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()

    result_root = args.result_root.expanduser().resolve()

    if args.output is None:
        output_path = result_root / "noise_boxplots.png"
    else:
        output_path = args.output.expanduser()

    paired_output_path = (
        result_root / "noise_paired_differences.png"
        if args.paired_output is None
        else args.paired_output.expanduser()
    )
    paired_csv_path = (
        result_root / "noise_paired_differences.csv"
        if args.paired_csv is None
        else args.paired_csv.expanduser()
    )
    paired_summary_path = paired_csv_path.with_name(
        paired_csv_path.stem + "_summary.csv"
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    conditions = discover_noise_conditions(
        result_root,
        requested_noise_levels=args.noise_levels,
    )

    noise_levels = [
        noise_std
        for noise_std, _ in conditions
    ]

    translation_data = collect_metric(
        conditions=conditions,
        metric="translation_error_mm",
        success_only=args.success_only,
    )

    rotation_data = collect_metric(
        conditions=conditions,
        metric="rotation_error_deg",
        success_only=args.success_only,
    )

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12.0, 4.8),
    )

    draw_grouped_boxplot(
        axis=axes[0],
        noise_levels=noise_levels,
        metric_data=translation_data,
        ylabel="Translation error [mm]",
        panel_label="(a)",
        hide_outliers=args.hide_outliers,
        use_log_scale=args.log_translation,
    )

    draw_grouped_boxplot(
        axis=axes[1],
        noise_levels=noise_levels,
        metric_data=rotation_data,
        ylabel="Rotation error [deg]",
        panel_label="(b)",
        hide_outliers=args.hide_outliers,
        use_log_scale=args.log_rotation,
    )

    legend_handles = [
        plt.Rectangle(
            (0, 0),
            1,
            1,
            facecolor="#7FA6D8",
            edgecolor="#303030",
        ),
        plt.Rectangle(
            (0, 0),
            1,
            1,
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
        bbox_to_anchor=(0.5, 1.03),
    )

    figure.tight_layout(
        rect=(0.0, 0.0, 1.0, 0.94)
    )

    figure.savefig(
        output_path,
        dpi=args.dpi,
        bbox_inches="tight",
    )
    plt.close(figure)

    paired = collect_paired_data(
        conditions=conditions,
        success_only=args.success_only,
    )
    paired_csv_path.parent.mkdir(parents=True, exist_ok=True)
    paired.to_csv(paired_csv_path, index=False)
    paired_summary(paired).to_csv(paired_summary_path, index=False)

    save_paired_plot(
        paired=paired,
        noise_levels=noise_levels,
        output_path=paired_output_path,
        hide_outliers=args.hide_outliers,
        dpi=args.dpi,
    )

    print(f"Noise levels: {noise_levels}")
    print(f"Saved grouped boxplot : {output_path}")
    print(f"Saved paired plot     : {paired_output_path}")
    print(f"Saved paired rows     : {paired_csv_path}")
    print(f"Saved paired summary  : {paired_summary_path}")


if __name__ == "__main__":
    main()
