#!/usr/bin/env python3
"""
PYTHONPATH=. python3 main/make_plot/plot_fixed_noise_line_count_boxplots.py \
  --result-root results/fair_plane_line_count_shared_global
"""


from __future__ import annotations

import argparse
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
            "Create grouped boxplots and paired-difference plots for fixed-noise "
            "line-count experiments. Expected structure: "
            "RESULT_ROOT/N*/{single_plane,three_plane}/trials.csv"
        )
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=Path("results/fair_plane_line_count_shared_global"),
    )
    parser.add_argument(
        "--scan-counts",
        nargs="+",
        type=int,
        default=None,
        help=(
            "Scan counts to include, in plotting order. "
            "When omitted, all complete N* directories are discovered."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Default: <result-root>/line_count_boxplots.png",
    )
    parser.add_argument(
        "--paired-output",
        type=Path,
        default=None,
        help="Default: <result-root>/line_count_paired_differences.png",
    )
    parser.add_argument(
        "--paired-csv",
        type=Path,
        default=None,
        help="Default: <result-root>/line_count_paired_differences.csv",
    )
    parser.add_argument(
        "--success-only",
        action="store_true",
        help="Use only trials whose success column is true.",
    )
    outlier_group = parser.add_mutually_exclusive_group()
    outlier_group.add_argument(
        "--hide-outliers",
        dest="hide_outliers",
        action="store_true",
        help=(
            "Hide individual boxplot fliers. This is the default so a rare "
            "catastrophic calibration does not collapse the useful y-axis."
        ),
    )
    outlier_group.add_argument(
        "--show-outliers",
        dest="hide_outliers",
        action="store_false",
        help=(
            "Show individual boxplot fliers and include them in axis "
            "autoscaling."
        ),
    )
    parser.set_defaults(hide_outliers=True)
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
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes"})
    )


def load_trials(path: Path, success_only: bool) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"trials.csv not found: {path}")

    frame = pd.read_csv(path)

    required = {
        "trial_index",
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
        ]

    if success_only:
        if "success" not in frame.columns:
            raise KeyError(
                f"--success-only requested, but success column is missing: {path}"
            )
        frame = frame[bool_series(frame["success"])]

    numeric_columns = [
        "trial_index",
        "translation_error_mm",
        "rotation_error_deg",
    ]
    optional_numeric = [
        "init_translation_error_mm",
        "init_rotation_error_deg",
    ]

    for column in numeric_columns + [
        name for name in optional_numeric if name in frame.columns
    ]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame[
        np.isfinite(frame["trial_index"])
        & np.isfinite(frame["translation_error_mm"])
        & np.isfinite(frame["rotation_error_deg"])
    ].copy()

    frame["trial_index"] = frame["trial_index"].astype(int)

    if frame["trial_index"].duplicated().any():
        duplicates = frame.loc[
            frame["trial_index"].duplicated(keep=False),
            "trial_index",
        ].tolist()
        raise RuntimeError(
            f"Duplicate trial_index values in {path}: {duplicates[:10]}"
        )

    return frame


def discover_scan_counts(
    result_root: Path,
    requested_scan_counts: list[int] | None = None,
) -> list[int]:
    if not result_root.is_dir():
        raise FileNotFoundError(f"Result root not found: {result_root}")

    counts: list[int] = []

    for path in result_root.iterdir():
        if not path.is_dir() or not path.name.startswith("N"):
            continue

        try:
            count = int(path.name[1:])
        except ValueError:
            continue

        complete = all(
            (path / method_dir / "trials.csv").is_file()
            for method_dir, _ in METHODS
        )
        if complete:
            counts.append(count)

    counts.sort()

    if not counts:
        raise RuntimeError(
            "No complete N*/single_plane/trials.csv and "
            f"N*/three_plane/trials.csv pairs found under {result_root}"
        )

    if requested_scan_counts is not None:
        missing = [
            count for count in requested_scan_counts if count not in counts
        ]
        if missing:
            raise RuntimeError(
                f"Requested scan-count results are incomplete or missing "
                f"under {result_root}: {missing}"
            )
        counts = list(requested_scan_counts)

    return counts


def collect_metric(
    result_root: Path,
    scan_counts: list[int],
    column: str,
    success_only: bool,
) -> dict[str, list[np.ndarray]]:
    result = {method_dir: [] for method_dir, _ in METHODS}

    for count in scan_counts:
        for method_dir, _ in METHODS:
            csv_path = result_root / f"N{count}" / method_dir / "trials.csv"
            frame = load_trials(csv_path, success_only)
            result[method_dir].append(frame[column].to_numpy(dtype=float))

    return result


def pair_count_condition(
    result_root: Path,
    scan_count: int,
    success_only: bool,
) -> pd.DataFrame:
    single = load_trials(
        result_root / f"N{scan_count}" / "single_plane" / "trials.csv",
        success_only,
    )
    three = load_trials(
        result_root / f"N{scan_count}" / "three_plane" / "trials.csv",
        success_only,
    )

    base = [
        "trial_index",
        "translation_error_mm",
        "rotation_error_deg",
    ]
    optional = [
        "init_translation_error_mm",
        "init_rotation_error_deg",
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

    paired["scan_count"] = int(scan_count)
    paired["translation_advantage_mm"] = (
        paired["translation_error_mm_single"]
        - paired["translation_error_mm_three"]
    )
    paired["rotation_advantage_deg"] = (
        paired["rotation_error_deg_single"]
        - paired["rotation_error_deg_three"]
    )

    for column in optional:
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
                f"warning: N{scan_count}: {int(mismatch.sum())} paired rows "
                f"have different {column}"
            )

    return paired


def collect_paired_data(
    result_root: Path,
    scan_counts: list[int],
    success_only: bool,
) -> pd.DataFrame:
    frames = [
        pair_count_condition(
            result_root=result_root,
            scan_count=count,
            success_only=success_only,
        )
        for count in scan_counts
    ]
    return pd.concat(frames, ignore_index=True)


def paired_summary(paired: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "scan_count",
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
    rows: list[dict[str, float | int]] = []

    for scan_count, group in paired.groupby("scan_count", sort=True):
        row: dict[str, float | int] = {
            "scan_count": int(scan_count),
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


def draw_grouped_boxes(
    axis: plt.Axes,
    scan_counts: list[int],
    metric_data: dict[str, list[np.ndarray]],
    ylabel: str,
    panel_label: str,
    hide_outliers: bool,
    use_log: bool,
) -> None:
    centers = np.arange(len(scan_counts), dtype=float)
    offset = 0.18
    width = 0.30

    single = axis.boxplot(
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
    three = axis.boxplot(
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

    style_boxplot(single, "#7FA6D8")
    style_boxplot(three, "#E6A36A")

    axis.set_xticks(centers)
    axis.set_xticklabels([str(value) for value in scan_counts])
    axis.set_xlabel("Total number of lines")
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


def draw_paired_boxes(
    axis: plt.Axes,
    scan_counts: list[int],
    paired: pd.DataFrame,
    value_column: str,
    ylabel: str,
    panel_label: str,
    hide_outliers: bool,
) -> None:
    data: list[np.ndarray] = []

    for scan_count in scan_counts:
        values = pd.to_numeric(
            paired.loc[
                paired["scan_count"].eq(scan_count),
                value_column,
            ],
            errors="coerce",
        ).to_numpy()
        data.append(values[np.isfinite(values)])

    positions = np.arange(len(scan_counts), dtype=float)

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

    axis.axhline(
        0.0,
        color="#202020",
        linestyle="--",
        linewidth=1.1,
    )
    axis.set_xticks(positions)
    axis.set_xticklabels([str(value) for value in scan_counts])
    axis.set_xlabel("Total number of lines")
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
        nonzero_abs = np.abs(finite_values[finite_values != 0.0])
        if nonzero_abs.size > 0:
            linear_threshold = max(
                float(np.percentile(nonzero_abs, 75)),
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


def save_grouped_plot(
    *,
    output: Path,
    scan_counts: list[int],
    translation: dict[str, list[np.ndarray]],
    rotation: dict[str, list[np.ndarray]],
    hide_outliers: bool,
    log_translation: bool,
    log_rotation: bool,
    dpi: int,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))

    draw_grouped_boxes(
        axes[0],
        scan_counts,
        translation,
        "Translation error [mm]",
        "(a)",
        hide_outliers,
        log_translation,
    )
    draw_grouped_boxes(
        axes[1],
        scan_counts,
        rotation,
        "Rotation error [deg]",
        "(b)",
        hide_outliers,
        log_rotation,
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
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def save_paired_plot(
    *,
    output: Path,
    scan_counts: list[int],
    paired: pd.DataFrame,
    hide_outliers: bool,
    dpi: int,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))

    draw_paired_boxes(
        axes[0],
        scan_counts,
        paired,
        "translation_advantage_mm",
        "Paired advantage [mm]\nSingle-plane − Three-plane",
        "(a)",
        hide_outliers,
    )
    draw_paired_boxes(
        axes[1],
        scan_counts,
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
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()

    result_root = args.result_root.expanduser().resolve()

    output = (
        args.output.expanduser()
        if args.output is not None
        else result_root / "line_count_boxplots.png"
    )
    paired_output = (
        args.paired_output.expanduser()
        if args.paired_output is not None
        else result_root / "line_count_paired_differences.png"
    )
    paired_csv = (
        args.paired_csv.expanduser()
        if args.paired_csv is not None
        else result_root / "line_count_paired_differences.csv"
    )
    paired_summary_csv = paired_csv.with_name(
        paired_csv.stem + "_summary.csv"
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    paired_output.parent.mkdir(parents=True, exist_ok=True)
    paired_csv.parent.mkdir(parents=True, exist_ok=True)

    scan_counts = discover_scan_counts(
        result_root,
        requested_scan_counts=args.scan_counts,
    )

    translation = collect_metric(
        result_root,
        scan_counts,
        "translation_error_mm",
        args.success_only,
    )
    rotation = collect_metric(
        result_root,
        scan_counts,
        "rotation_error_deg",
        args.success_only,
    )

    save_grouped_plot(
        output=output,
        scan_counts=scan_counts,
        translation=translation,
        rotation=rotation,
        hide_outliers=args.hide_outliers,
        log_translation=args.log_translation,
        log_rotation=args.log_rotation,
        dpi=args.dpi,
    )

    paired = collect_paired_data(
        result_root=result_root,
        scan_counts=scan_counts,
        success_only=args.success_only,
    )
    paired.to_csv(paired_csv, index=False)
    paired_summary(paired).to_csv(paired_summary_csv, index=False)

    save_paired_plot(
        output=paired_output,
        scan_counts=scan_counts,
        paired=paired,
        hide_outliers=args.hide_outliers,
        dpi=args.dpi,
    )

    print(f"Scan counts          : {scan_counts}")
    print(f"Saved grouped plot   : {output}")
    print(f"Saved paired plot    : {paired_output}")
    print(f"Saved paired rows    : {paired_csv}")
    print(f"Saved paired summary : {paired_summary_csv}")


if __name__ == "__main__":
    main()
