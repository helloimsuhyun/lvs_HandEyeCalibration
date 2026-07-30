#!/usr/bin/env python3
"""Plot single/three-plane Fisher-versus-random calibration outcomes."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np


METHODS = (
    ("single_random", "Single\nRandom", "single", "random", "#8da0cb"),
    ("single_fisher", "Single\nFisher", "single", "fisher", "#4c72b0"),
    ("three_random", "Three\nRandom", "three", "random", "#fc8d62"),
    ("three_fisher", "Three\nFisher", "three", "fisher", "#dd4b39"),
)
NUMERIC_FIELDS = (
    "translation_error_mm",
    "rotation_error_deg",
    "iterations",
    "condition_last",
)
Rows = list[dict[str, Any]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare calibration outcomes under random and Fisher pose "
            "selection. Expected: RESULT_ROOT/<method>/trials.csv, where "
            "method is single_random, single_fisher, three_random, or "
            "three_fisher."
        )
    )
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Default: <result-root>/plots or plots_success_only.",
    )
    parser.add_argument(
        "--success-only",
        action="store_true",
        help="Restrict error distributions and paired error ratios to successes.",
    )
    parser.add_argument(
        "--log-errors",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use logarithmic axes for final-error boxplots "
            "(default: enabled; disable with --no-log-errors)."
        ),
    )
    parser.add_argument(
        "--hide-outliers",
        action="store_true",
        help="Hide boxplot fliers.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "y", "t"}


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _load_trials(path: Path) -> Rows:
    if not path.is_file():
        raise FileNotFoundError(f"trials.csv not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "trial_index" not in reader.fieldnames:
            raise KeyError(f"trial_index is missing: {path}")
        rows: Rows = []
        for raw_row in reader:
            trial_value = _as_float(raw_row.get("trial_index"))
            if not np.isfinite(trial_value):
                continue
            row: dict[str, Any] = dict(raw_row)
            row["trial_index"] = int(trial_value)
            for field in NUMERIC_FIELDS:
                if field in row:
                    row[field] = _as_float(row[field])
            rows.append(row)
    if not rows:
        raise RuntimeError(f"no trial rows: {path}")
    trial_ids = [int(row["trial_index"]) for row in rows]
    if len(set(trial_ids)) != len(trial_ids):
        raise RuntimeError(f"duplicate trial_index values: {path}")
    return rows


def _error_rows(rows: Rows, success_only: bool) -> Rows:
    result = []
    for row in rows:
        translation = _as_float(row.get("translation_error_mm"))
        rotation = _as_float(row.get("rotation_error_deg"))
        if not np.isfinite(translation) or not np.isfinite(rotation):
            continue
        if str(row.get("error_type", "")).strip():
            continue
        if success_only and not _as_bool(row.get("success", False)):
            continue
        result.append(row)
    return result


def _rate(rows: Rows, field: str) -> float:
    if not rows or not any(field in row for row in rows):
        return float("nan")
    return float(np.mean([_as_bool(row.get(field, False)) for row in rows]))


def _values(rows: Rows, field: str) -> np.ndarray:
    values = np.asarray([_as_float(row.get(field)) for row in rows])
    return values[np.isfinite(values)]


def _percentile(rows: Rows, field: str, percentile: float) -> float:
    values = _values(rows, field)
    if len(values) == 0:
        return float("nan")
    return float(np.percentile(values, percentile))


def _safe_mean(values: Sequence[bool]) -> float:
    return float(np.mean(values)) if len(values) else float("nan")


def _safe_median(values: np.ndarray) -> float:
    return float(np.median(values)) if len(values) else float("nan")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"cannot write an empty summary: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _load_all(result_root: Path) -> dict[str, Rows]:
    return {
        method: _load_trials(result_root / method / "trials.csv")
        for method, *_unused in METHODS
    }


def _save_method_summary(
    frames: dict[str, Rows],
    output_path: Path,
) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for method, label, geometry, policy, _color in METHODS:
        rows = frames[method]
        errors = _error_rows(rows, success_only=False)
        summary.append(
            {
                "method": method,
                "label": label.replace("\n", " "),
                "geometry": geometry,
                "policy": policy,
                "trial_count": len(rows),
                "valid_error_trial_count": len(errors),
                "success_rate": _rate(rows, "success"),
                "convergence_rate": _rate(rows, "converged"),
                "translation_error_mm_median": _percentile(
                    errors, "translation_error_mm", 50.0
                ),
                "translation_error_mm_p95": _percentile(
                    errors, "translation_error_mm", 95.0
                ),
                "rotation_error_deg_median": _percentile(
                    errors, "rotation_error_deg", 50.0
                ),
                "rotation_error_deg_p95": _percentile(
                    errors, "rotation_error_deg", 95.0
                ),
            }
        )
    _write_csv(output_path, summary)
    return summary


def _paired_errors(
    frames: dict[str, Rows],
    geometry: str,
    success_only: bool,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    random_rows = {
        int(row["trial_index"]): row
        for row in _error_rows(
            frames[f"{geometry}_random"],
            success_only=success_only,
        )
    }
    fisher_rows = {
        int(row["trial_index"]): row
        for row in _error_rows(
            frames[f"{geometry}_fisher"],
            success_only=success_only,
        )
    }
    return [
        (random_rows[trial_id], fisher_rows[trial_id])
        for trial_id in sorted(random_rows.keys() & fisher_rows.keys())
    ]


def _success_pairs(
    frames: dict[str, Rows],
    geometry: str,
) -> list[tuple[bool, bool]]:
    random_rows = {
        int(row["trial_index"]): row
        for row in frames[f"{geometry}_random"]
    }
    fisher_rows = {
        int(row["trial_index"]): row
        for row in frames[f"{geometry}_fisher"]
    }
    return [
        (
            _as_bool(random_rows[trial_id].get("success", False)),
            _as_bool(fisher_rows[trial_id].get("success", False)),
        )
        for trial_id in sorted(random_rows.keys() & fisher_rows.keys())
    ]


def _error_ratios(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    field: str,
) -> np.ndarray:
    epsilon = 1e-12
    return np.asarray(
        [
            max(_as_float(random_row[field]), epsilon)
            / max(_as_float(fisher_row[field]), epsilon)
            for random_row, fisher_row in pairs
        ],
        dtype=float,
    )


def _save_paired_summary(
    frames: dict[str, Rows],
    output_path: Path,
) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for geometry in ("single", "three"):
        pairs = _paired_errors(frames, geometry, success_only=False)
        success_pairs = _success_pairs(frames, geometry)
        translation_ratio = _error_ratios(
            pairs, "translation_error_mm"
        )
        rotation_ratio = _error_ratios(pairs, "rotation_error_deg")
        summary.append(
            {
                "geometry": geometry,
                "paired_error_trial_count": len(pairs),
                "fisher_translation_win_rate": _safe_mean(
                    [
                        _as_float(fisher["translation_error_mm"])
                        < _as_float(random["translation_error_mm"])
                        for random, fisher in pairs
                    ]
                ),
                "fisher_rotation_win_rate": _safe_mean(
                    [
                        _as_float(fisher["rotation_error_deg"])
                        < _as_float(random["rotation_error_deg"])
                        for random, fisher in pairs
                    ]
                ),
                "translation_random_over_fisher_ratio_median": (
                    _safe_median(translation_ratio)
                ),
                "rotation_random_over_fisher_ratio_median": (
                    _safe_median(rotation_ratio)
                ),
                "fisher_only_success_rate": _safe_mean(
                    [
                        fisher_success and not random_success
                        for random_success, fisher_success in success_pairs
                    ]
                ),
                "random_only_success_rate": _safe_mean(
                    [
                        random_success and not fisher_success
                        for random_success, fisher_success in success_pairs
                    ]
                ),
                "both_success_rate": _safe_mean(
                    [
                        random_success and fisher_success
                        for random_success, fisher_success in success_pairs
                    ]
                ),
            }
        )
    _write_csv(output_path, summary)
    return summary


def _boxplot(
    axis: plt.Axes,
    values: list[np.ndarray],
    labels: list[str],
    colors: list[str],
    *,
    hide_outliers: bool,
) -> None:
    artists = axis.boxplot(
        values,
        labels=labels,
        patch_artist=True,
        showfliers=not hide_outliers,
    )
    for patch, color in zip(artists["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)


def _plot_final_errors(
    frames: dict[str, Rows],
    output_path: Path,
    *,
    success_only: bool,
    log_errors: bool,
    hide_outliers: bool,
    dpi: int,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    metrics = (
        ("translation_error_mm", "Translation error [mm]"),
        ("rotation_error_deg", "Rotation error [deg]"),
    )
    colors = [entry[4] for entry in METHODS]
    labels = [entry[1] for entry in METHODS]
    for axis, (metric, ylabel) in zip(axes, metrics):
        values = []
        for method, *_unused in METHODS:
            data = _values(
                _error_rows(frames[method], success_only),
                metric,
            )
            if log_errors:
                data = data[data > 0.0]
            values.append(data)
        _boxplot(
            axis,
            values,
            labels,
            colors,
            hide_outliers=hide_outliers,
        )
        if log_errors:
            axis.set_yscale("log")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
    suffix = " (successful trials)" if success_only else ""
    figure.suptitle(f"Fisher versus random pose selection{suffix}")
    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def _plot_rates(
    frames: dict[str, Rows],
    output_path: Path,
    *,
    dpi: int,
) -> None:
    labels = [entry[1] for entry in METHODS]
    colors = [entry[4] for entry in METHODS]
    success = [_rate(frames[entry[0]], "success") for entry in METHODS]
    convergence = [
        _rate(frames[entry[0]], "converged") for entry in METHODS
    ]
    x_values = np.arange(len(METHODS), dtype=float)
    width = 0.36
    figure, axis = plt.subplots(figsize=(8.6, 4.8))
    axis.bar(
        x_values - width / 2.0,
        success,
        width,
        label="Success",
        color=colors,
    )
    axis.bar(
        x_values + width / 2.0,
        convergence,
        width,
        label="Converged",
        color=colors,
        alpha=0.45,
        hatch="//",
    )
    axis.set_xticks(x_values, labels)
    axis.set_ylim(0.0, 1.05)
    axis.set_ylabel("Rate")
    axis.set_title("Calibration success and convergence rates")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def _plot_paired_ratios(
    frames: dict[str, Rows],
    output_path: Path,
    *,
    success_only: bool,
    hide_outliers: bool,
    dpi: int,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), sharey=True)
    metrics = (
        ("translation_error_mm", "Translation"),
        ("rotation_error_deg", "Rotation"),
    )
    for axis, (metric, title) in zip(axes, metrics):
        values = [
            np.log10(
                _error_ratios(
                    _paired_errors(frames, geometry, success_only),
                    metric,
                )
            )
            for geometry in ("single", "three")
        ]
        _boxplot(
            axis,
            values,
            ["Single", "Three"],
            ["#4c72b0", "#dd4b39"],
            hide_outliers=hide_outliers,
        )
        axis.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel(
        r"$\log_{10}(\mathrm{random\ error}/\mathrm{Fisher\ error})$"
        "\npositive = Fisher is better"
    )
    suffix = " (successful pairs)" if success_only else ""
    figure.suptitle(f"Paired policy advantage{suffix}")
    figure.tight_layout()
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def _print_table(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(rows[0])
    print(" | ".join(fields))
    for row in rows:
        print(" | ".join(str(row[field]) for field in fields))


def main() -> int:
    args = parse_args()
    if args.dpi <= 0:
        raise SystemExit("--dpi must be positive")
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = args.result_root / (
            "plots_success_only" if args.success_only else "plots"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = _load_all(args.result_root)
    method_summary = _save_method_summary(
        frames,
        output_dir / "method_summary.csv",
    )
    paired_summary = _save_paired_summary(
        frames,
        output_dir / "paired_policy_summary.csv",
    )
    _plot_final_errors(
        frames,
        output_dir / "final_error_boxplots.png",
        success_only=args.success_only,
        log_errors=args.log_errors,
        hide_outliers=args.hide_outliers,
        dpi=args.dpi,
    )
    _plot_rates(
        frames,
        output_dir / "success_convergence_rates.png",
        dpi=args.dpi,
    )
    _plot_paired_ratios(
        frames,
        output_dir / "paired_policy_ratios.png",
        success_only=args.success_only,
        hide_outliers=args.hide_outliers,
        dpi=args.dpi,
    )

    print("Method summary:")
    _print_table(method_summary)
    print("\nPaired Fisher-vs-random summary:")
    _print_table(paired_summary)
    print(f"\nPlots and CSV summaries: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
