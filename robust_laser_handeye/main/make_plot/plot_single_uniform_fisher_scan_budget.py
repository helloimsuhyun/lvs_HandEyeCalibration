#!/usr/bin/env python3
"""Plot paired Single Uniform/Fisher calibration performance versus scan budget."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np


METHODS = (
    ("single_uniform", "Uniform maximin", "#4C78A8"),
    ("single_fisher", "Active Fisher", "#B279A2"),
)
NUMERIC_FIELDS = (
    "trial_index",
    "translation_error_mm",
    "rotation_error_deg",
    "iterations",
    "condition_last",
    "n_scans",
)
Rows = list[dict[str, Any]]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--scan-counts", type=int, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--success-only", action="store_true")
    parser.add_argument("--hide-outliers", action="store_true")
    parser.add_argument(
        "--log-errors",
        action=argparse.BooleanOptionalAction,
        default=False,
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


def _load_rows(path: Path) -> Rows:
    if not path.is_file():
        raise FileNotFoundError(f"trials.csv not found: {path}")
    rows: Rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
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
        raise ValueError(f"empty or duplicate trial rows: {path}")
    return rows


def _valid_errors(rows: Rows, success_only: bool) -> Rows:
    output = []
    for row in rows:
        if str(row.get("error_type", "")).strip():
            continue
        if success_only and not _as_bool(row.get("success", False)):
            continue
        if not (
            np.isfinite(_as_float(row.get("translation_error_mm")))
            and np.isfinite(_as_float(row.get("rotation_error_deg")))
        ):
            continue
        output.append(row)
    return output


def _values(rows: Rows, field: str, positive: bool = False) -> np.ndarray:
    values = np.asarray([_as_float(row.get(field)) for row in rows], dtype=float)
    values = values[np.isfinite(values)]
    return values[values > 0.0] if positive else values


def _quartiles(values: np.ndarray) -> tuple[float, float, float]:
    if not values.size:
        return float("nan"), float("nan"), float("nan")
    return (
        float(np.median(values)),
        float(np.percentile(values, 25)),
        float(np.percentile(values, 75)),
    )


def _paired_rows(
    frames: dict[int, dict[str, Rows]],
    scan_counts: Sequence[int],
    success_only: bool,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for scan_count in scan_counts:
        uniform = {
            int(row["trial_index"]): row
            for row in _valid_errors(
                frames[scan_count]["single_uniform"],
                success_only,
            )
        }
        fisher = {
            int(row["trial_index"]): row
            for row in _valid_errors(
                frames[scan_count]["single_fisher"],
                success_only,
            )
        }
        for trial_index in sorted(uniform.keys() & fisher.keys()):
            left = uniform[trial_index]
            right = fisher[trial_index]
            output.append(
                {
                    "scan_count": scan_count,
                    "trial_index": trial_index,
                    "translation_uniform_mm": _as_float(
                        left["translation_error_mm"]
                    ),
                    "translation_fisher_mm": _as_float(
                        right["translation_error_mm"]
                    ),
                    "rotation_uniform_deg": _as_float(
                        left["rotation_error_deg"]
                    ),
                    "rotation_fisher_deg": _as_float(
                        right["rotation_error_deg"]
                    ),
                    "translation_advantage_mm": (
                        _as_float(left["translation_error_mm"])
                        - _as_float(right["translation_error_mm"])
                    ),
                    "rotation_advantage_deg": (
                        _as_float(left["rotation_error_deg"])
                        - _as_float(right["rotation_error_deg"])
                    ),
                }
            )
    return output


def _save_error_curves(
    frames: dict[int, dict[str, Rows]],
    scan_counts: Sequence[int],
    output: Path,
    args: argparse.Namespace,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.9))
    for axis, field, ylabel, panel in (
        (axes[0], "translation_error_mm", "Translation error [mm]", "(a)"),
        (axes[1], "rotation_error_deg", "Rotation error [deg]", "(b)"),
    ):
        for method, label, color in METHODS:
            quartiles = [
                _quartiles(
                    _values(
                        _valid_errors(
                            frames[count][method],
                            args.success_only,
                        ),
                        field,
                        positive=args.log_errors,
                    )
                )
                for count in scan_counts
            ]
            median = np.asarray([value[0] for value in quartiles])
            lower = np.asarray([value[1] for value in quartiles])
            upper = np.asarray([value[2] for value in quartiles])
            axis.plot(
                scan_counts,
                median,
                marker="o",
                linewidth=2.1,
                color=color,
                label=label,
            )
            axis.fill_between(
                scan_counts,
                lower,
                upper,
                color=color,
                alpha=0.17,
                linewidth=0,
            )
        if args.log_errors:
            axis.set_yscale("log")
            ylabel += " (log scale)"
        axis.set_xlabel("Number of acquired scan lines")
        axis.set_ylabel(ylabel)
        axis.set_xticks(scan_counts)
        axis.grid(linestyle="--", alpha=0.32)
        axis.set_axisbelow(True)
        axis.text(
            0.01,
            0.98,
            panel,
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontweight="bold",
        )
    axes[0].legend(frameon=False)
    figure.suptitle(
        "Single-plane pose-selection efficiency versus scan budget\n"
        "lines: median; bands: interquartile range"
    )
    figure.tight_layout()
    figure.savefig(output, dpi=args.dpi, bbox_inches="tight")
    plt.close(figure)


def _boxplot(
    axis: plt.Axes,
    values: list[np.ndarray],
    labels: list[str],
    hide_outliers: bool,
) -> None:
    artists = axis.boxplot(
        values,
        labels=labels,
        patch_artist=True,
        showfliers=not hide_outliers,
        showmeans=not hide_outliers,
        meanprops={
            "marker": "D",
            "markerfacecolor": "white",
            "markeredgecolor": "#222222",
            "markersize": 3.8,
        },
    )
    for box in artists["boxes"]:
        box.set_facecolor("#B279A2")
        box.set_alpha(0.68)
    for median in artists["medians"]:
        median.set_color("#111111")
        median.set_linewidth(1.7)
    for flier in artists["fliers"]:
        flier.set(
            marker="o",
            markersize=3.0,
            markerfacecolor="none",
            markeredgecolor="#555555",
            alpha=0.45,
        )


def _save_paired_advantage(
    paired: list[dict[str, Any]],
    scan_counts: Sequence[int],
    output: Path,
    args: argparse.Namespace,
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.9))
    for axis, field, ylabel, panel in (
        (
            axes[0],
            "translation_advantage_mm",
            "Paired advantage [mm]\nUniform − Fisher",
            "(a)",
        ),
        (
            axes[1],
            "rotation_advantage_deg",
            "Paired advantage [deg]\nUniform − Fisher",
            "(b)",
        ),
    ):
        values = [
            _values(
                [
                    row
                    for row in paired
                    if int(row["scan_count"]) == scan_count
                ],
                field,
            )
            for scan_count in scan_counts
        ]
        _boxplot(
            axis,
            values,
            [str(value) for value in scan_counts],
            args.hide_outliers,
        )
        if not args.hide_outliers:
            nonempty = [value for value in values if value.size]
            combined = (
                np.concatenate(nonempty)
                if nonempty
                else np.asarray([], dtype=float)
            )
            nonzero = np.abs(combined[combined != 0.0])
            if combined.size and nonzero.size:
                axis.set_yscale(
                    "symlog",
                    linthresh=max(
                        float(np.percentile(nonzero, 75)),
                        np.finfo(float).tiny,
                    ),
                )
        axis.axhline(0.0, color="#222222", linestyle="--", linewidth=1.1)
        axis.set_xlabel("Number of acquired scan lines")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", linestyle="--", alpha=0.32)
        axis.set_axisbelow(True)
        axis.text(
            0.01,
            0.98,
            panel,
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontweight="bold",
        )
    figure.suptitle("Positive values mean active Fisher has lower error")
    figure.tight_layout()
    figure.savefig(output, dpi=args.dpi, bbox_inches="tight")
    plt.close(figure)


def _rate(rows: Rows, field: str) -> float:
    return float(np.mean([_as_bool(row.get(field, False)) for row in rows]))


def _save_rates(
    frames: dict[int, dict[str, Rows]],
    scan_counts: Sequence[int],
    output: Path,
    dpi: int,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(14.0, 4.5))
    for axis, field, title in (
        (axes[0], "converged", "Convergence rate"),
        (axes[1], "success", "Success rate"),
        (axes[2], "outlier", "Outlier rate"),
    ):
        for method, label, color in METHODS:
            values = [
                100.0 * _rate(frames[count][method], field)
                for count in scan_counts
            ]
            axis.plot(
                scan_counts,
                values,
                marker="o",
                linewidth=2.0,
                color=color,
                label=label,
            )
        axis.set_title(title)
        axis.set_xlabel("Scan lines")
        axis.set_ylabel("Rate [%]")
        axis.set_xticks(scan_counts)
        axis.set_ylim(-2.0, 102.0)
        axis.grid(linestyle="--", alpha=0.3)
        axis.set_axisbelow(True)
    axes[0].legend(frameon=False)
    figure.tight_layout()
    figure.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def _summaries(
    frames: dict[int, dict[str, Rows]],
    paired: list[dict[str, Any]],
    scan_counts: Sequence[int],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for scan_count in scan_counts:
        paired_at_count = [
            row
            for row in paired
            if int(row["scan_count"]) == scan_count
        ]
        for method, label, _color in METHODS:
            rows = frames[scan_count][method]
            errors = _valid_errors(rows, success_only=False)
            t_median, t_q25, t_q75 = _quartiles(
                _values(errors, "translation_error_mm")
            )
            r_median, r_q25, r_q75 = _quartiles(
                _values(errors, "rotation_error_deg")
            )
            selected_t = _values(
                paired_at_count,
                "translation_advantage_mm",
            )
            selected_r = _values(
                paired_at_count,
                "rotation_advantage_deg",
            )
            output.append(
                {
                    "scan_count": scan_count,
                    "method": method,
                    "label": label,
                    "trials": len(rows),
                    "translation_median_mm": t_median,
                    "translation_q25_mm": t_q25,
                    "translation_q75_mm": t_q75,
                    "rotation_median_deg": r_median,
                    "rotation_q25_deg": r_q25,
                    "rotation_q75_deg": r_q75,
                    "convergence_rate": _rate(rows, "converged"),
                    "success_rate": _rate(rows, "success"),
                    "outlier_rate": _rate(rows, "outlier"),
                    "fisher_translation_better_fraction": (
                        float(np.mean(selected_t > 0.0))
                        if selected_t.size
                        else float("nan")
                    ),
                    "fisher_rotation_better_fraction": (
                        float(np.mean(selected_r > 0.0))
                        if selected_r.size
                        else float("nan")
                    ),
                }
            )
    return output


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _bootstrap_check(
    paired: list[dict[str, Any]],
    bootstrap_count: int,
) -> dict[str, Any]:
    selected = [
        row
        for row in paired
        if int(row["scan_count"]) == bootstrap_count
    ]
    translation = _values(selected, "translation_advantage_mm")
    rotation = _values(selected, "rotation_advantage_deg")
    maximum_translation = (
        float(np.max(np.abs(translation))) if translation.size else None
    )
    maximum_rotation = (
        float(np.max(np.abs(rotation))) if rotation.size else None
    )
    passed = bool(
        translation.size
        and rotation.size
        and maximum_translation is not None
        and maximum_rotation is not None
        and maximum_translation <= 1e-10
        and maximum_rotation <= 1e-10
    )
    return {
        "bootstrap_scan_count": bootstrap_count,
        "paired_trials": len(selected),
        "max_abs_translation_difference_mm": maximum_translation,
        "max_abs_rotation_difference_deg": maximum_rotation,
        "passed": passed,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    scan_counts = list(dict.fromkeys(args.scan_counts))
    if any(value <= 0 for value in scan_counts):
        raise SystemExit("--scan-counts must be positive")
    result_root = args.result_root.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else result_root / "plots"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = {
        scan_count: {
            method: _load_rows(
                result_root
                / f"N{scan_count}"
                / method
                / "trials.csv"
            )
            for method, _label, _color in METHODS
        }
        for scan_count in scan_counts
    }
    paired_all = _paired_rows(frames, scan_counts, success_only=False)
    paired = (
        _paired_rows(frames, scan_counts, success_only=True)
        if args.success_only
        else paired_all
    )

    _save_error_curves(
        frames,
        scan_counts,
        output_dir / "pose_selection_scan_budget_error_curves.png",
        args,
    )
    _save_paired_advantage(
        paired,
        scan_counts,
        output_dir / "pose_selection_scan_budget_paired_advantage.png",
        args,
    )
    _save_rates(
        frames,
        scan_counts,
        output_dir / "pose_selection_scan_budget_rates.png",
        args.dpi,
    )
    _write_csv(
        output_dir / "pose_selection_scan_budget_paired_rows.csv",
        paired,
    )
    _write_csv(
        output_dir / "pose_selection_scan_budget_summary.csv",
        _summaries(frames, paired_all, scan_counts),
    )
    bootstrap = _bootstrap_check(paired_all, min(scan_counts))
    (
        output_dir / "pose_selection_bootstrap_identity_check.json"
    ).write_text(
        json.dumps(bootstrap, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if not bootstrap["passed"]:
        raise RuntimeError(
            "Uniform/Fisher bootstrap identity check failed: "
            f"{bootstrap}"
        )
    print(f"Saved scan-budget comparison plots: {output_dir}")
    print(f"Bootstrap identity check: {bootstrap}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
