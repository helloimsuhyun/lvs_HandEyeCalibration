#!/usr/bin/env python3
"""Plot paired alternating-versus-joint-nonlinear calibration results."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


METHODS = (
    ("single_plane", "Single-plane"),
    ("three_plane", "Three-plane"),
)


def _as_bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


def _as_float(row: dict[str, str], field: str) -> float:
    try:
        return float(row.get(field, "nan"))
    except (TypeError, ValueError):
        return float("nan")


def _finite(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    return array[np.isfinite(array)]


def _load_trials(path: Path, nonlinear_success_only: bool) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"trials.csv not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    required = {
        "trial_index",
        "alternating_translation_error_mm",
        "alternating_rotation_error_deg",
        "translation_error_mm",
        "rotation_error_deg",
        "nonlinear_refined",
        "nonlinear_success",
    }
    missing = required.difference(rows[0] if rows else set())
    if missing:
        raise KeyError(f"{path} is missing columns: {sorted(missing)}")

    output: list[dict[str, str]] = []
    seen: set[int] = set()
    for row in rows:
        if str(row.get("error_type", "")).strip():
            continue
        if not _as_bool(row["nonlinear_refined"]):
            continue
        if nonlinear_success_only and not _as_bool(row["nonlinear_success"]):
            continue
        trial_index = int(row["trial_index"])
        if trial_index in seen:
            raise ValueError(f"duplicate trial_index={trial_index} in {path}")
        seen.add(trial_index)

        errors = [
            _as_float(row, "alternating_translation_error_mm"),
            _as_float(row, "alternating_rotation_error_deg"),
            _as_float(row, "translation_error_mm"),
            _as_float(row, "rotation_error_deg"),
        ]
        if all(np.isfinite(errors)):
            output.append(row)
    if not output:
        raise RuntimeError(f"no finite paired refinement rows in {path}")
    return output


def _paired(
    rows: list[dict[str, str]],
    before_field: str,
    after_field: str,
) -> tuple[np.ndarray, np.ndarray]:
    before = np.asarray([_as_float(row, before_field) for row in rows])
    after = np.asarray([_as_float(row, after_field) for row in rows])
    keep = np.isfinite(before) & np.isfinite(after)
    return before[keep], after[keep]


def _write_summary(
    output_path: Path,
    frames: dict[str, list[dict[str, str]]],
) -> None:
    fieldnames = [
        "method",
        "trials",
        "metric",
        "alternating_median",
        "joint_nonlinear_median",
        "median_improvement",
        "mean_improvement",
        "improved_fraction",
    ]
    metric_fields = (
        (
            "translation_error_mm",
            "alternating_translation_error_mm",
            "translation_error_mm",
        ),
        (
            "rotation_error_deg",
            "alternating_rotation_error_deg",
            "rotation_error_deg",
        ),
        (
            "plane_normal_error_deg",
            "alternating_plane_normal_error_deg",
            "nonlinear_plane_normal_error_deg",
        ),
        (
            "plane_offset_error_mm",
            "alternating_plane_offset_error_mm",
            "nonlinear_plane_offset_error_mm",
        ),
    )
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for method, display in METHODS:
            rows = frames[method]
            for metric, before_field, after_field in metric_fields:
                before, after = _paired(rows, before_field, after_field)
                if len(before) == 0:
                    continue
                improvement = before - after
                writer.writerow(
                    {
                        "method": display,
                        "trials": len(before),
                        "metric": metric,
                        "alternating_median": float(np.median(before)),
                        "joint_nonlinear_median": float(np.median(after)),
                        "median_improvement": float(np.median(improvement)),
                        "mean_improvement": float(np.mean(improvement)),
                        "improved_fraction": float(
                            np.mean(improvement > 0.0)
                        ),
                    }
                )


def _plot_paired_errors(
    output_path: Path,
    frames: dict[str, list[dict[str, str]]],
    dpi: int,
) -> None:
    metrics = (
        (
            "alternating_translation_error_mm",
            "translation_error_mm",
            "Translation error [mm]",
        ),
        (
            "alternating_rotation_error_deg",
            "rotation_error_deg",
            "Rotation error [deg]",
        ),
    )
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), constrained_layout=True)
    for row_index, (method, display) in enumerate(METHODS):
        rows = frames[method]
        for column_index, (before_field, after_field, ylabel) in enumerate(
            metrics
        ):
            axis = axes[row_index, column_index]
            before, after = _paired(rows, before_field, after_field)
            for left, right in zip(before, after, strict=True):
                axis.plot(
                    [0, 1],
                    [max(left, 1e-12), max(right, 1e-12)],
                    color="0.65",
                    linewidth=0.7,
                    alpha=0.35,
                )
            axis.plot(
                [0, 1],
                [np.median(before), np.median(after)],
                "o-",
                color="#d62728",
                linewidth=2.5,
                markersize=6,
                label="median",
            )
            axis.set_xticks([0, 1], ["Alternating", "Joint nonlinear"])
            axis.set_yscale("log")
            axis.set_ylabel(ylabel)
            axis.set_title(f"{display} (n={len(before)})")
            axis.grid(True, which="both", alpha=0.25)
            axis.legend(loc="best")
    fig.suptitle("Paired error change on the same noisy trial", fontsize=14)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def _plot_improvements(
    output_path: Path,
    frames: dict[str, list[dict[str, str]]],
    dpi: int,
) -> None:
    metrics = (
        (
            "alternating_translation_error_mm",
            "translation_error_mm",
            "Translation improvement [mm]",
        ),
        (
            "alternating_rotation_error_deg",
            "rotation_error_deg",
            "Rotation improvement [deg]",
        ),
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for axis, (before_field, after_field, ylabel) in zip(
        axes,
        metrics,
        strict=True,
    ):
        values = []
        labels = []
        for method, display in METHODS:
            before, after = _paired(
                frames[method],
                before_field,
                after_field,
            )
            values.append(before - after)
            labels.append(display)
        axis.boxplot(values, labels=labels, showmeans=True)
        axis.axhline(0.0, color="black", linewidth=1.0, linestyle="--")
        finite_groups = [
            value[np.isfinite(value)]
            for value in values
            if value.size > 0
        ]
        finite_values = (
            np.concatenate(finite_groups)
            if finite_groups
            else np.array([], dtype=float)
        )
        nonzero_abs = np.abs(
            finite_values[finite_values != 0.0]
        )
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
        axis.set_ylabel(ylabel)
        axis.set_title(
            "Positive means joint nonlinear is better (symmetric-log)"
        )
        axis.grid(True, axis="y", alpha=0.25)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def _plot_plane_errors(
    output_path: Path,
    frames: dict[str, list[dict[str, str]]],
    dpi: int,
) -> None:
    metrics = (
        (
            "alternating_plane_normal_error_deg",
            "nonlinear_plane_normal_error_deg",
            "Mean plane-normal error [deg]",
        ),
        (
            "alternating_plane_offset_error_mm",
            "nonlinear_plane_offset_error_mm",
            "Mean plane-offset error [mm]",
        ),
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for axis, (before_field, after_field, ylabel) in zip(
        axes,
        metrics,
        strict=True,
    ):
        data = []
        positions = []
        tick_positions = []
        tick_labels = []
        for method_index, (method, display) in enumerate(METHODS):
            before, after = _paired(
                frames[method],
                before_field,
                after_field,
            )
            base = 3 * method_index
            data.extend([before, after])
            positions.extend([base, base + 1])
            tick_positions.append(base + 0.5)
            tick_labels.append(display)
        box = axis.boxplot(data, positions=positions, patch_artist=True)
        for patch, color in zip(
            box["boxes"],
            ["#9ecae1", "#fb6a4a"] * len(METHODS),
            strict=True,
        ):
            patch.set_facecolor(color)
        axis.set_xticks(tick_positions, tick_labels)
        axis.set_ylabel(ylabel)
        axis.grid(True, axis="y", alpha=0.25)
        axis.legend(
            [box["boxes"][0], box["boxes"][1]],
            ["Alternating", "Joint nonlinear"],
            loc="best",
        )
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-root",
        type=Path,
        default=Path("results/alternating_joint_nonlinear_comparison"),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--nonlinear-success-only", action="store_true")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    output_dir = args.output_dir or args.result_root / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = {
        method: _load_trials(
            args.result_root / method / "trials.csv",
            args.nonlinear_success_only,
        )
        for method, _display in METHODS
    }

    summary_path = output_dir / "comparison_summary.csv"
    _write_summary(summary_path, frames)
    _plot_paired_errors(
        output_dir / "paired_calibration_errors.png",
        frames,
        args.dpi,
    )
    _plot_improvements(
        output_dir / "signed_error_improvements.png",
        frames,
        args.dpi,
    )
    _plot_plane_errors(
        output_dir / "plane_parameter_errors.png",
        frames,
        args.dpi,
    )

    for method, display in METHODS:
        rows = frames[method]
        before_t, after_t = _paired(
            rows,
            "alternating_translation_error_mm",
            "translation_error_mm",
        )
        before_r, after_r = _paired(
            rows,
            "alternating_rotation_error_deg",
            "rotation_error_deg",
        )
        full_rank = np.asarray(
            [
                _as_float(row, "nonlinear_jacobian_rank")
                == _as_float(row, "nonlinear_variable_count")
                for row in rows
            ],
            dtype=bool,
        )
        print(
            f"{display}: n={len(rows)}, "
            f"median t {np.median(before_t):.6g} -> "
            f"{np.median(after_t):.6g} mm, "
            f"median r {np.median(before_r):.6g} -> "
            f"{np.median(after_r):.6g} deg, "
            f"full-rank={np.mean(full_rank):.1%}"
        )
    print(f"saved: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
