"""Plot all fresh-seed candidate distributions, one box per POSE_COUNTS."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch


RESULT_DIR = Path("runs/pose_counts_candidate_ranking_robustness")
TRIAL_CSV = RESULT_DIR / "fresh_seed_trials.csv"
SUMMARY_CSV = RESULT_DIR / "robustness_summary.csv"
OUTPUT_PDF = RESULT_DIR / "all_candidates_fresh_seed_error_boxplots.pdf"

STATUS_COLORS = {
    "robust_pass": "#59A14F",
    "uncertain": "#F28E2B",
    "robust_fail": "#E15759",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def make_figure(
    n: int,
    summaries: list[dict[str, str]],
    trials: list[dict[str, str]],
) -> plt.Figure:
    candidates = sorted(
        (row for row in summaries if int(row["N"]) == n),
        key=lambda row: tuple(int(value) for value in row["pose_counts"].split()),
    )
    if not candidates:
        raise ValueError(f"No candidates for N={n}")

    labels: list[str] = []
    statuses: list[str] = []
    translation_data: list[list[float]] = []
    rotation_data: list[list[float]] = []
    translation_p95: list[float] = []
    rotation_p95: list[float] = []

    for candidate in candidates:
        counts = candidate["pose_counts"]
        matching = [
            row
            for row in trials
            if int(row["N"]) == n
            and row["pose_counts"] == counts
            and int(row["solver_success"]) == 1
        ]
        translation = [float(row["translation_error_mm"]) for row in matching]
        rotation = [float(row["rotation_error_deg"]) for row in matching]
        labels.append(f"({counts.replace(' ', ',')})")
        statuses.append(candidate["classification"])
        translation_data.append(translation)
        rotation_data.append(rotation)
        translation_p95.append(float(np.percentile(translation, 95.0)))
        rotation_p95.append(float(np.percentile(rotation, 95.0)))

    width = max(12.0, 0.82 * len(candidates))
    figure, axes = plt.subplots(2, 1, figsize=(width, 12.0), sharex=True)
    positions = np.arange(1, len(candidates) + 1)

    for axis, data, p95_values, ylabel in (
        (axes[0], translation_data, translation_p95, "Translation error [mm]"),
        (axes[1], rotation_data, rotation_p95, "Rotation error [deg]"),
    ):
        boxplot = axis.boxplot(
            data,
            labels=labels,
            widths=0.58,
            patch_artist=True,
            showfliers=True,
            flierprops={"markersize": 3.0, "alpha": 0.38},
            medianprops={"color": "black", "linewidth": 1.4},
        )
        for patch, status in zip(boxplot["boxes"], statuses):
            patch.set_facecolor(STATUS_COLORS[status])
            patch.set_alpha(0.70)

        axis.scatter(
            positions,
            p95_values,
            marker="D",
            s=31,
            color="#7A0019",
            edgecolor="white",
            linewidth=0.5,
            zorder=4,
        )
        axis.axhline(1.0, color="#C00000", linestyle="--", linewidth=1.3)
        axis.set_yscale("log")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", which="both", alpha=0.22)

    axes[0].set_title(f"N={n}: all candidate translation distributions (500 fresh trials)")
    axes[1].set_title(f"N={n}: all candidate rotation distributions (500 fresh trials)")
    axes[1].set_xlabel("POSE_COUNTS (5/60, 5/120, 40/60, 40/120)")
    axes[1].tick_params(axis="x", rotation=90, labelsize=8)
    axes[1].legend(
        handles=[
            Patch(facecolor=STATUS_COLORS[status], alpha=0.70, label=status)
            for status in ("robust_pass", "uncertain", "robust_fail")
        ]
        + [
            plt.Line2D([], [], marker="D", linestyle="none", color="#7A0019", label="p95"),
            plt.Line2D([], [], linestyle="--", color="#C00000", label="limit (1)"),
        ],
        loc="upper right",
        ncol=5,
        fontsize=9,
    )
    figure.tight_layout()
    return figure


def main() -> None:
    summaries = read_csv(SUMMARY_CSV)
    trials = read_csv(TRIAL_CSV)
    n_values = sorted({int(row["N"]) for row in summaries})

    with PdfPages(OUTPUT_PDF) as pdf:
        for n in n_values:
            figure = make_figure(n, summaries, trials)
            output_png = RESULT_DIR / f"n{n}_all_candidate_fresh_seed_boxplots.png"
            figure.savefig(output_png, dpi=220, bbox_inches="tight")
            pdf.savefig(figure, bbox_inches="tight")
            plt.close(figure)
            print(output_png)
    print(OUTPUT_PDF)


if __name__ == "__main__":
    main()
