"""Plot fresh-seed error distributions for the selected best candidate per N."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


RESULT_DIR = Path("runs/pose_counts_candidate_ranking_robustness")
TRIAL_CSV = RESULT_DIR / "fresh_seed_trials.csv"
BEST_CSV = RESULT_DIR / "best_candidate_by_n.csv"
OUTPUT_PNG = RESULT_DIR / "best_candidates_fresh_seed_error_boxplots.png"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def main() -> None:
    best_rows = read_csv(BEST_CSV)
    trial_rows = read_csv(TRIAL_CSV)

    labels: list[str] = []
    translation_data: list[list[float]] = []
    rotation_data: list[list[float]] = []
    translation_p95: list[float] = []
    rotation_p95: list[float] = []

    for best in sorted(best_rows, key=lambda row: int(row["N"])):
        n = int(best["N"])
        counts = best["pose_counts"]
        matching = [
            row
            for row in trial_rows
            if int(row["N"]) == n
            and row["pose_counts"] == counts
            and int(row["solver_success"]) == 1
        ]
        if not matching:
            raise RuntimeError(f"No successful trials for N={n}, POSE_COUNTS={counts}")

        translation = [float(row["translation_error_mm"]) for row in matching]
        rotation = [float(row["rotation_error_deg"]) for row in matching]
        labels.append(f"N={n}\n({counts.replace(' ', ',')})")
        translation_data.append(translation)
        rotation_data.append(rotation)
        translation_p95.append(float(np.percentile(translation, 95.0)))
        rotation_p95.append(float(np.percentile(rotation, 95.0)))

    figure, axes = plt.subplots(1, 2, figsize=(12.5, 5.8))
    positions = np.arange(1, len(labels) + 1)
    colors = ("#4C78A8", "#F58518", "#54A24B")

    for axis, data, p95_values, ylabel, threshold in (
        (
            axes[0],
            translation_data,
            translation_p95,
            "Translation error [mm]",
            1.0,
        ),
        (
            axes[1],
            rotation_data,
            rotation_p95,
            "Rotation error [deg]",
            1.0,
        ),
    ):
        boxplot = axis.boxplot(
            data,
            labels=labels,
            widths=0.55,
            patch_artist=True,
            showfliers=True,
            flierprops={"markersize": 3.5, "alpha": 0.45},
            medianprops={"color": "black", "linewidth": 1.6},
        )
        for patch, color in zip(boxplot["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.68)

        axis.scatter(
            positions,
            p95_values,
            marker="D",
            s=46,
            color="#7A0019",
            edgecolor="white",
            linewidth=0.7,
            zorder=4,
            label="p95",
        )
        axis.axhline(
            threshold,
            color="#C00000",
            linestyle="--",
            linewidth=1.4,
            label=f"acceptance limit ({threshold:g})",
        )
        axis.set_ylabel(ylabel)
        axis.set_xlabel("Selected POSE_COUNTS (5/60, 5/120, 40/60, 40/120)")
        axis.set_yscale("log")
        axis.grid(axis="y", which="both", alpha=0.25)
        axis.legend(loc="upper right")

    axes[0].set_title("Fresh-seed translation robustness (500 trials)")
    axes[1].set_title("Fresh-seed rotation robustness (500 trials)")
    figure.suptitle("Best well-conditioned candidate per N", fontsize=14)
    figure.tight_layout()
    figure.savefig(OUTPUT_PNG, dpi=220, bbox_inches="tight")
    plt.close(figure)
    print(OUTPUT_PNG)


if __name__ == "__main__":
    main()
