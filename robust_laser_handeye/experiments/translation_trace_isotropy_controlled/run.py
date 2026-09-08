#!/usr/bin/env python3
"""Controlled test of translation trace versus isotropy.

Stage 1 draws fresh subsets from the frozen Phase-1 candidate bank, rejects all
saved Phase-1 subsets, computes Cov(b), forms four extreme groups, and matches
subsets on the controlled axis and explicit confounders.  It never calibrates.

Stage 2 is a separate opt-in command.  It refuses to run unless Stage 1 marked
all required comparisons as matched, then evaluates only matched subsets using
the saved Phase-2A initial-error bank and identical slot-indexed noise arrays.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree
from scipy.stats import spearmanr, wilcoxon


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.lhs_random_subset_screening import run as phase1  # noqa: E402
from experiments.lhs_initialization_robustness import run as phase2a  # noqa: E402
from experiments.lhs_initialization_robustness import (  # noqa: E402
    analyze_phase2a_geometry as geometry,
)
from experiments.lhs_initialization_robustness import (  # noqa: E402
    analyze_phase2a_jacobian as jacobian_analysis,
)
from laser_handeye.data import LaserScan  # noqa: E402


SCHEMA = "laser_handeye.translation_trace_isotropy_controlled"
SCHEMA_VERSION = 1
PARAMETER_NAMES = phase1.PARAMETER_NAMES
GROUP_NAMES = ("A_high_trace_good_condition", "B_high_trace_bad_condition", "C_low_trace_good_condition", "D_low_trace_bad_condition")
COMPARISONS = {
    "A_vs_B": (GROUP_NAMES[0], GROUP_NAMES[1], "condition", "trace"),
    "C_vs_D": (GROUP_NAMES[2], GROUP_NAMES[3], "condition", "trace"),
    "A_vs_C": (GROUP_NAMES[0], GROUP_NAMES[2], "trace", "log_condition"),
    "B_vs_D": (GROUP_NAMES[1], GROUP_NAMES[3], "trace", "log_condition"),
}
CONFOUNDERS = (
    "position_cov_trace_mm2",
    "position_span_mm",
    "d_mean_mm",
    "d_std_mm",
    "d_range_mm",
    "uv_cov_trace_mm2",
    "u_range_mm",
    "v_range_mm",
    "tilt_range_deg",
    "azimuth_coverage_deg",
    "normal_azimuth_sensor_coverage_deg",
    "rotation_jacobian_logdet",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("stage1", "stage2"), default="stage1")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, Any]:
    with path.expanduser().open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _resolve(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _positive_float(value: Any, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _validate_config(config: dict[str, Any], smoke: bool) -> dict[str, Any]:
    geometry_config = config["geometry"]
    matching = config["matching"]
    scan_counts = sorted(set(_positive_int(n, "geometry.scan_counts[]") for n in geometry_config["scan_counts"]))
    population = _positive_int(geometry_config["fresh_subsets_per_n"], "geometry.fresh_subsets_per_n")
    pairs = _positive_int(matching["pairs_per_comparison"], "matching.pairs_per_comparison")
    minimum_pairs = _positive_int(matching["minimum_pairs_per_comparison"], "matching.minimum_pairs_per_comparison")
    extreme_fraction = float(geometry_config["extreme_fraction"])
    if not 0.0 < extreme_fraction < 0.5:
        raise ValueError("geometry.extreme_fraction must lie in (0,0.5)")
    if minimum_pairs > pairs:
        raise ValueError("minimum_pairs_per_comparison cannot exceed pairs_per_comparison")
    stage2_comparisons = tuple(config["stage2"].get("comparisons", ("A_vs_B", "A_vs_C")))
    invalid_comparisons = [name for name in stage2_comparisons if name not in COMPARISONS]
    if invalid_comparisons or not stage2_comparisons:
        raise ValueError(f"invalid stage2.comparisons: {invalid_comparisons}")
    return {
        "seed": int(config["seed"]),
        "phase1_dir": _resolve(config["phase1_dir"]),
        "phase2a_dir": _resolve(config["phase2a_dir"]),
        "output_dir": _resolve(config["output_dir"]),
        "scan_counts": scan_counts[:1] if smoke else scan_counts,
        "population": 1000 if smoke else population,
        "extreme_fraction": extreme_fraction,
        "eigen_floor": _positive_float(geometry_config["condition_eigenvalue_floor"], "geometry.condition_eigenvalue_floor"),
        "pairs": min(4, pairs) if smoke else pairs,
        "minimum_pairs": min(2, minimum_pairs) if smoke else minimum_pairs,
        "axis_tolerance": _positive_float(matching["controlled_axis_tolerance_std"], "matching.controlled_axis_tolerance_std"),
        "minimum_gap": _positive_float(matching["minimum_treatment_gap_std"], "matching.minimum_treatment_gap_std"),
        "maximum_confounder_rms": _positive_float(matching["maximum_confounder_rms_std"], "matching.maximum_confounder_rms_std"),
        "translation_success_mm": _positive_float(config["stage2"]["translation_success_mm"], "stage2.translation_success_mm"),
        "stage2_comparisons": stage2_comparisons,
        "smoke": bool(smoke),
    }


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(phase1._jsonable(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _circular_coverage_deg(values: np.ndarray) -> float:
    ordered = np.sort(np.mod(np.asarray(values, dtype=float), 360.0))
    gaps = np.diff(np.r_[ordered, ordered[0] + 360.0])
    return float(360.0 - np.max(gaps))


def _geometry_row(
    subset_id: int,
    candidate_ids: np.ndarray,
    bank: dict[str, np.ndarray],
    normal: np.ndarray,
    plane_offset_mm: float,
    eigen_floor: float,
) -> tuple[dict[str, Any], dict[str, float]]:
    """Compute pose-level Cov(b), confounders, and all requested identities."""
    transforms = bank["T_base_s"][candidate_ids]
    rotations = transforms[:, :3, :3]
    b = np.einsum("nji,j->ni", rotations, normal)
    b_norm_error = float(np.max(np.abs(np.linalg.norm(b, axis=1) - 1.0)))
    b_mean = np.mean(b, axis=0)
    centered = b - b_mean[None, :]
    covariance = centered.T @ centered / float(len(b))
    eigenvalues = np.linalg.eigvalsh(covariance)[::-1]
    trace = float(np.trace(covariance))
    lambda_min = float(eigenvalues[-1])
    condition = float("inf") if lambda_min <= eigen_floor else float(eigenvalues[0] / lambda_min)
    normalized = covariance / trace
    normalized_eigenvalues = np.linalg.eigvalsh(normalized)[::-1]
    normalized_min = float(normalized_eigenvalues[-1])
    normalized_condition = float("inf") if normalized_min <= eigen_floor else float(normalized_eigenvalues[0] / normalized_min)

    B = b
    projection = np.eye(len(B)) - np.ones((len(B), len(B))) / float(len(B))
    h_projection = B.T @ projection @ B
    h_covariance = len(B) * covariance
    sign, logdet = np.linalg.slogdet(covariance)

    positions = transforms[:, :3, 3]
    centered_positions = positions - np.mean(positions, axis=0)
    position_cov_trace = float(np.trace(centered_positions.T @ centered_positions / len(positions)))
    distances = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=2)
    parameters = bank["pose_parameters"][candidate_ids]
    uv = parameters[:, :2]
    uv_centered = uv - np.mean(uv, axis=0)

    joint, _ = jacobian_analysis._joint_jacobian(
        candidate_ids, bank, normal, plane_offset_mm
    )
    rotation_j = joint[:, :3] * jacobian_analysis.ROTATION_SCALE_RAD
    rotation_sign, rotation_logdet = np.linalg.slogdet(rotation_j.T @ rotation_j)

    row = {
        "N": len(candidate_ids),
        "subset_id": subset_id,
        "candidate_ids": json.dumps(candidate_ids.tolist(), separators=(",", ":")),
        "group": "unassigned",
        "b_resultant": float(np.linalg.norm(b_mean)),
        "b_cov_trace": trace,
        "b_cov_lambda_max": float(eigenvalues[0]),
        "b_cov_lambda_mid": float(eigenvalues[1]),
        "b_cov_lambda_min": lambda_min,
        "b_cov_rank": int(np.sum(eigenvalues > eigen_floor)),
        "b_cov_condition": condition,
        "b_cov_log_condition": float(np.log(condition)),
        "b_cov_logdet": float(logdet) if sign > 0.0 else float("-inf"),
        "b_cov_normalized_lambda_min": normalized_min,
        "b_cov_normalized_condition": normalized_condition,
        "position_cov_trace_mm2": position_cov_trace,
        "position_span_mm": float(np.max(distances)),
        "d_mean_mm": float(np.mean(parameters[:, 2])),
        "d_std_mm": float(np.std(parameters[:, 2])),
        "d_range_mm": float(np.ptp(parameters[:, 2])),
        "uv_cov_trace_mm2": float(np.trace(uv_centered.T @ uv_centered / len(uv))),
        "u_range_mm": float(np.ptp(parameters[:, 0])),
        "v_range_mm": float(np.ptp(parameters[:, 1])),
        "tilt_range_deg": float(np.ptp(parameters[:, 3])),
        "azimuth_coverage_deg": _circular_coverage_deg(parameters[:, 4]),
        "normal_azimuth_sensor_coverage_deg": _circular_coverage_deg(
            parameters[:, 5]
        ),
        "rotation_jacobian_logdet": float(rotation_logdet) if rotation_sign > 0.0 else float("-inf"),
    }
    checks = {
        "unit_b_max_abs": b_norm_error,
        "b_mean_max_abs": float(np.max(np.abs(b_mean - np.sum(B, axis=0) / len(B)))),
        "projection_vs_cov_max_abs": float(np.max(np.abs(h_projection - h_covariance))),
        "projection_vs_cov_relative_fro": float(np.linalg.norm(h_projection - h_covariance, ord="fro") / max(np.linalg.norm(h_covariance, ord="fro"), 1e-12)),
        "trace_resultant_max_abs": abs(trace - (1.0 - float(b_mean @ b_mean))),
    }
    return row, checks


def _load_source(resolved: dict[str, Any]) -> tuple[dict[str, Any], dict[str, np.ndarray], dict[int, set[tuple[int, ...]]]]:
    manifest, bank, _summaries, subsets = phase2a._load_phase1(resolved["phase1_dir"])
    excluded: dict[int, set[tuple[int, ...]]] = {}
    for n, matrix in subsets.items():
        excluded[n] = {tuple(sorted(map(int, row))) for row in matrix}
    return manifest, bank, excluded


def _fresh_subsets(
    rng: np.random.Generator,
    candidate_count: int,
    n: int,
    count: int,
    excluded: set[tuple[int, ...]],
) -> np.ndarray:
    """Draw unique unordered subsets and explicitly reject saved Phase-1 sets."""
    accepted: list[np.ndarray] = []
    seen = set(excluded)
    while len(accepted) < count:
        row = np.sort(rng.choice(candidate_count, n, replace=False))
        key = tuple(map(int, row))
        if key in seen:
            continue
        seen.add(key)
        accepted.append(row)
    return np.stack(accepted)


def _assign_groups(rows: list[dict[str, Any]], fraction: float) -> dict[str, float]:
    trace = np.asarray([row["b_cov_trace"] for row in rows], dtype=float)
    log_condition = np.asarray([row["b_cov_log_condition"] for row in rows], dtype=float)
    trace_low, trace_high = np.quantile(trace, [fraction, 1.0 - fraction])
    condition_good, condition_bad = np.quantile(log_condition, [fraction, 1.0 - fraction])
    for row in rows:
        high = row["b_cov_trace"] >= trace_high
        low = row["b_cov_trace"] <= trace_low
        good = row["b_cov_log_condition"] <= condition_good
        bad = row["b_cov_log_condition"] >= condition_bad
        if high and good:
            row["group"] = GROUP_NAMES[0]
        elif high and bad:
            row["group"] = GROUP_NAMES[1]
        elif low and good:
            row["group"] = GROUP_NAMES[2]
        elif low and bad:
            row["group"] = GROUP_NAMES[3]
    return {"trace_low": float(trace_low), "trace_high": float(trace_high), "log_condition_good": float(condition_good), "log_condition_bad": float(condition_bad)}


def _standardization(rows: Sequence[dict[str, Any]]) -> tuple[dict[str, float], dict[str, float]]:
    fields = ("b_cov_trace", "b_cov_log_condition", *CONFOUNDERS)
    means = {field: float(np.mean([row[field] for row in rows])) for field in fields}
    scales = {field: max(float(np.std([row[field] for row in rows])), 1e-12) for field in fields}
    return means, scales


def _match_comparison(
    rows: Sequence[dict[str, Any]],
    comparison: str,
    resolved: dict[str, Any],
    means: dict[str, float],
    scales: dict[str, float],
) -> list[dict[str, Any]]:
    """Greedily match on the controlled axis and standardized confounders."""
    left_group, right_group, treatment, controlled = COMPARISONS[comparison]
    left = [row for row in rows if row["group"] == left_group]
    right = [row for row in rows if row["group"] == right_group]
    controlled_field = "b_cov_trace" if controlled == "trace" else "b_cov_log_condition"
    treatment_field = "b_cov_trace" if treatment == "trace" else "b_cov_log_condition"
    match_fields = (controlled_field, *CONFOUNDERS)
    right_matrix = np.asarray([[(row[field] - means[field]) / scales[field] for field in match_fields] for row in right])
    right_matrix[:, 0] *= 8.0
    tree = cKDTree(right_matrix)
    candidates: list[tuple[float, int, int, float, float, float]] = []
    for left_index, row in enumerate(left):
        vector = np.asarray([(row[field] - means[field]) / scales[field] for field in match_fields])
        vector[0] *= 8.0
        distances, indices = tree.query(vector, k=min(80, len(right)))
        for distance, right_index in zip(np.atleast_1d(distances), np.atleast_1d(indices)):
            other = right[int(right_index)]
            controlled_diff = abs(row[controlled_field] - other[controlled_field]) / scales[controlled_field]
            treatment_gap = abs(row[treatment_field] - other[treatment_field]) / scales[treatment_field]
            confounder_rms = float(np.sqrt(np.mean([((row[field] - other[field]) / scales[field]) ** 2 for field in CONFOUNDERS])))
            if controlled_diff <= resolved["axis_tolerance"] and treatment_gap >= resolved["minimum_gap"]:
                candidates.append((float(distance), left_index, int(right_index), controlled_diff, treatment_gap, confounder_rms))
    candidates.sort()
    used_left: set[int] = set()
    used_right: set[int] = set()
    pairs: list[dict[str, Any]] = []
    for cost, left_index, right_index, controlled_diff, treatment_gap, confounder_rms in candidates:
        if left_index in used_left or right_index in used_right:
            continue
        if confounder_rms > resolved["maximum_confounder_rms"]:
            continue
        a, b = left[left_index], right[right_index]
        pairs.append({
            "N": int(a["N"]), "comparison": comparison, "pair_id": len(pairs),
            "left_group": left_group, "right_group": right_group,
            "left_subset_id": int(a["subset_id"]), "right_subset_id": int(b["subset_id"]),
            "controlled_axis": controlled, "treatment_axis": treatment,
            "controlled_abs_difference_std": controlled_diff,
            "treatment_abs_gap_std": treatment_gap,
            "confounder_rms_difference_std": confounder_rms,
            "left_trace": a["b_cov_trace"], "right_trace": b["b_cov_trace"],
            "left_condition": a["b_cov_condition"], "right_condition": b["b_cov_condition"],
            "left_b_resultant": a["b_resultant"], "right_b_resultant": b["b_resultant"],
        })
        used_left.add(left_index); used_right.add(right_index)
        if len(pairs) >= resolved["pairs"]:
            break
    return pairs


def _group_summary(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    fields = ("b_cov_trace", "b_cov_condition", "b_resultant", *CONFOUNDERS)
    output: list[dict[str, Any]] = []
    for n in sorted({int(row["N"]) for row in rows}):
        for group in GROUP_NAMES:
            selected = [row for row in rows if int(row["N"]) == n and row["group"] == group]
            for field in fields:
                values = np.asarray([row[field] for row in selected], dtype=float)
                output.append({"N": n, "group": group, "group_size": len(selected), "metric": field, "mean": float(np.mean(values)) if len(values) else math.nan, "std": float(np.std(values)) if len(values) else math.nan})
    return output


def _write_stage1_plots(rows: Sequence[dict[str, Any]], output: Path) -> None:
    plot_dir = output / "plots"; plot_dir.mkdir(exist_ok=True)
    for n in sorted({int(row["N"]) for row in rows}):
        selected = [row for row in rows if int(row["N"]) == n]
        trace = np.asarray([row["b_cov_trace"] for row in selected]); condition = np.asarray([row["b_cov_condition"] for row in selected])
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
        axes[0].hist(trace, bins=60, color="#4575b4"); axes[0].set_title(f"N={n} Cov(b) trace"); axes[0].set_xlabel("trace")
        axes[1].hist(np.log10(condition), bins=60, color="#f46d43"); axes[1].set_title(f"N={n} condition"); axes[1].set_xlabel("log10 condition")
        fig.tight_layout(); fig.savefig(plot_dir / f"N{n:03d}_distributions.png", dpi=170); plt.close(fig)
        fig, ax = plt.subplots(figsize=(7, 5.5))
        colors = {GROUP_NAMES[0]: "#1a9850", GROUP_NAMES[1]: "#d73027", GROUP_NAMES[2]: "#74add1", GROUP_NAMES[3]: "#984ea3", "unassigned": "#bdbdbd"}
        for group in ("unassigned", *GROUP_NAMES):
            group_rows = [row for row in selected if row["group"] == group]
            ax.scatter([row["b_cov_trace"] for row in group_rows], [row["b_cov_condition"] for row in group_rows], s=5 if group == "unassigned" else 10, alpha=0.25 if group == "unassigned" else 0.55, color=colors[group], label=group)
        ax.set_yscale("log"); ax.set_xlabel("trace(Cov(b))"); ax.set_ylabel("condition(Cov(b))"); ax.set_title(f"N={n}: trace vs isotropy groups"); ax.legend(fontsize=6)
        fig.tight_layout(); fig.savefig(plot_dir / f"N{n:03d}_trace_vs_condition.png", dpi=170); plt.close(fig)


def _matching_quality(pairs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for n in sorted({int(row["N"]) for row in pairs}):
        for comparison in sorted({row["comparison"] for row in pairs if int(row["N"]) == n}):
            selected = [
                row
                for row in pairs
                if int(row["N"]) == n and row["comparison"] == comparison
            ]
            output[f"N{n}_{comparison}"] = {
                "pair_count": len(selected),
                "controlled_difference_std_median": float(np.median([row["controlled_abs_difference_std"] for row in selected])) if selected else math.inf,
                "controlled_difference_std_max": float(np.max([row["controlled_abs_difference_std"] for row in selected])) if selected else math.inf,
                "treatment_gap_std_median": float(np.median([row["treatment_abs_gap_std"] for row in selected])) if selected else 0.0,
                "treatment_gap_std_min": float(np.min([row["treatment_abs_gap_std"] for row in selected])) if selected else 0.0,
                "confounder_rms_std_median": float(np.median([row["confounder_rms_difference_std"] for row in selected])) if selected else math.inf,
                "confounder_rms_std_max": float(np.max([row["confounder_rms_difference_std"] for row in selected])) if selected else math.inf,
            }
    return output


def _write_stage1_report(
    path: Path,
    rows: Sequence[dict[str, Any]],
    summary_rows: Sequence[dict[str, Any]],
    report: dict[str, Any],
) -> None:
    summary_lookup = {
        (int(row["N"]), str(row["group"]), str(row["metric"])): row
        for row in summary_rows
    }
    lines = [
        "# Translation trace–isotropy Stage 1 geometry report",
        "",
        "Calibration은 실행하지 않았다. 기존 Phase 1에 저장된 모든 unordered subset을 제외하고 fresh subset만 생성했다.",
        "",
        f"- Fresh subsets: {report['fresh_subset_count']}",
        f"- Unique matched subsets: {report['matched_subset_count']}",
        f"- Matching passed: `{str(report['matching_passed']).lower()}`",
        "",
        "## Numerical identities",
        "",
        "| identity error | maximum | tolerance |",
        "|---|---:|---:|",
    ]
    for name, value in report["numerical_validation_max_errors"].items():
        lines.append(f"| `{name}` | {value:.6e} | {report['numerical_tolerance']:.1e} |")
    for n in sorted({int(row["N"]) for row in rows}):
        lines += ["", f"## N={n} group geometry", "", "| group | count | trace mean±std | condition mean±std | b resultant mean±std |", "|---|---:|---:|---:|---:|"]
        for group in GROUP_NAMES:
            trace = summary_lookup[(n, group, "b_cov_trace")]
            condition = summary_lookup[(n, group, "b_cov_condition")]
            resultant = summary_lookup[(n, group, "b_resultant")]
            lines.append(
                f"| {group} | {int(trace['group_size'])} | "
                f"{float(trace['mean']):.4f}±{float(trace['std']):.4f} | "
                f"{float(condition['mean']):.2f}±{float(condition['std']):.2f} | "
                f"{float(resultant['mean']):.4f}±{float(resultant['std']):.4f} |"
            )
    lines += ["", "## Matching quality", "", "Standardized differences use the entire fresh population standard deviation.", "", "| comparison | pairs | controlled median/max | treatment gap median/min | confounder RMS median/max |", "|---|---:|---:|---:|---:|"]
    for comparison, quality in report["matching_quality"].items():
        lines.append(
            f"| {comparison} | {quality['pair_count']} | "
            f"{quality['controlled_difference_std_median']:.3f}/{quality['controlled_difference_std_max']:.3f} | "
            f"{quality['treatment_gap_std_median']:.3f}/{quality['treatment_gap_std_min']:.3f} | "
            f"{quality['confounder_rms_std_median']:.3f}/{quality['confounder_rms_std_max']:.3f} |"
        )
    lines += [
        "",
        "A–B와 C–D는 trace를 통제하고 condition을 변화시키며, A–C와 B–D는 log-condition을 통제하고 trace를 변화시킨다.",
        "",
        "`matching_passed=true`일 때만 별도 Stage 2 command가 calibration을 허용한다.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def run_stage1(config: dict[str, Any], output_override: Path | None, smoke: bool) -> Path:
    resolved = _validate_config(config, smoke)
    if output_override is not None: resolved["output_dir"] = _resolve(output_override)
    output = resolved["output_dir"]
    if output.exists() and next(output.iterdir(), None) is not None:
        raise FileExistsError(f"output is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    manifest, bank, excluded = _load_source(resolved)
    normal = np.asarray(manifest["resolved"]["plane"]["normal_base"], dtype=float)
    plane_offset_mm = float(manifest["resolved"]["plane"]["offset_mm"])
    candidate_count = len(bank["candidate_ids"])
    all_rows: list[dict[str, Any]] = []; all_pairs: list[dict[str, Any]] = []
    thresholds: dict[str, Any] = {}; validation_max: dict[str, float] = {}
    rng = np.random.default_rng(np.random.SeedSequence([resolved["seed"], 0x434F4E54]))
    matrices: dict[int, np.ndarray] = {}
    for n in resolved["scan_counts"]:
        print(f"[Stage 1 N={n}] drawing {resolved['population']} fresh subsets...", flush=True)
        matrix = _fresh_subsets(rng, candidate_count, n, resolved["population"], excluded.get(n, set()))
        matrices[n] = matrix
        n_rows: list[dict[str, Any]] = []
        for subset_id, candidate_ids in enumerate(matrix):
            row, checks = _geometry_row(
                subset_id,
                candidate_ids,
                bank,
                normal,
                plane_offset_mm,
                resolved["eigen_floor"],
            )
            n_rows.append(row)
            for name, value in checks.items(): validation_max[name] = max(validation_max.get(name, 0.0), value)
        thresholds[str(n)] = _assign_groups(n_rows, resolved["extreme_fraction"])
        means, scales = _standardization(n_rows)
        for comparison in COMPARISONS:
            pairs = _match_comparison(n_rows, comparison, resolved, means, scales)
            all_pairs.extend(pairs)
            print(f"[Stage 1 N={n}] {comparison}: {len(pairs)} matched pairs", flush=True)
        all_rows.extend(n_rows)

    validation_tolerance = 1e-9
    identity_passed = all(value <= validation_tolerance for value in validation_max.values())
    comparison_counts = {(int(n), comparison): sum(int(row["N"]) == n and row["comparison"] == comparison for row in all_pairs) for n in resolved["scan_counts"] for comparison in COMPARISONS}
    matching_quality = _matching_quality(all_pairs)
    matching_passed = identity_passed and all(count >= resolved["minimum_pairs"] for count in comparison_counts.values())
    selected_ids = {(int(pair["N"]), int(pair[side])) for pair in all_pairs for side in ("left_subset_id", "right_subset_id")}
    selected_rows = [row for row in all_rows if (int(row["N"]), int(row["subset_id"])) in selected_ids]

    _write_csv(output / "fresh_subset_geometry.csv", all_rows)
    group_summary_rows = _group_summary(all_rows)
    _write_csv(output / "group_geometry_summary.csv", group_summary_rows)
    _write_csv(output / "matched_pairs.csv", all_pairs)
    _write_csv(output / "matched_subsets.csv", selected_rows)
    payload = {f"candidate_ids_N{n:03d}": matrices[n] for n in matrices}
    np.savez_compressed(output / "fresh_subsets.npz", **payload)
    _write_stage1_plots(all_rows, output)
    report = {
        "schema": SCHEMA, "schema_version": SCHEMA_VERSION, "stage": 1,
        "status": "complete", "matching_passed": matching_passed,
        "config": config, "resolved": resolved,
        "source_phase1_manifest_sha256": _sha256(resolved["phase1_dir"] / "manifest.json"),
        "fresh_subset_count": len(all_rows), "matched_subset_count": len(selected_rows),
        "matched_pair_counts": {f"N{n}_{comparison}": count for (n, comparison), count in comparison_counts.items()},
        "matching_quality": matching_quality,
        "thresholds": thresholds, "numerical_validation_max_errors": validation_max,
        "numerical_tolerance": validation_tolerance,
        "freshness_policy": "reject every unordered subset saved in Phase 1; Phase 2A is a subset of these",
    }
    _write_json(output / "stage1_manifest.json", report)
    _write_stage1_report(output / "stage1_geometry_report.md", all_rows, group_summary_rows, report)
    print(f"Stage 1 matching_passed={matching_passed}; output={output}")
    if not matching_passed:
        print("Calibration is blocked. Increase population or relax declared matching tolerances after inspecting the report.")
    return output


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle: return list(csv.DictReader(handle))


def _paired_noisy_scans(resolved: dict[str, Any], bank: dict[str, np.ndarray], candidate_ids: np.ndarray, environment: int, repeat: int, noise_std: float, noise_axis: str) -> list[LaserScan]:
    """Apply exactly the same pointwise noise array to every subset by scan slot."""
    scans: list[LaserScan] = []
    for slot, candidate_id_value in enumerate(candidate_ids):
        candidate_id = int(candidate_id_value); points = bank["ideal_points_s"][candidate_id].copy()
        rng = np.random.default_rng(np.random.SeedSequence([resolved["seed"], 0x50414952, environment, repeat, slot]))
        if noise_axis == "z": points[:, 2] += rng.normal(0.0, noise_std, len(points))
        else: points[:, [0, 2]] += rng.normal(0.0, noise_std, (len(points), 2))
        scans.append(LaserScan(T_base_ef=bank["T_base_ef"][candidate_id], points_s=points, plane_id=0, scan_id=candidate_id, meta={"candidate_id": candidate_id, "environment_id": environment, "noise_repeat": repeat, "noise_slot": slot}))
    return scans


def _stage2_summary(subset: dict[str, Any], runs: Sequence[dict[str, Any]], translation_threshold: float) -> dict[str, Any]:
    count = len(runs); translation_success = sum(np.isfinite(row["translation_error_mm"]) and row["translation_error_mm"] <= translation_threshold for row in runs)
    return {**subset, "runs": count, "translation_success_rate": translation_success / count, "full_success_rate": sum(bool(row["passes_success_threshold"]) for row in runs) / count, "translation_error_median_mm": phase2a._finite_quantile(runs, "translation_error_mm", 0.5), "translation_error_p90_mm": phase2a._finite_quantile(runs, "translation_error_mm", 0.9), "rotation_error_median_deg": phase2a._finite_quantile(runs, "rotation_error_deg", 0.5), "rotation_error_p90_deg": phase2a._finite_quantile(runs, "rotation_error_deg", 0.9), "normalized_error_median": phase2a._finite_quantile(runs, "normalized_error", 0.5)}


def _analyze_stage2(summaries: Sequence[dict[str, Any]], pairs: Sequence[dict[str, str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    lookup = {(int(row["N"]), int(row["subset_id"])): row for row in summaries}
    paired: list[dict[str, Any]] = []
    for n in sorted({int(row["N"]) for row in pairs}):
        for comparison in sorted(
            {row["comparison"] for row in pairs if int(row["N"]) == n}
        ):
            for outcome in ("translation_success_rate", "translation_error_median_mm", "full_success_rate", "rotation_error_median_deg"):
                selected = [row for row in pairs if int(row["N"]) == n and row["comparison"] == comparison]
                left = np.asarray([lookup[(int(row["N"]), int(row["left_subset_id"]))][outcome] for row in selected], dtype=float)
                right = np.asarray([lookup[(int(row["N"]), int(row["right_subset_id"]))][outcome] for row in selected], dtype=float)
                difference = left - right; nonzero = difference[difference != 0.0]
                statistic, p_value = wilcoxon(left, right) if len(nonzero) else (0.0, 1.0)
                paired.append({"N": n, "comparison": comparison, "outcome": outcome, "pair_count": len(left), "left_median": float(np.median(left)), "right_median": float(np.median(right)), "median_paired_difference": float(np.median(difference)), "paired_sign_effect": float((np.sum(difference > 0)-np.sum(difference < 0))/len(difference)), "wilcoxon_statistic": float(statistic), "p_value": float(p_value), "fdr_q_value": math.nan})
        n_paired = [row for row in paired if int(row["N"]) == n]
        adjusted = geometry._bh_adjust([row["p_value"] for row in n_paired])
        for row, q_value in zip(n_paired, adjusted, strict=True):
            row["fdr_q_value"] = float(q_value)
    correlations: list[dict[str, Any]] = []
    for n in sorted({int(row["N"]) for row in summaries}):
        n_rows = [row for row in summaries if int(row["N"]) == n]
        for feature in ("b_cov_trace", "b_resultant", "b_cov_condition", "b_cov_normalized_lambda_min", "b_cov_logdet"):
            for outcome in ("translation_success_rate", "translation_error_median_mm", "full_success_rate"):
                x = np.asarray([row[feature] for row in n_rows], dtype=float)
                y = np.asarray([row[outcome] for row in n_rows], dtype=float)
                finite = np.isfinite(x) & np.isfinite(y)
                if np.sum(finite) >= 3 and np.unique(x[finite]).size >= 2 and np.unique(y[finite]).size >= 2:
                    result = spearmanr(x[finite], y[finite])
                    rho, p_value = float(result.correlation), float(result.pvalue)
                else:
                    rho, p_value = math.nan, math.nan
                correlations.append({"N": n, "population": "unique calibrated matched subsets", "feature": feature, "outcome": outcome, "subset_count": len(n_rows), "spearman_rho": rho, "p_value": p_value, "fdr_q_value": math.nan})
        n_correlations = [row for row in correlations if int(row["N"]) == n]
        adjusted = geometry._bh_adjust([row["p_value"] for row in n_correlations])
        for row, q_value in zip(n_correlations, adjusted, strict=True):
            row["fdr_q_value"] = float(q_value)
    return paired, correlations


def run_stage2(config: dict[str, Any], output_override: Path | None, smoke: bool) -> Path:
    resolved = _validate_config(config, smoke)
    if output_override is not None: resolved["output_dir"] = _resolve(output_override)
    output = resolved["output_dir"]; stage1_manifest = _load_json(output / "stage1_manifest.json")
    if stage1_manifest.get("matching_passed") is not True: raise RuntimeError("Stage 1 matching did not pass; calibration is blocked")
    if (output / "stage2_manifest.json").exists(): raise FileExistsError("Stage 2 output already exists")
    phase1_manifest, bank, _excluded = _load_source(resolved)
    phase2a_manifest = _load_json(resolved["phase2a_dir"] / "manifest.json")
    with np.load(resolved["phase2a_dir"] / "scenario_bank.npz", allow_pickle=False) as archive: scenarios = {name: archive[name].copy() for name in archive.files}
    noise = phase2a_manifest["config"]["noise"]; repeats = 1 if smoke else int(noise["repeats_per_environment"]); environments = scenarios["environment_ids"][:2] if smoke else scenarios["environment_ids"]
    subsets = _read_csv(output / "matched_subsets.csv"); pairs = _read_csv(output / "matched_pairs.csv")
    pairs = [row for row in pairs if row["comparison"] in resolved["stage2_comparisons"]]
    if smoke:
        pairs = [
            next(row for row in pairs if row["comparison"] == comparison)
            for comparison in resolved["stage2_comparisons"]
        ]
        selected_keys = {
            (int(row["N"]), int(row[side]))
            for row in pairs
            for side in ("left_subset_id", "right_subset_id")
        }
        subsets = [
            row
            for row in subsets
            if (int(row["N"]), int(row["subset_id"])) in selected_keys
        ]
    all_runs: list[dict[str, Any]] = []; summaries: list[dict[str, Any]] = []
    total = len(subsets) * len(environments) * repeats; completed = 0; started = time.perf_counter()
    for subset_position, subset_row in enumerate(subsets, start=1):
        candidate_ids = np.asarray(json.loads(subset_row["candidate_ids"]), dtype=np.int64); subset_runs: list[dict[str, Any]] = []
        for environment_value in environments:
            environment = int(environment_value)
            solver_resolved = phase2a._solver_resolved(phase1_manifest, {"classification_translation_mm": float(phase2a_manifest["config"]["classification"]["translation_mm"]), "classification_rotation_deg": float(phase2a_manifest["config"]["classification"]["rotation_deg"])}, scenarios["T_init"][environment])
            for repeat in range(repeats):
                scans = _paired_noisy_scans(resolved, bank, candidate_ids, environment, repeat, float(noise["std_mm"]), str(noise["axis"]))
                row = phase1._run_one_calibration(
                    resolved=solver_resolved,
                    scans=scans,
                    scan_count=int(subset_row["N"]),
                    subset_id=int(subset_row["subset_id"]),
                    repeat=repeat,
                    candidate_ids=candidate_ids,
                )
                row.update(source="translation_trace_isotropy_controlled", environment_id=environment, noise_repeat=repeat)
                subset_runs.append(row); all_runs.append(row); completed += 1
        numeric_subset = {key: (float(value) if key not in ("candidate_ids", "group") else value) for key, value in subset_row.items()}
        numeric_subset["N"] = int(subset_row["N"]); numeric_subset["subset_id"] = int(subset_row["subset_id"])
        summaries.append(_stage2_summary(numeric_subset, subset_runs, resolved["translation_success_mm"]))
        if subset_position % max(1, len(subsets)//20) == 0:
            elapsed = time.perf_counter()-started; eta=elapsed*(total-completed)/completed
            print(f"[Stage 2] subsets {subset_position}/{len(subsets)}, runs {completed}/{total}, ETA {phase1._format_duration(eta)}", flush=True)
    paired_stats, correlations = _analyze_stage2(summaries, pairs)
    _write_csv(output / "stage2_calibration_runs.csv", all_runs); _write_csv(output / "stage2_subset_summary.csv", summaries); _write_csv(output / "stage2_paired_statistics.csv", paired_stats); _write_csv(output / "stage2_subset_correlations.csv", correlations)
    _write_json(output / "stage2_manifest.json", {"schema": SCHEMA, "schema_version": SCHEMA_VERSION, "stage": 2, "status": "complete", "subset_count": len(subsets), "calibration_runs": len(all_runs), "subset_is_statistical_unit": True, "pairing": {"initial_error": "saved Phase2A scenario bank", "noise": "identical pointwise noise by environment/repeat/scan slot"}})
    print(f"Stage 2 complete: {output}"); return output


def main() -> int:
    args = _parser(); config = _load_json(args.config)
    if args.stage == "stage1": run_stage1(config, args.output_dir, args.smoke)
    else: run_stage2(config, args.output_dir, args.smoke)
    return 0


if __name__ == "__main__": raise SystemExit(main())
