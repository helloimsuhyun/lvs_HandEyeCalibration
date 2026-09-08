#!/usr/bin/env python3
"""Observe geometry patterns associated with Phase-2A calibration robustness.

This is an exploratory, read-only analysis of completed Phase 1 and Phase 2A
results.  It does not select or generate poses, run calibration, use Fisher or
D-optimal scores, or access Phase 2B data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mannwhitneyu, spearmanr


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


PHASE1_SCHEMA = "laser_handeye.phase1_random_subset_screening"
PHASE2A_SCHEMA = "laser_handeye.phase2a_initialization_robustness"
PRIMARY_N = (9, 11, 13, 15)
DIAGNOSTIC_N = (5, 7)
LEGACY_PARAMETER_NAMES = (
    "u_mm",
    "v_mm",
    "d_mm",
    "tilt_deg",
    "azimuth_deg",
    "roll_deg",
)
PARAMETER_NAMES = (
    "u_mm",
    "v_mm",
    "d_mm",
    "tilt_deg",
    "azimuth_deg",
    "normal_azimuth_sensor_deg",
)
PERFORMANCE_FIELDS = (
    "accuracy_success_rate",
    "normalized_error_p90",
    "normalized_error_median",
)
CORE_SUCCESS_RATE = 0.90
CORE_P90 = 1.0
BORDERLINE_SUCCESS_RATE = 0.80
BORDERLINE_P90 = 1.5
LARGE_EFFECT = 0.33
NUMERICAL_ZERO = 1e-12


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase1-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "phase1_lhs_screening",
    )
    parser.add_argument(
        "--phase2a-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "phase2a_initialization_robustness",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "phase2a_geometry_observation",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace analysis files with the same names; source results are untouched",
    )
    return parser.parse_args()


def _resolve(path: Path) -> Path:
    path = path.expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to create headerless CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_declared_file(
    root: Path, manifest: dict[str, Any], relative: str
) -> Path:
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(f"missing declared result file: {path}")
    expected = manifest.get("files", {}).get(relative)
    if expected and _sha256(path) != expected:
        raise ValueError(f"result file hash mismatch: {path}")
    return path


def _prepare_output(directory: Path, overwrite: bool) -> Path:
    source_names = {
        "subset_geometry_features.csv",
        "group_statistics.csv",
        "core_vs_non_effects.csv",
        "diagnostic_top_bottom_effects.csv",
        "continuous_correlations.csv",
        "consistent_candidate_features.csv",
        "transition_N007_N009.csv",
        "observations.md",
    }
    plot_names = {
        "accuracy_success_rate_by_n.png",
        "key_features_core_vs_non.png",
        "core_vs_non_cliffs_delta_heatmap.png",
        "continuous_spearman_heatmap.png",
        "key_features_vs_p90.png",
        "N007_vs_N009_top_group_transition.png",
    }
    existing = [directory / name for name in source_names if (directory / name).exists()]
    existing += [
        directory / "plots" / name
        for name in plot_names
        if (directory / "plots" / name).exists()
    ]
    if existing and not overwrite:
        raise FileExistsError(
            f"analysis output already exists; use --overwrite: {existing[0]}"
        )
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "plots").mkdir(exist_ok=True)
    return directory


def _float(row: dict[str, str], key: str) -> float:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"invalid numeric field {key!r} in row: {row}") from error


def _label(summary: dict[str, str]) -> str:
    success = _float(summary, "accuracy_success_rate")
    p90 = _float(summary, "normalized_error_p90")
    if success >= CORE_SUCCESS_RATE and p90 <= CORE_P90:
        return "core"
    if success >= BORDERLINE_SUCCESS_RATE and p90 <= BORDERLINE_P90:
        return "borderline"
    return "non_survivor"


def _quantile(values: np.ndarray, q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), q))


def _linear_descriptors(values: np.ndarray, prefix: str) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    return {
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
        f"{prefix}_range": float(np.ptp(values)),
        f"{prefix}_std": float(np.std(values, ddof=0)),
        f"{prefix}_iqr": _quantile(values, 0.75) - _quantile(values, 0.25),
    }


def _pairwise_circular_distance_deg(values: np.ndarray) -> np.ndarray:
    values = np.mod(np.asarray(values, dtype=float), 360.0)
    difference = np.abs(values[:, None] - values[None, :])
    difference = np.minimum(difference, 360.0 - difference)
    return difference[np.triu_indices(len(values), 1)]


def _circular_descriptors(values: np.ndarray, prefix: str) -> dict[str, float]:
    degrees = np.mod(np.asarray(values, dtype=float), 360.0)
    radians = np.deg2rad(degrees)
    resultant = float(
        np.hypot(np.mean(np.cos(radians)), np.mean(np.sin(radians)))
    )
    ordered = np.sort(degrees)
    gaps = np.diff(np.r_[ordered, ordered[0] + 360.0])
    largest_gap = float(np.max(gaps))
    pairwise = _pairwise_circular_distance_deg(degrees)
    return {
        f"{prefix}_resultant_length": resultant,
        f"{prefix}_circular_variance": 1.0 - resultant,
        f"{prefix}_largest_gap_deg": largest_gap,
        f"{prefix}_coverage_deg": 360.0 - largest_gap,
        f"{prefix}_pairwise_mean_deg": float(np.mean(pairwise)),
        f"{prefix}_pairwise_min_deg": float(np.min(pairwise)),
    }


def _unit_rows(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=float)
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(norms <= NUMERICAL_ZERO):
        raise ValueError("encountered zero-length geometry vector")
    return vectors / norms[:, None]


def _pairwise_vector_angles_deg(vectors: np.ndarray) -> np.ndarray:
    vectors = _unit_rows(vectors)
    cosine = np.clip(vectors @ vectors.T, -1.0, 1.0)
    return np.rad2deg(np.arccos(cosine[np.triu_indices(len(vectors), 1)]))


def _condition(maximum: float, minimum: float) -> float:
    return float("inf") if abs(minimum) <= NUMERICAL_ZERO else maximum / minimum


def _normal_descriptors(viewing_normals: np.ndarray) -> dict[str, float]:
    normals = _unit_rows(viewing_normals)
    gram = np.einsum("ni,nj->ij", normals, normals) / float(len(normals))
    eigenvalues = np.linalg.eigvalsh(gram)[::-1]
    angles = _pairwise_vector_angles_deg(normals)
    return {
        "normal_lambda_min": float(eigenvalues[-1]),
        "normal_lambda_max": float(eigenvalues[0]),
        "normal_condition": _condition(float(eigenvalues[0]), float(eigenvalues[-1])),
        "normal_eigenvalue_std": float(np.std(eigenvalues, ddof=0)),
        "normal_resultant": float(np.linalg.norm(np.mean(normals, axis=0))),
        "normal_pairwise_angle_min_deg": float(np.min(angles)),
        "normal_pairwise_angle_mean_deg": float(np.mean(angles)),
        "normal_pairwise_angle_max_deg": float(np.max(angles)),
    }


def _rotation_descriptors(rotations: np.ndarray) -> dict[str, float]:
    angles: list[float] = []
    for i in range(len(rotations)):
        for j in range(i + 1, len(rotations)):
            relative = rotations[i].T @ rotations[j]
            cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
            angles.append(float(np.rad2deg(np.arccos(cosine))))
    values = np.asarray(angles, dtype=float)
    return {
        "rotation_pair_min_deg": float(np.min(values)),
        "rotation_pair_mean_deg": float(np.mean(values)),
        "rotation_pair_max_deg": float(np.max(values)),
        "rotation_pair_std_deg": float(np.std(values, ddof=0)),
    }


def _position_descriptors(positions: np.ndarray) -> dict[str, float]:
    positions = np.asarray(positions, dtype=float)
    centered = positions - np.mean(positions, axis=0)
    covariance = centered.T @ centered / float(len(positions))
    eigenvalues = np.linalg.eigvalsh(covariance)[::-1]
    differences = positions[:, None, :] - positions[None, :, :]
    distances = np.linalg.norm(differences, axis=2)
    pairwise = distances[np.triu_indices(len(positions), 1)]
    return {
        "position_cov_trace": float(np.trace(covariance)),
        "position_lambda_min": float(eigenvalues[-1]),
        "position_lambda_max": float(eigenvalues[0]),
        "position_condition": _condition(
            float(eigenvalues[0]), float(eigenvalues[-1])
        ),
        "position_pairwise_min_mm": float(np.min(pairwise)),
        "position_pairwise_mean_mm": float(np.mean(pairwise)),
        "position_pairwise_max_mm": float(np.max(pairwise)),
    }


def _geometry_features(
    candidate_ids: np.ndarray,
    bank: dict[str, np.ndarray],
    plane_center: np.ndarray,
    plane_basis: np.ndarray,
) -> dict[str, float]:
    parameters = bank["pose_parameters"][candidate_ids]
    transforms = bank["T_base_s"][candidate_ids]
    features: dict[str, float] = {}
    for index, name in enumerate(("u", "v", "d", "tilt")):
        features.update(_linear_descriptors(parameters[:, index], name))
    features.update(_circular_descriptors(parameters[:, 4], "azimuth"))
    parameter_names = tuple(str(value) for value in bank["parameter_names"])
    sixth_name = (
        "roll"
        if parameter_names == LEGACY_PARAMETER_NAMES
        else "normal_azimuth_sensor"
    )
    features.update(_circular_descriptors(parameters[:, 5], sixth_name))

    # The canonical generator defines sensor +Z as its plane-viewing direction.
    # Expressing it in the saved Phase-1 plane frame keeps the convention explicit.
    viewing_base = transforms[:, :3, 2]
    viewing_plane = viewing_base @ plane_basis
    features.update(_normal_descriptors(viewing_plane))
    features.update(_rotation_descriptors(transforms[:, :3, :3]))

    origins_base = transforms[:, :3, 3]
    origins_plane = (origins_base - plane_center[None, :]) @ plane_basis
    features.update(_position_descriptors(origins_plane))
    return features


def _load_and_validate(
    phase1_dir: Path, phase2a_dir: Path
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, np.ndarray],
    dict[int, list[dict[str, str]]],
    dict[int, np.ndarray],
]:
    phase1_manifest = _load_json(phase1_dir / "manifest.json")
    phase2a_manifest = _load_json(phase2a_dir / "manifest.json")
    if phase1_manifest.get("schema") != PHASE1_SCHEMA:
        raise ValueError(f"unsupported Phase-1 schema: {phase1_manifest.get('schema')}")
    if phase2a_manifest.get("schema") != PHASE2A_SCHEMA:
        raise ValueError(f"unsupported Phase-2A schema: {phase2a_manifest.get('schema')}")
    if phase1_manifest.get("status") != "complete":
        raise ValueError("Phase-1 result must be complete")
    if phase2a_manifest.get("status") != "complete":
        raise ValueError("Phase-2A result must be complete")
    if set(map(int, phase2a_manifest.get("completed_scan_counts", []))) != set(
        PRIMARY_N + DIAGNOSTIC_N
    ):
        raise ValueError("Phase-2A completed scan counts do not match 5,7,9,11,13,15")

    bank_path = _verify_declared_file(
        phase1_dir, phase1_manifest, "candidate_bank.npz"
    )
    with np.load(bank_path, allow_pickle=False) as archive:
        bank = {name: archive[name].copy() for name in archive.files}
    names = tuple(str(value) for value in bank["parameter_names"])
    if names not in (LEGACY_PARAMETER_NAMES, PARAMETER_NAMES):
        raise ValueError(f"unexpected Phase-1 parameter order: {names}")

    selected_path = _verify_declared_file(
        phase2a_dir, phase2a_manifest, "selected_phase1_subsets.npz"
    )
    summaries: dict[int, list[dict[str, str]]] = {}
    candidate_matrices: dict[int, np.ndarray] = {}
    with np.load(selected_path, allow_pickle=False) as selected:
        for n in sorted(DIAGNOSTIC_N + PRIMARY_N):
            relative = f"per_n/N{n:03d}/subset_summary.csv"
            summary_path = _verify_declared_file(phase2a_dir, phase2a_manifest, relative)
            rows = _read_csv(summary_path)
            matrix = selected[f"candidate_ids_N{n:03d}"].copy()
            subset_ids = selected[f"subset_ids_N{n:03d}"].copy()
            if matrix.shape != (len(rows), n) or subset_ids.shape != (len(rows),):
                raise ValueError(f"selected subset archive shape mismatch for N={n}")
            by_id = {int(row["subset_id"]): row for row in rows}
            if set(by_id) != set(map(int, subset_ids)):
                raise ValueError(f"Phase-2A subset IDs mismatch for N={n}")
            ordered = [by_id[int(subset_id)] for subset_id in subset_ids]
            for index, row in enumerate(ordered):
                declared = np.asarray(json.loads(row["candidate_ids"]), dtype=np.int64)
                if not np.array_equal(declared, matrix[index]):
                    raise ValueError(f"candidate_ids mismatch for N={n}, row={index}")
            summaries[n] = ordered
            candidate_matrices[n] = matrix

            # Validate summary metrics against the saved run-level calibration data.
            run_relative = f"per_n/N{n:03d}/calibration_runs.csv"
            run_path = _verify_declared_file(phase2a_dir, phase2a_manifest, run_relative)
            runs_by_subset: dict[int, list[dict[str, str]]] = defaultdict(list)
            for run_row in _read_csv(run_path):
                runs_by_subset[int(run_row["subset_id"])].append(run_row)
            for row in ordered:
                subset_id = int(row["subset_id"])
                run_rows = runs_by_subset[subset_id]
                if len(run_rows) != int(row["runs"]):
                    raise ValueError(f"run count mismatch for N={n}, subset={subset_id}")
                success = np.mean(
                    [item["passes_success_threshold"].lower() == "true" for item in run_rows]
                )
                normalized = np.asarray(
                    [_float(item, "normalized_error") for item in run_rows], dtype=float
                )
                finite = normalized[np.isfinite(normalized)]
                if not np.isclose(success, _float(row, "accuracy_success_rate")):
                    raise ValueError(f"accuracy summary mismatch for N={n}, subset={subset_id}")
                if finite.size:
                    if not np.isclose(
                        np.quantile(finite, 0.9), _float(row, "normalized_error_p90")
                    ):
                        raise ValueError(f"p90 summary mismatch for N={n}, subset={subset_id}")
                    if not np.isclose(
                        np.quantile(finite, 0.5), _float(row, "normalized_error_median")
                    ):
                        raise ValueError(f"median summary mismatch for N={n}, subset={subset_id}")
    return phase1_manifest, phase2a_manifest, bank, summaries, candidate_matrices


def _build_feature_rows(
    phase1_manifest: dict[str, Any],
    bank: dict[str, np.ndarray],
    summaries: dict[int, list[dict[str, str]]],
    candidate_matrices: dict[int, np.ndarray],
) -> tuple[list[dict[str, Any]], list[str]]:
    plane = phase1_manifest["config"]["plane"]
    center = np.asarray(plane["center_base_mm"], dtype=float)
    u = np.asarray(plane["u_base"], dtype=float)
    v = np.asarray(plane["v_base"], dtype=float)
    normal = np.asarray(plane["normal_base"], dtype=float)
    basis = np.column_stack([u, v, normal])
    if not np.allclose(basis.T @ basis, np.eye(3), atol=1e-8):
        raise ValueError("saved Phase-1 plane frame is not orthonormal")

    rows: list[dict[str, Any]] = []
    feature_names: list[str] | None = None
    for n in sorted(summaries):
        for summary, candidate_ids in zip(summaries[n], candidate_matrices[n]):
            features = _geometry_features(candidate_ids, bank, center, basis)
            if feature_names is None:
                feature_names = list(features)
            elif list(features) != feature_names:
                raise RuntimeError("geometry feature order changed between subsets")
            rows.append(
                {
                    "N": n,
                    "subset_id": int(summary["subset_id"]),
                    "phase1_rank": int(summary["phase1_rank"]),
                    "phase2a_rank": int(summary["phase2a_rank_among_selected"]),
                    "candidate_ids": summary["candidate_ids"],
                    "robustness_group": _label(summary),
                    "accuracy_success_rate": _float(summary, "accuracy_success_rate"),
                    "normalized_error_p90": _float(summary, "normalized_error_p90"),
                    "normalized_error_median": _float(
                        summary, "normalized_error_median"
                    ),
                    "nonlinear_success_rate": _float(
                        summary, "nonlinear_success_rate"
                    ),
                    "finite_result_rate": _float(summary, "finite_result_rate"),
                    **features,
                }
            )
    assert feature_names is not None
    return rows, feature_names


def _finite_values(rows: Sequence[dict[str, Any]], field: str) -> np.ndarray:
    values = np.asarray([float(row[field]) for row in rows], dtype=float)
    return values[np.isfinite(values)]


def _group_statistics(
    rows: Sequence[dict[str, Any]], feature_names: Sequence[str]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for n in sorted({int(row["N"]) for row in rows}):
        n_rows = [row for row in rows if int(row["N"]) == n]
        groups: dict[str, list[dict[str, Any]]] = {
            name: [row for row in n_rows if row["robustness_group"] == name]
            for name in ("core", "borderline", "non_survivor")
        }
        if n in DIAGNOSTIC_N:
            ordered = sorted(n_rows, key=lambda row: int(row["phase2a_rank"]))
            count = max(1, int(math.ceil(0.10 * len(ordered))))
            groups["diagnostic_top_10pct"] = ordered[:count]
            groups["diagnostic_bottom_10pct"] = ordered[-count:]
        for group, group_rows in groups.items():
            for feature in feature_names:
                values = _finite_values(group_rows, feature)
                output.append(
                    {
                        "N": n,
                        "analysis_role": (
                            "negative_control" if group.startswith("diagnostic") else "label_summary"
                        ),
                        "group": group,
                        "feature": feature,
                        "group_size": len(group_rows),
                        "finite_count": int(values.size),
                        "nonfinite_count": len(group_rows) - int(values.size),
                        "median": _quantile(values, 0.50) if values.size else math.nan,
                        "iqr": (
                            _quantile(values, 0.75) - _quantile(values, 0.25)
                            if values.size
                            else math.nan
                        ),
                        "mean": float(np.mean(values)) if values.size else math.nan,
                        "std": float(np.std(values, ddof=1)) if values.size > 1 else math.nan,
                    }
                )
    return output


def _cliffs_delta(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    if not len(x) or not len(y):
        return math.nan, math.nan, math.nan
    result = mannwhitneyu(x, y, alternative="two-sided", method="auto")
    u = float(result.statistic)
    delta = 2.0 * u / float(len(x) * len(y)) - 1.0
    return u, float(result.pvalue), delta


def _bh_adjust(p_values: Sequence[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    adjusted = np.full(values.shape, np.nan, dtype=float)
    finite_indices = np.flatnonzero(np.isfinite(values))
    if not finite_indices.size:
        return adjusted
    ordered_indices = finite_indices[np.argsort(values[finite_indices])]
    ordered_p = values[ordered_indices]
    count = len(ordered_p)
    ordered_q = ordered_p * count / np.arange(1, count + 1)
    ordered_q = np.minimum.accumulate(ordered_q[::-1])[::-1]
    adjusted[ordered_indices] = np.clip(ordered_q, 0.0, 1.0)
    return adjusted


def _effects_for_comparison(
    rows: Sequence[dict[str, Any]],
    feature_names: Sequence[str],
    n_values: Sequence[int],
    left_group: str,
    right_group: str,
    comparison_kind: str,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for n in n_values:
        n_rows = [row for row in rows if int(row["N"]) == n]
        if left_group.startswith("diagnostic"):
            ordered = sorted(n_rows, key=lambda row: int(row["phase2a_rank"]))
            count = max(1, int(math.ceil(0.10 * len(ordered))))
            left_rows, right_rows = ordered[:count], ordered[-count:]
        else:
            left_rows = [row for row in n_rows if row["robustness_group"] == left_group]
            right_rows = [row for row in n_rows if row["robustness_group"] == right_group]
        n_start = len(output)
        for feature in feature_names:
            left = _finite_values(left_rows, feature)
            right = _finite_values(right_rows, feature)
            u, p, delta = _cliffs_delta(left, right)
            output.append(
                {
                    "comparison_kind": comparison_kind,
                    "N": n,
                    "feature": feature,
                    "left_group": left_group,
                    "right_group": right_group,
                    "left_size": len(left_rows),
                    "right_size": len(right_rows),
                    "left_finite": int(left.size),
                    "right_finite": int(right.size),
                    "left_median": _quantile(left, 0.5) if left.size else math.nan,
                    "right_median": _quantile(right, 0.5) if right.size else math.nan,
                    "median_difference": (
                        _quantile(left, 0.5) - _quantile(right, 0.5)
                        if left.size and right.size
                        else math.nan
                    ),
                    "mann_whitney_u": u,
                    "p_value": p,
                    "fdr_q_value": math.nan,
                    "cliffs_delta": delta,
                    "effect_direction": (
                        "+" if delta > 0.0 else "-" if delta < 0.0 else "0"
                    ) if np.isfinite(delta) else "NA",
                }
            )
        q_values = _bh_adjust([row["p_value"] for row in output[n_start:]])
        for row, q_value in zip(output[n_start:], q_values):
            row["fdr_q_value"] = float(q_value)
    return output


def _continuous_correlations(
    rows: Sequence[dict[str, Any]], feature_names: Sequence[str]
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for n in sorted({int(row["N"]) for row in rows}):
        n_rows = [row for row in rows if int(row["N"]) == n]
        n_start = len(output)
        for performance in PERFORMANCE_FIELDS:
            for feature in feature_names:
                feature_values = np.asarray(
                    [float(row[feature]) for row in n_rows], dtype=float
                )
                performance_values = np.asarray(
                    [float(row[performance]) for row in n_rows], dtype=float
                )
                finite = np.isfinite(feature_values) & np.isfinite(performance_values)
                if np.sum(finite) >= 3 and np.unique(feature_values[finite]).size >= 2:
                    result = spearmanr(feature_values[finite], performance_values[finite])
                    rho, p = float(result.correlation), float(result.pvalue)
                else:
                    rho, p = math.nan, math.nan
                output.append(
                    {
                        "N": n,
                        "feature": feature,
                        "performance_metric": performance,
                        "total_count": len(n_rows),
                        "finite_pair_count": int(np.sum(finite)),
                        "nonfinite_feature_count": int(np.sum(~np.isfinite(feature_values))),
                        "nonfinite_performance_count": int(
                            np.sum(~np.isfinite(performance_values))
                        ),
                        "spearman_rho": rho,
                        "p_value": p,
                        "fdr_q_value": math.nan,
                    }
                )
        # Correct across all feature × performance tests within an N.
        q_values = _bh_adjust([row["p_value"] for row in output[n_start:]])
        for row, q_value in zip(output[n_start:], q_values):
            row["fdr_q_value"] = float(q_value)
    return output


def _consistent_candidates(
    effects: Sequence[dict[str, Any]],
    correlations: Sequence[dict[str, Any]],
    feature_names: Sequence[str],
) -> list[dict[str, Any]]:
    effect_lookup = {
        (int(row["N"]), str(row["feature"])): float(row["cliffs_delta"])
        for row in effects
    }
    correlation_lookup = {
        (int(row["N"]), str(row["feature"]), str(row["performance_metric"])): float(
            row["spearman_rho"]
        )
        for row in correlations
    }
    output: list[dict[str, Any]] = []
    for feature in feature_names:
        deltas = [effect_lookup[(n, feature)] for n in PRIMARY_N]
        positive_count = sum(value > 0.0 for value in deltas)
        negative_count = sum(value < 0.0 for value in deltas)
        if positive_count >= negative_count:
            dominant_sign, same_count = 1, positive_count
        else:
            dominant_sign, same_count = -1, negative_count
        large_count = sum(
            np.isfinite(value)
            and np.sign(value) == dominant_sign
            and abs(value) >= LARGE_EFFECT
            for value in deltas
        )
        transformed_correlations: list[float] = []
        success_agreement = 0
        p90_agreement = 0
        median_agreement = 0
        for n in PRIMARY_N:
            success_rho = correlation_lookup[(n, feature, "accuracy_success_rate")]
            p90_rho = correlation_lookup[(n, feature, "normalized_error_p90")]
            median_rho = correlation_lookup[(n, feature, "normalized_error_median")]
            if np.isfinite(success_rho):
                transformed_correlations.append(success_rho)
                success_agreement += int(success_rho * dominant_sign >= 0.0)
            if np.isfinite(p90_rho):
                transformed_correlations.append(-p90_rho)
                p90_agreement += int(-p90_rho * dominant_sign >= 0.0)
            if np.isfinite(median_rho):
                transformed_correlations.append(-median_rho)
                median_agreement += int(-median_rho * dominant_sign >= 0.0)
        continuous_score = (
            float(np.median(transformed_correlations))
            if transformed_correlations
            else math.nan
        )
        continuous_consistent = bool(
            np.isfinite(continuous_score)
            and continuous_score != 0.0
            and int(np.sign(continuous_score)) == dominant_sign
            and success_agreement >= 3
            and p90_agreement >= 3
        )
        qualifies = same_count >= 3 and large_count >= 2 and continuous_consistent
        output.append(
            {
                "feature": feature,
                **{f"cliffs_delta_N{n:03d}": effect_lookup[(n, feature)] for n in PRIMARY_N},
                **{
                    f"effect_direction_N{n:03d}": (
                        "+" if effect_lookup[(n, feature)] > 0.0 else "-"
                        if effect_lookup[(n, feature)] < 0.0 else "0"
                    )
                    for n in PRIMARY_N
                },
                "dominant_direction": "+" if dominant_sign > 0 else "-",
                "same_direction_n_count": same_count,
                "large_effect_n_count": large_count,
                "median_abs_cliffs_delta": float(np.median(np.abs(deltas))),
                "continuous_direction_score": continuous_score,
                "continuous_success_agreement_n_count": success_agreement,
                "continuous_p90_agreement_n_count": p90_agreement,
                "continuous_median_agreement_n_count": median_agreement,
                "continuous_direction_consistent": continuous_consistent,
                "consistent_candidate_feature": qualifies,
                "criterion_note": (
                    "hypothesis-discovery heuristic; not a calibrated rule"
                ),
            }
        )
    return sorted(
        output,
        key=lambda row: (
            -int(bool(row["consistent_candidate_feature"])),
            -int(row["same_direction_n_count"]),
            -int(row["large_effect_n_count"]),
            -float(row["median_abs_cliffs_delta"]),
        ),
    )


def _transition_analysis(
    rows: Sequence[dict[str, Any]], feature_names: Sequence[str]
) -> list[dict[str, Any]]:
    groups: dict[int, list[dict[str, Any]]] = {}
    for n in (7, 9):
        ordered = sorted(
            [row for row in rows if int(row["N"]) == n],
            key=lambda row: int(row["phase2a_rank"]),
        )
        groups[n] = ordered[: max(1, int(math.ceil(0.10 * len(ordered))))]
    output: list[dict[str, Any]] = []
    for feature in feature_names:
        n7 = _finite_values(groups[7], feature)
        n9 = _finite_values(groups[9], feature)
        u, p, delta = _cliffs_delta(n9, n7)
        output.append(
            {
                "analysis_role": "negative-control N=7 to N=9 transition diagnostic",
                "feature": feature,
                "N007_top10_median": _quantile(n7, 0.5),
                "N009_top10_median": _quantile(n9, 0.5),
                "median_change_N009_minus_N007": _quantile(n9, 0.5)
                - _quantile(n7, 0.5),
                "cliffs_delta_N009_vs_N007": delta,
                "mann_whitney_u": u,
                "p_value": p,
                "fdr_q_value": math.nan,
            }
        )
    q_values = _bh_adjust([row["p_value"] for row in output])
    for row, q_value in zip(output, q_values):
        row["fdr_q_value"] = float(q_value)
    return sorted(output, key=lambda row: -abs(float(row["cliffs_delta_N009_vs_N007"])))


def _plot_accuracy_distribution(rows: Sequence[dict[str, Any]], path: Path) -> None:
    n_values = sorted({int(row["N"]) for row in rows})
    data = [
        [float(row["accuracy_success_rate"]) for row in rows if int(row["N"]) == n]
        for n in n_values
    ]
    figure, axis = plt.subplots(figsize=(9, 6))
    parts = axis.violinplot(data, positions=n_values, widths=1.2, showextrema=False)
    for body in parts["bodies"]:
        body.set_alpha(0.25)
        body.set_facecolor("tab:blue")
    axis.boxplot(data, positions=n_values, widths=0.55, showfliers=False)
    axis.axhline(CORE_SUCCESS_RATE, color="tab:green", linestyle="--", label="core cutoff")
    axis.axhline(
        BORDERLINE_SUCCESS_RATE, color="tab:orange", linestyle=":", label="borderline cutoff"
    )
    axis.set(
        xlabel="Scan count N",
        ylabel="Phase-2A accuracy success rate",
        ylim=(-0.03, 1.03),
        xticks=n_values,
        title="Initialization robustness of the Phase-1 selected subsets",
    )
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _top_features(
    consistent: Sequence[dict[str, Any]], limit: int = 8
) -> list[str]:
    qualified = [row for row in consistent if row["consistent_candidate_feature"]]
    source = qualified if len(qualified) >= limit else list(consistent)
    return [str(row["feature"]) for row in source[:limit]]


def _plot_key_group_features(
    rows: Sequence[dict[str, Any]], features: Sequence[str], path: Path
) -> None:
    figure, axes = plt.subplots(2, 4, figsize=(17, 9))
    for axis, feature in zip(axes.flat, features):
        positions: list[float] = []
        data: list[np.ndarray] = []
        colors: list[str] = []
        for index, n in enumerate(PRIMARY_N):
            for offset, group, color in (
                (-0.18, "core", "tab:green"),
                (0.18, "non_survivor", "tab:gray"),
            ):
                values = _finite_values(
                    [
                        row
                        for row in rows
                        if int(row["N"]) == n and row["robustness_group"] == group
                    ],
                    feature,
                )
                if values.size:
                    data.append(values)
                    positions.append(index + 1 + offset)
                    colors.append(color)
        boxes = axis.boxplot(
            data,
            positions=positions,
            widths=0.30,
            showfliers=False,
            patch_artist=True,
        )
        for patch, color in zip(boxes["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.45)
        axis.set_xticks(range(1, len(PRIMARY_N) + 1), [str(n) for n in PRIMARY_N])
        axis.set_title(feature, fontsize=10)
        axis.grid(alpha=0.2)
    figure.supxlabel("Scan count N (green=core, gray=non-survivor; borderline excluded)")
    figure.supylabel("Geometry descriptor value (outliers hidden, raw CSV preserved)")
    figure.suptitle("Core versus non-survivor geometry within each N")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_effect_heatmap(
    effects: Sequence[dict[str, Any]], feature_names: Sequence[str], path: Path
) -> None:
    lookup = {
        (str(row["feature"]), int(row["N"])): float(row["cliffs_delta"])
        for row in effects
    }
    matrix = np.asarray(
        [[lookup[(feature, n)] for n in PRIMARY_N] for feature in feature_names],
        dtype=float,
    )
    figure, axis = plt.subplots(figsize=(8, max(12, 0.28 * len(feature_names))))
    image = axis.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-1.0, vmax=1.0)
    axis.set_xticks(range(len(PRIMARY_N)), [f"N={n}" for n in PRIMARY_N])
    axis.set_yticks(range(len(feature_names)), feature_names, fontsize=7)
    axis.set_title("Cliff's delta: core minus non-survivor")
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("positive = larger in core survivors")
    figure.tight_layout()
    figure.savefig(path, dpi=190)
    plt.close(figure)


def _plot_correlation_heatmap(
    correlations: Sequence[dict[str, Any]], feature_names: Sequence[str], path: Path
) -> None:
    primary = [row for row in correlations if int(row["N"]) in PRIMARY_N]
    lookup = {
        (str(row["feature"]), int(row["N"]), str(row["performance_metric"])): float(
            row["spearman_rho"]
        )
        for row in primary
    }
    columns = [(n, metric) for n in PRIMARY_N for metric in PERFORMANCE_FIELDS]
    matrix = np.asarray(
        [[lookup[(feature, n, metric)] for n, metric in columns] for feature in feature_names]
    )
    labels = [
        f"N={n}\n{ {'accuracy_success_rate':'success', 'normalized_error_p90':'p90', 'normalized_error_median':'median'}[metric] }"
        for n, metric in columns
    ]
    figure, axis = plt.subplots(figsize=(14, max(12, 0.28 * len(feature_names))))
    image = axis.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-1.0, vmax=1.0)
    axis.set_xticks(range(len(columns)), labels, rotation=45, ha="right", fontsize=8)
    axis.set_yticks(range(len(feature_names)), feature_names, fontsize=7)
    axis.set_title("Spearman correlation with continuous Phase-2A performance")
    colorbar = figure.colorbar(image, ax=axis)
    colorbar.set_label("Spearman rho (finite pairs only)")
    figure.tight_layout()
    figure.savefig(path, dpi=190)
    plt.close(figure)


def _plot_features_vs_p90(
    rows: Sequence[dict[str, Any]], features: Sequence[str], path: Path
) -> None:
    figure, axes = plt.subplots(2, 4, figsize=(17, 9))
    colors = {9: "tab:blue", 11: "tab:orange", 13: "tab:green", 15: "tab:red"}
    for axis, feature in zip(axes.flat, features):
        excluded = 0
        for n in PRIMARY_N:
            n_rows = [row for row in rows if int(row["N"]) == n]
            x = np.asarray([float(row[feature]) for row in n_rows])
            y = np.asarray([float(row["normalized_error_p90"]) for row in n_rows])
            finite = np.isfinite(x) & np.isfinite(y) & (y > 0.0)
            excluded += int(np.sum(~finite))
            axis.scatter(x[finite], y[finite], s=16, alpha=0.55, color=colors[n], label=f"N={n}")
        axis.set_yscale("log")
        axis.set_title(feature + (f"\n({excluded} nonfinite omitted)" if excluded else ""), fontsize=9)
        axis.grid(alpha=0.2, which="both")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=4)
    figure.supxlabel("Geometry descriptor")
    figure.supylabel("Phase-2A normalized error p90 (log scale)")
    figure.suptitle("Leading exploratory geometry features versus robustness", y=0.995)
    figure.tight_layout(rect=(0, 0, 1, 0.92))
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _plot_transition(
    rows: Sequence[dict[str, Any]], features: Sequence[str], path: Path
) -> None:
    top: dict[int, list[dict[str, Any]]] = {}
    for n in (7, 9):
        ordered = sorted(
            [row for row in rows if int(row["N"]) == n],
            key=lambda row: int(row["phase2a_rank"]),
        )
        top[n] = ordered[:10]
    figure, axes = plt.subplots(2, 4, figsize=(17, 9))
    for axis, feature in zip(axes.flat, features):
        values = [_finite_values(top[n], feature) for n in (7, 9)]
        boxes = axis.boxplot(values, labels=["N=7 top 10", "N=9 top 10"], showfliers=True, patch_artist=True)
        for patch, color in zip(boxes["boxes"], ("tab:gray", "tab:blue")):
            patch.set_facecolor(color)
            patch.set_alpha(0.45)
        axis.set_title(feature, fontsize=10)
        axis.tick_params(axis="x", labelrotation=15)
        axis.grid(alpha=0.2)
    figure.suptitle("Negative-control diagnostic: Phase-2A top 10%, N=7 versus N=9")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _human_direction(feature: str, direction: str) -> str:
    return f"larger `{feature}`" if direction == "+" else f"smaller `{feature}`"


def _write_observations(
    path: Path,
    rows: Sequence[dict[str, Any]],
    consistent: Sequence[dict[str, Any]],
    transition: Sequence[dict[str, Any]],
) -> None:
    counts = {
        n: {
            group: sum(
                int(row["N"]) == n and row["robustness_group"] == group
                for row in rows
            )
            for group in ("core", "borderline", "non_survivor")
        }
        for n in sorted(DIAGNOSTIC_N + PRIMARY_N)
    }
    candidates = [row for row in consistent if row["consistent_candidate_feature"]]
    positives = [row for row in candidates if row["dominant_direction"] == "+"]
    negatives = [row for row in candidates if row["dominant_direction"] == "-"]
    changing = [row for row in consistent if int(row["same_direction_n_count"]) <= 2]
    negligible = sorted(
        consistent, key=lambda row: float(row["median_abs_cliffs_delta"])
    )[:5]
    top_transition = list(transition[:5])

    lines = [
        "# Phase 2A geometry observations",
        "",
        "This is an exploratory observation of Phase 1 + Phase 2A results only. "
        "It does not establish causality, necessity, optimality, or a guaranteed calibration rule. "
        "Phase 2B data, D-optimal/Fisher/Jacobian scores, exchange search, and newly generated subsets were not used.",
        "",
        "The viewing-normal descriptor uses the saved Phase-1 `T_base_s` sensor +Z axes, "
        "expressed in the saved plane frame. Spatial descriptors use the saved sensor origins "
        "expressed relative to the saved plane center. Thus the original pose convention is preserved.",
        "",
        "## Fixed group definitions",
        "",
        "- Core survivor: `accuracy_success_rate >= 0.90` and `normalized_error_p90 <= 1.0`.",
        "- Borderline: not core, `accuracy_success_rate >= 0.80` and `normalized_error_p90 <= 1.5`.",
        "- Non-survivor: all remaining selected subsets.",
        "",
        "## 1. Core survivor counts",
        "",
        "| N | Core | Borderline | Non-survivor | Total |",
        "|---:|---:|---:|---:|---:|",
    ]
    for n, group_counts in counts.items():
        lines.append(
            f"| {n} | {group_counts['core']} | {group_counts['borderline']} | "
            f"{group_counts['non_survivor']} | {sum(group_counts.values())} |"
        )

    def feature_lines(items: Sequence[dict[str, Any]], limit: int = 5) -> list[str]:
        if not items:
            return ["- No feature met the fixed exploratory consistency heuristic."]
        return [
            f"- `{row['feature']}`: dominant {row['dominant_direction']} direction in "
            f"{row['same_direction_n_count']}/4 N values; large effect in "
            f"{row['large_effect_n_count']}/4."
            for row in items[:limit]
        ]

    lines += [
        "",
        "## 2. Most consistent positive candidate features",
        "",
        *feature_lines(positives),
        "",
        "Core survivors tend to show larger values for the positive candidates above across multiple N values. "
        "These are candidate geometric characteristics associated with robustness, not rules.",
        "",
        "## 3. Most consistent negative candidate features",
        "",
        *feature_lines(negatives),
        "",
        "Core survivors tend to show smaller values for the negative candidates above across multiple N values. "
        "These associations still require independent validation.",
        "",
        "## 4. Features whose effect direction changes with N",
        "",
    ]
    if changing:
        for row in changing[:10]:
            directions = ", ".join(
                f"N={n}:{row[f'effect_direction_N{n:03d}']}" for n in PRIMARY_N
            )
            lines.append(f"- `{row['feature']}` — {directions}")
    else:
        lines.append("- None under the exact sign comparison.")

    lines += [
        "",
        "## 5. Features with little core/non separation",
        "",
        *[
            f"- `{row['feature']}`: median |Cliff's delta| = "
            f"{float(row['median_abs_cliffs_delta']):.3f}."
            for row in negligible
        ],
        "",
        "## 6. N=7 to N=9 transition diagnostic",
        "",
        "There are no N=5 or N=7 core survivors, so no forced core/non comparison was made for those N values. "
        "The following compares only the Phase-2A top 10% at N=7 and N=9 as a negative-control diagnostic; "
        "it is not primary evidence.",
        "",
    ]
    for row in top_transition:
        direction = "larger" if float(row["cliffs_delta_N009_vs_N007"]) > 0 else "smaller"
        lines.append(
            f"- `{row['feature']}` is {direction} in the N=9 top group "
            f"(Cliff's delta N9−N7 = {float(row['cliffs_delta_N009_vs_N007']):.3f})."
        )

    lines += [
        "",
        "## 7. Candidate hypotheses for a future independent Phase 2B validation",
        "",
    ]
    if candidates:
        for index, row in enumerate(candidates[:5], start=1):
            lines.append(
                f"{index}. Test whether {_human_direction(str(row['feature']), str(row['dominant_direction']))} "
                "remains associated with lower calibration error when GT hand-eye and plane geometry change."
            )
    else:
        lines.append(
            "No feature satisfied the fixed consistency heuristic; retain the strongest exploratory effects "
            "as unconfirmed observations rather than Phase 2B hypotheses."
        )

    lines += [
        "",
        "## Statistical interpretation",
        "",
        "Primary comparisons are within N for N=9,11,13,15 and exclude the borderline group. "
        "Mann–Whitney p-values are Benjamini–Hochberg corrected within each N across geometry features. "
        "Cliff's delta is positive when the descriptor is larger in core survivors. "
        "Continuous Spearman correlations are computed on finite pairs only; finite and non-finite counts are saved. "
        "A consistent candidate feature requires the same effect direction in at least three N values, "
        "|Cliff's delta| >= 0.33 in at least two, concordant success-rate and p90 correlation directions "
        "in at least three N values each, and a matching aggregate continuous-performance direction. "
        "This is a hypothesis-discovery heuristic, not a calibrated decision rule.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = _parser()
    phase1_dir = _resolve(args.phase1_dir)
    phase2a_dir = _resolve(args.phase2a_dir)
    output_dir = _prepare_output(_resolve(args.output_dir), args.overwrite)
    if output_dir == phase1_dir or output_dir == phase2a_dir:
        raise ValueError("output directory must differ from Phase-1 and Phase-2A inputs")

    print("Validating Phase 1 and Phase 2A source files...")
    phase1_manifest, _, bank, summaries, candidate_matrices = _load_and_validate(
        phase1_dir, phase2a_dir
    )
    rows, feature_names = _build_feature_rows(
        phase1_manifest, bank, summaries, candidate_matrices
    )
    group_statistics = _group_statistics(rows, feature_names)
    primary_effects = _effects_for_comparison(
        rows,
        feature_names,
        PRIMARY_N,
        "core",
        "non_survivor",
        "primary_within_N_core_vs_non_survivor",
    )
    diagnostic_effects = _effects_for_comparison(
        rows,
        feature_names,
        DIAGNOSTIC_N,
        "diagnostic_top_10pct",
        "diagnostic_bottom_10pct",
        "negative_control_within_N_top10_vs_bottom10",
    )
    correlations = _continuous_correlations(rows, feature_names)
    consistent = _consistent_candidates(primary_effects, correlations, feature_names)
    transition = _transition_analysis(rows, feature_names)

    _write_csv(output_dir / "subset_geometry_features.csv", rows)
    _write_csv(output_dir / "group_statistics.csv", group_statistics)
    _write_csv(output_dir / "core_vs_non_effects.csv", primary_effects)
    _write_csv(output_dir / "diagnostic_top_bottom_effects.csv", diagnostic_effects)
    _write_csv(output_dir / "continuous_correlations.csv", correlations)
    _write_csv(output_dir / "consistent_candidate_features.csv", consistent)
    _write_csv(output_dir / "transition_N007_N009.csv", transition)

    plots = output_dir / "plots"
    key_features = _top_features(consistent, limit=8)
    _plot_accuracy_distribution(rows, plots / "accuracy_success_rate_by_n.png")
    _plot_key_group_features(rows, key_features, plots / "key_features_core_vs_non.png")
    _plot_effect_heatmap(
        primary_effects, feature_names, plots / "core_vs_non_cliffs_delta_heatmap.png"
    )
    _plot_correlation_heatmap(
        correlations, feature_names, plots / "continuous_spearman_heatmap.png"
    )
    _plot_features_vs_p90(rows, key_features, plots / "key_features_vs_p90.png")
    transition_features = [str(row["feature"]) for row in transition[:8]]
    _plot_transition(
        rows, transition_features, plots / "N007_vs_N009_top_group_transition.png"
    )
    _write_observations(output_dir / "observations.md", rows, consistent, transition)

    print(f"Read {len(rows)} selected subsets")
    for n in sorted(summaries):
        n_rows = [row for row in rows if int(row["N"]) == n]
        counts = {
            group: sum(row["robustness_group"] == group for row in n_rows)
            for group in ("core", "borderline", "non_survivor")
        }
        print(
            f"N={n}: subsets={len(n_rows)}, core={counts['core']}, "
            f"borderline={counts['borderline']}, non={counts['non_survivor']}"
        )
    print(f"Generated {len(feature_names)} geometry features")
    print("Most consistent candidate features (top 10):")
    for row in consistent[:10]:
        marker = "candidate" if row["consistent_candidate_feature"] else "not-qualified"
        print(
            f"  {row['feature']}: direction={row['dominant_direction']}, "
            f"same_N={row['same_direction_n_count']}, "
            f"large_N={row['large_effect_n_count']}, {marker}"
        )
    print(f"Output: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
