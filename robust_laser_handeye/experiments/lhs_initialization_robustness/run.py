#!/usr/bin/env python3
"""Evaluate Phase-1 subsets under paired initial-error and noise scenarios."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys
import time
from typing import Any, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import spearmanr


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.lhs_random_subset_screening import run as phase1  # noqa: E402
from laser_handeye.data import LaserScan  # noqa: E402
from laser_handeye.se3 import make_T  # noqa: E402


SCHEMA = "laser_handeye.phase2a_initialization_robustness"
SCHEMA_VERSION = 1


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="evaluate two subsets for the first two N values in two environments",
    )
    parser.add_argument("--validate-only", action="store_true")
    return parser


def _load_json(path: Path) -> dict[str, Any]:
    with path.expanduser().open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _positive_float(value: Any, name: str) -> float:
    output = float(value)
    if not np.isfinite(output) or output <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return output


def _increasing_pair(value: Any, name: str) -> tuple[float, float]:
    values = np.asarray(value, dtype=float)
    if (
        values.shape != (2,)
        or not np.all(np.isfinite(values))
        or float(values[0]) >= float(values[1])
    ):
        raise ValueError(f"{name} must be an increasing finite pair")
    return float(values[0]), float(values[1])


def _resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def _validate_config(config: dict[str, Any]) -> dict[str, Any]:
    required = (
        "seed",
        "phase1_dir",
        "output_dir",
        "input",
        "initial_error",
        "noise",
        "classification",
    )
    missing = [name for name in required if name not in config]
    if missing:
        raise ValueError(f"config is missing fields: {missing}")
    seed = int(config["seed"])
    if isinstance(config["seed"], bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    top_fraction = float(config["input"]["top_fraction_per_n"])
    if not np.isfinite(top_fraction) or not 0.0 < top_fraction <= 1.0:
        raise ValueError("input.top_fraction_per_n must lie in (0, 1]")
    initial = config["initial_error"]
    translation_range = _increasing_pair(
        initial["translation_norm_mm"], "initial_error.translation_norm_mm"
    )
    if translation_range[0] < 0.0:
        raise ValueError("initial translation norm cannot be negative")
    rotation_range = _increasing_pair(
        initial["rotation_deg"], "initial_error.rotation_deg"
    )
    if rotation_range[0] < 0.0:
        raise ValueError("initial rotation angle cannot be negative")
    environments = _positive_int(initial["environments"], "initial_error.environments")
    noise = config["noise"]
    if noise["axis"] not in ("z", "xz"):
        raise ValueError("noise.axis must be 'z' or 'xz'")
    noise_std = _positive_float(noise["std_mm"], "noise.std_mm")
    noise_repeats = _positive_int(
        noise["repeats_per_environment"], "noise.repeats_per_environment"
    )
    classification = config["classification"]
    classification_t = _positive_float(
        classification["translation_mm"], "classification.translation_mm"
    )
    classification_r = _positive_float(
        classification["rotation_deg"], "classification.rotation_deg"
    )
    return {
        "seed": seed,
        "phase1_dir": _resolve_path(config["phase1_dir"]),
        "output_dir": _resolve_path(config["output_dir"]),
        "top_fraction": top_fraction,
        "translation_range_mm": translation_range,
        "rotation_range_deg": rotation_range,
        "environments": environments,
        "noise_axis": str(noise["axis"]),
        "noise_std_mm": noise_std,
        "noise_repeats": noise_repeats,
        "classification_translation_mm": classification_t,
        "classification_rotation_deg": classification_r,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(
            phase1._jsonable(payload), indent=2, sort_keys=True, allow_nan=False
        )
        + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    phase1._write_csv(path, rows)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _unit_sphere(rng: np.random.Generator) -> np.ndarray:
    vector = rng.normal(size=3)
    norm = float(np.linalg.norm(vector))
    while norm <= 1e-12:
        vector = rng.normal(size=3)
        norm = float(np.linalg.norm(vector))
    return vector / norm


def _load_phase1(
    phase1_dir: Path,
) -> tuple[
    dict[str, Any],
    dict[str, np.ndarray],
    dict[int, list[dict[str, str]]],
    dict[int, np.ndarray],
]:
    manifest_path = phase1_dir / "manifest.json"
    manifest = _load_json(manifest_path)
    if manifest.get("schema") != phase1.SCHEMA:
        raise ValueError(f"unsupported Phase-1 schema: {manifest.get('schema')!r}")
    if manifest.get("status") != "complete":
        raise ValueError("Phase-1 manifest must have status='complete'")
    scan_counts = [int(value) for value in manifest["completed_scan_counts"]]
    if not scan_counts:
        raise ValueError("Phase 1 has no completed scan counts")

    declared_hashes = manifest.get("files", {})

    def verify_file(relative_path: str) -> Path:
        path = phase1_dir / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"missing Phase-1 input: {path}")
        expected = declared_hashes.get(relative_path)
        if expected is not None and _sha256(path) != expected:
            raise ValueError(f"Phase-1 file hash mismatch: {relative_path}")
        return path

    candidate_path = verify_file("candidate_bank.npz")
    with np.load(candidate_path, allow_pickle=False) as archive:
        bank = {name: archive[name].copy() for name in archive.files}
    candidate_count = int(manifest["resolved"]["candidate_count"])
    if bank["candidate_ids"].shape != (candidate_count,):
        raise ValueError("Phase-1 candidate bank count does not match manifest")
    if bank["ideal_points_s"].shape[0] != candidate_count:
        raise ValueError("Phase-1 ideal profile count is inconsistent")

    summary_rows = _read_csv(verify_file("subset_summary.csv"))
    summaries: dict[int, list[dict[str, str]]] = {n: [] for n in scan_counts}
    for row in summary_rows:
        n = int(row["N"])
        if n in summaries:
            summaries[n].append(row)
    subsets: dict[int, np.ndarray] = {}
    for n in scan_counts:
        rows = summaries[n]
        if not rows:
            raise ValueError(f"Phase 1 has no subset summaries for N={n}")
        ranks = sorted(int(row["rank_in_N"]) for row in rows)
        if ranks != list(range(1, len(rows) + 1)):
            raise ValueError(f"Phase-1 rank_in_N is incomplete for N={n}")
        subset_relative = f"subsets/subsets_N{n:03d}.npz"
        subset_path = verify_file(subset_relative)
        with np.load(subset_path, allow_pickle=False) as archive:
            candidate_ids = archive["candidate_ids"].copy()
            subset_ids = archive["subset_ids"].copy()
        if candidate_ids.shape != (len(rows), n):
            raise ValueError(f"Phase-1 subset matrix has wrong shape for N={n}")
        if not np.array_equal(subset_ids, np.arange(len(rows))):
            raise ValueError(f"Phase-1 subset IDs are not contiguous for N={n}")
        subsets[n] = candidate_ids
    return manifest, bank, summaries, subsets


def _select_phase1_subsets(
    summaries: dict[int, list[dict[str, str]]],
    subsets: dict[int, np.ndarray],
    top_fraction: float,
    *,
    smoke: bool,
) -> dict[int, list[dict[str, Any]]]:
    selected: dict[int, list[dict[str, Any]]] = {}
    n_values = sorted(summaries)
    if smoke:
        n_values = n_values[:2]
    for n in n_values:
        ordered = sorted(summaries[n], key=lambda row: int(row["rank_in_N"]))
        count = int(math.ceil(len(ordered) * top_fraction))
        if smoke:
            count = min(2, count)
        chosen: list[dict[str, Any]] = []
        for row in ordered[:count]:
            subset_id = int(row["subset_id"])
            candidate_ids = subsets[n][subset_id].copy()
            declared = np.asarray(json.loads(row["candidate_ids"]), dtype=np.int64)
            if not np.array_equal(candidate_ids, declared):
                raise ValueError(
                    f"Phase-1 summary/subset candidate mismatch: N={n}, "
                    f"subset_id={subset_id}"
                )
            chosen.append(
                {
                    "N": n,
                    "subset_id": subset_id,
                    "phase1_rank": int(row["rank_in_N"]),
                    "candidate_ids_array": candidate_ids,
                    "phase1_threshold_pass_rate": float(row["threshold_pass_rate"]),
                    "phase1_nonlinear_success_rate": float(
                        row["nonlinear_success_rate"]
                    ),
                    "phase1_normalized_error_p90": float(
                        row["normalized_error_p90"]
                    ),
                    "phase1_normalized_error_median": float(
                        row["normalized_error_median"]
                    ),
                }
            )
        selected[n] = chosen
    return selected


def _make_scenario_bank(
    resolved: dict[str, Any], T_true: np.ndarray, *, smoke: bool
) -> dict[str, np.ndarray]:
    count = 2 if smoke else resolved["environments"]
    rng = np.random.default_rng(
        np.random.SeedSequence([resolved["seed"], 0x494E4954])
    )
    translation_norm = np.empty(count, dtype=float)
    translation_direction = np.empty((count, 3), dtype=float)
    translation_vector = np.empty((count, 3), dtype=float)
    rotation_deg = np.empty(count, dtype=float)
    rotation_axis = np.empty((count, 3), dtype=float)
    T_error = np.empty((count, 4, 4), dtype=float)
    T_init = np.empty((count, 4, 4), dtype=float)
    actual_translation_error = np.empty(count, dtype=float)
    actual_rotation_error = np.empty(count, dtype=float)

    for environment_id in range(count):
        translation_norm[environment_id] = rng.uniform(
            *resolved["translation_range_mm"]
        )
        translation_direction[environment_id] = _unit_sphere(rng)
        translation_vector[environment_id] = (
            translation_norm[environment_id]
            * translation_direction[environment_id]
        )
        rotation_deg[environment_id] = rng.uniform(*resolved["rotation_range_deg"])
        rotation_axis[environment_id] = _unit_sphere(rng)
        rotation = Rotation.from_rotvec(
            rotation_axis[environment_id]
            * np.deg2rad(rotation_deg[environment_id])
        ).as_matrix()
        T_error[environment_id] = make_T(
            rotation, translation_vector[environment_id]
        )
        T_init[environment_id] = T_true @ T_error[environment_id]
        (
            actual_translation_error[environment_id],
            actual_rotation_error[environment_id],
        ) = phase1._transform_errors(T_init[environment_id], T_true)

    return {
        "environment_ids": np.arange(count, dtype=np.int64),
        "translation_norm_mm": translation_norm,
        "translation_direction": translation_direction,
        "translation_vector_mm": translation_vector,
        "rotation_deg": rotation_deg,
        "rotation_axis": rotation_axis,
        "T_error": T_error,
        "T_init": T_init,
        "actual_translation_error_mm": actual_translation_error,
        "actual_rotation_error_deg": actual_rotation_error,
    }


def _scenario_rows(bank: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for environment_id in bank["environment_ids"]:
        index = int(environment_id)
        rows.append(
            {
                "environment_id": index,
                "translation_norm_mm": float(bank["translation_norm_mm"][index]),
                "translation_dx": float(bank["translation_direction"][index, 0]),
                "translation_dy": float(bank["translation_direction"][index, 1]),
                "translation_dz": float(bank["translation_direction"][index, 2]),
                "translation_x_mm": float(bank["translation_vector_mm"][index, 0]),
                "translation_y_mm": float(bank["translation_vector_mm"][index, 1]),
                "translation_z_mm": float(bank["translation_vector_mm"][index, 2]),
                "rotation_deg": float(bank["rotation_deg"][index]),
                "rotation_axis_x": float(bank["rotation_axis"][index, 0]),
                "rotation_axis_y": float(bank["rotation_axis"][index, 1]),
                "rotation_axis_z": float(bank["rotation_axis"][index, 2]),
                "actual_translation_error_mm": float(
                    bank["actual_translation_error_mm"][index]
                ),
                "actual_rotation_error_deg": float(
                    bank["actual_rotation_error_deg"][index]
                ),
            }
        )
    return rows


def _selected_rows(
    selected: dict[int, list[dict[str, Any]]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for n, subsets in sorted(selected.items()):
        for subset in subsets:
            rows.append(
                {
                    "N": n,
                    "subset_id": subset["subset_id"],
                    "phase1_rank": subset["phase1_rank"],
                    "candidate_ids": json.dumps(
                        subset["candidate_ids_array"].tolist(), separators=(",", ":")
                    ),
                    "phase1_threshold_pass_rate": subset[
                        "phase1_threshold_pass_rate"
                    ],
                    "phase1_nonlinear_success_rate": subset[
                        "phase1_nonlinear_success_rate"
                    ],
                    "phase1_normalized_error_p90": subset[
                        "phase1_normalized_error_p90"
                    ],
                    "phase1_normalized_error_median": subset[
                        "phase1_normalized_error_median"
                    ],
                }
            )
    return rows


def _write_selected_npz(
    path: Path,
    selected: dict[int, list[dict[str, Any]]],
    candidate_bank: dict[str, np.ndarray],
) -> None:
    payload: dict[str, np.ndarray] = {
        "parameter_names": candidate_bank["parameter_names"]
    }
    if "pose_convention" in candidate_bank:
        payload["pose_convention"] = candidate_bank["pose_convention"]
    for n, subsets in sorted(selected.items()):
        suffix = f"N{n:03d}"
        candidate_ids = np.stack(
            [subset["candidate_ids_array"] for subset in subsets]
        )
        payload[f"subset_ids_{suffix}"] = np.asarray(
            [subset["subset_id"] for subset in subsets], dtype=np.int64
        )
        payload[f"phase1_rank_{suffix}"] = np.asarray(
            [subset["phase1_rank"] for subset in subsets], dtype=np.int64
        )
        payload[f"candidate_ids_{suffix}"] = candidate_ids
        payload[f"normalized_pose_parameters_{suffix}"] = candidate_bank[
            "normalized_parameters"
        ][candidate_ids]
        payload[f"pose_parameters_{suffix}"] = candidate_bank["pose_parameters"][
            candidate_ids
        ]
    np.savez_compressed(path, **payload)


def _noisy_scans(
    *,
    resolved: dict[str, Any],
    candidate_bank: dict[str, np.ndarray],
    candidate_ids: np.ndarray,
    environment_id: int,
    noise_repeat: int,
) -> list[LaserScan]:
    scans: list[LaserScan] = []
    for candidate_id_value in candidate_ids:
        candidate_id = int(candidate_id_value)
        points = candidate_bank["ideal_points_s"][candidate_id].copy()
        rng = np.random.default_rng(
            np.random.SeedSequence(
                [
                    resolved["seed"],
                    0x4E4F4953,
                    environment_id,
                    noise_repeat,
                    candidate_id,
                ]
            )
        )
        if resolved["noise_axis"] == "z":
            points[:, 2] += rng.normal(
                0.0, resolved["noise_std_mm"], size=len(points)
            )
        else:
            points[:, [0, 2]] += rng.normal(
                0.0, resolved["noise_std_mm"], size=(len(points), 2)
            )
        scans.append(
            LaserScan(
                T_base_ef=candidate_bank["T_base_ef"][candidate_id],
                points_s=points,
                plane_id=0,
                scan_id=candidate_id,
                meta={
                    "candidate_id": candidate_id,
                    "environment_id": environment_id,
                    "noise_repeat": noise_repeat,
                },
            )
        )
    return scans


def _solver_resolved(
    phase1_manifest: dict[str, Any], resolved: dict[str, Any], T_init: np.ndarray
) -> dict[str, Any]:
    solver = phase1_manifest["config"]["solver"]
    return {
        "T_true": np.asarray(
            phase1_manifest["resolved"]["T_ef_s_true"], dtype=float
        ),
        "T_init": np.asarray(T_init, dtype=float),
        "iterative_plane_offset_mode": solver["iterative"]["plane_offset_mode"],
        "iterative_max_iter": int(solver["iterative"]["max_iter"]),
        "iterative_tol": float(solver["iterative"]["tol"]),
        "nonlinear_loss": solver["nonlinear"]["loss"],
        "nonlinear_max_nfev": int(solver["nonlinear"]["max_nfev"]),
        "nonlinear_tol": float(solver["nonlinear"]["tol"]),
        "classification_translation_mm": resolved[
            "classification_translation_mm"
        ],
        "classification_rotation_deg": resolved["classification_rotation_deg"],
    }


def _finite_quantile(rows: Sequence[dict[str, Any]], field: str, q: float) -> float:
    values = np.asarray([row[field] for row in rows], dtype=float)
    values = values[np.isfinite(values)]
    return float(np.quantile(values, q)) if len(values) else float("inf")


def _summarize(
    subset: dict[str, Any], rows: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    count = len(rows)
    finite_count = sum(not bool(row["catastrophic_failure"]) for row in rows)
    accuracy_count = sum(bool(row["passes_success_threshold"]) for row in rows)
    nonlinear_success_count = sum(bool(row["nonlinear_success"]) for row in rows)
    iterative_converged_count = sum(bool(row["iterative_converged"]) for row in rows)
    iterative_failure_count = sum(row["failure_stage"] == "iterative" for row in rows)
    nonlinear_failure_count = sum(
        bool(row["nonlinear_attempted"]) and not bool(row["nonlinear_success"])
        for row in rows
    )
    return {
        "source": "phase1_top_fraction",
        "N": subset["N"],
        "subset_id": subset["subset_id"],
        "phase1_rank": subset["phase1_rank"],
        "candidate_ids": json.dumps(
            subset["candidate_ids_array"].tolist(), separators=(",", ":")
        ),
        "runs": count,
        "iterative_convergence_rate": iterative_converged_count / count,
        "nonlinear_success_rate": nonlinear_success_count / count,
        "finite_result_rate": finite_count / count,
        "accuracy_success_rate": accuracy_count / count,
        "iterative_nonconverged_count": count - iterative_converged_count,
        "iterative_failure_count": iterative_failure_count,
        "nonlinear_failure_count": nonlinear_failure_count,
        "accuracy_failure_count": finite_count - accuracy_count,
        "catastrophic_failure_count": count - finite_count,
        "translation_error_median_mm": _finite_quantile(
            rows, "translation_error_mm", 0.50
        ),
        "translation_error_p90_mm": _finite_quantile(
            rows, "translation_error_mm", 0.90
        ),
        "rotation_error_median_deg": _finite_quantile(
            rows, "rotation_error_deg", 0.50
        ),
        "rotation_error_p90_deg": _finite_quantile(
            rows, "rotation_error_deg", 0.90
        ),
        "normalized_error_median": _finite_quantile(
            rows, "normalized_error", 0.50
        ),
        "normalized_error_p90": _finite_quantile(rows, "normalized_error", 0.90),
        "final_residual_rms_median_mm": _finite_quantile(
            rows, "nonlinear_final_rms_mm", 0.50
        ),
        "final_residual_rms_p90_mm": _finite_quantile(
            rows, "nonlinear_final_rms_mm", 0.90
        ),
        "phase1_threshold_pass_rate": subset["phase1_threshold_pass_rate"],
        "phase1_nonlinear_success_rate": subset[
            "phase1_nonlinear_success_rate"
        ],
        "phase1_normalized_error_p90": subset[
            "phase1_normalized_error_p90"
        ],
        "phase1_normalized_error_median": subset[
            "phase1_normalized_error_median"
        ],
        "phase2a_rank_among_selected": -1,
        "rank_change": 0,
        "runtime_sum_s": float(sum(float(row["runtime_s"]) for row in rows)),
    }


def _rank_summaries(summaries: list[dict[str, Any]]) -> float:
    def key(row: dict[str, Any]) -> tuple[float, ...]:
        return (
            -float(row["finite_result_rate"]),
            -float(row["nonlinear_success_rate"]),
            -float(row["accuracy_success_rate"]),
            float(row["normalized_error_p90"]),
            float(row["normalized_error_median"]),
            float(row["translation_error_p90_mm"]),
            float(row["rotation_error_p90_deg"]),
            int(row["subset_id"]),
        )

    ordered = sorted(summaries, key=key)
    for rank, row in enumerate(ordered, start=1):
        row["phase2a_rank_among_selected"] = rank
        row["rank_change"] = rank - int(row["phase1_rank"])
    if len(ordered) < 2:
        return float("nan")
    correlation_result = spearmanr(
        [int(row["phase1_rank"]) for row in ordered],
        [int(row["phase2a_rank_among_selected"]) for row in ordered],
    )
    correlation = getattr(
        correlation_result,
        "statistic",
        correlation_result.correlation,
    )
    return float(correlation)


def _finite_values(rows: Sequence[dict[str, Any]], field: str) -> np.ndarray:
    values = np.asarray([row[field] for row in rows], dtype=float)
    return values[np.isfinite(values)]


def _write_plots(
    directory: Path,
    n: int,
    runs: Sequence[dict[str, Any]],
    summaries: Sequence[dict[str, Any]],
    resolved: dict[str, Any],
) -> list[Path]:
    directory.mkdir()
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    panels = (
        (
            axes[0],
            _finite_values(runs, "translation_error_mm"),
            resolved["classification_translation_mm"],
            "Translation error [mm]",
        ),
        (
            axes[1],
            _finite_values(runs, "rotation_error_deg"),
            resolved["classification_rotation_deg"],
            "Rotation error [deg]",
        ),
        (
            axes[2],
            _finite_values(runs, "normalized_error"),
            1.0,
            "Normalized error",
        ),
    )
    for axis, values, threshold, label in panels:
        if len(values):
            axis.hist(values, bins=min(50, max(5, int(np.sqrt(len(values))))))
        axis.axvline(threshold, color="tab:red", linestyle="--")
        axis.set(xlabel=label, ylabel="Runs")
        axis.grid(alpha=0.2)
    figure.suptitle(f"Phase 2A N={n}: all finite calibration errors")
    figure.tight_layout()
    distribution_path = directory / "error_distributions.png"
    figure.savefig(distribution_path, dpi=160)
    plt.close(figure)

    ordered = sorted(summaries, key=lambda row: int(row["phase1_rank"]))
    phase1_rank = np.asarray([row["phase1_rank"] for row in ordered], dtype=int)
    success_rate = np.asarray(
        [row["accuracy_success_rate"] for row in ordered], dtype=float
    )
    p90 = np.asarray([row["normalized_error_p90"] for row in ordered], dtype=float)
    figure, primary = plt.subplots(figsize=(9, 5))
    primary.plot(phase1_rank, success_rate, "o-", markersize=3, label="accuracy success")
    primary.set(
        xlabel="Phase-1 rank",
        ylabel="Phase-2A accuracy success rate",
        ylim=(-0.02, 1.02),
        title=f"N={n}: Phase-1 rank versus initialization robustness",
    )
    secondary = primary.twinx()
    secondary.plot(phase1_rank, p90, ".", color="tab:orange", label="p90 normalized error")
    secondary.set_ylabel("Phase-2A p90 normalized error")
    primary.grid(alpha=0.2)
    lines = primary.lines + secondary.lines
    primary.legend(lines, [line.get_label() for line in lines], loc="best")
    figure.tight_layout()
    rank_metric_path = directory / "phase1_rank_vs_phase2a_metrics.png"
    figure.savefig(rank_metric_path, dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(6, 6))
    axis.scatter(
        [row["phase1_rank"] for row in summaries],
        [row["phase2a_rank_among_selected"] for row in summaries],
        s=22,
        alpha=0.7,
    )
    limit = len(summaries) + 1
    axis.plot([1, limit], [1, limit], "--", color="0.5")
    axis.set(
        xlabel="Phase-1 rank",
        ylabel="Phase-2A rank among selected",
        title=f"N={n}: rank stability",
        xlim=(0, limit),
        ylim=(0, limit),
    )
    axis.grid(alpha=0.2)
    figure.tight_layout()
    rank_path = directory / "phase1_vs_phase2a_rank.png"
    figure.savefig(rank_path, dpi=160)
    plt.close(figure)
    return [distribution_path, rank_metric_path, rank_path]


def _resolved_summary(
    resolved: dict[str, Any],
    phase1_manifest: dict[str, Any],
    selected: dict[int, list[dict[str, Any]]],
    scenario_count: int,
    noise_repeats: int,
) -> dict[str, Any]:
    selected_count = sum(len(values) for values in selected.values())
    total_runs = selected_count * scenario_count * noise_repeats
    return {
        "phase1_dir": resolved["phase1_dir"],
        "output_dir": resolved["output_dir"],
        "scan_counts": sorted(selected),
        "selected_subsets_per_n": {
            str(n): len(values) for n, values in sorted(selected.items())
        },
        "selected_subsets": selected_count,
        "initial_error_environments": scenario_count,
        "noise_repeats_per_environment": noise_repeats,
        "total_calibration_runs": total_runs,
        "T_ef_s_true": phase1_manifest["resolved"]["T_ef_s_true"],
        "plane": phase1_manifest["resolved"]["plane"],
        "solver": phase1_manifest["config"]["solver"],
        "pose_and_environment_policy": (
            "reuse Phase-1 candidate T_base_ef and ideal profiles; vary only "
            "right-local T_init and candidate-keyed measurement noise"
        ),
    }


def run(
    config: dict[str, Any],
    output_override: Path | None = None,
    *,
    smoke: bool = False,
) -> Path:
    resolved = _validate_config(config)
    if output_override is not None:
        resolved["output_dir"] = _resolve_path(output_override)
    output_dir: Path = resolved["output_dir"]
    if output_dir.exists() and next(output_dir.iterdir(), None) is not None:
        raise FileExistsError(
            f"output directory is not empty; refusing to overwrite: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    per_n_dir = output_dir / "per_n"
    per_n_dir.mkdir()

    phase1_manifest, candidate_bank, summaries, subset_banks = _load_phase1(
        resolved["phase1_dir"]
    )
    selected = _select_phase1_subsets(
        summaries, subset_banks, resolved["top_fraction"], smoke=smoke
    )
    T_true = np.asarray(phase1_manifest["resolved"]["T_ef_s_true"], dtype=float)
    scenarios = _make_scenario_bank(resolved, T_true, smoke=smoke)
    scenario_count = len(scenarios["environment_ids"])
    noise_repeats = 1 if smoke else resolved["noise_repeats"]

    selected_csv = output_dir / "selected_phase1_subsets.csv"
    selected_npz = output_dir / "selected_phase1_subsets.npz"
    scenario_csv = output_dir / "scenario_bank.csv"
    scenario_npz = output_dir / "scenario_bank.npz"
    _write_csv(selected_csv, _selected_rows(selected))
    _write_selected_npz(selected_npz, selected, candidate_bank)
    _write_csv(scenario_csv, _scenario_rows(scenarios))
    np.savez_compressed(scenario_npz, **scenarios)

    manifest = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "in_progress",
        "config": config,
        "resolved": _resolved_summary(
            resolved, phase1_manifest, selected, scenario_count, noise_repeats
        ),
        "smoke_override": bool(smoke),
        "phase1_manifest_sha256": _sha256(resolved["phase1_dir"] / "manifest.json"),
        "pairing": {
            "initial_error": "same scenario bank for every subset",
            "noise": (
                "SeedSequence(phase2a_seed, purpose, environment_id, "
                "noise_repeat, candidate_id)"
            ),
            "threshold_policy": "classification_only; no run or subset removal",
        },
        "completed_scan_counts": [],
        "per_n": {},
        "files": {},
    }
    for path in (selected_csv, selected_npz, scenario_csv, scenario_npz):
        manifest["files"][path.relative_to(output_dir).as_posix()] = _sha256(path)
    _write_json(output_dir / "manifest.json", manifest)

    selected_count = sum(len(values) for values in selected.values())
    total_runs = selected_count * scenario_count * noise_repeats
    completed_runs = 0
    experiment_started = time.perf_counter()
    all_runs: list[dict[str, Any]] = []
    all_summaries: list[dict[str, Any]] = []

    for n_index, (n, selected_subsets) in enumerate(sorted(selected.items()), start=1):
        n_started = time.perf_counter()
        n_runs: list[dict[str, Any]] = []
        n_summaries: list[dict[str, Any]] = []
        print(
            f"[N={n} {n_index}/{len(selected)}] evaluating "
            f"{len(selected_subsets)} Phase-1 subsets × {scenario_count} init "
            f"× {noise_repeats} noise...",
            flush=True,
        )
        for position, subset in enumerate(selected_subsets, start=1):
            subset_rows: list[dict[str, Any]] = []
            for environment_id in scenarios["environment_ids"]:
                environment = int(environment_id)
                solver_resolved = _solver_resolved(
                    phase1_manifest, resolved, scenarios["T_init"][environment]
                )
                for noise_repeat in range(noise_repeats):
                    scans = _noisy_scans(
                        resolved=resolved,
                        candidate_bank=candidate_bank,
                        candidate_ids=subset["candidate_ids_array"],
                        environment_id=environment,
                        noise_repeat=noise_repeat,
                    )
                    row = phase1._run_one_calibration(
                        resolved=solver_resolved,
                        scans=scans,
                        scan_count=n,
                        subset_id=int(subset["subset_id"]),
                        repeat=noise_repeat,
                        candidate_ids=subset["candidate_ids_array"],
                    )
                    row.update(
                        source="phase2a_initialization_robustness",
                        phase1_rank=int(subset["phase1_rank"]),
                        environment_id=environment,
                        initial_translation_error_mm=float(
                            scenarios["actual_translation_error_mm"][environment]
                        ),
                        initial_rotation_error_deg=float(
                            scenarios["actual_rotation_error_deg"][environment]
                        ),
                    )
                    subset_rows.append(row)
                    n_runs.append(row)
                    all_runs.append(row)
                    completed_runs += 1
            n_summaries.append(_summarize(subset, subset_rows))
            progress_interval = max(1, len(selected_subsets) // 20)
            if position % progress_interval == 0 or position == len(selected_subsets):
                elapsed = time.perf_counter() - experiment_started
                n_elapsed = time.perf_counter() - n_started
                total_eta = elapsed * (total_runs - completed_runs) / completed_runs
                n_eta = n_elapsed * (len(selected_subsets) - position) / position
                print(
                    f"[N={n} {n_index}/{len(selected)}] subsets "
                    f"{position}/{len(selected_subsets)} "
                    f"({100.0 * position / len(selected_subsets):5.1f}%), "
                    f"N ETA {phase1._format_duration(n_eta)} | runs "
                    f"{completed_runs}/{total_runs} "
                    f"({100.0 * completed_runs / total_runs:5.1f}%), "
                    f"elapsed {phase1._format_duration(elapsed)}, "
                    f"ETA {phase1._format_duration(total_eta)}",
                    flush=True,
                )

        spearman = _rank_summaries(n_summaries)
        all_summaries.extend(n_summaries)
        n_output = per_n_dir / f"N{n:03d}"
        n_output.mkdir()
        runs_path = n_output / "calibration_runs.csv"
        summary_path = n_output / "subset_summary.csv"
        _write_csv(runs_path, n_runs)
        _write_csv(summary_path, n_summaries)
        plot_paths = _write_plots(
            n_output / "plots", n, n_runs, n_summaries, resolved
        )
        completed_paths = (runs_path, summary_path, *plot_paths)
        for path in completed_paths:
            manifest["files"][path.relative_to(output_dir).as_posix()] = _sha256(path)
        manifest["per_n"][str(n)] = {
            "status": "complete",
            "selected_subsets": len(n_summaries),
            "calibration_runs": len(n_runs),
            "phase1_phase2a_rank_spearman": spearman,
            "directory": n_output.relative_to(output_dir).as_posix(),
        }
        manifest["completed_scan_counts"].append(n)
        _write_json(output_dir / "manifest.json", manifest)
        print(
            f"[N={n}] saved {len(n_runs)} runs; rank Spearman={spearman:.4g}; "
            f"time={phase1._format_duration(time.perf_counter() - n_started)}",
            flush=True,
        )

    runs_path = output_dir / "calibration_runs.csv"
    summary_path = output_dir / "subset_summary.csv"
    _write_csv(runs_path, all_runs)
    _write_csv(summary_path, all_summaries)
    manifest["files"][runs_path.name] = _sha256(runs_path)
    manifest["files"][summary_path.name] = _sha256(summary_path)
    manifest["status"] = "complete"
    manifest["counts"] = {
        "selected_subsets": len(all_summaries),
        "calibration_runs": len(all_runs),
        "accuracy_success_runs": sum(
            bool(row["passes_success_threshold"]) for row in all_runs
        ),
        "catastrophic_failures": sum(
            bool(row["catastrophic_failure"]) for row in all_runs
        ),
    }
    _write_json(output_dir / "manifest.json", manifest)
    return output_dir


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = _load_json(args.config)
    resolved = _validate_config(config)
    if args.validate_only:
        phase1_manifest, _bank, summaries, subset_banks = _load_phase1(
            resolved["phase1_dir"]
        )
        selected = _select_phase1_subsets(
            summaries, subset_banks, resolved["top_fraction"], smoke=args.smoke
        )
        scenario_count = 2 if args.smoke else resolved["environments"]
        noise_repeats = 1 if args.smoke else resolved["noise_repeats"]
        output_dir = (
            _resolve_path(args.output_dir)
            if args.output_dir is not None
            else resolved["output_dir"]
        )
        resolved["output_dir"] = output_dir
        print(
            json.dumps(
                phase1._jsonable(
                    _resolved_summary(
                        resolved,
                        phase1_manifest,
                        selected,
                        scenario_count,
                        noise_repeats,
                    )
                ),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    completed = run(config, args.output_dir, smoke=args.smoke)
    print(f"Completed Phase 2A initialization robustness: {completed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
