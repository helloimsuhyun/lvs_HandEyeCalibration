#!/usr/bin/env python3
"""Run paired nonlinear-refinement ablations on immutable collections.

Each trial is loaded once, receives one deterministic noise realization and
one deterministic raw initialization, and runs the alternating solver once.
The following estimates are then written in long form:

* ``alternating``
* ``alternating_fixed_planes``
* ``alternating_fixed_normals`` (offsets are profiled per candidate)
* ``alternating_refit``
* ``alternating_joint``
* ``initial_joint`` (joint nonlinear optimization from the raw initial guess)

Examples
--------

.. code-block:: bash

    PYTHONPATH=. python3 main/run_nonlinear_refinement_ablation.py \
      --collection single=dataset/fair_plane_initialization_shared_global/N108/single_plane \
      --collection three=dataset/fair_plane_initialization_shared_global/N108/three_plane \
      --initialization medium:100:15 \
      --initialization hard:200:30 \
      --output-dir results/nonlinear_refinement_ablation \
      --workers 4
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import itertools
import json
import math
import time
import traceback
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import wilcoxon

from laser_handeye.calibration_dataset import load_calibration_dataset
from laser_handeye.nonlinear_refinement import (
    point_to_plane_residuals,
    refine_handeye_nonlinear,
    refine_handeye_planes_nonlinear,
)
from main import calibrate as calibration_cli


SCHEMA = "laser_handeye.nonlinear_refinement_ablation"
SCHEMA_VERSION = 1

ARMS = (
    "alternating",
    "alternating_fixed_planes",
    "alternating_fixed_normals",
    "alternating_refit",
    "alternating_joint",
    "initial_joint",
)
ALTERNATING_DEPENDENT_ARMS = ARMS[1:5]

LONG_FIELDS = (
    "schema_version",
    "collection_label",
    "collection_path",
    "collection_manifest_sha256",
    "initialization_label",
    "trial_index",
    "pair_trial_index",
    "relative_path",
    "logical_dataset_sha256",
    "acquisition_mode",
    "arm",
    "parent_estimate",
    "noise_seed_key",
    "initialization_seed_key",
    "noise_axis",
    "noise_std_mm",
    "init_translation_range_mm",
    "init_angle_range_deg",
    "n_planes",
    "n_scans",
    "n_points",
    "init_translation_error_mm",
    "init_rotation_error_deg",
    "initial_estimate_sha256",
    "alternating_converged",
    "alternating_iterations",
    "alternating_runtime_s",
    "alternating_translation_error_mm",
    "alternating_rotation_error_deg",
    "alternating_estimate_sha256",
    "optimizer_success",
    "optimizer_status",
    "optimizer_message",
    "optimizer_iterations",
    "optimizer_nfev",
    "optimizer_njev",
    "optimizer_runtime_s",
    "pipeline_runtime_s",
    "translation_error_mm",
    "rotation_error_deg",
    "calibration_success",
    "outlier",
    "self_fit_plane_rms_mm",
    "true_plane_rms_mm",
    "plane_normal_error_deg",
    "plane_offset_error_mm",
    "initial_residual_rms_mm",
    "final_residual_rms_mm",
    "delta_translation_mm",
    "delta_rotation_deg",
    "mean_plane_normal_delta_deg",
    "max_plane_normal_delta_deg",
    "mean_plane_offset_delta_mm",
    "max_plane_offset_delta_mm",
    "jacobian_rank",
    "variable_count",
    "jacobian_condition",
    "scaled_jacobian_condition",
    "estimate_sha256",
    "estimate_T_json",
    "plane_normals_json",
    "plane_offsets_json",
    "error_type",
    "error_message",
    "error_traceback",
)

FLOAT_FIELDS = {
    "noise_std_mm",
    "init_translation_range_mm",
    "init_angle_range_deg",
    "init_translation_error_mm",
    "init_rotation_error_deg",
    "alternating_runtime_s",
    "alternating_translation_error_mm",
    "alternating_rotation_error_deg",
    "optimizer_runtime_s",
    "pipeline_runtime_s",
    "translation_error_mm",
    "rotation_error_deg",
    "self_fit_plane_rms_mm",
    "true_plane_rms_mm",
    "plane_normal_error_deg",
    "plane_offset_error_mm",
    "initial_residual_rms_mm",
    "final_residual_rms_mm",
    "delta_translation_mm",
    "delta_rotation_deg",
    "mean_plane_normal_delta_deg",
    "max_plane_normal_delta_deg",
    "mean_plane_offset_delta_mm",
    "max_plane_offset_delta_mm",
    "jacobian_condition",
    "scaled_jacobian_condition",
}

INTEGER_FIELDS = {
    "schema_version",
    "trial_index",
    "pair_trial_index",
    "n_planes",
    "n_scans",
    "n_points",
    "alternating_iterations",
    "optimizer_status",
    "optimizer_iterations",
    "optimizer_nfev",
    "optimizer_njev",
    "jacobian_rank",
    "variable_count",
}

BOOLEAN_FIELDS = {
    "alternating_converged",
    "optimizer_success",
    "calibration_success",
    "outlier",
}


@dataclass(frozen=True)
class CollectionSpec:
    label: str
    path: Path
    manifest_sha256: str
    acquisition_mode: str
    entries: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class InitializationSpec:
    label: str
    translation_range_mm: float
    angle_range_deg: float


@dataclass(frozen=True)
class TrialTask:
    collection_label: str
    collection_path: str
    collection_manifest_sha256: str
    acquisition_mode: str
    initialization_label: str
    init_translation_range_mm: float
    init_angle_range_deg: float
    trial_index: int
    pair_trial_index: int
    relative_path: str
    logical_dataset_sha256: str
    config: dict[str, Any]


def _positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return value


def _nonnegative_int(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return value


def _parse_labeled_path(text: str) -> tuple[str, Path]:
    if "=" in text:
        label, path_text = text.split("=", 1)
        label = label.strip()
        path_text = path_text.strip()
    else:
        path_text = text.strip()
        label = Path(path_text).name
    if not label or not path_text:
        raise argparse.ArgumentTypeError(
            "--collection must be LABEL=PATH or PATH"
        )
    return label, Path(path_text).expanduser()


def _parse_initialization(text: str) -> InitializationSpec:
    value = text.strip()
    if "=" in value:
        label, numbers = value.split("=", 1)
        parts = [label, *numbers.replace(":", ",").split(",")]
    else:
        parts = value.replace(",", ":").split(":")
    parts = [part.strip() for part in parts if part.strip()]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "--initialization must be LABEL:TRANSLATION_MM:ANGLE_DEG"
        )
    label = parts[0]
    try:
        translation = float(parts[1])
        angle = float(parts[2])
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "initialization ranges must be numbers"
        ) from exc
    if (
        not np.isfinite(translation)
        or not np.isfinite(angle)
        or translation < 0.0
        or angle < 0.0
    ):
        raise argparse.ArgumentTypeError(
            "initialization ranges must be finite and non-negative"
        )
    return InitializationSpec(label, translation, angle)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--collection",
        action="append",
        required=True,
        metavar="LABEL=PATH",
        help="Immutable collection; repeat for multiple collections.",
    )
    parser.add_argument(
        "--initialization",
        action="append",
        type=_parse_initialization,
        metavar="LABEL:TRANSLATION_MM:ANGLE_DEG",
        help=(
            "Initialization range; repeat for multiple conditions. "
            "Default: default:100:15."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--workers", type=_positive_int, default=1)

    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--max-trials", type=_positive_int)
    selection.add_argument("--only-trial", type=_nonnegative_int)
    parser.add_argument("--max-scans-per-trial", type=_positive_int)

    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument("--noise-axis", choices=("z", "xz"), default="xz")
    parser.add_argument("--noise-std-mm", type=float, default=0.2)
    parser.add_argument(
        "--init-mode",
        choices=("relative", "carlson"),
        default="carlson",
        help=(
            "The repeated LABEL:T:R ablation currently requires carlson. "
            "Relative initialization does not consume those error ranges."
        ),
    )
    parser.add_argument("--rel-offset", type=float, default=0.1)
    parser.add_argument(
        "--init-translation-perturbation",
        choices=("direction_norm", "box_xyz"),
        default="direction_norm",
    )
    parser.add_argument(
        "--init-rotation-perturbation",
        choices=("axis_angle", "euler_xyz"),
        default="axis_angle",
    )
    parser.add_argument("--max-iter", type=_positive_int, default=3000)
    parser.add_argument("--tol", type=float, default=1e-5)

    parser.add_argument(
        "--nonlinear-loss",
        choices=("linear", "soft_l1", "huber", "cauchy", "arctan"),
        default="linear",
        help=(
            "Use linear for the primary ablation. With robust losses, joint "
            "optimizes a robust plane objective while refit/fixed_normals "
            "still profile planes with PCA/arithmetic means."
        ),
    )
    parser.add_argument("--nonlinear-f-scale-mm", type=float, default=1.0)
    parser.add_argument(
        "--nonlinear-max-nfev",
        type=_positive_int,
        default=300,
    )
    parser.add_argument("--nonlinear-ftol", type=float, default=1e-10)
    parser.add_argument("--nonlinear-xtol", type=float, default=1e-10)
    parser.add_argument("--nonlinear-gtol", type=float, default=1e-10)

    parser.add_argument("--translation-success-mm", type=float, default=1.0)
    parser.add_argument("--rotation-success-deg", type=float, default=0.1)
    parser.add_argument("--outlier-translation-mm", type=float, default=5.0)
    parser.add_argument("--outlier-rotation-deg", type=float, default=1.0)

    parser.add_argument(
        "--bootstrap-samples",
        type=_positive_int,
        default=10_000,
    )
    parser.add_argument("--bootstrap-seed", type=int, default=20260731)
    parser.add_argument(
        "--translation-tie-tol-mm",
        type=float,
        default=1e-4,
        help="Practical/numerical equivalence margin for paired translation.",
    )
    parser.add_argument(
        "--rotation-tie-tol-deg",
        type=float,
        default=1e-4,
        help="Practical/numerical equivalence margin for paired rotation.",
    )
    parser.add_argument("--debug-traceback", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_transform(transform: np.ndarray) -> str:
    value = np.ascontiguousarray(
        np.asarray(transform, dtype="<f8").reshape(4, 4)
    )
    return hashlib.sha256(value.tobytes()).hexdigest()


def _stable_seed(base_seed: int, *parts: object) -> int:
    text = "\x1f".join([str(base_seed), *(str(part) for part in parts)])
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False)


def _default_row(task: TrialTask, arm: str) -> dict[str, Any]:
    row: dict[str, Any] = {field: "" for field in LONG_FIELDS}
    row.update({field: float("nan") for field in FLOAT_FIELDS})
    row.update({field: -1 for field in INTEGER_FIELDS})
    row.update({field: False for field in BOOLEAN_FIELDS})
    row.update(
        {
            "schema_version": SCHEMA_VERSION,
            "collection_label": task.collection_label,
            "collection_path": task.collection_path,
            "collection_manifest_sha256": (
                task.collection_manifest_sha256
            ),
            "initialization_label": task.initialization_label,
            "trial_index": task.trial_index,
            "pair_trial_index": task.pair_trial_index,
            "relative_path": task.relative_path,
            "logical_dataset_sha256": task.logical_dataset_sha256,
            "acquisition_mode": task.acquisition_mode,
            "arm": arm,
            "parent_estimate": (
                "raw_initial"
                if arm == "initial_joint"
                else ("none" if arm == "alternating" else "alternating")
            ),
            "noise_seed_key": f"{task.config['seed']}:{task.trial_index}:0",
            "initialization_seed_key": (
                f"{task.config['seed']}:{task.trial_index}:1"
            ),
            "noise_axis": task.config["noise_axis"],
            "noise_std_mm": task.config["noise_std_mm"],
            "init_translation_range_mm": (
                task.init_translation_range_mm
            ),
            "init_angle_range_deg": task.init_angle_range_deg,
        }
    )
    return row


def _failure(
    row: dict[str, Any],
    exc: BaseException,
    *,
    debug_traceback: bool,
) -> dict[str, Any]:
    row["optimizer_success"] = False
    row["calibration_success"] = False
    row["outlier"] = True
    row["error_type"] = type(exc).__name__
    row["error_message"] = str(exc)
    if debug_traceback:
        row["error_traceback"] = traceback.format_exc()
    return row


def _planes_to_json(
    planes: Mapping[int, tuple[np.ndarray, float]],
) -> tuple[str, str]:
    normals = {
        str(plane_id): np.asarray(normal, dtype=float).reshape(3).tolist()
        for plane_id, (normal, _offset) in sorted(planes.items())
    }
    offsets = {
        str(plane_id): float(offset)
        for plane_id, (_normal, offset) in sorted(planes.items())
    }
    return (
        json.dumps(normals, sort_keys=True, separators=(",", ":")),
        json.dumps(offsets, sort_keys=True, separators=(",", ":")),
    )


def _fixed_normal_planes(
    scans_by_plane: Mapping[int, Sequence[Any]],
    transform: np.ndarray,
    reference_planes: Mapping[int, tuple[np.ndarray, float]],
) -> dict[int, tuple[np.ndarray, float]]:
    output: dict[int, tuple[np.ndarray, float]] = {}
    for plane_id, scans in scans_by_plane.items():
        normal = np.asarray(
            reference_planes[plane_id][0],
            dtype=float,
        ).reshape(3)
        points = calibration_cli._reconstruct_points(scans, transform)
        output[plane_id] = (
            normal.copy(),
            float(np.mean(points @ normal)),
        )
    return output


def _residual_rms_refit(
    scans_by_plane: Mapping[int, Sequence[Any]],
    transform: np.ndarray,
) -> float:
    residuals = point_to_plane_residuals(
        list(scans_by_plane.items()),
        transform,
        plane_mode="refit",
    )
    return float(np.sqrt(np.mean(np.square(residuals))))


def _populate_estimate(
    row: dict[str, Any],
    *,
    transform: np.ndarray,
    planes: Mapping[int, tuple[np.ndarray, float]],
    scans_by_plane: Mapping[int, Sequence[Any]],
    true_transform: np.ndarray,
    true_planes: Mapping[int, tuple[np.ndarray, float]],
    optimizer_success: bool,
    config: Mapping[str, Any],
) -> None:
    translation_error, rotation_error = calibration_cli._transform_errors(
        transform,
        true_transform,
    )
    plane_normal_error, plane_offset_error = (
        calibration_cli._mean_plane_parameter_errors(
            planes,
            true_planes,
        )
    )
    normals_json, offsets_json = _planes_to_json(planes)
    row.update(
        {
            "optimizer_success": bool(optimizer_success),
            "translation_error_mm": translation_error,
            "rotation_error_deg": rotation_error,
            "calibration_success": bool(
                optimizer_success
                and translation_error
                <= float(config["translation_success_mm"])
                and rotation_error
                <= float(config["rotation_success_deg"])
            ),
            "outlier": bool(
                (not optimizer_success)
                or translation_error
                > float(config["outlier_translation_mm"])
                or rotation_error
                > float(config["outlier_rotation_deg"])
            ),
            "self_fit_plane_rms_mm": calibration_cli._self_fit_rms(
                dict(scans_by_plane),
                transform,
            ),
            "true_plane_rms_mm": calibration_cli._true_plane_rms(
                dict(scans_by_plane),
                transform,
                dict(true_planes),
            ),
            "plane_normal_error_deg": plane_normal_error,
            "plane_offset_error_mm": plane_offset_error,
            "estimate_sha256": _sha256_transform(transform),
            "estimate_T_json": json.dumps(
                np.asarray(transform, dtype=float).reshape(4, 4).tolist(),
                separators=(",", ":"),
            ),
            "plane_normals_json": normals_json,
            "plane_offsets_json": offsets_json,
        }
    )


def _populate_nonlinear_diagnostics(
    row: dict[str, Any],
    result: Any,
    runtime_s: float,
) -> None:
    row.update(
        {
            "optimizer_success": bool(result.success),
            "optimizer_status": int(result.status),
            "optimizer_message": str(result.message),
            "optimizer_nfev": int(result.nfev),
            "optimizer_njev": (
                -1 if result.njev is None else int(result.njev)
            ),
            "optimizer_runtime_s": float(runtime_s),
            "initial_residual_rms_mm": float(result.initial_rms_mm),
            "final_residual_rms_mm": float(result.final_rms_mm),
            "delta_translation_mm": float(result.delta_translation_mm),
            "delta_rotation_deg": float(result.delta_rotation_deg),
            "jacobian_rank": int(result.jacobian_rank),
            "variable_count": int(getattr(result, "variable_count", 6)),
            "jacobian_condition": float(result.jacobian_condition),
            "scaled_jacobian_condition": float(
                getattr(result, "scaled_jacobian_condition", float("nan"))
            ),
            "mean_plane_normal_delta_deg": float(
                getattr(
                    result,
                    "mean_plane_normal_delta_deg",
                    float("nan"),
                )
            ),
            "max_plane_normal_delta_deg": float(
                getattr(
                    result,
                    "max_plane_normal_delta_deg",
                    float("nan"),
                )
            ),
            "mean_plane_offset_delta_mm": float(
                getattr(
                    result,
                    "mean_plane_offset_delta_mm",
                    float("nan"),
                )
            ),
            "max_plane_offset_delta_mm": float(
                getattr(
                    result,
                    "max_plane_offset_delta_mm",
                    float("nan"),
                )
            ),
        }
    )


def _run_handeye_only_arm(
    row: dict[str, Any],
    *,
    plane_mode: str,
    scans_by_plane: Mapping[int, Sequence[Any]],
    start_transform: np.ndarray,
    reference_planes: Mapping[int, tuple[np.ndarray, float]],
    true_transform: np.ndarray,
    true_planes: Mapping[int, tuple[np.ndarray, float]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    started = time.perf_counter()
    kwargs: dict[str, Any] = {"plane_mode": plane_mode}
    if plane_mode != "refit":
        kwargs["planes"] = reference_planes
    result = refine_handeye_nonlinear(
        scans_by_plane,
        start_transform,
        loss=config["nonlinear_loss"],
        f_scale_mm=config["nonlinear_f_scale_mm"],
        max_nfev=config["nonlinear_max_nfev"],
        ftol=config["nonlinear_ftol"],
        xtol=config["nonlinear_xtol"],
        gtol=config["nonlinear_gtol"],
        **kwargs,
    )
    runtime_s = time.perf_counter() - started

    if plane_mode == "fixed":
        final_planes = dict(reference_planes)
    elif plane_mode == "fixed_normals":
        final_planes = _fixed_normal_planes(
            scans_by_plane,
            result.T_ef_s,
            reference_planes,
        )
    else:
        final_planes = calibration_cli._estimate_planes(
            dict(scans_by_plane),
            result.T_ef_s,
        )

    _populate_nonlinear_diagnostics(row, result, runtime_s)
    _populate_estimate(
        row,
        transform=result.T_ef_s,
        planes=final_planes,
        scans_by_plane=scans_by_plane,
        true_transform=true_transform,
        true_planes=true_planes,
        optimizer_success=result.success,
        config=config,
    )
    return row


def _run_joint_arm(
    row: dict[str, Any],
    *,
    scans_by_plane: Mapping[int, Sequence[Any]],
    start_transform: np.ndarray,
    reference_planes: Mapping[int, tuple[np.ndarray, float]],
    true_transform: np.ndarray,
    true_planes: Mapping[int, tuple[np.ndarray, float]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    started = time.perf_counter()
    result = refine_handeye_planes_nonlinear(
        scans_by_plane,
        start_transform,
        initial_planes=reference_planes,
        loss=config["nonlinear_loss"],
        f_scale_mm=config["nonlinear_f_scale_mm"],
        max_nfev=config["nonlinear_max_nfev"],
        ftol=config["nonlinear_ftol"],
        xtol=config["nonlinear_xtol"],
        gtol=config["nonlinear_gtol"],
    )
    runtime_s = time.perf_counter() - started
    final_planes = {
        plane_id: (
            result.plane_normals[plane_id],
            result.plane_offsets_mm[plane_id],
        )
        for plane_id in result.plane_normals
    }
    _populate_nonlinear_diagnostics(row, result, runtime_s)
    _populate_estimate(
        row,
        transform=result.T_ef_s,
        planes=final_planes,
        scans_by_plane=scans_by_plane,
        true_transform=true_transform,
        true_planes=true_planes,
        optimizer_success=result.success,
        config=config,
    )
    return row


def _run_trial_task(task: TrialTask) -> list[dict[str, Any]]:
    config = task.config
    rows = {arm: _default_row(task, arm) for arm in ARMS}

    try:
        dataset_clean = load_calibration_dataset(
            Path(task.collection_path) / task.relative_path
        )
        dataset_clean = calibration_cli._prefix_dataset(
            dataset_clean,
            config["max_scans_per_trial"],
        )
        true_transform = calibration_cli._extract_truth_transform(
            dataset_clean
        )
        true_planes = calibration_cli._extract_true_planes(dataset_clean)

        noise_seed, initialization_seed = np.random.SeedSequence(
            [config["seed"], task.trial_index]
        ).spawn(2)
        dataset = calibration_cli._add_noise(
            dataset_clean,
            np.random.default_rng(noise_seed),
            config["noise_std_mm"],
            config["noise_axis"],
        )
        scans_by_plane = calibration_cli._extract_scans_by_plane(dataset)

        helper_args = SimpleNamespace(
            init_mode=config["init_mode"],
            rel_offset=config["rel_offset"],
            init_translation_range_mm=task.init_translation_range_mm,
            init_angle_range_deg=task.init_angle_range_deg,
            init_translation_perturbation=(
                config["init_translation_perturbation"]
            ),
            init_rotation_perturbation=config["init_rotation_perturbation"],
            max_iter=config["max_iter"],
            tol=config["tol"],
        )
        initial_transform = calibration_cli._generic_initial_guess(
            true_transform,
            np.random.default_rng(initialization_seed),
            helper_args,
        )
        init_translation_error, init_rotation_error = (
            calibration_cli._transform_errors(
                initial_transform,
                true_transform,
            )
        )
        n_scans = sum(len(scans) for scans in scans_by_plane.values())
        n_points = sum(
            len(calibration_cli._scan_points(scan))
            for scans in scans_by_plane.values()
            for scan in scans
        )
        shared_fields = {
            "n_planes": len(scans_by_plane),
            "n_scans": n_scans,
            "n_points": n_points,
            "init_translation_error_mm": init_translation_error,
            "init_rotation_error_deg": init_rotation_error,
            "initial_estimate_sha256": _sha256_transform(initial_transform),
        }
        for row in rows.values():
            row.update(shared_fields)
    except Exception as exc:
        return [
            _failure(
                rows[arm],
                exc,
                debug_traceback=config["debug_traceback"],
            )
            for arm in ARMS
        ]

    alternating_transform: np.ndarray | None = None
    alternating_planes: dict[int, tuple[np.ndarray, float]] | None = None
    try:
        started = time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            alternating_result = calibration_cli._run_iterative(
                scans_by_plane,
                initial_transform,
                helper_args,
            )
        alternating_runtime = time.perf_counter() - started
        alternating_transform = np.asarray(
            alternating_result.T_ef_s,
            dtype=float,
        ).reshape(4, 4)
        alternating_planes = calibration_cli._estimate_planes(
            scans_by_plane,
            alternating_transform,
        )
        alternating_translation_error, alternating_rotation_error = (
            calibration_cli._transform_errors(
                alternating_transform,
                true_transform,
            )
        )
        iterations = int(
            getattr(
                alternating_result,
                "iterations",
                len(getattr(alternating_result, "T_history", [])),
            )
        )
        converged = bool(
            getattr(alternating_result, "converged", False)
        )
        alternating_fields = {
            "alternating_converged": converged,
            "alternating_iterations": iterations,
            "alternating_runtime_s": alternating_runtime,
            "alternating_translation_error_mm": (
                alternating_translation_error
            ),
            "alternating_rotation_error_deg": alternating_rotation_error,
            "alternating_estimate_sha256": _sha256_transform(
                alternating_transform
            ),
        }
        for row in rows.values():
            row.update(alternating_fields)

        alternating_row = rows["alternating"]
        alternating_row.update(
            {
                "optimizer_status": 1 if converged else 0,
                "optimizer_message": (
                    "alternating solver converged"
                    if converged
                    else "alternating solver did not converge"
                ),
                "optimizer_iterations": iterations,
                "optimizer_runtime_s": alternating_runtime,
                "initial_residual_rms_mm": _residual_rms_refit(
                    scans_by_plane,
                    initial_transform,
                ),
                "final_residual_rms_mm": _residual_rms_refit(
                    scans_by_plane,
                    alternating_transform,
                ),
                "delta_translation_mm": float(
                    np.linalg.norm(
                        alternating_transform[:3, 3]
                        - initial_transform[:3, 3]
                    )
                ),
                "delta_rotation_deg": calibration_cli.rot_error_deg(
                    alternating_transform[:3, :3],
                    initial_transform[:3, :3],
                ),
            }
        )
        rank_history = list(
            getattr(alternating_result, "rank_history", [])
        )
        condition_history = list(
            getattr(alternating_result, "cond_history", [])
        )
        if rank_history:
            alternating_row["jacobian_rank"] = int(rank_history[-1])
        if condition_history:
            alternating_row["jacobian_condition"] = float(
                condition_history[-1]
            )
        _populate_estimate(
            alternating_row,
            transform=alternating_transform,
            planes=alternating_planes,
            scans_by_plane=scans_by_plane,
            true_transform=true_transform,
            true_planes=true_planes,
            optimizer_success=converged,
            config=config,
        )
    except Exception as exc:
        alternating_transform = None
        alternating_planes = None
        _failure(
            rows["alternating"],
            exc,
            debug_traceback=config["debug_traceback"],
        )
        for arm in ALTERNATING_DEPENDENT_ARMS:
            dependency_error = RuntimeError(
                f"{arm} was not run because alternating failed: {exc}"
            )
            _failure(
                rows[arm],
                dependency_error,
                debug_traceback=False,
            )

    if alternating_transform is not None and alternating_planes is not None:
        handeye_modes = (
            ("alternating_fixed_planes", "fixed"),
            ("alternating_fixed_normals", "fixed_normals"),
            ("alternating_refit", "refit"),
        )
        for arm, plane_mode in handeye_modes:
            try:
                _run_handeye_only_arm(
                    rows[arm],
                    plane_mode=plane_mode,
                    scans_by_plane=scans_by_plane,
                    start_transform=alternating_transform,
                    reference_planes=alternating_planes,
                    true_transform=true_transform,
                    true_planes=true_planes,
                    config=config,
                )
            except Exception as exc:
                _failure(
                    rows[arm],
                    exc,
                    debug_traceback=config["debug_traceback"],
                )
        try:
            _run_joint_arm(
                rows["alternating_joint"],
                scans_by_plane=scans_by_plane,
                start_transform=alternating_transform,
                reference_planes=alternating_planes,
                true_transform=true_transform,
                true_planes=true_planes,
                config=config,
            )
        except Exception as exc:
            _failure(
                rows["alternating_joint"],
                exc,
                debug_traceback=config["debug_traceback"],
            )

    try:
        raw_planes = calibration_cli._estimate_planes(
            scans_by_plane,
            initial_transform,
        )
        _run_joint_arm(
            rows["initial_joint"],
            scans_by_plane=scans_by_plane,
            start_transform=initial_transform,
            reference_planes=raw_planes,
            true_transform=true_transform,
            true_planes=true_planes,
            config=config,
        )
    except Exception as exc:
        _failure(
            rows["initial_joint"],
            exc,
            debug_traceback=config["debug_traceback"],
        )

    for arm, row in rows.items():
        if row["error_type"]:
            continue
        incremental_runtime = float(row["optimizer_runtime_s"])
        if arm in ALTERNATING_DEPENDENT_ARMS:
            row["pipeline_runtime_s"] = (
                float(row["alternating_runtime_s"]) + incremental_runtime
            )
        else:
            # ``alternating`` and ``initial_joint`` each contain their whole
            # pipeline in optimizer_runtime_s.
            row["pipeline_runtime_s"] = incremental_runtime

    return [rows[arm] for arm in ARMS]


def _finite(values: Iterable[Any]) -> np.ndarray:
    output: list[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            output.append(number)
    return np.asarray(output, dtype=float)


def _distribution(values: Iterable[Any]) -> dict[str, Any]:
    array = _finite(values)
    if len(array) == 0:
        return {
            "count": 0,
            "median": None,
            "mean": None,
            "p95": None,
            "max": None,
        }
    return {
        "count": int(len(array)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "p95": float(np.percentile(array, 95)),
        "max": float(np.max(array)),
    }


def _method_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    completed = [
        row
        for row in rows
        if not row["error_type"]
        and np.isfinite(float(row["translation_error_mm"]))
        and np.isfinite(float(row["rotation_error_deg"]))
    ]
    return {
        "planned_trials": len(rows),
        "completed_trials": len(completed),
        "failure_trials": sum(bool(row["error_type"]) for row in rows),
        "optimizer_successful_trials": sum(
            bool(row["optimizer_success"]) for row in completed
        ),
        "calibration_successful_trials": sum(
            bool(row["calibration_success"]) for row in completed
        ),
        "outlier_trials": sum(bool(row["outlier"]) for row in rows),
        "translation_error_mm": _distribution(
            row["translation_error_mm"] for row in completed
        ),
        "rotation_error_deg": _distribution(
            row["rotation_error_deg"] for row in completed
        ),
        "final_residual_rms_mm": _distribution(
            row["final_residual_rms_mm"] for row in completed
        ),
        "plane_normal_error_deg": _distribution(
            row["plane_normal_error_deg"] for row in completed
        ),
        "plane_offset_error_mm": _distribution(
            row["plane_offset_error_mm"] for row in completed
        ),
        "optimizer_runtime_s": _distribution(
            row["optimizer_runtime_s"] for row in completed
        ),
        "pipeline_runtime_s": _distribution(
            row["pipeline_runtime_s"] for row in completed
        ),
        "optimizer_nfev": _distribution(
            row["optimizer_nfev"]
            for row in completed
            if int(row["optimizer_nfev"]) >= 0
        ),
        "full_rank_trials": sum(
            int(row["variable_count"]) > 0
            and int(row["jacobian_rank"]) == int(row["variable_count"])
            for row in completed
        ),
    }


def _bootstrap_median_ci(
    deltas: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> tuple[float | None, float | None]:
    if len(deltas) == 0:
        return None, None
    rng = np.random.default_rng(seed)
    # Chunking bounds peak memory for large user-supplied trial counts.
    estimates: list[np.ndarray] = []
    remaining = samples
    chunk_size = max(1, min(samples, 2048))
    while remaining:
        current = min(remaining, chunk_size)
        indices = rng.integers(
            0,
            len(deltas),
            size=(current, len(deltas)),
        )
        estimates.append(np.median(deltas[indices], axis=1))
        remaining -= current
    bootstrap = np.concatenate(estimates)
    lower, upper = np.percentile(bootstrap, [2.5, 97.5])
    return float(lower), float(upper)


def _wilcoxon_summary(
    deltas: np.ndarray,
    tie_tolerance: float,
) -> dict[str, Any]:
    nonzero = deltas[np.abs(deltas) > tie_tolerance]
    if len(nonzero) == 0:
        return {
            "n_nonzero": 0,
            "statistic": None,
            "pvalue_two_sided": None,
        }
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = wilcoxon(
                nonzero,
                zero_method="wilcox",
                alternative="two-sided",
                mode="auto",
            )
        return {
            "n_nonzero": int(len(nonzero)),
            "statistic": float(result.statistic),
            "pvalue_two_sided": float(result.pvalue),
        }
    except ValueError:
        return {
            "n_nonzero": int(len(nonzero)),
            "statistic": None,
            "pvalue_two_sided": None,
        }


def _wilson_interval(successes: int, trials: int) -> list[float | None]:
    """Return a two-sided 95% Wilson interval for a binomial fraction."""
    if trials <= 0:
        return [None, None]
    z = 1.959963984540054
    probability = successes / trials
    denominator = 1.0 + z * z / trials
    center = (probability + z * z / (2.0 * trials)) / denominator
    radius = (
        z
        * math.sqrt(
            probability * (1.0 - probability) / trials
            + z * z / (4.0 * trials * trials)
        )
        / denominator
    )
    return [float(center - radius), float(center + radius)]


def _paired_observations(
    reference: Sequence[dict[str, Any]],
    candidate: Sequence[dict[str, Any]],
    *,
    field: str,
) -> tuple[list[tuple[float, float]], dict[str, int]]:
    """Collect complete-case values and retain every exclusion denominator."""
    if len(reference) != len(candidate):
        raise ValueError("paired arms contain different row counts")

    counts = {
        "planned_pairs": len(reference),
        "complete_case_pairs": 0,
        "reference_exception_only": 0,
        "candidate_exception_only": 0,
        "both_exception": 0,
        "reference_nonfinite_only": 0,
        "candidate_nonfinite_only": 0,
        "both_nonfinite": 0,
        "both_optimizer_successful": 0,
        "reference_optimizer_failure_only": 0,
        "candidate_optimizer_failure_only": 0,
        "both_optimizer_failed": 0,
    }
    pairs: list[tuple[float, float]] = []

    for reference_row, candidate_row in zip(
        reference,
        candidate,
        strict=True,
    ):
        reference_error = bool(reference_row["error_type"])
        candidate_error = bool(candidate_row["error_type"])
        if reference_error or candidate_error:
            if reference_error and candidate_error:
                counts["both_exception"] += 1
            elif reference_error:
                counts["reference_exception_only"] += 1
            else:
                counts["candidate_exception_only"] += 1
            continue

        try:
            reference_value = float(reference_row[field])
            candidate_value = float(candidate_row[field])
        except (TypeError, ValueError):
            reference_value = float("nan")
            candidate_value = float("nan")
        reference_finite = bool(np.isfinite(reference_value))
        candidate_finite = bool(np.isfinite(candidate_value))
        if not (reference_finite and candidate_finite):
            if not reference_finite and not candidate_finite:
                counts["both_nonfinite"] += 1
            elif not reference_finite:
                counts["reference_nonfinite_only"] += 1
            else:
                counts["candidate_nonfinite_only"] += 1
            continue

        counts["complete_case_pairs"] += 1
        reference_success = bool(reference_row["optimizer_success"])
        candidate_success = bool(candidate_row["optimizer_success"])
        if reference_success and candidate_success:
            counts["both_optimizer_successful"] += 1
        elif not reference_success and not candidate_success:
            counts["both_optimizer_failed"] += 1
        elif not reference_success:
            counts["reference_optimizer_failure_only"] += 1
        else:
            counts["candidate_optimizer_failure_only"] += 1
        pairs.append((reference_value, candidate_value))

    return pairs, counts


def _paired_metric(
    reference: Sequence[dict[str, Any]],
    candidate: Sequence[dict[str, Any]],
    *,
    field: str,
    tie_tolerance: float,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    pairs, availability = _paired_observations(
        reference,
        candidate,
        field=field,
    )
    if not pairs:
        return {
            **availability,
            "analysis_population": "complete_case_accuracy",
            "paired_trials": 0,
            "reference_median": None,
            "candidate_median": None,
            "median_delta": None,
            "median_delta_bootstrap_95_ci": [None, None],
            "mean_delta": None,
            "win_fraction": None,
            "win_fraction_wilson_95_ci": [None, None],
            "tie_fraction": None,
            "loss_fraction": None,
            "wilcoxon": _wilcoxon_summary(
                np.asarray([], dtype=float),
                tie_tolerance,
            ),
        }

    values = np.asarray(pairs, dtype=float)
    deltas = values[:, 0] - values[:, 1]
    wins = int(np.sum(deltas > tie_tolerance))
    lower, upper = _bootstrap_median_ci(
        deltas,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    return {
        **availability,
        "analysis_population": "complete_case_accuracy",
        "paired_trials": int(len(deltas)),
        "both_optimizer_successful_trials": availability[
            "both_optimizer_successful"
        ],
        "reference_median": float(np.median(values[:, 0])),
        "candidate_median": float(np.median(values[:, 1])),
        "median_delta": float(np.median(deltas)),
        "median_delta_bootstrap_95_ci": [lower, upper],
        "mean_delta": float(np.mean(deltas)),
        "win_fraction": float(wins / len(deltas)),
        "win_fraction_wilson_95_ci": _wilson_interval(
            wins,
            len(deltas),
        ),
        "tie_fraction": float(
            np.mean(np.abs(deltas) <= tie_tolerance)
        ),
        "loss_fraction": float(np.mean(deltas < -tie_tolerance)),
        "wilcoxon": _wilcoxon_summary(deltas, tie_tolerance),
    }


def _paired_pareto(
    reference: Sequence[dict[str, Any]],
    candidate: Sequence[dict[str, Any]],
    *,
    translation_tolerance: float,
    rotation_tolerance: float,
) -> dict[str, Any]:
    categories = {
        "candidate_pareto_better": 0,
        "candidate_pareto_worse": 0,
        "tradeoff": 0,
        "tied": 0,
    }
    paired = 0
    for reference_row, candidate_row in zip(
        reference,
        candidate,
        strict=True,
    ):
        if reference_row["error_type"] or candidate_row["error_type"]:
            continue
        try:
            delta_translation = float(
                reference_row["translation_error_mm"]
            ) - float(candidate_row["translation_error_mm"])
            delta_rotation = float(
                reference_row["rotation_error_deg"]
            ) - float(candidate_row["rotation_error_deg"])
        except (TypeError, ValueError):
            continue
        if not (
            np.isfinite(delta_translation)
            and np.isfinite(delta_rotation)
        ):
            continue
        paired += 1
        translation_sign = (
            1
            if delta_translation > translation_tolerance
            else (
                -1
                if delta_translation < -translation_tolerance
                else 0
            )
        )
        rotation_sign = (
            1
            if delta_rotation > rotation_tolerance
            else (
                -1 if delta_rotation < -rotation_tolerance else 0
            )
        )
        if (
            translation_sign >= 0
            and rotation_sign >= 0
            and (translation_sign > 0 or rotation_sign > 0)
        ):
            categories["candidate_pareto_better"] += 1
        elif (
            translation_sign <= 0
            and rotation_sign <= 0
            and (translation_sign < 0 or rotation_sign < 0)
        ):
            categories["candidate_pareto_worse"] += 1
        elif translation_sign * rotation_sign < 0:
            categories["tradeoff"] += 1
        else:
            categories["tied"] += 1

    return {
        "paired_trials": paired,
        "counts": categories,
        "fractions": {
            key: (float(value / paired) if paired else None)
            for key, value in categories.items()
        },
    }


def _paired_outlier_transitions(
    reference: Sequence[dict[str, Any]],
    candidate: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    if len(reference) != len(candidate):
        raise ValueError("paired arms contain different row counts")
    counts = {
        "both_inlier": 0,
        "reference_outlier_candidate_inlier": 0,
        "reference_inlier_candidate_outlier": 0,
        "both_outlier": 0,
    }
    for reference_row, candidate_row in zip(
        reference,
        candidate,
        strict=True,
    ):
        reference_outlier = bool(reference_row["outlier"])
        candidate_outlier = bool(candidate_row["outlier"])
        if reference_outlier and candidate_outlier:
            counts["both_outlier"] += 1
        elif reference_outlier:
            counts["reference_outlier_candidate_inlier"] += 1
        elif candidate_outlier:
            counts["reference_inlier_candidate_outlier"] += 1
        else:
            counts["both_inlier"] += 1
    planned = len(reference)
    return {
        "planned_pairs": planned,
        "counts": counts,
        "fractions": {
            key: (float(value / planned) if planned else None)
            for key, value in counts.items()
        },
    }


def _paired_comparison(
    rows_by_arm: Mapping[str, Sequence[dict[str, Any]]],
    *,
    reference_arm: str,
    candidate_arm: str,
    condition_key: tuple[str, str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    reference = rows_by_arm[reference_arm]
    candidate = rows_by_arm[candidate_arm]
    seed_parts = (
        condition_key[0],
        condition_key[1],
        reference_arm,
        candidate_arm,
    )
    return {
        "reference_arm": reference_arm,
        "candidate_arm": candidate_arm,
        "delta_definition": "reference_error_minus_candidate_error",
        "positive_delta_means": "candidate_is_better",
        "translation_error_mm": _paired_metric(
            reference,
            candidate,
            field="translation_error_mm",
            tie_tolerance=args.translation_tie_tol_mm,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=_stable_seed(
                args.bootstrap_seed,
                *seed_parts,
                "translation",
            ),
        ),
        "rotation_error_deg": _paired_metric(
            reference,
            candidate,
            field="rotation_error_deg",
            tie_tolerance=args.rotation_tie_tol_deg,
            bootstrap_samples=args.bootstrap_samples,
            bootstrap_seed=_stable_seed(
                args.bootstrap_seed,
                *seed_parts,
                "rotation",
            ),
        ),
        "pareto": _paired_pareto(
            reference,
            candidate,
            translation_tolerance=args.translation_tie_tol_mm,
            rotation_tolerance=args.rotation_tie_tol_deg,
        ),
        "outlier_transitions": _paired_outlier_transitions(
            reference,
            candidate,
        ),
    }


def _validate_condition_pairing(
    condition_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Audit the exact within-trial pairing before computing statistics."""
    rows_by_trial: dict[int, list[dict[str, Any]]] = {}
    for row in condition_rows:
        rows_by_trial.setdefault(int(row["trial_index"]), []).append(row)

    shared_fields = (
        "pair_trial_index",
        "relative_path",
        "logical_dataset_sha256",
        "noise_seed_key",
        "initialization_seed_key",
        "initial_estimate_sha256",
    )
    expected_arms = set(ARMS)
    for trial_index, trial_rows in rows_by_trial.items():
        actual_arms = [str(row["arm"]) for row in trial_rows]
        if len(actual_arms) != len(ARMS) or set(actual_arms) != expected_arms:
            raise ValueError(
                f"trial {trial_index} does not contain exactly one row for "
                f"every arm: {actual_arms}"
            )
        for field in shared_fields:
            values = {str(row[field]) for row in trial_rows}
            if len(values) != 1:
                raise ValueError(
                    f"trial {trial_index} has mismatched {field}: {values}"
                )
        dependent_rows = [
            row for row in trial_rows if row["arm"] != "initial_joint"
        ]
        alternating_hashes = {
            str(row["alternating_estimate_sha256"])
            for row in dependent_rows
        }
        if len(alternating_hashes) != 1:
            raise ValueError(
                f"trial {trial_index} has mismatched alternating estimates"
            )

    return {
        "validated": True,
        "trial_count": len(rows_by_trial),
        "arms_per_trial": len(ARMS),
        "shared_fields": list(shared_fields),
        "alternating_hash_checked_for_dependent_arms": True,
    }


def _condition_summary(
    condition_rows: Sequence[dict[str, Any]],
    *,
    condition_key: tuple[str, str],
    args: argparse.Namespace,
) -> dict[str, Any]:
    pairing_audit = _validate_condition_pairing(condition_rows)
    rows_by_arm: dict[str, list[dict[str, Any]]] = {}
    for arm in ARMS:
        arm_rows = [
            row for row in condition_rows if row["arm"] == arm
        ]
        arm_rows.sort(key=lambda row: int(row["trial_index"]))
        rows_by_arm[arm] = arm_rows
    comparisons = [
        _paired_comparison(
            rows_by_arm,
            reference_arm=reference,
            candidate_arm=candidate,
            condition_key=condition_key,
            args=args,
        )
        for reference, candidate in itertools.combinations(ARMS, 2)
    ]
    example = condition_rows[0]
    return {
        "collection_label": condition_key[0],
        "collection_path": example["collection_path"],
        "collection_manifest_sha256": (
            example["collection_manifest_sha256"]
        ),
        "initialization_label": condition_key[1],
        "init_translation_range_mm": example[
            "init_translation_range_mm"
        ],
        "init_angle_range_deg": example["init_angle_range_deg"],
        "planned_trials": len(rows_by_arm[ARMS[0]]),
        "pairing_audit": pairing_audit,
        "methods": {
            arm: _method_summary(rows_by_arm[arm]) for arm in ARMS
        },
        "paired_comparisons": comparisons,
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(child) for child in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return value.as_posix()
    return value


def _write_long_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LONG_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in LONG_FIELDS})
    temporary.replace(path)


def _write_json(path: Path, document: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            _json_safe(document),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_collection_specs(
    raw_specs: Sequence[str],
    *,
    max_trials: int | None,
    only_trial: int | None,
) -> list[CollectionSpec]:
    output: list[CollectionSpec] = []
    seen_labels: set[str] = set()
    for raw_spec in raw_specs:
        label, path = _parse_labeled_path(raw_spec)
        path = path.resolve()
        if label in seen_labels:
            raise ValueError(f"duplicate collection label: {label!r}")
        seen_labels.add(label)
        manifest_path = path / "collection.json"
        collection = calibration_cli._read_collection(path)
        acquisition_mode = str(collection.get("acquisition_mode", ""))
        if not acquisition_mode:
            raise ValueError(
                f"{manifest_path} has no acquisition_mode"
            )
        entries = list(collection.get("trials", []))
        trial_indices = [int(entry["trial_index"]) for entry in entries]
        duplicate_indices = sorted(
            trial_index
            for trial_index, count in Counter(trial_indices).items()
            if count > 1
        )
        if duplicate_indices:
            raise ValueError(
                f"collection {label!r} contains duplicate trial_index "
                f"values: {duplicate_indices}"
            )
        if only_trial is not None:
            entries = [
                entry
                for entry in entries
                if int(entry["trial_index"]) == only_trial
            ]
            if not entries:
                raise ValueError(
                    f"trial {only_trial} not found in collection {label}"
                )
        if max_trials is not None:
            entries = entries[:max_trials]
        if not entries:
            raise ValueError(f"collection {label!r} has no selected trials")
        output.append(
            CollectionSpec(
                label=label,
                path=path,
                manifest_sha256=_sha256_file(manifest_path),
                acquisition_mode=acquisition_mode,
                entries=tuple(entries),
            )
        )
    return output


def _validate_args(args: argparse.Namespace) -> None:
    if args.seed < 0:
        raise ValueError("--seed must be non-negative")
    if args.init_mode != "carlson":
        raise ValueError(
            "this LABEL:T:R ablation requires --init-mode carlson; "
            "relative initialization ignores the requested error ranges"
        )
    finite_nonnegative = (
        ("--noise-std-mm", args.noise_std_mm),
        ("--translation-success-mm", args.translation_success_mm),
        ("--rotation-success-deg", args.rotation_success_deg),
        ("--outlier-translation-mm", args.outlier_translation_mm),
        ("--outlier-rotation-deg", args.outlier_rotation_deg),
        ("--translation-tie-tol-mm", args.translation_tie_tol_mm),
        ("--rotation-tie-tol-deg", args.rotation_tie_tol_deg),
    )
    for name, value in finite_nonnegative:
        if not np.isfinite(value) or value < 0.0:
            raise ValueError(f"{name} must be finite and non-negative")
    finite_positive = (
        ("--nonlinear-f-scale-mm", args.nonlinear_f_scale_mm),
        ("--tol", args.tol),
        ("--nonlinear-ftol", args.nonlinear_ftol),
        ("--nonlinear-xtol", args.nonlinear_xtol),
        ("--nonlinear-gtol", args.nonlinear_gtol),
    )
    for name, value in finite_positive:
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")


def _task_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "seed": args.seed,
        "noise_axis": args.noise_axis,
        "noise_std_mm": args.noise_std_mm,
        "init_mode": args.init_mode,
        "rel_offset": args.rel_offset,
        "init_translation_perturbation": (
            args.init_translation_perturbation
        ),
        "init_rotation_perturbation": args.init_rotation_perturbation,
        "max_iter": args.max_iter,
        "tol": args.tol,
        "max_scans_per_trial": args.max_scans_per_trial,
        "nonlinear_loss": args.nonlinear_loss,
        "nonlinear_f_scale_mm": args.nonlinear_f_scale_mm,
        "nonlinear_max_nfev": args.nonlinear_max_nfev,
        "nonlinear_ftol": args.nonlinear_ftol,
        "nonlinear_xtol": args.nonlinear_xtol,
        "nonlinear_gtol": args.nonlinear_gtol,
        "translation_success_mm": args.translation_success_mm,
        "rotation_success_deg": args.rotation_success_deg,
        "outlier_translation_mm": args.outlier_translation_mm,
        "outlier_rotation_deg": args.outlier_rotation_deg,
        "debug_traceback": args.debug_traceback,
    }


def _build_tasks(
    collections: Sequence[CollectionSpec],
    initializations: Sequence[InitializationSpec],
    config: dict[str, Any],
) -> list[TrialTask]:
    tasks: list[TrialTask] = []
    for collection in collections:
        for initialization in initializations:
            for entry in collection.entries:
                tasks.append(
                    TrialTask(
                        collection_label=collection.label,
                        collection_path=collection.path.as_posix(),
                        collection_manifest_sha256=(
                            collection.manifest_sha256
                        ),
                        acquisition_mode=collection.acquisition_mode,
                        initialization_label=initialization.label,
                        init_translation_range_mm=(
                            initialization.translation_range_mm
                        ),
                        init_angle_range_deg=(
                            initialization.angle_range_deg
                        ),
                        trial_index=int(entry["trial_index"]),
                        pair_trial_index=int(
                            entry.get(
                                "pair_trial_index",
                                entry["trial_index"],
                            )
                        ),
                        relative_path=str(entry["relative_path"]),
                        logical_dataset_sha256=str(
                            entry.get("logical_dataset_sha256", "")
                        ),
                        config=config,
                    )
                )
    return tasks


def _execute_tasks(
    tasks: Sequence[TrialTask],
    *,
    workers: int,
    verbose: bool,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    total = len(tasks)

    def report(completed: int, task: TrialTask, rows: Sequence[dict[str, Any]]) -> None:
        if not (
            verbose
            or completed == total
            or completed % max(1, total // 20) == 0
        ):
            return
        failures = sum(bool(row["error_type"]) for row in rows)
        print(
            f"[{completed:04d}/{total:04d}] "
            f"collection={task.collection_label} "
            f"init={task.initialization_label} "
            f"trial={task.trial_index:06d} "
            f"arm_failures={failures}/{len(ARMS)}",
            flush=True,
        )

    if workers == 1:
        for completed, task in enumerate(tasks, start=1):
            rows = _run_trial_task(task)
            output.extend(rows)
            report(completed, task, rows)
        return output

    with ProcessPoolExecutor(max_workers=workers) as executor:
        future_to_task = {
            executor.submit(_run_trial_task, task): task for task in tasks
        }
        for completed, future in enumerate(
            as_completed(future_to_task),
            start=1,
        ):
            task = future_to_task[future]
            try:
                rows = future.result()
            except BaseException as exc:
                rows = [
                    _failure(
                        _default_row(task, arm),
                        exc,
                        debug_traceback=task.config["debug_traceback"],
                    )
                    for arm in ARMS
                ]
            output.extend(rows)
            report(completed, task, rows)
    return output


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        _validate_args(args)
        collections = _load_collection_specs(
            args.collection,
            max_trials=args.max_trials,
            only_trial=args.only_trial,
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.nonlinear_loss != "linear":
        warnings.warn(
            "robust-loss arms do not profile the same inner plane objective; "
            "use loss='linear' for the primary normal-treatment ablation",
            RuntimeWarning,
            stacklevel=2,
        )

    initializations = args.initialization or [
        InitializationSpec("default", 100.0, 15.0)
    ]
    labels = [initialization.label for initialization in initializations]
    if len(labels) != len(set(labels)):
        raise SystemExit("initialization labels must be unique")

    output_dir = args.output_dir.expanduser().resolve()
    trials_path = output_dir / "trials_long.csv"
    summary_path = output_dir / "summary.json"
    existing = [path for path in (trials_path, summary_path) if path.exists()]
    if existing and not args.overwrite:
        paths = ", ".join(path.as_posix() for path in existing)
        raise SystemExit(
            f"output artifacts already exist: {paths}; use --overwrite"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    config = _task_config(args)
    tasks = _build_tasks(collections, initializations, config)
    print(
        f"[ablation] tasks={len(tasks)}, arms={len(ARMS)}, "
        f"planned_rows={len(tasks) * len(ARMS)}, workers={args.workers}",
        flush=True,
    )
    started = time.perf_counter()
    rows = _execute_tasks(
        tasks,
        workers=args.workers,
        verbose=args.verbose,
    )
    elapsed = time.perf_counter() - started

    collection_order = {
        collection.label: index
        for index, collection in enumerate(collections)
    }
    initialization_order = {
        initialization.label: index
        for index, initialization in enumerate(initializations)
    }
    arm_order = {arm: index for index, arm in enumerate(ARMS)}
    rows.sort(
        key=lambda row: (
            collection_order[row["collection_label"]],
            initialization_order[row["initialization_label"]],
            int(row["trial_index"]),
            arm_order[row["arm"]],
        )
    )
    _write_long_csv(trials_path, rows)

    conditions: list[dict[str, Any]] = []
    for collection in collections:
        for initialization in initializations:
            condition_key = (collection.label, initialization.label)
            condition_rows = [
                row
                for row in rows
                if row["collection_label"] == collection.label
                and row["initialization_label"] == initialization.label
            ]
            conditions.append(
                _condition_summary(
                    condition_rows,
                    condition_key=condition_key,
                    args=args,
                )
            )

    summary = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "arms": list(ARMS),
        "requested_tasks": len(tasks),
        "planned_rows": len(tasks) * len(ARMS),
        "written_rows": len(rows),
        "failure_rows": sum(bool(row["error_type"]) for row in rows),
        "optimizer_successful_rows": sum(
            not row["error_type"] and bool(row["optimizer_success"])
            for row in rows
        ),
        "elapsed_seconds": elapsed,
        "trials_long_csv": trials_path.name,
        "trials_long_sha256": _sha256_file(trials_path),
        "source_sha256": {
            "runner": _sha256_file(Path(__file__).resolve()),
            "main_calibrate": _sha256_file(
                Path(calibration_cli.__file__).resolve()
            ),
            "calibration_core": _sha256_file(
                Path(
                    calibration_cli.calibrate_planes.__code__.co_filename
                ).resolve()
            ),
            "nonlinear_refinement": _sha256_file(
                Path(refine_handeye_nonlinear.__code__.co_filename).resolve()
            ),
        },
        "config": {
            **{
                key: value
                for key, value in vars(args).items()
                if key
                not in {
                    "collection",
                    "initialization",
                    "output_dir",
                    "overwrite",
                    "verbose",
                    "debug_traceback",
                }
            },
            "collections": [
                {
                    "label": collection.label,
                    "path": collection.path.as_posix(),
                    "collection_manifest_sha256": (
                        collection.manifest_sha256
                    ),
                    "selected_trials": len(collection.entries),
                }
                for collection in collections
            ],
            "initializations": [
                {
                    "label": initialization.label,
                    "translation_range_mm": (
                        initialization.translation_range_mm
                    ),
                    "angle_range_deg": initialization.angle_range_deg,
                }
                for initialization in initializations
            ],
            "output_dir": output_dir.as_posix(),
            "overwrite": args.overwrite,
            "verbose": args.verbose,
            "debug_traceback": args.debug_traceback,
        },
        "statistics": {
            "paired_delta_definition": (
                "reference_error_minus_candidate_error"
            ),
            "positive_delta_means": "candidate_is_better",
            "bootstrap_samples": args.bootstrap_samples,
            "bootstrap_confidence_level": 0.95,
            "bootstrap_statistic": "paired_median_delta",
            "wilcoxon_alternative": "two-sided",
            "wilcoxon_multiplicity": (
                "unadjusted exploratory all-pairs; pre-specify primary "
                "contrasts before confirmatory use"
            ),
            "accuracy_analysis_population": "complete_case_accuracy",
            "failure_accounting": (
                "planned/excluded counts and outlier transitions are retained "
                "for every paired comparison"
            ),
            "robust_loss_caveat": (
                "joint/refit/fixed_normals share the same profiled objective "
                "only when nonlinear_loss is linear"
            ),
            "translation_tie_tolerance_mm": (
                args.translation_tie_tol_mm
            ),
            "rotation_tie_tolerance_deg": args.rotation_tie_tol_deg,
        },
        "conditions": conditions,
    }
    _write_json(summary_path, summary)

    print(
        f"[ablation] complete in {elapsed:.3f} s; "
        f"failures={summary['failure_rows']}/{len(rows)}",
        flush=True,
    )
    print(f"saved: {trials_path}", flush=True)
    print(f"saved: {summary_path}", flush=True)
    return 0 if summary["optimizer_successful_rows"] > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
