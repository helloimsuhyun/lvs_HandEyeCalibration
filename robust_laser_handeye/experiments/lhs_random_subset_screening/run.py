#!/usr/bin/env python3
"""Screen random subsets of a plane-relative LHS pose bank by calibration error."""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
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


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from laser_handeye.calibration import calibrate_planes  # noqa: E402
from laser_handeye.data import LaserScan  # noqa: E402
from laser_handeye.nonlinear_refinement import (  # noqa: E402
    refine_handeye_planes_nonlinear,
)
from laser_handeye.pose_design import (  # noqa: E402
    PLANE_RELATIVE_POSE_CONVENTION,
    PlaneFrame,
    PlaneRelativePoseBounds,
    audit_latin_hypercube,
    latin_hypercube_strata,
    plane_relative_pose_from_row,
    sensor_pose_from_plane_relative,
)
from laser_handeye.pose_generation import sensor_pose_to_robot_pose  # noqa: E402
from laser_handeye.se3 import euler_xyz_deg, make_T, rot_error_deg  # noqa: E402
from laser_handeye.simulation import simulate_profile_on_plane  # noqa: E402


SCHEMA = "laser_handeye.phase1_random_subset_screening"
SCHEMA_VERSION = 2
PARAMETER_NAMES = (
    "u_mm",
    "v_mm",
    "d_mm",
    "tilt_deg",
    "azimuth_deg",
    "normal_azimuth_sensor_deg",
)

# Implementation defaults deliberately stay out of the user-facing JSON.
PROFILE_POINTS = 100
PROFILE_HALF_WIDTH_MM = 25.0
PROFILE_Z_RANGE_MM = (20.0, 250.0)
MAX_WITHIN_STRATUM_RETRIES = 100


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="optional output override; relative paths are resolved from robust_laser_handeye",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="use a tiny in-memory override without modifying the JSON",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate and print the resolved experiment without running it",
    )
    return parser


def _load_config(path: Path) -> dict[str, Any]:
    with path.expanduser().open(encoding="utf-8") as stream:
        config = json.load(stream)
    if not isinstance(config, dict):
        raise ValueError("config root must be a JSON object")
    return config


def _finite_vector(value: Any, name: str, size: int = 3) -> np.ndarray:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (size,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain {size} finite values")
    return vector


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _nonnegative_float(value: Any, name: str) -> float:
    output = float(value)
    if not np.isfinite(output) or output < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return output


def _increasing_pair(value: Any, name: str) -> tuple[float, float]:
    pair = np.asarray(value, dtype=float)
    if (
        pair.shape != (2,)
        or not np.all(np.isfinite(pair))
        or float(pair[0]) >= float(pair[1])
    ):
        raise ValueError(f"{name} must be an increasing finite pair")
    return float(pair[0]), float(pair[1])


def _validate_config(config: dict[str, Any]) -> dict[str, Any]:
    required = (
        "seed",
        "output_dir",
        "lhs",
        "screening",
        "plane",
        "ground_truth_handeye",
        "initial_error",
        "noise",
        "solver",
        "classification",
    )
    missing = [name for name in required if name not in config]
    if missing:
        raise ValueError(f"config is missing fields: {missing}")

    seed = int(config["seed"])
    if isinstance(config["seed"], bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    lhs = config["lhs"]
    candidate_count = _positive_int(lhs["candidate_count"], "lhs.candidate_count")
    ranges = lhs["ranges"]
    parsed_ranges = {
        name: _increasing_pair(ranges[name], f"lhs.ranges.{name}")
        for name in PARAMETER_NAMES
    }

    screening = config["screening"]
    scan_counts = sorted(
        set(
            _positive_int(value, "screening.scan_counts[]")
            for value in screening["scan_counts"]
        )
    )
    if not scan_counts:
        raise ValueError("screening.scan_counts must not be empty")
    if scan_counts[-1] > candidate_count:
        raise ValueError("scan count cannot exceed lhs.candidate_count")
    subsets_per_n = _positive_int(
        screening["subsets_per_n"], "screening.subsets_per_n"
    )
    top_fraction = float(screening["top_fraction"])
    if not np.isfinite(top_fraction) or not 0.0 < top_fraction <= 1.0:
        raise ValueError("screening.top_fraction must lie in (0, 1]")

    plane = config["plane"]
    center = _finite_vector(plane["center_base_mm"], "plane.center_base_mm")
    frame = PlaneFrame(
        _finite_vector(plane["u_base"], "plane.u_base"),
        _finite_vector(plane["v_base"], "plane.v_base"),
        _finite_vector(plane["normal_base"], "plane.normal_base"),
        float(_finite_vector(plane["normal_base"], "plane.normal_base") @ center),
    )

    truth = config["ground_truth_handeye"]
    truth_angles = _finite_vector(
        truth["rotation_euler_xyz_deg"],
        "ground_truth_handeye.rotation_euler_xyz_deg",
    )
    truth_translation = _finite_vector(
        truth["translation_mm"], "ground_truth_handeye.translation_mm"
    )
    T_true = make_T(euler_xyz_deg(*truth_angles), truth_translation)

    initial = config["initial_error"]
    initial_translation = _finite_vector(
        initial["translation_mm"], "initial_error.translation_mm"
    )
    initial_axis = _finite_vector(
        initial["rotation_axis"], "initial_error.rotation_axis"
    )
    axis_norm = float(np.linalg.norm(initial_axis))
    if axis_norm <= 1e-12:
        raise ValueError("initial_error.rotation_axis must be non-zero")
    initial_axis = initial_axis / axis_norm
    initial_angle = _nonnegative_float(
        initial["rotation_deg"], "initial_error.rotation_deg"
    )
    T_error = make_T(
        Rotation.from_rotvec(initial_axis * np.deg2rad(initial_angle)).as_matrix(),
        initial_translation,
    )
    T_init = T_true @ T_error

    noise = config["noise"]
    if noise["axis"] not in ("z", "xz"):
        raise ValueError("noise.axis must be 'z' or 'xz'")
    noise_std = _nonnegative_float(noise["std_mm"], "noise.std_mm")
    noise_repeats = _positive_int(noise["repeats"], "noise.repeats")

    solver = config["solver"]
    iterative_solver = solver["iterative"]
    nonlinear_solver = solver["nonlinear"]
    if iterative_solver["plane_offset_mode"] != "joint":
        raise ValueError("solver.iterative.plane_offset_mode must be 'joint'")
    if nonlinear_solver["plane_mode"] != "joint":
        raise ValueError("solver.nonlinear.plane_mode must be 'joint'")
    if nonlinear_solver["loss"] != "linear":
        raise ValueError("solver.nonlinear.loss must be 'linear'")
    iterative_max_iter = _positive_int(
        iterative_solver["max_iter"], "solver.iterative.max_iter"
    )
    iterative_tol = _nonnegative_float(
        iterative_solver["tol"], "solver.iterative.tol"
    )
    nonlinear_max_nfev = _positive_int(
        nonlinear_solver["max_nfev"], "solver.nonlinear.max_nfev"
    )
    nonlinear_tol = _nonnegative_float(
        nonlinear_solver["tol"], "solver.nonlinear.tol"
    )

    classification = config["classification"]
    classification_translation = _nonnegative_float(
        classification["translation_mm"], "classification.translation_mm"
    )
    classification_rotation = _nonnegative_float(
        classification["rotation_deg"], "classification.rotation_deg"
    )
    if classification_translation == 0.0 or classification_rotation == 0.0:
        raise ValueError("classification thresholds must be positive")
    minimum_pass_rate = float(
        classification["minimum_pass_rate_for_promotion"]
    )
    if not np.isfinite(minimum_pass_rate) or not 0.0 <= minimum_pass_rate <= 1.0:
        raise ValueError(
            "classification.minimum_pass_rate_for_promotion must lie in [0, 1]"
        )

    return {
        "seed": seed,
        "output_dir": str(config["output_dir"]),
        "candidate_count": candidate_count,
        "ranges": parsed_ranges,
        "scan_counts": scan_counts,
        "subsets_per_n": subsets_per_n,
        "top_fraction": top_fraction,
        "plane_center": center,
        "plane_frame": frame,
        "T_true": T_true,
        "T_error": T_error,
        "T_init": T_init,
        "noise_axis": str(noise["axis"]),
        "noise_std_mm": noise_std,
        "noise_repeats": noise_repeats,
        "iterative_plane_offset_mode": str(iterative_solver["plane_offset_mode"]),
        "iterative_max_iter": iterative_max_iter,
        "iterative_tol": iterative_tol,
        "nonlinear_plane_mode": str(nonlinear_solver["plane_mode"]),
        "nonlinear_loss": str(nonlinear_solver["loss"]),
        "nonlinear_max_nfev": nonlinear_max_nfev,
        "nonlinear_tol": nonlinear_tol,
        "classification_translation_mm": classification_translation,
        "classification_rotation_deg": classification_rotation,
        "minimum_pass_rate_for_promotion": minimum_pass_rate,
    }


def _apply_smoke_override(config: dict[str, Any]) -> None:
    config["lhs"]["candidate_count"] = 40
    config["screening"]["scan_counts"] = [5, 6]
    config["screening"]["subsets_per_n"] = 3
    config["noise"]["repeats"] = 1
    config["solver"]["iterative"]["max_iter"] = min(
        int(config["solver"]["iterative"]["max_iter"]), 100
    )
    config["solver"]["nonlinear"]["max_nfev"] = min(
        int(config["solver"]["nonlinear"]["max_nfev"]), 100
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(_jsonable(payload), indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _write_csv(
    path: Path,
    rows: Sequence[dict[str, Any]],
    *,
    fieldnames: Sequence[str] | None = None,
) -> None:
    columns = list(fieldnames or (list(rows[0]) if rows else ()))
    with path.open("w", newline="", encoding="utf-8") as stream:
        if columns:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _format_duration(seconds: float) -> str:
    if not np.isfinite(seconds) or seconds < 0.0:
        return "?"
    total = int(round(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:d}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes:d}m {secs:02d}s"
    return f"{secs:d}s"


def _candidate_bounds(ranges: dict[str, tuple[float, float]]) -> PlaneRelativePoseBounds:
    return PlaneRelativePoseBounds(
        target_u_mm=ranges["u_mm"],
        target_v_mm=ranges["v_mm"],
        distance_mm=ranges["d_mm"],
        tilt_deg=ranges["tilt_deg"],
        azimuth_deg=ranges["azimuth_deg"],
        normal_azimuth_sensor_deg=ranges["normal_azimuth_sensor_deg"],
    )


def _valid_profile(points_s: np.ndarray) -> bool:
    points = np.asarray(points_s, dtype=float)
    return bool(
        points.shape == (PROFILE_POINTS, 3)
        and np.all(np.isfinite(points))
        and float(np.min(points[:, 2])) >= PROFILE_Z_RANGE_MM[0]
        and float(np.max(points[:, 2])) <= PROFILE_Z_RANGE_MM[1]
    )


def _build_candidate_bank(
    resolved: dict[str, Any],
) -> dict[str, np.ndarray]:
    count = resolved["candidate_count"]
    rng = np.random.default_rng(
        np.random.SeedSequence([resolved["seed"], 0x4C4853])
    )
    strata = latin_hypercube_strata(rng, count, len(PARAMETER_NAMES))
    normalized = np.empty((count, len(PARAMETER_NAMES)), dtype=float)
    parameters = np.empty_like(normalized)
    T_base_s = np.empty((count, 4, 4), dtype=float)
    T_base_ef = np.empty((count, 4, 4), dtype=float)
    ideal_points_s = np.empty((count, PROFILE_POINTS, 3), dtype=float)
    retries = np.empty(count, dtype=np.int64)

    frame: PlaneFrame = resolved["plane_frame"]
    center = resolved["plane_center"]
    T_true = resolved["T_true"]
    bounds = _candidate_bounds(resolved["ranges"])
    x_values = np.linspace(
        -PROFILE_HALF_WIDTH_MM, PROFILE_HALF_WIDTH_MM, PROFILE_POINTS
    )

    for candidate_id in range(count):
        lower = strata[candidate_id].astype(float) / float(count)
        accepted = False
        for retry in range(MAX_WITHIN_STRATUM_RETRIES):
            row = lower + rng.random(len(PARAMETER_NAMES)) / float(count)
            pose = plane_relative_pose_from_row(
                row, sample_id=candidate_id, bounds=bounds
            )
            try:
                sensor_pose = sensor_pose_from_plane_relative(frame, center, pose)
                flange_pose = sensor_pose_to_robot_pose(sensor_pose, T_true)
                scan = simulate_profile_on_plane(
                    T_base_ef=flange_pose,
                    T_ef_s_true=T_true,
                    plane_n=frame.n,
                    plane_l=frame.offset_mm,
                    x_values=x_values,
                    noise_std=0.0,
                    plane_id=0,
                    scan_id=candidate_id,
                )
            except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                continue
            if not _valid_profile(scan.points_s):
                continue
            normalized[candidate_id] = row
            parameters[candidate_id] = (
                pose.target_u_mm,
                pose.target_v_mm,
                pose.distance_mm,
                pose.tilt_deg,
                pose.azimuth_deg,
                pose.normal_azimuth_sensor_deg,
            )
            T_base_s[candidate_id] = sensor_pose
            T_base_ef[candidate_id] = flange_pose
            ideal_points_s[candidate_id] = scan.points_s
            retries[candidate_id] = retry
            accepted = True
            break
        if not accepted:
            raise RuntimeError(
                f"candidate {candidate_id} remained infeasible after "
                f"{MAX_WITHIN_STRATUM_RETRIES} within-stratum retries"
            )

    audit_latin_hypercube(normalized)
    return {
        "candidate_ids": np.arange(count, dtype=np.int64),
        "strata": strata,
        "normalized_parameters": normalized,
        "pose_parameters": parameters,
        "T_base_s": T_base_s,
        "T_base_ef": T_base_ef,
        "ideal_points_s": ideal_points_s,
        "within_stratum_retries": retries,
    }


def _candidate_rows(bank: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate_id in bank["candidate_ids"]:
        index = int(candidate_id)
        row: dict[str, Any] = {
            "candidate_id": index,
            "within_stratum_retries": int(bank["within_stratum_retries"][index]),
        }
        for dimension, name in enumerate(PARAMETER_NAMES):
            row[f"lhs_{name}"] = float(bank["normalized_parameters"][index, dimension])
            row[f"stratum_{name}"] = int(bank["strata"][index, dimension])
            row[name] = float(bank["pose_parameters"][index, dimension])
        rows.append(row)
    return rows


def _make_noisy_scan_bank(
    resolved: dict[str, Any], bank: dict[str, np.ndarray]
) -> list[list[LaserScan]]:
    output: list[list[LaserScan]] = []
    for repeat in range(resolved["noise_repeats"]):
        scans: list[LaserScan] = []
        for candidate_id in bank["candidate_ids"]:
            index = int(candidate_id)
            points = bank["ideal_points_s"][index].copy()
            if resolved["noise_std_mm"] > 0.0:
                rng = np.random.default_rng(
                    np.random.SeedSequence(
                        [resolved["seed"], 0x4E4F4953, repeat, index]
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
                    T_base_ef=bank["T_base_ef"][index],
                    points_s=points,
                    plane_id=0,
                    scan_id=index,
                    meta={"candidate_id": index, "noise_repeat": repeat},
                )
            )
        output.append(scans)
    return output


def _sample_subsets(
    *, candidate_count: int, scan_count: int, subset_count: int, seed: int
) -> np.ndarray:
    available = math.comb(candidate_count, scan_count)
    if subset_count > available:
        raise ValueError(
            f"requested {subset_count} unique N={scan_count} subsets, but only "
            f"{available} combinations exist"
        )
    rng = np.random.default_rng(
        np.random.SeedSequence([seed, 0x53554253, scan_count])
    )
    selected: set[tuple[int, ...]] = set()
    while len(selected) < subset_count:
        candidate_ids = tuple(
            sorted(
                map(
                    int,
                    rng.choice(candidate_count, size=scan_count, replace=False),
                )
            )
        )
        selected.add(candidate_ids)
    return np.asarray(sorted(selected), dtype=np.int64)


def _transform_errors(T_est: np.ndarray, T_true: np.ndarray) -> tuple[float, float]:
    return (
        float(np.linalg.norm(T_est[:3, 3] - T_true[:3, 3])),
        float(rot_error_deg(T_est[:3, :3], T_true[:3, :3])),
    )


def _run_one_calibration(
    *,
    resolved: dict[str, Any],
    scans: list[LaserScan],
    scan_count: int,
    subset_id: int,
    repeat: int,
    candidate_ids: np.ndarray,
) -> dict[str, Any]:
    started = time.perf_counter()
    row: dict[str, Any] = {
        "source": "random",
        "N": scan_count,
        "subset_id": subset_id,
        "noise_repeat": repeat,
        "candidate_ids": json.dumps(candidate_ids.tolist(), separators=(",", ":")),
        "iterative_converged": False,
        "iterative_finite": False,
        "iterative_iterations": 0,
        "iterative_translation_error_mm": float("nan"),
        "iterative_rotation_error_deg": float("nan"),
        "iterative_rank_last": -1,
        "iterative_condition_last": float("nan"),
        "nonlinear_success": False,
        "nonlinear_attempted": False,
        "nonlinear_status": 0,
        "nonlinear_nfev": 0,
        "nonlinear_final_rms_mm": float("nan"),
        "nonlinear_rank": -1,
        "nonlinear_condition": float("nan"),
        "translation_error_mm": float("inf"),
        "rotation_error_deg": float("inf"),
        "normalized_error": float("inf"),
        "passes_success_threshold": False,
        "catastrophic_failure": True,
        "runtime_s": float("nan"),
        "failure_stage": "",
        "error_type": "",
        "error_message": "",
    }
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            iterative = calibrate_planes(
                {0: scans},
                resolved["T_init"],
                max_iter=resolved["iterative_max_iter"],
                tol=resolved["iterative_tol"],
                plane_offset_mode=resolved["iterative_plane_offset_mode"],
            )
        iterative_t, iterative_r = _transform_errors(
            iterative.T_ef_s, resolved["T_true"]
        )
        iterative_finite = bool(np.all(np.isfinite(iterative.T_ef_s)))
        row.update(
            iterative_converged=bool(iterative.converged),
            iterative_finite=iterative_finite,
            iterative_iterations=int(iterative.iterations),
            iterative_translation_error_mm=iterative_t,
            iterative_rotation_error_deg=iterative_r,
            iterative_rank_last=(
                int(iterative.rank_history[-1]) if iterative.rank_history else -1
            ),
            iterative_condition_last=(
                float(iterative.cond_history[-1])
                if iterative.cond_history
                else float("nan")
            ),
        )
        row["nonlinear_attempted"] = True
        nonlinear = refine_handeye_planes_nonlinear(
            {0: scans},
            iterative.T_ef_s,
            loss=resolved["nonlinear_loss"],
            max_nfev=resolved["nonlinear_max_nfev"],
            ftol=resolved["nonlinear_tol"],
            xtol=resolved["nonlinear_tol"],
            gtol=resolved["nonlinear_tol"],
        )
        translation_error, rotation_error = _transform_errors(
            nonlinear.T_ef_s, resolved["T_true"]
        )
        normalized_error = max(
            translation_error / resolved["classification_translation_mm"],
            rotation_error / resolved["classification_rotation_deg"],
        )
        passes_threshold = bool(
            nonlinear.success
            and np.isfinite(translation_error)
            and np.isfinite(rotation_error)
            and translation_error <= resolved["classification_translation_mm"]
            and rotation_error <= resolved["classification_rotation_deg"]
        )
        finite_result = bool(
            np.all(np.isfinite(nonlinear.T_ef_s))
            and np.isfinite(translation_error)
            and np.isfinite(rotation_error)
        )
        row.update(
            nonlinear_success=bool(nonlinear.success),
            nonlinear_status=int(nonlinear.status),
            nonlinear_nfev=int(nonlinear.nfev),
            nonlinear_final_rms_mm=float(nonlinear.final_rms_mm),
            nonlinear_rank=int(nonlinear.jacobian_rank),
            nonlinear_condition=float(nonlinear.scaled_jacobian_condition),
            translation_error_mm=translation_error,
            rotation_error_deg=rotation_error,
            normalized_error=normalized_error,
            passes_success_threshold=passes_threshold,
            catastrophic_failure=not finite_result,
        )
    except Exception as exc:  # retain failed subsets in the result distribution
        row["failure_stage"] = (
            "nonlinear" if row["nonlinear_attempted"] else "iterative"
        )
        row["error_type"] = type(exc).__name__
        row["error_message"] = str(exc)
    row["runtime_s"] = time.perf_counter() - started
    return row


def _finite_quantile(values: Iterable[float], quantile: float) -> float:
    data = np.asarray(list(values), dtype=float)
    data = data[np.isfinite(data)]
    if len(data) == 0:
        return float("inf")
    return float(np.quantile(data, quantile))


def _summarize_subset(
    *, scan_count: int, subset_id: int, candidate_ids: np.ndarray, rows: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    threshold_pass_rate = float(
        np.mean([bool(row["passes_success_threshold"]) for row in rows])
    )
    nonlinear_success_rate = float(
        np.mean([bool(row["nonlinear_success"]) for row in rows])
    )
    finite_result_rate = float(
        np.mean(
            [
                np.isfinite(float(row["translation_error_mm"]))
                and np.isfinite(float(row["rotation_error_deg"]))
                for row in rows
            ]
        )
    )
    return {
        "source": "random",
        "N": scan_count,
        "subset_id": subset_id,
        "candidate_ids": json.dumps(candidate_ids.tolist(), separators=(",", ":")),
        "nonlinear_success_rate": nonlinear_success_rate,
        "finite_result_rate": finite_result_rate,
        "threshold_pass_rate": threshold_pass_rate,
        "normalized_error_p90": _finite_quantile(
            (row["normalized_error"] for row in rows), 0.90
        ),
        "normalized_error_median": _finite_quantile(
            (row["normalized_error"] for row in rows), 0.50
        ),
        "translation_error_p90_mm": _finite_quantile(
            (row["translation_error_mm"] for row in rows), 0.90
        ),
        "translation_error_median_mm": _finite_quantile(
            (row["translation_error_mm"] for row in rows), 0.50
        ),
        "rotation_error_p90_deg": _finite_quantile(
            (row["rotation_error_deg"] for row in rows), 0.90
        ),
        "rotation_error_median_deg": _finite_quantile(
            (row["rotation_error_deg"] for row in rows), 0.50
        ),
        "final_residual_rms_median_mm": _finite_quantile(
            (row["nonlinear_final_rms_mm"] for row in rows), 0.50
        ),
        "runtime_sum_s": float(sum(float(row["runtime_s"]) for row in rows)),
        "eligible_for_promotion": False,
        "rank_in_N": -1,
        "promoted": False,
    }


def _ranking_key(row: dict[str, Any]) -> tuple[float, ...]:
    return (
        -float(row["threshold_pass_rate"]),
        float(row["normalized_error_p90"]),
        float(row["normalized_error_median"]),
        float(row["translation_error_p90_mm"]),
        float(row["rotation_error_p90_deg"]),
        float(row["final_residual_rms_median_mm"]),
        int(row["subset_id"]),
    )


def _promote(
    summaries: list[dict[str, Any]], resolved: dict[str, Any]
) -> list[dict[str, Any]]:
    ordered = sorted(summaries, key=_ranking_key)
    for rank, row in enumerate(ordered, start=1):
        row["rank_in_N"] = rank
        row["eligible_for_promotion"] = bool(
            float(row["threshold_pass_rate"])
            >= resolved["minimum_pass_rate_for_promotion"]
        )
    eligible = [row for row in ordered if row["eligible_for_promotion"]]
    promoted_count = int(math.ceil(len(eligible) * resolved["top_fraction"]))
    promoted = eligible[:promoted_count]
    for row in promoted:
        row["promoted"] = True
    return promoted


def _write_promoted(
    *,
    directory: Path,
    scan_count: int,
    promoted: Sequence[dict[str, Any]],
    subsets: np.ndarray,
    bank: dict[str, np.ndarray],
) -> tuple[Path, Path]:
    candidate_sets = np.asarray(
        [subsets[int(row["subset_id"])] for row in promoted], dtype=np.int64
    ).reshape(len(promoted), scan_count)
    if len(promoted):
        normalized = bank["normalized_parameters"][candidate_sets]
        parameters = bank["pose_parameters"][candidate_sets]
        T_base_s = bank["T_base_s"][candidate_sets]
        T_base_ef = bank["T_base_ef"][candidate_sets]
    else:
        normalized = np.empty((0, scan_count, len(PARAMETER_NAMES)), dtype=float)
        parameters = np.empty_like(normalized)
        T_base_s = np.empty((0, scan_count, 4, 4), dtype=float)
        T_base_ef = np.empty_like(T_base_s)

    prefix = directory / f"promoted_N{scan_count:03d}"
    npz_path = prefix.with_suffix(".npz")
    np.savez_compressed(
        npz_path,
        parameter_names=np.asarray(PARAMETER_NAMES, dtype=np.str_),
        pose_convention=np.asarray(PLANE_RELATIVE_POSE_CONVENTION),
        subset_ids=np.asarray([row["subset_id"] for row in promoted], dtype=np.int64),
        rank_in_N=np.asarray([row["rank_in_N"] for row in promoted], dtype=np.int64),
        candidate_ids=candidate_sets,
        normalized_pose_parameters=normalized,
        pose_parameters=parameters,
        T_base_s=T_base_s,
        T_base_ef=T_base_ef,
        nonlinear_success_rate=np.asarray(
            [row["nonlinear_success_rate"] for row in promoted], dtype=float
        ),
        finite_result_rate=np.asarray(
            [row["finite_result_rate"] for row in promoted], dtype=float
        ),
        threshold_pass_rate=np.asarray(
            [row["threshold_pass_rate"] for row in promoted], dtype=float
        ),
        normalized_error_p90=np.asarray(
            [row["normalized_error_p90"] for row in promoted], dtype=float
        ),
        normalized_error_median=np.asarray(
            [row["normalized_error_median"] for row in promoted], dtype=float
        ),
    )

    csv_rows: list[dict[str, Any]] = []
    for subset_position, summary in enumerate(promoted):
        for pose_position, candidate_id in enumerate(candidate_sets[subset_position]):
            index = int(candidate_id)
            row = {
                "source": "random",
                "N": scan_count,
                "subset_id": int(summary["subset_id"]),
                "rank_in_N": int(summary["rank_in_N"]),
                "pose_position": pose_position,
                "candidate_id": index,
                "nonlinear_success_rate": float(summary["nonlinear_success_rate"]),
                "finite_result_rate": float(summary["finite_result_rate"]),
                "threshold_pass_rate": float(summary["threshold_pass_rate"]),
                "normalized_error_p90": float(summary["normalized_error_p90"]),
                "normalized_error_median": float(summary["normalized_error_median"]),
            }
            for dimension, name in enumerate(PARAMETER_NAMES):
                row[f"lhs_{name}"] = float(
                    bank["normalized_parameters"][index, dimension]
                )
                row[name] = float(bank["pose_parameters"][index, dimension])
            csv_rows.append(row)
    csv_path = prefix.with_suffix(".csv")
    promoted_columns = (
        "source",
        "N",
        "subset_id",
        "rank_in_N",
        "pose_position",
        "candidate_id",
        "nonlinear_success_rate",
        "finite_result_rate",
        "threshold_pass_rate",
        "normalized_error_p90",
        "normalized_error_median",
        *(f"lhs_{name}" for name in PARAMETER_NAMES),
        *PARAMETER_NAMES,
    )
    _write_csv(csv_path, csv_rows, fieldnames=promoted_columns)
    return csv_path, npz_path


def _finite_values(rows: Sequence[dict[str, Any]], field: str) -> np.ndarray:
    values = np.asarray([row[field] for row in rows], dtype=float)
    return values[np.isfinite(values)]


def _write_n_plots(
    *,
    directory: Path,
    scan_count: int,
    run_rows: Sequence[dict[str, Any]],
    summaries: Sequence[dict[str, Any]],
    resolved: dict[str, Any],
) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=False)
    translation = _finite_values(run_rows, "translation_error_mm")
    rotation = _finite_values(run_rows, "rotation_error_deg")
    normalized = _finite_values(run_rows, "normalized_error")
    pass_rates = _finite_values(summaries, "threshold_pass_rate")

    figure, axes = plt.subplots(2, 2, figsize=(11, 8))
    panels = (
        (
            axes[0, 0],
            translation,
            resolved["classification_translation_mm"],
            "Translation error [mm]",
        ),
        (
            axes[0, 1],
            rotation,
            resolved["classification_rotation_deg"],
            "Rotation error [deg]",
        ),
        (axes[1, 0], normalized, 1.0, "Normalized error"),
        (
            axes[1, 1],
            pass_rates,
            resolved["minimum_pass_rate_for_promotion"],
            "Subset threshold pass rate",
        ),
    )
    for axis, values, threshold, label in panels:
        if len(values):
            axis.hist(values, bins=min(40, max(5, int(np.sqrt(len(values))))), alpha=0.8)
        else:
            axis.text(0.5, 0.5, "No finite values", ha="center", va="center")
        axis.axvline(threshold, color="tab:red", linestyle="--", label="threshold")
        axis.set(xlabel=label, ylabel="Count")
        axis.grid(alpha=0.2)
        axis.legend()
    passed_runs = sum(bool(row["passes_success_threshold"]) for row in run_rows)
    figure.suptitle(
        f"N={scan_count}: all calibration runs "
        f"(threshold passes {passed_runs}/{len(run_rows)})"
    )
    figure.tight_layout()
    distribution_path = directory / "error_distributions.png"
    figure.savefig(distribution_path, dpi=160)
    plt.close(figure)

    finite_pairs = np.asarray(
        [
            (row["translation_error_mm"], row["rotation_error_deg"])
            for row in run_rows
            if np.isfinite(float(row["translation_error_mm"]))
            and np.isfinite(float(row["rotation_error_deg"]))
        ],
        dtype=float,
    ).reshape(-1, 2)
    figure, axis = plt.subplots(figsize=(7, 6))
    if len(finite_pairs):
        pass_mask = (
            (finite_pairs[:, 0] <= resolved["classification_translation_mm"])
            & (finite_pairs[:, 1] <= resolved["classification_rotation_deg"])
        )
        axis.scatter(
            finite_pairs[~pass_mask, 0],
            finite_pairs[~pass_mask, 1],
            s=12,
            alpha=0.45,
            label="outside threshold",
        )
        axis.scatter(
            finite_pairs[pass_mask, 0],
            finite_pairs[pass_mask, 1],
            s=14,
            alpha=0.65,
            label="inside threshold",
        )
    else:
        axis.text(0.5, 0.5, "No finite calibration results", ha="center", va="center")
    axis.axvline(
        resolved["classification_translation_mm"], color="tab:red", linestyle="--"
    )
    axis.axhline(
        resolved["classification_rotation_deg"], color="tab:red", linestyle="--"
    )
    axis.set(
        xlabel="Translation error [mm]",
        ylabel="Rotation error [deg]",
        title=f"N={scan_count}: calibration error pairs",
    )
    axis.grid(alpha=0.2)
    axis.legend()
    figure.tight_layout()
    scatter_path = directory / "translation_vs_rotation.png"
    figure.savefig(scatter_path, dpi=160)
    plt.close(figure)
    return [distribution_path, scatter_path]


def _resolved_summary(resolved: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    initial_t, initial_r = _transform_errors(resolved["T_init"], resolved["T_true"])
    frame: PlaneFrame = resolved["plane_frame"]
    return {
        "output_dir": output_dir,
        "candidate_count": resolved["candidate_count"],
        "scan_counts": resolved["scan_counts"],
        "subsets_per_n": resolved["subsets_per_n"],
        "noise_repeats": resolved["noise_repeats"],
        "total_calibrations": (
            len(resolved["scan_counts"])
            * resolved["subsets_per_n"]
            * resolved["noise_repeats"]
        ),
        "plane": {
            "center_base_mm": resolved["plane_center"],
            "u_base": frame.u,
            "v_base": frame.v,
            "normal_base": frame.n,
            "offset_mm": frame.offset_mm,
        },
        "T_ef_s_true": resolved["T_true"],
        "T_error_right_local": resolved["T_error"],
        "T_init": resolved["T_init"],
        "actual_initial_translation_error_mm": initial_t,
        "actual_initial_rotation_error_deg": initial_r,
    }


def run(config: dict[str, Any], output_override: Path | None = None) -> Path:
    resolved = _validate_config(config)
    raw_output = output_override or Path(resolved["output_dir"])
    output_dir = raw_output.expanduser()
    if not output_dir.is_absolute():
        output_dir = PROJECT_ROOT / output_dir
    if output_dir.exists() and next(output_dir.iterdir(), None) is not None:
        raise FileExistsError(
            f"output directory is not empty; refusing to overwrite: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    subsets_dir = output_dir / "subsets"
    promoted_dir = output_dir / "promoted"
    per_n_dir = output_dir / "per_n"
    subsets_dir.mkdir()
    promoted_dir.mkdir()
    per_n_dir.mkdir()

    manifest = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "in_progress",
        "config": config,
        "resolved": _resolved_summary(resolved, output_dir),
        "implementation_defaults": {
            "profile_points": PROFILE_POINTS,
            "profile_half_width_mm": PROFILE_HALF_WIDTH_MM,
            "profile_z_range_mm": PROFILE_Z_RANGE_MM,
            "max_within_stratum_retries": MAX_WITHIN_STRATUM_RETRIES,
            "pose_parameter_order": PARAMETER_NAMES,
            "pose_convention": PLANE_RELATIVE_POSE_CONVENTION,
            "normal_azimuth_definition": (
                "normal_azimuth_sensor_deg = atan2((R_BS^T n)_y, "
                "(R_BS^T n)_x)"
            ),
            "initial_error_composition": "T_init = T_true @ T_error",
            "subset_source": "random",
            "threshold_policy": "classification_only; never remove run or subset rows",
        },
        "completed_scan_counts": [],
        "per_n": {},
        "files": {},
    }
    _write_json(output_dir / "manifest.json", manifest)

    print(f"Generating exact {resolved['candidate_count']}-pose LHS bank...", flush=True)
    bank = _build_candidate_bank(resolved)
    candidate_csv = output_dir / "candidate_bank.csv"
    candidate_npz = output_dir / "candidate_bank.npz"
    _write_csv(candidate_csv, _candidate_rows(bank))
    np.savez_compressed(
        candidate_npz,
        parameter_names=np.asarray(PARAMETER_NAMES, dtype=np.str_),
        pose_convention=np.asarray(PLANE_RELATIVE_POSE_CONVENTION),
        **bank,
    )
    manifest["files"][candidate_csv.relative_to(output_dir).as_posix()] = _sha256(candidate_csv)
    manifest["files"][candidate_npz.relative_to(output_dir).as_posix()] = _sha256(candidate_npz)
    _write_json(output_dir / "manifest.json", manifest)

    noisy_scans = _make_noisy_scan_bank(resolved, bank)
    all_runs: list[dict[str, Any]] = []
    all_summaries: list[dict[str, Any]] = []
    experiment_started = time.perf_counter()
    total_calibrations = (
        len(resolved["scan_counts"])
        * resolved["subsets_per_n"]
        * resolved["noise_repeats"]
    )
    completed_calibrations = 0

    for n_index, scan_count in enumerate(resolved["scan_counts"], start=1):
        n_started = time.perf_counter()
        print(
            f"[N={scan_count} {n_index}/{len(resolved['scan_counts'])}] "
            f"sampling {resolved['subsets_per_n']} random subsets...",
            flush=True,
        )
        subsets = _sample_subsets(
            candidate_count=resolved["candidate_count"],
            scan_count=scan_count,
            subset_count=resolved["subsets_per_n"],
            seed=resolved["seed"],
        )
        subset_path = subsets_dir / f"subsets_N{scan_count:03d}.npz"
        np.savez_compressed(
            subset_path,
            source=np.asarray("random"),
            N=np.asarray(scan_count, dtype=np.int64),
            subset_ids=np.arange(len(subsets), dtype=np.int64),
            candidate_ids=subsets,
        )

        summaries: list[dict[str, Any]] = []
        n_runs: list[dict[str, Any]] = []
        for subset_id, candidate_ids in enumerate(subsets):
            subset_rows: list[dict[str, Any]] = []
            for repeat, repeat_scans in enumerate(noisy_scans):
                scans = [repeat_scans[int(index)] for index in candidate_ids]
                row = _run_one_calibration(
                    resolved=resolved,
                    scans=scans,
                    scan_count=scan_count,
                    subset_id=subset_id,
                    repeat=repeat,
                    candidate_ids=candidate_ids,
                )
                subset_rows.append(row)
                all_runs.append(row)
                n_runs.append(row)
                completed_calibrations += 1
            summaries.append(
                _summarize_subset(
                    scan_count=scan_count,
                    subset_id=subset_id,
                    candidate_ids=candidate_ids,
                    rows=subset_rows,
                )
            )
            completed_subsets = subset_id + 1
            progress_interval = max(1, len(subsets) // 20)
            if (
                completed_subsets % progress_interval == 0
                or completed_subsets == len(subsets)
            ):
                elapsed = time.perf_counter() - experiment_started
                elapsed_n = time.perf_counter() - n_started
                total_eta = (
                    elapsed
                    * (total_calibrations - completed_calibrations)
                    / completed_calibrations
                )
                n_eta = (
                    elapsed_n
                    * (len(subsets) - completed_subsets)
                    / completed_subsets
                )
                print(
                    f"[N={scan_count} {n_index}/{len(resolved['scan_counts'])}] "
                    f"subsets {completed_subsets}/{len(subsets)} "
                    f"({100.0 * completed_subsets / len(subsets):5.1f}%), "
                    f"N ETA {_format_duration(n_eta)} | "
                    f"total runs {completed_calibrations}/{total_calibrations} "
                    f"({100.0 * completed_calibrations / total_calibrations:5.1f}%), "
                    f"elapsed {_format_duration(elapsed)}, "
                    f"ETA {_format_duration(total_eta)}",
                    flush=True,
                )

        promoted = _promote(summaries, resolved)
        all_summaries.extend(summaries)
        n_output_dir = per_n_dir / f"N{scan_count:03d}"
        n_output_dir.mkdir()
        n_runs_path = n_output_dir / "calibration_runs.csv"
        n_summary_path = n_output_dir / "subset_summary.csv"
        _write_csv(n_runs_path, n_runs)
        _write_csv(n_summary_path, summaries)
        plot_paths = _write_n_plots(
            directory=n_output_dir / "plots",
            scan_count=scan_count,
            run_rows=n_runs,
            summaries=summaries,
            resolved=resolved,
        )
        promoted_csv, promoted_npz = _write_promoted(
            directory=promoted_dir,
            scan_count=scan_count,
            promoted=promoted,
            subsets=subsets,
            bank=bank,
        )
        eligible_count = sum(
            bool(row["eligible_for_promotion"]) for row in summaries
        )
        print(
            f"[N={scan_count} {n_index}/{len(resolved['scan_counts'])}] "
            f"eligible={eligible_count}/{len(summaries)}, promoted={len(promoted)}, "
            f"time={_format_duration(time.perf_counter() - n_started)}",
            flush=True,
        )

        completed_paths = (
            subset_path,
            n_runs_path,
            n_summary_path,
            *plot_paths,
            promoted_csv,
            promoted_npz,
        )
        for path in completed_paths:
            manifest["files"][path.relative_to(output_dir).as_posix()] = _sha256(path)
        manifest["per_n"][str(scan_count)] = {
            "status": "complete",
            "calibration_runs": len(n_runs),
            "subsets": len(summaries),
            "eligible_for_promotion": eligible_count,
            "promoted": len(promoted),
            "directory": n_output_dir.relative_to(output_dir).as_posix(),
        }
        manifest["completed_scan_counts"].append(scan_count)
        _write_json(output_dir / "manifest.json", manifest)

    runs_path = output_dir / "calibration_runs.csv"
    summary_path = output_dir / "subset_summary.csv"
    _write_csv(runs_path, all_runs)
    _write_csv(summary_path, all_summaries)
    manifest["files"][runs_path.relative_to(output_dir).as_posix()] = _sha256(runs_path)
    manifest["files"][summary_path.relative_to(output_dir).as_posix()] = _sha256(summary_path)
    manifest["status"] = "complete"
    manifest["counts"] = {
        "candidates": resolved["candidate_count"],
        "subsets": len(all_summaries),
        "calibration_runs": len(all_runs),
        "eligible_subsets": sum(
            bool(row["eligible_for_promotion"]) for row in all_summaries
        ),
        "promoted_subsets": sum(bool(row["promoted"]) for row in all_summaries),
    }
    _write_json(output_dir / "manifest.json", manifest)
    return output_dir


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = _load_config(args.config)
    if args.smoke:
        _apply_smoke_override(config)
    resolved = _validate_config(config)
    raw_output = args.output_dir or Path(resolved["output_dir"])
    output_dir = raw_output if raw_output.is_absolute() else PROJECT_ROOT / raw_output
    if args.validate_only:
        print(
            json.dumps(
                _jsonable(_resolved_summary(resolved, output_dir)),
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    completed = run(config, args.output_dir)
    print(f"Completed phase-1 screening: {completed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
