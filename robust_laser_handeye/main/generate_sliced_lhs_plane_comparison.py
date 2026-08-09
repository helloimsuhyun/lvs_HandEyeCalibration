#!/usr/bin/env python3
"""Generate paired single/three-plane datasets from an exact sliced LHS.

Design properties for S=3 slices and N=3m total poses:
  * the union is an N-point Latin hypercube in every active parameter;
  * every slice is independently an m-point Latin hypercube;
  * single-plane receives the complete union on plane 0;
  * three-plane receives slice k on physical plane k;
  * total scan counts and the union of relative-pose parameters are identical.

A whole sliced design is rejected and regenerated when any required profile is
infeasible. Individual points are never replaced, so the sliced-LHS property is
not damaged. The generator is resumable after interruption.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence

import numpy as np

from main import generate_independent_random_plane_comparison as base
from main import generate_plane_uniform_comparison as uniform

from laser_handeye.calibration_dataset import (
    AcquisitionGroup,
    CalibrationDataset,
    logical_dataset_sha256,
    save_calibration_dataset,
)
from laser_handeye.data import LaserScan
from laser_handeye.simulation import sample_random_handeye


SCHEMA = "laser_handeye.sliced_lhs_equal_budget_comparison"
SCHEMA_VERSION = 1
SLICE_COUNT = 3

PARAMETER_NAMES = (
    "target_u_mm",
    "target_v_mm",
    "center_depth_mm",
    "view_tilt_deg",
    "view_azimuth_deg",
    "sensor_roll_deg",
)


def _positive_int(text: str) -> int:
    value = int(text)
    if value <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return value


def _nonnegative_int(text: str) -> int:
    value = int(text)
    if value < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return value


def _finite_pair(values: Sequence[float], name: str) -> tuple[float, float]:
    if len(values) != 2:
        raise ValueError(f"{name} must contain exactly two values")
    lower, upper = map(float, values)
    if not np.isfinite(lower) or not np.isfinite(upper) or lower > upper:
        raise ValueError(f"invalid {name}: {values}")
    return lower, upper


def _parse_pose_range_spec(text: str) -> tuple[float, float, float, float, float, float]:
    parts = text.split(":")
    if len(parts) != 6:
        raise argparse.ArgumentTypeError(
            "pose range must be TILT_MIN:TILT_MAX:AZ_MIN:AZ_MAX:ROLL_MIN:ROLL_MAX"
        )
    try:
        values = tuple(float(value) for value in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid pose range: {text}") from exc
    t0, t1, a0, a1, r0, r1 = values
    if not (0.0 <= t0 <= t1 < 89.0 and a0 <= a1 and r0 <= r1):
        raise argparse.ArgumentTypeError(f"invalid pose range: {text}")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate resumable exact sliced-LHS single/three-plane datasets."
    )
    parser.add_argument("--trials", type=_positive_int, default=100)
    parser.add_argument("--seed", type=_nonnegative_int, default=17)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--total-scans", type=_positive_int, default=108)

    parser.add_argument("--profile-points", type=_positive_int, default=100)
    parser.add_argument("--profile-half-width-mm", type=float, default=25.0)
    parser.add_argument("--tangent-range-mm", type=float, default=100.0)
    parser.add_argument(
        "--profile-depth-range-mm",
        type=float,
        nargs=2,
        default=(77.0, 123.0),
        help=(
            "Valid sensor-profile Z interval. This is the sensor measurement "
            "range used by strict feasibility."
        ),
    )
    parser.add_argument(
        "--center-depth-range-mm",
        type=float,
        nargs=2,
        default=(95.0, 105.0),
        help=(
            "Sampling interval for the target-center depth d. This is separate "
            "from the valid profile-Z interval."
        ),
    )

    parser.add_argument(
        "--uniform-target-u-range-mm", type=float, nargs=2, default=(0.0, 0.0)
    )
    parser.add_argument(
        "--uniform-target-v-range-mm", type=float, nargs=2, default=(0.0, 0.0)
    )
    parser.add_argument(
        "--uniform-view-tilt-range-deg", type=float, nargs=2, default=(35.0, 75.0)
    )
    parser.add_argument(
        "--uniform-view-azimuth-range-deg", type=float, nargs=2, default=(20.0, 70.0)
    )
    parser.add_argument(
        "--uniform-sensor-roll-range-deg", type=float, nargs=2, default=(-90.0, 90.0)
    )

    parser.add_argument(
        "--shared-feasibility-pose-range",
        type=_parse_pose_range_spec,
        action="append",
        default=[],
        help=(
            "Additional pose range that must accept the same normalized sliced design. "
            "Repeat this option for restricted/moderate/wide to guarantee matched "
            "normalized samples across the pose-range experiment."
        ),
    )
    parser.add_argument(
        "--shared-feasibility-target-range-mm",
        type=float,
        nargs=4,
        action="append",
        default=[],
        metavar=("U_MIN", "U_MAX", "V_MIN", "V_MAX"),
        help=(
            "Additional target U/V range that must accept the same normalized "
            "sliced design. Repeat this option to obtain matched designs across "
            "target-range conditions."
        ),
    )
    parser.add_argument(
        "--max-design-attempts",
        type=_positive_int,
        default=200,
        help="Maximum whole-design resampling attempts per Monte-Carlo trial.",
    )
    parser.add_argument(
        "--feasibility-mode",
        choices=("strict", "geometric"),
        default="strict",
        help=(
            "strict enforces the original sensor-depth and finite-patch bounds; "
            "geometric only requires a finite plane/profile intersection and "
            "numerically valid point-to-plane geometry."
        ),
    )

    parser.add_argument(
        "--plane-angle-range-deg", type=float, nargs=2, default=(-15.0, 15.0)
    )
    parser.add_argument(
        "--plane-center-xy-range-mm", type=float, nargs=2, default=(-100.0, 100.0)
    )
    parser.add_argument(
        "--plane-center-z-range-mm", type=float, nargs=2, default=(400.0, 550.0)
    )
    parser.add_argument("--max-local-pose-trials", type=_positive_int, default=200_000)
    parser.add_argument("--min-abs-plane-normal-z", type=float, default=1e-4)
    parser.add_argument("--verification-atol", type=float, default=1e-8)
    return parser


def _validated_configs(
    args: argparse.Namespace,
) -> tuple[base.FairGenerationConfig, uniform.UniformConfig, list[uniform.UniformConfig]]:
    if args.total_scans < 9 or args.total_scans % SLICE_COUNT != 0:
        raise SystemExit("--total-scans must be at least 9 and divisible by 3")
    if args.profile_points < 2:
        raise SystemExit("--profile-points must be at least 2")
    if args.profile_half_width_mm <= 0.0 or args.tangent_range_mm <= 0.0:
        raise SystemExit("profile dimensions must be positive")

    profile_depth = _finite_pair(
        args.profile_depth_range_mm,
        "valid profile depth range",
    )
    if profile_depth[0] <= 0.0 or profile_depth[0] == profile_depth[1]:
        raise SystemExit(
            "valid profile depth range must be positive and non-zero"
        )

    center_depth = _finite_pair(
        args.center_depth_range_mm,
        "center depth range",
    )
    if center_depth[0] <= 0.0 or center_depth[0] == center_depth[1]:
        raise SystemExit(
            "center depth range must be positive and non-zero"
        )
    if (
        center_depth[0] < profile_depth[0]
        or center_depth[1] > profile_depth[1]
    ):
        raise SystemExit(
            "center depth range must lie inside valid profile depth range"
        )

    tilt = _finite_pair(args.uniform_view_tilt_range_deg, "tilt range")
    if tilt[0] < 0.0 or tilt[1] >= 89.0:
        raise SystemExit("tilt range must lie within [0, 89)")

    fair_config = base.FairGenerationConfig(
        total_scans=int(args.total_scans),
        profile_points=int(args.profile_points),
        profile_half_width_mm=float(args.profile_half_width_mm),
        tangent_range_mm=float(args.tangent_range_mm),
        profile_depth_range_mm=profile_depth,
        view_tilt_range_deg=tilt,
        view_azimuth_range_deg=_finite_pair(
            args.uniform_view_azimuth_range_deg, "azimuth range"
        ),
        sensor_roll_range_deg=_finite_pair(
            args.uniform_sensor_roll_range_deg, "roll range"
        ),
        plane_angle_range_deg=_finite_pair(
            args.plane_angle_range_deg, "plane angle range"
        ),
        plane_center_xy_range_mm=_finite_pair(
            args.plane_center_xy_range_mm, "plane center XY range"
        ),
        plane_center_z_range_mm=_finite_pair(
            args.plane_center_z_range_mm, "plane center Z range"
        ),
        max_local_pose_trials=int(args.max_local_pose_trials),
        min_abs_plane_normal_z=float(args.min_abs_plane_normal_z),
        verification_atol=float(args.verification_atol),
        min_effective_cross_plane_angle_deg=0.0,
    )
    output_config = uniform.UniformConfig(
        target_u_range_mm=_finite_pair(
            args.uniform_target_u_range_mm, "target U range"
        ),
        target_v_range_mm=_finite_pair(
            args.uniform_target_v_range_mm, "target V range"
        ),
        depth_range_mm=center_depth,
        tilt_range_deg=tilt,
        azimuth_range_deg=_finite_pair(
            args.uniform_view_azimuth_range_deg, "azimuth range"
        ),
        roll_range_deg=_finite_pair(
            args.uniform_sensor_roll_range_deg, "roll range"
        ),
        max_batches=int(args.max_design_attempts),
        batch_multiplier=1,
    )

    feasibility_configs = [output_config]

    def config_key(
        config: uniform.UniformConfig,
    ) -> tuple[tuple[float, float], ...]:
        return (
            config.target_u_range_mm,
            config.target_v_range_mm,
            config.depth_range_mm,
            config.tilt_range_deg,
            config.azimuth_range_deg,
            config.roll_range_deg,
        )

    def add_feasibility_config(candidate: uniform.UniformConfig) -> None:
        existing_keys = {config_key(config) for config in feasibility_configs}
        if config_key(candidate) not in existing_keys:
            feasibility_configs.append(candidate)

    for t0, t1, a0, a1, r0, r1 in args.shared_feasibility_pose_range:
        candidate = replace(
            output_config,
            tilt_range_deg=(t0, t1),
            azimuth_range_deg=(a0, a1),
            roll_range_deg=(r0, r1),
        )
        add_feasibility_config(candidate)

    for u0, u1, v0, v1 in args.shared_feasibility_target_range_mm:
        candidate = replace(
            output_config,
            target_u_range_mm=_finite_pair((u0, u1), "shared target U range"),
            target_v_range_mm=_finite_pair((v0, v1), "shared target V range"),
        )
        add_feasibility_config(candidate)

    return fair_config, output_config, feasibility_configs


def _active_parameter_indices(
    configs: Sequence[uniform.UniformConfig],
) -> list[int]:
    bounds_by_config = [
        (
            config.target_u_range_mm,
            config.target_v_range_mm,
            config.depth_range_mm,
            config.tilt_range_deg,
            config.azimuth_range_deg,
            config.roll_range_deg,
        )
        for config in configs
    ]
    indices = [
        index
        for index in range(len(PARAMETER_NAMES))
        if any(
            not np.isclose(
                bounds[index][0],
                bounds[index][1],
                rtol=0.0,
                atol=0.0,
            )
            for bounds in bounds_by_config
        )
    ]
    if not indices:
        raise ValueError("at least one relative-pose parameter must vary")
    return indices


def _sliced_lhs(
    rng: np.random.Generator,
    *,
    slice_count: int,
    points_per_slice: int,
    dimensions: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return values, fine-stratum IDs, and slice-level coarse IDs.

    values has shape (slice_count, points_per_slice, dimensions).
    """
    total = slice_count * points_per_slice
    values = np.empty((slice_count, points_per_slice, dimensions), dtype=float)
    fine = np.empty_like(values, dtype=np.int64)
    coarse = np.empty_like(values, dtype=np.int64)

    for dimension in range(dimensions):
        coarse_by_row = np.stack(
            [rng.permutation(points_per_slice) for _ in range(slice_count)],
            axis=0,
        )
        row_for_coarse = np.empty_like(coarse_by_row)
        for slice_id in range(slice_count):
            row_for_coarse[slice_id, coarse_by_row[slice_id]] = np.arange(
                points_per_slice
            )

        for coarse_id in range(points_per_slice):
            sub_strata = rng.permutation(slice_count)
            for slice_id in range(slice_count):
                row = int(row_for_coarse[slice_id, coarse_id])
                fine_id = coarse_id * slice_count + int(sub_strata[slice_id])
                fine[slice_id, row, dimension] = fine_id
                coarse[slice_id, row, dimension] = coarse_id
                values[slice_id, row, dimension] = (
                    fine_id + rng.random()
                ) / total

    _audit_sliced_lhs(values, slice_count=slice_count)
    return values, fine, coarse


def _audit_sliced_lhs(values: np.ndarray, *, slice_count: int) -> None:
    values = np.asarray(values, dtype=float)
    if values.ndim != 3 or values.shape[0] != slice_count:
        raise ValueError("invalid sliced-LHS array shape")
    _, points_per_slice, dimensions = values.shape
    total = slice_count * points_per_slice

    if np.any(values < 0.0) or np.any(values >= 1.0):
        raise RuntimeError("normalized sliced-LHS values left [0, 1)")

    for dimension in range(dimensions):
        union_bins = np.floor(values[:, :, dimension].reshape(-1) * total).astype(int)
        if not np.array_equal(np.sort(union_bins), np.arange(total)):
            raise RuntimeError(
                f"union is not an exact {total}-point LHS in dimension {dimension}"
            )
        for slice_id in range(slice_count):
            slice_bins = np.floor(
                values[slice_id, :, dimension] * points_per_slice
            ).astype(int)
            if not np.array_equal(np.sort(slice_bins), np.arange(points_per_slice)):
                raise RuntimeError(
                    f"slice {slice_id} is not an exact {points_per_slice}-point "
                    f"LHS in dimension {dimension}"
                )


def _expand_normalized_rows(
    active_values: np.ndarray,
    active_indices: Sequence[int],
) -> np.ndarray:
    rows = np.full((*active_values.shape[:-1], len(PARAMETER_NAMES)), 0.5, dtype=float)
    for source_dimension, parameter_index in enumerate(active_indices):
        rows[..., parameter_index] = active_values[..., source_dimension]
    return rows


def _normalized_design_sha256(values: np.ndarray) -> str:
    canonical = np.ascontiguousarray(np.asarray(values, dtype="<f8"))
    digest = hashlib.sha256()
    digest.update(np.asarray(canonical.shape, dtype="<i8").tobytes())
    digest.update(canonical.tobytes())
    return digest.hexdigest()


def _augment_scan_metadata(
    scan: LaserScan,
    *,
    normalized_row: np.ndarray,
    active_indices: Sequence[int],
    active_fine_strata: np.ndarray,
    active_coarse_strata: np.ndarray,
    design_attempt: int,
) -> LaserScan:
    metadata = deepcopy(scan.meta)
    metadata.update(
        {
            "comparison_schema": SCHEMA,
            "comparison_schema_version": SCHEMA_VERSION,
            "pose_selection_strategy": "exact_sliced_latin_hypercube",
            "sliced_lhs_slice_id": int(metadata["relative_pose_block_id"]),
            "sliced_lhs_design_attempt": int(design_attempt),
            "sliced_lhs_active_parameters": [
                PARAMETER_NAMES[index] for index in active_indices
            ],
            "sliced_lhs_normalized_parameters": {
                name: float(normalized_row[index])
                for index, name in enumerate(PARAMETER_NAMES)
            },
            "sliced_lhs_fine_strata": {
                PARAMETER_NAMES[index]: int(active_fine_strata[position])
                for position, index in enumerate(active_indices)
            },
            "sliced_lhs_coarse_strata": {
                PARAMETER_NAMES[index]: int(active_coarse_strata[position])
                for position, index in enumerate(active_indices)
            },
        }
    )
    return LaserScan(
        T_base_ef=scan.T_base_ef,
        points_s=scan.points_s,
        plane_id=scan.plane_id,
        scan_id=scan.scan_id,
        meta=metadata,
    )



def _profile_geometry(
    *,
    plane: base.PlaneFrame,
    common_center: np.ndarray,
    T_base_s: np.ndarray,
    x_values: np.ndarray,
    fair_config: base.FairGenerationConfig,
    feasibility_mode: str,
) -> base.ProfileFeasibility | None:
    if feasibility_mode == "strict":
        return base._profile_feasibility(
            plane=plane,
            common_center=common_center,
            T_base_s=T_base_s,
            x_values=x_values,
            config=fair_config,
        )

    normal_sensor = T_base_s[:3, :3].T @ plane.n
    if abs(float(normal_sensor[2])) < fair_config.min_abs_plane_normal_z:
        return None
    try:
        z_values = base._profile_z_values(plane, T_base_s, x_values)
    except (ValueError, FloatingPointError):
        return None
    if not np.all(np.isfinite(z_values)):
        return None

    points_s = np.column_stack(
        [x_values, np.zeros_like(x_values), z_values]
    )
    points_base = (
        T_base_s[:3, :3] @ points_s.T
    ).T + T_base_s[:3, 3]
    residual = points_base @ plane.n - plane.l
    residual_max = float(np.max(np.abs(residual)))
    if residual_max > fair_config.verification_atol:
        return None

    centered = (
        points_base
        - np.asarray(common_center, dtype=float).reshape(1, 3)
    )
    local_u = centered @ plane.u
    local_v = centered @ plane.v
    return base.ProfileFeasibility(
        profile_z_min_mm=float(np.min(z_values)),
        profile_z_max_mm=float(np.max(z_values)),
        profile_local_u_min_mm=float(np.min(local_u)),
        profile_local_u_max_mm=float(np.max(local_u)),
        profile_local_v_min_mm=float(np.min(local_v)),
        profile_local_v_max_mm=float(np.max(local_v)),
        ideal_plane_residual_max_abs_mm=residual_max,
    )


def _evaluate_design(
    *,
    normalized_rows: np.ndarray,
    active_fine: np.ndarray,
    active_coarse: np.ndarray,
    active_indices: Sequence[int],
    config: uniform.UniformConfig,
    master_seed: int,
    trial_index: int,
    frames: Sequence[base.PlaneFrame],
    common_center: np.ndarray,
    x_values: np.ndarray,
    fair_config: base.FairGenerationConfig,
    design_attempt: int,
    feasibility_mode: str,
) -> list[dict[str, Any]]:
    points_per_slice = normalized_rows.shape[1]
    evaluated: list[dict[str, Any]] = []

    for slice_id in range(SLICE_COUNT):
        for index_in_slice in range(points_per_slice):
            sample_id = slice_id * points_per_slice + index_in_slice
            row = normalized_rows[slice_id, index_in_slice]
            pose = uniform._uniform_candidate_from_row(
                row,
                sample_id=sample_id,
                block_id=slice_id,
                index_in_block=index_in_slice,
                master_seed=master_seed,
                trial_index=trial_index,
                config=config,
            )

            T_single = uniform._make_sensor_pose_relative_to_plane(
                frames[0], common_center, pose
            )
            T_three = uniform._make_sensor_pose_relative_to_plane(
                frames[slice_id], common_center, pose
            )
            single_feasibility = _profile_geometry(
                plane=frames[0],
                common_center=common_center,
                T_base_s=T_single,
                x_values=x_values,
                fair_config=fair_config,
                feasibility_mode=feasibility_mode,
            )
            if single_feasibility is None:
                raise RuntimeError(
                    f"single infeasible: slice={slice_id}, index={index_in_slice}"
                )
            three_feasibility = _profile_geometry(
                plane=frames[slice_id],
                common_center=common_center,
                T_base_s=T_three,
                x_values=x_values,
                fair_config=fair_config,
                feasibility_mode=feasibility_mode,
            )
            if three_feasibility is None:
                raise RuntimeError(
                    f"three infeasible: slice={slice_id}, index={index_in_slice}"
                )

            evaluated.append(
                {
                    "pose": pose,
                    "normalized_row": row.copy(),
                    "fine": active_fine[slice_id, index_in_slice].copy(),
                    "coarse": active_coarse[slice_id, index_in_slice].copy(),
                    "T_single": T_single,
                    "T_three": T_three,
                    "single_feasibility": single_feasibility,
                    "three_feasibility": three_feasibility,
                    "design_attempt": design_attempt,
                }
            )
    return evaluated


def _sample_jointly_feasible_sliced_design(
    *,
    master_seed: int,
    trial_index: int,
    fair_config: base.FairGenerationConfig,
    output_config: uniform.UniformConfig,
    feasibility_configs: Sequence[uniform.UniformConfig],
    frames: Sequence[base.PlaneFrame],
    common_center: np.ndarray,
    x_values: np.ndarray,
    max_attempts: int,
    feasibility_mode: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    points_per_slice = fair_config.total_scans // SLICE_COUNT
    active_indices = _active_parameter_indices(feasibility_configs)
    rejection_messages: list[str] = []

    for attempt in range(max_attempts):
        attempt_sequence = np.random.SeedSequence(
            [master_seed, trial_index, 0x534C4943, attempt]
        )
        active_values, active_fine, active_coarse = _sliced_lhs(
            np.random.default_rng(attempt_sequence),
            slice_count=SLICE_COUNT,
            points_per_slice=points_per_slice,
            dimensions=len(active_indices),
        )
        normalized_rows = _expand_normalized_rows(active_values, active_indices)

        output_evaluation: list[dict[str, Any]] | None = None
        rejected = False
        for config_index, config in enumerate(feasibility_configs):
            try:
                evaluation = _evaluate_design(
                    normalized_rows=normalized_rows,
                    active_fine=active_fine,
                    active_coarse=active_coarse,
                    active_indices=active_indices,
                    config=config,
                    master_seed=master_seed,
                    trial_index=trial_index,
                    frames=frames,
                    common_center=common_center,
                    x_values=x_values,
                    fair_config=fair_config,
                    design_attempt=attempt,
                    feasibility_mode=feasibility_mode,
                )
            except (ValueError, FloatingPointError, RuntimeError) as exc:
                if len(rejection_messages) < 10:
                    rejection_messages.append(
                        f"attempt={attempt}, range_index={config_index}: {exc}"
                    )
                rejected = True
                break
            if config_index == 0:
                output_evaluation = evaluation

        if rejected:
            continue
        if output_evaluation is None:
            raise AssertionError("output evaluation was not produced")

        normalized_hash = _normalized_design_sha256(active_values)
        stats = {
            "design_attempt": attempt,
            "attempts_used": attempt + 1,
            "normalized_design_sha256": normalized_hash,
            "slice_count": SLICE_COUNT,
            "points_per_slice": points_per_slice,
            "total_points": fair_config.total_scans,
            "active_parameter_indices": list(map(int, active_indices)),
            "active_parameter_names": [
                PARAMETER_NAMES[index] for index in active_indices
            ],
            "union_lhs_verified": True,
            "slice_lhs_verified": True,
            "joint_feasibility_range_count": len(feasibility_configs),
            "feasibility_mode": feasibility_mode,
            "rejection_examples": rejection_messages,
        }
        return output_evaluation, stats

    raise RuntimeError(
        f"no jointly feasible exact sliced LHS after {max_attempts} whole-design "
        f"attempts; examples={rejection_messages}"
    )


def _build_trial(
    *,
    fair_config: base.FairGenerationConfig,
    output_config: uniform.UniformConfig,
    feasibility_configs: Sequence[uniform.UniformConfig],
    master_seed: int,
    trial_index: int,
    max_design_attempts: int,
    feasibility_mode: str,
) -> tuple[CalibrationDataset, CalibrationDataset, dict[str, Any]]:
    trial_sequence = np.random.SeedSequence([master_seed, trial_index])
    handeye_sequence, plane_sequence = trial_sequence.spawn(2)

    T_ef_s_true, _, _ = sample_random_handeye(
        np.random.default_rng(handeye_sequence)
    )
    frames, common_center, frame_angles_deg = base._make_plane_frames(
        np.random.default_rng(plane_sequence),
        fair_config.plane_angle_range_deg,
        fair_config.plane_center_xy_range_mm,
        fair_config.plane_center_z_range_mm,
    )
    x_values = np.linspace(
        -fair_config.profile_half_width_mm,
        fair_config.profile_half_width_mm,
        fair_config.profile_points,
    )

    evaluated, design_stats = _sample_jointly_feasible_sliced_design(
        master_seed=master_seed,
        trial_index=trial_index,
        fair_config=fair_config,
        output_config=output_config,
        feasibility_configs=feasibility_configs,
        frames=frames,
        common_center=common_center,
        x_values=x_values,
        max_attempts=max_design_attempts,
        feasibility_mode=feasibility_mode,
    )

    active_indices = design_stats["active_parameter_indices"]
    single_scans: list[LaserScan] = []
    three_scans: list[LaserScan] = []

    for item in evaluated:
        pose = item["pose"]
        single_scan = uniform._simulate_uniform_scan(
            T_ef_s_true=T_ef_s_true,
            plane=frames[0],
            plane_id=0,
            T_base_s=item["T_single"],
            pose=pose,
            feasibility=item["single_feasibility"],
            strategy="single_plane_exact_sliced_lhs",
            x_values=x_values,
        )
        three_scan = uniform._simulate_uniform_scan(
            T_ef_s_true=T_ef_s_true,
            plane=frames[int(pose.block_id)],
            plane_id=int(pose.block_id),
            T_base_s=item["T_three"],
            pose=pose,
            feasibility=item["three_feasibility"],
            strategy="three_plane_exact_sliced_lhs",
            x_values=x_values,
        )
        single_scans.append(
            _augment_scan_metadata(
                single_scan,
                normalized_row=item["normalized_row"],
                active_indices=active_indices,
                active_fine_strata=item["fine"],
                active_coarse_strata=item["coarse"],
                design_attempt=item["design_attempt"],
            )
        )
        three_scans.append(
            _augment_scan_metadata(
                three_scan,
                normalized_row=item["normalized_row"],
                active_indices=active_indices,
                active_fine_strata=item["fine"],
                active_coarse_strata=item["coarse"],
                design_attempt=item["design_attempt"],
            )
        )

    points_per_slice = fair_config.total_scans // SLICE_COUNT
    three_group_ids = [
        f"plane_{int(scan.plane_id)}" for scan in three_scans
    ]
    three_sequence_indices = [
        int(scan.meta["relative_pose_index_in_block"]) for scan in three_scans
    ]

    selection_summary = {
        "uniform_design": "exact_sliced_latin_hypercube",
        "relative_pose_union_identical_between_single_and_three": True,
        "single_assignment": "all slices mapped to plane 0",
        "three_assignment": "slice k mapped to physical plane k",
        "equal_total_scan_budget": True,
        "uniform_config": asdict(output_config),
        **design_stats,
    }

    single_dataset = base._build_dataset(
        strategy="single_plane",
        scans=single_scans,
        scan_group_ids=["plane_0"] * len(single_scans),
        sequence_indices=list(range(len(single_scans))),
        groups=[
            AcquisitionGroup(
                group_id="plane_0",
                acquisition_role="calibration",
                motion_kind="plane_relative_exact_sliced_lhs",
                plane_id=0,
            )
        ],
        T_ef_s_true=T_ef_s_true,
        frames=frames,
        used_plane_ids=(0,),
        common_center=common_center,
        frame_angles_deg=frame_angles_deg,
        trial_index=trial_index,
        config=fair_config,
        selection_strategy="exact_sliced_lhs",
        selection_summary=selection_summary,
    )
    three_dataset = base._build_dataset(
        strategy="three_plane",
        scans=three_scans,
        scan_group_ids=three_group_ids,
        sequence_indices=three_sequence_indices,
        groups=[
            AcquisitionGroup(
                group_id=f"plane_{plane_id}",
                acquisition_role="calibration",
                motion_kind="plane_relative_exact_sliced_lhs",
                plane_id=plane_id,
            )
            for plane_id in range(SLICE_COUNT)
        ],
        T_ef_s_true=T_ef_s_true,
        frames=frames,
        used_plane_ids=(0, 1, 2),
        common_center=common_center,
        frame_angles_deg=frame_angles_deg,
        trial_index=trial_index,
        config=fair_config,
        selection_strategy="exact_sliced_lhs",
        selection_summary=selection_summary,
    )

    single_parameters = [
        scan.meta["relative_pose_parameters"] for scan in single_scans
    ]
    three_parameters = [
        scan.meta["relative_pose_parameters"] for scan in three_scans
    ]
    if single_parameters != three_parameters:
        raise RuntimeError("single and three relative-pose unions differ")

    plane_counts = {
        str(plane_id): int(
            sum(int(scan.plane_id) == plane_id for scan in three_scans)
        )
        for plane_id in range(SLICE_COUNT)
    }
    if any(count != points_per_slice for count in plane_counts.values()):
        raise RuntimeError(f"unbalanced three-plane sliced assignment: {plane_counts}")

    comparison = {
        "trial_index": trial_index,
        "T_ef_s_true_sha256": base._array_sha256(T_ef_s_true),
        "normalized_design_sha256": design_stats["normalized_design_sha256"],
        "design_attempt": design_stats["design_attempt"],
        "union_lhs_verified": True,
        "slice_lhs_verified": True,
        "relative_pose_union_identical": True,
        "equal_total_scan_budget": True,
        "single_scan_count": len(single_scans),
        "three_scan_count": len(three_scans),
        "three_scans_per_plane": plane_counts,
        "single_diversity": uniform._effective_normal_metrics(single_scans, frames),
        "three_diversity": uniform._effective_normal_metrics(three_scans, frames),
        "design_statistics": design_stats,
    }
    return single_dataset, three_dataset, comparison


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(
            _jsonable(payload),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _new_collection_manifest(
    *,
    key: str,
    acquisition_mode: str,
    trials: int,
    master_seed: int,
    fair_config: base.FairGenerationConfig,
    output_config: uniform.UniformConfig,
) -> dict[str, Any]:
    return {
        "schema": base.COLLECTION_SCHEMA,
        "schema_version": base.COLLECTION_SCHEMA_VERSION,
        "status": "in_progress",
        "generator_variant": SCHEMA,
        "generator_schema_version": SCHEMA_VERSION,
        "strategy": key,
        "acquisition_mode": acquisition_mode,
        "master_seed": master_seed,
        "requested_trials": trials,
        "completed_trials": 0,
        "profile_state": "ideal",
        "noise_applied": False,
        "robot_ik_and_collision_checked": False,
        "fair_config": asdict(fair_config),
        "uniform_config": asdict(output_config),
        "scans_per_trial": fair_config.total_scans,
        "equal_budget": True,
        "exact_sliced_lhs": True,
        "trials": [],
    }


def _initialize_output(
    output_root: Path,
    *,
    args: argparse.Namespace,
    fair_config: base.FairGenerationConfig,
    output_config: uniform.UniformConfig,
) -> tuple[dict[str, Path], dict[str, dict[str, Any]], dict[str, Any]]:
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(
            f"output directory is non-empty but has no compatible manifest: {output_root}"
        )
    output_root.mkdir(parents=True, exist_ok=True)

    roots = {
        "single": output_root / "single_plane_sliced_lhs",
        "three": output_root / "three_plane_sliced_lhs",
    }
    for root in roots.values():
        (root / "trials").mkdir(parents=True, exist_ok=False)

    manifests = {
        "single": _new_collection_manifest(
            key="single_plane_sliced_lhs",
            acquisition_mode="single_plane_exact_sliced_lhs",
            trials=args.trials,
            master_seed=args.seed,
            fair_config=fair_config,
            output_config=output_config,
        ),
        "three": _new_collection_manifest(
            key="three_plane_sliced_lhs",
            acquisition_mode="three_plane_exact_sliced_lhs",
            trials=args.trials,
            master_seed=args.seed,
            fair_config=fair_config,
            output_config=output_config,
        ),
    }
    comparison = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "in_progress",
        "master_seed": args.seed,
        "requested_trials": args.trials,
        "completed_trials": 0,
        "collections": {
            "single": roots["single"].name,
            "three": roots["three"].name,
        },
        "fair_config": asdict(fair_config),
        "uniform_config": asdict(output_config),
        "equal_budget": True,
        "exact_sliced_lhs": True,
        "slice_count": SLICE_COUNT,
        "trials": [],
    }
    for key, root in roots.items():
        _write_json(root / "collection.json", manifests[key])
    _write_json(output_root / "comparison_manifest.json", comparison)
    return roots, manifests, comparison


def _load_resumable_output(
    output_root: Path,
    *,
    args: argparse.Namespace,
    fair_config: base.FairGenerationConfig,
    output_config: uniform.UniformConfig,
) -> tuple[dict[str, Path], dict[str, dict[str, Any]], dict[str, Any]]:
    comparison_path = output_root / "comparison_manifest.json"
    if not comparison_path.is_file():
        return _initialize_output(
            output_root,
            args=args,
            fair_config=fair_config,
            output_config=output_config,
        )

    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    if comparison.get("schema") != SCHEMA:
        raise RuntimeError(
            f"incompatible generator schema in {comparison_path}: "
            f"{comparison.get('schema')!r}; use a new DATASET_ROOT or remove this directory"
        )
    if int(comparison.get("master_seed", -1)) != args.seed:
        raise RuntimeError("existing dataset uses a different generation seed")
    if int(comparison.get("requested_trials", -1)) != args.trials:
        raise RuntimeError("existing dataset uses a different requested trial count")
    if comparison.get("fair_config") != _jsonable(asdict(fair_config)):
        raise RuntimeError("existing dataset uses a different fair_config")
    if comparison.get("uniform_config") != _jsonable(asdict(output_config)):
        raise RuntimeError("existing dataset uses a different uniform_config")

    roots = {
        key: output_root / name
        for key, name in comparison["collections"].items()
    }
    manifests = {}
    for key, root in roots.items():
        manifest_path = root / "collection.json"
        manifests[key] = json.loads(manifest_path.read_text(encoding="utf-8"))

    counts = [
        len(comparison.get("trials", [])),
        *[len(manifest.get("trials", [])) for manifest in manifests.values()],
    ]
    completed = min(counts)
    if completed > args.trials:
        raise RuntimeError("existing dataset contains more trials than requested")

    # Reconcile a crash between writing datasets and writing all manifests.
    comparison["trials"] = comparison.get("trials", [])[:completed]
    comparison["completed_trials"] = completed
    comparison["status"] = "in_progress"
    for key, manifest in manifests.items():
        manifest["trials"] = manifest.get("trials", [])[:completed]
        manifest["completed_trials"] = completed
        manifest["status"] = "in_progress"
        root = roots[key]
        for trial_dir in sorted((root / "trials").glob("trial_*")):
            try:
                trial_index = int(trial_dir.name.split("_")[-1])
            except ValueError:
                continue
            if trial_index >= completed:
                shutil.rmtree(trial_dir)

    for key, root in roots.items():
        _write_json(root / "collection.json", manifests[key])
    _write_json(comparison_path, comparison)

    if completed == args.trials:
        comparison["status"] = "complete"
        comparison["completed_trials"] = args.trials
        for key, manifest in manifests.items():
            manifest["status"] = "complete"
            manifest["completed_trials"] = args.trials
            _write_json(roots[key] / "collection.json", manifest)
        _write_json(comparison_path, comparison)

    return roots, manifests, comparison


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    fair_config, output_config, feasibility_configs = _validated_configs(args)
    output_root = args.output_dir.expanduser()

    roots, manifests, comparison = _load_resumable_output(
        output_root,
        args=args,
        fair_config=fair_config,
        output_config=output_config,
    )
    start_trial = int(comparison["completed_trials"])

    if comparison.get("status") == "complete" and start_trial == args.trials:
        print(
            f"[sliced-lhs] complete dataset already exists: {output_root} "
            f"({start_trial}/{args.trials} trials)"
        )
        return 0

    for trial_index in range(start_trial, args.trials):
        relative_path = Path("trials") / f"trial_{trial_index:06d}"
        for root in roots.values():
            destination = root / relative_path
            if destination.exists():
                shutil.rmtree(destination)

        single_dataset, three_dataset, trial_comparison = _build_trial(
            fair_config=fair_config,
            output_config=output_config,
            feasibility_configs=feasibility_configs,
            master_seed=args.seed,
            trial_index=trial_index,
            max_design_attempts=args.max_design_attempts,
            feasibility_mode=args.feasibility_mode,
        )
        datasets = {
            "single": single_dataset,
            "three": three_dataset,
        }
        for key, dataset in datasets.items():
            save_calibration_dataset(dataset, roots[key] / relative_path)
            manifests[key]["trials"].append(
                {
                    "trial_index": trial_index,
                    "relative_path": relative_path.as_posix(),
                    "logical_dataset_sha256": logical_dataset_sha256(dataset),
                    "pair_trial_index": trial_index,
                    "T_ef_s_true_sha256": trial_comparison[
                        "T_ef_s_true_sha256"
                    ],
                    "normalized_design_sha256": trial_comparison[
                        "normalized_design_sha256"
                    ],
                }
            )
            manifests[key]["completed_trials"] = trial_index + 1

        comparison["trials"].append(
            {
                **trial_comparison,
                "single_relative_path": (
                    Path(roots["single"].name) / relative_path
                ).as_posix(),
                "three_relative_path": (
                    Path(roots["three"].name) / relative_path
                ).as_posix(),
            }
        )
        comparison["completed_trials"] = trial_index + 1

        for key, root in roots.items():
            _write_json(root / "collection.json", manifests[key])
        _write_json(output_root / "comparison_manifest.json", comparison)

        print(
            f"[sliced-lhs] {trial_index + 1}/{args.trials}: "
            f"N={fair_config.total_scans}, "
            f"slice={fair_config.total_scans // SLICE_COUNT}, "
            f"attempt={trial_comparison['design_attempt']}"
        )

    for key, root in roots.items():
        manifests[key]["status"] = "complete"
        manifests[key]["completed_trials"] = args.trials
        _write_json(root / "collection.json", manifests[key])
    comparison["status"] = "complete"
    comparison["completed_trials"] = args.trials
    _write_json(output_root / "comparison_manifest.json", comparison)

    print(f"single sliced LHS: {roots['single']}")
    print(f"three sliced LHS : {roots['three']}")
    print(f"comparison       : {output_root / 'comparison_manifest.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
