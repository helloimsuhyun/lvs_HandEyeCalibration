#!/usr/bin/env python3
"""Generate the two remaining plane-calibration experiments.

Collections written per Monte-Carlo trial
-----------------------------------------
1. single_plane_uniform
   All scans observe plane 0. Relative sensor poses are stratified uniformly
   in target-plane coordinates.

2. three_plane_random
   Existing shared-global random baseline from
   generate_independent_random_plane_comparison.py.

3. three_plane_uniform
   Every physical plane receives the complete relative-pose parameter set used
   by single_plane_uniform. With 108 base poses this collection has 324 scans.

This supports the two comparisons:

    Experiment 2 : single_uniform vs three_random
    Additional   : single_uniform vs three_uniform

Pose-replication rule for the uniform comparison
------------------------------------------------
For each trial, three Latin-hypercube blocks are generated, one block per
base-pose group. The complete three-block parameter set is applied to plane 0
for single_uniform and independently to every physical plane for
three_uniform. Thus every three_uniform plane receives all relative depth,
in-plane target position, tilt, azimuth, and roll samples.

Example
-------
PYTHONPATH=. python3 main/generate_plane_uniform_comparison.py \
  --trials 100 \
  --seed 17 \
  --output-dir dataset/single_uniform_three_random_uniform_81 \
  --total-scans 81 \
  --profile-points 100 \
  --profile-half-width-mm 25 \
  --tangent-range-mm 100 \
  --profile-depth-range-mm 60 150 \
  --uniform-target-u-range-mm -40 40 \
  --uniform-target-v-range-mm -40 40 \
  --uniform-view-tilt-range-deg 10 60 \
  --uniform-view-azimuth-range-deg -180 180 \
  --uniform-sensor-roll-range-deg -180 180

Notes
-----
* Generated profiles are ideal, matching the historical random-only generator.
  Apply the same downstream noise/initialization/calibration command to all
  three collections.
* Robot IK and collision are not checked.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

# Reuse the validated dataset, target, profile, and persistence implementation.
from main import generate_independent_random_plane_comparison as base

from laser_handeye.calibration_dataset import (
    AcquisitionGroup,
    CalibrationDataset,
    logical_dataset_sha256,
    save_calibration_dataset,
)
from laser_handeye.data import LaserScan
from laser_handeye.pose_design import (
    PlaneFrame as DesignPlaneFrame,
    PlaneRelativePose as DesignPlaneRelativePose,
    latin_hypercube,
    sensor_pose_from_plane_relative,
)
from laser_handeye.simulation import sample_random_handeye


SCHEMA = "laser_handeye.uniform_relative_pose_comparison"
SCHEMA_VERSION = 4

VIEW_POSE_CONVENTION = {
    "origin": (
        "simulated sensor origin; physical and sensor-coordinate origins "
        "coincide (offset 0 mm)"
    ),
    "u_v": "sensor +Z axis / target-plane intersection coordinates",
    "distance": (
        "Euclidean simulated-sensor-origin to axis/plane intersection distance; "
        "center_depth_mm is the legacy-compatible field name"
    ),
    "tilt": "angle(sensor -Z, oriented target-plane normal)",
    "azimuth": "target-plane azimuth of projected sensor +Z",
    "normal_azimuth_sensor": (
        "atan2 of the target-plane normal projected into sensor XY"
    ),
}


@dataclass(frozen=True)
class UniformRelativePose:
    sample_id: int
    block_id: int
    index_in_block: int
    target_u_mm: float
    target_v_mm: float
    center_depth_mm: float
    view_tilt_deg: float
    view_azimuth_deg: float
    normal_azimuth_sensor_deg: float
    noise_seed: int
    simulation_seed: int


@dataclass(frozen=True)
class UniformConfig:
    target_u_range_mm: tuple[float, float]
    target_v_range_mm: tuple[float, float]
    depth_range_mm: tuple[float, float]
    tilt_range_deg: tuple[float, float]
    azimuth_range_deg: tuple[float, float]
    normal_azimuth_sensor_range_deg: tuple[float, float]
    max_batches: int
    batch_multiplier: int


def _sensor_z_axis_from_view_angles(
    frame: base.PlaneFrame,
    view_tilt_deg: float,
    view_azimuth_deg: float,
    *,
    name: str = "canonical sensor +Z view direction",
) -> np.ndarray:
    """Return sensor +Z using the canonical target-plane view convention."""
    tilt = np.deg2rad(float(view_tilt_deg))
    azimuth = np.deg2rad(float(view_azimuth_deg))
    sensor_z = (
        np.sin(tilt) * np.cos(azimuth) * frame.u
        + np.sin(tilt) * np.sin(azimuth) * frame.v
        - np.cos(tilt) * frame.n
    )
    return base._normalize(sensor_z, name)


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate single-uniform, three-random, and three-uniform "
            "plane-calibration collections."
        )
    )
    parser.add_argument("--trials", type=_positive_int, default=100)
    parser.add_argument("--seed", type=_nonnegative_int, default=17)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--total-scans", type=_positive_int, default=81)
    parser.add_argument("--profile-points", type=_positive_int, default=100)
    parser.add_argument("--profile-half-width-mm", type=float, default=25.0)
    parser.add_argument("--tangent-range-mm", type=float, default=100.0)
    parser.add_argument(
        "--profile-depth-range-mm", type=float, nargs=2, default=(60.0, 150.0)
    )

    # Moderate shared-global random baseline used by the uniform comparison.
    parser.add_argument(
        "--random-view-tilt-range-deg", type=float, nargs=2, default=(35.0, 75.0)
    )
    parser.add_argument(
        "--random-view-azimuth-range-deg", type=float, nargs=2, default=(20.0, 70.0)
    )
    parser.add_argument(
        "--random-sensor-roll-range-deg", type=float, nargs=2, default=(-90.0, 90.0)
    )

    # Plane-relative uniform design ranges.
    parser.add_argument(
        "--uniform-target-u-range-mm", type=float, nargs=2, default=(-40.0, 40.0)
    )
    parser.add_argument(
        "--uniform-target-v-range-mm", type=float, nargs=2, default=(-40.0, 40.0)
    )
    parser.add_argument(
        "--uniform-view-tilt-range-deg", type=float, nargs=2, default=(10.0, 60.0)
    )
    parser.add_argument(
        "--uniform-view-azimuth-range-deg", type=float, nargs=2, default=(-180.0, 180.0)
    )
    parser.add_argument(
        "--uniform-normal-azimuth-sensor-range-deg",
        type=float,
        nargs=2,
        default=(-180.0, 180.0),
    )
    parser.add_argument(
        "--uniform-max-batches", type=_positive_int, default=200,
        help="Maximum Latin-hypercube feasibility-resampling batches per block.",
    )
    parser.add_argument(
        "--uniform-batch-multiplier", type=_positive_int, default=8,
        help="Candidates per batch = multiplier * remaining poses.",
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


def _validate_args(
    args: argparse.Namespace,
) -> tuple[base.FairGenerationConfig, UniformConfig]:
    if args.total_scans < 9 or args.total_scans % 3 != 0:
        raise SystemExit("--total-scans must be at least 9 and divisible by 3")
    if args.profile_points < 2:
        raise SystemExit("--profile-points must be at least 2")
    if args.profile_half_width_mm <= 0.0 or args.tangent_range_mm <= 0.0:
        raise SystemExit("profile half-width and tangent range must be positive")

    depth = _finite_pair(args.profile_depth_range_mm, "profile depth range")
    if depth[0] <= 0.0 or depth[0] == depth[1]:
        raise SystemExit("profile depth range must be positive and non-zero")

    random_tilt = _finite_pair(
        args.random_view_tilt_range_deg, "random view tilt range"
    )
    uniform_tilt = _finite_pair(
        args.uniform_view_tilt_range_deg, "uniform view tilt range"
    )
    if random_tilt[0] < 0.0 or random_tilt[1] >= 89.0:
        raise SystemExit("random tilt range must lie within [0, 89)")
    if uniform_tilt[0] <= 0.0 or uniform_tilt[1] >= 89.0:
        raise SystemExit("uniform tilt range must lie within (0, 89)")

    fair = base.FairGenerationConfig(
        total_scans=int(args.total_scans),
        profile_points=int(args.profile_points),
        profile_half_width_mm=float(args.profile_half_width_mm),
        tangent_range_mm=float(args.tangent_range_mm),
        profile_depth_range_mm=depth,
        view_tilt_range_deg=random_tilt,
        view_azimuth_range_deg=_finite_pair(
            args.random_view_azimuth_range_deg, "random view azimuth range"
        ),
        sensor_roll_range_deg=_finite_pair(
            args.random_sensor_roll_range_deg, "random sensor roll range"
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
    uniform = UniformConfig(
        target_u_range_mm=_finite_pair(
            args.uniform_target_u_range_mm, "uniform target U range"
        ),
        target_v_range_mm=_finite_pair(
            args.uniform_target_v_range_mm, "uniform target V range"
        ),
        depth_range_mm=depth,
        tilt_range_deg=uniform_tilt,
        azimuth_range_deg=_finite_pair(
            args.uniform_view_azimuth_range_deg, "uniform view azimuth range"
        ),
        normal_azimuth_sensor_range_deg=_finite_pair(
            args.uniform_normal_azimuth_sensor_range_deg,
            "uniform sensor-frame plane-normal azimuth range",
        ),
        max_batches=int(args.uniform_max_batches),
        batch_multiplier=int(args.uniform_batch_multiplier),
    )
    return fair, uniform


def _scale(values: np.ndarray, bounds: tuple[float, float]) -> np.ndarray:
    lower, upper = bounds
    return lower + values * (upper - lower)


def _make_sensor_pose_relative_to_plane(
    frame: base.PlaneFrame,
    common_center: np.ndarray,
    pose: UniformRelativePose,
) -> np.ndarray:
    """Construct T_base_sensor from one plane-relative pose parameterization."""
    return sensor_pose_from_plane_relative(
        DesignPlaneFrame(frame.u, frame.v, frame.n, frame.l),
        common_center,
        DesignPlaneRelativePose(
            sample_id=pose.sample_id,
            target_u_mm=pose.target_u_mm,
            target_v_mm=pose.target_v_mm,
            distance_mm=pose.center_depth_mm,
            tilt_deg=pose.view_tilt_deg,
            azimuth_deg=pose.view_azimuth_deg,
            normal_azimuth_sensor_deg=pose.normal_azimuth_sensor_deg,
        ),
    )


def _uniform_candidate_from_row(
    row: np.ndarray,
    *,
    sample_id: int,
    block_id: int,
    index_in_block: int,
    master_seed: int,
    trial_index: int,
    config: UniformConfig,
) -> UniformRelativePose:
    return UniformRelativePose(
        sample_id=sample_id,
        block_id=block_id,
        index_in_block=index_in_block,
        target_u_mm=float(_scale(row[0], config.target_u_range_mm)),
        target_v_mm=float(_scale(row[1], config.target_v_range_mm)),
        center_depth_mm=float(_scale(row[2], config.depth_range_mm)),
        view_tilt_deg=float(_scale(row[3], config.tilt_range_deg)),
        view_azimuth_deg=float(_scale(row[4], config.azimuth_range_deg)),
        normal_azimuth_sensor_deg=float(
            _scale(row[5], config.normal_azimuth_sensor_range_deg)
        ),
        noise_seed=base._derived_seed(
            master_seed, trial_index, 0x554E4946, sample_id, 0x4E4F4953
        ),
        simulation_seed=base._derived_seed(
            master_seed, trial_index, 0x554E4946, sample_id, 0x53494D55
        ),
    )


def _sample_uniform_block_pair(
    *,
    rng: np.random.Generator,
    block_id: int,
    block_size: int,
    master_seed: int,
    trial_index: int,
    plane0: base.PlaneFrame,
    target_plane: base.PlaneFrame,
    common_center: np.ndarray,
    x_values: np.ndarray,
    fair_config: base.FairGenerationConfig,
    uniform_config: UniformConfig,
) -> tuple[
    list[UniformRelativePose],
    list[np.ndarray],
    list[np.ndarray],
    list[base.ProfileFeasibility],
    list[base.ProfileFeasibility],
    dict[str, Any],
]:
    """Generate matched relative poses feasible on plane 0 and target plane."""
    poses: list[UniformRelativePose] = []
    single_transforms: list[np.ndarray] = []
    three_transforms: list[np.ndarray] = []
    single_feasibility: list[base.ProfileFeasibility] = []
    three_feasibility: list[base.ProfileFeasibility] = []
    attempted = 0
    rejected_single = 0
    rejected_three = 0

    for batch_index in range(uniform_config.max_batches):
        if len(poses) >= block_size:
            break
        remaining = block_size - len(poses)
        batch_size = max(remaining, uniform_config.batch_multiplier * remaining)
        rows = latin_hypercube(rng, batch_size, 6)
        for row in rows:
            if len(poses) >= block_size:
                break
            attempted += 1
            index_in_block = len(poses)
            sample_id = block_id * block_size + index_in_block
            pose = _uniform_candidate_from_row(
                row,
                sample_id=sample_id,
                block_id=block_id,
                index_in_block=index_in_block,
                master_seed=master_seed,
                trial_index=trial_index,
                config=uniform_config,
            )
            try:
                T_single = _make_sensor_pose_relative_to_plane(
                    plane0, common_center, pose
                )
                T_three = _make_sensor_pose_relative_to_plane(
                    target_plane, common_center, pose
                )
            except (ValueError, FloatingPointError):
                rejected_single += 1
                continue

            feasible_single = base._profile_feasibility(
                plane=plane0,
                common_center=common_center,
                T_base_s=T_single,
                x_values=x_values,
                config=fair_config,
            )
            if feasible_single is None:
                rejected_single += 1
                continue
            feasible_three = base._profile_feasibility(
                plane=target_plane,
                common_center=common_center,
                T_base_s=T_three,
                x_values=x_values,
                config=fair_config,
            )
            if feasible_three is None:
                rejected_three += 1
                continue

            poses.append(pose)
            single_transforms.append(T_single)
            three_transforms.append(T_three)
            single_feasibility.append(feasible_single)
            three_feasibility.append(feasible_three)

    if len(poses) != block_size:
        raise RuntimeError(
            f"uniform block {block_id}: generated only {len(poses)}/{block_size} "
            f"matched feasible poses after {attempted} candidates"
        )

    stats = {
        "block_id": block_id,
        "accepted": len(poses),
        "attempted": attempted,
        "acceptance_rate": float(len(poses) / attempted),
        "rejected_by_single_feasibility": rejected_single,
        "rejected_by_three_feasibility": rejected_three,
        "matched_relative_parameter_samples": True,
    }
    return (
        poses,
        single_transforms,
        three_transforms,
        single_feasibility,
        three_feasibility,
        stats,
    )


def _as_base_sample(
    pose: UniformRelativePose,
    *,
    plane_id: int,
    scan_id: int | None = None,
    noise_seed: int | None = None,
) -> base.SharedGlobalPoseSample:
    """Adapt metadata to the existing scan simulator interface."""
    return base.SharedGlobalPoseSample(
        scan_id=pose.sample_id if scan_id is None else int(scan_id),
        three_plane_assignment=plane_id,
        center_depth_mm=pose.center_depth_mm,
        global_view_tilt_deg=pose.view_tilt_deg,
        global_view_azimuth_deg=pose.view_azimuth_deg,
        # Adapter field only; T_base_s was already built with the new
        # plane-normal azimuth convention.
        sensor_roll_deg=pose.normal_azimuth_sensor_deg,
        noise_seed=pose.noise_seed if noise_seed is None else int(noise_seed),
        simulation_seed=pose.simulation_seed,
    )


def _simulate_uniform_scan(
    *,
    T_ef_s_true: np.ndarray,
    plane: base.PlaneFrame,
    plane_id: int,
    T_base_s: np.ndarray,
    pose: UniformRelativePose,
    feasibility: base.ProfileFeasibility,
    strategy: str,
    x_values: np.ndarray,
    scan_id: int | None = None,
    noise_seed: int | None = None,
) -> LaserScan:
    scan = base._simulate_scan(
        T_ef_s_true=T_ef_s_true,
        plane=plane,
        plane_id=plane_id,
        T_base_s=T_base_s,
        sample=_as_base_sample(
            pose,
            plane_id=plane_id,
            scan_id=scan_id,
            noise_seed=noise_seed,
        ),
        feasibility=feasibility,
        strategy=strategy,
        x_values=x_values,
    )
    metadata = dict(scan.meta)
    metadata.pop("sensor_roll_deg", None)
    metadata.update(
        {
            "comparison_schema": SCHEMA,
            "comparison_schema_version": SCHEMA_VERSION,
            "pose_sampling_frame": "assigned_plane_relative",
            "pose_selection_strategy": "latin_hypercube_uniform",
            "relative_pose_block_id": pose.block_id,
            "relative_pose_index_in_block": pose.index_in_block,
            "relative_pose_source_sample_id": pose.sample_id,
            "target_u_mm": pose.target_u_mm,
            "target_v_mm": pose.target_v_mm,
            "view_pose_convention": dict(VIEW_POSE_CONVENTION),
            "relative_pose_parameters": {
                "target_u_mm": pose.target_u_mm,
                "target_v_mm": pose.target_v_mm,
                "center_depth_mm": pose.center_depth_mm,
                "view_tilt_deg": pose.view_tilt_deg,
                "view_azimuth_deg": pose.view_azimuth_deg,
                "normal_azimuth_sensor_deg": pose.normal_azimuth_sensor_deg,
                "view_u_mm": pose.target_u_mm,
                "view_v_mm": pose.target_v_mm,
                "view_distance_mm": pose.center_depth_mm,
                "view_normal_azimuth_sensor_deg": pose.normal_azimuth_sensor_deg,
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


def _relative_pose_hash(poses: Sequence[UniformRelativePose]) -> str:
    payload = json.dumps(
        [asdict(pose) for pose in poses],
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _effective_normal_metrics(
    scans: Sequence[LaserScan], frames: Sequence[base.PlaneFrame]
) -> dict[str, Any]:
    effective = []
    for scan in scans:
        normal = frames[int(scan.plane_id)].n
        effective.append(scan.T_base_ef[:3, :3].T @ normal)
    values = np.asarray(effective, dtype=float)
    values /= np.linalg.norm(values, axis=1, keepdims=True)
    centered = values - np.mean(values, axis=0, keepdims=True)
    covariance = centered.T @ centered / len(values)
    eigenvalues = np.linalg.eigvalsh(covariance)
    pairwise = base._pairwise_vector_angles_deg(values)
    return {
        "effective_normal_pairwise_median_deg": base._strict_upper_median(pairwise),
        "effective_normal_covariance_eigenvalues": eigenvalues.tolist(),
        "effective_normal_min_eigenvalue": float(eigenvalues[0]),
        "effective_normal_condition": (
            float(eigenvalues[-1] / eigenvalues[0])
            if eigenvalues[0] > 0.0
            else float("inf")
        ),
    }


def _build_uniform_datasets(
    *,
    T_ef_s_true: np.ndarray,
    frames: Sequence[base.PlaneFrame],
    common_center: np.ndarray,
    frame_angles_deg: np.ndarray,
    trial_index: int,
    fair_config: base.FairGenerationConfig,
    uniform_config: UniformConfig,
    master_seed: int,
    uniform_rng: np.random.Generator,
) -> tuple[CalibrationDataset, CalibrationDataset, dict[str, Any]]:
    x_values = np.linspace(
        -fair_config.profile_half_width_mm,
        fair_config.profile_half_width_mm,
        fair_config.profile_points,
    )
    per_plane = fair_config.total_scans // 3

    all_poses: list[UniformRelativePose] = []
    single_scans: list[LaserScan] = []
    three_scans: list[LaserScan] = []
    block_stats: list[dict[str, Any]] = []

    for plane_id in range(3):
        (
            poses,
            single_transforms,
            _three_transforms_for_block_plane,
            single_feasibility,
            _three_feasibility_for_block_plane,
            stats,
        ) = _sample_uniform_block_pair(
            rng=uniform_rng,
            block_id=plane_id,
            block_size=per_plane,
            master_seed=master_seed,
            trial_index=trial_index,
            plane0=frames[0],
            target_plane=frames[plane_id],
            common_center=common_center,
            x_values=x_values,
            fair_config=fair_config,
            uniform_config=uniform_config,
        )
        all_poses.extend(poses)
        block_stats.append(stats)

        for pose, T_base_s, feasibility in zip(
            poses, single_transforms, single_feasibility
        ):
            single_scans.append(
                _simulate_uniform_scan(
                    T_ef_s_true=T_ef_s_true,
                    plane=frames[0],
                    plane_id=0,
                    T_base_s=T_base_s,
                    pose=pose,
                    feasibility=feasibility,
                    strategy="single_plane_uniform",
                    x_values=x_values,
                )
            )
    # Apply the complete base pose set to every physical plane. Plane-major
    # storage keeps plane 0's first TOTAL_SCANS profiles aligned with the
    # single_uniform collection while assigning globally unique scan IDs.
    for plane_id, plane in enumerate(frames):
        for pose in all_poses:
            T_base_s = _make_sensor_pose_relative_to_plane(
                plane,
                common_center,
                pose,
            )
            feasibility = base._profile_feasibility(
                plane=plane,
                common_center=common_center,
                T_base_s=T_base_s,
                x_values=x_values,
                config=fair_config,
            )
            if feasibility is None:
                raise RuntimeError(
                    "a pose accepted in its source block became infeasible "
                    f"when replicated on plane {plane_id}: "
                    f"sample_id={pose.sample_id}"
                )
            replicated_scan_id = (
                plane_id * fair_config.total_scans + pose.sample_id
            )
            replicated_noise_seed = (
                pose.noise_seed
                if plane_id == 0
                else base._derived_seed(
                    master_seed,
                    trial_index,
                    0x554E5245,
                    replicated_scan_id,
                    0x4E4F4953,
                )
            )
            three_scans.append(
                _simulate_uniform_scan(
                    T_ef_s_true=T_ef_s_true,
                    plane=plane,
                    plane_id=plane_id,
                    T_base_s=T_base_s,
                    pose=pose,
                    feasibility=feasibility,
                    strategy="three_plane_uniform_full_pose_set",
                    x_values=x_values,
                    scan_id=replicated_scan_id,
                    noise_seed=replicated_noise_seed,
                )
            )

    pose_hash = _relative_pose_hash(all_poses)
    common_metadata = {
        "uniform_design": "latin_hypercube_in_assigned_plane_relative_coordinates",
        "relative_pose_parameter_hash": pose_hash,
        "relative_pose_base_set_identical_between_uniform_branches": True,
        "three_uniform_full_pose_set_applied_to_every_plane": True,
        "three_uniform_replication_factor": len(frames),
        "view_pose_convention": dict(VIEW_POSE_CONVENTION),
        "uniform_config": asdict(uniform_config),
        "uniform_block_statistics": block_stats,
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
                motion_kind="plane_relative_latin_hypercube_uniform",
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
        selection_strategy="uniform",
        selection_summary=common_metadata,
    )

    three_group_ids = [f"plane_{int(scan.plane_id)}" for scan in three_scans]
    counts = [0, 0, 0]
    sequence_indices: list[int] = []
    for scan in three_scans:
        plane_id = int(scan.plane_id)
        sequence_indices.append(counts[plane_id])
        counts[plane_id] += 1
    three_dataset = base._build_dataset(
        strategy="three_plane",
        scans=three_scans,
        scan_group_ids=three_group_ids,
        sequence_indices=sequence_indices,
        groups=[
            AcquisitionGroup(
                group_id=f"plane_{plane_id}",
                acquisition_role="calibration",
                motion_kind="plane_relative_latin_hypercube_uniform",
                plane_id=plane_id,
            )
            for plane_id in range(3)
        ],
        T_ef_s_true=T_ef_s_true,
        frames=frames,
        used_plane_ids=(0, 1, 2),
        common_center=common_center,
        frame_angles_deg=frame_angles_deg,
        trial_index=trial_index,
        config=fair_config,
        selection_strategy="uniform",
        selection_summary=common_metadata,
    )

    # Exact audit: every physical plane must receive the complete single-plane
    # relative parameter list in the same order.
    single_parameters = [
        scan.meta["relative_pose_parameters"] for scan in single_scans
    ]
    for plane_id in range(3):
        three_parameters = [
            scan.meta["relative_pose_parameters"]
            for scan in three_scans
            if int(scan.plane_id) == plane_id
        ]
        if single_parameters != three_parameters:
            raise RuntimeError(
                "three_uniform plane does not contain the complete base "
                f"parameter set: plane_id={plane_id}"
            )

    comparison = {
        "relative_pose_parameter_hash": pose_hash,
        "relative_pose_base_set_identical": True,
        "three_uniform_full_pose_set_applied_to_every_plane": True,
        "three_uniform_replication_factor": 3,
        "single_uniform_scan_count": len(single_scans),
        "three_uniform_scan_count": len(three_scans),
        "single_uniform_pose_stack_sha256": base._array_sha256(
            np.stack([scan.T_base_ef for scan in single_scans], axis=0)
        ),
        "three_uniform_pose_stack_sha256": base._array_sha256(
            np.stack([scan.T_base_ef for scan in three_scans], axis=0)
        ),
        "single_uniform_diversity": _effective_normal_metrics(single_scans, frames),
        "three_uniform_diversity": _effective_normal_metrics(three_scans, frames),
        "three_uniform_scans_per_plane": {
            str(plane_id): counts[plane_id] for plane_id in range(3)
        },
        "uniform_block_statistics": block_stats,
    }
    return single_dataset, three_dataset, comparison


def _generate_trial(
    *,
    fair_config: base.FairGenerationConfig,
    uniform_config: UniformConfig,
    master_seed: int,
    trial_index: int,
) -> tuple[dict[str, CalibrationDataset], dict[str, Any]]:
    """Generate all three branches with common truth and physical planes."""
    trial_sequence = np.random.SeedSequence([master_seed, trial_index])
    (
        handeye_sequence,
        plane_sequence,
        random_pose_sequence,
        random_assignment_sequence,
        uniform_sequence,
    ) = trial_sequence.spawn(5)

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

    # Three-random branch: same global-random mechanism as Experiment 1.
    per_plane = fair_config.total_scans // 3
    assignments = np.repeat(np.arange(3, dtype=int), per_plane)
    np.random.default_rng(random_assignment_sequence).shuffle(assignments)
    (
        random_samples,
        random_transforms,
        _single_feasibility_unused,
        random_three_feasibility,
        random_sampling_statistics,
    ) = base._sample_shared_global_pose_library(
        rng=np.random.default_rng(random_pose_sequence),
        master_seed=master_seed,
        trial_index=trial_index,
        config=fair_config,
        x_values=x_values,
        frames=frames,
        common_center=common_center,
        plane_assignments=assignments,
    )

    three_random_scans: list[LaserScan] = []
    three_random_group_ids: list[str] = []
    three_random_sequence_indices: list[int] = []
    random_counts = [0, 0, 0]
    for sample, T_base_s, feasibility in zip(
        random_samples, random_transforms, random_three_feasibility
    ):
        plane_id = int(sample.three_plane_assignment)
        three_random_scans.append(
            base._simulate_scan(
                T_ef_s_true=T_ef_s_true,
                plane=frames[plane_id],
                plane_id=plane_id,
                T_base_s=T_base_s,
                sample=sample,
                feasibility=feasibility,
                strategy="three_plane_random",
                x_values=x_values,
            )
        )
        three_random_group_ids.append(f"plane_{plane_id}")
        three_random_sequence_indices.append(random_counts[plane_id])
        random_counts[plane_id] += 1

    three_random = base._build_dataset(
        strategy="three_plane",
        scans=three_random_scans,
        scan_group_ids=three_random_group_ids,
        sequence_indices=three_random_sequence_indices,
        groups=[
            AcquisitionGroup(
                group_id=f"plane_{plane_id}",
                acquisition_role="calibration",
                motion_kind="general_6dof_global_feasible",
                plane_id=plane_id,
            )
            for plane_id in range(3)
        ],
        T_ef_s_true=T_ef_s_true,
        frames=frames,
        used_plane_ids=(0, 1, 2),
        common_center=common_center,
        frame_angles_deg=frame_angles_deg,
        trial_index=trial_index,
        config=fair_config,
        selection_strategy="random",
        selection_summary={
            "sampling_statistics": random_sampling_statistics,
            "matches_experiment_1_generator": True,
        },
    )

    single_uniform, three_uniform, uniform_comparison = _build_uniform_datasets(
        T_ef_s_true=T_ef_s_true,
        frames=frames,
        common_center=common_center,
        frame_angles_deg=frame_angles_deg,
        trial_index=trial_index,
        fair_config=fair_config,
        uniform_config=uniform_config,
        master_seed=master_seed,
        uniform_rng=np.random.default_rng(uniform_sequence),
    )

    comparison = {
        "trial_index": trial_index,
        "T_ef_s_true_sha256": base._array_sha256(T_ef_s_true),
        "shared_plane_geometry": True,
        "shared_truth_between_all_branches": True,
        "single_uniform_vs_three_random": {
            "purpose": (
                "test whether plane-relative uniform single-plane poses "
                "reach the existing random three-plane baseline"
            )
        },
        "single_uniform_vs_three_uniform": {
            "purpose": (
                "measure the benefit of applying the complete single-plane "
                "relative-pose set to every physical plane"
            ),
            "relative_pose_base_set_identical": True,
            "three_uniform_has_more_scans": True,
            "three_uniform_replication_factor": 3,
        },
        "three_random_sampling_statistics": random_sampling_statistics,
        "three_random_diversity": _effective_normal_metrics(
            three_random_scans, frames
        ),
        **uniform_comparison,
    }
    return {
        "single_uniform": single_uniform,
        "three_random": three_random,
        "three_uniform": three_uniform,
    }, comparison


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
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(
            _jsonable(payload), indent=2, sort_keys=True, allow_nan=False
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _prepare_empty_root(path: Path) -> Path:
    output = path.expanduser()
    if output.exists():
        if not output.is_dir():
            raise FileExistsError(f"output path is not a directory: {output}")
        if next(output.iterdir(), None) is not None:
            raise FileExistsError(
                f"output directory is not empty; refusing to overwrite: {output}"
            )
    else:
        output.mkdir(parents=True, exist_ok=False)
    return output


def _prepare_collection(
    root: Path,
    *,
    key: str,
    acquisition_mode: str,
    trials: int,
    master_seed: int,
    fair_config: base.FairGenerationConfig,
    uniform_config: UniformConfig,
) -> dict[str, Any]:
    (root / "trials").mkdir(parents=True, exist_ok=False)
    manifest = {
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
        "view_pose_convention": dict(VIEW_POSE_CONVENTION),
        "fair_config": asdict(fair_config),
        "uniform_config": asdict(uniform_config),
        "scans_per_trial": (
            fair_config.total_scans * 3
            if key == "three_uniform"
            else fair_config.total_scans
        ),
        "full_uniform_pose_set_applied_to_every_plane": (
            key == "three_uniform"
        ),
        "trials": [],
    }
    _write_json(root / "collection.json", manifest)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    fair_config, uniform_config = _validate_args(args)
    output_root = _prepare_empty_root(args.output_dir)

    roots = {
        "single_uniform": output_root / "single_plane_uniform",
        "three_random": output_root / "three_plane_random",
        "three_uniform": output_root / "three_plane_uniform",
    }
    acquisition_modes = {
        "single_uniform": "single_plane_global_uniform",
        "three_random": "three_plane_global_random",
        "three_uniform": "three_plane_global_uniform",
    }
    manifests: dict[str, dict[str, Any]] = {}
    for key, root in roots.items():
        root.mkdir()
        manifests[key] = _prepare_collection(
            root,
            key=key,
            acquisition_mode=acquisition_modes[key],
            trials=args.trials,
            master_seed=args.seed,
            fair_config=fair_config,
            uniform_config=uniform_config,
        )

    comparison = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "in_progress",
        "master_seed": args.seed,
        "requested_trials": args.trials,
        "completed_trials": 0,
        "collections": {key: root.name for key, root in roots.items()},
        "experiments": {
            "experiment_2": ["single_uniform", "three_random"],
            "additional": ["single_uniform", "three_uniform"],
        },
        "fair_config": asdict(fair_config),
        "uniform_config": asdict(uniform_config),
        "scan_counts": {
            "single_uniform": fair_config.total_scans,
            "three_random": fair_config.total_scans,
            "three_uniform": 3 * fair_config.total_scans,
        },
        "three_uniform_full_pose_set_applied_to_every_plane": True,
        "trials": [],
    }
    comparison_path = output_root / "comparison_manifest.json"
    _write_json(comparison_path, comparison)

    for trial_index in range(args.trials):
        datasets, trial_comparison = _generate_trial(
            fair_config=fair_config,
            uniform_config=uniform_config,
            master_seed=args.seed,
            trial_index=trial_index,
        )
        relative_path = Path("trials") / f"trial_{trial_index:06d}"
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
                }
            )
            manifests[key]["completed_trials"] += 1

        trial_comparison.update(
            {
                f"{key}_relative_path": (
                    Path(root.name) / relative_path
                ).as_posix()
                for key, root in roots.items()
            }
        )
        comparison["trials"].append(trial_comparison)
        comparison["completed_trials"] += 1

        for key, root in roots.items():
            _write_json(root / "collection.json", manifests[key])
        _write_json(comparison_path, comparison)
        print(
            f"[uniform comparison] {trial_index + 1}/{args.trials}: "
            f"base_poses={fair_config.total_scans}, "
            f"three_random={fair_config.total_scans}, "
            f"three_uniform={3 * fair_config.total_scans}"
        )

    for key, root in roots.items():
        manifests[key]["status"] = "complete"
        _write_json(root / "collection.json", manifests[key])
    comparison["status"] = "complete"
    _write_json(comparison_path, comparison)

    print(f"single uniform : {roots['single_uniform']}")
    print(f"three random   : {roots['three_random']}")
    print(f"three uniform  : {roots['three_uniform']}")
    print(f"comparison     : {comparison_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
