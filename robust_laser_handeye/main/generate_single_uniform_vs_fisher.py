#!/usr/bin/env python3
"""Generate a fair single-plane Uniform-vs-Fisher calibration comparison.

The two branches share, per Monte-Carlo trial:

* the same hand-eye ground truth and physical plane,
* one broad plane-relative candidate bank,
* the same initial bootstrap poses,
* candidate-keyed measurement noise,
* the same nominal hand-eye initialization,
* the same total number of scans.

Candidate poses are unrestricted in plane-relative coordinates except for the
simulated laser-profile feasibility conditions: finite plane intersection,
complete profile inside the configured depth ROI, and profile points inside the
finite target patch. Robot IK and collision are deliberately not checked.

Branches
--------
``single_uniform``
    Selects poses by greedy maximin spacㅇing in normalized
    ``(u, v, depth, tilt, azimuth, roll)`` parameter space. This is a purely
    geometric uniform design and does not inspect Fisher information.

``single_fisher``
    Starts from exactly the same bootstrap set and then performs estimated-state
    active greedy Fisher selection. Plane normal/offset remain nuisance
    parameters and are Schur-marginalized before D- or E-optimal scoring.

Example
-------
PYTHONPATH=. python3 main/generate_single_uniform_vs_fisher.py \
  --trials 100 \
  --seed 17 \
  --output-dir dataset/single_uniform_vs_fisher_81 \
  --total-scans 81 \
  --initial-scans 18 \
  --candidate-pool-size 2000 \
  --profile-points 100 \
  --profile-half-width-mm 25 \
  --tangent-range-mm 100 \
  --profile-depth-range-mm 60 150 \
  --candidate-center-depth-range-mm 90 120 \
  --candidate-target-u-range-mm -70 70 \
  --candidate-target-v-range-mm -70 70 \
  --candidate-view-tilt-range-deg 5 80 \
  --candidate-view-azimuth-range-deg -180 180 \
  --candidate-sensor-roll-range-deg -180 180 \
  --fisher-objective e_optimal \
  --measurement-noise-std-mm 0.20

The generated datasets already contain the configured measurement noise because
that same noisy acquisition is used by the active Fisher policy. Run the same
calibration/evaluation command on both collections without adding noise again.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from main import generate_independent_random_plane_comparison as base
from main import generate_plane_uniform_comparison as pose_geometry

from laser_handeye.active_fisher import estimate_joint_calibration
from laser_handeye.calibration_dataset import (
    CalibrationDataset,
    logical_dataset_sha256,
    save_calibration_dataset,
)
from laser_handeye.data import LaserScan
from laser_handeye.simulation import sample_random_handeye


SCHEMA = "laser_handeye.single_plane_uniform_vs_active_fisher"
SCHEMA_VERSION = 2


@dataclass(frozen=True)
class PlaneRelativeCandidate:
    candidate_id: int
    target_u_mm: float
    target_v_mm: float
    center_depth_mm: float
    view_tilt_deg: float
    view_azimuth_deg: float
    sensor_roll_deg: float
    noise_seed: int
    simulation_seed: int


@dataclass(frozen=True)
class CandidateConfig:
    target_u_range_mm: tuple[float, float]
    target_v_range_mm: tuple[float, float]
    depth_range_mm: tuple[float, float]
    tilt_range_deg: tuple[float, float]
    azimuth_range_deg: tuple[float, float]
    roll_range_deg: tuple[float, float]
    max_batches: int
    batch_multiplier: int


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
            "Generate fair single-plane plane-relative Uniform and active "
            "Fisher calibration collections."
        )
    )
    parser.add_argument("--trials", type=_positive_int, default=100)
    parser.add_argument("--seed", type=_nonnegative_int, default=17)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--total-scans", type=_positive_int, default=81)
    parser.add_argument("--initial-scans", type=_positive_int, default=18)
    parser.add_argument(
        "--candidate-pool-size", type=_positive_int, default=2000
    )

    parser.add_argument("--profile-points", type=_positive_int, default=100)
    parser.add_argument("--profile-half-width-mm", type=float, default=25.0)
    parser.add_argument("--tangent-range-mm", type=float, default=100.0)
    parser.add_argument(
        "--profile-depth-range-mm",
        type=float,
        nargs=2,
        default=(60.0, 150.0),
        help=(
            "Valid sensor-Z ROI for every point in a simulated laser profile."
        ),
    )
    parser.add_argument(
        "--candidate-center-depth-range-mm",
        type=float,
        nargs=2,
        default=(90.0, 120.0),
        help=(
            "Plane-relative center-ray depth used to generate candidate poses. "
            "This is separate from --profile-depth-range-mm."
        ),
    )

    parser.add_argument(
        "--candidate-target-u-range-mm", type=float, nargs=2,
        default=(-70.0, 70.0),
    )
    parser.add_argument(
        "--candidate-target-v-range-mm", type=float, nargs=2,
        default=(-70.0, 70.0),
    )
    parser.add_argument(
        "--candidate-view-tilt-range-deg", type=float, nargs=2,
        default=(5.0, 80.0),
    )
    parser.add_argument(
        "--candidate-view-azimuth-range-deg", type=float, nargs=2,
        default=(-180.0, 180.0),
    )
    parser.add_argument(
        "--candidate-sensor-roll-range-deg", type=float, nargs=2,
        default=(-180.0, 180.0),
    )
    parser.add_argument(
        "--candidate-max-batches", type=_positive_int, default=300
    )
    parser.add_argument(
        "--candidate-batch-multiplier", type=_positive_int, default=8
    )

    parser.add_argument(
        "--plane-angle-range-deg", type=float, nargs=2, default=(-15.0, 15.0)
    )
    parser.add_argument(
        "--plane-center-xy-range-mm", type=float, nargs=2,
        default=(-100.0, 100.0),
    )
    parser.add_argument(
        "--plane-center-z-range-mm", type=float, nargs=2,
        default=(400.0, 550.0),
    )
    parser.add_argument("--min-abs-plane-normal-z", type=float, default=1e-4)
    parser.add_argument("--verification-atol", type=float, default=1e-8)

    parser.add_argument(
        "--fisher-objective", choices=("d_optimal", "e_optimal"),
        default="e_optimal",
    )
    parser.add_argument(
        "--fisher-profile-noise-std-mm", type=float, default=0.20
    )
    parser.add_argument("--fisher-rotation-scale-deg", type=float, default=2.0)
    parser.add_argument(
        "--fisher-translation-scale-mm", type=float, default=10.0
    )
    parser.add_argument(
        "--fisher-plane-normal-scale-deg", type=float, default=20.0
    )
    parser.add_argument(
        "--fisher-plane-offset-scale-mm", type=float, default=100.0
    )

    parser.add_argument(
        "--measurement-noise-std-mm", type=float, default=0.20
    )
    parser.add_argument(
        "--measurement-noise-axis", choices=("z", "xz"), default="xz"
    )
    parser.add_argument(
        "--measurement-seed", type=_nonnegative_int, default=1701
    )
    parser.add_argument(
        "--initialization-seed", type=_nonnegative_int, default=1701
    )
    parser.add_argument(
        "--initial-translation-range-mm", type=float, default=100.0
    )
    parser.add_argument(
        "--initial-angle-range-deg", type=float, default=15.0
    )
    parser.add_argument(
        "--initial-rotation-perturbation",
        choices=("axis_angle", "euler_xyz"), default="axis_angle",
    )
    parser.add_argument(
        "--initial-translation-perturbation",
        choices=("direction_norm", "box_xyz"), default="direction_norm",
    )
    parser.add_argument(
        "--estimator-max-iterations", type=_positive_int, default=60
    )
    parser.add_argument("--estimator-tolerance", type=float, default=1e-7)
    return parser


def _validate_args(
    args: argparse.Namespace,
) -> tuple[base.FairGenerationConfig, CandidateConfig, base.PoseSelectionConfig]:
    if args.total_scans < 9:
        raise SystemExit("--total-scans must be at least 9")
    if args.initial_scans < 9:
        raise SystemExit("--initial-scans must be at least 9")
    if args.initial_scans >= args.total_scans:
        raise SystemExit("--initial-scans must be smaller than --total-scans")
    if args.candidate_pool_size < args.total_scans:
        raise SystemExit("--candidate-pool-size must be >= --total-scans")
    if args.profile_points < 2:
        raise SystemExit("--profile-points must be at least 2")
    if args.profile_half_width_mm <= 0.0 or args.tangent_range_mm <= 0.0:
        raise SystemExit("profile half-width and tangent range must be positive")

    profile_depth = _finite_pair(
        args.profile_depth_range_mm, "profile depth ROI"
    )
    center_depth = _finite_pair(
        args.candidate_center_depth_range_mm,
        "candidate center depth range",
    )
    tilt = _finite_pair(
        args.candidate_view_tilt_range_deg, "candidate tilt range"
    )
    if profile_depth[0] <= 0.0 or profile_depth[0] == profile_depth[1]:
        raise SystemExit("profile depth ROI must be positive and non-zero")
    if center_depth[0] <= 0.0 or center_depth[0] == center_depth[1]:
        raise SystemExit(
            "candidate center depth range must be positive and non-zero"
        )
    if (
        center_depth[0] < profile_depth[0]
        or center_depth[1] > profile_depth[1]
    ):
        raise SystemExit(
            "candidate center depth range must lie inside the profile depth ROI"
        )
    if tilt[0] < 0.0 or tilt[1] >= 89.0:
        raise SystemExit("candidate tilt range must lie within [0, 89)")

    # Conservative guarantee for every azimuth/roll combination:
    # |Delta z|max <= profile_half_width * tan(max_tilt).
    max_profile_depth_swing = (
        float(args.profile_half_width_mm)
        * np.tan(np.deg2rad(tilt[1]))
    )
    safe_center_min = profile_depth[0] + max_profile_depth_swing
    safe_center_max = profile_depth[1] - max_profile_depth_swing
    if safe_center_min >= safe_center_max:
        raise SystemExit(
            "No center-depth interval can contain the complete profile. "
            "Reduce profile half-width or maximum tilt, or widen the depth ROI."
        )
    if (
        center_depth[0] < safe_center_min - 1e-9
        or center_depth[1] > safe_center_max + 1e-9
    ):
        raise SystemExit(
            "candidate center depth range is not conservatively feasible for "
            "the configured profile half-width and maximum tilt. "
            f"Safe range: [{safe_center_min:.3f}, "
            f"{safe_center_max:.3f}] mm"
        )

    positive = {
        "--min-abs-plane-normal-z": args.min_abs_plane_normal_z,
        "--verification-atol": args.verification_atol,
        "--fisher-profile-noise-std-mm": args.fisher_profile_noise_std_mm,
        "--fisher-rotation-scale-deg": args.fisher_rotation_scale_deg,
        "--fisher-translation-scale-mm": args.fisher_translation_scale_mm,
        "--fisher-plane-normal-scale-deg": args.fisher_plane_normal_scale_deg,
        "--fisher-plane-offset-scale-mm": args.fisher_plane_offset_scale_mm,
        "--estimator-tolerance": args.estimator_tolerance,
    }
    for name, raw in positive.items():
        if not np.isfinite(raw) or float(raw) <= 0.0:
            raise SystemExit(f"{name} must be positive and finite")
    nonnegative = {
        "--measurement-noise-std-mm": args.measurement_noise_std_mm,
        "--initial-translation-range-mm": args.initial_translation_range_mm,
        "--initial-angle-range-deg": args.initial_angle_range_deg,
    }
    for name, raw in nonnegative.items():
        if not np.isfinite(raw) or float(raw) < 0.0:
            raise SystemExit(f"{name} must be non-negative and finite")

    fair = base.FairGenerationConfig(
        total_scans=int(args.total_scans),
        profile_points=int(args.profile_points),
        profile_half_width_mm=float(args.profile_half_width_mm),
        tangent_range_mm=float(args.tangent_range_mm),
        profile_depth_range_mm=profile_depth,
        # These global ranges are unused by this generator but remain explicit
        # for compatibility with the common dataset metadata structure.
        view_tilt_range_deg=tilt,
        view_azimuth_range_deg=_finite_pair(
            args.candidate_view_azimuth_range_deg, "candidate azimuth range"
        ),
        sensor_roll_range_deg=_finite_pair(
            args.candidate_sensor_roll_range_deg, "candidate roll range"
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
        max_local_pose_trials=max(
            int(args.candidate_pool_size * args.candidate_batch_multiplier),
            200_000,
        ),
        min_abs_plane_normal_z=float(args.min_abs_plane_normal_z),
        verification_atol=float(args.verification_atol),
        min_effective_cross_plane_angle_deg=0.0,
    )
    candidates = CandidateConfig(
        target_u_range_mm=_finite_pair(
            args.candidate_target_u_range_mm, "candidate target U range"
        ),
        target_v_range_mm=_finite_pair(
            args.candidate_target_v_range_mm, "candidate target V range"
        ),
        depth_range_mm=center_depth,
        tilt_range_deg=tilt,
        azimuth_range_deg=fair.view_azimuth_range_deg,
        roll_range_deg=fair.sensor_roll_range_deg,
        max_batches=int(args.candidate_max_batches),
        batch_multiplier=int(args.candidate_batch_multiplier),
    )
    selection = base.PoseSelectionConfig(
        initial_random_scans=int(args.initial_scans),
        candidate_pool_size=int(args.candidate_pool_size),
        fisher_profile_noise_std_mm=float(args.fisher_profile_noise_std_mm),
        rotation_scale_deg=float(args.fisher_rotation_scale_deg),
        translation_scale_mm=float(args.fisher_translation_scale_mm),
        plane_normal_scale_deg=float(args.fisher_plane_normal_scale_deg),
        plane_offset_scale_mm=float(args.fisher_plane_offset_scale_mm),
        fisher_objective=str(args.fisher_objective),
        measurement_noise_std_mm=float(args.measurement_noise_std_mm),
        measurement_noise_axis=str(args.measurement_noise_axis),
        measurement_seed=int(args.measurement_seed),
        initialization_seed=int(args.initialization_seed),
        initial_translation_range_mm=float(args.initial_translation_range_mm),
        initial_angle_range_deg=float(args.initial_angle_range_deg),
        initial_rotation_perturbation=str(args.initial_rotation_perturbation),
        initial_translation_perturbation=str(
            args.initial_translation_perturbation
        ),
        estimator_max_iterations=int(args.estimator_max_iterations),
        estimator_tolerance=float(args.estimator_tolerance),
    )
    return fair, candidates, selection


def _latin_hypercube(
    rng: np.random.Generator, n: int, dimensions: int
) -> np.ndarray:
    values = np.empty((n, dimensions), dtype=float)
    for dimension in range(dimensions):
        permutation = rng.permutation(n)
        values[:, dimension] = (permutation + rng.random(n)) / n
    return values


def _scale(value: float, bounds: tuple[float, float]) -> float:
    lower, upper = bounds
    return float(lower + value * (upper - lower))


def _candidate_from_row(
    row: np.ndarray,
    *,
    candidate_id: int,
    master_seed: int,
    trial_index: int,
    config: CandidateConfig,
) -> PlaneRelativeCandidate:
    return PlaneRelativeCandidate(
        candidate_id=int(candidate_id),
        target_u_mm=_scale(row[0], config.target_u_range_mm),
        target_v_mm=_scale(row[1], config.target_v_range_mm),
        center_depth_mm=_scale(row[2], config.depth_range_mm),
        view_tilt_deg=_scale(row[3], config.tilt_range_deg),
        view_azimuth_deg=_scale(row[4], config.azimuth_range_deg),
        sensor_roll_deg=_scale(row[5], config.roll_range_deg),
        noise_seed=base._derived_seed(
            master_seed, trial_index, 0x53494E47, candidate_id, 0x4E4F4953
        ),
        simulation_seed=base._derived_seed(
            master_seed, trial_index, 0x53494E47, candidate_id, 0x53494D55
        ),
    )


def _make_sensor_pose_relative_to_plane(
    frame: base.PlaneFrame,
    common_center: np.ndarray,
    pose: PlaneRelativeCandidate,
) -> np.ndarray:
    roll = np.deg2rad(pose.sensor_roll_deg)

    z_axis = pose_geometry._sensor_z_axis_from_view_angles(
        frame,
        pose.view_tilt_deg,
        pose.view_azimuth_deg,
        name="plane-relative canonical sensor +Z view direction",
    )

    x_reference = frame.u - float(frame.u @ z_axis) * z_axis
    if np.linalg.norm(x_reference) <= 1e-10:
        x_reference = frame.v - float(frame.v @ z_axis) * z_axis
    x_zero = base._normalize(x_reference, "plane-relative zero-roll x")
    y_zero = base._normalize(
        np.cross(z_axis, x_zero), "plane-relative zero-roll y"
    )
    x_axis = base._normalize(
        np.cos(roll) * x_zero + np.sin(roll) * y_zero,
        "plane-relative rolled x",
    )
    y_axis = base._normalize(
        -np.sin(roll) * x_zero + np.cos(roll) * y_zero,
        "plane-relative rolled y",
    )

    target = (
        np.asarray(common_center, dtype=float).reshape(3)
        + pose.target_u_mm * frame.u
        + pose.target_v_mm * frame.v
    )
    sensor_origin = target - pose.center_depth_mm * z_axis

    transform = np.eye(4, dtype=float)
    transform[:3, :3] = np.column_stack([x_axis, y_axis, z_axis])
    transform[:3, 3] = sensor_origin
    return transform


def _as_base_sample(pose: PlaneRelativeCandidate) -> base.SharedGlobalPoseSample:
    return base.SharedGlobalPoseSample(
        scan_id=pose.candidate_id,
        three_plane_assignment=0,
        center_depth_mm=pose.center_depth_mm,
        global_view_tilt_deg=pose.view_tilt_deg,
        global_view_azimuth_deg=pose.view_azimuth_deg,
        sensor_roll_deg=pose.sensor_roll_deg,
        noise_seed=pose.noise_seed,
        simulation_seed=pose.simulation_seed,
    )


def _simulate_candidate_scan(
    *,
    T_ef_s_true: np.ndarray,
    plane: base.PlaneFrame,
    T_base_s: np.ndarray,
    pose: PlaneRelativeCandidate,
    feasibility: base.ProfileFeasibility,
    x_values: np.ndarray,
) -> LaserScan:
    scan = base._simulate_scan(
        T_ef_s_true=T_ef_s_true,
        plane=plane,
        plane_id=0,
        T_base_s=T_base_s,
        sample=_as_base_sample(pose),
        feasibility=feasibility,
        strategy="single_plane_candidate",
        x_values=x_values,
    )
    metadata = dict(scan.meta)
    metadata.update(
        {
            "comparison_schema": SCHEMA,
            "comparison_schema_version": SCHEMA_VERSION,
            "pose_sampling_frame": "plane_0_relative",
            "candidate_parameterization": "u_v_depth_tilt_azimuth_roll",
            "target_u_mm": pose.target_u_mm,
            "target_v_mm": pose.target_v_mm,
            "view_pose_convention": dict(pose_geometry.VIEW_POSE_CONVENTION),
            "relative_pose_parameters": {
                "target_u_mm": pose.target_u_mm,
                "target_v_mm": pose.target_v_mm,
                "center_depth_mm": pose.center_depth_mm,
                "view_tilt_deg": pose.view_tilt_deg,
                "view_azimuth_deg": pose.view_azimuth_deg,
                "sensor_roll_deg": pose.sensor_roll_deg,
                "view_u_mm": pose.target_u_mm,
                "view_v_mm": pose.target_v_mm,
                "view_distance_mm": pose.center_depth_mm,
                "view_roll_deg": pose.sensor_roll_deg,
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


def _generate_candidate_bank(
    *,
    rng: np.random.Generator,
    master_seed: int,
    trial_index: int,
    count: int,
    frame: base.PlaneFrame,
    common_center: np.ndarray,
    T_ef_s_true: np.ndarray,
    x_values: np.ndarray,
    fair_config: base.FairGenerationConfig,
    candidate_config: CandidateConfig,
) -> tuple[
    list[PlaneRelativeCandidate],
    list[LaserScan],
    np.ndarray,
    dict[str, Any],
]:
    poses: list[PlaneRelativeCandidate] = []
    scans: list[LaserScan] = []
    normalized_rows: list[np.ndarray] = []
    attempted = 0
    rejected_geometry = 0
    rejected_roi = 0

    for _batch in range(candidate_config.max_batches):
        if len(poses) >= count:
            break
        remaining = count - len(poses)
        batch_size = max(remaining, candidate_config.batch_multiplier * remaining)
        rows = _latin_hypercube(rng, batch_size, 6)
        for row in rows:
            if len(poses) >= count:
                break
            attempted += 1
            candidate_id = len(poses)
            pose = _candidate_from_row(
                row,
                candidate_id=candidate_id,
                master_seed=master_seed,
                trial_index=trial_index,
                config=candidate_config,
            )
            try:
                T_base_s = _make_sensor_pose_relative_to_plane(
                    frame, common_center, pose
                )
            except (ValueError, FloatingPointError):
                rejected_geometry += 1
                continue
            feasibility = base._profile_feasibility(
                plane=frame,
                common_center=common_center,
                T_base_s=T_base_s,
                x_values=x_values,
                config=fair_config,
            )
            if feasibility is None:
                rejected_roi += 1
                continue
            scans.append(
                _simulate_candidate_scan(
                    T_ef_s_true=T_ef_s_true,
                    plane=frame,
                    T_base_s=T_base_s,
                    pose=pose,
                    feasibility=feasibility,
                    x_values=x_values,
                )
            )
            poses.append(pose)
            normalized_rows.append(np.asarray(row, dtype=float))

    if len(poses) != count:
        raise RuntimeError(
            f"generated only {len(poses)}/{count} feasible candidates after "
            f"{attempted} attempts; reduce the pose ranges or increase "
            "--candidate-max-batches"
        )
    statistics = {
        "attempted": attempted,
        "accepted": len(poses),
        "acceptance_rate": float(len(poses) / attempted),
        "rejected_geometry": rejected_geometry,
        "rejected_profile_roi_or_patch": rejected_roi,
        "robot_ik_and_collision_checked": False,
        "feasibility_rule": (
            "finite_plane_intersection_complete_profile_depth_roi_and_patch"
        ),
    }
    return poses, scans, np.vstack(normalized_rows), statistics


def _wrapped_parameter_embedding(normalized: np.ndarray) -> np.ndarray:
    """Embed angular dimensions continuously for maximin distance.

    Input columns are [u, v, depth, tilt, azimuth, roll] in [0, 1]. Azimuth
    and roll are periodic; replacing each by sine/cosine prevents the -180/180
    boundary from looking artificially far apart.
    """
    values = np.asarray(normalized, dtype=float)
    azimuth = 2.0 * np.pi * values[:, 4]
    roll = 2.0 * np.pi * values[:, 5]
    return np.column_stack(
        [
            values[:, 0],
            values[:, 1],
            values[:, 2],
            values[:, 3],
            np.cos(azimuth),
            np.sin(azimuth),
            np.cos(roll),
            np.sin(roll),
        ]
    )


def _maximin_order(
    normalized_parameters: np.ndarray,
    total_count: int,
    rng: np.random.Generator,
) -> list[int]:
    """Greedy farthest-point order in normalized plane-relative coordinates."""
    embedded = _wrapped_parameter_embedding(normalized_parameters)
    n = len(embedded)
    if total_count > n:
        raise ValueError("total_count exceeds candidate count")

    centroid = np.mean(embedded, axis=0)
    first = int(np.argmax(np.linalg.norm(embedded - centroid, axis=1)))
    selected = [first]
    selected_mask = np.zeros(n, dtype=bool)
    selected_mask[first] = True
    min_sq_distance = np.sum((embedded - embedded[first]) ** 2, axis=1)

    while len(selected) < total_count:
        scores = min_sq_distance.copy()
        scores[selected_mask] = -np.inf
        best_value = float(np.max(scores))
        ties = np.flatnonzero(np.isclose(scores, best_value, rtol=1e-12, atol=1e-15))
        candidate_id = int(ties[rng.integers(len(ties))])
        selected.append(candidate_id)
        selected_mask[candidate_id] = True
        new_sq_distance = np.sum(
            (embedded - embedded[candidate_id]) ** 2, axis=1
        )
        min_sq_distance = np.minimum(min_sq_distance, new_sq_distance)
    return selected


def _candidate_bank_hash(poses: Sequence[PlaneRelativeCandidate]) -> str:
    payload = json.dumps(
        [asdict(pose) for pose in poses],
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _effective_normal_metrics(
    scans: Sequence[LaserScan], plane: base.PlaneFrame
) -> dict[str, Any]:
    values = np.stack(
        [scan.T_base_ef[:3, :3].T @ plane.n for scan in scans], axis=0
    )
    values /= np.linalg.norm(values, axis=1, keepdims=True)
    centered = values - np.mean(values, axis=0, keepdims=True)
    covariance = centered.T @ centered / len(values)
    eigenvalues = np.linalg.eigvalsh(covariance)
    pairwise = base._pairwise_vector_angles_deg(values)
    return {
        "pairwise_median_deg": base._strict_upper_median(pairwise),
        "covariance_eigenvalues": eigenvalues.tolist(),
        "minimum_eigenvalue": float(eigenvalues[0]),
        "condition": (
            float(eigenvalues[-1] / eigenvalues[0])
            if eigenvalues[0] > 0.0
            else float("inf")
        ),
    }


def _uniform_selection_output(
    *,
    selected_ids: Sequence[int],
    acquire_scan,
    T_initial: np.ndarray,
    parameter_scales: np.ndarray,
    selection_config: base.PoseSelectionConfig,
) -> tuple[list[int], list[LaserScan], list[dict[str, Any]], dict[str, Any]]:
    acquired = [acquire_scan(int(candidate_id)) for candidate_id in selected_ids]
    initial_count = selection_config.initial_random_scans
    trace = [
        {
            "acquisition_step": step,
            "candidate_pool_index": int(candidate_id),
            "assigned_plane_id": 0,
            "selection_stage": (
                "common_initial_maximin"
                if step < initial_count
                else "uniform_parameter_maximin"
            ),
            "selection_objective": "plane_relative_parameter_maximin",
            "objective_gain": None,
            "objective_value_predicted_after": None,
            "objective_value_after": None,
            "linearization_source": "not_used_by_uniform_policy",
        }
        for step, candidate_id in enumerate(selected_ids)
    ]

    estimate = estimate_joint_calibration(
        {0: acquired},
        T_initial,
        parameter_scales=parameter_scales,
        profile_noise_std_mm=selection_config.fisher_profile_noise_std_mm,
        noise_axis=selection_config.measurement_noise_axis,
        max_iterations=selection_config.estimator_max_iterations,
        tolerance=selection_config.estimator_tolerance,
    )
    summary = {
        "strategy": "uniform",
        "objective": "plane_relative_parameter_maximin",
        "initial_random_scan_count": initial_count,
        "selected_candidate_ids": list(map(int, selected_ids)),
        "selected_assigned_plane_counts": {"0": len(selected_ids), "1": 0, "2": 0},
        "final_information": base._observed_information_diagnostics(
            estimate.information, parameter_scales
        ),
        "final_estimate_sha256": base._estimate_hash(estimate),
        "final_estimate": base._estimate_snapshot(estimate),
        "final_estimator_whitened_cost": estimate.whitened_cost,
        "final_estimator_iterations": estimate.iterations,
        "final_estimator_converged": estimate.converged,
        "linearization_policy": "not_used_for_pose_selection",
        "information_prior": "none",
    }
    return list(map(int, selected_ids)), acquired, trace, summary


def _generate_trial(
    *,
    fair_config: base.FairGenerationConfig,
    candidate_config: CandidateConfig,
    selection_config: base.PoseSelectionConfig,
    master_seed: int,
    trial_index: int,
) -> tuple[dict[str, CalibrationDataset], dict[str, Any]]:
    trial_sequence = np.random.SeedSequence([master_seed, trial_index])
    handeye_seq, plane_seq, candidate_seq, maximin_seq = trial_sequence.spawn(4)

    T_ef_s_true, _, _ = sample_random_handeye(np.random.default_rng(handeye_seq))
    frames, common_center, frame_angles_deg = base._make_plane_frames(
        np.random.default_rng(plane_seq),
        fair_config.plane_angle_range_deg,
        fair_config.plane_center_xy_range_mm,
        fair_config.plane_center_z_range_mm,
    )
    plane = frames[0]
    x_values = np.linspace(
        -fair_config.profile_half_width_mm,
        fair_config.profile_half_width_mm,
        fair_config.profile_points,
    )

    poses, source_scans, normalized, bank_stats = _generate_candidate_bank(
        rng=np.random.default_rng(candidate_seq),
        master_seed=master_seed,
        trial_index=trial_index,
        count=selection_config.candidate_pool_size,
        frame=plane,
        common_center=common_center,
        T_ef_s_true=T_ef_s_true,
        x_values=x_values,
        fair_config=fair_config,
        candidate_config=candidate_config,
    )
    candidate_ids = np.arange(len(source_scans), dtype=int)
    candidates = base._candidate_kinematics(
        source_scans, np.zeros(len(source_scans), dtype=int)
    )
    acquire_scan = base._make_candidate_acquirer(
        source_scans,
        measurement_seed=selection_config.measurement_seed,
        trial_index=trial_index,
        noise_std_mm=selection_config.measurement_noise_std_mm,
        noise_axis=selection_config.measurement_noise_axis,
    )
    scales = base._information_parameter_scales(1, selection_config)
    T_initial = base._selection_initial_transform(
        T_ef_s_true, trial_index=trial_index, config=selection_config
    )

    uniform_order = _maximin_order(
        normalized,
        fair_config.total_scans,
        np.random.default_rng(maximin_seq),
    )
    initial_ids = uniform_order[: selection_config.initial_random_scans]

    uniform_output = _uniform_selection_output(
        selected_ids=uniform_order,
        acquire_scan=acquire_scan,
        T_initial=T_initial,
        parameter_scales=scales,
        selection_config=selection_config,
    )
    fisher_output = base._run_online_pose_selection(
        candidates=candidates,
        acquire_scan=acquire_scan,
        initial_candidate_ids=initial_ids,
        total_scans=fair_config.total_scans,
        strategy="fisher",
        fixed_random_sequence=None,
        enforce_three_plane_balance=False,
        T_initial=T_initial,
        x_values=x_values,
        parameter_scales=scales,
        selection_config=selection_config,
        profile_depth_range_mm=fair_config.profile_depth_range_mm,
    )

    outputs = {"single_uniform": uniform_output, "single_fisher": fisher_output}
    datasets: dict[str, CalibrationDataset] = {}
    for key, (selected, acquired, trace, summary) in outputs.items():
        strategy = "uniform" if key.endswith("uniform") else "fisher"
        datasets[key] = base._build_selected_dataset(
            geometry_strategy="single_plane",
            pose_selection_strategy=strategy,
            acquired_scans=acquired,
            selected_candidate_ids=selected,
            selection_trace=trace,
            selection_summary=summary,
            T_ef_s_true=T_ef_s_true,
            frames=frames,
            common_center=common_center,
            frame_angles_deg=frame_angles_deg,
            trial_index=trial_index,
            config=fair_config,
            selection_config=selection_config,
        )

    initial_pose_stacks = {
        key: np.stack(
            [scan.T_base_ef for scan in dataset.scans[: len(initial_ids)]], axis=0
        )
        for key, dataset in datasets.items()
    }
    initial_difference = float(
        np.max(
            np.abs(
                initial_pose_stacks["single_uniform"]
                - initial_pose_stacks["single_fisher"]
            )
        )
    )
    if initial_difference > fair_config.verification_atol:
        raise RuntimeError("Uniform and Fisher branches do not share bootstrap poses")

    selected_ids = {
        key: output[0] for key, output in outputs.items()
    }
    selected_scans = {
        key: [source_scans[index] for index in ids]
        for key, ids in selected_ids.items()
    }
    uniform_info = uniform_output[3]["final_information"]
    fisher_info = fisher_output[3]["final_information"]
    comparison = {
        "trial_index": int(trial_index),
        "T_ef_s_true_sha256": base._array_sha256(T_ef_s_true),
        "candidate_bank_sha256": _candidate_bank_hash(poses),
        "candidate_pool_size": len(poses),
        "candidate_bank_statistics": bank_stats,
        "candidate_config": asdict(candidate_config),
        "candidate_bank_shared": True,
        "candidate_bank_policy_visibility": (
            "kinematics_only_until_selected; simulated future points hidden"
        ),
        "candidate_bank_feasibility_uses_truth": True,
        "candidate_bank_is_oracle_simulation_action_bank": True,
        "robot_ik_and_collision_checked": False,
        "initial_candidate_ids": list(map(int, initial_ids)),
        "initial_pose_max_abs_difference": initial_difference,
        "same_initial_bootstrap": True,
        "same_measurement_noise_for_same_candidate": True,
        "same_total_scan_count": True,
        "single_uniform_selected_candidate_ids": selected_ids["single_uniform"],
        "single_fisher_selected_candidate_ids": selected_ids["single_fisher"],
        "single_uniform_diversity": _effective_normal_metrics(
            selected_scans["single_uniform"], plane
        ),
        "single_fisher_diversity": _effective_normal_metrics(
            selected_scans["single_fisher"], plane
        ),
        "single_uniform_final_information": uniform_info,
        "single_fisher_final_information": fisher_info,
        "fisher_minus_uniform_marginal_logdet": float(
            fisher_info["handeye_marginal_logdet"]
            - uniform_info["handeye_marginal_logdet"]
        ),
        "fisher_over_uniform_min_eigenvalue_ratio": float(
            fisher_info["handeye_marginal_min_eigenvalue"]
            / uniform_info["handeye_marginal_min_eigenvalue"]
        ),
        "uniform_over_fisher_predicted_rotation_std_ratio": float(
            uniform_info["predicted_rotation_std_deg"]
            / fisher_info["predicted_rotation_std_deg"]
        ),
        "uniform_over_fisher_predicted_translation_std_ratio": float(
            uniform_info["predicted_translation_std_mm"]
            / fisher_info["predicted_translation_std_mm"]
        ),
    }
    return datasets, comparison


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
        json.dumps(_jsonable(payload), indent=2, sort_keys=True, allow_nan=False)
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
    trials: int,
    master_seed: int,
    fair_config: base.FairGenerationConfig,
    candidate_config: CandidateConfig,
    selection_config: base.PoseSelectionConfig,
) -> dict[str, Any]:
    (root / "trials").mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema": base.COLLECTION_SCHEMA,
        "schema_version": base.COLLECTION_SCHEMA_VERSION,
        "status": "in_progress",
        "generator_variant": SCHEMA,
        "generator_schema_version": SCHEMA_VERSION,
        "strategy": key,
        "acquisition_mode": key,
        "master_seed": master_seed,
        "requested_trials": trials,
        "completed_trials": 0,
        "profile_state": "measured",
        "noise_applied": selection_config.measurement_noise_std_mm > 0.0,
        "robot_ik_and_collision_checked": False,
        "view_pose_convention": dict(pose_geometry.VIEW_POSE_CONVENTION),
        "fair_config": asdict(fair_config),
        "candidate_config": asdict(candidate_config),
        "pose_selection_config": asdict(selection_config),
        "trials": [],
    }
    _write_json(root / "collection.json", manifest)
    return manifest


def _aggregate(trials: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metrics = (
        "fisher_minus_uniform_marginal_logdet",
        "fisher_over_uniform_min_eigenvalue_ratio",
        "uniform_over_fisher_predicted_rotation_std_ratio",
        "uniform_over_fisher_predicted_translation_std_ratio",
    )
    result: dict[str, Any] = {"trial_count": len(trials)}
    for metric in metrics:
        values = np.asarray([float(trial[metric]) for trial in trials])
        result[metric] = {
            "mean": float(np.mean(values)),
            "median": float(np.median(values)),
            "p05": float(np.percentile(values, 5.0)),
            "p95": float(np.percentile(values, 95.0)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
        }
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    fair_config, candidate_config, selection_config = _validate_args(args)
    output_root = _prepare_empty_root(args.output_dir)

    roots = {
        "single_uniform": output_root / "single_plane_uniform",
        "single_fisher": output_root / "single_plane_fisher",
    }
    manifests: dict[str, dict[str, Any]] = {}
    for key, root in roots.items():
        root.mkdir()
        manifests[key] = _prepare_collection(
            root,
            key=key,
            trials=args.trials,
            master_seed=args.seed,
            fair_config=fair_config,
            candidate_config=candidate_config,
            selection_config=selection_config,
        )

    comparison = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "status": "in_progress",
        "master_seed": args.seed,
        "requested_trials": args.trials,
        "completed_trials": 0,
        "collections": {key: root.name for key, root in roots.items()},
        "fair_config": asdict(fair_config),
        "candidate_config": asdict(candidate_config),
        "pose_selection_config": asdict(selection_config),
        "comparison_rules": {
            "shared_truth": True,
            "shared_physical_plane": True,
            "shared_candidate_bank": True,
            "shared_initial_bootstrap": True,
            "shared_candidate_keyed_noise": True,
            "same_total_scans": True,
            "uniform_policy": "plane_relative_parameter_maximin",
            "fisher_policy": (
                "estimated_state_active_greedy_marginal_handeye_"
                + selection_config.fisher_objective
            ),
        },
        "trials": [],
    }
    comparison_path = output_root / "comparison_manifest.json"
    _write_json(comparison_path, comparison)

    for trial_index in range(args.trials):
        datasets, trial_comparison = _generate_trial(
            fair_config=fair_config,
            candidate_config=candidate_config,
            selection_config=selection_config,
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
                    "candidate_bank_sha256": trial_comparison[
                        "candidate_bank_sha256"
                    ],
                    "initial_candidate_ids": trial_comparison[
                        "initial_candidate_ids"
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
            f"[single Uniform-vs-Fisher] {trial_index + 1}/{args.trials}: "
            f"initial={selection_config.initial_random_scans}, "
            f"total={fair_config.total_scans}, "
            f"pool={selection_config.candidate_pool_size}, "
            f"objective={selection_config.fisher_objective}, "
            "lambda-min-ratio(F/U)="
            f"{trial_comparison['fisher_over_uniform_min_eigenvalue_ratio']:.3f}"
        )

    for key, root in roots.items():
        manifests[key]["status"] = "complete"
        _write_json(root / "collection.json", manifests[key])
    comparison["summary"] = _aggregate(comparison["trials"])
    comparison["status"] = "complete"
    _write_json(comparison_path, comparison)

    print(f"single uniform : {roots['single_uniform']}")
    print(f"single Fisher  : {roots['single_fisher']}")
    print(f"comparison     : {comparison_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
