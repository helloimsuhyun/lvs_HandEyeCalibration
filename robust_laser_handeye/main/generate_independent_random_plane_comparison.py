#!/usr/bin/env python3
"""Compare random and online Fisher pose selection for plane calibration.

PYTHONPATH=. python3 main/generate_independent_random_plane_comparison.py \
  --trials 100 \
  --seed 17 \
  --output-dir dataset/fisher_vs_random_single_three_108 \
  --pose-selection-mode fisher_vs_random \
  --total-scans 108 \
  --initial-random-scans 27 \
  --candidate-pool-size 324 \
  --profile-points 100 \
  --profile-half-width-mm 25 \
  --tangent-range-mm 100 \
  --profile-depth-range-mm 60 150 \
  --view-tilt-range-deg 48 62 \
  --view-azimuth-range-deg 35 55 \
  --sensor-roll-range-deg -20 20

Every branch starts with the same balanced noisy random bootstrap.  The
bootstrap measurements are jointly optimized for the hand-eye transform and
unknown physical planes.  Each later candidate profile is predicted at that
current estimate, the configured marginal hand-eye Fisher objective is scored,
and all acquired measurements are reoptimized after the selected noisy scan is
added.  A future simulated profile is never read by the selection policy.

Candidate robot poses come from one shared simulator-generated pool and are
retained only when the complete true profile is feasible on both single-plane
plane 0 and the assigned three-plane plane.  This is an action-bank simulation
constraint, not a Fisher linearization input.  Plane parameters remain in the
joint information matrix and are Schur-marginalized before hand-eye scoring.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from laser_handeye.active_fisher import (
    JointCalibrationEstimate,
    estimate_joint_calibration,
    fisher_objective_value,
    marginal_handeye_information,
    predicted_candidate_information,
)
from laser_handeye.calibration_dataset import (
    AcquisitionGroup,
    CalibrationDataset,
    CalibrationTruth,
    PlaneTruth,
    logical_dataset_sha256,
    save_calibration_dataset,
)
from laser_handeye.data import LaserScan
from laser_handeye.initialization import make_initial_guess
from laser_handeye.se3 import make_T
from laser_handeye.simulation import (
    sample_random_handeye,
    simulate_profile_on_plane,
)


COLLECTION_SCHEMA = "laser_handeye.calibration_dataset_collection"
COLLECTION_SCHEMA_VERSION = 1
COMPARISON_SCHEMA = "laser_handeye.shared_global_feasible_pose_comparison"
COMPARISON_SCHEMA_VERSION = 4
POSE_SAMPLING_FRAME = "shared_target_global"


@dataclass(frozen=True)
class PlaneFrame:
    u: np.ndarray
    v: np.ndarray
    n: np.ndarray
    l: float


@dataclass(frozen=True)
class SharedGlobalPoseSample:
    scan_id: int
    three_plane_assignment: int
    center_depth_mm: float
    global_view_tilt_deg: float
    global_view_azimuth_deg: float
    sensor_roll_deg: float
    noise_seed: int
    simulation_seed: int


@dataclass(frozen=True)
class CandidateKinematics:
    """Policy-visible candidate data; no unacquired profile points."""

    candidate_id: int
    T_base_ef: np.ndarray
    measurement_plane_id: int
    quota_group_id: int

    def __post_init__(self) -> None:
        candidate_id = int(self.candidate_id)
        transform = np.asarray(self.T_base_ef, dtype=float).reshape(4, 4)
        if candidate_id < 0:
            raise ValueError("candidate_id must be non-negative")
        if np.any(~np.isfinite(transform)):
            raise ValueError("candidate transform must be finite")
        object.__setattr__(self, "candidate_id", candidate_id)
        object.__setattr__(self, "T_base_ef", transform.copy())
        object.__setattr__(
            self,
            "measurement_plane_id",
            int(self.measurement_plane_id),
        )
        object.__setattr__(self, "quota_group_id", int(self.quota_group_id))


@dataclass(frozen=True)
class ProfileFeasibility:
    profile_z_min_mm: float
    profile_z_max_mm: float
    profile_local_u_min_mm: float
    profile_local_u_max_mm: float
    profile_local_v_min_mm: float
    profile_local_v_max_mm: float
    ideal_plane_residual_max_abs_mm: float


@dataclass(frozen=True)
class FairGenerationConfig:
    total_scans: int
    profile_points: int
    profile_half_width_mm: float
    tangent_range_mm: float
    profile_depth_range_mm: tuple[float, float]
    view_tilt_range_deg: tuple[float, float]
    view_azimuth_range_deg: tuple[float, float]
    sensor_roll_range_deg: tuple[float, float]
    plane_angle_range_deg: tuple[float, float]
    plane_center_xy_range_mm: tuple[float, float]
    plane_center_z_range_mm: tuple[float, float]
    max_local_pose_trials: int
    min_abs_plane_normal_z: float
    verification_atol: float
    min_effective_cross_plane_angle_deg: float


@dataclass(frozen=True)
class PoseSelectionConfig:
    """Sequential design shared by the single- and three-plane comparisons."""

    initial_random_scans: int
    candidate_pool_size: int
    fisher_profile_noise_std_mm: float
    rotation_scale_deg: float
    translation_scale_mm: float
    plane_normal_scale_deg: float
    plane_offset_scale_mm: float
    fisher_objective: str = "d_optimal"
    measurement_noise_std_mm: float = 0.25
    measurement_noise_axis: str = "xz"
    measurement_seed: int = 1701
    initialization_seed: int = 1701
    initial_translation_range_mm: float = 50.0
    initial_angle_range_deg: float = 5.0
    initial_rotation_perturbation: str = "axis_angle"
    initial_translation_perturbation: str = "direction_norm"
    estimator_max_iterations: int = 60
    estimator_tolerance: float = 1e-7


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
            "Generate single/three-plane Fisher-vs-random collections from a "
            "common initial random scan set and shared global candidate pool."
        )
    )
    parser.add_argument("--trials", type=_positive_int, default=100)
    parser.add_argument("--seed", type=_nonnegative_int, default=17)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--pose-selection-mode",
        choices=("random_only", "fisher_vs_random"),
        default="random_only",
        help=(
            "random_only writes the paired noise-free single/three datasets "
            "used by the standard experiments. fisher_vs_random runs the "
            "four-branch noisy online active-selection experiment."
        ),
    )
    parser.add_argument("--total-scans", type=_positive_int, default=108)
    parser.add_argument("--profile-points", type=_positive_int, default=100)
    parser.add_argument("--profile-half-width-mm", type=float, default=25.0)
    parser.add_argument("--tangent-range-mm", type=float, default=100.0)
    parser.add_argument(
        "--profile-depth-range-mm", type=float, nargs=2, default=(60.0, 150.0)
    )
    parser.add_argument(
        "--view-tilt-range-deg", type=float, nargs=2, default=(48.0, 62.0)
    )
    parser.add_argument(
        "--view-azimuth-range-deg", type=float, nargs=2, default=(35.0, 55.0)
    )
    parser.add_argument(
        "--sensor-roll-range-deg", type=float, nargs=2, default=(-20.0, 20.0)
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
    parser.add_argument(
        "--max-local-pose-trials",
        type=_positive_int,
        default=200_000,
    )
    parser.add_argument("--min-abs-plane-normal-z", type=float, default=1e-4)
    parser.add_argument("--verification-atol", type=float, default=1e-8)
    parser.add_argument(
        "--min-effective-cross-plane-angle-deg",
        type=float,
        default=0.0,
        help=(
            "Optional dataset quality gate: reject a trial when the median "
            "calibration-effective angle between different three-plane groups "
            "is below this value. Default 0 disables rejection so Monte Carlo "
            "trials are not selected using the diversity outcome metric."
        ),
    )
    parser.add_argument(
        "--initial-random-scans",
        type=_positive_int,
        default=27,
        help=(
            "Common random bootstrap scans used by every branch before "
            "Fisher or random sequential selection. Must be at least 9 and "
            "divisible by 3."
        ),
    )
    parser.add_argument(
        "--candidate-pool-size",
        type=_positive_int,
        default=None,
        help=(
            "Number of shared feasible global poses in the discrete selection "
            "pool. Default: 3 * total-scans. Must be divisible by 3."
        ),
    )
    parser.add_argument(
        "--fisher-profile-noise-std-mm",
        type=float,
        default=0.25,
        help=(
            "Assumed per-coordinate profile noise used to whiten the "
            "point-to-plane Jacobians."
        ),
    )
    parser.add_argument(
        "--fisher-objective",
        choices=("d_optimal", "e_optimal"),
        default="d_optimal",
        help=(
            "Sequential hand-eye marginal design objective. d_optimal "
            "maximizes log-determinant; e_optimal maximizes the weakest "
            "scaled information eigenvalue."
        ),
    )
    parser.add_argument(
        "--fisher-rotation-scale-deg",
        "--fisher-prior-rotation-std-deg",
        dest="fisher_rotation_scale_deg",
        type=float,
        default=2.0,
        help=(
            "Characteristic rotation scale used to nondimensionalize the "
            "Fisher state. This is not an information prior. The legacy "
            "--fisher-prior-rotation-std-deg spelling is accepted."
        ),
    )
    parser.add_argument(
        "--fisher-translation-scale-mm",
        "--fisher-prior-translation-std-mm",
        dest="fisher_translation_scale_mm",
        type=float,
        default=10.0,
        help=(
            "Characteristic translation scale used to nondimensionalize the "
            "Fisher state. This is not an information prior. The legacy "
            "--fisher-prior-translation-std-mm spelling is accepted."
        ),
    )
    parser.add_argument(
        "--fisher-plane-normal-scale-deg",
        "--fisher-prior-plane-normal-std-deg",
        dest="fisher_plane_normal_scale_deg",
        type=float,
        default=20.0,
        help=(
            "Plane-normal tangent coordinate scale; not an information prior."
        ),
    )
    parser.add_argument(
        "--fisher-plane-offset-scale-mm",
        "--fisher-prior-plane-offset-std-mm",
        dest="fisher_plane_offset_scale_mm",
        type=float,
        default=100.0,
        help="Plane-offset coordinate scale; not an information prior.",
    )
    parser.add_argument(
        "--selection-measurement-noise-std-mm",
        type=float,
        default=0.25,
        help=(
            "Noise applied once to acquired profiles during online selection. "
            "The saved datasets contain this same noise."
        ),
    )
    parser.add_argument(
        "--selection-measurement-noise-axis",
        choices=("z", "xz"),
        default="xz",
    )
    parser.add_argument(
        "--selection-measurement-seed",
        type=_nonnegative_int,
        default=1701,
    )
    parser.add_argument(
        "--selection-initialization-seed",
        type=_nonnegative_int,
        default=1701,
        help=(
            "Seed for the provided synthetic nominal hand-eye transform. Use "
            "the same value as the downstream calibration seed."
        ),
    )
    parser.add_argument(
        "--selection-initial-translation-range-mm",
        type=float,
        default=50.0,
    )
    parser.add_argument(
        "--selection-initial-angle-range-deg",
        type=float,
        default=5.0,
    )
    parser.add_argument(
        "--selection-initial-rotation-perturbation",
        choices=("axis_angle", "euler_xyz"),
        default="axis_angle",
        help=(
            "Rotation-error sampling for the nominal transform used by "
            "online pose selection."
        ),
    )
    parser.add_argument(
        "--selection-initial-translation-perturbation",
        choices=("direction_norm", "box_xyz"),
        default="direction_norm",
        help=(
            "Translation-error sampling for the nominal transform used by "
            "online pose selection."
        ),
    )
    parser.add_argument(
        "--selection-estimator-max-iterations",
        type=_positive_int,
        default=60,
    )
    parser.add_argument(
        "--selection-estimator-tolerance",
        type=float,
        default=1e-7,
    )
    return parser


def _normalize(vector: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(vector, dtype=float).reshape(3)
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError(f"{name} is zero or non-finite")
    return value / norm


def _make_plane_frames(
    rng: np.random.Generator,
    angle_range_deg: tuple[float, float],
    center_xy_range_mm: tuple[float, float],
    center_z_range_mm: tuple[float, float],
) -> tuple[list[PlaneFrame], np.ndarray, np.ndarray]:
    angles_deg = rng.uniform(*angle_range_deg, size=3)
    frame_R = Rotation.from_euler("xyz", angles_deg, degrees=True).as_matrix()
    center = np.array(
        [
            rng.uniform(*center_xy_range_mm),
            rng.uniform(*center_xy_range_mm),
            rng.uniform(*center_z_range_mm),
        ],
        dtype=float,
    )

    raw_frames = (
        (frame_R[:, 0], frame_R[:, 1], frame_R[:, 2]),
        (frame_R[:, 1], frame_R[:, 2], frame_R[:, 0]),
        (frame_R[:, 2], frame_R[:, 0], frame_R[:, 1]),
    )

    frames: list[PlaneFrame] = []
    for plane_id, (raw_u, raw_v, raw_n) in enumerate(raw_frames):
        u = _normalize(raw_u, f"plane {plane_id} u")
        v = _normalize(raw_v, f"plane {plane_id} v")
        n = _normalize(raw_n, f"plane {plane_id} n")
        l = float(n @ center)
        if l < 0.0:
            # Flip u and n together: u x v = n stays true.
            u = -u
            n = -n
            l = -l
        if not np.allclose(np.cross(u, v), n, atol=1e-10):
            raise RuntimeError("plane frame is not right-handed")
        frames.append(PlaneFrame(u=u, v=v, n=n, l=l))

    normals = np.column_stack([frame.n for frame in frames])
    if not np.allclose(normals.T @ normals, np.eye(3), atol=1e-10):
        raise RuntimeError("plane normals are not mutually orthogonal")
    return frames, center, angles_deg


def _plane_transform(frame: PlaneFrame, common_center: np.ndarray) -> np.ndarray:
    return make_T(
        np.column_stack([frame.u, frame.v, frame.n]),
        np.asarray(common_center, dtype=float).reshape(3),
    )


def _make_sensor_pose_from_global_sample(
    target_frame: PlaneFrame,
    common_center: np.ndarray,
    sample: SharedGlobalPoseSample,
) -> np.ndarray:
    """Build one sensor pose in the common trihedral target frame.

    ``target_frame`` is plane 0's right-handed frame.  Its x/y/z axes are also
    the three unsigned trihedral normal directions before plane-equation sign
    normalization.  Around the nominal ``tilt=acos(1/sqrt(3)), azimuth=45``
    view, the sensor looks toward all three mutually orthogonal planes.
    """
    azimuth = np.deg2rad(sample.global_view_azimuth_deg)
    tilt = np.deg2rad(sample.global_view_tilt_deg)
    roll = np.deg2rad(sample.sensor_roll_deg)

    view_from_target = (
        np.sin(tilt) * np.cos(azimuth) * target_frame.u
        + np.sin(tilt) * np.sin(azimuth) * target_frame.v
        + np.cos(tilt) * target_frame.n
    )
    # Sensor +z points from the sensor origin toward the common target center.
    z_axis = -_normalize(view_from_target, "global target view direction")
    # Project the shared target x axis into the sensor image/profile tangent
    # plane.  This fixes zero roll without referring to the assigned plane.
    x_reference = target_frame.u - float(target_frame.u @ z_axis) * z_axis
    x_zero = _normalize(x_reference, "zero-roll sensor x axis")
    y_zero = _normalize(np.cross(z_axis, x_zero), "zero-roll sensor y axis")
    z_axis = _normalize(
        z_axis,
        "global sensor z axis",
    )
    x_axis = _normalize(
        np.cos(roll) * x_zero + np.sin(roll) * y_zero,
        "rolled sensor x axis",
    )
    y_axis = _normalize(
        -np.sin(roll) * x_zero + np.cos(roll) * y_zero,
        "rolled sensor y axis",
    )
    rotation = np.column_stack([x_axis, y_axis, z_axis])

    target = np.asarray(common_center, dtype=float).reshape(3)
    sensor_origin = target - sample.center_depth_mm * z_axis

    T_base_s = np.eye(4, dtype=float)
    T_base_s[:3, :3] = rotation
    T_base_s[:3, 3] = sensor_origin
    return T_base_s


def _profile_z_values(
    plane: PlaneFrame,
    T_base_s: np.ndarray,
    x_values: np.ndarray,
) -> np.ndarray:
    rotation = T_base_s[:3, :3]
    origin = T_base_s[:3, 3]
    normal_sensor = rotation.T @ plane.n
    if abs(float(normal_sensor[2])) <= 1e-10:
        raise ValueError("scan plane is nearly parallel to target plane")
    rhs = float(plane.l - plane.n @ origin)
    return (rhs - normal_sensor[0] * x_values) / normal_sensor[2]


def _profile_feasibility(
    *,
    plane: PlaneFrame,
    common_center: np.ndarray,
    T_base_s: np.ndarray,
    x_values: np.ndarray,
    config: FairGenerationConfig,
) -> ProfileFeasibility | None:
    normal_sensor = T_base_s[:3, :3].T @ plane.n
    if abs(float(normal_sensor[2])) < config.min_abs_plane_normal_z:
        return None
    try:
        z_values = _profile_z_values(plane, T_base_s, x_values)
    except (ValueError, FloatingPointError):
        return None
    if not np.all(np.isfinite(z_values)):
        return None

    depth_min, depth_max = config.profile_depth_range_mm
    z_min = float(np.min(z_values))
    z_max = float(np.max(z_values))
    if z_min < depth_min or z_max > depth_max:
        return None

    points_s = np.column_stack(
        [x_values, np.zeros_like(x_values), z_values]
    )
    points_base = (
        T_base_s[:3, :3] @ points_s.T
    ).T + T_base_s[:3, 3]
    centered = points_base - np.asarray(common_center, dtype=float).reshape(1, 3)
    local_u = centered @ plane.u
    local_v = centered @ plane.v
    if (
        float(np.max(np.abs(local_u))) > config.tangent_range_mm
        or float(np.max(np.abs(local_v))) > config.tangent_range_mm
    ):
        return None

    residual = points_base @ plane.n - plane.l
    residual_max = float(np.max(np.abs(residual)))
    if residual_max > config.verification_atol:
        return None
    return ProfileFeasibility(
        profile_z_min_mm=z_min,
        profile_z_max_mm=z_max,
        profile_local_u_min_mm=float(np.min(local_u)),
        profile_local_u_max_mm=float(np.max(local_u)),
        profile_local_v_min_mm=float(np.min(local_v)),
        profile_local_v_max_mm=float(np.max(local_v)),
        ideal_plane_residual_max_abs_mm=residual_max,
    )


def _derived_seed(
    master_seed: int,
    trial_index: int,
    method_code: int,
    scan_id: int,
    purpose: int,
) -> int:
    return int(
        np.random.SeedSequence(
            [
                int(master_seed),
                int(trial_index),
                int(method_code),
                int(scan_id),
                int(purpose),
            ]
        ).generate_state(1, dtype=np.uint64)[0]
    )


def _sample_shared_global_pose_library(
    *,
    rng: np.random.Generator,
    master_seed: int,
    trial_index: int,
    config: FairGenerationConfig,
    x_values: np.ndarray,
    frames: Sequence[PlaneFrame],
    common_center: np.ndarray,
    plane_assignments: Sequence[int],
) -> tuple[
    list[SharedGlobalPoseSample],
    list[np.ndarray],
    list[ProfileFeasibility],
    list[ProfileFeasibility],
    dict[str, Any],
]:
    """Sample poses feasible for both sides of each paired comparison."""
    assignments = [int(value) for value in plane_assignments]
    if len(assignments) != config.total_scans:
        raise ValueError(
            "plane assignment count mismatch: "
            f"expected {config.total_scans}, got {len(assignments)}"
        )
    if any(plane_id < 0 or plane_id >= len(frames) for plane_id in assignments):
        raise ValueError("plane assignment contains an invalid plane ID")

    samples: list[SharedGlobalPoseSample] = []
    transforms: list[np.ndarray] = []
    single_feasibility: list[ProfileFeasibility] = []
    three_feasibility: list[ProfileFeasibility] = []
    attempts = 0
    rejected_by_single = 0
    rejected_by_three = 0
    depth_min, depth_max = config.profile_depth_range_mm

    while (
        len(samples) < len(assignments)
        and attempts < config.max_local_pose_trials
    ):
        attempts += 1
        scan_id = len(samples)
        plane_id = assignments[scan_id]
        sample = SharedGlobalPoseSample(
            scan_id=scan_id,
            three_plane_assignment=plane_id,
            center_depth_mm=float(rng.uniform(depth_min, depth_max)),
            global_view_tilt_deg=float(rng.uniform(*config.view_tilt_range_deg)),
            global_view_azimuth_deg=float(
                rng.uniform(*config.view_azimuth_range_deg)
            ),
            sensor_roll_deg=float(rng.uniform(*config.sensor_roll_range_deg)),
            noise_seed=_derived_seed(
                master_seed, trial_index, 0x53484152, scan_id, 0x4E4F4953
            ),
            simulation_seed=_derived_seed(
                master_seed, trial_index, 0x53484152, scan_id, 0x53494D55
            ),
        )
        try:
            T_base_s = _make_sensor_pose_from_global_sample(
                frames[0], common_center, sample
            )
        except (ValueError, FloatingPointError):
            rejected_by_single += 1
            continue

        single_profile = _profile_feasibility(
            plane=frames[0],
            common_center=common_center,
            T_base_s=T_base_s,
            x_values=x_values,
            config=config,
        )
        if single_profile is None:
            rejected_by_single += 1
            continue
        three_profile = _profile_feasibility(
            plane=frames[plane_id],
            common_center=common_center,
            T_base_s=T_base_s,
            x_values=x_values,
            config=config,
        )
        if three_profile is None:
            rejected_by_three += 1
            continue

        samples.append(sample)
        transforms.append(T_base_s)
        single_feasibility.append(single_profile)
        three_feasibility.append(three_profile)

    if len(samples) != len(assignments):
        raise RuntimeError(
            f"generated only {len(samples)}/{len(assignments)} valid poses "
            f"after {attempts} attempts"
        )
    statistics = {
        "attempts": attempts,
        "accepted": len(samples),
        "acceptance_rate": float(len(samples) / attempts),
        "rejected_by_single_plane_feasibility": rejected_by_single,
        "rejected_by_three_plane_feasibility": rejected_by_three,
        "pairing_rule": (
            "same_global_pose_feasible_for_plane_0_and_assigned_plane"
        ),
    }
    return (
        samples,
        transforms,
        single_feasibility,
        three_feasibility,
        statistics,
    )


def _scan_metadata(
    *,
    strategy: str,
    sample: SharedGlobalPoseSample,
    plane_id: int,
    T_base_s: np.ndarray,
    feasibility: ProfileFeasibility,
    n_points: int,
    pose_selection_strategy: str | None = None,
    selection_stage: str | None = None,
    acquisition_step: int | None = None,
    information_gain_nats: float | None = None,
) -> dict[str, Any]:
    metadata = {
        "comparison_schema": COMPARISON_SCHEMA,
        "comparison_schema_version": COMPARISON_SCHEMA_VERSION,
        "strategy": strategy,
        "scan_id": int(sample.scan_id),
        "assigned_plane_id": int(plane_id),
        "plane_id": int(plane_id),
        "three_plane_assignment": int(sample.three_plane_assignment),
        "noise_seed": int(sample.noise_seed),
        "center_depth_mm": float(sample.center_depth_mm),
        "global_view_tilt_deg": float(sample.global_view_tilt_deg),
        "global_view_azimuth_deg": float(sample.global_view_azimuth_deg),
        "sensor_roll_deg": float(sample.sensor_roll_deg),
        "pose_sampling_frame": POSE_SAMPLING_FRAME,
        "paired_global_pose": True,
        "sensor_origin_base_mm": T_base_s[:3, 3].tolist(),
        "sensor_quaternion_xyzw": Rotation.from_matrix(
            T_base_s[:3, :3]
        ).as_quat().tolist(),
        "channel_ids": np.arange(n_points, dtype=np.int64),
        "ideal_profile": True,
    }
    if pose_selection_strategy is not None:
        metadata["pose_selection_strategy"] = pose_selection_strategy
    if selection_stage is not None:
        metadata["selection_stage"] = selection_stage
    if acquisition_step is not None:
        metadata["acquisition_step"] = int(acquisition_step)
        metadata["candidate_pool_index"] = int(sample.scan_id)
    if information_gain_nats is not None:
        metadata["information_gain_nats"] = float(information_gain_nats)
    metadata.update(asdict(feasibility))
    return metadata


def _simulate_scan(
    *,
    T_ef_s_true: np.ndarray,
    plane: PlaneFrame,
    plane_id: int,
    T_base_s: np.ndarray,
    sample: SharedGlobalPoseSample,
    feasibility: ProfileFeasibility,
    strategy: str,
    x_values: np.ndarray,
) -> LaserScan:
    T_base_ef = T_base_s @ np.linalg.inv(T_ef_s_true)
    return simulate_profile_on_plane(
        T_base_ef=T_base_ef,
        T_ef_s_true=T_ef_s_true,
        plane_n=plane.n,
        plane_l=plane.l,
        x_values=x_values,
        noise_std=0.0,
        rng=np.random.default_rng(sample.simulation_seed),
        plane_id=plane_id,
        scan_id=sample.scan_id,
        meta=_scan_metadata(
            strategy=strategy,
            sample=sample,
            plane_id=plane_id,
            T_base_s=T_base_s,
            feasibility=feasibility,
            n_points=len(x_values),
        ),
    )



def _build_dataset(
    *,
    strategy: str,
    scans: list[LaserScan],
    scan_group_ids: list[str],
    sequence_indices: list[int],
    groups: list[AcquisitionGroup],
    T_ef_s_true: np.ndarray,
    frames: Sequence[PlaneFrame],
    used_plane_ids: Sequence[int],
    common_center: np.ndarray,
    frame_angles_deg: np.ndarray,
    trial_index: int,
    config: FairGenerationConfig,
    selection_strategy: str = "random",
    selection_summary: Mapping[str, Any] | None = None,
    profile_state: str = "ideal",
    noise_applied: bool = False,
    measurement_noise: Mapping[str, Any] | None = None,
) -> CalibrationDataset:
    pose_stack = np.stack([scan.T_base_ef for scan in scans], axis=0)
    truth = CalibrationTruth(
        T_ef_s_true=T_ef_s_true,
        planes=tuple(
            PlaneTruth(
                plane_id=plane_id,
                normal_base=frames[plane_id].n,
                offset_mm=frames[plane_id].l,
                T_base_plane=_plane_transform(frames[plane_id], common_center),
            )
            for plane_id in used_plane_ids
        ),
        T_base_ef_true=pose_stack,
        T_base_ef_commanded=pose_stack,
        metadata={"comparison_trial_index": int(trial_index)},
    )
    return CalibrationDataset(
        scans=scans,
        scan_group_ids=scan_group_ids,
        sequence_indices=sequence_indices,
        groups=groups,
        acquisition_mode=f"{strategy}_global_{selection_strategy}",
        source="simulation",
        profile_state=profile_state,
        truth=truth,
        metadata={
            "generator": COMPARISON_SCHEMA,
            "comparison_schema_version": COMPARISON_SCHEMA_VERSION,
            "strategy": strategy,
            "pose_selection_strategy": selection_strategy,
            "trial_index": int(trial_index),
            "shared_target_center_base_mm": common_center.tolist(),
            "shared_target_frame_euler_xyz_deg": frame_angles_deg.tolist(),
            "pose_sampling_frame": POSE_SAMPLING_FRAME,
            "single_and_three_robot_poses_identical": (
                selection_strategy == "random"
            ),
            "shares_common_initial_random_poses_with_all_branches": (
                selection_strategy in {"random", "fisher"}
            ),
            "sensor_feasibility_checked": True,
            "robot_ik_and_collision_checked": False,
            "generator_config": asdict(config),
            "noise_applied": bool(noise_applied),
            "measurement_noise": (
                {}
                if measurement_noise is None
                else dict(measurement_noise)
            ),
            "channel_layout": "stable_profile_sample_index",
            "scan_storage_order": "scan_id_ascending",
            "selection_summary": (
                {} if selection_summary is None else dict(selection_summary)
            ),
        },
        channel_id_semantics="stable_sensor_channel",
    )


def _array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(np.asarray(array, dtype=np.float64))
    return hashlib.sha256(value.tobytes()).hexdigest()


def _pose_library_sha256(samples: Sequence[SharedGlobalPoseSample]) -> str:
    payload = json.dumps(
        [asdict(sample) for sample in samples],
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _information_parameter_scales(
    plane_count: int,
    config: PoseSelectionConfig,
) -> np.ndarray:
    """Return fixed units for the dimensionless local Fisher coordinates.

    These values define the metric used by E-optimality; they are not added to
    the observed information as a Bayesian prior.  Fixed nonsingular scaling
    changes D-optimal log-determinants only by a candidate-independent
    constant, whereas it is part of the E-optimal objective definition.
    """
    return np.asarray(
        [
            *([np.deg2rad(config.rotation_scale_deg)] * 3),
            *([config.translation_scale_mm] * 3),
            *(
                [
                    np.deg2rad(config.plane_normal_scale_deg),
                    np.deg2rad(config.plane_normal_scale_deg),
                    config.plane_offset_scale_mm,
                ]
                * plane_count
            ),
        ],
        dtype=float,
    )


def _observed_information_diagnostics(
    information: np.ndarray,
    parameter_scales: np.ndarray,
) -> dict[str, Any]:
    """Summarize acquired-data information without an artificial prior."""
    data_information = 0.5 * (information + information.T)
    data_eigenvalues = np.linalg.eigvalsh(data_information)
    rank_threshold = max(
        float(np.max(np.abs(data_eigenvalues))) * 1e-10,
        np.finfo(float).eps,
    )
    marginal = marginal_handeye_information(data_information)
    marginal_eigenvalues = np.linalg.eigvalsh(marginal)
    scaled_covariance = np.linalg.inv(marginal)
    handeye_scale = np.diag(parameter_scales[:6])
    physical_covariance = (
        handeye_scale @ scaled_covariance @ handeye_scale
    )
    return {
        "data_information_rank": int(
            np.sum(data_eigenvalues > rank_threshold)
        ),
        "joint_parameter_dimension": int(len(data_information)),
        "handeye_marginal_logdet": float(
            np.linalg.slogdet(marginal)[1]
        ),
        "handeye_marginal_min_eigenvalue": float(
            marginal_eigenvalues[0]
        ),
        "handeye_marginal_condition": float(
            marginal_eigenvalues[-1] / marginal_eigenvalues[0]
        ),
        "predicted_rotation_std_deg": float(
            np.degrees(
                np.sqrt(np.trace(physical_covariance[:3, :3]))
            )
        ),
        "predicted_translation_std_mm": float(
            np.sqrt(np.trace(physical_covariance[3:, 3:]))
        ),
    }


def _common_random_pose_sequence(
    *,
    plane_assignments: np.ndarray,
    initial_random_scans: int,
    total_scans: int,
    rng: np.random.Generator,
) -> tuple[list[int], list[int]]:
    """Return a balanced common bootstrap and balanced random continuation."""
    initial_per_plane = initial_random_scans // 3
    total_per_plane = total_scans // 3
    initial: list[int] = []
    continuation: list[int] = []
    for plane_id in range(3):
        candidates = np.flatnonzero(plane_assignments == plane_id)
        order = candidates[rng.permutation(len(candidates))]
        initial.extend(order[:initial_per_plane].tolist())
        continuation.extend(
            order[initial_per_plane:total_per_plane].tolist()
        )
    rng.shuffle(initial)
    rng.shuffle(continuation)
    return initial, initial + continuation


def _selection_initial_transform(
    T_ef_s_true: np.ndarray,
    *,
    trial_index: int,
    config: PoseSelectionConfig,
) -> np.ndarray:
    """Create the policy-visible synthetic nominal prior.

    The simulator uses truth only here to draw a controlled-error nominal
    transform.  The online policy receives only the resulting transform.  The
    seed construction matches ``main/calibrate.py`` so downstream evaluation
    can start from the exact same nominal prior.
    """
    trial_seed = np.random.SeedSequence(
        [config.initialization_seed, int(trial_index)]
    )
    _unused_noise_seed, initialization_seed = trial_seed.spawn(2)
    true_transform = np.asarray(T_ef_s_true, dtype=float).reshape(4, 4)
    true_angles_deg = Rotation.from_matrix(
        true_transform[:3, :3]
    ).as_euler("xyz", degrees=True)
    return make_initial_guess(
        reference_angles_deg=true_angles_deg,
        reference_translation_mm=true_transform[:3, 3],
        rng=np.random.default_rng(initialization_seed),
        mode="carlson",
        translation_range_mm=config.initial_translation_range_mm,
        angle_range_deg=config.initial_angle_range_deg,
        rotation_perturbation=config.initial_rotation_perturbation,
        translation_perturbation=config.initial_translation_perturbation,
    )


def _candidate_noisy_scan(
    source: LaserScan,
    *,
    measurement_seed: int,
    trial_index: int,
    candidate_id: int,
    noise_std_mm: float,
    noise_axis: str,
) -> LaserScan:
    """Materialize one candidate-keyed noisy acquisition.

    Noise is keyed by candidate rather than acquisition order, so two policies
    that select the same candidate receive byte-identical measurement noise.
    """
    points = np.asarray(source.points_s, dtype=float).copy()
    if noise_std_mm < 0.0 or not np.isfinite(noise_std_mm):
        raise ValueError("measurement noise must be non-negative and finite")
    noise_seed = _derived_seed(
        measurement_seed,
        trial_index,
        0x41435456,
        int(candidate_id),
        0x4E4F4953,
    )
    rng = np.random.default_rng(noise_seed)
    if noise_std_mm > 0.0:
        if noise_axis == "xz":
            points[:, [0, 2]] += rng.normal(
                0.0,
                noise_std_mm,
                size=(len(points), 2),
            )
        elif noise_axis == "z":
            points[:, 2] += rng.normal(
                0.0,
                noise_std_mm,
                size=len(points),
            )
        else:
            raise ValueError("measurement noise axis must be z or xz")
    metadata = dict(source.meta)
    metadata.update(
        {
            "ideal_profile": False,
            "measurement_noise_applied": noise_std_mm > 0.0,
            "measurement_noise_std_mm": float(noise_std_mm),
            "measurement_noise_axis": noise_axis,
            "measurement_noise_seed": int(noise_seed),
            "noise_key": "trial_candidate_channel",
        }
    )
    return LaserScan(
        T_base_ef=source.T_base_ef,
        points_s=points,
        plane_id=source.plane_id,
        scan_id=source.scan_id,
        meta=metadata,
    )


def _candidate_kinematics(
    scans: Sequence[LaserScan],
    quota_group_ids: Sequence[int],
) -> list[CandidateKinematics]:
    """Strip simulated profiles down to policy-visible action metadata."""
    if len(scans) != len(quota_group_ids):
        raise ValueError("scan and quota-group counts differ")
    return [
        CandidateKinematics(
            candidate_id=candidate_id,
            T_base_ef=scan.T_base_ef,
            measurement_plane_id=int(scan.plane_id),
            quota_group_id=int(quota_group_ids[candidate_id]),
        )
        for candidate_id, scan in enumerate(scans)
    ]


def _make_candidate_acquirer(
    source_scans: Sequence[LaserScan],
    *,
    measurement_seed: int,
    trial_index: int,
    noise_std_mm: float,
    noise_axis: str,
) -> Callable[[int], LaserScan]:
    """Return a lazy simulator boundary that reveals only selected scans."""

    def acquire(candidate_id: int) -> LaserScan:
        candidate_id = int(candidate_id)
        if not 0 <= candidate_id < len(source_scans):
            raise IndexError(f"candidate ID is out of range: {candidate_id}")
        return _candidate_noisy_scan(
            source_scans[candidate_id],
            measurement_seed=measurement_seed,
            trial_index=trial_index,
            candidate_id=candidate_id,
            noise_std_mm=noise_std_mm,
            noise_axis=noise_axis,
        )

    return acquire


def _group_selected_scans(
    scans: Sequence[LaserScan],
) -> dict[int, list[LaserScan]]:
    grouped: dict[int, list[LaserScan]] = {}
    for scan in scans:
        grouped.setdefault(int(scan.plane_id), []).append(scan)
    return {plane_id: grouped[plane_id] for plane_id in sorted(grouped)}


def _estimate_hash(estimate: JointCalibrationEstimate) -> str:
    digest = hashlib.sha256()
    digest.update(
        np.ascontiguousarray(
            np.asarray(estimate.T_ef_s, dtype=np.float64)
        ).tobytes()
    )
    for plane_id in sorted(estimate.planes):
        plane = estimate.planes[plane_id]
        digest.update(np.asarray([plane_id], dtype=np.int64).tobytes())
        digest.update(
            np.ascontiguousarray(
                np.asarray(plane.normal_base, dtype=np.float64)
            ).tobytes()
        )
        digest.update(
            np.asarray([plane.offset_mm], dtype=np.float64).tobytes()
        )
    return digest.hexdigest()


def _estimate_snapshot(
    estimate: JointCalibrationEstimate,
) -> dict[str, Any]:
    """Serialize a policy-visible estimate for audit and error analysis."""
    return {
        "T_ef_s": np.asarray(estimate.T_ef_s, dtype=float).tolist(),
        "planes": {
            str(plane_id): {
                "normal_base": plane.normal_base.tolist(),
                "offset_mm": float(plane.offset_mm),
            }
            for plane_id, plane in sorted(estimate.planes.items())
        },
        "whitened_cost": float(estimate.whitened_cost),
        "iterations": int(estimate.iterations),
        "converged": bool(estimate.converged),
        "data_rank": int(estimate.data_rank),
    }


def _run_online_pose_selection(
    *,
    candidates: Sequence[CandidateKinematics],
    acquire_scan: Callable[[int], LaserScan],
    initial_candidate_ids: Sequence[int],
    total_scans: int,
    strategy: str,
    fixed_random_sequence: Sequence[int] | None,
    enforce_three_plane_balance: bool,
    T_initial: np.ndarray,
    x_values: np.ndarray,
    parameter_scales: np.ndarray,
    selection_config: PoseSelectionConfig,
    profile_depth_range_mm: tuple[float, float],
) -> tuple[
    list[int],
    list[LaserScan],
    list[dict[str, Any]],
    dict[str, Any],
]:
    """Paper-style closed-loop NBV selection from acquired measurements.

    Initial measurements are jointly calibrated first.  Every subsequent
    candidate is predicted at the current estimate, an actual noisy profile is
    acquired only after selection, and the complete acquired dataset is then
    re-estimated and relinearized.
    """
    if strategy not in {"fisher", "random"}:
        raise ValueError("strategy must be 'fisher' or 'random'")
    if any(
        candidate.candidate_id != index
        for index, candidate in enumerate(candidates)
    ):
        raise ValueError(
            "candidates must be ordered by contiguous candidate_id"
        )

    selected: list[int] = []
    acquired: list[LaserScan] = []
    selected_mask = np.zeros(len(candidates), dtype=bool)
    plane_counts = np.zeros(3, dtype=int)
    trace: list[dict[str, Any]] = []

    for candidate_id in initial_candidate_ids:
        candidate_id = int(candidate_id)
        if selected_mask[candidate_id]:
            raise RuntimeError("bootstrap candidate was selected twice")
        candidate = candidates[candidate_id]
        measured_scan = acquire_scan(candidate_id)
        if (
            int(measured_scan.plane_id)
            != candidate.measurement_plane_id
            or not np.array_equal(
                measured_scan.T_base_ef,
                candidate.T_base_ef,
            )
        ):
            raise RuntimeError(
                "acquisition does not match the selected candidate"
            )
        selected.append(candidate_id)
        acquired.append(measured_scan)
        selected_mask[candidate_id] = True
        assigned_plane = candidate.quota_group_id
        plane_counts[assigned_plane] += 1
        trace.append(
            {
                "acquisition_step": len(selected) - 1,
                "candidate_pool_index": candidate_id,
                "assigned_plane_id": assigned_plane,
                "selection_stage": "initial_random",
                "selection_objective": selection_config.fisher_objective,
                "objective_gain": None,
                "objective_value_predicted_after": None,
                "objective_value_after": None,
                "linearization_source": "bootstrap_not_yet_estimated",
            }
        )

    estimate = estimate_joint_calibration(
        _group_selected_scans(acquired),
        T_initial,
        parameter_scales=parameter_scales,
        profile_noise_std_mm=(
            selection_config.fisher_profile_noise_std_mm
        ),
        noise_axis=selection_config.measurement_noise_axis,
        max_iterations=selection_config.estimator_max_iterations,
        tolerance=selection_config.estimator_tolerance,
    )
    initial_diagnostics = _observed_information_diagnostics(
        estimate.information,
        parameter_scales,
    )
    initial_objective = fisher_objective_value(
        estimate.information,
        selection_config.fisher_objective,
    )
    initial_estimate_snapshot = _estimate_snapshot(estimate)
    bootstrap_hash = _estimate_hash(estimate)
    for trace_entry in trace:
        trace_entry["bootstrap_objective_value"] = initial_objective
        trace_entry["bootstrap_estimate_sha256"] = bootstrap_hash

    plane_quota = total_scans // 3
    while len(selected) < total_scans:
        allowed = ~selected_mask
        if enforce_three_plane_balance:
            allowed &= np.asarray(
                [
                    plane_counts[candidate.quota_group_id] < plane_quota
                    for candidate in candidates
                ],
                dtype=bool,
            )
        allowed_ids = np.flatnonzero(allowed)
        if len(allowed_ids) == 0:
            raise RuntimeError(
                "no unused candidate satisfies the active policy constraints"
            )

        current_objective = fisher_objective_value(
            estimate.information,
            selection_config.fisher_objective,
        )
        predicted_objective: float | None = None
        predicted_information: np.ndarray | None = None
        if strategy == "random":
            if fixed_random_sequence is None:
                raise ValueError("random selection requires a fixed sequence")
            candidate_id = int(fixed_random_sequence[len(selected)])
            if not allowed[candidate_id]:
                raise RuntimeError(
                    "fixed random candidate violates the plane quota"
                )
            candidate_information = predicted_candidate_information(
                T_base_ef=candidates[candidate_id].T_base_ef,
                plane_id=candidates[candidate_id].measurement_plane_id,
                estimate=estimate,
                x_values=x_values,
                parameter_scales=parameter_scales,
                profile_noise_std_mm=(
                    selection_config.fisher_profile_noise_std_mm
                ),
                noise_axis=selection_config.measurement_noise_axis,
                depth_range_mm=None,
            )
            if candidate_information is not None:
                predicted_information = (
                    estimate.information + candidate_information
                )
                predicted_objective = fisher_objective_value(
                    predicted_information,
                    selection_config.fisher_objective,
                )
        else:
            best_score = -float("inf")
            candidate_id = -1
            for raw_candidate_id in allowed_ids:
                considered_id = int(raw_candidate_id)
                candidate_information = predicted_candidate_information(
                    T_base_ef=candidates[considered_id].T_base_ef,
                    plane_id=(
                        candidates[considered_id].measurement_plane_id
                    ),
                    estimate=estimate,
                    x_values=x_values,
                    parameter_scales=parameter_scales,
                    profile_noise_std_mm=(
                        selection_config.fisher_profile_noise_std_mm
                    ),
                    noise_axis=selection_config.measurement_noise_axis,
                    depth_range_mm=profile_depth_range_mm,
                )
                if candidate_information is None:
                    continue
                posterior_information = (
                    estimate.information + candidate_information
                )
                score = fisher_objective_value(
                    posterior_information,
                    selection_config.fisher_objective,
                )
                if score > best_score:
                    best_score = score
                    candidate_id = considered_id
                    predicted_information = posterior_information
            if candidate_id < 0 or predicted_information is None:
                raise RuntimeError(
                    "no candidate profile is visible under the current "
                    "estimated hand-eye and plane state"
                )
            predicted_objective = best_score

        candidate = candidates[candidate_id]
        measured_scan = acquire_scan(candidate_id)
        if (
            int(measured_scan.plane_id)
            != candidate.measurement_plane_id
            or not np.array_equal(
                measured_scan.T_base_ef,
                candidate.T_base_ef,
            )
        ):
            raise RuntimeError(
                "acquisition does not match the selected candidate"
            )
        selected.append(candidate_id)
        acquired.append(measured_scan)
        selected_mask[candidate_id] = True
        assigned_plane = candidate.quota_group_id
        plane_counts[assigned_plane] += 1
        previous_estimate_hash = _estimate_hash(estimate)

        estimate = estimate_joint_calibration(
            _group_selected_scans(acquired),
            estimate.T_ef_s,
            planes_initial=estimate.planes,
            parameter_scales=parameter_scales,
            profile_noise_std_mm=(
                selection_config.fisher_profile_noise_std_mm
            ),
            noise_axis=selection_config.measurement_noise_axis,
            max_iterations=selection_config.estimator_max_iterations,
            tolerance=selection_config.estimator_tolerance,
        )
        observed_objective = fisher_objective_value(
            estimate.information,
            selection_config.fisher_objective,
        )
        gain = (
            None
            if predicted_objective is None
            else float(predicted_objective - current_objective)
        )
        trace_entry: dict[str, Any] = {
            "acquisition_step": len(selected) - 1,
            "candidate_pool_index": candidate_id,
            "assigned_plane_id": assigned_plane,
            "selection_stage": (
                "random_baseline"
                if strategy == "random"
                else f"fisher_{selection_config.fisher_objective}"
            ),
            "selection_objective": selection_config.fisher_objective,
            "objective_gain": gain,
            "objective_value_predicted_after": predicted_objective,
            "objective_value_after": observed_objective,
            "linearization_source": "estimated_from_acquired_measurements",
            "linearization_estimate_sha256": previous_estimate_hash,
            "reestimated_state_sha256": _estimate_hash(estimate),
            "estimator_iterations": estimate.iterations,
            "estimator_converged": estimate.converged,
            "estimator_whitened_cost": estimate.whitened_cost,
            "cumulative_handeye_marginal_logdet": float(
                np.linalg.slogdet(
                    marginal_handeye_information(estimate.information)
                )[1]
            ),
            "handeye_marginal_min_eigenvalue": float(
                np.linalg.eigvalsh(
                    marginal_handeye_information(estimate.information)
                )[0]
            ),
        }
        if selection_config.fisher_objective == "d_optimal":
            trace_entry["information_gain_nats"] = gain
        else:
            trace_entry["marginal_min_eigenvalue_gain"] = gain
        trace.append(trace_entry)

    if enforce_three_plane_balance and not np.array_equal(
        plane_counts,
        np.full(3, plane_quota),
    ):
        raise RuntimeError(
            f"three-plane selection is unbalanced: {plane_counts.tolist()}"
        )
    summary = {
        "strategy": strategy,
        "objective": (
            f"estimated_state_handeye_marginal_"
            f"{selection_config.fisher_objective}"
        ),
        "initial_random_scan_count": len(initial_candidate_ids),
        "selected_candidate_ids": selected,
        "selected_assigned_plane_counts": {
            str(index): int(value)
            for index, value in enumerate(plane_counts)
        },
        "initial_information": initial_diagnostics,
        "initial_estimate": initial_estimate_snapshot,
        "final_information": _observed_information_diagnostics(
            estimate.information,
            parameter_scales,
        ),
        "final_estimate_sha256": _estimate_hash(estimate),
        "final_estimate": _estimate_snapshot(estimate),
        "final_estimator_whitened_cost": estimate.whitened_cost,
        "final_estimator_iterations": estimate.iterations,
        "final_estimator_converged": estimate.converged,
        "linearization_policy": (
            "reestimate_and_relinearize_all_acquired_measurements"
        ),
        "information_prior": "none",
    }
    return selected, acquired, trace, summary


def _pairwise_vector_angles_deg(vectors: np.ndarray) -> np.ndarray:
    values = np.asarray(vectors, dtype=float)
    values = values / np.linalg.norm(values, axis=1, keepdims=True)
    return np.degrees(
        np.arccos(np.clip(values @ values.T, -1.0, 1.0))
    )


def _pairwise_rotation_angles_deg(rotations: np.ndarray) -> np.ndarray:
    values = np.asarray(rotations, dtype=float)
    relative_trace = np.einsum("aij,bij->ab", values, values)
    return np.degrees(
        np.arccos(np.clip((relative_trace - 1.0) / 2.0, -1.0, 1.0))
    )


def _strict_upper_median(matrix: np.ndarray) -> float:
    indices = np.triu_indices(len(matrix), 1)
    if len(indices[0]) == 0:
        return 0.0
    return float(np.median(matrix[indices]))


def _centered_effective_normal_spectrum(
    effective_normals: np.ndarray,
    plane_ids: np.ndarray,
) -> tuple[np.ndarray, float]:
    centered = np.empty_like(effective_normals)
    for plane_id in np.unique(plane_ids):
        mask = plane_ids == plane_id
        centered[mask] = (
            effective_normals[mask]
            - np.mean(effective_normals[mask], axis=0, keepdims=True)
        )
    eigenvalues = np.linalg.eigvalsh(
        centered.T @ centered / len(centered)
    )
    condition = (
        float(eigenvalues[-1] / eigenvalues[0])
        if eigenvalues[0] > 0.0
        else float("inf")
    )
    return eigenvalues, condition


def _diversity_metrics(
    *,
    frames: Sequence[PlaneFrame],
    T_base_ef_stack: np.ndarray,
    three_plane_ids: np.ndarray,
) -> dict[str, Any]:
    rotations = np.asarray(T_base_ef_stack, dtype=float)[:, :3, :3]
    single_ids = np.zeros(len(rotations), dtype=int)
    single_normals = np.repeat(frames[0].n[None, :], len(rotations), axis=0)
    three_normals = np.stack(
        [frames[int(plane_id)].n for plane_id in three_plane_ids],
        axis=0,
    )
    single_effective = np.einsum(
        "nji,nj->ni", rotations, single_normals
    )
    three_effective = np.einsum(
        "nji,nj->ni", rotations, three_normals
    )
    single_angles = _pairwise_vector_angles_deg(single_effective)
    three_angles = _pairwise_vector_angles_deg(three_effective)
    cross_mask = np.triu(
        three_plane_ids[:, None] != three_plane_ids[None, :],
        1,
    )
    cross_angles = three_angles[cross_mask]
    if len(cross_angles) == 0:
        raise RuntimeError("three-plane diversity requires multiple plane IDs")

    single_eigenvalues, single_condition = (
        _centered_effective_normal_spectrum(single_effective, single_ids)
    )
    three_eigenvalues, three_condition = (
        _centered_effective_normal_spectrum(
            three_effective,
            three_plane_ids,
        )
    )
    base_angles = _pairwise_vector_angles_deg(
        np.stack([frame.n for frame in frames], axis=0)
    )
    base_upper = base_angles[np.triu_indices(len(frames), 1)]
    pose_angles = _pairwise_rotation_angles_deg(rotations)
    return {
        "base_plane_normal_pairwise_min_deg": float(np.min(base_upper)),
        "base_plane_normal_pairwise_median_deg": float(np.median(base_upper)),
        "robot_pose_pairwise_rotation_median_deg": _strict_upper_median(
            pose_angles
        ),
        "single_effective_normal_pairwise_median_deg": (
            _strict_upper_median(single_angles)
        ),
        "three_effective_normal_pairwise_median_deg": (
            _strict_upper_median(three_angles)
        ),
        "three_cross_plane_effective_normal_median_deg": float(
            np.median(cross_angles)
        ),
        "single_centered_effective_normal_eigenvalues": (
            single_eigenvalues.tolist()
        ),
        "three_centered_effective_normal_eigenvalues": (
            three_eigenvalues.tolist()
        ),
        "single_centered_effective_normal_condition": single_condition,
        "three_centered_effective_normal_condition": three_condition,
        "centered_effective_normal_condition_improvement_ratio": float(
            single_condition / three_condition
        ),
    }


def _generate_shared_global_trial(
    *, config: FairGenerationConfig, master_seed: int, trial_index: int
) -> tuple[CalibrationDataset, CalibrationDataset, dict[str, Any]]:
    trial_sequence = np.random.SeedSequence([master_seed, trial_index])
    (
        handeye_sequence,
        plane_sequence,
        shared_pose_sequence,
        three_assignment_sequence,
    ) = trial_sequence.spawn(4)

    T_ef_s_true, _, _ = sample_random_handeye(
        np.random.default_rng(handeye_sequence)
    )
    frames, common_center, frame_angles_deg = _make_plane_frames(
        np.random.default_rng(plane_sequence),
        config.plane_angle_range_deg,
        config.plane_center_xy_range_mm,
        config.plane_center_z_range_mm,
    )
    x_values = np.linspace(
        -config.profile_half_width_mm,
        config.profile_half_width_mm,
        config.profile_points,
    )

    per_plane = config.total_scans // 3
    three_assignments = np.repeat(np.arange(3, dtype=int), per_plane)
    np.random.default_rng(three_assignment_sequence).shuffle(three_assignments)

    (
        shared_samples,
        T_base_s_values,
        single_profiles,
        three_profiles,
        sampling_statistics,
    ) = _sample_shared_global_pose_library(
        rng=np.random.default_rng(shared_pose_sequence),
        master_seed=master_seed,
        trial_index=trial_index,
        config=config,
        x_values=x_values,
        frames=frames,
        common_center=common_center,
        plane_assignments=three_assignments,
    )

    single_scans: list[LaserScan] = []
    for sample, T_base_s, feasibility in zip(
        shared_samples,
        T_base_s_values,
        single_profiles,
    ):
        single_scans.append(
            _simulate_scan(
                T_ef_s_true=T_ef_s_true,
                plane=frames[0],
                plane_id=0,
                T_base_s=T_base_s,
                sample=sample,
                feasibility=feasibility,
                strategy="single_plane",
                x_values=x_values,
            )
        )

    three_scans: list[LaserScan] = []
    three_group_ids: list[str] = []
    three_sequence_indices: list[int] = []
    counts = [0, 0, 0]
    for sample, T_base_s, feasibility in zip(
        shared_samples,
        T_base_s_values,
        three_profiles,
    ):
        plane_id = sample.three_plane_assignment
        three_scans.append(
            _simulate_scan(
                T_ef_s_true=T_ef_s_true,
                plane=frames[plane_id],
                plane_id=plane_id,
                T_base_s=T_base_s,
                sample=sample,
                feasibility=feasibility,
                strategy="three_plane",
                x_values=x_values,
            )
        )
        three_group_ids.append(f"plane_{plane_id}")
        three_sequence_indices.append(counts[plane_id])
        counts[plane_id] += 1

    if counts != [per_plane, per_plane, per_plane]:
        raise RuntimeError(f"unbalanced three-plane assignment: {counts}")

    single_dataset = _build_dataset(
        strategy="single_plane",
        scans=single_scans,
        scan_group_ids=["plane_0"] * len(single_scans),
        sequence_indices=list(range(len(single_scans))),
        groups=[
            AcquisitionGroup(
                group_id="plane_0",
                acquisition_role="calibration",
                motion_kind="general_6dof_global_feasible",
                plane_id=0,
            )
        ],
        T_ef_s_true=T_ef_s_true,
        frames=frames,
        used_plane_ids=(0,),
        common_center=common_center,
        frame_angles_deg=frame_angles_deg,
        trial_index=trial_index,
        config=config,
    )
    three_dataset = _build_dataset(
        strategy="three_plane",
        scans=three_scans,
        scan_group_ids=three_group_ids,
        sequence_indices=three_sequence_indices,
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
        config=config,
    )

    single_pose_stack = np.stack(
        [scan.T_base_ef for scan in single_scans],
        axis=0,
    )
    three_pose_stack = np.stack(
        [scan.T_base_ef for scan in three_scans],
        axis=0,
    )
    max_pose_difference = float(
        np.max(np.abs(single_pose_stack - three_pose_stack))
    )
    if max_pose_difference > config.verification_atol:
        raise RuntimeError(
            "paired single/three robot poses differ: "
            f"max_abs={max_pose_difference:.6g}"
        )
    diversity = _diversity_metrics(
        frames=frames,
        T_base_ef_stack=single_pose_stack,
        three_plane_ids=np.asarray(three_assignments, dtype=int),
    )
    if (
        diversity["three_cross_plane_effective_normal_median_deg"]
        < config.min_effective_cross_plane_angle_deg
    ):
        raise RuntimeError(
            "three-plane calibration-effective normal diversity is too low: "
            f"{diversity['three_cross_plane_effective_normal_median_deg']:.6g} "
            f"< {config.min_effective_cross_plane_angle_deg:.6g} deg"
        )

    shared_pose_hash = _pose_library_sha256(shared_samples)
    comparison = {
        "trial_index": int(trial_index),
        "T_ef_s_true_sha256": _array_sha256(T_ef_s_true),
        "single_pose_library_sha256": shared_pose_hash,
        "three_pose_library_sha256": shared_pose_hash,
        "shared_pose_library_sha256": shared_pose_hash,
        "single_robot_pose_stack_sha256": _array_sha256(single_pose_stack),
        "three_robot_pose_stack_sha256": _array_sha256(three_pose_stack),
        "pose_libraries_are_independent": False,
        "pose_libraries_are_shared": True,
        "single_and_three_robot_poses_identical": True,
        "paired_robot_pose_max_abs_difference": max_pose_difference,
        "pose_sampling_frame": POSE_SAMPLING_FRAME,
        "sensor_feasibility_rule": (
            "complete_profile_inside_depth_and_finite_patch_for_"
            "single_plane_0_and_assigned_three_plane"
        ),
        "sampling_statistics": sampling_statistics,
        "diversity_metrics": diversity,
        "single_scan_count": len(single_scans),
        "three_scan_count": len(three_scans),
        "three_scans_per_plane": {str(i): counts[i] for i in range(3)},
    }
    return single_dataset, three_dataset, comparison


def _selected_scan_copy(
    source: LaserScan,
    *,
    geometry_strategy: str,
    pose_selection_strategy: str,
    trace_entry: Mapping[str, Any],
) -> LaserScan:
    metadata = dict(source.meta)
    source_scan_id = metadata.get("scan_id", source.scan_id)
    if source_scan_id is not None:
        metadata["source_candidate_scan_id"] = int(source_scan_id)
    if "noise_seed" in metadata:
        metadata["source_candidate_noise_seed"] = int(
            metadata.pop("noise_seed")
        )
    metadata.update(
        {
            "strategy": geometry_strategy,
            "pose_selection_strategy": pose_selection_strategy,
            "selection_stage": str(trace_entry["selection_stage"]),
            "acquisition_step": int(trace_entry["acquisition_step"]),
            "scan_id": int(trace_entry["acquisition_step"]),
            "candidate_pool_index": int(
                trace_entry["candidate_pool_index"]
            ),
            "selection_objective": str(
                trace_entry["selection_objective"]
            ),
        }
    )
    for optional_numeric_field in (
        "objective_gain",
        "objective_value_predicted_after",
        "objective_value_after",
        "bootstrap_objective_value",
        "cumulative_handeye_marginal_logdet",
        "handeye_marginal_min_eigenvalue",
        "estimator_iterations",
        "estimator_whitened_cost",
    ):
        value = trace_entry.get(optional_numeric_field)
        if value is not None:
            metadata[optional_numeric_field] = float(value)
    for optional_text_field in (
        "linearization_source",
        "linearization_estimate_sha256",
        "reestimated_state_sha256",
        "bootstrap_estimate_sha256",
    ):
        if optional_text_field in trace_entry:
            metadata[optional_text_field] = str(
                trace_entry[optional_text_field]
            )
    if "estimator_converged" in trace_entry:
        metadata["estimator_converged"] = bool(
            trace_entry["estimator_converged"]
        )
    for optional_field in (
        "information_gain_nats",
        "marginal_min_eigenvalue_gain",
    ):
        value = trace_entry.get(optional_field)
        if value is not None:
            metadata[optional_field] = float(value)
    return LaserScan(
        T_base_ef=source.T_base_ef,
        points_s=source.points_s,
        plane_id=source.plane_id,
        scan_id=int(trace_entry["acquisition_step"]),
        meta=metadata,
    )


def _build_selected_dataset(
    *,
    geometry_strategy: str,
    pose_selection_strategy: str,
    acquired_scans: Sequence[LaserScan],
    selected_candidate_ids: Sequence[int],
    selection_trace: Sequence[Mapping[str, Any]],
    selection_summary: Mapping[str, Any],
    T_ef_s_true: np.ndarray,
    frames: Sequence[PlaneFrame],
    common_center: np.ndarray,
    frame_angles_deg: np.ndarray,
    trial_index: int,
    config: FairGenerationConfig,
    selection_config: PoseSelectionConfig,
) -> CalibrationDataset:
    if not (
        len(selected_candidate_ids)
        == len(acquired_scans)
        == len(selection_trace)
    ):
        raise ValueError(
            "selection IDs, acquired scans, and trace lengths differ"
        )
    for candidate_id, trace_entry in zip(
        selected_candidate_ids,
        selection_trace,
    ):
        if int(candidate_id) != int(trace_entry["candidate_pool_index"]):
            raise ValueError("selected candidate ID and trace disagree")
    scans = [
        _selected_scan_copy(
            acquired_scan,
            geometry_strategy=geometry_strategy,
            pose_selection_strategy=pose_selection_strategy,
            trace_entry=trace_entry,
        )
        for candidate_id, acquired_scan, trace_entry in zip(
            selected_candidate_ids,
            acquired_scans,
            selection_trace,
        )
    ]
    if geometry_strategy == "single_plane":
        group_ids = ["plane_0"] * len(scans)
        sequence_indices = list(range(len(scans)))
        groups = [
            AcquisitionGroup(
                group_id="plane_0",
                acquisition_role="calibration",
                motion_kind=f"sequential_{pose_selection_strategy}",
                plane_id=0,
            )
        ]
        used_plane_ids: Sequence[int] = (0,)
    else:
        counts = [0, 0, 0]
        group_ids = []
        sequence_indices = []
        for scan in scans:
            plane_id = int(scan.plane_id)
            group_ids.append(f"plane_{plane_id}")
            sequence_indices.append(counts[plane_id])
            counts[plane_id] += 1
        groups = [
            AcquisitionGroup(
                group_id=f"plane_{plane_id}",
                acquisition_role="calibration",
                motion_kind=f"sequential_{pose_selection_strategy}",
                plane_id=plane_id,
            )
            for plane_id in range(3)
        ]
        used_plane_ids = (0, 1, 2)

    return _build_dataset(
        strategy=geometry_strategy,
        scans=scans,
        scan_group_ids=group_ids,
        sequence_indices=sequence_indices,
        groups=groups,
        T_ef_s_true=T_ef_s_true,
        frames=frames,
        used_plane_ids=used_plane_ids,
        common_center=common_center,
        frame_angles_deg=frame_angles_deg,
        trial_index=trial_index,
        config=config,
        selection_strategy=pose_selection_strategy,
        selection_summary=selection_summary,
        profile_state="measured",
        noise_applied=selection_config.measurement_noise_std_mm > 0.0,
        measurement_noise={
            "std_mm": selection_config.measurement_noise_std_mm,
            "axis": selection_config.measurement_noise_axis,
            "seed": selection_config.measurement_seed,
            "keying": "trial_candidate_channel",
            "applied_during_active_acquisition": True,
        },
    )


def _selection_improvement(
    fisher_summary: Mapping[str, Any],
    random_summary: Mapping[str, Any],
) -> dict[str, float]:
    fisher = fisher_summary["final_information"]
    random = random_summary["final_information"]
    return {
        "fisher_minus_random_marginal_logdet": float(
            fisher["handeye_marginal_logdet"]
            - random["handeye_marginal_logdet"]
        ),
        "fisher_minus_random_marginal_min_eigenvalue": float(
            fisher["handeye_marginal_min_eigenvalue"]
            - random["handeye_marginal_min_eigenvalue"]
        ),
        "fisher_over_random_marginal_min_eigenvalue_ratio": float(
            fisher["handeye_marginal_min_eigenvalue"]
            / random["handeye_marginal_min_eigenvalue"]
        ),
        "random_over_fisher_predicted_rotation_std_ratio": float(
            random["predicted_rotation_std_deg"]
            / fisher["predicted_rotation_std_deg"]
        ),
        "random_over_fisher_predicted_translation_std_ratio": float(
            random["predicted_translation_std_mm"]
            / fisher["predicted_translation_std_mm"]
        ),
    }


def _generate_fisher_random_trial(
    *,
    config: FairGenerationConfig,
    selection_config: PoseSelectionConfig,
    master_seed: int,
    trial_index: int,
) -> tuple[dict[str, CalibrationDataset], dict[str, Any]]:
    """Generate four paired branches from one common feasible pose pool."""
    trial_sequence = np.random.SeedSequence([master_seed, trial_index])
    (
        handeye_sequence,
        plane_sequence,
        shared_pose_sequence,
        three_assignment_sequence,
        selection_sequence,
    ) = trial_sequence.spawn(5)

    T_ef_s_true, _, _ = sample_random_handeye(
        np.random.default_rng(handeye_sequence)
    )
    frames, common_center, frame_angles_deg = _make_plane_frames(
        np.random.default_rng(plane_sequence),
        config.plane_angle_range_deg,
        config.plane_center_xy_range_mm,
        config.plane_center_z_range_mm,
    )
    x_values = np.linspace(
        -config.profile_half_width_mm,
        config.profile_half_width_mm,
        config.profile_points,
    )

    pool_per_plane = selection_config.candidate_pool_size // 3
    plane_assignments = np.repeat(
        np.arange(3, dtype=int),
        pool_per_plane,
    )
    np.random.default_rng(three_assignment_sequence).shuffle(
        plane_assignments
    )
    pool_generation_config = replace(
        config,
        total_scans=selection_config.candidate_pool_size,
    )
    (
        shared_samples,
        T_base_s_values,
        single_profiles,
        three_profiles,
        sampling_statistics,
    ) = _sample_shared_global_pose_library(
        rng=np.random.default_rng(shared_pose_sequence),
        master_seed=master_seed,
        trial_index=trial_index,
        config=pool_generation_config,
        x_values=x_values,
        frames=frames,
        common_center=common_center,
        plane_assignments=plane_assignments,
    )

    single_pool: list[LaserScan] = []
    three_pool: list[LaserScan] = []
    for sample, T_base_s, single_feasibility, three_feasibility in zip(
        shared_samples,
        T_base_s_values,
        single_profiles,
        three_profiles,
    ):
        single_pool.append(
            _simulate_scan(
                T_ef_s_true=T_ef_s_true,
                plane=frames[0],
                plane_id=0,
                T_base_s=T_base_s,
                sample=sample,
                feasibility=single_feasibility,
                strategy="single_plane",
                x_values=x_values,
            )
        )
        assigned_plane = int(sample.three_plane_assignment)
        three_pool.append(
            _simulate_scan(
                T_ef_s_true=T_ef_s_true,
                plane=frames[assigned_plane],
                plane_id=assigned_plane,
                T_base_s=T_base_s,
                sample=sample,
                feasibility=three_feasibility,
                strategy="three_plane",
                x_values=x_values,
            )
        )

    single_candidates = _candidate_kinematics(
        single_pool,
        plane_assignments,
    )
    three_candidates = _candidate_kinematics(
        three_pool,
        plane_assignments,
    )
    single_acquire = _make_candidate_acquirer(
        single_pool,
        measurement_seed=selection_config.measurement_seed,
        trial_index=trial_index,
        noise_std_mm=selection_config.measurement_noise_std_mm,
        noise_axis=selection_config.measurement_noise_axis,
    )
    three_acquire = _make_candidate_acquirer(
        three_pool,
        measurement_seed=selection_config.measurement_seed,
        trial_index=trial_index,
        noise_std_mm=selection_config.measurement_noise_std_mm,
        noise_axis=selection_config.measurement_noise_axis,
    )
    single_scales = _information_parameter_scales(1, selection_config)
    three_scales = _information_parameter_scales(3, selection_config)
    T_selection_initial = _selection_initial_transform(
        T_ef_s_true,
        trial_index=trial_index,
        config=selection_config,
    )

    initial_ids, common_random_ids = _common_random_pose_sequence(
        plane_assignments=plane_assignments,
        initial_random_scans=selection_config.initial_random_scans,
        total_scans=config.total_scans,
        rng=np.random.default_rng(selection_sequence),
    )
    selection_outputs: dict[
        str,
        tuple[
            list[int],
            list[LaserScan],
            list[dict[str, Any]],
            dict[str, Any],
        ],
    ] = {}
    selection_outputs["single_random"] = _run_online_pose_selection(
        candidates=single_candidates,
        acquire_scan=single_acquire,
        parameter_scales=single_scales,
        initial_candidate_ids=initial_ids,
        total_scans=config.total_scans,
        strategy="random",
        fixed_random_sequence=common_random_ids,
        enforce_three_plane_balance=False,
        T_initial=T_selection_initial,
        x_values=x_values,
        selection_config=selection_config,
        profile_depth_range_mm=config.profile_depth_range_mm,
    )
    selection_outputs["single_fisher"] = _run_online_pose_selection(
        candidates=single_candidates,
        acquire_scan=single_acquire,
        parameter_scales=single_scales,
        initial_candidate_ids=initial_ids,
        total_scans=config.total_scans,
        strategy="fisher",
        fixed_random_sequence=None,
        enforce_three_plane_balance=False,
        T_initial=T_selection_initial,
        x_values=x_values,
        selection_config=selection_config,
        profile_depth_range_mm=config.profile_depth_range_mm,
    )
    selection_outputs["three_random"] = _run_online_pose_selection(
        candidates=three_candidates,
        acquire_scan=three_acquire,
        parameter_scales=three_scales,
        initial_candidate_ids=initial_ids,
        total_scans=config.total_scans,
        strategy="random",
        fixed_random_sequence=common_random_ids,
        enforce_three_plane_balance=True,
        T_initial=T_selection_initial,
        x_values=x_values,
        selection_config=selection_config,
        profile_depth_range_mm=config.profile_depth_range_mm,
    )
    selection_outputs["three_fisher"] = _run_online_pose_selection(
        candidates=three_candidates,
        acquire_scan=three_acquire,
        parameter_scales=three_scales,
        initial_candidate_ids=initial_ids,
        total_scans=config.total_scans,
        strategy="fisher",
        fixed_random_sequence=None,
        enforce_three_plane_balance=True,
        T_initial=T_selection_initial,
        x_values=x_values,
        selection_config=selection_config,
        profile_depth_range_mm=config.profile_depth_range_mm,
    )

    datasets: dict[str, CalibrationDataset] = {}
    for geometry in ("single", "three"):
        for selection_strategy in ("random", "fisher"):
            key = f"{geometry}_{selection_strategy}"
            selected_ids, acquired_scans, trace, summary = (
                selection_outputs[key]
            )
            datasets[key] = _build_selected_dataset(
                geometry_strategy=f"{geometry}_plane",
                pose_selection_strategy=selection_strategy,
                acquired_scans=acquired_scans,
                selected_candidate_ids=selected_ids,
                selection_trace=trace,
                selection_summary=summary,
                T_ef_s_true=T_ef_s_true,
                frames=frames,
                common_center=common_center,
                frame_angles_deg=frame_angles_deg,
                trial_index=trial_index,
                config=config,
                selection_config=selection_config,
            )

    pose_stacks = {
        key: np.stack(
            [scan.T_base_ef for scan in dataset.scans],
            axis=0,
        )
        for key, dataset in datasets.items()
    }
    random_pose_difference = float(
        np.max(
            np.abs(
                pose_stacks["single_random"]
                - pose_stacks["three_random"]
            )
        )
    )
    if random_pose_difference > config.verification_atol:
        raise RuntimeError(
            "single/three random baseline poses differ: "
            f"max_abs={random_pose_difference:.6g}"
        )
    initial_count = selection_config.initial_random_scans
    initial_reference = pose_stacks["single_random"][:initial_count]
    initial_max_difference = max(
        float(np.max(np.abs(stack[:initial_count] - initial_reference)))
        for stack in pose_stacks.values()
    )
    if initial_max_difference > config.verification_atol:
        raise RuntimeError(
            "the four branches do not share the same initial random scans"
        )

    random_three_ids = np.asarray(
        [
            plane_assignments[candidate_id]
            for candidate_id in common_random_ids
        ],
        dtype=int,
    )
    diversity = _diversity_metrics(
        frames=frames,
        T_base_ef_stack=pose_stacks["single_random"],
        three_plane_ids=random_three_ids,
    )
    if (
        diversity["three_cross_plane_effective_normal_median_deg"]
        < config.min_effective_cross_plane_angle_deg
    ):
        raise RuntimeError(
            "three-plane calibration-effective normal diversity is too low: "
            f"{diversity['three_cross_plane_effective_normal_median_deg']:.6g} "
            f"< {config.min_effective_cross_plane_angle_deg:.6g} deg"
        )

    selection_manifest: dict[str, Any] = {}
    for key, (
        selected_ids,
        _acquired_scans,
        trace,
        summary,
    ) in selection_outputs.items():
        selection_manifest[key] = {
            **summary,
            "selection_trace": trace,
            "robot_pose_stack_sha256": _array_sha256(pose_stacks[key]),
            "selected_pose_library_sha256": hashlib.sha256(
                np.asarray(selected_ids, dtype=np.int64).tobytes()
            ).hexdigest(),
        }
    comparison = {
        "trial_index": int(trial_index),
        "T_ef_s_true_sha256": _array_sha256(T_ef_s_true),
        "candidate_pose_library_sha256": _pose_library_sha256(
            shared_samples
        ),
        "candidate_pool_size": selection_config.candidate_pool_size,
        "candidate_pool_assignment_counts": {
            str(plane_id): int(
                np.count_nonzero(plane_assignments == plane_id)
            )
            for plane_id in range(3)
        },
        "initial_random_candidate_ids": initial_ids,
        "initial_random_pose_max_abs_difference": initial_max_difference,
        "all_branches_share_initial_random_poses": True,
        "single_pose_library_sha256": _array_sha256(
            pose_stacks["single_random"]
        ),
        "three_pose_library_sha256": _array_sha256(
            pose_stacks["three_random"]
        ),
        "shared_pose_library_sha256": _pose_library_sha256(shared_samples),
        "single_robot_pose_stack_sha256": _array_sha256(
            pose_stacks["single_random"]
        ),
        "three_robot_pose_stack_sha256": _array_sha256(
            pose_stacks["three_random"]
        ),
        "pose_libraries_are_independent": False,
        "pose_libraries_are_shared": True,
        "single_and_three_robot_poses_identical": True,
        "paired_robot_pose_max_abs_difference": random_pose_difference,
        "pose_sampling_frame": POSE_SAMPLING_FRAME,
        "sensor_feasibility_rule": (
            "complete_profile_inside_depth_and_finite_patch_for_"
            "single_plane_0_and_assigned_three_plane"
        ),
        "fisher_linearization": (
            "current_estimate_from_acquired_noisy_measurements"
        ),
        "fisher_relinearization": (
            "all_acquired_measurements_after_every_scan"
        ),
        "future_candidate_profile_source": (
            "predicted_from_current_handeye_and_plane_estimates"
        ),
        "future_ground_truth_profiles_visible_to_policy": False,
        "policy_candidate_interface": (
            "candidate_kinematics_plus_acquire_selected_callback"
        ),
        "selection_initial_transform_source": (
            "simulation_truth_perturbed_nominal_initial_estimate"
        ),
        "selection_initial_transform_is_truth_centered": True,
        "ground_truth_parameters_directly_visible_to_policy": False,
        "selection_initial_transform_sha256": _array_sha256(
            T_selection_initial
        ),
        "selection_measurements_saved_in_dataset": True,
        "selection_measurement_noise": {
            "std_mm": selection_config.measurement_noise_std_mm,
            "axis": selection_config.measurement_noise_axis,
            "seed": selection_config.measurement_seed,
            "keying": "trial_candidate_channel",
        },
        "fisher_objective": (
            "estimated_state_handeye_marginal_"
            f"{selection_config.fisher_objective}"
        ),
        "fisher_plane_treatment": (
            "unknown_plane_parameters_jointly_included_then_"
            "schur_marginalized"
        ),
        "fisher_information_prior": "none",
        "fisher_coordinate_metric": {
            "rotation_scale_deg": selection_config.rotation_scale_deg,
            "translation_scale_mm": (
                selection_config.translation_scale_mm
            ),
            "plane_normal_scale_deg": (
                selection_config.plane_normal_scale_deg
            ),
            "plane_offset_scale_mm": (
                selection_config.plane_offset_scale_mm
            ),
            "interpretation": (
                "fixed_dimensionless_state_scaling_not_information_prior"
            ),
        },
        "fisher_measurement_model": {
            "residual": "point_to_plane",
            "information": "J_transpose_Sigma_inverse_J",
            "point_noise": (
                "independent_zero_mean_gaussian_sensor_coordinates"
            ),
            "robot_pose_uncertainty": "not_modelled",
            "candidate_expected_residual": "zero",
            "covariance_derivative_in_candidate_fisher": False,
        },
        "fisher_method_references": [
            {
                "role": "estimated_state_sequential_nbv",
                "citation": "Yang_Rebello_Waslander_2023",
                "url": "https://arxiv.org/abs/2303.06766",
            },
            {
                "role": "interest_nuisance_schur_marginalization",
                "citation": "Peng_Sturm_ICCV_2019",
                "url": "https://arxiv.org/abs/1811.03264",
            },
            {
                "role": "weighted_fim_and_e_optimality",
                "citation": "Wilson_Schultz_Murphey_TRO_2014",
                "url": (
                    "https://doi.org/10.1109/TRO.2014.2345918"
                ),
            },
        ],
        "candidate_bank_geometry": (
            "simulator_generated_oracle_feasible_robot_pose_bank"
        ),
        "candidate_bank_is_real_deployment_model": False,
        "sampling_statistics": sampling_statistics,
        "diversity_metrics": diversity,
        "single_scan_count": config.total_scans,
        "three_scan_count": config.total_scans,
        "three_scans_per_plane": {
            str(index): config.total_scans // 3
            for index in range(3)
        },
        "selections": selection_manifest,
        "fisher_vs_random": {
            "single_plane": _selection_improvement(
                selection_outputs["single_fisher"][3],
                selection_outputs["single_random"][3],
            ),
            "three_plane": _selection_improvement(
                selection_outputs["three_fisher"][3],
                selection_outputs["three_random"][3],
            ),
        },
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
    text = json.dumps(
        _jsonable(payload),
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )
    temporary.write_text(
        text + "\n",
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
    acquisition_mode: str,
    strategy: str,
    master_seed: int,
    trials: int,
    config: FairGenerationConfig,
    selection_config: PoseSelectionConfig | None = None,
) -> dict[str, Any]:
    (root / "trials").mkdir(parents=True, exist_ok=False)
    manifest = {
        "schema": COLLECTION_SCHEMA,
        "schema_version": COLLECTION_SCHEMA_VERSION,
        "status": "in_progress",
        "master_seed": master_seed,
        "command_mode": (
            "shared-global-random"
            if selection_config is None
            else "shared-global-fisher-vs-random"
        ),
        "acquisition_mode": acquisition_mode,
        "profile_state": (
            "ideal" if selection_config is None else "measured"
        ),
        "noise_applied": (
            False
            if selection_config is None
            else selection_config.measurement_noise_std_mm > 0.0
        ),
        "generator_variant": COMPARISON_SCHEMA,
        "pose_sampling_frame": POSE_SAMPLING_FRAME,
        "single_and_three_robot_poses_identical": (
            selection_config is None or strategy.endswith("_random")
        ),
        "all_selection_branches_share_initial_random_poses": (
            selection_config is not None
        ),
        "strategy": strategy,
        "requested_trials": trials,
        "completed_trials": 0,
        "config": asdict(config),
        "pose_selection_config": (
            None
            if selection_config is None
            else asdict(selection_config)
        ),
        "trials": [],
    }
    _write_json(root / "collection.json", manifest)
    return manifest


def _validate_args(args: argparse.Namespace) -> FairGenerationConfig:
    if args.total_scans % 3 != 0:
        raise SystemExit("--total-scans must be divisible by 3")
    if args.total_scans < 9:
        raise SystemExit(
            "--total-scans must be at least 9 so every plane has at least "
            "three calibration poses"
        )
    if args.profile_points < 2:
        raise SystemExit("--profile-points must be at least 2")
    depth = _finite_pair(args.profile_depth_range_mm, "profile depth range")
    tilt = _finite_pair(args.view_tilt_range_deg, "view tilt range")
    if depth[0] <= 0.0 or depth[0] == depth[1]:
        raise SystemExit("profile depth range must be positive and non-zero")
    if tilt[0] < 0.0 or tilt[1] >= 89.0:
        raise SystemExit("view tilt range must lie within [0, 89)")
    if args.profile_half_width_mm <= 0.0:
        raise SystemExit("--profile-half-width-mm must be positive")
    if args.tangent_range_mm <= 0.0:
        raise SystemExit("--tangent-range-mm must be positive")
    if (
        not np.isfinite(args.min_abs_plane_normal_z)
        or args.min_abs_plane_normal_z <= 0.0
    ):
        raise SystemExit("--min-abs-plane-normal-z must be positive and finite")
    if (
        not np.isfinite(args.verification_atol)
        or args.verification_atol <= 0.0
    ):
        raise SystemExit("--verification-atol must be positive and finite")
    if (
        not np.isfinite(args.min_effective_cross_plane_angle_deg)
        or not 0.0 <= args.min_effective_cross_plane_angle_deg <= 180.0
    ):
        raise SystemExit(
            "--min-effective-cross-plane-angle-deg must lie within [0, 180]"
        )
    return FairGenerationConfig(
        total_scans=args.total_scans,
        profile_points=args.profile_points,
        profile_half_width_mm=float(args.profile_half_width_mm),
        tangent_range_mm=float(args.tangent_range_mm),
        profile_depth_range_mm=depth,
        view_tilt_range_deg=tilt,
        view_azimuth_range_deg=_finite_pair(
            args.view_azimuth_range_deg, "view azimuth range"
        ),
        sensor_roll_range_deg=_finite_pair(
            args.sensor_roll_range_deg, "sensor roll range"
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
        max_local_pose_trials=args.max_local_pose_trials,
        min_abs_plane_normal_z=float(args.min_abs_plane_normal_z),
        verification_atol=float(args.verification_atol),
        min_effective_cross_plane_angle_deg=float(
            args.min_effective_cross_plane_angle_deg
        ),
    )


def _validate_selection_args(
    args: argparse.Namespace,
    config: FairGenerationConfig,
) -> PoseSelectionConfig:
    initial_scans = int(args.initial_random_scans)
    if initial_scans < 9:
        raise SystemExit(
            "--initial-random-scans must be at least 9 so the no-prior "
            "single/three joint Fisher systems can be full rank"
        )
    if initial_scans >= config.total_scans:
        raise SystemExit(
            "--initial-random-scans must be smaller than --total-scans so "
            "the Fisher and random continuation policies are compared"
        )
    if initial_scans % 3 != 0:
        raise SystemExit("--initial-random-scans must be divisible by 3")
    candidate_pool_size = (
        3 * config.total_scans
        if args.candidate_pool_size is None
        else int(args.candidate_pool_size)
    )
    if candidate_pool_size < config.total_scans:
        raise SystemExit(
            "--candidate-pool-size must be at least --total-scans"
        )
    if candidate_pool_size % 3 != 0:
        raise SystemExit("--candidate-pool-size must be divisible by 3")

    positive_values = {
        "--fisher-profile-noise-std-mm": args.fisher_profile_noise_std_mm,
        "--fisher-rotation-scale-deg": (
            args.fisher_rotation_scale_deg
        ),
        "--fisher-translation-scale-mm": (
            args.fisher_translation_scale_mm
        ),
        "--fisher-plane-normal-scale-deg": (
            args.fisher_plane_normal_scale_deg
        ),
        "--fisher-plane-offset-scale-mm": (
            args.fisher_plane_offset_scale_mm
        ),
        "--selection-estimator-tolerance": (
            args.selection_estimator_tolerance
        ),
    }
    for name, raw_value in positive_values.items():
        value = float(raw_value)
        if not np.isfinite(value) or value <= 0.0:
            raise SystemExit(f"{name} must be positive and finite")
    nonnegative_values = {
        "--selection-measurement-noise-std-mm": (
            args.selection_measurement_noise_std_mm
        ),
        "--selection-initial-translation-range-mm": (
            args.selection_initial_translation_range_mm
        ),
        "--selection-initial-angle-range-deg": (
            args.selection_initial_angle_range_deg
        ),
    }
    for name, raw_value in nonnegative_values.items():
        value = float(raw_value)
        if not np.isfinite(value) or value < 0.0:
            raise SystemExit(f"{name} must be non-negative and finite")

    return PoseSelectionConfig(
        initial_random_scans=initial_scans,
        candidate_pool_size=candidate_pool_size,
        fisher_profile_noise_std_mm=float(
            args.fisher_profile_noise_std_mm
        ),
        rotation_scale_deg=float(
            args.fisher_rotation_scale_deg
        ),
        translation_scale_mm=float(
            args.fisher_translation_scale_mm
        ),
        plane_normal_scale_deg=float(
            args.fisher_plane_normal_scale_deg
        ),
        plane_offset_scale_mm=float(
            args.fisher_plane_offset_scale_mm
        ),
        fisher_objective=str(args.fisher_objective),
        measurement_noise_std_mm=float(
            args.selection_measurement_noise_std_mm
        ),
        measurement_noise_axis=str(
            args.selection_measurement_noise_axis
        ),
        measurement_seed=int(args.selection_measurement_seed),
        initialization_seed=int(args.selection_initialization_seed),
        initial_translation_range_mm=float(
            args.selection_initial_translation_range_mm
        ),
        initial_angle_range_deg=float(
            args.selection_initial_angle_range_deg
        ),
        initial_rotation_perturbation=str(
            args.selection_initial_rotation_perturbation
        ),
        initial_translation_perturbation=str(
            args.selection_initial_translation_perturbation
        ),
        estimator_max_iterations=int(
            args.selection_estimator_max_iterations
        ),
        estimator_tolerance=float(args.selection_estimator_tolerance),
    )


def _aggregate_fisher_random_metrics(
    trials: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for geometry in ("single_plane", "three_plane"):
        rows = [
            trial["fisher_vs_random"][geometry]
            for trial in trials
        ]
        geometry_summary: dict[str, Any] = {"trial_count": len(rows)}
        for metric in rows[0]:
            values = np.asarray(
                [float(row[metric]) for row in rows],
                dtype=float,
            )
            geometry_summary[metric] = {
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "p05": float(np.percentile(values, 5.0)),
                "p95": float(np.percentile(values, 95.0)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
        summary[geometry] = geometry_summary
    return summary


def _run_random_only_generation(
    *,
    args: argparse.Namespace,
    config: FairGenerationConfig,
    output_root: Path,
) -> int:
    """Write the historical paired, noise-free random collections."""
    collection_roots = {
        "single": output_root / "single_plane",
        "three": output_root / "three_plane",
    }
    collection_manifests: dict[str, dict[str, Any]] = {}
    for key, root in collection_roots.items():
        root.mkdir()
        collection_manifests[key] = _prepare_collection(
            root,
            acquisition_mode=f"{key}_plane_shared_global_random",
            strategy=f"{key}_plane_shared_global_random",
            master_seed=args.seed,
            trials=args.trials,
            config=config,
            selection_config=None,
        )

    comparison_manifest: dict[str, Any] = {
        "schema": COMPARISON_SCHEMA,
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "status": "in_progress",
        "master_seed": args.seed,
        "requested_trials": args.trials,
        "completed_trials": 0,
        "single_collection": "single_plane",
        "three_collection": "three_plane",
        "single_random_collection": "single_plane",
        "three_random_collection": "three_plane",
        "pose_sampling_frame": POSE_SAMPLING_FRAME,
        "random_single_and_three_robot_poses_identical": True,
        "single_and_three_robot_poses_identical": True,
        "all_branches_share_initial_random_poses": True,
        "robot_ik_and_collision_checked": False,
        "config": asdict(config),
        "pose_selection_config": None,
        "trials": [],
    }
    comparison_path = output_root / "comparison_manifest.json"
    _write_json(comparison_path, comparison_manifest)

    for trial_index in range(args.trials):
        single, three, comparison = _generate_shared_global_trial(
            config=config,
            master_seed=args.seed,
            trial_index=trial_index,
        )
        datasets = {"single": single, "three": three}
        relative_path = Path("trials") / f"trial_{trial_index:06d}"
        for key, dataset in datasets.items():
            root = collection_roots[key]
            save_calibration_dataset(dataset, root / relative_path)
            collection_manifests[key]["trials"].append(
                {
                    "trial_index": trial_index,
                    "relative_path": relative_path.as_posix(),
                    "logical_dataset_sha256": logical_dataset_sha256(
                        dataset
                    ),
                    "pair_trial_index": trial_index,
                    "T_ef_s_true_sha256": comparison[
                        "T_ef_s_true_sha256"
                    ],
                    "pose_library_sha256": comparison[
                        "shared_pose_library_sha256"
                    ],
                }
            )
            collection_manifests[key]["completed_trials"] += 1

        comparison.update(
            {
                "single_relative_path": (
                    Path("single_plane") / relative_path
                ).as_posix(),
                "three_relative_path": (
                    Path("three_plane") / relative_path
                ).as_posix(),
            }
        )
        comparison_manifest["trials"].append(comparison)
        comparison_manifest["completed_trials"] += 1
        for key, root in collection_roots.items():
            _write_json(
                root / "collection.json",
                collection_manifests[key],
            )
        _write_json(comparison_path, comparison_manifest)
        print(
            f"[shared-random] {trial_index + 1}/{args.trials}: "
            f"total={config.total_scans}"
        )

    for manifest in collection_manifests.values():
        manifest["status"] = "complete"
    comparison_manifest["status"] = "complete"
    for key, root in collection_roots.items():
        _write_json(root / "collection.json", collection_manifests[key])
    _write_json(comparison_path, comparison_manifest)
    print(f"single collection      : {collection_roots['single']}")
    print(f"three collection       : {collection_roots['three']}")
    print(f"comparison manifest    : {comparison_path}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = _validate_args(args)
    output_root = _prepare_empty_root(args.output_dir)
    if args.pose_selection_mode == "random_only":
        return _run_random_only_generation(
            args=args,
            config=config,
            output_root=output_root,
        )

    selection_config = _validate_selection_args(args, config)
    collection_roots = {
        # Keep the historical random paths for downstream scripts.
        "single_random": output_root / "single_plane",
        "three_random": output_root / "three_plane",
        "single_fisher": output_root / "single_plane_fisher",
        "three_fisher": output_root / "three_plane_fisher",
    }
    collection_manifests: dict[str, dict[str, Any]] = {}
    for key, root in collection_roots.items():
        geometry, selection_strategy = key.split("_")
        root.mkdir()
        collection_manifests[key] = _prepare_collection(
            root,
            acquisition_mode=(
                f"{geometry}_plane_global_{selection_strategy}"
            ),
            strategy=(
                f"{geometry}_plane_shared_global_{selection_strategy}"
            ),
            master_seed=args.seed,
            trials=args.trials,
            config=config,
            selection_config=selection_config,
        )

    comparison_manifest: dict[str, Any] = {
        "schema": COMPARISON_SCHEMA,
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "status": "in_progress",
        "master_seed": args.seed,
        "requested_trials": args.trials,
        "completed_trials": 0,
        "single_collection": "single_plane",
        "three_collection": "three_plane",
        "single_random_collection": "single_plane",
        "three_random_collection": "three_plane",
        "single_fisher_collection": "single_plane_fisher",
        "three_fisher_collection": "three_plane_fisher",
        "pose_sampling_frame": POSE_SAMPLING_FRAME,
        "random_single_and_three_robot_poses_identical": True,
        # Compatibility alias: the historical collection paths now identify
        # the random pair, which remains pose-identical.
        "single_and_three_robot_poses_identical": True,
        "all_branches_share_initial_random_poses": True,
        "robot_ik_and_collision_checked": False,
        "config": asdict(config),
        "pose_selection_config": asdict(selection_config),
        "trials": [],
    }
    comparison_path = output_root / "comparison_manifest.json"
    _write_json(comparison_path, comparison_manifest)

    for trial_index in range(args.trials):
        datasets, comparison = _generate_fisher_random_trial(
            config=config,
            selection_config=selection_config,
            master_seed=args.seed,
            trial_index=trial_index,
        )
        relative_path = Path("trials") / f"trial_{trial_index:06d}"
        for key, dataset in datasets.items():
            root = collection_roots[key]
            save_calibration_dataset(dataset, root / relative_path)
            entry = {
                "trial_index": trial_index,
                "relative_path": relative_path.as_posix(),
                "logical_dataset_sha256": logical_dataset_sha256(dataset),
                "pair_trial_index": trial_index,
                "T_ef_s_true_sha256": comparison["T_ef_s_true_sha256"],
                "pose_library_sha256": comparison["selections"][key][
                    "selected_pose_library_sha256"
                ],
                "initial_random_candidate_ids": comparison[
                    "initial_random_candidate_ids"
                ],
            }
            collection_manifests[key]["trials"].append(entry)
            collection_manifests[key]["completed_trials"] += 1

        relative_paths = {
            key: (
                Path(root.name) / relative_path
            ).as_posix()
            for key, root in collection_roots.items()
        }
        comparison.update(
            {
                "single_relative_path": relative_paths["single_random"],
                "three_relative_path": relative_paths["three_random"],
                "single_random_relative_path": relative_paths[
                    "single_random"
                ],
                "three_random_relative_path": relative_paths[
                    "three_random"
                ],
                "single_fisher_relative_path": relative_paths[
                    "single_fisher"
                ],
                "three_fisher_relative_path": relative_paths[
                    "three_fisher"
                ],
            }
        )
        comparison_manifest["trials"].append(comparison)
        comparison_manifest["completed_trials"] += 1
        for key, root in collection_roots.items():
            _write_json(
                root / "collection.json",
                collection_manifests[key],
            )
        _write_json(comparison_path, comparison_manifest)
        if selection_config.fisher_objective == "d_optimal":
            metric = "fisher_minus_random_marginal_logdet"
            metric_label = "delta-logdet"
        else:
            metric = "fisher_over_random_marginal_min_eigenvalue_ratio"
            metric_label = "lambda-min-ratio"
        single_gain = comparison["fisher_vs_random"]["single_plane"][metric]
        three_gain = comparison["fisher_vs_random"]["three_plane"][metric]
        print(
            f"[Fisher-vs-random] {trial_index + 1}/{args.trials}: "
            f"objective={selection_config.fisher_objective}, "
            f"initial={selection_config.initial_random_scans}, "
            f"total={config.total_scans}, "
            f"pool={selection_config.candidate_pool_size}, "
            f"{metric_label}(single)={single_gain:.3f}, "
            f"{metric_label}(three)={three_gain:.3f}"
        )

    for manifest in collection_manifests.values():
        manifest["status"] = "complete"
    comparison_manifest["fisher_vs_random_summary"] = (
        _aggregate_fisher_random_metrics(comparison_manifest["trials"])
    )
    comparison_manifest["status"] = "complete"
    for key, root in collection_roots.items():
        _write_json(root / "collection.json", collection_manifests[key])
    _write_json(comparison_path, comparison_manifest)
    print(f"single random collection: {collection_roots['single_random']}")
    print(f"single Fisher collection: {collection_roots['single_fisher']}")
    print(f"three random collection : {collection_roots['three_random']}")
    print(f"three Fisher collection : {collection_roots['three_fisher']}")
    print(f"comparison manifest      : {comparison_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
