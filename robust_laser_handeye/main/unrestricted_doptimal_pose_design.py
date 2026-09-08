
from __future__ import annotations

import argparse
import contextlib
import csv
from dataclasses import asdict, dataclass
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


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from laser_handeye.active_fisher import (  # noqa: E402
    PlaneEstimate,
    apply_joint_local_update,
    joint_residual_jacobian,
)
from laser_handeye.calibration import (  # noqa: E402
    calibrate_planes,
    mean_self_fitted_plane_rms,
)
from laser_handeye.data import LaserScan  # noqa: E402
from laser_handeye.initialization import make_initial_guess  # noqa: E402
from laser_handeye.pose_generation import sensor_pose_to_robot_pose  # noqa: E402
from laser_handeye.pose_design import (  # noqa: E402
    PlaneFrame as DesignPlaneFrame,
    PlaneRelativePoseBounds,
    normalized_design,
    plane_relative_pose_from_row,
    sensor_pose_from_plane_relative,
)
from laser_handeye.se3 import rot_error_deg  # noqa: E402
from laser_handeye.simulation import (  # noqa: E402
    is_reachable_simple,
    sample_random_handeye,
)
from main import generate_independent_random_plane_comparison as base  # noqa: E402


SCHEMA = "laser_handeye.unrestricted_doptimal_pose_design"
SCHEMA_VERSION = 2
JOINT_DIMENSION = 9
HAND_EYE_DIMENSION = 6
@dataclass(frozen=True)
class Candidate:
    candidate_id: int
    target_u_mm: float
    target_v_mm: float
    distance_mm: float
    theta_deg: float
    azimuth_deg: float
    normal_azimuth_sensor_deg: float
    T_base_s: np.ndarray
    T_base_ef: np.ndarray
    points_s: np.ndarray
    plane_normal_sensor: np.ndarray
    view_normal_base: np.ndarray
    feasibility: base.ProfileFeasibility


@dataclass(frozen=True)
class InformationMetrics:
    valid: bool
    logdet_H_eff: float
    lambda_min: float
    lambda_max: float
    condition_number: float
    rank: int
    eigenvalues: tuple[float, ...]
    reason: str = ""


@dataclass(frozen=True)
class SearchResult:
    start_id: int
    initial_indices: tuple[int, ...]
    selected_indices: tuple[int, ...]
    initial_score: float
    final_score: float
    iterations: int
    evaluated_exchanges: int


@dataclass(frozen=True)
class BaselineResult:
    subset_id: int
    selected_indices: tuple[int, ...]
    metrics: InformationMetrics


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
        raise ValueError(f"{name} must contain two values")
    lower, upper = map(float, values)
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        raise ValueError(f"invalid {name}: {values}")
    return lower, upper


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Select unrestricted single-plane poses by nuisance-marginalized "
            "hand-eye D-optimality."
        )
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("result_unrestricted_doptimal")
    )
    parser.add_argument("--seed", type=_nonnegative_int, default=17)
    parser.add_argument("--candidate-count", type=_positive_int, default=5_000)
    parser.add_argument(
        "--candidate-design",
        choices=("random", "lhs"),
        default="lhs",
        help="Plane-relative candidate sampler. Circular designs use the public pose_design API separately.",
    )
    parser.add_argument(
        "--candidate-batch-size",
        type=_positive_int,
        default=5_000,
        help=(
            "Rows per feasibility batch. Keeping this fixed makes a larger "
            "candidate bank retain the smaller bank as a prefix."
        ),
    )
    parser.add_argument(
        "--candidate-max-batches", type=_positive_int, default=100
    )
    parser.add_argument(
        "--subset-sizes", type=_positive_int, nargs="+", default=(3, 4, 5, 6, 8, 10, 12)
    )
    parser.add_argument("--random-starts", type=_positive_int, default=100)
    parser.add_argument(
        "--random-baseline-subsets", type=_positive_int, default=1_000
    )
    parser.add_argument("--max-exchange-iterations", type=_positive_int, default=200)
    parser.add_argument("--exchange-chunk-size", type=_positive_int, default=1_024)
    parser.add_argument("--improvement-tolerance", type=float, default=1e-9)
    parser.add_argument("--initialization-attempts", type=_positive_int, default=10_000)
    parser.add_argument(
        "--skip-pool-growth-check",
        action="store_true",
        help=(
            "Skip the half-bank versus full-bank monotonicity audit. By default "
            "the prefix optimum is also used as an extra full-bank search start."
        ),
    )
    parser.add_argument("--schur-regularization", type=float, default=0.0)
    parser.add_argument("--rank-relative-tolerance", type=float, default=1e-10)
    parser.add_argument("--nuisance-relative-tolerance", type=float, default=1e-12)

    parser.add_argument("--profile-points", type=_positive_int, default=100)
    parser.add_argument("--profile-half-width-mm", type=float, default=25.0)
    parser.add_argument(
        "--profile-depth-range-mm", type=float, nargs=2, default=(60.0, 150.0)
    )
    parser.add_argument("--target-half-size-mm", type=float, default=100.0)
    parser.add_argument(
        "--target-u-range-mm", type=float, nargs=2, default=(-40.0, 40.0)
    )
    parser.add_argument(
        "--target-v-range-mm", type=float, nargs=2, default=(-40.0, 40.0)
    )
    parser.add_argument(
        "--view-tilt-range-deg", type=float, nargs=2, default=(10.0, 60.0)
    )
    parser.add_argument(
        "--view-azimuth-range-deg", type=float, nargs=2, default=(-180.0, 180.0)
    )
    parser.add_argument(
        "--normal-azimuth-sensor-range-deg",
        type=float,
        nargs=2,
        default=(-180.0, 180.0),
        help="azimuth of the projected plane normal in sensor XY",
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
    parser.add_argument("--min-abs-plane-normal-z", type=float, default=1e-4)
    parser.add_argument("--verification-atol", type=float, default=1e-8)
    parser.add_argument(
        "--workspace-min-mm", type=float, nargs=3, default=(-700.0, -700.0, 50.0)
    )
    parser.add_argument(
        "--workspace-max-mm", type=float, nargs=3, default=(700.0, 700.0, 900.0)
    )

    parser.add_argument("--fisher-profile-noise-std-mm", type=float, default=1.0)
    parser.add_argument("--fisher-noise-axis", choices=("z", "xz"), default="xz")

    parser.add_argument("--mc-trials", type=_nonnegative_int, default=100)
    parser.add_argument("--mc-noise-std-mm", type=float, default=0.20)
    parser.add_argument("--mc-noise-axis", choices=("z", "xz"), default="xz")
    parser.add_argument("--mc-init-rotation-deg", type=float, default=5.0)
    parser.add_argument("--mc-init-translation-mm", type=float, default=20.0)
    parser.add_argument("--mc-max-iterations", type=_positive_int, default=100)
    parser.add_argument("--mc-tolerance", type=float, default=1e-9)
    parser.add_argument(
        "--mc-plane-offset-mode",
        choices=("fitted", "joint", "difference"),
        default="joint",
    )
    parser.add_argument(
        "--mc-verbose",
        action="store_true",
        help="Show the underlying alternating calibrator's iteration log.",
    )
    parser.add_argument(
        "--skip-plots", action="store_true", help="Do not create PNG diagnostics."
    )
    return parser


def validate_args(args: argparse.Namespace) -> dict[str, tuple[float, float]]:
    ranges = {
        "target_u_mm": _finite_pair(args.target_u_range_mm, "target U range"),
        "target_v_mm": _finite_pair(args.target_v_range_mm, "target V range"),
        "distance_mm": _finite_pair(args.profile_depth_range_mm, "profile depth range"),
        "theta_deg": _finite_pair(args.view_tilt_range_deg, "view tilt range"),
        "azimuth_deg": _finite_pair(args.view_azimuth_range_deg, "azimuth range"),
        "normal_azimuth_sensor_deg": _finite_pair(
            args.normal_azimuth_sensor_range_deg,
            "sensor-frame plane-normal azimuth range",
        ),
    }
    _finite_pair(args.plane_angle_range_deg, "plane angle range")
    _finite_pair(args.plane_center_xy_range_mm, "plane center XY range")
    _finite_pair(args.plane_center_z_range_mm, "plane center Z range")
    if min(args.subset_sizes) < 3:
        raise SystemExit("all subset sizes must be at least 3")
    if max(args.subset_sizes) > args.candidate_count:
        raise SystemExit("subset size cannot exceed candidate count")
    if args.profile_points < 2 or args.profile_half_width_mm <= 0.0:
        raise SystemExit("profile geometry must contain at least two points and positive width")
    if args.target_half_size_mm <= 0.0:
        raise SystemExit("--target-half-size-mm must be positive")
    if ranges["distance_mm"][0] <= 0.0:
        raise SystemExit("profile depth must be positive")
    if ranges["theta_deg"][0] <= 0.0 or ranges["theta_deg"][1] >= 89.0:
        raise SystemExit("view tilt range must lie within (0, 89) degrees")
    for name in (
        "improvement_tolerance",
        "schur_regularization",
        "rank_relative_tolerance",
        "nuisance_relative_tolerance",
    ):
        if float(getattr(args, name)) < 0.0:
            raise SystemExit(f"--{name.replace('_', '-')} must be non-negative")
    if args.fisher_profile_noise_std_mm <= 0.0 or args.mc_noise_std_mm < 0.0:
        raise SystemExit("noise standard deviations must be valid")
    if args.mc_init_rotation_deg < 0.0 or args.mc_init_translation_mm < 0.0:
        raise SystemExit("Monte-Carlo initialization bounds must be non-negative")
    if np.any(np.asarray(args.workspace_min_mm) >= np.asarray(args.workspace_max_mm)):
        raise SystemExit("workspace minimum must be below workspace maximum")
    return ranges


def _fair_config(args: argparse.Namespace) -> base.FairGenerationConfig:
    return base.FairGenerationConfig(
        total_scans=int(args.candidate_count),
        profile_points=int(args.profile_points),
        profile_half_width_mm=float(args.profile_half_width_mm),
        tangent_range_mm=float(args.target_half_size_mm),
        profile_depth_range_mm=tuple(map(float, args.profile_depth_range_mm)),
        view_tilt_range_deg=tuple(map(float, args.view_tilt_range_deg)),
        view_azimuth_range_deg=tuple(map(float, args.view_azimuth_range_deg)),
        # The legacy feasibility configuration only stores this unused angular
        # range under its historical field name.  Pose construction below uses
        # the new sensor-frame plane-normal azimuth convention.
        sensor_roll_range_deg=tuple(
            map(float, args.normal_azimuth_sensor_range_deg)
        ),
        plane_angle_range_deg=tuple(map(float, args.plane_angle_range_deg)),
        plane_center_xy_range_mm=tuple(map(float, args.plane_center_xy_range_mm)),
        plane_center_z_range_mm=tuple(map(float, args.plane_center_z_range_mm)),
        max_local_pose_trials=int(args.candidate_count * args.candidate_max_batches),
        min_abs_plane_normal_z=float(args.min_abs_plane_normal_z),
        verification_atol=float(args.verification_atol),
        min_effective_cross_plane_angle_deg=0.0,
    )


def build_candidate_bank(
    *,
    args: argparse.Namespace,
    ranges: dict[str, tuple[float, float]],
    rng: np.random.Generator,
    frame: base.PlaneFrame,
    common_center: np.ndarray,
    T_ef_s_true: np.ndarray,
    x_values: np.ndarray,
) -> tuple[list[Candidate], np.ndarray, dict[str, int]]:
    fair = _fair_config(args)
    plane = PlaneEstimate(frame.n, frame.l)
    design_frame = DesignPlaneFrame(frame.u, frame.v, frame.n, frame.l)
    design_bounds = PlaneRelativePoseBounds(
        target_u_mm=ranges["target_u_mm"],
        target_v_mm=ranges["target_v_mm"],
        distance_mm=ranges["distance_mm"],
        tilt_deg=ranges["theta_deg"],
        azimuth_deg=ranges["azimuth_deg"],
        normal_azimuth_sensor_deg=ranges["normal_azimuth_sensor_deg"],
    )
    candidates: list[Candidate] = []
    information: list[np.ndarray] = []
    counts = {
        "attempted": 0,
        "invalid_pose": 0,
        "profile_infeasible": 0,
        "robot_unreachable": 0,
        "jacobian_failure": 0,
    }

    for _batch_index in range(args.candidate_max_batches):
        if len(candidates) >= args.candidate_count:
            break
        rows = normalized_design(
            args.candidate_design,
            rng=rng,
            count=args.candidate_batch_size,
            dimensions=6,
        )
        for row in rows:
            if len(candidates) >= args.candidate_count:
                break
            counts["attempted"] += 1
            pose = plane_relative_pose_from_row(
                row,
                sample_id=len(candidates),
                bounds=design_bounds,
            )
            values = {
                "target_u_mm": pose.target_u_mm,
                "target_v_mm": pose.target_v_mm,
                "distance_mm": pose.distance_mm,
                "theta_deg": pose.tilt_deg,
                "azimuth_deg": pose.azimuth_deg,
                "normal_azimuth_sensor_deg": pose.normal_azimuth_sensor_deg,
            }
            try:
                T_base_s = sensor_pose_from_plane_relative(
                    design_frame, common_center, pose
                )
                T_base_ef = sensor_pose_to_robot_pose(T_base_s, T_ef_s_true)
            except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                counts["invalid_pose"] += 1
                continue

            feasible = base._profile_feasibility(
                plane=frame,
                common_center=common_center,
                T_base_s=T_base_s,
                x_values=x_values,
                config=fair,
            )
            if feasible is None:
                counts["profile_infeasible"] += 1
                continue
            if not is_reachable_simple(
                T_base_ef,
                xyz_min_mm=tuple(map(float, args.workspace_min_mm)),
                xyz_max_mm=tuple(map(float, args.workspace_max_mm)),
            ):
                counts["robot_unreachable"] += 1
                continue

            try:
                z_values = base._profile_z_values(frame, T_base_s, x_values)
                points_s = np.column_stack(
                    [x_values, np.zeros_like(x_values), z_values]
                )
                scan = LaserScan(
                    T_base_ef=T_base_ef,
                    points_s=points_s,
                    plane_id=0,
                    scan_id=len(candidates),
                    meta={"source": SCHEMA, **values},
                )
                residual, jacobian = joint_residual_jacobian(
                    {0: [scan]},
                    T_ef_s_true,
                    {0: plane},
                    profile_noise_std_mm=float(args.fisher_profile_noise_std_mm),
                    noise_axis=args.fisher_noise_axis,
                )
                if not np.all(np.isfinite(residual)) or not np.all(np.isfinite(jacobian)):
                    raise FloatingPointError("non-finite residual/Jacobian")
                H_i = jacobian.T @ jacobian
                H_i = 0.5 * (H_i + H_i.T)
            except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                counts["jacobian_failure"] += 1
                continue

            normal_sensor = T_base_s[:3, :3].T @ frame.n
            view_normal = -T_base_s[:3, 2]
            candidates.append(
                Candidate(
                    candidate_id=len(candidates),
                    target_u_mm=values["target_u_mm"],
                    target_v_mm=values["target_v_mm"],
                    distance_mm=values["distance_mm"],
                    theta_deg=values["theta_deg"],
                    azimuth_deg=values["azimuth_deg"],
                    normal_azimuth_sensor_deg=values[
                        "normal_azimuth_sensor_deg"
                    ],
                    T_base_s=T_base_s.copy(),
                    T_base_ef=T_base_ef.copy(),
                    points_s=points_s.copy(),
                    plane_normal_sensor=normal_sensor.copy(),
                    view_normal_base=view_normal.copy(),
                    feasibility=feasible,
                )
            )
            information.append(H_i)

    if len(candidates) != args.candidate_count:
        raise RuntimeError(
            f"generated only {len(candidates)}/{args.candidate_count} valid "
            f"candidates after {counts['attempted']} attempts"
        )
    return candidates, np.stack(information), counts


class InformationScorer:
    """Batched Schur-complement scorer with explicit singularity handling."""

    def __init__(
        self,
        *,
        regularization: float = 0.0,
        rank_relative_tolerance: float = 1e-10,
        nuisance_relative_tolerance: float = 1e-12,
    ) -> None:
        self.regularization = float(regularization)
        self.rank_relative_tolerance = float(rank_relative_tolerance)
        self.nuisance_relative_tolerance = float(nuisance_relative_tolerance)

    def effective_batch(self, matrices: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[str]]:
        values = np.asarray(matrices, dtype=float)
        if values.ndim == 2:
            values = values[None, :, :]
        if values.ndim != 3 or values.shape[1:] != (JOINT_DIMENSION, JOINT_DIMENSION):
            raise ValueError("information matrices must have shape (K, 9, 9)")
        values = 0.5 * (values + values.transpose(0, 2, 1))
        effective = np.full((len(values), 6, 6), np.nan, dtype=float)
        valid = np.all(np.isfinite(values), axis=(1, 2))
        reasons = ["" if item else "nonfinite_joint_information" for item in valid]

        indices = np.flatnonzero(valid)
        if len(indices) == 0:
            return effective, valid, reasons
        nuisance = values[indices, 6:, 6:].copy()
        if self.regularization:
            nuisance += self.regularization * np.eye(3)[None, :, :]
        nuisance_eigenvalues = np.linalg.eigvalsh(nuisance)
        nuisance_scale = np.maximum(np.max(np.abs(nuisance_eigenvalues), axis=1), 1.0)
        nuisance_ok = (
            nuisance_eigenvalues[:, 0]
            > self.nuisance_relative_tolerance * nuisance_scale
        )
        for local_index in np.flatnonzero(~nuisance_ok):
            original_index = int(indices[local_index])
            valid[original_index] = False
            reasons[original_index] = "singular_plane_information"

        solve_indices = indices[nuisance_ok]
        if len(solve_indices):
            nuisance_valid = nuisance[nuisance_ok]
            coupling_transpose = values[solve_indices, 6:, :6]
            try:
                # Deliberately solve H_pp X = H_ph; never form H_pp**-1.
                solved = np.linalg.solve(nuisance_valid, coupling_transpose)
                schur = (
                    values[solve_indices, :6, :6]
                    - values[solve_indices, :6, 6:] @ solved
                )
                effective[solve_indices] = 0.5 * (
                    schur + schur.transpose(0, 2, 1)
                )
            except np.linalg.LinAlgError:
                # A rare batch-level failure is retried one matrix at a time so
                # one invalid subset does not discard all other exchanges.
                for original_index in solve_indices:
                    try:
                        H = values[original_index]
                        Hpp = H[6:, 6:] + self.regularization * np.eye(3)
                        solved = np.linalg.solve(Hpp, H[6:, :6])
                        schur = H[:6, :6] - H[:6, 6:] @ solved
                        effective[original_index] = 0.5 * (schur + schur.T)
                    except np.linalg.LinAlgError:
                        valid[original_index] = False
                        reasons[original_index] = "plane_solve_failed"
        return effective, valid, reasons

    def metrics_batch(self, matrices: np.ndarray) -> list[InformationMetrics]:
        effective, nuisance_valid, reasons = self.effective_batch(matrices)
        outputs: list[InformationMetrics] = []
        for index, matrix in enumerate(effective):
            if not nuisance_valid[index] or not np.all(np.isfinite(matrix)):
                outputs.append(
                    InformationMetrics(
                        False, -math.inf, math.nan, math.nan, math.inf, 0,
                        (math.nan,) * 6, reasons[index] or "nonfinite_schur",
                    )
                )
                continue
            eigenvalues = np.linalg.eigvalsh(matrix)
            scale = max(float(np.max(np.abs(eigenvalues))), 1.0)
            tolerance = self.rank_relative_tolerance * scale
            rank = int(np.count_nonzero(eigenvalues > tolerance))
            sign, logdet = np.linalg.slogdet(matrix)
            valid = bool(
                rank == HAND_EYE_DIMENSION
                and sign > 0.0
                and np.isfinite(logdet)
                and eigenvalues[0] > 0.0
            )
            condition = (
                float(eigenvalues[-1] / eigenvalues[0])
                if eigenvalues[0] > 0.0
                else math.inf
            )
            outputs.append(
                InformationMetrics(
                    valid=valid,
                    logdet_H_eff=float(logdet) if valid else -math.inf,
                    lambda_min=float(eigenvalues[0]),
                    lambda_max=float(eigenvalues[-1]),
                    condition_number=condition,
                    rank=rank,
                    eigenvalues=tuple(map(float, eigenvalues)),
                    reason="" if valid else "rank_or_positive_definiteness",
                )
            )
        return outputs

    def metrics(self, matrix: np.ndarray) -> InformationMetrics:
        return self.metrics_batch(np.asarray(matrix)[None, :, :])[0]

    def score_batch(self, matrices: np.ndarray) -> np.ndarray:
        effective, nuisance_valid, _reasons = self.effective_batch(matrices)
        scores = np.full(len(effective), -math.inf, dtype=float)
        indices = np.flatnonzero(
            nuisance_valid & np.all(np.isfinite(effective), axis=(1, 2))
        )
        if len(indices) == 0:
            return scores
        matrices_valid = effective[indices]
        eigenvalues = np.linalg.eigvalsh(matrices_valid)
        scale = np.maximum(np.max(np.abs(eigenvalues), axis=1), 1.0)
        tolerance = self.rank_relative_tolerance * scale
        ranks = np.count_nonzero(eigenvalues > tolerance[:, None], axis=1)
        signs, logdets = np.linalg.slogdet(matrices_valid)
        valid = (
            (ranks == HAND_EYE_DIMENSION)
            & (signs > 0.0)
            & np.isfinite(logdets)
            & (eigenvalues[:, 0] > 0.0)
        )
        scores[indices[valid]] = logdets[valid]
        return scores


def sample_random_baselines(
    *,
    information: np.ndarray,
    subset_size: int,
    count: int,
    rng: np.random.Generator,
    scorer: InformationScorer,
    chunk_size: int,
) -> list[BaselineResult]:
    results: list[BaselineResult] = []
    candidate_count = len(information)
    for start in range(0, count, chunk_size):
        batch_count = min(chunk_size, count - start)
        subsets = np.empty((batch_count, subset_size), dtype=int)
        for row in range(batch_count):
            subsets[row] = rng.choice(candidate_count, size=subset_size, replace=False)
        matrices = np.sum(information[subsets], axis=1)
        metrics = scorer.metrics_batch(matrices)
        for row, metric in enumerate(metrics):
            results.append(
                BaselineResult(
                    subset_id=start + row,
                    selected_indices=tuple(sorted(map(int, subsets[row]))),
                    metrics=metric,
                )
            )
    return results


def _best_full_rank_random_subset(
    *,
    information: np.ndarray,
    subset_size: int,
    rng: np.random.Generator,
    scorer: InformationScorer,
    attempts: int,
) -> tuple[int, ...]:
    best: tuple[int, ...] | None = None
    best_score = -math.inf
    for _ in range(attempts):
        indices = tuple(
            sorted(map(int, rng.choice(len(information), size=subset_size, replace=False)))
        )
        score = scorer.metrics(np.sum(information[list(indices)], axis=0)).logdet_H_eff
        if score > best_score:
            best = indices
            best_score = score
        if np.isfinite(score):
            return indices
    if best is None or not np.isfinite(best_score):
        raise RuntimeError(
            f"failed to find a full-rank random initialization for N={subset_size}"
        )
    return best


def one_exchange_search(
    *,
    information: np.ndarray,
    initial_indices: Sequence[int],
    scorer: InformationScorer,
    start_id: int,
    max_iterations: int,
    chunk_size: int,
    improvement_tolerance: float,
) -> SearchResult:
    selected = np.asarray(sorted(map(int, initial_indices)), dtype=int)
    if len(np.unique(selected)) != len(selected):
        raise ValueError("initial subset must contain unique candidates")
    current_H = np.sum(information[selected], axis=0)
    initial_score = scorer.metrics(current_H).logdet_H_eff
    if not np.isfinite(initial_score):
        raise ValueError("initial subset is not full-rank")
    current_score = initial_score
    evaluated = 0
    iterations = 0

    for _iteration in range(max_iterations):
        selected_mask = np.zeros(len(information), dtype=bool)
        selected_mask[selected] = True
        unselected = np.flatnonzero(~selected_mask)
        best_score = current_score
        best_remove_position = -1
        best_add = -1

        for remove_position, remove_index in enumerate(selected):
            reduced = current_H - information[remove_index]
            for start in range(0, len(unselected), chunk_size):
                add_indices = unselected[start : start + chunk_size]
                trial_matrices = reduced[None, :, :] + information[add_indices]
                scores = scorer.score_batch(trial_matrices)
                evaluated += len(add_indices)
                local_position = int(np.argmax(scores))
                local_score = float(scores[local_position])
                local_add = int(add_indices[local_position])
                if (
                    local_score > best_score + improvement_tolerance
                    or (
                        abs(local_score - best_score) <= improvement_tolerance
                        and best_add >= 0
                        and (int(remove_index), local_add)
                        < (int(selected[best_remove_position]), best_add)
                    )
                ):
                    best_score = local_score
                    best_remove_position = remove_position
                    best_add = local_add

        if best_remove_position < 0:
            break
        remove_index = int(selected[best_remove_position])
        current_H = current_H - information[remove_index] + information[best_add]
        selected[best_remove_position] = best_add
        selected.sort()
        current_score = best_score
        iterations += 1

    return SearchResult(
        start_id=int(start_id),
        initial_indices=tuple(sorted(map(int, initial_indices))),
        selected_indices=tuple(map(int, selected)),
        initial_score=float(initial_score),
        final_score=float(current_score),
        iterations=iterations,
        evaluated_exchanges=evaluated,
    )


def multi_start_search(
    *,
    information: np.ndarray,
    subset_size: int,
    random_starts: int,
    rng: np.random.Generator,
    scorer: InformationScorer,
    args: argparse.Namespace,
    seeded_initial_subset: Sequence[int] | None = None,
) -> list[SearchResult]:
    results: list[SearchResult] = []
    for start_id in range(random_starts):
        if start_id == 0 and seeded_initial_subset is not None:
            initial = tuple(sorted(map(int, seeded_initial_subset)))
        else:
            initial = _best_full_rank_random_subset(
                information=information,
                subset_size=subset_size,
                rng=rng,
                scorer=scorer,
                attempts=args.initialization_attempts,
            )
        result = one_exchange_search(
            information=information,
            initial_indices=initial,
            scorer=scorer,
            start_id=start_id,
            max_iterations=args.max_exchange_iterations,
            chunk_size=args.exchange_chunk_size,
            improvement_tolerance=float(args.improvement_tolerance),
        )
        results.append(result)
        print(
            f"    start {start_id + 1:>3}/{random_starts}: "
            f"{result.initial_score:.6f} -> {result.final_score:.6f} "
            f"({result.iterations} exchanges)",
            flush=True,
        )
    return results


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: Sequence[str] | None = None) -> Path:
    materialized = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        if not materialized:
            raise ValueError("fieldnames are required for an empty CSV")
        fieldnames_list = list(materialized[0])
        known = set(fieldnames_list)
        for row in materialized[1:]:
            for name in row:
                if name not in known:
                    fieldnames_list.append(name)
                    known.add(name)
        fieldnames = fieldnames_list
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(materialized)
    return path


def _json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(_json_value(payload), file, indent=2, sort_keys=True)
        file.write("\n")
    return path


def candidate_row(candidate: Candidate) -> dict[str, Any]:
    row: dict[str, Any] = {
        "scan_id": candidate.candidate_id,
        "theta_deg": candidate.theta_deg,
        "azimuth_deg": candidate.azimuth_deg,
        "normal_azimuth_sensor_deg": candidate.normal_azimuth_sensor_deg,
        "distance_mm": candidate.distance_mm,
        "board_u_mm": candidate.target_u_mm,
        "board_v_mm": candidate.target_v_mm,
        "normal_sensor_x": candidate.plane_normal_sensor[0],
        "normal_sensor_y": candidate.plane_normal_sensor[1],
        "normal_sensor_z": candidate.plane_normal_sensor[2],
        "view_normal_base_x": candidate.view_normal_base[0],
        "view_normal_base_y": candidate.view_normal_base[1],
        "view_normal_base_z": candidate.view_normal_base[2],
        "sensor_origin_base_x_mm": candidate.T_base_s[0, 3],
        "sensor_origin_base_y_mm": candidate.T_base_s[1, 3],
        "sensor_origin_base_z_mm": candidate.T_base_s[2, 3],
        "profile_z_min_mm": candidate.feasibility.profile_z_min_mm,
        "profile_z_max_mm": candidate.feasibility.profile_z_max_mm,
        "profile_local_u_min_mm": candidate.feasibility.profile_local_u_min_mm,
        "profile_local_u_max_mm": candidate.feasibility.profile_local_u_max_mm,
        "profile_local_v_min_mm": candidate.feasibility.profile_local_v_min_mm,
        "profile_local_v_max_mm": candidate.feasibility.profile_local_v_max_mm,
    }
    quaternion = Rotation.from_matrix(candidate.T_base_s[:3, :3]).as_quat()
    for name, value in zip(("qx", "qy", "qz", "qw"), quaternion):
        row[f"sensor_{name}"] = float(value)
    return row


def metrics_row(metrics: InformationMetrics) -> dict[str, Any]:
    row = {
        "logdet_H_eff": metrics.logdet_H_eff,
        "lambda_min": metrics.lambda_min,
        "lambda_max": metrics.lambda_max,
        "condition_number": metrics.condition_number,
        "rank": metrics.rank,
        "valid": metrics.valid,
        "invalid_reason": metrics.reason,
    }
    row.update(
        {f"eigenvalue_{index + 1}": value for index, value in enumerate(metrics.eigenvalues)}
    )
    return row


def geometry_diagnostics(
    selected: Sequence[Candidate],
    ranges: dict[str, tuple[float, float]],
) -> dict[str, Any]:
    normals = np.stack([candidate.view_normal_base for candidate in selected])
    moment = normals.T @ normals / len(normals)
    moment_eigenvalues = np.linalg.eigvalsh(moment)
    moment_condition = (
        float(moment_eigenvalues[-1] / moment_eigenvalues[0])
        if moment_eigenvalues[0] > 0.0
        else math.inf
    )
    azimuth = np.deg2rad([candidate.azimuth_deg for candidate in selected])
    harmonics = {
        f"azimuth_harmonic_{order}": float(
            abs(np.mean(np.exp(1j * order * azimuth)))
        )
        for order in (1, 2, 3)
    }
    pairwise_angles: list[float] = []
    for first in range(len(normals)):
        for second in range(first + 1, len(normals)):
            cosine = float(np.clip(normals[first] @ normals[second], -1.0, 1.0))
            pairwise_angles.append(float(np.degrees(np.arccos(cosine))))

    boundary: dict[str, dict[str, float | int]] = {}
    candidate_attributes = {
        "target_u_mm": "target_u_mm",
        "target_v_mm": "target_v_mm",
        "distance_mm": "distance_mm",
        "theta_deg": "theta_deg",
        "azimuth_deg": "azimuth_deg",
        "normal_azimuth_sensor_deg": "normal_azimuth_sensor_deg",
    }
    for name, attribute in candidate_attributes.items():
        lower, upper = ranges[name]
        margin = 0.05 * (upper - lower)
        values = np.asarray([getattr(candidate, attribute) for candidate in selected])
        lower_count = int(np.count_nonzero(values <= lower + margin))
        upper_count = int(np.count_nonzero(values >= upper - margin))
        boundary[name] = {
            "lower_count": lower_count,
            "upper_count": upper_count,
            "lower_fraction": lower_count / len(values),
            "upper_fraction": upper_count / len(values),
        }
    return {
        "normal_definition": "sensor -Z viewing axis expressed in robot base",
        "normal_moment_matrix": moment,
        "normal_moment_eigenvalues": moment_eigenvalues,
        "normal_moment_condition_number": moment_condition,
        **harmonics,
        "pairwise_normal_angles_deg": pairwise_angles,
        "boundary_usage_within_5_percent": boundary,
    }


def straight_profile_structural_rank_upper_bound(subset_size: int) -> int:
    """Return the hand-eye rank bound for ideal straight-line profiles.

    Within one scan every Jacobian column is an affine function of profile x,
    so a scan contributes at most two independent residual rows.  Eliminating
    the three single-plane nuisance coordinates leaves rank at most ``2N-3``.
    Consequently N=3 and N=4 cannot satisfy the requested six-DoF rank test,
    irrespective of the pose optimizer.
    """
    return max(0, min(HAND_EYE_DIMENSION, 2 * int(subset_size) - 3))


def save_plots(
    *,
    output_dir: Path,
    best_candidates: dict[int, list[Candidate]],
    summary_rows: Sequence[dict[str, Any]],
) -> None:
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    sizes = sorted(best_candidates)
    color_map = plt.get_cmap("viridis")
    colors = {N: color_map(i / max(len(sizes) - 1, 1)) for i, N in enumerate(sizes)}

    figure, axis = plt.subplots(figsize=(8, 5))
    for N in sizes:
        axis.scatter(
            np.full(N, N), [item.theta_deg for item in best_candidates[N]],
            color=colors[N], alpha=0.8, label=f"N={N}",
        )
    axis.set(xlabel="subset size N", ylabel="tilt theta [deg]", title="Selected tilt distribution")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(plot_dir / "tilt_distribution.png", dpi=180)
    plt.close(figure)

    figure = plt.figure(figsize=(8, 7))
    axis = figure.add_subplot(111, projection="polar")
    for N in sizes:
        angle = np.deg2rad([item.azimuth_deg for item in best_candidates[N]])
        axis.scatter(angle, np.full(N, N), color=colors[N], label=f"N={N}")
    axis.set_title("Selected view azimuth")
    axis.legend(loc="upper left", bbox_to_anchor=(1.05, 1.05))
    figure.tight_layout()
    figure.savefig(plot_dir / "azimuth_polar.png", dpi=180)
    plt.close(figure)

    def scatter_plot(filename: str, x_attr: str, y_attr: str, xlabel: str, ylabel: str, title: str) -> None:
        figure, axis = plt.subplots(figsize=(8, 5))
        for N in sizes:
            items = best_candidates[N]
            axis.scatter(
                [getattr(item, x_attr) for item in items],
                [getattr(item, y_attr) for item in items],
                color=colors[N], label=f"N={N}", alpha=0.8,
            )
        axis.set(xlabel=xlabel, ylabel=ylabel, title=title)
        axis.grid(alpha=0.25)
        axis.legend(ncol=2)
        figure.tight_layout()
        figure.savefig(plot_dir / filename, dpi=180)
        plt.close(figure)

    scatter_plot("tilt_vs_distance.png", "theta_deg", "distance_mm", "tilt [deg]", "distance [mm]", "Tilt versus distance")
    scatter_plot("board_uv.png", "target_u_mm", "target_v_mm", "board u [mm]", "board v [mm]", "Profile-center position on target")
    scatter_plot(
        "tilt_vs_normal_azimuth_sensor.png",
        "theta_deg",
        "normal_azimuth_sensor_deg",
        "tilt [deg]",
        "normal azimuth in sensor XY [deg]",
        "Tilt versus sensor-frame plane-normal azimuth",
    )

    figure = plt.figure(figsize=(8, 7))
    axis = figure.add_subplot(111, projection="3d")
    longitude = np.linspace(0.0, 2.0 * np.pi, 50)
    latitude = np.linspace(0.0, np.pi, 25)
    x = np.outer(np.cos(longitude), np.sin(latitude))
    y = np.outer(np.sin(longitude), np.sin(latitude))
    z = np.outer(np.ones_like(longitude), np.cos(latitude))
    axis.plot_wireframe(x, y, z, color="0.8", linewidth=0.3, alpha=0.5)
    for N in sizes:
        normals = np.stack([item.view_normal_base for item in best_candidates[N]])
        axis.scatter(normals[:, 0], normals[:, 1], normals[:, 2], color=colors[N], label=f"N={N}")
    axis.set(xlabel="base x", ylabel="base y", zlabel="base z", title="Sensor -Z viewing-axis distribution")
    axis.set_box_aspect((1, 1, 1))
    axis.legend(ncol=2)
    figure.tight_layout()
    figure.savefig(plot_dir / "normal_distribution_3d.png", dpi=180)
    plt.close(figure)

    ordered = [
        row
        for row in sorted(summary_rows, key=lambda row: int(row["N"]))
        if np.isfinite(float(row["best_logdet"]))
    ]
    if not ordered:
        return
    N_values = np.asarray([row["N"] for row in ordered], dtype=int)
    objectives = np.asarray([row["best_logdet"] for row in ordered], dtype=float)
    figure, primary = plt.subplots(figsize=(8, 5))
    primary.plot(N_values, objectives, "o-", label="best logdet")
    primary.set(xlabel="subset size N", ylabel="logdet H_eff", title="D-optimal objective versus scan count")
    primary.grid(alpha=0.25)
    if len(N_values) > 1:
        secondary = primary.twinx()
        gains = np.diff(objectives)
        secondary.plot(N_values[1:], gains, "s--", color="tab:orange", label="incremental gain")
        secondary.set_ylabel("incremental logdet gain")
        lines = primary.lines + secondary.lines
        primary.legend(lines, [line.get_label() for line in lines], loc="best")
    figure.tight_layout()
    figure.savefig(plot_dir / "objective_vs_N.png", dpi=180)
    plt.close(figure)


def _add_profile_noise(
    candidate: Candidate,
    noise: np.ndarray,
    noise_axis: str,
    scan_id: int,
) -> LaserScan:
    points = candidate.points_s.copy()
    if noise_axis == "xz":
        points[:, 0] += noise[:, 0]
        points[:, 2] += noise[:, 1]
    else:
        points[:, 2] += noise[:, 1]
    return LaserScan(
        T_base_ef=candidate.T_base_ef,
        points_s=points,
        plane_id=0,
        scan_id=scan_id,
        meta={"candidate_id": candidate.candidate_id, "paired_mc_noise": True},
    )


def run_monte_carlo(
    *,
    args: argparse.Namespace,
    candidates: Sequence[Candidate],
    designs: dict[int, dict[str, tuple[int, ...]]],
    T_ef_s_true: np.ndarray,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[tuple[int, str], dict[str, float]]]:
    rows: list[dict[str, Any]] = []
    if args.mc_trials == 0:
        return rows, {}
    for N in sorted(designs):
        for trial in range(args.mc_trials):
            trial_seed = np.random.SeedSequence([seed, 0x4D43, N, trial])
            noise_rng, init_rng = [np.random.default_rng(child) for child in trial_seed.spawn(2)]
            common_noise = noise_rng.normal(
                0.0,
                float(args.mc_noise_std_mm),
                size=(N, args.profile_points, 2),
            )
            T_init = make_initial_guess(
                reference_angles_deg=None,
                reference_translation_mm=T_ef_s_true[:3, 3],
                reference_rotation=T_ef_s_true[:3, :3],
                rng=init_rng,
                mode="carlson",
                translation_range_mm=float(args.mc_init_translation_mm),
                angle_range_deg=float(args.mc_init_rotation_deg),
                rotation_perturbation="axis_angle",
                translation_perturbation="direction_norm",
            )
            for method, indices in designs[N].items():
                scans = [
                    _add_profile_noise(candidates[index], common_noise[position], args.mc_noise_axis, position)
                    for position, index in enumerate(indices)
                ]
                row: dict[str, Any] = {"N": N, "trial": trial, "method": method}
                try:
                    output_context = (
                        contextlib.nullcontext()
                        if args.mc_verbose
                        else contextlib.redirect_stdout(io.StringIO())
                    )
                    with output_context:
                        result = calibrate_planes(
                            {0: scans},
                            T_init=T_init.copy(),
                            max_iter=int(args.mc_max_iterations),
                            tol=float(args.mc_tolerance),
                            plane_offset_mode=args.mc_plane_offset_mode,
                        )
                    estimate = result.T_ef_s
                    row.update(
                        {
                            "success": bool(result.converged and np.all(np.isfinite(estimate))),
                            "converged": bool(result.converged),
                            "iterations": int(result.iterations),
                            "rotation_error_deg": rot_error_deg(estimate[:3, :3], T_ef_s_true[:3, :3]),
                            "translation_error_mm": float(np.linalg.norm(estimate[:3, 3] - T_ef_s_true[:3, 3])),
                            "residual_rms_mm": mean_self_fitted_plane_rms({0: scans}, estimate),
                            "error": "",
                        }
                    )
                except (ValueError, FloatingPointError, np.linalg.LinAlgError) as error:
                    row.update(
                        {
                            "success": False,
                            "converged": False,
                            "iterations": 0,
                            "rotation_error_deg": math.nan,
                            "translation_error_mm": math.nan,
                            "residual_rms_mm": math.nan,
                            "error": f"{type(error).__name__}: {error}",
                        }
                    )
                rows.append(row)
        print(f"  Monte Carlo N={N}: {args.mc_trials} paired trials complete", flush=True)

    aggregates: dict[tuple[int, str], dict[str, float]] = {}
    for N in sorted(designs):
        for method in designs[N]:
            group = [row for row in rows if row["N"] == N and row["method"] == method]
            successful = [row for row in group if row["success"]]
            aggregate = {"success_rate": len(successful) / len(group)}
            for field in ("rotation_error_deg", "translation_error_mm", "residual_rms_mm"):
                values = np.asarray([row[field] for row in successful], dtype=float)
                aggregate[f"{field}_median"] = float(np.median(values)) if len(values) else math.nan
                aggregate[f"{field}_mean"] = float(np.mean(values)) if len(values) else math.nan
            aggregates[(N, method)] = aggregate
    return rows, aggregates


def finite_difference_jacobian_check(
    *,
    candidate: Candidate,
    T_ef_s_true: np.ndarray,
    plane: PlaneEstimate,
    profile_noise_std_mm: float,
    noise_axis: str,
) -> dict[str, Any]:
    scan = LaserScan(
        T_base_ef=candidate.T_base_ef,
        points_s=candidate.points_s,
        plane_id=0,
        scan_id=candidate.candidate_id,
    )
    residual, analytic = joint_residual_jacobian(
        {0: [scan]}, T_ef_s_true, {0: plane},
        profile_noise_std_mm=profile_noise_std_mm,
        noise_axis=noise_axis,
    )
    numerical = np.empty_like(analytic)
    steps = np.asarray([1e-7] * 3 + [1e-5] * 3 + [1e-7] * 2 + [1e-5])
    for parameter, step in enumerate(steps):
        delta = np.zeros(9, dtype=float)
        delta[parameter] = step
        plus_T, plus_planes = apply_joint_local_update(T_ef_s_true, {0: plane}, delta)
        minus_T, minus_planes = apply_joint_local_update(T_ef_s_true, {0: plane}, -delta)
        plus, _ = joint_residual_jacobian(
            {0: [scan]}, plus_T, plus_planes,
            profile_noise_std_mm=profile_noise_std_mm, noise_axis=noise_axis,
        )
        minus, _ = joint_residual_jacobian(
            {0: [scan]}, minus_T, minus_planes,
            profile_noise_std_mm=profile_noise_std_mm, noise_axis=noise_axis,
        )
        numerical[:, parameter] = (plus - minus) / (2.0 * step)
    difference = analytic - numerical
    denominator = max(float(np.linalg.norm(numerical)), 1e-12)
    relative_error = float(np.linalg.norm(difference) / denominator)
    return {
        "relative_frobenius_error": relative_error,
        "max_absolute_error": float(np.max(np.abs(difference))),
        "passed": bool(relative_error < 2e-5),
        "residual_rms_at_linearization": float(np.sqrt(np.mean(residual**2))),
    }


def run_sanity_checks(
    *,
    scorer: InformationScorer,
    information: np.ndarray,
    best_results: dict[int, SearchResult],
    baseline_results: dict[int, list[BaselineResult]],
    candidates: Sequence[Candidate],
    T_ef_s_true: np.ndarray,
    plane: PlaneEstimate,
    args: argparse.Namespace,
    pool_growth_checks: dict[int, dict[str, Any]],
) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    if not best_results:
        return {
            "status": "no observable subset size was requested",
            "all_executed_checks_passed": True,
        }
    first_N = min(best_results)
    indices = best_results[first_N].selected_indices
    H = np.sum(information[list(indices)], axis=0)
    metrics = scorer.metrics(H)
    reversed_metrics = scorer.metrics(np.sum(information[list(reversed(indices))], axis=0))
    checks["subset_order_invariance"] = {
        "absolute_logdet_difference": abs(metrics.logdet_H_eff - reversed_metrics.logdet_H_eff),
        "passed": bool(np.isclose(metrics.logdet_H_eff, reversed_metrics.logdet_H_eff, rtol=1e-12, atol=1e-10)),
    }
    effective, valid, _ = scorer.effective_batch(np.stack([H, 2.0 * H]))
    scale_error = float(np.linalg.norm(effective[1] - 2.0 * effective[0])) if np.all(valid) else math.inf
    checks["repeated_information_scaling"] = {
        "matrix_error": scale_error,
        "expected_logdet_gain": 6.0 * math.log(2.0),
        "actual_logdet_gain": scorer.metrics(2.0 * H).logdet_H_eff - metrics.logdet_H_eff,
        "passed": bool(scale_error < 1e-7 * max(np.linalg.norm(effective[1]), 1.0)),
    }
    repeated_pose_metrics = scorer.metrics(first_N * information[indices[0]])
    checks["degenerate_repeated_pose_detection"] = {
        "rank": repeated_pose_metrics.rank,
        "valid": repeated_pose_metrics.valid,
        "reason": repeated_pose_metrics.reason,
        "passed": not repeated_pose_metrics.valid,
    }
    symmetry_error = float(np.max(np.abs(effective[0] - effective[0].T)))
    checks["schur_symmetry_and_psd"] = {
        "symmetry_error": symmetry_error,
        "lambda_min": metrics.lambda_min,
        "passed": bool(symmetry_error < 1e-10 and metrics.lambda_min >= -1e-9 * max(metrics.lambda_max, 1.0)),
    }
    checks["analytic_vs_finite_difference_jacobian"] = finite_difference_jacobian_check(
        candidate=candidates[indices[0]],
        T_ef_s_true=T_ef_s_true,
        plane=plane,
        profile_noise_std_mm=float(args.fisher_profile_noise_std_mm),
        noise_axis=args.fisher_noise_axis,
    )
    for N, best in best_results.items():
        valid_baselines = [item for item in baseline_results[N] if item.metrics.valid]
        if not valid_baselines:
            checks[f"optimized_beats_random_N{N}"] = {
                "passed": None,
                "status": "random baseline sample contained no valid subset",
            }
            continue
        random_best = max(item.metrics.logdet_H_eff for item in valid_baselines)
        checks[f"optimized_beats_random_N{N}"] = {
            "optimized_logdet": best.final_score,
            "random_best_logdet": random_best,
            "passed": bool(best.final_score + 1e-9 >= random_best),
        }
    if pool_growth_checks:
        checks["candidate_pool_growth"] = {
            "passed": all(bool(item["passed"]) for item in pool_growth_checks.values()),
            "per_subset_size": pool_growth_checks,
            "prefix_compatibility": "fixed --candidate-batch-size preserves earlier accepted candidates",
        }
    else:
        checks["candidate_pool_growth"] = {
            "passed": None,
            "status": "skipped by option or candidate bank too small",
        }
    checks["all_executed_checks_passed"] = all(
        item.get("passed") is True
        for item in checks.values()
        if item.get("passed") is not None
    )
    return checks


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    ranges = validate_args(args)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    root_seed = np.random.SeedSequence(args.seed)
    geometry_seed, sampler_seed, baseline_seed, search_seed = root_seed.spawn(4)
    geometry_rng = np.random.default_rng(geometry_seed)
    sampler_rng = np.random.default_rng(sampler_seed)
    baseline_rng = np.random.default_rng(baseline_seed)
    search_rng = np.random.default_rng(search_seed)

    frames, common_center, plane_angles_deg = base._make_plane_frames(
        geometry_rng,
        tuple(map(float, args.plane_angle_range_deg)),
        tuple(map(float, args.plane_center_xy_range_mm)),
        tuple(map(float, args.plane_center_z_range_mm)),
    )
    frame = frames[0]
    T_ef_s_true, true_angles_deg, true_translation_mm = sample_random_handeye(geometry_rng)
    plane = PlaneEstimate(frame.n, frame.l)
    x_values = np.linspace(
        -float(args.profile_half_width_mm),
        float(args.profile_half_width_mm),
        int(args.profile_points),
    )

    print(
        f"Generating {args.candidate_count} feasible "
        f"{args.candidate_design} candidates...",
        flush=True,
    )
    candidates, information, generation_counts = build_candidate_bank(
        args=args,
        ranges=ranges,
        rng=sampler_rng,
        frame=frame,
        common_center=common_center,
        T_ef_s_true=T_ef_s_true,
        x_values=x_values,
    )
    print(
        f"  valid={len(candidates)}, attempted={generation_counts['attempted']}, "
        f"acceptance={len(candidates) / generation_counts['attempted']:.3f}",
        flush=True,
    )
    _write_csv(output_dir / "candidate_bank.csv", (candidate_row(item) for item in candidates))
    np.savez_compressed(output_dir / "candidate_information.npz", H_i=information)

    scorer = InformationScorer(
        regularization=args.schur_regularization,
        rank_relative_tolerance=args.rank_relative_tolerance,
        nuisance_relative_tolerance=args.nuisance_relative_tolerance,
    )
    summary_rows: list[dict[str, Any]] = []
    best_results: dict[int, SearchResult] = {}
    baseline_by_N: dict[int, list[BaselineResult]] = {}
    best_candidates_by_N: dict[int, list[Candidate]] = {}
    mc_designs: dict[int, dict[str, tuple[int, ...]]] = {}
    pool_growth_checks: dict[int, dict[str, Any]] = {}

    for N in sorted(set(map(int, args.subset_sizes))):
        print(f"\n[N={N}] sampling random baseline...", flush=True)
        baselines = sample_random_baselines(
            information=information,
            subset_size=N,
            count=args.random_baseline_subsets,
            rng=baseline_rng,
            scorer=scorer,
            chunk_size=args.exchange_chunk_size,
        )
        baseline_by_N[N] = baselines
        _write_csv(
            output_dir / f"random_baseline_N{N}.csv",
            (
                {
                    "subset_id": item.subset_id,
                    **metrics_row(item.metrics),
                    "selected_indices": json.dumps(item.selected_indices),
                }
                for item in baselines
            ),
        )
        valid_baselines = [item for item in baselines if item.metrics.valid]
        structural_rank_bound = straight_profile_structural_rank_upper_bound(N)
        if structural_rank_bound < HAND_EYE_DIMENSION:
            empty_subset_fields = ("selection_order", *candidate_row(candidates[0]).keys())
            _write_csv(
                output_dir / f"best_subset_N{N}.csv", (),
                fieldnames=empty_subset_fields,
            )
            _write_csv(
                output_dir / f"search_starts_N{N}.csv", (),
                fieldnames=(
                    "start_id", "initial_score", "final_score", "iterations",
                    "evaluated_exchanges", "initial_indices", "selected_indices",
                ),
            )
            observed_rank = max((item.metrics.rank for item in baselines), default=0)
            status = (
                f"structurally unobservable: straight profiles give rank <= "
                f"2N-3={structural_rank_bound} < 6"
            )
            _write_json(
                output_dir / f"best_subset_N{N}.json",
                {
                    "N": N,
                    "status": status,
                    "selected_indices": [],
                    "maximum_observed_random_rank": observed_rank,
                },
            )
            summary_rows.append(
                {
                    "N": N,
                    "status": "structurally_unobservable",
                    "candidate_count": len(candidates),
                    "best_logdet": -math.inf,
                    "rank": observed_rank,
                    "structural_rank_upper_bound": structural_rank_bound,
                    "lambda_min": math.nan,
                    "lambda_max": math.nan,
                    "condition_number": math.inf,
                    "random_starts": 0,
                    "unique_local_optima": 0,
                    "best_exchange_iterations": 0,
                    "random_valid_count": 0,
                    "random_median_logdet": -math.inf,
                    "random_p90_logdet": -math.inf,
                    "random_best_logdet": -math.inf,
                }
            )
            print(f"[N={N}] {status}; D-optimal search skipped.", flush=True)
            continue

        if valid_baselines:
            valid_baselines.sort(key=lambda item: item.metrics.logdet_H_eff)
            random_best = valid_baselines[-1]
            baseline_scores = np.asarray(
                [item.metrics.logdet_H_eff for item in valid_baselines]
            )
            median_score = float(np.median(baseline_scores))
            random_median = min(
                valid_baselines,
                key=lambda item: abs(item.metrics.logdet_H_eff - median_score),
            )
            seeded_subset: tuple[int, ...] | None = random_best.selected_indices
        else:
            # A very small requested baseline sample can miss a valid subset.
            # Search initialization still gets its own larger attempt budget.
            seeded_subset = _best_full_rank_random_subset(
                information=information,
                subset_size=N,
                rng=search_rng,
                scorer=scorer,
                attempts=args.initialization_attempts,
            )
            seeded_metrics = scorer.metrics(
                np.sum(information[list(seeded_subset)], axis=0)
            )
            random_best = BaselineResult(-1, seeded_subset, seeded_metrics)
            random_median = random_best
            baseline_scores = np.asarray([seeded_metrics.logdet_H_eff])

        print(f"[N={N}] running {args.random_starts} one-exchange starts...", flush=True)
        starts = multi_start_search(
            information=information,
            subset_size=N,
            random_starts=args.random_starts,
            rng=search_rng,
            scorer=scorer,
            args=args,
            seeded_initial_subset=seeded_subset,
        )
        prefix_count = max(N, len(information) // 2)
        if not args.skip_pool_growth_check and prefix_count < len(information):
            prefix_initial = _best_full_rank_random_subset(
                information=information[:prefix_count],
                subset_size=N,
                rng=search_rng,
                scorer=scorer,
                attempts=args.initialization_attempts,
            )
            prefix_result = one_exchange_search(
                information=information[:prefix_count],
                initial_indices=prefix_initial,
                scorer=scorer,
                start_id=-1,
                max_iterations=args.max_exchange_iterations,
                chunk_size=args.exchange_chunk_size,
                improvement_tolerance=float(args.improvement_tolerance),
            )
            full_from_prefix = one_exchange_search(
                information=information,
                initial_indices=prefix_result.selected_indices,
                scorer=scorer,
                start_id=args.random_starts,
                max_iterations=args.max_exchange_iterations,
                chunk_size=args.exchange_chunk_size,
                improvement_tolerance=float(args.improvement_tolerance),
            )
            starts.append(full_from_prefix)
            pool_growth_checks[N] = {
                "prefix_candidate_count": prefix_count,
                "full_candidate_count": len(information),
                "prefix_best_logdet": prefix_result.final_score,
                "full_bank_logdet_from_prefix_start": full_from_prefix.final_score,
                "passed": bool(
                    full_from_prefix.final_score + float(args.improvement_tolerance)
                    >= prefix_result.final_score
                ),
            }
            print(
                f"    pool audit {prefix_count}->{len(information)}: "
                f"{prefix_result.final_score:.6f} -> {full_from_prefix.final_score:.6f}",
                flush=True,
            )
        best = max(starts, key=lambda item: (item.final_score, tuple(-v for v in item.selected_indices)))
        best_results[N] = best
        best_metrics = scorer.metrics(np.sum(information[list(best.selected_indices)], axis=0))
        selected_candidates = [candidates[index] for index in best.selected_indices]
        best_candidates_by_N[N] = selected_candidates
        unique_optima = len({item.selected_indices for item in starts})
        geometry = geometry_diagnostics(selected_candidates, ranges)

        _write_csv(
            output_dir / f"best_subset_N{N}.csv",
            (
                {"selection_order": order, **candidate_row(candidate)}
                for order, candidate in enumerate(selected_candidates)
            ),
        )
        _write_csv(
            output_dir / f"search_starts_N{N}.csv",
            (
                {
                    "start_id": item.start_id,
                    "initial_score": item.initial_score,
                    "final_score": item.final_score,
                    "iterations": item.iterations,
                    "evaluated_exchanges": item.evaluated_exchanges,
                    "initial_indices": json.dumps(item.initial_indices),
                    "selected_indices": json.dumps(item.selected_indices),
                }
                for item in starts
            ),
        )
        pairwise = geometry["pairwise_normal_angles_deg"]
        _write_csv(
            output_dir / f"pairwise_normal_angles_N{N}.csv",
            ({"pair_id": index, "angle_deg": angle} for index, angle in enumerate(pairwise)),
            fieldnames=("pair_id", "angle_deg"),
        )
        _write_json(
            output_dir / f"best_subset_N{N}.json",
            {
                "N": N,
                "selected_indices": best.selected_indices,
                "information_metrics": asdict(best_metrics),
                "geometry_diagnostics": geometry,
            },
        )
        summary_rows.append(
            {
                "N": N,
                "status": "valid",
                "candidate_count": len(candidates),
                "best_logdet": best_metrics.logdet_H_eff,
                "rank": best_metrics.rank,
                "lambda_min": best_metrics.lambda_min,
                "lambda_max": best_metrics.lambda_max,
                "condition_number": best_metrics.condition_number,
                "random_starts": args.random_starts,
                "total_search_starts_including_pool_audit": len(starts),
                "unique_local_optima": unique_optima,
                "best_exchange_iterations": best.iterations,
                "random_valid_count": len(valid_baselines),
                "random_median_logdet": float(np.median(baseline_scores)),
                "random_p90_logdet": float(np.percentile(baseline_scores, 90.0)),
                "random_best_logdet": float(np.max(baseline_scores)),
                "azimuth_harmonic_1": geometry["azimuth_harmonic_1"],
                "azimuth_harmonic_2": geometry["azimuth_harmonic_2"],
                "normal_moment_condition_number": geometry["normal_moment_condition_number"],
                **{f"eigenvalue_{index + 1}": value for index, value in enumerate(best_metrics.eigenvalues)},
            }
        )
        mc_designs[N] = {
            "d_optimal": best.selected_indices,
            "random_median": random_median.selected_indices,
        }
        print(
            f"[N={N}] BEST logdet={best_metrics.logdet_H_eff:.6f}, "
            f"rank={best_metrics.rank}, lambda_min={best_metrics.lambda_min:.6g}, "
            f"cond={best_metrics.condition_number:.6g}, unique_optima={unique_optima}",
            flush=True,
        )
        for candidate in selected_candidates:
            print(
                f"    id={candidate.candidate_id:>4} theta={candidate.theta_deg:7.3f} "
                f"az={candidate.azimuth_deg:8.3f} "
                f"normal_az_s={candidate.normal_azimuth_sensor_deg:8.3f} "
                f"d={candidate.distance_mm:7.3f} u={candidate.target_u_mm:7.3f} "
                f"v={candidate.target_v_mm:7.3f}",
                flush=True,
            )

    sanity = run_sanity_checks(
        scorer=scorer,
        information=information,
        best_results=best_results,
        baseline_results=baseline_by_N,
        candidates=candidates,
        T_ef_s_true=T_ef_s_true,
        plane=plane,
        args=args,
        pool_growth_checks=pool_growth_checks,
    )
    _write_json(output_dir / "sanity_checks.json", sanity)
    if not sanity["all_executed_checks_passed"]:
        raise RuntimeError(f"one or more sanity checks failed; see {output_dir / 'sanity_checks.json'}")

    print(f"\nRunning {args.mc_trials} paired Monte-Carlo trials per N...", flush=True)
    mc_rows, mc_aggregates = run_monte_carlo(
        args=args,
        candidates=candidates,
        designs=mc_designs,
        T_ef_s_true=T_ef_s_true,
        seed=args.seed,
    )
    if mc_rows:
        _write_csv(output_dir / "monte_carlo_trials.csv", mc_rows)
        _write_csv(
            output_dir / "monte_carlo_summary.csv",
            (
                {"N": N, "method": method, **values}
                for (N, method), values in sorted(mc_aggregates.items())
            ),
        )
        for row in summary_rows:
            N = int(row["N"])
            for method in ("d_optimal", "random_median"):
                aggregate = mc_aggregates[(N, method)]
                for name, value in aggregate.items():
                    row[f"mc_{method}_{name}"] = value

    _write_csv(output_dir / "summary.csv", summary_rows)
    if best_candidates_by_N and not args.skip_plots:
        save_plots(
            output_dir=output_dir,
            best_candidates=best_candidates_by_N,
            summary_rows=summary_rows,
        )

    elapsed = time.perf_counter() - started
    config = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "arguments": vars(args),
        "pose_ranges": ranges,
        "pose_convention": {
            "u_v": "sensor +Z axis / target-plane intersection coordinates",
            "distance": "sensor origin to axis/plane intersection distance",
            "tilt": "angle(sensor -Z, oriented target-plane normal)",
            "azimuth": "target-plane azimuth of projected sensor +Z",
            "roll": "signed sensor +Z roll from projected target +U to sensor +X",
        },
        "plane": {
            "normal_base": frame.n,
            "offset_mm": frame.l,
            "common_center_base_mm": common_center,
            "frame_angles_deg": plane_angles_deg,
        },
        "ground_truth_handeye": {
            "T_ef_s": T_ef_s_true,
            "euler_xyz_deg": true_angles_deg,
            "translation_mm": true_translation_mm,
        },
        "candidate_generation": generation_counts,
        "information": {
            "joint_parameter_order": [
                "handeye_right_rotation_xyz_rad",
                "handeye_right_translation_xyz_mm",
                "plane_normal_tangent_uv_rad",
                "plane_offset_mm",
            ],
            "weighting": (
                "existing geometry-dependent profile-noise whitening from "
                "laser_handeye.active_fisher.joint_residual_jacobian"
            ),
            "parameter_scales": [1.0] * 9,
            "objective": "logdet(H_eff)",
            "plane_elimination": "solve(H_pp, H_ph)",
        },
        "legacy_baseline": {
            "status": "not_run",
            "reason": "no existing pose set with the same six-dimensional ranges was supplied",
        },
        "elapsed_seconds": elapsed,
    }
    _write_json(output_dir / "run_config.json", config)
    print(
        f"\nDone in {elapsed:.1f} s. Results: {output_dir}\n"
        f"Sanity checks passed: {sanity['all_executed_checks_passed']}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
