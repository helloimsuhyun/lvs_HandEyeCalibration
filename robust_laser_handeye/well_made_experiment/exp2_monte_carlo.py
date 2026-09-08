"""
Monte-Carlo comparison of measurement budget vs pose-design geometry.

Designs
-------
GLOBAL_OPTIMAL uses the same optimized 12-pose geometry repeatedly:

    OPTIMAL-12   = 1 x 12 poses
    OPTIMAL-24   = 2 x 12 poses
    OPTIMAL-48   = 4 x 12 poses
    OPTIMAL-96   = 8 x 12 poses
    OPTIMAL-156  = 13 x 12 poses

PAPER uses the full two-theta paper-style radial design:

    9 radial lines
    x 3 distances
    x 2 theta values
    x 3 theta-adaptive beta values
    = 162 poses

The repeated GLOBAL_OPTIMAL measurements use the SAME 12 physical poses but
each repeated scan receives an independent measurement-noise sample.

Within a Monte-Carlo trial all designs use the same:
    - GT hand-eye transform
    - initial hand-eye estimate
    - noise sigma
    - solver settings
    - shared target radius
    - trial-level RNG seed

For the GLOBAL_OPTIMAL budget curve, every larger budget is nested:
OPTIMAL-24 contains the same first 12 noisy scans as OPTIMAL-12 plus 12 new
independent scans, OPTIMAL-48 contains the same first 24 plus 24 new scans,
and so on. This isolates the effect of repeated acquisition count.

Run
---
cd ~/lvs_HandEyeCalibration

PYTHONPATH=. python \
    robust_laser_handeye/well_made_experiment/exp2_monte_carlo.py \
    --trials 500

Expected exp2.py interface
--------------------------
This script expects exp2.py to contain:
    make_theta_adaptive_pose_grid
    make_uniform_radial_line_specs
    make_pose_parameters_from_paper_lines
    make_sensor_poses
    make_scans

Outputs
-------
    config.json
    trials.csv
    summary.csv
    summary.json
    final_error_boxplots.png
    final_error_boxplots_without_large_outliers.png
    success_rates.png
    error_vs_pose_count.png
    accuracy_vs_pose_count.png
    opt12_vs_paper162_fisher_information.png
"""

from __future__ import annotations

import argparse
import contextlib
import csv
from dataclasses import asdict, dataclass
import io
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from robust_laser_handeye.laser_handeye.calibration import (
    calibrate_single_plane_with_nonlinear,
)

from robust_laser_handeye.laser_handeye.active_fisher import (
    fit_plane_estimates,
    joint_residual_jacobian,
    marginal_handeye_information,
)

from robust_laser_handeye.laser_handeye.simulation import (
    sample_random_handeye,
    simulate_scan_from_sensor_pose,
)

from robust_laser_handeye.well_made_experiment import (
    exp1_optimal as exp1,
)

from robust_laser_handeye.well_made_experiment import (
    exp2,
)


# =============================================================================
# TOP-LEVEL POSE CONFIGURATION
# =============================================================================

# Shared by BOTH designs.
SHARED_RADIUS_MM = 70.0

# -----------------------------------------------------------------------------
# EXP1 GLOBAL_OPTIMAL 12-pose geometry
# -----------------------------------------------------------------------------

EXP1_DISTANCE_NEAR_MM = 65.0
EXP1_DISTANCE_FAR_MM = 90.0

EXP1_TILT_MIN_DEG = 5.0
EXP1_TILT_MAX_DEG = 35.0

# Repeat the exact same optimized 12 physical poses this many times.
#
#   1  ->  12 poses
#   2  ->  24 poses
#   4  ->  48 poses
#   8  ->  96 poses
#   13 -> 156 poses
GLOBAL_REPEAT_COUNTS = (
    1,
    2,
    4,
    8,
    13,
)

# -----------------------------------------------------------------------------
# PAPER radial design
# -----------------------------------------------------------------------------
#
# For every theta, exp2.make_theta_adaptive_pose_grid() generates:
#
#     beta = {90-theta, 90, 90+theta}
#
# Therefore:
#     theta=5  -> beta={85,90,95}
#     theta=35 -> beta={55,90,125}
#
# Current count:
#     3 d x 2 theta x 3 beta x 9 lines = 162 poses.
# -----------------------------------------------------------------------------

PAPER_DISTANCES_MM = (
    60.0,
    75.0,
    90.0,
)

PAPER_THETAS_DEG = (
    5.0,
    35.0,
)

PAPER_LINE_COUNT = 9
PAPER_LINE_STEP_DEG = 40.0

paper_poses = exp2.make_theta_adaptive_pose_grid(
    distances_mm=PAPER_DISTANCES_MM,
    thetas_deg=PAPER_THETAS_DEG,
)


# =============================================================================
# Design names
# =============================================================================

BASE_GLOBAL_POSE_COUNT = 12

GLOBAL_DESIGNS = tuple(
    f"GLOBAL_OPTIMAL_{BASE_GLOBAL_POSE_COUNT * repeat_count}"
    for repeat_count in GLOBAL_REPEAT_COUNTS
)

DESIGN_PAPER = "PAPER_RADIAL_162"

DESIGN_ORDER = (
    *GLOBAL_DESIGNS,
    DESIGN_PAPER,
)


# =============================================================================
# Shared Monte-Carlo protocol
# =============================================================================

MEDIUM_INIT_TRANSLATION_MM = 100.0
MEDIUM_INIT_ROTATION_DEG = 15.0

DEFAULT_NOISE_STD_MM = 0.25

# Common nondimensionalization used for every design before comparing FIMs.
# These are coordinate scales, NOT information priors.
FIM_ROTATION_SCALE_DEG = 2.0
FIM_TRANSLATION_SCALE_MM = 10.0
FIM_PLANE_NORMAL_SCALE_DEG = 20.0
FIM_PLANE_OFFSET_SCALE_MM = 100.0

GT_TRANSLATION_COMPONENT_RANGE_MM = (
    -100.0,
    200.0,
)

GT_EULER_COMPONENT_RANGE_DEG = (
    -180.0,
    180.0,
)

LARGE_TRANSLATION_OUTLIER_MM = 10.0
LARGE_ROTATION_OUTLIER_DEG = 5.0

PLOT_EXCLUSION_TRANSLATION_MM = 5.0
PLOT_EXCLUSION_ROTATION_DEG = 5.0

ACCURACY_TRANSLATION_THRESHOLD_MM = 1.0
ACCURACY_ROTATION_THRESHOLD_DEG = 0.25


# =============================================================================
# Result
# =============================================================================

@dataclass(frozen=True)
class TrialResult:
    trial: int

    design: str
    design_family: str

    pose_count: int
    repeat_count: int

    radius_mm: float

    initial_condition: str

    init_rotation_limit_deg: float
    init_translation_limit_mm: float
    noise_std_mm: float

    actual_init_rotation_error_deg: float
    actual_init_translation_error_mm: float

    gt_euler_x_deg: float
    gt_euler_y_deg: float
    gt_euler_z_deg: float

    gt_translation_x_mm: float
    gt_translation_y_mm: float
    gt_translation_z_mm: float

    solver_success: bool
    accuracy_success: bool

    translation_error_mm: float
    rotation_error_deg: float
    final_rms_mm: float

    # Marginal hand-eye Fisher information metrics (6 x 6 after Schur complement).
    d_optimal_half_logdet: float
    d_optimal_count_normalized: float
    e_optimal_min_eigenvalue: float
    e_optimal_count_normalized: float
    fim_condition_number: float
    fim_logdet: float

    error_message: str


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Monte-Carlo measurement-budget comparison: "
            "GLOBAL_OPTIMAL 12/24/48/96/156 vs PAPER 162."
        )
    )

    parser.add_argument(
        "--trials",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=20260904,
    )

    parser.add_argument(
        "--branch-mode",
        choices=(
            "alternating",
            "positive",
            "negative",
        ),
        default="alternating",
        help=(
            "Mirror branch policy for the PAPER design. "
            "Default: %(default)s"
        ),
    )

    parser.add_argument(
        "--max-iter",
        type=int,
        default=3000,
    )

    parser.add_argument(
        "--tol",
        type=float,
        default=1e-5,
    )

    parser.add_argument(
        "--nonlinear-max-nfev",
        type=int,
        default=300,
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "runs/exp2_monte_carlo_budget_curve"
        ),
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Also display plots interactively.",
    )

    parser.add_argument(
        "--verbose-solver",
        action="store_true",
        help="Print iterative solver output for every calibration.",
    )

    args = parser.parse_args()

    if args.trials <= 0:
        parser.error(
            "--trials must be positive"
        )

    if (
        args.max_iter <= 0
        or args.nonlinear_max_nfev <= 0
    ):
        parser.error(
            "iteration limits must be positive"
        )

    if (
        not np.isfinite(args.tol)
        or args.tol <= 0.0
    ):
        parser.error(
            "--tol must be finite and positive"
        )

    return args


# =============================================================================
# Shared geometry
# =============================================================================

def build_shared_frame():
    return exp1.make_plane_frame(
        exp1.PLANE_NORMAL,
        exp1.BOARD_CENTER,
    )


def validate_shared_profile_geometry() -> None:
    """Ensure both designs use the same simulated laser profile sampling."""

    x1 = np.asarray(
        exp1.X_VALUES,
        dtype=float,
    )

    x2 = np.asarray(
        exp2.X_VALUES,
        dtype=float,
    )

    if (
        x1.shape != x2.shape
        or not np.allclose(
            x1,
            x2,
            atol=1e-12,
            rtol=0.0,
        )
    ):
        raise RuntimeError(
            "exp1.X_VALUES and exp2.X_VALUES differ; "
            "Monte-Carlo comparison would not use the same profile sampling."
        )


# =============================================================================
# GLOBAL_OPTIMAL base 12-pose geometry
# =============================================================================

def build_global_base_design(
    *,
    frame,
) -> dict[str, Any]:

    # exp1.make_scan_pose_parameters() reads exp1.POSE_SUPPORTS internally.
    # Override it using this experiment's top-level configuration.
    exp1.POSE_SUPPORTS = (
        (
            float(EXP1_TILT_MIN_DEG),
            float(EXP1_DISTANCE_NEAR_MM),
        ),
        (
            float(EXP1_TILT_MIN_DEG),
            float(EXP1_DISTANCE_FAR_MM),
        ),
        (
            float(EXP1_TILT_MAX_DEG),
            float(EXP1_DISTANCE_NEAR_MM),
        ),
        (
            float(EXP1_TILT_MAX_DEG),
            float(EXP1_DISTANCE_FAR_MM),
        ),
    )

    uv = exp1.circular_uv_points(
        radius_mm=float(
            SHARED_RADIUS_MM
        ),
        N=exp1.N,
    )

    target_points = (
        exp1.plane_uv_to_base_points(
            uv=uv,
            target_center_base_mm=(
                exp1.BOARD_CENTER
            ),
            frame=frame,
        )
    )

    support_ids = (
        exp1.make_support_ids()
    )

    beta_optimal = (
        exp1.make_optimal_beta(
            support_ids
        )
    )

    pose_params = (
        exp1.make_scan_pose_parameters(
            uv=uv,
            beta_deg=beta_optimal,
        )
    )

    sensor_poses = (
        exp1.make_sensor_poses(
            pose_params=pose_params,
            target_points_base=(
                target_points
            ),
            frame=frame,
        )
    )

    if len(sensor_poses) != BASE_GLOBAL_POSE_COUNT:
        raise RuntimeError(
            "Expected the GLOBAL_OPTIMAL base design to contain "
            f"{BASE_GLOBAL_POSE_COUNT} poses, got {len(sensor_poses)}"
        )

    return {
        "pose_params": pose_params,
        "sensor_poses": sensor_poses,
        "uv": uv,
        "support_ids": support_ids,
        "beta_optimal": beta_optimal,
    }


def build_global_budget_designs(
    *,
    base_design: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Repeat the same 12 physical sensor poses for each requested budget."""

    base_sensor_poses = list(
        base_design[
            "sensor_poses"
        ]
    )

    designs: dict[
        str,
        dict[str, Any],
    ] = {}

    for repeat_count in GLOBAL_REPEAT_COUNTS:

        repeat_count = int(
            repeat_count
        )

        if repeat_count <= 0:
            raise ValueError(
                "GLOBAL_REPEAT_COUNTS must contain positive integers"
            )

        pose_count = (
            BASE_GLOBAL_POSE_COUNT
            * repeat_count
        )

        design_name = (
            f"GLOBAL_OPTIMAL_{pose_count}"
        )

        # List repetition intentionally means:
        # same physical geometry, new measurement at every occurrence.
        sensor_poses = (
            base_sensor_poses
            * repeat_count
        )

        if len(sensor_poses) != pose_count:
            raise RuntimeError(
                "GLOBAL repeated pose count mismatch"
            )

        designs[
            design_name
        ] = {
            "kind": "global",
            "design_family": "GLOBAL_OPTIMAL",
            "repeat_count": repeat_count,
            "pose_count": pose_count,
            "sensor_poses": sensor_poses,
            "pose_params": None,
        }

    return designs


# =============================================================================
# PAPER design
# =============================================================================

def build_paper_design(
    *,
    frame,
    branch_mode: str,
) -> dict[str, Any]:

    line_specs = (
        exp2.make_uniform_radial_line_specs(
            line_count=PAPER_LINE_COUNT,
            line_step_deg=PAPER_LINE_STEP_DEG,
            start_azimuth_deg=0.0,
            paper_poses=paper_poses,
            branch_mode=branch_mode,
        )
    )

    pose_params = (
        exp2.make_pose_parameters_from_paper_lines(
            line_specs=line_specs,
            radius_mm=float(
                SHARED_RADIUS_MM
            ),
        )
    )

    sensor_poses = (
        exp2.make_sensor_poses(
            pose_params=pose_params,
            frame=frame,
            board_center=(
                exp1.BOARD_CENTER
            ),
        )
    )

    expected_pose_count = (
        PAPER_LINE_COUNT
        * len(
            paper_poses
        )
    )

    if len(sensor_poses) != expected_pose_count:
        raise RuntimeError(
            "PAPER pose count mismatch: "
            f"expected {expected_pose_count}, got {len(sensor_poses)}"
        )

    # Current experiment is intentionally the 162-pose full paper design.
    if expected_pose_count != 162:
        raise RuntimeError(
            "This budget-comparison experiment expects PAPER=162 poses, "
            f"but current top-level configuration generates {expected_pose_count}. "
            "If you intentionally changed the PAPER design, update DESIGN_PAPER "
            "and the comparison interpretation accordingly."
        )

    return {
        "kind": "paper",
        "design_family": "PAPER_RADIAL",
        "repeat_count": 1,
        "pose_count": len(
            sensor_poses
        ),
        "paper_poses": paper_poses,
        "line_specs": line_specs,
        "pose_params": pose_params,
        "sensor_poses": sensor_poses,
    }


# =============================================================================
# Build all designs
# =============================================================================

def build_all_designs(
    *,
    frame,
    branch_mode: str,
) -> tuple[
    dict[str, dict[str, Any]],
    dict[str, Any],
]:

    global_base = (
        build_global_base_design(
            frame=frame,
        )
    )

    designs = (
        build_global_budget_designs(
            base_design=global_base,
        )
    )

    paper_design = (
        build_paper_design(
            frame=frame,
            branch_mode=branch_mode,
        )
    )

    designs[
        DESIGN_PAPER
    ] = paper_design

    missing = [
        name
        for name
        in DESIGN_ORDER
        if name not in designs
    ]

    if missing:
        raise RuntimeError(
            f"Missing designs: {missing}"
        )

    return (
        designs,
        global_base,
    )


# =============================================================================
# Initial error / accuracy
# =============================================================================

def actual_initial_errors(
    T_init: np.ndarray,
    T_true: np.ndarray,
) -> tuple[float, float]:

    translation = float(
        np.linalg.norm(
            T_init[:3, 3]
            - T_true[:3, 3]
        )
    )

    rotation = (
        exp1.rotation_error_deg(
            T_init[:3, :3],
            T_true[:3, :3],
        )
    )

    return (
        translation,
        rotation,
    )


def passes_accuracy_threshold(
    result: TrialResult,
) -> bool:

    return bool(
        result.solver_success
        and result.translation_error_mm
        <= ACCURACY_TRANSLATION_THRESHOLD_MM
        and result.rotation_error_deg
        <= ACCURACY_ROTATION_THRESHOLD_DEG
    )


# =============================================================================
# Scan generation
# =============================================================================

def make_global_repeated_scans(
    *,
    sensor_poses: list[np.ndarray],
    frame,
    noise_seed: int,
    noise_std_mm: float,
    T_true: np.ndarray,
) -> list:
    """Simulate repeated scans with unique IDs and independent noise.

    All GLOBAL budgets restart the RNG from the same trial noise_seed.
    Therefore larger budgets are nested:
        N=24 uses exactly the same first 12 noisy scans as N=12,
        then appends 12 new independent noisy scans, etc.
    """

    rng = np.random.default_rng(
        noise_seed
    )

    scans = []

    for scan_id, T_base_s in enumerate(
        sensor_poses
    ):

        scans.append(
            simulate_scan_from_sensor_pose(
                T_base_s=T_base_s,
                T_ef_s_true=T_true,
                frame=frame,
                x_values=exp1.X_VALUES,
                noise_std=noise_std_mm,
                rng=rng,
                plane_id=0,
                scan_id=scan_id,
            )
        )

    return scans


def make_design_scans(
    *,
    design_data: dict[str, Any],
    frame,
    noise_seed: int,
    noise_std_mm: float,
    T_true: np.ndarray,
) -> list:

    if (
        design_data[
            "kind"
        ]
        == "global"
    ):
        return make_global_repeated_scans(
            sensor_poses=(
                design_data[
                    "sensor_poses"
                ]
            ),
            frame=frame,
            noise_seed=noise_seed,
            noise_std_mm=noise_std_mm,
            T_true=T_true,
        )

    if (
        design_data[
            "kind"
        ]
        == "paper"
    ):
        return exp2.make_scans(
            pose_params=(
                design_data[
                    "pose_params"
                ]
            ),
            sensor_poses=(
                design_data[
                    "sensor_poses"
                ]
            ),
            frame=frame,
            noise_seed=noise_seed,
            noise_std_mm=noise_std_mm,
            T_ef_s_true=T_true,
        )

    raise ValueError(
        "Unknown design kind: "
        f"{design_data.get('kind')}"
    )


# =============================================================================
# Fisher-information diagnostics
# =============================================================================


def fim_parameter_scales() -> np.ndarray:
    """Common coordinate scaling for the 6 hand-eye + 3 single-plane states.

    The scaling only nondimensionalizes rotation/translation/plane coordinates so
    determinant-based metrics are comparable.  It is not an information prior.
    """
    return np.asarray(
        [
            *([np.deg2rad(FIM_ROTATION_SCALE_DEG)] * 3),
            *([FIM_TRANSLATION_SCALE_MM] * 3),
            np.deg2rad(FIM_PLANE_NORMAL_SCALE_DEG),
            np.deg2rad(FIM_PLANE_NORMAL_SCALE_DEG),
            FIM_PLANE_OFFSET_SCALE_MM,
        ],
        dtype=float,
    )


def compute_marginal_fim_metrics(
    scans: list,
    T_est: np.ndarray,
    *,
    noise_std_mm: float,
    pose_count: int,
) -> dict[str, float]:
    """Compute observed single-plane hand-eye FIM metrics at ``T_est``.

    Plane normal/offset are nuisance states.  They are fitted from the acquired
    scans and Schur-marginalized, leaving a 6x6 hand-eye information matrix.

    Two D-optimal scores are reported:
      total:       0.5 * log det(H)
      count-normalized: 0.5 * log det(H / N) = total - 3*log(N)

    The normalized version removes the trivial linear information growth caused
    by repeating measurements and is therefore the cleaner pose-geometry
    efficiency comparison between OPTIMAL-12 and PAPER-162.
    """
    if pose_count <= 0:
        raise ValueError("pose_count must be positive")

    grouped = {0: list(scans)}
    planes = fit_plane_estimates(grouped, T_est)
    _residual, jacobian = joint_residual_jacobian(
        grouped,
        T_est,
        planes,
        profile_noise_std_mm=float(noise_std_mm),
        noise_axis="xz",
        parameter_scales=fim_parameter_scales(),
    )

    joint_information = jacobian.T @ jacobian
    joint_information = 0.5 * (joint_information + joint_information.T)
    marginal = marginal_handeye_information(joint_information)
    eigenvalues = np.linalg.eigvalsh(marginal)

    if np.any(~np.isfinite(eigenvalues)) or eigenvalues[0] <= 0.0:
        raise np.linalg.LinAlgError(
            "marginal hand-eye FIM is not positive definite"
        )

    logdet = float(np.sum(np.log(eigenvalues)))
    d_opt = 0.5 * logdet
    e_opt = float(eigenvalues[0])
    condition = float(eigenvalues[-1] / eigenvalues[0])
    n = float(pose_count)

    return {
        "d_optimal_half_logdet": d_opt,
        "d_optimal_count_normalized": d_opt - 3.0 * float(np.log(n)),
        "e_optimal_min_eigenvalue": e_opt,
        "e_optimal_count_normalized": e_opt / n,
        "fim_condition_number": condition,
        "fim_logdet": logdet,
    }


# =============================================================================
# One design evaluation
# =============================================================================

def evaluate_design(
    *,
    trial: int,
    design_name: str,
    design_data: dict[str, Any],
    frame,
    T_init: np.ndarray,
    T_true: np.ndarray,
    gt_euler_deg: np.ndarray,
    gt_translation_mm: np.ndarray,
    actual_init_rotation_error_deg: float,
    actual_init_translation_error_mm: float,
    noise_std_mm: float,
    noise_seed: int,
    max_iter: int,
    tol: float,
    nonlinear_max_nfev: int,
    verbose_solver: bool,
) -> TrialResult:

    scans = (
        make_design_scans(
            design_data=design_data,
            frame=frame,
            noise_seed=noise_seed,
            noise_std_mm=noise_std_mm,
            T_true=T_true,
        )
    )

    try:
        output_context = (
            contextlib.nullcontext()
            if verbose_solver
            else contextlib.redirect_stdout(
                io.StringIO()
            )
        )

        with output_context:
            (
                _,
                nonlinear,
            ) = (
                calibrate_single_plane_with_nonlinear(
                    scans,
                    T_init=T_init.copy(),
                    plane_offset_mode="joint",
                    plane_mode="refit",
                    max_iter=max_iter,
                    tol=tol,
                    nonlinear_max_nfev=(
                        nonlinear_max_nfev
                    ),
                )
            )

        T_est = np.asarray(
            nonlinear.T_ef_s,
            dtype=float,
        ).reshape(
            4,
            4,
        )

        if not np.all(
            np.isfinite(
                T_est
            )
        ):
            raise RuntimeError(
                "non-finite calibration transform"
            )

        translation_error = float(
            np.linalg.norm(
                T_est[:3, 3]
                - T_true[:3, 3]
            )
        )

        rotation_error = (
            exp1.rotation_error_deg(
                T_est[:3, :3],
                T_true[:3, :3],
            )
        )

        final_rms = float(
            nonlinear.final_rms_mm
        )

        if not np.all(
            np.isfinite(
                [
                    translation_error,
                    rotation_error,
                    final_rms,
                ]
            )
        ):
            raise RuntimeError(
                "non-finite calibration metric"
            )

        fim_metrics = compute_marginal_fim_metrics(
            scans,
            T_est,
            noise_std_mm=noise_std_mm,
            pose_count=int(design_data["pose_count"]),
        )

        solver_success = bool(
            nonlinear.success
        )

        accuracy_success = bool(
            solver_success
            and translation_error
            <= ACCURACY_TRANSLATION_THRESHOLD_MM
            and rotation_error
            <= ACCURACY_ROTATION_THRESHOLD_DEG
        )

        error_message = (
            ""
            if solver_success
            else (
                "Nonlinear solver: "
                f"{nonlinear.message}"
            )
        )

    except Exception as exc:
        solver_success = False
        accuracy_success = False

        translation_error = float(
            "nan"
        )

        rotation_error = float(
            "nan"
        )

        final_rms = float(
            "nan"
        )

        fim_metrics = {
            "d_optimal_half_logdet": float("nan"),
            "d_optimal_count_normalized": float("nan"),
            "e_optimal_min_eigenvalue": float("nan"),
            "e_optimal_count_normalized": float("nan"),
            "fim_condition_number": float("nan"),
            "fim_logdet": float("nan"),
        }

        error_message = (
            f"{type(exc).__name__}: {exc}"
        )

    return TrialResult(
        trial=int(
            trial
        ),
        design=design_name,
        design_family=str(
            design_data[
                "design_family"
            ]
        ),
        pose_count=int(
            design_data[
                "pose_count"
            ]
        ),
        repeat_count=int(
            design_data[
                "repeat_count"
            ]
        ),
        radius_mm=float(
            SHARED_RADIUS_MM
        ),
        initial_condition="medium",
        init_rotation_limit_deg=(
            MEDIUM_INIT_ROTATION_DEG
        ),
        init_translation_limit_mm=(
            MEDIUM_INIT_TRANSLATION_MM
        ),
        noise_std_mm=float(
            noise_std_mm
        ),
        actual_init_rotation_error_deg=float(
            actual_init_rotation_error_deg
        ),
        actual_init_translation_error_mm=float(
            actual_init_translation_error_mm
        ),
        gt_euler_x_deg=float(
            gt_euler_deg[0]
        ),
        gt_euler_y_deg=float(
            gt_euler_deg[1]
        ),
        gt_euler_z_deg=float(
            gt_euler_deg[2]
        ),
        gt_translation_x_mm=float(
            gt_translation_mm[0]
        ),
        gt_translation_y_mm=float(
            gt_translation_mm[1]
        ),
        gt_translation_z_mm=float(
            gt_translation_mm[2]
        ),
        solver_success=(
            solver_success
        ),
        accuracy_success=(
            accuracy_success
        ),
        translation_error_mm=(
            translation_error
        ),
        rotation_error_deg=(
            rotation_error
        ),
        final_rms_mm=(
            final_rms
        ),
        d_optimal_half_logdet=float(
            fim_metrics["d_optimal_half_logdet"]
        ),
        d_optimal_count_normalized=float(
            fim_metrics["d_optimal_count_normalized"]
        ),
        e_optimal_min_eigenvalue=float(
            fim_metrics["e_optimal_min_eigenvalue"]
        ),
        e_optimal_count_normalized=float(
            fim_metrics["e_optimal_count_normalized"]
        ),
        fim_condition_number=float(
            fim_metrics["fim_condition_number"]
        ),
        fim_logdet=float(
            fim_metrics["fim_logdet"]
        ),
        error_message=(
            error_message
        ),
    )


# =============================================================================
# Monte-Carlo experiment
# =============================================================================

def run_experiment(
    args: argparse.Namespace,
) -> tuple[
    list[TrialResult],
    dict[str, dict[str, Any]],
    dict[str, Any],
]:

    validate_shared_profile_geometry()

    frame = (
        build_shared_frame()
    )

    (
        designs,
        global_base,
    ) = (
        build_all_designs(
            frame=frame,
            branch_mode=args.branch_mode,
        )
    )

    print(
        "\n"
        + "=" * 96
    )

    print(
        "MONTE-CARLO MEASUREMENT-BUDGET COMPARISON"
    )

    print(
        "=" * 96
    )

    print(
        "trials                 =",
        args.trials,
    )

    print(
        "shared radius [mm]     =",
        SHARED_RADIUS_MM,
    )

    print(
        "GLOBAL near/far [mm]   =",
        (
            EXP1_DISTANCE_NEAR_MM,
            EXP1_DISTANCE_FAR_MM,
        ),
    )

    print(
        "GLOBAL tilt min/max    =",
        (
            EXP1_TILT_MIN_DEG,
            EXP1_TILT_MAX_DEG,
        ),
        "deg",
    )

    print(
        "GLOBAL repetitions     =",
        GLOBAL_REPEAT_COUNTS,
    )

    print(
        "GLOBAL pose budgets    =",
        tuple(
            BASE_GLOBAL_POSE_COUNT
            * k
            for k
            in GLOBAL_REPEAT_COUNTS
        ),
    )

    print(
        "PAPER distances [mm]   =",
        PAPER_DISTANCES_MM,
    )

    print(
        "PAPER thetas [deg]     =",
        PAPER_THETAS_DEG,
    )

    print(
        "PAPER theta -> beta    =",
        {
            float(theta): [
                90.0 - float(theta),
                90.0,
                90.0 + float(theta),
            ]
            for theta
            in PAPER_THETAS_DEG
        },
    )

    print(
        "PAPER pose count       =",
        designs[
            DESIGN_PAPER
        ][
            "pose_count"
        ],
    )

    print(
        "xz noise sigma [mm]    =",
        DEFAULT_NOISE_STD_MM,
    )

    print(
        "init limits            = "
        f"{MEDIUM_INIT_TRANSLATION_MM:g} mm, "
        f"{MEDIUM_INIT_ROTATION_DEG:g} deg"
    )

    results: list[
        TrialResult
    ] = []

    for trial in range(
        args.trials
    ):

        seed_sequence = (
            np.random.SeedSequence(
                [
                    args.seed,
                    trial,
                ]
            )
        )

        (
            gt_seed,
            init_seed,
            noise_seed_sequence,
        ) = (
            seed_sequence.spawn(
                3
            )
        )

        (
            T_true,
            gt_euler_deg,
            gt_translation_mm,
        ) = (
            sample_random_handeye(
                rng=(
                    np.random.default_rng(
                        gt_seed
                    )
                ),
                trans_range_mm=(
                    GT_TRANSLATION_COMPONENT_RANGE_MM
                ),
                angle_range_deg=(
                    GT_EULER_COMPONENT_RANGE_DEG
                ),
            )
        )

        T_init = (
            exp1.make_initial_guess_GT(
                T_true,
                rng=(
                    np.random.default_rng(
                        init_seed
                    )
                ),
                max_rotation_error_deg=(
                    MEDIUM_INIT_ROTATION_DEG
                ),
                max_translation_error_mm=(
                    MEDIUM_INIT_TRANSLATION_MM
                ),
            )
        )

        (
            actual_trans,
            actual_rot,
        ) = (
            actual_initial_errors(
                T_init,
                T_true,
            )
        )

        noise_seed = int(
            noise_seed_sequence
            .generate_state(1)[0]
        )

        for design_name in DESIGN_ORDER:

            results.append(
                evaluate_design(
                    trial=trial,
                    design_name=design_name,
                    design_data=(
                        designs[
                            design_name
                        ]
                    ),
                    frame=frame,
                    T_init=T_init,
                    T_true=T_true,
                    gt_euler_deg=(
                        gt_euler_deg
                    ),
                    gt_translation_mm=(
                        gt_translation_mm
                    ),
                    actual_init_rotation_error_deg=(
                        actual_rot
                    ),
                    actual_init_translation_error_mm=(
                        actual_trans
                    ),
                    noise_std_mm=(
                        DEFAULT_NOISE_STD_MM
                    ),
                    noise_seed=(
                        noise_seed
                    ),
                    max_iter=(
                        args.max_iter
                    ),
                    tol=(
                        args.tol
                    ),
                    nonlinear_max_nfev=(
                        args.nonlinear_max_nfev
                    ),
                    verbose_solver=(
                        args.verbose_solver
                    ),
                )
            )

        completed = (
            trial
            + 1
        )

        if (
            completed == 1
            or completed % 10 == 0
            or completed == args.trials
        ):
            print(
                "  trials completed: "
                f"{completed}/{args.trials}"
                f"  ({len(DESIGN_ORDER)} calibrations/trial)"
            )

    return (
        results,
        designs,
        global_base,
    )


# =============================================================================
# Statistics
# =============================================================================

def percentile(
    values: np.ndarray,
    q: float,
) -> float:

    finite = values[
        np.isfinite(
            values
        )
    ]

    return (
        float(
            np.percentile(
                finite,
                q,
            )
        )
        if finite.size
        else float(
            "nan"
        )
    )


def rows_for_design(
    results: list[TrialResult],
    design: str,
) -> list[TrialResult]:

    return [
        result
        for result
        in results
        if result.design == design
    ]


def design_summary(
    rows: list[TrialResult],
) -> dict[str, object]:

    if not rows:
        raise ValueError(
            "Cannot summarize an empty design"
        )

    successful = [
        result
        for result
        in rows
        if result.solver_success
    ]

    translation = np.asarray(
        [
            result.translation_error_mm
            for result
            in successful
        ],
        dtype=float,
    )

    rotation = np.asarray(
        [
            result.rotation_error_deg
            for result
            in successful
        ],
        dtype=float,
    )

    rms = np.asarray(
        [
            result.final_rms_mm
            for result
            in successful
        ],
        dtype=float,
    )

    dopt = np.asarray(
        [result.d_optimal_half_logdet for result in successful],
        dtype=float,
    )
    dopt_norm = np.asarray(
        [result.d_optimal_count_normalized for result in successful],
        dtype=float,
    )
    eopt = np.asarray(
        [result.e_optimal_min_eigenvalue for result in successful],
        dtype=float,
    )
    eopt_norm = np.asarray(
        [result.e_optimal_count_normalized for result in successful],
        dtype=float,
    )
    fim_cond = np.asarray(
        [result.fim_condition_number for result in successful],
        dtype=float,
    )

    solver_count = int(
        sum(
            result.solver_success
            for result
            in rows
        )
    )

    accuracy_count = int(
        sum(
            passes_accuracy_threshold(
                result
            )
            for result
            in rows
        )
    )

    return {
        "design": rows[0].design,
        "design_family": rows[0].design_family,
        "pose_count": rows[0].pose_count,
        "repeat_count": rows[0].repeat_count,
        "radius_mm": rows[0].radius_mm,
        "trials": len(
            rows
        ),
        "solver_success_count": solver_count,
        "solver_success_rate": float(
            solver_count
            / len(rows)
        ),
        "accuracy_success_count": accuracy_count,
        "accuracy_success_rate": float(
            accuracy_count
            / len(rows)
        ),
        "median_translation_error_mm": (
            percentile(
                translation,
                50,
            )
        ),
        "p95_translation_error_mm": (
            percentile(
                translation,
                95,
            )
        ),
        "median_rotation_error_deg": (
            percentile(
                rotation,
                50,
            )
        ),
        "p95_rotation_error_deg": (
            percentile(
                rotation,
                95,
            )
        ),
        "median_final_rms_mm": (
            percentile(
                rms,
                50,
            )
        ),
        "median_d_optimal_half_logdet": percentile(dopt, 50),
        "p05_d_optimal_half_logdet": percentile(dopt, 5),
        "median_d_optimal_count_normalized": percentile(dopt_norm, 50),
        "p05_d_optimal_count_normalized": percentile(dopt_norm, 5),
        "median_e_optimal_min_eigenvalue": percentile(eopt, 50),
        "p05_e_optimal_min_eigenvalue": percentile(eopt, 5),
        "median_e_optimal_count_normalized": percentile(eopt_norm, 50),
        "p05_e_optimal_count_normalized": percentile(eopt_norm, 5),
        "median_fim_condition_number": percentile(fim_cond, 50),
        "p95_fim_condition_number": percentile(fim_cond, 95),
    }


def all_summaries(
    results: list[TrialResult],
) -> list[
    dict[
        str,
        object,
    ]
]:

    return [
        design_summary(
            rows_for_design(
                results,
                design,
            )
        )
        for design
        in DESIGN_ORDER
    ]


def paired_opt12_vs_paper_information_summary(
    results: list[TrialResult],
) -> dict[str, float]:
    """Paired Monte-Carlo OPT12-PAPER162 information differences."""
    lookup = {(r.trial, r.design): r for r in results}
    trial_ids = sorted({r.trial for r in results})
    d_delta = []
    dn_delta = []
    en_ratio = []
    for trial in trial_ids:
        opt = lookup.get((trial, "GLOBAL_OPTIMAL_12"))
        paper = lookup.get((trial, DESIGN_PAPER))
        if opt is None or paper is None:
            continue
        if (
            np.isfinite(opt.d_optimal_half_logdet)
            and np.isfinite(paper.d_optimal_half_logdet)
        ):
            d_delta.append(
                opt.d_optimal_half_logdet - paper.d_optimal_half_logdet
            )
        if (
            np.isfinite(opt.d_optimal_count_normalized)
            and np.isfinite(paper.d_optimal_count_normalized)
        ):
            dn_delta.append(
                opt.d_optimal_count_normalized
                - paper.d_optimal_count_normalized
            )
        if (
            np.isfinite(opt.e_optimal_count_normalized)
            and np.isfinite(paper.e_optimal_count_normalized)
            and paper.e_optimal_count_normalized > 0.0
        ):
            en_ratio.append(
                opt.e_optimal_count_normalized
                / paper.e_optimal_count_normalized
            )

    def q(values, pct):
        arr = np.asarray(values, dtype=float)
        return float(np.percentile(arr, pct)) if arr.size else float("nan")

    return {
        "paired_count": float(len(d_delta)),
        "total_dopt_delta_q25": q(d_delta, 25),
        "total_dopt_delta_median": q(d_delta, 50),
        "total_dopt_delta_q75": q(d_delta, 75),
        "count_normalized_dopt_delta_q25": q(dn_delta, 25),
        "count_normalized_dopt_delta_median": q(dn_delta, 50),
        "count_normalized_dopt_delta_q75": q(dn_delta, 75),
        "count_normalized_eopt_ratio_q25": q(en_ratio, 25),
        "count_normalized_eopt_ratio_median": q(en_ratio, 50),
        "count_normalized_eopt_ratio_q75": q(en_ratio, 75),
    }


def make_opt12_vs_paper_information_summary(
    summaries: list[dict[str, object]],
) -> dict[str, float]:
    """Direct OPTIMAL-12 vs PAPER-162 information comparison."""
    by_design = {str(row["design"]): row for row in summaries}
    opt = by_design["GLOBAL_OPTIMAL_12"]
    paper = by_design[DESIGN_PAPER]

    d_opt = float(opt["median_d_optimal_half_logdet"])
    d_paper = float(paper["median_d_optimal_half_logdet"])
    dn_opt = float(opt["median_d_optimal_count_normalized"])
    dn_paper = float(paper["median_d_optimal_count_normalized"])
    en_opt = float(opt["median_e_optimal_count_normalized"])
    en_paper = float(paper["median_e_optimal_count_normalized"])

    return {
        "opt12_scan_count": 12.0,
        "paper162_scan_count": 162.0,
        "measurement_fraction_opt12_over_paper162": 12.0 / 162.0,
        "median_total_dopt_opt12": d_opt,
        "median_total_dopt_paper162": d_paper,
        "median_total_dopt_delta_opt12_minus_paper162": d_opt - d_paper,
        # log(det(H_opt)/det(H_paper)) = 2 * Delta D
        "median_log_determinant_ratio_opt12_over_paper162": 2.0 * (d_opt - d_paper),
        "median_count_normalized_dopt_opt12": dn_opt,
        "median_count_normalized_dopt_paper162": dn_paper,
        "median_count_normalized_dopt_advantage_opt12_minus_paper162": (
            dn_opt - dn_paper
        ),
        "median_count_normalized_eopt_opt12": en_opt,
        "median_count_normalized_eopt_paper162": en_paper,
        "median_count_normalized_eopt_ratio_opt12_over_paper162": (
            en_opt / en_paper if en_paper > 0.0 else float("nan")
        ),
    }


# =============================================================================
# CSV / JSON
# =============================================================================

def write_trial_csv(
    results: list[TrialResult],
    path: Path,
) -> None:

    rows = [
        asdict(
            result
        )
        for result
        in results
    ]

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as stream:

        writer = csv.DictWriter(
            stream,
            fieldnames=list(
                rows[0]
            ),
        )

        writer.writeheader()
        writer.writerows(
            rows
        )


def write_summary_csv(
    summaries: list[
        dict[
            str,
            object,
        ]
    ],
    path: Path,
) -> None:

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as stream:

        writer = csv.DictWriter(
            stream,
            fieldnames=list(
                summaries[0]
            ),
        )

        writer.writeheader()
        writer.writerows(
            summaries
        )


def write_summary_json(
    *,
    summaries: list[
        dict[
            str,
            object,
        ]
    ],
    results: list[TrialResult],
    args: argparse.Namespace,
    path: Path,
) -> None:

    payload = {
        "schema": (
            "global_optimal_budget_curve_vs_paper_162"
        ),
        "shared_protocol": {
            "trials": args.trials,
            "seed": args.seed,
            "radius_mm": (
                SHARED_RADIUS_MM
            ),
            "noise_std_mm": (
                DEFAULT_NOISE_STD_MM
            ),
            "init_translation_limit_mm": (
                MEDIUM_INIT_TRANSLATION_MM
            ),
            "init_rotation_limit_deg": (
                MEDIUM_INIT_ROTATION_DEG
            ),
            "accuracy_translation_threshold_mm": (
                ACCURACY_TRANSLATION_THRESHOLD_MM
            ),
            "accuracy_rotation_threshold_deg": (
                ACCURACY_ROTATION_THRESHOLD_DEG
            ),
            "same_gt_per_trial": True,
            "same_initialization_per_trial": True,
            "same_noise_sigma": True,
            "same_trial_noise_seed": True,
            "global_budget_noise_is_nested": True,
            "repeated_global_scans_have_independent_noise": True,
        },
        "fim_definition": {
            "state": "6 hand-eye + 3 single-plane nuisance parameters",
            "plane_marginalization": "Schur complement",
            "d_optimal_total": "0.5 * log det(H_HE_marg)",
            "d_optimal_count_normalized": "0.5 * log det(H_HE_marg / N) = D - 3 log N",
            "e_optimal_total": "lambda_min(H_HE_marg)",
            "e_optimal_count_normalized": "lambda_min(H_HE_marg) / N",
            "rotation_scale_deg": FIM_ROTATION_SCALE_DEG,
            "translation_scale_mm": FIM_TRANSLATION_SCALE_MM,
            "plane_normal_scale_deg": FIM_PLANE_NORMAL_SCALE_DEG,
            "plane_offset_scale_mm": FIM_PLANE_OFFSET_SCALE_MM,
        },
        "opt12_vs_paper162_information": make_opt12_vs_paper_information_summary(
            summaries
        ),
        "opt12_vs_paper162_paired_information": (
            paired_opt12_vs_paper_information_summary(results)
        ),
        "summaries": summaries,
    }

    with path.open(
        "w",
        encoding="utf-8",
    ) as stream:

        json.dump(
            payload,
            stream,
            indent=2,
            allow_nan=False,
        )

        stream.write(
            "\n"
        )


# =============================================================================
# Plot helpers
# =============================================================================

def short_design_label(
    design: str,
) -> str:

    if design.startswith(
        "GLOBAL_OPTIMAL_"
    ):
        return (
            "Opt-"
            + design.rsplit(
                "_",
                1,
            )[
                -1
            ]
        )

    if design == DESIGN_PAPER:
        return "Paper-162"

    return design


def _usable_values(
    rows: list[TrialResult],
    *,
    field: str,
    exclude_large_trials: bool,
) -> np.ndarray:

    output = []

    for result in rows:

        if not result.solver_success:
            continue

        if (
            exclude_large_trials
            and (
                (
                    np.isfinite(
                        result.translation_error_mm
                    )
                    and result.translation_error_mm
                    >= PLOT_EXCLUSION_TRANSLATION_MM
                )
                or (
                    np.isfinite(
                        result.rotation_error_deg
                    )
                    and result.rotation_error_deg
                    >= PLOT_EXCLUSION_ROTATION_DEG
                )
            )
        ):
            continue

        value = float(
            getattr(
                result,
                field,
            )
        )

        if np.isfinite(
            value
        ):
            output.append(
                value
            )

    return np.asarray(
        output,
        dtype=float,
    )


def create_error_boxplots(
    results: list[TrialResult],
    path: Path,
    *,
    exclude_large_trials: bool = False,
):
    translation = []
    rotation = []

    for design in DESIGN_ORDER:
        rows = rows_for_design(
            results,
            design,
        )

        translation.append(
            _usable_values(
                rows,
                field=(
                    "translation_error_mm"
                ),
                exclude_large_trials=(
                    exclude_large_trials
                ),
            )
        )

        rotation.append(
            _usable_values(
                rows,
                field=(
                    "rotation_error_deg"
                ),
                exclude_large_trials=(
                    exclude_large_trials
                ),
            )
        )

    labels = [
        short_design_label(
            design
        )
        for design
        in DESIGN_ORDER
    ]

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(
            10.0,
            8.6,
        ),
        constrained_layout=True,
    )

    axes[0].boxplot(
        [
            values
            if values.size
            else np.asarray(
                [np.nan]
            )
            for values
            in translation
        ],
        labels=labels,
        showfliers=True,
    )

    axes[0].set_ylabel(
        "Translation error [mm]"
    )

    axes[0].grid(
        axis="y",
        alpha=0.25,
    )

    axes[1].boxplot(
        [
            values
            if values.size
            else np.asarray(
                [np.nan]
            )
            for values
            in rotation
        ],
        labels=labels,
        showfliers=True,
    )

    axes[1].set_ylabel(
        "Rotation error [deg]"
    )

    axes[1].grid(
        axis="y",
        alpha=0.25,
    )

    title = (
        "Measurement-budget Monte-Carlo calibration errors"
        f"\nshared radius={SHARED_RADIUS_MM:g} mm, "
        f"xz noise sigma={DEFAULT_NOISE_STD_MM:g} mm"
    )

    if exclude_large_trials:
        title += (
            "\nper-design removal of trials with "
            f"T>={PLOT_EXCLUSION_TRANSLATION_MM:g} mm "
            f"or R>={PLOT_EXCLUSION_ROTATION_DEG:g} deg"
        )

    fig.suptitle(
        title
    )

    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )

    return fig


def create_success_rate_plot(
    results: list[TrialResult],
    path: Path,
):
    solver_rates = []
    accuracy_rates = []

    for design in DESIGN_ORDER:

        rows = rows_for_design(
            results,
            design,
        )

        solver_rates.append(
            float(
                np.mean(
                    [
                        row.solver_success
                        for row
                        in rows
                    ]
                )
            )
        )

        accuracy_rates.append(
            float(
                np.mean(
                    [
                        passes_accuracy_threshold(
                            row
                        )
                        for row
                        in rows
                    ]
                )
            )
        )

    x = np.arange(
        len(
            DESIGN_ORDER
        ),
        dtype=float,
    )

    width = 0.36

    fig, axis = plt.subplots(
        figsize=(
            10.0,
            5.8,
        ),
        constrained_layout=True,
    )

    bars_solver = axis.bar(
        x - width / 2.0,
        solver_rates,
        width=width,
        label="Solver success",
    )

    bars_accuracy = axis.bar(
        x + width / 2.0,
        accuracy_rates,
        width=width,
        label="Accuracy success",
    )

    for bars, rates in (
        (
            bars_solver,
            solver_rates,
        ),
        (
            bars_accuracy,
            accuracy_rates,
        ),
    ):
        for bar, rate in zip(
            bars,
            rates,
        ):
            axis.text(
                bar.get_x()
                + bar.get_width() / 2.0,
                min(
                    rate + 0.02,
                    1.055,
                ),
                f"{100.0 * rate:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    axis.set_xticks(
        x,
        [
            short_design_label(
                design
            )
            for design
            in DESIGN_ORDER
        ],
    )

    axis.set_ylabel(
        "Success rate"
    )

    axis.set_ylim(
        0.0,
        1.12,
    )

    axis.yaxis.set_major_formatter(
        lambda value, _position:
        f"{100.0 * value:.0f}%"
    )

    axis.grid(
        axis="y",
        alpha=0.25,
    )

    axis.legend()

    axis.set_title(
        "Monte-Carlo success-rate comparison"
    )

    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )

    return fig


def create_error_vs_pose_count_plot(
    summaries: list[
        dict[
            str,
            object,
        ]
    ],
    path: Path,
):
    summary_by_design = {
        str(
            row[
                "design"
            ]
        ): row
        for row
        in summaries
    }

    global_rows = [
        summary_by_design[
            design
        ]
        for design
        in GLOBAL_DESIGNS
    ]

    paper_row = (
        summary_by_design[
            DESIGN_PAPER
        ]
    )

    x_global = np.asarray(
        [
            row[
                "pose_count"
            ]
            for row
            in global_rows
        ],
        dtype=float,
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(
            8.2,
            8.2,
        ),
        constrained_layout=True,
    )

    # Translation
    axes[0].plot(
        x_global,
        [
            row[
                "median_translation_error_mm"
            ]
            for row
            in global_rows
        ],
        marker="o",
        label="Optimal median",
    )

    axes[0].plot(
        x_global,
        [
            row[
                "p95_translation_error_mm"
            ]
            for row
            in global_rows
        ],
        marker="s",
        label="Optimal P95",
    )

    axes[0].scatter(
        [
            paper_row[
                "pose_count"
            ]
        ],
        [
            paper_row[
                "median_translation_error_mm"
            ]
        ],
        marker="D",
        s=60,
        label="Paper median",
    )

    axes[0].scatter(
        [
            paper_row[
                "pose_count"
            ]
        ],
        [
            paper_row[
                "p95_translation_error_mm"
            ]
        ],
        marker="X",
        s=65,
        label="Paper P95",
    )

    axes[0].set_xlabel(
        "Number of scans"
    )

    axes[0].set_ylabel(
        "Translation error [mm]"
    )

    axes[0].grid(
        alpha=0.25,
    )

    axes[0].legend()

    # Rotation
    axes[1].plot(
        x_global,
        [
            row[
                "median_rotation_error_deg"
            ]
            for row
            in global_rows
        ],
        marker="o",
        label="Optimal median",
    )

    axes[1].plot(
        x_global,
        [
            row[
                "p95_rotation_error_deg"
            ]
            for row
            in global_rows
        ],
        marker="s",
        label="Optimal P95",
    )

    axes[1].scatter(
        [
            paper_row[
                "pose_count"
            ]
        ],
        [
            paper_row[
                "median_rotation_error_deg"
            ]
        ],
        marker="D",
        s=60,
        label="Paper median",
    )

    axes[1].scatter(
        [
            paper_row[
                "pose_count"
            ]
        ],
        [
            paper_row[
                "p95_rotation_error_deg"
            ]
        ],
        marker="X",
        s=65,
        label="Paper P95",
    )

    axes[1].set_xlabel(
        "Number of scans"
    )

    axes[1].set_ylabel(
        "Rotation error [deg]"
    )

    axes[1].grid(
        alpha=0.25,
    )

    axes[1].legend()

    fig.suptitle(
        "Calibration error vs measurement budget"
    )

    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )

    return fig


def create_accuracy_vs_pose_count_plot(
    summaries: list[
        dict[
            str,
            object,
        ]
    ],
    path: Path,
):
    summary_by_design = {
        str(
            row[
                "design"
            ]
        ): row
        for row
        in summaries
    }

    global_rows = [
        summary_by_design[
            design
        ]
        for design
        in GLOBAL_DESIGNS
    ]

    paper_row = (
        summary_by_design[
            DESIGN_PAPER
        ]
    )

    x_global = np.asarray(
        [
            row[
                "pose_count"
            ]
            for row
            in global_rows
        ],
        dtype=float,
    )

    y_global = np.asarray(
        [
            row[
                "accuracy_success_rate"
            ]
            for row
            in global_rows
        ],
        dtype=float,
    )

    fig, axis = plt.subplots(
        figsize=(
            8.2,
            5.5,
        ),
        constrained_layout=True,
    )

    axis.plot(
        x_global,
        y_global,
        marker="o",
        label="GLOBAL_OPTIMAL repetitions",
    )

    axis.scatter(
        [
            paper_row[
                "pose_count"
            ]
        ],
        [
            paper_row[
                "accuracy_success_rate"
            ]
        ],
        marker="D",
        s=70,
        label="PAPER",
    )

    axis.set_xlabel(
        "Number of scans"
    )

    axis.set_ylabel(
        "Accuracy success rate"
    )

    axis.set_ylim(
        0.0,
        1.05,
    )

    axis.yaxis.set_major_formatter(
        lambda value, _position:
        f"{100.0 * value:.0f}%"
    )

    axis.grid(
        alpha=0.25,
    )

    axis.legend()

    axis.set_title(
        "Accuracy success vs measurement budget"
    )

    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )

    return fig


def create_information_comparison_plot(
    summaries: list[dict[str, object]],
    path: Path,
):
    """Plot total and count-normalized D-opt for OPT12 and PAPER162."""
    by_design = {str(row["design"]): row for row in summaries}
    labels = ["OPTIMAL-12", "PAPER-162"]
    rows = [by_design["GLOBAL_OPTIMAL_12"], by_design[DESIGN_PAPER]]

    total = [float(row["median_d_optimal_half_logdet"]) for row in rows]
    normalized = [
        float(row["median_d_optimal_count_normalized"]) for row in rows
    ]

    x = np.arange(2, dtype=float)
    width = 0.36
    fig, axis = plt.subplots(figsize=(7.6, 5.5), constrained_layout=True)
    axis.bar(x - width / 2.0, total, width=width, label="Total D-opt")
    axis.bar(
        x + width / 2.0,
        normalized,
        width=width,
        label="Count-normalized D-opt",
    )
    axis.set_xticks(x, labels)
    axis.set_ylabel(r"$\frac{1}{2}\log\det(H_{HE}^{marg})$")
    axis.set_title("OPTIMAL-12 vs PAPER-162 Fisher information")
    axis.grid(axis="y", alpha=0.25)
    axis.legend()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    return fig


# =============================================================================
# Terminal summary
# =============================================================================

def print_summary(
    summaries: list[
        dict[
            str,
            object,
        ]
    ],
    results: list[TrialResult],
) -> None:

    print(
        "\n"
        + "=" * 112
    )

    print(
        "MONTE-CARLO BUDGET-CURVE SUMMARY"
    )

    print(
        "=" * 112
    )

    print(
        "design          poses repeats  solver    accuracy   "
        "T median/P95 [mm]       R median/P95 [deg]"
    )

    print(
        "-" * 112
    )

    for row in summaries:

        print(
            f"{short_design_label(str(row['design'])):14s} "
            f"{int(row['pose_count']):5d} "
            f"{int(row['repeat_count']):7d}  "
            f"{float(row['solver_success_rate']):7.1%}   "
            f"{float(row['accuracy_success_rate']):7.1%}   "
            f"{float(row['median_translation_error_mm']):9.4g}/"
            f"{float(row['p95_translation_error_mm']):<9.4g}   "
            f"{float(row['median_rotation_error_deg']):9.4g}/"
            f"{float(row['p95_rotation_error_deg']):<9.4g}"
        )



    info = make_opt12_vs_paper_information_summary(summaries)
    print("\n" + "=" * 112)
    print("OPTIMAL-12 vs PAPER-162 FISHER-INFORMATION COMPARISON")
    print("=" * 112)
    print(
        f"measurements             : 12 vs 162 "
        f"({100.0 * info['measurement_fraction_opt12_over_paper162']:.2f}% as many)"
    )
    print(
        f"total D-opt median       : "
        f"{info['median_total_dopt_opt12']:.6f} vs "
        f"{info['median_total_dopt_paper162']:.6f}"
    )
    print(
        f"total D-opt delta        : "
        f"{info['median_total_dopt_delta_opt12_minus_paper162']:+.6f} "
        f"(OPT12 - PAPER162)"
    )
    print(
        f"count-normalized D-opt   : "
        f"{info['median_count_normalized_dopt_opt12']:.6f} vs "
        f"{info['median_count_normalized_dopt_paper162']:.6f}"
    )
    print(
        f"geometry-efficiency gain : "
        f"{info['median_count_normalized_dopt_advantage_opt12_minus_paper162']:+.6f} nats "
        f"(OPT12 - PAPER162)"
    )
    print(
        f"count-normalized E-opt   : "
        f"{info['median_count_normalized_eopt_opt12']:.6g} vs "
        f"{info['median_count_normalized_eopt_paper162']:.6g}"
    )
    paired = paired_opt12_vs_paper_information_summary(results)
    print(
        f"paired total D-opt delta : "
        f"{paired['total_dopt_delta_median']:+.6f} "
        f"[IQR {paired['total_dopt_delta_q25']:+.6f}, "
        f"{paired['total_dopt_delta_q75']:+.6f}]"
    )
    print(
        f"paired norm D-opt delta  : "
        f"{paired['count_normalized_dopt_delta_median']:+.6f} "
        f"[IQR {paired['count_normalized_dopt_delta_q25']:+.6f}, "
        f"{paired['count_normalized_dopt_delta_q75']:+.6f}]"
    )


# =============================================================================
# Config
# =============================================================================

def write_config(
    *,
    args: argparse.Namespace,
    designs: dict[
        str,
        dict[
            str,
            Any,
        ],
    ],
    global_base: dict[
        str,
        Any,
    ],
    path: Path,
) -> None:

    payload = {
        "experiment": (
            "GLOBAL_OPTIMAL_budget_curve_vs_PAPER_162"
        ),
        "shared_radius_mm": (
            SHARED_RADIUS_MM
        ),
        "trials": args.trials,
        "seed": args.seed,
        "noise_std_mm": (
            DEFAULT_NOISE_STD_MM
        ),
        "fim_scaling": {
            "rotation_scale_deg": FIM_ROTATION_SCALE_DEG,
            "translation_scale_mm": FIM_TRANSLATION_SCALE_MM,
            "plane_normal_scale_deg": FIM_PLANE_NORMAL_SCALE_DEG,
            "plane_offset_scale_mm": FIM_PLANE_OFFSET_SCALE_MM,
            "note": "coordinate scaling only; not an information prior",
        },
        "init_translation_limit_mm": (
            MEDIUM_INIT_TRANSLATION_MM
        ),
        "init_rotation_limit_deg": (
            MEDIUM_INIT_ROTATION_DEG
        ),
        "gt_translation_component_range_mm": (
            GT_TRANSLATION_COMPONENT_RANGE_MM
        ),
        "gt_euler_component_range_deg": (
            GT_EULER_COMPONENT_RANGE_DEG
        ),
        "max_iter": args.max_iter,
        "tol": args.tol,
        "nonlinear_max_nfev": (
            args.nonlinear_max_nfev
        ),
        "accuracy_threshold_translation_mm": (
            ACCURACY_TRANSLATION_THRESHOLD_MM
        ),
        "accuracy_threshold_rotation_deg": (
            ACCURACY_ROTATION_THRESHOLD_DEG
        ),
        "global_optimal": {
            "base_pose_count": (
                BASE_GLOBAL_POSE_COUNT
            ),
            "repeat_counts": list(
                GLOBAL_REPEAT_COUNTS
            ),
            "pose_budgets": [
                BASE_GLOBAL_POSE_COUNT
                * k
                for k
                in GLOBAL_REPEAT_COUNTS
            ],
            "distance_near_mm": float(
                EXP1_DISTANCE_NEAR_MM
            ),
            "distance_far_mm": float(
                EXP1_DISTANCE_FAR_MM
            ),
            "tilt_min_deg": float(
                EXP1_TILT_MIN_DEG
            ),
            "tilt_max_deg": float(
                EXP1_TILT_MAX_DEG
            ),
            "support_ids": (
                global_base[
                    "support_ids"
                ].tolist()
            ),
            "beta_optimal_deg": (
                global_base[
                    "beta_optimal"
                ].tolist()
            ),
            "same_12_physical_poses_repeated": True,
            "independent_noise_per_repeated_scan": True,
            "nested_noise_across_budgets": True,
        },
        "paper": {
            "design": DESIGN_PAPER,
            "pose_count": int(
                designs[
                    DESIGN_PAPER
                ][
                    "pose_count"
                ]
            ),
            "line_count": (
                PAPER_LINE_COUNT
            ),
            "line_step_deg": (
                PAPER_LINE_STEP_DEG
            ),
            "distances_mm": list(
                PAPER_DISTANCES_MM
            ),
            "thetas_deg": list(
                PAPER_THETAS_DEG
            ),
            "theta_to_beta_deg": {
                str(
                    float(
                        theta
                    )
                ): [
                    90.0
                    - float(
                        theta
                    ),
                    90.0,
                    90.0
                    + float(
                        theta
                    ),
                ]
                for theta
                in PAPER_THETAS_DEG
            },
            "branch_mode": (
                args.branch_mode
            ),
        },
        "fairness": {
            "same_radius": True,
            "same_gt_per_trial": True,
            "same_initialization_per_trial": True,
            "same_noise_sigma": True,
            "same_trial_noise_seed": True,
            "same_solver_settings": True,
            "same_profile_sampling": True,
            "equal_pose_count_at_endpoint": False,
            "closest_endpoint_comparison": (
                "GLOBAL_OPTIMAL_156 vs PAPER_RADIAL_162"
            ),
        },
    }

    with path.open(
        "w",
        encoding="utf-8",
    ) as stream:

        json.dump(
            payload,
            stream,
            indent=2,
        )

        stream.write(
            "\n"
        )


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    args = parse_args()

    plt.switch_backend(
        "Agg"
        if not args.show
        else plt.get_backend()
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        results,
        designs,
        global_base,
    ) = (
        run_experiment(
            args
        )
    )

    summaries = (
        all_summaries(
            results
        )
    )

    trial_path = (
        args.output_dir
        / "trials.csv"
    )

    summary_path = (
        args.output_dir
        / "summary.csv"
    )

    summary_json_path = (
        args.output_dir
        / "summary.json"
    )

    config_path = (
        args.output_dir
        / "config.json"
    )

    error_plot_path = (
        args.output_dir
        / "final_error_boxplots.png"
    )

    filtered_error_plot_path = (
        args.output_dir
        / "final_error_boxplots_without_large_outliers.png"
    )

    success_plot_path = (
        args.output_dir
        / "success_rates.png"
    )

    error_curve_path = (
        args.output_dir
        / "error_vs_pose_count.png"
    )

    accuracy_curve_path = (
        args.output_dir
        / "accuracy_vs_pose_count.png"
    )

    information_plot_path = (
        args.output_dir
        / "opt12_vs_paper162_fisher_information.png"
    )

    write_trial_csv(
        results,
        trial_path,
    )

    write_summary_csv(
        summaries,
        summary_path,
    )

    write_summary_json(
        summaries=summaries,
        results=results,
        args=args,
        path=summary_json_path,
    )

    write_config(
        args=args,
        designs=designs,
        global_base=global_base,
        path=config_path,
    )

    figures = [
        create_error_boxplots(
            results,
            error_plot_path,
        ),
        create_error_boxplots(
            results,
            filtered_error_plot_path,
            exclude_large_trials=True,
        ),
        create_success_rate_plot(
            results,
            success_plot_path,
        ),
        create_error_vs_pose_count_plot(
            summaries,
            error_curve_path,
        ),
        create_accuracy_vs_pose_count_plot(
            summaries,
            accuracy_curve_path,
        ),
        create_information_comparison_plot(
            summaries,
            information_plot_path,
        ),
    ]

    print_summary(
        summaries,
        results,
    )

    print(
        f"\nSaved:"
        f"\n  {config_path}"
        f"\n  {trial_path}"
        f"\n  {summary_path}"
        f"\n  {summary_json_path}"
        f"\n  {error_plot_path}"
        f"\n  {filtered_error_plot_path}"
        f"\n  {success_plot_path}"
        f"\n  {error_curve_path}"
        f"\n  {accuracy_curve_path}"
        f"\n  {information_plot_path}"
    )

    if args.show:
        plt.show()

    for figure in figures:
        plt.close(
            figure
        )


if __name__ == "__main__":
    main()