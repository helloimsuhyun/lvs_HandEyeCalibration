"""
Monte-Carlo evaluation for the 12-pose GLOBAL_OPTIMAL design.

The pose geometry is defined in ``exp1_optimal.py``.  Every trial uses the
same optimized 12-pose layout, while the GT hand-eye transform, initialization,
and measurement noise are resampled.

Example
-------
PYTHONPATH=. python \
  robust_laser_handeye/well_made_experiment/exp1_optimal_monte_carlo.py \
  --trials 500

Defaults follow the repository's medium initialization condition:
    - initialization translation limit: 100 mm
    - initialization rotation limit: 15 deg
    - xz-noise sigma: 0.25 mm
    - random GT hand-eye from ``simulation.sample_random_handeye``

Accuracy success is defined as:
    translation error <= 1.0 mm
    rotation error    <= 0.25 deg
"""

from __future__ import annotations

import argparse
import contextlib
import csv
from dataclasses import asdict, dataclass
import io
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from robust_laser_handeye.laser_handeye.calibration import (
    calibrate_single_plane_with_nonlinear,
)
from robust_laser_handeye.laser_handeye.simulation import (
    sample_random_handeye,
)
from robust_laser_handeye.well_made_experiment import (
    exp1_optimal as exp1,
)


DESIGN = "GLOBAL_OPTIMAL"

MEDIUM_INIT_TRANSLATION_MM = 100.0
MEDIUM_INIT_ROTATION_DEG = 15.0
DEFAULT_NOISE_STD_MM = 0.25

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

    error_message: str


# =============================================================================
# Accuracy
# =============================================================================

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
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Monte-Carlo evaluation of the 12-pose "
            "GLOBAL_OPTIMAL design from exp1_optimal.py."
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
        default=20260901,
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
            "runs/exp1_optimal_monte_carlo"
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
        help=(
            "Print the iterative solver trace for "
            "every calibration."
        ),
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
# Common geometry
# =============================================================================

def build_common_geometry():
    frame = exp1.make_plane_frame(
        exp1.PLANE_NORMAL,
        exp1.BOARD_CENTER,
    )

    uv = exp1.circular_uv_points(
        radius_mm=exp1.RADIUS_MM,
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

    return (
        frame,
        uv,
        target_points,
    )


# =============================================================================
# Initial-error measurement
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


# =============================================================================
# One Monte-Carlo trial
# =============================================================================

def run_one_trial(
    *,
    trial: int,
    beta_deg: np.ndarray,
    uv: np.ndarray,
    target_points: np.ndarray,
    frame,
    T_init: np.ndarray,
    T_true: np.ndarray,
    gt_euler_deg: np.ndarray,
    gt_translation_mm: np.ndarray,
    init_rotation_limit_deg: float,
    init_translation_limit_mm: float,
    actual_init_rotation_error_deg: float,
    actual_init_translation_error_mm: float,
    noise_std_mm: float,
    noise_seed: int,
    max_iter: int,
    tol: float,
    nonlinear_max_nfev: int,
    verbose_solver: bool,
) -> TrialResult:

    # -------------------------------------------------------------------------
    # The support assignment is fixed inside exp1_optimal.make_scan_pose_parameters.
    # beta_deg is the matching global-optimal beta assignment.
    # -------------------------------------------------------------------------

    pose_params = (
        exp1.make_scan_pose_parameters(
            uv=uv,
            beta_deg=beta_deg,
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

    scans = (
        exp1.make_scans(
            pose_params=pose_params,
            sensor_poses=sensor_poses,
            frame=frame,
            noise_seed=noise_seed,
            noise_std_mm=noise_std_mm,
            T_ef_s_true=T_true,
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
        # Failed trials remain visible in all rate calculations.
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

        error_message = (
            f"{type(exc).__name__}: {exc}"
        )

    return TrialResult(
        trial=trial,
        design=DESIGN,
        initial_condition="medium",
        init_rotation_limit_deg=(
            init_rotation_limit_deg
        ),
        init_translation_limit_mm=(
            init_translation_limit_mm
        ),
        noise_std_mm=noise_std_mm,
        actual_init_rotation_error_deg=(
            actual_init_rotation_error_deg
        ),
        actual_init_translation_error_mm=(
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
        error_message=(
            error_message
        ),
    )


# =============================================================================
# Monte-Carlo experiment
# =============================================================================

def run_experiment(
    args: argparse.Namespace,
) -> list[TrialResult]:

    (
        frame,
        uv,
        target_points,
    ) = build_common_geometry()

    # -------------------------------------------------------------------------
    # Fixed GLOBAL_OPTIMAL design.
    # -------------------------------------------------------------------------

    support_ids = (
        exp1.make_support_ids()
    )

    beta_optimal = (
        exp1.make_optimal_beta(
            support_ids
        )
    )

    trans_limit = (
        MEDIUM_INIT_TRANSLATION_MM
    )

    rot_limit = (
        MEDIUM_INIT_ROTATION_DEG
    )

    noise_std = (
        DEFAULT_NOISE_STD_MM
    )

    results: list[
        TrialResult
    ] = []

    print(
        f"\nMonte Carlo: {args.trials} trials"
    )

    print(
        f"design = {DESIGN}"
    )

    print(
        f"init <= "
        f"({trans_limit:g} mm, "
        f"{rot_limit:g} deg)"
    )

    print(
        f"xz noise = {noise_std:g} mm"
    )

    print(
        "support IDs =",
        support_ids,
    )

    print(
        "beta [deg] =",
        beta_optimal,
    )

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
                    rot_limit
                ),
                max_translation_error_mm=(
                    trans_limit
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

        results.append(
            run_one_trial(
                trial=trial,
                beta_deg=(
                    beta_optimal
                ),
                uv=uv,
                target_points=(
                    target_points
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
                init_rotation_limit_deg=(
                    rot_limit
                ),
                init_translation_limit_mm=(
                    trans_limit
                ),
                actual_init_rotation_error_deg=(
                    actual_rot
                ),
                actual_init_translation_error_mm=(
                    actual_trans
                ),
                noise_std_mm=(
                    noise_std
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
            )

    return results


# =============================================================================
# CSV
# =============================================================================

def write_trial_csv(
    results: list[TrialResult],
    path: Path,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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
        else float("nan")
    )


def write_summary_csv(
    results: list[TrialResult],
    path: Path,
) -> None:

    fieldnames = [
        "design",
        "initial_condition",
        "init_rotation_limit_deg",
        "init_translation_limit_mm",
        "noise_std_mm",
        "trials",
        "solver_success_count",
        "solver_success_rate",
        "accuracy_success_count",
        "accuracy_success_rate",
        "median_translation_error_mm",
        "p95_translation_error_mm",
        "median_rotation_error_deg",
        "p95_rotation_error_deg",
        "median_final_rms_mm",
    ]

    successful = [
        result
        for result
        in results
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

    first = results[0]

    row = {
        "design": DESIGN,
        "initial_condition": (
            first.initial_condition
        ),
        "init_rotation_limit_deg": (
            first.init_rotation_limit_deg
        ),
        "init_translation_limit_mm": (
            first.init_translation_limit_mm
        ),
        "noise_std_mm": (
            first.noise_std_mm
        ),
        "trials": len(
            results
        ),
        "solver_success_count": int(
            sum(
                result.solver_success
                for result
                in results
            )
        ),
        "solver_success_rate": float(
            np.mean(
                [
                    result.solver_success
                    for result
                    in results
                ]
            )
        ),
        "accuracy_success_count": int(
            sum(
                passes_accuracy_threshold(
                    result
                )
                for result
                in results
            )
        ),
        "accuracy_success_rate": float(
            np.mean(
                [
                    passes_accuracy_threshold(
                        result
                    )
                    for result
                    in results
                ]
            )
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
    }

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as stream:

        writer = csv.DictWriter(
            stream,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerow(
            row
        )


# =============================================================================
# JSON summary
# =============================================================================

def _error_statistics(
    values: list[float],
) -> dict[
    str,
    float | int | None,
]:

    finite = np.asarray(
        values,
        dtype=float,
    )

    finite = finite[
        np.isfinite(
            finite
        )
    ]

    if finite.size == 0:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p95": None,
            "max": None,
        }

    return {
        "count": int(
            finite.size
        ),
        "mean": float(
            np.mean(
                finite
            )
        ),
        "median": float(
            np.median(
                finite
            )
        ),
        "p95": float(
            np.percentile(
                finite,
                95,
            )
        ),
        "max": float(
            np.max(
                finite
            )
        ),
    }


def _outlier_record(
    result: TrialResult,
) -> dict[
    str,
    float | int,
]:

    return {
        "trial": int(
            result.trial
        ),
        "translation_error_mm": float(
            result.translation_error_mm
        ),
        "rotation_error_deg": float(
            result.rotation_error_deg
        ),
        "actual_init_translation_error_mm": float(
            result.actual_init_translation_error_mm
        ),
        "actual_init_rotation_error_deg": float(
            result.actual_init_rotation_error_deg
        ),
    }


def write_summary_json(
    results: list[TrialResult],
    path: Path,
) -> None:

    successful = [
        result
        for result
        in results
        if result.solver_success
    ]

    finite = [
        result
        for result
        in results
        if (
            np.isfinite(
                result.translation_error_mm
            )
            and np.isfinite(
                result.rotation_error_deg
            )
        )
    ]

    translation_outliers = sorted(
        [
            result
            for result
            in finite
            if (
                result.translation_error_mm
                >= LARGE_TRANSLATION_OUTLIER_MM
            )
        ],
        key=lambda result: (
            result.translation_error_mm
        ),
        reverse=True,
    )

    rotation_outliers = sorted(
        [
            result
            for result
            in finite
            if (
                result.rotation_error_deg
                >= LARGE_ROTATION_OUTLIER_DEG
            )
        ],
        key=lambda result: (
            result.rotation_error_deg
        ),
        reverse=True,
    )

    outlier_records = sorted(
        {
            result.trial: result
            for result
            in (
                *translation_outliers,
                *rotation_outliers,
            )
        }.values(),
        key=lambda result: max(
            result.translation_error_mm
            / LARGE_TRANSLATION_OUTLIER_MM,
            result.rotation_error_deg
            / LARGE_ROTATION_OUTLIER_DEG,
        ),
        reverse=True,
    )

    accuracy_filtered = [
        result
        for result
        in successful
        if passes_accuracy_threshold(
            result
        )
    ]

    def statistics(
        rows: list[TrialResult],
    ) -> dict[
        str,
        object,
    ]:
        return {
            "count": len(
                rows
            ),
            "translation_error_mm": (
                _error_statistics(
                    [
                        result.translation_error_mm
                        for result
                        in rows
                    ]
                )
            ),
            "rotation_error_deg": (
                _error_statistics(
                    [
                        result.rotation_error_deg
                        for result
                        in rows
                    ]
                )
            ),
        }

    payload: dict[
        str,
        object,
    ] = {
        "schema": (
            "exp1_global_optimal_monte_carlo_summary"
        ),
        "design": DESIGN,
        "accuracy_thresholds": {
            "translation_mm": (
                ACCURACY_TRANSLATION_THRESHOLD_MM
            ),
            "rotation_deg": (
                ACCURACY_ROTATION_THRESHOLD_DEG
            ),
        },
        "large_outlier_thresholds": {
            "translation_mm": (
                LARGE_TRANSLATION_OUTLIER_MM
            ),
            "rotation_deg": (
                LARGE_ROTATION_OUTLIER_DEG
            ),
        },
        "trials": len(
            results
        ),
        "solver_success": {
            "count": int(
                sum(
                    result.solver_success
                    for result
                    in results
                )
            ),
            "rate": float(
                np.mean(
                    [
                        result.solver_success
                        for result
                        in results
                    ]
                )
            ),
        },
        "accuracy_success": {
            "count": int(
                sum(
                    passes_accuracy_threshold(
                        result
                    )
                    for result
                    in results
                )
            ),
            "rate": float(
                np.mean(
                    [
                        passes_accuracy_threshold(
                            result
                        )
                        for result
                        in results
                    ]
                )
            ),
        },
        "all_solver_success_results": (
            statistics(
                successful
            )
        ),
        "accuracy_success_results_only": (
            statistics(
                accuracy_filtered
            )
        ),
        "large_outliers": {
            "translation_ge_10_mm_count": len(
                translation_outliers
            ),
            "rotation_ge_5_deg_count": len(
                rotation_outliers
            ),
            "unique_trial_count": len(
                outlier_records
            ),
            "records": [
                _outlier_record(
                    result
                )
                for result
                in outlier_records
            ],
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
            allow_nan=False,
        )

        stream.write(
            "\n"
        )


# =============================================================================
# Plot helpers
# =============================================================================

def create_error_boxplots(
    results: list[TrialResult],
    path: Path,
    *,
    exclude_large_trials: bool = False,
):
    excluded_trials: set[int] = set()

    if exclude_large_trials:
        excluded_trials = {
            result.trial
            for result
            in results
            if (
                np.isfinite(
                    result.translation_error_mm
                )
                and np.isfinite(
                    result.rotation_error_deg
                )
                and (
                    result.translation_error_mm
                    >= PLOT_EXCLUSION_TRANSLATION_MM
                    or result.rotation_error_deg
                    >= PLOT_EXCLUSION_ROTATION_DEG
                )
            )
        }

    usable = [
        result
        for result
        in results
        if (
            result.solver_success
            and result.trial
            not in excluded_trials
        )
    ]

    translation = np.asarray(
        [
            result.translation_error_mm
            for result
            in usable
        ],
        dtype=float,
    )

    rotation = np.asarray(
        [
            result.rotation_error_deg
            for result
            in usable
        ],
        dtype=float,
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(
            7.0,
            8.0,
        ),
        constrained_layout=True,
    )

    axes[0].boxplot(
        [
            translation
            if translation.size
            else np.asarray(
                [np.nan]
            )
        ],
        labels=[
            DESIGN
        ],
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
            rotation
            if rotation.size
            else np.asarray(
                [np.nan]
            )
        ],
        labels=[
            DESIGN
        ],
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
        "GLOBAL_OPTIMAL Monte-Carlo "
        "final calibration errors"
        f"\nxz noise sigma = "
        f"{DEFAULT_NOISE_STD_MM:g} mm"
    )

    if exclude_large_trials:
        title += (
            f"\nremoved "
            f"{len(excluded_trials)}/"
            f"{len(results)} trials with "
            f"T>={PLOT_EXCLUSION_TRANSLATION_MM:g} mm "
            f"or "
            f"R>={PLOT_EXCLUSION_ROTATION_DEG:g} deg"
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
    solver_rate = float(
        np.mean(
            [
                result.solver_success
                for result
                in results
            ]
        )
    )

    accuracy_rate = float(
        np.mean(
            [
                passes_accuracy_threshold(
                    result
                )
                for result
                in results
            ]
        )
    )

    rates = np.asarray(
        [
            solver_rate,
            accuracy_rate,
        ],
        dtype=float,
    )

    labels = [
        "Solver success",
        "Accuracy success",
    ]

    x = np.arange(
        len(
            labels
        ),
        dtype=float,
    )

    fig, axis = plt.subplots(
        figsize=(
            7.0,
            5.2,
        ),
        constrained_layout=True,
    )

    bars = axis.bar(
        x,
        rates,
        width=0.58,
    )

    for (
        bar,
        rate,
    ) in zip(
        bars,
        rates,
    ):
        successes = int(
            round(
                rate
                * len(results)
            )
        )

        axis.text(
            bar.get_x()
            + bar.get_width() / 2,
            min(
                rate + 0.025,
                1.055,
            ),
            f"{100.0 * rate:.1f}%\n"
            f"({successes}/{len(results)})",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    axis.set_xticks(
        x,
        labels,
    )

    axis.set_ylabel(
        "Success rate"
    )

    axis.set_ylim(
        0.0,
        1.12,
    )

    axis.set_yticks(
        np.linspace(
            0.0,
            1.0,
            6,
        )
    )

    axis.yaxis.set_major_formatter(
        lambda value, _position:
        f"{100 * value:.0f}%"
    )

    axis.grid(
        axis="y",
        alpha=0.25,
    )

    axis.set_title(
        "GLOBAL_OPTIMAL Monte-Carlo success rates\n"
        "medium initialization "
        "(100 mm, 15 deg), "
        "xz noise sigma=0.25 mm"
    )

    fig.savefig(
        path,
        dpi=180,
        bbox_inches="tight",
    )

    return fig


# =============================================================================
# Terminal summary
# =============================================================================

def print_summary(
    results: list[TrialResult],
) -> None:

    successful = [
        result
        for result
        in results
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

    solver_count = int(
        sum(
            result.solver_success
            for result
            in results
        )
    )

    accuracy_count = int(
        sum(
            passes_accuracy_threshold(
                result
            )
            for result
            in results
        )
    )

    large_outlier_count = int(
        sum(
            (
                np.isfinite(
                    result.translation_error_mm
                )
                and np.isfinite(
                    result.rotation_error_deg
                )
                and (
                    result.translation_error_mm
                    >= LARGE_TRANSLATION_OUTLIER_MM
                    or result.rotation_error_deg
                    >= LARGE_ROTATION_OUTLIER_DEG
                )
            )
            for result
            in results
        )
    )

    print(
        "\n"
        + "=" * 72
    )

    print(
        "GLOBAL_OPTIMAL MONTE-CARLO SUMMARY"
    )

    print(
        "=" * 72
    )

    print(
        f"trials            : "
        f"{len(results)}"
    )

    print(
        f"solver success    : "
        f"{solver_count}/{len(results)} "
        f"({solver_count / len(results):.1%})"
    )

    print(
        f"accuracy success  : "
        f"{accuracy_count}/{len(results)} "
        f"({accuracy_count / len(results):.1%})"
    )

    print(
        "translation median/P95 [mm] : "
        f"{percentile(translation, 50):.6g} / "
        f"{percentile(translation, 95):.6g}"
    )

    print(
        "rotation median/P95 [deg]   : "
        f"{percentile(rotation, 50):.6g} / "
        f"{percentile(rotation, 95):.6g}"
    )

    print(
        "large outlier trials        : "
        f"{large_outlier_count}"
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

    results = (
        run_experiment(
            args
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

    config_path = (
        args.output_dir
        / "config.json"
    )

    write_trial_csv(
        results,
        trial_path,
    )

    write_summary_csv(
        results,
        summary_path,
    )

    write_summary_json(
        results,
        summary_json_path,
    )

    support_ids = (
        exp1.make_support_ids()
    )

    beta_optimal = (
        exp1.make_optimal_beta(
            support_ids
        )
    )

    with config_path.open(
        "w",
        encoding="utf-8",
    ) as stream:

        json.dump(
            {
                "design": DESIGN,
                "trials": args.trials,
                "initial_condition": "medium",
                "init_rotation_deg": (
                    MEDIUM_INIT_ROTATION_DEG
                ),
                "init_translation_mm": (
                    MEDIUM_INIT_TRANSLATION_MM
                ),
                "noise_std_mm": (
                    DEFAULT_NOISE_STD_MM
                ),
                "gt_sampling": (
                    "sample_random_handeye_per_trial"
                ),
                "gt_translation_component_range_mm": (
                    GT_TRANSLATION_COMPONENT_RANGE_MM
                ),
                "gt_euler_component_range_deg": (
                    GT_EULER_COMPONENT_RANGE_DEG
                ),
                "seed": args.seed,
                "max_iter": args.max_iter,
                "tol": args.tol,
                "nonlinear_max_nfev": (
                    args.nonlinear_max_nfev
                ),
                "noise_axis": "xz",
                "support_ids": (
                    support_ids.tolist()
                ),
                "beta_optimal_deg": (
                    beta_optimal.tolist()
                ),
                "accuracy_threshold_translation_mm": (
                    ACCURACY_TRANSLATION_THRESHOLD_MM
                ),
                "accuracy_threshold_rotation_deg": (
                    ACCURACY_ROTATION_THRESHOLD_DEG
                ),
                "pose_source": (
                    "well_made_experiment.exp1_optimal"
                ),
                "protocol_sources": [
                    (
                        "result_refit/"
                        "fair_plane_initialization_shared_global_"
                        "init_t100_r15_direction_norm_axis_angle/"
                        "medium_t100_r15"
                    ),
                    (
                        "experiments/"
                        "lhs_initialization_robustness/config.json "
                        "(noise std 0.25 mm)"
                    ),
                    (
                        "laser_handeye.simulation."
                        "sample_random_handeye"
                    ),
                ],
            },
            stream,
            indent=2,
        )

        stream.write(
            "\n"
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
    ]

    print_summary(
        results
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
    )

    if args.show:
        plt.show()

    for figure in figures:
        plt.close(
            figure
        )


if __name__ == "__main__":
    main()