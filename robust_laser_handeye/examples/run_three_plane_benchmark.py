from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from laser_handeye.benchmark_analysis import (
    iter_frobenius_error,
    iter_physical_transform_errors,
    print_calibration_summary,
    save_calibration_plots,
    save_experiment_summary_csv,
    save_failures_csv,
    save_iteration_history_csv,
    save_plane_scene_plot_3d,
    save_results_csv,
)
from laser_handeye.calibration import (
    calibrate_planes,
    calibrate_with_known_planes,
    mean_self_fitted_plane_rms,
)
from laser_handeye.data import LaserScan
from laser_handeye.initialization import InitialGuessMode, make_initial_guess
from laser_handeye.geometry import fit_plane_pca
from laser_handeye.nonlinear_refinement import RobustLoss, refine_handeye_nonlinear
from laser_handeye.pose_generation import sample_robot_pose_for_plane
from laser_handeye.scene_generation import make_three_planes
from laser_handeye.se3 import (
    rot_error_deg,
    rotation_vector_error_deg,
    transform_points,
)
from laser_handeye.simulation import (
    sample_random_handeye,
    simulate_profile_on_plane,
)


@dataclass
class ThreePlaneTrialResult:
    system_idx: int
    mode: str
    init_mode: str
    plane_offset_mode: str
    n_planes: int
    n_scans: int
    n_points_per_scan: int
    converged: bool
    iterations: int
    rank_last: int
    cond_last: float
    trans_err_norm_mm: float
    rot_err_angle_deg: float
    err_tx_mm: float
    err_ty_mm: float
    err_tz_mm: float
    err_t_sensor_x_mm: float
    err_t_sensor_y_mm: float
    err_t_sensor_z_mm: float
    err_rx_deg: float
    err_ry_deg: float
    err_rz_deg: float
    err_r_sensor_x_deg: float
    err_r_sensor_y_deg: float
    err_r_sensor_z_deg: float
    paper_success: bool
    init_trans_err_norm_mm: float
    init_rot_err_angle_deg: float
    nonlinear_refined: bool = False
    nonlinear_success: bool = False
    nonlinear_nfev: int = 0
    nonlinear_initial_rms_mm: float = float("nan")
    nonlinear_final_rms_mm: float = float("nan")
    nonlinear_delta_translation_mm: float = float("nan")
    nonlinear_delta_rotation_deg: float = float("nan")
    plane_rms_history_mm: list[float] = field(default_factory=list)
    iter_T_frob_error: list[float] = field(default_factory=list)
    iter_translation_error_norm_mm: list[float] = field(default_factory=list)
    iter_rotation_geodesic_error_deg: list[float] = field(default_factory=list)


def sample_scan_dataset_for_planes(
    T_ef_s_true: np.ndarray,
    planes: list[tuple[np.ndarray, float]],
    rng: np.random.Generator,
    x_values: np.ndarray,
    poses_per_plane: int,
    noise_std: float,
    max_trials_per_plane: int = 50_000,
) -> dict[int, list[LaserScan]]:
    """Generate scans from three non-parallel calibration planes."""

    profile_depth_min_mm = 60.0
    profile_depth_max_mm = 150.0
    scans_by_plane: dict[int, list[LaserScan]] = {}

    for plane_id, (plane_n, plane_l) in enumerate(planes):
        scans: list[LaserScan] = []
        attempts = 0

        while len(scans) < poses_per_plane and attempts < max_trials_per_plane:
            attempts += 1

            try:
                T_base_ef = sample_robot_pose_for_plane(
                    T_ef_s=T_ef_s_true,
                    plane_n=plane_n,
                    plane_l=plane_l,
                    rng=rng,
                    tangent_range_mm=220.0,
                    depth_range_mm=(
                        profile_depth_min_mm,
                        profile_depth_max_mm,
                    ),
                    min_view_dot=0.0,
                )

                scan = simulate_profile_on_plane(
                    T_base_ef=T_base_ef,
                    T_ef_s_true=T_ef_s_true,
                    plane_n=plane_n,
                    plane_l=plane_l,
                    x_values=x_values,
                    noise_std=noise_std,
                    rng=rng,
                    plane_id=plane_id,
                    scan_id=len(scans),
                )
            except ValueError:
                continue

            points = scan.valid_points_s
            if len(points) != scan.num_points:
                continue

            z_min = float(np.min(points[:, 2]))
            z_max = float(np.max(points[:, 2]))
            if z_min < profile_depth_min_mm or z_max > profile_depth_max_mm:
                continue

            scans.append(scan)

        if len(scans) < poses_per_plane:
            raise RuntimeError(
                f"only generated {len(scans)} scans for plane {plane_id}; "
                f"requested {poses_per_plane}; attempts={attempts}"
            )

        scans_by_plane[plane_id] = scans

    return scans_by_plane



def estimate_planes_from_handeye(
    scans_by_plane: dict[int, list[LaserScan]],
    T_ef_s: np.ndarray,
) -> dict[int, tuple[np.ndarray, float]]:
    """Estimate each calibration plane once from the final hand-eye estimate.

    For every plane group, valid laser points are reconstructed in the robot
    base frame using ``T_ef_s``. A PCA plane is fitted once per group and the
    resulting planes are then held fixed during nonlinear SE(3) refinement.
    """
    estimated_planes: dict[int, tuple[np.ndarray, float]] = {}

    for plane_id, scans in scans_by_plane.items():
        point_sets: list[np.ndarray] = []

        for scan in scans:
            points_s = scan.valid_points_s
            if len(points_s) == 0:
                continue

            points_ef = transform_points(T_ef_s, points_s)
            points_base = transform_points(scan.T_base_ef, points_ef)
            point_sets.append(points_base)

        if not point_sets:
            raise ValueError(
                f"cannot estimate plane {plane_id}: no valid scan points"
            )

        points_base_all = np.vstack(point_sets)
        if len(points_base_all) < 3:
            raise ValueError(
                f"cannot estimate plane {plane_id} from fewer than three points"
            )

        plane_n, plane_l, _, _ = fit_plane_pca(points_base_all)
        plane_n = np.asarray(plane_n, dtype=float).reshape(3)
        plane_n /= np.linalg.norm(plane_n)
        plane_l = float(plane_l)

        # Keep a consistent representation n.T @ p = l with l >= 0.
        if plane_l < 0.0:
            plane_n = -plane_n
            plane_l = -plane_l

        estimated_planes[int(plane_id)] = (plane_n, plane_l)

    return estimated_planes


def make_and_save_debug_scene_plot(
    seed: int,
    x_values: np.ndarray,
    poses_per_plane: int,
    noise_std: float,
    out_path: Path,
    plane_size_mm: float = 1500.0,
) -> Path:
    """Generate one representative system and save its scene plot."""

    rng = np.random.default_rng(seed)
    T_true, _, _ = sample_random_handeye(rng)

    # Generate one random three-plane set for this debug system.
    planes = make_three_planes(rng=rng)

    scans_by_plane = sample_scan_dataset_for_planes(
        T_ef_s_true=T_true,
        planes=planes,
        rng=rng,
        x_values=x_values,
        poses_per_plane=poses_per_plane,
        noise_std=noise_std,
    )

    return save_plane_scene_plot_3d(
        T_ef_s_true=T_true,
        scans_by_plane=scans_by_plane,
        planes=planes,
        out_path=out_path,
        plane_size_mm=plane_size_mm,
    )


def run_one_trial(
    system_idx: int,
    rng: np.random.Generator,
    x_values: np.ndarray,
    poses_per_plane: int,
    noise_std: float,
    mode: str,
    max_iter: int,
    tol: float,
    init_mode: InitialGuessMode,
    rel_offset: float,
    init_translation_range_mm: float,
    init_angle_range_deg: float,
    plane_offset_mode: str,
    nonlinear_refine: bool,
    nonlinear_max_nfev: int,
    nonlinear_ftol: float,
    nonlinear_xtol: float,
    nonlinear_gtol: float,
    nonlinear_loss: RobustLoss,
    nonlinear_f_scale_mm: float,
    gt_rng: np.random.Generator | None = None,
    init_rng: np.random.Generator | None = None,
) -> ThreePlaneTrialResult:
    T_true, true_angles_deg, true_translation_mm = sample_random_handeye(
        rng if gt_rng is None else gt_rng
    )

    # One new random three-plane set per trial.
    # The same set is reused throughout this trial.
    planes = make_three_planes(rng=rng)

    scans_by_plane = sample_scan_dataset_for_planes(
        T_ef_s_true=T_true,
        planes=planes,
        rng=rng,
        x_values=x_values,
        poses_per_plane=poses_per_plane,
        noise_std=noise_std,
    )

    plane_rms_history_mm: list[float] = []
    iter_T_frob_error: list[float] = []
    init_trans_err_norm_mm = float("nan")
    init_rot_err_angle_deg = float("nan")
    nonlinear_success = False
    nonlinear_nfev = 0
    nonlinear_initial_rms_mm = float("nan")
    nonlinear_final_rms_mm = float("nan")
    nonlinear_delta_translation_mm = float("nan")
    nonlinear_delta_rotation_deg = float("nan")

    if mode == "known":
        known_planes = {
            plane_id: (scans_by_plane[plane_id], plane_n, plane_l)
            for plane_id, (plane_n, plane_l) in enumerate(planes)
        }
        T_linear = calibrate_with_known_planes(known_planes)
        T_est = T_linear
        converged = True
        iterations = 1
        rank_last = -1
        cond_last = float("nan")
        transform_history = [T_linear]
        plane_rms_history_mm = [
            mean_self_fitted_plane_rms(scans_by_plane, T_linear)
        ]

        if nonlinear_refine:
            nonlinear_result = refine_handeye_nonlinear(
                scans_by_plane,
                T_linear,
                plane_mode="fixed",
                planes=planes,
                loss=nonlinear_loss,
                f_scale_mm=nonlinear_f_scale_mm,
                max_nfev=nonlinear_max_nfev,
                ftol=nonlinear_ftol,
                xtol=nonlinear_xtol,
                gtol=nonlinear_gtol,
            )
            T_est = nonlinear_result.T_ef_s
            converged = bool(nonlinear_result.success)
            nonlinear_success = bool(nonlinear_result.success)
            nonlinear_nfev = int(nonlinear_result.nfev)
            nonlinear_initial_rms_mm = float(nonlinear_result.initial_rms_mm)
            nonlinear_final_rms_mm = float(nonlinear_result.final_rms_mm)
            nonlinear_delta_translation_mm = float(
                nonlinear_result.delta_translation_mm
            )
            nonlinear_delta_rotation_deg = float(
                nonlinear_result.delta_rotation_deg
            )
            transform_history.append(T_est)
            plane_rms_history_mm.append(
                mean_self_fitted_plane_rms(scans_by_plane, T_est)
            )

        iter_T_frob_error = iter_frobenius_error(transform_history, T_true)

    elif mode == "unknown":
        T_init = make_initial_guess(
            reference_angles_deg=true_angles_deg,
            reference_translation_mm=true_translation_mm,
            rng=rng if init_rng is None else init_rng,
            mode=init_mode,
            rel_offset=rel_offset,
            translation_range_mm=init_translation_range_mm,
            angle_range_deg=init_angle_range_deg,
        )

        init_trans_err_norm_mm = float(
            np.linalg.norm(T_init[:3, 3] - T_true[:3, 3])
        )
        init_rot_err_angle_deg = float(
            rot_error_deg(T_init[:3, :3], T_true[:3, :3])
        )

        result = calibrate_planes(
            scans_by_plane,
            T_init=T_init,
            max_iter=max_iter,
            tol=tol,
            plane_offset_mode=plane_offset_mode,
        )

        T_est = result.T_ef_s
        converged = bool(result.converged)
        iterations = int(result.iterations)
        rank_last = int(result.rank_history[-1]) if result.rank_history else -1
        cond_last = (
            float(result.cond_history[-1])
            if result.cond_history
            else float("nan")
        )
        plane_rms_history_mm = [
            float(value) for value in result.plane_rms_history
        ]
        transform_history = list(result.T_history)

        if nonlinear_refine:
            estimated_planes = estimate_planes_from_handeye(
                scans_by_plane=scans_by_plane,
                T_ef_s=T_est,
            )

            nonlinear_result = refine_handeye_nonlinear(
                scans_by_plane,
                T_est,
                plane_mode="fixed",
                planes=estimated_planes,
                loss=nonlinear_loss,
                f_scale_mm=nonlinear_f_scale_mm,
                max_nfev=nonlinear_max_nfev,
                ftol=nonlinear_ftol,
                xtol=nonlinear_xtol,
                gtol=nonlinear_gtol,
            )
            T_est = nonlinear_result.T_ef_s
            converged = bool(nonlinear_result.success)
            nonlinear_success = bool(nonlinear_result.success)
            nonlinear_nfev = int(nonlinear_result.nfev)
            nonlinear_initial_rms_mm = float(nonlinear_result.initial_rms_mm)
            nonlinear_final_rms_mm = float(nonlinear_result.final_rms_mm)
            nonlinear_delta_translation_mm = float(
                nonlinear_result.delta_translation_mm
            )
            nonlinear_delta_rotation_deg = float(
                nonlinear_result.delta_rotation_deg
            )
            transform_history.append(T_est)
            plane_rms_history_mm.append(
                mean_self_fitted_plane_rms(scans_by_plane, T_est)
            )

        iter_T_frob_error = iter_frobenius_error(
            transform_history,
            T_true,
            T_initial=T_init,
        )

    else:
        raise ValueError("mode must be 'known' or 'unknown'")

    iter_translation_error_norm_mm, iter_rotation_geodesic_error_deg = (
        iter_physical_transform_errors(
            transform_history,
            T_true,
            T_initial=T_init if mode == "unknown" else None,
        )
    )

    translation_error = T_est[:3, 3] - T_true[:3, 3]
    rotation_vector_error = rotation_vector_error_deg(
        T_est[:3, :3],
        T_true[:3, :3],
    )
    translation_error_sensor = T_true[:3, :3].T @ translation_error
    rotation_error_sensor = T_true[:3, :3].T @ rotation_vector_error
    paper_success = bool(
        converged
        and iterations <= 2000
        and np.all(np.abs(translation_error) < 0.01)
    )

    return ThreePlaneTrialResult(
        system_idx=system_idx,
        mode=mode,
        init_mode=init_mode,
        plane_offset_mode=plane_offset_mode,
        n_planes=len(scans_by_plane),
        n_scans=sum(len(scans) for scans in scans_by_plane.values()),
        n_points_per_scan=len(x_values),
        converged=converged,
        iterations=iterations,
        rank_last=rank_last,
        cond_last=cond_last,
        trans_err_norm_mm=float(np.linalg.norm(translation_error)),
        rot_err_angle_deg=float(
            rot_error_deg(T_est[:3, :3], T_true[:3, :3])
        ),
        err_tx_mm=float(translation_error[0]),
        err_ty_mm=float(translation_error[1]),
        err_tz_mm=float(translation_error[2]),
        err_t_sensor_x_mm=float(translation_error_sensor[0]),
        err_t_sensor_y_mm=float(translation_error_sensor[1]),
        err_t_sensor_z_mm=float(translation_error_sensor[2]),
        err_rx_deg=float(rotation_vector_error[0]),
        err_ry_deg=float(rotation_vector_error[1]),
        err_rz_deg=float(rotation_vector_error[2]),
        err_r_sensor_x_deg=float(rotation_error_sensor[0]),
        err_r_sensor_y_deg=float(rotation_error_sensor[1]),
        err_r_sensor_z_deg=float(rotation_error_sensor[2]),
        paper_success=paper_success,
        init_trans_err_norm_mm=init_trans_err_norm_mm,
        init_rot_err_angle_deg=init_rot_err_angle_deg,
        nonlinear_refined=nonlinear_refine,
        nonlinear_success=nonlinear_success,
        nonlinear_nfev=nonlinear_nfev,
        nonlinear_initial_rms_mm=nonlinear_initial_rms_mm,
        nonlinear_final_rms_mm=nonlinear_final_rms_mm,
        nonlinear_delta_translation_mm=nonlinear_delta_translation_mm,
        nonlinear_delta_rotation_deg=nonlinear_delta_rotation_deg,
        plane_rms_history_mm=plane_rms_history_mm,
        iter_T_frob_error=iter_T_frob_error,
        iter_translation_error_norm_mm=iter_translation_error_norm_mm,
        iter_rotation_geodesic_error_deg=iter_rotation_geodesic_error_deg,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Three-plane benchmark for the unknown-plane calibration solver."
    )
    parser.add_argument("--systems", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--mode", choices=["known", "unknown"], default="unknown")
    parser.add_argument("--poses-per-plane", type=int, default=35)
    parser.add_argument("--profile-points", type=int, default=100)
    parser.add_argument("--profile-half-width", type=float, default=25.0)
    parser.add_argument("--noise-std", type=float, default=0.5)
    parser.add_argument("--max-iter", type=int, default=30)
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-9,
        help="convergence tolerance; a negative value forces max_iter iterations",
    )
    parser.add_argument(
        "--init-mode",
        choices=["relative", "carlson"],
        default="relative",
    )
    parser.add_argument("--rel-offset", type=float, default=0.1)
    parser.add_argument("--init-translation-range-mm", type=float, default=200.0)
    parser.add_argument("--init-angle-range-deg", type=float, default=30.0)
    parser.add_argument(
        "--plane-offset-mode",
        choices=["joint", "fitted"],
        default="joint",
        help="use the same linear plane-offset update as the single-plane benchmark",
    )
    parser.add_argument(
        "--nonlinear-refine",
        action="store_true",
        help="run a final six-parameter SE(3) nonlinear least-squares refinement",
    )
    parser.add_argument("--nonlinear-max-nfev", type=int, default=200)
    parser.add_argument("--nonlinear-ftol", type=float, default=1e-10)
    parser.add_argument("--nonlinear-xtol", type=float, default=1e-10)
    parser.add_argument("--nonlinear-gtol", type=float, default=1e-10)
    parser.add_argument(
        "--nonlinear-loss",
        choices=["linear", "soft_l1", "huber", "cauchy", "arctan"],
        default="linear",
    )
    parser.add_argument("--nonlinear-f-scale-mm", type=float, default=1.0)
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("three_plane_benchmark.csv"),
    )
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument("--failures-csv", type=Path, default=None)
    parser.add_argument("--debug-scene-plot", type=Path, default=None)
    parser.add_argument("--debug-scene-seed", type=int, default=None)
    parser.add_argument("--debug-scene-plane-size", type=float, default=600.0)
    parser.add_argument("--plot-dir", type=Path, default=None)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--log-every", type=int, default=5)
    args = parser.parse_args()

    x_values = np.linspace(
        -args.profile_half_width,
        args.profile_half_width,
        args.profile_points,
    )

    if args.debug_scene_plot is not None:
        scene_seed = (
            args.seed if args.debug_scene_seed is None else args.debug_scene_seed
        )
        debug_path = make_and_save_debug_scene_plot(
            seed=scene_seed,
            x_values=x_values,
            poses_per_plane=args.poses_per_plane,
            noise_std=args.noise_std,
            out_path=args.debug_scene_plot,
            plane_size_mm=args.debug_scene_plane_size,
        )
        print(f"saved debug scene plot: {debug_path}")

    results: list[ThreePlaneTrialResult] = []
    failures = 0
    failure_records: list[dict[str, object]] = []
    first_failure: str | None = None
    start_time = time.perf_counter()
    log_every = max(1, args.log_every)

    if args.verbose:
        print(
            f"[three-plane] start | systems={args.systems} | mode={args.mode} | "
            f"poses/plane={args.poses_per_plane} | noise={args.noise_std} mm | "
            f"init_mode={args.init_mode} | offset_mode={args.plane_offset_mode} | "
            f"nonlinear={args.nonlinear_refine}"
        )

    for system_idx in range(args.systems):
        gt_seed, scene_seed, init_seed = np.random.SeedSequence(
            [args.seed, system_idx]
        ).spawn(3)
        gt_rng = np.random.default_rng(gt_seed)
        trial_rng = np.random.default_rng(scene_seed)
        init_rng = np.random.default_rng(init_seed)
        try:
            result = run_one_trial(
                system_idx=system_idx,
                rng=trial_rng,
                x_values=x_values,
                poses_per_plane=args.poses_per_plane,
                noise_std=args.noise_std,
                mode=args.mode,
                max_iter=args.max_iter,
                tol=args.tol,
                init_mode=args.init_mode,
                rel_offset=args.rel_offset,
                init_translation_range_mm=args.init_translation_range_mm,
                init_angle_range_deg=args.init_angle_range_deg,
                plane_offset_mode=args.plane_offset_mode,
                nonlinear_refine=args.nonlinear_refine,
                nonlinear_max_nfev=args.nonlinear_max_nfev,
                nonlinear_ftol=args.nonlinear_ftol,
                nonlinear_xtol=args.nonlinear_xtol,
                nonlinear_gtol=args.nonlinear_gtol,
                nonlinear_loss=args.nonlinear_loss,
                nonlinear_f_scale_mm=args.nonlinear_f_scale_mm,
                gt_rng=gt_rng,
                init_rng=init_rng,
            )
            results.append(result)
        except Exception as exc:
            failures += 1
            failure_records.append(
                {
                    "system_idx": system_idx,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
            if first_failure is None:
                first_failure = f"{type(exc).__name__}: {exc}"
            if args.verbose:
                print(
                    f"[trial {system_idx:04d}] failed | "
                    f"{type(exc).__name__}: {exc}"
                )

        if args.verbose and (
            (system_idx + 1) % log_every == 0
            or system_idx + 1 == args.systems
        ):
            elapsed = time.perf_counter() - start_time

            if results:
                median_translation_error = float(
                    np.median(
                        [result.trans_err_norm_mm for result in results]
                    )
                )
                converged = sum(result.converged for result in results)
                succeeded = sum(result.paper_success for result in results)
                print(
                    f"[three-plane] progress {system_idx + 1}/{args.systems} | "
                    f"ok={len(results)} | fail={failures} | "
                    f"conv={converged}/{len(results)} | "
                    f"paper_success={succeeded}/{len(results)} | "
                    f"median_t={median_translation_error:.6g} mm | "
                    f"elapsed={elapsed:.1f}s"
                )
            else:
                print(
                    f"[three-plane] progress {system_idx + 1}/{args.systems} | "
                    f"ok=0 | fail={failures} | elapsed={elapsed:.1f}s"
                )

    if args.verbose:
        elapsed = time.perf_counter() - start_time
        print(
            f"[three-plane] finished | ok={len(results)} | "
            f"failed={failures} | elapsed={elapsed:.1f}s"
        )

    print_calibration_summary(results)
    summary_csv = args.summary_csv or args.csv.with_name(
        f"{args.csv.stem}_summary.csv"
    )
    failures_csv = args.failures_csv or args.csv.with_name(
        f"{args.csv.stem}_failures.csv"
    )
    save_experiment_summary_csv(
        results,
        summary_csv,
        requested_systems=args.systems,
        failed_systems=failures,
        config={
            **vars(args),
            "elapsed_seconds": time.perf_counter() - start_time,
        },
    )
    save_failures_csv(failure_records, failures_csv)
    print(f"saved summary: {summary_csv}")
    print(f"saved failures: {failures_csv}")
    if not results:
        if first_failure is not None:
            print(f"first failure: {first_failure}")
        raise SystemExit(2)

    save_results_csv(results, args.csv)
    print(f"saved trials: {args.csv}")
    iterations_csv = args.csv.with_name(f"{args.csv.stem}_iterations.csv")
    save_iteration_history_csv(results, iterations_csv)
    print(f"saved iterations: {iterations_csv}")

    if args.no_plots:
        return

    plot_dir = (
        args.plot_dir
        if args.plot_dir is not None
        else args.csv.parent / f"{args.csv.stem}_plots"
    )
    for path in save_calibration_plots(
        results,
        plot_dir,
        sensor_noise_std_mm=args.noise_std,
    ):
        print(f"saved plot: {path}")


if __name__ == "__main__":
    main()
