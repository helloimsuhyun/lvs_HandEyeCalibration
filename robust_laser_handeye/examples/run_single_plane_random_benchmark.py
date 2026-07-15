from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from laser_handeye.benchmark_analysis import (
    iter_frobenius_error,
    print_calibration_summary,
    save_calibration_plots,
    save_plane_scene_plot,
    save_results_csv,
)
from laser_handeye.calibration import calibrate_planes, calibrate_with_known_planes
from laser_handeye.data import LaserScan
from laser_handeye.initialization import InitialGuessMode, make_initial_guess
from laser_handeye.geometry import fit_plane_pca
from laser_handeye.nonlinear_refinement import RobustLoss, refine_handeye_nonlinear
from laser_handeye.pose_generation import sample_robot_pose_for_plane
from scipy.spatial.transform import Rotation
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
class SinglePlaneTrialResult:
    system_idx: int
    mode: str
    init_mode: str
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
    err_rx_deg: float
    err_ry_deg: float
    err_rz_deg: float
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

def sample_random_plane(
    rng: np.random.Generator,
    tilt_min_deg: float = 1.0,
    tilt_max_deg: float = 15.0,
    yaw_min_deg: float = -180.0,
    yaw_max_deg: float = 180.0,
    center_x_range_mm: tuple[float, float] = (-100.0, 100.0),
    center_y_range_mm: tuple[float, float] = (-100.0, 100.0),
    center_z_range_mm: tuple[float, float] = (400.0, 550.0),
) -> tuple[np.ndarray, float]:
    """Sample one randomly positioned and oriented calibration plane.

    The plane is represented as:

        plane_n.T @ p = plane_l

    The X/Y tilt magnitudes are sampled away from zero so that the plane is
    not exactly aligned with the base coordinate system. Yaw is sampled over
    the requested range.
    """
    if tilt_min_deg < 0.0 or tilt_max_deg < tilt_min_deg:
        raise ValueError("invalid plane tilt range")

    def sample_signed_tilt() -> float:
        magnitude = float(rng.uniform(tilt_min_deg, tilt_max_deg))
        sign = -1.0 if rng.random() < 0.5 else 1.0
        return sign * magnitude

    rx_deg = sample_signed_tilt()
    ry_deg = sample_signed_tilt()
    rz_deg = float(rng.uniform(yaw_min_deg, yaw_max_deg))

    R_base_plane = Rotation.from_euler(
        "xyz",
        [rx_deg, ry_deg, rz_deg],
        degrees=True,
    ).as_matrix()

    plane_n = R_base_plane[:, 2]
    plane_n = plane_n / np.linalg.norm(plane_n)

    plane_center = np.array(
        [
            rng.uniform(*center_x_range_mm),
            rng.uniform(*center_y_range_mm),
            rng.uniform(*center_z_range_mm),
        ],
        dtype=float,
    )
    plane_l = float(plane_n @ plane_center)

    # Keep a consistent normal direction and non-negative offset.
    if plane_l < 0.0:
        plane_n = -plane_n
        plane_l = -plane_l

    return plane_n, plane_l


def sample_scan_dataset_for_plane(
    T_ef_s_true: np.ndarray,
    plane: tuple[np.ndarray, float],
    rng: np.random.Generator,
    x_values: np.ndarray,
    poses_per_plane: int,
    noise_std: float,
    tangent_range_mm: float,
    depth_range_mm: tuple[float, float],
    min_view_dot: float,
    max_trials_per_plane: int = 50_000,
) -> dict[int, list[LaserScan]]:
    """Generate random robot/sensor poses observing one calibration plane.

    No fixed nine-parameter scan pattern is used. Every accepted scan pose is
    sampled independently by ``sample_robot_pose_for_plane``.
    """
    plane_n, plane_l = plane
    depth_min_mm, depth_max_mm = map(float, depth_range_mm)

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
                tangent_range_mm=tangent_range_mm,
                depth_range_mm=(depth_min_mm, depth_max_mm),
                min_view_dot=min_view_dot,
            )

            scan = simulate_profile_on_plane(
                T_base_ef=T_base_ef,
                T_ef_s_true=T_ef_s_true,
                plane_n=plane_n,
                plane_l=plane_l,
                x_values=x_values,
                noise_std=noise_std,
                rng=rng,
                plane_id=0,
                scan_id=len(scans),
            )
        except ValueError:
            continue

        points = scan.valid_points_s
        if len(points) != scan.num_points:
            continue

        z_min = float(np.min(points[:, 2]))
        z_max = float(np.max(points[:, 2]))
        if z_min < depth_min_mm or z_max > depth_max_mm:
            continue

        scans.append(scan)

    if len(scans) < poses_per_plane:
        raise RuntimeError(
            f"only generated {len(scans)} scans for the single plane; "
            f"requested {poses_per_plane}; attempts={attempts}"
        )

    return {0: scans}



def estimate_plane_from_handeye(
    scans: list[LaserScan],
    T_ef_s: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Estimate the calibration plane once from the final hand-eye estimate.

    All valid laser points are reconstructed in the robot base frame using
    ``T_ef_s``. A PCA plane is fitted once and then held fixed during the
    nonlinear SE(3) refinement.
    """
    point_sets: list[np.ndarray] = []

    for scan in scans:
        points_s = scan.valid_points_s
        if len(points_s) == 0:
            continue

        points_ef = transform_points(T_ef_s, points_s)
        points_base = transform_points(scan.T_base_ef, points_ef)
        point_sets.append(points_base)

    if not point_sets:
        raise ValueError("cannot estimate a plane: no valid scan points")

    points_base_all = np.vstack(point_sets)
    if len(points_base_all) < 3:
        raise ValueError("cannot estimate a plane from fewer than three points")

    plane_n, plane_l, _, _ = fit_plane_pca(points_base_all)
    plane_n = np.asarray(plane_n, dtype=float).reshape(3)
    plane_n /= np.linalg.norm(plane_n)
    plane_l = float(plane_l)

    # Keep a consistent representation n.T @ p = l with l >= 0.
    if plane_l < 0.0:
        plane_n = -plane_n
        plane_l = -plane_l

    return plane_n, plane_l


def make_and_save_debug_scene_plot(
    seed: int,
    x_values: np.ndarray,
    poses_per_plane: int,
    noise_std: float,
    out_path: Path,
    plane_size_mm: float,
    tangent_range_mm: float,
    depth_range_mm: tuple[float, float],
    min_view_dot: float,
    plane_tilt_min_deg: float,
    plane_tilt_max_deg: float,
    plane_yaw_min_deg: float,
    plane_yaw_max_deg: float,
    plane_center_xy_range_mm: float,
    plane_center_z_range_mm: tuple[float, float],
) -> Path:
    """Generate one representative random single-plane system."""

    rng = np.random.default_rng(seed)
    T_true, _, _ = sample_random_handeye(rng)
    plane = sample_random_plane(
        rng=rng,
        tilt_min_deg=plane_tilt_min_deg,
        tilt_max_deg=plane_tilt_max_deg,
        yaw_min_deg=plane_yaw_min_deg,
        yaw_max_deg=plane_yaw_max_deg,
        center_x_range_mm=(
            -plane_center_xy_range_mm,
            plane_center_xy_range_mm,
        ),
        center_y_range_mm=(
            -plane_center_xy_range_mm,
            plane_center_xy_range_mm,
        ),
        center_z_range_mm=plane_center_z_range_mm,
    )
    scans_by_plane = sample_scan_dataset_for_plane(
        T_ef_s_true=T_true,
        plane=plane,
        rng=rng,
        x_values=x_values,
        poses_per_plane=poses_per_plane,
        noise_std=noise_std,
        tangent_range_mm=tangent_range_mm,
        depth_range_mm=depth_range_mm,
        min_view_dot=min_view_dot,
    )

    return save_plane_scene_plot(
        T_ef_s_true=T_true,
        scans_by_plane=scans_by_plane,
        planes=[plane],
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
    nonlinear_refine: bool,
    nonlinear_max_nfev: int,
    nonlinear_loss: RobustLoss,
    nonlinear_f_scale_mm: float,
    tangent_range_mm: float,
    depth_range_mm: tuple[float, float],
    min_view_dot: float,
    plane_tilt_min_deg: float,
    plane_tilt_max_deg: float,
    plane_yaw_min_deg: float,
    plane_yaw_max_deg: float,
    plane_center_xy_range_mm: float,
    plane_center_z_range_mm: tuple[float, float],
) -> SinglePlaneTrialResult:
    T_true, true_angles_deg, true_translation_mm = sample_random_handeye(rng)
    plane = sample_random_plane(
        rng=rng,
        tilt_min_deg=plane_tilt_min_deg,
        tilt_max_deg=plane_tilt_max_deg,
        yaw_min_deg=plane_yaw_min_deg,
        yaw_max_deg=plane_yaw_max_deg,
        center_x_range_mm=(-plane_center_xy_range_mm, plane_center_xy_range_mm),
        center_y_range_mm=(-plane_center_xy_range_mm, plane_center_xy_range_mm),
        center_z_range_mm=plane_center_z_range_mm,
    )
    scans_by_plane = sample_scan_dataset_for_plane(
        T_ef_s_true=T_true,
        plane=plane,
        rng=rng,
        x_values=x_values,
        poses_per_plane=poses_per_plane,
        noise_std=noise_std,
        tangent_range_mm=tangent_range_mm,
        depth_range_mm=depth_range_mm,
        min_view_dot=min_view_dot,
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
        plane_n, plane_l = plane
        known_planes = {
            0: (scans_by_plane[0], plane_n, plane_l)
        }
        T_linear = calibrate_with_known_planes(known_planes)
        T_est = T_linear
        converged = True
        iterations = 1
        rank_last = -1
        cond_last = float("nan")
        transform_history = [T_linear]

        if nonlinear_refine:
            nonlinear_result = refine_handeye_nonlinear(
                scans_by_plane,
                T_linear,
                plane_mode="fixed",
                planes=[plane],
                loss=nonlinear_loss,
                f_scale_mm=nonlinear_f_scale_mm,
                max_nfev=nonlinear_max_nfev,
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

        iter_T_frob_error = iter_frobenius_error(transform_history, T_true)

    elif mode == "unknown":
        T_init = make_initial_guess(
            reference_angles_deg=true_angles_deg,
            reference_translation_mm=true_translation_mm,
            rng=rng,
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
            estimated_plane = estimate_plane_from_handeye(
                scans=scans_by_plane[0],
                T_ef_s=T_est,
            )

            nonlinear_result = refine_handeye_nonlinear(
                scans_by_plane,
                T_est,
                plane_mode="fixed",
                planes={0: estimated_plane},
                loss=nonlinear_loss,
                f_scale_mm=nonlinear_f_scale_mm,
                max_nfev=nonlinear_max_nfev,
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

        iter_T_frob_error = iter_frobenius_error(
            transform_history,
            T_true,
            T_initial=T_init,
        )

    else:
        raise ValueError("mode must be 'known' or 'unknown'")

    translation_error = T_est[:3, 3] - T_true[:3, 3]
    rotation_vector_error = rotation_vector_error_deg(
        T_est[:3, :3],
        T_true[:3, :3],
    )
    paper_success = bool(
        converged
        and np.all(np.abs(translation_error) < 0.01)
        and np.all(np.abs(rotation_vector_error) < 5e-5)
    )

    return SinglePlaneTrialResult(
        system_idx=system_idx,
        mode=mode,
        init_mode=init_mode,
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
        err_rx_deg=float(rotation_vector_error[0]),
        err_ry_deg=float(rotation_vector_error[1]),
        err_rz_deg=float(rotation_vector_error[2]),
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
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Random single-plane benchmark for the hand-eye calibration solver."
    )
    parser.add_argument("--systems", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--mode", choices=["known", "unknown"], default="unknown")
    parser.add_argument("--poses-per-plane", type=int, default=10)
    parser.add_argument("--profile-points", type=int, default=100)
    parser.add_argument("--profile-half-width", type=float, default=25.0)
    parser.add_argument("--noise-std", type=float, default=0.5)
    parser.add_argument("--tangent-range-mm", type=float, default=220.0)
    parser.add_argument("--depth-min-mm", type=float, default=60.0)
    parser.add_argument("--depth-max-mm", type=float, default=150.0)
    parser.add_argument(
        "--min-view-dot",
        type=float,
        default=0.0,
        help="minimum accepted viewing-direction dot product",
    )
    parser.add_argument("--plane-tilt-min-deg", type=float, default=1.0)
    parser.add_argument("--plane-tilt-max-deg", type=float, default=15.0)
    parser.add_argument("--plane-yaw-min-deg", type=float, default=-180.0)
    parser.add_argument("--plane-yaw-max-deg", type=float, default=180.0)
    parser.add_argument("--plane-center-xy-range-mm", type=float, default=100.0)
    parser.add_argument("--plane-center-z-min-mm", type=float, default=400.0)
    parser.add_argument("--plane-center-z-max-mm", type=float, default=550.0)
    parser.add_argument("--max-iter", type=int, default=30)
    parser.add_argument(
        "--tol",
        type=float,
        default=-1.0,
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
        "--nonlinear-refine",
        action="store_true",
        help="run a final six-parameter SE(3) nonlinear least-squares refinement",
    )
    parser.add_argument("--nonlinear-max-nfev", type=int, default=200)
    parser.add_argument(
        "--nonlinear-loss",
        choices=["linear", "soft_l1", "huber", "cauchy", "arctan"],
        default="linear",
    )
    parser.add_argument("--nonlinear-f-scale-mm", type=float, default=1.0)
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("single_plane_random_benchmark.csv"),
    )
    parser.add_argument("--debug-scene-plot", type=Path, default=None)
    parser.add_argument("--debug-scene-seed", type=int, default=None)
    parser.add_argument("--debug-scene-plane-size", type=float, default=700.0)
    parser.add_argument("--plot-dir", type=Path, default=None)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--log-every", type=int, default=5)
    args = parser.parse_args()

    if args.depth_min_mm <= 0.0 or args.depth_max_mm <= args.depth_min_mm:
        parser.error("require 0 < depth-min-mm < depth-max-mm")
    if args.plane_tilt_min_deg < 0.0 or args.plane_tilt_max_deg < args.plane_tilt_min_deg:
        parser.error("invalid plane tilt range")
    if args.plane_center_xy_range_mm < 0.0:
        parser.error("plane-center-xy-range-mm must be non-negative")
    if args.plane_center_z_max_mm <= args.plane_center_z_min_mm:
        parser.error("invalid plane center Z range")

    depth_range_mm = (args.depth_min_mm, args.depth_max_mm)
    plane_center_z_range_mm = (
        args.plane_center_z_min_mm,
        args.plane_center_z_max_mm,
    )

    rng = np.random.default_rng(args.seed)
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
            tangent_range_mm=args.tangent_range_mm,
            depth_range_mm=depth_range_mm,
            min_view_dot=args.min_view_dot,
            plane_tilt_min_deg=args.plane_tilt_min_deg,
            plane_tilt_max_deg=args.plane_tilt_max_deg,
            plane_yaw_min_deg=args.plane_yaw_min_deg,
            plane_yaw_max_deg=args.plane_yaw_max_deg,
            plane_center_xy_range_mm=args.plane_center_xy_range_mm,
            plane_center_z_range_mm=plane_center_z_range_mm,
        )
        print(f"saved debug scene plot: {debug_path}")

    results: list[SinglePlaneTrialResult] = []
    failures = 0
    start_time = time.perf_counter()
    log_every = max(1, args.log_every)

    if args.verbose:
        print(
            f"[single-plane] start | systems={args.systems} | mode={args.mode} | "
            f"poses/plane={args.poses_per_plane} | noise={args.noise_std} mm | "
            f"init_mode={args.init_mode} | nonlinear={args.nonlinear_refine}"
        )

    for system_idx in range(args.systems):
        try:
            result = run_one_trial(
                system_idx=system_idx,
                rng=rng,
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
                nonlinear_refine=args.nonlinear_refine,
                nonlinear_max_nfev=args.nonlinear_max_nfev,
                nonlinear_loss=args.nonlinear_loss,
                nonlinear_f_scale_mm=args.nonlinear_f_scale_mm,
                tangent_range_mm=args.tangent_range_mm,
                depth_range_mm=depth_range_mm,
                min_view_dot=args.min_view_dot,
                plane_tilt_min_deg=args.plane_tilt_min_deg,
                plane_tilt_max_deg=args.plane_tilt_max_deg,
                plane_yaw_min_deg=args.plane_yaw_min_deg,
                plane_yaw_max_deg=args.plane_yaw_max_deg,
                plane_center_xy_range_mm=args.plane_center_xy_range_mm,
                plane_center_z_range_mm=plane_center_z_range_mm,
            )
            results.append(result)
        except Exception as exc:
            failures += 1
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
                    f"[single-plane] progress {system_idx + 1}/{args.systems} | "
                    f"ok={len(results)} | fail={failures} | "
                    f"conv={converged}/{len(results)} | "
                    f"paper_success={succeeded}/{len(results)} | "
                    f"median_t={median_translation_error:.6g} mm | "
                    f"elapsed={elapsed:.1f}s"
                )
            else:
                print(
                    f"[single-plane] progress {system_idx + 1}/{args.systems} | "
                    f"ok=0 | fail={failures} | elapsed={elapsed:.1f}s"
                )

    if args.verbose:
        elapsed = time.perf_counter() - start_time
        print(
            f"[single-plane] finished | ok={len(results)} | "
            f"failed={failures} | elapsed={elapsed:.1f}s"
        )

    print_calibration_summary(results)
    if not results:
        return

    save_results_csv(results, args.csv)
    print(f"saved: {args.csv}")

    if args.no_plots:
        return

    plot_dir = (
        args.plot_dir
        if args.plot_dir is not None
        else args.csv.parent / f"{args.csv.stem}_plots"
    )
    for path in save_calibration_plots(results, plot_dir):
        print(f"saved plot: {path}")


if __name__ == "__main__":
    main()