from __future__ import annotations

import argparse
import csv
import time
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
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
    noise_std_mm: float
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
    plane_size_mm: float = 700.0,
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

    return save_plane_scene_plot(
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

        if nonlinear_refine:
            nonlinear_result = refine_handeye_nonlinear(
                scans_by_plane,
                T_linear,
                plane_mode="fixed",
                planes=planes,
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
        and iterations <= 2000
        and np.all(np.abs(translation_error) < 0.01)
    )

    return ThreePlaneTrialResult(
        system_idx=system_idx,
        noise_std_mm=float(noise_std),
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



def _finite_values(
    results: list[ThreePlaneTrialResult],
    attribute: str,
) -> np.ndarray:
    """Return finite values of one scalar result attribute."""
    values = np.asarray(
        [float(getattr(result, attribute)) for result in results],
        dtype=float,
    )
    return values[np.isfinite(values)]


def summarize_noise_results(
    results: list[ThreePlaneTrialResult],
    noise_levels: np.ndarray,
    requested_systems: int,
) -> list[dict[str, float | int]]:
    """Aggregate calibration performance for every sensor-noise level."""
    summary_rows: list[dict[str, float | int]] = []

    for noise_std in noise_levels:
        group = [
            result
            for result in results
            if np.isclose(result.noise_std_mm, float(noise_std))
        ]

        translation = _finite_values(group, "trans_err_norm_mm")
        rotation = _finite_values(group, "rot_err_angle_deg")
        iterations = _finite_values(group, "iterations")
        cond = _finite_values(group, "cond_last")

        converged_count = int(sum(result.converged for result in group))
        paper_success_count = int(sum(result.paper_success for result in group))
        nonlinear_success_count = int(
            sum(result.nonlinear_success for result in group)
        )

        def stat(values: np.ndarray, fn, default=float("nan")) -> float:
            return float(fn(values)) if values.size else float(default)

        summary_rows.append(
            {
                "noise_std_mm": float(noise_std),
                "requested_systems": int(requested_systems),
                "completed_systems": int(len(group)),
                "failed_systems": int(requested_systems - len(group)),
                "converged_count": converged_count,
                "convergence_rate": (
                    converged_count / len(group) if group else 0.0
                ),
                "paper_success_count": paper_success_count,
                "paper_success_rate": (
                    paper_success_count / len(group) if group else 0.0
                ),
                "translation_mean_mm": stat(translation, np.mean),
                "translation_std_mm": stat(translation, np.std),
                "translation_median_mm": stat(translation, np.median),
                "translation_p25_mm": stat(
                    translation, lambda x: np.percentile(x, 25.0)
                ),
                "translation_p75_mm": stat(
                    translation, lambda x: np.percentile(x, 75.0)
                ),
                "translation_p95_mm": stat(
                    translation, lambda x: np.percentile(x, 95.0)
                ),
                "rotation_mean_deg": stat(rotation, np.mean),
                "rotation_std_deg": stat(rotation, np.std),
                "rotation_median_deg": stat(rotation, np.median),
                "rotation_p25_deg": stat(
                    rotation, lambda x: np.percentile(x, 25.0)
                ),
                "rotation_p75_deg": stat(
                    rotation, lambda x: np.percentile(x, 75.0)
                ),
                "rotation_p95_deg": stat(
                    rotation, lambda x: np.percentile(x, 95.0)
                ),
                "iterations_median": stat(iterations, np.median),
                "condition_number_median": stat(cond, np.median),
                "nonlinear_success_count": nonlinear_success_count,
                "nonlinear_success_rate": (
                    nonlinear_success_count / len(group) if group else 0.0
                ),
            }
        )

    return summary_rows


def save_noise_summary_csv(
    summary_rows: list[dict[str, float | int]],
    out_path: Path,
) -> Path:
    """Save one aggregate row per sensor-noise level."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not summary_rows:
        raise ValueError("summary_rows must not be empty")

    with out_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    return out_path


def save_noise_performance_plots(
    summary_rows: list[dict[str, float | int]],
    out_dir: Path,
) -> list[Path]:
    """Save translation, rotation, and success-rate noise sweep plots."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    noise = np.asarray(
        [row["noise_std_mm"] for row in summary_rows],
        dtype=float,
    )

    paths: list[Path] = []

    translation_median = np.asarray(
        [row["translation_median_mm"] for row in summary_rows],
        dtype=float,
    )
    translation_p25 = np.asarray(
        [row["translation_p25_mm"] for row in summary_rows],
        dtype=float,
    )
    translation_p75 = np.asarray(
        [row["translation_p75_mm"] for row in summary_rows],
        dtype=float,
    )
    translation_p95 = np.asarray(
        [row["translation_p95_mm"] for row in summary_rows],
        dtype=float,
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(noise, translation_median, marker="o", label="Median")
    ax.fill_between(
        noise,
        translation_p25,
        translation_p75,
        alpha=0.25,
        label="25–75 percentile",
    )
    ax.plot(
        noise,
        translation_p95,
        linestyle="--",
        marker=".",
        label="95th percentile",
    )
    ax.set_xlabel("Sensor Gaussian noise standard deviation [mm]")
    ax.set_ylabel("Translation error norm [mm]")
    ax.set_title("Translation performance vs sensor noise")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = out_dir / "noise_sweep_translation_error.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    paths.append(path)

    rotation_median = np.asarray(
        [row["rotation_median_deg"] for row in summary_rows],
        dtype=float,
    )
    rotation_p25 = np.asarray(
        [row["rotation_p25_deg"] for row in summary_rows],
        dtype=float,
    )
    rotation_p75 = np.asarray(
        [row["rotation_p75_deg"] for row in summary_rows],
        dtype=float,
    )
    rotation_p95 = np.asarray(
        [row["rotation_p95_deg"] for row in summary_rows],
        dtype=float,
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(noise, rotation_median, marker="o", label="Median")
    ax.fill_between(
        noise,
        rotation_p25,
        rotation_p75,
        alpha=0.25,
        label="25–75 percentile",
    )
    ax.plot(
        noise,
        rotation_p95,
        linestyle="--",
        marker=".",
        label="95th percentile",
    )
    ax.set_xlabel("Sensor Gaussian noise standard deviation [mm]")
    ax.set_ylabel("Rotation error angle [deg]")
    ax.set_title("Rotation performance vs sensor noise")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = out_dir / "noise_sweep_rotation_error.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    paths.append(path)

    convergence_rate = np.asarray(
        [row["convergence_rate"] for row in summary_rows],
        dtype=float,
    )
    paper_success_rate = np.asarray(
        [row["paper_success_rate"] for row in summary_rows],
        dtype=float,
    )

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(
        noise,
        100.0 * convergence_rate,
        marker="o",
        label="Convergence rate",
    )
    ax.plot(
        noise,
        100.0 * paper_success_rate,
        marker="s",
        label="Paper success rate",
    )
    ax.set_xlabel("Sensor Gaussian noise standard deviation [mm]")
    ax.set_ylabel("Rate [%]")
    ax.set_ylim(-2.0, 102.0)
    ax.set_title("Calibration success vs sensor noise")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    path = out_dir / "noise_sweep_success_rate.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    paths.append(path)

    return paths

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
    parser.add_argument(
        "--noise-levels",
        type=float,
        nargs="+",
        default=None,
        help=(
            "explicit Gaussian sensor-noise standard deviations in mm; "
            "overrides --noise-min/--noise-max/--noise-step"
        ),
    )
    parser.add_argument("--noise-min", type=float, default=0.0)
    parser.add_argument("--noise-max", type=float, default=0.5)
    parser.add_argument("--noise-step", type=float, default=0.05)
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
    parser.add_argument(
        "--nonlinear-loss",
        choices=["linear", "soft_l1", "huber", "cauchy", "arctan"],
        default="linear",
    )
    parser.add_argument("--nonlinear-f-scale-mm", type=float, default=1.0)
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("three_plane_noise_sweep_trials.csv"),
        help="raw per-trial CSV containing every noise level",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=Path("three_plane_noise_sweep_summary.csv"),
        help="aggregate CSV containing one row per noise level",
    )
    parser.add_argument("--debug-scene-plot", type=Path, default=None)
    parser.add_argument("--debug-scene-seed", type=int, default=None)
    parser.add_argument("--debug-scene-plane-size", type=float, default=700.0)
    parser.add_argument("--plot-dir", type=Path, default=None)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--log-every", type=int, default=5)
    args = parser.parse_args()

    if args.noise_levels is not None:
        noise_levels = np.asarray(args.noise_levels, dtype=float)
    else:
        if args.noise_step <= 0.0:
            raise ValueError("--noise-step must be positive")
        if args.noise_max < args.noise_min:
            raise ValueError("--noise-max must be >= --noise-min")
        count = int(
            np.floor(
                (args.noise_max - args.noise_min) / args.noise_step + 1e-12
            )
        ) + 1
        noise_levels = args.noise_min + args.noise_step * np.arange(count)
        if noise_levels[-1] < args.noise_max - 1e-12:
            noise_levels = np.append(noise_levels, args.noise_max)

    if np.any(~np.isfinite(noise_levels)) or np.any(noise_levels < 0.0):
        raise ValueError("all noise levels must be finite and non-negative")

    noise_levels = np.unique(np.round(noise_levels, decimals=12))

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
            noise_std=float(noise_levels[0]),
            out_path=args.debug_scene_plot,
            plane_size_mm=args.debug_scene_plane_size,
        )
        print(f"saved debug scene plot: {debug_path}")

    results: list[ThreePlaneTrialResult] = []
    failures = 0
    first_failure: str | None = None
    start_time = time.perf_counter()
    log_every = max(1, args.log_every)
    total_trials = int(len(noise_levels) * args.systems)
    completed_trials = 0

    if args.verbose:
        print(
            f"[three-plane noise sweep] start | systems/noise={args.systems} | "
            f"noise_levels={noise_levels.tolist()} mm | mode={args.mode} | "
            f"poses/plane={args.poses_per_plane} | init_mode={args.init_mode} | "
            f"offset_mode={args.plane_offset_mode} | "
            f"nonlinear={args.nonlinear_refine}"
        )

    for noise_idx, noise_std in enumerate(noise_levels):
        level_results_before = len(results)
        level_failures_before = failures

        for system_idx in range(args.systems):
            # The same system_idx receives the same GT, scene, and initial-guess
            # random streams at every noise level. This creates a paired noise
            # sweep rather than comparing unrelated random systems.
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
                    noise_std=float(noise_std),
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
                    nonlinear_loss=args.nonlinear_loss,
                    nonlinear_f_scale_mm=args.nonlinear_f_scale_mm,
                    gt_rng=gt_rng,
                    init_rng=init_rng,
                )
                results.append(result)
            except Exception as exc:
                failures += 1
                if first_failure is None:
                    first_failure = f"{type(exc).__name__}: {exc}"
                if args.verbose:
                    print(
                        f"[noise={noise_std:.6g} | trial={system_idx:04d}] "
                        f"failed | {type(exc).__name__}: {exc}"
                    )

            completed_trials += 1

            if args.verbose and (
                completed_trials % log_every == 0
                or completed_trials == total_trials
            ):
                elapsed = time.perf_counter() - start_time
                current_level_results = [
                    result
                    for result in results[level_results_before:]
                    if np.isclose(result.noise_std_mm, float(noise_std))
                ]
                if current_level_results:
                    median_translation_error = float(
                        np.median(
                            [
                                result.trans_err_norm_mm
                                for result in current_level_results
                            ]
                        )
                    )
                    converged = sum(
                        result.converged for result in current_level_results
                    )
                    print(
                        f"[noise sweep] {completed_trials}/{total_trials} | "
                        f"noise={noise_std:.6g} mm | "
                        f"ok={len(current_level_results)} | "
                        f"fail={failures - level_failures_before} | "
                        f"conv={converged}/{len(current_level_results)} | "
                        f"median_t={median_translation_error:.6g} mm | "
                        f"elapsed={elapsed:.1f}s"
                    )

        if args.verbose:
            level_results = results[level_results_before:]
            level_failures = failures - level_failures_before
            print(
                f"[noise={noise_std:.6g} mm] finished | "
                f"ok={len(level_results)} | fail={level_failures}"
            )

    if args.verbose:
        elapsed = time.perf_counter() - start_time
        print(
            f"[three-plane noise sweep] finished | ok={len(results)} | "
            f"failed={failures} | elapsed={elapsed:.1f}s"
        )

    print_calibration_summary(results)
    if not results:
        if first_failure is not None:
            print(f"first failure: {first_failure}")
        raise SystemExit(2)

    save_results_csv(results, args.csv)
    print(f"saved raw trials: {args.csv}")

    summary_rows = summarize_noise_results(
        results=results,
        noise_levels=noise_levels,
        requested_systems=args.systems,
    )
    summary_path = save_noise_summary_csv(summary_rows, args.summary_csv)
    print(f"saved noise summary: {summary_path}")

    if args.no_plots:
        return

    plot_dir = (
        args.plot_dir
        if args.plot_dir is not None
        else args.summary_csv.parent / f"{args.summary_csv.stem}_plots"
    )

    for path in save_noise_performance_plots(summary_rows, plot_dir):
        print(f"saved noise plot: {path}")


if __name__ == "__main__":
    main()