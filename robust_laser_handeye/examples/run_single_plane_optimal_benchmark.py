from __future__ import annotations

import argparse
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

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
from scipy.spatial.transform import Rotation
from laser_handeye.se3 import (
    rot_error_deg,
    rotation_vector_error_deg,
    transform_points,
)
from laser_handeye.patterns import scan_parameter_grid
from laser_handeye.simulation import (
    generate_circular_pattern_scans,
    generate_circular_reference_scans,
    sample_random_handeye,
)


@dataclass
class SinglePlaneTrialResult:
    system_idx: int
    mode: str
    init_mode: str
    plane_offset_mode: str
    pose_geometry: str
    n_reference_scans: int
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
    gauge_parallel_error_mm: float
    gauge_perpendicular_error_mm: float
    gauge_axis_angle_deg: float
    gauge_parallel_fraction: float
    paper_success: bool
    linear_multistart_used: bool
    linear_start_count: int
    final_linear_plane_rms_mm: float
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

def sample_random_plane_pose(
    rng: np.random.Generator,
    tilt_min_deg: float = 1.0,
    tilt_max_deg: float = 5.0,
    yaw_min_deg: float = -5.0,
    yaw_max_deg: float = 5.0,
    center_x_range_mm: tuple[float, float] = (-100.0, 100.0),
    center_y_range_mm: tuple[float, float] = (-100.0, 100.0),
    center_z_range_mm: tuple[float, float] = (400.0, 550.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Sample one random calibration-plane pose.

    Returns:
        plane_R:
            Plane-frame orientation in the robot base frame.
            ``plane_R[:, 2]`` is the plane normal.
        plane_t:
            Plane-frame origin in the robot base frame [mm].
        plane_n, plane_l:
            Equivalent plane equation ``plane_n.T @ p = plane_l``.
    """
    if tilt_min_deg < 0.0 or tilt_max_deg < tilt_min_deg:
        raise ValueError("invalid plane tilt range")

    def sample_signed_tilt() -> float:
        magnitude = float(rng.uniform(tilt_min_deg, tilt_max_deg))
        return magnitude if rng.random() >= 0.5 else -magnitude

    def sample_nonzero_yaw() -> float:
        for _ in range(10_000):
            value = float(rng.uniform(yaw_min_deg, yaw_max_deg))
            if abs(value) >= tilt_min_deg:
                return value
        raise ValueError(
            "yaw range cannot satisfy the requested minimum absolute angle"
        )

    plane_angles_deg = np.array(
        [
            sample_signed_tilt(),
            sample_signed_tilt(),
            sample_nonzero_yaw(),
        ],
        dtype=float,
    )
    plane_R = Rotation.from_euler(
        "xyz",
        plane_angles_deg,
        degrees=True,
    ).as_matrix()

    plane_t = np.array(
        [
            rng.uniform(*center_x_range_mm),
            rng.uniform(*center_y_range_mm),
            rng.uniform(*center_z_range_mm),
        ],
        dtype=float,
    )

    plane_n = np.asarray(plane_R[:, 2], dtype=float)
    plane_n /= np.linalg.norm(plane_n)
    plane_l = float(plane_n @ plane_t)

    # Preserve the same geometric plane while keeping l non-negative.
    if plane_l < 0.0:
        plane_n = -plane_n
        plane_l = -plane_l
        plane_R = plane_R.copy()
        plane_R[:, 0] = -plane_R[:, 0]
        plane_R[:, 2] = -plane_R[:, 2]

    return plane_R, plane_t, plane_n, plane_l


def make_scan_params(
    heights_mm: tuple[float, ...] = (60.0, 90.0, 120.0),
    theta_deg: tuple[float, ...] = (30.0,),
    beta_deg: tuple[float, ...] = (60.0, 90.0, 120.0),
) -> list[dict]:
    """Create the (d, theta, beta) scan-parameter grid.

    Defaults reproduce the paper-style 3 x 1 x 3 = 9 combinations.
    """
    if not heights_mm or not theta_deg or not beta_deg:
        raise ValueError("heights_mm, theta_deg and beta_deg must be non-empty")
    if any(float(d) <= 0.0 for d in heights_mm):
        raise ValueError("all sensor heights must be positive")

    return scan_parameter_grid(
        heights_mm=tuple(float(v) for v in heights_mm),
        projection_deg=tuple(float(v) for v in theta_deg),
        tilt_deg=tuple(float(v) for v in beta_deg),
    )


def generate_optimal_single_plane_dataset(
    T_ef_s_true: np.ndarray,
    plane_R: np.ndarray,
    plane_t: np.ndarray,
    rng: np.random.Generator,
    x_values: np.ndarray,
    radius_mm: float,
    noise_std: float,
    check_reachability: bool,
    scan_params: list[dict],
    pose_geometry: str = "paper_incidence",
    reference_scan_params: list[dict] | None = None,
    reference_scan_count: int = 0,
) -> dict[int, list[LaserScan]]:
    """Generate the 81-scan fixed-theta grid plus optional excitation.

    Nine target lines are arranged around a circular pattern at 40-degree
    intervals. Each target line is scanned using all nine optimal parameter
    combinations, yielding nominally 9 x 9 = 81 scans.

    A faithful fixed-theta incidence ring cannot distinguish one hand-eye
    translation component from the unknown plane offset. ``reference_scan_*``
    adds an explicitly counted engineering observability extension. It is not
    part of the paper's reduced nine-combination parameter grid.
    """
    scans = generate_circular_pattern_scans(
        plane_R=plane_R,
        plane_t=plane_t,
        T_ef_s_true=T_ef_s_true,
        radius_mm=radius_mm,
        x_values=x_values,
        noise_std=noise_std,
        rng=rng,
        scan_params=scan_params,
        check_reachability=check_reachability,
        plane_id=0,
        pose_geometry=pose_geometry,
    )

    if reference_scan_count > 0:
        if not reference_scan_params:
            raise ValueError(
                "reference_scan_params are required when reference_scan_count > 0"
            )
        scans.extend(
            generate_circular_reference_scans(
                plane_R=plane_R,
                plane_t=plane_t,
                T_ef_s_true=T_ef_s_true,
                radius_mm=radius_mm,
                x_values=x_values,
                scan_params=reference_scan_params,
                n_scans=reference_scan_count,
                noise_std=noise_std,
                rng=rng,
                check_reachability=check_reachability,
                plane_id=0,
                pose_geometry=pose_geometry,
            )
        )

    for scan_id, scan in enumerate(scans):
        scan.scan_id = scan_id

    if not scans:
        raise RuntimeError("no valid optimal-parameter scans were generated")

    return {0: scans}


def _rotation_angle_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    """Geodesic angle between two rotation matrices [deg]."""
    R_rel = np.asarray(R_a, dtype=float).reshape(3, 3).T @ np.asarray(
        R_b, dtype=float
    ).reshape(3, 3)
    return float(np.degrees(Rotation.from_matrix(R_rel).magnitude()))


def _linear_retry_initial_guesses(
    T_init: np.ndarray,
    angle_deg: float,
) -> list[np.ndarray]:
    """Return six deterministic Euler-neighbour starts around ``T_init``.

    These starts are used only when the first linear alternating solve ends in
    a high-residual PCA basin. They do not run the optional nonlinear SE(3)
    refinement and do not use the simulated ground truth.
    """
    angle_deg = float(angle_deg)
    if angle_deg <= 0.0:
        raise ValueError("linear multistart angle must be positive")
    initial_euler = Rotation.from_matrix(
        np.asarray(T_init, dtype=float)[:3, :3]
    ).as_euler("xyz", degrees=True)
    candidates: list[np.ndarray] = []
    for axis in range(3):
        for sign in (-1.0, 1.0):
            euler = initial_euler.copy()
            euler[axis] += sign * angle_deg
            candidate = np.asarray(T_init, dtype=float).copy()
            candidate[:3, :3] = Rotation.from_euler(
                "xyz", euler, degrees=True
            ).as_matrix()
            candidates.append(candidate)
    return candidates


def _reconstruct_points_base(
    scans: list[LaserScan],
    T_ef_s: np.ndarray,
) -> np.ndarray:
    """Reconstruct all valid scan points in the robot base frame."""
    point_sets: list[np.ndarray] = []
    for scan in scans:
        points_s = np.asarray(scan.valid_points_s, dtype=float)
        if points_s.size == 0:
            continue
        points_ef = transform_points(T_ef_s, points_s)
        points_base = transform_points(scan.T_base_ef, points_ef)
        point_sets.append(points_base)

    if not point_sets:
        return np.empty((0, 3), dtype=float)
    return np.vstack(point_sets)


def _final_self_fit_plane_rms_mm(
    scans_by_plane: dict[int, list[LaserScan]],
    T_ef_s: np.ndarray,
) -> float:
    """Evaluate plane RMS at the returned transform itself."""
    rms_values: list[float] = []
    for scans in scans_by_plane.values():
        points_base = _reconstruct_points_base(scans, T_ef_s)
        if len(points_base) < 3:
            return float("inf")
        _normal, _offset, _centroid, plane_rms = fit_plane_pca(points_base)
        rms_values.append(float(plane_rms))
    return float(np.mean(rms_values)) if rms_values else float("inf")


def _plane_residual_stats(
    points_base: np.ndarray,
    plane_n: np.ndarray,
    plane_l: float,
) -> dict[str, float]:
    """Point-to-plane residual statistics in millimetres."""
    points_base = np.asarray(points_base, dtype=float).reshape(-1, 3)
    plane_n = np.asarray(plane_n, dtype=float).reshape(3)
    plane_n /= np.linalg.norm(plane_n)
    if len(points_base) == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "rms": float("nan"),
            "max_abs": float("nan"),
        }

    residuals = points_base @ plane_n - float(plane_l)
    return {
        "count": int(len(residuals)),
        "mean": float(np.mean(residuals)),
        "std": float(np.std(residuals)),
        "rms": float(np.sqrt(np.mean(residuals**2))),
        "max_abs": float(np.max(np.abs(residuals))),
    }


def translation_error_sensor_z_components(
    T_est: np.ndarray,
    T_true: np.ndarray,
) -> dict[str, float | np.ndarray]:
    """Decompose translation error along the true sensor optical-Z axis.

    Hand-eye translation is expressed in the end-effector frame.  The sensor
    ``+Z`` direction in that frame is therefore ``R_ef_s_true @ e_z``, i.e.
    the third column of the true hand-eye rotation.  The fixed-theta gauge is
    an unoriented line, so ``axis_angle_deg`` uses the smaller angle to either
    ``+Z`` or ``-Z``.
    """
    T_est = np.asarray(T_est, dtype=float).reshape(4, 4)
    T_true = np.asarray(T_true, dtype=float).reshape(4, 4)
    error = T_est[:3, 3] - T_true[:3, 3]
    sensor_z_ef = T_true[:3, 2].copy()
    sensor_z_ef /= np.linalg.norm(sensor_z_ef)

    parallel_signed = float(error @ sensor_z_ef)
    parallel_vector = parallel_signed * sensor_z_ef
    perpendicular_vector = error - parallel_vector
    perpendicular_norm = float(np.linalg.norm(perpendicular_vector))
    error_norm = float(np.linalg.norm(error))
    if error_norm <= np.finfo(float).eps:
        axis_angle_deg = float("nan")
        parallel_fraction = float("nan")
    else:
        parallel_fraction = float(abs(parallel_signed) / error_norm)
        axis_angle_deg = float(
            np.degrees(
                np.arctan2(perpendicular_norm, abs(parallel_signed))
            )
        )

    return {
        "error_ef": error,
        "sensor_z_ef": sensor_z_ef,
        "parallel_signed_mm": parallel_signed,
        "parallel_abs_mm": abs(parallel_signed),
        "perpendicular_mm": perpendicular_norm,
        "axis_angle_deg": axis_angle_deg,
        "parallel_fraction": parallel_fraction,
    }


def translation_gauge_sweep(
    *,
    scans: list[LaserScan],
    T_true: np.ndarray,
    plane_n: np.ndarray,
    plane_l: float,
    shifts_mm: np.ndarray,
) -> dict[str, np.ndarray]:
    """Evaluate the fixed-normal plane fit along sensor-Z and sensor-X shifts.

    For every candidate hand-eye translation, only the plane offset is fitted;
    the ground-truth plane normal is held fixed.  A fixed-theta data set must
    have constant RMS along ``R_true @ e_z`` while its fitted offset changes.
    ``R_true @ e_x`` is included as a transverse control direction.
    """
    if not scans:
        raise ValueError("at least one scan is required")
    T_true = np.asarray(T_true, dtype=float).reshape(4, 4)
    plane_n = np.asarray(plane_n, dtype=float).reshape(3)
    plane_n /= np.linalg.norm(plane_n)
    shifts = np.asarray(shifts_mm, dtype=float).reshape(-1)
    if len(shifts) < 2 or np.any(~np.isfinite(shifts)):
        raise ValueError("shifts_mm must contain at least two finite values")

    directions = {
        "sensor_z": T_true[:3, 2],
        "sensor_x": T_true[:3, 0],
    }
    output: dict[str, np.ndarray] = {"shifts_mm": shifts.copy()}
    for name, direction in directions.items():
        direction = np.asarray(direction, dtype=float).reshape(3)
        direction /= np.linalg.norm(direction)
        rms_values: list[float] = []
        offset_values: list[float] = []
        for shift in shifts:
            T_candidate = T_true.copy()
            T_candidate[:3, 3] += float(shift) * direction
            points_base = _reconstruct_points_base(scans, T_candidate)
            if len(points_base) == 0:
                raise ValueError("no valid reconstructed points")
            normal_coordinates = points_base @ plane_n
            fitted_offset = float(np.mean(normal_coordinates))
            residuals = normal_coordinates - fitted_offset
            rms_values.append(float(np.sqrt(np.mean(residuals**2))))
            offset_values.append(fitted_offset - float(plane_l))
        output[f"{name}_rms_mm"] = np.asarray(rms_values, dtype=float)
        output[f"{name}_offset_delta_mm"] = np.asarray(
            offset_values,
            dtype=float,
        )
    return output


def save_translation_gauge_sweep_plot(
    *,
    sweep: dict[str, np.ndarray],
    theta_deg: float,
    out_path: Path,
) -> Path:
    """Plot the residual-flat gauge and its compensating plane offset."""
    shifts = np.asarray(sweep["shifts_mm"], dtype=float)
    z_rms = np.asarray(sweep["sensor_z_rms_mm"], dtype=float)
    x_rms = np.asarray(sweep["sensor_x_rms_mm"], dtype=float)
    z_offset = np.asarray(
        sweep["sensor_z_offset_delta_mm"],
        dtype=float,
    )
    x_offset = np.asarray(
        sweep["sensor_x_offset_delta_mm"],
        dtype=float,
    )
    theoretical_offset = -shifts * np.cos(np.radians(float(theta_deg)))

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    rms_floor = 1e-12
    axes[0].semilogy(
        shifts,
        np.maximum(z_rms, rms_floor),
        linewidth=2.2,
        label=r"shift along $R_{ES}e_z$ (gauge)",
    )
    axes[0].semilogy(
        shifts,
        np.maximum(x_rms, rms_floor),
        linewidth=2.0,
        label=r"shift along $R_{ES}e_x$ (control)",
    )
    axes[0].set_xlabel("forced hand-eye translation shift [mm]")
    axes[0].set_ylabel("RMS after fitting plane offset [mm]")
    axes[0].set_title("Residual response to translation direction")
    axes[0].grid(True, which="both", alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        shifts,
        z_offset,
        linewidth=2.2,
        label=r"fitted offset: $R_{ES}e_z$ shift",
    )
    axes[1].plot(
        shifts,
        theoretical_offset,
        "--",
        linewidth=1.8,
        label=rf"theory: $-\lambda\cos({float(theta_deg):g}^\circ)$",
    )
    axes[1].plot(
        shifts,
        x_offset,
        linewidth=1.5,
        alpha=0.8,
        label=r"fitted offset: $R_{ES}e_x$ shift",
    )
    axes[1].set_xlabel("forced hand-eye translation shift [mm]")
    axes[1].set_ylabel(r"fitted plane offset change $\hat d-d$ [mm]")
    axes[1].set_title("Plane offset that absorbs the shift")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    z_rms_span = float(np.max(z_rms) - np.min(z_rms))
    offset_mismatch = float(np.max(np.abs(z_offset - theoretical_offset)))
    fig.suptitle(
        "Fixed-theta translation/plane-offset gauge\n"
        rf"$\theta={float(theta_deg):g}^\circ$: sensor-Z RMS span="
        f"{z_rms_span:.3e} mm, offset-law max error={offset_mismatch:.3e} mm",
        fontsize=13,
    )
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_translation_error_direction_plot(
    results: list[SinglePlaneTrialResult],
    out_path: Path,
) -> Path | None:
    """Plot actual solver error parallel/perpendicular to sensor optical-Z."""
    rows = [
        result
        for result in results
        if np.isfinite(result.gauge_parallel_error_mm)
        and np.isfinite(result.gauge_perpendicular_error_mm)
    ]
    if not rows:
        return None
    parallel = np.asarray(
        [abs(result.gauge_parallel_error_mm) for result in rows],
        dtype=float,
    )
    perpendicular = np.asarray(
        [result.gauge_perpendicular_error_mm for result in rows],
        dtype=float,
    )
    angle = np.asarray(
        [result.gauge_axis_angle_deg for result in rows],
        dtype=float,
    )
    norm = np.hypot(parallel, perpendicular)
    order = np.argsort(norm)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    axes[0].scatter(parallel, perpendicular, s=34, alpha=0.8)
    upper = max(1.0, float(np.max(np.r_[parallel, perpendicular])) * 1.08)
    axes[0].plot([0.0, upper], [0.0, upper], "--", color="0.45", linewidth=1.0)
    axes[0].set_xlim(0.0, upper)
    axes[0].set_ylim(0.0, upper)
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_xlabel(r"$|\Delta t^T R_{ES}e_z|$ [mm]")
    axes[0].set_ylabel(r"$\|\Delta t_\perp\|$ [mm]")
    axes[0].set_title("Actual translation-error direction")
    axes[0].grid(True, alpha=0.3)

    axes[1].bar(
        np.arange(len(rows)),
        parallel[order],
        label="parallel to sensor-Z gauge",
    )
    axes[1].bar(
        np.arange(len(rows)),
        perpendicular[order],
        bottom=parallel[order],
        label="perpendicular component",
    )
    axes[1].set_xlabel("trial, sorted by translation error")
    axes[1].set_ylabel("component magnitude [mm]")
    axes[1].set_title("Translation-error decomposition")
    axes[1].grid(True, axis="y", alpha=0.3)
    axes[1].legend()

    finite_angle = angle[np.isfinite(angle)]
    angle_summary = (
        f"median axis angle={np.median(finite_angle):.3e} deg, "
        f"max={np.max(finite_angle):.3e} deg"
        if len(finite_angle)
        else "axis angle undefined for zero-error trials"
    )
    fig.suptitle(
        "Estimated translation error relative to true sensor optical-Z\n"
        + angle_summary,
        fontsize=13,
    )
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _plane_difference(
    estimated: tuple[np.ndarray, float],
    reference: tuple[np.ndarray, float],
) -> tuple[float, float]:
    """Return unsigned normal-angle error [deg] and offset error [mm]."""
    n_est, l_est = estimated
    n_ref, l_ref = reference
    n_est = np.asarray(n_est, dtype=float).reshape(3)
    n_ref = np.asarray(n_ref, dtype=float).reshape(3)
    n_est /= np.linalg.norm(n_est)
    n_ref /= np.linalg.norm(n_ref)

    # Plane equations (n,l) and (-n,-l) represent the same plane.
    if float(n_est @ n_ref) < 0.0:
        n_est = -n_est
        l_est = -float(l_est)

    dot = float(np.clip(n_est @ n_ref, -1.0, 1.0))
    normal_error_deg = float(np.degrees(np.arccos(dot)))
    offset_error_mm = float(l_est - l_ref)
    return normal_error_deg, offset_error_mm


def _align_plane_representation(
    plane: tuple[np.ndarray, float],
    reference_normal: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Flip (n, l) so that n points to the same hemisphere as reference_normal."""
    n, l = plane
    n = np.asarray(n, dtype=float).reshape(3)
    n /= np.linalg.norm(n)
    reference_normal = np.asarray(reference_normal, dtype=float).reshape(3)
    reference_normal /= np.linalg.norm(reference_normal)
    l = float(l)
    if float(n @ reference_normal) < 0.0:
        n = -n
        l = -l
    return n, l


def translation_offset_observability(
    scans: list[LaserScan],
    plane_n: np.ndarray,
) -> dict[str, object]:
    """Diagnose observability of hand-eye translation jointly with plane offset.

    For each flange pose, q_i = R_base_ef_i.T @ n.  The joint linear block is
    [q_i.T, -1].  Full observability requires rank 4.
    """
    n = np.asarray(plane_n, dtype=float).reshape(3)
    n /= np.linalg.norm(n)

    rows: list[np.ndarray] = []
    for scan in scans:
        R_base_ef = np.asarray(scan.T_base_ef[:3, :3], dtype=float)
        q_i = R_base_ef.T @ n
        rows.append(np.r_[q_i, -1.0])

    A_tl = np.asarray(rows, dtype=float)
    if A_tl.size == 0:
        return {
            "rank": 0,
            "singular_values": np.empty(0, dtype=float),
            "condition": float("inf"),
            "min_singular_value": 0.0,
        }

    singular_values = np.linalg.svd(A_tl, compute_uv=False)
    rank = int(np.linalg.matrix_rank(A_tl, tol=1e-10))
    min_sv = float(singular_values[-1])
    condition = (
        float(singular_values[0] / min_sv)
        if min_sv > np.finfo(float).eps
        else float("inf")
    )
    return {
        "rank": rank,
        "singular_values": singular_values,
        "condition": condition,
        "min_singular_value": min_sv,
    }



def build_paper_linear_system(
    scans: list[LaserScan],
    plane_n: np.ndarray,
    plane_l: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the paper-style linear system A w = b for one fixed plane.

    State ordering:
        w = [r1_x, r1_y, r1_z, r3_x, r3_y, r3_z, t_x, t_y, t_z]

    For a laser point p_s = [x, 0, z]^T and flange pose
    T_base_ef = [R_i, t_i], define

        q_i = R_i.T @ n.

    The plane constraint n.T p_base = l becomes

        [x q_i.T, z q_i.T, q_i.T] w = l - n.T t_i.

    Notes:
        - n is normalized here. Scaling (n, l) by a common factor only scales
          every equation and does not change the exact solution.
        - Each point in the same scan has the same translation row q_i.T;
          only x and z vary across profile points.
    """
    n = np.asarray(plane_n, dtype=float).reshape(3)
    n_norm = float(np.linalg.norm(n))
    if n_norm <= 0.0:
        raise ValueError("plane_n must be non-zero")
    n = n / n_norm
    l = float(plane_l) / n_norm

    rows: list[np.ndarray] = []
    rhs: list[float] = []

    for scan in scans:
        points_s = np.asarray(scan.valid_points_s, dtype=float).reshape(-1, 3)
        if len(points_s) == 0:
            continue

        T_base_ef = np.asarray(scan.T_base_ef, dtype=float).reshape(4, 4)
        R_base_ef = T_base_ef[:3, :3]
        t_base_ef = T_base_ef[:3, 3]
        q_i = R_base_ef.T @ n
        b_i = l - float(n @ t_base_ef)

        for point_s in points_s:
            x_ij = float(point_s[0])
            z_ij = float(point_s[2])
            rows.append(np.r_[x_ij * q_i, z_ij * q_i, q_i])
            rhs.append(b_i)

    if not rows:
        raise ValueError("cannot build A: no valid laser points")

    return np.asarray(rows, dtype=float), np.asarray(rhs, dtype=float)


def _svd_summary(
    matrix: np.ndarray,
    *,
    relative_rank_tol: float = 1e-10,
) -> dict[str, object]:
    """Return raw and column-normalized SVD diagnostics for a matrix."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("matrix must be a non-empty 2-D array")

    _, raw_singular_values, raw_vt = np.linalg.svd(matrix, full_matrices=False)
    raw_threshold = relative_rank_tol * raw_singular_values[0]
    raw_rank = int(np.sum(raw_singular_values > raw_threshold))
    raw_min_sv = float(raw_singular_values[-1])
    raw_condition = (
        float(raw_singular_values[0] / raw_min_sv)
        if raw_min_sv > np.finfo(float).eps
        else float("inf")
    )

    # The first six columns contain coordinates multiplied by rotation columns,
    # while the final three are translation columns. Their units/scales differ,
    # so use column normalization when comparing geometric conditioning.
    column_norms = np.linalg.norm(matrix, axis=0)
    safe_norms = np.where(column_norms > np.finfo(float).eps, column_norms, 1.0)
    normalized = matrix / safe_norms
    _, singular_values, vt = np.linalg.svd(normalized, full_matrices=False)
    threshold = relative_rank_tol * singular_values[0]
    rank = int(np.sum(singular_values > threshold))
    min_sv = float(singular_values[-1])
    condition = (
        float(singular_values[0] / min_sv)
        if min_sv > np.finfo(float).eps
        else float("inf")
    )

    weakest = np.asarray(vt[-1], dtype=float)
    weakest_scale = float(np.max(np.abs(weakest)))
    if weakest_scale > 0.0:
        weakest = weakest / weakest_scale

    raw_weakest = np.asarray(raw_vt[-1], dtype=float)
    raw_weakest_scale = float(np.max(np.abs(raw_weakest)))
    if raw_weakest_scale > 0.0:
        raw_weakest = raw_weakest / raw_weakest_scale

    return {
        "shape": matrix.shape,
        "rank": rank,
        "raw_rank": raw_rank,
        "singular_values": singular_values,
        "raw_singular_values": raw_singular_values,
        "condition": condition,
        "raw_condition": raw_condition,
        "min_singular_value": min_sv,
        "raw_min_singular_value": raw_min_sv,
        "column_norms": column_norms,
        "weakest_vector": weakest,
        "raw_weakest_vector": raw_weakest,
    }


def paper_linear_system_observability(
    scans: list[LaserScan],
    plane_n: np.ndarray,
    plane_l: float,
    *,
    relative_rank_tol: float = 1e-10,
) -> dict[str, object]:
    """Analyze A=[A_R A_t] used by the fixed-plane linear calibration step.

    The report separates three questions:

    1. Is the complete 9-column system A numerically observable?
    2. Does A_t alone span all three translation directions when rotation is
       assumed known?
    3. After removing residual directions explainable by A_R, how much
       independent translation information remains?

    The third quantity is computed from

        A_t_effective = P_R_perp A_t,
        P_R_perp = I - U_R U_R.T,

    where U_R spans col(A_R). This avoids forming a large dense projector.
    """
    A, b = build_paper_linear_system(scans, plane_n, plane_l)
    A_R = A[:, :6]
    A_t = A[:, 6:9]

    full = _svd_summary(A, relative_rank_tol=relative_rank_tol)
    rotation = _svd_summary(A_R, relative_rank_tol=relative_rank_tol)
    translation = _svd_summary(A_t, relative_rank_tol=relative_rank_tol)

    # Remove all residual directions that the six unconstrained rotation-column
    # variables can explain. Using U directly is equivalent to P_perp @ A_t but
    # avoids allocating an N x N projector.
    U_R, s_R, _ = np.linalg.svd(A_R, full_matrices=False)
    if len(s_R) == 0:
        rank_R = 0
    else:
        rank_R = int(np.sum(s_R > relative_rank_tol * s_R[0]))
    if rank_R > 0:
        A_t_effective = A_t - U_R[:, :rank_R] @ (U_R[:, :rank_R].T @ A_t)
    else:
        A_t_effective = A_t.copy()
    translation_given_rotation = _svd_summary(
        A_t_effective,
        relative_rank_tol=relative_rank_tol,
    )

    weakest = np.asarray(full["weakest_vector"], dtype=float)
    weakest_rotation = weakest[:6]
    weakest_translation = weakest[6:9]
    rotation_norm = float(np.linalg.norm(weakest_rotation))
    translation_norm = float(np.linalg.norm(weakest_translation))
    total_norm = float(np.linalg.norm(weakest))
    translation_ratio = translation_norm / total_norm if total_norm > 0.0 else 0.0

    # Verify the true R,t against the fixed-plane equations externally if needed.
    # Here only the matrix geometry is analyzed; b is returned for optional use.
    return {
        "A": A,
        "b": b,
        "A_R": A_R,
        "A_t": A_t,
        "A_t_effective": A_t_effective,
        "full": full,
        "rotation": rotation,
        "translation": translation,
        "translation_given_rotation": translation_given_rotation,
        "weakest_rotation": weakest_rotation,
        "weakest_translation": weakest_translation,
        "weakest_rotation_norm": rotation_norm,
        "weakest_translation_norm": translation_norm,
        "weakest_translation_ratio": translation_ratio,
    }


def _format_svd_line(name: str, report: dict[str, object]) -> str:
    return (
        f"  {name}: shape={report['shape']}, "
        f"rank_norm={report['rank']}/{report['shape'][1]}, "
        f"min_sv_norm={report['min_singular_value']:.6g}, "
        f"cond_norm={report['condition']:.6g}, "
        f"raw_rank={report['raw_rank']}/{report['shape'][1]}, "
        f"raw_cond={report['raw_condition']:.6g}"
    )


def print_paper_linear_system_diagnostics(
    scans: list[LaserScan],
    plane_n: np.ndarray,
    plane_l: float,
) -> None:
    """Print interpretable diagnostics for the paper-style A=[A_R A_t]."""
    report = paper_linear_system_observability(
        scans=scans,
        plane_n=plane_n,
        plane_l=plane_l,
    )

    full = report["full"]
    rotation = report["rotation"]
    translation = report["translation"]
    effective = report["translation_given_rotation"]

    print("  PAPER LINEAR SYSTEM A=[A_R A_t] (fixed true plane):")
    print(_format_svd_line("A full", full))
    print(_format_svd_line("A_R", rotation))
    print(_format_svd_line("A_t | rotation known", translation))
    print(_format_svd_line("P_R^perp A_t | rotation free", effective))

    print(
        "  A normalized singular values: "
        + np.array2string(
            np.asarray(full["singular_values"], dtype=float), precision=6
        )
    )
    print(
        "  A_t normalized singular values (rotation known): "
        + np.array2string(
            np.asarray(translation["singular_values"], dtype=float), precision=6
        )
    )
    print(
        "  effective translation singular values (rotation columns removed): "
        + np.array2string(
            np.asarray(effective["singular_values"], dtype=float), precision=6
        )
    )

    full_labels = (
        "r1_x", "r1_y", "r1_z", "r3_x", "r3_y", "r3_z",
        "t_x", "t_y", "t_z",
    )
    weakest_terms = sorted(
        zip(full_labels, np.asarray(full["weakest_vector"], dtype=float)),
        key=lambda item: abs(item[1]),
        reverse=True,
    )
    print(
        "  A weakest normalized state direction: "
        + ", ".join(f"{name}={value:+.4f}" for name, value in weakest_terms)
    )
    print(
        "  weakest-mode energy split: "
        f"rotation_norm={report['weakest_rotation_norm']:.6g}, "
        f"translation_norm={report['weakest_translation_norm']:.6g}, "
        f"translation_ratio={report['weakest_translation_ratio']:.6g}"
    )

    translation_weakest = np.asarray(
        translation["weakest_vector"], dtype=float
    )
    effective_weakest = np.asarray(
        effective["weakest_vector"], dtype=float
    )
    print(
        "  weakest translation direction if rotation is known [tx,ty,tz]: "
        + np.array2string(translation_weakest, precision=6)
    )
    print(
        "  weakest translation direction after allowing rotation compensation: "
        + np.array2string(effective_weakest, precision=6)
    )

    if int(translation["rank"]) < 3:
        print(
            "  INTERPRETATION: A_t itself is rank deficient. Even with exact "
            "rotation, at least one translation direction cannot be recovered."
        )
    elif float(translation["condition"]) > 1e6:
        print(
            "  INTERPRETATION: translation is theoretically observable with exact "
            "rotation, but A_t is severely ill-conditioned."
        )
    else:
        print(
            "  INTERPRETATION: with exact rotation, A_t contains three-dimensional "
            "translation information."
        )

    if int(effective["rank"]) < 3:
        print(
            "  INTERPRETATION: once rotation is estimated simultaneously, part of "
            "the translation effect lies inside col(A_R). Rotation and translation "
            "can compensate for each other in the unconstrained 9-variable LS."
        )
    elif float(effective["condition"]) > 1e6:
        print(
            "  INTERPRETATION: independent translation information remains full "
            "rank, but rotation-translation coupling makes it ill-conditioned."
        )
    else:
        print(
            "  INTERPRETATION: translation remains well excited even after removing "
            "directions explainable by the rotation columns."
        )



def _plane_tangent_basis(plane_n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return an orthonormal 2-D basis for the tangent space of a unit normal."""
    n = np.asarray(plane_n, dtype=float).reshape(3)
    n /= np.linalg.norm(n)

    # Choose the Cartesian axis least aligned with n for numerical stability.
    axis = np.eye(3)[int(np.argmin(np.abs(n)))]
    u = axis - float(axis @ n) * n
    u /= np.linalg.norm(u)
    v = np.cross(n, u)
    v /= np.linalg.norm(v)
    return u, v


def _full_problem_residuals(
    scans: list[LaserScan],
    T_ef_s: np.ndarray,
    plane_n: np.ndarray,
    plane_l: float,
) -> np.ndarray:
    """Stack point-to-plane residuals for the full hand-eye/plane problem."""
    n = np.asarray(plane_n, dtype=float).reshape(3)
    n /= np.linalg.norm(n)
    residual_sets: list[np.ndarray] = []

    for scan in scans:
        points_s = np.asarray(scan.valid_points_s, dtype=float)
        if points_s.size == 0:
            continue
        points_ef = transform_points(T_ef_s, points_s)
        points_base = transform_points(scan.T_base_ef, points_ef)
        residual_sets.append(points_base @ n - float(plane_l))

    if not residual_sets:
        return np.empty(0, dtype=float)
    return np.concatenate(residual_sets)


def full_local_observability(
    scans: list[LaserScan],
    T_reference: np.ndarray,
    plane_n: np.ndarray,
    plane_l: float,
    *,
    rotation_step_rad: float = 1e-6,
    translation_step_mm: float = 1e-4,
    normal_step: float = 1e-6,
    offset_step_mm: float = 1e-4,
    relative_rank_tol: float = 1e-8,
) -> dict[str, object]:
    """Numerically evaluate local observability of SE(3) plus one unknown plane.

    Local state ordering:
        [dphi_x, dphi_y, dphi_z, dt_x, dt_y, dt_z, dn_u, dn_v, dl]

    ``dphi`` is a right-multiplicative SO(3) perturbation in radians.
    ``dn_u`` and ``dn_v`` perturb the unit plane normal in its tangent space.
    The raw Jacobian mixes radians, millimetres and dimensionless normal
    coordinates, so rank and conditioning are assessed primarily using a
    column-normalized Jacobian. Raw singular values are also returned.
    """
    T_reference = np.asarray(T_reference, dtype=float).reshape(4, 4)
    n0 = np.asarray(plane_n, dtype=float).reshape(3)
    n0 /= np.linalg.norm(n0)
    l0 = float(plane_l)
    tangent_u, tangent_v = _plane_tangent_basis(n0)

    steps = np.array(
        [
            rotation_step_rad,
            rotation_step_rad,
            rotation_step_rad,
            translation_step_mm,
            translation_step_mm,
            translation_step_mm,
            normal_step,
            normal_step,
            offset_step_mm,
        ],
        dtype=float,
    )
    labels = [
        "dphi_x",
        "dphi_y",
        "dphi_z",
        "dt_x",
        "dt_y",
        "dt_z",
        "dn_u",
        "dn_v",
        "dl",
    ]

    def evaluate(delta: np.ndarray) -> np.ndarray:
        delta = np.asarray(delta, dtype=float).reshape(9)
        T = T_reference.copy()
        T[:3, :3] = T_reference[:3, :3] @ Rotation.from_rotvec(
            delta[:3]
        ).as_matrix()
        T[:3, 3] = T_reference[:3, 3] + delta[3:6]

        n = n0 + delta[6] * tangent_u + delta[7] * tangent_v
        n /= np.linalg.norm(n)
        l = l0 + float(delta[8])
        return _full_problem_residuals(scans, T, n, l)

    residual0 = evaluate(np.zeros(9, dtype=float))
    if residual0.size == 0:
        return {
            "rank": 0,
            "raw_rank": 0,
            "singular_values": np.empty(0, dtype=float),
            "raw_singular_values": np.empty(0, dtype=float),
            "condition": float("inf"),
            "raw_condition": float("inf"),
            "min_singular_value": 0.0,
            "raw_min_singular_value": 0.0,
            "column_norms": np.zeros(9, dtype=float),
            "null_vector": np.full(9, float("nan")),
            "labels": labels,
            "residual_rms_mm": float("nan"),
        }

    J = np.empty((residual0.size, 9), dtype=float)
    for column, step in enumerate(steps):
        plus = np.zeros(9, dtype=float)
        minus = np.zeros(9, dtype=float)
        plus[column] = step
        minus[column] = -step
        J[:, column] = (evaluate(plus) - evaluate(minus)) / (2.0 * step)

    raw_singular_values = np.linalg.svd(J, compute_uv=False)
    raw_threshold = relative_rank_tol * raw_singular_values[0]
    raw_rank = int(np.sum(raw_singular_values > raw_threshold))
    raw_min_sv = float(raw_singular_values[-1])
    raw_condition = (
        float(raw_singular_values[0] / raw_min_sv)
        if raw_min_sv > np.finfo(float).eps
        else float("inf")
    )

    column_norms = np.linalg.norm(J, axis=0)
    safe_norms = np.where(column_norms > np.finfo(float).eps, column_norms, 1.0)
    J_normalized = J / safe_norms
    _, singular_values, Vt = np.linalg.svd(J_normalized, full_matrices=False)
    threshold = relative_rank_tol * singular_values[0]
    rank = int(np.sum(singular_values > threshold))
    min_sv = float(singular_values[-1])
    condition = (
        float(singular_values[0] / min_sv)
        if min_sv > np.finfo(float).eps
        else float("inf")
    )

    # Right-singular vector associated with the weakest local state direction.
    null_vector = np.asarray(Vt[-1], dtype=float)
    scale = float(np.max(np.abs(null_vector)))
    if scale > 0.0:
        null_vector /= scale

    return {
        "rank": rank,
        "raw_rank": raw_rank,
        "singular_values": singular_values,
        "raw_singular_values": raw_singular_values,
        "condition": condition,
        "raw_condition": raw_condition,
        "min_singular_value": min_sv,
        "raw_min_singular_value": raw_min_sv,
        "column_norms": column_norms,
        "null_vector": null_vector,
        "labels": labels,
        "residual_rms_mm": float(np.sqrt(np.mean(residual0**2))),
    }

def print_dataset_diagnostics(
    *,
    system_idx: int,
    scans_by_plane: dict[int, list[LaserScan]],
    T_true: np.ndarray,
    plane_n: np.ndarray,
    plane_l: float,
    expected_scan_count: int,
    noise_std: float,
    scan_params: list[dict],
) -> None:
    """Print checks that separate simulation, geometry and solver failures."""
    scans = scans_by_plane[0]
    valid_counts = np.asarray(
        [len(scan.valid_points_s) for scan in scans], dtype=int
    )

    print(f"\n[debug trial {system_idx:04d}] dataset")
    print(
        f"  scans={len(scans)} / expected={expected_scan_count} | "
        f"points/scan min={valid_counts.min() if len(valid_counts) else 0}, "
        f"median={np.median(valid_counts) if len(valid_counts) else 0:.1f}, "
        f"max={valid_counts.max() if len(valid_counts) else 0}"
    )
    if len(scans) != expected_scan_count:
        print(
            "  WARNING: scan count is not nominal. Reachability filtering or "
            "generation failure changed the intended pose set."
        )

    params = scan_params
    print(f"  scan parameter combinations ({len(params)}):")
    for idx, param in enumerate(params):
        print(f"    {idx:02d}: {param}")

    points_gt = _reconstruct_points_base(scans, T_true)
    gt_stats = _plane_residual_stats(points_gt, plane_n, plane_l)
    print(
        "  GT reconstruction -> true plane residual [mm]: "
        f"mean={gt_stats['mean']:.6g}, std={gt_stats['std']:.6g}, "
        f"rms={gt_stats['rms']:.6g}, max={gt_stats['max_abs']:.6g} "
        f"(configured noise_std={noise_std:.6g})"
    )
    if np.isfinite(gt_stats["rms"]) and gt_stats["rms"] > max(5.0 * noise_std, 1e-6):
        print(
            "  ERROR LIKELY IN SIMULATION/FRAME CONVENTION: even GT hand-eye "
            "does not reconstruct points on the true plane."
        )

    if len(points_gt) >= 3:
        fit_n, fit_l, _, _ = fit_plane_pca(points_gt)
        fit_err_deg, fit_offset_mm = _plane_difference(
            (fit_n, fit_l), (plane_n, plane_l)
        )
        print(
            "  PCA plane from GT reconstruction: "
            f"normal_error={fit_err_deg:.6g} deg, "
            f"offset_error={fit_offset_mm:.6g} mm"
        )

    # Sensor poses reconstructed from flange pose and GT hand-eye.
    T_base_s_all = [scan.T_base_ef @ T_true for scan in scans]
    origins = np.asarray([T[:3, 3] for T in T_base_s_all], dtype=float)
    if len(origins):
        spans = np.ptp(origins, axis=0)
        centered = origins - np.mean(origins, axis=0, keepdims=True)
        singular_values = np.linalg.svd(centered, compute_uv=False)
        print(
            "  sensor-origin span xyz [mm]: "
            f"{np.array2string(spans, precision=3)}"
        )
        print(
            "  sensor-origin centered singular values [mm]: "
            f"{np.array2string(singular_values, precision=3)}"
        )
        if len(singular_values) >= 3 and singular_values[-1] < 1e-6:
            print("  WARNING: sensor origins are nearly lower-dimensional.")

    if len(T_base_s_all) >= 2:
        first_R = T_base_s_all[0][:3, :3]
        angles = np.asarray(
            [_rotation_angle_deg(first_R, T[:3, :3]) for T in T_base_s_all],
            dtype=float,
        )
        print(
            "  orientation diversity relative to first pose [deg]: "
            f"min={angles.min():.6g}, median={np.median(angles):.6g}, "
            f"max={angles.max():.6g}"
        )
        if angles.max() < 1.0:
            print("  WARNING: almost no rotational excitation in the scan set.")

    obs = translation_offset_observability(scans, plane_n)
    print(
        "  translation/offset observability: "
        f"rank={obs['rank']}/4, "
        f"min_sv={obs['min_singular_value']:.6g}, "
        f"condition={obs['condition']:.6g}"
    )
    print(
        "  translation/offset singular values: "
        + np.array2string(
            np.asarray(obs["singular_values"], dtype=float),
            precision=6,
        )
    )
    if int(obs["rank"]) < 4:
        print(
            "  WARNING: exact translation/plane-offset ambiguity remains "
            "(rank < 4). More beta/theta samples help only if they change q_i."
        )
    elif float(obs["condition"]) > 1e8:
        print(
            "  WARNING: translation/offset block is numerically ill-conditioned."
        )

    # Analyze the exact fixed-plane linear system used by the paper-style solver.
    # This is distinct from the generic point-to-plane SE(3)+plane Jacobian below.
    print_paper_linear_system_diagnostics(
        scans=scans,
        plane_n=plane_n,
        plane_l=plane_l,
    )

    full_obs = full_local_observability(
        scans=scans,
        T_reference=T_true,
        plane_n=plane_n,
        plane_l=plane_l,
    )
    print(
        "  FULL local observability (SE3 + unknown plane): "
        f"rank={full_obs['rank']}/9, "
        f"min_sv_norm={full_obs['min_singular_value']:.6g}, "
        f"condition_norm={full_obs['condition']:.6g}, "
        f"reference_rms={full_obs['residual_rms_mm']:.6g} mm"
    )
    print(
        "  FULL normalized singular values: "
        + np.array2string(
            np.asarray(full_obs["singular_values"], dtype=float),
            precision=6,
        )
    )
    print(
        "  FULL raw singular values: "
        + np.array2string(
            np.asarray(full_obs["raw_singular_values"], dtype=float),
            precision=6,
        )
    )
    print(
        "  FULL Jacobian column norms "
        "[rot(rad), trans(mm), normal, offset(mm)]: "
        + np.array2string(
            np.asarray(full_obs["column_norms"], dtype=float),
            precision=6,
        )
    )
    weakest_terms = sorted(
        zip(full_obs["labels"], np.asarray(full_obs["null_vector"], dtype=float)),
        key=lambda item: abs(item[1]),
        reverse=True,
    )
    print(
        "  FULL weakest state direction (normalized): "
        + ", ".join(f"{name}={value:+.4f}" for name, value in weakest_terms)
    )
    if int(full_obs["rank"]) < 9:
        print(
            "  WARNING: the complete SE(3)+plane problem is locally rank deficient. "
            "The weakest-state vector identifies the coupled unobservable DOFs."
        )
    elif float(full_obs["condition"]) > 1e6:
        print(
            "  WARNING: the complete problem is full rank but locally ill-conditioned."
        )



def _apply_se3_local_perturbation(
    T: np.ndarray,
    delta: np.ndarray,
) -> np.ndarray:
    """Apply right-multiplicative SO(3) and additive translation perturbation.

    State order is [dphi_x, dphi_y, dphi_z, dt_x, dt_y, dt_z],
    with rotation in radians and translation in millimetres.
    """
    T = np.asarray(T, dtype=float).reshape(4, 4)
    delta = np.asarray(delta, dtype=float).reshape(6)
    out = T.copy()
    out[:3, :3] = T[:3, :3] @ Rotation.from_rotvec(delta[:3]).as_matrix()
    out[:3, 3] = T[:3, 3] + delta[3:]
    return out


def _se3_local_difference(
    T_reference: np.ndarray,
    T_other: np.ndarray,
) -> np.ndarray:
    """Return local 6-vector from T_reference to T_other.

    Rotation is the right-invariant rotation vector [rad]; translation is [mm].
    """
    T_reference = np.asarray(T_reference, dtype=float).reshape(4, 4)
    T_other = np.asarray(T_other, dtype=float).reshape(4, 4)
    dR = T_reference[:3, :3].T @ T_other[:3, :3]
    dphi = Rotation.from_matrix(dR).as_rotvec()
    dt = T_other[:3, 3] - T_reference[:3, 3]
    return np.r_[dphi, dt]


def _one_alternating_iteration(
    scans_by_plane: dict[int, list[LaserScan]],
    T_input: np.ndarray,
    plane_offset_mode: str = "fitted",
) -> np.ndarray:
    """Evaluate one full plane-fit/hand-eye alternating iteration F(T)."""
    one = calibrate_planes(
        scans_by_plane,
        T_init=np.asarray(T_input, dtype=float).reshape(4, 4),
        max_iter=1,
        tol=-1.0,
        plane_offset_mode=plane_offset_mode,
    )
    return np.asarray(one.T_ef_s, dtype=float).reshape(4, 4)


def iteration_mapping_jacobian(
    scans_by_plane: dict[int, list[LaserScan]],
    T_reference: np.ndarray,
    rotation_eps_rad: float = 1e-6,
    translation_eps_mm: float = 1e-4,
    plane_offset_mode: str = "fitted",
) -> dict[str, object]:
    """Numerically linearize the actual one-step map T_{k+1}=F(T_k).

    The returned 6x6 Jacobian maps a small local input error to the next-step
    local output error. Its spectral radius predicts local linear convergence:
      rho < 1  -> locally contractive,
      rho ~= 1 -> very slow convergence,
      rho > 1  -> locally divergent.
    """
    T_reference = np.asarray(T_reference, dtype=float).reshape(4, 4)
    F0 = _one_alternating_iteration(
        scans_by_plane,
        T_reference,
        plane_offset_mode=plane_offset_mode,
    )
    eps = np.array(
        [rotation_eps_rad] * 3 + [translation_eps_mm] * 3, dtype=float
    )
    J = np.zeros((6, 6), dtype=float)

    for col in range(6):
        d = np.zeros(6, dtype=float)
        d[col] = eps[col]
        F_plus = _one_alternating_iteration(
            scans_by_plane,
            _apply_se3_local_perturbation(T_reference, d),
            plane_offset_mode=plane_offset_mode,
        )
        F_minus = _one_alternating_iteration(
            scans_by_plane,
            _apply_se3_local_perturbation(T_reference, -d),
            plane_offset_mode=plane_offset_mode,
        )
        y_plus = _se3_local_difference(F0, F_plus)
        y_minus = _se3_local_difference(F0, F_minus)
        J[:, col] = (y_plus - y_minus) / (2.0 * eps[col])

    eigvals = np.linalg.eigvals(J)
    singular_values = np.linalg.svd(J, compute_uv=False)
    spectral_radius = float(np.max(np.abs(eigvals)))

    return {
        "J": J,
        "eigenvalues": eigvals,
        "singular_values": singular_values,
        "spectral_radius": spectral_radius,
        "fixed_point_defect": _se3_local_difference(T_reference, F0),
        "labels": (
            "dphi_x", "dphi_y", "dphi_z",
            "dt_x", "dt_y", "dt_z",
        ),
    }


def _safe_successive_ratios(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    if len(values) < 2:
        return np.empty(0, dtype=float)
    prev = values[:-1]
    nxt = values[1:]
    valid = np.isfinite(prev) & np.isfinite(nxt) & (np.abs(prev) > 1e-15)
    out = np.full(len(prev), np.nan, dtype=float)
    out[valid] = nxt[valid] / prev[valid]
    return out


def print_iteration_convergence_diagnostics(
    scans_by_plane: dict[int, list[LaserScan]],
    T_true: np.ndarray,
    T_history: list[np.ndarray],
    plane_rms_history: list[float],
    plane_offset_mode: str = "fitted",
) -> None:
    """Print empirical convergence ratios and the one-step-map Jacobian."""
    if T_history:
        t_errors = np.asarray(
            [np.linalg.norm(T[:3, 3] - T_true[:3, 3]) for T in T_history],
            dtype=float,
        )
        r_errors_rad = np.asarray(
            [
                Rotation.from_matrix(T_true[:3, :3].T @ T[:3, :3]).magnitude()
                for T in T_history
            ],
            dtype=float,
        )
        t_ratios = _safe_successive_ratios(t_errors)
        r_ratios = _safe_successive_ratios(r_errors_rad)
        if len(t_ratios):
            finite = t_ratios[np.isfinite(t_ratios)]
            if len(finite):
                print(
                    "  empirical translation contraction ratio e[k+1]/e[k]: "
                    f"median={np.median(finite):.6g}, last={finite[-1]:.6g}"
                )
        if len(r_ratios):
            finite = r_ratios[np.isfinite(r_ratios)]
            if len(finite):
                print(
                    "  empirical rotation contraction ratio e[k+1]/e[k]: "
                    f"median={np.median(finite):.6g}, last={finite[-1]:.6g}"
                )

    rms_ratios = _safe_successive_ratios(np.asarray(plane_rms_history, dtype=float))
    finite_rms = rms_ratios[np.isfinite(rms_ratios)]
    if len(finite_rms):
        print(
            "  empirical plane-RMS contraction ratio r[k+1]/r[k]: "
            f"median={np.median(finite_rms):.6g}, last={finite_rms[-1]:.6g}"
        )

    try:
        mapping = iteration_mapping_jacobian(
            scans_by_plane=scans_by_plane,
            T_reference=T_true,
            plane_offset_mode=plane_offset_mode,
        )
        rho = float(mapping["spectral_radius"])
        defect = np.asarray(mapping["fixed_point_defect"], dtype=float)
        eigvals = np.asarray(mapping["eigenvalues"])
        svals = np.asarray(mapping["singular_values"], dtype=float)
        print(
            "  ITERATION MAP local analysis T[k+1]=F(T[k]): "
            f"spectral_radius={rho:.6g}, "
            f"fixed_point_defect_rot={np.linalg.norm(defect[:3]):.6g} rad, "
            f"fixed_point_defect_trans={np.linalg.norm(defect[3:]):.6g} mm"
        )
        print(
            "  ITERATION MAP singular values: "
            + np.array2string(svals, precision=6)
        )
        print(
            "  ITERATION MAP eigenvalue magnitudes: "
            + np.array2string(np.sort(np.abs(eigvals))[::-1], precision=6)
        )
        if rho < 1.0:
            if rho > 0.95:
                print(
                    "  INTERPRETATION: locally convergent but very slow; "
                    "the dominant error mode decays by roughly rho per iteration."
                )
            else:
                print(
                    "  INTERPRETATION: locally contractive; convergence should be linear."
                )
        elif np.isclose(rho, 1.0, rtol=1e-3, atol=1e-6):
            print(
                "  INTERPRETATION: nearly neutral mode; convergence can stagnate."
            )
        else:
            print(
                "  WARNING: spectral radius exceeds one; the alternating iteration "
                "is locally divergent around the reference state."
            )
    except Exception as exc:
        print(
            "  ITERATION MAP analysis failed: "
            f"{type(exc).__name__}: {exc}"
        )

def print_solver_diagnostics(
    *,
    system_idx: int,
    scans_by_plane: dict[int, list[LaserScan]],
    T_true: np.ndarray,
    T_init: np.ndarray | None,
    T_est: np.ndarray,
    plane_n: np.ndarray,
    plane_l: float,
    result: object | None,
    run_known_baseline: bool,
    offset_anchor_applied: bool = False,
    offset_anchor_plane: tuple[np.ndarray, float] | None = None,
) -> None:
    """Print initial, iterative, final and known-plane baseline diagnostics."""
    scans = scans_by_plane[0]
    print(f"[debug trial {system_idx:04d}] solver")

    if T_init is not None:
        init_t = float(np.linalg.norm(T_init[:3, 3] - T_true[:3, 3]))
        init_r = float(rot_error_deg(T_init[:3, :3], T_true[:3, :3]))
        init_points = _reconstruct_points_base(scans, T_init)
        init_plane = estimate_plane_from_handeye(scans, T_init)
        init_plane_err = _plane_difference(init_plane, (plane_n, plane_l))
        init_fit_stats = _plane_residual_stats(
            init_points, init_plane[0], init_plane[1]
        )
        print(
            f"  initial hand-eye error: translation={init_t:.6g} mm, "
            f"rotation={init_r:.6g} deg"
        )
        print(
            "  plane fitted using initial hand-eye: "
            f"normal_error={init_plane_err[0]:.6g} deg, "
            f"offset_error={init_plane_err[1]:.6g} mm, "
            f"self_fit_rms={init_fit_stats['rms']:.6g} mm"
        )

    final_t = float(np.linalg.norm(T_est[:3, 3] - T_true[:3, 3]))
    final_r = float(rot_error_deg(T_est[:3, :3], T_true[:3, :3]))
    final_points = _reconstruct_points_base(scans, T_est)
    final_true_stats = _plane_residual_stats(final_points, plane_n, plane_l)
    final_plane = estimate_plane_from_handeye(scans, T_est)
    final_plane_err = _plane_difference(final_plane, (plane_n, plane_l))
    final_self_stats = _plane_residual_stats(
        final_points, final_plane[0], final_plane[1]
    )
    print(
        f"  final hand-eye error: translation={final_t:.6g} mm, "
        f"rotation={final_r:.6g} deg"
    )
    print(
        "  final reconstruction -> TRUE plane residual [mm]: "
        f"rms={final_true_stats['rms']:.6g}, "
        f"max={final_true_stats['max_abs']:.6g}"
    )
    print(
        "  plane fitted using final hand-eye: "
        f"normal_error={final_plane_err[0]:.6g} deg, "
        f"offset_error={final_plane_err[1]:.6g} mm, "
        f"self_fit_rms={final_self_stats['rms']:.6g} mm"
    )
    if offset_anchor_applied and offset_anchor_plane is not None:
        anchor_err = _plane_difference(
            offset_anchor_plane, (plane_n, plane_l)
        )
        print(
            "  TRUE-OFFSET ANCHOR applied: "
            f"anchored_normal_error={anchor_err[0]:.6g} deg, "
            f"anchored_offset_error={anchor_err[1]:.6g} mm"
        )
    if final_self_stats["rms"] < 1.0 and final_t > 10.0:
        print(
            "  IMPORTANT: points fit some plane well but hand-eye is wrong. "
            "This indicates ambiguity/local minimum or insufficient excitation, "
            "not raw point noise."
        )

    if result is not None:
        rank_history = list(getattr(result, "rank_history", []))
        cond_history = list(getattr(result, "cond_history", []))
        rms_history = list(getattr(result, "plane_rms_history", []))
        T_history = list(getattr(result, "T_history", []))
        print(f"  iterations recorded={len(T_history)}")
        if rank_history:
            print(f"  rank history: {rank_history}")
        if cond_history:
            print(
                "  condition history: "
                + np.array2string(np.asarray(cond_history), precision=4)
            )
        if rms_history:
            print(
                "  plane RMS history [mm]: "
                + np.array2string(np.asarray(rms_history), precision=6)
            )
        if T_history:
            t_errors = np.asarray(
                [np.linalg.norm(T[:3, 3] - T_true[:3, 3]) for T in T_history]
            )
            r_errors = np.asarray(
                [rot_error_deg(T[:3, :3], T_true[:3, :3]) for T in T_history]
            )
            print(
                "  per-iteration translation error [mm]: "
                + np.array2string(t_errors, precision=6)
            )
            print(
                "  per-iteration rotation error [deg]: "
                + np.array2string(r_errors, precision=6)
            )

        print_iteration_convergence_diagnostics(
            scans_by_plane=scans_by_plane,
            T_true=T_true,
            T_history=T_history,
            plane_rms_history=rms_history,
            plane_offset_mode=str(
                getattr(result, "plane_offset_mode", "fitted")
            ),
        )

    if run_known_baseline:
        try:
            T_known = calibrate_with_known_planes(
                {0: (scans, plane_n, plane_l)}
            )
            known_t = float(np.linalg.norm(T_known[:3, 3] - T_true[:3, 3]))
            known_r = float(
                rot_error_deg(T_known[:3, :3], T_true[:3, :3])
            )
            known_points = _reconstruct_points_base(scans, T_known)
            known_stats = _plane_residual_stats(known_points, plane_n, plane_l)
            print(
                "  KNOWN-PLANE BASELINE: "
                f"translation_error={known_t:.6g} mm, "
                f"rotation_error={known_r:.6g} deg, "
                f"true_plane_rms={known_stats['rms']:.6g} mm"
            )
            if known_t > 10.0 or known_r > 1.0:
                print(
                    "  ERROR LIKELY BEFORE UNKNOWN-PLANE ITERATION: known-plane "
                    "solver, generated robot poses, or frame convention is inconsistent."
                )
            elif final_t > 10.0:
                print(
                    "  ISOLATED TO UNKNOWN-PLANE PIPELINE: data generation and "
                    "known-plane solver are consistent, but iterative plane/hand-eye "
                    "estimation is failing."
                )
        except Exception as exc:
            print(
                "  KNOWN-PLANE BASELINE FAILED: "
                f"{type(exc).__name__}: {exc}"
            )
    print()


def estimate_plane_from_handeye(
    scans: list[LaserScan],
    T_ef_s: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Estimate one base-frame plane using the final hand-eye estimate.

    The plane is fitted exactly once from all valid scan points reconstructed
    with ``T_ef_s``. The resulting plane is then kept fixed during nonlinear
    hand-eye refinement.
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



def _scan_param_number(
    param: dict,
    aliases: tuple[str, ...],
    default: float = float("nan"),
) -> float:
    """Read one numeric scan parameter while tolerating naming differences."""
    for key in aliases:
        if key in param:
            return float(param[key])
    return float(default)


def _sensor_pose_in_plane_frame(
    scan: LaserScan,
    T_ef_s_true: np.ndarray,
    plane_R: np.ndarray,
    plane_t: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return sensor origin and orientation expressed in the plane frame."""
    T_base_s = np.asarray(scan.T_base_ef, dtype=float) @ np.asarray(
        T_ef_s_true, dtype=float
    )
    R_base_s = T_base_s[:3, :3]
    t_base_s = T_base_s[:3, 3]

    plane_R = np.asarray(plane_R, dtype=float).reshape(3, 3)
    plane_t = np.asarray(plane_t, dtype=float).reshape(3)
    origin_plane = plane_R.T @ (t_base_s - plane_t)
    R_plane_s = plane_R.T @ R_base_s
    return origin_plane, R_plane_s


def save_sensor_poses_by_line_plot(
    *,
    T_ef_s_true: np.ndarray,
    scans: list[LaserScan],
    plane_R: np.ndarray,
    plane_t: np.ndarray,
    radius_mm: float,
    scan_params: list[dict],
    out_path: Path,
    frame_scale_mm: float = 25.0,
) -> Path:
    """Visualize all sensor coordinate frames separately for each target line.

    Scans are grouped by their explicit ``line_id`` metadata, so optional
    reference poses and reachability-filtered datasets are shown faithfully.
    Target poses are labelled ``p*`` and auxiliary reference poses ``r*``.

    All poses are transformed into the calibration-plane frame. Consequently,
    the target plane is always z=0 and the 40-degree rotation between target
    lines is directly visible, independent of the random base-frame plane pose.
    """
    n_lines = 9
    if not scan_params:
        raise ValueError("scan_params must be non-empty")
    if not scans:
        raise ValueError("at least one scan is required")
    if frame_scale_mm <= 0.0:
        raise ValueError("frame_scale_mm must be positive")

    grouped: list[list[tuple[str, bool, np.ndarray, np.ndarray]]] = [
        [] for _ in range(n_lines)
    ]
    all_origins: list[np.ndarray] = []
    fallback_target_index = [0] * n_lines
    for scan in scans:
        line_idx = int(scan.meta.get("line_id", -1))
        if not 0 <= line_idx < n_lines:
            raise ValueError("every circular scan must contain a valid line_id")
        origin, rotation = _sensor_pose_in_plane_frame(
            scan=scan,
            T_ef_s_true=T_ef_s_true,
            plane_R=plane_R,
            plane_t=plane_t,
        )
        is_reference = bool(scan.meta.get("reference_pose", False))
        if is_reference:
            label = f"r{int(scan.meta.get('reference_id', scan.scan_id))}"
        else:
            parameter_id = int(
                scan.meta.get("parameter_id", fallback_target_index[line_idx])
            )
            fallback_target_index[line_idx] += 1
            label = f"p{parameter_id}"
        grouped[line_idx].append((label, is_reference, origin, rotation))
        all_origins.append(origin)

    origins = np.asarray(all_origins, dtype=float)
    xy_limit = max(
        float(radius_mm) * 1.2,
        float(np.max(np.abs(origins[:, :2]))) + frame_scale_mm * 1.5,
    )
    z_min = min(0.0, float(np.min(origins[:, 2])) - frame_scale_mm * 1.5)
    z_max = max(0.0, float(np.max(origins[:, 2])) + frame_scale_mm * 1.5)
    if np.isclose(z_min, z_max):
        z_min -= frame_scale_mm
        z_max += frame_scale_mm

    fig = plt.figure(figsize=(18, 18))
    fig.subplots_adjust(
        left=0.035, right=0.985, bottom=0.085, top=0.925,
        wspace=0.16, hspace=0.24,
    )
    fig.suptitle(
        "Sensor poses for each circular target line\n"
        "coordinates expressed in the calibration-plane frame",
        fontsize=16,
    )

    for line_idx, line_poses in enumerate(grouped):
        ax = fig.add_subplot(3, 3, line_idx + 1, projection="3d")
        phi_deg = 40.0 * line_idx
        phi = np.deg2rad(phi_deg)
        endpoint = np.array(
            [radius_mm * np.cos(phi), radius_mm * np.sin(phi), 0.0],
            dtype=float,
        )

        # Calibration plane and the corresponding radial target line.
        plane_extent = xy_limit
        plane_x = np.array(
            [-plane_extent, plane_extent, plane_extent, -plane_extent, -plane_extent]
        )
        plane_y = np.array(
            [-plane_extent, -plane_extent, plane_extent, plane_extent, -plane_extent]
        )
        ax.plot(plane_x, plane_y, np.zeros_like(plane_x), linewidth=0.6, alpha=0.25)
        ax.plot(
            [0.0, endpoint[0]],
            [0.0, endpoint[1]],
            [0.0, 0.0],
            linewidth=2.2,
            label="target line",
        )
        ax.scatter([0.0], [0.0], [0.0], s=18)

        for label, is_reference, origin, R_plane_s in line_poses:
            ax.scatter(
                [origin[0]],
                [origin[1]],
                [origin[2]],
                s=24 if is_reference else 16,
                marker="^" if is_reference else "o",
                color="tab:orange" if is_reference else "tab:blue",
            )
            ax.text(
                origin[0],
                origin[1],
                origin[2],
                label,
                fontsize=6,
            )

            # Sensor-frame axes: X, Y and Z.
            axis_colors = ("tab:red", "tab:green", "tab:blue")
            for axis_idx, axis_name in enumerate(("X", "Y", "Z")):
                direction = R_plane_s[:, axis_idx]
                ax.quiver(
                    origin[0],
                    origin[1],
                    origin[2],
                    direction[0],
                    direction[1],
                    direction[2],
                    length=frame_scale_mm,
                    normalize=True,
                    arrow_length_ratio=0.18,
                    linewidth=0.8,
                    color=axis_colors[axis_idx],
                )

        ax.set_title(
            f"line {line_idx}: phi={phi_deg:.0f} deg, poses={len(line_poses)}"
        )
        ax.set_xlabel("plane X [mm]")
        ax.set_ylabel("plane Y [mm]")
        ax.set_zlabel("plane Z [mm]")
        ax.set_xlim(-xy_limit, xy_limit)
        ax.set_ylim(-xy_limit, xy_limit)
        ax.set_zlim(z_min, z_max)
        ax.set_box_aspect((2.0 * xy_limit, 2.0 * xy_limit, z_max - z_min))
        ax.view_init(elev=25.0, azim=-60.0)
        ax.grid(True, alpha=0.25)

    mapping_lines = []
    for idx, param in enumerate(scan_params):
        d_mm = _scan_param_number(
            param,
            ("d_mm", "height_mm", "distance_mm", "d"),
        )
        theta_deg = _scan_param_number(
            param,
            ("theta_deg", "projection_deg", "projection_angle_deg", "theta"),
        )
        beta_deg = _scan_param_number(
            param,
            ("beta_deg", "tilt_deg", "tilt_angle_deg", "beta"),
        )
        mapping_lines.append(
            f"p{idx}: d={d_mm:g} mm, theta={theta_deg:g} deg, beta={beta_deg:g} deg"
        )

    mapping_rows = [
        "   |   ".join(mapping_lines[start : start + 3])
        for start in range(0, len(mapping_lines), 3)
    ]
    fig.text(
        0.5,
        0.012,
        "Sensor axes: X=red, Y=green, Z=blue\n"
        "Target parameter index used in every panel; r*=auxiliary reference:\n"
        + "\n".join(mapping_rows),
        ha="center",
        va="bottom",
        fontsize=8,
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out_path


def make_and_save_sensor_pose_plot(
    seed: int,
    x_values: np.ndarray,
    noise_std: float,
    out_path: Path,
    radius_mm: float,
    check_reachability: bool,
    plane_tilt_min_deg: float,
    plane_tilt_max_deg: float,
    plane_yaw_min_deg: float,
    plane_yaw_max_deg: float,
    plane_center_xy_range_mm: float,
    plane_center_z_range_mm: tuple[float, float],
    scan_params: list[dict],
    pose_geometry: str,
    reference_scan_params: list[dict],
    reference_scan_count: int,
    frame_scale_mm: float,
) -> Path:
    """Generate one deterministic debug system and plot its sensor poses."""
    rng = np.random.default_rng(seed)
    T_true, _, _ = sample_random_handeye(rng)
    plane_R, plane_t, _, _ = sample_random_plane_pose(
        rng=rng,
        tilt_min_deg=plane_tilt_min_deg,
        tilt_max_deg=plane_tilt_max_deg,
        yaw_min_deg=plane_yaw_min_deg,
        yaw_max_deg=plane_yaw_max_deg,
        center_x_range_mm=(-plane_center_xy_range_mm, plane_center_xy_range_mm),
        center_y_range_mm=(-plane_center_xy_range_mm, plane_center_xy_range_mm),
        center_z_range_mm=plane_center_z_range_mm,
    )
    scans_by_plane = generate_optimal_single_plane_dataset(
        T_ef_s_true=T_true,
        plane_R=plane_R,
        plane_t=plane_t,
        rng=rng,
        x_values=x_values,
        radius_mm=radius_mm,
        noise_std=noise_std,
        check_reachability=check_reachability,
        scan_params=scan_params,
        pose_geometry=pose_geometry,
        reference_scan_params=reference_scan_params,
        reference_scan_count=reference_scan_count,
    )
    return save_sensor_poses_by_line_plot(
        T_ef_s_true=T_true,
        scans=scans_by_plane[0],
        plane_R=plane_R,
        plane_t=plane_t,
        radius_mm=radius_mm,
        scan_params=scan_params,
        out_path=out_path,
        frame_scale_mm=frame_scale_mm,
    )


def make_and_save_translation_gauge_plot(
    *,
    seed: int,
    x_values: np.ndarray,
    noise_std: float,
    out_path: Path,
    shift_span_mm: float,
    shift_samples: int,
    radius_mm: float,
    check_reachability: bool,
    plane_tilt_min_deg: float,
    plane_tilt_max_deg: float,
    plane_yaw_min_deg: float,
    plane_yaw_max_deg: float,
    plane_center_xy_range_mm: float,
    plane_center_z_range_mm: tuple[float, float],
    scan_params: list[dict],
    pose_geometry: str,
    reference_scan_params: list[dict],
    reference_scan_count: int,
) -> tuple[Path, dict[str, np.ndarray]]:
    """Generate one deterministic system and visualize its gauge sweep."""
    if shift_span_mm <= 0.0:
        raise ValueError("shift_span_mm must be positive")
    if shift_samples < 3:
        raise ValueError("shift_samples must be at least 3")
    theta_values = {
        round(
            abs(
                _scan_param_number(
                    param,
                    (
                        "theta_deg",
                        "projection_deg",
                        "projection_angle_deg",
                        "theta",
                    ),
                )
            ),
            12,
        )
        for param in scan_params
    }
    if reference_scan_count > 0:
        theta_values.update(
            round(
                abs(
                    _scan_param_number(
                        param,
                        (
                            "theta_deg",
                            "projection_deg",
                            "projection_angle_deg",
                            "theta",
                        ),
                    )
                ),
                12,
            )
            for param in reference_scan_params
        )
    if len(theta_values) != 1:
        raise ValueError(
            "--debug-gauge-plot requires one fixed |theta|; use "
            "--reference-scans 0 and a single --theta-deg value"
        )
    theta_deg = float(next(iter(theta_values)))

    rng = np.random.default_rng(seed)
    T_true, _, _ = sample_random_handeye(rng)
    plane_R, plane_t, plane_n, plane_l = sample_random_plane_pose(
        rng=rng,
        tilt_min_deg=plane_tilt_min_deg,
        tilt_max_deg=plane_tilt_max_deg,
        yaw_min_deg=plane_yaw_min_deg,
        yaw_max_deg=plane_yaw_max_deg,
        center_x_range_mm=(-plane_center_xy_range_mm, plane_center_xy_range_mm),
        center_y_range_mm=(-plane_center_xy_range_mm, plane_center_xy_range_mm),
        center_z_range_mm=plane_center_z_range_mm,
    )
    scans_by_plane = generate_optimal_single_plane_dataset(
        T_ef_s_true=T_true,
        plane_R=plane_R,
        plane_t=plane_t,
        rng=rng,
        x_values=x_values,
        radius_mm=radius_mm,
        noise_std=noise_std,
        check_reachability=check_reachability,
        scan_params=scan_params,
        pose_geometry=pose_geometry,
        reference_scan_params=reference_scan_params,
        reference_scan_count=reference_scan_count,
    )
    shifts = np.linspace(
        -float(shift_span_mm),
        float(shift_span_mm),
        int(shift_samples),
    )
    sweep = translation_gauge_sweep(
        scans=scans_by_plane[0],
        T_true=T_true,
        plane_n=plane_n,
        plane_l=plane_l,
        shifts_mm=shifts,
    )
    return (
        save_translation_gauge_sweep_plot(
            sweep=sweep,
            theta_deg=theta_deg,
            out_path=out_path,
        ),
        sweep,
    )

def make_and_save_debug_scene_plot(
    seed: int,
    x_values: np.ndarray,
    noise_std: float,
    out_path: Path,
    plane_size_mm: float,
    radius_mm: float,
    check_reachability: bool,
    plane_tilt_min_deg: float,
    plane_tilt_max_deg: float,
    plane_yaw_min_deg: float,
    plane_yaw_max_deg: float,
    plane_center_xy_range_mm: float,
    plane_center_z_range_mm: tuple[float, float],
    scan_params: list[dict],
    pose_geometry: str,
    reference_scan_params: list[dict],
    reference_scan_count: int,
) -> Path:
    """Generate and plot one optimal-parameter single-plane system."""

    rng = np.random.default_rng(seed)
    T_true, _, _ = sample_random_handeye(rng)
    plane_R, plane_t, plane_n, plane_l = sample_random_plane_pose(
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
    scans_by_plane = generate_optimal_single_plane_dataset(
        T_ef_s_true=T_true,
        plane_R=plane_R,
        plane_t=plane_t,
        rng=rng,
        x_values=x_values,
        radius_mm=radius_mm,
        noise_std=noise_std,
        check_reachability=check_reachability,
        scan_params=scan_params,
        pose_geometry=pose_geometry,
        reference_scan_params=reference_scan_params,
        reference_scan_count=reference_scan_count,
    )

    return save_plane_scene_plot(
        T_ef_s_true=T_true,
        scans_by_plane=scans_by_plane,
        planes=[(plane_n, plane_l)],
        out_path=out_path,
        plane_size_mm=plane_size_mm,
    )


def run_one_trial(
    system_idx: int,
    rng: np.random.Generator,
    x_values: np.ndarray,
    noise_std: float,
    mode: str,
    max_iter: int,
    tol: float,
    init_mode: InitialGuessMode,
    rel_offset: float,
    init_translation_range_mm: float,
    init_angle_range_deg: float,
    plane_offset_mode: str,
    linear_multistart: bool,
    linear_multistart_threshold_mm: float,
    linear_multistart_angle_deg: float,
    nonlinear_refine: bool,
    nonlinear_plane_mode: str,
    nonlinear_max_nfev: int,
    nonlinear_ftol: float,
    nonlinear_xtol: float,
    nonlinear_gtol: float,
    nonlinear_loss: RobustLoss,
    nonlinear_f_scale_mm: float,
    radius_mm: float,
    check_reachability: bool,
    plane_tilt_min_deg: float,
    plane_tilt_max_deg: float,
    plane_yaw_min_deg: float,
    plane_yaw_max_deg: float,
    plane_center_xy_range_mm: float,
    plane_center_z_range_mm: tuple[float, float],
    scan_params: list[dict],
    pose_geometry: str,
    reference_scan_params: list[dict],
    reference_scan_count: int,
    debug_diagnostics: bool = False,
    debug_known_baseline: bool = True,
    fix_true_plane_offset: bool = False,
    gt_rng: np.random.Generator | None = None,
    init_rng: np.random.Generator | None = None,
) -> SinglePlaneTrialResult:
    T_true, true_angles_deg, true_translation_mm = sample_random_handeye(
        rng if gt_rng is None else gt_rng
    )

    plane_R, plane_t, plane_n, plane_l = sample_random_plane_pose(
        rng=rng,
        tilt_min_deg=plane_tilt_min_deg,
        tilt_max_deg=plane_tilt_max_deg,
        yaw_min_deg=plane_yaw_min_deg,
        yaw_max_deg=plane_yaw_max_deg,
        center_x_range_mm=(-plane_center_xy_range_mm, plane_center_xy_range_mm),
        center_y_range_mm=(-plane_center_xy_range_mm, plane_center_xy_range_mm),
        center_z_range_mm=plane_center_z_range_mm,
    )
    plane = (plane_n, plane_l)

    scans_by_plane = generate_optimal_single_plane_dataset(
        T_ef_s_true=T_true,
        plane_R=plane_R,
        plane_t=plane_t,
        rng=rng,
        x_values=x_values,
        radius_mm=radius_mm,
        noise_std=noise_std,
        check_reachability=check_reachability,
        scan_params=scan_params,
        pose_geometry=pose_geometry,
        reference_scan_params=reference_scan_params,
        reference_scan_count=reference_scan_count,
    )

    if debug_diagnostics:
        print_dataset_diagnostics(
            system_idx=system_idx,
            scans_by_plane=scans_by_plane,
            T_true=T_true,
            plane_n=plane_n,
            plane_l=plane_l,
            expected_scan_count=9 * len(scan_params) + reference_scan_count,
            noise_std=noise_std,
            scan_params=scan_params,
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
    linear_multistart_used = False
    linear_start_count = 1
    final_linear_plane_rms_mm = float("nan")
    T_init: np.ndarray | None = None
    solver_result: object | None = None
    offset_anchor_applied = False
    offset_anchor_plane: tuple[np.ndarray, float] | None = None

    if mode == "known":
        known_planes = {0: (scans_by_plane[0], plane_n, plane_l)}
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
                T_est,
                plane_mode="refit",
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

        final_linear_plane_rms_mm = _final_self_fit_plane_rms_mm(
            scans_by_plane,
            result.T_ef_s,
        )
        if (
            linear_multistart
            and final_linear_plane_rms_mm
            > float(linear_multistart_threshold_mm)
        ):
            linear_multistart_used = True
            candidates = [result]
            for retry_init in _linear_retry_initial_guesses(
                T_init,
                linear_multistart_angle_deg,
            ):
                linear_start_count += 1
                try:
                    candidates.append(
                        calibrate_planes(
                            scans_by_plane,
                            T_init=retry_init,
                            max_iter=max_iter,
                            tol=tol,
                            plane_offset_mode=plane_offset_mode,
                        )
                    )
                except np.linalg.LinAlgError:
                    continue
            candidate_rms = [
                _final_self_fit_plane_rms_mm(
                    scans_by_plane,
                    candidate.T_ef_s,
                )
                for candidate in candidates
            ]
            eligible_indices = [
                index
                for index, candidate in enumerate(candidates)
                if candidate.converged
            ]
            if not eligible_indices:
                eligible_indices = list(range(len(candidates)))
            selected_index = min(
                eligible_indices,
                key=lambda index: candidate_rms[index],
            )
            result = candidates[selected_index]
            final_linear_plane_rms_mm = float(candidate_rms[selected_index])
        solver_result = result

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

        if fix_true_plane_offset:
            # Debug/oracle experiment:
            # keep the normal estimated by the unknown-plane iteration, but
            # replace only the unobservable plane offset with the simulated GT l.
            estimated_normal, estimated_l = estimate_plane_from_handeye(
                scans=scans_by_plane[0],
                T_ef_s=T_est,
            )
            estimated_normal, estimated_l = _align_plane_representation(
                (estimated_normal, estimated_l),
                plane_n,
            )
            offset_anchor_plane = (estimated_normal, float(plane_l))
            T_est = calibrate_with_known_planes(
                {
                    0: (
                        scans_by_plane[0],
                        offset_anchor_plane[0],
                        offset_anchor_plane[1],
                    )
                }
            )
            transform_history.append(T_est)
            offset_anchor_applied = True

        if nonlinear_refine:
            # Unknown-plane refinement must normally refit/eliminate the plane
            # for every candidate hand-eye transform. Fixing a plane fitted from
            # an inaccurate T_est freezes the remaining hand-eye error into that
            # plane and produces an almost-zero nonlinear update.
            if offset_anchor_applied and offset_anchor_plane is not None:
                nonlinear_result = refine_handeye_nonlinear(
                    scans_by_plane,
                    T_est,
                    plane_mode="fixed",
                    planes={0: offset_anchor_plane},
                    loss=nonlinear_loss,
                    f_scale_mm=nonlinear_f_scale_mm,
                    max_nfev=nonlinear_max_nfev,
                    ftol=nonlinear_ftol,
                    xtol=nonlinear_xtol,
                    gtol=nonlinear_gtol,
                )
            elif nonlinear_plane_mode == "refit":
                nonlinear_result = refine_handeye_nonlinear(
                    scans_by_plane,
                    T_est,
                    plane_mode="refit",
                    loss=nonlinear_loss,
                    f_scale_mm=nonlinear_f_scale_mm,
                    max_nfev=nonlinear_max_nfev,
                    ftol=nonlinear_ftol,
                    xtol=nonlinear_xtol,
                    gtol=nonlinear_gtol,
                )
            else:
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

        iter_T_frob_error = iter_frobenius_error(
            transform_history,
            T_true,
            T_initial=T_init,
        )

    else:
        raise ValueError("mode must be 'known' or 'unknown'")

    if debug_diagnostics:
        print_solver_diagnostics(
            system_idx=system_idx,
            scans_by_plane=scans_by_plane,
            T_true=T_true,
            T_init=T_init,
            T_est=T_est,
            plane_n=plane_n,
            plane_l=plane_l,
            result=solver_result,
            run_known_baseline=debug_known_baseline,
            offset_anchor_applied=offset_anchor_applied,
            offset_anchor_plane=offset_anchor_plane,
        )

    translation_error = T_est[:3, 3] - T_true[:3, 3]
    gauge_components = translation_error_sensor_z_components(T_est, T_true)
    rotation_vector_error = rotation_vector_error_deg(
        T_est[:3, :3],
        T_true[:3, :3],
    )
    paper_success = bool(
        converged
        and iterations <= 2000
        and np.all(np.abs(translation_error) < 0.01)
    )

    return SinglePlaneTrialResult(
        system_idx=system_idx,
        mode=mode,
        init_mode=init_mode,
        plane_offset_mode=plane_offset_mode,
        pose_geometry=pose_geometry,
        n_reference_scans=sum(
            bool(scan.meta.get("reference_pose", False))
            for scan in scans_by_plane[0]
        ),
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
        gauge_parallel_error_mm=float(
            gauge_components["parallel_signed_mm"]
        ),
        gauge_perpendicular_error_mm=float(
            gauge_components["perpendicular_mm"]
        ),
        gauge_axis_angle_deg=float(gauge_components["axis_angle_deg"]),
        gauge_parallel_fraction=float(
            gauge_components["parallel_fraction"]
        ),
        paper_success=paper_success,
        linear_multistart_used=linear_multistart_used,
        linear_start_count=linear_start_count,
        final_linear_plane_rms_mm=final_linear_plane_rms_mm,
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
        description="Optimal-parameter single-plane benchmark for hand-eye calibration."
    )
    parser.add_argument("--systems", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--mode", choices=["known", "unknown"], default="unknown")
    parser.add_argument("--profile-points", type=int, default=100)
    parser.add_argument("--profile-half-width", type=float, default=25.0)
    parser.add_argument("--noise-std", type=float, default=0.5)
    parser.add_argument("--radius-mm", type=float, default=100.0)
    parser.add_argument(
        "--heights-mm",
        type=float,
        nargs="+",
        default=[60.0, 90.0, 120.0],
        help="sensor distances d in mm",
    )
    parser.add_argument(
        "--theta-deg",
        type=float,
        nargs="+",
        default=[30.0],
        help="projection angles theta in degrees",
    )
    parser.add_argument(
        "--beta-deg",
        type=float,
        nargs="+",
        default=[60.0, 90.0, 120.0],
        help="sensor tilt angles beta in degrees",
    )
    parser.add_argument(
        "--pose-geometry",
        choices=["paper_incidence", "observable_dihedral"],
        default="paper_incidence",
        help=(
            "paper_incidence matches the published/deposited theta incidence "
            "geometry; observable_dihedral retains the repository's older "
            "engineering convention"
        ),
    )
    parser.add_argument(
        "--reference-scans",
        type=int,
        default=24,
        help=(
            "number of explicitly counted auxiliary second-theta scans used "
            "to break the fixed-theta translation/plane-offset gauge; 0 runs "
            "the strict but rank-deficient 81-scan grid"
        ),
    )
    parser.add_argument(
        "--reference-theta-deg",
        type=float,
        default=60.0,
        help=(
            "incidence angle of the auxiliary observability scans; these "
            "scans are an engineering extension, not part of the paper's "
            "reduced theta=30 grid"
        ),
    )
    parser.add_argument(
        "--reference-heights-mm",
        type=float,
        nargs="+",
        default=[60.0, 90.0, 120.0],
    )
    parser.add_argument(
        "--reference-beta-deg",
        type=float,
        nargs="+",
        default=[60.0, 90.0, 120.0],
    )
    parser.add_argument(
        "--check-reachability",
        action="store_true",
        help="discard scans whose flange pose fails the simple reachability test",
    )
    parser.add_argument("--plane-tilt-min-deg", type=float, default=1.0)
    parser.add_argument("--plane-tilt-max-deg", type=float, default=5.0)
    parser.add_argument("--plane-yaw-min-deg", type=float, default=-5.0)
    parser.add_argument("--plane-yaw-max-deg", type=float, default=5.0)
    parser.add_argument("--plane-center-xy-range-mm", type=float, default=100.0)
    parser.add_argument("--plane-center-z-min-mm", type=float, default=400.0)
    parser.add_argument("--plane-center-z-max-mm", type=float, default=550.0)
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
        help=(
            "joint estimates hand-eye translation and unknown plane offsets "
            "in the same linear update and rejects rank-deficient data; "
            "fitted uses the legacy fit-then-solve offset iteration"
        ),
    )
    parser.add_argument(
        "--linear-multistart",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "retry six deterministic linear starts only when the first "
            "alternating solve has a high plane residual; this is a linear "
            "PCA-basin safeguard, not nonlinear refinement"
        ),
    )
    parser.add_argument(
        "--linear-multistart-threshold-mm",
        type=float,
        default=None,
        help=(
            "trigger residual for linear multistart; default is "
            "max(1 mm, 3*noise_std)"
        ),
    )
    parser.add_argument(
        "--linear-multistart-angle-deg",
        type=float,
        default=30.0,
    )
    parser.add_argument(
        "--nonlinear-refine",
        action="store_true",
        help="run a final six-parameter SE(3) nonlinear least-squares refinement",
    )
    parser.add_argument(
        "--nonlinear-plane-mode",
        choices=["refit", "fixed"],
        default="refit",
        help=(
            "unknown-plane nonlinear objective: refit the plane for every "
            "candidate hand-eye (recommended), or freeze the plane estimated "
            "from the alternating result"
        ),
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
        default=Path("single_plane_optimal_benchmark.csv"),
    )
    parser.add_argument("--debug-scene-plot", type=Path, default=None)
    parser.add_argument("--debug-scene-seed", type=int, default=None)
    parser.add_argument("--debug-scene-plane-size", type=float, default=700.0)
    parser.add_argument(
        "--debug-sensor-pose-plot",
        type=Path,
        default=None,
        help="save a 3x3 plot of all sensor frames grouped by circular target line",
    )
    parser.add_argument(
        "--debug-sensor-pose-seed",
        type=int,
        default=None,
        help="random seed for the sensor-pose plot; defaults to --seed",
    )
    parser.add_argument(
        "--debug-sensor-frame-scale-mm",
        type=float,
        default=25.0,
        help="display length of each sensor coordinate axis in millimetres",
    )
    parser.add_argument(
        "--debug-gauge-plot",
        type=Path,
        default=None,
        help=(
            "save a fixed-theta proof plot: residual after fitting the plane "
            "offset versus forced translation along sensor Z and sensor X"
        ),
    )
    parser.add_argument(
        "--debug-gauge-seed",
        type=int,
        default=None,
        help="random seed for the gauge plot; defaults to --seed",
    )
    parser.add_argument(
        "--debug-gauge-span-mm",
        type=float,
        default=150.0,
        help="positive/negative translation range shown in the gauge plot",
    )
    parser.add_argument(
        "--debug-gauge-samples",
        type=int,
        default=121,
        help="number of translation samples in the gauge plot",
    )
    parser.add_argument("--plot-dir", type=Path, default=None)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument(
        "--fix-true-plane-offset",
        action="store_true",
        help=(
            "debug/oracle test for unknown-plane mode: keep the estimated plane "
            "normal but replace its offset l with the simulated ground-truth l"
        ),
    )
    parser.add_argument(
        "--debug-diagnostics",
        action="store_true",
        help="print simulation, plane, excitation, rank and per-iteration diagnostics",
    )
    parser.add_argument(
        "--debug-trial",
        type=int,
        default=0,
        help="zero-based trial index for detailed diagnostics",
    )
    parser.add_argument(
        "--debug-all-trials",
        action="store_true",
        help="print detailed diagnostics for every trial",
    )
    parser.add_argument(
        "--no-debug-known-baseline",
        action="store_true",
        help="skip the known-plane baseline during detailed diagnostics",
    )
    parser.add_argument(
        "--debug-traceback",
        action="store_true",
        help="print full traceback when a trial raises an exception",
    )
    args = parser.parse_args()

    if args.reference_scans < 0:
        parser.error("--reference-scans must be non-negative")
    if args.linear_multistart_angle_deg <= 0.0:
        parser.error("--linear-multistart-angle-deg must be positive")
    if args.debug_gauge_span_mm <= 0.0:
        parser.error("--debug-gauge-span-mm must be positive")
    if args.debug_gauge_samples < 3:
        parser.error("--debug-gauge-samples must be at least 3")
    linear_multistart_threshold_mm = (
        max(1.0, 3.0 * float(args.noise_std))
        if args.linear_multistart_threshold_mm is None
        else float(args.linear_multistart_threshold_mm)
    )
    if linear_multistart_threshold_mm <= 0.0:
        parser.error("--linear-multistart-threshold-mm must be positive")

    plane_center_z_range_mm = (
        args.plane_center_z_min_mm,
        args.plane_center_z_max_mm,
    )

    x_values = np.linspace(
        -args.profile_half_width,
        args.profile_half_width,
        args.profile_points,
    )

    scan_params = make_scan_params(
        heights_mm=tuple(args.heights_mm),
        theta_deg=tuple(args.theta_deg),
        beta_deg=tuple(args.beta_deg),
    )
    reference_scan_params = make_scan_params(
        heights_mm=tuple(args.reference_heights_mm),
        theta_deg=(float(args.reference_theta_deg),),
        beta_deg=tuple(args.reference_beta_deg),
    )
    nominal_scan_count = 9 * len(scan_params) + args.reference_scans
    incidence_magnitudes = {
        round(abs(float(value)), 12) for value in args.theta_deg
    }
    if args.reference_scans > 0:
        incidence_magnitudes.add(
            round(abs(float(args.reference_theta_deg)), 12)
        )
    if (
        args.mode == "unknown"
        and args.pose_geometry == "paper_incidence"
        and len(incidence_magnitudes) < 2
    ):
        print(
            "[single-plane-optimal] WARNING: fixed-|theta| paper-incidence "
            "data cannot separate sensor-optical-axis translation from the "
            "unknown plane offset. Use --plane-offset-mode joint to reject "
            "this gauge, and add scans at a different |theta| for absolute "
            "translation."
        )

    if args.debug_scene_plot is not None:
        scene_seed = (
            args.seed if args.debug_scene_seed is None else args.debug_scene_seed
        )
        debug_path = make_and_save_debug_scene_plot(
            seed=scene_seed,
            x_values=x_values,
            noise_std=args.noise_std,
            out_path=args.debug_scene_plot,
            plane_size_mm=args.debug_scene_plane_size,
            radius_mm=args.radius_mm,
            check_reachability=args.check_reachability,
            plane_tilt_min_deg=args.plane_tilt_min_deg,
            plane_tilt_max_deg=args.plane_tilt_max_deg,
            plane_yaw_min_deg=args.plane_yaw_min_deg,
            plane_yaw_max_deg=args.plane_yaw_max_deg,
            plane_center_xy_range_mm=args.plane_center_xy_range_mm,
            plane_center_z_range_mm=plane_center_z_range_mm,
            scan_params=scan_params,
            pose_geometry=args.pose_geometry,
            reference_scan_params=reference_scan_params,
            reference_scan_count=args.reference_scans,
        )
        print(f"saved debug scene plot: {debug_path}")

    if args.debug_sensor_pose_plot is not None:
        pose_seed = (
            args.seed
            if args.debug_sensor_pose_seed is None
            else args.debug_sensor_pose_seed
        )
        pose_path = make_and_save_sensor_pose_plot(
            seed=pose_seed,
            x_values=x_values,
            noise_std=args.noise_std,
            out_path=args.debug_sensor_pose_plot,
            radius_mm=args.radius_mm,
            check_reachability=args.check_reachability,
            plane_tilt_min_deg=args.plane_tilt_min_deg,
            plane_tilt_max_deg=args.plane_tilt_max_deg,
            plane_yaw_min_deg=args.plane_yaw_min_deg,
            plane_yaw_max_deg=args.plane_yaw_max_deg,
            plane_center_xy_range_mm=args.plane_center_xy_range_mm,
            plane_center_z_range_mm=plane_center_z_range_mm,
            scan_params=scan_params,
            pose_geometry=args.pose_geometry,
            reference_scan_params=reference_scan_params,
            reference_scan_count=args.reference_scans,
            frame_scale_mm=args.debug_sensor_frame_scale_mm,
        )
        print(f"saved sensor pose plot: {pose_path}")

    if args.debug_gauge_plot is not None:
        gauge_seed = (
            args.seed if args.debug_gauge_seed is None else args.debug_gauge_seed
        )
        try:
            gauge_path, gauge_sweep = make_and_save_translation_gauge_plot(
                seed=gauge_seed,
                x_values=x_values,
                noise_std=args.noise_std,
                out_path=args.debug_gauge_plot,
                shift_span_mm=args.debug_gauge_span_mm,
                shift_samples=args.debug_gauge_samples,
                radius_mm=args.radius_mm,
                check_reachability=args.check_reachability,
                plane_tilt_min_deg=args.plane_tilt_min_deg,
                plane_tilt_max_deg=args.plane_tilt_max_deg,
                plane_yaw_min_deg=args.plane_yaw_min_deg,
                plane_yaw_max_deg=args.plane_yaw_max_deg,
                plane_center_xy_range_mm=args.plane_center_xy_range_mm,
                plane_center_z_range_mm=plane_center_z_range_mm,
                scan_params=scan_params,
                pose_geometry=args.pose_geometry,
                reference_scan_params=reference_scan_params,
                reference_scan_count=args.reference_scans,
            )
        except ValueError as exc:
            parser.error(str(exc))
        z_rms = np.asarray(gauge_sweep["sensor_z_rms_mm"], dtype=float)
        x_rms = np.asarray(gauge_sweep["sensor_x_rms_mm"], dtype=float)
        z_offset = np.asarray(
            gauge_sweep["sensor_z_offset_delta_mm"],
            dtype=float,
        )
        shifts = np.asarray(gauge_sweep["shifts_mm"], dtype=float)
        theta_for_plot = abs(float(args.theta_deg[0]))
        offset_law_error = float(
            np.max(
                np.abs(
                    z_offset
                    + shifts * np.cos(np.radians(theta_for_plot))
                )
            )
        )
        print(
            "fixed-theta gauge check: "
            f"sensor-Z RMS span={np.ptp(z_rms):.3e} mm, "
            f"sensor-X max RMS={np.max(x_rms):.6g} mm, "
            f"offset-law max error={offset_law_error:.3e} mm"
        )
        print(f"saved translation gauge plot: {gauge_path}")

    results: list[SinglePlaneTrialResult] = []
    failures = 0
    first_failure: str | None = None
    start_time = time.perf_counter()
    log_every = max(1, args.log_every)

    if args.verbose:
        print(
            f"[single-plane-optimal] start | systems={args.systems} | mode={args.mode} | "
            f"nominal_scans={nominal_scan_count} | noise={args.noise_std} mm | "
            f"init_mode={args.init_mode} | nonlinear={args.nonlinear_refine} "
            f"({args.nonlinear_plane_mode}) | offset_mode={args.plane_offset_mode} | "
            f"linear_multistart={args.linear_multistart} "
            f"(trigger>{linear_multistart_threshold_mm:g} mm) | "
            f"fix_true_plane_offset={args.fix_true_plane_offset}"
        )
        print(
            f"[single-plane-optimal] grid | d={tuple(args.heights_mm)} | "
            f"theta={tuple(args.theta_deg)} | beta={tuple(args.beta_deg)} | "
            f"combinations={len(scan_params)} | geometry={args.pose_geometry}"
        )
        print(
            "[single-plane-optimal] observability extension | "
            f"reference_scans={args.reference_scans} | "
            f"reference_d={tuple(args.reference_heights_mm)} | "
            f"reference_theta={args.reference_theta_deg:g} | "
            f"reference_beta={tuple(args.reference_beta_deg)}"
        )
    elif args.reference_scans > 0:
        print(
            "[single-plane-optimal] note: adding "
            f"{args.reference_scans} auxiliary theta="
            f"{args.reference_theta_deg:g} deg observability scans; this is "
            "an engineering extension, not the paper's reduced 81-scan grid"
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
                noise_std=args.noise_std,
                mode=args.mode,
                max_iter=args.max_iter,
                tol=args.tol,
                init_mode=args.init_mode,
                rel_offset=args.rel_offset,
                init_translation_range_mm=args.init_translation_range_mm,
                init_angle_range_deg=args.init_angle_range_deg,
                plane_offset_mode=args.plane_offset_mode,
                linear_multistart=args.linear_multistart,
                linear_multistart_threshold_mm=linear_multistart_threshold_mm,
                linear_multistart_angle_deg=args.linear_multistart_angle_deg,
                nonlinear_refine=args.nonlinear_refine,
                nonlinear_plane_mode=args.nonlinear_plane_mode,
                nonlinear_max_nfev=args.nonlinear_max_nfev,
                nonlinear_ftol=args.nonlinear_ftol,
                nonlinear_xtol=args.nonlinear_xtol,
                nonlinear_gtol=args.nonlinear_gtol,
                nonlinear_loss=args.nonlinear_loss,
                nonlinear_f_scale_mm=args.nonlinear_f_scale_mm,
                radius_mm=args.radius_mm,
                check_reachability=args.check_reachability,
                plane_tilt_min_deg=args.plane_tilt_min_deg,
                plane_tilt_max_deg=args.plane_tilt_max_deg,
                plane_yaw_min_deg=args.plane_yaw_min_deg,
                plane_yaw_max_deg=args.plane_yaw_max_deg,
                plane_center_xy_range_mm=args.plane_center_xy_range_mm,
                plane_center_z_range_mm=plane_center_z_range_mm,
                scan_params=scan_params,
                pose_geometry=args.pose_geometry,
                reference_scan_params=reference_scan_params,
                reference_scan_count=args.reference_scans,
                debug_diagnostics=(
                    args.debug_diagnostics
                    and (args.debug_all_trials or system_idx == args.debug_trial)
                ),
                debug_known_baseline=not args.no_debug_known_baseline,
                fix_true_plane_offset=args.fix_true_plane_offset,
                gt_rng=gt_rng,
                init_rng=init_rng,
            )
            results.append(result)
            if args.verbose and result.linear_multistart_used:
                print(
                    f"[trial {system_idx:04d}] linear multistart selected | "
                    f"starts={result.linear_start_count} | "
                    "final_plane_rms="
                    f"{result.final_linear_plane_rms_mm:.6g} mm"
                )
        except Exception as exc:
            failures += 1
            if first_failure is None:
                first_failure = f"{type(exc).__name__}: {exc}"
            if args.verbose or args.debug_diagnostics:
                print(
                    f"[trial {system_idx:04d}] failed | "
                    f"{type(exc).__name__}: {exc}"
                )
            if args.debug_traceback:
                traceback.print_exc()

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
                    f"[single-plane-optimal] progress {system_idx + 1}/{args.systems} | "
                    f"ok={len(results)} | fail={failures} | "
                    f"conv={converged}/{len(results)} | "
                    f"paper_success={succeeded}/{len(results)} | "
                    f"median_t={median_translation_error:.6g} mm | "
                    f"elapsed={elapsed:.1f}s"
                )
            else:
                print(
                    f"[single-plane-optimal] progress {system_idx + 1}/{args.systems} | "
                    f"ok=0 | fail={failures} | elapsed={elapsed:.1f}s"
                )

    if args.verbose:
        elapsed = time.perf_counter() - start_time
        print(
            f"[single-plane-optimal] finished | ok={len(results)} | "
            f"failed={failures} | elapsed={elapsed:.1f}s"
        )

    print_calibration_summary(results)
    if not results:
        if first_failure is not None:
            print(f"first failure: {first_failure}")
        raise SystemExit(2)

    save_results_csv(results, args.csv)
    print(f"saved: {args.csv}")

    finite_gauge_angles = np.asarray(
        [
            result.gauge_axis_angle_deg
            for result in results
            if np.isfinite(result.gauge_axis_angle_deg)
        ],
        dtype=float,
    )
    if len(finite_gauge_angles):
        parallel = np.asarray(
            [abs(result.gauge_parallel_error_mm) for result in results],
            dtype=float,
        )
        perpendicular = np.asarray(
            [result.gauge_perpendicular_error_mm for result in results],
            dtype=float,
        )
        print(
            "translation error vs true sensor-Z gauge: "
            f"median |parallel|={np.median(parallel):.6g} mm, "
            f"median perpendicular={np.median(perpendicular):.6g} mm, "
            f"median axis angle={np.median(finite_gauge_angles):.6g} deg, "
            f"max axis angle={np.max(finite_gauge_angles):.6g} deg"
        )

    if args.no_plots:
        return

    plot_dir = (
        args.plot_dir
        if args.plot_dir is not None
        else args.csv.parent / f"{args.csv.stem}_plots"
    )
    for path in save_calibration_plots(results, plot_dir):
        print(f"saved plot: {path}")
    error_direction_path = save_translation_error_direction_plot(
        results,
        plot_dir / "translation_error_vs_sensor_z_gauge.png",
    )
    if error_direction_path is not None:
        print(f"saved plot: {error_direction_path}")


if __name__ == "__main__":
    main()
