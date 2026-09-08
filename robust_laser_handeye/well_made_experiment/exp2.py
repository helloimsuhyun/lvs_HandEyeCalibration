"""
Generic paper-style pose converter/generator.

Run
---
cd ~/lvs_HandEyeCalibration
PYTHONPATH=. python \
  robust_laser_handeye/well_made_experiment/exp2_extensible.py \
  --radius-mm 100

Core mapping
------------
For every paper pose (d, theta, beta_paper) attached to a radial line psi:

    user tilt = paper theta

    cos(gamma) = cot(theta) * cot(beta_paper)

    gamma = atan2(
        branch * sqrt(1 - cos(gamma)^2),
        cos(gamma)
    )

where gamma is normal_azimuth_sensor_deg.

The existing project rule is then reused unchanged so that the physical
laser scan line remains radial:

    delta = atan2(sin(gamma), cos(gamma) / cos(tilt))
    azimuth_deg = psi - delta

The target point is created directly from the requested radius:

    u = radius * cos(psi)
    v = radius * sin(psi)

Nothing in the generic converter assumes 81 poses.
In main(), edit the distances/thetas passed to
`make_theta_adaptive_pose_grid(...)`.  For each theta, the paper beta set
is generated automatically as {90-theta, 90, 90+theta}.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from itertools import product
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np

from robust_laser_handeye.laser_handeye.data import PlaneFrame
from robust_laser_handeye.laser_handeye.simulation import simulate_scan_from_sensor_pose
from robust_laser_handeye.laser_handeye.pose_design import sensor_pose_from_target_point
from robust_laser_handeye.laser_handeye.initialization import make_initial_guess
from robust_laser_handeye.laser_handeye.scene_generation import make_plane_from_point, plane_basis
from robust_laser_handeye.laser_handeye.calibration import calibrate_single_plane_with_nonlinear
from robust_laser_handeye.well_made_experiment.visualization import plot_single_plane_scans
from robust_laser_handeye.well_made_experiment.jacobian_analysis import analyze_calibration_jacobian


# =============================================================================
# Generic input / output data structures
# =============================================================================

@dataclass(frozen=True)
class PaperPoseSpec:
    """One paper-style pose tuple attached to a target line."""

    distance_mm: float
    theta_deg: float
    beta_deg: float


@dataclass(frozen=True)
class PaperLineSpec:
    """One radial target line and all paper poses executed on that line."""

    line_id: int
    scanline_azimuth_deg: float
    branch_sign: float
    poses: tuple[PaperPoseSpec, ...]


@dataclass(frozen=True)
class ScanPoseParameters:
    """Converted project-style pose parameters."""

    scan_id: int
    line_id: int
    line_pose_id: int
    branch_sign: float

    # Original paper values
    paper_distance_mm: float
    paper_theta_deg: float
    paper_beta_deg: float

    # Project values
    target_u_mm: float
    target_v_mm: float
    distance_mm: float
    tilt_deg: float
    azimuth_deg: float
    normal_azimuth_sensor_deg: float
    scanline_azimuth_deg: float


# =============================================================================
# Scene / experiment configuration
# =============================================================================

BOARD_CENTER = np.array([0.0, 0.0, 500.0], dtype=float)
PLANE_NORMAL = np.array([0.2, 0.1, 1.0], dtype=float)

T_EF_S_TRUE = np.array([
    [0.813797681349, -0.543838142482, -0.204874128703, 50.0],
    [0.469846310393,  0.823172944646, -0.318795777597, -100.0],
    [0.342020143326,  0.163175911167,  0.925416578398, 80.0],
    [0.0,             0.0,             0.0,            1.0],
], dtype=float)

PROFILE_HALF_WIDTH_MM = 25.0
PROFILE_POINT_COUNT = 100
X_VALUES = np.linspace(-PROFILE_HALF_WIDTH_MM, PROFILE_HALF_WIDTH_MM, PROFILE_POINT_COUNT)

DEFAULT_RADIUS_MM = 100.0
NOISE_STD_MM = 0.05
INIT_SEED = 0
NOISE_SEED = 5678
MAX_INIT_ROT_ERROR_DEG = 30.0
MAX_INIT_TRANS_ERROR_MM = 200.0


# =============================================================================
# Existing experiment helpers
# =============================================================================

def make_initial_guess_GT(
    T_true: np.ndarray,
    rng: np.random.Generator | None = None,
    max_rotation_error_deg: float = MAX_INIT_ROT_ERROR_DEG,
    max_translation_error_mm: float = MAX_INIT_TRANS_ERROR_MM,
) -> np.ndarray:
    T_true = np.asarray(T_true, dtype=float).reshape(4, 4)
    return make_initial_guess(
        reference_angles_deg=None,
        reference_translation_mm=T_true[:3, 3],
        reference_rotation=T_true[:3, :3],
        rng=rng,
        mode="carlson",
        angle_range_deg=max_rotation_error_deg,
        translation_range_mm=max_translation_error_mm,
        rotation_perturbation="axis_angle",
        translation_perturbation="direction_norm",
    )


def make_plane_frame(normal: np.ndarray, board_center: np.ndarray) -> PlaneFrame:
    plane_n, plane_l = make_plane_from_point(
        normal=normal,
        point_on_plane_mm=board_center,
    )
    u, v = plane_basis(plane_n)
    return PlaneFrame(u=u, v=v, n=plane_n, offset_mm=plane_l)


def rotation_error_deg(R_est: np.ndarray, R_true: np.ndarray) -> float:
    R_delta = (
        np.asarray(R_est, dtype=float).reshape(3, 3)
        @ np.asarray(R_true, dtype=float).reshape(3, 3).T
    )
    c = float(np.clip((np.trace(R_delta) - 1.0) / 2.0, -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


# =============================================================================
# Generic paper -> project conversion
# =============================================================================

def normalize_deg(angle_deg: float) -> float:
    return float(float(angle_deg) % 360.0)


def target_uv_from_radius_and_line(
    *,
    radius_mm: float,
    scanline_azimuth_deg: float,
) -> tuple[float, float]:
    """Place the target point on a circle of radius_mm at line direction psi."""

    radius_mm = float(radius_mm)
    if not np.isfinite(radius_mm) or radius_mm <= 0.0:
        raise ValueError("radius_mm must be positive and finite")

    psi = np.deg2rad(float(scanline_azimuth_deg))
    return (
        float(radius_mm * np.cos(psi)),
        float(radius_mm * np.sin(psi)),
    )


def paper_beta_to_normal_azimuth(
    *,
    theta_deg: float,
    paper_beta_deg: float,
    branch_sign: float,
) -> float:
    """Convert paper beta to normal_azimuth_sensor_deg.

    Mapping:
        cos(gamma) = cot(theta) * cot(beta_paper)

    branch_sign chooses the sign of sin(gamma).
    """

    theta_deg = float(theta_deg)
    paper_beta_deg = float(paper_beta_deg)
    branch_sign = float(branch_sign)

    if branch_sign not in (-1.0, 1.0):
        raise ValueError("branch_sign must be +1 or -1")

    theta = np.deg2rad(theta_deg)
    beta = np.deg2rad(paper_beta_deg)

    sin_theta = float(np.sin(theta))
    sin_beta = float(np.sin(beta))

    if abs(sin_theta) <= 1e-12:
        raise ValueError("theta is too close to 0 deg: view azimuth is degenerate")
    if abs(sin_beta) <= 1e-12:
        raise ValueError("paper beta must not be 0 or 180 deg")

    c = float(
        (np.cos(theta) / sin_theta)
        * (np.cos(beta) / sin_beta)
    )

    if c < -1.0 - 1e-10 or c > 1.0 + 1e-10:
        raise ValueError(
            "infeasible paper (theta, beta) for this mapping: "
            f"theta={theta_deg:.6g} deg, beta={paper_beta_deg:.6g} deg, "
            f"cot(theta)*cot(beta)={c:.12g}"
        )

    c = float(np.clip(c, -1.0, 1.0))
    s = float(branch_sign * np.sqrt(max(0.0, 1.0 - c * c)))

    return float(np.degrees(np.arctan2(s, c)) % 360.0)


def azimuth_for_scanline_direction(
    *,
    scanline_azimuth_deg: float,
    tilt_deg: float,
    normal_azimuth_sensor_deg: float,
) -> float:
    """Reuse the existing radial physical scan-line compensation unchanged."""

    psi = np.deg2rad(float(scanline_azimuth_deg))
    tilt = np.deg2rad(float(tilt_deg))
    beta_user = np.deg2rad(float(normal_azimuth_sensor_deg))

    cos_tilt = float(np.cos(tilt))
    if abs(cos_tilt) <= 1e-12:
        raise ValueError("tilt too close to 90 deg for radial azimuth compensation")

    delta = float(np.arctan2(
        np.sin(beta_user),
        np.cos(beta_user) / cos_tilt,
    ))

    return float(np.degrees(psi - delta) % 360.0)


def physical_scanline_azimuth_deg(
    *,
    azimuth_deg: float,
    tilt_deg: float,
    normal_azimuth_sensor_deg: float,
) -> float:
    """Inverse check for azimuth_for_scanline_direction()."""

    alpha = np.deg2rad(float(azimuth_deg))
    tilt = np.deg2rad(float(tilt_deg))
    beta_user = np.deg2rad(float(normal_azimuth_sensor_deg))

    delta = float(np.arctan2(
        np.sin(beta_user),
        np.cos(beta_user) / np.cos(tilt),
    ))

    return float(np.degrees(alpha + delta) % 360.0)


def convert_one_paper_pose(
    *,
    scan_id: int,
    line_spec: PaperLineSpec,
    line_pose_id: int,
    paper_pose: PaperPoseSpec,
    radius_mm: float,
) -> ScanPoseParameters:
    """Convert one line + (d, theta, beta) into project pose parameters."""

    psi_deg = normalize_deg(line_spec.scanline_azimuth_deg)
    u_mm, v_mm = target_uv_from_radius_and_line(
        radius_mm=radius_mm,
        scanline_azimuth_deg=psi_deg,
    )

    # paper theta == project tilt
    tilt_deg = float(paper_pose.theta_deg)

    # paper beta -> project normal azimuth
    view_azimuth_deg = paper_beta_to_normal_azimuth(
        theta_deg=paper_pose.theta_deg,
        paper_beta_deg=paper_pose.beta_deg,
        branch_sign=line_spec.branch_sign,
    )

    # Keep the physical scan line radial using the existing project rule.
    azimuth_deg = azimuth_for_scanline_direction(
        scanline_azimuth_deg=psi_deg,
        tilt_deg=tilt_deg,
        normal_azimuth_sensor_deg=view_azimuth_deg,
    )

    # Sanity check: actual line direction must recover psi modulo 180 deg.
    recovered_psi = physical_scanline_azimuth_deg(
        azimuth_deg=azimuth_deg,
        tilt_deg=tilt_deg,
        normal_azimuth_sensor_deg=view_azimuth_deg,
    )
    line_error = ((recovered_psi - psi_deg + 90.0) % 180.0) - 90.0
    if abs(line_error) > 1e-8:
        raise RuntimeError(
            f"radial scan-line check failed on line {line_spec.line_id}: "
            f"{line_error:.3e} deg"
        )

    return ScanPoseParameters(
        scan_id=int(scan_id),
        line_id=int(line_spec.line_id),
        line_pose_id=int(line_pose_id),
        branch_sign=float(line_spec.branch_sign),
        paper_distance_mm=float(paper_pose.distance_mm),
        paper_theta_deg=float(paper_pose.theta_deg),
        paper_beta_deg=float(paper_pose.beta_deg),
        target_u_mm=float(u_mm),
        target_v_mm=float(v_mm),
        distance_mm=float(paper_pose.distance_mm),
        tilt_deg=float(tilt_deg),
        azimuth_deg=float(azimuth_deg),
        normal_azimuth_sensor_deg=float(view_azimuth_deg),
        scanline_azimuth_deg=float(psi_deg),
    )


def make_pose_parameters_from_paper_lines(
    *,
    line_specs: Sequence[PaperLineSpec],
    radius_mm: float,
) -> list[ScanPoseParameters]:
    """Convert every paper pose on every line.

    This is the generic entry point. It assumes nothing about line count,
    line spacing, or which d/theta/beta values each line contains.
    """

    if not line_specs:
        raise ValueError("line_specs must contain at least one line")

    seen_line_ids: set[int] = set()
    output: list[ScanPoseParameters] = []

    for line_spec in line_specs:
        if line_spec.line_id in seen_line_ids:
            raise ValueError(f"duplicate line_id: {line_spec.line_id}")
        seen_line_ids.add(line_spec.line_id)

        if line_spec.branch_sign not in (-1.0, 1.0):
            raise ValueError(
                f"line {line_spec.line_id}: branch_sign must be +1 or -1"
            )
        if not line_spec.poses:
            raise ValueError(f"line {line_spec.line_id}: poses must not be empty")

        for line_pose_id, paper_pose in enumerate(line_spec.poses):
            if not np.isfinite(paper_pose.distance_mm) or paper_pose.distance_mm <= 0.0:
                raise ValueError(
                    f"line {line_spec.line_id}, pose {line_pose_id}: "
                    "distance_mm must be positive and finite"
                )

            output.append(
                convert_one_paper_pose(
                    scan_id=len(output),
                    line_spec=line_spec,
                    line_pose_id=line_pose_id,
                    paper_pose=paper_pose,
                    radius_mm=radius_mm,
                )
            )

    return output


# =============================================================================
# Convenience builders
# =============================================================================

def make_pose_grid(
    *,
    distances_mm: Iterable[float],
    thetas_deg: Iterable[float],
    betas_deg: Iterable[float],
) -> tuple[PaperPoseSpec, ...]:
    """Cartesian product of arbitrary paper d/theta/beta values."""

    return tuple(
        PaperPoseSpec(
            distance_mm=float(d),
            theta_deg=float(theta),
            beta_deg=float(beta),
        )
        for d, theta, beta in product(distances_mm, thetas_deg, betas_deg)
    )


def make_theta_adaptive_pose_grid(
    *,
    distances_mm: Iterable[float],
    thetas_deg: Iterable[float],
) -> tuple[PaperPoseSpec, ...]:
    """Build a feasible three-beta paper pose set for every theta.

    For each paper theta, choose

        beta = {90-theta, 90, 90+theta}

    which maps to the project's canonical view-azimuth set

        gamma = {0, 90/270, 180} deg

    depending on mirror branch.

    This avoids infeasible Cartesian products such as
    theta=5 deg with beta=60 deg.
    """

    poses: list[PaperPoseSpec] = []

    distances = tuple(
        float(d)
        for d in distances_mm
    )

    thetas = tuple(
        float(theta)
        for theta in thetas_deg
    )

    if not distances:
        raise ValueError(
            "distances_mm must not be empty"
        )

    if not thetas:
        raise ValueError(
            "thetas_deg must not be empty"
        )

    for theta in thetas:

        if (
            not np.isfinite(theta)
            or theta <= 0.0
            or theta >= 90.0
        ):
            raise ValueError(
                "theta must lie in (0, 90) deg for the adaptive beta mapping"
            )

        betas = (
            90.0 - theta,
            90.0,
            90.0 + theta,
        )

        poses.extend(
            make_pose_grid(
                distances_mm=distances,
                thetas_deg=(theta,),
                betas_deg=betas,
            )
        )

    return tuple(
        poses
    )


def branch_sign_for_line(*, line_index: int, branch_mode: str) -> float:
    if branch_mode == "positive":
        return 1.0
    if branch_mode == "negative":
        return -1.0
    if branch_mode == "alternating":
        return 1.0 if int(line_index) % 2 == 0 else -1.0
    raise ValueError("branch_mode must be 'alternating', 'positive', or 'negative'")


def make_uniform_radial_line_specs(
    *,
    line_count: int,
    paper_poses: Sequence[PaperPoseSpec],
    start_azimuth_deg: float = 0.0,
    line_step_deg: float | None = None,
    branch_mode: str = "alternating",
) -> list[PaperLineSpec]:
    """Equally spaced radial lines that share the same paper pose list."""

    line_count = int(line_count)
    if line_count < 1:
        raise ValueError("line_count must be >= 1")
    if not paper_poses:
        raise ValueError("paper_poses must not be empty")

    if line_step_deg is None:
        line_step_deg = 360.0 / float(line_count)

    return [
        PaperLineSpec(
            line_id=line_id,
            scanline_azimuth_deg=normalize_deg(
                float(start_azimuth_deg) + float(line_step_deg) * line_id
            ),
            branch_sign=branch_sign_for_line(
                line_index=line_id,
                branch_mode=branch_mode,
            ),
            poses=tuple(paper_poses),
        )
        for line_id in range(line_count)
    ]


def make_paper81_line_specs(
    *,
    branch_mode: str = "alternating",
) -> list[PaperLineSpec]:
    """Paper-style radial preset using theta-adaptive feasible beta values.

    With theta=(5, 30) deg and three distances:
        3 d x 2 theta x 3 adaptive beta = 18 poses/line
        9 lines -> 162 poses

    The function name is kept for backward compatibility.
    """

    paper_poses = make_theta_adaptive_pose_grid(
        distances_mm=(60.0, 90.0, 120.0),
        thetas_deg=(5.0, 30.0),
    )

    return make_uniform_radial_line_specs(
        line_count=9,
        line_step_deg=40.0,
        start_azimuth_deg=0.0,
        paper_poses=paper_poses,
        branch_mode=branch_mode,
    )


# =============================================================================
# Sensor pose / scan generation
# =============================================================================

def target_point_base_from_uv(
    *,
    u_mm: float,
    v_mm: float,
    frame: PlaneFrame,
    board_center: np.ndarray = BOARD_CENTER,
) -> np.ndarray:
    return (
        np.asarray(board_center, dtype=float).reshape(3)
        + float(u_mm) * np.asarray(frame.u, dtype=float).reshape(3)
        + float(v_mm) * np.asarray(frame.v, dtype=float).reshape(3)
    )


def make_sensor_poses(
    *,
    pose_params: Sequence[ScanPoseParameters],
    frame: PlaneFrame,
    board_center: np.ndarray = BOARD_CENTER,
) -> list[np.ndarray]:
    """Generate T_base_sensor for every converted pose."""

    sensor_poses: list[np.ndarray] = []

    for p in pose_params:
        target_point_base = target_point_base_from_uv(
            u_mm=p.target_u_mm,
            v_mm=p.target_v_mm,
            frame=frame,
            board_center=board_center,
        )

        sensor_poses.append(
            sensor_pose_from_target_point(
                target_point_base_mm=target_point_base,
                frame=frame,
                distance_mm=p.distance_mm,
                tilt_deg=p.tilt_deg,
                azimuth_deg=p.azimuth_deg,
                normal_azimuth_sensor_deg=p.normal_azimuth_sensor_deg,
            )
        )

    return sensor_poses


def make_scans(
    *,
    pose_params: Sequence[ScanPoseParameters],
    sensor_poses: Sequence[np.ndarray],
    frame: PlaneFrame,
    noise_seed: int = NOISE_SEED,
    noise_std_mm: float = NOISE_STD_MM,
    T_ef_s_true: np.ndarray = T_EF_S_TRUE,
) -> list:
    rng = np.random.default_rng(noise_seed)
    scans = []

    for p, T_base_s in zip(pose_params, sensor_poses):
        scans.append(
            simulate_scan_from_sensor_pose(
                T_base_s=T_base_s,
                T_ef_s_true=T_ef_s_true,
                frame=frame,
                x_values=X_VALUES,
                noise_std=noise_std_mm,
                rng=rng,
                plane_id=0,
                scan_id=p.scan_id,
            )
        )

    return scans


# =============================================================================
# Printing / experiment
# =============================================================================

def print_pose_parameters(
    *,
    label: str,
    pose_params: Sequence[ScanPoseParameters],
) -> None:
    print("\n" + "=" * 145)
    print(label)
    print("=" * 145)
    print(
        " id | line | br | paper(d,theta,beta) |"
        "      u       v    | tilt | view az | radial psi | pose azimuth"
    )
    print("-" * 145)

    for p in pose_params:
        print(
            f"{p.scan_id:3d} | "
            f"{p.line_id:4d} | "
            f"{p.branch_sign:+2.0f} | "
            f"({p.paper_distance_mm:5.0f},"
            f"{p.paper_theta_deg:5.1f},"
            f"{p.paper_beta_deg:5.1f}) | "
            f"{p.target_u_mm:7.2f} "
            f"{p.target_v_mm:7.2f} | "
            f"{p.tilt_deg:5.1f} | "
            f"{p.normal_azimuth_sensor_deg:7.1f} | "
            f"{p.scanline_azimuth_deg:10.1f} | "
            f"{p.azimuth_deg:12.1f}"
        )


def run_design(
    *,
    label: str,
    pose_params: Sequence[ScanPoseParameters],
    frame: PlaneFrame,
    T_init: np.ndarray,
) -> None:
    sensor_poses = make_sensor_poses(
        pose_params=pose_params,
        frame=frame,
    )

    print_pose_parameters(
        label=label,
        pose_params=pose_params,
    )

    print("\n" + "#" * 88)
    print(f"FIM: {label}")
    print("#" * 88)

    analyze_calibration_jacobian(
        sensor_poses=sensor_poses,
        frame=frame,
        T_ef_s_true=T_EF_S_TRUE,
        T_eval=T_EF_S_TRUE,
        x_values=X_VALUES,
        rotation_characteristic_length_mm=100.0,
        label=label,
    )

    scans = make_scans(
        pose_params=pose_params,
        sensor_poses=sensor_poses,
        frame=frame,
    )

    try:
        _linear_result, nonlinear_result = calibrate_single_plane_with_nonlinear(
            scans,
            T_init=T_init.copy(),
            plane_offset_mode="joint",
            plane_mode="refit",
            max_iter=200,
            nonlinear_max_nfev=300,
        )

        T_est = np.asarray(nonlinear_result.T_ef_s, dtype=float)
        translation_error = float(
            np.linalg.norm(T_est[:3, 3] - T_EF_S_TRUE[:3, 3])
        )
        rot_error = rotation_error_deg(
            T_est[:3, :3],
            T_EF_S_TRUE[:3, :3],
        )

        print("\n" + "-" * 72)
        print(f"CALIBRATION RESULT: {label}")
        print("-" * 72)
        print("translation error [mm] :", translation_error)
        print("rotation error [deg]   :", rot_error)
        print("final RMS [mm]         :", nonlinear_result.final_rms_mm)

    except Exception as exc:
        print("\n" + "!" * 72)
        print(f"Calibration failed: {label}")
        print(repr(exc))
        print("!" * 72)

    plot_single_plane_scans(
        frame=frame,
        board_center=BOARD_CENTER,
        sensor_poses=sensor_poses,
        scans=scans,
    )


# =============================================================================
# CLI / main
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert arbitrary paper-style line poses (d, theta, beta) "
            "into project pose parameters."
        )
    )
    parser.add_argument(
        "--radius-mm",
        type=float,
        default=DEFAULT_RADIUS_MM,
        help="Radius [mm] used directly for target (u,v). Default: %(default)s",
    )
    parser.add_argument(
        "--branch-mode",
        choices=("alternating", "positive", "negative"),
        default="alternating",
        help="Mirror branch policy applied to the configured radial lines.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    frame = make_plane_frame(
        normal=PLANE_NORMAL,
        board_center=BOARD_CENTER,
    )

    # =====================================================================
    # DESIGN INPUT
    # =====================================================================
    # EDIT THIS BLOCK to change the paper-style pose combinations.
    #
    # Every Cartesian-product combination of:
    #     distance x theta x beta
    # is generated, converted to the project's pose parameterization,
    # and repeated on every radial line configured below.
    #
    # Example:
    #   3 distances x 1 theta x 3 betas = 9 poses / line
    #   9 lines -> 81 total poses
    #
    # If changed to:
    #   2 distances x 3 thetas x 3 betas = 18 poses / line
    # the same 9 lines automatically produce 162 poses.
    # =====================================================================

    # For every theta, beta is generated automatically as:
    #
    #     beta = {90-theta, 90, 90+theta}
    #
    # so all paper (theta, beta) pairs remain geometrically feasible and
    # convert to the same canonical view-azimuth support
    # {0, 90/270, 180} deg.
    paper_poses = make_theta_adaptive_pose_grid(
        distances_mm=(60.0, 90.0, 120.0),
        thetas_deg=(5.0, 30.0),
    )

    # =====================================================================
    # LINE LAYOUT
    # =====================================================================
    # Keep this unchanged if you only want to vary (d, theta, beta).
    #
    # Paper layout:
    #     9 radial lines, 40 deg apart.
    #
    # Radius is supplied independently through --radius-mm.
    # =====================================================================

    line_specs = make_uniform_radial_line_specs(
        line_count=9,
        line_step_deg=40.0,
        start_azimuth_deg=0.0,
        paper_poses=paper_poses,
        branch_mode=args.branch_mode,
    )

    # Generic conversion.
    # There is intentionally NO fixed 81-pose assertion here.
    pose_params = make_pose_parameters_from_paper_lines(
        line_specs=line_specs,
        radius_mm=args.radius_mm,
    )

    init_rng = np.random.default_rng(INIT_SEED)
    T_init = make_initial_guess_GT(
        T_true=T_EF_S_TRUE,
        rng=init_rng,
    )

    print("\n" + "=" * 88)
    print("GENERIC PAPER -> PROJECT POSE CONVERTER")
    print("=" * 88)
    print("radius [mm] =", args.radius_mm)
    print("lines       =", len(line_specs))
    print("poses       =", len(pose_params))
    print("branch mode =", args.branch_mode)
    print("poses/line  =", len(paper_poses))
    print("expected total =", len(line_specs) * len(paper_poses))

    run_design(
        label="PAPER_81_CONVERTED",
        pose_params=pose_params,
        frame=frame,
        T_init=T_init,
    )

    plt.show()


if __name__ == "__main__":
    main()