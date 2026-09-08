"""
Run
----
cd ~/lvs_HandEyeCalibration

PYTHONPATH=. python \
    robust_laser_handeye/well_made_experiment/exp1_optimal.py


Experiment
----------
12-pose Jacobian-derived GLOBAL_OPTIMAL design.

Geometry:
    - 12 circular (u, v) scan centers
    - radius = 100 mm
    - theta in {5, 40} deg
    - distance in {60, 120} mm
    - beta in {0, 120, 240} deg
    - every (theta, distance) support contains beta={0,120,240} once
    - physical laser scan line remains radial at every (u, v)

What is optimized:
    - the assignment of (theta, distance, beta) tuples to the 12 circular
      (u, v) locations.

Optimal support assignment by radial position psi:
    psi [deg] :   0  30  60  90 120 150 180 210 240 270 300 330
    support   :   0   1   0   3   3   3   1   0   1   2   2   2

where:
    support 0 = (theta=5 deg,  d=60 mm)
    support 1 = (theta=5 deg,  d=120 mm)
    support 2 = (theta=40 deg, d=60 mm)
    support 3 = (theta=40 deg, d=120 mm)

Optimal beta assignment:
    psi [deg] :   0  30  60  90 120 150 180 210 240 270 300 330
    beta[deg] :   0 120 240   0 240 120   0 120 240   0 240 120

IMPORTANT
---------
Ordinary azimuth alpha is compensated for beta and tilt so that the ACTUAL
laser scan line on the calibration plane remains radial.

Desired physical scan-line direction:

    psi_i = atan2(v_i, u_i)

Actual scan-line direction:

    psi_i = alpha_i + delta_i

where

    delta_i = atan2(
        sin(beta_i),
        cos(beta_i) / cos(theta_i)
    )

Therefore

    alpha_i = psi_i - delta_i
"""


from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from robust_laser_handeye.laser_handeye.data import PlaneFrame

from robust_laser_handeye.laser_handeye.simulation import (
    simulate_scan_from_sensor_pose,
)

from robust_laser_handeye.laser_handeye.pose_design import (
    circular_uv_points,
    plane_uv_to_base_points,
    sensor_pose_from_target_point,
)

from robust_laser_handeye.laser_handeye.initialization import (
    make_initial_guess,
)

from robust_laser_handeye.laser_handeye.scene_generation import (
    make_plane_from_point,
    plane_basis,
)

from robust_laser_handeye.laser_handeye.calibration import (
    calibrate_single_plane_with_nonlinear,
)

from robust_laser_handeye.well_made_experiment.visualization import (
    plot_single_plane_scans,
)

from robust_laser_handeye.well_made_experiment.jacobian_analysis import (
    analyze_calibration_jacobian,
)


# =============================================================================
# Data class
# =============================================================================

@dataclass(frozen=True)
class ScanPoseParameters:

    scan_id: int

    target_u_mm: float
    target_v_mm: float

    distance_mm: float
    tilt_deg: float

    # ordinary azimuth alpha
    azimuth_deg: float

    # view azimuth beta
    normal_azimuth_sensor_deg: float

    # desired physical scan-line direction on the plane
    scanline_azimuth_deg: float

    # support index:
    #
    # 0 = (5, 60)
    # 1 = (5, 120)
    # 2 = (40, 60)
    # 3 = (40, 120)
    support_id: int


# =============================================================================
# Scene configuration
# =============================================================================

BOARD_CENTER = np.array(
    [0.0, 0.0, 500.0],
    dtype=float,
)

PLANE_NORMAL = np.array(
    [0.2, 0.1, 1.0],
    dtype=float,
)


T_EF_S_TRUE = np.array([
    [
        0.813797681349,
        -0.543838142482,
        -0.204874128703,
        50.0,
    ],
    [
        0.469846310393,
        0.823172944646,
        -0.318795777597,
        -100.0,
    ],
    [
        0.342020143326,
        0.163175911167,
        0.925416578398,
        80.0,
    ],
    [
        0.0,
        0.0,
        0.0,
        1.0,
    ],
], dtype=float)


# =============================================================================
# Laser profile
# =============================================================================

PROFILE_HALF_WIDTH_MM = 25.0
PROFILE_POINT_COUNT = 100

X_VALUES = np.linspace(
    -PROFILE_HALF_WIDTH_MM,
    PROFILE_HALF_WIDTH_MM,
    PROFILE_POINT_COUNT,
)


# =============================================================================
# Initial hand-eye
# =============================================================================

MAX_INIT_ROT_ERROR_DEG = 30.0
MAX_INIT_TRANS_ERROR_MM = 200.0


# =============================================================================
# Calibration pattern
# =============================================================================

N = 12

RADIUS_MM = 100.0


# Full factorial:
#
# theta ∈ {5, 40}
# d     ∈ {60, 120}
#
POSE_SUPPORTS = (
    (5.0, 60.0),
    (5.0, 120.0),
    (40.0, 60.0),
    (40.0, 120.0),
)


# =============================================================================
# Noise
# =============================================================================

NOISE_STD_MM = 0.05


# =============================================================================
# Seeds
# =============================================================================

INIT_SEED = 0
NOISE_SEED = 5678


# =============================================================================
# Initial guess
# =============================================================================

def make_initial_guess_GT(
    T_true: np.ndarray,
    rng: np.random.Generator | None = None,
    max_rotation_error_deg: float = 30.0,
    max_translation_error_mm: float = 200.0,
) -> np.ndarray:

    T_true = np.asarray(
        T_true,
        dtype=float,
    ).reshape(4, 4)

    return make_initial_guess(
        reference_angles_deg=None,

        reference_translation_mm=(
            T_true[:3, 3]
        ),

        reference_rotation=(
            T_true[:3, :3]
        ),

        rng=rng,

        mode="carlson",

        angle_range_deg=(
            max_rotation_error_deg
        ),

        translation_range_mm=(
            max_translation_error_mm
        ),

        rotation_perturbation="axis_angle",

        translation_perturbation="direction_norm",
    )


# =============================================================================
# Plane
# =============================================================================

def make_plane_frame(
    normal: np.ndarray,
    board_center: np.ndarray,
) -> PlaneFrame:

    plane_n, plane_l = (
        make_plane_from_point(
            normal=normal,
            point_on_plane_mm=board_center,
        )
    )

    u, v = plane_basis(
        plane_n
    )

    return PlaneFrame(
        u=u,
        v=v,
        n=plane_n,
        offset_mm=plane_l,
    )


# =============================================================================
# Desired physical scan-line direction
# =============================================================================

def radial_scanline_azimuth_from_uv(
    uv: np.ndarray,
) -> np.ndarray:
    """
    Desired PHYSICAL laser scan-line direction on the board.

    The line at each circular scan center is radial:

        psi_i = atan2(v_i, u_i)

    Since a line has no orientation sign, psi and psi + 180 deg
    describe the same physical line.
    """

    uv = np.asarray(
        uv,
        dtype=float,
    )

    if (
        uv.ndim != 2
        or uv.shape[1] != 2
    ):
        raise ValueError(
            "uv must have shape (N, 2)"
        )

    return (
        np.degrees(
            np.arctan2(
                uv[:, 1],
                uv[:, 0],
            )
        )
        % 360.0
    )


# =============================================================================
# Convert desired physical scan-line direction -> ordinary alpha
# =============================================================================

def alpha_for_scanline_direction(
    scanline_azimuth_deg: np.ndarray,
    tilt_deg: np.ndarray,
    beta_deg: np.ndarray,
) -> np.ndarray:
    """
    Compute ordinary azimuth alpha so that the actual scan line
    on the calibration plane has the desired direction psi.

    Plane scan-line vector:

        w =
            cos(beta)/cos(theta) * e(alpha)
            +
            sin(beta) * e_perp(alpha)

    Therefore the actual line direction is

        psi = alpha + delta

    with

        delta = atan2(
            sin(beta),
            cos(beta) / cos(theta)
        )

    Hence

        alpha = psi - delta
    """

    psi_deg = np.asarray(
        scanline_azimuth_deg,
        dtype=float,
    )

    tilt_deg = np.asarray(
        tilt_deg,
        dtype=float,
    )

    beta_deg = np.asarray(
        beta_deg,
        dtype=float,
    )

    if not (
        psi_deg.shape
        == tilt_deg.shape
        == beta_deg.shape
    ):
        raise ValueError(
            "psi, tilt and beta must have identical shapes"
        )

    theta = np.deg2rad(
        tilt_deg
    )

    beta = np.deg2rad(
        beta_deg
    )

    delta_deg = np.degrees(
        np.arctan2(
            np.sin(beta),
            np.cos(beta) / np.cos(theta),
        )
    )

    alpha_deg = (
        psi_deg
        - delta_deg
    ) % 360.0

    return alpha_deg


# =============================================================================
# Check actual scan-line direction
# =============================================================================

def physical_scanline_azimuth_deg(
    alpha_deg: np.ndarray,
    tilt_deg: np.ndarray,
    beta_deg: np.ndarray,
) -> np.ndarray:
    """
    Reconstruct the actual physical scan-line angle from
    alpha, theta and beta.

    Useful as a sanity check.
    """

    alpha_deg = np.asarray(
        alpha_deg,
        dtype=float,
    )

    tilt_deg = np.asarray(
        tilt_deg,
        dtype=float,
    )

    beta_deg = np.asarray(
        beta_deg,
        dtype=float,
    )

    theta = np.deg2rad(
        tilt_deg
    )

    beta = np.deg2rad(
        beta_deg
    )

    delta_deg = np.degrees(
        np.arctan2(
            np.sin(beta),
            np.cos(beta) / np.cos(theta),
        )
    )

    return (
        alpha_deg
        + delta_deg
    ) % 360.0


# =============================================================================
# Global-optimal support assignment
# =============================================================================

def make_support_ids() -> np.ndarray:
    """
    Global-optimal assignment of the four (tilt, distance) supports
    to the 12 circular scan centers.

    Circular points:
        psi(radial) = 0, 30, 60, ..., 330 deg

    support:
        0 = (5 deg,  60 mm)
        1 = (5 deg, 120 mm)
        2 = (40 deg, 60 mm)
        3 = (40 deg,120 mm)

    Optimal assignment:
        psi [deg] :   0  30  60  90 120 150 180 210 240 270 300 330
        support   :   0   1   0   3   3   3   1   0   1   2   2   2
    """

    if N != 12:
        raise ValueError(
            "This global-optimal design assumes N=12."
        )

    return np.array(
        [
            0,  #   0 deg
            1,  #  30 deg
            0,  #  60 deg
            3,  #  90 deg
            3,  # 120 deg
            3,  # 150 deg
            1,  # 180 deg
            0,  # 210 deg
            1,  # 240 deg
            2,  # 270 deg
            2,  # 300 deg
            2,  # 330 deg
        ],
        dtype=int,
    )


# =============================================================================
# Global-optimal view azimuth beta
# =============================================================================

def make_optimal_beta(
    support_ids: np.ndarray,
) -> np.ndarray:
    """
    Global-optimal beta-to-(u,v) assignment.

    Optimal assignment:
        psi [deg] :   0  30  60  90 120 150 180 210 240 270 300 330
        beta[deg] :   0 120 240   0 240 120   0 120 240   0 240 120

    Every individual (tilt, distance) support still contains
    beta = {0, 120, 240} exactly once.
    """

    support_ids = np.asarray(
        support_ids,
        dtype=int,
    ).reshape(-1)

    if len(support_ids) != 12:
        raise ValueError(
            "This global-optimal beta design assumes N=12."
        )

    expected_support_ids = make_support_ids()

    if not np.array_equal(
        support_ids,
        expected_support_ids,
    ):
        raise ValueError(
            "support_ids do not match the global-optimal support assignment."
        )

    beta = np.array(
        [
              0.0,  #   0 deg
            120.0,  #  30 deg
            240.0,  #  60 deg
              0.0,  #  90 deg
            240.0,  # 120 deg
            120.0,  # 150 deg
              0.0,  # 180 deg
            120.0,  # 210 deg
            240.0,  # 240 deg
              0.0,  # 270 deg
            240.0,  # 300 deg
            120.0,  # 330 deg
        ],
        dtype=float,
    )

    # Sanity check:
    # each of the four (tilt, distance) supports must contain
    # beta={0,120,240} exactly once.
    expected_beta_set = np.array(
        [0.0, 120.0, 240.0],
        dtype=float,
    )

    for support_id in range(
        len(POSE_SUPPORTS)
    ):
        support_beta = np.sort(
            beta[
                support_ids
                == support_id
            ]
        )

        if not np.allclose(
            support_beta,
            expected_beta_set,
        ):
            raise RuntimeError(
                f"support {support_id} does not have beta={{0,120,240}}."
            )

    return beta


# =============================================================================
# Pose parameters
# =============================================================================

def make_scan_pose_parameters(
    uv: np.ndarray,
    beta_deg: np.ndarray,
) -> list[ScanPoseParameters]:

    uv = np.asarray(
        uv,
        dtype=float,
    )

    beta_deg = np.asarray(
        beta_deg,
        dtype=float,
    ).reshape(N)

    if uv.shape != (N, 2):
        raise ValueError(
            f"uv must have shape ({N}, 2)"
        )


    # -------------------------------------------------------------------------
    # Global-optimal support
    # -------------------------------------------------------------------------

    support_ids = (
        make_support_ids()
    )


    # -------------------------------------------------------------------------
    # Extract tilt and distance for every scan
    # -------------------------------------------------------------------------

    tilt_deg_all = np.zeros(
        N,
        dtype=float,
    )

    distance_mm_all = np.zeros(
        N,
        dtype=float,
    )

    for scan_id in range(N):

        support_id = int(
            support_ids[
                scan_id
            ]
        )

        (
            tilt_deg,
            distance_mm,
        ) = POSE_SUPPORTS[
            support_id
        ]

        tilt_deg_all[
            scan_id
        ] = float(
            tilt_deg
        )

        distance_mm_all[
            scan_id
        ] = float(
            distance_mm
        )


    # -------------------------------------------------------------------------
    # Desired PHYSICAL scan-line direction
    #
    # Radial line at each (u,v).
    # -------------------------------------------------------------------------

    scanline_azimuths = (
        radial_scanline_azimuth_from_uv(
            uv
        )
    )


    # -------------------------------------------------------------------------
    # Compute alpha needed to preserve that scan-line direction.
    #
    # IMPORTANT:
    # alpha depends on beta.
    #
    # Alpha is computed from the optimal beta assignment so that
    # the PHYSICAL scan line remains radial.
    # -------------------------------------------------------------------------

    alpha_deg_all = (
        alpha_for_scanline_direction(
            scanline_azimuth_deg=(
                scanline_azimuths
            ),

            tilt_deg=(
                tilt_deg_all
            ),

            beta_deg=(
                beta_deg
            ),
        )
    )


    # -------------------------------------------------------------------------
    # Sanity check
    # -------------------------------------------------------------------------

    recovered_scanline = (
        physical_scanline_azimuth_deg(
            alpha_deg=(
                alpha_deg_all
            ),

            tilt_deg=(
                tilt_deg_all
            ),

            beta_deg=(
                beta_deg
            ),
        )
    )


    # A line is equivalent modulo 180 deg.
    line_error = (
        (
            recovered_scanline
            - scanline_azimuths
            + 90.0
        )
        % 180.0
        - 90.0
    )


    if np.max(
        np.abs(
            line_error
        )
    ) > 1e-8:

        raise RuntimeError(
            "Failed to construct radial physical scan lines."
        )


    # -------------------------------------------------------------------------
    # Build pose objects
    # -------------------------------------------------------------------------

    params: list[
        ScanPoseParameters
    ] = []


    for scan_id in range(N):

        support_id = int(
            support_ids[
                scan_id
            ]
        )


        params.append(
            ScanPoseParameters(
                scan_id=(
                    scan_id
                ),

                target_u_mm=float(
                    uv[
                        scan_id,
                        0,
                    ]
                ),

                target_v_mm=float(
                    uv[
                        scan_id,
                        1,
                    ]
                ),

                distance_mm=float(
                    distance_mm_all[
                        scan_id
                    ]
                ),

                tilt_deg=float(
                    tilt_deg_all[
                        scan_id
                    ]
                ),

                azimuth_deg=float(
                    alpha_deg_all[
                        scan_id
                    ]
                ),

                normal_azimuth_sensor_deg=float(
                    beta_deg[
                        scan_id
                    ]
                ),

                scanline_azimuth_deg=float(
                    scanline_azimuths[
                        scan_id
                    ]
                ),

                support_id=(
                    support_id
                ),
            )
        )

    return params


# =============================================================================
# Sensor poses
# =============================================================================

def make_sensor_poses(
    pose_params: list[ScanPoseParameters],
    target_points_base: np.ndarray,
    frame: PlaneFrame,
) -> list[np.ndarray]:

    sensor_poses = []

    for p in pose_params:

        T_base_s = (
            sensor_pose_from_target_point(
                target_point_base_mm=(
                    target_points_base[
                        p.scan_id
                    ]
                ),

                frame=frame,

                distance_mm=(
                    p.distance_mm
                ),

                tilt_deg=(
                    p.tilt_deg
                ),

                azimuth_deg=(
                    p.azimuth_deg
                ),

                normal_azimuth_sensor_deg=(
                    p.normal_azimuth_sensor_deg
                ),
            )
        )

        sensor_poses.append(
            T_base_s
        )

    return sensor_poses


# =============================================================================
# Scan simulation
# =============================================================================

def make_scans(
    pose_params: list[ScanPoseParameters],
    sensor_poses: list[np.ndarray],
    frame: PlaneFrame,
    noise_seed: int,
    noise_std_mm: float = NOISE_STD_MM,
    T_ef_s_true: np.ndarray = T_EF_S_TRUE,
) -> list:

    rng = np.random.default_rng(
        noise_seed
    )

    scans = []

    for p, T_base_s in zip(
        pose_params,
        sensor_poses,
    ):

        scan = (
            simulate_scan_from_sensor_pose(
                T_base_s=T_base_s,

                T_ef_s_true=(
                    T_ef_s_true
                ),

                frame=frame,

                x_values=(
                    X_VALUES
                ),

                noise_std=(
                    noise_std_mm
                ),

                rng=rng,

                plane_id=0,

                scan_id=(
                    p.scan_id
                ),
            )
        )

        scans.append(
            scan
        )

    return scans


# =============================================================================
# Printing
# =============================================================================

def print_pose_parameters(
    label: str,
    pose_params: list[ScanPoseParameters],
) -> None:

    print(
        "\n"
        + "=" * 108
    )

    print(
        label
    )

    print(
        "=" * 108
    )

    print(
        " id | sup |    u       v    | tilt |   d   |"
        " radial scan |  alpha |  beta"
    )

    print(
        "-" * 108
    )

    for p in pose_params:

        print(
            f"{p.scan_id:3d} | "
            f"{p.support_id:3d} | "
            f"{p.target_u_mm:7.2f} "
            f"{p.target_v_mm:7.2f} | "
            f"{p.tilt_deg:4.0f}° | "
            f"{p.distance_mm:5.0f} | "
            f"{p.scanline_azimuth_deg:11.1f} | "
            f"{p.azimuth_deg:6.1f} | "
            f"{p.normal_azimuth_sensor_deg:6.1f}"
        )


# =============================================================================
# Rotation error
# =============================================================================

def rotation_error_deg(
    R_est: np.ndarray,
    R_true: np.ndarray,
) -> float:

    R_delta = (
        R_est
        @ R_true.T
    )

    c = (
        np.trace(
            R_delta
        )
        - 1.0
    ) / 2.0

    c = np.clip(
        c,
        -1.0,
        1.0,
    )

    return float(
        np.degrees(
            np.arccos(
                c
            )
        )
    )


# =============================================================================
# One design experiment
# =============================================================================

def run_design(
    label: str,
    beta_deg: np.ndarray,
    uv: np.ndarray,
    target_points_base: np.ndarray,
    frame: PlaneFrame,
    T_init: np.ndarray,
) -> None:

    # -------------------------------------------------------------------------
    # Pose parameters
    # -------------------------------------------------------------------------

    pose_params = (
        make_scan_pose_parameters(
            uv=uv,
            beta_deg=beta_deg,
        )
    )


    # -------------------------------------------------------------------------
    # Sensor poses
    # -------------------------------------------------------------------------

    sensor_poses = (
        make_sensor_poses(
            pose_params=(
                pose_params
            ),

            target_points_base=(
                target_points_base
            ),

            frame=(
                frame
            ),
        )
    )


    # -------------------------------------------------------------------------
    # Print geometry
    # -------------------------------------------------------------------------

    print_pose_parameters(
        label=label,
        pose_params=pose_params,
    )


    # =========================================================================
    # FIM
    # =========================================================================

    print(
        "\n"
        + "#" * 88
    )

    print(
        f"FIM: {label}"
    )

    print(
        "#" * 88
    )


    analyze_calibration_jacobian(
        sensor_poses=(
            sensor_poses
        ),

        frame=(
            frame
        ),

        T_ef_s_true=(
            T_EF_S_TRUE
        ),

        T_eval=(
            T_EF_S_TRUE
        ),

        x_values=(
            X_VALUES
        ),

        rotation_characteristic_length_mm=(
            100.0
        ),

        label=(
            label
        ),
    )


    # =========================================================================
    # Actual calibration sanity check
    # =========================================================================

    scans = (
        make_scans(
            pose_params=(
                pose_params
            ),

            sensor_poses=(
                sensor_poses
            ),

            frame=(
                frame
            ),

            noise_seed=(
                NOISE_SEED
            ),
        )
    )


    try:

        (
            linear_result,
            nonlinear_result,
        ) = (
            calibrate_single_plane_with_nonlinear(
                scans,

                T_init=(
                    T_init.copy()
                ),

                plane_offset_mode=(
                    "joint"
                ),

                plane_mode=(
                    "refit"
                ),

                max_iter=(
                    200
                ),

                nonlinear_max_nfev=(
                    300
                ),
            )
        )


        T_est = np.asarray(
            nonlinear_result.T_ef_s,
            dtype=float,
        )


        translation_error = float(
            np.linalg.norm(
                T_est[
                    :3,
                    3,
                ]
                -
                T_EF_S_TRUE[
                    :3,
                    3,
                ]
            )
        )


        rotation_error = (
            rotation_error_deg(
                T_est[
                    :3,
                    :3,
                ],

                T_EF_S_TRUE[
                    :3,
                    :3,
                ],
            )
        )


        print(
            "\n"
            + "-" * 72
        )

        print(
            f"CALIBRATION RESULT: {label}"
        )

        print(
            "-" * 72
        )

        print(
            "translation error [mm] :",
            translation_error,
        )

        print(
            "rotation error [deg]   :",
            rotation_error,
        )

        print(
            "final RMS [mm]         :",
            nonlinear_result.final_rms_mm,
        )


    except Exception as exc:

        print(
            "\n"
            + "!" * 72
        )

        print(
            f"Calibration failed: {label}"
        )

        print(
            repr(
                exc
            )
        )

        print(
            "!" * 72
        )


    # =========================================================================
    # Visualization
    # =========================================================================

    plot_single_plane_scans(
        frame=(
            frame
        ),

        board_center=(
            BOARD_CENTER
        ),

        sensor_poses=(
            sensor_poses
        ),

        scans=(
            scans
        ),
    )


# =============================================================================
# Main
# =============================================================================

def main() -> None:

    # =========================================================================
    # Plane
    # =========================================================================

    frame = (
        make_plane_frame(
            normal=(
                PLANE_NORMAL
            ),

            board_center=(
                BOARD_CENTER
            ),
        )
    )


    # =========================================================================
    # Circular scan centers
    # =========================================================================

    uv = (
        circular_uv_points(
            radius_mm=(
                RADIUS_MM
            ),

            N=(
                N
            ),
        )
    )


    target_points_base = (
        plane_uv_to_base_points(
            uv=(
                uv
            ),

            target_center_base_mm=(
                BOARD_CENTER
            ),

            frame=(
                frame
            ),
        )
    )


    # =========================================================================
    # Initial hand-eye
    # =========================================================================

    init_rng = (
        np.random.default_rng(
            INIT_SEED
        )
    )


    T_init = (
        make_initial_guess_GT(
            T_true=(
                T_EF_S_TRUE
            ),

            rng=(
                init_rng
            ),

            max_rotation_error_deg=(
                MAX_INIT_ROT_ERROR_DEG
            ),

            max_translation_error_mm=(
                MAX_INIT_TRANS_ERROR_MM
            ),
        )
    )


    # =========================================================================
    # Global-optimal pose assignment
    # =========================================================================

    support_ids = (
        make_support_ids()
    )

    beta_optimal = (
        make_optimal_beta(
            support_ids
        )
    )


    # =========================================================================
    # Print configuration
    # =========================================================================

    print(
        "\n"
        + "=" * 88
    )

    print(
        "12-POSE GLOBAL-OPTIMAL CALIBRATION DESIGN"
    )

    print(
        "=" * 88
    )


    print(
        "\nN =",
        N,
    )

    print(
        "radius =",
        RADIUS_MM,
        "mm",
    )


    print(
        "\nSupports:"
    )

    for i, support in enumerate(
        POSE_SUPPORTS
    ):

        print(
            f"  {i}: "
            f"tilt={support[0]:.1f} deg, "
            f"d={support[1]:.1f} mm"
        )


    print(
        "\nUV:"
    )

    print(
        uv
    )


    print(
        "\nDesired radial physical scan-line angles:"
    )

    print(
        radial_scanline_azimuth_from_uv(
            uv
        )
    )


    print(
        "\nOptimal support IDs:"
    )

    print(
        support_ids
    )


    print(
        "\nOptimal beta:"
    )

    print(
        beta_optimal
    )


    print(
        "\nGT hand-eye:"
    )

    print(
        T_EF_S_TRUE
    )


    print(
        "\nInitial hand-eye:"
    )

    print(
        T_init
    )


    # =========================================================================
    # Run only the global-optimal design
    # =========================================================================

    run_design(
        label=(
            "GLOBAL_OPTIMAL"
        ),

        beta_deg=(
            beta_optimal
        ),

        uv=(
            uv
        ),

        target_points_base=(
            target_points_base
        ),

        frame=(
            frame
        ),

        T_init=(
            T_init
        ),
    )


    plt.show()


if __name__ == "__main__":
    main()