# 사전에 정의된 bootstrap pattern을 스캔하는 코드

"""
Input
-----
1. Calibration board center in robot BASE frame
2. Calibration board normal in robot BASE frame
3. Initial hand-eye transform
4. Pose-design parameters:
   - radius
   - tilt_min
   - tilt_max
   - distance_near
   - distance_far
5. Physical-to-measurement transform ``^P T_S``

Output
------
12 desired robot TCP poses:

       ^B T_TCP

Geometry
--------
12 circular scan centers, with distance/tilt applied at physical origin P:

    psi = 0, 30, 60, ..., 330 deg

Four supports:

    support 0 = (tilt_min, distance_near)
    support 1 = (tilt_min, distance_far)
    support 2 = (tilt_max, distance_near)
    support 3 = (tilt_max, distance_far)

Support assignment:

    psi [deg] :   0  30  60  90 120 150 180 210 240 270 300 330
    support   :   0   1   0   3   3   3   1   0   1   2   2   2

Beta assignment:

    psi [deg] :   0  30  60  90 120 150 180 210 240 270 300 330
    beta[deg] :   0 120 240   0 240 120   0 120 240   0 240 120

The physical laser scan line is kept radial by compensating ordinary
azimuth alpha:

    psi = alpha + delta

    delta = atan2(
        sin(beta),
        cos(beta) / cos(theta)
    )

Therefore:

    alpha = psi - delta


Transform convention
--------------------

Initial hand-eye:

    ^TCP T_S

Desired sensor pose:

    ^B T_S

Robot command:

    ^B T_TCP

Relation:

    ^B T_S = ^B T_TCP @ ^TCP T_S

Therefore:

    ^B T_TCP = ^B T_S @ inv(^TCP T_S)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from robust_laser_handeye.laser_handeye.data import (
    PlaneFrame,
)

from robust_laser_handeye.laser_handeye.pose_design import (
    circular_uv_points,
    plane_uv_to_base_points,
    sensor_pose_from_target_point,
)

from robust_laser_handeye.laser_handeye.scene_generation import plane_basis


# =============================================================================
# Constants
# =============================================================================

N_POSES = 12


# Fixed optimal support-to-position assignment.
#
# support 0 = (tilt_min, distance_near)
# support 1 = (tilt_min, distance_far)
# support 2 = (tilt_max, distance_near)
# support 3 = (tilt_max, distance_far)

SUPPORT_IDS = np.array(
    [
        0,  # psi =   0 deg
        1,  # psi =  30 deg
        0,  # psi =  60 deg
        3,  # psi =  90 deg
        3,  # psi = 120 deg
        3,  # psi = 150 deg
        1,  # psi = 180 deg
        0,  # psi = 210 deg
        1,  # psi = 240 deg
        2,  # psi = 270 deg
        2,  # psi = 300 deg
        2,  # psi = 330 deg
    ],
    dtype=int,
)


# Fixed beta assignment.
#
# Every individual support contains beta={0,120,240} once.

BETA_DEG = np.array(
    [
          0.0,  # psi =   0 deg
        120.0,  # psi =  30 deg
        240.0,  # psi =  60 deg
          0.0,  # psi =  90 deg
        240.0,  # psi = 120 deg
        120.0,  # psi = 150 deg
          0.0,  # psi = 180 deg
        120.0,  # psi = 210 deg
        240.0,  # psi = 240 deg
          0.0,  # psi = 270 deg
        240.0,  # psi = 300 deg
        120.0,  # psi = 330 deg
    ],
    dtype=float,
)


# =============================================================================
# Data classes
# =============================================================================

@dataclass(frozen=True)
class CalibrationPose:
    """
    Information for one generated calibration pose.
    """

    scan_id: int
    support_id: int

    target_u_mm: float
    target_v_mm: float

    tilt_deg: float
    distance_mm: float

    beta_deg: float
    alpha_deg: float

    # Desired physical scan-line direction on the board.
    scanline_azimuth_deg: float

    # Desired sensor pose:
    #
    #     ^B T_S
    T_base_sensor: np.ndarray

    # Desired physical sensor-origin pose used for path design:
    #
    #     ^B T_P
    T_base_physical: np.ndarray

    # Desired robot TCP pose:
    #
    #     ^B T_TCP
    T_base_tcp: np.ndarray


# =============================================================================
# Validation
# =============================================================================

def validate_transform(
    T: np.ndarray,
    name: str,
) -> np.ndarray:
    """
    Basic validation for a 4x4 homogeneous transform.
    """

    T = np.asarray(
        T,
        dtype=float,
    )

    if T.shape != (4, 4):
        raise ValueError(
            f"{name} must have shape (4, 4), "
            f"but got {T.shape}."
        )

    if not np.all(
        np.isfinite(T)
    ):
        raise ValueError(
            f"{name} contains NaN or Inf."
        )

    if not np.allclose(
        T[3],
        np.array(
            [0.0, 0.0, 0.0, 1.0]
        ),
        atol=1e-8,
    ):
        raise ValueError(
            f"{name} is not a valid homogeneous transform: "
            "last row must be [0, 0, 0, 1]."
        )

    R = T[:3, :3]

    if not np.allclose(
        R.T @ R,
        np.eye(3),
        atol=1e-5,
    ):
        raise ValueError(
            f"{name} rotation matrix is not orthonormal."
        )

    if not np.isclose(
        np.linalg.det(R),
        1.0,
        atol=1e-5,
    ):
        raise ValueError(
            f"{name} rotation matrix determinant is not +1."
        )

    return T


def validate_design_parameters(
    radius_mm: float,
    tilt_min_deg: float,
    tilt_max_deg: float,
    distance_near_mm: float,
    distance_far_mm: float,
) -> None:
    """
    Validate pose-design parameters.
    """

    if radius_mm <= 0.0:
        raise ValueError(
            "radius_mm must be > 0."
        )

    if tilt_min_deg < 0.0:
        raise ValueError(
            "tilt_min_deg must be >= 0."
        )

    if tilt_max_deg <= tilt_min_deg:
        raise ValueError(
            "tilt_max_deg must be larger than tilt_min_deg."
        )

    if tilt_max_deg >= 90.0:
        raise ValueError(
            "tilt_max_deg must be < 90 deg."
        )

    if distance_near_mm <= 0.0:
        raise ValueError(
            "distance_near_mm must be > 0."
        )

    if distance_far_mm <= distance_near_mm:
        raise ValueError(
            "distance_far_mm must be larger than distance_near_mm."
        )


# =============================================================================
# Plane
# =============================================================================

def make_plane_frame(
    board_center_base_mm: np.ndarray,
    board_normal_base: np.ndarray,
) -> PlaneFrame:
    """
    Build a PlaneFrame from the calibration-board geometry.

    Parameters
    ----------
    board_center_base_mm
        Point on the calibration board expressed in BASE frame.

    board_normal_base
        Plane normal expressed in BASE frame.

    Returns
    -------
    PlaneFrame
        Orthonormal plane basis (u, v, n) and plane offset.
    """

    board_center_base_mm = np.asarray(
        board_center_base_mm,
        dtype=float,
    ).reshape(3)

    board_normal_base = np.asarray(
        board_normal_base,
        dtype=float,
    ).reshape(3)

    if not np.all(
        np.isfinite(
            board_center_base_mm
        )
    ):
        raise ValueError(
            "board_center_base_mm contains NaN or Inf."
        )

    if not np.all(
        np.isfinite(
            board_normal_base
        )
    ):
        raise ValueError(
            "board_normal_base contains NaN or Inf."
        )

    normal_norm = float(
        np.linalg.norm(
            board_normal_base
        )
    )

    if normal_norm < 1e-12:
        raise ValueError(
            "board_normal_base must be non-zero."
        )

    # Preserve the normal sign selected by estimate_initial_plane.  The generic
    # scene helper canonicalizes plane offset to l >= 0 and may flip the normal,
    # which would turn the desired sensor viewing direction by 180 degrees.
    plane_n = board_normal_base / normal_norm
    plane_l = float(plane_n @ board_center_base_mm)

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
# Scan-line geometry
# =============================================================================

def radial_scanline_azimuth_from_uv(
    uv: np.ndarray,
) -> np.ndarray:
    """
    Desired physical scan-line direction.

    Each scan center lies on a circle and the laser line is chosen
    to point radially:

        psi_i = atan2(v_i, u_i)

    Note
    ----
    A physical line is equivalent modulo 180 deg.
    """

    uv = np.asarray(
        uv,
        dtype=float,
    )

    if uv.ndim != 2:
        raise ValueError(
            "uv must be a 2D array."
        )

    if uv.shape[1] != 2:
        raise ValueError(
            "uv must have shape (N, 2)."
        )

    psi_deg = np.degrees(
        np.arctan2(
            uv[:, 1],
            uv[:, 0],
        )
    )

    return (
        psi_deg
        % 360.0
    )


def alpha_for_scanline_direction(
    scanline_azimuth_deg: np.ndarray,
    tilt_deg: np.ndarray,
    beta_deg: np.ndarray,
) -> np.ndarray:
    """
    Convert desired physical scan-line direction psi into
    the ordinary sensor azimuth alpha.

    Plane scan-line vector gives:

        psi = alpha + delta

    where:

        delta = atan2(
            sin(beta),
            cos(beta) / cos(theta)
        )

    Therefore:

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
            "scanline_azimuth_deg, tilt_deg and beta_deg "
            "must have identical shapes."
        )

    theta_rad = np.deg2rad(
        tilt_deg
    )

    beta_rad = np.deg2rad(
        beta_deg
    )

    delta_deg = np.degrees(
        np.arctan2(
            np.sin(
                beta_rad
            ),
            np.cos(
                beta_rad
            )
            / np.cos(
                theta_rad
            ),
        )
    )

    alpha_deg = (
        psi_deg
        - delta_deg
    ) % 360.0

    return alpha_deg


def physical_scanline_azimuth_deg(
    alpha_deg: np.ndarray,
    tilt_deg: np.ndarray,
    beta_deg: np.ndarray,
) -> np.ndarray:
    """
    Reconstruct the physical scan-line direction from
    alpha, tilt and beta.

    Used only as a sanity check.
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

    theta_rad = np.deg2rad(
        tilt_deg
    )

    beta_rad = np.deg2rad(
        beta_deg
    )

    delta_deg = np.degrees(
        np.arctan2(
            np.sin(
                beta_rad
            ),
            np.cos(
                beta_rad
            )
            / np.cos(
                theta_rad
            ),
        )
    )

    return (
        alpha_deg
        + delta_deg
    ) % 360.0


# =============================================================================
# Support construction
# =============================================================================

def make_pose_supports(
    tilt_min_deg: float,
    tilt_max_deg: float,
    distance_near_mm: float,
    distance_far_mm: float,
) -> tuple[
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
]:
    """
    Construct the four (tilt, distance) supports.

    support 0:
        tilt_min, distance_near

    support 1:
        tilt_min, distance_far

    support 2:
        tilt_max, distance_near

    support 3:
        tilt_max, distance_far
    """

    return (
        (
            float(
                tilt_min_deg
            ),
            float(
                distance_near_mm
            ),
        ),
        (
            float(
                tilt_min_deg
            ),
            float(
                distance_far_mm
            ),
        ),
        (
            float(
                tilt_max_deg
            ),
            float(
                distance_near_mm
            ),
        ),
        (
            float(
                tilt_max_deg
            ),
            float(
                distance_far_mm
            ),
        ),
    )


# =============================================================================
# Main pose generator
# =============================================================================

def generate_optimal_tcp_poses(
    board_center_base_mm: np.ndarray,
    board_normal_base: np.ndarray,
    T_tcp_sensor_init: np.ndarray,

    radius_mm: float = 100.0,

    tilt_min_deg: float = 5.0,
    tilt_max_deg: float = 40.0,

    distance_near_mm: float = 60.0,
    distance_far_mm: float = 120.0,
    sensor_forward_axis: str = "+z",
    T_physical_measurement: np.ndarray | None = None,
    alpha_branch_offset_deg: float = 0.0,
) -> list[CalibrationPose]:
    """
    Generate the 12 calibration TCP poses.

    Parameters
    ----------
    board_center_base_mm : array-like, shape (3,)
        Calibration-board center expressed in robot BASE coordinates [mm].

    board_normal_base : array-like, shape (3,)
        Calibration-board normal expressed in robot BASE coordinates.

    T_tcp_sensor_init : array-like, shape (4,4)
        Current hand-eye estimate:

            ^TCP T_S

        Sensor frame expressed relative to the TCP frame.

    radius_mm : float
        Radius of the circular board scan-center arrangement.

    tilt_min_deg : float
        Small tilt support [deg].

    tilt_max_deg : float
        Large tilt support [deg].

    distance_near_mm : float
        Near sensor-to-target distance [mm].

    distance_far_mm : float
        Far sensor-to-target distance [mm].

    sensor_forward_axis : {"+z", "-z"}
        Axis of the physical planning frame P that looks toward the board.
        Keyence uses ``-z`` and P/S currently have identical axis directions.

    T_physical_measurement : array-like, shape (4,4), optional
        ``^P T_S`` from physical sensor origin P to measurement/profile origin
        S. The circular pattern, distance and tilt are constructed at P first;
        S and the commanded TCP pose are derived afterward. Identity preserves
        the legacy behavior when the two origins coincide.

    alpha_branch_offset_deg : float, optional
        Additional ordinary-azimuth branch offset. Use 0 deg for the original
        branch and 180 deg for the calibration-equivalent opposite branch.

    Returns
    -------
    poses : list[CalibrationPose]
        12 generated calibration poses.

        For actual robot control, use:

            pose.T_base_tcp

        which is:

            ^B T_TCP
    """

    # =========================================================================
    # Validate inputs
    # =========================================================================

    validate_design_parameters(
        radius_mm=radius_mm,
        tilt_min_deg=tilt_min_deg,
        tilt_max_deg=tilt_max_deg,
        distance_near_mm=distance_near_mm,
        distance_far_mm=distance_far_mm,
    )
    forward_axis = str(sensor_forward_axis).strip().lower()
    if forward_axis not in {"+z", "-z"}:
        raise ValueError("sensor_forward_axis must be '+z' or '-z'")

    alpha_branch_offset_deg = float(alpha_branch_offset_deg)
    if not np.isfinite(alpha_branch_offset_deg):
        raise ValueError("alpha_branch_offset_deg must be finite")

    board_center_base_mm = np.asarray(
        board_center_base_mm,
        dtype=float,
    ).reshape(3)

    board_normal_base = np.asarray(
        board_normal_base,
        dtype=float,
    ).reshape(3)

    T_tcp_sensor_init = (
        validate_transform(
            T=T_tcp_sensor_init,
            name="T_tcp_sensor_init",
        )
    )
    if T_physical_measurement is None:
        T_physical_measurement = np.eye(4)
    T_physical_measurement = validate_transform(
        T=np.asarray(T_physical_measurement, dtype=float),
        name="T_physical_measurement",
    )


    # =========================================================================
    # Board frame
    # =========================================================================

    frame = make_plane_frame(
        board_center_base_mm=(
            board_center_base_mm
        ),
        board_normal_base=(
            board_normal_base
        ),
    )


    # =========================================================================
    # Circular scan centers
    #
    # psi = 0, 30, ..., 330 deg
    # =========================================================================

    uv = circular_uv_points(
        radius_mm=(
            float(radius_mm)
        ),
        N=N_POSES,
    )

    uv = np.asarray(
        uv,
        dtype=float,
    ).reshape(
        N_POSES,
        2,
    )


    # Convert board (u,v) coordinates to robot BASE coordinates.
    target_points_base = (
        plane_uv_to_base_points(
            uv=uv,
            target_center_base_mm=(
                board_center_base_mm
            ),
            frame=frame,
        )
    )


    # =========================================================================
    # Four tilt-distance supports
    # =========================================================================

    pose_supports = (
        make_pose_supports(
            tilt_min_deg=(
                tilt_min_deg
            ),
            tilt_max_deg=(
                tilt_max_deg
            ),
            distance_near_mm=(
                distance_near_mm
            ),
            distance_far_mm=(
                distance_far_mm
            ),
        )
    )


    # =========================================================================
    # Assign tilt and distance to each circular position
    # =========================================================================

    tilt_deg = np.zeros(
        N_POSES,
        dtype=float,
    )

    distance_mm = np.zeros(
        N_POSES,
        dtype=float,
    )

    for scan_id in range(
        N_POSES
    ):

        support_id = int(
            SUPPORT_IDS[
                scan_id
            ]
        )

        (
            tilt_value,
            distance_value,
        ) = pose_supports[
            support_id
        ]

        tilt_deg[
            scan_id
        ] = tilt_value

        distance_mm[
            scan_id
        ] = distance_value


    # =========================================================================
    # Physical scan-line direction
    #
    # Radial scan line:
    #
    #   psi = atan2(v,u)
    # =========================================================================

    scanline_azimuth_deg = (
        radial_scanline_azimuth_from_uv(
            uv
        )
    )


    # =========================================================================
    # Calculate ordinary azimuth alpha
    #
    # beta changes the actual physical line direction, so alpha is
    # compensated to keep the PHYSICAL scan line radial.
    # =========================================================================

    alpha_deg = (
        alpha_for_scanline_direction(
            scanline_azimuth_deg=(
                scanline_azimuth_deg
            ),
            tilt_deg=(
                tilt_deg
            ),
            beta_deg=(
                BETA_DEG
            ),
        )
        + alpha_branch_offset_deg
    ) % 360.0


    # =========================================================================
    # Sanity check: physical scan line must actually be radial
    # =========================================================================

    recovered_scanline_deg = (
        physical_scanline_azimuth_deg(
            alpha_deg=(
                alpha_deg
            ),
            tilt_deg=(
                tilt_deg
            ),
            beta_deg=(
                BETA_DEG
            ),
        )
    )

    # Physical line angle is equivalent modulo 180 deg.
    line_error_deg = (
        (
            recovered_scanline_deg
            - scanline_azimuth_deg
            + 90.0
        )
        % 180.0
        - 90.0
    )

    max_line_error = float(
        np.max(
            np.abs(
                line_error_deg
            )
        )
    )

    if max_line_error > 1e-8:
        raise RuntimeError(
            "Failed to construct radial physical scan lines. "
            f"max error = {max_line_error} deg."
        )


    # =========================================================================
    # Hand-eye inverse
    #
    # Input:
    #
    #   ^TCP T_S
    #
    # Need:
    #
    #   ^S T_TCP
    # =========================================================================

    T_sensor_tcp = np.linalg.inv(
        T_tcp_sensor_init
    )


    # =========================================================================
    # Generate desired sensor poses and TCP poses
    # =========================================================================

    poses: list[
        CalibrationPose
    ] = []


    for scan_id in range(
        N_POSES
    ):

        support_id = int(
            SUPPORT_IDS[
                scan_id
            ]
        )


        # ---------------------------------------------------------------------
        # Desired sensor pose
        #
        #     ^B T_S
        # ---------------------------------------------------------------------

        T_base_physical = (
            sensor_pose_from_target_point(
                target_point_base_mm=(
                    target_points_base[
                        scan_id
                    ]
                ),

                frame=frame,

                distance_mm=float(
                    distance_mm[
                        scan_id
                    ]
                ),

                tilt_deg=float(
                    tilt_deg[
                        scan_id
                    ]
                ),

                azimuth_deg=float(
                    alpha_deg[
                        scan_id
                    ]
                ),

                normal_azimuth_sensor_deg=float(
                    BETA_DEG[
                        scan_id
                    ]
                ),
            )
        )

        if forward_axis == "-z":
            # The base pose generator constructs +Z toward the target. A
            # A 180-degree rotation about sensor Y reverses +Z. Sensor X also
            # reverses, which represents the same unoriented physical laser
            # line while avoiding an unnecessary wrist-roll branch change.
            T_base_physical = T_base_physical.copy()
            T_base_physical[:3, :3] = T_base_physical[:3, :3] @ np.diag(
                [-1.0, 1.0, -1.0]
            )

        T_base_physical = (
            validate_transform(
                T=T_base_physical,
                name=(
                    f"T_base_physical[{scan_id}]"
                ),
            )
        )

        # P is the path-design origin. The supplied hand-eye transform refers
        # to measurement origin S, so derive S only after P has been placed.
        T_base_sensor = validate_transform(
            T=T_base_physical @ T_physical_measurement,
            name=f"T_base_sensor[{scan_id}]",
        )


        # ---------------------------------------------------------------------
        # Desired TCP pose
        #
        #     ^B T_S
        #       =
        #     ^B T_TCP @ ^TCP T_S
        #
        # therefore
        #
        #     ^B T_TCP
        #       =
        #     ^B T_S @ inv(^TCP T_S)
        # ---------------------------------------------------------------------

        T_base_tcp = (
            T_base_sensor
            @ T_sensor_tcp
        )

        T_base_tcp = (
            validate_transform(
                T=T_base_tcp,
                name=(
                    f"T_base_tcp[{scan_id}]"
                ),
            )
        )


        # ---------------------------------------------------------------------
        # Store result
        # ---------------------------------------------------------------------

        poses.append(
            CalibrationPose(
                scan_id=scan_id,

                support_id=(
                    support_id
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

                tilt_deg=float(
                    tilt_deg[
                        scan_id
                    ]
                ),

                distance_mm=float(
                    distance_mm[
                        scan_id
                    ]
                ),

                beta_deg=float(
                    BETA_DEG[
                        scan_id
                    ]
                ),

                alpha_deg=float(
                    alpha_deg[
                        scan_id
                    ]
                ),

                scanline_azimuth_deg=float(
                    scanline_azimuth_deg[
                        scan_id
                    ]
                ),

                T_base_sensor=(
                    T_base_sensor
                ),

                T_base_physical=(
                    T_base_physical
                ),

                T_base_tcp=(
                    T_base_tcp
                ),
            )
        )


    return poses


# =============================================================================
# Convenience function
# =============================================================================

def get_tcp_pose_matrices(
    board_center_base_mm: np.ndarray,
    board_normal_base: np.ndarray,
    T_tcp_sensor_init: np.ndarray,

    radius_mm: float = 100.0,
    tilt_min_deg: float = 5.0,
    tilt_max_deg: float = 40.0,
    distance_near_mm: float = 60.0,
    distance_far_mm: float = 120.0,

    sensor_forward_axis: str = "-z",
    T_physical_measurement: np.ndarray | None = None,
) -> list[np.ndarray]:

    poses = generate_optimal_tcp_poses(
        board_center_base_mm=board_center_base_mm,
        board_normal_base=board_normal_base,
        T_tcp_sensor_init=T_tcp_sensor_init,

        radius_mm=radius_mm,
        tilt_min_deg=tilt_min_deg,
        tilt_max_deg=tilt_max_deg,
        distance_near_mm=distance_near_mm,
        distance_far_mm=distance_far_mm,

        sensor_forward_axis=sensor_forward_axis,
        T_physical_measurement=T_physical_measurement,
    )

    return [
        pose.T_base_tcp.copy()
        for pose in poses
    ]


# =============================================================================
# Printing
# =============================================================================

def print_pose_summary(
    poses: list[CalibrationPose],
) -> None:
    """
    Print generated calibration pose geometry.
    """

    print()
    print(
        "=" * 112
    )

    print(
        "GENERATED CALIBRATION TCP POSES"
    )

    print(
        "=" * 112
    )

    print(
        " id | sup |"
        "      u       v |"
        " tilt |"
        "    d |"
        "  psi(radial) |"
        "  alpha |"
        "   beta"
    )

    print(
        "-" * 112
    )


    for pose in poses:

        print(
            f"{pose.scan_id:3d} | "
            f"{pose.support_id:3d} | "
            f"{pose.target_u_mm:7.2f} "
            f"{pose.target_v_mm:7.2f} | "
            f"{pose.tilt_deg:4.1f}° | "
            f"{pose.distance_mm:6.1f} | "
            f"{pose.scanline_azimuth_deg:12.1f} | "
            f"{pose.alpha_deg:6.1f} | "
            f"{pose.beta_deg:6.1f}"
        )


def print_tcp_matrices(
    poses: list[CalibrationPose],
    precision: int = 6,
) -> None:
    """
    Print all ^B T_TCP matrices.
    """

    np.set_printoptions(
        precision=precision,
        suppress=True,
    )

    for pose in poses:

        print()
        print(
            "=" * 72
        )

        print(
            f"TCP POSE {pose.scan_id}"
        )

        print(
            "=" * 72
        )

        print(
            pose.T_base_tcp
        )


# =============================================================================
# Example
# =============================================================================

def main() -> None:
    """
    Example usage.

    Replace these values with the actual board geometry
    and current hand-eye estimate.
    """

    # =========================================================================
    # Board information in robot BASE frame
    # =========================================================================

    board_center_base_mm = np.array(
        [
            0.0,
            0.0,
            500.0,
        ],
        dtype=float,
    )


    board_normal_base = np.array(
        [
            0.2,
            0.1,
            1.0,
        ],
        dtype=float,
    )


    # =========================================================================
    # Initial hand-eye
    #
    #     ^TCP T_S
    #
    # IMPORTANT:
    # Confirm that "TCP" here is exactly the frame commanded by
    # the robot controller.
    # =========================================================================

    T_tcp_sensor_init = np.array(
        [
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
        ],
        dtype=float,
    )


    # =========================================================================
    # User-selected design range
    # =========================================================================

    radius_mm = 100.0

    tilt_min_deg = 5.0
    tilt_max_deg = 40.0

    distance_near_mm = 60.0
    distance_far_mm = 120.0


    # =========================================================================
    # Generate poses
    # =========================================================================

    poses = generate_optimal_tcp_poses(
        board_center_base_mm=(
            board_center_base_mm
        ),

        board_normal_base=(
            board_normal_base
        ),

        T_tcp_sensor_init=(
            T_tcp_sensor_init
        ),

        radius_mm=(
            radius_mm
        ),

        tilt_min_deg=(
            tilt_min_deg
        ),

        tilt_max_deg=(
            tilt_max_deg
        ),

        distance_near_mm=(
            distance_near_mm
        ),

        distance_far_mm=(
            distance_far_mm
        ),
    )


    # =========================================================================
    # Print design
    # =========================================================================

    print_pose_summary(
        poses
    )


    # =========================================================================
    # Print TCP matrices
    #
    # These are the actual desired robot poses:
    #
    #     ^B T_TCP
    # =========================================================================

    print_tcp_matrices(
        poses
    )


    # =========================================================================
    # If only matrices are required:
    # =========================================================================

    tcp_pose_matrices = [
        pose.T_base_tcp
        for pose in poses
    ]

    print()
    print(
        f"Generated {len(tcp_pose_matrices)} TCP poses."
    )


if __name__ == "__main__":
    main()