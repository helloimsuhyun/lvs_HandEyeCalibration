from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import select
import sys
import time
from typing import TYPE_CHECKING
import uuid

import numpy as np


try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
except ImportError:  # pragma: no cover
    plt = None
    Figure = None
    Poly3DCollection = None


try:
    from .laser_adapter import LaserAdapter
except ImportError:
    from laser_adapter import LaserAdapter

if TYPE_CHECKING:
    try:
        from .robot_adapter import RobotAdapter
    except ImportError:
        from real_laser_handeye.robot_adapter import RobotAdapter


# =============================================================================
# Data class
# =============================================================================


@dataclass(frozen=True)
class PlaneEstimate:
    """
    Estimated calibration plane.

    centroid_w
        Geometric AREA centroid of the convex hull of the observed
        laser profiles after projection onto the fitted plane.

        IMPORTANT:
        This is NOT simply the arithmetic mean of all measured points.

    normal_w
        Plane normal oriented toward the side where the sensor origins
        were located during initial plane acquisition.

    boundary_uv
        Convex hull vertices expressed relative to centroid_w
        in the estimated plane coordinate system.

    boundary_w
        Same convex hull vertices expressed in world/base coordinates.
    """

    centroid_w: np.ndarray
    normal_w: np.ndarray
    offset_w: float

    basis_u_w: np.ndarray
    basis_v_w: np.ndarray

    boundary_uv: np.ndarray
    boundary_w: np.ndarray

    rms_mm: float
    max_abs_mm: float

    singular_values: np.ndarray

    scan_count: int
    point_count: int


# =============================================================================
# Transform utilities
# =============================================================================


def validate_transform(
    value: np.ndarray,
    name: str,
) -> np.ndarray:

    transform = np.asarray(
        value,
        dtype=float,
    )

    if (
        transform.shape != (4, 4)
        or not np.all(
            np.isfinite(transform)
        )
    ):
        raise ValueError(
            f"{name} must be a finite 4x4 matrix"
        )

    if not np.allclose(
        transform[3],
        [0.0, 0.0, 0.0, 1.0],
        atol=1e-8,
    ):
        raise ValueError(
            f"{name} has an invalid homogeneous last row"
        )

    rotation = transform[:3, :3]

    if not np.allclose(
        rotation.T @ rotation,
        np.eye(3),
        atol=1e-5,
    ):
        raise ValueError(
            f"{name} rotation is not orthonormal"
        )

    if not np.isclose(
        np.linalg.det(rotation),
        1.0,
        atol=1e-5,
    ):
        raise ValueError(
            f"{name} rotation determinant must be +1"
        )

    return transform.copy()


def load_transform(
    path: Path,
) -> np.ndarray:

    if not path.exists():
        raise FileNotFoundError(
            path
        )

    if path.suffix.lower() == ".json":

        value = json.loads(
            path.read_text(
                encoding="utf-8"
            )
        )

        if isinstance(
            value,
            dict,
        ):

            for key in (
                "T_tcp_sensor",
                "T_ef_s",
                "transform",
            ):
                if key in value:
                    value = value[key]
                    break

        return validate_transform(
            np.asarray(
                value,
                dtype=float,
            ),
            str(path),
        )

    try:

        value = np.loadtxt(
            path,
            delimiter=",",
        )

    except ValueError:

        value = np.loadtxt(
            path
        )

    return validate_transform(
        value,
        str(path),
    )


def transform_points(
    T_a_b: np.ndarray,
    points_b: np.ndarray,
) -> np.ndarray:

    points = np.asarray(
        points_b,
        dtype=float,
    )

    if (
        points.ndim != 2
        or points.shape[1] != 3
    ):
        raise ValueError(
            "points must have shape (N, 3)"
        )

    return (
        points
        @ T_a_b[:3, :3].T
        + T_a_b[:3, 3]
    )


# =============================================================================
# File utilities
# =============================================================================


def atomic_json(
    path: Path,
    value: dict,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_name(
        f".{path.name}.{uuid.uuid4().hex}.tmp"
    )

    try:

        with temporary.open(
            "w",
            encoding="utf-8",
        ) as stream:

            json.dump(
                value,
                stream,
                indent=2,
                ensure_ascii=False,
            )

            stream.write("\n")
            stream.flush()

            os.fsync(
                stream.fileno()
            )

        temporary.replace(
            path
        )

    finally:

        if temporary.exists():
            temporary.unlink()


def next_capture_path(
    dataset_dir: Path,
) -> Path:

    ids: list[int] = []

    for path in dataset_dir.glob(
        "capture_*.npz"
    ):

        try:

            ids.append(
                int(
                    path.stem.split("_")[-1]
                )
            )

        except ValueError:
            pass

    return (
        dataset_dir
        / f"capture_{max(ids, default=0) + 1:04d}.npz"
    )


# =============================================================================
# Rotation distance
# =============================================================================


def rotation_distance_deg(
    first: np.ndarray,
    second: np.ndarray,
) -> float:

    relative = (
        first.T
        @ second
    )

    cosine = np.clip(
        (
            float(
                np.trace(relative)
            )
            - 1.0
        )
        * 0.5,
        -1.0,
        1.0,
    )

    return math.degrees(
        math.acos(
            float(cosine)
        )
    )


# =============================================================================
# Laser profile filtering
# =============================================================================


def filter_profile(
    points_s: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:

    points = np.asarray(
        points_s,
        dtype=float,
    )

    if (
        points.ndim != 2
        or points.shape[1] != 3
    ):
        raise ValueError(
            "laser adapter must return an (N, 3) array"
        )

    valid = np.all(
        np.isfinite(points),
        axis=1,
    )

    valid &= (
        np.abs(
            points[:, 1]
        )
        <= args.max_abs_sensor_y_mm
    )

    if args.min_sensor_x_mm is not None:
        valid &= (
            points[:, 0]
            >= args.min_sensor_x_mm
        )

    if args.max_sensor_x_mm is not None:
        valid &= (
            points[:, 0]
            <= args.max_sensor_x_mm
        )

    if args.min_sensor_z_mm is not None:
        valid &= (
            points[:, 2]
            >= args.min_sensor_z_mm
        )

    if args.max_sensor_z_mm is not None:
        valid &= (
            points[:, 2]
            <= args.max_sensor_z_mm
        )

    result = points[
        valid
    ]

    if len(result) < args.min_points:

        raise RuntimeError(
            f"profile has {len(result)} valid points; "
            f"need at least {args.min_points}"
        )

    return result


# =============================================================================
# Capture
# =============================================================================


def capture_once(
    robot: RobotAdapter,
    laser: LaserAdapter,
    args: argparse.Namespace,
) -> Path:

    # -------------------------------------------------------------------------
    # Robot pose before acquisition
    # -------------------------------------------------------------------------

    T_before = validate_transform(
        robot.read_T_base_tcp(),
        "pre-capture TCP",
    )

    # -------------------------------------------------------------------------
    # Laser profile
    # -------------------------------------------------------------------------

    points_s = filter_profile(
        laser.read_profile(
            timeout_s=args.timeout_s
        ),
        args,
    )

    profile_timestamp_ns = (
        time.time_ns()
    )

    # -------------------------------------------------------------------------
    # Robot pose after acquisition
    # -------------------------------------------------------------------------

    T_after = validate_transform(
        robot.read_T_base_tcp(),
        "post-capture TCP",
    )

    tcp_timestamp_ns = (
        time.time_ns()
    )

    # -------------------------------------------------------------------------
    # Stationarity check
    # -------------------------------------------------------------------------

    translation_delta = float(
        np.linalg.norm(
            T_after[:3, 3]
            - T_before[:3, 3]
        )
    )

    rotation_delta = (
        rotation_distance_deg(
            T_before[:3, :3],
            T_after[:3, :3],
        )
    )

    if (
        translation_delta
        > args.max_stationarity_translation_mm
    ):
        raise RuntimeError(
            f"robot moved {translation_delta:.3f} mm "
            "during capture"
        )

    if (
        rotation_delta
        > args.max_stationarity_rotation_deg
    ):
        raise RuntimeError(
            f"robot rotated {rotation_delta:.3f} deg "
            "during capture"
        )

    # -------------------------------------------------------------------------
    # Sensor pose
    #
    # ^W T_S
    # =
    # ^W T_TCP @ ^TCP T_S
    # -------------------------------------------------------------------------

    T_world_sensor = (
        T_after
        @ args.T_tcp_sensor
    )

    # -------------------------------------------------------------------------
    # Laser points -> world/base
    # -------------------------------------------------------------------------

    points_w = transform_points(
        T_world_sensor,
        points_s,
    )

    # -------------------------------------------------------------------------
    # Save capture
    # -------------------------------------------------------------------------

    args.dataset_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output = next_capture_path(
        args.dataset_dir
    )

    temporary = output.with_name(
        f".{output.name}.{uuid.uuid4().hex}.tmp"
    )

    try:

        with temporary.open(
            "xb"
        ) as stream:

            np.savez_compressed(
                stream,

                T_world_tcp=(
                    T_after
                ),

                # Compatibility alias for calibration tools and older
                # datasets that name the workflow planning frame ``base``.
                T_base_tcp=(
                    T_after
                ),

                T_tcp_sensor=(
                    args.T_tcp_sensor
                ),

                T_world_sensor=(
                    T_world_sensor
                ),

                points_s=(
                    points_s
                ),

                points_w=(
                    points_w
                ),

                tcp_timestamp_ns=(
                    np.int64(
                        tcp_timestamp_ns
                    )
                ),

                profile_timestamp_ns=(
                    np.int64(
                        profile_timestamp_ns
                    )
                ),

                captured_at=np.array(
                    datetime.now(
                        timezone.utc
                    ).isoformat()
                ),
            )

            stream.flush()

            os.fsync(
                stream.fileno()
            )

        temporary.replace(
            output
        )

    finally:

        if temporary.exists():
            temporary.unlink()

    # -------------------------------------------------------------------------
    # Optional simulated pose advancement
    # -------------------------------------------------------------------------

    advance_pose = getattr(
        robot,
        "advance_pose",
        None,
    )

    if callable(
        advance_pose
    ):
        advance_pose()

    print(
        f"saved {output} "
        f"({len(points_s)} points)"
    )

    print(
        "world endpoints: "
        f"{points_w[0]} -> {points_w[-1]}"
    )

    return output


# =============================================================================
# Load captures
# =============================================================================


def load_world_points(
    dataset_dir: Path,
    T_tcp_sensor: np.ndarray,
) -> tuple[
    list[np.ndarray],
    np.ndarray,
    list[Path],
]:
    """
    Load captured profiles.

    Returns
    -------
    scans_w
        Laser profiles expressed in world/base coordinates.

    sensor_origins_w
        Sensor origin for each capture expressed in world/base coordinates.

        Shape:

            (num_scans, 3)

        These positions are later used to resolve the +/- ambiguity
        of the plane normal.

    paths
        Capture files.
    """

    scans_w: list[np.ndarray] = []
    sensor_origins_w: list[np.ndarray] = []

    paths = sorted(
        dataset_dir.glob(
            "capture_*.npz"
        )
    )

    for path in paths:

        with np.load(
            path,
            allow_pickle=False,
        ) as pair:

            # -------------------------------------------------------------
            # TCP pose
            # -------------------------------------------------------------

            if "T_world_tcp" in pair:

                T_world_tcp = (
                    validate_transform(
                        pair[
                            "T_world_tcp"
                        ],
                        str(path),
                    )
                )

            elif "T_base_tcp" in pair:

                T_world_tcp = (
                    validate_transform(
                        pair[
                            "T_base_tcp"
                        ],
                        str(path),
                    )
                )

            else:

                raise ValueError(
                    f"{path} has no "
                    "T_world_tcp/T_base_tcp"
                )

            # -------------------------------------------------------------
            # Laser points
            # -------------------------------------------------------------

            if "points_s" not in pair:

                raise ValueError(
                    f"{path} has no points_s"
                )

            points_s = np.asarray(
                pair[
                    "points_s"
                ],
                dtype=float,
            )

        # -------------------------------------------------------------
        # Sensor pose
        #
        # ^W T_S
        # =
        # ^W T_TCP @ ^TCP T_S
        # -------------------------------------------------------------

        T_world_sensor = (
            T_world_tcp
            @ T_tcp_sensor
        )

        # -------------------------------------------------------------
        # Points -> world
        # -------------------------------------------------------------

        points_w = (
            transform_points(
                T_world_sensor,
                points_s,
            )
        )

        scans_w.append(
            points_w
        )

        # -------------------------------------------------------------
        # Sensor origin in world
        # -------------------------------------------------------------

        sensor_origins_w.append(
            T_world_sensor[
                :3,
                3,
            ].copy()
        )

    return (
        scans_w,

        np.asarray(
            sensor_origins_w,
            dtype=float,
        ).reshape(
            -1,
            3,
        ),

        paths,
    )


# =============================================================================
# Convex hull
# =============================================================================


def _cross_2d(
    o: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
) -> float:

    return float(
        (a[0] - o[0])
        * (b[1] - o[1])
        - (a[1] - o[1])
        * (b[0] - o[0])
    )


def convex_hull_2d(
    points_uv: np.ndarray,
) -> np.ndarray:
    """
    Andrew monotonic-chain convex hull.

    Returns ordered hull vertices.
    """

    points = np.unique(
        np.asarray(
            points_uv,
            dtype=float,
        ),
        axis=0,
    )

    if len(points) < 3:

        raise RuntimeError(
            "at least three non-collinear "
            "projected points are required"
        )

    order = np.lexsort(
        (
            points[:, 1],
            points[:, 0],
        )
    )

    points = points[
        order
    ]

    lower: list[np.ndarray] = []

    for point in points:

        while (
            len(lower) >= 2
            and _cross_2d(
                lower[-2],
                lower[-1],
                point,
            )
            <= 0.0
        ):
            lower.pop()

        lower.append(
            point
        )

    upper: list[np.ndarray] = []

    for point in reversed(
        points
    ):

        while (
            len(upper) >= 2
            and _cross_2d(
                upper[-2],
                upper[-1],
                point,
            )
            <= 0.0
        ):
            upper.pop()

        upper.append(
            point
        )

    hull = np.asarray(
        lower[:-1]
        + upper[:-1],
        dtype=float,
    )

    if len(hull) < 3:

        raise RuntimeError(
            "projected points are nearly collinear; "
            "vary robot pose more"
        )

    return hull


def polygon_area(
    points_uv: np.ndarray,
) -> float:

    x = points_uv[:, 0]
    y = points_uv[:, 1]

    return (
        0.5
        * abs(
            float(
                np.dot(
                    x,
                    np.roll(
                        y,
                        -1,
                    ),
                )
                - np.dot(
                    y,
                    np.roll(
                        x,
                        -1,
                    ),
                )
            )
        )
    )


def polygon_centroid_2d(
    points_uv: np.ndarray,
) -> np.ndarray:
    """
    Compute the geometric AREA centroid of a simple polygon.

    IMPORTANT
    ---------
    This is not:

        mean(vertices)

    and not:

        mean(all measured laser points)

    It is the centroid of the polygon AREA enclosed by the convex hull.

    For ordered polygon vertices:

        cross_i = x_i y_{i+1} - x_{i+1} y_i

        Cx =
            sum((x_i + x_{i+1}) cross_i)
            --------------------------------
                    3 sum(cross_i)

        Cy =
            sum((y_i + y_{i+1}) cross_i)
            --------------------------------
                    3 sum(cross_i)
    """

    points = np.asarray(
        points_uv,
        dtype=float,
    )

    if (
        points.ndim != 2
        or points.shape[1] != 2
    ):
        raise ValueError(
            "points_uv must have shape (N, 2)"
        )

    if len(points) < 3:

        raise ValueError(
            "polygon requires at least 3 vertices"
        )

    x0 = points[:, 0]
    y0 = points[:, 1]

    x1 = np.roll(
        x0,
        -1,
    )

    y1 = np.roll(
        y0,
        -1,
    )

    cross = (
        x0 * y1
        - x1 * y0
    )

    area2 = float(
        np.sum(
            cross
        )
    )

    if abs(area2) < 1e-12:

        raise RuntimeError(
            "polygon area is too small "
            "to compute centroid"
        )

    centroid_x = float(
        np.sum(
            (x0 + x1)
            * cross
        )
        / (
            3.0
            * area2
        )
    )

    centroid_y = float(
        np.sum(
            (y0 + y1)
            * cross
        )
        / (
            3.0
            * area2
        )
    )

    return np.array(
        [
            centroid_x,
            centroid_y,
        ],
        dtype=float,
    )


# =============================================================================
# Plane fitting
# =============================================================================


def fit_plane_and_boundary(
    scans_w: list[np.ndarray],
    *,
    sensor_origins_w: np.ndarray | None = None,
    normal_hint_w: np.ndarray | None = None,
) -> PlaneEstimate:
    """
    Fit calibration plane and observed convex-hull boundary.

    --------------------------------------------------------------------------
    Plane fitting
    --------------------------------------------------------------------------

    All laser points are transformed into the robot world/base frame.

    PCA/SVD is applied:

        centered_points = points - mean(points)

    The smallest right-singular vector gives the plane normal.

    --------------------------------------------------------------------------
    Plane-normal sign
    --------------------------------------------------------------------------

    PCA gives only an unoriented plane:

        n  <->  -n

    If sensor_origins_w is supplied, the sign is selected such that:

        n^T (sensor_origin - plane_point) > 0

    for the side containing the sensors.

    Therefore +normal points from the calibration board toward the sensor side.

    No arbitrary +World-Z convention is used.

    --------------------------------------------------------------------------
    Center
    --------------------------------------------------------------------------

    The PCA point mean is used ONLY internally to define the fitted plane.

    The final returned centroid_w is:

        all world laser points
            ->
        project onto fitted plane
            ->
        express in plane UV
            ->
        convex hull
            ->
        geometric AREA centroid of hull
            ->
        convert back to world coordinates

    Thus centroid_w is the geometric center of the OBSERVED convex-hull area.

    It is not necessarily the true physical board center unless the acquired
    initial profiles sufficiently cover the physical board.
    """

    # =========================================================================
    # Validate scans
    # =========================================================================

    if len(scans_w) < 2:

        raise RuntimeError(
            "one laser profile is only a 3D line; "
            "capture at least two non-collinear scans"
        )

    points_w = np.vstack(
        scans_w
    )

    if len(points_w) < 3:

        raise RuntimeError(
            "not enough world points"
        )

    if not np.all(
        np.isfinite(
            points_w
        )
    ):
        raise ValueError(
            "world points contain NaN or Inf"
        )

    # =========================================================================
    # PCA/SVD plane fitting
    #
    # IMPORTANT:
    #
    # fit_centroid_w is NOT the final board center.
    #
    # It is only the reference point used for fitting.
    # =========================================================================

    fit_centroid_w = np.mean(
        points_w,
        axis=0,
    )

    centered = (
        points_w
        - fit_centroid_w
    )

    _, singular_values, Vt = (
        np.linalg.svd(
            centered,
            full_matrices=False,
        )
    )

    if singular_values[1] <= 1e-9:

        raise RuntimeError(
            "all captured profiles are collinear; "
            "change sensor pose"
        )

    # -------------------------------------------------------------------------
    # PCA basis
    # -------------------------------------------------------------------------

    basis_u = (
        Vt[0].copy()
    )

    basis_v = (
        Vt[1].copy()
    )

    normal = (
        Vt[2].copy()
    )

    # =========================================================================
    # Right-handed frame
    #
    # u x v = n
    # =========================================================================

    if np.dot(
        np.cross(
            basis_u,
            basis_v,
        ),
        normal,
    ) < 0.0:

        basis_v *= -1.0

    # =========================================================================
    # Resolve normal sign using SENSOR SIDE
    # =========================================================================

    if sensor_origins_w is not None:

        sensor_origins = np.asarray(
            sensor_origins_w,
            dtype=float,
        )

        if (
            sensor_origins.ndim != 2
            or sensor_origins.shape[1] != 3
        ):
            raise ValueError(
                "sensor_origins_w must have shape (N, 3)"
            )

        if len(sensor_origins) == 0:

            raise ValueError(
                "sensor_origins_w is empty"
            )

        if not np.all(
            np.isfinite(
                sensor_origins
            )
        ):
            raise ValueError(
                "sensor_origins_w contains NaN or Inf"
            )

        # ---------------------------------------------------------------------
        # Signed position of each sensor relative to fitted plane.
        #
        # > 0 : sensor is on +normal side
        # < 0 : sensor is on -normal side
        # ---------------------------------------------------------------------

        sensor_side_scores = (
            sensor_origins
            - fit_centroid_w[None, :]
        ) @ normal

        # Median makes sign selection robust to one strange capture.
        side_score = float(
            np.median(
                sensor_side_scores
            )
        )

        if abs(
            side_score
        ) < 1e-9:

            raise RuntimeError(
                "cannot determine plane-normal direction "
                "from sensor positions: "
                "sensor origins are approximately on the fitted plane"
            )

        # ---------------------------------------------------------------------
        # Flip if sensors currently lie on -normal side.
        # ---------------------------------------------------------------------

        if side_score < 0.0:

            normal *= -1.0

            # Preserve:
            #
            #     u x v = n
            #
            basis_v *= -1.0

    # =========================================================================
    # Optional fallback normal hint
    #
    # This is used only when sensor origins are unavailable.
    # =========================================================================

    elif normal_hint_w is not None:

        hint = np.asarray(
            normal_hint_w,
            dtype=float,
        )

        norm = float(
            np.linalg.norm(
                hint
            )
        )

        if norm <= 0.0:

            raise ValueError(
                "normal hint must be nonzero"
            )

        hint /= norm

        if np.dot(
            normal,
            hint,
        ) < 0.0:

            normal *= -1.0
            basis_v *= -1.0

    # =========================================================================
    # IMPORTANT:
    #
    # No:
    #
    #     if normal[2] < 0:
    #         normal *= -1
    #
    # World Z has no physical meaning for board front/back.
    # =========================================================================

    # =========================================================================
    # Point-to-plane distances
    # =========================================================================

    signed_distances = (
        centered
        @ normal
    )

    # =========================================================================
    # Project every laser point onto fitted plane
    # =========================================================================

    projected_w = (
        points_w
        - signed_distances[:, None]
        * normal[None, :]
    )

    # =========================================================================
    # Temporary 2D plane coordinates
    #
    # Temporary origin = PCA fit centroid
    # =========================================================================

    projected_centered = (
        projected_w
        - fit_centroid_w
    )

    points_uv = np.column_stack(
        (
            projected_centered
            @ basis_u,

            projected_centered
            @ basis_v,
        )
    )

    # =========================================================================
    # Convex hull of all observed projected laser points
    # =========================================================================

    boundary_uv_temp = (
        convex_hull_2d(
            points_uv
        )
    )

    # =========================================================================
    # Geometric AREA centroid of convex hull
    #
    # This is robust to point-sampling density compared with mean(points).
    # =========================================================================

    hull_centroid_uv = (
        polygon_centroid_2d(
            boundary_uv_temp
        )
    )

    # =========================================================================
    # Hull centroid -> world coordinates
    # =========================================================================

    centroid_w = (
        fit_centroid_w
        + hull_centroid_uv[0]
        * basis_u
        + hull_centroid_uv[1]
        * basis_v
    )

    # =========================================================================
    # Re-center UV coordinates at geometric hull centroid
    #
    # Therefore:
    #
    #   plane frame origin = centroid_w
    #   hull centroid       = (0, 0)
    # =========================================================================

    boundary_uv = (
        boundary_uv_temp
        - hull_centroid_uv[None, :]
    )

    # =========================================================================
    # Hull -> world
    # =========================================================================

    boundary_w = (
        centroid_w[None, :]
        + boundary_uv[:, [0]]
        * basis_u[None, :]
        + boundary_uv[:, [1]]
        * basis_v[None, :]
    )

    # =========================================================================
    # Final frame sanity check
    # =========================================================================

    handedness = float(
        np.dot(
            np.cross(
                basis_u,
                basis_v,
            ),
            normal,
        )
    )

    if handedness < 0.999:

        raise RuntimeError(
            "generated plane frame is not right-handed"
        )

    # =========================================================================
    # Plane equation
    #
    # normal dot p = offset
    # =========================================================================

    offset_w = float(
        np.dot(
            normal,
            centroid_w,
        )
    )

    # =========================================================================
    # Result
    # =========================================================================

    return PlaneEstimate(

        centroid_w=(
            centroid_w
        ),

        normal_w=(
            normal
        ),

        offset_w=(
            offset_w
        ),

        basis_u_w=(
            basis_u
        ),

        basis_v_w=(
            basis_v
        ),

        boundary_uv=(
            boundary_uv
        ),

        boundary_w=(
            boundary_w
        ),

        rms_mm=float(
            np.sqrt(
                np.mean(
                    signed_distances**2
                )
            )
        ),

        max_abs_mm=float(
            np.max(
                np.abs(
                    signed_distances
                )
            )
        ),

        singular_values=(
            singular_values
        ),

        scan_count=(
            len(scans_w)
        ),

        point_count=(
            len(points_w)
        ),
    )


# =============================================================================
# Plot utilities
# =============================================================================


def set_axes_equal_3d(
    ax,
    all_points: np.ndarray,
) -> None:

    mins = np.min(
        all_points,
        axis=0,
    )

    maxs = np.max(
        all_points,
        axis=0,
    )

    centers = (
        0.5
        * (
            mins
            + maxs
        )
    )

    radius = (
        0.5
        * np.max(
            maxs
            - mins
        )
    )

    if (
        not np.isfinite(radius)
        or radius <= 0.0
    ):
        radius = 1.0

    ax.set_xlim(
        centers[0] - radius,
        centers[0] + radius,
    )

    ax.set_ylim(
        centers[1] - radius,
        centers[1] + radius,
    )

    ax.set_zlim(
        centers[2] - radius,
        centers[2] + radius,
    )


def plot_world_3d(
    scans_w: list[np.ndarray],
    estimate: PlaneEstimate,
    output_path: Path,
    *,
    show_plot: bool,
    axis_length_mm: float = 50.0,
) -> None:

    if (
        plt is None
        or Poly3DCollection is None
    ):
        raise RuntimeError(
            "matplotlib is required for 3D plotting"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # A plain Figure avoids creating a Tk GUI object from the workflow's
    # background estimation thread. Interactive CLI use still gets pyplot.
    fig = (
        plt.figure(figsize=(10, 8))
        if show_plot
        else Figure(figsize=(10, 8))
    )

    ax = fig.add_subplot(
        111,
        projection="3d",
    )

    # =========================================================================
    # Individual scans
    # =========================================================================

    for idx, scan in enumerate(
        scans_w,
        start=1,
    ):

        scan = np.asarray(
            scan,
            dtype=float,
        )

        ax.plot(
            scan[:, 0],
            scan[:, 1],
            scan[:, 2],
            linewidth=1.5,
            label=f"scan {idx}",
        )

    # =========================================================================
    # All points
    # =========================================================================

    all_points_w = np.vstack(
        scans_w
    )

    ax.scatter(
        all_points_w[:, 0],
        all_points_w[:, 1],
        all_points_w[:, 2],
        s=4,
        alpha=0.25,
        label="all world points",
    )

    # =========================================================================
    # Convex hull
    # =========================================================================

    boundary_closed = np.vstack(
        [
            estimate.boundary_w,
            estimate.boundary_w[0],
        ]
    )

    ax.plot(
        boundary_closed[:, 0],
        boundary_closed[:, 1],
        boundary_closed[:, 2],
        linewidth=2.5,
        label="observed convex hull",
    )

    polygon = Poly3DCollection(
        [
            estimate.boundary_w
        ],
        alpha=0.25,
    )

    ax.add_collection3d(
        polygon
    )

    # =========================================================================
    # Hull area centroid
    # =========================================================================

    c = estimate.centroid_w

    ax.scatter(
        [c[0]],
        [c[1]],
        [c[2]],
        s=60,
        marker="o",
        label="hull area centroid",
    )

    # =========================================================================
    # Plane frame
    # =========================================================================

    ax.quiver(
        c[0],
        c[1],
        c[2],

        estimate.basis_u_w[0],
        estimate.basis_u_w[1],
        estimate.basis_u_w[2],

        length=axis_length_mm,
        normalize=True,
        linewidth=2.0,
    )

    ax.quiver(
        c[0],
        c[1],
        c[2],

        estimate.basis_v_w[0],
        estimate.basis_v_w[1],
        estimate.basis_v_w[2],

        length=axis_length_mm,
        normalize=True,
        linewidth=2.0,
    )

    ax.quiver(
        c[0],
        c[1],
        c[2],

        estimate.normal_w[0],
        estimate.normal_w[1],
        estimate.normal_w[2],

        length=axis_length_mm,
        normalize=True,
        linewidth=2.5,
    )

    # =========================================================================
    # Axis labels
    # =========================================================================

    label_u = (
        c
        + axis_length_mm
        * estimate.basis_u_w
    )

    label_v = (
        c
        + axis_length_mm
        * estimate.basis_v_w
    )

    label_n = (
        c
        + axis_length_mm
        * estimate.normal_w
    )

    ax.text(
        label_u[0],
        label_u[1],
        label_u[2],
        "u",
    )

    ax.text(
        label_v[0],
        label_v[1],
        label_v[2],
        "v",
    )

    ax.text(
        label_n[0],
        label_n[1],
        label_n[2],
        "n (toward sensor)",
    )

    # =========================================================================
    # Equal axes
    # =========================================================================

    axes_reference = np.vstack(
        [
            all_points_w,
            estimate.boundary_w,
            c[None, :],
            label_u[None, :],
            label_v[None, :],
            label_n[None, :],
        ]
    )

    set_axes_equal_3d(
        ax,
        axes_reference,
    )

    ax.set_xlabel(
        "World X [mm]"
    )

    ax.set_ylabel(
        "World Y [mm]"
    )

    ax.set_zlabel(
        "World Z [mm]"
    )

    ax.set_title(
        "Estimated plane and observed profile lines\n"
        f"scans={estimate.scan_count}, "
        f"points={estimate.point_count}, "
        f"RMS={estimate.rms_mm:.4f} mm"
    )

    ax.legend(
        loc="best"
    )

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=180,
    )

    print(
        f"saved 3D plot to {output_path}"
    )

    if show_plot:
        plt.show()
    else:
        plt.close(
            fig
        )


# =============================================================================
# Save estimate
# =============================================================================


def save_estimate(
    estimate: PlaneEstimate,
    output_dir: Path,
    capture_paths: list[Path],
) -> None:

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # =========================================================================
    # Hull world coordinates
    # =========================================================================

    np.savetxt(
        output_dir
        / "plane_boundary_world.csv",

        estimate.boundary_w,

        delimiter=",",

        header=(
            "x_mm,y_mm,z_mm"
        ),

        comments="",
    )

    # =========================================================================
    # Hull plane coordinates
    #
    # Origin = hull area centroid
    # =========================================================================

    np.savetxt(
        output_dir
        / "plane_boundary_uv.csv",

        estimate.boundary_uv,

        delimiter=",",

        header=(
            "u_mm,v_mm"
        ),

        comments="",
    )

    # =========================================================================
    # Plane frame
    #
    # ^W T_P
    #
    # origin = convex-hull area centroid
    # z      = normal toward sensors
    # =========================================================================

    frame = np.eye(
        4
    )

    frame[:3, 0] = (
        estimate.basis_u_w
    )

    frame[:3, 1] = (
        estimate.basis_v_w
    )

    frame[:3, 2] = (
        estimate.normal_w
    )

    frame[:3, 3] = (
        estimate.centroid_w
    )

    np.savetxt(
        output_dir
        / "T_world_plane.csv",

        frame,

        delimiter=",",

        fmt="%.12g",
    )

    # =========================================================================
    # Metadata
    # =========================================================================

    atomic_json(
        output_dir
        / "plane_estimate.json",

        {
            "coordinate_frame": (
                "world/base"
            ),

            "center_definition": (
                "geometric area centroid of convex hull "
                "of projected observed laser points"
            ),

            "normal_definition": (
                "plane normal oriented toward captured sensor origins"
            ),

            "plane_equation": (
                "normal_w dot p_w = offset_w"
            ),

            "normal_w": (
                estimate.normal_w.tolist()
            ),

            "offset_w_mm": (
                estimate.offset_w
            ),

            "centroid_w_mm": (
                estimate.centroid_w.tolist()
            ),

            "basis_u_w": (
                estimate.basis_u_w.tolist()
            ),

            "basis_v_w": (
                estimate.basis_v_w.tolist()
            ),

            "rms_point_to_plane_mm": (
                estimate.rms_mm
            ),

            "max_abs_point_to_plane_mm": (
                estimate.max_abs_mm
            ),

            "singular_values": (
                estimate.singular_values.tolist()
            ),

            "scan_count": (
                estimate.scan_count
            ),

            "point_count": (
                estimate.point_count
            ),

            "boundary_vertex_count": (
                len(
                    estimate.boundary_w
                )
            ),

            "boundary_area_mm2": (
                polygon_area(
                    estimate.boundary_uv
                )
            ),

            "boundary_world_csv": (
                "plane_boundary_world.csv"
            ),

            "boundary_uv_csv": (
                "plane_boundary_uv.csv"
            ),

            "T_world_plane_csv": (
                "T_world_plane.csv"
            ),

            "world_plot_png": (
                "plane_estimate_3d.png"
            ),

            "captures": [
                str(path)
                for path in capture_paths
            ],

            "estimated_at": (
                datetime.now(
                    timezone.utc
                ).isoformat()
            ),
        },
    )


# =============================================================================
# Estimate plane from captured dataset
# =============================================================================


def estimate_from_dataset(
    args: argparse.Namespace,
) -> PlaneEstimate:

    # =========================================================================
    # Load scans AND capture-time sensor origins
    # =========================================================================

    (
        scans_w,
        sensor_origins_w,
        paths,
    ) = load_world_points(
        args.dataset_dir,
        args.T_tcp_sensor,
    )

    # =========================================================================
    # Optional fallback normal hint
    #
    # Normally unnecessary because sensor origins resolve the sign.
    # =========================================================================

    normal_hint = None

    if (
        args.normal_hint_world
        is not None
    ):

        normal_hint = np.asarray(
            args.normal_hint_world,
            dtype=float,
        )

    # =========================================================================
    # Fit
    # =========================================================================

    estimate = (
        fit_plane_and_boundary(

            scans_w,

            sensor_origins_w=(
                sensor_origins_w
            ),

            normal_hint_w=(
                normal_hint
            ),
        )
    )

    # =========================================================================
    # Save
    # =========================================================================

    save_estimate(
        estimate,
        args.estimate_dir,
        paths,
    )

    # =========================================================================
    # Plot
    # =========================================================================

    plot_world_3d(
        scans_w,

        estimate,

        args.estimate_dir
        / "plane_estimate_3d.png",

        show_plot=(
            args.show_plot
        ),

        axis_length_mm=(
            args.plot_axis_length_mm
        ),
    )

    # =========================================================================
    # Verify sensor side
    #
    # Since +normal must point toward sensors:
    #
    #     n^T (p_sensor - p_plane) > 0
    # =========================================================================

    sensor_signed_distance_mm = (
        sensor_origins_w
        - estimate.centroid_w[
            None,
            :
        ]
    ) @ estimate.normal_w

    # =========================================================================
    # Print
    # =========================================================================

    print(
        "\nPlane estimate"
    )

    print(
        f"  scans / points  : "
        f"{estimate.scan_count} / "
        f"{estimate.point_count}"
    )

    print(
        "  hull centroid_w : "
        f"{np.array2string(estimate.centroid_w, precision=8)}"
    )

    print(
        "  normal_w        : "
        f"{np.array2string(estimate.normal_w, precision=8)}"
    )

    print(
        f"  offset_w [mm]   : "
        f"{estimate.offset_w:.6f}"
    )

    print(
        f"  RMS [mm]        : "
        f"{estimate.rms_mm:.6f}"
    )

    print(
        f"  max abs [mm]    : "
        f"{estimate.max_abs_mm:.6f}"
    )

    print(
        f"  hull vertices   : "
        f"{len(estimate.boundary_w)}"
    )

    print(
        f"  hull area [mm2] : "
        f"{polygon_area(estimate.boundary_uv):.3f}"
    )

    # -------------------------------------------------------------------------
    # This should normally be positive.
    # -------------------------------------------------------------------------

    print(
        "  sensor side [mm]: "
        f"min={np.min(sensor_signed_distance_mm):.3f}, "
        f"median={np.median(sensor_signed_distance_mm):.3f}, "
        f"max={np.max(sensor_signed_distance_mm):.3f}"
    )

    if (
        np.median(
            sensor_signed_distance_mm
        )
        <= 0.0
    ):

        raise RuntimeError(
            "internal error: estimated normal does not point "
            "toward sensor side"
        )

    print(
        f"saved estimate to "
        f"{args.estimate_dir}"
    )

    return estimate


# =============================================================================
# Hardware connection
# =============================================================================


def connect_hardware(
    args: argparse.Namespace,
) -> tuple[
    RobotAdapter,
    LaserAdapter,
]:

    try:
        from .robot_adapter import RobotAdapter
    except ImportError:
        from real_laser_handeye.robot_adapter import RobotAdapter

    robot = RobotAdapter(
        args.robot_host,
        args.robot_port,
    )

    laser = LaserAdapter(
        ip=(
            args.laser_ip
        ),

        control_port=(
            args.laser_control_port
        ),

        high_speed_port=(
            args.laser_high_speed_port
        ),

        batch_profiles=(
            args.batch_profiles
        ),

        aggregate=(
            args.aggregate
        ),
    )

    robot.connect()

    try:

        laser.connect()

    except BaseException:

        robot.close()
        raise

    return (
        robot,
        laser,
    )


# =============================================================================
# Single capture
# =============================================================================


def run_capture(
    args: argparse.Namespace,
) -> None:

    robot, laser = (
        connect_hardware(
            args
        )
    )

    try:

        capture_once(
            robot,
            laser,
            args,
        )

    finally:

        laser.close()
        robot.close()


# =============================================================================
# Interactive session
# =============================================================================


def run_session(
    args: argparse.Namespace,
) -> None:

    robot, laser = (
        connect_hardware(
            args
        )
    )

    print(
        "Initial plane capture session"
    )

    print(
        "c + Enter: capture | "
        "p + Enter: estimate plane/boundary/3D plot | "
        "q + Enter: quit"
    )

    print(
        "> ",
        end="",
        flush=True,
    )

    try:

        running = True

        while running:

            readable, _, _ = (
                select.select(
                    [sys.stdin],
                    [],
                    [],
                    0.1,
                )
            )

            if not readable:
                continue

            command = (
                sys.stdin.readline()
                .strip()
                .lower()
            )

            try:

                if command in (
                    "",
                    "c",
                    "capture",
                ):

                    capture_once(
                        robot,
                        laser,
                        args,
                    )

                elif command in (
                    "p",
                    "plane",
                    "estimate",
                ):

                    estimate_from_dataset(
                        args
                    )

                elif command in (
                    "q",
                    "quit",
                    "exit",
                ):

                    running = False

                else:

                    print(
                        "unknown command: c, p, q"
                    )

            except Exception as exc:

                print(
                    f"ERROR: "
                    f"{type(exc).__name__}: "
                    f"{exc}"
                )

            if running:

                print(
                    "> ",
                    end="",
                    flush=True,
                )

    finally:

        laser.close()
        robot.close()


# =============================================================================
# Arguments
# =============================================================================


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Capture initial laser profiles and estimate "
            "a world-frame calibration plane boundary"
        )
    )

    parser.add_argument(
        "command",
        nargs="?",
        choices=(
            "session",
            "capture",
            "estimate",
        ),
        default="session",
    )

    # =========================================================================
    # Paths
    # =========================================================================

    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path(
            "runs/real/initial_plane/dataset"
        ),
    )

    parser.add_argument(
        "--estimate-dir",
        type=Path,
        default=Path(
            "runs/real/initial_plane/estimate"
        ),
    )

    parser.add_argument(
        "--handeye",
        type=Path,
        default=Path(
            "real_laser_handeye/"
            "initial_T_tcp_sensor.json"
        ),
        help=(
            "Current ^TCP T_S hand-eye transform"
        ),
    )

    # =========================================================================
    # Robot
    # =========================================================================

    parser.add_argument(
        "--robot-host",
        default="192.168.0.10",
    )

    parser.add_argument(
        "--robot-port",
        type=int,
    )

    # =========================================================================
    # Laser
    # =========================================================================

    parser.add_argument(
        "--laser-ip",
        default="192.168.1.1",
    )

    parser.add_argument(
        "--laser-control-port",
        type=int,
        default=24691,
    )

    parser.add_argument(
        "--laser-high-speed-port",
        type=int,
        default=24692,
    )

    parser.add_argument(
        "--batch-profiles",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--aggregate",
        choices=(
            "median",
            "latest",
        ),
        default="median",
    )

    parser.add_argument(
        "--timeout-s",
        type=float,
        default=3.0,
    )

    # =========================================================================
    # Profile filtering
    # =========================================================================

    parser.add_argument(
        "--min-points",
        type=int,
        default=50,
    )

    parser.add_argument(
        "--max-abs-sensor-y-mm",
        type=float,
        default=0.1,
    )

    parser.add_argument(
        "--min-sensor-x-mm",
        type=float,
    )

    parser.add_argument(
        "--max-sensor-x-mm",
        type=float,
    )

    parser.add_argument(
        "--min-sensor-z-mm",
        type=float,
    )

    parser.add_argument(
        "--max-sensor-z-mm",
        type=float,
    )

    # =========================================================================
    # Capture stationarity
    # =========================================================================

    parser.add_argument(
        "--max-stationarity-translation-mm",
        type=float,
        default=0.2,
    )

    parser.add_argument(
        "--max-stationarity-rotation-deg",
        type=float,
        default=0.2,
    )

    # =========================================================================
    # Normal hint
    #
    # Usually no longer needed.
    #
    # Sensor origins are the PRIMARY rule for selecting normal sign.
    # =========================================================================

    parser.add_argument(
        "--normal-hint-world",
        type=float,
        nargs=3,
        metavar=(
            "NX",
            "NY",
            "NZ",
        ),
        help=(
            "Fallback vector used to choose plane-normal sign "
            "only when sensor origins are unavailable"
        ),
    )

    # =========================================================================
    # Plot
    # =========================================================================

    parser.add_argument(
        "--plot-axis-length-mm",
        type=float,
        default=50.0,
        help=(
            "Axis length for the plotted plane frame"
        ),
    )

    parser.set_defaults(
        show_plot=True
    )

    parser.add_argument(
        "--show-plot",
        dest="show_plot",
        action="store_true",
        help=(
            "Show the world-coordinate 3D plot "
            "after estimation (default)"
        ),
    )

    parser.add_argument(
        "--no-show-plot",
        dest="show_plot",
        action="store_false",
        help=(
            "Only save the 3D plot PNG "
            "without opening a window"
        ),
    )

    # =========================================================================
    # Parse
    # =========================================================================

    args = (
        parser.parse_args()
    )

    args.T_tcp_sensor = (
        load_transform(
            args.handeye
        )
    )

    return args


# =============================================================================
# Main
# =============================================================================


def main() -> None:

    args = parse_args()

    if args.command == "capture":

        run_capture(
            args
        )

    elif args.command == "estimate":

        estimate_from_dataset(
            args
        )

    else:

        run_session(
            args
        )


if __name__ == "__main__":
    main()
