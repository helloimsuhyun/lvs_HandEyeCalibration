"""
surface_aware_harris3d.py
=========================

python surface_aware_harris3d.py \
    --input /home/choisuhyun/lvs_HandEyeCalibration/pose_estimation/data/part.stl \
    --voxel 0.3 \
    --normal-radius 3.0 \
    --spatial-sigma-ratio 0.5 \
    --sigma-e-deg 15 \
    --sigma-r 0.25 \
    --normal-max-iterations 10 \
    --noise-std 0.25 \
    --outlier-ratio 0.05 \
    --outlier-offset-min 0.3 \
    --outlier-offset-max 1.5 \
    --debug-view all

Minimal reusable module containing only:

1) Surface-aware iterative normal estimation
2) Gaussian-window Harris3D
3) Optional PCL-style corner refinement

Dependencies
------------
numpy
open3d
pclpybridge

All geometric parameters use the SAME length unit as the input point cloud.
If the point cloud is in millimeters, radius/sigma values are also in mm.

Typical usage
-------------
import surface_aware_harris3d as sah

normals = sah.estimate_surface_aware_normals(
    points,
    radius=3.0,
    sigma_e_deg=15.0,
    sigma_r=0.25,
    max_iterations=5,
)

corners, response = sah.gaussian_harris3d(
    points,
    normals,
    radius=4.0,
    sigma=1.5,
    threshold=1e-6,
    nonmax=True,
    refine=True,
    refine_max_iterations=5,
)

Or run the full pipeline:

result = sah.surface_aware_harris3d(
    points,
    normal_radius=3.0,
    harris_radius=4.0,
    gaussian_sigma=1.5,
    refine=True,
)
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import open3d as o3d
import pclpybridge as pcl


__all__ = [
    "estimate_pca_normals",
    "estimate_surface_aware_normals",
    "gaussian_harris_response",
    "gaussian_harris3d",
    "refine_corners",
    "surface_aware_harris3d",

    # Debug / inspection API
    "inspect_surface_aware_point",
    "visualize_surface_aware_point",
    "pick_query_points",
    "run_surface_aware_debugger",
]


# =====================================================================
# Input helpers
# =====================================================================

def _as_points(points: Any) -> np.ndarray:
    """
    Convert NumPy array or Open3D legacy PointCloud to (N,3) float64.
    """
    if isinstance(points, o3d.geometry.PointCloud):
        x = np.asarray(points.points, dtype=np.float64)
    else:
        x = np.asarray(points, dtype=np.float64)

    if x.ndim != 2 or x.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")

    if len(x) == 0:
        raise ValueError("points must not be empty")

    if not np.all(np.isfinite(x)):
        raise ValueError("points contain NaN/Inf")

    return np.ascontiguousarray(x)


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)

    norm = np.linalg.norm(
        x,
        axis=1,
        keepdims=True,
    )

    out = np.full_like(
        x,
        np.nan,
        dtype=np.float64,
    )

    valid = np.squeeze(
        norm > 1e-12
    )

    out[valid] = (
        x[valid]
        / norm[valid]
    )

    return out


def _finite_rows(x: np.ndarray) -> np.ndarray:
    return np.all(
        np.isfinite(x),
        axis=1,
    )


def _build_tree(points: np.ndarray):
    cloud = o3d.geometry.PointCloud()

    cloud.points = (
        o3d.utility.Vector3dVector(
            points
        )
    )

    return o3d.geometry.KDTreeFlann(
        cloud
    )


def _unpack_pcl_normals(result):
    """
    Supports either pclpybridge API style:

        normals, curvature = pcl.estimate_normals(...)

    or raw pybind dict style:

        {"normals": ..., "curvature": ...}
    """
    if isinstance(result, dict):
        normals = result["normals"]
        curvature = result.get(
            "curvature",
            None,
        )
    else:
        normals, curvature = result

    return (
        np.asarray(
            normals,
            dtype=np.float64,
        ),
        curvature,
    )


# =====================================================================
# Ordinary PCA/PCL rough normals
# =====================================================================

def estimate_pca_normals(
    points: Any,
    radius: float,
    *,
    viewpoint=None,
    threads: int = 0,
) -> np.ndarray:
    """
    Estimate ordinary radius-PCA normals using pclpybridge/PCL.

    Parameters
    ----------
    points
        (N,3) NumPy array or Open3D legacy PointCloud.
    radius
        Normal estimation support radius.
    viewpoint
        Optional PCL normal orientation viewpoint.
    threads
        PCL NormalEstimationOMP thread count.

    Returns
    -------
    normals : (N,3) float64
    """
    xyz = _as_points(points)

    kwargs = {
        "radius": float(radius),
        "threads": int(threads),
    }

    if viewpoint is not None:
        kwargs["viewpoint"] = np.asarray(
            viewpoint,
            dtype=np.float32,
        )

    result = pcl.estimate_normals(
        np.asarray(
            xyz,
            dtype=np.float32,
        ),
        **kwargs,
    )

    normals, _ = _unpack_pcl_normals(
        result
    )

    return _normalize_rows(
        normals
    )


# =====================================================================
# Surface-aware normal estimator
# =====================================================================

def _weighted_pca_normal(
    neighbor_points: np.ndarray,
    weights: np.ndarray,
    reference_normal: np.ndarray,
):
    valid = (
        _finite_rows(
            neighbor_points
        )
        & np.isfinite(weights)
        & (weights > 1e-12)
    )

    q = neighbor_points[
        valid
    ]

    w = weights[
        valid
    ]

    if len(q) < 3:
        return None

    sw = float(
        np.sum(w)
    )

    if sw <= 1e-12:
        return None

    centroid = (
        np.sum(
            q * w[:, None],
            axis=0,
        )
        / sw
    )

    centered = (
        q - centroid
    )

    covariance = (
        (
            centered
            * w[:, None]
        ).T
        @ centered
        / sw
    )

    try:
        _, eigvec = np.linalg.eigh(
            covariance
        )
    except np.linalg.LinAlgError:
        return None

    normal = eigvec[:, 0]

    n = np.linalg.norm(
        normal
    )

    if (
        not np.isfinite(n)
        or n < 1e-12
    ):
        return None

    normal /= n

    if (
        reference_normal is not None
        and np.all(
            np.isfinite(
                reference_normal
            )
        )
        and np.dot(
            normal,
            reference_normal,
        ) < 0.0
    ):
        normal = -normal

    return normal


def _surface_aware_weights(
    query_point: np.ndarray,
    neighbor_points: np.ndarray,
    rough_neighbor_normals: np.ndarray,
    current_normal: np.ndarray,
    *,
    sigma_d: float,
    sigma_e_rad: float,
    sigma_r: float,
):
    """
    Current proposed soft compatibility weighting.

    d_i     = ||q_i - p||
    theta_i = angle(n_i^rough, n_current)

    kappa_hat = median(theta_i / d_i)

    e_i = max(
        0,
        theta_i - kappa_hat*d_i
    )

    w_d = exp(-d_i^2 / (2 sigma_d^2))
    w_e = 1 / (1 + (e_i/sigma_e)^2)
    w_r = 1 / (1 + (r_i/sigma_r)^2)

    w = w_d * w_e * w_r
    """
    delta = (
        neighbor_points
        - query_point
    )

    distance = np.linalg.norm(
        delta,
        axis=1,
    )

    local_normals = np.asarray(
        rough_neighbor_normals,
        dtype=np.float64,
    ).copy()

    valid_normal = _finite_rows(
        local_normals
    )

    if np.any(
        valid_normal
    ):
        aligned = local_normals[
            valid_normal
        ]

        sign = (
            aligned
            @ current_normal
        )

        aligned[
            sign < 0.0
        ] *= -1.0

        local_normals[
            valid_normal
        ] = aligned

    theta = np.full(
        len(neighbor_points),
        np.nan,
        dtype=np.float64,
    )

    if np.any(
        valid_normal
    ):
        dot = np.clip(
            local_normals[
                valid_normal
            ]
            @ current_normal,
            -1.0,
            1.0,
        )

        theta[
            valid_normal
        ] = np.arccos(
            dot
        )

    valid_ratio = (
        valid_normal
        & np.isfinite(theta)
        & (distance > 1e-9)
    )

    if np.sum(
        valid_ratio
    ) >= 3:
        kappa_hat = float(
            np.median(
                theta[
                    valid_ratio
                ]
                / distance[
                    valid_ratio
                ]
            )
        )
    else:
        kappa_hat = 0.0

    expected_theta = (
        kappa_hat
        * distance
    )

    theta_safe = np.where(
        np.isfinite(theta),
        theta,
        np.inf,
    )

    excess = np.maximum(
        0.0,
        theta_safe
        - expected_theta,
    )

    plane_residual = np.abs(
        delta
        @ current_normal
    )

    spatial_weight = np.exp(
        -(
            distance ** 2
        )
        / (
            2.0
            * sigma_d ** 2
        )
    )

    normal_field_weight = (
        1.0
        / (
            1.0
            + (
                excess
                / sigma_e_rad
            ) ** 2
        )
    )

    residual_weight = (
        1.0
        / (
            1.0
            + (
                plane_residual
                / sigma_r
            ) ** 2
        )
    )

    weight = (
        spatial_weight
        * normal_field_weight
        * residual_weight
    )

    weight[
        ~valid_normal
    ] = 0.0

    return weight


def estimate_surface_aware_normals(
    points: Any,
    radius: float,
    *,
    sigma_e_deg: float = 15.0,
    sigma_r: float = 0.25,
    spatial_sigma_ratio: float = 0.5,
    max_iterations: int = 5,
    convergence_deg: float = 0.05,
    min_neighbors: int = 5,
    rough_normals: np.ndarray | None = None,
    threads: int = 0,
    return_info: bool = False,
):
    """
    Estimate the current proposed surface-aware normals.

    Algorithm
    ---------
    1. Compute ordinary radius-PCA rough normals once.
    2. For each query point:
       - align rough neighbor normals to the current query normal;
       - estimate local expected smooth normal variation;
       - assign continuous Cauchy compatibility weights;
       - perform weighted PCA;
       - repeat until convergence or max_iterations.

    Parameters
    ----------
    points
        (N,3) NumPy array or Open3D legacy PointCloud.
    radius
        Neighborhood radius.
    sigma_e_deg
        Cauchy scale of normal-field excess angle [deg].
    sigma_r
        Cauchy scale of plane residual.
        Uses the same length unit as points.
    spatial_sigma_ratio
        sigma_d = spatial_sigma_ratio * radius.
    max_iterations
        Maximum weighted-PCA refinement iterations per point.
    convergence_deg
        Stop when successive normal change is below this angle.
    min_neighbors
        Minimum radius-neighborhood size.
    rough_normals
        Optional externally computed rough normals.
        If None, ordinary PCL radius-PCA normals are used.
    threads
        PCL thread count for initial rough normals.
    return_info
        If True, also return a dict with fallback/iteration statistics.

    Returns
    -------
    normals
        (N,3) surface-aware normals.

    or, if return_info=True:
        normals, info
    """
    xyz = _as_points(
        points
    )

    radius = float(
        radius
    )

    if radius <= 0:
        raise ValueError(
            "radius must be > 0"
        )

    if sigma_e_deg <= 0:
        raise ValueError(
            "sigma_e_deg must be > 0"
        )

    if sigma_r <= 0:
        raise ValueError(
            "sigma_r must be > 0"
        )

    if spatial_sigma_ratio <= 0:
        raise ValueError(
            "spatial_sigma_ratio must be > 0"
        )

    if max_iterations < 1:
        raise ValueError(
            "max_iterations must be >= 1"
        )

    if min_neighbors < 3:
        raise ValueError(
            "min_neighbors must be >= 3"
        )

    if rough_normals is None:
        rough = estimate_pca_normals(
            xyz,
            radius,
            threads=threads,
        )

    else:
        rough = _normalize_rows(
            np.asarray(
                rough_normals,
                dtype=np.float64,
            )
        )

        if rough.shape != xyz.shape:
            raise ValueError(
                "rough_normals must have shape (N,3)"
            )

    tree = _build_tree(
        xyz
    )

    output = np.full_like(
        xyz,
        np.nan,
        dtype=np.float64,
    )

    sigma_d = (
        spatial_sigma_ratio
        * radius
    )

    sigma_e_rad = math.radians(
        sigma_e_deg
    )

    fallback_count = 0

    iteration_count = np.zeros(
        len(xyz),
        dtype=np.int32,
    )

    for i, query in enumerate(
        xyz
    ):
        current = rough[
            i
        ].copy()

        if not np.all(
            np.isfinite(current)
        ):
            continue

        _, indices, _ = (
            tree.search_radius_vector_3d(
                query,
                radius,
            )
        )

        indices = np.asarray(
            indices,
            dtype=np.int64,
        )

        if len(
            indices
        ) < min_neighbors:
            output[i] = current
            fallback_count += 1
            continue

        neighbor_points = xyz[
            indices
        ]

        neighbor_rough = rough[
            indices
        ]

        success = False

        for iteration in range(
            max_iterations
        ):
            weights = (
                _surface_aware_weights(
                    query,
                    neighbor_points,
                    neighbor_rough,
                    current,
                    sigma_d=sigma_d,
                    sigma_e_rad=(
                        sigma_e_rad
                    ),
                    sigma_r=sigma_r,
                )
            )

            new_normal = (
                _weighted_pca_normal(
                    neighbor_points,
                    weights,
                    current,
                )
            )

            if new_normal is None:
                break

            dot = np.clip(
                abs(
                    float(
                        np.dot(
                            new_normal,
                            current,
                        )
                    )
                ),
                0.0,
                1.0,
            )

            change_deg = math.degrees(
                math.acos(dot)
            )

            current = new_normal
            success = True

            iteration_count[
                i
            ] = iteration + 1

            if (
                change_deg
                < convergence_deg
            ):
                break

        if success:
            output[i] = current

        else:
            output[i] = rough[i]
            fallback_count += 1

    # Final safety fallback.
    invalid = (
        ~_finite_rows(output)
        | (
            np.linalg.norm(
                np.nan_to_num(
                    output,
                    nan=0.0,
                ),
                axis=1,
            )
            < 1e-12
        )
    )

    output[
        invalid
    ] = rough[
        invalid
    ]

    fallback_count += int(
        np.sum(invalid)
    )

    output = _normalize_rows(
        output
    )

    if not return_info:
        return output

    info = {
        "rough_normals": rough,
        "fallback_count": int(
            fallback_count
        ),
        "iterations": iteration_count,
        "mean_iterations": float(
            np.mean(
                iteration_count
            )
        ),
        "max_iterations_used": int(
            np.max(
                iteration_count
            )
        ),
    }

    return (
        output,
        info,
    )


# =====================================================================
# Gaussian-window Harris3D
# =====================================================================

def gaussian_harris_response(
    points: Any,
    normals: np.ndarray,
    radius: float,
    sigma: float | None = None,
) -> np.ndarray:
    """
    Compute dense Gaussian-window Harris3D response.

    Weighted normal covariance:

        M(p) =
            sum_i w_i n_i n_i^T
            -------------------
                 sum_i w_i

        w_i = exp(
            -||q_i-p||^2
            /(2 sigma^2)
        )

    PCL-style HARRIS response:

        R = 0.04 + det(M)
            - 0.04 * trace(M)^2

    Parameters
    ----------
    points
        Input point cloud.
    normals
        (N,3) normal vectors.
    radius
        Harris neighborhood radius.
    sigma
        Gaussian window sigma.
        If None, sigma = 0.5 * radius.

    Returns
    -------
    response : (N,)
    """
    xyz = _as_points(
        points
    )

    normals = _normalize_rows(
        np.asarray(
            normals,
            dtype=np.float64,
        )
    )

    if normals.shape != xyz.shape:
        raise ValueError(
            "normals must have shape (N,3)"
        )

    radius = float(
        radius
    )

    if radius <= 0:
        raise ValueError(
            "radius must be > 0"
        )

    if sigma is None:
        sigma = 0.5 * radius

    sigma = float(
        sigma
    )

    if sigma <= 0:
        raise ValueError(
            "sigma must be > 0"
        )

    tree = _build_tree(
        xyz
    )

    response = np.zeros(
        len(xyz),
        dtype=np.float64,
    )

    for i, query in enumerate(
        xyz
    ):
        _, indices, dist2 = (
            tree.search_radius_vector_3d(
                query,
                radius,
            )
        )

        indices = np.asarray(
            indices,
            dtype=np.int64,
        )

        dist2 = np.asarray(
            dist2,
            dtype=np.float64,
        )

        if len(indices) == 0:
            continue

        local_normals = normals[
            indices
        ]

        valid = _finite_rows(
            local_normals
        )

        if not np.any(
            valid
        ):
            continue

        local_normals = local_normals[
            valid
        ]

        local_dist2 = dist2[
            valid
        ]

        weight = np.exp(
            -local_dist2
            / (
                2.0
                * sigma ** 2
            )
        )

        sw = float(
            np.sum(weight)
        )

        if sw <= 1e-12:
            continue

        M = (
            (
                local_normals
                * weight[:, None]
            ).T
            @ local_normals
            / sw
        )

        trace = float(
            np.trace(M)
        )

        determinant = float(
            np.linalg.det(M)
        )

        if (
            not np.isfinite(trace)
            or not np.isfinite(
                determinant
            )
        ):
            continue

        response[i] = (
            0.04
            + determinant
            - 0.04
            * trace
            * trace
        )

    return response


def _nonmax_suppression(
    points: np.ndarray,
    response: np.ndarray,
    radius: float,
    threshold: float,
):
    """
    PCL-like radius NMS.

    A point is suppressed when any radius-neighbor has strictly larger
    response. Equal-response plateaus are retained.
    """
    tree = _build_tree(
        points
    )

    keep = []

    for i, query in enumerate(
        points
    ):
        r = response[
            i
        ]

        if (
            not np.isfinite(r)
            or r < threshold
        ):
            continue

        _, indices, _ = (
            tree.search_radius_vector_3d(
                query,
                radius,
            )
        )

        indices = np.asarray(
            indices,
            dtype=np.int64,
        )

        if np.any(
            response[
                indices
            ] > r
        ):
            continue

        keep.append(i)

    return np.asarray(
        keep,
        dtype=np.int64,
    )


# =====================================================================
# PCL-style geometric corner refinement
# =====================================================================

def refine_corners(
    points: Any,
    normals: np.ndarray,
    corners: np.ndarray,
    radius: float,
    *,
    max_iterations: int = 10,
    rcond_threshold: float = 1e-4,
    squared_stop: float = 1e-6,
) -> np.ndarray:
    """
    Refine corner positions using the same tangent-plane intersection
    principle as PCL HarrisKeypoint3D::refineCorners().

    For neighboring point p_i and normal n_i:

        NNT  = sum n_i n_i^T
        NNTp = sum n_i n_i^T p_i

        x* = solve(NNT, NNTp)

    The search neighborhood is recomputed after each update.

    Parameters
    ----------
    points
        Original point cloud.
    normals
        Normals used by Harris/refinement.
    corners
        Initial corner locations, shape (K,3).
    radius
        Refinement neighborhood radius.
    max_iterations
        Maximum refinement updates.
    rcond_threshold
        Reject nearly singular tangent-plane systems.
    squared_stop
        Stop when squared movement is below this value.

    Returns
    -------
    refined : (K,3)
    """
    xyz = _as_points(
        points
    )

    normals = _normalize_rows(
        np.asarray(
            normals,
            dtype=np.float64,
        )
    )

    if normals.shape != xyz.shape:
        raise ValueError(
            "normals must have shape (N,3)"
        )

    corners = np.asarray(
        corners,
        dtype=np.float64,
    )

    if (
        corners.ndim != 2
        or corners.shape[1] != 3
    ):
        raise ValueError(
            "corners must have shape (K,3)"
        )

    if len(corners) == 0:
        return corners.copy()

    tree = _build_tree(
        xyz
    )

    refined = corners.copy()

    for corner_id in range(
        len(refined)
    ):
        current = refined[
            corner_id
        ].copy()

        for _ in range(
            max_iterations
        ):
            _, indices, _ = (
                tree.search_radius_vector_3d(
                    current,
                    radius,
                )
            )

            indices = np.asarray(
                indices,
                dtype=np.int64,
            )

            if len(indices) < 3:
                break

            NNT = np.zeros(
                (3, 3),
                dtype=np.float64,
            )

            NNTp = np.zeros(
                3,
                dtype=np.float64,
            )

            valid_count = 0

            for idx in indices:
                normal = normals[
                    idx
                ]

                if not np.all(
                    np.isfinite(normal)
                ):
                    continue

                outer = np.outer(
                    normal,
                    normal,
                )

                NNT += outer

                NNTp += (
                    outer
                    @ xyz[idx]
                )

                valid_count += 1

            if valid_count < 3:
                break

            try:
                condition_number = (
                    np.linalg.cond(
                        NNT
                    )
                )
            except np.linalg.LinAlgError:
                break

            if (
                not np.isfinite(
                    condition_number
                )
                or condition_number <= 0
            ):
                break

            reciprocal_condition = (
                1.0
                / condition_number
            )

            if (
                reciprocal_condition
                <= rcond_threshold
            ):
                break

            old = current.copy()

            try:
                current = np.linalg.solve(
                    NNT,
                    NNTp,
                )
            except np.linalg.LinAlgError:
                break

            movement2 = float(
                np.sum(
                    (
                        current
                        - old
                    ) ** 2
                )
            )

            if (
                movement2
                <= squared_stop
            ):
                break

        refined[
            corner_id
        ] = current

    return refined


# =====================================================================
# Complete Gaussian Harris detector
# =====================================================================

def gaussian_harris3d(
    points: Any,
    normals: np.ndarray,
    radius: float,
    *,
    sigma: float | None = None,
    threshold: float = 0.0,
    nonmax: bool = True,
    refine: bool = False,
    refine_max_iterations: int = 10,
    return_indices: bool = False,
    return_dense_response: bool = False,
):
    """
    Gaussian-window Harris3D detector.

    Parameters
    ----------
    points
        (N,3) NumPy array or Open3D legacy PointCloud.
    normals
        (N,3) normals.
    radius
        Harris support / NMS / refinement radius.
    sigma
        Gaussian sigma. Default = 0.5 * radius.
    threshold
        Harris response threshold.
    nonmax
        If False, return all points with dense response.
        If True, perform radius NMS.
    refine
        Apply PCL-style tangent-plane corner refinement after NMS.
    refine_max_iterations
        Maximum refinement iterations.
    return_indices
        Include original NMS point indices.
    return_dense_response
        Include full per-point Harris response.

    Returns
    -------
    Default:
        keypoints, keypoint_response

    If nonmax=False:
        points, dense_response

    Optional extra return values are appended depending on
    return_indices / return_dense_response.
    """
    xyz = _as_points(
        points
    )

    normals = _normalize_rows(
        np.asarray(
            normals,
            dtype=np.float64,
        )
    )

    if normals.shape != xyz.shape:
        raise ValueError(
            "normals must have shape (N,3)"
        )

    response = (
        gaussian_harris_response(
            xyz,
            normals,
            radius,
            sigma=sigma,
        )
    )

    if not nonmax:
        result = [
            xyz.copy(),
            response.copy(),
        ]

        if return_indices:
            result.append(
                np.arange(
                    len(xyz),
                    dtype=np.int64,
                )
            )

        if return_dense_response:
            result.append(
                response.copy()
            )

        return tuple(result)

    indices = (
        _nonmax_suppression(
            xyz,
            response,
            float(radius),
            float(threshold),
        )
    )

    keypoints = xyz[
        indices
    ].copy()

    keypoint_response = response[
        indices
    ].copy()

    if refine and len(
        keypoints
    ):
        keypoints = refine_corners(
            xyz,
            normals,
            keypoints,
            float(radius),
            max_iterations=(
                int(
                    refine_max_iterations
                )
            ),
        )

    result = [
        keypoints,
        keypoint_response,
    ]

    if return_indices:
        result.append(
            indices
        )

    if return_dense_response:
        result.append(
            response
        )

    return tuple(result)


# =====================================================================
# Convenience full pipeline
# =====================================================================

def surface_aware_harris3d(
    points: Any,
    *,
    normal_radius: float,
    harris_radius: float,
    gaussian_sigma: float | None = None,
    harris_threshold: float = 0.0,
    refine: bool = True,
    refine_max_iterations: int = 10,
    sigma_e_deg: float = 15.0,
    sigma_r: float = 0.25,
    spatial_sigma_ratio: float = 0.5,
    normal_max_iterations: int = 5,
    normal_convergence_deg: float = 0.05,
    min_neighbors: int = 5,
    threads: int = 0,
):
    """
    Convenience pipeline:

        points
        -> surface-aware normals
        -> Gaussian Harris3D
        -> optional PCL-style refinement

    Returns
    -------
    dict with:
        points
        normals
        keypoints
        response
        keypoint_indices
        dense_response
    """
    xyz = _as_points(
        points
    )

    normals = (
        estimate_surface_aware_normals(
            xyz,
            radius=normal_radius,
            sigma_e_deg=sigma_e_deg,
            sigma_r=sigma_r,
            spatial_sigma_ratio=(
                spatial_sigma_ratio
            ),
            max_iterations=(
                normal_max_iterations
            ),
            convergence_deg=(
                normal_convergence_deg
            ),
            min_neighbors=(
                min_neighbors
            ),
            threads=threads,
        )
    )

    (
        keypoints,
        keypoint_response,
        keypoint_indices,
        dense_response,
    ) = gaussian_harris3d(
        xyz,
        normals,
        radius=harris_radius,
        sigma=gaussian_sigma,
        threshold=harris_threshold,
        nonmax=True,
        refine=refine,
        refine_max_iterations=(
            refine_max_iterations
        ),
        return_indices=True,
        return_dense_response=True,
    )

    return {
        "points": xyz,
        "normals": normals,
        "keypoints": keypoints,
        "response": keypoint_response,
        "keypoint_indices": (
            keypoint_indices
        ),
        "dense_response": (
            dense_response
        ),
    }


# =====================================================================
# Debug / inspection tools
# =====================================================================

def _surface_aware_weight_diagnostics(
    query_point: np.ndarray,
    neighbor_points: np.ndarray,
    rough_neighbor_normals: np.ndarray,
    current_normal: np.ndarray,
    *,
    sigma_d: float,
    sigma_e_rad: float,
    sigma_r: float,
):
    """
    Compute the exact final weight used by `_surface_aware_weights()` plus
    its interpretable components.

    IMPORTANT
    ---------
    `weight` is obtained by calling the actual estimator function
    `_surface_aware_weights()`.  Therefore the heatmap displayed by the
    debugger is the SAME weight used by weighted PCA, not a separate
    visualization-only approximation.

    Returns
    -------
    dict containing:
        weight
        spatial_weight
        normal_field_weight
        residual_weight
        distance
        theta_deg
        expected_theta_deg
        excess_deg
        plane_residual
        kappa_hat_rad_per_unit
        effective_sample_size
    """
    query_point = np.asarray(
        query_point,
        dtype=np.float64,
    )

    neighbor_points = np.asarray(
        neighbor_points,
        dtype=np.float64,
    )

    rough_neighbor_normals = np.asarray(
        rough_neighbor_normals,
        dtype=np.float64,
    )

    current_normal = np.asarray(
        current_normal,
        dtype=np.float64,
    )

    # Exact weight used by the normal estimator.
    weight = _surface_aware_weights(
        query_point,
        neighbor_points,
        rough_neighbor_normals,
        current_normal,
        sigma_d=sigma_d,
        sigma_e_rad=sigma_e_rad,
        sigma_r=sigma_r,
    )

    delta = (
        neighbor_points
        - query_point
    )

    distance = np.linalg.norm(
        delta,
        axis=1,
    )

    local_normals = (
        rough_neighbor_normals
        .copy()
    )

    valid_normal = _finite_rows(
        local_normals
    )

    if np.any(valid_normal):
        aligned = local_normals[
            valid_normal
        ]

        sign = (
            aligned
            @ current_normal
        )

        aligned[
            sign < 0.0
        ] *= -1.0

        local_normals[
            valid_normal
        ] = aligned

    theta = np.full(
        len(neighbor_points),
        np.nan,
        dtype=np.float64,
    )

    if np.any(valid_normal):
        dot = np.clip(
            local_normals[
                valid_normal
            ]
            @ current_normal,
            -1.0,
            1.0,
        )

        theta[
            valid_normal
        ] = np.arccos(
            dot
        )

    valid_ratio = (
        valid_normal
        & np.isfinite(theta)
        & (distance > 1e-9)
    )

    if np.sum(
        valid_ratio
    ) >= 3:
        kappa_hat = float(
            np.median(
                theta[
                    valid_ratio
                ]
                / distance[
                    valid_ratio
                ]
            )
        )
    else:
        kappa_hat = 0.0

    expected_theta = (
        kappa_hat
        * distance
    )

    theta_safe = np.where(
        np.isfinite(theta),
        theta,
        np.inf,
    )

    excess = np.maximum(
        0.0,
        theta_safe
        - expected_theta,
    )

    plane_residual = np.abs(
        delta
        @ current_normal
    )

    spatial_weight = np.exp(
        -(
            distance ** 2
        )
        / (
            2.0
            * sigma_d ** 2
        )
    )

    normal_field_weight = (
        1.0
        / (
            1.0
            + (
                excess
                / sigma_e_rad
            ) ** 2
        )
    )

    residual_weight = (
        1.0
        / (
            1.0
            + (
                plane_residual
                / sigma_r
            ) ** 2
        )
    )

    spatial_weight[
        ~valid_normal
    ] = 0.0

    normal_field_weight[
        ~valid_normal
    ] = 0.0

    residual_weight[
        ~valid_normal
    ] = 0.0

    sw = float(
        np.sum(weight)
    )

    sw2 = float(
        np.sum(
            weight ** 2
        )
    )

    effective_sample_size = (
        sw * sw / sw2
        if sw2 > 1e-12
        else 0.0
    )

    return {
        "weight": weight,
        "spatial_weight": spatial_weight,
        "normal_field_weight": (
            normal_field_weight
        ),
        "residual_weight": (
            residual_weight
        ),
        "distance": distance,
        "theta_deg": np.degrees(
            theta
        ),
        "expected_theta_deg": (
            np.degrees(
                expected_theta
            )
        ),
        "excess_deg": np.degrees(
            excess
        ),
        "plane_residual": (
            plane_residual
        ),
        "kappa_hat_rad_per_unit": (
            kappa_hat
        ),
        "effective_sample_size": (
            effective_sample_size
        ),
    }


def inspect_surface_aware_point(
    points: Any,
    query_index: int,
    radius: float,
    *,
    sigma_e_deg: float = 15.0,
    sigma_r: float = 0.25,
    spatial_sigma_ratio: float = 0.5,
    max_iterations: int = 5,
    convergence_deg: float = 0.05,
    min_neighbors: int = 5,
    rough_normals: np.ndarray | None = None,
    threads: int = 0,
    injected_outlier_mask: np.ndarray | None = None,
):
    """
    Inspect one query point using exactly the same iterative logic as
    `estimate_surface_aware_normals()`.

    The returned `final_weight` is the weight vector that was ACTUALLY
    used in the LAST weighted-PCA update producing `final_normal`.

    This is the central debugging API.

    Parameters
    ----------
    points
        (N,3) point cloud.
    query_index
        Index of the query point to inspect.
    radius
        Surface-aware normal support radius.
    sigma_e_deg, sigma_r, spatial_sigma_ratio, max_iterations,
    convergence_deg, min_neighbors
        Same parameters as `estimate_surface_aware_normals`.
    rough_normals
        Optional precomputed ordinary PCA normals. Supplying them is useful
        when many points will be inspected interactively.

    Returns
    -------
    debug : dict
        query_index
        query_point
        neighborhood_indices
        neighborhood_points
        rough_query_normal
        final_normal
        final_weight
        final_components
        iteration_trace
        converged
        fallback
    """
    xyz = _as_points(
        points
    )

    if injected_outlier_mask is None:
        injected_outlier_mask = np.zeros(
            len(xyz),
            dtype=bool,
        )
    else:
        injected_outlier_mask = np.asarray(
            injected_outlier_mask,
            dtype=bool,
        )

        if injected_outlier_mask.shape != (
            len(xyz),
        ):
            raise ValueError(
                "injected_outlier_mask must have shape (N,)"
            )

    query_index = int(
        query_index
    )

    if (
        query_index < 0
        or query_index >= len(xyz)
    ):
        raise IndexError(
            f"query_index {query_index} is outside "
            f"[0, {len(xyz)-1}]"
        )

    radius = float(
        radius
    )

    if radius <= 0:
        raise ValueError(
            "radius must be > 0"
        )

    if rough_normals is None:
        rough = estimate_pca_normals(
            xyz,
            radius,
            threads=threads,
        )

    else:
        rough = _normalize_rows(
            np.asarray(
                rough_normals,
                dtype=np.float64,
            )
        )

        if rough.shape != xyz.shape:
            raise ValueError(
                "rough_normals must have shape (N,3)"
            )

    query = xyz[
        query_index
    ]

    rough_query = rough[
        query_index
    ].copy()

    tree = _build_tree(
        xyz
    )

    _, indices, _ = (
        tree.search_radius_vector_3d(
            query,
            radius,
        )
    )

    indices = np.asarray(
        indices,
        dtype=np.int64,
    )

    neighbor_points = xyz[
        indices
    ]

    neighbor_rough = rough[
        indices
    ]

    debug = {
        "query_index": query_index,
        "query_point": query.copy(),
        "radius": float(radius),
        "neighborhood_indices": (
            indices.copy()
        ),
        "neighborhood_points": (
            neighbor_points.copy()
        ),
        "neighborhood_injected_outlier_mask": (
            injected_outlier_mask[
                indices
            ].copy()
        ),
        "query_is_injected_outlier": bool(
            injected_outlier_mask[
                query_index
            ]
        ),
        "rough_query_normal": (
            rough_query.copy()
        ),
        "final_normal": (
            rough_query.copy()
        ),
        "final_weight": np.zeros(
            len(indices),
            dtype=np.float64,
        ),
        "final_components": None,
        "iteration_trace": [],
        "converged": False,
        "fallback": False,
        "rough_normals": rough,
    }

    if not np.all(
        np.isfinite(
            rough_query
        )
    ):
        debug[
            "fallback"
        ] = True

        return debug

    if len(
        indices
    ) < min_neighbors:
        debug[
            "fallback"
        ] = True

        return debug

    sigma_d = (
        spatial_sigma_ratio
        * radius
    )

    sigma_e_rad = (
        math.radians(
            sigma_e_deg
        )
    )

    current = (
        rough_query.copy()
    )

    last_diag = None

    for iteration in range(
        int(max_iterations)
    ):
        diag = (
            _surface_aware_weight_diagnostics(
                query,
                neighbor_points,
                neighbor_rough,
                current,
                sigma_d=sigma_d,
                sigma_e_rad=(
                    sigma_e_rad
                ),
                sigma_r=sigma_r,
            )
        )

        new_normal = (
            _weighted_pca_normal(
                neighbor_points,
                diag["weight"],
                current,
            )
        )

        if new_normal is None:
            debug[
                "fallback"
            ] = True

            break

        dot = np.clip(
            abs(
                float(
                    np.dot(
                        new_normal,
                        current,
                    )
                )
            ),
            0.0,
            1.0,
        )

        change_deg = (
            math.degrees(
                math.acos(
                    dot
                )
            )
        )

        trace_item = {
            "iteration": (
                iteration + 1
            ),
            "normal_before": (
                current.copy()
            ),
            "normal_after": (
                new_normal.copy()
            ),
            "change_deg": (
                float(
                    change_deg
                )
            ),
            "kappa_hat_rad_per_unit": (
                float(
                    diag[
                        "kappa_hat_rad_per_unit"
                    ]
                )
            ),
            "effective_sample_size": (
                float(
                    diag[
                        "effective_sample_size"
                    ]
                )
            ),
            "weight_sum": float(
                np.sum(
                    diag["weight"]
                )
            ),
            "weight_mean": float(
                np.mean(
                    diag["weight"]
                )
            ),
            "weight_max": float(
                np.max(
                    diag["weight"]
                )
            ),
        }

        debug[
            "iteration_trace"
        ].append(
            trace_item
        )

        # This diag is EXACTLY the weight used to produce new_normal.
        last_diag = diag

        current = (
            new_normal
        )

        if (
            change_deg
            < convergence_deg
        ):
            debug[
                "converged"
            ] = True

            break

    debug[
        "final_normal"
    ] = current.copy()

    if last_diag is not None:
        debug[
            "final_weight"
        ] = last_diag[
            "weight"
        ].copy()

        debug[
            "final_components"
        ] = last_diag

    return debug


def _debug_colormap(
    values: np.ndarray,
    *,
    vmin: float = 0.0,
    vmax: float = 1.0,
):
    """
    Compact Turbo-like RGB mapping without a matplotlib dependency.
    """
    x = np.asarray(
        values,
        dtype=np.float64,
    )

    scale = max(
        float(vmax)
        - float(vmin),
        1e-12,
    )

    x = np.clip(
        (
            x - float(vmin)
        ) / scale,
        0.0,
        1.0,
    )

    kr = np.array([
        0.13572138,
        4.61539260,
        -42.66032258,
        132.13108234,
        -152.94239396,
        59.28637943,
    ])

    kg = np.array([
        0.09140261,
        2.19418839,
        4.84296658,
        -14.18503333,
        4.27729857,
        2.82956604,
    ])

    kb = np.array([
        0.10667330,
        12.64194608,
        -60.58204836,
        110.36276771,
        -89.90310912,
        27.34824973,
    ])

    X = np.stack(
        [
            np.ones_like(x),
            x,
            x ** 2,
            x ** 3,
            x ** 4,
            x ** 5,
        ],
        axis=1,
    )

    rgb = np.c_[
        X @ kr,
        X @ kg,
        X @ kb,
    ]

    return np.clip(
        rgb,
        0.0,
        1.0,
    )


def _make_debug_normal_lines(
    query_point: np.ndarray,
    rough_normal: np.ndarray,
    final_normal: np.ndarray,
    length: float,
):
    """
    LineSet:
        line 0: rough PCA normal
        line 1: final surface-aware normal
    """
    query_point = np.asarray(
        query_point,
        dtype=np.float64,
    )

    rough_normal = np.asarray(
        rough_normal,
        dtype=np.float64,
    )

    final_normal = np.asarray(
        final_normal,
        dtype=np.float64,
    )

    points = [
        query_point,
        (
            query_point
            + length
            * rough_normal
        ),
        (
            query_point
            + length
            * final_normal
        ),
    ]

    lines = [
        [0, 1],
        [0, 2],
    ]

    # Rough normal = magenta, final normal = green.
    colors = [
        [0.90, 0.10, 0.90],
        [0.10, 0.90, 0.20],
    ]

    line_set = (
        o3d.geometry.LineSet()
    )

    line_set.points = (
        o3d.utility.Vector3dVector(
            points
        )
    )

    line_set.lines = (
        o3d.utility.Vector2iVector(
            lines
        )
    )

    line_set.colors = (
        o3d.utility.Vector3dVector(
            colors
        )
    )

    return line_set


def _make_radius_wire_sphere(
    center: np.ndarray,
    radius: float,
    *,
    circle_segments: int = 72,
    n_latitude_rings: int = 5,
    n_longitude_rings: int = 8,
    color=(0.15, 0.75, 1.0),
):
    """
    Build a lightweight wireframe sphere showing the exact support radius.

    The sphere is centered on the selected query point and has radius equal
    to the normal-estimation neighborhood radius.

    It uses latitude/longitude rings instead of a dense triangle mesh so
    the neighborhood points remain visible through the sphere.
    """
    center = np.asarray(
        center,
        dtype=np.float64,
    )

    radius = float(
        radius
    )

    if radius <= 0:
        raise ValueError(
            "radius must be > 0"
        )

    circle_segments = max(
        12,
        int(circle_segments),
    )

    n_latitude_rings = max(
        1,
        int(n_latitude_rings),
    )

    n_longitude_rings = max(
        3,
        int(n_longitude_rings),
    )

    vertices = []
    lines = []
    colors = []

    def add_closed_polyline(polyline):
        base = len(vertices)

        vertices.extend(
            polyline.tolist()
        )

        n = len(polyline)

        for i in range(n):
            lines.append([
                base + i,
                base + ((i + 1) % n),
            ])

            colors.append(
                list(color)
            )

    t = np.linspace(
        0.0,
        2.0 * np.pi,
        circle_segments,
        endpoint=False,
    )

    # Latitude rings, excluding degenerate poles.
    latitudes = np.linspace(
        -0.5 * np.pi,
        0.5 * np.pi,
        n_latitude_rings + 2,
    )[1:-1]

    for latitude in latitudes:
        z = (
            radius
            * np.sin(latitude)
        )

        ring_radius = (
            radius
            * np.cos(latitude)
        )

        ring = np.column_stack([
            ring_radius
            * np.cos(t),

            ring_radius
            * np.sin(t),

            np.full_like(
                t,
                z,
            ),
        ])

        ring += center

        add_closed_polyline(
            ring
        )

    # Longitude rings: vertical great circles around Z.
    longitudes = np.linspace(
        0.0,
        np.pi,
        n_longitude_rings,
        endpoint=False,
    )

    for longitude in longitudes:
        # Great circle in a vertical plane whose XY direction is longitude.
        direction_xy = np.array([
            np.cos(longitude),
            np.sin(longitude),
            0.0,
        ])

        ring = (
            center[None, :]
            + radius
            * (
                np.cos(t)[:, None]
                * direction_xy[None, :]
                + np.sin(t)[:, None]
                * np.array(
                    [0.0, 0.0, 1.0]
                )[None, :]
            )
        )

        add_closed_polyline(
            ring
        )

    sphere = (
        o3d.geometry.LineSet()
    )

    sphere.points = (
        o3d.utility.Vector3dVector(
            np.asarray(
                vertices,
                dtype=np.float64,
            )
        )
    )

    sphere.lines = (
        o3d.utility.Vector2iVector(
            np.asarray(
                lines,
                dtype=np.int32,
            )
        )
    )

    sphere.colors = (
        o3d.utility.Vector3dVector(
            np.asarray(
                colors,
                dtype=np.float64,
            )
        )
    )

    return sphere


def _make_normal_arrow(
    origin: np.ndarray,
    normal: np.ndarray,
    length: float,
    *,
    color=(0.10, 0.95, 0.20),
):
    """
    Create a 3D arrow aligned with `normal`.

    Open3D create_arrow() is initially aligned with +Z.
    The mesh is rotated from +Z onto the requested normal and then
    translated to the selected query point.
    """
    origin = np.asarray(
        origin,
        dtype=np.float64,
    )

    normal = np.asarray(
        normal,
        dtype=np.float64,
    )

    n = float(
        np.linalg.norm(normal)
    )

    if (
        not np.isfinite(n)
        or n < 1e-12
    ):
        return None

    direction = (
        normal / n
    )

    length = float(
        max(
            length,
            1e-6,
        )
    )

    cylinder_height = (
        0.72 * length
    )

    cone_height = (
        0.28 * length
    )

    cylinder_radius = (
        0.025 * length
    )

    cone_radius = (
        0.065 * length
    )

    arrow = (
        o3d.geometry.TriangleMesh
        .create_arrow(
            cylinder_radius=(
                cylinder_radius
            ),
            cone_radius=(
                cone_radius
            ),
            cylinder_height=(
                cylinder_height
            ),
            cone_height=(
                cone_height
            ),
            resolution=20,
            cylinder_split=4,
            cone_split=1,
        )
    )

    z_axis = np.array(
        [0.0, 0.0, 1.0],
        dtype=np.float64,
    )

    dot = float(
        np.clip(
            np.dot(
                z_axis,
                direction,
            ),
            -1.0,
            1.0,
        )
    )

    if dot > 1.0 - 1e-12:
        R = np.eye(3)

    elif dot < -1.0 + 1e-12:
        # 180 degree rotation around X maps +Z -> -Z.
        R = (
            o3d.geometry
            .get_rotation_matrix_from_axis_angle(
                np.array(
                    [np.pi, 0.0, 0.0],
                    dtype=np.float64,
                )
            )
        )

    else:
        axis = np.cross(
            z_axis,
            direction,
        )

        axis_norm = float(
            np.linalg.norm(
                axis
            )
        )

        axis /= max(
            axis_norm,
            1e-12,
        )

        angle = float(
            np.arccos(dot)
        )

        R = (
            o3d.geometry
            .get_rotation_matrix_from_axis_angle(
                axis * angle
            )
        )

    arrow.rotate(
        R,
        center=(
            np.zeros(3)
        ),
    )

    arrow.translate(
        origin
    )

    arrow.paint_uniform_color(
        list(color)
    )

    arrow.compute_vertex_normals()

    return arrow


def _make_query_marker(
    point: np.ndarray,
    radius: float,
):
    marker = (
        o3d.geometry.TriangleMesh
        .create_sphere(
            radius=float(radius)
        )
    )

    marker.translate(
        np.asarray(
            point,
            dtype=np.float64,
        )
    )

    marker.paint_uniform_color(
        [1.0, 0.15, 0.15]
    )

    marker.compute_vertex_normals()

    return marker


def _debug_scalar_for_view(
    debug: dict,
    view: str,
):
    components = debug[
        "final_components"
    ]

    if components is None:
        return (
            np.zeros(
                len(
                    debug[
                        "neighborhood_indices"
                    ]
                )
            ),
            0.0,
            1.0,
            "No valid weighted-PCA iteration",
        )

    if view == "weight":
        return (
            components[
                "weight"
            ],
            0.0,
            1.0,
            "Final PCA weight w = w_d * w_e * w_r",
        )

    if view == "spatial":
        return (
            components[
                "spatial_weight"
            ],
            0.0,
            1.0,
            "Spatial Gaussian weight w_d",
        )

    if view == "normal_field":
        return (
            components[
                "normal_field_weight"
            ],
            0.0,
            1.0,
            "Normal-field compatibility weight w_e",
        )

    if view == "residual_weight":
        return (
            components[
                "residual_weight"
            ],
            0.0,
            1.0,
            "Plane-residual compatibility weight w_r",
        )

    raise ValueError(
        "view must be one of: "
        "weight, spatial, normal_field, residual_weight"
    )


def _make_outlier_cross_markers(
    points: np.ndarray,
    size: float,
    *,
    color=(1.0, 1.0, 1.0),
):
    """
    Mark injected outliers with white 3D X/cross lines.

    The center heatmap point remains visible, so its actual weight color
    can still be inspected.
    """
    xyz = np.asarray(
        points,
        dtype=np.float64,
    )

    vertices = []
    lines = []
    colors = []

    s = float(
        size
    )

    directions = np.asarray(
        [
            [1.0, 1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 0.0, 1.0],
            [1.0, 0.0, -1.0],
            [0.0, 1.0, 1.0],
            [0.0, 1.0, -1.0],
        ],
        dtype=np.float64,
    )

    directions /= np.linalg.norm(
        directions,
        axis=1,
        keepdims=True,
    )

    for p in xyz:
        for d in directions:
            a = (
                p
                - 0.5
                * s
                * d
            )
            b = (
                p
                + 0.5
                * s
                * d
            )

            base = len(
                vertices
            )

            vertices.extend([
                a,
                b,
            ])

            lines.append([
                base,
                base + 1,
            ])

            colors.append(
                list(
                    color
                )
            )

    marker = (
        o3d.geometry.LineSet()
    )

    if len(vertices):
        marker.points = (
            o3d.utility.Vector3dVector(
                np.asarray(
                    vertices,
                    dtype=np.float64,
                )
            )
        )

        marker.lines = (
            o3d.utility.Vector2iVector(
                np.asarray(
                    lines,
                    dtype=np.int32,
                )
            )
        )

        marker.colors = (
            o3d.utility.Vector3dVector(
                np.asarray(
                    colors,
                    dtype=np.float64,
                )
            )
        )

    return marker


def visualize_surface_aware_point(
    points: Any,
    debug: dict,
    *,
    view: str = "weight",
    normal_length: float | None = None,
    query_marker_radius: float | None = None,
    point_size: float = 7.0,
    show_full_cloud: bool = True,
    window_name: str | None = None,
):
    """
    Open3D visualization for one inspected query point.

    Full cloud
        dark gray

    Points inside normal radius
        heatmap according to selected debug scalar

    Query point
        red sphere

    Normal support radius
        cyan wireframe sphere centered at the query point

    Rough PCA normal
        magenta line

    Final surface-aware normal
        green 3D arrow

    For the default `view="weight"`:
        blue/cyan -> low final PCA weight
        yellow/red -> high final PCA weight

    High final weight means the estimator considers that neighbor highly
    compatible with the query's local surface for the FINAL weighted PCA.
    """
    xyz = _as_points(
        points
    )

    indices = debug[
        "neighborhood_indices"
    ]

    local_points = xyz[
        indices
    ]

    (
        scalar,
        vmin,
        vmax,
        description,
    ) = _debug_scalar_for_view(
        debug,
        view,
    )

    local_colors = (
        _debug_colormap(
            scalar,
            vmin=vmin,
            vmax=vmax,
        )
    )

    geometries = []

    if show_full_cloud:
        full = (
            o3d.geometry.PointCloud()
        )

        full.points = (
            o3d.utility.Vector3dVector(
                xyz
            )
        )

        full.paint_uniform_color(
            [0.32, 0.32, 0.32]
        )

        geometries.append(
            full
        )

    local = (
        o3d.geometry.PointCloud()
    )

    local.points = (
        o3d.utility.Vector3dVector(
            local_points
        )
    )

    local.colors = (
        o3d.utility.Vector3dVector(
            local_colors
        )
    )

    geometries.append(
        local
    )

    # Scale markers using neighborhood radius estimate if not supplied.
    local_distance = np.linalg.norm(
        local_points
        - debug["query_point"],
        axis=1,
    )

    estimated_radius = (
        float(
            np.max(local_distance)
        )
        if len(local_distance)
        else 1.0
    )

    support_radius = float(
        debug.get(
            "radius",
            estimated_radius,
        )
    )

    # Exact normal-radius boundary.
    geometries.append(
        _make_radius_wire_sphere(
            debug[
                "query_point"
            ],
            support_radius,
        )
    )

    if query_marker_radius is None:
        query_marker_radius = max(
            0.03
            * support_radius,
            1e-4,
        )

    if normal_length is None:
        normal_length = max(
            0.65
            * support_radius,
            1e-4,
        )

    geometries.append(
        _make_query_marker(
            debug[
                "query_point"
            ],
            query_marker_radius,
        )
    )

    # Synthetic outliers are overlaid with WHITE X markers.
    # Their heatmap point color itself is not replaced.
    local_outlier_mask = debug.get(
        "neighborhood_injected_outlier_mask",
        None,
    )

    if local_outlier_mask is not None:
        local_outlier_mask = np.asarray(
            local_outlier_mask,
            dtype=bool,
        )

        if np.any(
            local_outlier_mask
        ):
            outlier_points = local_points[
                local_outlier_mask
            ]

            geometries.append(
                _make_outlier_cross_markers(
                    outlier_points,
                    size=max(
                        0.10
                        * support_radius,
                        1e-4,
                    ),
                    color=(
                        1.0,
                        1.0,
                        1.0,
                    ),
                )
            )

    # Rough PCA normal remains a thin magenta reference line.
    rough_line = (
        _make_debug_normal_lines(
            debug[
                "query_point"
            ],
            debug[
                "rough_query_normal"
            ],
            debug[
                "rough_query_normal"
            ],
            normal_length,
        )
    )

    # Keep only the first line in the LineSet so the final normal is shown
    # separately as a proper 3D arrow.
    rough_line.lines = (
        o3d.utility.Vector2iVector(
            np.asarray(
                [[0, 1]],
                dtype=np.int32,
            )
        )
    )

    rough_line.colors = (
        o3d.utility.Vector3dVector(
            np.asarray(
                [[0.90, 0.10, 0.90]],
                dtype=np.float64,
            )
        )
    )

    geometries.append(
        rough_line
    )

    final_arrow = (
        _make_normal_arrow(
            debug[
                "query_point"
            ],
            debug[
                "final_normal"
            ],
            normal_length,
            color=(
                0.10,
                0.95,
                0.20,
            ),
        )
    )

    if final_arrow is not None:
        geometries.append(
            final_arrow
        )

    if window_name is None:
        window_name = (
            f"Surface-aware debug | "
            f"point {debug['query_index']} | "
            f"{view}"
        )

    print(
        "\n"
        + "=" * 72
    )

    print(
        f"[DEBUG VIEW] {description}"
    )

    print(
        f"query index       : "
        f"{debug['query_index']}"
    )

    print(
        f"neighbors         : "
        f"{len(indices)}"
    )

    print(
        "heatmap           : "
        "blue/cyan=low, yellow/red=high"
    )

    print(
        "rough normal      : magenta"
    )

    print(
        "final normal      : green 3D arrow"
    )

    print(
        "query point       : red sphere"
    )

    local_outlier_mask = debug.get(
        "neighborhood_injected_outlier_mask",
        None,
    )

    if local_outlier_mask is not None:
        n_local_outlier = int(
            np.sum(
                local_outlier_mask
            )
        )

        print(
            "injected outlier : white X marker "
            f"({n_local_outlier} in this neighborhood)"
        )

    print(
        f"normal radius     : "
        f"{support_radius:.6g}"
    )

    print(
        "radius boundary   : cyan wireframe sphere"
    )

    print(
        "Close the window to continue."
    )

    vis = (
        o3d.visualization.Visualizer()
    )

    vis.create_window(
        window_name=window_name,
        width=1280,
        height=900,
    )

    for geometry in geometries:
        vis.add_geometry(
            geometry
        )

    option = (
        vis.get_render_option()
    )

    option.background_color = (
        np.array(
            [0.04, 0.04, 0.04]
        )
    )

    option.point_size = float(
        point_size
    )

    # Open3D support for line_width depends on the backend/platform,
    # but setting it is harmless when unsupported.
    try:
        option.line_width = 1.5
    except Exception:
        pass

    option.show_coordinate_frame = True

    vis.run()
    vis.destroy_window()


def _print_surface_aware_debug_summary(
    debug: dict,
):
    print(
        "\n"
        + "#" * 84
    )

    print(
        f" SURFACE-AWARE NORMAL DEBUG | "
        f"query={debug['query_index']}"
    )

    print(
        "#" * 84
    )

    print(
        f"query point    : "
        f"{debug['query_point']}"
    )

    print(
        f"neighbors      : "
        f"{len(debug['neighborhood_indices'])}"
    )

    print(
        f"rough normal   : "
        f"{debug['rough_query_normal']}"
    )

    print(
        f"final normal   : "
        f"{debug['final_normal']}"
    )

    print(
        f"converged      : "
        f"{debug['converged']}"
    )

    print(
        f"fallback       : "
        f"{debug['fallback']}"
    )

    weight = debug[
        "final_weight"
    ]

    if len(weight):
        sw = float(
            np.sum(weight)
        )

        sw2 = float(
            np.sum(
                weight ** 2
            )
        )

        ess = (
            sw * sw / sw2
            if sw2 > 1e-12
            else 0.0
        )

        print(
            f"weight min/max : "
            f"{np.min(weight):.4f} / "
            f"{np.max(weight):.4f}"
        )

        print(
            f"weight mean    : "
            f"{np.mean(weight):.4f}"
        )

        print(
            f"ESS            : "
            f"{ess:.2f} / "
            f"{len(weight)}"
        )

        print(
            "soft support    : "
            f"w>=0.75 {np.sum(weight >= 0.75)}, "
            f"w>=0.50 {np.sum(weight >= 0.50)}, "
            f"w>=0.25 {np.sum(weight >= 0.25)}, "
            f"w>=0.10 {np.sum(weight >= 0.10)}"
        )

        outlier_mask = debug.get(
            "neighborhood_injected_outlier_mask",
            None,
        )

        if outlier_mask is not None:
            outlier_mask = np.asarray(
                outlier_mask,
                dtype=bool,
            )

            inlier_mask = ~outlier_mask

            print(
                "\nInjected-outlier rejection"
            )

            print(
                "--------------------------"
            )

            print(
                f"local inliers     : "
                f"{int(np.sum(inlier_mask))}"
            )

            print(
                f"local outliers    : "
                f"{int(np.sum(outlier_mask))}"
            )

            if np.any(
                inlier_mask
            ):
                inlier_w = weight[
                    inlier_mask
                ]

                print(
                    f"inlier weight     : "
                    f"mean={np.mean(inlier_w):.4f}, "
                    f"median={np.median(inlier_w):.4f}"
                )

            if np.any(
                outlier_mask
            ):
                outlier_w = weight[
                    outlier_mask
                ]

                print(
                    f"outlier weight    : "
                    f"mean={np.mean(outlier_w):.4f}, "
                    f"median={np.median(outlier_w):.4f}"
                )

                print(
                    f"outlier w < 0.25 : "
                    f"{100.0*np.mean(outlier_w < 0.25):.1f}%"
                )

                print(
                    f"outlier w < 0.10 : "
                    f"{100.0*np.mean(outlier_w < 0.10):.1f}%"
                )

                if np.any(
                    inlier_mask
                ):
                    inlier_w = weight[
                        inlier_mask
                    ]

                    ratio = (
                        np.median(
                            outlier_w
                        )
                        / max(
                            np.median(
                                inlier_w
                            ),
                            1e-12,
                        )
                    )

                    print(
                        f"median out/in     : "
                        f"{ratio:.4f}"
                    )

    print(
        "\nIteration trace"
    )

    print(
        "iter  change(deg)  kappa(rad/unit)  ESS    "
        "sum(w)   mean(w)  max(w)"
    )

    print(
        "----  -----------  ---------------  -----  "
        "-------  -------  ------"
    )

    for item in debug[
        "iteration_trace"
    ]:
        print(
            f"{item['iteration']:>4d}  "
            f"{item['change_deg']:>11.5f}  "
            f"{item['kappa_hat_rad_per_unit']:>15.6f}  "
            f"{item['effective_sample_size']:>5.1f}  "
            f"{item['weight_sum']:>7.3f}  "
            f"{item['weight_mean']:>7.4f}  "
            f"{item['weight_max']:>6.4f}"
        )


def pick_query_points(
    points: Any,
    *,
    window_name: str = (
        "Pick query points: "
        "Shift+Left Click | Shift+Right Undo | Q Finish"
    ),
):
    """
    Open Open3D's point picker and return selected point indices.

    Controls
    --------
    Shift + Left Click  : pick point
    Shift + Right Click : undo
    Q                   : finish
    """
    xyz = _as_points(
        points
    )

    cloud = (
        o3d.geometry.PointCloud()
    )

    cloud.points = (
        o3d.utility.Vector3dVector(
            xyz
        )
    )

    cloud.paint_uniform_color(
        [0.72, 0.72, 0.72]
    )

    print(
        "\n[PICK QUERY POINTS]"
    )

    print(
        "  Shift + LEFT CLICK  : pick"
    )

    print(
        "  Shift + RIGHT CLICK : undo"
    )

    print(
        "  Q                    : finish"
    )

    visualizer = (
        o3d.visualization
        .VisualizerWithEditing()
    )

    visualizer.create_window(
        window_name=window_name,
        width=1280,
        height=900,
    )

    visualizer.add_geometry(
        cloud
    )

    visualizer.run()

    picked = list(
        visualizer
        .get_picked_points()
    )

    visualizer.destroy_window()

    return [
        int(i)
        for i in picked
    ]


def _save_debug_npz(
    debug: dict,
    path,
):
    """
    Save numeric local-neighborhood diagnostics for later plotting.
    """
    components = debug[
        "final_components"
    ]

    payload = {
        "query_index": np.asarray(
            [debug["query_index"]],
            dtype=np.int64,
        ),
        "query_point": (
            debug["query_point"]
        ),
        "neighborhood_indices": (
            debug[
                "neighborhood_indices"
            ]
        ),
        "neighborhood_points": (
            debug[
                "neighborhood_points"
            ]
        ),
        "rough_query_normal": (
            debug[
                "rough_query_normal"
            ]
        ),
        "final_normal": (
            debug["final_normal"]
        ),
        "final_weight": (
            debug["final_weight"]
        ),
        "neighborhood_injected_outlier_mask": (
            debug.get(
                "neighborhood_injected_outlier_mask",
                np.zeros(
                    len(
                        debug[
                            "neighborhood_indices"
                        ]
                    ),
                    dtype=bool,
                ),
            )
        ),
    }

    if components is not None:
        for key in [
            "spatial_weight",
            "normal_field_weight",
            "residual_weight",
            "distance",
            "theta_deg",
            "expected_theta_deg",
            "excess_deg",
            "plane_residual",
        ]:
            payload[
                key
            ] = components[
                key
            ]

    np.savez(
        path,
        **payload,
    )


def run_surface_aware_debugger(
    points: Any,
    *,
    radius: float,
    sigma_e_deg: float = 15.0,
    sigma_r: float = 0.25,
    spatial_sigma_ratio: float = 0.5,
    max_iterations: int = 5,
    convergence_deg: float = 0.05,
    min_neighbors: int = 5,
    threads: int = 0,
    query_indices=None,
    views=("weight",),
    save_debug_dir=None,
    injected_outlier_mask: np.ndarray | None = None,
):
    """
    Interactive debugger entry point.

    Workflow
    --------
    1. Compute ordinary PCA rough normals ONCE.
    2. Let the user pick one or more query points unless indices were supplied.
    3. Re-run the surface-aware normal iteration for each selected point.
    4. Display the exact final weighted-PCA support as an Open3D heatmap.
    5. Optionally display component diagnostics and save NPZ files.

    Supported views
    ---------------
    weight
        Exact final weight used by weighted PCA.
    spatial
        w_d
    normal_field
        w_e
    residual_weight
        w_r
    """
    xyz = _as_points(
        points
    )

    if injected_outlier_mask is None:
        injected_outlier_mask = np.zeros(
            len(xyz),
            dtype=bool,
        )
    else:
        injected_outlier_mask = np.asarray(
            injected_outlier_mask,
            dtype=bool,
        )

        if injected_outlier_mask.shape != (
            len(xyz),
        ):
            raise ValueError(
                "injected_outlier_mask must have shape (N,)"
            )

    print(
        "\nComputing rough PCA normals once..."
    )

    rough = estimate_pca_normals(
        xyz,
        radius,
        threads=threads,
    )

    if query_indices is None:
        query_indices = (
            pick_query_points(
                xyz
            )
        )

    query_indices = [
        int(i)
        for i in query_indices
    ]

    if not query_indices:
        print(
            "No query point selected."
        )

        return []

    if save_debug_dir is not None:
        from pathlib import Path

        save_debug_dir = Path(
            save_debug_dir
        )

        save_debug_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    outputs = []

    for query_index in query_indices:
        debug = (
            inspect_surface_aware_point(
                xyz,
                query_index,
                radius,
                sigma_e_deg=(
                    sigma_e_deg
                ),
                sigma_r=sigma_r,
                spatial_sigma_ratio=(
                    spatial_sigma_ratio
                ),
                max_iterations=(
                    max_iterations
                ),
                convergence_deg=(
                    convergence_deg
                ),
                min_neighbors=(
                    min_neighbors
                ),
                rough_normals=rough,
                threads=threads,
                injected_outlier_mask=(
                    injected_outlier_mask
                ),
            )
        )

        _print_surface_aware_debug_summary(
            debug
        )

        if (
            save_debug_dir
            is not None
        ):
            _save_debug_npz(
                debug,
                save_debug_dir
                / (
                    f"query_"
                    f"{query_index:06d}"
                    f"_debug.npz"
                ),
            )

        for view in views:
            visualize_surface_aware_point(
                xyz,
                debug,
                view=view,
            )

        outputs.append(
            debug
        )

    return outputs


# =====================================================================
# Synthetic noise / outlier injection for robustness debugging
# =====================================================================

def _apply_synthetic_corruption(
    points: np.ndarray,
    *,
    noise_std: float = 0.0,
    outlier_ratio: float = 0.0,
    outlier_offset_min: float = 0.3,
    outlier_offset_max: float = 1.5,
    seed: int = 0,
):
    """
    Add controlled synthetic corruption AFTER mesh sampling / voxelization.

    Gaussian noise
    --------------
    Every original point receives isotropic position noise:

        p_noisy = p + eps
        eps ~ N(0, noise_std^2 I)

    Local outliers
    --------------
    Outliers are generated near randomly selected existing points:

        q_out = p_anchor + d * u

    where u is a random unit 3D vector and

        d ~ Uniform(outlier_offset_min, outlier_offset_max)

    This is intentionally harder than scattering outliers over a huge
    bounding box, because many injected outliers can remain inside the
    local normal-estimation radius.

    Returns
    -------
    corrupted_points : (N+M,3)
    injected_outlier_mask : (N+M,) bool
        True only for the synthetically appended outliers.
    """
    xyz = _as_points(
        points
    ).copy()

    noise_std = float(
        noise_std
    )

    outlier_ratio = float(
        outlier_ratio
    )

    outlier_offset_min = float(
        outlier_offset_min
    )

    outlier_offset_max = float(
        outlier_offset_max
    )

    if noise_std < 0:
        raise ValueError(
            "noise_std must be >= 0"
        )

    if outlier_ratio < 0:
        raise ValueError(
            "outlier_ratio must be >= 0"
        )

    if outlier_offset_min < 0:
        raise ValueError(
            "outlier_offset_min must be >= 0"
        )

    if (
        outlier_offset_max
        < outlier_offset_min
    ):
        raise ValueError(
            "outlier_offset_max must be >= outlier_offset_min"
        )

    rng = np.random.default_rng(
        int(seed)
    )

    # 1) Position noise on the original surface samples.
    if noise_std > 0:
        xyz += rng.normal(
            loc=0.0,
            scale=noise_std,
            size=xyz.shape,
        )

    n_inlier = len(xyz)

    # 2) Append local outliers.
    n_outlier = int(
        round(
            outlier_ratio
            * n_inlier
        )
    )

    if n_outlier <= 0:
        mask = np.zeros(
            n_inlier,
            dtype=bool,
        )

        print(
            "\n"
            + "=" * 72
        )
        print(
            " SYNTHETIC CORRUPTION"
        )
        print(
            "=" * 72
        )
        print(
            f"Gaussian noise std : {noise_std:.6g}"
        )
        print(
            "Injected outliers  : 0"
        )

        return (
            xyz,
            mask,
        )

    anchor_idx = rng.integers(
        0,
        n_inlier,
        size=n_outlier,
    )

    anchors = xyz[
        anchor_idx
    ]

    direction = rng.normal(
        size=(
            n_outlier,
            3,
        )
    )

    direction_norm = np.linalg.norm(
        direction,
        axis=1,
        keepdims=True,
    )

    direction /= np.maximum(
        direction_norm,
        1e-12,
    )

    offsets = rng.uniform(
        outlier_offset_min,
        outlier_offset_max,
        size=(
            n_outlier,
            1,
        ),
    )

    outliers = (
        anchors
        + offsets
        * direction
    )

    corrupted = np.vstack([
        xyz,
        outliers,
    ])

    mask = np.zeros(
        len(corrupted),
        dtype=bool,
    )

    mask[
        n_inlier:
    ] = True

    print(
        "\n"
        + "=" * 72
    )
    print(
        " SYNTHETIC CORRUPTION"
    )
    print(
        "=" * 72
    )
    print(
        f"Original points    : {n_inlier}"
    )
    print(
        f"Gaussian noise std : {noise_std:.6g}"
    )
    print(
        f"Injected outliers  : {n_outlier} "
        f"({100.0*outlier_ratio:.3f}% of original)"
    )
    print(
        f"Outlier offset     : "
        f"[{outlier_offset_min:.6g}, "
        f"{outlier_offset_max:.6g}]"
    )
    print(
        f"Final points       : {len(corrupted)}"
    )

    return (
        corrupted,
        mask,
    )


# =====================================================================
# Standalone Open3D debugger
# =====================================================================

def _load_debug_input(
    path,
    *,
    input_type: str = "auto",
    sample_points: int = 300000,
    voxel: float = 0.0,
    seed: int = 0,
):
    """
    Load either a mesh or point cloud for the standalone debugger.

    For a mesh:
        sample surface -> optional voxel downsample

    For a point cloud:
        load -> optional voxel downsample
    """
    from pathlib import Path

    path = Path(
        path
    )

    if not path.exists():
        raise FileNotFoundError(
            str(path)
        )

    mesh_suffix = {
        ".stl",
        ".obj",
        ".off",
        ".gltf",
        ".glb",
    }

    if input_type == "auto":
        is_mesh = (
            path.suffix.lower()
            in mesh_suffix
        )

    elif input_type == "mesh":
        is_mesh = True

    elif input_type == "pointcloud":
        is_mesh = False

    else:
        raise ValueError(
            "input_type must be "
            "auto/mesh/pointcloud"
        )

    if is_mesh:
        mesh = (
            o3d.io.read_triangle_mesh(
                str(path)
            )
        )

        if mesh.is_empty():
            raise RuntimeError(
                f"Failed to read mesh: "
                f"{path}"
            )

        mesh.remove_duplicated_vertices()
        mesh.remove_duplicated_triangles()
        mesh.remove_degenerate_triangles()

        o3d.utility.random.seed(
            int(seed)
        )

        cloud = (
            mesh.sample_points_uniformly(
                number_of_points=(
                    int(sample_points)
                )
            )
        )

    else:
        cloud = (
            o3d.io.read_point_cloud(
                str(path)
            )
        )

        if cloud.is_empty():
            raise RuntimeError(
                f"Failed to read point cloud: "
                f"{path}"
            )

    if voxel > 0:
        cloud = (
            cloud.voxel_down_sample(
                float(voxel)
            )
        )

    xyz = np.asarray(
        cloud.points,
        dtype=np.float64,
    )

    print(
        "\n"
        + "=" * 72
    )

    print(
        " SURFACE-AWARE NORMAL DEBUGGER"
    )

    print(
        "=" * 72
    )

    print(
        f"input      : {path}"
    )

    print(
        f"type       : "
        f"{'mesh' if is_mesh else 'pointcloud'}"
    )

    print(
        f"points     : {len(xyz)}"
    )

    if voxel > 0:
        print(
            f"voxel      : {voxel}"
        )

    return xyz


def _parse_debug_views(
    text: str,
):
    text = str(
        text
    ).strip()

    if text == "all":
        return (
            "weight",
            "spatial",
            "normal_field",
            "residual_weight",
        )

    views = tuple(
        x.strip()
        for x in text.split(",")
        if x.strip()
    )

    allowed = {
        "weight",
        "spatial",
        "normal_field",
        "residual_weight",
    }

    invalid = [
        x
        for x in views
        if x not in allowed
    ]

    if invalid:
        raise ValueError(
            f"Invalid debug views: "
            f"{invalid}"
        )

    if not views:
        return (
            "weight",
        )

    return views


def main():
    """
    Standalone visual debugger.

    Importing this file does NOT invoke main().
    """
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Interactive Open3D debugger for "
            "surface-aware normal weights."
        )
    )

    parser.add_argument(
        "--input",
        required=True,
        help=(
            "STL/OBJ/OFF mesh or "
            "PLY/PCD point cloud."
        ),
    )

    parser.add_argument(
        "--input-type",
        choices=[
            "auto",
            "mesh",
            "pointcloud",
        ],
        default="auto",
    )

    parser.add_argument(
        "--sample-points",
        type=int,
        default=300000,
        help=(
            "Surface samples when input is a mesh."
        ),
    )

    parser.add_argument(
        "--voxel",
        type=float,
        default=1.0,
        help=(
            "Optional voxel downsample size. "
            "Use <=0 to disable."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.0,
        help=(
            "Isotropic Gaussian position-noise standard deviation. "
            "Uses the same length unit as the point cloud."
        ),
    )

    parser.add_argument(
        "--outlier-ratio",
        type=float,
        default=0.0,
        help=(
            "Number of injected local outliers divided by the "
            "number of original points. Example: 0.05 = 5%%."
        ),
    )

    parser.add_argument(
        "--outlier-offset-min",
        type=float,
        default=0.3,
        help=(
            "Minimum random offset of a synthetic local outlier "
            "from its anchor surface point."
        ),
    )

    parser.add_argument(
        "--outlier-offset-max",
        type=float,
        default=1.5,
        help=(
            "Maximum random offset of a synthetic local outlier "
            "from its anchor surface point."
        ),
    )

    parser.add_argument(
        "--normal-radius",
        type=float,
        default=3.0,
    )

    parser.add_argument(
        "--sigma-e-deg",
        type=float,
        default=15.0,
    )

    parser.add_argument(
        "--sigma-r",
        type=float,
        default=0.25,
    )

    parser.add_argument(
        "--spatial-sigma-ratio",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--normal-max-iterations",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--normal-convergence-deg",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--min-neighbors",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--threads",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--query-index",
        action="append",
        type=int,
        default=None,
        help=(
            "Inspect this index directly. "
            "Can be repeated. "
            "If omitted, Open3D point picking is used."
        ),
    )

    parser.add_argument(
        "--debug-view",
        default="weight",
        help=(
            "Comma-separated views or 'all'. "
            "Available: weight, spatial, normal_field, "
            "residual_weight."
        ),
    )

    parser.add_argument(
        "--save-debug-dir",
        default=None,
        help=(
            "Optional folder for per-query NPZ diagnostics."
        ),
    )

    args = (
        parser.parse_args()
    )

    xyz = _load_debug_input(
        args.input,
        input_type=(
            args.input_type
        ),
        sample_points=(
            args.sample_points
        ),
        voxel=args.voxel,
        seed=args.seed,
    )

    xyz, injected_outlier_mask = (
        _apply_synthetic_corruption(
            xyz,
            noise_std=(
                args.noise_std
            ),
            outlier_ratio=(
                args.outlier_ratio
            ),
            outlier_offset_min=(
                args.outlier_offset_min
            ),
            outlier_offset_max=(
                args.outlier_offset_max
            ),
            seed=args.seed,
        )
    )

    views = (
        _parse_debug_views(
            args.debug_view
        )
    )

    run_surface_aware_debugger(
        xyz,
        radius=(
            args.normal_radius
        ),
        sigma_e_deg=(
            args.sigma_e_deg
        ),
        sigma_r=args.sigma_r,
        spatial_sigma_ratio=(
            args.spatial_sigma_ratio
        ),
        max_iterations=(
            args.normal_max_iterations
        ),
        convergence_deg=(
            args.normal_convergence_deg
        ),
        min_neighbors=(
            args.min_neighbors
        ),
        threads=args.threads,
        query_indices=(
            args.query_index
        ),
        views=views,
        save_debug_dir=(
            args.save_debug_dir
        ),
        injected_outlier_mask=(
            injected_outlier_mask
        ),
    )


if __name__ == "__main__":
    main()