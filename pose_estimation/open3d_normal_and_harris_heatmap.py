#!/usr/bin/env python3
"""
Open3D PCA vs Proposed Normal + Gaussian-Weighted Harris3D Heatmaps

Compare exactly two normal fields:

    1) PCA baseline
    2) Proposed method
       - ITERATIVE (default)
       - ONESHOT   (--proposed oneshot)

GT:
    Closest triangle face normal on the original CAD mesh.

Error:
    acos(abs(n_est dot n_gt))

Visualization:
    Open3D only.
    Four interactive windows are shown sequentially:
        PCA normal error -> Proposed normal error -> PCA Gaussian-Harris -> Proposed Gaussian-Harris

Example
-------
python open3d_normal_and_harris_heatmap.py \
    --stl /home/choisuhyun/lvs_HandEyeCalibration/pose_estimation/미그럼틀.stl \
    --voxel 0.5 \
    --normal-radius 3.0 \
    --proposed iterative \
    --sigma-e 15 \
    --sigma-r 0.25 \
    --max-iter 5 \
    --vmax-deg 30

Open3D:
    mouse drag : rotate
    wheel      : zoom
    shift+drag : pan
    Q / ESC    : close current window

Normal arrows:
    default: every point
    --normal-length 0.75    # explicit arrow length [mm]
    --normal-stride 2       # draw every 2nd point if too dense
"""

import argparse
import csv
import math
from pathlib import Path

import numpy as np
import open3d as o3d
import pclpybridge as pcl


# ============================================================
# Defaults
# ============================================================

DEFAULT_STL = (
    "/home/choisuhyun/lvs_HandEyeCalibration/"
    "pose_estimation/data/part.stl"
)

N_DENSE = 300000
VOXEL = 1.0
NORMAL_RADIUS = 2.0

SIGMA_E_DEG = 15.0
SIGMA_R_MM = 0.25
SPATIAL_SIGMA_RATIO = 0.5

MAX_ITER = 5
CONVERGENCE_DEG = 0.05
MIN_NEIGHBORS = 5

VMAX_DEG = 30.0

HARRIS_RADIUS = 3.0
HARRIS_GAUSSIAN_SIGMA = 0.0  # <=0 -> 0.5 * Harris radius
HARRIS_PERCENTILE_LOW = 1.0
HARRIS_PERCENTILE_HIGH = 99.0


# ============================================================
# Basic utility
# ============================================================

def normalize_rows(x):
    x = np.asarray(x, dtype=np.float64)
    n = np.linalg.norm(x, axis=1, keepdims=True)

    out = np.full_like(x, np.nan, dtype=np.float64)

    valid = np.squeeze(n > 1e-12)
    out[valid] = x[valid] / n[valid]

    return out


def normal_error_deg(n_est, n_gt):
    """
    Axial / sign-invariant normal error:
        n and -n are identical plane normals.
    """
    n_est = np.asarray(n_est, dtype=np.float64)
    n_gt = np.asarray(n_gt, dtype=np.float64)

    err = np.full(len(n_est), np.nan, dtype=np.float64)

    valid = (
        np.all(np.isfinite(n_est), axis=1)
        & np.all(np.isfinite(n_gt), axis=1)
    )

    if not np.any(valid):
        return err

    a = normalize_rows(n_est[valid])
    b = normalize_rows(n_gt[valid])

    dot = np.sum(a * b, axis=1)
    dot = np.clip(np.abs(dot), 0.0, 1.0)

    err[valid] = np.degrees(np.arccos(dot))

    return err


def stats(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "median": np.nan,
            "p95": np.nan,
            "max": np.nan,
        }

    return {
        "n": len(x),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "p95": float(np.percentile(x, 95)),
        "max": float(np.max(x)),
    }


def print_stats(name, error):
    s = stats(error)

    print(
        f"{name:<10s} "
        f"N={s['n']:6d} | "
        f"mean={s['mean']:7.3f} deg | "
        f"median={s['median']:7.3f} deg | "
        f"P95={s['p95']:7.3f} deg | "
        f"max={s['max']:7.3f} deg"
    )


# ============================================================
# CAD + sampling
# ============================================================

def load_mesh(path):
    mesh = o3d.io.read_triangle_mesh(str(path))

    if mesh.is_empty():
        raise RuntimeError(f"Failed to load mesh: {path}")

    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.compute_triangle_normals()

    return mesh


def sample_mesh(mesh, n_dense, voxel, seed):
    o3d.utility.random.seed(int(seed))

    dense = mesh.sample_points_uniformly(
        number_of_points=int(n_dense)
    )

    pcd = dense.voxel_down_sample(float(voxel))

    points = np.asarray(pcd.points, dtype=np.float64)

    nn = np.asarray(
        pcd.compute_nearest_neighbor_distance(),
        dtype=np.float64,
    )

    print("\nPOINT CLOUD")
    print("-" * 60)
    print(f"dense       : {len(dense.points)}")
    print(f"voxel       : {voxel:.3f} mm")
    print(f"after voxel : {len(points)}")

    if len(nn):
        print(f"mean NN     : {np.mean(nn):.4f} mm")
        print(f"median NN   : {np.median(nn):.4f} mm")

    return points


# ============================================================
# GT mesh normal
# ============================================================

def closest_triangle_gt_normals(mesh, points):
    """
    GT normal = normal of closest CAD triangle.

    This preserves sharp edges better than interpolated smooth vertex normals.
    """
    mesh.compute_triangle_normals()

    tri_normals = np.asarray(
        mesh.triangle_normals,
        dtype=np.float64,
    )

    tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tmesh)

    query = o3d.core.Tensor(
        np.asarray(points, dtype=np.float32)
    )

    ans = scene.compute_closest_points(query)

    triangle_ids = np.asarray(
        ans["primitive_ids"].numpy(),
        dtype=np.int64,
    )

    closest_points = np.asarray(
        ans["points"].numpy(),
        dtype=np.float64,
    )

    gt_normals = tri_normals[triangle_ids]
    gt_normals = normalize_rows(gt_normals)

    mesh_dist = np.linalg.norm(
        points - closest_points,
        axis=1,
    )

    return gt_normals, mesh_dist, triangle_ids


# ============================================================
# PCA baseline
# ============================================================

def estimate_pca_normals(points, radius):
    """
    PCL radius-PCA normal estimation.
    Also used as the fixed rough-normal field for the proposed method.
    """
    normals, curvature = pcl.estimate_normals(
        np.asarray(points, dtype=np.float32),
        radius=float(radius),
    )

    normals = np.asarray(normals, dtype=np.float64)
    normals = normalize_rows(normals)

    curvature = np.asarray(curvature, dtype=np.float64)

    return normals, curvature


# ============================================================
# Weighted PCA
# ============================================================

def weighted_pca(q, w, reference=None):
    q = np.asarray(q, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)

    valid = (
        np.all(np.isfinite(q), axis=1)
        & np.isfinite(w)
        & (w > 1e-12)
    )

    q = q[valid]
    w = w[valid]

    if len(q) < 3:
        return None

    sw = float(np.sum(w))
    if sw < 1e-12:
        return None

    c = np.sum(q * w[:, None], axis=0) / sw
    y = q - c

    cov = (y * w[:, None]).T @ y / sw

    _, vec = np.linalg.eigh(cov)
    n = vec[:, 0]

    n_norm = np.linalg.norm(n)

    if n_norm < 1e-12:
        return None

    n = n / n_norm

    if reference is not None and np.all(np.isfinite(reference)):
        reference = reference / max(np.linalg.norm(reference), 1e-12)

        if np.dot(n, reference) < 0:
            n = -n

    return n


# ============================================================
# Proposed soft weight
# ============================================================

def soft_weights(
    query,
    neighbors,
    rough_neighbor_normals,
    current_normal,
    sigma_d,
    sigma_e_rad,
    sigma_r,
):
    """
    d_i       = ||q_i - p||
    theta_i   = angle(n_i^rough, current_normal)
    kappa_hat = median(theta_i / d_i)

    expected:
        theta_expected = kappa_hat * d_i

    excess:
        e_i = max(0, theta_i - theta_expected)

    residual:
        r_i = |(q_i-p)^T current_normal|

    weight:
        w = w_d * w_e * w_r
    """
    delta = neighbors - query
    d = np.linalg.norm(delta, axis=1)

    local_n = np.asarray(
        rough_neighbor_normals,
        dtype=np.float64,
    ).copy()

    valid_n = np.all(np.isfinite(local_n), axis=1)

    if np.any(valid_n):
        temp = local_n[valid_n]

        signs = temp @ current_normal
        temp[signs < 0] *= -1

        local_n[valid_n] = temp

    theta = np.full(len(neighbors), np.nan, dtype=np.float64)

    if np.any(valid_n):
        dot = np.clip(
            local_n[valid_n] @ current_normal,
            -1.0,
            1.0,
        )

        theta[valid_n] = np.arccos(dot)

    mask = (
        valid_n
        & np.isfinite(theta)
        & (d > 1e-9)
    )

    if np.sum(mask) >= 3:
        kappa_hat = float(
            np.median(theta[mask] / d[mask])
        )
    else:
        kappa_hat = 0.0

    expected = kappa_hat * d

    theta_safe = np.where(
        np.isfinite(theta),
        theta,
        np.inf,
    )

    excess = np.maximum(
        0.0,
        theta_safe - expected,
    )

    residual = np.abs(delta @ current_normal)

    wd = np.exp(
        -(d * d) / (2 * sigma_d * sigma_d)
    )

    we = 1.0 / (
        1.0 + (excess / sigma_e_rad) ** 2
    )

    wr = 1.0 / (
        1.0 + (residual / sigma_r) ** 2
    )

    w = wd * we * wr
    w[~valid_n] = 0.0

    return w


# ============================================================
# KD-tree
# ============================================================

def build_tree(points):
    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(
        np.asarray(points, dtype=np.float64)
    )

    return o3d.geometry.KDTreeFlann(pcd)


# ============================================================
# ONESHOT
# ============================================================

def estimate_oneshot(
    points,
    rough_normals,
    radius,
    sigma_e_deg,
    sigma_r,
    spatial_sigma_ratio,
    min_neighbors,
):
    tree = build_tree(points)

    out = np.full_like(rough_normals, np.nan)

    sigma_d = max(
        spatial_sigma_ratio * radius,
        1e-9,
    )

    sigma_e_rad = math.radians(
        max(sigma_e_deg, 1e-6)
    )

    for i, p in enumerate(points):
        n0 = rough_normals[i]

        if np.any(~np.isfinite(n0)):
            continue

        _, idx, _ = tree.search_radius_vector_3d(
            p,
            radius,
        )

        idx = np.asarray(idx, dtype=np.int64)

        if len(idx) < min_neighbors:
            continue

        q = points[idx]

        w = soft_weights(
            query=p,
            neighbors=q,
            rough_neighbor_normals=rough_normals[idx],
            current_normal=n0,
            sigma_d=sigma_d,
            sigma_e_rad=sigma_e_rad,
            sigma_r=sigma_r,
        )

        n = weighted_pca(
            q,
            w,
            reference=n0,
        )

        if n is not None:
            out[i] = n

    return out


# ============================================================
# ITERATIVE
# ============================================================

def estimate_iterative(
    points,
    rough_normals,
    radius,
    sigma_e_deg,
    sigma_r,
    spatial_sigma_ratio,
    max_iter,
    convergence_deg,
    min_neighbors,
):
    tree = build_tree(points)

    out = np.full_like(rough_normals, np.nan)

    sigma_d = max(
        spatial_sigma_ratio * radius,
        1e-9,
    )

    sigma_e_rad = math.radians(
        max(sigma_e_deg, 1e-6)
    )

    for i, p in enumerate(points):
        n_current = rough_normals[i].copy()

        if np.any(~np.isfinite(n_current)):
            continue

        _, idx, _ = tree.search_radius_vector_3d(
            p,
            radius,
        )

        idx = np.asarray(idx, dtype=np.int64)

        if len(idx) < min_neighbors:
            continue

        q = points[idx]
        local_rough = rough_normals[idx]

        for _ in range(max_iter):
            w = soft_weights(
                query=p,
                neighbors=q,
                rough_neighbor_normals=local_rough,
                current_normal=n_current,
                sigma_d=sigma_d,
                sigma_e_rad=sigma_e_rad,
                sigma_r=sigma_r,
            )

            n_new = weighted_pca(
                q,
                w,
                reference=n_current,
            )

            if n_new is None:
                break

            dot = np.clip(
                abs(float(np.dot(n_new, n_current))),
                0.0,
                1.0,
            )

            change = math.degrees(math.acos(dot))

            n_current = n_new

            if change < convergence_deg:
                break

        out[i] = n_current

    return out


# ============================================================
# Error heatmap color
# ============================================================

def turbo_colormap(x):
    """
    Turbo polynomial approximation.
    No matplotlib required.
    """
    x = np.clip(
        np.asarray(x, dtype=np.float64),
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
            x**2,
            x**3,
            x**4,
            x**5,
        ],
        axis=1,
    )

    rgb = np.c_[
        X @ kr,
        X @ kg,
        X @ kb,
    ]

    return np.clip(rgb, 0, 1)


def errors_to_colors(error_deg, vmax_deg):
    error_deg = np.asarray(error_deg, dtype=np.float64)

    x = np.clip(
        error_deg / vmax_deg,
        0.0,
        1.0,
    )

    colors = turbo_colormap(
        np.nan_to_num(x, nan=0.0)
    )

    # invalid normals = gray
    colors[~np.isfinite(error_deg)] = [0.4, 0.4, 0.4]

    return colors


def make_heatmap_cloud(points, error_deg, vmax_deg):
    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(
        points
    )

    pcd.colors = o3d.utility.Vector3dVector(
        errors_to_colors(error_deg, vmax_deg)
    )

    return pcd



# ============================================================
# Harris3D response from a supplied normal field
# ============================================================

def harris3d_response_from_normals(
    points,
    normals,
    radius,
    gaussian_sigma,
    min_neighbors=5,
):
    """
    Gaussian-weighted Harris3D response from an externally supplied
    normal field.

    PCL HarrisKeypoint3D uses an UNIFORM average over radius neighbors:

        M_i = mean_j (n_j n_j^T)

    Here we intentionally replace that uniform window with a spatial
    Gaussian window centered at the query point p_i:

        w_ij = exp(-||q_j - p_i||^2 / (2 sigma_H^2))

        M_i = sum_j w_ij (n_j n_j^T) / sum_j w_ij

    The Harris response itself is kept identical to PCL's HARRIS form:

        R_i = 0.04 + det(M_i) - 0.04 * trace(M_i)^2

    Notes
    -----
    - This Gaussian weighting is NOT part of stock PCL HarrisKeypoint3D.
    - n and -n yield the same outer product, so normal sign does not
      affect the response.
    - Using the same point cloud, radius, sigma, and response equation for
      both normal estimators isolates the effect of the normal field.
    """
    points = np.asarray(points, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)

    radius = float(radius)
    gaussian_sigma = float(gaussian_sigma)
    if gaussian_sigma <= 0.0:
        gaussian_sigma = 0.5 * radius
    gaussian_sigma = max(gaussian_sigma, 1e-12)

    tree = build_tree(points)
    response = np.full(len(points), np.nan, dtype=np.float64)

    for i, p in enumerate(points):
        _, idx, dist2 = tree.search_radius_vector_3d(
            p,
            radius,
        )

        idx = np.asarray(idx, dtype=np.int64)
        dist2 = np.asarray(dist2, dtype=np.float64)

        if len(idx) < int(min_neighbors):
            continue

        local_n = normals[idx]
        valid = np.all(np.isfinite(local_n), axis=1) & np.isfinite(dist2)

        local_n = local_n[valid]
        local_dist2 = dist2[valid]

        if len(local_n) < int(min_neighbors):
            continue

        local_n = normalize_rows(local_n)
        valid = np.all(np.isfinite(local_n), axis=1)
        local_n = local_n[valid]
        local_dist2 = local_dist2[valid]

        if len(local_n) < int(min_neighbors):
            continue

        # Spatial Gaussian window around the query point.
        w = np.exp(
            -local_dist2 / (2.0 * gaussian_sigma * gaussian_sigma)
        )

        sw = float(np.sum(w))
        if sw < 1e-12:
            continue

        # Weighted normal structure tensor / second-moment matrix.
        M = (local_n * w[:, None]).T @ local_n / sw

        trace = float(np.trace(M))

        if abs(trace) < 1e-12:
            response[i] = 0.0
            continue

        det = float(np.linalg.det(M))
        response[i] = 0.04 + det - 0.04 * trace * trace

    return response

def scalar_stats(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return {
            "n": 0,
            "min": np.nan,
            "mean": np.nan,
            "median": np.nan,
            "p95": np.nan,
            "p99": np.nan,
            "max": np.nan,
        }

    return {
        "n": len(x),
        "min": float(np.min(x)),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "p95": float(np.percentile(x, 95)),
        "p99": float(np.percentile(x, 99)),
        "max": float(np.max(x)),
    }


def print_scalar_stats(name, values):
    s = scalar_stats(values)
    print(
        f"{name:<10s} "
        f"N={s['n']:6d} | "
        f"min={s['min']:+.6e} | "
        f"median={s['median']:+.6e} | "
        f"P95={s['p95']:+.6e} | "
        f"P99={s['p99']:+.6e} | "
        f"max={s['max']:+.6e}"
    )


def shared_scalar_range(
    a,
    b,
    percentile_low=1.0,
    percentile_high=99.0,
):
    """Shared robust scale so two response heatmaps are comparable."""
    values = np.concatenate([
        np.asarray(a, dtype=np.float64).ravel(),
        np.asarray(b, dtype=np.float64).ravel(),
    ])
    values = values[np.isfinite(values)]

    if len(values) == 0:
        return 0.0, 1.0

    lo = float(np.percentile(values, percentile_low))
    hi = float(np.percentile(values, percentile_high))

    if not np.isfinite(lo):
        lo = float(np.min(values))
    if not np.isfinite(hi):
        hi = float(np.max(values))

    if hi <= lo + 1e-15:
        lo = float(np.min(values))
        hi = float(np.max(values))

    if hi <= lo + 1e-15:
        hi = lo + 1.0

    return lo, hi


def scalar_to_colors(values, vmin, vmax):
    values = np.asarray(values, dtype=np.float64)
    denom = max(float(vmax) - float(vmin), 1e-15)

    x = (values - float(vmin)) / denom
    x = np.clip(x, 0.0, 1.0)

    colors = turbo_colormap(np.nan_to_num(x, nan=0.0))
    colors[~np.isfinite(values)] = [0.4, 0.4, 0.4]
    return colors


def make_scalar_heatmap_cloud(points, values, vmin, vmax):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(
        np.asarray(points, dtype=np.float64)
    )
    pcd.colors = o3d.utility.Vector3dVector(
        scalar_to_colors(values, vmin, vmax)
    )
    return pcd


# ============================================================
# Normal arrow visualization
# ============================================================

def make_normal_arrow_lineset(
    points,
    normals,
    length,
    stride=1,
    head_ratio=0.28,
    color=(1.0, 1.0, 1.0),
):
    """
    Lightweight per-point normal arrows using one Open3D LineSet.

    Each arrow:
        shaft:
            p -> p + L*n

        arrow head:
            tip -> tip - h*n + h*perp
            tip -> tip - h*n - h*perp

    This is much lighter than creating one TriangleMesh arrow per point.

    Parameters
    ----------
    points : (N,3)
    normals : (N,3)
    length : float
        Arrow shaft length in the same units as the point cloud.
    stride : int
        1 = every valid point.
        2 = every second valid point, etc.
    head_ratio : float
        Arrow-head size relative to shaft length.
    color : RGB tuple
    """
    points = np.asarray(points, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)

    stride = max(1, int(stride))
    length = max(float(length), 1e-9)
    head_len = max(length * float(head_ratio), 1e-9)

    valid = (
        np.all(np.isfinite(points), axis=1)
        & np.all(np.isfinite(normals), axis=1)
    )

    ids = np.flatnonzero(valid)[::stride]

    vertices = []
    lines = []

    for i in ids:
        p = points[i]

        n = normals[i]
        n_norm = np.linalg.norm(n)

        if n_norm < 1e-12:
            continue

        n = n / n_norm

        tip = p + length * n

        # Choose a stable helper axis not parallel to n.
        if abs(n[2]) < 0.9:
            helper_axis = np.array([0.0, 0.0, 1.0])
        else:
            helper_axis = np.array([0.0, 1.0, 0.0])

        perp = np.cross(n, helper_axis)
        perp_norm = np.linalg.norm(perp)

        if perp_norm < 1e-12:
            helper_axis = np.array([1.0, 0.0, 0.0])
            perp = np.cross(n, helper_axis)
            perp_norm = np.linalg.norm(perp)

        if perp_norm < 1e-12:
            continue

        perp = perp / perp_norm

        wing1 = tip - head_len * n + 0.55 * head_len * perp
        wing2 = tip - head_len * n - 0.55 * head_len * perp

        base = len(vertices)

        vertices.extend([
            p,
            tip,
            wing1,
            wing2,
        ])

        lines.extend([
            [base + 0, base + 1],  # shaft
            [base + 1, base + 2],  # head wing 1
            [base + 1, base + 3],  # head wing 2
        ])

    line_set = o3d.geometry.LineSet()

    if len(vertices) == 0:
        return line_set

    line_set.points = o3d.utility.Vector3dVector(
        np.asarray(vertices, dtype=np.float64)
    )

    line_set.lines = o3d.utility.Vector2iVector(
        np.asarray(lines, dtype=np.int32)
    )

    line_set.colors = o3d.utility.Vector3dVector(
        np.tile(
            np.asarray(color, dtype=np.float64),
            (len(lines), 1),
        )
    )

    return line_set


# ============================================================
# Open3D visualization
# ============================================================

def show_heatmap(
    pcd,
    normal_arrows,
    title,
    point_size,
    background,
):
    vis = o3d.visualization.Visualizer()

    vis.create_window(
        window_name=title,
        width=1100,
        height=850,
    )

    vis.add_geometry(pcd)

    if normal_arrows is not None:
        vis.add_geometry(normal_arrows)

    opt = vis.get_render_option()
    opt.point_size = float(point_size)
    opt.line_width = 1.0
    opt.show_coordinate_frame = True

    if background == "white":
        opt.background_color = np.array(
            [1.0, 1.0, 1.0]
        )
    else:
        opt.background_color = np.array(
            [0.0, 0.0, 0.0]
        )

    vis.run()
    vis.destroy_window()


# ============================================================
# Output
# ============================================================

def save_csv(
    out_dir,
    points,
    gt_normals,
    pca_normals,
    proposed_normals,
    pca_error,
    proposed_error,
    mesh_dist,
    triangle_ids,
    proposed_name,
):
    data = np.c_[
        points,
        gt_normals,
        pca_normals,
        proposed_normals,
        pca_error,
        proposed_error,
        mesh_dist,
        triangle_ids,
    ]

    header = ",".join([
        "x","y","z",
        "gt_nx","gt_ny","gt_nz",
        "pca_nx","pca_ny","pca_nz",
        "proposed_nx","proposed_ny","proposed_nz",
        "pca_error_deg",
        "proposed_error_deg",
        "mesh_distance_mm",
        "closest_triangle_id",
    ])

    np.savetxt(
        out_dir / "normal_error_points.csv",
        data,
        delimiter=",",
        header=header,
        comments="",
    )

    rows = []

    for name, err in [
        ("PCA", pca_error),
        (proposed_name, proposed_error),
    ]:
        s = stats(err)

        rows.append({
            "method": name,
            "n_valid": s["n"],
            "mean_deg": s["mean"],
            "median_deg": s["median"],
            "p95_deg": s["p95"],
            "max_deg": s["max"],
        })

    with open(
        out_dir / "summary.csv",
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


# ============================================================
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--stl",
        default=DEFAULT_STL,
    )

    p.add_argument(
        "--n-dense",
        type=int,
        default=N_DENSE,
    )

    p.add_argument(
        "--voxel",
        type=float,
        default=VOXEL,
    )

    p.add_argument(
        "--normal-radius",
        type=float,
        default=NORMAL_RADIUS,
    )

    p.add_argument(
        "--proposed",
        choices=["oneshot", "iterative"],
        default="iterative",
    )

    p.add_argument(
        "--sigma-e",
        type=float,
        default=SIGMA_E_DEG,
    )

    p.add_argument(
        "--sigma-r",
        type=float,
        default=SIGMA_R_MM,
    )

    p.add_argument(
        "--spatial-sigma-ratio",
        type=float,
        default=SPATIAL_SIGMA_RATIO,
    )

    p.add_argument(
        "--max-iter",
        type=int,
        default=MAX_ITER,
    )

    p.add_argument(
        "--convergence-deg",
        type=float,
        default=CONVERGENCE_DEG,
    )

    p.add_argument(
        "--min-neighbors",
        type=int,
        default=MIN_NEIGHBORS,
    )

    p.add_argument(
        "--vmax-deg",
        type=float,
        default=VMAX_DEG,
    )

    p.add_argument(
        "--harris-radius",
        type=float,
        default=HARRIS_RADIUS,
        help=(
            "Harris3D neighborhood radius [mm]. "
            "The same radius is used for PCA and Proposed normals."
        ),
    )

    p.add_argument(
        "--harris-gaussian-sigma",
        type=float,
        default=HARRIS_GAUSSIAN_SIGMA,
        help=(
            "Gaussian sigma [mm] for Harris3D spatial weighting. "
            "<=0 uses 0.5 * harris radius."
        ),
    )

    p.add_argument(
        "--harris-percentile-low",
        type=float,
        default=HARRIS_PERCENTILE_LOW,
        help="Lower percentile for the shared Harris heatmap color scale.",
    )

    p.add_argument(
        "--harris-percentile-high",
        type=float,
        default=HARRIS_PERCENTILE_HIGH,
        help="Upper percentile for the shared Harris heatmap color scale.",
    )

    p.add_argument(
        "--point-size",
        type=float,
        default=5.0,
    )

    p.add_argument(
        "--normal-length",
        type=float,
        default=0.0,
        help=(
            "Normal arrow length [mm]. "
            "<=0 uses 0.75 * voxel size."
        ),
    )

    p.add_argument(
        "--normal-stride",
        type=int,
        default=1,
        help=(
            "Draw one normal arrow every N valid points. "
            "1 draws every point."
        ),
    )

    p.add_argument(
        "--arrow-head-ratio",
        type=float,
        default=0.28,
        help="Arrow-head length / shaft length.",
    )

    p.add_argument(
        "--background",
        choices=["black", "white"],
        default="black",
    )

    p.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    p.add_argument(
        "--output-dir",
        default="normal_error_open3d_results",
    )

    return p.parse_args()


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mesh = load_mesh(args.stl)

    print("\nCAD")
    print("-" * 60)
    print(f"STL       : {args.stl}")
    print(f"vertices  : {len(mesh.vertices)}")
    print(f"triangles : {len(mesh.triangles)}")
    print(
        f"extent    : "
        f"{mesh.get_axis_aligned_bounding_box().get_extent()} mm"
    )

    points = sample_mesh(
        mesh,
        n_dense=args.n_dense,
        voxel=args.voxel,
        seed=args.seed,
    )

    gt_normals, mesh_dist, triangle_ids = (
        closest_triangle_gt_normals(
            mesh,
            points,
        )
    )

    print("\nEstimating PCA baseline...")
    pca_normals, _ = estimate_pca_normals(
        points,
        args.normal_radius,
    )

    if args.proposed == "oneshot":
        print("Estimating ONESHOT...")
        proposed_normals = estimate_oneshot(
            points=points,
            rough_normals=pca_normals,
            radius=args.normal_radius,
            sigma_e_deg=args.sigma_e,
            sigma_r=args.sigma_r,
            spatial_sigma_ratio=args.spatial_sigma_ratio,
            min_neighbors=args.min_neighbors,
        )
        proposed_name = "ONESHOT"

    else:
        print("Estimating ITERATIVE...")
        proposed_normals = estimate_iterative(
            points=points,
            rough_normals=pca_normals,
            radius=args.normal_radius,
            sigma_e_deg=args.sigma_e,
            sigma_r=args.sigma_r,
            spatial_sigma_ratio=args.spatial_sigma_ratio,
            max_iter=args.max_iter,
            convergence_deg=args.convergence_deg,
            min_neighbors=args.min_neighbors,
        )
        proposed_name = "ITERATIVE"

    pca_error = normal_error_deg(
        pca_normals,
        gt_normals,
    )

    proposed_error = normal_error_deg(
        proposed_normals,
        gt_normals,
    )

    print("\nGT NORMAL ERROR")
    print("-" * 90)
    print_stats("PCA", pca_error)
    print_stats(proposed_name, proposed_error)

    print(
        f"\nHeatmap range: "
        f"0 - {args.vmax_deg:.1f} deg"
    )
    print("gray = invalid normal")

    pca_cloud = make_heatmap_cloud(
        points,
        pca_error,
        args.vmax_deg,
    )

    proposed_cloud = make_heatmap_cloud(
        points,
        proposed_error,
        args.vmax_deg,
    )

    # --------------------------------------------------------
    # Harris3D response: SAME detector, ONLY normals differ
    # --------------------------------------------------------
    harris_sigma = (
        float(args.harris_gaussian_sigma)
        if float(args.harris_gaussian_sigma) > 0.0
        else 0.5 * float(args.harris_radius)
    )

    print(
        "\nEstimating Gaussian-weighted Harris3D response "
        f"from PCA normals (sigma={harris_sigma:.3f} mm)..."
    )
    pca_harris = harris3d_response_from_normals(
        points=points,
        normals=pca_normals,
        radius=args.harris_radius,
        gaussian_sigma=harris_sigma,
        min_neighbors=args.min_neighbors,
    )

    print(
        f"Estimating Gaussian-weighted Harris3D response from "
        f"{proposed_name} normals (sigma={harris_sigma:.3f} mm)..."
    )
    proposed_harris = harris3d_response_from_normals(
        points=points,
        normals=proposed_normals,
        radius=args.harris_radius,
        gaussian_sigma=harris_sigma,
        min_neighbors=args.min_neighbors,
    )

    print("\nGAUSSIAN-WEIGHTED HARRIS3D RESPONSE")
    print("-" * 120)
    print_scalar_stats("PCA", pca_harris)
    print_scalar_stats(proposed_name, proposed_harris)

    harris_vmin, harris_vmax = shared_scalar_range(
        pca_harris,
        proposed_harris,
        percentile_low=args.harris_percentile_low,
        percentile_high=args.harris_percentile_high,
    )

    print(
        f"\nShared Gaussian-Harris heatmap range: "
        f"[{harris_vmin:+.6e}, {harris_vmax:+.6e}] "
        f"from pooled P{args.harris_percentile_low:g}-"
        f"P{args.harris_percentile_high:g}"
    )
    print("blue = low response, red = high response, gray = invalid")

    pca_harris_cloud = make_scalar_heatmap_cloud(
        points,
        pca_harris,
        harris_vmin,
        harris_vmax,
    )

    proposed_harris_cloud = make_scalar_heatmap_cloud(
        points,
        proposed_harris,
        harris_vmin,
        harris_vmax,
    )

    # --------------------------------------------------------
    # Estimated-normal arrows
    # --------------------------------------------------------
    normal_length = (
        float(args.normal_length)
        if float(args.normal_length) > 0.0
        else 0.75 * float(args.voxel)
    )

    # Use a high-contrast arrow color against the chosen background.
    if args.background == "white":
        arrow_color = (0.05, 0.05, 0.05)
    else:
        arrow_color = (1.0, 1.0, 1.0)

    pca_arrows = make_normal_arrow_lineset(
        points=points,
        normals=pca_normals,
        length=normal_length,
        stride=args.normal_stride,
        head_ratio=args.arrow_head_ratio,
        color=arrow_color,
    )

    proposed_arrows = make_normal_arrow_lineset(
        points=points,
        normals=proposed_normals,
        length=normal_length,
        stride=args.normal_stride,
        head_ratio=args.arrow_head_ratio,
        color=arrow_color,
    )

    print(
        f"Normal arrows: length={normal_length:.3f} mm, "
        f"stride={args.normal_stride}"
    )

    # Save colored PLY for later screenshots / paper figures.
    o3d.io.write_point_cloud(
        str(out_dir / "pca_error_heatmap.ply"),
        pca_cloud,
    )

    o3d.io.write_point_cloud(
        str(out_dir / "proposed_error_heatmap.ply"),
        proposed_cloud,
    )

    o3d.io.write_point_cloud(
        str(out_dir / "pca_harris_gaussian_response_heatmap.ply"),
        pca_harris_cloud,
    )

    o3d.io.write_point_cloud(
        str(out_dir / "proposed_harris_gaussian_response_heatmap.ply"),
        proposed_harris_cloud,
    )

    np.savetxt(
        out_dir / "harris_gaussian_response_points.csv",
        np.c_[points, pca_harris, proposed_harris],
        delimiter=",",
        header=(
            "x,y,z,pca_harris_gaussian_response,"
            "proposed_harris_gaussian_response"
        ),
        comments="",
    )

    save_csv(
        out_dir=out_dir,
        points=points,
        gt_normals=gt_normals,
        pca_normals=pca_normals,
        proposed_normals=proposed_normals,
        pca_error=pca_error,
        proposed_error=proposed_error,
        mesh_dist=mesh_dist,
        triangle_ids=triangle_ids,
        proposed_name=proposed_name,
    )

    pca_s = stats(pca_error)
    prop_s = stats(proposed_error)

    print("\n[1/4] PCA normal-error heatmap")
    print("Q / ESC -> next window")

    show_heatmap(
        pca_cloud,
        normal_arrows=pca_arrows,
        title=(
            f"PCA GT Normal Error | "
            f"mean={pca_s['mean']:.3f} deg | "
            f"P95={pca_s['p95']:.3f} deg | "
            f"0-{args.vmax_deg:g} deg"
        ),
        point_size=args.point_size,
        background=args.background,
    )

    print(f"\n[2/4] {proposed_name} normal-error heatmap")
    print("Q / ESC -> next window")

    show_heatmap(
        proposed_cloud,
        normal_arrows=proposed_arrows,
        title=(
            f"{proposed_name} GT Normal Error | "
            f"mean={prop_s['mean']:.3f} deg | "
            f"P95={prop_s['p95']:.3f} deg | "
            f"0-{args.vmax_deg:g} deg"
        ),
        point_size=args.point_size,
        background=args.background,
    )

    print("\n[3/4] PCA Gaussian-Harris3D response heatmap")
    print("Q / ESC -> next window")

    show_heatmap(
        pca_harris_cloud,
        normal_arrows=None,
        title=(
            f"PCA normals -> Gaussian-Harris3D | "
            f"radius={args.harris_radius:g} mm | "
            f"sigma={harris_sigma:g} mm | "
            f"shared scale [{harris_vmin:+.3e}, {harris_vmax:+.3e}]"
        ),
        point_size=args.point_size,
        background=args.background,
    )

    print(f"\n[4/4] {proposed_name} Gaussian-Harris3D response heatmap")
    print("Q / ESC -> finish")

    show_heatmap(
        proposed_harris_cloud,
        normal_arrows=None,
        title=(
            f"{proposed_name} normals -> Gaussian-Harris3D | "
            f"radius={args.harris_radius:g} mm | "
            f"sigma={harris_sigma:g} mm | "
            f"shared scale [{harris_vmin:+.3e}, {harris_vmax:+.3e}]"
        ),
        point_size=args.point_size,
        background=args.background,
    )



if __name__ == "__main__":
    main()