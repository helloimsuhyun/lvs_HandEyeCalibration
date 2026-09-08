#!/usr/bin/env python3
"""
Example
-------
python /home/choisuhyun/lvs_HandEyeCalibration/pose_estimation/harris_prior_scan_registration.py \
  /home/choisuhyun/lvs_HandEyeCalibration/pose_estimation/미그럼틀.stl \
  --mesh-unit auto \
  --sample-points 120000 \
  --voxel-mm 0.5 \
  --harris-radius-mm 3.0 \
  --harris-percentile 85 \
  --nms-mm 5 \
  --candidate-rank 1 \
  --scan-length-mm 10 \
  --scan-width-mm 32 \
  --near-width-mm 25 \
  --far-width-mm 39 \
  --scan-step-mm 0.5 \
  --profile-step-mm 0.05 \
  --standoff-mm 80 \
  --z-range-mm 23 \
  --local-margin-mm 5 \
  --match-harris-radius-mm 3.0 \
  --match-harris-percentile 65 \
  --fpfh-radius-mm 3 \
  --match-mode mutual_ratio \
  --ratio-threshold 0.80 \
  --ransac-iterations 5000 \
  --ransac-inlier-mm 2.0 \
  --ransac-edge-tolerance-mm 1.5 \
  --ransac-min-inliers 3 \
  --keypoint-size 18 \
  --cad-keypoint-radius-mm 1.4 \
  --scan-keypoint-radius-mm 1.4 \
  --sensor-frame-size-mm 15 \
  --trajectory-line-width 4 \
  --show
"""

from __future__ import annotations

import argparse
import itertools
import csv
from pathlib import Path
import sys

import numpy as np

try:
    import open3d as o3d
except ImportError as exc:
    raise RuntimeError("Open3D is required: pip install open3d") from exc

try:
    import pclpybridge as pcl
except ImportError as exc:
    raise RuntimeError(
        "pclpybridge is required. Install it in the same Python environment."
    ) from exc

try:
    from scipy.spatial import cKDTree
    from scipy.spatial.distance import cdist
    from scipy.optimize import minimize
except ImportError as exc:
    raise RuntimeError("SciPy is required: pip install scipy") from exc


EPS = 1.0e-12


# -----------------------------------------------------------------------------
# Arguments
# -----------------------------------------------------------------------------


def positive_float(v: str) -> float:
    x = float(v)
    if x <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return x


def nonnegative_float(v: str) -> float:
    x = float(v)
    if x < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return x


def positive_int(v: str) -> int:
    x = int(v)
    if x <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return x


def percentile_0_100(v: str) -> float:
    x = float(v)
    if not 0.0 <= x <= 100.0:
        raise argparse.ArgumentTypeError("must be in [0,100]")
    return x


def ratio_0_1(v: str) -> float:
    x = float(v)
    if not 0.0 < x <= 1.0:
        raise argparse.ArgumentTypeError("must be in (0,1]")
    return x


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Select a high-Harris CAD point, determine a fixed LJ-V7080-like "
            "sensor orientation from the local CAD mean surface normal, "
            "detect local CAD/scan Harris keypoints, match FPFH, and connect matches."
        )
    )

    p.add_argument("cad", type=Path)
    p.add_argument("--mesh-unit", choices=("auto", "m", "mm"), default="auto")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--show", action="store_true")
    p.add_argument("--threads", type=int, default=0)

    g = p.add_argument_group("Global CAD Harris candidate")
    g.add_argument("--sample-points", type=positive_int, default=120000)
    g.add_argument("--voxel-mm", type=positive_float, default=0.5)
    g.add_argument("--harris-radius-mm", type=positive_float, default=3.0)
    g.add_argument(
        "--method",
        choices=("HARRIS", "NOBLE", "LOWE", "TOMASI", "CURVATURE"),
        default="HARRIS",
    )
    g.add_argument(
        "--harris-percentile",
        type=percentile_0_100,
        default=90.0,
        help="Keep global CAD Harris responses >= this percentile before NMS.",
    )
    g.add_argument("--nms-mm", type=positive_float, default=5.0)
    g.add_argument("--max-candidates", type=positive_int, default=250)
    g.add_argument(
        "--candidate-rank",
        type=positive_int,
        default=1,
        help="Harris-response rank used when not interactively picked.",
    )

    g = p.add_argument_group("Local-normal-structure LJ-V7080-like virtual scan")
    g.add_argument(
        "--view-normal-radius-mm",
        type=positive_float,
        default=5.0,
        help="Radius around the selected Harris point used to analyze CAD normals.",
    )
    g.add_argument(
        "--view-plane-spread-threshold",
        type=nonnegative_float,
        default=0.01,
        help=(
            "If trace(C_n) is below this value, treat the neighborhood as one "
            "dominant normal mode (plane-like)."
        ),
    )
    g.add_argument(
        "--view-corner-eigen-ratio",
        type=ratio_0_1,
        default=0.25,
        help=(
            "For non-planar neighborhoods, use 3 modes when lambda2/lambda1 "
            "is at least this value; otherwise use 2 modes."
        ),
    )
    g.add_argument(
        "--view-min-mode-fraction",
        type=ratio_0_1,
        default=0.08,
        help="Discard dominant-normal clusters with less than this support fraction.",
    )
    g.add_argument(
        "--profile-reference-axis",
        choices=("x", "y", "z"),
        default="x",
        help=(
            "Fallback global CAD axis for sensor +X when the dominant normal "
            "covariance direction is degenerate."
        ),
    )
    # LJ-V7080-like defaults at the 80 mm reference distance.
    # Scan length/step are robot-motion choices, not intrinsic sensor limits.
    g.add_argument("--scan-length-mm", type=positive_float, default=50.0)
    g.add_argument(
        "--scan-width-mm",
        type=positive_float,
        default=32.0,
        help=(
            "LJ-V7080 X width at the reference distance. "
            "Official default: 32 mm at 80 mm."
        ),
    )
    g.add_argument(
        "--near-width-mm",
        type=positive_float,
        default=25.0,
        help="LJ-V7080 X width at the NEAR Z limit. Official default: 25 mm.",
    )
    g.add_argument(
        "--far-width-mm",
        type=positive_float,
        default=39.0,
        help="LJ-V7080 X width at the FAR Z limit. Official default: 39 mm.",
    )
    g.add_argument("--scan-step-mm", type=positive_float, default=0.5)
    g.add_argument("--profile-step-mm", type=positive_float, default=0.05)
    g.add_argument("--standoff-mm", type=positive_float, default=80.0)
    g.add_argument(
        "--z-range-mm",
        type=positive_float,
        default=23.0,
        help="Allowed range around reference standoff: [d-z, d+z] [mm].",
    )
    g.add_argument(
        "--scan-noise-mm",
        type=nonnegative_float,
        default=0.0,
        help="Optional isotropic Gaussian noise applied after ray casting.",
    )
    g.add_argument("--seed", type=int, default=7)

    g = p.add_argument_group("Local CAD / scan Harris keypoints")
    g.add_argument(
        "--local-margin-mm",
        type=nonnegative_float,
        default=5.0,
        help=(
            "Margin added to the oriented CAD crop around the actual scan "
            "footprint in X/Y and LJ-V Z-range in depth."
        ),
    )
    g.add_argument(
        "--match-harris-radius-mm",
        type=positive_float,
        default=3.0,
        help="Harris3D radius used independently on local CAD and scan.",
    )
    g.add_argument(
        "--match-harris-percentile",
        type=percentile_0_100,
        default=90.0,
        help="Local CAD/scan Harris response percentile before NMS.",
    )
    g.add_argument("--match-nms-mm", type=positive_float, default=2.5)
    g.add_argument("--match-max-keypoints", type=positive_int, default=80)

    g = p.add_argument_group("FPFH correspondence")
    g.add_argument("--fpfh-radius-mm", type=positive_float, default=8.0)
    g.add_argument("--fpfh-max-nn", type=positive_int, default=100)
    g.add_argument(
        "--match-mode",
        choices=("top1", "mutual", "mutual_ratio"),
        default="mutual_ratio",
    )
    g.add_argument(
        "--ratio-threshold",
        type=ratio_0_1,
        default=0.95,
        help="Lowe-style d1/d2 threshold for mutual_ratio mode.",
    )
    g.add_argument(
        "--max-matches",
        type=positive_int,
        default=40,
        help="Show at most this many lowest-descriptor-distance correspondences.",
    )

    g = p.add_argument_group("Patch-geometry pose estimation")
    g.add_argument(
        "--fpfh-distance-threshold",
        type=positive_float,
        default=10.0,
        help=(
            "Keep only FPFH correspondences whose absolute descriptor "
            "distance is <= this threshold."
        ),
    )
    g.add_argument(
        "--geometry-patch-radius-mm",
        type=positive_float,
        default=6.0,
        help=(
            "Radius around each accepted Harris correspondence used to "
            "extract scan/CAD geometry patches."
        ),
    )
    g.add_argument(
        "--geometry-min-patch-points",
        type=positive_int,
        default=20,
        help="Minimum points required in both scan and CAD geometry patches.",
    )
    g.add_argument(
        "--geometry-inlier-mm",
        type=positive_float,
        default=1.5,
        help=(
            "Nearest-neighbor residual threshold used to score each "
            "patch-derived SE(3) candidate."
        ),
    )
    g.add_argument(
        "--geometry-normal-angle-deg",
        type=positive_float,
        default=45.0,
        help=(
            "Maximum scan-vs-CAD normal angle for a geometry inlier. "
            "Normal sign is ignored."
        ),
    )
    g.add_argument(
        "--geometry-icp-max-mm",
        type=positive_float,
        default=2.0,
        help="Maximum correspondence distance for final point-to-plane ICP refinement.",
    )
    g.add_argument(
        "--geometry-icp-iterations",
        type=positive_int,
        default=50,
        help="Maximum iterations for final point-to-plane ICP refinement.",
    )

    g = p.add_argument_group("Visualization")
    g.add_argument("--point-size", type=positive_float, default=4.0)
    g.add_argument("--keypoint-size", type=positive_float, default=18.0)
    g.add_argument(
        "--display-offset-mm",
        type=positive_float,
        default=60.0,
        help="Side-by-side offset applied to scan only in final visualization.",
    )
    g.add_argument(
        "--cad-keypoint-radius-mm",
        type=positive_float,
        default=1.4,
        help="Sphere radius used to visualize CAD Harris keypoints.",
    )
    g.add_argument(
        "--scan-keypoint-radius-mm",
        type=positive_float,
        default=1.4,
        help="Sphere radius used to visualize scan Harris keypoints.",
    )
    g.add_argument(
        "--sensor-frame-size-mm",
        type=positive_float,
        default=15.0,
        help="Displayed sensor coordinate-frame axis length.",
    )
    g.add_argument(
        "--trajectory-line-width",
        type=positive_float,
        default=4.0,
        help="Requested trajectory line width for visualization.",
    )

    return p.parse_args()


# -----------------------------------------------------------------------------
# Geometry helpers
# -----------------------------------------------------------------------------


def normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if n <= EPS:
        raise RuntimeError("Cannot normalize zero-length vector.")
    return v / n


def projected_reference_axis(normal: np.ndarray, preferred: str = "x") -> np.ndarray:
    """Project a global CAD axis onto the plane orthogonal to ``normal``."""
    n = normalize(normal)
    axes = {
        "x": np.array([1.0, 0.0, 0.0], dtype=np.float64),
        "y": np.array([0.0, 1.0, 0.0], dtype=np.float64),
        "z": np.array([0.0, 0.0, 1.0], dtype=np.float64),
    }
    order = [preferred] + [k for k in ("x", "y", "z") if k != preferred]
    for key in order:
        a = axes[key]
        t = a - np.dot(a, n) * n
        if np.linalg.norm(t) > 1.0e-6:
            return normalize(t)
    raise RuntimeError("Could not construct tangent reference axis.")


def central_ray_view_error(mesh, center, z_axis, standoff_mm: float):
    """Return central-ray hit error for choosing the global normal sign."""
    center = np.asarray(center, dtype=np.float64)
    z_axis = normalize(z_axis)
    d = standoff_mm / 1000.0
    origin = center + d * z_axis
    ray = np.hstack((origin, -z_axis)).astype(np.float32)[None, :]
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    ans = scene.cast_rays(o3d.core.Tensor(ray, dtype=o3d.core.Dtype.Float32))
    t = float(ans["t_hit"].numpy()[0])
    if not np.isfinite(t) or t <= 0.0:
        return np.inf, np.inf, None
    hit = origin - t * z_axis
    point_error = float(np.linalg.norm(hit - center))
    range_error = abs(t - d)
    return point_error, range_error, hit


def choose_global_normal_sign(mesh, center, normal, standoff_mm: float):
    """Choose one global +/- sign for CAD normals using the selected point ray."""
    n = normalize(normal)
    candidates = []
    for sign in (1.0, -1.0):
        z = sign * n
        point_err, range_err, hit = central_ray_view_error(
            mesh, center, z, standoff_mm
        )
        candidates.append((point_err, range_err, sign, z, hit))
    candidates.sort(key=lambda x: (x[0], x[1]))
    best = candidates[0]
    if not np.isfinite(best[0]):
        raise RuntimeError(
            "Neither +/- CAD normal sign produced a valid central ray hit."
        )
    return best[3], best[2], best[0], best[1]


def collect_local_normals(points, normals, center, radius_mm: float):
    """Collect unit CAD normals in a Euclidean radius around the Harris point."""
    pts = np.asarray(points, dtype=np.float64)
    nrm = renormalize_normals(np.asarray(normals, dtype=np.float64))
    center = np.asarray(center, dtype=np.float64)
    tree = cKDTree(pts)
    ids = np.asarray(
        tree.query_ball_point(center, r=float(radius_mm) / 1000.0),
        dtype=np.int64,
    )
    if len(ids) < 6:
        raise RuntimeError(
            f"Only {len(ids)} points in local-normal neighborhood. "
            "Increase --view-normal-radius-mm or reduce --voxel-mm."
        )
    local = nrm[ids]
    finite = np.all(np.isfinite(local), axis=1)
    ids = ids[finite]
    local = local[finite]
    if len(local) < 6:
        raise RuntimeError("Too few finite local CAD normals for orientation analysis.")
    return local, ids


def normal_covariance_eigensystem(normals: np.ndarray):
    """Mean-centered covariance of unit normals and descending eigensystem."""
    N = renormalize_normals(np.asarray(normals, dtype=np.float64))
    mean = np.mean(N, axis=0)
    centered = N - mean[None, :]
    C = (centered.T @ centered) / float(len(N))
    C = 0.5 * (C + C.T)
    vals, vecs = np.linalg.eigh(C)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    for j in range(3):
        v = vecs[:, j]
        dominant = int(np.argmax(np.abs(v)))
        if v[dominant] < 0.0:
            vecs[:, j] *= -1.0
    return mean, C, vals, vecs


def infer_normal_mode_count(
    eigenvalues: np.ndarray,
    plane_spread_threshold: float,
    corner_eigen_ratio: float,
):
    """Infer 1/2/3 dominant normal modes from covariance rank structure."""
    vals = np.asarray(eigenvalues, dtype=np.float64)
    spread = float(np.sum(np.maximum(vals, 0.0)))
    if spread <= float(plane_spread_threshold) or vals[0] <= EPS:
        return 1, "plane-like"
    ratio21 = float(max(vals[1], 0.0) / max(vals[0], EPS))
    if ratio21 < float(corner_eigen_ratio):
        return 2, "edge-like"
    return 3, "corner-like"


def spherical_kmeans(normals, k: int, reference_normal, max_iter: int = 60):
    """Deterministic spherical k-means for globally oriented unit normals."""
    X = renormalize_normals(np.asarray(normals, dtype=np.float64))
    ref = normalize(reference_normal)
    k = int(max(1, min(k, len(X))))

    # Deterministic farthest-point initialization on the unit sphere.
    first = int(np.argmax(X @ ref))
    centroids = [X[first].copy()]
    while len(centroids) < k:
        C = np.vstack(centroids)
        similarity = X @ C.T
        nearest_similarity = np.max(similarity, axis=1)
        idx = int(np.argmin(nearest_similarity))
        centroids.append(X[idx].copy())
    C = np.vstack(centroids)

    labels = np.zeros(len(X), dtype=np.int64)
    for _ in range(max_iter):
        new_labels = np.argmax(X @ C.T, axis=1)
        new_C = C.copy()
        for j in range(k):
            ids = np.flatnonzero(new_labels == j)
            if len(ids) == 0:
                # Re-seed with the point least represented by current centroids.
                similarity = X @ C.T
                idx = int(np.argmin(np.max(similarity, axis=1)))
                new_C[j] = X[idx]
            else:
                s = np.sum(X[ids], axis=0)
                if np.linalg.norm(s) <= EPS:
                    new_C[j] = X[ids[0]]
                else:
                    new_C[j] = normalize(s)
        if np.array_equal(new_labels, labels) and np.max(np.abs(new_C - C)) < 1e-10:
            labels = new_labels
            C = new_C
            break
        labels = new_labels
        C = new_C

    counts = np.bincount(labels, minlength=k).astype(np.int64)
    order = np.argsort(counts)[::-1]
    remap = np.empty(k, dtype=np.int64)
    remap[order] = np.arange(k)
    labels = remap[labels]
    C = C[order]
    counts = counts[order]
    return C, counts, labels


def prune_small_normal_modes(centroids, counts, min_fraction: float):
    counts = np.asarray(counts, dtype=np.int64)
    centroids = np.asarray(centroids, dtype=np.float64)
    fraction = counts / max(int(np.sum(counts)), 1)
    keep = fraction >= float(min_fraction)
    if not np.any(keep):
        keep[int(np.argmax(counts))] = True
    return centroids[keep], counts[keep], fraction[keep]


def maximin_view_direction(mode_normals, mode_counts, reference_normal):
    """
    Solve max_{||d||=1} min_k n_k^T d.

    ``d`` is the sensor +Z/outward direction; laser rays travel along -d.
    """
    M = renormalize_normals(np.asarray(mode_normals, dtype=np.float64))
    counts = np.asarray(mode_counts, dtype=np.float64)
    ref = normalize(reference_normal)

    weighted = np.sum(M * counts[:, None], axis=0)
    if np.linalg.norm(weighted) <= EPS:
        d0 = ref.copy()
    else:
        d0 = normalize(weighted)
    if np.dot(d0, ref) < 0.0:
        d0 *= -1.0
    gamma0 = float(np.min(M @ d0))
    x0 = np.hstack((d0, gamma0))

    def objective(x):
        return -float(x[3])

    constraints = [
        {"type": "eq", "fun": lambda x: float(np.dot(x[:3], x[:3]) - 1.0)},
        {"type": "ineq", "fun": lambda x: M @ x[:3] - x[3]},
    ]
    res = minimize(
        objective,
        x0,
        method="SLSQP",
        bounds=[(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)],
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 300, "disp": False},
    )

    candidates = [d0]
    if res.success and np.linalg.norm(res.x[:3]) > EPS:
        candidates.append(normalize(res.x[:3]))
    # Analytic-style fallback candidates are useful if SLSQP becomes ill-conditioned.
    candidates.extend([normalize(n) for n in M])
    for i in range(len(M)):
        for j in range(i + 1, len(M)):
            s = M[i] + M[j]
            if np.linalg.norm(s) > EPS:
                candidates.append(normalize(s))
    s_all = np.sum(M, axis=0)
    if np.linalg.norm(s_all) > EPS:
        candidates.append(normalize(s_all))

    best_d = None
    best_gamma = -np.inf
    for d in candidates:
        # Preserve the globally validated outward hemisphere when possible.
        if np.dot(d, ref) < 0.0:
            d = -d
        gamma = float(np.min(M @ d))
        if gamma > best_gamma:
            best_gamma = gamma
            best_d = d.copy()

    return normalize(best_d), float(best_gamma), bool(res.success), str(res.message)


def build_normal_structure_frame(
    view_direction: np.ndarray,
    covariance_eigenvalues: np.ndarray,
    covariance_eigenvectors: np.ndarray,
    preferred_axis: str = "x",
):
    """
    Build R_cad_sensor from maximin viewing direction and normal covariance.

    +Z = maximin outward viewing direction.
    +X = dominant normal-variation direction projected onto the +Z tangent plane.
         For a flat/degenerate neighborhood, fall back to a projected global axis.
    +Y = right-handed completion and scan-motion direction.
    """
    z_axis = normalize(view_direction)
    vals = np.asarray(covariance_eigenvalues, dtype=np.float64)
    vecs = np.asarray(covariance_eigenvectors, dtype=np.float64)

    x_source = "global-axis-fallback"
    x_axis = None
    if vals[0] > 1.0e-8:
        v1 = vecs[:, 0]
        t = v1 - np.dot(v1, z_axis) * z_axis
        if np.linalg.norm(t) > 1.0e-6:
            x_axis = normalize(t)
            # Resolve eigenvector sign deterministically using the preferred global axis.
            ref_x = projected_reference_axis(z_axis, preferred_axis)
            if np.dot(x_axis, ref_x) < 0.0:
                x_axis *= -1.0
            x_source = "normal-covariance-v1"

    if x_axis is None:
        x_axis = projected_reference_axis(z_axis, preferred_axis)

    y_axis = normalize(np.cross(z_axis, x_axis))
    x_axis = normalize(np.cross(y_axis, z_axis))
    R = np.column_stack((x_axis, y_axis, z_axis))
    if np.linalg.det(R) < 0.0:
        y_axis *= -1.0
        R = np.column_stack((x_axis, y_axis, z_axis))
    return R, x_source



def determine_mean_normal_scan_orientation(
    mesh,
    cad_points,
    cad_normals,
    center,
    selected_raw_normal,
    standoff_mm: float,
    normal_radius_mm: float,
    preferred_axis: str,
):
    """
    Simple viewing-orientation rule:

    1. Collect CAD surface normals in a Euclidean neighborhood around
       the selected Harris candidate.
    2. Choose one global +/- normal sign using a central ray test.
    3. Apply that same sign to every local normal.
    4. Compute the normalized mean surface normal.
    5. Use that mean normal as sensor +Z (outward viewing direction).
    6. Construct +X by projecting a fixed CAD reference axis onto the
       plane orthogonal to +Z, then +Y = +Z x +X.

    Laser rays travel along -Z.
    """
    reference_normal, global_sign, point_err, range_err = choose_global_normal_sign(
        mesh, center, selected_raw_normal, standoff_mm
    )

    local_normals, local_ids = collect_local_normals(
        cad_points, cad_normals, center, normal_radius_mm
    )

    # IMPORTANT:
    # Apply ONE global sign only. Do not flip individual normals toward
    # the mean, because that would destroy genuine multi-surface structure.
    local_normals = renormalize_normals(global_sign * local_normals)

    mean_vec = np.sum(local_normals, axis=0)
    mean_norm = float(np.linalg.norm(mean_vec))

    if mean_norm <= EPS:
        raise RuntimeError(
            "Local surface normals cancel out, so a stable mean viewing "
            "direction cannot be computed. Try a smaller neighborhood or "
            "fall back to normal-mode/maximin orientation."
        )

    view_direction = mean_vec / mean_norm

    # Keep the globally validated outward hemisphere.
    if np.dot(view_direction, reference_normal) < 0.0:
        view_direction *= -1.0

    # Build a deterministic right-handed sensor frame.
    # +Z: mean surface normal (outward)
    # +X: projected global reference axis (laser profile direction)
    # +Y: scan-motion direction for the current baseline
    z_axis = normalize(view_direction)
    x_axis = projected_reference_axis(z_axis, preferred_axis)
    y_axis = normalize(np.cross(z_axis, x_axis))
    x_axis = normalize(np.cross(y_axis, z_axis))

    R = np.column_stack((x_axis, y_axis, z_axis))
    if np.linalg.det(R) < 0.0:
        y_axis *= -1.0
        R = np.column_stack((x_axis, y_axis, z_axis))

    # Angular spread around the mean is useful as a diagnostic.
    dots = np.clip(local_normals @ z_axis, -1.0, 1.0)
    incidence_deg = np.degrees(np.arccos(dots))

    return {
        "R_cad_sensor": R,
        "view_direction": z_axis,
        "reference_normal": reference_normal,
        "global_normal_sign": global_sign,
        "central_point_error": point_err,
        "central_range_error": range_err,
        "local_ids": local_ids,
        "local_normals": local_normals,
        "mean_normal": z_axis,
        "mean_resultant_length": mean_norm / float(len(local_normals)),
        "incidence_mean_deg": float(np.mean(incidence_deg)),
        "incidence_median_deg": float(np.median(incidence_deg)),
        "incidence_max_deg": float(np.max(incidence_deg)),
        "profile_axis_source": f"projected-global-{preferred_axis}",
    }


def determine_fixed_scan_orientation(
    mesh,
    cad_points,
    cad_normals,
    center,
    selected_raw_normal,
    standoff_mm: float,
    normal_radius_mm: float,
    plane_spread_threshold: float,
    corner_eigen_ratio: float,
    min_mode_fraction: float,
    preferred_axis: str,
):
    """Complete local-normal covariance -> clustering -> maximin orientation pipeline."""
    reference_normal, global_sign, point_err, range_err = choose_global_normal_sign(
        mesh, center, selected_raw_normal, standoff_mm
    )

    local_normals, local_ids = collect_local_normals(
        cad_points, cad_normals, center, normal_radius_mm
    )
    # Apply ONE global sign only. Do not individually flip modes, because distinct
    # neighboring surfaces are exactly the structure we want to preserve.
    local_normals = renormalize_normals(global_sign * local_normals)

    mean_n, C, vals, vecs = normal_covariance_eigensystem(local_normals)
    requested_k, structure = infer_normal_mode_count(
        vals, plane_spread_threshold, corner_eigen_ratio
    )
    centroids, counts, labels = spherical_kmeans(
        local_normals, requested_k, reference_normal
    )
    centroids, counts, fractions = prune_small_normal_modes(
        centroids, counts, min_mode_fraction
    )

    view_direction, gamma, solver_success, solver_message = maximin_view_direction(
        centroids, counts, reference_normal
    )
    R, x_source = build_normal_structure_frame(
        view_direction, vals, vecs, preferred_axis
    )

    incidence_deg = np.degrees(
        np.arccos(np.clip(centroids @ view_direction, -1.0, 1.0))
    )
    return {
        "R_cad_sensor": R,
        "view_direction": view_direction,
        "reference_normal": reference_normal,
        "global_normal_sign": global_sign,
        "central_point_error": point_err,
        "central_range_error": range_err,
        "local_ids": local_ids,
        "local_normals": local_normals,
        "mean_normal": mean_n,
        "normal_covariance": C,
        "eigenvalues": vals,
        "eigenvectors": vecs,
        "requested_mode_count": requested_k,
        "mode_count": len(centroids),
        "structure": structure,
        "mode_normals": centroids,
        "mode_counts": counts,
        "mode_fractions": fractions,
        "mode_incidence_deg": incidence_deg,
        "maximin_gamma": gamma,
        "maximin_solver_success": solver_success,
        "maximin_solver_message": solver_message,
        "profile_axis_source": x_source,
    }

def load_centered_mesh(path: Path, mesh_unit: str):
    mesh = o3d.io.read_triangle_mesh(str(path), enable_post_processing=True)
    if mesh.is_empty() or len(mesh.triangles) == 0:
        raise RuntimeError(f"failed to load mesh: {path}")

    bbox = mesh.get_axis_aligned_bounding_box()
    center_raw = np.asarray(bbox.get_center(), dtype=np.float64)
    extent_raw = np.asarray(bbox.get_extent(), dtype=np.float64)
    diag_raw = float(np.linalg.norm(extent_raw))

    if mesh_unit == "m":
        scale = 1.0
        unit_label = "m"
    elif mesh_unit == "mm":
        scale = 0.001
        unit_label = "mm"
    else:
        scale = 0.001 if diag_raw > 10.0 else 1.0
        unit_label = "mm(auto)" if scale == 0.001 else "m(auto)"

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    vertices[:] = (vertices - center_raw) * scale

    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()

    return mesh, {
        "input_unit": unit_label,
        "diameter_mm": diag_raw * scale * 1000.0,
    }


def renormalize_normals(normals: np.ndarray) -> np.ndarray:
    normals = np.asarray(normals, dtype=np.float64)
    length = np.linalg.norm(normals, axis=1, keepdims=True)
    valid = np.isfinite(length[:, 0]) & (length[:, 0] > EPS)
    if not np.all(valid):
        raise RuntimeError(
            f"Found {int(np.count_nonzero(~valid))} invalid normals."
        )
    return normals / length


def prepare_cad_surface(mesh, sample_points: int, voxel_mm: float):
    dense = mesh.sample_points_uniformly(
        number_of_points=sample_points,
        use_triangle_normal=True,
    )
    if not dense.has_normals():
        raise RuntimeError("Open3D did not return triangle normals.")

    down = dense.voxel_down_sample(voxel_size=voxel_mm / 1000.0)
    points = np.asarray(down.points, dtype=np.float64)
    normals = renormalize_normals(np.asarray(down.normals, dtype=np.float64))

    if len(points) == 0:
        raise RuntimeError("Voxel downsampling produced an empty cloud.")

    return points, normals, len(dense.points)


def make_point_cloud(points, normals=None, color=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(
            renormalize_normals(np.asarray(normals, dtype=np.float64))
        )
    if color is not None:
        pcd.paint_uniform_color(color)
    return pcd


# -----------------------------------------------------------------------------
# Harris3D
# -----------------------------------------------------------------------------


def run_dense_harris(
    points: np.ndarray,
    normals: np.ndarray,
    radius_mm: float,
    method: str,
    threads: int,
):
    response_points, response = pcl.harris3d(
        np.asarray(points, dtype=np.float32),
        radius=radius_mm / 1000.0,
        threshold=0.0,
        nonmax=False,
        refine=False,
        method=method,
        normals=np.asarray(normals, dtype=np.float32),
        threads=threads,
    )

    response_points = np.asarray(response_points, dtype=np.float64)
    response = np.asarray(response, dtype=np.float64).reshape(-1)

    if len(response_points) != len(response):
        raise RuntimeError("PCL Harris returned mismatched output lengths.")

    if (
        len(response_points) == len(points)
        and np.allclose(response_points, points, atol=1.0e-7, rtol=0.0)
    ):
        response_normals = np.asarray(normals, dtype=np.float64)
    else:
        tree = cKDTree(points)
        _, ids = tree.query(response_points, k=1)
        response_normals = np.asarray(normals, dtype=np.float64)[ids]

    return response_points, response_normals, response


def extract_candidates(
    points: np.ndarray,
    normals: np.ndarray,
    response: np.ndarray,
    percentile_keep: float,
    nms_mm: float,
    max_candidates: int,
):
    r = np.asarray(response, dtype=np.float64)
    finite = np.isfinite(r)
    if not np.any(finite):
        raise RuntimeError("Harris returned no finite responses.")

    threshold = float(np.percentile(r[finite], percentile_keep))
    ids = np.flatnonzero(finite & (r >= threshold))
    ids = ids[np.argsort(r[ids])[::-1]]

    selected = []
    min_d2 = (nms_mm / 1000.0) ** 2

    for idx in ids:
        idx = int(idx)
        p = points[idx]
        if selected:
            q = points[np.asarray(selected, dtype=np.int64)]
            d = q - p[None, :]
            d2 = np.einsum("ij,ij->i", d, d)
            if np.any(d2 < min_d2):
                continue

        selected.append(idx)
        if len(selected) >= max_candidates:
            break

    selected = np.asarray(selected, dtype=np.int64)
    if len(selected) == 0:
        raise RuntimeError("No candidates survived Harris threshold + NMS.")

    return {
        "ids": selected,
        "points": np.asarray(points)[selected],
        "normals": np.asarray(normals)[selected],
        "response": r[selected],
        "threshold": threshold,
        "pre_nms_count": int(len(ids)),
    }


def candidate_rank_order(candidates):
    return np.argsort(np.asarray(candidates["response"], dtype=np.float64))[::-1]


def pick_candidate_interactively(mesh, candidates, fallback_rank: int, args):
    base = mesh.sample_points_uniformly(number_of_points=15000)
    base_points = np.asarray(base.points, dtype=np.float64)

    c_points = np.asarray(candidates["points"], dtype=np.float64)
    c_response = np.asarray(candidates["response"], dtype=np.float64)

    lo = float(np.min(c_response))
    hi = float(np.max(c_response))
    q = (c_response - lo) / max(hi - lo, EPS)

    candidate_colors = np.column_stack(
        (
            np.ones_like(q),
            0.15 + 0.55 * (1.0 - q),
            0.08 * np.ones_like(q),
        )
    )

    all_points = np.vstack((base_points, c_points))
    all_colors = np.vstack(
        (
            np.tile((0.55, 0.55, 0.58), (len(base_points), 1)),
            candidate_colors,
        )
    )

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(all_points)
    cloud.colors = o3d.utility.Vector3dVector(all_colors)

    print("\n[VIEW] Harris candidates")
    print("  Orange/red = retained candidates (enlarged display)")
    print("  Shift + LEFT CLICK near a candidate")
    print("  Shift + RIGHT CLICK = undo")
    print("  Q = finish")

    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(
        window_name="Pick Harris candidate | Shift+LeftClick -> Q",
        width=1440,
        height=900,
    )
    vis.add_geometry(cloud)
    opt = vis.get_render_option()
    opt.background_color = np.asarray((0.025, 0.025, 0.03))
    opt.point_size = float(args.keypoint_size)
    vis.run()
    picked = list(vis.get_picked_points())
    vis.destroy_window()

    order = candidate_rank_order(candidates)
    fallback_rank0 = min(max(int(fallback_rank), 1), len(order)) - 1
    fallback_idx = int(order[fallback_rank0])

    if not picked:
        return fallback_idx, fallback_rank0 + 1

    picked_idx = int(picked[-1])
    if not 0 <= picked_idx < len(all_points):
        return fallback_idx, fallback_rank0 + 1

    picked_point = all_points[picked_idx]
    d = np.linalg.norm(c_points - picked_point[None, :], axis=1)
    candidate_idx = int(np.argmin(d))

    inverse_rank = np.empty(len(order), dtype=np.int64)
    inverse_rank[order] = np.arange(len(order))
    rank = int(inverse_rank[candidate_idx]) + 1

    print(
        f"  nearest retained candidate = rank #{rank} | "
        f"click distance={d[candidate_idx] * 1000.0:.3f} mm"
    )
    return candidate_idx, rank


# -----------------------------------------------------------------------------
# Surface-aligned LJ-V7080-like virtual profile scan
# -----------------------------------------------------------------------------


def inclusive_axis(half_extent_m: float, step_m: float) -> np.ndarray:
    count = max(2, int(np.floor((2.0 * half_extent_m) / step_m + 0.5)) + 1)
    return np.linspace(-half_extent_m, half_extent_m, count, dtype=np.float64)


def virtual_profile_scan(
    mesh,
    candidate_center: np.ndarray,
    R_cad_sensor: np.ndarray,
    scan_length_mm: float,
    scan_width_mm: float,
    near_width_mm: float,
    far_width_mm: float,
    scan_step_mm: float,
    profile_step_mm: float,
    standoff_mm: float,
    z_range_mm: float,
    noise_mm: float,
    seed: int,
):
    """
    Ray-cast an LJ-V7080-like profile scan with a trapezoidal X-Z ROI.

    Official LJ-V7080 diffuse-reflection geometry:
      reference distance = 80 mm
      Z range            = +/-23 mm
      X width at NEAR    = 25 mm
      X width at REF     = 32 mm
      X width at FAR     = 39 mm

    Each normalized profile coordinate u in [-1, 1] becomes one fan ray.
    The lateral X footprint therefore widens with axial depth and fills the
    trapezoidal X-Z measurement ROI instead of a fixed rectangular prism.

    Sensor frame:
      +X = laser profile direction
      +Y = robot scan-motion direction
      +Z = outward from object
    Rays travel generally along -Z.
    """
    center = np.asarray(candidate_center, dtype=np.float64)
    R = np.asarray(R_cad_sensor, dtype=np.float64)

    x_axis = R[:, 0]
    y_axis = R[:, 1]
    z_axis = R[:, 2]

    d_ref_m = standoff_mm / 1000.0
    z_span_m = z_range_mm / 1000.0
    d_near_m = max(0.0, d_ref_m - z_span_m)
    d_far_m = d_ref_m + z_span_m

    half_near_m = 0.5 * near_width_mm / 1000.0
    half_ref_m = 0.5 * scan_width_mm / 1000.0
    half_far_m = 0.5 * far_width_mm / 1000.0

    if d_far_m <= d_near_m + EPS:
        raise RuntimeError("Invalid LJ-V7080 Z range.")
    if d_ref_m <= d_near_m + EPS or d_ref_m >= d_far_m - EPS:
        raise RuntimeError("Reference distance must lie strictly inside the Z range.")

    # Sample spacing is defined on the 32-mm reference plane.
    xs_ref = inclusive_axis(half_ref_m, profile_step_mm / 1000.0)
    u = np.clip(xs_ref / max(half_ref_m, EPS), -1.0, 1.0)
    ys = inclusive_axis(scan_length_mm / 2000.0, scan_step_mm / 1000.0)

    # Fan-ray geometry from the official NEAR and FAR widths.
    axial_span = d_far_m - d_near_m
    halfwidth_slope = (half_far_m - half_near_m) / axial_span
    halfwidth_at_sensor = half_near_m - halfwidth_slope * d_near_m

    sensor_center = center + z_axis * d_ref_m

    origins = []
    directions = []

    for y in ys:
        # Extrapolated ray origins at axial depth 0.
        origins_y = (
            sensor_center[None, :]
            + (u * halfwidth_at_sensor)[:, None] * x_axis[None, :]
            + y * y_axis[None, :]
        )

        dirs_y = (
            (u * halfwidth_slope)[:, None] * x_axis[None, :]
            - z_axis[None, :]
        )
        dirs_y = dirs_y / np.linalg.norm(dirs_y, axis=1, keepdims=True)

        origins.append(origins_y)
        directions.append(dirs_y)

    origins = np.vstack(origins)
    directions = np.vstack(directions)

    tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tmesh)

    rays = np.hstack((origins, directions)).astype(np.float32)
    ans = scene.cast_rays(o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32))

    t_hit = ans["t_hit"].numpy().astype(np.float64)
    finite = np.isfinite(t_hit) & (t_hit > 0.0)

    hit_points_all = np.full_like(origins, np.nan, dtype=np.float64)
    hit_points_all[finite] = (
        origins[finite] + t_hit[finite, None] * directions[finite]
    )

    # Convert hit position to sensor axial depth and profile X coordinate.
    axial_depth = np.full(len(origins), np.nan, dtype=np.float64)
    lateral_x = np.full(len(origins), np.nan, dtype=np.float64)

    if np.any(finite):
        rel = hit_points_all[finite] - sensor_center[None, :]
        axial_depth[finite] = rel @ (-z_axis)
        lateral_x[finite] = rel @ x_axis

    def half_width_at_depth(depth_m):
        depth_m = np.asarray(depth_m, dtype=np.float64)
        out = np.empty_like(depth_m)

        near_side = depth_m <= d_ref_m
        alpha_near = (
            (depth_m[near_side] - d_near_m)
            / max(d_ref_m - d_near_m, EPS)
        )
        out[near_side] = (
            half_near_m + alpha_near * (half_ref_m - half_near_m)
        )

        far_side = ~near_side
        alpha_far = (
            (depth_m[far_side] - d_ref_m)
            / max(d_far_m - d_ref_m, EPS)
        )
        out[far_side] = (
            half_ref_m + alpha_far * (half_far_m - half_ref_m)
        )
        return out

    allowed_half_width = np.full(len(origins), np.nan, dtype=np.float64)
    if np.any(finite):
        allowed_half_width[finite] = half_width_at_depth(axial_depth[finite])

    tol_m = 1.0e-6
    hit = (
        finite
        & (axial_depth >= d_near_m - tol_m)
        & (axial_depth <= d_far_m + tol_m)
        & (np.abs(lateral_x) <= allowed_half_width + tol_m)
    )

    if not np.any(hit):
        raise RuntimeError(
            "Virtual scan produced zero valid LJ-V7080 trapezoid-ROI hits."
        )

    scan_cad = hit_points_all[hit]

    if "primitive_normals" in ans:
        primitive_normals = ans["primitive_normals"].numpy().astype(np.float64)
        scan_normals_cad = primitive_normals[hit]
    else:
        scan_normals_cad = np.tile(z_axis[None, :], (len(scan_cad), 1))

    scan_normals_cad = renormalize_normals(scan_normals_cad)

    if noise_mm > 0.0:
        rng = np.random.default_rng(seed)
        scan_cad = scan_cad + rng.normal(
            0.0, noise_mm / 1000.0, size=scan_cad.shape
        )

    scan_local = (scan_cad - center[None, :]) @ R
    scan_normals_local = renormalize_normals(scan_normals_cad @ R)

    valid_depth_mm = axial_depth[hit] * 1000.0
    valid_width_mm = 2.0 * allowed_half_width[hit] * 1000.0

    return {
        "scan_points_cad": scan_cad,
        "scan_normals_cad": scan_normals_cad,
        "scan_points_local": scan_local,
        "scan_normals_local": scan_normals_local,
        "sensor_center_cad": sensor_center,
        "R_cad_sensor": R,
        "ray_count": int(len(rays)),
        "hit_count": int(np.count_nonzero(hit)),
        "profile_count": int(len(ys)),
        "samples_per_profile": int(len(u)),
        "min_range_mm": d_near_m * 1000.0,
        "max_range_mm": d_far_m * 1000.0,
        "near_width_mm": float(near_width_mm),
        "reference_width_mm": float(scan_width_mm),
        "far_width_mm": float(far_width_mm),
        "valid_depth_min_mm": float(np.min(valid_depth_mm)),
        "valid_depth_max_mm": float(np.max(valid_depth_mm)),
        "valid_roi_width_min_mm": float(np.min(valid_width_mm)),
        "valid_roi_width_max_mm": float(np.max(valid_width_mm)),
    }


# -----------------------------------------------------------------------------
# Oriented local crop, FPFH, matching
# -----------------------------------------------------------------------------


def crop_oriented_surface(
    points,
    normals,
    center,
    R_cad_sensor,
    scan_width_mm: float,
    scan_length_mm: float,
    z_range_mm: float,
    margin_mm: float,
):
    """Crop CAD surface by the sensor-aligned scan footprint + margin."""
    points = np.asarray(points, dtype=np.float64)
    normals = np.asarray(normals, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    R = np.asarray(R_cad_sensor, dtype=np.float64)

    local = (points - center[None, :]) @ R
    hx = (0.5 * scan_width_mm + margin_mm) / 1000.0
    hy = (0.5 * scan_length_mm + margin_mm) / 1000.0
    hz = (z_range_mm + margin_mm) / 1000.0

    keep = (
        (np.abs(local[:, 0]) <= hx)
        & (np.abs(local[:, 1]) <= hy)
        & (np.abs(local[:, 2]) <= hz)
    )
    ids = np.flatnonzero(keep).astype(np.int64)
    if len(ids) < 20:
        raise RuntimeError(
            f"Oriented local CAD crop has only {len(ids)} points. "
            "Increase --local-margin-mm or reduce --voxel-mm."
        )
    return points[ids], normals[ids], ids, local[ids]


def compute_fpfh(points, normals, radius_mm: float, max_nn: int):
    pcd = make_point_cloud(points, normals)
    feat = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius_mm / 1000.0,
            max_nn=int(max_nn),
        ),
    )
    F = np.asarray(feat.data, dtype=np.float64).T
    if F.shape != (len(points), 33):
        raise RuntimeError(f"Unexpected FPFH shape: {F.shape}")
    return F


def keypoint_base_indices(base_points, keypoints):
    tree = cKDTree(np.asarray(base_points, dtype=np.float64))
    d, ids = tree.query(np.asarray(keypoints, dtype=np.float64), k=1)
    return np.asarray(ids, dtype=np.int64), np.asarray(d, dtype=np.float64)


def match_descriptors(
    scan_desc: np.ndarray,
    cad_desc: np.ndarray,
    mode: str,
    ratio_threshold: float,
    max_matches: int,
):
    if len(scan_desc) == 0 or len(cad_desc) == 0:
        return []

    finite_scan = np.all(np.isfinite(scan_desc), axis=1)
    finite_cad = np.all(np.isfinite(cad_desc), axis=1)
    if not np.any(finite_scan) or not np.any(finite_cad):
        return []

    scan_ids = np.flatnonzero(finite_scan)
    cad_ids = np.flatnonzero(finite_cad)
    D = cdist(scan_desc[scan_ids], cad_desc[cad_ids], metric="euclidean")

    best_cad_local = np.argmin(D, axis=1)
    best_dist = D[np.arange(len(D)), best_cad_local]

    # CAD -> scan best mapping for mutual-NN gate.
    best_scan_local_for_cad = np.argmin(D, axis=0)

    if D.shape[1] >= 2:
        part = np.partition(D, kth=1, axis=1)
        second_dist = part[:, 1]
        ratio = best_dist / np.maximum(second_dist, EPS)
    else:
        ratio = np.zeros(len(D), dtype=np.float64)

    matches = []
    for s_local in range(len(scan_ids)):
        c_local = int(best_cad_local[s_local])
        mutual = int(best_scan_local_for_cad[c_local]) == s_local
        ratio_ok = float(ratio[s_local]) <= ratio_threshold

        keep = False
        if mode == "top1":
            keep = True
        elif mode == "mutual":
            keep = mutual
        elif mode == "mutual_ratio":
            keep = mutual and ratio_ok

        if keep:
            matches.append(
                {
                    "scan_kp": int(scan_ids[s_local]),
                    "cad_kp": int(cad_ids[c_local]),
                    "distance": float(best_dist[s_local]),
                    "ratio": float(ratio[s_local]),
                    "mutual": bool(mutual),
                }
            )

    matches.sort(key=lambda m: m["distance"])
    return matches[:max_matches]




# -----------------------------------------------------------------------------
# Patch geometry -> SE(3)
# -----------------------------------------------------------------------------


def extract_radius_patch(points, normals, center, radius_mm: float):
    """Extract a Euclidean-radius patch around ``center``."""
    P = np.asarray(points, dtype=np.float64)
    N = renormalize_normals(np.asarray(normals, dtype=np.float64))
    c = np.asarray(center, dtype=np.float64)
    tree = cKDTree(P)
    ids = np.asarray(
        tree.query_ball_point(c, r=float(radius_mm) / 1000.0),
        dtype=np.int64,
    )
    if len(ids) == 0:
        return P[:0], N[:0], ids
    return P[ids], N[ids], ids


def pca_local_frame(points, normals):
    """
    Build a local geometry frame from patch point covariance.

    The frame origin is the PATCH CENTROID, not the Harris point.
    PCA determines the three geometric axes.  The third axis sign is aligned
    as much as possible with the mean surface normal.  Remaining discrete PCA
    sign/permutation ambiguity is handled later by enumerating all 24 proper
    signed permutations.
    """
    P = np.asarray(points, dtype=np.float64)
    N = renormalize_normals(np.asarray(normals, dtype=np.float64))
    if len(P) < 3:
        raise ValueError("Need at least 3 points for patch PCA.")

    c = np.mean(P, axis=0)
    X = P - c[None, :]
    C = (X.T @ X) / float(len(P))
    C = 0.5 * (C + C.T)

    vals, vecs = np.linalg.eigh(C)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    F = vecs[:, order]

    # Ensure proper orientation.
    if np.linalg.det(F) < 0.0:
        F[:, 2] *= -1.0

    mean_n = np.sum(N, axis=0)
    if np.linalg.norm(mean_n) > EPS:
        mean_n = normalize(mean_n)
        # Prefer the least-variance PCA axis to point roughly with the mean normal.
        if np.dot(F[:, 2], mean_n) < 0.0:
            F[:, 2] *= -1.0
            F[:, 1] *= -1.0  # keep det +1

    return c, vals, F


def proper_signed_permutation_matrices():
    """Return the 24 right-handed signed permutation matrices in SO(3)."""
    mats = []
    eye = np.eye(3, dtype=np.float64)
    for perm in itertools.permutations(range(3)):
        P = eye[:, perm]
        for signs in itertools.product((-1.0, 1.0), repeat=3):
            S = P @ np.diag(signs)
            if np.linalg.det(S) > 0.5:
                mats.append(S)
    # Numerical uniqueness guard.
    unique = []
    for M in mats:
        if not any(np.allclose(M, Q) for Q in unique):
            unique.append(M)
    return unique


PROPER_SIGNED_PERMUTATIONS = proper_signed_permutation_matrices()


def transform_normals(R, normals):
    R = np.asarray(R, dtype=np.float64)
    N = np.asarray(normals, dtype=np.float64)
    return renormalize_normals(N @ R.T)


def score_pose_geometry(
    T,
    scan_points,
    scan_normals,
    cad_points,
    cad_normals,
    inlier_mm: float,
    normal_angle_deg: float,
):
    """
    Score a scan->CAD pose using actual surrounding geometry.

    A scan point is an inlier when:
      - nearest CAD point distance <= inlier_mm
      - |n_scan^T n_cad| >= cos(normal_angle_deg)

    Absolute normal dot is used because mesh/scan normal signs can differ.
    """
    S = transform_points(T, scan_points)
    SN = transform_normals(T[:3, :3], scan_normals)

    C = np.asarray(cad_points, dtype=np.float64)
    CN = renormalize_normals(np.asarray(cad_normals, dtype=np.float64))

    tree = cKDTree(C)
    d, ids = tree.query(S, k=1)
    matched_normals = CN[np.asarray(ids, dtype=np.int64)]

    normal_dot = np.abs(np.einsum("ij,ij->i", SN, matched_normals))
    cos_gate = float(np.cos(np.radians(normal_angle_deg)))

    inliers = (d <= float(inlier_mm) / 1000.0) & (normal_dot >= cos_gate)
    nin = int(np.count_nonzero(inliers))

    if nin:
        med = float(np.median(d[inliers]))
        mean = float(np.mean(d[inliers]))
        rmse = float(np.sqrt(np.mean(np.square(d[inliers]))))
    else:
        med = mean = rmse = np.inf

    return {
        "inliers": inliers,
        "inlier_count": nin,
        "inlier_ratio": float(nin / max(len(S), 1)),
        "median_m": med,
        "mean_m": mean,
        "rmse_m": rmse,
        "nn_distance_m": d,
        "normal_dot": normal_dot,
    }


def estimate_pose_from_matched_patches(
    accepted_matches,
    scan_points,
    scan_normals,
    scan_keypoints,
    cad_points,
    cad_normals,
    cad_keypoints,
    patch_radius_mm: float,
    min_patch_points: int,
    inlier_mm: float,
    normal_angle_deg: float,
):
    """
    Use FPFH matches ONLY as patch anchors.

    No SE(3) is computed from Harris keypoint coordinates themselves.

    For each accepted FPFH match:
      1) extract scan/CAD patches around the two matched Harris locations
      2) compute each patch centroid + PCA local frame
      3) enumerate the 24 proper PCA axis/sign correspondences
      4) obtain candidate SE(3) from PATCH CENTROIDS + PATCH FRAMES
      5) score each candidate using the full scan geometry against local CAD

    The best geometry-supported transform is returned.
    """
    S_all = np.asarray(scan_points, dtype=np.float64)
    SN_all = renormalize_normals(np.asarray(scan_normals, dtype=np.float64))
    C_all = np.asarray(cad_points, dtype=np.float64)
    CN_all = renormalize_normals(np.asarray(cad_normals, dtype=np.float64))

    candidates = []
    patch_records = []

    for match_id, m in enumerate(accepted_matches):
        s_anchor = np.asarray(scan_keypoints[m["scan_kp"]], dtype=np.float64)
        c_anchor = np.asarray(cad_keypoints[m["cad_kp"]], dtype=np.float64)

        s_patch, sn_patch, s_ids = extract_radius_patch(
            S_all, SN_all, s_anchor, patch_radius_mm
        )
        c_patch, cn_patch, c_ids = extract_radius_patch(
            C_all, CN_all, c_anchor, patch_radius_mm
        )

        rec = {
            "match_id": match_id,
            "scan_kp": m["scan_kp"],
            "cad_kp": m["cad_kp"],
            "fpfh_distance": m["distance"],
            "scan_patch_points": len(s_patch),
            "cad_patch_points": len(c_patch),
            "usable": False,
        }

        if len(s_patch) < int(min_patch_points) or len(c_patch) < int(min_patch_points):
            patch_records.append(rec)
            continue

        try:
            cs, vals_s, Fs = pca_local_frame(s_patch, sn_patch)
            cc, vals_c, Fc = pca_local_frame(c_patch, cn_patch)
        except (ValueError, np.linalg.LinAlgError):
            patch_records.append(rec)
            continue

        rec["usable"] = True
        rec["scan_eigenvalues"] = vals_s
        rec["cad_eigenvalues"] = vals_c
        patch_records.append(rec)

        for orient_id, Q in enumerate(PROPER_SIGNED_PERMUTATIONS):
            # Patch-frame alignment:
            #   Fc ~= R * Fs * Q^T
            # => R = Fc * Q * Fs^T
            R = Fc @ Q @ Fs.T
            if np.linalg.det(R) < 0.0:
                continue

            # Translation is from PATCH CENTROIDS, not Harris anchors.
            t = cc - R @ cs

            T = np.eye(4, dtype=np.float64)
            T[:3, :3] = R
            T[:3, 3] = t

            score = score_pose_geometry(
                T,
                S_all,
                SN_all,
                C_all,
                CN_all,
                inlier_mm=inlier_mm,
                normal_angle_deg=normal_angle_deg,
            )

            candidates.append(
                {
                    "T": T,
                    "score": score,
                    "match_id": match_id,
                    "orientation_id": orient_id,
                    "scan_centroid": cs,
                    "cad_centroid": cc,
                    "scan_eigenvalues": vals_s,
                    "cad_eigenvalues": vals_c,
                }
            )

    if not candidates:
        return {
            "success": False,
            "patch_records": patch_records,
            "candidate_count": 0,
        }

    # Primary: maximize geometry inliers.
    # Secondary: minimize median nearest-neighbor residual.
    candidates.sort(
        key=lambda x: (
            -x["score"]["inlier_count"],
            x["score"]["median_m"],
            x["score"]["rmse_m"],
        )
    )
    best = candidates[0]

    return {
        "success": best["score"]["inlier_count"] > 0,
        "T": best["T"],
        "score": best["score"],
        "source_match_id": best["match_id"],
        "orientation_id": best["orientation_id"],
        "patch_records": patch_records,
        "candidate_count": len(candidates),
        "all_candidates": candidates,
    }


def refine_pose_point_to_plane_icp(
    T_init,
    scan_points,
    scan_normals,
    cad_points,
    cad_normals,
    max_correspondence_mm: float,
    max_iterations: int,
):
    """Final local point-to-plane ICP refinement from the patch-geometry pose."""
    src = make_point_cloud(scan_points, scan_normals)
    tgt = make_point_cloud(cad_points, cad_normals)

    result = o3d.pipelines.registration.registration_icp(
        src,
        tgt,
        float(max_correspondence_mm) / 1000.0,
        np.asarray(T_init, dtype=np.float64),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=int(max_iterations)
        ),
    )
    return result


# -----------------------------------------------------------------------------
# Geometric consistency + RANSAC rigid pose estimation
# -----------------------------------------------------------------------------


def rigid_transform_kabsch(source: np.ndarray, target: np.ndarray):
    """
    Estimate R,t such that target ~= (R @ source.T).T + t.

    Uses the proper-rotation Kabsch/SVD solution (det(R)=+1).
    """
    src = np.asarray(source, dtype=np.float64)
    dst = np.asarray(target, dtype=np.float64)

    if src.shape != dst.shape or src.ndim != 2 or src.shape[1] != 3:
        raise ValueError("source and target must both have shape (N,3)")
    if len(src) < 3:
        raise ValueError("At least 3 correspondences are required.")

    cs = np.mean(src, axis=0)
    ct = np.mean(dst, axis=0)

    X = src - cs[None, :]
    Y = dst - ct[None, :]

    H = X.T @ Y
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Reflection guard.
    if np.linalg.det(R) < 0.0:
        Vt[-1, :] *= -1.0
        R = Vt.T @ U.T

    t = ct - R @ cs

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def transform_points(T: np.ndarray, points: np.ndarray):
    T = np.asarray(T, dtype=np.float64)
    P = np.asarray(points, dtype=np.float64)
    return (P @ T[:3, :3].T) + T[:3, 3][None, :]


def triangle_is_nondegenerate(points: np.ndarray, min_area_mm2: float = 0.05):
    """
    Reject nearly collinear 3-point samples.

    The returned triangle area is measured in m^2 internally; the threshold
    is exposed in mm^2 for readability.
    """
    P = np.asarray(points, dtype=np.float64)
    if P.shape != (3, 3):
        return False
    area2 = np.linalg.norm(np.cross(P[1] - P[0], P[2] - P[0]))
    # area2 = 2 * triangle area
    min_area2_m2 = 2.0 * float(min_area_mm2) * 1.0e-6
    return bool(area2 >= min_area2_m2)


def triplet_edge_consistent(
    source_triplet: np.ndarray,
    target_triplet: np.ndarray,
    tolerance_mm: float,
):
    """
    Geometric-consistency gate for a 3-correspondence RANSAC sample.

    Rigid transforms preserve pairwise distances.  Reject a triplet if any
    source-vs-target edge length differs by more than tolerance_mm.
    """
    S = np.asarray(source_triplet, dtype=np.float64)
    C = np.asarray(target_triplet, dtype=np.float64)
    tol_m = float(tolerance_mm) / 1000.0

    for i in range(3):
        for j in range(i + 1, 3):
            ds = float(np.linalg.norm(S[i] - S[j]))
            dc = float(np.linalg.norm(C[i] - C[j]))
            if abs(ds - dc) > tol_m:
                return False
    return True


def correspondence_residuals(T, source_points, target_points):
    pred = transform_points(T, source_points)
    return np.linalg.norm(pred - np.asarray(target_points, dtype=np.float64), axis=1)


def estimate_pose_ransac(
    source_points: np.ndarray,
    target_points: np.ndarray,
    iterations: int,
    inlier_mm: float,
    edge_tolerance_mm: float,
    min_inliers: int,
    seed: int,
):
    """
    Robustly estimate scan->CAD SE(3) from candidate correspondences.

    Pipeline:
      1) sample 3 correspondences
      2) reject collinear triplets
      3) pairwise edge-length geometric-consistency gate
      4) Kabsch SE(3)
      5) score all correspondences by 3D residual
      6) refine on all best inliers
      7) re-evaluate inliers once more
    """
    S = np.asarray(source_points, dtype=np.float64)
    C = np.asarray(target_points, dtype=np.float64)

    if S.shape != C.shape or S.ndim != 2 or S.shape[1] != 3:
        raise ValueError("RANSAC source/target must both have shape (N,3)")
    if len(S) < 3:
        return None

    threshold_m = float(inlier_mm) / 1000.0
    rng = np.random.default_rng(seed)

    best = None
    accepted_hypotheses = 0
    rejected_degenerate = 0
    rejected_edge = 0

    for _ in range(int(iterations)):
        ids = rng.choice(len(S), size=3, replace=False)
        s3 = S[ids]
        c3 = C[ids]

        if not triangle_is_nondegenerate(s3) or not triangle_is_nondegenerate(c3):
            rejected_degenerate += 1
            continue

        if not triplet_edge_consistent(s3, c3, edge_tolerance_mm):
            rejected_edge += 1
            continue

        try:
            T = rigid_transform_kabsch(s3, c3)
        except (ValueError, np.linalg.LinAlgError):
            rejected_degenerate += 1
            continue

        accepted_hypotheses += 1
        residuals = correspondence_residuals(T, S, C)
        inliers = residuals <= threshold_m
        nin = int(np.count_nonzero(inliers))

        if nin < 3:
            continue

        # RANSAC ranking:
        #   primary   = maximum inlier count
        #   secondary = minimum median inlier residual
        med = float(np.median(residuals[inliers]))
        mean = float(np.mean(residuals[inliers]))

        if (
            best is None
            or nin > best["inlier_count"]
            or (
                nin == best["inlier_count"]
                and med < best["median_residual_m"]
            )
        ):
            best = {
                "T": T,
                "inliers": inliers,
                "residuals": residuals,
                "inlier_count": nin,
                "median_residual_m": med,
                "mean_residual_m": mean,
                "sample_ids": np.asarray(ids, dtype=np.int64),
            }

    if best is None or best["inlier_count"] < int(min_inliers):
        return {
            "success": False,
            "accepted_hypotheses": accepted_hypotheses,
            "rejected_degenerate": rejected_degenerate,
            "rejected_edge": rejected_edge,
            "inlier_count": 0 if best is None else best["inlier_count"],
        }

    # Refine using every current inlier.
    inliers = best["inliers"]
    T_refined = rigid_transform_kabsch(S[inliers], C[inliers])

    # One final inlier update + final least-squares refit.
    residuals = correspondence_residuals(T_refined, S, C)
    inliers = residuals <= threshold_m

    if int(np.count_nonzero(inliers)) >= 3:
        T_refined = rigid_transform_kabsch(S[inliers], C[inliers])
        residuals = correspondence_residuals(T_refined, S, C)
        inliers = residuals <= threshold_m

    nin = int(np.count_nonzero(inliers))
    success = nin >= int(min_inliers)

    return {
        "success": bool(success),
        "T": T_refined if success else best["T"],
        "inliers": inliers,
        "residuals": residuals,
        "inlier_count": nin,
        "inlier_ratio": float(nin / len(S)),
        "mean_residual_m": float(np.mean(residuals[inliers])) if nin else np.inf,
        "median_residual_m": float(np.median(residuals[inliers])) if nin else np.inf,
        "max_residual_m": float(np.max(residuals[inliers])) if nin else np.inf,
        "accepted_hypotheses": accepted_hypotheses,
        "rejected_degenerate": rejected_degenerate,
        "rejected_edge": rejected_edge,
    }


def rotation_error_deg(R_est, R_gt):
    R_est = np.asarray(R_est, dtype=np.float64)
    R_gt = np.asarray(R_gt, dtype=np.float64)
    R_delta = R_est @ R_gt.T
    c = np.clip((np.trace(R_delta) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def make_transform_ground_truth(R_cad_sensor, selected_center):
    """
    Ground-truth transform for this virtual experiment only.

    scan_local was constructed as:
        p_local = (p_cad - center) @ R
    therefore:
        p_cad = R @ p_local + center
    """
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R_cad_sensor, dtype=np.float64)
    T[:3, 3] = np.asarray(selected_center, dtype=np.float64)
    return T


# -----------------------------------------------------------------------------
# Visualization / saving
# -----------------------------------------------------------------------------


def make_sphere(center, radius_mm, color):
    s = o3d.geometry.TriangleMesh.create_sphere(
        radius=radius_mm / 1000.0,
        resolution=12,
    )
    s.translate(np.asarray(center, dtype=np.float64))
    s.compute_vertex_normals()
    s.paint_uniform_color(color)
    return s


def draw_geometries(
    name: str,
    geoms,
    point_size: float = 4.0,
    line_width: float = 2.0,
):
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=name, width=1440, height=900)
    for g in geoms:
        vis.add_geometry(g)
    opt = vis.get_render_option()
    opt.background_color = np.asarray((0.025, 0.025, 0.03))
    opt.point_size = float(point_size)
    opt.line_width = float(line_width)
    vis.run()
    vis.destroy_window()



def make_sensor_frame(sensor_center, R_cad_sensor, axis_length_mm: float):
    """Create an Open3D LineSet for sensor +X/+Y/+Z axes in CAD frame."""
    c = np.asarray(sensor_center, dtype=np.float64)
    R = np.asarray(R_cad_sensor, dtype=np.float64)
    L = axis_length_mm / 1000.0

    pts = np.vstack(
        (
            c,
            c + L * R[:, 0],
            c + L * R[:, 1],
            c + L * R[:, 2],
        )
    )
    lines = np.asarray(
        [
            [0, 1],  # +X
            [0, 2],  # +Y
            [0, 3],  # +Z
        ],
        dtype=np.int32,
    )

    # Standard frame colors:
    # X=red, Y=green, Z=blue.
    colors = np.asarray(
        [
            [1.0, 0.15, 0.15],
            [0.15, 1.0, 0.15],
            [0.15, 0.35, 1.0],
        ],
        dtype=np.float64,
    )

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(colors)
    return ls


def make_scan_trajectory(
    selected_center,
    sensor_center,
    R_cad_sensor,
    scan_length_mm: float,
):
    """
    Draw the sensor-center trajectory for the current fixed-orientation scan.

    The current simulator translates the sensor along sensor +Y while keeping
    orientation fixed, centered about the selected Harris candidate.
    """
    center = np.asarray(sensor_center, dtype=np.float64)
    R = np.asarray(R_cad_sensor, dtype=np.float64)
    y_axis = R[:, 1]
    half = 0.5 * scan_length_mm / 1000.0

    p0 = center - half * y_axis
    p1 = center + half * y_axis

    pts = np.vstack((p0, p1))
    lines = np.asarray([[0, 1]], dtype=np.int32)
    colors = np.asarray([[1.0, 0.85, 0.10]], dtype=np.float64)

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(pts)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(colors)

    # Endpoints make trajectory direction/extent easier to see.
    start_sphere = make_sphere(p0, 1.2, (1.0, 0.65, 0.05))
    end_sphere = make_sphere(p1, 1.2, (1.0, 0.95, 0.15))

    return ls, start_sphere, end_sphere, p0, p1


def show_correspondences(
    mesh,
    selected_center,
    sensor_center,
    local_cad_points,
    cad_keypoints,
    scan_points_local,
    scan_keypoints_local,
    matches,
    R_cad_sensor,
    args,
):
    selected_center = np.asarray(selected_center, dtype=np.float64)
    R = np.asarray(R_cad_sensor, dtype=np.float64)

    # Bring scan to CAD frame only for display, then offset it along sensor +X
    # so the correspondence lines are visible instead of nearly zero-length.
    scan_points_cad = scan_points_local @ R.T + selected_center[None, :]
    scan_kp_cad = scan_keypoints_local @ R.T + selected_center[None, :]
    display_shift = R[:, 0] * (args.display_offset_mm / 1000.0)
    scan_points_disp = scan_points_cad + display_shift[None, :]
    scan_kp_disp = scan_kp_cad + display_shift[None, :]

    base = mesh.sample_points_uniformly(number_of_points=18000)
    base.paint_uniform_color((0.42, 0.42, 0.45))

    cad_cloud = make_point_cloud(local_cad_points, color=(0.15, 0.55, 1.0))
    scan_cloud = make_point_cloud(scan_points_disp, color=(1.0, 0.55, 0.08))

    # Draw Harris keypoints as actual spheres so they remain clearly visible
    # regardless of Open3D point-size/rendering backend behavior.
    cad_kp_spheres = [
        make_sphere(
            p,
            args.cad_keypoint_radius_mm,
            (0.15, 1.0, 0.22),
        )
        for p in np.asarray(cad_keypoints, dtype=np.float64)
    ]
    scan_kp_spheres = [
        make_sphere(
            p,
            args.scan_keypoint_radius_mm,
            (1.0, 0.12, 0.18),
        )
        for p in np.asarray(scan_kp_disp, dtype=np.float64)
    ]

    sensor_frame = make_sensor_frame(
        sensor_center,
        R,
        args.sensor_frame_size_mm,
    )
    traj_ls, traj_start, traj_end, traj_p0, traj_p1 = make_scan_trajectory(
        selected_center,
        sensor_center,
        R,
        args.scan_length_mm,
    )

    geoms = [
        base,
        cad_cloud,
        scan_cloud,
        make_sphere(selected_center, 1.6, (1.0, 1.0, 1.0)),
        sensor_frame,
        traj_ls,
        traj_start,
        traj_end,
        *cad_kp_spheres,
        *scan_kp_spheres,
    ]

    if matches:
        vertices = []
        lines = []
        colors = []
        for i, m in enumerate(matches):
            c = np.asarray(cad_keypoints[m["cad_kp"]], dtype=np.float64)
            s = np.asarray(scan_kp_disp[m["scan_kp"]], dtype=np.float64)
            vertices.extend([c, s])
            lines.append([2 * i, 2 * i + 1])
            colors.append([0.95, 0.95, 0.20])

        ls = o3d.geometry.LineSet()
        ls.points = o3d.utility.Vector3dVector(np.asarray(vertices, dtype=np.float64))
        ls.lines = o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
        ls.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))
        geoms.append(ls)

    print("\n[FINAL VIEW] Harris keypoint correspondences")
    print("  blue   = local CAD surface")
    print("  orange = virtual scan, shifted only for visualization")
    print("  green spheres = CAD Harris keypoints")
    print("  red spheres   = scan Harris keypoints")
    print("  yellow = FPFH correspondence lines")
    print("  white sphere = selected global Harris candidate")
    print("  sensor frame = X red / Y green / Z blue")
    print("  yellow trajectory = sensor-center scan path along sensor +Y")
    print(
        f"  trajectory start/end [mm] = "
        f"{np.round(traj_p0 * 1000.0, 3).tolist()} -> "
        f"{np.round(traj_p1 * 1000.0, 3).tolist()}"
    )

    draw_geometries(
        "Harris local CAD <-> scan FPFH correspondences",
        geoms,
        point_size=args.keypoint_size,
        line_width=args.trajectory_line_width,
    )


def save_xyz_csv(path: Path, points: np.ndarray, header_prefix=""):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            f"{header_prefix}x_mm",
            f"{header_prefix}y_mm",
            f"{header_prefix}z_mm",
        ])
        for p in np.asarray(points, dtype=np.float64) * 1000.0:
            w.writerow(p.tolist())


def save_matches(path: Path, matches, scan_kp, cad_kp):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "match_id",
                "scan_kp_id",
                "cad_kp_id",
                "fpfh_distance",
                "ratio",
                "mutual",
                "scan_x_mm",
                "scan_y_mm",
                "scan_z_mm",
                "cad_x_mm",
                "cad_y_mm",
                "cad_z_mm",
            ]
        )
        for i, m in enumerate(matches, start=1):
            s = np.asarray(scan_kp[m["scan_kp"]]) * 1000.0
            c = np.asarray(cad_kp[m["cad_kp"]]) * 1000.0
            w.writerow(
                [
                    i,
                    m["scan_kp"],
                    m["cad_kp"],
                    m["distance"],
                    m["ratio"],
                    int(m["mutual"]),
                    *s.tolist(),
                    *c.tolist(),
                ]
            )




def save_transform_csv(path: Path, T: np.ndarray):
    T = np.asarray(T, dtype=np.float64)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["row", "c0", "c1", "c2", "c3"])
        for i in range(4):
            w.writerow([i, *T[i].tolist()])


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> int:
    args = parse_args()

    if not args.cad.is_file():
        raise RuntimeError(f"CAD file does not exist: {args.cad}")
    if args.threads < 0:
        raise RuntimeError("--threads must be >= 0")

    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else Path(f"{args.cad.stem}_ljv7080_surface_correspondence")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Stage 1: global CAD Harris response/candidates
    # ------------------------------------------------------------------
    print("=" * 92)
    print("STAGE 1 - GLOBAL CAD HARRIS CANDIDATE")
    print("=" * 92)

    mesh, meta = load_centered_mesh(args.cad, args.mesh_unit)
    cad_points, cad_normals, dense_count = prepare_cad_surface(
        mesh,
        args.sample_points,
        args.voxel_mm,
    )

    response_points, response_normals, response = run_dense_harris(
        cad_points,
        cad_normals,
        args.harris_radius_mm,
        args.method,
        args.threads,
    )

    candidates = extract_candidates(
        response_points,
        response_normals,
        response,
        args.harris_percentile,
        args.nms_mm,
        args.max_candidates,
    )

    order = candidate_rank_order(candidates)
    print(
        f"CAD unit={meta['input_unit']} | diameter={meta['diameter_mm']:.3f} mm | "
        f"sample={dense_count:,} -> voxel={len(cad_points):,}"
    )
    print(
        f"global Harris threshold=P{args.harris_percentile:g}="
        f"{candidates['threshold']:.9g} | retained={len(candidates['points'])}"
    )
    for rank, candidate_idx in enumerate(order[:10], start=1):
        p = candidates["points"][candidate_idx] * 1000.0
        r = candidates["response"][candidate_idx]
        print(
            f"  #{rank:02d} response={r:.9g} | "
            f"({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}) mm"
        )

    if args.show:
        selected_idx, selected_rank = pick_candidate_interactively(
            mesh,
            candidates,
            args.candidate_rank,
            args,
        )
    else:
        rank0 = min(max(args.candidate_rank, 1), len(order)) - 1
        selected_idx = int(order[rank0])
        selected_rank = rank0 + 1

    selected_center = np.asarray(
        candidates["points"][selected_idx], dtype=np.float64
    )
    selected_response = float(candidates["response"][selected_idx])

    print(
        f"selected rank #{selected_rank} | response={selected_response:.9g} | "
        f"point(mm)={np.round(selected_center * 1000.0, 4).tolist()}"
    )

    # ------------------------------------------------------------------
    # Stage 2: surface-normal-aligned LJ-V7080-like virtual profile scan
    # ------------------------------------------------------------------
    print("\n" + "=" * 92)
    print("STAGE 2 - MEAN-SURFACE-NORMAL VIEW + LJ-V7080-LIKE SCAN")
    print("=" * 92)

    selected_normal_raw = np.asarray(
        candidates["normals"][selected_idx], dtype=np.float64
    )
    orientation = determine_mean_normal_scan_orientation(
        mesh=mesh,
        cad_points=response_points,
        cad_normals=response_normals,
        center=selected_center,
        selected_raw_normal=selected_normal_raw,
        standoff_mm=args.standoff_mm,
        normal_radius_mm=args.view_normal_radius_mm,
        preferred_axis=args.profile_reference_axis,
    )
    R_cad_sensor = orientation["R_cad_sensor"]

    scan = virtual_profile_scan(
        mesh,
        selected_center,
        R_cad_sensor,
        args.scan_length_mm,
        args.scan_width_mm,
        args.near_width_mm,
        args.far_width_mm,
        args.scan_step_mm,
        args.profile_step_mm,
        args.standoff_mm,
        args.z_range_mm,
        args.scan_noise_mm,
        args.seed,
    )

    print(
        f"normal neighborhood   = r={args.view_normal_radius_mm:.2f} mm | "
        f"N={len(orientation['local_normals'])}"
    )
    print(
        f"global normal sign    = {orientation['global_normal_sign']:+.0f} | "
        f"central hit error={orientation['central_point_error']*1000.0:.4f} mm | "
        f"range error={orientation['central_range_error']*1000.0:.4f} mm"
    )
    print(
        f"mean surface normal   = "
        f"{np.round(orientation['mean_normal'], 6).tolist()}"
    )
    print(
        f"mean resultant length = {orientation['mean_resultant_length']:.6f} "
        f"(1.0=all local normals agree)"
    )
    print(
        f"normal incidence      = mean {orientation['incidence_mean_deg']:.2f} deg | "
        f"median {orientation['incidence_median_deg']:.2f} deg | "
        f"max {orientation['incidence_max_deg']:.2f} deg"
    )
    print(
        f"view +Z              = {np.round(orientation['view_direction'], 6).tolist()}"
    )
    print(
        f"profile-axis source   = {orientation['profile_axis_source']}"
    )
    print(
        f"sensor +X profile     = {np.round(R_cad_sensor[:,0], 6).tolist()}"
    )
    print(
        f"sensor +Y motion      = {np.round(R_cad_sensor[:,1], 6).tolist()}"
    )
    print(
        f"sensor +Z outward     = {np.round(R_cad_sensor[:,2], 6).tolist()}"
    )
    print(
        f"LJ-V7080 X ROI        = near {scan['near_width_mm']:.2f} mm | "
        f"ref {scan['reference_width_mm']:.2f} mm | "
        f"far {scan['far_width_mm']:.2f} mm"
    )
    print(
        f"robot scan length     = {args.scan_length_mm:.2f} mm"
    )
    print(
        f"profile/scan interval = {args.profile_step_mm:.3f} / "
        f"{args.scan_step_mm:.3f} mm"
    )
    print(
        f"reference/Z range     = {args.standoff_mm:.2f} +/- "
        f"{args.z_range_mm:.2f} mm -> ray distance "
        f"[{scan['min_range_mm']:.2f}, {scan['max_range_mm']:.2f}] mm"
    )
    print(
        f"profile grid          = {scan['profile_count']} x "
        f"{scan['samples_per_profile']} = {scan['ray_count']:,} rays"
    )
    print(
        f"ray hits              = {scan['hit_count']:,} "
        f"({100.0 * scan['hit_count']/scan['ray_count']:.1f}%)"
    )
    print(
        f"valid hit depth       = {scan['valid_depth_min_mm']:.2f} .. "
        f"{scan['valid_depth_max_mm']:.2f} mm"
    )
    print(
        f"ROI width @ hit depth = {scan['valid_roi_width_min_mm']:.2f} .. "
        f"{scan['valid_roi_width_max_mm']:.2f} mm"
    )
    print(
        f"sensor center [mm]    = "
        f"{np.round(scan['sensor_center_cad'] * 1000.0, 4).tolist()}"
    )

    # ------------------------------------------------------------------
    # Stage 3: local CAD crop and independent Harris keypoints
    # ------------------------------------------------------------------
    print("\n" + "=" * 92)
    print("STAGE 3 - LOCAL CAD / SCAN HARRIS KEYPOINTS")
    print("=" * 92)

    local_cad_points, local_cad_normals, _, local_cad_sensor = crop_oriented_surface(
        response_points,
        response_normals,
        selected_center,
        R_cad_sensor,
        max(args.near_width_mm, args.scan_width_mm, args.far_width_mm),
        args.scan_length_mm,
        args.z_range_mm,
        args.local_margin_mm,
    )

    cad_h_pts, cad_h_normals, cad_h_response = run_dense_harris(
        local_cad_points,
        local_cad_normals,
        args.match_harris_radius_mm,
        args.method,
        args.threads,
    )
    cad_kp = extract_candidates(
        cad_h_pts,
        cad_h_normals,
        cad_h_response,
        args.match_harris_percentile,
        args.match_nms_mm,
        args.match_max_keypoints,
    )

    scan_h_pts, scan_h_normals, scan_h_response = run_dense_harris(
        scan["scan_points_local"],
        scan["scan_normals_local"],
        args.match_harris_radius_mm,
        args.method,
        args.threads,
    )
    scan_kp = extract_candidates(
        scan_h_pts,
        scan_h_normals,
        scan_h_response,
        args.match_harris_percentile,
        args.match_nms_mm,
        args.match_max_keypoints,
    )

    print(
        f"local CAD oriented box = "
        f"{max(args.near_width_mm, args.scan_width_mm, args.far_width_mm) + 2*args.local_margin_mm:.1f} x "
        f"{args.scan_length_mm + 2*args.local_margin_mm:.1f} x "
        f"{2*(args.z_range_mm + args.local_margin_mm):.1f} mm | "
        f"points={len(local_cad_points):,} | Harris kp={len(cad_kp['points'])}"
    )
    print(
        f"scan points           = {len(scan['scan_points_local']):,} | "
        f"Harris kp={len(scan_kp['points'])}"
    )

    # ------------------------------------------------------------------
    # Stage 4: FPFH descriptor correspondence
    # ------------------------------------------------------------------
    print("\n" + "=" * 92)
    print("STAGE 4 - FPFH KEYPOINT CORRESPONDENCE")
    print("=" * 92)

    cad_fpfh_all = compute_fpfh(
        local_cad_points,
        local_cad_normals,
        args.fpfh_radius_mm,
        args.fpfh_max_nn,
    )
    scan_fpfh_all = compute_fpfh(
        scan["scan_points_local"],
        scan["scan_normals_local"],
        args.fpfh_radius_mm,
        args.fpfh_max_nn,
    )

    cad_base_ids, cad_map_dist = keypoint_base_indices(
        local_cad_points,
        cad_kp["points"],
    )
    scan_base_ids, scan_map_dist = keypoint_base_indices(
        scan["scan_points_local"],
        scan_kp["points"],
    )

    cad_desc = cad_fpfh_all[cad_base_ids]
    scan_desc = scan_fpfh_all[scan_base_ids]

    matches = match_descriptors(
        scan_desc,
        cad_desc,
        args.match_mode,
        args.ratio_threshold,
        args.max_matches,
    )

    print(
        f"FPFH radius           = {args.fpfh_radius_mm:.2f} mm | "
        f"mode={args.match_mode} | ratio={args.ratio_threshold:.3f}"
    )
    print(
        f"keypoint->base NN max = CAD {1000.0*np.max(cad_map_dist):.5f} mm | "
        f"scan {1000.0*np.max(scan_map_dist):.5f} mm"
    )
    print(f"matches               = {len(matches)}")

    for i, m in enumerate(matches[:15], start=1):
        print(
            f"  #{i:02d} scan[{m['scan_kp']:02d}] -> cad[{m['cad_kp']:02d}] | "
            f"FPFH d={m['distance']:.5g} | ratio={m['ratio']:.4f} | "
            f"mutual={m['mutual']}"
        )

    if len(matches) == 0:
        print(
            "WARNING: no correspondence survived the selected gate. "
            "Try --match-mode mutual or top1, increase --ratio-threshold, "
            "or increase --local-margin-mm / --fpfh-radius-mm."
        )

    # ------------------------------------------------------------------
    # Stage 5: FPFH distance gate -> matched local geometry -> SE(3)
    # ------------------------------------------------------------------
    print("\n" + "=" * 92)
    print("STAGE 5 - FPFH DISTANCE GATE + LOCAL PATCH GEOMETRY SE(3)")
    print("=" * 92)

    # FPFH is used only to select trustworthy patch anchors.
    geometry_matches = [
        m for m in matches
        if m["distance"] <= args.fpfh_distance_threshold
    ]

    print(
        f"FPFH distance gate    = <= {args.fpfh_distance_threshold:.5g} | "
        f"kept={len(geometry_matches)}/{len(matches)}"
    )
    print(
        f"geometry patch        = radius {args.geometry_patch_radius_mm:.2f} mm | "
        f"min points {args.geometry_min_patch_points}"
    )
    print(
        f"geometry score gate   = distance <= {args.geometry_inlier_mm:.3f} mm | "
        f"normal <= {args.geometry_normal_angle_deg:.1f} deg"
    )

    pose_result = None

    if len(geometry_matches) == 0:
        print("pose estimation       = FAILED | no FPFH match passed absolute distance gate")
    else:
        pose_result = estimate_pose_from_matched_patches(
            accepted_matches=geometry_matches,
            scan_points=scan["scan_points_local"],
            scan_normals=scan["scan_normals_local"],
            scan_keypoints=scan_kp["points"],
            cad_points=local_cad_points,
            cad_normals=local_cad_normals,
            cad_keypoints=cad_kp["points"],
            patch_radius_mm=args.geometry_patch_radius_mm,
            min_patch_points=args.geometry_min_patch_points,
            inlier_mm=args.geometry_inlier_mm,
            normal_angle_deg=args.geometry_normal_angle_deg,
        )

        for rec in pose_result.get("patch_records", []):
            status = "USE" if rec["usable"] else "SKIP"
            print(
                f"  patch match#{rec['match_id']+1:02d} {status} | "
                f"FPFH d={rec['fpfh_distance']:.5g} | "
                f"scan N={rec['scan_patch_points']} | "
                f"CAD N={rec['cad_patch_points']}"
            )

        if not pose_result.get("success", False):
            print(
                f"pose estimation       = FAILED | "
                f"patch-derived candidates={pose_result.get('candidate_count', 0)}"
            )
        else:
            T_patch = pose_result["T"]
            sc = pose_result["score"]

            print(
                f"patch pose candidates = {pose_result['candidate_count']} | "
                f"best from accepted-match #{pose_result['source_match_id']+1}"
            )
            print(
                f"geometry support      = {sc['inlier_count']}/{len(scan['scan_points_local'])} "
                f"({100.0*sc['inlier_ratio']:.1f}%)"
            )
            print(
                f"geometry residual mm  = median {1000.0*sc['median_m']:.4f} | "
                f"mean {1000.0*sc['mean_m']:.4f} | "
                f"RMSE {1000.0*sc['rmse_m']:.4f}"
            )

            # Final refinement uses the surrounding surfaces, not keypoints.
            icp = refine_pose_point_to_plane_icp(
                T_patch,
                scan["scan_points_local"],
                scan["scan_normals_local"],
                local_cad_points,
                local_cad_normals,
                max_correspondence_mm=args.geometry_icp_max_mm,
                max_iterations=args.geometry_icp_iterations,
            )

            T_est = np.asarray(icp.transformation, dtype=np.float64)
            pose_result["T_patch"] = T_patch
            pose_result["T"] = T_est
            pose_result["icp_fitness"] = float(icp.fitness)
            pose_result["icp_rmse"] = float(icp.inlier_rmse)

            print(
                f"point-to-plane ICP    = fitness {icp.fitness:.4f} | "
                f"RMSE {1000.0*icp.inlier_rmse:.4f} mm"
            )
            print("T_cad_scan estimated:")
            for row in T_est:
                print("  " + " ".join(f"{x: .9f}" for x in row))

            # GT is evaluation only.
            T_gt = make_transform_ground_truth(R_cad_sensor, selected_center)
            t_error_mm = 1000.0 * float(
                np.linalg.norm(T_est[:3, 3] - T_gt[:3, 3])
            )
            r_error_deg = rotation_error_deg(T_est[:3, :3], T_gt[:3, :3])

            print(
                f"GT evaluation only    = translation error "
                f"{t_error_mm:.4f} mm | rotation error {r_error_deg:.4f} deg"
            )

    # Save useful intermediate data.
    save_xyz_csv(output_dir / "scan_points_local.csv", scan["scan_points_local"])
    save_xyz_csv(output_dir / "local_cad_points.csv", local_cad_points)
    save_xyz_csv(output_dir / "local_cad_points_sensor_frame.csv", local_cad_sensor)
    save_xyz_csv(output_dir / "scan_harris_keypoints_local.csv", scan_kp["points"])
    save_xyz_csv(output_dir / "cad_harris_keypoints.csv", cad_kp["points"])
    save_matches(
        output_dir / "fpfh_correspondences.csv",
        matches,
        scan_kp["points"],
        cad_kp["points"],
    )

    if pose_result is not None and pose_result.get("success", False):
        save_transform_csv(
            output_dir / "T_cad_scan_patch_geometry.csv",
            pose_result["T"],
        )
        if "T_patch" in pose_result:
            save_transform_csv(
                output_dir / "T_cad_scan_patch_geometry_pre_icp.csv",
                pose_result["T_patch"],
            )

        with (output_dir / "patch_geometry_matches.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            w = csv.writer(f)
            w.writerow(
                [
                    "accepted_match_id",
                    "scan_kp_id",
                    "cad_kp_id",
                    "fpfh_distance",
                    "scan_patch_points",
                    "cad_patch_points",
                    "usable",
                ]
            )
            for rec in pose_result.get("patch_records", []):
                w.writerow(
                    [
                        rec["match_id"] + 1,
                        rec["scan_kp"],
                        rec["cad_kp"],
                        rec["fpfh_distance"],
                        rec["scan_patch_points"],
                        rec["cad_patch_points"],
                        int(rec["usable"]),
                    ]
                )

    frame_path = output_dir / "sensor_frame_cad.csv"
    with frame_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item", "x", "y", "z"])
        w.writerow(["selected_center_m", *selected_center.tolist()])
        w.writerow(["sensor_center_m", *scan["sensor_center_cad"].tolist()])
        w.writerow(["sensor_X_in_CAD", *R_cad_sensor[:, 0].tolist()])
        w.writerow(["sensor_Y_in_CAD", *R_cad_sensor[:, 1].tolist()])
        w.writerow(["sensor_Z_in_CAD", *R_cad_sensor[:, 2].tolist()])

    print(f"outputs               = {output_dir.resolve()}")

    if args.show:
        show_correspondences(
            mesh,
            selected_center,
            scan["sensor_center_cad"],
            local_cad_points,
            cad_kp["points"],
            scan["scan_points_local"],
            scan_kp["points"],
            matches,
            R_cad_sensor,
            args,
        )

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)