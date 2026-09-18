#!/usr/bin/env python3
"""
Harris3D: Uniform(PCL) vs Gaussian window
PCA normals vs Proposed normals
PCL-style corner refinement (CLI-controlled iterations)
============================================================

For each normal field:
    - PCA
    - PROPOSED (ITERATIVE default, ONESHOT optional)

compare Harris windows:
    - UNIFORM  : native PCL Harris3D
    - GAUSSIAN : weighted normal covariance

For each combination:
    - Harris response heatmap
    - RAW/NMS corner
    - REFINED corner
    - GT corner localization error

Gaussian Harris:
    M(p) = sum_i w_i n_i n_i^T / sum_i w_i

    w_i = exp( -||q_i-p||^2 / (2 sigma_H^2) )

    R(p) = 0.04 + det(M) - 0.04 trace(M)^2

Gaussian NMS follows PCL semantics.

Gaussian refinement reproduces PCL HarrisKeypoint3D::refineCorners():
    NNT  = sum_i n_i n_i^T
    NNTp = sum_i n_i n_i^T p_i
    x*   = solve(NNT, NNTp)

It intentionally does NOT Gaussian-weight the refinement stage.
Only the Harris response aggregation is changed.

Example
-------
python harris3d_robustness_benchmark.py \
    --benchmark \
    --stl /home/choisuhyun/lvs_HandEyeCalibration/pose_estimation/data/4.stl \
    --gt-mode mesh_vertices \
    --normal-radius 3.0 \
    --proposed iterative \
    --harris-radius 4.0 \
    --gaussian-sigma 1.5 \
    --harris-threshold 1e-6 \
    --refine-max-iter 5 \
    --noise-stage post_voxel \
    --match-radii 0.25,0.5,1.0,2.0 \
    --benchmark-seeds 10 \
    --benchmark-plot

Visualization order
-------------------
For every selected Normal x Window combination:

    1) Harris response heatmap
    2) RAW NMS corner result
    3) REFINED corner result

Colors in corner windows:
    gray  : sampled point cloud
    red   : GT corner
    green : detected Harris corner
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
VOXEL_SIZE = 1.0

NORMAL_RADIUS = 2.0

SIGMA_E_DEG = 15.0
SIGMA_R_MM = 0.25
SPATIAL_SIGMA_RATIO = 0.5
PROPOSED_MAX_ITER = 5
PROPOSED_CONVERGENCE_DEG = 0.05
MIN_NEIGHBORS = 5

HARRIS_RADIUS = 3.0
HARRIS_THRESHOLD = 1e-6
MATCH_RADIUS = 2.0


# ============================================================
# Utility
# ============================================================

def finite_rows(x):
    x = np.asarray(x)
    return np.all(np.isfinite(x), axis=1)


def normalize_rows(x):
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


def safe_stats(x):
    x = np.asarray(
        x,
        dtype=np.float64,
    )

    x = x[
        np.isfinite(x)
    ]

    if len(x) == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "median": np.nan,
            "rmse": np.nan,
            "p95": np.nan,
            "max": np.nan,
        }

    return {
        "n": int(len(x)),
        "mean": float(np.mean(x)),
        "median": float(np.median(x)),
        "rmse": float(
            np.sqrt(
                np.mean(x * x)
            )
        ),
        "p95": float(
            np.percentile(x, 95)
        ),
        "max": float(np.max(x)),
    }


def save_dict_csv(path, rows):
    """
    Save heterogeneous dictionaries safely.

    Some conditions contain extra metadata
    (e.g. gaussian_sigma_mm) while others do not.
    Build the CSV header from the union of all keys
    instead of only rows[0].
    """
    if not rows:
        return

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = []
    seen = set()

    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open(
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        writer.writeheader()

        for row in rows:
            writer.writerow({
                key: row.get(key, "")
                for key in fieldnames
            })


def print_table(rows, headers):
    widths = [
        len(str(h))
        for h in headers
    ]

    for row in rows:
        for i, v in enumerate(row):
            widths[i] = max(
                widths[i],
                len(str(v)),
            )

    fmt = "  ".join(
        "{:<" + str(w) + "}"
        for w in widths
    )

    print(fmt.format(*headers))
    print(
        fmt.format(
            *[
                "-" * w
                for w in widths
            ]
        )
    )

    for row in rows:
        print(fmt.format(*row))


def unpack_harris_result(out):
    """
    Supports either:
        tuple(points, response)
    or:
        dict{"points":..., "response":...}
    """
    if isinstance(out, dict):
        points = out["points"]
        response = out["response"]
    else:
        points, response = out

    points = np.asarray(
        points,
        dtype=np.float64,
    )

    response = np.asarray(
        response,
        dtype=np.float64,
    ).reshape(-1)

    return points[:, :3].copy(), response


# ============================================================
# Mesh / sampling
# ============================================================

def load_mesh(stl_path):
    mesh = o3d.io.read_triangle_mesh(
        str(stl_path)
    )

    if mesh.is_empty():
        raise RuntimeError(
            f"Failed to load mesh: {stl_path}"
        )

    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()

    mesh.compute_triangle_normals()

    return mesh


def sample_mesh_voxel(
    mesh,
    n_dense,
    voxel_size,
    seed,
):
    o3d.utility.random.seed(
        int(seed)
    )

    dense = (
        mesh.sample_points_uniformly(
            number_of_points=int(n_dense)
        )
    )

    pcd = dense.voxel_down_sample(
        float(voxel_size)
    )

    points = np.asarray(
        pcd.points,
        dtype=np.float64,
    )

    nn = np.asarray(
        pcd.compute_nearest_neighbor_distance(),
        dtype=np.float64,
    )

    print("\n" + "=" * 72)
    print(" POINT CLOUD")
    print("=" * 72)

    print(
        f"Dense samples : "
        f"{len(dense.points)}"
    )

    print(
        f"After voxel   : "
        f"{len(points)}"
    )

    print(
        f"Voxel size    : "
        f"{voxel_size:.4f} mm"
    )

    if len(nn):
        print(
            f"Mean NN       : "
            f"{np.mean(nn):.4f} mm"
        )

        print(
            f"Median NN     : "
            f"{np.median(nn):.4f} mm"
        )

    return points


# ============================================================
# GT corners
# ============================================================

def load_gt_corner_file(path):
    path = Path(path)

    if path.suffix.lower() == ".npy":
        x = np.load(path)

    else:
        try:
            x = np.loadtxt(
                path,
                delimiter=",",
            )
        except Exception:
            x = np.loadtxt(path)

    x = np.asarray(
        x,
        dtype=np.float64,
    )

    if x.ndim == 1:
        x = x.reshape(
            1,
            -1,
        )

    if x.shape[1] < 3:
        raise ValueError(
            "GT corner file must contain x,y,z."
        )

    return x[:, :3].copy()


def cluster_incident_normals(
    normals,
    angle_deg,
):
    normals = normalize_rows(
        normals
    )

    normals = normals[
        finite_rows(normals)
    ]

    cos_thr = math.cos(
        math.radians(
            angle_deg
        )
    )

    clusters = []

    for n in normals:
        assigned = False

        for c in clusters:
            if (
                abs(
                    float(
                        np.dot(
                            n,
                            c["mean"],
                        )
                    )
                )
                >= cos_thr
            ):
                c["members"].append(
                    n
                )

                m = np.mean(
                    c["members"],
                    axis=0,
                )

                m /= max(
                    np.linalg.norm(m),
                    1e-12,
                )

                c["mean"] = m
                assigned = True
                break

        if not assigned:
            clusters.append({
                "mean": n.copy(),
                "members": [
                    n.copy()
                ],
            })

    return clusters


def extract_sharp_vertex_corners(
    mesh,
    normal_cluster_angle_deg,
    min_normal_clusters,
):
    m = o3d.geometry.TriangleMesh(
        mesh
    )

    m.remove_duplicated_vertices()
    m.remove_duplicated_triangles()
    m.remove_degenerate_triangles()
    m.compute_triangle_normals()

    vertices = np.asarray(
        m.vertices,
        dtype=np.float64,
    )

    triangles = np.asarray(
        m.triangles,
        dtype=np.int64,
    )

    face_normals = np.asarray(
        m.triangle_normals,
        dtype=np.float64,
    )

    incident = [
        []
        for _ in range(
            len(vertices)
        )
    ]

    for fi, tri in enumerate(
        triangles
    ):
        for vi in tri:
            incident[
                int(vi)
            ].append(fi)

    keep = []

    for vi, face_ids in enumerate(
        incident
    ):
        if (
            len(face_ids)
            < min_normal_clusters
        ):
            continue

        clusters = (
            cluster_incident_normals(
                face_normals[
                    np.asarray(
                        face_ids,
                        dtype=np.int64,
                    )
                ],
                normal_cluster_angle_deg,
            )
        )

        if (
            len(clusters)
            >= min_normal_clusters
        ):
            keep.append(vi)

    return vertices[
        np.asarray(
            keep,
            dtype=np.int64,
        )
    ]


def determine_gt_corners(
    mesh,
    args,
):
    if args.gt_corners is not None:
        gt = load_gt_corner_file(
            args.gt_corners
        )

        source = (
            f"file:{args.gt_corners}"
        )

    elif args.gt_mode == "mesh_vertices":
        m = o3d.geometry.TriangleMesh(
            mesh
        )

        m.remove_duplicated_vertices()

        gt = np.asarray(
            m.vertices,
            dtype=np.float64,
        ).copy()

        source = "mesh_vertices"

    else:
        gt = (
            extract_sharp_vertex_corners(
                mesh,
                normal_cluster_angle_deg=(
                    args.face_normal_cluster_angle
                ),
                min_normal_clusters=(
                    args.min_face_normal_clusters
                ),
            )
        )

        source = "sharp_vertices"

    if len(gt) == 0:
        raise RuntimeError(
            "No GT corners found."
        )

    return gt, source


# ============================================================
# PCA / Proposed normals
# ============================================================

def estimate_pca_normals(
    points,
    radius,
):
    normals, curvature = (
        pcl.estimate_normals(
            np.asarray(
                points,
                dtype=np.float32,
            ),
            radius=float(radius),
        )
    )

    normals = normalize_rows(
        np.asarray(
            normals,
            dtype=np.float64,
        )
    )

    return normals


def build_tree(points):
    pcd = o3d.geometry.PointCloud()

    pcd.points = (
        o3d.utility.Vector3dVector(
            np.asarray(
                points,
                dtype=np.float64,
            )
        )
    )

    return o3d.geometry.KDTreeFlann(
        pcd
    )


def weighted_pca_normal(
    q,
    weights,
    reference,
):
    q = np.asarray(
        q,
        dtype=np.float64,
    )

    weights = np.asarray(
        weights,
        dtype=np.float64,
    )

    valid = (
        finite_rows(q)
        & np.isfinite(weights)
        & (weights > 1e-12)
    )

    q = q[valid]
    weights = weights[valid]

    if len(q) < 3:
        return None

    sw = float(
        np.sum(weights)
    )

    if sw <= 1e-12:
        return None

    centroid = (
        np.sum(
            q * weights[:, None],
            axis=0,
        )
        / sw
    )

    centered = (
        q - centroid
    )

    C = (
        (centered * weights[:, None]).T
        @ centered
        / sw
    )

    _, V = np.linalg.eigh(C)

    n = V[:, 0]

    n_norm = np.linalg.norm(n)

    if (
        not np.isfinite(n_norm)
        or n_norm < 1e-12
    ):
        return None

    n = n / n_norm

    if (
        reference is not None
        and np.all(
            np.isfinite(reference)
        )
    ):
        ref = (
            reference
            / max(
                np.linalg.norm(reference),
                1e-12,
            )
        )

        if np.dot(n, ref) < 0.0:
            n = -n

    return n


def proposed_local_weights(
    p,
    q,
    rough_neighbor_normals,
    current_normal,
    sigma_d,
    sigma_e_rad,
    sigma_r,
):
    delta = q - p

    d = np.linalg.norm(
        delta,
        axis=1,
    )

    local_n = np.asarray(
        rough_neighbor_normals,
        dtype=np.float64,
    ).copy()

    valid_n = finite_rows(
        local_n
    )

    if np.any(valid_n):
        temp = local_n[
            valid_n
        ]

        dot = temp @ current_normal

        temp[
            dot < 0.0
        ] *= -1.0

        local_n[
            valid_n
        ] = temp

    theta = np.full(
        len(q),
        np.nan,
        dtype=np.float64,
    )

    if np.any(valid_n):
        dot = np.clip(
            local_n[valid_n]
            @ current_normal,
            -1.0,
            1.0,
        )

        theta[
            valid_n
        ] = np.arccos(dot)

    mask = (
        valid_n
        & np.isfinite(theta)
        & (d > 1e-9)
    )

    if np.sum(mask) >= 3:
        kappa_hat = float(
            np.median(
                theta[mask]
                / d[mask]
            )
        )
    else:
        kappa_hat = 0.0

    expected = (
        kappa_hat * d
    )

    theta_safe = np.where(
        np.isfinite(theta),
        theta,
        np.inf,
    )

    excess = np.maximum(
        0.0,
        theta_safe - expected,
    )

    residual = np.abs(
        delta @ current_normal
    )

    wd = np.exp(
        -(d ** 2)
        / (
            2.0
            * sigma_d
            * sigma_d
        )
    )

    we = (
        1.0
        / (
            1.0
            + (
                excess
                / sigma_e_rad
            ) ** 2
        )
    )

    wr = (
        1.0
        / (
            1.0
            + (
                residual
                / sigma_r
            ) ** 2
        )
    )

    w = wd * we * wr

    w[
        ~valid_n
    ] = 0.0

    return w


def estimate_proposed_oneshot(
    points,
    rough,
    radius,
    sigma_e_deg,
    sigma_r,
    spatial_sigma_ratio,
    min_neighbors,
):
    tree = build_tree(points)

    output = np.full_like(
        rough,
        np.nan,
    )

    sigma_d = max(
        spatial_sigma_ratio
        * radius,
        1e-9,
    )

    sigma_e_rad = (
        math.radians(
            max(
                sigma_e_deg,
                1e-6,
            )
        )
    )

    for i, p in enumerate(points):
        n0 = rough[i]

        if not np.all(
            np.isfinite(n0)
        ):
            continue

        _, idx, _ = (
            tree.search_radius_vector_3d(
                p,
                radius,
            )
        )

        idx = np.asarray(
            idx,
            dtype=np.int64,
        )

        if len(idx) < min_neighbors:
            continue

        q = points[idx]

        w = proposed_local_weights(
            p,
            q,
            rough[idx],
            n0,
            sigma_d,
            sigma_e_rad,
            sigma_r,
        )

        n = weighted_pca_normal(
            q,
            w,
            n0,
        )

        if n is not None:
            output[i] = n

    return output


def estimate_proposed_iterative(
    points,
    rough,
    radius,
    sigma_e_deg,
    sigma_r,
    spatial_sigma_ratio,
    max_iter,
    convergence_deg,
    min_neighbors,
):
    tree = build_tree(points)

    output = np.full_like(
        rough,
        np.nan,
    )

    sigma_d = max(
        spatial_sigma_ratio
        * radius,
        1e-9,
    )

    sigma_e_rad = (
        math.radians(
            max(
                sigma_e_deg,
                1e-6,
            )
        )
    )

    for i, p in enumerate(points):
        n_current = (
            rough[i].copy()
        )

        if not np.all(
            np.isfinite(
                n_current
            )
        ):
            continue

        _, idx, _ = (
            tree.search_radius_vector_3d(
                p,
                radius,
            )
        )

        idx = np.asarray(
            idx,
            dtype=np.int64,
        )

        if len(idx) < min_neighbors:
            continue

        q = points[idx]
        local_rough = rough[idx]

        for _ in range(max_iter):
            w = proposed_local_weights(
                p,
                q,
                local_rough,
                n_current,
                sigma_d,
                sigma_e_rad,
                sigma_r,
            )

            n_new = weighted_pca_normal(
                q,
                w,
                n_current,
            )

            if n_new is None:
                break

            dot = np.clip(
                abs(
                    float(
                        np.dot(
                            n_new,
                            n_current,
                        )
                    )
                ),
                0.0,
                1.0,
            )

            change_deg = (
                math.degrees(
                    math.acos(dot)
                )
            )

            n_current = n_new

            if (
                change_deg
                < convergence_deg
            ):
                break

        output[i] = n_current

    return output


def fill_invalid_with_fallback(
    normals,
    fallback,
):
    """
    pclpybridge rejects NaN normals.
    Proposed failures fall back to PCA normals.
    """
    normals = np.asarray(
        normals,
        dtype=np.float64,
    ).copy()

    fallback = np.asarray(
        fallback,
        dtype=np.float64,
    )

    invalid = (
        ~finite_rows(normals)
        | (
            np.linalg.norm(
                np.nan_to_num(
                    normals,
                    nan=0.0,
                ),
                axis=1,
            )
            < 1e-12
        )
    )

    normals[
        invalid
    ] = fallback[
        invalid
    ]

    return (
        normalize_rows(normals),
        int(np.sum(invalid)),
    )


# ============================================================
# Native PCL Uniform Harris
# ============================================================

def pcl_harris_dense_response(
    points,
    normals,
    radius,
):
    out = pcl.harris3d(
        np.asarray(
            points,
            dtype=np.float32,
        ),
        radius=float(radius),
        threshold=0.0,
        nonmax=False,
        refine=False,
        method="HARRIS",
        normals=np.asarray(
            normals,
            dtype=np.float32,
        ),
    )

    return unpack_harris_result(
        out
    )


def pcl_harris_keypoints(
    points,
    normals,
    radius,
    threshold,
    refine,
):
    out = pcl.harris3d(
        np.asarray(
            points,
            dtype=np.float32,
        ),
        radius=float(radius),
        threshold=float(threshold),
        nonmax=True,
        refine=bool(refine),
        method="HARRIS",
        normals=np.asarray(
            normals,
            dtype=np.float32,
        ),
    )

    return unpack_harris_result(
        out
    )


# ============================================================
# Gaussian Harris response
# ============================================================

def gaussian_harris_response(
    points,
    normals,
    radius,
    sigma,
):
    """
    Gaussian-weighted version of PCL Harris3D's normal covariance.

        M = sum_i w_i n_i n_i^T / sum_i w_i

        R = 0.04 + det(M) - 0.04 * trace(M)^2
    """
    points = np.asarray(
        points,
        dtype=np.float64,
    )

    normals = normalize_rows(
        normals
    )

    tree = build_tree(
        points
    )

    response = np.zeros(
        len(points),
        dtype=np.float64,
    )

    sigma = max(
        float(sigma),
        1e-9,
    )

    for i, p in enumerate(points):
        _, idx, dist2 = (
            tree.search_radius_vector_3d(
                p,
                float(radius),
            )
        )

        idx = np.asarray(
            idx,
            dtype=np.int64,
        )

        dist2 = np.asarray(
            dist2,
            dtype=np.float64,
        )

        if len(idx) == 0:
            continue

        n = normals[idx]

        valid = finite_rows(
            n
        )

        if not np.any(valid):
            continue

        n = n[valid]
        d2 = dist2[valid]

        w = np.exp(
            -d2
            / (
                2.0
                * sigma
                * sigma
            )
        )

        sw = float(
            np.sum(w)
        )

        if sw <= 1e-12:
            continue

        M = (
            (n * w[:, None]).T
            @ n
            / sw
        )

        trace = float(
            np.trace(M)
        )

        if (
            not np.isfinite(trace)
            or trace == 0.0
        ):
            continue

        det = float(
            np.linalg.det(M)
        )

        response[i] = (
            0.04
            + det
            - 0.04
            * trace
            * trace
        )

    return response


def nms_from_response(
    points,
    response,
    radius,
    threshold,
):
    """
    Same max test as PCL:
        suppress if current response < any neighbor response.
    Ties remain maxima.
    """
    points = np.asarray(
        points,
        dtype=np.float64,
    )

    response = np.asarray(
        response,
        dtype=np.float64,
    )

    tree = build_tree(
        points
    )

    keep = []

    for i, p in enumerate(points):
        r = response[i]

        if (
            not np.isfinite(r)
            or r < threshold
        ):
            continue

        _, idx, _ = (
            tree.search_radius_vector_3d(
                p,
                radius,
            )
        )

        idx = np.asarray(
            idx,
            dtype=np.int64,
        )

        if np.any(
            r < response[idx]
        ):
            continue

        keep.append(i)

    return np.asarray(
        keep,
        dtype=np.int64,
    )


# ============================================================
# PCL-equivalent refineCorners for Gaussian NMS output
# ============================================================

def pcl_style_refine_corners(
    points,
    normals,
    corner_xyz,
    radius,
    max_iterations=10,
    rcond_threshold=1e-4,
    squared_stop=1e-6,
):
    """
    Reproduces PCL HarrisKeypoint3D::refineCorners() logic.

    Per iteration:
        NNT  = sum n n^T
        NNTp = sum n n^T p
        x    = solve(NNT, NNTp)

    update only if reciprocal condition > 1e-4
    stop if squared displacement <= 1e-6
    max 10 iterations
    """
    points = np.asarray(
        points,
        dtype=np.float64,
    )

    normals = normalize_rows(
        normals
    )

    refined = np.asarray(
        corner_xyz,
        dtype=np.float64,
    ).copy()

    tree = build_tree(
        points
    )

    for c in range(
        len(refined)
    ):
        current = refined[c].copy()

        iterations = 0

        while True:
            _, idx, _ = (
                tree.search_radius_vector_3d(
                    current,
                    radius,
                )
            )

            idx = np.asarray(
                idx,
                dtype=np.int64,
            )

            NNT = np.zeros(
                (3, 3),
                dtype=np.float64,
            )

            NNTp = np.zeros(
                3,
                dtype=np.float64,
            )

            for j in idx:
                n = normals[j]

                if not np.all(
                    np.isfinite(n)
                ):
                    continue

                nnT = np.outer(
                    n,
                    n,
                )

                NNT += nnT

                NNTp += (
                    nnT
                    @ points[j]
                )

            old = current.copy()

            if np.all(
                np.isfinite(NNT)
            ):
                cond = np.linalg.cond(
                    NNT
                )

                if (
                    np.isfinite(cond)
                    and cond > 0.0
                ):
                    rcond = (
                        1.0 / cond
                    )
                else:
                    rcond = 0.0

                if (
                    rcond
                    > rcond_threshold
                ):
                    try:
                        current = (
                            np.linalg.solve(
                                NNT,
                                NNTp,
                            )
                        )
                    except np.linalg.LinAlgError:
                        pass

            diff2 = float(
                np.sum(
                    (
                        current
                        - old
                    ) ** 2
                )
            )

            if diff2 <= squared_stop:
                break

            iterations += 1

            if (
                iterations
                >= max_iterations
            ):
                break

        refined[c] = current

    return refined


def gaussian_harris_keypoints(
    points,
    normals,
    response,
    radius,
    threshold,
    refine,
    refine_max_iterations=10,
):
    kp_idx = nms_from_response(
        points,
        response,
        radius,
        threshold,
    )

    raw_xyz = np.asarray(
        points,
        dtype=np.float64,
    )[
        kp_idx
    ].copy()

    if not refine:
        return (
            raw_xyz,
            response[
                kp_idx
            ].copy(),
            kp_idx,
        )

    refined_xyz = (
        pcl_style_refine_corners(
            points=points,
            normals=normals,
            corner_xyz=raw_xyz,
            radius=radius,
            max_iterations=int(refine_max_iterations),
        )
    )

    return (
        refined_xyz,
        response[
            kp_idx
        ].copy(),
        kp_idx,
    )


# ============================================================
# GT matching / paired localization
# ============================================================

def one_to_one_match(
    gt,
    pred,
    tolerance,
):
    gt = np.asarray(
        gt,
        dtype=np.float64,
    )

    pred = np.asarray(
        pred,
        dtype=np.float64,
    )

    if (
        len(gt) == 0
        or len(pred) == 0
    ):
        return []

    D = np.linalg.norm(
        gt[:, None, :]
        - pred[None, :, :],
        axis=2,
    )

    candidates = np.argwhere(
        D <= tolerance
    )

    ordered = sorted(
        [
            (
                float(
                    D[g, p]
                ),
                int(g),
                int(p),
            )
            for g, p
            in candidates
        ],
        key=lambda x: x[0],
    )

    used_gt = set()
    used_pred = set()

    matches = []

    for d, g, p in ordered:
        if (
            g in used_gt
            or p in used_pred
        ):
            continue

        used_gt.add(g)
        used_pred.add(p)

        matches.append(
            (g, p, d)
        )

    return matches


def evaluate_raw_refined_pair(
    label,
    gt,
    raw_xyz,
    refined_xyz,
    match_radius,
):
    """
    Match GT to RAW NMS point once.

    Then evaluate RAW and REFINED coordinates against the same GT/pred pair.
    This isolates the effect of refinement.
    """
    matches = one_to_one_match(
        gt,
        raw_xyz,
        match_radius,
    )

    raw_error = []
    refined_error = []
    per_corner = []

    matched_gt = set()

    for gt_id, pred_id, raw_d in matches:
        matched_gt.add(
            gt_id
        )

        gt_xyz = gt[
            gt_id
        ]

        ref_d = float(
            np.linalg.norm(
                refined_xyz[pred_id]
                - gt_xyz
            )
        )

        raw_error.append(
            raw_d
        )

        refined_error.append(
            ref_d
        )

        per_corner.append({
            "method": label,
            "gt_corner_id": gt_id,
            "detected": 1,

            "gt_x": gt_xyz[0],
            "gt_y": gt_xyz[1],
            "gt_z": gt_xyz[2],

            "raw_x": raw_xyz[pred_id, 0],
            "raw_y": raw_xyz[pred_id, 1],
            "raw_z": raw_xyz[pred_id, 2],

            "refined_x": refined_xyz[pred_id, 0],
            "refined_y": refined_xyz[pred_id, 1],
            "refined_z": refined_xyz[pred_id, 2],

            "raw_error_mm": raw_d,
            "refined_error_mm": ref_d,
            "improvement_mm": (
                raw_d - ref_d
            ),
        })

    for gt_id, gt_xyz in enumerate(
        gt
    ):
        if gt_id in matched_gt:
            continue

        per_corner.append({
            "method": label,
            "gt_corner_id": gt_id,
            "detected": 0,

            "gt_x": gt_xyz[0],
            "gt_y": gt_xyz[1],
            "gt_z": gt_xyz[2],

            "raw_x": np.nan,
            "raw_y": np.nan,
            "raw_z": np.nan,

            "refined_x": np.nan,
            "refined_y": np.nan,
            "refined_z": np.nan,

            "raw_error_mm": np.nan,
            "refined_error_mm": np.nan,
            "improvement_mm": np.nan,
        })

    raw_error = np.asarray(
        raw_error,
        dtype=np.float64,
    )

    refined_error = np.asarray(
        refined_error,
        dtype=np.float64,
    )

    raw_s = safe_stats(
        raw_error
    )

    refined_s = safe_stats(
        refined_error
    )

    improvement = (
        raw_error
        - refined_error
    )

    return {
        "method": label,

        "n_gt": len(gt),
        "n_pred": len(raw_xyz),
        "n_matched": len(matches),

        "recall": (
            len(matches)
            / len(gt)
            if len(gt)
            else np.nan
        ),

        "precision": (
            len(matches)
            / len(raw_xyz)
            if len(raw_xyz)
            else np.nan
        ),

        "raw_mean_mm": (
            raw_s["mean"]
        ),
        "raw_median_mm": (
            raw_s["median"]
        ),
        "raw_rmse_mm": (
            raw_s["rmse"]
        ),
        "raw_p95_mm": (
            raw_s["p95"]
        ),

        "refined_mean_mm": (
            refined_s["mean"]
        ),
        "refined_median_mm": (
            refined_s["median"]
        ),
        "refined_rmse_mm": (
            refined_s["rmse"]
        ),
        "refined_p95_mm": (
            refined_s["p95"]
        ),

        "mean_improvement_mm": (
            float(
                np.mean(
                    improvement
                )
            )
            if len(improvement)
            else np.nan
        ),

        "improved_fraction": (
            float(
                np.mean(
                    improvement > 0
                )
            )
            if len(improvement)
            else np.nan
        ),
    }, per_corner


# ============================================================
# Open3D heatmap / corners
# ============================================================

def turbo_colormap(x):
    x = np.clip(
        np.asarray(
            x,
            dtype=np.float64,
        ),
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

    return np.clip(
        rgb,
        0.0,
        1.0,
    )


def show_response_heatmap(
    points,
    response,
    title,
    percentile,
    point_size,
):
    points = np.asarray(
        points,
        dtype=np.float64,
    )

    response = np.asarray(
        response,
        dtype=np.float64,
    )

    valid = np.isfinite(
        response
    )

    if np.any(valid):
        low = float(
            np.min(
                response[valid]
            )
        )

        high = float(
            np.percentile(
                response[valid],
                percentile,
            )
        )

        if (
            not np.isfinite(high)
            or high <= low
        ):
            high = float(
                np.max(
                    response[valid]
                )
            )

        scale = max(
            high - low,
            1e-12,
        )

        x = (
            response - low
        ) / scale

    else:
        low = np.nan
        high = np.nan
        x = np.zeros(
            len(response)
        )

    colors = turbo_colormap(
        np.nan_to_num(
            x,
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        )
    )

    colors[
        ~valid
    ] = [0.35, 0.35, 0.35]

    pcd = o3d.geometry.PointCloud()

    pcd.points = (
        o3d.utility.Vector3dVector(
            points
        )
    )

    pcd.colors = (
        o3d.utility.Vector3dVector(
            colors
        )
    )

    vis = (
        o3d.visualization.Visualizer()
    )

    vis.create_window(
        window_name=title,
        width=1200,
        height=900,
    )

    vis.add_geometry(
        pcd
    )

    opt = (
        vis.get_render_option()
    )

    opt.background_color = (
        np.array(
            [0.0, 0.0, 0.0]
        )
    )

    opt.point_size = float(
        point_size
    )

    opt.show_coordinate_frame = True

    print(
        f"\n{title}"
    )

    print(
        f"response color: "
        f"min={low:.6g}, "
        f"P{percentile:g}={high:.6g}"
    )

    print(
        "Close window -> next."
    )

    vis.run()
    vis.destroy_window()


def make_sphere(
    xyz,
    radius,
    color,
):
    s = (
        o3d.geometry.TriangleMesh
        .create_sphere(
            radius=float(radius)
        )
    )

    s.translate(
        np.asarray(
            xyz,
            dtype=np.float64,
        )
    )

    s.paint_uniform_color(
        color
    )

    s.compute_vertex_normals()

    return s


def show_corner_result(
    points,
    gt,
    corners,
    title,
    marker_radius,
):
    pcd = o3d.geometry.PointCloud()

    pcd.points = (
        o3d.utility.Vector3dVector(
            points
        )
    )

    pcd.paint_uniform_color(
        [0.55, 0.55, 0.55]
    )

    geometries = [
        pcd
    ]

    for p in gt:
        geometries.append(
            make_sphere(
                p,
                marker_radius,
                [1.0, 0.0, 0.0],
            )
        )

    for p in corners:
        geometries.append(
            make_sphere(
                p,
                marker_radius * 0.72,
                [0.0, 1.0, 0.0],
            )
        )

    print(
        f"\n{title}"
    )

    print(
        "red=GT, green=Harris corner"
    )

    print(
        "Close window -> next."
    )

    o3d.visualization.draw_geometries(
        geometries,
        window_name=title,
        width=1200,
        height=900,
    )


# ============================================================
# Run one Normal x Harris-window condition
# ============================================================

def run_uniform_condition(
    normal_name,
    points,
    normals,
    gt,
    args,
):
    dense_points, dense_response = (
        pcl_harris_dense_response(
            points,
            normals,
            args.harris_radius,
        )
    )

    raw_xyz, _ = (
        pcl_harris_keypoints(
            points,
            normals,
            args.harris_radius,
            args.harris_threshold,
            refine=False,
        )
    )

    refined_xyz, _ = (
        pcl_harris_keypoints(
            points,
            normals,
            args.harris_radius,
            args.harris_threshold,
            refine=True,
        )
    )

    if (
        len(raw_xyz)
        != len(refined_xyz)
    ):
        raise RuntimeError(
            "PCL raw/refined keypoint counts differ. "
            "Expected same NMS set."
        )

    label = (
        normal_name
        + "_UNIFORM"
    )

    summary, per_corner = (
        evaluate_raw_refined_pair(
            label,
            gt,
            raw_xyz,
            refined_xyz,
            args.match_radius,
        )
    )

    # Keep summary.csv schema consistent with Gaussian conditions.
    summary["gaussian_sigma_mm"] = np.nan

    return {
        "label": label,
        "response_points": (
            dense_points
        ),
        "response": (
            dense_response
        ),
        "raw_xyz": raw_xyz,
        "refined_xyz": (
            refined_xyz
        ),
        "summary": summary,
        "per_corner": (
            per_corner
        ),
    }


def run_gaussian_condition(
    normal_name,
    points,
    normals,
    gt,
    args,
    gaussian_sigma,
):
    response = (
        gaussian_harris_response(
            points=points,
            normals=normals,
            radius=args.harris_radius,
            sigma=gaussian_sigma,
        )
    )

    raw_xyz, _, kp_idx = (
        gaussian_harris_keypoints(
            points=points,
            normals=normals,
            response=response,
            radius=args.harris_radius,
            threshold=(
                args.harris_threshold
            ),
            refine=False,
        )
    )

    refined_xyz, _, kp_idx_ref = (
        gaussian_harris_keypoints(
            points=points,
            normals=normals,
            response=response,
            radius=args.harris_radius,
            threshold=(
                args.harris_threshold
            ),
            refine=True,
        )
    )

    if not np.array_equal(
        kp_idx,
        kp_idx_ref,
    ):
        raise RuntimeError(
            "Gaussian raw/refined NMS indices differ."
        )

    label = (
        normal_name
        + "_GAUSSIAN"
    )

    summary, per_corner = (
        evaluate_raw_refined_pair(
            label,
            gt,
            raw_xyz,
            refined_xyz,
            args.match_radius,
        )
    )

    summary[
        "gaussian_sigma_mm"
    ] = gaussian_sigma

    return {
        "label": label,
        "response_points": (
            np.asarray(
                points,
                dtype=np.float64,
            )
        ),
        "response": response,
        "raw_xyz": raw_xyz,
        "refined_xyz": (
            refined_xyz
        ),
        "summary": summary,
        "per_corner": (
            per_corner
        ),
    }


# ============================================================
# CLI
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Compare PCL uniform Harris3D "
            "against Gaussian-window Harris3D."
        )
    )

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
        default=VOXEL_SIZE,
    )

    p.add_argument(
        "--seed",
        type=int,
        default=0,
    )

    # Normal estimation
    p.add_argument(
        "--normal-radius",
        type=float,
        default=NORMAL_RADIUS,
    )

    p.add_argument(
        "--proposed",
        choices=[
            "oneshot",
            "iterative",
        ],
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
        default=PROPOSED_MAX_ITER,
    )

    p.add_argument(
        "--convergence-deg",
        type=float,
        default=(
            PROPOSED_CONVERGENCE_DEG
        ),
    )

    p.add_argument(
        "--min-neighbors",
        type=int,
        default=MIN_NEIGHBORS,
    )

    # Harris
    p.add_argument(
        "--harris-radius",
        type=float,
        default=HARRIS_RADIUS,
    )

    p.add_argument(
        "--harris-threshold",
        type=float,
        default=HARRIS_THRESHOLD,
    )

    p.add_argument(
        "--harris-window",
        choices=[
            "uniform",
            "gaussian",
            "both",
        ],
        default="both",
    )

    p.add_argument(
        "--gaussian-sigma",
        type=float,
        default=0.0,
        help=(
            "Gaussian Harris sigma [mm]. "
            "<=0 uses 0.5 * Harris radius."
        ),
    )

    # GT
    p.add_argument(
        "--gt-mode",
        choices=[
            "sharp_vertices",
            "mesh_vertices",
        ],
        default="sharp_vertices",
    )

    p.add_argument(
        "--gt-corners",
        default=None,
    )

    p.add_argument(
        "--face-normal-cluster-angle",
        type=float,
        default=12.0,
    )

    p.add_argument(
        "--min-face-normal-clusters",
        type=int,
        default=3,
    )

    p.add_argument(
        "--match-radius",
        type=float,
        default=MATCH_RADIUS,
    )

    # Visualization
    p.add_argument(
        "--show",
        action="store_true",
    )

    p.add_argument(
        "--response-percentile",
        type=float,
        default=99.0,
    )

    p.add_argument(
        "--heatmap-point-size",
        type=float,
        default=5.0,
    )

    p.add_argument(
        "--marker-radius",
        type=float,
        default=0.30,
    )

    # Output
    p.add_argument(
        "--output-dir",
        default=(
            "harris3d_uniform_gaussian_results"
        ),
    )

    return p.parse_args()


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    out_dir = Path(
        args.output_dir
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    mesh = load_mesh(
        args.stl
    )

    print("\n" + "=" * 72)
    print(" CAD")
    print("=" * 72)

    print(
        f"STL       : {args.stl}"
    )

    print(
        f"Vertices  : "
        f"{len(mesh.vertices)}"
    )

    print(
        f"Triangles : "
        f"{len(mesh.triangles)}"
    )

    print(
        f"Extent    : "
        f"{mesh.get_axis_aligned_bounding_box().get_extent()} mm"
    )

    gt, gt_source = (
        determine_gt_corners(
            mesh,
            args,
        )
    )

    print("\nGT CORNERS")
    print("-" * 72)

    print(
        f"source : {gt_source}"
    )

    print(
        f"count  : {len(gt)}"
    )

    points = sample_mesh_voxel(
        mesh,
        args.n_dense,
        args.voxel,
        args.seed,
    )

    # --------------------------------------------------------
    # Normal fields
    # --------------------------------------------------------
    print(
        "\nEstimating PCA normals..."
    )

    pca_normals = (
        estimate_pca_normals(
            points,
            args.normal_radius,
        )
    )

    if args.proposed == "oneshot":
        print(
            "Estimating proposed ONESHOT normals..."
        )

        proposed_normals = (
            estimate_proposed_oneshot(
                points=points,
                rough=pca_normals,
                radius=args.normal_radius,
                sigma_e_deg=args.sigma_e,
                sigma_r=args.sigma_r,
                spatial_sigma_ratio=(
                    args.spatial_sigma_ratio
                ),
                min_neighbors=(
                    args.min_neighbors
                ),
            )
        )

        proposed_name = (
            "PROPOSED_ONESHOT"
        )

    else:
        print(
            "Estimating proposed ITERATIVE normals..."
        )

        proposed_normals = (
            estimate_proposed_iterative(
                points=points,
                rough=pca_normals,
                radius=args.normal_radius,
                sigma_e_deg=args.sigma_e,
                sigma_r=args.sigma_r,
                spatial_sigma_ratio=(
                    args.spatial_sigma_ratio
                ),
                max_iter=args.max_iter,
                convergence_deg=(
                    args.convergence_deg
                ),
                min_neighbors=(
                    args.min_neighbors
                ),
            )
        )

        proposed_name = (
            "PROPOSED_ITERATIVE"
        )

    proposed_normals, fallback_count = (
        fill_invalid_with_fallback(
            proposed_normals,
            pca_normals,
        )
    )

    print(
        f"Proposed -> PCA fallback normals: "
        f"{fallback_count}/{len(points)}"
    )

    gaussian_sigma = (
        float(args.gaussian_sigma)
        if args.gaussian_sigma > 0.0
        else 0.5
        * float(
            args.harris_radius
        )
    )

    print("\nHARRIS SETTINGS")
    print("-" * 72)

    print(
        f"radius         : "
        f"{args.harris_radius:.4f} mm"
    )

    print(
        f"threshold      : "
        f"{args.harris_threshold:.6g}"
    )

    print(
        f"window         : "
        f"{args.harris_window}"
    )

    print(
        f"Gaussian sigma : "
        f"{gaussian_sigma:.4f} mm"
    )

    # --------------------------------------------------------
    # Conditions
    # --------------------------------------------------------
    normal_fields = [
        ("PCA", pca_normals),
        (
            proposed_name,
            proposed_normals,
        ),
    ]

    results = []

    for normal_name, normals in normal_fields:
        if args.harris_window in (
            "uniform",
            "both",
        ):
            print(
                f"\nRunning "
                f"{normal_name}_UNIFORM..."
            )

            results.append(
                run_uniform_condition(
                    normal_name,
                    points,
                    normals,
                    gt,
                    args,
                )
            )

        if args.harris_window in (
            "gaussian",
            "both",
        ):
            print(
                f"\nRunning "
                f"{normal_name}_GAUSSIAN..."
            )

            results.append(
                run_gaussian_condition(
                    normal_name,
                    points,
                    normals,
                    gt,
                    args,
                    gaussian_sigma,
                )
            )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------
    print("\n" + "#" * 126)
    print(
        " HARRIS3D GT CORNER LOCALIZATION "
        "- RAW vs PCL REFINE"
    )
    print("#" * 126)

    rows = []

    summary_rows = []
    per_corner_rows = []

    for result in results:
        s = result[
            "summary"
        ]

        summary_rows.append(s)

        per_corner_rows.extend(
            result[
                "per_corner"
            ]
        )

        rows.append([
            s["method"],
            str(s["n_pred"]),
            str(s["n_matched"]),
            f"{100*s['recall']:.1f}%",
            f"{100*s['precision']:.1f}%",
            f"{s['raw_median_mm']:.4f}",
            f"{s['raw_p95_mm']:.4f}",
            f"{s['refined_median_mm']:.4f}",
            f"{s['refined_p95_mm']:.4f}",
            f"{s['mean_improvement_mm']:.4f}",
            (
                f"{100*s['improved_fraction']:.1f}%"
                if np.isfinite(
                    s["improved_fraction"]
                )
                else "nan"
            ),
        ])

    print_table(
        rows,
        [
            "Method",
            "Npred",
            "Match",
            "Recall",
            "Precision",
            "RawMed",
            "RawP95",
            "RefMed",
            "RefP95",
            "MeanGain",
            "Improved",
        ],
    )

    print(
        "\nAll localization/gain units = mm."
    )

    print(
        "MeanGain > 0 means refinement "
        "reduced GT localization error."
    )

    save_dict_csv(
        out_dir
        / "summary.csv",
        summary_rows,
    )

    save_dict_csv(
        out_dir
        / "per_corner.csv",
        per_corner_rows,
    )

    # Save response + raw/refined coordinates per condition.
    for result in results:
        label = result[
            "label"
        ]

        np.savez(
            out_dir
            / (
                label
                + "_data.npz"
            ),
            response_points=(
                result[
                    "response_points"
                ]
            ),
            response=(
                result["response"]
            ),
            raw_xyz=(
                result["raw_xyz"]
            ),
            refined_xyz=(
                result[
                    "refined_xyz"
                ]
            ),
            gt_xyz=gt,
        )

    print(
        f"\nSaved results to: "
        f"{out_dir.resolve()}"
    )

    # --------------------------------------------------------
    # Open3D visualization
    # --------------------------------------------------------
    if args.show:
        for result in results:
            label = result[
                "label"
            ]

            show_response_heatmap(
                points=result[
                    "response_points"
                ],
                response=result[
                    "response"
                ],
                title=(
                    label
                    + " Harris response"
                ),
                percentile=(
                    args.response_percentile
                ),
                point_size=(
                    args.heatmap_point_size
                ),
            )

            show_corner_result(
                points=points,
                gt=gt,
                corners=result[
                    "raw_xyz"
                ],
                title=(
                    label
                    + " RAW corner"
                ),
                marker_radius=(
                    args.marker_radius
                ),
            )

            show_corner_result(
                points=points,
                gt=gt,
                corners=result[
                    "refined_xyz"
                ],
                title=(
                    label
                    + " REFINED corner"
                ),
                marker_radius=(
                    args.marker_radius
                ),
            )




# =====================================================================
# ROBUSTNESS BENCHMARK
# Noise / Outlier / Voxel / Laser-profile-like sweep
# v2: post-voxel corruption + multi-tolerance evaluation
# =====================================================================

def closest_triangle_normals_for_points(mesh, points):
    """
    GT surface normals used ONLY to synthesize normal-direction sensor noise.
    They are NOT used by Harris detection or the proposed estimator.
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

    primitive_ids = np.asarray(
        ans["primitive_ids"].numpy(),
        dtype=np.int64,
    )

    normals = tri_normals[primitive_ids]
    return normalize_rows(normals)


def sample_dense_surface(mesh, n_dense, seed):
    o3d.utility.random.seed(int(seed))

    pcd = mesh.sample_points_uniformly(
        number_of_points=int(n_dense)
    )

    return np.asarray(
        pcd.points,
        dtype=np.float64,
    )


def voxelize_points(points, voxel_size):
    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(
        np.asarray(points, dtype=np.float64)
    )

    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(
            float(voxel_size)
        )

    return np.asarray(
        pcd.points,
        dtype=np.float64,
    )


def add_noise(
    mesh,
    points,
    sigma_mm,
    rng,
    mode="isotropic",
):
    """
    Synthetic measurement noise.

    isotropic:
        xyz iid Gaussian noise

    surface_normal:
        scalar Gaussian displacement along the closest CAD face normal.
        This is closer to a depth/range error model.
    """
    points = np.asarray(
        points,
        dtype=np.float64,
    )

    sigma_mm = float(sigma_mm)

    if sigma_mm <= 0:
        return points.copy()

    if mode == "isotropic":
        noise = rng.normal(
            0.0,
            sigma_mm,
            size=points.shape,
        )

        return points + noise

    if mode == "surface_normal":
        gt_n = closest_triangle_normals_for_points(
            mesh,
            points,
        )

        scalar = rng.normal(
            0.0,
            sigma_mm,
            size=len(points),
        )

        return (
            points
            + scalar[:, None]
            * gt_n
        )

    raise ValueError(
        f"Unknown noise mode: {mode}"
    )



def add_near_surface_outliers(
    points,
    fraction,
    magnitude_mm,
    rng,
):
    """
    Add structured near-surface outliers AFTER voxelization.

    `fraction` is the desired outlier fraction in the FINAL input cloud:

        fraction = N_out / (N_in + N_out)

    Therefore:
        N_out = fraction / (1 - fraction) * N_in

    Examples for N_in = 400:
        fraction=0.01 -> about 4 outliers
        fraction=0.05 -> about 21 outliers
        fraction=0.10 -> about 44 outliers

    Outliers start from randomly selected clean points and are displaced
    by a random 3D vector. This makes them local-neighborhood contaminants,
    not trivially removable far-away points.

    Returns
    -------
    combined : (N,3)
    n_out    : number of inserted outliers
    actual_fraction : N_out / N_total
    """
    points = np.asarray(
        points,
        dtype=np.float64,
    )

    fraction = float(
        np.clip(
            fraction,
            0.0,
            0.95,
        )
    )

    if fraction <= 0.0 or len(points) == 0:
        return (
            points.copy(),
            0,
            0.0,
        )

    n_in = len(points)

    n_out = int(
        round(
            fraction
            / max(
                1.0 - fraction,
                1e-12,
            )
            * n_in
        )
    )

    n_out = max(
        n_out,
        1,
    )

    ids = rng.integers(
        0,
        n_in,
        size=n_out,
    )

    base = points[ids]

    direction = rng.normal(
        size=(n_out, 3)
    )

    direction /= np.maximum(
        np.linalg.norm(
            direction,
            axis=1,
            keepdims=True,
        ),
        1e-12,
    )

    magnitude = rng.uniform(
        0.25 * magnitude_mm,
        magnitude_mm,
        size=n_out,
    )

    outliers = (
        base
        + direction
        * magnitude[:, None]
    )

    combined = np.vstack([
        points,
        outliers,
    ])

    actual_fraction = (
        n_out
        / len(combined)
    )

    return (
        combined,
        n_out,
        float(actual_fraction),
    )

def laser_profile_like_sweep(
    points,
    axis,
    step_mm,
    band_mm,
    phase_mm=0.0,
):
    """
    Approximate an accumulated laser-profile sweep.

    Think of the laser profile as a plane. As the sensor moves,
    this plane is sampled at regular sweep positions.

    We retain surface samples that are within `band_mm/2`
    of one of those regularly spaced scan planes.

    Example:
        axis = "x"
        step = 1.0 mm

    then profiles are approximately:
        x = x0, x0+1, x0+2, ...

    This produces anisotropic stripe-like point sampling rather than
    ordinary random/full-surface sampling.
    """
    points = np.asarray(
        points,
        dtype=np.float64,
    )

    axis_map = {
        "x": 0,
        "y": 1,
        "z": 2,
    }

    if axis not in axis_map:
        raise ValueError(
            "sweep axis must be x/y/z"
        )

    step_mm = float(step_mm)
    band_mm = float(band_mm)

    if step_mm <= 0:
        raise ValueError(
            "sweep step must be > 0"
        )

    if band_mm <= 0:
        raise ValueError(
            "profile band must be > 0"
        )

    k = axis_map[axis]

    coordinate = points[:, k]

    origin = float(
        np.min(coordinate)
    ) + float(phase_mm)

    relative = (
        coordinate - origin
    )

    # Distance to the nearest regularly spaced profile plane.
    nearest = np.round(
        relative / step_mm
    ) * step_mm

    plane_distance = np.abs(
        relative - nearest
    )

    keep = (
        plane_distance
        <= 0.5 * band_mm
    )

    return points[keep]


def sanitize_after_normal_estimation(
    points,
    pca_normals,
):
    """
    The pybind normal converter rejects NaN/zero normals.

    Remove points where baseline PCA normal estimation failed.
    """
    points = np.asarray(
        points,
        dtype=np.float64,
    )

    pca_normals = np.asarray(
        pca_normals,
        dtype=np.float64,
    )

    n_norm = np.linalg.norm(
        np.nan_to_num(
            pca_normals,
            nan=0.0,
        ),
        axis=1,
    )

    valid = (
        finite_rows(points)
        & finite_rows(pca_normals)
        & (n_norm > 1e-12)
    )

    return (
        points[valid],
        normalize_rows(
            pca_normals[valid]
        ),
        int(
            np.sum(~valid)
        ),
    )


def independent_variant_metrics(
    gt,
    pred,
    tolerance,
):
    """
    Final detector metric.

    Matching is performed independently for the supplied prediction set.
    Use this separately for RAW and REFINED detections.
    """
    matches = one_to_one_match(
        gt,
        pred,
        tolerance,
    )

    error = np.asarray(
        [
            d
            for _, _, d
            in matches
        ],
        dtype=np.float64,
    )

    s = safe_stats(error)

    return {
        "matched": len(matches),

        "recall": (
            len(matches)
            / len(gt)
            if len(gt)
            else np.nan
        ),

        "precision": (
            len(matches)
            / len(pred)
            if len(pred)
            else np.nan
        ),

        "mean_mm": s["mean"],
        "median_mm": s["median"],
        "rmse_mm": s["rmse"],
        "p95_mm": s["p95"],
        "max_mm": s["max"],
    }


def evaluate_condition_extended(
    method,
    gt,
    raw_xyz,
    refined_xyz,
    match_radius,
):
    """
    Report BOTH:

    1) RAW detector metrics, independently matched.
    2) REFINED final detector metrics, independently matched.
    3) Paired refinement gain on RAW-matched corner pairs.

    This prevents two different questions from being mixed:
        "Did refinement improve the same detection?"
    vs
        "What is the final refined detector performance?"
    """
    raw = independent_variant_metrics(
        gt,
        raw_xyz,
        match_radius,
    )

    refined = independent_variant_metrics(
        gt,
        refined_xyz,
        match_radius,
    )

    paired_matches = one_to_one_match(
        gt,
        raw_xyz,
        match_radius,
    )

    paired_raw = []
    paired_refined = []

    for gt_id, pred_id, raw_d in paired_matches:
        ref_d = float(
            np.linalg.norm(
                refined_xyz[pred_id]
                - gt[gt_id]
            )
        )

        paired_raw.append(
            raw_d
        )

        paired_refined.append(
            ref_d
        )

    paired_raw = np.asarray(
        paired_raw,
        dtype=np.float64,
    )

    paired_refined = np.asarray(
        paired_refined,
        dtype=np.float64,
    )

    gain = (
        paired_raw
        - paired_refined
    )

    return {
        "method": method,

        "n_gt": len(gt),
        "n_pred_raw": len(raw_xyz),
        "n_pred_refined": len(refined_xyz),

        "raw_matched": raw["matched"],
        "raw_recall": raw["recall"],
        "raw_precision": raw["precision"],
        "raw_mean_mm": raw["mean_mm"],
        "raw_median_mm": raw["median_mm"],
        "raw_rmse_mm": raw["rmse_mm"],
        "raw_p95_mm": raw["p95_mm"],

        "refined_matched": refined["matched"],
        "refined_recall": refined["recall"],
        "refined_precision": refined["precision"],
        "refined_mean_mm": refined["mean_mm"],
        "refined_median_mm": refined["median_mm"],
        "refined_rmse_mm": refined["rmse_mm"],
        "refined_p95_mm": refined["p95_mm"],

        "paired_n": len(gain),

        "paired_mean_gain_mm": (
            float(
                np.mean(gain)
            )
            if len(gain)
            else np.nan
        ),

        "paired_median_gain_mm": (
            float(
                np.median(gain)
            )
            if len(gain)
            else np.nan
        ),

        "paired_improved_fraction": (
            float(
                np.mean(
                    gain > 0
                )
            )
            if len(gain)
            else np.nan
        ),
    }



def run_all_harris_methods_for_benchmark(
    points,
    gt,
    args,
    gaussian_sigma,
    match_radii,
):
    """
    Run the four detector variants ONCE:

        PCA + Uniform
        PCA + Gaussian
        Proposed + Uniform
        Proposed + Gaussian

    Then evaluate the exact same RAW/REFINED predictions at every
    match tolerance in `match_radii`.

    This cleanly separates:
        detector output
    from
        evaluation tolerance.
    """
    if len(points) < 20:
        return []

    # --------------------------------------------------------
    # PCA normals
    # --------------------------------------------------------
    pca_normals = estimate_pca_normals(
        points,
        args.normal_radius,
    )

    (
        points_valid,
        pca_normals,
        invalid_count,
    ) = sanitize_after_normal_estimation(
        points,
        pca_normals,
    )

    if len(points_valid) < 20:
        return []

    # --------------------------------------------------------
    # Proposed normals
    # --------------------------------------------------------
    if args.proposed == "oneshot":
        proposed = (
            estimate_proposed_oneshot(
                points=points_valid,
                rough=pca_normals,
                radius=args.normal_radius,
                sigma_e_deg=args.sigma_e,
                sigma_r=args.sigma_r,
                spatial_sigma_ratio=(
                    args.spatial_sigma_ratio
                ),
                min_neighbors=(
                    args.min_neighbors
                ),
            )
        )

        proposed_name = (
            "PROPOSED_ONESHOT"
        )

    else:
        proposed = (
            estimate_proposed_iterative(
                points=points_valid,
                rough=pca_normals,
                radius=args.normal_radius,
                sigma_e_deg=args.sigma_e,
                sigma_r=args.sigma_r,
                spatial_sigma_ratio=(
                    args.spatial_sigma_ratio
                ),
                max_iter=args.max_iter,
                convergence_deg=(
                    args.convergence_deg
                ),
                min_neighbors=(
                    args.min_neighbors
                ),
            )
        )

        proposed_name = (
            "PROPOSED_ITERATIVE"
        )

    proposed, fallback_count = (
        fill_invalid_with_fallback(
            proposed,
            pca_normals,
        )
    )

    normal_fields = [
        ("PCA", pca_normals),
        (
            proposed_name,
            proposed,
        ),
    ]

    rows = []

    def append_all_tolerances(
        method_name,
        raw_xyz,
        refined_xyz,
        fallback_value,
    ):
        for tolerance in match_radii:
            metrics = (
                evaluate_condition_extended(
                    method_name,
                    gt,
                    raw_xyz,
                    refined_xyz,
                    float(tolerance),
                )
            )

            metrics[
                "match_radius_mm"
            ] = float(tolerance)

            metrics[
                "n_observed_points"
            ] = len(points_valid)

            metrics[
                "invalid_normal_points_removed"
            ] = invalid_count

            metrics[
                "proposed_fallback_count"
            ] = fallback_value

            rows.append(
                metrics
            )

    for normal_name, normals in normal_fields:
        fallback_value = (
            fallback_count
            if normal_name != "PCA"
            else 0
        )

        # ----------------------------------------------------
        # Uniform PCL Harris: RAW NMS
        # ----------------------------------------------------
        raw_xyz, _ = (
            pcl_harris_keypoints(
                points_valid,
                normals,
                args.harris_radius,
                args.harris_threshold,
                refine=False,
            )
        )

        # Shared PCL-style refinement so the iteration count
        # is controlled identically for Uniform/Gaussian.
        refined_xyz = (
            pcl_style_refine_corners(
                points=points_valid,
                normals=normals,
                corner_xyz=raw_xyz,
                radius=args.harris_radius,
                max_iterations=(
                    args.refine_max_iter
                ),
            )
        )

        append_all_tolerances(
            normal_name
            + "_UNIFORM",
            raw_xyz,
            refined_xyz,
            fallback_value,
        )

        # ----------------------------------------------------
        # Gaussian Harris
        # ----------------------------------------------------
        response = (
            gaussian_harris_response(
                points=points_valid,
                normals=normals,
                radius=args.harris_radius,
                sigma=gaussian_sigma,
            )
        )

        raw_g, _, idx_g = (
            gaussian_harris_keypoints(
                points=points_valid,
                normals=normals,
                response=response,
                radius=args.harris_radius,
                threshold=(
                    args.harris_threshold
                ),
                refine=False,
            )
        )

        refined_g, _, idx_ref = (
            gaussian_harris_keypoints(
                points=points_valid,
                normals=normals,
                response=response,
                radius=args.harris_radius,
                threshold=(
                    args.harris_threshold
                ),
                refine=True,
                refine_max_iterations=(
                    args.refine_max_iter
                ),
            )
        )

        if not np.array_equal(
            idx_g,
            idx_ref,
        ):
            raise RuntimeError(
                "Gaussian RAW/REFINE NMS index sets differ."
            )

        append_all_tolerances(
            normal_name
            + "_GAUSSIAN",
            raw_g,
            refined_g,
            fallback_value,
        )

    return rows


def make_observation_for_benchmark(
    mesh,
    dense_points,
    experiment,
    level,
    args,
    rng,
):
    """
    Generate one observed point cloud with explicit corruption ordering.

    ------------------------------------------------------------
    noise, stage=pre_voxel:
        dense surface
        -> noise
        -> voxel

    noise, stage=post_voxel:
        dense surface
        -> voxel
        -> noise

        This is the recommended mode for testing the robustness of
        the normal/Harris algorithm itself because the requested sigma
        is directly present in the final point coordinates.

    outlier:
        dense surface
        -> voxel
        -> add outliers

        Therefore the requested outlier fraction refers to the FINAL
        point cloud, not to the 300k dense pre-voxel samples.

    voxel:
        dense surface
        -> requested voxel size

    sweep:
        dense surface
        -> profile-plane stripe sampling
        -> voxel
        -> optional final-cloud noise
        -> optional final-cloud outliers
    ------------------------------------------------------------

    Returns
    -------
    points
    voxel_size
    metadata
    """
    dense = np.asarray(
        dense_points,
        dtype=np.float64,
    ).copy()

    voxel = float(
        args.benchmark_base_voxel
    )

    metadata = {
        "noise_stage": "",
        "n_clean_points": np.nan,
        "n_outliers_added": 0,
        "actual_outlier_fraction": 0.0,
        "sweep_phase_mm": np.nan,
    }

    # --------------------------------------------------------
    # Noise robustness
    # --------------------------------------------------------
    if experiment == "noise":
        sigma = float(level)

        metadata[
            "noise_stage"
        ] = args.noise_stage

        if args.noise_stage == "pre_voxel":
            noisy = add_noise(
                mesh,
                dense,
                sigma_mm=sigma,
                rng=rng,
                mode=args.noise_mode,
            )

            p = voxelize_points(
                noisy,
                voxel,
            )

        elif args.noise_stage == "post_voxel":
            p = voxelize_points(
                dense,
                voxel,
            )

            metadata[
                "n_clean_points"
            ] = len(p)

            p = add_noise(
                mesh,
                p,
                sigma_mm=sigma,
                rng=rng,
                mode=args.noise_mode,
            )

        else:
            raise ValueError(
                f"Unknown noise stage: {args.noise_stage}"
            )

    # --------------------------------------------------------
    # Outlier robustness
    # --------------------------------------------------------
    elif experiment == "outlier":
        fraction = float(level)

        # Critical: voxelize FIRST.
        clean = voxelize_points(
            dense,
            voxel,
        )

        metadata[
            "n_clean_points"
        ] = len(clean)

        (
            p,
            n_out,
            actual_fraction,
        ) = add_near_surface_outliers(
            clean,
            fraction=fraction,
            magnitude_mm=(
                args.outlier_magnitude
            ),
            rng=rng,
        )

        metadata[
            "n_outliers_added"
        ] = n_out

        metadata[
            "actual_outlier_fraction"
        ] = actual_fraction

    # --------------------------------------------------------
    # Voxel robustness
    # --------------------------------------------------------
    elif experiment == "voxel":
        voxel = float(level)

        p = voxelize_points(
            dense,
            voxel,
        )

        metadata[
            "n_clean_points"
        ] = len(p)

    # --------------------------------------------------------
    # Laser-profile-like stripe sampling
    # --------------------------------------------------------
    elif experiment == "sweep":
        step = float(level)

        # Random phase across seeds prevents accidentally evaluating
        # only a favorable profile-grid alignment.
        phase = rng.uniform(
            0.0,
            step,
        )

        metadata[
            "sweep_phase_mm"
        ] = float(phase)

        striped = laser_profile_like_sweep(
            dense,
            axis=args.sweep_axis,
            step_mm=step,
            band_mm=(
                args.profile_band
            ),
            phase_mm=phase,
        )

        p = voxelize_points(
            striped,
            voxel,
        )

        metadata[
            "n_clean_points"
        ] = len(p)

        # Treat optional sweep noise as final measurement noise.
        if args.sweep_noise > 0:
            p = add_noise(
                mesh,
                p,
                sigma_mm=(
                    args.sweep_noise
                ),
                rng=rng,
                mode=args.noise_mode,
            )

        # Likewise, contaminate the final profile cloud.
        if args.sweep_outlier_ratio > 0:
            (
                p,
                n_out,
                actual_fraction,
            ) = add_near_surface_outliers(
                p,
                fraction=(
                    args.sweep_outlier_ratio
                ),
                magnitude_mm=(
                    args.outlier_magnitude
                ),
                rng=rng,
            )

            metadata[
                "n_outliers_added"
            ] = n_out

            metadata[
                "actual_outlier_fraction"
            ] = actual_fraction

    else:
        raise ValueError(
            f"Unknown experiment: {experiment}"
        )

    if not np.isfinite(
        metadata["n_clean_points"]
    ):
        metadata[
            "n_clean_points"
        ] = len(p)

    metadata[
        "n_final_points"
    ] = len(p)

    return (
        p,
        voxel,
        metadata,
    )

def parse_float_list(text):
    if isinstance(
        text,
        (list, tuple),
    ):
        return [
            float(x)
            for x in text
        ]

    return [
        float(x.strip())
        for x in str(text).split(",")
        if x.strip()
    ]



def aggregate_benchmark_rows(
    run_rows,
):
    """
    Aggregate across seeds for each:

        experiment
        x perturbation level
        x match tolerance
        x detector method
    """
    groups = {}

    for row in run_rows:
        key = (
            row["experiment"],
            row["level"],
            row["match_radius_mm"],
            row["method"],
        )

        groups.setdefault(
            key,
            [],
        ).append(row)

    metric_names = [
        "n_observed_points",

        "raw_recall",
        "raw_precision",
        "raw_median_mm",
        "raw_rmse_mm",
        "raw_p95_mm",

        "refined_recall",
        "refined_precision",
        "refined_median_mm",
        "refined_rmse_mm",
        "refined_p95_mm",

        "paired_mean_gain_mm",
        "paired_improved_fraction",
    ]

    out = []

    for (
        experiment,
        level,
        match_radius,
        method,
    ), rows in sorted(
        groups.items(),
        key=lambda x: (
            x[0][0],
            float(x[0][1]),
            float(x[0][2]),
            x[0][3],
        ),
    ):
        summary = {
            "experiment": experiment,
            "level": level,
            "match_radius_mm": match_radius,
            "method": method,
            "n_seeds": len(rows),
        }

        for metric in metric_names:
            values = np.asarray(
                [
                    r.get(
                        metric,
                        np.nan,
                    )
                    for r in rows
                ],
                dtype=np.float64,
            )

            valid = values[
                np.isfinite(values)
            ]

            if len(valid):
                summary[
                    metric + "_mean"
                ] = float(
                    np.mean(valid)
                )

                summary[
                    metric + "_std"
                ] = float(
                    np.std(
                        valid,
                        ddof=1,
                    )
                    if len(valid) > 1
                    else 0.0
                )

            else:
                summary[
                    metric + "_mean"
                ] = np.nan

                summary[
                    metric + "_std"
                ] = np.nan

        out.append(
            summary
        )

    return out


def plot_benchmark_summary(
    aggregate_rows,
    out_dir,
):
    """
    Save robustness curves separately for each matching tolerance.
    """
    try:
        import matplotlib.pyplot as plt

    except Exception as exc:
        print(
            f"[plot skipped] matplotlib unavailable: {exc}"
        )
        return

    experiments = sorted(
        set(
            row["experiment"]
            for row in aggregate_rows
        )
    )

    methods = sorted(
        set(
            row["method"]
            for row in aggregate_rows
        )
    )

    tolerances = sorted(
        set(
            float(
                row["match_radius_mm"]
            )
            for row in aggregate_rows
        )
    )

    for tolerance in tolerances:
        tol_tag = (
            f"{tolerance:g}"
            .replace(".", "p")
        )

        for experiment in experiments:
            rows_exp = [
                r
                for r in aggregate_rows
                if (
                    r["experiment"]
                    == experiment
                    and np.isclose(
                        float(
                            r[
                                "match_radius_mm"
                            ]
                        ),
                        tolerance,
                    )
                )
            ]

            if not rows_exp:
                continue

            # ---------------- Recall ----------------
            plt.figure(
                figsize=(8, 5)
            )

            for method in methods:
                rows_m = sorted(
                    [
                        r
                        for r in rows_exp
                        if r["method"]
                        == method
                    ],
                    key=lambda r:
                        float(
                            r["level"]
                        ),
                )

                if not rows_m:
                    continue

                x = [
                    float(
                        r["level"]
                    )
                    for r in rows_m
                ]

                y = [
                    r[
                        "refined_recall_mean"
                    ]
                    for r in rows_m
                ]

                plt.plot(
                    x,
                    y,
                    marker="o",
                    label=method,
                )

            plt.xlabel(
                experiment
            )

            plt.ylabel(
                "Refined recall"
            )

            plt.ylim(
                -0.02,
                1.02,
            )

            plt.title(
                f"{experiment}: refined recall "
                f"(match <= {tolerance:g} mm)"
            )

            plt.grid(
                True,
                alpha=0.3,
            )

            plt.legend(
                fontsize=8
            )

            plt.tight_layout()

            plt.savefig(
                out_dir
                / (
                    f"{experiment}_match_{tol_tag}"
                    "_refined_recall.png"
                ),
                dpi=220,
            )

            plt.close()

            # ---------------- Median localization ----------------
            plt.figure(
                figsize=(8, 5)
            )

            for method in methods:
                rows_m = sorted(
                    [
                        r
                        for r in rows_exp
                        if r["method"]
                        == method
                    ],
                    key=lambda r:
                        float(
                            r["level"]
                        ),
                )

                if not rows_m:
                    continue

                x = [
                    float(
                        r["level"]
                    )
                    for r in rows_m
                ]

                y = [
                    r[
                        "refined_median_mm_mean"
                    ]
                    for r in rows_m
                ]

                plt.plot(
                    x,
                    y,
                    marker="o",
                    label=method,
                )

            plt.xlabel(
                experiment
            )

            plt.ylabel(
                "Refined median error [mm]"
            )

            plt.title(
                f"{experiment}: localization "
                f"(match <= {tolerance:g} mm)"
            )

            plt.grid(
                True,
                alpha=0.3,
            )

            plt.legend(
                fontsize=8
            )

            plt.tight_layout()

            plt.savefig(
                out_dir
                / (
                    f"{experiment}_match_{tol_tag}"
                    "_refined_median_error.png"
                ),
                dpi=220,
            )

            plt.close()


def add_benchmark_arguments(parser):
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help=(
            "Run robustness benchmark instead of "
            "the single visualization experiment."
        ),
    )

    parser.add_argument(
        "--experiments",
        default=(
            "noise,outlier,voxel,sweep"
        ),
        help=(
            "Comma-separated subset of: "
            "noise,outlier,voxel,sweep"
        ),
    )

    parser.add_argument(
        "--benchmark-seeds",
        type=int,
        default=10,
    )

    # --------------------------------------------------------
    # Noise
    # --------------------------------------------------------
    parser.add_argument(
        "--noise-levels",
        default=(
            "0,0.02,0.05,0.10,0.20"
        ),
        help=(
            "Gaussian noise sigma [mm]."
        ),
    )

    parser.add_argument(
        "--noise-mode",
        choices=[
            "isotropic",
            "surface_normal",
        ],
        default="isotropic",
    )

    parser.add_argument(
        "--noise-stage",
        choices=[
            "pre_voxel",
            "post_voxel",
        ],
        default="post_voxel",
        help=(
            "pre_voxel: simulate sensor samples then voxel filtering; "
            "post_voxel: inject the requested sigma directly into the "
            "final point cloud. Recommended for algorithm robustness."
        ),
    )

    # --------------------------------------------------------
    # Outlier
    # --------------------------------------------------------
    parser.add_argument(
        "--outlier-levels",
        default=(
            "0,0.01,0.03,0.05,0.10"
        ),
        help=(
            "FINAL-cloud outlier fraction "
            "Nout/(Nin+Nout). Outliers are inserted after voxelization."
        ),
    )

    parser.add_argument(
        "--outlier-magnitude",
        type=float,
        default=2.0,
        help=(
            "Maximum near-surface outlier "
            "displacement [mm]."
        ),
    )

    # --------------------------------------------------------
    # Voxel
    # --------------------------------------------------------
    parser.add_argument(
        "--voxel-levels",
        default=(
            "0.5,0.75,1.0,1.25,1.5"
        ),
        help=(
            "Voxel sizes [mm]."
        ),
    )

    parser.add_argument(
        "--benchmark-base-voxel",
        type=float,
        default=1.0,
        help=(
            "Voxel used in noise/outlier/sweep "
            "experiments."
        ),
    )

    # --------------------------------------------------------
    # Laser-profile-like sweep
    # --------------------------------------------------------
    parser.add_argument(
        "--sweep-levels",
        default=(
            "0.5,0.75,1.0,1.5,2.0"
        ),
        help=(
            "Laser sweep profile-plane spacing [mm]."
        ),
    )

    parser.add_argument(
        "--sweep-axis",
        choices=[
            "x",
            "y",
            "z",
        ],
        default="x",
    )

    parser.add_argument(
        "--profile-band",
        type=float,
        default=0.20,
        help=(
            "Thickness of each simulated laser profile "
            "band [mm]."
        ),
    )

    parser.add_argument(
        "--sweep-noise",
        type=float,
        default=0.0,
        help=(
            "Noise sigma applied to the final voxelized "
            "profile-like point cloud [mm]."
        ),
    )

    parser.add_argument(
        "--sweep-outlier-ratio",
        type=float,
        default=0.0,
        help=(
            "Final-cloud outlier fraction for sweep experiment."
        ),
    )

    # --------------------------------------------------------
    # Refinement
    # --------------------------------------------------------
    parser.add_argument(
        "--refine-max-iter",
        type=int,
        default=10,
        help=(
            "Maximum iterations for PCL-style "
            "corner refinement. Default: 10."
        ),
    )

    # --------------------------------------------------------
    # Evaluation tolerance sweep
    # --------------------------------------------------------
    parser.add_argument(
        "--match-radii",
        default=(
            "0.25,0.5,1.0,2.0"
        ),
        help=(
            "Comma-separated GT matching tolerances [mm]. "
            "All values are evaluated from the same detections."
        ),
    )

    parser.add_argument(
        "--benchmark-output-dir",
        default=(
            "harris3d_robustness_benchmark_v2"
        ),
    )

    parser.add_argument(
        "--benchmark-plot",
        action="store_true",
    )

    return parser

def benchmark_parse_args():
    """
    Re-create the original CLI plus benchmark-specific options.

    We call the existing parse_args() only for its definition style,
    so this function mirrors the required arguments explicitly.
    """
    p = argparse.ArgumentParser(
        description=(
            "Robustness benchmark for "
            "Proposed normal + Gaussian Harris3D + refinement."
        )
    )

    # Core CAD / sampling
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
        "--seed",
        type=int,
        default=0,
    )

    # Normal
    p.add_argument(
        "--normal-radius",
        type=float,
        default=NORMAL_RADIUS,
    )

    p.add_argument(
        "--proposed",
        choices=[
            "oneshot",
            "iterative",
        ],
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
        default=PROPOSED_MAX_ITER,
    )

    p.add_argument(
        "--convergence-deg",
        type=float,
        default=(
            PROPOSED_CONVERGENCE_DEG
        ),
    )

    p.add_argument(
        "--min-neighbors",
        type=int,
        default=MIN_NEIGHBORS,
    )

    # Harris
    p.add_argument(
        "--harris-radius",
        type=float,
        default=HARRIS_RADIUS,
    )

    p.add_argument(
        "--gaussian-sigma",
        type=float,
        default=0.0,
    )

    p.add_argument(
        "--harris-threshold",
        type=float,
        default=HARRIS_THRESHOLD,
    )

    p.add_argument(
        "--match-radius",
        type=float,
        default=MATCH_RADIUS,
    )

    # GT
    p.add_argument(
        "--gt-mode",
        choices=[
            "sharp_vertices",
            "mesh_vertices",
        ],
        default="mesh_vertices",
    )

    p.add_argument(
        "--gt-corners",
        default=None,
    )

    p.add_argument(
        "--face-normal-cluster-angle",
        type=float,
        default=12.0,
    )

    p.add_argument(
        "--min-face-normal-clusters",
        type=int,
        default=3,
    )

    add_benchmark_arguments(
        p
    )

    return p.parse_args()


def benchmark_main():
    args = benchmark_parse_args()

    if not args.benchmark:
        print(
            "This file is the robustness benchmark version.\n"
            "Add --benchmark to run it."
        )
        return

    if args.refine_max_iter < 1:
        raise ValueError(
            "--refine-max-iter must be >= 1"
        )

    match_radii = sorted(
        set(
            parse_float_list(
                args.match_radii
            )
        )
    )

    if (
        not match_radii
        or any(
            r <= 0
            for r in match_radii
        )
    ):
        raise ValueError(
            "--match-radii must contain positive values"
        )

    out_dir = Path(
        args.benchmark_output_dir
    )

    out_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    mesh = load_mesh(
        args.stl
    )

    gt, gt_source = (
        determine_gt_corners(
            mesh,
            args,
        )
    )

    print("\n" + "=" * 90)
    print(" ROBUSTNESS BENCHMARK")
    print("=" * 90)

    print(
        f"STL             : {args.stl}"
    )

    print(
        f"GT source       : {gt_source}"
    )

    print(
        f"GT corners      : {len(gt)}"
    )

    print(
        f"Seeds           : {args.benchmark_seeds}"
    )

    gaussian_sigma = (
        float(args.gaussian_sigma)
        if args.gaussian_sigma > 0
        else 0.5
        * float(
            args.harris_radius
        )
    )

    print(
        f"Gaussian sigma  : {gaussian_sigma:.4f} mm"
    )

    print(
        f"Refine max iter : {args.refine_max_iter}"
    )

    print(
        f"Noise stage     : {args.noise_stage}"
    )

    print(
        "Match radii     : "
        + ", ".join(
            f"{r:g} mm"
            for r in match_radii
        )
    )

    experiments = [
        x.strip()
        for x in args.experiments.split(",")
        if x.strip()
    ]

    allowed = {
        "noise",
        "outlier",
        "voxel",
        "sweep",
    }

    invalid = [
        x
        for x in experiments
        if x not in allowed
    ]

    if invalid:
        raise ValueError(
            f"Unknown experiments: {invalid}"
        )

    level_map = {
        "noise": parse_float_list(
            args.noise_levels
        ),

        "outlier": parse_float_list(
            args.outlier_levels
        ),

        "voxel": parse_float_list(
            args.voxel_levels
        ),

        "sweep": parse_float_list(
            args.sweep_levels
        ),
    }

    all_rows = []

    # --------------------------------------------------------
    # For each seed, generate a new dense CAD sampling.
    # This tests point-sampling repeatability as well.
    # --------------------------------------------------------
    for seed_index in range(
        args.benchmark_seeds
    ):
        seed = (
            int(args.seed)
            + seed_index
        )

        dense_points = (
            sample_dense_surface(
                mesh,
                args.n_dense,
                seed,
            )
        )

        for experiment in experiments:
            for level in level_map[
                experiment
            ]:
                rng_seed = (
                    seed * 1000003
                    + int(
                        round(
                            float(level)
                            * 10000
                        )
                    )
                    + (
                        {
                            "noise": 11,
                            "outlier": 23,
                            "voxel": 37,
                            "sweep": 53,
                        }[
                            experiment
                        ]
                    )
                )

                rng = np.random.default_rng(
                    rng_seed
                )

                (
                    observed,
                    used_voxel,
                    observation_meta,
                ) = (
                    make_observation_for_benchmark(
                        mesh=mesh,
                        dense_points=(
                            dense_points
                        ),
                        experiment=experiment,
                        level=level,
                        args=args,
                        rng=rng,
                    )
                )

                extra = ""

                if experiment == "outlier":
                    extra = (
                        f" out={observation_meta['n_outliers_added']} "
                        f"actual={100.0*observation_meta['actual_outlier_fraction']:.2f}%"
                    )

                elif experiment == "noise":
                    extra = (
                        f" stage={observation_meta['noise_stage']}"
                    )

                print(
                    f"\n"
                    f"[seed {seed:02d}] "
                    f"{experiment:7s} "
                    f"level={level:g} "
                    f"N={len(observed)} "
                    f"voxel={used_voxel:g}"
                    f"{extra}"
                )

                method_rows = (
                    run_all_harris_methods_for_benchmark(
                        points=observed,
                        gt=gt,
                        args=args,
                        gaussian_sigma=(
                            gaussian_sigma
                        ),
                        match_radii=(
                            match_radii
                        ),
                    )
                )

                if not method_rows:
                    print(
                        "  -> skipped: too few usable points"
                    )

                for row in method_rows:
                    row[
                        "experiment"
                    ] = experiment

                    row[
                        "level"
                    ] = float(level)

                    row[
                        "seed"
                    ] = seed

                    row[
                        "used_voxel_mm"
                    ] = used_voxel

                    row[
                        "gaussian_sigma_mm"
                    ] = gaussian_sigma

                    row[
                        "normal_radius_mm"
                    ] = (
                        args.normal_radius
                    )

                    row[
                        "harris_radius_mm"
                    ] = (
                        args.harris_radius
                    )

                    row[
                        "refine_max_iter"
                    ] = (
                        args.refine_max_iter
                    )

                    row[
                        "noise_stage"
                    ] = (
                        observation_meta[
                            "noise_stage"
                        ]
                    )

                    row[
                        "n_clean_points"
                    ] = (
                        observation_meta[
                            "n_clean_points"
                        ]
                    )

                    row[
                        "n_outliers_added"
                    ] = (
                        observation_meta[
                            "n_outliers_added"
                        ]
                    )

                    row[
                        "actual_outlier_fraction"
                    ] = (
                        observation_meta[
                            "actual_outlier_fraction"
                        ]
                    )

                    row[
                        "sweep_phase_mm"
                    ] = (
                        observation_meta[
                            "sweep_phase_mm"
                        ]
                    )

                    all_rows.append(
                        row
                    )

    # --------------------------------------------------------
    # Save seed-level results
    # --------------------------------------------------------
    save_dict_csv(
        out_dir
        / "all_runs.csv",
        all_rows,
    )

    aggregate = (
        aggregate_benchmark_rows(
            all_rows
        )
    )

    save_dict_csv(
        out_dir
        / "summary_by_condition.csv",
        aggregate,
    )

    # --------------------------------------------------------
    # Print compact refined-performance table
    # --------------------------------------------------------
    print("\n" + "#" * 130)
    print(" ROBUSTNESS SUMMARY (mean across seeds)")
    print("#" * 130)

    table_rows = []

    for row in aggregate:
        table_rows.append([
            row[
                "experiment"
            ],

            f"{float(row['level']):g}",

            f"{float(row['match_radius_mm']):g}",

            row["method"],

            (
                f"{row['refined_recall_mean']:.3f}"
                if np.isfinite(
                    row[
                        "refined_recall_mean"
                    ]
                )
                else "nan"
            ),

            (
                f"{row['refined_precision_mean']:.3f}"
                if np.isfinite(
                    row[
                        "refined_precision_mean"
                    ]
                )
                else "nan"
            ),

            (
                f"{row['refined_median_mm_mean']:.4f}"
                if np.isfinite(
                    row[
                        "refined_median_mm_mean"
                    ]
                )
                else "nan"
            ),

            (
                f"{row['refined_p95_mm_mean']:.4f}"
                if np.isfinite(
                    row[
                        "refined_p95_mm_mean"
                    ]
                )
                else "nan"
            ),

            (
                f"{row['paired_mean_gain_mm_mean']:.4f}"
                if np.isfinite(
                    row[
                        "paired_mean_gain_mm_mean"
                    ]
                )
                else "nan"
            ),
        ])

    print_table(
        table_rows,
        [
            "Experiment",
            "Level",
            "MatchTol",
            "Method",
            "RefRecall",
            "RefPrec",
            "RefMed(mm)",
            "RefP95(mm)",
            "PairGain(mm)",
        ],
    )

    if args.benchmark_plot:
        plot_benchmark_summary(
            aggregate,
            out_dir,
        )

    print(
        f"\nSaved:\n"
        f"  {out_dir / 'all_runs.csv'}\n"
        f"  {out_dir / 'summary_by_condition.csv'}"
    )

    if args.benchmark_plot:
        print(
            f"  {out_dir}/*_match_*_refined_recall.png\n"
            f"  {out_dir}/*_match_*_refined_median_error.png"
        )


if __name__ == "__main__":
    benchmark_main()