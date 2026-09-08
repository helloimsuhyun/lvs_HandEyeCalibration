#!/usr/bin/env python3
"""
# 1. CAD + 선택된 PPF model points
python /home/choisuhyun/lvs_HandEyeCalibration/pose_estimation/PPF_uniqueness.py /home/choisuhyun/lvs_HandEyeCalibration/pose_estimation/미그럼틀.stl \
    --tau-d 0.025 \
    --viz-stage 1

python /home/choisuhyun/lvs_HandEyeCalibration/pose_estimation/PPF_uniqueness.py /home/choisuhyun/lvs_HandEyeCalibration/pose_estimation/미그럼틀.stl \
    --tau-d 0.025 \
    --viz-stage 2 \
    --inspect-ref-index 0

python /home/choisuhyun/lvs_HandEyeCalibration/pose_estimation/PPF_uniqueness.py \
    /home/choisuhyun/lvs_HandEyeCalibration/pose_estimation/미그럼틀.stl \
    --tau-d 0.025 \
    --patch-radius-mm 2.5 \
    --viz-stage 3 \
    --inspect-ref-index 0

python /home/choisuhyun/lvs_HandEyeCalibration/pose_estimation/PPF_uniqueness.py \
    /home/choisuhyun/lvs_HandEyeCalibration/pose_estimation/data/042_adjustable_wrench/google_16k/nontextured.stl \
    --tau-d 0.025 \
    --patch-radius-mm 3.0 \
    --viz-stage 3 \
    --inspect-ref-index 0
    
python /home/choisuhyun/lvs_HandEyeCalibration/pose_estimation/PPF_uniqueness.py /home/choisuhyun/lvs_HandEyeCalibration/pose_estimation/미그럼틀.stl \
    --tau-d 0.025 \
    --viz-stage 4 \
    --inspect-ref-index 0 \
    --inspect-target-index 10

python /home/choisuhyun/lvs_HandEyeCalibration/pose_estimation/PPF_uniqueness.py /home/choisuhyun/lvs_HandEyeCalibration/pose_estimation/미그럼틀.stl \
    --tau-d 0.025 \
    --patch-radius-mm 3.0 \
    --viz-stage 5

python /home/choisuhyun/lvs_HandEyeCalibration/pose_estimation/PPF_uniqueness.py /home/choisuhyun/lvs_HandEyeCalibration/pose_estimation/data/042_adjustable_wrench/google_16k/nontextured.stl \
    --tau-d 0.025 \
    --patch-radius-mm 5.0 \
    --viz-stage 6
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
from scipy.spatial import ConvexHull, cKDTree
from scipy.spatial.distance import cdist

try:
    import open3d as o3d
except ImportError as exc:
    raise SystemExit("open3d is required: pip install open3d") from exc

try:
    import matplotlib
    import matplotlib.pyplot as plt
except ImportError as exc:
    raise SystemExit("matplotlib is required: pip install matplotlib") from exc


EPS = 1e-12


def positive_float(s: str) -> float:
    x = float(s)
    if x <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return x


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Drost-style exact PPF global hash + offline local-patch uniqueness heatmap."
    )
    p.add_argument("mesh", type=Path, help="Input STL/OBJ/PLY mesh.")

    # Units / dense mesh sampling
    p.add_argument(
        "--unit-scale",
        default="auto",
        help=(
            "Scale mesh coordinates to millimeters. "
            "'auto' (default), 1 for mm CAD, 1000 for meter CAD."
        ),
    )
    p.add_argument(
        "--surface-samples",
        type=int,
        default=50000,
        help="Dense surface samples before Drost-style spacing suppression. Default: 50000",
    )

    # Drost paper parameters
    p.add_argument(
        "--tau-d",
        type=positive_float,
        default=0.025,
        help="Drost distance sampling ratio: ddist = tau_d * model diameter. Default: 0.025",
    )
    p.add_argument(
        "--nangle",
        type=int,
        default=30,
        help="Drost angular sampling count; dangle = 2*pi/nangle. Default: 30",
    )

    # The supplied Results section says normals are recalculated by plane fitting
    # but does not specify the neighborhood size.
    p.add_argument(
        "--normal-k",
        type=int,
        default=20,
        help=(
            "k-NN neighborhood for PCA plane-fit normal recalculation after resampling. "
            "The supplied paper excerpt does not specify k. Default: 20"
        ),
    )

    # Our local scan approximation
    p.add_argument(
        "--patch-radius-mm",
        type=positive_float,
        default=10.0,
        help="Radius of circular CAD patch used as a scan approximation. Default: 10 mm",
    )
    p.add_argument(
        "--min-patch-points",
        type=int,
        default=4,
        help="Skip candidate patches with fewer resampled points. Default: 4",
    )
    p.add_argument(
        "--candidate-stride",
        type=int,
        default=1,
        help=(
            "Evaluate every Nth Drost-resampled model point as a patch center. "
            "1 means every model point. Default: 1"
        ),
    )

    # Visualization / output
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--colormap", default="turbo")
    p.add_argument("--point-size", type=positive_float, default=4.0)
    p.add_argument("--no-show", action="store_true")

    # Step-by-step PPF visualization.
    p.add_argument(
        "--viz-stage",
        type=int,
        choices=[0, 1, 2, 3, 4, 5, 6],
        default=0,
        help=(
            "0: normal full pipeline, "
            "1: CAD + Drost-resampled model/reference points, "
            "2: one reference point + all target points/pairs, "
            "3: one patch center + radius + included points, "
            "4: one selected PPF pair + its quantized hash bin, "
            "5: final single-patch PPF uniqueness heatmap, "
            "6: click one already-scanned location and show next-patch heatmap."
        ),
    )
    p.add_argument(
        "--inspect-ref-index",
        type=int,
        default=0,
        help="Index in the Drost-resampled model cloud used for stages 2-4. Default: 0",
    )
    p.add_argument(
        "--inspect-target-index",
        type=int,
        default=-1,
        help=(
            "Target model-point index for stage 4. "
            "-1 automatically chooses the farthest point from the reference."
        ),
    )
    p.add_argument(
        "--viz-marker-radius-mm",
        type=float,
        default=0.0,
        help=(
            "Sphere radius for highlighted points. "
            "0 chooses a size automatically from ddist."
        ),
    )

    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ppf_uniqueness_drost_output"),
    )
    return p


def resolve_unit_scale(mesh: o3d.geometry.TriangleMesh, arg: str) -> float:
    v = np.asarray(mesh.vertices, dtype=np.float64)
    extent = v.max(axis=0) - v.min(axis=0)
    diag = float(np.linalg.norm(extent))

    a = str(arg).strip().lower()
    if a == "auto":
        # Practical heuristic only; STL itself has no unit metadata.
        scale = 1000.0 if diag < 2.0 else 1.0
        assumed = "meters" if scale == 1000.0 else "millimeters"
        print(
            f"      raw bbox extent={np.round(extent, 6)}, diagonal={diag:.6g} "
            f"-> auto assumes {assumed}, scale={scale:g}"
        )
        return scale

    try:
        scale = float(a)
    except ValueError as exc:
        raise ValueError("--unit-scale must be 'auto' or a positive number.") from exc
    if scale <= 0:
        raise ValueError("--unit-scale must be positive.")

    print(
        f"      raw bbox extent={np.round(extent, 6)}, diagonal={diag:.6g}, "
        f"using explicit scale={scale:g}"
    )
    return scale


def load_dense_surface(
    mesh_path: Path,
    surface_samples: int,
    unit_scale_arg: str,
) -> tuple[o3d.geometry.TriangleMesh, np.ndarray, np.ndarray]:
    if not mesh_path.exists():
        raise FileNotFoundError(mesh_path)

    mesh = o3d.io.read_triangle_mesh(str(mesh_path))
    if mesh.is_empty():
        raise RuntimeError(f"Could not load mesh: {mesh_path}")

    scale = resolve_unit_scale(mesh, unit_scale_arg)
    if scale != 1.0:
        v = np.asarray(mesh.vertices)
        v *= scale

    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()

    # Dense cloud is only an intermediate representation from which the
    # Drost-spaced model cloud is selected.
    pcd = mesh.sample_points_uniformly(
        number_of_points=surface_samples,
        use_triangle_normal=True,
    )
    points = np.asarray(pcd.points, dtype=np.float64)
    normals = np.asarray(pcd.normals, dtype=np.float64)

    if len(points) == 0:
        raise RuntimeError("Mesh sampling produced zero points.")

    nn = np.linalg.norm(normals, axis=1)
    valid = nn > EPS
    points = points[valid]
    normals = normals[valid]
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)

    extent = points.max(axis=0) - points.min(axis=0)
    print(
        f"      working bbox extent={np.round(extent, 3)} mm, "
        f"bbox diagonal={np.linalg.norm(extent):.3f} mm"
    )
    print(f"      dense surface points={len(points):,}")
    return mesh, points, normals


def exact_model_diameter(points: np.ndarray) -> float:
    """
    Euclidean model diameter = max ||p_i - p_j||.

    The farthest pair lies on the convex hull. We therefore compute the hull
    first and then scan hull-point distances in memory-safe blocks.
    """
    if len(points) < 2:
        raise RuntimeError("Need at least two points for model diameter.")

    try:
        hull = ConvexHull(points)
        hp = points[np.unique(hull.simplices)]
    except Exception:
        hp = points

    print(f"      convex-hull points for diameter={len(hp):,}")

    max_d2 = 0.0
    block = 512
    for start in range(0, len(hp), block):
        a = hp[start : start + block]
        d = cdist(a, hp, metric="sqeuclidean")
        local = float(np.max(d))
        if local > max_d2:
            max_d2 = local

    return math.sqrt(max_d2)


def minimum_distance_subsample(
    points: np.ndarray,
    source_normals: np.ndarray,
    min_distance: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Greedy radius suppression.

    Every retained pair is guaranteed to be >= min_distance apart.
    This implements the Results-section requirement:
      "subsampled such that all points have a minimum distance of ddist"

    Selection order is randomized only to avoid a systematic mesh-order bias.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(points))
    tree = cKDTree(points)

    blocked = np.zeros(len(points), dtype=bool)
    selected = []

    for idx in order:
        if blocked[idx]:
            continue
        selected.append(idx)
        neighbors = tree.query_ball_point(points[idx], r=min_distance)
        blocked[np.asarray(neighbors, dtype=np.int64)] = True
        # The selected point itself remains selected; blocked only affects future choices.

    selected = np.asarray(selected, dtype=np.int64)
    return points[selected], source_normals[selected]


def recalculate_normals_plane_fit(
    points: np.ndarray,
    orientation_reference_normals: np.ndarray,
    k: int,
) -> np.ndarray:
    """
    Recalculate normals on the RESAMPLED cloud by fitting a local PCA plane.

    Sign is oriented using the mesh-derived normal at the retained point.
    """
    n = len(points)
    if n < 3:
        raise RuntimeError("Too few model points to estimate normals.")

    k_eff = min(max(3, k), n)
    tree = cKDTree(points)
    _, nbr = tree.query(points, k=k_eff)
    if k_eff == 1:
        nbr = nbr[:, None]

    out = np.empty((n, 3), dtype=np.float64)

    for i in range(n):
        q = points[np.atleast_1d(nbr[i])]
        centroid = q.mean(axis=0)
        x = q - centroid
        cov = x.T @ x / max(len(q), 1)
        _, vec = np.linalg.eigh(cov)
        normal = vec[:, 0]
        normal /= max(np.linalg.norm(normal), EPS)

        # Plane-fit normal has arbitrary sign. Keep it consistent with the mesh.
        if np.dot(normal, orientation_reference_normals[i]) < 0:
            normal = -normal
        out[i] = normal

    return out


def encode_ppf_bins(
    distance_bin: np.ndarray,
    a1_bin: np.ndarray,
    a2_bin: np.ndarray,
    a3_bin: np.ndarray,
) -> np.ndarray:
    """
    Pack four nonnegative integer bins into uint64.
    16 bits per component is ample for the Drost parameter regime.
    """
    b0 = distance_bin.astype(np.uint64)
    b1 = a1_bin.astype(np.uint64)
    b2 = a2_bin.astype(np.uint64)
    b3 = a3_bin.astype(np.uint64)

    if (
        np.any(b0 >= 65536)
        or np.any(b1 >= 65536)
        or np.any(b2 >= 65536)
        or np.any(b3 >= 65536)
    ):
        raise RuntimeError("PPF bin index exceeds uint16 packing range.")

    return b0 | (b1 << 16) | (b2 << 32) | (b3 << 48)


def ppf_keys_one_reference(
    ref_idx: int,
    target_indices: np.ndarray,
    points: np.ndarray,
    normals: np.ndarray,
    ddist: float,
    dangle: float,
) -> np.ndarray:
    """
    Classic asymmetric Drost PPF:
      F = (||d||, angle(n1,d), angle(n2,d), angle(n1,n2))

    Quantization:
      distance step = ddist
      angular step  = dangle = 2*pi/nangle

    We use floor(value/step) to map the sampled feature to a discrete hash bin.
    """
    if len(target_indices) == 0:
        return np.empty(0, dtype=np.uint64)

    p1 = points[ref_idx]
    n1 = normals[ref_idx]
    p2 = points[target_indices]
    n2 = normals[target_indices]

    dvec = p2 - p1
    dist = np.linalg.norm(dvec, axis=1)

    valid = dist > EPS
    if not np.any(valid):
        return np.empty(0, dtype=np.uint64)

    dvec = dvec[valid]
    dist = dist[valid]
    n2 = n2[valid]
    dhat = dvec / dist[:, None]

    dot1 = np.clip(dhat @ n1, -1.0, 1.0)
    dot2 = np.clip(np.einsum("ij,ij->i", n2, dhat), -1.0, 1.0)
    dot3 = np.clip(n2 @ n1, -1.0, 1.0)

    a1 = np.arccos(dot1)
    a2 = np.arccos(dot2)
    a3 = np.arccos(dot3)

    b0 = np.floor(dist / ddist).astype(np.int64)
    b1 = np.floor(a1 / dangle).astype(np.int64)
    b2 = np.floor(a2 / dangle).astype(np.int64)
    b3 = np.floor(a3 / dangle).astype(np.int64)

    return encode_ppf_bins(b0, b1, b2, b3)


def build_exact_global_hash(
    points: np.ndarray,
    normals: np.ndarray,
    ddist: float,
    dangle: float,
) -> tuple[dict[int, int], int]:
    """
    Exact global PPF occupancy over ALL ordered model pairs i != j.

    The original recognition implementation stores each matching model pair
    (and precomputed alpha_m) in the hash bucket. For the present uniqueness
    heatmap we only need the bucket occupancy, so we store exact counts.
    """
    n = len(points)
    occupancy: dict[int, int] = {}
    total_pairs = n * (n - 1)

    all_idx = np.arange(n, dtype=np.int64)

    print(
        f"[3/6] Building exact global PPF hash from ALL ordered pairs: "
        f"{n:,} x {n-1:,} = {total_pairs:,}"
    )

    for i in range(n):
        targets = all_idx[all_idx != i]
        keys = ppf_keys_one_reference(
            i, targets, points, normals, ddist, dangle
        )

        unique, counts = np.unique(keys, return_counts=True)
        for key, count in zip(unique, counts):
            kk = int(key)
            occupancy[kk] = occupancy.get(kk, 0) + int(count)

        if (i + 1) % 100 == 0 or i + 1 == n:
            print(f"\r      references {i+1:,}/{n:,}", end="", flush=True)

    print()
    print(f"      occupied global PPF bins={len(occupancy):,}")
    return occupancy, total_pairs


def score_local_patch_exact(
    patch_indices: np.ndarray,
    points: np.ndarray,
    normals: np.ndarray,
    global_occupancy: dict[int, int],
    global_pair_count: int,
    ddist: float,
    dangle: float,
) -> tuple[float, int, float, float]:
    """
    All ordered pairs in the patch are used.

    Returns:
      score
      local_pair_count
      mean_global_occupancy
      median_global_occupancy
    """
    m = len(patch_indices)
    if m < 2:
        return float("nan"), 0, float("nan"), float("nan")

    occ_values = []
    local_pair_count = 0

    for local_ref_pos, ref_idx in enumerate(patch_indices):
        targets = patch_indices[patch_indices != ref_idx]
        keys = ppf_keys_one_reference(
            int(ref_idx), targets, points, normals, ddist, dangle
        )
        if len(keys) == 0:
            continue

        # Because the patch is a subset of the exact global model cloud,
        # each local key MUST already exist in the exact global table.
        occ = np.fromiter(
            (global_occupancy[int(k)] for k in keys),
            dtype=np.float64,
            count=len(keys),
        )
        occ_values.append(occ)
        local_pair_count += len(keys)

    if not occ_values:
        return float("nan"), 0, float("nan"), float("nan")

    occ = np.concatenate(occ_values)

    # IDF-like discrimination score on the exact PPF model hash.
    rarity = np.log(global_pair_count / occ)
    score = float(np.mean(rarity))

    return (
        score,
        int(local_pair_count),
        float(np.mean(occ)),
        float(np.median(occ)),
    )


def robust_normalize(x: np.ndarray) -> np.ndarray:
    """
    Legacy robust min-max normalization.
    Kept for compatibility with non-visual outputs.
    """
    out = np.zeros_like(x, dtype=np.float64)
    finite = np.isfinite(x)
    if not np.any(finite):
        return out

    v = x[finite]
    lo = float(np.percentile(v, 5))
    hi = float(np.percentile(v, 95))

    if hi <= lo + EPS:
        lo, hi = float(v.min()), float(v.max())

    if hi <= lo + EPS:
        out[finite] = 0.5
        return out

    out[finite] = np.clip((v - lo) / (hi - lo), 0.0, 1.0)
    return out


def percentile_rank_normalize(x: np.ndarray) -> np.ndarray:
    """
    Convert finite values to empirical percentile ranks in [0, 1].

    Important for the heatmap:
      - only the best few candidates become red,
      - median candidate is around 0.5,
      - bottom candidates stay in the cool half of the colormap.

    Ties receive the same average percentile rank.
    """
    out = np.full_like(x, np.nan, dtype=np.float64)
    finite = np.isfinite(x)

    if not np.any(finite):
        return out

    values = x[finite]
    n = len(values)

    if n == 1:
        out[finite] = 1.0
        return out

    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(n, dtype=np.float64)

    start = 0
    while start < n:
        end = start + 1
        while end < n and sorted_values[end] == sorted_values[start]:
            end += 1

        # Average 0-based rank for ties.
        avg_rank = 0.5 * (start + (end - 1))
        ranks[order[start:end]] = avg_rank
        start = end

    ranks /= (n - 1)
    out[finite] = ranks
    return out


def map_candidate_scores_to_dense(
    dense_points: np.ndarray,
    candidate_points: np.ndarray,
    candidate_scores: np.ndarray,
) -> np.ndarray:
    finite = np.isfinite(candidate_scores)
    cp = candidate_points[finite]
    cs = candidate_scores[finite]
    if len(cp) == 0:
        raise RuntimeError("No valid candidate scores.")

    tree = cKDTree(cp)
    _, idx = tree.query(dense_points, k=1)
    return cs[idx]


def save_csv(
    path: Path,
    candidate_indices: np.ndarray,
    candidate_points: np.ndarray,
    scores: np.ndarray,
    patch_sizes: np.ndarray,
    pair_counts: np.ndarray,
    mean_occ: np.ndarray,
    median_occ: np.ndarray,
) -> None:
    order = np.argsort(np.nan_to_num(scores, nan=-np.inf))[::-1]

    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "rank",
            "model_point_index",
            "x_mm", "y_mm", "z_mm",
            "score",
            "patch_points",
            "local_ordered_pairs",
            "mean_global_bin_occupancy",
            "median_global_bin_occupancy",
        ])

        rank = 0
        for row_idx in order:
            if not np.isfinite(scores[row_idx]):
                continue
            rank += 1
            p = candidate_points[row_idx]
            w.writerow([
                rank,
                int(candidate_indices[row_idx]),
                f"{p[0]:.6f}",
                f"{p[1]:.6f}",
                f"{p[2]:.6f}",
                f"{scores[row_idx]:.9f}",
                int(patch_sizes[row_idx]),
                int(pair_counts[row_idx]),
                f"{mean_occ[row_idx]:.6f}",
                f"{median_occ[row_idx]:.6f}",
            ])



def _paint_mesh_copy(mesh: o3d.geometry.TriangleMesh, color=(0.65, 0.65, 0.65)):
    out = o3d.geometry.TriangleMesh(mesh)
    out.compute_vertex_normals()
    out.paint_uniform_color(list(color))
    return out


def _make_point_cloud(points: np.ndarray, color) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    if len(points):
        colors = np.tile(np.asarray(color, dtype=np.float64), (len(points), 1))
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def _make_sphere(center: np.ndarray, radius: float, color):
    sphere = o3d.geometry.TriangleMesh.create_sphere(
        radius=float(radius),
        resolution=12,
    )
    sphere.compute_vertex_normals()
    sphere.translate(np.asarray(center, dtype=np.float64))
    sphere.paint_uniform_color(list(color))
    return sphere


def _make_lines_from_reference(
    ref_point: np.ndarray,
    target_points: np.ndarray,
    color=(0.25, 0.8, 0.25),
) -> o3d.geometry.LineSet:
    if len(target_points) == 0:
        return o3d.geometry.LineSet()

    vertices = np.vstack([ref_point[None, :], target_points])
    lines = np.column_stack([
        np.zeros(len(target_points), dtype=np.int32),
        np.arange(1, len(target_points) + 1, dtype=np.int32),
    ])

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(vertices)
    ls.lines = o3d.utility.Vector2iVector(lines)
    ls.colors = o3d.utility.Vector3dVector(
        np.tile(np.asarray(color, dtype=np.float64), (len(lines), 1))
    )
    return ls


def _make_pair_line(
    a: np.ndarray,
    b: np.ndarray,
    color=(1.0, 0.75, 0.0),
) -> o3d.geometry.LineSet:
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(np.vstack([a, b]))
    ls.lines = o3d.utility.Vector2iVector(np.array([[0, 1]], dtype=np.int32))
    ls.colors = o3d.utility.Vector3dVector(
        np.asarray([color], dtype=np.float64)
    )
    return ls


def _make_wire_sphere(
    center: np.ndarray,
    radius: float,
    color=(1.0, 0.35, 0.1),
    segments: int = 96,
) -> o3d.geometry.LineSet:
    """Three orthogonal circles showing the Euclidean patch radius."""
    t = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)

    circles = []
    for plane in range(3):
        pts = np.zeros((segments, 3), dtype=np.float64)
        if plane == 0:      # XY
            pts[:, 0] = radius * np.cos(t)
            pts[:, 1] = radius * np.sin(t)
        elif plane == 1:    # XZ
            pts[:, 0] = radius * np.cos(t)
            pts[:, 2] = radius * np.sin(t)
        else:               # YZ
            pts[:, 1] = radius * np.cos(t)
            pts[:, 2] = radius * np.sin(t)
        circles.append(pts + center[None, :])

    vertices = np.vstack(circles)
    lines = []
    for c in range(3):
        offset = c * segments
        for i in range(segments):
            lines.append([offset + i, offset + ((i + 1) % segments)])

    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(vertices)
    ls.lines = o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
    ls.colors = o3d.utility.Vector3dVector(
        np.tile(np.asarray(color, dtype=np.float64), (len(lines), 1))
    )
    return ls


def _show_open3d(
    geometries: list,
    title: str,
    point_size: float,
) -> None:
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=title, width=1400, height=900)
    for g in geometries:
        vis.add_geometry(g)

    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.03, 0.03, 0.03])
    opt.point_size = float(point_size)

    vis.run()
    vis.destroy_window()


def _validated_ref_index(ref_idx: int, n: int) -> int:
    if not 0 <= ref_idx < n:
        raise ValueError(
            f"--inspect-ref-index={ref_idx} is invalid; "
            f"resampled model has indices 0..{n-1}"
        )
    return ref_idx


def _marker_radius(ddist: float, requested: float) -> float:
    if requested > 0:
        return float(requested)
    return max(0.18 * ddist, 0.5)


def visualize_stage1(
    mesh: o3d.geometry.TriangleMesh,
    model_points: np.ndarray,
    ddist: float,
    point_size: float,
    marker_radius: float,
) -> None:
    """
    Stage 1
    Gray = original CAD
    Cyan spheres = points retained by Drost-style subsampling.
    """
    print()
    print("[VIS 1] CAD + selected Drost model/reference points")
    print(f"        selected points = {len(model_points)}")
    print(f"        minimum spacing ddist = {ddist:.4f} mm")
    print("        gray = original CAD")
    print("        cyan = selected points used by the PPF model")

    geometries = [_paint_mesh_copy(mesh)]

    # Point cloud gives the spatial distribution; spheres keep the points visible.
    geometries.append(_make_point_cloud(model_points, (0.0, 0.9, 1.0)))
    for p in model_points:
        geometries.append(
            _make_sphere(p, marker_radius, (0.0, 0.9, 1.0))
        )

    _show_open3d(
        geometries,
        "PPF Stage 1 - CAD + Drost model points",
        max(point_size, 5.0),
    )


def visualize_stage2(
    mesh: o3d.geometry.TriangleMesh,
    model_points: np.ndarray,
    ref_idx: int,
    ddist: float,
    point_size: float,
    marker_radius: float,
) -> None:
    """
    Stage 2
    Pick one model point as reference m_i.
    Pair it with every other model point m_j.
    """
    n = len(model_points)
    ref_idx = _validated_ref_index(ref_idx, n)

    target_idx = np.arange(n, dtype=np.int64)
    target_idx = target_idx[target_idx != ref_idx]

    ref = model_points[ref_idx]
    targets = model_points[target_idx]

    print()
    print("[VIS 2] One reference point + all PPF target points")
    print(f"        reference index = {ref_idx}")
    print(f"        target points   = {len(targets)}")
    print(f"        ordered pairs from this reference = {len(targets)}")
    print("        red   = selected reference point m_i")
    print("        cyan  = all possible target points m_j")
    print("        green lines = ordered pairs (m_i, m_j)")

    geometries = [
        _paint_mesh_copy(mesh, (0.42, 0.42, 0.42)),
        _make_point_cloud(targets, (0.0, 0.75, 1.0)),
        _make_lines_from_reference(ref, targets),
        _make_sphere(ref, marker_radius * 1.6, (1.0, 0.1, 0.1)),
    ]

    _show_open3d(
        geometries,
        f"PPF Stage 2 - reference {ref_idx} paired with all other points",
        max(point_size, 5.0),
    )


def visualize_stage3(
    mesh: o3d.geometry.TriangleMesh,
    model_points: np.ndarray,
    center_idx: int,
    patch_radius_mm: float,
    point_size: float,
    marker_radius: float,
) -> None:
    """
    Stage 3
    Show exactly what query_ball_point(center, r=patch_radius_mm) means.
    """
    n = len(model_points)
    center_idx = _validated_ref_index(center_idx, n)
    center = model_points[center_idx]

    tree = cKDTree(model_points)
    patch_idx = np.asarray(
        tree.query_ball_point(center, r=patch_radius_mm),
        dtype=np.int64,
    )

    mask = np.ones(n, dtype=bool)
    mask[patch_idx] = False

    patch_points = model_points[patch_idx]
    outside_points = model_points[mask]

    print()
    print("[VIS 3] Local CAD patch")
    print(f"        patch center index = {center_idx}")
    print(
        f"        center xyz = "
        f"({center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}) mm"
    )
    print(f"        patch radius = {patch_radius_mm:.3f} mm")
    print(f"        points inside radius = {len(patch_idx)}")
    print(f"        local ordered pairs = {len(patch_idx) * max(len(patch_idx)-1, 0)}")
    print("        red    = patch center")
    print("        orange = Drost model points inside the patch")
    print("        gray   = model points outside the patch")
    print("        orange wire sphere = Euclidean patch-radius boundary")

    geometries = [
        _paint_mesh_copy(mesh, (0.45, 0.45, 0.45)),
        _make_point_cloud(outside_points, (0.35, 0.35, 0.35)),
        _make_point_cloud(patch_points, (1.0, 0.55, 0.0)),
        _make_wire_sphere(center, patch_radius_mm),
        _make_sphere(center, marker_radius * 1.7, (1.0, 0.0, 0.0)),
    ]

    # Make the included patch points clearly visible.
    for p in patch_points:
        if np.linalg.norm(p - center) <= EPS:
            continue
        geometries.append(
            _make_sphere(p, marker_radius * 0.72, (1.0, 0.55, 0.0))
        )

    _show_open3d(
        geometries,
        f"PPF Stage 3 - patch center {center_idx}, radius {patch_radius_mm:.1f} mm",
        max(point_size, 5.0),
    )


def _raw_ppf_and_bins(
    ref_idx: int,
    target_idx: int,
    points: np.ndarray,
    normals: np.ndarray,
    ddist: float,
    dangle: float,
):
    p1 = points[ref_idx]
    p2 = points[target_idx]
    n1 = normals[ref_idx]
    n2 = normals[target_idx]

    dvec = p2 - p1
    dist = float(np.linalg.norm(dvec))
    if dist <= EPS:
        raise ValueError("Reference and target point are identical.")

    dhat = dvec / dist
    a1 = float(np.arccos(np.clip(np.dot(n1, dhat), -1.0, 1.0)))
    a2 = float(np.arccos(np.clip(np.dot(n2, dhat), -1.0, 1.0)))
    a3 = float(np.arccos(np.clip(np.dot(n1, n2), -1.0, 1.0)))

    bins = (
        int(np.floor(dist / ddist)),
        int(np.floor(a1 / dangle)),
        int(np.floor(a2 / dangle)),
        int(np.floor(a3 / dangle)),
    )

    key = int(
        encode_ppf_bins(
            np.asarray([bins[0]], dtype=np.int64),
            np.asarray([bins[1]], dtype=np.int64),
            np.asarray([bins[2]], dtype=np.int64),
            np.asarray([bins[3]], dtype=np.int64),
        )[0]
    )

    return dist, a1, a2, a3, bins, key


def visualize_stage4(
    mesh: o3d.geometry.TriangleMesh,
    model_points: np.ndarray,
    model_normals: np.ndarray,
    global_occ: dict[int, int],
    global_pair_count: int,
    ref_idx: int,
    target_idx: int,
    ddist: float,
    dangle: float,
    point_size: float,
    marker_radius: float,
) -> None:
    """
    Stage 4
    Pick ONE ordered pair, compute its PPF, quantize it, and show the
    corresponding global hash-table bucket and occupancy.
    """
    n = len(model_points)
    ref_idx = _validated_ref_index(ref_idx, n)

    if target_idx < 0:
        dist = np.linalg.norm(model_points - model_points[ref_idx], axis=1)
        dist[ref_idx] = -np.inf
        target_idx = int(np.argmax(dist))

    if not 0 <= target_idx < n:
        raise ValueError(
            f"--inspect-target-index={target_idx} is invalid; valid range is 0..{n-1}"
        )
    if target_idx == ref_idx:
        raise ValueError("--inspect-target-index must differ from --inspect-ref-index")

    p_ref = model_points[ref_idx]
    p_tar = model_points[target_idx]

    dist, a1, a2, a3, bins, key = _raw_ppf_and_bins(
        ref_idx,
        target_idx,
        model_points,
        model_normals,
        ddist,
        dangle,
    )

    occupancy = int(global_occ.get(key, 0))

    print()
    print("[VIS 4] One PPF -> one global hash-table bin")
    print(f"        ordered pair = ({ref_idx}, {target_idx})")
    print(
        "        raw PPF F = "
        f"(distance={dist:.4f} mm, "
        f"a1={math.degrees(a1):.2f} deg, "
        f"a2={math.degrees(a2):.2f} deg, "
        f"a3={math.degrees(a3):.2f} deg)"
    )
    print(f"        quantized bin = {bins}")
    print(f"        encoded hash key = {key}")
    print(f"        this bin contains {occupancy} / {global_pair_count} global ordered pairs")
    print("        red = reference, blue = target, yellow = selected ordered pair")

    geometries = [
        _paint_mesh_copy(mesh, (0.45, 0.45, 0.45)),
        _make_point_cloud(model_points, (0.25, 0.25, 0.25)),
        _make_pair_line(p_ref, p_tar),
        _make_sphere(p_ref, marker_radius * 1.7, (1.0, 0.0, 0.0)),
        _make_sphere(p_tar, marker_radius * 1.7, (0.0, 0.45, 1.0)),
    ]

    _show_open3d(
        geometries,
        f"PPF Stage 4 - pair ({ref_idx}, {target_idx})",
        max(point_size, 5.0),
    )

    # A second compact figure makes the 4-D bin explicit.
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.axis("off")

    rows = [
        ["PPF component", "Raw value", "Quantized bin"],
        ["distance", f"{dist:.4f} mm", str(bins[0])],
        ["angle(n_ref, d)", f"{math.degrees(a1):.2f} deg", str(bins[1])],
        ["angle(n_target, d)", f"{math.degrees(a2):.2f} deg", str(bins[2])],
        ["angle(n_ref, n_target)", f"{math.degrees(a3):.2f} deg", str(bins[3])],
    ]

    table = ax.table(
        cellText=rows[1:],
        colLabels=rows[0],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.7)

    ax.set_title(
        "Drost PPF Hash Bin\n"
        f"bin={bins}    occupancy={occupancy}/{global_pair_count}",
        fontsize=14,
        pad=20,
    )

    plt.tight_layout()
    plt.show()




def visualize_stage5_heatmap(
    dense_points: np.ndarray,
    dense_mesh_normals: np.ndarray,
    candidate_points: np.ndarray,
    candidate_scores: np.ndarray,
    top_k: int,
    colormap_name: str,
    point_size: float,
    marker_radius: float,
) -> o3d.geometry.PointCloud:
    """
    Stage 5
    Final PPF uniqueness heatmap.

    Blue-ish/low colormap values = common PPFs
    Red-ish/high colormap values = globally rare PPFs
    White spheres = top-ranked scan-patch centers
    """
    # Heatmap color is based on candidate percentile rank, NOT raw min-max.
    candidate_percentile = percentile_rank_normalize(candidate_scores)

    dense_norm_score = map_candidate_scores_to_dense(
        dense_points,
        candidate_points,
        candidate_percentile,
    )

    cmap = matplotlib.colormaps[colormap_name]
    rgb = cmap(np.nan_to_num(dense_norm_score, nan=0.0))[:, :3]

    heatmap = o3d.geometry.PointCloud()
    heatmap.points = o3d.utility.Vector3dVector(dense_points)
    heatmap.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64))
    heatmap.normals = o3d.utility.Vector3dVector(dense_mesh_normals)

    finite = np.isfinite(candidate_scores)
    valid_scores = candidate_scores[finite]

    print()
    print("[VIS 5] Final PPF uniqueness heatmap")
    if len(valid_scores):
        print(
            f"        candidate score min/median/max = "
            f"{np.min(valid_scores):.6f} / "
            f"{np.median(valid_scores):.6f} / "
            f"{np.max(valid_scores):.6f}"
        )
    print("        heatmap color = percentile rank among ALL candidate patches")
    print("        ~0.0 = bottom candidate, ~0.5 = median, ~1.0 = top candidate")
    print("        therefore only relative top candidates become red")
    print("        white spheres = top scan-patch centers")

    geometries = [heatmap]

    order = np.argsort(
        np.nan_to_num(candidate_scores, nan=-np.inf)
    )[::-1]
    top = [
        idx for idx in order
        if np.isfinite(candidate_scores[idx])
    ][:max(0, top_k)]

    for rank, idx in enumerate(top, start=1):
        p = candidate_points[idx]
        print(
            f"        TOP {rank:02d}: "
            f"score={candidate_scores[idx]:.6f}  "
            f"xyz=({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}) mm"
        )
        geometries.append(
            _make_sphere(
                p,
                marker_radius * 1.15,
                (1.0, 1.0, 1.0),
            )
        )

    _show_open3d(
        geometries,
        "PPF Stage 5 - Final Uniqueness Heatmap",
        max(point_size, 4.0),
    )

    # Small colorbar-only figure so the heatmap meaning is explicit.
    fig, ax = plt.subplots(figsize=(7, 1.6))
    fig.subplots_adjust(bottom=0.45)
    norm = matplotlib.colors.Normalize(vmin=0.0, vmax=1.0)
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=ax, orientation="horizontal")
    cbar.set_label("Candidate percentile rank of PPF uniqueness")
    cbar.set_ticks([0.0, 0.5, 1.0])
    cbar.set_ticklabels(["Low", "Medium", "High"])
    plt.show()

    return heatmap




def pick_existing_scan_center(
    dense_points: np.ndarray,
    model_points: np.ndarray,
    point_size: float,
) -> tuple[int, np.ndarray]:
    """Click one CAD surface point and map it to the nearest Drost model point."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(dense_points)

    colors = np.tile(
        np.asarray([0.72, 0.72, 0.72], dtype=np.float64),
        (len(dense_points), 1),
    )
    pcd.colors = o3d.utility.Vector3dVector(colors)

    print()
    print("[VIS 6 - PICK]")
    print("        Shift + Left Click : choose existing scan location")
    print("        Q                  : finish selection")

    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(
        window_name="PPF Stage 6 - Pick Existing Scan Location",
        width=1400,
        height=900,
    )
    vis.add_geometry(pcd)
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.03, 0.03, 0.03])
    opt.point_size = max(float(point_size), 3.0)

    vis.run()
    picked = list(vis.get_picked_points())
    vis.destroy_window()

    if not picked:
        raise RuntimeError(
            "No point selected. Shift+LeftClick one CAD point, then press Q."
        )

    dense_idx = int(picked[0])
    clicked_xyz = dense_points[dense_idx]

    tree = cKDTree(model_points)
    _, model_idx = tree.query(clicked_xyz, k=1)
    model_idx = int(model_idx)

    print(
        f"        clicked xyz = "
        f"({clicked_xyz[0]:.3f}, {clicked_xyz[1]:.3f}, {clicked_xyz[2]:.3f}) mm"
    )
    print(
        f"        nearest model index = {model_idx}, xyz = "
        f"({model_points[model_idx,0]:.3f}, "
        f"{model_points[model_idx,1]:.3f}, "
        f"{model_points[model_idx,2]:.3f}) mm"
    )

    return model_idx, clicked_xyz



def score_newly_introduced_ppf_exact(
    existing_indices: np.ndarray,
    new_indices: np.ndarray,
    points: np.ndarray,
    normals: np.ndarray,
    global_occupancy: dict[int, int],
    global_pair_count: int,
    ddist: float,
    dangle: float,
) -> tuple[float, int, float, float]:
    """
    Stage-6 information score.

    Existing scan A is already known, so A->A pairs provide NO new information.

    For a candidate patch, only PPFs newly introduced by new points B are used:
        A -> B   : included
        B -> A   : included
        B -> B   : included (ordered, i != j)
        A -> A   : excluded

    The score is the mean rarity of these newly introduced PPF bins in the
    ORIGINAL global CAD hash.
    """
    existing_indices = np.asarray(existing_indices, dtype=np.int64)
    new_indices = np.asarray(new_indices, dtype=np.int64)

    if len(new_indices) == 0:
        return float("nan"), 0, float("nan"), float("nan")

    occ_values = []
    new_pair_count = 0

    # Existing -> New
    for ref_idx in existing_indices:
        keys = ppf_keys_one_reference(
            int(ref_idx),
            new_indices,
            points,
            normals,
            ddist,
            dangle,
        )
        if len(keys):
            occ = np.fromiter(
                (global_occupancy[int(k)] for k in keys),
                dtype=np.float64,
                count=len(keys),
            )
            occ_values.append(occ)
            new_pair_count += len(keys)

    # New -> Existing and New -> New
    union_targets = np.concatenate([existing_indices, new_indices])

    for ref_idx in new_indices:
        targets = union_targets[union_targets != ref_idx]
        keys = ppf_keys_one_reference(
            int(ref_idx),
            targets,
            points,
            normals,
            ddist,
            dangle,
        )
        if len(keys):
            occ = np.fromiter(
                (global_occupancy[int(k)] for k in keys),
                dtype=np.float64,
                count=len(keys),
            )
            occ_values.append(occ)
            new_pair_count += len(keys)

    if not occ_values:
        return float("nan"), 0, float("nan"), float("nan")

    occ = np.concatenate(occ_values)

    rarity = np.log(global_pair_count / occ)
    score = float(np.mean(rarity))

    return (
        score,
        int(new_pair_count),
        float(np.mean(occ)),
        float(np.median(occ)),
    )



def score_existing_plus_candidate_patches(
    model_points: np.ndarray,
    model_normals: np.ndarray,
    global_occ: dict[int, int],
    global_pair_count: int,
    ddist: float,
    dangle: float,
    patch_radius_mm: float,
    existing_center_idx: int,
    candidate_indices: np.ndarray,
    min_patch_points: int,
) -> dict:
    """
    Score each candidate using the union:
        existing_patch U candidate_patch

    All ordered PPF pairs in the union are compared against the same
    exact global CAD PPF hash.
    """
    tree = cKDTree(model_points)

    existing_patch_idx = np.asarray(
        tree.query_ball_point(
            model_points[existing_center_idx],
            r=patch_radius_mm,
        ),
        dtype=np.int64,
    )
    existing_set = set(map(int, existing_patch_idx))

    n = len(candidate_indices)
    scores = np.full(n, np.nan, dtype=np.float64)
    added_points = np.zeros(n, dtype=np.int64)
    overlap_points = np.zeros(n, dtype=np.int64)
    union_sizes = np.zeros(n, dtype=np.int64)
    pair_counts = np.zeros(n, dtype=np.int64)
    mean_occ = np.full(n, np.nan, dtype=np.float64)
    median_occ = np.full(n, np.nan, dtype=np.float64)

    print()
    print("[VIS 6 - SCORE]")
    print(f"        existing center index = {existing_center_idx}")
    print(f"        existing patch points = {len(existing_patch_idx)}")
    print(f"        patch radius = {patch_radius_mm:.3f} mm")

    for cpos, candidate_idx in enumerate(candidate_indices):
        candidate_patch_idx = np.asarray(
            tree.query_ball_point(
                model_points[int(candidate_idx)],
                r=patch_radius_mm,
            ),
            dtype=np.int64,
        )
        candidate_set = set(map(int, candidate_patch_idx))

        new_points = candidate_set - existing_set
        added_points[cpos] = len(new_points)
        overlap_points[cpos] = len(candidate_set & existing_set)

        # If nothing new is added, it is not a meaningful next scan.
        if len(new_points) == 0:
            continue

        new_idx = np.asarray(sorted(new_points), dtype=np.int64)

        union_idx = np.union1d(
            existing_patch_idx,
            new_idx,
        ).astype(np.int64)

        union_sizes[cpos] = len(union_idx)

        if len(union_idx) < min_patch_points:
            continue

        # IMPORTANT:
        # Do NOT re-score existing->existing PPFs.
        # Score only PPFs introduced by the newly added points.
        score, n_pairs, m_occ, md_occ = score_newly_introduced_ppf_exact(
            existing_patch_idx,
            new_idx,
            model_points,
            model_normals,
            global_occ,
            global_pair_count,
            ddist,
            dangle,
        )

        scores[cpos] = score
        pair_counts[cpos] = n_pairs
        mean_occ[cpos] = m_occ
        median_occ[cpos] = md_occ

        if (cpos + 1) % 100 == 0 or cpos + 1 == n:
            print(
                f"\r        candidates {cpos+1:,}/{n:,}",
                end="",
                flush=True,
            )

    print()

    return {
        "existing_patch_idx": existing_patch_idx,
        "scores": scores,
        "added_points": added_points,
        "overlap_points": overlap_points,
        "union_sizes": union_sizes,
        "pair_counts": pair_counts,
        "mean_occ": mean_occ,
        "median_occ": median_occ,
    }


def visualize_stage6_next_scan_heatmap(
    dense_points: np.ndarray,
    dense_mesh_normals: np.ndarray,
    model_points: np.ndarray,
    candidate_points: np.ndarray,
    result: dict,
    existing_center_idx: int,
    patch_radius_mm: float,
    top_k: int,
    colormap_name: str,
    point_size: float,
    marker_radius: float,
) -> o3d.geometry.PointCloud:
    """
    Heatmap meaning:
        score(x) = uniqueness(existing_patch U candidate_patch(x))
    """
    scores = result["scores"]
    finite = np.isfinite(scores)
    if not np.any(finite):
        raise RuntimeError(
            "No valid next-scan candidate. "
            "Try smaller --patch-radius-mm or smaller --tau-d."
        )

    # Percentile rank over VALID next-scan candidates.
    candidate_percentile = percentile_rank_normalize(scores)

    dense_norm_score = map_candidate_scores_to_dense(
        dense_points,
        candidate_points,
        candidate_percentile,
    )

    cmap = matplotlib.colormaps[colormap_name]
    rgb = cmap(np.nan_to_num(dense_norm_score, nan=0.0))[:, :3]

    heatmap = o3d.geometry.PointCloud()
    heatmap.points = o3d.utility.Vector3dVector(dense_points)
    heatmap.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64))
    heatmap.normals = o3d.utility.Vector3dVector(dense_mesh_normals)

    print()
    print("[VIS 6] Next-scan heatmap")
    print(
        "        score = PPF uniqueness of "
        "(existing scanned patch + candidate patch)"
    )
    print("        white points  = already-scanned patch")
    print("        yellow center = already-scanned center")
    print("        magenta       = top next-scan candidates")

    valid_scores = scores[finite]
    print(
        f"        score min/median/max = "
        f"{np.min(valid_scores):.6f} / "
        f"{np.median(valid_scores):.6f} / "
        f"{np.max(valid_scores):.6f}"
    )

    geometries = [heatmap]

    existing_patch_idx = result["existing_patch_idx"]
    existing_points = model_points[existing_patch_idx]

    geometries.append(
        _make_point_cloud(existing_points, (1.0, 1.0, 1.0))
    )
    geometries.append(
        _make_wire_sphere(
            model_points[existing_center_idx],
            patch_radius_mm,
            color=(1.0, 1.0, 1.0),
        )
    )
    geometries.append(
        _make_sphere(
            model_points[existing_center_idx],
            marker_radius * 1.6,
            (1.0, 0.85, 0.0),
        )
    )

    order = np.argsort(np.nan_to_num(scores, nan=-np.inf))[::-1]
    top = [
        i for i in order
        if np.isfinite(scores[i])
    ][:max(0, top_k)]

    for rank, i in enumerate(top, start=1):
        p = candidate_points[i]
        print(
            f"        TOP {rank:02d}: "
            f"score={scores[i]:.6f}  "
            f"xyz=({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}) mm  "
            f"new_pts={result['added_points'][i]}  "
            f"union_pts={result['union_sizes'][i]}  "
            f"pairs={result['pair_counts'][i]}  "
            f"median_global_occ={result['median_occ'][i]:.1f}"
        )
        geometries.append(
            _make_sphere(
                p,
                marker_radius * 1.15,
                (1.0, 0.0, 1.0),
            )
        )

    _show_open3d(
        geometries,
        "PPF Stage 6 - Next Scan Heatmap",
        max(point_size, 4.0),
    )

    fig, ax = plt.subplots(figsize=(7, 1.6))
    fig.subplots_adjust(bottom=0.45)
    norm = matplotlib.colors.Normalize(vmin=0.0, vmax=1.0)
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=ax, orientation="horizontal")
    cbar.set_label(
        "Normalized uniqueness(existing scan + candidate patch)"
    )
    cbar.set_ticks([0.0, 0.5, 1.0])
    cbar.set_ticklabels(["Low", "Medium", "High"])
    plt.show()

    return heatmap


def save_stage6_outputs(
    output_dir: Path,
    heatmap: o3d.geometry.PointCloud,
    candidate_indices: np.ndarray,
    candidate_points: np.ndarray,
    result: dict,
    existing_center_idx: int,
    existing_center_xyz: np.ndarray,
    patch_radius_mm: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "ppf_stage6_next_scan_candidates.csv"
    npz_path = output_dir / "ppf_stage6_next_scan.npz"
    ply_path = output_dir / "ppf_stage6_next_scan_heatmap.ply"

    scores = result["scores"]
    order = np.argsort(np.nan_to_num(scores, nan=-np.inf))[::-1]

    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "rank",
            "candidate_model_index",
            "x_mm", "y_mm", "z_mm",
            "new_ppf_uniqueness_score",
            "new_points_added",
            "overlap_points",
            "union_points",
            "newly_introduced_ordered_pairs",
            "mean_global_bin_occupancy",
            "median_global_bin_occupancy",
        ])

        rank = 0
        for i in order:
            if not np.isfinite(scores[i]):
                continue
            rank += 1
            p = candidate_points[i]
            w.writerow([
                rank,
                int(candidate_indices[i]),
                f"{p[0]:.6f}",
                f"{p[1]:.6f}",
                f"{p[2]:.6f}",
                f"{scores[i]:.9f}",
                int(result["added_points"][i]),
                int(result["overlap_points"][i]),
                int(result["union_sizes"][i]),
                int(result["pair_counts"][i]),
                f"{result['mean_occ'][i]:.6f}",
                f"{result['median_occ'][i]:.6f}",
            ])

    np.savez_compressed(
        npz_path,
        candidate_indices=candidate_indices,
        candidate_points=candidate_points,
        combined_scores=scores,
        new_points_added=result["added_points"],
        overlap_points=result["overlap_points"],
        union_sizes=result["union_sizes"],
        pair_counts=result["pair_counts"],
        mean_global_occupancy=result["mean_occ"],
        median_global_occupancy=result["median_occ"],
        existing_center_idx=int(existing_center_idx),
        existing_center_xyz=np.asarray(existing_center_xyz),
        existing_patch_idx=result["existing_patch_idx"],
        patch_radius_mm=float(patch_radius_mm),
    )

    o3d.io.write_point_cloud(
        str(ply_path),
        heatmap,
        write_ascii=False,
        compressed=True,
    )

    print()
    print("[VIS 6] Saved outputs")
    print(f"        CSV : {csv_path}")
    print(f"        NPZ : {npz_path}")
    print(f"        PLY : {ply_path}")



def main() -> None:
    args = build_parser().parse_args()

    if args.surface_samples < 100:
        raise ValueError("--surface-samples must be >= 100")
    if args.nangle < 1:
        raise ValueError("--nangle must be >= 1")
    if args.normal_k < 3:
        raise ValueError("--normal-k must be >= 3")
    if args.min_patch_points < 2:
        raise ValueError("--min-patch-points must be >= 2")
    if args.candidate_stride < 1:
        raise ValueError("--candidate-stride must be >= 1")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("[0/6] Loading CAD and sampling dense surface ...")
    mesh, dense_points, dense_mesh_normals = load_dense_surface(
        args.mesh,
        args.surface_samples,
        args.unit_scale,
    )

    print("[1/6] Computing model diameter and Drost sampling parameters ...")
    diameter = exact_model_diameter(dense_points)
    ddist = args.tau_d * diameter
    dangle = 2.0 * math.pi / args.nangle

    print(f"      diam(M) = {diameter:.6f} mm")
    print(f"      tau_d   = {args.tau_d:.6f}")
    print(f"      ddist   = tau_d * diam(M) = {ddist:.6f} mm")
    print(f"      nangle  = {args.nangle}")
    print(f"      dangle  = 2*pi/nangle = {math.degrees(dangle):.6f} deg")

    if args.patch_radius_mm < ddist:
        print(
            f"WARNING: patch radius ({args.patch_radius_mm:.3f} mm) "
            f"is smaller than ddist ({ddist:.3f} mm); "
            "many patches may contain too few PPF model points."
        )

    print("[2/6] Drost-style model resampling and normal recalculation ...")
    model_points, retained_mesh_normals = minimum_distance_subsample(
        dense_points,
        dense_mesh_normals,
        ddist,
        args.seed,
    )
    print(
        f"      resampled model points={len(model_points):,} "
        f"(minimum spacing >= {ddist:.4f} mm)"
    )

    model_normals = recalculate_normals_plane_fit(
        model_points,
        retained_mesh_normals,
        args.normal_k,
    )
    print(
        f"      normals recalculated by PCA plane fit on "
        f"k={min(args.normal_k, len(model_points))} resampled neighbors"
    )

    if len(model_points) < 2:
        raise RuntimeError("Drost resampling left fewer than two model points.")

    marker_radius = _marker_radius(ddist, args.viz_marker_radius_mm)

    # Stop after the requested educational visualization stage.
    if args.viz_stage == 1:
        visualize_stage1(
            mesh,
            model_points,
            ddist,
            args.point_size,
            marker_radius,
        )
        return

    if args.viz_stage == 2:
        visualize_stage2(
            mesh,
            model_points,
            args.inspect_ref_index,
            ddist,
            args.point_size,
            marker_radius,
        )
        return

    if args.viz_stage == 3:
        visualize_stage3(
            mesh,
            model_points,
            args.inspect_ref_index,
            args.patch_radius_mm,
            args.point_size,
            marker_radius,
        )
        return

    global_occ, global_pair_count = build_exact_global_hash(
        model_points,
        model_normals,
        ddist,
        dangle,
    )

    if args.viz_stage == 6:
        existing_center_idx, clicked_xyz = pick_existing_scan_center(
            dense_points,
            model_points,
            args.point_size,
        )

        candidate_indices = np.arange(
            0,
            len(model_points),
            args.candidate_stride,
            dtype=np.int64,
        )
        candidate_points = model_points[candidate_indices]

        result = score_existing_plus_candidate_patches(
            model_points,
            model_normals,
            global_occ,
            global_pair_count,
            ddist,
            dangle,
            args.patch_radius_mm,
            existing_center_idx,
            candidate_indices,
            args.min_patch_points,
        )

        heatmap = visualize_stage6_next_scan_heatmap(
            dense_points,
            dense_mesh_normals,
            model_points,
            candidate_points,
            result,
            existing_center_idx,
            args.patch_radius_mm,
            args.top_k,
            args.colormap,
            args.point_size,
            marker_radius,
        )

        save_stage6_outputs(
            args.output_dir,
            heatmap,
            candidate_indices,
            candidate_points,
            result,
            existing_center_idx,
            model_points[existing_center_idx],
            args.patch_radius_mm,
        )
        return

    if args.viz_stage == 4:
        visualize_stage4(
            mesh,
            model_points,
            model_normals,
            global_occ,
            global_pair_count,
            args.inspect_ref_index,
            args.inspect_target_index,
            ddist,
            dangle,
            args.point_size,
            marker_radius,
        )
        return

    print("[4/6] Evaluating local scan-patch uniqueness ...")
    model_tree = cKDTree(model_points)

    candidate_indices = np.arange(
        0,
        len(model_points),
        args.candidate_stride,
        dtype=np.int64,
    )
    candidate_points = model_points[candidate_indices]

    n_c = len(candidate_indices)
    scores = np.full(n_c, np.nan, dtype=np.float64)
    patch_sizes = np.zeros(n_c, dtype=np.int64)
    pair_counts = np.zeros(n_c, dtype=np.int64)
    mean_occ = np.full(n_c, np.nan, dtype=np.float64)
    median_occ = np.full(n_c, np.nan, dtype=np.float64)

    for cpos, model_idx in enumerate(candidate_indices):
        center = model_points[model_idx]
        patch_idx = np.asarray(
            model_tree.query_ball_point(center, r=args.patch_radius_mm),
            dtype=np.int64,
        )
        patch_sizes[cpos] = len(patch_idx)

        if len(patch_idx) < args.min_patch_points:
            continue

        score, n_pairs, m_occ, md_occ = score_local_patch_exact(
            patch_idx,
            model_points,
            model_normals,
            global_occ,
            global_pair_count,
            ddist,
            dangle,
        )
        scores[cpos] = score
        pair_counts[cpos] = n_pairs
        mean_occ[cpos] = m_occ
        median_occ[cpos] = md_occ

        if (cpos + 1) % 100 == 0 or cpos + 1 == n_c:
            print(f"\r      candidates {cpos+1:,}/{n_c:,}", end="", flush=True)

    print()

    valid_count = int(np.sum(np.isfinite(scores)))
    if valid_count == 0:
        raise RuntimeError(
            "No valid local patches. Increase --patch-radius-mm or reduce "
            "--min-patch-points."
        )

    print(f"      valid candidate patches={valid_count:,}/{n_c:,}")

    if args.viz_stage == 5:
        visualize_stage5_heatmap(
            dense_points,
            dense_mesh_normals,
            candidate_points,
            scores,
            args.top_k,
            args.colormap,
            args.point_size,
            marker_radius,
        )
        return

    print("[5/6] Saving ranking and heatmap ...")
    csv_path = args.output_dir / "ppf_uniqueness_candidates.csv"
    npz_path = args.output_dir / "ppf_uniqueness_drost.npz"
    ply_path = args.output_dir / "ppf_uniqueness_heatmap.ply"

    save_csv(
        csv_path,
        candidate_indices,
        candidate_points,
        scores,
        patch_sizes,
        pair_counts,
        mean_occ,
        median_occ,
    )

    dense_raw_score = map_candidate_scores_to_dense(
        dense_points,
        candidate_points,
        scores,
    )
    dense_norm_score = robust_normalize(dense_raw_score)

    cmap = matplotlib.colormaps[args.colormap]
    rgb = cmap(dense_norm_score)[:, :3]

    heatmap = o3d.geometry.PointCloud()
    heatmap.points = o3d.utility.Vector3dVector(dense_points)
    heatmap.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64))
    heatmap.normals = o3d.utility.Vector3dVector(dense_mesh_normals)

    o3d.io.write_point_cloud(
        str(ply_path),
        heatmap,
        write_ascii=False,
        compressed=True,
    )

    np.savez_compressed(
        npz_path,
        dense_points=dense_points,
        dense_mesh_normals=dense_mesh_normals,
        model_points=model_points,
        model_normals=model_normals,
        candidate_indices=candidate_indices,
        candidate_points=candidate_points,
        candidate_scores=scores,
        candidate_patch_sizes=patch_sizes,
        candidate_pair_counts=pair_counts,
        candidate_mean_global_occupancy=mean_occ,
        candidate_median_global_occupancy=median_occ,
        model_diameter_mm=float(diameter),
        tau_d=float(args.tau_d),
        ddist_mm=float(ddist),
        nangle=int(args.nangle),
        dangle_rad=float(dangle),
        patch_radius_mm=float(args.patch_radius_mm),
        global_pair_count=int(global_pair_count),
        global_occupied_bins=int(len(global_occ)),
    )

    print(f"      CSV : {csv_path}")
    print(f"      NPZ : {npz_path}")
    print(f"      PLY : {ply_path}")

    order = np.argsort(np.nan_to_num(scores, nan=-np.inf))[::-1]
    top = [i for i in order if np.isfinite(scores[i])][: max(0, args.top_k)]

    print()
    print("Top scan-patch candidates:")
    geometries = [heatmap]

    marker_radius = max(0.12 * ddist, 0.25)
    for rank, row_idx in enumerate(top, start=1):
        p = candidate_points[row_idx]
        print(
            f"      TOP {rank:02d}: score={scores[row_idx]:.6f}  "
            f"xyz=({p[0]:.2f}, {p[1]:.2f}, {p[2]:.2f}) mm  "
            f"patch_pts={patch_sizes[row_idx]}  "
            f"pairs={pair_counts[row_idx]}  "
            f"median_global_occ={median_occ[row_idx]:.1f}"
        )

        sphere = o3d.geometry.TriangleMesh.create_sphere(
            radius=marker_radius,
            resolution=8,
        )
        sphere.compute_vertex_normals()
        sphere.translate(p)
        sphere.paint_uniform_color([1.0, 1.0, 1.0])
        geometries.append(sphere)

    if not args.no_show:
        vis = o3d.visualization.Visualizer()
        vis.create_window(
            window_name="Drost PPF Offline Scan Uniqueness Heatmap",
            width=1400,
            height=900,
        )
        for g in geometries:
            vis.add_geometry(g)

        opt = vis.get_render_option()
        opt.point_size = float(args.point_size)
        opt.background_color = np.asarray([0.03, 0.03, 0.03])
        vis.run()
        vis.destroy_window()


if __name__ == "__main__":
    main()