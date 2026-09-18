#!/usr/bin/env python3
"""
Controlled Harris3D experiment: uniform neighborhood vs Gaussian-windowed Harris.

Purpose
-------
The previous fixed-query experiments showed that independent resampling keeps the
broad Harris response region fairly similar, while the local winner/NMS location
changes strongly. This script tests a specific alternative explanation:

    "Is that instability mainly because the Harris moment matrix used a hard,
     uniform radius window instead of a Gaussian spatial window?"

Everything except the Harris spatial window is controlled:
  * same complete CAD mesh
  * two independent support samples A/B
  * exact triangle face normals (no normal-estimation error)
  * no noise / outliers / partial crop
  * one identical fixed query set Q for A and B
  * same Harris support radius and same NMS neighborhoods
  * NO post-response smoothing

For each Gaussian sigma, the local Harris matrix at fixed query q is

    M(q) = sum_i w_i n_i n_i^T / sum_i w_i

with

    w_i = exp(-||p_i-q||^2 / (2 sigma^2)).

sigma=0 means the previous uniform/box window:

    M(q) = (1/N) sum_i n_i n_i^T.

The Harris response remains the same PCL-style expression used in the previous
controlled scripts:

    R(q) = 0.04 + det(M) - 0.04 * trace(M)^2

The script reports, for every window sigma:
  * fixed-Q Pearson / Spearman response correlation A vs B
  * threshold-candidate Jaccard
  * NMS spatial repeatability at requested radii
  * exact same-query NMS retention
  * local Spearman around A NMS peaks
  * top-1/top-2 winner flip rate
  * same-local-winner rate
  * local-winner displacement

Interpretation
--------------
If a Gaussian window is the missing ingredient, one should see a substantial and
consistent improvement in point-level NMS repeatability and local ranking
stability compared with sigma=0, without relying on post-response smoothing.

Example
-------
python3 harris3d_gaussian_window_diagnosis.py \
  --cad /home/choisuhyun/lvs_HandEyeCalibration/pose_estimation/data/datasets/tless/models_cad/models_cad/obj_05.ply \
  --support-sample-points 120000 \
  --seed-a 0 \
  --seed-b 1 \
  --support-voxel-mm 1.0 \
  --query-sample-points 120000 \
  --seed-q 999 \
  --query-voxel-mm 0.5 \
  --harris-radius-mm 3.0 \
  --window-sigmas-mm 0,0.5,1.0,1.5,2.0,3.0 \
  --harris-threshold 1e-6 \
  --local-radius-mm 3.0 \
  --repeat-mm 1.0 \
  --report-mm 0.5,1,2,3,5 \
  --save-summary-csv gaussian_window_sweep.csv \
  --save-peak-csv gaussian_window_peaks.csv \
  --save-npz gaussian_window_fields.npz \
  --show
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Tuple

import numpy as np
import open3d as o3d

try:
    from scipy.spatial import cKDTree
    from scipy.stats import spearmanr
except Exception as exc:  # pragma: no cover
    raise ImportError("This script requires scipy (cKDTree + spearmanr)") from exc


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_positive_float_list(text: str) -> list[float]:
    try:
        vals = [float(x.strip()) for x in text.split(",") if x.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if not vals or any(v <= 0 for v in vals):
        raise argparse.ArgumentTypeError("Expected comma-separated positive values.")
    return sorted(set(vals))


def parse_nonnegative_float_list(text: str) -> list[float]:
    try:
        vals = [float(x.strip()) for x in text.split(",") if x.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if not vals or any(v < 0 for v in vals):
        raise argparse.ArgumentTypeError(
            "Expected comma-separated non-negative values; 0 means uniform window."
        )
    vals = sorted(set(vals))
    if 0.0 not in vals:
        vals = [0.0] + vals
    return vals


def parse_args():
    p = argparse.ArgumentParser(
        description="Fixed-query Harris3D Gaussian spatial-window diagnosis."
    )
    p.add_argument("--cad", type=Path, required=True)

    p.add_argument("--support-sample-points", type=int, default=120000)
    p.add_argument("--seed-a", type=int, default=0)
    p.add_argument("--seed-b", type=int, default=1)
    p.add_argument(
        "--support-voxel-mm",
        type=float,
        default=1.0,
        help="Representative thinning for support clouds A/B; 0 disables.",
    )

    p.add_argument("--query-source", choices=["sample", "vertices"], default="sample")
    p.add_argument("--query-sample-points", type=int, default=120000)
    p.add_argument("--seed-q", type=int, default=999)
    p.add_argument(
        "--query-voxel-mm",
        type=float,
        default=0.5,
        help="Representative thinning for common fixed query set Q; 0 disables.",
    )

    p.add_argument("--harris-radius-mm", type=float, default=3.0)
    p.add_argument(
        "--window-sigmas-mm",
        type=parse_nonnegative_float_list,
        default=[0.0, 0.5, 1.0, 1.5, 2.0, 3.0],
        help="Gaussian sigma sweep. 0 = uniform/box window baseline.",
    )
    p.add_argument("--harris-threshold", type=float, default=1e-6)
    p.add_argument("--min-neighbors", type=int, default=3)
    p.add_argument(
        "--keypoints-k",
        type=int,
        default=0,
        help="Keep strongest K NMS peaks per field. 0 keeps all.",
    )

    p.add_argument(
        "--local-radius-mm",
        type=float,
        default=3.0,
        help="Radius for local ranking/winner-flip diagnostics.",
    )
    p.add_argument("--repeat-mm", type=float, default=1.0)
    p.add_argument(
        "--report-mm",
        type=parse_positive_float_list,
        default=[0.5, 1.0, 2.0, 3.0, 5.0],
    )

    p.add_argument("--save-summary-csv", type=Path, default=None)
    p.add_argument("--save-peak-csv", type=Path, default=None)
    p.add_argument("--save-npz", type=Path, default=None)
    p.add_argument("--show", action="store_true")
    p.add_argument("--max-spheres", type=int, default=700)
    p.add_argument("--max-lines", type=int, default=350)
    return p.parse_args()


# -----------------------------------------------------------------------------
# Mesh / controlled sampling
# -----------------------------------------------------------------------------


def load_mesh(path: Path) -> o3d.geometry.TriangleMesh:
    if not path.exists():
        raise FileNotFoundError(path)
    mesh = o3d.io.read_triangle_mesh(str(path), enable_post_processing=False)
    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        raise ValueError(f"CAD must be a triangle mesh: {path}")

    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    if len(mesh.triangles) == 0:
        raise ValueError(f"CAD mesh became empty after cleanup: {path}")

    print(
        f"[CAD] TRIANGLE MESH: vertices={len(mesh.vertices)}, "
        f"triangles={len(mesh.triangles)}"
    )
    return mesh


def _triangle_geometry(mesh: o3d.geometry.TriangleMesh):
    V = np.asarray(mesh.vertices, dtype=np.float64)
    F = np.asarray(mesh.triangles, dtype=np.int64)
    a = V[F[:, 0]]
    b = V[F[:, 1]]
    c = V[F[:, 2]]
    cross = np.cross(b - a, c - a)
    twice_area = np.linalg.norm(cross, axis=1)
    valid = np.isfinite(twice_area) & (twice_area > 1e-15)
    if not np.any(valid):
        raise ValueError("Mesh has no valid non-degenerate triangles")
    return V, a, b, c, cross, twice_area, valid


def sample_mesh_with_face_normals(
    mesh: o3d.geometry.TriangleMesh,
    n_samples: int,
    seed: int,
    label: str,
) -> Tuple[np.ndarray, np.ndarray]:
    if n_samples < 10:
        raise ValueError("sample count must be >= 10")

    _, a, b, c, cross, twice_area, valid = _triangle_geometry(mesh)
    valid_ids = np.flatnonzero(valid)
    prob = twice_area[valid] / twice_area[valid].sum()

    rng = np.random.default_rng(seed)
    chosen_local = rng.choice(len(valid_ids), size=int(n_samples), replace=True, p=prob)
    tri_ids = valid_ids[chosen_local]

    aa = a[tri_ids]
    bb = b[tri_ids]
    cc = c[tri_ids]

    r1 = rng.random(n_samples)
    r2 = rng.random(n_samples)
    s1 = np.sqrt(r1)
    w0 = 1.0 - s1
    w1 = s1 * (1.0 - r2)
    w2 = s1 * r2
    points = w0[:, None] * aa + w1[:, None] * bb + w2[:, None] * cc

    normals = cross[tri_ids].copy()
    normals /= np.linalg.norm(normals, axis=1, keepdims=True)

    print(f"[{label}] area-uniform samples: {len(points)} (seed={seed})")
    return np.ascontiguousarray(points), np.ascontiguousarray(normals)


def sample_mesh_points_only(
    mesh: o3d.geometry.TriangleMesh,
    n_samples: int,
    seed: int,
    label: str,
) -> np.ndarray:
    p, _ = sample_mesh_with_face_normals(mesh, n_samples, seed, label)
    return p


def voxel_representative(
    points: np.ndarray,
    leaf: float,
    origin: np.ndarray,
    label: str,
    normals: np.ndarray | None = None,
):
    points = np.asarray(points, dtype=np.float64)
    if leaf <= 0:
        if normals is None:
            return np.ascontiguousarray(points)
        return np.ascontiguousarray(points), np.ascontiguousarray(normals)

    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    vox = np.floor((points - origin) / float(leaf)).astype(np.int64)
    _, first = np.unique(vox, axis=0, return_index=True)
    keep = np.sort(first)
    print(
        f"[{label}] representative voxel thinning @ {leaf:.3f} mm: "
        f"{len(points)} -> {len(keep)}"
    )
    if normals is None:
        return np.ascontiguousarray(points[keep])
    return np.ascontiguousarray(points[keep]), np.ascontiguousarray(normals[keep])


# -----------------------------------------------------------------------------
# Fixed-query support cache + Harris response
# -----------------------------------------------------------------------------


def build_support_cache(
    query_points: np.ndarray,
    support_points: np.ndarray,
    support_normals: np.ndarray,
    radius: float,
    label: str,
):
    """Cache radius-neighborhood indices and squared distances for one support cloud."""
    Q = np.asarray(query_points, dtype=np.float64)
    P = np.asarray(support_points, dtype=np.float64)
    N = np.asarray(support_normals, dtype=np.float64)
    if len(P) != len(N):
        raise ValueError(f"{label}: support points/normals size mismatch")

    n_norm = np.linalg.norm(N, axis=1)
    good_n = np.isfinite(N).all(axis=1) & (n_norm > 1e-12)
    Nn = N.copy()
    Nn[good_n] /= n_norm[good_n, None]

    tree = cKDTree(P)
    raw_neigh = tree.query_ball_point(Q, r=float(radius), workers=-1)

    ids_list: list[np.ndarray] = []
    d2_list: list[np.ndarray] = []
    count = np.zeros(len(Q), dtype=np.int32)

    for qi, raw in enumerate(raw_neigh):
        ids = np.asarray(raw, dtype=np.int64)
        if len(ids):
            ids = ids[good_n[ids]]
        if len(ids):
            diff = P[ids] - Q[qi]
            d2 = np.einsum("ij,ij->i", diff, diff)
        else:
            d2 = np.empty(0, dtype=np.float64)
        ids_list.append(ids)
        d2_list.append(np.asarray(d2, dtype=np.float64))
        count[qi] = len(ids)

    print(
        f"[{label}] fixed-Q support count: median={np.median(count):.1f}, "
        f"P10={np.percentile(count,10):.1f}, P90={np.percentile(count,90):.1f}"
    )
    return {
        "normals": Nn,
        "ids": ids_list,
        "d2": d2_list,
        "count": count,
    }


def compute_harris_from_cache(
    cache,
    sigma_mm: float,
    min_neighbors: int,
    label: str,
):
    """
    Compute PCL-style Harris response using either:
      sigma=0   : uniform/box support weighting
      sigma>0   : normalized Gaussian spatial weighting inside the Harris matrix
    """
    N = cache["normals"]
    ids_list = cache["ids"]
    d2_list = cache["d2"]

    response = np.zeros(len(ids_list), dtype=np.float64)
    effective_n = np.zeros(len(ids_list), dtype=np.float64)

    sigma = float(sigma_mm)
    denom = 2.0 * sigma * sigma if sigma > 0 else None

    for qi, ids in enumerate(ids_list):
        m = len(ids)
        if m < int(min_neighbors):
            continue

        NN = N[ids]
        if sigma <= 0:
            M = (NN.T @ NN) / float(m)
            effective_n[qi] = float(m)
        else:
            d2 = d2_list[qi]
            w = np.exp(-d2 / denom)
            sw = float(np.sum(w))
            if not np.isfinite(sw) or sw <= 1e-15:
                continue
            # Weighted normal second-moment / structure tensor.
            M = (NN.T * w) @ NN / sw
            # Kish effective sample size: useful diagnostic for overly narrow sigma.
            sw2 = float(np.dot(w, w))
            effective_n[qi] = (sw * sw / sw2) if sw2 > 1e-15 else 0.0

        tr = float(np.trace(M))
        if tr == 0.0 or not np.isfinite(tr):
            continue
        det = float(np.linalg.det(M))
        r = 0.04 + det - 0.04 * tr * tr
        if np.isfinite(r):
            response[qi] = r

    window_name = "UNIFORM" if sigma <= 0 else f"GAUSSIAN sigma={sigma:.3f} mm"
    valid_eff = effective_n[effective_n > 0]
    if len(valid_eff):
        print(
            f"[{label}] {window_name}: effective support median/P10/P90 = "
            f"{np.median(valid_eff):.1f}/{np.percentile(valid_eff,10):.1f}/"
            f"{np.percentile(valid_eff,90):.1f}"
        )
    return response, effective_n


# -----------------------------------------------------------------------------
# NMS + metrics
# -----------------------------------------------------------------------------


def build_radius_neighborhoods(points: np.ndarray, radius: float):
    tree = cKDTree(np.asarray(points, dtype=np.float64))
    return tree.query_ball_point(points, r=float(radius), workers=-1)


def radius_nms_on_fixed_queries(
    response: np.ndarray,
    query_neighborhoods,
    threshold: float,
    keypoints_k: int,
) -> np.ndarray:
    response = np.asarray(response, dtype=np.float64)
    candidates = np.flatnonzero(np.isfinite(response) & (response >= float(threshold)))

    maxima: list[int] = []
    for i in candidates:
        ids = np.asarray(query_neighborhoods[i], dtype=np.int64)
        if np.any(response[ids] > response[i]):
            continue
        maxima.append(int(i))

    if not maxima:
        return np.empty(0, dtype=np.int64)

    maxima = np.asarray(maxima, dtype=np.int64)
    order = np.argsort(response[maxima])[::-1]
    maxima = maxima[order]
    if keypoints_k > 0:
        maxima = maxima[:keypoints_k]
    return np.ascontiguousarray(maxima)


def safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(ok) < 3:
        return float("nan")
    xx, yy = x[ok], y[ok]
    if np.all(xx == xx[0]) or np.all(yy == yy[0]):
        return float("nan")
    r = spearmanr(xx, yy).statistic
    return float(r) if np.isfinite(r) else float("nan")


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ok = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(ok) < 2:
        return float("nan")
    xx, yy = x[ok], y[ok]
    if np.std(xx) == 0 or np.std(yy) == 0:
        return float("nan")
    return float(np.corrcoef(xx, yy)[0, 1])


def nearest_points(src: np.ndarray, dst: np.ndarray):
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if len(src) == 0:
        return np.empty(0), np.empty(0, dtype=np.int64)
    if len(dst) == 0:
        return np.full(len(src), np.inf), np.full(len(src), -1, dtype=np.int64)
    tree = cKDTree(dst)
    d, idx = tree.query(src, k=1, workers=-1)
    return np.asarray(d), np.asarray(idx, dtype=np.int64)


def nms_repeatability(Q, kpA, kpB, radii):
    d_ab, _ = nearest_points(Q[kpA], Q[kpB])
    d_ba, _ = nearest_points(Q[kpB], Q[kpA])
    out = {}
    for eps in radii:
        rab = float(np.mean(d_ab <= eps)) if len(kpA) else float("nan")
        rba = float(np.mean(d_ba <= eps)) if len(kpB) else float("nan")
        out[float(eps)] = (rab, rba, float(np.nanmean([rab, rba])))
    return out, d_ab, d_ba


def exact_nms_stats(kpA, kpB):
    a = set(map(int, np.asarray(kpA).tolist()))
    b = set(map(int, np.asarray(kpB).tolist()))
    inter = len(a & b)
    union = len(a | b)
    return {
        "intersection": inter,
        "jaccard": inter / union if union else float("nan"),
        "A_retention": inter / len(kpA) if len(kpA) else float("nan"),
        "B_retention": inter / len(kpB) if len(kpB) else float("nan"),
    }


def candidate_jaccard(RA, RB, threshold):
    ca = np.isfinite(RA) & (RA >= threshold)
    cb = np.isfinite(RB) & (RB >= threshold)
    inter = int(np.count_nonzero(ca & cb))
    union = int(np.count_nonzero(ca | cb))
    return (
        inter / union if union else float("nan"),
        int(np.count_nonzero(ca)),
        int(np.count_nonzero(cb)),
    )


# -----------------------------------------------------------------------------
# Local ranking diagnostic
# -----------------------------------------------------------------------------


def local_ranking_summary(Q, RA, RB, kpA, local_neighborhoods):
    rows = []
    for qi in np.asarray(kpA, dtype=np.int64):
        ids = np.asarray(local_neighborhoods[qi], dtype=np.int64)
        valid = np.isfinite(RA[ids]) & np.isfinite(RB[ids])
        ids = ids[valid]
        if len(ids) < 3:
            continue

        a = RA[ids]
        b = RB[ids]
        rho = safe_spearman(a, b)
        pear = safe_pearson(a, b)
        order_a = np.argsort(a)[::-1]
        order_b = np.argsort(b)[::-1]
        q1 = int(ids[order_a[0]])
        q2 = int(ids[order_a[1]])
        qb1 = int(ids[order_b[0]])

        rows.append(
            {
                "a_peak_query_index": int(qi),
                "a_local_winner_query_index": q1,
                "a_second_query_index": q2,
                "b_local_winner_query_index": qb1,
                "n_local_queries": int(len(ids)),
                "local_spearman": rho,
                "local_pearson": pear,
                "same_winner": bool(q1 == qb1),
                "top2_flip": bool(RB[q1] < RB[q2]),
                "a_winner_rank_in_b": int(1 + np.count_nonzero(b > RB[q1])),
                "winner_displacement_mm": float(np.linalg.norm(Q[qb1] - Q[q1])),
            }
        )

    if not rows:
        return rows, {
            "local_spearman_median": float("nan"),
            "local_pearson_median": float("nan"),
            "same_winner_rate": float("nan"),
            "top2_flip_rate": float("nan"),
            "winner_rank_b_median": float("nan"),
            "winner_disp_median": float("nan"),
            "winner_disp_p90": float("nan"),
        }

    rho = np.asarray([r["local_spearman"] for r in rows], dtype=float)
    pear = np.asarray([r["local_pearson"] for r in rows], dtype=float)
    same = np.asarray([r["same_winner"] for r in rows], dtype=bool)
    flip = np.asarray([r["top2_flip"] for r in rows], dtype=bool)
    rank = np.asarray([r["a_winner_rank_in_b"] for r in rows], dtype=float)
    disp = np.asarray([r["winner_displacement_mm"] for r in rows], dtype=float)

    good_rho = rho[np.isfinite(rho)]
    good_pear = pear[np.isfinite(pear)]
    summary = {
        "local_spearman_median": float(np.median(good_rho)) if len(good_rho) else float("nan"),
        "local_pearson_median": float(np.median(good_pear)) if len(good_pear) else float("nan"),
        "same_winner_rate": float(np.mean(same)),
        "top2_flip_rate": float(np.mean(flip)),
        "winner_rank_b_median": float(np.median(rank)),
        "winner_disp_median": float(np.median(disp)),
        "winner_disp_p90": float(np.percentile(disp, 90)),
    }
    return rows, summary


# -----------------------------------------------------------------------------
# Sweep
# -----------------------------------------------------------------------------


def sigma_tag(sigma: float) -> str:
    return "uniform" if sigma <= 0 else f"g{sigma:g}".replace(".", "p")


def run_window_sweep(Q, cacheA, cacheB, nms_neigh, local_neigh, args):
    summary_rows = []
    peak_rows_all = []
    fields = {}

    print("\n=== GAUSSIAN WINDOW SWEEP (INSIDE HARRIS, NO RESPONSE SMOOTHING) ===")
    header = (
        f"{'sigma':>8} | {'#A':>4} {'#B':>4} | {'RespSp':>6} | {'CandJ':>6} | "
        f"{'LocSp':>6} | {'Flip':>6} | {'SameW':>6} | {'Disp50':>7} | "
        + " | ".join([f"R@{e:g}" for e in args.report_mm])
    )
    print(header)
    print("-" * len(header))

    for sigma in args.window_sigmas_mm:
        sigma = float(sigma)
        label = "uniform" if sigma <= 0 else f"gauss-{sigma:g}mm"
        RA, effA = compute_harris_from_cache(
            cacheA, sigma, args.min_neighbors, f"A {label}"
        )
        RB, effB = compute_harris_from_cache(
            cacheB, sigma, args.min_neighbors, f"B {label}"
        )

        kpA = radius_nms_on_fixed_queries(
            RA, nms_neigh, args.harris_threshold, args.keypoints_k
        )
        kpB = radius_nms_on_fixed_queries(
            RB, nms_neigh, args.harris_threshold, args.keypoints_k
        )

        reps, d_ab, d_ba = nms_repeatability(Q, kpA, kpB, args.report_mm)
        exact = exact_nms_stats(kpA, kpB)
        cand_j, candA, candB = candidate_jaccard(RA, RB, args.harris_threshold)
        local_rows, local = local_ranking_summary(Q, RA, RB, kpA, local_neigh)

        row = {
            "window_sigma_mm": sigma,
            "window_type": "uniform" if sigma <= 0 else "gaussian",
            "response_pearson": safe_pearson(RA, RB),
            "response_spearman": safe_spearman(RA, RB),
            "candidate_jaccard": cand_j,
            "candidate_A": candA,
            "candidate_B": candB,
            "nms_A": len(kpA),
            "nms_B": len(kpB),
            "exact_intersection": exact["intersection"],
            "exact_A_retention": exact["A_retention"],
            "exact_B_retention": exact["B_retention"],
            "nms_jaccard": exact["jaccard"],
            **local,
            "effective_support_A_median": float(np.median(effA[effA > 0])) if np.any(effA > 0) else float("nan"),
            "effective_support_B_median": float(np.median(effB[effB > 0])) if np.any(effB > 0) else float("nan"),
        }
        for eps, (rab, rba, sym) in reps.items():
            row[f"AtoB_{eps:g}mm"] = rab
            row[f"BtoA_{eps:g}mm"] = rba
            row[f"sym_{eps:g}mm"] = sym
        summary_rows.append(row)

        for pr in local_rows:
            pr = dict(pr)
            pr["window_sigma_mm"] = sigma
            pr["window_type"] = row["window_type"]
            peak_rows_all.append(pr)

        fields[sigma] = {
            "RA": RA,
            "RB": RB,
            "kpA": kpA,
            "kpB": kpB,
            "effA": effA,
            "effB": effB,
            "d_ab": d_ab,
            "d_ba": d_ba,
        }

        rep_txt = " | ".join(f"{100*reps[e][2]:5.1f}%" for e in args.report_mm)
        print(
            f"{sigma:8.3f} | {len(kpA):4d} {len(kpB):4d} | "
            f"{row['response_spearman']:6.3f} | {100*cand_j:5.1f}% | "
            f"{row['local_spearman_median']:6.3f} | {100*row['top2_flip_rate']:5.1f}% | "
            f"{100*row['same_winner_rate']:5.1f}% | {row['winner_disp_median']:7.3f} | "
            f"{rep_txt}"
        )

    # Find best Gaussian by primary spatial repeatability.
    key = f"sym_{args.repeat_mm:g}mm"
    uniform = next(r for r in summary_rows if r["window_sigma_mm"] <= 0)
    gauss_rows = [r for r in summary_rows if r["window_sigma_mm"] > 0]
    best_gauss = max(gauss_rows, key=lambda r: r.get(key, float("-inf"))) if gauss_rows else uniform

    print("\n=== PRIMARY COMPARISON ===")
    print(f"primary repeatability radius : {args.repeat_mm:.3f} mm")
    print(f"uniform window               : {100*uniform[key]:.2f}%")
    if best_gauss["window_sigma_mm"] > 0:
        print(f"best Gaussian sigma          : {best_gauss['window_sigma_mm']:.3f} mm")
        print(f"best Gaussian repeatability  : {100*best_gauss[key]:.2f}%")
        print(
            "absolute change               : "
            f"{100*(best_gauss[key]-uniform[key]):+.2f} %-points"
        )
        print(
            "winner displacement median    : "
            f"{uniform['winner_disp_median']:.3f} -> "
            f"{best_gauss['winner_disp_median']:.3f} mm"
        )
        print(
            "top1/top2 flip rate           : "
            f"{100*uniform['top2_flip_rate']:.2f}% -> "
            f"{100*best_gauss['top2_flip_rate']:.2f}%"
        )
        print(
            "local Spearman median         : "
            f"{uniform['local_spearman_median']:.4f} -> "
            f"{best_gauss['local_spearman_median']:.4f}"
        )

    return summary_rows, peak_rows_all, fields, uniform, best_gauss


# -----------------------------------------------------------------------------
# Save
# -----------------------------------------------------------------------------


def save_summary_csv(path: Path, rows, report_mm):
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "window_sigma_mm", "window_type",
        "response_pearson", "response_spearman",
        "candidate_jaccard", "candidate_A", "candidate_B",
        "nms_A", "nms_B", "exact_intersection",
        "exact_A_retention", "exact_B_retention", "nms_jaccard",
        "local_spearman_median", "local_pearson_median",
        "same_winner_rate", "top2_flip_rate", "winner_rank_b_median",
        "winner_disp_median", "winner_disp_p90",
        "effective_support_A_median", "effective_support_B_median",
    ]
    for eps in report_mm:
        fields += [f"AtoB_{eps:g}mm", f"BtoA_{eps:g}mm", f"sym_{eps:g}mm"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in fields})
    print(f"Saved Gaussian-window summary CSV: {path}")


def save_peak_csv(path: Path, rows):
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = [
        "window_sigma_mm", "window_type",
        "a_peak_query_index", "a_local_winner_query_index",
        "a_second_query_index", "b_local_winner_query_index",
        "n_local_queries", "local_spearman", "local_pearson",
        "same_winner", "top2_flip", "a_winner_rank_in_b",
        "winner_displacement_mm",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k) for k in fields})
    print(f"Saved Gaussian-window per-peak CSV: {path}")


def save_npz(path: Path, Q, fields):
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"query_points": Q}
    for sigma, d in fields.items():
        tag = sigma_tag(float(sigma))
        payload[f"response_A_{tag}"] = d["RA"]
        payload[f"response_B_{tag}"] = d["RB"]
        payload[f"nms_A_{tag}"] = d["kpA"]
        payload[f"nms_B_{tag}"] = d["kpB"]
        payload[f"effective_support_A_{tag}"] = d["effA"]
        payload[f"effective_support_B_{tag}"] = d["effB"]
    np.savez_compressed(path, **payload)
    print(f"Saved Gaussian-window response fields NPZ: {path}")


# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------


def np_to_o3d(points: np.ndarray, color=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    if color is not None:
        pcd.paint_uniform_color(color)
    return pcd


def sphere_cloud(points: np.ndarray, color, radius: float, max_n: int):
    geoms = []
    for p in np.asarray(points)[:max_n]:
        s = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        s.translate(np.asarray(p, dtype=np.float64))
        s.paint_uniform_color(color)
        geoms.append(s)
    return geoms


def make_lines(a: np.ndarray, b: np.ndarray, color, max_lines: int):
    n = min(len(a), len(b), int(max_lines))
    if n <= 0:
        return None
    pts, lines, colors = [], [], []
    for k in range(n):
        pts.extend([a[k], b[k]])
        lines.append([2*k, 2*k+1])
        colors.append(color)
    ls = o3d.geometry.LineSet()
    ls.points = o3d.utility.Vector3dVector(np.asarray(pts, dtype=np.float64))
    ls.lines = o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
    ls.colors = o3d.utility.Vector3dVector(np.asarray(colors, dtype=np.float64))
    return ls


def visualize(Q, A, B, fields, uniform_row, best_row, args):
    geoms = [
        np_to_o3d(A, [0.72, 0.72, 0.72]),
        np_to_o3d(B, [0.25, 0.55, 1.0]),
    ]

    fu = fields[uniform_row["window_sigma_mm"]]
    fg = fields[best_row["window_sigma_mm"]]

    # Uniform A peaks in red, best-Gaussian A peaks in green, B Gaussian in purple.
    geoms += sphere_cloud(Q[fu["kpA"]], [1.0, 0.0, 0.0], 1.15, args.max_spheres)
    geoms += sphere_cloud(Q[fg["kpA"]], [0.0, 0.85, 0.1], 1.05, args.max_spheres)
    geoms += sphere_cloud(Q[fg["kpB"]], [0.65, 0.0, 0.9], 0.8, args.max_spheres)

    # Draw best Gaussian A -> nearest best Gaussian B for missed @ primary radius.
    d, idx = nearest_points(Q[fg["kpA"]], Q[fg["kpB"]])
    missed = np.flatnonzero(d > args.repeat_mm)
    if len(missed):
        apts = Q[fg["kpA"][missed]]
        bpts = Q[fg["kpB"][idx[missed]]]
        line = make_lines(apts, bpts, [1.0, 0.55, 0.0], args.max_lines)
        if line is not None:
            geoms.append(line)

    print("\n[VIS]")
    print("  gray / blue surfaces = independent support samples A/B")
    print("  red spheres          = uniform-window A NMS peaks")
    print(
        f"  green spheres        = best Gaussian-window A NMS peaks "
        f"(sigma={best_row['window_sigma_mm']:.3f} mm)"
    )
    print("  purple spheres       = best Gaussian-window B NMS peaks")
    print("  orange lines         = Gaussian A peaks missed in B @ primary radius")

    o3d.visualization.draw_geometries(
        geoms,
        window_name="Harris3D Gaussian-window diagnosis",
    )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main():
    args = parse_args()

    if args.support_sample_points < 10:
        raise ValueError("--support-sample-points must be >= 10")
    if args.query_source == "sample" and args.query_sample_points < 10:
        raise ValueError("--query-sample-points must be >= 10")
    if args.support_voxel_mm < 0 or args.query_voxel_mm < 0:
        raise ValueError("voxel sizes must be >= 0")
    if args.harris_radius_mm <= 0 or args.harris_threshold <= 0:
        raise ValueError("Harris radius/threshold must be positive")
    if args.local_radius_mm <= 0 or args.repeat_mm <= 0:
        raise ValueError("local/repeat radii must be positive")
    if args.keypoints_k < 0:
        raise ValueError("--keypoints-k must be >= 0")

    print("=== HARRIS3D GAUSSIAN SPATIAL-WINDOW DIAGNOSIS ===")
    print(f"CAD                  : {args.cad}")
    print(f"support samples      : {args.support_sample_points}")
    print(f"seed A / B           : {args.seed_a} / {args.seed_b}")
    print(f"support voxel        : {args.support_voxel_mm:.3f} mm")
    print(f"query source         : {args.query_source}")
    if args.query_source == "sample":
        print(f"query samples        : {args.query_sample_points}, seed={args.seed_q}")
    print(f"query voxel          : {args.query_voxel_mm:.3f} mm")
    print(f"Harris radius        : {args.harris_radius_mm:.3f} mm")
    print(f"window sigmas        : {args.window_sigmas_mm}")
    print(f"local rank radius    : {args.local_radius_mm:.3f} mm")
    print(f"Harris threshold     : {args.harris_threshold:g}")
    print(f"primary repeat radius: {args.repeat_mm:.3f} mm")
    print("[CONTROL] exact face normals; identical Q; no noise/outliers/partiality")
    print("[VARIABLE] only the spatial weighting INSIDE the Harris moment matrix")
    print("[IMPORTANT] no post-response smoothing is used in this script")

    mesh = load_mesh(args.cad)
    V = np.asarray(mesh.vertices, dtype=np.float64)
    common_origin = V.min(axis=0) - 1e-9

    A_raw, NA_raw = sample_mesh_with_face_normals(
        mesh, args.support_sample_points, args.seed_a, "A-support"
    )
    B_raw, NB_raw = sample_mesh_with_face_normals(
        mesh, args.support_sample_points, args.seed_b, "B-support"
    )
    A, NA = voxel_representative(
        A_raw, args.support_voxel_mm, common_origin, "A-support", normals=NA_raw
    )
    B, NB = voxel_representative(
        B_raw, args.support_voxel_mm, common_origin, "B-support", normals=NB_raw
    )

    if args.query_source == "vertices":
        Q_raw = V.copy()
        print(f"[Q] mesh vertices: {len(Q_raw)}")
    else:
        Q_raw = sample_mesh_points_only(
            mesh, args.query_sample_points, args.seed_q, "Q-fixed"
        )
    Q = voxel_representative(
        Q_raw, args.query_voxel_mm, common_origin, "Q-fixed", normals=None
    )

    cacheA = build_support_cache(Q, A, NA, args.harris_radius_mm, "A->Q")
    cacheB = build_support_cache(Q, B, NB, args.harris_radius_mm, "B->Q")

    print("[Q] building shared NMS neighborhood graph ...")
    nms_neigh = build_radius_neighborhoods(Q, args.harris_radius_mm)
    if abs(args.local_radius_mm - args.harris_radius_mm) < 1e-12:
        local_neigh = nms_neigh
    else:
        print("[Q] building local-ranking neighborhood graph ...")
        local_neigh = build_radius_neighborhoods(Q, args.local_radius_mm)

    summary_rows, peak_rows, fields, uniform_row, best_row = run_window_sweep(
        Q, cacheA, cacheB, nms_neigh, local_neigh, args
    )

    print("\n=== INTERPRETATION ===")
    print("* If Gaussian windowing raises local Spearman, lowers top1/top2 flips,")
    print("  lowers winner displacement, AND raises NMS repeatability, then the hard")
    print("  uniform support window was a meaningful source of sampling instability.")
    print("* If candidate/response similarity stays high but point repeatability remains")
    print("  low even with the best Gaussian sigma, then Gaussian weighting alone does")
    print("  not solve the sampling-dependent localization/NMS problem.")
    print("* Very small sigma can reduce effective support too much; inspect the printed")
    print("  effective-support values before judging a narrow Gaussian as 'better/worse'.")

    if args.save_summary_csv is not None:
        save_summary_csv(args.save_summary_csv, summary_rows, args.report_mm)
    if args.save_peak_csv is not None:
        save_peak_csv(args.save_peak_csv, peak_rows)
    if args.save_npz is not None:
        save_npz(args.save_npz, Q, fields)
    if args.show:
        visualize(Q, A, B, fields, uniform_row, best_row, args)


if __name__ == "__main__":
    main()