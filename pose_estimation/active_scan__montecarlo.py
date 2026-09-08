#!/usr/bin/env python3
"""
Random 2-D laser scan -> Drost-style PPF voting -> SE(3) pose clustering
-> Top-1 cluster -> point-to-plane ICP.

This stops before active Scan-2 selection.

Example:
python /home/choisuhyun/lvs_HandEyeCalibration/pose_estimation/active_scan__montecarlo.py /home/choisuhyun/lvs_HandEyeCalibration/pose_estimation/data/042_adjustable_wrench/google_16k/nontextured.stl --mesh-unit m --show
"""

from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys
from typing import Any

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

EPS = 1.0e-12


def require_open3d():
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError("Open3D is required: pip install open3d") from exc
    return o3d


@dataclass
class RandomScanPlan:
    point_cad_m: np.ndarray
    normal_cad: np.ndarray
    angle_deg: float
    profile_axis_cad: np.ndarray
    sweep_axis_cad: np.ndarray


@dataclass
class ScanGeometry:
    clean_points_m: np.ndarray
    ray_directions: np.ndarray
    lateral_directions: np.ndarray
    path_origins_m: np.ndarray
    sensor_viewpoint_m: np.ndarray


@dataclass
class PPFModel:
    points_m: np.ndarray
    normals: np.ndarray
    diameter_m: float
    dist_step_m: float
    angle_step_rad: float
    nangle: int
    hash_table: dict[tuple[int, int, int, int], np.ndarray]


@dataclass
class PoseHypothesis:
    transform_model_to_world: np.ndarray
    votes: float
    scene_reference_id: int
    model_reference_id: int
    alpha_bin: int


@dataclass
class PoseCluster:
    rank: int
    transform_model_to_world: np.ndarray
    score: float
    member_count: int
    members: list[int]
    translation_spread_mm: float
    rotation_spread_deg: float


def positive_float(v: str) -> float:
    x = float(v)
    if x <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return x


def positive_int(v: str) -> int:
    x = int(v)
    if x <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return x


def probability(v: str) -> float:
    x = float(v)
    if not 0.0 <= x < 1.0:
        raise argparse.ArgumentTypeError("must be in [0,1)")
    return x


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Random laser scan + PPF pose clustering + Top-1 ICP"
    )
    p.add_argument("cad", type=Path)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--mesh-unit", choices=("auto", "m", "mm"), default="auto")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--show", action="store_true")

    g = p.add_argument_group("PPF model")
    g.add_argument("--cad-sample-points", type=positive_int, default=120000)
    g.add_argument("--ppf-model-points", type=positive_int, default=800)
    g.add_argument("--ppf-tau-d", type=positive_float, default=0.05)
    g.add_argument("--ppf-nangle", type=positive_int, default=30)
    g.add_argument("--max-model-pairs-per-bin", type=positive_int, default=300)

    g = p.add_argument_group("Random virtual 2-D laser scan")
    g.add_argument("--scan-length-mm", type=positive_float, default=50.0)
    g.add_argument("--scan-step-mm", type=positive_float, default=0.5)
    g.add_argument("--profile-width-mm", type=positive_float, default=32.0)
    g.add_argument("--profile-points", type=positive_int, default=321)
    g.add_argument("--standoff-mm", type=positive_float, default=80.0)
    g.add_argument("--first-scan-angle-deg", type=float, default=None)
    g.add_argument("--range-sigma-mm", type=positive_float, default=0.20)
    g.add_argument("--lateral-sigma-mm", type=positive_float, default=0.08)
    g.add_argument("--dropout-rate", type=probability, default=0.03)
    g.add_argument("--outlier-rate", type=probability, default=0.01)
    g.add_argument("--outlier-sigma-mm", type=positive_float, default=3.0)

    g = p.add_argument_group("Hidden GT")
    g.add_argument("--gt-translation-mm", type=float, default=40.0)
    g.add_argument("--gt-rotation-deg", type=float, default=20.0)

    g = p.add_argument_group("PPF voting")
    g.add_argument("--ppf-scene-voxel-mm", type=positive_float, default=2.0)
    g.add_argument("--ppf-scene-normal-radius-mm", type=positive_float, default=6.0)
    g.add_argument("--ppf-reference-fraction", type=float, default=0.20)
    g.add_argument("--ppf-peak-relative", type=float, default=0.55)
    g.add_argument("--ppf-peaks-per-reference", type=positive_int, default=5)
    g.add_argument("--ppf-max-hypotheses", type=positive_int, default=500)

    g = p.add_argument_group("Pose clustering")
    g.add_argument("--cluster-translation-mm", type=positive_float, default=10.0)
    g.add_argument("--cluster-rotation-deg", type=positive_float, default=10.0)
    g.add_argument("--plot-top-clusters", type=positive_int, default=12)

    g = p.add_argument_group("Top-1 ICP")
    g.add_argument("--icp-cad-voxel-mm", type=positive_float, default=1.0)
    g.add_argument("--icp-normal-radius-mm", type=positive_float, default=3.0)
    g.add_argument("--icp-distance-mm", type=positive_float, default=3.0)


    g = p.add_argument_group("Greedy information next scan")
    g.add_argument("--candidate-count", type=positive_int, default=24)
    g.add_argument("--candidate-nms-mm", type=positive_float, default=15.0)
    g.add_argument("--orientation-samples", type=positive_int, default=4)
    g.add_argument("--max-correspondence-mm", type=positive_float, default=3.0)
    g.add_argument("--prior-translation-mm", type=positive_float, default=10.0)
    g.add_argument("--prior-rotation-deg", type=positive_float, default=10.0)
    g.add_argument(
        "--candidate-exclude-first-mm",
        type=float,
        default=10.0,
        help="Do not choose a Scan-2 candidate this close to the random Scan-1 center.",
    )

    g = p.add_argument_group("Monte Carlo")
    g.add_argument("--mc-trials", type=positive_int, default=20)
    g.add_argument("--success-translation-mm", type=positive_float, default=2.0)
    g.add_argument("--success-rotation-deg", type=positive_float, default=2.0)
    g.add_argument(
        "--save-first-trial-debug",
        action="store_true",
        help="Save PPF cluster vote plot and point clouds for the first successful trial.",
    )

    return p.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.cad.is_file():
        raise ValueError(f"CAD file does not exist: {args.cad}")
    if not 0.0 < args.ppf_reference_fraction <= 1.0:
        raise ValueError("--ppf-reference-fraction must be in (0,1]")
    if not 0.0 < args.ppf_peak_relative <= 1.0:
        raise ValueError("--ppf-peak-relative must be in (0,1]")
    if args.dropout_rate + args.outlier_rate >= 1.0:
        raise ValueError("dropout + outlier rate must be < 1")
    if args.gt_translation_mm < 0 or args.gt_rotation_deg < 0:
        raise ValueError("GT magnitudes must be non-negative")
    if args.candidate_exclude_first_mm < 0:
        raise ValueError("--candidate-exclude-first-mm must be non-negative")


def normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(v))
    if n <= EPS:
        raise ValueError("cannot normalize zero vector")
    return v / n


def invert_transform(T: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    R = np.asarray(T[:3, :3], dtype=np.float64)
    out[:3, :3] = R.T
    out[:3, 3] = -(R.T @ np.asarray(T[:3, 3], dtype=np.float64))
    return out


def transform_points(points: np.ndarray, T: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64)
    return points @ T[:3, :3].T + T[:3, 3]


def rotation_distance_deg(R1: np.ndarray, R2: np.ndarray) -> float:
    R = R1.T @ R2
    c = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    return math.degrees(math.acos(c))


def pose_error_model_to_world(est: np.ndarray, gt: np.ndarray) -> dict[str, float]:
    E = invert_transform(gt) @ est
    return {
        "translation_mm": float(np.linalg.norm(E[:3, 3]) * 1000.0),
        "rotation_deg": float(
            np.degrees(np.linalg.norm(Rotation.from_matrix(E[:3, :3]).as_rotvec()))
        ),
    }


def tangent_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = normalize(normal)
    axes = np.eye(3)
    helper = axes[int(np.argmin(np.abs(axes @ n)))]
    t1 = normalize(np.cross(n, helper))
    t2 = normalize(np.cross(n, t1))
    return t1, t2


def make_cloud(points: np.ndarray, color=None):
    o3d = require_open3d()
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    if color is not None:
        cloud.paint_uniform_color(color)
    return cloud


def make_path_line(points: np.ndarray, color):
    o3d = require_open3d()
    line = o3d.geometry.LineSet()
    line.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    if len(points) >= 2:
        ids = np.column_stack((np.arange(len(points)-1), np.arange(1, len(points))))
        line.lines = o3d.utility.Vector2iVector(ids.astype(np.int32))
        line.colors = o3d.utility.Vector3dVector(
            np.tile(np.asarray(color, dtype=np.float64), (len(ids), 1))
        )
    return line


def load_centered_mesh(path: Path, mesh_unit: str):
    o3d = require_open3d()
    mesh = o3d.io.read_triangle_mesh(str(path), enable_post_processing=True)
    if mesh.is_empty() or len(mesh.triangles) == 0:
        raise ValueError(f"failed to load mesh: {path}")

    bbox = mesh.get_axis_aligned_bounding_box()
    extent_raw = np.asarray(bbox.get_extent(), dtype=np.float64)
    diag_raw = float(np.linalg.norm(extent_raw))
    center_raw = np.asarray(bbox.get_center(), dtype=np.float64)

    if mesh_unit == "m":
        scale, unit = 1.0, "m"
    elif mesh_unit == "mm":
        scale, unit = 0.001, "mm"
    else:
        scale = 0.001 if diag_raw > 10.0 else 1.0
        unit = "mm (auto)" if scale == 0.001 else "m (auto)"

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    vertices[:] = (vertices - center_raw) * scale
    mesh.compute_vertex_normals()

    return mesh, {
        "input_unit": unit,
        "scale_input_to_m": scale,
        "input_bbox_center": center_raw.tolist(),
        "input_bbox_extent": extent_raw.tolist(),
        "diameter_mm": diag_raw * scale * 1000.0,
    }


def sample_oriented_cad(mesh, sample_points: int, target_points: int):
    dense = mesh.sample_points_uniformly(number_of_points=sample_points)
    if not dense.has_normals():
        dense.estimate_normals()

    diagonal = float(np.linalg.norm(dense.get_axis_aligned_bounding_box().get_extent()))
    voxel = max(diagonal / 80.0, 0.0005)
    cloud = dense.voxel_down_sample(voxel)

    for _ in range(20):
        n = len(cloud.points)
        if 0.75 * target_points <= n <= 1.25 * target_points:
            break
        voxel *= 1.10 if n > target_points else 0.90
        cloud = dense.voxel_down_sample(voxel)

    pts = np.asarray(cloud.points, dtype=np.float64)
    nrm = np.asarray(cloud.normals, dtype=np.float64)
    if len(pts) > target_points:
        r = np.random.default_rng(12345)
        ids = np.sort(r.choice(len(pts), size=target_points, replace=False))
        pts, nrm = pts[ids], nrm[ids]

    nrm = np.asarray([normalize(x) for x in nrm])
    center = pts.mean(axis=0)
    if float(np.median(np.einsum("ij,ij->i", pts-center, nrm))) < 0:
        nrm *= -1.0
    return pts, nrm


def random_gt_transform(args, rng):
    T = np.eye(4)
    if args.gt_rotation_deg > 0:
        axis = normalize(rng.normal(size=3))
        T[:3, :3] = Rotation.from_rotvec(
            axis * math.radians(args.gt_rotation_deg)
        ).as_matrix()
    if args.gt_translation_mm > 0:
        direction = normalize(rng.normal(size=3))
        T[:3, 3] = direction * args.gt_translation_mm / 1000.0
    return T


def random_surface_scan_plan(mesh, args, rng) -> RandomScanPlan:
    sample = mesh.sample_points_uniformly(number_of_points=20000)
    if not sample.has_normals():
        sample.estimate_normals()
    pts = np.asarray(sample.points)
    nrm = np.asarray(sample.normals)
    idx = int(rng.integers(0, len(pts)))
    point = pts[idx].copy()
    normal = normalize(nrm[idx].copy())
    if np.dot(point, normal) < 0:
        normal *= -1.0

    t1, t2 = tangent_basis(normal)
    angle = (
        float(rng.uniform(0.0, math.pi))
        if args.first_scan_angle_deg is None
        else math.radians(args.first_scan_angle_deg)
    )
    return RandomScanPlan(
        point_cad_m=point,
        normal_cad=normal,
        angle_deg=math.degrees(angle),
        profile_axis_cad=normalize(math.cos(angle)*t1 + math.sin(angle)*t2),
        sweep_axis_cad=normalize(-math.sin(angle)*t1 + math.cos(angle)*t2),
    )


def build_raycast_scene(mesh):
    o3d = require_open3d()
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return scene


def simulate_scan_from_plan_world(world_scene, plan, cad_to_world, args):
    o3d = require_open3d()
    R = cad_to_world[:3, :3]
    center = R @ plan.point_cad_m + cad_to_world[:3, 3]
    normal = normalize(R @ plan.normal_cad)
    profile_axis = normalize(R @ plan.profile_axis_cad)
    sweep_axis = normalize(R @ plan.sweep_axis_cad)

    scan_offsets = np.arange(
        -0.5*args.scan_length_mm,
        0.5*args.scan_length_mm + 0.5*args.scan_step_mm,
        args.scan_step_mm,
    ) / 1000.0
    profile_offsets = np.linspace(
        -0.5*args.profile_width_mm,
        0.5*args.profile_width_mm,
        args.profile_points,
    ) / 1000.0
    standoff = args.standoff_mm / 1000.0

    path = center[None, :] + scan_offsets[:, None]*sweep_axis + standoff*normal
    origins = np.repeat(path, args.profile_points, axis=0)
    centers = center[None, :] + scan_offsets[:, None]*sweep_axis
    targets = (
        centers[:, None, :] + profile_offsets[None, :, None]*profile_axis
    ).reshape(-1, 3)
    directions = targets - origins
    directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), EPS)

    rays = np.hstack((origins, directions)).astype(np.float32)
    t_hit = world_scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy().astype(float)
    valid = (
        np.isfinite(t_hit)
        & (t_hit > 0.25*standoff)
        & (t_hit < 1.75*standoff)
    )
    points = origins[valid] + t_hit[valid, None]*directions[valid]
    lateral = np.repeat(profile_axis[None, :], len(points), axis=0)
    return ScanGeometry(
        clean_points_m=points,
        ray_directions=directions[valid],
        lateral_directions=lateral,
        path_origins_m=path,
        sensor_viewpoint_m=np.mean(path, axis=0),
    )


def noisy_scan_points(scan, args, rng):
    n = len(scan.clean_points_m)
    keep = rng.random(n) >= args.dropout_rate
    pts = scan.clean_points_m[keep].copy()
    rays = scan.ray_directions[keep]
    lateral = scan.lateral_directions[keep]

    pts += rng.normal(
        0.0, args.range_sigma_mm/1000.0, size=(len(pts), 1)
    ) * rays
    pts += rng.normal(
        0.0, args.lateral_sigma_mm/1000.0, size=(len(pts), 1)
    ) * lateral

    outlier = rng.random(len(pts)) < args.outlier_rate
    if np.any(outlier):
        pts[outlier] += rng.normal(
            0.0,
            args.outlier_sigma_mm/1000.0,
            size=(int(outlier.sum()), 1),
        ) * rays[outlier]
    return pts


def canonical_rotation_world_to_local(normal):
    x = normalize(normal)
    axes = np.eye(3)
    helper = axes[int(np.argmin(np.abs(axes @ x)))]
    y = normalize(np.cross(helper, x))
    z = normalize(np.cross(x, y))
    return np.vstack((x, y, z))


def canonical_transform_world_to_local(point, normal):
    R = canonical_rotation_world_to_local(normal)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = -(R @ point)
    return T


def pair_alpha(ref_point, ref_normal, other_point):
    R = canonical_rotation_world_to_local(ref_normal)
    v = R @ (other_point - ref_point)
    return math.atan2(-float(v[2]), float(v[1])) % (2.0*math.pi)


def quantize_feature(distance, a1, a2, a3, dist_step, angle_step):
    return (
        int(np.rint(distance / dist_step)),
        int(np.rint(a1 / angle_step)),
        int(np.rint(a2 / angle_step)),
        int(np.rint(a3 / angle_step)),
    )


def build_ppf_model(model_points, model_normals, diameter_m, args, rng):
    dist_step = args.ppf_tau_d * diameter_m
    angle_step = 2.0*math.pi / args.ppf_nangle
    table = {}
    n = len(model_points)

    print(
        f"  model points={n}, ordered pairs≈{n*(n-1):,}, "
        f"ddist={dist_step*1000:.3f} mm, dangle={math.degrees(angle_step):.1f} deg"
    )

    for i in range(n):
        dvec = model_points - model_points[i]
        dist = np.linalg.norm(dvec, axis=1)
        ids = np.where((np.arange(n) != i) & (dist > EPS))[0]
        vec = dvec[ids]
        dhat = vec / dist[ids, None]
        nj = model_normals[ids]

        a1 = np.arccos(np.clip(dhat @ model_normals[i], -1.0, 1.0))
        a2 = np.arccos(np.clip(np.einsum("ij,ij->i", nj, dhat), -1.0, 1.0))
        a3 = np.arccos(
            np.clip(
                np.einsum(
                    "ij,ij->i",
                    np.repeat(model_normals[i][None, :], len(ids), axis=0),
                    nj,
                ),
                -1.0,
                1.0,
            )
        )

        Rloc = canonical_rotation_world_to_local(model_normals[i])
        vlocal = vec @ Rloc.T
        alpha_m = np.mod(np.arctan2(-vlocal[:, 2], vlocal[:, 1]), 2.0*math.pi)

        qd = np.rint(dist[ids] / dist_step).astype(np.int32)
        q1 = np.rint(a1 / angle_step).astype(np.int16)
        q2 = np.rint(a2 / angle_step).astype(np.int16)
        q3 = np.rint(a3 / angle_step).astype(np.int16)

        for k in range(len(ids)):
            key = (int(qd[k]), int(q1[k]), int(q2[k]), int(q3[k]))
            table.setdefault(key, []).append((i, float(alpha_m[k])))

        if (i + 1) % max(1, n // 10) == 0 or i + 1 == n:
            print(f"    hash {i+1:4d}/{n}: bins={len(table):,}")

    compact = {}
    cap = args.max_model_pairs_per_bin
    for key, items in table.items():
        if len(items) > cap:
            sel = rng.choice(len(items), size=cap, replace=False)
            items = [items[j] for j in sel]
        compact[key] = np.asarray(items, dtype=np.float64)

    return PPFModel(
        points_m=model_points,
        normals=model_normals,
        diameter_m=diameter_m,
        dist_step_m=dist_step,
        angle_step_rad=angle_step,
        nangle=args.ppf_nangle,
        hash_table=compact,
    )


def preprocess_scene_for_ppf(scan_points_world, viewpoint_world, args):
    o3d = require_open3d()
    cloud = make_cloud(scan_points_world)
    cloud = cloud.voxel_down_sample(args.ppf_scene_voxel_mm / 1000.0)
    if len(cloud.points) < 20:
        raise RuntimeError(f"PPF scene has only {len(cloud.points)} points")

    cloud.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=args.ppf_scene_normal_radius_mm / 1000.0,
            max_nn=60,
        )
    )
    cloud.orient_normals_towards_camera_location(np.asarray(viewpoint_world))
    pts = np.asarray(cloud.points, dtype=np.float64)
    nrm = np.asarray(cloud.normals, dtype=np.float64)
    nrm = np.asarray([normalize(x) for x in nrm])
    return cloud, pts, nrm


def circular_bin_distance(a, b, n):
    d = abs(a-b) % n
    return min(d, n-d)


def local_accumulator_peaks(accumulator, peak_relative, max_peaks, nangle):
    if not accumulator:
        return []
    ordered = sorted(
        ((votes, mr, ab) for (mr, ab), votes in accumulator.items()),
        reverse=True,
    )
    local_max = ordered[0][0]
    threshold = peak_relative * local_max

    peaks = []
    for votes, mr, ab in ordered:
        if votes < threshold:
            break
        duplicate = any(
            pmr == mr and circular_bin_distance(ab, pab, nangle) <= 1
            for _, pmr, pab in peaks
        )
        if duplicate:
            continue
        peaks.append((votes, mr, ab))
        if len(peaks) >= max_peaks:
            break
    return peaks


def pose_from_ppf_local_coordinates(
    model_point,
    model_normal,
    scene_point,
    scene_normal,
    alpha,
):
    Tm = canonical_transform_world_to_local(model_point, model_normal)
    Ts = canonical_transform_world_to_local(scene_point, scene_normal)

    c, s = math.cos(alpha), math.sin(alpha)
    Rx = np.eye(4)
    Rx[:3, :3] = np.array(
        [[1.0, 0.0, 0.0],
         [0.0, c, -s],
         [0.0, s,  c]]
    )
    return invert_transform(Ts) @ Rx @ Tm


def ppf_vote_pose_hypotheses(model, scene_points, scene_normals, args, rng):
    n_scene = len(scene_points)
    ref_count = min(
        n_scene,
        max(1, int(round(args.ppf_reference_fraction * n_scene))),
    )
    reference_ids = np.sort(rng.choice(n_scene, size=ref_count, replace=False))

    print(
        f"  scene points={n_scene}, refs={ref_count} "
        f"({100.0*ref_count/n_scene:.1f}%)"
    )

    hypotheses = []

    for ref_order, sr_id in enumerate(reference_ids, start=1):
        sr = scene_points[sr_id]
        nr = scene_normals[sr_id]
        accumulator = {}

        dvec = scene_points - sr
        dist = np.linalg.norm(dvec, axis=1)

        for sj_id in range(n_scene):
            if sj_id == sr_id or dist[sj_id] <= EPS:
                continue

            d = dvec[sj_id]
            dhat = d / dist[sj_id]
            nj = scene_normals[sj_id]

            a1 = math.acos(float(np.clip(np.dot(nr, dhat), -1.0, 1.0)))
            a2 = math.acos(float(np.clip(np.dot(nj, dhat), -1.0, 1.0)))
            a3 = math.acos(float(np.clip(np.dot(nr, nj), -1.0, 1.0)))
            key = quantize_feature(
                float(dist[sj_id]), a1, a2, a3,
                model.dist_step_m, model.angle_step_rad,
            )

            model_pairs = model.hash_table.get(key)
            if model_pairs is None:
                continue

            alpha_s = pair_alpha(sr, nr, scene_points[sj_id])

            for row in model_pairs:
                mr_id = int(row[0])
                alpha_m = float(row[1])
                alpha = (alpha_m - alpha_s) % (2.0*math.pi)
                alpha_bin = int(np.rint(alpha / model.angle_step_rad)) % model.nangle
                key_acc = (mr_id, alpha_bin)
                accumulator[key_acc] = accumulator.get(key_acc, 0) + 1

        peaks = local_accumulator_peaks(
            accumulator,
            args.ppf_peak_relative,
            args.ppf_peaks_per_reference,
            model.nangle,
        )

        for votes, mr_id, alpha_bin in peaks:
            alpha = alpha_bin * model.angle_step_rad
            T = pose_from_ppf_local_coordinates(
                model.points_m[mr_id],
                model.normals[mr_id],
                sr,
                nr,
                alpha,
            )
            hypotheses.append(
                PoseHypothesis(
                    transform_model_to_world=T,
                    votes=float(votes),
                    scene_reference_id=int(sr_id),
                    model_reference_id=int(mr_id),
                    alpha_bin=int(alpha_bin),
                )
            )

        if ref_order % max(1, ref_count // 10) == 0 or ref_order == ref_count:
            print(
                f"    voting {ref_order:4d}/{ref_count}: "
                f"hypotheses={len(hypotheses)}"
            )

    hypotheses.sort(key=lambda h: h.votes, reverse=True)
    hypotheses = hypotheses[: args.ppf_max_hypotheses]

    if not hypotheses:
        raise RuntimeError(
            "PPF produced no hypotheses. Try another seed, larger scan, "
            "smaller --ppf-tau-d, or denser PPF scene."
        )
    return hypotheses


def weighted_rotation_mean(rotations, weights):
    rots = Rotation.from_matrix(np.asarray(rotations))
    try:
        return rots.mean(weights=np.asarray(weights)).as_matrix()
    except TypeError:
        q = rots.as_quat()
        q = np.column_stack((q[:, 3], q[:, 0], q[:, 1], q[:, 2]))
        A = np.zeros((4, 4))
        for qi, wi in zip(q, weights):
            if qi[0] < 0:
                qi = -qi
            A += float(wi) * np.outer(qi, qi)
        _, vecs = np.linalg.eigh(A)
        qmean = vecs[:, -1]
        qmean /= np.linalg.norm(qmean)
        return Rotation.from_quat(
            [qmean[1], qmean[2], qmean[3], qmean[0]]
        ).as_matrix()


def representative_from_members(hypotheses, member_ids):
    weights = np.asarray([hypotheses[i].votes for i in member_ids], dtype=float)
    translations = np.asarray(
        [hypotheses[i].transform_model_to_world[:3, 3] for i in member_ids]
    )
    rotations = [
        hypotheses[i].transform_model_to_world[:3, :3] for i in member_ids
    ]

    T = np.eye(4)
    T[:3, 3] = np.average(translations, axis=0, weights=weights)
    T[:3, :3] = weighted_rotation_mean(rotations, weights)
    return T


def cluster_pose_hypotheses(hypotheses, translation_threshold_mm, rotation_threshold_deg):
    t_thresh = translation_threshold_mm / 1000.0
    working = []

    for hyp_id, hyp in enumerate(hypotheses):
        T = hyp.transform_model_to_world
        best_id = None
        best_metric = float("inf")

        for cid, c in enumerate(working):
            Rep = c["representative"]
            dt = float(np.linalg.norm(T[:3, 3] - Rep[:3, 3]))
            dr = rotation_distance_deg(T[:3, :3], Rep[:3, :3])
            if dt <= t_thresh and dr <= rotation_threshold_deg:
                metric = dt/max(t_thresh, EPS) + dr/max(rotation_threshold_deg, EPS)
                if metric < best_metric:
                    best_metric = metric
                    best_id = cid

        if best_id is None:
            working.append({
                "members": [hyp_id],
                "representative": T.copy(),
            })
        else:
            c = working[best_id]
            c["members"].append(hyp_id)
            c["representative"] = representative_from_members(
                hypotheses, c["members"]
            )

    clusters = []
    for item in working:
        members = item["members"]
        rep = representative_from_members(hypotheses, members)
        score = float(sum(hypotheses[i].votes for i in members))

        t_spread = [
            np.linalg.norm(
                hypotheses[i].transform_model_to_world[:3, 3] - rep[:3, 3]
            ) * 1000.0
            for i in members
        ]
        r_spread = [
            rotation_distance_deg(
                hypotheses[i].transform_model_to_world[:3, :3],
                rep[:3, :3],
            )
            for i in members
        ]

        clusters.append(
            PoseCluster(
                rank=0,
                transform_model_to_world=rep,
                score=score,
                member_count=len(members),
                members=members,
                translation_spread_mm=float(max(t_spread, default=0.0)),
                rotation_spread_deg=float(max(r_spread, default=0.0)),
            )
        )

    clusters.sort(key=lambda c: c.score, reverse=True)
    for rank, c in enumerate(clusters, start=1):
        c.rank = rank
    return clusters


def make_icp_target(mesh, args):
    o3d = require_open3d()
    cloud = mesh.sample_points_uniformly(number_of_points=args.cad_sample_points)
    cloud = cloud.voxel_down_sample(args.icp_cad_voxel_mm / 1000.0)
    cloud.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=args.icp_normal_radius_mm / 1000.0,
            max_nn=60,
        )
    )
    return cloud


def refine_top1_icp(scan_points_world, target_cad, top1_model_to_world, args):
    o3d = require_open3d()
    source = make_cloud(scan_points_world)
    init_world_to_model = invert_transform(top1_model_to_world)

    result = o3d.pipelines.registration.registration_icp(
        source,
        target_cad,
        args.icp_distance_mm / 1000.0,
        init_world_to_model,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=80),
    )
    est_model_to_world = invert_transform(
        np.asarray(result.transformation, dtype=np.float64)
    )
    return result, est_model_to_world


CLUSTER_COLORS = [
    (0.10, 0.85, 0.25),
    (0.15, 0.55, 1.00),
    (1.00, 0.60, 0.10),
    (0.75, 0.25, 1.00),
    (0.95, 0.20, 0.45),
    (0.20, 0.85, 0.85),
    (0.85, 0.85, 0.20),
    (0.65, 0.65, 0.65),
]


def draw_stage(name, geometries, note, point_size=4.0):
    print(f"\n[VIEW] {name}")
    print(f"  {note}")
    print("  Close window to continue.")
    o3d = require_open3d()
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=name, width=1440, height=900)
    for g in geometries:
        vis.add_geometry(g)
    opt = vis.get_render_option()
    opt.background_color = np.asarray((0.025, 0.025, 0.03))
    opt.point_size = float(point_size)
    vis.run()
    vis.destroy_window()


def show_debug_views(
    mesh,
    scan,
    scan_points_world,
    gt_model_to_world,
    clusters,
    icp_result,
    icp_model_to_world,
    args,
):
    draw_stage(
        "01 - Random Scan 1",
        [
            make_cloud(scan_points_world, (1.0, 0.12, 0.12)),
            make_path_line(scan.path_origins_m, (0.10, 0.65, 1.0)),
        ],
        f"Red=random Scan 1, blue=sensor path, points={len(scan_points_world)}",
        point_size=4.5,
    )

    base = mesh.sample_points_uniformly(number_of_points=8000)

    geoms = [make_cloud(scan_points_world, (1.0, 0.10, 0.10))]
    topn = min(args.plot_top_clusters, len(clusters))
    for i, c in enumerate(clusters[:topn]):
        cloud = copy.deepcopy(base)
        cloud.transform(c.transform_model_to_world)
        cloud.paint_uniform_color(CLUSTER_COLORS[i % len(CLUSTER_COLORS)])
        geoms.append(cloud)

    draw_stage(
        "02 - PPF Pose Clusters",
        geoms,
        f"Red=scan, colored CADs=Top-{topn} cluster representatives by summed PPF votes.",
        point_size=2.5,
    )

    top1_cloud = copy.deepcopy(base)
    top1_cloud.transform(clusters[0].transform_model_to_world)
    top1_cloud.paint_uniform_color((0.10, 0.85, 0.25))
    draw_stage(
        "03 - PPF Top-1 Before ICP",
        [make_cloud(scan_points_world, (1.0, 0.10, 0.10)), top1_cloud],
        (
            f"Top-1 score={clusters[0].score:.1f}, "
            f"members={clusters[0].member_count}"
        ),
        point_size=3.0,
    )

    gt_cloud = copy.deepcopy(base)
    gt_cloud.transform(gt_model_to_world)
    gt_cloud.paint_uniform_color((0.65, 0.65, 0.65))

    icp_cloud = copy.deepcopy(base)
    icp_cloud.transform(icp_model_to_world)
    icp_cloud.paint_uniform_color((0.10, 0.85, 0.25))

    draw_stage(
        "04 - Top-1 + ICP vs Hidden GT",
        [
            gt_cloud,
            icp_cloud,
            make_cloud(scan_points_world, (1.0, 0.10, 0.10)),
        ],
        (
            f"Gray=GT, green=ICP result, red=scan. "
            f"fitness={float(icp_result.fitness):.3f}, "
            f"RMSE={float(icp_result.inlier_rmse)*1000.0:.3f} mm"
        ),
        point_size=3.0,
    )


def save_cluster_plot(output_dir, clusters, top_n, show=False):
    """Plot how much PPF voting support each pose cluster received.

    y-axis = cluster.score = sum of PPF accumulator-peak votes assigned
    to that SE(3) cluster.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib unavailable -> skip ppf_cluster_votes.png")
        return

    top_n = min(top_n, len(clusters))
    shown = clusters[:top_n]

    ranks = [f"#{c.rank}" for c in shown]
    scores = np.asarray([c.score for c in shown], dtype=np.float64)
    members = np.asarray([c.member_count for c in shown], dtype=np.int64)

    total_score = float(np.sum(scores))
    shares = (
        100.0 * scores / total_score
        if total_score > EPS
        else np.zeros_like(scores)
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(ranks, scores)

    # Each bar: raw summed votes + share among displayed clusters.
    for bar, score, share, member_count in zip(bars, scores, shares, members):
        ax.annotate(
            f"{score:.0f}\n({share:.1f}%, {member_count} poses)",
            xy=(bar.get_x() + bar.get_width() / 2.0, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_xlabel("Pose cluster rank")
    ax.set_ylabel("Summed PPF votes")
    ax.set_title("PPF pose-cluster voting support")
    ax.grid(axis="y", alpha=0.25)

    if len(shown) >= 2:
        ratio = scores[0] / max(scores[1], EPS)
        ax.text(
            0.99,
            0.98,
            f"Top-1 / Top-2 = {ratio:.3f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
        )

    fig.tight_layout()
    out = output_dir / "ppf_cluster_votes.png"
    fig.savefig(out, dpi=180)
    print(f"  saved vote plot: {out}")

    if show:
        plt.show()
    plt.close(fig)


def save_outputs(
    output_dir,
    scan_points_world,
    scan,
    hypotheses,
    clusters,
    gt_model_to_world,
    top1_icp,
    top1_icp_model_to_world,
    args,
    metadata,
):
    o3d = require_open3d()
    output_dir.mkdir(parents=True, exist_ok=True)

    o3d.io.write_point_cloud(
        str(output_dir / "random_scan1_world.ply"),
        make_cloud(scan_points_world, (1.0, 0.1, 0.1)),
    )
    o3d.io.write_line_set(
        str(output_dir / "random_scan1_sensor_path.ply"),
        make_path_line(scan.path_origins_m, (0.1, 0.65, 1.0)),
    )

    with (output_dir / "ppf_pose_hypotheses.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        fields = [
            "hypothesis_id", "votes", "scene_reference_id",
            "model_reference_id", "alpha_bin",
            "tx_mm", "ty_mm", "tz_mm",
            "rx_deg", "ry_deg", "rz_deg",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, h in enumerate(hypotheses):
            T = h.transform_model_to_world
            e = Rotation.from_matrix(T[:3, :3]).as_euler("xyz", degrees=True)
            w.writerow({
                "hypothesis_id": i,
                "votes": h.votes,
                "scene_reference_id": h.scene_reference_id,
                "model_reference_id": h.model_reference_id,
                "alpha_bin": h.alpha_bin,
                "tx_mm": T[0, 3]*1000.0,
                "ty_mm": T[1, 3]*1000.0,
                "tz_mm": T[2, 3]*1000.0,
                "rx_deg": e[0], "ry_deg": e[1], "rz_deg": e[2],
            })

    with (output_dir / "ppf_pose_clusters.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        fields = [
            "rank", "score", "member_count",
            "translation_spread_mm", "rotation_spread_deg",
            "gt_translation_error_mm", "gt_rotation_error_deg",
            "tx_mm", "ty_mm", "tz_mm",
            "rx_deg", "ry_deg", "rz_deg",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for c in clusters:
            T = c.transform_model_to_world
            e = Rotation.from_matrix(T[:3, :3]).as_euler("xyz", degrees=True)
            err = pose_error_model_to_world(T, gt_model_to_world)
            w.writerow({
                "rank": c.rank,
                "score": c.score,
                "member_count": c.member_count,
                "translation_spread_mm": c.translation_spread_mm,
                "rotation_spread_deg": c.rotation_spread_deg,
                "gt_translation_error_mm": err["translation_mm"],
                "gt_rotation_error_deg": err["rotation_deg"],
                "tx_mm": T[0, 3]*1000.0,
                "ty_mm": T[1, 3]*1000.0,
                "tz_mm": T[2, 3]*1000.0,
                "rx_deg": e[0], "ry_deg": e[1], "rz_deg": e[2],
            })

    pre = pose_error_model_to_world(
        clusters[0].transform_model_to_world, gt_model_to_world
    )
    post = pose_error_model_to_world(
        top1_icp_model_to_world, gt_model_to_world
    )

    report = {
        "input": str(args.cad),
        "geometry": metadata,
        "random_scan_1": {
            "points": int(len(scan_points_world)),
            "sensor_viewpoint_m": scan.sensor_viewpoint_m.tolist(),
        },
        "ppf": {
            "hypothesis_count": len(hypotheses),
            "cluster_count": len(clusters),
            "cluster_translation_threshold_mm": args.cluster_translation_mm,
            "cluster_rotation_threshold_deg": args.cluster_rotation_deg,
        },
        "ground_truth": {
            "T_model_to_world": gt_model_to_world.tolist()
        },
        "top_clusters": [
            {
                "rank": c.rank,
                "score": c.score,
                "member_count": c.member_count,
                "translation_spread_mm": c.translation_spread_mm,
                "rotation_spread_deg": c.rotation_spread_deg,
                "T_model_to_world": c.transform_model_to_world.tolist(),
                "gt_error": pose_error_model_to_world(
                    c.transform_model_to_world, gt_model_to_world
                ),
            }
            for c in clusters[:min(20, len(clusters))]
        ],
        "top1_before_icp": {
            "T_model_to_world": clusters[0].transform_model_to_world.tolist(),
            "gt_error": pre,
        },
        "top1_after_icp": {
            "T_model_to_world": top1_icp_model_to_world.tolist(),
            "fitness": float(top1_icp.fitness),
            "inlier_rmse_mm": float(top1_icp.inlier_rmse)*1000.0,
            "gt_error": post,
        },
    }

    with (output_dir / "ppf_pose_cluster_report.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    save_cluster_plot(output_dir, clusters, args.plot_top_clusters, show=args.show)



# ---------------------------------------------------------------------------
# Greedy information-based Scan 2 + Monte Carlo
# ---------------------------------------------------------------------------

def orient_surface_normal_for_raycast(
    scene,
    point_m: np.ndarray,
    normal: np.ndarray,
    standoff_mm: float,
) -> np.ndarray:
    """Pick the normal sign that places the sensor outside the mesh."""
    o3d = require_open3d()
    standoff_m = standoff_mm / 1000.0
    best = normalize(normal)
    best_error = float("inf")

    for sign in (1.0, -1.0):
        n = sign * normalize(normal)
        origin = point_m + standoff_m * n
        ray = np.r_[origin, -n].astype(np.float32)[None, :]
        hit = scene.cast_rays(o3d.core.Tensor(ray))["t_hit"].numpy()[0]
        if np.isfinite(hit):
            error = abs(float(hit) - standoff_m)
            if error < best_error:
                best_error = error
                best = n
    return normalize(best)


def make_surface_pool(mesh, number_of_points: int = 50000) -> tuple[np.ndarray, np.ndarray]:
    cloud = mesh.sample_points_uniformly(number_of_points=number_of_points)
    if not cloud.has_normals():
        cloud.estimate_normals()
    points = np.asarray(cloud.points, dtype=np.float64)
    normals = np.asarray(cloud.normals, dtype=np.float64)
    normals = np.asarray([normalize(n) for n in normals], dtype=np.float64)
    return points, normals


def random_scan_plan_from_surface_pool(
    surface_points: np.ndarray,
    surface_normals: np.ndarray,
    cad_scene,
    args,
    rng: np.random.Generator,
) -> RandomScanPlan:
    idx = int(rng.integers(0, len(surface_points)))
    point = surface_points[idx].copy()
    normal = orient_surface_normal_for_raycast(
        cad_scene,
        point,
        surface_normals[idx],
        args.standoff_mm,
    )
    t1, t2 = tangent_basis(normal)
    angle = (
        float(rng.uniform(0.0, math.pi))
        if args.first_scan_angle_deg is None
        else math.radians(args.first_scan_angle_deg)
    )
    return RandomScanPlan(
        point_cad_m=point,
        normal_cad=normal,
        angle_deg=float(math.degrees(angle)),
        profile_axis_cad=normalize(math.cos(angle) * t1 + math.sin(angle) * t2),
        sweep_axis_cad=normalize(-math.sin(angle) * t1 + math.cos(angle) * t2),
    )


def build_information_candidate_pool(
    surface_points: np.ndarray,
    surface_normals: np.ndarray,
    cad_scene,
    args,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    """Create a fixed CAD-surface candidate pool, independent of the Monte Carlo GT."""
    order = rng.permutation(len(surface_points))
    selected: list[int] = []
    nms_m = args.candidate_nms_mm / 1000.0

    for idx in order:
        p = surface_points[idx]
        if selected:
            chosen = surface_points[np.asarray(selected, dtype=np.int64)]
            if np.any(np.linalg.norm(chosen - p, axis=1) < nms_m):
                continue
        selected.append(int(idx))
        if len(selected) >= args.candidate_count:
            break

    if len(selected) < 2:
        raise RuntimeError(
            "Too few next-scan candidates survived NMS; reduce --candidate-nms-mm"
        )

    candidates: list[dict[str, Any]] = []
    identity = np.eye(4, dtype=np.float64)

    for candidate_id, idx in enumerate(selected, start=1):
        point = surface_points[idx].copy()
        normal = orient_surface_normal_for_raycast(
            cad_scene,
            point,
            surface_normals[idx],
            args.standoff_mm,
        )
        t1, t2 = tangent_basis(normal)
        orientation_options = []

        for angle in np.linspace(
            0.0,
            math.pi,
            args.orientation_samples,
            endpoint=False,
        ):
            plan = RandomScanPlan(
                point_cad_m=point.copy(),
                normal_cad=normal.copy(),
                angle_deg=float(math.degrees(angle)),
                profile_axis_cad=normalize(math.cos(angle) * t1 + math.sin(angle) * t2),
                sweep_axis_cad=normalize(-math.sin(angle) * t1 + math.cos(angle) * t2),
            )
            scan_cad = simulate_scan_from_plan_world(
                cad_scene,
                plan,
                identity,
                args,
            )
            if len(scan_cad.clean_points_m) >= 20:
                orientation_options.append(
                    {
                        "plan": plan,
                        "scan_cad": scan_cad,
                    }
                )

        if orientation_options:
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "point_cad_m": point,
                    "normal_cad": normal,
                    "orientations": orientation_options,
                }
            )

    if len(candidates) < 2:
        raise RuntimeError("Fewer than two valid information candidates remain")
    return candidates


def prior_information(
    rotation_scale_mm: float,
    translation_sigma_mm: float,
    rotation_sigma_deg: float,
) -> np.ndarray:
    rotation_sigma_scaled_mm = rotation_scale_mm * math.radians(rotation_sigma_deg)
    sigmas = np.array(
        [rotation_sigma_scaled_mm] * 3 + [translation_sigma_mm] * 3,
        dtype=np.float64,
    )
    return np.diag(1.0 / np.maximum(sigmas * sigmas, EPS))


def point_to_plane_information(
    scan_points_cad_m: np.ndarray,
    target_points_m: np.ndarray,
    target_normals: np.ndarray,
    target_tree: cKDTree,
    rotation_scale_mm: float,
    residual_sigma_mm: float,
    max_correspondence_mm: float,
) -> tuple[np.ndarray, int]:
    """Local 6-DoF point-to-plane Fisher/Gauss-Newton information."""
    if len(scan_points_cad_m) == 0:
        return np.zeros((6, 6), dtype=np.float64), 0

    distances_m, indices = target_tree.query(
        scan_points_cad_m,
        k=1,
        workers=-1,
    )
    valid = distances_m <= max_correspondence_mm / 1000.0
    if not np.any(valid):
        return np.zeros((6, 6), dtype=np.float64), 0

    p_mm = scan_points_cad_m[valid] * 1000.0
    n = target_normals[indices[valid]]

    J_rot = np.cross(p_mm, n) / max(rotation_scale_mm, EPS)
    J = np.hstack((J_rot, n))

    # Robustly reduce the contribution of points already near the correspondence gate.
    residual_mm = distances_m[valid] * 1000.0
    q = residual_mm / max(max_correspondence_mm, EPS)
    w = np.square(np.clip(1.0 - q * q, 0.0, 1.0))
    Jw = J * np.sqrt(w[:, None])

    H = Jw.T @ Jw / max(residual_sigma_mm ** 2, EPS)
    return H, int(np.count_nonzero(valid))


def logdet_information_gain(base_H: np.ndarray, candidate_H: np.ndarray) -> float:
    s0, ld0 = np.linalg.slogdet(base_H)
    s1, ld1 = np.linalg.slogdet(base_H + candidate_H)
    if s0 <= 0 or s1 <= 0:
        return float("-inf")
    return 0.5 * float(ld1 - ld0)


def prepare_information_target(mesh, args):
    target = make_icp_target(mesh, args)
    points = np.asarray(target.points, dtype=np.float64)
    normals = np.asarray(target.normals, dtype=np.float64)
    normals = np.asarray([normalize(n) for n in normals], dtype=np.float64)
    return target, points, normals, cKDTree(points)


def select_greedy_next_scan(
    candidates: list[dict[str, Any]],
    first_plan: RandomScanPlan,
    first_scan_points_cad_est: np.ndarray,
    target_points: np.ndarray,
    target_normals: np.ndarray,
    target_tree: cKDTree,
    prior_H: np.ndarray,
    rotation_scale_mm: float,
    args,
) -> tuple[dict[str, Any], np.ndarray, int]:
    """Pure greedy D-optimal choice: argmax 0.5 log det increase."""
    first_H, first_corr = point_to_plane_information(
        first_scan_points_cad_est,
        target_points,
        target_normals,
        target_tree,
        rotation_scale_mm,
        max(args.range_sigma_mm, 0.01),
        args.max_correspondence_mm,
    )
    base_H = prior_H + first_H

    best: dict[str, Any] | None = None
    exclude_m = args.candidate_exclude_first_mm / 1000.0

    for candidate in candidates:
        distance_from_first = float(
            np.linalg.norm(candidate["point_cad_m"] - first_plan.point_cad_m)
        )
        if distance_from_first < exclude_m:
            continue

        for option in candidate["orientations"]:
            scan_cad = option["scan_cad"]
            candidate_H, corr = point_to_plane_information(
                scan_cad.clean_points_m,
                target_points,
                target_normals,
                target_tree,
                rotation_scale_mm,
                max(args.range_sigma_mm, 0.01),
                args.max_correspondence_mm,
            )
            gain = logdet_information_gain(base_H, candidate_H)

            if best is None or gain > best["information_gain"]:
                best = {
                    "candidate_id": candidate["candidate_id"],
                    "point_cad_m": candidate["point_cad_m"],
                    "normal_cad": candidate["normal_cad"],
                    "plan": option["plan"],
                    "expected_scan_cad": scan_cad,
                    "information_gain": float(gain),
                    "candidate_correspondences": int(corr),
                    "distance_from_first_mm": distance_from_first * 1000.0,
                }

    if best is None:
        raise RuntimeError("No valid next-scan candidate could be evaluated")

    return best, base_H, first_corr


def refine_cumulative_icp(
    cumulative_world_points: np.ndarray,
    target_cad,
    init_model_to_world: np.ndarray,
    args,
):
    o3d = require_open3d()
    source = make_cloud(cumulative_world_points)
    init_world_to_model = invert_transform(init_model_to_world)
    result = o3d.pipelines.registration.registration_icp(
        source,
        target_cad,
        args.icp_distance_mm / 1000.0,
        init_world_to_model,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=100),
    )
    est_model_to_world = invert_transform(
        np.asarray(result.transformation, dtype=np.float64)
    )
    return result, est_model_to_world


def is_pose_success(error: dict[str, float], args) -> bool:
    return (
        error["translation_mm"] <= args.success_translation_mm
        and error["rotation_deg"] <= args.success_rotation_deg
    )


def plot_mc_summary(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib unavailable -> skipping MC plots")
        return

    valid = [r for r in rows if r.get("status") == "ok"]
    if not valid:
        return

    ids = np.asarray([r["trial"] for r in valid])
    t1 = np.asarray([r["scan1_translation_error_mm"] for r in valid])
    t2 = np.asarray([r["scan2_translation_error_mm"] for r in valid])
    r1 = np.asarray([r["scan1_rotation_error_deg"] for r in valid])
    r2 = np.asarray([r["scan2_rotation_error_deg"] for r in valid])

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(ids, t1, marker="o", label="After random Scan 1")
    ax.plot(ids, t2, marker="o", label="After greedy Scan 2")
    ax.set_xlabel("Monte Carlo trial")
    ax.set_ylabel("Translation error [mm]")
    ax.set_title("Translation error: random Scan 1 vs greedy information Scan 2")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "mc_translation_error.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(ids, r1, marker="o", label="After random Scan 1")
    ax.plot(ids, r2, marker="o", label="After greedy Scan 2")
    ax.set_xlabel("Monte Carlo trial")
    ax.set_ylabel("Rotation error [deg]")
    ax.set_title("Rotation error: random Scan 1 vs greedy information Scan 2")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "mc_rotation_error.png", dpi=180)
    plt.close(fig)

    gain = np.asarray([r["selected_information_gain"] for r in valid])
    improvement = t1 - t2
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.scatter(gain, improvement)
    ax.axhline(0.0, linewidth=1.0)
    ax.set_xlabel("Predicted information gain [nats]")
    ax.set_ylabel("Translation-error improvement [mm]")
    ax.set_title("Predicted information gain vs realized improvement")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_dir / "mc_gain_vs_improvement.png", dpi=180)
    plt.close(fig)


def summarize_mc(rows: list[dict[str, Any]], args) -> dict[str, Any]:
    total = len(rows)
    valid = [r for r in rows if r.get("status") == "ok"]

    summary: dict[str, Any] = {
        "requested_trials": total,
        "valid_trials": len(valid),
        "failed_trials": total - len(valid),
        "success_threshold": {
            "translation_mm": args.success_translation_mm,
            "rotation_deg": args.success_rotation_deg,
        },
    }
    if not valid:
        return summary

    def arr(key):
        return np.asarray([float(r[key]) for r in valid], dtype=np.float64)

    t1 = arr("scan1_translation_error_mm")
    r1 = arr("scan1_rotation_error_deg")
    t2 = arr("scan2_translation_error_mm")
    r2 = arr("scan2_rotation_error_deg")
    gain = arr("selected_information_gain")
    ratio = arr("ppf_top1_top2_ratio")

    s1 = (
        (t1 <= args.success_translation_mm)
        & (r1 <= args.success_rotation_deg)
    )
    s2 = (
        (t2 <= args.success_translation_mm)
        & (r2 <= args.success_rotation_deg)
    )

    summary.update({
        "after_random_scan1": {
            "success_rate": float(np.mean(s1)),
            "translation_median_mm": float(np.median(t1)),
            "translation_p95_mm": float(np.percentile(t1, 95)),
            "rotation_median_deg": float(np.median(r1)),
            "rotation_p95_deg": float(np.percentile(r1, 95)),
        },
        "after_greedy_scan2": {
            "success_rate": float(np.mean(s2)),
            "translation_median_mm": float(np.median(t2)),
            "translation_p95_mm": float(np.percentile(t2, 95)),
            "rotation_median_deg": float(np.median(r2)),
            "rotation_p95_deg": float(np.percentile(r2, 95)),
        },
        "improvement": {
            "translation_median_mm": float(np.median(t1 - t2)),
            "rotation_median_deg": float(np.median(r1 - r2)),
            "fraction_translation_improved": float(np.mean(t2 < t1)),
            "fraction_rotation_improved": float(np.mean(r2 < r1)),
        },
        "selection": {
            "information_gain_median": float(np.median(gain)),
            "ppf_top1_top2_ratio_median": float(np.median(ratio)),
        },
    })
    return summary


def show_active_trial(
    mesh,
    scan1_points_world,
    scan2_points_world,
    gt_model_to_world,
    scan1_est_model_to_world,
    scan2_est_model_to_world,
    selected,
):
    base = mesh.sample_points_uniformly(number_of_points=8000)

    expected = copy.deepcopy(base)
    expected.transform(scan1_est_model_to_world)
    expected.paint_uniform_color((0.20, 0.55, 1.0))

    draw_stage(
        "MC debug - selected greedy Scan 2",
        [
            expected,
            make_cloud(scan1_points_world, (1.0, 0.10, 0.10)),
            make_path_line(
                transform_points(
                    selected["expected_scan_cad"].path_origins_m,
                    scan1_est_model_to_world,
                ),
                (0.80, 0.15, 1.0),
            ),
        ],
        (
            f"Blue=Scan-1 estimated object pose, red=Scan 1, purple=selected Scan-2 path. "
            f"candidate={selected['candidate_id']}, "
            f"predicted gain={selected['information_gain']:.3f}"
        ),
        point_size=3.0,
    )

    gt_cloud = copy.deepcopy(base)
    gt_cloud.transform(gt_model_to_world)
    gt_cloud.paint_uniform_color((0.65, 0.65, 0.65))
    final_cloud = copy.deepcopy(base)
    final_cloud.transform(scan2_est_model_to_world)
    final_cloud.paint_uniform_color((0.10, 0.85, 0.25))

    draw_stage(
        "MC debug - after greedy Scan 2",
        [
            gt_cloud,
            final_cloud,
            make_cloud(scan1_points_world, (1.0, 0.10, 0.10)),
            make_cloud(scan2_points_world, (1.0, 0.65, 0.0)),
        ],
        "Gray=GT, green=final estimate, red=Scan 1, orange=Scan 2.",
        point_size=3.0,
    )


def main() -> int:
    args = parse_args()
    validate_args(args)

    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else Path(f"{args.cad.stem}_ppf_greedy_info_mc")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    setup_rng = np.random.default_rng(args.seed)

    print("[setup 1/5] Load and center CAD")
    mesh, metadata = load_centered_mesh(args.cad, args.mesh_unit)
    diameter_m = metadata["diameter_mm"] / 1000.0
    rotation_scale_mm = max(0.5 * metadata["diameter_mm"], 10.0)
    cad_scene = build_raycast_scene(mesh)

    print("[setup 2/5] Build PPF model hash once")
    model_points, model_normals = sample_oriented_cad(
        mesh,
        args.cad_sample_points,
        args.ppf_model_points,
    )
    model = build_ppf_model(
        model_points,
        model_normals,
        diameter_m,
        args,
        setup_rng,
    )
    print(f"  PPF hash bins={len(model.hash_table):,}")

    print("[setup 3/5] Build reusable CAD surface pool")
    surface_points, surface_normals = make_surface_pool(mesh, number_of_points=50000)

    print("[setup 4/5] Build fixed Scan-2 candidate pool")
    candidates = build_information_candidate_pool(
        surface_points,
        surface_normals,
        cad_scene,
        args,
        np.random.default_rng(args.seed + 500),
    )
    print(
        f"  valid candidates={len(candidates)}, "
        f"orientations/candidate<= {args.orientation_samples}"
    )

    print("[setup 5/5] Build ICP/information target")
    target_cloud, target_points, target_normals, target_tree = prepare_information_target(
        mesh,
        args,
    )
    prior_H = prior_information(
        rotation_scale_mm,
        args.prior_translation_mm,
        args.prior_rotation_deg,
    )

    rows: list[dict[str, Any]] = []
    first_debug_saved = False

    print(f"\n[Monte Carlo] trials={args.mc_trials}")
    for trial in range(args.mc_trials):
        trial_seed = args.seed + 10000 + trial
        rng = np.random.default_rng(trial_seed)
        row: dict[str, Any] = {
            "trial": trial,
            "seed": trial_seed,
            "status": "failed",
            "failure_reason": "",
        }

        print(f"\n=== Trial {trial + 1}/{args.mc_trials} | seed={trial_seed} ===")

        try:
            # A) New hidden object pose for this trial.
            gt_model_to_world = random_gt_transform(args, rng)
            world_mesh = copy.deepcopy(mesh)
            world_mesh.transform(gt_model_to_world)
            world_scene = build_raycast_scene(world_mesh)

            # B) Scan 1: completely random CAD surface location + random/fixed direction.
            first_plan = random_scan_plan_from_surface_pool(
                surface_points,
                surface_normals,
                cad_scene,
                args,
                rng,
            )
            first_scan = simulate_scan_from_plan_world(
                world_scene,
                first_plan,
                gt_model_to_world,
                args,
            )
            first_points_world = noisy_scan_points(first_scan, args, rng)
            if len(first_points_world) < 30:
                raise RuntimeError(
                    f"Scan 1 produced only {len(first_points_world)} points"
                )

            # C) PPF -> SE(3) clusters -> Top-1 -> ICP.
            _, scene_points, scene_normals = preprocess_scene_for_ppf(
                first_points_world,
                first_scan.sensor_viewpoint_m,
                args,
            )
            hypotheses = ppf_vote_pose_hypotheses(
                model,
                scene_points,
                scene_normals,
                args,
                rng,
            )
            clusters = cluster_pose_hypotheses(
                hypotheses,
                args.cluster_translation_mm,
                args.cluster_rotation_deg,
            )
            if not clusters:
                raise RuntimeError("PPF returned no pose clusters")

            top1_top2 = (
                clusters[0].score / max(clusters[1].score, EPS)
                if len(clusters) >= 2
                else float(clusters[0].score)
            )

            ppf_top1_error = pose_error_model_to_world(
                clusters[0].transform_model_to_world,
                gt_model_to_world,
            )

            first_icp, first_est_model_to_world = refine_top1_icp(
                first_points_world,
                target_cloud,
                clusters[0].transform_model_to_world,
                args,
            )
            first_error = pose_error_model_to_world(
                first_est_model_to_world,
                gt_model_to_world,
            )

            # D) Convert Scan 1 using the *estimated* Top-1+ICP pose.
            first_world_to_cad_est = invert_transform(first_est_model_to_world)
            first_points_cad_est = transform_points(
                first_points_world,
                first_world_to_cad_est,
            )

            # E) Pure greedy D-optimal next-scan selection.
            selected, base_H, first_corr = select_greedy_next_scan(
                candidates,
                first_plan,
                first_points_cad_est,
                target_points,
                target_normals,
                target_tree,
                prior_H,
                rotation_scale_mm,
                args,
            )

            # F) Aim selected CAD scan using T_hat_1, but raycast the hidden-GT world.
            second_scan = simulate_scan_from_plan_world(
                world_scene,
                selected["plan"],
                first_est_model_to_world,
                args,
            )
            second_points_world = noisy_scan_points(second_scan, args, rng)
            if len(second_points_world) < 20:
                raise RuntimeError(
                    f"Greedy Scan 2 hit only {len(second_points_world)} points "
                    "(aiming failure under T_hat_1)"
                )

            # G) Scan 1 + Scan 2 cumulative ICP, initialized by T_hat_1.
            cumulative_world = np.vstack(
                (first_points_world, second_points_world)
            )
            second_icp, second_est_model_to_world = refine_cumulative_icp(
                cumulative_world,
                target_cloud,
                first_est_model_to_world,
                args,
            )
            second_error = pose_error_model_to_world(
                second_est_model_to_world,
                gt_model_to_world,
            )

            row.update({
                "status": "ok",
                "scan1_points": len(first_points_world),
                "ppf_hypotheses": len(hypotheses),
                "ppf_clusters": len(clusters),
                "ppf_top1_score": clusters[0].score,
                "ppf_top2_score": clusters[1].score if len(clusters) >= 2 else 0.0,
                "ppf_top1_top2_ratio": top1_top2,
                "ppf_top1_translation_error_mm": ppf_top1_error["translation_mm"],
                "ppf_top1_rotation_error_deg": ppf_top1_error["rotation_deg"],
                "scan1_translation_error_mm": first_error["translation_mm"],
                "scan1_rotation_error_deg": first_error["rotation_deg"],
                "scan1_success": int(is_pose_success(first_error, args)),
                "scan1_icp_fitness": float(first_icp.fitness),
                "scan1_icp_rmse_mm": float(first_icp.inlier_rmse) * 1000.0,
                "scan1_information_correspondences": int(first_corr),
                "selected_candidate_id": selected["candidate_id"],
                "selected_scan_angle_deg": selected["plan"].angle_deg,
                "selected_information_gain": selected["information_gain"],
                "selected_candidate_correspondences": selected["candidate_correspondences"],
                "selected_distance_from_first_mm": selected["distance_from_first_mm"],
                "scan2_points": len(second_points_world),
                "scan2_translation_error_mm": second_error["translation_mm"],
                "scan2_rotation_error_deg": second_error["rotation_deg"],
                "scan2_success": int(is_pose_success(second_error, args)),
                "scan2_icp_fitness": float(second_icp.fitness),
                "scan2_icp_rmse_mm": float(second_icp.inlier_rmse) * 1000.0,
                "translation_improvement_mm": (
                    first_error["translation_mm"] - second_error["translation_mm"]
                ),
                "rotation_improvement_deg": (
                    first_error["rotation_deg"] - second_error["rotation_deg"]
                ),
            })

            print(
                f"  PPF Top1/Top2={top1_top2:.3f} | "
                f"Top1 PPF err={ppf_top1_error['translation_mm']:.2f} mm/"
                f"{ppf_top1_error['rotation_deg']:.2f} deg"
            )
            print(
                f"  Scan1+ICP      ={first_error['translation_mm']:.3f} mm/"
                f"{first_error['rotation_deg']:.3f} deg"
            )
            print(
                f"  Greedy Scan2   =candidate #{selected['candidate_id']}, "
                f"angle={selected['plan'].angle_deg:.1f} deg, "
                f"gain={selected['information_gain']:.3f}"
            )
            print(
                f"  Scan1+2 ICP    ={second_error['translation_mm']:.3f} mm/"
                f"{second_error['rotation_deg']:.3f} deg"
            )

            if args.save_first_trial_debug and not first_debug_saved:
                debug_dir = output_dir / "first_successful_trial"
                debug_dir.mkdir(parents=True, exist_ok=True)
                save_cluster_plot(
                    debug_dir,
                    clusters,
                    args.plot_top_clusters,
                    show=False,
                )
                require_open3d().io.write_point_cloud(
                    str(debug_dir / "scan1_world.ply"),
                    make_cloud(first_points_world, (1.0, 0.1, 0.1)),
                )
                require_open3d().io.write_point_cloud(
                    str(debug_dir / "scan2_world.ply"),
                    make_cloud(second_points_world, (1.0, 0.65, 0.0)),
                )
                first_debug_saved = True

            if args.show and not first_debug_saved:
                show_active_trial(
                    mesh,
                    first_points_world,
                    second_points_world,
                    gt_model_to_world,
                    first_est_model_to_world,
                    second_est_model_to_world,
                    selected,
                )
                first_debug_saved = True

        except (RuntimeError, ValueError, np.linalg.LinAlgError) as exc:
            row["failure_reason"] = str(exc)
            print(f"  FAILED: {exc}")

        rows.append(row)

    # Save one row per MC trial.
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with (output_dir / "mc_trials.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = summarize_mc(rows, args)
    summary["input"] = str(args.cad)
    summary["geometry"] = metadata
    summary["greedy_policy"] = {
        "criterion": "D-optimal information gain only",
        "formula": "0.5 * [logdet(H_base + H_candidate) - logdet(H_base)]",
        "candidate_count": len(candidates),
        "orientation_samples": args.orientation_samples,
        "note": (
            "Scan 2 is aimed using Top-1 PPF + ICP pose estimate; GT is used only "
            "to generate the hidden world and evaluate error."
        ),
    }
    with (output_dir / "mc_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    plot_mc_summary(output_dir, rows)

    print("\n" + "=" * 90)
    print("MONTE-CARLO SUMMARY")
    print("=" * 90)
    print(
        f"valid={summary['valid_trials']}/{summary['requested_trials']}, "
        f"failed={summary['failed_trials']}"
    )

    if summary["valid_trials"]:
        a = summary["after_random_scan1"]
        b = summary["after_greedy_scan2"]
        imp = summary["improvement"]
        print(
            f"Random Scan1 : success={100*a['success_rate']:.1f}% | "
            f"T med/P95={a['translation_median_mm']:.3f}/"
            f"{a['translation_p95_mm']:.3f} mm | "
            f"R med/P95={a['rotation_median_deg']:.3f}/"
            f"{a['rotation_p95_deg']:.3f} deg"
        )
        print(
            f"+ Greedy Scan2: success={100*b['success_rate']:.1f}% | "
            f"T med/P95={b['translation_median_mm']:.3f}/"
            f"{b['translation_p95_mm']:.3f} mm | "
            f"R med/P95={b['rotation_median_deg']:.3f}/"
            f"{b['rotation_p95_deg']:.3f} deg"
        )
        print(
            f"Median improvement: "
            f"T={imp['translation_median_mm']:+.3f} mm, "
            f"R={imp['rotation_median_deg']:+.3f} deg"
        )

    print(f"outputs: {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)