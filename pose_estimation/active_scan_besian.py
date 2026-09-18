"""

Example:
python /home/choisuhyun/lvs_HandEyeCalibration/pose_estimation/active_scan_besian.py \
    /home/choisuhyun/lvs_HandEyeCalibration/pose_estimation/미그럼틀.stl \
    --mesh-unit mm \
    --scan-length-mm 10 \
    --show
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

try:
    import pclpybridge as pclb
except ImportError as exc:
    raise RuntimeError(
        "pclpybridge is required. Install it in the same Python environment with "
        "`cd /home/choisuhyun/lvs_HandEyeCalibration/pclpybridge && "
        "python -m pip install --no-cache-dir .`"
    ) from exc

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
        description=(
            "Random 2-D laser scanning: PPF pose proposals + sensor-agnostic Bayesian "
            "SE(3) Gaussian-mixture posterior. PPF votes generate pose proposals only; "
            "all scan locations/directions are sampled randomly from the CAD surface."
        )
    )
    p.add_argument("cad", type=Path)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--mesh-unit", choices=("auto", "m", "mm"), default="auto")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument(
        "--show",
        action="store_true",
        help="Show Open3D stages and matplotlib diagnostic plots interactively.",
    )

    g = p.add_argument_group("PPF model preprocessing")
    g.add_argument(
        "--cad-sample-points",
        type=positive_int,
        default=120000,
        help=(
            "Dense mesh samples used only before Drost-style spatial resampling. "
            "The final PPF model size is determined by d_dist=tau_d*diameter."
        ),
    )
    g.add_argument(
        "--ppf-tau-d",
        type=positive_float,
        default=0.05,
        help="Drost sampling rate: d_dist = tau_d * model diameter (paper default 0.05).",
    )
    g.add_argument(
        "--ppf-nangle",
        type=positive_int,
        default=30,
        help="Drost angular sampling count; d_angle=2*pi/nangle (paper default 30).",
    )
    g.add_argument(
        "--ppf-normal-radius-factor",
        type=positive_float,
        default=2.0,
        help=(
            "Plane-fit normal neighborhood radius divided by d_dist. "
            "The paper requires normals to be recomputed after resampling but "
            "does not specify a numeric radius; 2.0 is the implementation choice here."
        ),
    )
    g.add_argument(
        "--ppf-normal-max-nn",
        type=positive_int,
        default=30,
        help="Maximum neighbors for the post-resampling plane fit.",
    )

    g = p.add_argument_group("Virtual 2-D laser scan")
    # Keep the latest uploaded defaults unchanged.
    g.add_argument(
        "--scan-length-mm",
        type=positive_float,
        default=30.0,
        help="Linear sweep length of EVERY random scan in mm.",
    )
    g.add_argument("--scan-step-mm", type=positive_float, default=1.0)
    g.add_argument("--profile-width-mm", type=positive_float, default=32.0)
    g.add_argument("--profile-points", type=positive_int, default=321)
    g.add_argument("--standoff-mm", type=positive_float, default=80.0)
    g.add_argument("--first-scan-angle-deg", type=float, default=None)
    g.add_argument("--range-sigma-mm", type=positive_float, default=0.20)
    g.add_argument("--lateral-sigma-mm", type=positive_float, default=0.08)
    g.add_argument("--dropout-rate", type=probability, default=0.03)
    g.add_argument("--outlier-rate", type=probability, default=0.01)
    g.add_argument("--outlier-sigma-mm", type=positive_float, default=3.0)
    g.add_argument(
        "--random-scan-max-attempts",
        type=positive_int,
        default=100,
        help=(
            "Maximum random re-draws when a sampled scan is too close to a previous "
            "scan or produces fewer than --min-new-scan-points."
        ),
    )

    g = p.add_argument_group("Hidden GT (simulation only)")
    g.add_argument(
        "--gt-generation",
        choices=("prior-gaussian", "fixed-magnitude"),
        default="prior-gaussian",
        help=(
            "How hidden GT is generated. 'prior-gaussian' samples GT around the supplied "
            "initial pose using the same SE(3) Gaussian prior sigmas. 'fixed-magnitude' "
            "uses --gt-translation-mm/--gt-rotation-deg around the supplied initial pose."
        ),
    )
    g.add_argument("--gt-translation-mm", type=float, default=30.0)
    g.add_argument("--gt-rotation-deg", type=float, default=5.0)

    g = p.add_argument_group("PCL PPF registration")
    g.add_argument(
        "--ppf-reference-fraction",
        type=float,
        default=0.20,
        help=(
            "Scene reference fraction. It is converted to PCL's "
            "scene_reference_rate ~= 1/fraction. Default 0.20 -> rate 5."
        ),
    )
    g.add_argument(
        "--ppf-max-candidates",
        type=positive_int,
        default=50,
        help="Maximum clustered PCL PPF pose candidates returned to Python.",
    )
    g.add_argument(
        "--ppf-min-sampled-scene-points",
        type=positive_int,
        default=6,
        help=(
            "Hard sanity minimum after Drost spatial resampling. If a very partial 2-D scan "
            "falls below this, reduce --ppf-tau-d (e.g. 0.025)."
        ),
    )

    g = p.add_argument_group("PPF proposal clustering")
    g.add_argument("--cluster-translation-mm", type=positive_float, default=10.0)
    g.add_argument("--cluster-rotation-deg", type=positive_float, default=10.0)
    g.add_argument("--plot-top-clusters", type=positive_int, default=20)
    g.add_argument("--visualize-max-hypotheses", type=positive_int, default=8)

    g = p.add_argument_group("Bayesian pose posterior (sensor-agnostic geometry)")
    g.add_argument(
        "--prior-pose-json",
        type=Path,
        default=None,
        help=(
            "Optional JSON file containing a 4x4 initial T_model_to_world, either as "
            "a bare 4x4 array or under key 'T_model_to_world'. In simulation, the "
            "default prior mean is identity, i.e. the nominal initial object pose."
        ),
    )
    g.add_argument("--prior-translation-sigma-mm", type=positive_float, default=30.0)
    g.add_argument("--prior-rotation-sigma-deg", type=positive_float, default=5.0)
    g.add_argument(
        "--prior-gate-mahalanobis",
        type=positive_float,
        default=4.48,
        help=(
            "Hard SE(3) prior gate on PPF/refined poses. 4.48 is approximately the "
            "99.73%% chi-square radius for 6 DoF. Candidates outside this ellipsoid "
            "are rejected instead of being allowed to overpower the prior via geometry."
        ),
    )
    g.add_argument("--posterior-max-hypotheses", type=positive_int, default=20)
    g.add_argument(
        "--posterior-plausible-relative",
        type=float,
        default=0.20,
        help="Keep hypotheses whose posterior weight is at least this fraction of MAP.",
    )
    g.add_argument(
        "--posterior-stop-runnerup-relative",
        type=float,
        default=0.05,
        help="Stop when posterior runner-up / MAP weight falls below this value.",
    )
    g.add_argument(
        "--geom-sigma-normal-mm",
        type=positive_float,
        default=1.0,
        help="Gaussian surface thickness along the CAD normal; geometric-model scale.",
    )
    g.add_argument(
        "--geom-sigma-tangent-mm",
        type=positive_float,
        default=3.0,
        help="Gaussian scale in the CAD tangent plane; geometric-model scale.",
    )
    g.add_argument(
        "--geom-object-prior",
        type=float,
        default=0.80,
        help="Mixture prior P(point came from target object), in (0,1).",
    )
    g.add_argument(
        "--geom-outlier-halfwidth-mm",
        type=positive_float,
        default=100.0,
        help="Half-width of the 3-D uniform background/outlier component.",
    )
    g.add_argument(
        "--geom-max-points",
        type=positive_int,
        default=3000,
        help="Deterministic cap on cumulative scan points used for likelihood scoring.",
    )
    g.add_argument(
        "--geom-observation-spacing-mm",
        type=positive_float,
        default=2.0,
        help=(
            "Minimum spatial spacing for geometric-likelihood observations. "
            "This reduces overconfidence from densely correlated points."
        ),
    )
    g.add_argument(
        "--posterior-local-refine-iterations",
        type=positive_int,
        default=8,
        help="Robust Gauss-Newton iterations used to refine each PPF proposal locally.",
    )
    g.add_argument(
        "--posterior-local-refine-max-translation-step-mm",
        type=positive_float,
        default=5.0,
        help="Maximum translation norm of one local SE(3) refinement step.",
    )
    g.add_argument(
        "--posterior-local-refine-max-rotation-step-deg",
        type=positive_float,
        default=2.0,
        help="Maximum rotation norm of one local SE(3) refinement step.",
    )
    g.add_argument(
        "--posterior-stop-translation-std-mm",
        type=positive_float,
        default=1.0,
        help="Require every MAP local translation std axis to be below this to stop.",
    )
    g.add_argument(
        "--posterior-stop-rotation-std-deg",
        type=positive_float,
        default=1.0,
        help="Require every MAP local rotation std axis to be below this to stop.",
    )

    g = p.add_argument_group("CAD FPFH")
    g.add_argument("--fpfh-voxel-mm", type=positive_float, default=2.0)
    g.add_argument("--fpfh-normal-radius-mm", type=positive_float, default=4.0)
    g.add_argument("--fpfh-feature-radius-mm", type=positive_float, default=10.0)
    g.add_argument("--fpfh-normal-max-nn", type=positive_int, default=30)
    g.add_argument("--fpfh-feature-max-nn", type=positive_int, default=100)
    g.add_argument(
        "--fpfh-distance-metric",
        choices=("open3d_l2", "hist_intersection"),
        default="open3d_l2",
    )
    g.add_argument("--fpfh-patch-depth-mm", type=positive_float, default=12.0)
    g.add_argument("--fpfh-min-patch-points", type=positive_int, default=3)
    g.add_argument("--fpfh-max-incidence-deg", type=positive_float, default=75.0)

    g = p.add_argument_group("Offline global-unique scan candidates")
    g.add_argument("--candidate-count", type=positive_int, default=24)
    g.add_argument("--candidate-nms-mm", type=positive_float, default=15.0)
    g.add_argument("--orientation-samples", type=positive_int, default=4)
    g.add_argument(
        "--global-uniqueness-knn",
        type=positive_int,
        default=10,
        help="Number of globally separated patch neighbors used in uniqueness.",
    )
    g.add_argument(
        "--global-uniqueness-exclude-mm",
        type=positive_float,
        default=15.0,
        help="Do not compare a patch to spatially nearby patches.",
    )
    g.add_argument(
        "--rescan-exclusion-mm",
        type=float,
        default=10.0,
        help="Avoid accepted scan centers closer than this CAD distance.",
    )

    g = p.add_argument_group("Iterative active scan")
    g.add_argument("--max-scans", type=positive_int, default=5)
    g.add_argument(
        "--min-new-scan-points",
        type=positive_int,
        default=20,
        help=(
            "Minimum raw points used only as a hard sanity check. "
            "There is no incremental-PPF ACCEPT/REJECT fallback."
        ),
    )

    g = p.add_argument_group("Final ICP")
    g.add_argument("--icp-cad-voxel-mm", type=positive_float, default=1.0)
    g.add_argument("--icp-normal-radius-mm", type=positive_float, default=3.0)
    g.add_argument("--icp-distance-mm", type=positive_float, default=3.0)

    return p.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.cad.is_file():
        raise ValueError(f"CAD file does not exist: {args.cad}")
    if not 0.0 < args.ppf_reference_fraction <= 1.0:
        raise ValueError("--ppf-reference-fraction must be in (0,1]")
    if args.dropout_rate + args.outlier_rate >= 1.0:
        raise ValueError("dropout + outlier rate must be < 1")
    if args.gt_translation_mm < 0 or args.gt_rotation_deg < 0:
        raise ValueError("GT magnitudes must be non-negative")
    if args.prior_pose_json is not None and not args.prior_pose_json.is_file():
        raise ValueError(f"--prior-pose-json does not exist: {args.prior_pose_json}")
    if not 0.0 < args.posterior_plausible_relative <= 1.0:
        raise ValueError("--posterior-plausible-relative must be in (0,1]")
    if not 0.0 <= args.posterior_stop_runnerup_relative < 1.0:
        raise ValueError("--posterior-stop-runnerup-relative must be in [0,1)")
    if args.posterior_stop_runnerup_relative >= args.posterior_plausible_relative:
        raise ValueError(
            "--posterior-stop-runnerup-relative should be smaller than "
            "--posterior-plausible-relative."
        )
    if not 0.0 < args.geom_object_prior < 1.0:
        raise ValueError("--geom-object-prior must be in (0,1)")
    if not 0.0 < args.fpfh_max_incidence_deg < 90.0:
        raise ValueError("--fpfh-max-incidence-deg must be in (0,90)")
    if args.rescan_exclusion_mm < 0:
        raise ValueError("--rescan-exclusion-mm must be non-negative")

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


def skew(v: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(v, dtype=np.float64)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def se3_log_vector(T: np.ndarray) -> np.ndarray:
    """Return xi=[rho, phi] with T=Exp(xi^) using the SE(3) logarithm."""
    T = np.asarray(T, dtype=np.float64)
    phi = Rotation.from_matrix(T[:3, :3]).as_rotvec()
    theta = float(np.linalg.norm(phi))
    Omega = skew(phi)
    if theta < 1.0e-8:
        V = np.eye(3) + 0.5 * Omega + (1.0 / 6.0) * (Omega @ Omega)
    else:
        theta2 = theta * theta
        V = (
            np.eye(3)
            + ((1.0 - math.cos(theta)) / theta2) * Omega
            + ((theta - math.sin(theta)) / (theta2 * theta)) * (Omega @ Omega)
        )
    rho = np.linalg.solve(V, T[:3, 3])
    return np.r_[rho, phi]


def se3_exp_vector(xi: np.ndarray) -> np.ndarray:
    """Return T=Exp(xi^) for xi=[rho, phi]."""
    xi = np.asarray(xi, dtype=np.float64).reshape(6)
    rho = xi[:3]
    phi = xi[3:]
    theta = float(np.linalg.norm(phi))
    Omega = skew(phi)
    if theta < 1.0e-8:
        Rm = np.eye(3) + Omega + 0.5 * (Omega @ Omega)
        V = np.eye(3) + 0.5 * Omega + (1.0 / 6.0) * (Omega @ Omega)
    else:
        theta2 = theta * theta
        Rm = Rotation.from_rotvec(phi).as_matrix()
        V = (
            np.eye(3)
            + ((1.0 - math.cos(theta)) / theta2) * Omega
            + ((theta - math.sin(theta)) / (theta2 * theta)) * (Omega @ Omega)
        )
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rm
    T[:3, 3] = V @ rho
    return T


def load_prior_pose(args) -> np.ndarray:
    """Load the externally supplied nominal object pose; identity is the simulation default."""
    if args.prior_pose_json is None:
        return np.eye(4, dtype=np.float64)
    with args.prior_pose_json.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    raw = obj.get("T_model_to_world") if isinstance(obj, dict) else obj
    T = np.asarray(raw, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError("--prior-pose-json must contain a 4x4 T_model_to_world")
    if not np.all(np.isfinite(T)):
        raise ValueError("--prior-pose-json contains non-finite values")
    if not np.allclose(T[3], np.array([0.0, 0.0, 0.0, 1.0]), atol=1e-8):
        raise ValueError("--prior-pose-json last row must be [0,0,0,1]")
    return T


def pose_prior_log_score(T: np.ndarray, prior_T: np.ndarray, args) -> tuple[float, float, np.ndarray]:
    """Unnormalized log N_SE3(T; prior_T, Sigma0) and its Mahalanobis distance."""
    xi = se3_log_vector(invert_transform(prior_T) @ T)
    sigma_t = args.prior_translation_sigma_mm / 1000.0
    sigma_r = math.radians(args.prior_rotation_sigma_deg)
    inv_var = np.array(
        [1.0 / sigma_t**2] * 3 + [1.0 / sigma_r**2] * 3,
        dtype=np.float64,
    )
    mahal2 = float(np.sum(xi * xi * inv_var))
    return -0.5 * mahal2, math.sqrt(max(mahal2, 0.0)), xi

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

def deterministic_min_distance_sample_indices(
    points: np.ndarray,
    min_distance_m: float,
) -> np.ndarray:
    """Deterministically keep a subset with pairwise spacing >= min_distance_m.

    Drost et al. state that model and scene are subsampled so all retained
    points have a minimum distance d_dist.  The paper does not prescribe the
    exact data structure/order used to realize that constraint.  This function
    uses a lexicographically ordered greedy spatial-hash implementation, so the
    result is deterministic and the spacing condition is explicit.
    """
    pts = np.asarray(points, dtype=np.float64)
    if len(pts) == 0:
        return np.empty(0, dtype=np.int64)
    if min_distance_m <= 0.0:
        return np.arange(len(pts), dtype=np.int64)

    order = np.lexsort((pts[:, 2], pts[:, 1], pts[:, 0]))
    cell = float(min_distance_m)
    inv_cell = 1.0 / cell
    min_d2 = cell * cell
    grid: dict[tuple[int, int, int], list[int]] = {}
    selected: list[int] = []

    for idx in order:
        idx = int(idx)
        p = pts[idx]
        key_arr = np.floor(p * inv_cell).astype(np.int64)
        key = (int(key_arr[0]), int(key_arr[1]), int(key_arr[2]))

        too_close = False
        for dx in (-1, 0, 1):
            if too_close:
                break
            for dy in (-1, 0, 1):
                if too_close:
                    break
                for dz in (-1, 0, 1):
                    for j in grid.get((key[0] + dx, key[1] + dy, key[2] + dz), ()):
                        d = p - pts[j]
                        if float(np.dot(d, d)) < min_d2 - EPS:
                            too_close = True
                            break
                    if too_close:
                        break

        if too_close:
            continue
        selected.append(idx)
        grid.setdefault(key, []).append(idx)

    return np.asarray(selected, dtype=np.int64)


def estimate_normals_after_ppf_resampling(
    points: np.ndarray,
    d_dist_m: float,
    args,
) -> tuple[Any, np.ndarray]:
    """Recalculate normals after PPF resampling by local plane fitting.

    This follows the Drost preprocessing order.  Open3D's normal estimator uses
    a covariance/plane fit in the local neighborhood.  The paper does not give
    the neighborhood radius, so this implementation uses
    --ppf-normal-radius-factor * d_dist.
    """
    o3d = require_open3d()
    cloud = make_cloud(points)
    radius = max(args.ppf_normal_radius_factor * d_dist_m, d_dist_m * 1.01)
    cloud.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius,
            max_nn=args.ppf_normal_max_nn,
        )
    )
    normals = np.asarray(cloud.normals, dtype=np.float64)
    if len(normals) != len(points):
        raise RuntimeError("PPF normal estimation returned an unexpected size")
    normals = np.asarray([normalize(n) for n in normals], dtype=np.float64)
    return cloud, normals


def sample_oriented_cad_drost(mesh, sample_points: int, diameter_m: float, args):
    """Drost-style model preprocessing.

    1) d_dist = tau_d * diam(M)
    2) spatially subsample so retained model points are >= d_dist apart
    3) recalculate normals at that sampling level
    4) orient the recalculated normal to agree with the mesh-sampled normal
    """
    d_dist_m = args.ppf_tau_d * diameter_m
    dense = mesh.sample_points_uniformly(number_of_points=sample_points)
    if not dense.has_normals():
        dense.estimate_normals()

    dense_points = np.asarray(dense.points, dtype=np.float64)
    dense_normals = np.asarray(dense.normals, dtype=np.float64)
    if len(dense_points) == 0:
        raise RuntimeError("Cannot build PPF model from an empty CAD sample")

    selected_ids = deterministic_min_distance_sample_indices(dense_points, d_dist_m)
    points = dense_points[selected_ids]
    seed_normals = dense_normals[selected_ids]

    _, normals = estimate_normals_after_ppf_resampling(points, d_dist_m, args)
    # Plane-fit normals have a sign ambiguity.  The CAD mesh provides the
    # orientation sign; only the direction is re-estimated after resampling.
    flip = np.einsum("ij,ij->i", normals, seed_normals) < 0.0
    normals[flip] *= -1.0
    normals = np.asarray([normalize(n) for n in normals], dtype=np.float64)

    if len(points) < 2:
        raise RuntimeError(
            f"Drost model resampling left only {len(points)} point(s); "
            "reduce --ppf-tau-d or increase --cad-sample-points."
        )

    print(
        f"  Drost model sampling: dense={len(dense_points):,} -> "
        f"sampled={len(points):,}, d_dist={d_dist_m*1000.0:.3f} mm"
    )
    return points, normals

def random_gt_transform(prior_T: np.ndarray, args, rng):
    """Generate hidden GT *around the supplied initial pose*.

    This is important: the initial pose is the prior mean.  The hidden true pose must
    therefore be a perturbation of that mean, not an unrelated transform around identity.
    """
    prior_T = np.asarray(prior_T, dtype=np.float64)

    if args.gt_generation == "prior-gaussian":
        sigma_t = args.prior_translation_sigma_mm / 1000.0
        sigma_r = math.radians(args.prior_rotation_sigma_deg)
        xi = np.r_[
            rng.normal(0.0, sigma_t, size=3),
            rng.normal(0.0, sigma_r, size=3),
        ]
        return prior_T @ se3_exp_vector(xi)

    # Fixed-magnitude perturbation, still centered on the supplied initial pose.
    delta = np.eye(4, dtype=np.float64)
    if args.gt_rotation_deg > 0:
        axis = normalize(rng.normal(size=3))
        delta[:3, :3] = Rotation.from_rotvec(
            axis * math.radians(args.gt_rotation_deg)
        ).as_matrix()
    if args.gt_translation_mm > 0:
        direction = normalize(rng.normal(size=3))
        delta[:3, 3] = direction * args.gt_translation_mm / 1000.0
    return prior_T @ delta

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


def build_ppf_model(model_points, model_normals, diameter_m, args):
    """Prepare the sampled/oriented PPF model used by PCL.

    PPF feature hashing, voting, pose generation, and pose clustering are NOT
    implemented here. They are delegated to pclpybridge -> PCL.
    """
    dist_step = args.ppf_tau_d * diameter_m
    angle_step = 2.0 * math.pi / args.ppf_nangle
    print(
        f"  PCL PPF model points={len(model_points):,}, "
        f"distance_step={dist_step*1000.0:.3f} mm, "
        f"angle_step={math.degrees(angle_step):.1f} deg"
    )
    return PPFModel(
        points_m=np.asarray(model_points, dtype=np.float64),
        normals=np.asarray(model_normals, dtype=np.float64),
        diameter_m=float(diameter_m),
        dist_step_m=float(dist_step),
        angle_step_rad=float(angle_step),
        nangle=int(args.ppf_nangle),
    )


def preprocess_multiple_scans_for_ppf(
    scans: list[tuple[np.ndarray, np.ndarray]],
    d_dist_m: float,
    args,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Preprocess the cumulative scene before sending it to PCL PPF.

    The cumulative raw scans are spatially resampled with the same d_dist used
    for the CAD model. Normals are recomputed after resampling and oriented
    toward the viewpoint of the scan that contributed each retained point.

    The actual PPF descriptor/hash/voting/clustering is handled by PCL.
    """
    if not scans:
        raise RuntimeError("No scans supplied for cumulative PPF")

    raw_points: list[np.ndarray] = []
    raw_source_ids: list[np.ndarray] = []
    viewpoints = [
        np.asarray(viewpoint_world, dtype=np.float64)
        for _, viewpoint_world in scans
    ]
    raw_counts: list[int] = []

    for scan_id, (points_world, _viewpoint_world) in enumerate(scans):
        pts = np.asarray(points_world, dtype=np.float64)
        raw_counts.append(len(pts))
        if len(pts) == 0:
            continue
        raw_points.append(pts)
        raw_source_ids.append(np.full(len(pts), scan_id, dtype=np.int32))

    if not raw_points:
        raise RuntimeError("All cumulative scans are empty")

    points_raw = np.vstack(raw_points)
    source_raw = np.concatenate(raw_source_ids)
    selected_ids = deterministic_min_distance_sample_indices(points_raw, d_dist_m)
    points = points_raw[selected_ids]
    source_ids = source_raw[selected_ids]

    if len(points) < args.ppf_min_sampled_scene_points:
        raise RuntimeError(
            f"PPF scene resampling left only {len(points)} points at "
            f"d_dist={d_dist_m*1000.0:.3f} mm. Reduce --ppf-tau-d."
        )

    _, normals = estimate_normals_after_ppf_resampling(points, d_dist_m, args)

    viewpoint_array = np.asarray(viewpoints, dtype=np.float64)
    for i in range(len(points)):
        sid = int(source_ids[i])
        to_view = viewpoint_array[sid] - points[i]
        if float(np.dot(normals[i], to_view)) < 0.0:
            normals[i] *= -1.0
    normals = np.asarray([normalize(n) for n in normals], dtype=np.float64)

    counts = [int(np.count_nonzero(source_ids == i)) for i in range(len(scans))]
    print(
        f"    PPF scene sampling: raw={len(points_raw):,} -> sampled={len(points):,}, "
        f"d_dist={d_dist_m*1000.0:.3f} mm"
    )
    for i, (raw_n, sampled_n) in enumerate(zip(raw_counts, counts), start=1):
        print(f"      Scan{i}: raw={raw_n:,}, retained={sampled_n:,}")

    return points, normals, counts


def pcl_ppf_result_to_clusters(ppf_result) -> list[PoseCluster]:
    """Convert pclpybridge PPFResult into this simulator's PoseCluster objects."""
    transforms = np.asarray(ppf_result.transforms, dtype=np.float64)
    votes = np.asarray(ppf_result.votes, dtype=np.float64)

    if transforms.ndim != 3 or transforms.shape[1:] != (4, 4):
        raise RuntimeError(
            f"Unexpected PCL PPF transform shape: {transforms.shape}"
        )
    if len(transforms) != len(votes):
        raise RuntimeError("PCL PPF transform/vote count mismatch")
    if len(transforms) == 0:
        raise RuntimeError("PCL PPF returned no pose candidates")

    clusters: list[PoseCluster] = []
    for rank, (T, vote) in enumerate(zip(transforms, votes), start=1):
        clusters.append(
            PoseCluster(
                rank=rank,
                transform_model_to_world=T.copy(),
                score=float(vote),
                member_count=1,
                members=[],
                translation_spread_mm=0.0,
                rotation_spread_deg=0.0,
            )
        )
    return clusters


def run_pcl_ppf_registration(
    model: PPFModel,
    scene_points: np.ndarray,
    scene_normals: np.ndarray,
    args,
):
    """Run PCL PPF through pclpybridge.

    Returned transforms map CAD/model -> cumulative scene/world coordinates.
    """
    reference_rate = max(
        1, int(round(1.0 / float(args.ppf_reference_fraction)))
    )

    result = pclb.ppf_register(
        model.points_m,
        scene_points,
        model_normals=model.normals,
        scene_normals=scene_normals,
        # Normals are supplied explicitly, so this radius is only a fallback.
        normal_radius=max(
            model.dist_step_m * args.ppf_normal_radius_factor,
            model.dist_step_m * 1.01,
        ),
        angle_step_deg=math.degrees(model.angle_step_rad),
        distance_step=model.dist_step_m,
        scene_reference_rate=reference_rate,
        position_cluster_threshold=args.cluster_translation_mm / 1000.0,
        rotation_cluster_threshold_deg=args.cluster_rotation_deg,
        max_candidates=args.ppf_max_candidates,
    )

    print(
        f"  PCL PPF backend={result.candidate_backend}, "
        f"converged={result.converged}, candidates={len(result.transforms)}"
    )
    if len(result.votes):
        top = result.votes[:min(8, len(result.votes))]
        print("  PCL PPF votes:", ", ".join(str(int(v)) for v in top))

    return result

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


def prepare_geometric_likelihood_target(target_cloud) -> dict[str, Any]:
    """Common CAD surface representation used by every sensor type after 3-D conversion."""
    points = np.asarray(target_cloud.points, dtype=np.float64)
    normals = np.asarray(target_cloud.normals, dtype=np.float64)
    if len(points) == 0 or len(normals) != len(points):
        raise RuntimeError("Geometric likelihood target requires CAD points and normals")
    normals = np.asarray([normalize(n) for n in normals], dtype=np.float64)
    return {"points_m": points, "normals": normals, "tree": cKDTree(points)}


def prepare_geometric_observations(points_world: np.ndarray, args) -> np.ndarray:
    """Spatially decorrelate dense observations before likelihood/information use."""
    pts = np.asarray(points_world, dtype=np.float64)
    if len(pts) == 0:
        raise RuntimeError("Cannot evaluate geometric likelihood on an empty scan")

    spacing_m = args.geom_observation_spacing_mm / 1000.0
    if spacing_m > 0.0 and len(pts) > 1:
        keep = deterministic_min_distance_sample_indices(pts, spacing_m)
        pts = pts[keep]

    if len(pts) > args.geom_max_points:
        ids = np.linspace(0, len(pts) - 1, args.geom_max_points, dtype=np.int64)
        pts = pts[ids]
    return pts


def geometric_mixture_terms(
    prepared_points_world: np.ndarray,
    hypothesis_model_to_world: np.ndarray,
    geom_db: dict[str, Any],
    args,
) -> dict[str, Any]:
    """Evaluate robust object/background mixture and local CAD correspondences."""
    pts = np.asarray(prepared_points_world, dtype=np.float64)
    world_to_model = invert_transform(hypothesis_model_to_world)
    x = transform_points(pts, world_to_model)
    nn_dist, nn_id = geom_db["tree"].query(x, k=1, workers=-1)
    m = geom_db["points_m"][nn_id]
    n = geom_db["normals"][nn_id]
    delta = x - m

    e_n = np.einsum("ij,ij->i", delta, n)
    d2 = np.einsum("ij,ij->i", delta, delta)
    e_t2 = np.maximum(d2 - e_n * e_n, 0.0)

    sigma_n = args.geom_sigma_normal_mm / 1000.0
    sigma_t = args.geom_sigma_tangent_mm / 1000.0
    log_norm_obj = (
        -1.5 * math.log(2.0 * math.pi)
        - math.log(sigma_n)
        - 2.0 * math.log(sigma_t)
    )
    logp_obj = log_norm_obj - 0.5 * (
        (e_n / sigma_n) ** 2 + e_t2 / (sigma_t * sigma_t)
    )

    pi_obj = float(args.geom_object_prior)
    half = args.geom_outlier_halfwidth_mm / 1000.0
    outlier_volume = max((2.0 * half) ** 3, EPS)
    logp_out = -math.log(outlier_volume)

    a = math.log(pi_obj) + logp_obj
    b = math.log(1.0 - pi_obj) + logp_out
    log_mix = np.logaddexp(a, b)
    gamma_obj = np.exp(a - log_mix)

    return {
        "points_world": pts,
        "points_model": x,
        "cad_points": m,
        "cad_normals": n,
        "delta": delta,
        "nn_dist": nn_dist,
        "normal_residual": e_n,
        "gamma_obj": gamma_obj,
        "log_mix": log_mix,
    }


def geometric_information_and_gradient(
    terms: dict[str, Any],
    args,
) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Newton information/gradient for the robust anisotropic surface model.

    The latent object responsibility gamma downweights background/outlier points.
    A right perturbation T' = T Exp(dxi^) gives, in the model frame,
        x' ~= x + [-I, [x]_x] dxi.
    """
    sigma_n = args.geom_sigma_normal_mm / 1000.0
    sigma_t = args.geom_sigma_tangent_mm / 1000.0
    inv_t2 = 1.0 / (sigma_t * sigma_t)
    inv_n2 = 1.0 / (sigma_n * sigma_n)

    H = np.zeros((6, 6), dtype=np.float64)
    g = np.zeros(6, dtype=np.float64)
    I3 = np.eye(3, dtype=np.float64)

    for x, n, delta, gamma in zip(
        terms["points_model"],
        terms["cad_normals"],
        terms["delta"],
        terms["gamma_obj"],
    ):
        # Inverse covariance: tangent variance sigma_t^2, normal variance sigma_n^2.
        W = inv_t2 * I3 + (inv_n2 - inv_t2) * np.outer(n, n)
        J = np.hstack((-I3, skew(x)))
        wg = float(gamma)
        H += wg * (J.T @ W @ J)
        g += wg * (J.T @ W @ delta)
    return H, g


def prior_information_matrix(args) -> np.ndarray:
    sigma_t = args.prior_translation_sigma_mm / 1000.0
    sigma_r = math.radians(args.prior_rotation_sigma_deg)
    return np.diag([1.0 / sigma_t**2] * 3 + [1.0 / sigma_r**2] * 3)


def robust_geometric_log_likelihood(
    points_world: np.ndarray,
    hypothesis_model_to_world: np.ndarray,
    geom_db: dict[str, Any],
    args,
) -> dict[str, float]:
    """Sensor-agnostic robust surface likelihood on spatially decorrelated points."""
    pts = prepare_geometric_observations(points_world, args)
    terms = geometric_mixture_terms(pts, hypothesis_model_to_world, geom_db, args)
    return {
        "log_likelihood": float(np.sum(terms["log_mix"])),
        "mean_log_likelihood": float(np.mean(terms["log_mix"])),
        "expected_object_fraction": float(np.mean(terms["gamma_obj"])),
        "mean_abs_normal_residual_mm": float(
            np.mean(np.abs(terms["normal_residual"])) * 1000.0
        ),
        "median_nn_distance_mm": float(np.median(terms["nn_dist"]) * 1000.0),
        "evaluated_points": int(len(pts)),
    }


def local_refine_and_covariance(
    initial_T: np.ndarray,
    prepared_points_world: np.ndarray,
    prior_T: np.ndarray,
    geom_db: dict[str, Any],
    args,
) -> dict[str, Any]:
    """Refine one PPF proposal and build its Laplace/Gauss-Newton covariance.

    This turns a discrete PPF proposal into one Gaussian component (T_i, Sigma_i).
    The prior is included as local information.  The SE(3) prior Jacobian is
    approximated as identity in the local tangent space, which is the standard
    small-perturbation Gauss-Newton approximation used here.
    """
    T = np.asarray(initial_T, dtype=np.float64).copy()
    Lambda0 = prior_information_matrix(args)

    def score(Tq: np.ndarray) -> tuple[float, dict[str, Any], np.ndarray, float]:
        terms_q = geometric_mixture_terms(prepared_points_world, Tq, geom_db, args)
        prior_log, prior_mahal, xi_q = pose_prior_log_score(Tq, prior_T, args)
        if prior_mahal > args.prior_gate_mahalanobis:
            total = -float("inf")
        else:
            total = float(prior_log + np.sum(terms_q["log_mix"]))
        return total, terms_q, xi_q, float(prior_mahal)

    current_score, terms, xi_prior, prior_mahal = score(T)
    iterations = 0

    for it in range(args.posterior_local_refine_iterations):
        H_geom, g_geom = geometric_information_and_gradient(terms, args)
        H = H_geom + Lambda0
        # Local approximation of the prior gradient.
        g = g_geom + Lambda0 @ xi_prior

        # Small Levenberg damping only for numerical stability; prior keeps H SPD.
        damp = max(float(np.max(np.diag(H))), 1.0) * 1.0e-10
        try:
            dxi = -np.linalg.solve(H + damp * np.eye(6), g)
        except np.linalg.LinAlgError:
            dxi = -np.linalg.pinv(H + damp * np.eye(6), rcond=1.0e-12) @ g

        max_t = args.posterior_local_refine_max_translation_step_mm / 1000.0
        max_r = math.radians(args.posterior_local_refine_max_rotation_step_deg)
        nt = float(np.linalg.norm(dxi[:3]))
        nr = float(np.linalg.norm(dxi[3:]))
        if nt > max_t:
            dxi[:3] *= max_t / max(nt, EPS)
        if nr > max_r:
            dxi[3:] *= max_r / max(nr, EPS)

        if np.linalg.norm(dxi[:3]) < 1.0e-7 and np.linalg.norm(dxi[3:]) < 1.0e-6:
            break

        accepted = False
        for alpha in (1.0, 0.5, 0.25, 0.1):
            T_try = T @ se3_exp_vector(alpha * dxi)
            try_score, try_terms, try_xi, try_mahal = score(T_try)
            if try_score > current_score + 1.0e-9:
                T = T_try
                current_score = try_score
                terms = try_terms
                xi_prior = try_xi
                prior_mahal = try_mahal
                accepted = True
                break
        iterations = it + 1
        if not accepted:
            break

    # Laplace covariance at the locally refined component center.
    H_geom, _ = geometric_information_and_gradient(terms, args)
    H = H_geom + Lambda0
    H = 0.5 * (H + H.T)
    eigval, eigvec = np.linalg.eigh(H)
    floor = max(float(np.max(eigval)), 1.0) * 1.0e-12
    eigval = np.maximum(eigval, floor)
    covariance = (eigvec * (1.0 / eigval)[None, :]) @ eigvec.T
    covariance = 0.5 * (covariance + covariance.T)

    std_t_mm = np.sqrt(np.maximum(np.diag(covariance)[:3], 0.0)) * 1000.0
    std_r_deg = np.degrees(np.sqrt(np.maximum(np.diag(covariance)[3:], 0.0)))

    return {
        "T_model_to_world": T,
        "covariance_6x6": covariance,
        "local_std_translation_mm": std_t_mm,
        "local_std_rotation_deg": std_r_deg,
        "log_posterior_unnormalized": float(current_score),
        "prior_mahalanobis": float(prior_mahal),
        "prior_xi": xi_prior,
        "log_likelihood": float(np.sum(terms["log_mix"])),
        "mean_log_likelihood": float(np.mean(terms["log_mix"])),
        "expected_object_fraction": float(np.mean(terms["gamma_obj"])),
        "mean_abs_normal_residual_mm": float(
            np.mean(np.abs(terms["normal_residual"])) * 1000.0
        ),
        "median_nn_distance_mm": float(np.median(terms["nn_dist"]) * 1000.0),
        "evaluated_points": int(len(prepared_points_world)),
        "refine_iterations": int(iterations),
    }


def expected_covariance_information_gain(
    candidate: dict[str, Any],
    current_covariance: np.ndarray,
    geom_db: dict[str, Any],
    args,
) -> dict[str, float]:
    """Expected local information gain from the CAD geometry in one scan footprint."""
    pts = np.asarray(candidate["expected_scan_cad"].clean_points_m, dtype=np.float64)
    if len(pts) == 0:
        return {"information_gain": 0.0, "predicted_max_t_std_mm": float("inf"),
                "predicted_max_r_std_deg": float("inf")}
    pts = prepare_geometric_observations(pts, args)
    _, nn_id = geom_db["tree"].query(pts, k=1, workers=-1)
    n = geom_db["normals"][nn_id]

    sigma_n = args.geom_sigma_normal_mm / 1000.0
    sigma_t = args.geom_sigma_tangent_mm / 1000.0
    inv_t2 = 1.0 / (sigma_t * sigma_t)
    inv_n2 = 1.0 / (sigma_n * sigma_n)
    I3 = np.eye(3, dtype=np.float64)
    Hscan = np.zeros((6, 6), dtype=np.float64)
    # Expected inlier weight: use the object-mixture prior before seeing the scan.
    gamma_expected = float(args.geom_object_prior)
    for x, ni in zip(pts, n):
        W = inv_t2 * I3 + (inv_n2 - inv_t2) * np.outer(ni, ni)
        J = np.hstack((-I3, skew(x)))
        Hscan += gamma_expected * (J.T @ W @ J)

    current_covariance = 0.5 * (current_covariance + current_covariance.T)
    current_info = np.linalg.pinv(current_covariance, rcond=1.0e-12)
    post_info = 0.5 * (current_info + Hscan + (current_info + Hscan).T)
    post_cov = np.linalg.pinv(post_info, rcond=1.0e-12)
    post_cov = 0.5 * (post_cov + post_cov.T)

    sign0, logdet0 = np.linalg.slogdet(current_covariance)
    sign1, logdet1 = np.linalg.slogdet(post_cov)
    info_gain = 0.0 if sign0 <= 0 or sign1 <= 0 else 0.5 * float(logdet0 - logdet1)
    std_t = np.sqrt(np.maximum(np.diag(post_cov)[:3], 0.0)) * 1000.0
    std_r = np.degrees(np.sqrt(np.maximum(np.diag(post_cov)[3:], 0.0)))
    return {
        "information_gain": float(max(info_gain, 0.0)),
        "predicted_max_t_std_mm": float(np.max(std_t)),
        "predicted_max_r_std_deg": float(np.max(std_r)),
    }

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

def prepare_cad_fpfh_database(mesh, args) -> dict[str, Any]:
    """Precompute Open3D FPFH descriptors on the complete CAD surface.

    The planner does not synthesize noisy laser ranges here.  It only asks:
    under each retained pose hypothesis, which CAD surface patch would occupy
    the same world scan footprint, and how different are those FPFH patches?
    """
    o3d = require_open3d()

    # Dense enough to make voxel_down_sample define the actual feature support.
    dense_count = max(int(args.cad_sample_points), 100000)
    cloud = mesh.sample_points_uniformly(number_of_points=dense_count)

    if not cloud.has_normals():
        cloud.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=args.fpfh_normal_radius_mm / 1000.0,
                max_nn=args.fpfh_normal_max_nn,
            )
        )

    cloud = cloud.voxel_down_sample(args.fpfh_voxel_mm / 1000.0)
    cloud.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=args.fpfh_normal_radius_mm / 1000.0,
            max_nn=args.fpfh_normal_max_nn,
        )
    )
    cloud.normalize_normals()

    feature = o3d.pipelines.registration.compute_fpfh_feature(
        cloud,
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=args.fpfh_feature_radius_mm / 1000.0,
            max_nn=args.fpfh_feature_max_nn,
        ),
    )

    points = np.asarray(cloud.points, dtype=np.float64)
    normals = np.asarray(cloud.normals, dtype=np.float64)
    descriptors = np.asarray(feature.data, dtype=np.float64).T

    if descriptors.ndim != 2 or descriptors.shape[1] != 33:
        raise RuntimeError(
            f"Expected Open3D FPFH dimension 33, got {descriptors.shape}"
        )
    if len(points) != len(descriptors):
        raise RuntimeError("CAD FPFH point/descriptor count mismatch")
    if len(points) < args.fpfh_min_patch_points:
        raise RuntimeError("Too few CAD FPFH points")

    # Robust CAD-wide scale for the native Open3D FPFH L2 distance.
    # This only normalizes feature distance; no hit/miss reward is mixed in.
    rng = np.random.default_rng(args.seed + 1701)
    sample_n = min(len(descriptors), 2000)
    ids = rng.choice(len(descriptors), size=sample_n, replace=False)
    a = descriptors[ids]
    b = descriptors[rng.permutation(ids)]
    pair_d = np.linalg.norm(a - b, axis=1)
    positive = pair_d[pair_d > EPS]
    l2_scale = float(np.median(positive)) if len(positive) else 1.0
    l2_scale = max(l2_scale, EPS)

    return {
        "cloud": cloud,
        "points_m": points,
        "normals": normals,
        "descriptors": descriptors,
        "tree": cKDTree(points),
        "open3d_l2_scale": l2_scale,
    }

def fpfh_descriptor_distance(
    f: np.ndarray,
    g: np.ndarray,
    metric: str,
    open3d_l2_scale: float,
) -> tuple[float, float]:
    """Return (planner_distance, raw_distance).

    open3d_l2:
      raw_distance = ||f-g||_2 in the native 33-D Open3D FPFH space.
      planner_distance = raw_distance / robust CAD-wide median L2 scale.

    hist_intersection:
      First L1-normalize nonnegative descriptors.  The FPFH paper used the
      histogram-intersection kernel sum(min(.)) to compare mean signatures.
      Here we convert that similarity to a distance: 1 - intersection.
    """
    f = np.asarray(f, dtype=np.float64)
    g = np.asarray(g, dtype=np.float64)

    if metric == "open3d_l2":
        raw = float(np.linalg.norm(f - g))
        return raw / max(open3d_l2_scale, EPS), raw

    if metric == "hist_intersection":
        fp = np.maximum(f, 0.0)
        gp = np.maximum(g, 0.0)
        sf = float(np.sum(fp))
        sg = float(np.sum(gp))
        if sf <= EPS or sg <= EPS:
            return 1.0, 1.0
        fp /= sf
        gp /= sg
        similarity = float(np.sum(np.minimum(fp, gp)))
        distance = float(np.clip(1.0 - similarity, 0.0, 1.0))
        return distance, distance

    raise ValueError(f"Unsupported FPFH distance metric: {metric}")

def predicted_cad_fpfh_patch(
    plan: RandomScanPlan,
    map_model_to_world: np.ndarray,
    hypothesis_model_to_world: np.ndarray,
    fpfh_db: dict[str, Any],
    args,
) -> dict[str, Any]:
    """Predict which CAD patch occupies one fixed world scan footprint."""
    T_mapcad_to_hcad = invert_transform(hypothesis_model_to_world) @ map_model_to_world
    R_rel = T_mapcad_to_hcad[:3, :3]

    center = transform_points(plan.point_cad_m[None, :], T_mapcad_to_hcad)[0]
    profile_axis = normalize(R_rel @ plan.profile_axis_cad)
    sweep_axis = normalize(R_rel @ plan.sweep_axis_cad)
    scan_normal = normalize(R_rel @ plan.normal_cad)

    half_profile = 0.5 * args.profile_width_mm / 1000.0
    half_sweep = 0.5 * args.scan_length_mm / 1000.0
    half_depth = args.fpfh_patch_depth_mm / 1000.0
    radius = math.sqrt(
        half_profile * half_profile + half_sweep * half_sweep + half_depth * half_depth
    )

    candidate_ids = fpfh_db["tree"].query_ball_point(center, radius)
    if not candidate_ids:
        return {
            "hit": False,
            "descriptor": None,
            "point_count": 0,
            "center_cad_m": center,
            "point_ids": np.empty(0, dtype=np.int64),
        }

    ids = np.asarray(candidate_ids, dtype=np.int64)
    delta = fpfh_db["points_m"][ids] - center[None, :]
    u = delta @ profile_axis
    v = delta @ sweep_axis
    d = delta @ scan_normal

    inside = (
        (np.abs(u) <= half_profile)
        & (np.abs(v) <= half_sweep)
        & (np.abs(d) <= half_depth)
    )

    cos_min = math.cos(math.radians(args.fpfh_max_incidence_deg))
    incidence = np.abs(fpfh_db["normals"][ids] @ scan_normal)
    inside &= incidence >= cos_min

    kept = ids[inside]
    if len(kept) < args.fpfh_min_patch_points:
        return {
            "hit": False,
            "descriptor": None,
            "point_count": int(len(kept)),
            "center_cad_m": center,
            "point_ids": kept,
        }

    patch_descriptor = np.mean(fpfh_db["descriptors"][kept], axis=0)
    return {
        "hit": True,
        "descriptor": patch_descriptor,
        "point_count": int(len(kept)),
        "center_cad_m": center,
        "point_ids": kept,
    }

def fpfh_hypothesis_disagreement_score(
    predicted_patches: list[dict[str, Any]],
    weights: np.ndarray,
    fpfh_db: dict[str, Any],
    args,
) -> dict[str, Any]:
    """Pure CAD-FPFH disagreement for one next-scan action.

    Policy:
        1) The action is valid only when every active PPF hypothesis predicts
           a usable CAD patch in the same world scan footprint.
        2) For a valid action, compare only the CAD patch FPFH descriptors:

            U(a) = sum_{i<j} w_i w_j D_FPFH(F_i(a), F_j(a))

    No synthetic sensor likelihood, hit/miss bonus, incremental PPF score,
    residual, ICP score, or Bayesian term is used in next-scan selection.
    """
    if len(predicted_patches) < 2:
        return {
            "valid": False,
            "score": 0.0,
            "valid_patch_count": int(sum(bool(p["hit"]) for p in predicted_patches)),
            "hypothesis_count": int(len(predicted_patches)),
            "max_raw_feature_distance": 0.0,
            "mean_raw_feature_distance": 0.0,
        }

    valid_patch_count = int(sum(bool(p["hit"]) for p in predicted_patches))
    if valid_patch_count != len(predicted_patches):
        return {
            "valid": False,
            "score": 0.0,
            "valid_patch_count": valid_patch_count,
            "hypothesis_count": int(len(predicted_patches)),
            "max_raw_feature_distance": 0.0,
            "mean_raw_feature_distance": 0.0,
        }

    w = np.asarray(weights, dtype=np.float64)
    w = w / np.sum(w)

    score = 0.0
    raw_feature_distances: list[float] = []

    for i in range(len(predicted_patches)):
        for j in range(i + 1, len(predicted_patches)):
            d, raw = fpfh_descriptor_distance(
                predicted_patches[i]["descriptor"],
                predicted_patches[j]["descriptor"],
                args.fpfh_distance_metric,
                fpfh_db["open3d_l2_scale"],
            )
            score += float(w[i] * w[j]) * d
            raw_feature_distances.append(float(raw))

    return {
        "valid": True,
        "score": float(score),
        "valid_patch_count": valid_patch_count,
        "hypothesis_count": int(len(predicted_patches)),
        "max_raw_feature_distance": (
            float(max(raw_feature_distances))
            if raw_feature_distances else 0.0
        ),
        "mean_raw_feature_distance": (
            float(np.mean(raw_feature_distances))
            if raw_feature_distances else 0.0
        ),
    }

def make_scan_plan(
    point_cad_m: np.ndarray,
    normal_cad: np.ndarray,
    angle_rad: float,
) -> RandomScanPlan:
    t1, t2 = tangent_basis(normal_cad)
    return RandomScanPlan(
        point_cad_m=np.asarray(point_cad_m, dtype=np.float64).copy(),
        normal_cad=normalize(normal_cad),
        angle_deg=float(math.degrees(angle_rad)),
        profile_axis_cad=normalize(math.cos(angle_rad) * t1 + math.sin(angle_rad) * t2),
        sweep_axis_cad=normalize(-math.sin(angle_rad) * t1 + math.cos(angle_rad) * t2),
    )


def build_global_unique_candidate_pool(
    cad_scene,
    fpfh_db: dict[str, Any],
    args,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build offline scan actions from global CAD-patch FPFH uniqueness.

    Every CAD FPFH point is considered as a scan center.  For each center we
    test the requested scan directions, represent the expected footprint by the
    mean FPFH of that CAD patch, then ask how far its descriptor is from the K
    nearest *spatially distant* patch descriptors.  High distance means the
    patch is globally distinctive on this CAD.

    Spatial NMS is applied only after uniqueness is known, so a random proposal
    stage cannot discard the best distinctive point beforehand.
    """
    identity = np.eye(4, dtype=np.float64)
    points = fpfh_db["points_m"]
    normals = fpfh_db["normals"]

    actions: list[dict[str, Any]] = []
    angles = np.linspace(0.0, math.pi, args.orientation_samples, endpoint=False)

    print(
        f"  evaluate global patch uniqueness: centers={len(points)}, "
        f"orientations={len(angles)}"
    )

    for idx, (point, normal0) in enumerate(zip(points, normals)):
        normal = orient_surface_normal_for_raycast(
            cad_scene, point, normal0, args.standoff_mm
        )
        for angle in angles:
            plan = make_scan_plan(point, normal, float(angle))
            patch = predicted_cad_fpfh_patch(
                plan, identity, identity, fpfh_db, args
            )
            if not patch["hit"]:
                continue
            actions.append({
                "source_point_id": int(idx),
                "point_cad_m": point.copy(),
                "normal_cad": normal.copy(),
                "plan": plan,
                "descriptor": patch["descriptor"].copy(),
                "patch_point_count": int(patch["point_count"]),
            })
        if (idx + 1) % max(1, len(points) // 10) == 0 or idx + 1 == len(points):
            print(f"    patch actions {idx+1:5d}/{len(points)}: valid={len(actions):,}")

    if len(actions) < 2:
        raise RuntimeError("Too few valid CAD patch actions for global uniqueness")

    descriptors = np.asarray([a["descriptor"] for a in actions], dtype=np.float64)
    centers = np.asarray([a["point_cad_m"] for a in actions], dtype=np.float64)
    desc_tree = cKDTree(descriptors)
    n = len(actions)
    k_need = min(args.global_uniqueness_knn, max(1, n - 1))
    query_k = min(n, max(64, 8 * k_need + 1))
    _, neighbor_ids = desc_tree.query(descriptors, k=query_k, workers=-1)
    if neighbor_ids.ndim == 1:
        neighbor_ids = neighbor_ids[:, None]

    exclude_m = args.global_uniqueness_exclude_mm / 1000.0
    scale = fpfh_db["open3d_l2_scale"]

    for i, action in enumerate(actions):
        selected_distances: list[float] = []
        for j in np.asarray(neighbor_ids[i]).reshape(-1):
            j = int(j)
            if j == i:
                continue
            if np.linalg.norm(centers[j] - centers[i]) < exclude_m:
                continue
            d, _ = fpfh_descriptor_distance(
                descriptors[i], descriptors[j], args.fpfh_distance_metric, scale
            )
            selected_distances.append(float(d))
            if len(selected_distances) >= k_need:
                break

        # Rare fallback when the descriptor-neighbor query was dominated by
        # spatially local patches.  Exact L2 over the remaining spatial set.
        if len(selected_distances) < k_need:
            spatial_ok = np.linalg.norm(centers - centers[i], axis=1) >= exclude_m
            spatial_ok[i] = False
            ids = np.flatnonzero(spatial_ok)
            if len(ids):
                raw_l2 = np.linalg.norm(descriptors[ids] - descriptors[i], axis=1)
                order = ids[np.argsort(raw_l2)]
                for j in order:
                    d, _ = fpfh_descriptor_distance(
                        descriptors[i], descriptors[int(j)],
                        args.fpfh_distance_metric, scale,
                    )
                    selected_distances.append(float(d))
                    if len(selected_distances) >= k_need:
                        break

        action["global_uniqueness"] = (
            float(np.mean(selected_distances)) if selected_distances else 0.0
        )

    order = sorted(
        range(len(actions)),
        key=lambda i: actions[i]["global_uniqueness"],
        reverse=True,
    )

    candidates: list[dict[str, Any]] = []
    selected_centers: list[np.ndarray] = []
    nms_m = args.candidate_nms_mm / 1000.0

    for global_rank, action_id in enumerate(order, start=1):
        action = actions[action_id]
        p = action["point_cad_m"]
        if selected_centers:
            chosen = np.asarray(selected_centers)
            if np.any(np.linalg.norm(chosen - p, axis=1) < nms_m):
                continue

        scan_cad = simulate_scan_from_plan_world(
            cad_scene,
            action["plan"],
            identity,
            args,
        )
        if len(scan_cad.clean_points_m) < args.min_new_scan_points:
            continue

        candidate = {
            "candidate_id": len(candidates) + 1,
            "global_rank": int(global_rank),
            "source_point_id": action["source_point_id"],
            "point_cad_m": action["point_cad_m"].copy(),
            "normal_cad": action["normal_cad"].copy(),
            "plan": action["plan"],
            "expected_scan_cad": scan_cad,
            "global_uniqueness": float(action["global_uniqueness"]),
            "patch_point_count": int(action["patch_point_count"]),
        }
        candidates.append(candidate)
        selected_centers.append(p.copy())
        if len(candidates) >= args.candidate_count:
            break

    if len(candidates) < 2:
        raise RuntimeError(
            "Fewer than two global-unique candidates survived NMS/raycast. "
            "Reduce --candidate-nms-mm or --global-uniqueness-exclude-mm."
        )

    return candidates, actions


def build_posterior_state(
    clusters: list[PoseCluster],
    cumulative_world_points: np.ndarray,
    prior_T: np.ndarray,
    geom_db: dict[str, Any],
    args,
) -> dict[str, Any]:
    """Convert PPF proposals into a Gaussian-mixture posterior on SE(3).

    Each PPF proposal is only an initializer.  It is locally refined under the
    robust geometric likelihood and SE(3) prior, then assigned a Laplace/GN
    covariance Sigma_i.  Refined proposals that converge to the same local basin
    are de-duplicated so proposal multiplicity does not become probability mass.
    """
    if not clusters:
        raise RuntimeError("Cannot build posterior from zero PPF proposals")

    prepared_points = prepare_geometric_observations(cumulative_world_points, args)
    proposals = clusters[: min(len(clusters), args.posterior_max_hypotheses)]
    raw_records: list[dict[str, Any]] = []

    rejected_by_prior = 0
    nearest_rejected_mahal = float("inf")
    for cluster in proposals:
        _, proposal_prior_mahal, _ = pose_prior_log_score(
            cluster.transform_model_to_world, prior_T, args
        )
        if proposal_prior_mahal > args.prior_gate_mahalanobis:
            rejected_by_prior += 1
            nearest_rejected_mahal = min(nearest_rejected_mahal, proposal_prior_mahal)
            continue

        local = local_refine_and_covariance(
            cluster.transform_model_to_world,
            prepared_points,
            prior_T,
            geom_db,
            args,
        )
        if not np.isfinite(local["log_posterior_unnormalized"]):
            rejected_by_prior += 1
            nearest_rejected_mahal = min(nearest_rejected_mahal, local["prior_mahalanobis"])
            continue
        refined_cluster = PoseCluster(
            rank=int(cluster.rank),
            transform_model_to_world=local["T_model_to_world"].copy(),
            score=float(cluster.score),
            member_count=int(cluster.member_count),
            members=list(cluster.members),
            translation_spread_mm=float(cluster.translation_spread_mm),
            rotation_spread_deg=float(cluster.rotation_spread_deg),
        )
        raw_records.append({
            "cluster": refined_cluster,
            "proposal_transform_model_to_world": cluster.transform_model_to_world.copy(),
            "source_cluster_rank": int(cluster.rank),
            "ppf_vote": float(cluster.score),
            "prior_log_score": float(-0.5 * local["prior_mahalanobis"] ** 2),
            **local,
            "merged_proposal_count": 1,
        })

    if not raw_records:
        nearest_msg = (
            f", nearest rejected priorD={nearest_rejected_mahal:.2f}"
            if np.isfinite(nearest_rejected_mahal) else ""
        )
        raise RuntimeError(
            "All PPF proposals were rejected by the initial-pose prior gate "
            f"(Mahalanobis radius <= {args.prior_gate_mahalanobis:.2f}{nearest_msg}). "
            "This means PPF did not return a pose consistent with the supplied initial "
            "estimate. Increase the prior sigmas/gate only if the initial estimate is "
            "actually that uncertain; do not silently snap to a distant pose."
        )

    # Remove duplicate local modes.  PPF proposal multiplicity is not Bayesian mass.
    raw_records.sort(key=lambda r: r["log_posterior_unnormalized"], reverse=True)
    records: list[dict[str, Any]] = []
    trans_thr = args.cluster_translation_mm / 1000.0
    rot_thr = args.cluster_rotation_deg
    for rec in raw_records:
        T = rec["T_model_to_world"]
        duplicate_of = None
        for kept in records:
            Tk = kept["T_model_to_world"]
            dt = float(np.linalg.norm(T[:3, 3] - Tk[:3, 3]))
            dr = rotation_distance_deg(T[:3, :3], Tk[:3, :3])
            if dt <= trans_thr and dr <= rot_thr:
                duplicate_of = kept
                break
        if duplicate_of is None:
            records.append(rec)
        else:
            duplicate_of["merged_proposal_count"] += 1

    logw = np.asarray([r["log_posterior_unnormalized"] for r in records], dtype=float)
    logw -= float(np.max(logw))
    w = np.exp(logw)
    w /= max(float(np.sum(w)), EPS)
    for r, wi in zip(records, w):
        r["posterior_weight"] = float(wi)

    records.sort(key=lambda r: r["posterior_weight"], reverse=True)
    weights = np.asarray([r["posterior_weight"] for r in records], dtype=float)
    relative = weights / max(weights[0], EPS)

    plausible_count = max(
        1,
        int(np.count_nonzero(relative >= args.posterior_plausible_relative)),
    )
    if len(records) == 1:
        runnerup_relative = 0.0
        mode_separated = True
    else:
        runnerup_relative = float(relative[1])
        mode_separated = runnerup_relative < args.posterior_stop_runnerup_relative

    map_rec = records[0]
    max_t_std_mm = float(np.max(map_rec["local_std_translation_mm"]))
    max_r_std_deg = float(np.max(map_rec["local_std_rotation_deg"]))
    local_confident = (
        max_t_std_mm <= args.posterior_stop_translation_std_mm
        and max_r_std_deg <= args.posterior_stop_rotation_std_deg
    )
    stop_ready = bool(mode_separated and local_confident)

    if not mode_separated:
        active_count = plausible_count
        comparison_guard = False
        if active_count == 1 and len(records) >= 2:
            active_count = 2
            comparison_guard = True
        planning_mode = "mode_disambiguation"
    elif not local_confident:
        # One mode is dominant, but its continuous covariance is still too large.
        active_count = 1
        comparison_guard = False
        planning_mode = "local_covariance_reduction"
    else:
        active_count = 1
        comparison_guard = False
        planning_mode = "stop"

    active = records[:active_count]
    plausible = records[:plausible_count]
    return {
        "records": records,
        "active_records": active,
        "plausible_records": plausible,
        "clusters": [r["cluster"] for r in active],
        "plausible_clusters": [r["cluster"] for r in plausible],
        "count": int(active_count),
        "plausible_count": int(plausible_count),
        "relative_support": relative,
        "runnerup_relative": float(runnerup_relative),
        "plausible_threshold": float(args.posterior_plausible_relative),
        "stop_threshold": float(args.posterior_stop_runnerup_relative),
        "method": "se3_gmm_prior_x_robust_geometric_likelihood",
        "comparison_guard": bool(comparison_guard),
        "mode_separated": bool(mode_separated),
        "local_confident": bool(local_confident),
        "map_max_translation_std_mm": max_t_std_mm,
        "map_max_rotation_std_deg": max_r_std_deg,
        "planning_mode": planning_mode,
        "stop_ready": stop_ready,
        "prepared_geometric_points": int(len(prepared_points)),
        "ppf_rejected_by_prior_gate": int(rejected_by_prior),
    }


def build_hypothesis_bank(
    posterior_state: dict[str, Any],
) -> list[dict[str, Any]]:
    bank: list[dict[str, Any]] = []
    for k, rec in enumerate(posterior_state["active_records"], start=1):
        bank.append({
            "bank_id": k,
            "source_cluster_rank": rec["source_cluster_rank"],
            "ppf_score": rec["ppf_vote"],
            "relative_to_top1": float(
                rec["posterior_weight"]
                / max(posterior_state["active_records"][0]["posterior_weight"], EPS)
            ),
            "weight": rec["posterior_weight"],
            "posterior_weight": rec["posterior_weight"],
            "prior_mahalanobis": rec["prior_mahalanobis"],
            "geometric_log_likelihood": rec["log_likelihood"],
            "expected_object_fraction": rec["expected_object_fraction"],
            "mean_abs_normal_residual_mm": rec["mean_abs_normal_residual_mm"],
            "median_nn_distance_mm": rec["median_nn_distance_mm"],
            "local_covariance_6x6": rec["covariance_6x6"].copy(),
            "local_std_translation_mm": rec["local_std_translation_mm"].copy(),
            "local_std_rotation_deg": rec["local_std_rotation_deg"].copy(),
            "refine_iterations": rec["refine_iterations"],
            "merged_proposal_count": rec["merged_proposal_count"],
            "T_model_to_world": rec["T_model_to_world"].copy(),
        })
    z = sum(h["weight"] for h in bank)
    for h in bank:
        h["weight"] /= max(z, EPS)
    return bank


def posterior_gmm_covariance(bank: list[dict[str, Any]]) -> np.ndarray:
    """Total GMM covariance = within-component + between-component covariance."""
    if not bank:
        return np.full((6, 6), np.nan)
    T_ref = bank[0]["T_model_to_world"]
    w = np.asarray([h["weight"] for h in bank], dtype=float)
    w /= max(float(np.sum(w)), EPS)
    xi = np.asarray([
        se3_log_vector(invert_transform(T_ref) @ h["T_model_to_world"])
        for h in bank
    ])
    mu = np.sum(w[:, None] * xi, axis=0)
    cov = np.zeros((6, 6), dtype=np.float64)
    for wi, h, xii in zip(w, bank, xi):
        d = xii - mu
        cov += wi * (h["local_covariance_6x6"] + np.outer(d, d))
    return 0.5 * (cov + cov.T)


def posterior_local_covariance(bank: list[dict[str, Any]]) -> np.ndarray:
    """Backward-compatible alias: return the MAP component's local covariance."""
    if not bank:
        return np.full((6, 6), np.nan)
    return np.asarray(bank[0]["local_covariance_6x6"], dtype=np.float64)

def select_discriminative_next_scan(
    candidates: list[dict[str, Any]],
    previous_plans: list[RandomScanPlan],
    used_candidate_ids: set[int],
    bank: list[dict[str, Any]],
    map_model_to_world: np.ndarray,
    fpfh_db: dict[str, Any],
    posterior_state: dict[str, Any],
    geom_db: dict[str, Any],
    args,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Adaptive next-scan planner.

    - Multi-modal posterior: choose a scan that maximally separates pose modes
      using weighted CAD-FPFH disagreement.
    - Single dominant mode but large local covariance: choose a scan with maximal
      expected local Fisher information / covariance reduction.
    """
    mode = posterior_state["planning_mode"]
    weights = np.asarray([h["weight"] for h in bank], dtype=float)
    weights /= max(float(np.sum(weights)), EPS)

    best: dict[str, Any] | None = None
    evaluations: list[dict[str, Any]] = []
    exclude_m = args.rescan_exclusion_mm / 1000.0
    current_cov = np.asarray(bank[0]["local_covariance_6x6"], dtype=np.float64)

    for candidate in candidates:
        candidate_id = int(candidate["candidate_id"])
        if candidate_id in used_candidate_ids:
            continue

        if previous_plans and exclude_m > 0.0:
            dmin = min(
                float(np.linalg.norm(candidate["point_cad_m"] - p.point_cad_m))
                for p in previous_plans
            )
            if dmin < exclude_m:
                continue

        if mode == "mode_disambiguation":
            predicted = [
                predicted_cad_fpfh_patch(
                    candidate["plan"],
                    map_model_to_world,
                    h["T_model_to_world"],
                    fpfh_db,
                    args,
                )
                for h in bank
            ]
            utility = fpfh_hypothesis_disagreement_score(
                predicted, weights, fpfh_db, args
            )
            row = {
                "candidate_id": candidate_id,
                "planner_mode": mode,
                "global_uniqueness": float(candidate["global_uniqueness"]),
                "valid_under_all_hypotheses": bool(utility["valid"]),
                "valid_patch_count": int(utility["valid_patch_count"]),
                "hypothesis_count": int(utility["hypothesis_count"]),
                "fpfh_disagreement": float(utility["score"]),
                "information_gain": 0.0,
                "max_raw_feature_distance": float(utility["max_raw_feature_distance"]),
                "mean_raw_feature_distance": float(utility["mean_raw_feature_distance"]),
                "planner_score": float(utility["score"]),
            }
            evaluations.append(row)
            if not utility["valid"]:
                continue
            result = dict(candidate)
            result.update(row)
            result["predicted_patches"] = predicted

        elif mode == "local_covariance_reduction":
            info = expected_covariance_information_gain(
                candidate, current_cov, geom_db, args
            )
            row = {
                "candidate_id": candidate_id,
                "planner_mode": mode,
                "global_uniqueness": float(candidate["global_uniqueness"]),
                "valid_under_all_hypotheses": True,
                "valid_patch_count": 1,
                "hypothesis_count": 1,
                "fpfh_disagreement": 0.0,
                "information_gain": float(info["information_gain"]),
                "predicted_max_t_std_mm": float(info["predicted_max_t_std_mm"]),
                "predicted_max_r_std_deg": float(info["predicted_max_r_std_deg"]),
                "max_raw_feature_distance": 0.0,
                "mean_raw_feature_distance": 0.0,
                "planner_score": float(info["information_gain"]),
            }
            evaluations.append(row)
            result = dict(candidate)
            result.update(row)
            result["predicted_patches"] = []
        else:
            raise RuntimeError(f"Unexpected planning mode: {mode}")

        if best is None or result["planner_score"] > best["planner_score"]:
            best = result

    if best is None:
        raise RuntimeError(
            "No unused offline candidate is feasible for the current posterior planner. "
            "Increase --candidate-count or reduce --rescan-exclusion-mm."
        )

    return best, evaluations

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

CLUSTER_COLORS = [
    (0.10, 0.85, 0.25), (0.15, 0.55, 1.00),
    (1.00, 0.60, 0.10), (0.75, 0.25, 1.00),
    (0.95, 0.20, 0.45), (0.20, 0.85, 0.85),
    (0.85, 0.85, 0.20), (0.65, 0.65, 0.65),
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

def make_sphere_marker(center_m, radius_m, color):
    o3d = require_open3d()
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=float(radius_m))
    sphere.translate(np.asarray(center_m, dtype=np.float64))
    sphere.compute_vertex_normals()
    sphere.paint_uniform_color(color)
    return sphere

def accumulated_scan_geometries(accepted_scans):
    """Return colored point clouds for all accepted scans."""
    colors = [
        (1.00, 0.12, 0.12),  # Scan1
        (1.00, 0.62, 0.05),  # Scan2
        (0.15, 0.60, 1.00),  # Scan3
        (0.75, 0.30, 1.00),  # Scan4
        (0.15, 0.85, 0.70),  # Scan5+
    ]
    geoms = []
    for i, scan_record in enumerate(accepted_scans):
        geoms.append(
            make_cloud(
                scan_record["points_world"],
                colors[i % len(colors)],
            )
        )
    return geoms

def plot_global_uniqueness(
    output_dir: Path,
    actions: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    args,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    scores = np.asarray(
        sorted((a["global_uniqueness"] for a in actions), reverse=True),
        dtype=float,
    )
    selected = np.asarray([c["global_uniqueness"] for c in candidates], dtype=float)
    selected_rank = np.asarray([c["global_rank"] for c in candidates], dtype=int)

    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.plot(np.arange(1, len(scores) + 1), scores, linewidth=1.2, label="All valid scan actions")
    ax.scatter(
        selected_rank, selected,
        s=55, marker="o", label="Selected after spatial NMS",
    )
    ax.set_xlabel("Global uniqueness rank")
    ax.set_ylabel("Mean KNN FPFH distance")
    ax.set_title("Offline CAD scan-patch global FPFH uniqueness")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out = output_dir / "00_offline_global_uniqueness.png"
    fig.savefig(out, dpi=180)
    print(f"  saved: {out}")
    if args.show:
        plt.show()
    plt.close(fig)


def plot_posterior_support(
    output_dir: Path,
    posterior_state: dict[str, Any],
    round_id: int,
    args,
) -> None:
    """Plot posterior hypothesis weights; PPF votes are shown only as annotations."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    records = posterior_state["records"][: min(args.plot_top_clusters, len(posterior_state["records"]))]
    if not records:
        return
    weights = np.asarray([r["posterior_weight"] for r in records], dtype=float)
    x = np.arange(1, len(records) + 1)
    colors = []
    for i in range(len(records)):
        if i < posterior_state["plausible_count"]:
            colors.append("tab:blue")
        elif i < posterior_state["count"]:
            colors.append("tab:orange")
        else:
            colors.append("0.72")

    fig, ax = plt.subplots(figsize=(10.8, 6.0))
    ax.bar(x, weights, color=colors)
    ax.set_xlabel("Posterior pose-hypothesis rank")
    ax.set_ylabel("Posterior weight")
    ax.set_title(
        f"SE(3) prior x robust geometric likelihood — round {round_id} | "
        f"plausible={posterior_state['plausible_count']} | "
        f"W2/W1={posterior_state['runnerup_relative']:.3f}"
    )
    ax.grid(axis="y", alpha=0.25)
    for i, r in enumerate(records[:8]):
        ax.annotate(
            f"PPF#{r['source_cluster_rank']}\nV={r['ppf_vote']:.0f}",
            xy=(x[i], weights[i]), xytext=(0, 4), textcoords="offset points",
            ha="center", va="bottom", fontsize=7,
        )
    fig.tight_layout()
    out = output_dir / f"posterior_weights_round_{round_id:02d}.png"
    fig.savefig(out, dpi=180)
    print(f"  saved: {out}")
    if args.show:
        plt.show()
    plt.close(fig)


def plot_candidate_scores(
    output_dir: Path,
    evaluations: list[dict[str, Any]],
    selected_id: int,
    attempt_id: int,
    args,
) -> None:
    """Plot whichever posterior-driven next-scan utility is active."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    if not evaluations:
        return

    mode = evaluations[0].get("planner_mode", "mode_disambiguation")
    valid_rows = [r for r in evaluations if r.get("valid_under_all_hypotheses", True)]
    if not valid_rows:
        return
    rows = sorted(valid_rows, key=lambda r: r["planner_score"], reverse=True)
    rows = rows[:min(24, len(rows))]
    if not any(r["candidate_id"] == selected_id for r in rows):
        selected_row = next(r for r in valid_rows if r["candidate_id"] == selected_id)
        rows[-1] = selected_row

    labels = [f"C{r['candidate_id']}" for r in rows]
    scores = np.asarray([r["planner_score"] for r in rows], dtype=float)
    colors = [
        "tab:pink" if r["candidate_id"] == selected_id else "tab:blue"
        for r in rows
    ]

    fig, ax = plt.subplots(figsize=(11.5, 6.0))
    ax.bar(np.arange(len(rows)), scores, color=colors)
    ax.set_xticks(np.arange(len(rows)), labels, rotation=45, ha="right")
    ax.set_xlabel("Offline global-unique candidate")
    if mode == "mode_disambiguation":
        ax.set_ylabel("Weighted CAD-FPFH disagreement U(a)")
        ax.set_title(f"Next-scan mode disambiguation — selection {attempt_id}")
    else:
        ax.set_ylabel("Expected information gain 0.5 log det ratio")
        ax.set_title(f"Next-scan local covariance reduction — selection {attempt_id}")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    out = output_dir / f"candidate_scores_attempt_{attempt_id:02d}.png"
    fig.savefig(out, dpi=180)
    print(f"  saved: {out}")
    if args.show:
        plt.show()
    plt.close(fig)

def show_offline_candidates(mesh, candidates, args) -> None:
    if not args.show:
        return
    base = mesh.sample_points_uniformly(number_of_points=9000)
    base.paint_uniform_color((0.55, 0.55, 0.58))
    centers = np.asarray([c["point_cad_m"] for c in candidates], dtype=float)
    cloud = make_cloud(centers)
    scores = np.asarray([c["global_uniqueness"] for c in candidates], dtype=float)
    lo, hi = float(np.min(scores)), float(np.max(scores))
    q = (scores - lo) / max(hi - lo, EPS)
    colors = np.column_stack((q, 0.25 + 0.55 * (1.0 - q), 1.0 - 0.75 * q))
    cloud.colors = require_open3d().utility.Vector3dVector(colors)
    geoms = [base, cloud]
    for c in candidates[:min(8, len(candidates))]:
        geoms.append(make_path_line(c["expected_scan_cad"].path_origins_m, (1.0, 0.25, 0.75)))
    draw_stage(
        "00 - Offline global-unique CAD candidates",
        geoms,
        (
            f"Candidate centers={len(candidates)}; brighter/redder = higher global patch uniqueness. "
            "Magenta paths show the top candidate scan directions."
        ),
        point_size=10.0,
    )


def show_scan1_measurement(mesh, scan, points_world, map_gt, args) -> None:
    if not args.show:
        return
    gt = mesh.sample_points_uniformly(number_of_points=8000)
    gt.transform(map_gt)
    gt.paint_uniform_color((0.55, 0.55, 0.58))
    draw_stage(
        "01 - Initial Scan1",
        [gt, make_cloud(points_world, (1.0, 0.12, 0.12)),
         make_path_line(scan.path_origins_m, (0.1, 0.65, 1.0))],
        "Gray=hidden object (simulation display only), red=Scan1, blue=sensor path.",
        point_size=4.5,
    )


def show_current_ppf_state(mesh, accepted_scans, bank, relative_state, args) -> None:
    if not args.show:
        return
    base = mesh.sample_points_uniformly(number_of_points=7000)
    geoms = accumulated_scan_geometries(accepted_scans)
    shown = bank[:min(args.visualize_max_hypotheses, len(bank))]
    for i, h in enumerate(shown):
        cad = copy.deepcopy(base)
        cad.transform(h["T_model_to_world"])
        cad.paint_uniform_color(CLUSTER_COLORS[i % len(CLUSTER_COLORS)])
        geoms.append(cad)
    draw_stage(
        f"02 - Current Bayesian pose hypotheses ({len(accepted_scans)} scan(s))",
        geoms,
        (
            f"Plausible={relative_state['plausible_count']}; active bank={len(bank)}; "
            f"W2/W1={relative_state['runnerup_relative']:.2f}; "
            f"MAP max std={relative_state['map_max_translation_std_mm']:.2f} mm / "
            f"{relative_state['map_max_rotation_std_deg']:.2f} deg; "
            f"planner={relative_state['planning_mode']}."
        ),
        point_size=3.0,
    )


def show_next_scan_candidates(
    mesh, accepted_scans, candidates, used_candidate_ids,
    selected, map_model_to_world, args,
) -> None:
    if not args.show:
        return
    base = mesh.sample_points_uniformly(number_of_points=8000)
    current = copy.deepcopy(base)
    current.transform(map_model_to_world)
    current.paint_uniform_color((0.55, 0.55, 0.58))
    geoms = [current]
    geoms.extend(accumulated_scan_geometries(accepted_scans))

    available, used = [], []
    for c in candidates:
        p = transform_points(c["point_cad_m"][None, :], map_model_to_world)[0]
        (used if int(c["candidate_id"]) in used_candidate_ids else available).append(p)
    if available:
        geoms.append(make_cloud(np.asarray(available), (0.15, 0.85, 0.85)))
    if used:
        geoms.append(make_cloud(np.asarray(used), (0.35, 0.35, 0.38)))

    center = transform_points(selected["point_cad_m"][None, :], map_model_to_world)[0]
    geoms.append(make_sphere_marker(center, max(0.0025, args.fpfh_voxel_mm * 0.0015), (1.0, 0.2, 0.75)))
    geoms.append(make_path_line(
        transform_points(selected["expected_scan_cad"].path_origins_m, map_model_to_world),
        (1.0, 0.2, 0.75),
    ))
    if selected["planner_mode"] == "mode_disambiguation":
        note = (
            f"C{selected['candidate_id']}: FPFH mode-disagreement={selected['fpfh_disagreement']:.3f}; "
            "chosen to separate remaining posterior modes."
        )
    else:
        note = (
            f"C{selected['candidate_id']}: expected information gain={selected['information_gain']:.3f}; "
            f"predicted max std={selected['predicted_max_t_std_mm']:.2f} mm / "
            f"{selected['predicted_max_r_std_deg']:.2f} deg."
        )
    draw_stage("03 - Selected next scan", geoms, note, point_size=8.0)

def show_selected_prediction_patches(mesh, selected, bank, fpfh_db, args) -> None:
    if not args.show:
        return
    base = mesh.sample_points_uniformly(number_of_points=9000)
    base.paint_uniform_color((0.55, 0.55, 0.58))
    geoms = [base]
    shown = min(args.visualize_max_hypotheses, len(bank))
    hit_count = 0
    for i in range(shown):
        patch = selected["predicted_patches"][i]
        if not patch["hit"]:
            continue
        ids = patch["point_ids"]
        if len(ids):
            geoms.append(make_cloud(
                fpfh_db["points_m"][ids],
                CLUSTER_COLORS[i % len(CLUSTER_COLORS)],
            ))
            hit_count += 1
    geoms.append(make_path_line(selected["expected_scan_cad"].path_origins_m, (1.0, 0.2, 0.75)))
    draw_stage(
        "04 - Predicted CAD patches under current hypotheses",
        geoms,
        (
            f"Same planned world action mapped back to each hypothesis CAD frame; "
            f"colored patches={hit_count}/{shown} shown hypotheses. "
            "Only FPFH descriptor differences contribute to the online score."
        ),
        point_size=5.0,
    )


def show_new_scan_measurement(
    mesh, accepted_scans, new_points_world, new_scan,
    map_model_to_world, selected, args,
) -> None:
    """Show the newly acquired scan before it is unconditionally accumulated."""
    if not args.show:
        return
    base = mesh.sample_points_uniformly(number_of_points=8000)
    current = copy.deepcopy(base)
    current.transform(map_model_to_world)
    current.paint_uniform_color((0.55, 0.55, 0.58))
    geoms = [current]
    geoms.extend(accumulated_scan_geometries(accepted_scans))
    geoms.append(make_cloud(new_points_world, (1.0, 0.85, 0.05)))
    geoms.append(make_path_line(new_scan.path_origins_m, (1.0, 0.2, 0.75)))
    draw_stage(
        "05 - New measurement (accumulate)",
        geoms,
        (
            f"C{selected['candidate_id']}: yellow=new measured scan; "
            "this scan is accumulated directly; cumulative PPF is rerun afterward."
        ),
        point_size=4.5,
    )


def show_updated_registration(mesh, accepted_scans, relative_state, args) -> None:
    if not args.show:
        return
    base = mesh.sample_points_uniformly(number_of_points=8000)
    top1 = copy.deepcopy(base)
    top1.transform(relative_state["clusters"][0].transform_model_to_world)
    top1.paint_uniform_color((0.10, 0.85, 0.25))
    geoms = accumulated_scan_geometries(accepted_scans) + [top1]
    draw_stage(
        f"06 - Bayesian posterior updated ({len(accepted_scans)} scans)",
        geoms,
        (
            f"Green=new posterior MAP; plausible={relative_state['plausible_count']}; "
            f"W2/W1={relative_state['runnerup_relative']:.2f}."
        ),
        point_size=3.5,
    )


def show_final_result(
    mesh, accepted_scans, gt_model_to_world, final_model_to_world,
    final_icp, final_error, args,
) -> None:
    if not args.show:
        return
    base = mesh.sample_points_uniformly(number_of_points=8000)
    gt = copy.deepcopy(base)
    gt.transform(gt_model_to_world)
    gt.paint_uniform_color((0.65, 0.65, 0.65))
    est = copy.deepcopy(base)
    est.transform(final_model_to_world)
    est.paint_uniform_color((0.10, 0.85, 0.25))
    geoms = [gt, est] + accumulated_scan_geometries(accepted_scans)
    draw_stage(
        "07 - Final ICP vs hidden GT",
        geoms,
        (
            f"Gray=GT, green=final ICP; error={final_error['translation_mm']:.3f} mm / "
            f"{final_error['rotation_deg']:.3f} deg; ICP RMSE={float(final_icp.inlier_rmse)*1000:.3f} mm."
        ),
        point_size=3.0,
    )


def run_cumulative_ppf(
    accepted_scans: list[dict[str, Any]],
    model: PPFModel,
    args,
):
    scan_inputs = [
        (s["points_world"], s["scan_geometry"].sensor_viewpoint_m)
        for s in accepted_scans
    ]
    scene_points, scene_normals, counts = preprocess_multiple_scans_for_ppf(
        scan_inputs, model.dist_step_m, args
    )
    print(
        "  cumulative PPF points: "
        + ", ".join(f"Scan{i+1}={n}" for i, n in enumerate(counts))
    )

    ppf_result = run_pcl_ppf_registration(
        model,
        scene_points,
        scene_normals,
        args,
    )
    clusters = pcl_ppf_result_to_clusters(ppf_result)
    return ppf_result, clusters


def save_candidate_csv(output_dir: Path, candidates: list[dict[str, Any]]) -> None:
    with (output_dir / "offline_global_unique_candidates.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        fields = [
            "candidate_id", "global_rank", "global_uniqueness", "angle_deg",
            "x_mm", "y_mm", "z_mm", "patch_point_count", "raycast_points",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for c in candidates:
            p = c["point_cad_m"] * 1000.0
            w.writerow({
                "candidate_id": c["candidate_id"],
                "global_rank": c["global_rank"],
                "global_uniqueness": c["global_uniqueness"],
                "angle_deg": c["plan"].angle_deg,
                "x_mm": p[0], "y_mm": p[1], "z_mm": p[2],
                "patch_point_count": c["patch_point_count"],
                "raycast_points": len(c["expected_scan_cad"].clean_points_m),
            })


def save_active_scan_outputs(
    output_dir: Path,
    accepted_scans: list[dict[str, Any]],
    attempt_history: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
    clusters: list[PoseCluster],
    final_relative_state: dict[str, Any],
    final_icp,
    final_model_to_world: np.ndarray,
    gt_model_to_world: np.ndarray,
    stop_reason: str,
    prior_T: np.ndarray,
    args,
    metadata: dict[str, Any],
) -> None:
    o3d = require_open3d()
    output_dir.mkdir(parents=True, exist_ok=True)
    save_candidate_csv(output_dir, candidates)

    for i, s in enumerate(accepted_scans, start=1):
        o3d.io.write_point_cloud(
            str(output_dir / f"scan_{i:02d}_world.ply"),
            make_cloud(s["points_world"]),
        )
        o3d.io.write_line_set(
            str(output_dir / f"scan_{i:02d}_sensor_path.ply"),
            make_path_line(s["scan_geometry"].path_origins_m, (0.1, 0.65, 1.0)),
        )

    if attempt_history:
        fields = sorted({k for row in attempt_history for k in row})
        with (output_dir / "scan_history.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(attempt_history)

    plausible_ranks = {r["source_cluster_rank"] for r in final_relative_state["plausible_records"]}
    posterior_by_rank = {r["source_cluster_rank"]: r for r in final_relative_state["records"]}
    with (output_dir / "final_pose_hypotheses.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        fields = [
            "ppf_rank", "ppf_vote", "posterior_weight", "plausible",
            "prior_mahalanobis", "geometric_log_likelihood",
            "expected_object_fraction", "mean_abs_normal_residual_mm",
            "median_nn_distance_mm", "merged_proposal_count", "refine_iterations",
            "max_translation_std_mm", "max_rotation_std_deg",
            "gt_translation_error_mm", "gt_rotation_error_deg",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for c in clusters:
            if c.rank not in posterior_by_rank:
                continue
            r = posterior_by_rank[c.rank]
            err = pose_error_model_to_world(r["T_model_to_world"], gt_model_to_world)
            w.writerow({
                "ppf_rank": c.rank,
                "ppf_vote": c.score,
                "posterior_weight": r["posterior_weight"],
                "plausible": int(c.rank in plausible_ranks),
                "prior_mahalanobis": r["prior_mahalanobis"],
                "geometric_log_likelihood": r["log_likelihood"],
                "expected_object_fraction": r["expected_object_fraction"],
                "mean_abs_normal_residual_mm": r["mean_abs_normal_residual_mm"],
                "median_nn_distance_mm": r["median_nn_distance_mm"],
                "merged_proposal_count": r["merged_proposal_count"],
                "refine_iterations": r["refine_iterations"],
                "max_translation_std_mm": float(np.max(r["local_std_translation_mm"])),
                "max_rotation_std_deg": float(np.max(r["local_std_rotation_deg"])),
                "gt_translation_error_mm": err["translation_mm"],
                "gt_rotation_error_deg": err["rotation_deg"],
            })

    final_error = pose_error_model_to_world(final_model_to_world, gt_model_to_world)
    report = {
        "input": str(args.cad),
        "geometry": metadata,
        "ppf_drost2010": {
            "tau_d": args.ppf_tau_d,
            "d_dist_mm": metadata["diameter_mm"] * args.ppf_tau_d,
            "n_angle": args.ppf_nangle,
            "d_angle_deg": 360.0 / args.ppf_nangle,
            "reference_fraction": args.ppf_reference_fraction,
            "reference_selection": "deterministic_uniform_after_spatial_resampling",
            "same_model_scene_sampling_distance": True,
            "normals_recomputed_after_resampling": True,
            "normal_radius_factor": args.ppf_normal_radius_factor,
            "registration_backend": "pclpybridge -> PCL PPFRegistration",
            "max_candidates": args.ppf_max_candidates,
        },
        "scan_planner": {
            "method": "random_cad_surface_point_and_random_tangent_direction",
            "fpfh_used_for_scan_selection": False,
            "information_gain_used_for_scan_selection": False,
            "scan_length_mm": args.scan_length_mm,
            "scan_step_mm": args.scan_step_mm,
            "rescan_exclusion_mm": args.rescan_exclusion_mm,
            "random_scan_max_attempts": args.random_scan_max_attempts,
        },
        "accepted_scan_count": len(accepted_scans),
        "scan_selection_count_after_scan1": len(attempt_history),
        "stop_reason": stop_reason,
        "final_active_hypothesis_count": final_relative_state["count"],
        "final_plausible_hypothesis_count": final_relative_state["plausible_count"],
        "posterior_plausible_relative_threshold": args.posterior_plausible_relative,
        "posterior_stop_runnerup_relative_threshold": args.posterior_stop_runnerup_relative,
        "final_runnerup_relative_to_map": final_relative_state["runnerup_relative"],
        "bayesian_pose_model": {
            "method": final_relative_state["method"],
            "prior_T_model_to_world": prior_T.tolist(),
            "prior_translation_sigma_mm": args.prior_translation_sigma_mm,
            "prior_rotation_sigma_deg": args.prior_rotation_sigma_deg,
            "prior_gate_mahalanobis": args.prior_gate_mahalanobis,
            "gt_generation_mode": args.gt_generation,
            "geom_sigma_normal_mm": args.geom_sigma_normal_mm,
            "geom_sigma_tangent_mm": args.geom_sigma_tangent_mm,
            "geom_object_prior": args.geom_object_prior,
            "geom_outlier_halfwidth_mm": args.geom_outlier_halfwidth_mm,
            "geom_observation_spacing_mm": args.geom_observation_spacing_mm,
            "local_refine_iterations": args.posterior_local_refine_iterations,
            "stop_translation_std_mm": args.posterior_stop_translation_std_mm,
            "stop_rotation_std_deg": args.posterior_stop_rotation_std_deg,
            "ppf_votes_used_as_probability": False,
            "component_model": "SE3 Gaussian mixture; Laplace/Gauss-Newton local covariance",
        },
        "ground_truth": {"T_model_to_world": gt_model_to_world.tolist()},
        "final_posterior_map": {
            "T_model_to_world": final_relative_state["clusters"][0].transform_model_to_world.tolist(),
            "posterior_weight": final_relative_state["active_records"][0]["posterior_weight"],
            "source_ppf_rank": final_relative_state["active_records"][0]["source_cluster_rank"],
            "local_covariance_6x6": final_relative_state["active_records"][0]["covariance_6x6"].tolist(),
            "local_std_translation_mm": final_relative_state["active_records"][0]["local_std_translation_mm"].tolist(),
            "local_std_rotation_deg": final_relative_state["active_records"][0]["local_std_rotation_deg"].tolist(),
            "mode_separated": final_relative_state["mode_separated"],
            "local_confident": final_relative_state["local_confident"],
            "gt_error": pose_error_model_to_world(
                final_relative_state["clusters"][0].transform_model_to_world, gt_model_to_world
            ),
        },
        "posterior_local_covariance_note": (
            "Each retained PPF basin is a Gaussian component (T_i, Sigma_i, w_i). "
            "Sigma_i is a robust Gauss-Newton/Laplace covariance from the geometric "
            "likelihood plus the SE(3) prior; PPF votes are not used as probabilities."
        ),
        "final_icp": {
            "T_model_to_world": final_model_to_world.tolist(),
            "fitness": float(final_icp.fitness),
            "inlier_rmse_mm": float(final_icp.inlier_rmse) * 1000.0,
            "gt_error": final_error,
        },
    }
    with (output_dir / "active_scan_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


def sample_valid_random_scan(
    surface_points: np.ndarray,
    surface_normals: np.ndarray,
    cad_scene,
    world_scene,
    aim_model_to_world: np.ndarray,
    previous_plans: list[RandomScanPlan],
    args,
    rng: np.random.Generator,
) -> tuple[RandomScanPlan, ScanGeometry, np.ndarray, int]:
    """Draw a random CAD surface point + random tangent sweep direction.

    There is deliberately no FPFH, Fisher-information, uniqueness, or posterior-driven
    scan selection here.  The only rejection rules are operational sanity checks:
      1) optional minimum CAD-center spacing from earlier scans, and
      2) the simulated measurement must contain at least --min-new-scan-points.

    --scan-length-mm directly controls the linear sweep length used by
    simulate_scan_from_plan_world().
    """
    exclude_m = args.rescan_exclusion_mm / 1000.0
    last_reason = ""

    for draw_id in range(1, args.random_scan_max_attempts + 1):
        plan = random_scan_plan_from_surface_pool(
            surface_points,
            surface_normals,
            cad_scene,
            args,
            rng,
        )

        if previous_plans and exclude_m > 0.0:
            dmin = min(
                float(np.linalg.norm(plan.point_cad_m - prev.point_cad_m))
                for prev in previous_plans
            )
            if dmin < exclude_m:
                last_reason = f"too close to previous scan ({dmin*1000.0:.2f} mm)"
                continue

        scan = simulate_scan_from_plan_world(
            world_scene,
            plan,
            aim_model_to_world,
            args,
        )
        points_world = noisy_scan_points(scan, args, rng)
        if len(points_world) < args.min_new_scan_points:
            last_reason = f"only {len(points_world)} measured points"
            continue

        return plan, scan, points_world, draw_id

    raise RuntimeError(
        f"Could not draw a usable random scan after {args.random_scan_max_attempts} attempts"
        + (f": {last_reason}" if last_reason else "")
    )


def main() -> int:
    args = parse_args()
    validate_args(args)

    output_dir = (
        args.output_dir
        if args.output_dir is not None
        else Path(f"{args.cad.stem}_ppf_bayesian_gmm_random_scan")
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print("[setup 1/3] Load CAD")
    mesh, metadata = load_centered_mesh(args.cad, args.mesh_unit)
    diameter_m = metadata["diameter_mm"] / 1000.0
    cad_scene = build_raycast_scene(mesh)

    print("[setup 2/3] Prepare PCL PPF model")
    model_points, model_normals = sample_oriented_cad_drost(
        mesh, args.cad_sample_points, diameter_m, args
    )
    model = build_ppf_model(model_points, model_normals, diameter_m, args)

    print("[setup 3/3] Build ICP + sensor-agnostic geometric-likelihood target")
    target_cloud = make_icp_target(mesh, args)
    geom_db = prepare_geometric_likelihood_target(target_cloud)
    random_surface_points = geom_db["points_m"]
    random_surface_normals = geom_db["normals"]
    prior_T = load_prior_pose(args)
    print(
        f"  prior sigma: {args.prior_translation_sigma_mm:.1f} mm / "
        f"{args.prior_rotation_sigma_deg:.1f} deg | "
        f"random scan length={args.scan_length_mm:.1f} mm | "
        f"surface pool={len(random_surface_points):,} points"
    )
    print("  next-scan policy: RANDOM surface point + RANDOM tangent direction (no FPFH selection)")

    # Hidden GT is simulation-only and is generated around the supplied initial pose.
    # The estimator never receives GT; it only receives prior_T and its covariance.
    gt_model_to_world = random_gt_transform(prior_T, args, rng)
    init_err = pose_error_model_to_world(prior_T, gt_model_to_world)
    print(
        f"  supplied initial pose -> hidden GT error: "
        f"{init_err['translation_mm']:.2f} mm / {init_err['rotation_deg']:.2f} deg "
        f"(GT mode={args.gt_generation})"
    )
    print(f"  prior hard gate Mahalanobis radius: {args.prior_gate_mahalanobis:.2f}")
    world_mesh = copy.deepcopy(mesh)
    world_mesh.transform(gt_model_to_world)
    world_scene = build_raycast_scene(world_mesh)

    print("\n" + "=" * 96)
    print("SCAN 1 - RANDOM INITIAL MEASUREMENT")
    print("=" * 96)
    first_plan, first_scan, first_points_world, first_draws = sample_valid_random_scan(
        random_surface_points,
        random_surface_normals,
        cad_scene,
        world_scene,
        prior_T,
        [],
        args,
        rng,
    )
    print(
        f"  random draw={first_draws}, length={args.scan_length_mm:.1f} mm, "
        f"angle={first_plan.angle_deg:.1f} deg, points={len(first_points_world)}"
    )
    show_scan1_measurement(mesh, first_scan, first_points_world, gt_model_to_world, args)

    accepted_scans: list[dict[str, Any]] = [{
        "scan_index": 1,
        "points_world": first_points_world,
        "scan_geometry": first_scan,
        "plan": first_plan,
        "candidate_id": None,
    }]
    previous_plans: list[RandomScanPlan] = [first_plan]
    attempt_history: list[dict[str, Any]] = []

    ppf_round = 1
    hypotheses, clusters = run_cumulative_ppf(accepted_scans, model, args)

    stop_reason = ""
    final_relative_state: dict[str, Any] | None = None

    while True:
        accepted_count = len(accepted_scans)
        cumulative_points_world = np.vstack([s["points_world"] for s in accepted_scans])
        relative_state = build_posterior_state(
            clusters, cumulative_points_world, prior_T, geom_db, args
        )
        final_relative_state = relative_state
        bank = build_hypothesis_bank(relative_state)

        print("\n" + "-" * 96)
        print(
            f"STATE AFTER {accepted_count} ACCEPTED SCAN(S): "
            f"PPF-proposals={len(clusters)}, posterior-plausible={relative_state['plausible_count']}, "
            f"active-bank={relative_state['count']}, "
            f"W2/W1={relative_state['runnerup_relative']:.3f}, "
            f"MAP-std={relative_state['map_max_translation_std_mm']:.2f} mm/"
            f"{relative_state['map_max_rotation_std_deg']:.2f} deg, "
            f"prior-rejected={relative_state['ppf_rejected_by_prior_gate']}, planner=random"
        )
        print("-" * 96)
        for h in bank:
            c = relative_state["clusters"][h["bank_id"] - 1]
            err = pose_error_model_to_world(c.transform_model_to_world, gt_model_to_world)
            print(
                f"  H{h['bank_id']:02d} = cluster#{c.rank:02d}: "
                f"PPFvote={c.score:.0f}, posterior={h['posterior_weight']:.4f}, "
                f"priorD={h['prior_mahalanobis']:.2f}, obj={h['expected_object_fraction']:.2f}, "
                f"|e_n|={h['mean_abs_normal_residual_mm']:.2f} mm, "
                f"stdMax={np.max(h['local_std_translation_mm']):.2f} mm/"
                f"{np.max(h['local_std_rotation_deg']):.2f} deg, "
                f"merged={h['merged_proposal_count']}, refine={h['refine_iterations']}, "
                f"GT={err['translation_mm']:.2f} mm/{err['rotation_deg']:.2f} deg"
            )

        plot_posterior_support(output_dir, relative_state, ppf_round, args)
        show_current_ppf_state(mesh, accepted_scans, bank, relative_state, args)

        if relative_state["stop_ready"]:
            stop_reason = (
                f"POSTERIOR_CONFIDENT:W2/W1={relative_state['runnerup_relative']:.3f}"
                f"<{args.posterior_stop_runnerup_relative:.3f},"
                f"stdT={relative_state['map_max_translation_std_mm']:.3f}"
                f"<={args.posterior_stop_translation_std_mm:.3f}mm,"
                f"stdR={relative_state['map_max_rotation_std_deg']:.3f}"
                f"<={args.posterior_stop_rotation_std_deg:.3f}deg"
            )
            print(f"\nSTOP: {stop_reason}")
            break
        if accepted_count >= args.max_scans:
            stop_reason = f"MAX_SCANS:{args.max_scans}"
            print(f"\nSTOP: {stop_reason}")
            break

        map_T = bank[0]["T_model_to_world"]
        attempt_id = len(attempt_history) + 1
        next_scan_number = accepted_count + 1

        try:
            random_plan, new_scan, new_points_world, random_draws = sample_valid_random_scan(
                random_surface_points,
                random_surface_normals,
                cad_scene,
                world_scene,
                map_T,
                previous_plans,
                args,
                rng,
            )
        except RuntimeError as exc:
            stop_reason = f"NO_USABLE_RANDOM_SCAN:{exc}"
            print(f"\nSTOP: {stop_reason}")
            break

        selected = {
            "candidate_id": next_scan_number,
            "planner_mode": "random",
            "plan": random_plan,
        }
        print(
            f"\n  Random Scan{next_scan_number}: draw={random_draws}, "
            f"length={args.scan_length_mm:.1f} mm, angle={random_plan.angle_deg:.1f} deg, "
            f"points={len(new_points_world)}"
        )

        show_new_scan_measurement(
            mesh,
            accepted_scans,
            new_points_world,
            new_scan,
            map_T,
            selected,
            args,
        )

        history_row = {
            "attempt_index": attempt_id,
            "accepted_scans_before": accepted_count,
            "plausible_hypotheses_before": relative_state["plausible_count"],
            "active_hypotheses_before": relative_state["count"],
            "runnerup_relative_before": relative_state["runnerup_relative"],
            "planner_mode": "random",
            "map_max_translation_std_mm_before": relative_state["map_max_translation_std_mm"],
            "map_max_rotation_std_deg_before": relative_state["map_max_rotation_std_deg"],
            "random_draws_needed": random_draws,
            "scan_length_mm": args.scan_length_mm,
            "scan_angle_deg": random_plan.angle_deg,
            "scan_center_x_mm": float(random_plan.point_cad_m[0] * 1000.0),
            "scan_center_y_mm": float(random_plan.point_cad_m[1] * 1000.0),
            "scan_center_z_mm": float(random_plan.point_cad_m[2] * 1000.0),
            "measured_points": len(new_points_world),
        }

        accepted_scans.append({
            "scan_index": len(accepted_scans) + 1,
            "points_world": new_points_world,
            "scan_geometry": new_scan,
            "plan": random_plan,
            "candidate_id": None,
        })
        previous_plans.append(random_plan)
        history_row["accepted_scan_index"] = len(accepted_scans)
        attempt_history.append(history_row)

        print(
            f"  -> accumulated as Scan{len(accepted_scans)}; "
            "rerun cumulative PPF from scratch"
        )
        ppf_round += 1
        hypotheses, clusters = run_cumulative_ppf(accepted_scans, model, args)
        cumulative_points_world = np.vstack([s["points_world"] for s in accepted_scans])
        updated_relative = build_posterior_state(
            clusters, cumulative_points_world, prior_T, geom_db, args
        )
        show_updated_registration(mesh, accepted_scans, updated_relative, args)

    cumulative_points_world = np.vstack([s["points_world"] for s in accepted_scans])
    final_relative_state = build_posterior_state(
        clusters, cumulative_points_world, prior_T, geom_db, args
    )
    final_map_T = final_relative_state["clusters"][0].transform_model_to_world
    final_map_error = pose_error_model_to_world(final_map_T, gt_model_to_world)
    final_icp, final_T = refine_cumulative_icp(
        cumulative_points_world, target_cloud, final_map_T, args
    )
    final_error = pose_error_model_to_world(final_T, gt_model_to_world)

    print("\n" + "=" * 96)
    print("FINAL RESULT")
    print("=" * 96)
    print(f"accepted scans        : {len(accepted_scans)}")
    print(f"random scan length    : {args.scan_length_mm:.1f} mm")
    print(f"next-scan selections  : {len(attempt_history)} (all RANDOM)")
    print(f"stop reason           : {stop_reason}")
    print(f"plausible hypotheses  : {final_relative_state['plausible_count']}")
    print(f"active hypothesis bank: {final_relative_state['count']}")
    print(f"W2 / W1              : {final_relative_state['runnerup_relative']:.3f}")
    print(f"plausible threshold   : {args.posterior_plausible_relative:.2f} x MAP weight")
    print(f"stop W2/W1 threshold  : {args.posterior_stop_runnerup_relative:.2f}")
    print(
        f"stop local std thresh : {args.posterior_stop_translation_std_mm:.2f} mm / "
        f"{args.posterior_stop_rotation_std_deg:.2f} deg"
    )
    print(
        f"Posterior MAP pre-ICP : {final_map_error['translation_mm']:.3f} mm / "
        f"{final_map_error['rotation_deg']:.3f} deg"
    )
    final_bank = build_hypothesis_bank(final_relative_state)
    Sigma_local = posterior_local_covariance(final_bank)
    std_t_mm = np.sqrt(np.maximum(np.diag(Sigma_local)[:3], 0.0)) * 1000.0
    std_r_deg = np.degrees(np.sqrt(np.maximum(np.diag(Sigma_local)[3:], 0.0)))
    print(
        "MAP component local std: "
        f"t=[{std_t_mm[0]:.2f},{std_t_mm[1]:.2f},{std_t_mm[2]:.2f}] mm, "
        f"r=[{std_r_deg[0]:.2f},{std_r_deg[1]:.2f},{std_r_deg[2]:.2f}] deg"
    )
    Sigma_gmm = posterior_gmm_covariance(final_bank)
    gmm_t_mm = np.sqrt(np.maximum(np.diag(Sigma_gmm)[:3], 0.0)) * 1000.0
    gmm_r_deg = np.degrees(np.sqrt(np.maximum(np.diag(Sigma_gmm)[3:], 0.0)))
    print(
        "active GMM total std   : "
        f"t=[{gmm_t_mm[0]:.2f},{gmm_t_mm[1]:.2f},{gmm_t_mm[2]:.2f}] mm, "
        f"r=[{gmm_r_deg[0]:.2f},{gmm_r_deg[1]:.2f},{gmm_r_deg[2]:.2f}] deg"
    )
    print(
        f"Final ICP             : {final_error['translation_mm']:.3f} mm / "
        f"{final_error['rotation_deg']:.3f} deg"
    )
    print(
        f"ICP fitness/RMSE      : {float(final_icp.fitness):.3f} / "
        f"{float(final_icp.inlier_rmse)*1000.0:.3f} mm"
    )

    show_final_result(
        mesh, accepted_scans, gt_model_to_world, final_T,
        final_icp, final_error, args,
    )
    # Empty candidate list is intentional: FPFH/offline scan selection is disabled.
    save_active_scan_outputs(
        output_dir,
        accepted_scans,
        attempt_history,
        [],
        clusters,
        final_relative_state,
        final_icp,
        final_T,
        gt_model_to_world,
        stop_reason,
        prior_T,
        args,
        metadata,
    )
    print(f"outputs: {output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)