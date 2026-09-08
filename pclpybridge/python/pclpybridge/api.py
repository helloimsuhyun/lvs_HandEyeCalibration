from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from . import _core


def _as_points(x: Any) -> np.ndarray:
    """Accept Nx3 NumPy-like data or an Open3D legacy PointCloud."""
    if hasattr(x, "points"):
        x = np.asarray(x.points)
    a = np.asarray(x, dtype=np.float32)
    if a.ndim != 2 or a.shape[1] != 3:
        raise ValueError(f"expected points with shape (N,3), got {a.shape}")
    return np.ascontiguousarray(a)


def _normals_from_cloud_or_value(cloud: Any, normals: Any | None) -> np.ndarray | None:
    if normals is not None:
        a = np.asarray(normals, dtype=np.float32)
        if a.ndim != 2 or a.shape[1] != 3:
            raise ValueError(f"expected normals with shape (N,3), got {a.shape}")
        return np.ascontiguousarray(a)

    # Open3D legacy PointCloud: use existing normals if present.
    if hasattr(cloud, "has_normals") and cloud.has_normals():
        a = np.asarray(cloud.normals, dtype=np.float32)
        if len(a):
            return np.ascontiguousarray(a)
    return None


def _viewpoint(v: Any | None):
    if v is None:
        return None
    a = np.asarray(v, dtype=np.float32).reshape(-1)
    if a.size != 3:
        raise ValueError("viewpoint must be length 3")
    return np.ascontiguousarray(a)


def voxel_downsample(points: Any, leaf_size: float) -> np.ndarray:
    """PCL VoxelGrid. All distances use the same unit as the input coordinates."""
    return _core.voxel_downsample(_as_points(points), float(leaf_size))


def estimate_normals(
    points: Any,
    radius: float,
    *,
    viewpoint: Any | None = None,
    threads: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """PCL NormalEstimationOMP -> (normals[N,3], curvature[N])."""
    d = _core.estimate_normals(
        _as_points(points),
        float(radius),
        _viewpoint(viewpoint),
        int(threads),
    )
    return np.asarray(d["normals"]), np.asarray(d["curvature"])


def harris3d(
    points: Any,
    radius: float,
    *,
    threshold: float = 0.0,
    nonmax: bool = True,
    refine: bool = False,
    method: str = "HARRIS",
    normals: Any | None = None,
    threads: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """PCL HarrisKeypoint3D -> (keypoints[N,3], response[N])."""
    pts = _as_points(points)
    nrm = _normals_from_cloud_or_value(points, normals)
    d = _core.harris3d(
        pts,
        float(radius),
        float(threshold),
        bool(nonmax),
        bool(refine),
        str(method).upper(),
        nrm,
        int(threads),
    )
    return np.asarray(d["points"]), np.asarray(d["response"])


def iss3d(
    points: Any,
    salient_radius: float,
    nonmax_radius: float,
    *,
    gamma21: float = 0.975,
    gamma32: float = 0.975,
    min_neighbors: int = 5,
    threads: int = 0,
) -> np.ndarray:
    """PCL ISSKeypoint3D."""
    return np.asarray(
        _core.iss3d(
            _as_points(points),
            float(salient_radius),
            float(nonmax_radius),
            float(gamma21),
            float(gamma32),
            int(min_neighbors),
            int(threads),
        )
    )


def sift3d(
    points: Any,
    intensity: Any | None = None,
    *,
    min_scale: float,
    n_octaves: int = 3,
    n_scales_per_octave: int = 4,
    min_contrast: float = 0.001,
) -> np.ndarray:
    """PCL SIFTKeypoint -> [x,y,z,scale].

    Important: PCL's SIFTKeypoint is intensity-based. If `points` is an Nx4
    NumPy array, column 4 is used as intensity when `intensity=None`.
    """
    raw = np.asarray(points)
    if intensity is None and raw.ndim == 2 and raw.shape[1] == 4:
        intensity = raw[:, 3]
        points = raw[:, :3]
    if intensity is None:
        raise ValueError(
            "PCL SIFT3D requires intensity. Pass intensity=(N,) or an Nx4 array."
        )
    pts = _as_points(points)
    it = np.asarray(intensity, dtype=np.float32).reshape(-1)
    if len(it) != len(pts):
        raise ValueError("intensity length must equal number of points")
    return np.asarray(
        _core.sift3d(
            pts,
            np.ascontiguousarray(it),
            float(min_scale),
            int(n_octaves),
            int(n_scales_per_octave),
            float(min_contrast),
        )
    )


def fpfh(
    points: Any,
    radius: float,
    *,
    normals: Any | None = None,
    normal_radius: float = 0.005,
    viewpoint: Any | None = None,
    threads: int = 0,
) -> np.ndarray:
    """PCL FPFHEstimationOMP -> descriptors[N,33]."""
    return np.asarray(
        _core.fpfh(
            _as_points(points),
            float(radius),
            _normals_from_cloud_or_value(points, normals),
            float(normal_radius),
            _viewpoint(viewpoint),
            int(threads),
        )
    )


def shot(
    points: Any,
    radius: float,
    *,
    normals: Any | None = None,
    normal_radius: float = 0.005,
    viewpoint: Any | None = None,
) -> np.ndarray:
    """PCL SHOTEstimation -> descriptors[N,352]."""
    return np.asarray(
        _core.shot(
            _as_points(points),
            float(radius),
            _normals_from_cloud_or_value(points, normals),
            float(normal_radius),
            _viewpoint(viewpoint),
        )
    )


@dataclass(frozen=True)
class PPFResult:
    """PCL PPFRegistration result.

    `transform` and every row in `transforms` map MODEL -> SCENE.
    `votes[k]` is PCL's clustered vote count for `transforms[k]`.
    """
    converged: bool
    transform: np.ndarray
    transforms: np.ndarray
    votes: np.ndarray
    model_feature_count: int
    model_point_count: int
    scene_point_count: int
    candidate_backend: str

    @property
    def top_transform(self) -> np.ndarray:
        return self.transforms[0] if len(self.transforms) else self.transform

    @property
    def top_votes(self) -> int:
        return int(self.votes[0]) if len(self.votes) else 0

    @property
    def relative_votes(self) -> np.ndarray:
        if not len(self.votes) or self.votes[0] == 0:
            return np.zeros_like(self.votes, dtype=np.float64)
        return self.votes.astype(np.float64) / float(self.votes[0])


def ppf_register(
    model_points: Any,
    scene_points: Any,
    *,
    model_normals: Any | None = None,
    scene_normals: Any | None = None,
    normal_radius: float = 0.005,
    model_viewpoint: Any | None = None,
    scene_viewpoint: Any | None = None,
    angle_step_deg: float = 12.0,
    distance_step: float = 0.01,
    scene_reference_rate: int = 5,
    position_cluster_threshold: float = 0.01,
    rotation_cluster_threshold_deg: float = 20.0,
    max_candidates: int = 20,
    threads: int = 0,
) -> PPFResult:
    """Run PCL PPFRegistration and return ranked pose candidates + votes.

    Distance parameters use the same unit as point coordinates.
    For meter point clouds, e.g. distance_step=0.005 means 5 mm.

    For 2-D laser data, supplying `scene_normals` explicitly is recommended,
    because normals from different scans can be oriented using each scan's own
    sensor viewpoint before concatenation.
    """
    d = _core.ppf_register(
        _as_points(model_points),
        _as_points(scene_points),
        _normals_from_cloud_or_value(model_points, model_normals),
        _normals_from_cloud_or_value(scene_points, scene_normals),
        float(normal_radius),
        _viewpoint(model_viewpoint),
        _viewpoint(scene_viewpoint),
        float(angle_step_deg),
        float(distance_step),
        int(scene_reference_rate),
        float(position_cluster_threshold),
        float(rotation_cluster_threshold_deg),
        int(max_candidates),
        int(threads),
    )
    return PPFResult(
        converged=bool(d["converged"]),
        transform=np.asarray(d["transform"], dtype=np.float64),
        transforms=np.asarray(d["transforms"], dtype=np.float64),
        votes=np.asarray(d["votes"], dtype=np.uint32),
        model_feature_count=int(d["model_feature_count"]),
        model_point_count=int(d["model_point_count"]),
        scene_point_count=int(d["scene_point_count"]),
        candidate_backend=str(d["candidate_backend"]),
    )
