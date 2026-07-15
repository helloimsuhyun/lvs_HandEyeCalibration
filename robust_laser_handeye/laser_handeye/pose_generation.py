from __future__ import annotations

import numpy as np

from .scene_generation import normalize_vector, plane_basis
from .se3 import inv_T, make_T


def sample_random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Sample a uniform random rotation matrix in SO(3)."""
    quaternion = rng.normal(size=4)
    quaternion_norm = float(np.linalg.norm(quaternion))
    if quaternion_norm <= 0.0:
        raise RuntimeError("failed to sample a valid quaternion")

    w, x, y, z = quaternion / quaternion_norm
    return np.array(
        [
            [
                1.0 - 2.0 * (y * y + z * z),
                2.0 * (x * y - z * w),
                2.0 * (x * z + y * w),
            ],
            [
                2.0 * (x * y + z * w),
                1.0 - 2.0 * (x * x + z * z),
                2.0 * (y * z - x * w),
            ],
            [
                2.0 * (x * z - y * w),
                2.0 * (y * z + x * w),
                1.0 - 2.0 * (x * x + y * y),
            ],
        ],
        dtype=float,
    )


def sample_sensor_pose_for_plane(
    plane_n: np.ndarray,
    plane_l: float,
    rng: np.random.Generator,
    tangent_range_mm: float = 220.0,
    depth_range_mm: tuple[float, float] = (60.0, 150.0),
    min_view_dot: float = 0.0,
    max_trials: int = 200,
) -> np.ndarray:
    """Sample a sensor pose ``T_base_s`` that observes a plane."""
    plane_n = normalize_vector(plane_n)
    plane_l = float(plane_l)
    depth_min, depth_max = map(float, depth_range_mm)

    if depth_min <= 0.0 or depth_max <= depth_min:
        raise ValueError("invalid depth_range_mm")
    if tangent_range_mm < 0.0:
        raise ValueError("tangent_range_mm must be non-negative")
    if not -1.0 <= min_view_dot <= 1.0:
        raise ValueError("min_view_dot must be in [-1, 1]")
    if max_trials <= 0:
        raise ValueError("max_trials must be positive")

    u, v = plane_basis(plane_n)
    plane_center = plane_n * plane_l
    target = (
        plane_center
        + rng.uniform(-tangent_range_mm, tangent_range_mm) * u
        + rng.uniform(-tangent_range_mm, tangent_range_mm) * v
    )

    R_base_s: np.ndarray | None = None
    for _ in range(max_trials):
        candidate = sample_random_rotation(rng)
        if float(candidate[:, 2] @ (-plane_n)) > min_view_dot:
            R_base_s = candidate
            break

    if R_base_s is None:
        raise RuntimeError("failed to generate a sensor rotation facing the plane")

    depth_mm = float(rng.uniform(depth_min, depth_max))
    sensor_origin = target - depth_mm * R_base_s[:, 2]
    return make_T(R_base_s, sensor_origin)


def sensor_pose_to_robot_pose(
    T_base_s: np.ndarray,
    T_ef_s: np.ndarray,
) -> np.ndarray:
    """Convert sensor pose to flange pose using ``T_base_s = T_base_ef @ T_ef_s``."""
    T_base_s = np.asarray(T_base_s, dtype=float).reshape(4, 4)
    T_ef_s = np.asarray(T_ef_s, dtype=float).reshape(4, 4)
    return T_base_s @ inv_T(T_ef_s)


def sample_robot_pose_for_plane(
    T_ef_s: np.ndarray,
    plane_n: np.ndarray,
    plane_l: float,
    rng: np.random.Generator,
    tangent_range_mm: float = 220.0,
    depth_range_mm: tuple[float, float] = (60.0, 150.0),
    min_view_dot: float = 0.0,
    max_trials: int = 200,
) -> np.ndarray:
    """Sample one flange pose whose mounted sensor observes the plane."""
    T_base_s = sample_sensor_pose_for_plane(
        plane_n=plane_n,
        plane_l=plane_l,
        rng=rng,
        tangent_range_mm=tangent_range_mm,
        depth_range_mm=depth_range_mm,
        min_view_dot=min_view_dot,
        max_trials=max_trials,
    )
    return sensor_pose_to_robot_pose(T_base_s, T_ef_s)