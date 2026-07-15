from __future__ import annotations

import numpy as np
from .se3 import euler_xyz_deg
from scipy.spatial.transform import Rotation

Plane = tuple[np.ndarray, float]


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    """Return a normalized three-dimensional vector."""
    vector = np.asarray(vector, dtype=float).reshape(3)
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        raise ValueError("cannot normalize a zero vector")
    return vector / norm


def plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return two orthonormal directions lying on a plane."""
    normal = normalize_vector(normal)
    reference = np.array([0.0, 0.0, 1.0])

    if abs(float(normal @ reference)) > 0.9:
        reference = np.array([0.0, 1.0, 0.0])

    u = normalize_vector(np.cross(normal, reference))
    v = normalize_vector(np.cross(normal, u))
    return u, v


def make_plane(normal: np.ndarray, distance_mm: float) -> Plane:
    """Create a canonical plane n.T @ p = l with unit normal and l >= 0."""
    normal = normalize_vector(normal)
    distance_mm = float(distance_mm)

    if not np.isfinite(distance_mm):
        raise ValueError("distance_mm must be finite")

    if distance_mm < 0.0:
        normal = -normal
        distance_mm = -distance_mm

    return normal, distance_mm


def _angle_to_nearest_base_axis_deg(normal: np.ndarray) -> float:
    """Return the acute angle to the nearest ±X/±Y/±Z base axis."""
    normal = normalize_vector(normal)

    # abs(component) gives cosine to the nearest signed base axis.
    max_abs_component = float(np.max(np.abs(normal)))
    max_abs_component = float(np.clip(max_abs_component, -1.0, 1.0))

    return float(np.degrees(np.arccos(max_abs_component)))


def _sample_non_axis_aligned_normal(
    rng: np.random.Generator,
    min_axis_angle_deg: float,
    max_trials: int = 10_000,
) -> np.ndarray:
    """Sample a random unit normal not aligned with any base axis.

    A candidate is rejected when its acute angle to any of the signed base
    axes ±X, ±Y, ±Z is smaller than ``min_axis_angle_deg``.
    """
    min_axis_angle_deg = float(min_axis_angle_deg)

    if not 0.0 <= min_axis_angle_deg < 54.7356:
        raise ValueError(
            "min_axis_angle_deg must be in [0, 54.7356) degrees"
        )

    for _ in range(max_trials):
        normal = normalize_vector(rng.normal(size=3))

        if _angle_to_nearest_base_axis_deg(normal) >= min_axis_angle_deg:
            return normal

    raise RuntimeError(
        "failed to sample a plane normal satisfying the base-axis constraint"
    )


def _perturb_normal(
    normal: np.ndarray,
    rng: np.random.Generator,
    max_jitter_deg: float,
) -> np.ndarray:
    """Apply a small random rotation to a unit normal."""
    normal = normalize_vector(normal)
    max_jitter_deg = float(max_jitter_deg)

    if max_jitter_deg <= 0.0:
        return normal

    axis = normalize_vector(rng.normal(size=3))
    angle_deg = float(rng.uniform(-max_jitter_deg, max_jitter_deg))

    rotation = Rotation.from_rotvec(
        np.deg2rad(angle_deg) * axis
    ).as_matrix()

    return normalize_vector(rotation @ normal)




def make_three_planes(
    rng: np.random.Generator,
    distance_range_mm: tuple[float, float] = (650.0, 1000.0),
    min_axis_angle_deg: float = 1.0,
) -> list[Plane]:
    distance_min, distance_max = map(float, distance_range_mm)

    if distance_min <= 0.0 or distance_max <= distance_min:
        raise ValueError("invalid distance_range_mm")

    if min_axis_angle_deg < 0.0:
        raise ValueError("min_axis_angle_deg must be non-negative")

    cos_threshold = float(np.cos(np.deg2rad(min_axis_angle_deg)))

    # 첫 번째 평면의 랜덤 법선 생성
    while True:
        n0 = normalize_vector(rng.normal(size=3))

        # ±X, ±Y, ±Z 중 하나와 1도 이내로 정렬되면 제외
        nearest_axis_cos = float(np.max(np.abs(n0)))

        if nearest_axis_cos < cos_threshold:
            break

    # 첫 번째 평면 기준 직교 방향 2개
    n1, n2 = plane_basis(n0)

    normals = [n0, n1, n2]

    # 각 평면 위치만 독립 랜덤
    distances = rng.uniform(
        distance_min,
        distance_max,
        size=3,
    )

    return [
        make_plane(normal, distance_mm)
        for normal, distance_mm in zip(normals, distances)
    ]


def make_random_plane(
    rng: np.random.Generator,
    distance_range_mm: tuple[float, float] = (350.0, 550.0),
    min_axis_angle_deg: float = 1.0,
) -> Plane:
    """Create one random plane not aligned with any robot-base axis."""
    distance_min, distance_max = map(float, distance_range_mm)

    if distance_min <= 0.0 or distance_max <= distance_min:
        raise ValueError("invalid distance_range_mm")

    normal = _sample_non_axis_aligned_normal(
        rng=rng,
        min_axis_angle_deg=min_axis_angle_deg,
    )

    return make_plane(
        normal=normal,
        distance_mm=rng.uniform(distance_min, distance_max),
    )


def make_plane_from_point(
    normal: np.ndarray,
    point_on_plane_mm: np.ndarray,
) -> Plane:
    """Create a plane from its normal and one point lying on it."""
    normal = normalize_vector(normal)
    point_on_plane_mm = np.asarray(
        point_on_plane_mm,
        dtype=float,
    ).reshape(3)

    if not np.all(np.isfinite(point_on_plane_mm)):
        raise ValueError(
            "point_on_plane_mm must contain only finite values"
        )

    return make_plane(
        normal,
        float(normal @ point_on_plane_mm),
    )