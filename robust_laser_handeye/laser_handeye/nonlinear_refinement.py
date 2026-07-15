from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .data import LaserScan
from .geometry import fit_plane_pca
from .se3 import make_T, transform_points


Plane = tuple[np.ndarray, float]
PlaneMode = Literal["refit", "fixed"]
RobustLoss = Literal["linear", "soft_l1", "huber", "cauchy", "arctan"]
ScanGroups = Mapping[int, Sequence[LaserScan]] | Sequence[Sequence[LaserScan]]
PlaneGroups = Mapping[int, Plane] | Sequence[Plane]


@dataclass
class NonlinearRefinementResult:
    """Result of the six-parameter SE(3) nonlinear least-squares refinement."""

    T_ef_s: np.ndarray
    success: bool
    status: int
    message: str
    nfev: int
    njev: int | None
    cost: float
    optimality: float
    initial_rms_mm: float
    final_rms_mm: float
    delta_rotation_deg: float
    delta_translation_mm: float
    jacobian_rank: int
    jacobian_condition: float
    local_update: np.ndarray


def refine_handeye_nonlinear(
    scans_by_plane: ScanGroups,
    T_init: np.ndarray,
    *,
    plane_mode: PlaneMode = "refit",
    planes: PlaneGroups | None = None,
    loss: RobustLoss = "linear",
    f_scale_mm: float = 1.0,
    max_nfev: int = 200,
    ftol: float = 1e-10,
    xtol: float = 1e-10,
    gtol: float = 1e-10,
) -> NonlinearRefinementResult:
    """Refine a hand-eye transform using six-dimensional nonlinear least squares.

    The optimized variable is a local six-vector::

        x = [d_rx, d_ry, d_rz, d_tx, d_ty, d_tz]

    where the rotation part is a rotation vector in radians and the translation
    part is in millimetres. Every candidate remains on SE(3) through::

        T_candidate = T_init @ T_delta(x)

    ``plane_mode='refit'`` is intended for unknown-plane calibration. For every
    candidate hand-eye transform, each physical plane is re-estimated by PCA and
    the signed point-to-fitted-plane distances are returned as residuals. Plane
    parameters are therefore eliminated analytically and only six hand-eye
    parameters are optimized.

    ``plane_mode='fixed'`` uses the supplied plane equations ``n.T @ p = l``.
    This is appropriate when the planes are independently known.
    """
    groups = _normalize_scan_groups(scans_by_plane)
    T_reference = _validate_transform(T_init, "T_init")

    if plane_mode not in ("refit", "fixed"):
        raise ValueError("plane_mode must be 'refit' or 'fixed'")

    fixed_planes: dict[int, Plane] | None = None
    if plane_mode == "fixed":
        if planes is None:
            raise ValueError("planes are required when plane_mode='fixed'")
        fixed_planes = _normalize_planes(planes, groups)
    elif planes is not None:
        raise ValueError("planes must be omitted when plane_mode='refit'")

    f_scale_mm = float(f_scale_mm)
    if not np.isfinite(f_scale_mm) or f_scale_mm <= 0.0:
        raise ValueError("f_scale_mm must be finite and positive")
    if max_nfev <= 0:
        raise ValueError("max_nfev must be positive")

    def residual_function(local_update: np.ndarray) -> np.ndarray:
        T_candidate = apply_local_se3_update(T_reference, local_update)
        return point_to_plane_residuals(
            groups,
            T_candidate,
            plane_mode=plane_mode,
            fixed_planes=fixed_planes,
        )

    x0 = np.zeros(6, dtype=float)
    residual_initial = residual_function(x0)
    if residual_initial.size < 6:
        raise ValueError("at least six point residuals are required")

    optimization = least_squares(
        residual_function,
        x0,
        method="trf",
        jac="2-point",
        x_scale="jac",
        loss=loss,
        f_scale=f_scale_mm,
        max_nfev=int(max_nfev),
        ftol=float(ftol),
        xtol=float(xtol),
        gtol=float(gtol),
    )

    residual_final = residual_function(optimization.x)
    T_refined = apply_local_se3_update(T_reference, optimization.x)

    jacobian_rank, jacobian_condition = _jacobian_diagnostics(
        np.asarray(optimization.jac, dtype=float)
    )

    return NonlinearRefinementResult(
        T_ef_s=T_refined,
        success=bool(optimization.success),
        status=int(optimization.status),
        message=str(optimization.message),
        nfev=int(optimization.nfev),
        njev=(None if optimization.njev is None else int(optimization.njev)),
        cost=float(optimization.cost),
        optimality=float(optimization.optimality),
        initial_rms_mm=_rms(residual_initial),
        final_rms_mm=_rms(residual_final),
        delta_rotation_deg=float(np.degrees(np.linalg.norm(optimization.x[:3]))),
        delta_translation_mm=float(np.linalg.norm(optimization.x[3:])),
        jacobian_rank=jacobian_rank,
        jacobian_condition=jacobian_condition,
        local_update=np.asarray(optimization.x, dtype=float).copy(),
    )


def point_to_plane_residuals(
    groups: list[tuple[int, list[LaserScan]]],
    T_ef_s: np.ndarray,
    *,
    plane_mode: PlaneMode,
    fixed_planes: Mapping[int, Plane] | None = None,
) -> np.ndarray:
    """Return signed point-to-plane residuals in millimetres."""
    T_ef_s = _validate_transform(T_ef_s, "T_ef_s")
    residual_blocks: list[np.ndarray] = []

    for plane_id, scans in groups:
        points_base = _reconstruct_group_points(scans, T_ef_s)

        if plane_mode == "refit":
            normal, distance_mm, _, _ = fit_plane_pca(points_base)
        else:
            if fixed_planes is None or plane_id not in fixed_planes:
                raise ValueError(f"missing fixed plane for plane_id={plane_id}")
            normal, distance_mm = fixed_planes[plane_id]

        residual_blocks.append(points_base @ normal - distance_mm)

    if not residual_blocks:
        raise ValueError("no residuals could be constructed")
    return np.concatenate(residual_blocks)


def apply_local_se3_update(
    T_reference: np.ndarray,
    local_update: np.ndarray,
) -> np.ndarray:
    """Apply a right-local 6D update and return a valid SE(3) transform."""
    T_reference = _validate_transform(T_reference, "T_reference")
    update = np.asarray(local_update, dtype=float).reshape(6)
    if not np.all(np.isfinite(update)):
        raise ValueError("local_update must contain only finite values")

    R_delta = Rotation.from_rotvec(update[:3]).as_matrix()
    T_delta = make_T(R_delta, update[3:])
    return T_reference @ T_delta


def _reconstruct_group_points(
    scans: Sequence[LaserScan],
    T_ef_s: np.ndarray,
) -> np.ndarray:
    point_sets: list[np.ndarray] = []

    for scan in scans:
        points_sensor = scan.valid_points_s
        if len(points_sensor) == 0:
            continue
        points_ef = transform_points(T_ef_s, points_sensor)
        points_base = transform_points(scan.T_base_ef, points_ef)
        point_sets.append(points_base)

    if not point_sets:
        raise ValueError("a plane group contains no valid points")

    points = np.vstack(point_sets)
    if len(points) < 3:
        raise ValueError("each plane group requires at least three valid points")
    return points


def _normalize_scan_groups(
    scans_by_plane: ScanGroups,
) -> list[tuple[int, list[LaserScan]]]:
    if isinstance(scans_by_plane, Mapping):
        groups = [(int(plane_id), list(scans)) for plane_id, scans in scans_by_plane.items()]
    else:
        groups = [(plane_id, list(scans)) for plane_id, scans in enumerate(scans_by_plane)]

    normalized: list[tuple[int, list[LaserScan]]] = []
    for plane_id, scans in groups:
        if not scans:
            continue
        if any(not isinstance(scan, LaserScan) for scan in scans):
            raise TypeError("all scans must be LaserScan instances")
        normalized.append((plane_id, scans))

    if not normalized:
        raise ValueError("at least one non-empty plane group is required")
    return normalized


def _normalize_planes(
    planes: PlaneGroups,
    groups: Sequence[tuple[int, list[LaserScan]]],
) -> dict[int, Plane]:
    if isinstance(planes, Mapping):
        items = [(int(plane_id), plane) for plane_id, plane in planes.items()]
    else:
        items = [(plane_id, plane) for plane_id, plane in enumerate(planes)]

    normalized: dict[int, Plane] = {}
    for plane_id, (normal, distance_mm) in items:
        normal = np.asarray(normal, dtype=float).reshape(3)
        norm = float(np.linalg.norm(normal))
        if not np.isfinite(norm) or norm <= 0.0:
            raise ValueError(f"plane {plane_id} has an invalid normal")
        normal = normal / norm
        distance_mm = float(distance_mm)
        if not np.isfinite(distance_mm):
            raise ValueError(f"plane {plane_id} has a non-finite distance")
        if distance_mm < 0.0:
            normal = -normal
            distance_mm = -distance_mm
        normalized[plane_id] = (normal, distance_mm)

    missing = [plane_id for plane_id, _ in groups if plane_id not in normalized]
    if missing:
        raise ValueError(f"missing plane equations for plane IDs: {missing}")
    return normalized


def _validate_transform(T: np.ndarray, name: str) -> np.ndarray:
    T = np.asarray(T, dtype=float).reshape(4, 4).copy()
    if not np.all(np.isfinite(T)):
        raise ValueError(f"{name} must contain only finite values")
    return T


def _rms(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float).reshape(-1)
    return float(np.sqrt(np.mean(values * values)))


def _jacobian_diagnostics(jacobian: np.ndarray) -> tuple[int, float]:
    if jacobian.ndim != 2 or jacobian.shape[1] != 6:
        return 0, float("inf")

    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    rank = int(np.linalg.matrix_rank(jacobian))
    if len(singular_values) == 0 or singular_values[-1] <= 0.0:
        return rank, float("inf")
    return rank, float(singular_values[0] / singular_values[-1])