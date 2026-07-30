"""Estimated-state Fisher information for active laser hand-eye calibration.

The policy-facing functions in this module deliberately accept only acquired
laser scans, robot poses, and the current parameter estimate.  Simulation
truth is neither required nor accepted.

The local joint state is

    [hand-eye right rotation (3), hand-eye right translation (3),
     plane-1 normal tangent (2), plane-1 offset (1), ...].

Plane parameters are retained in the joint observed information matrix.  They
can subsequently be marginalized with a Schur complement when scoring only
the six hand-eye parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np
from scipy.spatial.transform import Rotation

from .data import LaserScan
from .geometry import fit_plane_pca
from .se3 import make_T, transform_points


NoiseAxis = Literal["z", "xz"]
FisherObjective = Literal["d_optimal", "e_optimal"]


@dataclass(frozen=True)
class PlaneEstimate:
    """One estimated plane in ``normal.T @ point = offset_mm`` form."""

    normal_base: np.ndarray
    offset_mm: float

    def __post_init__(self) -> None:
        normal = np.asarray(self.normal_base, dtype=float).reshape(3)
        norm = float(np.linalg.norm(normal))
        if not np.isfinite(norm) or norm <= 1e-12:
            raise ValueError("plane normal must be finite and non-zero")
        normal = normal / norm
        offset = float(self.offset_mm)
        if not np.isfinite(offset):
            raise ValueError("plane offset must be finite")
        if offset < 0.0:
            normal = -normal
            offset = -offset
        object.__setattr__(self, "normal_base", normal)
        object.__setattr__(self, "offset_mm", offset)


@dataclass(frozen=True)
class JointCalibrationEstimate:
    """Current joint maximum-likelihood estimate and observed information."""

    T_ef_s: np.ndarray
    planes: dict[int, PlaneEstimate]
    information: np.ndarray
    whitened_cost: float
    iterations: int
    converged: bool
    data_rank: int


def _normalize(vector: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(vector, dtype=float).reshape(3)
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError(f"{name} must be finite and non-zero")
    return value / norm


def plane_tangent_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return a deterministic orthonormal tangent basis for a unit normal."""
    normal = _normalize(normal, "plane normal")
    reference = np.eye(3, dtype=float)[int(np.argmin(np.abs(normal)))]
    tangent_u = _normalize(np.cross(normal, reference), "plane tangent u")
    tangent_v = _normalize(np.cross(normal, tangent_u), "plane tangent v")
    return tangent_u, tangent_v


def _ordered_groups(
    scans_by_plane: Mapping[int, Sequence[LaserScan]],
) -> list[tuple[int, list[LaserScan]]]:
    groups = [
        (int(plane_id), list(scans))
        for plane_id, scans in sorted(scans_by_plane.items())
        if scans
    ]
    if not groups:
        raise ValueError("at least one acquired plane group is required")
    if any(
        not isinstance(scan, LaserScan)
        for _plane_id, scans in groups
        for scan in scans
    ):
        raise TypeError("all acquired measurements must be LaserScan objects")
    return groups


def fit_plane_estimates(
    scans_by_plane: Mapping[int, Sequence[LaserScan]],
    T_ef_s: np.ndarray,
) -> dict[int, PlaneEstimate]:
    """Fit each unknown physical plane using only acquired measurements."""
    transform = np.asarray(T_ef_s, dtype=float).reshape(4, 4)
    estimates: dict[int, PlaneEstimate] = {}
    for plane_id, scans in _ordered_groups(scans_by_plane):
        point_blocks: list[np.ndarray] = []
        for scan in scans:
            points_sensor = scan.valid_points_s
            if len(points_sensor) == 0:
                continue
            points_ef = transform_points(transform, points_sensor)
            point_blocks.append(
                transform_points(scan.T_base_ef, points_ef)
            )
        if not point_blocks:
            raise ValueError(
                f"plane {plane_id} has no finite acquired profile points"
            )
        normal, offset, _centroid, _rms = fit_plane_pca(
            np.vstack(point_blocks)
        )
        estimates[plane_id] = PlaneEstimate(normal, float(offset))
    return estimates


def _residual_noise_projection(
    normal_sensor: np.ndarray,
    noise_axis: NoiseAxis,
) -> tuple[float, np.ndarray]:
    """Return the residual noise gain and its normal-vector gradient."""
    normal_sensor = np.asarray(normal_sensor, dtype=float).reshape(3)
    if noise_axis == "xz":
        projection = float(np.linalg.norm(normal_sensor[[0, 2]]))
        gradient = (
            np.asarray(
                [
                    normal_sensor[0] / projection,
                    0.0,
                    normal_sensor[2] / projection,
                ],
                dtype=float,
            )
            if projection > 1e-12
            else np.zeros(3, dtype=float)
        )
    elif noise_axis == "z":
        projection = abs(float(normal_sensor[2]))
        gradient = np.asarray(
            [0.0, 0.0, np.sign(normal_sensor[2])],
            dtype=float,
        )
    else:
        raise ValueError("noise_axis must be 'z' or 'xz'")
    if projection <= 1e-12:
        raise ValueError("profile has zero point-to-plane noise projection")
    return projection, gradient


def joint_residual_jacobian(
    scans_by_plane: Mapping[int, Sequence[LaserScan]],
    T_ef_s: np.ndarray,
    planes: Mapping[int, PlaneEstimate],
    *,
    profile_noise_std_mm: float,
    noise_axis: NoiseAxis,
    parameter_scales: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return whitened residuals and the joint local analytic Jacobian.

    The hand-eye update is the right-local retraction

    ``T_new = T_ef_s @ [Exp(delta_rotation), delta_translation]``.

    Consequently both local rotation and local translation are mapped through
    ``R_base_ef @ R_ef_s``.  For acquired nonzero residuals, the Jacobian also
    differentiates the geometry-dependent whitening gain.  At a predicted
    candidate's zero expected residual this term vanishes, yielding the usual
    Gauss-Newton mean-sensitivity Fisher approximation.
    """
    groups = _ordered_groups(scans_by_plane)
    plane_ids = [plane_id for plane_id, _scans in groups]
    if set(plane_ids) != set(map(int, planes)):
        raise ValueError("plane estimates must match the acquired groups")

    transform = np.asarray(T_ef_s, dtype=float).reshape(4, 4)
    R_ef_s = transform[:3, :3]
    t_ef_s = transform[:3, 3]
    dimension = 6 + 3 * len(groups)
    scales = (
        np.ones(dimension, dtype=float)
        if parameter_scales is None
        else np.asarray(parameter_scales, dtype=float).reshape(dimension)
    )
    if np.any(~np.isfinite(scales)) or np.any(scales <= 0.0):
        raise ValueError("parameter scales must be positive and finite")

    residual_blocks: list[np.ndarray] = []
    jacobian_blocks: list[np.ndarray] = []
    for plane_index, (plane_id, scans) in enumerate(groups):
        plane = planes[plane_id]
        normal = plane.normal_base
        tangent_u, tangent_v = plane_tangent_basis(normal)
        plane_start = 6 + 3 * plane_index

        for scan in scans:
            points_sensor = scan.valid_points_s
            if len(points_sensor) == 0:
                continue
            R_base_ef = scan.T_base_ef[:3, :3]
            t_base_ef = scan.T_base_ef[:3, 3]
            R_base_s = R_base_ef @ R_ef_s
            points_base = (
                R_base_ef
                @ (
                    R_ef_s @ points_sensor.T
                    + t_ef_s[:, None]
                )
                + t_base_ef[:, None]
            ).T
            normal_sensor = normal @ R_base_s
            projection, projection_gradient = (
                _residual_noise_projection(
                    normal_sensor,
                    noise_axis,
                )
            )
            profile_sigma = float(profile_noise_std_mm)
            if not np.isfinite(profile_sigma) or profile_sigma <= 0.0:
                raise ValueError(
                    "profile noise standard deviation must be positive"
                )
            residual_sigma = profile_sigma * projection

            physical_residual = (
                points_base @ normal - plane.offset_mm
            )
            residual = physical_residual / residual_sigma
            physical_jacobian = np.zeros(
                (len(points_sensor), dimension),
                dtype=float,
            )
            # -a @ [p]_x == p x a. Write the components directly because
            # np.cross has substantial axis-management overhead here.
            physical_jacobian[:, 0] = (
                points_sensor[:, 1] * normal_sensor[2]
                - points_sensor[:, 2] * normal_sensor[1]
            )
            physical_jacobian[:, 1] = (
                points_sensor[:, 2] * normal_sensor[0]
                - points_sensor[:, 0] * normal_sensor[2]
            )
            physical_jacobian[:, 2] = (
                points_sensor[:, 0] * normal_sensor[1]
                - points_sensor[:, 1] * normal_sensor[0]
            )
            physical_jacobian[:, 3:6] = normal_sensor
            physical_jacobian[
                :, plane_start : plane_start + 2
            ] = (
                points_base @ np.column_stack([tangent_u, tangent_v])
            )
            physical_jacobian[:, plane_start + 2] = -1.0

            sigma_jacobian = np.zeros(dimension, dtype=float)
            sigma_jacobian[:3] = profile_sigma * np.cross(
                projection_gradient,
                normal_sensor,
            )
            sigma_jacobian[
                plane_start : plane_start + 2
            ] = profile_sigma * np.asarray(
                [
                    projection_gradient
                    @ (R_base_s.T @ tangent_u),
                    projection_gradient
                    @ (R_base_s.T @ tangent_v),
                ],
                dtype=float,
            )
            jacobian = (
                physical_jacobian / residual_sigma
                - physical_residual[:, None]
                * sigma_jacobian[None, :]
                / (residual_sigma**2)
            )

            residual_blocks.append(residual)
            jacobian_blocks.append(jacobian * scales[None, :])

    if not residual_blocks:
        raise ValueError("acquired scans contain no finite profile points")
    return np.concatenate(residual_blocks), np.vstack(jacobian_blocks)


def apply_joint_local_update(
    T_ef_s: np.ndarray,
    planes: Mapping[int, PlaneEstimate],
    physical_update: np.ndarray,
) -> tuple[np.ndarray, dict[int, PlaneEstimate]]:
    """Apply one right-local hand-eye and tangent-plane update."""
    ordered_plane_ids = sorted(map(int, planes))
    update = np.asarray(physical_update, dtype=float).reshape(
        6 + 3 * len(ordered_plane_ids)
    )
    if np.any(~np.isfinite(update)):
        raise ValueError("joint update must contain only finite values")

    transform = np.asarray(T_ef_s, dtype=float).reshape(4, 4)
    delta_transform = make_T(
        Rotation.from_rotvec(update[:3]).as_matrix(),
        update[3:6],
    )
    updated_transform = transform @ delta_transform

    updated_planes: dict[int, PlaneEstimate] = {}
    for plane_index, plane_id in enumerate(ordered_plane_ids):
        plane = planes[plane_id]
        tangent_u, tangent_v = plane_tangent_basis(plane.normal_base)
        start = 6 + 3 * plane_index
        plane_update = update[start : start + 3]
        updated_normal = _normalize(
            plane.normal_base
            + tangent_u * plane_update[0]
            + tangent_v * plane_update[1],
            "updated plane normal",
        )
        updated_planes[plane_id] = PlaneEstimate(
            updated_normal,
            plane.offset_mm + float(plane_update[2]),
        )
    return updated_transform, updated_planes


def estimate_joint_calibration(
    scans_by_plane: Mapping[int, Sequence[LaserScan]],
    T_initial: np.ndarray,
    *,
    parameter_scales: np.ndarray,
    profile_noise_std_mm: float,
    noise_axis: NoiseAxis,
    planes_initial: Mapping[int, PlaneEstimate] | None = None,
    max_iterations: int = 30,
    tolerance: float = 1e-7,
    max_normalized_step: float = 1.0,
    max_line_search_steps: int = 12,
) -> JointCalibrationEstimate:
    """Jointly refine hand-eye and unknown planes with weighted Gauss-Newton.

    The solve is warm-startable and relinearizes every acquired residual after
    each accepted update.  No truth transform, truth plane, or artificial
    identity information prior is used.
    """
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    if tolerance <= 0.0 or not np.isfinite(tolerance):
        raise ValueError("tolerance must be positive and finite")
    if max_normalized_step <= 0.0:
        raise ValueError("max_normalized_step must be positive")
    if max_line_search_steps <= 0:
        raise ValueError("max_line_search_steps must be positive")

    transform = np.asarray(T_initial, dtype=float).reshape(4, 4).copy()
    planes = (
        fit_plane_estimates(scans_by_plane, transform)
        if planes_initial is None
        else {
            int(plane_id): PlaneEstimate(
                plane.normal_base,
                plane.offset_mm,
            )
            for plane_id, plane in planes_initial.items()
        }
    )
    scales = np.asarray(parameter_scales, dtype=float).reshape(
        6 + 3 * len(planes)
    )

    converged = False
    completed_iterations = 0
    previous_cost = float("inf")
    for iteration in range(max_iterations):
        residual, jacobian = joint_residual_jacobian(
            scans_by_plane,
            transform,
            planes,
            profile_noise_std_mm=profile_noise_std_mm,
            noise_axis=noise_axis,
            parameter_scales=scales,
        )
        cost = float(residual @ residual)
        rank = int(np.linalg.matrix_rank(jacobian))
        if rank < jacobian.shape[1]:
            raise np.linalg.LinAlgError(
                "acquired-data Fisher matrix is rank deficient: "
                f"rank={rank}, dimension={jacobian.shape[1]}"
            )

        normalized_update, *_ = np.linalg.lstsq(
            jacobian,
            -residual,
            rcond=None,
        )
        update_norm = float(np.linalg.norm(normalized_update))
        if update_norm > max_normalized_step:
            normalized_update *= max_normalized_step / update_norm
            update_norm = max_normalized_step
        physical_update = scales * normalized_update

        accepted = False
        accepted_scale = 0.0
        accepted_cost = cost
        for line_search_step in range(max_line_search_steps):
            step_scale = float(0.5**line_search_step)
            candidate_transform, candidate_planes = (
                apply_joint_local_update(
                    transform,
                    planes,
                    step_scale * physical_update,
                )
            )
            candidate_residual, _candidate_jacobian = (
                joint_residual_jacobian(
                    scans_by_plane,
                    candidate_transform,
                    candidate_planes,
                    profile_noise_std_mm=profile_noise_std_mm,
                    noise_axis=noise_axis,
                    parameter_scales=scales,
                )
            )
            candidate_cost = float(
                candidate_residual @ candidate_residual
            )
            if candidate_cost < cost:
                transform = candidate_transform
                planes = candidate_planes
                accepted = True
                accepted_scale = step_scale
                accepted_cost = candidate_cost
                break

        completed_iterations = iteration + 1
        scaled_step_norm = accepted_scale * update_norm
        relative_improvement = (
            (cost - accepted_cost) / max(cost, 1.0)
            if accepted
            else 0.0
        )
        if (
            not accepted
            or scaled_step_norm < tolerance
            or relative_improvement < tolerance
        ):
            converged = True
            break
        if accepted_cost > previous_cost:
            raise RuntimeError("joint calibration cost increased")
        previous_cost = accepted_cost

    final_residual, final_jacobian = joint_residual_jacobian(
        scans_by_plane,
        transform,
        planes,
        profile_noise_std_mm=profile_noise_std_mm,
        noise_axis=noise_axis,
        parameter_scales=scales,
    )
    information = final_jacobian.T @ final_jacobian
    information = 0.5 * (information + information.T)
    return JointCalibrationEstimate(
        T_ef_s=transform,
        planes=planes,
        information=information,
        whitened_cost=float(final_residual @ final_residual),
        iterations=completed_iterations,
        converged=converged,
        data_rank=int(np.linalg.matrix_rank(final_jacobian)),
    )


def predict_profile_points(
    T_base_ef: np.ndarray,
    T_ef_s: np.ndarray,
    plane: PlaneEstimate,
    x_values: np.ndarray,
    *,
    depth_range_mm: tuple[float, float] | None = None,
    min_abs_normal_z: float = 1e-6,
) -> np.ndarray | None:
    """Predict a candidate profile from the current estimate only."""
    T_base_s = (
        np.asarray(T_base_ef, dtype=float).reshape(4, 4)
        @ np.asarray(T_ef_s, dtype=float).reshape(4, 4)
    )
    normal_sensor = T_base_s[:3, :3].T @ plane.normal_base
    if abs(float(normal_sensor[2])) < float(min_abs_normal_z):
        return None
    x_values = np.asarray(x_values, dtype=float).reshape(-1)
    rhs = float(
        plane.offset_mm
        - plane.normal_base @ T_base_s[:3, 3]
    )
    z_values = (
        rhs - normal_sensor[0] * x_values
    ) / normal_sensor[2]
    if np.any(~np.isfinite(z_values)):
        return None
    if depth_range_mm is not None:
        depth_min, depth_max = map(float, depth_range_mm)
        if (
            float(np.min(z_values)) < depth_min
            or float(np.max(z_values)) > depth_max
        ):
            return None
    return np.column_stack(
        [x_values, np.zeros_like(x_values), z_values]
    )


def predicted_candidate_information(
    *,
    T_base_ef: np.ndarray,
    plane_id: int,
    estimate: JointCalibrationEstimate,
    x_values: np.ndarray,
    parameter_scales: np.ndarray,
    profile_noise_std_mm: float,
    noise_axis: NoiseAxis,
    depth_range_mm: tuple[float, float] | None,
) -> np.ndarray | None:
    """Return an expected candidate FIM without reading a future profile."""
    plane_id = int(plane_id)
    if plane_id not in estimate.planes:
        raise KeyError(f"missing estimated plane {plane_id}")
    predicted_points = predict_profile_points(
        T_base_ef,
        estimate.T_ef_s,
        estimate.planes[plane_id],
        x_values,
        depth_range_mm=depth_range_mm,
    )
    if predicted_points is None:
        return None
    predicted_scan = LaserScan(
        T_base_ef=np.asarray(T_base_ef, dtype=float).reshape(4, 4),
        points_s=predicted_points,
        plane_id=plane_id,
        scan_id=0,
        meta={"profile_prediction": "current_estimate"},
    )
    _residual, jacobian = joint_residual_jacobian(
        {plane_id: [predicted_scan]},
        estimate.T_ef_s,
        {plane_id: estimate.planes[plane_id]},
        profile_noise_std_mm=profile_noise_std_mm,
        noise_axis=noise_axis,
        parameter_scales=np.asarray(parameter_scales, dtype=float)[
            np.r_[
                np.arange(6),
                np.arange(
                    6 + 3 * sorted(estimate.planes).index(plane_id),
                    9 + 3 * sorted(estimate.planes).index(plane_id),
                ),
            ]
        ],
    )

    full_dimension = 6 + 3 * len(estimate.planes)
    full_jacobian = np.zeros(
        (len(jacobian), full_dimension),
        dtype=float,
    )
    full_jacobian[:, :6] = jacobian[:, :6]
    plane_index = sorted(estimate.planes).index(plane_id)
    full_start = 6 + 3 * plane_index
    full_jacobian[:, full_start : full_start + 3] = jacobian[:, 6:9]
    information = full_jacobian.T @ full_jacobian
    return 0.5 * (information + information.T)


def marginal_handeye_information(
    joint_information: np.ndarray,
) -> np.ndarray:
    """Schur-marginalize plane nuisance parameters using Cholesky solves."""
    information = np.asarray(joint_information, dtype=float)
    if (
        information.ndim != 2
        or information.shape[0] != information.shape[1]
        or information.shape[0] <= 6
    ):
        raise ValueError("joint information must be square and larger than 6")
    information = 0.5 * (information + information.T)
    nuisance = information[6:, 6:]
    coupling = information[:6, 6:]
    cholesky = np.linalg.cholesky(nuisance)
    solved = np.linalg.solve(
        cholesky.T,
        np.linalg.solve(cholesky, coupling.T),
    )
    marginal = information[:6, :6] - coupling @ solved
    return 0.5 * (marginal + marginal.T)


def fisher_objective_value(
    joint_information: np.ndarray,
    objective: FisherObjective,
) -> float:
    """Evaluate D- or E-optimality on marginal hand-eye information."""
    marginal = marginal_handeye_information(joint_information)
    if objective == "d_optimal":
        sign, logdet = np.linalg.slogdet(marginal)
        if sign <= 0.0:
            raise np.linalg.LinAlgError(
                "marginal hand-eye information is not positive definite"
            )
        return 0.5 * float(logdet)
    if objective == "e_optimal":
        eigenvalues = np.linalg.eigvalsh(marginal)
        if eigenvalues[0] <= 0.0:
            raise np.linalg.LinAlgError(
                "marginal hand-eye information is not positive definite"
            )
        return float(eigenvalues[0])
    raise ValueError(f"unsupported Fisher objective: {objective}")
