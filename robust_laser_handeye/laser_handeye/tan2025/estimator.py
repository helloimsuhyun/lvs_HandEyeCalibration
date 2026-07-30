from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
import time

import numpy as np

from ..se3 import inv_T, make_T, project_to_so3
from .models import Tan2025Dataset


@dataclass(frozen=True)
class LinearSystemDiagnostics:
    """Rank and conditioning information for one paper equation block."""

    rows: int
    columns: int
    rank: int
    condition: float
    singular_values: tuple[float, ...]
    residual_rms: float


@dataclass(frozen=True)
class Tan2025EstimatorConfig:
    """Numerical and data-quality policy for the closed-form estimator."""

    line_constraint_weight: float = 1.0
    center_constraint_weight: float = 1.0
    max_translation_rotation_spread_deg: float = 0.1
    max_abs_profile_y_mm: float = 1e-6
    max_condition: float = 1e12
    rank_relative_tolerance: float = 1e-10
    min_translation_excitation_mm: float = 1.0
    min_composite_rotation_excitation: float = 1e-3
    min_profile_line_extent_mm: float = 1.0
    max_profile_linearity_ratio: float = 0.1

    def __post_init__(self) -> None:
        if not np.isfinite(self.line_constraint_weight) or self.line_constraint_weight <= 0.0:
            raise ValueError("line_constraint_weight must be positive")
        if not np.isfinite(self.center_constraint_weight) or self.center_constraint_weight <= 0.0:
            raise ValueError("center_constraint_weight must be positive")
        if (
            not np.isfinite(self.max_translation_rotation_spread_deg)
            or self.max_translation_rotation_spread_deg < 0.0
        ):
            raise ValueError(
                "max_translation_rotation_spread_deg must be finite and non-negative"
            )
        if not np.isfinite(self.max_abs_profile_y_mm) or self.max_abs_profile_y_mm < 0.0:
            raise ValueError("max_abs_profile_y_mm must be finite and non-negative")
        if not np.isfinite(self.max_condition) or self.max_condition <= 1.0:
            raise ValueError("max_condition must be finite and greater than one")
        if (
            not np.isfinite(self.rank_relative_tolerance)
            or not 0.0 < self.rank_relative_tolerance < 1.0
        ):
            raise ValueError("rank_relative_tolerance must lie between zero and one")
        if (
            not np.isfinite(self.min_translation_excitation_mm)
            or self.min_translation_excitation_mm <= 0.0
        ):
            raise ValueError("min_translation_excitation_mm must be finite and positive")
        if (
            not np.isfinite(self.min_composite_rotation_excitation)
            or self.min_composite_rotation_excitation <= 0.0
        ):
            raise ValueError(
                "min_composite_rotation_excitation must be finite and positive"
            )
        if (
            not np.isfinite(self.min_profile_line_extent_mm)
            or self.min_profile_line_extent_mm <= 0.0
        ):
            raise ValueError("min_profile_line_extent_mm must be finite and positive")
        if (
            not np.isfinite(self.max_profile_linearity_ratio)
            or not 0.0 <= self.max_profile_linearity_ratio < 1.0
        ):
            raise ValueError(
                "max_profile_linearity_ratio must be finite and lie in [0, 1)"
            )


@dataclass
class Tan2025Estimate:
    """Closed-form hand-eye result and observability diagnostics."""

    T_ef_s: np.ndarray
    plane_normal_base: np.ndarray
    elapsed_s: float
    normal_system: LinearSystemDiagnostics
    line_system: LinearSystemDiagnostics
    center_system: LinearSystemDiagnostics
    rotation_system: LinearSystemDiagnostics
    translation_system: LinearSystemDiagnostics
    constraint_rms: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.T_ef_s = np.asarray(self.T_ef_s, dtype=float).reshape(4, 4).copy()
        normal = np.asarray(self.plane_normal_base, dtype=float).reshape(3)
        self.plane_normal_base = normal / np.linalg.norm(normal)
        self.elapsed_s = float(self.elapsed_s)
        self.constraint_rms = {
            str(key): float(value) for key, value in self.constraint_rms.items()
        }


def _rotation_distance_deg(first: np.ndarray, second: np.ndarray) -> float:
    relative = np.asarray(first).reshape(3, 3).T @ np.asarray(second).reshape(3, 3)
    cosine = np.clip((float(np.trace(relative)) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _validate_transform(transform: np.ndarray, name: str) -> None:
    transform = np.asarray(transform, dtype=float)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError(f"{name} must be a finite 4x4 transform")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError(f"{name} has an invalid homogeneous last row")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError(f"{name} rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
        raise ValueError(f"{name} rotation determinant must be +1")


def _diagnostics(
    matrix: np.ndarray,
    rhs: np.ndarray,
    solution: np.ndarray,
    relative_tolerance: float,
) -> LinearSystemDiagnostics:
    matrix = np.asarray(matrix, dtype=float)
    rhs = np.asarray(rhs, dtype=float).reshape(-1)
    singular_values = np.linalg.svd(matrix, compute_uv=False)
    if singular_values.size == 0 or singular_values[0] <= 0.0:
        rank = 0
        condition = float("inf")
    else:
        threshold = float(singular_values[0]) * float(relative_tolerance)
        rank = int(np.count_nonzero(singular_values > threshold))
        condition = float(singular_values[0] / singular_values[rank - 1])
    residual = matrix @ np.asarray(solution, dtype=float).reshape(-1) - rhs
    residual_rms = float(np.sqrt(np.mean(np.square(residual))))
    return LinearSystemDiagnostics(
        rows=int(matrix.shape[0]),
        columns=int(matrix.shape[1]),
        rank=rank,
        condition=condition,
        singular_values=tuple(float(value) for value in singular_values),
        residual_rms=residual_rms,
    )


def _check_system(
    name: str,
    diagnostics: LinearSystemDiagnostics,
    required_rank: int,
    max_condition: float,
) -> None:
    if diagnostics.rank < required_rank:
        raise np.linalg.LinAlgError(
            f"{name} is unobservable: rank {diagnostics.rank} < "
            f"{required_rank}; collect more independent robot motions"
        )
    if diagnostics.condition > max_condition:
        raise np.linalg.LinAlgError(
            f"{name} is ill-conditioned: condition "
            f"{diagnostics.condition:.6g} > {max_condition:.6g}; "
            "increase pose diversity"
        )


def _profile_line(
    points_s: np.ndarray,
    *,
    max_abs_y_mm: float,
    min_line_extent_mm: float,
    max_linearity_ratio: float,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points_s, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{label} points must have shape (N, 3)")
    points = points[np.all(np.isfinite(points), axis=1)]
    if len(points) < 3:
        raise ValueError(f"{label} needs at least three finite points")
    if np.max(np.abs(points[:, 1])) > max_abs_y_mm:
        raise ValueError(
            f"{label} is not an X-Z laser profile: |y| exceeds "
            f"{max_abs_y_mm:g} mm"
        )
    center = np.mean(points, axis=0)
    _, singular_values, vh = np.linalg.svd(points - center, full_matrices=False)
    if singular_values[0] <= np.finfo(float).eps:
        raise ValueError(f"{label} has no measurable line extent")
    direction = vh[0]
    direction[1] = 0.0
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm <= np.finfo(float).eps:
        raise ValueError(f"{label} has an invalid PCA line direction")
    direction /= direction_norm
    if direction[0] < 0.0 or (
        np.isclose(direction[0], 0.0) and direction[2] < 0.0
    ):
        direction = -direction
    extent = float(np.ptp((points - center) @ direction))
    if extent < min_line_extent_mm:
        raise ValueError(
            f"{label} line extent {extent:.6g} mm is below "
            f"{min_line_extent_mm:.6g} mm"
        )
    linearity_ratio = float(
        singular_values[1] / singular_values[0]
        if len(singular_values) > 1
        else 0.0
    )
    if linearity_ratio > max_linearity_ratio:
        raise ValueError(
            f"{label} is not line-like: PCA ratio {linearity_ratio:.6g} > "
            f"{max_linearity_ratio:.6g}"
        )
    center[1] = 0.0
    return center, direction


class Tan2025ClosedFormEstimator:
    """Three-step single-plane estimator from Tan et al., TIM 2025.

    The implementation follows equations (17)--(54), using SVD-backed least
    squares instead of explicit normal-equation inverses.  The two rotation
    blocks are kept at their literal paper scale by default; their weights are
    exposed because the article does not state a normalization policy.
    """

    name = "tan2025_closed_form"

    def __init__(self, config: Tan2025EstimatorConfig | None = None) -> None:
        self.config = config or Tan2025EstimatorConfig()

    def estimate(self, dataset: Tan2025Dataset) -> Tan2025Estimate:
        started = time.perf_counter()
        self._validate_dataset(dataset)

        normal_base, normal_diag = self._recover_plane_normal(dataset)
        (
            rotation,
            line_diag,
            center_diag,
            rotation_diag,
        ) = self._recover_rotation(
            dataset,
            normal_base,
        )
        translation, translation_diag = self._recover_translation(
            dataset,
            normal_base,
            rotation,
        )
        transform = make_T(rotation, translation)

        return Tan2025Estimate(
            T_ef_s=transform,
            plane_normal_base=normal_base,
            elapsed_s=time.perf_counter() - started,
            normal_system=normal_diag,
            line_system=line_diag,
            center_system=center_diag,
            rotation_system=rotation_diag,
            translation_system=translation_diag,
            constraint_rms={
                "normal_range_mm": normal_diag.residual_rms,
                "line_orthogonality": line_diag.residual_rms,
                "center_pair_mm": center_diag.residual_rms,
                "rotation_combined": rotation_diag.residual_rms,
                "translation_pair_mm": translation_diag.residual_rms,
            },
        )

    def _validate_dataset(self, dataset: Tan2025Dataset) -> None:
        if len(dataset.translation_scans) < 4:
            raise ValueError(
                "Tan2025 needs at least four translation poses: one reference "
                "and three independent translations"
            )
        if len(dataset.composite_scans) < 6:
            raise ValueError(
                "Tan2025 needs at least six composite poses with distinct rotations"
            )

        for index, scan in enumerate(dataset.all_scans):
            _validate_transform(scan.T_base_ef, f"scan[{index}].T_base_ef")
            points = scan.valid_points_s
            if len(points) < 3:
                raise ValueError(f"scan[{index}] needs at least three finite points")
            if np.max(np.abs(points[:, 1])) > self.config.max_abs_profile_y_mm:
                raise ValueError(
                    f"scan[{index}] is not an X-Z laser profile: |y| exceeds "
                    f"{self.config.max_abs_profile_y_mm:g} mm"
                )

        reference_rotation = dataset.translation_scans[0].T_base_ef[:3, :3]
        spreads = [
            _rotation_distance_deg(reference_rotation, scan.T_base_ef[:3, :3])
            for scan in dataset.translation_scans[1:]
        ]
        maximum = max(spreads, default=0.0)
        if maximum > self.config.max_translation_rotation_spread_deg:
            raise ValueError(
                "translation motion group changes tool orientation by up to "
                f"{maximum:.6g} deg; limit is "
                f"{self.config.max_translation_rotation_spread_deg:.6g} deg"
            )

    def _recover_plane_normal(
        self,
        dataset: Tan2025Dataset,
    ) -> tuple[np.ndarray, LinearSystemDiagnostics]:
        reference_pose = dataset.translation_scans[0].T_base_ef
        relative_translations: list[np.ndarray] = []
        for scan in dataset.translation_scans[1:]:
            relative = inv_T(reference_pose) @ scan.T_base_ef
            relative_translations.append(relative[:3, 3].copy())
        translation_matrix = np.vstack(relative_translations)
        translation_singular_values = np.linalg.svd(
            translation_matrix, compute_uv=False
        )
        translation_rank = int(
            np.count_nonzero(
                translation_singular_values
                > translation_singular_values[0]
                * self.config.rank_relative_tolerance
            )
        )
        if translation_rank < 3:
            raise np.linalg.LinAlgError(
                "plane-normal translation system is unobservable: rank "
                f"{translation_rank} < 3; collect translations spanning "
                "three dimensions"
            )
        if translation_singular_values[2] < self.config.min_translation_excitation_mm:
            raise np.linalg.LinAlgError(
                "plane-normal translations are too small: weakest excitation "
                f"{translation_singular_values[2]:.6g} mm < "
                f"{self.config.min_translation_excitation_mm:.6g} mm"
            )

        channel_maps = dataset.translation_channel_maps()
        reference_channels = channel_maps[0]
        channel_solutions: list[np.ndarray] = []
        diagnostic_rows: list[np.ndarray] = []
        diagnostic_residuals: list[np.ndarray] = []
        for channel, reference_point in reference_channels.items():
            if not np.all(np.isfinite(reference_point)):
                continue
            rows: list[np.ndarray] = []
            target_values: list[float] = []
            reference_range = float(np.linalg.norm(reference_point))
            for translation, channel_map in zip(
                relative_translations,
                channel_maps[1:],
            ):
                point = channel_map.get(channel)
                if point is None or not np.all(np.isfinite(point)):
                    continue
                rows.append(translation)
                target_values.append(reference_range - float(np.linalg.norm(point)))
            if len(rows) < 3:
                continue
            channel_matrix = np.vstack(rows)
            channel_target = np.asarray(target_values, dtype=float)
            singular_values = np.linalg.svd(channel_matrix, compute_uv=False)
            if singular_values[0] <= 0.0:
                continue
            channel_rank = int(
                np.count_nonzero(
                    singular_values
                    > singular_values[0] * self.config.rank_relative_tolerance
                )
            )
            if channel_rank < 3:
                continue
            channel_condition = float(
                singular_values[0] / singular_values[channel_rank - 1]
            )
            if channel_condition > self.config.max_condition:
                continue
            channel_normal, *_ = np.linalg.lstsq(
                channel_matrix,
                channel_target,
                rcond=None,
            )
            if np.linalg.norm(channel_normal) <= np.finfo(float).eps:
                continue
            if channel_solutions and float(channel_normal @ channel_solutions[0]) < 0.0:
                channel_normal = -channel_normal
            channel_solutions.append(channel_normal)
            diagnostic_rows.append(channel_matrix)
            diagnostic_residuals.append(
                channel_matrix @ channel_normal - channel_target
            )

        if not channel_solutions:
            raise np.linalg.LinAlgError(
                "plane-normal system has no channel observed across three "
                "independent translations; preserve channel IDs/NaNs and "
                "collect more translation poses"
            )
        normal_raw = np.sum(channel_solutions, axis=0)
        matrix = np.vstack(diagnostic_rows)
        singular_values = np.linalg.svd(matrix, compute_uv=False)
        threshold = singular_values[0] * self.config.rank_relative_tolerance
        rank = int(np.count_nonzero(singular_values > threshold))
        condition = (
            float(singular_values[0] / singular_values[rank - 1])
            if rank > 0
            else float("inf")
        )
        residual = np.concatenate(diagnostic_residuals)
        diagnostics = LinearSystemDiagnostics(
            rows=int(matrix.shape[0]),
            columns=3,
            rank=rank,
            condition=condition,
            singular_values=tuple(float(value) for value in singular_values),
            residual_rms=float(np.sqrt(np.mean(np.square(residual)))),
        )
        _check_system(
            "plane-normal translation system",
            diagnostics,
            required_rank=3,
            max_condition=self.config.max_condition,
        )
        norm = float(np.linalg.norm(normal_raw))
        if norm <= np.finfo(float).eps:
            raise np.linalg.LinAlgError(
                "plane-normal solution is zero; translations have no range variation"
            )
        normal_f0 = normal_raw / norm
        normal_base = reference_pose[:3, :3] @ normal_f0
        normal_base /= np.linalg.norm(normal_base)
        return normal_base, diagnostics

    def _recover_rotation(
        self,
        dataset: Tan2025Dataset,
        normal_base: np.ndarray,
    ) -> tuple[
        np.ndarray,
        LinearSystemDiagnostics,
        LinearSystemDiagnostics,
        LinearSystemDiagnostics,
    ]:
        line_rows: list[np.ndarray] = []
        for index, scan in enumerate(dataset.composite_scans):
            _, direction = _profile_line(
                scan.points_s,
                max_abs_y_mm=self.config.max_abs_profile_y_mm,
                min_line_extent_mm=self.config.min_profile_line_extent_mm,
                max_linearity_ratio=self.config.max_profile_linearity_ratio,
                label=f"composite scan {index}",
            )
            projected_normal = normal_base @ scan.T_base_ef[:3, :3]
            line_rows.append(
                np.concatenate(
                    [direction[0] * projected_normal, direction[2] * projected_normal]
                )
            )
        line_matrix = np.vstack(line_rows)
        line_rhs = np.zeros(len(line_matrix), dtype=float)

        reference_rotation = dataset.translation_scans[0].T_base_ef[:3, :3]
        projected_reference_normal = normal_base @ reference_rotation
        translation_centers = [
            _profile_line(
                scan.points_s,
                max_abs_y_mm=self.config.max_abs_profile_y_mm,
                min_line_extent_mm=self.config.min_profile_line_extent_mm,
                max_linearity_ratio=self.config.max_profile_linearity_ratio,
                label=f"translation scan {index}",
            )[0]
            for index, scan in enumerate(dataset.translation_scans)
        ]
        center_rows: list[np.ndarray] = []
        center_rhs_values: list[float] = []
        for first, second in combinations(
            range(len(dataset.translation_scans)), 2
        ):
            delta_profile = translation_centers[first] - translation_centers[second]
            center_rows.append(
                np.concatenate(
                    [
                        delta_profile[0] * projected_reference_normal,
                        delta_profile[2] * projected_reference_normal,
                    ]
                )
            )
            first_t = dataset.translation_scans[first].T_base_ef[:3, 3]
            second_t = dataset.translation_scans[second].T_base_ef[:3, 3]
            center_rhs_values.append(float(normal_base @ (second_t - first_t)))
        center_matrix = np.vstack(center_rows)
        center_rhs = np.asarray(center_rhs_values, dtype=float)

        matrix = np.vstack(
            [
                self.config.line_constraint_weight * line_matrix,
                self.config.center_constraint_weight * center_matrix,
            ]
        )
        target = np.concatenate(
            [
                self.config.line_constraint_weight * line_rhs,
                self.config.center_constraint_weight * center_rhs,
            ]
        )
        w, *_ = np.linalg.lstsq(matrix, target, rcond=None)

        line_diag = _diagnostics(
            line_matrix,
            line_rhs,
            w,
            self.config.rank_relative_tolerance,
        )
        center_diag = _diagnostics(
            center_matrix,
            center_rhs,
            w,
            self.config.rank_relative_tolerance,
        )
        rotation_diag = _diagnostics(
            matrix,
            target,
            w,
            self.config.rank_relative_tolerance,
        )
        _check_system(
            "rotation system",
            rotation_diag,
            required_rank=6,
            max_condition=self.config.max_condition,
        )

        first_column = w[:3]
        third_column = w[3:]
        second_column = np.cross(third_column, first_column)
        raw_rotation = np.column_stack(
            [first_column, second_column, third_column]
        )
        rotation = project_to_so3(raw_rotation)
        return rotation, line_diag, center_diag, rotation_diag

    def _recover_translation(
        self,
        dataset: Tan2025Dataset,
        normal_base: np.ndarray,
        rotation_ef_s: np.ndarray,
    ) -> tuple[np.ndarray, LinearSystemDiagnostics]:
        centers = [
            _profile_line(
                scan.points_s,
                max_abs_y_mm=self.config.max_abs_profile_y_mm,
                min_line_extent_mm=self.config.min_profile_line_extent_mm,
                max_linearity_ratio=self.config.max_profile_linearity_ratio,
                label=f"composite scan {index}",
            )[0]
            for index, scan in enumerate(dataset.composite_scans)
        ]
        rows: list[np.ndarray] = []
        rhs_values: list[float] = []

        for first, second in combinations(range(len(centers)), 2):
            first_pose = dataset.composite_scans[first].T_base_ef
            second_pose = dataset.composite_scans[second].T_base_ef
            first_rotation = first_pose[:3, :3]
            second_rotation = second_pose[:3, :3]
            delta = (
                first_rotation @ rotation_ef_s @ centers[first]
                - second_rotation @ rotation_ef_s @ centers[second]
                + first_pose[:3, 3]
                - second_pose[:3, 3]
            )
            rows.append(normal_base @ (first_rotation - second_rotation))
            rhs_values.append(-float(normal_base @ delta))

        matrix = np.vstack(rows)
        target = np.asarray(rhs_values, dtype=float)
        translation, *_ = np.linalg.lstsq(matrix, target, rcond=None)
        diagnostics = _diagnostics(
            matrix,
            target,
            translation,
            self.config.rank_relative_tolerance,
        )
        _check_system(
            "hand-eye translation system",
            diagnostics,
            required_rank=3,
            max_condition=self.config.max_condition,
        )
        weakest = diagnostics.singular_values[-1]
        if weakest < self.config.min_composite_rotation_excitation:
            raise np.linalg.LinAlgError(
                "composite rotations are too small: weakest translation-system "
                f"excitation {weakest:.6g} < "
                f"{self.config.min_composite_rotation_excitation:.6g}"
            )
        return translation, diagnostics
