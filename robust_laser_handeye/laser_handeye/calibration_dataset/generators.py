from __future__ import annotations

from dataclasses import asdict, dataclass
from numbers import Integral
from typing import Literal, Sequence

import numpy as np

from ..data import LaserScan
from ..patterns import scan_parameter_grid
from ..pose_generation import sample_robot_pose_for_plane
from ..scene_generation import make_three_planes, plane_basis
from ..se3 import make_T
from ..simulation import (
    PoseGeometry,
    generate_circular_pattern_scans,
    sample_random_handeye,
    sample_random_plane_pose,
    simulate_profile_on_plane,
)
from .models import (
    AcquisitionGroup,
    CalibrationDataset,
    CalibrationTruth,
    PlaneTruth,
)


@dataclass(frozen=True)
class GenerationSeeds:
    """Independent random streams for one immutable ideal acquisition."""

    master_seed: int
    trial_index: int
    handeye_seed: int
    environment_seed: int
    motion_seed: int

    def __post_init__(self) -> None:
        for name in (
            "master_seed",
            "trial_index",
            "handeye_seed",
            "environment_seed",
            "motion_seed",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError(f"{name} must be an integer")
            if int(value) < 0:
                raise ValueError(f"{name} must be non-negative")
            object.__setattr__(self, name, int(value))

    @classmethod
    def derive(cls, master_seed: int, trial_index: int) -> "GenerationSeeds":
        for name, value in (
            ("master_seed", master_seed),
            ("trial_index", trial_index),
        ):
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError(f"{name} must be an integer")
            if int(value) < 0:
                raise ValueError(f"{name} must be non-negative")
        children = np.random.SeedSequence(
            [int(master_seed), int(trial_index)]
        ).spawn(3)
        values = [
            int(child.generate_state(1, dtype=np.uint32)[0])
            for child in children
        ]
        return cls(
            master_seed=int(master_seed),
            trial_index=int(trial_index),
            handeye_seed=values[0],
            environment_seed=values[1],
            motion_seed=values[2],
        )


@dataclass(frozen=True)
class SinglePlaneCircularGenerationConfig:
    """Current optimal-benchmark circular pattern, stored without noise."""

    profile_points: int = 100
    profile_half_width_mm: float = 25.0
    radius_mm: float = 100.0
    heights_mm: tuple[float, ...] = (60.0, 90.0, 120.0)
    theta_deg: tuple[float, ...] = (30.0,)
    beta_deg: tuple[float, ...] = (60.0, 90.0, 120.0)
    pose_geometry: PoseGeometry = "paper_incidence"
    projection_branch_mode: Literal["alternating", "positive"] = "alternating"
    reference_line_ids: tuple[int, ...] = (1, 2, 5, 6)
    reference_heights_mm: tuple[float, ...] = (60.0, 90.0, 120.0)
    reference_theta_deg: float = 60.0
    reference_beta_deg: tuple[float, ...] = (60.0, 90.0, 120.0)
    plane_angle_range_deg: tuple[float, float] = (-5.0, 5.0)
    plane_min_abs_angle_deg: float = 1.0
    plane_xy_range_mm: tuple[float, float] = (-100.0, 100.0)
    plane_z_range_mm: tuple[float, float] = (400.0, 550.0)
    check_reachability: bool = False

    def __post_init__(self) -> None:
        _validate_positive_integer(self.profile_points, "profile_points", minimum=3)
        _validate_positive(self.profile_half_width_mm, "profile_half_width_mm")
        _validate_positive(self.radius_mm, "radius_mm")
        _validate_nonempty_finite(self.heights_mm, "heights_mm", positive=True)
        _validate_nonempty_finite(self.theta_deg, "theta_deg")
        _validate_nonempty_finite(self.beta_deg, "beta_deg")
        _validate_nonempty_finite(
            self.reference_heights_mm,
            "reference_heights_mm",
            positive=True,
        )
        _validate_nonempty_finite(self.reference_beta_deg, "reference_beta_deg")
        if not np.isfinite(self.reference_theta_deg):
            raise ValueError("reference_theta_deg must be finite")
        if self.pose_geometry not in ("paper_incidence", "observable_dihedral"):
            raise ValueError("unsupported pose_geometry")
        if self.projection_branch_mode not in ("alternating", "positive"):
            raise ValueError("unsupported projection_branch_mode")
        line_ids = tuple(int(value) for value in self.reference_line_ids)
        if len(set(line_ids)) != len(line_ids) or any(
            value < 0 or value >= 9 for value in line_ids
        ):
            raise ValueError("reference_line_ids must be unique values in [0, 8]")
        object.__setattr__(self, "reference_line_ids", line_ids)
        _validate_range(self.plane_angle_range_deg, "plane_angle_range_deg")
        _validate_range(self.plane_xy_range_mm, "plane_xy_range_mm")
        _validate_range(self.plane_z_range_mm, "plane_z_range_mm")
        if (
            not np.isfinite(self.plane_min_abs_angle_deg)
            or self.plane_min_abs_angle_deg < 0.0
        ):
            raise ValueError("plane_min_abs_angle_deg must be finite and non-negative")
        angle_low, angle_high = self.plane_angle_range_deg
        if not (
            angle_low < -self.plane_min_abs_angle_deg
            or angle_high > self.plane_min_abs_angle_deg
        ):
            raise ValueError(
                "plane_angle_range_deg cannot satisfy plane_min_abs_angle_deg"
            )


@dataclass(frozen=True)
class ThreePlaneGenerationConfig:
    """Existing three-orthogonal-plane random acquisition, stored without noise."""

    poses_per_plane: int = 35
    profile_points: int = 100
    profile_half_width_mm: float = 25.0
    plane_distance_range_mm: tuple[float, float] = (650.0, 1000.0)
    plane_min_axis_angle_deg: float = 1.0
    tangent_range_mm: float = 220.0
    profile_depth_range_mm: tuple[float, float] = (60.0, 150.0)
    min_view_dot: float = 0.0
    max_trials_per_plane: int = 50_000

    def __post_init__(self) -> None:
        _validate_positive_integer(self.poses_per_plane, "poses_per_plane")
        _validate_positive_integer(self.profile_points, "profile_points", minimum=3)
        _validate_positive(self.profile_half_width_mm, "profile_half_width_mm")
        _validate_range(self.plane_distance_range_mm, "plane_distance_range_mm")
        _validate_range(self.profile_depth_range_mm, "profile_depth_range_mm")
        if self.plane_distance_range_mm[0] <= 0.0:
            raise ValueError("plane distances must be positive")
        if self.profile_depth_range_mm[0] <= 0.0:
            raise ValueError("profile depths must be positive")
        if (
            not np.isfinite(self.plane_min_axis_angle_deg)
            or not 0.0 <= self.plane_min_axis_angle_deg < 54.7356
        ):
            raise ValueError(
                "plane_min_axis_angle_deg must lie in [0, 54.7356)"
            )
        if not np.isfinite(self.tangent_range_mm) or self.tangent_range_mm < 0.0:
            raise ValueError("tangent_range_mm must be finite and non-negative")
        if not np.isfinite(self.min_view_dot) or not -1.0 <= self.min_view_dot < 1.0:
            raise ValueError("min_view_dot must lie in [-1, 1)")
        _validate_positive_integer(
            self.max_trials_per_plane,
            "max_trials_per_plane",
        )


def _validate_positive_integer(value: int, name: str, minimum: int = 1) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    if int(value) < minimum:
        raise ValueError(f"{name} must be at least {minimum}")


def _validate_positive(value: float, name: str) -> None:
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and positive")


def _validate_range(value: Sequence[float], name: str) -> None:
    values = np.asarray(value, dtype=float)
    if values.shape != (2,) or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain two finite values")
    if values[1] <= values[0]:
        raise ValueError(f"{name} must be increasing")


def _validate_nonempty_finite(
    value: Sequence[float],
    name: str,
    *,
    positive: bool = False,
) -> None:
    values = np.asarray(value, dtype=float)
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be a non-empty finite sequence")
    if positive and np.any(values <= 0.0):
        raise ValueError(f"{name} values must be positive")


def _sample_handeye(seed: int) -> np.ndarray:
    transform, _, _ = sample_random_handeye(np.random.default_rng(seed))
    return transform


def _jsonable(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _config_metadata_without_truth(config: object) -> dict[str, object]:
    payload = _jsonable(asdict(config))
    if not isinstance(payload, dict):
        raise RuntimeError("generation config must serialize to a dictionary")
    simulation = payload.get("simulation")
    if isinstance(simulation, dict):
        simulation.pop("T_ef_s_true", None)
        simulation.pop("plane_center_base_mm", None)
        simulation.pop("plane_normal_base", None)
    return payload


def _normalize_simulation_scans(
    grouped_scans: Sequence[tuple[str, Sequence[LaserScan]]],
) -> tuple[list[LaserScan], list[str], list[int]]:
    """Normalize analytic fixed-grid scans into the portable capture contract."""
    scans: list[LaserScan] = []
    group_ids: list[str] = []
    sequence_indices: list[int] = []
    for group_id, group_scans in grouped_scans:
        for sequence_index, original in enumerate(group_scans):
            meta = dict(original.meta)
            # Simulation truth belongs only in CalibrationTruth. Keeping it in
            # ordinary scan metadata would leak GT into blind solver inputs.
            meta.pop("true_T_base_ef", None)
            if original.scan_id is not None:
                meta.setdefault("source_scan_id", int(original.scan_id))
            meta["scan_uid"] = f"{group_id}:{sequence_index:06d}"
            meta["group_id"] = group_id
            meta.setdefault(
                "channel_ids",
                np.arange(original.num_points, dtype=np.int64),
            )
            scans.append(
                LaserScan(
                    T_base_ef=original.T_base_ef,
                    points_s=original.points_s,
                    plane_id=original.plane_id,
                    scan_id=len(scans),
                    meta=meta,
                )
            )
            group_ids.append(group_id)
            sequence_indices.append(sequence_index)
    return scans, group_ids, sequence_indices


def _dataset(
    *,
    grouped_scans: Sequence[tuple[str, Sequence[LaserScan]]],
    groups: Sequence[AcquisitionGroup],
    acquisition_mode: str,
    truth: CalibrationTruth,
    seeds: GenerationSeeds,
    metadata: dict[str, object],
) -> CalibrationDataset:
    scans, group_ids, sequence_indices = _normalize_simulation_scans(
        grouped_scans
    )
    pose_stack = np.stack([scan.T_base_ef for scan in scans], axis=0)
    normalized_truth = CalibrationTruth(
        T_ef_s_true=truth.T_ef_s_true,
        planes=truth.planes,
        T_base_ef_true=(
            pose_stack
            if truth.T_base_ef_true is None
            else truth.T_base_ef_true
        ),
        T_base_ef_commanded=(
            pose_stack
            if truth.T_base_ef_commanded is None
            else truth.T_base_ef_commanded
        ),
        metadata=truth.metadata,
    )
    return CalibrationDataset(
        scans=scans,
        scan_group_ids=group_ids,
        sequence_indices=sequence_indices,
        groups=list(groups),
        acquisition_mode=acquisition_mode,
        source="simulation",
        profile_state="ideal",
        truth=normalized_truth,
        metadata={
            **metadata,
            "generation_seeds": asdict(seeds),
            "noise_applied": False,
            "channel_layout": "stable_profile_sample_index",
            "profile_order_contract": (
                "group declaration order plus sequence_index_in_group are "
                "stable for prefix sweeps"
            ),
        },
        channel_id_semantics="stable_sensor_channel",
    )


def generate_single_plane_circular_dataset(
    config: SinglePlaneCircularGenerationConfig,
    seeds: GenerationSeeds,
) -> CalibrationDataset:
    true_handeye = _sample_handeye(seeds.handeye_seed)
    environment_rng = np.random.default_rng(seeds.environment_seed)
    motion_rng = np.random.default_rng(seeds.motion_seed)
    plane_R, plane_t, plane_n, plane_l, _plane_angles = sample_random_plane_pose(
        rng=environment_rng,
        angle_range_deg=config.plane_angle_range_deg,
        min_abs_angle_deg=config.plane_min_abs_angle_deg,
        xy_range_mm=config.plane_xy_range_mm,
        z_range_mm=config.plane_z_range_mm,
    )
    x_values = np.linspace(
        -config.profile_half_width_mm,
        config.profile_half_width_mm,
        config.profile_points,
    )
    primary_params = scan_parameter_grid(
        heights_mm=config.heights_mm,
        projection_deg=config.theta_deg,
        tilt_deg=config.beta_deg,
    )
    primary_scans = generate_circular_pattern_scans(
        plane_R=plane_R,
        plane_t=plane_t,
        T_ef_s_true=true_handeye,
        radius_mm=config.radius_mm,
        x_values=x_values,
        noise_std=0.0,
        rng=motion_rng,
        scan_params=primary_params,
        check_reachability=config.check_reachability,
        plane_id=0,
        projection_branch_mode=config.projection_branch_mode,
        pose_geometry=config.pose_geometry,
    )
    expected_primary = 9 * len(primary_params)
    if not config.check_reachability and len(primary_scans) != expected_primary:
        raise RuntimeError(
            f"circular primary generation produced {len(primary_scans)}/"
            f"{expected_primary} scans"
        )

    reference_scans: list[LaserScan] = []
    reference_params: list[dict[str, float]] = []
    if config.reference_line_ids:
        reference_params = scan_parameter_grid(
            heights_mm=config.reference_heights_mm,
            projection_deg=(config.reference_theta_deg,),
            tilt_deg=config.reference_beta_deg,
        )
        reference_all = generate_circular_pattern_scans(
            plane_R=plane_R,
            plane_t=plane_t,
            T_ef_s_true=true_handeye,
            radius_mm=config.radius_mm,
            x_values=x_values,
            noise_std=0.0,
            rng=motion_rng,
            scan_params=reference_params,
            check_reachability=config.check_reachability,
            plane_id=0,
            projection_branch_mode=config.projection_branch_mode,
            pose_geometry=config.pose_geometry,
        )
        selected_lines = set(config.reference_line_ids)
        for scan in reference_all:
            if int(scan.meta["line_id"]) not in selected_lines:
                continue
            scan.meta["reference_pose"] = True
            scan.meta["additional_scan"] = True
            reference_scans.append(scan)
        expected_reference = len(selected_lines) * len(reference_params)
        if not config.check_reachability and len(reference_scans) != expected_reference:
            raise RuntimeError(
                f"circular reference generation produced {len(reference_scans)}/"
                f"{expected_reference} scans"
            )

    groups = [
        AcquisitionGroup(
            group_id="primary_ring",
            acquisition_role="calibration",
            motion_kind="circular",
            plane_id=0,
            include_in_calibration=True,
        )
    ]
    grouped_scans: list[tuple[str, Sequence[LaserScan]]] = [
        ("primary_ring", primary_scans)
    ]
    if reference_scans:
        groups.append(
            AcquisitionGroup(
                group_id="reference_ring",
                acquisition_role="reference",
                motion_kind="circular_reference",
                plane_id=0,
                include_in_calibration=True,
            )
        )
        grouped_scans.append(("reference_ring", reference_scans))

    truth = CalibrationTruth(
        T_ef_s_true=true_handeye,
        planes=(
            PlaneTruth(
                plane_id=0,
                normal_base=plane_n,
                offset_mm=plane_l,
                T_base_plane=_plane_transform_with_normal(
                    plane_R,
                    plane_t,
                    plane_n,
                ),
            ),
        ),
        metadata={"handeye_source": "random"},
    )
    return _dataset(
        grouped_scans=grouped_scans,
        groups=groups,
        acquisition_mode="single_plane_circular",
        truth=truth,
        seeds=seeds,
        metadata={
            "generator": "optimal_single_plane_circular",
            "generator_config": _config_metadata_without_truth(config),
            "environment": {"target_plane": "infinite"},
            "expected_and_generated_counts": {
                "primary_expected": expected_primary,
                "primary_generated": len(primary_scans),
                "reference_expected": (
                    len(config.reference_line_ids) * len(reference_params)
                ),
                "reference_generated": len(reference_scans),
            },
            "reachability_check": (
                "simple workspace box" if config.check_reachability else "disabled"
            ),
        },
    )


def _canonical_plane_transform_at_point(
    normal: np.ndarray,
    point_on_plane: np.ndarray,
) -> np.ndarray:
    u_axis, v_axis = plane_basis(normal)
    rotation = np.column_stack([u_axis, v_axis, normal])
    return make_T(rotation, point_on_plane)


def _canonical_plane_transform(normal: np.ndarray, offset_mm: float) -> np.ndarray:
    point = np.asarray(normal, dtype=float) * float(offset_mm)
    return _canonical_plane_transform_at_point(normal, point)


def _plane_transform_with_normal(
    rotation: np.ndarray,
    origin: np.ndarray,
    normal: np.ndarray,
) -> np.ndarray:
    """Keep the planning-frame origin/X direction while aligning signed Z."""
    aligned = np.asarray(rotation, dtype=float).reshape(3, 3).copy()
    normal = np.asarray(normal, dtype=float).reshape(3)
    if float(aligned[:, 2] @ normal) < 0.0:
        # Flip two axes to preserve a right-handed SO(3) frame.
        aligned[:, 0] *= -1.0
        aligned[:, 2] *= -1.0
    return make_T(aligned, origin)


def generate_three_plane_dataset(
    config: ThreePlaneGenerationConfig,
    seeds: GenerationSeeds,
) -> CalibrationDataset:
    true_handeye = _sample_handeye(seeds.handeye_seed)
    environment_rng = np.random.default_rng(seeds.environment_seed)
    motion_rng = np.random.default_rng(seeds.motion_seed)
    planes = make_three_planes(
        rng=environment_rng,
        distance_range_mm=config.plane_distance_range_mm,
        min_axis_angle_deg=config.plane_min_axis_angle_deg,
    )
    x_values = np.linspace(
        -config.profile_half_width_mm,
        config.profile_half_width_mm,
        config.profile_points,
    )
    grouped_scans: list[tuple[str, Sequence[LaserScan]]] = []
    groups: list[AcquisitionGroup] = []
    attempts_by_plane: dict[str, int] = {}
    depth_min, depth_max = config.profile_depth_range_mm

    for plane_id, (plane_n, plane_l) in enumerate(planes):
        scans: list[LaserScan] = []
        attempts = 0
        while (
            len(scans) < config.poses_per_plane
            and attempts < config.max_trials_per_plane
        ):
            attempts += 1
            try:
                pose = sample_robot_pose_for_plane(
                    T_ef_s=true_handeye,
                    plane_n=plane_n,
                    plane_l=plane_l,
                    rng=motion_rng,
                    tangent_range_mm=config.tangent_range_mm,
                    depth_range_mm=config.profile_depth_range_mm,
                    min_view_dot=config.min_view_dot,
                )
                scan = simulate_profile_on_plane(
                    T_base_ef=pose,
                    T_ef_s_true=true_handeye,
                    plane_n=plane_n,
                    plane_l=plane_l,
                    x_values=x_values,
                    noise_std=0.0,
                    rng=motion_rng,
                    plane_id=plane_id,
                    scan_id=len(scans),
                    meta={"generation_attempt": attempts},
                )
            except (RuntimeError, ValueError):
                continue
            points = scan.valid_points_s
            if len(points) != scan.num_points:
                continue
            if np.min(points[:, 2]) < depth_min or np.max(points[:, 2]) > depth_max:
                continue
            scans.append(scan)

        if len(scans) != config.poses_per_plane:
            raise RuntimeError(
                f"only generated {len(scans)} scans for plane {plane_id}; "
                f"requested {config.poses_per_plane}; attempts={attempts}"
            )
        group_id = f"plane_{plane_id}"
        grouped_scans.append((group_id, scans))
        groups.append(
            AcquisitionGroup(
                group_id=group_id,
                acquisition_role="calibration",
                motion_kind="general_6dof",
                plane_id=plane_id,
                include_in_calibration=True,
            )
        )
        attempts_by_plane[group_id] = attempts

    truth = CalibrationTruth(
        T_ef_s_true=true_handeye,
        planes=tuple(
            PlaneTruth(
                plane_id=plane_id,
                normal_base=normal,
                offset_mm=offset,
                T_base_plane=_canonical_plane_transform(normal, offset),
            )
            for plane_id, (normal, offset) in enumerate(planes)
        ),
        metadata={"handeye_source": "random"},
    )
    return _dataset(
        grouped_scans=grouped_scans,
        groups=groups,
        acquisition_mode="three_plane_random",
        truth=truth,
        seeds=seeds,
        metadata={
            "generator": "random_three_orthogonal_planes",
            "generator_config": _config_metadata_without_truth(config),
            "environment": {
                "target_planes": "three infinite mutually orthogonal planes",
            },
            "generation_attempts_by_plane": attempts_by_plane,
            "reachability_check": "view direction and profile depth only; no IK",
        },
    )


__all__ = [
    "GenerationSeeds",
    "SinglePlaneCircularGenerationConfig",
    "ThreePlaneGenerationConfig",
    "generate_single_plane_circular_dataset",
    "generate_three_plane_dataset",
]
