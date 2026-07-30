from __future__ import annotations

from dataclasses import dataclass, field
from numbers import Integral
from typing import Literal

import numpy as np

from ..data import LaserScan
from ..se3 import euler_xyz_deg, inv_T, make_T, project_to_so3
from .models import Tan2025Dataset, Tan2025GroundTruth


NoiseMode = Literal[
    "none",
    "gaussian_xz",
    "uniform_xz",
    "constant_range_bias",
]


def _normalize(vector: np.ndarray) -> np.ndarray:
    value = np.asarray(vector, dtype=float).reshape(3)
    norm = float(np.linalg.norm(value))
    if not np.isfinite(norm) or norm <= np.finfo(float).eps:
        raise ValueError("cannot normalize a zero or non-finite vector")
    return value / norm


def paper_handeye_transform() -> np.ndarray:
    """Return paper equation (55), using exact 90/0/5 degree rotation."""
    rotation = euler_xyz_deg(5.0, 0.0, 90.0)
    translation = np.array([-22.86848, 83.73314, 153.08619], dtype=float)
    return make_T(rotation, translation)


def _paper_plane_normal() -> np.ndarray:
    return _normalize(np.array([0.337918, 0.1050427, -0.935296], dtype=float))


def analytic_model_dict(config: "PaperSimulationConfig") -> dict[str, object]:
    """Describe simulator conventions that are not PaperSimulationConfig fields."""
    return {
        "version": 1,
        "target_plane": "infinite",
        "ray_sampling": "uniform_angle",
        "ray_direction_sensor": "[sin(theta), 0, cos(theta)]",
        "visibility": "positive range and sensor-Z clipping",
        "translation_anchors_mm": {
            "tangent_u": min(30.0, config.translation_tangent_span_mm),
            "tangent_v": min(30.0, config.translation_tangent_span_mm),
            "normal": min(18.0, config.translation_normal_span_mm),
        },
        "pose_sampling": (
            "fixed observability anchors followed by seeded uniform draws"
        ),
    }


@dataclass(frozen=True)
class SensorNoiseConfig:
    """Explicit profile-noise definition.

    The article never defines whether its "absolute noise" is a bias, uniform
    noise, or fixed-magnitude random noise.  This API avoids silently choosing
    one interpretation.  ``constant_range_bias`` is the closest named model
    to a systematic absolute range error; ``gaussian_xz`` applies independent
    zero-mean noise to the measured X and Z coordinates.
    """

    mode: NoiseMode = "none"
    magnitude_mm: float = 0.0
    dropout_probability: float = 0.0

    def __post_init__(self) -> None:
        allowed = {
            "none",
            "gaussian_xz",
            "uniform_xz",
            "constant_range_bias",
        }
        if self.mode not in allowed:
            raise ValueError(f"unsupported noise mode: {self.mode}")
        if self.magnitude_mm < 0.0 or not np.isfinite(self.magnitude_mm):
            raise ValueError("magnitude_mm must be finite and non-negative")
        if not 0.0 <= self.dropout_probability < 1.0:
            raise ValueError("dropout_probability must lie in [0, 1)")
        if self.mode == "none" and self.magnitude_mm != 0.0:
            raise ValueError("noise mode 'none' requires magnitude_mm=0")


@dataclass(frozen=True)
class PaperSimulationConfig:
    """Analytic reconstruction of the paper's disclosed synthetic setup.

    Values explicitly supplied by the paper are the hand-eye transform, plane
    normal, 640 samples, 21.4 degree fan, and 190--290 mm Z range.  The plane
    offset and robot-pose distributions are not published; the defaults below
    are therefore documented implementation choices and are stored in every
    generated dataset's metadata.
    """

    num_translation_poses: int = 36
    num_composite_poses: int = 30
    num_profile_points: int = 640
    scan_angle_deg: float = 21.4
    sensor_z_min_mm: float = 190.0
    sensor_z_max_mm: float = 290.0
    nominal_sensor_distance_mm: float = 240.0
    plane_center_base_mm: tuple[float, float, float] = (500.0, 0.0, 300.0)
    plane_normal_base: np.ndarray = field(default_factory=_paper_plane_normal)
    T_ef_s_true: np.ndarray = field(default_factory=paper_handeye_transform)
    translation_tangent_span_mm: float = 55.0
    translation_normal_span_mm: float = 22.0
    composite_target_span_mm: float = 55.0
    composite_tilt_span_deg: float = 30.0
    composite_roll_span_deg: float = 170.0
    composite_distance_span_mm: float = 12.0
    pose_translation_noise_std_mm: float = 0.0
    pose_rotation_noise_std_deg: float = 0.0
    minimum_valid_fraction: float = 0.8
    sensor_noise: SensorNoiseConfig = field(default_factory=SensorNoiseConfig)

    def __post_init__(self) -> None:
        for name in (
            "num_translation_poses",
            "num_composite_poses",
            "num_profile_points",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral):
                raise ValueError(f"{name} must be an integer")
            object.__setattr__(self, name, int(value))

        if self.num_translation_poses < 4:
            raise ValueError("num_translation_poses must be at least four")
        if self.num_composite_poses < 6:
            raise ValueError("num_composite_poses must be at least six")
        if self.num_profile_points < 3:
            raise ValueError("num_profile_points must be at least three")
        if not np.isfinite(self.scan_angle_deg) or not 0.0 < self.scan_angle_deg < 180.0:
            raise ValueError("scan_angle_deg must lie between zero and 180")
        if (
            not np.isfinite(self.sensor_z_min_mm)
            or not np.isfinite(self.sensor_z_max_mm)
            or not 0.0 < self.sensor_z_min_mm < self.sensor_z_max_mm
        ):
            raise ValueError("sensor Z range must be positive and increasing")
        if (
            not np.isfinite(self.nominal_sensor_distance_mm)
            or not self.sensor_z_min_mm
            < self.nominal_sensor_distance_mm
            < self.sensor_z_max_mm
        ):
            raise ValueError("nominal sensor distance must lie inside the Z range")

        for name in (
            "translation_tangent_span_mm",
            "translation_normal_span_mm",
            "composite_tilt_span_deg",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        for name in (
            "composite_target_span_mm",
            "composite_roll_span_deg",
            "composite_distance_span_mm",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if (
            not np.isfinite(self.pose_translation_noise_std_mm)
            or self.pose_translation_noise_std_mm < 0.0
        ):
            raise ValueError(
                "pose_translation_noise_std_mm must be finite and non-negative"
            )
        if (
            not np.isfinite(self.pose_rotation_noise_std_deg)
            or self.pose_rotation_noise_std_deg < 0.0
        ):
            raise ValueError(
                "pose_rotation_noise_std_deg must be finite and non-negative"
            )
        if (
            not np.isfinite(self.minimum_valid_fraction)
            or not 0.0 < self.minimum_valid_fraction <= 1.0
        ):
            raise ValueError("minimum_valid_fraction must lie in (0, 1]")
        if not isinstance(self.sensor_noise, SensorNoiseConfig):
            raise ValueError("sensor_noise must be a SensorNoiseConfig")

        normal = _normalize(self.plane_normal_base)
        center = np.asarray(self.plane_center_base_mm, dtype=float)
        if center.shape != (3,) or not np.all(np.isfinite(center)):
            raise ValueError("plane_center_base_mm must contain three finite values")
        transform = np.asarray(self.T_ef_s_true, dtype=float)
        if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
            raise ValueError("T_ef_s_true must be a finite 4x4 transform")
        if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
            raise ValueError("T_ef_s_true has an invalid homogeneous last row")
        rotation = transform[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7):
            raise ValueError("T_ef_s_true rotation must be orthonormal")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-7):
            raise ValueError("T_ef_s_true rotation determinant must be +1")
        object.__setattr__(self, "plane_normal_base", normal)
        object.__setattr__(self, "plane_center_base_mm", tuple(center.tolist()))
        object.__setattr__(self, "T_ef_s_true", transform.copy())


@dataclass
class SimulatedTan2025Dataset:
    dataset: Tan2025Dataset
    truth: Tan2025GroundTruth


def _reference_sensor_rotation(normal: np.ndarray) -> np.ndarray:
    z_axis = -_normalize(normal)
    hint = np.array([1.0, 0.0, 0.0], dtype=float)
    if abs(float(hint @ z_axis)) > 0.9:
        hint = np.array([0.0, 1.0, 0.0], dtype=float)
    x_axis = _normalize(hint - float(hint @ z_axis) * z_axis)
    y_axis = _normalize(np.cross(z_axis, x_axis))
    rotation = np.column_stack([x_axis, y_axis, z_axis])
    return project_to_so3(rotation)


def _plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    reference = _reference_sensor_rotation(normal)
    return reference[:, 0], reference[:, 1]


def _rotation_from_rotvec(rotvec: np.ndarray) -> np.ndarray:
    vector = np.asarray(rotvec, dtype=float).reshape(3)
    angle = float(np.linalg.norm(vector))
    if angle < 1e-14:
        return np.eye(3)
    axis = vector / angle
    x, y, z = axis
    skew = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(angle) * skew + (1.0 - np.cos(angle)) * (skew @ skew)


class AnalyticTan2025Simulator:
    """Vectorized plane/ray simulator with no physics-engine dependency."""

    name = "analytic_plane_ray"

    def __init__(self, config: PaperSimulationConfig | None = None) -> None:
        self.config = config or PaperSimulationConfig()

    def generate(
        self,
        *,
        seed: int | None = 7,
        rng: np.random.Generator | None = None,
    ) -> SimulatedTan2025Dataset:
        if rng is not None and seed is not None:
            raise ValueError("pass either seed or rng, not both")
        generator = np.random.default_rng(seed) if rng is None else rng
        config = self.config
        normal = config.plane_normal_base
        center = np.asarray(config.plane_center_base_mm, dtype=float).reshape(3)
        plane_offset = float(normal @ center)

        translation_true = self._translation_poses(generator, center, normal)
        composite_true = self._composite_poses(generator, center, normal)
        translation_scans = [
            self._make_scan(
                true_pose,
                generator,
                scan_id=index,
                motion_kind="translation",
                plane_normal=normal,
                plane_offset=plane_offset,
            )
            for index, true_pose in enumerate(translation_true)
        ]
        composite_scans = [
            self._make_scan(
                true_pose,
                generator,
                scan_id=index,
                motion_kind="composite",
                plane_normal=normal,
                plane_offset=plane_offset,
            )
            for index, true_pose in enumerate(composite_true)
        ]

        metadata = {
            "source": self.name,
            "paper": "Tan et al., IEEE TIM 2025, doi:10.1109/TIM.2025.3551450",
            "seed": seed,
            "units": {"length": "mm", "angle": "deg"},
            "disclosed_by_paper": {
                "num_profile_points": config.num_profile_points,
                "scan_angle_deg": config.scan_angle_deg,
                "sensor_z_range_mm": [
                    config.sensor_z_min_mm,
                    config.sensor_z_max_mm,
                ],
                "plane_normal_base": normal.tolist(),
                "T_ef_s_true": config.T_ef_s_true.tolist(),
            },
            "implementation_choices_not_disclosed_by_paper": {
                "plane_center_base_mm": center.tolist(),
                "nominal_sensor_distance_mm": config.nominal_sensor_distance_mm,
                "translation_tangent_span_mm": config.translation_tangent_span_mm,
                "translation_normal_span_mm": config.translation_normal_span_mm,
                "composite_target_span_mm": config.composite_target_span_mm,
                "composite_tilt_span_deg": config.composite_tilt_span_deg,
                "composite_roll_span_deg": config.composite_roll_span_deg,
                "composite_distance_span_mm": config.composite_distance_span_mm,
            },
            "analytic_model": analytic_model_dict(config),
            "noise": {
                "mode": config.sensor_noise.mode,
                "magnitude_mm": config.sensor_noise.magnitude_mm,
                "dropout_probability": config.sensor_noise.dropout_probability,
                "pose_translation_noise_std_mm": config.pose_translation_noise_std_mm,
                "pose_rotation_noise_std_deg": config.pose_rotation_noise_std_deg,
            },
        }
        dataset = Tan2025Dataset(
            translation_scans=translation_scans,
            composite_scans=composite_scans,
            metadata=metadata,
        )
        truth = Tan2025GroundTruth(
            T_ef_s=config.T_ef_s_true,
            plane_normal_base=normal,
            plane_offset_mm=plane_offset,
            true_translation_poses=translation_true,
            true_composite_poses=composite_true,
        )
        return SimulatedTan2025Dataset(dataset=dataset, truth=truth)

    def _translation_poses(
        self,
        rng: np.random.Generator,
        plane_center: np.ndarray,
        plane_normal: np.ndarray,
    ) -> list[np.ndarray]:
        config = self.config
        tangent_u, tangent_v = _plane_basis(plane_normal)
        reference_rotation = _reference_sensor_rotation(plane_normal)
        reference_rotation = reference_rotation @ euler_xyz_deg(2.0, -4.0, 15.0)
        sensor_origin = (
            plane_center
            - config.nominal_sensor_distance_mm * reference_rotation[:, 2]
        )
        T_base_sensor_reference = make_T(reference_rotation, sensor_origin)
        T_base_ef_reference = T_base_sensor_reference @ inv_T(config.T_ef_s_true)

        tangent_anchor = min(30.0, config.translation_tangent_span_mm)
        normal_anchor = min(18.0, config.translation_normal_span_mm)
        offsets = [
            np.zeros(3),
            tangent_anchor * tangent_u,
            tangent_anchor * tangent_v,
            normal_anchor * plane_normal,
        ]
        for _ in range(4, config.num_translation_poses):
            du, dv = rng.uniform(
                -config.translation_tangent_span_mm,
                config.translation_tangent_span_mm,
                size=2,
            )
            dn = rng.uniform(
                -config.translation_normal_span_mm,
                config.translation_normal_span_mm,
            )
            offsets.append(du * tangent_u + dv * tangent_v + dn * plane_normal)

        poses = []
        for offset in offsets:
            pose = T_base_ef_reference.copy()
            pose[:3, 3] += offset
            poses.append(pose)
        return poses

    def _composite_poses(
        self,
        rng: np.random.Generator,
        plane_center: np.ndarray,
        plane_normal: np.ndarray,
    ) -> list[np.ndarray]:
        config = self.config
        tangent_u, tangent_v = _plane_basis(plane_normal)
        reference_rotation = _reference_sensor_rotation(plane_normal)
        poses: list[np.ndarray] = []

        for index in range(config.num_composite_poses):
            if index < 6:
                anchors = (
                    (-0.75, -0.55, -0.8),
                    (0.70, -0.45, 0.7),
                    (-0.55, 0.70, 0.3),
                    (0.60, 0.65, -0.3),
                    (0.10, -0.10, 1.0),
                    (-0.15, 0.15, -1.0),
                )
                ax, ay, roll_scale = anchors[index]
                tilt_x = ax * config.composite_tilt_span_deg
                tilt_y = ay * config.composite_tilt_span_deg
                roll = roll_scale * config.composite_roll_span_deg
            else:
                tilt_x, tilt_y = rng.uniform(
                    -config.composite_tilt_span_deg,
                    config.composite_tilt_span_deg,
                    size=2,
                )
                roll = rng.uniform(
                    -config.composite_roll_span_deg,
                    config.composite_roll_span_deg,
                )
            rotation = reference_rotation @ euler_xyz_deg(tilt_x, tilt_y, roll)

            target_u, target_v = rng.uniform(
                -config.composite_target_span_mm,
                config.composite_target_span_mm,
                size=2,
            )
            target = plane_center + target_u * tangent_u + target_v * tangent_v
            distance = config.nominal_sensor_distance_mm + rng.uniform(
                -config.composite_distance_span_mm,
                config.composite_distance_span_mm,
            )
            sensor_origin = target - distance * rotation[:, 2]
            T_base_sensor = make_T(rotation, sensor_origin)
            poses.append(T_base_sensor @ inv_T(config.T_ef_s_true))
        return poses

    def _make_scan(
        self,
        true_T_base_ef: np.ndarray,
        rng: np.random.Generator,
        *,
        scan_id: int,
        motion_kind: str,
        plane_normal: np.ndarray,
        plane_offset: float,
    ) -> LaserScan:
        config = self.config
        true_T_base_sensor = true_T_base_ef @ config.T_ef_s_true
        half_angle = np.radians(config.scan_angle_deg * 0.5)
        angles = np.linspace(-half_angle, half_angle, config.num_profile_points)
        rays = np.column_stack(
            [np.sin(angles), np.zeros_like(angles), np.cos(angles)]
        )
        rays_base = rays @ true_T_base_sensor[:3, :3].T
        numerator = plane_offset - float(
            plane_normal @ true_T_base_sensor[:3, 3]
        )
        denominator = rays_base @ plane_normal
        ranges = np.full(config.num_profile_points, np.nan, dtype=float)
        stable = np.abs(denominator) > 1e-9
        ranges[stable] = numerator / denominator[stable]
        points = ranges[:, None] * rays
        valid = np.isfinite(ranges) & (ranges > 0.0)
        valid &= points[:, 2] >= config.sensor_z_min_mm
        valid &= points[:, 2] <= config.sensor_z_max_mm
        points[~valid] = np.nan

        if np.count_nonzero(valid) < int(
            np.ceil(config.minimum_valid_fraction * config.num_profile_points)
        ):
            raise RuntimeError(
                f"generated {motion_kind} scan {scan_id} has only "
                f"{np.count_nonzero(valid)}/{config.num_profile_points} valid rays; "
                "reduce pose spans or widen the sensor range"
            )

        points = self._apply_sensor_noise(points, rng)
        measured_pose = self._apply_pose_readback_noise(true_T_base_ef, rng)
        return LaserScan(
            T_base_ef=measured_pose,
            points_s=points,
            plane_id=0,
            scan_id=scan_id,
            meta={
                "motion_kind": motion_kind,
                "channel_ids": np.arange(config.num_profile_points, dtype=np.int64),
                "true_T_base_ef": true_T_base_ef.copy(),
            },
        )

    def _apply_sensor_noise(
        self,
        points_s: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        noise = self.config.sensor_noise
        points = np.asarray(points_s, dtype=float).copy()
        valid = np.all(np.isfinite(points), axis=1)
        if noise.mode == "gaussian_xz" and noise.magnitude_mm > 0.0:
            perturbation = rng.normal(
                0.0,
                noise.magnitude_mm,
                size=(int(np.count_nonzero(valid)), 2),
            )
            points[valid, 0] += perturbation[:, 0]
            points[valid, 2] += perturbation[:, 1]
        elif noise.mode == "uniform_xz" and noise.magnitude_mm > 0.0:
            perturbation = rng.uniform(
                -noise.magnitude_mm,
                noise.magnitude_mm,
                size=(int(np.count_nonzero(valid)), 2),
            )
            points[valid, 0] += perturbation[:, 0]
            points[valid, 2] += perturbation[:, 1]
        elif noise.mode == "constant_range_bias" and noise.magnitude_mm > 0.0:
            lengths = np.linalg.norm(points[valid], axis=1)
            points[valid] *= ((lengths + noise.magnitude_mm) / lengths)[:, None]

        if noise.dropout_probability > 0.0:
            dropout = rng.random(len(points)) < noise.dropout_probability
            points[dropout] = np.nan
        return points

    def _apply_pose_readback_noise(
        self,
        true_pose: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        config = self.config
        measured = np.asarray(true_pose, dtype=float).copy()
        if config.pose_translation_noise_std_mm > 0.0:
            measured[:3, 3] += rng.normal(
                0.0,
                config.pose_translation_noise_std_mm,
                size=3,
            )
        if config.pose_rotation_noise_std_deg > 0.0:
            rotvec = np.radians(
                rng.normal(0.0, config.pose_rotation_noise_std_deg, size=3)
            )
            measured[:3, :3] = (
                _rotation_from_rotvec(rotvec) @ measured[:3, :3]
            )
        return measured
