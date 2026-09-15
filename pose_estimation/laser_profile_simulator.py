#!/usr/bin/env python3
"""
Laser-profile scan simulator for CAD surface-scan trajectories.

Designed to consume the output of surface_scan_path.py without modifying the
path planner itself.

Coordinate convention expected from the planner
-----------------------------------------------
    sensor frame +X : laser profile direction
    sensor frame +Y : scan-motion direction
    sensor frame +Z : outward CAD normal
    viewing direction: -Z

The measurement area is represented by a parameterized trapezoidal ROI in the
sensor X-Z plane. The near/far depth, width, and lateral center can be changed
without changing simulator code, so the same module can model different laser
profile sensors. KEYENCE LJ-V7080 is provided only as a preset.

Each normalized profile coordinate u in [-1,+1] connects its corresponding point
on the near ROI edge to the far ROI edge. These lines form the ray family used
for CAD ray casting. This supports expanding, rectangular, contracting, and
laterally shifted/skewed trapezoidal ROIs without requiring a shared virtual
ray origin.

Import usage
------------
    from surface_scan_path import SurfaceScanPlanner, ScanPlannerConfig
    from laser_profile_simulator import (
        LaserProfileSimulator,
        LaserProfileConfig,
        TrapezoidalROI,
    )

    planner = SurfaceScanPlanner("part.stl", ScanPlannerConfig(sensor_standoff_mm=80.0))
    plan = planner.plan(waypoints_m)

    simulator = LaserProfileSimulator(
        planner.mesh,
        LaserProfileConfig.lj_v7080(),
    )
    result = simulator.simulate_plan(plan)
    print(result.summary())
    result.save_npz("simulated_scan.npz")

Standalone usage
----------------
    python laser_profile_simulator.py part.stl \
        --scan-json scan_path.json \
        --mesh-unit auto \
        --show \
        --save-ply simulated_scan.ply \
        --save-npz simulated_scan.npz

Notes
-----
* This is a geometric simulator, not an optical/material renderer.
* Reflectance, exposure, multi-path, saturation and controller-side filtering are
  intentionally not synthesized yet.
* Optional Gaussian X/Z noise and random dropout can be enabled explicitly.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Iterable

import numpy as np

try:
    import open3d as o3d
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise RuntimeError("Open3D is required: pip install open3d") from exc


EPS = 1.0e-12


def _normalize_rows(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    n = np.linalg.norm(v, axis=1)
    if np.any(n <= EPS):
        raise ValueError("Cannot normalize zero-length ray direction.")
    return v / n[:, None]


@dataclass(frozen=True)
class TrapezoidalROI:
    """Parameterized 2-D measurement ROI in the sensor X-Z plane.

    The ROI is defined by two horizontal cross-sections:

        near plane: depth = near_depth_mm, center = near_center_x_mm,
                    width = near_width_mm
        far plane : depth = far_depth_mm,  center = far_center_x_mm,
                    width = far_width_mm

    The left/right boundaries are linearly interpolated between those planes.
    This represents a general trapezoid, including:
      * expanding FOV (far_width > near_width),
      * constant-width rectangular ROI,
      * contracting FOV, and
      * laterally shifted / skewed trapezoids through different X centers.

    All values are millimetres in the sensor frame. Positive depth means the
    viewing direction (-Z in the planner convention).
    """

    near_depth_mm: float = 57.0
    far_depth_mm: float = 103.0
    near_width_mm: float = 25.0
    far_width_mm: float = 39.0
    near_center_x_mm: float = 0.0
    far_center_x_mm: float = 0.0

    def validate(self) -> None:
        if self.near_depth_mm <= 0.0:
            raise ValueError("ROI near_depth_mm must be positive")
        if self.far_depth_mm <= self.near_depth_mm:
            raise ValueError("ROI far_depth_mm must be larger than near_depth_mm")
        if self.near_width_mm <= 0.0 or self.far_width_mm <= 0.0:
            raise ValueError("ROI near/far widths must be positive")
        vals = np.asarray([
            self.near_depth_mm, self.far_depth_mm,
            self.near_width_mm, self.far_width_mm,
            self.near_center_x_mm, self.far_center_x_mm,
        ], dtype=np.float64)
        if not np.all(np.isfinite(vals)):
            raise ValueError("ROI parameters must be finite")

    def _alpha(self, depth_mm: np.ndarray | float) -> np.ndarray:
        d = np.asarray(depth_mm, dtype=np.float64)
        return (d - self.near_depth_mm) / (self.far_depth_mm - self.near_depth_mm)

    def width_mm(self, depth_mm: np.ndarray | float) -> np.ndarray:
        """Linearly interpolated ROI width at the requested depth."""
        a = self._alpha(depth_mm)
        return (1.0 - a) * self.near_width_mm + a * self.far_width_mm

    def center_x_mm(self, depth_mm: np.ndarray | float) -> np.ndarray:
        """Linearly interpolated lateral ROI center at the requested depth."""
        a = self._alpha(depth_mm)
        return (1.0 - a) * self.near_center_x_mm + a * self.far_center_x_mm

    def x_bounds_mm(self, depth_mm: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
        center = self.center_x_mm(depth_mm)
        half = 0.5 * self.width_mm(depth_mm)
        return center - half, center + half

    def x_for_u_mm(self, u: np.ndarray, depth_mm: float) -> np.ndarray:
        """Map normalized profile coordinate u in [-1,+1] to sensor X."""
        u = np.asarray(u, dtype=np.float64)
        return self.center_x_mm(depth_mm) + 0.5 * self.width_mm(depth_mm) * u

    def ray_endpoints_mm(self, u: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return near/far X-Z endpoints for each normalized profile sample.

        Connecting corresponding near/far points creates a straight ray family
        that exactly tiles the parameterized trapezoid. Unlike the old single
        virtual-origin approximation, this works for expanding, rectangular,
        contracting, and skewed trapezoidal ROIs.
        """
        u = np.asarray(u, dtype=np.float64)
        near = np.column_stack([
            self.x_for_u_mm(u, self.near_depth_mm),
            np.zeros_like(u),
            -np.full_like(u, self.near_depth_mm),
        ])
        far = np.column_stack([
            self.x_for_u_mm(u, self.far_depth_mm),
            np.zeros_like(u),
            -np.full_like(u, self.far_depth_mm),
        ])
        return near, far


@dataclass(frozen=True)
class LaserProfileConfig:
    """Generic geometric laser-profile sensor model.

    The sensor-independent geometry lives in ``roi``. ``reference_distance_mm``
    is only the plane at which profile sampling pitch is specified; it does not
    define the ROI shape. Set ``profile_samples`` when the sensor has a fixed
    number of output points, or leave it as None to infer the count from
    ``x_interval_mm`` at the reference plane.
    """

    model_name: str = "KEYENCE LJ-V7080"
    roi: TrapezoidalROI = field(default_factory=TrapezoidalROI)

    reference_distance_mm: float = 80.0
    x_interval_mm: float = 0.05
    profile_samples: int | None = None

    # Optional synthetic measurement corruption. Defaults intentionally represent
    # ideal geometric ray casting rather than claiming real single-shot noise.
    x_noise_sigma_mm: float = 0.0
    z_noise_sigma_mm: float = 0.0
    dropout_rate: float = 0.0
    random_seed: int | None = 0

    # Optional geometric rejection. None disables incidence rejection.
    max_incidence_angle_deg: float | None = None

    # Small tolerance used when checking depth/FOV limits.
    range_tolerance_mm: float = 1.0e-3

    @classmethod
    def lj_v7080(cls, **overrides) -> "LaserProfileConfig":
        roi = overrides.pop(
            "roi",
            TrapezoidalROI(
                near_depth_mm=57.0,
                far_depth_mm=103.0,
                near_width_mm=25.0,
                far_width_mm=39.0,
                near_center_x_mm=0.0,
                far_center_x_mm=0.0,
            ),
        )
        values = dict(
            model_name="KEYENCE LJ-V7080",
            roi=roi,
            reference_distance_mm=80.0,
            x_interval_mm=0.05,
            profile_samples=None,
        )
        values.update(overrides)
        return cls(**values)

    @classmethod
    def custom_trapezoid(
        cls,
        *,
        model_name: str = "Custom laser profile sensor",
        near_depth_mm: float,
        far_depth_mm: float,
        near_width_mm: float,
        far_width_mm: float,
        reference_distance_mm: float,
        x_interval_mm: float = 0.05,
        profile_samples: int | None = None,
        near_center_x_mm: float = 0.0,
        far_center_x_mm: float = 0.0,
        **kwargs,
    ) -> "LaserProfileConfig":
        return cls(
            model_name=model_name,
            roi=TrapezoidalROI(
                near_depth_mm=near_depth_mm,
                far_depth_mm=far_depth_mm,
                near_width_mm=near_width_mm,
                far_width_mm=far_width_mm,
                near_center_x_mm=near_center_x_mm,
                far_center_x_mm=far_center_x_mm,
            ),
            reference_distance_mm=reference_distance_mm,
            x_interval_mm=x_interval_mm,
            profile_samples=profile_samples,
            **kwargs,
        )

    # Compatibility/readability aliases used by existing diagnostics.
    @property
    def near_depth_mm(self) -> float:
        return float(self.roi.near_depth_mm)

    @property
    def far_depth_mm(self) -> float:
        return float(self.roi.far_depth_mm)

    @property
    def x_width_near_mm(self) -> float:
        return float(self.roi.near_width_mm)

    @property
    def x_width_far_mm(self) -> float:
        return float(self.roi.far_width_mm)

    @property
    def x_width_reference_mm(self) -> float:
        return float(self.roi.width_mm(self.reference_distance_mm))

    def validate(self) -> None:
        self.roi.validate()
        if not (self.near_depth_mm <= self.reference_distance_mm <= self.far_depth_mm):
            raise ValueError(
                "reference_distance_mm must lie inside the trapezoidal ROI depth range"
            )
        if self.x_interval_mm <= 0.0:
            raise ValueError("x_interval_mm must be positive")
        if self.profile_samples is not None and int(self.profile_samples) < 2:
            raise ValueError("profile_samples must be >= 2 when specified")
        if self.x_noise_sigma_mm < 0.0 or self.z_noise_sigma_mm < 0.0:
            raise ValueError("noise sigmas must be non-negative")
        if not (0.0 <= self.dropout_rate <= 1.0):
            raise ValueError("dropout_rate must lie in [0, 1]")
        if self.max_incidence_angle_deg is not None:
            if not (0.0 < self.max_incidence_angle_deg <= 90.0):
                raise ValueError("max_incidence_angle_deg must lie in (0, 90]")

    def width_mm(self, depth_mm: np.ndarray | float) -> np.ndarray:
        return self.roi.width_mm(depth_mm)

    def center_x_mm(self, depth_mm: np.ndarray | float) -> np.ndarray:
        return self.roi.center_x_mm(depth_mm)

    def x_bounds_mm(self, depth_mm: np.ndarray | float) -> tuple[np.ndarray, np.ndarray]:
        return self.roi.x_bounds_mm(depth_mm)

    def sample_count(self) -> int:
        if self.profile_samples is not None:
            return int(self.profile_samples)
        width = self.x_width_reference_mm
        n_intervals = max(1, int(round(width / self.x_interval_mm)))
        return n_intervals + 1

    def normalized_samples(self) -> np.ndarray:
        return np.linspace(-1.0, 1.0, self.sample_count(), dtype=np.float64)

    def reference_x_samples_mm(self) -> np.ndarray:
        """Profile X coordinates at ``reference_distance_mm``."""
        return self.roi.x_for_u_mm(self.normalized_samples(), self.reference_distance_mm)


@dataclass
class ProfileScan:
    segment_id: int
    pose_index: int
    sensor_position_m: np.ndarray
    R_cad_sensor: np.ndarray
    reference_x_m: np.ndarray
    true_points_m: np.ndarray
    measured_points_m: np.ndarray
    valid_mask: np.ndarray
    primitive_ids: np.ndarray
    local_x_m: np.ndarray
    depth_m: np.ndarray
    incidence_angle_deg: np.ndarray

    @property
    def valid_count(self) -> int:
        return int(np.count_nonzero(self.valid_mask))

    @property
    def sample_count(self) -> int:
        return int(len(self.valid_mask))


@dataclass
class ScanSimulationResult:
    config: LaserProfileConfig
    profiles: list[ProfileScan] = field(default_factory=list)

    def _flatten(self, measured: bool = True):
        points = []
        segment_ids = []
        pose_ids = []
        sample_ids = []
        depths = []
        local_x = []
        incidence = []

        for profile in self.profiles:
            ids = np.flatnonzero(profile.valid_mask)
            if len(ids) == 0:
                continue
            P = profile.measured_points_m if measured else profile.true_points_m
            points.append(P[ids])
            segment_ids.append(np.full(len(ids), profile.segment_id, dtype=np.int32))
            pose_ids.append(np.full(len(ids), profile.pose_index, dtype=np.int32))
            sample_ids.append(ids.astype(np.int32))
            depths.append(profile.depth_m[ids])
            local_x.append(profile.local_x_m[ids])
            incidence.append(profile.incidence_angle_deg[ids])

        if not points:
            return {
                "points_m": np.empty((0, 3), dtype=np.float64),
                "segment_ids": np.empty(0, dtype=np.int32),
                "pose_ids": np.empty(0, dtype=np.int32),
                "sample_ids": np.empty(0, dtype=np.int32),
                "depth_m": np.empty(0, dtype=np.float64),
                "local_x_m": np.empty(0, dtype=np.float64),
                "incidence_angle_deg": np.empty(0, dtype=np.float64),
            }

        return {
            "points_m": np.concatenate(points, axis=0),
            "segment_ids": np.concatenate(segment_ids),
            "pose_ids": np.concatenate(pose_ids),
            "sample_ids": np.concatenate(sample_ids),
            "depth_m": np.concatenate(depths),
            "local_x_m": np.concatenate(local_x),
            "incidence_angle_deg": np.concatenate(incidence),
        }

    @property
    def points_m(self) -> np.ndarray:
        return self._flatten(measured=True)["points_m"]

    @property
    def true_points_m(self) -> np.ndarray:
        return self._flatten(measured=False)["points_m"]

    def summary(self) -> dict:
        total_samples = int(sum(p.sample_count for p in self.profiles))
        total_valid = int(sum(p.valid_count for p in self.profiles))
        profile_counts = np.asarray([p.valid_count for p in self.profiles], dtype=np.int64)
        return {
            "sensor_model": self.config.model_name,
            "profiles": int(len(self.profiles)),
            "samples_per_profile": int(self.profiles[0].sample_count) if self.profiles else 0,
            "total_rays": total_samples,
            "valid_hits": total_valid,
            "hit_rate": (float(total_valid) / total_samples) if total_samples else 0.0,
            "valid_points_per_profile_median": (
                float(np.median(profile_counts)) if len(profile_counts) else 0.0
            ),
            "valid_points_per_profile_min": int(np.min(profile_counts)) if len(profile_counts) else 0,
            "valid_points_per_profile_max": int(np.max(profile_counts)) if len(profile_counts) else 0,
        }

    def print_summary(self) -> None:
        s = self.summary()
        print("\n[LASER PROFILE SIMULATION]")
        print(f"  sensor              : {s['sensor_model']}")
        print(f"  profiles            : {s['profiles']}")
        print(f"  samples/profile     : {s['samples_per_profile']}")
        print(f"  valid hits          : {s['valid_hits']:,} / {s['total_rays']:,} ({100*s['hit_rate']:.2f}%)")
        print(
            "  valid/profile       : "
            f"min {s['valid_points_per_profile_min']} | "
            f"median {s['valid_points_per_profile_median']:.1f} | "
            f"max {s['valid_points_per_profile_max']}"
        )

    def save_npz(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        m = self._flatten(measured=True)
        t = self._flatten(measured=False)
        np.savez_compressed(
            path,
            points_m=m["points_m"],
            true_points_m=t["points_m"],
            segment_ids=m["segment_ids"],
            pose_ids=m["pose_ids"],
            sample_ids=m["sample_ids"],
            depth_m=m["depth_m"],
            local_x_m=m["local_x_m"],
            incidence_angle_deg=m["incidence_angle_deg"],
            sensor_model=np.asarray(self.config.model_name),
            roi_near_depth_mm=np.asarray(self.config.roi.near_depth_mm),
            roi_far_depth_mm=np.asarray(self.config.roi.far_depth_mm),
            roi_near_width_mm=np.asarray(self.config.roi.near_width_mm),
            roi_far_width_mm=np.asarray(self.config.roi.far_width_mm),
            roi_near_center_x_mm=np.asarray(self.config.roi.near_center_x_mm),
            roi_far_center_x_mm=np.asarray(self.config.roi.far_center_x_mm),
            reference_distance_mm=np.asarray(self.config.reference_distance_mm),
            profile_samples=np.asarray(self.config.sample_count()),
        )

    def save_ply(self, path: str | Path, measured: bool = True) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        P = self.points_m if measured else self.true_points_m
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(P)
        if not o3d.io.write_point_cloud(str(path), cloud, write_ascii=False):
            raise RuntimeError(f"Failed to write PLY: {path}")

    def save_csv(self, path: str | Path, measured: bool = True) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        f = self._flatten(measured=measured)
        P = f["points_m"]
        with path.open("w", newline="", encoding="utf-8") as fp:
            writer = csv.writer(fp)
            writer.writerow(
                [
                    "segment_id", "pose_id", "sample_id",
                    "x_m", "y_m", "z_m",
                    "profile_x_m", "depth_m", "incidence_angle_deg",
                ]
            )
            for i in range(len(P)):
                writer.writerow(
                    [
                        int(f["segment_ids"][i]),
                        int(f["pose_ids"][i]),
                        int(f["sample_ids"][i]),
                        float(P[i, 0]), float(P[i, 1]), float(P[i, 2]),
                        float(f["local_x_m"][i]),
                        float(f["depth_m"][i]),
                        float(f["incidence_angle_deg"][i]),
                    ]
                )


class LaserProfileSimulator:
    """Ray-cast laser-profile returns from a triangle-mesh CAD model."""

    def __init__(self, mesh, config: LaserProfileConfig | None = None):
        self.config = config or LaserProfileConfig.lj_v7080()
        self.config.validate()

        self.mesh = o3d.geometry.TriangleMesh(mesh)
        if self.mesh.is_empty() or len(self.mesh.triangles) == 0:
            raise ValueError("mesh must be a non-empty triangle mesh")
        self.mesh.compute_triangle_normals()

        self.scene = o3d.t.geometry.RaycastingScene()
        tensor_mesh = o3d.t.geometry.TriangleMesh.from_legacy(self.mesh)
        self.scene.add_triangles(tensor_mesh)

        self.reference_x_m = self.config.reference_x_samples_mm() / 1000.0
        self.reference_depth_m = self.config.reference_distance_mm / 1000.0
        self.profile_u = self.config.normalized_samples()

        # Generic trapezoidal ROI ray family. Each normalized profile coordinate u
        # has a point on the near ROI edge and the corresponding point on the far
        # ROI edge. The connecting line is the measurement ray for that sample.
        # This removes the old requirement that all rays share one virtual origin.
        near_mm, far_mm = self.config.roi.ray_endpoints_mm(self.profile_u)
        self.local_ray_origins = near_mm / 1000.0
        self.local_ray_dirs = _normalize_rows((far_mm - near_mm) / 1000.0)

        self.rng = np.random.default_rng(self.config.random_seed)

    def _world_rays(self, sensor_position_m: np.ndarray, R_cad_sensor: np.ndarray):
        p = np.asarray(sensor_position_m, dtype=np.float64).reshape(3)
        R = np.asarray(R_cad_sensor, dtype=np.float64).reshape(3, 3)

        origins = p[None, :] + (R @ self.local_ray_origins.T).T
        directions = (R @ self.local_ray_dirs.T).T
        directions = _normalize_rows(directions)
        return origins, directions

    def simulate_pose(
        self,
        sensor_position_m: np.ndarray,
        R_cad_sensor: np.ndarray,
        *,
        segment_id: int = 0,
        pose_index: int = 0,
    ) -> ProfileScan:
        cfg = self.config
        p = np.asarray(sensor_position_m, dtype=np.float64).reshape(3)
        R = np.asarray(R_cad_sensor, dtype=np.float64).reshape(3, 3)

        origins, directions = self._world_rays(p, R)
        rays_np = np.hstack([origins, directions]).astype(np.float32)
        rays = o3d.core.Tensor(rays_np, dtype=o3d.core.Dtype.Float32)
        ans = self.scene.cast_rays(rays)

        t_hit = np.asarray(ans["t_hit"].numpy(), dtype=np.float64).reshape(-1)
        primitive_ids = np.asarray(ans["primitive_ids"].numpy(), dtype=np.int64).reshape(-1)

        true_points = np.full((len(t_hit), 3), np.nan, dtype=np.float64)
        finite = np.isfinite(t_hit) & (t_hit >= 0.0)
        true_points[finite] = origins[finite] + t_hit[finite, None] * directions[finite]

        local = np.full_like(true_points, np.nan)
        if np.any(finite):
            local[finite] = (R.T @ (true_points[finite] - p[None, :]).T).T

        local_x = local[:, 0].copy()
        depth = -local[:, 2].copy()  # positive forward distance
        depth_mm = 1000.0 * depth
        tol = float(cfg.range_tolerance_mm)
        valid = finite.copy()
        valid &= depth_mm >= cfg.near_depth_mm - tol
        valid &= depth_mm <= cfg.far_depth_mm + tol

        left_mm, right_mm = cfg.x_bounds_mm(depth_mm)
        signed_x_mm = 1000.0 * local_x
        valid &= signed_x_mm >= left_mm - tol
        valid &= signed_x_mm <= right_mm + tol

        incidence = np.full(len(t_hit), np.nan, dtype=np.float64)
        if np.any(finite):
            # primitive_normals are CAD/world normals at the hit triangle.  Flip-insensitive
            # incidence uses |n . (-ray)| so STL winding does not change rejection.
            if "primitive_normals" in ans:
                N = np.asarray(ans["primitive_normals"].numpy(), dtype=np.float64).reshape(-1, 3)
                nmag = np.linalg.norm(N, axis=1)
                good_n = finite & np.isfinite(nmag) & (nmag > EPS)
                cos_i = np.full(len(t_hit), np.nan, dtype=np.float64)
                cos_i[good_n] = np.abs(
                    np.sum((N[good_n] / nmag[good_n, None]) * (-directions[good_n]), axis=1)
                )
                cos_i = np.clip(cos_i, 0.0, 1.0)
                incidence[good_n] = np.degrees(np.arccos(cos_i[good_n]))

        if cfg.max_incidence_angle_deg is not None:
            valid &= np.isfinite(incidence)
            valid &= incidence <= float(cfg.max_incidence_angle_deg)

        measured = true_points.copy()

        if np.any(valid) and (cfg.x_noise_sigma_mm > 0.0 or cfg.z_noise_sigma_mm > 0.0):
            ids = np.flatnonzero(valid)
            measured_local = local[ids].copy()
            if cfg.x_noise_sigma_mm > 0.0:
                measured_local[:, 0] += self.rng.normal(
                    0.0, cfg.x_noise_sigma_mm / 1000.0, size=len(ids)
                )
            if cfg.z_noise_sigma_mm > 0.0:
                # Noise is applied to positive forward depth, therefore local z gets
                # the opposite sign.
                dz = self.rng.normal(0.0, cfg.z_noise_sigma_mm / 1000.0, size=len(ids))
                measured_local[:, 2] -= dz
            measured[ids] = p[None, :] + (R @ measured_local.T).T

        if cfg.dropout_rate > 0.0 and np.any(valid):
            ids = np.flatnonzero(valid)
            dropped = self.rng.random(len(ids)) < cfg.dropout_rate
            valid[ids[dropped]] = False

        measured[~valid] = np.nan
        true_points[~valid] = np.nan
        local_x[~valid] = np.nan
        depth[~valid] = np.nan
        incidence[~valid] = np.nan
        primitive_ids[~valid] = -1

        return ProfileScan(
            segment_id=int(segment_id),
            pose_index=int(pose_index),
            sensor_position_m=p.copy(),
            R_cad_sensor=R.copy(),
            reference_x_m=self.reference_x_m.copy(),
            true_points_m=true_points,
            measured_points_m=measured,
            valid_mask=valid,
            primitive_ids=primitive_ids,
            local_x_m=local_x,
            depth_m=depth,
            incidence_angle_deg=incidence,
        )

    def simulate_segments(self, segments: Iterable[dict]) -> ScanSimulationResult:
        profiles: list[ProfileScan] = []
        for seg in segments:
            sid = int(seg.get("segment_id", len(profiles)))
            sensor_points = np.asarray(seg["sensor_points"], dtype=np.float64)
            frames = np.asarray(seg["frames"], dtype=np.float64)
            if sensor_points.ndim != 2 or sensor_points.shape[1] != 3:
                raise ValueError(f"S{sid}: sensor_points must be N x 3")
            if frames.shape != (len(sensor_points), 3, 3):
                raise ValueError(f"S{sid}: frames must have shape (N,3,3)")

            for pose_i, (p, R) in enumerate(zip(sensor_points, frames)):
                profiles.append(
                    self.simulate_pose(
                        p,
                        R,
                        segment_id=sid,
                        pose_index=pose_i,
                    )
                )

        return ScanSimulationResult(config=self.config, profiles=profiles)

    def simulate_plan(self, plan) -> ScanSimulationResult:
        """Simulate a surface_scan_path.ScanPlan (duck-typed via .segments)."""
        if not hasattr(plan, "segments"):
            raise TypeError("plan must expose a .segments list")
        return self.simulate_segments(plan.segments)


# -----------------------------------------------------------------------------
# JSON bridge for standalone usage
# -----------------------------------------------------------------------------


def load_segments_from_scan_json(path: str | Path) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    raw_segments = data.get("segments", [])
    if not raw_segments:
        raise RuntimeError(f"No segments found in scan JSON: {path}")

    segments = []
    for raw in raw_segments:
        sid = int(raw["segment_id"])
        poses = raw.get("poses", [])
        if not poses:
            raise RuntimeError(f"S{sid} has no poses")
        sensor_points = np.asarray([p["sensor_position_m"] for p in poses], dtype=np.float64)
        frames = np.asarray([p["R_cad_sensor"] for p in poses], dtype=np.float64)
        surface_points = np.asarray(
            [p.get("surface_position_m", [np.nan, np.nan, np.nan]) for p in poses],
            dtype=np.float64,
        )
        segments.append(
            {
                "segment_id": sid,
                "sensor_points": sensor_points,
                "frames": frames,
                "surface_points": surface_points,
            }
        )
    return segments


def _load_mesh_same_as_planner(cad_path: Path, mesh_unit: str, weld_tolerance_mm: float):
    """Reuse the planner's loader so standalone simulation stays in the same CAD frame."""
    try:
        from surface_scan_path import load_mesh_preserve_frame
    except ImportError as exc:
        raise RuntimeError(
            "Standalone mode expects laser_profile_simulator.py to be placed beside "
            "surface_scan_path.py so it can reuse load_mesh_preserve_frame()."
        ) from exc
    mesh, info = load_mesh_preserve_frame(cad_path, mesh_unit, weld_tolerance_mm)
    return mesh, info


# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------


def _make_segmented_sensor_path(segments: Iterable[dict]):
    points = []
    lines = []
    base = 0
    for seg in segments:
        P = np.asarray(seg["sensor_points"], dtype=np.float64)
        points.extend(P)
        if len(P) >= 2:
            lines.extend([[base + i, base + i + 1] for i in range(len(P) - 1)])
        base += len(P)

    ls = o3d.geometry.LineSet()
    if points:
        ls.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    if lines:
        ls.lines = o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
        ls.colors = o3d.utility.Vector3dVector(
            np.repeat(np.asarray([[0.05, 0.35, 1.0]]), len(lines), axis=0)
        )
    return ls


def _profile_lines(result: ScanSimulationResult, stride: int = 10):
    points = []
    lines = []
    base = 0
    stride = max(1, int(stride))

    for profile_i, profile in enumerate(result.profiles):
        if profile_i % stride != 0:
            continue
        P = profile.measured_points_m
        valid = profile.valid_mask
        ids = np.flatnonzero(valid)
        if len(ids) < 2:
            continue

        # Add only adjacent valid samples so gaps/occlusions are not bridged.
        local_map = {}
        for sample_id in ids:
            local_map[int(sample_id)] = base + len(points)
            points.append(P[sample_id])

        for a, b in zip(ids[:-1], ids[1:]):
            if int(b) == int(a) + 1:
                lines.append([local_map[int(a)], local_map[int(b)]])

    ls = o3d.geometry.LineSet()
    if points:
        ls.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    if lines:
        ls.lines = o3d.utility.Vector2iVector(np.asarray(lines, dtype=np.int32))
        ls.colors = o3d.utility.Vector3dVector(
            np.repeat(np.asarray([[1.0, 0.05, 0.75]]), len(lines), axis=0)
        )
    return ls


def visualize_simulation(
    mesh,
    segments: Iterable[dict],
    result: ScanSimulationResult,
    *,
    profile_stride: int = 10,
    point_size: float = 2.0,
):
    cad = o3d.geometry.TriangleMesh(mesh)
    cad.paint_uniform_color([0.65, 0.65, 0.68])

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(result.points_m)
    if len(result.points_m):
        cloud.paint_uniform_color([0.95, 0.10, 0.10])

    sensor_path = _make_segmented_sensor_path(segments)
    profile_lines = _profile_lines(result, stride=profile_stride)

    print("\n[SIMULATION VIEW]")
    print("  Gray CAD      : triangle mesh")
    print("  Red points    : simulated valid laser-profile returns")
    print("  Magenta lines : selected scan profiles")
    print("  Blue line     : sensor trajectory")

    vis = o3d.visualization.Visualizer()
    vis.create_window("Laser profile scan simulation", width=1500, height=920)
    vis.add_geometry(cad)
    if len(result.points_m):
        vis.add_geometry(cloud)
    if len(sensor_path.lines):
        vis.add_geometry(sensor_path)
    if len(profile_lines.lines):
        vis.add_geometry(profile_lines)
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.96, 0.97, 0.98])
    opt.point_size = float(point_size)
    vis.run()
    vis.destroy_window()


# -----------------------------------------------------------------------------
# Diagnostics
# -----------------------------------------------------------------------------


def planned_center_depth_report(segments: Iterable[dict], config: LaserProfileConfig) -> dict:
    """Check whether planner standoff places nominal surface centers inside Z range."""
    depths = []
    for seg in segments:
        if "surface_points" not in seg:
            continue
        surf = np.asarray(seg["surface_points"], dtype=np.float64)
        sens = np.asarray(seg["sensor_points"], dtype=np.float64)
        R = np.asarray(seg["frames"], dtype=np.float64)
        if surf.shape != sens.shape or R.shape != (len(sens), 3, 3):
            continue
        finite = np.all(np.isfinite(surf), axis=1)
        for p_surf, p_sens, rot in zip(surf[finite], sens[finite], R[finite]):
            local = rot.T @ (p_surf - p_sens)
            depths.append(-float(local[2]))

    d = np.asarray(depths, dtype=np.float64)
    if len(d) == 0:
        return {"count": 0}

    near = config.near_depth_mm / 1000.0
    far = config.far_depth_mm / 1000.0
    inside = (d >= near) & (d <= far)
    return {
        "count": int(len(d)),
        "min_mm": float(np.min(d) * 1000.0),
        "median_mm": float(np.median(d) * 1000.0),
        "max_mm": float(np.max(d) * 1000.0),
        "inside_fraction": float(np.mean(inside)),
    }


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Ray-cast a generic trapezoidal-ROI laser profile sensor along a "
            "generated CAD scan path."
        )
    )
    p.add_argument("cad", type=Path)
    p.add_argument("--scan-json", type=Path, required=True)
    p.add_argument("--mesh-unit", choices=("auto", "m", "mm"), default="auto")
    p.add_argument("--weld-tolerance-mm", type=float, default=0.001)

    # Sensor geometry. LJ-V7080 remains the default preset; every ROI field can be
    # overridden independently, so the same simulator can represent other sensors.
    p.add_argument("--sensor-preset", choices=("lj_v7080", "custom"), default="lj_v7080")
    p.add_argument("--model-name", type=str, default=None)
    p.add_argument("--roi-near-depth-mm", type=float, default=None)
    p.add_argument("--roi-far-depth-mm", type=float, default=None)
    p.add_argument("--roi-near-width-mm", type=float, default=None)
    p.add_argument("--roi-far-width-mm", type=float, default=None)
    p.add_argument("--roi-near-center-x-mm", type=float, default=None)
    p.add_argument("--roi-far-center-x-mm", type=float, default=None)
    p.add_argument("--reference-distance-mm", type=float, default=None)
    p.add_argument("--x-interval-mm", type=float, default=None)
    p.add_argument(
        "--profile-samples",
        type=int,
        default=None,
        help=(
            "Fixed output samples per profile. If omitted, infer count from the "
            "reference-plane width / --x-interval-mm."
        ),
    )

    p.add_argument("--x-noise-sigma-mm", type=float, default=0.0)
    p.add_argument("--z-noise-sigma-mm", type=float, default=0.0)
    p.add_argument("--dropout-rate", type=float, default=0.0)
    p.add_argument("--max-incidence-angle-deg", type=float, default=None)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--save-npz", type=Path, default=None)
    p.add_argument("--save-ply", type=Path, default=None)
    p.add_argument("--save-csv", type=Path, default=None)
    p.add_argument("--show", action="store_true")
    p.add_argument("--profile-stride", type=int, default=10)
    p.add_argument("--point-size", type=float, default=2.0)
    return p.parse_args()


def config_from_args(args) -> LaserProfileConfig:
    """Build a sensor config from a preset plus optional trapezoid overrides."""
    if args.sensor_preset == "lj_v7080":
        base = LaserProfileConfig.lj_v7080()
        default_name = base.model_name
    else:
        # Neutral custom defaults make CLI experimentation possible without
        # changing code, but users should normally specify the actual datasheet.
        base = LaserProfileConfig(
            model_name="Custom laser profile sensor",
            roi=TrapezoidalROI(),
            reference_distance_mm=80.0,
            x_interval_mm=0.05,
        )
        default_name = base.model_name

    b = base.roi
    roi = TrapezoidalROI(
        near_depth_mm=b.near_depth_mm if args.roi_near_depth_mm is None else args.roi_near_depth_mm,
        far_depth_mm=b.far_depth_mm if args.roi_far_depth_mm is None else args.roi_far_depth_mm,
        near_width_mm=b.near_width_mm if args.roi_near_width_mm is None else args.roi_near_width_mm,
        far_width_mm=b.far_width_mm if args.roi_far_width_mm is None else args.roi_far_width_mm,
        near_center_x_mm=(
            b.near_center_x_mm
            if args.roi_near_center_x_mm is None
            else args.roi_near_center_x_mm
        ),
        far_center_x_mm=(
            b.far_center_x_mm
            if args.roi_far_center_x_mm is None
            else args.roi_far_center_x_mm
        ),
    )

    cfg = LaserProfileConfig(
        model_name=args.model_name or default_name,
        roi=roi,
        reference_distance_mm=(
            base.reference_distance_mm
            if args.reference_distance_mm is None
            else args.reference_distance_mm
        ),
        x_interval_mm=(
            base.x_interval_mm if args.x_interval_mm is None else args.x_interval_mm
        ),
        profile_samples=args.profile_samples,
        x_noise_sigma_mm=args.x_noise_sigma_mm,
        z_noise_sigma_mm=args.z_noise_sigma_mm,
        dropout_rate=args.dropout_rate,
        max_incidence_angle_deg=args.max_incidence_angle_deg,
        random_seed=args.seed,
    )
    cfg.validate()
    return cfg


def main():
    args = parse_args()
    if args.weld_tolerance_mm < 0.0:
        raise ValueError("--weld-tolerance-mm must be >= 0")
    if args.x_noise_sigma_mm < 0.0 or args.z_noise_sigma_mm < 0.0:
        raise ValueError("noise sigmas must be >= 0")
    if not (0.0 <= args.dropout_rate <= 1.0):
        raise ValueError("--dropout-rate must be in [0,1]")

    mesh, mesh_info = _load_mesh_same_as_planner(
        args.cad, args.mesh_unit, args.weld_tolerance_mm
    )
    segments = load_segments_from_scan_json(args.scan_json)

    cfg = config_from_args(args)

    print("[CAD]")
    print(f"  unit interpretation : {mesh_info['input_unit']}")
    print(f"  diameter            : {mesh_info['diameter_mm']:.3f} mm")
    print("[SENSOR]")
    print(f"  model               : {cfg.model_name}")
    print(f"  ROI depth           : {cfg.near_depth_mm:.3f} .. {cfg.far_depth_mm:.3f} mm")
    print(
        "  ROI width           : "
        f"{cfg.x_width_near_mm:.3f} -> {cfg.x_width_far_mm:.3f} mm (near -> far)"
    )
    print(
        "  ROI center X        : "
        f"{cfg.roi.near_center_x_mm:.3f} -> {cfg.roi.far_center_x_mm:.3f} mm"
    )
    print(f"  reference distance  : {cfg.reference_distance_mm:.3f} mm")
    print(f"  reference width     : {cfg.x_width_reference_mm:.3f} mm")
    if cfg.profile_samples is None:
        print(f"  reference X interval: {cfg.x_interval_mm:.3f} mm")
    else:
        print(f"  fixed profile count : {cfg.profile_samples}")
    print(f"  samples/profile     : {cfg.sample_count()}")

    depth_report = planned_center_depth_report(segments, cfg)
    if depth_report.get("count", 0):
        print("[PLANNED CENTER DEPTH]")
        print(
            f"  min/median/max      : {depth_report['min_mm']:.3f} / "
            f"{depth_report['median_mm']:.3f} / {depth_report['max_mm']:.3f} mm"
        )
        print(f"  inside sensor ROI depth: {100*depth_report['inside_fraction']:.1f}%")
        if depth_report["inside_fraction"] < 0.999:
            print(
                f"  [WARN] Some planned surface centers are outside "
                f"{cfg.near_depth_mm:.3f}..{cfg.far_depth_mm:.3f} mm. "
                f"Choose planner standoff near the sensor working distance."
            )

    simulator = LaserProfileSimulator(mesh, cfg)
    result = simulator.simulate_segments(segments)
    result.print_summary()

    if args.save_npz is not None:
        result.save_npz(args.save_npz)
        print(f"Saved NPZ : {args.save_npz}")
    if args.save_ply is not None:
        result.save_ply(args.save_ply)
        print(f"Saved PLY : {args.save_ply}")
    if args.save_csv is not None:
        result.save_csv(args.save_csv)
        print(f"Saved CSV : {args.save_csv}")

    if args.show:
        visualize_simulation(
            mesh,
            segments,
            result,
            profile_stride=args.profile_stride,
            point_size=args.point_size,
        )


if __name__ == "__main__":
    main()