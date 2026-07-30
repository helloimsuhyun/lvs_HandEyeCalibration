from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


LIVE_ACKNOWLEDGEMENT = "I_CHECKED_THE_ROBOT_CELL"


def rotation_distance_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    relative = np.asarray(R_a).reshape(3, 3).T @ np.asarray(R_b).reshape(3, 3)
    return float(np.degrees(Rotation.from_matrix(relative).magnitude()))


def validate_transform(T: np.ndarray, name: str = "T") -> np.ndarray:
    T = np.asarray(T, dtype=float)
    if T.shape != (4, 4) or not np.all(np.isfinite(T)):
        raise ValueError(f"{name} must be a finite 4x4 matrix")
    if not np.allclose(T[3], [0.0, 0.0, 0.0, 1.0], atol=1e-8):
        raise ValueError(f"{name} has an invalid homogeneous last row")
    R = T[:3, :3]
    if not np.allclose(R.T @ R, np.eye(3), atol=1e-6) or not np.isclose(
        np.linalg.det(R), 1.0, atol=1e-6
    ):
        raise ValueError(f"{name} rotation is not in SO(3)")
    return T.copy()


@dataclass(frozen=True)
class AxisAlignedBox:
    minimum_mm: np.ndarray
    maximum_mm: np.ndarray
    name: str = "box"

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, name: str) -> "AxisAlignedBox":
        minimum = np.asarray(data["min"], dtype=float).reshape(3)
        maximum = np.asarray(data["max"], dtype=float).reshape(3)
        if np.any(minimum >= maximum):
            raise ValueError(f"{name}: every min must be smaller than max")
        return cls(minimum, maximum, name)

    def contains(self, point_mm: np.ndarray, margin_mm: float = 0.0) -> bool:
        point = np.asarray(point_mm, dtype=float).reshape(3)
        return bool(
            np.all(point >= self.minimum_mm - float(margin_mm))
            and np.all(point <= self.maximum_mm + float(margin_mm))
        )

    def intersects_segment(self, start_mm: np.ndarray, end_mm: np.ndarray) -> bool:
        """Return whether a closed TCP line segment intersects this box."""
        start = np.asarray(start_mm, dtype=float).reshape(3)
        end = np.asarray(end_mm, dtype=float).reshape(3)
        direction = end - start
        lower, upper = 0.0, 1.0
        for axis in range(3):
            if abs(float(direction[axis])) < 1e-12:
                if start[axis] < self.minimum_mm[axis] or start[axis] > self.maximum_mm[axis]:
                    return False
                continue
            first = (self.minimum_mm[axis] - start[axis]) / direction[axis]
            second = (self.maximum_mm[axis] - start[axis]) / direction[axis]
            entry, leave = min(first, second), max(first, second)
            lower = max(lower, float(entry))
            upper = min(upper, float(leave))
            if lower > upper:
                return False
        return True


@dataclass
class SafetyConfig:
    workspace: AxisAlignedBox
    no_go_boxes: list[AxisAlignedBox] = field(default_factory=list)
    safe_transit_T_base_tcp: np.ndarray | None = None
    approach_clearance_mm: float = 40.0
    max_linear_step_mm: float = 10.0
    max_angular_step_deg: float = 5.0
    linear_speed_mm_s: float = 20.0
    angular_speed_deg_s: float = 10.0
    motion_timeout_s: float = 30.0
    settle_time_s: float = 0.25
    readback_position_tolerance_mm: float = 1.0
    readback_rotation_tolerance_deg: float = 1.0
    stationarity_position_tolerance_mm: float = 0.2
    stationarity_rotation_tolerance_deg: float = 0.2
    min_sensor_plane_clearance_mm: float = 15.0
    initial_handeye_uncertainty_mm: float = 0.0
    initial_handeye_uncertainty_deg: float = 0.0
    require_controller_collision_check: bool = True
    controller_collision_check_acknowledged: bool = False
    live_enabled: bool = False

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SafetyConfig":
        if "workspace_mm" not in data:
            raise ValueError("safety.workspace_mm is required")
        workspace = AxisAlignedBox.from_dict(data["workspace_mm"], name="workspace")
        boxes = [
            AxisAlignedBox.from_dict(item, name=str(item.get("name", f"no_go_{i}")))
            for i, item in enumerate(data.get("no_go_boxes_mm", []))
        ]
        transit_raw = data.get("safe_transit_T_base_tcp")
        transit = None if transit_raw is None else validate_transform(
            np.asarray(transit_raw, dtype=float), "safe_transit_T_base_tcp"
        )
        kwargs = {
            key: data[key]
            for key in (
                "approach_clearance_mm",
                "max_linear_step_mm",
                "max_angular_step_deg",
                "linear_speed_mm_s",
                "angular_speed_deg_s",
                "motion_timeout_s",
                "settle_time_s",
                "readback_position_tolerance_mm",
                "readback_rotation_tolerance_deg",
                "stationarity_position_tolerance_mm",
                "stationarity_rotation_tolerance_deg",
                "min_sensor_plane_clearance_mm",
                "initial_handeye_uncertainty_mm",
                "initial_handeye_uncertainty_deg",
                "require_controller_collision_check",
                "controller_collision_check_acknowledged",
                "live_enabled",
            )
            if key in data
        }
        config = cls(workspace=workspace, no_go_boxes=boxes, safe_transit_T_base_tcp=transit, **kwargs)
        config.validate_values()
        return config

    def validate_values(self) -> None:
        positive = {
            "approach_clearance_mm": self.approach_clearance_mm,
            "max_linear_step_mm": self.max_linear_step_mm,
            "max_angular_step_deg": self.max_angular_step_deg,
            "linear_speed_mm_s": self.linear_speed_mm_s,
            "angular_speed_deg_s": self.angular_speed_deg_s,
            "motion_timeout_s": self.motion_timeout_s,
            "readback_position_tolerance_mm": self.readback_position_tolerance_mm,
            "readback_rotation_tolerance_deg": self.readback_rotation_tolerance_deg,
            "stationarity_position_tolerance_mm": self.stationarity_position_tolerance_mm,
            "stationarity_rotation_tolerance_deg": self.stationarity_rotation_tolerance_deg,
            "min_sensor_plane_clearance_mm": self.min_sensor_plane_clearance_mm,
        }
        for name, value in positive.items():
            if float(value) <= 0.0:
                raise ValueError(f"safety.{name} must be positive")
        if self.settle_time_s < 0.0:
            raise ValueError("safety.settle_time_s must be non-negative")
        if self.initial_handeye_uncertainty_mm < 0.0:
            raise ValueError("safety.initial_handeye_uncertainty_mm must be non-negative")
        if not 0.0 <= self.initial_handeye_uncertainty_deg <= 180.0:
            raise ValueError(
                "safety.initial_handeye_uncertainty_deg must be in [0, 180]"
            )

    def assert_pose_safe(self, T_base_tcp: np.ndarray, *, name: str) -> None:
        T = validate_transform(T_base_tcp, name)
        point = T[:3, 3]
        if not self.workspace.contains(point):
            raise ValueError(f"{name} TCP {point.tolist()} is outside workspace")
        for box in self.no_go_boxes:
            if box.contains(point):
                raise ValueError(f"{name} TCP {point.tolist()} is inside no-go box '{box.name}'")

    def assert_live_unlocked(self, acknowledgement: str | None) -> None:
        if not self.live_enabled:
            raise RuntimeError("live motion is disabled in config (safety.live_enabled=false)")
        if acknowledgement != LIVE_ACKNOWLEDGEMENT:
            raise RuntimeError(
                "live acknowledgement mismatch; pass --acknowledge-risk "
                f"{LIVE_ACKNOWLEDGEMENT} after checking the cell"
            )
        if self.safe_transit_T_base_tcp is None:
            raise RuntimeError("live motion requires safety.safe_transit_T_base_tcp")
        if (
            self.require_controller_collision_check
            and not self.controller_collision_check_acknowledged
        ):
            raise RuntimeError(
                "confirm the robot controller's full-link collision/joint-limit checks "
                "and set safety.controller_collision_check_acknowledged=true"
            )


def interpolate_segment(
    start: np.ndarray,
    end: np.ndarray,
    *,
    max_linear_step_mm: float,
    max_angular_step_deg: float,
) -> list[np.ndarray]:
    start = validate_transform(start, "segment start")
    end = validate_transform(end, "segment end")
    linear = float(np.linalg.norm(end[:3, 3] - start[:3, 3]))
    angular = rotation_distance_deg(start[:3, :3], end[:3, :3])
    count = max(
        1,
        int(np.ceil(linear / float(max_linear_step_mm))),
        int(np.ceil(angular / float(max_angular_step_deg))),
    )
    key_rotations = Rotation.from_matrix(np.stack([start[:3, :3], end[:3, :3]]))
    slerp = Slerp([0.0, 1.0], key_rotations)
    result: list[np.ndarray] = []
    for fraction, rotation in zip(
        np.linspace(1.0 / count, 1.0, count),
        slerp(np.linspace(1.0 / count, 1.0, count)).as_matrix(),
    ):
        T = np.eye(4)
        T[:3, :3] = rotation
        T[:3, 3] = (1.0 - fraction) * start[:3, 3] + fraction * end[:3, 3]
        result.append(T)
    return result


def validate_segment(start: np.ndarray, end: np.ndarray, safety: SafetyConfig, *, name: str) -> list[np.ndarray]:
    poses = interpolate_segment(
        start,
        end,
        max_linear_step_mm=safety.max_linear_step_mm,
        max_angular_step_deg=safety.max_angular_step_deg,
    )
    safety.assert_pose_safe(start, name=f"{name} start")
    previous = np.asarray(start, dtype=float)
    for index, pose in enumerate(poses):
        safety.assert_pose_safe(pose, name=f"{name} waypoint {index}")
        for box in safety.no_go_boxes:
            if box.intersects_segment(previous[:3, 3], pose[:3, 3]):
                raise ValueError(
                    f"{name} segment {index} intersects no-go box '{box.name}'"
                )
        previous = pose
    return poses
