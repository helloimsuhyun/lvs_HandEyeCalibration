from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import importlib
import time
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass(frozen=True)
class ProfileSample:
    """One laser profile in the sensor frame and its host timestamp."""

    points_s: np.ndarray
    timestamp_ns: int

    def __post_init__(self) -> None:
        points = np.asarray(self.points_s, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points_s must have shape (N, 3)")
        object.__setattr__(self, "points_s", points.copy())
        object.__setattr__(self, "timestamp_ns", int(self.timestamp_ns))


class RobotInterface(ABC):
    """Minimal robot contract.

    ``current_T_base_tcp`` and ``move_tcp`` are the only robot-specific
    operations that must be implemented.  A vendor implementation should keep
    the controller's joint limits, singularity checks and collision monitoring
    enabled.  Positions are millimetres and rotations are proper matrices.
    """

    def connect(self) -> None:
        pass

    def close(self) -> None:
        pass

    @abstractmethod
    def current_T_base_tcp(self) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def move_tcp(
        self,
        T_base_tcp: np.ndarray,
        *,
        linear_speed_mm_s: float,
        angular_speed_deg_s: float,
        timeout_s: float,
    ) -> None:
        """Execute the reviewed Cartesian-linear TCP segment and block until stopped.

        Do not silently replace this with an unconstrained MoveJ path: the
        workspace/no-go visualization describes the Cartesian segment between
        the supplied interpolated waypoints.
        """
        raise NotImplementedError

    def stop(self) -> None:
        """Best-effort controlled stop. Override with the vendor stop call."""

    def supports_independent_stop(self) -> bool:
        """Whether stop can interrupt a concurrently blocking move command."""
        return False

    def controller_path_is_safe(self, poses: list[np.ndarray]) -> bool | None:
        """Optional controller/scene collision query.

        Return True only when the controller has checked the complete robot,
        not merely the TCP.  None means that no programmatic query is exposed.
        """
        return None


class LaserInterface(ABC):
    def connect(self) -> None:
        pass

    def close(self) -> None:
        pass

    @abstractmethod
    def capture_profile(self, *, timeout_s: float) -> ProfileSample:
        raise NotImplementedError


def transform_from_xyz_rpy_deg(values: list[float] | np.ndarray) -> np.ndarray:
    """Build ``T`` from [x, y, z, rx, ry, rz], extrinsic XYZ degrees."""
    values = np.asarray(values, dtype=float).reshape(6)
    T = np.eye(4)
    T[:3, :3] = Rotation.from_euler("xyz", values[3:], degrees=True).as_matrix()
    T[:3, 3] = values[:3]
    return T


def xyz_rpy_deg_from_transform(T: np.ndarray) -> np.ndarray:
    """Return [x, y, z, rx, ry, rz], extrinsic XYZ degrees."""
    T = np.asarray(T, dtype=float).reshape(4, 4)
    return np.concatenate(
        [T[:3, 3], Rotation.from_matrix(T[:3, :3]).as_euler("xyz", degrees=True)]
    )


def load_adapter(spec: str, kwargs: dict[str, Any]) -> Any:
    """Instantiate ``package.module:ClassName`` without importing vendor SDKs early."""
    if ":" not in spec:
        raise ValueError("adapter must have the form 'package.module:ClassName'")
    module_name, class_name = spec.split(":", 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    return cls(**dict(kwargs))


class MockRobot(RobotInterface):
    """Instantaneous in-memory robot used by tests and the mock demo."""

    def __init__(self, initial_T_base_tcp: list[list[float]] | None = None) -> None:
        self._pose = np.eye(4) if initial_T_base_tcp is None else np.asarray(
            initial_T_base_tcp, dtype=float
        ).reshape(4, 4)
        self.connected = False
        self.motion_log: list[np.ndarray] = []

    def connect(self) -> None:
        self.connected = True

    def close(self) -> None:
        self.connected = False

    def current_T_base_tcp(self) -> np.ndarray:
        return self._pose.copy()

    def move_tcp(
        self,
        T_base_tcp: np.ndarray,
        *,
        linear_speed_mm_s: float,
        angular_speed_deg_s: float,
        timeout_s: float,
    ) -> None:
        del linear_speed_mm_s, angular_speed_deg_s, timeout_s
        self._pose = np.asarray(T_base_tcp, dtype=float).reshape(4, 4).copy()
        self.motion_log.append(self._pose.copy())

    def controller_path_is_safe(self, poses: list[np.ndarray]) -> bool:
        return bool(poses)

    def supports_independent_stop(self) -> bool:
        return True


class MockPlanarLaser(LaserInterface):
    """Analytic planar profile source tied to a ``MockRobot``."""

    def __init__(
        self,
        robot: RobotInterface,
        T_tcp_sensor: np.ndarray,
        plane_normal_base: np.ndarray,
        plane_offset_mm: float,
        x_min_mm: float = -35.0,
        x_max_mm: float = 35.0,
        point_count: int = 151,
        noise_std_mm: float = 0.0,
        seed: int = 1,
    ) -> None:
        self.robot = robot
        self.T_tcp_sensor = np.asarray(T_tcp_sensor, dtype=float).reshape(4, 4)
        self.normal = np.asarray(plane_normal_base, dtype=float).reshape(3)
        self.normal /= np.linalg.norm(self.normal)
        self.offset = float(plane_offset_mm)
        self.x_values = np.linspace(float(x_min_mm), float(x_max_mm), int(point_count))
        self.noise_std = float(noise_std_mm)
        self.rng = np.random.default_rng(seed)

    def capture_profile(self, *, timeout_s: float) -> ProfileSample:
        del timeout_s
        T_base_sensor = self.robot.current_T_base_tcp() @ self.T_tcp_sensor
        R = T_base_sensor[:3, :3]
        origin = T_base_sensor[:3, 3]
        denominator = float(self.normal @ R[:, 2])
        if abs(denominator) < 1e-9:
            raise RuntimeError("mock scan plane is parallel to the target plane")
        z = (
            self.offset
            - float(self.normal @ origin)
            - self.x_values * float(self.normal @ R[:, 0])
        ) / denominator
        if self.noise_std > 0.0:
            z = z + self.rng.normal(0.0, self.noise_std, size=len(z))
        points = np.column_stack([self.x_values, np.zeros(len(z)), z])
        return ProfileSample(points_s=points, timestamp_ns=time.time_ns())
