from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from .data import LaserScan

class RobotAdapter(ABC):
    """Implement this for Fanuc, RB5, ROS2, RoboDK, etc."""

    @abstractmethod
    def current_T_base_ef(self) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def move_linear(self, T_base_ef: np.ndarray, speed: float | None = None) -> None:
        raise NotImplementedError

class LaserAdapter(ABC):
    """Implement this for Keyence LJ-X, Micro-Epsilon scanCONTROL, etc."""

    @abstractmethod
    def capture_profile(self) -> np.ndarray:
        """Return Nx3 sensor-frame points. For 2D profile, y must be zero."""
        raise NotImplementedError


def capture_synchronized_scan(robot: RobotAdapter, laser: LaserAdapter) -> LaserScan:
    """Capture a profile with the matching flange pose.

    For moving scans, replace this with timestamp interpolation between robot pose
    samples and sensor profile timestamps.
    """
    T = robot.current_T_base_ef()
    pts = laser.capture_profile()
    return LaserScan(T_base_ef=T, points_s=pts)
