from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass
class LaserScan:
    """One 2D laser profile and the synchronized robot flange pose.

    T_base_ef: 4x4 homogeneous transform from end-effector/flange to robot base.
    points_s:  Nx3 points in laser sensor frame. For a 2D profile sensor, y should be 0.
    meta: optional dictionary for line id, scan parameters, timestamp, etc.
    """
    T_base_ef: np.ndarray
    points_s: np.ndarray
    meta: dict | None = None

    def __post_init__(self) -> None:
        self.T_base_ef = np.asarray(self.T_base_ef, dtype=float).reshape(4, 4)
        self.points_s = np.asarray(self.points_s, dtype=float)
        if self.points_s.ndim != 2 or self.points_s.shape[1] != 3:
            raise ValueError('points_s must have shape (N, 3)')
