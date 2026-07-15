from __future__ import annotations

from dataclasses import dataclass, field
import numpy as np

# helper 함수들 --------------------------------------------------------------
def _validate_transform(T: np.ndarray, name: str) -> np.ndarray:
    T = np.asarray(T, dtype=float)

    if T.shape != (4, 4):
        raise ValueError(f"{name} must have shape (4, 4)")
    if not np.all(np.isfinite(T)):
        raise ValueError(f"{name} must contain only finite values")

    return T.copy()


def _validate_points(points: np.ndarray, name: str) -> np.ndarray:
    points = np.asarray(points, dtype=float)

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3)")

    return points.copy()


# 1개 프로파일 데이터와 해당하는 TCP transform을 담는 dataclass
@dataclass
class LaserScan:
    """One laser profile and its corresponding robot flange pose.

    Coordinate convention
    ---------------------
    T_base_ef:
        End-effector coordinates to robot-base coordinates. [mm]

    points_s:
        Laser profile points expressed in the sensor frame. [mm]
    """

    T_base_ef: np.ndarray
    points_s: np.ndarray

    plane_id: int = 0
    scan_id: int | None = None

    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.T_base_ef = _validate_transform(
            self.T_base_ef,
            "T_base_ef",
        )

        self.points_s = _validate_points(
            self.points_s,
            "points_s",
        )

        self.plane_id = int(self.plane_id)

        if self.scan_id is not None:
            self.scan_id = int(self.scan_id)

        self.meta = dict(self.meta)

    @property
    def num_points(self) -> int:
        return len(self.points_s)

    @property
    def valid_points_s(self) -> np.ndarray:
        """Return points whose coordinates are all finite."""
        valid = np.all(np.isfinite(self.points_s), axis=1)
        return self.points_s[valid]

# plane_id -> 해당 평면에서 획득한 LaserScan mapping
ScansByPlane = dict[int, list[LaserScan]]
