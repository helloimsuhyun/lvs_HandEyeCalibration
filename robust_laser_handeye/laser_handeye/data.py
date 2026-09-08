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


def _normalize_vector(
    value: np.ndarray,
    name: str,
) -> np.ndarray:
    vector = np.asarray(value, dtype=float).reshape(3)

    norm = float(np.linalg.norm(vector))

    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError(
            f"{name} must be finite and non-zero"
        )

    return vector / norm


@dataclass(frozen=True)
class PlaneFrame:
    """Right-handed target-plane frame in robot-base coordinates."""

    u: np.ndarray
    v: np.ndarray
    n: np.ndarray
    offset_mm: float

    def __post_init__(self) -> None:
        u = _normalize_vector(self.u, "plane u")
        v = _normalize_vector(self.v, "plane v")
        n = _normalize_vector(self.n, "plane normal")

        if not np.allclose(
            np.cross(u, v),
            n,
            atol=1e-8,
        ):
            raise ValueError(
                "plane frame must be right-handed and orthonormal"
            )

        if not np.isfinite(self.offset_mm):
            raise ValueError(
                "plane offset must be finite"
            )

        object.__setattr__(self, "u", u)
        object.__setattr__(self, "v", v)
        object.__setattr__(self, "n", n)
        object.__setattr__(
            self,
            "offset_mm",
            float(self.offset_mm),
        )

    @property
    def l(self) -> float:
        return self.offset_mm


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
