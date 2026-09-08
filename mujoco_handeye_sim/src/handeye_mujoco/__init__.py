"""Build and query a hand-eye MuJoCo model."""

from .builder import BuildError, build_model
from .planning import (
    automatic_scan_radius_mm,
    centered_convex_inradius,
    interpolate_joint_path,
    write_model_with_convex_board,
    write_model_with_infinite_plane,
    write_model_with_plane_box,
)
from .simulation import CollisionReport, HandEyeSimulation, IKResult

__all__ = [
    "BuildError",
    "CollisionReport",
    "HandEyeSimulation",
    "IKResult",
    "automatic_scan_radius_mm",
    "build_model",
    "centered_convex_inradius",
    "interpolate_joint_path",
    "write_model_with_convex_board",
    "write_model_with_infinite_plane",
    "write_model_with_plane_box",
]
