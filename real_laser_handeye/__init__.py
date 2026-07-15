"""Real-hardware workflow for single-plane 2D-laser hand-eye calibration.

All transforms use millimetres and the convention ``T_A_B`` (coordinates in
frame B to coordinates in frame A).  The robot TCP is the end-effector frame
used by the calibration solver, so the hand-eye transform is ``T_tcp_sensor``.
"""

from .hardware import LaserInterface, ProfileSample, RobotInterface

__all__ = ["LaserInterface", "ProfileSample", "RobotInterface"]
