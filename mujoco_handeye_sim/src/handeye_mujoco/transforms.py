from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def rpy_matrix(rpy: Iterable[float]) -> np.ndarray:
    """URDF fixed-axis roll-pitch-yaw rotation (Rz(yaw) Ry(pitch) Rx(roll))."""
    roll, pitch, yaw = (float(v) for v in rpy)
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=float,
    )


def xyz_rpy_transform(xyz: Iterable[float], rpy: Iterable[float]) -> np.ndarray:
    transform = np.eye(4)
    transform[:3, :3] = rpy_matrix(rpy)
    transform[:3, 3] = np.asarray(tuple(xyz), dtype=float)
    return transform


def matrix_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    """Return a normalized MuJoCo quaternion in w, x, y, z order."""
    matrix = np.asarray(rotation, dtype=float)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        quat = np.array(
            [0.25 * s, (matrix[2, 1] - matrix[1, 2]) / s,
             (matrix[0, 2] - matrix[2, 0]) / s,
             (matrix[1, 0] - matrix[0, 1]) / s]
        )
    else:
        axis = int(np.argmax(np.diag(matrix)))
        if axis == 0:
            s = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quat = np.array([(matrix[2, 1] - matrix[1, 2]) / s, 0.25 * s,
                             (matrix[0, 1] + matrix[1, 0]) / s,
                             (matrix[0, 2] + matrix[2, 0]) / s])
        elif axis == 1:
            s = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quat = np.array([(matrix[0, 2] - matrix[2, 0]) / s,
                             (matrix[0, 1] + matrix[1, 0]) / s, 0.25 * s,
                             (matrix[1, 2] + matrix[2, 1]) / s])
        else:
            s = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quat = np.array([(matrix[1, 0] - matrix[0, 1]) / s,
                             (matrix[0, 2] + matrix[2, 0]) / s,
                             (matrix[1, 2] + matrix[2, 1]) / s, 0.25 * s])
    if quat[0] < 0:
        quat *= -1
    return quat / np.linalg.norm(quat)


def validate_transform(matrix: np.ndarray, name: str = "transform") -> np.ndarray:
    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (4, 4):
        raise ValueError(f"{name} must be 4x4; received {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} contains NaN or infinity")
    if not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-8):
        raise ValueError(f"{name} last row must be [0, 0, 0, 1]")
    rotation = matrix[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
        raise ValueError(f"{name} rotation is not orthonormal")
    if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5):
        raise ValueError(f"{name} rotation determinant must be +1")
    return matrix


def format_vector(values: Iterable[float]) -> str:
    return " ".join(f"{float(value):.10g}" for value in values)
