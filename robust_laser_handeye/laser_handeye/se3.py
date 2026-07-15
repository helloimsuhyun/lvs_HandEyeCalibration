from __future__ import annotations
import numpy as np


def euler_xyz_deg(rx: float, ry: float, rz: float) -> np.ndarray:
    """Return ``Rz(rz) @ Ry(ry) @ Rx(rx)`` for angles in degrees."""
    ax, ay, az = np.radians([rx, ry, rz])
    cx, sx = np.cos(ax), np.sin(ax)
    cy, sy = np.cos(ay), np.sin(ay)
    cz, sz = np.cos(az), np.sin(az)

    R_x = np.array(
        [[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]],
        dtype=float,
    )
    R_y = np.array(
        [[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]],
        dtype=float,
    )
    R_z = np.array(
        [[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    return R_z @ R_y @ R_x

# R,t -> T (SE3)
def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=float).reshape(3, 3)
    t = np.asarray(t, dtype=float).reshape(3)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

# inv_T
def inv_T(T: np.ndarray) -> np.ndarray:
    T = np.asarray(T, dtype=float).reshape(4, 4)
    R = T[:3, :3]
    t = T[:3, 3]
    return make_T(R.T, -R.T @ t)

# p' = T * p
def transform_points(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError('pts must have shape (N, 3)')
    return pts @ T[:3, :3].T + T[:3, 3]

# cross product 통해 구한 R이 SO(3) constraint를 만족하도록 수정
def project_to_so3(R: np.ndarray) -> np.ndarray:
    U, _, Vt = np.linalg.svd(np.asarray(R, dtype=float).reshape(3, 3))
    Rp = U @ Vt
    if np.linalg.det(Rp) < 0:
        U[:, -1] *= -1
        Rp = U @ Vt
    return Rp


def rot_error_deg(R_est: np.ndarray, R_true: np.ndarray) -> float:
    R = R_est @ R_true.T
    c = (np.trace(R) - 1.0) / 2.0
    c = np.clip(c, -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))

def rotation_vector_error_deg(
    R_est: np.ndarray,
    R_true: np.ndarray,
) -> np.ndarray:
    """Return the relative SO(3) rotation vector in degrees.

    The relative rotation is defined as:

        R_error = R_est @ R_true.T

    The output is a 3-vector:

        rotation_vector = angle * axis

    where each component is expressed in degrees.

    Notes
    -----
    The norm of the returned vector is approximately equal to:

        rot_error_deg(R_est, R_true)

    for ordinary non-singular rotations.
    """
    R_est = np.asarray(R_est, dtype=float).reshape(3, 3)
    R_true = np.asarray(R_true, dtype=float).reshape(3, 3)

    R_error = R_est @ R_true.T

    cos_angle = (np.trace(R_error) - 1.0) / 2.0
    cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
    angle_rad = float(np.arccos(cos_angle))

    # Almost identical rotations
    if angle_rad < 1e-12:
        return np.zeros(3, dtype=float)

    # General case, away from 180 degrees
    sin_angle = float(np.sin(angle_rad))
    if abs(sin_angle) > 1e-8:
        axis = np.array(
            [
                R_error[2, 1] - R_error[1, 2],
                R_error[0, 2] - R_error[2, 0],
                R_error[1, 0] - R_error[0, 1],
            ],
            dtype=float,
        ) / (2.0 * sin_angle)

        axis_norm = float(np.linalg.norm(axis))
        if axis_norm > 0.0:
            axis /= axis_norm

        return np.degrees(angle_rad) * axis

    # Near 180 degrees, the usual 1 / sin(angle) formula is unstable.
    # Recover the axis from the diagonal terms.
    diagonal = np.diag(R_error)
    axis = np.sqrt(np.maximum((diagonal + 1.0) / 2.0, 0.0))

    largest = int(np.argmax(axis))

    if largest == 0 and axis[0] > 1e-8:
        axis[1] = (
            R_error[0, 1] + R_error[1, 0]
        ) / (4.0 * axis[0])
        axis[2] = (
            R_error[0, 2] + R_error[2, 0]
        ) / (4.0 * axis[0])

    elif largest == 1 and axis[1] > 1e-8:
        axis[0] = (
            R_error[0, 1] + R_error[1, 0]
        ) / (4.0 * axis[1])
        axis[2] = (
            R_error[1, 2] + R_error[2, 1]
        ) / (4.0 * axis[1])

    elif largest == 2 and axis[2] > 1e-8:
        axis[0] = (
            R_error[0, 2] + R_error[2, 0]
        ) / (4.0 * axis[2])
        axis[1] = (
            R_error[1, 2] + R_error[2, 1]
        ) / (4.0 * axis[2])

    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-12:
        return np.zeros(3, dtype=float)

    axis /= axis_norm

    # Resolve the remaining sign ambiguity using the skew-symmetric part.
    skew_vector = np.array(
        [
            R_error[2, 1] - R_error[1, 2],
            R_error[0, 2] - R_error[2, 0],
            R_error[1, 0] - R_error[0, 1],
        ],
        dtype=float,
    )

    if float(axis @ skew_vector) < 0.0:
        axis = -axis

    return np.degrees(angle_rad) * axis