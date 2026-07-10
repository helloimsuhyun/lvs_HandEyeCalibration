from __future__ import annotations
import numpy as np

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
