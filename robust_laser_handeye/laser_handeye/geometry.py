from __future__ import annotations
import numpy as np

# 스캔 Point들의 PCA와 무게중심으로 추정 board 방정식 구함
def fit_plane_pca(points: np.ndarray) -> tuple[np.ndarray, float, np.ndarray, float]:

    P = np.asarray(points, dtype=float)
    if P.ndim != 2 or P.shape[1] != 3 or len(P) < 3:
        raise ValueError('points must have shape (N, 3), N>=3')
    c = P.mean(axis=0)
    _, _, Vt = np.linalg.svd(P - c, full_matrices=False)
    n = Vt[-1]
    n /= np.linalg.norm(n)
    l = float(n @ c)
    if l < 0:
        n = -n
        l = -l
    residuals = P @ n - l
    rms = float(np.sqrt(np.mean(residuals**2)))
    return n, l, c, rms

# scale encoging된 normal vector 구함
def scaled_normal_from_plane(points: np.ndarray) -> tuple[np.ndarray, float]:
    """Return paper-style scaled normal n = l*n_unit and plane fit RMS.

    With this convention, points on the plane satisfy n^T p = ||n||^2.
    """
    n_unit, l, _, rms = fit_plane_pca(points)
    return l * n_unit, rms
