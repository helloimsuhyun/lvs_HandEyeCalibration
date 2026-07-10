from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .data import LaserScan
from .geometry import scaled_normal_from_plane
from .se3 import make_T, project_to_so3, transform_points

@dataclass
class CalibrationResult:
    T_ef_s: np.ndarray
    iterations: int
    converged: bool
    plane_rms_history: list[float]
    delta_history: list[float]
    rank_history: list[int]
    cond_history: list[float]

# 현재 handeye 추정값을 사용해 point를 base coordinate로 표현
def reconstruct_points_base(scans: list[LaserScan], T_ef_s: np.ndarray) -> np.ndarray:
    all_pts = []
    for scan in scans:
        pts_ef = transform_points(T_ef_s, scan.points_s)
        pts_b = transform_points(scan.T_base_ef, pts_ef)
        all_pts.append(pts_b)
    return np.vstack(all_pts)

# scan data ps와 scaled 된 n을 사용해 Aw = Y 선형식 build
def build_linear_system(scans: list[LaserScan], n_scaled: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Unknown :
        w = [r11, r21, r31, r13, r23, r33, tx, ty, tz]^T
    """
    n = np.asarray(n_scaled, dtype=float).reshape(3)
    rows = []
    rhs = []
    nn = float(n @ n)
    for scan in scans:
        Rbe = scan.T_base_ef[:3, :3]
        tbe = scan.T_base_ef[:3, 3]
        a = n @ Rbe  # 1x3 row vector
        base_rhs = nn - float(n @ tbe)
        for p in scan.points_s:
            x, _, z = p
            rows.append(np.r_[x * a, z * a, a])
            rhs.append(base_rhs)
    return np.asarray(rows), np.asarray(rhs)

# R1, R3를 사용해 R2를 복원 > 이후 회전행렬을 SO(3) 만족하도록 변형
def solve_rotation_translation_linear(A: np.ndarray, y: np.ndarray) -> np.ndarray:
    w, *_ = np.linalg.lstsq(A, y, rcond=None)
    r1 = w[0:3]
    r3 = w[3:6]
    r2 = np.cross(r3, r1)  # R2 = R3 x R1
    R_raw = np.column_stack([r1, r2, r3])
    R = project_to_so3(R_raw) # RO(3) 만족하도록 R을 변형
    t = w[6:9]
    return make_T(R, t)

# SO(3)로 fixed 된 rotation로 만든 T를 사용해 R은 고정한 상태로 다시 t를 least - square 최적화
def refine_translation_with_fixed_rotation(scans: list[LaserScan], n_scaled: np.ndarray, R_ef_s: np.ndarray) -> np.ndarray:
    n = np.asarray(n_scaled, dtype=float).reshape(3)
    r1 = R_ef_s[:, 0]
    r3 = R_ef_s[:, 2]
    rows = []
    rhs = []
    nn = float(n @ n)
    for scan in scans:
        Rbe = scan.T_base_ef[:3, :3]
        tbe = scan.T_base_ef[:3, 3]
        a = n @ Rbe
        for p in scan.points_s:
            x, _, z = p
            y_i = nn - float(n @ tbe) - float(x * (a @ r1)) - float(z * (a @ r3))
            rows.append(a)
            rhs.append(y_i)
    t, *_ = np.linalg.lstsq(np.asarray(rows), np.asarray(rhs), rcond=None)
    return t


def calibrate_single_plane(
    scans: list[LaserScan],
    T_init: np.ndarray,
    max_iter: int = 100,
    tol: float = 1e-9,
    min_rank: int = 9,
) -> CalibrationResult:
    """
    input : 
    scans - scan 프로파일 데이터
    T_init - 초기의 hand-eye transform

    """
    if len(scans) == 0:
        raise ValueError('at least one scan is required')
    T = np.asarray(T_init, dtype=float).reshape(4, 4).copy()
    plane_rms_history: list[float] = []
    delta_history: list[float] = []
    rank_history: list[int] = []
    cond_history: list[float] = []

    converged = False
    for k in range(max_iter):
        pts_b = reconstruct_points_base(scans, T) # point와 초기 T를 가지고 scan data를 recon
        n_scaled, plane_rms = scaled_normal_from_plane(pts_b) # recon된 평면을 가지고 PCA를 통해 scaled n을 구함
        A, y = build_linear_system(scans, n_scaled) # Aw = Y를 구함
        rank = int(np.linalg.matrix_rank(A)) # A행렬의 rank
        s = np.linalg.svd(A, compute_uv=False)
        cond = float(s[0] / s[-1]) if s[-1] > 0 else float('inf')
        if rank < min_rank:
            raise np.linalg.LinAlgError(f'coefficient matrix rank {rank} < {min_rank}; calibration data lacks variation')

        T_new = solve_rotation_translation_linear(A, y)
        t_refined = refine_translation_with_fixed_rotation(scans, n_scaled, T_new[:3, :3])
        T_new[:3, 3] = t_refined

        delta = float(np.linalg.norm(T_new - T))
        plane_rms_history.append(plane_rms)
        delta_history.append(delta)
        rank_history.append(rank)
        cond_history.append(cond)
        T = T_new
        if delta < tol:
            converged = True
            break

    return CalibrationResult(T, k + 1, converged, plane_rms_history, delta_history, rank_history, cond_history)


def calibrate_with_known_plane(scans: list[LaserScan], plane_n_unit: np.ndarray, plane_l: float) -> np.ndarray:
    """One-shot LS calibration when the calibration plane in robot base is known.

    This is useful for synthetic validation or a fixture whose plane pose was
    measured independently. The plane equation is n_unit^T p = plane_l.
    """
    n_unit = np.asarray(plane_n_unit, dtype=float).reshape(3)
    n_unit = n_unit / np.linalg.norm(n_unit)
    n_scaled = float(plane_l) * n_unit
    A, y = build_linear_system(scans, n_scaled)
    if np.linalg.matrix_rank(A) < 9:
        raise np.linalg.LinAlgError('coefficient matrix is rank deficient')
    T = solve_rotation_translation_linear(A, y)
    T[:3, 3] = refine_translation_with_fixed_rotation(scans, n_scaled, T[:3, :3])
    return T
