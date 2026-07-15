# 사용자가 single plane에서 4개의 point를 찍으면
# 초기 handeye와 4개 포인트에 대한 스캔 데이터를 바탕으로 
# 현재 평면에 대한 n, l을 추정 -> 캘리브레이션을 진행할 plane의 boudary를 제약


# 1. 얻어진 4개의 프로파일 데이터를 바탕으로 초기 handeye로 평면을 추정
# 2. 4개 프로파일의 경계로 min max u,v를 구해 평면 좌표계에서 캘리브레이션 스캔 boundary를 잡음

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def load_T(path: str | Path) -> np.ndarray:
    T = np.loadtxt(path, delimiter=",")
    if T.shape != (4, 4):
        T = np.loadtxt(path)

    T = np.asarray(T, dtype=float)
    if T.shape != (4, 4):
        raise ValueError(f"{path}: expected 4x4 matrix, got {T.shape}")
    return T


def load_profile(path: str | Path) -> np.ndarray:
    """
    Load laser profile points.

    Supported:
        N x 2: [x_s, z_s]
        N x 3: [x_s, y_s, z_s]
    """
    try:
        points = np.loadtxt(path, delimiter=",", skiprows=1)
    except ValueError:
        points = np.loadtxt(path, delimiter=",")

    points = np.asarray(points, dtype=float)

    if points.ndim == 1:
        points = points.reshape(1, -1)

    if points.shape[1] == 2:
        # [x, z] -> [x, 0, z]
        points = np.column_stack(
            [points[:, 0], np.zeros(len(points)), points[:, 1]]
        )
    elif points.shape[1] != 3:
        raise ValueError(
            f"{path}: expected Nx2 [x,z] or Nx3 [x,y,z], got {points.shape}"
        )

    points = points[np.all(np.isfinite(points), axis=1)]
    return points

# 점 좌표계 변환
def transform_points(T: np.ndarray, points: np.ndarray) -> np.ndarray:
    """p_out = R p_in + t."""
    return points @ T[:3, :3].T + T[:3, 3]


def normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm < 1e-12:
        raise ValueError("zero-length vector")
    return v / norm


def fit_plane(points_base: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Fit plane with SVD.

    Plane equation:
        n^T p = l
    """
    center = np.mean(points_base, axis=0)
    _, _, Vt = np.linalg.svd(points_base - center, full_matrices=False)

    normal = normalize(Vt[-1])
    offset = float(normal @ center)

    return center, normal, offset


def first_scan_direction(points_base: np.ndarray) -> np.ndarray:
    """Principal direction of the first laser line."""
    center = np.mean(points_base, axis=0)
    _, _, Vt = np.linalg.svd(points_base - center, full_matrices=False)
    return normalize(Vt[0])


def make_plane_frame(
    center: np.ndarray,
    normal: np.ndarray,
    first_scan_points: np.ndarray,
    sensor_origins: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Plane frame:
        u: first scan-line direction
        v: n x u
        n: plane normal
    """

    # Make the normal point toward the sensors.
    mean_sensor_origin = np.mean(sensor_origins, axis=0)
    if normal @ (mean_sensor_origin - center) < 0:
        normal = -normal

    line_direction = first_scan_direction(first_scan_points)

    # Project the scan direction onto the plane.
    u = line_direction - (line_direction @ normal) * normal
    u = normalize(u)
    v = normalize(np.cross(normal, u))

    T_base_plane = np.eye(4)
    T_base_plane[:3, 0] = u
    T_base_plane[:3, 1] = v
    T_base_plane[:3, 2] = normal
    T_base_plane[:3, 3] = center

    return u, v, normal, T_base_plane


def project_to_plane_uv(
    points_base: np.ndarray,
    center: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
) -> np.ndarray:
    q = points_base - center
    return np.column_stack([q @ u, q @ v])


def uv_to_base(
    uv: np.ndarray,
    center: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
) -> np.ndarray:
    uv = np.asarray(uv, dtype=float)
    return center + uv[:, [0]] * u + uv[:, [1]] * v


def estimate_plane_and_boundary(
    T_ef_s_init: np.ndarray,
    T_base_ef_list: list[np.ndarray],
    profile_list: list[np.ndarray],
    margin_mm: float = 10.0,
) -> dict:
    """
    Estimate plane and rectangular safe boundary from four scans.

    Transform convention:
        T_base_s = T_base_ef @ T_ef_s_init
    """
    if len(T_base_ef_list) != 4 or len(profile_list) != 4:
        raise ValueError("exactly four robot poses and four profiles are required")
    margin_mm = float(margin_mm)
    if not np.isfinite(margin_mm) or margin_mm < 0.0:
        raise ValueError("margin_mm must be finite and non-negative")

    scan_points_base = []
    sensor_origins = []

    for T_base_ef, points_s in zip(T_base_ef_list, profile_list):
        T_base_s = T_base_ef @ T_ef_s_init
        points_base = transform_points(T_base_s, points_s)

        scan_points_base.append(points_base)
        sensor_origins.append(T_base_s[:3, 3])

    all_points_base = np.vstack(scan_points_base)
    sensor_origins = np.asarray(sensor_origins)

    center, normal, offset = fit_plane(all_points_base)

    u, v, normal, T_base_plane = make_plane_frame(
        center=center,
        normal=normal,
        first_scan_points=scan_points_base[0],
        sensor_origins=sensor_origins,
    )
    offset = float(normal @ center)

    uv = project_to_plane_uv(all_points_base, center, u, v)

    observed = {
        "u_min": float(np.min(uv[:, 0])),
        "u_max": float(np.max(uv[:, 0])),
        "v_min": float(np.min(uv[:, 1])),
        "v_max": float(np.max(uv[:, 1])),
    }

    safe = {
        "u_min": observed["u_min"] + margin_mm,
        "u_max": observed["u_max"] - margin_mm,
        "v_min": observed["v_min"] + margin_mm,
        "v_max": observed["v_max"] - margin_mm,
    }

    if safe["u_min"] >= safe["u_max"] or safe["v_min"] >= safe["v_max"]:
        raise ValueError("margin is too large for the observed region")

    safe_uv_corners = np.array(
        [
            [safe["u_min"], safe["v_min"]],
            [safe["u_max"], safe["v_min"]],
            [safe["u_max"], safe["v_max"]],
            [safe["u_min"], safe["v_max"]],
        ]
    )

    safe_base_corners = uv_to_base(
        safe_uv_corners,
        center,
        u,
        v,
    )

    distances = (all_points_base - center) @ normal

    return {
        "plane": {
            "normal_base": normal.tolist(),
            "offset_mm": offset,
            "center_base_mm": center.tolist(),
            "rms_error_mm": float(np.sqrt(np.mean(distances**2))),
        },
        "plane_frame": {
            "T_base_plane": T_base_plane.tolist(),
        },
        "observed_bounds_uv_mm": observed,
        "safe_bounds_uv_mm": safe,
        "safe_corners_base_mm": safe_base_corners.tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--handeye", required=True)
    parser.add_argument("--poses", nargs=4, required=True)
    parser.add_argument("--profiles", nargs=4, required=True)

    parser.add_argument("--margin-mm", type=float, default=0.0)
    parser.add_argument("--output", default="plane_boundary.json")

    args = parser.parse_args()

    T_ef_s_init = load_T(args.handeye)
    T_base_ef_list = [load_T(path) for path in args.poses]
    profile_list = [load_profile(path) for path in args.profiles]

    result = estimate_plane_and_boundary(
        T_ef_s_init=T_ef_s_init,
        T_base_ef_list=T_base_ef_list,
        profile_list=profile_list,
        margin_mm=args.margin_mm,
    )

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(f"saved: {args.output}")
    print("plane normal:", result["plane"]["normal_base"])
    print("plane RMS [mm]:", result["plane"]["rms_error_mm"])
    print("safe bounds [uv, mm]:", result["safe_bounds_uv_mm"])


if __name__ == "__main__":
    main()
