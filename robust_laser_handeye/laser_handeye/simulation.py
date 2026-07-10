from __future__ import annotations
import numpy as np
from .data import LaserScan
from .se3 import make_T, inv_T, transform_points


def euler_xyz_deg(rx: float, ry: float, rz: float) -> np.ndarray:
    ax, ay, az = np.radians([rx, ry, rz])
    cx, sx = np.cos(ax), np.sin(ax)
    cy, sy = np.cos(ay), np.sin(ay)
    cz, sz = np.cos(az), np.sin(az)
    Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
    Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
    Rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])
    return Rz @ Ry @ Rx

# board pose로부터 board pose 방정식을 계산
def make_plane_from_pose(R_base_plane: np.ndarray, t_base_plane: np.ndarray) -> tuple[np.ndarray, float]:
    n = R_base_plane[:, 2]
    n = n / np.linalg.norm(n)
    l = float(n @ np.asarray(t_base_plane).reshape(3))
    if l < 0:
        n, l = -n, -l
    return n, l

# gt hand eye와 로봇 FK, 평면 방정식을 통해 실제 획득되는 프로파일을 시뮬레이션
def simulate_profile_on_plane(
    T_base_ef: np.ndarray,
    T_ef_s_true: np.ndarray,
    plane_n: np.ndarray,
    plane_l: float,
    x_values: np.ndarray,
    noise_std: float = 0.0,
    rng: np.random.Generator | None = None,
) -> LaserScan:
    """
    return ps [x, 0, z] in sensor coordinates
    """
    rng = np.random.default_rng() if rng is None else rng
    T_base_s = T_base_ef @ T_ef_s_true
    Rbs = T_base_s[:3, :3]
    tbs = T_base_s[:3, 3]
    n_s = Rbs.T @ plane_n
    c = float(plane_n @ tbs - plane_l)
    if abs(n_s[2]) < 1e-9:
        raise ValueError('sensor scan plane is nearly parallel to target plane; cannot solve z(x)')
    pts = []
    for x in x_values:
        z = -(n_s[0] * x + c) / n_s[2]
        pts.append([x, 0.0, z])
    pts = np.asarray(pts)
    if noise_std > 0:
        pts[:, [0, 2]] += rng.normal(0.0, noise_std, size=(len(pts), 2))
    return LaserScan(T_base_ef=T_base_ef, points_s=pts)


def random_robot_poses(n: int, rng: np.random.Generator | None = None) -> list[np.ndarray]:
    """Generate generic flange poses for simulation/debugging.

    For a real robot, replace this with IK-generated reachable poses.
    """
    rng = np.random.default_rng() if rng is None else rng
    poses = []
    for _ in range(n):
        R = euler_xyz_deg(*rng.uniform([-25, -25, -180], [25, 25, 180]))
        t = rng.uniform([-250, -250, 250], [250, 250, 650])
        poses.append(make_T(R, t))
    return poses

def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """벡터를 단위벡터로 정규화한다."""
    v = np.asarray(v, dtype=float).reshape(3)
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError("cannot normalize near-zero vector")
    return v / n


def sensor_pose_from_target_line(
    plane_R: np.ndarray,
    plane_t: np.ndarray,
    line_p0: np.ndarray,
    line_p1: np.ndarray,
    d_mm: float,
    theta_deg: float,
    beta_deg: float,
) -> np.ndarray:
    """
    평면 위 target line과 scan parameter (d, theta, beta)로 sensor pose T_base_s를 만든다.

    입력:
        plane_R, plane_t:
            calibration plane frame을 robot base frame으로 보내는 pose.
            plane frame에서 z=0이 calibration plane이다.

        line_p0, line_p1:
            plane local 2D 좌표계에서의 line endpoint.
            shape은 각각 (2,)이다.

        d_mm:
            sensor origin에서 target line 기준점까지의 거리.

        theta_deg:
            sensor scan plane 안에서 profile x-axis와 target line 사이의 기울기 성분.

        beta_deg:
            target line을 기준으로 sensor scan plane normal을 기울이는 성분.
            여기서는 beta=90 deg일 때 기본 자세가 되도록 구현한다.

    출력:
        T_base_s:
            robot base frame에서 본 sensor frame pose.
    """
    plane_R = np.asarray(plane_R, dtype=float).reshape(3, 3)
    plane_t = np.asarray(plane_t, dtype=float).reshape(3)

    line_p0 = np.asarray(line_p0, dtype=float).reshape(2)
    line_p1 = np.asarray(line_p1, dtype=float).reshape(2)

    # plane local 2D point [u, v]를 base frame의 3D point [u, v, 0]으로 변환한다.
    q0 = plane_t + plane_R @ np.array([line_p0[0], line_p0[1], 0.0])
    q1 = plane_t + plane_R @ np.array([line_p1[0], line_p1[1], 0.0])

    # target line direction in base frame
    line_dir = _normalize(q1 - q0)

    # calibration plane normal in base frame
    plane_n = _normalize(plane_R[:, 2])

    # line이 plane 위에 있으므로 line_dir과 plane_n은 수직이어야 한다.
    if abs(float(line_dir @ plane_n)) > 1e-6:
        raise ValueError("target line direction is not perpendicular to plane normal")

    # line의 대표점. 여기서는 line midpoint를 sensor가 보는 기준점으로 둔다.
    q_mid = 0.5 * (q0 + q1)

    # 기본 scan plane normal.
    # sensor y-axis가 이 방향이면 sensor의 y=0 scan plane과 calibration plane의 교선이 target line이 된다.
    y0 = _normalize(np.cross(plane_n, line_dir))

    # beta는 target line을 축으로 scan plane normal을 기울이는 값으로 사용한다.
    # beta=90 deg이면 y_axis = y0.
    beta_offset = np.radians(beta_deg - 90.0)
    y_axis = _normalize(
        np.cos(beta_offset) * y0
        + np.sin(beta_offset) * plane_n
    )

    # sensor scan plane 안에서 target line과 수직인 방향.
    # 이 방향은 대략 plane normal 쪽을 향한다.
    v_axis = _normalize(np.cross(line_dir, y_axis))

    # theta는 scan plane 내부에서 x/z basis를 회전시키는 값으로 사용한다.
    theta = np.radians(theta_deg)

    x_axis = _normalize(
        np.cos(theta) * line_dir
        + np.sin(theta) * v_axis
    )

    z_axis = _normalize(
        -np.sin(theta) * line_dir
        + np.cos(theta) * v_axis
    )

    # 오른손 좌표계 보정: x × y = z가 되도록 한다.
    z_axis = _normalize(np.cross(x_axis, y_axis))
    y_axis = _normalize(np.cross(z_axis, x_axis))

    R_base_s = np.column_stack([x_axis, y_axis, z_axis])

    # det(R)=+1인지 확인한다.
    if np.linalg.det(R_base_s) < 0:
        y_axis = -y_axis
        R_base_s = np.column_stack([x_axis, y_axis, z_axis])

    # q_mid가 sensor 좌표계에서 [0, 0, d]에 오도록 sensor origin을 배치한다.
    t_base_s = q_mid - float(d_mm) * z_axis

    return make_T(R_base_s, t_base_s)


def generate_circular_pattern_scans(
    plane_R: np.ndarray,
    plane_t: np.ndarray,
    T_ef_s_true: np.ndarray,
    radius_mm: float,
    x_values: np.ndarray,
    noise_std: float = 0.0,
    rng: np.random.Generator | None = None,
) -> list[LaserScan]:
    """
    논문식 circular pattern 기반 synthetic scan dataset을 생성한다.

    흐름:
        1. circular 9-line pattern 생성
        2. d/theta/beta scan parameter grid 생성
        3. 각 line과 parameter 조합으로 sensor pose T_base_s 생성
        4. T_base_ef = T_base_s @ inv(T_ef_s_true) 계산
        5. simulate_profile_on_plane()으로 LaserScan 생성
    """
    from .patterns import circular_lines, scan_parameter_grid

    rng = np.random.default_rng() if rng is None else rng

    lines = circular_lines(radius_mm, n_lines=9)
    params = scan_parameter_grid()

    plane_n, plane_l = make_plane_from_pose(plane_R, plane_t)

    scans: list[LaserScan] = []

    for line_p0, line_p1 in lines:
        for prm in params:
            T_base_s = sensor_pose_from_target_line(
                plane_R=plane_R,
                plane_t=plane_t,
                line_p0=line_p0,
                line_p1=line_p1,
                d_mm=prm["d_mm"],
                theta_deg=prm["theta_deg"],
                beta_deg=prm["beta_deg"],
            )

            # T_base_s = T_base_ef @ T_ef_s_true
            # 따라서 T_base_ef = T_base_s @ inv(T_ef_s_true)
            T_base_ef = T_base_s @ inv_T(T_ef_s_true)

            try:
                scan = simulate_profile_on_plane(
                    T_base_ef=T_base_ef,
                    T_ef_s_true=T_ef_s_true,
                    plane_n=plane_n,
                    plane_l=plane_l,
                    x_values=x_values,
                    noise_std=noise_std,
                    rng=rng,
                )
                scans.append(scan)
            except ValueError:
                # sensor scan plane이 target plane과 거의 평행하면 해당 pose는 버린다.
                continue

    return scans