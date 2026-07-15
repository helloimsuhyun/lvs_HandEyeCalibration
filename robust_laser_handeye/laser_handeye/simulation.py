from __future__ import annotations
from typing import Literal
import numpy as np
from .data import LaserScan
from .se3 import make_T, inv_T, euler_xyz_deg


PoseGeometry = Literal["paper_incidence", "observable_dihedral"]


# board pose로부터 board pose 방정식을 계산
def make_plane_from_pose(R_base_plane: np.ndarray, t_base_plane: np.ndarray) -> tuple[np.ndarray, float]:
    n = R_base_plane[:, 2]
    n = n / np.linalg.norm(n)
    l = float(n @ np.asarray(t_base_plane).reshape(3))
    if l < 0:
        n, l = -n, -l
    return n, l

# 보드 위에 맺히는 레이저 프로파일을 시뮬레이션
def simulate_profile_on_plane(
    T_base_ef: np.ndarray,
    T_ef_s_true: np.ndarray,
    plane_n: np.ndarray,
    plane_l: float,
    x_values: np.ndarray,
    noise_std: float = 0.0,
    rng: np.random.Generator | None = None,
    plane_id: int = 0,
    scan_id: int | None = None,
    meta: dict | None = None,
) -> LaserScan:
    """
    The 2D laser profile is represented in the sensor frame as points [x, 0, z].
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
    meta_out = dict(meta or {})
    if noise_std > 0:
        noise_xz = rng.normal(0.0, noise_std, size=(len(pts), 2))
        pts[:, 0] += noise_xz[:, 0]
        pts[:, 2] += noise_xz[:, 1]

        meta_out["noise_x_mm"] = noise_xz[:, 0].copy()
        meta_out["noise_z_mm"] = noise_xz[:, 1].copy()
        meta_out["noise_std_command_mm"] = float(noise_std)

    return LaserScan(
        T_base_ef=T_base_ef,
        points_s=pts,
        plane_id=plane_id,
        scan_id=scan_id,
        meta=meta_out,
    )


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


# 벡터를 단위벡터로 정규화
def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """벡터를 단위벡터로 정규화한다."""
    v = np.asarray(v, dtype=float).reshape(3)
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError("cannot normalize near-zero vector")
    return v / n

# random한 GT handeye 생성
def sample_random_handeye(
    rng: np.random.Generator | None = None,
    trans_range_mm: tuple[float, float] = (-100.0, 200.0),
    angle_range_deg: tuple[float, float] = (-180.0, 180.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample a paper-style random GT hand-eye transform.

    Returns:
        T_ef_s_true, euler_xyz_angles_deg, translation_mm
    """
    rng = np.random.default_rng() if rng is None else rng
    angles_deg = rng.uniform(angle_range_deg[0], angle_range_deg[1], size=3)
    t_mm = rng.uniform(trans_range_mm[0], trans_range_mm[1], size=3)
    return make_T(euler_xyz_deg(*angles_deg), t_mm), angles_deg, t_mm

# random한 캘리브레이션 borad를 생성
def sample_random_plane_pose(
    rng: np.random.Generator | None = None,
    angle_range_deg: tuple[float, float] = (-30.0, 30.0),
    min_abs_angle_deg: float = 1.0,
    xy_range_mm: tuple[float, float] = (-100.0, 100.0),
    z_range_mm: tuple[float, float] = (350.0, 600.0),
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, np.ndarray]:
    """Sample a single-plane calibration artefact pose in the robot base frame.
    """
    rng = np.random.default_rng() if rng is None else rng
    for _ in range(10_000):
        angles_deg = rng.uniform(angle_range_deg[0], angle_range_deg[1], size=3)
        if np.all(np.abs(angles_deg) > min_abs_angle_deg):
            break
    else:
        raise RuntimeError("failed to sample plane angles satisfying min_abs_angle_deg")

    R_base_plane = euler_xyz_deg(*angles_deg)
    t_base_plane = np.array([
        rng.uniform(xy_range_mm[0], xy_range_mm[1]),
        rng.uniform(xy_range_mm[0], xy_range_mm[1]),
        rng.uniform(z_range_mm[0], z_range_mm[1]),
    ])
    plane_n, plane_l = make_plane_from_pose(R_base_plane, t_base_plane)
    return R_base_plane, t_base_plane, plane_n, plane_l, angles_deg

# GT기반 초기 hand eye 행렬 생성
def make_initial_guess_from_true(
    true_angles_deg: np.ndarray,
    true_translation_mm: np.ndarray,
    rng: np.random.Generator | None = None,
    rel_offset: float = 0.10,
    min_angle_offset_deg: float = 0.0,
    min_translation_offset_mm: float = 0.0,
) -> np.ndarray:
    """Generate the paper-style initial guess by adding uniform ±10% offsets.

    Optional minimum additive offsets are useful when a nominal parameter is close
    to zero and a purely relative perturbation becomes unrealistically tiny.
    """
    rng = np.random.default_rng() if rng is None else rng
    true_angles_deg = np.asarray(true_angles_deg, dtype=float).reshape(3)
    true_translation_mm = np.asarray(true_translation_mm, dtype=float).reshape(3)

    angle_delta = true_angles_deg * rng.uniform(-rel_offset, rel_offset, size=3)
    trans_delta = true_translation_mm * rng.uniform(-rel_offset, rel_offset, size=3)

    if min_angle_offset_deg > 0:
        angle_delta += rng.uniform(-min_angle_offset_deg, min_angle_offset_deg, size=3)
    if min_translation_offset_mm > 0:
        trans_delta += rng.uniform(-min_translation_offset_mm, min_translation_offset_mm, size=3)

    init_angles = true_angles_deg + angle_delta
    init_translation = true_translation_mm + trans_delta
    return make_T(euler_xyz_deg(*init_angles), init_translation)


def is_reachable_simple(
    T_base_ef: np.ndarray,
    xyz_min_mm: tuple[float, float, float] = (-700.0, -700.0, 50.0),
    xyz_max_mm: tuple[float, float, float] = (700.0, 700.0, 900.0),
) -> bool:
    """Simple workspace-box reachability approximation.

    This is only a stand-in for robot IK + joint-limit checking.
    """
    p = np.asarray(T_base_ef, dtype=float).reshape(4, 4)[:3, 3]
    return bool(np.all(p >= np.asarray(xyz_min_mm)) and np.all(p <= np.asarray(xyz_max_mm)))

# target line과 d, theta, beta를 이용해
# 논문 Fig. 2 정의에 맞는 sensor pose T_base_s 생성
def _sensor_pose_from_target_line_dihedral(
    plane_R: np.ndarray,
    plane_t: np.ndarray,
    line_p0: np.ndarray,
    line_p1: np.ndarray,
    d_mm: float,
    theta_deg: float,
    beta_deg: float,
    branch_sign: float = 1.0,
) -> np.ndarray:
    """Generate the legacy/observable dihedral-angle pose convention.

    This simulator convention uses ``theta`` as
    the dihedral angle between the laser scan plane and calibration plane, and
    ``beta`` as the in-scan-plane angle from the target line to ``Z_s``.  The
    convention is deterministic and its realized angles are checked below; it
    should not be described as the unique pose construction from the paper.

    Geometry:
        L = target-line unit direction
        n = calibration-plane unit normal
        s = n x L

    theta is the projection angle of the laser scan plane relative to the
    calibration plane.  Equivalently, the sensor scan-plane normal is

        Y_s = cos(theta) s + branch_sign sin(theta) n.

    Let u be the direction in the sensor X-Z scan plane that is perpendicular
    to L and points toward the plane normal.  beta is the angle from L to Z_s:

        Z_s = cos(beta) L + sin(beta) u.

    The target-line midpoint is placed at [0, 0, d] in the sensor frame.
    """
    plane_R = np.asarray(plane_R, dtype=float).reshape(3, 3)
    plane_t = np.asarray(plane_t, dtype=float).reshape(3)
    line_p0 = np.asarray(line_p0, dtype=float).reshape(2)
    line_p1 = np.asarray(line_p1, dtype=float).reshape(2)

    if d_mm <= 0.0:
        raise ValueError("d_mm must be positive")
    if branch_sign not in (-1.0, 1.0):
        raise ValueError("branch_sign must be +1 or -1")

    q0 = plane_t + plane_R @ np.array([line_p0[0], line_p0[1], 0.0])
    q1 = plane_t + plane_R @ np.array([line_p1[0], line_p1[1], 0.0])
    q_mid = 0.5 * (q0 + q1)

    line_dir = _normalize(q1 - q0)
    plane_n = _normalize(plane_R[:, 2])
    if abs(float(line_dir @ plane_n)) > 1e-6:
        raise ValueError("target line is not contained in the plane")

    side_dir = _normalize(np.cross(plane_n, line_dir))
    theta = float(np.radians(theta_deg))
    beta = float(np.radians(beta_deg))

    # Sensor Y is the normal of the laser X-Z scan plane.
    y_axis = _normalize(
        np.cos(theta) * side_dir
        + float(branch_sign) * np.sin(theta) * plane_n
    )

    # In-scan-plane direction perpendicular to the target line.
    u_axis = plane_n - float(plane_n @ y_axis) * y_axis
    u_axis = u_axis - float(u_axis @ line_dir) * line_dir
    u_axis = _normalize(u_axis)

    z_axis = _normalize(
        np.cos(beta) * line_dir
        + np.sin(beta) * u_axis
    )
    x_axis = _normalize(np.cross(y_axis, z_axis))

    # Re-orthogonalize to suppress floating-point drift.
    z_axis = _normalize(np.cross(x_axis, y_axis))
    y_axis = _normalize(np.cross(z_axis, x_axis))

    R_base_s = np.column_stack([x_axis, y_axis, z_axis])
    if np.linalg.det(R_base_s) < 0.0:
        x_axis = -x_axis
        R_base_s = np.column_stack([x_axis, y_axis, z_axis])

    # Sensor +Z points from the sensor origin to the target-line midpoint.
    t_base_s = q_mid - float(d_mm) * z_axis

    # Geometry checks.
    realized_theta = np.degrees(
        np.arcsin(np.clip(abs(float(plane_n @ y_axis)), 0.0, 1.0))
    )
    realized_beta = np.degrees(
        np.arccos(np.clip(float(line_dir @ z_axis), -1.0, 1.0))
    )
    if not np.isclose(realized_theta, abs(theta_deg), atol=1e-6):
        raise RuntimeError(
            f"theta construction failed: command={theta_deg}, "
            f"actual={realized_theta}"
        )
    if not np.isclose(realized_beta, beta_deg, atol=1e-6):
        raise RuntimeError(
            f"beta construction failed: command={beta_deg}, "
            f"actual={realized_beta}"
        )
    if abs(float(y_axis @ line_dir)) > 1e-8:
        raise RuntimeError("target line is not in the sensor X-Z scan plane")

    return make_T(R_base_s, t_base_s)


def _sensor_pose_from_target_line_paper_incidence(
    plane_R: np.ndarray,
    plane_t: np.ndarray,
    line_p0: np.ndarray,
    line_p1: np.ndarray,
    d_mm: float,
    theta_deg: float,
    beta_deg: float,
    branch_sign: float,
) -> np.ndarray:
    """Construct the incidence-angle geometry evidenced by the paper data.

    Let ``L`` be the directed target line and ``n`` the plane normal.  In the
    sensor frame the target line is in the X-Z laser plane and the published
    angles impose

        L_s = [sin(beta), 0, cos(beta)]
        n_s = [cos(theta) cot(beta), mirror_term, -cos(theta)].

    The mirror term follows from unit length and has either sign.  This gives
    ``angle(L_s, +Z_s)=beta`` and ``angle(n, -Z_s)=theta``.  It also reproduces
    the four normal directions in the deposited theta=30, beta=60/90/120
    trajectory.  The paper does not print this vector equation, but Fig. 2,
    its theta=0/beta=90 statement, and the public poses all agree with it.
    """
    plane_R = np.asarray(plane_R, dtype=float).reshape(3, 3)
    plane_t = np.asarray(plane_t, dtype=float).reshape(3)
    line_p0 = np.asarray(line_p0, dtype=float).reshape(2)
    line_p1 = np.asarray(line_p1, dtype=float).reshape(2)
    if d_mm <= 0.0:
        raise ValueError("d_mm must be positive")
    if branch_sign not in (-1.0, 1.0):
        raise ValueError("branch_sign must be +1 or -1")

    theta = float(np.radians(theta_deg))
    beta = float(np.radians(beta_deg))
    if not (0.0 <= theta < 0.5 * np.pi):
        raise ValueError("paper-incidence theta must be in [0, 90) degrees")
    sin_beta = float(np.sin(beta))
    if sin_beta <= 1e-12:
        raise ValueError("paper-incidence beta must be in (0, 180) degrees")

    cos_theta = float(np.cos(theta))
    mirror_squared = 1.0 - (cos_theta / sin_beta) ** 2
    if mirror_squared < -1e-10:
        raise ValueError(
            "infeasible paper-incidence angles: require "
            "sin(beta) >= cos(theta)"
        )
    mirror_component = float(np.sqrt(max(0.0, mirror_squared)))

    line_s = np.array(
        [sin_beta, 0.0, float(np.cos(beta))],
        dtype=float,
    )
    normal_s = np.array(
        [
            cos_theta / float(np.tan(beta)),
            float(branch_sign) * mirror_component,
            -cos_theta,
        ],
        dtype=float,
    )
    line_s = _normalize(line_s)
    normal_s = _normalize(normal_s)
    cross_s = _normalize(np.cross(line_s, normal_s))

    q0 = plane_t + plane_R @ np.array([line_p0[0], line_p0[1], 0.0])
    q1 = plane_t + plane_R @ np.array([line_p1[0], line_p1[1], 0.0])
    q_mid = 0.5 * (q0 + q1)
    line_base = _normalize(q1 - q0)
    normal_base = _normalize(plane_R[:, 2])
    cross_base = _normalize(np.cross(line_base, normal_base))

    basis_base = np.column_stack([line_base, normal_base, cross_base])
    basis_sensor = np.column_stack([line_s, normal_s, cross_s])
    R_base_s = basis_base @ basis_sensor.T
    if not np.isclose(np.linalg.det(R_base_s), 1.0, atol=1e-9):
        raise RuntimeError("paper-incidence pose construction is not right-handed")

    # q_mid is [0, 0, d] in sensor coordinates.
    t_base_s = q_mid - float(d_mm) * R_base_s[:, 2]
    return make_T(R_base_s, t_base_s)


def sensor_pose_from_target_line(
    plane_R: np.ndarray,
    plane_t: np.ndarray,
    line_p0: np.ndarray,
    line_p1: np.ndarray,
    d_mm: float,
    theta_deg: float,
    beta_deg: float,
    branch_sign: float = 1.0,
    pose_geometry: PoseGeometry = "observable_dihedral",
) -> np.ndarray:
    """Convert one target line and scan-parameter triple into a sensor pose.

    ``paper_incidence`` follows the incidence geometry reconstructed from the
    publication and deposited trajectory. ``observable_dihedral`` retains the
    repository's earlier angle convention as an explicit engineering extension.
    """
    if pose_geometry == "paper_incidence":
        return _sensor_pose_from_target_line_paper_incidence(
            plane_R,
            plane_t,
            line_p0,
            line_p1,
            d_mm,
            theta_deg,
            beta_deg,
            branch_sign,
        )
    if pose_geometry == "observable_dihedral":
        return _sensor_pose_from_target_line_dihedral(
            plane_R,
            plane_t,
            line_p0,
            line_p1,
            d_mm,
            theta_deg,
            beta_deg,
            branch_sign,
        )
    raise ValueError(
        "pose_geometry must be 'paper_incidence' or 'observable_dihedral'"
    )

# 원형 패턴으로 스캔되는 프로파일을 simulation
def generate_circular_pattern_scans(
    plane_R: np.ndarray,
    plane_t: np.ndarray,
    T_ef_s_true: np.ndarray,
    radius_mm: float,
    x_values: np.ndarray,
    noise_std: float = 0.0,
    rng: np.random.Generator | None = None,
    scan_params: list[dict] | None = None,
    check_reachability: bool = False,
    plane_id: int = 0,
    projection_branch_mode: Literal["alternating", "positive"] = "alternating",
    pose_geometry: PoseGeometry = "observable_dihedral",
) -> list[LaserScan]:
    """Generate a paper-style circular-pattern synthetic scan dataset.

    The circular pattern has nine target lines, each 40 deg apart. For each line,
    a set of scan parameters (d, theta, beta) is converted into a sensor pose;
    the corresponding robot flange pose is then obtained from the GT hand-eye.
    """
    from .patterns import circular_lines, scan_parameter_grid

    rng = np.random.default_rng() if rng is None else rng
    if projection_branch_mode not in ("alternating", "positive"):
        raise ValueError(
            "projection_branch_mode must be 'alternating' or 'positive'"
        )
    lines = circular_lines(radius_mm, n_lines=9)
    params = scan_parameter_grid() if scan_params is None else scan_params
    plane_n, plane_l = make_plane_from_pose(plane_R, plane_t)

    scans: list[LaserScan] = []
    for line_id, (line_p0, line_p1) in enumerate(lines):
        # theta is unsigned in the paper and therefore has two mirror pose
        # solutions.  The publication never states how the branches are
        # allocated, so adjacent target lines alternate them. This repairs the
        # one-sided ambiguity in the legacy dihedral convention. In the
        # paper-incidence convention, however, all fixed-|theta| normals keep
        # the same optical-Z component; both branches still leave translation
        # along that direction coupled to the unknown plane offset.
        branch_sign = (
            1.0
            if projection_branch_mode == "positive" or line_id % 2 == 0
            else -1.0
        )

        for parameter_id, prm in enumerate(params):
            try:
                T_base_s = sensor_pose_from_target_line(
                    plane_R=plane_R,
                    plane_t=plane_t,
                    line_p0=line_p0,
                    line_p1=line_p1,
                    d_mm=prm["d_mm"],
                    theta_deg=prm["theta_deg"],
                    beta_deg=prm["beta_deg"],
                    branch_sign=branch_sign,
                    pose_geometry=pose_geometry,
                )
                T_base_ef = T_base_s @ inv_T(T_ef_s_true)
                if check_reachability and not is_reachable_simple(T_base_ef):
                    continue
                scan = simulate_profile_on_plane(
                    T_base_ef=T_base_ef,
                    T_ef_s_true=T_ef_s_true,
                    plane_n=plane_n,
                    plane_l=plane_l,
                    x_values=x_values,
                    noise_std=noise_std,
                    rng=rng,
                    plane_id=plane_id,
                    scan_id=len(scans),
                    meta={
                        "line_id": line_id,
                        "parameter_id": parameter_id,
                        "theta_branch_sign": branch_sign,
                        "signed_theta_deg": branch_sign * prm["theta_deg"],
                        "pose_geometry": pose_geometry,
                        **prm,
                    },
                )
                scans.append(scan)
            except ValueError:
                continue
    return scans


def generate_circular_reference_scans(
    plane_R: np.ndarray,
    plane_t: np.ndarray,
    T_ef_s_true: np.ndarray,
    radius_mm: float,
    x_values: np.ndarray,
    scan_params: list[dict],
    n_scans: int,
    noise_std: float = 0.0,
    rng: np.random.Generator | None = None,
    check_reachability: bool = False,
    plane_id: int = 0,
    pose_geometry: PoseGeometry = "paper_incidence",
) -> list[LaserScan]:
    """Generate an explicit off-ring/reference subset around the circle.

    A fixed incidence magnitude leaves absolute translation coupled to the
    unknown plane offset.  This helper distributes a requested number of poses
    from a second parameter list over the same nine target lines.  It is kept
    separate from the reported 81-pose grid so benchmark output cannot silently
    mistake the added observability constraint for a paper-provided parameter.
    """
    from .patterns import circular_lines

    if n_scans < 0:
        raise ValueError("n_scans must be non-negative")
    if n_scans > 0 and not scan_params:
        raise ValueError("scan_params must be non-empty when n_scans > 0")
    rng = np.random.default_rng() if rng is None else rng
    lines = circular_lines(radius_mm, n_lines=9)
    plane_n, plane_l = make_plane_from_pose(plane_R, plane_t)
    scans: list[LaserScan] = []

    for reference_id in range(int(n_scans)):
        line_id = reference_id % len(lines)
        cycle = reference_id // len(lines)
        # A coprime stride avoids tying one parameter to one circular line.
        parameter_id = (4 * line_id + cycle) % len(scan_params)
        line_p0, line_p1 = lines[line_id]
        prm = scan_params[parameter_id]
        branch_sign = 1.0 if reference_id % 2 == 0 else -1.0
        try:
            T_base_s = sensor_pose_from_target_line(
                plane_R=plane_R,
                plane_t=plane_t,
                line_p0=line_p0,
                line_p1=line_p1,
                d_mm=prm["d_mm"],
                theta_deg=prm["theta_deg"],
                beta_deg=prm["beta_deg"],
                branch_sign=branch_sign,
                pose_geometry=pose_geometry,
            )
            T_base_ef = T_base_s @ inv_T(T_ef_s_true)
            if check_reachability and not is_reachable_simple(T_base_ef):
                continue
            scans.append(
                simulate_profile_on_plane(
                    T_base_ef=T_base_ef,
                    T_ef_s_true=T_ef_s_true,
                    plane_n=plane_n,
                    plane_l=plane_l,
                    x_values=x_values,
                    noise_std=noise_std,
                    rng=rng,
                    plane_id=plane_id,
                    scan_id=reference_id,
                    meta={
                        "line_id": line_id,
                        "reference_pose": True,
                        "reference_id": reference_id,
                        "reference_parameter_id": parameter_id,
                        "theta_branch_sign": branch_sign,
                        "pose_geometry": pose_geometry,
                        **prm,
                    },
                )
            )
        except ValueError:
            continue

    return scans
