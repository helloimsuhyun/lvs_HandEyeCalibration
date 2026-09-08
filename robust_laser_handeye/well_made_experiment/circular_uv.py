"""
cd ~/lvs_HandEyeCalibration

PYTHONPATH=. python robust_laser_handeye/well_made_experiment/circular_uv.py

"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from robust_laser_handeye.laser_handeye.data import PlaneFrame
from robust_laser_handeye.laser_handeye.simulation import (
    sample_random_handeye,
    simulate_scan_from_sensor_pose,
)
from robust_laser_handeye.laser_handeye.pose_design import (
    circular_uv_points,
    plane_uv_to_base_points,
    sensor_pose_from_target_point,
)

from robust_laser_handeye.laser_handeye.initialization import make_initial_guess
from robust_laser_handeye.laser_handeye.scene_generation import make_plane_from_point,plane_basis
from robust_laser_handeye.laser_handeye.calibration import calibrate_single_plane_with_nonlinear

from robust_laser_handeye.well_made_experiment.visualization import (
    plot_single_plane_scans,
)

from robust_laser_handeye.well_made_experiment.jacobian_analysis import (
    analyze_calibration_jacobian,
)

@dataclass(frozen=True)
class ScanPoseParameters:
    scan_id: int

    target_u_mm: float
    target_v_mm: float

    distance_mm: float
    tilt_deg: float
    azimuth_deg: float
    normal_azimuth_sensor_deg: float


## 나중에 config json으로 옮길 부분 ============================================================================

BOARD_CENTER = np.array([0.0, 0.0, 500.0])
PLANE_NORMAL = np.array([0.2, 0.1, 1.0])

T_EF_S_TRUE = np.array([
    [ 0.813797681349, -0.543838142482, -0.204874128703,   50.0],
    [ 0.469846310393,  0.823172944646, -0.318795777597, -100.0],
    [ 0.342020143326,  0.163175911167,  0.925416578398,   80.0],
    [ 0.0,             0.0,             0.0,               1.0],
], dtype=float)

PROFILE_HALF_WIDTH_MM = 25.0
PROFILE_POINT_COUNT = 100

X_VALUES = np.linspace(
    -PROFILE_HALF_WIDTH_MM,
    PROFILE_HALF_WIDTH_MM,
    PROFILE_POINT_COUNT,
)

MAX_INIT_ROT_ERROR_DEG = 30.0
MAX_INIT_TRANS_ERROR_MM = 200.0

# pattern
N = 5
RADIUS_MM = 100.0

POSE_SUPPORTS = (
    (20.0, 60.0),
    (20.0, 120.0),
    (55.0, 60.0),
    (55.0, 120.0),
)

POSE_RATIOS = (
    0.1,
    0.1,
    0.4,
    0.4,
)

SEED = 0


## 나중에 config json으로 옮길 부분 ============================================================================


# make GT Handeye / initial GT handeye -----------

def make_initial_guess_GT(
    T_true: np.ndarray,
    rng: np.random.Generator | None = None,
    max_rotation_error_deg: float = 30.0,
    max_translation_error_mm: float = 200.0,
) -> np.ndarray:

    T_true = np.asarray(
        T_true,
        dtype=float,
    ).reshape(4, 4)

    return make_initial_guess(
        reference_angles_deg=None,
        reference_translation_mm=T_true[:3, 3],
        reference_rotation=T_true[:3, :3],

        rng=rng,
        mode="carlson",

        angle_range_deg=max_rotation_error_deg,
        translation_range_mm=max_translation_error_mm,

        rotation_perturbation="axis_angle",
        translation_perturbation="direction_norm",
    )

# make plane --------------------- 

def make_plane_frame(
    normal: np.ndarray,
    board_center: np.ndarray,
) -> PlaneFrame:

    plane_n, plane_l = make_plane_from_point(
        normal=normal,
        point_on_plane_mm=board_center,
    )

    u, v = plane_basis(plane_n)

    frame = PlaneFrame(
        u=u,
        v=v,
        n=plane_n,
        offset_mm=plane_l,
    )
    return frame

# =======================================================================================
# --------------------------------------  make scan parametor 

# 주어진 tilt와 beta(view azimuth)를 기반으로 자연스럽게 원형패턴을 만들도록
# plane azimuth 파라미터(alpha)를 연산
def alpha_for_radial_scanline(
    u: float,
    v: float,
    tilt_deg: float,
    beta_deg: float,
) -> float:

    # 원하는 실제 scan-line 방향
    psi_deg = (
        np.degrees(
            np.arctan2(v, u)
        )
        % 360.0
    )

    theta = np.deg2rad(
        tilt_deg
    )

    beta = np.deg2rad(
        beta_deg
    )

    delta_deg = np.degrees(
        np.arctan2(
            np.sin(beta),
            np.cos(beta) / np.cos(theta),
        )
    )

    alpha_deg = (
        psi_deg
        - delta_deg
    ) % 360.0

    return float(alpha_deg)

def make_scan_pose_parameters(
    uv: np.ndarray,
    pose_supports: tuple[tuple[float, float], ...],
    pose_ratios: tuple[float, ...],
) -> list[ScanPoseParameters]:

    uv = np.asarray(uv, dtype=float)

    if uv.ndim != 2 or uv.shape[1] != 2:
        raise ValueError("uv must have shape (N, 2)")

    N = len(uv)

    if len(pose_supports) != len(pose_ratios):
        raise ValueError(
            "pose_supports and pose_ratios must have the same length"
        )

    ratios = np.asarray(
        pose_ratios,
        dtype=float,
    )

    if np.any(ratios < 0.0):
        raise ValueError("pose_ratios must be non-negative")

    if ratios.sum() <= 0.0:
        raise ValueError("pose_ratios must have positive sum")

    ratios = ratios / ratios.sum()

    # ---------------------------------------------------------------
    # 1. Continuous weights -> integer scan counts
    # ---------------------------------------------------------------
    raw_counts = N * ratios
    counts = np.floor(raw_counts).astype(int)

    remainder = N - counts.sum()

    # 남은 scan은 fractional part가 큰 support에 우선 배정
    fractions = raw_counts - counts

    order = np.argsort(-fractions)

    for i in order[:remainder]:
        counts[i] += 1


    # ---------------------------------------------------------------
    # 3. Support별 parameter 생성
    # ---------------------------------------------------------------
    params: list[ScanPoseParameters] = []

    assigned = np.zeros(len(counts), dtype=int)
    support_ids = []

    for scan_id in range(N):
        deficit = (scan_id + 1) * counts / N - assigned
        deficit[assigned >= counts] = -np.inf

        support_id = int(np.argmax(deficit))
        support_ids.append(support_id)
        assigned[support_id] += 1

    # Distribute normal azimuth uniformly across every scan that shares the
    # same tilt.  Distance only selects the support label; it does not start a
    # separate azimuth ring.
    tilt_total_counts: dict[float, int] = {}

    for support_id, (tilt_deg, _distance_mm) in enumerate(
        pose_supports
    ):
        tilt_key = float(tilt_deg)
        tilt_total_counts[tilt_key] = (
            tilt_total_counts.get(tilt_key, 0)
            + int(counts[support_id])
        )

    tilt_local_ids = {
        tilt_key: 0
        for tilt_key in tilt_total_counts
    }

    for scan_id, support_id in enumerate(support_ids):

        tilt_deg, distance_mm = pose_supports[support_id]

        tilt_key = float(tilt_deg)
        tilt_count = tilt_total_counts[tilt_key]
        tilt_local_id = tilt_local_ids[tilt_key]

        # -----------------------------------------------------------
        # View azimuth beta
        # -----------------------------------------------------------

        normal_azimuth_deg = (
            360.0 * tilt_local_id / tilt_count
        )

        # -----------------------------------------------------------
        # Ordinary azimuth alpha
        #
        # 실제 laser scan line이 원의 radial direction을
        # 따르도록 theta와 beta를 고려해 alpha 보정
        # -----------------------------------------------------------

        azimuth_deg = alpha_for_radial_scanline(
            u=float(uv[scan_id, 0]),
            v=float(uv[scan_id, 1]),
            tilt_deg=float(tilt_deg),
            beta_deg=float(normal_azimuth_deg),
        )

        params.append(
            ScanPoseParameters(
                scan_id=scan_id,

                target_u_mm=float(uv[scan_id, 0]),
                target_v_mm=float(uv[scan_id, 1]),

                distance_mm=float(distance_mm),
                tilt_deg=float(tilt_deg),

                azimuth_deg=float(
                    azimuth_deg
                ),

                normal_azimuth_sensor_deg=float(
                    normal_azimuth_deg
                ),
            )
        )

        tilt_local_ids[tilt_key] += 1

    return params


def main() -> None:

    rng = np.random.default_rng(SEED)

    # GT / initial handeye
    T_ef_s_true = T_EF_S_TRUE.copy()
    T_init = make_initial_guess_GT(
        T_true=T_ef_s_true,
        rng=rng,
        max_rotation_error_deg=MAX_INIT_ROT_ERROR_DEG,
        max_translation_error_mm=MAX_INIT_TRANS_ERROR_MM,
    )

    # calibration pattern
    frame = make_plane_frame(
        normal=PLANE_NORMAL,
        board_center=BOARD_CENTER,
    )

    # Circular target points

    uv = circular_uv_points(
        radius_mm=RADIUS_MM,
        N=N,
    )
    
    target_points_base = plane_uv_to_base_points(
        uv=uv,
        target_center_base_mm=BOARD_CENTER,
        frame=frame,
    )

    pose_params = make_scan_pose_parameters(
        uv=uv,
        pose_supports=POSE_SUPPORTS,
        pose_ratios=POSE_RATIOS,
    )

    print(
        "\nGT:\n", T_ef_s_true,
        "\n\nInit:\n", T_init,
        "\n\nPlane:",
        "\nu =", frame.u,
        "\nv =", frame.v,
        "\nn =", frame.n,
        "\nl =", frame.offset_mm,
        "\n\nUV:\n", uv,
        "\n\nTarget points:\n", target_points_base,
    )

    print("\nPose parameters:")
    for p in pose_params:
        print(
            f"id={p.scan_id:2d} | "
            f"u={p.target_u_mm:7.2f} "
            f"v={p.target_v_mm:7.2f} | "
            f"d={p.distance_mm:6.1f} | "
            f"tilt={p.tilt_deg:5.1f} | "
            f"az={p.azimuth_deg:6.1f} | "
            f"normal_az={p.normal_azimuth_sensor_deg:6.1f}"
        )

    # -------------------------------------------------

    scans = []
    sensor_poses = []

    for p in pose_params:

        T_base_s = sensor_pose_from_target_point(
            target_point_base_mm=target_points_base[p.scan_id],
            frame=frame,
            distance_mm=p.distance_mm,
            tilt_deg=p.tilt_deg,
            azimuth_deg=p.azimuth_deg,
            normal_azimuth_sensor_deg=p.normal_azimuth_sensor_deg,
        )

        sensor_poses.append(T_base_s)

        scan = simulate_scan_from_sensor_pose(
            T_base_s=T_base_s,
            T_ef_s_true=T_ef_s_true,
            frame=frame,
            x_values=X_VALUES,
            noise_std=0.0,
            rng=rng,
            plane_id=0,
            scan_id=p.scan_id,
        )

        scans.append(scan)

    linear_result, nonlinear_result = calibrate_single_plane_with_nonlinear(
        scans,
        T_init=T_init,
        plane_offset_mode="joint",
        plane_mode="refit",
        max_iter=200,
        nonlinear_max_nfev=200,
    )

    print("\nGT:\n", T_ef_s_true)
    print("\nEstimated:\n", nonlinear_result.T_ef_s)
    print("\nRMS:", nonlinear_result.final_rms_mm)

    plot_single_plane_scans(
        frame=frame,
        board_center=BOARD_CENTER,
        sensor_poses=sensor_poses,
        scans=scans,
    )

    gt_metrics = analyze_calibration_jacobian(
        sensor_poses=sensor_poses,
        frame=frame,
        T_ef_s_true=T_ef_s_true,
        T_eval=T_ef_s_true,
        x_values=X_VALUES,
        rotation_characteristic_length_mm=100.0,
        label="GT",
    )
    
if __name__ == "__main__":
    main()
