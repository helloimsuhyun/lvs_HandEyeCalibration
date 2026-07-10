import numpy as np
from laser_handeye.se3 import make_T, rot_error_deg
from laser_handeye.simulation import euler_xyz_deg, make_plane_from_pose, random_robot_poses, simulate_profile_on_plane
from laser_handeye.calibration import calibrate_with_known_plane, calibrate_single_plane

rng = np.random.default_rng(7)

# Ground-truth hand-eye: T_ef_s, units: mm
T_true = make_T(euler_xyz_deg(20, -15, 35), np.array([80.0, -40.0, 130.0]))

R_bp = euler_xyz_deg(5, 5, 5)
t_bp = np.array([0.0, 0.0, 450.0])
plane_n, plane_l = make_plane_from_pose(R_bp, t_bp) # 캘리브리에션 보드 위치

# Simulated robot flange poses and 2D profiles.
from laser_handeye.simulation import generate_circular_pattern_scans

x_values = np.linspace(-25, 25, 40)

scans = generate_circular_pattern_scans(
    plane_R=R_bp,
    plane_t=t_bp,
    T_ef_s_true=T_true,
    radius_mm=100.0,
    x_values=x_values,
    noise_std=0.0,
    rng=rng,
)

# 1) One-shot sanity check when the plane is known/measured.
T_known = calibrate_with_known_plane(scans, plane_n, plane_l)
print('[known plane] translation error [mm]:', np.linalg.norm(T_known[:3, 3] - T_true[:3, 3]))
print('[known plane] rotation error [deg]:', rot_error_deg(T_known[:3, :3], T_true[:3, :3]))

# 2) Paper-style iterative unknown-plane path. This requires a good initial guess
# and well-designed scan pose variation in real use.
T_init = make_T(euler_xyz_deg(21, -14, 36), np.array([81.0, -39.0, 131.0]))
res = calibrate_single_plane(scans, T_init, max_iter=1000, tol=1e-7)
print('[unknown plane] converged:', res.converged, 'iterations:', res.iterations)
print('[unknown plane] rank last:', res.rank_history[-1], 'cond last:', res.cond_history[-1])
print('[unknown plane] translation error [mm]:', np.linalg.norm(res.T_ef_s[:3, 3] - T_true[:3, 3]))
print('[unknown plane] rotation error [deg]:', rot_error_deg(res.T_ef_s[:3, :3], T_true[:3, :3]))
