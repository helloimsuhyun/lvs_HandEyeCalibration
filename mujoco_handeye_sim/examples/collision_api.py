"""Minimal integration example for a calibration trajectory planner."""

from pathlib import Path

import numpy as np

from handeye_mujoco import HandEyeSimulation


ROOT = Path(__file__).resolve().parents[1]
simulation = HandEyeSimulation(ROOT / "build/rb5_ljv7080/model.xml")
simulation.reset_home()

print("joint order:", simulation.joint_names)
print("T_world_sensor:\n", simulation.sensor_pose_world())

# Replace this interpolation with IK-generated calibration waypoints.
q_start = simulation.data.qpos.copy()
q_goal = q_start + np.deg2rad([10, -5, 5, 0, 5, 10])
trajectory = np.linspace(q_start, q_goal, 101)
safe, bad_index, report = simulation.trajectory_is_collision_free(trajectory)
print("safe:", safe, "first unsafe sample:", bad_index)
if report:
    print(report)
