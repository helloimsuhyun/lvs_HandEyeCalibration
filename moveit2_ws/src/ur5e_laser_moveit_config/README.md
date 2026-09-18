# UR5e + LJ-V7080 MoveIt configuration

This package is the UR5e counterpart of `rb5_laser_moveit_config`.

Contract:

- planning group: `ur5e_arm`
- planning base: `base_link`
- planning tip: `sensor_physical_origin`
- IK: TRAC-IK, `solve_type: Distance`
- planner: OMPL `RRTConnect`
- sensor attachment in URDF: `flange -> sensor_measurement_frame`
- application JSON convention: `tcp == tool0`, so `T_tcp_sensor == ^tool0 T_sensor`
- sensor collision: padded head + connector + cable primitives

Current transforms:

```text
^tool0 T_sensor rotation = diag(+1,-1,-1)
^tool0 p_sensor = [0,0,151] mm
^sensor p_physical = [0,0,80] mm
```

This rotation includes the sensor's physical 180-degree turn about the final
`wrist_3_joint` axis, which is `+Z_tool0` in the UR5e model.

The equivalent URDF mounting is:

```text
^flange p_sensor = [151,0,0] mm
^flange R_sensor = RPY(-90 deg, 0, +90 deg)
```

## Build

Do ROS/MoveIt builds outside the Conda environment:

```bash
conda deactivate
source /opt/ros/humble/setup.bash
cd ~/lvs_HandEyeCalibration/moveit2_ws
colcon build --symlink-install --packages-select ur5e_laser_moveit_config
source install/setup.bash
```

## Start MoveIt

```bash
ros2 launch ur5e_laser_moveit_config move_group.launch.py
```

Your planner client must use `ur5e_arm` instead of `rb5_arm` when UR5e is selected.
Trajectory execution can remain in the existing robot adapter, exactly as in the RB5 workflow.

## Transform sanity check

```bash
python3 src/ur5e_laser_moveit_config/scripts/check_sensor_transform.py
```
