# UR5e MoveIt extension

`src/ur5e_laser_moveit_config` now mirrors the RB5 planning stack:

- official UR5e URDF/kinematics from `ur_description`
- Keyence visual + padded collision proxy
- flange -> sensor -> physical-origin fixed chain
- `sensor_physical_origin` as planning tip
- TRAC-IK / Distance
- OMPL RRTConnect
- conservative joint velocity/acceleration limits

The existing `ur5e_laser_visualization` package is kept as a coordinate-frame preview.
The new `ur5e_laser_moveit_config` is the actual MoveIt planning model.

Build with Conda deactivated. `ros-humble-ur`, MoveIt 2 and the TRAC-IK plugin must be installed.
