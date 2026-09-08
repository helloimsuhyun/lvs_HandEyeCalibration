# UR5e + LJ-V7080 visualization preview

This package is intentionally visualization-only. It does not add a UR5e MoveIt planning configuration yet.

The mounting used by the preview is:

- `T_tool0_sensor = diag(-1,+1,-1)` with translation `[0,0,151] mm`
- equivalently `T_flange_sensor`: translation `[151,0,0] mm`, RPY `[+90°,0,-90°]`
- `T_sensor_physical`: translation `[0,0,+80] mm`

Origin marker colors:

- orange: `flange`
- cyan: `tool0`
- magenta: sensor measurement origin `S`
- yellow: sensor physical origin `P`
- axis colors: X red, Y green, Z blue

Run:

```bash
cd ~/lvs_HandEyeCalibration/moveit2_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select ur5e_laser_visualization
source install/setup.bash
ros2 launch ur5e_laser_visualization display.launch.py
```

Expected geometry:

- sensor origin is 151 mm outward from `tool0` along `+Z_tool0` / from `flange` along `+X_flange`
- sensor `+Z` points opposite `+Z_tool0`
- physical origin is 80 mm back from the sensor origin along sensor `+Z`
- therefore P is 71 mm outward from the flange/tool0 origin along `+Z_tool0` (= `+X_flange`)


Axis mapping used in this revision:
- `+X_sensor = -X_tool0`
- `+Y_sensor = +Y_tool0`
- `+Z_sensor = -Z_tool0`

This keeps the sensor Z direction unchanged from the previous preview while flipping X and Y together to preserve a proper right-handed frame.
