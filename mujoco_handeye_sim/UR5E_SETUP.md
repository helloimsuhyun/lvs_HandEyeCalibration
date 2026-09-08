# UR5e + LJ-V7080 MuJoCo setup

The UR5e path now has the same simulation ingredients as the RB5 path:

- official UR5e collision meshes from `ur_description`
- six actuated joints and home pose
- Keyence CAD visual
- padded sensor head / connector / cable collision proxy
- `tool0 -> sensor` hand-eye transform
- `sensor -> physical_origin` transform
- flange/tool0/sensor/physical diagnostic frame triads
- self/environment collision reporting
- trajectory collision checking through `HandEyeSimulation`

Current hand-eye JSON:

```text
^tool0 R_sensor = diag(-1,+1,-1)
^tool0 p_sensor = [0,0,151] mm
^sensor p_physical = [0,0,80] mm
```

Thus the physical origin is 71 mm along `+Z_tool0` from tool0.

## Run

The MuJoCo Python environment may remain active. ROS is sourced only so the
installed official UR description can be flattened and its mesh paths resolved.

```bash
cd ~/lvs_HandEyeCalibration/mujoco_handeye_sim
./run_ur5e.sh
```

Collision proxy view:

```bash
./run_ur5e_collision.sh
```

Manual build:

```bash
source /opt/ros/humble/setup.bash
python3 scripts/prepare_ur5e.py
PYTHONPATH=src python3 -m handeye_mujoco build --config configs/ur5e_ljv7080.yaml
PYTHONPATH=src python3 -m handeye_mujoco check --model build/ur5e_ljv7080/model.xml
```
