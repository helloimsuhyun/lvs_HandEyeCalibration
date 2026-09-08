# RB5 + UR5e ROS2 RobotAdapter bundle

The integrated GUI starts the selected official ROS 2 driver from the supplied
robot IP, verifies its interfaces, and stops the owned driver process on exit.
It connects to:

- `/joint_states`
- TF (`base -> tcp`)
- `/joint_trajectory_controller/follow_joint_trajectory`

## Integrated files

Copy these files into your existing `real_laser_handeye/` package:

```text
real_laser_handeye/
├── ros2_robot_adapter_base.py
├── robot_adapter_rb5_ros.py
├── robot_adapter_ur5e_ros.py
└── ros2_driver_launcher.py
```

Driver launch settings are in the matching real workflow YAML files.

---

# 1. RB5-850E

## Official repositories

```bash
sudo apt update
sudo apt install -y \
  build-essential cmake git \
  ros-humble-ament-cmake \
  ros-humble-joint-state-publisher \
  ros-humble-moveit \
  ros-humble-pluginlib \
  ros-humble-robot-state-publisher \
  ros-humble-ros2-controllers \
  ros-humble-ros2-control \
  ros-humble-rviz2 \
  ros-humble-urdf-launch \
  ros-humble-xacro

cd ~
git clone https://github.com/RainbowRobotics/rbpodo.git
mkdir -p rbpodo/build
cd rbpodo/build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
sudo make install

mkdir -p ~/rbpodo_ros2_ws/src
cd ~/rbpodo_ros2_ws
git clone https://github.com/RainbowRobotics/rbpodo_ros2.git src/rbpodo_ros2

source /opt/ros/humble/setup.bash
rosdep install --from-paths src --ignore-src -r -y
colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
```

### Official RB5 launch reference

Official repository example:

```bash
source /opt/ros/humble/setup.bash
source ~/rbpodo_ros2_ws/install/setup.bash

ros2 launch rbpodo_moveit_config moveit.launch.py \
  model_id:=rb5_850e \
  use_fake_hardware:=false \
  cb_simulation:=false \
  robot_ip:=169.254.186.20
```

This is a reference check only. During normal GUI use, do not run it in
parallel: the GUI launches `rb5_laser_moveit_config/real_driver.launch.py`,
which uses the same official hardware and activates the trajectory controller
without starting a duplicate MoveGroup.

Check:

```bash
ros2 topic echo /joint_states --once
ros2 action list | grep follow_joint_trajectory
ros2 run tf2_ros tf2_echo link0 tcp
```

Expected action:

```text
/joint_trajectory_controller/follow_joint_trajectory
```

## Hand-eye config change

```yaml
equipment:
  robot_adapter: real_laser_handeye.robot_adapter_rb5_ros:RobotAdapter

  joint_names:
    - base
    - shoulder
    - elbow
    - wrist1
    - wrist2
    - wrist3
```

Your MoveIt section should match the RB5 MoveIt model.  If your existing custom
`rb5_laser_moveit_config` already contains the Keyence collision geometry,
keep using that package instead of replacing it.

---

# 2. UR5e

## Recommended install

Universal Robots recommends binary installation unless you specifically need
to modify the driver:

```bash
source /opt/ros/humble/setup.bash
sudo apt update
sudo apt install ros-humble-ur
```

If you explicitly want the GitHub source instead:

```bash
mkdir -p ~/ur_ros2_ws/src
cd ~/ur_ros2_ws

git clone -b humble \
  https://github.com/UniversalRobots/Universal_Robots_ROS2_Driver.git \
  src/Universal_Robots_ROS2_Driver

vcs import src --skip-existing \
  --input src/Universal_Robots_ROS2_Driver/Universal_Robots_ROS2_Driver-not-released.humble.repos

rosdep update
rosdep install --ignore-src --from-paths src -y
colcon build --cmake-args -DCMAKE_BUILD_TYPE=Release
source install/setup.bash
```

## Real UR5e: one-time robot setup

Install the External Control URCap and create an External Control program on the
teach pendant.

The GUI extracts the factory calibration automatically on the first connection.
The equivalent official command is shown only for troubleshooting:

```bash
ros2 launch ur_calibration calibration_correction.launch.py \
  robot_ip:=<UR_IP> \
  target_filename:=/tmp/ur5e_calibration.yaml
```

## Official UR5e launch reference

```bash
source /opt/ros/humble/setup.bash
# source ~/ur_ros2_ws/install/setup.bash   # only if built from source

ros2 launch ur_robot_driver ur_control.launch.py \
  ur_type:=ur5e \
  robot_ip:=<UR_IP> \
  kinematics_params_file:=/tmp/ur5e_calibration.yaml \
  initial_joint_controller:=joint_trajectory_controller
```

Do not run this beside the GUI. The GUI launches the scaled trajectory
controller and passes its cached calibration file automatically.

The following official MoveIt command is also a standalone reference; this
project launches its sensor-aware MoveGroup itself:

```bash
source /opt/ros/humble/setup.bash
ros2 launch ur_moveit_config ur_moveit.launch.py \
  ur_type:=ur5e \
  launch_rviz:=true
```

Check:

```bash
ros2 topic echo /joint_states --once
ros2 action list | grep follow_joint_trajectory
ros2 run tf2_ros tf2_echo base tool0
```

Expected action:

```text
/scaled_joint_trajectory_controller/follow_joint_trajectory
# fallback: /joint_trajectory_controller/follow_joint_trajectory
```

## Hand-eye config change

```yaml
equipment:
  robot_adapter: real_laser_handeye.robot_adapter_ur5e_ros:RobotAdapter

  joint_names:
    - shoulder_pan_joint
    - shoulder_lift_joint
    - elbow_joint
    - wrist_1_joint
    - wrist_2_joint
    - wrist_3_joint
```

---

# 3. Important: Keyence collision model

The official UR repository knows the UR5e robot, but it does not know your
Keyence LJ-V7080 mount.

For collision-aware planning you still need your project-specific end-effector
description:

```text
UR5e tool0
    |
    +-- Keyence mount
    +-- Keyence collision geometry
    +-- sensor_physical_origin
```

Keep `T_tcp_sensor` consistent with the TCP frame chosen by the adapter.
This bundle uses:

- RB5: `link0 -> tcp`
- UR5e: `base_link -> tool0`

If you instead define a dedicated robot TCP frame in your URDF, change
`TCP_FRAME` in `robot_adapter_ur5e_ros.py` to that frame name.

---

# 4. Run your workflow

Source ROS and the correct driver workspace, then run the GUI.  REAL mode now
starts and owns the selected robot driver automatically; do not launch a second
driver manually.

```bash
cd ~/lvs_HandEyeCalibration
conda activate laser_handeye

source /opt/ros/humble/setup.bash

# RB5:
source ~/rbpodo_ros2_ws/install/setup.bash

# OR UR5e source build:
# source ~/ur_ros2_ws/install/setup.bash

export PYTHONPATH="$PWD:$PWD/mujoco_handeye_sim/src${PYTHONPATH:+:$PYTHONPATH}"

python -m real_laser_handeye.workflow_gui \
  --mode real \
  --robot rb5 \
  --robot-ip <ROBOT_IP> \
  --laser-ip <LASER_IP>
```

For UR5e choose `--robot ur5e`.  On its first connection the official
`ur_calibration` extractor writes a factory-kinematics YAML under the real
session's `ros_driver` directory, then that file is passed to
`ur_control.launch.py` and the project's sensor-aware MoveGroup. Start the
teach-pendant External Control program before enabling motion.

After `Connect`, real movement remains interlocked.  Enabling **Real motion**
sends one low-speed, zero-displacement hold trajectory and checks the action,
measured joints and TCP before scan motion is allowed.

## Trajectory execution

MoveIt returns a time-parameterized joint trajectory (positions plus
`time_from_start`, and velocity/acceleration arrays when available). The
workflow preserves that trajectory and sends the complete path as **one**
`FollowJointTrajectory` action goal. The ROS trajectory controller therefore
owns interpolation/tracking between waypoints instead of stopping at every
MoveIt waypoint.

`move_j()` remains only for the zero-displacement motion verification and
legacy fallback paths. Normal MoveIt scan/return execution uses
`execute_joint_trajectory()`. Return-to-safe is independently planned and
time-parameterized by MoveIt; it is not created by reversing the forward
trajectory array.
