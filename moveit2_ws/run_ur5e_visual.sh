#!/usr/bin/env bash
set -eo pipefail
source /opt/ros/humble/setup.bash
cd "$(dirname "$0")"
colcon build --symlink-install --packages-select ur5e_laser_visualization
source install/setup.bash
exec ros2 launch ur5e_laser_visualization display.launch.py
