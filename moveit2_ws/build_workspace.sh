#!/usr/bin/env bash
set -eo pipefail

WORKSPACE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "${WORKSPACE_DIR}/.." && pwd)"

source /opt/ros/humble/setup.bash
set -u
python3 "${WORKSPACE_DIR}/src/rb5_laser_moveit_config/scripts/generate_robot_description.py"
cd "${WORKSPACE_DIR}"
colcon build --symlink-install --packages-select \
  rbpodo_description \
  rb5_laser_moveit_config \
  ur5e_laser_moveit_config

echo "Built ${WORKSPACE_DIR}/install/setup.bash"
echo "The generated P/S fixed joints came from ${PROJECT_DIR}/real_laser_handeye/initial_T_tcp_sensor.json"
