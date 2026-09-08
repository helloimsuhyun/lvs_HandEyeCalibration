#!/usr/bin/env bash
set -eo pipefail

if [[ -n "${CONDA_PREFIX:-}" ]]; then
  echo "ERROR: Conda environment is active: ${CONDA_PREFIX}" >&2
  echo "Run 'conda deactivate' before building ROS 2 / MoveIt packages." >&2
  exit 2
fi

ROOT="$(cd "$(dirname "$0")" && pwd)"
source /opt/ros/humble/setup.bash
cd "$ROOT"

colcon build --symlink-install --packages-select ur5e_laser_moveit_config
source install/setup.bash
exec ros2 launch ur5e_laser_moveit_config move_group.launch.py
