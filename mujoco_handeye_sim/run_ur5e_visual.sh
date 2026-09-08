#!/usr/bin/env bash
set -euo pipefail
source /opt/ros/humble/setup.bash
cd "$(dirname "$0")"
python3 scripts/prepare_ur5e_visual.py
PYTHONPATH=src python3 -m handeye_mujoco build --config configs/ur5e_ljv7080_visual.yaml
exec env PYTHONPATH=src python3 -m handeye_mujoco view --model build/ur5e_ljv7080_visual/model.xml
