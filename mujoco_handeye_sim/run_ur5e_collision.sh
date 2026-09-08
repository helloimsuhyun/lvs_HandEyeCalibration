#!/usr/bin/env bash
set -eo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
source /opt/ros/humble/setup.bash

python3 "$ROOT/scripts/prepare_ur5e.py"
PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" python3 -m handeye_mujoco build \
  --config "$ROOT/configs/ur5e_ljv7080.yaml"
PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" python3 -m handeye_mujoco view \
  --collision-only \
  --model "$ROOT/build/ur5e_ljv7080/model.xml"
