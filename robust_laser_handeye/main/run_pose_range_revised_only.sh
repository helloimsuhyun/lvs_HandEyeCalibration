#!/usr/bin/env bash
set -euo pipefail

# Revised pose-range experiment only.
#
# Pose ranges:
#   Restricted: Tilt 25-35 deg, Azimuth -30-30 deg, Roll -15-15 deg
#   Moderate:   Tilt 10-50 deg, Azimuth -135-135 deg, Roll -60-60 deg
#   Wide:       Tilt 5-65 deg, Azimuth -180-180 deg, Roll -90-90 deg
#
# Quick test:
#   MAX_TRIALS=10 bash main/run_pose_range_revised_only.sh
#
# Final run:
#   bash main/run_pose_range_revised_only.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

BASE_SCRIPT="main/run_sliced_lhs_5level_experiments.sh"

if [[ ! -f "${BASE_SCRIPT}" ]]; then
  echo "ERROR: base script not found: ${BASE_SCRIPT}" >&2
  echo "Place this script in the repository's main directory." >&2
  exit 1
fi

EXPERIMENTS="pose_range"

POSE_LEVELS="\
restricted:25:35:-30:30:-15:15 \
moderate:10:50:-135:135:-60:60 \
wide:5:65:-180:180:-90:90"

# Required so all three levels use the same normalized sliced-LHS design.
SHARED_POSE_RANGE_SPECS="\
25:35:-30:30:-15:15 \
10:50:-135:135:-60:60 \
5:65:-180:180:-90:90"

BASELINE_TOTAL_SCANS="${BASELINE_TOTAL_SCANS:-108}"
BASELINE_NOISE_STD_MM="${BASELINE_NOISE_STD_MM:-0.20}"
MAX_TRIALS="${MAX_TRIALS:-100}"

# Separate paths prevent mixing with previous pose-range results.
DATASET_ROOT="${DATASET_ROOT:-dataset/sliced_lhs_pose_range_revised_mc100}"
RESULT_ROOT="${RESULT_ROOT:-results/sliced_lhs_pose_range_revised_mc100_nonlinear}"

# Regenerate because the pose ranges changed.
FORCE_REGENERATE="${FORCE_REGENERATE:-1}"
FORCE_RERUN="${FORCE_RERUN:-1}"

MAKE_PLOTS="${MAKE_PLOTS:-1}"
PLOT_SUCCESS_ONLY="${PLOT_SUCCESS_ONLY:-0}"

EXPERIMENTS="${EXPERIMENTS}" \
POSE_LEVELS="${POSE_LEVELS}" \
SHARED_POSE_RANGE_SPECS="${SHARED_POSE_RANGE_SPECS}" \
BASELINE_TOTAL_SCANS="${BASELINE_TOTAL_SCANS}" \
BASELINE_NOISE_STD_MM="${BASELINE_NOISE_STD_MM}" \
MAX_TRIALS="${MAX_TRIALS}" \
DATASET_ROOT="${DATASET_ROOT}" \
RESULT_ROOT="${RESULT_ROOT}" \
FORCE_REGENERATE="${FORCE_REGENERATE}" \
FORCE_RERUN="${FORCE_RERUN}" \
MAKE_PLOTS="${MAKE_PLOTS}" \
PLOT_SUCCESS_ONLY="${PLOT_SUCCESS_ONLY}" \
bash "${BASE_SCRIPT}"
