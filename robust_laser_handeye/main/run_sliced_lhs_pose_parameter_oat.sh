#!/usr/bin/env bash
set -euo pipefail

# One-at-a-time pose-diversity ablation around the moderate baseline.
#
# Conditions:
#   1) moderate baseline: all pose spans retained at 100%
#   2) mild:   retain 50% of one pose-parameter span at a time
#   3) strong: retain 33.333% of one pose-parameter span at a time
#
# All non-ablated pose ranges remain at the moderate setting. Target U/V is
# held at +/-50 mm by default. This script uses separate dataset/result roots
# and delegates generation, calibration, validation, and plotting to the
# established sliced-LHS runner.
#
# Run from anywhere:
#   bash main/run_sliced_lhs_pose_parameter_oat.sh
#
# Fresh full rerun:
#   FORCE_REGENERATE=1 FORCE_RERUN=1 \
#   bash main/run_sliced_lhs_pose_parameter_oat.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPOSITORY_ROOT}"

# Moderate baseline:
#   tilt     10..50       (center=30, span=40)
#   azimuth -135..135     (center=0,  span=270)
#   roll     -60..60      (center=0,  span=120)
#
# Mild retains 50% of the corresponding span:
#   tilt     20..40       (span=20)
#   azimuth  -67.5..67.5  (span=135)
#   roll     -30..30      (span=60)
#
# Strong retains exactly 1/3 of the corresponding span:
#   tilt     23.3333333333..36.6666666667 (span=13.3333333334)
#   azimuth  -45..45                     (span=90)
#   roll     -20..20                     (span=40)
POSE_LEVELS="${POSE_LEVELS:-\
moderate:10:50:-135:135:-60:60 \
tilt_mild:20:40:-135:135:-60:60 \
azimuth_mild:10:50:-67.5:67.5:-60:60 \
roll_mild:10:50:-135:135:-30:30 \
tilt_strong:23.3333333333:36.6666666667:-135:135:-60:60 \
azimuth_strong:10:50:-45:45:-60:60 \
roll_strong:10:50:-135:135:-20:20}"

# Every condition must accept the same normalized sliced-LHS design so that
# trial-wise differences remain paired and attributable to the changed span.
SHARED_POSE_RANGE_SPECS="${SHARED_POSE_RANGE_SPECS:-\
10:50:-135:135:-60:60 \
20:40:-135:135:-60:60 \
10:50:-67.5:67.5:-60:60 \
10:50:-135:135:-30:30 \
23.3333333333:36.6666666667:-135:135:-60:60 \
10:50:-45:45:-60:60 \
10:50:-135:135:-20:20}"

TARGET_U_MIN_MM="${TARGET_U_MIN_MM:--50}"
TARGET_U_MAX_MM="${TARGET_U_MAX_MM:-50}"
TARGET_V_MIN_MM="${TARGET_V_MIN_MM:--50}"
TARGET_V_MAX_MM="${TARGET_V_MAX_MM:-50}"

DATASET_ROOT="${DATASET_ROOT:-dataset/sliced_lhs_pose_parameter_oat_mc100_nonlinear}"
RESULT_ROOT="${RESULT_ROOT:-results/sliced_lhs_pose_parameter_oat_mc100_nonlinear}"

export EXPERIMENTS="pose_range"
export POSE_LEVELS SHARED_POSE_RANGE_SPECS
export TARGET_U_MIN_MM TARGET_U_MAX_MM TARGET_V_MIN_MM TARGET_V_MAX_MM
export DATASET_ROOT RESULT_ROOT

bash main/run_sliced_lhs_5level_experiments.sh

MAKE_PLOTS="${MAKE_PLOTS:-1}"
PLOT_SUCCESS_ONLY="${PLOT_SUCCESS_ONLY:-0}"
PLOT_DPI="${PLOT_DPI:-300}"
if [[ "${MAKE_PLOTS}" == "1" ]]; then
  oat_plot_args=(
    --result-root "${RESULT_ROOT}/pose_range"
    --dpi "${PLOT_DPI}"
  )
  if [[ "${PLOT_SUCCESS_ONLY}" == "1" ]]; then
    oat_plot_args+=(--success-only)
  fi
  MPLBACKEND="${MPLBACKEND:-Agg}" PYTHONPATH=. \
    python3 main/make_plot/plot_pose_parameter_oat_percent_change.py \
      "${oat_plot_args[@]}"
fi
