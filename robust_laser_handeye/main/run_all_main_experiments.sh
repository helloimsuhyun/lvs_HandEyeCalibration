#!/usr/bin/env bash
set -euo pipefail

# Run every main experiment with one standardized initialization:
#   translation norm: at most 100 mm
#   rotation angle  : at most 15 deg
# Both directions are sampled isotropically by default.
#
# The Fisher experiment additionally uses a balanced 27-scan bootstrap
# (9 scans/plane for the three-plane branch), leaving 81/108 scans for the
# random-versus-Fisher continuation.
#
# Full run:
#   bash main/run_all_main_experiments.sh
#
# Validate orchestration without running experiments:
#   DRY_RUN=1 bash main/run_all_main_experiments.sh
#
# Common overrides are forwarded to every child runner:
#   MAX_TRIALS=3 MAKE_PLOTS=0 \
#     bash main/run_all_main_experiments.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
cd "${REPOSITORY_ROOT}"

INIT_TRANSLATION_RANGE_MM="${INIT_TRANSLATION_RANGE_MM:-100}"
INIT_ANGLE_RANGE_DEG="${INIT_ANGLE_RANGE_DEG:-15}"
INIT_ROTATION_PERTURBATION="${INIT_ROTATION_PERTURBATION:-axis_angle}"
INIT_TRANSLATION_PERTURBATION="${INIT_TRANSLATION_PERTURBATION:-direction_norm}"
INIT_LEVELS="${INIT_LEVELS:-easy:25:5 medium:100:15 hard:200:30}"

INITIAL_RANDOM_SCANS="${INITIAL_RANDOM_SCANS:-27}"
SELECTION_ESTIMATOR_MAX_ITER="${SELECTION_ESTIMATOR_MAX_ITER:-60}"
FISHER_OBJECTIVE="${FISHER_OBJECTIVE:-d_optimal}"
FISHER_ROTATION_SCALE_DEG="${FISHER_ROTATION_SCALE_DEG:-2}"
FISHER_TRANSLATION_SCALE_MM="${FISHER_TRANSLATION_SCALE_MM:-10}"

MAX_TRIALS="${MAX_TRIALS:-100}"
GENERATION_SEED="${GENERATION_SEED:-17}"
CALIBRATION_SEED="${CALIBRATION_SEED:-1701}"
MAX_ITER="${MAX_ITER:-3000}"
TOL="${TOL:-1e-5}"
MAKE_PLOTS="${MAKE_PLOTS:-1}"
FORCE_REGENERATE="${FORCE_REGENERATE:-0}"
FORCE_RERUN="${FORCE_RERUN:-0}"

MASTER_DATASET_ROOT="${MASTER_DATASET_ROOT:-dataset}"
MASTER_RESULT_ROOT="${MASTER_RESULT_ROOT:-results}"
MASTER_LOG_ROOT="${MASTER_LOG_ROOT:-results/all_main_experiment_logs}"
DRY_RUN="${DRY_RUN:-0}"

INIT_T_TAG="${INIT_TRANSLATION_RANGE_MM//./p}"
INIT_R_TAG="${INIT_ANGLE_RANGE_DEG//./p}"
INIT_RESULT_TAG="init_t${INIT_T_TAG}_r${INIT_R_TAG}_${INIT_TRANSLATION_PERTURBATION}_${INIT_ROTATION_PERTURBATION}"

RUN_INITIALIZATION_STANDARD="${RUN_INITIALIZATION_STANDARD:-1}"
RUN_FIXED_LINE_NOISE="${RUN_FIXED_LINE_NOISE:-1}"
RUN_FIXED_NOISE_LINE_COUNT="${RUN_FIXED_NOISE_LINE_COUNT:-1}"
RUN_POSE_DIVERSITY="${RUN_POSE_DIVERSITY:-1}"
RUN_FISHER_RANDOM="${RUN_FISHER_RANDOM:-1}"

for binary_setting in \
  "${MAKE_PLOTS}" \
  "${FORCE_REGENERATE}" \
  "${FORCE_RERUN}" \
  "${DRY_RUN}" \
  "${RUN_INITIALIZATION_STANDARD}" \
  "${RUN_FIXED_LINE_NOISE}" \
  "${RUN_FIXED_NOISE_LINE_COUNT}" \
  "${RUN_POSE_DIVERSITY}" \
  "${RUN_FISHER_RANDOM}"; do
  if [[ "${binary_setting}" != "0" && "${binary_setting}" != "1" ]]; then
    echo "ERROR: binary settings must be 0 or 1." >&2
    exit 1
  fi
done

if [[ "${FISHER_OBJECTIVE}" != "d_optimal" \
      && "${FISHER_OBJECTIVE}" != "e_optimal" ]]; then
  echo "ERROR: FISHER_OBJECTIVE must be d_optimal or e_optimal." >&2
  exit 1
fi

if [[ "${INIT_ROTATION_PERTURBATION}" != "axis_angle" \
      && "${INIT_ROTATION_PERTURBATION}" != "euler_xyz" ]]; then
  echo "ERROR: INIT_ROTATION_PERTURBATION must be axis_angle or euler_xyz." >&2
  exit 1
fi

if [[ "${INIT_TRANSLATION_PERTURBATION}" != "direction_norm" \
      && "${INIT_TRANSLATION_PERTURBATION}" != "box_xyz" ]]; then
  echo "ERROR: INIT_TRANSLATION_PERTURBATION must be direction_norm or box_xyz." >&2
  exit 1
fi

export \
  INIT_TRANSLATION_RANGE_MM \
  INIT_ANGLE_RANGE_DEG \
  INIT_ROTATION_PERTURBATION \
  INIT_TRANSLATION_PERTURBATION \
  INIT_LEVELS \
  INITIAL_RANDOM_SCANS \
  SELECTION_ESTIMATOR_MAX_ITER \
  FISHER_OBJECTIVE \
  FISHER_ROTATION_SCALE_DEG \
  FISHER_TRANSLATION_SCALE_MM \
  MAX_TRIALS \
  GENERATION_SEED \
  CALIBRATION_SEED \
  MAX_ITER \
  TOL \
  MAKE_PLOTS \
  FORCE_REGENERATE \
  FORCE_RERUN

if [[ "${DRY_RUN}" == "0" ]]; then
  mkdir -p \
    "${MASTER_DATASET_ROOT}" \
    "${MASTER_RESULT_ROOT}" \
    "${MASTER_LOG_ROOT}"
fi

run_experiment() {
  local enabled="$1"
  local label="$2"
  local script="$3"
  local dataset_subdir="$4"
  local result_subdir="$5"

  if [[ "${enabled}" == "0" ]]; then
    echo "[skip] ${label}"
    return
  fi
  if [[ ! -f "${script}" ]]; then
    echo "ERROR: experiment runner not found: ${script}" >&2
    exit 1
  fi

  local dataset_root="${MASTER_DATASET_ROOT}/${dataset_subdir}"
  local result_root="${MASTER_RESULT_ROOT}/${result_subdir}_${INIT_RESULT_TAG}"
  echo
  echo "============================================================"
  echo "[run-all] ${label}"
  echo "script  : ${script}"
  echo "dataset : ${dataset_root}"
  echo "result  : ${result_root}"
  echo "init    : ${INIT_TRANSLATION_RANGE_MM} mm / ${INIT_ANGLE_RANGE_DEG} deg max"
  echo "sampler : ${INIT_TRANSLATION_PERTURBATION} / ${INIT_ROTATION_PERTURBATION}"
  echo "============================================================"

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "DATASET_ROOT=${dataset_root} RESULT_ROOT=${result_root} bash ${script}"
    return
  fi

  DATASET_ROOT="${dataset_root}" \
  RESULT_ROOT="${result_root}" \
    bash "${script}" 2>&1 \
      | tee "${MASTER_LOG_ROOT}/${label}_${INIT_RESULT_TAG}.log"
}

echo "============================================================"
echo "All main experiments"
echo "============================================================"
echo "Trials per condition : ${MAX_TRIALS}"
echo "Initialization       : ${INIT_TRANSLATION_RANGE_MM} mm / ${INIT_ANGLE_RANGE_DEG} deg max"
echo "Perturbation model   : translation=${INIT_TRANSLATION_PERTURBATION}, rotation=${INIT_ROTATION_PERTURBATION}"
if [[ "${RUN_FISHER_RANDOM}" == "1" ]]; then
  echo "Fisher bootstrap     : ${INITIAL_RANDOM_SCANS}"
  echo "Fisher objective     : ${FISHER_OBJECTIVE}"
  echo "Fisher coord. metric : ${FISHER_ROTATION_SCALE_DEG} deg / ${FISHER_TRANSLATION_SCALE_MM} mm"
else
  echo "Fisher experiment    : excluded"
fi
echo "Dry run              : ${DRY_RUN}"
echo "============================================================"

run_experiment \
  "${RUN_INITIALIZATION_STANDARD}" \
  "initialization_standard" \
  "main/run_fair_plane_initialization_levels.sh" \
  "fair_plane_initialization_shared_global" \
  "fair_plane_initialization_shared_global"

run_experiment \
  "${RUN_FIXED_LINE_NOISE}" \
  "fixed_line_noise" \
  "main/run_fixed_line_noise_comparison.sh" \
  "fair_plane_noise_fixed_line_shared_global" \
  "fair_plane_noise_fixed_line_shared_global"

run_experiment \
  "${RUN_FIXED_NOISE_LINE_COUNT}" \
  "fixed_noise_line_count" \
  "main/run_fixed_noise_line_count_comparison.sh" \
  "fair_plane_line_count_shared_global" \
  "fair_plane_line_count_shared_global"

run_experiment \
  "${RUN_POSE_DIVERSITY}" \
  "pose_diversity" \
  "main/run_pose_diversity_comparison.sh" \
  "fair_plane_global_pose_diversity" \
  "fair_plane_global_pose_diversity"

run_experiment \
  "${RUN_FISHER_RANDOM}" \
  "fisher_random" \
  "main/run_fisher_random_policy_comparison.sh" \
  "fisher_random_policy_comparison" \
  "fisher_random_policy_comparison"

echo
echo "============================================================"
echo "All enabled main experiments completed."
echo "Logs: ${MASTER_LOG_ROOT}"
echo "============================================================"
