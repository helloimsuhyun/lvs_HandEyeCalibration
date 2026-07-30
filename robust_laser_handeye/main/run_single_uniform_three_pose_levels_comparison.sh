#!/usr/bin/env bash
set -euo pipefail

# Four-way equal-scan comparison using the already generated paired sources:
#   single_uniform, three_hard(restricted), three_moderate, three_easy(wide).

TOTAL_SCANS="${TOTAL_SCANS:-108}"
MAX_TRIALS="${MAX_TRIALS:-100}"
CALIBRATION_SEED="${CALIBRATION_SEED:-1701}"
NOISE_AXIS="${NOISE_AXIS:-xz}"
NOISE_STD_MM="${NOISE_STD_MM:-0.20}"
INIT_TRANSLATION_RANGE_MM="${INIT_TRANSLATION_RANGE_MM:-100}"
INIT_ANGLE_RANGE_DEG="${INIT_ANGLE_RANGE_DEG:-15}"
INIT_ROTATION_PERTURBATION="${INIT_ROTATION_PERTURBATION:-axis_angle}"
INIT_TRANSLATION_PERTURBATION="${INIT_TRANSLATION_PERTURBATION:-direction_norm}"
MAX_ITER="${MAX_ITER:-3000}"
TOL="${TOL:-1e-5}"
FORCE_RERUN="${FORCE_RERUN:-0}"
MAKE_PLOTS="${MAKE_PLOTS:-1}"
PLOT_DPI="${PLOT_DPI:-300}"
PLOT_SUCCESS_ONLY="${PLOT_SUCCESS_ONLY:-0}"
POSE_VIS_TRIAL_INDEX="${POSE_VIS_TRIAL_INDEX:-0}"
POSE_VIS_MAX_ORIENTATION_AXES="${POSE_VIS_MAX_ORIENTATION_AXES:-18}"

SINGLE_UNIFORM_COLLECTION="${SINGLE_UNIFORM_COLLECTION:-dataset/plane_uniform_comparison/N${TOTAL_SCANS}/single_plane_uniform}"
POSE_DIVERSITY_DATASET_ROOT="${POSE_DIVERSITY_DATASET_ROOT:-dataset/fair_plane_global_pose_diversity}"
THREE_HARD_COLLECTION="${THREE_HARD_COLLECTION:-${POSE_DIVERSITY_DATASET_ROOT}/restricted_N${TOTAL_SCANS}/three_plane}"
THREE_MODERATE_COLLECTION="${THREE_MODERATE_COLLECTION:-${POSE_DIVERSITY_DATASET_ROOT}/moderate_N${TOTAL_SCANS}/three_plane}"
THREE_EASY_COLLECTION="${THREE_EASY_COLLECTION:-${POSE_DIVERSITY_DATASET_ROOT}/wide_N${TOTAL_SCANS}/three_plane}"

COMBINED_DATASET_ROOT="${COMBINED_DATASET_ROOT:-dataset/single_uniform_three_pose_levels/N${TOTAL_SCANS}}"
NOISE_TAG="${NOISE_STD_MM//./p}"
INIT_T_TAG="${INIT_TRANSLATION_RANGE_MM//./p}"
INIT_R_TAG="${INIT_ANGLE_RANGE_DEG//./p}"
RESULT_ROOT="${RESULT_ROOT:-results/single_uniform_three_pose_levels_N${TOTAL_SCANS}_noise${NOISE_TAG}_init_t${INIT_T_TAG}_r${INIT_R_TAG}_${INIT_TRANSLATION_PERTURBATION}_${INIT_ROTATION_PERTURBATION}}"

CALIBRATOR="main/calibrate.py"
VALIDATOR="main/validate_experiment_artifact.py"
PLOTTER="main/make_plot/plot_plane_uniform_comparison.py"
POSE_PLOTTER="main/make_plot/plot_plane_uniform_pose_geometry.py"

METHODS=(single_uniform three_hard three_moderate three_easy)
SOURCES=(
  "${SINGLE_UNIFORM_COLLECTION}"
  "${THREE_HARD_COLLECTION}"
  "${THREE_MODERATE_COLLECTION}"
  "${THREE_EASY_COLLECTION}"
)

if ! [[ "${TOTAL_SCANS}" =~ ^[1-9][0-9]*$ ]] \
  || (( TOTAL_SCANS % 3 != 0 )); then
  echo "ERROR: TOTAL_SCANS must be positive and divisible by 3." >&2
  exit 1
fi
if ! [[ "${MAX_TRIALS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: MAX_TRIALS must be a positive integer." >&2
  exit 1
fi
for FLAG in FORCE_RERUN MAKE_PLOTS PLOT_SUCCESS_ONLY; do
  if [[ "${!FLAG}" != "0" && "${!FLAG}" != "1" ]]; then
    echo "ERROR: ${FLAG} must be 0 or 1." >&2
    exit 1
  fi
done
for FILE in "${CALIBRATOR}" "${VALIDATOR}" "${PLOTTER}" "${POSE_PLOTTER}"; do
  if [[ ! -f "${FILE}" ]]; then
    echo "ERROR: required file not found: ${FILE}" >&2
    exit 1
  fi
done
for SOURCE in "${SOURCES[@]}"; do
  if [[ ! -f "${SOURCE}/collection.json" ]]; then
    echo "ERROR: source collection is missing: ${SOURCE}" >&2
    echo "Generate the plane-uniform and pose-diversity datasets first." >&2
    exit 1
  fi
done

mkdir -p "${COMBINED_DATASET_ROOT}" "${RESULT_ROOT}"
for INDEX in "${!METHODS[@]}"; do
  METHOD="${METHODS[INDEX]}"
  SOURCE="${SOURCES[INDEX]}"
  LINK="${COMBINED_DATASET_ROOT}/${METHOD}"
  SOURCE_REAL="$(realpath -e "${SOURCE}")"
  if [[ -L "${LINK}" ]]; then
    if [[ "$(realpath -e "${LINK}")" != "${SOURCE_REAL}" ]]; then
      echo "ERROR: ${LINK} points to a different collection." >&2
      exit 1
    fi
  elif [[ -e "${LINK}" ]]; then
    echo "ERROR: expected a collection link but found another path: ${LINK}" >&2
    exit 1
  else
    ln -s "${SOURCE_REAL}" "${LINK}"
  fi
done

echo "============================================================"
echo "Single-uniform vs three-plane pose levels"
echo "============================================================"
echo "Scans per method : ${TOTAL_SCANS}"
echo "Three split      : $((TOTAL_SCANS / 3)) scans/plane × 3"
echo "Methods          : single_uniform / three_hard / three_moderate / three_easy"
echo "Noise            : ${NOISE_AXIS}, sigma=${NOISE_STD_MM} mm"
echo "Initialization   : ${INIT_TRANSLATION_RANGE_MM} mm / ${INIT_ANGLE_RANGE_DEG} deg"
echo "Results          : ${RESULT_ROOT}"
echo "============================================================"

for INDEX in "${!METHODS[@]}"; do
  METHOD="${METHODS[INDEX]}"
  COLLECTION="${COMBINED_DATASET_ROOT}/${METHOD}"
  METHOD_RESULT="${RESULT_ROOT}/${METHOD}"

  if [[ "${FORCE_RERUN}" == "1" && -e "${METHOD_RESULT}" ]]; then
    rm -rf "${METHOD_RESULT}"
  fi
  if [[ ! -f "${METHOD_RESULT}/summary.json" ]]; then
    echo "[${METHOD}] calibrating..."
    PYTHONPATH=. python3 "${CALIBRATOR}" \
      --collection "${COLLECTION}" \
      --output-dir "${METHOD_RESULT}" \
      --mode iterative \
      --seed "${CALIBRATION_SEED}" \
      --noise-axis "${NOISE_AXIS}" \
      --noise-std-mm "${NOISE_STD_MM}" \
      --init-mode carlson \
      --init-translation-range-mm "${INIT_TRANSLATION_RANGE_MM}" \
      --init-angle-range-deg "${INIT_ANGLE_RANGE_DEG}" \
      --init-rotation-perturbation "${INIT_ROTATION_PERTURBATION}" \
      --init-translation-perturbation "${INIT_TRANSLATION_PERTURBATION}" \
      --max-trials "${MAX_TRIALS}" \
      --max-iter "${MAX_ITER}" \
      --tol "${TOL}"
  else
    echo "[${METHOD}] completed result found; validating."
  fi

  PYTHONPATH=. python3 "${VALIDATOR}" result \
    --summary "${METHOD_RESULT}/summary.json" \
    --collection "${COLLECTION}" \
    --trials "${MAX_TRIALS}" \
    --seed "${CALIBRATION_SEED}" \
    --noise-axis "${NOISE_AXIS}" \
    --noise-std-mm "${NOISE_STD_MM}" \
    --init-translation-range-mm "${INIT_TRANSLATION_RANGE_MM}" \
    --init-angle-range-deg "${INIT_ANGLE_RANGE_DEG}" \
    --init-rotation-perturbation "${INIT_ROTATION_PERTURBATION}" \
    --init-translation-perturbation "${INIT_TRANSLATION_PERTURBATION}" \
    --max-iter "${MAX_ITER}" \
    --tol "${TOL}"
done

if [[ "${MAKE_PLOTS}" == "1" ]]; then
  COMMON_PLOT_ARGS=(
    --design three_pose_levels
    --result-root "${RESULT_ROOT}"
    --dpi "${PLOT_DPI}"
  )
  if [[ "${PLOT_SUCCESS_ONLY}" == "1" ]]; then
    COMMON_PLOT_ARGS+=(--success-only)
  fi
  MPLBACKEND="${MPLBACKEND:-Agg}" \
  MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/lvs-matplotlib-cache}" \
  PYTHONPATH=. python3 "${PLOTTER}" \
    "${COMMON_PLOT_ARGS[@]}" \
    --output-dir "${RESULT_ROOT}/plots_hide_outliers" \
    --hide-outliers --no-log-errors --no-log-iterations
  MPLBACKEND="${MPLBACKEND:-Agg}" \
  MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/lvs-matplotlib-cache}" \
  PYTHONPATH=. python3 "${PLOTTER}" \
    "${COMMON_PLOT_ARGS[@]}" \
    --output-dir "${RESULT_ROOT}/plots_show_outliers" \
    --log-errors --log-iterations
  MPLBACKEND="${MPLBACKEND:-Agg}" \
  MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/lvs-matplotlib-cache}" \
  PYTHONPATH=. python3 "${POSE_PLOTTER}" \
    --design three_pose_levels \
    --dataset-root "${COMBINED_DATASET_ROOT}" \
    --output-dir "${RESULT_ROOT}/pose_visualization" \
    --trial-index "${POSE_VIS_TRIAL_INDEX}" \
    --max-orientation-axes "${POSE_VIS_MAX_ORIENTATION_AXES}" \
    --dpi "${PLOT_DPI}"
fi

echo "Completed: ${RESULT_ROOT}"
