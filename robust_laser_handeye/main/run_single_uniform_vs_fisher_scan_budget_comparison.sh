#!/usr/bin/env bash
set -euo pipefail

# Evaluate prefixes of one completed Single Uniform/Fisher sequential dataset.
# No acquisition is regenerated: N-budget results use the first N saved scans.
#
# Full experiment:
#   MAX_TRIALS=100 \
#   bash main/run_single_uniform_vs_fisher_scan_budget_comparison.sh
#
# Recompute calibration outputs:
#   FORCE_RERUN=1 MAX_TRIALS=100 \
#   bash main/run_single_uniform_vs_fisher_scan_budget_comparison.sh

TOTAL_SCANS="${TOTAL_SCANS:-108}"
INITIAL_SCANS="${INITIAL_SCANS:-27}"
CANDIDATE_POOL_SIZE="${CANDIDATE_POOL_SIZE:-2000}"
SCAN_COUNTS="${SCAN_COUNTS:-27 36 45 54 72 90 108}"
SOURCE_TRIALS="${SOURCE_TRIALS:-100}"
MAX_TRIALS="${MAX_TRIALS:-100}"

GENERATION_SEED="${GENERATION_SEED:-17}"
CALIBRATION_SEED="${CALIBRATION_SEED:-1701}"
MEASUREMENT_SEED="${MEASUREMENT_SEED:-1701}"
SELECTION_INITIALIZATION_SEED="${SELECTION_INITIALIZATION_SEED:-1701}"

PROFILE_POINTS="${PROFILE_POINTS:-100}"
PROFILE_HALF_WIDTH_MM="${PROFILE_HALF_WIDTH_MM:-25}"
TANGENT_RANGE_MM="${TANGENT_RANGE_MM:-100}"
PROFILE_DEPTH_RANGE_MM="${PROFILE_DEPTH_RANGE_MM:-60 150}"
CANDIDATE_TARGET_U_RANGE_MM="${CANDIDATE_TARGET_U_RANGE_MM:--70 70}"
CANDIDATE_TARGET_V_RANGE_MM="${CANDIDATE_TARGET_V_RANGE_MM:--70 70}"
CANDIDATE_VIEW_TILT_RANGE_DEG="${CANDIDATE_VIEW_TILT_RANGE_DEG:-5 80}"
CANDIDATE_VIEW_AZIMUTH_RANGE_DEG="${CANDIDATE_VIEW_AZIMUTH_RANGE_DEG:--180 180}"
CANDIDATE_SENSOR_ROLL_RANGE_DEG="${CANDIDATE_SENSOR_ROLL_RANGE_DEG:--180 180}"
CANDIDATE_MAX_BATCHES="${CANDIDATE_MAX_BATCHES:-300}"
CANDIDATE_BATCH_MULTIPLIER="${CANDIDATE_BATCH_MULTIPLIER:-8}"
PLANE_ANGLE_RANGE_DEG="${PLANE_ANGLE_RANGE_DEG:--15 15}"
PLANE_CENTER_XY_RANGE_MM="${PLANE_CENTER_XY_RANGE_MM:--100 100}"
PLANE_CENTER_Z_RANGE_MM="${PLANE_CENTER_Z_RANGE_MM:-400 550}"
MIN_ABS_PLANE_NORMAL_Z="${MIN_ABS_PLANE_NORMAL_Z:-1e-4}"
VERIFICATION_ATOL="${VERIFICATION_ATOL:-1e-8}"

FISHER_OBJECTIVE="${FISHER_OBJECTIVE:-d_optimal}"
FISHER_PROFILE_NOISE_STD_MM="${FISHER_PROFILE_NOISE_STD_MM:-0.20}"
FISHER_ROTATION_SCALE_DEG="${FISHER_ROTATION_SCALE_DEG:-2}"
FISHER_TRANSLATION_SCALE_MM="${FISHER_TRANSLATION_SCALE_MM:-10}"
FISHER_PLANE_NORMAL_SCALE_DEG="${FISHER_PLANE_NORMAL_SCALE_DEG:-20}"
FISHER_PLANE_OFFSET_SCALE_MM="${FISHER_PLANE_OFFSET_SCALE_MM:-100}"
MEASUREMENT_NOISE_STD_MM="${MEASUREMENT_NOISE_STD_MM:-0.20}"
MEASUREMENT_NOISE_AXIS="${MEASUREMENT_NOISE_AXIS:-xz}"

INIT_TRANSLATION_RANGE_MM="${INIT_TRANSLATION_RANGE_MM:-100}"
INIT_ANGLE_RANGE_DEG="${INIT_ANGLE_RANGE_DEG:-15}"
INIT_ROTATION_PERTURBATION="${INIT_ROTATION_PERTURBATION:-axis_angle}"
INIT_TRANSLATION_PERTURBATION="${INIT_TRANSLATION_PERTURBATION:-direction_norm}"
ESTIMATOR_MAX_ITER="${ESTIMATOR_MAX_ITER:-60}"
ESTIMATOR_TOL="${ESTIMATOR_TOL:-1e-7}"
MAX_ITER="${MAX_ITER:-3000}"
TOL="${TOL:-1e-5}"

DATASET_ROOT="${DATASET_ROOT:-dataset/single_uniform_vs_fisher}"
RESULT_ROOT="${RESULT_ROOT:-results/single_uniform_vs_fisher_scan_budget}"
FORCE_RERUN="${FORCE_RERUN:-0}"
MAKE_PLOTS="${MAKE_PLOTS:-1}"
PLOT_SUCCESS_ONLY="${PLOT_SUCCESS_ONLY:-0}"
PLOT_DPI="${PLOT_DPI:-300}"

CALIBRATOR="main/calibrate.py"
VALIDATOR="main/validate_experiment_artifact.py"
PLOTTER="main/make_plot/plot_single_uniform_fisher_scan_budget.py"
METHODS=("single_uniform" "single_fisher")

parse_pair() {
  local name="$1"
  local text="$2"
  local -n output="$3"
  read -r -a output <<< "${text}"
  if (( ${#output[@]} != 2 )); then
    echo "ERROR: ${name} must contain exactly two values." >&2
    exit 1
  fi
}

collection_for_method() {
  case "$1" in
    single_uniform) printf "%s" "${SOURCE_DATASET_DIR}/single_plane_uniform" ;;
    single_fisher) printf "%s" "${SOURCE_DATASET_DIR}/single_plane_fisher" ;;
    *)
      echo "ERROR: unknown method: $1" >&2
      return 1
      ;;
  esac
}

for file in "${CALIBRATOR}" "${VALIDATOR}" "${PLOTTER}"; do
  if [[ ! -f "${file}" ]]; then
    echo "ERROR: required file not found: ${file}" >&2
    echo "Run this script from the robust_laser_handeye repository root." >&2
    exit 1
  fi
done
for value in \
  "${TOTAL_SCANS}" "${INITIAL_SCANS}" "${CANDIDATE_POOL_SIZE}" \
  "${SOURCE_TRIALS}" "${MAX_TRIALS}" "${PROFILE_POINTS}" \
  "${CANDIDATE_MAX_BATCHES}" "${CANDIDATE_BATCH_MULTIPLIER}" \
  "${ESTIMATOR_MAX_ITER}" "${MAX_ITER}" "${PLOT_DPI}"; do
  if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: expected a positive integer: ${value}" >&2
    exit 1
  fi
done
if (( MAX_TRIALS > SOURCE_TRIALS )); then
  echo "ERROR: MAX_TRIALS cannot exceed SOURCE_TRIALS." >&2
  exit 1
fi
if [[ "${FORCE_RERUN}" != "0" && "${FORCE_RERUN}" != "1" ]]; then
  echo "ERROR: FORCE_RERUN must be 0 or 1." >&2
  exit 1
fi
if [[ "${MAKE_PLOTS}" != "0" && "${MAKE_PLOTS}" != "1" ]]; then
  echo "ERROR: MAKE_PLOTS must be 0 or 1." >&2
  exit 1
fi
if [[ "${PLOT_SUCCESS_ONLY}" != "0" \
      && "${PLOT_SUCCESS_ONLY}" != "1" ]]; then
  echo "ERROR: PLOT_SUCCESS_ONLY must be 0 or 1." >&2
  exit 1
fi

read -r -a SCAN_COUNT_VALUES <<< "${SCAN_COUNTS}"
if (( ${#SCAN_COUNT_VALUES[@]} < 2 )); then
  echo "ERROR: SCAN_COUNTS must contain at least two budgets." >&2
  exit 1
fi
previous=0
for count in "${SCAN_COUNT_VALUES[@]}"; do
  if ! [[ "${count}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: invalid scan count: ${count}" >&2
    exit 1
  fi
  if (( count <= previous || count > TOTAL_SCANS )); then
    echo "ERROR: SCAN_COUNTS must be increasing and <= TOTAL_SCANS." >&2
    exit 1
  fi
  previous="${count}"
done
if (( SCAN_COUNT_VALUES[0] != INITIAL_SCANS )); then
  echo "ERROR: first SCAN_COUNTS value must equal INITIAL_SCANS." >&2
  exit 1
fi
if (( SCAN_COUNT_VALUES[${#SCAN_COUNT_VALUES[@]} - 1] != TOTAL_SCANS )); then
  echo "ERROR: last SCAN_COUNTS value must equal TOTAL_SCANS." >&2
  exit 1
fi

parse_pair "PROFILE_DEPTH_RANGE_MM" \
  "${PROFILE_DEPTH_RANGE_MM}" PROFILE_DEPTH
parse_pair "CANDIDATE_TARGET_U_RANGE_MM" \
  "${CANDIDATE_TARGET_U_RANGE_MM}" CANDIDATE_U
parse_pair "CANDIDATE_TARGET_V_RANGE_MM" \
  "${CANDIDATE_TARGET_V_RANGE_MM}" CANDIDATE_V
parse_pair "CANDIDATE_VIEW_TILT_RANGE_DEG" \
  "${CANDIDATE_VIEW_TILT_RANGE_DEG}" CANDIDATE_TILT
parse_pair "CANDIDATE_VIEW_AZIMUTH_RANGE_DEG" \
  "${CANDIDATE_VIEW_AZIMUTH_RANGE_DEG}" CANDIDATE_AZIMUTH
parse_pair "CANDIDATE_SENSOR_ROLL_RANGE_DEG" \
  "${CANDIDATE_SENSOR_ROLL_RANGE_DEG}" CANDIDATE_ROLL
parse_pair "PLANE_ANGLE_RANGE_DEG" \
  "${PLANE_ANGLE_RANGE_DEG}" PLANE_ANGLE
parse_pair "PLANE_CENTER_XY_RANGE_MM" \
  "${PLANE_CENTER_XY_RANGE_MM}" PLANE_CENTER_XY
parse_pair "PLANE_CENTER_Z_RANGE_MM" \
  "${PLANE_CENTER_Z_RANGE_MM}" PLANE_CENTER_Z

NOISE_TAG="${MEASUREMENT_NOISE_STD_MM//./p}"
INIT_T_TAG="${INIT_TRANSLATION_RANGE_MM//./p}"
INIT_R_TAG="${INIT_ANGLE_RANGE_DEG//./p}"
CONDITION_TAG="N${TOTAL_SCANS}_I${INITIAL_SCANS}_P${CANDIDATE_POOL_SIZE}_${FISHER_OBJECTIVE}_noise${NOISE_TAG}_init_t${INIT_T_TAG}_r${INIT_R_TAG}_${INIT_TRANSLATION_PERTURBATION}_${INIT_ROTATION_PERTURBATION}"
SOURCE_DATASET_DIR="${DATASET_ROOT}/${CONDITION_TAG}"
CONDITION_RESULT_ROOT="${RESULT_ROOT}/${CONDITION_TAG}"

if [[ ! -f "${SOURCE_DATASET_DIR}/comparison_manifest.json" ]]; then
  echo "ERROR: source Uniform/Fisher dataset is missing:" >&2
  echo "  ${SOURCE_DATASET_DIR}" >&2
  echo "Generate it first with:" >&2
  echo "  MAX_TRIALS=${SOURCE_TRIALS} bash main/run_single_uniform_vs_fisher_comparison.sh" >&2
  exit 1
fi

VALIDATION_ARGS=(
  single-uniform-fisher-dataset
  --manifest "${SOURCE_DATASET_DIR}/comparison_manifest.json"
  --trials "${SOURCE_TRIALS}"
  --seed "${GENERATION_SEED}"
  --total-scans "${TOTAL_SCANS}"
  --initial-scans "${INITIAL_SCANS}"
  --candidate-pool-size "${CANDIDATE_POOL_SIZE}"
  --profile-points "${PROFILE_POINTS}"
  --profile-half-width-mm "${PROFILE_HALF_WIDTH_MM}"
  --tangent-range-mm "${TANGENT_RANGE_MM}"
  --profile-depth-range-mm "${PROFILE_DEPTH[@]}"
  --candidate-target-u-range-mm "${CANDIDATE_U[@]}"
  --candidate-target-v-range-mm "${CANDIDATE_V[@]}"
  --candidate-view-tilt-range-deg "${CANDIDATE_TILT[@]}"
  --candidate-view-azimuth-range-deg "${CANDIDATE_AZIMUTH[@]}"
  --candidate-sensor-roll-range-deg "${CANDIDATE_ROLL[@]}"
  --candidate-max-batches "${CANDIDATE_MAX_BATCHES}"
  --candidate-batch-multiplier "${CANDIDATE_BATCH_MULTIPLIER}"
  --plane-angle-range-deg "${PLANE_ANGLE[@]}"
  --plane-center-xy-range-mm "${PLANE_CENTER_XY[@]}"
  --plane-center-z-range-mm "${PLANE_CENTER_Z[@]}"
  --min-abs-plane-normal-z "${MIN_ABS_PLANE_NORMAL_Z}"
  --verification-atol "${VERIFICATION_ATOL}"
  --fisher-objective "${FISHER_OBJECTIVE}"
  --fisher-profile-noise-std-mm "${FISHER_PROFILE_NOISE_STD_MM}"
  --fisher-rotation-scale-deg "${FISHER_ROTATION_SCALE_DEG}"
  --fisher-translation-scale-mm "${FISHER_TRANSLATION_SCALE_MM}"
  --fisher-plane-normal-scale-deg "${FISHER_PLANE_NORMAL_SCALE_DEG}"
  --fisher-plane-offset-scale-mm "${FISHER_PLANE_OFFSET_SCALE_MM}"
  --measurement-noise-std-mm "${MEASUREMENT_NOISE_STD_MM}"
  --measurement-noise-axis "${MEASUREMENT_NOISE_AXIS}"
  --measurement-seed "${MEASUREMENT_SEED}"
  --initialization-seed "${SELECTION_INITIALIZATION_SEED}"
  --initial-translation-range-mm "${INIT_TRANSLATION_RANGE_MM}"
  --initial-angle-range-deg "${INIT_ANGLE_RANGE_DEG}"
  --initial-rotation-perturbation "${INIT_ROTATION_PERTURBATION}"
  --initial-translation-perturbation "${INIT_TRANSLATION_PERTURBATION}"
  --estimator-max-iterations "${ESTIMATOR_MAX_ITER}"
  --estimator-tolerance "${ESTIMATOR_TOL}"
)
PYTHONPATH=. python3 "${VALIDATOR}" "${VALIDATION_ARGS[@]}"

mkdir -p "${CONDITION_RESULT_ROOT}"

echo "============================================================"
echo "Single Uniform/Fisher scan-budget comparison"
echo "============================================================"
echo "Trials             : ${MAX_TRIALS}"
echo "Shared bootstrap   : ${INITIAL_SCANS}"
echo "Scan budgets       : ${SCAN_COUNTS}"
echo "Source scans       : ${TOTAL_SCANS}"
echo "Candidate pool     : ${CANDIDATE_POOL_SIZE}"
echo "Fisher objective   : ${FISHER_OBJECTIVE}"
echo "Saved noise        : ${MEASUREMENT_NOISE_STD_MM} mm, ${MEASUREMENT_NOISE_AXIS}"
echo "Calibration noise  : 0 mm"
echo "Initialization     : ${INIT_TRANSLATION_RANGE_MM} mm / ${INIT_ANGLE_RANGE_DEG} deg"
echo "Source dataset     : ${SOURCE_DATASET_DIR}"
echo "Results            : ${CONDITION_RESULT_ROOT}"
echo "============================================================"

for scan_count in "${SCAN_COUNT_VALUES[@]}"; do
  for method in "${METHODS[@]}"; do
    collection="$(collection_for_method "${method}")"
    result_dir="${CONDITION_RESULT_ROOT}/N${scan_count}/${method}"
    digest_file="${result_dir}/input_collection_manifest.sha256"
    read -r collection_digest _ < <(sha256sum "${collection}/collection.json")

    echo
    echo "[N=${scan_count}] ${method}"
    if [[ "${FORCE_RERUN}" == "1" && -e "${result_dir}" ]]; then
      rm -rf "${result_dir}"
    fi
    if [[ -f "${result_dir}/summary.json" ]]; then
      if [[ ! -f "${digest_file}" ]] \
        || [[ "$(<"${digest_file}")" != "${collection_digest}" ]]; then
        echo "ERROR: cached result does not match the source collection." >&2
        echo "Use FORCE_RERUN=1." >&2
        exit 1
      fi
      echo "  completed result found; validating."
    else
      PYTHONPATH=. python3 "${CALIBRATOR}" \
        --collection "${collection}" \
        --output-dir "${result_dir}" \
        --mode iterative \
        --max-trials "${MAX_TRIALS}" \
        --max-scans-per-trial "${scan_count}" \
        --seed "${CALIBRATION_SEED}" \
        --noise-axis "${MEASUREMENT_NOISE_AXIS}" \
        --noise-std-mm 0 \
        --init-mode carlson \
        --init-translation-range-mm "${INIT_TRANSLATION_RANGE_MM}" \
        --init-angle-range-deg "${INIT_ANGLE_RANGE_DEG}" \
        --init-rotation-perturbation "${INIT_ROTATION_PERTURBATION}" \
        --init-translation-perturbation \
          "${INIT_TRANSLATION_PERTURBATION}" \
        --max-iter "${MAX_ITER}" \
        --tol "${TOL}"
      printf "%s\n" "${collection_digest}" > "${digest_file}"
    fi

    PYTHONPATH=. python3 "${VALIDATOR}" result \
      --summary "${result_dir}/summary.json" \
      --collection "${collection}" \
      --trials "${MAX_TRIALS}" \
      --seed "${CALIBRATION_SEED}" \
      --noise-axis "${MEASUREMENT_NOISE_AXIS}" \
      --noise-std-mm 0 \
      --init-translation-range-mm "${INIT_TRANSLATION_RANGE_MM}" \
      --init-angle-range-deg "${INIT_ANGLE_RANGE_DEG}" \
      --init-rotation-perturbation "${INIT_ROTATION_PERTURBATION}" \
      --init-translation-perturbation \
        "${INIT_TRANSLATION_PERTURBATION}" \
      --max-scans-per-trial "${scan_count}" \
      --max-iter "${MAX_ITER}" \
      --tol "${TOL}"
  done
done

if [[ "${MAKE_PLOTS}" == "1" ]]; then
  COMMON_PLOT_ARGS=(
    --result-root "${CONDITION_RESULT_ROOT}"
    --scan-counts "${SCAN_COUNT_VALUES[@]}"
    --dpi "${PLOT_DPI}"
  )
  if [[ "${PLOT_SUCCESS_ONLY}" == "1" ]]; then
    COMMON_PLOT_ARGS+=(--success-only)
  fi
  MPLBACKEND="${MPLBACKEND:-Agg}" \
  MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/lvs-matplotlib-cache}" \
  PYTHONPATH=. python3 "${PLOTTER}" \
    "${COMMON_PLOT_ARGS[@]}" \
    --output-dir "${CONDITION_RESULT_ROOT}/plots_hide_outliers" \
    --hide-outliers \
    --no-log-errors
  MPLBACKEND="${MPLBACKEND:-Agg}" \
  MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/lvs-matplotlib-cache}" \
  PYTHONPATH=. python3 "${PLOTTER}" \
    "${COMMON_PLOT_ARGS[@]}" \
    --output-dir "${CONDITION_RESULT_ROOT}/plots_show_outliers" \
    --log-errors
fi

echo
echo "============================================================"
echo "Scan-budget comparison complete."
echo "Results: ${CONDITION_RESULT_ROOT}"
if [[ "${MAKE_PLOTS}" == "1" ]]; then
  echo "Plots: ${CONDITION_RESULT_ROOT}/plots_hide_outliers"
  echo "Plots: ${CONDITION_RESULT_ROOT}/plots_show_outliers"
fi
echo "============================================================"
