#!/usr/bin/env bash
set -euo pipefail

# Fair single-plane comparison:
#   single_uniform : greedy maximin in normalized plane-relative pose space
#   single_fisher  : same bootstrap, then sequential active Fisher selection
#
# Both branches share the ground truth, plane, candidate bank, bootstrap poses,
# candidate-keyed measurement noise, initialization, and total scan count.
#
# Full run:
#   MAX_TRIALS=100 \
#   bash main/run_single_uniform_vs_fisher_comparison.sh
#
# Recreate all cached artifacts:
#   FORCE_REGENERATE=1 FORCE_RERUN=1 MAX_TRIALS=100 \
#   bash main/run_single_uniform_vs_fisher_comparison.sh
#
# Quick smoke test:
#   MAX_TRIALS=1 TOTAL_SCANS=10 INITIAL_SCANS=9 \
#   CANDIDATE_POOL_SIZE=18 PROFILE_POINTS=12 \
#   ESTIMATOR_MAX_ITER=10 MAX_ITER=100 PLOT_DPI=100 \
#   bash main/run_single_uniform_vs_fisher_comparison.sh

TOTAL_SCANS="${TOTAL_SCANS:-108}"
INITIAL_SCANS="${INITIAL_SCANS:-27}"
CANDIDATE_POOL_SIZE="${CANDIDATE_POOL_SIZE:-2000}"
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
RESULT_ROOT="${RESULT_ROOT:-results/single_uniform_vs_fisher}"
FORCE_REGENERATE="${FORCE_REGENERATE:-0}"
FORCE_RERUN="${FORCE_RERUN:-0}"
MAKE_PLOTS="${MAKE_PLOTS:-1}"
PLOT_SUCCESS_ONLY="${PLOT_SUCCESS_ONLY:-0}"
PLOT_DPI="${PLOT_DPI:-300}"
POSE_VIS_TRIAL_INDEX="${POSE_VIS_TRIAL_INDEX:-0}"
POSE_VIS_MAX_ORIENTATION_AXES="${POSE_VIS_MAX_ORIENTATION_AXES:-18}"

GENERATOR="main/generate_single_uniform_vs_fisher.py"
CALIBRATOR="main/calibrate.py"
VALIDATOR="main/validate_experiment_artifact.py"
PLOTTER="main/make_plot/plot_plane_uniform_comparison.py"
POSE_PLOTTER="main/make_plot/plot_plane_uniform_pose_geometry.py"

METHODS=("single_uniform" "single_fisher")

is_positive_decimal() {
  [[ "$1" =~ ^[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?$ ]] \
    && awk -v value="$1" 'BEGIN { exit !(value > 0) }'
}

is_nonnegative_decimal() {
  [[ "$1" =~ ^[0-9]+([.][0-9]+)?([eE][-+]?[0-9]+)?$ ]] \
    && awk -v value="$1" 'BEGIN { exit !(value >= 0) }'
}

require_binary_flag() {
  local name="$1"
  local value="$2"
  if [[ "${value}" != "0" && "${value}" != "1" ]]; then
    echo "ERROR: ${name} must be 0 or 1." >&2
    exit 1
  fi
}

parse_pair() {
  local name="$1"
  local text="$2"
  local -n output="$3"
  read -r -a output <<< "${text}"
  if (( ${#output[@]} != 2 )); then
    echo "ERROR: ${name} must contain exactly two numbers." >&2
    exit 1
  fi
}

collection_for_method() {
  case "$1" in
    single_uniform) printf "%s" "${DATASET_DIR}/single_plane_uniform" ;;
    single_fisher) printf "%s" "${DATASET_DIR}/single_plane_fisher" ;;
    *)
      echo "ERROR: unknown method: $1" >&2
      return 1
      ;;
  esac
}

for required_file in \
  "${GENERATOR}" "${CALIBRATOR}" "${VALIDATOR}" \
  "${PLOTTER}" "${POSE_PLOTTER}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "ERROR: required file not found: ${required_file}" >&2
    echo "Run this script from the robust_laser_handeye repository root." >&2
    exit 1
  fi
done

for value in \
  "${TOTAL_SCANS}" "${INITIAL_SCANS}" "${CANDIDATE_POOL_SIZE}" \
  "${MAX_TRIALS}" "${PROFILE_POINTS}" "${CANDIDATE_MAX_BATCHES}" \
  "${CANDIDATE_BATCH_MULTIPLIER}" "${ESTIMATOR_MAX_ITER}" \
  "${MAX_ITER}" "${PLOT_DPI}"; do
  if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: expected a positive integer: ${value}" >&2
    exit 1
  fi
done
if (( TOTAL_SCANS < 9 )); then
  echo "ERROR: TOTAL_SCANS must be at least 9." >&2
  exit 1
fi
if (( INITIAL_SCANS < 9 || INITIAL_SCANS >= TOTAL_SCANS )); then
  echo "ERROR: INITIAL_SCANS must be >= 9 and < TOTAL_SCANS." >&2
  exit 1
fi
if (( CANDIDATE_POOL_SIZE < TOTAL_SCANS )); then
  echo "ERROR: CANDIDATE_POOL_SIZE must be >= TOTAL_SCANS." >&2
  exit 1
fi
if (( PROFILE_POINTS < 2 )); then
  echo "ERROR: PROFILE_POINTS must be at least 2." >&2
  exit 1
fi
if (( POSE_VIS_TRIAL_INDEX < 0 || POSE_VIS_TRIAL_INDEX >= MAX_TRIALS )); then
  echo "ERROR: POSE_VIS_TRIAL_INDEX must lie in [0, MAX_TRIALS)." >&2
  exit 1
fi
if ! [[ "${POSE_VIS_MAX_ORIENTATION_AXES}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: POSE_VIS_MAX_ORIENTATION_AXES must be positive." >&2
  exit 1
fi

if [[ "${FISHER_OBJECTIVE}" != "d_optimal" \
      && "${FISHER_OBJECTIVE}" != "e_optimal" ]]; then
  echo "ERROR: FISHER_OBJECTIVE must be d_optimal or e_optimal." >&2
  exit 1
fi
if [[ "${MEASUREMENT_NOISE_AXIS}" != "z" \
      && "${MEASUREMENT_NOISE_AXIS}" != "xz" ]]; then
  echo "ERROR: MEASUREMENT_NOISE_AXIS must be z or xz." >&2
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

for value in \
  "${PROFILE_HALF_WIDTH_MM}" "${TANGENT_RANGE_MM}" \
  "${MIN_ABS_PLANE_NORMAL_Z}" "${VERIFICATION_ATOL}" \
  "${FISHER_PROFILE_NOISE_STD_MM}" "${FISHER_ROTATION_SCALE_DEG}" \
  "${FISHER_TRANSLATION_SCALE_MM}" "${FISHER_PLANE_NORMAL_SCALE_DEG}" \
  "${FISHER_PLANE_OFFSET_SCALE_MM}" "${ESTIMATOR_TOL}" "${TOL}"; do
  if ! is_positive_decimal "${value}"; then
    echo "ERROR: expected a positive decimal: ${value}" >&2
    exit 1
  fi
done
for value in \
  "${MEASUREMENT_NOISE_STD_MM}" "${INIT_TRANSLATION_RANGE_MM}" \
  "${INIT_ANGLE_RANGE_DEG}"; do
  if ! is_nonnegative_decimal "${value}"; then
    echo "ERROR: expected a non-negative decimal: ${value}" >&2
    exit 1
  fi
done

require_binary_flag "FORCE_REGENERATE" "${FORCE_REGENERATE}"
require_binary_flag "FORCE_RERUN" "${FORCE_RERUN}"
require_binary_flag "MAKE_PLOTS" "${MAKE_PLOTS}"
require_binary_flag "PLOT_SUCCESS_ONLY" "${PLOT_SUCCESS_ONLY}"

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

if [[ "${MAKE_PLOTS}" == "1" ]] \
  && ! MPLBACKEND="${MPLBACKEND:-Agg}" PYTHONPATH=. \
    python3 "${PLOTTER}" --help >/dev/null; then
  echo "ERROR: plotting dependencies are unavailable." >&2
  echo "Install requirements.txt or set MAKE_PLOTS=0." >&2
  exit 1
fi

NOISE_TAG="${MEASUREMENT_NOISE_STD_MM//./p}"
INIT_T_TAG="${INIT_TRANSLATION_RANGE_MM//./p}"
INIT_R_TAG="${INIT_ANGLE_RANGE_DEG//./p}"
CONDITION_TAG="N${TOTAL_SCANS}_I${INITIAL_SCANS}_P${CANDIDATE_POOL_SIZE}_${FISHER_OBJECTIVE}_noise${NOISE_TAG}_init_t${INIT_T_TAG}_r${INIT_R_TAG}_${INIT_TRANSLATION_PERTURBATION}_${INIT_ROTATION_PERTURBATION}"
DATASET_DIR="${DATASET_ROOT}/${CONDITION_TAG}"
CONDITION_RESULT_ROOT="${RESULT_ROOT}/${CONDITION_TAG}"

mkdir -p "${DATASET_ROOT}" "${RESULT_ROOT}"

echo "============================================================"
echo "Single-plane Uniform maximin vs active Fisher"
echo "============================================================"
echo "Trials                 : ${MAX_TRIALS}"
echo "Scans                  : bootstrap=${INITIAL_SCANS}, total=${TOTAL_SCANS}"
echo "Candidate pool         : ${CANDIDATE_POOL_SIZE}"
echo "Candidate u/v [mm]     : ${CANDIDATE_TARGET_U_RANGE_MM} / ${CANDIDATE_TARGET_V_RANGE_MM}"
echo "Candidate depth [mm]   : ${PROFILE_DEPTH_RANGE_MM}"
echo "Tilt / azimuth / roll  : ${CANDIDATE_VIEW_TILT_RANGE_DEG} / ${CANDIDATE_VIEW_AZIMUTH_RANGE_DEG} / ${CANDIDATE_SENSOR_ROLL_RANGE_DEG}"
echo "Fisher objective       : ${FISHER_OBJECTIVE}"
echo "Fisher design noise    : ${FISHER_PROFILE_NOISE_STD_MM} mm"
echo "Measurement noise      : ${MEASUREMENT_NOISE_STD_MM} mm, axis=${MEASUREMENT_NOISE_AXIS}"
echo "Calibration re-noise   : disabled (saved noisy profiles are reused)"
echo "Initialization         : ${INIT_TRANSLATION_RANGE_MM} mm / ${INIT_ANGLE_RANGE_DEG} deg"
echo "Perturbation model     : ${INIT_TRANSLATION_PERTURBATION} / ${INIT_ROTATION_PERTURBATION}"
echo "Dataset                : ${DATASET_DIR}"
echo "Results                : ${CONDITION_RESULT_ROOT}"
echo "============================================================"

DATASET_REBUILT=0
if [[ "${FORCE_REGENERATE}" == "1" && -e "${DATASET_DIR}" ]]; then
  echo "[generate] removing existing dataset: ${DATASET_DIR}"
  rm -rf "${DATASET_DIR}"
fi

DATASET_COMPLETE=1
if [[ ! -f "${DATASET_DIR}/comparison_manifest.json" ]]; then
  DATASET_COMPLETE=0
fi
for method in "${METHODS[@]}"; do
  collection="$(collection_for_method "${method}")"
  if [[ ! -f "${collection}/collection.json" ]]; then
    DATASET_COMPLETE=0
  fi
done

if [[ "${DATASET_COMPLETE}" == "1" ]]; then
  echo "[generate] complete dataset found; validating cache."
else
  if [[ -e "${DATASET_DIR}" ]]; then
    echo "ERROR: incomplete dataset directory exists: ${DATASET_DIR}" >&2
    echo "Use FORCE_REGENERATE=1 to recreate it." >&2
    exit 1
  fi
  echo "[generate] creating paired Uniform/Fisher collections..."
  PYTHONPATH=. python3 "${GENERATOR}" \
    --trials "${MAX_TRIALS}" \
    --seed "${GENERATION_SEED}" \
    --output-dir "${DATASET_DIR}" \
    --total-scans "${TOTAL_SCANS}" \
    --initial-scans "${INITIAL_SCANS}" \
    --candidate-pool-size "${CANDIDATE_POOL_SIZE}" \
    --profile-points "${PROFILE_POINTS}" \
    --profile-half-width-mm "${PROFILE_HALF_WIDTH_MM}" \
    --tangent-range-mm "${TANGENT_RANGE_MM}" \
    --profile-depth-range-mm "${PROFILE_DEPTH[@]}" \
    --candidate-target-u-range-mm "${CANDIDATE_U[@]}" \
    --candidate-target-v-range-mm "${CANDIDATE_V[@]}" \
    --candidate-view-tilt-range-deg "${CANDIDATE_TILT[@]}" \
    --candidate-view-azimuth-range-deg "${CANDIDATE_AZIMUTH[@]}" \
    --candidate-sensor-roll-range-deg "${CANDIDATE_ROLL[@]}" \
    --candidate-max-batches "${CANDIDATE_MAX_BATCHES}" \
    --candidate-batch-multiplier "${CANDIDATE_BATCH_MULTIPLIER}" \
    --plane-angle-range-deg "${PLANE_ANGLE[@]}" \
    --plane-center-xy-range-mm "${PLANE_CENTER_XY[@]}" \
    --plane-center-z-range-mm "${PLANE_CENTER_Z[@]}" \
    --min-abs-plane-normal-z "${MIN_ABS_PLANE_NORMAL_Z}" \
    --verification-atol "${VERIFICATION_ATOL}" \
    --fisher-objective "${FISHER_OBJECTIVE}" \
    --fisher-profile-noise-std-mm "${FISHER_PROFILE_NOISE_STD_MM}" \
    --fisher-rotation-scale-deg "${FISHER_ROTATION_SCALE_DEG}" \
    --fisher-translation-scale-mm "${FISHER_TRANSLATION_SCALE_MM}" \
    --fisher-plane-normal-scale-deg "${FISHER_PLANE_NORMAL_SCALE_DEG}" \
    --fisher-plane-offset-scale-mm "${FISHER_PLANE_OFFSET_SCALE_MM}" \
    --measurement-noise-std-mm "${MEASUREMENT_NOISE_STD_MM}" \
    --measurement-noise-axis "${MEASUREMENT_NOISE_AXIS}" \
    --measurement-seed "${MEASUREMENT_SEED}" \
    --initialization-seed "${SELECTION_INITIALIZATION_SEED}" \
    --initial-translation-range-mm "${INIT_TRANSLATION_RANGE_MM}" \
    --initial-angle-range-deg "${INIT_ANGLE_RANGE_DEG}" \
    --initial-rotation-perturbation "${INIT_ROTATION_PERTURBATION}" \
    --initial-translation-perturbation \
      "${INIT_TRANSLATION_PERTURBATION}" \
    --estimator-max-iterations "${ESTIMATOR_MAX_ITER}" \
    --estimator-tolerance "${ESTIMATOR_TOL}"
  DATASET_REBUILT=1
fi

VALIDATION_ARGS=(
  single-uniform-fisher-dataset
  --manifest "${DATASET_DIR}/comparison_manifest.json"
  --trials "${MAX_TRIALS}"
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
if ! PYTHONPATH=. python3 "${VALIDATOR}" "${VALIDATION_ARGS[@]}"; then
  echo "Use FORCE_REGENERATE=1 to replace the incompatible dataset." >&2
  exit 1
fi

if [[ "${DATASET_REBUILT}" == "1" && "${FORCE_RERUN}" != "1" ]]; then
  echo "[cache] dataset changed; forcing dependent calibrations to rerun."
  FORCE_RERUN=1
fi

for method in "${METHODS[@]}"; do
  collection="$(collection_for_method "${method}")"
  result_dir="${CONDITION_RESULT_ROOT}/${method}"
  digest_file="${result_dir}/input_collection_manifest.sha256"
  read -r collection_digest _ < <(sha256sum "${collection}/collection.json")

  echo
  echo "------------------------------------------------------------"
  echo "Method     : ${method}"
  echo "Collection : ${collection}"
  echo "------------------------------------------------------------"

  if [[ "${FORCE_RERUN}" == "1" && -e "${result_dir}" ]]; then
    echo "[${method}] removing existing result: ${result_dir}"
    rm -rf "${result_dir}"
  fi

  if [[ -f "${result_dir}/summary.json" ]]; then
    if [[ ! -f "${digest_file}" ]] \
      || [[ "$(<"${digest_file}")" != "${collection_digest}" ]]; then
      echo "ERROR: cached ${method} result does not match this dataset." >&2
      echo "Use FORCE_RERUN=1 to recompute it." >&2
      exit 1
    fi
    echo "[${method}] completed result found; validating cache."
  else
    echo "[${method}] calibrating..."
    PYTHONPATH=. python3 "${CALIBRATOR}" \
      --collection "${collection}" \
      --output-dir "${result_dir}" \
      --mode iterative \
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

  if ! PYTHONPATH=. python3 "${VALIDATOR}" result \
    --summary "${result_dir}/summary.json" \
    --collection "${collection}" \
    --trials "${MAX_TRIALS}" \
    --seed "${CALIBRATION_SEED}" \
    --noise-axis "${MEASUREMENT_NOISE_AXIS}" \
    --noise-std-mm 0 \
    --init-translation-range-mm "${INIT_TRANSLATION_RANGE_MM}" \
    --init-angle-range-deg "${INIT_ANGLE_RANGE_DEG}" \
    --init-rotation-perturbation "${INIT_ROTATION_PERTURBATION}" \
    --init-translation-perturbation "${INIT_TRANSLATION_PERTURBATION}" \
    --max-iter "${MAX_ITER}" \
    --tol "${TOL}"; then
    echo "Use FORCE_RERUN=1 to replace incompatible ${method} results." >&2
    exit 1
  fi
done

if [[ "${MAKE_PLOTS}" == "1" ]]; then
  echo
  echo "[plot] creating error, paired, diagnostic, and 3D pose plots..."
  PLOT_ARGS=(
    --result-root "${CONDITION_RESULT_ROOT}"
    --design single_uniform_vs_fisher
    --dpi "${PLOT_DPI}"
  )
  if [[ "${PLOT_SUCCESS_ONLY}" == "1" ]]; then
    PLOT_ARGS+=(--success-only)
  fi

  MPLBACKEND="${MPLBACKEND:-Agg}" PYTHONPATH=. \
    python3 "${PLOTTER}" \
      "${PLOT_ARGS[@]}" \
      --output-dir "${CONDITION_RESULT_ROOT}/plots_hide_outliers" \
      --hide-outliers \
      --no-log-errors \
      --no-log-iterations
  MPLBACKEND="${MPLBACKEND:-Agg}" PYTHONPATH=. \
    python3 "${PLOTTER}" \
      "${PLOT_ARGS[@]}" \
      --output-dir "${CONDITION_RESULT_ROOT}/plots_show_outliers" \
      --log-errors \
      --log-iterations
  MPLBACKEND="${MPLBACKEND:-Agg}" PYTHONPATH=. \
    python3 "${POSE_PLOTTER}" \
      --dataset-root "${DATASET_DIR}" \
      --output-dir "${CONDITION_RESULT_ROOT}/plots_pose_3d" \
      --design single_uniform_vs_fisher \
      --trial-index "${POSE_VIS_TRIAL_INDEX}" \
      --max-orientation-axes "${POSE_VIS_MAX_ORIENTATION_AXES}" \
      --tilt-range-deg "${CANDIDATE_TILT[@]}" \
      --azimuth-range-deg "${CANDIDATE_AZIMUTH[@]}" \
      --roll-range-deg "${CANDIDATE_ROLL[@]}" \
      --dpi "${PLOT_DPI}"
fi

echo
echo "============================================================"
echo "Single Uniform/Fisher comparison complete."
echo "Dataset manifest : ${DATASET_DIR}/comparison_manifest.json"
echo "Results          : ${CONDITION_RESULT_ROOT}"
if [[ "${MAKE_PLOTS}" == "1" ]]; then
  echo "Plots (robust)   : ${CONDITION_RESULT_ROOT}/plots_hide_outliers"
  echo "Plots (all)      : ${CONDITION_RESULT_ROOT}/plots_show_outliers"
  echo "3D pose plots    : ${CONDITION_RESULT_ROOT}/plots_pose_3d"
fi
echo "============================================================"
