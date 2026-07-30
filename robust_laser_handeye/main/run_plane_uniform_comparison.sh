#!/usr/bin/env bash
set -euo pipefail

# Plane-relative pose sampling comparison:
#   1) single_uniform : one physical plane, Latin-hypercube relative poses
#   2) three_random   : three physical planes, shared-global random poses
#   3) three_uniform  : every physical plane receives the complete matched
#                       Latin-hypercube base pose set
#
# Primary comparison : single_uniform vs three_random
# Full-set comparison: single_uniform vs three_uniform (3x scan count)

TOTAL_SCANS="${TOTAL_SCANS:-108}"
MAX_TRIALS="${MAX_TRIALS:-100}"

GENERATION_SEED="${GENERATION_SEED:-17}"
CALIBRATION_SEED="${CALIBRATION_SEED:-1701}"

PROFILE_POINTS="${PROFILE_POINTS:-100}"
PROFILE_HALF_WIDTH_MM="${PROFILE_HALF_WIDTH_MM:-25}"
TANGENT_RANGE_MM="${TANGENT_RANGE_MM:-100}"
PROFILE_DEPTH_RANGE_MM="${PROFILE_DEPTH_RANGE_MM:-60 150}"

# Moderate shared-global random baseline. The restricted ranges used by the
# other historical experiments were 48–62 / 35–55 / ±20 degrees.
RANDOM_VIEW_TILT_RANGE_DEG="${RANDOM_VIEW_TILT_RANGE_DEG:-35 75}"
RANDOM_VIEW_AZIMUTH_RANGE_DEG="${RANDOM_VIEW_AZIMUTH_RANGE_DEG:-20 70}"
RANDOM_SENSOR_ROLL_RANGE_DEG="${RANDOM_SENSOR_ROLL_RANGE_DEG:--90 90}"
if [[ -z "${RANDOM_POSE_TAG+x}" ]]; then
  if [[ "${RANDOM_VIEW_TILT_RANGE_DEG}" == "35 75" \
        && "${RANDOM_VIEW_AZIMUTH_RANGE_DEG}" == "20 70" \
        && "${RANDOM_SENSOR_ROLL_RANGE_DEG}" == "-90 90" ]]; then
    RANDOM_POSE_TAG="moderate"
  elif [[ "${RANDOM_VIEW_TILT_RANGE_DEG}" == "48 62" \
          && "${RANDOM_VIEW_AZIMUTH_RANGE_DEG}" == "35 55" \
          && "${RANDOM_SENSOR_ROLL_RANGE_DEG}" == "-20 20" ]]; then
    RANDOM_POSE_TAG="restricted"
  else
    RANDOM_POSE_TAG="custom"
  fi
fi

UNIFORM_TARGET_U_RANGE_MM="${UNIFORM_TARGET_U_RANGE_MM:--40 40}"
UNIFORM_TARGET_V_RANGE_MM="${UNIFORM_TARGET_V_RANGE_MM:--40 40}"
UNIFORM_VIEW_TILT_RANGE_DEG="${UNIFORM_VIEW_TILT_RANGE_DEG:-10 60}"
UNIFORM_VIEW_AZIMUTH_RANGE_DEG="${UNIFORM_VIEW_AZIMUTH_RANGE_DEG:--180 180}"
UNIFORM_SENSOR_ROLL_RANGE_DEG="${UNIFORM_SENSOR_ROLL_RANGE_DEG:--180 180}"
UNIFORM_MAX_BATCHES="${UNIFORM_MAX_BATCHES:-200}"
UNIFORM_BATCH_MULTIPLIER="${UNIFORM_BATCH_MULTIPLIER:-8}"

PLANE_ANGLE_RANGE_DEG="${PLANE_ANGLE_RANGE_DEG:--15 15}"
PLANE_CENTER_XY_RANGE_MM="${PLANE_CENTER_XY_RANGE_MM:--100 100}"
PLANE_CENTER_Z_RANGE_MM="${PLANE_CENTER_Z_RANGE_MM:-400 550}"
MAX_LOCAL_POSE_TRIALS="${MAX_LOCAL_POSE_TRIALS:-200000}"
MIN_ABS_PLANE_NORMAL_Z="${MIN_ABS_PLANE_NORMAL_Z:-1e-4}"
VERIFICATION_ATOL="${VERIFICATION_ATOL:-1e-8}"

NOISE_AXIS="${NOISE_AXIS:-xz}"
NOISE_STD_MM="${NOISE_STD_MM:-0.20}"

INIT_TRANSLATION_RANGE_MM="${INIT_TRANSLATION_RANGE_MM:-100}"
INIT_ANGLE_RANGE_DEG="${INIT_ANGLE_RANGE_DEG:-15}"
INIT_ROTATION_PERTURBATION="${INIT_ROTATION_PERTURBATION:-axis_angle}"
INIT_TRANSLATION_PERTURBATION="${INIT_TRANSLATION_PERTURBATION:-direction_norm}"

MAX_ITER="${MAX_ITER:-3000}"
TOL="${TOL:-1e-5}"

DATASET_ROOT="${DATASET_ROOT:-dataset/plane_uniform_comparison}"
DATASET_DIR="${DATASET_ROOT}/N${TOTAL_SCANS}_random_${RANDOM_POSE_TAG}_three_uniform_fullset"
INIT_T_TAG="${INIT_TRANSLATION_RANGE_MM//./p}"
INIT_R_TAG="${INIT_ANGLE_RANGE_DEG//./p}"
NOISE_TAG="${NOISE_STD_MM//./p}"
RESULT_ROOT="${RESULT_ROOT:-results/plane_uniform_comparison_N${TOTAL_SCANS}_random_${RANDOM_POSE_TAG}_three_uniform_fullset_noise${NOISE_TAG}_init_t${INIT_T_TAG}_r${INIT_R_TAG}_${INIT_TRANSLATION_PERTURBATION}_${INIT_ROTATION_PERTURBATION}}"

FORCE_REGENERATE="${FORCE_REGENERATE:-0}"
FORCE_RERUN="${FORCE_RERUN:-0}"
MAKE_PLOTS="${MAKE_PLOTS:-1}"
PLOT_SUCCESS_ONLY="${PLOT_SUCCESS_ONLY:-0}"
PLOT_DPI="${PLOT_DPI:-300}"
POSE_VIS_TRIAL_INDEX="${POSE_VIS_TRIAL_INDEX:-0}"
POSE_VIS_MAX_ORIENTATION_AXES="${POSE_VIS_MAX_ORIENTATION_AXES:-18}"

GENERATOR="main/generate_plane_uniform_comparison.py"
CALIBRATOR="main/calibrate.py"
VALIDATOR="main/validate_experiment_artifact.py"
PLOTTER="main/make_plot/plot_plane_uniform_comparison.py"
POSE_PLOTTER="main/make_plot/plot_plane_uniform_pose_geometry.py"

for REQUIRED_FILE in \
  "${GENERATOR}" "${CALIBRATOR}" "${VALIDATOR}" \
  "${PLOTTER}" "${POSE_PLOTTER}"; do
  if [[ ! -f "${REQUIRED_FILE}" ]]; then
    echo "ERROR: required file not found: ${REQUIRED_FILE}" >&2
    exit 1
  fi
done

for FLAG_NAME in FORCE_REGENERATE FORCE_RERUN MAKE_PLOTS PLOT_SUCCESS_ONLY; do
  FLAG_VALUE="${!FLAG_NAME}"
  if [[ "${FLAG_VALUE}" != "0" && "${FLAG_VALUE}" != "1" ]]; then
    echo "ERROR: ${FLAG_NAME} must be 0 or 1." >&2
    exit 1
  fi
done

if ! [[ "${TOTAL_SCANS}" =~ ^[1-9][0-9]*$ ]] \
  || (( TOTAL_SCANS < 9 || TOTAL_SCANS % 3 != 0 )); then
  echo "ERROR: TOTAL_SCANS must be at least 9 and divisible by 3." >&2
  exit 1
fi
if ! [[ "${MAX_TRIALS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: MAX_TRIALS must be a positive integer." >&2
  exit 1
fi
if ! [[ "${PLOT_DPI}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: PLOT_DPI must be a positive integer." >&2
  exit 1
fi
if ! [[ "${RANDOM_POSE_TAG}" =~ ^[A-Za-z0-9_-]+$ ]]; then
  echo "ERROR: RANDOM_POSE_TAG may contain only letters, digits, _ and -." >&2
  exit 1
fi
if ! [[ "${POSE_VIS_TRIAL_INDEX}" =~ ^[0-9]+$ ]] \
  || (( POSE_VIS_TRIAL_INDEX >= MAX_TRIALS )); then
  echo "ERROR: POSE_VIS_TRIAL_INDEX must be in [0, MAX_TRIALS)." >&2
  exit 1
fi
if ! [[ "${POSE_VIS_MAX_ORIENTATION_AXES}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: POSE_VIS_MAX_ORIENTATION_AXES must be a positive integer." >&2
  exit 1
fi

if [[ "${MAKE_PLOTS}" == "1" ]] \
  && ! MPLBACKEND="${MPLBACKEND:-Agg}" \
    MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/lvs-matplotlib-cache}" \
    PYTHONPATH=. \
    python3 "${PLOTTER}" --help >/dev/null; then
  echo "ERROR: plot dependencies are unavailable; set MAKE_PLOTS=0 or install them." >&2
  exit 1
fi
if [[ "${MAKE_PLOTS}" == "1" ]] \
  && ! MPLBACKEND="${MPLBACKEND:-Agg}" \
    MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/lvs-matplotlib-cache}" \
    PYTHONPATH=. \
    python3 "${POSE_PLOTTER}" --help >/dev/null; then
  echo "ERROR: pose-plot dependencies are unavailable." >&2
  exit 1
fi

read -r -a PROFILE_DEPTH_ARGS <<< "${PROFILE_DEPTH_RANGE_MM}"
read -r -a RANDOM_TILT_ARGS <<< "${RANDOM_VIEW_TILT_RANGE_DEG}"
read -r -a RANDOM_AZIMUTH_ARGS <<< "${RANDOM_VIEW_AZIMUTH_RANGE_DEG}"
read -r -a RANDOM_ROLL_ARGS <<< "${RANDOM_SENSOR_ROLL_RANGE_DEG}"
read -r -a UNIFORM_U_ARGS <<< "${UNIFORM_TARGET_U_RANGE_MM}"
read -r -a UNIFORM_V_ARGS <<< "${UNIFORM_TARGET_V_RANGE_MM}"
read -r -a UNIFORM_TILT_ARGS <<< "${UNIFORM_VIEW_TILT_RANGE_DEG}"
read -r -a UNIFORM_AZIMUTH_ARGS <<< "${UNIFORM_VIEW_AZIMUTH_RANGE_DEG}"
read -r -a UNIFORM_ROLL_ARGS <<< "${UNIFORM_SENSOR_ROLL_RANGE_DEG}"
read -r -a PLANE_ANGLE_ARGS <<< "${PLANE_ANGLE_RANGE_DEG}"
read -r -a PLANE_XY_ARGS <<< "${PLANE_CENTER_XY_RANGE_MM}"
read -r -a PLANE_Z_ARGS <<< "${PLANE_CENTER_Z_RANGE_MM}"

PAIR_ARRAY_NAMES=(
  PROFILE_DEPTH_ARGS RANDOM_TILT_ARGS RANDOM_AZIMUTH_ARGS RANDOM_ROLL_ARGS
  UNIFORM_U_ARGS UNIFORM_V_ARGS UNIFORM_TILT_ARGS UNIFORM_AZIMUTH_ARGS
  UNIFORM_ROLL_ARGS PLANE_ANGLE_ARGS PLANE_XY_ARGS PLANE_Z_ARGS
)

validate_pair_array() {
  local ARRAY_NAME="$1"
  local -n PAIR_VALUES="${ARRAY_NAME}"
  if (( ${#PAIR_VALUES[@]} != 2 )); then
    echo "ERROR: ${ARRAY_NAME} must contain exactly two numbers." >&2
    exit 1
  fi
}

for ARRAY_NAME in "${PAIR_ARRAY_NAMES[@]}"; do
  validate_pair_array "${ARRAY_NAME}"
done

SINGLE_COLLECTION="${DATASET_DIR}/single_plane_uniform"
THREE_RANDOM_COLLECTION="${DATASET_DIR}/three_plane_random"
THREE_UNIFORM_COLLECTION="${DATASET_DIR}/three_plane_uniform"

mkdir -p "${DATASET_ROOT}" "${RESULT_ROOT}"

echo "============================================================"
echo "Plane-relative uniform pose comparison"
echo "============================================================"
echo "Base pose set      : ${TOTAL_SCANS}"
echo "Single uniform     : ${TOTAL_SCANS} scans on plane 0"
echo "Three random       : $((TOTAL_SCANS / 3)) scans/plane × 3 = ${TOTAL_SCANS}"
echo "Three uniform      : ${TOTAL_SCANS} scans/plane × 3 = $((3 * TOTAL_SCANS))"
echo "Trials per method  : ${MAX_TRIALS}"
echo "Profile noise      : ${NOISE_AXIS}, sigma=${NOISE_STD_MM} mm"
echo "Initialization     : ${INIT_TRANSLATION_RANGE_MM} mm norm / ${INIT_ANGLE_RANGE_DEG} deg axis-angle"
echo "Random pose level  : ${RANDOM_POSE_TAG}"
echo "Random tilt/az/roll: ${RANDOM_VIEW_TILT_RANGE_DEG} / ${RANDOM_VIEW_AZIMUTH_RANGE_DEG} / ${RANDOM_SENSOR_ROLL_RANGE_DEG} deg"
echo "Methods            : single_uniform, three_random, three_uniform"
echo "Results            : ${RESULT_ROOT}"
echo "============================================================"

if [[ "${FORCE_REGENERATE}" == "1" && -e "${DATASET_DIR}" ]]; then
  echo "[generate] removing existing dataset: ${DATASET_DIR}"
  rm -rf "${DATASET_DIR}"
fi

if [[ -f "${DATASET_DIR}/comparison_manifest.json" \
      && -f "${SINGLE_COLLECTION}/collection.json" \
      && -f "${THREE_RANDOM_COLLECTION}/collection.json" \
      && -f "${THREE_UNIFORM_COLLECTION}/collection.json" ]]; then
  echo "[generate] existing dataset found; validating before reuse."
else
  if [[ -e "${DATASET_DIR}" ]]; then
    echo "ERROR: incomplete dataset directory exists: ${DATASET_DIR}" >&2
    echo "Use FORCE_REGENERATE=1 to recreate it." >&2
    exit 1
  fi

  echo "[generate] creating three paired collections..."
  PYTHONPATH=. python3 "${GENERATOR}" \
    --trials "${MAX_TRIALS}" \
    --seed "${GENERATION_SEED}" \
    --output-dir "${DATASET_DIR}" \
    --total-scans "${TOTAL_SCANS}" \
    --profile-points "${PROFILE_POINTS}" \
    --profile-half-width-mm "${PROFILE_HALF_WIDTH_MM}" \
    --tangent-range-mm "${TANGENT_RANGE_MM}" \
    --profile-depth-range-mm "${PROFILE_DEPTH_ARGS[@]}" \
    --random-view-tilt-range-deg "${RANDOM_TILT_ARGS[@]}" \
    --random-view-azimuth-range-deg "${RANDOM_AZIMUTH_ARGS[@]}" \
    --random-sensor-roll-range-deg "${RANDOM_ROLL_ARGS[@]}" \
    --uniform-target-u-range-mm "${UNIFORM_U_ARGS[@]}" \
    --uniform-target-v-range-mm "${UNIFORM_V_ARGS[@]}" \
    --uniform-view-tilt-range-deg "${UNIFORM_TILT_ARGS[@]}" \
    --uniform-view-azimuth-range-deg "${UNIFORM_AZIMUTH_ARGS[@]}" \
    --uniform-sensor-roll-range-deg "${UNIFORM_ROLL_ARGS[@]}" \
    --uniform-max-batches "${UNIFORM_MAX_BATCHES}" \
    --uniform-batch-multiplier "${UNIFORM_BATCH_MULTIPLIER}" \
    --plane-angle-range-deg "${PLANE_ANGLE_ARGS[@]}" \
    --plane-center-xy-range-mm "${PLANE_XY_ARGS[@]}" \
    --plane-center-z-range-mm "${PLANE_Z_ARGS[@]}" \
    --max-local-pose-trials "${MAX_LOCAL_POSE_TRIALS}" \
    --min-abs-plane-normal-z "${MIN_ABS_PLANE_NORMAL_Z}" \
    --verification-atol "${VERIFICATION_ATOL}"
fi

if ! PYTHONPATH=. python3 "${VALIDATOR}" uniform-dataset \
  --manifest "${DATASET_DIR}/comparison_manifest.json" \
  --trials "${MAX_TRIALS}" \
  --seed "${GENERATION_SEED}" \
  --total-scans "${TOTAL_SCANS}" \
  --profile-points "${PROFILE_POINTS}" \
  --profile-half-width-mm "${PROFILE_HALF_WIDTH_MM}" \
  --tangent-range-mm "${TANGENT_RANGE_MM}" \
  --profile-depth-range-mm "${PROFILE_DEPTH_ARGS[@]}" \
  --random-view-tilt-range-deg "${RANDOM_TILT_ARGS[@]}" \
  --random-view-azimuth-range-deg "${RANDOM_AZIMUTH_ARGS[@]}" \
  --random-sensor-roll-range-deg "${RANDOM_ROLL_ARGS[@]}" \
  --uniform-target-u-range-mm "${UNIFORM_U_ARGS[@]}" \
  --uniform-target-v-range-mm "${UNIFORM_V_ARGS[@]}" \
  --uniform-view-tilt-range-deg "${UNIFORM_TILT_ARGS[@]}" \
  --uniform-view-azimuth-range-deg "${UNIFORM_AZIMUTH_ARGS[@]}" \
  --uniform-sensor-roll-range-deg "${UNIFORM_ROLL_ARGS[@]}" \
  --uniform-max-batches "${UNIFORM_MAX_BATCHES}" \
  --uniform-batch-multiplier "${UNIFORM_BATCH_MULTIPLIER}" \
  --plane-angle-range-deg "${PLANE_ANGLE_ARGS[@]}" \
  --plane-center-xy-range-mm "${PLANE_XY_ARGS[@]}" \
  --plane-center-z-range-mm "${PLANE_Z_ARGS[@]}" \
  --max-local-pose-trials "${MAX_LOCAL_POSE_TRIALS}" \
  --min-abs-plane-normal-z "${MIN_ABS_PLANE_NORMAL_Z}" \
  --verification-atol "${VERIFICATION_ATOL}"; then
  echo "Use FORCE_REGENERATE=1 to replace the incompatible dataset." >&2
  exit 1
fi

METHOD_KEYS=(single_uniform three_random three_uniform)
COLLECTIONS=(
  "${SINGLE_COLLECTION}"
  "${THREE_RANDOM_COLLECTION}"
  "${THREE_UNIFORM_COLLECTION}"
)

for INDEX in "${!METHOD_KEYS[@]}"; do
  METHOD="${METHOD_KEYS[INDEX]}"
  COLLECTION="${COLLECTIONS[INDEX]}"
  METHOD_RESULT="${RESULT_ROOT}/${METHOD}"

  echo
  echo "------------------------------------------------------------"
  echo "[${METHOD}] collection=${COLLECTION}"
  echo "------------------------------------------------------------"

  if [[ "${FORCE_RERUN}" == "1" && -e "${METHOD_RESULT}" ]]; then
    echo "[rerun] removing existing result: ${METHOD_RESULT}"
    rm -rf "${METHOD_RESULT}"
  fi

  if [[ -f "${METHOD_RESULT}/summary.json" ]]; then
    echo "[${METHOD}] completed result found; validating before reuse."
  else
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
      --max-iter "${MAX_ITER}" \
      --tol "${TOL}"
  fi

  if ! PYTHONPATH=. python3 "${VALIDATOR}" result \
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
    --tol "${TOL}"; then
    echo "Use FORCE_RERUN=1 to replace the incompatible ${METHOD} result." >&2
    exit 1
  fi
done

if [[ "${MAKE_PLOTS}" == "1" ]]; then
  echo
  echo "[plot] creating outlier-hidden and outlier-visible figures..."
  PLOT_ARGS_COMMON=(
    --result-root "${RESULT_ROOT}"
    --dpi "${PLOT_DPI}"
  )
  if [[ "${PLOT_SUCCESS_ONLY}" == "1" ]]; then
    PLOT_ARGS_COMMON+=(--success-only)
  fi

  MPLBACKEND="${MPLBACKEND:-Agg}" \
  MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/lvs-matplotlib-cache}" \
  PYTHONPATH=. python3 "${PLOTTER}" \
    "${PLOT_ARGS_COMMON[@]}" \
    --output-dir "${RESULT_ROOT}/plots_hide_outliers" \
    --hide-outliers \
    --no-log-errors \
    --no-log-iterations

  MPLBACKEND="${MPLBACKEND:-Agg}" \
  MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/lvs-matplotlib-cache}" \
  PYTHONPATH=. python3 "${PLOTTER}" \
    "${PLOT_ARGS_COMMON[@]}" \
    --output-dir "${RESULT_ROOT}/plots_show_outliers" \
    --log-errors \
    --log-iterations

  MPLBACKEND="${MPLBACKEND:-Agg}" \
  MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/lvs-matplotlib-cache}" \
  PYTHONPATH=. python3 "${POSE_PLOTTER}" \
    --dataset-root "${DATASET_DIR}" \
    --output-dir "${RESULT_ROOT}/pose_visualization" \
    --trial-index "${POSE_VIS_TRIAL_INDEX}" \
    --max-orientation-axes "${POSE_VIS_MAX_ORIENTATION_AXES}" \
    --dpi "${PLOT_DPI}"
fi

echo
echo "============================================================"
echo "Plane-relative uniform comparison completed."
echo "Primary    : single_uniform vs three_random"
echo "Additional : single_uniform vs three_uniform full pose set/plane"
echo "Results    : ${RESULT_ROOT}"
echo "============================================================"
