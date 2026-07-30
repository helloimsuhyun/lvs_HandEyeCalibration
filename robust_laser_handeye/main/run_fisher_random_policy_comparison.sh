#!/usr/bin/env bash
set -euo pipefail

# Compare Fisher-information pose selection against a seeded random policy for:
#   1) single-plane calibration
#   2) three-plane calibration
#
# All four branches share the exact same balanced initial random scans.
# Random single/three additionally share the complete pose sequence. Fisher
# branches maximize the configured marginal hand-eye objective (D-optimal by
# default; E-optimal remains available for ablation).
#
# Run from the robust_laser_handeye repository root.
#
# Quick smoke test:
#   MAX_TRIALS=3 TOTAL_SCANS=18 INITIAL_RANDOM_SCANS=9 \
#   CANDIDATE_POOL_SIZE=36 PROFILE_POINTS=16 MAX_ITER=300 MAKE_PLOTS=1 \
#   bash main/run_fisher_random_policy_comparison.sh
#
# Full run:
#   MAX_TRIALS=100 \
#   bash main/run_fisher_random_policy_comparison.sh
#
# Recreate cached artifacts:
#   FORCE_REGENERATE=1 FORCE_RERUN=1 \
#   bash main/run_fisher_random_policy_comparison.sh

# ---------------------------------------------------------------------------
# Experiment settings
# ---------------------------------------------------------------------------

TOTAL_SCANS="${TOTAL_SCANS:-108}"
INITIAL_RANDOM_SCANS="${INITIAL_RANDOM_SCANS:-27}"
CANDIDATE_POOL_SIZE="${CANDIDATE_POOL_SIZE:-}"
MAX_TRIALS="${MAX_TRIALS:-100}"

GENERATION_SEED="${GENERATION_SEED:-17}"
CALIBRATION_SEED="${CALIBRATION_SEED:-1701}"

PROFILE_POINTS="${PROFILE_POINTS:-100}"
PROFILE_HALF_WIDTH_MM="${PROFILE_HALF_WIDTH_MM:-25}"
TANGENT_RANGE_MM="${TANGENT_RANGE_MM:-100}"
PROFILE_DEPTH_MIN_MM="${PROFILE_DEPTH_MIN_MM:-60}"
PROFILE_DEPTH_MAX_MM="${PROFILE_DEPTH_MAX_MM:-150}"
VIEW_TILT_MIN_DEG="${VIEW_TILT_MIN_DEG:-48}"
VIEW_TILT_MAX_DEG="${VIEW_TILT_MAX_DEG:-62}"
VIEW_AZIMUTH_MIN_DEG="${VIEW_AZIMUTH_MIN_DEG:-35}"
VIEW_AZIMUTH_MAX_DEG="${VIEW_AZIMUTH_MAX_DEG:-55}"
SENSOR_ROLL_MIN_DEG="${SENSOR_ROLL_MIN_DEG:--20}"
SENSOR_ROLL_MAX_DEG="${SENSOR_ROLL_MAX_DEG:-20}"

FISHER_PROFILE_NOISE_STD_MM="${FISHER_PROFILE_NOISE_STD_MM:-0.20}"
FISHER_OBJECTIVE="${FISHER_OBJECTIVE:-d_optimal}"
FISHER_ROTATION_SCALE_DEG="${FISHER_ROTATION_SCALE_DEG:-2}"
FISHER_TRANSLATION_SCALE_MM="${FISHER_TRANSLATION_SCALE_MM:-10}"
FISHER_PLANE_NORMAL_SCALE_DEG="${FISHER_PLANE_NORMAL_SCALE_DEG:-20}"
FISHER_PLANE_OFFSET_SCALE_MM="${FISHER_PLANE_OFFSET_SCALE_MM:-100}"

NOISE_STD_MM="${NOISE_STD_MM:-0.20}"
NOISE_AXIS="${NOISE_AXIS:-xz}"
INIT_TRANSLATION_RANGE_MM="${INIT_TRANSLATION_RANGE_MM:-100}"
INIT_ANGLE_RANGE_DEG="${INIT_ANGLE_RANGE_DEG:-15}"
INIT_ROTATION_PERTURBATION="${INIT_ROTATION_PERTURBATION:-axis_angle}"
INIT_TRANSLATION_PERTURBATION="${INIT_TRANSLATION_PERTURBATION:-direction_norm}"
SELECTION_ESTIMATOR_MAX_ITER="${SELECTION_ESTIMATOR_MAX_ITER:-60}"
SELECTION_ESTIMATOR_TOL="${SELECTION_ESTIMATOR_TOL:-1e-7}"
MAX_ITER="${MAX_ITER:-3000}"
TOL="${TOL:-1e-5}"

DATASET_ROOT="${DATASET_ROOT:-dataset/fisher_random_policy_comparison}"
RESULT_ROOT="${RESULT_ROOT:-results/fisher_random_policy_comparison}"

FORCE_REGENERATE="${FORCE_REGENERATE:-0}"
FORCE_RERUN="${FORCE_RERUN:-0}"
MAKE_PLOTS="${MAKE_PLOTS:-1}"
PLOT_SUCCESS_ONLY="${PLOT_SUCCESS_ONLY:-0}"
PLOT_DPI="${PLOT_DPI:-300}"

GENERATOR="main/generate_independent_random_plane_comparison.py"
CALIBRATOR="main/calibrate.py"
VALIDATOR="main/validate_experiment_artifact.py"
PLOTTER="main/make_plot/plot_fisher_random_policy_comparison.py"

METHODS=(
  "single_random"
  "single_fisher"
  "three_random"
  "three_fisher"
)

# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

is_positive_decimal() {
  [[ "$1" =~ ^[0-9]+([.][0-9]+)?$ ]] \
    && awk -v value="$1" 'BEGIN { exit !(value > 0) }'
}

is_nonnegative_decimal() {
  [[ "$1" =~ ^[0-9]+([.][0-9]+)?$ ]]
}

require_binary_input() {
  local name="$1"
  local value="$2"
  if [[ "${value}" != "0" && "${value}" != "1" ]]; then
    echo "ERROR: ${name} must be 0 or 1." >&2
    exit 1
  fi
}

collection_for_method() {
  local method="$1"
  case "${method}" in
    single_random) printf "%s" "${DATASET_DIR}/single_plane" ;;
    single_fisher) printf "%s" "${DATASET_DIR}/single_plane_fisher" ;;
    three_random) printf "%s" "${DATASET_DIR}/three_plane" ;;
    three_fisher) printf "%s" "${DATASET_DIR}/three_plane_fisher" ;;
    *)
      echo "ERROR: unknown method: ${method}" >&2
      return 1
      ;;
  esac
}

for required_file in \
  "${GENERATOR}" \
  "${CALIBRATOR}" \
  "${VALIDATOR}" \
  "${PLOTTER}"; do
  if [[ ! -f "${required_file}" ]]; then
    echo "ERROR: required file not found: ${required_file}" >&2
    echo "Run this script from the robust_laser_handeye repository root." >&2
    exit 1
  fi
done

for integer_setting in \
  "${TOTAL_SCANS}" \
  "${INITIAL_RANDOM_SCANS}" \
  "${MAX_TRIALS}" \
  "${PROFILE_POINTS}" \
  "${SELECTION_ESTIMATOR_MAX_ITER}" \
  "${MAX_ITER}" \
  "${PLOT_DPI}"; do
  if ! [[ "${integer_setting}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: integer settings must be positive: ${integer_setting}" >&2
    exit 1
  fi
done

if [[ -z "${CANDIDATE_POOL_SIZE}" ]]; then
  CANDIDATE_POOL_SIZE="$((3 * TOTAL_SCANS))"
fi
if ! [[ "${CANDIDATE_POOL_SIZE}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: CANDIDATE_POOL_SIZE must be a positive integer." >&2
  exit 1
fi

if (( TOTAL_SCANS < 9 || TOTAL_SCANS % 3 != 0 )); then
  echo "ERROR: TOTAL_SCANS must be at least 9 and divisible by 3." >&2
  exit 1
fi
if (( INITIAL_RANDOM_SCANS < 9 \
      || INITIAL_RANDOM_SCANS >= TOTAL_SCANS \
      || INITIAL_RANDOM_SCANS % 3 != 0 )); then
  echo "ERROR: INITIAL_RANDOM_SCANS must be >= 9, < TOTAL_SCANS, and divisible by 3." >&2
  exit 1
fi
if (( CANDIDATE_POOL_SIZE < TOTAL_SCANS \
      || CANDIDATE_POOL_SIZE % 3 != 0 )); then
  echo "ERROR: CANDIDATE_POOL_SIZE must be at least TOTAL_SCANS and divisible by 3." >&2
  exit 1
fi

if [[ "${FISHER_OBJECTIVE}" != "d_optimal" \
      && "${FISHER_OBJECTIVE}" != "e_optimal" ]]; then
  echo "ERROR: FISHER_OBJECTIVE must be d_optimal or e_optimal." >&2
  exit 1
fi

if [[ "${NOISE_AXIS}" != "z" && "${NOISE_AXIS}" != "xz" ]]; then
  echo "ERROR: NOISE_AXIS must be z or xz." >&2
  exit 1
fi

for value in \
  "${PROFILE_HALF_WIDTH_MM}" \
  "${TANGENT_RANGE_MM}" \
  "${FISHER_PROFILE_NOISE_STD_MM}" \
  "${FISHER_ROTATION_SCALE_DEG}" \
  "${FISHER_TRANSLATION_SCALE_MM}" \
  "${FISHER_PLANE_NORMAL_SCALE_DEG}" \
  "${FISHER_PLANE_OFFSET_SCALE_MM}"; do
  if ! is_positive_decimal "${value}"; then
    echo "ERROR: expected a positive decimal value: ${value}" >&2
    exit 1
  fi
done

for value in \
  "${NOISE_STD_MM}" \
  "${INIT_TRANSLATION_RANGE_MM}" \
  "${INIT_ANGLE_RANGE_DEG}"; do
  if ! is_nonnegative_decimal "${value}"; then
    echo "ERROR: expected a non-negative decimal value: ${value}" >&2
    exit 1
  fi
done

require_binary_input "FORCE_REGENERATE" "${FORCE_REGENERATE}"
require_binary_input "FORCE_RERUN" "${FORCE_RERUN}"
require_binary_input "MAKE_PLOTS" "${MAKE_PLOTS}"
require_binary_input "PLOT_SUCCESS_ONLY" "${PLOT_SUCCESS_ONLY}"

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

if [[ "${MAKE_PLOTS}" == "1" ]] \
  && ! MPLBACKEND="${MPLBACKEND:-Agg}" PYTHONPATH=. \
    python3 "${PLOTTER}" --help >/dev/null; then
  echo "ERROR: plotting dependencies are unavailable." >&2
  echo "Install requirements.txt or set MAKE_PLOTS=0." >&2
  exit 1
fi

INIT_T_TAG="${INIT_TRANSLATION_RANGE_MM//./p}"
INIT_R_TAG="${INIT_ANGLE_RANGE_DEG//./p}"
CONDITION_TAG="online_v4_N${TOTAL_SCANS}_I${INITIAL_RANDOM_SCANS}_P${CANDIDATE_POOL_SIZE}_${FISHER_OBJECTIVE}_init_t${INIT_T_TAG}_r${INIT_R_TAG}_${INIT_TRANSLATION_PERTURBATION}_${INIT_ROTATION_PERTURBATION}"
DATASET_DIR="${DATASET_ROOT}/${CONDITION_TAG}"
CONDITION_RESULT_ROOT="${RESULT_ROOT}/${CONDITION_TAG}"

mkdir -p "${DATASET_ROOT}" "${RESULT_ROOT}"

echo "============================================================"
echo "Fisher versus random pose-policy comparison"
echo "============================================================"
echo "Trials              : ${MAX_TRIALS}"
echo "Scans               : initial=${INITIAL_RANDOM_SCANS}, total=${TOTAL_SCANS}"
echo "Candidate pool      : ${CANDIDATE_POOL_SIZE}"
echo "Three-plane quota   : $((TOTAL_SCANS / 3)) scans/plane"
echo "Fisher objective    : ${FISHER_OBJECTIVE}"
echo "Profile design noise: ${FISHER_PROFILE_NOISE_STD_MM} mm"
echo "Fisher metric       : rotation=${FISHER_ROTATION_SCALE_DEG} deg, translation=${FISHER_TRANSLATION_SCALE_MM} mm"
echo "Acquisition noise   : sigma=${NOISE_STD_MM} mm, axis=${NOISE_AXIS}"
echo "Calibration re-noise: disabled (saved measured profiles are reused)"
echo "Initialization      : ${INIT_TRANSLATION_RANGE_MM} mm / ${INIT_ANGLE_RANGE_DEG} deg max"
echo "Perturbation model  : translation=${INIT_TRANSLATION_PERTURBATION}, rotation=${INIT_ROTATION_PERTURBATION}"
echo "Dataset             : ${DATASET_DIR}"
echo "Results             : ${CONDITION_RESULT_ROOT}"
echo "============================================================"

# ---------------------------------------------------------------------------
# 1) Generate the four paired measured online-acquisition collections
# ---------------------------------------------------------------------------

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

  echo "[generate] creating four policy/geometry collections..."
  PYTHONPATH=. python3 "${GENERATOR}" \
    --trials "${MAX_TRIALS}" \
    --seed "${GENERATION_SEED}" \
    --output-dir "${DATASET_DIR}" \
    --pose-selection-mode fisher_vs_random \
    --total-scans "${TOTAL_SCANS}" \
    --initial-random-scans "${INITIAL_RANDOM_SCANS}" \
    --candidate-pool-size "${CANDIDATE_POOL_SIZE}" \
    --fisher-objective "${FISHER_OBJECTIVE}" \
    --profile-points "${PROFILE_POINTS}" \
    --profile-half-width-mm "${PROFILE_HALF_WIDTH_MM}" \
    --tangent-range-mm "${TANGENT_RANGE_MM}" \
    --profile-depth-range-mm \
      "${PROFILE_DEPTH_MIN_MM}" "${PROFILE_DEPTH_MAX_MM}" \
    --view-tilt-range-deg \
      "${VIEW_TILT_MIN_DEG}" "${VIEW_TILT_MAX_DEG}" \
    --view-azimuth-range-deg \
      "${VIEW_AZIMUTH_MIN_DEG}" "${VIEW_AZIMUTH_MAX_DEG}" \
    --sensor-roll-range-deg \
      "${SENSOR_ROLL_MIN_DEG}" "${SENSOR_ROLL_MAX_DEG}" \
    --fisher-profile-noise-std-mm \
      "${FISHER_PROFILE_NOISE_STD_MM}" \
    --fisher-rotation-scale-deg \
      "${FISHER_ROTATION_SCALE_DEG}" \
    --fisher-translation-scale-mm \
      "${FISHER_TRANSLATION_SCALE_MM}" \
    --fisher-plane-normal-scale-deg \
      "${FISHER_PLANE_NORMAL_SCALE_DEG}" \
    --fisher-plane-offset-scale-mm \
      "${FISHER_PLANE_OFFSET_SCALE_MM}" \
    --selection-measurement-noise-std-mm "${NOISE_STD_MM}" \
    --selection-measurement-noise-axis "${NOISE_AXIS}" \
    --selection-measurement-seed "${CALIBRATION_SEED}" \
    --selection-initialization-seed "${CALIBRATION_SEED}" \
    --selection-initial-translation-range-mm \
      "${INIT_TRANSLATION_RANGE_MM}" \
    --selection-initial-angle-range-deg "${INIT_ANGLE_RANGE_DEG}" \
    --selection-initial-rotation-perturbation \
      "${INIT_ROTATION_PERTURBATION}" \
    --selection-initial-translation-perturbation \
      "${INIT_TRANSLATION_PERTURBATION}" \
    --selection-estimator-max-iterations \
      "${SELECTION_ESTIMATOR_MAX_ITER}" \
    --selection-estimator-tolerance "${SELECTION_ESTIMATOR_TOL}"
  DATASET_REBUILT=1
fi

if ! PYTHONPATH=. python3 "${VALIDATOR}" dataset \
  --manifest "${DATASET_DIR}/comparison_manifest.json" \
  --trials "${MAX_TRIALS}" \
  --seed "${GENERATION_SEED}" \
  --total-scans "${TOTAL_SCANS}" \
  --profile-points "${PROFILE_POINTS}" \
  --profile-half-width-mm "${PROFILE_HALF_WIDTH_MM}" \
  --tangent-range-mm "${TANGENT_RANGE_MM}" \
  --profile-depth-range-mm \
    "${PROFILE_DEPTH_MIN_MM}" "${PROFILE_DEPTH_MAX_MM}" \
  --view-tilt-range-deg \
    "${VIEW_TILT_MIN_DEG}" "${VIEW_TILT_MAX_DEG}" \
  --view-azimuth-range-deg \
    "${VIEW_AZIMUTH_MIN_DEG}" "${VIEW_AZIMUTH_MAX_DEG}" \
  --sensor-roll-range-deg \
    "${SENSOR_ROLL_MIN_DEG}" "${SENSOR_ROLL_MAX_DEG}" \
  --initial-random-scans "${INITIAL_RANDOM_SCANS}" \
  --candidate-pool-size "${CANDIDATE_POOL_SIZE}" \
  --fisher-objective "${FISHER_OBJECTIVE}" \
  --fisher-profile-noise-std-mm "${FISHER_PROFILE_NOISE_STD_MM}" \
  --fisher-rotation-scale-deg \
    "${FISHER_ROTATION_SCALE_DEG}" \
  --fisher-translation-scale-mm \
    "${FISHER_TRANSLATION_SCALE_MM}" \
  --fisher-plane-normal-scale-deg \
    "${FISHER_PLANE_NORMAL_SCALE_DEG}" \
  --fisher-plane-offset-scale-mm \
    "${FISHER_PLANE_OFFSET_SCALE_MM}" \
  --selection-measurement-noise-std-mm "${NOISE_STD_MM}" \
  --selection-measurement-noise-axis "${NOISE_AXIS}" \
  --selection-measurement-seed "${CALIBRATION_SEED}" \
  --selection-initialization-seed "${CALIBRATION_SEED}" \
  --selection-initial-translation-range-mm \
    "${INIT_TRANSLATION_RANGE_MM}" \
  --selection-initial-angle-range-deg "${INIT_ANGLE_RANGE_DEG}" \
  --selection-initial-rotation-perturbation \
    "${INIT_ROTATION_PERTURBATION}" \
  --selection-initial-translation-perturbation \
    "${INIT_TRANSLATION_PERTURBATION}" \
  --selection-estimator-max-iterations \
    "${SELECTION_ESTIMATOR_MAX_ITER}" \
  --selection-estimator-tolerance "${SELECTION_ESTIMATOR_TOL}"; then
  echo "Use FORCE_REGENERATE=1 to replace the incompatible dataset." >&2
  exit 1
fi

if [[ "${DATASET_REBUILT}" == "1" && "${FORCE_RERUN}" != "1" ]]; then
  echo "[cache] dataset was rebuilt; forcing dependent calibrations to rerun."
  FORCE_RERUN=1
fi

# ---------------------------------------------------------------------------
# 2) Run all four calibrations with identical stochastic settings
# ---------------------------------------------------------------------------

for method in "${METHODS[@]}"; do
  collection="$(collection_for_method "${method}")"
  result_dir="${CONDITION_RESULT_ROOT}/${method}"
  collection_digest_file="${result_dir}/input_collection_manifest.sha256"
  read -r collection_digest _ < <(
    sha256sum "${collection}/collection.json"
  )

  echo
  echo "------------------------------------------------------------"
  echo "Method: ${method}"
  echo "Collection: ${collection}"
  echo "------------------------------------------------------------"

  if [[ "${FORCE_RERUN}" == "1" && -e "${result_dir}" ]]; then
    echo "[${method}] removing existing result: ${result_dir}"
    rm -rf "${result_dir}"
  fi

  if [[ -f "${result_dir}/summary.json" ]]; then
    if [[ ! -f "${collection_digest_file}" ]] \
      || [[ "$(<"${collection_digest_file}")" != "${collection_digest}" ]]; then
      echo "ERROR: cached ${method} result does not match the current dataset." >&2
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
      --noise-axis "${NOISE_AXIS}" \
      --noise-std-mm 0 \
      --init-mode carlson \
      --init-translation-range-mm "${INIT_TRANSLATION_RANGE_MM}" \
      --init-angle-range-deg "${INIT_ANGLE_RANGE_DEG}" \
      --init-rotation-perturbation "${INIT_ROTATION_PERTURBATION}" \
      --init-translation-perturbation "${INIT_TRANSLATION_PERTURBATION}" \
      --max-iter "${MAX_ITER}" \
      --tol "${TOL}"
    printf "%s\n" "${collection_digest}" \
      > "${collection_digest_file}"
  fi

  if ! PYTHONPATH=. python3 "${VALIDATOR}" result \
    --summary "${result_dir}/summary.json" \
    --collection "${collection}" \
    --trials "${MAX_TRIALS}" \
    --seed "${CALIBRATION_SEED}" \
    --noise-axis "${NOISE_AXIS}" \
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

# ---------------------------------------------------------------------------
# 3) Plot final errors, rates, and paired Fisher/random improvement
# ---------------------------------------------------------------------------

if [[ "${MAKE_PLOTS}" == "1" ]]; then
  echo
  echo "[plot] creating Fisher/random policy comparison plots..."
  PLOT_ARGS_COMMON=(
    --result-root "${CONDITION_RESULT_ROOT}"
    --dpi "${PLOT_DPI}"
  )
  if [[ "${PLOT_SUCCESS_ONLY}" == "1" ]]; then
    PLOT_ARGS_COMMON+=(--success-only)
  fi

  PLOT_ARGS_HIDE=(
    "${PLOT_ARGS_COMMON[@]}"
    --output-dir "${CONDITION_RESULT_ROOT}/plots_hide_outliers"
    --hide-outliers
    --no-log-errors
  )
  PLOT_ARGS_SHOW=(
    "${PLOT_ARGS_COMMON[@]}"
    --output-dir "${CONDITION_RESULT_ROOT}/plots_show_outliers"
    --log-errors
  )

  MPLBACKEND="${MPLBACKEND:-Agg}" PYTHONPATH=. \
    python3 "${PLOTTER}" "${PLOT_ARGS_HIDE[@]}"
  MPLBACKEND="${MPLBACKEND:-Agg}" PYTHONPATH=. \
    python3 "${PLOTTER}" "${PLOT_ARGS_SHOW[@]}"
fi

echo
echo "============================================================"
echo "Fisher/random policy comparison complete."
echo "Dataset manifest: ${DATASET_DIR}/comparison_manifest.json"
echo "Calibration results: ${CONDITION_RESULT_ROOT}"
if [[ "${MAKE_PLOTS}" == "1" ]]; then
  if [[ "${PLOT_SUCCESS_ONLY}" == "1" ]]; then
    echo "Plots: ${CONDITION_RESULT_ROOT}/plots_success_only"
  else
    echo "Plots: ${CONDITION_RESULT_ROOT}/plots"
  fi
fi
echo "============================================================"
