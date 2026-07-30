#!/usr/bin/env bash
set -euo pipefail

# Initialization-robustness comparison at fixed profile noise.
#
#   Single-plane: N scans on one plane
#   Three-plane : N/3 scans per plane, total N scans
#
# The same ideal datasets and calibration seed are reused for every
# initialization level.
#
# The default sweep uses exact translation-norm and SO(3) geodesic bounds.
# Other main experiments use the medium 100 mm / 15 deg level.
#
# Run:
#   bash main/run_fair_plane_initialization_levels.sh
#
# Quick test:
#   MAX_TRIALS=3 INIT_LEVELS="medium:100:15" \
#   bash main/run_fair_plane_initialization_levels.sh

TOTAL_SCANS="${TOTAL_SCANS:-108}"
NOISE_STD_MM="${NOISE_STD_MM:-0.20}"
NOISE_AXIS="${NOISE_AXIS:-xz}"

MAX_TRIALS="${MAX_TRIALS:-100}"

GENERATION_SEED="${GENERATION_SEED:-17}"
CALIBRATION_SEED="${CALIBRATION_SEED:-1701}"

PROFILE_POINTS="${PROFILE_POINTS:-100}"
PROFILE_HALF_WIDTH_MM="${PROFILE_HALF_WIDTH_MM:-25}"
TANGENT_RANGE_MM="${TANGENT_RANGE_MM:-100}"

# Format: LABEL:MAX_TRANSLATION_NORM_MM:MAX_ROTATION_ANGLE_DEG
INIT_LEVELS="${INIT_LEVELS:-easy:25:5 medium:100:15 hard:200:30}"
INIT_ROTATION_PERTURBATION="${INIT_ROTATION_PERTURBATION:-axis_angle}"
INIT_TRANSLATION_PERTURBATION="${INIT_TRANSLATION_PERTURBATION:-direction_norm}"

MAX_ITER="${MAX_ITER:-3000}"
TOL="${TOL:-1e-5}"

DATASET_ROOT="${DATASET_ROOT:-dataset/fair_plane_initialization_shared_global}"
INIT_RESULT_TAG="${INIT_TRANSLATION_PERTURBATION}_${INIT_ROTATION_PERTURBATION}"
RESULT_ROOT="${RESULT_ROOT:-results/fair_plane_initialization_shared_global_${INIT_RESULT_TAG}}"

FORCE_REGENERATE="${FORCE_REGENERATE:-0}"
FORCE_RERUN="${FORCE_RERUN:-0}"
MAKE_PLOTS="${MAKE_PLOTS:-1}"
PLOT_SUCCESS_ONLY="${PLOT_SUCCESS_ONLY:-0}"
PLOT_DPI="${PLOT_DPI:-300}"

GENERATOR="main/generate_independent_random_plane_comparison.py"
CALIBRATOR="main/calibrate.py"
PLOTTER="main/make_plot/plot_initialization_robustness.py"
VALIDATOR="main/validate_experiment_artifact.py"

if [[ ! -f "${GENERATOR}" ]]; then
  echo "ERROR: generator not found: ${GENERATOR}" >&2
  exit 1
fi

if [[ ! -f "${CALIBRATOR}" ]]; then
  echo "ERROR: calibrator not found: ${CALIBRATOR}" >&2
  exit 1
fi

if [[ ! -f "${VALIDATOR}" ]]; then
  echo "ERROR: artifact validator not found: ${VALIDATOR}" >&2
  exit 1
fi

if [[ "${MAKE_PLOTS}" != "0" && "${MAKE_PLOTS}" != "1" ]]; then
  echo "ERROR: MAKE_PLOTS must be 0 or 1." >&2
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

if [[ "${PLOT_SUCCESS_ONLY}" != "0" && "${PLOT_SUCCESS_ONLY}" != "1" ]]; then
  echo "ERROR: PLOT_SUCCESS_ONLY must be 0 or 1." >&2
  exit 1
fi

if ! [[ "${PLOT_DPI}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: PLOT_DPI must be a positive integer." >&2
  exit 1
fi

if [[ "${MAKE_PLOTS}" == "1" && ! -f "${PLOTTER}" ]]; then
  echo "ERROR: plotter not found: ${PLOTTER}" >&2
  exit 1
fi

if [[ "${MAKE_PLOTS}" == "1" ]] \
  && ! MPLBACKEND="${MPLBACKEND:-Agg}" PYTHONPATH=. \
    python3 "${PLOTTER}" --help >/dev/null; then
  echo "ERROR: plot dependencies are unavailable; install requirements.txt or set MAKE_PLOTS=0." >&2
  exit 1
fi

if ! [[ "${TOTAL_SCANS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: TOTAL_SCANS must be a positive integer." >&2
  exit 1
fi

if (( TOTAL_SCANS % 3 != 0 )); then
  echo "ERROR: TOTAL_SCANS must be divisible by 3." >&2
  exit 1
fi

if ! [[ "${MAX_TRIALS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: MAX_TRIALS must be a positive integer." >&2
  exit 1
fi

DATASET_DIR="${DATASET_ROOT}/N${TOTAL_SCANS}"
SINGLE_COLLECTION="${DATASET_DIR}/single_plane"
THREE_COLLECTION="${DATASET_DIR}/three_plane"

mkdir -p "${DATASET_ROOT}" "${RESULT_ROOT}"

echo "============================================================"
echo "Initialization robustness comparison"
echo "============================================================"
echo "Total scans        : ${TOTAL_SCANS}"
echo "Three-plane split  : $((TOTAL_SCANS / 3)) scans/plane × 3"
echo "Fixed noise [mm]   : ${NOISE_STD_MM}"
echo "Noise axis         : ${NOISE_AXIS}"
echo "Trials per level   : ${MAX_TRIALS}"
echo "Initialization     : ${INIT_LEVELS}"
echo "Perturbation model : translation=${INIT_TRANSLATION_PERTURBATION}, rotation=${INIT_ROTATION_PERTURBATION}"
echo "============================================================"

# ---------------------------------------------------------------------------
# 1) Generate one shared ideal dataset
# ---------------------------------------------------------------------------

if [[ "${FORCE_REGENERATE}" == "1" && -e "${DATASET_DIR}" ]]; then
  echo "[generate] removing existing dataset: ${DATASET_DIR}"
  rm -rf "${DATASET_DIR}"
fi

if [[ -f "${DATASET_DIR}/comparison_manifest.json" \
      && -f "${SINGLE_COLLECTION}/collection.json" \
      && -f "${THREE_COLLECTION}/collection.json" ]]; then
  echo "[generate] existing shared dataset found; skipping."
else
  if [[ -e "${DATASET_DIR}" ]]; then
    echo "ERROR: incomplete dataset directory exists: ${DATASET_DIR}" >&2
    echo "Use FORCE_REGENERATE=1 to recreate it." >&2
    exit 1
  fi

  echo "[generate] creating shared ideal dataset..."
  PYTHONPATH=. python3 "${GENERATOR}" \
    --trials "${MAX_TRIALS}" \
    --seed "${GENERATION_SEED}" \
    --output-dir "${DATASET_DIR}" \
    --pose-selection-mode random_only \
    --total-scans "${TOTAL_SCANS}" \
    --profile-points "${PROFILE_POINTS}" \
    --profile-half-width-mm "${PROFILE_HALF_WIDTH_MM}" \
    --tangent-range-mm "${TANGENT_RANGE_MM}" \
    --profile-depth-range-mm 60 150 \
    --view-tilt-range-deg 48 62 \
    --view-azimuth-range-deg 35 55 \
    --sensor-roll-range-deg -20 20
fi

if ! PYTHONPATH=. python3 "${VALIDATOR}" dataset \
  --manifest "${DATASET_DIR}/comparison_manifest.json" \
  --trials "${MAX_TRIALS}" \
  --seed "${GENERATION_SEED}" \
  --total-scans "${TOTAL_SCANS}" \
  --profile-points "${PROFILE_POINTS}" \
  --profile-half-width-mm "${PROFILE_HALF_WIDTH_MM}" \
  --tangent-range-mm "${TANGENT_RANGE_MM}" \
  --profile-depth-range-mm 60 150 \
  --view-tilt-range-deg 48 62 \
  --view-azimuth-range-deg 35 55 \
  --sensor-roll-range-deg -20 20; then
  echo "Use FORCE_REGENERATE=1 to replace the incompatible dataset." >&2
  exit 1
fi

# ---------------------------------------------------------------------------
# 2) Sweep initialization levels at fixed noise
# ---------------------------------------------------------------------------

INIT_CONDITION_NAMES=()

for LEVEL_SPEC in ${INIT_LEVELS}; do
  IFS=":" read -r LEVEL_NAME INIT_T INIT_R <<< "${LEVEL_SPEC}"

  if [[ -z "${LEVEL_NAME}" || -z "${INIT_T}" || -z "${INIT_R}" ]]; then
    echo "ERROR: invalid INIT_LEVELS entry: ${LEVEL_SPEC}" >&2
    echo "Expected LABEL:TRANSLATION_MM:ANGLE_DEG" >&2
    exit 1
  fi

  if ! [[ "${LEVEL_NAME}" =~ ^[A-Za-z0-9_-]+$ ]]; then
    echo "ERROR: initialization label must contain only letters, digits, _ or -: ${LEVEL_NAME}" >&2
    exit 1
  fi
  if ! [[ "${INIT_T}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "ERROR: translation range must be a non-negative decimal: ${INIT_T}" >&2
    exit 1
  fi
  if ! [[ "${INIT_R}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "ERROR: rotation range must be a non-negative decimal: ${INIT_R}" >&2
    exit 1
  fi

  INIT_T_TAG="${INIT_T//./p}"
  INIT_R_TAG="${INIT_R//./p}"
  CONDITION_NAME="${LEVEL_NAME}_t${INIT_T_TAG}_r${INIT_R_TAG}"
  INIT_CONDITION_NAMES+=("${CONDITION_NAME}")

  CONDITION_ROOT="${RESULT_ROOT}/${CONDITION_NAME}"
  SINGLE_RESULT="${CONDITION_ROOT}/single_plane"
  THREE_RESULT="${CONDITION_ROOT}/three_plane"

  echo
  echo "------------------------------------------------------------"
  echo "Initialization level=${LEVEL_NAME}"
  echo "Translation bound  = ${INIT_T} mm (${INIT_TRANSLATION_PERTURBATION})"
  echo "Rotation bound     = ${INIT_R} deg (${INIT_ROTATION_PERTURBATION})"
  echo "Noise sigma        = ${NOISE_STD_MM} mm"
  echo "------------------------------------------------------------"

  if [[ "${FORCE_RERUN}" == "1" && -e "${CONDITION_ROOT}" ]]; then
    echo "[rerun] removing existing result: ${CONDITION_ROOT}"
    rm -rf "${CONDITION_ROOT}"
  fi

  if [[ -f "${SINGLE_RESULT}/summary.json" ]]; then
    echo "[single] completed result found; skipping."
  else
    echo "[single] calibrating..."
    PYTHONPATH=. python3 "${CALIBRATOR}" \
      --collection "${SINGLE_COLLECTION}" \
      --output-dir "${SINGLE_RESULT}" \
      --mode iterative \
      --seed "${CALIBRATION_SEED}" \
      --noise-axis "${NOISE_AXIS}" \
      --noise-std-mm "${NOISE_STD_MM}" \
      --init-mode carlson \
      --init-translation-range-mm "${INIT_T}" \
      --init-angle-range-deg "${INIT_R}" \
      --init-rotation-perturbation "${INIT_ROTATION_PERTURBATION}" \
      --init-translation-perturbation "${INIT_TRANSLATION_PERTURBATION}" \
      --max-iter "${MAX_ITER}" \
      --tol "${TOL}"
  fi

  if ! PYTHONPATH=. python3 "${VALIDATOR}" result \
    --summary "${SINGLE_RESULT}/summary.json" \
    --collection "${SINGLE_COLLECTION}" \
    --trials "${MAX_TRIALS}" \
    --seed "${CALIBRATION_SEED}" \
    --noise-axis "${NOISE_AXIS}" \
    --noise-std-mm "${NOISE_STD_MM}" \
    --init-translation-range-mm "${INIT_T}" \
    --init-angle-range-deg "${INIT_R}" \
    --init-rotation-perturbation "${INIT_ROTATION_PERTURBATION}" \
    --init-translation-perturbation "${INIT_TRANSLATION_PERTURBATION}" \
    --max-iter "${MAX_ITER}" \
    --tol "${TOL}"; then
    echo "Use FORCE_RERUN=1 to replace the incompatible single-plane result." >&2
    exit 1
  fi

  if [[ -f "${THREE_RESULT}/summary.json" ]]; then
    echo "[three] completed result found; skipping."
  else
    echo "[three] calibrating..."
    PYTHONPATH=. python3 "${CALIBRATOR}" \
      --collection "${THREE_COLLECTION}" \
      --output-dir "${THREE_RESULT}" \
      --mode iterative \
      --seed "${CALIBRATION_SEED}" \
      --noise-axis "${NOISE_AXIS}" \
      --noise-std-mm "${NOISE_STD_MM}" \
      --init-mode carlson \
      --init-translation-range-mm "${INIT_T}" \
      --init-angle-range-deg "${INIT_R}" \
      --init-rotation-perturbation "${INIT_ROTATION_PERTURBATION}" \
      --init-translation-perturbation "${INIT_TRANSLATION_PERTURBATION}" \
      --max-iter "${MAX_ITER}" \
      --tol "${TOL}"
  fi

  if ! PYTHONPATH=. python3 "${VALIDATOR}" result \
    --summary "${THREE_RESULT}/summary.json" \
    --collection "${THREE_COLLECTION}" \
    --trials "${MAX_TRIALS}" \
    --seed "${CALIBRATION_SEED}" \
    --noise-axis "${NOISE_AXIS}" \
    --noise-std-mm "${NOISE_STD_MM}" \
    --init-translation-range-mm "${INIT_T}" \
    --init-angle-range-deg "${INIT_R}" \
    --init-rotation-perturbation "${INIT_ROTATION_PERTURBATION}" \
    --init-translation-perturbation "${INIT_TRANSLATION_PERTURBATION}" \
    --max-iter "${MAX_ITER}" \
    --tol "${TOL}"; then
    echo "Use FORCE_RERUN=1 to replace the incompatible three-plane result." >&2
    exit 1
  fi

  echo "[done] ${LEVEL_NAME}"
done

if [[ "${MAKE_PLOTS}" == "1" ]]; then
  echo
  echo "[plot] creating initialization-robustness plots..."
  PLOT_ARGS_COMMON=(
    --result-root "${RESULT_ROOT}"
    --conditions "${INIT_CONDITION_NAMES[@]}"
    --dpi "${PLOT_DPI}"
  )
  if [[ "${PLOT_SUCCESS_ONLY}" == "1" ]]; then
    PLOT_ARGS_COMMON+=(--success-only)
  fi

  PLOT_ARGS_HIDE=(
    "${PLOT_ARGS_COMMON[@]}"
    --output-dir "${RESULT_ROOT}/plots_hide_outliers"
    --hide-outliers
    --no-log-translation
    --no-log-rotation
    --no-log-iterations
  )
  PLOT_ARGS_SHOW=(
    "${PLOT_ARGS_COMMON[@]}"
    --output-dir "${RESULT_ROOT}/plots_show_outliers"
    --log-translation
    --log-rotation
    --log-iterations
  )

  MPLBACKEND="${MPLBACKEND:-Agg}" PYTHONPATH=. \
    python3 "${PLOTTER}" "${PLOT_ARGS_HIDE[@]}"
  MPLBACKEND="${MPLBACKEND:-Agg}" PYTHONPATH=. \
    python3 "${PLOTTER}" "${PLOT_ARGS_SHOW[@]}"
fi

echo
echo "============================================================"
echo "All initialization-level experiments completed."
echo "Results: ${RESULT_ROOT}"
echo "============================================================"
