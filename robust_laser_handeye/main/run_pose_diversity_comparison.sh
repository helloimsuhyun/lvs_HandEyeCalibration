#!/usr/bin/env bash
set -euo pipefail

# Shared-global pose-diversity comparison for single-plane vs three-plane.
#
# Fixed:
#   - total scans
#   - profile noise
#   - initialization range
#   - the exact robot pose stack paired between single and three plane
#   - GT hand-eye generation seed
#   - calibration seed
#
# Varied:
#   - view tilt range
#   - view azimuth range
#   - sensor roll range
#
# Angles are expressed in the shared trihedral target frame.  The nominal
# direction that sees all three orthogonal planes is tilt=54.7356, azimuth=45.
#
# Default pose levels:
#   restricted : tilt 48–62 deg, azimuth 35–55 deg, roll ±20 deg
#   moderate   : tilt 35–75 deg, azimuth 20–70 deg, roll ±90 deg
#   wide       : tilt 25–82 deg, azimuth 10–80 deg, roll ±180 deg
#
# "wide" is a negative-control regime: a sufficiently diverse single-plane
# trajectory may close the observability gap, so three-plane is not assumed to
# win there.
#
# Run from the robust_laser_handeye repository root.
#
# Quick test:
#   MAX_TRIALS=3 \
#   bash main/run_pose_diversity_comparison.sh
#
# Final run:
#   MAX_TRIALS=100 \
#   bash main/run_pose_diversity_comparison.sh
#
# Regenerate and rerun:
#   FORCE_REGENERATE=1 FORCE_RERUN=1 \
#   bash main/run_pose_diversity_comparison.sh

# ---------------------------------------------------------------------------
# Fixed experiment settings
# ---------------------------------------------------------------------------

TOTAL_SCANS="${TOTAL_SCANS:-108}"
MAX_TRIALS="${MAX_TRIALS:-100}"

NOISE_STD_MM="${NOISE_STD_MM:-0.20}"
NOISE_AXIS="${NOISE_AXIS:-xz}"

GENERATION_SEED="${GENERATION_SEED:-17}"
CALIBRATION_SEED="${CALIBRATION_SEED:-1701}"

PROFILE_POINTS="${PROFILE_POINTS:-100}"
PROFILE_HALF_WIDTH_MM="${PROFILE_HALF_WIDTH_MM:-25}"
TANGENT_RANGE_MM="${TANGENT_RANGE_MM:-100}"

# Standardized across the main experiment runners.
INIT_TRANSLATION_RANGE_MM="${INIT_TRANSLATION_RANGE_MM:-100}"
INIT_ANGLE_RANGE_DEG="${INIT_ANGLE_RANGE_DEG:-15}"
INIT_ROTATION_PERTURBATION="${INIT_ROTATION_PERTURBATION:-axis_angle}"
INIT_TRANSLATION_PERTURBATION="${INIT_TRANSLATION_PERTURBATION:-direction_norm}"

MAX_ITER="${MAX_ITER:-3000}"
TOL="${TOL:-1e-5}"
CALIBRATION_MODE="${CALIBRATION_MODE:-iterative}"
NONLINEAR_LOSS="${NONLINEAR_LOSS:-linear}"
NONLINEAR_F_SCALE_MM="${NONLINEAR_F_SCALE_MM:-1.0}"
NONLINEAR_MAX_NFEV="${NONLINEAR_MAX_NFEV:-300}"
NONLINEAR_FTOL="${NONLINEAR_FTOL:-1e-10}"
NONLINEAR_XTOL="${NONLINEAR_XTOL:-1e-10}"
NONLINEAR_GTOL="${NONLINEAR_GTOL:-1e-10}"
NONLINEAR_ARGS=(
  --nonlinear-loss "${NONLINEAR_LOSS}"
  --nonlinear-f-scale-mm "${NONLINEAR_F_SCALE_MM}"
  --nonlinear-max-nfev "${NONLINEAR_MAX_NFEV}"
  --nonlinear-ftol "${NONLINEAR_FTOL}"
  --nonlinear-xtol "${NONLINEAR_XTOL}"
  --nonlinear-gtol "${NONLINEAR_GTOL}"
)

DATASET_ROOT="${DATASET_ROOT:-dataset/fair_plane_global_pose_diversity}"
INIT_T_TAG="${INIT_TRANSLATION_RANGE_MM//./p}"
INIT_R_TAG="${INIT_ANGLE_RANGE_DEG//./p}"
INIT_RESULT_TAG="init_t${INIT_T_TAG}_r${INIT_R_TAG}_${INIT_TRANSLATION_PERTURBATION}_${INIT_ROTATION_PERTURBATION}"
RESULT_ROOT="${RESULT_ROOT:-results/fair_plane_global_pose_diversity_${INIT_RESULT_TAG}}"

FORCE_REGENERATE="${FORCE_REGENERATE:-0}"
FORCE_RERUN="${FORCE_RERUN:-0}"
MAKE_PLOTS="${MAKE_PLOTS:-1}"
MAKE_HTML_POSE_PLOTS="${MAKE_HTML_POSE_PLOTS:-1}"
PLOT_SUCCESS_ONLY="${PLOT_SUCCESS_ONLY:-0}"
PLOT_DPI="${PLOT_DPI:-300}"
POSE_VIS_TRIAL_INDEX="${POSE_VIS_TRIAL_INDEX:-0}"
POSE_VIS_MAX_ORIENTATION_AXES="${POSE_VIS_MAX_ORIENTATION_AXES:-18}"

GENERATOR="main/generate_independent_random_plane_comparison.py"
CALIBRATOR="main/calibrate.py"
PLOTTER="main/make_plot/plot_pose_diversity_comparison.py"
HTML_POSE_PLOTTER="main/make_plot/plot_pose_diversity_condition_html.py"
VALIDATOR="main/validate_experiment_artifact.py"

# Format:
#   LABEL:TILT_MIN:TILT_MAX:AZIMUTH_MIN:AZIMUTH_MAX:ROLL_MIN:ROLL_MAX
POSE_LEVELS="${POSE_LEVELS:-\
restricted:48:62:35:55:-20:20 \
moderate:35:75:20:70:-90:90 \
wide:25:82:10:80:-180:180}"

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

if [[ ! -f "${GENERATOR}" ]]; then
  echo "ERROR: generator not found: ${GENERATOR}" >&2
  echo "Run this script from the repository root." >&2
  exit 1
fi

if [[ ! -f "${CALIBRATOR}" ]]; then
  echo "ERROR: calibrator not found: ${CALIBRATOR}" >&2
  echo "Run this script from the repository root." >&2
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

case "${CALIBRATION_MODE}" in
  iterative|iterative_refit_nonlinear|iterative_joint_nonlinear) ;;
  *)
    echo "ERROR: unsupported CALIBRATION_MODE: ${CALIBRATION_MODE}" >&2
    exit 1
    ;;
esac

if [[ "${MAKE_HTML_POSE_PLOTS}" != "0" \
      && "${MAKE_HTML_POSE_PLOTS}" != "1" ]]; then
  echo "ERROR: MAKE_HTML_POSE_PLOTS must be 0 or 1." >&2
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

if [[ "${MAKE_HTML_POSE_PLOTS}" == "1" \
      && ! -f "${HTML_POSE_PLOTTER}" ]]; then
  echo "ERROR: HTML pose plotter not found: ${HTML_POSE_PLOTTER}" >&2
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
if ! [[ "${POSE_VIS_TRIAL_INDEX}" =~ ^[0-9]+$ ]] \
  || (( POSE_VIS_TRIAL_INDEX >= MAX_TRIALS )); then
  echo "ERROR: POSE_VIS_TRIAL_INDEX must lie in [0, MAX_TRIALS)." >&2
  exit 1
fi
if ! [[ "${POSE_VIS_MAX_ORIENTATION_AXES}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: POSE_VIS_MAX_ORIENTATION_AXES must be positive." >&2
  exit 1
fi

mkdir -p "${DATASET_ROOT}" "${RESULT_ROOT}"

echo "============================================================"
echo "Pose-diversity comparison"
echo "============================================================"
echo "Total scans        : ${TOTAL_SCANS}"
echo "Three-plane split  : $((TOTAL_SCANS / 3)) scans/plane × 3"
echo "Trials per setting : ${MAX_TRIALS}"
echo "Noise              : sigma=${NOISE_STD_MM} mm, axis=${NOISE_AXIS}"
echo "Initialization     : ${INIT_TRANSLATION_RANGE_MM} mm / ${INIT_ANGLE_RANGE_DEG} deg max"
echo "Perturbation model : translation=${INIT_TRANSLATION_PERTURBATION}, rotation=${INIT_ROTATION_PERTURBATION}"
echo "Pose levels        : ${POSE_LEVELS}"
echo "Dataset root       : ${DATASET_ROOT}"
echo "Result root        : ${RESULT_ROOT}"
echo "============================================================"

# ---------------------------------------------------------------------------
# Run each pose-diversity condition
# ---------------------------------------------------------------------------

POSE_LEVEL_NAMES=()

for LEVEL_SPEC in ${POSE_LEVELS}; do
  IFS=":" read -r \
    LEVEL_NAME \
    TILT_MIN TILT_MAX \
    AZIMUTH_MIN AZIMUTH_MAX \
    ROLL_MIN ROLL_MAX \
    <<< "${LEVEL_SPEC}"

  if [[ -z "${LEVEL_NAME}" \
        || -z "${TILT_MIN}" || -z "${TILT_MAX}" \
        || -z "${AZIMUTH_MIN}" || -z "${AZIMUTH_MAX}" \
        || -z "${ROLL_MIN}" || -z "${ROLL_MAX}" ]]; then
    echo "ERROR: invalid POSE_LEVELS entry: ${LEVEL_SPEC}" >&2
    echo "Expected:" >&2
    echo "  LABEL:TILT_MIN:TILT_MAX:AZ_MIN:AZ_MAX:ROLL_MIN:ROLL_MAX" >&2
    exit 1
  fi

  if ! [[ "${LEVEL_NAME}" =~ ^[A-Za-z0-9_-]+$ ]]; then
    echo "ERROR: pose level label must contain only letters, digits, _ or -: ${LEVEL_NAME}" >&2
    exit 1
  fi

  POSE_LEVEL_NAMES+=("${LEVEL_NAME}")

  DATASET_DIR="${DATASET_ROOT}/${LEVEL_NAME}_N${TOTAL_SCANS}"
  SINGLE_COLLECTION="${DATASET_DIR}/single_plane"
  THREE_COLLECTION="${DATASET_DIR}/three_plane"

  CONDITION_RESULT_ROOT="${RESULT_ROOT}/${LEVEL_NAME}"
  SINGLE_RESULT="${CONDITION_RESULT_ROOT}/single_plane"
  THREE_RESULT="${CONDITION_RESULT_ROOT}/three_plane"

  echo
  echo "------------------------------------------------------------"
  echo "Pose level       : ${LEVEL_NAME}"
  echo "Tilt range       : ${TILT_MIN} to ${TILT_MAX} deg"
  echo "Azimuth range    : ${AZIMUTH_MIN} to ${AZIMUTH_MAX} deg"
  echo "Sensor-roll range: ${ROLL_MIN} to ${ROLL_MAX} deg"
  echo "------------------------------------------------------------"

  # -------------------------------------------------------------------------
  # 1) Generate ideal paired datasets
  # -------------------------------------------------------------------------

  if [[ "${FORCE_REGENERATE}" == "1" && -e "${DATASET_DIR}" ]]; then
    echo "[generate] removing existing dataset: ${DATASET_DIR}"
    rm -rf "${DATASET_DIR}"
  fi

  if [[ -f "${DATASET_DIR}/comparison_manifest.json" \
        && -f "${SINGLE_COLLECTION}/collection.json" \
        && -f "${THREE_COLLECTION}/collection.json" ]]; then
    echo "[generate] existing dataset found; skipping."
  else
    if [[ -e "${DATASET_DIR}" ]]; then
      echo "ERROR: incomplete dataset directory exists: ${DATASET_DIR}" >&2
      echo "Use FORCE_REGENERATE=1 to recreate it." >&2
      exit 1
    fi

    echo "[generate] creating ${LEVEL_NAME} pose-diversity dataset..."
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
      --view-tilt-range-deg "${TILT_MIN}" "${TILT_MAX}" \
      --view-azimuth-range-deg "${AZIMUTH_MIN}" "${AZIMUTH_MAX}" \
      --sensor-roll-range-deg "${ROLL_MIN}" "${ROLL_MAX}"
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
    --view-tilt-range-deg "${TILT_MIN}" "${TILT_MAX}" \
    --view-azimuth-range-deg "${AZIMUTH_MIN}" "${AZIMUTH_MAX}" \
    --sensor-roll-range-deg "${ROLL_MIN}" "${ROLL_MAX}"; then
    echo "Use FORCE_REGENERATE=1 to replace the incompatible dataset." >&2
    exit 1
  fi

  # -------------------------------------------------------------------------
  # 2) Single-plane calibration
  # -------------------------------------------------------------------------

  if [[ "${FORCE_RERUN}" == "1" && -e "${SINGLE_RESULT}" ]]; then
    echo "[single] removing existing result: ${SINGLE_RESULT}"
    rm -rf "${SINGLE_RESULT}"
  fi

  if [[ -f "${SINGLE_RESULT}/summary.json" ]]; then
    echo "[single] completed result found; skipping."
  else
    echo "[single] calibrating..."
    PYTHONPATH=. python3 "${CALIBRATOR}" \
      --collection "${SINGLE_COLLECTION}" \
      --output-dir "${SINGLE_RESULT}" \
      --mode "${CALIBRATION_MODE}" \
      "${NONLINEAR_ARGS[@]}" \
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
    --mode "${CALIBRATION_MODE}" \
    "${NONLINEAR_ARGS[@]}" \
    --summary "${SINGLE_RESULT}/summary.json" \
    --collection "${SINGLE_COLLECTION}" \
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
    echo "Use FORCE_RERUN=1 to replace the incompatible single-plane result." >&2
    exit 1
  fi

  # -------------------------------------------------------------------------
  # 3) Three-plane calibration
  # -------------------------------------------------------------------------

  if [[ "${FORCE_RERUN}" == "1" && -e "${THREE_RESULT}" ]]; then
    echo "[three] removing existing result: ${THREE_RESULT}"
    rm -rf "${THREE_RESULT}"
  fi

  if [[ -f "${THREE_RESULT}/summary.json" ]]; then
    echo "[three] completed result found; skipping."
  else
    echo "[three] calibrating..."
    PYTHONPATH=. python3 "${CALIBRATOR}" \
      --collection "${THREE_COLLECTION}" \
      --output-dir "${THREE_RESULT}" \
      --mode "${CALIBRATION_MODE}" \
      "${NONLINEAR_ARGS[@]}" \
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
    --mode "${CALIBRATION_MODE}" \
    "${NONLINEAR_ARGS[@]}" \
    --summary "${THREE_RESULT}/summary.json" \
    --collection "${THREE_COLLECTION}" \
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
    echo "Use FORCE_RERUN=1 to replace the incompatible three-plane result." >&2
    exit 1
  fi

  echo "[done] ${LEVEL_NAME}"
  echo "  single: ${SINGLE_RESULT}/trials.csv"
  echo "  three : ${THREE_RESULT}/trials.csv"
done

if [[ "${MAKE_PLOTS}" == "1" ]]; then
  echo
  echo "[plot] creating pose-diversity plots..."
  PLOT_ARGS_COMMON=(
    --result-root "${RESULT_ROOT}"
    --levels "${POSE_LEVEL_NAMES[@]}"
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
    --log-iterations
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

if [[ "${MAKE_HTML_POSE_PLOTS}" == "1" ]]; then
  echo
  echo "[plot] creating per-condition interactive 3D HTML..."
  PYTHONPATH=. python3 "${HTML_POSE_PLOTTER}" \
    --dataset-root "${DATASET_ROOT}" \
    --output-dir "${RESULT_ROOT}/pose_visualization_html" \
    --levels "${POSE_LEVEL_NAMES[@]}" \
    --total-scans "${TOTAL_SCANS}" \
    --trial-index "${POSE_VIS_TRIAL_INDEX}" \
    --max-orientation-axes "${POSE_VIS_MAX_ORIENTATION_AXES}" \
    --no-full-azimuth-presentation \
    --plotly-js inline
fi

echo
echo "============================================================"
echo "All pose-diversity experiments completed."
echo "Results: ${RESULT_ROOT}"
if [[ "${MAKE_HTML_POSE_PLOTS}" == "1" ]]; then
  echo "3D HTML: ${RESULT_ROOT}/pose_visualization_html/index.html"
fi
echo "============================================================"
