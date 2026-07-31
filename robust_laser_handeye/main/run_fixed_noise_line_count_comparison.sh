#!/usr/bin/env bash
set -euo pipefail

# Fixed-noise line-count comparison:
#   Single-plane: N scans on one plane
#   Three-plane : N/3 scans per plane, total N scans
#
# Run this script from the robust_laser_handeye repository root.
#
# Quick test:
#   MAX_TRIALS=3 SCAN_COUNTS="81" \
#   bash main/run_fixed_noise_line_count_comparison.sh
#
# Recommended run:
#   bash main/run_fixed_noise_line_count_comparison.sh
#
# Final paper run:
#   MAX_TRIALS=100 \
#   bash main/run_fixed_noise_line_count_comparison.sh
#
# Regenerate existing datasets/results:
#   FORCE_REGENERATE=1 FORCE_RERUN=1 \
#   bash main/run_fixed_noise_line_count_comparison.sh

# ---------------------------------------------------------------------------
# Experiment settings
# ---------------------------------------------------------------------------

# Five representative totals around the 108-scan baseline. Every value must
# be divisible by 3 so the three-plane branch remains balanced.
SCAN_COUNTS="${SCAN_COUNTS:-27 54 108 162 243}"

# Practical default; override with MAX_TRIALS=100 for the final experiment.
MAX_TRIALS="${MAX_TRIALS:-100}"

# Gaussian noise applied independently to every profile point.
NOISE_STD_MM="${NOISE_STD_MM:-0.20}"
NOISE_AXIS="${NOISE_AXIS:-xz}"

GENERATION_SEED="${GENERATION_SEED:-17}"
CALIBRATION_SEED="${CALIBRATION_SEED:-1701}"

PROFILE_POINTS="${PROFILE_POINTS:-100}"
PROFILE_HALF_WIDTH_MM="${PROFILE_HALF_WIDTH_MM:-25}"
TANGENT_RANGE_MM="${TANGENT_RANGE_MM:-100}"

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

DATASET_ROOT="${DATASET_ROOT:-dataset/fair_plane_line_count_shared_global}"
INIT_T_TAG="${INIT_TRANSLATION_RANGE_MM//./p}"
INIT_R_TAG="${INIT_ANGLE_RANGE_DEG//./p}"
INIT_RESULT_TAG="init_t${INIT_T_TAG}_r${INIT_R_TAG}_${INIT_TRANSLATION_PERTURBATION}_${INIT_ROTATION_PERTURBATION}"
RESULT_ROOT="${RESULT_ROOT:-results/fair_plane_line_count_shared_global_${INIT_RESULT_TAG}}"

FORCE_REGENERATE="${FORCE_REGENERATE:-0}"
FORCE_RERUN="${FORCE_RERUN:-0}"
MAKE_PLOTS="${MAKE_PLOTS:-1}"
PLOT_SUCCESS_ONLY="${PLOT_SUCCESS_ONLY:-0}"
PLOT_DPI="${PLOT_DPI:-300}"

GENERATOR="main/generate_independent_random_plane_comparison.py"
CALIBRATOR="main/calibrate.py"
PLOTTER="main/make_plot/plot_fixed_noise_line_count_boxplots.py"
VALIDATOR="main/validate_experiment_artifact.py"

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

if [[ ! -f "${GENERATOR}" ]]; then
  echo "ERROR: generator not found: ${GENERATOR}" >&2
  echo "Run this script from the robust_laser_handeye repository root." >&2
  exit 1
fi

if [[ ! -f "${CALIBRATOR}" ]]; then
  echo "ERROR: calibrator not found: ${CALIBRATOR}" >&2
  echo "Run this script from the robust_laser_handeye repository root." >&2
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

if ! [[ "${MAX_TRIALS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: MAX_TRIALS must be a positive integer." >&2
  exit 1
fi

for N in ${SCAN_COUNTS}; do
  if ! [[ "${N}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: invalid scan count: ${N}" >&2
    exit 1
  fi
  if (( N % 3 != 0 )); then
    echo "ERROR: scan count ${N} is not divisible by 3." >&2
    exit 1
  fi
done

mkdir -p "${DATASET_ROOT}" "${RESULT_ROOT}"

echo "============================================================"
echo "Fixed-noise line-count comparison"
echo "============================================================"
echo "Scan counts        : ${SCAN_COUNTS}"
echo "Trials per setting : ${MAX_TRIALS}"
echo "Point noise        : sigma=${NOISE_STD_MM} mm, axis=${NOISE_AXIS}"
echo "Initialization     : ${INIT_TRANSLATION_RANGE_MM} mm / ${INIT_ANGLE_RANGE_DEG} deg max"
echo "Perturbation model : translation=${INIT_TRANSLATION_PERTURBATION}, rotation=${INIT_ROTATION_PERTURBATION}"
echo "Dataset root       : ${DATASET_ROOT}"
echo "Result root        : ${RESULT_ROOT}"
echo "============================================================"

# ---------------------------------------------------------------------------
# Run each line-count condition
# ---------------------------------------------------------------------------

for N in ${SCAN_COUNTS}; do
  DATASET_DIR="${DATASET_ROOT}/N${N}"
  SINGLE_COLLECTION="${DATASET_DIR}/single_plane"
  THREE_COLLECTION="${DATASET_DIR}/three_plane"

  CONDITION_RESULT_ROOT="${RESULT_ROOT}/N${N}"
  SINGLE_RESULT="${CONDITION_RESULT_ROOT}/single_plane"
  THREE_RESULT="${CONDITION_RESULT_ROOT}/three_plane"

  echo
  echo "------------------------------------------------------------"
  echo "N=${N}: single=${N}, three=$((N / 3)) scans/plane × 3"
  echo "------------------------------------------------------------"

  # -------------------------------------------------------------------------
  # 1) Generate ideal datasets
  # -------------------------------------------------------------------------

  if [[ "${FORCE_REGENERATE}" == "1" && -e "${DATASET_DIR}" ]]; then
    echo "[generate] removing existing dataset: ${DATASET_DIR}"
    rm -rf "${DATASET_DIR}"
  fi

  if [[ -f "${DATASET_DIR}/comparison_manifest.json" \
        && -f "${SINGLE_COLLECTION}/collection.json" \
        && -f "${THREE_COLLECTION}/collection.json" ]]; then
    echo "[generate] existing dataset found; skipping: ${DATASET_DIR}"
  else
    if [[ -e "${DATASET_DIR}" ]]; then
      echo "ERROR: incomplete or non-empty dataset directory exists:" >&2
      echo "  ${DATASET_DIR}" >&2
      echo "Use FORCE_REGENERATE=1 to recreate it." >&2
      exit 1
    fi

    echo "[generate] creating N=${N} dataset..."
    PYTHONPATH=. python3 "${GENERATOR}" \
      --trials "${MAX_TRIALS}" \
      --seed "${GENERATION_SEED}" \
      --output-dir "${DATASET_DIR}" \
      --pose-selection-mode random_only \
      --total-scans "${N}" \
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
    --total-scans "${N}" \
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

  # -------------------------------------------------------------------------
  # 2) Single-plane iterative calibration
  # -------------------------------------------------------------------------

  if [[ "${FORCE_RERUN}" == "1" && -e "${SINGLE_RESULT}" ]]; then
    echo "[single] removing existing result: ${SINGLE_RESULT}"
    rm -rf "${SINGLE_RESULT}"
  fi

  if [[ -f "${SINGLE_RESULT}/summary.json" ]]; then
    echo "[single] completed result found; skipping."
  else
    echo "[single] calibrating N=${N}..."
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
  # 3) Three-plane iterative calibration
  # -------------------------------------------------------------------------

  if [[ "${FORCE_RERUN}" == "1" && -e "${THREE_RESULT}" ]]; then
    echo "[three] removing existing result: ${THREE_RESULT}"
    rm -rf "${THREE_RESULT}"
  fi

  if [[ -f "${THREE_RESULT}/summary.json" ]]; then
    echo "[three] completed result found; skipping."
  else
    echo "[three] calibrating N=${N}..."
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

  echo "[done] N=${N}"
  echo "  single: ${SINGLE_RESULT}/trials.csv"
  echo "  three : ${THREE_RESULT}/trials.csv"
done

if [[ "${MAKE_PLOTS}" == "1" ]]; then
  echo
  echo "[plot] creating fixed-noise line-count plots..."
  read -r -a PLOT_SCAN_COUNTS <<< "${SCAN_COUNTS}"
  PLOT_ARGS_COMMON=(
    --result-root "${RESULT_ROOT}"
    --scan-counts "${PLOT_SCAN_COUNTS[@]}"
    --dpi "${PLOT_DPI}"
    --paired-csv "${RESULT_ROOT}/line_count_paired_differences.csv"
  )
  if [[ "${PLOT_SUCCESS_ONLY}" == "1" ]]; then
    PLOT_ARGS_COMMON+=(--success-only)
  fi

  PLOT_ARGS_HIDE=(
    "${PLOT_ARGS_COMMON[@]}"
    --output "${RESULT_ROOT}/line_count_boxplots_hide_outliers.png"
    --paired-output "${RESULT_ROOT}/line_count_paired_differences_hide_outliers.png"
    --hide-outliers
    --no-log-translation
    --no-log-rotation
  )
  PLOT_ARGS_SHOW=(
    "${PLOT_ARGS_COMMON[@]}"
    --output "${RESULT_ROOT}/line_count_boxplots_show_outliers.png"
    --paired-output "${RESULT_ROOT}/line_count_paired_differences_show_outliers.png"
    --show-outliers
    --log-translation
    --log-rotation
  )

  MPLBACKEND="${MPLBACKEND:-Agg}" PYTHONPATH=. \
    python3 "${PLOTTER}" "${PLOT_ARGS_HIDE[@]}"
  MPLBACKEND="${MPLBACKEND:-Agg}" PYTHONPATH=. \
    python3 "${PLOTTER}" "${PLOT_ARGS_SHOW[@]}"
fi

echo
echo "============================================================"
echo "All fixed-noise line-count experiments completed."
echo "Results: ${RESULT_ROOT}"
echo "============================================================"
