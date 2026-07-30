#!/usr/bin/env bash
set -euo pipefail

# Paired comparison on identical noisy trials:
#   alternating unknown-plane solver
#       vs
#   alternating -> joint nonlinear refinement of T_ef_s + every plane (n, d)
#
# Both the before/after estimates are stored in the same trials.csv row, so
# the alternating stage is run only once per trial.
#
# Full run:
#   bash main/run_alternating_joint_nonlinear_comparison.sh
#
# Quick test:
#   MAX_TRIALS=3 bash main/run_alternating_joint_nonlinear_comparison.sh

TOTAL_SCANS="${TOTAL_SCANS:-108}"
DATASET_TRIALS="${DATASET_TRIALS:-100}"
MAX_TRIALS="${MAX_TRIALS:-100}"
NOISE_STD_MM="${NOISE_STD_MM:-0.20}"
NOISE_AXIS="${NOISE_AXIS:-xz}"

GENERATION_SEED="${GENERATION_SEED:-17}"
CALIBRATION_SEED="${CALIBRATION_SEED:-1701}"

INIT_TRANSLATION_RANGE_MM="${INIT_TRANSLATION_RANGE_MM:-100}"
INIT_ANGLE_RANGE_DEG="${INIT_ANGLE_RANGE_DEG:-15}"
INIT_TRANSLATION_PERTURBATION="${INIT_TRANSLATION_PERTURBATION:-direction_norm}"
INIT_ROTATION_PERTURBATION="${INIT_ROTATION_PERTURBATION:-axis_angle}"

MAX_ITER="${MAX_ITER:-3000}"
TOL="${TOL:-1e-5}"

NONLINEAR_LOSS="${NONLINEAR_LOSS:-linear}"
NONLINEAR_F_SCALE_MM="${NONLINEAR_F_SCALE_MM:-1.0}"
NONLINEAR_MAX_NFEV="${NONLINEAR_MAX_NFEV:-300}"

PROFILE_POINTS="${PROFILE_POINTS:-100}"
PROFILE_HALF_WIDTH_MM="${PROFILE_HALF_WIDTH_MM:-25}"
TANGENT_RANGE_MM="${TANGENT_RANGE_MM:-100}"

DATASET_ROOT="${DATASET_ROOT:-dataset/fair_plane_initialization_shared_global}"
DATASET_DIR="${DATASET_ROOT}/N${TOTAL_SCANS}"
RESULT_ROOT="${RESULT_ROOT:-results/alternating_joint_nonlinear_comparison_t${INIT_TRANSLATION_RANGE_MM}_r${INIT_ANGLE_RANGE_DEG}}"

FORCE_REGENERATE="${FORCE_REGENERATE:-0}"
FORCE_RERUN="${FORCE_RERUN:-0}"
MAKE_PLOTS="${MAKE_PLOTS:-1}"
PLOT_DPI="${PLOT_DPI:-300}"

GENERATOR="main/generate_independent_random_plane_comparison.py"
CALIBRATOR="main/calibrate.py"
VALIDATOR="main/validate_experiment_artifact.py"
PLOTTER="main/make_plot/plot_alternating_joint_nonlinear_comparison.py"

if (( TOTAL_SCANS % 3 != 0 )); then
  echo "ERROR: TOTAL_SCANS must be divisible by 3." >&2
  exit 1
fi
if (( MAX_TRIALS > DATASET_TRIALS )); then
  echo "ERROR: MAX_TRIALS cannot exceed DATASET_TRIALS." >&2
  exit 1
fi

for path in "${GENERATOR}" "${CALIBRATOR}" "${VALIDATOR}"; do
  if [[ ! -f "${path}" ]]; then
    echo "ERROR: required file not found: ${path}" >&2
    exit 1
  fi
done

if [[ "${MAKE_PLOTS}" == "1" && ! -f "${PLOTTER}" ]]; then
  echo "ERROR: plotter not found: ${PLOTTER}" >&2
  exit 1
fi

SINGLE_COLLECTION="${DATASET_DIR}/single_plane"
THREE_COLLECTION="${DATASET_DIR}/three_plane"

echo "============================================================"
echo "Alternating -> joint nonlinear comparison"
echo "============================================================"
echo "Total scans       : ${TOTAL_SCANS}"
echo "Single-plane      : ${TOTAL_SCANS} scans on one plane"
echo "Three-plane       : $((TOTAL_SCANS / 3)) scans/plane x 3"
echo "Dataset trials    : ${DATASET_TRIALS}"
echo "Trials to run     : ${MAX_TRIALS}"
echo "Profile noise     : ${NOISE_STD_MM} mm (${NOISE_AXIS})"
echo "Initial error     : <= ${INIT_TRANSLATION_RANGE_MM} mm norm,"
echo "                    <= ${INIT_ANGLE_RANGE_DEG} deg axis-angle"
echo "Alternating       : max_iter=${MAX_ITER}, tol=${TOL}"
echo "Joint nonlinear   : loss=${NONLINEAR_LOSS},"
echo "                    f_scale=${NONLINEAR_F_SCALE_MM} mm,"
echo "                    max_nfev=${NONLINEAR_MAX_NFEV}"
echo "============================================================"

if [[ "${FORCE_REGENERATE}" == "1" && -e "${DATASET_DIR}" ]]; then
  echo "[generate] removing explicitly selected dataset: ${DATASET_DIR}"
  rm -rf "${DATASET_DIR}"
fi

if [[ -f "${DATASET_DIR}/comparison_manifest.json" \
      && -f "${SINGLE_COLLECTION}/collection.json" \
      && -f "${THREE_COLLECTION}/collection.json" ]]; then
  echo "[generate] compatible-looking dataset found; validating."
else
  if [[ -e "${DATASET_DIR}" ]]; then
    echo "ERROR: incomplete dataset directory exists: ${DATASET_DIR}" >&2
    echo "Use FORCE_REGENERATE=1 to recreate it." >&2
    exit 1
  fi
  echo "[generate] creating shared ideal single/three-plane datasets..."
  PYTHONPATH=. python3 "${GENERATOR}" \
    --trials "${DATASET_TRIALS}" \
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
  --trials "${DATASET_TRIALS}" \
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

mkdir -p "${RESULT_ROOT}"

for METHOD in single_plane three_plane; do
  if [[ "${METHOD}" == "single_plane" ]]; then
    COLLECTION="${SINGLE_COLLECTION}"
  else
    COLLECTION="${THREE_COLLECTION}"
  fi
  OUTPUT_DIR="${RESULT_ROOT}/${METHOD}"

  if [[ "${FORCE_RERUN}" == "1" && -e "${OUTPUT_DIR}" ]]; then
    echo "[${METHOD}] removing explicitly selected result: ${OUTPUT_DIR}"
    rm -rf "${OUTPUT_DIR}"
  fi

  if [[ -f "${OUTPUT_DIR}/summary.json" ]]; then
    echo "[${METHOD}] completed result found; validating."
  else
    echo "[${METHOD}] calibrating alternating + joint nonlinear..."
    PYTHONPATH=. python3 "${CALIBRATOR}" \
      --collection "${COLLECTION}" \
      --output-dir "${OUTPUT_DIR}" \
      --mode iterative_joint_nonlinear \
      --max-trials "${MAX_TRIALS}" \
      --seed "${CALIBRATION_SEED}" \
      --noise-axis "${NOISE_AXIS}" \
      --noise-std-mm "${NOISE_STD_MM}" \
      --init-mode carlson \
      --init-translation-range-mm "${INIT_TRANSLATION_RANGE_MM}" \
      --init-angle-range-deg "${INIT_ANGLE_RANGE_DEG}" \
      --init-translation-perturbation "${INIT_TRANSLATION_PERTURBATION}" \
      --init-rotation-perturbation "${INIT_ROTATION_PERTURBATION}" \
      --max-iter "${MAX_ITER}" \
      --tol "${TOL}" \
      --nonlinear-loss "${NONLINEAR_LOSS}" \
      --nonlinear-f-scale-mm "${NONLINEAR_F_SCALE_MM}" \
      --nonlinear-max-nfev "${NONLINEAR_MAX_NFEV}"
  fi

  if ! PYTHONPATH=. python3 "${VALIDATOR}" result \
    --summary "${OUTPUT_DIR}/summary.json" \
    --mode iterative_joint_nonlinear \
    --collection "${COLLECTION}" \
    --trials "${MAX_TRIALS}" \
    --seed "${CALIBRATION_SEED}" \
    --noise-axis "${NOISE_AXIS}" \
    --noise-std-mm "${NOISE_STD_MM}" \
    --init-translation-range-mm "${INIT_TRANSLATION_RANGE_MM}" \
    --init-angle-range-deg "${INIT_ANGLE_RANGE_DEG}" \
    --init-translation-perturbation "${INIT_TRANSLATION_PERTURBATION}" \
    --init-rotation-perturbation "${INIT_ROTATION_PERTURBATION}" \
    --max-iter "${MAX_ITER}" \
    --tol "${TOL}" \
    --nonlinear-loss "${NONLINEAR_LOSS}" \
    --nonlinear-f-scale-mm "${NONLINEAR_F_SCALE_MM}" \
    --nonlinear-max-nfev "${NONLINEAR_MAX_NFEV}"; then
    echo "Use FORCE_RERUN=1 to replace the incompatible result." >&2
    exit 1
  fi
done

if [[ "${MAKE_PLOTS}" == "1" ]]; then
  echo "[plot] creating paired comparison plots..."
  mkdir -p "${MPLCONFIGDIR:-/tmp/lvs-matplotlib-cache}"
  MPLBACKEND="${MPLBACKEND:-Agg}" \
  MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/lvs-matplotlib-cache}" \
  PYTHONPATH=. python3 "${PLOTTER}" \
    --result-root "${RESULT_ROOT}" \
    --dpi "${PLOT_DPI}"
fi

echo "============================================================"
echo "Completed: ${RESULT_ROOT}"
echo "Each trials.csv contains both alternating_* and final refined fields."
echo "============================================================"
