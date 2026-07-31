#!/usr/bin/env bash
set -euo pipefail

# Fixed-bootstrap scan-budget comparison:
#
#   shared INITIAL_SCANS:
#       sequential Sliced-LHS bootstrap
#
#   for every later scan budget N:
#       A) use the first N scans of the Sliced-LHS continuation sequence
#       B) use the first N scans of the active Fisher sequence
#
# The maximum-length dataset is generated only once. Every N-budget evaluation
# uses a prefix of that same sequential acquisition, so earlier scans never
# change when the budget increases.
#
# Default:
#   INITIAL_SCANS=27
#   TOTAL_SCANS=135
#   SCAN_COUNTS="27 54 81 108 135"  (generated automatically)
#
# Full rerun:
#   INITIAL_SCANS=27 TOTAL_SCANS=135 \
#   FORCE_REGENERATE=1 FORCE_RERUN=1 \
#   bash main/run_single_sliced_lhs_vs_fisher_scan_budget.sh
#
# Another bootstrap:
#   INITIAL_SCANS=18 TOTAL_SCANS=108 \
#   FORCE_REGENERATE=1 FORCE_RERUN=1 \
#   bash main/run_single_sliced_lhs_vs_fisher_scan_budget.sh
#
# Complete Sliced-LHS batches require:
#   TOTAL_SCANS % INITIAL_SCANS == 0
#
# Custom budgets are allowed through SCAN_COUNTS, but by default this script
# requires every budget to be a multiple of INITIAL_SCANS so each evaluation
# ends after a complete continuation slice.

INITIAL_SCANS="${INITIAL_SCANS:-27}"
TOTAL_SCANS="${TOTAL_SCANS:-135}"
CANDIDATE_POOL_SIZE="${CANDIDATE_POOL_SIZE:-5000}"
SOURCE_TRIALS="${SOURCE_TRIALS:-100}"
MAX_TRIALS="${MAX_TRIALS:-${SOURCE_TRIALS}}"

if [[ -z "${SCAN_COUNTS+x}" ]]; then
  SCAN_COUNTS=""
  for ((n = INITIAL_SCANS; n <= TOTAL_SCANS; n += INITIAL_SCANS)); do
    SCAN_COUNTS+="${n} "
  done
  SCAN_COUNTS="${SCAN_COUNTS% }"
fi

GENERATION_SEED="${GENERATION_SEED:-17}"
CALIBRATION_SEED="${CALIBRATION_SEED:-1701}"
MEASUREMENT_SEED="${MEASUREMENT_SEED:-1701}"
SELECTION_INITIALIZATION_SEED="${SELECTION_INITIALIZATION_SEED:-1701}"

PROFILE_POINTS="${PROFILE_POINTS:-100}"
PROFILE_HALF_WIDTH_MM="${PROFILE_HALF_WIDTH_MM:-25}"
TANGENT_RANGE_MM="${TANGENT_RANGE_MM:-100}"
PROFILE_DEPTH_RANGE_MM="${PROFILE_DEPTH_RANGE_MM:-60 150}"
CANDIDATE_CENTER_DEPTH_RANGE_MM="${CANDIDATE_CENTER_DEPTH_RANGE_MM:-90 120}"

# Plane-relative pose domain. This matches the selected moderate condition.
CANDIDATE_TARGET_U_RANGE_MM="${CANDIDATE_TARGET_U_RANGE_MM:-0 0}"
CANDIDATE_TARGET_V_RANGE_MM="${CANDIDATE_TARGET_V_RANGE_MM:-0 0}"
CANDIDATE_VIEW_TILT_RANGE_DEG="${CANDIDATE_VIEW_TILT_RANGE_DEG:-10 50}"
CANDIDATE_VIEW_AZIMUTH_RANGE_DEG="${CANDIDATE_VIEW_AZIMUTH_RANGE_DEG:--135 135}"
CANDIDATE_SENSOR_ROLL_RANGE_DEG="${CANDIDATE_SENSOR_ROLL_RANGE_DEG:--60 60}"
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
CALIBRATION_MODE="${CALIBRATION_MODE:-iterative_refit_nonlinear}"
NONLINEAR_LOSS="${NONLINEAR_LOSS:-linear}"
NONLINEAR_F_SCALE_MM="${NONLINEAR_F_SCALE_MM:-1.0}"
NONLINEAR_MAX_NFEV="${NONLINEAR_MAX_NFEV:-300}"
NONLINEAR_FTOL="${NONLINEAR_FTOL:-1e-10}"
NONLINEAR_XTOL="${NONLINEAR_XTOL:-1e-10}"
NONLINEAR_GTOL="${NONLINEAR_GTOL:-1e-10}"

DATASET_ROOT="${DATASET_ROOT:-dataset/single_sliced_lhs_vs_fisher}"
RESULT_ROOT="${RESULT_ROOT:-results/single_sliced_lhs_vs_fisher_scan_budget}"

FORCE_REGENERATE="${FORCE_REGENERATE:-0}"
FORCE_RERUN="${FORCE_RERUN:-0}"
AUTO_REPAIR_INCOMPATIBLE="${AUTO_REPAIR_INCOMPATIBLE:-1}"
MAKE_PLOTS="${MAKE_PLOTS:-1}"
PLOT_SUCCESS_ONLY="${PLOT_SUCCESS_ONLY:-0}"
PLOT_DPI="${PLOT_DPI:-300}"
POSE_VIS_TRIAL_INDEX="${POSE_VIS_TRIAL_INDEX:-0}"
POSE_VIS_MAX_ORIENTATION_AXES="${POSE_VIS_MAX_ORIENTATION_AXES:-18}"

GENERATOR="main/generate_single_sliced_lhs_vs_fisher.py"
CALIBRATOR="main/calibrate.py"
VALIDATOR="main/validate_experiment_artifact.py"
BUDGET_PLOTTER="main/make_plot/plot_single_uniform_fisher_scan_budget.py"
POSE_PLOTTER="main/make_plot/plot_plane_uniform_pose_geometry.py"

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

number_tag() {
  printf '%s' "$1" | sed -e 's/-/m/g' -e 's/\./p/g'
}

range_tag() {
  local -a values
  read -r -a values <<< "$1"
  printf '%s_%s' "$(number_tag "${values[0]}")" "$(number_tag "${values[1]}")"
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

for file in \
  "${GENERATOR}" "${CALIBRATOR}" "${VALIDATOR}" \
  "${BUDGET_PLOTTER}" "${POSE_PLOTTER}"; do
  if [[ ! -f "${file}" ]]; then
    echo "ERROR: required file not found: ${file}" >&2
    echo "Run this script from the robust_laser_handeye repository root." >&2
    exit 1
  fi
done

for value in \
  "${INITIAL_SCANS}" "${TOTAL_SCANS}" "${CANDIDATE_POOL_SIZE}" \
  "${SOURCE_TRIALS}" "${MAX_TRIALS}" "${PROFILE_POINTS}" \
  "${CANDIDATE_MAX_BATCHES}" "${CANDIDATE_BATCH_MULTIPLIER}" \
  "${ESTIMATOR_MAX_ITER}" "${MAX_ITER}" "${PLOT_DPI}"; do
  if ! [[ "${value}" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: expected a positive integer: ${value}" >&2
    exit 1
  fi
done

if (( INITIAL_SCANS < 9 || INITIAL_SCANS >= TOTAL_SCANS )); then
  echo "ERROR: INITIAL_SCANS must be >=9 and smaller than TOTAL_SCANS." >&2
  exit 1
fi
if (( TOTAL_SCANS % INITIAL_SCANS != 0 )); then
  echo "ERROR: TOTAL_SCANS must be divisible by INITIAL_SCANS." >&2
  echo "This keeps every default evaluation at a complete Sliced-LHS slice." >&2
  exit 1
fi
if (( CANDIDATE_POOL_SIZE < TOTAL_SCANS )); then
  echo "ERROR: CANDIDATE_POOL_SIZE must be >= TOTAL_SCANS." >&2
  exit 1
fi
if (( MAX_TRIALS > SOURCE_TRIALS )); then
  echo "ERROR: MAX_TRIALS cannot exceed SOURCE_TRIALS." >&2
  exit 1
fi
if (( POSE_VIS_TRIAL_INDEX < 0 || POSE_VIS_TRIAL_INDEX >= SOURCE_TRIALS )); then
  echo "ERROR: POSE_VIS_TRIAL_INDEX must lie in [0, SOURCE_TRIALS)." >&2
  exit 1
fi

for flag in \
  FORCE_REGENERATE FORCE_RERUN AUTO_REPAIR_INCOMPATIBLE \
  MAKE_PLOTS PLOT_SUCCESS_ONLY; do
  if [[ "${!flag}" != "0" && "${!flag}" != "1" ]]; then
    echo "ERROR: ${flag} must be 0 or 1." >&2
    exit 1
  fi
done

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
  if (( count % INITIAL_SCANS != 0 )); then
    echo "ERROR: scan budget ${count} is not a multiple of INITIAL_SCANS." >&2
    echo "Use complete Sliced-LHS continuation slices for a clean comparison." >&2
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
parse_pair "CANDIDATE_CENTER_DEPTH_RANGE_MM" \
  "${CANDIDATE_CENTER_DEPTH_RANGE_MM}" CANDIDATE_CENTER_DEPTH
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

NOISE_TAG="$(number_tag "${MEASUREMENT_NOISE_STD_MM}")"
CENTER_DEPTH_TAG="$(range_tag "${CANDIDATE_CENTER_DEPTH_RANGE_MM}")"
TILT_TAG="$(range_tag "${CANDIDATE_VIEW_TILT_RANGE_DEG}")"
AZIMUTH_TAG="$(range_tag "${CANDIDATE_VIEW_AZIMUTH_RANGE_DEG}")"
ROLL_TAG="$(range_tag "${CANDIDATE_SENSOR_ROLL_RANGE_DEG}")"
CONDITION_TAG="N${TOTAL_SCANS}_I${INITIAL_SCANS}_P${CANDIDATE_POOL_SIZE}_${FISHER_OBJECTIVE}_noise${NOISE_TAG}_d${CENTER_DEPTH_TAG}_t${TILT_TAG}_a${AZIMUTH_TAG}_r${ROLL_TAG}"

SOURCE_DATASET_DIR="${DATASET_ROOT}/${CONDITION_TAG}"
CONDITION_RESULT_ROOT="${RESULT_ROOT}/${CONDITION_TAG}"

mkdir -p "${DATASET_ROOT}" "${RESULT_ROOT}"

echo "============================================================"
echo "Sliced-LHS continuation vs active Fisher: scan budgets"
echo "============================================================"
echo "Trials to generate  : ${SOURCE_TRIALS}"
echo "Trials to evaluate  : ${MAX_TRIALS}"
echo "Shared bootstrap    : ${INITIAL_SCANS}"
echo "Maximum scans       : ${TOTAL_SCANS}"
echo "Scan budgets        : ${SCAN_COUNTS}"
echo "Candidate pool      : ${CANDIDATE_POOL_SIZE}"
echo "Profile depth ROI   : ${PROFILE_DEPTH_RANGE_MM} mm"
echo "Center depth range  : ${CANDIDATE_CENTER_DEPTH_RANGE_MM} mm"
echo "Tilt/azimuth/roll   : ${CANDIDATE_VIEW_TILT_RANGE_DEG} / ${CANDIDATE_VIEW_AZIMUTH_RANGE_DEG} / ${CANDIDATE_SENSOR_ROLL_RANGE_DEG}"
echo "Saved noise         : ${MEASUREMENT_NOISE_STD_MM} mm, ${MEASUREMENT_NOISE_AXIS}"
echo "Calibration mode    : ${CALIBRATION_MODE}"
echo "Source dataset      : ${SOURCE_DATASET_DIR}"
echo "Results             : ${CONDITION_RESULT_ROOT}"
echo "============================================================"

DATASET_REBUILT=0
if [[ "${FORCE_REGENERATE}" == "1" && -e "${SOURCE_DATASET_DIR}" ]]; then
  echo "[generate] removing existing dataset: ${SOURCE_DATASET_DIR}"
  rm -rf "${SOURCE_DATASET_DIR}"
fi

if [[ ! -f "${SOURCE_DATASET_DIR}/comparison_manifest.json" ]]; then
  if [[ -e "${SOURCE_DATASET_DIR}" ]]; then
    echo "ERROR: incomplete dataset directory exists: ${SOURCE_DATASET_DIR}" >&2
    echo "Use FORCE_REGENERATE=1." >&2
    exit 1
  fi

  echo "[generate] creating one maximum-length paired sequence..."
  PYTHONPATH=. python3 "${GENERATOR}" \
    --trials "${SOURCE_TRIALS}" \
    --seed "${GENERATION_SEED}" \
    --output-dir "${SOURCE_DATASET_DIR}" \
    --total-scans "${TOTAL_SCANS}" \
    --initial-scans "${INITIAL_SCANS}" \
    --candidate-pool-size "${CANDIDATE_POOL_SIZE}" \
    --profile-points "${PROFILE_POINTS}" \
    --profile-half-width-mm "${PROFILE_HALF_WIDTH_MM}" \
    --tangent-range-mm "${TANGENT_RANGE_MM}" \
    --profile-depth-range-mm "${PROFILE_DEPTH[@]}" \
    --candidate-center-depth-range-mm "${CANDIDATE_CENTER_DEPTH[@]}" \
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
    --initial-translation-perturbation "${INIT_TRANSLATION_PERTURBATION}" \
    --estimator-max-iterations "${ESTIMATOR_MAX_ITER}" \
    --estimator-tolerance "${ESTIMATOR_TOL}"
  DATASET_REBUILT=1
fi

# Structural audit independent of the old maximin-specific validator.
PYTHONPATH=. python3 - \
  "${SOURCE_DATASET_DIR}" \
  "${SOURCE_TRIALS}" \
  "${TOTAL_SCANS}" \
  "${INITIAL_SCANS}" \
  "${CANDIDATE_POOL_SIZE}" \
  "${PROFILE_DEPTH[0]}" "${PROFILE_DEPTH[1]}" \
  "${CANDIDATE_CENTER_DEPTH[0]}" "${CANDIDATE_CENTER_DEPTH[1]}" <<'PY'
from pathlib import Path
import json
import sys

root = Path(sys.argv[1])
expected_trials = int(sys.argv[2])
total_scans = int(sys.argv[3])
initial_scans = int(sys.argv[4])
candidate_pool = int(sys.argv[5])
expected_profile_depth = [float(sys.argv[6]), float(sys.argv[7])]
expected_center_depth = [float(sys.argv[8]), float(sys.argv[9])]

manifest = json.loads(
    (root / "comparison_manifest.json").read_text(encoding="utf-8")
)
if manifest.get("status") != "complete":
    raise SystemExit(f"incomplete comparison manifest: {manifest.get('status')}")
if int(manifest.get("completed_trials", -1)) != expected_trials:
    raise SystemExit("source trial count mismatch")

selection = manifest.get("pose_selection_config", {})
fair = manifest.get("fair_config", {})
if int(selection.get("initial_random_scans", -1)) != initial_scans:
    raise SystemExit("initial scan count mismatch")
if int(selection.get("candidate_pool_size", -1)) != candidate_pool:
    raise SystemExit("candidate pool mismatch")
if int(fair.get("total_scans", -1)) != total_scans:
    raise SystemExit("total scan count mismatch")
if list(map(float, fair.get("profile_depth_range_mm", []))) != expected_profile_depth:
    raise SystemExit("profile depth ROI mismatch")
candidate = manifest.get("candidate_config", {})
if list(map(float, candidate.get("depth_range_mm", []))) != expected_center_depth:
    raise SystemExit("candidate center depth range mismatch")

for item in manifest.get("trials", []):
    if not item.get("same_initial_bootstrap", False):
        raise SystemExit("a trial does not share the bootstrap")
    uniform_ids = item["single_uniform_selected_candidate_ids"]
    fisher_ids = item["single_fisher_selected_candidate_ids"]
    initial_ids = item["initial_candidate_ids"]
    if uniform_ids[:initial_scans] != initial_ids:
        raise SystemExit("uniform bootstrap ID mismatch")
    if fisher_ids[:initial_scans] != initial_ids:
        raise SystemExit("Fisher bootstrap ID mismatch")
    if len(uniform_ids) != total_scans or len(fisher_ids) != total_scans:
        raise SystemExit("saved sequential acquisition length mismatch")

for collection_name in ("single_plane_uniform", "single_plane_fisher"):
    collection = json.loads(
        (root / collection_name / "collection.json").read_text(encoding="utf-8")
    )
    if collection.get("status") != "complete":
        raise SystemExit(f"incomplete collection: {collection_name}")
    if int(collection.get("completed_trials", -1)) != expected_trials:
        raise SystemExit(f"collection trial mismatch: {collection_name}")

print(
    f"[dataset audit] trials={expected_trials}, bootstrap={initial_scans}, "
    f"maximum={total_scans}, pool={candidate_pool}"
)
PY

if [[ "${DATASET_REBUILT}" == "1" && "${FORCE_RERUN}" != "1" ]]; then
  echo "[cache] source sequence changed; forcing all budget calibrations."
  FORCE_RERUN=1
fi

NONLINEAR_ARGS=(
  --nonlinear-loss "${NONLINEAR_LOSS}"
  --nonlinear-f-scale-mm "${NONLINEAR_F_SCALE_MM}"
  --nonlinear-max-nfev "${NONLINEAR_MAX_NFEV}"
  --nonlinear-ftol "${NONLINEAR_FTOL}"
  --nonlinear-xtol "${NONLINEAR_XTOL}"
  --nonlinear-gtol "${NONLINEAR_GTOL}"
)

validate_result() {
  local summary="$1"
  local collection="$2"
  local scan_count="$3"

  PYTHONPATH=. python3 "${VALIDATOR}" result \
    --mode "${CALIBRATION_MODE}" \
    "${NONLINEAR_ARGS[@]}" \
    --summary "${summary}" \
    --collection "${collection}" \
    --trials "${MAX_TRIALS}" \
    --seed "${CALIBRATION_SEED}" \
    --noise-axis "${MEASUREMENT_NOISE_AXIS}" \
    --noise-std-mm 0 \
    --init-translation-range-mm "${INIT_TRANSLATION_RANGE_MM}" \
    --init-angle-range-deg "${INIT_ANGLE_RANGE_DEG}" \
    --init-rotation-perturbation "${INIT_ROTATION_PERTURBATION}" \
    --init-translation-perturbation "${INIT_TRANSLATION_PERTURBATION}" \
    --max-scans-per-trial "${scan_count}" \
    --max-iter "${MAX_ITER}" \
    --tol "${TOL}"
}

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
      compatible=1
      if [[ ! -f "${digest_file}" ]] \
        || [[ "$(<"${digest_file}")" != "${collection_digest}" ]]; then
        compatible=0
      elif ! validate_result \
        "${result_dir}/summary.json" "${collection}" "${scan_count}"; then
        compatible=0
      fi

      if [[ "${compatible}" == "1" ]]; then
        echo "  compatible completed result found."
        continue
      fi

      if [[ "${AUTO_REPAIR_INCOMPATIBLE}" == "1" ]]; then
        echo "  removing incompatible cached result."
        rm -rf "${result_dir}"
      else
        echo "ERROR: incompatible cached result: ${result_dir}" >&2
        exit 1
      fi
    fi

    PYTHONPATH=. python3 "${CALIBRATOR}" \
      --collection "${collection}" \
      --output-dir "${result_dir}" \
      --mode "${CALIBRATION_MODE}" \
      "${NONLINEAR_ARGS[@]}" \
      --max-trials "${MAX_TRIALS}" \
      --max-scans-per-trial "${scan_count}" \
      --seed "${CALIBRATION_SEED}" \
      --noise-axis "${MEASUREMENT_NOISE_AXIS}" \
      --noise-std-mm 0 \
      --init-mode carlson \
      --init-translation-range-mm "${INIT_TRANSLATION_RANGE_MM}" \
      --init-angle-range-deg "${INIT_ANGLE_RANGE_DEG}" \
      --init-rotation-perturbation "${INIT_ROTATION_PERTURBATION}" \
      --init-translation-perturbation "${INIT_TRANSLATION_PERTURBATION}" \
      --max-iter "${MAX_ITER}" \
      --tol "${TOL}"

    mkdir -p "${result_dir}"
    printf "%s\n" "${collection_digest}" > "${digest_file}"
    validate_result "${result_dir}/summary.json" "${collection}" "${scan_count}"
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

  echo
  echo "[plot] scan-budget error curves and paired comparisons..."
  MPLBACKEND="${MPLBACKEND:-Agg}" \
  MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/lvs-matplotlib-cache}" \
  PYTHONPATH=. python3 "${BUDGET_PLOTTER}" \
    "${COMMON_PLOT_ARGS[@]}" \
    --output-dir "${CONDITION_RESULT_ROOT}/plots_hide_outliers" \
    --hide-outliers \
    --no-log-errors

  MPLBACKEND="${MPLBACKEND:-Agg}" \
  MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/lvs-matplotlib-cache}" \
  PYTHONPATH=. python3 "${BUDGET_PLOTTER}" \
    "${COMMON_PLOT_ARGS[@]}" \
    --output-dir "${CONDITION_RESULT_ROOT}/plots_show_outliers" \
    --log-errors

  echo "[plot] maximum-budget 3D pose geometry..."
  MPLBACKEND="${MPLBACKEND:-Agg}" \
  MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/lvs-matplotlib-cache}" \
  PYTHONPATH=. python3 "${POSE_PLOTTER}" \
    --dataset-root "${SOURCE_DATASET_DIR}" \
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
echo "Scan-budget comparison complete."
echo "Source dataset : ${SOURCE_DATASET_DIR}"
echo "Results        : ${CONDITION_RESULT_ROOT}"
if [[ "${MAKE_PLOTS}" == "1" ]]; then
  echo "Plots (robust) : ${CONDITION_RESULT_ROOT}/plots_hide_outliers"
  echo "Plots (all)    : ${CONDITION_RESULT_ROOT}/plots_show_outliers"
  echo "3D poses       : ${CONDITION_RESULT_ROOT}/plots_pose_3d"
fi
echo "============================================================"
