#!/usr/bin/env bash
set -euo pipefail

# Exact sliced-LHS equal-budget experiments.

# FORCE_REGENERATE=1 \
# FORCE_RERUN=1 \
# bash main/run_sliced_lhs_5level_experiments.sh


EXPERIMENTS="${EXPERIMENTS:-pose_range scan_count noise}"
POSE_LEVELS="${POSE_LEVELS:-\
restricted:20:40:-90:90:-45:45 \
moderate:10:50:-135:135:-60:60 \
wide:0:60:-180:180:-90:90}"
SHARED_POSE_RANGE_SPECS="${SHARED_POSE_RANGE_SPECS:-\
20:40:-90:90:-45:45 \
10:50:-135:135:-60:60 \
0:60:-180:180:-90:90}"

SCAN_COUNTS="${SCAN_COUNTS:-27 54 81 108 135}"
NOISE_LEVELS="${NOISE_LEVELS:-0.00 0.10 0.20 0.30 0.40}"

BASELINE_TOTAL_SCANS="${BASELINE_TOTAL_SCANS:-108}"
BASELINE_NOISE_STD_MM="${BASELINE_NOISE_STD_MM:-0.20}"
MAX_TRIALS="${MAX_TRIALS:-100}"

GENERATION_SEED="${GENERATION_SEED:-17}"
CALIBRATION_SEED="${CALIBRATION_SEED:-1701}"

PROFILE_POINTS="${PROFILE_POINTS:-100}"
PROFILE_HALF_WIDTH_MM="${PROFILE_HALF_WIDTH_MM:-25}"
TANGENT_RANGE_MM="${TANGENT_RANGE_MM:-100}"
PROFILE_DEPTH_MIN_MM="${PROFILE_DEPTH_MIN_MM:-60}"
PROFILE_DEPTH_MAX_MM="${PROFILE_DEPTH_MAX_MM:-150}"

TARGET_U_MIN_MM="${TARGET_U_MIN_MM:-0}"
TARGET_U_MAX_MM="${TARGET_U_MAX_MM:-0}"
TARGET_V_MIN_MM="${TARGET_V_MIN_MM:-0}"
TARGET_V_MAX_MM="${TARGET_V_MAX_MM:-0}"

MAX_DESIGN_ATTEMPTS="${MAX_DESIGN_ATTEMPTS:-200}"
FEASIBILITY_MODE="${FEASIBILITY_MODE:-geometric}"

NOISE_AXIS="${NOISE_AXIS:-xz}"
INIT_TRANSLATION_RANGE_MM="${INIT_TRANSLATION_RANGE_MM:-100}"
INIT_ANGLE_RANGE_DEG="${INIT_ANGLE_RANGE_DEG:-15}"
INIT_ROTATION_PERTURBATION="${INIT_ROTATION_PERTURBATION:-axis_angle}"
INIT_TRANSLATION_PERTURBATION="${INIT_TRANSLATION_PERTURBATION:-direction_norm}"

MAX_ITER="${MAX_ITER:-3000}"
TOL="${TOL:-1e-5}"
CALIBRATION_MODE="${CALIBRATION_MODE:-iterative_refit_nonlinear}"
NONLINEAR_LOSS="${NONLINEAR_LOSS:-linear}"
NONLINEAR_F_SCALE_MM="${NONLINEAR_F_SCALE_MM:-1.0}"
NONLINEAR_MAX_NFEV="${NONLINEAR_MAX_NFEV:-300}"
NONLINEAR_FTOL="${NONLINEAR_FTOL:-1e-10}"
NONLINEAR_XTOL="${NONLINEAR_XTOL:-1e-10}"
NONLINEAR_GTOL="${NONLINEAR_GTOL:-1e-10}"

# New roots intentionally avoid the old interrupted non-sliced dataset.
DATASET_ROOT="${DATASET_ROOT:-dataset/sliced_lhs_equal_budget_mc100_nonlinear}"
RESULT_ROOT="${RESULT_ROOT:-results/sliced_lhs_equal_budget_mc100_nonlinear}"

FORCE_REGENERATE="${FORCE_REGENERATE:-0}"
FORCE_RERUN="${FORCE_RERUN:-0}"
AUTO_REPAIR_INCOMPLETE_RESULTS="${AUTO_REPAIR_INCOMPLETE_RESULTS:-1}"
MAKE_PLOTS="${MAKE_PLOTS:-1}"
PLOT_SUCCESS_ONLY="${PLOT_SUCCESS_ONLY:-0}"
PLOT_DPI="${PLOT_DPI:-300}"

GENERATOR="main/generate_sliced_lhs_plane_comparison.py"
CALIBRATOR="main/calibrate.py"
VALIDATOR="main/validate_experiment_artifact.py"
POSE_PLOTTER="main/make_plot/plot_pose_diversity_comparison.py"
SCAN_PLOTTER="main/make_plot/plot_fixed_noise_line_count_boxplots.py"
NOISE_PLOTTER="main/make_plot/plot_fixed_line_noise_boxplots.py"

NONLINEAR_ARGS=(
  --nonlinear-loss "${NONLINEAR_LOSS}"
  --nonlinear-f-scale-mm "${NONLINEAR_F_SCALE_MM}"
  --nonlinear-max-nfev "${NONLINEAR_MAX_NFEV}"
  --nonlinear-ftol "${NONLINEAR_FTOL}"
  --nonlinear-xtol "${NONLINEAR_XTOL}"
  --nonlinear-gtol "${NONLINEAR_GTOL}"
)

for required in "${GENERATOR}" "${CALIBRATOR}" "${VALIDATOR}"; do
  if [[ ! -f "${required}" ]]; then
    echo "ERROR: required file not found: ${required}" >&2
    echo "Run from the robust_laser_handeye repository root." >&2
    exit 1
  fi
done

if [[ "${MAKE_PLOTS}" == "1" ]]; then
  for plotter in "${POSE_PLOTTER}" "${SCAN_PLOTTER}" "${NOISE_PLOTTER}"; do
    if [[ ! -f "${plotter}" ]]; then
      echo "ERROR: plotter not found: ${plotter}" >&2
      exit 1
    fi
  done
fi

for flag_name in \
  FORCE_REGENERATE FORCE_RERUN AUTO_REPAIR_INCOMPLETE_RESULTS \
  MAKE_PLOTS PLOT_SUCCESS_ONLY; do
  flag_value="${!flag_name}"
  if [[ "${flag_value}" != "0" && "${flag_value}" != "1" ]]; then
    echo "ERROR: ${flag_name} must be 0 or 1." >&2
    exit 1
  fi
done

case "${CALIBRATION_MODE}" in
  iterative|iterative_refit_nonlinear|iterative_joint_nonlinear) ;;
  *)
    echo "ERROR: unsupported CALIBRATION_MODE=${CALIBRATION_MODE}" >&2
    exit 1
    ;;
esac

if ! [[ "${MAX_TRIALS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: MAX_TRIALS must be positive." >&2
  exit 1
fi

for n in ${SCAN_COUNTS} "${BASELINE_TOTAL_SCANS}"; do
  if ! [[ "${n}" =~ ^[1-9][0-9]*$ ]] || (( n < 9 || n % 3 != 0 )); then
    echo "ERROR: scan count must be >=9 and divisible by 3: ${n}" >&2
    exit 1
  fi
done

mkdir -p "${DATASET_ROOT}" "${RESULT_ROOT}"

contains_experiment() {
  local name="$1"
  [[ " ${EXPERIMENTS} " == *" ${name} "* ]]
}

number_tag() {
  printf '%s' "$1" | sed -e 's/-/m/g' -e 's/\./p/g'
}

range_tag() {
  printf 't%s_%s_a%s_%s_r%s_%s' \
    "$(number_tag "$1")" "$(number_tag "$2")" \
    "$(number_tag "$3")" "$(number_tag "$4")" \
    "$(number_tag "$5")" "$(number_tag "$6")"
}

CURRENT_DATASET_DIR=""
CURRENT_SINGLE_COLLECTION=""
CURRENT_THREE_COLLECTION=""

ensure_sliced_pair() {
  local total_scans="$1"
  local tilt_min="$2"
  local tilt_max="$3"
  local azimuth_min="$4"
  local azimuth_max="$5"
  local roll_min="$6"
  local roll_max="$7"

  local pose_tag
  pose_tag="$(range_tag \
    "${tilt_min}" "${tilt_max}" \
    "${azimuth_min}" "${azimuth_max}" \
    "${roll_min}" "${roll_max}")"

  local dataset_dir="${DATASET_ROOT}/N${total_scans}_${pose_tag}"
  local single_collection="${dataset_dir}/single_plane_sliced_lhs"
  local three_collection="${dataset_dir}/three_plane_sliced_lhs"

  if [[ "${FORCE_REGENERATE}" == "1" && -e "${dataset_dir}" ]]; then
    echo "[generate] removing dataset: ${dataset_dir}"
    rm -rf "${dataset_dir}"
  fi

  generator_args=(
    --trials "${MAX_TRIALS}"
    --seed "${GENERATION_SEED}"
    --output-dir "${dataset_dir}"
    --total-scans "${total_scans}"
    --profile-points "${PROFILE_POINTS}"
    --profile-half-width-mm "${PROFILE_HALF_WIDTH_MM}"
    --tangent-range-mm "${TANGENT_RANGE_MM}"
    --profile-depth-range-mm \
      "${PROFILE_DEPTH_MIN_MM}" "${PROFILE_DEPTH_MAX_MM}"
    --uniform-target-u-range-mm \
      "${TARGET_U_MIN_MM}" "${TARGET_U_MAX_MM}"
    --uniform-target-v-range-mm \
      "${TARGET_V_MIN_MM}" "${TARGET_V_MAX_MM}"
    --uniform-view-tilt-range-deg "${tilt_min}" "${tilt_max}"
    --uniform-view-azimuth-range-deg \
      "${azimuth_min}" "${azimuth_max}"
    --uniform-sensor-roll-range-deg "${roll_min}" "${roll_max}"
    --max-design-attempts "${MAX_DESIGN_ATTEMPTS}"
    --feasibility-mode "${FEASIBILITY_MODE}"
  )

  for shared_spec in ${SHARED_POSE_RANGE_SPECS}; do
    generator_args+=(--shared-feasibility-pose-range "${shared_spec}")
  done

  echo "[generate/resume] exact sliced LHS: ${dataset_dir}"
  PYTHONPATH=. python3 "${GENERATOR}" "${generator_args[@]}"

  PYTHONPATH=. python3 - \
    "${single_collection}" "${three_collection}" "${MAX_TRIALS}" <<'PY'
from pathlib import Path
import json
import sys

expected = int(sys.argv[3])
for raw in sys.argv[1:3]:
    path = Path(raw) / "collection.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete":
        raise SystemExit(f"collection is not complete: {path}, status={payload.get('status')}")
    if int(payload.get("completed_trials", -1)) != expected:
        raise SystemExit(
            f"collection trial mismatch: {path}, "
            f"completed={payload.get('completed_trials')}, expected={expected}"
        )
print(f"[dataset-check] complete: {expected} trials")
PY

  CURRENT_DATASET_DIR="${dataset_dir}"
  CURRENT_SINGLE_COLLECTION="${single_collection}"
  CURRENT_THREE_COLLECTION="${three_collection}"
}

validate_result() {
  local summary="$1"
  local collection="$2"
  local noise_std_mm="$3"

  PYTHONPATH=. python3 "${VALIDATOR}" result \
    --mode "${CALIBRATION_MODE}" \
    "${NONLINEAR_ARGS[@]}" \
    --summary "${summary}" \
    --collection "${collection}" \
    --trials "${MAX_TRIALS}" \
    --seed "${CALIBRATION_SEED}" \
    --noise-axis "${NOISE_AXIS}" \
    --noise-std-mm "${noise_std_mm}" \
    --init-translation-range-mm "${INIT_TRANSLATION_RANGE_MM}" \
    --init-angle-range-deg "${INIT_ANGLE_RANGE_DEG}" \
    --init-rotation-perturbation "${INIT_ROTATION_PERTURBATION}" \
    --init-translation-perturbation "${INIT_TRANSLATION_PERTURBATION}" \
    --max-iter "${MAX_ITER}" \
    --tol "${TOL}"
}

run_calibration() {
  local method_label="$1"
  local collection="$2"
  local output_dir="$3"
  local noise_std_mm="$4"

  if [[ "${FORCE_RERUN}" == "1" && -e "${output_dir}" ]]; then
    rm -rf "${output_dir}"
  fi

  if [[ -e "${output_dir}" && ! -f "${output_dir}/summary.json" ]]; then
    if [[ "${AUTO_REPAIR_INCOMPLETE_RESULTS}" == "1" ]]; then
      echo "[${method_label}] removing incomplete result: ${output_dir}"
      rm -rf "${output_dir}"
    else
      echo "ERROR: incomplete result directory: ${output_dir}" >&2
      exit 1
    fi
  fi

  if [[ -f "${output_dir}/summary.json" ]]; then
    if validate_result \
      "${output_dir}/summary.json" "${collection}" "${noise_std_mm}"; then
      echo "[${method_label}] compatible completed result found: ${output_dir}"
      return
    fi
    if [[ "${AUTO_REPAIR_INCOMPLETE_RESULTS}" == "1" ]]; then
      echo "[${method_label}] removing incompatible result: ${output_dir}"
      rm -rf "${output_dir}"
    else
      exit 1
    fi
  fi

  echo "[${method_label}] calibrating: mode=${CALIBRATION_MODE}, noise=${noise_std_mm}"
  PYTHONPATH=. python3 "${CALIBRATOR}" \
    --collection "${collection}" \
    --output-dir "${output_dir}" \
    --mode "${CALIBRATION_MODE}" \
    "${NONLINEAR_ARGS[@]}" \
    --seed "${CALIBRATION_SEED}" \
    --noise-axis "${NOISE_AXIS}" \
    --noise-std-mm "${noise_std_mm}" \
    --init-mode carlson \
    --init-translation-range-mm "${INIT_TRANSLATION_RANGE_MM}" \
    --init-angle-range-deg "${INIT_ANGLE_RANGE_DEG}" \
    --init-rotation-perturbation "${INIT_ROTATION_PERTURBATION}" \
    --init-translation-perturbation "${INIT_TRANSLATION_PERTURBATION}" \
    --max-iter "${MAX_ITER}" \
    --tol "${TOL}"

  validate_result \
    "${output_dir}/summary.json" "${collection}" "${noise_std_mm}"
}

calibrate_pair() {
  local condition_root="$1"
  local noise_std_mm="$2"

  run_calibration \
    "single" \
    "${CURRENT_SINGLE_COLLECTION}" \
    "${condition_root}/single_plane" \
    "${noise_std_mm}"

  run_calibration \
    "three" \
    "${CURRENT_THREE_COLLECTION}" \
    "${condition_root}/three_plane" \
    "${noise_std_mm}"
}

if contains_experiment pose_range; then
  echo
  echo "============================================================"
  echo "Exact sliced-LHS pose-range analysis"
  echo "============================================================"

  pose_result_root="${RESULT_ROOT}/pose_range"
  mkdir -p "${pose_result_root}"
  pose_dataset_dirs=()
  pose_level_names=()

  for level_spec in ${POSE_LEVELS}; do
    IFS=":" read -r \
      level_name \
      tilt_min tilt_max \
      azimuth_min azimuth_max \
      roll_min roll_max \
      <<< "${level_spec}"

    echo
    echo "[pose_range] ${level_name}"
    ensure_sliced_pair \
      "${BASELINE_TOTAL_SCANS}" \
      "${tilt_min}" "${tilt_max}" \
      "${azimuth_min}" "${azimuth_max}" \
      "${roll_min}" "${roll_max}"

    pose_dataset_dirs+=("${CURRENT_DATASET_DIR}")
    pose_level_names+=("${level_name}")

    calibrate_pair \
      "${pose_result_root}/${level_name}" \
      "${BASELINE_NOISE_STD_MM}"
  done

  # Verify that restricted/moderate/wide use the same normalized sliced design
  # in every corresponding Monte-Carlo trial.
  PYTHONPATH=. python3 - "${pose_dataset_dirs[@]}" <<'PY'
from pathlib import Path
import json
import sys

manifests = [
    json.loads((Path(path) / "comparison_manifest.json").read_text(encoding="utf-8"))
    for path in sys.argv[1:]
]
reference = [
    trial["normalized_design_sha256"]
    for trial in manifests[0]["trials"]
]
for index, manifest in enumerate(manifests[1:], start=1):
    current = [
        trial["normalized_design_sha256"]
        for trial in manifest["trials"]
    ]
    if current != reference:
        raise SystemExit(
            f"normalized sliced-LHS designs differ across pose ranges: index={index}"
        )
print(
    f"[pose-range audit] identical normalized sliced designs across "
    f"{len(manifests)} ranges and {len(reference)} trials"
)
PY

  if [[ "${MAKE_PLOTS}" == "1" ]]; then
    echo "[plot] pose-range figures -> ${pose_result_root}/plots"
    plot_args=(
      --result-root "${pose_result_root}"
      --levels "${pose_level_names[@]}"
      --output-dir "${pose_result_root}/plots"
      --dpi "${PLOT_DPI}"
      --no-log-translation
      --no-log-rotation
      --no-log-iterations
    )
    if [[ "${PLOT_SUCCESS_ONLY}" == "1" ]]; then
      plot_args+=(--success-only)
    fi
    MPLBACKEND="${MPLBACKEND:-Agg}" PYTHONPATH=. \
      python3 "${POSE_PLOTTER}" "${plot_args[@]}"
    find "${pose_result_root}/plots" -maxdepth 1 \
      -type f -name '*.png' -print | sort
  fi
fi

if contains_experiment scan_count; then
  echo
  echo "============================================================"
  echo "Exact sliced-LHS scan-count robustness"
  echo "============================================================"

  scan_result_root="${RESULT_ROOT}/scan_count"
  mkdir -p "${scan_result_root}"

  for n in ${SCAN_COUNTS}; do
    echo
    echo "[scan_count] N=${n}"
    ensure_sliced_pair "${n}" 10 50 -135 135 -60 60
    calibrate_pair \
      "${scan_result_root}/N${n}" \
      "${BASELINE_NOISE_STD_MM}"
  done

  if [[ "${MAKE_PLOTS}" == "1" ]]; then
    echo "[plot] scan-count figures -> ${scan_result_root}"
    read -r -a plot_scan_counts <<< "${SCAN_COUNTS}"
    common_plot_args=(
      --result-root "${scan_result_root}"
      --scan-counts "${plot_scan_counts[@]}"
      --dpi "${PLOT_DPI}"
      --paired-csv "${scan_result_root}/line_count_paired_differences.csv"
    )
    if [[ "${PLOT_SUCCESS_ONLY}" == "1" ]]; then
      common_plot_args+=(--success-only)
    fi

    MPLBACKEND="${MPLBACKEND:-Agg}" PYTHONPATH=. \
      python3 "${SCAN_PLOTTER}" \
        "${common_plot_args[@]}" \
        --output "${scan_result_root}/line_count_boxplots.png" \
        --paired-output "${scan_result_root}/line_count_paired_differences.png" \
        --hide-outliers \
        --no-log-translation \
        --no-log-rotation

    find "${scan_result_root}" -maxdepth 1 -type f \
      \( -name '*.png' -o -name '*paired*.csv' \) -print | sort
  fi
fi

if contains_experiment noise; then
  echo
  echo "============================================================"
  echo "Exact sliced-LHS noise robustness"
  echo "============================================================"

  noise_result_root="${RESULT_ROOT}/noise"
  mkdir -p "${noise_result_root}"

  ensure_sliced_pair "${BASELINE_TOTAL_SCANS}" 10 50 -135 135 -60 60

  for sigma in ${NOISE_LEVELS}; do
    sigma_tag="$(number_tag "${sigma}")"
    echo
    echo "[noise] sigma=${sigma} mm"
    calibrate_pair \
      "${noise_result_root}/noise_${sigma_tag}" \
      "${sigma}"
  done

  if [[ "${MAKE_PLOTS}" == "1" ]]; then
    echo "[plot] noise figures -> ${noise_result_root}"
    read -r -a plot_noise_levels <<< "${NOISE_LEVELS}"
    common_plot_args=(
      --result-root "${noise_result_root}"
      --noise-levels "${plot_noise_levels[@]}"
      --dpi "${PLOT_DPI}"
      --paired-csv "${noise_result_root}/noise_paired_differences.csv"
    )
    if [[ "${PLOT_SUCCESS_ONLY}" == "1" ]]; then
      common_plot_args+=(--success-only)
    fi

    MPLBACKEND="${MPLBACKEND:-Agg}" PYTHONPATH=. \
      python3 "${NOISE_PLOTTER}" \
        "${common_plot_args[@]}" \
        --output "${noise_result_root}/noise_boxplots.png" \
        --paired-output "${noise_result_root}/noise_paired_differences.png" \
        --hide-outliers \
        --no-log-translation \
        --no-log-rotation

    find "${noise_result_root}" -maxdepth 1 -type f \
      \( -name '*.png' -o -name '*paired*.csv' \) -print | sort
  fi
fi

echo
echo "============================================================"
echo "Exact sliced-LHS experiments completed."
echo "Datasets: ${DATASET_ROOT}"
echo "Results : ${RESULT_ROOT}"
echo "Trials  : ${MAX_TRIALS}"
echo "Mode    : ${CALIBRATION_MODE}"
echo "Feasible: ${FEASIBILITY_MODE}"
echo "============================================================"