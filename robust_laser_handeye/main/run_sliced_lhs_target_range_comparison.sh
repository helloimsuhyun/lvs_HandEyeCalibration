#!/usr/bin/env bash
set -euo pipefail

# Moderate-pose target-position comparison:
#   fixed U/V=0, U/V=+/-10 mm, U/V=+/-20 mm, and U/V=+/-50 mm.

TARGET_HALF_WIDTHS_MM="${TARGET_HALF_WIDTHS_MM:-0 10 20 50}"

TOTAL_SCANS="${TOTAL_SCANS:-108}"
NOISE_STD_MM="${NOISE_STD_MM:-0.20}"
MAX_TRIALS="${MAX_TRIALS:-100}"
GENERATION_SEED="${GENERATION_SEED:-17}"
CALIBRATION_SEED="${CALIBRATION_SEED:-1701}"

# Keep pose constraints fixed at the moderate level.
TILT_MIN_DEG="${TILT_MIN_DEG:-10}"
TILT_MAX_DEG="${TILT_MAX_DEG:-50}"
AZIMUTH_MIN_DEG="${AZIMUTH_MIN_DEG:--135}"
AZIMUTH_MAX_DEG="${AZIMUTH_MAX_DEG:-135}"
ROLL_MIN_DEG="${ROLL_MIN_DEG:--60}"
ROLL_MAX_DEG="${ROLL_MAX_DEG:-60}"

PROFILE_POINTS="${PROFILE_POINTS:-100}"
PROFILE_HALF_WIDTH_MM="${PROFILE_HALF_WIDTH_MM:-25}"
TANGENT_RANGE_MM="${TANGENT_RANGE_MM:-100}"
PROFILE_DEPTH_MIN_MM="${PROFILE_DEPTH_MIN_MM:-60}"
PROFILE_DEPTH_MAX_MM="${PROFILE_DEPTH_MAX_MM:-150}"
CENTER_DEPTH_MIN_MM="${CENTER_DEPTH_MIN_MM:-60}"
CENTER_DEPTH_MAX_MM="${CENTER_DEPTH_MAX_MM:-150}"
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

DATASET_ROOT="${DATASET_ROOT:-dataset/sliced_lhs_target_range_moderate_mc100_nonlinear}"
RESULT_ROOT="${RESULT_ROOT:-results/sliced_lhs_target_range_moderate_mc100_nonlinear}"

FORCE_REGENERATE="${FORCE_REGENERATE:-0}"
FORCE_RERUN="${FORCE_RERUN:-0}"
AUTO_REPAIR_INCOMPLETE_RESULTS="${AUTO_REPAIR_INCOMPLETE_RESULTS:-1}"
MAKE_PLOTS="${MAKE_PLOTS:-1}"
PLOT_SUCCESS_ONLY="${PLOT_SUCCESS_ONLY:-0}"
PLOT_DPI="${PLOT_DPI:-300}"

GENERATOR="main/generate_sliced_lhs_plane_comparison.py"
CALIBRATOR="main/calibrate.py"
VALIDATOR="main/validate_experiment_artifact.py"
PLOTTER="main/make_plot/plot_target_range_comparison.py"

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
if [[ "${MAKE_PLOTS}" == "1" && ! -f "${PLOTTER}" ]]; then
  echo "ERROR: plotter not found: ${PLOTTER}" >&2
  exit 1
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

if ! [[ "${MAX_TRIALS}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: MAX_TRIALS must be positive." >&2
  exit 1
fi
if ! [[ "${TOTAL_SCANS}" =~ ^[1-9][0-9]*$ ]] \
  || (( TOTAL_SCANS < 9 || TOTAL_SCANS % 3 != 0 )); then
  echo "ERROR: TOTAL_SCANS must be >=9 and divisible by 3." >&2
  exit 1
fi
for half_width in ${TARGET_HALF_WIDTHS_MM}; do
  if ! [[ "${half_width}" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
    echo "ERROR: target half-width must be non-negative: ${half_width}" >&2
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

mkdir -p "${DATASET_ROOT}" "${RESULT_ROOT}"

number_tag() {
  printf '%s' "$1" | sed -e 's/-/m/g' -e 's/\./p/g'
}

negative_half_width() {
  local value="$1"
  if [[ "${value}" =~ ^0+([.]0+)?$ ]]; then
    printf '0'
  else
    printf -- '-%s' "${value}"
  fi
}

CURRENT_DATASET_DIR=""
CURRENT_SINGLE_COLLECTION=""
CURRENT_THREE_COLLECTION=""

ensure_sliced_pair() {
  local half_width="$1"
  local lower
  local level_name
  lower="$(negative_half_width "${half_width}")"
  level_name="uv_$(number_tag "${half_width}")mm"

  local pose_tag
  pose_tag="t$(number_tag "${TILT_MIN_DEG}")_$(number_tag "${TILT_MAX_DEG}")"
  pose_tag+="_a$(number_tag "${AZIMUTH_MIN_DEG}")_$(number_tag "${AZIMUTH_MAX_DEG}")"
  pose_tag+="_r$(number_tag "${ROLL_MIN_DEG}")_$(number_tag "${ROLL_MAX_DEG}")"

  local dataset_dir="${DATASET_ROOT}/${level_name}/N${TOTAL_SCANS}_${pose_tag}"
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
    --total-scans "${TOTAL_SCANS}"
    --profile-points "${PROFILE_POINTS}"
    --profile-half-width-mm "${PROFILE_HALF_WIDTH_MM}"
    --tangent-range-mm "${TANGENT_RANGE_MM}"
    --profile-depth-range-mm "${PROFILE_DEPTH_MIN_MM}" "${PROFILE_DEPTH_MAX_MM}"
    --center-depth-range-mm "${CENTER_DEPTH_MIN_MM}" "${CENTER_DEPTH_MAX_MM}"
    --uniform-target-u-range-mm "${lower}" "${half_width}"
    --uniform-target-v-range-mm "${lower}" "${half_width}"
    --uniform-view-tilt-range-deg "${TILT_MIN_DEG}" "${TILT_MAX_DEG}"
    --uniform-view-azimuth-range-deg "${AZIMUTH_MIN_DEG}" "${AZIMUTH_MAX_DEG}"
    --uniform-sensor-roll-range-deg "${ROLL_MIN_DEG}" "${ROLL_MAX_DEG}"
    --max-design-attempts "${MAX_DESIGN_ATTEMPTS}"
    --feasibility-mode "${FEASIBILITY_MODE}"
  )

  # Requiring every target range during design generation keeps all six
  # normalized sliced-LHS dimensions matched, even when output U/V is fixed.
  for shared_half_width in ${TARGET_HALF_WIDTHS_MM}; do
    shared_lower="$(negative_half_width "${shared_half_width}")"
    generator_args+=(
      --shared-feasibility-target-range-mm
      "${shared_lower}" "${shared_half_width}"
      "${shared_lower}" "${shared_half_width}"
    )
  done

  echo "[generate/resume] U/V=${lower}..${half_width} mm: ${dataset_dir}"
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
        raise SystemExit(f"collection is not complete: {path}")
    if int(payload.get("completed_trials", -1)) != expected:
        raise SystemExit(f"collection trial mismatch: {path}")
print(f"[dataset-check] complete: {expected} trials")
PY

  CURRENT_DATASET_DIR="${dataset_dir}"
  CURRENT_SINGLE_COLLECTION="${single_collection}"
  CURRENT_THREE_COLLECTION="${three_collection}"
}

validate_result() {
  local summary="$1"
  local collection="$2"

  PYTHONPATH=. python3 "${VALIDATOR}" result \
    --mode "${CALIBRATION_MODE}" \
    "${NONLINEAR_ARGS[@]}" \
    --summary "${summary}" \
    --collection "${collection}" \
    --trials "${MAX_TRIALS}" \
    --seed "${CALIBRATION_SEED}" \
    --noise-axis "${NOISE_AXIS}" \
    --noise-std-mm "${NOISE_STD_MM}" \
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
    if validate_result "${output_dir}/summary.json" "${collection}"; then
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

  echo "[${method_label}] calibrating: mode=${CALIBRATION_MODE}, noise=${NOISE_STD_MM}"
  PYTHONPATH=. python3 "${CALIBRATOR}" \
    --collection "${collection}" \
    --output-dir "${output_dir}" \
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

  validate_result "${output_dir}/summary.json" "${collection}"
}

echo
echo "============================================================"
echo "Target U/V range comparison with moderate pose constraints"
echo "============================================================"

dataset_dirs=()
level_names=()
level_labels=()

for half_width in ${TARGET_HALF_WIDTHS_MM}; do
  level_name="uv_$(number_tag "${half_width}")mm"
  lower="$(negative_half_width "${half_width}")"
  if [[ "${lower}" == "0" ]]; then
    level_label="0 mm (fixed)"
  else
    level_label="+/-${half_width} mm"
  fi

  echo
  echo "[target_range] ${level_label}"
  ensure_sliced_pair "${half_width}"
  dataset_dirs+=("${CURRENT_DATASET_DIR}")
  level_names+=("${level_name}")
  level_labels+=("${level_label}")

  run_calibration \
    single "${CURRENT_SINGLE_COLLECTION}" \
    "${RESULT_ROOT}/${level_name}/single_plane"
  run_calibration \
    three "${CURRENT_THREE_COLLECTION}" \
    "${RESULT_ROOT}/${level_name}/three_plane"
done

PYTHONPATH=. python3 - "${dataset_dirs[@]}" <<'PY'
from pathlib import Path
import json
import sys

manifests = [
    json.loads((Path(path) / "comparison_manifest.json").read_text(encoding="utf-8"))
    for path in sys.argv[1:]
]
reference = [trial["normalized_design_sha256"] for trial in manifests[0]["trials"]]
for index, manifest in enumerate(manifests[1:], start=1):
    current = [trial["normalized_design_sha256"] for trial in manifest["trials"]]
    if current != reference:
        raise SystemExit(
            f"normalized sliced-LHS designs differ across target ranges: index={index}"
        )
print(
    f"[target-range audit] identical normalized sliced designs across "
    f"{len(manifests)} ranges and {len(reference)} trials"
)
PY

if [[ "${MAKE_PLOTS}" == "1" ]]; then
  plot_args=(
    --result-root "${RESULT_ROOT}"
    --levels "${level_names[@]}"
    --level-labels "${level_labels[@]}"
    --output-dir "${RESULT_ROOT}/plots"
    --outlier-thresholds-mm 1 2
    --dpi "${PLOT_DPI}"
  )
  if [[ "${PLOT_SUCCESS_ONLY}" == "1" ]]; then
    plot_args+=(--success-only)
  fi
  MPLBACKEND="${MPLBACKEND:-Agg}" PYTHONPATH=. \
    python3 "${PLOTTER}" "${plot_args[@]}"
  find "${RESULT_ROOT}/plots" -maxdepth 1 \
    -type f \( -name '*.png' -o -name '*.csv' \) -print | sort
fi

echo
echo "============================================================"
echo "Target-range comparison completed."
echo "Datasets: ${DATASET_ROOT}"
echo "Results : ${RESULT_ROOT}"
echo "Trials  : ${MAX_TRIALS}"
echo "Pose    : moderate"
echo "============================================================"
