#!/usr/bin/env bash
set -euo pipefail

# Run every main experiment with:
#   alternating linear calibration -> six-parameter nonlinear refit
#
# Existing datasets are reused. Results and logs are written below
# result_refit/ by default.
#
# Full run:
#   bash main/run_all_main_experiments_refit.sh
#
# Quick orchestration check:
#   DRY_RUN=1 bash main/run_all_main_experiments_refit.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export CALIBRATION_MODE="iterative_refit_nonlinear"
export MASTER_RESULT_ROOT="${MASTER_RESULT_ROOT:-result_refit}"
export MASTER_LOG_ROOT="${MASTER_LOG_ROOT:-${MASTER_RESULT_ROOT}/all_main_experiment_logs}"

exec bash "${SCRIPT_DIR}/run_all_main_experiments.sh" "$@"
