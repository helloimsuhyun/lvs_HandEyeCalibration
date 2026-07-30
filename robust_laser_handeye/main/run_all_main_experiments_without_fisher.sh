#!/usr/bin/env bash
set -euo pipefail

# Run every main experiment except the Fisher-vs-random policy comparison.
#
# Included:
#   1) initialization standard
#   2) fixed-line noise
#   3) fixed-noise line count
#   4) pose diversity
#
# All common settings and per-experiment logs are managed by
# run_all_main_experiments.sh. The default standardized initialization is
# norm-bounded 100 mm translation and axis-angle-bounded 15 deg rotation.
#
# Full run:
#   bash main/run_all_main_experiments_without_fisher.sh
#
# Inspect commands without running:
#   DRY_RUN=1 bash main/run_all_main_experiments_without_fisher.sh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

export RUN_FISHER_RANDOM=0
exec bash "${SCRIPT_DIR}/run_all_main_experiments.sh" "$@"
