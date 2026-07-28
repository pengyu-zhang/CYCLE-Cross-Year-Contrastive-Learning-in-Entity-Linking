#!/usr/bin/env bash
# Evaluate a trained run on one or more test years.
#   bash scripts/evaluate.sh <run_dir> [test_years] [extra args...]
# Example:
#   bash scripts/evaluate.sh outputs/new_2019_default_seed52313 2019,2020,2021,2022
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"

RUN_DIR="${1:?usage: evaluate.sh <run_dir> [test_years]}"
TEST_YEARS="${2:-2019,2020,2021,2022}"
shift $(( $# > 2 ? 2 : $# ))

"$PYTHON" -m cycle.evaluate --run-dir "$RUN_DIR" --test-years "$TEST_YEARS" "$@"
