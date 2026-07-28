#!/usr/bin/env bash
# Train one model.
#   bash scripts/train.sh [config] [entity_set] [train_year] [seed] [extra --set overrides...]
# Defaults: configs/default.yaml, new, 2019, 52313
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"

CONFIG="${1:-configs/default.yaml}"
ENTITY_SET="${2:-new}"
TRAIN_YEAR="${3:-2019}"
SEED="${4:-52313}"
shift $(( $# > 4 ? 4 : $# ))

"$PYTHON" -m cycle.train --config "$CONFIG" \
    --set "data.entity_set=$ENTITY_SET" \
    --set "data.train_year=$TRAIN_YEAR" \
    --set "run.seed=$SEED" \
    "$@"
