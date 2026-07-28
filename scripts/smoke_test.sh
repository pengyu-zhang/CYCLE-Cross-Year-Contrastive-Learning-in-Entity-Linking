#!/usr/bin/env bash
# End-to-end smoke test (a few minutes): tiny training runs for every config
# plus a capped evaluation, exercising the full pipeline.
# Requires prepared data (scripts/prepare_data.sh).
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"

OUT="outputs/smoke"
rm -rf "$OUT"

for CONFIG in baseline paper default; do
    echo "== smoke: train ($CONFIG) =="
    "$PYTHON" -m cycle.train --config "configs/$CONFIG.yaml" \
        --set data.entity_set=new \
        --set data.train_year=2019 \
        --set data.debug_n=128 \
        --set data.valid_n=64 \
        --set run.output_dir="$OUT" \
        --set train.eval_interval=4
    echo "== smoke: evaluate ($CONFIG) =="
    "$PYTHON" -m cycle.evaluate \
        --run-dir "$OUT/new_2019_${CONFIG}_seed52313" \
        --test-years 2020 --limit 256
done

echo "== smoke test passed =="
