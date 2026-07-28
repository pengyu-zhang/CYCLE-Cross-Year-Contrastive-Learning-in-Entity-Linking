#!/usr/bin/env bash
# Full experiment grid (paper main table): train on each year 2019-2022 for
# both entity sets, evaluate every model on test years 2019-2022.
#   baseline.yaml / paper.yaml : 1 seed  (controlled comparison / faithfulness)
#   default.yaml               : 3 seeds (reported as mean±std)
#
# Fail-fast: the script stops at the first failing job (set -e). Completed
# runs are skipped via .done markers, so re-running resumes where it stopped.
#
#   CANARY=1 bash scripts/run_all.sh   # one train-year per config × entity set
#   bash scripts/run_all.sh            # everything
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"

TEST_YEARS="2019,2020,2021,2022"
TRAIN_YEARS=(2019 2020 2021 2022)
ENTITY_SETS=(new continual)
DEFAULT_SEEDS=(52313 52314 52315)
BASE_SEED=52313

run_one () {
    local config="$1" entity_set="$2" year="$3" seed="$4"
    local name run_dir
    name="$(basename "$config" .yaml)"
    run_dir="outputs/${entity_set}_${year}_${name}_seed${seed}"
    if [ -f "$run_dir/.done" ]; then
        echo "== skip (done): $run_dir =="
        return 0
    fi
    echo "== train: $run_dir =="
    "$PYTHON" -m cycle.train --config "$config" \
        --set "data.entity_set=$entity_set" \
        --set "data.train_year=$year" \
        --set "run.seed=$seed"
    echo "== evaluate: $run_dir on $TEST_YEARS =="
    "$PYTHON" -m cycle.evaluate --run-dir "$run_dir" --test-years "$TEST_YEARS"
    touch "$run_dir/.done"
}

if [ "${CANARY:-0}" = "1" ]; then
    for entity_set in "${ENTITY_SETS[@]}"; do
        for config in configs/baseline.yaml configs/paper.yaml configs/default.yaml; do
            run_one "$config" "$entity_set" 2019 "$BASE_SEED"
        done
    done
    echo "== canary finished =="
    exit 0
fi

for entity_set in "${ENTITY_SETS[@]}"; do
    for year in "${TRAIN_YEARS[@]}"; do
        run_one configs/baseline.yaml "$entity_set" "$year" "$BASE_SEED"
        run_one configs/paper.yaml    "$entity_set" "$year" "$BASE_SEED"
        for seed in "${DEFAULT_SEEDS[@]}"; do
            run_one configs/default.yaml "$entity_set" "$year" "$seed"
        done
    done
done
echo "== full grid finished =="
