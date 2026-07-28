#!/usr/bin/env bash
# Download GCL-TempEL (Zenodo, with GitHub-Release mirror fallback), verify
# its MD5, lay it out under data/raw/ and build the processed per-year
# graphs used by configs/paper.yaml and configs/default.yaml.
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
export PYTHONPATH="src${PYTHONPATH:+:$PYTHONPATH}"

"$PYTHON" -m cycle.prepare_data --data-root data "$@"
"$PYTHON" -m cycle.prepare_graphs --data-root data
echo "== data ready =="
