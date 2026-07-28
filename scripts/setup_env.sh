#!/usr/bin/env bash
# Install dependencies into the current Python environment.
#   CUDA_TAG=cu132 bash scripts/setup_env.sh     # CUDA build (default)
#   CUDA_TAG=cpu   bash scripts/setup_env.sh     # CPU-only build
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-python}"
CUDA_TAG="${CUDA_TAG:-cu132}"

echo "== installing torch (${CUDA_TAG}) =="
if [ "${CUDA_TAG}" = "cpu" ]; then
    "$PYTHON" -m pip install torch
else
    "$PYTHON" -m pip install torch --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"
fi

echo "== installing remaining requirements =="
"$PYTHON" -m pip install -r requirements.txt

"$PYTHON" - <<'EOF'
import torch
print(f"torch {torch.__version__}, cuda available: {torch.cuda.is_available()}")
EOF
echo "== done =="
