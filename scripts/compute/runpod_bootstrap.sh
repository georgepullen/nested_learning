#!/usr/bin/env bash
# Bootstrap a RunPod instance for this repository.
set -euo pipefail

REPO_DIR=${1:-/workspace/nested_learning}
PYTHON_VERSION=${PYTHON_VERSION:-3.12}

if [[ ! -d "${REPO_DIR}" ]]; then
  echo "[runpod-bootstrap] repo dir not found: ${REPO_DIR}"
  exit 1
fi

cd "${REPO_DIR}"
export UV_CACHE_DIR=${UV_CACHE_DIR:-/workspace/.cache/uv}
export UV_LINK_MODE=${UV_LINK_MODE:-copy}

mkdir -p "${UV_CACHE_DIR}" logs artifacts eval

uv python install "${PYTHON_VERSION}"
uv sync --python "${PYTHON_VERSION}" --all-extras --dev
uv run python - <<'PY'
import torch
print(f"torch={torch.__version__} cuda={torch.version.cuda} cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"device0={torch.cuda.get_device_name(0)}")
PY

echo "[runpod-bootstrap] complete"
