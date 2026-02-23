#!/usr/bin/env bash
# Validate a checkpoint + telemetry pair before/after RunPod resume operations.
set -euo pipefail

CHECKPOINT=${1:-}
LOG_PATH=${2:-}

if [[ -z "${CHECKPOINT}" || -z "${LOG_PATH}" ]]; then
  echo "Usage: $0 <checkpoint.pt> <metrics.json>"
  exit 1
fi

uv run python scripts/checkpoint/verify.py --checkpoint "${CHECKPOINT}"
uv run python scripts/checks/validate_fidelity_telemetry.py --log "${LOG_PATH}"

echo "[runpod-validate] checkpoint + telemetry validation passed"
