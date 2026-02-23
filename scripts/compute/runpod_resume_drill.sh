#!/usr/bin/env bash
# Forced stop/resume drill using synthetic pilot_smoke config.
set -euo pipefail

DEVICE=${DEVICE:-cuda:0}
CKPT_DIR=${CKPT_DIR:-artifacts/checkpoints/runpod_resume_drill}
LOG_PHASE1=${LOG_PHASE1:-logs/runpod_resume_drill_phase1.json}
LOG_PHASE2=${LOG_PHASE2:-logs/runpod_resume_drill_phase2.json}

mkdir -p "${CKPT_DIR}" logs

echo "[resume-drill] phase 1: create resumable checkpoint"
uv run python train.py --config-name pilot_smoke \
  train.device="${DEVICE}" \
  train.steps=6 \
  train.checkpoint.enable=true \
  train.checkpoint.dir="${CKPT_DIR}" \
  train.checkpoint.save_interval=3 \
  train.checkpoint.save_last=true \
  logging.enabled=true \
  logging.backend=json \
  logging.path="${LOG_PHASE1}"

RESUME_CKPT=$(ls -1t "${CKPT_DIR}"/step_*.pt | head -n 1)
if [[ -z "${RESUME_CKPT}" ]]; then
  echo "[resume-drill] no checkpoint produced in ${CKPT_DIR}"
  exit 1
fi

echo "[resume-drill] checkpoint selected: ${RESUME_CKPT}"

echo "[resume-drill] phase 2: resume and continue"
uv run python train.py --config-name pilot_smoke \
  train.device="${DEVICE}" \
  train.steps=9 \
  train.checkpoint.enable=true \
  train.checkpoint.dir="${CKPT_DIR}" \
  train.checkpoint.save_interval=9 \
  train.checkpoint.save_last=true \
  +train.checkpoint.resume_path="${RESUME_CKPT}" \
  logging.enabled=true \
  logging.backend=json \
  logging.path="${LOG_PHASE2}"

FINAL_CKPT="${CKPT_DIR}/step_000009.pt"
if [[ ! -f "${FINAL_CKPT}" ]]; then
  echo "[resume-drill] expected final checkpoint missing: ${FINAL_CKPT}"
  exit 1
fi

uv run python scripts/checkpoint/verify.py --checkpoint "${RESUME_CKPT}"
uv run python scripts/checkpoint/verify.py --checkpoint "${FINAL_CKPT}"
uv run python scripts/checks/validate_fidelity_telemetry.py --log "${LOG_PHASE1}" --log "${LOG_PHASE2}"

echo "[resume-drill] success"
