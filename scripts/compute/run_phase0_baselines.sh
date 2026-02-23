#!/usr/bin/env bash
# Canonical Phase 0 baseline runbook for paper-faithful comparison runs.
set -euo pipefail

DEVICE=${DEVICE:-cuda:0}
TRAIN_STEPS=${TRAIN_STEPS:-2000}
TRAIN_LOG_INTERVAL=${TRAIN_LOG_INTERVAL:-}
TRAIN_CHECKPOINT_INTERVAL=${TRAIN_CHECKPOINT_INTERVAL:-}
TRAIN_EXTRA_OVERRIDES=${TRAIN_EXTRA_OVERRIDES:-}
TOKENIZER_PATH=${TOKENIZER_PATH:-artifacts/tokenizer/refinedweb_mix/spm_32000_unigram.model}
SEGMENTS_YAML=${SEGMENTS_YAML:-configs/data/continual_segments_sample.yaml}
ZEROSHOT_MAX_SAMPLES=${ZEROSHOT_MAX_SAMPLES:-256}
NIAH_CONTEXTS=${NIAH_CONTEXTS:-"2048 4096 8192 16384"}
NIAH_SAMPLES=${NIAH_SAMPLES:-8}
CONT_BATCH=${CONT_BATCH:-4}
CONT_MAX_BATCHES=${CONT_MAX_BATCHES:-20}
PASSKEY_SAMPLES=${PASSKEY_SAMPLES:-64}
PASSKEY_FILLER=${PASSKEY_FILLER:-256}
PG19_SAMPLES=${PG19_SAMPLES:-32}

SELFMOD_CONFIG_NAME=${SELFMOD_CONFIG_NAME:-pilot_selfmod_paper_faithful}
ATTENTION_CONFIG_NAME=${ATTENTION_CONFIG_NAME:-pilot_attention_paper_faithful}
SELFMOD_CONFIG_PATH=${SELFMOD_CONFIG_PATH:-configs/pilot_selfmod_paper_faithful.yaml}
ATTENTION_CONFIG_PATH=${ATTENTION_CONFIG_PATH:-configs/pilot_attention_paper_faithful.yaml}

SELFMOD_CKPT_DIR=${SELFMOD_CKPT_DIR:-artifacts/checkpoints/phase0_selfmod}
ATTENTION_CKPT_DIR=${ATTENTION_CKPT_DIR:-artifacts/checkpoints/phase0_attention}
SELFMOD_LOG=${SELFMOD_LOG:-logs/phase0_selfmod_metrics.json}
ATTENTION_LOG=${ATTENTION_LOG:-logs/phase0_attention_metrics.json}

run_eval_suite() {
  local tag="$1"
  local config_path="$2"
  local checkpoint="$3"

  local niah_args=()
  IFS=' ' read -r -a _contexts <<< "${NIAH_CONTEXTS}"
  for ctx in "${_contexts[@]}"; do
    niah_args+=(--context-lengths "${ctx}")
  done

  UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run python scripts/eval/zeroshot.py \
    --config "${config_path}" \
    --checkpoint "${checkpoint}" \
    --tokenizer-path "${TOKENIZER_PATH}" \
    --tasks all \
    --max-samples "${ZEROSHOT_MAX_SAMPLES}" \
    --device "${DEVICE}" \
    --output "eval/zeroshot_${tag}.json"

  UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run python scripts/eval/niah.py \
    --config "${config_path}" \
    --checkpoint "${checkpoint}" \
    --tokenizer-path "${TOKENIZER_PATH}" \
    "${niah_args[@]}" \
    --samples-per-length "${NIAH_SAMPLES}" \
    --device "${DEVICE}" \
    --output "eval/niah_${tag}.json"

  UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run python scripts/eval/continual.py \
    --config "${config_path}" \
    --checkpoints "${checkpoint}" \
    --segments-yaml "${SEGMENTS_YAML}" \
    --tokenizer-path "${TOKENIZER_PATH}" \
    --batch-size "${CONT_BATCH}" \
    --max-batches "${CONT_MAX_BATCHES}" \
    --device "${DEVICE}" \
    --output "eval/continual_${tag}.json"

  UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run python scripts/eval/passkey.py \
    --config "${config_path}" \
    --checkpoint "${checkpoint}" \
    --tokenizer-path "${TOKENIZER_PATH}" \
    --samples "${PASSKEY_SAMPLES}" \
    --filler-sentences "${PASSKEY_FILLER}" \
    --device "${DEVICE}" \
    --output "eval/passkey_${tag}.json"

  UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run python scripts/eval/pg19_perplexity.py \
    --config "${config_path}" \
    --checkpoint "${checkpoint}" \
    --tokenizer-path "${TOKENIZER_PATH}" \
    --max-samples "${PG19_SAMPLES}" \
    --device "${DEVICE}" \
    --output "eval/pg19_${tag}.json"
}

mkdir -p artifacts/checkpoints eval logs

train_overrides=()
if [[ -n "${TRAIN_LOG_INTERVAL}" ]]; then
  train_overrides+=("train.log_interval=${TRAIN_LOG_INTERVAL}")
fi
if [[ -n "${TRAIN_CHECKPOINT_INTERVAL}" ]]; then
  train_overrides+=("train.checkpoint.save_interval=${TRAIN_CHECKPOINT_INTERVAL}")
fi
if [[ -n "${TRAIN_EXTRA_OVERRIDES}" ]]; then
  IFS=' ' read -r -a _extra <<< "${TRAIN_EXTRA_OVERRIDES}"
  train_overrides+=("${_extra[@]}")
fi
TRAIN_OVERRIDES_STRING="${train_overrides[*]}"

SELFMOD_CMD="uv run python train.py --config-name ${SELFMOD_CONFIG_NAME} train.steps=${TRAIN_STEPS} train.device=${DEVICE} logging.enabled=true logging.backend=json logging.path=${SELFMOD_LOG} logging.run_name=phase0-selfmod train.checkpoint.dir=${SELFMOD_CKPT_DIR} ${TRAIN_OVERRIDES_STRING}"
ATTENTION_CMD="uv run python train.py --config-name ${ATTENTION_CONFIG_NAME} train.steps=${TRAIN_STEPS} train.device=${DEVICE} logging.enabled=true logging.backend=json logging.path=${ATTENTION_LOG} logging.run_name=phase0-attention train.checkpoint.dir=${ATTENTION_CKPT_DIR} ${TRAIN_OVERRIDES_STRING}"

echo "[phase0] training selfmod baseline"
eval "${SELFMOD_CMD}"

echo "[phase0] training attention baseline"
eval "${ATTENTION_CMD}"

SELFMOD_CKPT=$(ls -1t "${SELFMOD_CKPT_DIR}"/step_*.pt | head -n 1)
ATTENTION_CKPT=$(ls -1t "${ATTENTION_CKPT_DIR}"/step_*.pt | head -n 1)

echo "[phase0] selfmod checkpoint: ${SELFMOD_CKPT}"
echo "[phase0] attention checkpoint: ${ATTENTION_CKPT}"

echo "[phase0] validating checkpoint sidecars"
uv run python scripts/checkpoint/verify.py --checkpoint "${SELFMOD_CKPT}"
uv run python scripts/checkpoint/verify.py --checkpoint "${ATTENTION_CKPT}"

echo "[phase0] validating telemetry"
uv run python scripts/checks/validate_fidelity_telemetry.py --log "${SELFMOD_LOG}" --log "${ATTENTION_LOG}"

echo "[phase0] running eval suites"
run_eval_suite "phase0_selfmod" "${SELFMOD_CONFIG_PATH}" "${SELFMOD_CKPT}"
run_eval_suite "phase0_attention" "${ATTENTION_CONFIG_PATH}" "${ATTENTION_CKPT}"

echo "[phase0] packaging baseline bundle"
uv run python scripts/checkpoint/package_phase0_baselines.py \
  --selfmod-checkpoint "${SELFMOD_CKPT}" \
  --attention-checkpoint "${ATTENTION_CKPT}" \
  --selfmod-log "${SELFMOD_LOG}" \
  --attention-log "${ATTENTION_LOG}" \
  --selfmod-train-command "${SELFMOD_CMD}" \
  --attention-train-command "${ATTENTION_CMD}"

echo "[phase0] complete"
