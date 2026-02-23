#!/usr/bin/env bash
set -euo pipefail

TOKENIZER_MODEL=${1:-artifacts/tokenizer/refinedweb_mix/spm_32000_unigram.model}
TOKENIZER_DIR="$(dirname "${TOKENIZER_MODEL}")"
RPJ_DATASET=${RPJ_DATASET:-cerebras/SlimPajama-627B}
RPJ_DATASET_CANDIDATES=${RPJ_DATASET_CANDIDATES:-MBZUAI-LLM/SlimPajama-627B-DC,DKYoon/SlimPajama-6B,gmongaras/SlimPajama-627B_Reupload}

if [[ ! -f "data/filtered/refinedweb_en_sample.txt" ]]; then
  echo "[Data] Creating filtered RefinedWeb sample"
  uv run python scripts/data/filter_corpus.py \
    --dataset HuggingFaceFW/fineweb \
    "--subset=sample-10BT" \
    --split train \
    --text-column text \
    --target-lang en \
    --lang-threshold 0.85 \
    --min-chars 200 \
    --max-chars 8000 \
    --limit 2000 \
    --output-path data/filtered/refinedweb_en_sample.txt \
    --force-exit
fi

if [[ ! -f "data/filtered/wikipedia_en_sample.txt" ]]; then
  echo "[Data] Creating filtered Wikipedia sample"
  uv run python scripts/data/filter_corpus.py \
    --dataset wikimedia/wikipedia \
    "--subset=20231101.en" \
    --split train \
    --text-column text \
    --target-lang en \
    --lang-threshold 0.85 \
    --min-chars 200 \
    --max-chars 8000 \
    --limit 1000 \
    --output-path data/filtered/wikipedia_en_sample.txt \
    --force-exit
fi

if [[ ! -f "data/filtered/c4_en_sample.txt" ]]; then
  echo "[Data] Creating filtered C4 sample"
  uv run python scripts/data/filter_corpus.py \
    --dataset allenai/c4 --subset en --split train \
    --text-column text --target-lang en --lang-threshold 0.85 \
    --min-chars 200 --max-chars 8000 --limit 1000 \
    --output-path data/filtered/c4_en_sample.txt --force-exit
fi

if [[ ! -f "data/filtered/redpajama_en_sample.txt" ]]; then
  echo "[Data] Creating filtered SlimPajama sample"
  IFS=',' read -r -a _rpj_fallbacks <<< "${RPJ_DATASET_CANDIDATES}"
  _rpj_sources=("${RPJ_DATASET}")
  for _fallback in "${_rpj_fallbacks[@]}"; do
    _fallback="${_fallback//[[:space:]]/}"
    [[ -n "${_fallback}" ]] && _rpj_sources+=("${_fallback}")
  done

  _rpj_ok=0
  for _dataset in "${_rpj_sources[@]}"; do
    echo "[Data] SlimPajama candidate: ${_dataset}"
    rm -f data/filtered/redpajama_en_sample.txt
    if uv run python scripts/data/filter_corpus.py \
      "--dataset=${_dataset}" \
      --split train \
      --text-column text \
      --target-lang en \
      --lang-threshold 0.85 \
      --min-chars 200 \
      --max-chars 8000 \
      --limit 1000 \
      --output-path data/filtered/redpajama_en_sample.txt \
      --force-exit; then
      _rpj_ok=1
      break
    fi
  done

  if [[ "${_rpj_ok}" != "1" ]]; then
    echo "[Data] ERROR: unable to fetch SlimPajama sample from any configured source."
    echo "[Data] Tried: ${_rpj_sources[*]}"
    exit 1
  fi
fi

if [[ ! -f "data/filtered/code_en_sample.txt" ]]; then
  echo "[Data] Creating filtered code sample"
  uv run python scripts/data/filter_corpus.py \
    --dataset codeparrot/codeparrot-clean-train --split train \
    --text-column content --target-lang en --lang-threshold 0.5 \
    --min-chars 200 --max-chars 12000 --limit 1000 \
    --output-path data/filtered/code_en_sample.txt --force-exit
fi

if [[ ! -f "${TOKENIZER_MODEL}" ]]; then
  echo "[Data] Training tokenizer (sample) -> ${TOKENIZER_DIR}"
  uv run python scripts/data/train_tokenizer.py \
    --manifest configs/data/refinedweb_mixture_filtered.yaml \
    --vocab-size 32000 \
    --no-hard-vocab-limit \
    --output-dir "${TOKENIZER_DIR}" \
    --log-file data/mixtures/refinedweb_mix_tokenizer_sample.json
fi

echo "[Data] Sharding filtered samples"
uv run python scripts/data/process_mixture.py \
  configs/data/refinedweb_mixture_filtered.yaml \
  --tokenizer-path ${TOKENIZER_MODEL} \
  --log-file data/mixtures/refinedweb_mix_filtered_shards.json

echo "[Data] Sample pipeline complete"
