#!/usr/bin/env bash
set -euo pipefail

STAMP="$(date -u +%Y%m%dT%H%M%SZ)_fullmatrix3090"
OUT="logs/${STAMP}"
mkdir -p "${OUT}"

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

if command -v uv >/dev/null 2>&1; then
  RUN_PY=(uv run python)
elif [[ -x ".venv/bin/python" ]]; then
  RUN_PY=(.venv/bin/python)
else
  RUN_PY=(python)
fi

echo "[run] output_dir=${OUT}"
echo "[run] python_runner=${RUN_PY[*]}"

# Baseline B=1
"${RUN_PY[@]}" train.py --config-name bench_small_b1 \
  train.device=cuda:0 \
  train.steps=2000 \
  train.grad_accum_steps=1 \
  train.online_chunk_size=2 \
  logging.path="${OUT}/b1_s2000.json" \
  logging.run_name="${STAMP}_b1_s2000" \
  | tee "${OUT}/b1_s2000.log"

# B=1 with grad accumulation = 4 (same tokens, ~4x fewer optimizer steps)
"${RUN_PY[@]}" train.py --config-name bench_small_b1 \
  train.device=cuda:0 \
  train.steps=2000 \
  train.grad_accum_steps=4 \
  train.online_chunk_size=2 \
  logging.path="${OUT}/b1_accum4_s2000.json" \
  logging.run_name="${STAMP}_b1_accum4_s2000" \
  | tee "${OUT}/b1_accum4_s2000.log"

# B=4 list-mode LR sweep
for LR in 0.002 0.004 0.006; do
  "${RUN_PY[@]}" train.py --config-name bench_small_isolated_b4 \
    train.device=cuda:0 \
    train.steps=500 \
    train.fast_state_batch_mode=per_sample_list \
    train.online_chunk_size=2 \
    optim.lr="${LR}" \
    logging.path="${OUT}/b4_list_lr${LR}_s500.json" \
    logging.run_name="${STAMP}_b4_list_lr${LR}_s500" \
    | tee "${OUT}/b4_list_lr${LR}_s500.log"
done

"${RUN_PY[@]}" scripts/analyze_token_normalized.py \
  --runs \
    "${OUT}/b1_s2000.json" \
    "${OUT}/b1_accum4_s2000.json" \
    "${OUT}/b4_list_lr0.002_s500.json" \
    "${OUT}/b4_list_lr0.004_s500.json" \
    "${OUT}/b4_list_lr0.006_s500.json" \
  --labels b1 b1_accum4 b4_lr2e-3 b4_lr4e-3 b4_lr6e-3 \
  --out_dir "${OUT}/token_analysis"

echo "[done] ${OUT}"
