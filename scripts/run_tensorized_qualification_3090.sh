#!/usr/bin/env bash
set -euo pipefail

STAMP="$(date -u +%Y%m%dT%H%M%SZ)_tensorized_qual_3090"
OUT="logs/${STAMP}"
mkdir -p "${OUT}"

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

if command -v uv >/dev/null 2>&1; then
  RUN_PY=(uv run python)
elif [[ -x ".venv/bin/python" ]]; then
  RUN_PY=(.venv/bin/python)
else
  RUN_PY=(python)
fi

echo "[env] output_dir=${OUT}" | tee "${OUT}/env.log"
echo "[env] python_runner=${RUN_PY[*]}" | tee -a "${OUT}/env.log"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi | tee -a "${OUT}/env.log"
fi
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  git rev-parse HEAD | sed 's/^/[env] git_sha=/' | tee -a "${OUT}/env.log"
else
  echo "[env] git_sha=unknown (no .git metadata in this workspace copy)" | tee -a "${OUT}/env.log"
fi
"${RUN_PY[@]}" -c "import torch; print('[env] torch', torch.__version__, 'cuda', torch.version.cuda)" \
  | tee -a "${OUT}/env.log"

echo "[tests] running tensorized qualification tests..." | tee "${OUT}/tests.log"
"${RUN_PY[@]}" -m pytest -q \
  tests/test_teach_signal_batch_equivalence.py \
  tests/test_fast_state_isolation.py \
  tests/test_batched_equals_sequential.py \
  tests/test_checkpoint_resume_parity.py \
  tests/test_tensorized_cms_matches_listmode.py \
  tests/test_tensorized_cms_multilevel_matches_listmode.py \
  tests/test_tensorized_cms_hybrid_wiring_matches_listmode.py \
  | tee -a "${OUT}/tests.log"

echo "[throughput] small bench..." | tee "${OUT}/throughput_small.log"
for MODE in shared per_sample_list tensorized_cms; do
  BS=1
  if [[ "${MODE}" != "shared" ]]; then
    BS=4
  fi
  "${RUN_PY[@]}" scripts/bench_throughput.py \
    --config bench_micro_200 \
    --override train.device=cuda:0 \
    --override data.batch_size=${BS} \
    --override train.fast_state_batch_mode=${MODE} \
    --trials 3 --warmup_steps 20 --measure_steps 200 \
    --output "${OUT}/bench_small_${MODE}_b${BS}.json" \
    --resolved-config-out "${OUT}/bench_small_${MODE}_b${BS}.resolved.yaml" \
    | tee -a "${OUT}/throughput_small.log"
done

echo "[throughput] medium bench..." | tee "${OUT}/throughput_medium.log"
for MODE in shared per_sample_list tensorized_cms; do
  BS=1
  if [[ "${MODE}" != "shared" ]]; then
    BS=4
  fi
  "${RUN_PY[@]}" scripts/bench_throughput.py \
    --config bench_micro_medium \
    --override train.device=cuda:0 \
    --override data.batch_size=${BS} \
    --override train.fast_state_batch_mode=${MODE} \
    --trials 3 --warmup_steps 20 --measure_steps 200 \
    --output "${OUT}/bench_medium_${MODE}_b${BS}.json" \
    --resolved-config-out "${OUT}/bench_medium_${MODE}_b${BS}.resolved.yaml" \
    | tee -a "${OUT}/throughput_medium.log"
done

echo "[guard] throughput thresholds..." | tee "${OUT}/throughput_guard.log"
"${RUN_PY[@]}" scripts/checks/validate_throughput_gate.py \
  --label small \
  --b1-json "${OUT}/bench_small_shared_b1.json" \
  --list-json "${OUT}/bench_small_per_sample_list_b4.json" \
  --tensorized-json "${OUT}/bench_small_tensorized_cms_b4.json" \
  --min-list-ratio 1.5 \
  --min-b1-ratio 1.8 \
  | tee -a "${OUT}/throughput_guard.log"
"${RUN_PY[@]}" scripts/checks/validate_throughput_gate.py \
  --label medium \
  --b1-json "${OUT}/bench_medium_shared_b1.json" \
  --list-json "${OUT}/bench_medium_per_sample_list_b4.json" \
  --tensorized-json "${OUT}/bench_medium_tensorized_cms_b4.json" \
  --min-list-ratio 1.5 \
  --min-b1-ratio 1.8 \
  | tee -a "${OUT}/throughput_guard.log"

echo "[train] token-matched list vs tensorized (B=4, LR=0.004)..." | tee "${OUT}/train.log"
"${RUN_PY[@]}" train.py --config-name bench_small_isolated_b4 \
  train.device=cuda:0 \
  train.steps=500 \
  train.fast_state_batch_mode=per_sample_list \
  train.online_chunk_size=2 \
  optim.lr=0.004 \
  logging.path="${OUT}/b4_list_lr0.004_s500.json" \
  logging.run_name="${STAMP}_b4_list_lr0.004_s500" \
  | tee "${OUT}/b4_list_lr0.004_s500.log"

"${RUN_PY[@]}" train.py --config-name bench_small_isolated_b4 \
  train.device=cuda:0 \
  train.steps=500 \
  train.fast_state_batch_mode=tensorized_cms \
  train.online_chunk_size=2 \
  optim.lr=0.004 \
  logging.path="${OUT}/b4_tensorized_lr0.004_s500.json" \
  logging.run_name="${STAMP}_b4_tensorized_lr0.004_s500" \
  | tee "${OUT}/b4_tensorized_lr0.004_s500.log"

echo "[analysis] token-normalized..." | tee "${OUT}/analysis.log"
"${RUN_PY[@]}" scripts/analyze_token_normalized.py \
  --runs "${OUT}/b4_list_lr0.004_s500.json" "${OUT}/b4_tensorized_lr0.004_s500.json" \
  --labels b4_list b4_tensorized \
  --out_dir "${OUT}/token_analysis" \
  | tee -a "${OUT}/analysis.log"

echo "[done] ${OUT}"
