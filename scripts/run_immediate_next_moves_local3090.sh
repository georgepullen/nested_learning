#!/usr/bin/env bash
set -euo pipefail

STAMP="$(date -u +%Y%m%dT%H%M%SZ)_local3090_immediate"
OUT_DIR="logs/${STAMP}"
mkdir -p "${OUT_DIR}"

echo "[env] collecting system info..." | tee "${OUT_DIR}/env.log"
uname -a | tee -a "${OUT_DIR}/env.log"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi | tee -a "${OUT_DIR}/env.log"
fi
echo "git_sha=$(git rev-parse HEAD)" | tee -a "${OUT_DIR}/env.log"

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export PYTHONUNBUFFERED=1

if command -v uv >/dev/null 2>&1; then
  echo "[deps] uv sync..." | tee "${OUT_DIR}/deps.log"
  uv sync --extra dev | tee -a "${OUT_DIR}/deps.log"
else
  echo "uv not found; install uv first" | tee -a "${OUT_DIR}/deps.log"
  exit 1
fi

echo "[test] semantic + resume gates..." | tee "${OUT_DIR}/pytest.log"
uv run python -m pytest -q \
  tests/test_teach_signal_batch_equivalence.py \
  tests/test_online_updates_nontrivial.py \
  tests/test_fast_state_isolation.py \
  tests/test_batched_equals_sequential.py \
  tests/test_fast_state_batch_semantics.py \
  tests/test_checkpoint_resume_parity.py \
  | tee -a "${OUT_DIR}/pytest.log"

echo "[bench] throughput protocol b1..." | tee "${OUT_DIR}/throughput.log"
uv run python scripts/bench_throughput.py \
  --config bench_micro_200 \
  --override train.device=cuda:0 \
  --override data.batch_size=1 \
  --override train.fast_state_batch_mode=shared \
  --trials 3 --warmup_steps 20 --measure_steps 200 \
  --output "${OUT_DIR}/bench_baseline_b1.json" \
  --resolved-config-out "${OUT_DIR}/bench_baseline_b1.resolved.yaml" \
  | tee -a "${OUT_DIR}/throughput.log"

echo "[bench] throughput protocol b4 list-mode..." | tee -a "${OUT_DIR}/throughput.log"
uv run python scripts/bench_throughput.py \
  --config bench_micro_200 \
  --override train.device=cuda:0 \
  --override data.batch_size=4 \
  --override train.fast_state_batch_mode=per_sample_list \
  --trials 3 --warmup_steps 20 --measure_steps 200 \
  --output "${OUT_DIR}/bench_listmode_b4.json" \
  --resolved-config-out "${OUT_DIR}/bench_listmode_b4.resolved.yaml" \
  | tee -a "${OUT_DIR}/throughput.log"

echo "[bench] probe tensorized_cms mode..." | tee -a "${OUT_DIR}/throughput.log"
if uv run python scripts/bench_throughput.py \
  --config bench_micro_200 \
  --override train.device=cuda:0 \
  --override data.batch_size=4 \
  --override train.fast_state_batch_mode=tensorized_cms \
  --trials 1 --warmup_steps 1 --measure_steps 2 \
  --output "${OUT_DIR}/bench_tensorized_probe.json" \
  --resolved-config-out "${OUT_DIR}/bench_tensorized_probe.resolved.yaml" \
  >/dev/null 2>&1; then
  echo "[bench] tensorized_cms available; running full 3-trial protocol" | tee -a "${OUT_DIR}/throughput.log"
  uv run python scripts/bench_throughput.py \
    --config bench_micro_200 \
    --override train.device=cuda:0 \
    --override data.batch_size=4 \
    --override train.fast_state_batch_mode=tensorized_cms \
    --trials 3 --warmup_steps 20 --measure_steps 200 \
    --output "${OUT_DIR}/bench_tensorized_cms_b4.json" \
    --resolved-config-out "${OUT_DIR}/bench_tensorized_cms_b4.resolved.yaml" \
    | tee -a "${OUT_DIR}/throughput.log"
  TENSORIZED_READY=1
else
  echo "[bench] tensorized_cms not implemented yet; skipping full tensorized benchmarks" | tee -a "${OUT_DIR}/throughput.log"
  TENSORIZED_READY=0
fi

BASE_STEPS=2000
B4_STEPS=500

echo "[train] baseline b1 token budget run..." | tee "${OUT_DIR}/train.log"
uv run python train.py --config-name bench_small_b1 \
  train.device=cuda:0 \
  train.steps=${BASE_STEPS} \
  train.max_runtime_seconds=$((23*3600)) \
  logging.path="${OUT_DIR}/train_baseline_b1_metrics.json" \
  logging.run_name="${STAMP}_baseline_b1" \
  | tee "${OUT_DIR}/train_baseline_b1.console.log"

echo "[train] list-mode b4 token-matched run..." | tee -a "${OUT_DIR}/train.log"
uv run python train.py --config-name bench_small_isolated_b4 \
  train.device=cuda:0 \
  train.steps=${B4_STEPS} \
  train.fast_state_batch_mode=per_sample_list \
  train.max_runtime_seconds=$((23*3600)) \
  logging.path="${OUT_DIR}/train_listmode_b4_metrics.json" \
  logging.run_name="${STAMP}_listmode_b4" \
  | tee "${OUT_DIR}/train_listmode_b4.console.log"

RUNS=("${OUT_DIR}/train_baseline_b1_metrics.json" "${OUT_DIR}/train_listmode_b4_metrics.json")
LABELS=("baseline_b1" "listmode_b4")
CFGS=("${OUT_DIR}/bench_baseline_b1.resolved.yaml" "${OUT_DIR}/bench_listmode_b4.resolved.yaml")

if [[ "${TENSORIZED_READY}" == "1" ]]; then
  echo "[train] tensorized CMS b4 token-matched run..." | tee -a "${OUT_DIR}/train.log"
  uv run python train.py --config-name bench_small_isolated_b4 \
    train.device=cuda:0 \
    train.steps=${B4_STEPS} \
    train.fast_state_batch_mode=tensorized_cms \
    train.max_runtime_seconds=$((23*3600)) \
    logging.path="${OUT_DIR}/train_tensorized_cms_b4_metrics.json" \
    logging.run_name="${STAMP}_tensorized_cms_b4" \
    | tee "${OUT_DIR}/train_tensorized_cms_b4.console.log"

  RUNS+=("${OUT_DIR}/train_tensorized_cms_b4_metrics.json")
  LABELS+=("tensorized_cms_b4")
  CFGS+=("${OUT_DIR}/bench_tensorized_cms_b4.resolved.yaml")
fi

echo "[analysis] token-normalized comparison..." | tee "${OUT_DIR}/analysis.log"
uv run python scripts/analyze_token_normalized.py \
  --runs "${RUNS[@]}" \
  --labels "${LABELS[@]}" \
  --resolved-configs "${CFGS[@]}" \
  --out_dir "${OUT_DIR}/token_analysis" \
  | tee -a "${OUT_DIR}/analysis.log"

echo "[done] outputs: ${OUT_DIR}"
ls -lah "${OUT_DIR}"
