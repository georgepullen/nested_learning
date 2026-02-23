#!/usr/bin/env bash
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1
cd /workspace/nested_learning
mkdir -p logs reports

TS=$(date -u +%Y%m%dT%H%M%SZ)
META_LOG="logs/dedicated_${TS}_suite_meta.log"
START_STAGE=${SUITE_START_STAGE:-A}

_stage_order() {
  case "$1" in
    A) echo 1 ;;
    B) echo 2 ;;
    C) echo 3 ;;
    *) echo 999 ;;
  esac
}

_should_run_stage() {
  local stage="$1"
  local start_ord stage_ord
  start_ord=$(_stage_order "${START_STAGE}")
  stage_ord=$(_stage_order "${stage}")
  [[ "${stage_ord}" -ge "${start_ord}" ]]
}

_dump_cfg() {
  local cfg_name="$1"
  local out_path="$2"
  shift 2
  uv run python - "$cfg_name" "$out_path" "$@" <<'PY'
from pathlib import Path
import sys
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from nested_learning.training import unwrap_config

cfg_name = sys.argv[1]
out_path = Path(sys.argv[2])
overrides = sys.argv[3:]
config_dir = Path('/workspace/nested_learning/configs')
with initialize_config_dir(config_dir=str(config_dir), version_base=None):
    cfg = compose(config_name=cfg_name, overrides=overrides)
cfg = unwrap_config(cfg)
out_path.write_text(OmegaConf.to_yaml(cfg, resolve=True))
print(f"wrote {out_path}")
PY
}

echo "[suite] started at $(date -u --iso-8601=seconds)" | tee -a "$META_LOG"
echo "[suite] ts=$TS start_stage=$START_STAGE" | tee -a "$META_LOG"

if _should_run_stage A; then
  echo "[suite] stage=A bench_throughput_micro200" | tee -a "$META_LOG"
  uv run python scripts/bench_throughput.py \
    --config bench_micro_200 \
    --override train.device=cuda:0 \
    --override train.steps="${SUITE_STAGE_A_STEPS:-200}" \
    --override train.log_interval="${SUITE_STAGE_A_LOG_INTERVAL:-10}" \
    --output "logs/dedicated_${TS}_bench_throughput_micro200.json" \
    --resolved-config-out "logs/dedicated_${TS}_bench_throughput_micro200.resolved.yaml" \
    | tee -a "logs/dedicated_${TS}_bench_throughput_micro200.console.log"
fi

if _should_run_stage B; then
  echo "[suite] stage=B bench_small_b1_2k" | tee -a "$META_LOG"
  _dump_cfg \
    "bench_small_b1" \
    "logs/dedicated_${TS}_bench_small_b1_2k.resolved.yaml" \
    "train.device=cuda:0" \
    "logging.path=logs/dedicated_${TS}_bench_small_b1_2k_metrics.json"
  uv run python train.py \
    --config-name bench_small_b1 \
    train.device=cuda:0 \
    logging.path="logs/dedicated_${TS}_bench_small_b1_2k_metrics.json" \
    | tee -a "logs/dedicated_${TS}_bench_small_b1_2k.console.log"
fi

if _should_run_stage C; then
  echo "[suite] stage=C bench_small_isolated_b4_2k" | tee -a "$META_LOG"
  _dump_cfg \
    "bench_small_isolated_b4" \
    "logs/dedicated_${TS}_bench_small_isolated_b4_2k.resolved.yaml" \
    "train.device=cuda:0" \
    "logging.path=logs/dedicated_${TS}_bench_small_isolated_b4_2k_metrics.json"
  uv run python train.py \
    --config-name bench_small_isolated_b4 \
    train.device=cuda:0 \
    logging.path="logs/dedicated_${TS}_bench_small_isolated_b4_2k_metrics.json" \
    | tee -a "logs/dedicated_${TS}_bench_small_isolated_b4_2k.console.log"
fi

echo "[suite] completed at $(date -u --iso-8601=seconds)" | tee -a "$META_LOG"
