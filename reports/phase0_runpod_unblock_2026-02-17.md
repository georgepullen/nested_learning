# Phase 0 RunPod Unblock Log (2026-02-17)

This log records the concrete unblock work to align the repo execution flow with the paper-faithful Phase 0 runbook in `reports/NL_IMPLEMENTATION_ORACLE.md`.

## What was blocked

- RunPod data bootstrap failed in `scripts/data/run_sample.sh` at SlimPajama source:
  - `cerebras/SlimPajama-627B` no longer resolves from current HF access path.
- RunPod bootstrap selected Python 3.14 by default in this environment, which broke `datasets` processing during filtering.
- Muon optimizer selection accepted non-2D tensors and crashed on paper-faithful selfmod local-conv weights.
- `ppl` telemetry overflowed to `inf` on early high-loss steps, causing telemetry quality-gate failures.

## Changes applied

- Data pipeline source resilience:
  - `scripts/data/run_sample.sh`
  - `scripts/data/run_full.sh`
  - Added SlimPajama fallback candidate chain:
    - primary: `cerebras/SlimPajama-627B`
    - fallbacks: `MBZUAI-LLM/SlimPajama-627B-DC`, `DKYoon/SlimPajama-6B`, `gmongaras/SlimPajama-627B_Reupload`
- RunPod reproducibility:
  - `scripts/compute/runpod_bootstrap.sh`
  - Defaulted bootstrap to Python `3.12` (`PYTHON_VERSION` override supported).
- RunPod sync reliability:
  - `scripts/compute/runpod_sync.sh`
  - Added SSH key support via `RUNPOD_SSH_KEY`.
  - Added tar-over-SSH fallback when `rsync` is missing on either side.
- Phase 0 eval path consistency:
  - `configs/data/continual_segments_sample.yaml`
  - Pointed continual segments to `*_filtered` shard outputs produced by sample/full filtering scripts.
- Muon compatibility fix:
  - `src/nested_learning/training.py`
  - `_is_muon_candidate` now requires exactly 2D tensors.
  - Added regression test: `tests/test_optimizer_param_policy.py`.
- Telemetry overflow fix:
  - `src/nested_learning/training.py`
  - Added `_safe_perplexity()` to clamp exponential overflow.
  - Added regression test: `tests/test_training_metrics.py`.

## RunPod validation runs completed

Pod used:
- `6sbhkq9580btzo` (`nl-phase0-worker`, RTX 3080)
- SSH endpoint: `root@94.61.157.224 -p 49093`

### 1) Data pipeline re-run (previous hard block)

Command:

```bash
bash scripts/data/run_sample.sh artifacts/tokenizer/refinedweb_mix/spm_32000_unigram.model
```

Outcome:
- Canonical source failed, fallback source succeeded.
- Completed end-to-end and produced:
  - `data/filtered/redpajama_en_sample.txt`
  - `data/filtered/code_en_sample.txt`
  - `data/mixtures/refinedweb_mix_filtered_shards.json`
  - `data/shards/{refinedweb,wikipedia,c4,redpajama,code}_filtered/*`

### 2) Paper-faithful optimizer/telemetry gates (fast validation)

SelfMod:

```bash
uv run python train.py --config-name pilot_selfmod_paper_faithful \
  train.steps=1 train.log_interval=1 train.device=cuda:0 \
  data.seq_len=128 data.batch_size=1 data.num_workers=0 \
  logging.path=logs/tmp_selfmod_metrics.json \
  train.checkpoint.dir=artifacts/checkpoints/tmp_selfmod
uv run python scripts/checkpoint/verify.py --checkpoint artifacts/checkpoints/tmp_selfmod/step_000001.pt
uv run python scripts/checks/validate_fidelity_telemetry.py --log logs/tmp_selfmod_metrics.json
```

Attention:

```bash
uv run python train.py --config-name pilot_attention_paper_faithful \
  train.steps=1 train.log_interval=1 train.device=cuda:0 \
  data.seq_len=128 data.batch_size=1 data.num_workers=0 \
  logging.path=logs/tmp_attention_metrics.json \
  train.checkpoint.dir=artifacts/checkpoints/tmp_attention
uv run python scripts/checkpoint/verify.py --checkpoint artifacts/checkpoints/tmp_attention/step_000001.pt
uv run python scripts/checks/validate_fidelity_telemetry.py --log logs/tmp_attention_metrics.json
```

Outcome:
- Muon crash resolved.
- Telemetry gate passes for both quick validation logs.

## Context7 references used

- RunPod CLI operational patterns (`runpodctl` create/get/ssh):
  - Context7 library: `/runpod/runpodctl`
- PyTorch release/support policy context (for Python compatibility decisions):
  - Context7 library: `/pytorch/pytorch/v2.5.1`

## Remaining action for canonical Phase 0

- Run full canonical command in `scripts/compute/run_phase0_baselines.sh` on a larger GPU class than RTX 3080 (runtime on 3080 is not practical for `train.steps=2000` with current model/data defaults).
- Keep quality gates unchanged:
  - `scripts/checkpoint/verify.py`
  - `scripts/checks/validate_fidelity_telemetry.py`

## Live canonical run status

Launched on the current pod as a background job:

```bash
nohup env DEVICE=cuda:0 bash scripts/compute/run_phase0_baselines.sh \
  > logs/phase0_runpod_full_2026-02-17.log 2>&1 &
```

Observed process:
- PID: `2285` (`bash scripts/compute/run_phase0_baselines.sh`)
- Log: `logs/phase0_runpod_full_2026-02-17.log`

Monitoring commands:

```bash
tail -f logs/phase0_runpod_full_2026-02-17.log
ps -eo pid,etimes,cmd | grep run_phase0_baselines.sh | grep -v grep
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader
```
