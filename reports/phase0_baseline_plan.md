# Phase 0 Canonical Baseline Plan

This runbook locks the Phase 0 paper-faithful baseline commands and artifact packaging contract.

## Canonical training commands

- HOPE SelfMod baseline:
  - `uv run python train.py --config-name pilot_selfmod_paper_faithful train.steps=2000`
- HOPE Attention baseline:
  - `uv run python train.py --config-name pilot_attention_paper_faithful train.steps=2000`

The canonical orchestrator is:

- `bash scripts/compute/run_phase0_baselines.sh`

## Expected outputs

- Checkpoints + sidecars:
  - `artifacts/checkpoints/phase0_selfmod/step_*.pt`
  - `artifacts/checkpoints/phase0_attention/step_*.pt`
- Logs:
  - `logs/phase0_selfmod_metrics.json`
  - `logs/phase0_attention_metrics.json`
- Eval suite JSONs:
  - `eval/{zeroshot,niah,continual,passkey,pg19}_phase0_selfmod.json`
  - `eval/{zeroshot,niah,continual,passkey,pg19}_phase0_attention.json`
- Frozen baseline bundle + checksums:
  - `artifacts/phase0_baselines/phase0_*/manifest.json`
- Report table with provenance + metrics:
  - `reports/phase0_baseline_table.md`

## Quality gates

- `uv run python scripts/checkpoint/verify.py --checkpoint <checkpoint.pt>` passes for both variants.
- `uv run python scripts/checks/validate_fidelity_telemetry.py --log <metrics.json>` passes for both variants.
- No NaN/Inf in training telemetry (`loss`, `ppl`, `teach_signal_norm`, `surprise_value`, and layer CMS metrics).
