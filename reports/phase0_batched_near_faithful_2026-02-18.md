# Phase 0 Batched Near-Faithful Run Log (2026-02-18)

## Request

Cancel the active Phase 0 run and re-launch with batching while staying as close to paper-faithful semantics as possible.

## Cancellation

Cancelled on RunPod pod `6sbhkq9580btzo` (`root@94.61.157.224:49093`) by terminating active Phase 0 selfmod processes.

Post-cancel verification:

- No active `run_phase0_baselines.sh` / `train.py --config-name pilot_selfmod_paper_faithful` processes.
- GPU idle (`nvidia-smi`: `0 %, 1 MiB, 10240 MiB`).

## Fidelity constraint (strict paper-faithful vs batching)

Strict paper-faithful + fast state (`train.use_fast_state=true`) currently requires `data.batch_size=1` in this repo.

Evidence:

- `src/nested_learning/training.py` (`_validate_fast_state_batch_semantics`) raises when `batch_size>1` and `train.fail_if_paper_faithful_disabled=true`.
- `configs/pilot_paper_faithful.yaml` sets:
  - `data.batch_size: 1`
  - `train.use_fast_state: true`
  - `train.fail_if_paper_faithful_disabled: true`
- `tests/test_fast_state_batch_semantics.py` and `tests/test_paper_faithful_configs.py` lock this behavior.

Reason: CMS/TITAN fast state is currently shared across the batch for training semantics, so batching introduces cross-sample context interaction.

## Validation probes run

1. Strict probe (`batch_size=4`, strict fail-fast on) correctly fails with:

`RuntimeError: train.use_fast_state=true currently shares CMS/TITAN fast state across the batch. For strict per-context semantics, set data.batch_size=1.`

2. Near-faithful probe (`batch_size=4`, `train.fail_if_paper_faithful_disabled=false`) runs and checkpoints successfully.

## Relaunch configuration used

To maximize faithfulness while enabling batching, all paper-faithful settings were preserved except strict fail-fast and batch size:

- `data.batch_size=4`
- `train.fail_if_paper_faithful_disabled=false`

Launch command:

```bash
nohup env DEVICE=cuda:0 \
  TRAIN_LOG_INTERVAL=10 \
  TRAIN_CHECKPOINT_INTERVAL=100 \
  TRAIN_EXTRA_OVERRIDES="data.batch_size=4 train.fail_if_paper_faithful_disabled=false" \
  bash scripts/compute/run_phase0_baselines.sh \
  > logs/phase0_runpod_batched_near_faithful_2026-02-18T091825Z.log 2>&1 &
```

Observed start state:

- PID: `3016` (`bash scripts/compute/run_phase0_baselines.sh`)
- Child training process active with above overrides.
- Log header confirms expected warning about shared fast state across batch.

## Interpretation

This run is **near-faithful**, not strict paper-faithful. It is the closest available batched mode without implementing per-sample isolated fast-state updates in training.

## Per-step logging remediation

To remove the delayed JSON flush behavior, `JSONLogger` was patched to flush atomically on every `log()` call (while preserving the JSON-array format expected by downstream scripts).

Changed file:

- `src/nested_learning/logging_utils.py`
  - `log()` now calls `_flush()` immediately.
  - logger loads existing JSON arrays on startup to append across restarts.
  - flush uses `*.tmp` + atomic replace.

Regression tests:

- `tests/test_logging_utils.py`
  - `test_json_logger_flushes_on_each_log`
  - `test_json_logger_appends_when_log_file_exists`

Validation command:

```bash
UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run pytest \
  tests/test_logging_utils.py tests/test_validate_fidelity_telemetry.py
```

Result: `5 passed`.

## Relaunch with true per-step logging

Phase 0 run was relaunched with `TRAIN_LOG_INTERVAL=1` so every training step is persisted immediately:

```bash
nohup env DEVICE=cuda:0 \
  TRAIN_LOG_INTERVAL=1 \
  TRAIN_CHECKPOINT_INTERVAL=100 \
  TRAIN_EXTRA_OVERRIDES="data.batch_size=4 train.fail_if_paper_faithful_disabled=false" \
  bash scripts/compute/run_phase0_baselines.sh \
  > logs/phase0_runpod_batched_near_faithful_2026-02-18T093623Z.log 2>&1 &
```

Live evidence during run:

- `logs/phase0_selfmod_metrics.json` updated while process was still running (`step=-1`, `step=0`, `step=1`, `step=2` present).

## Early runtime sample (batch=4 near-faithful)

From live per-step metrics:

- `step=0` appeared at `2026-02-18T09:39:01Z`
- `step=1` appeared at `2026-02-18T09:41:02Z` (~121s after step 0)
- `step=2` appeared at `2026-02-18T09:42:59Z` (~117s after step 1)

Observed steady-state estimate from this short sample:

- ~`119s/step` (about `30 steps/hour`) for selfmod stage on current pod/runtime.
