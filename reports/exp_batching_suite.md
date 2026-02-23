# Fast-State Batching Experiment Suite (Progress Report)

Date: 2026-02-18

## Scope completed in this pass

1. Experiment 0 gates added:
   - `tests/test_online_updates_nontrivial.py`
2. Experiment 1 implemented:
   - `compute_teach_signal(..., normalization=\"per_sample\")`
   - config plumbing: `train.teach_signal_normalization`
   - `tests/test_teach_signal_batch_equivalence.py`
3. Experiment 2 implemented (minimal-risk per-sample list mode):
   - config plumbing: `train.fast_state_batch_mode` with `shared|per_sample_list`
   - per-sample fast-state storage in `src/nested_learning/fast_state.py`
   - model initialization support in `src/nested_learning/model.py`
   - CMS/TITAN fast-update list-mode handling in `src/nested_learning/hope/block.py`
   - batch semantics validation updated in `src/nested_learning/training.py`
   - isolation/equivalence tests:
     - `tests/test_fast_state_isolation.py`
     - `tests/test_batched_equals_sequential.py`
4. Harness additions:
   - runtime watchdog in train loop (`train.max_runtime_seconds`)
   - fast-state instrumentation metrics:
     - `fast_state.layer*.{cms|titan}.*.delta_norm`
     - `...delta_change`
     - `...isolation_checksum*`
5. Throughput harness:
   - `scripts/bench_throughput.py`
   - configs:
     - `configs/bench_small_b1.yaml`
     - `configs/bench_small_b4.yaml`
     - `configs/bench_micro_200.yaml`
     - `configs/bench_small_isolated_b4.yaml`

## Gate results

### G0: unit tests

Executed:

```bash
uv run pytest -q \
  tests/test_teach_signal_batch_equivalence.py \
  tests/test_online_updates_nontrivial.py \
  tests/test_fast_state_isolation.py \
  tests/test_batched_equals_sequential.py \
  tests/test_fast_state_batch_semantics.py
```

Result: pass (`7 passed`).

Additional regression subset:

```bash
uv run pytest -q \
  tests/test_teach_signal.py \
  tests/test_fast_state_forward_equivalence.py \
  tests/test_self_modifying_titans.py
```

Result: pass (`11 passed`).

Note: `tests/test_memorization.py` collection requires `sentencepiece`; local environment did not have it at collection time.

### G1: throughput bench

RunPod execution command used:

```bash
uv run python scripts/bench_throughput.py \
  --config bench_micro_200 \
  --override train.device=cuda:0 \
  --override train.steps=20 \
  --override train.log_interval=5 \
  --output logs/bench_throughput_micro20_gpu.json \
  --resolved-config-out logs/bench_throughput_micro20_gpu.resolved.yaml
```

Key output (`logs/bench_throughput_micro20_gpu.json`):

- `steps`: 20
- `elapsed_seconds`: `366.9240`
- `steps_per_second`: `0.0545`
- `tokens_per_second`: `27.9077`
- `peak_vram_gb`: `0.6780`

### G2: stability

Observed from run output/logs:
- No NaN/Inf surfaced in logged metrics.
- Loss remained finite through the run (`step 0 -> step 15`: `79.49 -> 75.71`).

## RunPod notes

- Used Context7 docs for runpodctl command semantics (create/get/ssh/start/stop/remove).
- Created pod `v7d0cpnnktut8t` (`nl-batch-suite`) successfully, but SSH endpoint stayed unavailable (connection refused). Pod was stopped and removed.
- Continued execution on accessible existing RTX 3080 pod `pngijmxmcu9fte`.

## What remains for full suite closure

1. Complete full 200-step micro bench with isolated pod occupancy (no competing workloads).
2. Run bounded 2k-step comparison:
   - `bench_small_b1` (baseline)
   - `bench_small_isolated_b4` (isolated batched)
3. Add Experiment 6 microbatch fallback (`train.fast_state_microbatch`) and validation.
4. Implement and evaluate Experiments 7/8 (LoRA fast deltas, latent gating path).
