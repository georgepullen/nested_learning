# Tensorized CMS Qualification Rerun Report
Date: 2026-02-21 (UTC)
Run ID: `20260221T120209Z_tensorized_qual_3090`
Primary artifact dir: `/home/george/research/Cyril/kmccleary_nested_learning/logs/20260221T120209Z_tensorized_qual_3090`

## 1) Scope
This rerun executed the full qualification flow on the 3090 after landing tensorized CMS batching changes:
1. Tensorized correctness gate tests.
2. Throughput protocol on small + medium benches (`B=1 shared`, `B=4 per_sample_list`, `B=4 tensorized_cms`).
3. Token-matched quality comparison (`B=4`, `steps=500`, `lr=0.004`) between `per_sample_list` and `tensorized_cms`.
4. Token-normalized analysis.

## 2) Code Changes Implemented
The implementation focus was removing residual per-sample optimizer-apply overhead in the tensorized CMS path while preserving per-sample semantics.

### 2.1 Batched CMS forward path helper
- Added `call_with_batched_deltas(...)` with `vmap` for per-sample batched delta execution.
- File: `/Users/georgepullen/Documents/research/Cyril/kmccleary_nested_learning/src/nested_learning/functional.py:51`

### 2.2 Batched optimizer apply API
- Added `apply_grads_batched(...)` with optional `sample_mask` and batched grad/param shape checks.
- File: `/Users/georgepullen/Documents/research/Cyril/kmccleary_nested_learning/src/nested_learning/optim/manager.py:141`

### 2.3 DeepMomentum batched-state handling
- Added `sample_mask` support to preserve inactive-sample optimizer state.
- Added batched context handling for NL preconditioning.
- Key entries:
  - `/Users/georgepullen/Documents/research/Cyril/kmccleary_nested_learning/src/nested_learning/optim/deep.py:16`
  - `/Users/georgepullen/Documents/research/Cyril/kmccleary_nested_learning/src/nested_learning/optim/deep.py:80`
  - `/Users/georgepullen/Documents/research/Cyril/kmccleary_nested_learning/src/nested_learning/optim/deep.py:248`

### 2.4 Tensorized CMS update hot path wiring
- Added `_apply_tensorized_cms_grads(...)` using one autograd pass + batched manager apply.
- Added helper routing for shared/list/tensorized delta calls.
- Key entries:
  - `/Users/georgepullen/Documents/research/Cyril/kmccleary_nested_learning/src/nested_learning/hope/block.py:104`
  - `/Users/georgepullen/Documents/research/Cyril/kmccleary_nested_learning/src/nested_learning/hope/block.py:157`
  - `/Users/georgepullen/Documents/research/Cyril/kmccleary_nested_learning/src/nested_learning/hope/block.py:67`

### 2.5 Fast-state batch mode support
- Fast-state construction now supports `shared`, `per_sample_list`, `tensorized_cms` with batched delta initialization.
- File: `/Users/georgepullen/Documents/research/Cyril/kmccleary_nested_learning/src/nested_learning/fast_state.py:58`

### 2.6 Training controls and fairness instrumentation (used by this run)
- `grad_accum_steps` validation/use and logging.
- `optimizer_steps_total` logging.
- Explicit rejection of `online_chunk_size=1`.
- Batch-mode aware teach-signal normalization.
- Key entries:
  - `/Users/georgepullen/Documents/research/Cyril/kmccleary_nested_learning/src/nested_learning/training.py:520`
  - `/Users/georgepullen/Documents/research/Cyril/kmccleary_nested_learning/src/nested_learning/training.py:640`
  - `/Users/georgepullen/Documents/research/Cyril/kmccleary_nested_learning/src/nested_learning/training.py:700`

### 2.7 New/extended tests relevant to these changes
- Batched optimizer behavior tests:
  - `/Users/georgepullen/Documents/research/Cyril/kmccleary_nested_learning/tests/test_optim_manager_batched.py:25`
  - `/Users/georgepullen/Documents/research/Cyril/kmccleary_nested_learning/tests/test_optim_manager_batched.py:48`
- Checkpoint parity includes `tensorized_cms` mode:
  - `/Users/georgepullen/Documents/research/Cyril/kmccleary_nested_learning/tests/test_checkpoint_resume_parity.py:115`

## 3) Execution Artifacts
- Env: `/home/george/research/Cyril/kmccleary_nested_learning/logs/20260221T120209Z_tensorized_qual_3090/env.log`
- Tests: `/home/george/research/Cyril/kmccleary_nested_learning/logs/20260221T120209Z_tensorized_qual_3090/tests.log`
- Throughput:
  - `/home/george/research/Cyril/kmccleary_nested_learning/logs/20260221T120209Z_tensorized_qual_3090/throughput_small.log`
  - `/home/george/research/Cyril/kmccleary_nested_learning/logs/20260221T120209Z_tensorized_qual_3090/throughput_medium.log`
- Train:
  - `/home/george/research/Cyril/kmccleary_nested_learning/logs/20260221T120209Z_tensorized_qual_3090/b4_list_lr0.004_s500.json`
  - `/home/george/research/Cyril/kmccleary_nested_learning/logs/20260221T120209Z_tensorized_qual_3090/b4_tensorized_lr0.004_s500.json`
- Analysis:
  - `/home/george/research/Cyril/kmccleary_nested_learning/logs/20260221T120209Z_tensorized_qual_3090/token_analysis/summary.md`
  - `/home/george/research/Cyril/kmccleary_nested_learning/logs/20260221T120209Z_tensorized_qual_3090/token_analysis/loss_vs_tokens.csv`

## 4) Results Analysis

### 4.1 Correctness gate tests
From `tests.log`:
- Tensorized qualification test set passed (`8 passed`).
- Warnings observed (PyTorch pin_memory deprecation), no failures.

### 4.2 Throughput (tokens/s)

| Bench | shared B=1 | list B=4 | tensorized B=4 | tensorized/list | tensorized/B1 |
|---|---:|---:|---:|---:|---:|
| small | 102.20 | 148.38 | 203.21 | 1.37x | 1.99x |
| medium | 67.58 | 96.87 | 133.10 | 1.37x | 1.97x |

Source JSON files:
- `bench_small_shared_b1.json`, `bench_small_per_sample_list_b4.json`, `bench_small_tensorized_cms_b4.json`
- `bench_medium_shared_b1.json`, `bench_medium_per_sample_list_b4.json`, `bench_medium_tensorized_cms_b4.json`

VRAM means (GB):
- small: shared `0.438`, list `0.674`, tensorized `0.674`
- medium: shared `1.121`, list `1.916`, tensorized `1.916`

Interpretation:
1. Tensorized mode materially outperforms list mode (+37% in both benches).
2. Tensorized mode exceeds the secondary gate vs B1 (`>=1.8x`) in both benches.
3. Tensorized mode does not meet the primary gate vs list (`>=1.5x`); achieved `~1.37x`.

### 4.3 Token-matched quality parity (`B=4`, `lr=0.004`, `steps=500`)
From `token_analysis/summary.md`:

Terminal (246,272 tokens):
- `b4_list` loss: `10.467682`
- `b4_tensorized` loss: `10.482269`
- absolute delta: `+0.014587`
- relative delta vs list: `+0.139%`

Milestones:
- 50k: list `14.517237`, tensorized `14.369434`
- 100k: list `10.546992`, tensorized `10.529455`
- 150k: list `10.521387`, tensorized `10.522708`
- 200k: list `10.482813`, tensorized `10.478714`

Interpretation:
- Quality parity is tight and stable across milestones.
- Tensorized path is not introducing measurable degradation relative to list-mode oracle in this benchmark.

## 5) Gate Status
Using current senior-defined gates:

1. Semantic/correctness gate (`T0`): **PASS**
- Qualification tests passed in this rerun.

2. Quality parity gate (`T1`, tensorized vs list at matched tokens): **PASS**
- Terminal delta `+0.139%` (well within ±2% band).

3. Throughput gate (`T2` primary, tensorized >=1.5x list): **FAIL**
- Achieved `~1.37x` (small and medium).

4. Throughput gate (`T2` secondary, tensorized >=1.8x B1): **PASS**
- Achieved `1.99x` (small), `1.97x` (medium).

## 6) Reviewer Decisions Requested
1. Promotion decision for `tensorized_cms` default (`batch_size>1`):
- Option A: Promote now (quality parity + strong B1 gain), keep list-mode as oracle.
- Option B: Hold promotion until primary throughput gate (`>=1.5x list`) is met.

2. Throughput optimization priority:
- Option A: Focus on removing remaining list/tensorized overhead in batched optimizer/state paths and re-run throughput only.
- Option B: Accept current throughput and proceed to B=8/B=16 scaling matrix.

3. Benchmark realism:
- Option A: Keep synthetic bench for fast iteration and move to larger model/seq for perf validation.
- Option B: Move immediately to real-corpus quality/throughput confirmation.

## 7) Recommended Next Step (if no further direction)
Run a targeted throughput optimization cycle (no semantic changes), then rerun only throughput benchmarks on the same protocol to try to clear the `>=1.5x list` primary gate.
