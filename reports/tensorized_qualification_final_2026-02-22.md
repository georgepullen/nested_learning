# Tensorized CMS Qualification Final Report
Date: 2026-02-22 (UTC)
Run ID: `logs/20260222T111721Z_tensorized_qual_3090`
Host: `ssh 3090` (`george-linux-server`, RTX 3090 24GB)

## 1) Scope and Protocol
This run executed the full qualification protocol from `scripts/run_tensorized_qualification_3090.sh`:
- qualification tests (8 targeted tests)
- throughput protocol, both benches, `trials=3`, `warmup_steps=20`, `measure_steps=200`
- token-matched quality comparison at B=4 and LR=0.004:
  - `per_sample_list` (`steps=500`)
  - `tensorized_cms` (`steps=500`)
- token-normalized post-analysis via `scripts/analyze_token_normalized.py`

Approximate wall-clock for this full run: ~5h46m (start around 11:17 UTC, completed around 17:03 UTC).

## 2) Code/Infra Changes Used in This Qualification
Changes applied before this rerun to close the 1.5x gate focused on removing residual overhead in tensorized fast-state execution and avoiding stale remote code paths:
- `src/nested_learning/functional.py`
  - optional hot-path validation bypass in batched call path.
- `src/nested_learning/hope/block.py`
  - fast-path mode dispatch for tensorized/list/shared call sites.
  - batched TITAN update path for batched deltas.
  - all-active mask fast path and reduced context work in batched update sections.
- `src/nested_learning/optim/manager.py`
  - batched gradient apply path improvements and lower-overhead norm handling.
  - context-needed checks.
- `src/nested_learning/fast_state.py`
  - tensorized TITAN fast-state stored in batched tensor form for tensorized mode.

Operational correction made during rerun: fixed remote sync target paths so the 3090 run consumed updated files.

## 3) Qualification Tests (T0)
Source: `logs/20260222T111721Z_tensorized_qual_3090/tests.log`

Result:
- `8 passed` (`........ [100%]`)
- no test failures.

Interpretation:
- semantic/correctness gate (T0) passed for the targeted tensorized qualification suite.

## 4) Throughput Results (T2)
Sources:
- `bench_small_shared_b1.json`
- `bench_small_per_sample_list_b4.json`
- `bench_small_tensorized_cms_b4.json`
- `bench_medium_shared_b1.json`
- `bench_medium_per_sample_list_b4.json`
- `bench_medium_tensorized_cms_b4.json`

### 4.1 Small bench (`bench_micro_200`)
- shared B=1: `103.8984` tok/s, peak VRAM `0.4378` GB
- list B=4: `147.9915` tok/s, peak VRAM `0.6740` GB
- tensorized B=4: `267.4566` tok/s, peak VRAM `0.7369` GB

Ratios:
- tensorized/list: **1.8072x**
- tensorized/B1: **2.5742x**

### 4.2 Medium bench (`bench_micro_medium`)
- shared B=1: `67.8786` tok/s, peak VRAM `1.1213` GB
- list B=4: `96.9815` tok/s, peak VRAM `1.9158` GB
- tensorized B=4: `158.4526` tok/s, peak VRAM `2.1278` GB

Ratios:
- tensorized/list: **1.6338x**
- tensorized/B1: **2.3344x**

### 4.3 Throughput gate verdict
Primary gate (>=1.5x list-mode) and secondary gate (>=1.8x B1) are both satisfied on both benchmark regimes.

- Primary (tensorized/list): PASS (`1.807x` small, `1.634x` medium)
- Secondary (tensorized/B1): PASS (`2.574x` small, `2.334x` medium)

## 5) Token-Matched Quality Parity (T1)
Source: `logs/20260222T111721Z_tensorized_qual_3090/token_analysis/summary.md`

Runs:
- list: `b4_list_lr0.004_s500.json`
- tensorized: `b4_tensorized_lr0.004_s500.json`

Terminal point (246,272 tokens):
- list loss: `10.467682`
- tensorized loss: `10.482635`
- absolute delta: `+0.014953`
- relative delta: `+0.143%`

Milestones:
- 50k tokens: list `14.517237`, tensorized `14.571945` (delta `+0.054708`)
- 100k tokens: list `10.546992`, tensorized `10.527964` (delta `-0.019028`)
- 150k tokens: list `10.521387`, tensorized `10.524732` (delta `+0.003345`)
- 200k tokens: list `10.482813`, tensorized `10.487421` (delta `+0.004608`)

T1 verdict:
- quality parity vs list-mode is maintained well within the prior ±2% acceptance framing.

## 6) Gate Summary
- T0 Semantics/Correctness: **PASS**
- T1 Quality parity vs list-mode (token-matched): **PASS**
- T2 Throughput >=1.5x list-mode: **PASS**

Overall qualification status for `tensorized_cms`: **PASS**.

## 7) Key Artifacts
- Launch log: `logs/20260222T111721Z_run_tensorized_qualification_3090.launch.log`
- Output directory: `logs/20260222T111721Z_tensorized_qual_3090`
- Throughput JSONs:
  - `logs/20260222T111721Z_tensorized_qual_3090/bench_small_shared_b1.json`
  - `logs/20260222T111721Z_tensorized_qual_3090/bench_small_per_sample_list_b4.json`
  - `logs/20260222T111721Z_tensorized_qual_3090/bench_small_tensorized_cms_b4.json`
  - `logs/20260222T111721Z_tensorized_qual_3090/bench_medium_shared_b1.json`
  - `logs/20260222T111721Z_tensorized_qual_3090/bench_medium_per_sample_list_b4.json`
  - `logs/20260222T111721Z_tensorized_qual_3090/bench_medium_tensorized_cms_b4.json`
- Quality runs:
  - `logs/20260222T111721Z_tensorized_qual_3090/b4_list_lr0.004_s500.json`
  - `logs/20260222T111721Z_tensorized_qual_3090/b4_tensorized_lr0.004_s500.json`
- Analysis:
  - `logs/20260222T111721Z_tensorized_qual_3090/token_analysis/summary.md`
  - `logs/20260222T111721Z_tensorized_qual_3090/token_analysis/loss_vs_tokens.csv`

## 8) Decisions Recommended to Senior Agent
1. Promote `tensorized_cms` as default for `batch_size>1` in qualification/benchmark paths, keeping `per_sample_list` as oracle fallback.
2. Freeze the current throughput gate as achieved, and shift next effort from “gate clearing” to “scaling characterization” (B=8/B=16 sweep with the same fairness framing).
3. Keep LR anchor at `0.004` for B=4 token-matched comparisons; introduce a minimal LR check only when scaling B beyond 4.
4. Add one regression CI check on the medium bench ratio threshold (coarse smoke threshold, e.g., tensorized/list > 1.45x) to catch performance regressions early.
