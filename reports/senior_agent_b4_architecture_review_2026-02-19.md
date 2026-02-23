# Senior Review Memo: B4 Architecture Status and Next Direction
Date: 2026-02-19 (UTC)
Scope: Local RTX 3090 run `20260219T163555Z_local3090_immediate`

## 1) Executive Decision Summary
The `B=4` list-mode architecture is now semantically valid and operational, but it is not quality-competitive at matched token budget.

Current conclusion:
1. Keep `per_sample_list` as the correctness oracle.
2. Do not treat current `B=4` list-mode as the scaling endpoint.
3. Prioritize `tensorized_cms` implementation as the next architectural milestone.

Reason: at equal tokens, `B=4` list-mode is substantially worse than `B=1` while only delivering moderate throughput gains.

## 2) What Was Actually Measured
Run artifacts:
1. `/home/george/research/Cyril/kmccleary_nested_learning/logs/20260219T163555Z_local3090_immediate/bench_baseline_b1.json`
2. `/home/george/research/Cyril/kmccleary_nested_learning/logs/20260219T163555Z_local3090_immediate/bench_listmode_b4.json`
3. `/home/george/research/Cyril/kmccleary_nested_learning/logs/20260219T163555Z_local3090_immediate/train_baseline_b1_metrics.json`
4. `/home/george/research/Cyril/kmccleary_nested_learning/logs/20260219T163555Z_local3090_immediate/train_listmode_b4_metrics.json`
5. `/home/george/research/Cyril/kmccleary_nested_learning/logs/20260219T163555Z_local3090_immediate/token_analysis/summary.md`
6. `/home/george/research/Cyril/kmccleary_nested_learning/logs/20260219T163555Z_local3090_immediate/token_analysis/loss_vs_tokens.csv`

Gates also passed in this run:
1. teach-signal equivalence
2. fast-state isolation
3. batched equals sequential
4. online updates non-trivial
5. checkpoint/resume parity

## 3) Key Results
### 3.1 Throughput protocol (3 trials each)
`B=1 shared`:
1. mean steps/s: `0.8121`
2. mean tokens/s: `103.9433`
3. mean peak VRAM: `0.4434 GB`

`B=4 per_sample_list`:
1. mean steps/s: `0.2870`
2. mean tokens/s: `146.9641`
3. mean peak VRAM: `0.6795 GB`

Relative effect (`B=4` vs `B=1`):
1. tokens/s: `+41.4%`
2. steps/s: `-64.7%`

Interpretation: list-mode buys some token throughput, but scaling is weak and step time inflates sharply.

### 3.2 Token-matched training comparison
Runs configured for approximate token parity:
1. `B=1`, 2000 steps
2. `B=4`, 500 steps

Observed terminal points:
1. `B=1`: step `1980`, tokens `253,568`, loss `15.7678`
2. `B=4`: step `480`, tokens `246,272`, loss `27.8362`

Token-normalized milestone losses:
1. 50k tokens: `35.9337` (B1) vs `44.9075` (B4), B4 `+24.97%`
2. 100k tokens: `27.4697` (B1) vs `40.1536` (B4), B4 `+46.17%`
3. 150k tokens: `21.4147` (B1) vs `35.0213` (B4), B4 `+63.54%`
4. 200k tokens: `17.3239` (B1) vs `31.8654` (B4), B4 `+83.94%`

Interpretation: quality gap grows with training progress, not just a fixed offset.

## 4) Deep Analysis: Why B4 List-Mode Is Underperforming
This section focuses on architecture and optimization flaws, not cosmetic issues.

### 4.1 Optimization geometry mismatch (primary)
Token matching was fixed, but update matching was not:
1. `B=1`: 2000 optimizer updates
2. `B=4`: 500 optimizer updates

With unchanged optimizer settings, `B=4` is receiving fewer parameter updates at lower gradient noise. This often requires explicit retuning (LR schedule, momentum, warmup) to match convergence behavior.

Consequence: the current result is a valid failure signal for this configuration, not proof that batching is intrinsically worse.

### 4.2 List-mode algorithmic structure is expensive and distorting
`per_sample_list` updates fast-state per sample via Python-side loops and repeated autograd work. This has two effects:
1. Throughput ceiling stays low (confirmed by only `+41%` tokens/s with 4x batch).
2. Training dynamics differ from an ideal batched implementation because execution order and update granularity are per-sample/per-chunk, not truly vectorized.

Consequence: list-mode is a correctness scaffold, not a performant or necessarily dynamics-equivalent scaling architecture.

### 4.3 Chunk-size pathology indicates config/algorithm tension
Repeated warning during all runs:
1. `online_chunk_size=1 is too small; clamping to 2`

This means runtime repeatedly enters a degenerate path and gets corrected on the fly. Even if semantically safe, this is a sign that the configured update cadence and sequence chunking are misaligned with next-token CE constraints.

Consequence:
1. avoidable overhead
2. possible instability in effective update frequency
3. harder interpretability of per-step behavior

### 4.4 B4 quality failure is not due to broken semantics
Isolation/equivalence tests are passing, and token accounting is now explicit (`tokens_seen_total`). The regression is therefore likely optimization/architecture efficiency, not cross-sample contamination.

Consequence: do not back out the semantics work; build on it.

### 4.5 Tensorized mode absent: hard ceiling remains
`tensorized_cms` probe was skipped because mode is not implemented yet. Without it, the project cannot answer the main performance question: whether strict semantics can coexist with materially better throughput and acceptable quality.

Consequence: the highest-value unknown remains unresolved.

## 5) What This Means for Architectural Direction
## Direction choice
Choose: `tensorized_cms` as next architecture milestone, while keeping list-mode as a reference oracle.

Rationale:
1. current B4 list-mode has proven correctness but failed quality at token parity and has limited scaling efficiency
2. list-mode cannot realistically deliver required throughput targets as model size grows
3. tensorization is the only path that can remove Python per-sample update overhead while preserving per-sample semantics by construction

## 6) Required Next Experiment Set (decision-grade)
### 6.1 Fast remediation of current B4 underperformance
Before concluding on method quality, run controlled optimization sweeps:
1. LR sweep for B4 outer optimizer (at least 3 points around current LR)
2. warmup/decay schedule sweep
3. optional gradient accumulation baseline to separate "batch size" from "update count" effects

Acceptance test:
1. at 200k tokens, B4 loss gap vs B1 <= 20% after retuning

### 6.2 Architectural milestone: implement `tensorized_cms`
Implement and validate:
1. batched delta storage `[B, ...]`
2. batched forward/update path with one autograd pass for summed per-sample losses
3. per-sample optimizer state on batched tensors

Must pass existing semantic gates:
1. teach-signal batch equivalence
2. fast-state isolation
3. batched equals sequential
4. checkpoint/resume parity

### 6.3 Throughput + quality go/no-go criteria
Use these thresholds for advancement:
1. throughput: `tensorized_cms B=4` tokens/s >= `1.8x` current B1 tokens/s on same protocol
2. quality: token-normalized loss gap at 200k tokens <= `15%` vs B1 after B4 retuning
3. stability: no NaN/Inf, full gate pass

If not met, pivot to microbatch fallback + retune, then reassess before TITAN tensorization.

## 7) Risks If We Continue With List-Mode As-Is
1. misleading scaling story: modest throughput gain but growing quality deficit
2. inability to justify larger-scale runs economically
3. delayed detection of real bottlenecks because Python overhead dominates profiles

## 8) Bottom Line
The project has cleared the semantics barrier, which is significant. The current B4 architecture itself is not yet an acceptable scaling solution.

Recommended immediate path:
1. keep list-mode for correctness regression tests
2. retune B4 once for fairness
3. prioritize `tensorized_cms` implementation and re-run token-normalized comparison

This is the shortest path to a defensible architectural decision and a meaningful systems contribution.
