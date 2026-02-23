# Senior Agent Progress Update
Date: 2026-02-20 (UTC)  
Scope: post-review execution on fairness controls, full decision matrix, and `tensorized_cms` milestone.

## 1) Executive Summary
- Root-cause hypothesis is now decision-grade confirmed at full horizon: update-count mismatch was a major factor in the earlier B=4 regression.
- Full 2000/500 token-matched matrix completed on 3090 with LR retune sweep.
- `tensorized_cms` v0 is implemented (batched CMS deltas + one autograd pass for CMS grads + per-sample optimizer apply), tested, and smoke-run on CUDA.
- Next decisions are now about promotion criteria and whether to shift immediately to tensorized throughput/quality qualification.

## 2) Completed Work

### 2.1 Fairness and instrumentation controls
Implemented:
- `train.grad_accum_steps` support with correct loss scaling and optimizer-step cadence.
- explicit logging:
  - `optimizer_steps_total`
  - `grad_accum_steps`
- strict config validation:
  - reject `train.online_chunk_size=1` (hard fail)
  - benchmark config pins `train.online_chunk_size=2`.

Primary files:
- `src/nested_learning/training.py`
- `configs/bench_small_b1.yaml`
- `tests/test_training_grad_accum.py`

### 2.2 Full matrix automation on 3090
Added script:
- `scripts/run_full_matrix_3090.sh`

Run ID:
- `logs/20260220T141330Z_fullmatrix3090`

Produced artifacts:
- `logs/20260220T141330Z_fullmatrix3090/b1_s2000.json`
- `logs/20260220T141330Z_fullmatrix3090/b1_accum4_s2000.json`
- `logs/20260220T141330Z_fullmatrix3090/b4_list_lr0.002_s500.json`
- `logs/20260220T141330Z_fullmatrix3090/b4_list_lr0.004_s500.json`
- `logs/20260220T141330Z_fullmatrix3090/b4_list_lr0.006_s500.json`
- `logs/20260220T141330Z_fullmatrix3090/token_analysis/summary.md`
- `logs/20260220T141330Z_fullmatrix3090/token_analysis/loss_vs_tokens.csv`

### 2.3 `tensorized_cms` milestone (v0)
Implemented:
- new batch mode plumbing for fast state:
  - `shared`
  - `per_sample_list`
  - `tensorized_cms`
- batched CMS delta tensors with shape `[B, *param.shape]`.
- batched forward helper using `torch.func.vmap` + `functional_call`.
- CMS fast update path:
  - one batched autograd call for CMS deltas
  - per-sample optimizer-state updates applied via existing manager list
  - per-sample context vectors preserved (`chunk_inputs.mean(dim=1)`).

Primary files:
- `src/nested_learning/fast_state.py`
- `src/nested_learning/functional.py`
- `src/nested_learning/hope/block.py`

New/updated tests:
- `tests/test_tensorized_cms_matches_listmode.py`
- `tests/test_fast_state_batch_semantics.py`

## 3) Full-Matrix Results (Decision Grade)
Source:
- `logs/20260220T141330Z_fullmatrix3090/token_analysis/summary.md`

### 3.1 Terminal points
- `b1` (steps=2000): tokens `253,568`, loss `15.7678`
- `b1_accum4` (steps=2000, grad_accum=4): tokens `253,568`, loss `29.1925`
- `b4_lr2e-3` (steps=500): tokens `246,272`, loss `10.4945`
- `b4_lr4e-3` (steps=500): tokens `246,272`, loss `10.4677`
- `b4_lr6e-3` (steps=500): tokens `246,272`, loss `10.4701`

### 3.2 Token-normalized milestones
| tokens | b1 | b1_accum4 | b4_lr2e-3 | b4_lr4e-3 | b4_lr6e-3 |
|---|---:|---:|---:|---:|---:|
| 50k | 35.9337 | 46.2713 | 19.7509 | 14.5172 | 11.7459 |
| 100k | 27.4697 | 39.5536 | 13.6527 | 10.5470 | 10.5437 |
| 150k | 21.4147 | 37.4428 | 11.5769 | 10.5214 | 10.5230 |
| 200k | 17.3239 | 30.5460 | 10.5138 | 10.4828 | 10.4899 |

### 3.3 Interpretation
1. Root-cause claim remains supported:
   - `b1_accum4` is materially worse than `b1` at matched tokens, consistent with fewer slow-weight updates harming convergence.
2. LR retune materially shifts B=4 behavior:
   - all swept B=4 LRs outperform `b1` at this workload/data regime.
3. Within B=4 sweep, `0.004` and `0.006` are effectively tied by 200k tokens.

## 4) Validation Status

### 4.1 Local targeted tests passed
- `tests/test_tensorized_cms_matches_listmode.py`
- `tests/test_fast_state_batch_semantics.py`
- `tests/test_batched_equals_sequential.py`
- `tests/test_fast_state_isolation.py`
- `tests/test_training_grad_accum.py`
- `tests/test_checkpoint_resume_parity.py`

### 4.2 Remote 3090 targeted tests passed
- `tests/test_tensorized_cms_matches_listmode.py`
- `tests/test_fast_state_batch_semantics.py`
- `tests/test_training_grad_accum.py`

### 4.3 CUDA smoke
- `tensorized_cms` smoke run succeeded:
  - `train.py --config-name bench_small_isolated_b4 ... train.fast_state_batch_mode=tensorized_cms train.steps=2`

## 5) Known Gaps / Risks
1. `tensorized_cms` equivalence test currently covers tiny HOPE-attention setup; broader variant coverage (e.g., hybrid/titan path) is still needed.
2. Performance target for tensorized mode (e.g., >=1.8x B1 tokens/s) has not yet been measured after implementation.
3. Full matrix above is list-mode B=4 only; tensorized quality/throughput matrix is pending.
4. Benchmark is on synthetic workload; real-corpus confirmation is still required before broader claims.

## 6) Decision Requests for Senior Agent
Please choose on the following:

1. Canonical B=4 LR for next phase:
   - `0.004` (recommended, slightly best at 200k and less aggressive than 0.006)
   - `0.006`
   - run tie-breaker on longer horizon/alternate seed

2. Promotion criterion for B=4 list-mode:
   - Treat current results as sufficient and freeze list-mode as correctness oracle only.
   - Or require additional seeds before freezing.

3. Immediate next matrix:
   - Proceed directly to tensorized throughput+quality matrix (`B=4 tensorized_cms` vs `B=4 list` vs `B=1`).
   - Or first expand tensorized equivalence testing across more block variants before performance runs.

4. Acceptance gates for tensorized milestone:
   - keep current: semantic parity + throughput >=1.8x B1 + stable quality at 200k tokens
   - or adjust threshold/criteria before execution.

## 7) Proposed Next Actions (Pending Decision)
If approved, execute in order:
1. Add tensorized equivalence tests for additional block variants.
2. Run throughput protocol including `tensorized_cms`.
3. Run token-matched training matrix with best selected B=4 LR using `tensorized_cms`.
4. Publish comparative report (list-mode oracle vs tensorized endpoint).

## 8) Senior Review Decisions Applied (2026-02-20 follow-up)
Decisions adopted from latest senior guidance:
1. Canonical B=4 LR for tensorized qualification defaults to `0.004`.
2. `per_sample_list` is treated as correctness oracle/fallback, not scaling endpoint.
3. Quality gate for tensorized milestone is parity vs list-mode at matched tokens.
4. Throughput gates for tensorized:
   - primary: `tensorized_cms B=4 >= 1.5x list-mode B=4` tokens/s
   - secondary: `tensorized_cms B=4 >= 1.8x B=1` tokens/s

Execution artifacts added for this phase:
- tests:
  - `tests/test_tensorized_cms_multilevel_matches_listmode.py`
  - `tests/test_tensorized_cms_hybrid_wiring_matches_listmode.py`
- throughput config:
  - `configs/bench_micro_medium.yaml`
- 3090 qualification runner:
  - `scripts/run_tensorized_qualification_3090.sh`

## 9) Comprehensive Reflection and Decision Packet (2026-02-21)
This section merges the full reflection packet into the progress report so senior review can happen from a single document.

### 9.1 Executive Reflection
The project is now in a materially better state than the pre-review baseline.

What is conclusively true from artifacts:
1. Update-count mismatch was the core cause of the earlier B=4 failure signal.
2. LR-retuned B=4 list-mode is stable and materially better than the original B=1 baseline on the synthetic benchmark at 200k tokens.
3. `tensorized_cms` v0 preserves list-mode training behavior at matched tokens (tight parity).
4. `tensorized_cms` improves throughput vs list-mode, but not enough to clear the current primary throughput gate (`>=1.5x list`) in either small or medium bench.

Pragmatic state: semantics and quality parity are de-risked; scaling/perf work remains.

### 9.2 Evidence Package (Artifacts)
#### 9.2.1 Full fairness matrix (completed)
- `/home/george/research/Cyril/kmccleary_nested_learning/logs/20260220T141330Z_fullmatrix3090/`
- `/home/george/research/Cyril/kmccleary_nested_learning/logs/20260220T141330Z_fullmatrix3090/token_analysis/summary.md`
- `/home/george/research/Cyril/kmccleary_nested_learning/logs/20260220T141330Z_fullmatrix3090/token_analysis/loss_vs_tokens.csv`

Included runs:
- `b1_s2000.json`
- `b1_accum4_s2000.json`
- `b4_list_lr0.002_s500.json`
- `b4_list_lr0.004_s500.json`
- `b4_list_lr0.006_s500.json`

#### 9.2.2 Tensorized qualification (completed)
Primary completed run:
- `/home/george/research/Cyril/kmccleary_nested_learning/logs/20260220T173614Z_tensorized_qual_3090/`
- launch log: `/home/george/research/Cyril/kmccleary_nested_learning/logs/20260220T173614Z_run_tensorized_qualification_3090.launch.log`

Run components:
- Tests: `tests.log`
- Throughput small: `bench_small_*.json`, `throughput_small.log`
- Throughput medium: `bench_medium_*.json`, `throughput_medium.log`
- Token-matched train: `b4_list_lr0.004_s500.json`, `b4_tensorized_lr0.004_s500.json`
- Analysis: `token_analysis/summary.md`, `token_analysis/loss_vs_tokens.csv`
- Env capture: `env.log`

Superseded/aborted attempt (git metadata check issue):
- `/home/george/research/Cyril/kmccleary_nested_learning/logs/20260220T173526Z_run_tensorized_qualification_3090.launch.log`
- `/home/george/research/Cyril/kmccleary_nested_learning/logs/20260220T173526Z_tensorized_qual_3090/`

#### 9.2.3 Code and test artifacts introduced for this phase
- `/Users/georgepullen/Documents/research/Cyril/kmccleary_nested_learning/scripts/run_tensorized_qualification_3090.sh`
- `/Users/georgepullen/Documents/research/Cyril/kmccleary_nested_learning/configs/bench_micro_medium.yaml`
- `/Users/georgepullen/Documents/research/Cyril/kmccleary_nested_learning/tests/test_tensorized_cms_multilevel_matches_listmode.py`
- `/Users/georgepullen/Documents/research/Cyril/kmccleary_nested_learning/tests/test_tensorized_cms_hybrid_wiring_matches_listmode.py`

### 9.3 Chronology and Runtime Reality
Full-matrix timeline (`20260220T141330Z_fullmatrix3090`):
- 14:55: `b1_s2000.json`
- 15:36: `b1_accum4_s2000.json`
- 16:04: `b4_list_lr0.002_s500.json`
- 16:33: `b4_list_lr0.004_s500.json`
- 17:02: `b4_list_lr0.006_s500.json` + summary

Tensorized qualification timeline (`20260220T173614Z_tensorized_qual_3090`):
- 17:50: small shared done
- 18:28: small list done
- 18:56: small tensorized done
- 19:37: medium shared done
- 21:32: medium list done
- 22:56: medium tensorized done
- 23:25: B=4 list train done
- 23:46: B=4 tensorized train done + analysis summary

Observation: medium-list stage dominates wall time, confirming list-mode as perf bottleneck even at moderate scale.

### 9.4 Results: Full Fairness Matrix
Source: `/home/george/research/Cyril/kmccleary_nested_learning/logs/20260220T141330Z_fullmatrix3090/token_analysis/summary.md`

#### 9.4.1 Terminal points
- `b1` (`steps=2000`): tokens `253,568`, loss `15.767822`
- `b1_accum4` (`steps=2000`, `grad_accum=4`): tokens `253,568`, loss `29.192505`
- `b4_lr2e-3` (`steps=500`): tokens `246,272`, loss `10.494537`
- `b4_lr4e-3` (`steps=500`): tokens `246,272`, loss `10.467682`
- `b4_lr6e-3` (`steps=500`): tokens `246,272`, loss `10.470123`

#### 9.4.2 Token-normalized milestone table
| tokens | b1 | b1_accum4 | b4_lr2e-3 | b4_lr4e-3 | b4_lr6e-3 |
|---|---:|---:|---:|---:|---:|
| 50k | 35.933704 | 46.271271 | 19.750859 | 14.517237 | 11.745854 |
| 100k | 27.469663 | 39.553597 | 13.652719 | 10.546992 | 10.543719 |
| 150k | 21.414668 | 37.442814 | 11.576905 | 10.521387 | 10.522952 |
| 200k | 17.323944 | 30.545993 | 10.513782 | 10.482813 | 10.489945 |

Reflection:
- Root-cause diagnosis is strongly reinforced at full horizon.
- `b1_accum4` degradation is aligned with fewer optimizer updates.
- Retuned B=4 list-mode (`lr=0.004` or `0.006`) is effectively tied by 200k tokens.

### 9.5 Results: Tensorized Qualification
#### 9.5.1 Test gate evidence (T0 semantic/test foundation)
Source: `/home/george/research/Cyril/kmccleary_nested_learning/logs/20260220T173614Z_tensorized_qual_3090/tests.log`

- `........ [100%]` (8 selected tests passed)
- Included checks:
  - `test_teach_signal_batch_equivalence`
  - `test_fast_state_isolation`
  - `test_batched_equals_sequential`
  - `test_checkpoint_resume_parity`
  - `test_tensorized_cms_matches_listmode`
  - `test_tensorized_cms_multilevel_matches_listmode`
  - `test_tensorized_cms_hybrid_wiring_matches_listmode`

Warnings observed (non-fatal):
- `Tensor.pin_memory(device)` deprecation warnings in checkpoint parity tests.

#### 9.5.2 Quality parity vs list-mode (T1)
Source: `/home/george/research/Cyril/kmccleary_nested_learning/logs/20260220T173614Z_tensorized_qual_3090/token_analysis/summary.md`

Terminal:
- `b4_list`: loss `10.467682`
- `b4_tensorized`: loss `10.482269`
- Absolute terminal delta: `+0.014587` (tensorized slightly higher), relative `+0.139%`

Token milestones:
| tokens | b4_list | b4_tensorized | delta (tensor-list) | relative |
|---|---:|---:|---:|---:|
| 50k | 14.517237 | 14.369434 | -0.147803 | -1.018% |
| 100k | 10.546992 | 10.529455 | -0.017537 | -0.166% |
| 150k | 10.521387 | 10.522708 | +0.001321 | +0.013% |
| 200k | 10.482813 | 10.478714 | -0.004099 | -0.039% |

Reflection:
- Quality parity is tight and well within the requested parity band (±2%).
- No divergence pattern between list and tensorized over the 200k-token horizon.

#### 9.5.3 Throughput qualification (T2)
Source: `bench_small_*.json`, `bench_medium_*.json` under `/home/george/research/Cyril/kmccleary_nested_learning/logs/20260220T173614Z_tensorized_qual_3090/`

Small bench (`dim=256`, `layers=4`, `seq=128`):
- B=1 shared: `102.300710 tok/s`
- B=4 list: `146.367137 tok/s`
- B=4 tensorized: `205.033808 tok/s`
- tensorized/list: `1.4008x`
- tensorized/B1: `2.0042x`

Medium bench (`dim=384`, `layers=6`, `seq=256`):
- B=1 shared: `68.149821 tok/s`
- B=4 list: `98.124289 tok/s`
- B=4 tensorized: `133.880103 tok/s`
- tensorized/list: `1.3644x`
- tensorized/B1: `1.9645x`

Reflection:
- Tensorized delivers clear speedups vs list and B1 in both regimes.
- Current primary gate (`>=1.5x list`) is **not** met.
- Secondary gate (`>=1.8x B1`) **is** met.

### 9.6 Gate-by-Gate Status Against Senior Decisions
Decision baseline:
1. Canonical B=4 LR: `0.004`
2. list-mode as oracle/fallback
3. Tensorized quality gate: parity vs list-mode
4. Throughput gates:
   - primary: tensorized B4 `>=1.5x` list B4
   - secondary: tensorized B4 `>=1.8x` B1

Status:
- Canonical LR selection: **Applied**
- List-mode role: **Applied**
- T0 semantics/tests: **PASS**
- T1 quality parity: **PASS**
- T2 throughput primary (`>=1.5x list`): **FAIL** (1.40x small, 1.36x medium)
- T2 throughput secondary (`>=1.8x B1`): **PASS** (2.00x small, 1.96x medium)

### 9.7 Decision Requests for Senior Agent
Please decide on each item explicitly.

1. Primary throughput gate disposition:
   - Option A: keep `>=1.5x list` hard gate and block promotion
   - Option B: temporary `>=1.35x list` for CMS v0
   - Option C: weighted criterion (`>=1.35x list` and `>=1.9x B1`)

2. Promotion of `tensorized_cms` default for B>1:
   - Option A: promote now, keep `per_sample_list` as oracle fallback
   - Option B: keep opt-in until primary throughput gate is met

3. Next perf engineering target:
   - Option A (recommended): vectorize/compile residual per-sample CMS optimizer apply path
   - Option B: expand benchmark matrix first
   - Option C: start TITAN tensorization first

4. Validation breadth before broader claims:
   - Option A: run one real-corpus list-vs-tensorized parity sweep
   - Option B: defer real-corpus until throughput gate is settled

5. B1 retune sanity check (claims discipline):
   - Option A: small B1 LR sweep
   - Option B: skip for now and scope claims to current baseline

### 9.8 Proposed Next Execution Plan (Pending Decision)
1. Profile medium throughput path with `torch.profiler` focused on CMS update/apply hotspots.
2. Optimize residual Python loop in per-sample optimizer apply for tensorized CMS path.
3. Rerun throughput-only qualification (`small+medium`) with the same protocol for gate re-check.
4. If throughput gate is satisfied or redefined: promote default for B>1 and run one real-data parity confirmation.
5. Publish final promotion memo with updated gate outcomes.

### 9.9 Appendix: Repro Metadata
From `/home/george/research/Cyril/kmccleary_nested_learning/logs/20260220T173614Z_tensorized_qual_3090/env.log`:
- GPU: RTX 3090 (24 GB)
- Driver: 550.163.01
- CUDA runtime in env print: `torch 2.9.0+cu128`, `cuda 12.8`
- Git SHA unavailable in this workspace copy (`no .git metadata`)

Operational warnings observed during run:
- `PYTORCH_CUDA_ALLOC_CONF` deprecation notice (suggests `PYTORCH_ALLOC_CONF`).
- `torch.backends.cuda.sdp_kernel()` future deprecation warning.

## 10) NL Paper Summary (with Verbatim Formulations)
Source summarized: `kmccleary_nested_learning/reports/paper/NL-print.extracted.clean.txt`

### 10.1 High-level summary
- The paper introduces Nested Learning (NL): a perspective that models architectures and optimization as nested, multi-level, and/or parallel optimization problems with distinct context flows.
- Core claim: familiar deep learning components can be reframed as associative memory systems that compress context flow; this includes both architectures and optimizers.
- Practical direction: add more levels/frequencies of learning to improve continual adaptation and in-context capabilities.
- Three concrete directions in the paper:
  - Deep Optimizers: reinterpret optimizer state as learned memory and extend it with richer associations/objectives/modules.
  - Self-Modifying Titans: sequence modules that learn update algorithms.
  - Continuum Memory System (CMS): memory across multiple update frequencies beyond binary short/long-term framing.

### 10.2 Verbatim formulations from `NL-print.extracted.clean.txt`
Note: copied verbatim from extracted text; line breaks/formatting follow the extraction file.

```text
Definition 1(Associative Memory).Given a set of keys K ⊆R dk and values V ⊆R dv, associative
memory is an operator M:K → V that maps two sets of keys K and values V. To learn such
mapping from the data, an objective ˜L(·;·) measures the quality of the mapping and M can be
defined as:
M∗ = arg min
M
˜L(M(K);V).(1)
```

```text
W ∗ = arg min
W
L(W;D train),(2)
whose optimization by gradient descent results in a weight update rule equivalent to:
Wt+1 =W t −η t+1∇Wt L(Wt;x t+1)(3)
=W t −η t+1∇yt+1 L(Wt;x t+1)⊗x t+1,wherex t+1 ∼D train,(4)
```

```text
Wt+1 = arg min
W
⟨W xt+1, ut+1⟩+ 1
2ηt+1
∥W−W t∥2
2 (5)
= arg min
W
⟨W xt,∇ yt+1 L(Wt;x t+1)⟩+ 1
2ηt+1
∥W−W t∥2
2.(6)

Wt+1 =W t −m t+1,(7)
mt+1 =m t −η t+1∇Wt L(Wt;x t+1) =m t −η t+1∇yt+1 L(Wt;x t+1)⊗x t+1.(8)

Wt+1 =W t −m t+1,(9)
mt+1 = arg min
m
−⟨m,∇ Wt L(Wt;x t+1)⟩+η t+1 ∥m−m t∥2
2 (10)
= arg min
m
−⟨mx t+1,∇ yt+1 L(Wt;x t+1)⟩+η t+1 ∥m−m t∥2
2,(11)
```

```text
kt =x tWk,v t =x tWv,q t =x tWq,(12)
Mt =M t−1 +v tk⊤
t ,(13)
yt =M tqt .(14)

Mt+1 = arg min
M
⟨Mkt+1,v t+1⟩+∥M − M t∥2
2 with gradient descent,(15)
⇒ M t+1 =M t − ∇ ˜L(Mt;k t+1,v t+1) =M t +v t+1k⊤
t+1,(16)
```

```text
Wi+1 =W i +m i+1
mi+1 =α i+1mi −η t∇L(W i;x i),(17)

min
m
⟨m∇L(W i;x i)⊤,I⟩.(18)

min
m
⟨m∇L(W i;x i)⊤,P i⟩,(19)
using gradient descent, resulting in the update rule:
Wi+1 =W i +m i+1
mi+1 =α i+1mi −η tPi∇L(W i;x i).(20)

Wi+1 =W i +m i+1,(21)
mi+1 =
\left(
αi+1I− ∇L(W i;x i)⊤ ∇L(W i;x i)
\right)
mi −η tPi∇L(W i;x i),(22)

Wi+1 =W i +m i+1 (ui),andm i+1 =α i+1mi −η t∇L(2)(mi;u i,I),(23)
where ui =∇L(W i;x i) and ∇L(2)(·) is the internal objective of momentum (e.g., dot product
similarity⟨m(u ⊤
i ),1⟩). We refer to this variant as Deep Momentum Gradient Descent (DMGD).

Wi+1 =W i +σ(m i+1 (ui)),andm i+1 =α i+1mi −η t∇L(2)(mi;u i,I),(24)
```

```text
Wt+1 =W t −η t+1∇Wt L(Wt;x t) =W t −η t+1∇yt L(Wt;x t)⊗x t,wherex t ∼D train,(25)
which from the associative memory perspective is equivalent to one step of gradient descent in
optimization process of:
min
W
⟨W xt,∇ yt L(Wt;x t)⟩.(26)

min
W
∥W xt − ∇yt L(Wt;x t)∥2
2.(27)

Wt+1 =W t

I−x tx⊤
t
\right)
−η t+1∇Wt L(Wt;x t)(28)
=W t

I−x tx⊤
t
\right)
−η t+1∇yt L(Wt;x t)⊗x t,wherex t ∼D train,(29)

yt =MLP (fk)(MLP(fk−1)(· · ·MLP (f1)(xt))),(30)
where the parameters ofℓ-th MLP block, i.e.,θ (fℓ), are updated everyC (ℓ) steps:
θ(fℓ)
i+1 =θ (fℓ)
i −
(Pi
t=i−C (ℓ) η(ℓ)
t f(θ (fℓ)
t ;x t)ifi≡0 (modC (ℓ)),
0otherwise. (31)
```

### 10.3 Mapping this paper summary to current implementation status
- The current code path in this repo has implemented and validated a `tensorized_cms` path for batched fast-state updates while preserving list-mode semantics on covered tests.
- The qualification run shows strong quality parity vs list-mode but partial throughput-gate attainment (secondary gate met, primary gate missed).
- This aligns with the paper’s idea that richer multi-level memory/update structure can be made trainable, while implementation bottlenecks can still dominate wall-clock performance.

## 11) Codebase Deep Dive Appendix (Repomix Bundle, <30k tokens)
This appendix is included for senior review assuming no prior codebase context.

Generation details:
- Tool: `repomix v1.11.1`
- Output file: `reports/_tmp_repomix_senior_codepack_30k.md`
- Included files: 12
- Total tokens: 29,749 (o200k_base; from repomix summary)
- Total chars: 139,871
- Generation command:
```bash
repomix --style markdown --output reports/_tmp_repomix_senior_codepack_30k.md --include "src/nested_learning/hope/block.py,src/nested_learning/model.py,src/nested_learning/fast_state.py,src/nested_learning/functional.py,src/nested_learning/optim/manager.py,scripts/run_tensorized_qualification_3090.sh,configs/bench_small_isolated_b4.yaml,tests/test_fast_state_batch_semantics.py,tests/test_fast_state_isolation.py,tests/test_tensorized_cms_matches_listmode.py,tests/test_tensorized_cms_multilevel_matches_listmode.py,tests/test_tensorized_cms_hybrid_wiring_matches_listmode.py" --token-count-tree
```

Repomix-packed code follows:

This file is a merged representation of a subset of the codebase, containing specifically included files, combined into a single document by Repomix.

# File Summary

## Purpose
This file contains a packed representation of a subset of the repository's contents that is considered the most important context.
It is designed to be easily consumable by AI systems for analysis, code review,
or other automated processes.

## File Format
The content is organized as follows:
1. This summary section
2. Repository information
3. Directory structure
4. Repository files (if enabled)
5. Multiple file entries, each consisting of:
  a. A header with the file path (## File: path/to/file)
  b. The full contents of the file in a code block

## Usage Guidelines
- This file should be treated as read-only. Any changes should be made to the
  original repository files, not this packed version.
- When processing this file, use the file path to distinguish
  between different files in the repository.
- Be aware that this file may contain sensitive information. Handle it with
  the same level of security as you would the original repository.

## Notes
- Some files may have been excluded based on .gitignore rules and Repomix's configuration
- Binary files are not included in this packed representation. Please refer to the Repository Structure section for a complete list of file paths, including binary files
- Only files matching these patterns are included: src/nested_learning/hope/block.py, src/nested_learning/model.py, src/nested_learning/fast_state.py, src/nested_learning/functional.py, src/nested_learning/optim/manager.py, scripts/run_tensorized_qualification_3090.sh, configs/bench_small_isolated_b4.yaml, tests/test_fast_state_batch_semantics.py, tests/test_fast_state_isolation.py, tests/test_tensorized_cms_matches_listmode.py, tests/test_tensorized_cms_multilevel_matches_listmode.py, tests/test_tensorized_cms_hybrid_wiring_matches_listmode.py
- Files matching patterns in .gitignore are excluded
- Files matching default ignore patterns are excluded
- Files are sorted by Git change count (files with more changes are at the bottom)

# Directory Structure
```
configs/
  bench_small_isolated_b4.yaml
scripts/
  run_tensorized_qualification_3090.sh
src/
  nested_learning/
    hope/
      block.py
    optim/
      manager.py
    fast_state.py
    functional.py
    model.py
tests/
  test_fast_state_batch_semantics.py
  test_fast_state_isolation.py
  test_tensorized_cms_hybrid_wiring_matches_listmode.py
  test_tensorized_cms_matches_listmode.py
  test_tensorized_cms_multilevel_matches_listmode.py
```

# Files

## File: configs/bench_small_isolated_b4.yaml
```yaml
hydra:
  run:
    dir: .
  output_subdir: null
  job:
    chdir: false

defaults:
  - /bench_small_b4
  - _self_

train:
  steps: 2000
  max_runtime_seconds: 82800

logging:
  path: logs/bench_small_isolated_b4_metrics.json
  run_name: bench-small-isolated-b4
```

## File: scripts/run_tensorized_qualification_3090.sh
```bash
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
```

## File: tests/test_fast_state_isolation.py
```python
import torch

from nested_learning.fast_state import BlockFastState
from nested_learning.levels import LevelSpec
from nested_learning.model import HOPEModel, ModelConfig
from nested_learning.training import compute_teach_signal


def _tiny_config() -> ModelConfig:
    titan = LevelSpec(name="titan", update_period=1, optimizer_key="titan_opt")
    cms = [LevelSpec(name="cms_fast", update_period=1, optimizer_key="cms_opt")]
    return ModelConfig(
        vocab_size=64,
        dim=32,
        num_layers=1,
        heads=4,
        titan_level=titan,
        cms_levels=cms,
        optimizers=None,
        teach_scale=0.1,
    )


def _checksum(params: dict[str, torch.Tensor]) -> float:
    total = 0.0
    for idx, (_, value) in enumerate(sorted(params.items())):
        total += float((idx + 1) * value.detach().float().mean().item())
    return total


def _block_checksums(state: BlockFastState, sample_idx: int) -> dict[str, float]:
    payload: dict[str, float] = {}
    titan = state.titan_params
    if isinstance(titan, list):
        payload["titan"] = _checksum(titan[sample_idx])
    elif isinstance(titan, dict):
        payload["titan"] = _checksum(titan)
    for level_name, store in state.cms_params.items():
        if isinstance(store, list):
            payload[f"cms.{level_name}"] = _checksum(store[sample_idx])
        else:
            payload[f"cms.{level_name}"] = _checksum(store)
    return payload


def _run_once(model: HOPEModel, tokens: torch.Tensor) -> BlockFastState:
    state = model.init_fast_state(batch_size=tokens.size(0), fast_state_batch_mode="per_sample_list")
    with torch.no_grad():
        logits = model(tokens, fast_state=state)
        teach = compute_teach_signal(model, logits, tokens, normalization="per_sample")
        _ = model(tokens, teach_signal=teach, fast_state=state)
    return state.blocks[0]


def test_fast_state_isolation() -> None:
    torch.manual_seed(12)
    model = HOPEModel(_tiny_config())
    tokens = torch.randint(0, model.config.vocab_size, (2, 12))
    perturbed = tokens.clone()
    perturbed[0] = torch.randint(0, model.config.vocab_size, (tokens.size(1),))

    state_a = _run_once(model, tokens)
    state_b = _run_once(model, perturbed)

    sample_0_a = _block_checksums(state_a, sample_idx=0)
    sample_0_b = _block_checksums(state_b, sample_idx=0)
    sample_1_a = _block_checksums(state_a, sample_idx=1)
    sample_1_b = _block_checksums(state_b, sample_idx=1)

    for key in sample_1_a:
        assert torch.isclose(torch.tensor(sample_1_a[key]), torch.tensor(sample_1_b[key]), atol=1e-6)
    changed = any(abs(sample_0_a[k] - sample_0_b[k]) > 1e-8 for k in sample_0_a)
    assert changed
```

## File: tests/test_tensorized_cms_hybrid_wiring_matches_listmode.py
```python
from __future__ import annotations

import torch

from nested_learning.levels import LevelSpec
from nested_learning.model import HOPEModel, ModelConfig
from nested_learning.training import compute_teach_signal


def _tiny_hybrid_config() -> ModelConfig:
    titan = LevelSpec(name="titan", update_period=1, optimizer_key="titan_opt")
    cms = [
        LevelSpec(name="cms_fast", update_period=1, optimizer_key="cms_opt"),
        LevelSpec(name="cms_mid", update_period=2, optimizer_key="cms_opt"),
    ]
    return ModelConfig(
        vocab_size=64,
        dim=32,
        num_layers=1,
        heads=4,
        block_variant="hope_hybrid",
        titan_level=titan,
        cms_levels=cms,
        optimizers=None,
        teach_scale=0.1,
    )


def _sample_cms_deltas(state, sample_idx: int) -> dict[str, dict[str, torch.Tensor]]:
    block = state.blocks[0]
    out: dict[str, dict[str, torch.Tensor]] = {}
    for level_name, store in block.cms_params.items():
        if isinstance(store, list):
            out[level_name] = {name: value.detach().clone() for name, value in store[sample_idx].items()}
        else:
            out[level_name] = {
                name: value[sample_idx].detach().clone() for name, value in store.items()
            }
    return out


def _titan_checksum(state, sample_idx: int) -> float:
    block = state.blocks[0]
    titan = block.titan_params
    if titan is None:
        return 0.0
    if isinstance(titan, list):
        params = titan[sample_idx]
    else:
        params = {k: v[sample_idx] for k, v in titan.items()}
    total = 0.0
    for idx, (_, value) in enumerate(sorted(params.items())):
        total += float((idx + 1) * value.detach().float().mean().item())
    return total


def test_tensorized_cms_hybrid_wiring_matches_listmode() -> None:
    torch.manual_seed(29)
    model = HOPEModel(_tiny_hybrid_config())
    # Force CMS-only updates so hybrid wiring is exercised while titan updates are disabled.
    model.set_allowed_update_levels({"cms_fast", "cms_mid"})
    tokens = torch.randint(0, model.config.vocab_size, (3, 14))

    list_state = model.init_fast_state(batch_size=tokens.size(0), fast_state_batch_mode="per_sample_list")
    tensor_state = model.init_fast_state(
        batch_size=tokens.size(0),
        fast_state_batch_mode="tensorized_cms",
    )

    titan_before_list = [_titan_checksum(list_state, i) for i in range(tokens.size(0))]
    titan_before_tensor = [_titan_checksum(tensor_state, i) for i in range(tokens.size(0))]

    with torch.no_grad():
        logits_list = model(tokens, fast_state=list_state)
        teach_list = compute_teach_signal(model, logits_list, tokens, normalization="per_sample")
        _ = model(tokens, teach_signal=teach_list, fast_state=list_state)
        logits_after_list = model(tokens, fast_state=list_state)

        logits_tensor = model(tokens, fast_state=tensor_state)
        teach_tensor = compute_teach_signal(model, logits_tensor, tokens, normalization="per_sample")
        _ = model(tokens, teach_signal=teach_tensor, fast_state=tensor_state)
        logits_after_tensor = model(tokens, fast_state=tensor_state)

    diff = (logits_after_list - logits_after_tensor).abs().float()
    assert diff.mean().item() < 8e-2
    assert diff.max().item() < 7e-1

    for sample_idx in range(tokens.size(0)):
        list_deltas = _sample_cms_deltas(list_state, sample_idx)
        tensor_deltas = _sample_cms_deltas(tensor_state, sample_idx)
        assert list_deltas.keys() == tensor_deltas.keys()
        for level_name in list_deltas:
            assert list_deltas[level_name].keys() == tensor_deltas[level_name].keys()
            for param_name in list_deltas[level_name]:
                assert torch.allclose(
                    list_deltas[level_name][param_name],
                    tensor_deltas[level_name][param_name],
                    atol=7e-4,
                    rtol=7e-4,
                )

    titan_after_list = [_titan_checksum(list_state, i) for i in range(tokens.size(0))]
    titan_after_tensor = [_titan_checksum(tensor_state, i) for i in range(tokens.size(0))]
    for before, after in zip(titan_before_list, titan_after_list, strict=True):
        assert abs(before - after) <= 1e-7
    for before, after in zip(titan_before_tensor, titan_after_tensor, strict=True):
        assert abs(before - after) <= 1e-7
```

## File: tests/test_tensorized_cms_matches_listmode.py
```python
from __future__ import annotations

import torch

from nested_learning.levels import LevelSpec
from nested_learning.model import HOPEModel, ModelConfig
from nested_learning.training import compute_teach_signal


def _tiny_attention_config() -> ModelConfig:
    titan = LevelSpec(name="titan", update_period=1, optimizer_key="titan_opt")
    cms = [LevelSpec(name="cms_fast", update_period=1, optimizer_key="cms_opt")]
    return ModelConfig(
        vocab_size=64,
        dim=32,
        num_layers=1,
        heads=4,
        block_variant="hope_attention",
        titan_level=titan,
        cms_levels=cms,
        optimizers=None,
        teach_scale=0.1,
    )


def _sample_cms_deltas(state, sample_idx: int) -> dict[str, dict[str, torch.Tensor]]:
    block = state.blocks[0]
    out: dict[str, dict[str, torch.Tensor]] = {}
    for level_name, store in block.cms_params.items():
        if isinstance(store, list):
            out[level_name] = {name: value.detach().clone() for name, value in store[sample_idx].items()}
        else:
            out[level_name] = {
                name: value[sample_idx].detach().clone() for name, value in store.items()
            }
    return out


def test_tensorized_cms_matches_listmode() -> None:
    torch.manual_seed(17)
    model = HOPEModel(_tiny_attention_config())
    tokens = torch.randint(0, model.config.vocab_size, (3, 12))

    list_state = model.init_fast_state(batch_size=tokens.size(0), fast_state_batch_mode="per_sample_list")
    tensor_state = model.init_fast_state(
        batch_size=tokens.size(0),
        fast_state_batch_mode="tensorized_cms",
    )

    with torch.no_grad():
        logits_list = model(tokens, fast_state=list_state)
        teach_list = compute_teach_signal(model, logits_list, tokens, normalization="per_sample")
        _ = model(tokens, teach_signal=teach_list, fast_state=list_state)
        logits_after_list = model(tokens, fast_state=list_state)

        logits_tensor = model(tokens, fast_state=tensor_state)
        teach_tensor = compute_teach_signal(model, logits_tensor, tokens, normalization="per_sample")
        _ = model(tokens, teach_signal=teach_tensor, fast_state=tensor_state)
        logits_after_tensor = model(tokens, fast_state=tensor_state)

    diff = (logits_after_list - logits_after_tensor).abs().float()
    assert diff.mean().item() < 5e-2
    assert diff.max().item() < 5e-1

    for sample_idx in range(tokens.size(0)):
        list_deltas = _sample_cms_deltas(list_state, sample_idx)
        tensor_deltas = _sample_cms_deltas(tensor_state, sample_idx)
        assert list_deltas.keys() == tensor_deltas.keys()
        for level_name in list_deltas:
            assert list_deltas[level_name].keys() == tensor_deltas[level_name].keys()
            for param_name in list_deltas[level_name]:
                assert torch.allclose(
                    list_deltas[level_name][param_name],
                    tensor_deltas[level_name][param_name],
                    atol=5e-4,
                    rtol=5e-4,
                )
```

## File: tests/test_tensorized_cms_multilevel_matches_listmode.py
```python
from __future__ import annotations

import torch

from nested_learning.levels import LevelSpec
from nested_learning.model import HOPEModel, ModelConfig
from nested_learning.training import compute_teach_signal


def _tiny_attention_multilevel_config() -> ModelConfig:
    titan = LevelSpec(name="titan", update_period=1, optimizer_key="titan_opt")
    cms = [
        LevelSpec(name="cms_fast", update_period=1, optimizer_key="cms_opt"),
        LevelSpec(name="cms_mid", update_period=2, optimizer_key="cms_opt"),
    ]
    return ModelConfig(
        vocab_size=64,
        dim=32,
        num_layers=1,
        heads=4,
        block_variant="hope_attention",
        titan_level=titan,
        cms_levels=cms,
        optimizers=None,
        teach_scale=0.1,
    )


def _sample_cms_deltas(state, sample_idx: int) -> dict[str, dict[str, torch.Tensor]]:
    block = state.blocks[0]
    out: dict[str, dict[str, torch.Tensor]] = {}
    for level_name, store in block.cms_params.items():
        if isinstance(store, list):
            out[level_name] = {name: value.detach().clone() for name, value in store[sample_idx].items()}
        else:
            out[level_name] = {
                name: value[sample_idx].detach().clone() for name, value in store.items()
            }
    return out


def test_tensorized_cms_multilevel_matches_listmode() -> None:
    torch.manual_seed(23)
    model = HOPEModel(_tiny_attention_multilevel_config())
    tokens = torch.randint(0, model.config.vocab_size, (3, 14))

    list_state = model.init_fast_state(batch_size=tokens.size(0), fast_state_batch_mode="per_sample_list")
    tensor_state = model.init_fast_state(
        batch_size=tokens.size(0),
        fast_state_batch_mode="tensorized_cms",
    )

    with torch.no_grad():
        logits_list = model(tokens, fast_state=list_state)
        teach_list = compute_teach_signal(model, logits_list, tokens, normalization="per_sample")
        _ = model(tokens, teach_signal=teach_list, fast_state=list_state)
        logits_after_list = model(tokens, fast_state=list_state)

        logits_tensor = model(tokens, fast_state=tensor_state)
        teach_tensor = compute_teach_signal(model, logits_tensor, tokens, normalization="per_sample")
        _ = model(tokens, teach_signal=teach_tensor, fast_state=tensor_state)
        logits_after_tensor = model(tokens, fast_state=tensor_state)

    diff = (logits_after_list - logits_after_tensor).abs().float()
    assert diff.mean().item() < 7e-2
    assert diff.max().item() < 6e-1

    for sample_idx in range(tokens.size(0)):
        list_deltas = _sample_cms_deltas(list_state, sample_idx)
        tensor_deltas = _sample_cms_deltas(tensor_state, sample_idx)
        assert list_deltas.keys() == tensor_deltas.keys()
        for level_name in list_deltas:
            assert list_deltas[level_name].keys() == tensor_deltas[level_name].keys()
            for param_name in list_deltas[level_name]:
                assert torch.allclose(
                    list_deltas[level_name][param_name],
                    tensor_deltas[level_name][param_name],
                    atol=6e-4,
                    rtol=6e-4,
                )
```

## File: tests/test_fast_state_batch_semantics.py
```python
import pytest
from omegaconf import OmegaConf

from nested_learning.training import _validate_fast_state_batch_semantics


def test_fast_state_batch_semantics_raises_when_strict() -> None:
    cfg = OmegaConf.create(
        {
            "train": {"use_fast_state": True, "fail_if_paper_faithful_disabled": True},
            "data": {"batch_size": 2},
        }
    )
    with pytest.raises(RuntimeError, match="fast-state"):
        _validate_fast_state_batch_semantics(cfg)


def test_fast_state_batch_semantics_allows_batch1() -> None:
    cfg = OmegaConf.create(
        {
            "train": {"use_fast_state": True, "fail_if_paper_faithful_disabled": True},
            "data": {"batch_size": 1},
        }
    )
    _validate_fast_state_batch_semantics(cfg)


def test_fast_state_batch_semantics_allows_per_sample_list_mode() -> None:
    cfg = OmegaConf.create(
        {
            "train": {
                "use_fast_state": True,
                "fail_if_paper_faithful_disabled": True,
                "fast_state_batch_mode": "per_sample_list",
            },
            "data": {"batch_size": 4},
        }
    )
    _validate_fast_state_batch_semantics(cfg)


def test_fast_state_batch_semantics_allows_tensorized_cms_mode() -> None:
    cfg = OmegaConf.create(
        {
            "train": {
                "use_fast_state": True,
                "fail_if_paper_faithful_disabled": True,
                "fast_state_batch_mode": "tensorized_cms",
            },
            "data": {"batch_size": 4},
        }
    )
    _validate_fast_state_batch_semantics(cfg)
```

## File: src/nested_learning/fast_state.py
```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, cast

import torch
from torch import nn

from .optim.manager import LevelConfig, LevelOptimizerManager
from .titan.self_modifying import SelfModifyingTitansState

ParamDict = Dict[str, torch.Tensor]
ParamStore = ParamDict | list[ParamDict]
ManagerStore = LevelOptimizerManager | list[LevelOptimizerManager]
SelfModStore = SelfModifyingTitansState | list[SelfModifyingTitansState] | None


def init_module_deltas(module: nn.Module) -> ParamDict:
    """
    Initialize a per-parameter "fast state" delta dict for meta+delta fast state.

    The fast state stores *deltas* (initialized to 0) rather than detached parameter clones so that
    forward passes can use `meta_param + delta`, allowing outer gradients to flow to meta params
    while keeping online updates as stop-grad writes into the delta tensors.
    """

    return {name: torch.zeros_like(param).detach() for name, param in module.named_parameters()}


def init_module_deltas_batched(module: nn.Module, *, batch_size: int) -> ParamDict:
    return {
        name: torch.zeros((batch_size, *param.shape), device=param.device, dtype=param.dtype).detach()
        for name, param in module.named_parameters()
    }


@dataclass
class BlockFastState:
    titan_params: ParamStore | None
    cms_params: Dict[str, ParamStore]
    level_manager: ManagerStore
    selfmod_state: SelfModStore = None
    batch_mode: str = "shared"


def build_block_fast_state(
    *,
    titan_module: nn.Module | None,
    cms_blocks: Dict[str, nn.Module],
    selfmod_module: nn.Module | None = None,
    specs,
    optimizer_configs: Dict[str, dict],
    default_lr: float,
    batch_size: int = 1,
    fast_state_batch_mode: str = "shared",
) -> BlockFastState:
    mode = str(fast_state_batch_mode).strip().lower()
    if mode not in {"shared", "per_sample_list", "tensorized_cms"}:
        raise ValueError(
            f"Unsupported fast_state_batch_mode={fast_state_batch_mode!r}; "
            "expected one of ['shared', 'per_sample_list', 'tensorized_cms']"
        )
    if mode in {"per_sample_list", "tensorized_cms"} and batch_size <= 0:
        raise ValueError(f"batch_size must be > 0 for fast_state_batch_mode={mode!r}")

    level_cfg = LevelConfig(specs=specs, optimizer_configs=optimizer_configs, default_lr=default_lr)

    titan_params: ParamStore | None = None
    if titan_module is not None:
        if mode == "shared":
            titan_params = init_module_deltas(titan_module)
        else:
            titan_params = [init_module_deltas(titan_module) for _ in range(batch_size)]

    cms_params: Dict[str, ParamStore] = {}
    for name, block in cms_blocks.items():
        if mode == "shared":
            cms_params[name] = init_module_deltas(block)
        elif mode == "tensorized_cms":
            cms_params[name] = init_module_deltas_batched(block, batch_size=batch_size)
        else:
            cms_params[name] = [init_module_deltas(block) for _ in range(batch_size)]

    if mode == "shared":
        level_manager: ManagerStore = LevelOptimizerManager(level_cfg)
    else:
        level_manager = [LevelOptimizerManager(level_cfg) for _ in range(batch_size)]

    selfmod_state: SelfModStore = None
    if selfmod_module is not None:
        init_fn = getattr(selfmod_module, "init_fast_state", None)
        if callable(init_fn):
            if mode == "shared":
                selfmod_state = cast(SelfModifyingTitansState, init_fn())
            else:
                selfmod_state = [cast(SelfModifyingTitansState, init_fn()) for _ in range(batch_size)]
    return BlockFastState(
        titan_params=titan_params,
        cms_params=cms_params,
        level_manager=level_manager,
        selfmod_state=selfmod_state,
        batch_mode=mode,
    )


@dataclass
class ModelFastState:
    blocks: list[BlockFastState]
```

## File: src/nested_learning/functional.py
```python
from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

import torch
from torch import nn
from torch.func import functional_call, vmap

ParamDict = Dict[str, torch.Tensor]


def params_with_deltas(module: nn.Module, deltas: ParamDict) -> ParamDict:
    params: ParamDict = {}
    missing: list[str] = []
    for name, param in module.named_parameters():
        delta = deltas.get(name)
        if delta is None:
            missing.append(name)
            continue
        params[name] = param + delta
    if missing:
        raise KeyError(
            f"Missing fast-state delta(s) for {module.__class__.__name__}: {sorted(missing)[:10]}"
        )
    return params


def module_buffers(module: nn.Module) -> ParamDict:
    return {name: buf for name, buf in module.named_buffers()}


def call_with_params(
    module: nn.Module,
    params: ParamDict,
    *args: Any,
    **kwargs: Any,
) -> Any:
    buffers = module_buffers(module)
    return functional_call(module, (params, buffers), args, kwargs, strict=True)


def call_with_deltas(
    module: nn.Module,
    deltas: ParamDict,
    *args: Any,
    **kwargs: Any,
) -> Any:
    return call_with_params(module, params_with_deltas(module, deltas), *args, **kwargs)


def call_with_batched_deltas(
    module: nn.Module,
    batched_deltas: ParamDict,
    inputs: torch.Tensor,
) -> torch.Tensor:
    """
    Vectorized per-sample fast-state forward.

    `batched_deltas` stores one delta slice per sample with shape [B, *param.shape].
    """

    if inputs.ndim < 1:
        raise ValueError("inputs must include a batch dimension")
    batch_size = int(inputs.size(0))
    for name, param in module.named_parameters():
        delta = batched_deltas.get(name)
        if delta is None:
            raise KeyError(f"Missing batched delta for parameter {name!r}")
        if delta.ndim != param.ndim + 1:
            raise ValueError(
                f"Batched delta for {name!r} has rank {delta.ndim}; expected {param.ndim + 1}"
            )
        if int(delta.size(0)) != batch_size:
            raise ValueError(
                f"Batched delta for {name!r} has batch {int(delta.size(0))}; expected {batch_size}"
            )

    def _single_forward(single_deltas: ParamDict, single_inputs: torch.Tensor) -> torch.Tensor:
        return call_with_deltas(module, single_deltas, single_inputs.unsqueeze(0)).squeeze(0)

    return vmap(_single_forward)(batched_deltas, inputs)


def require_grad_params(params: Mapping[str, torch.Tensor]) -> ParamDict:
    return {name: value.detach().requires_grad_(True) for name, value in params.items()}


def grads_to_dict(params: ParamDict, grads: Tuple[torch.Tensor | None, ...]) -> ParamDict:
    out: ParamDict = {}
    for (name, _), grad in zip(params.items(), grads, strict=True):
        if grad is None:
            continue
        out[name] = grad
    return out
```

## File: src/nested_learning/optim/manager.py
```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Sequence, Tuple

import torch
from torch import nn

from ..levels import LevelClock, LevelSpec
from .factory import build_optimizer


@dataclass
class LevelConfig:
    specs: Sequence[LevelSpec]
    optimizer_configs: Dict[str, dict]
    default_lr: float


class LevelOptimizerManager:
    def __init__(self, config: LevelConfig):
        self.clock = LevelClock(config.specs)
        self.learning_rates: Dict[str, float] = {}
        self.optimizers = {}
        self._last_metrics: Dict[str, Dict[str, float]] = {}
        for spec in config.specs:
            key = spec.optimizer_key or "default"
            optim_cfg = config.optimizer_configs.get(key, {"type": "deep_momentum", "params": {}})
            lr = optim_cfg.get("lr", config.default_lr)
            params_cfg = optim_cfg.get("params", {})
            optimizer = build_optimizer(
                {"type": optim_cfg.get("type", "deep_momentum"), "params": params_cfg}
            )
            self.optimizers[spec.name] = optimizer
            self.learning_rates[spec.name] = lr

    def should_update(self, level: str) -> bool:
        return self.clock.should_update(level)

    def optimize(
        self,
        level: str,
        module: nn.Module,
        loss: torch.Tensor,
        *,
        context: torch.Tensor | None = None,
        force: bool = False,
    ) -> float:
        if (not force) and (not self.should_update(level)):
            return 0.0
        named_params: Tuple[Tuple[str, torch.nn.Parameter], ...] = tuple(
            (name, param) for name, param in module.named_parameters() if param.requires_grad
        )
        if not named_params:
            return 0.0
        params = tuple(param for _, param in named_params)
        grads = torch.autograd.grad(loss, params, retain_graph=False, allow_unused=True)
        grads_dict: Dict[str, torch.Tensor] = {}
        for (name, _), grad in zip(named_params, grads, strict=True):
            if grad is None:
                continue
            grads_dict[name] = grad
        return self.apply_module_grads(
            level,
            module,
            grads_dict,
            context=context,
            force=True,
        )

    def apply_module_grads(
        self,
        level: str,
        module: nn.Module,
        grads: Dict[str, torch.Tensor],
        *,
        context: torch.Tensor | None = None,
        force: bool = False,
    ) -> float:
        if (not force) and (not self.should_update(level)):
            return 0.0
        optimizer = self.optimizers[level]
        lr = self.learning_rates[level]
        total_norm = 0.0
        for name, param in module.named_parameters():
            if not param.requires_grad:
                continue
            grad = grads.get(name)
            if grad is None:
                continue
            update = optimizer(grad, context=context, param_key=name)
            with torch.no_grad():
                param.add_(update, alpha=-lr)
            total_norm += grad.norm().item()
        self.clock.record_update(level)
        metrics = getattr(optimizer, "last_metrics", None)
        if metrics:
            self._last_metrics[level] = dict(metrics)
        else:
            self._last_metrics[level] = {}
        return total_norm

    def tick(self) -> None:
        self.clock.tick()

    def pop_last_metrics(self, level: str) -> Dict[str, float]:
        return self._last_metrics.pop(level, {})

    def apply_grads(
        self,
        level: str,
        params: Dict[str, torch.Tensor],
        grads: Dict[str, torch.Tensor],
        *,
        context: torch.Tensor | None = None,
        force: bool = False,
    ) -> tuple[Dict[str, torch.Tensor], float]:
        if (not force) and (not self.should_update(level)):
            return params, 0.0
        optimizer = self.optimizers[level]
        lr = self.learning_rates[level]
        updated: Dict[str, torch.Tensor] = {}
        total_norm = 0.0
        for name, param in params.items():
            grad = grads.get(name)
            if grad is None:
                updated[name] = param
                continue
            update = optimizer(grad, context=context, param_key=name)
            with torch.no_grad():
                updated[name] = (param - lr * update).detach()
            total_norm += grad.norm().item()
        self.clock.record_update(level)
        metrics = getattr(optimizer, "last_metrics", None)
        if metrics:
            self._last_metrics[level] = dict(metrics)
        else:
            self._last_metrics[level] = {}
        return updated, total_norm
```

## File: src/nested_learning/hope/block.py
```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Sequence, Set

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..backbones import AttentionConfig, SelfAttention
from ..cms import CMS
from ..fast_state import BlockFastState
from ..functional import (
    call_with_batched_deltas,
    call_with_deltas,
    call_with_params,
    grads_to_dict,
    params_with_deltas,
    require_grad_params,
)
from ..levels import LevelSpec
from ..optim.manager import LevelConfig, LevelOptimizerManager
from ..titan.memory import TitanMemory, TitanMemoryConfig
from ..titan.self_modifying import SelfModifyingTitans, SelfModifyingTitansConfig
from .self_mod import SelfModifier


def _chunk_loss(
    prediction: torch.Tensor,
    delta_target: torch.Tensor,
    mask_f: torch.Tensor,
    *,
    reduction: str,
) -> torch.Tensor:
    target = (prediction.detach() - delta_target).detach()
    diff_sq = (prediction - target).pow(2)
    masked = diff_sq * mask_f
    if reduction == "mean":
        return masked.sum() / mask_f.sum().clamp(min=1.0)
    if reduction == "sum":
        return masked.sum()
    raise ValueError(f"Unsupported cms_chunk_reduction={reduction}")


def _min_update_period(levels: Sequence[LevelSpec]) -> int:
    periods = [int(spec.update_period) for spec in levels if int(spec.update_period) > 0]
    return min(periods) if periods else 1


def _tick_manager(level_manager: LevelOptimizerManager | list[LevelOptimizerManager]) -> None:
    if isinstance(level_manager, list):
        for manager in level_manager:
            manager.tick()
        return
    level_manager.tick()


def _manager_for_sample(
    level_manager: LevelOptimizerManager | list[LevelOptimizerManager],
    sample_idx: int,
) -> LevelOptimizerManager:
    if isinstance(level_manager, list):
        return level_manager[sample_idx]
    return level_manager


def _call_with_deltas_maybe_list(
    module: nn.Module,
    deltas: Dict[str, torch.Tensor] | list[Dict[str, torch.Tensor]],
    inputs: torch.Tensor,
) -> torch.Tensor:
    if isinstance(deltas, list):
        outputs = []
        for idx, sample_params in enumerate(deltas):
            outputs.append(call_with_deltas(module, sample_params, inputs[idx : idx + 1]))
        return torch.cat(outputs, dim=0)
    if _is_batched_delta_dict(module, deltas):
        return call_with_batched_deltas(module, deltas, inputs)
    return call_with_deltas(module, deltas, inputs)


def _is_batched_delta_dict(module: nn.Module, deltas: Dict[str, torch.Tensor]) -> bool:
    flags: list[bool] = []
    for name, param in module.named_parameters():
        delta = deltas.get(name)
        if delta is None:
            continue
        if delta.ndim == param.ndim:
            flags.append(False)
            continue
        if delta.ndim == param.ndim + 1:
            flags.append(True)
            continue
        raise ValueError(
            f"Delta rank mismatch for {name!r}: got {delta.ndim}, expected {param.ndim} or {param.ndim + 1}"
        )
    if not flags:
        return False
    if any(flags) and not all(flags):
        raise ValueError("Mixed batched and unbatched fast-state deltas are not supported")
    return all(flags)


def _apply_tensorized_cms_grads(
    *,
    module: nn.Module,
    level_name: str,
    base_params: Dict[str, torch.Tensor],
    chunk_inputs: torch.Tensor,
    chunk_teach: torch.Tensor,
    chunk_active: torch.Tensor,
    reduction: str,
    level_manager: LevelOptimizerManager | list[LevelOptimizerManager],
) -> tuple[Dict[str, torch.Tensor], float]:
    mask_f = chunk_active.unsqueeze(-1).float()
    params_req = require_grad_params(base_params)
    with torch.enable_grad():
        prediction = call_with_batched_deltas(module, params_req, chunk_inputs)
        loss = _chunk_loss(
            prediction,
            chunk_teach,
            mask_f,
            reduction=reduction,
        )
    grads = torch.autograd.grad(
        loss,
        tuple(params_req.values()),
        retain_graph=False,
        allow_unused=True,
    )
    grads_dict = grads_to_dict(params_req, grads)
    contexts = chunk_inputs.mean(dim=1)
    updated_batched = {name: value.detach().clone() for name, value in base_params.items()}
    active_samples = chunk_active.any(dim=1)
    total_magnitude = 0.0
    for sample_idx in range(int(chunk_inputs.size(0))):
        if not bool(active_samples[sample_idx]):
            continue
        sample_params = {name: value[sample_idx] for name, value in base_params.items()}
        sample_grads = {name: value[sample_idx] for name, value in grads_dict.items()}
        manager = _manager_for_sample(level_manager, sample_idx)
        updated, magnitude = manager.apply_grads(
            level_name,
            sample_params,
            sample_grads,
            context=contexts[sample_idx],
            force=True,
        )
        manager.pop_last_metrics(level_name)
        for name, value in updated.items():
            updated_batched[name][sample_idx] = value
        total_magnitude += magnitude
    return {name: value.detach() for name, value in updated_batched.items()}, total_magnitude


@dataclass
class _CmsBuffer:
    inputs: list[torch.Tensor]
    teach: list[torch.Tensor]
    active: list[torch.Tensor]
    count: int = 0


def _pop_buffer_chunk(
    buffer: _CmsBuffer,
    count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if count <= 0:
        raise ValueError("count must be positive")
    result_inputs: list[torch.Tensor] = []
    result_teach: list[torch.Tensor] = []
    result_active: list[torch.Tensor] = []
    remaining = count
    while remaining > 0:
        first = buffer.inputs[0]
        chunk_len = first.size(1)
        take = min(remaining, chunk_len)
        src_inputs = buffer.inputs[0]
        src_teach = buffer.teach[0]
        src_active = buffer.active[0]
        result_inputs.append(src_inputs[:, :take])
        result_teach.append(src_teach[:, :take])
        result_active.append(src_active[:, :take])
        if take == chunk_len:
            buffer.inputs.pop(0)
            buffer.teach.pop(0)
            buffer.active.pop(0)
        else:
            buffer.inputs[0] = src_inputs[:, take:]
            buffer.teach[0] = src_teach[:, take:]
            buffer.active[0] = src_active[:, take:]
        remaining -= take
    return (
        torch.cat(result_inputs, dim=1),
        torch.cat(result_teach, dim=1),
        torch.cat(result_active, dim=1),
    )


@dataclass
class HOPEBlockConfig:
    dim: int
    heads: int
    titan_level: LevelSpec
    cms_levels: Sequence[LevelSpec]
    titan_hidden_multiplier: int = 4
    cms_hidden_multiplier: int = 4
    cms_use_layernorm: bool = True
    activation: str = "gelu"
    qk_l2_norm: bool = False
    local_conv_window: int | None = None
    self_mod_hidden: int = 4
    self_mod_lr: float = 1e-3
    cms_chunk_reduction: str = "sum"
    cms_online_updates: bool = True
    cms_flush_partial_at_end: bool = False
    optimizer_configs: Dict[str, dict] = field(default_factory=dict)


@dataclass
class HOPEAttentionBlockConfig:
    dim: int
    heads: int
    cms_levels: Sequence[LevelSpec]
    cms_hidden_multiplier: int = 4
    cms_use_layernorm: bool = True
    activation: str = "gelu"
    qk_l2_norm: bool = False
    local_conv_window: int | None = None
    self_mod_lr: float = 1e-3
    cms_chunk_reduction: str = "sum"
    cms_online_updates: bool = True
    cms_flush_partial_at_end: bool = False
    optimizer_configs: Dict[str, dict] = field(default_factory=dict)


class HOPEAttentionBlock(nn.Module):
    """
    Paper-defined HOPE-Attention variant: softmax attention followed by CMS.

    Reference: Nested Learning paper, HOPE-Attention note under Eqs. 94–97.
    """

    def __init__(self, config: HOPEAttentionBlockConfig):
        super().__init__()
        self.config = config
        self.last_update_stats: Dict[str, Dict[str, float]] = {}
        self.surprise_threshold: float | None = None
        self.surprise_metric: str = "l2"
        self.allowed_levels: Set[str] | None = None
        self.attn = SelfAttention(
            AttentionConfig(
                dim=config.dim,
                heads=config.heads,
                qk_l2_norm=config.qk_l2_norm,
                local_conv_window=config.local_conv_window,
            )
        )
        self.cms = CMS(
            dim=config.dim,
            levels=config.cms_levels,
            hidden_multiplier=config.cms_hidden_multiplier,
            activation=config.activation,
            use_layernorm=config.cms_use_layernorm,
        )
        level_config = LevelConfig(
            specs=config.cms_levels,
            optimizer_configs=config.optimizer_configs,
            default_lr=config.self_mod_lr,
        )
        self.level_manager = LevelOptimizerManager(level_config)

    def forward(
        self,
        x: torch.Tensor,
        *,
        teach_signal: torch.Tensor | None = None,
        surprise_value: float | None = None,
        fast_state: BlockFastState | None = None,
    ) -> torch.Tensor:
        attn_out = self.attn(x)
        if fast_state is None:
            if teach_signal is not None and self.config.cms_online_updates:
                cms_out = self._cms_forward_online(attn_out, teach_signal, surprise_value)
            else:
                cms_result = self.cms(attn_out, return_intermediates=True)
                cms_out, cms_inputs, cms_outputs = cms_result
                if teach_signal is not None:
                    self._update_cms(cms_inputs, cms_outputs, teach_signal, surprise_value)
            self.level_manager.tick()
            return cms_out
        if teach_signal is not None and self.config.cms_online_updates:
            cms_out = self._cms_forward_online_fast(
                attn_out, fast_state, teach_signal, surprise_value
            )
        else:
            cms_out, cms_inputs = self._cms_forward_fast(attn_out, fast_state)
            if teach_signal is not None:
                self._update_cms_fast(fast_state, cms_inputs, teach_signal, surprise_value)
        _tick_manager(fast_state.level_manager)
        return cms_out

    def set_surprise_threshold(self, threshold: float | None) -> None:
        self.surprise_threshold = threshold

    def set_surprise_metric(self, metric: str) -> None:
        self.surprise_metric = str(metric).strip().lower()

    def set_allowed_levels(self, allowed: Set[str] | None) -> None:
        self.allowed_levels = allowed.copy() if allowed is not None else None

    def pop_update_stats(self) -> Dict[str, Dict[str, float]]:
        stats = self.last_update_stats
        self.last_update_stats = {}
        return stats

    def _cms_forward_fast(
        self,
        x: torch.Tensor,
        fast_state: BlockFastState,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        current = x
        inputs: dict[str, torch.Tensor] = {}
        for spec in self.config.cms_levels:
            level_name = spec.name
            inputs[level_name] = current
            params = fast_state.cms_params[level_name]
            current = _call_with_deltas_maybe_list(self.cms.blocks[level_name], params, current)
        return current, inputs

    def _cms_forward_online(
        self,
        x: torch.Tensor,
        teach_signal: torch.Tensor,
        surprise_value: float | None,
    ) -> torch.Tensor:
        seq_len = x.shape[1]
        base_chunk = _min_update_period(self.config.cms_levels)
        active_mask = teach_signal.detach().abs().sum(dim=-1) > 0
        outputs: list[torch.Tensor] = []
        stats: dict[str, Dict[str, float]] = {}
        buffers: dict[str, _CmsBuffer] = {}
        for spec in self.config.cms_levels:
            buffers[spec.name] = _CmsBuffer(inputs=[], teach=[], active=[], count=0)
            stats[spec.name] = {"grad_norm": 0.0, "chunk_tokens": 0.0, "gate_hit": 0.0}

        for start in range(0, seq_len, base_chunk):
            end = min(start + base_chunk, seq_len)
            chunk_in = x[:, start:end, :]
            chunk_teach = teach_signal[:, start:end, :]
            chunk_active = active_mask[:, start:end]

            current = chunk_in
            level_inputs: dict[str, torch.Tensor] = {}
            for spec in self.config.cms_levels:
                level_name = spec.name
                level_inputs[level_name] = current
                current = self.cms.blocks[level_name](current)
            outputs.append(current)

            for spec in self.config.cms_levels:
                level_name = spec.name
                buffer = buffers[level_name]
                buffer.inputs.append(level_inputs[level_name].detach())
                buffer.teach.append(chunk_teach)
                buffer.active.append(chunk_active)
                buffer.count += end - start
                update_period = int(spec.update_period)
                while update_period > 0 and buffer.count >= update_period:
                    chunk_inputs, chunk_teach, chunk_active = _pop_buffer_chunk(
                        buffer, update_period
                    )
                    buffer.count -= update_period
                    magnitude = self._update_cms_chunk(
                        level_name,
                        chunk_inputs,
                        chunk_teach,
                        chunk_active,
                        surprise_value,
                    )
                    if magnitude > 0:
                        stats[level_name]["grad_norm"] += magnitude
                        stats[level_name]["chunk_tokens"] += float(update_period)
                        stats[level_name]["gate_hit"] += 1.0
        if self.config.cms_flush_partial_at_end:
            for spec in self.config.cms_levels:
                level_name = spec.name
                buffer = buffers[level_name]
                remaining = int(buffer.count)
                if remaining <= 0:
                    continue
                chunk_inputs, chunk_teach, chunk_active = _pop_buffer_chunk(buffer, remaining)
                buffer.count -= remaining
                if not bool(chunk_active.any()):
                    continue
                magnitude = self._update_cms_chunk(
                    level_name,
                    chunk_inputs,
                    chunk_teach,
                    chunk_active,
                    surprise_value,
                )
                if magnitude > 0:
                    stats[level_name]["grad_norm"] += magnitude
                    stats[level_name]["chunk_tokens"] += float(remaining)
                    stats[level_name]["gate_hit"] += 1.0
        for level_name, payload in stats.items():
            if payload["gate_hit"] <= 0:
                continue
            if surprise_value is not None:
                payload["surprise_value"] = surprise_value
            self.last_update_stats[f"cms.{level_name}"] = payload
        return torch.cat(outputs, dim=1)

    def _cms_forward_online_fast(
        self,
        x: torch.Tensor,
        fast_state: BlockFastState,
        teach_signal: torch.Tensor,
        surprise_value: float | None,
    ) -> torch.Tensor:
        seq_len = x.shape[1]
        base_chunk = _min_update_period(self.config.cms_levels)
        active_mask = teach_signal.detach().abs().sum(dim=-1) > 0
        outputs: list[torch.Tensor] = []
        stats: dict[str, Dict[str, float]] = {}
        buffers: dict[str, _CmsBuffer] = {}
        for spec in self.config.cms_levels:
            buffers[spec.name] = _CmsBuffer(inputs=[], teach=[], active=[], count=0)
            stats[spec.name] = {"grad_norm": 0.0, "chunk_tokens": 0.0, "gate_hit": 0.0}

        for start in range(0, seq_len, base_chunk):
            end = min(start + base_chunk, seq_len)
            chunk_in = x[:, start:end, :]
            chunk_teach = teach_signal[:, start:end, :]
            chunk_active = active_mask[:, start:end]

            current = chunk_in
            level_inputs: dict[str, torch.Tensor] = {}
            for spec in self.config.cms_levels:
                level_name = spec.name
                level_inputs[level_name] = current
                params = fast_state.cms_params[level_name]
                current = _call_with_deltas_maybe_list(
                    self.cms.blocks[level_name], params, current
                )
            outputs.append(current)

            for spec in self.config.cms_levels:
                level_name = spec.name
                buffer = buffers[level_name]
                buffer.inputs.append(level_inputs[level_name].detach())
                buffer.teach.append(chunk_teach)
                buffer.active.append(chunk_active)
                buffer.count += end - start
                update_period = int(spec.update_period)
                while update_period > 0 and buffer.count >= update_period:
                    chunk_inputs, chunk_teach, chunk_active = _pop_buffer_chunk(
                        buffer, update_period
                    )
                    buffer.count -= update_period
                    magnitude = self._update_cms_chunk_fast(
                        fast_state,
                        level_name,
                        chunk_inputs,
                        chunk_teach,
                        chunk_active,
                        surprise_value,
                    )
                    if magnitude > 0:
                        stats[level_name]["grad_norm"] += magnitude
                        stats[level_name]["chunk_tokens"] += float(update_period)
                        stats[level_name]["gate_hit"] += 1.0
        if self.config.cms_flush_partial_at_end:
            for spec in self.config.cms_levels:
                level_name = spec.name
                buffer = buffers[level_name]
                remaining = int(buffer.count)
                if remaining <= 0:
                    continue
                chunk_inputs, chunk_teach, chunk_active = _pop_buffer_chunk(buffer, remaining)
                buffer.count -= remaining
                if not bool(chunk_active.any()):
                    continue
                magnitude = self._update_cms_chunk_fast(
                    fast_state,
                    level_name,
                    chunk_inputs,
                    chunk_teach,
                    chunk_active,
                    surprise_value,
                )
                if magnitude > 0:
                    stats[level_name]["grad_norm"] += magnitude
                    stats[level_name]["chunk_tokens"] += float(remaining)
                    stats[level_name]["gate_hit"] += 1.0
        for level_name, payload in stats.items():
            if payload["gate_hit"] <= 0:
                continue
            if surprise_value is not None:
                payload["surprise_value"] = surprise_value
            self.last_update_stats[f"cms.{level_name}"] = payload
        return torch.cat(outputs, dim=1)

    def _update_cms_fast(
        self,
        fast_state: BlockFastState,
        cms_inputs: dict[str, torch.Tensor],
        teach_signal: torch.Tensor,
        surprise_value: float | None,
    ) -> None:
        teach = teach_signal.detach()
        active_mask = teach.abs().sum(dim=-1) > 0
        for spec in self.config.cms_levels:
            level_name = spec.name
            if not self._is_level_allowed(level_name):
                continue
            if not self._passes_surprise(surprise_value):
                self._record_gate(level_name, hit=False)
                continue
            inputs = cms_inputs[level_name]
            seq_len = inputs.shape[1]
            chunk_size = int(spec.update_period)
            if chunk_size <= 0:
                continue
            total_norm = 0.0
            update_events = 0
            token_events = 0
            for start in range(0, seq_len, chunk_size):
                end = min(start + chunk_size, seq_len)
                chunk_len = end - start
                chunk_inputs = inputs[:, start:end, :].detach()
                chunk_teach = teach[:, start:end, :]
                chunk_active = active_mask[:, start:end]
                if not bool(chunk_active.any()):
                    continue
                magnitude = self._update_cms_chunk_fast(
                    fast_state,
                    level_name,
                    chunk_inputs,
                    chunk_teach,
                    chunk_active,
                    surprise_value,
                )
                if magnitude <= 0:
                    continue
                total_norm += magnitude
                token_events += chunk_len
                update_events += 1
            if update_events == 0:
                continue
            stats_payload: Dict[str, float] = {
                "grad_norm": total_norm,
                "chunk_tokens": float(token_events),
                "gate_hit": float(update_events),
            }
            if surprise_value is not None:
                stats_payload["surprise_value"] = surprise_value
            self.last_update_stats[f"cms.{level_name}"] = stats_payload

    def _is_level_allowed(self, level_name: str) -> bool:
        if self.allowed_levels is None:
            return True
        return level_name in self.allowed_levels

    def _passes_surprise(self, surprise_value: float | None) -> bool:
        if self.surprise_threshold is None:
            return True
        if surprise_value is None:
            return False
        return surprise_value >= self.surprise_threshold

    def _record_gate(self, level_name: str, *, hit: bool) -> None:
        stats_key = f"gate.{level_name}"
        self.last_update_stats.setdefault(stats_key, {})
        self.last_update_stats[stats_key]["gate_hit"] = 1.0 if hit else 0.0

    def _update_cms(
        self,
        cms_inputs: dict[str, torch.Tensor],
        cms_outputs: dict[str, torch.Tensor],
        teach_signal: torch.Tensor,
        surprise_value: float | None,
    ) -> None:
        teach = teach_signal.detach()
        active_mask = teach.abs().sum(dim=-1) > 0
        for spec in self.config.cms_levels:
            level_name = spec.name
            if not self._is_level_allowed(level_name):
                continue
            if not self._passes_surprise(surprise_value):
                self._record_gate(level_name, hit=False)
                continue
            inputs = cms_inputs[level_name]
            seq_len = inputs.shape[1]
            chunk_size = int(spec.update_period)
            if chunk_size <= 0:
                continue
            total_norm = 0.0
            update_events = 0
            token_events = 0
            for start in range(0, seq_len, chunk_size):
                end = min(start + chunk_size, seq_len)
                chunk_len = end - start
                chunk_inputs = inputs[:, start:end, :].detach()
                chunk_teach = teach[:, start:end, :]
                chunk_active = active_mask[:, start:end]
                if not bool(chunk_active.any()):
                    continue
                magnitude = self._update_cms_chunk(
                    level_name,
                    chunk_inputs,
                    chunk_teach,
                    chunk_active,
                    surprise_value,
                )
                if magnitude <= 0:
                    continue
                total_norm += magnitude
                token_events += chunk_len
                update_events += 1
            if update_events == 0:
                continue
            stats_payload: Dict[str, float] = {
                "grad_norm": total_norm,
                "chunk_tokens": float(token_events),
                "gate_hit": float(update_events),
            }
            if surprise_value is not None:
                stats_payload["surprise_value"] = surprise_value
            self.last_update_stats[f"cms.{level_name}"] = stats_payload

    def _update_cms_chunk(
        self,
        level_name: str,
        chunk_inputs: torch.Tensor,
        chunk_teach: torch.Tensor,
        chunk_active: torch.Tensor,
        surprise_value: float | None,
    ) -> float:
        if not self._is_level_allowed(level_name):
            return 0.0
        if not self._passes_surprise(surprise_value):
            self._record_gate(level_name, hit=False)
            return 0.0
        mask_f = chunk_active.unsqueeze(-1).float()
        with torch.enable_grad():
            prediction = self.cms.blocks[level_name](chunk_inputs)
            loss = _chunk_loss(
                prediction,
                chunk_teach,
                mask_f,
                reduction=self.config.cms_chunk_reduction,
            )
        context_vec = chunk_inputs.mean(dim=(0, 1))
        magnitude = self.level_manager.optimize(
            level_name,
            self.cms.blocks[level_name],
            loss,
            context=context_vec,
            force=True,
        )
        self.level_manager.pop_last_metrics(level_name)
        return magnitude

    def _update_cms_chunk_fast(
        self,
        fast_state: BlockFastState,
        level_name: str,
        chunk_inputs: torch.Tensor,
        chunk_teach: torch.Tensor,
        chunk_active: torch.Tensor,
        surprise_value: float | None,
    ) -> float:
        if not self._is_level_allowed(level_name):
            return 0.0
        if not self._passes_surprise(surprise_value):
            self._record_gate(level_name, hit=False)
            return 0.0
        mask_f = chunk_active.unsqueeze(-1).float()
        base_params = fast_state.cms_params[level_name]
        if isinstance(base_params, list):
            updated_params: list[Dict[str, torch.Tensor]] = []
            total_magnitude = 0.0
            for sample_idx, sample_params in enumerate(base_params):
                sample_active = chunk_active[sample_idx : sample_idx + 1]
                if not bool(sample_active.any()):
                    updated_params.append(sample_params)
                    continue
                sample_inputs = chunk_inputs[sample_idx : sample_idx + 1]
                sample_teach = chunk_teach[sample_idx : sample_idx + 1]
                sample_mask = mask_f[sample_idx : sample_idx + 1]
                forward_params = params_with_deltas(self.cms.blocks[level_name], sample_params)
                params_req = require_grad_params(forward_params)
                with torch.enable_grad():
                    prediction = call_with_params(self.cms.blocks[level_name], params_req, sample_inputs)
                    loss = _chunk_loss(
                        prediction,
                        sample_teach,
                        sample_mask,
                        reduction=self.config.cms_chunk_reduction,
                    )
                grads = torch.autograd.grad(
                    loss,
                    tuple(params_req.values()),
                    retain_graph=False,
                    allow_unused=True,
                )
                grads_dict = grads_to_dict(params_req, grads)
                context_vec = sample_inputs.mean(dim=(0, 1))
                level_manager = _manager_for_sample(fast_state.level_manager, sample_idx)
                updated, magnitude = level_manager.apply_grads(
                    level_name,
                    sample_params,
                    grads_dict,
                    context=context_vec,
                    force=True,
                )
                level_manager.pop_last_metrics(level_name)
                updated_params.append(updated)
                total_magnitude += magnitude
            fast_state.cms_params[level_name] = updated_params
            return total_magnitude
        if _is_batched_delta_dict(self.cms.blocks[level_name], base_params):
            updated, magnitude = _apply_tensorized_cms_grads(
                module=self.cms.blocks[level_name],
                level_name=level_name,
                base_params=base_params,
                chunk_inputs=chunk_inputs,
                chunk_teach=chunk_teach,
                chunk_active=chunk_active,
                reduction=self.config.cms_chunk_reduction,
                level_manager=fast_state.level_manager,
            )
            fast_state.cms_params[level_name] = updated
            return magnitude

        forward_params = params_with_deltas(self.cms.blocks[level_name], base_params)
        params_req = require_grad_params(forward_params)
        with torch.enable_grad():
            prediction = call_with_params(self.cms.blocks[level_name], params_req, chunk_inputs)
            loss = _chunk_loss(
                prediction,
                chunk_teach,
                mask_f,
                reduction=self.config.cms_chunk_reduction,
            )
        grads = torch.autograd.grad(
            loss,
            tuple(params_req.values()),
            retain_graph=False,
            allow_unused=True,
        )
        grads_dict = grads_to_dict(params_req, grads)
        context_vec = chunk_inputs.mean(dim=(0, 1))
        level_manager = _manager_for_sample(fast_state.level_manager, 0)
        updated, magnitude = level_manager.apply_grads(
            level_name,
            base_params,
            grads_dict,
            context=context_vec,
            force=True,
        )
        fast_state.cms_params[level_name] = updated
        level_manager.pop_last_metrics(level_name)
        return magnitude


@dataclass
class HOPESelfModBlockConfig:
    dim: int
    cms_levels: Sequence[LevelSpec]
    cms_hidden_multiplier: int = 4
    cms_use_layernorm: bool = True
    activation: str = "gelu"
    qk_l2_norm: bool = True
    cms_flush_partial_at_end: bool = False
    selfmod_adaptive_q: bool = False
    selfmod_local_conv_window: int | None = 4
    eta_scale: float = 1e-3
    selfmod_chunk_size: int = 1
    selfmod_chunk_size_memory: int | None = None
    selfmod_objective: str = "l2"
    selfmod_stopgrad_vhat: bool = True
    selfmod_use_rank1_precond: bool = True
    selfmod_use_alpha: bool = True
    selfmod_use_skip: bool = True
    selfmod_momentum: float = 0.0
    selfmod_online_updates: bool = True
    self_mod_lr: float = 1e-3
    cms_chunk_reduction: str = "sum"
    cms_online_updates: bool = True
    optimizer_configs: Dict[str, dict] = field(default_factory=dict)


class HOPESelfModBlock(nn.Module):
    """
    Paper-defined HOPE block (Eqs. 94–97): self-modifying Titans followed by CMS.

    Fast-state is required for in-context self-mod updates.
    """

    def __init__(self, config: HOPESelfModBlockConfig):
        super().__init__()
        self.config = config
        self.last_update_stats: Dict[str, Dict[str, float]] = {}
        self.surprise_threshold: float | None = None
        self.surprise_metric: str = "l2"
        self.allowed_levels: Set[str] | None = None
        self.selfmod = SelfModifyingTitans(
            SelfModifyingTitansConfig(
                dim=config.dim,
                eta_scale=config.eta_scale,
                chunk_size_other=config.selfmod_chunk_size,
                chunk_size_memory=config.selfmod_chunk_size_memory,
                objective=config.selfmod_objective,
                stopgrad_vhat=config.selfmod_stopgrad_vhat,
                use_rank1_precond=config.selfmod_use_rank1_precond,
                use_alpha=config.selfmod_use_alpha,
                use_skip=config.selfmod_use_skip,
                momentum=config.selfmod_momentum,
                qk_l2_norm=config.qk_l2_norm,
                adaptive_q=config.selfmod_adaptive_q,
                local_conv_window=config.selfmod_local_conv_window,
            )
        )
        self.cms = CMS(
            dim=config.dim,
            levels=config.cms_levels,
            hidden_multiplier=config.cms_hidden_multiplier,
            activation=config.activation,
            use_layernorm=config.cms_use_layernorm,
        )
        level_config = LevelConfig(
            specs=config.cms_levels,
            optimizer_configs=config.optimizer_configs,
            default_lr=config.self_mod_lr,
        )
        self.level_manager = LevelOptimizerManager(level_config)

    def forward(
        self,
        x: torch.Tensor,
        *,
        teach_signal: torch.Tensor | None = None,
        surprise_value: float | None = None,
        fast_state: BlockFastState | None = None,
    ) -> torch.Tensor:
        if fast_state is None:
            # Differentiable read path (used for the outer loss).
            o = self.selfmod(x)
            # Explicit update pass (typically called under `torch.no_grad()` after backward).
            if teach_signal is not None and self.config.selfmod_online_updates:
                self.selfmod.apply_updates_inplace(x)
            if teach_signal is not None and self.config.cms_online_updates:
                cms_out = self._cms_forward_online(o, teach_signal, surprise_value)
            else:
                cms_out, cms_inputs, cms_outputs = self.cms(o, return_intermediates=True)
                if teach_signal is not None:
                    self._update_cms(cms_inputs, cms_outputs, teach_signal, surprise_value)
            self.level_manager.tick()
            return cms_out

        if fast_state.selfmod_state is None:
            raise ValueError("fast_state.selfmod_state is required for hope_selfmod variant")
        if isinstance(fast_state.selfmod_state, list):
            outputs: list[torch.Tensor] = []
            if self.config.selfmod_online_updates and teach_signal is not None:
                updated_states = []
                for sample_idx, sample_state in enumerate(fast_state.selfmod_state):
                    sample_out, sample_updated = self.selfmod.forward_with_updates(
                        x[sample_idx : sample_idx + 1],
                        sample_state,
                    )
                    outputs.append(sample_out)
                    updated_states.append(sample_updated)
                fast_state.selfmod_state = updated_states
            else:
                for sample_idx, sample_state in enumerate(fast_state.selfmod_state):
                    sample_out = self.selfmod.forward_with_state(
                        x[sample_idx : sample_idx + 1],
                        sample_state,
                    )
                    outputs.append(sample_out)
            o = torch.cat(outputs, dim=0)
        elif self.config.selfmod_online_updates and teach_signal is not None:
            o, updated = self.selfmod.forward_with_updates(x, fast_state.selfmod_state)
            fast_state.selfmod_state = updated
        else:
            o = self.selfmod.forward_with_state(x, fast_state.selfmod_state)
        if teach_signal is not None and self.config.cms_online_updates:
            cms_out = self._cms_forward_online_fast(o, fast_state, teach_signal, surprise_value)
        else:
            cms_out, cms_inputs = self._cms_forward_fast(o, fast_state)
            if teach_signal is not None:
                self._update_cms_fast(fast_state, cms_inputs, teach_signal, surprise_value)
        _tick_manager(fast_state.level_manager)
        return cms_out

    def set_surprise_threshold(self, threshold: float | None) -> None:
        self.surprise_threshold = threshold

    def set_surprise_metric(self, metric: str) -> None:
        self.surprise_metric = str(metric).strip().lower()

    def set_allowed_levels(self, allowed: Set[str] | None) -> None:
        self.allowed_levels = allowed.copy() if allowed is not None else None

    def pop_update_stats(self) -> Dict[str, Dict[str, float]]:
        stats = self.last_update_stats
        self.last_update_stats = {}
        return stats

    def _cms_forward_fast(
        self,
        x: torch.Tensor,
        fast_state: BlockFastState,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        current = x
        inputs: dict[str, torch.Tensor] = {}
        for spec in self.config.cms_levels:
            level_name = spec.name
            inputs[level_name] = current
            params = fast_state.cms_params[level_name]
            current = _call_with_deltas_maybe_list(self.cms.blocks[level_name], params, current)
        return current, inputs

    def _cms_forward_online(
        self,
        x: torch.Tensor,
        teach_signal: torch.Tensor,
        surprise_value: float | None,
    ) -> torch.Tensor:
        seq_len = x.shape[1]
        base_chunk = _min_update_period(self.config.cms_levels)
        active_mask = teach_signal.detach().abs().sum(dim=-1) > 0
        outputs: list[torch.Tensor] = []
        stats: dict[str, Dict[str, float]] = {}
        buffers: dict[str, _CmsBuffer] = {}
        for spec in self.config.cms_levels:
            buffers[spec.name] = _CmsBuffer(inputs=[], teach=[], active=[], count=0)
            stats[spec.name] = {"grad_norm": 0.0, "chunk_tokens": 0.0, "gate_hit": 0.0}

        for start in range(0, seq_len, base_chunk):
            end = min(start + base_chunk, seq_len)
            chunk_in = x[:, start:end, :]
            chunk_teach = teach_signal[:, start:end, :]
            chunk_active = active_mask[:, start:end]

            current = chunk_in
            level_inputs: dict[str, torch.Tensor] = {}
            for spec in self.config.cms_levels:
                level_name = spec.name
                level_inputs[level_name] = current
                current = self.cms.blocks[level_name](current)
            outputs.append(current)

            for spec in self.config.cms_levels:
                level_name = spec.name
                buffer = buffers[level_name]
                buffer.inputs.append(level_inputs[level_name].detach())
                buffer.teach.append(chunk_teach)
                buffer.active.append(chunk_active)
                buffer.count += end - start
                update_period = int(spec.update_period)
                while update_period > 0 and buffer.count >= update_period:
                    chunk_inputs, chunk_teach, chunk_active = _pop_buffer_chunk(
                        buffer, update_period
                    )
                    buffer.count -= update_period
                    magnitude = self._update_cms_chunk(
                        level_name,
                        chunk_inputs,
                        chunk_teach,
                        chunk_active,
                        surprise_value,
                    )
                    if magnitude > 0:
                        stats[level_name]["grad_norm"] += magnitude
                        stats[level_name]["chunk_tokens"] += float(update_period)
                        stats[level_name]["gate_hit"] += 1.0
        if self.config.cms_flush_partial_at_end:
            for spec in self.config.cms_levels:
                level_name = spec.name
                buffer = buffers[level_name]
                remaining = int(buffer.count)
                if remaining <= 0:
                    continue
                chunk_inputs, chunk_teach, chunk_active = _pop_buffer_chunk(buffer, remaining)
                buffer.count -= remaining
                if not bool(chunk_active.any()):
                    continue
                magnitude = self._update_cms_chunk(
                    level_name,
                    chunk_inputs,
                    chunk_teach,
                    chunk_active,
                    surprise_value,
                )
                if magnitude > 0:
                    stats[level_name]["grad_norm"] += magnitude
                    stats[level_name]["chunk_tokens"] += float(remaining)
                    stats[level_name]["gate_hit"] += 1.0
        for level_name, payload in stats.items():
            if payload["gate_hit"] <= 0:
                continue
            if surprise_value is not None:
                payload["surprise_value"] = surprise_value
            self.last_update_stats[f"cms.{level_name}"] = payload
        return torch.cat(outputs, dim=1)

    def _cms_forward_online_fast(
        self,
        x: torch.Tensor,
        fast_state: BlockFastState,
        teach_signal: torch.Tensor,
        surprise_value: float | None,
    ) -> torch.Tensor:
        seq_len = x.shape[1]
        base_chunk = _min_update_period(self.config.cms_levels)
        active_mask = teach_signal.detach().abs().sum(dim=-1) > 0
        outputs: list[torch.Tensor] = []
        stats: dict[str, Dict[str, float]] = {}
        buffers: dict[str, _CmsBuffer] = {}
        for spec in self.config.cms_levels:
            buffers[spec.name] = _CmsBuffer(inputs=[], teach=[], active=[], count=0)
            stats[spec.name] = {"grad_norm": 0.0, "chunk_tokens": 0.0, "gate_hit": 0.0}

        for start in range(0, seq_len, base_chunk):
            end = min(start + base_chunk, seq_len)
            chunk_in = x[:, start:end, :]
            chunk_teach = teach_signal[:, start:end, :]
            chunk_active = active_mask[:, start:end]

            current = chunk_in
            level_inputs: dict[str, torch.Tensor] = {}
            for spec in self.config.cms_levels:
                level_name = spec.name
                level_inputs[level_name] = current
                params = fast_state.cms_params[level_name]
                current = _call_with_deltas_maybe_list(
                    self.cms.blocks[level_name], params, current
                )
            outputs.append(current)

            for spec in self.config.cms_levels:
                level_name = spec.name
                buffer = buffers[level_name]
                buffer.inputs.append(level_inputs[level_name].detach())
                buffer.teach.append(chunk_teach)
                buffer.active.append(chunk_active)
                buffer.count += end - start
                update_period = int(spec.update_period)
                while update_period > 0 and buffer.count >= update_period:
                    chunk_inputs, chunk_teach, chunk_active = _pop_buffer_chunk(
                        buffer, update_period
                    )
                    buffer.count -= update_period
                    magnitude = self._update_cms_chunk_fast(
                        fast_state,
                        level_name,
                        chunk_inputs,
                        chunk_teach,
                        chunk_active,
                        surprise_value,
                    )
                    if magnitude > 0:
                        stats[level_name]["grad_norm"] += magnitude
                        stats[level_name]["chunk_tokens"] += float(update_period)
                        stats[level_name]["gate_hit"] += 1.0
        if self.config.cms_flush_partial_at_end:
            for spec in self.config.cms_levels:
                level_name = spec.name
                buffer = buffers[level_name]
                remaining = int(buffer.count)
                if remaining <= 0:
                    continue
                chunk_inputs, chunk_teach, chunk_active = _pop_buffer_chunk(buffer, remaining)
                buffer.count -= remaining
                if not bool(chunk_active.any()):
                    continue
                magnitude = self._update_cms_chunk_fast(
                    fast_state,
                    level_name,
                    chunk_inputs,
                    chunk_teach,
                    chunk_active,
                    surprise_value,
                )
                if magnitude > 0:
                    stats[level_name]["grad_norm"] += magnitude
                    stats[level_name]["chunk_tokens"] += float(remaining)
                    stats[level_name]["gate_hit"] += 1.0
        for level_name, payload in stats.items():
            if payload["gate_hit"] <= 0:
                continue
            if surprise_value is not None:
                payload["surprise_value"] = surprise_value
            self.last_update_stats[f"cms.{level_name}"] = payload
        return torch.cat(outputs, dim=1)

    def _is_level_allowed(self, level_name: str) -> bool:
        if self.allowed_levels is None:
            return True
        return level_name in self.allowed_levels

    def _passes_surprise(self, surprise_value: float | None) -> bool:
        if self.surprise_threshold is None:
            return True
        if surprise_value is None:
            return False
        return surprise_value >= self.surprise_threshold

    def _record_gate(self, level_name: str, *, hit: bool) -> None:
        stats_key = f"gate.{level_name}"
        self.last_update_stats.setdefault(stats_key, {})
        self.last_update_stats[stats_key]["gate_hit"] = 1.0 if hit else 0.0

    def _update_cms(
        self,
        cms_inputs: dict[str, torch.Tensor],
        cms_outputs: dict[str, torch.Tensor],
        teach_signal: torch.Tensor,
        surprise_value: float | None,
    ) -> None:
        teach = teach_signal.detach()
        active_mask = teach.abs().sum(dim=-1) > 0
        for spec in self.config.cms_levels:
            level_name = spec.name
            if not self._is_level_allowed(level_name):
                continue
            if not self._passes_surprise(surprise_value):
                self._record_gate(level_name, hit=False)
                continue
            inputs = cms_inputs[level_name]
            seq_len = inputs.shape[1]
            chunk_size = int(spec.update_period)
            if chunk_size <= 0:
                continue
            total_norm = 0.0
            update_events = 0
            token_events = 0
            for start in range(0, seq_len, chunk_size):
                end = min(start + chunk_size, seq_len)
                chunk_len = end - start
                chunk_inputs = inputs[:, start:end, :].detach()
                chunk_teach = teach[:, start:end, :]
                chunk_active = active_mask[:, start:end]
                if not bool(chunk_active.any()):
                    continue
                magnitude = self._update_cms_chunk(
                    level_name,
                    chunk_inputs,
                    chunk_teach,
                    chunk_active,
                    surprise_value,
                )
                if magnitude <= 0:
                    continue
                total_norm += magnitude
                token_events += chunk_len
                update_events += 1
            if update_events == 0:
                continue
            stats_payload: Dict[str, float] = {
                "grad_norm": total_norm,
                "chunk_tokens": float(token_events),
                "gate_hit": float(update_events),
            }
            if surprise_value is not None:
                stats_payload["surprise_value"] = surprise_value
            self.last_update_stats[f"cms.{level_name}"] = stats_payload

    def _update_cms_fast(
        self,
        fast_state: BlockFastState,
        cms_inputs: dict[str, torch.Tensor],
        teach_signal: torch.Tensor,
        surprise_value: float | None,
    ) -> None:
        teach = teach_signal.detach()
        active_mask = teach.abs().sum(dim=-1) > 0
        for spec in self.config.cms_levels:
            level_name = spec.name
            if not self._is_level_allowed(level_name):
                continue
            if not self._passes_surprise(surprise_value):
                self._record_gate(level_name, hit=False)
                continue
            inputs = cms_inputs[level_name]
            seq_len = inputs.shape[1]
            chunk_size = int(spec.update_period)
            if chunk_size <= 0:
                continue
            total_norm = 0.0
            update_events = 0
            token_events = 0
            for start in range(0, seq_len, chunk_size):
                end = min(start + chunk_size, seq_len)
                chunk_len = end - start
                chunk_inputs = inputs[:, start:end, :].detach()
                chunk_teach = teach[:, start:end, :]
                chunk_active = active_mask[:, start:end]
                if not bool(chunk_active.any()):
                    continue
                magnitude = self._update_cms_chunk_fast(
                    fast_state,
                    level_name,
                    chunk_inputs,
                    chunk_teach,
                    chunk_active,
                    surprise_value,
                )
                if magnitude <= 0:
                    continue
                total_norm += magnitude
                token_events += chunk_len
                update_events += 1
            if update_events == 0:
                continue
            stats_payload: Dict[str, float] = {
                "grad_norm": total_norm,
                "chunk_tokens": float(token_events),
                "gate_hit": float(update_events),
            }
            if surprise_value is not None:
                stats_payload["surprise_value"] = surprise_value
            self.last_update_stats[f"cms.{level_name}"] = stats_payload

    def _update_cms_chunk(
        self,
        level_name: str,
        chunk_inputs: torch.Tensor,
        chunk_teach: torch.Tensor,
        chunk_active: torch.Tensor,
        surprise_value: float | None,
    ) -> float:
        if not self._is_level_allowed(level_name):
            return 0.0
        if not self._passes_surprise(surprise_value):
            self._record_gate(level_name, hit=False)
            return 0.0
        mask_f = chunk_active.unsqueeze(-1).float()
        with torch.enable_grad():
            prediction = self.cms.blocks[level_name](chunk_inputs)
            loss = _chunk_loss(
                prediction,
                chunk_teach,
                mask_f,
                reduction=self.config.cms_chunk_reduction,
            )
        context_vec = chunk_inputs.mean(dim=(0, 1))
        magnitude = self.level_manager.optimize(
            level_name,
            self.cms.blocks[level_name],
            loss,
            context=context_vec,
            force=True,
        )
        self.level_manager.pop_last_metrics(level_name)
        return magnitude

    def _update_cms_chunk_fast(
        self,
        fast_state: BlockFastState,
        level_name: str,
        chunk_inputs: torch.Tensor,
        chunk_teach: torch.Tensor,
        chunk_active: torch.Tensor,
        surprise_value: float | None,
    ) -> float:
        if not self._is_level_allowed(level_name):
            return 0.0
        if not self._passes_surprise(surprise_value):
            self._record_gate(level_name, hit=False)
            return 0.0
        mask_f = chunk_active.unsqueeze(-1).float()
        base_params = fast_state.cms_params[level_name]
        if isinstance(base_params, list):
            updated_params: list[Dict[str, torch.Tensor]] = []
            total_magnitude = 0.0
            for sample_idx, sample_params in enumerate(base_params):
                sample_active = chunk_active[sample_idx : sample_idx + 1]
                if not bool(sample_active.any()):
                    updated_params.append(sample_params)
                    continue
                sample_inputs = chunk_inputs[sample_idx : sample_idx + 1]
                sample_teach = chunk_teach[sample_idx : sample_idx + 1]
                sample_mask = mask_f[sample_idx : sample_idx + 1]
                forward_params = params_with_deltas(self.cms.blocks[level_name], sample_params)
                params_req = require_grad_params(forward_params)
                with torch.enable_grad():
                    prediction = call_with_params(self.cms.blocks[level_name], params_req, sample_inputs)
                    loss = _chunk_loss(
                        prediction,
                        sample_teach,
                        sample_mask,
                        reduction=self.config.cms_chunk_reduction,
                    )
                grads = torch.autograd.grad(
                    loss,
                    tuple(params_req.values()),
                    retain_graph=False,
                    allow_unused=True,
                )
                grads_dict = grads_to_dict(params_req, grads)
                context_vec = sample_inputs.mean(dim=(0, 1))
                level_manager = _manager_for_sample(fast_state.level_manager, sample_idx)
                updated, magnitude = level_manager.apply_grads(
                    level_name,
                    sample_params,
                    grads_dict,
                    context=context_vec,
                    force=True,
                )
                level_manager.pop_last_metrics(level_name)
                updated_params.append(updated)
                total_magnitude += magnitude
            fast_state.cms_params[level_name] = updated_params
            return total_magnitude

        if _is_batched_delta_dict(self.cms.blocks[level_name], base_params):
            updated, magnitude = _apply_tensorized_cms_grads(
                module=self.cms.blocks[level_name],
                level_name=level_name,
                base_params=base_params,
                chunk_inputs=chunk_inputs,
                chunk_teach=chunk_teach,
                chunk_active=chunk_active,
                reduction=self.config.cms_chunk_reduction,
                level_manager=fast_state.level_manager,
            )
            fast_state.cms_params[level_name] = updated
            return magnitude

        forward_params = params_with_deltas(self.cms.blocks[level_name], base_params)
        params_req = require_grad_params(forward_params)
        with torch.enable_grad():
            prediction = call_with_params(self.cms.blocks[level_name], params_req, chunk_inputs)
            loss = _chunk_loss(
                prediction,
                chunk_teach,
                mask_f,
                reduction=self.config.cms_chunk_reduction,
            )
        grads = torch.autograd.grad(
            loss,
            tuple(params_req.values()),
            retain_graph=False,
            allow_unused=True,
        )
        grads_dict = grads_to_dict(params_req, grads)
        context_vec = chunk_inputs.mean(dim=(0, 1))
        level_manager = _manager_for_sample(fast_state.level_manager, 0)
        updated, magnitude = level_manager.apply_grads(
            level_name,
            base_params,
            grads_dict,
            context=context_vec,
            force=True,
        )
        fast_state.cms_params[level_name] = updated
        level_manager.pop_last_metrics(level_name)
        return magnitude


class HOPEBlock(nn.Module):
    def __init__(self, config: HOPEBlockConfig):
        super().__init__()
        self.config = config
        self.last_update_stats: Dict[str, Dict[str, float]] = {}
        self.surprise_threshold: float | None = None
        self.surprise_metric: str = "l2"
        self.allowed_levels: Set[str] | None = None
        self.attn = SelfAttention(
            AttentionConfig(
                dim=config.dim,
                heads=config.heads,
                qk_l2_norm=config.qk_l2_norm,
                local_conv_window=config.local_conv_window,
            )
        )
        titan_config = TitanMemoryConfig(
            dim=config.dim,
            hidden_multiplier=config.titan_hidden_multiplier,
            activation=config.activation,
        )
        self.titan_memory = TitanMemory(titan_config)
        self.cms = CMS(
            dim=config.dim,
            levels=config.cms_levels,
            hidden_multiplier=config.cms_hidden_multiplier,
            activation=config.activation,
            use_layernorm=config.cms_use_layernorm,
        )
        self.self_modifier = SelfModifier(config.dim, hidden_multiplier=config.self_mod_hidden)
        self.dropout = nn.Dropout(0.0)
        specs = [config.titan_level, *config.cms_levels]
        level_config = LevelConfig(
            specs=specs,
            optimizer_configs=config.optimizer_configs,
            default_lr=config.self_mod_lr,
        )
        self.level_manager = LevelOptimizerManager(level_config)

    def forward(
        self,
        x: torch.Tensor,
        *,
        teach_signal: torch.Tensor | None = None,
        surprise_value: float | None = None,
        fast_state: BlockFastState | None = None,
    ) -> torch.Tensor:
        attn_out = self.attn(x)
        if fast_state is None:
            mem_out = self.titan_memory(attn_out)
            combined = attn_out + mem_out
            if teach_signal is not None and self.config.cms_online_updates:
                cms_out = self._cms_forward_online(combined, teach_signal, surprise_value)
                self._update_titan(attn_out, mem_out, teach_signal, surprise_value)
            else:
                cms_result = self.cms(combined, return_intermediates=True)
                cms_out, cms_inputs, cms_outputs = cms_result
                if teach_signal is not None:
                    self._update_titan(attn_out, mem_out, teach_signal, surprise_value)
                    self._update_cms(cms_inputs, cms_outputs, teach_signal, surprise_value)
            self.level_manager.tick()
            return cms_out

        if fast_state.titan_params is None:
            raise ValueError("fast_state.titan_params is required for HOPEBlock fast-state forward")
        mem_out = _call_with_deltas_maybe_list(self.titan_memory, fast_state.titan_params, attn_out)
        combined = attn_out + mem_out
        if teach_signal is not None and self.config.cms_online_updates:
            cms_out = self._cms_forward_online_fast(
                combined, fast_state, teach_signal, surprise_value
            )
            self._update_titan_fast(fast_state, attn_out, mem_out, teach_signal, surprise_value)
        else:
            cms_out, cms_inputs = self._cms_forward_fast(combined, fast_state)
            if teach_signal is not None:
                self._update_titan_fast(fast_state, attn_out, mem_out, teach_signal, surprise_value)
                self._update_cms_fast(fast_state, cms_inputs, teach_signal, surprise_value)
        _tick_manager(fast_state.level_manager)
        return cms_out

    def set_surprise_threshold(self, threshold: float | None) -> None:
        self.surprise_threshold = threshold

    def set_surprise_metric(self, metric: str) -> None:
        self.surprise_metric = str(metric).strip().lower()

    def set_allowed_levels(self, allowed: Set[str] | None) -> None:
        self.allowed_levels = allowed.copy() if allowed is not None else None

    def _cms_forward_fast(
        self,
        x: torch.Tensor,
        fast_state: BlockFastState,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        current = x
        inputs: dict[str, torch.Tensor] = {}
        for spec in self.config.cms_levels:
            level_name = spec.name
            inputs[level_name] = current
            params = fast_state.cms_params[level_name]
            current = _call_with_deltas_maybe_list(self.cms.blocks[level_name], params, current)
        return current, inputs


    def _cms_forward_online(
        self,
        x: torch.Tensor,
        teach_signal: torch.Tensor,
        surprise_value: float | None,
    ) -> torch.Tensor:
        seq_len = x.shape[1]
        base_chunk = _min_update_period(self.config.cms_levels)
        active_mask = teach_signal.detach().abs().sum(dim=-1) > 0
        outputs: list[torch.Tensor] = []
        stats: dict[str, Dict[str, float]] = {}
        buffers: dict[str, _CmsBuffer] = {}
        for spec in self.config.cms_levels:
            buffers[spec.name] = _CmsBuffer(inputs=[], teach=[], active=[], count=0)
            stats[spec.name] = {"grad_norm": 0.0, "chunk_tokens": 0.0, "gate_hit": 0.0}

        for start in range(0, seq_len, base_chunk):
            end = min(start + base_chunk, seq_len)
            chunk_in = x[:, start:end, :]
            chunk_teach = teach_signal[:, start:end, :]
            chunk_active = active_mask[:, start:end]

            current = chunk_in
            level_inputs: dict[str, torch.Tensor] = {}
            for spec in self.config.cms_levels:
                level_name = spec.name
                level_inputs[level_name] = current
                current = self.cms.blocks[level_name](current)
            outputs.append(current)

            for spec in self.config.cms_levels:
                level_name = spec.name
                buffer = buffers[level_name]
                buffer.inputs.append(level_inputs[level_name].detach())
                buffer.teach.append(chunk_teach)
                buffer.active.append(chunk_active)
                buffer.count += end - start
                update_period = int(spec.update_period)
                while update_period > 0 and buffer.count >= update_period:
                    chunk_inputs, chunk_teach, chunk_active = _pop_buffer_chunk(
                        buffer, update_period
                    )
                    buffer.count -= update_period
                    magnitude = self._update_cms_chunk(
                        level_name,
                        chunk_inputs,
                        chunk_teach,
                        chunk_active,
                        surprise_value,
                    )
                    if magnitude > 0:
                        stats[level_name]["grad_norm"] += magnitude
                        stats[level_name]["chunk_tokens"] += float(update_period)
                        stats[level_name]["gate_hit"] += 1.0
        if self.config.cms_flush_partial_at_end:
            for spec in self.config.cms_levels:
                level_name = spec.name
                buffer = buffers[level_name]
                remaining = int(buffer.count)
                if remaining <= 0:
                    continue
                chunk_inputs, chunk_teach, chunk_active = _pop_buffer_chunk(buffer, remaining)
                buffer.count -= remaining
                if not bool(chunk_active.any()):
                    continue
                magnitude = self._update_cms_chunk(
                    level_name,
                    chunk_inputs,
                    chunk_teach,
                    chunk_active,
                    surprise_value,
                )
                if magnitude > 0:
                    stats[level_name]["grad_norm"] += magnitude
                    stats[level_name]["chunk_tokens"] += float(remaining)
                    stats[level_name]["gate_hit"] += 1.0
        for level_name, payload in stats.items():
            if payload["gate_hit"] <= 0:
                continue
            if surprise_value is not None:
                payload["surprise_value"] = surprise_value
            self.last_update_stats[f"cms.{level_name}"] = payload
        return torch.cat(outputs, dim=1)

    def _cms_forward_online_fast(
        self,
        x: torch.Tensor,
        fast_state: BlockFastState,
        teach_signal: torch.Tensor,
        surprise_value: float | None,
    ) -> torch.Tensor:
        seq_len = x.shape[1]
        base_chunk = _min_update_period(self.config.cms_levels)
        active_mask = teach_signal.detach().abs().sum(dim=-1) > 0
        outputs: list[torch.Tensor] = []
        stats: dict[str, Dict[str, float]] = {}
        buffers: dict[str, _CmsBuffer] = {}
        for spec in self.config.cms_levels:
            buffers[spec.name] = _CmsBuffer(inputs=[], teach=[], active=[], count=0)
            stats[spec.name] = {"grad_norm": 0.0, "chunk_tokens": 0.0, "gate_hit": 0.0}

        for start in range(0, seq_len, base_chunk):
            end = min(start + base_chunk, seq_len)
            chunk_in = x[:, start:end, :]
            chunk_teach = teach_signal[:, start:end, :]
            chunk_active = active_mask[:, start:end]

            current = chunk_in
            level_inputs: dict[str, torch.Tensor] = {}
            for spec in self.config.cms_levels:
                level_name = spec.name
                level_inputs[level_name] = current
                params = fast_state.cms_params[level_name]
                current = _call_with_deltas_maybe_list(
                    self.cms.blocks[level_name], params, current
                )
            outputs.append(current)

            for spec in self.config.cms_levels:
                level_name = spec.name
                buffer = buffers[level_name]
                buffer.inputs.append(level_inputs[level_name].detach())
                buffer.teach.append(chunk_teach)
                buffer.active.append(chunk_active)
                buffer.count += end - start
                update_period = int(spec.update_period)
                while update_period > 0 and buffer.count >= update_period:
                    chunk_inputs, chunk_teach, chunk_active = _pop_buffer_chunk(
                        buffer, update_period
                    )
                    buffer.count -= update_period
                    magnitude = self._update_cms_chunk_fast(
                        fast_state,
                        level_name,
                        chunk_inputs,
                        chunk_teach,
                        chunk_active,
                        surprise_value,
                    )
                    if magnitude > 0:
                        stats[level_name]["grad_norm"] += magnitude
                        stats[level_name]["chunk_tokens"] += float(update_period)
                        stats[level_name]["gate_hit"] += 1.0
        if self.config.cms_flush_partial_at_end:
            for spec in self.config.cms_levels:
                level_name = spec.name
                buffer = buffers[level_name]
                remaining = int(buffer.count)
                if remaining <= 0:
                    continue
                chunk_inputs, chunk_teach, chunk_active = _pop_buffer_chunk(buffer, remaining)
                buffer.count -= remaining
                if not bool(chunk_active.any()):
                    continue
                magnitude = self._update_cms_chunk_fast(
                    fast_state,
                    level_name,
                    chunk_inputs,
                    chunk_teach,
                    chunk_active,
                    surprise_value,
                )
                if magnitude > 0:
                    stats[level_name]["grad_norm"] += magnitude
                    stats[level_name]["chunk_tokens"] += float(remaining)
                    stats[level_name]["gate_hit"] += 1.0
        for level_name, payload in stats.items():
            if payload["gate_hit"] <= 0:
                continue
            if surprise_value is not None:
                payload["surprise_value"] = surprise_value
            self.last_update_stats[f"cms.{level_name}"] = payload
        return torch.cat(outputs, dim=1)
    def _update_titan(
        self,
        attn_out: torch.Tensor,
        mem_out: torch.Tensor,
        teach_signal: torch.Tensor,
        surprise_value: float | None,
    ) -> None:
        level_name = self.config.titan_level.name
        if not self._is_level_allowed("titan"):
            return
        if not self.level_manager.should_update(level_name):
            return
        if not self._passes_surprise(surprise_value):
            self._record_gate(level_name, hit=False)
            return
        # Use full sequence for granular updates (Critique P1)
        # Note: We intentionally do not pool over dim=1 (sequence) here.
        # teach_signal is (B, T, D), attn_out is (B, T, D)
        modifier = self.self_modifier(
            key=attn_out.detach(),
            value=mem_out.detach(),
            error_signal=teach_signal.detach(),
        )
        context_vec = attn_out.detach().mean(dim=(0, 1))

        with torch.enable_grad():
            query = attn_out.detach()
            target = (modifier - teach_signal.detach()).detach()
            base_params = {name: param for name, param in self.titan_memory.named_parameters()}
            params_req = require_grad_params(base_params)
            prediction = call_with_params(self.titan_memory, params_req, query)
            loss_terms = F.mse_loss(prediction, target, reduction="none")
            active = teach_signal.detach().abs().sum(dim=-1, keepdim=True) > 0
            mask = active.float()
            if self.surprise_threshold is not None and self.surprise_metric == "l2":
                norms = teach_signal.norm(dim=-1, keepdim=True)
                mask = mask * (norms >= self.surprise_threshold).float()
            loss = (loss_terms * mask).sum() / mask.sum().clamp(min=1.0)

        grads = torch.autograd.grad(
            loss,
            tuple(params_req.values()),
            retain_graph=False,
            allow_unused=True,
        )
        grads_dict = grads_to_dict(params_req, grads)
        magnitude = self.level_manager.apply_module_grads(
            level_name,
            self.titan_memory,
            grads_dict,
            context=context_vec,
            force=True,
        )
        extra_metrics = self.level_manager.pop_last_metrics(level_name)
        stats = {"grad_norm": magnitude, "gate_hit": 1.0}
        if surprise_value is not None:
            stats["surprise_value"] = surprise_value
        stats.update(extra_metrics)
        self.last_update_stats[f"titan.{level_name}"] = stats

    def _update_titan_fast(
        self,
        fast_state: BlockFastState,
        attn_out: torch.Tensor,
        mem_out: torch.Tensor,
        teach_signal: torch.Tensor,
        surprise_value: float | None,
    ) -> None:
        level_name = self.config.titan_level.name
        if not self._is_level_allowed("titan"):
            return
        if not self._passes_surprise(surprise_value):
            self._record_gate(level_name, hit=False)
            return
        if fast_state.titan_params is None:
            return
        base_params = fast_state.titan_params
        if isinstance(base_params, list):
            updated_params: list[Dict[str, torch.Tensor]] = []
            total_magnitude = 0.0
            total_hits = 0.0
            for sample_idx, sample_params in enumerate(base_params):
                level_manager = _manager_for_sample(fast_state.level_manager, sample_idx)
                if not level_manager.should_update(level_name):
                    updated_params.append(sample_params)
                    continue
                sample_attn = attn_out[sample_idx : sample_idx + 1].detach()
                sample_mem = mem_out[sample_idx : sample_idx + 1].detach()
                sample_teach = teach_signal[sample_idx : sample_idx + 1].detach()
                modifier = self.self_modifier(
                    key=sample_attn,
                    value=sample_mem,
                    error_signal=sample_teach,
                )
                context_vec = sample_attn.mean(dim=(0, 1))
                forward_params = params_with_deltas(self.titan_memory, sample_params)
                params_req = require_grad_params(forward_params)
                with torch.enable_grad():
                    target = (modifier - sample_teach).detach()
                    prediction = call_with_params(self.titan_memory, params_req, sample_attn)
                    loss_terms = F.mse_loss(prediction, target, reduction="none")
                    active = sample_teach.abs().sum(dim=-1, keepdim=True) > 0
                    mask = active.float()
                    if self.surprise_threshold is not None and self.surprise_metric == "l2":
                        norms = sample_teach.norm(dim=-1, keepdim=True)
                        mask = mask * (norms >= self.surprise_threshold).float()
                    loss = (loss_terms * mask).sum() / mask.sum().clamp(min=1.0)
                grads = torch.autograd.grad(
                    loss,
                    tuple(params_req.values()),
                    retain_graph=False,
                    allow_unused=True,
                )
                grads_dict = grads_to_dict(params_req, grads)
                updated, magnitude = level_manager.apply_grads(
                    level_name,
                    sample_params,
                    grads_dict,
                    context=context_vec,
                    force=False,
                )
                updated_params.append(updated)
                level_manager.pop_last_metrics(level_name)
                total_magnitude += magnitude
                total_hits += 1.0
            fast_state.titan_params = updated_params
            if total_hits <= 0:
                return
            stats = {"grad_norm": total_magnitude, "gate_hit": total_hits}
            if surprise_value is not None:
                stats["surprise_value"] = surprise_value
            self.last_update_stats[f"titan.{level_name}"] = stats
            return

        level_manager = _manager_for_sample(fast_state.level_manager, 0)
        if not level_manager.should_update(level_name):
            return
        modifier = self.self_modifier(
            key=attn_out.detach(),
            value=mem_out.detach(),
            error_signal=teach_signal.detach(),
        )
        context_vec = attn_out.detach().mean(dim=(0, 1))
        forward_params = params_with_deltas(self.titan_memory, base_params)
        params_req = require_grad_params(forward_params)
        with torch.enable_grad():
            query = attn_out.detach()
            target = (modifier - teach_signal.detach()).detach()
            prediction = call_with_params(self.titan_memory, params_req, query)
            loss_terms = F.mse_loss(prediction, target, reduction="none")
            active = teach_signal.detach().abs().sum(dim=-1, keepdim=True) > 0
            mask = active.float()
            if self.surprise_threshold is not None and self.surprise_metric == "l2":
                norms = teach_signal.norm(dim=-1, keepdim=True)
                mask = mask * (norms >= self.surprise_threshold).float()
            loss = (loss_terms * mask).sum() / mask.sum().clamp(min=1.0)
        grads = torch.autograd.grad(
            loss,
            tuple(params_req.values()),
            retain_graph=False,
            allow_unused=True,
        )
        grads_dict = grads_to_dict(params_req, grads)
        updated, magnitude = level_manager.apply_grads(
            level_name,
            base_params,
            grads_dict,
            context=context_vec,
            force=False,
        )
        fast_state.titan_params = updated
        extra_metrics = level_manager.pop_last_metrics(level_name)
        stats = {"grad_norm": magnitude, "gate_hit": 1.0}
        if surprise_value is not None:
            stats["surprise_value"] = surprise_value
        stats.update(extra_metrics)
        self.last_update_stats[f"titan.{level_name}"] = stats

    def _update_cms(
        self,
        cms_inputs: dict[str, torch.Tensor],
        cms_outputs: dict[str, torch.Tensor],
        teach_signal: torch.Tensor,
        surprise_value: float | None,
    ) -> None:
        teach = teach_signal.detach()
        active_mask = teach.abs().sum(dim=-1) > 0
        for spec in self.config.cms_levels:
            level_name = spec.name
            if not self._is_level_allowed(level_name):
                continue
            if not self._passes_surprise(surprise_value):
                self._record_gate(level_name, hit=False)
                continue
            inputs = cms_inputs[level_name]
            seq_len = inputs.shape[1]
            chunk_size = int(spec.update_period)
            if chunk_size <= 0:
                continue
            total_norm = 0.0
            update_events = 0
            token_events = 0
            for start in range(0, seq_len, chunk_size):
                end = min(start + chunk_size, seq_len)
                chunk_len = end - start
                chunk_inputs = inputs[:, start:end, :].detach()
                chunk_teach = teach[:, start:end, :]
                chunk_active = active_mask[:, start:end]
                if not bool(chunk_active.any()):
                    continue
                magnitude = self._update_cms_chunk(
                    level_name,
                    chunk_inputs,
                    chunk_teach,
                    chunk_active,
                    surprise_value,
                )
                if magnitude <= 0:
                    continue
                total_norm += magnitude
                token_events += chunk_len
                update_events += 1
            if update_events == 0:
                continue
            stats_payload: Dict[str, float] = {
                "grad_norm": total_norm,
                "chunk_tokens": float(token_events),
                "gate_hit": float(update_events),
            }
            if surprise_value is not None:
                stats_payload["surprise_value"] = surprise_value
            self.last_update_stats[f"cms.{level_name}"] = stats_payload

    def _update_cms_fast(
        self,
        fast_state: BlockFastState,
        cms_inputs: dict[str, torch.Tensor],
        teach_signal: torch.Tensor,
        surprise_value: float | None,
    ) -> None:
        teach = teach_signal.detach()
        active_mask = teach.abs().sum(dim=-1) > 0
        for spec in self.config.cms_levels:
            level_name = spec.name
            if not self._is_level_allowed(level_name):
                continue
            if not self._passes_surprise(surprise_value):
                self._record_gate(level_name, hit=False)
                continue
            inputs = cms_inputs[level_name]
            seq_len = inputs.shape[1]
            chunk_size = int(spec.update_period)
            if chunk_size <= 0:
                continue
            total_norm = 0.0
            update_events = 0
            token_events = 0
            for start in range(0, seq_len, chunk_size):
                end = min(start + chunk_size, seq_len)
                chunk_len = end - start
                chunk_inputs = inputs[:, start:end, :].detach()
                chunk_teach = teach[:, start:end, :]
                chunk_active = active_mask[:, start:end]
                if not bool(chunk_active.any()):
                    continue
                magnitude = self._update_cms_chunk_fast(
                    fast_state,
                    level_name,
                    chunk_inputs,
                    chunk_teach,
                    chunk_active,
                    surprise_value,
                )
                if magnitude <= 0:
                    continue
                total_norm += magnitude
                token_events += chunk_len
                update_events += 1
            if update_events == 0:
                continue
            stats_payload: Dict[str, float] = {
                "grad_norm": total_norm,
                "chunk_tokens": float(token_events),
                "gate_hit": float(update_events),
            }
            if surprise_value is not None:
                stats_payload["surprise_value"] = surprise_value
            self.last_update_stats[f"cms.{level_name}"] = stats_payload

    def _update_cms_chunk(
        self,
        level_name: str,
        chunk_inputs: torch.Tensor,
        chunk_teach: torch.Tensor,
        chunk_active: torch.Tensor,
        surprise_value: float | None,
    ) -> float:
        if not self._is_level_allowed(level_name):
            return 0.0
        if not self._passes_surprise(surprise_value):
            self._record_gate(level_name, hit=False)
            return 0.0
        mask_f = chunk_active.unsqueeze(-1).float()
        with torch.enable_grad():
            prediction = self.cms.blocks[level_name](chunk_inputs)
            loss = _chunk_loss(
                prediction,
                chunk_teach,
                mask_f,
                reduction=self.config.cms_chunk_reduction,
            )
        context_vec = chunk_inputs.mean(dim=(0, 1))
        magnitude = self.level_manager.optimize(
            level_name,
            self.cms.blocks[level_name],
            loss,
            context=context_vec,
            force=True,
        )
        self.level_manager.pop_last_metrics(level_name)
        return magnitude

    def _update_cms_chunk_fast(
        self,
        fast_state: BlockFastState,
        level_name: str,
        chunk_inputs: torch.Tensor,
        chunk_teach: torch.Tensor,
        chunk_active: torch.Tensor,
        surprise_value: float | None,
    ) -> float:
        if not self._is_level_allowed(level_name):
            return 0.0
        if not self._passes_surprise(surprise_value):
            self._record_gate(level_name, hit=False)
            return 0.0
        mask_f = chunk_active.unsqueeze(-1).float()
        base_params = fast_state.cms_params[level_name]
        if isinstance(base_params, list):
            updated_params: list[Dict[str, torch.Tensor]] = []
            total_magnitude = 0.0
            for sample_idx, sample_params in enumerate(base_params):
                sample_active = chunk_active[sample_idx : sample_idx + 1]
                if not bool(sample_active.any()):
                    updated_params.append(sample_params)
                    continue
                sample_inputs = chunk_inputs[sample_idx : sample_idx + 1]
                sample_teach = chunk_teach[sample_idx : sample_idx + 1]
                sample_mask = mask_f[sample_idx : sample_idx + 1]
                forward_params = params_with_deltas(self.cms.blocks[level_name], sample_params)
                params_req = require_grad_params(forward_params)
                with torch.enable_grad():
                    prediction = call_with_params(self.cms.blocks[level_name], params_req, sample_inputs)
                    loss = _chunk_loss(
                        prediction,
                        sample_teach,
                        sample_mask,
                        reduction=self.config.cms_chunk_reduction,
                    )
                grads = torch.autograd.grad(
                    loss,
                    tuple(params_req.values()),
                    retain_graph=False,
                    allow_unused=True,
                )
                grads_dict = grads_to_dict(params_req, grads)
                context_vec = sample_inputs.mean(dim=(0, 1))
                level_manager = _manager_for_sample(fast_state.level_manager, sample_idx)
                updated, magnitude = level_manager.apply_grads(
                    level_name,
                    sample_params,
                    grads_dict,
                    context=context_vec,
                    force=True,
                )
                level_manager.pop_last_metrics(level_name)
                updated_params.append(updated)
                total_magnitude += magnitude
            fast_state.cms_params[level_name] = updated_params
            return total_magnitude

        if _is_batched_delta_dict(self.cms.blocks[level_name], base_params):
            updated, magnitude = _apply_tensorized_cms_grads(
                module=self.cms.blocks[level_name],
                level_name=level_name,
                base_params=base_params,
                chunk_inputs=chunk_inputs,
                chunk_teach=chunk_teach,
                chunk_active=chunk_active,
                reduction=self.config.cms_chunk_reduction,
                level_manager=fast_state.level_manager,
            )
            fast_state.cms_params[level_name] = updated
            return magnitude

        forward_params = params_with_deltas(self.cms.blocks[level_name], base_params)
        params_req = require_grad_params(forward_params)
        with torch.enable_grad():
            prediction = call_with_params(self.cms.blocks[level_name], params_req, chunk_inputs)
            loss = _chunk_loss(
                prediction,
                chunk_teach,
                mask_f,
                reduction=self.config.cms_chunk_reduction,
            )
        grads = torch.autograd.grad(
            loss,
            tuple(params_req.values()),
            retain_graph=False,
            allow_unused=True,
        )
        grads_dict = grads_to_dict(params_req, grads)
        context_vec = chunk_inputs.mean(dim=(0, 1))
        level_manager = _manager_for_sample(fast_state.level_manager, 0)
        updated, magnitude = level_manager.apply_grads(
            level_name,
            base_params,
            grads_dict,
            context=context_vec,
            force=True,
        )
        fast_state.cms_params[level_name] = updated
        level_manager.pop_last_metrics(level_name)
        return magnitude

    def pop_update_stats(self) -> Dict[str, Dict[str, float]]:
        stats = self.last_update_stats
        self.last_update_stats = {}
        return stats

    def _passes_surprise(self, surprise_value: float | None) -> bool:
        if self.surprise_threshold is None:
            return True
        if surprise_value is None:
            return False
        return surprise_value >= self.surprise_threshold

    def _is_level_allowed(self, level_name: str) -> bool:
        if self.allowed_levels is None:
            return True
        return level_name in self.allowed_levels or (
            level_name.startswith("titan") and "titan" in self.allowed_levels
        )

    def _record_gate(self, level_name: str, *, hit: bool) -> None:
        stats_key = f"gate.{level_name}"
        self.last_update_stats.setdefault(stats_key, {})
        self.last_update_stats[stats_key]["gate_hit"] = 1.0 if hit else 0.0
```

## File: src/nested_learning/model.py
```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Protocol, Sequence, cast

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .fast_state import ModelFastState, build_block_fast_state
from .hope.block import (
    HOPEAttentionBlock,
    HOPEAttentionBlockConfig,
    HOPEBlock,
    HOPEBlockConfig,
    HOPESelfModBlock,
    HOPESelfModBlockConfig,
)
from .levels import LevelSpec
from .transformer import TransformerBlock, TransformerBlockConfig


@dataclass
class ModelConfig:
    vocab_size: int
    dim: int
    num_layers: int
    heads: int
    titan_level: LevelSpec
    cms_levels: Sequence[LevelSpec]
    cms_flush_partial_at_end: bool = False
    cms_use_layernorm: bool = True
    optimizers: Dict[str, dict] | None = None
    teach_scale: float = 1.0
    teach_clip: float = 0.0
    teach_schedule: Dict[str, float] | None = None
    gradient_checkpointing: bool = False
    surprise_threshold: float | None = None
    surprise_metric: str = "l2"
    freeze_backbone: bool = False
    qk_l2_norm: bool = False
    local_conv_window: int | None = None
    self_mod_lr: float = 1e-3
    self_mod_hidden: int = 4
    self_mod_chunk_size: int = 1
    self_mod_chunk_size_memory: int | None = None
    self_mod_objective: str = "l2"
    self_mod_stopgrad_vhat: bool = True
    self_mod_use_rank1_precond: bool = True
    self_mod_use_alpha: bool = True
    self_mod_use_skip: bool = True
    self_mod_momentum: float = 0.0
    self_mod_adaptive_q: bool = False
    self_mod_local_conv_window: int | None = 4
    transformer_mlp_hidden_multiplier: int = 4
    transformer_activation: str = "gelu"
    block_variant: str = "hope_hybrid"


class HOPEModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.dim)
        self.base_teach_scale = config.teach_scale
        self.base_teach_clip = config.teach_clip
        self._runtime_teach_scale = config.teach_scale
        self._runtime_teach_clip = config.teach_clip
        self.gradient_checkpointing = config.gradient_checkpointing
        self._surprise_threshold = config.surprise_threshold
        self._surprise_metric = "l2"
        self._allowed_update_levels: set[str] | None = None
        self._allowed_update_layers: set[int] | None = None
        variant = str(config.block_variant).strip().lower()
        if variant == "hope_attention":
            attn_block_config = HOPEAttentionBlockConfig(
                dim=config.dim,
                heads=config.heads,
                cms_levels=config.cms_levels,
                cms_flush_partial_at_end=config.cms_flush_partial_at_end,
                cms_use_layernorm=config.cms_use_layernorm,
                qk_l2_norm=config.qk_l2_norm,
                local_conv_window=config.local_conv_window,
                self_mod_lr=config.self_mod_lr,
                optimizer_configs=config.optimizers or {},
            )
            self.blocks = nn.ModuleList(
                [HOPEAttentionBlock(attn_block_config) for _ in range(config.num_layers)]
            )
        elif variant == "hope_hybrid":
            hybrid_block_config = HOPEBlockConfig(
                dim=config.dim,
                heads=config.heads,
                titan_level=config.titan_level,
                cms_levels=config.cms_levels,
                cms_flush_partial_at_end=config.cms_flush_partial_at_end,
                cms_use_layernorm=config.cms_use_layernorm,
                qk_l2_norm=config.qk_l2_norm,
                local_conv_window=config.local_conv_window,
                self_mod_lr=config.self_mod_lr,
                self_mod_hidden=config.self_mod_hidden,
                optimizer_configs=config.optimizers or {},
            )
            self.blocks = nn.ModuleList(
                [HOPEBlock(hybrid_block_config) for _ in range(config.num_layers)]
            )
        elif variant == "hope_selfmod":
            selfmod_block_config = HOPESelfModBlockConfig(
                dim=config.dim,
                cms_levels=config.cms_levels,
                cms_flush_partial_at_end=config.cms_flush_partial_at_end,
                cms_use_layernorm=config.cms_use_layernorm,
                qk_l2_norm=config.qk_l2_norm,
                selfmod_adaptive_q=config.self_mod_adaptive_q,
                selfmod_local_conv_window=config.self_mod_local_conv_window,
                eta_scale=config.self_mod_lr,
                selfmod_chunk_size=config.self_mod_chunk_size,
                selfmod_chunk_size_memory=config.self_mod_chunk_size_memory,
                selfmod_objective=config.self_mod_objective,
                selfmod_stopgrad_vhat=config.self_mod_stopgrad_vhat,
                selfmod_use_rank1_precond=config.self_mod_use_rank1_precond,
                selfmod_use_alpha=config.self_mod_use_alpha,
                selfmod_use_skip=config.self_mod_use_skip,
                selfmod_momentum=config.self_mod_momentum,
                self_mod_lr=config.self_mod_lr,
                optimizer_configs=config.optimizers or {},
            )
            self.blocks = nn.ModuleList(
                [HOPESelfModBlock(selfmod_block_config) for _ in range(config.num_layers)]
            )
        elif variant == "transformer":
            transformer_block_config = TransformerBlockConfig(
                dim=config.dim,
                heads=config.heads,
                mlp_hidden_multiplier=config.transformer_mlp_hidden_multiplier,
                activation=config.transformer_activation,
                qk_l2_norm=config.qk_l2_norm,
                local_conv_window=config.local_conv_window,
            )
            self.blocks = nn.ModuleList(
                [TransformerBlock(transformer_block_config) for _ in range(config.num_layers)]
            )
        else:
            raise ValueError(
                f"Unsupported block_variant={config.block_variant!r}; expected one of "
                "['hope_attention', 'hope_hybrid', 'hope_selfmod', 'transformer']"
            )
        self.norm = nn.LayerNorm(config.dim)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        # Weight tying keeps the LM head gradient aligned with the embedding space.
        self.lm_head.weight = self.embed.weight
        self._latest_update_metrics: Dict[str, float] = {}
        self.set_surprise_metric(config.surprise_metric)
        self.set_surprise_threshold(self._surprise_threshold)
        if config.freeze_backbone:
            self.freeze_backbone()

    def set_teach_runtime(self, *, scale: float | None = None, clip: float | None = None) -> None:
        if scale is not None:
            self._runtime_teach_scale = scale
        if clip is not None:
            self._runtime_teach_clip = clip

    def set_surprise_threshold(self, threshold: float | None) -> None:
        self._surprise_threshold = threshold
        for block in self.blocks:
            cast(_UpdateControlledBlock, block).set_surprise_threshold(threshold)

    def get_surprise_threshold(self) -> float | None:
        return self._surprise_threshold

    def set_surprise_metric(self, metric: str) -> None:
        normalized = str(metric).strip().lower()
        allowed = {"l2", "loss", "logit_entropy"}
        if normalized not in allowed:
            raise ValueError(
                f"Unsupported surprise_metric={metric!r}; expected one of {sorted(allowed)}"
            )
        self._surprise_metric = normalized
        for block in self.blocks:
            cast(_UpdateControlledBlock, block).set_surprise_metric(normalized)

    def get_surprise_metric(self) -> str:
        return self._surprise_metric

    def set_allowed_update_levels(self, levels: set[str] | None) -> None:
        self._allowed_update_levels = levels.copy() if levels is not None else None
        for block in self.blocks:
            cast(_UpdateControlledBlock, block).set_allowed_levels(self._allowed_update_levels)

    def get_allowed_update_levels(self) -> set[str] | None:
        return None if self._allowed_update_levels is None else self._allowed_update_levels.copy()

    def set_allowed_update_layers(self, layers: set[int] | None) -> None:
        if layers is None:
            self._allowed_update_layers = None
            return
        normalized: set[int] = set()
        total = len(self.blocks)
        for idx in layers:
            layer_idx = int(idx)
            if layer_idx < 0:
                layer_idx = total + layer_idx
            if not (0 <= layer_idx < total):
                raise ValueError(f"Invalid layer index {idx} for model with {total} layers")
            normalized.add(layer_idx)
        self._allowed_update_layers = normalized

    def get_allowed_update_layers(self) -> set[int] | None:
        return None if self._allowed_update_layers is None else self._allowed_update_layers.copy()

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        teach_signal: torch.Tensor | None = None,
        teach_signals: list[torch.Tensor] | None = None,
        fast_state: ModelFastState | None = None,
        surprise_value: float | None = None,
    ) -> torch.Tensor:
        logits, _pre_norm = self.forward_with_pre_norm(
            tokens,
            teach_signal=teach_signal,
            teach_signals=teach_signals,
            fast_state=fast_state,
            surprise_value=surprise_value,
        )
        return logits

    def forward_with_pre_norm(
        self,
        tokens: torch.Tensor,
        *,
        teach_signal: torch.Tensor | None = None,
        teach_signals: list[torch.Tensor] | None = None,
        fast_state: ModelFastState | None = None,
        surprise_value: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = self._run_blocks(
            tokens,
            teach_signal=teach_signal,
            teach_signals=teach_signals,
            fast_state=fast_state,
            surprise_value=surprise_value,
        )
        pre_norm = cast(torch.Tensor, x)
        x = self.norm(pre_norm)
        logits = self.lm_head(x)
        if teach_signal is not None or teach_signals is not None:
            self._latest_update_metrics = self._gather_block_stats()
        return logits, pre_norm

    def forward_with_block_outputs(
        self,
        tokens: torch.Tensor,
        *,
        teach_signal: torch.Tensor | None = None,
        teach_signals: list[torch.Tensor] | None = None,
        fast_state: ModelFastState | None = None,
        surprise_value: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        x, block_outputs = self._run_blocks(
            tokens,
            teach_signal=teach_signal,
            teach_signals=teach_signals,
            fast_state=fast_state,
            surprise_value=surprise_value,
            collect_outputs=True,
        )
        pre_norm = x
        x = self.norm(x)
        logits = self.lm_head(x)
        if teach_signal is not None or teach_signals is not None:
            self._latest_update_metrics = self._gather_block_stats()
        return logits, pre_norm, block_outputs

    def _run_blocks(
        self,
        tokens: torch.Tensor,
        *,
        teach_signal: torch.Tensor | None,
        fast_state: ModelFastState | None,
        teach_signals: list[torch.Tensor] | None = None,
        surprise_value: float | None = None,
        collect_outputs: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, list[torch.Tensor]]:
        x = self.embed(tokens)
        block_outputs: list[torch.Tensor] = []
        runtime_scale = self._runtime_teach_scale
        runtime_clip = self._runtime_teach_clip
        if teach_signals is not None:
            if len(teach_signals) != len(self.blocks):
                raise ValueError(
                    f"teach_signals length {len(teach_signals)} "
                    f"does not match blocks {len(self.blocks)}"
                )
            if teach_signal is not None:
                raise ValueError("Provide either teach_signal or teach_signals, not both.")
        if fast_state is not None and len(fast_state.blocks) != len(self.blocks):
            raise ValueError("fast_state.blocks length does not match model.blocks")

        require_external = self._surprise_metric in {"loss", "logit_entropy"}
        if require_external and self._surprise_threshold is not None:
            if (teach_signal is not None or teach_signals is not None) and surprise_value is None:
                raise ValueError(
                    f"surprise_metric={self._surprise_metric} requires passing surprise_value "
                    "when model.surprise_threshold is set."
                )

        base_surprise = surprise_value
        scaled_global_signal: torch.Tensor | None = None
        if base_surprise is None and teach_signal is not None and self._surprise_metric == "l2":
            scaled_global_signal = teach_signal * runtime_scale
            if runtime_clip > 0:
                norm = scaled_global_signal.norm(dim=-1, keepdim=True)
                scale = torch.clamp(norm / runtime_clip, min=1.0)
                scaled_global_signal = scaled_global_signal / scale
            base_surprise = float(scaled_global_signal.norm(dim=-1).mean().item())

        for idx, block in enumerate(self.blocks):
            block_state = None if fast_state is None else fast_state.blocks[idx]
            scaled_signal = None
            block_surprise = base_surprise
            if teach_signal is not None:
                if scaled_global_signal is None:
                    scaled_signal = teach_signal * runtime_scale
                    if runtime_clip > 0:
                        norm = scaled_signal.norm(dim=-1, keepdim=True)
                        scale = torch.clamp(norm / runtime_clip, min=1.0)
                        scaled_signal = scaled_signal / scale
                else:
                    scaled_signal = scaled_global_signal
                if (
                    self._allowed_update_layers is not None
                    and idx not in self._allowed_update_layers
                ):
                    scaled_signal = None
            if teach_signals is not None:
                scaled_signal = teach_signals[idx] * self._runtime_teach_scale
                if self._surprise_metric == "l2" and base_surprise is None:
                    block_surprise = float(scaled_signal.norm(dim=-1).mean().item())
                if self._runtime_teach_clip > 0:
                    norm = scaled_signal.norm(dim=-1, keepdim=True)
                    scale = torch.clamp(norm / self._runtime_teach_clip, min=1.0)
                    scaled_signal = scaled_signal / scale
                if (
                    self._allowed_update_layers is not None
                    and idx not in self._allowed_update_layers
                ):
                    scaled_signal = None

            def block_call(
                hidden: torch.Tensor,
                *,
                blk=block,
                sig=scaled_signal,
                st=block_state,
                sv=block_surprise,
            ) -> torch.Tensor:
                return blk(
                    hidden,
                    teach_signal=sig,
                    surprise_value=sv,
                    fast_state=st,
                )

            if torch.is_grad_enabled() and self.training and self.gradient_checkpointing:
                x = checkpoint(block_call, x, use_reentrant=False)
            else:
                x = block_call(x)
            if collect_outputs:
                block_outputs.append(x)
        if collect_outputs:
            return x, block_outputs
        return x

    def _gather_block_stats(self) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        for idx, block in enumerate(self.blocks):
            pop_fn = getattr(block, "pop_update_stats", None)
            if callable(pop_fn):
                stats = cast(Dict[str, Dict[str, float]], pop_fn())
                for level_name, payload in stats.items():
                    prefix = f"layer{idx}.{level_name}"
                    for key, value in payload.items():
                        metrics[f"{prefix}.{key}"] = value
        return metrics

    def pop_update_metrics(self) -> Dict[str, float]:
        metrics = self._latest_update_metrics
        self._latest_update_metrics = {}
        return metrics

    def init_fast_state(
        self,
        *,
        batch_size: int = 1,
        fast_state_batch_mode: str = "shared",
    ) -> ModelFastState:
        states = []
        for block in self.blocks:
            if isinstance(block, HOPEBlock):
                specs = [block.config.titan_level, *block.config.cms_levels]
                state = build_block_fast_state(
                    titan_module=block.titan_memory,
                    cms_blocks=dict(block.cms.blocks.items()),
                    specs=specs,
                    optimizer_configs=block.config.optimizer_configs,
                    default_lr=block.config.self_mod_lr,
                    batch_size=batch_size,
                    fast_state_batch_mode=fast_state_batch_mode,
                )
                states.append(state)
            elif isinstance(block, HOPEAttentionBlock):
                specs = list(block.config.cms_levels)
                state = build_block_fast_state(
                    titan_module=None,
                    cms_blocks=dict(block.cms.blocks.items()),
                    specs=specs,
                    optimizer_configs=block.config.optimizer_configs,
                    default_lr=block.config.self_mod_lr,
                    batch_size=batch_size,
                    fast_state_batch_mode=fast_state_batch_mode,
                )
                states.append(state)
            elif isinstance(block, HOPESelfModBlock):
                specs = list(block.config.cms_levels)
                state = build_block_fast_state(
                    titan_module=None,
                    cms_blocks=dict(block.cms.blocks.items()),
                    selfmod_module=block.selfmod,
                    specs=specs,
                    optimizer_configs=block.config.optimizer_configs,
                    default_lr=block.config.self_mod_lr,
                    batch_size=batch_size,
                    fast_state_batch_mode=fast_state_batch_mode,
                )
                states.append(state)
            elif isinstance(block, TransformerBlock):
                state = build_block_fast_state(
                    titan_module=None,
                    cms_blocks={},
                    specs=(),
                    optimizer_configs={},
                    default_lr=0.0,
                    batch_size=batch_size,
                    fast_state_batch_mode=fast_state_batch_mode,
                )
                states.append(state)
            else:
                raise TypeError(f"Unsupported block type for fast state: {type(block)}")
        return ModelFastState(blocks=states)

    def freeze_backbone(self) -> None:
        """
        Freeze the shared transformer spine (embeddings, attention blocks, norm, LM head).
        HOPE/TITAN/CMS memories remain trainable for adapter-style finetuning.
        """
        for p in self.embed.parameters():
            p.requires_grad = False
        for p in self.norm.parameters():
            p.requires_grad = False
        for p in self.lm_head.parameters():
            p.requires_grad = False
        for block in self.blocks:
            attn = getattr(block, "attn", None)
            if isinstance(attn, nn.Module):
                for p in attn.parameters():
                    p.requires_grad = False


class _UpdateControlledBlock(Protocol):
    def set_surprise_threshold(self, threshold: float | None) -> None: ...

    def set_surprise_metric(self, metric: str) -> None: ...

    def set_allowed_levels(self, allowed: set[str] | None) -> None: ...
```

## 12) Senior Review Follow-up Implementation (2026-02-21)
Applied core systems recommendations from the latest senior review to reduce residual Python/per-sample overhead in `tensorized_cms`.

### 12.1 Implemented code changes
1. Batched optimizer-state updates in DeepMomentum
- File: `src/nested_learning/optim/deep.py`
- Added optional `sample_mask` support to preserve inactive-sample state.
- Added batched-context handling for `nl_l2_precond` (`context: [B, D]`).
- Added mask-aware first/second-moment updates for batched gradients.

2. Batched apply API in manager
- File: `src/nested_learning/optim/manager.py`
- Added `apply_grads_batched(...)` for `[B, ...]` params/grads with optional `sample_mask`.
- Preserves no-op behavior for all-inactive masks.

3. Tensorized fast-state manager strategy
- File: `src/nested_learning/fast_state.py`
- `fast_state_batch_mode=tensorized_cms` now uses a single `LevelOptimizerManager` with batched optimizer states.

4. Tensorized CMS hot-path reroute
- File: `src/nested_learning/hope/block.py`
- `_apply_tensorized_cms_grads(...)` now uses manager-level batched apply when manager is shared.
- Keeps list-manager fallback path as oracle-safe fallback.

### 12.2 Added/updated tests
- Added: `tests/test_optim_manager_batched.py`
  - validates masked batched updates and all-inactive no-op behavior.
- Updated: `tests/test_optim.py`
  - added batched `sample_mask` state-preservation test.
  - added batched-context `nl_l2_precond` projection test.
- Updated: `tests/test_checkpoint_resume_parity.py`
  - added `(batch_size=4, fast_state_batch_mode=tensorized_cms)` coverage.

### 12.3 Validation run (local)
Executed and passing:
- `tests/test_optim_manager_batched.py`
- `tests/test_optim.py`
- `tests/test_tensorized_cms_matches_listmode.py`
- `tests/test_tensorized_cms_multilevel_matches_listmode.py`
- `tests/test_tensorized_cms_hybrid_wiring_matches_listmode.py`
- `tests/test_fast_state_isolation.py`
- `tests/test_batched_equals_sequential.py`
- `tests/test_checkpoint_resume_parity.py`

### 12.4 Pending for decision-grade closure
- Rerun 3090 throughput+quality protocol with the new batched-apply implementation.
- Re-evaluate throughput gate (`tensorized/list >= 1.5x`) on small+medium benches and B-scaling beyond 4.
