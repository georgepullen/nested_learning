# Multi-Batch Fast-State Architecture: Problem Evaluation for Expert Design

Date: 2026-02-18
Repo: `georgepullen/nested_learning` fork
Primary objective: enable semantics-preserving multi-batch training for NL/HOPE fast-state paths, while reducing single-run wall-clock below 24h on available RunPod budget.

## 1) Direct answer up front

### Does the paper claim this is already solved?

Short answer: **No explicit claim in the provided print text.**

- The paper emphasizes multi-level/parallel optimization with per-component context flow, not a concrete implementation pattern for batched, per-sample isolated fast-state training.
- The extracted print text does not specify mini-batch/fast-state isolation mechanics (no explicit `batch`/`mini-batch` term found in `reports/paper/NL-print.extracted.clean.txt`).
- The paper states experimental results and token budgets, but detailed setup is deferred to appendix material not included in this extracted main text (`reports/paper/NL-print.extracted.clean.txt:549-553`).

### Would solving this be a novel contribution?

Short answer: **Likely yes as an incremental systems/method contribution for NL implementations.**

- It is not a new NL theory claim by itself.
- It is a high-value architectural contribution if it preserves per-context semantics and materially improves throughput/cost.
- It becomes publishable-quality incremental work if paired with: (1) semantic equivalence tests vs strict `batch_size=1`, (2) speed/cost gains, and (3) downstream eval parity.

## 2) Evidence snapshot (Oracle + paper + code)

### Paper-level semantic anchors

- NL is framed as nested/multi-level/parallel optimization, each with its own context flow (`reports/paper/NL-print.extracted.clean.txt:31-33`).
- Update frequency is defined per component; each component has its own context/optimization problem (`reports/paper/NL-print.extracted.clean.txt:349-367`).

Interpretation: context separation is central; accidental cross-context contamination at fast-update levels is semantically risky.

### Oracle-level implementation anchors

- Oracle identifies fast-state support as per-context adaptation (`reports/NL_IMPLEMENTATION_ORACLE.md:81`).
- Oracle states large-scale parity is not guaranteed and paper-scale reproduction remains incomplete (`reports/NL_IMPLEMENTATION_ORACLE.md:173-182`).
- Oracle includes the exact fail-fast logic currently used:
  - shared fast-state across batch warning/error
  - strict mode enforces `batch_size=1`
  (`reports/NL_IMPLEMENTATION_ORACLE.md:12877-12896`).

### Repo implementation anchors (current bottleneck)

- Paper-faithful config encodes the constraint:
  - comment: shared context when `batch_size>1`
  - `data.batch_size: 1`
  (`configs/pilot_paper_faithful.yaml:16-18`).
- Training initializes a single fast-state object for the step from batched tokens (`src/nested_learning/training.py:575-582`).
- Fast-state CMS parameters are delta tensors per parameter without sample axis (`src/nested_learning/fast_state.py:24`, `src/nested_learning/fast_state.py:28-32`).
- Fast CMS update aggregates chunk loss and writes one updated state for the level (`src/nested_learning/hope/block.py:571-599`).
- Strict semantics guard is tested (`tests/test_fast_state_batch_semantics.py:7-25`).

## 3) Problem statement

Current state:

- Strict paper-faithful path requires `batch_size=1`.
- Near-faithful path (`batch_size>1`, strict guard off) mixes contexts via shared fast-state writes.
- Run-time throughput at current settings is too slow for cost constraints (target: sub-24h runs).

Architectural problem:

- Design a multi-batch execution model that **preserves per-context fast-state semantics** while materially increasing throughput.

## 4) Why naive batching fails (mechanics)

At each step:

1. A batched token tensor `tokens` with shape `[B, T]` is sampled.
2. One fast-state object is created for the model/block.
3. CMS/TITAN online updates are computed from chunk loss over batch+time.
4. Updated deltas are written back into the same fast-state parameter dict.

Failure mode:

- Different samples in the batch contribute to the same delta update tensors.
- This couples unrelated context trajectories inside a single “fast memory” object.
- That violates the intended per-context isolation used by strict paper-faithful semantics.

## 5) Non-negotiable requirements for a viable architecture

### Semantic requirements

1. Per-sample fast-state isolation during online updates.
2. No cross-sample writes to fast-state deltas.
3. Compatibility with current strict paper-faithful mode as reference baseline.

### Performance requirements

1. Practical target: `>= 83.3 steps/hour` for 2k-step sub-24h runs.
2. No catastrophic memory blow-up from batched fast-state tensors.
3. Works on commodity RunPod classes (not only very high-end pods).

### Reliability requirements

1. Checkpoint/resume must preserve per-sample fast-state invariants.
2. Telemetry must expose contamination/leak checks.
3. Deterministic equivalence mode for testing (`B=1` vs isolated `B>1`).

## 6) Candidate architecture patterns

### A) Batched per-sample fast-state tensorization (recommended primary design)

Core idea:

- Add an explicit batch axis to fast-state deltas and optimizer states for memory modules.
- Represent each memory parameter delta as `[B, ...param_shape]`.
- Apply forward/update functionally per sample (vectorized), not as one shared module update.

Pros:

- Preserves semantics by construction.
- Enables real batch parallelism.
- Clean equivalence story against `batch_size=1`.

Cons:

- Larger memory footprint.
- Requires refactor of level optimizer manager + fast update path.

### B) Micro-batch sequential accumulation with isolated fast-state pools

Core idea:

- Keep per-sample isolation, but process small micro-batches sequentially (or grouped by similar level schedules).

Pros:

- Lower implementation risk than full tensorization.
- Better memory control.

Cons:

- Smaller speedup ceiling.
- May still miss sub-24h target on current hardware.

### C) Sample-sharded actor model (asynchronous fast-state workers)

Core idea:

- Distribute samples across worker lanes, each with independent fast-state evolution; synchronize only outer/meta gradients.

Pros:

- Strong isolation.
- Potentially good scale-out path.

Cons:

- High systems complexity.
- Harder determinism and reproducibility.

## 7) Recommended design direction

Primary recommendation: **A with B as fallback mode.**

Stage the rollout:

1. Add “isolated multi-batch fast-state” mode behind explicit config flag.
2. Implement CMS path first (biggest current pain), then TITAN self-mod path.
3. Keep current stop-grad semantics unchanged initially; do not combine with differentiable online writes in first milestone.
4. Use micro-batch fallback when memory pressure exceeds threshold.

## 8) Concrete implementation touchpoints

Priority files:

- `src/nested_learning/fast_state.py`
- `src/nested_learning/functional.py`
- `src/nested_learning/hope/block.py`
- `src/nested_learning/titan/self_modifying.py`
- `src/nested_learning/optim/manager.py`
- `src/nested_learning/training.py`
- `configs/pilot_paper_faithful.yaml` and new batched-faithful config(s)

New tests to require:

1. Per-sample isolation test:
   - changing one sample in batch does not alter another sample’s fast-state trajectory.
2. Equivalence test:
   - isolated `B=K` run equals concatenated `K x B=1` runs (within tolerance).
3. Leakage sentinel test:
   - inject orthogonal synthetic contexts; verify no cross-sample delta contamination.
4. Resume parity test:
   - checkpoint/resume keeps per-sample state mapping stable.

## 9) Quality gates and success criteria

### Gate G0 (semantic correctness)

- All new isolation/equivalence tests pass.
- Existing strict `batch_size=1` tests remain green.

### Gate G1 (throughput)

- On target RunPod profile, sustained `>= 83.3 steps/hour` for canonical 2k baseline config.

### Gate G2 (model quality)

- No NaN/Inf regressions.
- Eval drift vs strict baseline within agreed tolerance on pilot suite.

### Gate G3 (operability)

- Resume drill passes.
- Telemetry includes explicit per-sample fast-state diagnostics.

## 10) Interaction with current GitHub project plan

This work is a partial pivot from pure equation-fidelity sequence to execution-feasibility work, but it is not a strategic fork away from paper alignment.

Practical effect:

- It should be inserted before heavy RunPod matrix execution (#11/#12), otherwise cost/time will dominate and block milestone completion.
- It can be framed as “fidelity-preserving execution architecture” that unlocks the rest of the board.

## 11) Novelty framing for publication

Reasonable claim:

- “A semantics-preserving batched fast-state execution architecture for Nested Learning/HOPE training that retains per-context adaptation behavior while enabling practical throughput.”

Avoid over-claiming:

- Do not claim new NL theory.
- Claim implementation-level and empirical contributions:
  - formal isolation invariants,
  - efficient batched algorithm,
  - measured cost/performance gains,
  - parity/quality evidence.

## 12) Open architect questions

1. What is acceptable memory overhead for batched fast-state tensors per layer?
2. Should optimizer state be fully per-sample, or can parts be shared without semantic leakage?
3. What determinism guarantees are required for parity testing?
4. Is single-GPU first sufficient, or should FSDP compatibility be designed in from day one?
5. What eval tolerance thresholds define “semantics preserved” for publication claims?

## 13) Decision recommendation

Given current cost constraints, proceed with a targeted architecture effort for **isolated multi-batch fast-state** before scaling run matrices.

This is the highest-leverage path to keep the project viable financially while preserving the paper-alignment mission.
