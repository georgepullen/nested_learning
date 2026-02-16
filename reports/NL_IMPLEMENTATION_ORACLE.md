# Nested Learning Repository Oracle Analysis

Generated at: `2026-02-16T11:43:13`

Repository: `/Users/georgepullen/Documents/research/Cyril/kmccleary_nested_learning`

Branch: `main`

Commit: `5606f404d65da0612f61bbd2c7e268035f75472f`

Last commit: `2026-01-10 13:48:43 -0600 | 5606f40 | Fix CI mypy + setup-python`

Paper analyzed: `/Users/georgepullen/Documents/research/papers/NL-print.pdf`

Paper extraction used: `reports/paper/NL-print.extracted.clean.txt`

## 1) Scope and Method

This document is an implementation oracle for this repository. It is designed to let a reader:

1. Understand the full architecture and training/evaluation pipeline.
2. Map implementation details back to the attached NeurIPS print paper equations (Eq. 1–31 available in this version).
3. See what is paper-faithful vs practical/deferred.
4. Inspect every functional code file verbatim in one place.

Inclusion policy:

- Included verbatim: every `*.py`, `*.sh`, `*.yaml`, `*.yml`, and `*.toml` file in the repository.
- Summarized (not verbatim): markdown documentation files.
- Excluded from verbatim appendix: generated artifacts/non-functional outputs (e.g., metrics JSON, binary artifacts, lock/cache outputs).

## 2) Repository Functional Index

- Total functional files included verbatim: **147**
- Python files: **101**
- Shell scripts: **9**
- YAML/YML configs: **36**
- TOML files: **1**
- Approximate total lines included in verbatim appendix: **15428**

### Architecture at a glance

- Core library: `src/nested_learning/`
- Train entrypoints: `train.py`, `train_dist.py`, `train_fsdp.py`, `train_deepspeed.py`
- Configs: `configs/`
- Data/eval/checkpoint scripts: `scripts/`
- Tests: `tests/`

## 3) Paper Equation-to-Code Mapping (Attached NL-print.pdf)

The attached NeurIPS print version exposes Eq. (1)–(31). Later equation references in comments/docs (e.g., 83–97) correspond to fuller versions discussed in project docs, not fully present in this print PDF.

| Paper Eq. | Mathematical intent | Implementation mapping |
|---|---|---|
| Eq. (1) | Associative memory objective M* = argmin L~(M(K);V) | Conceptual foundation; reflected in memory-module optimization framing in `src/nested_learning/titan/memory.py` and update managers `src/nested_learning/optim/manager.py`. Not implemented as one direct callable objective API. |
| Eq. (2)–(6) | MLP training + LSS reinterpretation of gradient descent | `src/nested_learning/training.py` (`compute_teach_signal`) computes a closed-form `dL/dh` for next-token CE via softmax-residual projection through the detached LM-head weight (`residual @ head_weight`). Under this setup it matches autograd in tests (`tests/test_teach_signal.py`), while still being an engineering realization rather than an explicit solver for the paper’s outer-product associative-memory objective form in Eq. 5–6. The update-pass logic in `src/nested_learning/hope/block.py` consumes this signal. |
| Eq. (7)–(11) | Momentum as nested memory | `src/nested_learning/optim/deep.py` (`DeepMomentum`) implements an EMA-based momentum (`grad_avg.mul_(beta).add_(update, alpha=1-beta)`). This captures the *spirit* of momentum as a memory that accumulates past gradients (Eq. 10–11), but uses a standard exponential moving average rather than the paper's explicit associative-memory optimization formulation `argmin_m −⟨m, ∇L⟩ + η‖m − m_t‖²`. |
| Eq. (12)–(16) | Linear attention associative memory update | Repository uses softmax attention backbone in `src/nested_learning/backbones.py` and TITAN/CMS memory modules; unnormalized linear attention equations are used conceptually, not as a literal module. |
| Eq. (17)–(24) | Deep optimizer extensions (preconditioning, L2 objective, nonlinear outputs) | `src/nested_learning/optim/deep.py` provides five `DeepMomentum` variants, but these are **simplified heuristics**, not direct implementations of the paper equations. Specifically: (a) `preconditioned` uses Adam-style second-moment preconditioning (≈ Eq. 20 direction but via EMA, not the paper's associative-memory framing); (b) `l2_objective` adds `0.1 * mean(grad)` — a loose heuristic that does **not** implement the paper's delta-rule update `(αI − ∇L⊤∇L)m − ηP∇L` from Eq. 21–22; (c) `dmgd` applies `tanh` nonlinearity to the EMA update — the paper's DMGD (Eq. 23) uses an MLP-parameterized momentum (`m(u)` is a neural network), not a scalar nonlinearity on a linear EMA; (d) `muon` is `preconditioned` + `tanh` — the paper's Muon (Eq. 24) uses `Newton-Schulz(·)` as σ(·), not `tanh`; Newton-Schulz is only implemented in the *outer* M3 optimizer (`src/nested_learning/optim/m3.py`), not in the inner `DeepMomentum` module; (e) `nl_l2_precond` implements rank-1 context-orthogonal projection, loosely inspired by Eq. 19 preconditioning but not a direct mapping. |
| Eq. (25)–(29) | Backprop as associative memory + L2-style variant | The paper proposes `W_{t+1} = W_t(I − x_t x_t⊤) − η∇L` (Eq. 28–29), a GD variant with an explicit outer-product projection that removes the component of W along the current input. The implementation does **not** implement this projection. Instead, `_chunk_loss` in `src/nested_learning/hope/block.py` constructs a standard MSE-style loss `‖pred − stopgrad(pred − δ)‖²` whose gradient w.r.t. prediction is proportional to the teach-signal δ, and `compute_teach_signal` in `training.py` provides the δ. This is a practical gradient-shaping construction that achieves δ-directed updates, but omits the `(I − x_t x_t⊤)` weight decay term that is central to Eq. 28–29. |
| Eq. (30) | CMS chained MLP forward | Directly represented by `src/nested_learning/cms.py` (`CMS.forward`) and HOPE block compositions in `src/nested_learning/hope/block.py`. |
| Eq. (31) | Per-level update periodicity C(l) | Enforced with `LevelSpec.update_period`, `LevelClock`, online chunk buffering, and level-manager stepping in `src/nested_learning/levels.py` and `src/nested_learning/hope/block.py`. |


## 4) Core Implementation Analysis

### 4.1 Model variants and runtime composition

`src/nested_learning/model.py` defines `ModelConfig` and `HOPEModel` with four variants:

- `hope_attention`: attention -> CMS (paper-faithful minimal variant)
- `hope_hybrid`: attention + TITAN memory + CMS (legacy/exploratory — uses `SelfModifier` hypernetwork from `hope/self_mod.py`, not the paper's self-modifying Titans)
- `hope_selfmod`: self-modifying Titans -> CMS (paper-faithful primary variant)
- `transformer`: baseline attention -> MLP (comparison baseline)

Key implementation details:

- Embedding and LM head are weight tied.
- Teach-signal runtime scaling/clipping is configurable.
- Surprise gating supports `l2`, `loss`, and `logit_entropy` metrics.
- Fast-state is supported for per-context online adaptation.

### 4.2 Frequency semantics and nested levels

`src/nested_learning/levels.py` (`LevelSpec`, `LevelClock`) is the explicit realization of the paper’s update-frequency ordering idea (Def. 2 and Eq. 31 behavior).

- `update_period` is the core timescale parameter.
- `warmup_steps`/`jitter` extend scheduling behavior.
- `LevelOptimizerManager` in `src/nested_learning/optim/manager.py` applies level-specific updates and tracks optimizer-local metrics.

### 4.3 CMS implementation

`src/nested_learning/cms.py` implements the chain of MLP memory blocks (paper Eq. 30 concept).

`src/nested_learning/hope/block.py` implements practical CMS update behavior:

- online chunk buffering by update period,
- active-token masking,
- partial remainder flush option,
- surprise-threshold gating,
- both normal module updates and fast-state delta updates.

The update objective is implemented via `_chunk_loss`, a gradient-shaping construction that makes `d(loss)/d(prediction)` proportional to the desired teach delta under masking/reduction.

### 4.4 Self-modifying Titans

`src/nested_learning/titan/self_modifying.py` is the most mathematically dense module.

Highlights:

- Memory stack (`M_k`, `M_v`, `M_q`, `M_eta`, `M_alpha`, `M_memory`) represented as residual 2-layer MLP memories.
- Chunked updates for “other” memories and `M_memory` are separated.
- DGD-like updates use per-token gradients computed efficiently with `torch.func.grad` + `vmap`.
- Optional rank-1 preconditioning, alpha retention, momentum buffers, adaptive q, and causal local conv are exposed as config toggles.

### 4.5 Training loop and teach-signal mechanics

`src/nested_learning/training.py` controls end-to-end training.

Critical behaviors:

- `compute_teach_signal` computes `(softmax(logits) − one_hot(target)) @ W_head` on detached tensors, i.e., a closed-form gradient for CE loss w.r.t. the pre-head hidden state (`dL/dh`). In this repository’s CE path, it is numerically aligned with autograd (validated in `tests/test_teach_signal.py`). It still differs from the paper’s exposition style because the code computes the gradient directly rather than instantiating Eq. 25–26 as an explicit associative-memory optimization problem.
- Supports per-layer teach signals (`delta_l`) via `forward_with_block_outputs` + `_compute_layer_teach_signals` (real autograd through block outputs).
- Supports online chunked training where inner updates happen between chunk losses.
- Enforces fail-fast behavior for paper-faithful constraints in unsupported distributed modes.
- Supports AdamW / Muon hybrid / M3 outer optimizer choices.

### 4.6 Fast-state design

`src/nested_learning/fast_state.py` + `src/nested_learning/functional.py` implement meta+delta fast-state:

- forward uses `meta + delta`,
- online updates write only deltas,
- enabling context-local adaptation while preserving meta-parameter trainability.

### 4.7 Data, tokenizer, and eval stack

- Data construction: `src/nested_learning/data.py`, `scripts/data/*`
- Tokenizer support: `src/nested_learning/tokenizer.py`, `scripts/data/train_tokenizer.py`
- Evaluation scripts: `scripts/eval/*` (zeroshot, NIAH, continual, passkey, PG19, plotting/summaries)
- Memorization path: `src/nested_learning/memorize.py`

## 5) Fidelity, Ambiguities, and Gaps

Based on code + docs (`docs/PAPER_COMPLIANCE.md`) and the attached paper text:

### 5.1 Implemented strongly

- Multi-frequency update semantics (CMS/TITAN-level scheduling).
- Explicit online update paths with surprise gating.
- Rich self-modifying memory update machinery.
- Test coverage on many invariants (teach signal, CMS updates, fast-state behavior, optimizer policies).

### 5.2 Ambiguities or pragmatic approximations

1. **Paper objective details vs implementation objective shaping:**
   - CMS and memory updates use practical losses (`_chunk_loss`, MSE-style constructs) that are faithful in gradient direction but are engineering realizations, not one-to-one symbolic forms for every paper equation.
2. **Print paper is condensed:**
   - This NeurIPS print PDF references appendices/expanded formulations not fully present here; code/docs include interpretations of those broader equations.
3. **Surprise metric variants:**
   - `loss` and `logit_entropy` are offered in addition to L2 teach norm, useful for ablations but beyond strict single-metric reading.
4. **Teach signal computation style differs from paper presentation:**
   - `compute_teach_signal` produces `dL/dh` directly via a closed-form softmax-residual calculation (and is validated against autograd in tests). The paper presents this quantity in an associative-memory optimization framing (Eq. 5–6); the code computes it directly without instantiating that optimization.

### 5.3 Deep optimizer variants are heuristic simplifications

The `DeepMomentum` variants in `src/nested_learning/optim/deep.py` are the **largest divergence** between the implementation and the paper's mathematical formulations (Eq. 17–24). Specifically:

1. **No MLP-parameterized momentum (Eq. 23 DMGD):** The paper defines DMGD with momentum as a multi-layer neural network `m(u)` whose parameters are updated by an inner objective. The implementation uses a scalar EMA with `tanh` nonlinearity — structurally different from a learned MLP momentum.
2. **No delta-rule momentum update (Eq. 21–22):** The paper's L2-objective extension produces `m_{i+1} = (αI − ∇L⊤∇L)m − ηP∇L`, a Widrow-Hoff / delta-rule that allows the momentum to manage capacity by subtracting previously stored gradient directions. The `l2_objective` variant adds `0.1 * mean(grad)` — a fixed heuristic unrelated to this formula.
3. **Muon nonlinearity mismatch (Eq. 24):** The paper specifies `σ(·) = Newton-Schulz(·)` for the Muon optimizer. `DeepMomentum(variant="muon")` uses `tanh`. Newton-Schulz orthogonalization is only present in the outer `M3` optimizer (`optim/m3.py`), not in the inner momentum module.
4. **Preconditioning is Adam-style, not associative (Eq. 19–20):** The paper frames preconditioning as the momentum learning a mapping between a value matrix P and gradients. The `preconditioned` variant uses standard Adam second-moment EMA (`v = β₂v + (1−β₂)g²; g/√v`), which achieves diagonal preconditioning but not the key-value associative memory framing.

These variants are useful engineering baselines and correctly labeled in code, but should not be cited as direct implementations of Eq. 17–24.

### 5.4 Not fully built out relative to strict paper-faithful large-scale path

1. Full bi-level meta-learning experiments over explicit task episodes are not present.
2. No backprop-through-online-writes boundary-state training procedure; writes are stop-grad explicit passes.
3. Distributed paper-faithful parity is intentionally limited:
   - DDP disables/guards some online + per-layer mechanisms,
   - FSDP path is practical/offline-oriented.
4. Large-scale benchmark parity (paper-scale compute/data) is not guaranteed by this repo alone. The paper reports results at 340M / 760M / 1.3B parameters trained on 30B–100B tokens; this repo has run smoke and pilot-scale experiments, not full-scale reproduction runs.
5. **Undiscussed files:** `src/nested_learning/titan/model.py` (364 lines) defines a standalone `TitanOnlyModel` — an older/alternative architecture variant not referenced in the main HOPE pipeline. `src/nested_learning/hope/self_mod.py` (41 lines) defines a `SelfModifier` hypernetwork (concatenates key+value+error → MLP → delta) used only by the legacy `HOPEBlock` hybrid variant, not by the paper-faithful `HOPESelfModBlock`.
6. **Hyperparameter correspondence is unverified:** The paper does not publish all hyperparameters (eta_scale, chunk sizes, surprise thresholds, CMS level periods). The repo's defaults are reasonable engineering choices; whether they match the paper's internal configurations is unknown.

## 6) Gap-to-Paper Roadmap: What Needs to Be Done to Match the Exact Paper

This section enumerates every known divergence between this implementation and the paper's mathematical formulations, ordered by estimated impact on reproducing the paper's results.

### 6.1 High impact — likely required for paper-scale results

#### 6.1.1 Implement MLP-parameterized deep momentum (Eq. 23–24, DMGD)

**Paper:** Momentum `m` is a multi-layer neural network. The inner objective `L^(2)(m; u, I)` (e.g., dot-product similarity `⟨m(u⊤), 1⟩`) is optimized w.r.t. the MLP parameters of `m`. The weight update becomes `W_{i+1} = W_i + σ(m_{i+1}(u_i))`, where σ can be Newton-Schulz orthogonalization.

**Current state:** `DeepMomentum` uses a scalar EMA (`grad_avg = β·grad_avg + (1−β)·update`) with optional `tanh` nonlinearity. No MLP, no inner objective optimization, no Newton-Schulz on the momentum output.

**Work required:**
- Replace `DeepMomentum`'s linear EMA state with a small MLP (e.g., 2-layer residual, matching `ResidualMLPMemory` architecture).
- Add an inner optimization step: compute `∇_{m_params} L^(2)(m; u, I)` and apply it to the MLP's parameters each outer step.
- Add Newton-Schulz as a σ option on the momentum output (can reuse `_newton_schulz` from `m3.py`).
- Expose variant selection: `dmgd_mlp`, `muon_ns` alongside existing heuristic variants.

**Files:** `src/nested_learning/optim/deep.py`, new inner-loop logic.

#### 6.1.2 Implement delta-rule momentum update (Eq. 21–22)

**Paper:** `m_{i+1} = (α_{i+1}I − ∇L(W_i;x_i)⊤ ∇L(W_i;x_i)) m_i − η_t P_i ∇L(W_i;x_i)`. The `−∇L⊤∇L · m` term is a Widrow-Hoff correction that subtracts previously stored gradient directions from momentum, enabling better capacity management.

**Current state:** The `l2_objective` variant adds `0.1 * mean(grad)` — unrelated to the delta rule.

**Work required:**
- Implement: `m_new = (α·I − g⊤g) · m_old − η · P · g` where `g = ∇L(W;x)`.
- For the preconditioning matrix P, support at minimum: identity (→ Eq. 22 without preconditioning) and second-moment diagonal (→ Adam-style).
- This is a matrix-vector operation per parameter tensor; compute cost is modest.

**Files:** `src/nested_learning/optim/deep.py` (new variant `delta_rule`).

#### 6.1.3 Implement the L2-variant of gradient descent (Eq. 28–29)

**Paper:** `W_{t+1} = W_t(I − x_t x_t⊤) − η ∇_y L ⊗ x_t`. The `W_t(I − x_t x_t⊤)` term is a rank-1 weight decay that removes the component of W projecting onto the current input, making the update aware of input dependencies.

**Current state:** The inner update uses standard MSE-style `_chunk_loss` with teach-signal-shaped targets. No `(I − x_t x_t⊤)` projection is applied.

**Work required:**
- In the CMS / memory update path, after computing the gradient update, apply the additional weight modification: `W = W @ (I − x x⊤ / ‖x‖²)` (or the batched equivalent over chunk tokens).
- This can be implemented as a post-step hook in `LevelOptimizerManager.apply_grads()` or directly in `_update_cms_chunk`.
- Note: the paper says "we use this optimizer as the internal optimizer of our HOPE architecture" (after Eq. 29), so this is specifically intended for the CMS inner updates.

**Files:** `src/nested_learning/hope/block.py` (`_update_cms_chunk`), `src/nested_learning/optim/manager.py`.

#### 6.1.4 Backpropagation through online writes (boundary-state gradients)

**Paper:** Implies a chunk-parallel training procedure where the outer loss gradient flows *through* the online memory updates (boundary states between chunks are differentiable).

**Current state:** All online writes use `torch.no_grad()` / `.detach()`. The `PAPER_COMPLIANCE.md` explicitly documents this as a known semantic gap.

**Work required:**
- Make the CMS delta updates differentiable: instead of `stop_grad(delta)`, allow gradients to flow from later chunk losses back through the CMS parameter updates of earlier chunks.
- For self-modifying Titans: make `_apply_chunk_update_seq` differentiable (currently uses `vmap(grad(...))` in a stop-grad context).
- This is architecturally the hardest item — it requires careful memory management (gradient checkpointing through the update chain) and may need `torch.utils.checkpoint` per chunk.
- Validate: the outer loss gradient w.r.t. meta-parameters should now include terms from the online update pathway, not just the read pathway.

**Files:** `src/nested_learning/hope/block.py` (all `_update_*` methods), `src/nested_learning/titan/self_modifying.py` (`_apply_chunk_update_seq`), `src/nested_learning/training.py` (online chunk loop).

### 6.2 Medium impact — improves fidelity for specific experiments

#### 6.2.1 Adam as optimal associative memory (Paper Section C.4)

**Paper:** Claims Adam (with a small modification) is the optimal associative memory for gradients. The appendix presumably derives the specific form.

**Current state:** Not implemented or tested. The repo supports multiple outer optimizers (Muon is the primary default in pilot/mid/target configs, with AdamW and M3 available), but none implement the specific “optimal associative memory” Adam variant from paper Section C.4.

**Work required:**
- Once the full arXiv appendix is available, extract the specific Adam modification from Section C.4.
- Implement as a variant of the outer optimizer or as an alternative inner memory update rule.
- Compare against standard AdamW in ablations.

**Files:** `src/nested_learning/optim/` (new module or variant).

#### 6.2.2 Preconditioning as associative key-value mapping (Eq. 19–20)

**Paper:** The momentum `m` maps gradients (keys) to preconditioning values P_i (values). The paper argues that functions of the Hessian provide the most meaningful P_i.

**Current state:** `preconditioned` uses diagonal Adam-style second-moment. `nl_l2_precond` projects gradients orthogonal to a context vector. Neither frames P as a key-value mapping or uses Hessian information.

**Work required:**
- Implement a variant where P_i is derived from curvature information (e.g., diagonal Fisher, or Hessian-vector products via `torch.autograd.functional.hvp`).
- Frame the preconditioning step as key-value storage: gradient → Hessian-informed direction.
- This is expensive at scale; consider low-rank Hessian approximations (e.g., K-FAC style).

**Files:** `src/nested_learning/optim/deep.py` (new variant).

#### 6.2.3 Full self-referential learning loop

**Paper (abstract):** HOPE "learns how to modify itself by learning its own update algorithm." The self-modifying Titans learn eta/alpha gates that control the update, but the *update rule structure itself* (DGD-like gradient descent on the memory) is fixed.

**Current state:** The learned eta/alpha gates provide input-dependent learning rates and retention, which is a form of learned update control. But the update rule template (`w_new = alpha·w − eta·P·grad`) is hardcoded, not itself learned.

**Work required:**
- To fully match the paper's vision: the update rule itself should be parameterizable and meta-learned.
- Possible approach: replace the fixed DGD template with a small hypernetwork that takes (current_state, gradient, input) and outputs the full parameter delta.
- Note: The existing `SelfModifier` in `hope/self_mod.py` (concatenates key+value+error → MLP → delta) is closer to this idea but is only used in the legacy `HOPEBlock`, not the paper-faithful `HOPESelfModBlock`.

**Files:** `src/nested_learning/titan/self_modifying.py`, possibly integrate `hope/self_mod.py` pattern.

#### 6.2.4 Bi-level meta-learning evaluation (task-episode format)

**Paper:** Discusses models that can "continually learn" and "fast adapt to a new task." The NL framework supports explicit inner/outer task structure.

**Current state:** Training uses standard language modeling (next-token prediction). No task-episode evaluation where the model adapts to a new task distribution within context and is evaluated on held-out queries.

**Work required:**
- Implement a few-shot evaluation harness: sample task → provide k-shot context → evaluate on held-out examples → measure adaptation quality.
- Add continual-learning benchmarks with explicit domain-shift boundaries (partially started in `scripts/eval/continual.py` and `continual_classification.py`).
- This is primarily an evaluation/experiment concern, not an architecture change.

**Files:** `scripts/eval/` (new evaluation scripts), configs for task-episode data.

### 6.3 Lower impact — completeness and rigor

#### 6.3.1 Linear attention as associative memory module (Eq. 12–16)

**Paper:** Uses unnormalized linear attention to demonstrate that the recurrence `M_{t+1} = M_t + v_t k_t⊤` is equivalent to one step of gradient descent on an associative memory objective (Eq. 15–16).

**Current state:** The repo uses softmax attention (standard `F.scaled_dot_product_attention`). Linear attention is used only as theoretical scaffolding in the paper.

**Work required (optional):**
- Implement an unnormalized linear attention module matching Eq. 13–14.
- Use it to validate the associative-memory equivalence claim (Eq. 15–16) via unit tests.
- Not required for HOPE results, but would complete the NL framework demonstration.

**Files:** New module in `src/nested_learning/backbones.py` or `src/nested_learning/linear_attention.py`.

#### 6.3.2 Expose the Eq. 1 associative memory objective as a callable API

**Paper:** Definition 1 is the foundation — `M* = argmin L̃(M(K); V)`. All other formulations derive from specific choices of L̃ and optimization method.

**Current state:** `AssocMemory` protocol has `forward()` and `update()` but no explicit `objective()` method. The associative memory formulation is implicit in the update rules.

**Work required:**
- Add an `objective(keys, values) → loss` method to the `AssocMemory` protocol.
- Each memory type computes its specific L̃ (dot-product, L2, etc.).
- Enables direct testing of the associative memory equivalence claims.

**Files:** `src/nested_learning/assoc_memory.py`, implementors (`titan/memory.py`, `titan/self_modifying.py`, `cms.py`).

#### 6.3.3 Large-scale training reproduction

**Paper:** Reports results at 340M, 760M (30B tokens), and 1.3B (100B tokens) parameter scales. In the attached NeurIPS print text, the explicit token counts visible are 30B and 100B.

**Current state:** Configs exist for mid (760M) and target (1.3B) scales (`configs/hope/mid.yaml`, `configs/hope/target.yaml`). FSDP scaling guide exists. Actual training has been smoke/pilot-scale only.

**Work required:**
- Compute budget: dual RTX 6000 Ada (2×48GB) is documented but may be insufficient for 100B-token 1.3B runs without significant wall-clock time.
- Run full training at each scale with paper-matched hyperparameters (where known).
- Evaluate on the paper's benchmark suite (WikiText, LMB, PIQA, HellaSwag, WinoGrande, ARC, SIQA, BoolQ).
- Compare against the paper's Table 1 numbers.

**Files:** `configs/hope/target.yaml`, `scripts/eval/zeroshot.py`, `docs/compute_plan.md`.

#### 6.3.4 Hyperparameter alignment with paper

**Paper:** Does not publish all internal hyperparameters. Key unknowns include:
- Exact eta_scale values for self-modifying memories
- CMS level update periods and how they scale with model size
- Surprise threshold values used during training
- Inner optimizer learning rates for CMS and TITAN updates
- Chunk sizes for self-modifying memory updates

**Work required:**
- Extract and incorporate all hyperparameter tables/details from the fuller arXiv material referenced by the print paper (and verify against the latest public version).
- Align `configs/hope/*.yaml` defaults to match.
- Run sensitivity ablations to understand impact of each hyperparameter.

**Files:** `configs/hope/*.yaml`, `docs/` (new hyperparameter alignment doc).

### 6.4 Summary matrix

| Gap | Paper Reference | Difficulty | Impact | Blocking for paper parity? |
|---|---|---|---|---|
| MLP deep momentum (DMGD) | Eq. 23–24 | Medium | High | Yes (optimizer expressivity) |
| Delta-rule momentum | Eq. 21–22 | Low | Medium–High | Yes (capacity management) |
| L2-variant GD for CMS | Eq. 28–29 | Low–Medium | High | Yes (paper says HOPE uses this) |
| Backprop through online writes | Implicit in training procedure | High | High | Yes (gradient flow fidelity) |
| Adam as optimal assoc. memory | Section C.4 | Unknown | Medium | Needs appendix |
| Hessian-based preconditioning | Eq. 19–20 | Medium–High | Medium | No (ablation target) |
| Full self-referential update learning | Abstract / Section 3 | High | Medium | No (eta/alpha gates are partial) |
| Bi-level meta-learning eval | Paper discussion | Medium | Medium | No (eval only) |
| Linear attention module | Eq. 12–16 | Low | Low | No (theoretical only) |
| Assoc memory objective API | Eq. 1 | Low | Low | No (testing convenience) |
| Large-scale training | Table 1 | Low (code) / High (compute) | High | Yes (results parity) |
| Hyperparameter alignment | Full arXiv version | Low | High | Blocked until appendix details are integrated into this repo |

## 7) Test Evidence (including newly added hypothesis tests)

New tests added in this task:

- `tests/test_paper_hypotheses.py`
  - `test_cms_frequency_matches_floor_schedule_without_partial_flush`
  - `test_cms_frequency_matches_ceil_schedule_with_partial_flush`
  - `test_online_training_smoke_produces_nonzero_teach_and_update_metrics`

Executed command:

```bash
PYTHONPATH=src python3.12 -m pytest   tests/test_teach_signal.py   tests/test_cms.py   tests/test_selfmod_dgd_linear.py   tests/test_m3.py   tests/test_paper_hypotheses.py
```

Observed result:

- `15 passed, 1 warning`
- Warning is dataloader `pin_memory` on MPS/CPU environment; not a correctness failure.

Interpretation:

- Equation-like CMS frequency laws (floor/ceil under flush mode) are now explicitly regression-tested.
- Tiny online training path with inferred chunk-size clamping was exercised and produced finite metrics with nonzero teach/update telemetry.
- Core mathematical invariants (teach signal, CMS behavior, optimizer update flow) remain green in targeted suite.

## 8) Markdown File Summaries

- `CHANGELOG.md`: **Changelog**. All notable changes to this project will be documented here. The format loosely follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and uses semantic versioning once tagged releases begin.
- `README.md`: **Nested Learning Reproduction**. ![Python](https://img.shields.io/badge/python-3.12+-blue)
- `TODO.md`: **Project TODOs**. - [ ] **Data Engineering**
- `docs/FSDP_SCALING_GUIDE.md`: **FSDP/ZeRO Scaling Guide (RTX 6000 Ada Dual-GPU Rig)**. This note captures the configuration we will use for the Stage 2 mid (≈760 M) and target (≈1.3 B) HOPE models when running on the dual RTX 6000 Ada workstation (2× 48 GB). It accompanies the new Hydra configs `configs/hope/mid_fsdp.yaml` and `configs/hope/target_fsdp.yaml`.
- `docs/P4_REMEDIATION_PLAN.md`: **P4 Remediation Plan — Status & Tracking (Paper-Faithful HOPE/Nested Learning)**. This file started as an execution checklist for the P4 “paper faithfulness” sprint. It is now maintained as a **status page** so contributors can quickly see what’s implemented, what is verified by tests, and what follow‑ups remain.
- `docs/PAPER_COMPLIANCE.md`: **Paper Compliance / Fidelity Guide (Nested Learning / HOPE)**. This doc explains the **fidelity‑critical behaviors** (what the paper relies on) and how they map to this repo’s code, flags, and tests.
- `docs/PHASE2_LONG_CONTEXT_COMPARISON.md`: **Phase 2 – HOPE-Attention vs Transformer (Long-Context Sanity)**. This repo includes a lightweight Phase‑2 sanity check that compares **HOPE-Attention** (Attention → CMS) against a **baseline Transformer** on synthetic long‑context retrieval prompts.
- `docs/PHASE_2_PLAN.md`: **Phase 2 Plan – Execution & Results Packaging**. Before resuming large-scale runs, we must land the following **P0 faithfulness fixes** plus high-priority engineering upgrades. Each item lists the concrete code touchpoints, validation criteria, and downstream dependencies.
- `docs/compute_plan.md`: **Compute Reservation Plan (Stage 2)**. - Cluster: 2× nodes with dual NVIDIA RTX 6000 Ada (48 GB VRAM) + 64-core CPU + 512 GB RAM.
- `docs/continual_classification_eval.md`: **Continual Classification Evaluation (CLINC / Banking77 / DBpedia14)**. The Nested Learning paper highlights **class-incremental continual learning** in the text classification
- `docs/continual_eval.md`: **Continual-Learning Evaluation Guide**. Use `scripts/eval/continual.py` to quantify forgetting across streaming segments. Supply:
- `docs/data_pipeline.md`: **Data Pipeline (Stage 2)**. This document explains how to generate tokenizer artifacts and token shards for Stage 2 training.
- `docs/env_matrix.md`: **Environment Matrix – Stage 2**. This document captures the exact runtime state used for the Stage 2 sprint so collaborators can reproduce the setup without guesswork.
- `docs/experiments_report.md`: **Experiments Report – Nested Learning Reproduction**. _Draft covering work completed through 9 Nov 2025. This document is meant to accompany the initial public release so contributors understand what has been reproduced and what remains._
- `docs/future_directions.md`: **Future Directions – Nested Learning Reproduction**. This roadmap outlines high-impact areas for contributors once the initial public release is out. Items are organized by theme and roughly prioritized.
- `docs/guide.md`: **Nested Learning Reproduction Guide**. This guide is a self-contained reference for reproducing Google's Nested Learning (HOPE) architecture within this repository. It captures the expectations set by the Nested Learning and TITAN papers, plus the quality bar demonstrated by lucidrains’ TITAN implementation. Follow it end-to-end to prepare the environment, process data, train smoke models, and run the evaluation suite.
- `docs/phase2_comparison.md`: **Phase 2 – HOPE-Attention vs Transformer Baseline**. Phase 2 is “implementation-complete” when we can compare the **paper-defined HOPE-Attention** variant
- `docs/planner_convo_01.md`: **0) One‑paragraph TL;DR**. ***User***
- `docs/release_checklist.md`: **Release Checklist (Stage 2)**. Use this list before tagging/publishing any checkpoint bundle.
- `docs/release_plan.md`: **Release Readiness Checklist (v0.1)**. Goal: Package the Nested Learning reproduction so others can run data prep, pilot training, and evaluation out of the box using modest hardware (dual RTX 6000 Ada). Larger-scale configs remain documented for future scaling/community contributions.
- `docs/scaling_guidance.md`: **Scaling Guidance – Nested Learning Reproduction**. This document describes how to extend the current smoke-tested Nested Learning (HOPE) stack to larger datasets, hardware targets, and experiment scopes without changing the core codebase.
- `docs/spec_interfaces.md`: **Interface Notes for Nested Learning Modules**. - `LevelSpec`: name, update_period, warmup, jitter, optimizer binding.
- `docs/sprint_next_plan.md`: **Sprint Plan – Stage 2 Pilot & Results Sprint**. **Window:** Nov 10 – Nov 17, 2025 (7 days)
- `docs/stability_journal.md`: **Stability Journal – Nested Learning Reproduction**. _Chronological notes on debugging and stabilizing the HOPE/TITAN implementation. Useful for future contributors digging into NaN fixes or regression hunting._
- `docs/stage1_plan.md`: **Stage 1 Plan – Nested Learning (HOPE) Architecture Reproduction**. This document specifies the full execution plan for Stage 1: reproducing the Nested Learning (NL) architecture—specifically the HOPE model comprising multi-frequency levels, TITAN-style memory, Continuum Memory System (CMS), self-modification pathways, and deep optimizers. It is self-contained and assumes no additional context.
- `docs/stage2_plan.md`: **Stage 2 Plan – Nested Learning (HOPE) Results Reproduction**. This document details Stage 2 goals: reproduce the key experimental results from Google’s Nested Learning (HOPE) paper/blog using the Stage 1 codebase. It is self-contained and assumes Stage 1 deliverables (architecture, training harness, tests, `uv` environment) are ready.
- `docs/stage2_progress.md`: **Stage 2 Progress Report (Nov 9, 2025)**. This note captures the current state of Stage 2 (results reproduction) so collaborators can pick up the dual-GPU workflow immediately.
- `docs/templates/checkpoint_report.md`: **Checkpoint Report Template**. Copy this template into `reports/checkpoints/<run>.md` (or similar) for every published checkpoint.
- `docs/zeroshot_eval.md`: **Zero-shot Evaluation Guide**. The script `scripts/eval/zeroshot.py` evaluates HOPE checkpoints on
- `google_papers/Nested_Learning/Nested_Learning.md`: **Nested Learning: The Illusion of Deep Learning Architectures**. PAGE 1
- `google_papers/TITANs/TITANs.md`: **Titans: Learning to Memorize at Test Time**. PAGE 1
- `reports/ablations.md`: **Planned Ablations – Pilot Run**. This document tracks the ablation studies we intend to run once the 3 B-token pilot checkpoint is available. The goal is to isolate the contributions of teach-signal scaling, CMS chunk accumulation, self-modifiers, and optimizer choices (AdamW vs Muon) before moving to larger configs.
- `reports/stage2_smoke.md`: **Stage 2 Smoke Artifact Summary**. - 2× NVIDIA RTX 6000 Ada (49 GB VRAM each)


## 9) Functional File Index (included verbatim below)

| File | Role | Lines |
|---|---|---|
| `.github/ISSUE_TEMPLATE/config.yml` | functional file | 3 |
| `.github/workflows/ci.yml` | functional file | 81 |
| `configs/ablations/cms_sparse.yaml` | runtime configuration | 47 |
| `configs/ablations/selfmod_chunked_8_64.yaml` | runtime configuration | 25 |
| `configs/ablations/selfmod_momentum_off.yaml` | runtime configuration | 24 |
| `configs/ablations/selfmod_momentum_on.yaml` | runtime configuration | 24 |
| `configs/ablations/selfmod_no_alpha.yaml` | runtime configuration | 24 |
| `configs/ablations/selfmod_no_cms.yaml` | runtime configuration | 24 |
| `configs/ablations/selfmod_rank1_precond_off.yaml` | runtime configuration | 24 |
| `configs/data/continual_segments_sample.yaml` | runtime configuration | 10 |
| `configs/data/fineweb_edu_longdoc_filtered_sample.yaml` | runtime configuration | 15 |
| `configs/data/fineweb_edu_mixture_full.yaml` | runtime configuration | 15 |
| `configs/data/fineweb_edu_mixture_sample.yaml` | runtime configuration | 15 |
| `configs/data/refinedweb_mixture.yaml` | runtime configuration | 49 |
| `configs/data/refinedweb_mixture_filtered.yaml` | runtime configuration | 49 |
| `configs/data/refinedweb_mixture_full.yaml` | runtime configuration | 49 |
| `configs/data/refinedweb_mixture_sample.yaml` | runtime configuration | 52 |
| `configs/hope/mid.yaml` | runtime configuration | 116 |
| `configs/hope/mid_fsdp.yaml` | runtime configuration | 45 |
| `configs/hope/pilot.yaml` | runtime configuration | 3 |
| `configs/hope/pilot_attention.yaml` | runtime configuration | 10 |
| `configs/hope/pilot_selfmod.yaml` | runtime configuration | 21 |
| `configs/hope/pilot_transformer.yaml` | runtime configuration | 10 |
| `configs/hope/target.yaml` | runtime configuration | 143 |
| `configs/hope/target_fsdp.yaml` | runtime configuration | 45 |
| `configs/mid_smoke.yaml` | runtime configuration | 97 |
| `configs/mid_stage2.yaml` | runtime configuration | 108 |
| `configs/mid_stage2_smoke.yaml` | runtime configuration | 100 |
| `configs/mid_titan_baseline.yaml` | runtime configuration | 90 |
| `configs/pilot.yaml` | runtime configuration | 122 |
| `configs/pilot_paper_faithful.yaml` | runtime configuration | 33 |
| `configs/pilot_selfmod_paper_faithful.yaml` | runtime configuration | 19 |
| `configs/pilot_smoke.yaml` | runtime configuration | 78 |
| `configs/resolved/cms_sparse_eval.yaml` | runtime configuration | 106 |
| `configs/resolved/phase2_pilot_attention_eval.yaml` | runtime configuration | 50 |
| `configs/resolved/phase2_pilot_transformer_eval.yaml` | runtime configuration | 50 |
| `pyproject.toml` | packaging/toolchain config | 69 |
| `scripts/__init__.py` | automation/ops script | 2 |
| `scripts/checkpoint/verify.py` | validation/check tooling | 25 |
| `scripts/checks/tokenizer_coverage_guard.py` | validation/check tooling | 99 |
| `scripts/compute/create_reservations.sh` | automation/ops script | 39 |
| `scripts/data/__init__.py` | data pipeline tooling | 3 |
| `scripts/data/check_tokenizer.py` | data pipeline tooling | 81 |
| `scripts/data/check_tokenizer_coverage.py` | data pipeline tooling | 35 |
| `scripts/data/filter_corpus.py` | data pipeline tooling | 119 |
| `scripts/data/process_mixture.py` | data pipeline tooling | 52 |
| `scripts/data/run_full.sh` | data pipeline tooling | 147 |
| `scripts/data/run_sample.sh` | data pipeline tooling | 89 |
| `scripts/data/shard_corpus.py` | data pipeline tooling | 156 |
| `scripts/data/train_tokenizer.py` | data pipeline tooling | 166 |
| `scripts/data/validate_mixture.py` | data pipeline tooling | 74 |
| `scripts/eval/__init__.py` | evaluation tooling | 2 |
| `scripts/eval/compare_variants.py` | evaluation tooling | 524 |
| `scripts/eval/continual.py` | evaluation tooling | 199 |
| `scripts/eval/continual_classification.py` | evaluation tooling | 178 |
| `scripts/eval/niah.py` | evaluation tooling | 203 |
| `scripts/eval/niah_suite.py` | evaluation tooling | 409 |
| `scripts/eval/passkey.py` | evaluation tooling | 170 |
| `scripts/eval/pg19_perplexity.py` | evaluation tooling | 168 |
| `scripts/eval/phase2_memorization_delta_smoke.py` | evaluation tooling | 101 |
| `scripts/eval/plot_continual_classification.py` | evaluation tooling | 67 |
| `scripts/eval/plot_forgetting.py` | evaluation tooling | 45 |
| `scripts/eval/plot_niah_suite.py` | evaluation tooling | 62 |
| `scripts/eval/run_pilot_suite.sh` | evaluation tooling | 201 |
| `scripts/eval/summarize_eval.py` | evaluation tooling | 111 |
| `scripts/eval/zeroshot.py` | evaluation tooling | 391 |
| `scripts/package_pilot_release.sh` | automation/ops script | 137 |
| `scripts/run_cpu_ddp_smoke.sh` | automation/ops script | 9 |
| `scripts/run_e2e_smoke.sh` | automation/ops script | 65 |
| `scripts/run_smoke.sh` | automation/ops script | 21 |
| `scripts/tests/run_passkey_smoke.sh` | automation/ops script | 36 |
| `src/nested_learning/__init__.py` | model/runtime core | 4 |
| `src/nested_learning/assoc_memory.py` | model/runtime core | 24 |
| `src/nested_learning/backbones.py` | model/runtime core | 112 |
| `src/nested_learning/cms.py` | model/runtime core | 93 |
| `src/nested_learning/continual_classification.py` | model/runtime core | 137 |
| `src/nested_learning/continual_streaming.py` | model/runtime core | 284 |
| `src/nested_learning/data.py` | model/runtime core | 154 |
| `src/nested_learning/device.py` | model/runtime core | 22 |
| `src/nested_learning/fast_state.py` | model/runtime core | 66 |
| `src/nested_learning/functional.py` | model/runtime core | 62 |
| `src/nested_learning/hope/__init__.py` | HOPE block/update core | 1 |
| `src/nested_learning/hope/block.py` | HOPE block/update core | 1728 |
| `src/nested_learning/hope/self_mod.py` | HOPE block/update core | 41 |
| `src/nested_learning/instrumentation.py` | model/runtime core | 39 |
| `src/nested_learning/levels.py` | model/runtime core | 95 |
| `src/nested_learning/logging_utils.py` | model/runtime core | 65 |
| `src/nested_learning/memorize.py` | model/runtime core | 383 |
| `src/nested_learning/model.py` | model/runtime core | 465 |
| `src/nested_learning/optim/__init__.py` | optimizer core | 1 |
| `src/nested_learning/optim/deep.py` | optimizer core | 103 |
| `src/nested_learning/optim/factory.py` | optimizer core | 14 |
| `src/nested_learning/optim/m3.py` | optimizer core | 122 |
| `src/nested_learning/optim/manager.py` | optimizer core | 140 |
| `src/nested_learning/titan/__init__.py` | TITAN/self-mod core | 1 |
| `src/nested_learning/titan/memory.py` | TITAN/self-mod core | 89 |
| `src/nested_learning/titan/model.py` | TITAN/self-mod core | 364 |
| `src/nested_learning/titan/self_modifying.py` | TITAN/self-mod core | 725 |
| `src/nested_learning/tokenizer.py` | model/runtime core | 29 |
| `src/nested_learning/tokenizer_coverage.py` | model/runtime core | 78 |
| `src/nested_learning/training.py` | model/runtime core | 1193 |
| `src/nested_learning/transformer.py` | model/runtime core | 93 |
| `tests/conftest.py` | test coverage | 8 |
| `tests/test_attention_features.py` | test coverage | 46 |
| `tests/test_build_model_from_cfg_selfmod.py` | test coverage | 45 |
| `tests/test_cms.py` | test coverage | 92 |
| `tests/test_cms_delta_rule.py` | test coverage | 44 |
| `tests/test_cms_flush_partial.py` | test coverage | 48 |
| `tests/test_compare_variants_cli.py` | test coverage | 101 |
| `tests/test_continual_classification.py` | test coverage | 120 |
| `tests/test_data_split_fallbacks.py` | test coverage | 113 |
| `tests/test_device_resolution.py` | test coverage | 11 |
| `tests/test_distributed_fail_fast.py` | test coverage | 47 |
| `tests/test_eval_builders.py` | test coverage | 37 |
| `tests/test_faithfulness_harness.py` | test coverage | 58 |
| `tests/test_fast_state_batch_semantics.py` | test coverage | 27 |
| `tests/test_fast_state_forward_equivalence.py` | test coverage | 26 |
| `tests/test_fast_state_meta_grads.py` | test coverage | 38 |
| `tests/test_fast_state_selfmod_meta_grads.py` | test coverage | 55 |
| `tests/test_hope_block.py` | test coverage | 30 |
| `tests/test_hope_selfmod_fast_state_meta_unchanged.py` | test coverage | 32 |
| `tests/test_hope_selfmod_integration.py` | test coverage | 31 |
| `tests/test_hope_selfmod_update_pass.py` | test coverage | 35 |
| `tests/test_levels.py` | test coverage | 16 |
| `tests/test_m3.py` | test coverage | 28 |
| `tests/test_m3_slow_timing.py` | test coverage | 28 |
| `tests/test_memorization.py` | test coverage | 180 |
| `tests/test_model.py` | test coverage | 20 |
| `tests/test_optim.py` | test coverage | 51 |
| `tests/test_optimizer_param_policy.py` | test coverage | 77 |
| `tests/test_paper_faithful_configs.py` | test coverage | 38 |
| `tests/test_paper_hypotheses.py` | test coverage | 134 |
| `tests/test_phase2_memorization_delta.py` | test coverage | 51 |
| `tests/test_residual_mlp_memory.py` | test coverage | 32 |
| `tests/test_self_modifying_titans.py` | test coverage | 76 |
| `tests/test_selfmod_adaptive_q.py` | test coverage | 25 |
| `tests/test_selfmod_dgd_linear.py` | test coverage | 44 |
| `tests/test_selfmod_grad_flow.py` | test coverage | 34 |
| `tests/test_selfmod_local_conv.py` | test coverage | 18 |
| `tests/test_selfmod_online.py` | test coverage | 31 |
| `tests/test_surprise_metric.py` | test coverage | 135 |
| `tests/test_teach_signal.py` | test coverage | 162 |
| `tests/test_variants.py` | test coverage | 68 |
| `train.py` | training entrypoint | 19 |
| `train_deepspeed.py` | training entrypoint | 122 |
| `train_dist.py` | training entrypoint | 37 |
| `train_fsdp.py` | training entrypoint | 200 |


## 10) Verbatim Functional Code Appendix

The following sections print each functional code file exactly as present at generation time.


### File: `.github/ISSUE_TEMPLATE/config.yml`

```yaml
blank_issues_enabled: false
contact_links: []
```

### File: `.github/workflows/ci.yml`

```yaml
name: CI

on:
  push:
    branches: ["main"]
  pull_request:
    branches: ["main"]

jobs:
  lint-and-test:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Set up uv
        uses: astral-sh/setup-uv@v3
        with:
          version: "0.9.8"

      - name: Sync dependencies
        run: uv sync --all-extras --dev

      - name: Ruff
        run: uv run ruff check .

      - name: Mypy
        run: uv run mypy src

      - name: Pytest
        run: uv run pytest

  cpu-ddp-smoke:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Set up uv
        uses: astral-sh/setup-uv@v3
        with:
          version: "0.9.8"

      - name: Sync dependencies
        run: uv sync --all-extras --dev

      - name: CPU DDP smoke (gloo backend)
        run: bash scripts/run_cpu_ddp_smoke.sh

  passkey-smoke:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - name: Set up uv
        uses: astral-sh/setup-uv@v3
        with:
          version: "0.9.8"

      - name: Sync dependencies
        run: uv sync --all-extras --dev

      - name: Run synthetic passkey memorization test
        run: bash scripts/tests/run_passkey_smoke.sh
```

### File: `configs/ablations/cms_sparse.yaml`

```yaml
defaults:
  - /pilot
  - _self_

model:
  dim: 384
  num_layers: 8
  heads: 6
  titan_level:
    name: titan
    update_period: 8
    optimizer_key: titan_opt
  cms_hidden_multiplier: 2
  cms_levels:
    - name: cms_fast
      update_period: 8
      optimizer_key: cms_opt
    - name: cms_mid
      update_period: 32
      optimizer_key: cms_opt
    - name: cms_slow
      update_period: 128
      optimizer_key: cms_opt
    - name: cms_ultra
      update_period: 512
      optimizer_key: cms_opt

data:
  seq_len: 1024
  batch_size: 2
  num_workers: 2

train:
  online_updates: true
  online_chunk_size: 0
  per_layer_teach_signal: true
  steps: 5000
  device: "cuda:1"
  checkpoint:
    dir: artifacts/checkpoints/pilot_cms_sparse
    save_interval: 1000
  log_interval: 25

logging:
  path: logs/pilot_cms_sparse_metrics.json
  run_name: pilot-cms-sparse
```

### File: `configs/ablations/selfmod_chunked_8_64.yaml`

```yaml
defaults:
  - /pilot
  - _self_

model:
  block_variant: hope_selfmod
  self_mod_chunk_size: 8
  self_mod_chunk_size_memory: 64

train:
  online_updates: true
  online_chunk_size: 0
  per_layer_teach_signal: true
  steps: 5000
  device: "cuda:1"
  checkpoint:
    dir: artifacts/checkpoints/pilot_selfmod_chunked_8_64
    save_interval: 1000

logging:
  enabled: true
  backend: json
  path: logs/pilot_selfmod_chunked_8_64_metrics.json
  run_name: pilot-selfmod-chunked-8-64
```

### File: `configs/ablations/selfmod_momentum_off.yaml`

```yaml
defaults:
  - /pilot
  - _self_

model:
  block_variant: hope_selfmod
  self_mod_momentum: 0.0

train:
  online_updates: true
  online_chunk_size: 0
  per_layer_teach_signal: true
  steps: 5000
  device: "cuda:1"
  checkpoint:
    dir: artifacts/checkpoints/pilot_selfmod_momentum_off
    save_interval: 1000

logging:
  enabled: true
  backend: json
  path: logs/pilot_selfmod_momentum_off_metrics.json
  run_name: pilot-selfmod-momentum-off
```

### File: `configs/ablations/selfmod_momentum_on.yaml`

```yaml
defaults:
  - /pilot
  - _self_

model:
  block_variant: hope_selfmod
  self_mod_momentum: 0.9

train:
  online_updates: true
  online_chunk_size: 0
  per_layer_teach_signal: true
  steps: 5000
  device: "cuda:1"
  checkpoint:
    dir: artifacts/checkpoints/pilot_selfmod_momentum_on
    save_interval: 1000

logging:
  enabled: true
  backend: json
  path: logs/pilot_selfmod_momentum_on_metrics.json
  run_name: pilot-selfmod-momentum-on
```

### File: `configs/ablations/selfmod_no_alpha.yaml`

```yaml
defaults:
  - /pilot
  - _self_

model:
  block_variant: hope_selfmod
  self_mod_use_alpha: false

train:
  online_updates: true
  online_chunk_size: 0
  per_layer_teach_signal: true
  steps: 5000
  device: "cuda:1"
  checkpoint:
    dir: artifacts/checkpoints/pilot_selfmod_no_alpha
    save_interval: 1000

logging:
  enabled: true
  backend: json
  path: logs/pilot_selfmod_no_alpha_metrics.json
  run_name: pilot-selfmod-no-alpha
```

### File: `configs/ablations/selfmod_no_cms.yaml`

```yaml
defaults:
  - /pilot
  - _self_

model:
  block_variant: hope_selfmod
  cms_levels: []

train:
  online_updates: true
  online_chunk_size: 0
  per_layer_teach_signal: true
  steps: 5000
  device: "cuda:1"
  checkpoint:
    dir: artifacts/checkpoints/pilot_selfmod_no_cms
    save_interval: 1000

logging:
  enabled: true
  backend: json
  path: logs/pilot_selfmod_no_cms_metrics.json
  run_name: pilot-selfmod-no-cms
```

### File: `configs/ablations/selfmod_rank1_precond_off.yaml`

```yaml
defaults:
  - /pilot
  - _self_

model:
  block_variant: hope_selfmod
  self_mod_use_rank1_precond: false

train:
  online_updates: true
  online_chunk_size: 0
  per_layer_teach_signal: true
  steps: 5000
  device: "cuda:1"
  checkpoint:
    dir: artifacts/checkpoints/pilot_selfmod_rank1_off
    save_interval: 1000

logging:
  enabled: true
  backend: json
  path: logs/pilot_selfmod_rank1_off_metrics.json
  run_name: pilot-selfmod-rank1-off
```

### File: `configs/data/continual_segments_sample.yaml`

```yaml
segments:
  - name: refinedweb_2018
    shards_dir: data/shards/refinedweb_sample
  - name: wikipedia_sample
    shards_dir: data/shards/wikipedia_sample
  - name: c4_sample
    shards_dir: data/shards/c4_sample
  - name: redpajama_sample
    shards_dir: data/shards/redpajama_sample
```

### File: `configs/data/fineweb_edu_longdoc_filtered_sample.yaml`

```yaml
name: fineweb_edu_longdoc_filtered_sample
tokenizer_output_dir: artifacts/tokenizer/fineweb_edu_longdoc
datasets:
  - name: fineweb_edu_longdoc
    dataset: text
    split: train
    text_column: text
    data_files: data/filtered/fineweb_edu_longdoc_en_sample.txt
    sample_limit: 5000
    seq_len: 4096
    sequences_per_shard: 1024
    output_dir: data/shards/fineweb_edu_longdoc_sample
    max_records: null

```

### File: `configs/data/fineweb_edu_mixture_full.yaml`

```yaml
name: fineweb_edu_full
tokenizer_output_dir: artifacts/tokenizer/fineweb_edu
datasets:
  - name: fineweb_edu
    dataset: HuggingFaceFW/fineweb-edu
    subset: sample-100BT
    split: train
    text_column: text
    sample_limit: 100000
    seq_len: 4096
    sequences_per_shard: 1024
    output_dir: data/shards/fineweb_edu_full
    max_records: null

```

### File: `configs/data/fineweb_edu_mixture_sample.yaml`

```yaml
name: fineweb_edu_sample
tokenizer_output_dir: artifacts/tokenizer/fineweb_edu
datasets:
  - name: fineweb_edu
    dataset: HuggingFaceFW/fineweb-edu
    subset: sample-10BT
    split: train
    text_column: text
    sample_limit: 5000
    seq_len: 2048
    sequences_per_shard: 1024
    output_dir: data/shards/fineweb_edu_sample
    max_records: 10000

```

### File: `configs/data/refinedweb_mixture.yaml`

```yaml
name: refinedweb_mix_v1
tokenizer_output_dir: artifacts/tokenizer/refinedweb_mix
datasets:
  - name: refinedweb
    dataset: text
    split: train
    text_column: text
    data_files: data/filtered/refinedweb_en_full.txt
    seq_len: 2048
    sequences_per_shard: 2048
    output_dir: data/shards/refinedweb
    max_records: null
  - name: books
    dataset: text
    split: train
    text_column: text
    data_files: data/filtered/wikipedia_en_full.txt
    seq_len: 2048
    sequences_per_shard: 2048
    output_dir: data/shards/wikipedia
    max_records: null
  - name: c4
    dataset: text
    split: train
    text_column: text
    data_files: data/filtered/c4_en_full.txt
    seq_len: 2048
    sequences_per_shard: 2048
    output_dir: data/shards/c4
    max_records: null
  - name: redpajama
    dataset: text
    split: train
    text_column: text
    data_files: data/filtered/redpajama_en_full.txt
    seq_len: 2048
    sequences_per_shard: 2048
    output_dir: data/shards/redpajama
    max_records: null
  - name: code
    dataset: text
    split: train
    text_column: text
    data_files: data/filtered/code_en_full.txt
    seq_len: 2048
    sequences_per_shard: 2048
    output_dir: data/shards/code
    max_records: null
```

### File: `configs/data/refinedweb_mixture_filtered.yaml`

```yaml
name: refinedweb_mix_filtered
tokenizer_output_dir: artifacts/tokenizer/refinedweb_mix
datasets:
  - name: refinedweb
    dataset: text
    split: train
    text_column: text
    data_files: data/filtered/refinedweb_en_sample.txt
    seq_len: 512
    sequences_per_shard: 512
    output_dir: data/shards/refinedweb_filtered
    max_records: null
  - name: wikipedia
    dataset: text
    split: train
    text_column: text
    data_files: data/filtered/wikipedia_en_sample.txt
    seq_len: 512
    sequences_per_shard: 512
    output_dir: data/shards/wikipedia_filtered
    max_records: null
  - name: c4
    dataset: text
    split: train
    text_column: text
    data_files: data/filtered/c4_en_sample.txt
    seq_len: 512
    sequences_per_shard: 512
    output_dir: data/shards/c4_filtered
    max_records: null
  - name: redpajama
    dataset: text
    split: train
    text_column: text
    data_files: data/filtered/redpajama_en_sample.txt
    seq_len: 512
    sequences_per_shard: 512
    output_dir: data/shards/redpajama_filtered
    max_records: null
  - name: code
    dataset: text
    split: train
    text_column: text
    data_files: data/filtered/code_en_sample.txt
    seq_len: 512
    sequences_per_shard: 512
    output_dir: data/shards/code_filtered
    max_records: null
```

### File: `configs/data/refinedweb_mixture_full.yaml`

```yaml
name: refinedweb_mix_full
tokenizer_output_dir: artifacts/tokenizer/refinedweb_mix
datasets:
  - name: refinedweb
    dataset: text
    split: train
    text_column: text
    data_files: data/filtered/refinedweb_en_full.txt
    seq_len: 2048
    sequences_per_shard: 1024
    output_dir: data/shards/refinedweb_full
    max_records: null
  - name: wikipedia
    dataset: text
    split: train
    text_column: text
    data_files: data/filtered/wikipedia_en_full.txt
    seq_len: 2048
    sequences_per_shard: 1024
    output_dir: data/shards/wikipedia_full
    max_records: null
  - name: c4
    dataset: text
    split: train
    text_column: text
    data_files: data/filtered/c4_en_full.txt
    seq_len: 2048
    sequences_per_shard: 1024
    output_dir: data/shards/c4_full
    max_records: null
  - name: redpajama
    dataset: text
    split: train
    text_column: text
    data_files: data/filtered/redpajama_en_full.txt
    seq_len: 2048
    sequences_per_shard: 1024
    output_dir: data/shards/redpajama_full
    max_records: null
  - name: code
    dataset: text
    split: train
    text_column: text
    data_files: data/filtered/code_en_full.txt
    seq_len: 2048
    sequences_per_shard: 1024
    output_dir: data/shards/code_full
    max_records: null
```

### File: `configs/data/refinedweb_mixture_sample.yaml`

```yaml
name: refinedweb_mix_sample
tokenizer_output_dir: artifacts/tokenizer/refinedweb_mix
datasets:
  - name: refinedweb
    dataset: HuggingFaceFW/fineweb
    subset: sample-10BT
    split: train
    text_column: text
    sample_limit: 5000
    seq_len: 512
    sequences_per_shard: 512
    output_dir: data/shards/refinedweb_sample
    max_records: 10000
  - name: books
    dataset: wikimedia/wikipedia
    subset: 20231101.en
    split: train
    text_column: text
    sample_limit: 2000
    seq_len: 512
    sequences_per_shard: 512
    output_dir: data/shards/wikipedia_sample
    max_records: 5000
  - name: c4
    dataset: allenai/c4
    subset: en
    split: train
    text_column: text
    sample_limit: 2000
    seq_len: 512
    sequences_per_shard: 512
    output_dir: data/shards/c4_sample
    max_records: 4000
  - name: redpajama
    dataset: cerebras/SlimPajama-627B
    split: train
    text_column: text
    sample_limit: 2000
    seq_len: 512
    sequences_per_shard: 512
    output_dir: data/shards/redpajama_sample
    max_records: 4000
  - name: code
    dataset: codeparrot/codeparrot-clean-train
    split: train
    text_column: content
    sample_limit: 2000
    seq_len: 512
    sequences_per_shard: 512
    output_dir: data/shards/code_sample
    max_records: 4000
```

### File: `configs/hope/mid.yaml`

```yaml
defaults:
  - _self_

hydra:
  run:
    dir: .
  output_subdir: null
  job:
    chdir: false

model:
  vocab_size: 32000
  dim: 1024
  num_layers: 24
  heads: 16
  surprise_threshold: null
  freeze_backbone: false
  titan_level:
    name: titan
    update_period: 16
    optimizer_key: titan_opt
  cms_levels:
    - name: cms_fast
      update_period: 1
      optimizer_key: cms_opt
    - name: cms_mid
      update_period: 4
      optimizer_key: cms_opt
    - name: cms_slow
      update_period: 32
      optimizer_key: cms_opt
    - name: cms_ultra
      update_period: 128
      optimizer_key: cms_opt
  optimizers:
    titan_opt:
      type: deep_momentum
      lr: 8.0e-4
      params:
        beta: 0.9
        beta2: 0.999
        variant: nl_l2_precond
    cms_opt:
      type: deep_momentum
      lr: 4.0e-4
      params:
        beta: 0.9
        beta2: 0.999
        variant: nl_l2_precond

data:
  source: mixture
  batch_size: 16
  num_workers: 4
  mixture:
    samples_per_epoch: 8192
    seed: 42
    sources:
      - name: refinedweb
        shards_dir: data/shards/refinedweb_full
        weight: 0.4
      - name: wikipedia
        shards_dir: data/shards/wikipedia_full
        weight: 0.2
      - name: c4
        shards_dir: data/shards/c4_full
        weight: 0.15
      - name: redpajama
        shards_dir: data/shards/redpajama_full
        weight: 0.15
      - name: code
        shards_dir: data/shards/code_full
        weight: 0.1

train:
  online_updates: true
  online_chunk_size: 0
  per_layer_teach_signal: true
  steps: 100
  log_interval: 10
  device: "cuda:1"
  seed: 808
  deterministic: false
  step_offset: 0
  mixed_precision:
    enabled: true
    dtype: bf16
  compile:
    enable: true
    mode: max-autotune
  fsdp:
    auto_wrap_min_params: 2000000
    cpu_offload: false
  checkpoint:
    enable: true
    dir: checkpoints/mid
    save_interval: 50
    resume_path: null
    resume_tag: null

optim:
  type: muon
  lr: 2.0e-4
  weight_decay: 0.02
  momentum: 0.95
  betas:
    - 0.9
    - 0.999

logging:
  enabled: false
  backend: wandb
  project: nested-learning
  run_name: mid-${now:%Y%m%d%H%M%S}
  path: logs/mid_metrics.json
```

### File: `configs/hope/mid_fsdp.yaml`

```yaml
defaults:
  - mid
  - _self_

model:
  gradient_checkpointing: true

data:
  batch_size: 8  # per-rank micro-batch for 2× RTX 6000 Ada
  num_workers: 6

train:
  online_updates: true
  online_chunk_size: 0
  per_layer_teach_signal: true
  steps: 250000
  log_interval: 20
  device: "cuda"
  mixed_precision:
    enabled: true
    dtype: bf16
  compile:
    enable: false
  fsdp:
    auto_wrap_min_params: 2000000
    cpu_offload: false
  checkpoint:
    enable: true
    dir: artifacts/checkpoints/mid_fsdp
    save_interval: 1000
    resume_path: null
    resume_tag: null

optim:
  type: muon
  lr: 2.0e-4
  weight_decay: 0.01

logging:
  enabled: true
  backend: wandb
  project: nested-learning
  run_name: hope-mid-fsdp-${now:%Y%m%d%H%M%S}
  path: logs/mid_fsdp_metrics.json
```

### File: `configs/hope/pilot.yaml`

```yaml
defaults:
  - /pilot
```

### File: `configs/hope/pilot_attention.yaml`

```yaml
defaults:
  - /pilot
  - _self_

model:
  block_variant: hope_attention
  qk_l2_norm: true
  local_conv_window: 4

```

### File: `configs/hope/pilot_selfmod.yaml`

```yaml
defaults:
  - /pilot
  - _self_

model:
  block_variant: hope_selfmod
  # Chunk update cadence (paper §8.2): other memories update more often than M_memory.
  self_mod_chunk_size: 8
  self_mod_chunk_size_memory: 64

train:
  online_updates: true
  online_chunk_size: 0
  per_layer_teach_signal: true
  checkpoint:
    dir: artifacts/checkpoints/pilot_selfmod

logging:
  run_name: pilot-selfmod
  path: logs/pilot_selfmod_metrics.json
```

### File: `configs/hope/pilot_transformer.yaml`

```yaml
defaults:
  - /pilot
  - _self_

model:
  block_variant: transformer
  qk_l2_norm: true
  local_conv_window: 4

```

### File: `configs/hope/target.yaml`

```yaml
defaults:
  - _self_

hydra:
  run:
    dir: .
  output_subdir: null
  job:
    chdir: false

model:
  vocab_size: 32000
  dim: 1536
  num_layers: 32
  heads: 24
  surprise_threshold: null
  freeze_backbone: false
  titan_level:
    name: titan
    update_period: 32
    optimizer_key: titan_opt
  cms_levels:
    - name: cms_fast
      update_period: 1
      optimizer_key: cms_fast_opt
    - name: cms_mid
      update_period: 4
      optimizer_key: cms_mid_opt
    - name: cms_slow
      update_period: 32
      optimizer_key: cms_slow_opt
    - name: cms_ultra
      update_period: 128
      optimizer_key: cms_slow_opt
    - name: cms_anchor
      update_period: 512
      optimizer_key: cms_anchor_opt
  optimizers:
    titan_opt:
      type: deep_momentum
      lr: 6.0e-4
      params:
        beta: 0.9
        beta2: 0.999
        variant: nl_l2_precond
    cms_fast_opt:
      type: deep_momentum
      lr: 3.0e-4
      params:
        beta: 0.9
        beta2: 0.999
        variant: nl_l2_precond
    cms_mid_opt:
      type: deep_momentum
      lr: 2.5e-4
      params:
        beta: 0.9
        beta2: 0.999
        variant: nl_l2_precond
    cms_slow_opt:
      type: deep_momentum
      lr: 2.0e-4
      params:
        beta: 0.9
        beta2: 0.999
        variant: nl_l2_precond
    cms_anchor_opt:
      type: deep_momentum
      lr: 1.5e-4
      params:
        beta: 0.9
        beta2: 0.999
        variant: nl_l2_precond

data:
  source: mixture
  batch_size: 32
  num_workers: 8
  mixture:
    samples_per_epoch: 32768
    seed: 123
    sources:
      - name: refinedweb
        shards_dir: data/shards/refinedweb_filtered
        weight: 0.35
      - name: wikipedia
        shards_dir: data/shards/wikipedia_filtered
        weight: 0.2
      - name: c4
        shards_dir: data/shards/c4_filtered
        weight: 0.15
      - name: redpajama
        shards_dir: data/shards/redpajama_filtered
        weight: 0.2
      - name: code
        shards_dir: data/shards/code_filtered
        weight: 0.1

train:
  online_updates: true
  online_chunk_size: 0
  per_layer_teach_signal: true
  steps: 200
  log_interval: 10
  device: "cuda:1"
  seed: 9001
  deterministic: false
  step_offset: 0
  mixed_precision:
    enabled: true
    dtype: bf16
  compile:
    enable: true
    mode: max-autotune
  fsdp:
    auto_wrap_min_params: 2000000
    cpu_offload: false
  checkpoint:
    enable: true
    dir: checkpoints/target
    save_interval: 100
    resume_path: null
    resume_tag: null

optim:
  type: muon
  lr: 1.5e-4
  weight_decay: 0.02
  momentum: 0.95
  betas:
    - 0.9
    - 0.999

logging:
  enabled: false
  backend: wandb
  project: nested-learning
  run_name: target-${now:%Y%m%d%H%M%S}
  path: logs/target_metrics.json

deepspeed:
  config: configs/deepspeed/zero3.json
```

### File: `configs/hope/target_fsdp.yaml`

```yaml
defaults:
  - target
  - _self_

model:
  gradient_checkpointing: true

data:
  batch_size: 4  # per-rank micro-batch
  num_workers: 8

train:
  online_updates: true
  online_chunk_size: 0
  per_layer_teach_signal: true
  steps: 300000
  log_interval: 20
  device: "cuda"
  mixed_precision:
    enabled: true
    dtype: bf16
  compile:
    enable: false
  fsdp:
    auto_wrap_min_params: 2500000
    cpu_offload: false
  checkpoint:
    enable: true
    dir: artifacts/checkpoints/target_fsdp
    save_interval: 1000
    resume_path: null
    resume_tag: null

optim:
  type: muon
  lr: 1.5e-4
  weight_decay: 0.01

logging:
  enabled: true
  backend: wandb
  project: nested-learning
  run_name: hope-target-fsdp-${now:%Y%m%d%H%M%S}
  path: logs/target_fsdp_metrics.json
```

### File: `configs/mid_smoke.yaml`

```yaml
hydra:
  run:
    dir: .
  output_subdir: null
  job:
    chdir: false

model:
  vocab_size: 32000
  dim: 256
  num_layers: 4
  heads: 8
  titan_level:
    name: titan
    update_period: 16
    optimizer_key: titan_opt
  cms_levels:
    - name: cms_fast
      update_period: 1
      optimizer_key: cms_opt
    - name: cms_mid
      update_period: 4
      optimizer_key: cms_opt
    - name: cms_slow
      update_period: 16
      optimizer_key: cms_opt
    - name: cms_ultra
      update_period: 64
      optimizer_key: cms_opt
  optimizers:
    titan_opt:
      type: deep_momentum
      lr: 8.0e-4
      params:
        beta: 0.9
        beta2: 0.999
    cms_opt:
      type: deep_momentum
      lr: 4.0e-4
      params:
        beta: 0.9
        beta2: 0.999

data:
  source: mixture
  batch_size: 4
  num_workers: 0
  mixture:
    samples_per_epoch: 128
    seed: 0
    sources:
      - name: refinedweb
        shards_dir: data/shards/refinedweb_filtered
        weight: 0.4
      - name: wikipedia
        shards_dir: data/shards/wikipedia_filtered
        weight: 0.2
      - name: c4
        shards_dir: data/shards/c4_filtered
        weight: 0.15
      - name: redpajama
        shards_dir: data/shards/redpajama_filtered
        weight: 0.15
      - name: code
        shards_dir: data/shards/code_filtered
        weight: 0.1

train:
  online_updates: true
  online_chunk_size: 0
  per_layer_teach_signal: true
  steps: 10
  log_interval: 1
  device: "cpu"
  seed: 2024
  deterministic: true
  mixed_precision:
    enabled: false
    dtype: bf16
  compile:
    enable: false
  checkpoint:
    enable: true
    dir: artifacts/checkpoints/mid_smoke
    save_interval: 10
    save_last: true

optim:
  type: adamw
  lr: 2.0e-4
  fused: false

logging:
  enabled: true
  backend: json
  path: logs/mid_smoke.json
```

### File: `configs/mid_stage2.yaml`

```yaml
hydra:
  run:
    dir: .
  output_subdir: null
  job:
    chdir: false

model:
  vocab_size: 32000
  dim: 768
  num_layers: 18
  heads: 12
  teach_scale: 0.05
  teach_clip: 5.0
  teach_schedule:
    warmup_steps: 20
    decay_start: 80
    decay_duration: 40
  titan_level:
    name: titan
    update_period: 16
    optimizer_key: titan_opt
  cms_levels:
    - name: cms_fast
      update_period: 1
      optimizer_key: cms_opt
    - name: cms_mid
      update_period: 4
      optimizer_key: cms_opt
    - name: cms_slow
      update_period: 32
      optimizer_key: cms_opt
    - name: cms_ultra
      update_period: 128
      optimizer_key: cms_opt
  optimizers:
    titan_opt:
      type: deep_momentum
      lr: 8.0e-4
      params:
        beta: 0.9
        beta2: 0.999
    cms_opt:
      type: deep_momentum
      lr: 4.0e-4
      params:
        beta: 0.9
        beta2: 0.999

data:
  source: mixture
  batch_size: 8
  num_workers: 2
  mixture:
    samples_per_epoch: 1024
    seed: 42
    sources:
      - name: refinedweb
        shards_dir: data/shards/refinedweb_full
        weight: 0.4
      - name: wikipedia
        shards_dir: data/shards/wikipedia_full
        weight: 0.2
      - name: c4
        shards_dir: data/shards/c4_full
        weight: 0.15
      - name: redpajama
        shards_dir: data/shards/redpajama_full
        weight: 0.15
      - name: code
        shards_dir: data/shards/code_full
        weight: 0.1

train:
  online_updates: true
  online_chunk_size: 0
  per_layer_teach_signal: true
  steps: 100
  log_interval: 10
  device: "cuda"
  seed: 3401
  deterministic: false
  mixed_precision:
    enabled: true
    dtype: bf16
  compile:
    enable: true
    mode: max-autotune
  fsdp:
    auto_wrap_min_params: 2000000
    cpu_offload: false
  checkpoint:
    enable: true
    dir: artifacts/checkpoints/mid_stage2
    save_interval: 100
    resume_path: null
    resume_tag: null

optim:
  type: adamw
  lr: 3.0e-5
  fused: auto

logging:
  enabled: true
  backend: json
  path: logs/mid_stage2.json
```

### File: `configs/mid_stage2_smoke.yaml`

```yaml
hydra:
  run:
    dir: .
  output_subdir: null
  job:
    chdir: false

model:
  vocab_size: 32000
  dim: 512
  num_layers: 12
  heads: 8
  teach_scale: 0.2
  teach_clip: 2.0
  titan_level:
    name: titan
    update_period: 16
    optimizer_key: titan_opt
  cms_levels:
    - name: cms_fast
      update_period: 1
      optimizer_key: cms_opt
    - name: cms_mid
      update_period: 4
      optimizer_key: cms_opt
    - name: cms_slow
      update_period: 16
      optimizer_key: cms_opt
  optimizers:
    titan_opt:
      type: deep_momentum
      lr: 6.0e-4
      params:
        beta: 0.9
        beta2: 0.999
    cms_opt:
      type: deep_momentum
      lr: 3.0e-4
      params:
        beta: 0.9
        beta2: 0.999

data:
  source: mixture
  batch_size: 8
  num_workers: 2
  mixture:
    samples_per_epoch: 512
    seed: 0
    sources:
      - name: refinedweb
        shards_dir: data/shards/refinedweb_filtered
        weight: 0.4
      - name: wikipedia
        shards_dir: data/shards/wikipedia_filtered
        weight: 0.2
      - name: c4
        shards_dir: data/shards/c4_filtered
        weight: 0.15
      - name: redpajama
        shards_dir: data/shards/redpajama_filtered
        weight: 0.15
      - name: code
        shards_dir: data/shards/code_filtered
        weight: 0.1

train:
  online_updates: true
  online_chunk_size: 0
  per_layer_teach_signal: true
  steps: 60
  log_interval: 5
  device: "cuda"
  seed: 777
  deterministic: false
  mixed_precision:
    enabled: true
    dtype: bf16
  compile:
    enable: false
  fsdp:
    auto_wrap_min_params: 1500000
    cpu_offload: false
  checkpoint:
    enable: true
    dir: artifacts/checkpoints/mid_stage2_smoke
    save_interval: 60
    resume_path: null
    resume_tag: null

optim:
  type: adamw
  lr: 1.0e-4
  fused: auto

logging:
  enabled: true
  backend: json
  path: logs/mid_stage2_smoke.json
```

### File: `configs/mid_titan_baseline.yaml`

```yaml
hydra:
  run:
    dir: .
  output_subdir: null
  job:
    chdir: false

model:
  type: titan
  vocab_size: 32000
  dim: 768
  num_layers: 18
  heads: 12
  surprise_threshold: 0.02
  freeze_backbone: false
  titan_level:
    name: titan
    update_period: 16
    optimizer_key: titan_opt
  optimizers:
    titan_opt:
      type: deep_momentum
      lr: 8.0e-4
      params:
        beta: 0.9
        beta2: 0.999
  teach_scale: 0.10
  teach_clip: 4.0
  teach_schedule:
    warmup_steps: 60
    decay_start: 140
    decay_duration: 80

data:
  source: mixture
  batch_size: 4
  num_workers: 2
  mixture:
    samples_per_epoch: 1024
    seed: 42
    sources:
      - name: refinedweb
        shards_dir: data/shards/refinedweb_full
        weight: 0.4
      - name: wikipedia
        shards_dir: data/shards/wikipedia_full
        weight: 0.2
      - name: c4
        shards_dir: data/shards/c4_full
        weight: 0.15
      - name: redpajama
        shards_dir: data/shards/redpajama_full
        weight: 0.15
      - name: code
        shards_dir: data/shards/code_full
        weight: 0.1

train:
  online_updates: true
  online_chunk_size: 0
  per_layer_teach_signal: true
  steps: 220
  log_interval: 20
  device: "cuda:1"
  seed: 451
  deterministic: false
  step_offset: 0
  mixed_precision:
    enabled: true
    dtype: bf16
  compile:
    enable: false
  checkpoint:
    enable: true
    dir: artifacts/checkpoints/mid_titan_baseline
    save_interval: 100
    resume_path: null
    resume_tag: null

optim:
  type: adamw
  lr: 1.0e-5
  fused: auto

logging:
  enabled: true
  backend: json
  path: logs/mid_titan_baseline.json
  run_name: mid_titan_baseline
```

### File: `configs/pilot.yaml`

```yaml
defaults:
  - _self_

hydra:
  run:
    dir: .
  output_subdir: null
  job:
    chdir: false

model:
  vocab_size: 32000
  dim: 512
  num_layers: 12
  heads: 8
  teach_scale: 0.10
  teach_clip: 5.0
  surprise_threshold: 0.02
  freeze_backbone: false
  self_mod_lr: 0.001
  teach_schedule:
    warmup_steps: 2000
    decay_start: 120000
    decay_duration: 20000
  titan_level:
    name: titan
    update_period: 8
    optimizer_key: titan_opt
  cms_levels:
    - name: cms_fast
      update_period: 1
      optimizer_key: cms_opt
    - name: cms_mid
      update_period: 4
      optimizer_key: cms_opt
    - name: cms_slow
      update_period: 32
      optimizer_key: cms_opt
    - name: cms_ultra
      update_period: 128
      optimizer_key: cms_opt
  optimizers:
    titan_opt:
      type: deep_momentum
      lr: 6.0e-4
      params:
        beta: 0.9
        beta2: 0.999
        variant: nl_l2_precond
    cms_opt:
      type: deep_momentum
      lr: 3.0e-4
      params:
        beta: 0.9
        beta2: 0.999
        variant: nl_l2_precond

data:
  source: mixture
  seq_len: 2048
  batch_size: 6
  num_workers: 4
  mixture:
    samples_per_epoch: 65536
    seed: 1337
    sources:
      - name: refinedweb
        shards_dir: data/shards/refinedweb_filtered
        weight: 0.4
      - name: wikipedia
        shards_dir: data/shards/wikipedia_filtered
        weight: 0.2
      - name: c4
        shards_dir: data/shards/c4_filtered
        weight: 0.15
      - name: redpajama
        shards_dir: data/shards/redpajama_filtered
        weight: 0.15
      - name: code
        shards_dir: data/shards/code_filtered
        weight: 0.1

train:
  online_updates: true
  online_chunk_size: 0
  per_layer_teach_signal: true
  steps: 246667
  log_interval: 50
  device: "cuda:1"
  seed: 1337
  deterministic: false
  step_offset: 0
  mixed_precision:
    enabled: true
    dtype: bf16
  compile:
    enable: false
    mode: max-autotune
  checkpoint:
    enable: true
    dir: artifacts/checkpoints/pilot
    save_interval: 1000
    save_last: true
    resume_path: null
    resume_tag: null

optim:
  type: muon
  lr: 2.5e-4
  weight_decay: 0.02
  momentum: 0.95
  betas:
    - 0.9
    - 0.999

logging:
  enabled: true
  backend: json
  path: logs/pilot_metrics.json
  project: nested-learning
  run_name: pilot-main
```

### File: `configs/pilot_paper_faithful.yaml`

```yaml
defaults:
  - /pilot
  - _self_

model:
  # Paper-faithful: treat "surprise" as the (scaled) teach signal itself, without threshold gating.
  surprise_threshold: null
  # Paper updates on the last (possibly partial) chunk; enable flush for non-multiple seq lengths.
  cms_flush_partial_at_end: true
  # Paper: q is non-adaptive and uses a fixed projection.
  self_mod_adaptive_q: false
  # Paper: local causal conv in the HOPE self-mod module.
  self_mod_local_conv_window: 4

data:
  # Paper-faithful semantics: CMS/TITAN fast state is per-context; this repo currently treats
  # each *batch* as a single shared context when batch_size>1.
  batch_size: 1

train:
  # Paper: re-initialize fast memories per context (sequence).
  use_fast_state: true
  # Fail fast if DDP would silently disable paper-critical features.
  fail_if_paper_faithful_disabled: true

optim:
  # Ensure meta-learning updates include memory module initial states (paper §8.2).
  param_policy: all

logging:
  run_name: pilot-paper-faithful
  path: logs/pilot_paper_faithful_metrics.json
```

### File: `configs/pilot_selfmod_paper_faithful.yaml`

```yaml
defaults:
  - /pilot_paper_faithful
  - _self_

model:
  block_variant: hope_selfmod
  # Chunk update cadence (paper §8.2): other memories update more often than M_memory.
  self_mod_chunk_size: 8
  self_mod_chunk_size_memory: 64
  self_mod_use_skip: false

train:
  checkpoint:
    dir: artifacts/checkpoints/pilot_selfmod_paper_faithful

logging:
  run_name: pilot-selfmod-paper-faithful
  path: logs/pilot_selfmod_paper_faithful_metrics.json
```

### File: `configs/pilot_smoke.yaml`

```yaml
hydra:
  run:
    dir: .
  output_subdir: null
  job:
    chdir: false

model:
  vocab_size: 32000
  dim: 128
  num_layers: 2
  heads: 4
  titan_level:
    name: titan
    update_period: 8
    optimizer_key: titan_opt
  cms_levels:
    - name: cms_fast
      update_period: 1
      optimizer_key: cms_opt
    - name: cms_mid
      update_period: 4
      optimizer_key: cms_opt
    - name: cms_slow
      update_period: 16
      optimizer_key: cms_opt
  optimizers:
    titan_opt:
      type: deep_momentum
      lr: 1.0e-3
      params:
        beta: 0.9
        beta2: 0.999
    cms_opt:
      type: deep_momentum
      lr: 5.0e-4
      params:
        beta: 0.9
        beta2: 0.999

data:
  source: synthetic
  vocab_size: 32000
  seq_len: 64
  dataset_size: 1024
  batch_size: 4
  num_workers: 0

train:
  online_updates: true
  online_chunk_size: 0
  per_layer_teach_signal: true
  steps: 10
  log_interval: 1
  device: "cpu"
  seed: 1234
  deterministic: true
  mixed_precision:
    enabled: false
    dtype: bf16
  compile:
    enable: false
  checkpoint:
    enable: true
    dir: artifacts/checkpoints/pilot_smoke
    save_interval: 10
    save_last: true

optim:
  type: adamw
  lr: 3.0e-4
  fused: false

logging:
  enabled: true
  backend: json
  path: logs/pilot_smoke.json
```

### File: `configs/resolved/cms_sparse_eval.yaml`

```yaml
hydra:
  run:
    dir: .
  output_subdir: null
  job:
    chdir: false
model:
  vocab_size: 32000
  dim: 384
  num_layers: 8
  heads: 6
  teach_scale: 0.1
  teach_clip: 5.0
  self_mod_lr: 0.001
  teach_schedule:
    warmup_steps: 2000
    decay_start: 120000
    decay_duration: 20000
  titan_level:
    name: titan
    update_period: 8
    optimizer_key: titan_opt
  cms_levels:
  - name: cms_fast
    update_period: 8
    optimizer_key: cms_opt
  - name: cms_mid
    update_period: 32
    optimizer_key: cms_opt
  - name: cms_slow
    update_period: 128
    optimizer_key: cms_opt
  - name: cms_ultra
    update_period: 512
    optimizer_key: cms_opt
  optimizers:
    titan_opt:
      type: deep_momentum
      lr: 0.0006
      params:
        beta: 0.9
        beta2: 0.999
    cms_opt:
      type: deep_momentum
      lr: 0.0003
      params:
        beta: 0.9
        beta2: 0.999
  cms_hidden_multiplier: 2
data:
  source: mixture
  seq_len: 1024
  batch_size: 2
  num_workers: 2
  mixture:
    samples_per_epoch: 65536
    seed: 1337
    sources:
    - name: refinedweb
      shards_dir: data/shards/refinedweb_filtered
      weight: 0.4
    - name: wikipedia
      shards_dir: data/shards/wikipedia_filtered
      weight: 0.2
    - name: c4
      shards_dir: data/shards/c4_filtered
      weight: 0.15
    - name: redpajama
      shards_dir: data/shards/redpajama_filtered
      weight: 0.15
    - name: code
      shards_dir: data/shards/code_filtered
      weight: 0.1
train:
  online_updates: true
  online_chunk_size: 0
  per_layer_teach_signal: true
  steps: 5000
  log_interval: 25
  device: cuda:1
  seed: 1337
  deterministic: false
  mixed_precision:
    enabled: true
    dtype: bf16
  compile:
    enable: false
    mode: max-autotune
  checkpoint:
    enable: true
    dir: artifacts/checkpoints/pilot_cms_sparse
    save_interval: 1000
    save_last: true
    resume_path: null
    resume_tag: null
optim:
  type: adamw
  lr: 0.00025
  fused: auto
logging:
  enabled: true
  backend: json
  path: logs/pilot_cms_sparse_metrics.json
  project: nested-learning
  run_name: pilot-cms-sparse
```

### File: `configs/resolved/phase2_pilot_attention_eval.yaml`

```yaml
model:
  vocab_size: 32000
  dim: 512
  num_layers: 12
  heads: 8
  teach_scale: 0.10
  teach_clip: 5.0
  surprise_threshold: 0.02
  freeze_backbone: false
  qk_l2_norm: true
  local_conv_window: 4
  block_variant: hope_attention
  teach_schedule:
    warmup_steps: 2000
    decay_start: 120000
    decay_duration: 20000
  titan_level:
    name: titan
    update_period: 8
    optimizer_key: titan_opt
  cms_levels:
    - name: cms_fast
      update_period: 1
      optimizer_key: cms_opt
    - name: cms_mid
      update_period: 4
      optimizer_key: cms_opt
    - name: cms_slow
      update_period: 32
      optimizer_key: cms_opt
    - name: cms_ultra
      update_period: 128
      optimizer_key: cms_opt
  optimizers:
    titan_opt:
      type: deep_momentum
      lr: 6.0e-4
      params:
        beta: 0.9
        beta2: 0.999
        variant: nl_l2_precond
    cms_opt:
      type: deep_momentum
      lr: 3.0e-4
      params:
        beta: 0.9
        beta2: 0.999
        variant: nl_l2_precond

```

### File: `configs/resolved/phase2_pilot_transformer_eval.yaml`

```yaml
model:
  vocab_size: 32000
  dim: 512
  num_layers: 12
  heads: 8
  teach_scale: 0.10
  teach_clip: 5.0
  surprise_threshold: 0.02
  freeze_backbone: false
  qk_l2_norm: true
  local_conv_window: 4
  block_variant: transformer
  teach_schedule:
    warmup_steps: 2000
    decay_start: 120000
    decay_duration: 20000
  titan_level:
    name: titan
    update_period: 8
    optimizer_key: titan_opt
  cms_levels:
    - name: cms_fast
      update_period: 1
      optimizer_key: cms_opt
    - name: cms_mid
      update_period: 4
      optimizer_key: cms_opt
    - name: cms_slow
      update_period: 32
      optimizer_key: cms_opt
    - name: cms_ultra
      update_period: 128
      optimizer_key: cms_opt
  optimizers:
    titan_opt:
      type: deep_momentum
      lr: 6.0e-4
      params:
        beta: 0.9
        beta2: 0.999
        variant: nl_l2_precond
    cms_opt:
      type: deep_momentum
      lr: 3.0e-4
      params:
        beta: 0.9
        beta2: 0.999
        variant: nl_l2_precond

```

### File: `pyproject.toml`

```toml
[project]
name = "nested-learning"
version = "0.1.0"
description = "Reproduction of Google's Nested Learning (HOPE) architecture"
license = {text = "Apache-2.0"}
authors = [
  {name = "Nested Learning Team", email = "nested-learning@example.com"}
]
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
  "torch==2.9.0",
  "torchvision==0.24.0",
  "torchaudio==2.9.0",
  "einops>=0.7.0",
  "numpy>=1.26",
  "hydra-core>=1.3.2",
  "omegaconf>=2.3.0",
  "pyyaml>=6.0",
  "tqdm>=4.66",
  "typing-extensions>=4.9",
  "datasets>=2.19,<3.0",
  "sentencepiece>=0.2.0",
  "huggingface-hub>=0.23,<1.0",
  "zstandard>=0.22.0",
  "wandb>=0.18.0",
  "langdetect>=1.0.9",
  "typer>=0.12",
  "matplotlib>=3.8",
]

[project.optional-dependencies]
dev = [
  "pytest>=7.4",
  "pytest-cov>=4.1",
  "ruff>=0.6.8",
  "mypy>=1.11",
  "types-PyYAML",
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
minversion = "7.0"
addopts = "-ra -q"
testpaths = ["tests"]

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I"]
ignore = []

[tool.mypy]
python_version = 3.12
warn_unused_configs = true
# PyTorch-style codebases inevitably interact with untyped third-party deps and dynamic module calls.
# Keep mypy enabled, but avoid noisy `Any` return warnings for nn.Module forward calls.
ignore_missing_imports = true
warn_return_any = false
strict_optional = true
show_error_codes = true
pretty = true
packages = ["nested_learning"]
```

### File: `scripts/__init__.py`

```python
# Makes `scripts` a package for intra-eval imports.
```

### File: `scripts/checkpoint/verify.py`

```python
#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path

import typer

from nested_learning.training import verify_checkpoint_integrity

app = typer.Typer(help="Verify checkpoint metadata hashes, config, and RNG sidecars.")


@app.command()
def main(
    checkpoint: Path = typer.Option(..., help="Path to checkpoint .pt file."),
) -> None:
    metadata = verify_checkpoint_integrity(checkpoint)
    typer.echo(f"[verify] {checkpoint} OK (step {metadata.get('step')})")
    typer.echo(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    app()
```

### File: `scripts/checks/tokenizer_coverage_guard.py`

```python
#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from nested_learning.tokenizer_coverage import compute_tokenizer_coverage_stats

app = typer.Typer(
    add_completion=False,
    help="Regress coverage stats against a recorded baseline to catch tokenizer drift.",
)


@app.command()
def main(
    baseline: Path = typer.Option(
        ...,
        help="Reference JSON produced by scripts/data/check_tokenizer_coverage.py.",
    ),
    tokenizer_path: Path = typer.Option(..., help="SentencePiece tokenizer to evaluate."),
    sample_file: Path = typer.Option(..., help="Representative text sample."),
    max_lines: int = typer.Option(10_000, help="Maximum lines to consume."),
    avg_tokens_tolerance: float = typer.Option(
        0.05,
        help="Allowed increase in avg tokens per word before failing.",
    ),
    single_token_drop_tolerance: float = typer.Option(
        0.02,
        help="Allowed decrease in pct_single_token_words before failing.",
    ),
    two_token_drop_tolerance: float = typer.Option(
        0.02,
        help="Allowed decrease in pct_two_or_less_tokens_words before failing.",
    ),
    output: Optional[Path] = typer.Option(
        None,
        help="Optional path to write the freshly computed coverage JSON.",
    ),
) -> None:
    if not baseline.exists():
        raise typer.BadParameter(f"Baseline JSON {baseline} was not found.")
    baseline_stats = json.loads(baseline.read_text())
    current_stats = compute_tokenizer_coverage_stats(
        tokenizer_path, sample_file, max_lines=max_lines
    )
    violations: list[str] = []

    delta_avg = current_stats["avg_tokens_per_word"] - baseline_stats["avg_tokens_per_word"]
    if delta_avg > avg_tokens_tolerance:
        violations.append(
            f"avg_tokens_per_word regressed by {delta_avg:.4f} (limit {avg_tokens_tolerance:.4f})."
        )

    delta_single = (
        baseline_stats["pct_single_token_words"] - current_stats["pct_single_token_words"]
    )
    if delta_single > single_token_drop_tolerance:
        violations.append(
            f"pct_single_token_words dropped by {delta_single:.4f} "
            f"(limit {single_token_drop_tolerance:.4f})."
        )

    delta_two = (
        baseline_stats["pct_two_or_less_tokens_words"]
        - current_stats["pct_two_or_less_tokens_words"]
    )
    if delta_two > two_token_drop_tolerance:
        violations.append(
            f"pct_two_or_less_tokens_words dropped by {delta_two:.4f} "
            f"(limit {two_token_drop_tolerance:.4f})."
        )

    payload = json.dumps(current_stats, indent=2)
    typer.echo("# Tokenizer coverage guard")
    typer.echo(f"- Baseline: {baseline}")
    typer.echo(f"- Tokenizer: {tokenizer_path}")
    typer.echo(f"- Sample: {sample_file}")
    typer.echo(payload)

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload)

    if violations:
        typer.echo("Guard failed:")
        for violation in violations:
            typer.echo(f"  - {violation}")
        raise typer.Exit(code=1)

    typer.echo("Guard passed: tokenizer coverage within tolerance.")


if __name__ == "__main__":
    app()
```

### File: `scripts/compute/create_reservations.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

# Example Slurm reservations for Stage 2 (edit dates/times as needed).

PARTITION="gpu-a6000"
ACCOUNT="${ACCOUNT:-research}"

function reserve() {
  local name="$1"
  local start="$2"
  local duration="$3"
  local nodes="$4"
  scontrol create reservation="Name=${name},StartTime=${start},Duration=${duration},Nodes=${nodes},PartitionName=${PARTITION},Users=${USER},Accounts=${ACCOUNT}"
}

# Pilot run (1 node, 2 GPUs)
reserve "NL_Pilot" "2025-02-10T08:00:00" "3-00:00:00" 1

# Ablations (1 node)
reserve "NL_Ablations" "2025-02-13T08:00:00" "2-00:00:00" 1

# Mid-scale (2 nodes)
reserve "NL_Mid" "2025-02-17T08:00:00" "10-00:00:00" 2

# Mid evals (1 node)
reserve "NL_MidEval" "2025-02-27T08:00:00" "2-00:00:00" 1

# Target warmup (2 nodes)
reserve "NL_TargetWarmup" "2025-03-03T08:00:00" "3-00:00:00" 2

# Target full run (2 nodes)
reserve "NL_TargetFull" "2025-03-06T08:00:00" "14-00:00:00" 2

# Final evals (1 node)
reserve "NL_FinalEval" "2025-03-20T08:00:00" "3-00:00:00" 1

echo "Submitted reservations for Stage 2 (check with scontrol show reservation)."
```

### File: `scripts/data/__init__.py`

```python
"""Data preparation scripts (tokenizer/filtering/sharding)."""

```

### File: `scripts/data/check_tokenizer.py`

```python
#!/usr/bin/env python3
"""Utility to record and verify tokenizer artifact checksums."""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Optional


def compute_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def dump_metadata(path: Path, sha256: str, output: Optional[Path]) -> None:
    if not output:
        return
    payload = {
        "tokenizer_path": str(path),
        "sha256": sha256,
    }
    output.write_text(json.dumps(payload, indent=2) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        required=True,
        help="Path to the SentencePiece tokenizer model (.model).",
    )
    parser.add_argument(
        "--expected-sha256",
        type=str,
        default=None,
        help=(
            "Optional expected checksum; if provided and mismatch occurs, "
            "exits with non-zero status."
        ),
    )
    parser.add_argument(
        "--metadata-json",
        type=Path,
        default=None,
        help="Optional path to write checksum metadata as JSON.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Only emit errors; suppress the default stdout line.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tokenizer_path = args.tokenizer_path
    if not tokenizer_path.exists():
        raise SystemExit(f"Tokenizer file not found: {tokenizer_path}")

    sha256 = compute_sha256(tokenizer_path)
    if not args.quiet:
        print(f"{sha256}  {tokenizer_path}")

    dump_metadata(tokenizer_path, sha256, args.metadata_json)

    expected = args.expected_sha256
    if expected and expected.lower() != sha256.lower():
        raise SystemExit(
            f"Checksum mismatch for {tokenizer_path} (expected {expected}, got {sha256})"
        )


if __name__ == "__main__":
    main()
```

### File: `scripts/data/check_tokenizer_coverage.py`

```python
#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from nested_learning.tokenizer_coverage import compute_tokenizer_coverage_stats

app = typer.Typer(add_completion=False, help="Compute tokenizer coverage stats on a text sample.")


@app.command()
def main(
    tokenizer_path: Path = typer.Option(..., help="SentencePiece model path."),
    sample_file: Path = typer.Option(..., help="Text file with representative lines."),
    max_lines: int = typer.Option(10000, help="Maximum lines to process."),
    output: Optional[Path] = typer.Option(None, help="Optional JSON output path."),
) -> None:
    try:
        result = compute_tokenizer_coverage_stats(tokenizer_path, sample_file, max_lines=max_lines)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    payload = json.dumps(result, indent=2)
    typer.echo(payload)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload)


if __name__ == "__main__":
    app()
```

### File: `scripts/data/filter_corpus.py`

```python
#!/usr/bin/env python
from __future__ import annotations

import os
from collections import deque
from pathlib import Path
from typing import Optional

import typer
from datasets import load_dataset
from langdetect import DetectorFactory, LangDetectException, detect_langs
from tqdm import tqdm

DetectorFactory.seed = 0

app = typer.Typer(
    add_completion=False, help="Filter datasets by language/length and deduplicate lines."
)


def normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def is_target_language(text: str, target_lang: str, threshold: float) -> bool:
    try:
        langs = detect_langs(text)
    except LangDetectException:
        return False
    return any(lang.lang == target_lang and lang.prob >= threshold for lang in langs)


@app.command()
def main(
    dataset: str = typer.Option(..., help="HF dataset name, e.g. HuggingFaceFW/fineweb"),
    subset: Optional[str] = typer.Option(None, help="Optional dataset subset/config name."),
    split: str = typer.Option("train", help="Dataset split."),
    text_column: str = typer.Option("text", help="Column containing text."),
    target_lang: str = typer.Option("en", help="Language code to keep."),
    lang_threshold: float = typer.Option(0.80, help="Minimum probability for language detection."),
    min_chars: int = typer.Option(200, help="Minimum character count."),
    max_chars: int = typer.Option(10000, help="Maximum character count."),
    output_path: Path = typer.Option(
        Path("data/filtered/output.jsonl"), help="Destination JSONL file."
    ),
    dedup_window: int = typer.Option(
        50000, help="Number of recent hashes to retain for deduplication."
    ),
    limit: Optional[int] = typer.Option(None, help="Optional limit on records processed."),
    streaming: bool = typer.Option(True, help="Use HF streaming mode."),
    data_files: Optional[str] = typer.Option(
        None, help="Optional data_files argument (e.g., local text file)."
    ),
    force_exit: bool = typer.Option(
        False, help="Force os._exit(0) to avoid async finalization issues."
    ),
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    load_kwargs = {}
    if data_files is not None:
        # Ensure the requested split exists for local files (HF `text` dataset defaults can be odd).
        load_kwargs["data_files"] = {split: data_files}
    try:
        dataset_obj = load_dataset(
            dataset, subset, split=split, streaming=streaming, **load_kwargs
        )
    except ValueError as err:
        msg = str(err)
        if "Bad split" not in msg:
            raise
        ds_dict = load_dataset(dataset, subset, streaming=streaming, **load_kwargs)
        if not hasattr(ds_dict, "keys"):
            raise
        available = list(ds_dict.keys())
        if not available:
            raise
        fallback = (
            "train"
            if "train" in available
            else ("test" if "test" in available else available[0])
        )
        typer.echo(f"[Filter] Requested split '{split}' unavailable; using '{fallback}'")
        dataset_obj = ds_dict[fallback]
    iterator = dataset_obj if streaming else iter(dataset_obj)
    seen_hashes = set()
    hash_queue = deque()
    kept = 0
    total = 0
    with output_path.open("w", encoding="utf-8") as writer:
        for row in tqdm(iterator, desc="Filtering dataset"):
            total += 1
            text = row.get(text_column)
            if not isinstance(text, str):
                continue
            normalized = normalize_text(text)
            if len(normalized) < min_chars or len(normalized) > max_chars:
                continue
            if not is_target_language(normalized, target_lang, lang_threshold):
                continue
            hashed = hash(normalized)
            if hashed in seen_hashes:
                continue
            writer.write(normalized + "\n")
            kept += 1
            seen_hashes.add(hashed)
            hash_queue.append(hashed)
            if len(hash_queue) > dedup_window:
                old_hash = hash_queue.popleft()
                seen_hashes.discard(old_hash)
            if limit and kept >= limit:
                break
    typer.echo(f"[Filter] Processed={total} kept={kept} -> {output_path}")
    if force_exit:
        os._exit(0)


if __name__ == "__main__":
    app()
```

### File: `scripts/data/process_mixture.py`

```python
#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import typer
import yaml
from shard_corpus import ShardConfig, shard_dataset

app = typer.Typer(
    add_completion=False, help="Process a dataset manifest to shard multiple corpora."
)


@app.command()
def main(
    manifest: Path = typer.Argument(..., help="YAML manifest describing datasets."),
    tokenizer_path: Path = typer.Option(..., help="SentencePiece model to tokenize with."),
    log_file: Path = typer.Option(
        Path("data/mixtures/mixture_stats.json"), help="Output stats JSON."
    ),
) -> None:
    data = yaml.safe_load(manifest.read_text())
    datasets = data.get("datasets", data)
    stats: List[Dict[str, Any]] = []
    for entry in datasets:
        name = entry["name"]
        config = ShardConfig(
            name=name,
            dataset=entry["dataset"],
            split=entry.get("split", "train"),
            subset=entry.get("subset"),
            text_column=entry.get("text_column", "text"),
            tokenizer_path=tokenizer_path,
            seq_len=entry.get("seq_len", 2048),
            sequences_per_shard=entry.get("sequences_per_shard", 1024),
            output_dir=Path(entry.get("output_dir", f"data/shards/{name}")),
            eos_id=entry.get("eos_id", -1),
            max_records=entry.get("max_records"),
            data_files=entry.get("data_files"),
        )
        stats.append(shard_dataset(config))
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.write_text(json.dumps({"manifest": str(manifest), "stats": stats}, indent=2))
    typer.echo(f"[Mixture] Logged stats for {len(stats)} datasets -> {log_file}")


if __name__ == "__main__":
    app()
```

### File: `scripts/data/run_full.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

# General controls
TOKENIZER_MANIFEST=${TOKENIZER_MANIFEST:-configs/data/refinedweb_mixture.yaml}
TOKENIZER_OUTPUT_DIR=${TOKENIZER_OUTPUT_DIR:-artifacts/tokenizer/refinedweb_mix}
TOKENIZER_MODEL=${TOKENIZER_MODEL:-${TOKENIZER_OUTPUT_DIR}/spm_32000_unigram.model}
VOCAB_SIZE=${VOCAB_SIZE:-32000}
TOKENIZER_LOG=${TOKENIZER_LOG:-data/mixtures/refinedweb_mix_tokenizer_full.json}
MIXTURE_CONFIG=${MIXTURE_CONFIG:-configs/data/refinedweb_mixture_full.yaml}
SHARD_LOG=${SHARD_LOG:-data/mixtures/refinedweb_mix_full_shards.json}
FORCE_FILTER=${FORCE_FILTER:-0}
RETRAIN_TOKENIZER=${RETRAIN_TOKENIZER:-0}
FALLBACK_SPLIT=${FALLBACK_SPLIT:-test}

mkdir -p data/filtered data/shards artifacts/tokenizer data/mixtures

filter_dataset() {
  local name=$1
  local dataset=$2
  local subset=$3
  local split=$4
  local text_column=$5
  local limit=$6
  local output=$7
  local target_lang=${8:-en}
  local lang_threshold=${9:-0.85}
  local min_chars=${10:-200}
  local max_chars=${11:-12000}

  if [[ "${FORCE_FILTER}" != "1" && -f "${output}" ]]; then
    echo "[Data][${name}] Found existing ${output}, skipping filter step (set FORCE_FILTER=1 to rebuild)"
    return
  fi

  echo "[Data][${name}] Filtering ${dataset}${subset:+/${subset}} -> ${output}"
  run_filter() {
    local split_value=$1
    cmd=(uv run python scripts/data/filter_corpus.py
      --dataset "${dataset}"
      --split "${split_value}"
      --text-column "${text_column}"
      --target-lang "${target_lang}"
      --lang-threshold "${lang_threshold}"
      --min-chars "${min_chars}"
      --max-chars "${max_chars}"
      --output-path "${output}"
      --force-exit)
    if [[ -n "${subset}" ]]; then
      cmd+=(--subset "${subset}")
    fi
    if [[ -n "${limit}" ]]; then
      cmd+=(--limit "${limit}")
    fi
    "${cmd[@]}"
  }

  if ! run_filter "${split}"; then
    if [[ -n "${FALLBACK_SPLIT}" && "${FALLBACK_SPLIT}" != "${split}" ]]; then
      echo "[Data][${name}] Primary split '${split}' failed; retrying with fallback '${FALLBACK_SPLIT}'"
      run_filter "${FALLBACK_SPLIT}"
    else
      exit 1
    fi
  fi
}

echo "[Data] === Stage 1: Filtering corpora ==="
filter_dataset "refinedweb" \
  "${RW_DATASET:-HuggingFaceFW/fineweb}" \
  "${RW_SUBSET:-sample-10BT}" \
  "${RW_SPLIT:-train}" \
  "${RW_TEXT_COLUMN:-text}" \
  "${RW_LIMIT:-100000}" \
  "${RW_OUTPUT:-data/filtered/refinedweb_en_full.txt}" \
  "${RW_LANG:-en}" \
  "${RW_LANG_THRESHOLD:-0.85}" \
  "${RW_MIN_CHARS:-200}" \
  "${RW_MAX_CHARS:-8000}"

filter_dataset "wikipedia" \
  "${WIKI_DATASET:-wikimedia/wikipedia}" \
  "${WIKI_SUBSET:-20231101.en}" \
  "${WIKI_SPLIT:-train}" \
  "${WIKI_TEXT_COLUMN:-text}" \
  "${WIKI_LIMIT:-50000}" \
  "${WIKI_OUTPUT:-data/filtered/wikipedia_en_full.txt}" \
  "${WIKI_LANG:-en}" \
  "${WIKI_LANG_THRESHOLD:-0.85}" \
  "${WIKI_MIN_CHARS:-200}" \
  "${WIKI_MAX_CHARS:-8000}"

filter_dataset "c4" \
  "${C4_DATASET:-allenai/c4}" \
  "${C4_SUBSET:-en}" \
  "${C4_SPLIT:-train}" \
  "${C4_TEXT_COLUMN:-text}" \
  "${C4_LIMIT:-50000}" \
  "${C4_OUTPUT:-data/filtered/c4_en_full.txt}" \
  "${C4_LANG:-en}" \
  "${C4_LANG_THRESHOLD:-0.85}" \
  "${C4_MIN_CHARS:-200}" \
  "${C4_MAX_CHARS:-8000}"

filter_dataset "redpajama" \
  "${RPJ_DATASET:-cerebras/SlimPajama-627B}" \
  "${RPJ_SUBSET:-}" \
  "${RPJ_SPLIT:-train}" \
  "${RPJ_TEXT_COLUMN:-text}" \
  "${RPJ_LIMIT:-50000}" \
  "${RPJ_OUTPUT:-data/filtered/redpajama_en_full.txt}" \
  "${RPJ_LANG:-en}" \
  "${RPJ_LANG_THRESHOLD:-0.85}" \
  "${RPJ_MIN_CHARS:-200}" \
  "${RPJ_MAX_CHARS:-8000}"

filter_dataset "code" \
  "${CODE_DATASET:-codeparrot/codeparrot-clean-train}" \
  "${CODE_SUBSET:-}" \
  "${CODE_SPLIT:-train}" \
  "${CODE_TEXT_COLUMN:-content}" \
  "${CODE_LIMIT:-50000}" \
  "${CODE_OUTPUT:-data/filtered/code_en_full.txt}" \
  "${CODE_LANG:-en}" \
  "${CODE_LANG_THRESHOLD:-0.50}" \
  "${CODE_MIN_CHARS:-200}" \
  "${CODE_MAX_CHARS:-16000}"

echo "[Data] === Stage 2: Tokenizer training ==="
if [[ ! -f "${TOKENIZER_MODEL}" || "${RETRAIN_TOKENIZER}" == "1" ]]; then
  uv run python scripts/data/train_tokenizer.py \
    --manifest "${TOKENIZER_MANIFEST}" \
    --vocab-size "${VOCAB_SIZE}" \
    --output-dir "${TOKENIZER_OUTPUT_DIR}" \
    --log-file "${TOKENIZER_LOG}"
else
  echo "[Data] Tokenizer already exists at ${TOKENIZER_MODEL}; set RETRAIN_TOKENIZER=1 to rebuild."
fi

echo "[Data] === Stage 3: Sharding filtered corpora ==="
uv run python scripts/data/process_mixture.py \
  "${MIXTURE_CONFIG}" \
  --tokenizer-path "${TOKENIZER_MODEL}" \
  --log-file "${SHARD_LOG}"

echo "[Data] Full pipeline complete."
```

### File: `scripts/data/run_sample.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

TOKENIZER_MODEL=${1:-artifacts/tokenizer/refinedweb_mix/spm_32000_unigram.model}
TOKENIZER_DIR="$(dirname "${TOKENIZER_MODEL}")"

if [[ ! -f "data/filtered/refinedweb_en_sample.txt" ]]; then
  echo "[Data] Creating filtered RefinedWeb sample"
  uv run python scripts/data/filter_corpus.py \
    --dataset HuggingFaceFW/fineweb \
    "--subset=sample-10BT" \
    --split train \
    --text-column text \
    --target-lang en \
    --lang-threshold 0.85 \
    --min-chars 200 \
    --max-chars 8000 \
    --limit 2000 \
    --output-path data/filtered/refinedweb_en_sample.txt \
    --force-exit
fi

if [[ ! -f "data/filtered/wikipedia_en_sample.txt" ]]; then
  echo "[Data] Creating filtered Wikipedia sample"
  uv run python scripts/data/filter_corpus.py \
    --dataset wikimedia/wikipedia \
    "--subset=20231101.en" \
    --split train \
    --text-column text \
    --target-lang en \
    --lang-threshold 0.85 \
    --min-chars 200 \
    --max-chars 8000 \
    --limit 1000 \
    --output-path data/filtered/wikipedia_en_sample.txt \
    --force-exit
fi

if [[ ! -f "data/filtered/c4_en_sample.txt" ]]; then
  echo "[Data] Creating filtered C4 sample"
  uv run python scripts/data/filter_corpus.py \
    --dataset allenai/c4 --subset en --split train \
    --text-column text --target-lang en --lang-threshold 0.85 \
    --min-chars 200 --max-chars 8000 --limit 1000 \
    --output-path data/filtered/c4_en_sample.txt --force-exit
fi

if [[ ! -f "data/filtered/redpajama_en_sample.txt" ]]; then
  echo "[Data] Creating filtered SlimPajama sample"
  uv run python scripts/data/filter_corpus.py \
    "--dataset=cerebras/SlimPajama-627B" \
    --split train \
    --text-column text \
    --target-lang en \
    --lang-threshold 0.85 \
    --min-chars 200 \
    --max-chars 8000 \
    --limit 1000 \
    --output-path data/filtered/redpajama_en_sample.txt \
    --force-exit
fi

if [[ ! -f "data/filtered/code_en_sample.txt" ]]; then
  echo "[Data] Creating filtered code sample"
  uv run python scripts/data/filter_corpus.py \
    --dataset codeparrot/codeparrot-clean-train --split train \
    --text-column content --target-lang en --lang-threshold 0.5 \
    --min-chars 200 --max-chars 12000 --limit 1000 \
    --output-path data/filtered/code_en_sample.txt --force-exit
fi

if [[ ! -f "${TOKENIZER_MODEL}" ]]; then
  echo "[Data] Training tokenizer (sample) -> ${TOKENIZER_DIR}"
  uv run python scripts/data/train_tokenizer.py \
    --manifest configs/data/refinedweb_mixture_filtered.yaml \
    --vocab-size 32000 \
    --no-hard-vocab-limit \
    --output-dir "${TOKENIZER_DIR}" \
    --log-file data/mixtures/refinedweb_mix_tokenizer_sample.json
fi

echo "[Data] Sharding filtered samples"
uv run python scripts/data/process_mixture.py \
  configs/data/refinedweb_mixture_filtered.yaml \
  --tokenizer-path ${TOKENIZER_MODEL} \
  --log-file data/mixtures/refinedweb_mix_filtered_shards.json

echo "[Data] Sample pipeline complete"
```

### File: `scripts/data/shard_corpus.py`

```python
#!/usr/bin/env python
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import sentencepiece as spm
import typer
from datasets import load_dataset
from tqdm import tqdm

app = typer.Typer(add_completion=False, help="Shard datasets into tokenized numpy binaries.")


@dataclass
class ShardConfig:
    name: str
    dataset: str
    split: str = "train"
    subset: str | None = None
    text_column: str = "text"
    tokenizer_path: Path = Path()
    seq_len: int = 2048
    sequences_per_shard: int = 1024
    output_dir: Path = Path("data/shards")
    eos_id: int = -1
    max_records: Optional[int] = None
    data_files: Optional[str] = None


def shard_dataset(config: ShardConfig) -> dict:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    processor = spm.SentencePieceProcessor(model_file=str(config.tokenizer_path))
    eos = config.eos_id if config.eos_id >= 0 else processor.eos_id()
    load_kwargs = {}
    if config.data_files is not None:
        load_kwargs["data_files"] = {config.split: config.data_files}
    try:
        ds = load_dataset(
            config.dataset, config.subset, split=config.split, streaming=True, **load_kwargs
        )
    except ValueError as err:
        msg = str(err)
        if "Bad split" not in msg:
            raise
        ds_dict = load_dataset(config.dataset, config.subset, streaming=True, **load_kwargs)
        available = list(ds_dict.keys())
        if not available:
            raise
        fallback = (
            "train"
            if "train" in available
            else ("test" if "test" in available else available[0])
        )
        typer.echo(f"[Shard] Requested split '{config.split}' unavailable; using '{fallback}'")
        ds = ds_dict[fallback]

    buffer: List[int] = []
    sequences: List[List[int]] = []
    shard_idx = 0
    records = 0
    sequences_total = 0
    tokens_total = 0

    for row in tqdm(ds, desc=f"Sharding {config.name}", unit="record"):
        text = row.get(config.text_column)
        if not isinstance(text, str):
            continue
        tokens = processor.encode(text)
        tokens.append(eos)
        tokens_total += len(tokens)
        buffer.extend(tokens)
        records += 1
        while len(buffer) >= config.seq_len:
            seq = buffer[: config.seq_len]
            buffer = buffer[config.seq_len :]
            sequences.append(seq)
            sequences_total += 1
            if len(sequences) >= config.sequences_per_shard:
                _write_shard(sequences, config.output_dir, shard_idx)
                shard_idx += 1
                sequences = []
        if config.max_records and records >= config.max_records:
            break
    if sequences:
        _write_shard(sequences, config.output_dir, shard_idx)
        shard_idx += 1

    stats = {
        "name": config.name,
        "dataset": config.dataset,
        "subset": config.subset,
        "records": records,
        "sequences": sequences_total,
        "tokens": tokens_total,
        "shards": shard_idx,
        "output_dir": str(config.output_dir),
    }
    typer.echo(
        f"[Shard] {config.name}: records={records} sequences={sequences_total} "
        f"shards={shard_idx} -> {config.output_dir}"
    )
    return stats


@app.command()
def main(
    dataset: str = typer.Option("roneneldan/TinyStories", help="HF dataset name."),
    split: str = typer.Option("train", help="Dataset split."),
    subset: Optional[str] = typer.Option(None, help="Optional dataset subset/config."),
    text_column: str = typer.Option("text", help="Text column."),
    tokenizer_path: Path = typer.Option(..., help="Path to SentencePiece model."),
    seq_len: int = typer.Option(2048, help="Sequence length (tokens per sample)."),
    sequences_per_shard: int = typer.Option(1024, help="Number of sequences per shard."),
    output_dir: Path = typer.Option(Path("data/shards"), help="Directory for shard files."),
    eos_id: int = typer.Option(-1, help="EOS token id (defaults to tokenizer default)."),
    max_records: Optional[int] = typer.Option(None, help="Optional max records to process."),
    name: Optional[str] = typer.Option(None, help="Friendly name for logging."),
    log_file: Optional[Path] = typer.Option(
        Path("data/mixtures/shard_stats.json"), help="Where to save shard stats JSON."
    ),
    data_files: Optional[str] = typer.Option(None, help="Optional data_files argument."),
) -> None:
    config = ShardConfig(
        name=name or dataset.split("/")[-1],
        dataset=dataset,
        split=split,
        subset=subset,
        text_column=text_column,
        tokenizer_path=tokenizer_path,
        seq_len=seq_len,
        sequences_per_shard=sequences_per_shard,
        output_dir=output_dir,
        eos_id=eos_id,
        max_records=max_records,
        data_files=data_files,
    )
    stats = shard_dataset(config)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_file.write_text(json.dumps(stats, indent=2))
        typer.echo(f"[Shard] Stats logged to {log_file}")


def _write_shard(sequences: List[List[int]], output_dir: Path, shard_idx: int) -> None:
    array = np.asarray(sequences, dtype=np.int32)
    target = output_dir / f"shard_{shard_idx:05d}.npy"
    np.save(target, array)


if __name__ == "__main__":
    app()
```

### File: `scripts/data/train_tokenizer.py`

```python
#!/usr/bin/env python
from __future__ import annotations

import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import sentencepiece as spm
import typer
import yaml
from datasets import load_dataset

app = typer.Typer(add_completion=False, help="Train a SentencePiece tokenizer from HF datasets.")


@dataclass
class DatasetSpec:
    name: str
    dataset: str
    split: str = "train"
    subset: str | None = None
    text_column: str = "text"
    sample_limit: int = 100_000
    data_files: str | None = None


def _load_specs_from_manifest(manifest: Path) -> List[DatasetSpec]:
    data = yaml.safe_load(manifest.read_text())
    entries = data.get("datasets", data)
    specs = []
    for entry in entries:
        specs.append(
            DatasetSpec(
                name=entry.get("name") or entry["dataset"].split("/")[-1],
                dataset=entry["dataset"],
                split=entry.get("split", "train"),
                subset=entry.get("subset"),
                text_column=entry.get("text_column", "text"),
                sample_limit=entry.get("sample_limit", 100_000),
                data_files=entry.get("data_files"),
            )
        )
    return specs


def _write_samples(spec: DatasetSpec, handle) -> int:
    load_kwargs = {}
    if spec.data_files is not None:
        load_kwargs["data_files"] = {spec.split: spec.data_files}
    try:
        ds = load_dataset(
            spec.dataset, spec.subset, split=spec.split, streaming=True, **load_kwargs
        )
    except ValueError as err:
        msg = str(err)
        if "Bad split" not in msg:
            raise
        ds_dict = load_dataset(spec.dataset, spec.subset, streaming=True, **load_kwargs)
        available = list(ds_dict.keys())
        if not available:
            raise
        fallback = (
            "train"
            if "train" in available
            else ("test" if "test" in available else available[0])
        )
        typer.echo(
            f"[Tokenizer] Requested split '{spec.split}' unavailable for {spec.dataset}; "
            f"using '{fallback}'"
        )
        ds = ds_dict[fallback]
    count = 0
    for row in ds:
        text = row.get(spec.text_column)
        if not isinstance(text, str):
            continue
        handle.write(text.replace("\n", " ") + "\n")
        count += 1
        if spec.sample_limit > 0 and count >= spec.sample_limit:
            break
    return count


@app.command()
def main(
    dataset: str = typer.Option(
        "roneneldan/TinyStories", help="HF dataset name (ignored if manifest set)."
    ),
    split: str = typer.Option("train", help="Dataset split (ignored if manifest set)."),
    text_column: str = typer.Option("text", help="Text column (ignored if manifest set)."),
    sample_limit: int = typer.Option(
        100_000, help="Sample limit per dataset (ignored if manifest set)."
    ),
    vocab_size: int = typer.Option(32_000, help="SentencePiece vocabulary size."),
    model_type: str = typer.Option("unigram", help="SentencePiece model type."),
    character_coverage: float = typer.Option(0.9995, help="Character coverage target."),
    hard_vocab_limit: bool = typer.Option(
        True,
        help=(
            "Require the trained vocab to match vocab_size exactly. "
            "Disable for tiny sample corpora where vocab_size is unattainable."
        ),
    ),
    output_dir: Path = typer.Option(
        Path("artifacts/tokenizer"), help="Directory for tokenizer artifacts."
    ),
    manifest: Optional[Path] = typer.Option(
        None, help="YAML manifest describing multiple datasets."
    ),
    log_file: Optional[Path] = typer.Option(
        Path("data/mixtures/tokenizer_samples.json"), help="Where to log dataset sample stats."
    ),
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = (
        _load_specs_from_manifest(manifest)
        if manifest is not None
        else [
            DatasetSpec(
                name=dataset.split("/")[-1],
                dataset=dataset,
                split=split,
                text_column=text_column,
                sample_limit=sample_limit,
            )
        ]
    )
    model_prefix = output_dir / f"spm_{vocab_size}_{model_type}"
    stats = []
    with tempfile.NamedTemporaryFile("w+", encoding="utf-8", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        typer.echo(f"[Tokenizer] Writing samples to {tmp_path}")
        with tmp_path.open("w", encoding="utf-8") as handle:
            for spec in specs:
                typer.echo(
                    f"[Tokenizer] Streaming {spec.name} ({spec.dataset}) limit={spec.sample_limit}"
                )
                count = _write_samples(spec, handle)
                stats.append({"name": spec.name, "dataset": spec.dataset, "samples": count})
    typer.echo(f"[Tokenizer] Training SentencePiece -> {model_prefix}")
    total_samples = sum(s["samples"] for s in stats)
    spm.SentencePieceTrainer.train(
        input=str(tmp_path),
        model_prefix=str(model_prefix),
        vocab_size=vocab_size,
        model_type=model_type,
        character_coverage=character_coverage,
        hard_vocab_limit=hard_vocab_limit,
        # SentencePiece requires input_sentence_size <= 0 or > 100.
        input_sentence_size=(total_samples if total_samples > 100 else 0),
        shuffle_input_sentence=True,
        train_extremely_large_corpus=True,
    )
    typer.echo(f"[Tokenizer] Saved model to {model_prefix}.model")
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        log_payload = {"model": str(model_prefix), "datasets": stats}
        log_file.write_text(json.dumps(log_payload, indent=2))
        typer.echo(f"[Tokenizer] Logged sample stats to {log_file}")


if __name__ == "__main__":
    app()
```

### File: `scripts/data/validate_mixture.py`

```python
#!/usr/bin/env python
from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(add_completion=False, help="Validate mixture manifests and shard inventories.")


def _dir_stats(path: Path, sample_limit: int = 2000) -> tuple[dict[str, float], set[str]]:
    total_bytes = 0
    file_count = 0
    sampled_names: set[str] = set()
    for entry in sorted(path.rglob("*.npy")):
        total_bytes += entry.stat().st_size
        file_count += 1
        if len(sampled_names) < sample_limit:
            sampled_names.add(f"{path.name}/{entry.relative_to(path).as_posix()}")
    return {"files": file_count, "bytes": total_bytes}, sampled_names


@app.command()
def main(
    manifest: Path = typer.Option(..., help="Path to data/manifest/*.json file."),
    output: Optional[Path] = typer.Option(None, help="Optional JSON output path for the report."),
    overlap_threshold: float = typer.Option(
        0.05, help="Warn when filename overlap exceeds this Jaccard."
    ),
) -> None:
    spec = json.loads(manifest.read_text())
    report = {"manifest": spec.get("name"), "sources": []}
    sampled_sets: dict[str, set[str]] = {}
    for entry in spec.get("sources", []):
        shards_dir = Path(entry["shards_dir"])
        source_report = dict(entry)
        source_report["exists"] = shards_dir.exists()
        if shards_dir.exists():
            stats, sampled = _dir_stats(shards_dir)
            source_report.update(stats)
            sampled_sets[entry["name"]] = sampled
        stats_file = entry.get("stats_file")
        if stats_file and Path(stats_file).exists():
            try:
                stats_payload = json.loads(Path(stats_file).read_text())
                source_report["stats_snapshot"] = stats_payload.get(entry["name"])
            except json.JSONDecodeError:
                source_report["stats_snapshot"] = "unreadable"
        report["sources"].append(source_report)
    overlaps = []
    for a, b in combinations(sampled_sets.keys(), 2):
        set_a = sampled_sets[a]
        set_b = sampled_sets[b]
        if not set_a or not set_b:
            continue
        jaccard = len(set_a & set_b) / len(set_a | set_b)
        entry = {"pair": [a, b], "jaccard": jaccard}
        if jaccard >= overlap_threshold:
            entry["warning"] = True
        overlaps.append(entry)
    report["filename_overlap"] = overlaps
    summary = json.dumps(report, indent=2)
    typer.echo(summary)
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(summary)


if __name__ == "__main__":
    app()
```

### File: `scripts/eval/__init__.py`

```python
# Eval utilities package marker.
```

### File: `scripts/eval/compare_variants.py`

```python
#!/usr/bin/env python
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import torch
import typer
from omegaconf import OmegaConf
from tqdm import tqdm

from nested_learning.device import resolve_device
from nested_learning.memorize import (
    MemorizeConfig,
    memorize_sequence,
    memorize_tokens,
    restore_state_dict,
    snapshot_state_dict,
)
from nested_learning.tokenizer import SentencePieceTokenizer
from nested_learning.training import build_model_from_cfg, unwrap_config

app = typer.Typer(
    add_completion=False, help="Compare long-context metrics across two model variants."
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    config: Path
    checkpoint: Path


def _load_model(spec: ModelSpec, device: torch.device) -> torch.nn.Module:
    cfg = OmegaConf.load(spec.config)
    cfg = unwrap_config(cfg)
    model = build_model_from_cfg(cfg.model)
    state = torch.load(spec.checkpoint, map_location="cpu", weights_only=False)
    state_dict = state["model"] if "model" in state else state
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(
            f"[compare] {spec.name}: state_dict mismatch "
            f"(missing={len(missing)} unexpected={len(unexpected)}) – continuing."
        )
    return model.to(device).eval()


def _logprob_answer(
    model: torch.nn.Module,
    tokenizer: SentencePieceTokenizer,
    prompt: str,
    answer: str,
    device: torch.device,
    *,
    fast_state=None,
) -> float:
    prompt_ids = tokenizer.encode(prompt, add_bos=True)
    answer_ids = tokenizer.encode(" " + answer, add_bos=False)
    inputs = torch.cat([prompt_ids, answer_ids], dim=0).to(device)
    with torch.no_grad():
        logits = (
            model(inputs.unsqueeze(0), fast_state=fast_state)
            if fast_state is not None
            else model(inputs.unsqueeze(0))
        )
        log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
        target = inputs.unsqueeze(0)[:, 1:]
        gathered = log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        prompt_len = prompt_ids.numel()
        answer_logprob = gathered[0, prompt_len - 1 :].sum().item()
    return float(answer_logprob)


def _memorize_prompt_answer_only(
    model: torch.nn.Module,
    tokenizer: SentencePieceTokenizer,
    prompt: str,
    answer: str,
    device: torch.device,
    memorize_cfg: MemorizeConfig,
    *,
    fast_state=None,
) -> Dict[str, float]:
    """
    Memorize using gradients for the answer tokens only.

    This avoids updating on long filler/haystack tokens when `use_correct_answer=True`,
    which otherwise makes the comparison noisy for randomly-initialized checkpoints.
    """
    prompt_ids = tokenizer.encode(prompt, add_bos=True)
    answer_ids = tokenizer.encode(" " + answer, add_bos=False)
    inputs = torch.cat([prompt_ids, answer_ids], dim=0).to(device)
    batch = inputs.unsqueeze(0)
    teach_mask = torch.zeros((1, batch.size(1)), device=device)
    start = max(0, prompt_ids.numel() - 1)
    end = min(batch.size(1), start + answer_ids.numel())
    teach_mask[:, start:end] = 1.0
    return memorize_tokens(
        model,
        batch,
        memorize_cfg,
        fast_state=fast_state,
        teach_mask=teach_mask,
    )


def _make_passkey_prompt(*, filler_sentences: int, key: str) -> str:
    sentences = [f"This is filler sentence number {idx}." for idx in range(filler_sentences)]
    random.shuffle(sentences)
    filler = " ".join(sentences)
    return (
        f"{filler}\nRemember that the passkey for this document is {key}. "
        "Later we will ask about it.\nQuestion: What is the passkey?\nAnswer:"
    )


def _run_passkey(
    model: torch.nn.Module,
    tokenizer: SentencePieceTokenizer,
    device: torch.device,
    *,
    samples: int,
    filler_sentences: int,
    memorize_cfg: MemorizeConfig,
) -> Dict[str, Any]:
    base_state: Dict[str, torch.Tensor] | None = None
    fast_state = None
    if memorize_cfg.enabled and (not memorize_cfg.use_fast_state) and memorize_cfg.reset:
        base_state = snapshot_state_dict(model)

    correct_base = 0
    correct_mem = 0
    true_lp_base_sum = 0.0
    false_lp_base_sum = 0.0
    margin_base_sum = 0.0
    true_lp_mem_sum = 0.0
    false_lp_mem_sum = 0.0
    margin_mem_sum = 0.0
    path_stats: Dict[str, float] = {}
    for _ in tqdm(range(samples), desc="passkey"):
        key = f"PASSKEY-{random.randint(1000, 9999)}"
        prompt = _make_passkey_prompt(filler_sentences=filler_sentences, key=key)
        distractor = f"PASSKEY-{random.randint(1000, 9999)}"

        lp_true = _logprob_answer(model, tokenizer, prompt, key, device, fast_state=fast_state)
        lp_false = _logprob_answer(
            model, tokenizer, prompt, distractor, device, fast_state=fast_state
        )
        correct_base += int(lp_true > lp_false)
        true_lp_base_sum += lp_true
        false_lp_base_sum += lp_false
        margin_base_sum += lp_true - lp_false

        if memorize_cfg.enabled:
            if memorize_cfg.use_fast_state:
                if fast_state is None or memorize_cfg.reset:
                    if not hasattr(model, "init_fast_state"):
                        raise RuntimeError("Model does not support fast state memorization")
                    fast_state = model.init_fast_state()
                if memorize_cfg.use_correct_answer:
                    stats = _memorize_prompt_answer_only(
                        model,
                        tokenizer,
                        prompt,
                        key,
                        device,
                        memorize_cfg,
                        fast_state=fast_state,
                    )
                else:
                    stats = memorize_sequence(
                        model,
                        tokenizer,
                        prompt,
                        device,
                        memorize_cfg,
                        fast_state=fast_state,
                    )
                for k, v in stats.items():
                    path_stats[k] = path_stats.get(k, 0.0) + v
                lp_true_mem = _logprob_answer(
                    model, tokenizer, prompt, key, device, fast_state=fast_state
                )
                lp_false_mem = _logprob_answer(
                    model, tokenizer, prompt, distractor, device, fast_state=fast_state
                )
                correct_mem += int(lp_true_mem > lp_false_mem)
                true_lp_mem_sum += lp_true_mem
                false_lp_mem_sum += lp_false_mem
                margin_mem_sum += lp_true_mem - lp_false_mem
            else:
                memorize_text = prompt if not memorize_cfg.use_correct_answer else f"{prompt} {key}"
                stats = memorize_sequence(model, tokenizer, memorize_text, device, memorize_cfg)
                for k, v in stats.items():
                    path_stats[k] = path_stats.get(k, 0.0) + v
                lp_true_mem = _logprob_answer(model, tokenizer, prompt, key, device)
                lp_false_mem = _logprob_answer(model, tokenizer, prompt, distractor, device)
                correct_mem += int(lp_true_mem > lp_false_mem)
                true_lp_mem_sum += lp_true_mem
                false_lp_mem_sum += lp_false_mem
                margin_mem_sum += lp_true_mem - lp_false_mem
                if memorize_cfg.reset and base_state is not None:
                    restore_state_dict(model, base_state)
        else:
            correct_mem += int(lp_true > lp_false)
            true_lp_mem_sum += lp_true
            false_lp_mem_sum += lp_false
            margin_mem_sum += lp_true - lp_false

    denom = float(samples) if samples else 1.0
    base_acc = correct_base / denom
    mem_acc = correct_mem / denom
    payload: Dict[str, Any] = {
        "samples": samples,
        "filler_sentences": filler_sentences,
        "accuracy_base": base_acc,
        "accuracy_memorize": mem_acc,
        "accuracy_delta": mem_acc - base_acc,
        "mean_logprob_true_base": true_lp_base_sum / denom,
        "mean_logprob_true_memorize": true_lp_mem_sum / denom,
        "mean_logprob_true_delta": (true_lp_mem_sum - true_lp_base_sum) / denom,
        "mean_logprob_false_base": false_lp_base_sum / denom,
        "mean_logprob_false_memorize": false_lp_mem_sum / denom,
        "mean_logprob_false_delta": (false_lp_mem_sum - false_lp_base_sum) / denom,
        "mean_margin_base": margin_base_sum / denom,
        "mean_margin_memorize": margin_mem_sum / denom,
        "mean_margin_delta": (margin_mem_sum - margin_base_sum) / denom,
    }
    if memorize_cfg.enabled:
        payload["memorize_paths"] = (
            "all" if memorize_cfg.paths is None else ",".join(memorize_cfg.paths)
        )
        if memorize_cfg.surprise_threshold is not None:
            payload["memorize_surprise_threshold"] = memorize_cfg.surprise_threshold
        payload["memorize_use_correct_answer"] = bool(memorize_cfg.use_correct_answer)
        if path_stats:
            payload["memorize_stats"] = path_stats
    return payload


def _make_niah_prompt(*, needle: str, filler_tokens: int) -> str:
    filler_chunks = ["This is filler sentence number {}.".format(i) for i in range(filler_tokens)]
    random.shuffle(filler_chunks)
    haystack = " ".join(filler_chunks)
    prompt = (
        f"{haystack} Remember that the secret key is {needle}. Later you might be asked about it. "
    )
    prompt += "Now answer the question truthfully. What is the secret key? Answer:"
    return prompt


def _run_niah(
    model: torch.nn.Module,
    tokenizer: SentencePieceTokenizer,
    device: torch.device,
    *,
    context_lengths: List[int],
    samples_per_length: int,
    memorize_cfg: MemorizeConfig,
) -> Dict[str, Any]:
    base_state: Dict[str, torch.Tensor] | None = None
    fast_state = None
    if memorize_cfg.enabled and (not memorize_cfg.use_fast_state) and memorize_cfg.reset:
        base_state = snapshot_state_dict(model)

    results: Dict[str, Any] = {}
    path_stats: Dict[str, float] = {}
    for length in context_lengths:
        correct_base = 0
        correct_mem = 0
        true_lp_base_sum = 0.0
        false_lp_base_sum = 0.0
        margin_base_sum = 0.0
        true_lp_mem_sum = 0.0
        false_lp_mem_sum = 0.0
        margin_mem_sum = 0.0
        for _ in tqdm(range(samples_per_length), desc=f"niah@{length}"):
            needle = f"KEY-{random.randint(1000, 9999)}"
            prompt = _make_niah_prompt(needle=needle, filler_tokens=max(1, length // 128))
            distractor = f"KEY-{random.randint(1000, 9999)}"

            lp_true_base = _logprob_answer(
                model, tokenizer, prompt, needle, device, fast_state=fast_state
            )
            lp_false_base = _logprob_answer(
                model, tokenizer, prompt, distractor, device, fast_state=fast_state
            )
            correct_base += int(lp_true_base > lp_false_base)
            true_lp_base_sum += lp_true_base
            false_lp_base_sum += lp_false_base
            margin_base_sum += lp_true_base - lp_false_base

            if memorize_cfg.enabled:
                if memorize_cfg.use_fast_state:
                    if fast_state is None or memorize_cfg.reset:
                        if not hasattr(model, "init_fast_state"):
                            raise RuntimeError("Model does not support fast state memorization")
                        fast_state = model.init_fast_state()
                    if memorize_cfg.use_correct_answer:
                        stats = _memorize_prompt_answer_only(
                            model,
                            tokenizer,
                            prompt,
                            needle,
                            device,
                            memorize_cfg,
                            fast_state=fast_state,
                        )
                    else:
                        stats = memorize_sequence(
                            model,
                            tokenizer,
                            prompt,
                            device,
                            memorize_cfg,
                            fast_state=fast_state,
                        )
                    for k, v in stats.items():
                        path_stats[k] = path_stats.get(k, 0.0) + v
                    lp_true_mem = _logprob_answer(
                        model, tokenizer, prompt, needle, device, fast_state=fast_state
                    )
                    lp_false_mem = _logprob_answer(
                        model, tokenizer, prompt, distractor, device, fast_state=fast_state
                    )
                    correct_mem += int(lp_true_mem > lp_false_mem)
                    true_lp_mem_sum += lp_true_mem
                    false_lp_mem_sum += lp_false_mem
                    margin_mem_sum += lp_true_mem - lp_false_mem
                else:
                    memorize_text = (
                        prompt if not memorize_cfg.use_correct_answer else f"{prompt} {needle}"
                    )
                    stats = memorize_sequence(model, tokenizer, memorize_text, device, memorize_cfg)
                    for k, v in stats.items():
                        path_stats[k] = path_stats.get(k, 0.0) + v
                    lp_true_mem = _logprob_answer(model, tokenizer, prompt, needle, device)
                    lp_false_mem = _logprob_answer(model, tokenizer, prompt, distractor, device)
                    correct_mem += int(lp_true_mem > lp_false_mem)
                    true_lp_mem_sum += lp_true_mem
                    false_lp_mem_sum += lp_false_mem
                    margin_mem_sum += lp_true_mem - lp_false_mem
                    if memorize_cfg.reset and base_state is not None:
                        restore_state_dict(model, base_state)
            else:
                correct_mem += int(lp_true_base > lp_false_base)
                true_lp_mem_sum += lp_true_base
                false_lp_mem_sum += lp_false_base
                margin_mem_sum += lp_true_base - lp_false_base

        base_acc = correct_base / samples_per_length if samples_per_length else 0.0
        mem_acc = correct_mem / samples_per_length if samples_per_length else 0.0
        results[f"niah_{length}_baseline_accuracy"] = base_acc
        results[f"niah_{length}_memorize_accuracy"] = mem_acc
        results[f"niah_{length}_memorize_delta"] = mem_acc - base_acc
        denom = float(samples_per_length) if samples_per_length else 1.0
        results[f"niah_{length}_mean_logprob_true_base"] = true_lp_base_sum / denom
        results[f"niah_{length}_mean_logprob_true_memorize"] = true_lp_mem_sum / denom
        results[f"niah_{length}_mean_logprob_true_delta"] = (
            true_lp_mem_sum - true_lp_base_sum
        ) / denom
        results[f"niah_{length}_mean_margin_base"] = margin_base_sum / denom
        results[f"niah_{length}_mean_margin_memorize"] = margin_mem_sum / denom
        results[f"niah_{length}_mean_margin_delta"] = (margin_mem_sum - margin_base_sum) / denom

    if memorize_cfg.enabled:
        results["memorize_paths"] = (
            "all" if memorize_cfg.paths is None else ",".join(memorize_cfg.paths)
        )
        if memorize_cfg.surprise_threshold is not None:
            results["memorize_surprise_threshold"] = memorize_cfg.surprise_threshold
        results["memorize_use_correct_answer"] = bool(memorize_cfg.use_correct_answer)
        if path_stats:
            results["memorize_stats"] = path_stats
    return results


@app.command()
def main(
    a_config: Path = typer.Option(..., help="Hydra config for model A."),
    a_checkpoint: Path = typer.Option(..., help="Checkpoint for model A."),
    b_config: Path = typer.Option(..., help="Hydra config for model B."),
    b_checkpoint: Path = typer.Option(..., help="Checkpoint for model B."),
    tokenizer_path: Path = typer.Option(..., help="SentencePiece tokenizer path."),
    device: str = typer.Option("cuda:0" if torch.cuda.is_available() else "cpu"),
    output: Path = typer.Option(Path("eval/compare_variants.json")),
    seed: int = typer.Option(0, help="PRNG seed for prompt generation."),
    smoke: bool = typer.Option(False, help="Use tiny settings for quick sanity checks."),
    passkey_samples: int = typer.Option(64, help="Passkey prompts per model."),
    passkey_filler_sentences: int = typer.Option(200, help="Filler sentences for passkey."),
    niah_context_lengths: List[int] = typer.Option(
        [2048, 4096, 8192], help="Context lengths for NIAH."
    ),
    niah_samples_per_length: int = typer.Option(50, help="Samples per NIAH length."),
    memorize: bool = typer.Option(False, help="Enable test-time memorization for both models."),
    memorize_steps: int = typer.Option(1, help="Memorization passes per prompt."),
    memorize_use_correct_answer: bool = typer.Option(
        False, help="Append ground truth during memorization."
    ),
    memorize_no_reset: bool = typer.Option(False, help="Retain memory between samples."),
    memorize_surprise_threshold: float = typer.Option(
        None, help="Minimum teach-signal norm required to trigger memorization."
    ),
    memorize_layers: str = typer.Option(
        "all",
        help=(
            "Comma-separated layer indices to update during memorization "
            "(e.g., '11' or '0,11'), or 'last', or 'all'."
        ),
    ),
    memorize_paths: str = typer.Option(
        "all",
        help=(
            "Comma-separated memory paths to update (e.g., 'titan,cms_fast'); "
            "use 'all' for no restriction."
        ),
    ),
) -> None:
    random.seed(seed)
    torch_device = resolve_device(device)
    tokenizer = SentencePieceTokenizer(tokenizer_path)

    if smoke:
        passkey_samples = min(passkey_samples, 8)
        passkey_filler_sentences = min(passkey_filler_sentences, 20)
        niah_context_lengths = [256]
        niah_samples_per_length = min(niah_samples_per_length, 8)

    if memorize_paths.lower() == "all":
        allowed_paths = None
    else:
        allowed_paths = tuple(path.strip() for path in memorize_paths.split(",") if path.strip())

    layers_raw = memorize_layers.strip().lower()
    if layers_raw == "all":
        allowed_layers = None
    elif layers_raw == "last":
        allowed_layers = (-1,)
    else:
        parsed: list[int] = []
        for part in memorize_layers.split(","):
            part = part.strip()
            if not part:
                continue
            parsed.append(int(part))
        allowed_layers = tuple(parsed) if parsed else None
    memorize_cfg = MemorizeConfig(
        enabled=memorize,
        steps=max(1, memorize_steps),
        reset=not memorize_no_reset,
        use_correct_answer=memorize_use_correct_answer,
        surprise_threshold=memorize_surprise_threshold,
        paths=allowed_paths,
        layers=allowed_layers,
    )

    spec_a = ModelSpec(name="A", config=a_config, checkpoint=a_checkpoint)
    spec_b = ModelSpec(name="B", config=b_config, checkpoint=b_checkpoint)
    model_a = _load_model(spec_a, torch_device)
    model_b = _load_model(spec_b, torch_device)

    payload: Dict[str, Any] = {
        "seed": seed,
        "device": str(torch_device),
        "tokenizer_path": str(tokenizer_path),
        "a": {"config": str(a_config), "checkpoint": str(a_checkpoint)},
        "b": {"config": str(b_config), "checkpoint": str(b_checkpoint)},
        "memorize": {
            "enabled": memorize_cfg.enabled,
            "steps": memorize_cfg.steps,
            "reset": memorize_cfg.reset,
            "use_correct_answer": bool(memorize_cfg.use_correct_answer),
            "paths": "all" if memorize_cfg.paths is None else ",".join(memorize_cfg.paths),
            "surprise_threshold": memorize_cfg.surprise_threshold,
        },
    }

    payload["a"]["passkey"] = _run_passkey(
        model_a,
        tokenizer,
        torch_device,
        samples=passkey_samples,
        filler_sentences=passkey_filler_sentences,
        memorize_cfg=memorize_cfg,
    )
    payload["b"]["passkey"] = _run_passkey(
        model_b,
        tokenizer,
        torch_device,
        samples=passkey_samples,
        filler_sentences=passkey_filler_sentences,
        memorize_cfg=memorize_cfg,
    )

    payload["a"]["niah"] = _run_niah(
        model_a,
        tokenizer,
        torch_device,
        context_lengths=niah_context_lengths,
        samples_per_length=niah_samples_per_length,
        memorize_cfg=memorize_cfg,
    )
    payload["b"]["niah"] = _run_niah(
        model_b,
        tokenizer,
        torch_device,
        context_lengths=niah_context_lengths,
        samples_per_length=niah_samples_per_length,
        memorize_cfg=memorize_cfg,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    typer.echo(f"[compare] Saved comparison to {output}")


if __name__ == "__main__":
    app()
```

### File: `scripts/eval/continual.py`

```python
#!/usr/bin/env python
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import torch
import typer
import yaml
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from nested_learning.data import TokenShardDataset, collate_batch
from nested_learning.device import resolve_device
from nested_learning.memorize import (
    MemorizeConfig,
    memorize_tokens,
    restore_state_dict,
    snapshot_state_dict,
)
from nested_learning.training import build_model_from_cfg, unwrap_config

app = typer.Typer(add_completion=False, help="Continual learning evaluation harness.")


def load_segments(yaml_path: Path) -> List[Dict[str, str]]:
    payload = yaml.safe_load(yaml_path.read_text())
    return payload.get("segments", [])


def evaluate_segment(
    model,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: int | None,
    memorize_cfg: MemorizeConfig,
) -> tuple[float, float, Dict[str, float]]:
    model.eval()
    total_loss_base = 0.0
    total_loss_mem = 0.0
    total_tokens = 0
    batches = 0
    path_stats: Dict[str, float] = defaultdict(float)
    base_state: Dict[str, torch.Tensor] | None = None
    fast_state = None
    if memorize_cfg.enabled and (not memorize_cfg.use_fast_state) and memorize_cfg.reset:
        base_state = snapshot_state_dict(model)
    for batch in dataloader:
        tokens = batch.to(device)
        with torch.no_grad():
            logits = model(tokens)
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                tokens[:, 1:].reshape(-1),
                reduction="sum",
            )
        total_loss_base += loss.item()
        if memorize_cfg.enabled:
            if memorize_cfg.use_fast_state:
                if fast_state is None or memorize_cfg.reset:
                    if not hasattr(model, "init_fast_state"):
                        raise RuntimeError("Model does not support fast state memorization")
                    fast_state = model.init_fast_state()
                stats = memorize_tokens(model, tokens, memorize_cfg, fast_state=fast_state)
            else:
                stats = memorize_tokens(model, tokens, memorize_cfg)
            for key, value in stats.items():
                path_stats[key] += value
            with torch.no_grad():
                logits_mem = (
                    model(tokens, fast_state=fast_state)
                    if memorize_cfg.use_fast_state
                    else model(tokens)
                )
                loss_mem = torch.nn.functional.cross_entropy(
                    logits_mem[:, :-1].reshape(-1, logits_mem.size(-1)),
                    tokens[:, 1:].reshape(-1),
                    reduction="sum",
                )
            total_loss_mem += loss_mem.item()
            if (not memorize_cfg.use_fast_state) and memorize_cfg.reset and base_state is not None:
                restore_state_dict(model, base_state)
        else:
            total_loss_mem += loss.item()
        total_tokens += tokens[:, 1:].numel()
        batches += 1
        if max_batches and batches >= max_batches:
            break
    base_ce = total_loss_base / total_tokens if total_tokens > 0 else float("nan")
    mem_ce = total_loss_mem / total_tokens if total_tokens > 0 else float("nan")
    return base_ce, mem_ce, path_stats


@app.command()
def main(
    config: Path = typer.Option(..., help="Hydra model config for HOPE."),
    checkpoints: List[Path] = typer.Option(
        ..., help="Ordered list of checkpoints (chronological)."
    ),
    segments_yaml: Path = typer.Option(..., help="YAML describing shard directories per segment."),
    tokenizer_path: Path = typer.Option(..., help="SentencePiece model path (unused for now)."),
    batch_size: int = typer.Option(4, help="Batch size for evaluation."),
    max_batches: int = typer.Option(50, help="Max batches per segment (0 = entire dataset)."),
    device: str = typer.Option("cuda:0" if torch.cuda.is_available() else "cpu"),
    output: Path = typer.Option(Path("eval/continual_results.json")),
    memorize: bool = typer.Option(False, help="Enable memorization while evaluating segments."),
    memorize_steps: int = typer.Option(1, help="Memorization passes per batch."),
    memorize_no_reset: bool = typer.Option(True, help="Keep memory between segments by default."),
    memorize_surprise_threshold: float = typer.Option(
        None, help="Minimum teach-signal norm needed to memorize a batch."
    ),
    memorize_paths: str = typer.Option(
        "all",
        help=(
            "Comma-separated memory paths to update (e.g., 'titan,cms_fast'); "
            "use 'all' for default behavior."
        ),
    ),
) -> None:
    segments = load_segments(segments_yaml)
    if not segments:
        raise typer.BadParameter("No segments found in YAML.")

    cfg = OmegaConf.load(config)
    cfg = unwrap_config(cfg)
    device_obj = resolve_device(device)
    results = []

    if memorize_paths.lower() == "all":
        allowed_paths = None
    else:
        allowed_paths = tuple(path.strip() for path in memorize_paths.split(",") if path.strip())
    memorize_cfg = MemorizeConfig(
        enabled=memorize,
        steps=max(1, memorize_steps),
        reset=not memorize_no_reset,
        use_correct_answer=False,
        surprise_threshold=memorize_surprise_threshold,
        paths=allowed_paths,
    )

    for step_idx, ckpt_path in enumerate(checkpoints):
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model = build_model_from_cfg(cfg.model)
        state_dict = state["model"] if "model" in state else state
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            print(
                "[continual] Warning: state_dict mismatch "
                f"(missing={len(missing)} unexpected={len(unexpected)}) – continuing."
            )
        model = model.to(device_obj)

        segment_losses = {}
        baseline_losses = {}
        segment_stats = {}
        for segment in segments:
            name = segment["name"]
            shards_dir = Path(segment["shards_dir"])
            dataset = TokenShardDataset(shards_dir)
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                collate_fn=collate_batch,
            )
            base_loss, mem_loss, stats = evaluate_segment(
                model,
                loader,
                device_obj,
                None if max_batches <= 0 else max_batches,
                memorize_cfg,
            )
            baseline_losses[name] = base_loss
            segment_losses[name] = mem_loss
            if stats:
                segment_stats[name] = stats

        entry = {"checkpoint": str(ckpt_path), "segment_losses": segment_losses}
        if memorize_cfg.enabled:
            entry["segment_baseline_losses"] = baseline_losses
            entry["segment_memorize_delta"] = {
                name: baseline_losses[name] - segment_losses[name] for name in segment_losses
            }
            if segment_stats:
                entry["memorize_stats"] = segment_stats
        results.append(entry)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2))
    typer.echo(f"[Continual] Saved results to {output}")


if __name__ == "__main__":
    app()
```

### File: `scripts/eval/continual_classification.py`

```python
#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import torch
import typer
from omegaconf import OmegaConf

from nested_learning.continual_classification import (
    ClassificationExample,
    load_banking77,
    load_clinc_oos,
    load_dbpedia14,
)
from nested_learning.continual_streaming import (
    ContinualEvalConfig,
    build_streaming_tasks,
    evaluate_continual_classification,
)
from nested_learning.device import resolve_device
from nested_learning.memorize import MemorizeConfig
from nested_learning.tokenizer import SentencePieceTokenizer
from nested_learning.training import build_model_from_cfg, unwrap_config

app = typer.Typer(
    add_completion=False,
    help="Class-incremental continual-learning harness (CLINC/Banking/DBpedia).",
)


def _load_local_jsonl(path: Path) -> List[ClassificationExample]:
    examples: List[ClassificationExample] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        examples.append(ClassificationExample(text=str(row["text"]), label=str(row["label"])))
    return examples


def _load_examples(
    dataset: str, *, split: str, max_samples: int | None
) -> List[ClassificationExample]:
    dataset = dataset.strip().lower()
    if dataset == "clinc":
        return load_clinc_oos(split=split, max_samples=max_samples).examples
    if dataset == "banking77":
        return load_banking77(split=split, max_samples=max_samples).examples
    if dataset == "dbpedia14":
        return load_dbpedia14(split=split, max_samples=max_samples).examples
    raise typer.BadParameter("dataset must be one of: clinc, banking77, dbpedia14")


@app.command()
def main(
    config: Path = typer.Option(..., help="Hydra model config path."),
    checkpoint: Path = typer.Option(..., help="Checkpoint path."),
    tokenizer_path: Path = typer.Option(..., help="SentencePiece tokenizer path."),
    dataset: str = typer.Option("clinc", help="Dataset: clinc | banking77 | dbpedia14."),
    split: str = typer.Option("test", help="HF split to load."),
    local_jsonl: Path = typer.Option(
        None,
        help="Optional local JSONL (each line: {'text':..., 'label':...}); bypasses HF datasets.",
    ),
    task_size: int = typer.Option(10, help="Number of classes per task."),
    train_per_label: int = typer.Option(25, help="Streaming examples per label."),
    eval_per_label: int = typer.Option(25, help="Eval examples per label."),
    seed: int = typer.Option(0, help="Label/task shuffle seed."),
    task_aware: bool = typer.Option(True, help="Restrict candidates to current task labels."),
    max_samples: int = typer.Option(
        0, help="Max dataset samples (0 = no limit); recommended for smoke runs."
    ),
    device: str = typer.Option("cuda:0" if torch.cuda.is_available() else "cpu"),
    output: Path = typer.Option(Path("eval/continual_classification.json")),
    smoke: bool = typer.Option(False, help="Tiny settings for quick sanity checks."),
    memorize: bool = typer.Option(False, help="Enable test-time memorization during streaming."),
    memorize_steps: int = typer.Option(1, help="Memorization passes per example."),
    memorize_no_reset: bool = typer.Option(
        True, help="Keep memory across examples/tasks by default."
    ),
    memorize_surprise_threshold: float = typer.Option(
        None, help="Minimum teach-signal norm required to trigger memorization."
    ),
    memorize_paths: str = typer.Option(
        "all",
        help=(
            "Comma-separated memory paths to update (e.g., 'titan,cms_fast'); "
            "use 'all' for no restriction."
        ),
    ),
) -> None:
    torch_device = resolve_device(device)
    cfg = OmegaConf.load(config)
    cfg = unwrap_config(cfg)
    model = build_model_from_cfg(cfg.model)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = state["model"] if "model" in state else state
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(
            "[continual_cls] Warning: state_dict mismatch "
            f"(missing={len(missing)} unexpected={len(unexpected)}) – continuing."
        )
    model = model.to(torch_device).eval()
    tokenizer = SentencePieceTokenizer(tokenizer_path)

    resolved_max = None if max_samples <= 0 else int(max_samples)
    if smoke:
        resolved_max = 500
        task_size = min(task_size, 3)
        train_per_label = min(train_per_label, 2)
        eval_per_label = min(eval_per_label, 2)

    if local_jsonl is not None:
        examples = _load_local_jsonl(local_jsonl)
    else:
        examples = _load_examples(dataset, split=split, max_samples=resolved_max)

    eval_cfg = ContinualEvalConfig(
        task_size=task_size,
        seed=seed,
        train_per_label=train_per_label,
        eval_per_label=eval_per_label,
        task_aware=task_aware,
    )
    tasks = build_streaming_tasks(examples, cfg=eval_cfg)

    if memorize_paths.lower() == "all":
        allowed_paths = None
    else:
        allowed_paths = tuple(path.strip() for path in memorize_paths.split(",") if path.strip())
    memorize_cfg = MemorizeConfig(
        enabled=memorize,
        steps=max(1, memorize_steps),
        reset=not memorize_no_reset,
        use_correct_answer=True,
        surprise_threshold=memorize_surprise_threshold,
        paths=allowed_paths,
    )

    result, meta = evaluate_continual_classification(
        model,
        tokenizer,
        tasks,
        torch_device,
        cfg=eval_cfg,
        memorize_cfg=memorize_cfg,
    )
    payload = {
        "dataset": dataset if local_jsonl is None else str(local_jsonl),
        "split": split,
        "config": str(config),
        "checkpoint": str(checkpoint),
        "tokenizer_path": str(tokenizer_path),
        "device": str(torch_device),
        "tasks": [
            {"task_id": t.task_id, "labels": t.labels, "train": len(t.train), "eval": len(t.eval)}
            for t in tasks
        ],
        "result": {
            "avg_accuracy_final": result.avg_accuracy_final,
            "avg_forgetting": result.avg_forgetting,
            "per_task_forgetting": result.per_task_forgetting,
            "task_accuracy_matrix": result.task_accuracy_matrix,
        },
        "meta": meta,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    typer.echo(f"[continual_cls] Saved results to {output}")


if __name__ == "__main__":
    app()
```

### File: `scripts/eval/niah.py`

```python
#!/usr/bin/env python
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import torch
import typer
from omegaconf import OmegaConf
from tqdm import tqdm

from nested_learning.device import resolve_device
from nested_learning.memorize import (
    MemorizeConfig,
    memorize_sequence,
    restore_state_dict,
    snapshot_state_dict,
)
from nested_learning.model import HOPEModel
from nested_learning.tokenizer import SentencePieceTokenizer
from nested_learning.training import build_model_from_cfg, unwrap_config

app = typer.Typer(add_completion=False, help="Needle-in-a-haystack evaluation scaffolding.")


def load_model(config_path: Path, checkpoint: Path, device: torch.device) -> HOPEModel:
    cfg = OmegaConf.load(config_path)
    cfg = unwrap_config(cfg)
    model = build_model_from_cfg(cfg.model)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = state["model"] if "model" in state else state
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(
            "[eval] Warning: state_dict mismatch "
            f"(missing={len(missing)} unexpected={len(unexpected)}) – continuing."
        )
    return model.to(device).eval()


def make_prompt(needle: str, filler_tokens: int) -> str:
    filler_chunks = ["This is filler sentence number {}.".format(i) for i in range(filler_tokens)]
    random.shuffle(filler_chunks)
    haystack = " ".join(filler_chunks)
    prompt = (
        f"{haystack} Remember that the secret key is {needle}. Later you might be asked about it. "
    )
    prompt += "Now answer the question truthfully. What is the secret key? Answer:"
    return prompt


def logprob_answer(
    model: HOPEModel,
    tokenizer: SentencePieceTokenizer,
    prompt: str,
    answer: str,
    device: torch.device,
    *,
    fast_state=None,
) -> float:
    prompt_ids = tokenizer.encode(prompt, add_bos=True)
    answer_ids = tokenizer.encode(" " + answer, add_bos=False)
    inputs = torch.cat([prompt_ids, answer_ids], dim=0).to(device)
    with torch.no_grad():
        logits = (
            model(inputs.unsqueeze(0), fast_state=fast_state)
            if fast_state is not None
            else model(inputs.unsqueeze(0))
        )
        log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
        target = inputs.unsqueeze(0)[:, 1:]
        gathered = log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        prompt_len = prompt_ids.numel()
        answer_logprob = gathered[0, prompt_len - 1 :].sum().item()
    return answer_logprob


@app.command()
def main(
    config: Path = typer.Option(..., help="Hydra config path."),
    checkpoint: Path = typer.Option(..., help="Checkpoint to evaluate."),
    tokenizer_path: Path = typer.Option(..., help="SentencePiece tokenizer path."),
    context_lengths: List[int] = typer.Option([2048, 4096, 8192], help="Context lengths to probe."),
    samples_per_length: int = typer.Option(50, help="Samples per context length."),
    device: str = typer.Option("cuda:0" if torch.cuda.is_available() else "cpu"),
    output: Path = typer.Option(Path("eval/niah_results.json")),
    memorize: bool = typer.Option(False, help="Enable test-time memorization for each prompt."),
    memorize_steps: int = typer.Option(1, help="Memorization passes per prompt."),
    memorize_use_correct_answer: bool = typer.Option(
        False, help="Include correct key when memorizing."
    ),
    memorize_no_reset: bool = typer.Option(False, help="Retain memory between samples."),
    memorize_surprise_threshold: float = typer.Option(
        None, help="Minimum teach-signal norm required to trigger memorization."
    ),
    memorize_paths: str = typer.Option(
        "all",
        help=(
            "Comma-separated memory paths to update (e.g., 'titan,cms_fast'); "
            "use 'all' for no restriction."
        ),
    ),
) -> None:
    torch_device = resolve_device(device)
    model = load_model(config, checkpoint, torch_device)
    tokenizer = SentencePieceTokenizer(tokenizer_path)
    if memorize_paths.lower() == "all":
        allowed_paths = None
    else:
        allowed_paths = tuple(path.strip() for path in memorize_paths.split(",") if path.strip())
    memorize_cfg = MemorizeConfig(
        enabled=memorize,
        steps=max(1, memorize_steps),
        reset=not memorize_no_reset,
        use_correct_answer=memorize_use_correct_answer,
        surprise_threshold=memorize_surprise_threshold,
        paths=allowed_paths,
    )
    base_state: Dict[str, torch.Tensor] | None = None
    fast_state = None
    results = {}
    path_stats: Dict[str, float] = defaultdict(float)
    for length in context_lengths:
        correct_base = 0
        correct_mem = 0
        for _ in tqdm(range(samples_per_length), desc=f"NIAH@{length}"):
            needle = f"KEY-{random.randint(1000, 9999)}"
            prompt = make_prompt(needle, filler_tokens=max(1, length // 128))
            distractor = f"KEY-{random.randint(1000, 9999)}"
            logprob_true_base = logprob_answer(model, tokenizer, prompt, needle, torch_device)
            logprob_false_base = logprob_answer(model, tokenizer, prompt, distractor, torch_device)
            correct_base += int(logprob_true_base > logprob_false_base)
            if memorize_cfg.enabled:
                memorize_text = prompt
                if memorize_cfg.use_correct_answer:
                    memorize_text = f"{prompt} {needle}"
                if memorize_cfg.use_fast_state:
                    if fast_state is None or memorize_cfg.reset:
                        if not hasattr(model, "init_fast_state"):
                            raise RuntimeError("Model does not support fast state memorization")
                        fast_state = model.init_fast_state()
                    stats = memorize_sequence(
                        model,
                        tokenizer,
                        memorize_text,
                        torch_device,
                        memorize_cfg,
                        fast_state=fast_state,
                    )
                    for key, value in stats.items():
                        path_stats[key] += value
                    logprob_true_mem = logprob_answer(
                        model, tokenizer, prompt, needle, torch_device, fast_state=fast_state
                    )
                    logprob_false_mem = logprob_answer(
                        model, tokenizer, prompt, distractor, torch_device, fast_state=fast_state
                    )
                    correct_mem += int(logprob_true_mem > logprob_false_mem)
                else:
                    if memorize_cfg.reset and base_state is None:
                        base_state = snapshot_state_dict(model)
                    stats = memorize_sequence(
                        model, tokenizer, memorize_text, torch_device, memorize_cfg
                    )
                    for key, value in stats.items():
                        path_stats[key] += value
                    logprob_true_mem = logprob_answer(
                        model, tokenizer, prompt, needle, torch_device
                    )
                    logprob_false_mem = logprob_answer(
                        model, tokenizer, prompt, distractor, torch_device
                    )
                    correct_mem += int(logprob_true_mem > logprob_false_mem)
                    if memorize_cfg.reset and base_state is not None:
                        restore_state_dict(model, base_state)
            else:
                correct_mem += int(logprob_true_base > logprob_false_base)
        base_acc = correct_base / samples_per_length if samples_per_length else 0.0
        mem_acc = correct_mem / samples_per_length if samples_per_length else 0.0
        results[f"niah_{length}"] = mem_acc
        if memorize_cfg.enabled:
            results[f"niah_{length}_baseline_accuracy"] = base_acc
            results[f"niah_{length}_memorize_accuracy"] = mem_acc
            results[f"niah_{length}_memorize_delta"] = mem_acc - base_acc
    if memorize_cfg.enabled:
        for key, value in path_stats.items():
            results[f"niah_{key}"] = value
        results["niah_memorize_paths"] = (
            "all" if memorize_cfg.paths is None else ",".join(memorize_cfg.paths)
        )
        if memorize_cfg.surprise_threshold is not None:
            results["niah_memorize_surprise_threshold"] = memorize_cfg.surprise_threshold
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2))
    typer.echo(f"[Eval] Saved NIAH metrics to {output}")


if __name__ == "__main__":
    app()
```

### File: `scripts/eval/niah_suite.py`

```python
#!/usr/bin/env python
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import torch
import typer
from omegaconf import OmegaConf
from tqdm import tqdm

from nested_learning.device import resolve_device
from nested_learning.memorize import (
    MemorizeConfig,
    memorize_sequence,
    restore_state_dict,
    snapshot_state_dict,
)
from nested_learning.tokenizer import SentencePieceTokenizer
from nested_learning.training import build_model_from_cfg, unwrap_config

app = typer.Typer(add_completion=False, help="RULER-ish NIAH suite (multiple retrieval variants).")


def load_model(config_path: Path, checkpoint: Path, device: torch.device):
    cfg = OmegaConf.load(config_path)
    cfg = unwrap_config(cfg)
    model = build_model_from_cfg(cfg.model)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = state["model"] if "model" in state else state
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(
            "[niah_suite] Warning: state_dict mismatch "
            f"(missing={len(missing)} unexpected={len(unexpected)}) – continuing."
        )
    return model.to(device).eval()


def _logprob_answer(
    model,
    tokenizer: SentencePieceTokenizer,
    prompt: str,
    answer: str,
    device: torch.device,
    *,
    fast_state=None,
) -> float:
    prompt_ids = tokenizer.encode(prompt, add_bos=True)
    answer_ids = tokenizer.encode(" " + answer, add_bos=False)
    inputs = torch.cat([prompt_ids, answer_ids], dim=0).to(device)
    with torch.no_grad():
        logits = (
            model(inputs.unsqueeze(0), fast_state=fast_state)
            if fast_state is not None
            else model(inputs.unsqueeze(0))
        )
        log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
        target = inputs.unsqueeze(0)[:, 1:]
        gathered = log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        prompt_len = prompt_ids.numel()
        return float(gathered[0, prompt_len - 1 :].sum().item())


def _filler_sentences(count: int) -> List[str]:
    return [f"This is filler sentence number {idx}." for idx in range(count)]


def _ensure_prompt_length(
    tokenizer: SentencePieceTokenizer,
    *,
    base_lines: List[str],
    target_tokens: int,
    rng: random.Random,
    max_filler: int = 50_000,
) -> str:
    filler = []
    filler_count = max(1, target_tokens // 32)
    while True:
        filler = _filler_sentences(filler_count)
        rng.shuffle(filler)
        prompt = "\n".join([*filler, *base_lines])
        token_len = int(tokenizer.encode(prompt, add_bos=True).numel())
        if token_len >= target_tokens:
            return prompt
        if filler_count >= max_filler:
            return prompt
        missing = target_tokens - token_len
        filler_count += max(1, missing // 16)


@dataclass(frozen=True)
class VariantCase:
    prompt: str
    answer: str
    distractor: str


def _case_single_needle(rng: random.Random) -> VariantCase:
    needle = f"KEY-{rng.randint(1000, 9999)}"
    prompt_lines = [
        f"Remember that the secret key is {needle}.",
        "Later you might be asked about it.",
        "Question: What is the secret key?",
        "Answer:",
    ]
    distractor = f"KEY-{rng.randint(1000, 9999)}"
    return VariantCase(prompt="\n".join(prompt_lines), answer=needle, distractor=distractor)


def _case_multi_needle(rng: random.Random, *, needles: int) -> VariantCase:
    keys = [f"KEY-{rng.randint(1000, 9999)}" for _ in range(max(2, needles))]
    query_idx = rng.randrange(len(keys))
    prompt_lines = ["Memorize the following secret keys:"]
    for idx, key in enumerate(keys, start=1):
        prompt_lines.append(f"Key {idx}: {key}.")
    prompt_lines.extend(
        [
            f"Question: What is Key {query_idx + 1}?",
            "Answer:",
        ]
    )
    distractor = f"KEY-{rng.randint(1000, 9999)}"
    return VariantCase(
        prompt="\n".join(prompt_lines), answer=keys[query_idx], distractor=distractor
    )


def _case_kv_single(rng: random.Random) -> VariantCase:
    key = f"ITEM-{rng.randint(100, 999)}"
    value = f"VALUE-{rng.randint(1000, 9999)}"
    prompt_lines = [
        "Memorize this key-value pair:",
        f"{key} -> {value}.",
        f"Question: What is the value for {key}?",
        "Answer:",
    ]
    distractor = f"VALUE-{rng.randint(1000, 9999)}"
    return VariantCase(prompt="\n".join(prompt_lines), answer=value, distractor=distractor)


def _case_kv_multi(rng: random.Random, *, pairs: int) -> VariantCase:
    pairs = max(2, pairs)
    keys = [f"ITEM-{rng.randint(100, 999)}" for _ in range(pairs)]
    values = [f"VALUE-{rng.randint(1000, 9999)}" for _ in range(pairs)]
    query_idx = rng.randrange(pairs)
    prompt_lines = ["Memorize the following key-value pairs:"]
    for k, v in zip(keys, values, strict=True):
        prompt_lines.append(f"{k} -> {v}.")
    prompt_lines.extend(
        [
            f"Question: What is the value for {keys[query_idx]}?",
            "Answer:",
        ]
    )
    distractor = f"VALUE-{rng.randint(1000, 9999)}"
    return VariantCase(
        prompt="\n".join(prompt_lines), answer=values[query_idx], distractor=distractor
    )


def _case_positioned_needle(rng: random.Random, *, position: str) -> VariantCase:
    needle = f"KEY-{rng.randint(1000, 9999)}"
    prompt_lines = [
        f"Remember that the secret key is {needle}.",
        "Question: What is the secret key?",
        "Answer:",
    ]
    distractor = f"KEY-{rng.randint(1000, 9999)}"
    return VariantCase(prompt="\n".join(prompt_lines), answer=needle, distractor=distractor)


def _variant_cases(rng: random.Random, *, variant: str) -> VariantCase:
    if variant == "single_needle":
        return _case_single_needle(rng)
    if variant == "multi_needle":
        return _case_multi_needle(rng, needles=4)
    if variant == "kv_single":
        return _case_kv_single(rng)
    if variant == "kv_multi":
        return _case_kv_multi(rng, pairs=6)
    if variant in {"needle_early", "needle_mid", "needle_late"}:
        pos = variant.split("_", 1)[1]
        return _case_positioned_needle(rng, position=pos)
    raise ValueError(f"Unknown variant: {variant}")


def _evaluate_variant(
    model,
    tokenizer: SentencePieceTokenizer,
    device: torch.device,
    *,
    variant: str,
    context_tokens: int,
    samples: int,
    rng: random.Random,
    memorize_cfg: MemorizeConfig,
) -> Dict[str, Any]:
    base_state: Dict[str, torch.Tensor] | None = None
    fast_state = None
    if memorize_cfg.enabled and (not memorize_cfg.use_fast_state) and memorize_cfg.reset:
        base_state = snapshot_state_dict(model)

    correct_base = 0
    correct_mem = 0
    path_stats: Dict[str, float] = {}
    for _ in tqdm(range(samples), desc=f"{variant}@{context_tokens}"):
        case = _variant_cases(rng, variant=variant)
        if variant in {"needle_early", "needle_mid", "needle_late"}:
            memory_line, question_line, answer_line = case.prompt.split("\n", 2)
            if variant == "needle_early":
                ratio = 0.1
            elif variant == "needle_late":
                ratio = 0.9
            else:
                ratio = 0.5
            filler_count = max(1, context_tokens // 32)
            while True:
                filler = _filler_sentences(filler_count)
                rng.shuffle(filler)
                insert_at = int(ratio * max(1, len(filler)))
                insert_at = max(0, min(insert_at, len(filler)))
                with_memory = filler[:insert_at] + [memory_line] + filler[insert_at:]
                prompt = "\n".join([*with_memory, question_line, answer_line])
                token_len = int(tokenizer.encode(prompt, add_bos=True).numel())
                if token_len >= context_tokens:
                    break
                filler_count += max(1, (context_tokens - token_len) // 16)
        else:
            prompt = _ensure_prompt_length(
                tokenizer,
                base_lines=[case.prompt],
                target_tokens=context_tokens,
                rng=rng,
            )
        lp_true_base = _logprob_answer(
            model, tokenizer, prompt, case.answer, device, fast_state=fast_state
        )
        lp_false_base = _logprob_answer(
            model, tokenizer, prompt, case.distractor, device, fast_state=fast_state
        )
        correct_base += int(lp_true_base > lp_false_base)
        if memorize_cfg.enabled:
            memorize_text = (
                prompt if not memorize_cfg.use_correct_answer else f"{prompt} {case.answer}"
            )
            if memorize_cfg.use_fast_state:
                if fast_state is None or memorize_cfg.reset:
                    if not hasattr(model, "init_fast_state"):
                        raise RuntimeError("Model does not support fast state memorization")
                    fast_state = model.init_fast_state()
                stats = memorize_sequence(
                    model, tokenizer, memorize_text, device, memorize_cfg, fast_state=fast_state
                )
                for k, v in stats.items():
                    path_stats[k] = path_stats.get(k, 0.0) + v
                lp_true_mem = _logprob_answer(
                    model, tokenizer, prompt, case.answer, device, fast_state=fast_state
                )
                lp_false_mem = _logprob_answer(
                    model, tokenizer, prompt, case.distractor, device, fast_state=fast_state
                )
                correct_mem += int(lp_true_mem > lp_false_mem)
            else:
                stats = memorize_sequence(model, tokenizer, memorize_text, device, memorize_cfg)
                for k, v in stats.items():
                    path_stats[k] = path_stats.get(k, 0.0) + v
                lp_true_mem = _logprob_answer(model, tokenizer, prompt, case.answer, device)
                lp_false_mem = _logprob_answer(model, tokenizer, prompt, case.distractor, device)
                correct_mem += int(lp_true_mem > lp_false_mem)
                if memorize_cfg.reset and base_state is not None:
                    restore_state_dict(model, base_state)
        else:
            correct_mem += int(lp_true_base > lp_false_base)

    base_acc = correct_base / samples if samples else 0.0
    mem_acc = correct_mem / samples if samples else 0.0
    payload: Dict[str, Any] = {
        "variant": variant,
        "context_tokens": context_tokens,
        "samples": samples,
        "baseline_accuracy": base_acc,
        "memorize_accuracy": mem_acc,
        "memorize_delta": mem_acc - base_acc,
    }
    if memorize_cfg.enabled:
        payload["memorize_paths"] = (
            "all" if memorize_cfg.paths is None else ",".join(memorize_cfg.paths)
        )
        payload["memorize_use_correct_answer"] = bool(memorize_cfg.use_correct_answer)
        if memorize_cfg.surprise_threshold is not None:
            payload["memorize_surprise_threshold"] = memorize_cfg.surprise_threshold
        if path_stats:
            payload["memorize_stats"] = path_stats
    return payload


@app.command()
def main(
    config: Path = typer.Option(..., help="Hydra config path."),
    checkpoint: Path = typer.Option(..., help="Checkpoint to evaluate."),
    tokenizer_path: Path = typer.Option(..., help="SentencePiece tokenizer path."),
    context_tokens: List[int] = typer.Option(
        [2048, 4096, 8192], help="Target prompt token lengths."
    ),
    samples_per_length: int = typer.Option(50, help="Samples per (variant, length)."),
    variants: List[str] = typer.Option(
        [
            "single_needle",
            "multi_needle",
            "kv_single",
            "kv_multi",
            "needle_early",
            "needle_mid",
            "needle_late",
        ],
        help="Variant names to run.",
    ),
    seed: int = typer.Option(0, help="Random seed."),
    device: str = typer.Option("cuda:0" if torch.cuda.is_available() else "cpu"),
    output: Path = typer.Option(Path("eval/niah_suite_results.json")),
    smoke: bool = typer.Option(False, help="Tiny settings for quick sanity checks."),
    memorize: bool = typer.Option(False, help="Enable test-time memorization for each prompt."),
    memorize_steps: int = typer.Option(1, help="Memorization passes per prompt."),
    memorize_use_correct_answer: bool = typer.Option(
        False, help="Append ground truth during memorization."
    ),
    memorize_no_reset: bool = typer.Option(False, help="Retain memory between samples."),
    memorize_surprise_threshold: float = typer.Option(
        None, help="Minimum teach-signal norm required to trigger memorization."
    ),
    memorize_paths: str = typer.Option(
        "all",
        help=(
            "Comma-separated memory paths to update (e.g., 'titan,cms_fast'); "
            "use 'all' for no restriction."
        ),
    ),
) -> None:
    rng = random.Random(seed)
    torch_device = resolve_device(device)
    model = load_model(config, checkpoint, torch_device)
    tokenizer = SentencePieceTokenizer(tokenizer_path)

    if smoke:
        context_tokens = [256]
        samples_per_length = min(samples_per_length, 8)
        variants = ["single_needle", "kv_single"]

    if memorize_paths.lower() == "all":
        allowed_paths = None
    else:
        allowed_paths = tuple(path.strip() for path in memorize_paths.split(",") if path.strip())
    memorize_cfg = MemorizeConfig(
        enabled=memorize,
        steps=max(1, memorize_steps),
        reset=not memorize_no_reset,
        use_correct_answer=memorize_use_correct_answer,
        surprise_threshold=memorize_surprise_threshold,
        paths=allowed_paths,
    )

    results: List[Dict[str, Any]] = []
    for variant in variants:
        for length in context_tokens:
            results.append(
                _evaluate_variant(
                    model,
                    tokenizer,
                    torch_device,
                    variant=variant,
                    context_tokens=length,
                    samples=samples_per_length,
                    rng=rng,
                    memorize_cfg=memorize_cfg,
                )
            )

    payload = {
        "seed": seed,
        "device": str(torch_device),
        "config": str(config),
        "checkpoint": str(checkpoint),
        "tokenizer_path": str(tokenizer_path),
        "variants": variants,
        "context_tokens": context_tokens,
        "samples_per_length": samples_per_length,
        "memorize": {
            "enabled": memorize_cfg.enabled,
            "steps": memorize_cfg.steps,
            "reset": memorize_cfg.reset,
            "use_correct_answer": bool(memorize_cfg.use_correct_answer),
            "paths": "all" if memorize_cfg.paths is None else ",".join(memorize_cfg.paths),
            "surprise_threshold": memorize_cfg.surprise_threshold,
        },
        "results": results,
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    typer.echo(f"[niah_suite] Saved results to {output}")


if __name__ == "__main__":
    app()
```

### File: `scripts/eval/passkey.py`

```python
#!/usr/bin/env python
from __future__ import annotations

import json
import random
from pathlib import Path

import torch
import typer
from omegaconf import OmegaConf

from nested_learning.device import resolve_device
from nested_learning.memorize import (
    MemorizeConfig,
    memorize_sequence,
    restore_state_dict,
    snapshot_state_dict,
)
from nested_learning.tokenizer import SentencePieceTokenizer
from nested_learning.training import build_model_from_cfg, unwrap_config

app = typer.Typer(add_completion=False, help="Synthetic passkey evaluation (LongBench-style).")

PROMPT_TEMPLATE = (
    "{filler}\nRemember that the passkey for this document is {key}. "
    "Later we will ask about it.\nQuestion: What is the passkey?\nAnswer:"
)


def load_model(config: Path, checkpoint: Path, device: torch.device):
    cfg = OmegaConf.load(config)
    cfg = unwrap_config(cfg)
    model = build_model_from_cfg(cfg.model)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = state["model"] if "model" in state else state
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[passkey] Warning: mismatch missing={len(missing)} unexpected={len(unexpected)}")
    return model.to(device).eval()


def make_prompt(context_tokens: int, key: str) -> str:
    sentences = [f"This is filler sentence number {idx}." for idx in range(context_tokens)]
    random.shuffle(sentences)
    filler = " ".join(sentences)
    return PROMPT_TEMPLATE.format(filler=filler, key=key)


def logprob(
    model, tokenizer, prompt: str, answer: str, device: torch.device, *, fast_state=None
) -> float:
    prompt_ids = tokenizer.encode(prompt, add_bos=True)
    answer_ids = tokenizer.encode(" " + answer, add_bos=False, add_eos=True)
    tokens = torch.cat([prompt_ids, answer_ids], dim=0).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tokens, fast_state=fast_state) if fast_state is not None else model(tokens)
        log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
        targets = tokens[:, 1:]
        gathered = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        prompt_len = prompt_ids.numel()
        return gathered[:, prompt_len - 1 :].sum().item()


@app.command()
def main(
    config: Path = typer.Option(..., help="Hydra model config."),
    checkpoint: Path = typer.Option(..., help="Checkpoint path."),
    tokenizer_path: Path = typer.Option(..., help="SentencePiece tokenizer."),
    samples: int = typer.Option(64, help="Number of synthetic prompts."),
    filler_sentences: int = typer.Option(200, help="Number of filler sentences (controls length)."),
    device: str = typer.Option("cuda:0" if torch.cuda.is_available() else "cpu"),
    output: Path = typer.Option(Path("eval/passkey_results.json")),
    memorize: bool = typer.Option(False, help="Enable memorization before answering."),
    memorize_steps: int = typer.Option(1, help="Memorization iterations."),
    memorize_no_reset: bool = typer.Option(False, help="Retain memory between prompts."),
    memorize_surprise_threshold: float = typer.Option(
        None, help="Minimum teach-signal norm required before memorizing a prompt."
    ),
    memorize_paths: str = typer.Option(
        "all",
        help=(
            "Comma-separated memory paths to update (e.g., 'titan,cms_fast'); "
            "use 'all' for unrestricted paths."
        ),
    ),
) -> None:
    torch_device = resolve_device(device)
    model = load_model(config, checkpoint, torch_device)
    tokenizer = SentencePieceTokenizer(tokenizer_path)
    if memorize_paths.lower() == "all":
        allowed_paths = None
    else:
        allowed_paths = tuple(path.strip() for path in memorize_paths.split(",") if path.strip())
    memorize_cfg = MemorizeConfig(
        enabled=memorize,
        steps=max(1, memorize_steps),
        reset=not memorize_no_reset,
        use_correct_answer=True,
        surprise_threshold=memorize_surprise_threshold,
        paths=allowed_paths,
    )
    base_state = (
        snapshot_state_dict(model)
        if memorize_cfg.enabled and (not memorize_cfg.use_fast_state) and memorize_cfg.reset
        else None
    )
    fast_state = None
    correct_base = 0
    correct_mem = 0
    path_stats: dict[str, float] = {}
    for _ in range(samples):
        key = f"PASSKEY-{random.randint(1000, 9999)}"
        prompt = make_prompt(filler_sentences, key)
        distractor = f"PASSKEY-{random.randint(1000, 9999)}"
        lp_true = logprob(model, tokenizer, prompt, key, torch_device)
        lp_false = logprob(model, tokenizer, prompt, distractor, torch_device)
        correct_base += int(lp_true > lp_false)
        if memorize_cfg.enabled:
            if memorize_cfg.use_fast_state:
                if fast_state is None or memorize_cfg.reset:
                    if not hasattr(model, "init_fast_state"):
                        raise RuntimeError("Model does not support fast state memorization")
                    fast_state = model.init_fast_state()
                stats = memorize_sequence(
                    model, tokenizer, prompt, torch_device, memorize_cfg, fast_state=fast_state
                )
                for k, v in stats.items():
                    path_stats[k] = path_stats.get(k, 0.0) + v
                lp_true_mem = logprob(
                    model, tokenizer, prompt, key, torch_device, fast_state=fast_state
                )
                lp_false_mem = logprob(
                    model, tokenizer, prompt, distractor, torch_device, fast_state=fast_state
                )
                correct_mem += int(lp_true_mem > lp_false_mem)
            else:
                stats = memorize_sequence(model, tokenizer, prompt, torch_device, memorize_cfg)
                for k, v in stats.items():
                    path_stats[k] = path_stats.get(k, 0.0) + v
                lp_true_mem = logprob(model, tokenizer, prompt, key, torch_device)
                lp_false_mem = logprob(model, tokenizer, prompt, distractor, torch_device)
                correct_mem += int(lp_true_mem > lp_false_mem)
                if memorize_cfg.reset and base_state is not None:
                    restore_state_dict(model, base_state)
        else:
            correct_mem += int(lp_true > lp_false)
    base_acc = correct_base / samples if samples else 0.0
    mem_acc = correct_mem / samples if samples else 0.0
    result = {
        "samples": samples,
        "filler_sentences": filler_sentences,
        "accuracy_base": base_acc,
        "accuracy_memorize": mem_acc,
        "accuracy_delta": mem_acc - base_acc,
        "path_stats": path_stats,
    }
    if memorize_cfg.enabled:
        result["memorize_paths"] = (
            "all" if memorize_cfg.paths is None else ",".join(memorize_cfg.paths)
        )
        if memorize_cfg.surprise_threshold is not None:
            result["memorize_surprise_threshold"] = memorize_cfg.surprise_threshold
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    typer.echo(f"[passkey] Saved results to {output}")


if __name__ == "__main__":
    app()
```

### File: `scripts/eval/pg19_perplexity.py`

```python
#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F
import typer
from datasets import load_dataset
from omegaconf import OmegaConf

from nested_learning.device import resolve_device
from nested_learning.memorize import (
    MemorizeConfig,
    memorize_sequence,
    restore_state_dict,
    snapshot_state_dict,
)
from nested_learning.tokenizer import SentencePieceTokenizer
from nested_learning.training import build_model_from_cfg, unwrap_config

app = typer.Typer(add_completion=False, help="Compute PG-19 perplexity for a checkpoint.")


def load_model(config: Path, checkpoint: Path, device: torch.device):
    cfg = OmegaConf.load(config)
    cfg = unwrap_config(cfg)
    model = build_model_from_cfg(cfg.model)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = state["model"] if "model" in state else state
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[pg19] Warning: mismatch missing={len(missing)} unexpected={len(unexpected)}")
    return model.to(device).eval()


def _nll_for_text(
    model,
    tokenizer,
    text: str,
    device: torch.device,
    max_seq: int,
    *,
    fast_state=None,
) -> tuple[float, int] | None:
    tokens = tokenizer.encode(text, add_bos=True, add_eos=True)
    if tokens.size(0) < 2:
        return None
    if tokens.size(0) > max_seq:
        tokens = tokens[:max_seq]
    tokens = tokens.to(device).unsqueeze(0)
    with torch.no_grad():
        logits = model(tokens, fast_state=fast_state) if fast_state is not None else model(tokens)
        log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
        targets = tokens[:, 1:]
        gathered = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        return -gathered.sum().item(), targets.numel()


@app.command()
def main(
    config: Path = typer.Option(..., help="Hydra model config."),
    checkpoint: Path = typer.Option(..., help="Checkpoint path."),
    tokenizer_path: Path = typer.Option(..., help="SentencePiece model path."),
    max_samples: int = typer.Option(64, help="Number of PG-19 samples."),
    device: str = typer.Option("cuda:0" if torch.cuda.is_available() else "cpu"),
    output: Path = typer.Option(Path("eval/pg19_perplexity.json")),
    context_tokens: int = typer.Option(
        2048, help="Truncate text to this many tokens before scoring."
    ),
    memorize: bool = typer.Option(False, help="Apply test-time memorization to each excerpt."),
    memorize_steps: int = typer.Option(1, help="Memorization passes per excerpt."),
    memorize_no_reset: bool = typer.Option(False, help="Retain memory between excerpts."),
    memorize_paths: str = typer.Option(
        "all",
        help="Comma-separated memory paths to update during memorization (e.g., 'titan,cms_fast').",
    ),
    memorize_surprise_threshold: float = typer.Option(
        None, help="Minimum teach-signal norm required before memorizing an excerpt."
    ),
) -> None:
    torch_device = resolve_device(device)
    model = load_model(config, checkpoint, torch_device)
    tokenizer = SentencePieceTokenizer(tokenizer_path)
    dataset = load_dataset("pg19", split="test", streaming=True, trust_remote_code=True).shuffle(
        seed=42
    )
    total_tokens = 0
    total_nll_base = 0.0
    total_nll_mem = 0.0
    if memorize_paths.lower() == "all":
        allowed_paths = None
    else:
        allowed_paths = tuple(path.strip() for path in memorize_paths.split(",") if path.strip())
    memorize_cfg = MemorizeConfig(
        enabled=memorize,
        steps=max(1, memorize_steps),
        reset=not memorize_no_reset,
        use_correct_answer=False,
        surprise_threshold=memorize_surprise_threshold,
        paths=allowed_paths,
    )
    base_state = (
        snapshot_state_dict(model)
        if memorize_cfg.enabled and (not memorize_cfg.use_fast_state) and memorize_cfg.reset
        else None
    )
    fast_state = None
    processed = 0
    for idx, sample in enumerate(dataset):
        if idx >= max_samples:
            break
        text = sample.get("text") or sample.get("passage")
        if not text:
            continue
        nll_tot = _nll_for_text(model, tokenizer, text, torch_device, context_tokens)
        if nll_tot is None:
            continue
        nll_base, tokens_seen = nll_tot
        total_nll_base += nll_base
        total_tokens += tokens_seen
        if memorize_cfg.enabled:
            if memorize_cfg.use_fast_state:
                if fast_state is None or memorize_cfg.reset:
                    if not hasattr(model, "init_fast_state"):
                        raise RuntimeError("Model does not support fast state memorization")
                    fast_state = model.init_fast_state()
                memorize_sequence(
                    model, tokenizer, text[:1024], torch_device, memorize_cfg, fast_state=fast_state
                )
                nll_mem = _nll_for_text(
                    model, tokenizer, text, torch_device, context_tokens, fast_state=fast_state
                )
                if nll_mem is not None:
                    total_nll_mem += nll_mem[0]
            else:
                memorize_sequence(model, tokenizer, text[:1024], torch_device, memorize_cfg)
                nll_mem = _nll_for_text(model, tokenizer, text, torch_device, context_tokens)
                if nll_mem is not None:
                    total_nll_mem += nll_mem[0]
                if memorize_cfg.reset and base_state is not None:
                    restore_state_dict(model, base_state)
        else:
            total_nll_mem += nll_base
        processed += 1
    ppl_base = float(torch.exp(torch.tensor(total_nll_base / max(1, total_tokens))))
    ppl_mem = float(torch.exp(torch.tensor(total_nll_mem / max(1, total_tokens))))
    payload = {
        "samples": processed,
        "tokens": total_tokens,
        "ppl_base": ppl_base,
        "ppl_memorize": ppl_mem,
        "ppl_delta": ppl_base - ppl_mem,
    }
    if memorize_cfg.enabled:
        payload["memorize_paths"] = (
            "all" if memorize_cfg.paths is None else ",".join(memorize_cfg.paths)
        )
        payload["memorize_surprise_threshold"] = memorize_cfg.surprise_threshold
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2))
    typer.echo(f"[pg19] Saved perplexity to {output}")


if __name__ == "__main__":
    app()
```

### File: `scripts/eval/phase2_memorization_delta_smoke.py`

```python
#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path

import torch
import typer

from nested_learning.device import resolve_device
from nested_learning.levels import LevelSpec
from nested_learning.memorize import MemorizeConfig, memorize_tokens
from nested_learning.model import HOPEModel, ModelConfig

app = typer.Typer(
    add_completion=False,
    help=(
        "CPU-friendly smoke: show HOPE-Attention adapts via CMS updates while Transformer does not."
    ),
)


def _build_model(*, variant: str, vocab_size: int, dim: int, layers: int, heads: int) -> HOPEModel:
    titan = LevelSpec(name="titan", update_period=1, optimizer_key="titan_opt")
    cms = (LevelSpec(name="cms_fast", update_period=1, optimizer_key="cms_opt"),)
    cfg = ModelConfig(
        vocab_size=vocab_size,
        dim=dim,
        num_layers=layers,
        heads=heads,
        titan_level=titan,
        cms_levels=cms,
        optimizers=None,
        teach_scale=0.1,
        block_variant=variant,
    )
    return HOPEModel(cfg).eval()


def _run_once(
    *,
    variant: str,
    tokens: torch.Tensor,
    seed: int,
) -> dict:
    torch.manual_seed(seed)
    model = _build_model(
        variant=variant,
        vocab_size=int(tokens.max().item() + 1),
        dim=16,
        layers=1,
        heads=4,
    ).to(tokens.device)
    fast_state = model.init_fast_state()
    with torch.no_grad():
        before = model(tokens, fast_state=fast_state).detach()
    cfg = MemorizeConfig(enabled=True, steps=1, use_fast_state=True, paths=("cms_fast",))
    stats = memorize_tokens(model, tokens, cfg, fast_state=fast_state)
    with torch.no_grad():
        after = model(tokens, fast_state=fast_state).detach()
    return {
        "delta_mean_abs": float((after - before).abs().mean().item()),
        "outputs_identical": bool(torch.allclose(before, after, atol=0.0, rtol=0.0)),
        "cms_fast_update_events": float(stats.get("cms_fast_update_events", 0.0)),
        "cms_fast_updates": float(stats.get("cms_fast_updates", 0.0)),
        "titan_update_events": float(stats.get("titan_update_events", 0.0)),
    }


@app.command()
def main(
    seed: int = typer.Option(0, help="Torch RNG seed (affects weights)."),
    vocab_size: int = typer.Option(32, help="Synthetic vocab size."),
    seq_len: int = typer.Option(16, help="Token sequence length."),
    batch_size: int = typer.Option(1, help="Batch size."),
    device: str = typer.Option("cpu", help="cpu or cuda:<idx>."),
    output: Path = typer.Option(
        Path("eval/phase2_memorization_delta_smoke.json"), help="Where to write results."
    ),
) -> None:
    torch_device = resolve_device(device)
    token_gen = torch.Generator(device="cpu").manual_seed(1337)
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len), generator=token_gen).to(
        torch_device
    )
    results = {
        "seed": int(seed),
        "vocab_size": int(vocab_size),
        "seq_len": int(seq_len),
        "batch_size": int(batch_size),
        "hope_attention": _run_once(variant="hope_attention", tokens=tokens, seed=seed),
        "transformer": _run_once(variant="transformer", tokens=tokens, seed=seed),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2))
    typer.echo(f"[phase2] wrote {output}")


if __name__ == "__main__":
    app()
```

### File: `scripts/eval/plot_continual_classification.py`

```python
#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import typer

app = typer.Typer(
    add_completion=False, help="Plot continual classification task matrix + forgetting bars."
)


@app.command()
def main(
    continual_json: Path = typer.Option(
        ..., help="Output JSON from scripts/eval/continual_classification.py"
    ),
    output: Path = typer.Option(Path("reports/plots/continual_classification.png")),
    title: str = typer.Option("Continual Classification", help="Plot title"),
) -> None:
    payload = json.loads(continual_json.read_text())
    tasks = payload.get("tasks", [])
    matrix = payload.get("result", {}).get("task_accuracy_matrix", [])
    forgetting = payload.get("result", {}).get("per_task_forgetting", [])

    task_ids: List[str] = [str(t.get("task_id", idx)) for idx, t in enumerate(tasks)]
    data = np.array(matrix, dtype=np.float32)
    mask = np.isnan(data)
    masked = np.ma.array(data, mask=mask)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [2, 1]})
    im = ax0.imshow(masked, vmin=0.0, vmax=1.0, cmap="viridis")
    ax0.set_title(f"{title} – Task Accuracy Matrix")
    ax0.set_xlabel("After Task")
    ax0.set_ylabel("Eval Task")
    ax0.set_xticks(range(len(task_ids)))
    ax0.set_yticks(range(len(task_ids)))
    ax0.set_xticklabels(task_ids, rotation=90)
    ax0.set_yticklabels(task_ids)
    fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.04, label="Accuracy")

    f = (
        np.array(forgetting, dtype=np.float32)
        if forgetting
        else np.zeros((len(task_ids),), dtype=np.float32)
    )
    ax1.bar(range(len(task_ids)), f)
    ax1.set_title("Forgetting per Task")
    ax1.set_xlabel("Task")
    ax1.set_ylabel("Max - Final Acc")
    ax1.set_xticks(range(len(task_ids)))
    ax1.set_xticklabels(task_ids, rotation=90)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    typer.echo(f"[plot] Wrote {output}")


if __name__ == "__main__":
    app()
```

### File: `scripts/eval/plot_forgetting.py`

```python
#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import typer

app = typer.Typer(add_completion=False, help="Plot continual-learning forgetting curves.")


@app.command()
def main(
    continual_json: Path = typer.Option(..., help="Path to eval/continual_*.json output."),
    output: Path = typer.Option(Path("reports/plots/continual_forgetting.png")),
    segment: str = typer.Option(None, help="Specific segment to plot (default: all)."),
) -> None:
    data = json.loads(continual_json.read_text())
    checkpoints = []
    baseline = []
    memorize = []
    for entry in data:
        checkpoints.append(entry.get("checkpoint"))
        seg_losses = entry.get("segment_losses", {})
        base_losses = entry.get("segment_baseline_losses", seg_losses)
        key = segment or next(iter(seg_losses))
        baseline.append(base_losses.get(key))
        memorize.append(seg_losses.get(key))
    plt.figure(figsize=(8, 4))
    plt.plot(checkpoints, baseline, label="baseline CE", marker="o")
    plt.plot(checkpoints, memorize, label="memorize CE", marker="o")
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Cross-entropy")
    plt.title(f"Continual forgetting ({segment or 'default segment'})")
    plt.legend()
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output)
    typer.echo(f"[plot] Saved plot to {output}")


if __name__ == "__main__":
    app()
```

### File: `scripts/eval/plot_niah_suite.py`

```python
#!/usr/bin/env python
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import typer

app = typer.Typer(add_completion=False, help="Plot NIAH suite accuracy vs context length.")


@app.command()
def main(
    niah_suite_json: Path = typer.Option(..., help="Output JSON from scripts/eval/niah_suite.py"),
    output: Path = typer.Option(Path("reports/plots/niah_suite.png")),
    title: str = typer.Option("NIAH Suite", help="Plot title"),
) -> None:
    payload = json.loads(niah_suite_json.read_text())
    results = payload.get("results", [])
    grouped: Dict[str, List[Tuple[int, float, float]]] = defaultdict(list)
    for row in results:
        variant = str(row.get("variant", "unknown"))
        length = int(row.get("context_tokens", 0))
        base = float(row.get("baseline_accuracy", 0.0))
        mem = float(row.get("memorize_accuracy", base))
        grouped[variant].append((length, base, mem))

    variants = sorted(grouped.keys())
    ncols = 2
    nrows = (len(variants) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, max(3, 3 * nrows)), squeeze=False)
    axes_flat = axes.flatten()
    for ax, variant in zip(axes_flat, variants, strict=False):
        series = sorted(grouped[variant], key=lambda t: t[0])
        xs = [t[0] for t in series]
        base = [t[1] for t in series]
        mem = [t[2] for t in series]
        ax.plot(xs, base, label="baseline")
        ax.plot(xs, mem, label="memorize")
        ax.set_title(variant)
        ax.set_xlabel("context_tokens")
        ax.set_ylabel("accuracy")
        ax.set_ylim(0.0, 1.0)
        ax.legend()

    for ax in axes_flat[len(variants) :]:
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)
    typer.echo(f"[plot] Wrote {output}")


if __name__ == "__main__":
    app()
```

### File: `scripts/eval/run_pilot_suite.sh`

```bash
#!/usr/bin/env bash
#
# Convenience wrapper to run the Stage 2 evaluation suite (zero-shot, NIAH, continual)
# on the pilot HOPE checkpoint and optional TITAN baseline.
#
# Environment variables (override as needed):
#   HOPE_CONFIG          (default configs/pilot.yaml)
#   HOPE_CHECKPOINT      (default artifacts/checkpoints/pilot/step_latest.pt)
#   TITAN_CONFIG         (optional)
#   TITAN_CHECKPOINT     (optional)
#   TOKENIZER_PATH       (default artifacts/tokenizer/refinedweb_mix/spm_32000_unigram.model)
#   DEVICE               (default cuda:1)
#   MAX_SAMPLES          (default 256 for zero-shot)
#   NIAH_CONTEXTS        (space-separated list, default "2048 4096 8192 16384 32768 65536")
#   NIAH_SAMPLES         (default 8 per context)
#   CONT_BATCH           (default 4)
#   CONT_MAX_BATCHES     (default 20)

set -euo pipefail

HOPE_CONFIG=${HOPE_CONFIG:-configs/pilot.yaml}
TOKENIZER_PATH=${TOKENIZER_PATH:-artifacts/tokenizer/refinedweb_mix/spm_32000_unigram.model}
DEVICE=${DEVICE:-cuda:1}
MAX_SAMPLES=${MAX_SAMPLES:-256}
NIAH_CONTEXTS=${NIAH_CONTEXTS:-"2048 4096 8192 16384 32768 65536"}
NIAH_SAMPLES=${NIAH_SAMPLES:-8}
CONT_BATCH=${CONT_BATCH:-4}
CONT_MAX_BATCHES=${CONT_MAX_BATCHES:-20}
SEGMENTS_YAML=${SEGMENTS_YAML:-configs/data/continual_segments_sample.yaml}
HOPE_CONT_CHECKPOINTS=${HOPE_CONT_CHECKPOINTS:-}
PASSKEY_SAMPLES=${PASSKEY_SAMPLES:-64}
PASSKEY_FILLER=${PASSKEY_FILLER:-256}
PG19_SAMPLES=${PG19_SAMPLES:-32}
CONT_PLOT_SEGMENT=${CONT_PLOT_SEGMENT:-refinedweb_2018}
MEMORIZE_PATHS=${MEMORIZE_PATHS:-titan,cms_fast}
HOPE_MEMORIZE_PATHS=${HOPE_MEMORIZE_PATHS:-${MEMORIZE_PATHS}}
TITAN_MEMORIZE_PATHS=${TITAN_MEMORIZE_PATHS:-titan}
MEMORIZE_SURPRISE_THRESHOLD=${MEMORIZE_SURPRISE_THRESHOLD:-0.02}

resolve_checkpoint() {
  local path="$1"
  if [[ -n "${path}" ]]; then
    echo "${path}"
    return
  fi
  local latest
  latest=$(ls -1t artifacts/checkpoints/pilot/step_*.pt 2>/dev/null | head -n 1 || true)
  if [[ -z "${latest}" ]]; then
    echo ""
  else
    echo "${latest}"
  fi
}

HOPE_CHECKPOINT=${HOPE_CHECKPOINT:-$(resolve_checkpoint "")}
if [[ -z "${HOPE_CHECKPOINT}" ]]; then
  echo "[eval] No HOPE checkpoint supplied and none found under artifacts/checkpoints/pilot."
  exit 1
fi
if [[ -z "${HOPE_CONT_CHECKPOINTS}" ]]; then
  HOPE_CONT_CHECKPOINTS="${HOPE_CHECKPOINT}"
fi

mkdir -p eval
IFS=' ' read -r -a HOPE_CONT_LIST <<< "${HOPE_CONT_CHECKPOINTS}"

run_zero_shot() {
  local config=$1
  local ckpt=$2
  local tag=$3
  local memorize_paths=$4
  UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run python scripts/eval/zeroshot.py \
    --config "${config}" \
    --checkpoint "${ckpt}" \
    --tokenizer-path "${TOKENIZER_PATH}" \
    --tasks all \
    --max-samples "${MAX_SAMPLES}" \
    --device "${DEVICE}" \
    --output "eval/zeroshot_${tag}.json" \
    --memorize \
    --memorize-steps 2 \
    --memorize-use-correct-answer \
    --memorize-paths "${memorize_paths}" \
    --memorize-surprise-threshold "${MEMORIZE_SURPRISE_THRESHOLD}"
}

run_niah() {
  local config=$1
  local ckpt=$2
  local tag=$3
  local memorize_paths=$4
  local args=()
  for ctx in ${NIAH_CONTEXTS}; do
    args+=(--context-lengths "${ctx}")
  done
  UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run python scripts/eval/niah.py \
    --config "${config}" \
    --checkpoint "${ckpt}" \
    --tokenizer-path "${TOKENIZER_PATH}" \
    "${args[@]}" \
    --samples-per-length "${NIAH_SAMPLES}" \
    --device "${DEVICE}" \
    --output "eval/niah_${tag}.json" \
    --memorize \
    --memorize-steps 2 \
    --memorize-use-correct-answer \
    --memorize-paths "${memorize_paths}" \
    --memorize-surprise-threshold "${MEMORIZE_SURPRISE_THRESHOLD}"
}

run_continual() {
  local config=$1
  local tag=$2
  local memorize_paths=$3
  shift 3
  local ckpts=("$@")
  if [[ ${#ckpts[@]} -eq 0 ]]; then
    echo "[eval] No checkpoints provided for continual eval (${tag}); skipping."
    return
  fi
  UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run python scripts/eval/continual.py \
    --config "${config}" \
    --checkpoints "${ckpts[@]}" \
    --segments-yaml "${SEGMENTS_YAML}" \
    --tokenizer-path "${TOKENIZER_PATH}" \
    --batch-size "${CONT_BATCH}" \
    --max-batches "${CONT_MAX_BATCHES}" \
    --device "${DEVICE}" \
    --output "eval/continual_${tag}.json" \
    --memorize \
    --memorize-steps 1 \
    --memorize-paths "${memorize_paths}" \
    --memorize-surprise-threshold "${MEMORIZE_SURPRISE_THRESHOLD}"
  if [[ ${#ckpts[@]} -gt 1 ]]; then
    local plot_target="reports/plots/continual_${tag}_${CONT_PLOT_SEGMENT}.png"
    mkdir -p reports/plots
    UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run python scripts/eval/plot_forgetting.py \
      --continual-json "eval/continual_${tag}.json" \
      --segment "${CONT_PLOT_SEGMENT}" \
      --output "${plot_target}"
    echo "[eval] Forgetting plot saved to ${plot_target}"
  fi
}

run_passkey() {
  local config=$1
  local ckpt=$2
  local tag=$3
  local memorize_paths=$4
  UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run python scripts/eval/passkey.py \
    --config "${config}" \
    --checkpoint "${ckpt}" \
    --tokenizer-path "${TOKENIZER_PATH}" \
    --samples "${PASSKEY_SAMPLES}" \
    --filler-sentences "${PASSKEY_FILLER}" \
    --device "${DEVICE}" \
    --output "eval/passkey_${tag}.json" \
    --memorize \
    --memorize-steps 2 \
    --memorize-paths "${memorize_paths}" \
    --memorize-surprise-threshold "${MEMORIZE_SURPRISE_THRESHOLD}"
}

run_pg19() {
  local config=$1
  local ckpt=$2
  local tag=$3
  local memorize_paths=$4
  UV_CACHE_DIR=/tmp/uv-cache UV_LINK_MODE=copy uv run python scripts/eval/pg19_perplexity.py \
    --config "${config}" \
    --checkpoint "${ckpt}" \
    --tokenizer-path "${TOKENIZER_PATH}" \
    --max-samples "${PG19_SAMPLES}" \
    --device "${DEVICE}" \
    --output "eval/pg19_${tag}.json" \
    --memorize \
    --memorize-paths "${memorize_paths}" \
    --memorize-surprise-threshold "${MEMORIZE_SURPRISE_THRESHOLD}"
}

echo "[eval] Running suite for HOPE (${HOPE_CHECKPOINT})"
run_zero_shot "${HOPE_CONFIG}" "${HOPE_CHECKPOINT}" "pilot" "${HOPE_MEMORIZE_PATHS}"
run_niah "${HOPE_CONFIG}" "${HOPE_CHECKPOINT}" "pilot" "${HOPE_MEMORIZE_PATHS}"
run_continual "${HOPE_CONFIG}" "pilot" "${HOPE_MEMORIZE_PATHS}" "${HOPE_CONT_LIST[@]}"
run_passkey "${HOPE_CONFIG}" "${HOPE_CHECKPOINT}" "pilot" "${HOPE_MEMORIZE_PATHS}"
run_pg19 "${HOPE_CONFIG}" "${HOPE_CHECKPOINT}" "pilot" "${HOPE_MEMORIZE_PATHS}"

if [[ -n "${TITAN_CONFIG:-}" && -n "${TITAN_CHECKPOINT:-}" ]]; then
  echo "[eval] Running suite for TITAN baseline (${TITAN_CHECKPOINT})"
  run_zero_shot "${TITAN_CONFIG}" "${TITAN_CHECKPOINT}" "titan" "${TITAN_MEMORIZE_PATHS}"
  run_niah "${TITAN_CONFIG}" "${TITAN_CHECKPOINT}" "titan" "${TITAN_MEMORIZE_PATHS}"
  IFS=' ' read -r -a TITAN_CONT_LIST <<< "${TITAN_CHECKPOINTS:-$TITAN_CHECKPOINT}"
  run_continual "${TITAN_CONFIG}" "titan" "${TITAN_MEMORIZE_PATHS}" "${TITAN_CONT_LIST[@]}"
  run_passkey "${TITAN_CONFIG}" "${TITAN_CHECKPOINT}" "titan" "${TITAN_MEMORIZE_PATHS}"
  run_pg19 "${TITAN_CONFIG}" "${TITAN_CHECKPOINT}" "titan" "${TITAN_MEMORIZE_PATHS}"
else
  echo "[eval] TITAN baseline skipped (set TITAN_CONFIG and TITAN_CHECKPOINT to enable)."
fi

echo "[eval] Pilot suite complete. Outputs saved under eval/."
```

### File: `scripts/eval/summarize_eval.py`

```python
#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import typer

app = typer.Typer(add_completion=False, help="Summarize eval JSONs into a small markdown table.")


def _flatten_numeric(obj: Any, *, prefix: str = "") -> Dict[str, float]:
    out: Dict[str, float] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.update(_flatten_numeric(v, prefix=key))
        return out
    if isinstance(obj, list):
        # Avoid exploding large lists; only summarize scalar numeric lists.
        if obj and all(isinstance(v, (int, float)) for v in obj):
            out[prefix] = float(sum(float(v) for v in obj) / len(obj))
        return out
    if isinstance(obj, (int, float)):
        out[prefix] = float(obj)
    return out


def _expand_keys(flat: Dict[str, float], keys: Iterable[str]) -> List[str]:
    resolved: List[str] = []
    for key in keys:
        key = key.strip()
        if not key:
            continue
        if key.endswith("*"):
            prefix = key[:-1]
            matches = sorted(k for k in flat.keys() if k.startswith(prefix))
            resolved.extend(matches)
        else:
            resolved.append(key)
    # De-duplicate while preserving order.
    seen = set()
    ordered: List[str] = []
    for k in resolved:
        if k in seen:
            continue
        seen.add(k)
        ordered.append(k)
    return ordered


def _render_table(rows: List[Tuple[str, Dict[str, float]]], keys: List[str]) -> str:
    header = ["file", *keys]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for name, flat in rows:
        cells = [name]
        for key in keys:
            value = flat.get(key)
            if value is None:
                cells.append("")
            else:
                cells.append(f"{value:.6g}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


@app.command()
def main(
    inputs: List[Path] = typer.Option(..., help="Eval JSON files to summarize."),
    keys: List[str] = typer.Option(
        [],
        help=(
            "Dotted numeric keys to include (supports '*' suffix prefix expansion). "
            "If omitted, uses a small default set."
        ),
    ),
    output: Path = typer.Option(Path("eval/summary.md"), help="Markdown output path."),
) -> None:
    rows: List[Tuple[str, Dict[str, float]]] = []
    for path in inputs:
        payload = json.loads(path.read_text())
        flat = _flatten_numeric(payload)
        rows.append((path.name, flat))

    if not rows:
        raise typer.BadParameter("No input files provided.")

    if not keys:
        # Reasonable defaults across our eval scripts.
        keys = [
            "accuracy",
            "accuracy_base",
            "accuracy_memorize",
            "accuracy_delta",
            "avg_accuracy_final",
            "avg_forgetting",
        ]

    expanded = _expand_keys(rows[0][1], keys)
    for _name, flat in rows[1:]:
        expanded = sorted(set(expanded) | set(_expand_keys(flat, keys)))

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_render_table(rows, expanded))
    typer.echo(f"[summary] Wrote {output}")


if __name__ == "__main__":
    app()
```

### File: `scripts/eval/zeroshot.py`

```python
#!/usr/bin/env python
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import torch
import typer
from datasets import load_dataset
from omegaconf import OmegaConf
from tqdm import tqdm

from nested_learning.device import resolve_device
from nested_learning.memorize import (
    MemorizeConfig,
    memorize_sequence,
    restore_state_dict,
    snapshot_state_dict,
)
from nested_learning.tokenizer import SentencePieceTokenizer
from nested_learning.training import build_model_from_cfg, unwrap_config

app = typer.Typer(add_completion=False, help="Zero-shot evaluation harness for HOPE.")
HF_DATASET_KWARGS = {"trust_remote_code": True}


def load_model(config_path: Path, checkpoint: Path, device: torch.device):
    cfg = OmegaConf.load(config_path)
    cfg = unwrap_config(cfg)
    model = build_model_from_cfg(cfg.model)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state_dict = state["model"] if "model" in state else state
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(
            "[eval] Warning: state_dict mismatch "
            f"(missing={len(missing)} unexpected={len(unexpected)}) – continuing."
        )
    return model.to(device).eval()


def score_text(
    model, tokenizer: SentencePieceTokenizer, text: str, device: torch.device, *, fast_state=None
) -> float:
    tokens = tokenizer.encode(text)
    tokens = tokens.to(device)
    with torch.no_grad():
        logits = (
            model(tokens.unsqueeze(0), fast_state=fast_state)
            if fast_state is not None
            else model(tokens.unsqueeze(0))
        )
        log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
        target = tokens.unsqueeze(0)[:, 1:]
        gathered = log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        return gathered.sum().item()


def evaluate_multiple_choice(
    task_name: str,
    dataset_iter: Iterable[dict],
    build_texts_fn: Callable[[dict], Tuple[str, List[str], int]],
    tokenizer: SentencePieceTokenizer,
    model,
    device: torch.device,
    max_samples: int | None,
    memorize_cfg: MemorizeConfig,
) -> Dict[str, float]:
    correct_mem = 0
    correct_base = 0
    total = 0
    base_state: Dict[str, torch.Tensor] | None = None
    fast_state = None
    path_stats: Dict[str, float] = defaultdict(float)
    for sample in tqdm(dataset_iter, desc=task_name.upper()):
        prompt, texts, answer_idx = build_texts_fn(sample)
        scores_base = [score_text(model, tokenizer, t, device) for t in texts]
        pred_base = int(max(range(len(scores_base)), key=lambda i: scores_base[i]))
        correct_base += int(pred_base == answer_idx)
        if memorize_cfg.enabled:
            memorize_text = prompt
            if memorize_cfg.use_correct_answer:
                memorize_text = f"{prompt} {texts[answer_idx]}".strip()
            if memorize_cfg.use_fast_state:
                if fast_state is None or memorize_cfg.reset:
                    if not hasattr(model, "init_fast_state"):
                        raise RuntimeError("Model does not support fast state memorization")
                    fast_state = model.init_fast_state()
                stats = memorize_sequence(
                    model, tokenizer, memorize_text, device, memorize_cfg, fast_state=fast_state
                )
                for key, value in stats.items():
                    path_stats[key] += value
                scores_eval = [
                    score_text(model, tokenizer, t, device, fast_state=fast_state) for t in texts
                ]
                pred_eval = int(max(range(len(scores_eval)), key=lambda i: scores_eval[i]))
                correct_mem += int(pred_eval == answer_idx)
            else:
                if memorize_cfg.reset and base_state is None:
                    base_state = snapshot_state_dict(model)
                stats = memorize_sequence(model, tokenizer, memorize_text, device, memorize_cfg)
                for key, value in stats.items():
                    path_stats[key] += value
                scores_eval = [score_text(model, tokenizer, t, device) for t in texts]
                pred_eval = int(max(range(len(scores_eval)), key=lambda i: scores_eval[i]))
                correct_mem += int(pred_eval == answer_idx)
        else:
            correct_mem += int(pred_base == answer_idx)
        total += 1
        if (
            memorize_cfg.enabled
            and (not memorize_cfg.use_fast_state)
            and memorize_cfg.reset
            and base_state is not None
        ):
            restore_state_dict(model, base_state)
        if max_samples and total >= max_samples:
            break
    accuracy = correct_mem / total if total else 0.0
    result: Dict[str, float] = {f"{task_name}_accuracy": accuracy, f"{task_name}_samples": total}
    if memorize_cfg.enabled:
        baseline_acc = correct_base / total if total else 0.0
        result[f"{task_name}_baseline_accuracy"] = baseline_acc
        result[f"{task_name}_memorize_accuracy"] = accuracy
        result[f"{task_name}_memorize_delta"] = accuracy - baseline_acc
        if memorize_cfg.paths is None:
            result[f"{task_name}_memorize_paths"] = "all"
        else:
            result[f"{task_name}_memorize_paths"] = ",".join(memorize_cfg.paths)
        if memorize_cfg.surprise_threshold is not None:
            result[f"{task_name}_memorize_surprise_threshold"] = memorize_cfg.surprise_threshold
        for key, value in path_stats.items():
            result[f"{task_name}_{key}"] = value
    return result


def build_piqa_texts(sample: dict) -> Tuple[str, List[str], int]:
    prompt = sample["goal"].strip()
    options = [sample["sol1"].strip(), sample["sol2"].strip()]
    texts = [f"{prompt} {opt}" for opt in options]
    target = sample["label"]
    return prompt, texts, target


def eval_piqa(model, tokenizer, device, max_samples, memorize_cfg):
    dataset = load_dataset("piqa", split="validation", **HF_DATASET_KWARGS)
    return evaluate_multiple_choice(
        "piqa", dataset, build_piqa_texts, tokenizer, model, device, max_samples, memorize_cfg
    )


def build_hellaswag_texts(sample: dict) -> Tuple[str, List[str], int]:
    prompt = f"{sample['ctx_a'].strip()} {sample['ctx_b'].strip()}".strip()
    endings = [ending.strip() for ending in sample["endings"]]
    texts = [f"{prompt} {ending}" for ending in endings]
    label = sample["label"]
    target = int(label) if not isinstance(label, int) else label
    return prompt, texts, target


def eval_hellaswag(model, tokenizer, device, max_samples, memorize_cfg):
    dataset = load_dataset("hellaswag", split="validation", **HF_DATASET_KWARGS)
    return evaluate_multiple_choice(
        "hellaswag",
        dataset,
        build_hellaswag_texts,
        tokenizer,
        model,
        device,
        max_samples,
        memorize_cfg,
    )


def build_winogrande_texts(sample: dict) -> Tuple[str, List[str], int]:
    sentence = sample["sentence"]
    options = [sample["option1"].strip(), sample["option2"].strip()]
    texts = [sentence.replace("_", opt) for opt in options]
    target = int(sample["answer"]) - 1
    return sentence, texts, target


def eval_winogrande(model, tokenizer, device, max_samples, memorize_cfg):
    dataset = load_dataset("winogrande", "winogrande_xl", split="validation", **HF_DATASET_KWARGS)
    return evaluate_multiple_choice(
        "winogrande",
        dataset,
        build_winogrande_texts,
        tokenizer,
        model,
        device,
        max_samples,
        memorize_cfg,
    )


def build_arc_texts(sample: dict) -> Tuple[str, List[str], int]:
    prompt = sample["question"].strip()
    choice_texts = sample["choices"]["text"]
    labels = sample["choices"]["label"]
    texts = [f"{prompt} {choice.strip()}" for choice in choice_texts]
    target = labels.index(sample["answerKey"])
    return prompt, texts, target


def eval_arc(
    model, tokenizer, device, max_samples, difficulty: str, memorize_cfg: MemorizeConfig
) -> Dict[str, float]:
    dataset = load_dataset("ai2_arc", difficulty, split="validation", **HF_DATASET_KWARGS)
    return evaluate_multiple_choice(
        f"arc_{difficulty.lower()}",
        dataset,
        build_arc_texts,
        tokenizer,
        model,
        device,
        max_samples,
        memorize_cfg,
    )


def build_boolq_texts(sample: dict) -> Tuple[str, List[str], int]:
    prompt = f"{sample['passage'].strip()}\nQuestion: {sample['question'].strip()}\nAnswer:"
    texts = [f"{prompt} yes", f"{prompt} no"]
    target = 0 if sample["answer"] else 1
    return prompt, texts, target


def eval_boolq(model, tokenizer, device, max_samples, memorize_cfg):
    dataset = load_dataset("boolq", split="validation", **HF_DATASET_KWARGS)
    return evaluate_multiple_choice(
        "boolq", dataset, build_boolq_texts, tokenizer, model, device, max_samples, memorize_cfg
    )


def build_siqa_texts(sample: dict) -> Tuple[str, List[str], int]:
    prompt = f"Context: {sample['context'].strip()} Question: {sample['question'].strip()} Answer:"
    options = [sample["answerA"].strip(), sample["answerB"].strip(), sample["answerC"].strip()]
    texts = [f"{prompt} {opt}" for opt in options]
    target = int(sample["label"]) - 1
    return prompt, texts, target


def eval_siqa(model, tokenizer, device, max_samples, memorize_cfg):
    dataset = load_dataset("social_i_qa", split="validation", **HF_DATASET_KWARGS)
    return evaluate_multiple_choice(
        "siqa", dataset, build_siqa_texts, tokenizer, model, device, max_samples, memorize_cfg
    )


def build_commonsenseqa_texts(sample: dict) -> Tuple[str, List[str], int]:
    prompt = sample["question"].strip()
    choice_texts = sample["choices"]["text"]
    labels = sample["choices"]["label"]
    texts = [f"{prompt} {choice.strip()}" for choice in choice_texts]
    target = labels.index(sample["answerKey"])
    return prompt, texts, target


def eval_commonsenseqa(model, tokenizer, device, max_samples, memorize_cfg):
    dataset = load_dataset("commonsense_qa", split="validation", **HF_DATASET_KWARGS)
    return evaluate_multiple_choice(
        "commonsenseqa",
        dataset,
        build_commonsenseqa_texts,
        tokenizer,
        model,
        device,
        max_samples,
        memorize_cfg,
    )


def build_openbookqa_texts(sample: dict) -> Tuple[str, List[str], int]:
    prompt = sample["question_stem"].strip()
    choice_texts = sample["choices"]["text"]
    labels = sample["choices"]["label"]
    texts = [f"{prompt} {choice.strip()}" for choice in choice_texts]
    target = labels.index(sample["answerKey"])
    return prompt, texts, target


def eval_openbookqa(model, tokenizer, device, max_samples, memorize_cfg):
    dataset = load_dataset("openbookqa", "main", split="validation", **HF_DATASET_KWARGS)
    return evaluate_multiple_choice(
        "openbookqa",
        dataset,
        build_openbookqa_texts,
        tokenizer,
        model,
        device,
        max_samples,
        memorize_cfg,
    )


TASK_EVALUATORS = {
    "piqa": eval_piqa,
    "hellaswag": eval_hellaswag,
    "winogrande": eval_winogrande,
    "arc_easy": lambda model, tok, dev, n, mem: eval_arc(model, tok, dev, n, "ARC-Easy", mem),
    "arc_challenge": lambda model, tok, dev, n, mem: eval_arc(
        model, tok, dev, n, "ARC-Challenge", mem
    ),
    "boolq": eval_boolq,
    "siqa": eval_siqa,
    "commonsenseqa": eval_commonsenseqa,
    "openbookqa": eval_openbookqa,
}


@app.command()
def main(
    config: Path = typer.Option(..., help="Hydra model config path."),
    checkpoint: Path = typer.Option(..., help="Checkpoint file (state dict)."),
    tokenizer_path: Path = typer.Option(..., help="SentencePiece model path."),
    tasks: str = typer.Option("piqa", help="Comma-separated list of tasks or 'all'."),
    max_samples: int = typer.Option(500, help="Max samples per task (0 = entire split)."),
    output: Path = typer.Option(Path("eval/zeroshot_results.json"), help="Output JSON file."),
    device: str = typer.Option(
        "cuda:0" if torch.cuda.is_available() else "cpu", help="Device to run eval on."
    ),
    list_tasks: bool = typer.Option(False, "--list-tasks", help="List available tasks and exit."),
    memorize: bool = typer.Option(False, help="Enable test-time memorization updates."),
    memorize_steps: int = typer.Option(1, help="Number of memorize passes per sample."),
    memorize_use_correct_answer: bool = typer.Option(
        False, help="When memorizing, include the correct answer text (for ablations)."
    ),
    memorize_no_reset: bool = typer.Option(
        False, help="If set, retain memorization across samples."
    ),
    memorize_surprise_threshold: float = typer.Option(
        None, help="Minimum teach-signal norm required before applying memorization."
    ),
    memorize_paths: str = typer.Option(
        "all",
        help=(
            "Comma-separated memory paths to update (e.g., 'titan,cms_fast'); "
            "use 'all' to allow every path."
        ),
    ),
) -> None:
    available = list(TASK_EVALUATORS.keys())
    if list_tasks:
        typer.echo("Available tasks: " + ", ".join(available))
        raise typer.Exit(0)

    selected_tasks = (
        available if tasks.lower() == "all" else [t.strip().lower() for t in tasks.split(",")]
    )
    torch_device = resolve_device(device)
    model = load_model(config, checkpoint, torch_device)
    tokenizer = SentencePieceTokenizer(tokenizer_path)
    if memorize_paths.lower() == "all":
        allowed_paths = None
    else:
        allowed_paths = tuple(path.strip() for path in memorize_paths.split(",") if path.strip())
    memorize_cfg = MemorizeConfig(
        enabled=memorize,
        steps=max(1, memorize_steps),
        reset=not memorize_no_reset,
        use_correct_answer=memorize_use_correct_answer,
        surprise_threshold=memorize_surprise_threshold,
        paths=allowed_paths,
    )

    results: Dict[str, float] = {}
    for task in selected_tasks:
        evaluator = TASK_EVALUATORS.get(task)
        if evaluator is None:
            raise ValueError(f"Unsupported task '{task}'. Valid tasks: {available}")
        metrics = evaluator(
            model,
            tokenizer,
            torch_device,
            None if max_samples <= 0 else max_samples,
            memorize_cfg,
        )
        results.update(metrics)

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2))
    typer.echo(f"[Eval] Saved metrics for tasks {selected_tasks} -> {output}")


if __name__ == "__main__":
    app()
```

### File: `scripts/package_pilot_release.sh`

```bash
#!/usr/bin/env bash
#
# Bundle the latest pilot checkpoint + metadata into artifacts/pilot_release/.
# Usage:
#   scripts/package_pilot_release.sh [hope_checkpoint_path] [titan_checkpoint_path]
# If no path is provided, the newest file under artifacts/checkpoints/pilot is used.

set -euo pipefail

RELEASE_DIR="artifacts/pilot_release"
CHECKPOINT_DIR="artifacts/checkpoints/pilot"
CONFIG_PATH="configs/pilot.yaml"
LOG_PATTERNS=( "logs/pilot_train*.log" "logs/pilot_train*.json" "logs/pilot_relaunch*.log" "logs/pilot_relaunch*.json" )
METADATA_PATH="${RELEASE_DIR}/metadata.json"
MANIFEST_PATH="${RELEASE_DIR}/MANIFEST.txt"
EVAL_PATTERNS=( "eval/*_pilot.json" "eval/*_titan.json" )
PLOT_PATTERNS=( "reports/plots/continual_pilot_*.png" "reports/plots/continual_titan_*.png" )

mkdir -p "${RELEASE_DIR}"

copy_sidecars() {
  local ckpt_path="$1"
  local dest_prefix="$2"
  local src_prefix="${ckpt_path%.pt}"
  local exts=("sha256" "meta.json" "yaml")
  for ext in "${exts[@]}"; do
    local src="${src_prefix}.${ext}"
    if [[ -f "${src}" ]]; then
      cp "${src}" "${dest_prefix}.${ext}"
    fi
  done
}

copy_patterns() {
  local dest_dir="$1"
  shift
  mkdir -p "${dest_dir}"
  shopt -s nullglob
  for pattern in "$@"; do
    for path in ${pattern}; do
      cp "${path}" "${dest_dir}/"
    done
  done
  shopt -u nullglob
}

if [[ $# -ge 1 ]]; then
  HOPE_CHECKPOINT="$1"
else
  HOPE_CHECKPOINT=$(ls -1t ${CHECKPOINT_DIR}/step_*.pt 2>/dev/null | head -n 1 || true)
fi

TITAN_CHECKPOINT="${2:-}"

if [[ -z "${HOPE_CHECKPOINT}" ]]; then
  echo "[package] No checkpoint found. Pass the path explicitly or ensure ${CHECKPOINT_DIR}/step_*.pt exists."
  exit 1
fi

HOPE_CHECKPOINT_BASENAME=$(basename "${HOPE_CHECKPOINT}")
DEST_CKPT="${RELEASE_DIR}/checkpoint.pt"
cp "${HOPE_CHECKPOINT}" "${DEST_CKPT}"
copy_sidecars "${HOPE_CHECKPOINT}" "${RELEASE_DIR}/checkpoint"

# Copy config snapshot
cp "${CONFIG_PATH}" "${RELEASE_DIR}/config.yaml"

# Copy relevant logs (if they exist)
LOG_DEST="${RELEASE_DIR}/logs"
mkdir -p "${LOG_DEST}"
shopt -s nullglob
for pattern in "${LOG_PATTERNS[@]}"; do
  for log_path in ${pattern}; do
    cp "${log_path}" "${LOG_DEST}/"
  done
done
shopt -u nullglob

# Copy latest eval outputs / plots if present.
copy_patterns "${RELEASE_DIR}" "${EVAL_PATTERNS[@]}"
copy_patterns "${RELEASE_DIR}/plots" "${PLOT_PATTERNS[@]}"

TITAN_RELEASE_BASENAME=""
if [[ -n "${TITAN_CHECKPOINT}" ]]; then
  if [[ ! -f "${TITAN_CHECKPOINT}" ]]; then
    echo "[package] TITAN checkpoint not found: ${TITAN_CHECKPOINT}"
    exit 1
  fi
  TITAN_BASENAME=$(basename "${TITAN_CHECKPOINT}")
  TITAN_RELEASE_BASENAME="titan_${TITAN_BASENAME}"
  cp "${TITAN_CHECKPOINT}" "${RELEASE_DIR}/${TITAN_RELEASE_BASENAME}"
  copy_sidecars "${TITAN_CHECKPOINT}" "${RELEASE_DIR}/titan_${TITAN_BASENAME%.pt}"
fi

# Update metadata stub with checkpoint information if present
if [[ -f "${METADATA_PATH}" ]]; then
  python - "$HOPE_CHECKPOINT_BASENAME" "$TITAN_RELEASE_BASENAME" "$METADATA_PATH" <<'PY' || true
import json, sys, pathlib
ckpt = sys.argv[1]
titan = sys.argv[2]
path = pathlib.Path(sys.argv[3])
meta = json.loads(path.read_text())
meta["checkpoint_step"] = ckpt
if titan:
    meta["titan_checkpoint_step"] = titan
path.write_text(json.dumps(meta, indent=2))
PY
fi

# Emit manifest with quick reference info
{
  echo "Pilot Release Manifest"
  echo "======================"
  echo "HOPE Checkpoint: ${HOPE_CHECKPOINT_BASENAME}"
  if [[ -n "${TITAN_RELEASE_BASENAME}" ]]; then
    echo "TITAN Checkpoint: ${TITAN_RELEASE_BASENAME}"
  fi
  echo "Config: ${CONFIG_PATH}"
  echo "Logs copied from patterns: ${LOG_PATTERNS[*]}"
  echo "Eval copied from patterns: ${EVAL_PATTERNS[*]}"
  echo "Plots copied from patterns: ${PLOT_PATTERNS[*]}"
  date "+Packaged at: %Y-%m-%d %H:%M:%S"
} > "${MANIFEST_PATH}"

echo "[package] Release bundle updated:"
echo "  - ${DEST_CKPT}"
echo "  - ${RELEASE_DIR}/checkpoint.* (sidecars, when available)"
if [[ -n "${TITAN_RELEASE_BASENAME}" ]]; then
  echo "  - ${RELEASE_DIR}/${TITAN_RELEASE_BASENAME}"
  echo "  - ${RELEASE_DIR}/titan_${TITAN_BASENAME%.pt}.* (sidecars, when available)"
fi
echo "  - ${RELEASE_DIR}/config.yaml"
echo "  - ${LOG_DEST}/"
echo "  - ${RELEASE_DIR}/*_pilot.json and ${RELEASE_DIR}/*_titan.json (when available)"
echo "  - ${RELEASE_DIR}/plots/ (when available)"
echo "  - ${METADATA_PATH} (if present)"
```

### File: `scripts/run_cpu_ddp_smoke.sh`

```bash
#!/usr/bin/env bash

set -euo pipefail

# Force CPU execution so torchrun selects the gloo backend.
export CUDA_VISIBLE_DEVICES=""

uv run torchrun --standalone --nproc_per_node=2 train_dist.py --config-name pilot_smoke "$@"
```

### File: `scripts/run_e2e_smoke.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

DEVICE=${DEVICE:-cpu}
TRAIN_CONFIG=${TRAIN_CONFIG:-pilot_smoke}
DEFAULT_MODEL_CONFIG="configs/${TRAIN_CONFIG}.yaml"
if [[ ! -f "${DEFAULT_MODEL_CONFIG}" ]]; then
  DEFAULT_MODEL_CONFIG="configs/hope/pilot.yaml"
fi
MODEL_CONFIG=${MODEL_CONFIG:-${DEFAULT_MODEL_CONFIG}}
TOKENIZER_PATH=${TOKENIZER_PATH:-artifacts/tokenizer/refinedweb_mix/spm_32000_unigram.model}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-artifacts/checkpoints/${TRAIN_CONFIG}}
LOG_PATH=${LOG_PATH:-logs/${TRAIN_CONFIG}_release.json}
EVAL_OUTPUT=${EVAL_OUTPUT:-eval/zeroshot_smoke.json}
TASKS=${TASKS:-piqa}
MAX_SAMPLES=${MAX_SAMPLES:-32}

step() {
  echo
  echo "[$(date +%H:%M:%S)] $1"
  echo "------------------------------------------------------------"
}

step "1/4: Syncing environment (uv sync --all-extras)"
uv sync --all-extras

step "2/4: Preparing filtered sample data"
uv run bash scripts/data/run_sample.sh

step "3/4: Running ${TRAIN_CONFIG} smoke training on device=${DEVICE}"
mkdir -p "$(dirname "${LOG_PATH}")"
uv run python train.py \
  --config-name "${TRAIN_CONFIG}" \
  train.device="${DEVICE}" \
  logging.enabled=true \
  logging.backend=json \
  logging.path="${LOG_PATH}" \
  train.checkpoint.enable=true \
  train.checkpoint.dir="${CHECKPOINT_DIR}" \
  train.checkpoint.save_interval=999999 \
  train.checkpoint.save_last=true

if ! ls "${CHECKPOINT_DIR}"/step_*.pt >/dev/null 2>&1; then
  echo "No checkpoints found in ${CHECKPOINT_DIR}. Training may have failed."
  exit 1
fi
LATEST_CKPT=$(ls -1 "${CHECKPOINT_DIR}"/step_*.pt | sort | tail -n 1)
echo "[Info] Using checkpoint ${LATEST_CKPT}"

step "4/4: Running zero-shot eval (${TASKS})"
mkdir -p "$(dirname "${EVAL_OUTPUT}")"
uv run python scripts/eval/zeroshot.py \
  --config "${MODEL_CONFIG}" \
  --checkpoint "${LATEST_CKPT}" \
  --tokenizer-path "${TOKENIZER_PATH}" \
  --tasks "${TASKS}" \
  --max-samples "${MAX_SAMPLES}" \
  --output "${EVAL_OUTPUT}" \
  --device "${DEVICE}"

echo
echo "[Done] Logs -> ${LOG_PATH}"
echo "[Done] Checkpoint -> ${LATEST_CKPT}"
echo "[Done] Eval metrics -> ${EVAL_OUTPUT}"
```

### File: `scripts/run_smoke.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

MODE=${1:-pilot}

if [[ "${MODE}" == "pilot" ]]; then
  echo "[Smoke] Running pilot config on CPU"
  uv run python train.py --config-name pilot_smoke
elif [[ "${MODE}" == "mid" ]]; then
  echo "[Smoke] Ensuring filtered shards exist"
  if [[ ! -d "data/shards/refinedweb_filtered" ]]; then
    echo "Filtered shards missing. Generate them first via configs/data/refinedweb_mixture_filtered.yaml"
    exit 1
  fi
  echo "[Smoke] Running mid mixture config on CPU"
  uv run python train.py --config-name mid_smoke
else
  echo "Usage: scripts/run_smoke.sh [pilot|mid]"
  exit 1
fi
```

### File: `scripts/tests/run_passkey_smoke.sh`

```bash
#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT_DIR="artifacts/checkpoints/pilot_smoke"
CHECKPOINT_PATH="${CHECKPOINT_DIR}/step_000010.pt"
TOKENIZER="tests/data/tiny_tokenizer.model"
OUTPUT_JSON="eval/passkey_ci.json"

rm -rf "${CHECKPOINT_DIR}"

echo "[passkey-ci] training pilot_smoke for 10 steps"
uv run python train.py --config-name pilot_smoke

echo "[passkey-ci] running synthetic passkey eval with memorization"
uv run python scripts/eval/passkey.py \
  --config configs/pilot_smoke.yaml \
  --checkpoint "${CHECKPOINT_PATH}" \
  --tokenizer-path "${TOKENIZER}" \
  --samples 8 \
  --filler-sentences 32 \
  --device cpu \
  --output "${OUTPUT_JSON}" \
  --memorize \
  --memorize-steps 1

uv run python - <<'PY'
import json
from pathlib import Path

data = json.loads(Path("eval/passkey_ci.json").read_text())
delta = data.get("accuracy_delta", 0.0)
if delta < 0:
    raise SystemExit(f"Memorization delta negative: {delta}")
print(f"[passkey-ci] Memorization delta OK ({delta:.3f})")
PY
```

### File: `src/nested_learning/__init__.py`

```python
"""Nested Learning (HOPE) reproduction package."""

from .levels import LevelClock, LevelSpec  # noqa: F401
```

### File: `src/nested_learning/assoc_memory.py`

```python
from __future__ import annotations

from typing import Protocol

import torch
import torch.nn as nn


class AssocMemory(nn.Module):
    """Base class for associative memories with explicit update hooks."""

    def forward(self, query: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        raise NotImplementedError

    @torch.no_grad()
    def update(
        self, *, key: torch.Tensor, value: torch.Tensor, error_signal: torch.Tensor | None = None
    ) -> None:
        raise NotImplementedError


class SupportsReset(Protocol):
    def reset_state(self) -> None: ...
```

### File: `src/nested_learning/backbones.py`

```python
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AttentionConfig:
    dim: int
    heads: int
    dropout: float = 0.0
    use_flash: bool = True
    causal: bool = True
    qk_l2_norm: bool = False
    qk_norm_eps: float = 1e-6
    local_conv_window: int | None = None


class SelfAttention(nn.Module):
    def __init__(self, config: AttentionConfig):
        super().__init__()
        if config.dim % config.heads != 0:
            msg = f"dim must be divisible by heads (got dim={config.dim}, heads={config.heads})"
            raise ValueError(msg)
        self.config = config
        self.heads = config.heads
        self.head_dim = config.dim // config.heads
        self.qkv = nn.Linear(config.dim, config.dim * 3, bias=False)
        self.out_proj = nn.Linear(config.dim, config.dim, bias=False)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.norm = nn.LayerNorm(config.dim)
        self.local_conv: nn.Conv1d | None = None
        if config.local_conv_window is not None:
            window = int(config.local_conv_window)
            if window <= 0:
                raise ValueError("local_conv_window must be positive")
            self.local_conv = nn.Conv1d(
                config.dim,
                config.dim,
                kernel_size=window,
                groups=config.dim,
                padding=0,
                bias=False,
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        residual = x
        attn_inp = x
        if self.local_conv is not None:
            kernel = self.local_conv.kernel_size[0]
            attn_inp = attn_inp.transpose(1, 2)
            # Causal depthwise conv: only attends to past tokens.
            attn_inp = F.pad(attn_inp, (kernel - 1, 0))
            attn_inp = self.local_conv(attn_inp).transpose(1, 2)
        q, k, v = self._compute_qkv(attn_inp)
        attn_output = self._scaled_dot_product_attn(q, k, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(x.size(0), x.size(1), -1)
        attn_output = self.out_proj(attn_output)
        attn_output = self.resid_dropout(attn_output)
        return self.norm(residual + attn_output)

    def _compute_qkv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        shape = (x.size(0), x.size(1), self.heads, self.head_dim)
        q = q.view(*shape).transpose(1, 2)
        k = k.view(*shape).transpose(1, 2)
        v = v.view(*shape).transpose(1, 2)
        if self.config.qk_l2_norm:
            q = F.normalize(q, dim=-1, eps=self.config.qk_norm_eps)
            k = F.normalize(k, dim=-1, eps=self.config.qk_norm_eps)
        return q, k, v

    def _scaled_dot_product_attn(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        dropout_p = self.config.dropout if self.training else 0.0
        device_type = q.device.type
        if (
            device_type == "cuda"
            and torch.cuda.is_available()
            and hasattr(torch.backends, "cuda")
            and hasattr(torch.backends.cuda, "sdp_kernel")
        ):
            with torch.backends.cuda.sdp_kernel(  # type: ignore[attr-defined]
                enable_flash=self.config.use_flash,
                enable_mem_efficient=True,
                enable_math=not self.config.use_flash,
            ):
                return F.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=None,
                    dropout_p=dropout_p,
                    is_causal=self.config.causal,
                )
        return F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=dropout_p,
            is_causal=self.config.causal,
        )
```

### File: `src/nested_learning/cms.py`

```python
from __future__ import annotations

from typing import Dict, Sequence

import torch
import torch.nn as nn

from .levels import LevelSpec, ensure_level_specs


class CMSBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_multiplier: int = 4,
        activation: str = "gelu",
        grad_clip: float = 1.0,
        use_layernorm: bool = True,
    ):
        super().__init__()
        hidden = dim * hidden_multiplier
        act: nn.Module
        if activation == "relu":
            act = nn.ReLU()
        elif activation == "silu":
            act = nn.SiLU()
        else:
            act = nn.GELU()
        norm: nn.Module = nn.LayerNorm(dim) if use_layernorm else nn.Identity()
        self.net = nn.Sequential(
            norm,
            nn.Linear(dim, hidden),
            act,
            nn.Linear(hidden, dim),
        )
        self.grad_clip = grad_clip

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        delta = self.net(x)
        if self.training and self.grad_clip > 0:
            with torch.no_grad():
                norm = delta.norm(dim=-1, keepdim=True)
                scale = torch.clamp(norm / self.grad_clip, min=1.0)
            delta = delta / scale
        return x + delta


class CMS(nn.Module):
    """Continuum Memory System with multi-frequency updates."""

    def __init__(
        self,
        *,
        dim: int,
        levels: Sequence[LevelSpec],
        hidden_multiplier: int = 4,
        activation: str = "gelu",
        use_layernorm: bool = True,
    ) -> None:
        super().__init__()
        ordered = ensure_level_specs(levels)
        self.level_specs: Sequence[LevelSpec] = tuple(ordered)
        self.blocks = nn.ModuleDict(
            {
                spec.name: CMSBlock(
                    dim,
                    hidden_multiplier=hidden_multiplier,
                    activation=activation,
                    grad_clip=1.0,
                    use_layernorm=use_layernorm,
                )
                for spec in self.level_specs
            }
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        return_intermediates: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        current = x
        inputs: Dict[str, torch.Tensor] = {}
        outputs: Dict[str, torch.Tensor] = {}
        for spec in self.level_specs:
            block = self.blocks[spec.name]
            inputs[spec.name] = current
            current = block(current)
            outputs[spec.name] = current
        if return_intermediates:
            return current, inputs, outputs
        return current
```

### File: `src/nested_learning/continual_classification.py`

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence


@dataclass(frozen=True)
class ClassificationExample:
    text: str
    label: str


@dataclass(frozen=True)
class LoadedClassificationDataset:
    name: str
    split: str
    examples: List[ClassificationExample]
    label_names: List[str]


def load_hf_classification_dataset(
    dataset: str,
    *,
    split: str,
    text_field: str,
    label_field: str,
    name: str | None = None,
    max_samples: int | None = None,
) -> LoadedClassificationDataset:
    """
    Load a HuggingFace `datasets` text classification dataset into a simple in-memory format.

    This is used by the Phase 4 continual-learning harness (CLINC/Banking/DBpedia).
    """
    try:
        from datasets import load_dataset  # type: ignore[import-not-found]
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "`datasets` dependency is required for continual classification."
        ) from exc

    ds = load_dataset(dataset, name=name, split=split)
    features = getattr(ds, "features", None)
    label_names: List[str] = []
    if features is not None and label_field in features:
        feature = features[label_field]
        if getattr(feature, "names", None) is not None:
            label_names = list(feature.names)

    examples: List[ClassificationExample] = []
    count = 0
    for row in ds:
        if max_samples is not None and count >= max_samples:
            break
        text = str(row[text_field])
        raw_label = row[label_field]
        if isinstance(raw_label, int) and label_names:
            label = label_names[raw_label]
        else:
            label = str(raw_label)
        examples.append(ClassificationExample(text=text, label=label))
        count += 1

    if not label_names:
        label_names = sorted({ex.label for ex in examples})

    return LoadedClassificationDataset(
        name=dataset if name is None else f"{dataset}:{name}",
        split=split,
        examples=examples,
        label_names=label_names,
    )


def load_clinc_oos(
    *,
    split: str = "test",
    max_samples: int | None = None,
) -> LoadedClassificationDataset:
    # HF dataset: "clinc_oos" with fields {"text", "intent"}.
    return load_hf_classification_dataset(
        "clinc_oos",
        split=split,
        text_field="text",
        label_field="intent",
        max_samples=max_samples,
    )


def load_banking77(
    *,
    split: str = "test",
    max_samples: int | None = None,
) -> LoadedClassificationDataset:
    # HF dataset: "banking77" with fields {"text", "label"}.
    return load_hf_classification_dataset(
        "banking77",
        split=split,
        text_field="text",
        label_field="label",
        max_samples=max_samples,
    )


def load_dbpedia14(
    *,
    split: str = "test",
    max_samples: int | None = None,
) -> LoadedClassificationDataset:
    # HF dataset: "dbpedia_14" with fields {"content", "label"}.
    return load_hf_classification_dataset(
        "dbpedia_14",
        split=split,
        text_field="content",
        label_field="label",
        max_samples=max_samples,
    )


def unique_labels(examples: Iterable[ClassificationExample]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for ex in examples:
        if ex.label in seen:
            continue
        seen.add(ex.label)
        ordered.append(ex.label)
    return ordered


def filter_examples_by_labels(
    examples: Sequence[ClassificationExample],
    *,
    allowed: set[str],
) -> List[ClassificationExample]:
    return [ex for ex in examples if ex.label in allowed]
```

### File: `src/nested_learning/continual_streaming.py`

```python
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import torch

from .continual_classification import ClassificationExample, unique_labels
from .memorize import MemorizeConfig, memorize_sequence
from .tokenizer import SentencePieceTokenizer


@dataclass(frozen=True)
class StreamingTask:
    task_id: int
    labels: List[str]
    train: List[ClassificationExample]
    eval: List[ClassificationExample]


@dataclass(frozen=True)
class ContinualEvalConfig:
    task_size: int = 10
    seed: int = 0
    train_per_label: int = 50
    eval_per_label: int = 50
    prompt_template: str = "Text: {text}\nLabel:"
    label_template: str = "{label}"
    task_aware: bool = True


def _logprob_completion(
    model,
    tokenizer: SentencePieceTokenizer,
    prompt: str,
    completion: str,
    device: torch.device,
    *,
    fast_state=None,
) -> float:
    prompt_ids = tokenizer.encode(prompt, add_bos=True)
    completion_ids = tokenizer.encode(" " + completion, add_bos=False)
    tokens = torch.cat([prompt_ids, completion_ids], dim=0).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tokens, fast_state=fast_state) if fast_state is not None else model(tokens)
        log_probs = torch.log_softmax(logits[:, :-1, :], dim=-1)
        target = tokens[:, 1:]
        gathered = log_probs.gather(-1, target.unsqueeze(-1)).squeeze(-1)
        prompt_len = prompt_ids.numel()
        return float(gathered[0, prompt_len - 1 :].sum().item())


def predict_label(
    model,
    tokenizer: SentencePieceTokenizer,
    text: str,
    candidates: Sequence[str],
    device: torch.device,
    *,
    prompt_template: str,
    label_template: str,
    fast_state=None,
) -> str:
    if not candidates:
        raise ValueError("predict_label requires at least one candidate label")
    prompt = prompt_template.format(text=text)
    best_label = candidates[0]
    best_score = -math.inf
    for label in candidates:
        label_str = label_template.format(label=label)
        score = _logprob_completion(
            model, tokenizer, prompt, label_str, device, fast_state=fast_state
        )
        if score > best_score:
            best_score = score
            best_label = label
    return best_label


def _balanced_split(
    examples: Sequence[ClassificationExample],
    *,
    labels: Sequence[str],
    train_per_label: int,
    eval_per_label: int,
) -> tuple[List[ClassificationExample], List[ClassificationExample]]:
    train: List[ClassificationExample] = []
    eval_: List[ClassificationExample] = []
    counts_train: Dict[str, int] = {lbl: 0 for lbl in labels}
    counts_eval: Dict[str, int] = {lbl: 0 for lbl in labels}
    for ex in examples:
        lbl = ex.label
        if lbl not in counts_train:
            continue
        if counts_train[lbl] < train_per_label:
            train.append(ex)
            counts_train[lbl] += 1
        elif counts_eval[lbl] < eval_per_label:
            eval_.append(ex)
            counts_eval[lbl] += 1
        if all(v >= train_per_label for v in counts_train.values()) and all(
            v >= eval_per_label for v in counts_eval.values()
        ):
            break
    return train, eval_


def build_streaming_tasks(
    examples: Sequence[ClassificationExample],
    *,
    cfg: ContinualEvalConfig,
    label_order: Sequence[str] | None = None,
) -> List[StreamingTask]:
    labels = list(label_order) if label_order is not None else unique_labels(examples)
    if label_order is None:
        import random

        rng = random.Random(cfg.seed)
        rng.shuffle(labels)
    if cfg.task_size <= 0:
        raise ValueError("task_size must be positive")
    tasks: List[StreamingTask] = []
    for task_id, start in enumerate(range(0, len(labels), cfg.task_size)):
        task_labels = labels[start : start + cfg.task_size]
        if not task_labels:
            break
        task_examples = [ex for ex in examples if ex.label in set(task_labels)]
        train, eval_ = _balanced_split(
            task_examples,
            labels=task_labels,
            train_per_label=cfg.train_per_label,
            eval_per_label=cfg.eval_per_label,
        )
        tasks.append(
            StreamingTask(task_id=task_id, labels=list(task_labels), train=train, eval=eval_)
        )
    return tasks


@dataclass(frozen=True)
class ContinualEvalResult:
    task_accuracy_matrix: List[List[float]]
    per_task_forgetting: List[float]
    avg_accuracy_final: float
    avg_forgetting: float


def evaluate_continual_classification(
    model,
    tokenizer: SentencePieceTokenizer,
    tasks: Sequence[StreamingTask],
    device: torch.device,
    *,
    cfg: ContinualEvalConfig,
    memorize_cfg: MemorizeConfig,
) -> tuple[ContinualEvalResult, Dict[str, Any]]:
    """
    Streaming class-incremental evaluation using generative classification + optional
    test-time memorization.

    - If `memorize_cfg.enabled`, each training example is memorized by appending the correct
      label string.
    - Accuracy is computed after each task on each task's eval set, producing a task-accuracy
      matrix.
    """
    meta_snapshot: Dict[str, torch.Tensor] | None = None
    if memorize_cfg.enabled and (not memorize_cfg.use_fast_state) and memorize_cfg.reset:
        from .memorize import snapshot_state_dict  # local import to avoid cycles

        meta_snapshot = snapshot_state_dict(model)

    fast_state = None
    if memorize_cfg.enabled and memorize_cfg.use_fast_state:
        if not hasattr(model, "init_fast_state"):
            raise RuntimeError("Model does not support fast state memorization")
        fast_state = model.init_fast_state()

    task_acc: List[List[float]] = [[float("nan") for _ in tasks] for _ in tasks]
    best_acc: List[float] = [0.0 for _ in tasks]

    memorize_stats_total: Dict[str, float] = {}

    def _eval_task(task_idx: int) -> float:
        task = tasks[task_idx]
        candidates = (
            task.labels
            if cfg.task_aware
            else [lbl for t in tasks[: current_task + 1] for lbl in t.labels]
        )
        if not task.eval:
            return float("nan")
        correct = 0
        for ex in task.eval:
            pred = predict_label(
                model,
                tokenizer,
                ex.text,
                candidates,
                device,
                prompt_template=cfg.prompt_template,
                label_template=cfg.label_template,
                fast_state=fast_state,
            )
            correct += int(pred == ex.label)
        return correct / len(task.eval) if task.eval else float("nan")

    for current_task, task in enumerate(tasks):
        # Online "training" on this task's examples via optional memorization.
        for ex in task.train:
            candidates = (
                task.labels
                if cfg.task_aware
                else [lbl for t in tasks[: current_task + 1] for lbl in t.labels]
            )
            _ = predict_label(
                model,
                tokenizer,
                ex.text,
                candidates,
                device,
                prompt_template=cfg.prompt_template,
                label_template=cfg.label_template,
                fast_state=fast_state,
            )
            if memorize_cfg.enabled:
                prompt = cfg.prompt_template.format(text=ex.text)
                target = cfg.label_template.format(label=ex.label)
                memorize_text = f"{prompt} {target}"
                if memorize_cfg.use_fast_state and memorize_cfg.reset:
                    fast_state = model.init_fast_state()
                stats = memorize_sequence(
                    model, tokenizer, memorize_text, device, memorize_cfg, fast_state=fast_state
                )
                for k, v in stats.items():
                    memorize_stats_total[k] = memorize_stats_total.get(k, 0.0) + v
                if (
                    (not memorize_cfg.use_fast_state)
                    and memorize_cfg.reset
                    and meta_snapshot is not None
                ):
                    from .memorize import restore_state_dict  # local import to avoid cycles

                    restore_state_dict(model, meta_snapshot)

        # Evaluate on all tasks seen so far.
        for task_idx in range(current_task + 1):
            acc = _eval_task(task_idx)
            task_acc[task_idx][current_task] = acc
            if not math.isnan(acc):
                best_acc[task_idx] = max(best_acc[task_idx], acc)

    final_accs = [task_acc[i][-1] for i in range(len(tasks)) if not math.isnan(task_acc[i][-1])]
    avg_accuracy_final = sum(final_accs) / len(final_accs) if final_accs else float("nan")

    per_task_forgetting: List[float] = []
    for i in range(len(tasks)):
        last = task_acc[i][-1]
        if math.isnan(last):
            per_task_forgetting.append(float("nan"))
            continue
        per_task_forgetting.append(best_acc[i] - last)
    valid_forgetting = [f for f in per_task_forgetting if not math.isnan(f)]
    avg_forgetting = (
        sum(valid_forgetting) / len(valid_forgetting) if valid_forgetting else float("nan")
    )

    result = ContinualEvalResult(
        task_accuracy_matrix=task_acc,
        per_task_forgetting=per_task_forgetting,
        avg_accuracy_final=avg_accuracy_final,
        avg_forgetting=avg_forgetting,
    )
    meta = {
        "task_size": cfg.task_size,
        "train_per_label": cfg.train_per_label,
        "eval_per_label": cfg.eval_per_label,
        "task_aware": cfg.task_aware,
        "prompt_template": cfg.prompt_template,
        "label_template": cfg.label_template,
        "memorize_stats": memorize_stats_total,
    }
    return result, meta
```

### File: `src/nested_learning/data.py`

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info


@dataclass
class SyntheticTextConfig:
    vocab_size: int
    seq_len: int
    dataset_size: int


class SyntheticTextDataset(Dataset[torch.Tensor]):
    def __init__(self, config: SyntheticTextConfig):
        self.config = config

    def __len__(self) -> int:
        return self.config.dataset_size

    def __getitem__(self, idx: int) -> torch.Tensor:
        g = torch.Generator().manual_seed(idx)
        return torch.randint(0, self.config.vocab_size, (self.config.seq_len,), generator=g)


class TokenShardDataset(Dataset[torch.Tensor]):
    """Memory-mapped dataset over NumPy shards produced by shard_corpus.py."""

    def __init__(self, shard_dir: str | Path):
        self.shard_dir = Path(shard_dir)
        if not self.shard_dir.exists():
            msg = f"Shard directory {self.shard_dir} does not exist"
            raise FileNotFoundError(msg)
        self.paths = sorted(self.shard_dir.glob("*.npy"))
        if not self.paths:
            msg = f"No shard files found in {self.shard_dir}"
            raise ValueError(msg)
        self.metadata: List[tuple[int, int]] = []
        self._cache: dict[int, np.memmap] = {}
        total = 0
        for idx, path in enumerate(self.paths):
            arr = np.load(path, mmap_mode="r")
            length = arr.shape[0]
            self.metadata.append((total, length))
            total += length
        self.total_sequences = total

    def __len__(self) -> int:
        return self.total_sequences

    def _load_array(self, shard_idx: int) -> np.memmap:
        if shard_idx not in self._cache:
            self._cache[shard_idx] = np.load(self.paths[shard_idx], mmap_mode="r")
        return self._cache[shard_idx]

    def __getitem__(self, idx: int) -> torch.Tensor:
        if idx < 0 or idx >= self.total_sequences:
            raise IndexError(idx)
        shard_idx = self._find_shard(idx)
        start_offset = self.metadata[shard_idx][0]
        arr = self._load_array(shard_idx)
        local_idx = idx - start_offset
        tokens = torch.from_numpy(arr[local_idx])
        return tokens.long()

    def _find_shard(self, idx: int) -> int:
        lo, hi = 0, len(self.metadata) - 1
        while lo <= hi:
            mid = (lo + hi) // 2
            start, length = self.metadata[mid]
            if idx < start:
                hi = mid - 1
            elif idx >= start + length:
                lo = mid + 1
            else:
                return mid
        return len(self.metadata) - 1


@dataclass
class ShardSourceConfig:
    name: str
    shards_dir: str
    weight: float


class ShardSource:
    def __init__(self, config: ShardSourceConfig):
        self.name = config.name
        self.weight = config.weight
        self.dir = Path(config.shards_dir)
        if not self.dir.exists():
            msg = f"Shard directory {self.dir} missing for source {self.name}"
            raise FileNotFoundError(msg)
        self.paths = sorted(self.dir.glob("*.npy"))
        if not self.paths:
            raise ValueError(f"No shard files in {self.dir}")
        self._cache: dict[Path, np.memmap] = {}

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        shard_path = self.paths[rng.integers(0, len(self.paths))]
        if shard_path not in self._cache:
            self._cache[shard_path] = np.load(shard_path, mmap_mode="r")
        shard = self._cache[shard_path]
        idx = rng.integers(0, shard.shape[0])
        return shard[idx]


class MixtureShardDataset(IterableDataset[torch.Tensor]):
    def __init__(
        self,
        sources: Sequence[ShardSourceConfig],
        *,
        samples_per_epoch: int,
        seed: int = 0,
    ):
        super().__init__()
        self.sources = [ShardSource(cfg) for cfg in sources]
        total_weight = sum(max(src.weight, 0.0) for src in self.sources)
        if total_weight <= 0:
            raise ValueError("Mixture weights must sum to > 0")
        self.weights = np.array([max(src.weight, 0.0) / total_weight for src in self.sources])
        self.samples_per_epoch = samples_per_epoch
        self.seed = seed

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __iter__(self) -> Iterator[torch.Tensor]:
        worker = get_worker_info()
        if worker is None:
            start = 0
            end = self.samples_per_epoch
            worker_seed = self.seed
        else:
            per_worker = (self.samples_per_epoch + worker.num_workers - 1) // worker.num_workers
            start = worker.id * per_worker
            end = min(start + per_worker, self.samples_per_epoch)
            worker_seed = self.seed + worker.id
        rng = np.random.default_rng(worker_seed)
        for _ in range(start, end):
            idx = rng.choice(len(self.sources), p=self.weights)
            sample = np.array(self.sources[idx].sample(rng), copy=True)
            yield torch.from_numpy(sample).long()


def collate_batch(batch: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack(batch, dim=0)
```

### File: `src/nested_learning/device.py`

```python
from __future__ import annotations

import torch


def resolve_device(device_str: str) -> torch.device:
    normalized = str(device_str).strip().lower()
    if normalized.startswith("cuda"):
        if not torch.cuda.is_available():
            return torch.device("cpu")
        parts = normalized.split(":")
        idx = int(parts[1]) if len(parts) > 1 else 0
        if idx >= torch.cuda.device_count():
            idx = max(torch.cuda.device_count() - 1, 0)
        return torch.device(f"cuda:{idx}")
    if normalized.startswith("mps"):
        if not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
            return torch.device("cpu")
        return torch.device("mps")
    return torch.device(device_str)

```

### File: `src/nested_learning/fast_state.py`

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, cast

import torch
from torch import nn

from .optim.manager import LevelConfig, LevelOptimizerManager
from .titan.self_modifying import SelfModifyingTitansState

ParamDict = Dict[str, torch.Tensor]


def init_module_deltas(module: nn.Module) -> ParamDict:
    """
    Initialize a per-parameter "fast state" delta dict for meta+delta fast state.

    The fast state stores *deltas* (initialized to 0) rather than detached parameter clones so that
    forward passes can use `meta_param + delta`, allowing outer gradients to flow to meta params
    while keeping online updates as stop-grad writes into the delta tensors.
    """

    return {name: torch.zeros_like(param).detach() for name, param in module.named_parameters()}


@dataclass
class BlockFastState:
    titan_params: ParamDict | None
    cms_params: Dict[str, ParamDict]
    level_manager: LevelOptimizerManager
    selfmod_state: SelfModifyingTitansState | None = None


def build_block_fast_state(
    *,
    titan_module: nn.Module | None,
    cms_blocks: Dict[str, nn.Module],
    selfmod_module: nn.Module | None = None,
    specs,
    optimizer_configs: Dict[str, dict],
    default_lr: float,
) -> BlockFastState:
    titan_params = None
    if titan_module is not None:
        titan_params = init_module_deltas(titan_module)
    cms_params = {name: init_module_deltas(block) for name, block in cms_blocks.items()}
    level_cfg = LevelConfig(specs=specs, optimizer_configs=optimizer_configs, default_lr=default_lr)
    level_manager = LevelOptimizerManager(level_cfg)
    selfmod_state = None
    if selfmod_module is not None:
        init_fn = getattr(selfmod_module, "init_fast_state", None)
        if callable(init_fn):
            selfmod_state = cast(SelfModifyingTitansState, init_fn())
    return BlockFastState(
        titan_params=titan_params,
        cms_params=cms_params,
        level_manager=level_manager,
        selfmod_state=selfmod_state,
    )


@dataclass
class ModelFastState:
    blocks: list[BlockFastState]
```

### File: `src/nested_learning/functional.py`

```python
from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

import torch
from torch import nn
from torch.func import functional_call

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

### File: `src/nested_learning/hope/__init__.py`

```python

```

### File: `src/nested_learning/hope/block.py`

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
        fast_state.level_manager.tick()
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
            current = call_with_deltas(self.cms.blocks[level_name], params, current)
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
                current = call_with_deltas(self.cms.blocks[level_name], params, current)
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
        updated, magnitude = fast_state.level_manager.apply_grads(
            level_name,
            base_params,
            grads_dict,
            context=context_vec,
            force=True,
        )
        fast_state.cms_params[level_name] = updated
        fast_state.level_manager.pop_last_metrics(level_name)
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
        if self.config.selfmod_online_updates and teach_signal is not None:
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
        fast_state.level_manager.tick()
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
            current = call_with_deltas(self.cms.blocks[level_name], params, current)
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
                current = call_with_deltas(self.cms.blocks[level_name], params, current)
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
        updated, magnitude = fast_state.level_manager.apply_grads(
            level_name,
            base_params,
            grads_dict,
            context=context_vec,
            force=True,
        )
        fast_state.cms_params[level_name] = updated
        fast_state.level_manager.pop_last_metrics(level_name)
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
        mem_out = call_with_deltas(self.titan_memory, fast_state.titan_params, attn_out)
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
        fast_state.level_manager.tick()
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
            current = call_with_deltas(self.cms.blocks[level_name], params, current)
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
                current = call_with_deltas(self.cms.blocks[level_name], params, current)
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
        if not fast_state.level_manager.should_update(level_name):
            return
        if not self._passes_surprise(surprise_value):
            self._record_gate(level_name, hit=False)
            return
        if fast_state.titan_params is None:
            return
        modifier = self.self_modifier(
            key=attn_out.detach(),
            value=mem_out.detach(),
            error_signal=teach_signal.detach(),
        )
        context_vec = attn_out.detach().mean(dim=(0, 1))
        base_params = fast_state.titan_params
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
        updated, magnitude = fast_state.level_manager.apply_grads(
            level_name,
            base_params,
            grads_dict,
            context=context_vec,
            force=False,
        )
        fast_state.titan_params = updated
        extra_metrics = fast_state.level_manager.pop_last_metrics(level_name)
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
        updated, magnitude = fast_state.level_manager.apply_grads(
            level_name,
            base_params,
            grads_dict,
            context=context_vec,
            force=True,
        )
        fast_state.cms_params[level_name] = updated
        fast_state.level_manager.pop_last_metrics(level_name)
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

### File: `src/nested_learning/hope/self_mod.py`

```python
from __future__ import annotations

import torch
import torch.nn as nn


class SelfModifier(nn.Module):
    """
    Learns parameter updates conditioned on key/value/error signals.

    Note: In this implementation, we predict a 'target modification' (delta to the error signal)
    rather than directly predicting weight deltas (Delta W). Mathematically, modifying the
    target y to (y + delta) in the inner optimization step:
        L = || f(x) - (y + delta) ||^2
    results in a gradient update that is shifted by the gradient of delta.
    This is functionally equivalent to a 'Learned Optimization Step' or 'Hypernetwork'
    that modulates the update direction, but is more efficient to implement for
    large memory modules than generating O(d^2) weight parameters directly.
    """

    def __init__(self, dim: int, hidden_multiplier: int = 4):
        super().__init__()
        hidden = dim * hidden_multiplier
        self.net = nn.Sequential(
            nn.Linear(dim * 3, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )

    def forward(
        self,
        *,
        key: torch.Tensor,
        value: torch.Tensor,
        error_signal: torch.Tensor,
    ) -> torch.Tensor:
        concat = torch.cat([key, value, error_signal], dim=-1)
        return self.net(concat)
```

### File: `src/nested_learning/instrumentation.py`

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class UpdateEvent:
    step: int
    level: str
    magnitude: float | None = None


@dataclass
class UpdateLog:
    """Lightweight container for tracking update magnitudes per level."""

    events: List[UpdateEvent] = field(default_factory=list)

    def record(self, *, step: int, level: str, magnitude: float | None = None) -> None:
        self.events.append(UpdateEvent(step=step, level=level, magnitude=magnitude))

    def summary(self) -> Dict[str, Dict[str, float]]:
        counts: Dict[str, int] = {}
        totals: Dict[str, float] = {}
        for event in self.events:
            counts[event.level] = counts.get(event.level, 0) + 1
            if event.magnitude is not None:
                totals[event.level] = totals.get(event.level, 0.0) + event.magnitude
        return {
            level: {
                "updates": counts[level],
                "avg_magnitude": (
                    totals[level] / counts[level] if level in totals else float("nan")
                ),
            }
            for level in counts
        }
```

### File: `src/nested_learning/levels.py`

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, MutableMapping, Sequence


@dataclass(frozen=True)
class LevelSpec:
    """Configuration for a nested-learning level."""

    name: str
    update_period: int
    warmup_steps: int = 0
    jitter: int = 0
    optimizer_key: str | None = None

    def __post_init__(self) -> None:
        if self.update_period <= 0:
            msg = f"update_period for level {self.name} must be positive"
            raise ValueError(msg)
        if self.warmup_steps < 0:
            msg = f"warmup_steps for level {self.name} must be non-negative"
            raise ValueError(msg)
        if self.jitter < 0:
            msg = f"jitter for level {self.name} must be non-negative"
            raise ValueError(msg)


@dataclass
class LevelState:
    last_step: int = -1
    updates: int = 0


class LevelClock:
    """Deterministic scheduler for Nested Learning level updates."""

    def __init__(self, specs: Sequence[LevelSpec]):
        self._specs: Dict[str, LevelSpec] = {spec.name: spec for spec in specs}
        if len(self._specs) != len(specs):
            raise ValueError("Duplicate level names provided to LevelClock")
        self._state: MutableMapping[str, LevelState] = {name: LevelState() for name in self._specs}
        self._step: int = 0
        self._timeline: List[dict] = []

    @property
    def step(self) -> int:
        return self._step

    def tick(self) -> None:
        self._step += 1

    def should_update(self, name: str) -> bool:
        spec = self._specs[name]
        state = self._state[name]
        if self._step < spec.warmup_steps:
            return False
        delta = self._step - state.last_step
        period = spec.update_period
        if spec.jitter:
            period = period + (self._step % (spec.jitter + 1))
        return state.last_step < 0 or delta >= period

    def record_update(self, name: str) -> None:
        state = self._state[name]
        state.last_step = self._step
        state.updates += 1
        self._timeline.append({"step": self._step, "level": name})

    def levels_in_frequency_order(self) -> List[LevelSpec]:
        return sorted(self._specs.values(), key=lambda spec: spec.update_period)

    def stats(self) -> Dict[str, LevelState]:
        return {
            name: LevelState(state.last_step, state.updates) for name, state in self._state.items()
        }

    def timeline(self) -> List[dict]:
        return list(self._timeline)


def ensure_level_specs(entries: Iterable[LevelSpec]) -> List[LevelSpec]:
    """Ensure deterministic ordering and validate duplicates."""

    specs = list(entries)
    seen = set()
    ordered: List[LevelSpec] = []
    for spec in specs:
        if spec.name in seen:
            msg = f"Duplicate level spec {spec.name}"
            raise ValueError(msg)
        seen.add(spec.name)
        ordered.append(spec)
    return ordered
```

### File: `src/nested_learning/logging_utils.py`

```python
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, cast

from omegaconf import DictConfig, OmegaConf


class BaseLogger:
    def log(self, metrics: Dict[str, Any], step: int) -> None:
        raise NotImplementedError

    def finish(self) -> None:
        pass


class NullLogger(BaseLogger):
    def log(self, metrics: Dict[str, Any], step: int) -> None:
        return


class JSONLogger(BaseLogger):
    def __init__(self, path: Path):
        self.path = path
        self.records: list[Dict[str, Any]] = []

    def log(self, metrics: Dict[str, Any], step: int) -> None:
        payload = {"step": step, **metrics}
        self.records.append(payload)

    def finish(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.records, indent=2))


class WandbLogger(BaseLogger):
    def __init__(self, cfg: DictConfig, full_cfg: DictConfig):
        import wandb

        project = cfg.get("project", "nested-learning")
        run_name = cfg.get("run_name")
        config_dict = cast(dict[str, Any], OmegaConf.to_container(full_cfg, resolve=True))
        self.run = wandb.init(project=project, name=run_name, config=config_dict)

    def log(self, metrics: Dict[str, Any], step: int) -> None:
        if self.run is not None:
            self.run.log(metrics, step=step)

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()


def init_logger(logging_cfg: DictConfig | None, full_cfg: DictConfig) -> BaseLogger:
    if logging_cfg is None or not logging_cfg.get("enabled", False):
        return NullLogger()
    backend = logging_cfg.get("backend", "wandb").lower()
    if backend == "wandb":
        return WandbLogger(logging_cfg, full_cfg)
    if backend == "json":
        path = Path(logging_cfg.get("path", "logs/train_metrics.json"))
        return JSONLogger(path)
    return NullLogger()
```

### File: `src/nested_learning/memorize.py`

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn

from .tokenizer import SentencePieceTokenizer
from .training import compute_teach_signal


@dataclass
class MemorizeConfig:
    enabled: bool = False
    steps: int = 1
    reset: bool = True
    use_correct_answer: bool = False
    use_fast_state: bool = True
    surprise_threshold: float | None = None
    paths: tuple[str, ...] | None = None
    layers: tuple[int, ...] | None = None
    online_chunk_size: int | None = None  # If set, use online chunked updates


def snapshot_state_dict(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def restore_state_dict(model: torch.nn.Module, state: Dict[str, torch.Tensor]) -> None:
    model.load_state_dict(state, strict=False)


def _setup_memorization_context(model, cfg: MemorizeConfig):
    """Helper to setup model state for memorization."""
    prev_allowed = getattr(model, "get_allowed_update_levels", lambda: None)()
    prev_threshold = getattr(model, "get_surprise_threshold", lambda: None)()
    prev_layers = getattr(model, "get_allowed_update_layers", lambda: None)()

    if hasattr(model, "set_allowed_update_levels"):
        allowed = None
        if cfg.paths is not None:
            allowed = {path.strip() for path in cfg.paths if path.strip()}
        getattr(model, "set_allowed_update_levels")(allowed)

    if cfg.surprise_threshold is not None and hasattr(model, "set_surprise_threshold"):
        getattr(model, "set_surprise_threshold")(cfg.surprise_threshold)

    if hasattr(model, "set_allowed_update_layers"):
        layers = None
        if cfg.layers is not None:
            layers = {int(idx) for idx in cfg.layers}
        getattr(model, "set_allowed_update_layers")(layers)

    return prev_allowed, prev_threshold, prev_layers


def _teardown_memorization_context(model, prev_allowed, prev_threshold, prev_layers):
    """Helper to restore model state after memorization."""
    if hasattr(model, "set_allowed_update_levels"):
        getattr(model, "set_allowed_update_levels")(
            prev_allowed if prev_allowed is None else set(prev_allowed)
        )
    if hasattr(model, "set_surprise_threshold"):
        getattr(model, "set_surprise_threshold")(prev_threshold)
    if hasattr(model, "set_allowed_update_layers"):
        getattr(model, "set_allowed_update_layers")(
            None if prev_layers is None else {int(idx) for idx in prev_layers}
        )


def _collect_metrics(model, stats: dict[str, float]):
    """Helper to collect and aggregate update metrics."""
    if hasattr(model, "pop_update_metrics"):
        metrics = model.pop_update_metrics()
        titan_updates = sum(
            value for key, value in metrics.items() if key.endswith("titan.titan.grad_norm")
        )
        titan_hits = sum(
            value for key, value in metrics.items() if key.endswith("titan.titan.gate_hit")
        )
        stats["titan_mem_updates"] += titan_updates
        stats["titan_update_events"] += titan_hits

        # Aggregate CMS updates per level: keys look like "layer{idx}.cms.<level>.<metric>".
        for key, value in metrics.items():
            parts = key.split(".")
            if len(parts) < 4:
                continue
            if parts[-3] != "cms":
                continue
            level = parts[-2]
            metric = parts[-1]
            if metric == "grad_norm":
                stats_key = f"{level}_updates"
                stats[stats_key] = stats.get(stats_key, 0.0) + float(value)
            elif metric == "gate_hit":
                stats_key = f"{level}_update_events"
                stats[stats_key] = stats.get(stats_key, 0.0) + float(value)


def _layernorm_backward(
    grad_out: torch.Tensor,
    pre_norm: torch.Tensor,
    norm: nn.LayerNorm,
) -> torch.Tensor:
    """
    Convert gradient w.r.t. LayerNorm output into gradient w.r.t. LayerNorm input.

    This aligns the teach signal with the pre-norm hidden state that the blocks actually update.
    """
    if grad_out.shape != pre_norm.shape:
        raise ValueError("grad_out and pre_norm must have identical shapes")
    weight = norm.weight
    if weight is None:
        weight = torch.ones(pre_norm.shape[-1], device=pre_norm.device, dtype=pre_norm.dtype)
    grad_hat = grad_out * weight.to(grad_out.dtype).view(1, 1, -1)
    mean = pre_norm.mean(dim=-1, keepdim=True)
    var = pre_norm.var(dim=-1, unbiased=False, keepdim=True)
    inv_std = torch.rsqrt(var + norm.eps)
    x_hat = (pre_norm - mean) * inv_std
    grad_mean = grad_hat.mean(dim=-1, keepdim=True)
    grad_proj = (grad_hat * x_hat).mean(dim=-1, keepdim=True)
    return (grad_hat - grad_mean - x_hat * grad_proj) * inv_std


def _get_model_surprise_metric(model) -> str:
    getter = getattr(model, "get_surprise_metric", None)
    if callable(getter):
        return str(getter()).strip().lower()
    return "l2"


def _compute_surprise_value(
    *,
    model,
    metric: str,
    logits: torch.Tensor,
    tokens: torch.Tensor,
    teach_signal: torch.Tensor,
) -> tuple[float, float | None]:
    normalized = str(metric).strip().lower()
    if normalized == "l2":
        runtime_scale = float(getattr(model, "_runtime_teach_scale", 1.0))
        runtime_clip = float(getattr(model, "_runtime_teach_clip", 0.0))
        scaled = teach_signal * runtime_scale
        if runtime_clip > 0:
            norm = scaled.norm(dim=-1, keepdim=True)
            scale = torch.clamp(norm / runtime_clip, min=1.0)
            scaled = scaled / scale
        value = float(scaled.norm(dim=-1).mean().item())
        return value, None
    if normalized == "loss":
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)),
            tokens[:, 1:].reshape(-1),
        )
        value = float(loss.detach().item())
        return value, value
    if normalized == "logit_entropy":
        logits_detached = logits[:, :-1].detach().float()
        probs = torch.softmax(logits_detached, dim=-1)
        entropy = -(probs * torch.log(probs.clamp(min=1e-9))).sum(dim=-1).mean()
        value = float(entropy.item())
        return value, value
    raise ValueError(f"Unsupported surprise_metric={metric!r}")


def memorize_tokens(
    model,
    token_batch: torch.Tensor,
    cfg: MemorizeConfig,
    *,
    fast_state=None,
    teach_mask: torch.Tensor | None = None,
) -> dict[str, float]:
    if token_batch.size(1) < 2:
        return {}

    if cfg.use_fast_state and fast_state is None:
        raise ValueError("cfg.use_fast_state=True requires passing fast_state")

    with torch.no_grad():
        stats: dict[str, float] = {
            "titan_mem_updates": 0.0,
            "titan_update_events": 0.0,
            "cms_fast_updates": 0.0,
            "cms_fast_update_events": 0.0,
            "cms_mid_updates": 0.0,
            "cms_mid_update_events": 0.0,
            "cms_slow_updates": 0.0,
            "cms_slow_update_events": 0.0,
            "cms_ultra_updates": 0.0,
            "cms_ultra_update_events": 0.0,
        }
        prev_allowed, prev_threshold, prev_layers = _setup_memorization_context(model, cfg)

        if cfg.online_chunk_size and cfg.online_chunk_size > 0:
            # Online / Chunked Learning Mode
            seq_len = token_batch.size(1)
            chunk_size = cfg.online_chunk_size

            # We process the sequence in increasing windows
            # But to avoid O(N^2) cost for very long sequences, this is an approximation
            # where we re-process the history. For faithful online learning, this is necessary
            # without external KV cache management.

            # Note: compute_teach_signal computes gradients for predicting tokens[1:]
            # token_batch: [t0, t1, t2, t3]
            # logits: [p1, p2, p3, p4] (aligned with t0..t3 input)
            # teach_signal index i corresponds to error on token[i+1]

            # We iterate over target token indices (1..seq_len-1) in chunks.
            # For targets up to index K (exclusive end), feed tokens[:, :K] as context.
            target_start = 1
            while target_start < seq_len:
                target_end = min(target_start + chunk_size, seq_len)
                # We want to learn targets [target_start ... target_end]
                # (python slice style end index).
                # Range: target_start until target_end.

                # To compute error for target at index K, we need input 0..K.
                # So we need input up to target_end-1? No, up to target_end.
                # Because compute_teach_signal aligns logits[:-1] with tokens[1:].
                # If tokens is [A, B], logits[:-1] is preds for [B].
                # So if we have input [A, B], we get error for B.
                # If we have input [A, B, C], we get error for B, C.

                # So to get error for targets up to target_end-1 (python slice),
                # we need input tokens[:, :target_end].

                context_tokens = token_batch[:, :target_end]

                pre_norm = None
                if hasattr(model, "forward_with_pre_norm"):
                    forward_fn = getattr(model, "forward_with_pre_norm")
                    logits, pre_norm = (
                        forward_fn(context_tokens, fast_state=fast_state)
                        if cfg.use_fast_state
                        else forward_fn(context_tokens)
                    )
                else:
                    logits = (
                        model(context_tokens, fast_state=fast_state)
                        if cfg.use_fast_state
                        else model(context_tokens)
                    )
                full_signal = compute_teach_signal(model, logits, context_tokens)
                if pre_norm is not None:
                    norm = getattr(model, "norm", None)
                    if isinstance(norm, nn.LayerNorm):
                        full_signal = _layernorm_backward(full_signal, pre_norm, norm)

                # full_signal length is target_end.
                # indices correspond to errors for targets at 1 ... target_end.
                # idx 0 -> target 1.
                # idx k -> target k+1.

                # We want to keep errors for targets [target_start ... target_end-1].
                # These correspond to signal indices [target_start-1 ... target_end-2].

                # Example: [A, B, C]. target_start=1 (B). target_end=2 (up to B).
                # chunk=1.
                # context [A, B].
                # signal len 2. idx 0->B. idx 1->pad.
                # We want B. idx 0.
                # signal indices: target_start-1 (0) to target_end-1 (1)?
                # Wait, if target_end is 2 (slice), we processed B.
                # signal indices: 1-1=0. 2-2=0. Range 0:1.

                mask = torch.zeros_like(full_signal)
                mask_start = target_start - 1
                mask_end = target_end - 1
                mask[:, mask_start:mask_end, :] = 1.0

                masked_signal = full_signal * mask
                if teach_mask is not None:
                    if teach_mask.ndim != 2:
                        raise ValueError("teach_mask must have shape (B, T)")
                    if teach_mask.shape[0] != token_batch.shape[0]:
                        raise ValueError("teach_mask batch size mismatch")
                    mask_slice = teach_mask[:, :target_end].to(masked_signal.device).float()
                    masked_signal = masked_signal * mask_slice.unsqueeze(-1)
                surprise_metric = _get_model_surprise_metric(model)
                surprise_value, surprise_override = _compute_surprise_value(
                    model=model,
                    metric=surprise_metric,
                    logits=logits,
                    tokens=context_tokens,
                    teach_signal=masked_signal,
                )
                if cfg.surprise_threshold is not None and surprise_value < cfg.surprise_threshold:
                    target_start = target_end
                    continue
                if cfg.use_fast_state:
                    model(
                        context_tokens,
                        teach_signal=masked_signal,
                        surprise_value=surprise_override,
                        fast_state=fast_state,
                    )
                else:
                    model(
                        context_tokens,
                        teach_signal=masked_signal,
                        surprise_value=surprise_override,
                    )
                _collect_metrics(model, stats)

                target_start = target_end

        else:
            # Batch Mode (Default)
            for _ in range(cfg.steps):
                pre_norm = None
                if hasattr(model, "forward_with_pre_norm"):
                    forward_fn = getattr(model, "forward_with_pre_norm")
                    logits, pre_norm = (
                        forward_fn(token_batch, fast_state=fast_state)
                        if cfg.use_fast_state
                        else forward_fn(token_batch)
                    )
                else:
                    logits = (
                        model(token_batch, fast_state=fast_state)
                        if cfg.use_fast_state
                        else model(token_batch)
                    )
                teach_signal = compute_teach_signal(model, logits, token_batch)
                if pre_norm is not None:
                    norm = getattr(model, "norm", None)
                    if isinstance(norm, nn.LayerNorm):
                        teach_signal = _layernorm_backward(teach_signal, pre_norm, norm)
                if teach_mask is not None:
                    if teach_mask.ndim != 2:
                        raise ValueError("teach_mask must have shape (B, T)")
                    if teach_mask.shape[:2] != teach_signal.shape[:2]:
                        raise ValueError("teach_mask shape mismatch")
                    mask_f = teach_mask.to(teach_signal.device).float().unsqueeze(-1)
                    teach_signal = teach_signal * mask_f
                surprise_metric = _get_model_surprise_metric(model)
                surprise_value, surprise_override = _compute_surprise_value(
                    model=model,
                    metric=surprise_metric,
                    logits=logits,
                    tokens=token_batch,
                    teach_signal=teach_signal,
                )
                if cfg.surprise_threshold is not None and surprise_value < cfg.surprise_threshold:
                    continue
                if cfg.use_fast_state:
                    model(
                        token_batch,
                        teach_signal=teach_signal,
                        surprise_value=surprise_override,
                        fast_state=fast_state,
                    )
                else:
                    model(token_batch, teach_signal=teach_signal, surprise_value=surprise_override)
                _collect_metrics(model, stats)

        _teardown_memorization_context(model, prev_allowed, prev_threshold, prev_layers)
        return stats


def memorize_sequence(
    model,
    tokenizer: SentencePieceTokenizer,
    text: str,
    device: torch.device,
    cfg: MemorizeConfig,
    *,
    fast_state=None,
    teach_mask: torch.Tensor | None = None,
) -> dict[str, float]:
    if not text:
        return {}
    tokens = tokenizer.encode(text)
    if tokens.size(0) < 2:
        return {}
    batch = tokens.to(device).unsqueeze(0)
    return memorize_tokens(model, batch, cfg, fast_state=fast_state, teach_mask=teach_mask)
```

### File: `src/nested_learning/model.py`

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
        if teach_signal is not None:
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

    def init_fast_state(self) -> ModelFastState:
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
                )
                states.append(state)
            elif isinstance(block, TransformerBlock):
                state = build_block_fast_state(
                    titan_module=None,
                    cms_blocks={},
                    specs=(),
                    optimizer_configs={},
                    default_lr=0.0,
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

### File: `src/nested_learning/optim/__init__.py`

```python

```

### File: `src/nested_learning/optim/deep.py`

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class DeepMomentumState:
    grad_avg: Optional[torch.Tensor] = None
    sq_avg: Optional[torch.Tensor] = None


class DeepMomentum(nn.Module):
    """Implements momentum variants described in the NL paper."""

    def __init__(
        self,
        *,
        beta: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        variant: str = "preconditioned",
    ) -> None:
        super().__init__()
        self.beta = beta
        self.beta2 = beta2
        self.eps = eps
        self.variant = variant
        self.state: dict[str, DeepMomentumState] = {}
        self.nonlinearity = nn.Tanh() if variant in {"dmgd", "muon"} else nn.Identity()
        self.last_metrics: dict[str, float] = {}

    def reset_state(self) -> None:
        self.state.clear()

    def _precondition(self, grad: torch.Tensor, state: DeepMomentumState) -> torch.Tensor:
        if state.sq_avg is None or state.sq_avg.shape != grad.shape:
            state.sq_avg = torch.zeros_like(grad)
        state.sq_avg.mul_(self.beta2).addcmul_(grad, grad, value=1 - self.beta2)
        denom = state.sq_avg.sqrt().add_(self.eps)
        return grad / denom

    def _nl_precondition(
        self,
        grad: torch.Tensor,
        context: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        metrics: dict[str, float] = {
            "ctx_norm": 0.0,
            "proj_norm": 0.0,
            "proj_skipped": 0.0,
        }
        if context is None:
            return grad, metrics
        ctx = context
        if ctx.ndim > 1:
            ctx = ctx.reshape(-1, ctx.shape[-1]).mean(dim=0)
        ctx_norm = torch.norm(ctx)
        metrics["ctx_norm"] = ctx_norm.item()

        if ctx_norm > 0:
            if grad.ndim == 0 or grad.shape[-1] != ctx.shape[-1]:
                metrics["proj_skipped"] = 1.0
                return grad, metrics
            unit = ctx / (ctx_norm + self.eps)
            # Project grad orthogonal to context (rank-1 projector).
            projection = (grad * unit).sum(dim=-1, keepdim=True) * unit
            update = grad - projection
            metrics["proj_norm"] = torch.norm(update).item()
            return update, metrics
        return grad, metrics

    def forward(  # type: ignore[override]
        self,
        grad: torch.Tensor,
        *,
        context: torch.Tensor | None = None,
        param_key: str | None = None,
    ) -> torch.Tensor:
        key = param_key or "__default__"
        state = self.state.get(key)
        if state is None:
            state = DeepMomentumState()
            self.state[key] = state
        if state.grad_avg is None or state.grad_avg.shape != grad.shape:
            state.grad_avg = torch.zeros_like(grad)
        self.last_metrics = {}
        update = grad
        if self.variant in {"preconditioned", "muon"}:
            update = self._precondition(grad, state)
        if self.variant == "l2_objective":
            update = grad + 0.1 * torch.mean(grad, dim=-1, keepdim=True)
        if self.variant == "nl_l2_precond":
            update, metrics = self._nl_precondition(grad, context)
            self.last_metrics.update(metrics)
        if self.variant in {"dmgd", "muon"}:
            update = self.nonlinearity(update)
        state.grad_avg.mul_(self.beta).add_(update, alpha=1 - self.beta)
        return state.grad_avg
```

### File: `src/nested_learning/optim/factory.py`

```python
from __future__ import annotations

from typing import Any, Dict

from .deep import DeepMomentum


def build_optimizer(config: Dict[str, Any]) -> DeepMomentum:
    opt_type = config.get("type", "deep_momentum").lower()
    if opt_type != "deep_momentum":
        raise ValueError(f"Unsupported optimizer type {opt_type}")
    params = config.get("params", {})
    return DeepMomentum(**params)
```

### File: `src/nested_learning/optim/m3.py`

```python
from __future__ import annotations

from typing import Iterable

import torch


def _newton_schulz(matrix: torch.Tensor, steps: int, eps: float = 1e-6) -> torch.Tensor:
    if matrix.ndim != 2:
        raise ValueError("Newton-Schulz expects a 2D matrix")
    dtype = matrix.dtype
    device = matrix.device
    m, n = matrix.shape
    x = matrix
    norm = torch.linalg.norm(x)
    x = x / (norm + eps)
    eye = torch.eye(n, device=device, dtype=dtype)
    for _ in range(steps):
        x = 0.5 * x @ (3.0 * eye - x.T @ x)
    return x


def _orthogonalize(tensor: torch.Tensor, steps: int, eps: float) -> torch.Tensor:
    if tensor.ndim < 2:
        return tensor
    mat = tensor.reshape(tensor.shape[0], -1)
    ortho = _newton_schulz(mat, steps=steps, eps=eps)
    return ortho.reshape_as(tensor)


class M3(torch.optim.Optimizer):
    """
    Multi-scale Momentum Muon (M3) optimizer (Nested Learning paper, Algorithm 1).

    This is a paper-faithful implementation for 2D weight tensors:
      - M1: fast momentum
      - M2: slow momentum (updated every `slow_chunk` steps)
      - V: second moment
      - O1/O2: Newton-Schulz orthogonalized momenta
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        *,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        beta3: float = 0.9,
        alpha: float = 1.0,
        eps: float = 1e-8,
        ns_steps: int = 3,
        slow_chunk: int = 100,
        weight_decay: float = 0.0,
    ) -> None:
        defaults = dict(
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            beta3=beta3,
            alpha=alpha,
            eps=eps,
            ns_steps=ns_steps,
            slow_chunk=slow_chunk,
            weight_decay=weight_decay,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            beta3 = group["beta3"]
            alpha = group["alpha"]
            eps = group["eps"]
            ns_steps = group["ns_steps"]
            slow_chunk = group["slow_chunk"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if weight_decay != 0.0:
                    grad = grad.add(p, alpha=weight_decay)
                state = self.state[p]
                if not state:
                    state["step"] = 0
                    state["m1"] = torch.zeros_like(p)
                    state["m2"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                    state["slow_buffer"] = torch.zeros_like(p)
                    state["o2"] = torch.zeros_like(p)
                state["step"] += 1
                m1 = state["m1"]
                m2 = state["m2"]
                v = state["v"]
                slow_buffer = state["slow_buffer"]

                m1.add_(grad, alpha=beta1)
                v.addcmul_(grad, grad, value=beta2)
                slow_buffer.add_(grad)

                o1 = _orthogonalize(m1, steps=ns_steps, eps=eps)
                o2 = state["o2"]
                denom = v.sqrt().add_(eps)
                update = (o1 + alpha * o2) / denom
                p.add_(update, alpha=-lr)

                if slow_chunk > 0 and state["step"] % slow_chunk == 0:
                    # Paper Algorithm 1 uses the updated slow momentum term in the *next* chunk.
                    # Compute it after applying the current step update to avoid off-by-one usage.
                    m2.add_(slow_buffer, alpha=beta3)
                    slow_buffer.zero_()
                    state["o2"] = _orthogonalize(m2, steps=ns_steps, eps=eps)
        return loss
```

### File: `src/nested_learning/optim/manager.py`

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
        with torch.no_grad():
            for name, param in module.named_parameters():
                if not param.requires_grad:
                    continue
                grad = grads.get(name)
                if grad is None:
                    continue
                update = optimizer(grad, context=context, param_key=name)
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
        with torch.no_grad():
            for name, param in params.items():
                grad = grads.get(name)
                if grad is None:
                    updated[name] = param
                    continue
                update = optimizer(grad, context=context, param_key=name)
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

### File: `src/nested_learning/titan/__init__.py`

```python

```

### File: `src/nested_learning/titan/memory.py`

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn

from ..assoc_memory import AssocMemory


@dataclass
class TitanMemoryConfig:
    dim: int
    hidden_multiplier: int = 4
    layers: int = 2
    activation: str = "gelu"


def _activation(name: str) -> nn.Module:
    if name.lower() == "relu":
        return nn.ReLU()
    if name.lower() == "gelu":
        return nn.GELU()
    if name.lower() == "silu":
        return nn.SiLU()
    msg = f"Unsupported activation {name}"
    raise ValueError(msg)


class TitanMemory(AssocMemory):
    """Simplified TITAN-style associative memory."""

    def __init__(self, config: TitanMemoryConfig):
        super().__init__()
        self.config = config
        hidden = config.dim * config.hidden_multiplier
        blocks = []
        activation = _activation(config.activation)
        for layer_idx in range(config.layers - 1):
            blocks.extend([nn.Linear(config.dim if layer_idx == 0 else hidden, hidden), activation])
        blocks.append(nn.Linear(hidden if config.layers > 1 else config.dim, config.dim))
        self.net = nn.Sequential(*blocks)
        self.norm = nn.LayerNorm(config.dim)
        self.grad_clip = 1.0

    def forward(self, query: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        attn = self.net(query)
        if self.training and self.grad_clip > 0:
            with torch.no_grad():
                norm = attn.norm(dim=-1, keepdim=True)
                scale = torch.clamp(norm / self.grad_clip, min=1.0)
            attn = attn / scale
        return self.norm(attn)

    def surprise(self, residual: torch.Tensor) -> torch.Tensor:
        return residual.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def update(
        self,
        *,
        key: torch.Tensor,
        value: torch.Tensor,
        error_signal: torch.Tensor | None = None,
        lr: float = 1e-3,
    ) -> None:
        with torch.enable_grad():
            key_detached = key.detach().requires_grad_(True)
            prediction = self.forward(key_detached)
            target = value.detach()
            if error_signal is None:
                loss = torch.mean((prediction - target) ** 2)
            else:
                loss = torch.mean(error_signal * prediction)
        grads = torch.autograd.grad(loss, list(self.net.parameters()), retain_graph=False)
        for param, grad in zip(self.net.parameters(), grads, strict=False):
            if grad is None:
                continue
            param.add_(grad, alpha=-lr)

    @torch.no_grad()
    def apply_deltas(self, deltas: Dict[str, torch.Tensor], scale: float = 1.0) -> None:
        for name, tensor in deltas.items():
            target = dict(self.named_parameters()).get(name)
            if target is None:
                continue
            target.add_(tensor, alpha=scale)
```

### File: `src/nested_learning/titan/model.py`

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, cast

import torch
import torch.nn as nn

from ..backbones import AttentionConfig, SelfAttention
from ..fast_state import BlockFastState, ModelFastState, build_block_fast_state
from ..functional import (
    call_with_deltas,
    call_with_params,
    grads_to_dict,
    params_with_deltas,
    require_grad_params,
)
from ..hope.self_mod import SelfModifier
from ..levels import LevelSpec
from ..optim.manager import LevelConfig, LevelOptimizerManager
from ..titan.memory import TitanMemory, TitanMemoryConfig


@dataclass
class TitanOnlyModelConfig:
    vocab_size: int
    dim: int
    num_layers: int
    heads: int
    titan_level: LevelSpec
    optimizers: Dict[str, dict] | None = None
    teach_scale: float = 1.0
    teach_clip: float = 0.0
    teach_schedule: Dict[str, float] | None = None
    qk_l2_norm: bool = False
    local_conv_window: int | None = None
    titan_hidden_multiplier: int = 4
    activation: str = "gelu"
    self_mod_hidden: int = 4
    self_mod_lr: float = 1e-3
    surprise_threshold: float | None = None
    surprise_metric: str = "l2"
    freeze_backbone: bool = False


class TitanOnlyBlock(nn.Module):
    def __init__(self, config: TitanOnlyModelConfig):
        super().__init__()
        self.config = config
        self.surprise_threshold: float | None = None
        self.surprise_metric: str = "l2"
        self.enabled: bool = True
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
        self.self_modifier = SelfModifier(config.dim, hidden_multiplier=config.self_mod_hidden)
        self.dropout = nn.Dropout(0.0)
        self.norm = nn.LayerNorm(config.dim)
        level_config = LevelConfig(
            specs=[config.titan_level],
            optimizer_configs=config.optimizers or {},
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
        else:
            if fast_state.titan_params is None:
                raise ValueError(
                    "fast_state.titan_params is required for TitanOnlyBlock fast-state forward"
                )
            mem_out = call_with_deltas(self.titan_memory, fast_state.titan_params, attn_out)
        combined = attn_out + mem_out
        if teach_signal is not None:
            if fast_state is None:
                self._update_titan(attn_out, mem_out, teach_signal, surprise_value)
            else:
                self._update_titan_fast(fast_state, attn_out, mem_out, teach_signal, surprise_value)
        if fast_state is None:
            self.level_manager.tick()
        else:
            fast_state.level_manager.tick()
        return self.norm(combined)

    def set_surprise_threshold(self, threshold: float | None) -> None:
        self.surprise_threshold = threshold

    def set_surprise_metric(self, metric: str) -> None:
        self.surprise_metric = str(metric).strip().lower()

    def set_enabled(self, enabled: bool) -> None:
        self.enabled = enabled

    def _passes_surprise(self, surprise_value: float | None) -> bool:
        if self.surprise_threshold is None:
            return True
        if surprise_value is None:
            return False
        return surprise_value >= self.surprise_threshold

    def _update_titan(
        self,
        attn_out: torch.Tensor,
        mem_out: torch.Tensor,
        teach_signal: torch.Tensor,
        surprise_value: float | None,
    ) -> None:
        level_name = self.config.titan_level.name
        if not self.enabled:
            return
        if not self.level_manager.should_update(level_name):
            return
        if not self._passes_surprise(surprise_value):
            return
        # Use full sequence for granular updates (Critique P1)
        # Note: We intentionally do not pool over dim=1 (sequence) here.
        modifier = self.self_modifier(
            key=attn_out.detach(),
            value=mem_out.detach(),
            error_signal=teach_signal.detach(),
        )
        context_vec = attn_out.detach().mean(dim=(0, 1))
        with torch.enable_grad():
            query = attn_out.detach()
            target = (teach_signal.detach() + modifier).detach()
            base_params = {name: param for name, param in self.titan_memory.named_parameters()}
            params_req = require_grad_params(base_params)
            prediction = call_with_params(self.titan_memory, params_req, query)
            loss_terms = nn.functional.mse_loss(prediction, target, reduction="none")
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
        self.level_manager.apply_module_grads(
            level_name,
            self.titan_memory,
            grads_dict,
            context=context_vec,
            force=True,
        )
        # Pop metrics to avoid stale entries even if we do not log them yet.
        self.level_manager.pop_last_metrics(level_name)

    def _update_titan_fast(
        self,
        fast_state: BlockFastState,
        attn_out: torch.Tensor,
        mem_out: torch.Tensor,
        teach_signal: torch.Tensor,
        surprise_value: float | None,
    ) -> None:
        level_name = self.config.titan_level.name
        if not self.enabled:
            return
        if not fast_state.level_manager.should_update(level_name):
            return
        if not self._passes_surprise(surprise_value):
            return
        if fast_state.titan_params is None:
            return
        modifier = self.self_modifier(
            key=attn_out.detach(),
            value=mem_out.detach(),
            error_signal=teach_signal.detach(),
        )
        context_vec = attn_out.detach().mean(dim=(0, 1))
        base_params = fast_state.titan_params
        forward_params = params_with_deltas(self.titan_memory, base_params)
        params_req = require_grad_params(forward_params)
        with torch.enable_grad():
            query = attn_out.detach()
            target = (teach_signal.detach() + modifier).detach()
            prediction = call_with_params(self.titan_memory, params_req, query)
            loss_terms = nn.functional.mse_loss(prediction, target, reduction="none")
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
        updated, _magnitude = fast_state.level_manager.apply_grads(
            level_name,
            base_params,
            grads_dict,
            context=context_vec,
            force=False,
        )
        fast_state.titan_params = updated
        fast_state.level_manager.pop_last_metrics(level_name)


class TitanOnlyModel(nn.Module):
    def __init__(self, config: TitanOnlyModelConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.dim)
        self.blocks = nn.ModuleList([TitanOnlyBlock(config) for _ in range(config.num_layers)])
        self.norm = nn.LayerNorm(config.dim)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight
        self._runtime_teach_scale = config.teach_scale
        self._runtime_teach_clip = config.teach_clip
        self._surprise_threshold: float | None = None
        self._surprise_metric = "l2"
        self._updates_enabled: bool = True
        self.set_surprise_metric(config.surprise_metric)
        self.set_surprise_threshold(config.surprise_threshold)
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
            cast(TitanOnlyBlock, block).set_surprise_threshold(threshold)

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
            cast(TitanOnlyBlock, block).set_surprise_metric(normalized)

    def get_surprise_metric(self) -> str:
        return self._surprise_metric

    def set_allowed_update_levels(self, levels: set[str] | None) -> None:
        enabled = True
        if levels is not None and "titan" not in levels and len(levels) > 0:
            enabled = False
        self._updates_enabled = enabled
        for block in self.blocks:
            cast(TitanOnlyBlock, block).set_enabled(enabled)

    def get_allowed_update_levels(self) -> set[str] | None:
        if self._updates_enabled:
            return {"titan"}
        return set()

    def forward(
        self,
        tokens: torch.Tensor,
        *,
        teach_signal: torch.Tensor | None = None,
        fast_state: ModelFastState | None = None,
        surprise_value: float | None = None,
    ) -> torch.Tensor:
        require_external = self._surprise_metric in {"loss", "logit_entropy"}
        if require_external and self._surprise_threshold is not None:
            if teach_signal is not None and surprise_value is None:
                raise ValueError(
                    f"surprise_metric={self._surprise_metric} requires passing surprise_value "
                    "when model.surprise_threshold is set."
                )
        x = self.embed(tokens)
        if fast_state is not None and len(fast_state.blocks) != len(self.blocks):
            raise ValueError("fast_state.blocks length does not match model.blocks")
        base_surprise = surprise_value
        for idx, block in enumerate(self.blocks):
            scaled_signal = None
            if teach_signal is not None:
                scaled_signal = teach_signal * self._runtime_teach_scale
                if self._runtime_teach_clip > 0:
                    with torch.no_grad():
                        norm = scaled_signal.norm(dim=-1, keepdim=True)
                        scale = torch.clamp(norm / self._runtime_teach_clip, min=1.0)
                    scaled_signal = scaled_signal / scale
            block_surprise = base_surprise
            if (
                scaled_signal is not None
                and base_surprise is None
                and self._surprise_metric == "l2"
            ):
                block_surprise = float(scaled_signal.norm(dim=-1).mean().item())
            block_state = None if fast_state is None else fast_state.blocks[idx]
            x = block(  # type: ignore[arg-type]
                x,
                teach_signal=scaled_signal,
                surprise_value=block_surprise,
                fast_state=block_state,
            )
        x = self.norm(x)
        return self.lm_head(x)

    def freeze_backbone(self) -> None:
        """
        Freeze shared transformer components; leave TITAN memory/trainable paths active.
        """
        for p in self.embed.parameters():
            p.requires_grad = False
        for p in self.norm.parameters():
            p.requires_grad = False
        for p in self.lm_head.parameters():
            p.requires_grad = False
        for block in self.blocks:
            typed_block = cast(TitanOnlyBlock, block)
            for p in typed_block.attn.parameters():
                p.requires_grad = False

    def init_fast_state(self) -> ModelFastState:
        states = []
        for block in self.blocks:
            typed_block = cast(TitanOnlyBlock, block)
            specs = [typed_block.config.titan_level]
            state = build_block_fast_state(
                titan_module=typed_block.titan_memory,
                cms_blocks={},
                specs=specs,
                optimizer_configs=typed_block.config.optimizers or {},
                default_lr=typed_block.config.self_mod_lr,
            )
            states.append(state)
        return ModelFastState(blocks=states)
```

### File: `src/nested_learning/titan/self_modifying.py`

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F
from torch import nn
from torch.func import grad, vmap


@dataclass(frozen=True)
class SelfModifyingTitansConfig:
    dim: int
    eta_scale: float = 1e-3
    chunk_size_other: int = 1
    chunk_size_memory: int | None = None
    objective: str = "l2"
    stopgrad_vhat: bool = True
    use_rank1_precond: bool = True
    use_alpha: bool = True
    momentum: float = 0.0
    qk_l2_norm: bool = True
    adaptive_q: bool = False
    use_skip: bool = True
    local_conv_window: int | None = 4
    eps: float = 1e-6

    def __post_init__(self) -> None:
        if self.dim <= 0:
            raise ValueError("dim must be positive")
        if self.eta_scale <= 0:
            raise ValueError("eta_scale must be positive")
        if self.chunk_size_other <= 0:
            raise ValueError("chunk_size_other must be positive")
        if self.chunk_size_memory is not None and self.chunk_size_memory <= 0:
            raise ValueError("chunk_size_memory must be positive")
        if self.objective not in {"l2", "dot"}:
            raise ValueError("objective must be one of {'l2', 'dot'}")
        if not (0.0 <= self.momentum < 1.0):
            raise ValueError("momentum must be in [0, 1)")
        if self.local_conv_window is not None and int(self.local_conv_window) <= 0:
            raise ValueError("local_conv_window must be positive")
        if self.chunk_size_memory is None:
            object.__setattr__(self, "chunk_size_memory", int(self.chunk_size_other))


@dataclass
class ResidualMLPMemoryState:
    w1: torch.Tensor
    w2: torch.Tensor
    w_skip: torch.Tensor | None = None
    m_w1: torch.Tensor | None = None
    m_w2: torch.Tensor | None = None
    m_w_skip: torch.Tensor | None = None

    def clone(self) -> "ResidualMLPMemoryState":
        return ResidualMLPMemoryState(
            w1=self.w1.detach().clone(),
            w2=self.w2.detach().clone(),
            w_skip=None if self.w_skip is None else self.w_skip.detach().clone(),
            m_w1=None if self.m_w1 is None else self.m_w1.detach().clone(),
            m_w2=None if self.m_w2 is None else self.m_w2.detach().clone(),
            m_w_skip=None if self.m_w_skip is None else self.m_w_skip.detach().clone(),
        )


@dataclass
class SelfModifyingTitansState:
    """
    Fast state for self-modifying Titans.

    Each memory M_□ is a residual MLP (Eq. 91) whose initial parameters are meta-learned
    (stored in the module) and cloned into this fast state per context.
    """

    k: ResidualMLPMemoryState
    v: ResidualMLPMemoryState
    q: ResidualMLPMemoryState
    eta: ResidualMLPMemoryState
    alpha: ResidualMLPMemoryState
    memory: ResidualMLPMemoryState

    def clone(self) -> "SelfModifyingTitansState":
        return SelfModifyingTitansState(
            k=self.k.clone(),
            v=self.v.clone(),
            q=self.q.clone(),
            eta=self.eta.clone(),
            alpha=self.alpha.clone(),
            memory=self.memory.clone(),
        )


class ResidualMLPMemory(nn.Module):
    def __init__(
        self,
        *,
        in_dim: int,
        out_dim: int,
        hidden_dim: int,
        activation: Callable[[torch.Tensor], torch.Tensor],
        use_skip: bool = True,
    ) -> None:
        super().__init__()
        if in_dim <= 0 or out_dim <= 0 or hidden_dim <= 0:
            raise ValueError("in_dim/out_dim/hidden_dim must be positive")
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.hidden_dim = int(hidden_dim)
        self.activation = activation
        self.use_skip = bool(use_skip)
        self.w2 = nn.Linear(self.in_dim, self.hidden_dim, bias=False)
        self.w1 = nn.Linear(self.hidden_dim, self.out_dim, bias=False)
        self.w_skip: nn.Linear | None = None
        if self.use_skip and self.in_dim != self.out_dim:
            self.w_skip = nn.Linear(self.in_dim, self.out_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        hidden = self.activation(self.w2(x))
        out = self.w1(hidden)
        if self.w_skip is not None:
            return self.w_skip(x) + out
        if out.shape[-1] == x.shape[-1]:
            return x + out
        return out


class SelfModifyingTitans(nn.Module):
    """
    Self-modifying Titans (Nested Learning paper, Eqs. 83–93), correctness-first.

    - Multiple memories: M_k, M_v, M_q, M_eta, M_alpha, M_memory.
    - Each memory is a 2-layer residual MLP (Eq. 91).
    - Updates are performed on fast state using chunked DGD-like rule (Eq. 90/93).

    Note: This implementation prioritizes semantic fidelity and testability over speed.
    """

    def __init__(self, config: SelfModifyingTitansConfig):
        super().__init__()
        self.config = config
        dim = config.dim
        hidden = dim
        act = F.gelu
        self.local_conv: nn.Conv1d | None = None
        if config.local_conv_window is not None:
            window = int(config.local_conv_window)
            self.local_conv = nn.Conv1d(
                dim,
                dim,
                kernel_size=window,
                groups=dim,
                padding=0,
                bias=False,
            )
        self.w_q = nn.Linear(dim, dim, bias=False)
        self.m_k = ResidualMLPMemory(
            in_dim=dim, out_dim=dim, hidden_dim=hidden, activation=act, use_skip=config.use_skip
        )
        self.m_v = ResidualMLPMemory(
            in_dim=dim, out_dim=dim, hidden_dim=hidden, activation=act, use_skip=config.use_skip
        )
        self.m_q = ResidualMLPMemory(
            in_dim=dim, out_dim=dim, hidden_dim=hidden, activation=act, use_skip=config.use_skip
        )
        self.m_eta = ResidualMLPMemory(
            in_dim=dim, out_dim=1, hidden_dim=hidden, activation=act, use_skip=config.use_skip
        )
        self.m_alpha = ResidualMLPMemory(
            in_dim=dim, out_dim=1, hidden_dim=hidden, activation=act, use_skip=config.use_skip
        )
        self.m_memory = ResidualMLPMemory(
            in_dim=dim, out_dim=dim, hidden_dim=hidden, activation=act, use_skip=config.use_skip
        )

    def init_fast_state(self) -> SelfModifyingTitansState:
        return SelfModifyingTitansState(
            k=self._init_memory_state(self.m_k),
            v=self._init_memory_state(self.m_v),
            q=self._init_memory_state(self.m_q),
            eta=self._init_memory_state(self.m_eta),
            alpha=self._init_memory_state(self.m_alpha),
            memory=self._init_memory_state(self.m_memory),
        )

    def apply_updates_inplace(
        self,
        x: torch.Tensor,
        *,
        chunk_size_other: int | None = None,
        chunk_size_memory: int | None = None,
    ) -> None:
        """
        Apply the self-modifying update rule to the *module parameters* in-place.

        This is intended to be called in an explicit "update pass" under `torch.no_grad()`
        (e.g., after an outer backward), so we avoid mixing differentiable reads with
        in-place writes during the same autograd graph.
        """
        state = self.init_fast_state()
        _out, updated = self.forward_with_updates(
            x,
            state,
            chunk_size_other=chunk_size_other,
            chunk_size_memory=chunk_size_memory,
        )
        self._load_state_mean_(updated)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self._apply_local_conv(x)
        q = self.m_q(x) if self.config.adaptive_q else self.w_q(x)
        if self.config.qk_l2_norm:
            q = F.normalize(q, dim=-1, eps=self.config.eps)
        return self.m_memory(q)

    def forward_with_state(
        self,
        x: torch.Tensor,
        state: SelfModifyingTitansState,
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("Expected x to have shape (B, T, D)")
        batch, _seq_len, dim = x.shape
        if dim != self.config.dim:
            raise ValueError(f"Expected dim={self.config.dim}, got {dim}")
        state = self._ensure_batched_state(state, batch)
        x = self._apply_local_conv(x)
        q = (
            self._memory_forward(x, state.q, meta=self.m_q)
            if self.config.adaptive_q
            else self.w_q(x)
        )
        if self.config.qk_l2_norm:
            q = F.normalize(q, dim=-1, eps=self.config.eps)
        return self._memory_forward(q, state.memory, meta=self.m_memory)

    def forward_with_updates(
        self,
        x: torch.Tensor,
        state: SelfModifyingTitansState,
        *,
        chunk_size_other: int | None = None,
        chunk_size_memory: int | None = None,
    ) -> tuple[torch.Tensor, SelfModifyingTitansState]:
        if x.ndim != 3:
            raise ValueError("Expected x to have shape (B, T, D)")
        batch, seq_len, dim = x.shape
        if dim != self.config.dim:
            raise ValueError(f"Expected dim={self.config.dim}, got {dim}")
        state = self._ensure_batched_state(state, batch)
        x = self._apply_local_conv(x)
        other_chunk = int(
            self.config.chunk_size_other if chunk_size_other is None else chunk_size_other
        )
        memory_chunk_cfg = self.config.chunk_size_memory
        if memory_chunk_cfg is None:
            memory_chunk_cfg = self.config.chunk_size_other
        memory_chunk = int(memory_chunk_cfg if chunk_size_memory is None else chunk_size_memory)
        if other_chunk <= 0 or memory_chunk <= 0:
            raise ValueError("chunk sizes must be positive")

        outputs: list[torch.Tensor] = []
        other_k: list[torch.Tensor] = []
        other_v: list[torch.Tensor] = []
        other_eta: list[torch.Tensor] = []
        other_alpha: list[torch.Tensor] = []
        memory_k: list[torch.Tensor] = []
        memory_v: list[torch.Tensor] = []
        memory_eta: list[torch.Tensor] = []
        memory_alpha: list[torch.Tensor] = []

        def _next_boundary(idx: int, *, chunk_size: int) -> int:
            if chunk_size <= 0:
                raise ValueError("chunk_size must be positive")
            return min(((idx // chunk_size) + 1) * chunk_size, seq_len)

        with torch.no_grad():
            idx = 0
            while idx < seq_len:
                next_other = _next_boundary(idx, chunk_size=other_chunk)
                next_memory = _next_boundary(idx, chunk_size=memory_chunk)
                end = min(next_other, next_memory, seq_len)
                x_chunk = x[:, idx:end, :]

                k_chunk = self._memory_forward(x_chunk, state.k)
                v_chunk = self._memory_forward(x_chunk, state.v)
                q_chunk = (
                    self._memory_forward(x_chunk, state.q)
                    if self.config.adaptive_q
                    else self.w_q(x_chunk)
                )
                if self.config.qk_l2_norm:
                    k_chunk = F.normalize(k_chunk, dim=-1, eps=self.config.eps)
                    q_chunk = F.normalize(q_chunk, dim=-1, eps=self.config.eps)
                eta_chunk = self._memory_forward(x_chunk, state.eta).squeeze(-1)
                eta_chunk = F.softplus(eta_chunk) * self.config.eta_scale
                if self.config.use_alpha:
                    alpha_chunk = self._memory_forward(x_chunk, state.alpha).squeeze(-1)
                    alpha_chunk = torch.sigmoid(alpha_chunk)
                else:
                    alpha_chunk = torch.ones_like(eta_chunk)
                o_chunk = self._memory_forward(q_chunk, state.memory)
                outputs.append(o_chunk)

                other_k.append(k_chunk)
                other_v.append(v_chunk)
                other_eta.append(eta_chunk)
                other_alpha.append(alpha_chunk)
                memory_k.append(k_chunk)
                memory_v.append(v_chunk)
                memory_eta.append(eta_chunk)
                memory_alpha.append(alpha_chunk)

                idx = end

                if idx == next_other and other_k:
                    other_memories: tuple[str, ...] = ("k", "v", "eta")
                    if self.config.adaptive_q:
                        other_memories = (*other_memories, "q")
                    if self.config.use_alpha:
                        other_memories = (*other_memories, "alpha")
                    self._apply_chunk_update_seq(
                        state,
                        k_seq=torch.cat(other_k, dim=1),
                        v_seq=torch.cat(other_v, dim=1),
                        eta_seq=torch.cat(other_eta, dim=1),
                        alpha_seq=torch.cat(other_alpha, dim=1),
                        memories=other_memories,
                    )
                    other_k.clear()
                    other_v.clear()
                    other_eta.clear()
                    other_alpha.clear()

                if idx == next_memory and memory_k:
                    self._apply_chunk_update_seq(
                        state,
                        k_seq=torch.cat(memory_k, dim=1),
                        v_seq=torch.cat(memory_v, dim=1),
                        eta_seq=torch.cat(memory_eta, dim=1),
                        alpha_seq=torch.cat(memory_alpha, dim=1),
                        memories=("memory",),
                    )
                    memory_k.clear()
                    memory_v.clear()
                    memory_eta.clear()
                    memory_alpha.clear()

            if other_k:
                other_memories = ("k", "v", "eta")
                if self.config.adaptive_q:
                    other_memories = (*other_memories, "q")
                if self.config.use_alpha:
                    other_memories = (*other_memories, "alpha")
                self._apply_chunk_update_seq(
                    state,
                    k_seq=torch.cat(other_k, dim=1),
                    v_seq=torch.cat(other_v, dim=1),
                    eta_seq=torch.cat(other_eta, dim=1),
                    alpha_seq=torch.cat(other_alpha, dim=1),
                    memories=other_memories,
                )
            if memory_k:
                self._apply_chunk_update_seq(
                    state,
                    k_seq=torch.cat(memory_k, dim=1),
                    v_seq=torch.cat(memory_v, dim=1),
                    eta_seq=torch.cat(memory_eta, dim=1),
                    alpha_seq=torch.cat(memory_alpha, dim=1),
                    memories=("memory",),
                )

        return torch.cat(outputs, dim=1), state

    def _apply_local_conv(self, x: torch.Tensor) -> torch.Tensor:
        if self.local_conv is None:
            return x
        if x.ndim != 3:
            raise ValueError("Expected x to have shape (B, T, D)")
        kernel = int(self.local_conv.kernel_size[0])
        # Causal depthwise conv: only attends to past tokens.
        x_t = x.transpose(1, 2)
        x_t = F.pad(x_t, (kernel - 1, 0))
        x_t = self.local_conv(x_t)
        return x_t.transpose(1, 2)

    def _load_state_mean_(self, state: SelfModifyingTitansState) -> None:
        def _mean_weight(weight: torch.Tensor) -> torch.Tensor:
            return weight.mean(dim=0) if weight.ndim == 3 else weight

        def _copy(module: ResidualMLPMemory, mem: ResidualMLPMemoryState) -> None:
            module.w1.weight.copy_(_mean_weight(mem.w1))
            module.w2.weight.copy_(_mean_weight(mem.w2))
            if module.w_skip is None:
                return
            if mem.w_skip is None:
                raise RuntimeError("Expected w_skip state for projected residual memory")
            module.w_skip.weight.copy_(_mean_weight(mem.w_skip))

        with torch.no_grad():
            _copy(self.m_k, state.k)
            _copy(self.m_v, state.v)
            _copy(self.m_eta, state.eta)
            if self.config.use_alpha:
                _copy(self.m_alpha, state.alpha)
            _copy(self.m_memory, state.memory)
            if self.config.adaptive_q:
                _copy(self.m_q, state.q)

    def _apply_chunk_update(
        self,
        state: SelfModifyingTitansState,
        buffer: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]],
        *,
        memories: tuple[str, ...],
    ) -> None:
        if not buffer:
            return
        k_seq = torch.stack([item[0] for item in buffer], dim=1)
        v_seq = torch.stack([item[1] for item in buffer], dim=1)
        eta_seq = torch.stack([item[2] for item in buffer], dim=1)
        alpha_seq = torch.stack([item[3] for item in buffer], dim=1)
        self._apply_chunk_update_seq(
            state,
            k_seq=k_seq,
            v_seq=v_seq,
            eta_seq=eta_seq,
            alpha_seq=alpha_seq,
            memories=memories,
        )

    def _apply_chunk_update_seq(
        self,
        state: SelfModifyingTitansState,
        *,
        k_seq: torch.Tensor,
        v_seq: torch.Tensor,
        eta_seq: torch.Tensor,
        alpha_seq: torch.Tensor,
        memories: tuple[str, ...],
    ) -> None:
        steps = k_seq.size(1)
        dim = self.config.dim
        eye = (
            torch.eye(dim, device=k_seq.device, dtype=k_seq.dtype)
            .unsqueeze(0)
            .expand(k_seq.size(0), -1, -1)
        )

        boundary: dict[str, ResidualMLPMemoryState] = {
            name: getattr(state, name).clone() for name in memories
        }
        grads = {name: self._memory_grads_chunk(boundary[name], k_seq, v_seq) for name in memories}

        for t in range(steps):
            k_t = k_seq[:, t, :]
            eta_t = eta_seq[:, t]
            alpha_t = alpha_seq[:, t]
            kk = torch.einsum("bi,bj->bij", k_t, k_t)
            precond = alpha_t[:, None, None] * eye - eta_t[:, None, None] * kk
            for name in memories:
                fast = getattr(state, name)
                g1, g2, gskip = grads[name]
                self._apply_param_update(
                    fast,
                    (
                        g1[:, t, ...],
                        g2[:, t, ...],
                        None if gskip is None else gskip[:, t, ...],
                    ),
                    eta_t,
                    alpha_t,
                    precond,
                )

    def _memory_grads(
        self,
        frozen: ResidualMLPMemoryState,
        k_t: torch.Tensor,
        v_t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        with torch.enable_grad():
            w1 = frozen.w1.detach().requires_grad_(True)
            w2 = frozen.w2.detach().requires_grad_(True)
            w_skip = None
            if frozen.w_skip is not None:
                w_skip = frozen.w_skip.detach().requires_grad_(True)

            pred = self._memory_forward(k_t, ResidualMLPMemoryState(w1=w1, w2=w2, w_skip=w_skip))
            vhat = self._memory_forward(v_t, ResidualMLPMemoryState(w1=w1, w2=w2, w_skip=w_skip))
            if self.config.stopgrad_vhat:
                vhat = vhat.detach()

            if self.config.objective == "dot":
                loss = -(pred * vhat).sum(dim=-1)
            else:
                loss = (pred - vhat).pow(2).sum(dim=-1)
            loss_scalar = loss.sum()

            grads = torch.autograd.grad(
                loss_scalar,
                (w1, w2, w_skip) if w_skip is not None else (w1, w2),
                retain_graph=False,
                create_graph=False,
                allow_unused=False,
            )
        if w_skip is None:
            g1, g2 = grads
            return g1, g2, None
        g1, g2, gskip = grads
        return g1, g2, gskip

    def _memory_grads_chunk(
        self,
        frozen: ResidualMLPMemoryState,
        k_seq: torch.Tensor,
        v_seq: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Compute per-token gradients for an entire chunk in parallel (paper §8.2).

        Returns gradients with leading shape (B, T, ...).
        """
        w1 = frozen.w1.detach()
        w2 = frozen.w2.detach()
        w_skip = None if frozen.w_skip is None else frozen.w_skip.detach()

        k_tokens = k_seq.transpose(0, 1)
        v_tokens = v_seq.transpose(0, 1)

        if w_skip is None:

            def loss_fn_noskip(
                w1_t: torch.Tensor,
                w2_t: torch.Tensor,
                k_t: torch.Tensor,
                v_t: torch.Tensor,
            ) -> torch.Tensor:
                mem = ResidualMLPMemoryState(w1=w1_t, w2=w2_t)
                pred = self._memory_forward(k_t, mem)
                vhat = self._memory_forward(v_t, mem)
                if self.config.stopgrad_vhat:
                    vhat = vhat.detach()
                if self.config.objective == "dot":
                    loss = -(pred * vhat).sum(dim=-1)
                else:
                    loss = (pred - vhat).pow(2).sum(dim=-1)
                return loss.sum()

            grad_fn = grad(loss_fn_noskip, argnums=(0, 1))
            g1_tokens, g2_tokens = vmap(grad_fn, in_dims=(None, None, 0, 0))(
                w1,
                w2,
                k_tokens,
                v_tokens,
            )
            return g1_tokens.transpose(0, 1), g2_tokens.transpose(0, 1), None

        def loss_fn_skip(
            w1_t: torch.Tensor,
            w2_t: torch.Tensor,
            w_skip_t: torch.Tensor,
            k_t: torch.Tensor,
            v_t: torch.Tensor,
        ) -> torch.Tensor:
            mem = ResidualMLPMemoryState(w1=w1_t, w2=w2_t, w_skip=w_skip_t)
            pred = self._memory_forward(k_t, mem)
            vhat = self._memory_forward(v_t, mem)
            if self.config.stopgrad_vhat:
                vhat = vhat.detach()
            if self.config.objective == "dot":
                loss = -(pred * vhat).sum(dim=-1)
            else:
                loss = (pred - vhat).pow(2).sum(dim=-1)
            return loss.sum()

        grad_fn = grad(loss_fn_skip, argnums=(0, 1, 2))
        g1_tokens, g2_tokens, gskip_tokens = vmap(
            grad_fn,
            in_dims=(None, None, None, 0, 0),
        )(w1, w2, w_skip, k_tokens, v_tokens)
        return (
            g1_tokens.transpose(0, 1),
            g2_tokens.transpose(0, 1),
            gskip_tokens.transpose(0, 1),
        )

    def _apply_param_update(
        self,
        fast: ResidualMLPMemoryState,
        grads: tuple[torch.Tensor, torch.Tensor, torch.Tensor | None],
        eta_t: torch.Tensor,
        alpha_t: torch.Tensor,
        precond: torch.Tensor,
    ) -> None:
        g1, g2, gskip = grads
        g1 = self._apply_momentum(fast, "m_w1", g1)
        g2 = self._apply_momentum(fast, "m_w2", g2)
        if self.config.use_rank1_precond:
            fast.w2 = torch.matmul(fast.w2, precond) - eta_t[:, None, None] * g2
        else:
            fast.w2 = alpha_t[:, None, None] * fast.w2 - eta_t[:, None, None] * g2
        fast.w1 = alpha_t[:, None, None] * fast.w1 - eta_t[:, None, None] * g1

        if fast.w_skip is None:
            return
        if gskip is None:
            raise RuntimeError("Expected w_skip grad to be present")
        gskip = self._apply_momentum(fast, "m_w_skip", gskip)
        if self.config.use_rank1_precond:
            fast.w_skip = torch.matmul(fast.w_skip, precond) - eta_t[:, None, None] * gskip
        else:
            fast.w_skip = alpha_t[:, None, None] * fast.w_skip - eta_t[:, None, None] * gskip

    def _apply_momentum(
        self,
        fast: ResidualMLPMemoryState,
        attr_name: str,
        grad: torch.Tensor,
    ) -> torch.Tensor:
        beta = float(self.config.momentum)
        if beta <= 0.0:
            return grad
        buf = getattr(fast, attr_name)
        if buf is None:
            buf = torch.zeros_like(grad)
        buf = beta * buf + grad
        setattr(fast, attr_name, buf)
        return buf

    def _init_memory_state(self, module: ResidualMLPMemory) -> ResidualMLPMemoryState:
        skip = None if module.w_skip is None else module.w_skip.weight.detach().clone()
        return ResidualMLPMemoryState(
            w1=module.w1.weight.detach().clone(),
            w2=module.w2.weight.detach().clone(),
            w_skip=skip,
        )

    def _ensure_batched_state(
        self, state: SelfModifyingTitansState, batch: int
    ) -> SelfModifyingTitansState:
        if state.k.w1.ndim == 2:
            return SelfModifyingTitansState(
                k=self._expand_memory_state(state.k, batch),
                v=self._expand_memory_state(state.v, batch),
                q=self._expand_memory_state(state.q, batch),
                eta=self._expand_memory_state(state.eta, batch),
                alpha=self._expand_memory_state(state.alpha, batch),
                memory=self._expand_memory_state(state.memory, batch),
            )
        if state.k.w1.ndim != 3:
            raise ValueError("SelfModifyingTitansState weights must be 2D or 3D tensors")
        if state.k.w1.size(0) != batch:
            raise ValueError(
                f"State batch mismatch: expected batch={batch}, got {state.k.w1.size(0)}"
            )
        return state

    def _expand_memory_state(
        self, mem: ResidualMLPMemoryState, batch: int
    ) -> ResidualMLPMemoryState:
        def _expand(t: torch.Tensor) -> torch.Tensor:
            return t.detach().clone().unsqueeze(0).repeat(batch, 1, 1)

        def _expand_opt(t: torch.Tensor | None) -> torch.Tensor | None:
            return None if t is None else _expand(t)

        return ResidualMLPMemoryState(
            w1=_expand(mem.w1),
            w2=_expand(mem.w2),
            w_skip=_expand_opt(mem.w_skip),
            m_w1=_expand_opt(mem.m_w1),
            m_w2=_expand_opt(mem.m_w2),
            m_w_skip=_expand_opt(mem.m_w_skip),
        )

    def _memory_forward(
        self,
        x: torch.Tensor,
        mem: ResidualMLPMemoryState,
        *,
        meta: ResidualMLPMemory | None = None,
    ) -> torch.Tensor:
        if meta is None:
            w2 = mem.w2
            w1 = mem.w1
            w_skip = mem.w_skip
        else:
            w2 = self._straight_through_meta(mem.w2, meta.w2.weight)
            w1 = self._straight_through_meta(mem.w1, meta.w1.weight)
            w_skip = None
            if mem.w_skip is not None:
                if meta.w_skip is None:
                    raise RuntimeError("Expected meta w_skip for projected residual memory")
                w_skip = self._straight_through_meta(mem.w_skip, meta.w_skip.weight)
        if x.ndim == 2:
            x_seq = x.unsqueeze(1)
            squeeze = True
        else:
            x_seq = x
            squeeze = False
        w2_t = w2.transpose(-1, -2)
        hidden = torch.matmul(x_seq, w2_t)
        hidden = F.gelu(hidden)
        w1_t = w1.transpose(-1, -2)
        out = torch.matmul(hidden, w1_t)
        if w_skip is not None:
            w_skip_t = w_skip.transpose(-1, -2)
            out = out + torch.matmul(x_seq, w_skip_t)
        elif out.size(-1) == x_seq.size(-1):
            out = out + x_seq
        if squeeze:
            return out.squeeze(1)
        return out

    @staticmethod
    def _straight_through_meta(fast: torch.Tensor, meta: torch.Tensor) -> torch.Tensor:
        if meta.ndim > fast.ndim:
            raise ValueError("meta tensor must have <= fast tensor rank")
        expanded = meta
        while expanded.ndim < fast.ndim:
            expanded = expanded.unsqueeze(0)
        return fast + (expanded - expanded.detach())
```

### File: `src/nested_learning/tokenizer.py`

```python
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import sentencepiece as spm
import torch


class SentencePieceTokenizer:
    def __init__(self, model_path: str | Path):
        self.processor = spm.SentencePieceProcessor(model_file=str(model_path))

    @property
    def vocab_size(self) -> int:
        return self.processor.vocab_size()

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = True) -> torch.Tensor:
        tokens: list[int] = []
        if add_bos:
            tokens.append(self.processor.bos_id())
        tokens.extend(self.processor.encode(text))
        if add_eos:
            tokens.append(self.processor.eos_id())
        return torch.tensor(tokens, dtype=torch.long)

    def batch_encode(self, texts: Sequence[str]) -> list[torch.Tensor]:
        return [self.encode(text) for text in texts]
```

### File: `src/nested_learning/tokenizer_coverage.py`

```python
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict

from .tokenizer import SentencePieceTokenizer


def compute_tokenizer_coverage_stats(
    tokenizer_path: Path,
    sample_file: Path,
    max_lines: int = 10_000,
) -> Dict[str, object]:
    """
    Compute tokenizer coverage statistics on a representative text sample.

    Returns a JSON-serialisable dictionary; shared by both the coverage CLI and
    the regression guard so they cannot drift apart silently.
    """

    tokenizer = SentencePieceTokenizer(tokenizer_path)
    total_words = 0
    total_tokens = 0
    total_chars = 0
    processed_lines = 0
    word_token_lengths: list[int] = []
    piece_lengths: Counter[int] = Counter()

    with sample_file.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            if idx >= max_lines:
                break
            stripped = line.strip()
            if not stripped:
                continue
            processed_lines += 1
            total_chars += len(stripped)
            words = stripped.split()
            if not words:
                continue
            total_words += len(words)
            encoded = tokenizer.encode(stripped, add_bos=False, add_eos=False)
            ids = encoded.tolist()
            total_tokens += len(ids)
            for word in words:
                word_tokens = tokenizer.encode(word, add_bos=False, add_eos=False).tolist()
                if not word_tokens:
                    continue
                word_token_lengths.append(len(word_tokens))
            for token_id in ids:
                piece = tokenizer.processor.id_to_piece(token_id)
                piece_lengths[len(piece)] += 1

    if total_words == 0 or not word_token_lengths:
        raise ValueError("Sample produced no words; double-check the sample_file path.")

    avg_tokens_per_word = total_tokens / total_words if total_words else 0.0
    pct_single_token = sum(1 for length in word_token_lengths if length == 1) / len(
        word_token_lengths
    )
    pct_two_or_less = sum(1 for length in word_token_lengths if length <= 2) / len(
        word_token_lengths
    )

    return {
        "tokenizer": str(tokenizer_path),
        "sample_file": str(sample_file),
        "lines_processed": processed_lines,
        "total_words": total_words,
        "total_tokens": total_tokens,
        "avg_tokens_per_word": avg_tokens_per_word,
        "pct_single_token_words": pct_single_token,
        "pct_two_or_less_tokens_words": pct_two_or_less,
        "avg_chars_per_word": total_chars / total_words,
        "piece_length_histogram": dict(piece_lengths.most_common(20)),
    }
```

### File: `src/nested_learning/training.py`

```python
from __future__ import annotations

import base64
import json
import os
import pickle
import random
from contextlib import nullcontext
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Dict, Protocol, Tuple, cast

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, DistributedSampler, IterableDataset

from .data import (
    MixtureShardDataset,
    ShardSourceConfig,
    SyntheticTextConfig,
    SyntheticTextDataset,
    TokenShardDataset,
    collate_batch,
)
from .levels import LevelSpec
from .logging_utils import BaseLogger, NullLogger, init_logger
from .model import HOPEModel, ModelConfig
from .optim.m3 import M3
from .titan.model import TitanOnlyModel, TitanOnlyModelConfig


@dataclass
class DistributedContext:
    rank: int
    world_size: int
    device: torch.device


def unwrap_config(cfg: DictConfig) -> DictConfig:
    """Hydra can wrap grouped configs (e.g., hope/pilot) under the group name."""
    if "model" in cfg:
        return cfg
    if "hope" in cfg:
        return cast(DictConfig, cfg.hope)
    if "ablations" in cfg:
        return cast(DictConfig, cfg.ablations)
    return cfg


def build_model_from_cfg(model_cfg: DictConfig) -> torch.nn.Module:
    model_type = model_cfg.get("type", "hope")
    optimizer_cfg: Dict[str, dict] = {}
    if "optimizers" in model_cfg:
        optimizer_cfg = cast(
            Dict[str, dict],
            OmegaConf.to_container(model_cfg.optimizers, resolve=True),
        )
    teach_scale = model_cfg.get("teach_scale", 1.0)
    teach_clip = model_cfg.get("teach_clip", 0.0)
    teach_schedule: Dict[str, float] = {}
    if "teach_schedule" in model_cfg:
        teach_schedule = cast(
            Dict[str, float],
            OmegaConf.to_container(model_cfg.teach_schedule, resolve=True),
        )
    qk_l2_norm = bool(model_cfg.get("qk_l2_norm", False))
    local_conv_window_raw = model_cfg.get("local_conv_window")
    local_conv_window = None if local_conv_window_raw is None else int(local_conv_window_raw)
    surprise_threshold_raw = model_cfg.get("surprise_threshold")
    surprise_threshold = (
        None if surprise_threshold_raw is None else float(surprise_threshold_raw)
    )
    surprise_metric = str(model_cfg.get("surprise_metric", "l2"))
    cms_use_layernorm = bool(model_cfg.get("cms_use_layernorm", True))
    if model_type == "titan":
        titan_spec = LevelSpec(**model_cfg.titan_level)
        titan_cfg = TitanOnlyModelConfig(
            vocab_size=model_cfg.vocab_size,
            dim=model_cfg.dim,
            num_layers=model_cfg.num_layers,
            heads=model_cfg.heads,
            titan_level=titan_spec,
            optimizers=optimizer_cfg,
            teach_scale=teach_scale,
            teach_clip=teach_clip,
            teach_schedule=teach_schedule,
            qk_l2_norm=qk_l2_norm,
            local_conv_window=local_conv_window,
            surprise_threshold=surprise_threshold,
            surprise_metric=surprise_metric,
            freeze_backbone=model_cfg.get("freeze_backbone", False),
            self_mod_lr=float(model_cfg.get("self_mod_lr", 1e-3)),
            self_mod_hidden=int(model_cfg.get("self_mod_hidden", 4)),
        )
        return TitanOnlyModel(titan_cfg)
    titan_spec = LevelSpec(**model_cfg.titan_level)
    cms_specs = [LevelSpec(**entry) for entry in model_cfg.cms_levels]
    self_mod_chunk_size_memory_raw = model_cfg.get("self_mod_chunk_size_memory")
    self_mod_chunk_size_memory = (
        None if self_mod_chunk_size_memory_raw is None else int(self_mod_chunk_size_memory_raw)
    )
    self_mod_local_conv_window_raw = model_cfg.get("self_mod_local_conv_window", 4)
    self_mod_local_conv_window = (
        None if self_mod_local_conv_window_raw is None else int(self_mod_local_conv_window_raw)
    )
    hope_cfg = ModelConfig(
        vocab_size=model_cfg.vocab_size,
        dim=model_cfg.dim,
        num_layers=model_cfg.num_layers,
        heads=model_cfg.heads,
        titan_level=titan_spec,
        cms_levels=cms_specs,
        cms_flush_partial_at_end=bool(model_cfg.get("cms_flush_partial_at_end", False)),
        cms_use_layernorm=cms_use_layernorm,
        optimizers=optimizer_cfg,
        teach_scale=teach_scale,
        teach_clip=teach_clip,
        teach_schedule=teach_schedule,
        gradient_checkpointing=model_cfg.get("gradient_checkpointing", False),
        surprise_threshold=surprise_threshold,
        surprise_metric=surprise_metric,
        freeze_backbone=model_cfg.get("freeze_backbone", False),
        qk_l2_norm=qk_l2_norm,
        local_conv_window=local_conv_window,
        self_mod_lr=float(model_cfg.get("self_mod_lr", 1e-3)),
        self_mod_hidden=int(model_cfg.get("self_mod_hidden", 4)),
        self_mod_chunk_size=int(model_cfg.get("self_mod_chunk_size", 1)),
        self_mod_chunk_size_memory=self_mod_chunk_size_memory,
        self_mod_objective=str(model_cfg.get("self_mod_objective", "l2")),
        self_mod_stopgrad_vhat=bool(model_cfg.get("self_mod_stopgrad_vhat", True)),
        self_mod_use_rank1_precond=bool(model_cfg.get("self_mod_use_rank1_precond", True)),
        self_mod_use_alpha=bool(model_cfg.get("self_mod_use_alpha", True)),
        self_mod_use_skip=bool(model_cfg.get("self_mod_use_skip", True)),
        self_mod_momentum=float(model_cfg.get("self_mod_momentum", 0.0)),
        self_mod_adaptive_q=bool(model_cfg.get("self_mod_adaptive_q", False)),
        self_mod_local_conv_window=self_mod_local_conv_window,
        transformer_mlp_hidden_multiplier=int(
            model_cfg.get("transformer_mlp_hidden_multiplier", 4)
        ),
        transformer_activation=str(model_cfg.get("transformer_activation", "gelu")),
        block_variant=str(model_cfg.get("block_variant", "hope_hybrid")),
    )
    return HOPEModel(hope_cfg)


def build_dataloader(
    data_cfg: DictConfig,
    *,
    distributed: bool,
    dist_ctx: DistributedContext | None,
    seed: int | None = None,
) -> Tuple[DataLoader, DistributedSampler | None]:
    dataset = _build_dataset(data_cfg)
    use_sampler = distributed and not isinstance(dataset, IterableDataset)
    if use_sampler:
        assert dist_ctx is not None
        sampler: DistributedSampler | None = DistributedSampler(
            dataset,
            num_replicas=dist_ctx.world_size,
            rank=dist_ctx.rank,
            shuffle=True,
            drop_last=False,
        )
        shuffle = False
    else:
        sampler = None
        shuffle = True
    if isinstance(dataset, IterableDataset):
        shuffle = False
    generator = None
    worker_init_fn = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
        worker_init_fn = _make_worker_init_fn(seed)
    dataloader = DataLoader(
        dataset,
        batch_size=data_cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        collate_fn=collate_batch,
        num_workers=data_cfg.get("num_workers", 0),
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        generator=generator,
    )
    return dataloader, sampler


def _build_dataset(data_cfg: DictConfig):
    source = data_cfg.source
    if source == "synthetic":
        synth_cfg = SyntheticTextConfig(
            vocab_size=data_cfg.vocab_size,
            seq_len=data_cfg.seq_len,
            dataset_size=data_cfg.dataset_size,
        )
        return SyntheticTextDataset(synth_cfg)
    if source == "shards":
        shard_dir = data_cfg.shards_dir
        return TokenShardDataset(shard_dir)
    if source == "mixture":
        mixture_cfg = data_cfg.mixture
        sources = [
            ShardSourceConfig(
                name=entry.name,
                shards_dir=entry.shards_dir,
                weight=entry.weight,
            )
            for entry in mixture_cfg.sources
        ]
        samples_per_epoch = mixture_cfg.samples_per_epoch
        seed = mixture_cfg.get("seed", 0)
        return MixtureShardDataset(
            sources,
            samples_per_epoch=samples_per_epoch,
            seed=seed,
        )
    msg = f"Unsupported data source {source}"
    raise ValueError(msg)


def compute_teach_signal(
    model: "_HasLMHead",
    logits: torch.Tensor,
    tokens: torch.Tensor,
    *,
    ignore_index: int | None = None,
) -> torch.Tensor:
    """
    Approximate dL/dh where h is the hidden state before the LM head.

    This matches the gradient of mean next-token CE:
      CE(logits[:, :-1], tokens[:, 1:]) with mean reduction.

    If ignore_index is provided, targets equal to ignore_index are masked out and
    the mean reduction denominator becomes the number of active targets (matching
    PyTorch CE semantics).
    """
    logits_detached = logits.detach()
    probs = torch.softmax(logits_detached, dim=-1)
    target_tokens = tokens[:, 1:]
    residual = probs[:, :-1].clone()

    if ignore_index is None:
        safe_targets = target_tokens
        src = -torch.ones(
            (*safe_targets.shape, 1),
            device=residual.device,
            dtype=residual.dtype,
        )
        denom: torch.Tensor | float = max(1, tokens.size(0) * max(1, tokens.size(1) - 1))
    else:
        active = target_tokens != ignore_index
        safe_targets = torch.where(active, target_tokens, torch.zeros_like(target_tokens))
        active_f = active.to(dtype=residual.dtype)
        residual.mul_(active_f.unsqueeze(-1))
        src = -active_f.unsqueeze(-1)
        denom = active_f.sum().clamp(min=1.0)

    residual.scatter_add_(-1, safe_targets.unsqueeze(-1), src)
    residual = residual / denom

    head_weight = model.lm_head.weight.detach()
    if head_weight.dtype != residual.dtype:
        head_weight = head_weight.to(dtype=residual.dtype)
    grad = residual @ head_weight
    pad = torch.zeros(
        grad.size(0),
        1,
        grad.size(-1),
        device=grad.device,
        dtype=grad.dtype,
    )
    return torch.cat([grad, pad], dim=1)


def _compute_layer_teach_signals(
    loss: torch.Tensor, block_outputs: list[torch.Tensor]
) -> list[torch.Tensor]:
    grads = torch.autograd.grad(
        loss,
        block_outputs,
        retain_graph=True,
        allow_unused=False,
    )
    return [g.detach() for g in grads]


def _compute_surprise_override(
    metric: str,
    *,
    logits: torch.Tensor,
    tokens: torch.Tensor,
    loss: torch.Tensor,
) -> float | None:
    normalized = str(metric).strip().lower()
    if normalized == "loss":
        return float(loss.detach().item())
    if normalized == "logit_entropy":
        logits_detached = logits[:, :-1].detach().float()
        probs = torch.softmax(logits_detached, dim=-1)
        entropy = -(probs * torch.log(probs.clamp(min=1e-9))).sum(dim=-1).mean()
        return float(entropy.item())
    return None


def _infer_online_chunk_size(model: HOPEModel) -> int | None:
    min_period: int | None = None
    blocks = getattr(model, "blocks", [])
    for block in blocks:
        cfg = getattr(block, "config", None)
        levels = getattr(cfg, "cms_levels", None)
        if not levels:
            continue
        for spec in levels:
            period = int(spec.update_period)
            if period <= 0:
                continue
            min_period = period if min_period is None else min(min_period, period)
    return min_period


class _HasLMHead(Protocol):
    lm_head: torch.nn.Linear


def _checksum_path(path: str | None) -> str | None:
    if not path:
        return None
    candidate = Path(path)
    if not candidate.exists() or not candidate.is_file():
        return None
    digest = sha256()
    with candidate.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def maybe_save_checkpoint(
    cfg: DictConfig,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    step: int,
    total_steps: int,
    distributed: bool,
    dist_ctx: DistributedContext | None,
    step_offset: int = 0,
) -> None:
    ckpt_cfg = cfg.train.get("checkpoint")
    if not ckpt_cfg or not ckpt_cfg.get("enable", False):
        return
    if distributed and dist_ctx is not None and dist_ctx.rank != 0:
        return
    save_interval = ckpt_cfg.get("save_interval", total_steps)
    save_last = ckpt_cfg.get("save_last", True)
    is_last_step = (step + 1) >= total_steps
    should_save = ((step + 1) % max(1, save_interval) == 0) or (save_last and is_last_step)
    if not should_save:
        return
    ckpt_dir = Path(ckpt_cfg.get("dir", "checkpoints/default"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    global_step = step + 1 + int(step_offset)
    ckpt_path = ckpt_dir / f"step_{global_step:06d}.pt"
    tmp_path = ckpt_path.with_suffix(".tmp")
    resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": step + 1,
        "config": resolved_cfg,
    }
    torch.save(state, tmp_path)
    os.replace(tmp_path, ckpt_path)
    write_checkpoint_metadata(cfg, ckpt_path, global_step)
    prefix = "[checkpoint]"
    if distributed and dist_ctx is not None:
        prefix = f"[checkpoint rank={dist_ctx.rank}]"
    print(f"{prefix} saved {ckpt_path} (global_step={global_step})")


def _validate_distributed_config(cfg: DictConfig, distributed: bool) -> None:
    if not distributed:
        return
    fail_if_faithful_disabled = bool(cfg.train.get("fail_if_paper_faithful_disabled", False))
    if not fail_if_faithful_disabled:
        return
    if bool(cfg.train.get("per_layer_teach_signal", False)):
        raise RuntimeError(
            "train.per_layer_teach_signal=true is not supported under DDP in this repo. "
            "Set train.fail_if_paper_faithful_disabled=false to allow the fallback, "
            "or run single-process training."
        )
    if bool(cfg.train.get("online_updates", False)):
        raise RuntimeError(
            "train.online_updates=true is not supported under DDP in this repo. "
            "Set train.fail_if_paper_faithful_disabled=false to allow the fallback, "
            "or run single-process training."
        )


def _validate_fast_state_batch_semantics(cfg: DictConfig) -> None:
    if not bool(cfg.train.get("use_fast_state", False)):
        return
    data_cfg = cfg.get("data")
    if data_cfg is None:
        return
    batch_size_raw = data_cfg.get("batch_size", 1)
    try:
        batch_size = int(batch_size_raw)
    except (TypeError, ValueError):
        return
    if batch_size <= 1:
        return
    msg = (
        "train.use_fast_state=true currently shares CMS/TITAN fast state across the batch. "
        "For strict per-context semantics, set data.batch_size=1."
    )
    if bool(cfg.train.get("fail_if_paper_faithful_disabled", False)):
        raise RuntimeError(msg)
    print(f"[train] {msg}")


def run_training_loop(
    cfg: DictConfig,
    *,
    device: torch.device,
    distributed: bool = False,
    dist_ctx: DistributedContext | None = None,
) -> Dict[str, float]:
    _validate_distributed_config(cfg, distributed)
    _validate_fast_state_batch_semantics(cfg)
    model = build_model_from_cfg(cfg.model).to(device)
    train_seed = cfg.train.get("seed")
    deterministic = cfg.train.get("deterministic", False)
    if train_seed is not None:
        _seed_everything(int(train_seed), deterministic=bool(deterministic))
    model = _maybe_compile_model(model, cfg.train.get("compile"))
    if distributed:
        assert dist_ctx is not None
        if device.type == "cuda":
            idx = device.index if device.index is not None else 0
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[idx],
                output_device=idx,
                find_unused_parameters=True,
            )
        else:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                find_unused_parameters=True,
            )
        base_model = model.module
    else:
        base_model = model

    seed_offset = 0
    if train_seed is not None and dist_ctx is not None:
        seed_offset = dist_ctx.rank
    dataloader_seed = None if train_seed is None else int(train_seed) + seed_offset
    dataloader, sampler = build_dataloader(
        cfg.data,
        distributed=distributed,
        dist_ctx=dist_ctx,
        seed=dataloader_seed,
    )
    optimizer = _build_optimizer(base_model, cfg, device=device)
    autocast_factory = _make_autocast_factory(device, cfg.train.get("mixed_precision"))
    logger = init_logger(getattr(cfg, "logging", None), cfg)
    if distributed and dist_ctx is not None and dist_ctx.rank != 0:
        logger = NullLogger()
    _log_run_features(logger, base_model, cfg, optimizer, device)
    steps = cfg.train.steps
    log_interval = cfg.train.get("log_interval", 1)
    per_layer_teach = bool(cfg.train.get("per_layer_teach_signal", False))
    online_updates = bool(cfg.train.get("online_updates", False))
    online_chunk_size = int(cfg.train.get("online_chunk_size", 0) or 0)
    use_fast_state = bool(cfg.train.get("use_fast_state", False))
    fail_if_faithful_disabled = bool(cfg.train.get("fail_if_paper_faithful_disabled", False))
    if distributed and per_layer_teach:
        msg = "[train] per_layer_teach_signal disabled under DDP (uses base model methods)"
        if fail_if_faithful_disabled:
            raise RuntimeError(
                f"{msg}. Set train.fail_if_paper_faithful_disabled=false to allow the fallback, "
                "or run single-process training."
            )
        print(msg)
        per_layer_teach = False
    if distributed and online_updates:
        msg = "[train] online_updates disabled under DDP (uses base model methods)"
        if fail_if_faithful_disabled:
            raise RuntimeError(
                f"{msg}. Set train.fail_if_paper_faithful_disabled=false to allow the fallback, "
                "or run single-process training."
            )
        print(msg)
        online_updates = False
    step_iter = iter(dataloader)
    epoch = 0
    metrics: Dict[str, float] = {}
    surprise_metric_getter = getattr(base_model, "get_surprise_metric", None)
    surprise_metric = (
        str(surprise_metric_getter()).strip().lower()
        if callable(surprise_metric_getter)
        else str(cfg.model.get("surprise_metric", "l2")).strip().lower()
    )
    for step in range(steps):
        if sampler is not None and step % len(dataloader) == 0:
            sampler.set_epoch(epoch)
            epoch += 1
        try:
            batch = next(step_iter)
        except StopIteration:
            step_iter = iter(dataloader)
            batch = next(step_iter)
        tokens = batch.to(device)
        fast_state = None
        if use_fast_state:
            init_fn = getattr(base_model, "init_fast_state", None)
            if not callable(init_fn):
                raise ValueError("train.use_fast_state=true requires model.init_fast_state()")
            fast_state = init_fn()
        _apply_teach_schedule(base_model, cfg, step)
        update_metrics: Dict[str, float] = {}
        if online_updates and hasattr(base_model, "forward_with_block_outputs"):
            total_loss = 0.0
            total_tokens = 0
            teach_signal_norm = 0.0
            optimizer.zero_grad()
            chunk_size = online_chunk_size
            if chunk_size <= 0:
                inferred = _infer_online_chunk_size(base_model)
                chunk_size = inferred if inferred is not None else tokens.size(1)
            if chunk_size < 2:
                # Next-token CE needs at least 2 tokens per chunk. Clamp to avoid a silent no-op
                # when configs infer update_period=1.
                print(f"[train] online_chunk_size={chunk_size} is too small; clamping to 2")
                chunk_size = 2
            for start in range(0, tokens.size(1), chunk_size):
                end = min(start + chunk_size, tokens.size(1))
                chunk_tokens = tokens[:, start:end]
                if chunk_tokens.size(1) <= 1:
                    continue
                with autocast_factory():
                    logits, _pre, block_outputs = (
                        base_model.forward_with_block_outputs(chunk_tokens, fast_state=fast_state)
                        if fast_state is not None
                        else base_model.forward_with_block_outputs(chunk_tokens)
                    )
                    loss = torch.nn.functional.cross_entropy(
                        logits[:, :-1].reshape(-1, logits.size(-1)),
                        chunk_tokens[:, 1:].reshape(-1),
                    )
                surprise_override = _compute_surprise_override(
                    surprise_metric,
                    logits=logits,
                    tokens=chunk_tokens,
                    loss=loss,
                )
                if per_layer_teach:
                    teach_signals = _compute_layer_teach_signals(loss, block_outputs)
                    teach_signal_norm += float(
                        torch.stack([sig.norm(dim=-1).mean() for sig in teach_signals]).mean()
                    ) * (chunk_tokens.size(1) - 1)
                else:
                    teach_signal = compute_teach_signal(base_model, logits, chunk_tokens)
                    teach_signal_norm += (
                        teach_signal.norm(dim=-1).mean().item() * (chunk_tokens.size(1) - 1)
                    )
                loss.backward()
                with torch.no_grad():
                    if per_layer_teach:
                        base_model(
                            chunk_tokens,
                            teach_signals=teach_signals,
                            surprise_value=surprise_override,
                            fast_state=fast_state,
                        )
                    else:
                        base_model(
                            chunk_tokens,
                            teach_signal=teach_signal,
                            surprise_value=surprise_override,
                            fast_state=fast_state,
                        )
                    if hasattr(base_model, "pop_update_metrics"):
                        update_metrics = base_model.pop_update_metrics()
                total_loss += loss.item() * (chunk_tokens.size(1) - 1)
                total_tokens += chunk_tokens.size(1) - 1
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), max_norm=1.0)
            optimizer.step()
            loss = torch.tensor(total_loss / max(total_tokens, 1), device=device)
            teach_signal_norm = teach_signal_norm / max(total_tokens, 1)
        else:
            with autocast_factory():
                if per_layer_teach and hasattr(base_model, "forward_with_block_outputs"):
                    logits, _pre, block_outputs = (
                        base_model.forward_with_block_outputs(tokens, fast_state=fast_state)
                        if fast_state is not None
                        else base_model.forward_with_block_outputs(tokens)
                    )
                    loss = torch.nn.functional.cross_entropy(
                        logits[:, :-1].reshape(-1, logits.size(-1)),
                        tokens[:, 1:].reshape(-1),
                    )
                else:
                    if fast_state is not None:
                        logits = model(tokens, fast_state=fast_state)
                    else:
                        logits = model(tokens)
                    loss = torch.nn.functional.cross_entropy(
                        logits[:, :-1].reshape(-1, logits.size(-1)),
                        tokens[:, 1:].reshape(-1),
                    )
            surprise_override = _compute_surprise_override(
                surprise_metric,
                logits=logits,
                tokens=tokens,
                loss=loss,
            )
            optimizer.zero_grad()
            if per_layer_teach and hasattr(base_model, "forward_with_block_outputs"):
                teach_signals = _compute_layer_teach_signals(loss, block_outputs)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(base_model.parameters(), max_norm=1.0)
            optimizer.step()
            with torch.no_grad():
                if per_layer_teach and hasattr(base_model, "forward_with_block_outputs"):
                    teach_signal_norm = float(
                        torch.stack([sig.norm(dim=-1).mean() for sig in teach_signals]).mean()
                    )
                    base_model(
                        tokens,
                        teach_signals=teach_signals,
                        surprise_value=surprise_override,
                        fast_state=fast_state,
                    )
                else:
                    teach_signal = compute_teach_signal(base_model, logits, tokens)
                    teach_signal_norm = teach_signal.norm(dim=-1).mean().item()
                    base_model(
                        tokens,
                        teach_signal=teach_signal,
                        surprise_value=surprise_override,
                        fast_state=fast_state,
                    )
                if hasattr(base_model, "pop_update_metrics"):
                    update_metrics = base_model.pop_update_metrics()
        if step % log_interval == 0:
            ppl = torch.exp(loss.detach()).item()
            metrics_payload = {
                "loss": loss.item(),
                "ppl": ppl,
                "teach_signal_norm": teach_signal_norm,
            }
            metrics_payload.update(update_metrics)
            logger.log(metrics_payload, step=step)
            if (not distributed) or (dist_ctx and dist_ctx.rank == 0):
                print(
                    f"[train] step={step} loss={loss.item():.4f} "
                    f"ppl={ppl:.2f} teach_norm={teach_signal_norm:.4f}"
                )
            metrics = metrics_payload
        maybe_save_checkpoint(
            cfg,
            base_model,
            optimizer,
            step=step,
            total_steps=steps,
            distributed=distributed,
            dist_ctx=dist_ctx,
            step_offset=int(cfg.train.get("step_offset", 0) or 0),
        )
    logger.finish()
    return metrics


def _apply_teach_schedule(model: HOPEModel, cfg: DictConfig, step: int) -> None:
    schedule = cfg.model.get("teach_schedule")
    base_scale = cfg.model.get("teach_scale", 1.0)
    scale = base_scale
    if schedule:
        warmup = schedule.get("warmup_steps", 0)
        if warmup and warmup > 0:
            scale *= min(1.0, (step + 1) / warmup)
        decay_start = schedule.get("decay_start")
        decay_duration = schedule.get("decay_duration")
        if (
            decay_start is not None
            and decay_duration
            and decay_duration > 0
            and (step + 1) > decay_start
        ):
            progress = min(1.0, (step + 1 - decay_start) / decay_duration)
            scale *= max(0.0, 1.0 - progress)
    model.set_teach_runtime(scale=scale)


def _maybe_compile_model(model: torch.nn.Module, compile_cfg: dict | None) -> torch.nn.Module:
    if not compile_cfg or not compile_cfg.get("enable", False):
        return model
    kwargs = {}
    if "mode" in compile_cfg:
        kwargs["mode"] = compile_cfg["mode"]
    if "backend" in compile_cfg:
        kwargs["backend"] = compile_cfg["backend"]
    try:
        return cast(torch.nn.Module, torch.compile(model, **kwargs))  # type: ignore[attr-defined]
    except Exception as err:  # pragma: no cover - compile is optional
        if compile_cfg.get("strict", False):
            raise
        print(f"[compile] fallback to eager due to: {err}")
        return model


def _make_autocast_factory(device: torch.device, mp_cfg: dict | None):
    if not mp_cfg or not mp_cfg.get("enabled", False):
        return lambda: nullcontext()
    dtype = _resolve_autocast_dtype(mp_cfg.get("dtype", "bf16"))
    device_type = device.type
    if device_type not in {"cuda", "cpu", "mps"}:
        device_type = "cpu"

    def factory():
        try:
            return torch.autocast(device_type=device_type, dtype=dtype)
        except Exception as err:  # pragma: no cover - device/dtype support varies by backend
            print(f"[autocast] disabled for device_type={device_type} dtype={dtype}: {err}")
            return nullcontext()

    return factory


def _resolve_autocast_dtype(name: str) -> torch.dtype:
    normalized = str(name).lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16", "half"}:
        return torch.float16
    msg = f"Unsupported autocast dtype {name}"
    raise ValueError(msg)


def _build_optimizer(
    model: torch.nn.Module, cfg: DictConfig, *, device: torch.device
) -> torch.optim.Optimizer:
    optimizer_cfg_raw = cfg.get("optim")
    if isinstance(optimizer_cfg_raw, DictConfig):
        optimizer_cfg = optimizer_cfg_raw
    else:
        optimizer_cfg = cast(DictConfig, OmegaConf.create(optimizer_cfg_raw or {}))
    param_policy_raw = optimizer_cfg.get("param_policy")
    if param_policy_raw is None:
        outer_updates_memory_modules = optimizer_cfg.get("outer_updates_memory_modules")
        if outer_updates_memory_modules is None:
            param_policy = "all"
        else:
            param_policy = "all" if bool(outer_updates_memory_modules) else "exclude_memory"
    else:
        param_policy = str(param_policy_raw).strip().lower()
    named_params = _select_outer_named_parameters(model, param_policy)
    if not named_params:
        raise ValueError(
            f"No trainable parameters selected for optim.param_policy={param_policy!r}. "
            "Check freeze_backbone, requires_grad flags, or adjust the policy."
        )
    optim_type = str(optimizer_cfg.get("type", "adamw")).lower()
    if optim_type == "muon":
        return _build_muon_optimizer(
            model,
            optimizer_cfg,
            device=device,
            named_params=named_params,
            param_policy=param_policy,
        )
    if optim_type == "m3":
        return _build_m3_optimizer(
            model,
            optimizer_cfg,
            device=device,
            named_params=named_params,
            param_policy=param_policy,
        )
    lr = optimizer_cfg.get("lr", 1e-3)
    betas = optimizer_cfg.get("betas", (0.9, 0.999))
    weight_decay = optimizer_cfg.get("weight_decay", 0.0)
    fused_cfg = optimizer_cfg.get("fused", "auto")
    fused = False
    if fused_cfg == "auto":
        fused = device.type == "cuda" and torch.cuda.is_available()
    else:
        fused = bool(fused_cfg)
    kwargs = {"lr": lr, "betas": betas, "weight_decay": weight_decay}
    if fused:
        kwargs["fused"] = True
    params = [param for _, param in named_params]
    return torch.optim.AdamW(params, **kwargs)


def _build_muon_optimizer(
    model: torch.nn.Module,
    optimizer_cfg: DictConfig,
    *,
    device: torch.device,
    named_params: list[tuple[str, torch.nn.Parameter]] | None = None,
    param_policy: str | None = None,
):
    if not hasattr(torch.optim, "Muon"):
        raise RuntimeError("torch.optim.Muon is not available in this PyTorch build")
    lr = optimizer_cfg.get("lr", 1e-3)
    weight_decay = optimizer_cfg.get("weight_decay", 0.01)
    momentum = optimizer_cfg.get("momentum", 0.95)
    ns_coefficients = optimizer_cfg.get("ns_coefficients")
    ns_steps = optimizer_cfg.get("ns_steps")
    eps = optimizer_cfg.get("eps", 1e-7)
    fused_cfg = optimizer_cfg.get("fused", "auto")
    fused = False
    if fused_cfg == "auto":
        fused = device.type == "cuda" and torch.cuda.is_available()
    else:
        fused = bool(fused_cfg)
    muon_params: list[torch.nn.Parameter] = []
    adamw_params: list[torch.nn.Parameter] = []
    source = named_params if named_params is not None else model.named_parameters()
    for name, param in source:
        if not param.requires_grad:
            continue
        if _is_muon_candidate(name, param):
            muon_params.append(param)
        else:
            adamw_params.append(param)
    muon_kwargs = {
        "lr": lr,
        "weight_decay": weight_decay,
        "momentum": momentum,
        "eps": eps,
    }
    if ns_coefficients is not None:
        muon_kwargs["ns_coefficients"] = tuple(ns_coefficients)
    if ns_steps is not None:
        muon_kwargs["ns_steps"] = int(ns_steps)
    muon_opt = torch.optim.Muon(muon_params, **muon_kwargs) if muon_params else None  # type: ignore[attr-defined]
    adamw_kwargs = {
        "lr": lr,
        "betas": optimizer_cfg.get("betas", (0.9, 0.999)),
        "weight_decay": weight_decay,
    }
    if fused:
        adamw_kwargs["fused"] = True
    adamw_opt = torch.optim.AdamW(adamw_params, **adamw_kwargs) if adamw_params else None
    muon_elems = int(sum(p.numel() for p in muon_params))
    adamw_elems = int(sum(p.numel() for p in adamw_params))
    return _HybridOptimizer(
        muon_opt,
        adamw_opt,
        muon_elems,
        adamw_elems,
        primary_name="muon",
        param_policy=param_policy,
    )


def _build_m3_optimizer(
    model: torch.nn.Module,
    optimizer_cfg: DictConfig,
    *,
    device: torch.device,
    named_params: list[tuple[str, torch.nn.Parameter]] | None = None,
    param_policy: str | None = None,
):
    lr = optimizer_cfg.get("lr", 1e-3)
    weight_decay = optimizer_cfg.get("weight_decay", 0.01)
    beta1 = optimizer_cfg.get("beta1", 0.9)
    beta2 = optimizer_cfg.get("beta2", 0.999)
    beta3 = optimizer_cfg.get("beta3", 0.9)
    alpha = optimizer_cfg.get("alpha", 1.0)
    ns_steps = int(optimizer_cfg.get("ns_steps", 3))
    slow_chunk = int(optimizer_cfg.get("slow_chunk", 100))
    eps = optimizer_cfg.get("eps", 1e-8)
    fused_cfg = optimizer_cfg.get("fused", "auto")
    fused = False
    if fused_cfg == "auto":
        fused = device.type == "cuda" and torch.cuda.is_available()
    else:
        fused = bool(fused_cfg)

    m3_params: list[torch.nn.Parameter] = []
    adamw_params: list[torch.nn.Parameter] = []
    source = named_params if named_params is not None else model.named_parameters()
    for name, param in source:
        if not param.requires_grad:
            continue
        if _is_muon_candidate(name, param):
            m3_params.append(param)
        else:
            adamw_params.append(param)
    m3_opt = (
        M3(
            m3_params,
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            beta3=beta3,
            alpha=alpha,
            eps=eps,
            ns_steps=ns_steps,
            slow_chunk=slow_chunk,
            weight_decay=weight_decay,
        )
        if m3_params
        else None
    )
    adamw_kwargs = {
        "lr": lr,
        "betas": optimizer_cfg.get("betas", (0.9, 0.999)),
        "weight_decay": weight_decay,
    }
    if fused:
        adamw_kwargs["fused"] = True
    adamw_opt = torch.optim.AdamW(adamw_params, **adamw_kwargs) if adamw_params else None
    m3_elems = int(sum(p.numel() for p in m3_params))
    adamw_elems = int(sum(p.numel() for p in adamw_params))
    return _HybridOptimizer(
        m3_opt,
        adamw_opt,
        m3_elems,
        adamw_elems,
        primary_name="m3",
        param_policy=param_policy,
    )


def _select_outer_named_parameters(
    model: torch.nn.Module, param_policy: str
) -> list[tuple[str, torch.nn.Parameter]]:
    policy = str(param_policy).strip().lower()
    trainable: list[tuple[str, torch.nn.Parameter]] = [
        (name, param) for name, param in model.named_parameters() if param.requires_grad
    ]
    if policy in {"all", "full"}:
        return trainable
    if policy in {"exclude_memory", "no_memory"}:
        return [(name, param) for name, param in trainable if not _is_memory_param_name(name)]
    if policy in {"only_memory", "memory_only"}:
        return [(name, param) for name, param in trainable if _is_memory_param_name(name)]
    raise ValueError(
        f"Unsupported optim.param_policy={param_policy!r}. "
        "Expected one of ['all', 'exclude_memory', 'only_memory']."
    )


def _is_memory_param_name(name: str) -> bool:
    lowered = name.lower()
    return any(token in lowered for token in (".cms.", ".titan_memory.", ".selfmod."))


def _is_muon_candidate(name: str, param: torch.nn.Parameter) -> bool:
    if param.ndim < 2:
        return False
    lowered = name.lower()
    if "norm" in lowered or "embed" in lowered:
        return False
    return True


class _HybridOptimizer:
    def __init__(
        self,
        primary_opt: torch.optim.Optimizer | None,
        secondary_opt: torch.optim.Optimizer | None,
        primary_param_elems: int,
        secondary_param_elems: int,
        *,
        primary_name: str = "muon",
        param_policy: str | None = None,
    ):
        self.primary_opt = primary_opt
        self.secondary_opt = secondary_opt
        self.primary_param_elems = primary_param_elems
        self.secondary_param_elems = secondary_param_elems
        self.primary_name = primary_name
        self.param_policy = param_policy

    def zero_grad(self) -> None:
        if self.primary_opt:
            self.primary_opt.zero_grad()
        if self.secondary_opt:
            self.secondary_opt.zero_grad()

    def step(self) -> None:
        if self.primary_opt:
            self.primary_opt.step()
        if self.secondary_opt:
            self.secondary_opt.step()

    def state_dict(self) -> dict:
        return {
            self.primary_name: self.primary_opt.state_dict() if self.primary_opt else None,
            "adamw": self.secondary_opt.state_dict() if self.secondary_opt else None,
        }

    def load_state_dict(self, state: dict) -> None:
        if self.primary_opt and state.get(self.primary_name) is not None:
            self.primary_opt.load_state_dict(state[self.primary_name])
        if self.secondary_opt and state.get("adamw") is not None:
            self.secondary_opt.load_state_dict(state["adamw"])

    @property
    def param_groups(self):
        groups = []
        if self.primary_opt:
            groups.extend(self.primary_opt.param_groups)
        if self.secondary_opt:
            groups.extend(self.secondary_opt.param_groups)
        return groups

    def get_param_split(self) -> dict[str, int]:
        return {
            self.primary_name: self.primary_param_elems,
            "adamw": self.secondary_param_elems,
        }


def _log_run_features(
    logger: BaseLogger,
    model: torch.nn.Module,
    cfg: DictConfig,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> None:
    mp_cfg = cfg.train.get("mixed_precision", {})
    compile_cfg = cfg.train.get("compile", {})
    features: dict[str, object] = {
        "train.mixed_precision_enabled": bool(mp_cfg.get("enabled", False)),
        "train.mixed_precision_dtype": str(mp_cfg.get("dtype", "bf16")),
        "train.compile_enabled": bool(compile_cfg.get("enable", False)),
        "train.compile_mode": str(compile_cfg.get("mode", "default")) if compile_cfg else "default",
        "attention.flash_enabled": _detect_flash_attention(model),
        "device": device.type,
    }
    optimizer_cfg_raw = cfg.get("optim")
    if isinstance(optimizer_cfg_raw, DictConfig):
        optimizer_cfg = optimizer_cfg_raw
    else:
        optimizer_cfg = cast(DictConfig, OmegaConf.create(optimizer_cfg_raw or {}))
    param_policy_raw = optimizer_cfg.get("param_policy")
    if param_policy_raw is None:
        outer_updates_memory_modules = optimizer_cfg.get("outer_updates_memory_modules")
        if outer_updates_memory_modules is None:
            param_policy = "all"
        else:
            param_policy = "all" if bool(outer_updates_memory_modules) else "exclude_memory"
    else:
        param_policy = str(param_policy_raw).strip().lower()
    try:
        selected = _select_outer_named_parameters(model, param_policy)
        total_elems = int(sum(param.numel() for _, param in selected))
        memory_elems = int(
            sum(param.numel() for name, param in selected if _is_memory_param_name(name))
        )
        features["optim.param_policy"] = param_policy
        features["optim.param_policy_param_elems"] = total_elems
        features["optim.param_policy_memory_param_elems"] = memory_elems
        features["optim.param_policy_non_memory_param_elems"] = total_elems - memory_elems
    except Exception as err:  # pragma: no cover - purely diagnostic
        features["optim.param_policy"] = param_policy
        features["optim.param_policy_error"] = str(err)
    split_fn = getattr(optimizer, "get_param_split", None)
    if callable(split_fn):
        split = split_fn()
        for key, value in split.items():
            features[f"optim.{key}_param_elems"] = int(value)
    logger.log(features, step=-1)
    print(f"[train] run_features {features}")


def _detect_flash_attention(model: torch.nn.Module) -> bool:
    blocks = getattr(model, "blocks", [])
    for block in blocks:
        attn = getattr(block, "attn", None)
        config = getattr(attn, "config", None)
        if config is not None and hasattr(config, "use_flash"):
            return bool(config.use_flash)
    return False


def write_checkpoint_metadata(cfg: DictConfig, ckpt_path: Path, step: int) -> None:
    config_yaml = OmegaConf.to_yaml(cfg)
    config_path = ckpt_path.with_suffix(".yaml")
    config_path.write_text(config_yaml)
    config_hash = sha256(config_yaml.encode("utf-8")).hexdigest()
    ckpt_hash = _checksum_path(str(ckpt_path))
    sha_path = ckpt_path.with_suffix(".sha256")
    if ckpt_hash:
        sha_path.write_text(f"{ckpt_hash}  {ckpt_path.name}\n")
    tokenizer_path = cfg.data.get("tokenizer_path") if hasattr(cfg, "data") else None
    metadata = {
        "step": step,
        "checkpoint_sha256": ckpt_hash,
        "config_sha256": config_hash,
        "tokenizer_hash": _checksum_path(tokenizer_path) if tokenizer_path else None,
        "config_path": str(config_path),
        "rng_states": _capture_rng_states(),
    }
    ckpt_path.with_suffix(".meta.json").write_text(json.dumps(metadata, indent=2))


def verify_checkpoint_integrity(ckpt_path: Path) -> Dict[str, object]:
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint {ckpt_path} not found")
    meta_path = ckpt_path.with_suffix(".meta.json")
    if not meta_path.exists():
        raise FileNotFoundError(f"Metadata file {meta_path} missing")
    metadata = json.loads(meta_path.read_text())
    computed_sha = _checksum_path(str(ckpt_path))
    recorded_sha = metadata.get("checkpoint_sha256")
    if recorded_sha and computed_sha and recorded_sha != computed_sha:
        raise ValueError(
            f"Checkpoint SHA mismatch: recorded {recorded_sha} vs computed {computed_sha}"
        )
    sha_file = ckpt_path.with_suffix(".sha256")
    if sha_file.exists() and computed_sha:
        recorded_line = sha_file.read_text().strip().split()
        if recorded_line:
            recorded = recorded_line[0]
            if recorded != computed_sha:
                raise ValueError(f".sha256 mismatch: {recorded} vs {computed_sha}")
    config_path = ckpt_path.with_suffix(".yaml")
    if not config_path.exists():
        raise FileNotFoundError(f"Config file {config_path} missing")
    config_hash = sha256(config_path.read_text().encode("utf-8")).hexdigest()
    recorded_cfg_hash = metadata.get("config_sha256")
    if recorded_cfg_hash and recorded_cfg_hash != config_hash:
        raise ValueError(
            f"Config SHA mismatch: recorded {recorded_cfg_hash} vs computed {config_hash}"
        )
    if "rng_states" not in metadata:
        raise ValueError("Metadata missing rng_states")
    return metadata


def _capture_rng_states() -> Dict[str, object]:
    payload: Dict[str, object] = {
        "python": _encode_pickle(random.getstate()),
        "numpy": _encode_pickle(np.random.get_state()),
        "torch": _tensor_state_to_hex(torch.random.get_rng_state()),
    }
    if torch.cuda.is_available():
        payload["torch_cuda"] = [
            _tensor_state_to_hex(state) for state in torch.cuda.get_rng_state_all()
        ]  # type: ignore[attr-defined]
    return payload


def _encode_pickle(obj: object) -> str:
    return base64.b64encode(pickle.dumps(obj)).decode("ascii")


def _tensor_state_to_hex(state: torch.Tensor) -> str:
    return state.cpu().numpy().tobytes().hex()


def _seed_everything(seed: int, *, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
            torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
    else:
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = True  # type: ignore[attr-defined]
            torch.backends.cudnn.deterministic = False  # type: ignore[attr-defined]


def _make_worker_init_fn(base_seed: int):
    def _init_fn(worker_id: int) -> None:
        worker_seed = base_seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return _init_fn
```

### File: `src/nested_learning/transformer.py`

```python
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .backbones import AttentionConfig, SelfAttention


@dataclass
class TransformerBlockConfig:
    dim: int
    heads: int
    mlp_hidden_multiplier: int = 4
    activation: str = "gelu"
    qk_l2_norm: bool = False
    local_conv_window: int | None = None


class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        *,
        hidden_multiplier: int = 4,
        activation: str = "gelu",
    ) -> None:
        super().__init__()
        hidden = dim * hidden_multiplier
        if activation == "relu":
            act: nn.Module = nn.ReLU()
        elif activation == "silu":
            act = nn.SiLU()
        else:
            act = nn.GELU()
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden, bias=False),
            act,
            nn.Linear(hidden, dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        residual = x
        x = self.norm(x)
        return residual + self.net(x)


class TransformerBlock(nn.Module):
    """
    Baseline Transformer block: Attention -> MLP (no TITAN/CMS learning updates).

    This is used for Phase 2 comparisons (HOPE-Attention vs standard Transformer).
    """

    def __init__(self, config: TransformerBlockConfig) -> None:
        super().__init__()
        self.config = config
        self.attn = SelfAttention(
            AttentionConfig(
                dim=config.dim,
                heads=config.heads,
                qk_l2_norm=config.qk_l2_norm,
                local_conv_window=config.local_conv_window,
            )
        )
        self.mlp = FeedForward(
            config.dim,
            hidden_multiplier=config.mlp_hidden_multiplier,
            activation=config.activation,
        )

    def forward(
        self,
        x: torch.Tensor,
        *,
        teach_signal: torch.Tensor | None = None,
        surprise_value: float | None = None,
        fast_state=None,
    ) -> torch.Tensor:
        _ = (teach_signal, surprise_value, fast_state)
        return self.mlp(self.attn(x))

    def set_surprise_threshold(self, threshold: float | None) -> None:
        _ = threshold

    def set_surprise_metric(self, metric: str) -> None:
        _ = metric

    def set_allowed_levels(self, allowed) -> None:
        _ = allowed
```

### File: `tests/conftest.py`

```python
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
```

### File: `tests/test_attention_features.py`

```python
import torch

from nested_learning.backbones import AttentionConfig, SelfAttention


def test_self_attention_qk_l2_norm_unit_vectors() -> None:
    attn = SelfAttention(AttentionConfig(dim=16, heads=4, qk_l2_norm=True, use_flash=False))
    x = torch.randn(2, 5, 16)
    q, k, _v = attn._compute_qkv(x)
    q_norm = q.norm(dim=-1)
    k_norm = k.norm(dim=-1)
    assert torch.allclose(q_norm, torch.ones_like(q_norm), atol=1e-4, rtol=1e-4)
    assert torch.allclose(k_norm, torch.ones_like(k_norm), atol=1e-4, rtol=1e-4)


def test_self_attention_local_conv_window_preserves_shape() -> None:
    attn = SelfAttention(AttentionConfig(dim=16, heads=4, local_conv_window=4, use_flash=False))
    assert attn.local_conv is not None
    assert attn.local_conv.kernel_size == (4,)
    x = torch.randn(2, 8, 16)
    out = attn(x)
    assert out.shape == x.shape


def test_self_attention_local_conv_is_causal() -> None:
    torch.manual_seed(0)
    dim = 4
    attn = SelfAttention(
        AttentionConfig(dim=dim, heads=2, local_conv_window=4, use_flash=False, dropout=0.0)
    ).eval()
    assert attn.local_conv is not None
    with torch.no_grad():
        attn.local_conv.weight.fill_(1.0)
        eye = torch.eye(dim)
        attn.qkv.weight.zero_()
        attn.qkv.weight[:dim].copy_(eye)
        attn.qkv.weight[dim : 2 * dim].copy_(eye)
        attn.qkv.weight[2 * dim :].copy_(eye)
        attn.out_proj.weight.copy_(eye)
    x1 = torch.randn(1, 8, dim)
    x2 = x1.clone()
    x2[:, 4:, :] = torch.randn_like(x2[:, 4:, :])
    out1 = attn(x1)
    out2 = attn(x2)
    assert torch.allclose(out1[:, :4, :], out2[:, :4, :], atol=1e-5, rtol=1e-5)
```

### File: `tests/test_build_model_from_cfg_selfmod.py`

```python
from omegaconf import OmegaConf

from nested_learning.hope.block import HOPESelfModBlock
from nested_learning.training import build_model_from_cfg


def test_build_model_from_cfg_plumbs_selfmod_fields() -> None:
    model_cfg = OmegaConf.create(
        {
            "type": "hope",
            "vocab_size": 32,
            "dim": 16,
            "num_layers": 1,
            "heads": 4,
            "titan_level": {"name": "titan", "update_period": 1, "optimizer_key": "titan_opt"},
            "cms_levels": [{"name": "cms_fast", "update_period": 1, "optimizer_key": "cms_opt"}],
            "block_variant": "hope_selfmod",
            "self_mod_chunk_size": 3,
            "self_mod_chunk_size_memory": 7,
            "self_mod_objective": "dot",
            "self_mod_stopgrad_vhat": False,
            "self_mod_use_rank1_precond": False,
            "self_mod_use_alpha": False,
            "self_mod_momentum": 0.5,
        }
    )
    model = build_model_from_cfg(model_cfg)
    assert model.config.self_mod_chunk_size == 3
    assert model.config.self_mod_chunk_size_memory == 7
    assert model.config.self_mod_objective == "dot"
    assert model.config.self_mod_stopgrad_vhat is False
    assert model.config.self_mod_use_rank1_precond is False
    assert model.config.self_mod_use_alpha is False
    assert abs(model.config.self_mod_momentum - 0.5) < 1e-9

    block = model.blocks[0]
    assert isinstance(block, HOPESelfModBlock)
    assert block.selfmod.config.chunk_size_other == 3
    assert block.selfmod.config.chunk_size_memory == 7
    assert block.selfmod.config.objective == "dot"
    assert block.selfmod.config.stopgrad_vhat is False
    assert block.selfmod.config.use_rank1_precond is False
    assert block.selfmod.config.use_alpha is False
    assert abs(block.selfmod.config.momentum - 0.5) < 1e-9
```

### File: `tests/test_cms.py`

```python
import torch

from nested_learning.cms import CMS
from nested_learning.hope.block import HOPEAttentionBlock, HOPEAttentionBlockConfig
from nested_learning.levels import LevelSpec


def test_cms_forward_preserves_shape() -> None:
    cms = CMS(
        dim=16,
        levels=[LevelSpec(name="fast", update_period=2), LevelSpec(name="slow", update_period=4)],
    )
    x = torch.randn(2, 9, 16)
    out, inputs, outputs = cms(x, return_intermediates=True)
    assert out.shape == x.shape
    assert set(inputs.keys()) == {"fast", "slow"}
    assert set(outputs.keys()) == {"fast", "slow"}


def test_cms_can_disable_layernorm() -> None:
    cms = CMS(
        dim=16,
        levels=[LevelSpec(name="fast", update_period=2)],
        use_layernorm=False,
    )
    assert not any("net.0" in name for name, _ in cms.named_parameters())
    x = torch.randn(2, 9, 16)
    out = cms(x)
    assert isinstance(out, torch.Tensor)
    assert out.shape == x.shape


def test_cms_updates_respect_update_period_tokens() -> None:
    cfg = HOPEAttentionBlockConfig(
        dim=16,
        heads=4,
        cms_levels=[
            LevelSpec(name="fast", update_period=2),
            LevelSpec(name="slow", update_period=4),
        ],
        optimizer_configs={},
    )
    block = HOPEAttentionBlock(cfg)
    x = torch.randn(1, 9, 16)
    teach = torch.randn_like(x)
    _ = block(x, teach_signal=teach)
    stats = block.pop_update_stats()
    assert stats["cms.fast"]["gate_hit"] == 4.0
    assert stats["cms.fast"]["chunk_tokens"] == 8.0
    assert stats["cms.slow"]["gate_hit"] == 2.0
    assert stats["cms.slow"]["chunk_tokens"] == 8.0


def test_cms_updates_skip_when_no_signal() -> None:
    cfg = HOPEAttentionBlockConfig(
        dim=16,
        heads=4,
        cms_levels=[LevelSpec(name="fast", update_period=2)],
        optimizer_configs={},
    )
    block = HOPEAttentionBlock(cfg)
    x = torch.randn(1, 8, 16)
    teach = torch.zeros_like(x)
    _ = block(x, teach_signal=teach)
    stats = block.pop_update_stats()
    assert stats == {}


def test_cms_online_updates_affect_later_tokens() -> None:
    torch.manual_seed(0)
    cfg_online = HOPEAttentionBlockConfig(
        dim=16,
        heads=4,
        cms_levels=[LevelSpec(name="fast", update_period=2)],
        optimizer_configs={},
        cms_online_updates=True,
    )
    cfg_offline = HOPEAttentionBlockConfig(
        dim=16,
        heads=4,
        cms_levels=[LevelSpec(name="fast", update_period=2)],
        optimizer_configs={},
        cms_online_updates=False,
    )
    block_online = HOPEAttentionBlock(cfg_online)
    block_offline = HOPEAttentionBlock(cfg_offline)
    x = torch.randn(1, 6, 16)
    teach = torch.randn_like(x)
    out_online = block_online(x, teach_signal=teach)
    out_offline = block_offline(x, teach_signal=teach)
    assert not torch.allclose(out_online[:, 2:], out_offline[:, 2:])
```

### File: `tests/test_cms_delta_rule.py`

```python
import torch

from nested_learning.hope.block import _chunk_loss


def test_cms_target_shift_loss_grad_is_proportional_to_delta() -> None:
    torch.manual_seed(0)
    prediction = torch.randn(2, 5, 7, requires_grad=True)
    delta = torch.randn(2, 5, 7)
    active = torch.tensor(
        [
            [1, 1, 0, 1, 1],
            [0, 1, 1, 1, 0],
        ],
        dtype=torch.float32,
    )
    mask_f = active.unsqueeze(-1)
    loss = _chunk_loss(prediction, delta, mask_f, reduction="sum")
    loss.backward()
    assert prediction.grad is not None
    expected = 2.0 * delta * mask_f
    assert torch.allclose(prediction.grad, expected, atol=1e-6, rtol=1e-6)


def test_cms_chunk_loss_sum_scales_relative_to_mean() -> None:
    torch.manual_seed(0)
    prediction = torch.randn(1, 4, 5, requires_grad=True)
    delta = torch.randn(1, 4, 5)
    mask_f = torch.ones(1, 4, 1)

    loss_sum = _chunk_loss(prediction, delta, mask_f, reduction="sum")
    loss_sum.backward()
    assert prediction.grad is not None
    grad_sum = prediction.grad.detach().clone()

    prediction.grad.zero_()
    loss_mean = _chunk_loss(prediction, delta, mask_f, reduction="mean")
    loss_mean.backward()
    assert prediction.grad is not None
    grad_mean = prediction.grad.detach().clone()

    scale = float(mask_f.sum().item())
    assert torch.allclose(grad_sum, grad_mean * scale, atol=1e-6, rtol=1e-6)
```

### File: `tests/test_cms_flush_partial.py`

```python
import torch

from nested_learning.fast_state import build_block_fast_state
from nested_learning.hope.block import HOPEAttentionBlock, HOPEAttentionBlockConfig
from nested_learning.levels import LevelSpec


def _run_block(*, flush_partial: bool, use_fast_state: bool) -> dict[str, float]:
    torch.manual_seed(0)
    cfg = HOPEAttentionBlockConfig(
        dim=8,
        heads=1,
        cms_levels=(LevelSpec(name="fast", update_period=4),),
        cms_flush_partial_at_end=flush_partial,
        cms_online_updates=True,
        cms_chunk_reduction="sum",
    )
    block = HOPEAttentionBlock(cfg)
    x = torch.randn(1, 6, 8)
    teach = torch.ones(1, 6, 8)
    fast_state = None
    if use_fast_state:
        fast_state = build_block_fast_state(
            titan_module=None,
            cms_blocks=dict(block.cms.blocks.items()),
            specs=cfg.cms_levels,
            optimizer_configs=cfg.optimizer_configs,
            default_lr=cfg.self_mod_lr,
        )
    _out = block(x, teach_signal=teach, fast_state=fast_state)
    stats = block.pop_update_stats()
    return stats["cms.fast"]


def test_cms_flush_partial_disabled_leaves_remainder_unupdated() -> None:
    for use_fast_state in (False, True):
        payload = _run_block(flush_partial=False, use_fast_state=use_fast_state)
        assert payload["gate_hit"] == 1.0
        assert payload["chunk_tokens"] == 4.0


def test_cms_flush_partial_enabled_updates_final_remainder() -> None:
    for use_fast_state in (False, True):
        payload = _run_block(flush_partial=True, use_fast_state=use_fast_state)
        assert payload["gate_hit"] == 2.0
        assert payload["chunk_tokens"] == 6.0

```

### File: `tests/test_compare_variants_cli.py`

```python
import json
import subprocess
import sys
from pathlib import Path

import sentencepiece as spm
import torch
from omegaconf import OmegaConf

from nested_learning.training import build_model_from_cfg


def _train_tiny_sentencepiece(tmp_path: Path, *, vocab_size: int) -> Path:
    corpus_path = tmp_path / "corpus.txt"
    corpus_path.write_text(
        "\n".join(
            [
                "This is a tiny corpus for sentencepiece.",
                "Remember that the secret key is KEY-1234.",
                "Question: What is the passkey? Answer: PASSKEY-1234.",
            ]
        )
    )
    model_prefix = tmp_path / "spm_test"
    spm.SentencePieceTrainer.Train(
        input=str(corpus_path),
        model_prefix=str(model_prefix),
        vocab_size=vocab_size,
        hard_vocab_limit=False,
        model_type="unigram",
        bos_id=1,
        eos_id=2,
        pad_id=0,
        unk_id=3,
        character_coverage=1.0,
    )
    return model_prefix.with_suffix(".model")


def _write_minimal_model_config(path: Path, *, vocab_size: int, block_variant: str) -> None:
    payload = {
        "model": {
            "vocab_size": vocab_size,
            "dim": 16,
            "num_layers": 1,
            "heads": 4,
            "block_variant": block_variant,
            "titan_level": {"name": "titan", "update_period": 1},
            "cms_levels": [{"name": "cms_fast", "update_period": 1}],
        }
    }
    path.write_text(OmegaConf.to_yaml(OmegaConf.create(payload)))


def _write_checkpoint(path: Path, config_path: Path) -> None:
    cfg = OmegaConf.load(config_path)
    model = build_model_from_cfg(cfg.model)
    torch.save({"model": model.state_dict()}, path)


def test_compare_variants_cli_smoke(tmp_path: Path) -> None:
    vocab_size = 64
    spm_model = _train_tiny_sentencepiece(tmp_path, vocab_size=vocab_size)

    config_a = tmp_path / "a.yaml"
    config_b = tmp_path / "b.yaml"
    _write_minimal_model_config(config_a, vocab_size=vocab_size, block_variant="transformer")
    _write_minimal_model_config(config_b, vocab_size=vocab_size, block_variant="transformer")

    ckpt_a = tmp_path / "a.pt"
    ckpt_b = tmp_path / "b.pt"
    _write_checkpoint(ckpt_a, config_a)
    _write_checkpoint(ckpt_b, config_b)

    out_path = tmp_path / "out.json"
    cmd = [
        sys.executable,
        "scripts/eval/compare_variants.py",
        "--a-config",
        str(config_a),
        "--a-checkpoint",
        str(ckpt_a),
        "--b-config",
        str(config_b),
        "--b-checkpoint",
        str(ckpt_b),
        "--tokenizer-path",
        str(spm_model),
        "--device",
        "cpu",
        "--smoke",
        "--output",
        str(out_path),
    ]
    completed = subprocess.run(cmd, check=True, capture_output=True, text=True)
    assert completed.returncode == 0
    data = json.loads(out_path.read_text())
    assert "a" in data and "b" in data
    assert "passkey" in data["a"] and "niah" in data["a"]
    assert "accuracy_base" in data["a"]["passkey"]
```

### File: `tests/test_continual_classification.py`

```python
from pathlib import Path

import sentencepiece as spm
import torch

from nested_learning.continual_classification import ClassificationExample
from nested_learning.continual_streaming import (
    ContinualEvalConfig,
    build_streaming_tasks,
    evaluate_continual_classification,
)
from nested_learning.levels import LevelSpec
from nested_learning.memorize import MemorizeConfig
from nested_learning.model import HOPEModel, ModelConfig
from nested_learning.tokenizer import SentencePieceTokenizer


def _train_tiny_sentencepiece(tmp_path: Path, *, vocab_size: int) -> Path:
    corpus_path = tmp_path / "corpus.txt"
    corpus_path.write_text(
        "\n".join(
            [
                "Text: hello world Label: A",
                "Text: goodbye world Label: B",
                "Text: foo bar Label: C",
                "Text: baz qux Label: D",
            ]
        )
    )
    model_prefix = tmp_path / "spm_continual"
    spm.SentencePieceTrainer.Train(
        input=str(corpus_path),
        model_prefix=str(model_prefix),
        vocab_size=vocab_size,
        hard_vocab_limit=False,
        model_type="unigram",
        bos_id=1,
        eos_id=2,
        pad_id=0,
        unk_id=3,
        character_coverage=1.0,
    )
    return model_prefix.with_suffix(".model")


def _tiny_transformer_model(vocab_size: int) -> HOPEModel:
    cfg = ModelConfig(
        vocab_size=vocab_size,
        dim=16,
        num_layers=1,
        heads=4,
        titan_level=LevelSpec(name="titan", update_period=1),
        cms_levels=[LevelSpec(name="cms_fast", update_period=1)],
        block_variant="transformer",
    )
    return HOPEModel(cfg).eval()


def _toy_examples() -> list[ClassificationExample]:
    examples = []
    for label in ["A", "B", "C", "D"]:
        for idx in range(3):
            examples.append(ClassificationExample(text=f"example {idx} for {label}", label=label))
    return examples


def test_build_streaming_tasks_balanced_split() -> None:
    cfg = ContinualEvalConfig(task_size=2, seed=0, train_per_label=2, eval_per_label=1)
    tasks = build_streaming_tasks(_toy_examples(), cfg=cfg)
    assert len(tasks) == 2
    for task in tasks:
        assert len(task.labels) == 2
        assert len(task.train) == 4
        assert len(task.eval) == 2


def test_evaluate_continual_classification_runs(tmp_path: Path) -> None:
    vocab_size = 64
    spm_model = _train_tiny_sentencepiece(tmp_path, vocab_size=vocab_size)
    tokenizer = SentencePieceTokenizer(spm_model)
    model = _tiny_transformer_model(vocab_size)

    eval_cfg = ContinualEvalConfig(task_size=2, seed=0, train_per_label=2, eval_per_label=1)
    tasks = build_streaming_tasks(_toy_examples(), cfg=eval_cfg)

    memorize_cfg = MemorizeConfig(enabled=False)
    result, meta = evaluate_continual_classification(
        model,
        tokenizer,
        tasks,
        torch.device("cpu"),
        cfg=eval_cfg,
        memorize_cfg=memorize_cfg,
    )
    assert len(result.task_accuracy_matrix) == len(tasks)
    assert len(result.task_accuracy_matrix[0]) == len(tasks)
    assert 0.0 <= result.avg_accuracy_final <= 1.0
    assert "task_size" in meta


def test_evaluate_continual_classification_with_memorize_fast_state(tmp_path: Path) -> None:
    vocab_size = 64
    spm_model = _train_tiny_sentencepiece(tmp_path, vocab_size=vocab_size)
    tokenizer = SentencePieceTokenizer(spm_model)
    model = _tiny_transformer_model(vocab_size)

    eval_cfg = ContinualEvalConfig(task_size=2, seed=0, train_per_label=2, eval_per_label=1)
    tasks = build_streaming_tasks(_toy_examples(), cfg=eval_cfg)

    memorize_cfg = MemorizeConfig(enabled=True, steps=1, reset=False, use_fast_state=True)
    result, _meta = evaluate_continual_classification(
        model,
        tokenizer,
        tasks,
        torch.device("cpu"),
        cfg=eval_cfg,
        memorize_cfg=memorize_cfg,
    )
    assert len(result.per_task_forgetting) == len(tasks)
```

### File: `tests/test_data_split_fallbacks.py`

```python
from __future__ import annotations

import io
import subprocess
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.data import shard_corpus, train_tokenizer


def test_train_tokenizer_manifest_supports_text_data_files(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("hello world\nthis is a test\nanother line\n", encoding="utf-8")
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        "\n".join(
            [
                "datasets:",
                "  - name: local",
                "    dataset: text",
                "    split: train",
                "    text_column: text",
                f"    data_files: {corpus}",
                "    sample_limit: 10",
                "",
            ]
        ),
        encoding="utf-8",
    )
    specs = train_tokenizer._load_specs_from_manifest(manifest)  # noqa: SLF001
    assert len(specs) == 1
    assert specs[0].dataset == "text"
    assert specs[0].split == "train"
    assert specs[0].data_files == str(corpus)
    buf = io.StringIO()
    count = train_tokenizer._write_samples(specs[0], buf)  # noqa: SLF001
    assert count == 3


def test_shard_corpus_accepts_text_data_files_with_train_split(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(("hello world " * 100).strip() + "\n", encoding="utf-8")
    out_dir = tmp_path / "shards"
    cfg = shard_corpus.ShardConfig(
        name="local",
        dataset="text",
        split="train",
        subset=None,
        text_column="text",
        tokenizer_path=Path("tests/data/tiny_tokenizer.model"),
        seq_len=4,
        sequences_per_shard=2,
        output_dir=out_dir,
        eos_id=-1,
        max_records=10,
        data_files=str(corpus),
    )
    stats = shard_corpus.shard_dataset(cfg)
    assert stats["records"] > 0
    assert stats["sequences"] > 0
    assert stats["shards"] > 0
    assert list(out_dir.glob("shard_*.npy"))


def test_train_tokenizer_allows_small_corpus_with_no_hard_vocab_limit(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(("hello world\n" * 20).strip() + "\n", encoding="utf-8")
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        "\n".join(
            [
                "datasets:",
                "  - name: local",
                "    dataset: text",
                "    split: train",
                "    text_column: text",
                f"    data_files: {corpus}",
                "    sample_limit: 50",
                "",
            ]
        ),
        encoding="utf-8",
    )
    out_dir = tmp_path / "tokenizer"
    log_file = tmp_path / "tokenizer_log.json"
    repo_root = Path(__file__).resolve().parents[1]
    subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts/data/train_tokenizer.py"),
            "--manifest",
            str(manifest),
            "--vocab-size",
            "1000",
            "--model-type",
            "unigram",
            "--output-dir",
            str(out_dir),
            "--log-file",
            str(log_file),
            "--no-hard-vocab-limit",
        ],
        check=True,
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert (out_dir / "spm_1000_unigram.model").exists()
    assert log_file.exists()
```

### File: `tests/test_device_resolution.py`

```python
import torch

from nested_learning.device import resolve_device


def test_resolve_device_mps_falls_back_when_unavailable() -> None:
    device = resolve_device("mps")
    mps_available = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    assert device.type == ("mps" if mps_available else "cpu")

```

### File: `tests/test_distributed_fail_fast.py`

```python
import pytest
from omegaconf import OmegaConf

from nested_learning.training import _validate_distributed_config


def test_fail_if_paper_faithful_disabled_blocks_ddp_per_layer_teach() -> None:
    cfg = OmegaConf.create(
        {
            "train": {
                "fail_if_paper_faithful_disabled": True,
                "per_layer_teach_signal": True,
                "online_updates": False,
            }
        }
    )
    with pytest.raises(RuntimeError, match="per_layer_teach_signal"):
        _validate_distributed_config(cfg, distributed=True)


def test_fail_if_paper_faithful_disabled_blocks_ddp_online_updates() -> None:
    cfg = OmegaConf.create(
        {
            "train": {
                "fail_if_paper_faithful_disabled": True,
                "per_layer_teach_signal": False,
                "online_updates": True,
            }
        }
    )
    with pytest.raises(RuntimeError, match="online_updates"):
        _validate_distributed_config(cfg, distributed=True)


def test_fail_if_paper_faithful_disabled_allows_single_process() -> None:
    cfg = OmegaConf.create(
        {
            "train": {
                "fail_if_paper_faithful_disabled": True,
                "per_layer_teach_signal": True,
                "online_updates": True,
            }
        }
    )
    _validate_distributed_config(cfg, distributed=False)

```

### File: `tests/test_eval_builders.py`

```python
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from scripts.eval import zeroshot


def test_commonsenseqa_builder() -> None:
    sample = {
        "question": "Where would you most likely find a revolving door?",
        "choices": {
            "label": ["A", "B", "C"],
            "text": ["bank", "library", "garden"],
        },
        "answerKey": "B",
    }
    _, texts, target = zeroshot.build_commonsenseqa_texts(sample)
    assert len(texts) == 3
    assert target == 1
    assert "library" in texts[target]


def test_openbookqa_builder() -> None:
    sample = {
        "question_stem": "Plants need what to make food?",
        "choices": {
            "label": ["A", "B", "C", "D"],
            "text": ["sunlight", "soil", "wind", "music"],
        },
        "answerKey": "A",
    }
    _, texts, target = zeroshot.build_openbookqa_texts(sample)
    assert len(texts) == 4
    assert target == 0
    assert "sunlight" in texts[target]
```

### File: `tests/test_faithfulness_harness.py`

```python
import torch

from nested_learning.levels import LevelSpec
from nested_learning.model import HOPEModel, ModelConfig
from nested_learning.training import compute_teach_signal


def _cms_delta_l1(state, level_name: str) -> float:
    params = state.blocks[0].cms_params[level_name]
    return float(sum(delta.abs().sum().item() for delta in params.values()))


def test_e2e_update_paths_and_surprise_gate() -> None:
    torch.manual_seed(0)
    cfg = ModelConfig(
        vocab_size=32,
        dim=16,
        num_layers=1,
        heads=4,
        titan_level=LevelSpec(name="titan", update_period=1),
        cms_levels=(LevelSpec(name="cms_fast", update_period=2),),
        block_variant="hope_selfmod",
    )
    model = HOPEModel(cfg).eval()
    tokens = torch.randint(0, cfg.vocab_size, (1, 8))

    # Baseline: both selfmod and CMS should update in fast-state mode.
    state = model.init_fast_state()
    assert state.blocks[0].selfmod_state is not None
    cms_before = _cms_delta_l1(state, "cms_fast")
    assert cms_before == 0.0
    selfmod_before = state.blocks[0].selfmod_state.memory.w2.detach().clone()
    with torch.no_grad():
        logits_before = model(tokens, fast_state=state)
        teach = compute_teach_signal(model, logits_before, tokens)
        _ = model(tokens, teach_signal=teach, fast_state=state)
        logits_after = model(tokens, fast_state=state)
    cms_after = _cms_delta_l1(state, "cms_fast")
    selfmod_after = state.blocks[0].selfmod_state.memory.w2.detach().clone()
    assert cms_after > 0.0
    assert not torch.allclose(selfmod_before, selfmod_after)
    assert not torch.allclose(logits_before, logits_after)

    # Surprise gate: CMS updates should be blocked when threshold exceeds the computed surprise.
    gated_state = model.init_fast_state()
    with torch.no_grad():
        gated_logits = model(tokens, fast_state=gated_state)
        gated_teach = compute_teach_signal(model, gated_logits, tokens)
    surprise = float(gated_teach.norm(dim=-1).mean().item())
    model.set_surprise_threshold(surprise + 1.0)
    try:
        with torch.no_grad():
            _ = model(tokens, teach_signal=gated_teach, fast_state=gated_state)
        assert _cms_delta_l1(gated_state, "cms_fast") == 0.0
    finally:
        model.set_surprise_threshold(None)

```

### File: `tests/test_fast_state_batch_semantics.py`

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
    with pytest.raises(RuntimeError, match="fast state"):
        _validate_fast_state_batch_semantics(cfg)


def test_fast_state_batch_semantics_allows_batch1() -> None:
    cfg = OmegaConf.create(
        {
            "train": {"use_fast_state": True, "fail_if_paper_faithful_disabled": True},
            "data": {"batch_size": 1},
        }
    )
    _validate_fast_state_batch_semantics(cfg)

```

### File: `tests/test_fast_state_forward_equivalence.py`

```python
import torch

from nested_learning.levels import LevelSpec
from nested_learning.model import HOPEModel, ModelConfig


def test_fast_state_zero_deltas_matches_meta_forward() -> None:
    torch.manual_seed(0)
    cfg = ModelConfig(
        vocab_size=64,
        dim=16,
        num_layers=1,
        heads=2,
        titan_level=LevelSpec(name="titan", update_period=1),
        cms_levels=(LevelSpec(name="cms_fast", update_period=1),),
        block_variant="hope_hybrid",
    )
    model = HOPEModel(cfg).eval()
    tokens = torch.randint(0, cfg.vocab_size, (1, 8))
    fast_state = model.init_fast_state()
    with torch.no_grad():
        logits_meta = model(tokens)
        logits_fast = model(tokens, fast_state=fast_state)
    assert torch.allclose(logits_meta, logits_fast, atol=1e-6)

```

### File: `tests/test_fast_state_meta_grads.py`

```python
import torch

from nested_learning.levels import LevelSpec
from nested_learning.model import HOPEModel, ModelConfig
from nested_learning.training import _is_memory_param_name


def test_fast_state_preserves_outer_grads_for_memory_meta_params() -> None:
    torch.manual_seed(0)
    cfg = ModelConfig(
        vocab_size=64,
        dim=16,
        num_layers=1,
        heads=2,
        titan_level=LevelSpec(name="titan", update_period=1),
        cms_levels=(LevelSpec(name="cms_fast", update_period=1),),
        block_variant="hope_hybrid",
    )
    model = HOPEModel(cfg)
    fast_state = model.init_fast_state()
    tokens = torch.randint(0, cfg.vocab_size, (1, 8))
    logits = model(tokens, fast_state=fast_state)
    loss = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)),
        tokens[:, 1:].reshape(-1),
    )
    loss.backward()

    memory_param_names = [
        name
        for name, param in model.named_parameters()
        if param.requires_grad and _is_memory_param_name(name)
    ]
    assert memory_param_names, "Test expected at least one memory parameter"
    assert any(
        model.get_parameter(name).grad is not None for name in memory_param_names
    ), "Expected at least one memory parameter grad in fast_state mode"
```

### File: `tests/test_fast_state_selfmod_meta_grads.py`

```python
import torch
import torch.nn.functional as F

from nested_learning.levels import LevelSpec
from nested_learning.model import HOPEModel, ModelConfig


def test_hope_selfmod_fast_state_preserves_meta_forward_at_init() -> None:
    torch.manual_seed(0)
    cfg = ModelConfig(
        vocab_size=64,
        dim=16,
        num_layers=1,
        heads=2,
        titan_level=LevelSpec(name="titan", update_period=1),
        cms_levels=(),
        block_variant="hope_selfmod",
    )
    model = HOPEModel(cfg).eval()
    tokens = torch.randint(0, cfg.vocab_size, (1, 8))
    fast_state = model.init_fast_state()
    with torch.no_grad():
        logits_meta = model(tokens)
        logits_fast = model(tokens, fast_state=fast_state)
    assert torch.allclose(logits_meta, logits_fast, atol=1e-6)


def test_hope_selfmod_fast_state_preserves_outer_grads_for_meta_memory_init() -> None:
    torch.manual_seed(0)
    cfg = ModelConfig(
        vocab_size=64,
        dim=16,
        num_layers=1,
        heads=2,
        titan_level=LevelSpec(name="titan", update_period=1),
        cms_levels=(),
        block_variant="hope_selfmod",
    )
    model = HOPEModel(cfg)
    tokens = torch.randint(0, cfg.vocab_size, (1, 8))
    fast_state = model.init_fast_state()
    logits = model(tokens, fast_state=fast_state)
    loss = F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)),
        tokens[:, 1:].reshape(-1),
    )
    loss.backward()
    block = model.blocks[0]
    selfmod = getattr(block, "selfmod", None)
    assert selfmod is not None
    grad = selfmod.m_memory.w1.weight.grad
    assert grad is not None
    assert grad.abs().sum().item() > 0.0

```

### File: `tests/test_hope_block.py`

```python
import torch

from nested_learning.hope.block import HOPEBlock, HOPEBlockConfig
from nested_learning.levels import LevelSpec


def make_block() -> HOPEBlock:
    config = HOPEBlockConfig(
        dim=32,
        heads=4,
        titan_level=LevelSpec(name="titan", update_period=2),
        cms_levels=[LevelSpec(name="fast", update_period=1)],
    )
    return HOPEBlock(config)


def test_hope_block_forward() -> None:
    block = make_block()
    tokens = torch.randn(2, 8, 32)
    out = block(tokens)
    assert out.shape == tokens.shape


def test_hope_block_self_mod() -> None:
    block = make_block()
    tokens = torch.randn(2, 8, 32)
    teach = torch.randn_like(tokens)
    out = block(tokens, teach_signal=teach)
    assert out.shape == tokens.shape
```

### File: `tests/test_hope_selfmod_fast_state_meta_unchanged.py`

```python
import torch

from nested_learning.levels import LevelSpec
from nested_learning.memorize import snapshot_state_dict
from nested_learning.model import HOPEModel, ModelConfig
from nested_learning.training import compute_teach_signal


def test_hope_selfmod_fast_state_updates_do_not_mutate_meta_params() -> None:
    torch.manual_seed(0)
    cfg = ModelConfig(
        vocab_size=32,
        dim=16,
        num_layers=1,
        heads=4,
        titan_level=LevelSpec(name="titan", update_period=1),
        cms_levels=(),
        block_variant="hope_selfmod",
        self_mod_lr=1.0,
    )
    model = HOPEModel(cfg)
    baseline = snapshot_state_dict(model)
    fast_state = model.init_fast_state()
    tokens = torch.randint(0, cfg.vocab_size, (1, 8))
    with torch.no_grad():
        logits = model(tokens, fast_state=fast_state)
        teach = compute_teach_signal(model, logits, tokens)
        _ = model(tokens, teach_signal=teach, fast_state=fast_state)
    for name, value in model.state_dict().items():
        assert torch.allclose(baseline[name], value.cpu(), atol=1e-6)

```

### File: `tests/test_hope_selfmod_integration.py`

```python
import torch

from nested_learning.levels import LevelSpec
from nested_learning.model import HOPEModel, ModelConfig
from nested_learning.training import compute_teach_signal


def test_hope_selfmod_variant_updates_selfmod_state_in_fast_mode() -> None:
    cfg = ModelConfig(
        vocab_size=32,
        dim=16,
        num_layers=1,
        heads=4,
        titan_level=LevelSpec(name="titan", update_period=1),
        cms_levels=[LevelSpec(name="cms_fast", update_period=2)],
        block_variant="hope_selfmod",
    )
    model = HOPEModel(cfg)
    state = model.init_fast_state()
    assert state.blocks[0].selfmod_state is not None
    before = state.blocks[0].selfmod_state.memory.w2.detach().clone()

    tokens = torch.randint(0, cfg.vocab_size, (1, 6))
    with torch.no_grad():
        logits = model(tokens, fast_state=state)
        teach = compute_teach_signal(model, logits, tokens)
        _ = model(tokens, teach_signal=teach, fast_state=state)

    after = state.blocks[0].selfmod_state.memory.w2.detach().clone()
    assert not torch.allclose(before.unsqueeze(0), after)
```

### File: `tests/test_hope_selfmod_update_pass.py`

```python
import torch

from nested_learning.levels import LevelSpec
from nested_learning.model import HOPEModel, ModelConfig
from nested_learning.training import compute_teach_signal


def test_hope_selfmod_updates_module_params_only_in_update_pass() -> None:
    torch.manual_seed(0)
    cfg = ModelConfig(
        vocab_size=32,
        dim=16,
        num_layers=1,
        heads=4,
        titan_level=LevelSpec(name="titan", update_period=1),
        cms_levels=(),
        block_variant="hope_selfmod",
        self_mod_lr=1.0,
    )
    model = HOPEModel(cfg)
    tokens = torch.randint(0, cfg.vocab_size, (1, 8))
    before = model.blocks[0].selfmod.m_memory.w2.weight.detach().clone()

    _ = model(tokens)
    after_forward = model.blocks[0].selfmod.m_memory.w2.weight.detach().clone()
    assert torch.allclose(before, after_forward, atol=1e-6, rtol=1e-6)

    with torch.no_grad():
        logits = model(tokens)
        teach = compute_teach_signal(model, logits, tokens)
        _ = model(tokens, teach_signal=teach)
    after_update = model.blocks[0].selfmod.m_memory.w2.weight.detach().clone()
    assert not torch.allclose(after_forward, after_update)

```

### File: `tests/test_levels.py`

```python
from nested_learning.levels import LevelClock, LevelSpec


def test_level_clock_updates_on_schedule() -> None:
    specs = [LevelSpec(name="fast", update_period=1), LevelSpec(name="slow", update_period=3)]
    clock = LevelClock(specs)
    updates = []
    for step in range(5):
        for spec in specs:
            if clock.should_update(spec.name):
                updates.append((step, spec.name))
                clock.record_update(spec.name)
        clock.tick()
    assert updates[0] == (0, "fast")
    assert any(level == "slow" for _, level in updates)
```

### File: `tests/test_m3.py`

```python
import torch

from nested_learning.optim.m3 import M3


def test_m3_updates_and_slow_momentum() -> None:
    torch.manual_seed(0)
    param = torch.nn.Parameter(torch.ones(2, 2))
    opt = M3(
        [param],
        lr=0.1,
        beta1=0.9,
        beta2=0.9,
        beta3=0.5,
        alpha=1.0,
        ns_steps=1,
        slow_chunk=2,
        eps=1e-6,
    )
    param.grad = torch.ones_like(param)
    opt.step()
    first = param.detach().clone()
    param.grad = torch.ones_like(param)
    opt.step()
    state = opt.state[param]
    assert not torch.allclose(first, param)
    assert torch.any(state["o2"] != 0)
```

### File: `tests/test_m3_slow_timing.py`

```python
import torch

from nested_learning.optim.m3 import M3


def test_m3_slow_momentum_applies_next_chunk_not_boundary_step() -> None:
    param = torch.nn.Parameter(torch.tensor([0.0]))
    opt = M3(
        [param],
        lr=1.0,
        beta1=1.0,
        beta2=0.0,
        beta3=1.0,
        alpha=1.0,
        eps=1.0,
        ns_steps=0,
        slow_chunk=2,
        weight_decay=0.0,
    )
    param.grad = torch.tensor([1.0])
    opt.step()
    param.grad = torch.tensor([1.0])
    opt.step()
    # With correct timing, the slow momentum (o2) is updated after step 2 and therefore
    # does not affect the step-2 update itself.
    assert torch.allclose(param.detach(), torch.tensor([-3.0]))

```

### File: `tests/test_memorization.py`

```python
import torch

from nested_learning.levels import LevelSpec
from nested_learning.memorize import MemorizeConfig, memorize_tokens, snapshot_state_dict
from nested_learning.model import HOPEModel, ModelConfig


def _tiny_model() -> HOPEModel:
    titan = LevelSpec(name="titan", update_period=2, optimizer_key="titan_opt")
    cms = [
        LevelSpec(name="cms_fast", update_period=1, optimizer_key="cms_opt"),
        LevelSpec(name="cms_mid", update_period=2, optimizer_key="cms_opt"),
    ]
    cfg = ModelConfig(
        vocab_size=32,
        dim=16,
        num_layers=1,
        heads=4,
        titan_level=titan,
        cms_levels=cms,
        optimizers=None,
        teach_scale=0.1,
    )
    return HOPEModel(cfg)


def _tiny_model_update_every_call() -> HOPEModel:
    titan = LevelSpec(name="titan", update_period=1, optimizer_key="titan_opt")
    cms = [
        LevelSpec(name="cms_fast", update_period=1, optimizer_key="cms_opt"),
        LevelSpec(name="cms_mid", update_period=1, optimizer_key="cms_opt"),
    ]
    cfg = ModelConfig(
        vocab_size=32,
        dim=16,
        num_layers=1,
        heads=4,
        titan_level=titan,
        cms_levels=cms,
        optimizers=None,
        teach_scale=0.1,
    )
    return HOPEModel(cfg)


def _tiny_model_with_self_mod_lr(lr: float) -> HOPEModel:
    titan = LevelSpec(name="titan", update_period=1, optimizer_key="titan_opt")
    cfg = ModelConfig(
        vocab_size=32,
        dim=16,
        num_layers=1,
        heads=4,
        titan_level=titan,
        cms_levels=(),
        optimizers=None,
        teach_scale=0.1,
        self_mod_lr=lr,
    )
    return HOPEModel(cfg)


def _fast_titan_delta_norm(fast_state, before: dict[str, torch.Tensor]) -> float:
    block_state = fast_state.blocks[0]
    if block_state.titan_params is None:
        return 0.0
    total = 0.0
    for name, value in block_state.titan_params.items():
        total += (value.cpu() - before[name]).norm().item()
    return total


def test_memorize_fast_state_does_not_mutate_meta_params() -> None:
    torch.manual_seed(0)
    model = _tiny_model()
    tokens = torch.randint(0, model.config.vocab_size, (1, 8))
    baseline = snapshot_state_dict(model)
    fast_state = model.init_fast_state()
    cfg = MemorizeConfig(enabled=True, steps=2, use_fast_state=True)
    memorize_tokens(model, tokens, cfg, fast_state=fast_state)
    for name, param in model.state_dict().items():
        assert torch.allclose(baseline[name], param.cpu(), atol=1e-6)


def test_memorize_fast_state_changes_outputs_and_resets() -> None:
    torch.manual_seed(0)
    model = _tiny_model()
    tokens = torch.randint(0, model.config.vocab_size, (1, 8))
    fast_state = model.init_fast_state()
    with torch.no_grad():
        logits_before = model(tokens, fast_state=fast_state).detach().clone()

    cfg = MemorizeConfig(enabled=True, steps=1, use_fast_state=True)
    memorize_tokens(model, tokens, cfg, fast_state=fast_state)
    with torch.no_grad():
        logits_after = model(tokens, fast_state=fast_state).detach().clone()

    assert not torch.allclose(logits_before, logits_after)

    reset_state = model.init_fast_state()
    with torch.no_grad():
        logits_reset = model(tokens, fast_state=reset_state).detach().clone()
    assert torch.allclose(logits_before, logits_reset, atol=1e-6)


def test_memorize_respects_surprise_threshold() -> None:
    model = _tiny_model()
    tokens = torch.randint(0, model.config.vocab_size, (1, 8))
    fast_state = model.init_fast_state()
    block_state = fast_state.blocks[0]
    titan_before = {k: v.cpu().clone() for k, v in block_state.titan_params.items()}  # type: ignore[union-attr]
    cfg = MemorizeConfig(enabled=True, steps=1, surprise_threshold=1e6, use_fast_state=True)
    memorize_tokens(model, tokens, cfg, fast_state=fast_state)
    assert _fast_titan_delta_norm(fast_state, titan_before) == 0.0


def test_memorize_paths_filter_blocks_updates() -> None:
    model = _tiny_model()
    tokens = torch.randint(0, model.config.vocab_size, (1, 8))
    fast_state = model.init_fast_state()
    block_state = fast_state.blocks[0]
    titan_before = {k: v.cpu().clone() for k, v in block_state.titan_params.items()}  # type: ignore[union-attr]
    cfg = MemorizeConfig(enabled=True, steps=1, paths=(), use_fast_state=True)
    memorize_tokens(model, tokens, cfg, fast_state=fast_state)
    assert _fast_titan_delta_norm(fast_state, titan_before) == 0.0


def test_memorize_online_chunking_updates_once_per_target() -> None:
    model = _tiny_model_update_every_call()
    tokens = torch.randint(0, model.config.vocab_size, (1, 8))
    fast_state = model.init_fast_state()
    cfg = MemorizeConfig(enabled=True, online_chunk_size=1, use_fast_state=True)
    stats = memorize_tokens(model, tokens, cfg, fast_state=fast_state)
    assert stats["titan_update_events"] == float(tokens.size(1) - 1)


def test_teach_mask_restricts_memorization_updates() -> None:
    torch.manual_seed(0)
    model = _tiny_model_update_every_call()
    tokens = torch.randint(0, model.config.vocab_size, (1, 8))
    cfg = MemorizeConfig(enabled=True, steps=1, use_fast_state=True, paths=("cms_fast",))

    fast_state_masked = model.init_fast_state()
    zero_mask = torch.zeros((tokens.size(0), tokens.size(1)))
    stats_masked = memorize_tokens(
        model, tokens, cfg, fast_state=fast_state_masked, teach_mask=zero_mask
    )
    assert stats_masked["cms_fast_update_events"] == 0.0

    fast_state_full = model.init_fast_state()
    one_mask = torch.ones((tokens.size(0), tokens.size(1)))
    stats_full = memorize_tokens(
        model,
        tokens,
        cfg,
        fast_state=fast_state_full,
        teach_mask=one_mask,
    )
    assert stats_full["cms_fast_update_events"] > 0.0


def test_self_mod_lr_scales_fast_state_update_magnitude() -> None:
    torch.manual_seed(0)
    model_hi = _tiny_model_with_self_mod_lr(1e-3)
    torch.manual_seed(0)
    model_lo = _tiny_model_with_self_mod_lr(1e-4)
    tokens = torch.randint(0, model_hi.config.vocab_size, (1, 8))

    state_hi = model_hi.init_fast_state()
    state_lo = model_lo.init_fast_state()
    titan_hi_before = {k: v.cpu().clone() for k, v in state_hi.blocks[0].titan_params.items()}  # type: ignore[union-attr]
    titan_lo_before = {k: v.cpu().clone() for k, v in state_lo.blocks[0].titan_params.items()}  # type: ignore[union-attr]

    cfg = MemorizeConfig(enabled=True, steps=1, paths=("titan",), use_fast_state=True)
    memorize_tokens(model_hi, tokens, cfg, fast_state=state_hi)
    memorize_tokens(model_lo, tokens, cfg, fast_state=state_lo)

    hi = _fast_titan_delta_norm(state_hi, titan_hi_before)
    lo = _fast_titan_delta_norm(state_lo, titan_lo_before)
    assert hi > lo * 5.0
```

### File: `tests/test_model.py`

```python
import torch

from nested_learning.levels import LevelSpec
from nested_learning.model import HOPEModel, ModelConfig


def test_model_forward() -> None:
    config = ModelConfig(
        vocab_size=100,
        dim=32,
        num_layers=1,
        heads=4,
        titan_level=LevelSpec(name="titan", update_period=2),
        cms_levels=[LevelSpec(name="fast", update_period=1)],
    )
    model = HOPEModel(config)
    tokens = torch.randint(0, 100, (2, 10))
    logits = model(tokens)
    assert logits.shape == (2, 10, 100)
```

### File: `tests/test_optim.py`

```python
import torch

from nested_learning.optim.deep import DeepMomentum


def test_deep_momentum_nl_preconditioner_projects_grad() -> None:
    grad = torch.randn(4, 6)
    context = torch.randn(6)
    optimizer = DeepMomentum(beta=0.0, beta2=0.0, variant="nl_l2_precond")
    update = optimizer(grad, context=context)
    unit = context / context.norm()
    expected = grad - (grad * unit).sum(dim=-1, keepdim=True) * unit
    assert torch.allclose(update, expected, atol=1e-5, rtol=1e-4)
    assert optimizer.last_metrics["ctx_norm"] > 0
    assert optimizer.last_metrics["proj_norm"] >= 0


def test_deep_momentum_nl_preconditioner_reduces_simple_objective() -> None:
    torch.manual_seed(0)
    context = torch.randn(6)
    weights = torch.randn(6)
    grad = torch.dot(weights, context) * context
    optimizer = DeepMomentum(beta=0.0, beta2=0.0, variant="nl_l2_precond")
    update = optimizer(grad, context=context)
    with torch.no_grad():
        old_obj = 0.5 * torch.dot(weights, context) ** 2
        new_weights = weights - 0.1 * update
        new_obj = 0.5 * torch.dot(new_weights, context) ** 2
    assert new_obj < old_obj


def test_deep_momentum_keeps_state_per_param_key() -> None:
    optimizer = DeepMomentum(beta=0.5, beta2=0.0, variant="preconditioned")
    grad_a = torch.ones(2, 3)
    grad_b = torch.ones(5)
    out_a1 = optimizer(grad_a, param_key="a").detach().clone()
    _ = optimizer(grad_b, param_key="b")
    out_a2 = optimizer(grad_a, param_key="a").detach().clone()
    assert out_a2.shape == out_a1.shape
    assert torch.all(out_a2 > out_a1)
    assert set(optimizer.state.keys()) == {"a", "b"}


def test_deep_momentum_nl_preconditioner_skips_mismatched_shapes() -> None:
    optimizer = DeepMomentum(beta=0.0, beta2=0.0, variant="nl_l2_precond")
    context = torch.randn(512)
    grad_bias = torch.randn(2048)
    out = optimizer(grad_bias, context=context)
    assert torch.allclose(out, grad_bias)
    assert optimizer.last_metrics["proj_skipped"] == 1.0
```

### File: `tests/test_optimizer_param_policy.py`

```python
import torch
from omegaconf import OmegaConf

from nested_learning.levels import LevelSpec
from nested_learning.model import HOPEModel, ModelConfig
from nested_learning.training import _build_optimizer, _is_memory_param_name


def _make_small_hope_model() -> HOPEModel:
    return HOPEModel(
        ModelConfig(
            vocab_size=128,
            dim=16,
            num_layers=2,
            heads=2,
            titan_level=LevelSpec(name="titan", update_period=2),
            cms_levels=(LevelSpec(name="cms_fast", update_period=1),),
            block_variant="hope_hybrid",
        )
    )


def _optimizer_param_set(optimizer: torch.optim.Optimizer) -> set[torch.nn.Parameter]:
    params: set[torch.nn.Parameter] = set()
    for group in optimizer.param_groups:
        for param in group["params"]:
            params.add(param)
    return params


def test_param_policy_all_includes_all_trainable_params() -> None:
    model = _make_small_hope_model()
    cfg = OmegaConf.create({"optim": {"type": "adamw", "lr": 1e-3, "param_policy": "all"}})
    optimizer = _build_optimizer(model, cfg, device=torch.device("cpu"))
    opt_params = _optimizer_param_set(optimizer)
    expected = {p for _n, p in model.named_parameters() if p.requires_grad}
    assert opt_params == expected
    has_memory = any(
        _is_memory_param_name(name) for name, p in model.named_parameters() if p.requires_grad
    )
    assert has_memory


def test_param_policy_exclude_memory_drops_memory_params() -> None:
    model = _make_small_hope_model()
    cfg = OmegaConf.create(
        {"optim": {"type": "adamw", "lr": 1e-3, "param_policy": "exclude_memory"}}
    )
    optimizer = _build_optimizer(model, cfg, device=torch.device("cpu"))
    opt_params = _optimizer_param_set(optimizer)
    expected = {
        p
        for name, p in model.named_parameters()
        if p.requires_grad and not _is_memory_param_name(name)
    }
    assert opt_params == expected
    contains_memory = any(
        _is_memory_param_name(name) for name, p in model.named_parameters() if p in opt_params
    )
    assert not contains_memory


def test_param_policy_only_memory_keeps_only_memory_params() -> None:
    model = _make_small_hope_model()
    cfg = OmegaConf.create(
        {"optim": {"type": "adamw", "lr": 1e-3, "param_policy": "only_memory"}}
    )
    optimizer = _build_optimizer(model, cfg, device=torch.device("cpu"))
    opt_params = _optimizer_param_set(optimizer)
    expected = {
        p
        for name, p in model.named_parameters()
        if p.requires_grad and _is_memory_param_name(name)
    }
    assert expected
    assert opt_params == expected
```

### File: `tests/test_paper_faithful_configs.py`

```python
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from nested_learning.training import build_model_from_cfg


def _compose_config(name: str):
    config_dir = Path(__file__).resolve().parents[1] / "configs"
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        return compose(config_name=name)


def test_pilot_paper_faithful_config_composes() -> None:
    cfg = _compose_config("pilot_paper_faithful")
    assert cfg.model.cms_flush_partial_at_end is True
    assert cfg.model.surprise_threshold is None
    assert cfg.data.batch_size == 1
    assert cfg.train.use_fast_state is True
    assert cfg.train.fail_if_paper_faithful_disabled is True
    assert cfg.optim.param_policy == "all"
    build_model_from_cfg(cfg.model)


def test_pilot_selfmod_paper_faithful_config_composes() -> None:
    cfg = _compose_config("pilot_selfmod_paper_faithful")
    assert cfg.model.block_variant == "hope_selfmod"
    assert cfg.model.cms_flush_partial_at_end is True
    assert cfg.model.surprise_threshold is None
    assert cfg.model.self_mod_use_skip is False
    assert cfg.data.batch_size == 1
    assert cfg.train.use_fast_state is True
    assert cfg.train.fail_if_paper_faithful_disabled is True
    assert cfg.optim.param_policy == "all"
    build_model_from_cfg(cfg.model)
```

### File: `tests/test_paper_hypotheses.py`

```python
import math

import torch
from omegaconf import OmegaConf

from nested_learning.fast_state import build_block_fast_state
from nested_learning.hope.block import HOPEAttentionBlock, HOPEAttentionBlockConfig
from nested_learning.levels import LevelSpec
from nested_learning.training import run_training_loop


def _run_multilevel_cms_block(
    *,
    seq_len: int,
    flush_partial: bool,
    use_fast_state: bool,
) -> dict[str, dict[str, float]]:
    torch.manual_seed(0)
    cfg = HOPEAttentionBlockConfig(
        dim=8,
        heads=1,
        cms_levels=(
            LevelSpec(name="fast", update_period=2),
            LevelSpec(name="slow", update_period=4),
        ),
        cms_flush_partial_at_end=flush_partial,
        cms_online_updates=True,
        cms_chunk_reduction="sum",
    )
    block = HOPEAttentionBlock(cfg)
    x = torch.randn(1, seq_len, 8)
    teach = torch.ones_like(x)
    fast_state = None
    if use_fast_state:
        fast_state = build_block_fast_state(
            titan_module=None,
            cms_blocks=dict(block.cms.blocks.items()),
            specs=cfg.cms_levels,
            optimizer_configs=cfg.optimizer_configs,
            default_lr=cfg.self_mod_lr,
        )
    _out = block(x, teach_signal=teach, fast_state=fast_state)
    return block.pop_update_stats()


def test_cms_frequency_matches_floor_schedule_without_partial_flush() -> None:
    # Hypothesis (Eq. 31 semantics): with flush disabled, each level applies floor(T / C_l) updates.
    seq_len = 9
    for use_fast_state in (False, True):
        stats = _run_multilevel_cms_block(
            seq_len=seq_len,
            flush_partial=False,
            use_fast_state=use_fast_state,
        )
        assert stats["cms.fast"]["gate_hit"] == 4.0  # floor(9/2)
        assert stats["cms.fast"]["chunk_tokens"] == 8.0
        assert stats["cms.slow"]["gate_hit"] == 2.0  # floor(9/4)
        assert stats["cms.slow"]["chunk_tokens"] == 8.0


def test_cms_frequency_matches_ceil_schedule_with_partial_flush() -> None:
    # Hypothesis (Eq. 31 semantics + flush): with flush enabled, each level applies ceil(T / C_l).
    seq_len = 9
    for use_fast_state in (False, True):
        stats = _run_multilevel_cms_block(
            seq_len=seq_len,
            flush_partial=True,
            use_fast_state=use_fast_state,
        )
        assert stats["cms.fast"]["gate_hit"] == 5.0  # ceil(9/2)
        assert stats["cms.fast"]["chunk_tokens"] == 9.0
        assert stats["cms.slow"]["gate_hit"] == 3.0  # ceil(9/4)
        assert stats["cms.slow"]["chunk_tokens"] == 9.0


def test_online_training_smoke_produces_nonzero_teach_and_update_metrics() -> None:
    # Hypothesis: paper-faithful online updates (with inferred chunk size from period=1) remain
    # numerically stable and produce non-zero teach/update telemetry in a tiny run.
    cfg = OmegaConf.create(
        {
            "model": {
                "vocab_size": 64,
                "dim": 16,
                "num_layers": 1,
                "heads": 2,
                "block_variant": "hope_attention",
                "titan_level": {"name": "titan", "update_period": 1, "optimizer_key": "titan_opt"},
                "cms_levels": [
                    {"name": "cms_fast", "update_period": 1, "optimizer_key": "cms_opt"},
                ],
                "optimizers": {
                    "cms_opt": {
                        "type": "deep_momentum",
                        "params": {"beta": 0.9, "beta2": 0.999},
                    }
                },
                "teach_scale": 0.1,
            },
            "data": {
                "source": "synthetic",
                "vocab_size": 64,
                "seq_len": 6,
                "dataset_size": 64,
                "batch_size": 1,
                "num_workers": 0,
            },
            "train": {
                "steps": 2,
                "log_interval": 1,
                "device": "cpu",
                "seed": 1234,
                "deterministic": True,
                "online_updates": True,
                "online_chunk_size": 0,
                "per_layer_teach_signal": False,
                "use_fast_state": True,
                "fail_if_paper_faithful_disabled": True,
                "mixed_precision": {"enabled": False, "dtype": "bf16"},
                "compile": {"enable": False},
                "checkpoint": {"enable": False},
            },
            "optim": {"type": "adamw", "lr": 3e-4, "fused": False},
            "logging": {"enabled": False},
        }
    )

    metrics = run_training_loop(cfg, device=torch.device("cpu"), distributed=False)
    assert math.isfinite(metrics["loss"])
    assert math.isfinite(metrics["ppl"])
    assert metrics["teach_signal_norm"] > 0.0
    grad_keys = [k for k in metrics if k.endswith(".grad_norm")]
    assert grad_keys
    assert any(metrics[k] > 0.0 for k in grad_keys)
```

### File: `tests/test_phase2_memorization_delta.py`

```python
import torch

from nested_learning.levels import LevelSpec
from nested_learning.memorize import MemorizeConfig, memorize_tokens
from nested_learning.model import HOPEModel, ModelConfig


def _tiny_variant(variant: str) -> HOPEModel:
    titan = LevelSpec(name="titan", update_period=1, optimizer_key="titan_opt")
    cms = (LevelSpec(name="cms_fast", update_period=1, optimizer_key="cms_opt"),)
    cfg = ModelConfig(
        vocab_size=32,
        dim=16,
        num_layers=1,
        heads=4,
        titan_level=titan,
        cms_levels=cms,
        optimizers=None,
        teach_scale=0.1,
        block_variant=variant,
    )
    return HOPEModel(cfg).eval()


def test_hope_attention_adapts_transformer_does_not() -> None:
    tokens = torch.randint(0, 32, (1, 16), generator=torch.Generator().manual_seed(1337))
    cfg = MemorizeConfig(enabled=True, steps=1, use_fast_state=True, paths=("cms_fast",))

    torch.manual_seed(0)
    hope = _tiny_variant("hope_attention")
    state = hope.init_fast_state()
    with torch.no_grad():
        before = hope(tokens, fast_state=state).detach().clone()
    stats = memorize_tokens(hope, tokens, cfg, fast_state=state)
    with torch.no_grad():
        after = hope(tokens, fast_state=state).detach().clone()
    assert not torch.allclose(before, after)
    assert stats["cms_fast_update_events"] > 0.0

    torch.manual_seed(0)
    transformer = _tiny_variant("transformer")
    state = transformer.init_fast_state()
    with torch.no_grad():
        before = transformer(tokens, fast_state=state).detach().clone()
    stats = memorize_tokens(transformer, tokens, cfg, fast_state=state)
    with torch.no_grad():
        after = transformer(tokens, fast_state=state).detach().clone()
    assert torch.allclose(before, after, atol=0.0, rtol=0.0)
    assert stats["cms_fast_update_events"] == 0.0

```

### File: `tests/test_residual_mlp_memory.py`

```python
import torch
import torch.nn.functional as F

from nested_learning.titan.self_modifying import ResidualMLPMemory


def test_residual_mlp_memory_matches_eq91_when_dims_match() -> None:
    torch.manual_seed(0)
    mem = ResidualMLPMemory(in_dim=8, out_dim=8, hidden_dim=8, activation=F.gelu, use_skip=False)
    assert mem.w_skip is None
    x = torch.randn(2, 5, 8)
    with torch.no_grad():
        expected = x + mem.w1(mem.activation(mem.w2(x)))
        actual = mem(x)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_residual_mlp_memory_uses_projection_skip_when_dims_differ() -> None:
    mem = ResidualMLPMemory(in_dim=8, out_dim=1, hidden_dim=8, activation=F.gelu, use_skip=True)
    assert mem.w_skip is not None


def test_residual_mlp_memory_disables_projection_skip_in_faithful_mode() -> None:
    torch.manual_seed(0)
    mem = ResidualMLPMemory(in_dim=8, out_dim=1, hidden_dim=8, activation=F.gelu, use_skip=False)
    assert mem.w_skip is None
    x = torch.randn(2, 5, 8)
    with torch.no_grad():
        expected = mem.w1(mem.activation(mem.w2(x)))
        actual = mem(x)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
```

### File: `tests/test_self_modifying_titans.py`

```python
import torch

from nested_learning.titan.self_modifying import SelfModifyingTitans, SelfModifyingTitansConfig


def test_self_modifying_titans_forward_shape() -> None:
    model = SelfModifyingTitans(SelfModifyingTitansConfig(dim=8))
    x = torch.randn(2, 5, 8)
    out = model(x)
    assert out.shape == x.shape


def test_self_modifying_titans_updates_fast_state() -> None:
    torch.manual_seed(0)
    model = SelfModifyingTitans(SelfModifyingTitansConfig(dim=8, eta_scale=1.0))
    x = torch.randn(1, 6, 8)
    state = model.init_fast_state()
    before = state.memory.w2.detach().clone()
    out, updated = model.forward_with_updates(x, state)
    assert out.shape == (1, 6, 8)
    assert not torch.allclose(before.unsqueeze(0), updated.memory.w2)


def test_self_modifying_titans_supports_batch_fast_state_updates() -> None:
    torch.manual_seed(0)
    model = SelfModifyingTitans(SelfModifyingTitansConfig(dim=8, eta_scale=1.0))
    x = torch.randn(2, 6, 8)
    state = model.init_fast_state()
    out, updated = model.forward_with_updates(x, state)
    assert out.shape == (2, 6, 8)
    assert updated.memory.w2.shape == (2, 8, 8)
    assert not torch.allclose(updated.memory.w2[0], updated.memory.w2[1])


def test_self_modifying_titans_chunked_outputs_match_no_update_with_single_chunk() -> None:
    torch.manual_seed(0)
    seq_len = 6
    model = SelfModifyingTitans(
        SelfModifyingTitansConfig(
            dim=8,
            eta_scale=1.0,
            chunk_size_other=seq_len,
            chunk_size_memory=seq_len,
        )
    )
    x = torch.randn(1, seq_len, 8)
    state = model.init_fast_state()
    before = state.memory.w2.detach().clone()

    out_no_update = model.forward_with_state(x, state)
    out_chunked, updated = model.forward_with_updates(x, state)

    assert torch.allclose(out_chunked, out_no_update, atol=1e-6)
    assert not torch.allclose(before.unsqueeze(0), updated.memory.w2)


def test_self_modifying_titans_flushes_partial_chunks_for_memory_updates() -> None:
    torch.manual_seed(0)
    model = SelfModifyingTitans(
        SelfModifyingTitansConfig(
            dim=8,
            eta_scale=1.0,
            chunk_size_other=1,
            chunk_size_memory=4,
        )
    )
    state = model.init_fast_state()
    x = torch.randn(1, 3, 8)
    before_other = state.k.w2.detach().clone()
    before_memory = state.memory.w2.detach().clone()

    _out, updated = model.forward_with_updates(x, state)

    assert not torch.allclose(before_other.unsqueeze(0), updated.k.w2)
    assert not torch.allclose(before_memory.unsqueeze(0), updated.memory.w2)
```

### File: `tests/test_selfmod_adaptive_q.py`

```python
import torch

from nested_learning.titan.self_modifying import SelfModifyingTitans, SelfModifyingTitansConfig


def test_selfmod_fixed_q_does_not_update_q_memory() -> None:
    torch.manual_seed(0)
    model = SelfModifyingTitans(SelfModifyingTitansConfig(dim=8, eta_scale=1.0, adaptive_q=False))
    x = torch.randn(1, 6, 8)
    state = model.init_fast_state()
    before = state.q.w2.detach().clone()
    _out, updated = model.forward_with_updates(x, state)
    assert torch.allclose(before.unsqueeze(0), updated.q.w2, atol=1e-6, rtol=1e-6)


def test_selfmod_adaptive_q_updates_q_memory() -> None:
    torch.manual_seed(0)
    model = SelfModifyingTitans(SelfModifyingTitansConfig(dim=8, eta_scale=1.0, adaptive_q=True))
    x = torch.randn(1, 6, 8)
    state = model.init_fast_state()
    before = state.q.w2.detach().clone()
    _out, updated = model.forward_with_updates(x, state)
    assert not torch.allclose(before.unsqueeze(0), updated.q.w2)

```

### File: `tests/test_selfmod_dgd_linear.py`

```python
import torch

from nested_learning.titan.self_modifying import (
    ResidualMLPMemoryState,
    SelfModifyingTitans,
    SelfModifyingTitansConfig,
)


def test_selfmod_linear_memory_l2_grad_matches_analytic() -> None:
    torch.manual_seed(0)
    model = SelfModifyingTitans(
        SelfModifyingTitansConfig(
            dim=4,
            eta_scale=1.0,
            objective="l2",
            stopgrad_vhat=True,
            use_rank1_precond=False,
            use_alpha=False,
            local_conv_window=None,
        )
    )
    w_skip = torch.randn(4, 4)
    frozen = ResidualMLPMemoryState(
        w1=torch.zeros_like(model.m_memory.w1.weight),
        w2=torch.zeros_like(model.m_memory.w2.weight),
        w_skip=w_skip.clone(),
    )
    k = torch.randn(3, 4)
    v = torch.randn(3, 4)
    g1, g2, gskip = model._memory_grads(frozen, k, v)
    assert gskip is not None

    with torch.no_grad():
        pred = k @ w_skip.t()
        vhat = v @ w_skip.t()
        diff = pred - vhat
        expected = 2.0 * torch.einsum("bi,bj->bij", diff, k).sum(dim=0)

    assert torch.allclose(gskip, expected, atol=1e-6, rtol=1e-6)
    assert torch.allclose(g1, torch.zeros_like(g1), atol=1e-6, rtol=1e-6)
    assert torch.allclose(g2, torch.zeros_like(g2), atol=1e-6, rtol=1e-6)

```

### File: `tests/test_selfmod_grad_flow.py`

```python
import torch
import torch.nn.functional as F

from nested_learning.levels import LevelSpec
from nested_learning.model import HOPEModel, ModelConfig


def test_hope_selfmod_forward_allows_outer_gradients() -> None:
    torch.manual_seed(0)
    cfg = ModelConfig(
        vocab_size=32,
        dim=16,
        num_layers=1,
        heads=4,
        titan_level=LevelSpec(name="titan", update_period=1),
        cms_levels=(),
        block_variant="hope_selfmod",
    )
    model = HOPEModel(cfg)
    tokens = torch.randint(0, cfg.vocab_size, (2, 6))
    logits = model(tokens)
    loss = F.cross_entropy(
        logits[:, :-1].reshape(-1, cfg.vocab_size),
        tokens[:, 1:].reshape(-1),
    )
    loss.backward()
    block = model.blocks[0]
    selfmod = getattr(block, "selfmod", None)
    assert selfmod is not None
    grad = selfmod.m_memory.w1.weight.grad
    assert grad is not None
    assert grad.abs().sum().item() > 0.0

```

### File: `tests/test_selfmod_local_conv.py`

```python
import torch

from nested_learning.titan.self_modifying import SelfModifyingTitans, SelfModifyingTitansConfig


def test_selfmod_local_conv_is_causal() -> None:
    torch.manual_seed(0)
    model = SelfModifyingTitans(SelfModifyingTitansConfig(dim=4, local_conv_window=4))
    assert model.local_conv is not None
    with torch.no_grad():
        model.local_conv.weight.fill_(1.0)
    x = torch.zeros(1, 6, 4)
    x[0, 4, 0] = 1.0
    y = model._apply_local_conv(x)
    assert torch.allclose(y[0, :4, 0], torch.zeros(4))
    assert y[0, 4, 0].item() != 0.0

```

### File: `tests/test_selfmod_online.py`

```python
import torch

from nested_learning.fast_state import build_block_fast_state
from nested_learning.hope.block import HOPESelfModBlock, HOPESelfModBlockConfig
from nested_learning.levels import LevelSpec


def test_selfmod_updates_on_update_pass_even_with_zero_teach_signal() -> None:
    cfg = HOPESelfModBlockConfig(
        dim=8,
        cms_levels=[LevelSpec(name="fast", update_period=1)],
        optimizer_configs={},
        selfmod_online_updates=True,
    )
    block = HOPESelfModBlock(cfg)
    fast_state = build_block_fast_state(
        titan_module=None,
        cms_blocks=block.cms.blocks,
        selfmod_module=block.selfmod,
        specs=cfg.cms_levels,
        optimizer_configs={},
        default_lr=cfg.self_mod_lr,
    )
    assert fast_state.selfmod_state is not None
    x = torch.randn(1, 4, 8)
    teach = torch.zeros_like(x)
    before = fast_state.selfmod_state.memory.w1.clone()
    _ = block(x, teach_signal=teach, fast_state=fast_state)
    after = fast_state.selfmod_state.memory.w1
    assert not torch.allclose(before, after)
```

### File: `tests/test_surprise_metric.py`

```python
import torch

from nested_learning.levels import LevelSpec
from nested_learning.model import HOPEModel, ModelConfig
from nested_learning.training import compute_teach_signal


def _cms_delta_l1(state, level_name: str) -> float:
    params = state.blocks[0].cms_params[level_name]
    return float(sum(delta.abs().sum().item() for delta in params.values()))


def _logit_entropy(logits: torch.Tensor) -> float:
    logits_detached = logits[:, :-1].detach().float()
    probs = torch.softmax(logits_detached, dim=-1)
    entropy = -(probs * torch.log(probs.clamp(min=1e-9))).sum(dim=-1).mean()
    return float(entropy.item())


def _next_token_loss(logits: torch.Tensor, tokens: torch.Tensor) -> float:
    loss = torch.nn.functional.cross_entropy(
        logits[:, :-1].reshape(-1, logits.size(-1)),
        tokens[:, 1:].reshape(-1),
    )
    return float(loss.detach().item())


def test_surprise_metric_loss_gates_updates_when_threshold_set() -> None:
    torch.manual_seed(0)
    cfg = ModelConfig(
        vocab_size=32,
        dim=16,
        num_layers=1,
        heads=2,
        titan_level=LevelSpec(name="titan", update_period=1),
        cms_levels=(LevelSpec(name="cms_fast", update_period=2),),
        block_variant="hope_attention",
        surprise_metric="loss",
        surprise_threshold=0.0,
    )
    model = HOPEModel(cfg).eval()
    state = model.init_fast_state()
    tokens = torch.randint(0, cfg.vocab_size, (1, 8))

    with torch.no_grad():
        logits = model(tokens, fast_state=state)
        teach = compute_teach_signal(model, logits, tokens)
        loss_value = _next_token_loss(logits, tokens)
        model.set_surprise_threshold(loss_value + 1.0)
        _ = model(
            tokens,
            teach_signal=teach,
            surprise_value=loss_value,
            fast_state=state,
        )
    assert _cms_delta_l1(state, "cms_fast") == 0.0

    with torch.no_grad():
        model.set_surprise_threshold(loss_value - 1.0)
        _ = model(
            tokens,
            teach_signal=teach,
            surprise_value=loss_value,
            fast_state=state,
        )
    assert _cms_delta_l1(state, "cms_fast") > 0.0


def test_surprise_metric_entropy_gates_updates_when_threshold_set() -> None:
    torch.manual_seed(0)
    cfg = ModelConfig(
        vocab_size=32,
        dim=16,
        num_layers=1,
        heads=2,
        titan_level=LevelSpec(name="titan", update_period=1),
        cms_levels=(LevelSpec(name="cms_fast", update_period=2),),
        block_variant="hope_attention",
        surprise_metric="logit_entropy",
        surprise_threshold=0.0,
    )
    model = HOPEModel(cfg).eval()
    state = model.init_fast_state()
    tokens = torch.randint(0, cfg.vocab_size, (1, 8))

    with torch.no_grad():
        logits = model(tokens, fast_state=state)
        teach = compute_teach_signal(model, logits, tokens)
        entropy_value = _logit_entropy(logits)
        model.set_surprise_threshold(entropy_value + 1.0)
        _ = model(
            tokens,
            teach_signal=teach,
            surprise_value=entropy_value,
            fast_state=state,
        )
    assert _cms_delta_l1(state, "cms_fast") == 0.0

    with torch.no_grad():
        model.set_surprise_threshold(entropy_value - 1.0)
        _ = model(
            tokens,
            teach_signal=teach,
            surprise_value=entropy_value,
            fast_state=state,
        )
    assert _cms_delta_l1(state, "cms_fast") > 0.0


def test_surprise_metric_requires_external_value_when_threshold_set() -> None:
    cfg = ModelConfig(
        vocab_size=32,
        dim=16,
        num_layers=1,
        heads=2,
        titan_level=LevelSpec(name="titan", update_period=1),
        cms_levels=(LevelSpec(name="cms_fast", update_period=2),),
        block_variant="hope_attention",
        surprise_metric="loss",
        surprise_threshold=0.1,
    )
    model = HOPEModel(cfg).eval()
    tokens = torch.randint(0, cfg.vocab_size, (1, 8))
    state = model.init_fast_state()
    with torch.no_grad():
        logits = model(tokens, fast_state=state)
        teach = compute_teach_signal(model, logits, tokens)
        try:
            _ = model(tokens, teach_signal=teach, fast_state=state)
        except ValueError as err:
            assert "requires passing surprise_value" in str(err)
        else:
            raise AssertionError("Expected ValueError when surprise_value is omitted")

```

### File: `tests/test_teach_signal.py`

```python
import torch
import torch.nn.functional as F

from nested_learning.levels import LevelSpec
from nested_learning.model import HOPEModel, ModelConfig
from nested_learning.titan.model import TitanOnlyModel, TitanOnlyModelConfig
from nested_learning.training import _compute_layer_teach_signals, compute_teach_signal


def _tiny_config() -> ModelConfig:
    titan = LevelSpec(name="titan", update_period=2, optimizer_key="titan_opt")
    cms = [
        LevelSpec(name="cms_fast", update_period=1, optimizer_key="cms_opt"),
        LevelSpec(name="cms_mid", update_period=4, optimizer_key="cms_opt"),
    ]
    return ModelConfig(
        vocab_size=32,
        dim=16,
        num_layers=2,
        heads=4,
        titan_level=titan,
        cms_levels=cms,
        optimizers=None,
        teach_scale=0.1,
    )


def _tiny_titan_config() -> TitanOnlyModelConfig:
    titan = LevelSpec(name="titan", update_period=2, optimizer_key="titan_opt")
    return TitanOnlyModelConfig(
        vocab_size=32,
        dim=16,
        num_layers=2,
        heads=4,
        titan_level=titan,
        optimizers=None,
        teach_scale=0.1,
    )


def test_teach_signal_matches_gradient() -> None:
    torch.manual_seed(0)
    cfg = _tiny_config()
    model = HOPEModel(cfg)
    tokens = torch.randint(0, cfg.vocab_size, (2, 6))

    hidden_cache: dict[str, torch.Tensor] = {}

    def hook(_, __, output: torch.Tensor) -> None:
        output.retain_grad()
        hidden_cache["hidden"] = output

    handle = model.norm.register_forward_hook(hook)
    logits = model(tokens)
    teach_signal = compute_teach_signal(model, logits, tokens)
    loss = F.cross_entropy(
        logits[:, :-1].reshape(-1, cfg.vocab_size),
        tokens[:, 1:].reshape(-1),
    )
    loss.backward()
    handle.remove()

    hidden = hidden_cache["hidden"]
    assert hidden.grad is not None
    grad = hidden.grad
    assert torch.allclose(teach_signal, grad, atol=1e-5, rtol=1e-4)


def test_teach_signal_matches_gradient_titan() -> None:
    torch.manual_seed(0)
    cfg = _tiny_titan_config()
    model = TitanOnlyModel(cfg)
    tokens = torch.randint(0, cfg.vocab_size, (2, 6))

    hidden_cache: dict[str, torch.Tensor] = {}

    def hook(_, __, output: torch.Tensor) -> None:
        output.retain_grad()
        hidden_cache["hidden"] = output

    handle = model.norm.register_forward_hook(hook)
    logits = model(tokens)
    teach_signal = compute_teach_signal(model, logits, tokens)
    loss = F.cross_entropy(
        logits[:, :-1].reshape(-1, cfg.vocab_size),
        tokens[:, 1:].reshape(-1),
    )
    loss.backward()
    handle.remove()

    hidden = hidden_cache["hidden"]
    assert hidden.grad is not None
    grad = hidden.grad
    assert torch.allclose(teach_signal, grad, atol=1e-5, rtol=1e-4)


def test_per_layer_teach_signal_shapes() -> None:
    torch.manual_seed(0)
    cfg = _tiny_config()
    model = HOPEModel(cfg)
    tokens = torch.randint(0, cfg.vocab_size, (2, 6))
    logits, _pre, block_outputs = model.forward_with_block_outputs(tokens)
    loss = F.cross_entropy(
        logits[:, :-1].reshape(-1, cfg.vocab_size),
        tokens[:, 1:].reshape(-1),
    )
    teach_signals = _compute_layer_teach_signals(loss, block_outputs)
    assert len(teach_signals) == cfg.num_layers
    for signal, output in zip(teach_signals, block_outputs):
        assert signal.shape == output.shape


def test_per_layer_teach_signal_matches_autograd_grads() -> None:
    torch.manual_seed(0)
    cfg = _tiny_config()
    model = HOPEModel(cfg)
    tokens = torch.randint(0, cfg.vocab_size, (2, 6))
    logits, _pre, block_outputs = model.forward_with_block_outputs(tokens)
    loss = F.cross_entropy(
        logits[:, :-1].reshape(-1, cfg.vocab_size),
        tokens[:, 1:].reshape(-1),
    )
    expected = torch.autograd.grad(loss, block_outputs, retain_graph=True, allow_unused=False)
    teach_signals = _compute_layer_teach_signals(loss, block_outputs)
    for exp, actual in zip(expected, teach_signals):
        assert torch.allclose(actual, exp.detach(), atol=1e-6, rtol=1e-6)


def test_teach_signal_matches_gradient_with_ignore_index() -> None:
    torch.manual_seed(0)
    cfg = _tiny_config()
    model = HOPEModel(cfg)
    tokens = torch.randint(1, cfg.vocab_size, (2, 6))
    tokens[0, 2] = 0
    tokens[1, 4] = 0

    hidden_cache: dict[str, torch.Tensor] = {}

    def hook(_, __, output: torch.Tensor) -> None:
        output.retain_grad()
        hidden_cache["hidden"] = output

    handle = model.norm.register_forward_hook(hook)
    logits = model(tokens)
    teach_signal = compute_teach_signal(model, logits, tokens, ignore_index=0)
    loss = F.cross_entropy(
        logits[:, :-1].reshape(-1, cfg.vocab_size),
        tokens[:, 1:].reshape(-1),
        ignore_index=0,
    )
    loss.backward()
    handle.remove()

    hidden = hidden_cache["hidden"]
    assert hidden.grad is not None
    grad = hidden.grad
    assert torch.allclose(teach_signal, grad, atol=1e-5, rtol=1e-4)

    ignored = tokens[:, 1:] == 0
    masked = teach_signal[:, :-1][ignored]
    assert torch.allclose(masked, torch.zeros_like(masked))
```

### File: `tests/test_variants.py`

```python
import torch

from nested_learning.hope.block import HOPEAttentionBlock, HOPEBlock, HOPESelfModBlock
from nested_learning.levels import LevelSpec
from nested_learning.model import HOPEModel, ModelConfig
from nested_learning.titan.memory import TitanMemory
from nested_learning.transformer import TransformerBlock


def _base_cfg(*, block_variant: str) -> ModelConfig:
    titan = LevelSpec(name="titan", update_period=1, optimizer_key="titan_opt")
    cms = [LevelSpec(name="cms_fast", update_period=1, optimizer_key="cms_opt")]
    return ModelConfig(
        vocab_size=32,
        dim=16,
        num_layers=1,
        heads=4,
        titan_level=titan,
        cms_levels=cms,
        block_variant=block_variant,
        optimizers=None,
    )


def test_hope_hybrid_variant_contains_titan_memory() -> None:
    model = HOPEModel(_base_cfg(block_variant="hope_hybrid"))
    block = model.blocks[0]
    assert isinstance(block, HOPEBlock)
    assert isinstance(block.titan_memory, TitanMemory)


def test_hope_attention_variant_excludes_titan_memory() -> None:
    model = HOPEModel(_base_cfg(block_variant="hope_attention"))
    block = model.blocks[0]
    assert isinstance(block, HOPEAttentionBlock)
    assert not hasattr(block, "titan_memory")

    tokens = torch.randint(0, model.config.vocab_size, (2, 5))
    logits = model(tokens)
    assert logits.shape == (2, 5, model.config.vocab_size)


def test_hope_selfmod_variant_excludes_titan_memory() -> None:
    model = HOPEModel(_base_cfg(block_variant="hope_selfmod"))
    block = model.blocks[0]
    assert isinstance(block, HOPESelfModBlock)
    assert not hasattr(block, "titan_memory")
    assert hasattr(block, "selfmod")

    fast_state = model.init_fast_state()
    tokens = torch.randint(0, model.config.vocab_size, (1, 5))
    logits = model(tokens, fast_state=fast_state)
    assert logits.shape == (1, 5, model.config.vocab_size)


def test_transformer_variant_runs_with_and_without_fast_state() -> None:
    model = HOPEModel(_base_cfg(block_variant="transformer"))
    block = model.blocks[0]
    assert isinstance(block, TransformerBlock)

    tokens = torch.randint(0, model.config.vocab_size, (2, 5))
    logits = model(tokens)
    assert logits.shape == (2, 5, model.config.vocab_size)

    fast_state = model.init_fast_state()
    logits_fast = model(tokens, fast_state=fast_state)
    assert logits_fast.shape == (2, 5, model.config.vocab_size)
```

### File: `train.py`

```python
from __future__ import annotations

import hydra
from omegaconf import DictConfig

from nested_learning.device import resolve_device
from nested_learning.training import run_training_loop, unwrap_config


@hydra.main(config_path="configs", config_name="pilot", version_base=None)
def main(cfg: DictConfig) -> None:
    cfg = unwrap_config(cfg)
    device = resolve_device(cfg.train.device)
    run_training_loop(cfg, device=device, distributed=False)


if __name__ == "__main__":
    main()
```

### File: `train_deepspeed.py`

```python
from __future__ import annotations

import json
import os
from pathlib import Path

import hydra
import torch
import torch.distributed as dist
from omegaconf import DictConfig

from nested_learning.logging_utils import NullLogger, init_logger
from nested_learning.training import (
    DistributedContext,
    _seed_everything,
    build_dataloader,
    build_model_from_cfg,
    compute_teach_signal,
    unwrap_config,
)

try:
    import deepspeed
except ImportError as exc:  # pragma: no cover - optional dependency
    raise RuntimeError(
        "DeepSpeed is not installed. Install it in this environment to use train_deepspeed.py."
    ) from exc


def setup_distributed() -> DistributedContext:
    deepspeed.init_distributed()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")
    return DistributedContext(rank=rank, world_size=world_size, device=device)


def load_ds_config(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


@hydra.main(config_path="configs", config_name="hope/target", version_base=None)
def main(cfg: DictConfig) -> None:
    cfg = unwrap_config(cfg)
    dist_ctx = setup_distributed()
    train_seed = cfg.train.get("seed")
    deterministic = cfg.train.get("deterministic", False)
    if train_seed is not None:
        _seed_everything(int(train_seed), deterministic=bool(deterministic))
    model = build_model_from_cfg(cfg.model)
    ds_config = load_ds_config(cfg.deepspeed.config)
    engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=[p for p in model.parameters() if p.requires_grad],
        config=ds_config,
    )

    train_seed = cfg.train.get("seed")
    loader_seed = None if train_seed is None else int(train_seed) + dist_ctx.rank
    dataloader, sampler = build_dataloader(
        cfg.data,
        distributed=True,
        dist_ctx=dist_ctx,
        seed=loader_seed,
    )
    logger = (
        init_logger(getattr(cfg, "logging", None), cfg) if engine.global_rank == 0 else NullLogger()
    )
    steps = cfg.train.steps
    log_interval = cfg.train.get("log_interval", 10)
    checkpoint_cfg = cfg.train.get("checkpoint", {})
    ckpt_dir = Path(checkpoint_cfg.get("dir", "checkpoints/deepspeed"))

    if checkpoint_cfg.get("resume_tag"):
        tag = checkpoint_cfg["resume_tag"]
        engine.load_checkpoint(str(ckpt_dir), tag=tag)
        if engine.global_rank == 0:
            print(f"[DeepSpeed] Resumed from {ckpt_dir} tag={tag}")

    step_iter = iter(dataloader)
    epoch = 0
    for step in range(steps):
        if sampler is not None and step % len(dataloader) == 0:
            sampler.set_epoch(epoch)
            epoch += 1
        try:
            batch = next(step_iter)
        except StopIteration:
            step_iter = iter(dataloader)
            batch = next(step_iter)
        tokens = batch.to(dist_ctx.device)
        logits = engine(tokens)
        loss = torch.nn.functional.cross_entropy(
            logits[:, :-1].reshape(-1, logits.size(-1)), tokens[:, 1:].reshape(-1)
        )
        engine.backward(loss)
        engine.step()
        with torch.no_grad():
            teach_signal = compute_teach_signal(engine.module, logits, tokens)
            engine.module(tokens, teach_signal=teach_signal)
        if step % log_interval == 0 and engine.global_rank == 0:
            ppl = torch.exp(loss.detach()).item()
            logger.log({"loss": loss.item(), "ppl": ppl}, step=step)
            print(f"[DeepSpeed] step={step} loss={loss.item():.4f} ppl={ppl:.2f}")
        if (
            checkpoint_cfg.get("enable", False)
            and step % checkpoint_cfg.get("save_interval", 100) == 0
            and engine.global_rank == 0
        ):
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            engine.save_checkpoint(str(ckpt_dir), tag=f"step_{step:06d}")

    logger.finish()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
```

### File: `train_dist.py`

```python
from __future__ import annotations

import os

import hydra
import torch
import torch.distributed as dist
from omegaconf import DictConfig

from nested_learning.training import DistributedContext, run_training_loop, unwrap_config


def setup_distributed(backend: str | None = None) -> DistributedContext:
    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = dist.get_world_size()
    if backend == "nccl":
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    return DistributedContext(rank=dist.get_rank(), world_size=world_size, device=device)


@hydra.main(config_path="configs", config_name="hope/mid", version_base=None)
def main(cfg: DictConfig) -> None:
    cfg = unwrap_config(cfg)
    dist_ctx = setup_distributed()
    run_training_loop(cfg, device=dist_ctx.device, distributed=True, dist_ctx=dist_ctx)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
```

### File: `train_fsdp.py`

```python
from __future__ import annotations

import os
from functools import partial
from pathlib import Path

import hydra
import torch
import torch.distributed as dist
import torch.nn as nn
from omegaconf import DictConfig
from torch.distributed.fsdp import (
    CPUOffload,
    FullStateDictConfig,
    StateDictType,
)
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

from nested_learning.logging_utils import NullLogger, init_logger
from nested_learning.training import (
    DistributedContext,
    _build_optimizer,
    _make_autocast_factory,
    _maybe_compile_model,
    _seed_everything,
    build_dataloader,
    build_model_from_cfg,
    compute_teach_signal,
    unwrap_config,
    verify_checkpoint_integrity,
    write_checkpoint_metadata,
)


def setup_distributed() -> DistributedContext:
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{local_rank}")
    return DistributedContext(rank=rank, world_size=world_size, device=device)


def build_fsdp_model(cfg: DictConfig, device: torch.device) -> tuple[FSDP, torch.nn.Module]:
    base_model = build_model_from_cfg(cfg.model).to(device)
    base_model = _maybe_compile_model(base_model, cfg.train.get("compile"))
    fsdp_cfg = cfg.train.get("fsdp", {})
    min_params = fsdp_cfg.get("auto_wrap_min_params", 2_000_000)
    # Avoid wrapping tied-weight modules (embed/lm_head) into separate FSDP instances.
    # Parameter sharing across FSDP wrappers can produce shape mismatches at runtime.
    exclude = {nn.Embedding, nn.Linear}
    auto_wrap_policy = partial(
        size_based_auto_wrap_policy,
        min_num_params=min_params,
        exclude_wrap_modules=exclude,
    )
    cpu_offload = CPUOffload(offload_params=fsdp_cfg.get("cpu_offload", False))
    model = FSDP(
        base_model,
        device_id=device.index,
        auto_wrap_policy=auto_wrap_policy,
        cpu_offload=cpu_offload,
        use_orig_params=True,  # Required for custom inner optimizers / in-place updates
    )
    return model, base_model


def unwrap_model(module: torch.nn.Module) -> torch.nn.Module:
    if hasattr(module, "_fsdp_wrapped_module"):
        return module._fsdp_wrapped_module  # type: ignore[attr-defined]
    if hasattr(module, "module"):
        return module.module  # type: ignore[attr-defined]
    return module


def save_checkpoint(
    cfg: DictConfig,
    model: FSDP,
    optimizer: torch.optim.Optimizer,
    step: int,
    rank: int,
    step_offset: int = 0,
) -> None:
    ckpt_cfg = cfg.train.get("checkpoint")
    if not ckpt_cfg or not ckpt_cfg.get("enable", False):
        return
    save_interval = ckpt_cfg.get("save_interval", 1000)
    save_last = ckpt_cfg.get("save_last", True)
    total_steps = cfg.train.get("steps", step + 1)
    next_step = step + 1
    should_save = (next_step % save_interval == 0) or (save_last and next_step >= total_steps)
    if not should_save:
        return
    ckpt_dir = Path(ckpt_cfg.get("dir", "checkpoints/fsdp"))
    global_step = next_step + int(step_offset)
    ckpt_path = ckpt_dir / f"step_{global_step:06d}.pt"
    if rank == 0:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    full_state_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_state_cfg):
        model_state = model.state_dict()
    if rank != 0:
        return
    state = {"model": model_state, "optimizer": optimizer.state_dict(), "step": global_step}
    torch.save(state, ckpt_path)
    write_checkpoint_metadata(cfg, ckpt_path, global_step)


def maybe_resume(cfg: DictConfig, model: FSDP, optimizer: torch.optim.Optimizer, rank: int) -> int:
    ckpt_cfg = cfg.train.get("checkpoint")
    if not ckpt_cfg:
        return 0
    resume_path = ckpt_cfg.get("resume_path")
    if not resume_path:
        return 0
    if not Path(resume_path).exists():
        raise FileNotFoundError(f"Resume checkpoint {resume_path} not found")
    map_location = "cpu"
    verify_checkpoint_integrity(Path(resume_path))
    state = torch.load(resume_path, map_location=map_location)
    full_state_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=False)
    with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, full_state_cfg):
        model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    if rank == 0:
        print(f"[FSDP] Resumed from {resume_path} at step {state.get('step', 0)}")
    return state.get("step", 0)


@hydra.main(config_path="configs", config_name="hope/mid", version_base=None)
def main(cfg: DictConfig) -> None:
    cfg = unwrap_config(cfg)
    dist_ctx = setup_distributed()
    train_seed = cfg.train.get("seed")
    deterministic = cfg.train.get("deterministic", False)
    if train_seed is not None:
        _seed_everything(int(train_seed), deterministic=bool(deterministic))
    model, _ = build_fsdp_model(cfg, dist_ctx.device)
    train_seed = cfg.train.get("seed")
    loader_seed = None if train_seed is None else int(train_seed) + dist_ctx.rank
    dataloader, sampler = build_dataloader(
        cfg.data,
        distributed=True,
        dist_ctx=dist_ctx,
        seed=loader_seed,
    )
    optimizer = _build_optimizer(model, cfg, device=dist_ctx.device)
    start_step = maybe_resume(cfg, model, optimizer, dist_ctx.rank)
    logger = init_logger(getattr(cfg, "logging", None), cfg) if dist_ctx.rank == 0 else NullLogger()
    autocast_factory = _make_autocast_factory(dist_ctx.device, cfg.train.get("mixed_precision"))

    steps = cfg.train.steps
    log_interval = cfg.train.get("log_interval", 10)
    step_iter = iter(dataloader)
    epoch = 0
    for step in range(start_step, steps):
        if sampler is not None and step % len(dataloader) == 0:
            sampler.set_epoch(epoch)
            epoch += 1
        try:
            batch = next(step_iter)
        except StopIteration:
            step_iter = iter(dataloader)
            batch = next(step_iter)
        tokens = batch.to(dist_ctx.device)
        with autocast_factory():
            logits = model(tokens)
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)), tokens[:, 1:].reshape(-1)
            )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        with torch.no_grad():
            # FSDP shards parameters at rest; compute_teach_signal needs access to the full
            # (tied) LM head weight matrix. Summon the root parameters (embed/lm_head) to
            # materialize the full 2D weight before computing the teach signal.
            with FSDP.summon_full_params(model, recurse=False):
                inner_full = unwrap_model(model)
                teach_signal = compute_teach_signal(inner_full, logits, tokens)
            _ = model(tokens, teach_signal=teach_signal)
        if step % log_interval == 0 and dist_ctx.rank == 0:
            ppl = torch.exp(loss.detach()).item()
            logger.log({"loss": loss.item(), "ppl": ppl}, step=step)
            print(f"[fsdp] step={step} loss={loss.item():.4f} ppl={ppl:.2f}")
        save_checkpoint(cfg, model, optimizer, step, dist_ctx.rank)

    logger.finish()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
```
