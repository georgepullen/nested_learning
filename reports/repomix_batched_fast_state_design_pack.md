# Files

## File: reports/batching_architecture_problem_evaluation.md
```markdown
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
```

## File: configs/pilot_paper_faithful.yaml
```yaml
defaults:
  - /pilot
  - _self_
model:
  surprise_threshold: null
  cms_flush_partial_at_end: true
  self_mod_adaptive_q: false
  self_mod_local_conv_window: 4
data:
  batch_size: 1
train:
  use_fast_state: true
  fail_if_paper_faithful_disabled: true
optim:
  param_policy: all
logging:
  run_name: pilot-paper-faithful
  path: logs/pilot_paper_faithful_metrics.json
```

## File: tests/test_fast_state_batch_semantics.py
```python
def test_fast_state_batch_semantics_raises_when_strict() -> None
⋮----
cfg = OmegaConf.create(
⋮----
def test_fast_state_batch_semantics_allows_batch1() -> None
```

## File: tests/test_self_modifying_titans.py
```python
def test_self_modifying_titans_forward_shape() -> None
⋮----
model = SelfModifyingTitans(SelfModifyingTitansConfig(dim=8))
x = torch.randn(2, 5, 8)
out = model(x)
⋮----
def test_self_modifying_titans_updates_fast_state() -> None
⋮----
model = SelfModifyingTitans(SelfModifyingTitansConfig(dim=8, eta_scale=1.0))
x = torch.randn(1, 6, 8)
state = model.init_fast_state()
before = state.memory.w2.detach().clone()
⋮----
def test_self_modifying_titans_supports_batch_fast_state_updates() -> None
⋮----
x = torch.randn(2, 6, 8)
⋮----
def test_self_modifying_titans_chunked_outputs_match_no_update_with_single_chunk() -> None
⋮----
seq_len = 6
model = SelfModifyingTitans(
x = torch.randn(1, seq_len, 8)
⋮----
out_no_update = model.forward_with_state(x, state)
⋮----
def test_self_modifying_titans_flushes_partial_chunks_for_memory_updates() -> None
⋮----
x = torch.randn(1, 3, 8)
before_other = state.k.w2.detach().clone()
before_memory = state.memory.w2.detach().clone()
```

## File: src/nested_learning/fast_state.py
```python
ParamDict = Dict[str, torch.Tensor]
def init_module_deltas(module: nn.Module) -> ParamDict
⋮----
@dataclass
class BlockFastState
⋮----
titan_params: ParamDict | None
cms_params: Dict[str, ParamDict]
level_manager: LevelOptimizerManager
selfmod_state: SelfModifyingTitansState | None = None
⋮----
titan_params = None
⋮----
titan_params = init_module_deltas(titan_module)
cms_params = {name: init_module_deltas(block) for name, block in cms_blocks.items()}
level_cfg = LevelConfig(specs=specs, optimizer_configs=optimizer_configs, default_lr=default_lr)
level_manager = LevelOptimizerManager(level_cfg)
selfmod_state = None
⋮----
init_fn = getattr(selfmod_module, "init_fast_state", None)
⋮----
selfmod_state = cast(SelfModifyingTitansState, init_fn())
⋮----
@dataclass
class ModelFastState
⋮----
blocks: list[BlockFastState]
```

## File: src/nested_learning/functional.py
```python
ParamDict = Dict[str, torch.Tensor]
def params_with_deltas(module: nn.Module, deltas: ParamDict) -> ParamDict
⋮----
params: ParamDict = {}
missing: list[str] = []
⋮----
delta = deltas.get(name)
⋮----
def module_buffers(module: nn.Module) -> ParamDict
⋮----
buffers = module_buffers(module)
⋮----
def require_grad_params(params: Mapping[str, torch.Tensor]) -> ParamDict
def grads_to_dict(params: ParamDict, grads: Tuple[torch.Tensor | None, ...]) -> ParamDict
⋮----
out: ParamDict = {}
```

## File: src/nested_learning/titan/self_modifying.py
```python
@dataclass(frozen=True)
class SelfModifyingTitansConfig
⋮----
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
def __post_init__(self) -> None
⋮----
@dataclass
class ResidualMLPMemoryState
⋮----
w1: torch.Tensor
w2: torch.Tensor
w_skip: torch.Tensor | None = None
m_w1: torch.Tensor | None = None
m_w2: torch.Tensor | None = None
m_w_skip: torch.Tensor | None = None
def clone(self) -> "ResidualMLPMemoryState"
⋮----
@dataclass
class SelfModifyingTitansState
⋮----
k: ResidualMLPMemoryState
v: ResidualMLPMemoryState
q: ResidualMLPMemoryState
eta: ResidualMLPMemoryState
alpha: ResidualMLPMemoryState
memory: ResidualMLPMemoryState
def clone(self) -> "SelfModifyingTitansState"
class ResidualMLPMemory(nn.Module)
⋮----
def forward(self, x: torch.Tensor) -> torch.Tensor
⋮----
hidden = self.activation(self.w2(x))
out = self.w1(hidden)
⋮----
class SelfModifyingTitans(nn.Module)
⋮----
def __init__(self, config: SelfModifyingTitansConfig)
⋮----
dim = config.dim
hidden = dim
act = F.gelu
⋮----
window = int(config.local_conv_window)
⋮----
def init_fast_state(self) -> SelfModifyingTitansState
⋮----
state = self.init_fast_state()
⋮----
x = self._apply_local_conv(x)
q = self.m_q(x) if self.config.adaptive_q else self.w_q(x)
⋮----
q = F.normalize(q, dim=-1, eps=self.config.eps)
⋮----
state = self._ensure_batched_state(state, batch)
⋮----
q = (
⋮----
other_chunk = int(
memory_chunk_cfg = self.config.chunk_size_memory
⋮----
memory_chunk_cfg = self.config.chunk_size_other
memory_chunk = int(memory_chunk_cfg if chunk_size_memory is None else chunk_size_memory)
⋮----
outputs: list[torch.Tensor] = []
other_k: list[torch.Tensor] = []
other_v: list[torch.Tensor] = []
other_eta: list[torch.Tensor] = []
other_alpha: list[torch.Tensor] = []
memory_k: list[torch.Tensor] = []
memory_v: list[torch.Tensor] = []
memory_eta: list[torch.Tensor] = []
memory_alpha: list[torch.Tensor] = []
def _next_boundary(idx: int, *, chunk_size: int) -> int
⋮----
idx = 0
⋮----
next_other = _next_boundary(idx, chunk_size=other_chunk)
next_memory = _next_boundary(idx, chunk_size=memory_chunk)
end = min(next_other, next_memory, seq_len)
x_chunk = x[:, idx:end, :]
k_chunk = self._memory_forward(x_chunk, state.k)
v_chunk = self._memory_forward(x_chunk, state.v)
q_chunk = (
⋮----
k_chunk = F.normalize(k_chunk, dim=-1, eps=self.config.eps)
q_chunk = F.normalize(q_chunk, dim=-1, eps=self.config.eps)
eta_chunk = self._memory_forward(x_chunk, state.eta).squeeze(-1)
eta_chunk = F.softplus(eta_chunk) * self.config.eta_scale
⋮----
alpha_chunk = self._memory_forward(x_chunk, state.alpha).squeeze(-1)
alpha_chunk = torch.sigmoid(alpha_chunk)
⋮----
alpha_chunk = torch.ones_like(eta_chunk)
o_chunk = self._memory_forward(q_chunk, state.memory)
⋮----
idx = end
⋮----
other_memories: tuple[str, ...] = ("k", "v", "eta")
⋮----
other_memories = (*other_memories, "q")
⋮----
other_memories = (*other_memories, "alpha")
⋮----
other_memories = ("k", "v", "eta")
⋮----
def _apply_local_conv(self, x: torch.Tensor) -> torch.Tensor
⋮----
kernel = int(self.local_conv.kernel_size[0])
x_t = x.transpose(1, 2)
x_t = F.pad(x_t, (kernel - 1, 0))
x_t = self.local_conv(x_t)
⋮----
def _load_state_mean_(self, state: SelfModifyingTitansState) -> None
⋮----
def _mean_weight(weight: torch.Tensor) -> torch.Tensor
def _copy(module: ResidualMLPMemory, mem: ResidualMLPMemoryState) -> None
⋮----
k_seq = torch.stack([item[0] for item in buffer], dim=1)
v_seq = torch.stack([item[1] for item in buffer], dim=1)
eta_seq = torch.stack([item[2] for item in buffer], dim=1)
alpha_seq = torch.stack([item[3] for item in buffer], dim=1)
⋮----
steps = k_seq.size(1)
dim = self.config.dim
eye = (
boundary: dict[str, ResidualMLPMemoryState] = {
grads = {name: self._memory_grads_chunk(boundary[name], k_seq, v_seq) for name in memories}
⋮----
k_t = k_seq[:, t, :]
eta_t = eta_seq[:, t]
alpha_t = alpha_seq[:, t]
kk = torch.einsum("bi,bj->bij", k_t, k_t)
precond = alpha_t[:, None, None] * eye - eta_t[:, None, None] * kk
⋮----
fast = getattr(state, name)
⋮----
w1 = frozen.w1.detach().requires_grad_(True)
w2 = frozen.w2.detach().requires_grad_(True)
w_skip = None
⋮----
w_skip = frozen.w_skip.detach().requires_grad_(True)
pred = self._memory_forward(k_t, ResidualMLPMemoryState(w1=w1, w2=w2, w_skip=w_skip))
vhat = self._memory_forward(v_t, ResidualMLPMemoryState(w1=w1, w2=w2, w_skip=w_skip))
⋮----
vhat = vhat.detach()
⋮----
loss = -(pred * vhat).sum(dim=-1)
⋮----
loss = (pred - vhat).pow(2).sum(dim=-1)
loss_scalar = loss.sum()
grads = torch.autograd.grad(
⋮----
w1 = frozen.w1.detach()
w2 = frozen.w2.detach()
w_skip = None if frozen.w_skip is None else frozen.w_skip.detach()
k_tokens = k_seq.transpose(0, 1)
v_tokens = v_seq.transpose(0, 1)
⋮----
mem = ResidualMLPMemoryState(w1=w1_t, w2=w2_t)
pred = self._memory_forward(k_t, mem)
vhat = self._memory_forward(v_t, mem)
⋮----
grad_fn = grad(loss_fn_noskip, argnums=(0, 1))
⋮----
mem = ResidualMLPMemoryState(w1=w1_t, w2=w2_t, w_skip=w_skip_t)
⋮----
grad_fn = grad(loss_fn_skip, argnums=(0, 1, 2))
⋮----
g1 = self._apply_momentum(fast, "m_w1", g1)
g2 = self._apply_momentum(fast, "m_w2", g2)
⋮----
gskip = self._apply_momentum(fast, "m_w_skip", gskip)
⋮----
beta = float(self.config.momentum)
⋮----
buf = getattr(fast, attr_name)
⋮----
buf = torch.zeros_like(grad)
buf = beta * buf + grad
⋮----
def _init_memory_state(self, module: ResidualMLPMemory) -> ResidualMLPMemoryState
⋮----
skip = None if module.w_skip is None else module.w_skip.weight.detach().clone()
⋮----
def _expand(t: torch.Tensor) -> torch.Tensor
def _expand_opt(t: torch.Tensor | None) -> torch.Tensor | None
⋮----
w2 = mem.w2
w1 = mem.w1
w_skip = mem.w_skip
⋮----
w2 = self._straight_through_meta(mem.w2, meta.w2.weight)
w1 = self._straight_through_meta(mem.w1, meta.w1.weight)
⋮----
w_skip = self._straight_through_meta(mem.w_skip, meta.w_skip.weight)
⋮----
x_seq = x.unsqueeze(1)
squeeze = True
⋮----
x_seq = x
squeeze = False
w2_t = w2.transpose(-1, -2)
hidden = torch.matmul(x_seq, w2_t)
hidden = F.gelu(hidden)
w1_t = w1.transpose(-1, -2)
out = torch.matmul(hidden, w1_t)
⋮----
w_skip_t = w_skip.transpose(-1, -2)
out = out + torch.matmul(x_seq, w_skip_t)
⋮----
out = out + x_seq
⋮----
@staticmethod
    def _straight_through_meta(fast: torch.Tensor, meta: torch.Tensor) -> torch.Tensor
⋮----
expanded = meta
⋮----
expanded = expanded.unsqueeze(0)
```

## File: src/nested_learning/optim/manager.py
```python
@dataclass
class LevelConfig
⋮----
specs: Sequence[LevelSpec]
optimizer_configs: Dict[str, dict]
default_lr: float
class LevelOptimizerManager
⋮----
def __init__(self, config: LevelConfig)
⋮----
key = spec.optimizer_key or "default"
optim_cfg = config.optimizer_configs.get(key, {"type": "deep_momentum", "params": {}})
lr = optim_cfg.get("lr", config.default_lr)
params_cfg = optim_cfg.get("params", {})
optimizer = build_optimizer(
⋮----
def should_update(self, level: str) -> bool
⋮----
named_params: Tuple[Tuple[str, torch.nn.Parameter], ...] = tuple(
⋮----
params = tuple(param for _, param in named_params)
grads = torch.autograd.grad(loss, params, retain_graph=False, allow_unused=True)
grads_dict: Dict[str, torch.Tensor] = {}
⋮----
optimizer = self.optimizers[level]
lr = self.learning_rates[level]
total_norm = 0.0
⋮----
grad = grads.get(name)
⋮----
update = optimizer(grad, context=context, param_key=name)
⋮----
metrics = getattr(optimizer, "last_metrics", None)
⋮----
def tick(self) -> None
def pop_last_metrics(self, level: str) -> Dict[str, float]
⋮----
updated: Dict[str, torch.Tensor] = {}
```

## File: src/nested_learning/hope/block.py
```python
target = (prediction.detach() - delta_target).detach()
diff_sq = (prediction - target).pow(2)
masked = diff_sq * mask_f
⋮----
def _min_update_period(levels: Sequence[LevelSpec]) -> int
⋮----
periods = [int(spec.update_period) for spec in levels if int(spec.update_period) > 0]
⋮----
@dataclass
class _CmsBuffer
⋮----
inputs: list[torch.Tensor]
teach: list[torch.Tensor]
active: list[torch.Tensor]
count: int = 0
⋮----
result_inputs: list[torch.Tensor] = []
result_teach: list[torch.Tensor] = []
result_active: list[torch.Tensor] = []
remaining = count
⋮----
first = buffer.inputs[0]
chunk_len = first.size(1)
take = min(remaining, chunk_len)
src_inputs = buffer.inputs[0]
src_teach = buffer.teach[0]
src_active = buffer.active[0]
⋮----
@dataclass
class HOPEBlockConfig
⋮----
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
⋮----
@dataclass
class HOPEAttentionBlockConfig
class HOPEAttentionBlock(nn.Module)
⋮----
def __init__(self, config: HOPEAttentionBlockConfig)
⋮----
level_config = LevelConfig(
⋮----
attn_out = self.attn(x)
⋮----
cms_out = self._cms_forward_online(attn_out, teach_signal, surprise_value)
⋮----
cms_result = self.cms(attn_out, return_intermediates=True)
⋮----
cms_out = self._cms_forward_online_fast(
⋮----
def set_surprise_threshold(self, threshold: float | None) -> None
def set_surprise_metric(self, metric: str) -> None
def set_allowed_levels(self, allowed: Set[str] | None) -> None
def pop_update_stats(self) -> Dict[str, Dict[str, float]]
⋮----
stats = self.last_update_stats
⋮----
current = x
inputs: dict[str, torch.Tensor] = {}
⋮----
level_name = spec.name
⋮----
params = fast_state.cms_params[level_name]
current = call_with_deltas(self.cms.blocks[level_name], params, current)
⋮----
seq_len = x.shape[1]
base_chunk = _min_update_period(self.config.cms_levels)
active_mask = teach_signal.detach().abs().sum(dim=-1) > 0
outputs: list[torch.Tensor] = []
stats: dict[str, Dict[str, float]] = {}
buffers: dict[str, _CmsBuffer] = {}
⋮----
end = min(start + base_chunk, seq_len)
chunk_in = x[:, start:end, :]
chunk_teach = teach_signal[:, start:end, :]
chunk_active = active_mask[:, start:end]
current = chunk_in
level_inputs: dict[str, torch.Tensor] = {}
⋮----
current = self.cms.blocks[level_name](current)
⋮----
buffer = buffers[level_name]
⋮----
update_period = int(spec.update_period)
⋮----
magnitude = self._update_cms_chunk(
⋮----
remaining = int(buffer.count)
⋮----
magnitude = self._update_cms_chunk_fast(
⋮----
teach = teach_signal.detach()
active_mask = teach.abs().sum(dim=-1) > 0
⋮----
inputs = cms_inputs[level_name]
seq_len = inputs.shape[1]
chunk_size = int(spec.update_period)
⋮----
total_norm = 0.0
update_events = 0
token_events = 0
⋮----
end = min(start + chunk_size, seq_len)
chunk_len = end - start
chunk_inputs = inputs[:, start:end, :].detach()
chunk_teach = teach[:, start:end, :]
⋮----
stats_payload: Dict[str, float] = {
⋮----
def _is_level_allowed(self, level_name: str) -> bool
def _passes_surprise(self, surprise_value: float | None) -> bool
def _record_gate(self, level_name: str, *, hit: bool) -> None
⋮----
stats_key = f"gate.{level_name}"
⋮----
mask_f = chunk_active.unsqueeze(-1).float()
⋮----
prediction = self.cms.blocks[level_name](chunk_inputs)
loss = _chunk_loss(
context_vec = chunk_inputs.mean(dim=(0, 1))
magnitude = self.level_manager.optimize(
⋮----
base_params = fast_state.cms_params[level_name]
forward_params = params_with_deltas(self.cms.blocks[level_name], base_params)
params_req = require_grad_params(forward_params)
⋮----
prediction = call_with_params(self.cms.blocks[level_name], params_req, chunk_inputs)
⋮----
grads = torch.autograd.grad(
grads_dict = grads_to_dict(params_req, grads)
⋮----
@dataclass
class HOPESelfModBlockConfig
⋮----
qk_l2_norm: bool = True
⋮----
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
⋮----
class HOPESelfModBlock(nn.Module)
⋮----
def __init__(self, config: HOPESelfModBlockConfig)
⋮----
o = self.selfmod(x)
⋮----
cms_out = self._cms_forward_online(o, teach_signal, surprise_value)
⋮----
o = self.selfmod.forward_with_state(x, fast_state.selfmod_state)
⋮----
cms_out = self._cms_forward_online_fast(o, fast_state, teach_signal, surprise_value)
⋮----
class HOPEBlock(nn.Module)
⋮----
def __init__(self, config: HOPEBlockConfig)
⋮----
titan_config = TitanMemoryConfig(
⋮----
specs = [config.titan_level, *config.cms_levels]
⋮----
mem_out = self.titan_memory(attn_out)
combined = attn_out + mem_out
⋮----
cms_out = self._cms_forward_online(combined, teach_signal, surprise_value)
⋮----
cms_result = self.cms(combined, return_intermediates=True)
⋮----
mem_out = call_with_deltas(self.titan_memory, fast_state.titan_params, attn_out)
⋮----
level_name = self.config.titan_level.name
⋮----
modifier = self.self_modifier(
context_vec = attn_out.detach().mean(dim=(0, 1))
⋮----
query = attn_out.detach()
target = (modifier - teach_signal.detach()).detach()
base_params = {name: param for name, param in self.titan_memory.named_parameters()}
params_req = require_grad_params(base_params)
prediction = call_with_params(self.titan_memory, params_req, query)
loss_terms = F.mse_loss(prediction, target, reduction="none")
active = teach_signal.detach().abs().sum(dim=-1, keepdim=True) > 0
mask = active.float()
⋮----
norms = teach_signal.norm(dim=-1, keepdim=True)
mask = mask * (norms >= self.surprise_threshold).float()
loss = (loss_terms * mask).sum() / mask.sum().clamp(min=1.0)
⋮----
magnitude = self.level_manager.apply_module_grads(
extra_metrics = self.level_manager.pop_last_metrics(level_name)
stats = {"grad_norm": magnitude, "gate_hit": 1.0}
⋮----
base_params = fast_state.titan_params
forward_params = params_with_deltas(self.titan_memory, base_params)
⋮----
extra_metrics = fast_state.level_manager.pop_last_metrics(level_name)
```

## File: src/nested_learning/model.py
```python
@dataclass
class ModelConfig
⋮----
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
class HOPEModel(nn.Module)
⋮----
def __init__(self, config: ModelConfig)
⋮----
variant = str(config.block_variant).strip().lower()
⋮----
attn_block_config = HOPEAttentionBlockConfig(
⋮----
hybrid_block_config = HOPEBlockConfig(
⋮----
selfmod_block_config = HOPESelfModBlockConfig(
⋮----
transformer_block_config = TransformerBlockConfig(
⋮----
# Weight tying keeps the LM head gradient aligned with the embedding space.
⋮----
def set_teach_runtime(self, *, scale: float | None = None, clip: float | None = None) -> None
def set_surprise_threshold(self, threshold: float | None) -> None
def get_surprise_threshold(self) -> float | None
def set_surprise_metric(self, metric: str) -> None
⋮----
normalized = str(metric).strip().lower()
allowed = {"l2", "loss", "logit_entropy"}
⋮----
def get_surprise_metric(self) -> str
def set_allowed_update_levels(self, levels: set[str] | None) -> None
def get_allowed_update_levels(self) -> set[str] | None
def set_allowed_update_layers(self, layers: set[int] | None) -> None
⋮----
normalized: set[int] = set()
total = len(self.blocks)
⋮----
layer_idx = int(idx)
⋮----
layer_idx = total + layer_idx
⋮----
def get_allowed_update_layers(self) -> set[int] | None
⋮----
x = self._run_blocks(
pre_norm = cast(torch.Tensor, x)
x = self.norm(pre_norm)
logits = self.lm_head(x)
⋮----
pre_norm = x
x = self.norm(x)
⋮----
x = self.embed(tokens)
block_outputs: list[torch.Tensor] = []
runtime_scale = self._runtime_teach_scale
runtime_clip = self._runtime_teach_clip
⋮----
require_external = self._surprise_metric in {"loss", "logit_entropy"}
⋮----
base_surprise = surprise_value
scaled_global_signal: torch.Tensor | None = None
⋮----
scaled_global_signal = teach_signal * runtime_scale
⋮----
norm = scaled_global_signal.norm(dim=-1, keepdim=True)
scale = torch.clamp(norm / runtime_clip, min=1.0)
scaled_global_signal = scaled_global_signal / scale
base_surprise = float(scaled_global_signal.norm(dim=-1).mean().item())
⋮----
block_state = None if fast_state is None else fast_state.blocks[idx]
scaled_signal = None
block_surprise = base_surprise
⋮----
scaled_signal = teach_signal * runtime_scale
⋮----
norm = scaled_signal.norm(dim=-1, keepdim=True)
⋮----
scaled_signal = scaled_signal / scale
⋮----
scaled_signal = scaled_global_signal
⋮----
scaled_signal = teach_signals[idx] * self._runtime_teach_scale
⋮----
block_surprise = float(scaled_signal.norm(dim=-1).mean().item())
⋮----
scale = torch.clamp(norm / self._runtime_teach_clip, min=1.0)
⋮----
x = checkpoint(block_call, x, use_reentrant=False)
⋮----
x = block_call(x)
⋮----
def _gather_block_stats(self) -> Dict[str, float]
⋮----
metrics: Dict[str, float] = {}
⋮----
pop_fn = getattr(block, "pop_update_stats", None)
⋮----
stats = cast(Dict[str, Dict[str, float]], pop_fn())
⋮----
prefix = f"layer{idx}.{level_name}"
⋮----
def pop_update_metrics(self) -> Dict[str, float]
⋮----
metrics = self._latest_update_metrics
⋮----
def init_fast_state(self) -> ModelFastState
⋮----
states = []
⋮----
specs = [block.config.titan_level, *block.config.cms_levels]
state = build_block_fast_state(
⋮----
specs = list(block.config.cms_levels)
⋮----
def freeze_backbone(self) -> None
⋮----
"""
        Freeze the shared transformer spine (embeddings, attention blocks, norm, LM head).
        HOPE/TITAN/CMS memories remain trainable for adapter-style finetuning.
        """
⋮----
attn = getattr(block, "attn", None)
⋮----
class _UpdateControlledBlock(Protocol)
⋮----
def set_surprise_threshold(self, threshold: float | None) -> None: ...
def set_surprise_metric(self, metric: str) -> None: ...
def set_allowed_levels(self, allowed: set[str] | None) -> None: ...
```

## File: src/nested_learning/training.py
```python
@dataclass
class DistributedContext
⋮----
rank: int
world_size: int
device: torch.device
def unwrap_config(cfg: DictConfig) -> DictConfig
def build_model_from_cfg(model_cfg: DictConfig) -> torch.nn.Module
⋮----
model_type = model_cfg.get("type", "hope")
optimizer_cfg: Dict[str, dict] = {}
⋮----
optimizer_cfg = cast(
teach_scale = model_cfg.get("teach_scale", 1.0)
teach_clip = model_cfg.get("teach_clip", 0.0)
teach_schedule: Dict[str, float] = {}
⋮----
teach_schedule = cast(
qk_l2_norm = bool(model_cfg.get("qk_l2_norm", False))
local_conv_window_raw = model_cfg.get("local_conv_window")
local_conv_window = None if local_conv_window_raw is None else int(local_conv_window_raw)
surprise_threshold_raw = model_cfg.get("surprise_threshold")
surprise_threshold = (
surprise_metric = str(model_cfg.get("surprise_metric", "l2"))
cms_use_layernorm = bool(model_cfg.get("cms_use_layernorm", True))
⋮----
titan_spec = LevelSpec(**model_cfg.titan_level)
titan_cfg = TitanOnlyModelConfig(
⋮----
cms_specs = [LevelSpec(**entry) for entry in model_cfg.cms_levels]
self_mod_chunk_size_memory_raw = model_cfg.get("self_mod_chunk_size_memory")
self_mod_chunk_size_memory = (
self_mod_local_conv_window_raw = model_cfg.get("self_mod_local_conv_window", 4)
self_mod_local_conv_window = (
hope_cfg = ModelConfig(
⋮----
dataset = _build_dataset(data_cfg)
use_sampler = distributed and not isinstance(dataset, IterableDataset)
⋮----
sampler: DistributedSampler | None = DistributedSampler(
shuffle = False
⋮----
sampler = None
shuffle = True
⋮----
generator = None
worker_init_fn = None
⋮----
generator = torch.Generator()
⋮----
worker_init_fn = _make_worker_init_fn(seed)
dataloader = DataLoader(
⋮----
def _build_dataset(data_cfg: DictConfig)
⋮----
source = data_cfg.source
⋮----
synth_cfg = SyntheticTextConfig(
⋮----
shard_dir = data_cfg.shards_dir
⋮----
mixture_cfg = data_cfg.mixture
sources = [
samples_per_epoch = mixture_cfg.samples_per_epoch
seed = mixture_cfg.get("seed", 0)
⋮----
msg = f"Unsupported data source {source}"
⋮----
logits_detached = logits.detach()
probs = torch.softmax(logits_detached, dim=-1)
target_tokens = tokens[:, 1:]
residual = probs[:, :-1].clone()
⋮----
safe_targets = target_tokens
src = -torch.ones(
denom: torch.Tensor | float = max(1, tokens.size(0) * max(1, tokens.size(1) - 1))
⋮----
active = target_tokens != ignore_index
safe_targets = torch.where(active, target_tokens, torch.zeros_like(target_tokens))
active_f = active.to(dtype=residual.dtype)
⋮----
src = -active_f.unsqueeze(-1)
denom = active_f.sum().clamp(min=1.0)
⋮----
residual = residual / denom
head_weight = model.lm_head.weight.detach()
⋮----
head_weight = head_weight.to(dtype=residual.dtype)
grad = residual @ head_weight
pad = torch.zeros(
⋮----
grads = torch.autograd.grad(
⋮----
normalized = str(metric).strip().lower()
⋮----
logits_detached = logits[:, :-1].detach().float()
⋮----
entropy = -(probs * torch.log(probs.clamp(min=1e-9))).sum(dim=-1).mean()
⋮----
def _infer_online_chunk_size(model: HOPEModel) -> int | None
⋮----
min_period: int | None = None
blocks = getattr(model, "blocks", [])
⋮----
cfg = getattr(block, "config", None)
levels = getattr(cfg, "cms_levels", None)
⋮----
period = int(spec.update_period)
⋮----
min_period = period if min_period is None else min(min_period, period)
⋮----
class _HasLMHead(Protocol)
⋮----
lm_head: torch.nn.Linear
def _checksum_path(path: str | None) -> str | None
⋮----
candidate = Path(path)
⋮----
digest = sha256()
⋮----
ckpt_cfg = cfg.train.get("checkpoint")
⋮----
save_interval = ckpt_cfg.get("save_interval", total_steps)
save_last = ckpt_cfg.get("save_last", True)
is_last_step = (step + 1) >= total_steps
should_save = ((step + 1) % max(1, save_interval) == 0) or (save_last and is_last_step)
⋮----
ckpt_dir = Path(ckpt_cfg.get("dir", "checkpoints/default"))
⋮----
global_step = step + 1 + int(step_offset)
ckpt_path = ckpt_dir / f"step_{global_step:06d}.pt"
tmp_path = ckpt_path.with_suffix(".tmp")
resolved_cfg = OmegaConf.to_container(cfg, resolve=True)
state = {
⋮----
prefix = "[checkpoint]"
⋮----
prefix = f"[checkpoint rank={dist_ctx.rank}]"
⋮----
def _validate_distributed_config(cfg: DictConfig, distributed: bool) -> None
⋮----
fail_if_faithful_disabled = bool(cfg.train.get("fail_if_paper_faithful_disabled", False))
⋮----
def _validate_fast_state_batch_semantics(cfg: DictConfig) -> None
⋮----
data_cfg = cfg.get("data")
⋮----
batch_size_raw = data_cfg.get("batch_size", 1)
⋮----
batch_size = int(batch_size_raw)
⋮----
msg = (
⋮----
resume_path_raw = ckpt_cfg.get("resume_path")
⋮----
resume_path = Path(str(resume_path_raw))
⋮----
state = torch.load(resume_path, map_location=device, weights_only=False)
⋮----
model_state = state.get("model")
⋮----
model_state = state
⋮----
optimizer_state = state.get("optimizer")
⋮----
start_step = int(state.get("step", 0) or 0)
⋮----
model = build_model_from_cfg(cfg.model).to(device)
train_seed = cfg.train.get("seed")
deterministic = cfg.train.get("deterministic", False)
⋮----
model = _maybe_compile_model(model, cfg.train.get("compile"))
⋮----
idx = device.index if device.index is not None else 0
model = torch.nn.parallel.DistributedDataParallel(
⋮----
base_model = model.module
⋮----
base_model = model
seed_offset = 0
⋮----
seed_offset = dist_ctx.rank
dataloader_seed = None if train_seed is None else int(train_seed) + seed_offset
⋮----
optimizer = _build_optimizer(base_model, cfg, device=device)
start_step = _maybe_resume_training(
autocast_factory = _make_autocast_factory(device, cfg.train.get("mixed_precision"))
logger = init_logger(getattr(cfg, "logging", None), cfg)
⋮----
logger = NullLogger()
⋮----
steps = cfg.train.steps
log_interval = cfg.train.get("log_interval", 1)
per_layer_teach = bool(cfg.train.get("per_layer_teach_signal", False))
online_updates = bool(cfg.train.get("online_updates", False))
online_chunk_size = int(cfg.train.get("online_chunk_size", 0) or 0)
use_fast_state = bool(cfg.train.get("use_fast_state", False))
⋮----
msg = "[train] per_layer_teach_signal disabled under DDP (uses base model methods)"
⋮----
per_layer_teach = False
⋮----
msg = "[train] online_updates disabled under DDP (uses base model methods)"
⋮----
online_updates = False
step_iter = iter(dataloader)
epoch = 0
metrics: Dict[str, float] = {}
surprise_metric_getter = getattr(base_model, "get_surprise_metric", None)
surprise_metric = (
⋮----
batch = next(step_iter)
⋮----
tokens = batch.to(device)
fast_state = None
⋮----
init_fn = getattr(base_model, "init_fast_state", None)
⋮----
fast_state = init_fn()
⋮----
update_metrics: Dict[str, float] = {}
surprise_value_for_step = 0.0
⋮----
total_loss = 0.0
total_tokens = 0
teach_signal_norm = 0.0
total_surprise = 0.0
⋮----
chunk_size = online_chunk_size
⋮----
inferred = _infer_online_chunk_size(base_model)
chunk_size = inferred if inferred is not None else tokens.size(1)
⋮----
chunk_size = 2
⋮----
end = min(start + chunk_size, tokens.size(1))
chunk_tokens = tokens[:, start:end]
⋮----
loss = torch.nn.functional.cross_entropy(
surprise_override = _compute_surprise_override(
⋮----
teach_signals = _compute_layer_teach_signals(loss, block_outputs)
chunk_teach_norm = float(
⋮----
teach_signal = compute_teach_signal(base_model, logits, chunk_tokens)
chunk_teach_norm = teach_signal.norm(dim=-1).mean().item()
⋮----
chunk_surprise = (
⋮----
chunk_metrics = base_model.pop_update_metrics()
⋮----
loss = torch.tensor(total_loss / max(total_tokens, 1), device=device)
teach_signal_norm = teach_signal_norm / max(total_tokens, 1)
surprise_value_for_step = total_surprise / max(total_tokens, 1)
⋮----
logits = model(tokens, fast_state=fast_state)
⋮----
logits = model(tokens)
⋮----
teach_signal_norm = float(
⋮----
teach_signal = compute_teach_signal(base_model, logits, tokens)
teach_signal_norm = teach_signal.norm(dim=-1).mean().item()
⋮----
update_metrics = base_model.pop_update_metrics()
surprise_value_for_step = (
⋮----
ppl = _safe_perplexity(loss)
metrics_payload = {
⋮----
metrics = metrics_payload
⋮----
def _safe_perplexity(loss: torch.Tensor, max_value: float = 1.0e38) -> float
⋮----
clamped_loss = min(float(loss.detach().double().item()), math.log(max_value))
⋮----
def _apply_teach_schedule(model: HOPEModel, cfg: DictConfig, step: int) -> None
⋮----
schedule = cfg.model.get("teach_schedule")
base_scale = cfg.model.get("teach_scale", 1.0)
scale = base_scale
⋮----
warmup = schedule.get("warmup_steps", 0)
⋮----
decay_start = schedule.get("decay_start")
decay_duration = schedule.get("decay_duration")
⋮----
progress = min(1.0, (step + 1 - decay_start) / decay_duration)
⋮----
def _maybe_compile_model(model: torch.nn.Module, compile_cfg: dict | None) -> torch.nn.Module
⋮----
kwargs = {}
⋮----
def _make_autocast_factory(device: torch.device, mp_cfg: dict | None)
⋮----
dtype = _resolve_autocast_dtype(mp_cfg.get("dtype", "bf16"))
device_type = device.type
⋮----
device_type = "cpu"
def factory()
⋮----
def _resolve_autocast_dtype(name: str) -> torch.dtype
⋮----
normalized = str(name).lower()
⋮----
msg = f"Unsupported autocast dtype {name}"
⋮----
optimizer_cfg_raw = cfg.get("optim")
⋮----
optimizer_cfg = optimizer_cfg_raw
⋮----
optimizer_cfg = cast(DictConfig, OmegaConf.create(optimizer_cfg_raw or {}))
param_policy_raw = optimizer_cfg.get("param_policy")
⋮----
outer_updates_memory_modules = optimizer_cfg.get("outer_updates_memory_modules")
⋮----
param_policy = "all"
⋮----
param_policy = "all" if bool(outer_updates_memory_modules) else "exclude_memory"
⋮----
param_policy = str(param_policy_raw).strip().lower()
named_params = _select_outer_named_parameters(model, param_policy)
⋮----
optim_type = str(optimizer_cfg.get("type", "adamw")).lower()
⋮----
lr = optimizer_cfg.get("lr", 1e-3)
betas = optimizer_cfg.get("betas", (0.9, 0.999))
weight_decay = optimizer_cfg.get("weight_decay", 0.0)
fused_cfg = optimizer_cfg.get("fused", "auto")
fused = False
⋮----
fused = device.type == "cuda" and torch.cuda.is_available()
⋮----
fused = bool(fused_cfg)
kwargs = {"lr": lr, "betas": betas, "weight_decay": weight_decay}
⋮----
params = [param for _, param in named_params]
⋮----
weight_decay = optimizer_cfg.get("weight_decay", 0.01)
momentum = optimizer_cfg.get("momentum", 0.95)
ns_coefficients = optimizer_cfg.get("ns_coefficients")
ns_steps = optimizer_cfg.get("ns_steps")
eps = optimizer_cfg.get("eps", 1e-7)
⋮----
muon_params: list[torch.nn.Parameter] = []
adamw_params: list[torch.nn.Parameter] = []
source = named_params if named_params is not None else model.named_parameters()
⋮----
muon_kwargs = {
⋮----
muon_opt = (
adamw_kwargs = {
⋮----
adamw_opt = torch.optim.AdamW(adamw_params, **adamw_kwargs) if adamw_params else None
muon_elems = int(sum(p.numel() for p in muon_params))
adamw_elems = int(sum(p.numel() for p in adamw_params))
⋮----
beta1 = optimizer_cfg.get("beta1", 0.9)
beta2 = optimizer_cfg.get("beta2", 0.999)
beta3 = optimizer_cfg.get("beta3", 0.9)
alpha = optimizer_cfg.get("alpha", 1.0)
ns_steps = int(optimizer_cfg.get("ns_steps", 3))
slow_chunk = int(optimizer_cfg.get("slow_chunk", 100))
eps = optimizer_cfg.get("eps", 1e-8)
⋮----
m3_params: list[torch.nn.Parameter] = []
⋮----
m3_opt = (
⋮----
m3_elems = int(sum(p.numel() for p in m3_params))
⋮----
policy = str(param_policy).strip().lower()
trainable: list[tuple[str, torch.nn.Parameter]] = [
⋮----
def _is_memory_param_name(name: str) -> bool
⋮----
lowered = name.lower()
⋮----
def _is_muon_candidate(name: str, param: torch.nn.Parameter) -> bool
class _HybridOptimizer
⋮----
def zero_grad(self) -> None
def step(self) -> None
def state_dict(self) -> dict
def load_state_dict(self, state: dict) -> None
⋮----
@property
    def param_groups(self)
⋮----
groups = []
⋮----
def get_param_split(self) -> dict[str, int]
⋮----
mp_cfg = cfg.train.get("mixed_precision", {})
compile_cfg = cfg.train.get("compile", {})
features: dict[str, object] = {
surprise_metric_getter = getattr(model, "get_surprise_metric", None)
surprise_threshold_getter = getattr(model, "get_surprise_threshold", None)
⋮----
selected = _select_outer_named_parameters(model, param_policy)
total_elems = int(sum(param.numel() for _, param in selected))
memory_elems = int(
⋮----
split_fn = getattr(optimizer, "get_param_split", None)
⋮----
split = split_fn()
⋮----
def _detect_flash_attention(model: torch.nn.Module) -> bool
⋮----
attn = getattr(block, "attn", None)
config = getattr(attn, "config", None)
⋮----
def write_checkpoint_metadata(cfg: DictConfig, ckpt_path: Path, step: int) -> None
⋮----
config_yaml = OmegaConf.to_yaml(cfg)
config_path = ckpt_path.with_suffix(".yaml")
⋮----
config_hash = sha256(config_yaml.encode("utf-8")).hexdigest()
ckpt_hash = _checksum_path(str(ckpt_path))
sha_path = ckpt_path.with_suffix(".sha256")
⋮----
tokenizer_path = cfg.data.get("tokenizer_path") if hasattr(cfg, "data") else None
metadata = {
⋮----
def verify_checkpoint_integrity(ckpt_path: Path) -> Dict[str, object]
⋮----
meta_path = ckpt_path.with_suffix(".meta.json")
⋮----
metadata = json.loads(meta_path.read_text())
computed_sha = _checksum_path(str(ckpt_path))
recorded_sha = metadata.get("checkpoint_sha256")
⋮----
sha_file = ckpt_path.with_suffix(".sha256")
⋮----
recorded_line = sha_file.read_text().strip().split()
⋮----
recorded = recorded_line[0]
⋮----
config_hash = sha256(config_path.read_text().encode("utf-8")).hexdigest()
recorded_cfg_hash = metadata.get("config_sha256")
⋮----
def _capture_rng_states() -> Dict[str, object]
⋮----
payload: Dict[str, object] = {
⋮----
def _encode_pickle(obj: object) -> str
def _tensor_state_to_hex(state: torch.Tensor) -> str
def _seed_everything(seed: int, *, deterministic: bool = False) -> None
def _make_worker_init_fn(base_seed: int)
⋮----
def _init_fn(worker_id: int) -> None
⋮----
worker_seed = base_seed + worker_id
```
