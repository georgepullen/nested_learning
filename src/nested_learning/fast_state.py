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
        elif mode == "tensorized_cms":
            titan_params = init_module_deltas_batched(titan_module, batch_size=batch_size)
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
    elif mode == "tensorized_cms":
        # Keep one manager with batched optimizer states for tensorized CMS updates.
        level_manager = LevelOptimizerManager(level_cfg)
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


@dataclass
class AttentionKVCache:
    """
    Per-layer autoregressive attention cache.

    Shapes:
    - key:   [batch, heads, cached_tokens, head_dim]
    - value: [batch, heads, cached_tokens, head_dim]
    """

    key: torch.Tensor
    value: torch.Tensor


@dataclass
class ModelAttentionCache:
    """
    Model-level container for per-block attention caches.

    Blocks without attention store `None` entries.
    """

    blocks: list[AttentionKVCache | None]
