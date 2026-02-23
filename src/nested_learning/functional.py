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
    *,
    validate: bool = True,
) -> torch.Tensor:
    """
    Vectorized per-sample fast-state forward.

    `batched_deltas` stores one delta slice per sample with shape [B, *param.shape].
    """

    if inputs.ndim < 1:
        raise ValueError("inputs must include a batch dimension")
    if validate:
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
