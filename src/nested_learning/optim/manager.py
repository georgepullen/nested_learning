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
        self._optimizer_uses_context: Dict[str, bool] = {}
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
            variant = str(getattr(optimizer, "variant", "")).strip().lower()
            self._optimizer_uses_context[spec.name] = variant == "nl_l2_precond"

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

    def needs_context(self, level: str) -> bool:
        return bool(self._optimizer_uses_context.get(level, False))

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

    def apply_grads_batched(
        self,
        level: str,
        params: Dict[str, torch.Tensor],
        grads: Dict[str, torch.Tensor],
        *,
        context: torch.Tensor | None = None,
        sample_mask: torch.Tensor | None = None,
        validate: bool = True,
        force: bool = False,
    ) -> tuple[Dict[str, torch.Tensor], float]:
        """
        Batched variant of apply_grads().

        Expects each param/grad tensor to have leading batch dimension [B, ...].
        `sample_mask` (bool [B]) can be used to skip updates for inactive samples while
        preserving their optimizer state.
        """
        if (not force) and (not self.should_update(level)):
            return params, 0.0
        if not params:
            return params, 0.0
        if sample_mask is not None and validate:
            if sample_mask.dtype != torch.bool:
                raise ValueError("sample_mask must be a bool tensor")
            if sample_mask.ndim != 1:
                raise ValueError("sample_mask must have shape [B]")
            if not bool(sample_mask.any().item()):
                return params, 0.0
        if sample_mask is not None and bool(sample_mask.all().item()):
            sample_mask = None

        optimizer = self.optimizers[level]
        lr = self.learning_rates[level]
        updated: Dict[str, torch.Tensor] = {}
        total_norm = 0.0
        variant = str(getattr(optimizer, "variant", "")).strip().lower()
        can_flatten_fast_path = (
            sample_mask is None and context is None and variant in {"preconditioned", "muon"}
        )

        if can_flatten_fast_path:
            items: list[tuple[str, torch.Tensor, torch.Tensor]] = []
            for name, param in params.items():
                grad = grads.get(name)
                if grad is None:
                    updated[name] = param
                    continue
                if validate:
                    if param.ndim < 1 or grad.ndim < 1:
                        raise ValueError(
                            f"Batched apply requires tensors with leading batch dim; got {name} "
                            f"param.ndim={param.ndim}, grad.ndim={grad.ndim}"
                        )
                    if grad.shape != param.shape:
                        raise ValueError(
                            f"Grad/param shape mismatch for {name}: {tuple(grad.shape)} vs {tuple(param.shape)}"
                        )
                items.append((name, param, grad))
            if items:
                batch_size = int(items[0][1].size(0))
                grad_flat = torch.cat(
                    [grad.reshape(batch_size, -1) for _, _, grad in items],
                    dim=1,
                )
                update_flat = optimizer(
                    grad_flat,
                    context=None,
                    param_key="__batched_flat__",
                    sample_mask=None,
                )
                offset = 0
                with torch.no_grad():
                    for name, param, grad in items:
                        numel = grad[0].numel()
                        update = update_flat[:, offset : offset + numel].reshape_as(param)
                        updated[name] = (param - lr * update).detach()
                        offset += numel
                total_norm = float(grad_flat.norm().item())
            self.clock.record_update(level)
            metrics = getattr(optimizer, "last_metrics", None)
            if metrics:
                self._last_metrics[level] = dict(metrics)
            else:
                self._last_metrics[level] = {}
            return updated, total_norm

        with torch.no_grad():
            for name, param in params.items():
                grad = grads.get(name)
                if grad is None:
                    updated[name] = param
                    continue
                if validate:
                    if param.ndim < 1 or grad.ndim < 1:
                        raise ValueError(
                            f"Batched apply requires tensors with leading batch dim; got {name} "
                            f"param.ndim={param.ndim}, grad.ndim={grad.ndim}"
                        )
                    if grad.shape != param.shape:
                        raise ValueError(
                            f"Grad/param shape mismatch for {name}: {tuple(grad.shape)} vs {tuple(param.shape)}"
                        )
                    if sample_mask is not None and sample_mask.size(0) != grad.size(0):
                        raise ValueError(
                            f"sample_mask batch={sample_mask.size(0)} does not match grad batch={grad.size(0)}"
                        )

                update = optimizer(
                    grad,
                    context=context,
                    param_key=name,
                    sample_mask=sample_mask,
                )

                if sample_mask is None:
                    updated[name] = (param - lr * update).detach()
                    total_norm += float(grad.norm().item())
                else:
                    mask_f = sample_mask.view(param.size(0), *([1] * (param.ndim - 1))).to(
                        device=param.device, dtype=param.dtype
                    )
                    updated[name] = (param - lr * update * mask_f).detach()
                    total_norm += float((grad * mask_f).norm().item())
        self.clock.record_update(level)
        metrics = getattr(optimizer, "last_metrics", None)
        if metrics:
            self._last_metrics[level] = dict(metrics)
        else:
            self._last_metrics[level] = {}
        return updated, total_norm
