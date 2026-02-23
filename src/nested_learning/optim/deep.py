from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class DeepMomentumState:
    grad_avg: Optional[torch.Tensor] = None
    sq_avg: Optional[torch.Tensor] = None


def _broadcast_sample_mask(sample_mask: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """
    Broadcast a [B] boolean mask to match `like` (assumed batched on dim 0).
    Returns a bool tensor of shape [B, 1, 1, ...].
    """
    if sample_mask.ndim != 1:
        raise ValueError("sample_mask must have shape [B]")
    if like.ndim < 1 or like.size(0) != sample_mask.size(0):
        raise ValueError("sample_mask batch must match grad batch")
    shape = (sample_mask.size(0),) + (1,) * (like.ndim - 1)
    return sample_mask.view(shape).to(device=like.device, dtype=torch.bool)


class DeepMomentum(nn.Module):
    """Implements momentum variants described in the NL paper."""

    def __init__(
        self,
        *,
        beta: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        variant: str = "preconditioned",
        delta_alpha: float | None = None,
        delta_eta: float = 1.0,
        delta_preconditioner: str = "identity",
        dmgd_hidden_dim: int = 16,
        dmgd_inner_lr: float = 1e-3,
        dmgd_inner_objective: str = "mse",
    ) -> None:
        super().__init__()
        self.beta = beta
        self.beta2 = beta2
        self.eps = eps
        self.variant = variant
        self.delta_alpha = delta_alpha
        self.delta_eta = delta_eta
        self.delta_preconditioner = str(delta_preconditioner).strip().lower()
        self.dmgd_hidden_dim = int(dmgd_hidden_dim)
        self.dmgd_inner_lr = float(dmgd_inner_lr)
        self.dmgd_inner_objective = str(dmgd_inner_objective).strip().lower()
        self.state: dict[str, DeepMomentumState] = {}
        self.nonlinearity = nn.Tanh() if variant in {"dmgd", "muon"} else nn.Identity()
        self.dmgd_mlp: nn.Sequential | None = None
        if self.variant == "dmgd_mlp":
            if self.dmgd_hidden_dim <= 0:
                raise ValueError("dmgd_hidden_dim must be > 0")
            if self.dmgd_inner_lr <= 0:
                raise ValueError("dmgd_inner_lr must be > 0")
            if self.dmgd_inner_objective not in {"mse", "cosine"}:
                raise ValueError(
                    "dmgd_inner_objective must be one of {'mse', 'cosine'}; "
                    f"got {self.dmgd_inner_objective!r}"
                )
            self.dmgd_mlp = nn.Sequential(
                nn.Linear(5, self.dmgd_hidden_dim),
                nn.SiLU(),
                nn.Linear(self.dmgd_hidden_dim, 2),
            )
        self.last_metrics: dict[str, float] = {}

    def reset_state(self) -> None:
        self.state.clear()

    def _precondition(
        self,
        grad: torch.Tensor,
        state: DeepMomentumState,
        *,
        sample_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if state.sq_avg is None or state.sq_avg.shape != grad.shape:
            state.sq_avg = torch.zeros_like(grad)
        if sample_mask is None:
            state.sq_avg.mul_(self.beta2).addcmul_(grad, grad, value=1 - self.beta2)
        else:
            mask_b = _broadcast_sample_mask(sample_mask, grad)
            sq_new = state.sq_avg.mul(self.beta2).addcmul(grad, grad, value=1 - self.beta2)
            state.sq_avg = torch.where(mask_b, sq_new, state.sq_avg)
        denom = state.sq_avg.sqrt().add_(self.eps)
        return grad / denom

    def _nl_precondition(
        self,
        grad: torch.Tensor,
        context: torch.Tensor | None,
        *,
        sample_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        metrics: dict[str, float] = {
            "ctx_norm": 0.0,
            "proj_norm": 0.0,
            "proj_skipped": 0.0,
        }
        if context is None:
            return grad, metrics
        ctx = context
        if grad.ndim >= 2 and ctx.ndim == 2 and ctx.size(0) == grad.size(0):
            if grad.shape[-1] != ctx.shape[-1]:
                metrics["proj_skipped"] = 1.0
                return grad, metrics
            ctx_norm = torch.norm(ctx, dim=-1, keepdim=True)
            if sample_mask is not None and sample_mask.any():
                metrics["ctx_norm"] = float(ctx_norm[sample_mask].mean().item())
            else:
                metrics["ctx_norm"] = float(ctx_norm.mean().item())
            unit = ctx / (ctx_norm + self.eps)
            unit_expanded = unit
            while unit_expanded.ndim < grad.ndim:
                unit_expanded = unit_expanded.unsqueeze(1)
            projection = (grad * unit_expanded).sum(dim=-1, keepdim=True) * unit_expanded
            update = grad - projection
            metrics["proj_norm"] = float(torch.norm(update).item())
            return update, metrics
        if ctx.ndim > 1:
            ctx = ctx.reshape(-1, ctx.shape[-1]).mean(dim=0)
        ctx_norm = torch.norm(ctx)
        metrics["ctx_norm"] = float(ctx_norm.item())

        if ctx_norm > 0:
            if grad.ndim == 0 or grad.shape[-1] != ctx.shape[-1]:
                metrics["proj_skipped"] = 1.0
                return grad, metrics
            unit = ctx / (ctx_norm + self.eps)
            projection = (grad * unit).sum(dim=-1, keepdim=True) * unit
            update = grad - projection
            metrics["proj_norm"] = float(torch.norm(update).item())
            return update, metrics
        return grad, metrics

    def _delta_rule_update(
        self,
        grad: torch.Tensor,
        state: DeepMomentumState,
    ) -> torch.Tensor:
        m_prev = state.grad_avg if state.grad_avg is not None else torch.zeros_like(grad)
        g_flat = grad.reshape(-1)
        m_flat = m_prev.reshape(-1)
        dot = torch.dot(g_flat, m_flat)
        correction = (g_flat * dot).reshape_as(grad)

        if self.delta_preconditioner == "identity":
            p_grad = grad
        elif self.delta_preconditioner == "adam_diag":
            p_grad = self._precondition(grad, state)
        else:
            raise ValueError(
                "delta_preconditioner must be one of {'identity', 'adam_diag'}; "
                f"got {self.delta_preconditioner!r}"
            )

        alpha = self.beta if self.delta_alpha is None else self.delta_alpha
        update = alpha * m_prev - correction - self.delta_eta * p_grad
        self.last_metrics = {
            "delta_rule.grad_norm": float(torch.norm(grad).item()),
            "delta_rule.correction_norm": float(torch.norm(correction).item()),
            "delta_rule.precond_grad_norm": float(torch.norm(p_grad).item()),
        }
        return update

    def _dmgd_features(self, grad: torch.Tensor, m_prev: torch.Tensor) -> torch.Tensor:
        grad_f = grad.detach().float()
        m_prev_f = m_prev.detach().float()
        scale = max(1.0, float(grad_f.numel()) ** 0.5)
        features = torch.stack(
            (
                grad_f.mean(),
                grad_f.std(unbiased=False),
                grad_f.norm() / scale,
                m_prev_f.mean(),
                m_prev_f.std(unbiased=False),
            )
        )
        return features

    def _dmgd_predict(
        self,
        grad: torch.Tensor,
        m_prev: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.dmgd_mlp is None:
            raise RuntimeError("dmgd_mlp variant requested but network is not initialized")
        if self.dmgd_mlp[0].weight.device != grad.device:
            self.dmgd_mlp = self.dmgd_mlp.to(device=grad.device)
        features = self._dmgd_features(grad, m_prev).to(device=grad.device)
        coeffs = self.dmgd_mlp(features.unsqueeze(0)).squeeze(0)
        scale = coeffs[0].to(dtype=grad.dtype)
        bias = coeffs[1].to(dtype=grad.dtype)
        candidate = scale * grad + bias
        return candidate, scale, bias

    def _dmgd_inner_loss(self, momentum: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
        grad_target = grad.detach()
        if self.dmgd_inner_objective == "mse":
            return torch.mean((momentum - grad_target) ** 2)
        if self.dmgd_inner_objective == "cosine":
            dot = torch.sum(momentum * grad_target)
            denom = momentum.norm() * grad_target.norm() + self.eps
            return -dot / denom
        raise ValueError(f"Unsupported dmgd_inner_objective {self.dmgd_inner_objective!r}")

    def _dmgd_mlp_update(
        self,
        grad: torch.Tensor,
        state: DeepMomentumState,
    ) -> torch.Tensor:
        m_prev = state.grad_avg if state.grad_avg is not None else torch.zeros_like(grad)
        with torch.enable_grad():
            candidate, scale, bias = self._dmgd_predict(grad, m_prev)
            momentum = self.beta * m_prev + (1.0 - self.beta) * candidate
            inner_loss = self._dmgd_inner_loss(momentum, grad)
            params = tuple(self.dmgd_mlp.parameters()) if self.dmgd_mlp is not None else ()
            grads = torch.autograd.grad(inner_loss, params, retain_graph=False, allow_unused=False)

        grad_norm = 0.0
        with torch.no_grad():
            for param, grad_param in zip(params, grads, strict=True):
                param.mul_(self.beta).add_(grad_param, alpha=-self.dmgd_inner_lr)
                grad_norm += float(grad_param.norm().item())
            updated, updated_scale, updated_bias = self._dmgd_predict(grad, m_prev)
            momentum = self.beta * m_prev + (1.0 - self.beta) * updated

        self.last_metrics = {
            "dmgd_mlp.inner_loss": float(inner_loss.item()),
            "dmgd_mlp.mlp_grad_norm": grad_norm,
            "dmgd_mlp.scale_before": float(scale.item()),
            "dmgd_mlp.bias_before": float(bias.item()),
            "dmgd_mlp.scale_after": float(updated_scale.item()),
            "dmgd_mlp.bias_after": float(updated_bias.item()),
        }
        return momentum

    def forward(  # type: ignore[override]
        self,
        grad: torch.Tensor,
        *,
        context: torch.Tensor | None = None,
        param_key: str | None = None,
        sample_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if sample_mask is not None:
            sample_mask = sample_mask.to(device=grad.device, dtype=torch.bool)
            if grad.ndim < 1:
                raise ValueError("sample_mask requires batched grad with leading batch dimension")
            if sample_mask.ndim != 1 or sample_mask.size(0) != grad.size(0):
                raise ValueError(
                    f"sample_mask must have shape [B]={grad.size(0)}; got {tuple(sample_mask.shape)}"
                )
        key = param_key or "__default__"
        state = self.state.get(key)
        if state is None:
            state = DeepMomentumState()
            self.state[key] = state
        if state.grad_avg is None or state.grad_avg.shape != grad.shape:
            state.grad_avg = torch.zeros_like(grad)
        self.last_metrics = {}
        if self.variant == "delta_rule":
            if sample_mask is not None:
                raise ValueError("delta_rule variant does not support batched sample_mask updates")
            update = self._delta_rule_update(grad, state)
            state.grad_avg.copy_(update)
            return state.grad_avg
        if self.variant == "dmgd_mlp":
            if sample_mask is not None:
                raise ValueError("dmgd_mlp variant does not support batched sample_mask updates")
            update = self._dmgd_mlp_update(grad, state)
            state.grad_avg.copy_(update)
            return state.grad_avg
        update = grad
        if self.variant in {"preconditioned", "muon"}:
            update = self._precondition(grad, state, sample_mask=sample_mask)
        if self.variant == "l2_objective":
            update = grad + 0.1 * torch.mean(grad, dim=-1, keepdim=True)
        if self.variant == "nl_l2_precond":
            update, metrics = self._nl_precondition(grad, context, sample_mask=sample_mask)
            self.last_metrics.update(metrics)
        if self.variant in {"dmgd", "muon"}:
            update = self.nonlinearity(update)
        if sample_mask is None:
            state.grad_avg.mul_(self.beta).add_(update, alpha=1 - self.beta)
        else:
            mask_b = _broadcast_sample_mask(sample_mask, grad)
            grad_new = state.grad_avg.mul(self.beta).add(update, alpha=1 - self.beta)
            state.grad_avg = torch.where(mask_b, grad_new, state.grad_avg)
        return state.grad_avg
