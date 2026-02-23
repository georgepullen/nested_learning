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
    assert new_obj <= old_obj + 1e-6


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


def test_deep_momentum_delta_rule_identity_matches_expected_recurrence() -> None:
    grad = torch.tensor([1.0, 2.0])
    optimizer = DeepMomentum(
        beta=0.9,
        beta2=0.0,
        variant="delta_rule",
        delta_eta=1.0,
        delta_preconditioner="identity",
    )

    out1 = optimizer(grad, param_key="p").detach().clone()
    assert torch.allclose(out1, -grad)

    out2 = optimizer(grad, param_key="p")
    dot = torch.dot(grad, out1)
    expected = 0.9 * out1 - grad * dot - grad
    assert torch.allclose(out2, expected)
    assert optimizer.last_metrics["delta_rule.grad_norm"] > 0.0
    assert optimizer.last_metrics["delta_rule.precond_grad_norm"] > 0.0


def test_deep_momentum_delta_rule_supports_adam_diagonal_preconditioner() -> None:
    grad = torch.tensor([3.0, 4.0])
    optimizer = DeepMomentum(
        beta=0.0,
        beta2=0.9,
        eps=1e-8,
        variant="delta_rule",
        delta_eta=1.0,
        delta_preconditioner="adam_diag",
    )
    out = optimizer(grad, param_key="p")
    # With beta2=0.9 and m_0=0, sq_avg=(1-beta2)*g^2 and P*g = g/sqrt(sq_avg).
    expected = -grad / ((1.0 - 0.9) ** 0.5 * grad.abs().clamp(min=1e-8))
    assert torch.allclose(out, expected, atol=1e-5, rtol=1e-5)


def test_deep_momentum_dmgd_mlp_updates_network_params() -> None:
    torch.manual_seed(0)
    optimizer = DeepMomentum(
        beta=0.9,
        beta2=0.0,
        variant="dmgd_mlp",
        dmgd_hidden_dim=8,
        dmgd_inner_lr=1e-2,
        dmgd_inner_objective="mse",
    )
    assert optimizer.dmgd_mlp is not None
    before = [param.detach().clone() for param in optimizer.dmgd_mlp.parameters()]
    grad = torch.randn(4, 3)
    out = optimizer(grad, param_key="p")
    after = list(optimizer.dmgd_mlp.parameters())

    assert out.shape == grad.shape
    assert any(not torch.allclose(old, new) for old, new in zip(before, after, strict=True))
    assert optimizer.last_metrics["dmgd_mlp.inner_loss"] >= 0.0
    assert optimizer.last_metrics["dmgd_mlp.mlp_grad_norm"] > 0.0


def test_deep_momentum_dmgd_mlp_is_deterministic_for_fixed_seed() -> None:
    torch.manual_seed(1234)
    grads = [torch.randn(2, 2), torch.randn(2, 2), torch.randn(2, 2)]

    torch.manual_seed(99)
    opt_a = DeepMomentum(
        beta=0.8,
        beta2=0.0,
        variant="dmgd_mlp",
        dmgd_hidden_dim=6,
        dmgd_inner_lr=5e-3,
        dmgd_inner_objective="cosine",
    )
    torch.manual_seed(99)
    opt_b = DeepMomentum(
        beta=0.8,
        beta2=0.0,
        variant="dmgd_mlp",
        dmgd_hidden_dim=6,
        dmgd_inner_lr=5e-3,
        dmgd_inner_objective="cosine",
    )

    outs_a = [opt_a(g, param_key="p").detach().clone() for g in grads]
    outs_b = [opt_b(g, param_key="p").detach().clone() for g in grads]
    for left, right in zip(outs_a, outs_b, strict=True):
        assert torch.allclose(left, right)


def test_deep_momentum_sample_mask_preserves_inactive_state() -> None:
    optimizer = DeepMomentum(beta=0.5, beta2=0.9, variant="preconditioned")
    grad1 = torch.tensor([[1.0, 2.0], [3.0, 4.0], [1.5, -2.0]])
    grad2 = torch.tensor([[2.0, 1.0], [7.0, -3.0], [0.5, 0.25]])
    mask = torch.tensor([True, False, True], dtype=torch.bool)

    out1 = optimizer(grad1, param_key="p", sample_mask=mask).detach().clone()
    out2 = optimizer(grad2, param_key="p", sample_mask=mask).detach().clone()

    # Inactive sample keeps its state unchanged across calls.
    assert torch.allclose(out1[1], torch.zeros_like(out1[1]))
    assert torch.allclose(out2[1], out1[1])
    # Active samples accumulate non-zero state.
    assert torch.norm(out2[0]) > 0
    assert torch.norm(out2[2]) > 0


def test_deep_momentum_nl_preconditioner_supports_batched_context() -> None:
    torch.manual_seed(123)
    grad = torch.randn(2, 5)
    context = torch.randn(2, 5)
    optimizer = DeepMomentum(beta=0.0, beta2=0.0, variant="nl_l2_precond")
    mask = torch.tensor([True, True], dtype=torch.bool)
    update = optimizer(grad, context=context, sample_mask=mask)

    ctx_norm = context.norm(dim=-1, keepdim=True)
    unit = context / ctx_norm
    expected = grad - (grad * unit).sum(dim=-1, keepdim=True) * unit
    assert torch.allclose(update, expected, atol=1e-5, rtol=1e-4)
