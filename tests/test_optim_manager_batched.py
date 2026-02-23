from __future__ import annotations

import torch

from nested_learning.levels import LevelSpec
from nested_learning.optim.manager import LevelConfig, LevelOptimizerManager


def _make_manager() -> LevelOptimizerManager:
    spec = LevelSpec(name="cms_fast", update_period=1, optimizer_key="cms_opt")
    cfg = LevelConfig(
        specs=[spec],
        optimizer_configs={
            "cms_opt": {
                "type": "deep_momentum",
                "lr": 1e-3,
                "params": {"beta": 0.0, "beta2": 0.0, "variant": "preconditioned"},
            }
        },
        default_lr=1e-3,
    )
    return LevelOptimizerManager(cfg)


def test_apply_grads_batched_respects_sample_mask() -> None:
    manager = _make_manager()
    params = {"w": torch.zeros(3, 2, 2)}
    grads = {"w": torch.ones(3, 2, 2)}
    context = torch.randn(3, 2)
    mask = torch.tensor([True, False, True], dtype=torch.bool)

    updated, magnitude = manager.apply_grads_batched(
        "cms_fast",
        params,
        grads,
        context=context,
        sample_mask=mask,
        force=True,
    )
    assert magnitude > 0
    assert not torch.allclose(updated["w"][0], params["w"][0])
    assert torch.allclose(updated["w"][1], params["w"][1])
    assert not torch.allclose(updated["w"][2], params["w"][2])
    stats = manager.clock.stats()
    assert stats["cms_fast"].updates == 1


def test_apply_grads_batched_all_inactive_is_noop() -> None:
    manager = _make_manager()
    params = {"w": torch.zeros(2, 2)}
    grads = {"w": torch.ones(2, 2)}
    before = manager.clock.stats()["cms_fast"].updates

    updated, magnitude = manager.apply_grads_batched(
        "cms_fast",
        params,
        grads,
        sample_mask=torch.tensor([False, False], dtype=torch.bool),
        force=True,
    )
    assert magnitude == 0.0
    assert torch.allclose(updated["w"], params["w"])
    after = manager.clock.stats()["cms_fast"].updates
    assert after == before
