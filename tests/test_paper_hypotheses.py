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
