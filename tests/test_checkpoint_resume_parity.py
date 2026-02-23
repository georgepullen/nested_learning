from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from nested_learning.training import run_training_loop


def _base_cfg(
    checkpoint_dir: Path,
    log_path: Path,
    *,
    batch_size: int,
    fast_state_batch_mode: str,
    steps: int,
    resume_path: str | None = None,
    save_interval: int | None = None,
):
    return OmegaConf.create(
        {
            "model": {
                "vocab_size": 64,
                "dim": 16,
                "num_layers": 1,
                "heads": 2,
                "block_variant": "hope_attention",
                "titan_level": {
                    "name": "titan",
                    "update_period": 1,
                    "optimizer_key": "titan_opt",
                },
                "cms_levels": [
                    {"name": "cms_fast", "update_period": 1, "optimizer_key": "cms_opt"}
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
                "batch_size": batch_size,
                "num_workers": 0,
            },
            "train": {
                "steps": steps,
                "log_interval": 1,
                "device": "cpu",
                "seed": 2026,
                "deterministic": True,
                "online_updates": True,
                "online_chunk_size": 0,
                "per_layer_teach_signal": False,
                "use_fast_state": True,
                "fast_state_batch_mode": fast_state_batch_mode,
                "teach_signal_normalization": "per_sample",
                "fail_if_paper_faithful_disabled": True,
                "mixed_precision": {"enabled": False, "dtype": "bf16"},
                "compile": {"enable": False},
                "checkpoint": {
                    "enable": True,
                    "dir": str(checkpoint_dir),
                    "save_interval": int(save_interval if save_interval is not None else steps),
                    "save_last": True,
                    "resume_path": resume_path,
                },
            },
            "optim": {"type": "adamw", "lr": 3e-4, "fused": False},
            "logging": {"enabled": True, "backend": "json", "path": str(log_path)},
        }
    )


def _loss_and_tokens(path: Path) -> tuple[list[float], list[int]]:
    payload = json.loads(path.read_text())
    losses: list[float] = []
    tokens: list[int] = []
    for row in payload:
        if not isinstance(row, dict):
            continue
        if not isinstance(row.get("step"), int) or row["step"] < 0:
            continue
        loss = row.get("loss")
        tok = row.get("tokens_seen_total")
        if isinstance(loss, (int, float)):
            losses.append(float(loss))
        if isinstance(tok, (int, float)):
            tokens.append(int(tok))
    return losses, tokens


def _load_model_state(ckpt_path: Path) -> dict[str, torch.Tensor]:
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert isinstance(state, dict)
    model_state = state.get("model")
    assert isinstance(model_state, dict)
    return model_state


@pytest.mark.parametrize(
    ("batch_size", "fast_state_batch_mode"),
    [
        (1, "shared"),
        (4, "per_sample_list"),
        (4, "tensorized_cms"),
    ],
)
def test_checkpoint_resume_parity(batch_size: int, fast_state_batch_mode: str, tmp_path: Path) -> None:
    total_steps = 8
    split_step = 4

    ckpt_full = tmp_path / "ckpt_full"
    ckpt_resume = tmp_path / "ckpt_resume"

    full_log = tmp_path / f"full_{fast_state_batch_mode}.json"
    phase1_log = tmp_path / f"phase1_{fast_state_batch_mode}.json"
    phase2_log = tmp_path / f"phase2_{fast_state_batch_mode}.json"

    cfg_full = _base_cfg(
        ckpt_full,
        full_log,
        batch_size=batch_size,
        fast_state_batch_mode=fast_state_batch_mode,
        steps=total_steps,
        save_interval=total_steps,
    )
    run_training_loop(cfg_full, device=torch.device("cpu"), distributed=False)

    split_ckpt = ckpt_resume / f"step_{split_step:06d}.pt"
    final_full_ckpt = ckpt_full / f"step_{total_steps:06d}.pt"
    final_resume_ckpt = ckpt_resume / f"step_{total_steps:06d}.pt"

    cfg_phase1 = _base_cfg(
        ckpt_resume,
        phase1_log,
        batch_size=batch_size,
        fast_state_batch_mode=fast_state_batch_mode,
        steps=split_step,
        save_interval=split_step,
    )
    run_training_loop(cfg_phase1, device=torch.device("cpu"), distributed=False)
    assert split_ckpt.exists()

    cfg_phase2 = _base_cfg(
        ckpt_resume,
        phase2_log,
        batch_size=batch_size,
        fast_state_batch_mode=fast_state_batch_mode,
        steps=total_steps,
        resume_path=str(split_ckpt),
        save_interval=total_steps,
    )
    run_training_loop(cfg_phase2, device=torch.device("cpu"), distributed=False)

    assert final_full_ckpt.exists()
    assert final_resume_ckpt.exists()

    full_losses, full_tokens = _loss_and_tokens(full_log)
    phase1_losses, phase1_tokens = _loss_and_tokens(phase1_log)
    phase2_losses, phase2_tokens = _loss_and_tokens(phase2_log)

    resumed_losses = phase1_losses + phase2_losses
    resumed_tokens = phase1_tokens + phase2_tokens

    assert len(full_losses) == len(resumed_losses)
    assert len(full_tokens) == len(resumed_tokens)

    for got, exp in zip(resumed_losses, full_losses):
        assert abs(got - exp) <= 1e-6
    for got, exp in zip(resumed_tokens, full_tokens):
        assert got == exp

    full_state = _load_model_state(final_full_ckpt)
    resumed_state = _load_model_state(final_resume_ckpt)

    assert full_state.keys() == resumed_state.keys()
    for key in full_state:
        assert torch.allclose(full_state[key], resumed_state[key], atol=1e-6, rtol=1e-6), key
