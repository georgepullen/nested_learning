from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from nested_learning.training import run_training_loop


def _base_cfg(log_path: Path, *, grad_accum_steps: int, online_chunk_size: int = 0):
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
                "batch_size": 1,
                "num_workers": 0,
            },
            "train": {
                "steps": 4,
                "log_interval": 1,
                "device": "cpu",
                "seed": 2026,
                "deterministic": True,
                "online_updates": False,
                "online_chunk_size": online_chunk_size,
                "per_layer_teach_signal": False,
                "use_fast_state": False,
                "fast_state_batch_mode": "shared",
                "grad_accum_steps": grad_accum_steps,
                "mixed_precision": {"enabled": False, "dtype": "bf16"},
                "compile": {"enable": False},
                "checkpoint": {"enable": False},
            },
            "optim": {"type": "adamw", "lr": 3e-4, "fused": False},
            "logging": {"enabled": True, "backend": "json", "path": str(log_path)},
        }
    )


def _last_logged_row(log_path: Path) -> dict:
    payload = json.loads(log_path.read_text())
    rows = [row for row in payload if isinstance(row, dict) and isinstance(row.get("step"), int)]
    assert rows
    return rows[-1]


def test_grad_accum_steps_controls_optimizer_step_count(tmp_path: Path) -> None:
    log1 = tmp_path / "accum1.json"
    cfg1 = _base_cfg(log1, grad_accum_steps=1)
    metrics1 = run_training_loop(cfg1, device=torch.device("cpu"), distributed=False)
    row1 = _last_logged_row(log1)
    assert int(metrics1["optimizer_steps_total"]) == 4
    assert int(row1["optimizer_steps_total"]) == 4
    assert int(row1["grad_accum_steps"]) == 1

    log2 = tmp_path / "accum2.json"
    cfg2 = _base_cfg(log2, grad_accum_steps=2)
    metrics2 = run_training_loop(cfg2, device=torch.device("cpu"), distributed=False)
    row2 = _last_logged_row(log2)
    assert int(metrics2["optimizer_steps_total"]) == 2
    assert int(row2["optimizer_steps_total"]) == 2
    assert int(row2["grad_accum_steps"]) == 2


def test_online_chunk_size_one_raises_in_training_loop(tmp_path: Path) -> None:
    cfg = _base_cfg(tmp_path / "invalid.json", grad_accum_steps=1, online_chunk_size=1)
    cfg.train.online_updates = True
    cfg.train.use_fast_state = True
    cfg.train.fast_state_batch_mode = "shared"
    with pytest.raises(ValueError, match="online_chunk_size=1"):
        run_training_loop(cfg, device=torch.device("cpu"), distributed=False)
