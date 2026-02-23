from pathlib import Path

import torch
from omegaconf import OmegaConf

from nested_learning.training import run_training_loop


def _base_cfg(checkpoint_dir: Path, log_path: Path):
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
                "steps": 3,
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
                "checkpoint": {
                    "enable": True,
                    "dir": str(checkpoint_dir),
                    "save_interval": 3,
                    "save_last": True,
                    "resume_path": None,
                },
            },
            "optim": {"type": "adamw", "lr": 3e-4, "fused": False},
            "logging": {"enabled": True, "backend": "json", "path": str(log_path)},
        }
    )


def test_training_loop_can_resume_from_checkpoint(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "ckpt"
    log1 = tmp_path / "log_phase1.json"
    cfg1 = _base_cfg(checkpoint_dir, log1)
    run_training_loop(cfg1, device=torch.device("cpu"), distributed=False)

    ckpt3 = checkpoint_dir / "step_000003.pt"
    assert ckpt3.exists()

    log2 = tmp_path / "log_phase2.json"
    cfg2 = _base_cfg(checkpoint_dir, log2)
    cfg2.train.steps = 5
    cfg2.train.checkpoint.resume_path = str(ckpt3)
    cfg2.train.checkpoint.save_interval = 5

    run_training_loop(cfg2, device=torch.device("cpu"), distributed=False)

    assert (checkpoint_dir / "step_000005.pt").exists()
