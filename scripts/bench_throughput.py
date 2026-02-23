#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from hashlib import sha256
from pathlib import Path
from typing import Any

import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from nested_learning.device import resolve_device
from nested_learning.training import run_training_loop, unwrap_config


def _load_cfg(config_name: str, overrides: list[str]) -> Any:
    root = Path(__file__).resolve().parents[1]
    config_dir = root / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name=config_name, overrides=overrides)
    return unwrap_config(cfg)


def _clone_cfg(cfg: Any) -> Any:
    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))


def _query_gpu() -> dict[str, str]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,temperature.gpu,clocks.sm,clocks.mem,power.draw,utilization.gpu,utilization.memory,memory.used",
        "--format=csv,noheader,nounits",
    ]
    try:
        out = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except (subprocess.SubprocessError, FileNotFoundError):
        return {}
    line = out.stdout.strip().splitlines()
    if not line:
        return {}
    values = [item.strip() for item in line[0].split(",")]
    keys = [
        "name",
        "driver_version",
        "temperature_c",
        "clock_sm_mhz",
        "clock_mem_mhz",
        "power_draw_w",
        "utilization_gpu_pct",
        "utilization_mem_pct",
        "memory_used_mib",
    ]
    return dict(zip(keys, values))


def _sync_cuda(device: torch.device) -> None:
    if device.type != "cuda" or not torch.cuda.is_available():
        return
    idx = device.index if device.index is not None else torch.cuda.current_device()
    torch.cuda.synchronize(idx)


def _trial(cfg: Any, device: torch.device, warmup_steps: int, measure_steps: int, sync_cuda: bool) -> dict[str, Any]:
    trial_cfg = _clone_cfg(cfg)
    total_steps = warmup_steps + measure_steps
    trial_cfg.train.steps = total_steps
    trial_cfg.train.log_interval = max(total_steps + 1, 10_000)
    trial_cfg.logging.enabled = False
    if "checkpoint" not in trial_cfg.train:
        trial_cfg.train.checkpoint = OmegaConf.create({})
    trial_cfg.train.checkpoint.enable = False

    tokens_per_step = int(trial_cfg.data.batch_size) * int(trial_cfg.data.get("seq_len", 0) or 0)
    markers: dict[str, float] = {}

    if sync_cuda:
        _sync_cuda(device)
    if warmup_steps == 0:
        markers["t0"] = time.perf_counter()

    gpu_start = _query_gpu()

    cuda_device: int | None = None
    if device.type == "cuda" and torch.cuda.is_available():
        cuda_device = device.index if device.index is not None else torch.cuda.current_device()
        torch.cuda.reset_peak_memory_stats(cuda_device)

    def _on_step_end(step: int) -> None:
        if step == warmup_steps - 1:
            if sync_cuda:
                _sync_cuda(device)
            markers["t0"] = time.perf_counter()
        if step == (warmup_steps + measure_steps - 1):
            if sync_cuda:
                _sync_cuda(device)
            markers["t1"] = time.perf_counter()

    final_metrics = run_training_loop(
        trial_cfg,
        device=device,
        distributed=False,
        step_end_callback=_on_step_end,
    )

    if sync_cuda:
        _sync_cuda(device)
    markers.setdefault("t1", time.perf_counter())
    markers.setdefault("t0", markers["t1"])

    gpu_end = _query_gpu()

    peak_vram_bytes = 0
    if cuda_device is not None:
        peak_vram_bytes = int(torch.cuda.max_memory_allocated(cuda_device))

    elapsed = max(markers["t1"] - markers["t0"], 1e-9)
    measured_tokens = measure_steps * tokens_per_step

    return {
        "warmup_steps": warmup_steps,
        "measure_steps": measure_steps,
        "elapsed_seconds": elapsed,
        "steps_per_second": measure_steps / elapsed,
        "tokens_per_second": (measured_tokens / elapsed) if measured_tokens > 0 else 0.0,
        "peak_vram_bytes": peak_vram_bytes,
        "peak_vram_gb": peak_vram_bytes / (1024**3),
        "gpu_start": gpu_start,
        "gpu_end": gpu_end,
        "final_metrics": final_metrics,
    }


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    if len(values) == 1:
        return values[0], 0.0
    return statistics.fmean(values), statistics.stdev(values)


def _git_sha() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.SubprocessError:
        return None
    return out.stdout.strip() or None


def main() -> None:
    parser = argparse.ArgumentParser(description="Standardized throughput benchmark for fast-state runs.")
    parser.add_argument("--config", default="bench_micro_200", help="Hydra config name")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Hydra override (repeatable), e.g. --override data.batch_size=8",
    )
    parser.add_argument("--trials", type=int, default=3, help="Number of measured trials")
    parser.add_argument("--warmup_steps", type=int, default=20, help="Warmup steps excluded from timing")
    parser.add_argument("--measure_steps", type=int, default=200, help="Measured steps")
    parser.add_argument(
        "--sync_cuda",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Synchronize CUDA before/after timing windows",
    )
    parser.add_argument(
        "--output",
        default="logs/bench_throughput_metrics.json",
        help="Where to write the throughput JSON payload",
    )
    parser.add_argument(
        "--resolved-config-out",
        default="logs/bench_throughput_resolved.yaml",
        help="Where to write the resolved config",
    )
    args = parser.parse_args()

    cfg = _load_cfg(args.config, args.override)
    device = resolve_device(cfg.train.device)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_cfg_path = Path(args.resolved_config_out)
    resolved_cfg_path.parent.mkdir(parents=True, exist_ok=True)

    resolved_yaml = OmegaConf.to_yaml(cfg, resolve=True)
    resolved_cfg_path.write_text(resolved_yaml)

    trial_results: list[dict[str, Any]] = []
    for idx in range(max(1, int(args.trials))):
        trial = _trial(
            cfg,
            device,
            warmup_steps=max(0, int(args.warmup_steps)),
            measure_steps=max(1, int(args.measure_steps)),
            sync_cuda=bool(args.sync_cuda),
        )
        trial["trial"] = idx
        trial_results.append(trial)

    steps_vals = [float(t["steps_per_second"]) for t in trial_results]
    tokens_vals = [float(t["tokens_per_second"]) for t in trial_results]
    vram_vals = [float(t["peak_vram_gb"]) for t in trial_results]

    mean_steps, std_steps = _mean_std(steps_vals)
    mean_tokens, std_tokens = _mean_std(tokens_vals)
    mean_vram, std_vram = _mean_std(vram_vals)

    payload = {
        "config": args.config,
        "overrides": args.override,
        "trials": trial_results,
        "warmup_steps": int(args.warmup_steps),
        "measure_steps": int(args.measure_steps),
        "sync_cuda": bool(args.sync_cuda),
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "git_sha": _git_sha(),
        "config_hash": sha256(resolved_yaml.encode("utf-8")).hexdigest(),
        "mean": {
            "steps_per_second": mean_steps,
            "tokens_per_second": mean_tokens,
            "peak_vram_gb": mean_vram,
        },
        "std": {
            "steps_per_second": std_steps,
            "tokens_per_second": std_tokens,
            "peak_vram_gb": std_vram,
        },
        # Backward-compatible top-level fields used by prior reports/scripts.
        "steps_per_second": mean_steps,
        "tokens_per_second": mean_tokens,
        "peak_vram_gb": mean_vram,
    }

    output_path.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
