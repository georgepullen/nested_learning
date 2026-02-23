# Throughput Protocol

This protocol standardizes throughput measurements for fast-state modes and removes single-run noise.

## Hardware + Runtime Controls

- Run one benchmark job at a time per GPU.
- Keep GPU shape fixed for all compared runs.
- Use synthetic data config for throughput tests.
- Keep seed/config constant except the variable under test.

## Bench Script

Use `scripts/bench_throughput.py` with:

- `--trials` (default `3`)
- `--warmup_steps` (default `20`, excluded from timing)
- `--measure_steps` (default `200`)
- `--sync-cuda` (default enabled)

The script reports per-trial and aggregate:

- `steps_per_second`
- `tokens_per_second`
- `peak_vram_gb`
- `gpu_start` / `gpu_end` telemetry snapshot

## Example Commands

```bash
uv run python scripts/bench_throughput.py \
  --config bench_micro_200 \
  --override train.device=cuda:0 \
  --override data.batch_size=1 \
  --override train.fast_state_batch_mode=shared \
  --trials 3 --warmup_steps 20 --measure_steps 200 \
  --output logs/bench_b1_protocol.json \
  --resolved-config-out logs/bench_b1_protocol.resolved.yaml

uv run python scripts/bench_throughput.py \
  --config bench_micro_200 \
  --override train.device=cuda:0 \
  --override data.batch_size=4 \
  --override train.fast_state_batch_mode=per_sample_list \
  --trials 3 --warmup_steps 20 --measure_steps 200 \
  --output logs/bench_b4_list_protocol.json \
  --resolved-config-out logs/bench_b4_list_protocol.resolved.yaml
```

## Comparison Rules

- Compare **mean** and **std** across all trials, not single values.
- Reject claims from one-off runs when variance overlaps.
- Store resolved config YAML with each benchmark JSON.

