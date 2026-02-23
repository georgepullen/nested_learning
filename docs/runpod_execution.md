# RunPod Execution Playbook

This playbook defines the reproducible RunPod workflow for this repo: pod provisioning, persistent storage, sync, checkpoint cadence, and resume drills.

## 1) Pod and storage contract

Create pods with:

- Persistent storage mounted at `/workspace`.
- SSH (`22/tcp`) enabled.
- Fixed image tag (do not use floating tags).

Recommended naming:

- Repo root: `/workspace/nested_learning`
- Checkpoints: `/workspace/nested_learning/artifacts/checkpoints/...`
- Logs: `/workspace/nested_learning/logs/...`
- Eval outputs: `/workspace/nested_learning/eval/...`

Use network volumes for persistence across pod lifecycle events.

## 2) Bootstrap a fresh pod

```bash
bash scripts/compute/runpod_bootstrap.sh /workspace/nested_learning
```

This installs/pins Python `3.12` by default (`PYTHON_VERSION` override supported), runs `uv sync --python 3.12 --all-extras --dev`, creates output folders, and prints PyTorch/CUDA runtime details.

## 3) Sync workflow

Push local workspace to pod:

```bash
bash scripts/compute/runpod_sync.sh push root@<host> <ssh_port> /workspace/nested_learning <local_repo_path>
```

If RunPod generated its own keypair, pass it explicitly:

```bash
RUNPOD_SSH_KEY=~/.runpod/ssh/RunPod-Key-Go \
  bash scripts/compute/runpod_sync.sh push root@<host> <ssh_port> /workspace/nested_learning <local_repo_path>
```

Pull artifacts/logs/eval/reports back:

```bash
bash scripts/compute/runpod_sync.sh pull root@<host> <ssh_port> /workspace/nested_learning <local_repo_path>
```

`runpod_sync.sh` prefers `rsync` when present and automatically falls back to tar-over-SSH when `rsync` is unavailable.

## 4) Training and checkpoint cadence

For interruption safety:

- On interruptible/spot pods: checkpoint every 100-500 steps.
- On on-demand pods: checkpoint every 500-2000 steps.
- Always keep `save_last=true`.
- Validate each published checkpoint with `scripts/checkpoint/verify.py`.

Phase-0 canonical run:

```bash
bash scripts/compute/run_phase0_baselines.sh
```

## 5) Forced stop/resume drill

Run synthetic resume drill end-to-end:

```bash
DEVICE=cuda:0 bash scripts/compute/runpod_resume_drill.sh
```

This creates a checkpoint, resumes from it, and validates both checkpoint integrity and telemetry.

Run validation manually for any run:

```bash
bash scripts/compute/runpod_validate.sh <checkpoint.pt> <metrics.json>
```

## 6) Pod lifecycle operations

RunPod API lifecycle operations for stop/start/restart/reset should be used instead of deleting pods during active experiments, so mounted volume state remains intact.

## 7) Quality gates before marking a run usable

- `scripts/checkpoint/verify.py` passes.
- `scripts/checks/validate_fidelity_telemetry.py` passes.
- No NaN/Inf in `loss`, `ppl`, `teach_signal_norm`, `surprise_value`.
- Required per-layer keys exist: `layer*.cms.*.{grad_norm,chunk_tokens,gate_hit}`.
- Resume drill executed at least once on the target image/runtime.

## References

- RunPod pod management: [runpod/docs pods/manage-pods.mdx](https://github.com/runpod/docs/blob/main/pods/manage-pods.mdx)
- RunPod network volumes: [runpod/docs pods/storage/create-network-volumes.mdx](https://github.com/runpod/docs/blob/main/pods/storage/create-network-volumes.mdx)
- RunPod storage types: [runpod/docs pods/storage/types.mdx](https://github.com/runpod/docs/blob/main/pods/storage/types.mdx)
- RunPod SSH/SCP: [runpod/docs pods/configuration/use-ssh.mdx](https://github.com/runpod/docs/blob/main/pods/configuration/use-ssh.mdx)
- RunPod lifecycle API examples: [Context7 /runpod/docs llms.txt](https://context7.com/runpod/docs/llms.txt)
- PyTorch AMP guidance: [pytorch/docs amp_examples.rst](https://github.com/pytorch/pytorch/blob/main/docs/source/notes/amp_examples.rst)
- PyTorch checkpoint pattern: [Context7 /pytorch/pytorch llms.txt](https://context7.com/pytorch/pytorch/llms.txt)
- PyTorch FSDP notes: [pytorch/docs distributed.fsdp.fully_shard.md](https://github.com/pytorch/pytorch/blob/main/docs/source/distributed.fsdp.fully_shard.md)
