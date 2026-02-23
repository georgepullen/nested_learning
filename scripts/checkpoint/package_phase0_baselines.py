#!/usr/bin/env python
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import typer

from nested_learning.training import verify_checkpoint_integrity

app = typer.Typer(add_completion=False, help="Package and checksum Phase 0 baseline artifacts.")
_EVAL_KINDS = ("zeroshot", "niah", "continual", "passkey", "pg19")


@dataclass(frozen=True)
class VariantInputs:
    tag: str
    checkpoint: Path
    log_path: Path
    train_command: str


def _checksum(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _sidecars(checkpoint: Path) -> list[Path]:
    stem = checkpoint.with_suffix("")
    candidates = [
        Path(f"{stem}.sha256"),
        Path(f"{stem}.meta.json"),
        Path(f"{stem}.yaml"),
    ]
    return [path for path in candidates if path.exists()]


def _load_last_step_metrics(log_path: Path) -> dict[str, Any]:
    payload = json.loads(log_path.read_text())
    if not isinstance(payload, list):
        raise ValueError(f"Expected JSON list in {log_path}")
    rows = [row for row in payload if isinstance(row, dict) and isinstance(row.get("step"), int)]
    train_rows = [row for row in rows if int(row["step"]) >= 0]
    if not train_rows:
        return {}
    latest = max(train_rows, key=lambda row: int(row["step"]))
    keys = ("step", "loss", "ppl", "teach_signal_norm", "surprise_metric", "surprise_value")
    return {key: latest[key] for key in keys if key in latest}


def _collect_evals(eval_dir: Path, tag: str) -> dict[str, Path]:
    payload: dict[str, Path] = {}
    missing: list[str] = []
    for kind in _EVAL_KINDS:
        path = eval_dir / f"{kind}_{tag}.json"
        if not path.exists():
            missing.append(str(path))
            continue
        payload[kind] = path
    if missing:
        raise FileNotFoundError(
            "Missing expected eval outputs for "
            f"{tag}:\n- "
            + "\n- ".join(missing)
        )
    return payload


def _stage_variant(
    variant: VariantInputs,
    *,
    eval_dir: Path,
    bundle_root: Path,
) -> dict[str, Any]:
    verify_meta = verify_checkpoint_integrity(variant.checkpoint)
    if not variant.log_path.exists():
        raise FileNotFoundError(f"Missing training log: {variant.log_path}")

    evals = _collect_evals(eval_dir, variant.tag)
    variant_dir = bundle_root / variant.tag
    copied: list[Path] = []

    ckpt_dst = variant_dir / "checkpoint.pt"
    _copy(variant.checkpoint, ckpt_dst)
    copied.append(ckpt_dst)

    for sidecar in _sidecars(variant.checkpoint):
        dst = variant_dir / sidecar.name
        _copy(sidecar, dst)
        copied.append(dst)

    log_dst = variant_dir / "metrics.json"
    _copy(variant.log_path, log_dst)
    copied.append(log_dst)

    eval_manifest: dict[str, dict[str, str]] = {}
    for kind, src in evals.items():
        dst = variant_dir / "eval" / src.name
        _copy(src, dst)
        copied.append(dst)
        eval_manifest[kind] = {
            "source": str(src),
            "bundle_path": str(dst),
            "sha256": _checksum(dst),
        }

    files_manifest: dict[str, str] = {
        str(path.relative_to(bundle_root)): _checksum(path) for path in copied
    }

    return {
        "tag": variant.tag,
        "train_command": variant.train_command,
        "checkpoint": str(variant.checkpoint),
        "checkpoint_step": verify_meta.get("step"),
        "log": str(variant.log_path),
        "latest_metrics": _load_last_step_metrics(variant.log_path),
        "eval": eval_manifest,
        "files": files_manifest,
    }


def _write_report(report_path: Path, bundle_root: Path, manifest: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# Phase 0 Baseline Table")
    lines.append("")
    lines.append(f"Bundle: `{bundle_root}`")
    lines.append("")
    lines.append("## Canonical Commands")
    lines.append("")
    for variant in manifest["variants"]:
        lines.append(f"- `{variant['tag']}`: `{variant['train_command']}`")
    lines.append("")
    lines.append("## Metrics Snapshot")
    lines.append("")
    lines.append("| Variant | Step | Loss | PPL | Teach Norm | Surprise Metric | Surprise Value |")
    lines.append("|---|---:|---:|---:|---:|---|---:|")
    for variant in manifest["variants"]:
        metrics = variant.get("latest_metrics", {})
        lines.append(
            "| {tag} | {step} | {loss} | {ppl} | {teach} | {sm} | {sv} |".format(
                tag=variant["tag"],
                step=metrics.get("step", "n/a"),
                loss=metrics.get("loss", "n/a"),
                ppl=metrics.get("ppl", "n/a"),
                teach=metrics.get("teach_signal_norm", "n/a"),
                sm=metrics.get("surprise_metric", "n/a"),
                sv=metrics.get("surprise_value", "n/a"),
            )
        )
    lines.append("")
    lines.append("## Eval Provenance")
    lines.append("")
    for variant in manifest["variants"]:
        lines.append(f"### {variant['tag']}")
        for kind, payload in variant["eval"].items():
            lines.append(
                f"- `{kind}`: `{payload['source']}` -> `{payload['bundle_path']}` "
                f"(sha256 `{payload['sha256']}`)"
            )
        lines.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines).strip() + "\n")


@app.command()
def main(
    selfmod_checkpoint: Path = typer.Option(..., help="Path to phase0 selfmod checkpoint (.pt)."),
    attention_checkpoint: Path = typer.Option(
        ..., help="Path to phase0 attention checkpoint (.pt)."
    ),
    selfmod_log: Path = typer.Option(..., help="Path to phase0 selfmod metrics JSON log."),
    attention_log: Path = typer.Option(..., help="Path to phase0 attention metrics JSON log."),
    selfmod_train_command: str = typer.Option(..., help="Exact command used for selfmod baseline."),
    attention_train_command: str = typer.Option(
        ..., help="Exact command used for attention baseline."
    ),
    eval_dir: Path = typer.Option(Path("eval"), help="Directory containing eval JSON outputs."),
    output_dir: Path = typer.Option(
        Path("artifacts/phase0_baselines"), help="Output root for baseline bundles."
    ),
    report: Path = typer.Option(
        Path("reports/phase0_baseline_table.md"),
        help="Markdown report output path.",
    ),
    bundle_id: str = typer.Option(
        "",
        help="Optional bundle ID (default is UTC timestamp).",
    ),
) -> None:
    bundle_name = bundle_id.strip() or datetime.now(UTC).strftime("phase0_%Y%m%dT%H%M%SZ")
    bundle_root = output_dir / bundle_name
    bundle_root.mkdir(parents=True, exist_ok=True)

    variants = [
        VariantInputs(
            tag="phase0_selfmod",
            checkpoint=selfmod_checkpoint,
            log_path=selfmod_log,
            train_command=selfmod_train_command,
        ),
        VariantInputs(
            tag="phase0_attention",
            checkpoint=attention_checkpoint,
            log_path=attention_log,
            train_command=attention_train_command,
        ),
    ]

    manifest_variants: list[dict[str, Any]] = []
    for variant in variants:
        manifest_variants.append(
            _stage_variant(variant, eval_dir=eval_dir, bundle_root=bundle_root)
        )

    manifest = {
        "bundle_id": bundle_name,
        "bundle_root": str(bundle_root),
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "variants": manifest_variants,
    }

    manifest_path = bundle_root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    _write_report(report, bundle_root, manifest)

    typer.echo(f"[phase0] wrote bundle: {bundle_root}")
    typer.echo(f"[phase0] wrote manifest: {manifest_path}")
    typer.echo(f"[phase0] wrote report: {report}")


if __name__ == "__main__":
    app()
