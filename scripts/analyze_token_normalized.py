#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


@dataclass
class RunSeries:
    label: str
    run_path: Path
    points: list[tuple[int, float, int]]  # (step, loss, tokens_seen_total)
    used_fallback_tokens: bool



def _load_metrics(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict) and isinstance(payload.get("records"), list):
        records = payload["records"]
    else:
        raise ValueError(f"Unsupported metrics format in {path}")
    return [r for r in records if isinstance(r, dict)]



def _load_tokens_per_step_from_cfg(path: Path | None) -> int | None:
    if path is None:
        return None
    cfg = OmegaConf.load(path)
    batch_size = cfg.get("data", {}).get("batch_size")
    seq_len = cfg.get("data", {}).get("seq_len")
    if batch_size is None or seq_len is None:
        return None
    try:
        return int(batch_size) * int(seq_len)
    except (TypeError, ValueError):
        return None



def _extract_points(records: list[dict[str, Any]], default_tokens_per_step: int | None) -> tuple[list[tuple[int, float, int]], bool]:
    points: list[tuple[int, float, int]] = []
    used_fallback = False
    for row in records:
        step = row.get("step")
        loss = row.get("loss")
        if not isinstance(step, int) or not isinstance(loss, (int, float)):
            continue
        tokens_seen = row.get("tokens_seen_total")
        if isinstance(tokens_seen, (int, float)):
            token_count = int(tokens_seen)
        else:
            fallback_tokens = row.get("tokens_per_step")
            if isinstance(fallback_tokens, (int, float)):
                tokens_per_step = int(fallback_tokens)
            else:
                tokens_per_step = default_tokens_per_step
            if tokens_per_step is None or tokens_per_step <= 0:
                raise ValueError(
                    "tokens_seen_total missing and no usable tokens_per_step/config fallback for "
                    f"step={step}"
                )
            token_count = (step + 1) * tokens_per_step
            used_fallback = True
        points.append((step, float(loss), token_count))
    points.sort(key=lambda x: x[2])
    return points, used_fallback



def _interp(points: list[tuple[int, float, int]], milestone: int) -> float | None:
    if not points:
        return None
    if milestone < points[0][2] or milestone > points[-1][2]:
        return None
    for idx, (_, loss, tok) in enumerate(points):
        if tok == milestone:
            return loss
        if tok > milestone and idx > 0:
            _, prev_loss, prev_tok = points[idx - 1]
            span = tok - prev_tok
            if span <= 0:
                return loss
            ratio = (milestone - prev_tok) / span
            return prev_loss + ratio * (loss - prev_loss)
    return points[-1][1]



def _default_milestones(series: list[RunSeries]) -> list[int]:
    if not series:
        return []
    max_common = min(s.points[-1][2] for s in series if s.points)
    if max_common <= 0:
        return []
    step = 50_000
    milestones = list(range(step, max_common + 1, step))
    if not milestones:
        milestones = [max_common]
    return milestones



def main() -> None:
    parser = argparse.ArgumentParser(description="Token-normalized training run comparison.")
    parser.add_argument("--runs", nargs="+", required=True, help="Metrics JSON files")
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Labels aligned with --runs (default: file stem)",
    )
    parser.add_argument(
        "--resolved-configs",
        nargs="*",
        default=None,
        help="Optional resolved YAML files aligned with --runs (for token fallback)",
    )
    parser.add_argument(
        "--milestones",
        nargs="*",
        type=int,
        default=None,
        help="Token milestones for interpolation (default: every 50k up to common max)",
    )
    parser.add_argument("--out_dir", required=True, help="Output directory")
    args = parser.parse_args()

    run_paths = [Path(p) for p in args.runs]
    labels = args.labels if args.labels is not None else [p.stem for p in run_paths]
    if len(labels) != len(run_paths):
        raise ValueError("--labels must have the same count as --runs")

    resolved_cfgs: list[Path | None]
    if args.resolved_configs is None:
        resolved_cfgs = [None] * len(run_paths)
    else:
        if len(args.resolved_configs) not in {0, len(run_paths)}:
            raise ValueError("--resolved-configs must be omitted, empty, or aligned with --runs")
        if len(args.resolved_configs) == 0:
            resolved_cfgs = [None] * len(run_paths)
        else:
            resolved_cfgs = [Path(p) for p in args.resolved_configs]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    series: list[RunSeries] = []
    for label, run_path, cfg_path in zip(labels, run_paths, resolved_cfgs):
        records = _load_metrics(run_path)
        fallback_tokens_per_step = _load_tokens_per_step_from_cfg(cfg_path)
        points, used_fallback = _extract_points(records, fallback_tokens_per_step)
        if not points:
            raise ValueError(f"No step/loss records found in {run_path}")
        series.append(
            RunSeries(
                label=label,
                run_path=run_path,
                points=points,
                used_fallback_tokens=used_fallback,
            )
        )

    csv_path = out_dir / "loss_vs_tokens.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["label", "step", "tokens_seen_total", "loss"],
        )
        writer.writeheader()
        for run in series:
            for step, loss, tokens in run.points:
                writer.writerow(
                    {
                        "label": run.label,
                        "step": step,
                        "tokens_seen_total": tokens,
                        "loss": loss,
                    }
                )

    milestones = args.milestones if args.milestones else _default_milestones(series)
    summary_lines: list[str] = []
    summary_lines.append("# Token-Normalized Summary")
    summary_lines.append("")
    summary_lines.append("## Runs")
    summary_lines.append("")
    for run in series:
        first_step, first_loss, first_tokens = run.points[0]
        last_step, last_loss, last_tokens = run.points[-1]
        summary_lines.append(f"- `{run.label}`")
        summary_lines.append(f"  run: `{run.run_path}`")
        summary_lines.append(f"  first: step={first_step}, tokens={first_tokens}, loss={first_loss:.6f}")
        summary_lines.append(f"  last: step={last_step}, tokens={last_tokens}, loss={last_loss:.6f}")
        summary_lines.append(
            "  token source: "
            + ("fallback (step*tokens_per_step)" if run.used_fallback_tokens else "tokens_seen_total")
        )
    summary_lines.append("")

    summary_lines.append("## Loss At Token Milestones")
    summary_lines.append("")
    if milestones:
        header = ["tokens"] + [run.label for run in series]
        summary_lines.append("| " + " | ".join(header) + " |")
        summary_lines.append("|" + "|".join(["---"] * len(header)) + "|")
        for milestone in milestones:
            row = [f"{milestone}"]
            for run in series:
                val = _interp(run.points, milestone)
                row.append("n/a" if val is None else f"{val:.6f}")
            summary_lines.append("| " + " | ".join(row) + " |")
    else:
        summary_lines.append("No common milestones available across runs.")

    summary_path = out_dir / "summary.md"
    summary_path.write_text("\n".join(summary_lines) + "\n")

    print(f"wrote {csv_path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
