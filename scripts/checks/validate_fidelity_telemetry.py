#!/usr/bin/env python
from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Optional

import typer

app = typer.Typer(
    add_completion=False,
    help="Validate that training telemetry includes paper-fidelity keys and per-layer CMS stats.",
)

_REQUIRED_SCALARS = ("loss", "ppl", "teach_signal_norm")
_REQUIRED_LAYER_METRICS = {"grad_norm", "chunk_tokens", "gate_hit"}
_LAYER_METRIC_RE = re.compile(r"^layer(?P<layer>\d+)\.cms\.(?P<level>[^.]+)\.(?P<metric>[^.]+)$")
_ALLOWED_SURPRISE_METRICS = {"l2", "loss", "logit_entropy"}


def _is_finite_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    return False


def _load_records(path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as err:
        raise typer.BadParameter(f"Invalid JSON in {path}: {err}") from err
    if not isinstance(payload, list):
        raise typer.BadParameter(f"Telemetry file must be a JSON list: {path}")
    records: list[dict[str, Any]] = []
    for idx, row in enumerate(payload):
        if not isinstance(row, dict):
            raise typer.BadParameter(f"Telemetry row {idx} in {path} is not an object")
        records.append(row)
    return records


def _train_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    train_rows: list[dict[str, Any]] = []
    for row in records:
        step = row.get("step")
        if isinstance(step, int) and step >= 0:
            train_rows.append(row)
    return train_rows


def _infer_gating_enabled(records: list[dict[str, Any]]) -> bool:
    for row in records:
        step = row.get("step")
        if step != -1:
            continue
        threshold = row.get("model.surprise_threshold")
        if threshold is None:
            continue
        if isinstance(threshold, (int, float)):
            return True
        return True
    return False


def validate_records(
    records: list[dict[str, Any]],
    *,
    gating_enabled: Optional[bool] = None,
) -> list[str]:
    errors: list[str] = []
    train_rows = _train_records(records)
    if not train_rows:
        return ["No training telemetry rows found (need at least one row with step >= 0)."]

    gate_required = _infer_gating_enabled(records) if gating_enabled is None else gating_enabled
    layer_metrics_seen: dict[tuple[str, str], set[str]] = {}

    for idx, row in enumerate(train_rows):
        for key in _REQUIRED_SCALARS:
            if key not in row:
                errors.append(f"step_row[{idx}] missing required key '{key}'")
                continue
            if not _is_finite_number(row[key]):
                errors.append(f"step_row[{idx}] key '{key}' must be a finite number")

        surprise_metric = row.get("surprise_metric")
        surprise_value = row.get("surprise_value")
        if surprise_metric is not None and str(surprise_metric) not in _ALLOWED_SURPRISE_METRICS:
            errors.append(
                f"step_row[{idx}] has invalid surprise_metric '{surprise_metric}' "
                f"(expected one of {sorted(_ALLOWED_SURPRISE_METRICS)})"
            )
        if surprise_value is not None and not _is_finite_number(surprise_value):
            errors.append(f"step_row[{idx}] key 'surprise_value' must be a finite number")
        if gate_required:
            if surprise_metric is None:
                errors.append(
                    f"step_row[{idx}] missing 'surprise_metric' while gating is enabled"
                )
            if surprise_value is None:
                errors.append(f"step_row[{idx}] missing 'surprise_value' while gating is enabled")

        for key, value in row.items():
            match = _LAYER_METRIC_RE.match(key)
            if not match:
                continue
            level_key = (match.group("layer"), match.group("level"))
            metric = match.group("metric")
            layer_metrics_seen.setdefault(level_key, set()).add(metric)
            if metric in _REQUIRED_LAYER_METRICS and not _is_finite_number(value):
                errors.append(f"step_row[{idx}] key '{key}' must be a finite number")

    if not layer_metrics_seen:
        errors.append(
            "No per-layer CMS metrics found (expected keys like "
            "layer0.cms.cms_fast.grad_norm/chunk_tokens/gate_hit)."
        )
    else:
        for level_key, metrics in sorted(layer_metrics_seen.items()):
            missing = _REQUIRED_LAYER_METRICS.difference(metrics)
            if missing:
                layer, level = level_key
                errors.append(
                    f"layer{layer}.cms.{level} missing required metrics: {sorted(missing)}"
                )

    return errors


@app.command()
def main(
    log: list[Path] = typer.Option(
        ...,
        "--log",
        help="Path to JSON telemetry log written by logging.backend=json. Repeatable.",
    ),
    gating_enabled: Optional[bool] = typer.Option(
        None,
        "--gating-enabled/--no-gating-enabled",
        help=(
            "Override surprise-gating expectation. By default this is inferred from "
            "step=-1 run_features model.surprise_threshold."
        ),
    ),
) -> None:
    any_failures = False
    for path in log:
        if not path.exists():
            raise typer.BadParameter(f"Telemetry log not found: {path}")
        records = _load_records(path)
        errors = validate_records(records, gating_enabled=gating_enabled)
        if errors:
            any_failures = True
            typer.echo(f"[telemetry] FAIL: {path}")
            for err in errors:
                typer.echo(f"  - {err}")
        else:
            typer.echo(f"[telemetry] PASS: {path}")

    if any_failures:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
