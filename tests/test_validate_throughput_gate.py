from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _write_tps(path: Path, tps: float) -> None:
    path.write_text(json.dumps({"tokens_per_second": float(tps)}))


def test_validate_throughput_gate_passes(tmp_path: Path) -> None:
    script = Path("scripts/checks/validate_throughput_gate.py")
    list_json = tmp_path / "list.json"
    tensor_json = tmp_path / "tensor.json"
    b1_json = tmp_path / "b1.json"
    _write_tps(list_json, 100.0)
    _write_tps(tensor_json, 170.0)
    _write_tps(b1_json, 90.0)

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--list-json",
            str(list_json),
            "--tensorized-json",
            str(tensor_json),
            "--b1-json",
            str(b1_json),
            "--min-list-ratio",
            "1.5",
            "--min-b1-ratio",
            "1.8",
            "--label",
            "test",
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    assert "[throughput-gate] PASS" in result.stdout


def test_validate_throughput_gate_fails_when_ratio_below_threshold(tmp_path: Path) -> None:
    script = Path("scripts/checks/validate_throughput_gate.py")
    list_json = tmp_path / "list.json"
    tensor_json = tmp_path / "tensor.json"
    b1_json = tmp_path / "b1.json"
    _write_tps(list_json, 100.0)
    _write_tps(tensor_json, 120.0)
    _write_tps(b1_json, 90.0)

    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--list-json",
            str(list_json),
            "--tensorized-json",
            str(tensor_json),
            "--b1-json",
            str(b1_json),
            "--min-list-ratio",
            "1.5",
            "--min-b1-ratio",
            "1.2",
            "--label",
            "test",
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "ratio below threshold" in result.stderr
