import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.checks.validate_fidelity_telemetry import validate_records


def _load_fixture(name: str):
    fixture = Path(__file__).resolve().parent / "data" / "telemetry" / name
    payload = json.loads(fixture.read_text())
    assert isinstance(payload, list)
    return payload


def test_fidelity_validator_accepts_passing_fixture() -> None:
    records = _load_fixture("passing_metrics.json")
    errors = validate_records(records)
    assert errors == []


def test_fidelity_validator_rejects_missing_layer_metrics() -> None:
    records = _load_fixture("failing_missing_layer_metrics.json")
    errors = validate_records(records)
    assert any("missing required metrics" in err for err in errors)


def test_fidelity_validator_requires_surprise_when_gated() -> None:
    records = _load_fixture("failing_missing_surprise_metrics.json")
    errors = validate_records(records)
    assert any("missing 'surprise_metric'" in err for err in errors)
    assert any("missing 'surprise_value'" in err for err in errors)
