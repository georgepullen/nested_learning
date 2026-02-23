import json
from pathlib import Path

from nested_learning.logging_utils import JSONLogger


def test_json_logger_flushes_on_each_log(tmp_path: Path) -> None:
    log_path = tmp_path / "metrics.json"
    logger = JSONLogger(log_path)

    logger.log({"loss": 1.25}, step=0)
    assert log_path.exists()
    payload = json.loads(log_path.read_text())
    assert payload == [{"step": 0, "loss": 1.25}]

    logger.log({"loss": 0.75}, step=1)
    payload = json.loads(log_path.read_text())
    assert payload == [{"step": 0, "loss": 1.25}, {"step": 1, "loss": 0.75}]


def test_json_logger_appends_when_log_file_exists(tmp_path: Path) -> None:
    log_path = tmp_path / "metrics.json"
    log_path.write_text(json.dumps([{"step": 5, "loss": 3.0}], indent=2))

    logger = JSONLogger(log_path)
    logger.log({"loss": 2.0}, step=6)

    payload = json.loads(log_path.read_text())
    assert payload == [{"step": 5, "loss": 3.0}, {"step": 6, "loss": 2.0}]
