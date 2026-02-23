import pytest
from omegaconf import OmegaConf

from nested_learning.training import (
    _resolve_fast_state_batch_mode,
    _validate_fast_state_batch_semantics,
)


def test_fast_state_batch_semantics_raises_when_strict() -> None:
    cfg = OmegaConf.create(
        {
            "train": {
                "use_fast_state": True,
                "fail_if_paper_faithful_disabled": True,
                "fast_state_batch_mode": "shared",
            },
            "data": {"batch_size": 2},
        }
    )
    with pytest.raises(RuntimeError, match="fast-state"):
        _validate_fast_state_batch_semantics(cfg)


def test_fast_state_batch_semantics_allows_batch1() -> None:
    cfg = OmegaConf.create(
        {
            "train": {"use_fast_state": True, "fail_if_paper_faithful_disabled": True},
            "data": {"batch_size": 1},
        }
    )
    _validate_fast_state_batch_semantics(cfg)


def test_fast_state_batch_semantics_allows_per_sample_list_mode() -> None:
    cfg = OmegaConf.create(
        {
            "train": {
                "use_fast_state": True,
                "fail_if_paper_faithful_disabled": True,
                "fast_state_batch_mode": "per_sample_list",
            },
            "data": {"batch_size": 4},
        }
    )
    _validate_fast_state_batch_semantics(cfg)


def test_fast_state_batch_semantics_allows_tensorized_cms_mode() -> None:
    cfg = OmegaConf.create(
        {
            "train": {
                "use_fast_state": True,
                "fail_if_paper_faithful_disabled": True,
                "fast_state_batch_mode": "tensorized_cms",
            },
            "data": {"batch_size": 4},
        }
    )
    _validate_fast_state_batch_semantics(cfg)


def test_fast_state_batch_mode_auto_resolves_tensorized_for_batched_fast_state() -> None:
    cfg = OmegaConf.create(
        {
            "train": {"use_fast_state": True, "fast_state_batch_mode": "auto"},
            "data": {"batch_size": 4},
        }
    )
    assert _resolve_fast_state_batch_mode(cfg, batch_size=4) == "tensorized_cms"


def test_fast_state_batch_semantics_auto_allows_batched_fast_state() -> None:
    cfg = OmegaConf.create(
        {
            "train": {
                "use_fast_state": True,
                "fail_if_paper_faithful_disabled": True,
                "fast_state_batch_mode": "auto",
            },
            "data": {"batch_size": 4},
        }
    )
    _validate_fast_state_batch_semantics(cfg)
