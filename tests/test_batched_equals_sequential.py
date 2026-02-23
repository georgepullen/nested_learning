import torch

from nested_learning.fast_state import BlockFastState
from nested_learning.levels import LevelSpec
from nested_learning.model import HOPEModel, ModelConfig
from nested_learning.training import compute_teach_signal


def _tiny_config() -> ModelConfig:
    titan = LevelSpec(name="titan", update_period=1, optimizer_key="titan_opt")
    cms = [LevelSpec(name="cms_fast", update_period=1, optimizer_key="cms_opt")]
    return ModelConfig(
        vocab_size=64,
        dim=32,
        num_layers=1,
        heads=4,
        titan_level=titan,
        cms_levels=cms,
        optimizers=None,
        teach_scale=0.1,
    )


def _checksum(params: dict[str, torch.Tensor]) -> float:
    total = 0.0
    for idx, (_, value) in enumerate(sorted(params.items())):
        total += float((idx + 1) * value.detach().float().mean().item())
    return total


def _state_signature(block_state: BlockFastState, sample_idx: int) -> dict[str, float]:
    out: dict[str, float] = {}
    titan = block_state.titan_params
    if isinstance(titan, list):
        out["titan"] = _checksum(titan[sample_idx])
    elif isinstance(titan, dict):
        out["titan"] = _checksum(titan)
    for level_name, value in block_state.cms_params.items():
        if isinstance(value, list):
            out[f"cms.{level_name}"] = _checksum(value[sample_idx])
        else:
            out[f"cms.{level_name}"] = _checksum(value)
    return out


def test_batched_equals_sequential() -> None:
    torch.manual_seed(33)
    model = HOPEModel(_tiny_config())
    tokens = torch.randint(0, model.config.vocab_size, (3, 12))

    state_batched = model.init_fast_state(
        batch_size=tokens.size(0),
        fast_state_batch_mode="per_sample_list",
    )
    with torch.no_grad():
        logits_before = model(tokens, fast_state=state_batched)
        teach_batched = compute_teach_signal(
            model,
            logits_before,
            tokens,
            normalization="per_sample",
        )
        _ = model(tokens, teach_signal=teach_batched, fast_state=state_batched)
        logits_after_batched = model(tokens, fast_state=state_batched)

    seq_logits_after = []
    seq_signatures = []
    for sample_idx in range(tokens.size(0)):
        sample = tokens[sample_idx : sample_idx + 1]
        state_single = model.init_fast_state(batch_size=1, fast_state_batch_mode="shared")
        with torch.no_grad():
            logits_single = model(sample, fast_state=state_single)
            teach_single = compute_teach_signal(
                model,
                logits_single,
                sample,
                normalization="per_sample",
            )
            _ = model(sample, teach_signal=teach_single, fast_state=state_single)
            logits_after_single = model(sample, fast_state=state_single)
        seq_logits_after.append(logits_after_single)
        seq_signatures.append(_state_signature(state_single.blocks[0], sample_idx=0))

    seq_logits_after_stacked = torch.cat(seq_logits_after, dim=0)
    diff = (logits_after_batched - seq_logits_after_stacked).abs().float()
    assert diff.mean().item() < 5e-2
    assert diff.max().item() < 5e-1

    for sample_idx in range(tokens.size(0)):
        batched_sig = _state_signature(state_batched.blocks[0], sample_idx=sample_idx)
        for key, value in batched_sig.items():
            assert torch.isclose(
                torch.tensor(value),
                torch.tensor(seq_signatures[sample_idx][key]),
                atol=2e-3,
            )
