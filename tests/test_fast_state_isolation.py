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


def _block_checksums(state: BlockFastState, sample_idx: int) -> dict[str, float]:
    payload: dict[str, float] = {}
    titan = state.titan_params
    if isinstance(titan, list):
        payload["titan"] = _checksum(titan[sample_idx])
    elif isinstance(titan, dict):
        payload["titan"] = _checksum(titan)
    for level_name, store in state.cms_params.items():
        if isinstance(store, list):
            payload[f"cms.{level_name}"] = _checksum(store[sample_idx])
        else:
            payload[f"cms.{level_name}"] = _checksum(store)
    return payload


def _run_once(model: HOPEModel, tokens: torch.Tensor) -> BlockFastState:
    state = model.init_fast_state(batch_size=tokens.size(0), fast_state_batch_mode="per_sample_list")
    with torch.no_grad():
        logits = model(tokens, fast_state=state)
        teach = compute_teach_signal(model, logits, tokens, normalization="per_sample")
        _ = model(tokens, teach_signal=teach, fast_state=state)
    return state.blocks[0]


def test_fast_state_isolation() -> None:
    torch.manual_seed(12)
    model = HOPEModel(_tiny_config())
    tokens = torch.randint(0, model.config.vocab_size, (2, 12))
    perturbed = tokens.clone()
    perturbed[0] = torch.randint(0, model.config.vocab_size, (tokens.size(1),))

    state_a = _run_once(model, tokens)
    state_b = _run_once(model, perturbed)

    sample_0_a = _block_checksums(state_a, sample_idx=0)
    sample_0_b = _block_checksums(state_b, sample_idx=0)
    sample_1_a = _block_checksums(state_a, sample_idx=1)
    sample_1_b = _block_checksums(state_b, sample_idx=1)

    for key in sample_1_a:
        assert torch.isclose(torch.tensor(sample_1_a[key]), torch.tensor(sample_1_b[key]), atol=1e-6)
    changed = any(abs(sample_0_a[k] - sample_0_b[k]) > 1e-8 for k in sample_0_a)
    assert changed
