from __future__ import annotations

import torch

from nested_learning.levels import LevelSpec
from nested_learning.model import HOPEModel, ModelConfig
from nested_learning.training import compute_teach_signal


def _tiny_attention_config() -> ModelConfig:
    titan = LevelSpec(name="titan", update_period=1, optimizer_key="titan_opt")
    cms = [LevelSpec(name="cms_fast", update_period=1, optimizer_key="cms_opt")]
    return ModelConfig(
        vocab_size=64,
        dim=32,
        num_layers=1,
        heads=4,
        block_variant="hope_attention",
        titan_level=titan,
        cms_levels=cms,
        optimizers=None,
        teach_scale=0.1,
    )


def _sample_cms_deltas(state, sample_idx: int) -> dict[str, dict[str, torch.Tensor]]:
    block = state.blocks[0]
    out: dict[str, dict[str, torch.Tensor]] = {}
    for level_name, store in block.cms_params.items():
        if isinstance(store, list):
            out[level_name] = {name: value.detach().clone() for name, value in store[sample_idx].items()}
        else:
            out[level_name] = {
                name: value[sample_idx].detach().clone() for name, value in store.items()
            }
    return out


def test_tensorized_cms_matches_listmode() -> None:
    torch.manual_seed(17)
    model = HOPEModel(_tiny_attention_config())
    tokens = torch.randint(0, model.config.vocab_size, (3, 12))

    list_state = model.init_fast_state(batch_size=tokens.size(0), fast_state_batch_mode="per_sample_list")
    tensor_state = model.init_fast_state(
        batch_size=tokens.size(0),
        fast_state_batch_mode="tensorized_cms",
    )

    with torch.no_grad():
        logits_list = model(tokens, fast_state=list_state)
        teach_list = compute_teach_signal(model, logits_list, tokens, normalization="per_sample")
        _ = model(tokens, teach_signal=teach_list, fast_state=list_state)
        logits_after_list = model(tokens, fast_state=list_state)

        logits_tensor = model(tokens, fast_state=tensor_state)
        teach_tensor = compute_teach_signal(model, logits_tensor, tokens, normalization="per_sample")
        _ = model(tokens, teach_signal=teach_tensor, fast_state=tensor_state)
        logits_after_tensor = model(tokens, fast_state=tensor_state)

    diff = (logits_after_list - logits_after_tensor).abs().float()
    assert diff.mean().item() < 5e-2
    assert diff.max().item() < 5e-1

    for sample_idx in range(tokens.size(0)):
        list_deltas = _sample_cms_deltas(list_state, sample_idx)
        tensor_deltas = _sample_cms_deltas(tensor_state, sample_idx)
        assert list_deltas.keys() == tensor_deltas.keys()
        for level_name in list_deltas:
            assert list_deltas[level_name].keys() == tensor_deltas[level_name].keys()
            for param_name in list_deltas[level_name]:
                assert torch.allclose(
                    list_deltas[level_name][param_name],
                    tensor_deltas[level_name][param_name],
                    atol=5e-4,
                    rtol=5e-4,
                )
