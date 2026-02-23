import torch

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


def test_teach_signal_batch_equivalence() -> None:
    torch.manual_seed(7)
    cfg = _tiny_config()
    model = HOPEModel(cfg)
    tokens = torch.randint(0, cfg.vocab_size, (4, 9))
    logits = model(tokens)

    batched = compute_teach_signal(model, logits, tokens, normalization="per_sample")
    expected = []
    for idx in range(tokens.size(0)):
        expected.append(
            compute_teach_signal(
                model,
                logits[idx : idx + 1],
                tokens[idx : idx + 1],
                normalization="per_sample",
            )
        )
    expected_stacked = torch.cat(expected, dim=0)
    assert torch.allclose(batched, expected_stacked, atol=1e-6, rtol=1e-6)
