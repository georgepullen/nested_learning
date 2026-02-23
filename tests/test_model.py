import torch

from nested_learning.levels import LevelSpec
from nested_learning.model import HOPEModel, ModelConfig


def test_model_forward() -> None:
    config = ModelConfig(
        vocab_size=100,
        dim=32,
        num_layers=1,
        heads=4,
        titan_level=LevelSpec(name="titan", update_period=2),
        cms_levels=[LevelSpec(name="fast", update_period=1)],
    )
    model = HOPEModel(config)
    tokens = torch.randint(0, 100, (2, 10))
    logits = model(tokens)
    assert logits.shape == (2, 10, 100)


def test_model_collects_update_metrics_for_per_layer_teach_signals() -> None:
    config = ModelConfig(
        vocab_size=100,
        dim=32,
        num_layers=1,
        heads=4,
        block_variant="hope_attention",
        titan_level=LevelSpec(name="titan", update_period=2),
        cms_levels=[LevelSpec(name="fast", update_period=1)],
    )
    model = HOPEModel(config)
    tokens = torch.randint(0, 100, (2, 6))
    teach_signals = [torch.randn(2, 6, 32)]
    _ = model(tokens, teach_signals=teach_signals)
    metrics = model.pop_update_metrics()

    assert metrics
    assert any(key.startswith("layer0.cms.fast.") for key in metrics)
