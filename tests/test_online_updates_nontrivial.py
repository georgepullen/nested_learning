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


def test_online_updates_nontrivial() -> None:
    torch.manual_seed(0)
    model = HOPEModel(_tiny_config())
    tokens = torch.randint(0, model.config.vocab_size, (1, 16))

    fast_state = model.init_fast_state(batch_size=1, fast_state_batch_mode="shared")
    with torch.no_grad():
        logits_before = model(tokens, fast_state=fast_state)
        teach_signal = compute_teach_signal(
            model,
            logits_before,
            tokens,
            normalization="per_sample",
        )
        _ = model(tokens, teach_signal=teach_signal, fast_state=fast_state)
        logits_after = model(tokens, fast_state=fast_state)

    diff = (logits_after - logits_before).float().abs().mean().item()
    assert diff > 1e-6
