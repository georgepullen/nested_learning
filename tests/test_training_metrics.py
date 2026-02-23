import math

import torch

from nested_learning.training import _safe_perplexity


def test_safe_perplexity_matches_exp_for_normal_loss() -> None:
    ppl = _safe_perplexity(torch.tensor(2.0))
    assert math.isclose(ppl, math.exp(2.0), rel_tol=1e-6)


def test_safe_perplexity_clamps_large_losses() -> None:
    ppl = _safe_perplexity(torch.tensor(1000.0))
    assert math.isfinite(ppl)
    assert ppl == 1.0e38
