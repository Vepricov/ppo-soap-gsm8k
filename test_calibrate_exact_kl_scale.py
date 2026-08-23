import pytest
import torch

from calibrate_exact_kl_scale import bisection_match_scale, interpolate_state_dict


def test_interpolation_scales_only_soap_hidden_matrices():
    base = {
        "model.layers.0.self_attn.q_proj.weight": torch.tensor([[1.0]]),
        "model.embed_tokens.weight": torch.tensor([[2.0]]),
    }
    endpoint = {
        "model.layers.0.self_attn.q_proj.weight": torch.tensor([[5.0]]),
        "model.embed_tokens.weight": torch.tensor([[7.0]]),
    }

    matched = interpolate_state_dict(base, endpoint, 0.25)

    assert matched["model.layers.0.self_attn.q_proj.weight"].item() == pytest.approx(2.0)
    assert matched["model.embed_tokens.weight"].item() == pytest.approx(7.0)


def test_bisection_matches_exact_kl_target():
    scale, value, iterations = bisection_match_scale(
        target=4.0,
        evaluate=lambda candidate: candidate * candidate,
        lower=0.0,
        upper=4.0,
        relative_tolerance=1e-4,
        max_iterations=40,
    )

    assert scale == pytest.approx(2.0, rel=1e-4)
    assert value == pytest.approx(4.0, rel=1e-4)
    assert iterations > 0


def test_bisection_rejects_unbracketed_target():
    with pytest.raises(ValueError, match="bracket"):
        bisection_match_scale(
            target=10.0,
            evaluate=lambda candidate: candidate,
            lower=0.0,
            upper=4.0,
        )
