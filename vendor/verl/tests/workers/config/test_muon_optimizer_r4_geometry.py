import pytest
import torch
from torch import nn

from verl.utils.optimizers import Muon


def _reference_zeropower(gradient):
    update = gradient.float()
    transposed = update.size(0) > update.size(1)
    if transposed:
        update = update.T
    update = update / update.norm().clamp(min=1e-7)
    for _ in range(5):
        gram = update @ update.T
        update = (
            3.4445 * update
            + (-4.7750 * gram + 2.0315 * (gram @ gram)) @ update
        )
    return update.T if transposed else update


@pytest.mark.parametrize("shape", ((2, 3), (3, 2)))
def test_muon_fp32_geometry_and_momentum(shape):
    initial = torch.arange(
        1, 1 + shape[0] * shape[1], dtype=torch.float32
    ).reshape(shape)
    gradient = torch.tensor(
        [[0.1, -0.2, 0.3], [0.4, -0.5, 0.6], [0.7, -0.8, 0.9]],
        dtype=torch.float32,
    )[: shape[0], : shape[1]]
    parameter = nn.Parameter(initial.clone())
    parameter.grad = gradient.clone()
    optimizer = Muon([parameter], lr=1e-3, weight_decay=0.01)
    optimizer.step()

    state = optimizer.state[parameter]
    torch.testing.assert_close(
        state["momentum_buffer"].float(),
        0.05 * gradient,
        rtol=1e-5,
        atol=1e-6,
    )
    reference = _reference_zeropower(gradient * 0.0975)
    adjusted_lr = 1e-3 * 0.2 * (max(shape) ** 0.5)
    actual_update = (
        initial * (1 - 1e-3 * 0.01) - parameter.detach().float()
    ) / adjusted_lr
    cosine = torch.nn.functional.cosine_similarity(
        actual_update.flatten(), reference.flatten(), dim=0
    )
    assert float(cosine) > 0.999
    assert float(actual_update.norm() / reference.norm()) == pytest.approx(
        1.0, rel=2e-2, abs=2e-2
    )
    assert torch.isfinite(parameter).all()
