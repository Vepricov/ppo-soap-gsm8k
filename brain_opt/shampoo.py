"""Shampoo — Preconditioned Stochastic Tensor Optimization.

Single-file implementation of Algorithm 2 from
    Gupta, Koren, Singer. "Shampoo: Preconditioned Stochastic Tensor
    Optimization." ICML 2018. arXiv:1802.09568

Port of the canonical PyTorch reference at
    https://github.com/moskomule/shampoo.pytorch
which is also the version shipped in ``jettify/pytorch-optimizer``.

For every tensor of order ``k`` Shampoo maintains ``k`` left preconditioners
``H_i = sum_t (mat_i G_t) (mat_i G_t)^T``, applies their
inverse ``k``-th roots ``H_i^{-1/k}`` to the corresponding modes of the
gradient, and steps in that direction.
"""
from typing import Iterable, Optional, Union

import torch
from torch.optim.optimizer import Optimizer

_Params = Union[Iterable[torch.Tensor], Iterable[dict]]


def _matrix_power(matrix: torch.Tensor, power: float) -> torch.Tensor:
    """Compute a matrix power via SVD. Computed in float32 on CPU for
    numerical stability; result is cast back to the input device & dtype."""
    device = matrix.device
    dtype = matrix.dtype
    m = matrix.detach().to(device="cpu", dtype=torch.float32)
    U, S, Vh = torch.linalg.svd(m, full_matrices=False)
    out = (U * S.pow(power).unsqueeze(0)) @ Vh
    return out.to(device=device, dtype=dtype)


class Shampoo(Optimizer):
    """Shampoo optimizer (Gupta-Koren-Singer 2018).

    Args:
        params: iterable of parameters or parameter group dicts.
        lr: learning rate.
        momentum: heavy-ball momentum applied to the *preconditioned*
            gradient (default 0).
        weight_decay: coupled L2 weight-decay coefficient
            (gradient is augmented by ``weight_decay * p`` before
            preconditioning; same convention as the reference code).
        epsilon: ridge added to the preconditioners as ``epsilon * I``
            on initialization for numerical stability.
        update_freq: how often (in steps) to recompute the inverse
            preconditioners. ``1`` recomputes every step (slow but
            accurate); larger values amortize the SVD cost.
    """

    def __init__(
        self,
        params: _Params,
        lr: float = 1e-1,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        epsilon: float = 1e-4,
        update_freq: int = 1,
    ) -> None:
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if epsilon < 0.0:
            raise ValueError(f"Invalid epsilon value: {epsilon}")
        if update_freq < 1:
            raise ValueError(f"Invalid update_freq value: {update_freq}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            epsilon=epsilon,
            update_freq=update_freq,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]
            epsilon = group["epsilon"]
            update_freq = group["update_freq"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Shampoo does not support sparse gradients")

                order = grad.ndim
                original_size = grad.size()
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    if momentum > 0:
                        state["momentum_buffer"] = grad.clone()
                    for dim_id, dim in enumerate(grad.size()):
                        # Preconditioner accumulator H_i and its inverse root H_i^{-1/order}.
                        state[f"precond_{dim_id}"] = epsilon * torch.eye(
                            dim, device=grad.device, dtype=grad.dtype
                        )
                        state[f"inv_precond_{dim_id}"] = torch.zeros(
                            dim, dim, device=grad.device, dtype=grad.dtype
                        )

                if momentum > 0:
                    grad = grad.mul(1 - momentum).add_(
                        state["momentum_buffer"], alpha=momentum
                    )

                if weight_decay > 0:
                    grad = grad.add(p.data, alpha=weight_decay)

                # Algorithm 2 — apply mode-wise preconditioning.
                for dim_id, dim in enumerate(grad.size()):
                    precond = state[f"precond_{dim_id}"]
                    inv_precond = state[f"inv_precond_{dim_id}"]

                    # Matricize grad along this mode.
                    grad = grad.transpose(0, dim_id).contiguous()
                    transposed_size = grad.size()
                    grad = grad.view(dim, -1)

                    grad_t = grad.t()
                    precond.add_(grad @ grad_t)
                    if state["step"] % update_freq == 0:
                        inv_precond.copy_(_matrix_power(precond, -1.0 / order))

                    if dim_id == order - 1:
                        grad = grad_t @ inv_precond
                        grad = grad.view(original_size)
                    else:
                        grad = inv_precond @ grad
                        grad = grad.view(transposed_size)

                state["step"] += 1
                if momentum > 0:
                    state["momentum_buffer"] = grad
                p.data.add_(grad, alpha=-lr)

        return loss
