"""Lion — EvoLved Sign Momentum (Chen et al., 2023).

Drop-in replacement for AdamW with roughly half the memory (only first-moment
buffer, no second moment).

Reference: https://arxiv.org/abs/2302.06675
Original code: https://github.com/google/automl/tree/master/lion
"""
from typing import Iterable, Optional, Tuple, Union

import torch
from torch.optim.optimizer import Optimizer

_Params = Union[Iterable[torch.Tensor], Iterable[dict]]


class Lion(Optimizer):
    """Implements the Lion optimizer.

    Args:
        params: iterable of parameters or parameter group dicts.
        lr: learning rate.
        betas: coefficients (beta1, beta2) for update interpolation and
            momentum EMA decay (defaults ``(0.9, 0.99)``).
        weight_decay: decoupled weight-decay coefficient.
    """

    def __init__(
        self,
        params: _Params,
        lr: float = 1e-4,
        betas: Tuple[float, float] = (0.9, 0.99),
        weight_decay: float = 0.0,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")

        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("Lion does not support sparse gradients")

                # Decoupled weight decay (AdamW style).
                if weight_decay != 0:
                    p.data.mul_(1 - lr * weight_decay)

                state = self.state[p]
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(p)

                exp_avg = state["exp_avg"]

                # Update direction: sign(beta1 * m + (1 - beta1) * g)
                update = exp_avg.mul(beta1).add_(grad, alpha=1 - beta1).sign_()
                p.data.add_(update, alpha=-lr)

                # Decay the momentum running average.
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)

        return loss
