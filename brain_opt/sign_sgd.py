"""SignSGD / Signum.

When ``momentum == 0`` this is plain sign-SGD: ``p <- p - lr * sign(g)``.
With ``momentum > 0`` it becomes Signum (Bernstein et al., 2018):
``p <- p - lr * sign(m_t)`` where ``m_t`` is an EMA of gradients.
"""
from typing import Iterable, Optional, Union

import torch
from torch.optim.optimizer import Optimizer

_Params = Union[Iterable[torch.Tensor], Iterable[dict]]


class SignSGD(Optimizer):
    """Sign-based SGD (a.k.a. Signum when ``momentum > 0``).

    Args:
        params: iterable of parameters or parameter group dicts.
        lr: learning rate.
        momentum: momentum factor (default 0 → pure signSGD).
        dampening: dampening for momentum.
        weight_decay: decoupled weight-decay coefficient.
        nesterov: enable Nesterov momentum (requires ``momentum > 0`` and
            ``dampening == 0``).
    """

    def __init__(
        self,
        params: _Params,
        lr: float = 1e-3,
        momentum: float = 0.0,
        dampening: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError(
                "Nesterov momentum requires a positive momentum and zero dampening"
            )

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
        super().__init__(params, defaults)

    def __setstate__(self, state):  # type: ignore[override]
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            weight_decay = group["weight_decay"]
            nesterov = group["nesterov"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("SignSGD does not support sparse gradients")

                if weight_decay != 0:
                    p.data.mul_(1 - lr * weight_decay)

                if momentum != 0:
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.clone(grad).detach()
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(grad, alpha=1 - dampening)
                    update = grad.add(buf, alpha=momentum) if nesterov else buf
                else:
                    update = grad

                p.data.add_(update.sign(), alpha=-lr)

        return loss
