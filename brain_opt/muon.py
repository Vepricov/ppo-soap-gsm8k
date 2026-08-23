"""D-Muon — Muon with decoupled weight decay.

Muon orthogonalizes the SGD-momentum update of every >=2-D parameter via a
quintic Newton-Schulz iteration, then takes a spectral-norm step. 1-D
parameters (norms, biases) and embedding-like matrices fall back to AdamW.

This file implements the *D-Muon* variant from Liu et al., "Muon is Scalable
for LLM Training" (arXiv:2502.16982): decoupled weight decay applied
multiplicatively to the parameters, and the per-parameter learning rate
rescaled by ``0.2 * sqrt(max(fan_in, fan_out))`` so that Muon matches AdamW
update magnitudes across layer shapes.

The Newton-Schulz iteration is the bfloat16 quintic version from
https://github.com/KellerJordan/Muon — runs fine on CPU/CUDA, no
``torch.distributed`` and no ``torch.compile`` dependency.
"""
import math
from typing import Iterable, List, Optional, Tuple, Union

import torch
from torch.optim.optimizer import Optimizer

_Params = Union[Iterable[torch.Tensor], Iterable[dict]]


@torch.no_grad()
def _zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Compute the zeroth power (a.k.a. orthogonalization) of a 2-D matrix
    by a quintic Newton-Schulz iteration. Inputs and outputs are 2-D.

    Coefficients ``(3.4445, -4.7750, 2.0315)`` are taken from
    https://github.com/KellerJordan/Muon and are optimized to maximize the
    slope at zero; the iteration does not produce an exact UV^T but rather
    something close enough to act as a spectral-norm normaliser in
    optimizer updates.
    """
    assert G.ndim == 2, f"Newton-Schulz expects a 2-D tensor, got {G.shape}"
    a, b, c = 3.4445, -4.7750, 2.0315
    use_bf16 = G.is_cuda  # bfloat16 only on CUDA; CPU uses float32 to avoid slow bf16 matmul
    work_dtype = torch.bfloat16 if use_bf16 else torch.float32
    X = G.to(work_dtype)
    X = X / (X.norm() + eps)  # spectral norm <= 1
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.T
    return X


def _is_muon_param(p: torch.Tensor) -> bool:
    """Default heuristic: use Muon for >=2-D parameters whose first dim
    is < 10000 (i.e. avoid embedding / lm_head matrices). This is the same
    rule as the reference Muon implementation."""
    return p.ndim >= 2 and p.size(0) < 10000


class Muon(Optimizer):
    """D-Muon: matrix-orthogonalised momentum optimizer with decoupled
    weight decay and AdamW fallback.

    Args:
        params: iterable of parameters or parameter group dicts. Every
            parameter is auto-classified: ``ndim >= 2`` and
            ``size(0) < 10000`` → Muon update; otherwise → AdamW update.
            You can override the rule by passing a parameter group with
            ``"use_muon": True`` / ``False``.
        lr: base learning rate. Inside Muon it is rescaled by
            ``0.2 * sqrt(max(A, B))`` for a parameter with leading shape
            ``(A, B)``; the AdamW fallback uses ``lr`` directly.
        weight_decay: decoupled weight-decay coefficient, applied to both
            Muon-managed and AdamW-managed parameters.
        momentum: SGD momentum for the Muon path.
        nesterov: enable Nesterov-style momentum in the Muon path.
        ns_steps: number of Newton-Schulz iterations (5 is a good default).
        adamw_betas: ``(beta1, beta2)`` for the AdamW fallback.
        adamw_eps: epsilon for the AdamW fallback.
        adamw_lr_ratio: multiplier applied to ``lr`` when computing the
            AdamW path learning rate (default 1.0). Allows independent
            scheduling without breaking the standard LR-scheduler API.

    Example:
        >>> opt = Muon(model.parameters(), lr=0.02, weight_decay=0.1)
        >>> opt.step()
    """

    def __init__(
        self,
        params: _Params,
        lr: float = 0.02,
        weight_decay: float = 0.0,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        adamw_betas: Tuple[float, float] = (0.9, 0.95),
        adamw_eps: float = 1e-8,
        adamw_lr_ratio: float = 1.0,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        if not 0.0 <= momentum < 1.0:
            raise ValueError(f"Invalid momentum: {momentum}")
        if ns_steps < 1:
            raise ValueError(f"ns_steps must be >= 1, got {ns_steps}")

        defaults = dict(
            lr=lr,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
            adamw_lr_ratio=adamw_lr_ratio,
        )
        super().__init__(params, defaults)

        # Auto-classify every param. A user can override by setting
        # state[p]["use_muon"] manually or by using "use_muon" in groups.
        for group in self.param_groups:
            override = group.get("use_muon", None)
            for p in group["params"]:
                if override is None:
                    self.state[p]["use_muon"] = _is_muon_param(p)
                else:
                    self.state[p]["use_muon"] = bool(override)

    @staticmethod
    def _adjusted_lr(lr: float, shape: Tuple[int, ...]) -> float:
        """Per-parameter LR scaling from Liu et al. 2025 (D-Muon)."""
        A, B = shape[0], shape[1]
        return lr * 0.2 * math.sqrt(max(A, B))

    @torch.no_grad()
    def step(self, closure: Optional[callable] = None):  # type: ignore[override]
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            wd = group["weight_decay"]
            momentum = group["momentum"]
            nesterov = group["nesterov"]
            ns_steps = group["ns_steps"]
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            adamw_lr = lr * group["adamw_lr_ratio"]

            # ---------- Muon path ----------
            for p in group["params"]:
                if p.grad is None or not self.state[p].get("use_muon", False):
                    continue
                g = p.grad
                original_shape = g.shape
                if g.ndim > 2:
                    g = g.view(g.size(0), -1)

                state = self.state[p]
                if "momentum_buffer" not in state:
                    state["momentum_buffer"] = torch.zeros_like(g)
                buf = state["momentum_buffer"]
                buf.mul_(momentum).add_(g)
                g_update = g.add(buf, alpha=momentum) if nesterov else buf
                u = _zeropower_via_newtonschulz5(g_update, steps=ns_steps).to(p.dtype)
                if u.shape != original_shape:
                    u = u.view(original_shape)

                adj_lr = self._adjusted_lr(lr, original_shape)
                if wd != 0:
                    p.data.mul_(1 - lr * wd)
                p.data.add_(u, alpha=-adj_lr)

            # ---------- AdamW fallback ----------
            for p in group["params"]:
                if p.grad is None or self.state[p].get("use_muon", False):
                    continue
                g = p.grad
                if g.is_sparse:
                    raise RuntimeError("Muon AdamW fallback does not support sparse gradients")
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)
                state["step"] += 1
                step = state["step"]
                buf1 = state["moment1"]
                buf2 = state["moment2"]
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)

                denom = eps + buf2.sqrt()
                update = buf1 / denom

                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                scale = bias_correction1 / math.sqrt(bias_correction2)
                if wd != 0:
                    p.data.mul_(1 - adamw_lr * wd)
                p.data.add_(update, alpha=-adamw_lr / scale)

        return loss
