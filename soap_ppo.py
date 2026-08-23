"""PPO-specific optimizer routing and exact categorical KL telemetry.

The actor route is deliberately composite: genuine stateful SOAP is used only
for hidden attention/MLP matrices, while every remaining trainable actor
parameter uses AdamW.  The critic route remains ordinary ``torch.optim.AdamW``.
The composite exposes one standard PyTorch optimizer state dict, so FSDP and
VERL checkpoint managers persist both routes plus SOAP's ``GG``, ``Q``, first
moment, and rotated second moment.
"""
from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor

from brain_opt.soap import SOAP


def partition_actor_parameters(
    named_parameters: Iterable[tuple[str, Tensor]],
) -> tuple[list[tuple[str, Tensor]], list[tuple[str, Tensor]]]:
    """Return disjoint, exhaustive SOAP-matrix and AdamW-auxiliary routes."""
    soap, auxiliary = [], []
    for name, parameter in named_parameters:
        path = f".{name}."
        hidden_matrix = parameter.ndim == 2 and (
            ".self_attn." in path or ".mlp." in path
        )
        (soap if hidden_matrix else auxiliary).append((name, parameter))
    return soap, auxiliary


class SOAPWithAuxAdamW(SOAP):
    """Stateful SOAP for actor hidden matrices, AdamW for actor auxiliaries.

    This is one standard :class:`torch.optim.Optimizer`, rather than a wrapper
    around two optimizers.  Consequently PyTorch/FSDP checkpoint conversion and
    LR schedulers see all parameter groups and all optimizer tensors directly.
    """

    requires_named_parameters = True
    route_label = "soap_actor_composite_aux_adamw"

    def __init__(
        self,
        named_parameters: Iterable[tuple[str, Tensor]],
        lr: float,
        weight_decay: float,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        soap_lr: float | None = None,
        soap_weight_decay: float | None = None,
        soap_betas: tuple[float, float] = (0.95, 0.95),
        soap_eps: float = 1e-8,
        soap_shampoo_beta: float = -1.0,
        soap_precondition_frequency: int = 10,
        soap_max_precond_dim: int = 2_048,
        auxiliary_lr: float | None = None,
        auxiliary_weight_decay: float | None = None,
        auxiliary_eps: float | None = None,
    ) -> None:
        named_parameters = list(named_parameters)
        soap_named, auxiliary_named = partition_actor_parameters(named_parameters)
        if not soap_named or not auxiliary_named:
            raise ValueError(
                "SOAPWithAuxAdamW requires non-empty SOAP matrix and auxiliary AdamW routes"
            )
        all_names = [name for name, _ in named_parameters]
        routed_names = [name for name, _ in soap_named + auxiliary_named]
        if len(set(routed_names)) != len(routed_names) or set(routed_names) != set(all_names):
            raise ValueError("actor optimizer routes must be disjoint and exhaustive")

        super().__init__(
            [{
                "params": [parameter for _, parameter in soap_named],
                "route": "soap_matrix",
                "parameter_names": tuple(name for name, _ in soap_named),
            }],
            lr=soap_lr if soap_lr is not None else lr,
            weight_decay=(
                soap_weight_decay if soap_weight_decay is not None else weight_decay
            ),
            betas=soap_betas,
            eps=soap_eps,
            shampoo_beta=soap_shampoo_beta,
            precondition_frequency=soap_precondition_frequency,
            max_precond_dim=soap_max_precond_dim,
            merge_dims=False,
            precondition_1d=False,
        )
        self.add_param_group({
            "params": [parameter for _, parameter in auxiliary_named],
            "route": "auxiliary_adamw",
            "parameter_names": tuple(name for name, _ in auxiliary_named),
            "lr": auxiliary_lr if auxiliary_lr is not None else lr,
            "weight_decay": (
                auxiliary_weight_decay
                if auxiliary_weight_decay is not None
                else weight_decay
            ),
            "betas": betas,
            "eps": auxiliary_eps if auxiliary_eps is not None else eps,
        })
        self.parameter_routes = {
            "soap_matrix": tuple(name for name, _ in soap_named),
            "auxiliary_adamw": tuple(name for name, _ in auxiliary_named),
        }

    @torch.no_grad()
    def step(self, closure=None):
        soap_groups = [group for group in self.param_groups if group["route"] == "soap_matrix"]
        auxiliary_groups = [
            group for group in self.param_groups if group["route"] == "auxiliary_adamw"
        ]
        all_groups = self.param_groups
        try:
            self.param_groups = soap_groups
            loss = super().step(closure)
        finally:
            self.param_groups = all_groups

        for group in auxiliary_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.is_sparse:
                    raise RuntimeError("AdamW auxiliary route does not support sparse gradients")
                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)
                state["step"] += 1
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                exp_avg.lerp_(gradient, 1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)
                parameter.mul_(1.0 - group["lr"] * group["weight_decay"])
                correction1 = 1.0 - beta1 ** state["step"]
                correction2 = 1.0 - beta2 ** state["step"]
                denominator = exp_avg_sq.sqrt().div_(correction2 ** 0.5).add_(group["eps"])
                parameter.addcdiv_(exp_avg, denominator, value=-group["lr"] / correction1)
        return loss


def categorical_kl_summary(
    old_logits: Tensor,
    new_logits: Tensor,
    response_mask: Tensor | None = None,
) -> dict[str, Any]:
    """Exact full-vocabulary ``KL(old || new)`` mean and q95 over states.

    Inputs are logits captured before and after a PPO actor update.  This is
    intentionally not the sampled-action log-ratio often reported as
    ``ppo_kl``.  Callers should stream minibatches and log these stable metric
    names alongside the existing AdamW baseline validation metrics.
    """
    if old_logits.shape != new_logits.shape or old_logits.ndim < 2:
        raise ValueError("old_logits and new_logits must have the same [..., vocab] shape")
    old_log_probs = old_logits.float().log_softmax(dim=-1)
    new_log_probs = new_logits.float().log_softmax(dim=-1)
    values = (old_log_probs.exp() * (old_log_probs - new_log_probs)).sum(dim=-1)
    if response_mask is not None:
        if response_mask.shape != values.shape:
            raise ValueError("response_mask must match logits without the vocab dimension")
        values = values[response_mask.bool()]
    else:
        values = values.reshape(-1)
    if values.numel() == 0:
        raise ValueError("categorical KL requires at least one response state")
    if not torch.isfinite(values).all():
        raise FloatingPointError("non-finite categorical KL")
    return {
        "actor/categorical_kl_mean": values.mean().item(),
        "actor/categorical_kl_q95": torch.quantile(values, 0.95).item(),
        "actor/categorical_kl_states": values.numel(),
    }
