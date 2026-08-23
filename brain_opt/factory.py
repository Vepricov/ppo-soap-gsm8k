"""Optimizer factory for brain_opt.

Usage:
    >>> import brain_opt
    >>> opt = brain_opt.get_optimizer(model.parameters(),
    ...                               name="Lion", lr=1e-3)

Pass ``lr=1e-3`` for any method — the factory rescales it to the
per-method canonical magnitude (see :mod:`brain_opt.lr_scaling`).
"""
from typing import Any, Dict, Iterable, Type, Union

import torch
from torch.optim.optimizer import Optimizer

from .lion import Lion
from .lr_scaling import scale_lr
from .muon import Muon
from .shampoo import Shampoo
from .sign_sgd import SignSGD
from .soap import SOAP

_Params = Union[Iterable[torch.Tensor], Iterable[dict]]


_REGISTRY: Dict[str, Type[Optimizer]] = {
    "SGD":     torch.optim.SGD,
    "AdamW":   torch.optim.AdamW,
    "SignSGD": SignSGD,
    "Lion":    Lion,
    "Muon":    Muon,
    "Shampoo": Shampoo,
    "SOAP":    SOAP,
}


def _normalize(name: str) -> str:
    return name.lower().replace("-", "").replace("_", "")


_LOOKUP: Dict[str, str] = {_normalize(k): k for k in _REGISTRY}
# Extra aliases.
_LOOKUP[_normalize("signum")] = "SignSGD"
_LOOKUP[_normalize("adam")] = "AdamW"


def available_optimizers() -> list:
    """Return the list of canonical optimizer names."""
    return list(_REGISTRY.keys())


def get_optimizer(
    params: _Params,
    name: str = "AdamW",
    *,
    auto_scale_lr: bool = True,
    **hyperparameters: Any,
) -> Optimizer:
    """Instantiate an optimizer by name with unified learning rate.

    Args:
        params: iterable of parameters or parameter group dicts, as passed
            to any ``torch.optim.Optimizer``.
        name: optimizer name. Case- and separator-insensitive. Accepted:
            ``"SGD"``, ``"AdamW"`` (alias ``"Adam"``), ``"SignSGD"``
            (alias ``"Signum"``), ``"Lion"``, ``"Muon"``, ``"Shampoo"``,
            ``"SOAP"``.
        auto_scale_lr: if ``True`` (default) and ``"lr"`` is present in
            ``hyperparameters``, rescale it through
            :func:`brain_opt.scale_lr` so that ``lr=1e-3`` works across
            every method. Set to ``False`` to forward the raw value.
        **hyperparameters: forwarded verbatim to the optimizer
            constructor.

    Returns:
        Instantiated optimizer.

    Raises:
        KeyError: if ``name`` is not registered.

    Example:
        >>> opt = get_optimizer(model.parameters(), name="Muon", lr=1e-3)
        >>> # → effective lr = 0.02
    """
    key = _normalize(name)
    if key not in _LOOKUP:
        raise KeyError(
            f"Unknown optimizer {name!r}. Available: {available_optimizers()}"
        )
    canonical = _LOOKUP[key]

    if auto_scale_lr and "lr" in hyperparameters:
        hyperparameters["lr"] = scale_lr(canonical, hyperparameters["lr"])

    cls = _REGISTRY[canonical]
    return cls(params, **hyperparameters)
