"""brain_opt — drop-in optimizer collection for PyTorch.

Public API:
    SGD       — re-export of ``torch.optim.SGD``.
    AdamW     — re-export of ``torch.optim.AdamW``.
    SignSGD   — sign-based SGD with optional momentum (Signum).
                Memory-efficient.
    Lion      — EvoLved Sign Momentum (Chen et al., 2023).
                Memory-efficient.
    Muon      — D-Muon: Newton-Schulz orthogonalized momentum with
                decoupled weight decay (Liu et al., 2025); falls back to
                AdamW for 1-D parameters and embeddings. Matrix-method.
    Shampoo   — Algorithm 2 from Gupta-Koren-Singer (2018). Matrix-method.
    SOAP      — ShampoO with Adam in the Preconditioner's eigenbasis
                (Vyas et al., 2024). Matrix-method.

    get_optimizer(params, name, **kwargs)  — factory by name with
        unified learning rate (``lr=1e-3`` works across all methods).
    scale_lr(name, base_lr)                — multiplier function used by
        the factory; exposed for callers that bypass ``get_optimizer``.

All optimizer classes follow the standard ``torch.optim.Optimizer`` API
and can be used as drop-in replacements (e.g. ``brain_opt.Lion`` in place
of ``torch.optim.AdamW``).
"""
from torch.optim import SGD, AdamW

from .factory import get_optimizer
from .lion import Lion
from .lr_scaling import LR_MULTIPLIERS, scale_lr
from .muon import Muon
from .shampoo import Shampoo
from .sign_sgd import SignSGD
from .soap import SOAP

__all__ = [
    "SGD",
    "AdamW",
    "SignSGD",
    "Lion",
    "Muon",
    "Shampoo",
    "SOAP",
    "get_optimizer",
    "scale_lr",
    "LR_MULTIPLIERS",
]

__version__ = "0.1.0"
