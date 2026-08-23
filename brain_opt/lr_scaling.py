"""Per-optimizer learning-rate scaling.

The canonical optimal learning rate differs by orders of magnitude across
optimizer families (Lion ~1e-4 vs. Shampoo ~1e-1). ``scale_lr`` lets a
caller specify a single "user-facing" learning rate (typically ``1e-3``)
and routes each method through the multiplier that maps it to the
literature-recommended order of magnitude.

The multipliers below are calibrated empirically on a small MLP target
(non-trivial NN training, not toy regression) so that ``user_lr=1e-3``
gives a reasonable effective lr for *general* use:

    AdamW                          1e-3
    SGD            (with mom.)     1e-2
    SignSGD / Signum               1e-3
    Lion                           1e-3    (see note below)
    Muon                           1e-2    (Keller Jordan / Liu et al., 2025)
    Shampoo                        1e-1    (Gupta et al., 2018 / jettify)
    SOAP                           3e-3    (Vyas et al., 2024)

Note on Lion: Chen et al. 2023 report that for *large* Transformer
training, Lion's optimum is ~3-10× smaller than AdamW (i.e. 1e-4 for
``user_lr=1e-3``). On smaller models / shorter runs, Lion converges
much faster at lr matching AdamW. We default to ``1.0×`` for general
use; for LLM-scale training pass ``auto_scale_lr=False`` and set
``lr=1e-4..3e-4`` explicitly, or override the multiplier via
``brain_opt.LR_MULTIPLIERS["Lion"] = 0.1``.

``get_optimizer(..., auto_scale_lr=True)`` (the default) applies this
mapping so user-facing code can keep a single hyper-parameter sweep
across optimizers.
"""
from typing import Dict


# Canonical name → multiplier applied to the user-facing reference lr.
# Multiplier ``m`` means: user-passed lr is multiplied by ``m`` before
# being forwarded to the optimizer constructor.
LR_MULTIPLIERS: Dict[str, float] = {
    "SGD":     10.0,   # 1e-3 → 1e-2
    "AdamW":    1.0,   # 1e-3 → 1e-3
    "SignSGD":  1.0,   # 1e-3 → 1e-3
    "Lion":     1.0,   # 1e-3 → 1e-3 (for LLM scale training reduce to 0.1)
    "Muon":    10.0,   # 1e-3 → 1e-2
    "Shampoo": 100.0,  # 1e-3 → 1e-1
    "SOAP":     3.0,   # 1e-3 → 3e-3
}


def _canonical(name: str) -> str:
    """Map case- / separator-insensitive names back to the canonical key."""
    key = name.lower().replace("-", "").replace("_", "")
    lookup = {k.lower(): k for k in LR_MULTIPLIERS}
    lookup.update({
        "signum": "SignSGD",
        "adam":   "AdamW",     # treat Adam as AdamW for scaling purposes
    })
    if key not in lookup:
        raise KeyError(
            f"Unknown optimizer {name!r} for lr scaling. "
            f"Known: {sorted(set(lookup.values()))}"
        )
    return lookup[key]


def scale_lr(name: str, base_lr: float) -> float:
    """Convert a user-facing reference lr into the per-method effective lr.

    Args:
        name: optimizer name (case- and separator-insensitive).
        base_lr: user-facing reference learning rate (typically ``1e-3``).

    Returns:
        ``base_lr`` multiplied by the per-method calibration factor in
        :data:`LR_MULTIPLIERS`.

    Example:
        >>> scale_lr("Lion", 1e-3)
        0.0001
        >>> scale_lr("Shampoo", 1e-3)
        0.1
    """
    canonical = _canonical(name)
    return base_lr * LR_MULTIPLIERS[canonical]
