#!/usr/bin/env python3
"""Calibrate a global SOAP-matrix scale to AdamW endpoint exact categorical KL.

The calibration is pre-run and uses only the fixed teacher-forced GSM8K states.
It scales the nominal SOAP endpoint displacement for hidden attention/MLP matrices,
keeps the composite optimizer's auxiliary AdamW endpoint unchanged, and solves for
the scale whose full-vocabulary KL(base || scaled SOAP endpoint) matches AdamW.
"""
from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from evaluate_exact_categorical_kl import (
    DEFAULT_PROMPT_INDICES,
    _atomic_json,
    _model_logits,
    _unwrap_state_dict,
    build_shared_teacher_forced_states,
    exact_kl_over_shared_states,
)


def _is_soap_matrix(name: str, tensor: Tensor) -> bool:
    path = f".{name}."
    return tensor.ndim == 2 and (".self_attn." in path or ".mlp." in path)


def interpolate_state_dict(
    base: Mapping[str, Tensor], endpoint: Mapping[str, Tensor], scale: float
) -> dict[str, Tensor]:
    if base.keys() != endpoint.keys():
        raise ValueError("base and endpoint state dict keys differ")
    if scale < 0.0:
        raise ValueError("scale must be nonnegative")
    result: dict[str, Tensor] = {}
    for name, endpoint_tensor in endpoint.items():
        base_tensor = base[name]
        if base_tensor.shape != endpoint_tensor.shape:
            raise ValueError(f"shape mismatch for {name}")
        if _is_soap_matrix(name, endpoint_tensor) and endpoint_tensor.is_floating_point():
            result[name] = base_tensor + (endpoint_tensor - base_tensor) * scale
        else:
            result[name] = endpoint_tensor.clone()
    return result


def bisection_match_scale(
    *,
    target: float,
    evaluate: Callable[[float], float],
    lower: float,
    upper: float,
    relative_tolerance: float = 1e-3,
    max_iterations: int = 32,
) -> tuple[float, float, int]:
    if not 0.0 <= lower < upper or target <= 0.0:
        raise ValueError("invalid bisection bounds or target")
    low_value, high_value = evaluate(lower), evaluate(upper)
    if not low_value <= target <= high_value:
        raise ValueError(
            f"target is not bracketed: low={low_value}, target={target}, high={high_value}"
        )
    best_scale, best_value = lower, low_value
    for iteration in range(1, max_iterations + 1):
        scale = (lower + upper) / 2.0
        value = evaluate(scale)
        if abs(value - target) < abs(best_value - target):
            best_scale, best_value = scale, value
        if abs(value - target) <= relative_tolerance * target:
            return scale, value, iteration
        if value < target:
            lower = scale
        else:
            upper = scale
    return best_scale, best_value, max_iterations


def _load_tensor_state(path: Path) -> Mapping[str, Tensor]:
    raw = torch.load(path, map_location="cpu", weights_only=True)
    return _unwrap_state_dict(raw)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--adamw-checkpoint", type=Path, required=True)
    parser.add_argument("--soap-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--lower", type=float, default=0.5)
    parser.add_argument("--upper", type=float, default=2.0)
    parser.add_argument("--relative-tolerance", type=float, default=1e-3)
    parser.add_argument("--prompt-indices", default=",".join(map(str, DEFAULT_PROMPT_INDICES)))
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    indices: Sequence[int] = tuple(int(value) for value in args.prompt_indices.split(",") if value)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=False)
    input_ids, attention_mask, occupied = build_shared_teacher_forced_states(
        args.dataset, tokenizer, indices
    )
    dtype = torch.bfloat16 if torch.device(args.device).type == "cuda" else torch.float32

    def factory():
        return AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=dtype,
            attn_implementation="sdpa",
            trust_remote_code=False,
        )

    base_model = factory()
    base_state = {name: tensor.detach().cpu().clone() for name, tensor in base_model.state_dict().items()}
    base_logits = _model_logits(base_model, input_ids, attention_mask, torch.device(args.device))
    del base_model

    adamw_model = factory()
    adamw_model.load_state_dict(_load_tensor_state(args.adamw_checkpoint), strict=True)
    adamw_logits = _model_logits(adamw_model, input_ids, attention_mask, torch.device(args.device))
    del adamw_model
    target_summary = exact_kl_over_shared_states(base_logits, adamw_logits, occupied)

    soap_state = _load_tensor_state(args.soap_checkpoint)
    candidate_model = factory()
    evaluations: list[dict[str, float]] = []

    def evaluate(scale: float) -> float:
        candidate_model.load_state_dict(interpolate_state_dict(base_state, soap_state, scale), strict=True)
        logits = _model_logits(candidate_model, input_ids, attention_mask, torch.device(args.device))
        summary = exact_kl_over_shared_states(base_logits, logits, occupied)
        evaluations.append({"scale": scale, "mean": summary["mean"], "q95": summary["q95"]})
        return float(summary["mean"])

    scale, value, iterations = bisection_match_scale(
        target=float(target_summary["mean"]),
        evaluate=evaluate,
        lower=args.lower,
        upper=args.upper,
        relative_tolerance=args.relative_tolerance,
    )
    final = next(item for item in reversed(evaluations) if item["scale"] == scale)
    artifact: dict[str, Any] = {
        "schema": "soap_exact_kl_scale_calibration_v1",
        "scale": scale,
        "soap_lr": scale * 1e-6,
        "target": target_summary,
        "matched": {"mean": value, "q95": final["q95"]},
        "relative_mean_error": abs(value - target_summary["mean"]) / target_summary["mean"],
        "iterations": iterations,
        "bounds": [args.lower, args.upper],
        "prompt_indices": list(indices),
        "adamw_checkpoint": str(args.adamw_checkpoint.resolve()),
        "soap_checkpoint": str(args.soap_checkpoint.resolve()),
        "evaluations": evaluations,
        "method": "endpoint hidden-matrix displacement interpolation; auxiliary AdamW endpoint fixed",
    }
    _atomic_json(args.output, artifact)
    print(json.dumps(artifact, sort_keys=True))


if __name__ == "__main__":
    main()
