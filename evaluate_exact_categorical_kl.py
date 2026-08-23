#!/usr/bin/env python3
"""Evaluate exact endpoint categorical KL on fixed, shared GSM8K states.

This is deliberately an endpoint evaluator, not PPO sampled-action telemetry.  It
loads the AdamW and SOAP actor checkpoints strictly, evaluates both against the
same immutable base model on identical teacher-forced response prefixes, and
reduces full-vocabulary ``KL(base || endpoint)`` over occupied response states.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn


ARTIFACT_SCHEMA = "soap_exact_categorical_kl_v1"
DEFAULT_PROMPT_INDICES = tuple(range(16))
DIRECTION = "KL(base_model || endpoint)"


def exact_kl_over_shared_states(
    baseline_logits: Tensor,
    soap_logits: Tensor,
    occupied_response_states: Tensor,
) -> dict[str, Any]:
    """Reduce exact full-vocabulary ``KL(reference || endpoint)`` on shared states."""
    if baseline_logits.shape != soap_logits.shape or baseline_logits.ndim != 3:
        raise ValueError("endpoint logits must have the same [batch, sequence, vocab] shape")
    if occupied_response_states.shape != baseline_logits.shape[:-1]:
        raise ValueError("occupied response-state mask must match [batch, sequence]")
    baseline_logp = baseline_logits.float().log_softmax(dim=-1)
    soap_logp = soap_logits.float().log_softmax(dim=-1)
    values = (baseline_logp.exp() * (baseline_logp - soap_logp)).sum(dim=-1)
    values = values[occupied_response_states.bool()]
    if values.numel() == 0:
        raise ValueError("exact categorical KL requires occupied response states")
    if not torch.isfinite(values).all():
        raise FloatingPointError("non-finite exact categorical KL")
    return {
        "mean": values.mean().item(),
        "q95": torch.quantile(values, 0.95).item(),
        "states": values.numel(),
        "direction": DIRECTION,
        "vocabulary": "full",
    }


def _unwrap_state_dict(raw: Any) -> Mapping[str, Tensor]:
    if isinstance(raw, Mapping):
        for key in ("state_dict", "model"):
            candidate = raw.get(key)
            if isinstance(candidate, Mapping):
                raw = candidate
                break
    if not isinstance(raw, Mapping) or not raw or not all(
        isinstance(key, str) and isinstance(value, Tensor) for key, value in raw.items()
    ):
        raise RuntimeError("checkpoint does not contain a tensor-only model state dict")
    return raw


def load_state_dict_strict(model: nn.Module, checkpoint: Path) -> nn.Module:
    """Load an actor checkpoint with no missing or unexpected tensor names."""
    checkpoint = Path(checkpoint)
    if not checkpoint.is_file():
        raise RuntimeError(f"missing actor checkpoint: {checkpoint}")
    try:
        raw = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(_unwrap_state_dict(raw), strict=True)
    except Exception as exc:
        raise RuntimeError(f"strict state-dict load failed for {checkpoint}: {exc}") from exc
    return model


def _model_logits(
    model: nn.Module,
    input_ids: Tensor,
    attention_mask: Tensor,
    device: torch.device,
) -> Tensor:
    model.to(device).eval()
    with torch.inference_mode():
        logits = model(
            input_ids=input_ids.to(device),
            attention_mask=attention_mask.to(device),
        ).logits.float().cpu()
    model.to("cpu")
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return logits


def _evaluation_digest(
    input_ids: Tensor,
    attention_mask: Tensor,
    occupied: Tensor,
    prompt_indices: Sequence[int],
) -> str:
    digest = hashlib.sha256()
    digest.update(json.dumps(list(prompt_indices), separators=(",", ":")).encode())
    for tensor in (input_ids, attention_mask, occupied):
        contiguous = tensor.detach().cpu().contiguous()
        digest.update(str(tuple(contiguous.shape)).encode())
        digest.update(contiguous.numpy().tobytes())
    return digest.hexdigest()


def evaluate_checkpoint_pair(
    *,
    model_factory: Callable[[], nn.Module],
    baseline_checkpoint: Path,
    soap_checkpoint: Path,
    input_ids: Tensor,
    attention_mask: Tensor,
    occupied_response_states: Tensor,
    expected_step: int,
    prompt_indices: Sequence[int],
    dataset_path: Path,
    model_path: Path,
    device: str | torch.device = "cpu",
) -> dict[str, Any]:
    """Strictly load both endpoints and compare each with the base policy."""
    if expected_step < 1:
        raise ValueError("expected_step must be positive")
    if not prompt_indices or len(set(prompt_indices)) != len(prompt_indices):
        raise ValueError("prompt indices must be non-empty and unique")
    device = torch.device(device)
    base_model = model_factory()
    base_logits = _model_logits(base_model, input_ids, attention_mask, device)
    del base_model
    baseline_model = load_state_dict_strict(model_factory(), Path(baseline_checkpoint))
    baseline_logits = _model_logits(baseline_model, input_ids, attention_mask, device)
    del baseline_model
    soap_model = load_state_dict_strict(model_factory(), Path(soap_checkpoint))
    soap_logits = _model_logits(soap_model, input_ids, attention_mask, device)
    del soap_model
    baseline_summary = exact_kl_over_shared_states(
        base_logits, baseline_logits, occupied_response_states
    )
    soap_summary = exact_kl_over_shared_states(
        base_logits, soap_logits, occupied_response_states
    )
    baseline_checkpoint = Path(baseline_checkpoint).resolve()
    soap_checkpoint = Path(soap_checkpoint).resolve()
    return {
        "schema": ARTIFACT_SCHEMA,
        "step": expected_step,
        "metrics": {
            "actor/categorical_kl_mean": soap_summary["mean"],
            "actor/categorical_kl_q95": soap_summary["q95"],
            "actor/categorical_kl_states": soap_summary["states"],
            "baseline/categorical_kl_mean": baseline_summary["mean"],
            "baseline/categorical_kl_q95": baseline_summary["q95"],
            "baseline/categorical_kl_states": baseline_summary["states"],
        },
        "provenance": {
            "direction": soap_summary["direction"],
            "vocabulary": soap_summary["vocabulary"],
            "baseline_checkpoint": str(baseline_checkpoint),
            "soap_checkpoint": str(soap_checkpoint),
            "baseline_step": expected_step,
            "soap_step": expected_step,
            "model_path": str(Path(model_path).resolve()),
            "dataset_path": str(Path(dataset_path).resolve()),
            "prompt_indices": list(prompt_indices),
            "prompt_subset_policy": "fixed_indices_v1",
            "shared_occupied_response_states": True,
            "sampled_action_kl": False,
            "evaluation_digest_sha256": _evaluation_digest(
                input_ids, attention_mask, occupied_response_states, prompt_indices
            ),
        },
    }


def _prompt_and_answer(row: Mapping[str, Any]) -> tuple[Any, str]:
    prompt = row.get("prompt")
    reward_model = row.get("reward_model")
    answer = reward_model.get("ground_truth") if isinstance(reward_model, Mapping) else None
    if prompt is None or answer is None:
        raise RuntimeError("GSM8K row must contain prompt and reward_model.ground_truth")
    if hasattr(prompt, "tolist"):
        prompt = prompt.tolist()
    if not isinstance(answer, str) or not answer:
        raise RuntimeError("GSM8K ground-truth response must be a non-empty string")
    return prompt, answer


def build_shared_teacher_forced_states(
    dataset_path: Path,
    tokenizer: Any,
    prompt_indices: Sequence[int],
) -> tuple[Tensor, Tensor, Tensor]:
    """Tokenize fixed GSM8K rows and mark states predicting response tokens."""
    import pandas as pd

    frame = pd.read_parquet(dataset_path)
    if not prompt_indices or min(prompt_indices) < 0 or max(prompt_indices) >= len(frame):
        raise RuntimeError("pre-registered prompt indices are outside the test parquet")
    examples: list[tuple[list[int], int]] = []
    eos = tokenizer.eos_token_id
    if eos is None:
        raise RuntimeError("tokenizer must define eos_token_id")
    for index in prompt_indices:
        prompt, answer = _prompt_and_answer(frame.iloc[index].to_dict())
        prompt_ids = tokenizer.apply_chat_template(
            prompt, tokenize=True, add_generation_prompt=True
        )
        answer_ids = tokenizer.encode(answer, add_special_tokens=False) + [eos]
        if not prompt_ids or not answer_ids:
            raise RuntimeError(f"empty tokenization for test row {index}")
        examples.append((list(prompt_ids) + answer_ids, len(prompt_ids)))

    max_length = max(len(ids) for ids, _ in examples)
    pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos
    input_ids = torch.full((len(examples), max_length), pad, dtype=torch.long)
    attention_mask = torch.zeros_like(input_ids)
    occupied = torch.zeros_like(input_ids, dtype=torch.bool)
    for row, (ids, prompt_length) in enumerate(examples):
        length = len(ids)
        input_ids[row, :length] = torch.tensor(ids)
        attention_mask[row, :length] = 1
        # Logit at position t predicts token t+1. Include every response token,
        # including EOS, and no padding or prompt-only state.
        occupied[row, prompt_length - 1 : length - 1] = True
    return input_ids, attention_mask, occupied


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path, required=True)
    parser.add_argument("--soap-checkpoint", type=Path, required=True)
    parser.add_argument("--expected-step", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prompt-indices", default=",".join(map(str, DEFAULT_PROMPT_INDICES)))
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    indices = tuple(int(value) for value in args.prompt_indices.split(",") if value)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=False)
    input_ids, attention_mask, occupied = build_shared_teacher_forced_states(
        args.dataset, tokenizer, indices
    )

    def model_factory() -> nn.Module:
        return AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16 if torch.device(args.device).type == "cuda" else torch.float32,
            attn_implementation="sdpa",
            trust_remote_code=False,
        )

    artifact = evaluate_checkpoint_pair(
        model_factory=model_factory,
        baseline_checkpoint=args.baseline_checkpoint,
        soap_checkpoint=args.soap_checkpoint,
        input_ids=input_ids,
        attention_mask=attention_mask,
        occupied_response_states=occupied,
        expected_step=args.expected_step,
        prompt_indices=indices,
        dataset_path=args.dataset,
        model_path=args.model_path,
        device=args.device,
    )
    _atomic_json(args.output, artifact)
    print(json.dumps(artifact, sort_keys=True))


if __name__ == "__main__":
    main()
