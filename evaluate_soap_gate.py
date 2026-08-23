#!/usr/bin/env python3
"""Evaluate the preregistered Stateful SOAP seed-0 gate."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any


VALIDATION_SUFFIXES = ("/acc/mean@1", "/reward/mean@1")
VALIDATION_PREFIXES = ("val-core/", "val-aux/")


def _trajectory(metrics_path: Path, expected_step: int) -> list[tuple[int, float]]:
    rows = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
    candidates = {
        key
        for row in rows
        for key in row.get("data", {})
        if key.startswith(VALIDATION_PREFIXES) and key.endswith(VALIDATION_SUFFIXES)
    }
    preferred = {key for key in candidates if key.startswith("val-core/")}
    if len(preferred) == 1:
        key = next(iter(preferred))
    elif not preferred and len(candidates) == 1:
        key = next(iter(candidates))
    else:
        raise RuntimeError(f"expected one validation metric in {metrics_path}, found {sorted(candidates)}")
    values: dict[int, float] = {}
    for row in rows:
        data = row.get("data", {})
        if key in data:
            step = int(row["step"])
            value = float(data[key])
            if not math.isfinite(value):
                raise RuntimeError(f"non-finite validation at step {step}")
            values[step] = value
    trajectory = sorted(values.items())
    if not trajectory or trajectory[0][0] != 0 or trajectory[-1][0] != expected_step:
        raise RuntimeError(f"validation trajectory must span 0..{expected_step}: {trajectory}")
    return trajectory


def _normalized_auc(trajectory: list[tuple[int, float]]) -> float:
    width = trajectory[-1][0] - trajectory[0][0]
    if width <= 0:
        raise RuntimeError("AUC requires at least two distinct validation steps")
    area = sum(
        (right_step - left_step) * (left_value + right_value) / 2
        for (left_step, left_value), (right_step, right_value) in zip(trajectory, trajectory[1:])
    )
    return area / width


def evaluate_gate(
    baseline_metrics: Path,
    soap_metrics: Path,
    kl_artifact: Path,
    expected_step: int,
) -> dict[str, Any]:
    baseline_trajectory = _trajectory(baseline_metrics, expected_step)
    soap_trajectory = _trajectory(soap_metrics, expected_step)
    if [step for step, _ in baseline_trajectory] != [step for step, _ in soap_trajectory]:
        raise RuntimeError("AdamW and SOAP validation checkpoints do not match")
    artifact = json.loads(kl_artifact.read_text())
    if artifact.get("step") != expected_step:
        raise RuntimeError("exact-KL artifact step does not match")
    metrics = artifact["metrics"]
    baseline_auc = _normalized_auc(baseline_trajectory)
    soap_auc = _normalized_auc(soap_trajectory)
    checks = {
        "auc_plus_0_01": soap_auc >= baseline_auc + 0.01,
        "endpoint_noninferior": soap_trajectory[-1][1] >= baseline_trajectory[-1][1],
        "categorical_kl_mean_noninferior": metrics["actor/categorical_kl_mean"] <= metrics["baseline/categorical_kl_mean"],
        "categorical_kl_q95_noninferior": metrics["actor/categorical_kl_q95"] <= metrics["baseline/categorical_kl_q95"],
    }
    return {
        "decision": "GO" if all(checks.values()) else "NO_GO",
        "expected_step": expected_step,
        "checks": checks,
        "baseline": {
            "validation_auc": baseline_auc,
            "validation_endpoint": baseline_trajectory[-1][1],
            "categorical_kl_mean": metrics["baseline/categorical_kl_mean"],
            "categorical_kl_q95": metrics["baseline/categorical_kl_q95"],
        },
        "soap": {
            "validation_auc": soap_auc,
            "validation_endpoint": soap_trajectory[-1][1],
            "categorical_kl_mean": metrics["actor/categorical_kl_mean"],
            "categorical_kl_q95": metrics["actor/categorical_kl_q95"],
        },
        "delta": {
            "validation_auc": soap_auc - baseline_auc,
            "validation_endpoint": soap_trajectory[-1][1] - baseline_trajectory[-1][1],
        },
    }


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-metrics", type=Path, required=True)
    parser.add_argument("--soap-metrics", type=Path, required=True)
    parser.add_argument("--kl-artifact", type=Path, required=True)
    parser.add_argument("--expected-step", type=int, default=150)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate_gate(
        args.baseline_metrics, args.soap_metrics, args.kl_artifact, args.expected_step
    )
    _atomic_json(args.output, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
