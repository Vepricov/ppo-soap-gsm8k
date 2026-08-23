#!/usr/bin/env python3
"""Verify the terminal SOAP pilot and dedicated exact-KL artifact."""
import json
import math
import re
import sys
from pathlib import Path


ARTIFACT_SCHEMA = "soap_exact_categorical_kl_v1"
REQUIRED_METRICS = {
    "actor/categorical_kl_mean",
    "actor/categorical_kl_q95",
    "actor/categorical_kl_states",
    "baseline/categorical_kl_mean",
    "baseline/categorical_kl_q95",
    "baseline/categorical_kl_states",
}


def _verify_exact_kl(run_dir: Path, expected_step: int, soap_actor: Path) -> Path:
    artifact_path = run_dir / "exact_categorical_kl.json"
    if not artifact_path.is_file():
        raise RuntimeError(f"missing exact categorical KL artifact: {artifact_path}")
    try:
        artifact = json.loads(artifact_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid exact categorical KL artifact: {exc}") from exc
    if artifact.get("schema") != ARTIFACT_SCHEMA or artifact.get("step") != expected_step:
        raise RuntimeError("categorical KL artifact schema or step does not match the run")

    metrics = artifact.get("metrics")
    if not isinstance(metrics, dict) or not REQUIRED_METRICS <= metrics.keys():
        raise RuntimeError("categorical KL artifact is missing required exact metrics")
    for key in REQUIRED_METRICS:
        value = metrics[key]
        if not isinstance(value, (float, int)) or not math.isfinite(float(value)):
            raise RuntimeError(f"categorical KL metric is non-finite or non-numeric: {key}")
    if metrics["actor/categorical_kl_states"] <= 0:
        raise RuntimeError("categorical KL artifact has no occupied response states")

    provenance = artifact.get("provenance")
    required_provenance = {
        "direction", "vocabulary", "baseline_checkpoint", "soap_checkpoint",
        "baseline_step", "soap_step", "prompt_indices", "prompt_subset_policy",
        "shared_occupied_response_states", "sampled_action_kl",
        "evaluation_digest_sha256",
    }
    if not isinstance(provenance, dict) or not required_provenance <= provenance.keys():
        raise RuntimeError("categorical KL provenance is incomplete")
    if provenance["direction"] != "KL(base_model || endpoint)":
        raise RuntimeError("categorical KL direction is not the preregistered endpoint direction")
    if provenance["vocabulary"] != "full" or provenance["sampled_action_kl"] is not False:
        raise RuntimeError("categorical KL must be exact full-vocabulary, not sampled-action KL")
    if provenance["shared_occupied_response_states"] is not True:
        raise RuntimeError("categorical KL endpoints did not use shared occupied response states")
    if provenance["baseline_step"] != expected_step or provenance["soap_step"] != expected_step:
        raise RuntimeError("categorical KL endpoints are not from the requested same step")
    indices = provenance["prompt_indices"]
    if (
        provenance["prompt_subset_policy"] != "fixed_indices_v1"
        or not isinstance(indices, list) or not indices
        or any(not isinstance(index, int) or index < 0 for index in indices)
        or len(set(indices)) != len(indices)
    ):
        raise RuntimeError("categorical KL prompt subset provenance is invalid")
    digest = provenance["evaluation_digest_sha256"]
    if not isinstance(digest, str) or re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise RuntimeError("categorical KL evaluation digest is invalid")

    recorded_soap = Path(provenance["soap_checkpoint"])
    recorded_baseline = Path(provenance["baseline_checkpoint"])
    if recorded_soap.resolve() != soap_actor.resolve():
        raise RuntimeError("categorical KL SOAP checkpoint provenance does not match terminal actor")
    if not recorded_soap.is_file() or not recorded_baseline.is_file():
        raise RuntimeError("categorical KL endpoint checkpoint provenance points to a missing file")
    expected_component = f"global_step_{expected_step}"
    if expected_component not in recorded_baseline.parts or expected_component not in recorded_soap.parts:
        raise RuntimeError("categorical KL endpoint checkpoint paths do not encode the same step")
    return artifact_path


def verify(run_dir: Path, expected_step: int) -> dict:
    checkpoint = run_dir / "checkpoints" / f"global_step_{expected_step}"
    if not checkpoint.is_dir():
        raise RuntimeError(f"missing terminal checkpoint: {checkpoint}")
    if not (checkpoint / "actor").is_dir() or not (checkpoint / "critic").is_dir():
        raise RuntimeError("terminal checkpoint must contain actor and critic directories")
    if not (checkpoint / "data.pt").is_file():
        raise RuntimeError("terminal checkpoint is missing trainer/dataloader state data.pt")

    files = [path.name.lower() for path in checkpoint.rglob("*") if path.is_file()]
    for role in ("actor", "critic"):
        role_files = [path.name.lower() for path in (checkpoint / role).rglob("*") if path.is_file()]
        for kind, alternatives in {
            "weights": ("model", "weight", "safetensor"),
            "optimizer": ("optim",),
            "scheduler/RNG worker state": ("extra_state", "scheduler", "rng"),
        }.items():
            if not any(any(token in name for token in alternatives) for name in role_files):
                raise RuntimeError(f"{role} checkpoint is missing {kind}: {role_files}")

    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.is_file():
        raise RuntimeError("missing metrics.jsonl")
    rows = [json.loads(line) for line in metrics_path.read_text().splitlines() if line.strip()]
    if not rows or max(int(row["step"]) for row in rows) != expected_step:
        raise RuntimeError("metrics do not end at the requested global step")
    observed = set().union(*(row.get("data", {}).keys() for row in rows))
    validation = [
        key for key in observed
        if key.startswith(("val-core/", "val-aux/"))
        and (key.endswith("/reward/mean@1") or key.endswith("/acc/mean@1"))
    ]
    if not validation:
        raise RuntimeError("missing validation metric compatible with the AdamW baseline")
    for row in rows:
        for key, value in row.get("data", {}).items():
            if isinstance(value, (float, int)) and not math.isfinite(float(value)):
                raise RuntimeError(f"non-finite metric at step={row['step']} key={key}")

    soap_actor = checkpoint / "actor" / "model_world_size_1_rank_0.pt"
    if not soap_actor.is_file():
        raise RuntimeError(f"missing canonical terminal SOAP actor checkpoint: {soap_actor}")
    kl_artifact = _verify_exact_kl(run_dir, expected_step, soap_actor)
    return {
        "run_dir": str(run_dir),
        "terminal_step": expected_step,
        "checkpoint_files": len(files),
        "validation_metrics": sorted(validation),
        "categorical_kl_metrics": sorted(REQUIRED_METRICS),
        "categorical_kl_artifact": str(kl_artifact),
    }


if __name__ == "__main__":
    result = verify(Path(sys.argv[1]), int(sys.argv[2]))
    print(json.dumps(result, sort_keys=True))
