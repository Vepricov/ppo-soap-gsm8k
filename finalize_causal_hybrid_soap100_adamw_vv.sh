#!/usr/bin/env bash
set -euo pipefail

CAMPAIGN=/data/users/shkodnik1917/rl_muon/jarvis-gsm8k-r4/campaign-4cdf62757063
SCRIPT_ROOT=/data/users/shkodnik1917/rl_muon/causal-kfac-soap-current
HYBRID_ROOT=$CAMPAIGN/causal-hybrid-soap100-adamw-seed0
HYBRID_RUN=$HYBRID_ROOT/qwen2.5-0.5b_gsm8k_ppo_causal_soap100_adamw_seed0
SOAP_RUN=$CAMPAIGN/causal-kfac-soap-fast-ppo-mb4-gate25-seed0/qwen2.5-0.5b_gsm8k_ppo_kl_matched_soap_seed0
ADAMW_RUN=$CAMPAIGN/causal-kfac-adamw-exact-r6-seed2/qwen2.5-0.5b_gsm8k_ppo_causal_adamw_control_seed2
STATUS=$HYBRID_ROOT/harness.status
FINALIZE_STATUS=$HYBRID_ROOT/finalize.status
LOG=$HYBRID_ROOT/finalize.log
KL=$HYBRID_RUN/exact_categorical_kl_common_control_seed2.json
GATE=$HYBRID_ROOT/gate.json

exec >>"$LOG" 2>&1
trap 'printf "failed %s\n" "$(date -u +%FT%TZ)" > "$FINALIZE_STATUS"' ERR
printf 'waiting-training %s\n' "$(date -u +%FT%TZ)" > "$FINALIZE_STATUS"
while :; do
    state=$(awk 'NR==1 {print $1}' "$STATUS" 2>/dev/null || true)
    case "$state" in
        complete) break ;;
        failed) printf 'source-training-failed %s\n' "$(date -u +%FT%TZ)" > "$FINALIZE_STATUS"; exit 1 ;;
    esac
    sleep 30
done

printf 'waiting-eval-gpu %s\n' "$(date -u +%FT%TZ)" > "$FINALIZE_STATUS"
while :; do
    selected=$(
        nvidia-smi --query-gpu=uuid,memory.free --format=csv,noheader,nounits |
        sort -t, -k2,2nr |
        awk -F, '$2+0 >= 13312 {gsub(/^[[:space:]]+|[[:space:]]+$/, "", $1); print $1; exit}'
    )
    [[ -n "$selected" ]] && break
    sleep 30
done
free=$(nvidia-smi --query-gpu=uuid,memory.free --format=csv,noheader,nounits |
    awk -F, -v uuid="$selected" '$1 == uuid {gsub(/[[:space:]]/, "", $2); print $2}')
printf 'exact-kl gpu=%s free=%sMiB bound=8192MiB projected_remaining=%sMiB %s\n' \
    "$selected" "$free" "$((free - 8192))" "$(date -u +%FT%TZ)"
printf 'running-exact-kl %s\n' "$(date -u +%FT%TZ)" > "$FINALIZE_STATUS"
CUDA_VISIBLE_DEVICES="$selected" "$CAMPAIGN/venv/bin/python3" \
    "$SCRIPT_ROOT/evaluate_exact_categorical_kl.py" \
    --model-path "$CAMPAIGN/models/qwen2.5-0.5b-instruct" \
    --dataset "$CAMPAIGN/data/gsm8k/test.parquet" \
    --baseline-checkpoint "$ADAMW_RUN/checkpoints/global_step_150/actor/model_world_size_1_rank_0.pt" \
    --soap-checkpoint "$HYBRID_RUN/checkpoints/global_step_150/actor/model_world_size_1_rank_0.pt" \
    --expected-step 150 --device cuda --output "$KL"

"$CAMPAIGN/venv/bin/python3" - "$HYBRID_RUN/train.log" "$SOAP_RUN/metrics.jsonl" \
    "$ADAMW_RUN/metrics.jsonl" "$KL" "$GATE" <<'PY'
import json
import math
import os
import re
import sys
from pathlib import Path

hybrid_log, soap_metrics, adamw_metrics, kl_path, output = map(Path, sys.argv[1:])
metric = "val-core/openai/gsm8k/acc/mean@1"
pattern = re.compile(r"step:(\d+) - .*?val-core/openai/gsm8k/acc/mean@1:([0-9eE+\-.]+)")
first = {}
for step, value in pattern.findall(hybrid_log.read_text(errors="replace")):
    first.setdefault(int(step), float(value))
hybrid_required = (0, 25, 50, 75, 100, 125, 150)
if tuple(sorted(first)) != hybrid_required:
    raise RuntimeError(f"hybrid first-reached validation grid mismatch: {sorted(first)}")

def trajectory(path):
    out = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        data = row.get("data") or {}
        if metric in data:
            step = int(data.get("training/global_step", row.get("step")))
            out.setdefault(step, float(data[metric]))
    return out

soap = trajectory(soap_metrics)
adamw = trajectory(adamw_metrics)
control_requirements = {
    "soap": ((25, 50, 75, 100), soap),
    "adamw": ((150,), adamw),
}
for name, (required, values) in control_requirements.items():
    missing = [step for step in required if step not in values]
    if missing:
        raise RuntimeError(f"{name} missing validation steps {missing}")

def auc(values, grid):
    area = sum((b-a) * (values[a]+values[b]) / 2 for a, b in zip(grid, grid[1:]))
    return area / (grid[-1] - grid[0])

early_grid = (25, 50, 75, 100)
kl = json.loads(kl_path.read_text())
expected_provenance = {
    "schema": "soap_exact_categorical_kl_v1",
    "step": 150,
    "direction": "KL(base_model || endpoint)",
    "vocabulary": "full",
    "shared_occupied_response_states": True,
    "sampled_action_kl": False,
    "prompt_indices": list(range(16)),
    "baseline_step": 150,
    "soap_step": 150,
}
actual_provenance = {
    "schema": kl.get("schema"),
    "step": kl.get("step"),
    **{key: kl.get("provenance", {}).get(key) for key in expected_provenance if key not in {"schema", "step"}},
}
if actual_provenance != expected_provenance:
    raise RuntimeError(
        f"exact-KL provenance mismatch: {actual_provenance}, expected {expected_provenance}"
    )
for key, expected_suffix in {
    "baseline_checkpoint": "causal-kfac-adamw-exact-r6-seed2/qwen2.5-0.5b_gsm8k_ppo_causal_adamw_control_seed2/checkpoints/global_step_150/actor/model_world_size_1_rank_0.pt",
    "soap_checkpoint": "causal-hybrid-soap100-adamw-seed0/qwen2.5-0.5b_gsm8k_ppo_causal_soap100_adamw_seed0/checkpoints/global_step_150/actor/model_world_size_1_rank_0.pt",
}.items():
    if not kl["provenance"].get(key, "").endswith(expected_suffix):
        raise RuntimeError(f"unexpected exact-KL {key}: {kl['provenance'].get(key)}")
km = kl["metrics"]
checks = {
    "early_auc_25_100_not_below_soap": auc(first, early_grid) >= auc(soap, early_grid),
    "endpoint_150_not_below_adamw_common_control": first[150] >= adamw[150],
    "categorical_kl_mean_not_above_adamw_common_control": km["actor/categorical_kl_mean"] <= km["baseline/categorical_kl_mean"],
    "categorical_kl_q95_not_above_adamw_common_control": km["actor/categorical_kl_q95"] <= km["baseline/categorical_kl_q95"],
}
result = {
    "schema": "causal_hybrid_soap100_adamw_gate_v1",
    "decision": "GO" if all(checks.values()) else "NO_GO",
    "evidence_scope": "single_seed_mechanism_screen",
    "next_action": "replicate_seeds_1_2" if all(checks.values()) else "close_temporal_hybrid_branch",
    "checks": checks,
    "first_reached_resume_rule": True,
    "hybrid": {"early_auc_25_100": auc(first, early_grid), "endpoint_150": first[150],
               "categorical_kl_mean": km["actor/categorical_kl_mean"],
               "categorical_kl_q95": km["actor/categorical_kl_q95"],
               "validation": first},
    "soap_control": {"early_auc_25_100": auc(soap, early_grid), "endpoint_150": soap[150]},
    "adamw_common_control_seed2": {"endpoint_150": adamw[150],
               "categorical_kl_mean": km["baseline/categorical_kl_mean"],
               "categorical_kl_q95": km["baseline/categorical_kl_q95"]},
    "exact_kl_provenance": kl["provenance"],
    "claim_boundary": "AdamW comparator is the preregistered common seed-2 control; no strict paired seed-0 AdamW trajectory exists.",
}
temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
os.replace(temporary, output)
print(json.dumps(result, sort_keys=True))
PY
trap - ERR
printf 'complete %s\n' "$(date -u +%FT%TZ)" > "$FINALIZE_STATUS"
