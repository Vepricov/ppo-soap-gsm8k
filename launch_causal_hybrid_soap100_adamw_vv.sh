#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CAMPAIGN=${CAMPAIGN:-/data/users/shkodnik1917/rl_muon/jarvis-gsm8k-r4/campaign-4cdf62757063}
export CAMPAIGN
export OUTPUT_ROOT=${OUTPUT_ROOT:-$CAMPAIGN/causal-hybrid-soap100-adamw-seed0}
export RUN_NAME=${RUN_NAME:-qwen2.5-0.5b_gsm8k_ppo_causal_soap100_adamw_seed0}
export ACTOR_OPTIMIZER=KLMatchedSOAPThenAdamW
export ACTOR_OPTIMIZER_IMPL=verl.utils.kl_matched_soap
export SWITCH_AFTER_GLOBAL_STEP=100 OPTIMIZER_UPDATES_PER_GLOBAL_STEP=4
export SAVE_FREQ=25 TEST_FREQ=25
STATUS=$OUTPUT_ROOT/harness.status
trap 'printf "failed hybrid_production_verification %s\n" "$(date -u +%FT%TZ)" > "$STATUS"' ERR

if [[ -e "$OUTPUT_ROOT/$RUN_NAME" ]]; then
    echo "refusing to overwrite causal hybrid production: $OUTPUT_ROOT/$RUN_NAME" >&2
    exit 73
fi

EXPECTED_STEP=100 "$ROOT/launch_causal_fast_mb4_gate25_vv.sh"
ALLOW_RESUME=1 EXPECTED_STEP=150 "$ROOT/launch_causal_fast_mb4_gate25_vv.sh"

"$CAMPAIGN/venv/bin/python3" - "$OUTPUT_ROOT/$RUN_NAME" <<'PY'
import json
import sys
from pathlib import Path
import torch

run_root = Path(sys.argv[1])
checkpoint = torch.load(
    run_root / "checkpoints/global_step_100/actor/optim_world_size_1_rank_0.pt",
    map_location="cpu",
    weights_only=False,
)
metadata = checkpoint["kl_matched_soap"]
if (metadata["update_generation"], metadata["latest_telemetry"]["actor/hybrid/phase"]) != (400, 0.0):
    raise SystemExit("step-100 checkpoint is not the terminal SOAP phase")
phase_by_step = {}
with (run_root / "metrics.jsonl").open() as handle:
    for line in handle:
        if line.strip():
            data = json.loads(line).get("data") or {}
            step = data.get("training/global_step")
            phase = data.get("actor/hybrid/phase")
            if step in (101, 150) and phase is not None:
                phase_by_step[step] = phase
expected = {101: 1.0, 150: 1.0}
if phase_by_step != expected:
    raise SystemExit(f"production hybrid telemetry mismatch: {phase_by_step}, expected {expected}")
print(json.dumps({"state": "complete", "soap_checkpoint_update": 400, "phase_by_step": phase_by_step}, sort_keys=True))
PY
trap - ERR
printf 'complete hybrid_production_verified step=150 %s\n' "$(date -u +%FT%TZ)" > "$STATUS"
