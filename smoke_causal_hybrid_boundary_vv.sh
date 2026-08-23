#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CAMPAIGN=${CAMPAIGN:-/data/users/shkodnik1917/rl_muon/jarvis-gsm8k-r4/campaign-4cdf62757063}
OUTPUT_ROOT=${OUTPUT_ROOT:-$CAMPAIGN/causal-hybrid-boundary-smoke-seed0}
RUN_NAME=${RUN_NAME:-qwen2.5-0.5b_gsm8k_ppo_causal_hybrid_boundary_seed0}
STATUS=$OUTPUT_ROOT/harness.status
export CAMPAIGN OUTPUT_ROOT RUN_NAME
export ACTOR_OPTIMIZER=KLMatchedSOAPThenAdamW
export ACTOR_OPTIMIZER_IMPL=verl.utils.kl_matched_soap
export SWITCH_AFTER_GLOBAL_STEP=1 OPTIMIZER_UPDATES_PER_GLOBAL_STEP=4
export SAVE_FREQ=1 TEST_FREQ=1
trap 'printf "failed hybrid_boundary_verification %s\n" "$(date -u +%FT%TZ)" > "$STATUS"' ERR

if [[ "${VERIFY_ONLY:-0}" != 1 ]]; then
    if [[ -e "$OUTPUT_ROOT/$RUN_NAME" ]]; then
        echo "refusing to overwrite hybrid boundary smoke: $OUTPUT_ROOT/$RUN_NAME" >&2
        exit 73
    fi
    EXPECTED_STEP=1 "$ROOT/launch_causal_fast_mb4_gate25_vv.sh"
    ALLOW_RESUME=1 EXPECTED_STEP=2 "$ROOT/launch_causal_fast_mb4_gate25_vv.sh"
fi

"$CAMPAIGN/venv/bin/python3" - "$OUTPUT_ROOT/$RUN_NAME" <<'PY'
import sys
from pathlib import Path
import torch

run_root = Path(sys.argv[1])

def optimizer_state(step):
    path = run_root / "checkpoints" / f"global_step_{step}" / "actor" / "optim_world_size_1_rank_0.pt"
    return torch.load(path, map_location="cpu", weights_only=False)

soap = optimizer_state(1)
adamw = optimizer_state(2)
soap_meta = soap["kl_matched_soap"]
adamw_meta = adamw["kl_matched_soap"]
if (soap_meta["update_generation"], soap_meta["latest_telemetry"]["actor/hybrid/phase"]) != (4, 0.0):
    raise SystemExit("step-1 checkpoint is not the terminal SOAP phase")
if (adamw_meta["update_generation"], adamw_meta["latest_telemetry"]["actor/hybrid/phase"]) != (8, 1.0):
    raise SystemExit("step-2 checkpoint is not the resumed AdamW phase")
if any("hybrid_adamw_step" in state for state in soap["state"].values()):
    raise SystemExit("fresh AdamW state appeared before the boundary")
adamw_steps = {state.get("hybrid_adamw_step") for state in adamw["state"].values() if "hybrid_adamw_step" in state}
if adamw_steps != {4}:
    raise SystemExit(f"unexpected resumed AdamW state: {adamw_steps}")
for step in (1, 2):
    checkpoint = run_root / "checkpoints" / f"global_step_{step}" / "actor"
    for name in ("model_world_size_1_rank_0.pt", "optim_world_size_1_rank_0.pt", "extra_state_world_size_1_rank_0.pt"):
        artifact = checkpoint / name
        if not artifact.is_file() or artifact.stat().st_size == 0:
            raise SystemExit(f"missing checkpoint artifact: {artifact}")
print("hybrid boundary checkpoint/resume verified: SOAP updates=4, AdamW updates=4")
PY
trap - ERR
printf 'complete hybrid_boundary_verified %s\n' "$(date -u +%FT%TZ)" > "$STATUS"
