#!/usr/bin/env bash
set -euo pipefail

GPU_UUID=${GPU_UUID:-GPU-81dfcc32-ea97-c0f1-778e-ff44c4992c1a}
CAMPAIGN=${CAMPAIGN:-/data/users/shkodnik1917/rl_muon/jarvis-gsm8k-r4/campaign-4cdf62757063}
SOURCE=${SOURCE:-/data/users/shkodnik1917/rl_muon/causal-kfac-soap-current}
OUTPUT_ROOT=${OUTPUT_ROOT:-$CAMPAIGN/causal-kfac-soap-fast-ppo-mb4-gate25-seed0}
RUN_NAME=${RUN_NAME:-qwen2.5-0.5b_gsm8k_ppo_kl_matched_soap_seed0}
RUN_ROOT=$OUTPUT_ROOT/$RUN_NAME
STATUS=$OUTPUT_ROOT/harness.status
GPU_LOG=$OUTPUT_ROOT/gpu-memory.csv
mkdir -p "$OUTPUT_ROOT"

grep -q '^class KLMatchedSOAPThenAdamW' "$SOURCE/vendor/verl/verl/utils/kl_matched_soap.py" || {
    echo "reviewed hybrid optimizer is absent from SOURCE=$SOURCE" >&2
    exit 69
}

EXPECTED_STEP=${EXPECTED_STEP:-25}
SAVE_FREQ=${SAVE_FREQ:-25}
TEST_FREQ=${TEST_FREQ:-5}
if [[ "${ALLOW_RESUME:-0}" != 1 && ( -e "$RUN_ROOT/metrics.jsonl" || -e "$RUN_ROOT/checkpoints/global_step_$EXPECTED_STEP" ) ]]; then
    echo "refusing to overwrite an existing gate run: $RUN_ROOT" >&2
    exit 73
fi

resolved_uuid=$(nvidia-smi --id="$GPU_UUID" --query-gpu=uuid --format=csv,noheader,nounits | tr -d '[:space:]')
[[ "$resolved_uuid" == "$GPU_UUID" ]] || {
    echo "GPU UUID mismatch: requested=$GPU_UUID resolved=$resolved_uuid" >&2
    exit 66
}

baseline=$(nvidia-smi --id="$GPU_UUID" --query-gpu=memory.used,memory.free,memory.total --format=csv,noheader,nounits)
printf 'timestamp,memory_used_mib,memory_free_mib,memory_total_mib\n' > "$GPU_LOG"
printf 'running seed=0 gpu_uuid=%s baseline=%s %s\n' "$GPU_UUID" "$baseline" "$(date -u +%FT%TZ)" > "$STATUS"

training_pid=
monitor_pid=
cleanup() {
    code=$?
    trap - EXIT INT TERM
    if [[ -n "$monitor_pid" ]]; then
        kill "$monitor_pid" 2>/dev/null || true
        wait "$monitor_pid" 2>/dev/null || true
    fi
    if [[ -n "$training_pid" ]]; then
        kill -TERM -- "-$training_pid" 2>/dev/null || true
        wait "$training_pid" 2>/dev/null || true
    fi
    exit "$code"
}
trap cleanup EXIT INT TERM

monitor_gpu() {
    while :; do
        sample=$(nvidia-smi --id="$GPU_UUID" --query-gpu=memory.used,memory.free,memory.total --format=csv,noheader,nounits || true)
        printf '%s,%s\n' "$(date -u +%FT%TZ)" "$sample" >> "$GPU_LOG"
        free_mib=$(printf '%s\n' "$sample" | cut -d, -f2 | tr -d '[:space:]')
        if [[ "$free_mib" =~ ^[0-9]+$ ]] && (( free_mib < 5120 )); then
            printf 'failed gpu_safety_reserve free_mib=%s %s\n' "$free_mib" "$(date -u +%FT%TZ)" > "$STATUS"
            kill -TERM -- "-$training_pid" 2>/dev/null || true
            return
        fi
        sleep 2
    done
}

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU_UUID"
export RL_MUON_CAMPAIGN_ROOT="$CAMPAIGN"
export RL_MUON_VERL_ROOT="$SOURCE/vendor/verl"
export OUTPUT_ROOT
export SEED=0 EXPECTED_STEP SAVE_FREQ TEST_FREQ RUN_NAME
export PPO_MICRO_BATCH_SIZE=4
export FISHER_PROMPT_INDICES='[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]'
export FISHER_MICRO_BATCH_SIZE=4 FISHER_PROBE_COUNT=2 FISHER_PROBE_SEED=0
export FISHER_EXPECTED_STATES=57 FISHER_FACTOR_RANK=16 FISHER_DENSE_THRESHOLD=256 FISHER_REFRESH_FREQUENCY=4
export GPU_MEMORY_UTILIZATION=0.65 RL_MUON_VLLM_KV_CACHE_CAP_MIB=512
export TMPDIR=/tmp/rmb4g25t-$$
export RAY_TMPDIR=/tmp/rmb4g25r-$$
rm -rf "$TMPDIR" "$RAY_TMPDIR"
mkdir -p "$TMPDIR" "$RAY_TMPDIR"

setsid bash "$SOURCE/run_matched_soap_config_adamw.sh" > "$OUTPUT_ROOT/train.log" 2>&1 &
training_pid=$!
monitor_gpu &
monitor_pid=$!
if wait "$training_pid"; then
    training_pid=
else
    code=$?
    training_pid=
    if ! grep -q '^failed gpu_safety_reserve' "$STATUS"; then
        printf 'failed exit=%s %s\n' "$code" "$(date -u +%FT%TZ)" > "$STATUS"
    fi
    exit "$code"
fi

"$CAMPAIGN/venv/bin/python3" - "$RUN_ROOT" <<'PY'
import json
import sys
from pathlib import Path

run_root = Path(sys.argv[1])
metrics = run_root / "metrics.jsonl"
last_step = None
with metrics.open() as handle:
    for line in handle:
        if not line.strip():
            continue
        row = json.loads(line)
        data = row.get("data") or {}
        step = data.get("training/global_step", row.get("step"))
        if isinstance(step, int):
            last_step = step
expected_step = int(__import__("os").environ["EXPECTED_STEP"])
if last_step != expected_step:
    raise SystemExit(f"expected terminal step {expected_step}, got {last_step}")
checkpoint = run_root / "checkpoints" / f"global_step_{expected_step}"
required = [
    "actor/model_world_size_1_rank_0.pt",
    "actor/optim_world_size_1_rank_0.pt",
    "actor/extra_state_world_size_1_rank_0.pt",
    "critic/model_world_size_1_rank_0.pt",
    "critic/optim_world_size_1_rank_0.pt",
    "critic/extra_state_world_size_1_rank_0.pt",
    "data.pt",
]
missing = [name for name in required if not (checkpoint / name).is_file() or (checkpoint / name).stat().st_size == 0]
if missing:
    raise SystemExit(f"missing terminal artifacts: {missing}")
PY

printf 'complete seed=0 step=%s gpu_uuid=%s %s\n' "$EXPECTED_STEP" "$GPU_UUID" "$(date -u +%FT%TZ)" > "$STATUS"
