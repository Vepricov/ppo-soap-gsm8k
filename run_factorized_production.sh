#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SEED=${SEED:?SEED must be one of 0, 1, or 2}
case "$SEED" in 0|1|2) ;; *) echo "SEED must be one of 0, 1, or 2" >&2; exit 64 ;; esac
CAMPAIGN_ROOT=${RL_MUON_CAMPAIGN_ROOT:-/data/users/shkodnik1917/rl_muon/jarvis-gsm8k-r4/campaign-4cdf62757063}
GPU_UUID=${GPU_UUID:?GPU_UUID must identify the selected physical GPU}
OUTPUT_ROOT=${OUTPUT_ROOT:-$CAMPAIGN_ROOT/causal-kfac-soap-production-seed$SEED}
STATUS_PATH=$OUTPUT_ROOT/harness.status
GPU_LOG=$OUTPUT_ROOT/gpu-memory.csv
MAX_GPU_DELTA_MIB=${MAX_GPU_DELTA_MIB:-35840}
MIN_GPU_FREE_MIB=${MIN_GPU_FREE_MIB:-5120}
mkdir -p "$OUTPUT_ROOT"
printf 'running seed=%s gpu_uuid=%s %s\n' "$SEED" "$GPU_UUID" "$(date -u +%FT%TZ)" >"$STATUS_PATH"
printf 'timestamp,memory_used_mib,memory_free_mib,memory_total_mib,own_memory_used_mib\n' >"$GPU_LOG"

resolved_uuid=$(nvidia-smi --id="$GPU_UUID" --query-gpu=uuid --format=csv,noheader,nounits)
resolved_uuid=${resolved_uuid//[[:space:]]/}
[[ "$resolved_uuid" == "$GPU_UUID" ]] || {
    echo "physical GPU UUID mismatch: requested=$GPU_UUID resolved=$resolved_uuid" >&2
    exit 66
}
baseline_sample=$(nvidia-smi --id="$GPU_UUID" --query-gpu=memory.used --format=csv,noheader,nounits)
BASELINE_GPU_USED_MIB=${baseline_sample//[[:space:]]/}
[[ "$BASELINE_GPU_USED_MIB" =~ ^[0-9]+$ ]]
printf 'gpu_uuid=%s\nbaseline_gpu_used_mib=%s\n' "$GPU_UUID" "$BASELINE_GPU_USED_MIB" >"$OUTPUT_ROOT/gpu-memory-baseline.txt"

process_belongs_to_harness() {
    local pid=$1 entry
    [[ -r "/proc/$pid/environ" ]] || return 1
    while IFS= read -r -d '' entry; do
        [[ "$entry" == "OUTPUT_ROOT=$OUTPUT_ROOT" ]] && return 0
    done <"/proc/$pid/environ"
    return 1
}

own_gpu_memory_mib() {
    local total=0 uuid pid memory
    while IFS=, read -r uuid pid memory; do
        uuid=${uuid//[[:space:]]/}
        pid=${pid//[[:space:]]/}
        memory=${memory//[[:space:]]/}
        [[ "$uuid" == "$GPU_UUID" && "$pid" =~ ^[0-9]+$ && "$memory" =~ ^[0-9]+$ ]] || continue
        if process_belongs_to_harness "$pid"; then
            total=$((total + memory))
        fi
    done < <(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits 2>/dev/null || true)
    printf '%s\n' "$total"
}

training_pid=
monitor_pid=
monitor_gpu() {
    while :; do
        timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)
        sample=$(nvidia-smi --id="$GPU_UUID" --query-gpu=memory.used,memory.free,memory.total --format=csv,noheader,nounits || true)
        own_used=$(own_gpu_memory_mib)
        printf '%s,%s,%s\n' "$timestamp" "$sample" "$own_used" >>"$GPU_LOG"
        IFS=, read -r used free total <<<"$sample"
        used=${used//[[:space:]]/}
        free=${free//[[:space:]]/}
        if [[ "$own_used" =~ ^[0-9]+$ ]] && (( own_used > MAX_GPU_DELTA_MIB )); then
            printf 'gpu memory estimate exceeded: own_used=%s MiB estimate=%s MiB; continuing while free memory remains above the safety reserve\n' \
                "$own_used" "$MAX_GPU_DELTA_MIB" \
                >"$OUTPUT_ROOT/memory-estimate-warning.txt"
        fi
        if [[ "$used" =~ ^[0-9]+$ && "$free" =~ ^[0-9]+$ ]] && \
           (( free < MIN_GPU_FREE_MIB )); then
            printf 'gpu memory guard exceeded: total_used=%s MiB own_used=%s MiB own_cap=%s MiB free=%s MiB minimum_free=%s MiB\n' \
                "$used" "$own_used" "$MAX_GPU_DELTA_MIB" "$free" "$MIN_GPU_FREE_MIB" \
                >"$OUTPUT_ROOT/memory-cap-breach.txt"
            kill -TERM -- "-$training_pid" 2>/dev/null || true
            return
        fi
        sleep 2
    done
}

cleanup() {
    if [[ -n "$training_pid" ]]; then kill -TERM -- "-$training_pid" 2>/dev/null || true; fi
    if [[ -n "$monitor_pid" ]]; then
        kill "$monitor_pid" 2>/dev/null || true
        wait "$monitor_pid" 2>/dev/null || true
    fi
    local peak=0 peak_own=0 timestamp used free total own_used
    while IFS=, read -r timestamp used free total own_used; do
        used=${used//[[:space:]]/}
        own_used=${own_used//[[:space:]]/}
        [[ "$used" =~ ^[0-9]+$ ]] && (( used > peak )) && peak=$used
        [[ "$own_used" =~ ^[0-9]+$ ]] && (( own_used > peak_own )) && peak_own=$own_used
    done <"$GPU_LOG"
    printf 'peak_memory_used_mib=%s\nbaseline_memory_used_mib=%s\npeak_device_delta_memory_used_mib=%s\npeak_own_memory_used_mib=%s\n' \
        "$peak" "$BASELINE_GPU_USED_MIB" "$((peak - BASELINE_GPU_USED_MIB))" "$peak_own" \
        >"$OUTPUT_ROOT/gpu-memory-peak.txt"
}
trap cleanup EXIT

terminal_artifacts_complete() {
    "$CAMPAIGN_ROOT/venv/bin/python3" - "$OUTPUT_ROOT" "$SEED" <<'PY'
import json
import sys
from pathlib import Path

output_root = Path(sys.argv[1])
seed = int(sys.argv[2])
metrics_paths = list(output_root.glob(f"*seed{seed}/metrics.jsonl"))
if len(metrics_paths) != 1:
    raise SystemExit(1)
run_root = metrics_paths[0].parent
last_step = None
with metrics_paths[0].open() as handle:
    for line in handle:
        if not line.strip():
            continue
        row = json.loads(line)
        data = row.get("data") or {}
        step = data.get("training/global_step", row.get("step"))
        if isinstance(step, int):
            last_step = step
if last_step != 150:
    raise SystemExit(1)
checkpoint = run_root / "checkpoints" / "global_step_150"
required = [
    "actor/model_world_size_1_rank_0.pt",
    "actor/optim_world_size_1_rank_0.pt",
    "actor/extra_state_world_size_1_rank_0.pt",
    "critic/model_world_size_1_rank_0.pt",
    "critic/optim_world_size_1_rank_0.pt",
    "critic/extra_state_world_size_1_rank_0.pt",
    "data.pt",
]
if not all((checkpoint / name).is_file() and (checkpoint / name).stat().st_size > 0 for name in required):
    raise SystemExit(1)
PY
}

export CUDA_VISIBLE_DEVICES="$GPU_UUID"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export RL_MUON_CAMPAIGN_ROOT="$CAMPAIGN_ROOT"
export RL_MUON_VERL_ROOT="$SCRIPT_ROOT/vendor/verl"
export OUTPUT_ROOT
export GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.85}
export RL_MUON_VLLM_KV_CACHE_CAP_MIB=${RL_MUON_VLLM_KV_CACHE_CAP_MIB:-512}
export TMPDIR=${TMPDIR:-/tmp/rlm-kfac-prod-tmp-s$SEED-$$}
export RAY_TMPDIR=${RAY_TMPDIR:-/tmp/rlm-kfac-prod-ray-s$SEED-$$}
rm -rf "$TMPDIR" "$RAY_TMPDIR"
mkdir -p "$TMPDIR" "$RAY_TMPDIR"

setsid env \
    SEED="$SEED" \
    FISHER_PROMPT_INDICES="[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]" \
    FISHER_MICRO_BATCH_SIZE=1 \
    FISHER_PROBE_COUNT=4 \
    FISHER_PROBE_SEED=0 \
    FISHER_EXPECTED_STATES=57 \
    FISHER_FACTOR_RANK=16 \
    FISHER_DENSE_THRESHOLD=256 \
    FISHER_REFRESH_FREQUENCY=4 \
    bash "$SCRIPT_ROOT/run_kl_matched_soap_seed.sh" \
    critic.fsdp.param_offload=True \
    critic.fsdp.optimizer_offload=True &
training_pid=$!
monitor_gpu &
monitor_pid=$!
if wait "$training_pid"; then
    training_pid=
    printf 'complete seed=%s gpu_uuid=%s %s\n' "$SEED" "$GPU_UUID" "$(date -u +%FT%TZ)" >"$STATUS_PATH"
else
    code=$?
    training_pid=
    if terminal_artifacts_complete; then
        printf 'complete seed=%s gpu_uuid=%s recovered_after_exit=%s %s\n' \
            "$SEED" "$GPU_UUID" "$code" "$(date -u +%FT%TZ)" >"$STATUS_PATH"
    else
        printf 'failed seed=%s gpu_uuid=%s exit=%s %s\n' "$SEED" "$GPU_UUID" "$code" "$(date -u +%FT%TZ)" >"$STATUS_PATH"
        exit "$code"
    fi
fi
