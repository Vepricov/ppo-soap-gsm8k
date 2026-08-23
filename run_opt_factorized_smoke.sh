#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CAMPAIGN_ROOT=${RL_MUON_CAMPAIGN_ROOT:-/home/shkodnik/rl_muon/jarvis-gsm8k-r4/campaign-4cdf62757063}
GPU_UUID=${GPU_UUID:-GPU-5430d3bb-f055-7a03-62f9-36ce1cf238c8}
OUTPUT_ROOT=${OUTPUT_ROOT:-$CAMPAIGN_ROOT/causal-kfac-soap-smoke-seed0}
STATUS_PATH=$OUTPUT_ROOT/harness.status
GPU_LOG=$OUTPUT_ROOT/gpu-memory.csv
MAX_GPU_DELTA_MIB=${MAX_GPU_DELTA_MIB:-45056}
MIN_GPU_FREE_MIB=${MIN_GPU_FREE_MIB:-5120}
mkdir -p "$OUTPUT_ROOT"
printf 'running\n' >"$STATUS_PATH"
printf 'timestamp,memory_used_mib,memory_free_mib,memory_total_mib,own_memory_used_mib\n' >"$GPU_LOG"
baseline_sample=$(nvidia-smi --id="$GPU_UUID" --query-gpu=memory.used --format=csv,noheader,nounits)
BASELINE_GPU_USED_MIB=${baseline_sample//[[:space:]]/}
[[ "$BASELINE_GPU_USED_MIB" =~ ^[0-9]+$ ]]
printf 'baseline_gpu_used_mib=%s\n' "$BASELINE_GPU_USED_MIB" >"$OUTPUT_ROOT/gpu-memory-baseline.txt"

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

monitor_gpu() {
    while :; do
        timestamp=$(date -u +%Y-%m-%dT%H:%M:%SZ)
        sample=$(nvidia-smi --id="$GPU_UUID" --query-gpu=memory.used,memory.free,memory.total --format=csv,noheader,nounits || true)
        own_used=$(own_gpu_memory_mib)
        printf '%s,%s,%s\n' "$timestamp" "$sample" "$own_used" >>"$GPU_LOG"
        IFS=, read -r used free total <<<"$sample"
        used=${used//[[:space:]]/}
        free=${free//[[:space:]]/}
        if [[ ! "$used" =~ ^[0-9]+$ || ! "$free" =~ ^[0-9]+$ ]]; then
            sleep 2
            continue
        fi
        if (( own_used > MAX_GPU_DELTA_MIB || free < MIN_GPU_FREE_MIB )); then
            printf 'gpu memory guard exceeded: total_used=%s MiB own_used=%s MiB own_cap=%s MiB free=%s MiB minimum_free=%s MiB\n' \
                "$used" "$own_used" "$MAX_GPU_DELTA_MIB" \
                "$free" "$MIN_GPU_FREE_MIB" >"$OUTPUT_ROOT/memory-cap-breach.txt"
            kill -TERM -- "-$training_pid" 2>/dev/null || true
            return
        fi
        sleep 2
    done
}
training_pid=
monitor_pid=
cleanup() {
    if [[ -n "$training_pid" ]]; then
        kill -TERM -- "-$training_pid" 2>/dev/null || true
    fi
    if [[ -n "$monitor_pid" ]]; then
        kill "$monitor_pid" 2>/dev/null || true
        wait "$monitor_pid" 2>/dev/null || true
    fi
    local peak=0 peak_own=0 timestamp used free total own_used
    while IFS=, read -r timestamp used free total own_used; do
        used=${used//[[:space:]]/}
        own_used=${own_used//[[:space:]]/}
        if [[ "$used" =~ ^[0-9]+$ ]] && (( used > peak )); then
            peak=$used
        fi
        if [[ "$own_used" =~ ^[0-9]+$ ]] && (( own_used > peak_own )); then
            peak_own=$own_used
        fi
    done <"$GPU_LOG"
    local peak_delta=$((peak - BASELINE_GPU_USED_MIB))
    printf 'peak_memory_used_mib=%s\nbaseline_memory_used_mib=%s\npeak_device_delta_memory_used_mib=%s\npeak_own_memory_used_mib=%s\n' \
        "$peak" "$BASELINE_GPU_USED_MIB" "$peak_delta" "$peak_own" >"$OUTPUT_ROOT/gpu-memory-peak.txt"
}
trap cleanup EXIT

export CUDA_VISIBLE_DEVICES="$GPU_UUID"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export RL_MUON_CAMPAIGN_ROOT="$CAMPAIGN_ROOT"
export RL_MUON_VERL_ROOT="$SCRIPT_ROOT/vendor/verl"
export OUTPUT_ROOT
export GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.12}
# Ray's Unix sockets must be private to this harness.  A second smoke used to
# rm the shared directory out from under a live raylet, leaving the driver and
# actors alive while every replacement worker failed to connect forever.
export TMPDIR=${TMPDIR:-/tmp/rlm-kfac-smoke-$$}
export RAY_TMPDIR=${RAY_TMPDIR:-/tmp/rlm-kfac-ray-$$}
rm -rf "$TMPDIR" "$RAY_TMPDIR"
mkdir -p "$TMPDIR" "$RAY_TMPDIR"

setsid bash "$SCRIPT_ROOT/smoke_kl_matched_soap_resume.sh" "$@" &
training_pid=$!
monitor_gpu &
monitor_pid=$!
if wait "$training_pid"; then
    training_pid=
    printf 'complete\n' >"$STATUS_PATH"
else
    code=$?
    training_pid=
    printf 'failed exit=%s\n' "$code" >"$STATUS_PATH"
    exit "$code"
fi
