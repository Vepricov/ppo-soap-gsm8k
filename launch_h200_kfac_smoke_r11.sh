#!/usr/bin/env bash
set -euo pipefail

SRC=/data/users/shkodnik1917/rl_muon/causal-kfac-soap-current
CAMPAIGN=/data/users/shkodnik1917/rl_muon/jarvis-gsm8k-r4/campaign-4cdf62757063
OUT=$CAMPAIGN/causal-kfac-soap-smoke-seed0-h200-gpu2-gmu063-r11
GPU_UUID=GPU-bbba09e7-ef87-452c-7d34-cb2a03510f2c
PROJECTED_NEED_MIB=35840
MIN_GPU_FREE_MIB=5120

mkdir -p "$OUT"
free=$(nvidia-smi --id="$GPU_UUID" --query-gpu=memory.free --format=csv,noheader,nounits)
free=${free//[[:space:]]/}
remaining=$((free - PROJECTED_NEED_MIB))
printf 'gpu_uuid=%s free=%s MiB projected_need=%s MiB projected_remaining=%s MiB\n' \
    "$GPU_UUID" "$free" "$PROJECTED_NEED_MIB" "$remaining" | tee "$OUT/placement.txt"
(( remaining >= MIN_GPU_FREE_MIB )) || exit 75

exec env \
    RL_MUON_CAMPAIGN_ROOT="$CAMPAIGN" \
    GPU_UUID="$GPU_UUID" \
    OUTPUT_ROOT="$OUT" \
    GPU_MEMORY_UTILIZATION=0.63 \
    ROLLOUT_MAX_MODEL_LEN=768 \
    ROLLOUT_MAX_NUM_SEQS=64 \
    ROLLOUT_MAX_NUM_BATCHED_TOKENS=2048 \
    MAX_GPU_DELTA_MIB="$PROJECTED_NEED_MIB" \
    MIN_GPU_FREE_MIB="$MIN_GPU_FREE_MIB" \
    bash "$SRC/run_opt_factorized_smoke.sh"
