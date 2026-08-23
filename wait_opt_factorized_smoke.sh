#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CAMPAIGN_ROOT=${RL_MUON_CAMPAIGN_ROOT:-/home/shkodnik/rl_muon/jarvis-gsm8k-r4/campaign-4cdf62757063}
OUTPUT_ROOT=${OUTPUT_ROOT:-$CAMPAIGN_ROOT/causal-kfac-soap-smoke-seed0-gmu020-reserve}
WAIT_LOG=${WAIT_LOG:-$OUTPUT_ROOT.wait.log}
PROJECTED_NEED_MIB=${PROJECTED_NEED_MIB:-33006}
MIN_GPU_FREE_MIB=${MIN_GPU_FREE_MIB:-5120}
MIN_START_FREE_MIB=${MIN_START_FREE_MIB:-$((PROJECTED_NEED_MIB + MIN_GPU_FREE_MIB))}
mkdir -p "$(dirname -- "$WAIT_LOG")"

while :; do
    best_free=-1
    best_uuid=
    while IFS=, read -r uuid free total; do
        uuid=${uuid//[[:space:]]/}
        free=${free//[[:space:]]/}
        total=${total//[[:space:]]/}
        [[ "$free" =~ ^[0-9]+$ && "$total" =~ ^[0-9]+$ ]] || continue
        (( total >= 40000 && total <= 42000 )) || continue
        if (( free > best_free )); then
            best_free=$free
            best_uuid=$uuid
        fi
    done < <(nvidia-smi --query-gpu=uuid,memory.free,memory.total --format=csv,noheader,nounits)

    printf '%s best_a100_40gb=%s free=%s MiB required=%s MiB\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${best_uuid:-none}" "$best_free" \
        "$MIN_START_FREE_MIB" >>"$WAIT_LOG"
    if [[ -n "$best_uuid" ]] && (( best_free >= MIN_START_FREE_MIB )); then
        sleep 5
        rechecked_free=$(nvidia-smi --id="$best_uuid" --query-gpu=memory.free --format=csv,noheader,nounits)
        rechecked_free=${rechecked_free//[[:space:]]/}
        if [[ "$rechecked_free" =~ ^[0-9]+$ ]] && (( rechecked_free >= MIN_START_FREE_MIB )); then
            printf '%s launching gpu=%s free=%s MiB projected_need=%s MiB projected_remaining=%s MiB\n' \
                "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$best_uuid" "$rechecked_free" \
                "$PROJECTED_NEED_MIB" "$((rechecked_free - PROJECTED_NEED_MIB))" >>"$WAIT_LOG"
            exec env \
                RL_MUON_CAMPAIGN_ROOT="$CAMPAIGN_ROOT" \
                GPU_UUID="$best_uuid" \
                OUTPUT_ROOT="$OUTPUT_ROOT" \
                GPU_MEMORY_UTILIZATION=0.12 \
                MAX_GPU_USED_MIB=35840 \
                MIN_GPU_FREE_MIB="$MIN_GPU_FREE_MIB" \
                bash "$SCRIPT_ROOT/run_opt_factorized_smoke.sh"
        fi
    fi
    sleep 10
done