#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SEED=${SEED:?SEED must be one of 0, 1, or 2}
case "$SEED" in 0|1|2) ;; *) echo "SEED must be one of 0, 1, or 2" >&2; exit 64 ;; esac
CAMPAIGN_ROOT=${RL_MUON_CAMPAIGN_ROOT:-/data/users/shkodnik1917/rl_muon/jarvis-gsm8k-r4/campaign-4cdf62757063}
OUTPUT_ROOT=${OUTPUT_ROOT:-$CAMPAIGN_ROOT/causal-kfac-adamw-production-seed$SEED}
WAIT_LOG=${WAIT_LOG:-$OUTPUT_ROOT.wait.log}
PROJECTED_NEED_MIB=${PROJECTED_NEED_MIB:-35840}
MIN_GPU_FREE_MIB=${MIN_GPU_FREE_MIB:-5120}
MIN_START_FREE_MIB=$((PROJECTED_NEED_MIB + MIN_GPU_FREE_MIB))
HOST_RAM_PEAK_MIB=${HOST_RAM_PEAK_MIB:-161500}
HOST_RAM_RESERVE_MIB=${HOST_RAM_RESERVE_MIB:-20480}
MIN_MEM_AVAILABLE_MIB=$((HOST_RAM_PEAK_MIB + HOST_RAM_RESERVE_MIB))
CLAIM_ROOT=${CLAIM_ROOT:-/tmp/rl_muon_factorized_gpu_claims}
mkdir -p "$(dirname -- "$WAIT_LOG")" "$CLAIM_ROOT"

while :; do
    mem_available_kib=$(python3 - <<'PY'
from pathlib import Path
for line in Path('/proc/meminfo').read_text().splitlines():
    if line.startswith('MemAvailable:'):
        print(line.split()[1])
        break
PY
)
    mem_available_mib=$((mem_available_kib / 1024))
    selected=
    selected_free=-1
    selected_fd=
    while IFS=, read -r uuid free total; do
        uuid=${uuid//[[:space:]]/}
        free=${free//[[:space:]]/}
        total=${total//[[:space:]]/}
        [[ "$free" =~ ^[0-9]+$ && "$total" =~ ^[0-9]+$ ]] || continue
        (( total >= 80000 && free >= MIN_START_FREE_MIB )) || continue
        lock_path="$CLAIM_ROOT/${uuid}.lock"
        exec {candidate_fd}>"$lock_path"
        if flock -n "$candidate_fd"; then
            selected=$uuid
            selected_free=$free
            selected_fd=$candidate_fd
            break
        fi
        exec {candidate_fd}>&-
    done < <(nvidia-smi --query-gpu=uuid,memory.free,memory.total --format=csv,noheader,nounits | sort -t, -k2,2nr)

    printf '%s seed=%s selected=%s free=%s MiB required=%s MiB mem_available=%s MiB required_mem=%s MiB\n' \
        "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$SEED" "${selected:-none}" "$selected_free" \
        "$MIN_START_FREE_MIB" "$mem_available_mib" "$MIN_MEM_AVAILABLE_MIB" >>"$WAIT_LOG"

    if [[ -n "$selected" ]] && (( mem_available_mib >= MIN_MEM_AVAILABLE_MIB )); then
        sleep 5
        rechecked_free=$(nvidia-smi --id="$selected" --query-gpu=memory.free --format=csv,noheader,nounits)
        rechecked_free=${rechecked_free//[[:space:]]/}
        if [[ "$rechecked_free" =~ ^[0-9]+$ ]] && (( rechecked_free >= MIN_START_FREE_MIB )); then
            printf '%s seed=%s launching gpu_uuid=%s free=%s MiB projected_need=%s MiB projected_remaining=%s MiB mem_available=%s MiB\n' \
                "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$SEED" "$selected" "$rechecked_free" \
                "$PROJECTED_NEED_MIB" "$((rechecked_free - PROJECTED_NEED_MIB))" "$mem_available_mib" \
                >>"$WAIT_LOG"
            env \
                SEED="$SEED" \
                RL_MUON_CAMPAIGN_ROOT="$CAMPAIGN_ROOT" \
                GPU_UUID="$selected" \
                OUTPUT_ROOT="$OUTPUT_ROOT" \
                GPU_MEMORY_UTILIZATION=0.85 \
                RL_MUON_VLLM_KV_CACHE_CAP_MIB=512 \
                MAX_GPU_DELTA_MIB="$PROJECTED_NEED_MIB" \
                MIN_GPU_FREE_MIB="$MIN_GPU_FREE_MIB" \
                bash "$SCRIPT_ROOT/run_factorized_adamw_production.sh"
            exit $?
        fi
    fi
    if [[ -n "${selected_fd:-}" ]]; then exec {selected_fd}>&-; fi
    sleep 10
done
