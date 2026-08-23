#!/usr/bin/env bash
# Wait for the source step-125 checkpoint and a GPU with the measured capture headroom.
set -euo pipefail
ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CAMPAIGN=${CAMPAIGN:-/data/users/shkodnik1917/rl_muon/jarvis-gsm8k-r4/campaign-4cdf62757063}
SOURCE_RUN=${SOURCE_RUN:-$CAMPAIGN/causal-hybrid-soap100-adamw-seed0/qwen2.5-0.5b_gsm8k_ppo_causal_soap100_adamw_seed0}
CHECKPOINT=$SOURCE_RUN/checkpoints/global_step_125
STATUS=${STATUS:-$CAMPAIGN/h-rlm-008-basis-lag-audit/capture-step-125-wait.status}
# 40 GiB conservative measured workload bound plus the mandatory 5 GiB residual.
MIN_FREE_MIB=${MIN_FREE_MIB:-46080}
mkdir -p "$(dirname "$STATUS")"

while [[ ! -s "$CHECKPOINT/actor/model_world_size_1_rank_0.pt" || ! -s "$CHECKPOINT/actor/optim_world_size_1_rank_0.pt" ]]; do
    printf 'waiting checkpoint=125 %s\n' "$(date -u +%FT%TZ)" > "$STATUS"
    sleep 30
done

while :; do
    selection=$(nvidia-smi --query-gpu=uuid,memory.free --format=csv,noheader,nounits |
        awk -F, -v minimum="$MIN_FREE_MIB" '{gsub(/ /,"",$1); gsub(/ /,"",$2); if ($2 >= minimum && $2 > best) {best=$2; uuid=$1}} END {if (uuid) print uuid, best}')
    if [[ -n "$selection" ]]; then
        read -r GPU_UUID free_mib <<<"$selection"
        remaining_mib=$((free_mib - 40960))
        printf 'launching checkpoint=125 gpu_uuid=%s free_mib=%s bound_mib=40960 remaining_mib=%s %s\n' \
            "$GPU_UUID" "$free_mib" "$remaining_mib" "$(date -u +%FT%TZ)" > "$STATUS"
        export GPU_UUID
        exec "$ROOT/capture_h_rlm_008_frozen_batches_vv.sh" 125
    fi
    printf 'waiting checkpoint=125 resource=min_free_%s_mib %s\n' "$MIN_FREE_MIB" "$(date -u +%FT%TZ)" > "$STATUS"
    sleep 30
done
