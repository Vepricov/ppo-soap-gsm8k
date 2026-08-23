#!/usr/bin/env bash
set -euo pipefail

SEED=${SEED:?SEED is required}
GPU_INDEX=${GPU_INDEX:?GPU_INDEX is required}
CAMPAIGN_ROOT=${RL_MUON_CAMPAIGN_ROOT:?RL_MUON_CAMPAIGN_ROOT is required}
SCRIPT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SOURCE=$SCRIPT_ROOT/run_matched_soap_pair.sh
OUTPUT_ROOT=${OUTPUT_ROOT:-$CAMPAIGN_ROOT/stateful-soap-replication-seed$SEED}
PAIR_STATUS=${PAIR_STATUS:-$OUTPUT_ROOT/pair.status}
LOG=${LOG:-$OUTPUT_ROOT/pair.log}
mkdir -p "$OUTPUT_ROOT"
exec >>"$LOG" 2>&1

IFS=, read -r used total < <(
    nvidia-smi -i "$GPU_INDEX" --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | tr -d ' '
)
free_mib=$((total-used))
needed_mib=40960
remaining_mib=$((free_mib-needed_mib))
printf 'gpu=%d used=%dMiB total=%dMiB free=%dMiB needed_bound=%dMiB remaining=%dMiB\n' \
    "$GPU_INDEX" "$used" "$total" "$free_mib" "$needed_mib" "$remaining_mib"
(( remaining_mib >= 5120 )) || { echo 'GPU headroom gate failed' >&2; exit 75; }

available_kib=$(python3 - <<'PY'
from pathlib import Path
for line in Path('/proc/meminfo').read_text().splitlines():
    if line.startswith('MemAvailable:'):
        print(line.split()[1])
        break
else:
    raise SystemExit('MemAvailable missing')
PY
)
required_kib=$((64*1024*1024))
printf 'host_available_kib=%d host_required_kib=%d\n' "$available_kib" "$required_kib"
(( available_kib >= required_kib )) || { echo 'Host RAM gate failed' >&2; exit 75; }
[[ $(awk '/full / {sub("avg10=", "", $2); print (($2 + 0) <= 0.01 ? 1 : 0)}' /proc/pressure/memory) == 1 ]] || {
    echo 'Host memory pressure gate failed' >&2
    exit 75
}

env \
    SEED="$SEED" \
    RL_MUON_CAMPAIGN_ROOT="$CAMPAIGN_ROOT" \
    OUTPUT_ROOT="$OUTPUT_ROOT" \
    PAIR_STATUS="$PAIR_STATUS" \
    CUDA_VISIBLE_DEVICES="$GPU_INDEX" \
    GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.45}" \
    bash "$SOURCE"
