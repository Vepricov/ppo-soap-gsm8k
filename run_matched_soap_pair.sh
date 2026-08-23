#!/usr/bin/env bash
set -euo pipefail

SEED=${SEED:?SEED is required}
[[ "$SEED" =~ ^[0-9]+$ ]] || { echo "SEED must be a nonnegative integer" >&2; exit 64; }
EXPECTED_STEP=${EXPECTED_STEP:-150}
SCRIPT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CAMPAIGN_ROOT=${RL_MUON_CAMPAIGN_ROOT:?RL_MUON_CAMPAIGN_ROOT is required}
OUTPUT_ROOT=${OUTPUT_ROOT:-$CAMPAIGN_ROOT/stateful-soap-replication-seed$SEED}
PAIR_STATUS=${PAIR_STATUS:-$OUTPUT_ROOT/pair.status}
mkdir -p "$OUTPUT_ROOT"

finish() {
    code=$?
    if (( code == 0 )); then state=complete; else state=failed; fi
    printf '%s seed=%s exit=%d %s\n' "$state" "$SEED" "$code" "$(date -u +%FT%TZ)" > "$PAIR_STATUS"
}
trap finish EXIT
printf 'running-adamw seed=%s %s\n' "$SEED" "$(date -u +%FT%TZ)" > "$PAIR_STATUS"

COMMON_ENV=(
    SEED="$SEED"
    EXPECTED_STEP="$EXPECTED_STEP"
    SAVE_FREQ=25
    TEST_FREQ=10
    RL_MUON_CAMPAIGN_ROOT="$CAMPAIGN_ROOT"
    OUTPUT_ROOT="$OUTPUT_ROOT"
    GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.20}"
)

env "${COMMON_ENV[@]}" \
    ACTOR_ROUTE=adamw \
    SKIP_ENDPOINT_EVAL=1 \
    TMPDIR="/dev/shm/rlm-soap-adamw-tmp-s$SEED" \
    RAY_TMPDIR="/dev/shm/rlm-soap-adamw-ray-s$SEED" \
    bash "$SCRIPT_ROOT/run_matched_soap_config_adamw.sh"

ADAMW_RUN="$OUTPUT_ROOT/qwen2.5-0.5b_gsm8k_ppo_adamw_replay_seed$SEED"
ADAMW_CHECKPOINT="$ADAMW_RUN/checkpoints/global_step_$EXPECTED_STEP/actor/model_world_size_1_rank_0.pt"
[[ -s "$ADAMW_CHECKPOINT" ]] || { echo "AdamW terminal checkpoint is missing" >&2; exit 70; }
printf 'running-soap seed=%s %s\n' "$SEED" "$(date -u +%FT%TZ)" > "$PAIR_STATUS"

env "${COMMON_ENV[@]}" \
    ACTOR_ROUTE=soap \
    ADAMW_ACTOR_CHECKPOINT="$ADAMW_CHECKPOINT" \
    SKIP_ENDPOINT_EVAL=0 \
    TMPDIR="/dev/shm/rlm-soap-actor-tmp-s$SEED" \
    RAY_TMPDIR="/dev/shm/rlm-soap-actor-ray-s$SEED" \
    bash "$SCRIPT_ROOT/run_matched_soap_config_adamw.sh"

SOAP_RUN="$OUTPUT_ROOT/qwen2.5-0.5b_gsm8k_ppo_soap_actor_adamw_critic_seed$SEED"
python3 "$SCRIPT_ROOT/evaluate_soap_gate.py" \
    --baseline-metrics "$ADAMW_RUN/metrics.jsonl" \
    --soap-metrics "$SOAP_RUN/metrics.jsonl" \
    --kl-artifact "$SOAP_RUN/exact_categorical_kl.json" \
    --expected-step "$EXPECTED_STEP" \
    --output "$OUTPUT_ROOT/gate.json"
