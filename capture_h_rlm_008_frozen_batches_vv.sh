#!/usr/bin/env bash
# Generate three disjoint PPO batches from one frozen actor/critic checkpoint.
# Audit mode skips both actor and critic optimizer updates; this is rollout capture, not training.
set -euo pipefail

SOURCE_STEP=${1:?usage: capture_h_rlm_008_frozen_batches_vv.sh SOURCE_STEP}
case "$SOURCE_STEP" in
    25|75|125) ;;
    *) echo "SOURCE_STEP must be 25, 75, or 125" >&2; exit 64 ;;
esac

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CAMPAIGN=${CAMPAIGN:-/data/users/shkodnik1917/rl_muon/jarvis-gsm8k-r4/campaign-4cdf62757063}
SOURCE=${SOURCE:-/data/users/shkodnik1917/rl_muon/causal-kfac-soap-current}
SOURCE_RUN=${SOURCE_RUN:-$CAMPAIGN/causal-hybrid-soap100-adamw-seed0/qwen2.5-0.5b_gsm8k_ppo_causal_soap100_adamw_seed0}
SOURCE_CHECKPOINT=$SOURCE_RUN/checkpoints/global_step_$SOURCE_STEP
CAPTURE_BASE=${CAPTURE_BASE:-$CAMPAIGN/h-rlm-008-basis-lag-audit/capture-step-$SOURCE_STEP}
CAPTURE_ROOT=$CAPTURE_BASE/frozen-batches
RUN_NAME=h_rlm_008_capture_step_$SOURCE_STEP

for path in \
    "$SOURCE_CHECKPOINT/actor/model_world_size_1_rank_0.pt" \
    "$SOURCE_CHECKPOINT/actor/optim_world_size_1_rank_0.pt" \
    "$SOURCE_CHECKPOINT/critic/model_world_size_1_rank_0.pt" \
    "$SOURCE_CHECKPOINT/data.pt"; do
    [[ -s "$path" ]] || { echo "missing source checkpoint artifact: $path" >&2; exit 66; }
done
[[ ! -e "$CAPTURE_BASE/$RUN_NAME" ]] || {
    echo "refusing to overwrite frozen capture run: $CAPTURE_BASE/$RUN_NAME" >&2
    exit 73
}
mkdir -p "$CAPTURE_ROOT"

export SOURCE CAMPAIGN OUTPUT_ROOT="$CAPTURE_BASE" RUN_NAME
export EXPECTED_STEP=$((SOURCE_STEP + 3)) SAVE_FREQ=-1 TEST_FREQ=-1
export RESUME_MODE=resume_path RESUME_FROM_PATH="$SOURCE_CHECKPOINT"
export SAVE_INITIAL_CHECKPOINT=False VAL_BEFORE_TRAIN=False
export H_RLM_008_FROZEN_CAPTURE_ROOT="$CAPTURE_ROOT"
export H_RLM_008_SOURCE_STEP="$SOURCE_STEP"
export ALLOW_RESUME=1

"$ROOT/launch_causal_fast_mb4_gate25_vv.sh"

for step in $(seq $((SOURCE_STEP + 1)) $((SOURCE_STEP + 3))); do
    [[ -s "$CAPTURE_ROOT/global_step_$step.dp" ]] || {
        echo "capture missing global step $step" >&2
        exit 70
    }
done
printf 'complete frozen_capture source_step=%s batches=3 model_optimizer_updates=0 %s\n' \
    "$SOURCE_STEP" "$(date -u +%FT%TZ)" > "$CAPTURE_BASE/capture.status"
