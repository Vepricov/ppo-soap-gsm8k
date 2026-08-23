#!/usr/bin/env bash
set -euo pipefail

SEED=${SEED:?SEED must be one of 0, 1, or 2}
case "$SEED" in 0|1|2) ;; *) echo "SEED must be one of 0, 1, or 2" >&2; exit 64 ;; esac
SCRIPT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CAMPAIGN_ROOT=${RL_MUON_CAMPAIGN_ROOT:-/home/shkodnik1917/rl_muon/jarvis-gsm8k-r4/campaign-4cdf62757063}
OUTPUT_ROOT=${OUTPUT_ROOT:-$CAMPAIGN_ROOT/kl-matched-soap-production}

env \
    SEED="$SEED" \
    EXPECTED_STEP=150 \
    SAVE_FREQ=25 \
    TEST_FREQ=25 \
    RL_MUON_CAMPAIGN_ROOT="$CAMPAIGN_ROOT" \
    OUTPUT_ROOT="$OUTPUT_ROOT" \
    TMPDIR="${TMPDIR:-/dev/shm/rlm-kl-soap-tmp-s$SEED}" \
    RAY_TMPDIR="${RAY_TMPDIR:-/dev/shm/rlm-kl-soap-ray-s$SEED}" \
    GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.20}" \
    bash "$SCRIPT_ROOT/run_matched_soap_config_adamw.sh" "$@"

run_dir="$OUTPUT_ROOT/qwen2.5-0.5b_gsm8k_ppo_kl_matched_soap_seed$SEED/checkpoints"
for step in 0 25 50 75 100 125 150; do
    checkpoint="$run_dir/global_step_$step"
    [[ -s "$checkpoint/actor/model_world_size_1_rank_0.pt" ]] || {
        echo "missing actor model checkpoint at step $step" >&2
        exit 70
    }
    [[ -s "$checkpoint/actor/optim_world_size_1_rank_0.pt" ]] || {
        echo "missing actor optimizer checkpoint at step $step" >&2
        exit 70
    }
    [[ -s "$checkpoint/actor/extra_state_world_size_1_rank_0.pt" ]] || {
        echo "missing actor scheduler/RNG checkpoint at step $step" >&2
        exit 70
    }
    [[ -s "$checkpoint/critic/optim_world_size_1_rank_0.pt" ]] || {
        echo "missing critic optimizer checkpoint at step $step" >&2
        exit 70
    }
    [[ -s "$checkpoint/critic/extra_state_world_size_1_rank_0.pt" ]] || {
        echo "missing critic scheduler/RNG checkpoint at step $step" >&2
        exit 70
    }
    [[ -s "$checkpoint/data.pt" ]] || {
        echo "missing dataloader checkpoint at step $step" >&2
        exit 70
    }
done
