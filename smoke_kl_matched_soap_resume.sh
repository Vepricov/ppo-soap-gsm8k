#!/usr/bin/env bash
# Real one-update checkpoint followed by normal VERL auto-resume to update two.
set -euo pipefail

SCRIPT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
SEED=${SEED:-0}
CAMPAIGN_ROOT=${RL_MUON_CAMPAIGN_ROOT:-/home/shkodnik1917/rl_muon/jarvis-gsm8k-r4/campaign-4cdf62757063}
OUTPUT_ROOT=${OUTPUT_ROOT:-$CAMPAIGN_ROOT/kl-matched-soap-resume-smoke-$$}
run_dir="$OUTPUT_ROOT/qwen2.5-0.5b_gsm8k_ppo_kl_matched_soap_seed$SEED/checkpoints"
[[ ! -e "$run_dir/latest_checkpointed_iteration.txt" ]] || {
    echo "smoke OUTPUT_ROOT must not contain an existing resumable run: $OUTPUT_ROOT" >&2
    exit 64
}

common=(
    SEED="$SEED"
    SAVE_FREQ=1
    TEST_FREQ=1
    RL_MUON_CAMPAIGN_ROOT="$CAMPAIGN_ROOT"
    OUTPUT_ROOT="$OUTPUT_ROOT"
    GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.45}"
)
env "${common[@]}" EXPECTED_STEP=1 bash "$SCRIPT_ROOT/run_matched_soap_config_adamw.sh" "$@"

[[ -s "$run_dir/global_step_1/actor/optim_world_size_1_rank_0.pt" ]] || {
    echo "one-step actor optimizer checkpoint was not saved" >&2
    exit 70
}

# The unchanged run directory and trainer.resume_mode=auto are the resume
# contract. Step two must be produced from the restored step-one optimizer.
env "${common[@]}" EXPECTED_STEP=2 bash "$SCRIPT_ROOT/run_matched_soap_config_adamw.sh" "$@"
[[ -s "$run_dir/global_step_2/actor/optim_world_size_1_rank_0.pt" ]] || {
    echo "resume did not produce the step-two actor optimizer checkpoint" >&2
    exit 70
}
