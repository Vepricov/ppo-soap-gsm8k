#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CAMPAIGN_ROOT=${RL_MUON_CAMPAIGN_ROOT:-/data/users/shkodnik1917/rl_muon/jarvis-gsm8k-r4/campaign-4cdf62757063}
STATE_ROOT=${STATE_ROOT:-$CAMPAIGN_ROOT/causal-kfac-factorized-comparison-exact-r6}
SOAP_OUTPUT_PREFIX=${SOAP_OUTPUT_PREFIX:-causal-kfac-soap-exact-r6-seed}
ADAMW_OUTPUT_PREFIX=${ADAMW_OUTPUT_PREFIX:-causal-kfac-adamw-exact-r6-seed}
CONTROL_SEED=${CONTROL_SEED:-2}
ORIGINAL_STATUS=$STATE_ROOT/supervisor.status
STATUS=$STATE_ROOT/single-control-finalizer.status
LOG=$STATE_ROOT/single-control-finalizer.log
CLAIM_ROOT=${CLAIM_ROOT:-/tmp/rl_muon_factorized_gpu_claims}
mkdir -p "$STATE_ROOT" "$CLAIM_ROOT"
exec >>"$LOG" 2>&1

finish() {
    code=$?
    if (( code == 0 )); then state=complete; else state=failed; fi
    printf '%s exit=%s %s\n' "$state" "$code" "$(date -u +%FT%TZ)" >"$STATUS"
}
trap finish EXIT

soap_root() {
    printf '%s/%s%s\n' "$CAMPAIGN_ROOT" "$SOAP_OUTPUT_PREFIX" "$1"
}

adamw_root() {
    printf '%s/%s%s\n' "$CAMPAIGN_ROOT" "$ADAMW_OUTPUT_PREFIX" "$1"
}

printf 'waiting-original-supervisor %s\n' "$(date -u +%FT%TZ)" >"$STATUS"
while :; do
    if [[ -s "$ORIGINAL_STATUS" ]]; then
        read -r original_state _ <"$ORIGINAL_STATUS"
        case "$original_state" in
            complete|failed) break ;;
        esac
    fi
    sleep 30
done

control_status=$(adamw_root "$CONTROL_SEED")/harness.status
read -r control_state _ <"$control_status"
if [[ "$control_state" != complete ]]; then
    echo "control seed $CONTROL_SEED is not complete: $(<"$control_status")" >&2
    exit 1
fi

printf 'running-exact-kl-common-control %s\n' "$(date -u +%FT%TZ)" >"$STATUS"
run_eval() {
    local seed=$1 selected= free= fd=
    local soap_run_root baseline_run_root soap_run baseline_run soap_metrics baseline_metrics
    soap_run_root=$(soap_root "$seed")
    baseline_run_root=$(adamw_root "$CONTROL_SEED")
    soap_run=$soap_run_root/qwen2.5-0.5b_gsm8k_ppo_kl_matched_soap_seed$seed
    baseline_run=$baseline_run_root/qwen2.5-0.5b_gsm8k_ppo_causal_adamw_control_seed$CONTROL_SEED
    soap_metrics=$soap_run/validation_trajectory.first_reached.jsonl
    baseline_metrics=$baseline_run/validation_trajectory.first_reached.jsonl
    "$CAMPAIGN_ROOT/venv/bin/python3" "$SCRIPT_ROOT/recover_first_reached_validation.py" \
        "$soap_run/train.log" "$soap_metrics" --expected-step 150 --interval 25
    "$CAMPAIGN_ROOT/venv/bin/python3" "$SCRIPT_ROOT/recover_first_reached_validation.py" \
        "$baseline_run/train.log" "$baseline_metrics" --expected-step 150 --interval 25
    while :; do
        while IFS=, read -r uuid candidate_free candidate_total; do
            uuid=${uuid//[[:space:]]/}
            candidate_free=${candidate_free//[[:space:]]/}
            candidate_total=${candidate_total//[[:space:]]/}
            [[ "$candidate_free" =~ ^[0-9]+$ && "$candidate_total" =~ ^[0-9]+$ ]] || continue
            (( candidate_total >= 80000 && candidate_free >= 13312 )) || continue
            exec {candidate_fd}>"$CLAIM_ROOT/${uuid}.lock"
            if flock -n "$candidate_fd"; then
                selected=$uuid
                free=$candidate_free
                fd=$candidate_fd
                break
            fi
            exec {candidate_fd}>&-
        done < <(nvidia-smi --query-gpu=uuid,memory.free,memory.total --format=csv,noheader,nounits | sort -t, -k2,2nr)
        [[ -n "$selected" ]] && break
        sleep 10
    done
    sleep 5
    free=$(nvidia-smi --id="$selected" --query-gpu=memory.free --format=csv,noheader,nounits | tr -d ' ')
    if (( free < 13312 )); then
        exec {fd}>&-
        run_eval "$seed"
        return
    fi

    printf 'soap_seed=%s control_seed=%s exact-kl gpu=%s free=%sMiB bound=8192MiB projected_remaining=%sMiB %s\n' \
        "$seed" "$CONTROL_SEED" "$selected" "$free" "$((free - 8192))" "$(date -u +%FT%TZ)"
    CUDA_VISIBLE_DEVICES="$selected" "$CAMPAIGN_ROOT/venv/bin/python3" \
        "$SCRIPT_ROOT/evaluate_exact_categorical_kl.py" \
        --model-path "$CAMPAIGN_ROOT/models/qwen2.5-0.5b-instruct" \
        --dataset "$CAMPAIGN_ROOT/data/gsm8k/test.parquet" \
        --baseline-checkpoint "$baseline_run/checkpoints/global_step_150/actor/model_world_size_1_rank_0.pt" \
        --soap-checkpoint "$soap_run/checkpoints/global_step_150/actor/model_world_size_1_rank_0.pt" \
        --expected-step 150 \
        --device cuda \
        --output "$soap_run/exact_categorical_kl_common_control_seed${CONTROL_SEED}.json"
    "$CAMPAIGN_ROOT/venv/bin/python3" "$SCRIPT_ROOT/evaluate_soap_gate.py" \
        --baseline-metrics "$baseline_metrics" \
        --soap-metrics "$soap_metrics" \
        --kl-artifact "$soap_run/exact_categorical_kl_common_control_seed${CONTROL_SEED}.json" \
        --expected-step 150 \
        --output "$soap_run_root/gate_common_control_seed${CONTROL_SEED}.json"
    exec {fd}>&-
}

for seed in 0 1 2; do
    run_eval "$seed"
done

"$CAMPAIGN_ROOT/venv/bin/python3" - "$STATE_ROOT" "$CONTROL_SEED" \
    "$(soap_root 0)/gate_common_control_seed${CONTROL_SEED}.json" \
    "$(soap_root 1)/gate_common_control_seed${CONTROL_SEED}.json" \
    "$(soap_root 2)/gate_common_control_seed${CONTROL_SEED}.json" <<'PY'
import json
import os
import sys
from pathlib import Path

state_root = Path(sys.argv[1])
control_seed = int(sys.argv[2])
gates = [json.loads(Path(path).read_text()) for path in sys.argv[3:]]
passing_seeds = [seed for seed, gate in enumerate(gates) if gate["decision"] == "GO"]
result = {
    "decision": "GO" if len(passing_seeds) >= 2 else "NO_GO",
    "control_seed": control_seed,
    "shared_control": True,
    "passing_soap_seeds": passing_seeds,
    "required_passing_soap_seeds": 2,
    "per_seed": gates,
}
output = state_root / "single_control_gate.json"
temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
os.replace(temporary, output)
print(json.dumps(result, sort_keys=True))
PY
