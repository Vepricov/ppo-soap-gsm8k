#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CAMPAIGN_ROOT=${RL_MUON_CAMPAIGN_ROOT:-/data/users/shkodnik1917/rl_muon/jarvis-gsm8k-r4/campaign-4cdf62757063}
STATE_ROOT=${STATE_ROOT:-$CAMPAIGN_ROOT/causal-kfac-factorized-comparison}
SOAP_OUTPUT_PREFIX=${SOAP_OUTPUT_PREFIX:-causal-kfac-soap-production-seed}
ADAMW_OUTPUT_PREFIX=${ADAMW_OUTPUT_PREFIX:-causal-kfac-adamw-production-seed}
CONTROL_SEED=${CONTROL_SEED:-2}
STATUS=$STATE_ROOT/supervisor.status
LOG=$STATE_ROOT/supervisor.log
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

wait_complete() {
    local path=$1 label=$2 state
    while :; do
        if [[ -s "$path" ]]; then
            read -r state _ <"$path"
            case "$state" in
                complete)
                    echo "$label complete $(date -u +%FT%TZ)"
                    return 0
                    ;;
                failed)
                    echo "$label failed: $(<"$path")" >&2
                    return 1
                    ;;
            esac
        fi
        sleep 30
    done
}

printf 'waiting-factorized %s\n' "$(date -u +%FT%TZ)" >"$STATUS"
for seed in 0 1 2; do
    root=$(soap_root "$seed")
    wait_complete "$root/harness.status" "factorized-seed$seed"
done

printf 'running-adamw-controls %s\n' "$(date -u +%FT%TZ)" >"$STATUS"
out=$(adamw_root "$CONTROL_SEED")
mkdir -p "$out"
env \
    SEED="$CONTROL_SEED" \
    RL_MUON_CAMPAIGN_ROOT="$CAMPAIGN_ROOT" \
    OUTPUT_ROOT="$out" \
    WAIT_LOG="$out/wait.log" \
    bash "$SCRIPT_ROOT/wait_factorized_adamw_production.sh" \
    >"$out/waiter.stdout.log" 2>&1
wait_complete "$out/harness.status" "adamw-seed$CONTROL_SEED"

printf 'running-exact-kl %s\n' "$(date -u +%FT%TZ)" >"$STATUS"
run_eval() {
    local seed=$1 selected= free= fd=
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

    local soap_run_root baseline_run_root soap_run baseline_run
    soap_run_root=$(soap_root "$seed")
    baseline_run_root=$(adamw_root "$CONTROL_SEED")
    soap_run=$soap_run_root/qwen2.5-0.5b_gsm8k_ppo_kl_matched_soap_seed$seed
    baseline_run=$baseline_run_root/qwen2.5-0.5b_gsm8k_ppo_causal_adamw_control_seed$CONTROL_SEED
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
        --baseline-metrics "$baseline_run/metrics.jsonl" \
        --soap-metrics "$soap_run/metrics.jsonl" \
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
