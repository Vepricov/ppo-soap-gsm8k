#!/usr/bin/env bash
set -euo pipefail
mode=${1:?mode must be smoke or full}
seed=${2:-0}
case "$mode" in smoke|full) ;; *) exit 64;; esac
case "$seed" in 0|1|2) ;; *) exit 64;; esac
repo=$(cd "$(dirname "$0")" && pwd)
commit=${RL_MUON_SOURCE_COMMIT:?missing source commit}
[[ "$(git -C "$repo" rev-parse HEAD)" == "$commit" ]] || { echo source_commit_mismatch; exit 73; }
campaign=${RL_MUON_CAMPAIGN_ROOT:-/home/jovyan/rl_muon/campaign-5ec9878-145650}
[[ -s "$campaign/bootstrap/status.json" ]] || { echo campaign_missing; exit 75; }
export RL_MUON_CAMPAIGN_ROOT="$campaign"
export RL_MUON_VERL_ROOT="$repo/vendor/verl"
export SEED="$seed"
export GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.80}
export TMPDIR=/dev/shm/rlm-kl-soap-${mode}-s${seed}-tmp
export RAY_TMPDIR=/dev/shm/rlm-kl-soap-${mode}-s${seed}-ray
export OUTPUT_ROOT="$campaign/causal-kl-matched-${mode}-s${seed}-${RL_MUON_ATTEMPT:-cloud1}"
finish() {
  rc=$?
  state=failed; [[ $rc -eq 0 ]] && state=complete
  printf 'RL_MUON_TERMINAL {"state":"%s","phase":"%s","seed":%s,"source_commit":"%s","exit":%s}\n' "$state" "$mode" "$seed" "$commit" "$rc"
  exit 0
}
trap finish EXIT
if [[ "$mode" == smoke ]]; then
  bash "$repo/smoke_kl_matched_soap_resume.sh"
else
  bash "$repo/run_kl_matched_soap_seed.sh"
fi
