#!/usr/bin/env bash
set -euo pipefail

SCRIPT_ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
for seed in 0 1 2; do
    SEED="$seed" bash "$SCRIPT_ROOT/run_kl_matched_soap_seed.sh" "$@"
done
