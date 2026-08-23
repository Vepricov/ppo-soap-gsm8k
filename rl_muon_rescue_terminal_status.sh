#!/usr/bin/env bash
set -euo pipefail
OUTPUT_ROOT=${1:?output root required}
SEED=${2:?seed required}
VALIDATOR=${3:?validator required}
STATUS=$OUTPUT_ROOT/harness.status
while :; do
    if [[ -s "$STATUS" ]]; then
        read -r state _ <"$STATUS"
        case "$state" in
            complete)
                exit 0
                ;;
            failed)
                python3 "$VALIDATOR" "$OUTPUT_ROOT" "$SEED"
                printf 'complete seed=%s recovered_terminal_artifacts_after_teardown_failure=1 %s\n' \
                    "$SEED" "$(date -u +%FT%TZ)" >"$STATUS"
                exit 0
                ;;
        esac
    fi
    sleep 2
done
