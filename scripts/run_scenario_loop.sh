#!/bin/bash
# Continuously loop the SeQaM experiment scenario.
#
# SeQaM's API accepts ExperimentConfig JSON and dispatches it immediately when
# execute_immediately is true. The request stays open until the scenario ends,
# so posting it again starts the next cycle.
#
# Usage:
#   ./scripts/run_scenario_loop.sh [path/to/scenario.json]
#
# Defaults:
#   SEQAM_API_URL  - http://localhost:8000
#   SCENARIO       - seqam/scenario.json relative to the repo root

set -u

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SEQAM_API_URL=${SEQAM_API_URL:-http://localhost:8000}
SCENARIO=${1:-"$REPO_ROOT/seqam/scenario.json"}
ENDPOINT="${SEQAM_API_URL%/}/config/ExperimentConfig.json"
LOOP=0

if [ ! -r "$SCENARIO" ]; then
    echo "Scenario file is not readable: $SCENARIO" >&2
    exit 1
fi

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) Starting continuous scenario loop: $SCENARIO -> $ENDPOINT"

while true; do
    LOOP=$((LOOP + 1))
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) === Iteration $LOOP ==="

    if ! curl --fail --silent --show-error \
        --header "Content-Type: application/json" \
        --data-binary "@$SCENARIO" \
        "$ENDPOINT"
    then
        echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) SeQaM API request failed; retrying in 2s" >&2
        sleep 2
    else
        printf '\n'
    fi
done
