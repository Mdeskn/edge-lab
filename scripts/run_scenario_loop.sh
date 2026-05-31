#!/bin/bash
# Continuously loop the SeQaM experiment scenario.
#
# SeQaM runs the scenario once then exits (the "exit" command at t=120000
# terminates the SeQaM process). This wrapper restarts it immediately so
# the lab runs continuously without manual intervention.
#
# Usage:
#   ./scripts/run_scenario_loop.sh [path/to/seqam] [path/to/scenario.json]
#
# Defaults:
#   SEQAM_BIN   - seqam (must be on PATH, or set this env var)
#   SCENARIO    - seqam/scenario.json relative to the repo root

set -e

SEQAM_BIN=${SEQAM_BIN:-seqam}
SCENARIO=${1:-seqam/scenario.json}
LOOP=0

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) Starting continuous scenario loop: $SCENARIO"

while true; do
    LOOP=$((LOOP + 1))
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) === Iteration $LOOP ==="
    "$SEQAM_BIN" run --scenario "$SCENARIO" || {
        echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) seqam exited with error $?; restarting in 2s"
        sleep 2
    }
done
