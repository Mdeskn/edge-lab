#!/bin/bash
# Apply netem rules to a network interface.
# Usage: ./tc_apply.sh [interface] [delay_ms] [jitter_ms] [loss_pct]
# Example: ./tc_apply.sh eth0 100 20 3

set -e
INTERFACE=${1:-eth0}
DELAY_MS=${2:-50}
JITTER_MS=${3:-10}
LOSS_PCT=${4:-0}

tc qdisc del dev "$INTERFACE" root 2>/dev/null || true
tc qdisc add dev "$INTERFACE" root netem \
    delay "${DELAY_MS}ms" "${JITTER_MS}ms" \
    loss "${LOSS_PCT}%"

echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) tc_apply: interface=$INTERFACE delay=${DELAY_MS}ms jitter=${JITTER_MS}ms loss=${LOSS_PCT}%"

# Signal publisher so it publishes immediately.
# The publisher runs inside the edge-lab-net-publisher Docker container, so
# pkill on the host process namespace would never reach it. Use docker exec instead.
docker exec edge-lab-net-publisher kill -USR1 1 2>/dev/null || true
