#!/bin/bash
INTERFACE=${1:-eth0}
tc qdisc del dev "$INTERFACE" root 2>/dev/null || true
echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) tc_clear: cleared rules on $INTERFACE"
docker exec edge-lab-net-publisher kill -USR1 1 2>/dev/null || true
