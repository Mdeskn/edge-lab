#!/usr/bin/env bash
# EdgeLab GPU load wrapper for LC1, using perf_analyzer (Eldiyar's tool).
#
# SeQaM-callable with colon-free arguments. The Triton URL contains a colon and
# SeQaM's command parser breaks on colons, so the URL stays inside this script
# and never reaches the SeQaM command line.
#
# Unlike a benchmark sweep, this holds a FIXED concurrency continuously so it
# acts as a sustained stressor for a phase. A very large measurement window
# keeps perf_analyzer loading until it is stopped (or the timeout fires).
#
# Usage:
#   run_gpu_load.sh start [concurrency] [duration]   start background load
#   run_gpu_load.sh stop                             stop the load
#   run_gpu_load.sh status                           show whether load is running
#
# duration 0 (default) = run until stopped; >0 = self-terminate after N seconds.
#
# SeQaM event example (no colons anywhere):
#   ssh router load-vm bash /home/lc1/edgelab-load-client/run_gpu_load.sh start 8 0
set -u

IMAGE="nvcr.io/nvidia/tritonserver:26.01-py3-sdk"
TRITON_URL="172.22.174.145:8001"
MODEL="resnet50_full"
NAME="edgelab_gpu_load"
LOGFILE="$(cd "$(dirname "$0")" && pwd)/gpu_load.log"

action="${1:-}"
concurrency="${2:-8}"
duration="${3:-0}"

running() {
  docker ps --format '{{.Names}}' | grep -qx "$NAME"
}

case "$action" in
  start)
    if running; then
      echo "gpu load already running"
      exit 0
    fi
    docker rm -f "$NAME" >/dev/null 2>&1 || true
    if [ "${duration:-0}" -gt 0 ] 2>/dev/null; then
      runner=(timeout "${duration}s" docker run --rm --name "$NAME" --net=host "$IMAGE")
    else
      runner=(docker run --rm --name "$NAME" --net=host "$IMAGE")
    fi
    nohup "${runner[@]}" \
      perf_analyzer -m "$MODEL" -i grpc -u "$TRITON_URL" \
      --input-data random \
      --concurrency-range "$concurrency" \
      --measurement-interval 999999 \
      >> "$LOGFILE" 2>&1 &
    echo "started gpu load (perf_analyzer) concurrency $concurrency duration $duration"
    ;;
  stop)
    if running; then
      docker rm -f "$NAME" >/dev/null 2>&1
      echo "stopped gpu load"
    else
      echo "gpu load not running"
    fi
    ;;
  status)
    if running; then
      echo "gpu load running"
    else
      echo "gpu load not running"
    fi
    ;;
  *)
    echo "usage: $0 start [concurrency] [duration] | stop | status"
    exit 1
    ;;
esac
