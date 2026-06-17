#!/usr/bin/env bash
# EdgeLab GPU load wrapper for LC1, using perf_analyzer (Eldiyar's tool).
#
# SeQaM-callable with colon-free arguments. The Triton URL contains a colon and
# SeQaM's command parser breaks on colons, so the URL stays inside this script
# and never reaches the SeQaM command line.
#
# The start action holds a fixed concurrency continuously. The pattern action
# cycles through concurrency levels so GPU queue metrics vary inside a phase
# instead of being a trivial phase-name lookup.
#
# Usage:
#   run_gpu_load.sh start [concurrency] [duration]   start background load
#   run_gpu_load.sh pattern [levels] [step] [duration]
#   run_gpu_load.sh stop                             stop the load
#   run_gpu_load.sh status                           show whether load is running
#
# duration 0 (default) = run until stopped; >0 = self-terminate after N seconds.
# levels default = 30,80. step default = 7 seconds.
#
# SeQaM event example (no colons anywhere):
#   ssh router load-vm bash /home/lc1/edgelab-load-client/run_gpu_load.sh pattern 30,80 7 0
set -u

IMAGE="nvcr.io/nvidia/tritonserver:26.01-py3-sdk"
TRITON_URL="172.22.174.145:8001"
MODEL="resnet50_full"
NAME="edgelab_gpu_load"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LOGFILE="$SCRIPT_DIR/gpu_load.log"
PATTERN_PID_FILE="$SCRIPT_DIR/gpu_load_pattern.pid"

action="${1:-}"

running() {
  docker ps --format '{{.Names}}' | grep -qx "$NAME"
}

pattern_pid() {
  if [ -f "$PATTERN_PID_FILE" ]; then
    cat "$PATTERN_PID_FILE" 2>/dev/null || true
  fi
}

pattern_running() {
  pid="$(pattern_pid)"
  [ -n "${pid:-}" ] && kill -0 "$pid" >/dev/null 2>&1
}

stop_pattern() {
  pid="$(pattern_pid)"
  if [ -n "${pid:-}" ] && kill -0 "$pid" >/dev/null 2>&1; then
    kill "$pid" >/dev/null 2>&1 || true
  fi
  rm -f "$PATTERN_PID_FILE"
}

start_load_container() {
  load_concurrency="$1"
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  docker run --rm --name "$NAME" --net=host "$IMAGE" \
    perf_analyzer -m "$MODEL" -i grpc -u "$TRITON_URL" \
    --input-data random \
    --concurrency-range "$load_concurrency" \
    --measurement-interval 999999 \
    >> "$LOGFILE" 2>&1 &
}

start_fixed_load() {
  load_concurrency="${1:-32}"
  load_duration="${2:-0}"

  stop_pattern
  if running; then
    echo "gpu load already running"
    exit 0
  fi
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  nohup docker run --rm --name "$NAME" --net=host "$IMAGE" \
    perf_analyzer -m "$MODEL" -i grpc -u "$TRITON_URL" \
    --input-data random \
    --concurrency-range "$load_concurrency" \
    --measurement-interval 999999 \
    >> "$LOGFILE" 2>&1 &
  if [ "${load_duration:-0}" -gt 0 ] 2>/dev/null; then
    ( sleep "${load_duration}"; docker rm -f "$NAME" >/dev/null 2>&1 || true ) &
  fi
  echo "started gpu load (perf_analyzer) concurrency $load_concurrency duration $load_duration"
}

run_pattern_worker() {
  levels_csv="${1:-30,80}"
  step_seconds="${2:-7}"
  total_duration="${3:-0}"

  if ! [ "$step_seconds" -gt 0 ] 2>/dev/null; then
    echo "pattern step must be a positive number of seconds"
    exit 1
  fi
  if ! [ "$total_duration" -ge 0 ] 2>/dev/null; then
    echo "pattern duration must be 0 or a positive number of seconds"
    exit 1
  fi

  IFS=',' read -r -a levels <<< "$levels_csv"
  if [ "${#levels[@]}" -eq 0 ]; then
    echo "pattern levels must not be empty"
    exit 1
  fi

  cleanup() {
    docker rm -f "$NAME" >/dev/null 2>&1 || true
    rm -f "$PATTERN_PID_FILE"
  }
  trap cleanup EXIT INT TERM

  echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') starting gpu load pattern levels=$levels_csv step=${step_seconds}s duration=${total_duration}s"
  started_at="$(date +%s)"
  index=0

  while true; do
    now="$(date +%s)"
    elapsed=$((now - started_at))
    if [ "$total_duration" -gt 0 ] && [ "$elapsed" -ge "$total_duration" ]; then
      break
    fi

    load_concurrency="${levels[$index]}"
    load_concurrency="${load_concurrency//[[:space:]]/}"
    if ! [ "$load_concurrency" -ge 0 ] 2>/dev/null; then
      echo "skipping invalid concurrency level: ${levels[$index]}"
    elif [ "$load_concurrency" -eq 0 ]; then
      docker rm -f "$NAME" >/dev/null 2>&1 || true
      echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') gpu load idle"
    else
      echo "$(date -u '+%Y-%m-%dT%H:%M:%SZ') gpu load concurrency $load_concurrency"
      start_load_container "$load_concurrency"
    fi

    sleep_for="$step_seconds"
    if [ "$total_duration" -gt 0 ]; then
      now="$(date +%s)"
      elapsed=$((now - started_at))
      remaining=$((total_duration - elapsed))
      if [ "$remaining" -le 0 ]; then
        break
      fi
      if [ "$remaining" -lt "$sleep_for" ]; then
        sleep_for="$remaining"
      fi
    fi
    sleep "$sleep_for"
    index=$(((index + 1) % ${#levels[@]}))
  done
}

case "$action" in
  start)
    start_fixed_load "${2:-32}" "${3:-0}"
    ;;
  pattern)
    stop_pattern
    docker rm -f "$NAME" >/dev/null 2>&1 || true
    nohup bash "$SCRIPT_DIR/run_gpu_load.sh" pattern_worker "${2:-30,80}" "${3:-7}" "${4:-0}" \
      >> "$LOGFILE" 2>&1 &
    echo "$!" > "$PATTERN_PID_FILE"
    echo "started gpu load pattern levels ${2:-30,80} step ${3:-7} duration ${4:-0}"
    ;;
  pattern_worker)
    run_pattern_worker "${2:-30,80}" "${3:-7}" "${4:-0}"
    ;;
  stop)
    stop_pattern
    if running; then
      docker rm -f "$NAME" >/dev/null 2>&1
      echo "stopped gpu load"
    else
      echo "gpu load not running"
    fi
    ;;
  status)
    if pattern_running; then
      echo "gpu load pattern running (pid $(pattern_pid))"
    elif running; then
      echo "gpu load running"
    else
      echo "gpu load not running"
    fi
    ;;
  *)
    echo "usage: $0 start [concurrency] [duration] | pattern [levels] [step_seconds] [duration] | stop | status"
    exit 1
    ;;
esac
