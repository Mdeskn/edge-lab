#!/usr/bin/env bash
# Lab-day readiness check. Exits 0 only when all required services look healthy.
set -u

VM1_SSH="${VM1_SSH:-mae@172.22.174.149}"
VM2_SSH="${VM2_SSH:-mae@172.22.174.145}"
VM3_SSH="${VM3_SSH:-mae@172.22.174.148}"
LC1_SSH="${LC1_SSH:-lc1@172.22.229.169}"
PI_SSH="${PI_SSH:-mae@172.22.229.167}"

VM2_HOST="${VM2_HOST:-172.22.174.145}"
ROUTER_HOST="${ROUTER_HOST:-172.22.174.148}"
REMOTE_INFERENCE_URL="${REMOTE_INFERENCE_URL:-http://172.22.174.148:8100}"
KAFKA_CONTAINER="${KAFKA_CONTAINER:-dnn-partition-kafka}"
KAFKA_TOPICS="${KAFKA_TOPICS:-dnn_partition.server_metrics edgelab.network.metrics edgelab.phase}"
KAFKA_MESSAGE_TIMEOUT_MS="${KAFKA_MESSAGE_TIMEOUT_MS:-12000}"
SSH_TIMEOUT="${SSH_TIMEOUT:-5}"

SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout="$SSH_TIMEOUT")
failures=()

ok() {
  printf '[ OK ] %s\n' "$1"
}

fail() {
  printf '[FAIL] %s\n' "$1"
  failures+=("$1")
}

run_remote() {
  target="$1"
  command="$2"
  ssh "${SSH_OPTS[@]}" "$target" "$command"
}

check_ssh() {
  label="$1"
  target="$2"
  if run_remote "$target" "true" >/dev/null 2>&1; then
    ok "$label reachable over SSH ($target)"
  else
    fail "$label is not reachable over SSH ($target)"
  fi
}

check_tcp() {
  label="$1"
  host="$2"
  port="$3"
  if nc -z -w 3 "$host" "$port" >/dev/null 2>&1; then
    ok "$label reachable at $host:$port"
  else
    fail "$label is not reachable at $host:$port"
  fi
}

check_http() {
  label="$1"
  url="$2"
  if curl -fsS --max-time 5 "$url" >/dev/null 2>&1; then
    ok "$label responded at $url"
  else
    fail "$label did not respond at $url"
  fi
}

check_kafka_topics() {
  output="$(
    run_remote "$VM1_SSH" \
      "docker exec $KAFKA_CONTAINER kafka-topics.sh --bootstrap-server localhost:9092 --list" \
      2>/dev/null
  )"
  status=$?
  if [ "$status" -ne 0 ]; then
    fail "Kafka topic list failed on $VM1_SSH container $KAFKA_CONTAINER"
    return
  fi

  for topic in $KAFKA_TOPICS; do
    if printf '%s\n' "$output" | grep -qx "$topic"; then
      ok "Kafka topic exists: $topic"
    else
      fail "Kafka topic is missing: $topic"
    fi
  done
}

check_kafka_recent_message() {
  topic="$1"
  command="docker exec $KAFKA_CONTAINER kafka-console-consumer.sh --bootstrap-server localhost:9092 --topic $topic --max-messages 1 --timeout-ms $KAFKA_MESSAGE_TIMEOUT_MS"
  output="$(run_remote "$VM1_SSH" "$command" 2>/dev/null)"
  status=$?
  if [ "$status" -eq 0 ] && [ -n "$output" ]; then
    ok "Kafka topic has a recent message: $topic"
  else
    fail "Kafka topic has no recent message within ${KAFKA_MESSAGE_TIMEOUT_MS}ms: $topic"
  fi
}

check_pi_router_resolution() {
  command="getent hosts $ROUTER_HOST >/dev/null 2>&1 || ping -c 1 -W 2 $ROUTER_HOST >/dev/null 2>&1 || nc -z -w 3 $ROUTER_HOST 8100 >/dev/null 2>&1"
  if run_remote "$PI_SSH" "$command" >/dev/null 2>&1; then
    ok "Pi can resolve/reach router VM ($ROUTER_HOST)"
  else
    fail "Pi cannot resolve/reach router VM ($ROUTER_HOST); set PI_SSH if the default is wrong"
  fi
}

check_gpu_metrics_publisher() {
  if run_remote "$VM2_SSH" "pgrep -af kafka_metrics_publisher.py" >/dev/null 2>&1; then
    ok "GPU metrics publisher is running on VM2"
  else
    fail "GPU metrics publisher is not running on VM2"
  fi
}

check_gpu_idle() {
  message="$(
    run_remote "$VM1_SSH" \
      "docker exec $KAFKA_CONTAINER kafka-console-consumer.sh --bootstrap-server localhost:9092 --topic dnn_partition.server_metrics --max-messages 1 --timeout-ms $KAFKA_MESSAGE_TIMEOUT_MS" \
      2>/dev/null | grep -m1 '^{'
  )"

  if [ -z "$message" ]; then
    fail "GPU idle check: no recent message on dnn_partition.server_metrics"
    return
  fi

  detail="$(printf '%s' "$message" | python3 -c '
import json, sys
data = json.load(sys.stdin)
busy = [m for m in data.get("models", []) if m.get("pending_requests", 0)]
gpu_util = (data.get("server") or {}).get("gpu_util_percent")
util_str = "{:.0f}%".format(gpu_util) if isinstance(gpu_util, (int, float)) else "n/a"
if busy:
    parts = ", ".join("{}={}".format(m.get("model_name"), m.get("pending_requests")) for m in busy)
    print("FAIL leftover pending requests: {} (gpu_util={})".format(parts, util_str))
else:
    print("OK gpu_util={}".format(util_str))
' 2>&1)"

  case "$detail" in
    OK*) ok "GPU is idle (${detail#OK })" ;;
    FAIL*) fail "GPU is not idle, ${detail#FAIL }" ;;
    *) fail "GPU idle check: could not parse metrics message ($detail)" ;;
  esac
}

check_ssh "VM1 Kafka/Grafana" "$VM1_SSH"
check_ssh "VM2 GPU server" "$VM2_SSH"
check_ssh "VM3 router/network" "$VM3_SSH"
check_ssh "LC1 GPU load client" "$LC1_SSH"

check_kafka_topics
for topic in $KAFKA_TOPICS; do
  check_kafka_recent_message "$topic"
done

check_tcp "Triton gRPC" "$VM2_HOST" 8001
check_http "JPEG gateway" "$REMOTE_INFERENCE_URL/health"
check_pi_router_resolution
check_gpu_metrics_publisher
check_gpu_idle

if [ "${#failures[@]}" -eq 0 ]; then
  printf '\nPreflight passed: all checks green.\n'
  exit 0
fi

printf '\nPreflight failed with %d issue(s):\n' "${#failures[@]}"
for item in "${failures[@]}"; do
  printf ' - %s\n' "$item"
done
exit 1
