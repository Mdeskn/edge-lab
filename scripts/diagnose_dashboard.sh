#!/usr/bin/env bash
# Find where the dashboard data chain is broken. Run this ON THE PI.
#
#   bash scripts/diagnose_dashboard.sh
#
# The chain has four links, and each one fails with the same symptom on screen
# ("Waiting for metrics", or charts that look overwritten instead of
# accumulating), so guessing from the browser is unreliable:
#
#   client container -> dashboard backend -> WebSocket payload -> browser JS
#
# Frame charts are fed by the Pi client. GPU and network charts are fed from
# Kafka by VM2 and VM3. So frame charts empty while GPU and network still move
# means the client is the broken link, not the dashboard.
set -u

BACKEND="${BACKEND:-http://localhost:8080}"
FRONTEND="${FRONTEND:-http://localhost:5173}"
COMPOSE_DIR="${COMPOSE_DIR:-$HOME/edge-lab}"

pass() { printf '[ OK ] %s\n' "$1"; }
fail() { printf '[FAIL] %s\n' "$1"; PROBLEMS+=("$1"); }
info() { printf '       %s\n' "$1"; }
PROBLEMS=()

echo "=== 1. Containers ==="
if ! command -v docker >/dev/null 2>&1; then
  fail "docker is not on PATH"
else
  for svc in client dashboard-backend dashboard-frontend; do
    line="$(docker ps -a --filter "name=$svc" --format '{{.Names}}\t{{.Status}}' 2>/dev/null | head -1)"
    if [ -z "$line" ]; then
      fail "no container found for $svc"
    elif echo "$line" | grep -q "Up "; then
      pass "$line"
    else
      fail "$line"
    fi
  done
fi

echo
echo "=== 2. Client startup errors ==="
CLIENT="$(docker ps -a --filter 'name=client' --format '{{.Names}}' 2>/dev/null | head -1)"
if [ -n "${CLIENT:-}" ]; then
  ERRORS="$(docker logs --tail 200 "$CLIENT" 2>&1 \
    | grep -E 'Traceback|ModuleNotFoundError|ImportError|ValueError|CRITICAL|Error:' | tail -5)"
  if [ -n "$ERRORS" ]; then
    fail "client logged startup errors"
    printf '       %s\n' "$ERRORS"
    case "$ERRORS" in
      *"No module named 'common'"*)
        info ""
        info "The image is missing the shared 'common' package. It is built from"
        info "the repository root, not from client/. Check docker-compose.yml has:"
        info "    build:"
        info "      context: ."
        info "      dockerfile: client/Dockerfile"
        info "then rebuild:  docker compose build --no-cache client"
        ;;
    esac
  else
    pass "no startup errors in the last 200 log lines"
  fi

  if docker logs --tail 200 "$CLIENT" 2>&1 | grep -q "FrameReader started"; then
    pass "FrameReader started"
  else
    fail "FrameReader never started: the client is not producing frames"
  fi

  if docker logs --tail 200 "$CLIENT" 2>&1 | grep -q "Dashboard publishing enabled"; then
    pass "client is publishing to the dashboard"
  else
    fail "client is not publishing to the dashboard (check DASHBOARD_ENABLED and DASHBOARD_URL in .env)"
  fi
fi

echo
echo "=== 3. Backend state ==="
STATE="$(curl -fsS -m 5 "$BACKEND/api/state" 2>/dev/null)"
if [ -z "$STATE" ]; then
  fail "dashboard backend did not answer at $BACKEND/api/state"
else
  pass "backend answered"
  echo "$STATE" | python3 -c '
import json, sys
s = json.load(sys.stdin)
h = s.get("history", {})
frames, gpu, net = len(h.get("frames", [])), len(h.get("gpu", [])), len(h.get("network", []))
print(f"       frames in history : {frames}")
print(f"       gpu in history    : {gpu}")
print(f"       network in history: {net}")
print(f"       history_mode      : {s.get(\"history_mode\")}")
print(f"       history_seq       : {s.get(\"history_seq\")}")
print(f"       kafka             : {s.get(\"infrastructure\",{}).get(\"kafka\",{}).get(\"detail\")}")
if s.get("history_mode") is None:
    print("[FAIL] backend predates the delta protocol; rebuild dashboard-backend")
elif frames == 0 and (gpu or net):
    print("[FAIL] GPU/network arriving but no frames: the CLIENT is the broken link")
elif frames == 0:
    print("[FAIL] no data at all reaching the backend")
else:
    print("[ OK ] frames are reaching the backend")
'
fi

echo
echo "=== 4. Is the client still publishing right now? ==="
A="$(curl -fsS -m 5 "$BACKEND/api/state" 2>/dev/null | python3 -c 'import json,sys; print(json.load(sys.stdin)["summary"]["total_frames"])' 2>/dev/null)"
sleep 4
B="$(curl -fsS -m 5 "$BACKEND/api/state" 2>/dev/null | python3 -c 'import json,sys; print(json.load(sys.stdin)["summary"]["total_frames"])' 2>/dev/null)"
if [ -n "${A:-}" ] && [ -n "${B:-}" ]; then
  if [ "$B" -gt "$A" ]; then
    pass "frame count rising ($A -> $B over 4s)"
  else
    fail "frame count static at $A: nothing is arriving from the client"
  fi
else
  fail "could not read the frame counter"
fi

echo
echo "=== 5. Which JavaScript is being served ==="
JS="$(curl -fsS -m 5 "$FRONTEND/app.js" 2>/dev/null)"
if [ -z "$JS" ]; then
  fail "could not fetch $FRONTEND/app.js"
else
  if echo "$JS" | grep -q "historySeeded"; then
    pass "frontend has the current history merge"
  else
    fail "frontend is older than the backend: it cannot merge deltas"
    info "charts will show ~5 points that look overwritten instead of accumulating"
    info "fix:  docker compose up -d --build dashboard-frontend   then hard-reload the browser"
  fi
fi

echo
if [ "${#PROBLEMS[@]}" -eq 0 ]; then
  echo "All checks passed. If the charts still look wrong, hard-reload the browser"
  echo "(Cmd-Shift-R / Ctrl-Shift-R) so it drops any cached app.js."
  exit 0
fi
echo "${#PROBLEMS[@]} problem(s):"
for p in "${PROBLEMS[@]}"; do printf ' - %s\n' "$p"; done
exit 1
