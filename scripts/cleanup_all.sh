#!/usr/bin/env bash
# Reset the lab after a run: Pi app down, GPU load stopped, VM3 network baseline.
set -u

PI_SSH="${PI_SSH:-mae@172.22.229.167}"
LC1_SSH="${LC1_SSH:-emulate@172.22.229.235}"
GPU_SSH="${GPU_SSH:-mae@172.22.174.145}"
VM3_SSH="${VM3_SSH:-mae@172.22.174.148}"

PI_EDGE_LAB_DIR="${PI_EDGE_LAB_DIR:-~/edge-lab}"
GPU_LOAD_DIR="${GPU_LOAD_DIR:-/home/emulate/edgelab-load-client}"
GPU_SERVER_DIR="${GPU_SERVER_DIR:-/home/mae/server}"
VM3_NETWORK_DIR="${VM3_NETWORK_DIR:-/home/mae/network_load}"
SSH_TIMEOUT="${SSH_TIMEOUT:-5}"
DRY_RUN="${DRY_RUN:-0}"

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
  label="$1"
  target="$2"
  command="$3"

  if [ "$DRY_RUN" = "1" ]; then
    printf '[DRY] %s: ssh %s %q\n' "$label" "$target" "$command"
    return 0
  fi

  if ssh "${SSH_OPTS[@]}" "$target" "$command"; then
    ok "$label"
  else
    fail "$label"
  fi
}

run_remote \
  "Stopped Pi app" \
  "$PI_SSH" \
  "if [ -d $PI_EDGE_LAB_DIR ]; then cd $PI_EDGE_LAB_DIR && docker compose -f docker-compose.pi.yml down; fi; pkill -f 'python[0-9.]* .*client/main.py' 2>/dev/null || true; pkill -f 'python[0-9.]* main.py' 2>/dev/null || true"

run_remote \
  "Stopped load VM GPU load (force)" \
  "$LC1_SSH" \
  "bash $GPU_LOAD_DIR/force_stop_gpu_load.sh"

run_remote \
  "Reset VM3 phase and traffic shaping" \
  "$VM3_SSH" \
  "$VM3_NETWORK_DIR/set_phase.sh baseline && sudo -n $VM3_NETWORK_DIR/tc_control.sh clear && sudo -n $VM3_NETWORK_DIR/tc_control.sh show"

run_remote \
  "GPU idle check" \
  "$GPU_SSH" \
  "bash $GPU_SERVER_DIR/check_gpu_idle.sh"

if [ "${#failures[@]}" -eq 0 ]; then
  printf '\nCleanup complete.\n'
  exit 0
fi

printf '\nCleanup finished with %d issue(s):\n' "${#failures[@]}"
for item in "${failures[@]}"; do
  printf ' - %s\n' "$item"
done
exit 1
