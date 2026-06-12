# VM 3: Network / Router VM

**IP:** `172.22.174.148`

This machine has two roles:

1. **Router**: forwards client traffic from `172.22.174.148:8100` to the GPU server remote inference API at `172.22.174.145:8100`. All Pi clients connect via this address so that network impairments affect the inference path. Port `8001` may still be forwarded for legacy direct Triton debugging.
2. **Network controller**: applies `tc` traffic shaping rules and publishes live network conditions and experiment phase events to Kafka.

---

## What runs here

| Component | Script | Kafka topic |
|-----------|--------|-------------|
| Network conditions publisher | `network_conditions_publisher.py` | `edgelab.network.metrics` |
| Phase controller | `tc_controller.py` | `edgelab.phase` |
| tc_control.sh | `/home/mae/network_load/tc_control.sh` | (called by tc_controller) |
| Port forward: 8100 to GPU API | `iptables` / kernel routing | (not Kafka) |
| Optional port forward: 8001 to Triton | `iptables` / kernel routing | Legacy/debug only |

---

## Script directory

Both Python scripts are deployed to `/home/mae/network_load/` on this VM. Copy them from the repo:

```bash
scp publishers/network_conditions/network_conditions_publisher.py \
    publishers/network_conditions/tc_controller.py \
    publishers/network_conditions/requirements.txt \
    mae@172.22.174.148:/home/mae/network_load/
```

---

## Full reference

`publishers/network_conditions/README.md` contains the complete documentation for these scripts, including:

- One-time setup and virtualenv creation
- Passwordless sudo configuration for `tc_control.sh`
- Run commands (with and without dry-run)
- All CLI arguments and environment variables
- Phase trigger commands
- The full phase map with tc_control.sh commands
- Published message formats for both Kafka topics
- Troubleshooting

---

## Quick reference

### Start the network conditions publisher

Publishes live `tc qdisc` state to `edgelab.network.metrics` every second:

```bash
ssh mae@172.22.174.148
source /home/mae/network_load/.venv/bin/activate
python3 -u /home/mae/network_load/network_conditions_publisher.py \
    --kafka-bootstrap-servers 172.22.174.149:9092 \
    --topic edgelab.network.metrics \
    --interface ens18
```

### Start the phase controller

Watches `/tmp/edgelab_phase`, runs `tc_control.sh`, publishes to `edgelab.phase`:

```bash
# Dry-run first (safe, no sudo called)
python3 -u /home/mae/network_load/tc_controller.py \
    --dry-run \
    --phase-file /tmp/edgelab_phase \
    --kafka-bootstrap-servers 172.22.174.149:9092

# Real mode
python3 -u /home/mae/network_load/tc_controller.py \
    --phase-file /tmp/edgelab_phase \
    --kafka-bootstrap-servers 172.22.174.149:9092
```

### Trigger a phase manually

```bash
echo baseline      > /tmp/edgelab_phase   # clear tc rules
echo gpu_load      > /tmp/edgelab_phase   # clear tc rules, GPU load external
echo bandwidth_50  > /tmp/edgelab_phase   # 50 mbit cap
echo jitter_light  > /tmp/edgelab_phase   # light netem delay + jitter
```

SeQaM writes phase names directly to this file over SSH.

### Check and clear tc state

```bash
sudo /home/mae/network_load/tc_control.sh show
sudo /home/mae/network_load/tc_control.sh clear
```

---

## Phase controller vs SeQaM scenario phases

The tc_controller's `PHASE_MAP` defines which `tc_control.sh` command runs for each phase name. `seqam/scenario.json` writes only phase names that are in PHASE_MAP:

- `baseline`: clears tc rules
- `gpu_load`: clears tc rules; GPU stressor applied separately via SeQaM SSH
- `bandwidth_50`: applies 50 Mbit/s tbf cap
- `mixed`: keeps 50 Mbit/s cap; GPU stressor also active

If you add a new phase to the scenario, add a matching entry to `PHASE_MAP` in `tc_controller.py` or tc_controller will log a warning and skip publishing that phase to Kafka.

---

## Network interface

The interface that connects to the Pi clients is `ens18`. Verify:

```bash
ssh mae@172.22.174.148 "ip link show ens18"
```

All `tc` commands operate on this interface.
