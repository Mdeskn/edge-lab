# VM 3: Network / Router VM

**IP:** `172.22.174.148`

This machine has two roles:

1. **Router**: forwards client traffic from `172.22.174.148:8001` to the GPU server at `172.22.174.145:8001`. All Pi clients connect via this address so that network impairments affect the inference path.
2. **Network controller**: applies `tc` traffic shaping rules and publishes live network conditions and experiment phase events to Kafka.

---

## What runs here

| Component | Script | Kafka topic |
|-----------|--------|-------------|
| Network conditions publisher | `network_conditions_publisher.py` | `edgelab.network.metrics` |
| Phase controller | `tc_controller.py` | `edgelab.phase` |
| tc_control.sh | `/home/mae/network_load/tc_control.sh` | (called by tc_controller) |
| Port forward: 8001 to GPU | `iptables` / kernel routing | (not Kafka) |

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

The tc_controller's `PHASE_MAP` defines which `tc_control.sh` command runs for each phase name. The SeQaM scenario (`seqam/scenario.json`) writes the following phase names to the phase file:

- `baseline` (in PHASE_MAP: clears tc rules)
- `gpu_load` (in PHASE_MAP: clears tc rules; GPU load applied separately)
- `network_load` (SeQaM also calls `tc_apply.sh` directly for this phase)
- `combined` (SeQaM calls both the stressor and `tc_apply.sh`)

If `network_load` or `combined` are not in the tc_controller's PHASE_MAP, tc_controller will log a warning and not publish those phases to Kafka. Students would then not see those phase values in `self.experiment_phase`. Add them to `PHASE_MAP` in `tc_controller.py` to fix this.

---

## Network interface

The interface that connects to the Pi clients is `ens18`. Verify:

```bash
ssh mae@172.22.174.148 "ip link show ens18"
```

All `tc` commands operate on this interface.
