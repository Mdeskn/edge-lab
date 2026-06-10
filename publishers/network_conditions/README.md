# Network Conditions Publisher - Network VM Setup

Scripts for the **Network / Router VM** (`172.22.174.148`).

These two scripts run on the Network VM and together handle experiment phase
management and network state reporting:

| Script | Role |
|--------|------|
| `tc_controller.py` | Watches the phase file, calls `tc_control.sh`, publishes phase events to Kafka |
| `network_conditions_publisher.py` | Reads live `tc` state from the interface, publishes network metrics to Kafka |

`tc_control.sh` is already on the Network VM at `/home/mae/network_load/tc_control.sh`.
These Python scripts do not replace it; they drive it and observe its effects.

---

## Lab infrastructure

| Machine | IP | Role |
|---------|----|------|
| SeQaM / Kafka VM | `172.22.174.149` | Kafka broker, Kafka UI, Grafana |
| GPU Server | `172.22.174.145` | Triton (yolov10n, resnet50_full) |
| Network VM | `172.22.174.148` | Routes client traffic, applies tc rules |

Kafka UI: `http://172.22.174.149:8080`
Grafana: `http://172.22.174.149:3000`

Client traffic path during experiments:

```
Raspberry Pi -> 172.22.174.148:8100 -> 172.22.174.145:8100 (remote inference API)
                                                 -> Triton on the GPU server
```

---

## Kafka topics

| Topic | Publisher | Consumers |
|-------|-----------|-----------|
| `edgelab.phase` | `tc_controller.py` | SP agent on Pi, dashboard |
| `edgelab.network.metrics` | `network_conditions_publisher.py` | SP agent on Pi, dashboard |
| `dnn_partition.server_metrics` | Eldiyar's publisher on GPU server | SP agent on Pi, dashboard |
| `dnn_partition.client_metrics` | Client app on Raspberry Pi | Dashboard |

---

## One-time setup on the Network VM

SSH into the Network VM and run:

```bash
ssh mae@172.22.174.148
cd /home/mae/network_load

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Passwordless sudo for tc_control.sh (required)

`tc_controller.py` runs `sudo tc_control.sh` to apply traffic rules.
The user `mae` needs passwordless sudo for that script specifically:

```bash
echo "mae ALL=(ALL) NOPASSWD: /home/mae/network_load/tc_control.sh" \
    | sudo tee /etc/sudoers.d/edgelab-tc
sudo chmod 440 /etc/sudoers.d/edgelab-tc
```

Verify it works:

```bash
sudo /home/mae/network_load/tc_control.sh show
```

---

## Copy files to the Network VM

From your local machine (run once after each edit):

```bash
scp publishers/network_conditions/network_conditions_publisher.py \
    publishers/network_conditions/tc_controller.py \
    publishers/network_conditions/requirements.txt \
    mae@172.22.174.148:/home/mae/network_load/
```

---

## Running the scripts

Open two terminal sessions on the Network VM.

### Terminal 1: network_conditions_publisher.py

Reads `tc -s qdisc show dev ens18` every second and publishes the parsed
network state to Kafka topic `edgelab.network.metrics`.

```bash
source /home/mae/network_load/.venv/bin/activate

python3 -u /home/mae/network_load/network_conditions_publisher.py \
    --kafka-bootstrap-servers 172.22.174.149:9092 \
    --topic edgelab.network.metrics \
    --interface ens18 \
    --interval-s 1
```

### Terminal 2: tc_controller.py

Watches `/tmp/edgelab_phase`. When its contents change, runs the matching
`tc_control.sh` command and publishes a phase event to Kafka topic `edgelab.phase`.

**Dry-run first** (safe: logs what it would do, no sudo called):

```bash
source /home/mae/network_load/.venv/bin/activate

python3 -u /home/mae/network_load/tc_controller.py \
    --dry-run \
    --phase-file /tmp/edgelab_phase \
    --kafka-bootstrap-servers 172.22.174.149:9092 \
    --topic edgelab.phase \
    --tc-script /home/mae/network_load/tc_control.sh
```

**Real mode** (applies tc rules):

```bash
source /home/mae/network_load/.venv/bin/activate

python3 -u /home/mae/network_load/tc_controller.py \
    --phase-file /tmp/edgelab_phase \
    --kafka-bootstrap-servers 172.22.174.149:9092 \
    --topic edgelab.phase \
    --tc-script /home/mae/network_load/tc_control.sh
```

---

## Triggering phase changes

Write a phase name or number to the phase file. The controller detects the
change on its next poll (default: 1 second) and applies the corresponding
`tc_control.sh` command.

```bash
echo baseline      > /tmp/edgelab_phase   # 0: clears tc rules
echo bandwidth_200 > /tmp/edgelab_phase   # 1: limits to 200 mbit
echo bandwidth_50  > /tmp/edgelab_phase   # 2: limits to 50 mbit
echo jitter_light  > /tmp/edgelab_phase   # 3: 0.1ms delay, 0.4ms jitter, 1gbit
echo gpu_load      > /tmp/edgelab_phase   # 4: clears tc (GPU load handled externally)
echo mixed         > /tmp/edgelab_phase   # 5: limits to 50 mbit
```

Numeric aliases work too:

```bash
echo 0 > /tmp/edgelab_phase   # baseline
echo 2 > /tmp/edgelab_phase   # bandwidth_50
echo 3 > /tmp/edgelab_phase   # jitter_light
```

### What each phase does

| Phase | Index | tc_control.sh command | Notes |
|-------|-------|-----------------------|-------|
| `baseline` | 0 | `clear` | No shaping, full bandwidth |
| `bandwidth_200` | 1 | `tbf 200mbit 2mbit 50ms` | 200 mbit cap |
| `bandwidth_50` | 2 | `tbf 50mbit 2mbit 50ms` | 50 mbit cap |
| `jitter_light` | 3 | `netem_tbf 0.1ms 0.4ms 1gbit 2mbit 50ms` | Tiny delay + jitter |
| `gpu_load` | 4 | `clear` | Network clear; GPU load triggered externally |
| `mixed` | 5 | `tbf 50mbit 2mbit 50ms` | 50 mbit cap, GPU load may be active too |

**Important:** no duration argument is passed to `tc_control.sh`. Rules stay
active until the next phase change or a manual `clear`. SeQaM controls
experiment timing by writing to the phase file at the right moment.

---

## Checking and clearing tc state

```bash
# Show current qdisc rules and statistics:
sudo /home/mae/network_load/tc_control.sh show

# Clear all tc rules manually:
sudo /home/mae/network_load/tc_control.sh clear

# Read raw tc output directly:
tc -s qdisc show dev ens18
```

---

## Verifying Kafka messages

Open Kafka UI in a browser: `http://172.22.174.149:8080`

Topics to check: `edgelab.phase` and `edgelab.network.metrics`.

Or from any machine with Kafka tools installed:

```bash
# Watch phase events:
kafka-console-consumer.sh \
    --bootstrap-server 172.22.174.149:9092 \
    --topic edgelab.phase \
    --from-beginning

# Watch network metrics:
kafka-console-consumer.sh \
    --bootstrap-server 172.22.174.149:9092 \
    --topic edgelab.network.metrics \
    --from-beginning
```

---

## Published message formats

### edgelab.phase (from tc_controller.py)

```json
{
  "timestamp": 1780400000.0,
  "source": "network-vm",
  "phase": "bandwidth_50",
  "phase_index": 2,
  "description": "Limit bandwidth to 50mbit.",
  "tc_command": "sudo /home/mae/network_load/tc_control.sh tbf 50mbit 2mbit 50ms",
  "tc_parameters": {
    "mode": "tbf",
    "bandwidth": "50mbit",
    "burst": "2mbit",
    "tbf_latency": "50ms",
    "delay_ms": 0.0,
    "jitter_ms": 0.0,
    "packet_loss_percent": 0.0,
    "duration_seconds": null
  },
  "status": "applied"
}
```

`status` is one of `"applied"`, `"failed"`, or `"dry_run"`.
An `"error"` field is added when `status == "failed"`.

### edgelab.network.metrics (from network_conditions_publisher.py)

**No shaping (baseline / gpu_load):**

```json
{
  "timestamp": 1780400001.0,
  "source": "network-vm",
  "interface": "ens18",
  "tc_active": false,
  "mode": "clear",
  "bandwidth": "unlimited",
  "rate": "unlimited",
  "delay_ms": 0.0,
  "jitter_ms": 0.0,
  "packet_loss_pct": 0.0,
  "packet_loss_percent": 0.0,
  "tbf_latency": null,
  "raw": "qdisc fq_codel 0: root ..."
}
```

**TBF bandwidth shaping:**

```json
{
  "tc_active": true,
  "mode": "tbf",
  "bandwidth": "50Mbit",
  "rate": "50Mbit",
  "tbf_latency": "50ms",
  "delay_ms": 0.0,
  "jitter_ms": 0.0,
  ...
}
```

**netem + TBF (jitter_light):**

```json
{
  "tc_active": true,
  "mode": "netem_tbf",
  "bandwidth": "1Gbit",
  "rate": "1Gbit",
  "tbf_latency": "50ms",
  "delay_ms": 0.1,
  "jitter_ms": 0.4,
  ...
}
```

---

## CLI reference

### network_conditions_publisher.py

```
--kafka-bootstrap-servers  Kafka broker host:port  (default: 172.22.174.149:9092)
--topic                    Kafka topic             (default: edgelab.network.metrics)
--interface                Network interface       (default: ens18)
--interval-s               Publish interval (s)   (default: 1)
```

### tc_controller.py

```
--phase-file               Phase file path         (default: /tmp/edgelab_phase)
--kafka-bootstrap-servers  Kafka broker host:port  (default: 172.22.174.149:9092)
--topic                    Kafka topic             (default: edgelab.phase)
--tc-script                Path to tc_control.sh   (default: /home/mae/network_load/tc_control.sh)
--poll-interval-s          File poll interval (s)  (default: 1.0)
--command-timeout-s        tc_control.sh timeout   (default: 10)
--dry-run                  Skip sudo, log only
```

All CLI args can also be set via environment variables (see the top of each script for the full list).

---

## Troubleshooting

**tc_controller.py logs `status=failed` with a sudo error:**
The passwordless sudo entry is missing. See the setup section above.

**tc_controller.py logs "Waiting for phase file":**
The phase file does not exist yet. Write any phase to create it:
```bash
echo baseline > /tmp/edgelab_phase
```

**network_conditions_publisher.py logs "'tc' command not found":**
Install iproute2:
```bash
sudo apt-get install -y iproute2
```

**Kafka connection refused:**
Check that the SeQaM VM is reachable and Kafka is running:
```bash
nc -zv 172.22.174.149 9092
```
