# Operator / Tutor Guide

This guide covers what the tutor/operator needs to do before, during, and after each experiment session. For detailed VM-specific setup, see the per-VM docs linked at the end.

---

## Lab infrastructure at a glance

| Machine | IP | Role |
|---------|----|------|
| SeQaM / Kafka VM | `172.22.174.149` | Kafka broker, Kafka UI, Grafana, SeQaM platform |
| GPU server | `172.22.174.145` | Triton Inference Server, GPU metrics publisher (Eldiyar) |
| Network / Router VM | `172.22.174.148` | Routes client traffic, applies tc rules, publishes network metrics and phase events |
| Raspberry Pi (per group) | group-assigned IP | Pi client app, group dashboard |

Clients connect to Triton via the Router VM (`172.22.174.148:8001`), not directly to the GPU server. This is what allows network impairments to affect the inference path.

---

## Pre-experiment checklist

### 1. Verify Kafka is reachable

```bash
nc -zv 172.22.174.149 9092
```

Open Kafka UI to confirm topics exist: `http://172.22.174.149:8080`

Expected topics: `dnn_partition.server_metrics`, `dnn_partition.client_metrics`, `edgelab.network.metrics`, `edgelab.phase`

### 2. Verify Triton is running (GPU server)

```bash
curl http://172.22.174.145:8001/v2/health/live
```

Also check via the Router VM (the path clients actually use):

```bash
curl http://172.22.174.148:8001/v2/health/live
```

Both should return HTTP 200.

### 3. Verify GPU metrics are flowing to Kafka

Watch the topic in Kafka UI: `dnn_partition.server_metrics`

Or from terminal:

```bash
kafka-console-consumer.sh \
    --bootstrap-server 172.22.174.149:9092 \
    --topic dnn_partition.server_metrics \
    --max-messages 3
```

### 4. Start Network VM scripts

SSH to the Network VM and start both scripts if they are not already running:

```bash
ssh mae@172.22.174.148
cd /home/mae/network_load
source .venv/bin/activate

# Terminal 1: network metrics publisher (reads tc qdisc, publishes every 1s)
python3 -u network_conditions_publisher.py

# Terminal 2: phase controller (watches /tmp/edgelab_phase, runs tc_control.sh)
python3 -u tc_controller.py
```

See `publishers/network_conditions/README.md` for full startup instructions, dry-run mode, and troubleshooting.

### 5. Clear any leftover tc rules

```bash
ssh mae@172.22.174.148 "sudo /home/mae/network_load/tc_control.sh clear"
```

Verify with:

```bash
ssh mae@172.22.174.148 "sudo /home/mae/network_load/tc_control.sh show"
```

### 6. Initialize the phase file

```bash
ssh mae@172.22.174.148 "echo baseline > /tmp/edgelab_phase"
```

### 7. Verify network metrics are flowing to Kafka

```bash
kafka-console-consumer.sh \
    --bootstrap-server 172.22.174.149:9092 \
    --topic edgelab.network.metrics \
    --max-messages 3
```

The `mode` field should be `"clear"` and `tc_active` should be `false`.

---

## Running the SeQaM experiment

1. Make sure all VMs are in their initial state (steps 1-7 above).
2. Start all student Pi clients (`AUTO_STOP=true`). They will print "Waiting for experiment phase to start..." and block.
3. Start the SeQaM scenario:

```bash
SEQAM_API_URL=http://172.22.174.149:8000 \
./scripts/run_scenario_loop.sh
```

This posts `seqam/scenario.json` to SeQaM after each 120-second run. SeQaM triggers the stressors and writes phase names to `/tmp/edgelab_phase` on the Network VM. The tc_controller publishes phase changes to Kafka. All Pi clients receive the phase events and start processing.

4. Monitor via Grafana (`http://172.22.174.149:3000`) or Kafka UI (`http://172.22.174.149:8080`).
5. After the run, Pi clients auto-stop and print their score. Collect scores.

---

## Manual phase testing (without SeQaM)

Write a phase name to the phase file on the Network VM:

```bash
ssh mae@172.22.174.148 "echo gpu_load > /tmp/edgelab_phase"
```

The tc_controller detects the change within 1 second, applies the corresponding tc rule, and publishes to Kafka. To apply network impairment manually:

```bash
ssh mae@172.22.174.148 "sudo /home/mae/network_load/tc_control.sh tbf 50mbit 2mbit 50ms"
```

To clear:

```bash
ssh mae@172.22.174.148 "echo baseline > /tmp/edgelab_phase"
```

---

## Resetting between groups

Each student group runs the experiment independently. Between runs:

1. Clear tc rules on the Network VM:

```bash
ssh mae@172.22.174.148 "sudo /home/mae/network_load/tc_control.sh clear && echo baseline > /tmp/edgelab_phase"
```

2. Reset the dashboard on each Pi (clears cumulative counters, keeps video feed):

```bash
curl -X POST http://<pi-ip>:8080/api/reset
```

3. Confirm no stressors are running on the GPU server.

---

## Dashboard setup for students

Each group runs its own dashboard on their Pi. In `.env` on the Pi:

```dotenv
DASHBOARD_ENABLED=true
DASHBOARD_URL=http://localhost:8080
```

Start the dashboard backend:

```bash
cd dashboard/backend
source .venv/bin/activate   # or the shared venv
uvicorn main:app --host 0.0.0.0 --port 8080
```

Or via Docker:

```bash
docker compose -f docker-compose.pi.yml up -d
```

Students open the dashboard at `http://<pi-ip>:5173`.

---

## Troubleshooting

### tc_controller logs "Unknown phase"

The tc_controller's PHASE_MAP does not include that phase name. Check `/tmp/edgelab_phase` and compare with the PHASE_MAP in `publishers/network_conditions/tc_controller.py`.

### Phase events not appearing in Kafka

Confirm tc_controller is running on the Network VM. Write a known phase manually:

```bash
ssh mae@172.22.174.148 "echo baseline > /tmp/edgelab_phase"
```

Watch the `edgelab.phase` topic in Kafka UI.

### Pi clients see only "baseline" phase

tc_controller only publishes phases that appear in its PHASE_MAP. SeQaM's scenario writes "baseline", "gpu_load", "network_load", "combined". Make sure all four are in PHASE_MAP (or use the manual echo approach above to test).

### GPU server overloaded before experiment starts

Check that no leftover stressor processes are running on the GPU server. Kill them:

```bash
ssh mae@172.22.174.145 "pkill -f gpu_stressor"
```

### Student Pi cannot reach Triton

The Pi must use `172.22.174.148:8001`, not `172.22.174.145:8001`. Verify the student's `.env` has `TRITON_URL=172.22.174.148:8001`.

---

## See also

| Document | Contents |
|----------|---------|
| `docs/vm1-kafka-grafana.md` | Kafka/Grafana VM details |
| `docs/vm2-gpu-server.md` | GPU server and Triton |
| `docs/vm3-network-vm.md` | Network VM and tc scripts |
| `docs/seqam-integration.md` | SeQaM scenario configuration |
| `publishers/network_conditions/README.md` | Network VM script reference |
