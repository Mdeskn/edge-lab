# SeQaM Integration

SeQaM is the experiment orchestration platform hosted on VM 1 (`172.22.174.149`). It executes load commands via SSH at scheduled times and drives the four-phase experiment loop.

---

## How the phase flow works

```
SeQaM (172.22.174.149)
   |
   | SSH at t=0s     echo baseline > /tmp/edgelab_phase
   | SSH at t=30s    echo gpu_load > /tmp/edgelab_phase
   |                 + launch gpu_stressor on GPU server
   | SSH at t=60s    echo bandwidth_50 > /tmp/edgelab_phase
   |                 (tc_controller applies 50 Mbit/s cap via PHASE_MAP)
   | SSH at t=90s    echo mixed > /tmp/edgelab_phase
   |                 + launch gpu_stressor
   v

Network VM (172.22.174.148) /tmp/edgelab_phase
   |
   v (tc_controller.py polls every 1s)
   |
   | Runs tc_control.sh (applies or clears tc rules)
   | Publishes phase event to Kafka
   v

Kafka 172.22.174.149:9092  topic: edgelab.phase
   |
   v (SPAgentBase Kafka consumer)
   |
self.experiment_phase updated on every Pi
```

---

## Experiment phases

| Time | Phase | GPU load | Network load |
|------|-------|----------|--------------|
| 0-30 s | `baseline` | No | No |
| 30-60 s | `gpu_load` | Yes (100 concurrent requests) | No |
| 60-90 s | `bandwidth_50` | No | Yes (50 Mbit/s tbf cap via tc_controller) |
| 90-120 s | `mixed` | Yes | Yes (50 Mbit/s cap + GPU flooded) |

At t=120 s the scenario ends; `run_scenario_loop.sh` immediately re-posts it, keeping tc_controller in sync.

The full loop is 120 seconds. `run_scenario_loop.sh` re-posts the scenario after each run to keep it going continuously.

---

## The scenario file

`seqam/scenario.json` defines the event list. Key fields:

- `execute_immediately: true`: SeQaM starts the scenario as soon as it receives the POST.
- Each event has a `command` (SSH command string) and `executionTime` (ms from scenario start).

The SSH target names (`src_device_name: net-vm`, `src_device_name: gpu-server`) must match entries in SeQaM's `ScenarioConfig.json`. See below.

---

## SeQaM SSH target configuration

Merge the entries from `seqam/ScenarioConfig.json` into SeQaM's installed config at:

```
~/.seqam_fh_dortmund_project_emulate/ScenarioConfig.json
```

Both entries appear under `router` because this SeQaM revision treats `router` entries as static SSH targets. The `server` category is used for dynamically registered components.

### SSH key setup

SeQaM connects to the Network VM and GPU server via SSH. Install SeQaM's `ecdsa.pub` key for user `mae` on both machines:

```bash
# On the Network VM
ssh mae@172.22.174.148
echo "<seqam-ecdsa-pub-key>" >> ~/.ssh/authorized_keys

# On the GPU server
ssh mae@172.22.174.145
echo "<seqam-ecdsa-pub-key>" >> ~/.ssh/authorized_keys
```

### Script placement

SeQaM calls scripts by absolute path (`/scripts/...`). Place them on both machines:

```
/scripts/gpu_stressor.sh    (on GPU server)
/scripts/tc_apply.sh        (on Network VM)
/scripts/tc_clear.sh        (on Network VM)
```

Copy from this repo:

```bash
scp scripts/gpu_stressor.sh mae@172.22.174.145:/scripts/
scp scripts/tc_apply.sh scripts/tc_clear.sh mae@172.22.174.148:/scripts/
```

---

## Running the scenario

### One-shot (single 120-second run)

```bash
curl -X POST http://172.22.174.149:8000/config/ExperimentConfig.json \
    -H "Content-Type: application/json" \
    -d @seqam/scenario.json
```

### Continuous loop (reruns automatically)

```bash
SEQAM_API_URL=http://172.22.174.149:8000 \
./scripts/run_scenario_loop.sh
```

This wrapper posts the scenario JSON to SeQaM after each 120-second run. Let it run for as many groups or iterations as needed, then `Ctrl+C`.

---

## Auto-stop on Pi clients

When students set `AUTO_STOP=true`, `main.py` behaves as follows:

1. Subscribes to `edgelab.phase` on Kafka.
2. Blocks until the first phase message arrives (prints "Waiting for experiment phase to start...").
3. Records the starting phase.
4. Starts all threads (FrameReader, Dispatcher, Scorer, SPAgent).
5. Monitors subsequent phase transitions.
6. When all four phases have been seen AND the phase returns to the starting phase, calls `request_shutdown()`.
7. Threads stop, the summary is printed, and the process exits.

For development without SeQaM, use `AUTO_STOP=false` and let the app run until `Ctrl+C`.

---

## Verifying SeQaM is running

```bash
curl http://172.22.174.149:8000/health
```

Or check whether phase events are appearing in Kafka after triggering a scenario run:

```bash
kafka-console-consumer.sh \
    --bootstrap-server 172.22.174.149:9092 \
    --topic edgelab.phase
```

---

## Phase alignment between SeQaM and tc_controller

`seqam/scenario.json` writes only phase names that exist in `tc_controller.py`'s `PHASE_MAP`: `baseline`, `gpu_load`, `bandwidth_50`, and `mixed`. tc_controller sees each write, runs the corresponding `tc_control.sh` command, and publishes the phase to Kafka. Students see exactly these four values in `self.experiment_phase`.

If you add a new phase to the scenario, add a matching entry to `PHASE_MAP` in `tc_controller.py` before running the scenario.
