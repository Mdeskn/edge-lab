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
   | SSH at t=60s    echo network_load > /tmp/edgelab_phase
   |                 + run tc_apply.sh on Network VM
   | SSH at t=90s    echo combined > /tmp/edgelab_phase
   |                 + launch gpu_stressor
   | SSH at t=119s   run tc_clear.sh
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
| 60-90 s | `network_load` | No | Yes (100 ms delay, 20 ms jitter, 2% loss) |
| 90-120 s | `combined` | Yes | Yes |

At t=119 s, tc rules are cleared so the next loop iteration starts clean.

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

## Known integration gap

The `tc_controller.py` PHASE_MAP currently covers `baseline`, `gpu_load`, `bandwidth_50`, `bandwidth_200`, `jitter_light`, and `mixed`. SeQaM's scenario also writes `network_load` and `combined`. If those phase names are not in PHASE_MAP, tc_controller logs a warning and does not publish those phase events to Kafka.

Students would then only see `"baseline"` and `"gpu_load"` in `self.experiment_phase`, not `"network_load"` or `"combined"`. Adding those entries to PHASE_MAP in `tc_controller.py` (with `tc_args: ["passthrough"]` or similar) and updating the apply logic to skip tc execution for them would fix this. See `docs/vm3-network-vm.md` for context.
