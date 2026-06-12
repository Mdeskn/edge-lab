# EdgeLab Startup Instructions

This guide explains the correct startup order for bringing the full EdgeLab pipeline online.

The main idea:

```text
VM1  = SeQaM + Kafka + Grafana/Prometheus
VM2  = GPU server + Triton + server metrics publisher
VM3  = network/router VM + tc_controller + network metrics publisher
LC1  = external GPU load client
Pi   = EdgeLab client app + dashboard
```

Start the machines in this order:

```text
1. VM1
2. VM2
3. VM3
4. LC1
5. Raspberry Pi
6. SeQaM experiment
```

---

## 0. Machine map

| Role | Host/IP | User | Main path |
|---|---|---|---|
| VM1 / SeQaM + Kafka | `172.22.174.149` | `mae` | `/home/mae/grafana-kafka` and `~/.seqam_fh_dortmund_project_emulate` |
| VM2 / GPU Triton server | `172.22.174.145` | `mae` | `/home/mae/server` |
| VM3 / Network control VM | `172.22.174.148` | `mae` | `/home/mae/network_load` |
| LC1 / GPU load client | `172.22.232.19` | `lc1` | `/home/lc1/edgelab-load-client` |
| Raspberry Pi / client + dashboard | Raspberry Pi hostname/IP | `mae` | `~/edge-lab` |

---

# 1. Start VM1: SeQaM, Kafka, Grafana, Prometheus

SSH into VM1:

```bash
ssh mae@172.22.174.149
```

Start the Kafka/Grafana/Prometheus stack:

```bash
cd /home/mae/grafana-kafka
docker compose up -d
```

Check that the important containers are running:

```bash
docker ps | grep -E 'kafka|ui|grafana|prometheus'
```

Expected containers include:

```text
dnn-partition-kafka
dnn-partition-kafka-ui
dnn-partition-grafana
dnn-partition-prometheus
```

Check Kafka topics:

```bash
docker exec -it dnn-partition-kafka kafka-topics.sh \
  --bootstrap-server localhost:9092 \
  --list
```

Expected important topics:

```text
dnn_partition.client_metrics
dnn_partition.server_metrics
edgelab.network.metrics
edgelab.phase
```

Check SeQaM UI/API:

```bash
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8000
```

Expected:

```text
200
```

---

# 2. Start VM2: GPU server and Triton

SSH into VM2:

```bash
ssh mae@172.22.174.145
```

Start Triton:

```bash
cd /home/mae/server
docker compose up -d
```

Check Triton health:

```bash
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8000/v2/health/ready
```

Expected:

```text
200
```

Check that Triton is listening:

```bash
ss -tulpn | grep -E '8000|8001|8002'
```

Expected ports:

```text
8000  HTTP
8001  gRPC
8002  metrics
```

Start the server/GPU metrics publisher if it is not already running:

```bash
ps aux | grep kafka_metrics_publisher | grep -v grep
```

If nothing appears, start it:

```bash
cd /home/mae/server
source .venv/bin/activate

nohup python3 -u kafka_metrics_publisher.py \
  --kafka-bootstrap-servers 172.22.174.149:9092 \
  > server_metrics.log 2>&1 &
```

Check the log:

```bash
tail -30 /home/mae/server/server_metrics.log
```

Optional: watch Triton model inference count:

```bash
watch -n 2 "curl -s http://localhost:8002/metrics | grep 'nv_inference_count{model=\"resnet50_full\"}'"
```

---

# 3. Start VM3: network publisher and phase controller

SSH into VM3:

```bash
ssh mae@172.22.174.148
```

Go to the network control folder:

```bash
cd /home/mae/network_load
```

Set clean baseline and clear traffic control rules:

```bash
printf "baseline\n" > /tmp/edgelab_phase
sudo -n /home/mae/network_load/tc_control.sh clear
```

The `-n` flag is important: it fails immediately if passwordless sudo is not configured. The expected behavior is no password prompt and successful clear.

Start the network metrics publisher:

```bash
/home/mae/network_load/network_publisher_service.sh start
```

Start the phase controller:

```bash
/home/mae/network_load/tc_controller_service.sh start
```

Check both services:

```bash
/home/mae/network_load/network_publisher_service.sh status
/home/mae/network_load/tc_controller_service.sh status
```

Expected:

```text
network_conditions_publisher running
tc_controller running
```

Check current network state:

```bash
cat /tmp/edgelab_phase
sudo -n /home/mae/network_load/tc_control.sh show
```

Expected:

```text
baseline
qdisc fq_codel ...
```

Useful logs:

```bash
/home/mae/network_load/network_publisher_service.sh log
/home/mae/network_load/tc_controller_service.sh log
```

Follow logs live:

```bash
/home/mae/network_load/tc_controller_service.sh follow
```

---

# 4. Prepare LC1: GPU load client

SSH into LC1:

```bash
ssh lc1@172.22.232.19
```

Go to the load client folder:

```bash
cd /home/lc1/edgelab-load-client
```

Make sure GPU load is stopped before starting an experiment:

```bash
./run_gpu_load.sh stop
./run_gpu_load.sh status
```

Expected:

```text
gpu load not running
```

During SeQaM experiments, SeQaM will start and stop this automatically.

Optional watcher:

```bash
watch -n 1 './run_gpu_load.sh status; docker ps --filter name=edgelab_gpu_load'
```

---

# 5. Start the Raspberry Pi app and dashboard

SSH into the Raspberry Pi:

```bash
ssh mae@raspberrypi
```

Go to the EdgeLab repo:

```bash
cd ~/edge-lab
```

Start the Pi client and dashboard:

```bash
docker compose -f docker-compose.pi.yml down
docker compose -f docker-compose.pi.yml up
```

Expected logs should show:

```text
dashboard-backend started
dashboard-frontend started
client started
RemoteClient connected to Triton
Kafka producer connected
Dashboard Kafka consumer subscribed
Frames posted to /api/frame
```

Open the dashboard in the browser.

Expected dashboard state before experiment:

```text
Video visible
Kafka metrics green
Client frames green
Experiment phase: baseline
Network mode: clear
Bandwidth: unlimited
Delay: 0.0 ms
Jitter: 0.0 ms
```

---

# 6. Run the SeQaM experiment

Open SeQaM in the browser:

```text
http://172.22.174.149:8000
```

In the SeQaM console, run:

```text
start_module module:experiment_dispatcher source:cache
```

Use `source:cache` when the experiment config was posted through the SeQaM API docs page and is stored in cache.

Expected experiment phases:

```text
baseline
gpu_load
jitter_light
bandwidth_50
mixed
baseline
```

The final `baseline` phase is important because it returns the network to a clean state.

---

# 7. Kafka-side verification

On VM1, open live consumers to confirm Kafka messages.

## Phase topic

```bash
docker exec -it dnn-partition-kafka kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic edgelab.phase
```

Expected messages:

```text
phase: gpu_load, status: applied
phase: jitter_light, status: applied
phase: bandwidth_50, status: applied
phase: mixed, status: applied
phase: baseline, status: applied
```

## Network metrics topic

```bash
docker exec -it dnn-partition-kafka kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic edgelab.network.metrics
```

Expected changes:

```text
mode: clear
mode: netem_tbf
mode: tbf
bandwidth: 50mbit
tc_active: true/false
delay_ms / jitter_ms values changing during network phases
```

## Server/GPU metrics topic

```bash
docker exec -it dnn-partition-kafka kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic dnn_partition.server_metrics
```

Expected:

```text
GPU/Triton/server metrics arriving regularly
Values change during gpu_load and mixed phases
```

## Client metrics topic

```bash
docker exec -it dnn-partition-kafka kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic dnn_partition.client_metrics
```

Expected:

```text
client mode
latency
displacement
cumulative score
frame metrics
```

---

# 8. Normal cleanup after testing

The SeQaM experiment should return to baseline automatically at the end. Still, when finished for the day, run this on VM3:

```bash
ssh mae@172.22.174.148
/home/mae/network_load/set_phase.sh baseline
sudo -n /home/mae/network_load/tc_control.sh clear
sudo -n /home/mae/network_load/tc_control.sh show
```

Stop GPU load on LC1:

```bash
ssh lc1@172.22.232.19
cd /home/lc1/edgelab-load-client
./run_gpu_load.sh stop
./run_gpu_load.sh status
```

Stop the Pi app:

```bash
ssh mae@raspberrypi
cd ~/edge-lab
docker compose -f docker-compose.pi.yml down
```

Optional: stop VM3 background services if you are done for the day:

```bash
ssh mae@172.22.174.148
/home/mae/network_load/tc_controller_service.sh stop
/home/mae/network_load/network_publisher_service.sh stop
```

For demo/testing days, it is okay to leave the VM3 services running.

---

# 9. Emergency reset

Use this if an experiment is interrupted and the network stays shaped.

## VM3 reset

```bash
ssh mae@172.22.174.148
/home/mae/network_load/set_phase.sh baseline
sudo -n /home/mae/network_load/tc_control.sh clear
sudo -n /home/mae/network_load/tc_control.sh show
```

## LC1 reset

```bash
ssh lc1@172.22.232.19
cd /home/lc1/edgelab-load-client
./run_gpu_load.sh stop
```

## Pi reset

```bash
ssh mae@raspberrypi
cd ~/edge-lab
docker compose -f docker-compose.pi.yml down
docker compose -f docker-compose.pi.yml up
```

---

# 10. Quick checklist before a demo

```text
VM1:
[ ] Kafka container healthy
[ ] SeQaM UI reachable
[ ] Required topics exist

VM2:
[ ] Triton health returns 200
[ ] kafka_metrics_publisher running

VM3:
[ ] network_conditions_publisher running
[ ] tc_controller running
[ ] phase is baseline
[ ] tc_control.sh show is clear/fq_codel
[ ] passwordless sudo works for tc_control.sh

LC1:
[ ] GPU load is stopped before experiment

Pi:
[ ] docker compose app is running
[ ] dashboard shows video
[ ] Kafka metrics green
[ ] client frames green

SeQaM:
[ ] experiment starts
[ ] phases appear on dashboard
[ ] experiment ends with baseline
```

---

# 11. Troubleshooting

## 11.1 Dashboard phase changes, but Network path stays `clear`

Symptom:

```text
Dashboard top-right phase changes to MIXED / JITTER_LIGHT / BANDWIDTH_50
but Infrastructure → Network path still shows:
Mode: clear
Bandwidth: unlimited
Delay: 0.0 ms
Jitter: 0.0 ms
```

Most likely cause: `tc_controller.py` received the phase but failed to apply `tc_control.sh`.

Check on VM3:

```bash
ssh mae@172.22.174.148
cd /home/mae/network_load
tail -80 tc_controller.log
```

If you see:

```text
sudo: a terminal is required to read the password
sudo: a password is required
status=failed
```

then passwordless sudo is missing or broken.

Fix on VM3:

```bash
sudo visudo -f /etc/sudoers.d/edgelab-tc-control
```

Add exactly this line:

```text
mae ALL=(root) NOPASSWD: /home/mae/network_load/tc_control.sh
```

Then:

```bash
sudo chmod 440 /etc/sudoers.d/edgelab-tc-control
sudo -n /home/mae/network_load/tc_control.sh clear
echo $?
```

Expected:

```text
0
```

Restart controller:

```bash
/home/mae/network_load/tc_controller_service.sh restart
```

Test:

```bash
/home/mae/network_load/set_phase.sh jitter_light
sleep 2
tail -20 /home/mae/network_load/tc_controller.log
sudo -n /home/mae/network_load/tc_control.sh show
```

Expected:

```text
Phase 'jitter_light' applied successfully
qdisc netem ...
```

Reset:

```bash
/home/mae/network_load/set_phase.sh baseline
```

---

## 11.2 `tc_controller.py` says `status=failed`

Check the controller log:

```bash
cd /home/mae/network_load
tail -80 tc_controller.log
```

Common causes:

### Cause A: sudo password problem

Error:

```text
sudo: a terminal is required to read the password
sudo: a password is required
```

Fix: configure passwordless sudo for `/home/mae/network_load/tc_control.sh` as described in section 11.1.

### Cause B: script path changed

Check:

```bash
ls -l /home/mae/network_load/tc_control.sh
```

If the file moved or was renamed, update the sudoers rule and `tc_controller_service.sh`.

### Cause C: invalid phase name

Valid phase names:

```text
baseline
bandwidth_200
bandwidth_50
jitter_light
gpu_load
mixed
```

Check current phase:

```bash
cat /tmp/edgelab_phase
```

Reset:

```bash
/home/mae/network_load/set_phase.sh baseline
```

---

## 11.3 SeQaM says experiment finished successfully, but network did not change

SeQaM only proves that the SSH commands ran. It does not prove that VM3 applied traffic control rules.

Check the real chain:

```text
SeQaM SSH command
→ set_phase.sh writes /tmp/edgelab_phase
→ tc_controller.py notices phase
→ tc_control.sh applies qdisc
→ network_conditions_publisher.py publishes real network state
→ Kafka edgelab.network.metrics
→ dashboard
```

On VM3:

```bash
cd /home/mae/network_load
cat /tmp/edgelab_phase
tail -80 tc_controller.log
tail -30 network_conditions_publisher.log
sudo -n /home/mae/network_load/tc_control.sh show
```

On VM1, watch Kafka:

```bash
docker exec -it dnn-partition-kafka kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic edgelab.network.metrics
```

If `edgelab.phase` changes but `edgelab.network.metrics` stays `clear`, the issue is on VM3 traffic control, not SeQaM.

---

## 11.4 `--phase-file: command not found` or `--kafka-bootstrap-servers: command not found`

Symptom:

```text
--phase-file: command not found
--kafka-bootstrap-servers: command not found
--topic: command not found
```

Cause: a multi-line command was pasted with blank lines after the `\` continuation character.

Bad:

```bash
nohup python3 -u tc_controller.py \

  --phase-file /tmp/edgelab_phase \
```

Good: use the service script instead:

```bash
/home/mae/network_load/tc_controller_service.sh restart
```

Or paste as one single line:

```bash
cd /home/mae/network_load && source .venv/bin/activate && nohup python3 -u tc_controller.py --phase-file /tmp/edgelab_phase --kafka-bootstrap-servers 172.22.174.149:9092 --topic edgelab.phase --tc-script /home/mae/network_load/tc_control.sh > tc_controller.log 2>&1 &
```

---

## 11.5 Background process is `Stopped`

Symptom:

```text
[2]+ Stopped ...
```

or `ps` shows:

```text
Tl
```

Cause: a background process tried to ask for sudo password or was interrupted.

Fix:

```bash
jobs
kill %2 %3 2>/dev/null || true
ps aux | grep tc_controller | grep -v grep
```

If old stopped processes remain, kill them by PID:

```bash
kill -9 PID_HERE
```

Then restart cleanly:

```bash
/home/mae/network_load/tc_controller_service.sh restart
```

---

## 11.6 Network publisher is running, but always publishes `clear`

Check if real `tc` state is actually clear:

```bash
sudo -n /home/mae/network_load/tc_control.sh show
```

If it shows:

```text
qdisc fq_codel ...
```

then the publisher is correct: the network really is clear.

Apply a phase:

```bash
/home/mae/network_load/set_phase.sh jitter_light
sleep 2
sudo -n /home/mae/network_load/tc_control.sh show
tail -10 /home/mae/network_load/network_conditions_publisher.log
```

Expected:

```text
mode=netem_tbf
tc_active=True
delay_ms=...
bandwidth=...
```

If `tc_control.sh show` shows `netem` or `tbf` but publisher still logs `clear`, restart publisher:

```bash
/home/mae/network_load/network_publisher_service.sh restart
```

---

## 11.7 GPU load changes show on dashboard, but network changes do not

This means VM2/LC1/server metrics are working, but VM3 network metrics are not.

Check VM3:

```bash
/home/mae/network_load/network_publisher_service.sh status
/home/mae/network_load/tc_controller_service.sh status
tail -80 /home/mae/network_load/tc_controller.log
tail -30 /home/mae/network_load/network_conditions_publisher.log
```

Check Kafka network topic on VM1:

```bash
docker exec -it dnn-partition-kafka kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic edgelab.network.metrics
```

If Kafka network messages change but dashboard does not, restart the Pi dashboard/app:

```bash
ssh mae@raspberrypi
cd ~/edge-lab
docker compose -f docker-compose.pi.yml down
docker compose -f docker-compose.pi.yml up
```

---

## 11.8 Dashboard shows `Experiment phase: MIXED`, but Network path shows `clear`

This can happen if the dashboard receives the `edgelab.phase` topic but the actual network shaping failed.

Check `edgelab.phase` messages on Kafka:

```bash
docker exec -it dnn-partition-kafka kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic edgelab.phase \
  --from-beginning \
  --timeout-ms 20000
```

Look at `status`:

```text
"status": "applied"  = tc_controller applied the phase
"status": "failed"   = phase label was published, but tc_control.sh failed
"status": "dry_run"  = old test messages; ignore for current live runs
```

If `status=failed`, go to section 11.1.

---

## 11.9 SeQaM direct sudo command fails

Error:

```text
sudo: a terminal is required to read the password
sudo: a password is required
```

Do not put direct sudo commands in `ExperimentConfig.json`, such as:

```json
{
  "command": "ssh router net-vm sudo /home/mae/network_load/tc_control.sh clear",
  "executionTime": 0
}
```

Instead, use:

```json
{
  "command": "ssh router net-vm bash /home/mae/network_load/set_phase.sh baseline",
  "executionTime": 0
}
```

Then `tc_controller.py` applies the actual sudo command locally on VM3.

---

## 11.10 Kafka phase topic has old `dry_run` messages

This is normal. Old messages remain in Kafka if consumed with `--from-beginning`.

Current live runs should show:

```text
"status": "applied"
```

Ignore older messages with:

```text
"status": "dry_run"
```

To watch only new messages, omit `--from-beginning`:

```bash
docker exec -it dnn-partition-kafka kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic edgelab.phase
```

---

## 11.11 Kafka consumer exits with `TimeoutException`

This is normal when using:

```bash
--timeout-ms 20000
```

It means no new message arrived before the timeout. It does not mean Kafka is broken.

Use no timeout for live watching:

```bash
docker exec -it dnn-partition-kafka kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic edgelab.phase
```

Stop with `Ctrl+C`.

---

## 11.12 Dashboard video blanks or flickers

The dashboard does not play the original MP4 directly. It shows processed frames sent by the client:

```text
client → POST /api/frame → dashboard backend → frontend GET /api/frame
```

Possible causes:

```text
1. Pi CPU is busy and cannot send frames smoothly.
2. Dashboard requests frames faster than backend receives them.
3. Frontend briefly clears an image before the next one loads.
4. Network or Docker resource hiccup.
```

Check logs on the Pi:

```bash
cd ~/edge-lab
docker compose -f docker-compose.pi.yml logs -f dashboard-backend
docker compose -f docker-compose.pi.yml logs -f client
```

If needed, reduce dashboard frame publishing rate in the app configuration if available.

---

## 11.13 Pi cannot connect to Triton

The Pi should connect to the VM3 router, not directly to the GPU server.

Expected Triton URL:

```text
172.22.174.148:8001
```

Check Pi app logs:

```bash
cd ~/edge-lab
docker compose -f docker-compose.pi.yml logs -f client
```

Check VM3 forwarding/NAT and VM2 Triton:

```bash
# VM2
ssh mae@172.22.174.145
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8000/v2/health/ready

# VM3
ssh mae@172.22.174.148
ss -tulpn | grep 8001
```

---

## 11.14 Final known-good VM3 state

Before every experiment, VM3 should look like this:

```bash
cd /home/mae/network_load

/home/mae/network_load/network_publisher_service.sh status
/home/mae/network_load/tc_controller_service.sh status
cat /tmp/edgelab_phase
sudo -n /home/mae/network_load/tc_control.sh show
sudo -n /home/mae/network_load/tc_control.sh clear
```

Expected:

```text
network_conditions_publisher running
tc_controller running
baseline
qdisc fq_codel ...
```

---

# 12. Mental model

```text
SeQaM does the schedule.
VM3 services listen and apply network phases.
LC1 creates GPU load when SeQaM tells it to.
VM2 serves Triton and publishes GPU/server metrics.
Pi runs the student app and dashboard.
Kafka carries phase, network, server, and client metrics.
Dashboard visualizes everything live.
```
