# EdgeLab Starter Instructions

This document explains the startup order for the EdgeLab demo/lab environment.

Goal: bring up the central services, GPU/Triton server, network-control VM, GPU-load client, and Raspberry Pi application/dashboard in the correct order.

---

## 0. Machine overview

| Machine | Role | User | IP / Host | Main path |
|---|---|---|---|---|
| VM1 | SeQaM + Kafka + Grafana + Prometheus | `mae` | `172.22.174.149` | `/home/mae/grafana-kafka` and `~/.seqam_fh_dortmund_project_emulate` |
| VM2 | GPU / Triton server | `mae` | `172.22.174.145` | `/home/mae/server` |
| VM3 | Network router / traffic shaping VM | `mae` | `172.22.174.148` | `/home/mae/network_load` |
| LC1 | External GPU-load client | `lc1` | `172.22.232.19` | `/home/lc1/edgelab-load-client` |
| Raspberry Pi | Edge app + dashboard | `mae` | Pi address depends on network | `~/edge-lab` |

The correct startup order is:

```text
1. VM1 central stack
2. VM2 Triton GPU server + server metrics publisher
3. VM3 network publisher + tc_controller
4. LC1 GPU-load client ready/stopped
5. Raspberry Pi app/dashboard
6. SeQaM experiment dispatcher
```

---

## 1. Start/check VM1: SeQaM, Kafka, Grafana, Prometheus

SSH into VM1:

```bash
ssh mae@172.22.174.149
```

Go to the Kafka/Grafana stack:

```bash
cd /home/mae/grafana-kafka
```

Check that the core containers are running:

```bash
docker ps | grep -E 'kafka|ui|grafana|prometheus'
```

Expected containers include:

```text
dnn-partition-kafka
dnn-partition-kafka-ui
dnn-partition-grafana
dnn-partition-prometheus
grafana
```

Check that Kafka topics exist:

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

`spans` may also exist. It is observability/tracing noise for this lab. The EdgeLab app/dashboard does not depend on it.

Check SeQaM API:

```bash
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8000/health
```

Expected:

```text
200
```

If the central stack is down, use the existing stack manager:

```bash
/home/mae/edgelab-stack-manager.sh
```

Then choose the start option.

---

## 2. Start/check VM2: GPU / Triton server

SSH into VM2:

```bash
ssh mae@172.22.174.145
```

Go to the server directory:

```bash
cd /home/mae/server
```

Start Triton:

```bash
docker compose up -d
```

Check container status:

```bash
docker ps | grep triton
```

Check Triton health:

```bash
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8000/v2/health/ready
```

Expected:

```text
200
```

Check GPU:

```bash
nvidia-smi
```

Start the server metrics publisher if it is not already running:

```bash
ps aux | grep kafka_metrics_publisher | grep -v grep
```

If no process is shown, start it:

```bash
cd /home/mae/server
source .venv/bin/activate

nohup python3 -u kafka_metrics_publisher.py \
  --kafka-bootstrap-servers 172.22.174.149:9092 \
  > server_metrics.log 2>&1 &
```

Check the log:

```bash
tail -40 server_metrics.log
```

Expected: no Kafka connection errors.

Optional: confirm VM2 can reach Kafka:

```bash
nc -vz 172.22.174.149 9092
```

---

## 3. Start/check VM3: network router + traffic shaping services

SSH into VM3:

```bash
ssh mae@172.22.174.148
```

Go to the network-load directory:

```bash
cd /home/mae/network_load
```

Set baseline and clear traffic shaping:

```bash
printf "baseline\n" > /tmp/edgelab_phase
sudo /home/mae/network_load/tc_control.sh clear
```

Start the network metrics publisher:

```bash
/home/mae/network_load/network_publisher_service.sh start
```

Start the traffic-control phase controller:

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

Check current phase:

```bash
cat /tmp/edgelab_phase
```

Expected:

```text
baseline
```

Check current traffic control state:

```bash
sudo /home/mae/network_load/tc_control.sh show
```

Expected: clear/default qdisc, or no active shaping.

Important: SeQaM should not run direct `sudo tc_control.sh` commands. SeQaM should only call:

```bash
/home/mae/network_load/set_phase.sh baseline
/home/mae/network_load/set_phase.sh jitter_light
/home/mae/network_load/set_phase.sh bandwidth_50
/home/mae/network_load/set_phase.sh mixed
```

Then `tc_controller.py` applies the sudo traffic-control commands locally on VM3.

---

## 4. Prepare LC1: GPU-load client

SSH into LC1:

```bash
ssh lc1@172.22.232.19
```

Go to the load-client directory:

```bash
cd /home/lc1/edgelab-load-client
```

Make sure GPU load is stopped before starting a new experiment:

```bash
./run_gpu_load.sh stop
```

Check status:

```bash
./run_gpu_load.sh status
```

Expected before experiment:

```text
gpu load not running
```

Optional manual test:

```bash
./run_gpu_load.sh start 8 30
```

Then check:

```bash
./run_gpu_load.sh status
```

Stop it again:

```bash
./run_gpu_load.sh stop
```

For normal experiments, SeQaM starts and stops this automatically.

---

## 5. Start the Raspberry Pi app + dashboard

SSH into the Raspberry Pi:

```bash
ssh mae@<PI_IP_ADDRESS>
```

Go to the EdgeLab repository:

```bash
cd ~/edge-lab
```

Optional: make sure old containers are gone:

```bash
docker compose -f docker-compose.pi.yml down
```

Start the app and dashboard:

```bash
docker compose -f docker-compose.pi.yml up
```

Expected logs should show:

```text
RemoteClient connected to Triton at 172.22.174.148:8001
AppMetricsPublisher connected to Kafka
Dashboard backend started on 8080
Dashboard Kafka consumer subscribed
FrameReader started
Dispatcher started
Scorer started
```

The important config values are:

```text
Triton URL:     172.22.174.148:8001
Kafka broker:   172.22.174.149:9092
Dashboard URL:  http://localhost:8080
```

The Pi talks to Triton through VM3, not directly to VM2.

Correct path:

```text
Pi client -> VM3 router 172.22.174.148:8001 -> VM2 Triton 172.22.174.145:8001
```

---

## 6. Open the dashboard

From the browser, open the dashboard address for the Pi.

Depending on how ports are exposed, use one of these:

```text
http://<PI_IP_ADDRESS>:3000
http://<PI_IP_ADDRESS>:8080
```

Expected dashboard signals:

```text
Video frame visible
Ground truth/prediction visible
Current mode LOCAL/REMOTE visible
Latency visible
Cumulative displacement score visible
Phase visible
Network metrics visible
GPU/server metrics visible
Data links green
```

If video sometimes blanks, the app may still be working. Check backend/client logs:

```bash
docker compose -f docker-compose.pi.yml logs -f dashboard-backend
docker compose -f docker-compose.pi.yml logs -f client
```

---

## 7. Run the SeQaM experiment

Go back to VM1:

```bash
ssh mae@172.22.174.149
```

Check the active ExperimentConfig:

```bash
cd ~/.seqam_fh_dortmund_project_emulate
python3 -m json.tool ExperimentConfig.json >/dev/null && echo "JSON OK"
cat ExperimentConfig.json
```

Start the experiment dispatcher from the SeQaM console or shell.

In the SeQaM console:

```text
start_module module:experiment_dispatcher
```

Expected phase sequence:

```text
baseline
gpu_load
jitter_light
bandwidth_50
mixed
baseline
```

The final `baseline` is important. It cleans up network shaping and returns the lab to a safe state.

---

## 8. Live Kafka validation during an experiment

On VM1, open separate terminals if possible.

### Terminal A: phase topic

```bash
cd /home/mae/grafana-kafka

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

### Terminal B: network metrics topic

```bash
cd /home/mae/grafana-kafka

docker exec -it dnn-partition-kafka kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic edgelab.network.metrics
```

Expected messages should change with phases:

```text
mode: clear
mode: netem_tbf
mode: tbf
bandwidth: 50mbit
delay_ms: ...
jitter_ms: ...
packet_loss_percent: ...
```

### Terminal C: server/GPU metrics topic

```bash
cd /home/mae/grafana-kafka

docker exec -it dnn-partition-kafka kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic dnn_partition.server_metrics
```

Expected: live GPU/Triton/server metrics.

During `gpu_load` and `mixed`, GPU/server metrics should change because LC1 is generating load.

### Terminal D: client metrics topic

```bash
cd /home/mae/grafana-kafka

docker exec -it dnn-partition-kafka kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic dnn_partition.client_metrics
```

Expected: live Pi/client metrics such as mode, latency, displacement, and cumulative score.

---

## 9. Normal shutdown / cleanup

### Stop Pi app

On the Raspberry Pi:

```bash
cd ~/edge-lab
docker compose -f docker-compose.pi.yml down
```

### Stop GPU load

On LC1:

```bash
cd /home/lc1/edgelab-load-client
./run_gpu_load.sh stop
```

### Reset network shaping

On VM3:

```bash
cd /home/mae/network_load
/home/mae/network_load/set_phase.sh baseline
sudo /home/mae/network_load/tc_control.sh clear
cat /tmp/edgelab_phase
sudo /home/mae/network_load/tc_control.sh show
```

Expected:

```text
baseline
```

For demo/testing days, it is okay to leave these two VM3 services running:

```text
network_conditions_publisher
tc_controller
```

They are listeners/controllers and do not create load by themselves.

If ending the day completely, stop them:

```bash
/home/mae/network_load/tc_controller_service.sh stop
/home/mae/network_load/network_publisher_service.sh stop
```

### Optional: keep central services running

Usually VM1 Kafka/Grafana/Prometheus/SeQaM can stay running.

If needed, stop via:

```bash
/home/mae/edgelab-stack-manager.sh
```

---

## 10. Emergency cleanup

Use this if the experiment was interrupted halfway or the network still feels shaped.

### VM3 emergency reset

```bash
ssh mae@172.22.174.148

cd /home/mae/network_load
printf "baseline\n" > /tmp/edgelab_phase
sudo /home/mae/network_load/tc_control.sh clear
/home/mae/network_load/tc_controller_service.sh status
/home/mae/network_load/network_publisher_service.sh status
sudo /home/mae/network_load/tc_control.sh show
```

### LC1 emergency stop

```bash
ssh lc1@172.22.232.19

cd /home/lc1/edgelab-load-client
./run_gpu_load.sh stop
./run_gpu_load.sh status
```

### Pi emergency stop

```bash
ssh mae@<PI_IP_ADDRESS>

cd ~/edge-lab
docker compose -f docker-compose.pi.yml down
```

---

## 11. Quick full startup checklist

Use this before a demo.

```text
[ ] VM1 Kafka container healthy
[ ] VM1 topics exist
[ ] VM1 SeQaM API returns 200
[ ] VM2 Triton health returns 200
[ ] VM2 server metrics publisher running
[ ] VM3 phase is baseline
[ ] VM3 network shaping is clear
[ ] VM3 network publisher running
[ ] VM3 tc_controller running
[ ] LC1 GPU load stopped
[ ] Pi docker compose starts app/dashboard
[ ] Dashboard shows video
[ ] Dashboard shows phase/network/GPU/client metrics
[ ] SeQaM experiment runs
[ ] edgelab.phase messages show status applied
[ ] Experiment ends in baseline
```

---

## 12. What each Kafka topic means

| Topic | Meaning | Producer | Consumer |
|---|---|---|---|
| `edgelab.phase` | Current experiment phase | VM3 `tc_controller.py` | Dashboard, SP-Agent |
| `edgelab.network.metrics` | Current network shaping state | VM3 `network_conditions_publisher.py` | Dashboard, SP-Agent |
| `dnn_partition.server_metrics` | GPU/Triton/server metrics | VM2 `kafka_metrics_publisher.py` | Dashboard, SP-Agent |
| `dnn_partition.client_metrics` | Pi/client latency/mode/score | Pi client app | Dashboard, SP-Agent |
| `spans` | Observability/tracing data | SeQaM/SigNoz/OpenTelemetry stack | Not needed for EdgeLab app |

For this lab, focus on:

```text
edgelab.phase
edgelab.network.metrics
dnn_partition.server_metrics
dnn_partition.client_metrics
```

Ignore `spans` unless someone explicitly wants tracing/debug observability.

---

## 13. Most common problems

### Problem: SeQaM command fails with sudo error

Symptom:

```text
sudo: a terminal is required to read the password
```

Fix: do not run direct sudo from SeQaM. Use:

```bash
ssh router net-vm bash /home/mae/network_load/set_phase.sh baseline
```

The VM3 `tc_controller.py` handles sudo locally.

### Problem: Kafka topic exists but no new messages

Check producer service:

```bash
# VM3
/home/mae/network_load/network_publisher_service.sh status
/home/mae/network_load/tc_controller_service.sh status

# VM2
ps aux | grep kafka_metrics_publisher | grep -v grep

# Pi
docker compose -f docker-compose.pi.yml logs -f client
```

### Problem: Pi cannot connect to Triton

Check path:

```text
Pi should use 172.22.174.148:8001
VM3 forwards to 172.22.174.145:8001
```

On VM3:

```bash
nc -vz 172.22.174.145 8001
```

On VM2:

```bash
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8000/v2/health/ready
```

### Problem: GPU load remains running after experiment

On LC1:

```bash
cd /home/lc1/edgelab-load-client
./run_gpu_load.sh stop
./run_gpu_load.sh status
```

### Problem: Network shaping remains active after experiment

On VM3:

```bash
cd /home/mae/network_load
/home/mae/network_load/set_phase.sh baseline
sudo /home/mae/network_load/tc_control.sh clear
sudo /home/mae/network_load/tc_control.sh show
```
