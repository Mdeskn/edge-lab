# EdgeLab

EdgeLab is a hands-on edge-computing lab for the IoT and Edge Computing course
at FH Dortmund. A Raspberry Pi processes a pre-recorded drone-view video and
tracks one target car frame by frame. For every frame, the student service
placement agent decides whether to run YOLO locally on the Pi CPU or send the
frame through the network to a remote GPU server.

The student task is intentionally small:

```python
# client/student/sp_agent.py
def decide(self) -> str:
    return "local"  # or "remote"
```

Everything else in the pipeline is already wired: video reading, local and
remote inference, Kafka metrics, scoring, phase updates, and the live dashboard.

## Goal

Minimize cumulative displacement: the total pixel distance between the predicted
target center and the ground-truth target center over all scored frames. Lower
is better.

Fast inference matters because the car is moving. A late prediction can be
technically correct for the frame that was processed, but stale by the time the
result arrives. The lab teaches when remote GPU inference is worth the network
trip, and when local edge inference is safer.

## System Map

| Machine | IP / host | Main role |
| --- | --- | --- |
| VM1 | `172.22.174.149` | Kafka broker, Kafka UI, Grafana/Prometheus, SeQaM API when enabled |
| VM2 | `172.22.174.145` | GPU server, Triton, JPEG inference API, GPU metrics publisher |
| VM3 | `172.22.174.148` | Network/router VM, traffic shaping, network metrics, phase controller |
| LC1 | `172.22.229.169` | External GPU load client |
| Raspberry Pi | `172.22.229.167` | Student client app, local inference, dashboard |

Primary inference path:

```text
Raspberry Pi
  -> VM3 Router API endpoint 172.22.174.148:8100
  -> VM2 remote inference API 172.22.174.145:8100
  -> Triton gRPC on VM2
```

Legacy fallback path:

```text
Raspberry Pi
  -> VM3 Router Triton endpoint 172.22.174.148:8001
  -> VM2 Triton gRPC 172.22.174.145:8001
```

Always route student inference traffic through VM3. Direct Pi traffic to
`172.22.174.145` bypasses network impairment and invalidates the network part of
the experiment.

Metrics and phase data flow through Kafka on VM1:

```text
VM2 GPU metrics      -> dnn_partition.server_metrics
VM3 network metrics  -> edgelab.network.metrics
VM3 phase events     -> edgelab.phase
Pi frame results     -> dnn_partition.client_metrics
```

## Repository Layout

| Path | Purpose |
| --- | --- |
| `client/` | Raspberry Pi app: frame reader, dispatcher, scorer, SP agent, local/remote inference |
| `client/student/sp_agent.py` | The only file students edit |
| `client/student/sp_agent_base.py` | Base class exposing Kafka metrics and latency history |
| `dashboard/` | Per-group FastAPI + React live dashboard |
| `ground_truth/` | HSV-based ground-truth CSV generator |
| `publishers/network_conditions/` | VM3 network metrics publisher and phase controller |
| `remote_inference/` | VM2 JPEG-to-Triton gateway API |
| `seqam/` | SeQaM scenario and SSH target configuration |
| `scripts/` | Benchmarking and deployment/helper scripts |
| `docker-compose.*.yml` | Compose files for Pi, GPU server, Network VM, and local runs |

## Student Quick Start

1. Clone the repository and prepare lab data:

```bash
git clone <repo-url> edge-lab
cd edge-lab
mkdir -p data
```

Copy the supplied files into `data/`:

```text
data/video.mp4
data/ground_truth.csv
data/yolov10n.onnx
```

Some older notes refer to `data/test_video.mp4`; the current default
configuration uses `data/video.mp4`. Use whatever file name is set in `.env`.

2. Configure the environment:

```bash
cp .env.example .env
```

Minimum student settings:

```dotenv
GROUP_ID=1

VIDEO_PATH=data/video.mp4
GROUND_TRUTH_PATH=data/ground_truth.csv
MODEL_PATH=data/yolov10n.onnx

REMOTE_INFERENCE_URL=http://172.22.174.148:8100
REMOTE_JPEG_QUALITY=80
TRITON_URL=172.22.174.148:8001

KAFKA_BROKERS=172.22.174.149:9092
KAFKA_GPU_TOPIC=dnn_partition.server_metrics
KAFKA_NET_TOPIC=edgelab.network.metrics
KAFKA_PHASE_TOPIC=edgelab.phase
APP_METRICS_TOPIC=dnn_partition.client_metrics

TARGET_CLASS_ID=2,5,7
TARGET_CONFIDENCE_THRESHOLD=0.1

DISPLAY_OUTPUT=false
AUTO_STOP=false
```

Use `DISPLAY_OUTPUT=true` only when a display is attached. Use
`AUTO_STOP=true` for the real experiment run so the app waits for the first
phase message, runs one full phase cycle, then stops.

3. Install and run locally:

```bash
python3 -m venv .venv-client
source .venv-client/bin/activate
pip install -r client/requirements.txt

cd client
python main.py
```

4. Edit the placement logic:

```text
client/student/sp_agent.py
```

Restart the app after each change and inspect `data/results.csv` or
`results.csv`, depending on `RESULTS_LOG_PATH`.

## Docker On The Pi

The Pi can run the client and dashboard together:

```bash
docker compose -f docker-compose.pi.yml up --build
```

Open the dashboard frontend at:

```text
http://<pi-ip>:5173
```

The dashboard backend health endpoint is:

```text
http://<pi-ip>:8080/health
```

The client and dashboard backend run on the same Pi, so keep:

```dotenv
DASHBOARD_ENABLED=true
DASHBOARD_URL=http://localhost:8080
```

## Placement Agent API

`SPAgentBase` keeps the following values updated from Kafka and local app
history. Inside `decide()`, always use `.get(key, default)` when reading metric
dictionaries so the first missing Kafka message does not crash the agent.

Current phase:

```python
phase = self.experiment_phase  # defaults to "baseline"
```

GPU metrics from `dnn_partition.server_metrics`:

```python
gpu_util   = self.gpu_metrics.get("gpu_util_pct", 0)
gpu_temp   = self.gpu_metrics.get("gpu_temp_c", 0)
gpu_mem_mb = self.gpu_metrics.get("gpu_mem_used_mb", 0)
cpu_util   = self.gpu_metrics.get("cpu_util_pct", 0)
power_w    = self.gpu_metrics.get("power_w", 0)

yolo_queue = self.gpu_metrics.get("yolo_queue_ms", 0)
yolo_infer = self.gpu_metrics.get("yolo_infer_ms", 0)
yolo_rps   = self.gpu_metrics.get("yolo_success_rps", 0)
total_rps  = self.gpu_metrics.get("total_rps", 0)
total_pend = self.gpu_metrics.get("total_pending", 0)
```

`yolo_queue_ms` is usually the strongest remote-server warning signal. A rising
queue means remote inference will slow down before GPU utilization alone looks
obvious.

Network metrics from `edgelab.network.metrics`:

```python
delay     = self.net_metrics.get("delay_ms", 0)
jitter    = self.net_metrics.get("jitter_ms", 0)
loss      = self.net_metrics.get("packet_loss_pct", 0)
bandwidth = self.net_metrics.get("bandwidth", "unlimited")
mode      = self.net_metrics.get("mode", "clear")
```

Local latency history:

```python
avg_lat  = self.avg_latency       # mean of last 20 latencies, or None
all_lats = self.recent_latencies  # list[float], last 20 frame latencies
mode     = self.current_mode      # "local" or "remote"
```

Example strategy:

```python
def decide(self) -> str:
    phase = self.experiment_phase

    if phase in ("gpu_load", "jitter_light", "bandwidth_5", "mixed"):
        return "local"

    if self.gpu_metrics.get("yolo_queue_ms", 0) > 50:
        return "local"
    if self.gpu_metrics.get("gpu_util_pct", 0) > 85:
        return "local"
    if self.net_metrics.get("delay_ms", 0) > 40:
        return "local"

    return "remote"
```

Stronger agents also avoid thrashing by requiring a bad condition to persist for
two or three `decide()` calls before switching modes.

## Scoring

For each scored frame:

```text
displacement_px = sqrt((predicted_x - true_x)^2 + (predicted_y - true_y)^2)
```

| Situation | Score behavior |
| --- | --- |
| Target absent in ground truth | Frame is excluded from cumulative score |
| Target present and model detects it | Euclidean distance in pixels |
| Target present and model misses it | `MISS_PENALTY_PX`, default `100.0` |

The fixed miss penalty keeps a single missed detection from dominating the score
as a 1000+ px distance from origin. Set `MISS_PENALTY_PX=0` only if you want to
skip missed detections entirely.

The app writes per-frame results with:

```text
timestamp
frame_number
group_id
experiment_phase
processing_mode
latency_ms
true_x,true_y
predicted_x,predicted_y
displacement_px
cumulative_displacement_px
```

`processing_mode` can be `local`, `remote`, or `local_fallback`. A
`local_fallback` frame means remote inference failed or timed out, then the
dispatcher paid that cost and ran local inference.

## Experiment Phases

The current SeQaM scenario emits these phase names:

| Phase | Index | Network action | GPU load |
| --- | ---: | --- | --- |
| `baseline` | 0 | clear | no |
| `gpu_load` | 4 | clear | yes, triggered externally |
| `jitter_light` | 3 | `netem_tbf 0.1ms 0.4ms 1gbit 2mbit 50ms` | no |
| `bandwidth_5` | 6 | `tbf 5mbit 256kb 50ms` | no |
| `mixed` | 5 | `tbf 5mbit 256kb 50ms` | yes, triggered externally |

Phase rules stay active until the next phase. Do not pass a duration to
`tc_control.sh` from automated phase control; SeQaM controls timing.

`tc_controller.py` also accepts legacy operator phases such as `bandwidth_200`
and `bandwidth_50`, but they are not emitted by the current SeQaM scenario.
Student-facing examples should use only the current scenario phases above.

The current checked-in SeQaM scenario is a short heavy-load cycle:

| Time | Action |
| ---: | --- |
| 0 s | Stop GPU load; set `baseline` |
| 10 s | Set `gpu_load`; start LC1 GPU load at concurrency `32` |
| 25 s | Stop GPU load; set `jitter_light` |
| 40 s | Set `bandwidth_5` |
| 55 s | Set `mixed`; start LC1 GPU load at concurrency `32` |
| 70 s | Stop GPU load; set `baseline` |
| 75 s | Exit |

If the SeQaM scenario changes, update `client/student/sp_agent.py`,
`client/student/sp_agent_base.py`, and `client/main.py` at the same time.

## Dashboard

The dashboard is optional and best-effort. It should never slow or crash the
experiment. The Pi client sends throttled, JPEG-compressed annotated frames to a
FastAPI backend. The backend also consumes Kafka metrics when configured and
broadcasts state to the React frontend over WebSocket.

Main features:

| Feature | What it shows |
| --- | --- |
| Annotated video | Ground-truth car center, predicted box, displacement line |
| Processing mode | `LOCAL`, `REMOTE`, or `LOCAL_FALLBACK` |
| Latency | Current, average, min, max, p95 |
| Displacement | Current, rolling average, cumulative score |
| Phase | Current experiment phase from Kafka |
| Infrastructure | GPU utilization, GPU memory, Triton queue, network delay/jitter/loss |
| Charts | Rolling latency, displacement, cumulative score, placement mode, GPU, network |
| Summary | Local/remote percentages, total frames, final score, best/worst values |

Backend API:

```text
GET  /health
GET  /api/state
GET  /api/history
POST /api/frame
GET  /api/frame
POST /api/reset
WS   /ws
```

Important dashboard variables:

```dotenv
DASHBOARD_ENABLED=true
DASHBOARD_URL=http://localhost:8080
DASHBOARD_FPS=5
DASHBOARD_JPEG_QUALITY=70
DASHBOARD_FRAME_WIDTH=960

DASHBOARD_HOST=0.0.0.0
DASHBOARD_PORT=8080
DASHBOARD_MAX_HISTORY=300
DASHBOARD_PUBLIC_API_URL=
```

If the browser says reconnecting, check:

```bash
curl http://<pi-ip>:8080/health
```

If video is missing but charts update, check `DASHBOARD_ENABLED=true` and the
client logs for dashboard publisher startup.

## Ground Truth

Ground truth is generated offline with HSV color segmentation. It does not use
YOLO, so YOLO misses remain measurable.

```bash
python ground_truth/generate_ground_truth.py \
    --video data/video.mp4 \
    --output data/ground_truth.csv
```

The generator:

1. Converts each frame to HSV.
2. Masks both red hue bands, 0-10 and 160-179.
3. Cleans the mask with morphological open and close operations.
4. Selects the largest red blob within car-sized bounds.
5. Writes the bounding-box center to CSV.
6. Linearly interpolates remaining missing rows.

CSV columns:

| Column | Description |
| --- | --- |
| `frame_number` | 1-indexed frame counter |
| `center_x` | Car center X coordinate |
| `center_y` | Car center Y coordinate |
| `confidence` | Always `1.0` for color segmentation |
| `class_id` | Always `2` |
| `class_name` | Always `car` |

To export the ONNX model for inference:

```bash
pip install ultralytics
yolo export model=yolov10n.pt format=onnx
```

## Operator Startup

Use this order for a demo or lab session:

```text
1. VM1 central Kafka/Grafana/SeQaM stack
2. VM2 Triton GPU server and GPU metrics publisher
3. VM3 network publisher and tc_controller
4. LC1 GPU load client ready and stopped
5. Raspberry Pi app and dashboard
6. SeQaM experiment dispatcher
```

### VM1: Kafka, UI, Grafana, SeQaM

```bash
ssh mae@172.22.174.149
cd /home/mae/grafana-kafka
docker compose up -d
docker ps | grep -E 'kafka|ui|grafana|prometheus'
```

Check Kafka topics:

```bash
docker exec -it dnn-partition-kafka kafka-topics.sh \
  --bootstrap-server localhost:9092 \
  --list
```

Expected topics:

```text
dnn_partition.client_metrics
dnn_partition.server_metrics
edgelab.network.metrics
edgelab.phase
```

Useful URLs:

```text
Kafka broker: 172.22.174.149:9092
Kafka UI:     http://172.22.174.149:8080
Grafana:      http://172.22.174.149:3000
Prometheus:   http://172.22.174.149:9090
SeQaM API:    http://172.22.174.149:8000
```

Kafka has topic auto-creation enabled, so `edgelab.network.metrics` and
`edgelab.phase` appear once VM3 publishers send their first messages.

### VM2: GPU Server

```bash
ssh mae@172.22.174.145
cd /home/mae/server
docker compose up -d
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8000/v2/health/ready
nvidia-smi
```

Expected Triton ports:

```text
8000 HTTP
8001 gRPC
8002 metrics
8100 JPEG remote inference API, if deployed
```

Models:

| Model | Purpose |
| --- | --- |
| `yolov10n` | Student remote inference |
| `resnet50_full` | GPU load generation |

Start GPU metrics publisher if needed:

```bash
cd /home/mae/server
source .venv/bin/activate
nohup python3 -u kafka_metrics_publisher.py \
  --kafka-bootstrap-servers 172.22.174.149:9092 \
  > server_metrics.log 2>&1 &
```

The publisher reads `nvidia-smi` and Triton metrics at
`http://localhost:8002/metrics`, then publishes to
`dnn_partition.server_metrics`.

Start the remote JPEG API with:

```bash
docker compose -f docker-compose.gpu-server.yml up -d --build
```

Health checks:

```bash
curl http://172.22.174.145:8100/health  # direct, bypass check
curl http://172.22.174.148:8100/health  # router path, student path
```

### VM3: Network / Router VM

```bash
ssh mae@172.22.174.148
cd /home/mae/network_load
printf "baseline\n" > /tmp/edgelab_phase
sudo -n /home/mae/network_load/tc_control.sh clear
```

Start services:

```bash
/home/mae/network_load/network_publisher_service.sh start
/home/mae/network_load/tc_controller_service.sh start

/home/mae/network_load/network_publisher_service.sh status
/home/mae/network_load/tc_controller_service.sh status
```

Expected state:

```text
network_conditions_publisher running
tc_controller running
baseline
qdisc fq_codel ...
```

The VM3 interface is `ens18`. `tc_control.sh` supports:

```bash
sudo /home/mae/network_load/tc_control.sh show
sudo /home/mae/network_load/tc_control.sh clear
sudo /home/mae/network_load/tc_control.sh tbf 50mbit 2mbit 50ms
sudo /home/mae/network_load/tc_control.sh netem_tbf 0.1ms 0.4ms 1gbit 2mbit 50ms
```

Passwordless sudo for the tc script is required:

```text
mae ALL=(root) NOPASSWD: /home/mae/network_load/tc_control.sh
```

Put that in `/etc/sudoers.d/edgelab-tc-control` with mode `440`, then verify:

```bash
sudo -n /home/mae/network_load/tc_control.sh clear
```

VM3 routing checks:

```bash
sysctl net.ipv4.ip_forward
sudo iptables -t nat -L -n -v
nc -vz 172.22.174.145 8001
```

A missing `ss` listener on `8001` on VM3 is normal when forwarding is handled by
iptables DNAT rather than a user-space process.

### LC1: GPU Load Client

```bash
ssh lc1@172.22.229.169
cd /home/lc1/edgelab-load-client
./run_gpu_load.sh stop
./run_gpu_load.sh status
```

Manual test:

```bash
./run_gpu_load.sh start 8 30
./run_gpu_load.sh status
./run_gpu_load.sh stop
```

LC1 should load the GPU server directly, not through VM3:

```text
LC1 -> 172.22.174.145:8001
```

This stresses the server, not the shaped student network path.

### Raspberry Pi

```bash
ssh mae@<pi-ip>
cd ~/edge-lab
docker compose -f docker-compose.pi.yml down
docker compose -f docker-compose.pi.yml up
```

Expected logs:

```text
RemoteClient connected to JPEG inference API at http://172.22.174.148:8100
Kafka producer connected
Dashboard Kafka consumer subscribed
FrameReader started
Dispatcher started
Scorer started
```

## SeQaM

SeQaM executes timed SSH commands. The current SSH target config is
`seqam/ScenarioConfig.json`; merge it into:

```text
~/.seqam_fh_dortmund_project_emulate/ScenarioConfig.json
```

Current targets:

| Target | Host | User |
| --- | --- | --- |
| `net-vm` | `172.22.174.148` | `mae` |
| `gpu-server` | `172.22.174.145` | `mae` |
| `load-vm` | `172.22.229.169` | `lc1` |

In this SeQaM setup, these targets live under `router` in `ScenarioConfig.json`.

Run a one-shot scenario:

```bash
curl -X POST http://172.22.174.149:8000/config/ExperimentConfig.json \
  -H "Content-Type: application/json" \
  -d @seqam/scenario.json
```

Or use the SeQaM console:

```text
start_module module:experiment_dispatcher source:cache
```

The safest network-control pattern is:

```text
SeQaM -> set_phase.sh <phase> on VM3
VM3 tc_controller.py -> applies sudo tc_control.sh locally
VM3 tc_controller.py -> publishes edgelab.phase
VM3 network_conditions_publisher.py -> publishes actual tc state
```

Avoid direct `sudo tc_control.sh` commands in the SeQaM scenario unless the
sudoers setup has been deliberately verified. Using `set_phase.sh` keeps the
sudo operation local to VM3 and easier to debug.

## Kafka Validation

Run these from VM1 while an experiment is active.

Phase events:

```bash
docker exec -it dnn-partition-kafka kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic edgelab.phase
```

Network metrics:

```bash
docker exec -it dnn-partition-kafka kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic edgelab.network.metrics
```

GPU metrics:

```bash
docker exec -it dnn-partition-kafka kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic dnn_partition.server_metrics
```

Client metrics:

```bash
docker exec -it dnn-partition-kafka kafka-console-consumer.sh \
  --bootstrap-server localhost:9092 \
  --topic dnn_partition.client_metrics
```

Watch only new live messages by omitting `--from-beginning`. Old `dry_run`
messages in `edgelab.phase` are normal if they came from earlier testing.

## Benchmarking

Use this before writing strategy:

```bash
python scripts/benchmark_inference.py --mode local \
  --model data/yolov10n.onnx \
  --video data/video.mp4

python scripts/benchmark_inference.py --mode remote \
  --model data/yolov10n.onnx \
  --remote-inference-url http://172.22.174.148:8100

python scripts/benchmark_inference.py --mode both \
  --model data/yolov10n.onnx \
  --video data/video.mp4 \
  --remote-inference-url http://172.22.174.148:8100
```

The legacy gRPC path sends a 4.9 MB FP32 tensor per request. At a 50 Mbit/s cap,
that can produce hundreds of milliseconds of latency. The preferred JPEG gateway
reduces that network payload significantly.

## Cleanup And Reset

Normal cleanup:

```bash
# Pi
ssh mae@<pi-ip>
cd ~/edge-lab
docker compose -f docker-compose.pi.yml down

# LC1
ssh lc1@172.22.229.169
cd /home/lc1/edgelab-load-client
./run_gpu_load.sh stop

# VM3
ssh mae@172.22.174.148
/home/mae/network_load/set_phase.sh baseline
sudo -n /home/mae/network_load/tc_control.sh clear
sudo -n /home/mae/network_load/tc_control.sh show
```

Reset a dashboard between groups:

```bash
curl -X POST http://<pi-ip>:8080/api/reset
```

It is okay to leave VM3 `network_conditions_publisher` and `tc_controller`
running between demos. They only listen, publish state, and react to phase
changes.

## Troubleshooting

### App crashes with FileNotFoundError

Check `VIDEO_PATH`, `GROUND_TRUTH_PATH`, and `MODEL_PATH` in `.env`, then verify
the files exist.

### Remote inference is unreachable

Check:

```bash
curl http://172.22.174.148:8100/health
```

The Pi should use:

```dotenv
REMOTE_INFERENCE_URL=http://172.22.174.148:8100
```

Only use the legacy gRPC fallback when `REMOTE_INFERENCE_URL` is blank:

```dotenv
TRITON_URL=172.22.174.148:8001
```

### Kafka is unreachable

```bash
nc -zv 172.22.174.149 9092
```

If Kafka is down, the app continues without Kafka. The phase remains
`baseline`, GPU metrics are empty, and network metrics default to clear.

### Dashboard phase changes but network path stays clear

On VM3:

```bash
cd /home/mae/network_load
tail -80 tc_controller.log
sudo -n /home/mae/network_load/tc_control.sh show
```

If the log contains a sudo password error, fix passwordless sudo for
`tc_control.sh` and restart the controller:

```bash
/home/mae/network_load/tc_controller_service.sh restart
```

### `tc_controller.py` reports unknown phase

Check the phase file:

```bash
cat /tmp/edgelab_phase
```

Then compare with `PHASE_MAP` in:

```text
publishers/network_conditions/tc_controller.py
```

Every phase emitted by `seqam/scenario.json` must be accepted by VM3
`set_phase.sh` and defined in `PHASE_MAP`.

### Network publisher always reports clear

Check whether the network really is clear:

```bash
sudo -n /home/mae/network_load/tc_control.sh show
```

Apply a known phase and watch logs:

```bash
/home/mae/network_load/set_phase.sh jitter_light
sleep 2
tail -20 /home/mae/network_load/network_conditions_publisher.log
```

### SeQaM says the experiment finished but network did not change

Follow the real chain:

```text
SeQaM SSH command
-> set_phase.sh writes /tmp/edgelab_phase
-> tc_controller.py notices phase
-> tc_control.sh applies qdisc
-> network_conditions_publisher.py publishes actual state
-> Kafka edgelab.network.metrics
-> dashboard and SP agent
```

Check VM3 logs and Kafka network messages before debugging the dashboard.

### Multi-line shell command says `--phase-file: command not found`

A blank line after a `\` continuation broke the command. Use the service script:

```bash
/home/mae/network_load/tc_controller_service.sh restart
```

Or paste the command as one single line.

### Background process is stopped

Usually a background process tried to ask for a sudo password. Kill old jobs and
restart with service scripts:

```bash
jobs
ps aux | grep tc_controller | grep -v grep
/home/mae/network_load/tc_controller_service.sh restart
```

### GPU load keeps running

```bash
ssh lc1@172.22.229.169
cd /home/lc1/edgelab-load-client
./run_gpu_load.sh stop
./run_gpu_load.sh status
```

### Very low detection rate

Use the current car-tracking settings:

```dotenv
TARGET_CLASS_ID=2,5,7
TARGET_CONFIDENCE_THRESHOLD=0.1
```

Also verify that `ground_truth.csv` matches the video.

### High cumulative displacement

Common causes:

| Cause | What to check |
| --- | --- |
| Remote during bad network/GPU load | Filter results by phase and mode |
| Local during clean baseline | Compare local/remote benchmark results |
| Remote failures | Count `local_fallback` rows |
| Mode thrashing | Add switching hysteresis in `decide()` |

### Pi is overloaded

Try:

```dotenv
FRAME_INTERVAL_MS=200
DISPLAY_OUTPUT=false
LOG_LEVEL=WARNING
DASHBOARD_FPS=3
DASHBOARD_FRAME_WIDTH=640
```

## Environment Reference

| Variable | Purpose |
| --- | --- |
| `GROUP_ID` | Student group number |
| `VIDEO_PATH` | Input video path |
| `GROUND_TRUTH_PATH` | Ground-truth CSV path |
| `MODEL_PATH` | Local ONNX model path |
| `REMOTE_INFERENCE_URL` | Preferred JPEG API, normally `http://172.22.174.148:8100` |
| `REMOTE_JPEG_QUALITY` | JPEG quality for remote inference |
| `REMOTE_INFERENCE_TIMEOUT_SEC` | Remote API timeout |
| `TRITON_URL` | Legacy gRPC fallback, normally `172.22.174.148:8001` |
| `TRITON_MODEL_NAME` | Triton model name, normally `yolov10n` |
| `KAFKA_BROKERS` | Kafka broker list |
| `KAFKA_GPU_TOPIC` | GPU metrics topic |
| `KAFKA_NET_TOPIC` | Network metrics topic |
| `KAFKA_PHASE_TOPIC` | Phase topic |
| `APP_METRICS_TOPIC` | Client metrics topic |
| `INITIAL_PROCESSING_MODE` | `local` or `remote` before first agent decision |
| `FRAME_INTERVAL_MS` | Frame interval, `100` means 10 fps |
| `DISPLAY_OUTPUT` | OpenCV display window |
| `AUTO_STOP` | Wait for phase cycle and exit automatically |
| `CONFIDENCE_THRESHOLD` | YOLO detection threshold |
| `TARGET_CLASS_ID` | Target class filter, `2,5,7` for this video |
| `TARGET_CONFIDENCE_THRESHOLD` | Target-specific confidence threshold |
| `SP_AGENT_INTERVAL_MS` | How often `decide()` runs |
| `MISS_PENALTY_PX` | Fixed score penalty for missed detections |
| `SP_AGENT_DEBUG_METRICS` | Periodic metrics logging from the agent |
| `DASHBOARD_ENABLED` | Enable client frame publishing to dashboard |
| `DASHBOARD_URL` | Dashboard backend URL from the client |
| `DASHBOARD_FPS` | Dashboard frame publish limit |
| `DASHBOARD_JPEG_QUALITY` | Dashboard JPEG quality |
| `DASHBOARD_FRAME_WIDTH` | Dashboard frame width limit |
| `NETWORK_INTERFACE` | VM3 network interface, normally `ens18` |
| `PHASE_FILE` | VM3 phase file, normally `/tmp/edgelab_phase` |

## Documentation Cleanup

This README is the canonical, deduplicated project guide. Older Markdown files
from previous drafts, roadmaps, per-component READMEs, and testing notes have
been archived under:

```text
docs/archive/markdown-sources/
```

Those files are kept only as historical source material. The generated full
project inventory, if present in the archive, remains ignored because it is a
large code dump and may contain local environment snapshots. Prefer this README
for current setup, operation, and student instructions.
