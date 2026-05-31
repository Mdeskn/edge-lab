# Edge Computing Lab

A hands-on lab project for the **IoT and Edge Computing** course. A Raspberry Pi 5 watches a video, tracks a tennis ball, and decides on its own whether to run the detection on its own CPU or send the frame to a powerful GPU server over the network. Your job as a student is to write the brain that makes that decision.

## Live browser dashboard

The optional browser dashboard makes service-placement decisions visible during
the experiment: live annotated video, local/remote mode, latency, displacement,
cumulative score, experiment phase, GPU and network metrics, rolling charts, and
a run summary.

The tennis-ball experiment uses `TARGET_CLASS_ID=32`, the COCO `sports ball`
class. Local Pi inference and remote GPU-server inference ignore people and
other objects. The supplied ground-truth CSV tracks only the tennis ball.

See [dashboard/README.md](dashboard/README.md) for dashboard setup, Pi publishing
configuration, and troubleshooting.

See [dashboard/FEATURES_AND_ARCHITECTURE.md](dashboard/FEATURES_AND_ARCHITECTURE.md)
for a feature-by-feature explanation and the exact implementation data flow.

---

## Table of contents

1. [What this project does](#1-what-this-project-does)
2. [How it works (big picture)](#2-how-it-works-big-picture)
3. [Project structure](#3-project-structure)
4. [The lab machines](#4-the-lab-machines)
5. [The processing pipeline step by step](#5-the-processing-pipeline-step-by-step)
6. [The SP-Agent: the only file you write](#6-the-sp-agent-the-only-file-you-write)
7. [All metrics available to your agent](#7-all-metrics-available-to-your-agent)
8. [The four experiment phases](#8-the-four-experiment-phases)
9. [How scoring works](#9-how-scoring-works)
10. [Environment variables reference](#10-environment-variables-reference)
11. [Installation and setup](#11-installation-and-setup)
12. [Running the experiment](#12-running-the-experiment)
13. [Reading the results](#13-reading-the-results)
14. [Troubleshooting](#14-troubleshooting)
15. [Full file reference](#15-full-file-reference)

---

## 1. What this project does

Imagine you have a small computer (Raspberry Pi 5) watching a video. It needs to locate a tennis ball in every frame, about 10 frames per second. Object detection is a heavy calculation. The Pi's CPU is not very fast, so it takes a while. But there is a powerful GPU server on the same network that can do the same calculation many times faster.

The catch is: sending a frame over the network takes time too. Sometimes the network is slow or lossy. Sometimes the GPU server is already overloaded. So the "right" choice (local or remote) changes constantly.

This project gives you a real system where:

- The Pi reads frames from a pre-recorded video at about 10 fps.
- For each frame, it runs YOLOv10n object detection and keeps only the tennis-ball prediction before calculating its center coordinates (x, y).
- It compares the predicted coordinates against a pre-computed **ground truth** (the correct answer) and measures how far off the prediction was. This distance is called **displacement**.
- It adds up the displacement over the whole experiment. This is the **cumulative displacement**, and it is your score. Lower is better.
- Your SP-Agent decides for every frame: run locally or remotely. A good agent adapts to changing conditions and always picks the faster, more accurate option.

---

## 2. How it works (big picture)

```
+-----------------------------------------------------+
|                  Raspberry Pi 5                     |
|                                                     |
|  Video file  -->  Frame Reader                      |
|                        |                            |
|                   reader_queue                      |
|                        |                            |
|                   Dispatcher  <--  SP-Agent decides |
|                   /        \                        |
|          Local CPU        Remote GPU (Triton)       |
|                   \        /                        |
|                  scorer_queue                       |
|                        |                            |
|                     Scorer                          |
|                   /        \                        |
|           CSV file        Kafka broker              |
+-----------------------------------------------------+
                                 |
              +------------------+------------------+
              |                                     |
   +----------+----------+             +------------+----------+
   |     GPU Server      |             |       Network VM      |
   |                     |             |                       |
   |  Triton Inference   |             |  tc netem rules       |
   |  Server (NVIDIA)    |             |  (adds delay/loss)    |
   |                     |             |                       |
   |  GPU Metrics        |             |  Network Conditions   |
   |  Publisher          |             |  Publisher            |
   +---------------------+             +-----------------------+
              |                                     |
              +------------------+------------------+
                                 |
                        Kafka message bus
                        (SeQaM platform)
```

All the machines communicate through **Kafka**, a message bus hosted on the SeQaM platform. The Pi reads metrics from Kafka (GPU utilization, network conditions, experiment phase) and writes its per-frame results back to Kafka.

---

## 3. Project structure

```
edge-lab/
|
|-- README.md                          <- this file
|-- .env.example                       <- template for your configuration
|
|-- client/                            <- runs on the Raspberry Pi
|   |-- main.py                        <- starts everything, creates all threads
|   |-- config.py                      <- reads all environment variables
|   |-- shared_state.py                <- thread-safe shared memory between threads
|   |-- requirements.txt               <- Python packages needed on the Pi
|   |-- Dockerfile                     <- container definition for the Pi
|   |
|   |-- inference/
|   |   |-- local_server.py            <- runs YOLO on the Pi's CPU
|   |   |-- remote_client.py           <- sends frames to Triton on the GPU server
|   |
|   |-- threads/
|   |   |-- frame_reader.py            <- reads video frames, looks up ground truth
|   |   |-- dispatcher.py              <- preprocesses frames, calls local or remote
|   |   |-- scorer.py                  <- measures displacement, writes CSV, publishes
|   |
|   |-- student/
|   |   |-- sp_agent_base.py           <- base class (do not edit)
|   |   |-- sp_agent.py                <- YOUR FILE: implement decide() here
|   |
|   |-- metrics/
|       |-- kafka_publisher.py         <- publishes per-frame results to Kafka
|       |-- dashboard_publisher.py     <- sends annotated JPEG snapshots to the dashboard
|       |-- telemetry.py               <- OpenTelemetry tracing setup
|
|-- publishers/
|   |-- gpu_metrics/
|   |   |-- gpu_metrics_publisher.py   <- runs on GPU server, polls nvidia-smi + Triton
|   |   |-- requirements.txt
|   |   |-- Dockerfile
|   |
|   |-- network_conditions/
|       |-- network_conditions_publisher.py  <- runs on Network VM, reads tc rules
|       |-- requirements.txt
|       |-- Dockerfile
|
|-- ground_truth/
|   |-- generate_ground_truth.py       <- run once on a fast machine to make the CSV
|   |-- README.md
|
|-- triton/
|   |-- model_repository/
|       |-- yolov10n/
|           |-- config.pbtxt           <- tells Triton how to load the model
|           |-- 1/                     <- model version folder (put model.onnx here)
|
|-- scripts/
|   |-- benchmark_inference.py         <- measure local vs remote latency before the experiment
|   |-- tc_apply.sh                    <- adds network delay/jitter/loss via tc
|   |-- tc_clear.sh                    <- removes all network rules
|   |-- gpu_stressor.sh                <- floods Triton with requests to stress GPU
|   |-- setup_pi.sh                    <- one-time Pi dependency installer
|
|-- seqam/
|   |-- scenario.json                  <- the 4-phase load schedule for SeQaM
|   |-- README.md
|
|-- dashboard/                         <- optional browser dashboard
|   |-- backend/                       <- FastAPI API, Kafka consumer, WebSocket server
|   |-- frontend/                      <- React live visualization
|   |-- docker-compose.yml             <- starts the dashboard host services
|
|-- docker-compose.pi.yml              <- Docker setup for the Pi client
|-- docker-compose.gpu-server.yml      <- Docker setup for Triton + GPU publisher
|-- docker-compose.netvm.yml           <- Docker setup for the network publisher
```

---

## 4. The lab machines

The student lab experiment uses the Pi, remote GPU server, Network VM, and SeQaM
platform together. The browser dashboard runs on a reachable dashboard host.

| Machine | What runs on it | Minimum requirements |
|---------|----------------|----------------------|
| Raspberry Pi 5 | The main client app | Python 3.11, ARM64 |
| GPU Server | Triton Inference Server + GPU metrics publisher | Docker, NVIDIA GPU, nvidia-container-toolkit |
| Network VM | Network conditions publisher, tc netem | Docker, iproute2 (tc) |
| SeQaM platform | Kafka broker, experiment phase controller | Provided by the lab |
| Dashboard Host | FastAPI backend + React frontend | Docker |

For an optional Pi-only smoke test, Kafka and Triton can be left empty. The
course experiment itself uses the Pi and remote GPU server together.

---

## 5. The processing pipeline step by step

Every 100 ms (configurable with `FRAME_INTERVAL_MS`), this is exactly what happens:

### Step 1: FrameReader reads a frame

`client/threads/frame_reader.py`

- Opens the video file with OpenCV.
- Reads the next frame.
- Looks up the ground truth for that frame number from the CSV file (the correct x, y coordinates of the tennis ball, or no coordinates when the ball is not visible).
- Stores the current ground truth in shared state for observers.
- Puts the raw frame, frame number, and matching ground-truth coordinates into `reader_queue` together so delayed results remain aligned with the correct video frame.
- If `DISPLAY_OUTPUT=true`, shows the raw frame in an OpenCV window.
- When the video reaches the end, it loops back to frame 0.

### Step 2: Dispatcher preprocesses and routes the frame

`client/threads/dispatcher.py`

- Picks up a frame from `reader_queue`.
- **Preprocesses** the frame:
  - Resizes it to 640x640 pixels (what YOLO expects).
  - Converts color from BGR to RGB.
  - Normalizes pixel values from 0-255 to 0.0-1.0.
  - Rearranges dimensions from HxWxC to CxHxW (channels first).
  - Adds a batch dimension so the shape becomes (1, 3, 640, 640).
- Checks `shared_state.processing_mode` (set by your SP-Agent).
  - If `"remote"` and the GPU server is reachable: sends the frame to Triton over HTTP.
  - If Triton fails (timeout, error): falls back to local CPU and logs a warning.
  - If `"local"` or Triton not configured: runs YOLO locally on the Pi CPU.
- Records how long the inference took (latency in milliseconds).
- Puts the result (predicted x, y, latency, actual mode used) into `scorer_queue`.

### Step 3: LocalServer runs inference on the Pi CPU

`client/inference/local_server.py`

- Loads the `yolov10n.onnx` model using onnxruntime at startup (fails loud if file missing).
- Uses 4 CPU threads for inference.
- Runs the ONNX model on the preprocessed frame.
- Parses the YOLOv10 output: it is a tensor of shape (num_boxes, 6) where each row is [x1, y1, x2, y2, confidence, class_id].
- Keeps only COCO class `TARGET_CLASS_ID` (`32`, sports ball, for this lab).
- Filters target boxes below `TARGET_CONFIDENCE_THRESHOLD`.
- Takes the remaining ball box with the highest confidence.
- Converts the box from 640x640 space back to the original video resolution.
- Returns the center coordinates (x, y) of that box.

### Step 4: RemoteClient sends a frame to Triton

`client/inference/remote_client.py`

- At startup, performs a health check to `http://[TRITON_URL]/v2/health/live`.
- If unreachable, marks itself as unavailable (no crash, just fallback to local).
- For each inference call, uses `tritonclient.http` to send the preprocessed FP32 tensor as the named `images` input and requests the `output0` tensor.
- Triton runs the ONNX model on the GPU (much faster than CPU).
- Receives the output, parses it the same way as LocalServer.
- Returns the center coordinates (x, y).
- Has a 5-second timeout per request.

### Step 5: Scorer measures displacement and writes results

`client/threads/scorer.py`

- Picks up the result from `scorer_queue`.
- Reads the matching ground truth (x, y) that travelled through the queues with this frame.
- Calculates displacement: the straight-line distance in pixels between the predicted center and the ground truth center.

  ```
  displacement = sqrt((predicted_x - true_x)^2 + (predicted_y - true_y)^2)
  ```

- Adds the displacement to the running total (cumulative displacement).
- If the supplied CSV marks the ball as absent, records `N/A` displacement and does not add that frame to the score.
- If `DISPLAY_OUTPUT=true`, draws an overlay on the frame:
  - Green circle: ground truth position.
  - Red circle: predicted position.
  - Line connecting the two.
  - HUD text showing current mode, latency, per-frame displacement, cumulative displacement.
- Writes one row to the CSV file.
- Publishes the result to Kafka (if configured).

### Step 6: SP-Agent calls decide() every 500 ms

`client/student/sp_agent.py` and `client/student/sp_agent_base.py`

- Runs in its own thread.
- Every `SP_AGENT_INTERVAL_MS` (default 500 ms), calls `decide()`.
- Your `decide()` returns either `"local"` or `"remote"`.
- The base class writes that value into `shared_state.processing_mode`.
- The Dispatcher reads that value before every single frame.

The base class also runs a background Kafka consumer thread that subscribes to four topics and stores the data privately inside the agent:
- `/edgelab/server/metrics`: GPU utilization, memory, temperature, Triton queue stats.
- `/edgelab/network/metrics`: current delay, jitter, packet loss.
- `/edgelab/server/events/phase`: which load phase is currently active. The base class also writes this to SharedState so the Scorer can display and record it.
- `/edgelab/app/metrics/group{N}`: per-frame results published by the Scorer; the agent extracts `latency_ms` from each message to maintain a rolling latency history.

All four are available to your `decide()` as read-only properties on `self`. The agent is entirely self-contained: it does not read any external metrics from shared state. Its only interaction with the rest of the pipeline is writing the processing mode.

---

## 6. The SP-Agent: the only file you write

**File: `client/student/sp_agent.py`**

This is the only file you need to edit. Everything else is infrastructure that you should not touch.

### The default (starter) implementation

```python
class SPAgent(SPAgentBase):

    def __init__(self, shared_state, config):
        super().__init__(shared_state, config)
        # Add your own state here if you need it, for example:
        # self.my_counter = 0

    def decide(self) -> str:
        # Return "local" or "remote"
        return "local"
```

This always returns `"local"`. It will work but it will not be optimal.

### A smarter example

```python
def decide(self) -> str:
    gpu_load = self.gpu_metrics.get("gpu_utilization_pct", 0)
    network_delay = self.net_metrics.get("delay_ms", 0)

    # If GPU is very busy, don't bother sending to it
    if gpu_load > 80:
        return "local"

    # If network delay is very high, it's faster to just run locally
    if network_delay > 50:
        return "local"

    # Otherwise the GPU server is fast and the network is fine
    return "remote"
```

### An even smarter example using the experiment phase

```python
def decide(self) -> str:
    # We know in advance what each phase does
    phase = self.experiment_phase

    if phase == "network_load":
        # Network is bad during this phase, stay local
        return "local"

    if phase == "gpu_load":
        # GPU is overloaded during this phase, stay local
        return "local"

    if phase == "combined":
        # Both are bad, definitely local
        return "local"

    # "baseline" phase: GPU is free and network is clean
    return "remote"
```

You can add any logic you want. You can keep state between calls by using `self`. You can use all the metrics described in the next section.

---

## 7. All metrics available to your agent

Inside `decide()`, you can read the following properties. The base class subscribes to Kafka topics in a background thread and keeps them up to date automatically. You never need to connect to Kafka yourself.

### GPU metrics (`self.gpu_metrics`)

This is a Python `dict`. The keys below are the ones you can safely read:

| Key | Type | What it means |
|-----|------|---------------|
| `"gpu_utilization_pct"` | float (0-100) | How busy the GPU is right now. 0 means idle, 100 means fully saturated. |
| `"gpu_memory_used_mb"` | float | How many MB of GPU memory are currently used. |
| `"gpu_memory_total_mb"` | float | Total GPU memory in MB. |
| `"gpu_temperature_c"` | float | GPU temperature in Celsius. |
| `"triton_requests_per_sec"` | float | How many inference requests Triton is handling per second. |
| `"triton_queue_duration_ms"` | float | Average time a request spends waiting in the Triton queue before processing starts. High value = GPU is overloaded. |
| `"triton_inference_duration_ms"` | float | Average time the GPU actually takes to run inference once it starts. |

**Safe access pattern:**
```python
gpu_load = self.gpu_metrics.get("gpu_utilization_pct", 0)
# The second argument (0) is the default if the key is missing
```

### Network metrics (`self.net_metrics`)

| Key | Type | What it means |
|-----|------|---------------|
| `"delay_ms"` | float | Extra network delay added by the Network VM in milliseconds. 0 means no artificial delay. |
| `"jitter_ms"` | float | Variation in delay in milliseconds. High jitter makes latency unpredictable. |
| `"packet_loss_pct"` | float | Percentage of packets being dropped. 2.0 means 2% of packets never arrive. |

### Latency history (`self.recent_latencies` and `self.avg_latency`)

These come from the `/edgelab/app/metrics/group{N}` Kafka topic. Every time the Scorer finishes a frame it publishes the result, and the agent picks up the `latency_ms` field.

| Property | Type | What it means |
|----------|------|---------------|
| `self.recent_latencies` | `list[float]` | The last 20 end-to-end inference latencies in milliseconds (from frame dequeue to result). |
| `self.avg_latency` | `float` or `None` | The average of those 20 values. `None` if no frames have been processed yet. |

### Experiment phase (`self.experiment_phase`)

| Value | Meaning |
|-------|---------|
| `"baseline"` | Normal conditions. GPU free, no network degradation. |
| `"gpu_load"` | GPU is being flooded with 100 concurrent requests by the stressor. |
| `"network_load"` | Network VM is adding 100ms delay, 20ms jitter, 2% packet loss. |
| `"combined"` | Both GPU and network are stressed simultaneously. |

### Current mode (`self.current_mode`)

The mode that is currently active. Either `"local"` or `"remote"`. Useful if you want to avoid switching too frequently (mode thrashing).

---

## 8. The four experiment phases

The SeQaM platform runs a 120-second loop that cycles through four phases, 30 seconds each. It controls this via `seqam/scenario.json`.

```
0s --------- 30s ---------- 60s ---------- 90s ---------- 120s
  baseline      gpu_load      network_load    combined
  (clean)      (GPU busy)    (net degraded)  (both bad)
     |               |              |              |
     |          gpu_stressor   tc_apply.sh    both active
     |          sends 100      adds 100ms
     |          req/sec to     delay + 20ms
     |          Triton         jitter + 2%
     |                         packet loss
```

The scenario also publishes the phase name to the `/edgelab/server/events/phase` Kafka topic so your agent can react proactively before performance actually degrades.

### Auto-stop behavior

When `AUTO_STOP=true` (the default) and `KAFKA_BROKERS` is set, the app does not start processing immediately. Instead it prints "Waiting for experiment phase to start..." and blocks until a phase message arrives on the phase topic. Once the first phase message arrives, all threads start. The app then monitors phase transitions and stops automatically after observing all four phases and returning to the starting phase (one full 120-second cycle). The final per-phase summary is printed on exit.

For development without a running SeQaM platform, set `AUTO_STOP=false` in your `.env`. The app then starts immediately and runs until Ctrl+C, while still subscribing to Kafka topics and updating metrics if `KAFKA_BROKERS` is set.

### What each stressor does

**GPU stressor (`scripts/gpu_stressor.sh`)**: Uses Triton's `perf_analyzer` tool to fire 100 concurrent inference requests at the GPU server continuously. This saturates the GPU so your legitimate inference requests have to wait in the queue.

**Network stressor (`scripts/tc_apply.sh`)**: Uses Linux `tc` (traffic control) with the `netem` module to add artificial impairments to the network interface. It adds delay, jitter, and packet drops so that sending a frame to the GPU server and getting the result back takes much longer.

---

## 9. How scoring works

### What gets measured

For every frame, the Scorer calculates:

```
displacement = sqrt((predicted_x - true_x)^2 + (predicted_y - true_y)^2)
```

This is simply the Euclidean distance in pixels between where your model said the tennis ball is and where it actually is. Frames where the supplied CSV marks the ball as absent have `N/A` displacement and are excluded from the cumulative score.

The score for the whole experiment is:

```
cumulative_displacement = sum of displacement for all frames
```

### Why does inference quality vary?

- **Local inference** is slower (higher latency) but the result is always computed locally. If the GPU server is overloaded or the network is degraded, remote inference might time out and fall back to local anyway, wasting time.
- **Remote inference** is faster on a free GPU, which means more accurate predictions (because the model gets more time to be precise and the queue is short). But if the GPU queue is long or the network adds 100ms+ of delay, a "remote" request can be slower than just running locally.
- **Fallback**: If you request `"remote"` but the request fails, the Dispatcher automatically falls back to local and marks the result as `"local_fallback"`. You still pay the time cost of the failed remote attempt.

### Per-phase scoring

In addition to the overall cumulative total, displacement is tracked separately for each experiment phase (baseline, gpu_load, network_load, combined). When the experiment ends, the final summary shows the average displacement per phase alongside the overall total. This breakdown helps you understand which phases your agent handles well and which it does not.

If no Kafka phase messages are received (local-only development), all frames are recorded under the "unknown" phase.

### Goal

Write a `decide()` function that picks the fastest option given current conditions. Faster inference = result arrives sooner = result is more in sync with ground truth = lower displacement = better score.

---

## 10. Environment variables reference

Copy `.env.example` to `.env` and fill in the values. Variables marked **required** must be set. Variables with defaults can be omitted.

### Identity

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GROUP_ID` | No | `1` | Your student group number (1 to 4). Used as part of the Kafka topic name for your results. |

### File paths

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `VIDEO_PATH` | Yes | - | Absolute path to the input video file on the Pi. |
| `GROUND_TRUTH_PATH` | Yes | - | Absolute path to the ground_truth.csv file. |
| `MODEL_PATH` | Yes | - | Absolute path to the yolov10n.onnx model file. |
| `RESULTS_LOG_PATH` | No | `results.csv` | Where to write the per-frame results CSV. |

### Remote inference

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `TRITON_URL` | No | `""` | Address of the Triton server, e.g. `192.168.1.100:8000`. Leave blank for local-only mode. |
| `TRITON_MODEL_NAME` | No | `yolov10n` | Name of the model as registered in Triton's model repository. |

### Kafka connection

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `KAFKA_BROKERS` | No | `""` | Kafka broker address, e.g. `192.168.1.200:9092`. Leave blank to disable all Kafka features. |
| `KAFKA_GPU_TOPIC` | No | `/edgelab/server/metrics` | Topic the GPU metrics publisher writes to. |
| `KAFKA_NET_TOPIC` | No | `/edgelab/network/metrics` | Topic the network conditions publisher writes to. |
| `KAFKA_PHASE_TOPIC` | No | `/edgelab/server/events/phase` | Topic the SeQaM platform writes experiment phases to. |

### Processing behavior

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `INITIAL_PROCESSING_MODE` | No | `local` | Starting mode before the SP-Agent makes its first decision. Either `local` or `remote`. |
| `FRAME_INTERVAL_MS` | No | `100` | Time between frames in milliseconds. 100ms = 10 fps. |
| `MODEL_INPUT_WIDTH` | No | `640` | Width the model expects. Do not change unless you use a different model. |
| `MODEL_INPUT_HEIGHT` | No | `640` | Height the model expects. Do not change unless you use a different model. |
| `CONFIDENCE_THRESHOLD` | No | `0.3` | Minimum detection confidence. Lower = more detections but noisier. Higher = fewer detections but more precise. |
| `TARGET_CLASS_ID` | No | `32` in `.env.example` | Optional COCO class filter. Keep `32` for the tennis-ball experiment so people and other objects are ignored. Unset it to disable class filtering. |
| `TARGET_CONFIDENCE_THRESHOLD` | No | `0.1` in `.env.example` | Minimum confidence for the selected target class. If omitted, uses `CONFIDENCE_THRESHOLD`. |
| `SP_AGENT_INTERVAL_MS` | No | `500` | How often `decide()` is called, in milliseconds. |
| `QUEUE_MAX_SIZE` | No | `10` | Maximum number of frames waiting in each internal queue. Frames are dropped if the queue is full. |
| `DISPLAY_OUTPUT` | No | `true` | Show OpenCV windows with the overlay. Set `false` for headless (SSH or Docker without X11). |
| `LOG_LEVEL` | No | `INFO` | Logging verbosity. Options: `DEBUG`, `INFO`, `WARNING`, `ERROR`. |
| `AUTO_STOP` | No | `true` | When `true` and `KAFKA_BROKERS` is set: wait for first phase message, run one full 120s cycle, then stop automatically. Set `false` for development (runs until Ctrl+C). |

### OpenTelemetry tracing

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `OTLP_ENDPOINT` | No | `""` | gRPC endpoint for trace export, e.g. `http://192.168.1.200:4317`. Leave blank to disable tracing. |

### Browser dashboard

Pi client variables:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `DASHBOARD_ENABLED` | No | `false` | Send annotated JPEG snapshots and frame metrics to the dashboard backend. |
| `DASHBOARD_URL` | No | `http://localhost:8080` | Dashboard backend URL reachable from the Pi. Use `http://<dashboard-host-ip>:8080` in the lab. |
| `DASHBOARD_FPS` | No | `5` | Maximum dashboard image updates per second. |
| `DASHBOARD_JPEG_QUALITY` | No | `70` | JPEG compression quality for dashboard snapshots. |
| `DASHBOARD_FRAME_WIDTH` | No | `960` | Maximum JPEG width. Smaller values reduce Pi and network overhead. |

Dashboard host variables:

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `DASHBOARD_HOST` | No | `0.0.0.0` | Backend listen address. |
| `DASHBOARD_PORT` | No | `8080` | Backend listen port. |
| `DASHBOARD_MAX_HISTORY` | No | `300` | Rolling metric samples kept in memory. |
| `DASHBOARD_PUBLIC_API_URL` | No | `http://localhost:8080` | Backend URL used by students' browsers. Set to `http://<dashboard-host-ip>:8080` before starting the dashboard compose stack. |
| `APP_METRICS_TOPICS` | No | all four group topics | Kafka app-metric topics consumed by the dashboard backend. |

### GPU metrics publisher (runs on GPU server, not the Pi)

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `TRITON_METRICS_URL` | No | `http://localhost:8002/metrics` | Prometheus endpoint exposed by Triton. |
| `NVIDIA_SMI_PATH` | No | `/usr/bin/nvidia-smi` | Full path to the nvidia-smi binary. |
| `POLL_INTERVAL_SEC` | No | `1` | How often to poll nvidia-smi and Triton, in seconds. |

### Network conditions publisher (runs on Network VM, not the Pi)

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `NETWORK_INTERFACE` | No | `eth0` | The network interface to read tc rules from. |

---

## 11. Installation and setup

For the course exercise, follow
[Option C: Student lab deployment](#option-c-student-lab-deployment-with-pi-and-remote-gpu-server).
Options A and B are optional Pi smoke tests.

### Option A: Optional Pi-only smoke test

Use this only to verify the Pi installation before connecting the lab
infrastructure. It is not the student experiment: the real placement exercise
requires both the Pi and remote GPU server.

**Step 1: Get the code onto the Pi**

```bash
git clone <repo-url> edge-lab
cd edge-lab
```

**Step 2: Prepare the data files**

Use the three files supplied for the lab. Put them in a `data/` folder on the Pi:

- `video.mp4`: the pre-recorded video file provided by the lab.
- `ground_truth.csv`: the supplied ball-only answer key for that video.
- `yolov10n.onnx`: the supplied ONNX model.

**Step 3: Create the `.env` file on the Pi**

```bash
cp .env.example .env
```

Open `.env` and set at minimum these three variables:

```
VIDEO_PATH=/home/pi/edge-lab/data/video.mp4
GROUND_TRUTH_PATH=/home/pi/edge-lab/data/ground_truth.csv
MODEL_PATH=/home/pi/edge-lab/data/yolov10n.onnx
```

Leave `TRITON_URL` and `KAFKA_BROKERS` empty (or just do not set them).

**Step 4: Install Python dependencies on the Pi**

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r client/requirements.txt
```

Alternatively, use the provided setup script:

```bash
chmod +x scripts/setup_pi.sh
./scripts/setup_pi.sh
source /opt/edge-lab-venv/bin/activate
```

**Step 5: Run the app**

```bash
cd client
python main.py
```

You should see an OpenCV window with the video playing and an overlay showing detections. Press `q` to quit.

---

### Option B: Run with Docker on the Pi

**Step 1: Build and run**

```bash
# Make sure .env is filled in (same as Option A Step 3)
docker compose -f docker-compose.pi.yml up
```

The Docker container mounts the `./data/` folder inside the container. Make sure your `.env` uses `/data/video.mp4`, `/data/ground_truth.csv`, etc. (the container path, not the host path).

---

### Option C: Student lab deployment with Pi and remote GPU server

This is the student experiment. The SP-Agent decides whether each video frame is
processed on the Pi CPU or sent over the network to the remote Triton GPU server.

#### On the GPU server

**Step 1: Copy the model**

```bash
cp yolov10n.onnx triton/model_repository/yolov10n/1/model.onnx
```

**Step 2: Configure `.env`**

```bash
cp .env.example .env
```

Set these variables:

```
KAFKA_BROKERS=<seqam-ip>:9092
TRITON_METRICS_URL=http://localhost:8002/metrics
NVIDIA_SMI_PATH=/usr/bin/nvidia-smi
```

**Step 3: Start Triton and the GPU metrics publisher**

```bash
docker compose -f docker-compose.gpu-server.yml up -d
```

This starts:
- **Triton Inference Server** on ports 8000 (HTTP), 8001 (gRPC), 8002 (Prometheus metrics).
- **GPU metrics publisher** which polls `nvidia-smi` and Triton every second and publishes to the `/edgelab/server/metrics` Kafka topic.

**Verify Triton is running:**

```bash
curl http://localhost:8000/v2/health/live
# Should return HTTP 200
```

---

#### On the Network VM

**Step 1: Configure `.env`**

```bash
cp .env.example .env
```

Set these variables:

```
KAFKA_BROKERS=<seqam-ip>:9092
NETWORK_INTERFACE=eth0   # or whatever interface connects to the Pi
```

**Step 2: Start the network conditions publisher**

```bash
docker compose -f docker-compose.netvm.yml up -d
```

This starts the **network conditions publisher** which reads the current `tc netem` rules every 2 seconds and publishes them to the `/edgelab/network/metrics` Kafka topic.

---

#### On the Raspberry Pi

**Step 1: Prepare the supplied data files**

```bash
mkdir -p data
cp /path/to/video.mp4        data/
cp /path/to/ground_truth.csv data/
cp /path/to/yolov10n.onnx    data/
```

**Step 2: Configure `.env`**

```bash
cp .env.example .env
```

Set all variables including:

```
GROUP_ID=1
VIDEO_PATH=/data/video.mp4
GROUND_TRUTH_PATH=/data/ground_truth.csv
MODEL_PATH=/data/yolov10n.onnx
TRITON_URL=<gpu-server-ip>:8000
KAFKA_BROKERS=<seqam-ip>:9092
DISPLAY_OUTPUT=false   # running headless
DASHBOARD_ENABLED=true
DASHBOARD_URL=http://<dashboard-host-ip>:8080
TARGET_CLASS_ID=32
TARGET_CONFIDENCE_THRESHOLD=0.1
```

**Step 3: Start the client**

```bash
docker compose -f docker-compose.pi.yml up
```

---

#### On the Dashboard Host

**Step 1: Configure `.env`**

```bash
cp .env.example .env
```

Set the Kafka broker and dashboard settings:

```dotenv
KAFKA_BROKERS=<seqam-ip>:9092
DASHBOARD_HOST=0.0.0.0
DASHBOARD_PORT=8080
DASHBOARD_MAX_HISTORY=300
DASHBOARD_PUBLIC_API_URL=http://<dashboard-host-ip>:8080
```

**Step 2: Start the dashboard**

```bash
docker compose -f dashboard/docker-compose.yml up --build -d
```

Open the dashboard in a browser:

```text
http://<dashboard-host-ip>:5173
```

The dashboard shows the annotated tennis-ball video, actual `LOCAL` or `REMOTE`
processing mode, latency, displacement score, experiment phase, GPU metrics,
and network conditions.

---

#### Benchmark inference latency (do this before starting the experiment)

Before writing your SP-Agent strategy, you need to know how fast each backend actually is on your specific hardware. The script has three modes. Run them in order as you bring up each machine.

```bash
# Step 1: test the Pi alone (no GPU server needed yet)
python scripts/benchmark_inference.py --mode local \
    --model /data/yolov10n.onnx \
    --video /data/video.mp4

# Step 2: test the GPU server alone (once Triton is up)
python scripts/benchmark_inference.py --mode remote \
    --model /data/yolov10n.onnx \
    --triton-url <gpu-server-ip>:8000

# Step 3: compare both and get a recommendation
python scripts/benchmark_inference.py --mode both \
    --model /data/yolov10n.onnx \
    --video /data/video.mp4 \
    --triton-url <gpu-server-ip>:8000 \
    --runs 100 --warmup 10
```

The `--mode local` result also tells you whether the Pi is fast enough to make the lab interesting. If it is too fast, pass `--threads 1` to slow it down. The `--mode both` result prints mean/min/max/p95/p99 for both backends and gives a plain-language recommendation. For example, if local takes 400ms and remote takes 30ms, "always remote" is already a very strong baseline and your agent only needs to fall back to local when the GPU or network is under severe stress.

---

#### On the SeQaM platform

Upload `seqam/scenario.json` to the SeQaM web interface and start the experiment. SeQaM will:
- Publish experiment phases to Kafka every 30 seconds.
- Trigger the GPU stressor and tc scripts via SSH commands to the GPU server and Network VM.

---

## 12. Running the experiment

### Development workflow

1. Run `scripts/benchmark_inference.py` once to understand baseline local vs remote latency on your hardware.
2. Edit `client/student/sp_agent.py` based on what you learned.
3. Run `python client/main.py`.
4. Watch the overlay or the log output.
5. Press `q` or Ctrl+C to stop.
6. Read your results from `results.csv`.
7. Repeat.

### Tips

- Set `LOG_LEVEL=DEBUG` to see every decision your agent makes and which backend was actually used.
- Set `DISPLAY_OUTPUT=false` when running over SSH without X11 forwarding. Without this, the app will crash when it tries to open an OpenCV window.
- Set `FRAME_INTERVAL_MS=200` to slow down to 5 fps if the Pi is struggling to keep up.
- The app loops the video automatically so you can let it run as long as you want.

### Forcing network conditions for local testing

If you have the Network VM set up, you can manually apply and remove network degradation:

```bash
# Add 100ms delay, 20ms jitter, 2% packet loss on eth0
./scripts/tc_apply.sh eth0 100 20 2

# Remove all conditions
./scripts/tc_clear.sh eth0
```

These scripts also send a signal to the network conditions publisher so Kafka gets updated immediately.

---

## 13. Reading the results

### The CSV file

Written to `RESULTS_LOG_PATH` (default: `results.csv`). One row per processed frame.

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | float | Unix timestamp when the frame was scored. |
| `frame_number` | int | Frame number from the video (1-indexed). |
| `group_id` | int | Your GROUP_ID setting. |
| `experiment_phase` | string | Current phase name: `"baseline"`, `"gpu_load"`, `"network_load"`, `"combined"`, or `"unknown"` before the first phase message. |
| `processing_mode` | string | `"local"`, `"remote"`, or `"local_fallback"` (remote requested but failed). |
| `latency_ms` | float | Total time from frame dequeue to result, in milliseconds. |
| `true_x` | float or blank | Ground truth X coordinate (pixels), or blank when the ball is absent. |
| `true_y` | float or blank | Ground truth Y coordinate (pixels), or blank when the ball is absent. |
| `predicted_x` | float | Model-predicted X coordinate (pixels). |
| `predicted_y` | float | Model-predicted Y coordinate (pixels). |
| `displacement_px` | float or blank | Distance between prediction and ground truth for this frame, or blank when the ball is absent. |
| `cumulative_displacement_px` | float | Running total of all displacements so far. |

### The final summary

When the experiment ends (auto-stop after one full cycle, or Ctrl+C), the app prints a per-phase breakdown followed by the overall totals:

```
Experiment complete. Results by phase:
------------------------------------------------------------
  baseline          avg displacement:   12.3 px  (300 frames)
  gpu_load          avg displacement:   45.6 px  (300 frames)
  network_load      avg displacement:   38.9 px  (300 frames)
  combined          avg displacement:   61.2 px  (300 frames)
------------------------------------------------------------
  overall           avg displacement:   39.50 px  (1200 frames)
  cumulative        1542.3 px  (your score, lower is better)
```

`cumulative` is your score. Lower is better. The per-phase averages show which conditions your agent handled well and which it struggled with.

### The on-screen overlay

If `DISPLAY_OUTPUT=true`, you will see:

- **Green circle**: the ground truth position.
- **Red circle**: the model's prediction.
- **Line**: the distance between them (the displacement for this frame).
- **HUD text** in the top-left corner: current mode (local/remote), current experiment phase, rolling-average latency (last 5 frames), per-frame displacement, cumulative displacement, total frame count.

---

## 14. Troubleshooting

### App crashes immediately with "FileNotFoundError"

The model file is missing or the path in `.env` is wrong.

Check: Does the file actually exist at the path you set in `MODEL_PATH`?

```bash
ls -la /path/to/yolov10n.onnx
```

### "Cannot open video" or "Video file not found"

Same issue but for the video file. Check `VIDEO_PATH` in `.env`.

### Triton not reachable

If you see warnings like `"Triton health check failed"`, the app will still run in local-only mode. To fix the remote connection:

1. Check that `TRITON_URL` is set to `<gpu-server-ip>:8000` (not 8001 or 8002).
2. Test it manually from the Pi:
   ```bash
   curl http://<gpu-server-ip>:8000/v2/health/live
   ```
3. Make sure port 8000 is open in any firewalls between the Pi and the GPU server.
4. Make sure Triton is actually running on the GPU server (`docker compose ps`).

### Kafka not reachable

You will see warnings like `"Failed to connect to Kafka"`. The app continues without Kafka. To fix:

1. Check that `KAFKA_BROKERS` is set correctly.
2. Test the port:
   ```bash
   nc -zv <kafka-broker-ip> 9092
   ```
3. Your agent's metrics (`gpu_metrics`, `net_metrics`, `experiment_phase`) will all be empty or default values. This is fine for local development.

### No video window appears

If you set `DISPLAY_OUTPUT=true` but nothing appears:

1. You are probably on SSH without X11 forwarding. Either:
   - Add `-X` to your SSH command: `ssh -X pi@<ip>`
   - Or set `DISPLAY_OUTPUT=false` in `.env`.

### Very low detection rate (too many missed frames)

1. Keep `TARGET_CLASS_ID=32` and try lowering `TARGET_CONFIDENCE_THRESHOLD` in `.env`.
2. Make sure `MODEL_INPUT_WIDTH` and `MODEL_INPUT_HEIGHT` are both `640`.
3. Make sure the supplied `yolov10n.onnx` model and ball-only `ground_truth.csv` are in the configured paths.

### High cumulative displacement

This is what you are trying to improve. Common causes:

1. Your agent always returns `"local"` during `"baseline"` phase when the GPU would be faster. Try returning `"remote"` when conditions are good.
2. Your agent returns `"remote"` during `"network_load"` or `"gpu_load"` phase, causing timeouts and fallbacks. Use `self.experiment_phase` to anticipate bad conditions.
3. The remote request times out and falls back to local anyway (you can see `"local_fallback"` in the CSV). This means you wasted time on a failed remote attempt. Consider switching to local earlier.

### Pi is very slow, frames are being dropped

You will see `"Queue full, dropping frame"` in the logs at DEBUG level. Try:

1. Set `FRAME_INTERVAL_MS=200` to reduce throughput to 5 fps.
2. Set `LOG_LEVEL=WARNING` to reduce logging overhead.
3. Make sure you are using Python 3.11 (not an older version). Older Python is slower.

---

## 15. Full file reference

### Client (runs on the Raspberry Pi)

| File | Purpose |
|------|---------|
| `client/main.py` | Entry point. Reads config, creates shared state, starts all four threads (FrameReader, Dispatcher, Scorer, SPAgent). When AUTO_STOP is enabled, waits for the first phase message before starting threads, then monitors phases and stops after one full cycle. Prints per-phase and overall summary on exit. |
| `client/config.py` | Reads all environment variables into a `Config` dataclass. Every other file reads config from here, never from `os.environ` directly. |
| `client/shared_state.py` | Thread-safe object holding shared data between the pipeline threads: current processing mode, ground truth coordinates, recent latency history, cumulative displacement, current experiment phase, per-phase displacement scores, and the shutdown event. Uses one lock for all access. |
| `client/inference/local_server.py` | Wraps the ONNX model for CPU inference. Loads model on startup. `infer(frame, shape)` returns (x, y). Raises `FileNotFoundError` if model is missing. |
| `client/inference/remote_client.py` | HTTP client for Triton. Performs health check on startup. `infer(frame, shape)` sends HTTP request to Triton and returns (x, y). Sets itself unavailable if health check fails. |
| `client/threads/frame_reader.py` | Reads video frames at `FRAME_INTERVAL_MS` rate. Loads ground truth CSV at startup. Puts each frame, frame number, matching ground truth, and enqueue timestamp into `reader_queue`. |
| `client/threads/dispatcher.py` | Picks up frames from `reader_queue`. Preprocesses (resize, normalize, transpose). Routes to local or remote based on `shared_state.processing_mode`. Records latency. Puts result into `scorer_queue`. |
| `client/threads/scorer.py` | Picks up results from `scorer_queue`. Calculates displacement. Draws overlay if display is on. Writes CSV row. Publishes to Kafka. |
| `client/student/sp_agent_base.py` | Base class for the SP-Agent. Subscribes to all four Kafka topics (/edgelab/server/metrics, /edgelab/network/metrics, /edgelab/server/events/phase, /edgelab/app/metrics/groupN) and stores the data in private fields inside the agent. Exposes `gpu_metrics`, `net_metrics`, `recent_latencies`, `avg_latency`, `experiment_phase`, `current_mode` as read-only properties. Calls `decide()` on interval and writes the result to the pipeline via `set_mode()`. Do not edit this file. |
| `client/student/sp_agent.py` | **The only file you write.** Extend `SPAgentBase` and implement `decide() -> str`. Return `"local"` or `"remote"`. |
| `client/metrics/kafka_publisher.py` | Publishes per-frame results to Kafka topic `/edgelab/app/metrics/group{N}`. Silently disabled if `KAFKA_BROKERS` is empty. |
| `client/metrics/dashboard_publisher.py` | Best-effort dashboard publisher. Sends throttled annotated JPEG snapshots from the Pi on a background thread and drops stale snapshots instead of slowing inference. |
| `client/metrics/telemetry.py` | Sets up OpenTelemetry tracing. Returns a no-op tracer if `OTLP_ENDPOINT` is empty. |

### Publishers (run on other machines)

| File | Purpose |
|------|---------|
| `publishers/gpu_metrics/gpu_metrics_publisher.py` | Runs on GPU server. Polls `nvidia-smi` for GPU utilization/memory/temperature. Polls Triton's Prometheus endpoint for queue and inference timing. Publishes to `/edgelab/server/metrics` Kafka topic every second (override with `KAFKA_GPU_TOPIC`). |
| `publishers/network_conditions/network_conditions_publisher.py` | Runs on Network VM. Parses `tc qdisc show` output to read current netem rules (delay, jitter, loss). Publishes to `/edgelab/network/metrics` Kafka topic every 2 seconds (override with `KAFKA_NET_TOPIC`). Also publishes immediately on SIGUSR1 signal (sent by tc_apply.sh and tc_clear.sh). |

### Ground truth generation

| File | Purpose |
|------|---------|
| `ground_truth/generate_ground_truth.py` | Offline tool. Run once on a fast machine. Writes the selected target position for each video frame to a CSV. The tennis-ball mode uses color tracking so its CSV is independent of measured YOLO inference. |

### Infrastructure and scripts

| File | Purpose |
|------|---------|
| `triton/model_repository/yolov10n/config.pbtxt` | Triton model configuration. Defines input shape (1, 3, 640, 640) and output shape (-1, 6). Sets dynamic batching. Copy `yolov10n.onnx` to the `1/` folder as `model.onnx`. |
| `seqam/scenario.json` | Experiment scenario for the SeQaM platform. Defines the 4-phase loop: baseline, gpu_load, network_load, combined. Each phase is 30 seconds. |
| `scripts/tc_apply.sh` | Applies netem traffic control rules. Usage: `./tc_apply.sh [interface] [delay_ms] [jitter_ms] [loss_pct]`. |
| `scripts/tc_clear.sh` | Removes all tc rules. Usage: `./tc_clear.sh [interface]`. |
| `scripts/benchmark_inference.py` | Measures local CPU inference latency vs remote Triton latency. Run once before the experiment to calibrate your SP-Agent strategy. Prints mean/min/max/p95/p99 for both backends and a recommendation. |
| `scripts/gpu_stressor.sh` | Stresses the GPU with many concurrent Triton requests. Usage: `./gpu_stressor.sh [model_name] [concurrency] [duration_seconds]`. |
| `scripts/setup_pi.sh` | One-time setup script for the Pi. Installs Python 3.11, OpenCV, and all Python dependencies into `/opt/edge-lab-venv/`. |

### Docker compose files

| File | Machine | What it starts |
|------|---------|---------------|
| `docker-compose.pi.yml` | Raspberry Pi | The client app. Mounts `./data` to `/data` inside the container. |
| `docker-compose.gpu-server.yml` | GPU Server | Triton Inference Server (ports 8000/8001/8002) + GPU metrics publisher. |
| `docker-compose.netvm.yml` | Network VM | Network conditions publisher. |
| `dashboard/docker-compose.yml` | Dashboard Host | FastAPI backend and React frontend. |
