# Student Guide

This guide covers everything you need to run the experiment, understand the metrics, and improve your score. The root `README.md` has the quick overview; this document has the full detail.

---

## Contents

1. [What you are building](#1-what-you-are-building)
2. [Installation and setup](#2-installation-and-setup)
3. [Running the experiment](#3-running-the-experiment)
4. [The SP-Agent API: all available signals](#4-the-sp-agent-api-all-available-signals)
5. [The four experiment phases](#5-the-four-experiment-phases)
6. [Scoring in detail](#6-scoring-in-detail)
7. [Reading your results](#7-reading-your-results)
8. [Strategy ideas](#8-strategy-ideas)
9. [Environment variables reference](#9-environment-variables-reference)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. What you are building

You implement one method:

```python
# client/student/sp_agent.py
def decide(self) -> str:
    ...
    return "local"   # or "remote"
```

The pipeline calls `decide()` every 500 ms. Your return value becomes the active inference backend for all frames until the next call. Local means YOLO runs on the Pi CPU; remote means the frame is sent to the Triton GPU server over the network.

The base class (`sp_agent_base.py`) runs a background Kafka consumer that subscribes to GPU metrics, network conditions, experiment phase, and your own frame results. All of that is available to you as properties on `self`.

---

## 2. Installation and setup

### Data files

The lab supplies three files. Put them in `data/` at the repo root:

```
data/
  video.mp4          pre-recorded drone-view single-car video
  ground_truth.csv   car coordinates per frame (the correct answers)
  yolov10n.onnx      YOLOv10n model in ONNX format
```

### Python environment

```bash
python3 -m venv .venv-client
source .venv-client/bin/activate
pip install -r client/requirements.txt
```

The requirements are: `onnxruntime`, `opencv-python`, `numpy`, `tritonclient[grpc]`, `confluent-kafka`, `python-dotenv`, and optional OpenTelemetry packages.

### Configuration file

```bash
cp .env.example .env
```

Minimum settings for the student lab experiment:

```dotenv
GROUP_ID=1

VIDEO_PATH=data/video.mp4
GROUND_TRUTH_PATH=data/ground_truth.csv
MODEL_PATH=data/yolov10n.onnx

# Always use the Router VM address, not the GPU server directly.
REMOTE_INFERENCE_URL=http://172.22.174.148:8100
REMOTE_JPEG_QUALITY=80

# Legacy direct Triton fallback, used only when REMOTE_INFERENCE_URL is blank.
TRITON_URL=172.22.174.148:8001

KAFKA_BROKERS=172.22.174.149:9092

TARGET_CLASS_ID=2,5,7
TARGET_CONFIDENCE_THRESHOLD=0.1
DISPLAY_OUTPUT=false
AUTO_STOP=false
```

Set `DISPLAY_OUTPUT=true` if you have a local monitor attached; it will open an OpenCV window with the annotated video. Leave it `false` when running over SSH.

Set `AUTO_STOP=true` for the real graded experiment run. The app will then wait for the first SeQaM phase message before starting, run one full 120-second cycle, and stop automatically.

---

## 3. Running the experiment

```bash
cd client
python main.py
```

On startup, you will see a banner showing all configuration values. The app then starts four threads: FrameReader, Dispatcher, Scorer, and SPAgent.

Press `Ctrl+C` to stop at any time. The app prints a per-phase summary on exit.

### Development workflow

```
1. Edit client/student/sp_agent.py
2. Run python client/main.py
3. Watch the log output (or the OpenCV window if DISPLAY_OUTPUT=true)
4. Press Ctrl+C to stop
5. Open data/results.csv to review per-frame results
6. Repeat
```

### Benchmarking before you write strategy

Before writing your logic, understand how fast each backend actually is on the lab hardware:

```bash
# Test local inference on the Pi
python scripts/benchmark_inference.py --mode local \
    --model data/yolov10n.onnx \
    --video data/video.mp4

# Test remote JPEG inference through the GPU-server API
python scripts/benchmark_inference.py --mode remote \
    --model data/yolov10n.onnx \
    --remote-inference-url http://172.22.174.148:8100

# Compare both
python scripts/benchmark_inference.py --mode both \
    --model data/yolov10n.onnx \
    --video data/video.mp4 \
    --remote-inference-url http://172.22.174.148:8100
```

The output shows mean, min, max, p95, and p99 latencies for both backends and prints a plain-language recommendation. Knowing these numbers tells you roughly when remote is worth it and when it is not.

---

## 4. The SP-Agent API: all available signals

All signals are available inside `decide()` via `self`. The base class keeps them up to date in a background thread.

### `self.experiment_phase` (str)

The current experiment phase. Defaults to `"baseline"` until the first Kafka message arrives.

Valid values from the SeQaM scenario:

| Value | Conditions |
|-------|-----------|
| `"baseline"` | No load; GPU free, network at full speed |
| `"gpu_load"` | GPU server flooded with 100 concurrent requests |
| `"bandwidth_50"` | Bandwidth capped at 50 mbit by the Router VM |
| `"mixed"` | 50 mbit cap AND GPU server flooded simultaneously |

The phase changes before the load is actually visible in metrics, so you can react proactively.

### `self.gpu_metrics` (dict)

Normalized from Eldiyar's server metrics publisher on the GPU server. Empty dict until the first Kafka message arrives. Always use `.get(key, default)`.

```python
gpu_util    = self.gpu_metrics.get("gpu_util_pct", 0)     # GPU utilization 0-100 %
gpu_temp    = self.gpu_metrics.get("gpu_temp_c", 0)       # GPU temperature in Celsius
gpu_mem_mb  = self.gpu_metrics.get("gpu_mem_used_mb", 0)  # GPU memory used in MB
cpu_util    = self.gpu_metrics.get("cpu_util_pct", 0)     # server CPU utilization
power_w     = self.gpu_metrics.get("power_w", 0)          # server power draw in Watts

yolo_queue  = self.gpu_metrics.get("yolo_queue_ms", 0)    # avg time waiting in queue (ms)
yolo_infer  = self.gpu_metrics.get("yolo_infer_ms", 0)    # avg GPU compute time (ms)
yolo_rps    = self.gpu_metrics.get("yolo_success_rps", 0) # successful requests per second

total_rps   = self.gpu_metrics.get("total_rps", 0)        # total requests/s all models
total_pend  = self.gpu_metrics.get("total_pending", 0)    # total queued requests
```

`yolo_queue_ms` is the most useful signal. A rising queue means remote inference will be slow before `gpu_util_pct` even looks alarming.

### `self.net_metrics` (dict)

From the network conditions publisher on the Network VM. Defaults to zeros until the first Kafka message arrives.

```python
delay     = self.net_metrics.get("delay_ms", 0)            # added one-way delay in ms
jitter    = self.net_metrics.get("jitter_ms", 0)           # delay variation in ms
loss      = self.net_metrics.get("packet_loss_pct", 0)     # % of packets dropped
bandwidth = self.net_metrics.get("bandwidth", "unlimited") # "50Mbit", "1Gbit", etc.
mode      = self.net_metrics.get("mode", "clear")          # "clear", "tbf", "netem_tbf"
```

### `self.avg_latency` (float or None)

Mean of the last 20 end-to-end inference latencies in milliseconds, as measured by the Scorer. `None` if fewer than one frame has been processed.

```python
avg_lat = self.avg_latency or 0
```

### `self.recent_latencies` (list[float])

The last 20 latency values that `avg_latency` averages over.

### `self.current_mode` (str)

The processing mode currently active: `"local"` or `"remote"`. Useful to avoid unnecessary switching.

```python
if self.current_mode == "remote" and some_bad_condition:
    return "local"
```

---

## 5. The four experiment phases

The SeQaM platform runs a 120-second loop that cycles through four phases:

```
0 s          30 s         60 s          90 s         120 s
  baseline      gpu_load     bandwidth_50   mixed
  (clean)      (GPU busy)   (50mbit cap)   (both)
     |               |             |              |
     v               v             v              v
  tc cleared     gpu_stressor  tc tbf 50mbit  tc tbf 50mbit
  (no shaping)   starts        applied        + gpu_stressor
```

### What each stressor does

**GPU load (`gpu_load`, `mixed`)**: the GPU stressor fires 100 concurrent YOLOv10n inference requests at the GPU server continuously. The Triton queue fills up. `yolo_queue_ms` rises significantly. Inference still completes but takes longer.

**Network cap (`bandwidth_50`, `mixed`)**: the Network VM applies a token bucket filter (`tbf`) that caps throughput at 50 mbit. Sending a large inference frame and receiving the result becomes slower; the effective round-trip time rises. Under heavy use, multiple concurrent requests compete for the capped bandwidth.

### The `"local_fallback"` mode

If you request `"remote"` but the Triton request fails (timeout or network error), the Dispatcher automatically runs the frame locally and records the result as `"local_fallback"`. You pay the full round-trip time of the failed attempt. This is why reacting early, before fallbacks start happening, improves your score.

---

## 6. Scoring in detail

Every scored frame produces a displacement:

```
displacement_px = sqrt((predicted_x - true_x)^2 + (predicted_y - true_y)^2)
```

Three cases:

| Situation | Score |
|-----------|-------|
| Target not on screen (ground truth absent) | Frame excluded from cumulative total |
| Target present, model returned a detection | Euclidean distance in pixels |
| Target present, model returned no detection | `MISS_PENALTY_PX` (default 100 px) |

The fixed miss penalty matters: without it, a missed detection would be scored as the distance from (0,0) to wherever the target is, typically 1000-1500 px. That swamps the latency signal the lab is designed to measure.

**Final score:**

```
cumulative_displacement = sum of displacement for every scored frame
```

Lower is better.

### Why inference speed affects displacement

The video plays at 10 fps. Each frame has a ground truth target position at the moment it was captured. If inference takes 400 ms (typical local), the result arrives 4 frames late; by then, the target may have moved 20-40 px. Remote inference at 30 ms means the result arrives nearly in sync.

### Missed frames vs. late frames

Both hurt, differently. A frame that produces no detection scores `MISS_PENALTY_PX`. A frame that produces a detection but the target moved while the inference ran scores the actual distance. Under high GPU load or network degradation, `local_fallback` frames take longer than clean local frames (you paid the failed remote attempt first).

---

## 7. Reading your results

### Terminal summary on exit

```
Experiment complete. Results by phase:
------------------------------------------------------------
  baseline          avg displacement:   12.3 px  (300 frames)
  gpu_load          avg displacement:   45.6 px  (300 frames)
  bandwidth_50      avg displacement:   38.9 px  (300 frames)
  mixed             avg displacement:   61.2 px  (300 frames)
------------------------------------------------------------
  overall           avg displacement:   39.50 px  (1200 frames)
  cumulative        1542.3 px  (your score, lower is better)
```

The per-phase rows tell you which conditions your agent handles well.

### The CSV (`data/results.csv`)

One row per scored frame:

| Column | Type | Description |
|--------|------|-------------|
| `timestamp` | float | Unix time when the frame was scored |
| `frame_number` | int | Frame number in the video |
| `group_id` | str | Your `GROUP_ID` |
| `experiment_phase` | str | Phase at the time of scoring |
| `processing_mode` | str | `"local"`, `"remote"`, or `"local_fallback"` |
| `latency_ms` | float | Total time from frame read to result |
| `true_x`, `true_y` | float | Ground truth target center (blank if absent) |
| `predicted_x`, `predicted_y` | float | Detected target center |
| `displacement_px` | float | Distance in pixels (blank if ball absent) |
| `cumulative_displacement_px` | float | Running total |

Open this in a spreadsheet and filter by `processing_mode = "local_fallback"` to see how many remote requests failed and cost you time.

### Live dashboard

If `DASHBOARD_ENABLED=true` and the dashboard backend is running, open a browser at `http://<pi-ip>:5173` to see: live annotated video, current mode and latency, displacement charts, GPU and network signals, and the experiment phase.

See `dashboard/README.md` for dashboard setup.

---

## 8. Strategy ideas

### Start simple

Always return `"remote"` is a surprisingly strong baseline if the GPU server is fast and the network is clean. Measure with the benchmark script first; if remote is 10x faster than local, the baseline phase alone will give you a big advantage.

### Use the phase as a predictive signal

The phase changes before the stressors are fully active. Switching to local as soon as you see `"gpu_load"` avoids the worst of the queue buildup. Waiting for `yolo_queue_ms` to rise means you have already spent several slow frames.

### Mix phase prediction with measured signals

The phase tells you what is coming; the measured signals tell you what is actually happening. A robust agent uses both:

```python
def decide(self) -> str:
    phase = self.experiment_phase

    # Phase-based prediction: return "local" for any stressful phase
    if phase in ("gpu_load", "bandwidth_50", "mixed"):
        return "local"

    # Measurement-based reaction
    if self.net_metrics.get("delay_ms", 0) > 40:
        return "local"
    if self.gpu_metrics.get("gpu_util_pct", 0) > 85:
        return "local"

    return "remote"
```

### Avoid thrashing

Switching every 500 ms adds mode-change overhead and can cause fallbacks mid-transition. Track how many consecutive calls have returned the same decision and only switch after seeing the condition for at least 2-3 calls.

### Use avg_latency as a lagging indicator

`self.avg_latency` reflects the last 20 frames. If it jumps above your measured baseline, something is degraded. This catches anomalies that the phase and individual metrics might miss.

---

## 9. Environment variables reference

Copy `.env.example` to `.env`. All variables are optional except those marked required.

### Identity

| Variable | Default | Description |
|----------|---------|-------------|
| `GROUP_ID` | `1` | Your group number (1-4). Used in your Kafka topic name. |

### File paths (required)

| Variable | Description |
|----------|-------------|
| `VIDEO_PATH` | Path to `video.mp4` |
| `GROUND_TRUTH_PATH` | Path to `ground_truth.csv` |
| `MODEL_PATH` | Path to `yolov10n.onnx` |
| `RESULTS_LOG_PATH` | Output CSV path (default: `results.csv`) |

### Remote inference

| Variable | Default | Description |
|----------|---------|-------------|
| `REMOTE_INFERENCE_URL` | (empty) | Preferred Router VM URL for compressed JPEG inference: `http://172.22.174.148:8100`. Leave blank for local-only or legacy direct Triton mode. |
| `REMOTE_JPEG_QUALITY` | `80` | JPEG quality for remote inference requests. Higher is larger and more accurate; lower is smaller and faster on bandwidth-limited links. |
| `REMOTE_INFERENCE_TIMEOUT_SEC` | `5.0` | Timeout for the remote inference API request. |
| `TRITON_URL` | (empty) | Legacy direct Triton gRPC address: `172.22.174.148:8001`. Used only when `REMOTE_INFERENCE_URL` is blank. |
| `TRITON_MODEL_NAME` | `yolov10n` | Model name in Triton |

### Kafka

| Variable | Default | Description |
|----------|---------|-------------|
| `KAFKA_BROKERS` | (empty) | SeQaM Kafka: `172.22.174.149:9092`. Leave blank to disable all Kafka. |
| `KAFKA_GPU_TOPIC` | `dnn_partition.server_metrics` | GPU metrics topic |
| `KAFKA_NET_TOPIC` | `edgelab.network.metrics` | Network metrics topic |
| `KAFKA_PHASE_TOPIC` | `edgelab.phase` | Experiment phase topic |
| `APP_METRICS_TOPIC` | `dnn_partition.client_metrics` | Your group's result topic |

### Processing behavior

| Variable | Default | Description |
|----------|---------|-------------|
| `INITIAL_PROCESSING_MODE` | `local` | Starting mode before first `decide()` call |
| `FRAME_INTERVAL_MS` | `100` | Time between frames (100 = 10 fps) |
| `DISPLAY_OUTPUT` | `true` | Show OpenCV window; set `false` on SSH/headless |
| `AUTO_STOP` | `false` | Wait for first phase, run one full cycle, then stop |
| `CONFIDENCE_THRESHOLD` | `0.3` | YOLO detection confidence threshold |
| `TARGET_CLASS_ID` | (not set) | Set to `2,6,67,4,0` for the drone-view car video (car + misclassification helpers) |
| `TARGET_CONFIDENCE_THRESHOLD` | `0.1` | Confidence for the target class |
| `SP_AGENT_INTERVAL_MS` | `500` | How often `decide()` is called |
| `MISS_PENALTY_PX` | `100.0` | Score penalty for missed detections |
| `LOG_LEVEL` | `INFO` | Log verbosity: `DEBUG`, `INFO`, `WARNING`, `ERROR` |

### Debug

| Variable | Default | Description |
|----------|---------|-------------|
| `SP_AGENT_DEBUG_METRICS` | `false` | Print a one-line metrics summary every ~2 s |

Set `SP_AGENT_DEBUG_METRICS=true` to confirm that Kafka metrics are arriving. The log line shows phase, GPU util, YOLO queue time, net delay, and bandwidth.

### Dashboard (client side)

| Variable | Default | Description |
|----------|---------|-------------|
| `DASHBOARD_ENABLED` | `false` | Send frames and metrics to the dashboard backend |
| `DASHBOARD_URL` | `http://localhost:8080` | Dashboard backend address |
| `DASHBOARD_FPS` | `5` | Max dashboard frame rate |
| `DASHBOARD_JPEG_QUALITY` | `70` | JPEG quality for dashboard snapshots |
| `DASHBOARD_FRAME_WIDTH` | `960` | Max snapshot width |

---

## 10. Troubleshooting

### App crashes immediately with "FileNotFoundError"

The model or video file path is wrong. Check `MODEL_PATH` and `VIDEO_PATH` in `.env`.

### "Cannot open video"

`VIDEO_PATH` is wrong or the file does not exist.

### Remote inference not reachable

You will see a remote-client connection error in the log. The app continues in local-only mode.

Check that `REMOTE_INFERENCE_URL` is `http://172.22.174.148:8100` (Router VM, not the GPU server directly). Test manually:

```bash
curl http://172.22.174.148:8100/health
```

If you intentionally use the legacy direct Triton path, check `TRITON_URL=172.22.174.148:8001`.

### Kafka not reachable

You will see `"Failed to connect to Kafka"`. The app continues without Kafka; `experiment_phase` stays `"baseline"`, GPU and network metrics stay at their defaults. This is fine for local development without the lab infrastructure.

Test the connection:

```bash
nc -zv 172.22.174.149 9092
```

### No video window despite DISPLAY_OUTPUT=true

You are on SSH without X11 forwarding. Either add `-X` to your SSH command or set `DISPLAY_OUTPUT=false`.

### Very low detection rate

Check `TARGET_CLASS_ID=2,5,7` and `TARGET_CONFIDENCE_THRESHOLD=0.1` are set. Also confirm you are using the lab-supplied `yolov10n.onnx` and the matching `ground_truth.csv`.

### High cumulative displacement

Common causes:

- Returning `"remote"` during heavy GPU or network load produces slow frames or fallbacks.
- Returning `"local"` during `"baseline"` misses the fast GPU window.
- Mode thrashing: switching on every call burns time on transitions.

Look at `data/results.csv`. Filter for `processing_mode = "local_fallback"` to count failed remote attempts. Filter by `experiment_phase` to see which phase costs the most displacement.

### Pi is dropping frames

You will see `"Queue full, dropping frame"` at DEBUG log level. Try:

- `FRAME_INTERVAL_MS=200` (drops to 5 fps)
- `LOG_LEVEL=WARNING` (reduces logging overhead)
- `DISPLAY_OUTPUT=false` (removes OpenCV rendering cost)
