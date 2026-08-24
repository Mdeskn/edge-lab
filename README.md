# EdgeLab

An edge-computing teaching lab built at FH Dortmund. Students implement a **Service Placement (SP) Agent** that decides, frame by frame, whether to run object-detection inference locally on a Raspberry Pi (CPU, low latency) or offload it to a remote GPU server (higher throughput, network-dependent latency). The system runs real experiments with live network shaping and GPU load injection, then scores each student's strategy automatically.

---

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  Raspberry Pi  (this repo)                                      │
│  ┌──────────┐   ┌────────────┐   ┌────────┐   ┌─────────────┐  │
│  │FrameReader│→ │ Dispatcher │→  │ Scorer │→  │  Dashboard  │  │
│  └──────────┘   └─────┬──────┘   └────────┘   │  Publisher  │  │
│                        │                        └─────────────┘  │
│               ┌────────┴────────┐                                │
│          local│                 │remote                          │
│         ┌─────▼─────┐   ┌──────▼──────┐                        │
│         │ ONNX/CPU  │   │ JPEG Client │                        │
│         └───────────┘   └──────┬──────┘                        │
│                                 │                                │
│         ┌───────────────────────┤                                │
│         │  SP Agent  (student)  │                                │
│         └───────────────────────┘                                │
└────────────────────────────────┬────────────────────────────────┘
                                  │ JPEG frames / predictions
                    ┌─────────────▼───────────────┐
                    │  VM3: Network / Router       │
                    │  tc traffic shaping          │
                    │  Network Conditions → Kafka  │
                    └─────────────┬───────────────┘
                                  │
                    ┌─────────────▼───────────────┐
                    │  VM2: GPU Server             │
                    │  Remote Inference API        │
                    │  Triton Inference Server     │
                    │  GPU Metrics → Kafka         │
                    └─────────────────────────────┘

                    ┌─────────────────────────────┐
                    │  VM1: Infrastructure         │
                    │  Kafka  •  Grafana           │
                    │  SeQaM Scenario Runner       │
                    └─────────────────────────────┘

                    ┌─────────────────────────────┐
                    │  Load VM                     │
                    │  GPU stress client           │
                    └─────────────────────────────┘
```

**This repository contains only the Raspberry Pi application** (client, dashboard, and scenario config). VM-side deployment files live outside this repo.

---

## Repository Layout

```
edge-lab/
├── client/                  # Pi application
│   ├── student/
│   │   └── sp_agent.py      # ← student submission (only file to edit)
│   ├── inference/           # Local ONNX and remote JPEG inference
│   ├── metrics/             # Kafka publishers
│   ├── threads/             # FrameReader, Dispatcher, Scorer threads
│   ├── config.py            # Environment-based config
│   └── main.py
├── dashboard/
│   ├── backend/             # FastAPI + WebSocket + Kafka consumer
│   └── frontend/            # Vanilla JS single-page app (Nginx)
├── data/                    # Runtime data (mounted into containers)
│   ├── yolov10n.onnx        # Detection model
│   ├── test_video.mp4       # Input video
│   └── ground_truth.csv     # Per-frame ground-truth centers
├── seqam/
│   └── ExperimentConfig.json  # Phase timing (read-only mount)
├── docker-compose.yml
└── .env.example
```

---

## Quick Start (Raspberry Pi)

### Prerequisites

- Docker and Docker Compose installed on the Pi
- `data/` directory populated with the model, video, and ground-truth CSV
- The lab infrastructure (Kafka, GPU server) reachable over the network

### 1. Configure the environment

```bash
cp .env.example .env
```

Edit `.env` and set at minimum:

| Variable | What to set |
|---|---|
| `GROUP_ID` | Your assigned group number |
| `KAFKA_BROKERS` | `<VM1-IP>:9092` |
| `REMOTE_INFERENCE_URL` | `http://<VM3-IP>:8100` |

### 2. Start the stack

```bash
docker compose up -d --build
```

Three containers start: `client`, `dashboard-backend`, `dashboard-frontend`.

### 3. Open the dashboard

```
http://<pi-host>:5173
```

### Useful commands

```bash
docker compose ps                       # container status
docker compose logs -f client           # live client logs
docker compose logs -f dashboard-backend
docker compose down                     # stop everything
```

---

## The Student Task

Each group edits **one file only**: `client/student/sp_agent.py`.

The agent runs every `SP_AGENT_INTERVAL_MS` milliseconds and must return either
`"local"` or `"remote"`. `decide()` takes no arguments: metrics are read from
`self`, and the base class keeps them current from Kafka in the background.

Minimal skeleton:

```python
from student.sp_agent_base import SPAgentBase


class SPAgent(SPAgentBase):
    def decide(self) -> str:
        if self.gpu_metrics.get("gpu_util_pct", 0) > 70:
            return "local"
        return "remote"
```

The import has no `client.` prefix. The client runs with its own directory as
the import root, so `from client.student...` fails inside the container even
though it may resolve when you run from the repository root.

Always read metric dictionaries with `.get(key, default)`. They are empty until
the first Kafka message arrives, so indexing raises `KeyError` on the very first
decision.

The most useful signals, with the exact key names:

| Expression | Meaning |
|---|---|
| `self.gpu_metrics.get("gpu_util_pct", 0)` | GPU utilization, 0-100 % |
| `self.gpu_metrics.get("gpu_mem_used_mb", 0)` | GPU memory in use, MB |
| `self.gpu_metrics.get("yolo_queue_ms", 0)` | Time requests wait in the Triton queue, ms |
| `self.gpu_metrics.get("total_pending", 0)` | Requests queued on the GPU server |
| `self.net_metrics.get("delay_ms", 0)` | Added one-way network delay, ms |
| `self.net_metrics.get("jitter_ms", 0)` | Delay variation, ms |
| `self.net_metrics.get("packet_loss_pct", 0)` | Packet loss, 0-100 % |
| `self.avg_latency` | Mean of recent end-to-end latencies, or `None` |
| `self.avg_remote_latency` | Mean of recent remote samples only, or `None` |
| `self.last_remote_probe_latency` | Latest probe of the idle remote backend |
| `self.last_remote_probe_status` | `"ok"`, `"failed"`, or `"unavailable"` |
| `self.current_mode` | `"local"` or `"remote"` right now |
| `self.experiment_phase` | Current phase name |

`yolo_queue_ms` is usually the earliest warning that remote inference is about
to degrade: the queue grows before utilization looks alarming.

The full list, including the Triton and ResNet fields, is in the module
docstring at the top of [`client/student/sp_agent_base.py`](client/student/sp_agent_base.py).

Check a submission without the lab hardware:

```bash
python scripts/check_sp_agent.py
```

It loads the agent and runs `decide()` against synthetic metric snapshots,
including the cold-start case where nothing has arrived from Kafka yet.

Set `SP_AGENT_CLASS=student` in `.env` to activate the submission. Set `SP_AGENT_CLASS=example` to compare against the reference implementation.

---

## Scoring

The score is **cumulative displacement in pixels, and lower is better.** For each
scored frame, the distance is measured between the predicted box centre and
where the target actually is *at the moment the prediction arrives*:

```
displacement = distance(prediction, ground_truth_now)
score        = sum(displacement) over all scored frames
```

Comparing against the target's current position, rather than its position when
the frame was captured, is what makes latency cost score. A correct answer that
arrives 100 ms late is penalized, because the car kept moving while inference
ran. This makes the score a measure of timeliness as much as accuracy, which is
the point of the lab.

A missed detection (no box above the confidence threshold) takes a fixed
`MISS_PENALTY_PX` penalty, 100 px by default.

`LATENCY_DEADLINE_MS` (300 ms by default) does **not** enter the score. Frames
slower than it are counted as deadline misses and reported per phase, as a
diagnostic for students analysing where a strategy struggled.

Some frames are deliberately excluded from the score and flagged in the CSV:

| Column | Excluded because |
|---|---|
| `excluded_warmup_spike` | One outlier right after a phase transition, while a backend absorbs the new load |
| `excluded_video_wrap` | The clip looped while the frame was in flight, so "where the target is now" refers to the start of the clip rather than to real movement |

Both counts are printed at the end of a run. Results are written to
`data/results.csv` and `data/results_by_phase.csv` at the end of each collection
cycle.

---

## Experiment Phases

The SeQaM scenario runner on VM1 drives a repeating cycle of approximately 115 seconds. Network and GPU conditions change per phase:

| Phase | Conditions |
|---|---|
| `cycle_start` | Baseline (clean network, idle GPU) |
| `gpu_load` | Heavy GPU load injected from Load VM |
| `jitter_light` | Light network jitter added |
| `bandwidth_20` | Bandwidth capped at 20 Mbit/s |
| `mixed` | GPU load + bandwidth cap combined |
| `cycle_end` | Results collected, conditions reset |

Phase transitions arrive over Kafka (`edgelab.phase` topic) and are shown live on the dashboard.

---

## Configuration Reference

All settings are read from environment variables (`.env` file). Key variables:

### Identity

| Variable | Default | Description |
|---|---|---|
| `GROUP_ID` | `1` | Student group identifier |

### Inference

| Variable | Default | Description |
|---|---|---|
| `INITIAL_PROCESSING_MODE` | `local` | Starting mode before SP Agent runs |
| `FRAME_INTERVAL_MS` | `100` | Target frame processing interval |
| `CONFIDENCE_THRESHOLD` | `0.25` | YOLO detection confidence cutoff |
| `TARGET_CLASS_ID` | `2,7` | COCO class IDs to track (car, truck) |
| `LATENCY_DEADLINE_MS` | `300` | Deadline for latency penalty |
| `MISS_PENALTY_PX` | `100.0` | Max displacement for zero score |

### Remote Inference

| Variable | Default | Description |
|---|---|---|
| `REMOTE_INFERENCE_URL` | `http://172.22.174.148:8100` | JPEG gateway on the router VM |
| `REMOTE_JPEG_QUALITY` | `80` | JPEG compression for frame transfer |
| `REMOTE_INFERENCE_TIMEOUT_SEC` | `12.0` | Per-frame remote timeout |
| `REMOTE_FALLBACK_TO_LOCAL` | `true` | Fall back to local on remote failure |

### SP Agent

| Variable | Default | Description |
|---|---|---|
| `SP_AGENT_CLASS` | `student` | `student` or `example` |
| `SP_AGENT_INTERVAL_MS` | `500` | Agent decision interval |
| `MANUAL_PLACEMENT_CONTROL` | `false` | Enable the dashboard Local / Remote / Auto buttons. Local and Remote lock placement; Auto returns control to the SP-Agent |
| `LATENCY_PROBES_ENABLED` | `true` | Probe the idle backend at a low rate, so `last_remote_probe_*` is populated for every agent |

### Collection Sync

| Variable | Options | Description |
|---|---|---|
| `SYNC_MODE` | `manual` / `wait_for_cycle` / `off` | When to start collecting results |

`manual`: click **Start Collection** in the dashboard, then wait for the next `cycle_start` message.  
`wait_for_cycle`: collection begins automatically at the next `cycle_start`.  
`off`: collect continuously (development only).

### Kafka Topics

| Variable | Default topic |
|---|---|
| `APP_METRICS_TOPIC` | `dnn_partition.client_metrics` |
| `KAFKA_GPU_TOPIC` | `dnn_partition.server_metrics` |
| `KAFKA_NET_TOPIC` | `edgelab.network.metrics` |
| `KAFKA_PHASE_TOPIC` | `edgelab.phase` |

### Dashboard

| Variable | Default | Description |
|---|---|---|
| `DASHBOARD_PORT` | `8080` | Backend port (internal) |
| `DASHBOARD_FPS` | `10` | Frame preview send rate |
| `DASHBOARD_WEBSOCKET_FPS` | `2` | WebSocket broadcast rate |

---

## Data Directory

The `data/` folder is mounted read-write into all containers. Populate it before starting:

```
data/
├── yolov10n.onnx          # ONNX model (provided by instructor)
├── test_video.mp4         # Input video clip (provided by instructor)
├── ground_truth.csv       # Frame-level ground truth (provided by instructor)
├── results.csv            # Written at end of each collection (auto-created)
└── results_by_phase.csv   # Per-phase breakdown (auto-created)
```

---

## Dashboard Features

| Panel | Description |
|---|---|
| Live video | Annotated frame with predicted box, ground-truth center, displacement line |
| Latency chart | Local and remote inference latency over time |
| Displacement chart | Per-frame pixel error against ground truth |
| Cumulative score | Running total score across the collection cycle |
| Mode timeline | When the SP Agent chose local vs. remote |
| GPU metrics | GPU utilization, memory, temperature from VM2 |
| Network metrics | Delay, jitter, packet loss from the router VM |
| Phase indicator | Current experiment phase and time in cycle |
| Collection controls | Start / stop collection; current collection state |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Object detection | YOLOv10n, ONNX Runtime (CPU), Triton (GPU) |
| Messaging | Apache Kafka (Confluent Python client) |
| Backend | FastAPI, Uvicorn, WebSockets |
| Frontend | Vanilla JavaScript, Nginx |
| Containerization | Docker, Docker Compose |
| Network shaping | Linux `tc` (traffic control) |
| Scenario automation | SeQaM |

---

## License

This project is developed for educational use at FH Dortmund. All rights reserved.
