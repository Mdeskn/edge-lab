# EdgeLab

A hands-on edge-computing lab for the IoT and Edge Computing course at FH Dortmund. A Raspberry Pi watches a pre-recorded video, tracks a tennis ball frame by frame, and has to decide for every frame: run YOLO locally on the Pi CPU, or ship the frame over the network to a remote GPU server?

Your job as a student: write the placement logic (the Service Placement Agent) that adapts to live conditions and keeps the prediction close to the ground truth. Everything else is already wired up.

---

## What is the goal?

Minimize **cumulative displacement**: the total distance (in pixels) between where your detector predicted the ball and where it actually was, summed over every scored frame. Faster inference means the result arrives before the ball has moved far, which means lower displacement and a better score.

---

## System architecture

```
Raspberry Pi app
   |
   +-- LOCAL: YOLOv10n on Pi CPU (onnxruntime)
   |
   +-- REMOTE: 172.22.174.148:8001 (Network/Router VM)
                       |
                       v
               172.22.174.145:8001 (Triton GPU server)

Metrics and phase data:

  GPU server  (172.22.174.145)  ---> Kafka  172.22.174.149:9092
  Network VM  (172.22.174.148)  ---> Kafka  172.22.174.149:9092
  Pi client   (your Raspberry Pi) -> Kafka  172.22.174.149:9092
```

The Pi reads GPU utilization, network conditions, and experiment phase from Kafka. It writes its per-frame results back to Kafka. The live dashboard reads all of that.

**Important:** the Pi must connect via the Router VM (`172.22.174.148:8001`), not directly to the GPU server. The router is the node that applies network impairments during experiments.

---

## Key files

| File | What it is |
|------|-----------|
| `client/student/sp_agent.py` | The only file you edit |
| `client/student/sp_agent_base.py` | Base class that provides live metrics; read but do not edit |
| `client/main.py` | Starts the full pipeline |
| `client/config.py` | All configuration loaded from `.env` |
| `publishers/network_conditions/` | Scripts that run on the Network VM |
| `dashboard/` | Live browser dashboard (FastAPI backend + React frontend) |

---

## Quick start for students

**Step 1: Clone and prepare data**

```bash
git clone <repo-url> edge-lab
cd edge-lab
mkdir -p data
# Copy the three lab-supplied files into data/:
#   video.mp4, ground_truth.csv, yolov10n.onnx
```

**Step 2: Configure `.env`**

```bash
cp .env.example .env
```

Open `.env` and fill in at minimum:

```dotenv
GROUP_ID=1

VIDEO_PATH=data/video.mp4
GROUND_TRUTH_PATH=data/ground_truth.csv
MODEL_PATH=data/yolov10n.onnx

# Router VM forwards to the GPU server; always use this address, not the GPU server directly
TRITON_URL=172.22.174.148:8001

KAFKA_BROKERS=172.22.174.149:9092

DISPLAY_OUTPUT=false   # set true if you have a local display
AUTO_STOP=false        # set true during the real experiment run
```

**Step 3: Install dependencies**

```bash
python3 -m venv .venv-client
source .venv-client/bin/activate
pip install -r client/requirements.txt
```

**Step 4: Run the app**

```bash
cd client
python main.py
```

**Step 5: Edit your agent and iterate**

Open `client/student/sp_agent.py` in your editor. Implement `decide()`. Restart the app. Read `data/results.csv` to see how you did.

---

## The file you edit: `client/student/sp_agent.py`

Implement `decide()` to return `"local"` or `"remote"`. The base class calls it every 500 ms and applies the result.

```python
def decide(self) -> str:
    # Read any signal from self, then return "local" or "remote"
    return "local"
```

You can use `self` to store state between calls. You can call any method on `self`. The only constraint: return one of the two strings.

---

## Signals available in decide()

All of these are kept up to date automatically by the background Kafka consumer in the base class. Use `.get(key, default)` for dict values so you get a safe zero when the first message has not arrived yet.

### Experiment phase

```python
phase = self.experiment_phase   # str, defaults to "baseline"
```

| Phase | What is happening |
|-------|------------------|
| `"baseline"` | Clean conditions: GPU free, no network impairment |
| `"gpu_load"` | GPU server is flooded with 100 concurrent requests |
| `"bandwidth_50"` | Network path capped at 50 Mbit/s by tc |
| `"mixed"` | Both GPU server flooded and 50 Mbit/s cap active |

### GPU server metrics (`self.gpu_metrics`)

```python
gpu_util   = self.gpu_metrics.get("gpu_util_pct", 0)    # GPU busy 0-100 %
gpu_temp   = self.gpu_metrics.get("gpu_temp_c", 0)      # temperature in Celsius
yolo_queue = self.gpu_metrics.get("yolo_queue_ms", 0)   # time waiting in Triton queue
yolo_infer = self.gpu_metrics.get("yolo_infer_ms", 0)   # GPU compute time
total_rps  = self.gpu_metrics.get("total_rps", 0)       # total requests per second
mem_used   = self.gpu_metrics.get("gpu_mem_used_mb", 0) # GPU memory used in MB
```

High `yolo_queue_ms` is the strongest signal that the GPU server is overloaded and remote inference will be slow.

### Network metrics (`self.net_metrics`)

```python
delay     = self.net_metrics.get("delay_ms", 0)           # one-way delay added in ms
jitter    = self.net_metrics.get("jitter_ms", 0)          # delay variation in ms
loss      = self.net_metrics.get("packet_loss_pct", 0)    # percentage of packets dropped
bandwidth = self.net_metrics.get("bandwidth", "unlimited") # e.g. "50Mbit" or "unlimited"
```

### Own latency history

```python
avg_lat  = self.avg_latency        # float or None; mean of last 20 frames
all_lats = self.recent_latencies   # list[float]; last 20 end-to-end times in ms
```

### Current mode

```python
mode = self.current_mode   # "local" or "remote"; what is active right now
```

---

## Example strategy

```python
def decide(self) -> str:
    phase = self.experiment_phase

    # React to known bad conditions immediately
    if phase in ("gpu_load", "bandwidth_50", "mixed"):
        return "local"

    # React to measured bad conditions
    yolo_queue = self.gpu_metrics.get("yolo_queue_ms", 0)
    gpu_util   = self.gpu_metrics.get("gpu_util_pct", 0)
    net_delay  = self.net_metrics.get("delay_ms", 0)

    if yolo_queue > 50 or gpu_util > 80:
        return "local"
    if net_delay > 30:
        return "local"

    # Conditions look good: use the fast GPU
    return "remote"
```

A stronger agent would also react to `self.avg_latency` rising, avoid switching too rapidly, and use the phase as a predictive signal (the phase changes before performance actually degrades).

---

## Scoring

Each scored frame produces a displacement in pixels:

```
displacement = sqrt((predicted_x - true_x)^2 + (predicted_y - true_y)^2)
```

Frames where the ball is not on screen are skipped. Frames where the model returns no detection score a fixed `MISS_PENALTY_PX` (default 100 px) instead of distance-from-origin.

Your final score is the sum of all per-frame displacements. **Lower is better.**

The per-frame CSV (`data/results.csv`) and the end-of-run terminal summary both show a breakdown by phase so you can see where your agent is performing well and where it is not.

---

## Documentation index

| Document | Audience | What it covers |
|----------|----------|---------------|
| `docs/student-guide.md` | Students | Full setup, all metrics, env vars, scoring, troubleshooting |
| `docs/operator-guide.md` | Tutors | Day-of checklist, VM startup order, reset between groups |
| `docs/vm1-kafka-grafana.md` | Operators | SeQaM/Kafka/Grafana VM (172.22.174.149) |
| `docs/vm2-gpu-server.md` | Operators | GPU server and Triton (172.22.174.145) |
| `docs/vm3-network-vm.md` | Operators | Network VM and tc scripts (172.22.174.148) |
| `docs/seqam-integration.md` | Operators | SeQaM scenario setup and phase-trigger flow |
| `publishers/network_conditions/README.md` | Operators | Network VM scripts: full setup and CLI reference |
| `dashboard/README.md` | Everyone | Dashboard setup, Pi configuration, troubleshooting |
| `seqam/README.md` | Operators | SeQaM scenario configuration |
