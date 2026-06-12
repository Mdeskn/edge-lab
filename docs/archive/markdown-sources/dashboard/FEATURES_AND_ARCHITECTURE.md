# Edge-Lab Dashboard: Features and Architecture

This document explains what the Edge-Lab dashboard shows, why each feature is
useful in the lab, and how data moves from the drone-view car video to the browser.

## 1. Educational Goal

The experiment teaches service placement: deciding whether inference should run
locally on the Raspberry Pi or remotely on the Triton GPU server.

Students edit only:

```text
client/student/sp_agent.py
```

The agent returns:

```text
local
```

or:

```text
remote
```

The dashboard makes the consequences visible. A good decision should keep
inference latency low and the predicted car position close to the
ground-truth car position. The competition score is cumulative
displacement in pixels. Lower is better.

## 2. What The Dashboard Shows

Open the dashboard at:

```text
http://PI_IP:5173
```

### Live Annotated Video

The video panel displays the newest scored video frame.

| Overlay | Color | Meaning |
| --- | --- | --- |
| `GT` dot | Green | Ground-truth car center (from HSV colour segmentation) |
| `PRED` box | Red | YOLO-predicted car bounding box |
| Connecting line | Yellow | Distance between ground truth center and predicted center |

The overlay is drawn by the Python client before the JPEG is sent to the
dashboard. The browser displays the already-annotated image, so browser resizing
cannot move a marker away from the car.

There are three important cases:

| Situation | Green GT dot | Red PRED box | Displacement |
| --- | --- | --- | --- |
| Car visible and YOLO finds it | Visible | Visible | Distance between GT center and predicted center |
| Car visible but YOLO misses it | Visible | Hidden | Penalized as distance from GT to `(0, 0)` |
| Car not visible / warmup frame | Hidden | Hidden | `N/A`; frame is not added to the cumulative score |

### Current Processing Mode

The large processing-mode card shows the backend used for the current frame:

| Value | Meaning |
| --- | --- |
| `LOCAL` | ONNX YOLO inference ran on the Pi or local CPU |
| `REMOTE` | The frame was sent to Triton on the GPU server |
| `LOCAL_FALLBACK` | Remote inference failed and the dispatcher used local inference |

The student agent writes its choice into shared state. The dispatcher reads that
choice immediately before processing each frame. The card therefore reflects
the backend that actually handled the scored frame.

### Latency

The latency card shows:

| Metric | Calculation |
| --- | --- |
| Current frame latency | Time from dispatcher start through preprocessing and inference |
| Rolling average | Average of the latest 20 dashboard frame samples |
| Minimum | Lowest latency in the retained dashboard history |
| Maximum | Highest latency in the retained dashboard history |
| P95 | 95th-percentile latency in the retained dashboard history |

Latency matters because a slow prediction can lag behind a moving car. Remote
inference may be fast when the network and GPU server are healthy, but slower
when the network path or Triton queue is under load.

### Displacement

The displacement card shows:

| Metric | Calculation |
| --- | --- |
| Current frame displacement | `sqrt((true_x - predicted_x)^2 + (true_y - predicted_y)^2)` |
| Rolling average | Average displacement of the latest 20 retained samples with a visible ball |
| Cumulative score | Sum of displacement across all scored frames |

The cumulative score card is intentionally prominent. This is the competition
metric and lower values are better.

### Experiment Phase

The top bar displays the current experiment phase:

| Phase | Meaning |
| --- | --- |
| `baseline` | No artificial load |
| `gpu_load` | GPU server flooded with 100 concurrent requests |
| `bandwidth_50` | Network path capped at 50 Mbit/s by tc |
| `mixed` | Both GPU flooded and 50 Mbit/s cap active |
| `bandwidth_200` | Network path capped at 200 Mbit/s |
| `jitter_light` | Light jitter added to the network path |
| `unknown` | No phase message has arrived yet |

Phase values arrive from Kafka. Before the first phase message arrives, the
phase remains `unknown`.

### Infrastructure Panel

The infrastructure panel shows the newest available external metrics:

| Section | Metrics |
| --- | --- |
| GPU server | GPU utilization, GPU memory utilization, Triton queue duration, Triton requests per second |
| Network path | Delay, jitter, packet loss, monitored interface |
| Data links | Kafka metrics status and client-frame status |

Missing values display as `N/A` instead of crashing the dashboard. If a value is
missing during the lab, check the relevant publisher and Kafka topic.

### Live Trend Charts

The dashboard keeps a rolling history rather than an unbounded log. The default
window is the latest `300` samples.

| Chart | Purpose |
| --- | --- |
| Latency | Reveals slow inference and spikes |
| Frame displacement | Reveals tracking error for individual scored frames |
| Cumulative displacement | Shows the score increasing over time |
| Placement mode | Plots local as `0` and remote as `1` |
| GPU utilization | Shows server load when GPU metrics are available |
| Network conditions | Plots delay and jitter together |

The frontend draws lightweight SVG polylines. It does not use a heavyweight
charting engine, which keeps the dashboard responsive on modest hardware.

### Run Summary

The summary panel accumulates:

| Metric | Meaning |
| --- | --- |
| Total frames | Number of unique dashboard frame metrics received |
| Local frames | Frames handled locally, including `local_fallback` |
| Remote frames | Frames handled remotely |
| Local / remote percentages | Placement distribution |
| Average latency | Average latency across received frames |
| Average displacement | Average across frames with a displacement value |
| Final score | Latest cumulative displacement |
| Best / worst latency | Lowest and highest received latency |
| Best / worst displacement | Lowest and highest scored displacement |

### Interpretation Panel

The “What does this mean?” panel converts metrics into a short educational hint.
Its rules are deliberately simple:

| Condition | Message |
| --- | --- |
| Delay at least `60 ms` or packet loss at least `2%` | Network conditions are poor; local processing may be safer |
| GPU utilization at least `85%` or Triton queue at least `25 ms` | Server is under load; local processing may reduce delay |
| Recent displacement rises quickly | Prediction is lagging further behind GT |
| Remote mode with healthy GPU and network | Remote inference is currently beneficial |
| Local mode | Local inference avoids network and server variability |

### Per-Group Deployment And Connection Status

Each student group deploys its own dashboard instance with its `GROUP_ID`.
There is no central dashboard and no browser-side group switcher.

The top-bar status shows whether the browser WebSocket is live. If disconnected,
the frontend retries after `1.5` seconds.

## 3. Single-Car Tracking

The lab tracks a single car in a drone-view video. Other detected objects
must never become the target.

### Ground Truth

Ground truth is generated offline using HSV colour segmentation — no YOLO model
needed and no background model, so it works correctly with a moving camera (drone):

```bash
python ground_truth/generate_ground_truth.py \
  --video data/test_video.mp4 \
  --output data/ground_truth.csv
```

The HSV colour tracker:

1. Converts each frame from BGR to HSV.
2. Masks both red hue bands (hue 0–10 and hue 160–179, the two sides of red in OpenCV).
3. Cleans the mask with morphological open and close operations.
4. Finds contours and selects the largest red blob within car-sized area bounds.
5. Writes its bounding-box centre to `data/ground_truth.csv`.
6. Linearly interpolates any remaining NaN rows between known detections.

This tracker runs only while generating the answer key. It is not part of the
measured local or remote inference path. Keeping ground truth independent of
YOLO means a YOLO miss remains measurable.

### Predicted Position

Measured inference uses YOLOv10 ONNX. Both local CPU inference and remote
Triton inference apply:

```dotenv
TARGET_CLASS_ID=2,5,7
TARGET_CONFIDENCE_THRESHOLD=0.1
```

Primary class is `2` (car). The helper classes (train=6, cell phone=67, airplane=4,
person=0) cover frames where YOLO misclassifies the car from a drone viewpoint — the
car rooftop viewed from above resembles flat rectangular shapes from the COCO training
distribution. Each inference path:

1. Runs YOLO.
2. Removes detections below `TARGET_CONFIDENCE_THRESHOLD`.
3. Removes every class except those in `TARGET_CLASS_ID`.
4. Selects the highest-confidence remaining detection.
5. Converts its bounding box and center from model space back to the original video resolution.
6. Returns `(0.0, 0.0)` when no detection remains.

This maximises detection rate (~99%) while keeping the tracker on the car.

## 4. Frame-By-Frame Data Flow

The complete frame flow is:

```text
test_video.mp4
      |
      v
FrameReader ---- ground_truth.csv
      |
      v
Dispatcher ---- SP-Agent choice: local or remote
      |
      +---- local ----> LocalServer: ONNX Runtime CPU
      |
      +---- remote ---> RemoteClient: Triton HTTP
      |
      v
Scorer
      |
      +---- results.csv
      |
      +---- Kafka app metric, when configured
      |
      +---- annotated JPEG + metric payload
                    |
                    v
             FastAPI dashboard backend
                    |
                    +---- in-memory latest state and rolling history
                    |
                    +---- WebSocket JSON updates
                    |
                    +---- latest JPEG endpoint
                                  |
                                  v
                           React dashboard
```

### Step 1: Read The Frame

`client/threads/frame_reader.py`:

1. Reads the next frame from the configured video.
2. Looks up the matching row in `ground_truth.csv`.
3. Uses `(None, None)` when the CSV row contains `NaN`.
4. Adds the frame and its GT coordinate to `reader_queue`.
5. Loops to frame `1` after the video ends.

### Step 2: Choose Local Or Remote Inference

`client/threads/dispatcher.py`:

1. Reads the latest placement choice from shared state.
2. Preprocesses the frame into YOLO input shape `(1, 3, 640, 640)`.
3. Uses local ONNX inference when mode is `local`.
4. Uses remote Triton inference when mode is `remote` and Triton is reachable.
5. Falls back to local inference if remote inference fails.
6. Measures latency.
7. Adds the result to `scorer_queue`.

### Step 3: Score And Draw

`client/threads/scorer.py`:

1. Calculates displacement when GT exists.
2. Adds displacement to the cumulative score.
3. Draws GT, PRED, and the connecting line when their coordinates exist.
4. Draws a small HUD with mode, phase, latency, displacement, cumulative score,
   and scored-frame count.
5. Appends a CSV row to `RESULTS_LOG_PATH`.
6. Publishes a compact metric record to Kafka when Kafka is configured.
7. Calls the optional dashboard publisher.

### Step 4: Publish A Lightweight Dashboard Frame

`client/metrics/dashboard_publisher.py` is intentionally best-effort:

1. It is enabled only when `DASHBOARD_ENABLED=true`.
2. It throttles outgoing images to `DASHBOARD_FPS`.
3. It optionally downsizes frames to `DASHBOARD_FRAME_WIDTH`.
4. It JPEG-compresses frames using `DASHBOARD_JPEG_QUALITY`.
5. It puts the newest payload into a queue with capacity `1`.
6. If the queue is full, it discards the older payload and keeps the newest one.
7. A daemon thread sends HTTP POST requests to the backend with a short timeout.
8. Backend outages are logged at debug level and never crash inference.

This design prevents dashboard traffic from slowing down the experiment.

## 5. Backend Architecture

The backend lives in:

```text
dashboard/backend/
```

It is a small FastAPI application with in-memory state. There is no database.

### REST And WebSocket API

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Backend health, configured group, Kafka status, and history size |
| `GET /api/state` | Full initial dashboard snapshot |
| `GET /api/history` | Rolling chart histories only |
| `POST /api/frame` | Receive annotated JPEG and frame metrics from this group's client |
| `GET /api/frame` | Return the latest JPEG |
| `POST /api/reset` | Reset retained counters and history between runs |
| `WS /ws` | Push live JSON state snapshots |
| `GET /docs` | FastAPI-generated API documentation |

### In-Memory State

`dashboard/backend/state.py` stores:

- Latest frame metrics for the configured group.
- Latest JPEG for the configured group.
- A frame sequence number used as a cache-busting query parameter.
- Latest GPU and network metrics.
- Rolling frame, GPU, and network histories.
- Summary counters.
- Kafka status.

The default retained history is:

```dotenv
DASHBOARD_MAX_HISTORY=300
```

The state object is protected by a reentrant lock because FastAPI handlers and
the Kafka consumer thread may update it concurrently.

### Duplicate Suppression

In a full lab run, one scored frame can reach the backend twice:

1. Through Kafka as an app metric.
2. Through the HTTP JPEG upload from the client.

The backend identifies a sample by group, frame number, and timestamp. It keeps
the newest metric state but counts each unique sample only once in histories and
summary statistics.

### Kafka Consumer

`dashboard/backend/kafka_consumer.py` is optional. If `KAFKA_BROKERS` is empty,
the backend runs normally without Kafka.

When enabled, its background thread subscribes to:

| Environment variable | Default topic | Data |
| --- | --- | --- |
| `APP_METRICS_TOPIC` | `dnn_partition.client_metrics` | This group's per-frame app metrics |
| `KAFKA_GPU_TOPIC` | `dnn_partition.server_metrics` | GPU and Triton metrics |
| `KAFKA_NET_TOPIC` | `edgelab.network.metrics` | Delay, jitter, packet loss |
| `KAFKA_PHASE_TOPIC` | `edgelab.phase` | Current experiment phase |

Kafka messages update state and schedule a WebSocket broadcast on FastAPI's
async event loop.

## 6. Frontend Architecture

The frontend lives in:

```text
dashboard/frontend/
```

It is a React and Vite application.

### Startup And Live Updates

`dashboard/frontend/src/api.ts`:

1. Fetches the initial snapshot with `GET /api/state`.
2. Opens `WS /ws`.
3. Replaces React state whenever a WebSocket message arrives.
4. Closes and reconnects after `1.5` seconds if the socket disconnects.

### Video Refresh

The backend returns a JPEG URL such as:

```text
/api/frame?v=1251
```

The sequence value changes for each received image. This prevents browser image
caching. The backend also returns `Cache-Control: no-store`.

### Missing Metrics

The frontend consistently renders unavailable values as:

```text
N/A
```

This allows the UI to remain usable while publishers connect or when an
infrastructure metric is temporarily unavailable.

## 7. Metric Payload

Each dashboard JPEG POST contains:

```json
{
  "timestamp": 1780250000.0,
  "frame_number": 875,
  "true_x": 1088.0,
  "true_y": 534.0,
  "predicted_x": 1091.0,
  "predicted_y": 536.2,
  "processing_mode": "local",
  "latency_ms": 36.5,
  "displacement_px": 3.72,
  "cumulative_displacement_px": 12345.67,
  "experiment_phase": "unknown",
  "image_base64": "...JPEG bytes encoded as base64..."
}
```

When the car is not visible (or during MOG2 warmup):

```json
{
  "true_x": null,
  "true_y": null,
  "displacement_px": null
}
```

When the car is visible but YOLO misses it:

```json
{
  "true_x": 663.0,
  "true_y": 582.5,
  "predicted_x": 0.0,
  "predicted_y": 0.0
}
```

## 8. Student Lab Deployment

Students run the experiment using the lab infrastructure described in the root
[README.md](../README.md). The dashboard is a visualization layer for that
experiment.

The deployed lab uses:

| Machine | Services |
| --- | --- |
| Raspberry Pi | Client, SP-Agent, local ONNX inference, and one group's dashboard |
| GPU server | Triton and GPU metrics publisher |
| Network VM | Network conditions publisher and Linux `tc netem` |
| SeQaM platform | Kafka broker and phase controller |

Configure:

```dotenv
GROUP_ID=1
KAFKA_BROKERS=HOST:PORT
TRITON_URL=GPU_SERVER:8000
DASHBOARD_ENABLED=true
DASHBOARD_URL=http://localhost:8080
TARGET_CLASS_ID=2,5,7
TARGET_CONFIDENCE_THRESHOLD=0.1
```

The browser derives the Pi dashboard API address from the hostname used to open
the frontend. An explicit override is optional:

```dotenv
DASHBOARD_PUBLIC_API_URL=
```

In the full lab, the dashboard shows external GPU metrics, network conditions,
Kafka status, phase transitions, and the effect of the student's placement
strategy.

### Student Processing Choice

For every video frame, the SP-Agent returns one of two values:

| Agent return value | Processing location |
| --- | --- |
| `local` | Raspberry Pi CPU using ONNX Runtime |
| `remote` | Triton on the remote GPU server over the network |

The dispatcher measures the actual frame latency, scores the resulting
car position, and reports the actual processing mode to the dashboard.
Students can then see whether their placement decision was appropriate for the
current GPU and network conditions.

For setup and run commands, follow the root [README.md](../README.md).

## 9. Important Configuration

| Variable | Default | Purpose |
| --- | --- | --- |
| `DASHBOARD_ENABLED` | `false` | Enables optional client JPEG publishing |
| `DASHBOARD_URL` | `http://localhost:8080` | Dashboard backend URL used by the client on the same Pi |
| `DASHBOARD_FPS` | `5` | Maximum dashboard JPEG updates per second |
| `DASHBOARD_JPEG_QUALITY` | `70` | JPEG compression quality |
| `DASHBOARD_FRAME_WIDTH` | `960` | Maximum published image width |
| `GROUP_ID` | `1` | Student group number; selects this dashboard's app-metrics topic |
| `DASHBOARD_MAX_HISTORY` | `300` | Rolling backend sample count |
| `DASHBOARD_HOST` | `0.0.0.0` | Backend listen host |
| `DASHBOARD_PORT` | `8080` | Backend listen port |
| `DASHBOARD_PUBLIC_API_URL` | empty | Optional browser-facing backend override; otherwise derive `http://<pi-host>:8080` |
| `TARGET_CLASS_ID` | empty | YOLO COCO class filter; use `2,6,67,4,0` for drone-view car tracking |
| `TARGET_CONFIDENCE_THRESHOLD` | same as `CONFIDENCE_THRESHOLD` | Detection threshold after target filtering |

## 10. Main Files

| File | Responsibility |
| --- | --- |
| `ground_truth/generate_ground_truth.py` | Generates independent car answer key via MOG2 |
| `client/inference/local_server.py` | Runs local car-only YOLO inference |
| `client/inference/remote_client.py` | Runs remote Triton car-only YOLO inference |
| `client/threads/frame_reader.py` | Reads frames and car GT coordinates |
| `client/threads/dispatcher.py` | Applies SP-Agent placement and measures latency |
| `client/threads/scorer.py` | Scores, annotates, logs, and publishes frames |
| `client/metrics/dashboard_publisher.py` | Sends throttled JPEG snapshots without blocking inference |
| `dashboard/backend/main.py` | Defines REST endpoints and WebSocket broadcasting |
| `dashboard/backend/state.py` | Maintains bounded in-memory histories and summaries |
| `dashboard/backend/kafka_consumer.py` | Optionally consumes Kafka metrics |
| `dashboard/frontend/src/App.tsx` | Composes the browser dashboard |
| `dashboard/frontend/src/api.ts` | Loads initial state and reconnects WebSocket |

## 11. Troubleshooting

### The prediction box follows the wrong object

Confirm:

```dotenv
TARGET_CLASS_ID=2,5,7
TARGET_CONFIDENCE_THRESHOLD=0.1
```

For instructor setup or maintenance, regenerate car ground truth:

```bash
python ground_truth/generate_ground_truth.py \
  --video data/test_video.mp4 \
  --output data/ground_truth.csv
```

### The green dot appears but the red box is missing

The car is visible in the ground truth, but YOLO did not emit a `car` detection
above the configured threshold. This is a valid measured miss.

### Both the dot and box are missing

The HSV ground-truth tracker did not find a red blob of car size in that frame
(car fully out of frame or heavily occluded). The frame is displayed but not
added to the cumulative displacement score.

### GPU and network values show N/A

Check Kafka, Triton, and the infrastructure publishers described in the root
[README.md](../README.md).

### The browser says reconnecting

Check:

```bash
curl http://PI_IP:8080/health
```

Restart the dashboard services on the group's Pi if needed.
