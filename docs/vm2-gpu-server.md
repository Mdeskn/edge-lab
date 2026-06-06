# VM 2: GPU Server

**IP:** `172.22.174.145`

This machine runs the Triton Inference Server and Eldiyar's GPU metrics publisher. Students never connect to it directly; all traffic goes via the Router VM (VM 3) at `172.22.174.148:8001`.

---

## What runs here

| Service | Port | Notes |
|---------|------|-------|
| Triton HTTP | 8000 | Internal only; students use the router |
| Triton gRPC | 8001 | Proxied by the Router VM at `172.22.174.148:8001` |
| Triton Prometheus metrics | 8002 | Read by Eldiyar's publisher |
| GPU metrics publisher | (no port) | Background process; publishes to Kafka |

---

## Models loaded in Triton

| Model | Purpose |
|-------|---------|
| `yolov10n` | YOLOv10n object detection; used by the Pi client for remote inference |
| `resnet50_full` | ResNet50; loaded for capacity/stress testing |

The model repository is at `/path/to/triton/model_repository/` on the GPU server. The `yolov10n` model was copied from `triton/model_repository/yolov10n/1/model.onnx` in this repo.

---

## Why students use the Router VM address

During network experiments, the Network VM (Router VM, `172.22.174.148`) applies `tc` traffic shaping rules on the path between the Pi and the GPU server. For those rules to affect the inference traffic, the Pi must route through the Router VM. A direct connection to `172.22.174.145:8001` bypasses all network impairments.

**Always set `TRITON_URL=172.22.174.148:8001` in `.env`, not `172.22.174.145:8001`.**

---

## Verifying Triton is running

From any machine:

```bash
curl http://172.22.174.148:8001/v2/health/live   # via Router VM (correct path)
curl http://172.22.174.145:8001/v2/health/live   # direct (bypass check only)
```

List loaded models:

```bash
curl http://172.22.174.145:8000/v2/models
```

Test inference:

```bash
curl http://172.22.174.145:8000/v2/models/yolov10n/ready
```

---

## GPU metrics publisher (Eldiyar)

The publisher at `/home/mae/server/kafka_metrics_publisher.py` runs on the GPU server and publishes to `dnn_partition.server_metrics` every second.

**Do not replace or modify this script.** The SPAgentBase and dashboard backend both know how to parse its nested message format (see `docs/vm1-kafka-grafana.md` for the schema).

The publisher polls:
- `nvidia-smi` for GPU utilization, temperature, memory, power draw
- `http://localhost:8002/metrics` (Triton Prometheus) for queue depths, inference times, and request rates per model

---

## GPU stressor

The stressor script (`scripts/gpu_stressor.sh`) floods Triton with concurrent requests to simulate the `gpu_load` experiment phase. SeQaM calls it via SSH:

```bash
bash /scripts/gpu_stressor.sh yolov10n 100 30
# model_name, concurrency, duration_seconds
```

Running this command directly stresses the GPU for 30 seconds with 100 parallel requests. It uses Triton's `perf_analyzer` tool which must be installed on the GPU server.

To stop a running stressor manually:

```bash
ssh mae@172.22.174.145 "pkill -f perf_analyzer"
```

---

## Confirming GPU metrics flow to Kafka

```bash
kafka-console-consumer.sh \
    --bootstrap-server 172.22.174.149:9092 \
    --topic dnn_partition.server_metrics \
    --max-messages 3 | python3 -m json.tool
```

You should see a nested JSON with `server`, `totals`, and `models` keys (see `docs/vm1-kafka-grafana.md` for the full schema).

---

## Triton model configuration

The model configuration for `yolov10n` is in `triton/model_repository/yolov10n/config.pbtxt`. It defines:
- Input: `images`, shape `[1, 3, 640, 640]`, dtype `FP32`
- Output: `output0`, shape `[-1, 6]`, dtype `FP32`
- Dynamic batching enabled

The model file (`model.onnx`) must be placed at `triton/model_repository/yolov10n/1/model.onnx` on the GPU server. Copy it from `yolov10n.onnx` in this repo.
