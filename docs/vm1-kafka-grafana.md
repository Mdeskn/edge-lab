# VM 1: SeQaM / Kafka / Grafana

**IP:** `172.22.174.149`

This is the shared lab infrastructure machine. It runs the Kafka message bus, Kafka UI, Grafana dashboards, and the SeQaM experiment orchestration platform. It is maintained by the lab and does not require student or operator configuration.

---

## What runs here

| Service | Port | URL |
|---------|------|-----|
| Kafka broker | 9092 | `172.22.174.149:9092` |
| Kafka UI | 8080 | `http://172.22.174.149:8080` |
| Grafana | 3000 | `http://172.22.174.149:3000` |
| SeQaM API | 8000 | `http://172.22.174.149:8000` |

---

## Kafka topics

| Topic | Publisher | Consumers |
|-------|-----------|-----------|
| `dnn_partition.server_metrics` | Eldiyar's GPU metrics publisher on VM 2 | Pi SP-Agent, dashboard |
| `dnn_partition.client_metrics` | Pi client app (each group) | Dashboard, SP-Agent latency feedback |
| `edgelab.network.metrics` | Network conditions publisher on VM 3 | Pi SP-Agent, dashboard |
| `edgelab.phase` | tc_controller on VM 3 | Pi SP-Agent, dashboard, main.py auto-stop |

---

## Verifying Kafka connectivity

From any machine with network access:

```bash
# Check that port 9092 is open
nc -zv 172.22.174.149 9092

# List topics (requires kafka-topics.sh on the path)
kafka-topics.sh --bootstrap-server 172.22.174.149:9092 --list

# Watch GPU metrics messages live
kafka-console-consumer.sh \
    --bootstrap-server 172.22.174.149:9092 \
    --topic dnn_partition.server_metrics

# Watch phase events live
kafka-console-consumer.sh \
    --bootstrap-server 172.22.174.149:9092 \
    --topic edgelab.phase

# Watch network metrics live
kafka-console-consumer.sh \
    --bootstrap-server 172.22.174.149:9092 \
    --topic edgelab.network.metrics
```

Or use Kafka UI at `http://172.22.174.149:8080`: browse topics, inspect messages, and see consumer group lag.

---

## Configuration for Pi clients

All clients and publishers point to this VM for Kafka:

```dotenv
KAFKA_BROKERS=172.22.174.149:9092
```

This is set in `.env` on each Pi and in each publisher's configuration.

---

## GPU metrics message format

Eldiyar's publisher on VM 2 sends messages to `dnn_partition.server_metrics` with a nested structure:

```json
{
  "timestamp": 1780400000.0,
  "server_id": "gpu-server",
  "server": {
    "gpu_util_percent": 45.3,
    "gpu_freq_mhz": 1350.0,
    "gpu_temp_c": 65.0,
    "gpu_mem_used_mb": 2048.0,
    "gpu_mem_total_mb": 8192.0,
    "cpu_util_percent": 22.0,
    "mem_util_percent": 34.0,
    "power_w": 180.0
  },
  "totals": {
    "total_rps": 12.5,
    "total_success_rps": 12.3,
    "total_failure_rps": 0.2,
    "total_pending_requests": 4
  },
  "models": [
    {
      "model_name": "yolov10n",
      "success_rps": 10.1,
      "inference_rps": 10.2,
      "pending_requests": 3,
      "avg_queue_time_ms": 8.5,
      "avg_compute_input_ms": 0.2,
      "avg_compute_infer_ms": 12.4,
      "avg_compute_output_ms": 0.1
    }
  ]
}
```

The Pi's SPAgentBase and the dashboard backend both flatten this into the normalized field names (`gpu_util_pct`, `yolo_queue_ms`, etc.) before exposing them to students.

---

## Grafana

Grafana at `http://172.22.174.149:3000` provides experiment-wide dashboards. Default login: check with the lab administrator. The dashboards show GPU utilization, network conditions, and per-group inference latency over time.

---

## SeQaM

SeQaM is the experiment orchestration platform. It reads `seqam/scenario.json` and executes load commands via SSH at specified times. See `docs/seqam-integration.md` for how to run the scenario.
