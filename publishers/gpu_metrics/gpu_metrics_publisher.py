"""
Runs on the GPU server. Polls nvidia-smi and Triton metrics endpoint.
Publishes to Kafka topic /edgelab/server/metrics every POLL_INTERVAL_SEC seconds.

Environment variables required:
    KAFKA_BROKERS, KAFKA_GPU_TOPIC, TRITON_METRICS_URL, NVIDIA_SMI_PATH,
    POLL_INTERVAL_SEC (default: 1), LOG_LEVEL
"""
import json
import logging
import os
import subprocess
import time

import requests
from confluent_kafka import Producer
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=getattr(logging, os.environ.get("LOG_LEVEL", "INFO").upper(), logging.INFO),
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Prometheus counter state for rate calculation
_prev_requests: float = 0.0
_prev_queue_us: float = 0.0
_prev_infer_us: float = 0.0
_prev_ts: float = 0.0


def get_nvidia_smi_metrics(nvidia_smi_path: str) -> dict:
    """
    Query nvidia-smi for GPU utilisation, memory, temperature, and power.

    Runs nvidia-smi with CSV output. Returns a dict with keys:
        gpu_utilization_pct, gpu_memory_used_mb, gpu_memory_total_mb,
        gpu_temperature_c, gpu_power_draw_w
    Returns an empty dict on any error.
    """
    try:
        result = subprocess.run(
            [
                nvidia_smi_path,
                "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            logger.warning("nvidia-smi exited with code %d", result.returncode)
            return {}

        parts = [p.strip() for p in result.stdout.strip().split(",")]
        if len(parts) < 5:
            logger.warning("Unexpected nvidia-smi output: %s", result.stdout.strip())
            return {}

        return {
            "gpu_utilization_pct": float(parts[0]),
            "gpu_memory_used_mb": float(parts[1]),
            "gpu_memory_total_mb": float(parts[2]),
            "gpu_temperature_c": float(parts[3]),
            "gpu_power_draw_w": float(parts[4]),
        }
    except Exception as exc:
        logger.error("nvidia-smi error: %s", exc)
        return {}


def get_triton_metrics(triton_metrics_url: str, poll_interval: float) -> dict:
    """
    Fetch Prometheus metrics from Triton and compute per-second rates.

    Extracts:
        nv_inference_request_success
        nv_inference_queue_duration_us
        nv_inference_compute_infer_duration_us

    Returns a dict with:
        triton_requests_per_sec, triton_queue_duration_ms,
        triton_inference_duration_ms
    Returns an empty dict on any error.
    """
    global _prev_requests, _prev_queue_us, _prev_infer_us, _prev_ts

    try:
        resp = requests.get(triton_metrics_url, timeout=5)
        resp.raise_for_status()
        text = resp.text
    except Exception as exc:
        logger.error("Triton metrics fetch error: %s", exc)
        return {}

    raw: dict = {}
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) >= 2:
            key = parts[0].split("{")[0]
            try:
                raw[key] = float(parts[-1])
            except ValueError:
                pass

    now = time.time()
    dt = now - _prev_ts if _prev_ts > 0 else poll_interval

    cur_req = raw.get("nv_inference_request_success", 0.0)
    cur_queue = raw.get("nv_inference_queue_duration_us", 0.0)
    cur_infer = raw.get("nv_inference_compute_infer_duration_us", 0.0)

    delta_req = max(cur_req - _prev_requests, 0.0)
    delta_queue = max(cur_queue - _prev_queue_us, 0.0)
    delta_infer = max(cur_infer - _prev_infer_us, 0.0)

    _prev_requests = cur_req
    _prev_queue_us = cur_queue
    _prev_infer_us = cur_infer
    _prev_ts = now

    rps = delta_req / dt if dt > 0 else 0.0
    queue_ms = (delta_queue / (delta_req * 1000.0)) if delta_req > 0 else 0.0
    infer_ms = (delta_infer / (delta_req * 1000.0)) if delta_req > 0 else 0.0

    return {
        "triton_requests_per_sec": round(rps, 3),
        "triton_queue_duration_ms": round(queue_ms, 3),
        "triton_inference_duration_ms": round(infer_ms, 3),
    }


def main() -> None:
    """
    Initialise Kafka producer and poll GPU + Triton metrics in a loop,
    publishing a JSON record every POLL_INTERVAL_SEC seconds.
    """
    kafka_brokers = os.environ["KAFKA_BROKERS"]
    gpu_topic = os.environ.get("KAFKA_GPU_TOPIC", "/edgelab/server/metrics")
    triton_metrics_url = os.environ.get(
        "TRITON_METRICS_URL", "http://localhost:8002/metrics"
    )
    nvidia_smi_path = os.environ.get("NVIDIA_SMI_PATH", "/usr/bin/nvidia-smi")
    poll_interval = float(os.environ.get("POLL_INTERVAL_SEC", "1"))

    producer = Producer(
        {
            "bootstrap.servers": kafka_brokers,
            "client.id": "gpu-metrics-publisher",
        }
    )
    logger.info(
        "GPU metrics publisher started: brokers=%s topic=%s interval=%.1fs",
        kafka_brokers,
        gpu_topic,
        poll_interval,
    )

    def _delivery_report(err, msg):
        if err:
            logger.error("Kafka delivery failed: %s", err)

    while True:
        metrics = {
            **get_nvidia_smi_metrics(nvidia_smi_path),
            **get_triton_metrics(triton_metrics_url, poll_interval),
            "timestamp": time.time(),
        }
        try:
            producer.produce(
                gpu_topic,
                key="gpu",
                value=json.dumps(metrics).encode("utf-8"),
                callback=_delivery_report,
            )
            producer.poll(0)
        except Exception as exc:
            logger.error("Kafka produce error: %s", exc)

        time.sleep(poll_interval)


if __name__ == "__main__":
    main()
