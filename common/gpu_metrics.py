"""Normalization for GPU / Triton server metrics messages.

Both the SP-Agent (client side) and the dashboard backend consume
KAFKA_GPU_TOPIC, and both previously carried their own near-identical copy of
this logic. The copies had already drifted: the dashboard's flat branch had
lost the ResNet fields. One implementation lives here so the students' metric
keys and the dashboard's metric keys can never disagree.

Two producer formats appear on the same topic:

  nested : has "server" / "totals" / "models" keys (the GPU server publisher)
  flat   : has "gpu_utilization_pct" / "triton_requests_per_sec" keys

`normalize_gpu_metrics` accepts either and always returns the same flat,
snake_case key set, so downstream code never has to branch on the format.
"""
from typing import Any

#: Every key `normalize_gpu_metrics` guarantees to return, with its zero value.
#: Consumers can rely on all of these being present after normalization.
GPU_METRIC_DEFAULTS: dict[str, Any] = {
    # GPU / host hardware
    "gpu_util_pct": 0.0,
    "gpu_freq_mhz": 0.0,
    "gpu_temp_c": 0.0,
    "gpu_mem_used_mb": 0.0,
    "gpu_mem_total_mb": 0.0,
    "cpu_util_pct": 0.0,
    "mem_util_pct": 0.0,
    "power_w": 0.0,
    # Triton aggregate throughput
    "total_rps": 0.0,
    "total_success_rps": 0.0,
    "total_failure_rps": 0.0,
    "total_pending": 0,
    # YOLOv10n model
    "yolo_success_rps": 0.0,
    "yolo_inference_rps": 0.0,
    "yolo_pending": 0,
    "yolo_queue_ms": 0.0,
    "yolo_input_ms": 0.0,
    "yolo_infer_ms": 0.0,
    "yolo_output_ms": 0.0,
    # ResNet50 model
    "resnet_success_rps": 0.0,
    "resnet_inference_rps": 0.0,
    "resnet_pending": 0,
    "resnet_queue_ms": 0.0,
    "resnet_infer_ms": 0.0,
}

_YOLO_MODEL_NAME = "yolov10n"
_RESNET_MODEL_NAME = "resnet50_full"


def is_nested_format(message: dict) -> bool:
    """Return True when `message` uses the nested server/totals/models layout."""
    return "server" in message or "models" in message or "totals" in message


def normalize_gpu_metrics(message: dict, include_raw: bool = True) -> dict[str, Any]:
    """
    Return a flat metric dict for either producer format.

    Every key in GPU_METRIC_DEFAULTS is always present. "timestamp" is copied
    from the message when available. "raw" carries the original message and is
    included only when `include_raw` is set: the SP-Agent exposes it to
    students, the dashboard omits it to keep WebSocket payloads small.
    """
    if not isinstance(message, dict):
        message = {}

    if is_nested_format(message):
        metrics = _from_nested(message)
    else:
        metrics = _from_flat(message)

    normalized = dict(GPU_METRIC_DEFAULTS)
    normalized.update(metrics)
    normalized["timestamp"] = message.get("timestamp", 0.0)
    if include_raw:
        normalized["raw"] = message
    return normalized


def _from_nested(message: dict) -> dict[str, Any]:
    """Read the nested server/totals/models layout used by the GPU server."""
    server = message.get("server") or {}
    totals = message.get("totals") or {}
    models = message.get("models") or []

    yolo = _find_model(models, _YOLO_MODEL_NAME)
    resnet = _find_model(models, _RESNET_MODEL_NAME)

    return {
        "gpu_util_pct": server.get("gpu_util_percent", 0.0),
        "gpu_freq_mhz": server.get("gpu_freq_mhz", 0.0),
        "gpu_temp_c": server.get("gpu_temp_c", 0.0),
        "gpu_mem_used_mb": server.get("gpu_mem_used_mb", 0.0),
        "gpu_mem_total_mb": server.get("gpu_mem_total_mb", 0.0),
        "cpu_util_pct": server.get("cpu_util_percent", 0.0),
        "mem_util_pct": server.get("mem_util_percent", 0.0),
        "power_w": server.get("power_w", 0.0),

        "total_rps": totals.get("total_rps", 0.0),
        "total_success_rps": totals.get("total_success_rps", 0.0),
        "total_failure_rps": totals.get("total_failure_rps", 0.0),
        "total_pending": totals.get("total_pending_requests", 0),

        "yolo_success_rps": yolo.get("success_rps", 0.0),
        "yolo_inference_rps": yolo.get("inference_rps", 0.0),
        "yolo_pending": yolo.get("pending_requests", 0),
        "yolo_queue_ms": yolo.get("avg_queue_time_ms", 0.0),
        "yolo_input_ms": yolo.get("avg_compute_input_ms", 0.0),
        "yolo_infer_ms": yolo.get("avg_compute_infer_ms", 0.0),
        "yolo_output_ms": yolo.get("avg_compute_output_ms", 0.0),

        "resnet_success_rps": resnet.get("success_rps", 0.0),
        "resnet_inference_rps": resnet.get("inference_rps", 0.0),
        "resnet_pending": resnet.get("pending_requests", 0),
        "resnet_queue_ms": resnet.get("avg_queue_time_ms", 0.0),
        "resnet_infer_ms": resnet.get("avg_compute_infer_ms", 0.0),
    }


def _from_flat(message: dict) -> dict[str, Any]:
    """Read the flat layout published by gpu_metrics_publisher.py."""
    rps = message.get("triton_requests_per_sec", 0.0)
    queue_ms = message.get("triton_queue_duration_ms", 0.0)
    infer_ms = message.get("triton_inference_duration_ms", 0.0)

    return {
        "gpu_util_pct": message.get("gpu_utilization_pct", 0.0),
        "gpu_temp_c": message.get("gpu_temperature_c", 0.0),
        "gpu_mem_used_mb": message.get("gpu_memory_used_mb", 0.0),
        "gpu_mem_total_mb": message.get("gpu_memory_total_mb", 0.0),
        "power_w": message.get("gpu_power_draw_w", 0.0),

        "total_rps": rps,
        "total_success_rps": rps,

        "yolo_success_rps": rps,
        "yolo_inference_rps": rps,
        "yolo_queue_ms": queue_ms,
        "yolo_infer_ms": infer_ms,
    }


def _find_model(models: list, model_name: str) -> dict:
    """Return the per-model stats block for `model_name`, or an empty dict."""
    for model in models:
        if isinstance(model, dict) and model.get("model_name") == model_name:
            return model
    return {}


_DEFAULT_NET_METRICS: dict[str, Any] = {
    "delay_ms": 0.0,
    "jitter_ms": 0.0,
    "packet_loss_pct": 0.0,
    "packet_loss_percent": 0.0,
    "bandwidth": "unknown",
}


def default_net_metrics() -> dict[str, Any]:
    """Return the zero-condition network metrics used before the first message."""
    return dict(_DEFAULT_NET_METRICS)


def normalize_net_metrics(message: dict, include_raw: bool = True) -> dict[str, Any]:
    """
    Return flat network conditions with both packet-loss key spellings.

    Missing fields fall back to zero delay / loss and an "unknown" bandwidth,
    which is what students see before the network publisher's first message.
    """
    if not isinstance(message, dict):
        message = {}

    loss = message.get("packet_loss_pct", message.get("packet_loss_percent", 0.0))
    normalized = {
        "delay_ms": _as_float(message.get("delay_ms")),
        "jitter_ms": _as_float(message.get("jitter_ms")),
        "packet_loss_pct": _as_float(loss),
        "packet_loss_percent": _as_float(loss),
        "bandwidth": str(message.get("bandwidth", "unknown")),
    }
    if "timestamp" in message:
        normalized["timestamp"] = message["timestamp"]
    if include_raw:
        normalized["raw"] = message
    return normalized


def _as_float(value: Any) -> float:
    """Coerce a metric field to float, treating anything unparseable as zero."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
