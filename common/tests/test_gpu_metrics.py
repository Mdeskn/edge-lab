"""The shared metric normalizers.

The client and the dashboard each used to carry their own copy of this logic
and the copies had already drifted (the dashboard's flat branch had lost the
ResNet fields), so students and the dashboard could disagree about the same
Kafka message.
"""
from common.gpu_metrics import (
    GPU_METRIC_DEFAULTS,
    default_net_metrics,
    normalize_gpu_metrics,
    normalize_net_metrics,
)

NESTED = {
    "timestamp": 1700.0,
    "server": {
        "gpu_util_percent": 85.5,
        "gpu_temp_c": 71.0,
        "gpu_mem_used_mb": 4096.0,
        "cpu_util_percent": 30.0,
        "power_w": 210.0,
    },
    "totals": {"total_rps": 42.0, "total_pending_requests": 7},
    "models": [
        {"model_name": "yolov10n", "avg_queue_time_ms": 12.5, "pending_requests": 3},
        {"model_name": "resnet50_full", "avg_queue_time_ms": 4.0, "success_rps": 9.0},
    ],
}

FLAT = {
    "timestamp": 1800.0,
    "gpu_utilization_pct": 55.0,
    "gpu_temperature_c": 60.0,
    "gpu_memory_used_mb": 2048.0,
    "gpu_power_draw_w": 150.0,
    "triton_requests_per_sec": 30.0,
    "triton_queue_duration_ms": 8.0,
    "triton_inference_duration_ms": 5.0,
}


def test_nested_format_is_flattened() -> None:
    m = normalize_gpu_metrics(NESTED)
    assert m["gpu_util_pct"] == 85.5
    assert m["gpu_temp_c"] == 71.0
    assert m["cpu_util_pct"] == 30.0
    assert m["total_rps"] == 42.0
    assert m["total_pending"] == 7
    assert m["yolo_queue_ms"] == 12.5
    assert m["yolo_pending"] == 3
    assert m["resnet_queue_ms"] == 4.0
    assert m["resnet_success_rps"] == 9.0


def test_flat_format_is_mapped_onto_the_same_keys() -> None:
    m = normalize_gpu_metrics(FLAT)
    assert m["gpu_util_pct"] == 55.0
    assert m["gpu_temp_c"] == 60.0
    assert m["power_w"] == 150.0
    assert m["total_rps"] == 30.0
    assert m["yolo_queue_ms"] == 8.0
    assert m["yolo_infer_ms"] == 5.0


def test_both_formats_produce_the_same_key_set() -> None:
    """Students' .get() calls must work regardless of which publisher runs."""
    assert set(normalize_gpu_metrics(NESTED)) == set(normalize_gpu_metrics(FLAT))


def test_every_documented_key_is_always_present() -> None:
    for message in (NESTED, FLAT, {}, {"unexpected": 1}):
        m = normalize_gpu_metrics(message)
        missing = set(GPU_METRIC_DEFAULTS) - set(m)
        assert not missing, missing


def test_raw_can_be_omitted() -> None:
    """The dashboard drops the original message to keep payloads small."""
    assert "raw" in normalize_gpu_metrics(NESTED, include_raw=True)
    assert "raw" not in normalize_gpu_metrics(NESTED, include_raw=False)


def test_timestamp_is_carried_through() -> None:
    assert normalize_gpu_metrics(NESTED)["timestamp"] == 1700.0
    assert normalize_gpu_metrics(FLAT)["timestamp"] == 1800.0


def test_unparseable_message_does_not_raise() -> None:
    assert normalize_gpu_metrics(None)["gpu_util_pct"] == 0.0


# --- Network conditions --------------------------------------------------


def test_net_metrics_populate_both_packet_loss_spellings() -> None:
    m = normalize_net_metrics({"packet_loss_pct": 2.5})
    assert m["packet_loss_pct"] == 2.5
    assert m["packet_loss_percent"] == 2.5

    m = normalize_net_metrics({"packet_loss_percent": 1.5})
    assert m["packet_loss_pct"] == 1.5
    assert m["packet_loss_percent"] == 1.5


def test_net_metrics_default_to_clean_conditions() -> None:
    m = normalize_net_metrics({})
    assert m["delay_ms"] == 0.0
    assert m["jitter_ms"] == 0.0
    assert m["packet_loss_pct"] == 0.0
    assert m["bandwidth"] == "unknown"


def test_net_metric_non_numeric_field_becomes_zero() -> None:
    assert normalize_net_metrics({"delay_ms": "n/a"})["delay_ms"] == 0.0


def test_default_net_metrics_are_a_fresh_copy() -> None:
    first = default_net_metrics()
    first["delay_ms"] = 99.0
    assert default_net_metrics()["delay_ms"] == 0.0
