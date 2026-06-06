"""Thread-safe rolling dashboard state."""
from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field
from threading import RLock
import time
from typing import Any


def normalize_group_id(group_id: str | int | None) -> str:
    """Return a stable groupN identifier for topics and dashboard state."""
    value = str(group_id or "group1").strip().lower()
    if value.startswith("group"):
        suffix = value[5:]
    else:
        suffix = value
    return f"group{suffix}" if suffix else "group1"


def _numbers(items: deque, key: str) -> list[float]:
    return [
        float(item[key])
        for item in items
        if item.get(key) is not None
    ]


def _average(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * percentile))))
    return ordered[index]


@dataclass
class GroupState:
    """Mutable state for one student group."""

    history: deque
    seen_samples: deque
    seen_sample_set: set[str] = field(default_factory=set)
    latest_metric: dict[str, Any] = field(default_factory=dict)
    frame_image: bytes | None = None
    frame_sequence: int = 0
    frame_updated_at: float | None = None
    total_frames: int = 0
    local_frames: int = 0
    remote_frames: int = 0
    total_latency: float = 0.0
    latency_count: int = 0
    total_displacement: float = 0.0
    displacement_count: int = 0
    min_latency: float | None = None
    max_latency: float | None = None
    min_displacement: float | None = None
    max_displacement: float | None = None
    cumulative_displacement: float = 0.0


class DashboardState:
    """Maintain one group's latest values and rolling history."""

    def __init__(self, max_history: int = 300, group_id: str = "1"):
        self.max_history = max(10, max_history)
        self.group_id = normalize_group_id(group_id)
        self._lock = RLock()
        self._group = self._new_group()
        self._gpu_metrics: dict[str, Any] = {}
        self._network_metrics: dict[str, Any] = {}
        self._phase = "unknown"
        self._gpu_history: deque = deque(maxlen=self.max_history)
        self._network_history: deque = deque(maxlen=self.max_history)
        self._kafka_status: dict[str, Any] = {
            "connected": False,
            "detail": "disabled",
            "last_message_at": None,
        }

    def update_app_metric(self, metric: dict[str, Any]) -> bool:
        """Record one scored frame, returning True when it is a new sample."""
        with self._lock:
            group = self._group
            clean = self._normalize_app_metric(metric)
            group.latest_metric = clean
            if clean.get("experiment_phase"):
                self._phase = clean["experiment_phase"]

            sample_key = self._sample_key(clean)
            if sample_key in group.seen_sample_set:
                return False

            if len(group.seen_samples) == group.seen_samples.maxlen:
                oldest = group.seen_samples.popleft()
                group.seen_sample_set.discard(oldest)
            group.seen_samples.append(sample_key)
            group.seen_sample_set.add(sample_key)

            group.history.append(clean)
            group.total_frames += 1

            mode = str(clean.get("processing_mode", "")).lower()
            if mode.startswith("local"):
                group.local_frames += 1
            elif mode == "remote":
                group.remote_frames += 1

            latency = clean.get("latency_ms")
            if latency is not None:
                latency = float(latency)
                group.total_latency += latency
                group.latency_count += 1
                group.min_latency = latency if group.min_latency is None else min(group.min_latency, latency)
                group.max_latency = latency if group.max_latency is None else max(group.max_latency, latency)

            displacement = clean.get("displacement_px")
            if displacement is not None:
                displacement = float(displacement)
                group.total_displacement += displacement
                group.displacement_count += 1
                group.min_displacement = (
                    displacement
                    if group.min_displacement is None
                    else min(group.min_displacement, displacement)
                )
                group.max_displacement = (
                    displacement
                    if group.max_displacement is None
                    else max(group.max_displacement, displacement)
                )

            cumulative = clean.get("cumulative_displacement_px")
            if cumulative is not None:
                group.cumulative_displacement = float(cumulative)
            return True

    def update_frame(self, metric: dict[str, Any], image: bytes) -> None:
        """Store a decoded JPEG and its scored metric record."""
        self.update_app_metric(metric)
        with self._lock:
            group = self._group
            group.frame_image = image
            group.frame_sequence += 1
            group.frame_updated_at = time.time()

    def update_gpu_metrics(self, metric: dict[str, Any]) -> None:
        with self._lock:
            clean = self._flatten_gpu_metrics(metric)
            clean.setdefault("timestamp", time.time())
            self._gpu_metrics = clean
            self._gpu_history.append(clean)

    def update_network_metrics(self, metric: dict[str, Any]) -> None:
        with self._lock:
            clean = deepcopy(metric)
            clean.setdefault("timestamp", time.time())
            self._network_metrics = clean
            self._network_history.append(clean)

    def update_phase(self, phase: str) -> None:
        with self._lock:
            self._phase = phase or "unknown"

    def update_kafka_status(self, connected: bool, detail: str) -> None:
        with self._lock:
            self._kafka_status = {
                "connected": connected,
                "detail": detail,
                "last_message_at": self._kafka_status.get("last_message_at"),
            }

    def touch_kafka(self) -> None:
        with self._lock:
            self._kafka_status["connected"] = True
            self._kafka_status["detail"] = "receiving metrics"
            self._kafka_status["last_message_at"] = time.time()

    def snapshot(self, include_history: bool = True) -> dict[str, Any]:
        """Return a JSON-ready immutable dashboard view."""
        with self._lock:
            group = self._group
            latency_values = _numbers(group.history, "latency_ms")
            displacement_values = _numbers(group.history, "displacement_px")
            dashboard_connected = (
                group.frame_updated_at is not None
                and time.time() - group.frame_updated_at < 5.0
            )

            history = {
                "frames": list(group.history) if include_history else [],
                "gpu": list(self._gpu_history) if include_history else [],
                "network": list(self._network_history) if include_history else [],
            }
            return {
                "group_id": self.group_id,
                "latest": deepcopy(group.latest_metric),
                "experiment_phase": self._phase,
                "frame": {
                    "sequence": group.frame_sequence,
                    "frame_number": group.latest_metric.get("frame_number"),
                    "url": (
                        f"/api/frame?v={group.frame_sequence}"
                        if group.frame_image is not None
                        else None
                    ),
                    "updated_at": group.frame_updated_at,
                },
                "latency": {
                    "rolling_average_ms": _average(latency_values[-20:]),
                    "min_ms": min(latency_values) if latency_values else None,
                    "max_ms": max(latency_values) if latency_values else None,
                    "p95_ms": _percentile(latency_values, 0.95),
                },
                "displacement": {
                    "rolling_average_px": _average(displacement_values[-20:]),
                },
                "infrastructure": {
                    "gpu": deepcopy(self._gpu_metrics),
                    "network": deepcopy(self._network_metrics),
                    "kafka": deepcopy(self._kafka_status),
                    "dashboard": {
                        "connected": dashboard_connected,
                        "detail": "receiving frames" if dashboard_connected else "waiting for frames",
                        "last_frame_at": group.frame_updated_at,
                    },
                },
                "summary": self._summary(group),
                "history": history,
                "updated_at": time.time(),
            }

    def history(self) -> dict[str, Any]:
        """Return only rolling chart history."""
        return self.snapshot()["history"]

    def frame_image(self) -> bytes | None:
        """Return the newest JPEG bytes."""
        with self._lock:
            return self._group.frame_image

    def reset(self) -> None:
        """Reset cumulative counters and history.

        The last received JPEG frame is preserved so the video panel stays live
        instead of going blank until the next frame arrives.
        """
        with self._lock:
            old = self._group
            fresh = self._new_group()
            fresh.frame_image = old.frame_image
            fresh.frame_sequence = old.frame_sequence
            fresh.frame_updated_at = old.frame_updated_at
            self._group = fresh

    def health(self) -> dict[str, Any]:
        """Return backend health and live source status."""
        with self._lock:
            return {
                "status": "ok",
                "group_id": self.group_id,
                "kafka": deepcopy(self._kafka_status),
                "max_history": self.max_history,
            }

    def _new_group(self) -> GroupState:
        return GroupState(
            history=deque(maxlen=self.max_history),
            seen_samples=deque(maxlen=self.max_history * 4),
        )

    def _normalize_app_metric(self, metric: dict[str, Any]) -> dict[str, Any]:
        clean = deepcopy(metric)
        clean["group_id"] = self.group_id
        clean.setdefault("timestamp", time.time())
        clean.setdefault("experiment_phase", "unknown")
        clean.setdefault("processing_mode", "unknown")
        return clean

    @staticmethod
    def _sample_key(metric: dict[str, Any]) -> str:
        return (
            f"{metric.get('group_id')}:{metric.get('frame_number')}:"
            f"{float(metric.get('timestamp', 0.0)):.6f}"
        )

    @staticmethod
    def _flatten_gpu_metrics(message: dict[str, Any]) -> dict[str, Any]:
        """Flatten Eldiyar's nested server metrics into the same keys used by SPAgentBase."""
        server = message.get("server", {})
        totals = message.get("totals", {})
        models = message.get("models", [])
        yolo = next((m for m in models if m.get("model_name") == "yolov10n"), {})
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
            "yolo_pending": yolo.get("pending_requests", 0),
            "yolo_queue_ms": yolo.get("avg_queue_time_ms", 0.0),
            "yolo_infer_ms": yolo.get("avg_compute_infer_ms", 0.0),
            "timestamp": message.get("timestamp", 0.0),
        }

    @staticmethod
    def _summary(group: GroupState) -> dict[str, Any]:
        total = group.total_frames
        return {
            "total_frames": total,
            "local_frames": group.local_frames,
            "remote_frames": group.remote_frames,
            "local_percentage": (group.local_frames / total * 100.0) if total else 0.0,
            "remote_percentage": (group.remote_frames / total * 100.0) if total else 0.0,
            "average_latency_ms": (
                group.total_latency / group.latency_count if group.latency_count else None
            ),
            "average_displacement_px": (
                group.total_displacement / group.displacement_count
                if group.displacement_count
                else None
            ),
            "cumulative_displacement_px": group.cumulative_displacement,
            "best_latency_ms": group.min_latency,
            "worst_latency_ms": group.max_latency,
            "best_displacement_px": group.min_displacement,
            "worst_displacement_px": group.max_displacement,
        }
