"""Thread-safe rolling dashboard state."""
from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field
from threading import RLock
import time
from typing import Any


GROUPS = ("group1", "group2", "group3", "group4")


def normalize_group_id(group_id: str | int | None) -> str:
    """Return a stable groupN identifier for URLs, topics, and client state."""
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
    """Maintain latest values and rolling history for REST and WebSocket clients."""

    def __init__(self, max_history: int = 300):
        self.max_history = max(10, max_history)
        self._lock = RLock()
        self._groups = {
            group_id: GroupState(
                history=deque(maxlen=self.max_history),
                seen_samples=deque(maxlen=self.max_history * 4),
            )
            for group_id in GROUPS
        }
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

    def update_app_metric(self, group_id: str, metric: dict[str, Any]) -> bool:
        """Record one scored frame, returning True when it is a new sample."""
        normalized_group = normalize_group_id(group_id)
        with self._lock:
            group = self._get_group(normalized_group)
            clean = self._normalize_app_metric(normalized_group, metric)
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

    def update_frame(self, group_id: str, metric: dict[str, Any], image: bytes) -> None:
        """Store a decoded JPEG and its scored metric record."""
        normalized_group = normalize_group_id(group_id)
        self.update_app_metric(normalized_group, metric)
        with self._lock:
            group = self._get_group(normalized_group)
            group.frame_image = image
            group.frame_sequence += 1
            group.frame_updated_at = time.time()

    def update_gpu_metrics(self, metric: dict[str, Any]) -> None:
        with self._lock:
            clean = deepcopy(metric)
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

    def snapshot(self, group_id: str = "group1", include_history: bool = True) -> dict[str, Any]:
        """Return a JSON-ready immutable view of one group's dashboard."""
        normalized_group = normalize_group_id(group_id)
        with self._lock:
            group = self._get_group(normalized_group)
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
                "group_id": normalized_group,
                "groups": list(GROUPS),
                "latest": deepcopy(group.latest_metric),
                "experiment_phase": self._phase,
                "frame": {
                    "sequence": group.frame_sequence,
                    "frame_number": group.latest_metric.get("frame_number"),
                    "url": (
                        f"/api/frame/{normalized_group}?v={group.frame_sequence}"
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

    def history(self, group_id: str = "group1") -> dict[str, Any]:
        """Return only rolling chart history."""
        return self.snapshot(group_id)["history"]

    def frame_image(self, group_id: str) -> bytes | None:
        """Return the newest JPEG bytes for one group."""
        with self._lock:
            return self._get_group(normalize_group_id(group_id)).frame_image

    def health(self) -> dict[str, Any]:
        """Return backend health and live source status."""
        with self._lock:
            return {
                "status": "ok",
                "groups": list(GROUPS),
                "kafka": deepcopy(self._kafka_status),
                "max_history": self.max_history,
            }

    def _get_group(self, group_id: str) -> GroupState:
        if group_id not in self._groups:
            self._groups[group_id] = GroupState(
                history=deque(maxlen=self.max_history),
                seen_samples=deque(maxlen=self.max_history * 4),
            )
        return self._groups[group_id]

    @staticmethod
    def _normalize_app_metric(group_id: str, metric: dict[str, Any]) -> dict[str, Any]:
        clean = deepcopy(metric)
        clean["group_id"] = group_id
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
