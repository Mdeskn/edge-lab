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


def _mode_latency_summary(items: deque, mode: str, source: str = "scored") -> dict[str, Any]:
    mode_items = [
        item for item in items
        if str(item.get("processing_mode", "")).lower() == mode
        and item.get("latency_ms") is not None
    ]
    values = [float(item["latency_ms"]) for item in mode_items]
    latest = mode_items[-1] if mode_items else {}
    return {
        "latest_ms": values[-1] if values else None,
        "rolling_average_ms": _average(values[-20:]),
        "min_ms": min(values) if values else None,
        "max_ms": max(values) if values else None,
        "p95_ms": _percentile(values, 0.95),
        "sample_count": len(values),
        "frame_number": latest.get("frame_number"),
        "timestamp": latest.get("timestamp"),
        "source": source,
    }


def _latency_summary_for_mode(group: "GroupState", mode: str) -> dict[str, Any]:
    scored = _mode_latency_summary(group.history, mode, source="scored")
    probes = group.latency_probes.get(mode, deque())
    probe = _mode_latency_summary(probes, mode, source="probe")

    latest_mode = str(group.latest_metric.get("processing_mode", "")).lower()
    active = latest_mode == mode or (mode == "local" and latest_mode.startswith("local"))
    if active and scored["sample_count"]:
        return scored
    if probe["sample_count"]:
        return probe
    return scored


@dataclass
class GroupState:
    """Mutable state for one student group."""

    history: deque
    seen_samples: deque
    seen_sample_set: set[str] = field(default_factory=set)
    latency_probes: dict[str, deque] = field(default_factory=dict)
    latest_metric: dict[str, Any] = field(default_factory=dict)
    frame_image: bytes | None = None
    frame_sequence: int = 0
    frame_updated_at: float | None = None
    preview_metric: dict[str, Any] = field(default_factory=dict)
    preview_image: bytes | None = None
    preview_sequence: int = 0
    preview_updated_at: float | None = None
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

    def __init__(
        self,
        max_history: int = 300,
        group_id: str = "1",
        placement_control_enabled: bool = False,
        placement_control_topic: str = "",
    ):
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
        self._placement_control: dict[str, Any] = {
            "enabled": placement_control_enabled,
            "topic": placement_control_topic,
            "requested_mode": None,
            "status": "enabled" if placement_control_enabled else "disabled",
            "detail": (
                "manual controls available"
                if placement_control_enabled
                else "SP-agent automatic decisions"
            ),
            "updated_at": None,
        }
        self._collection_state: dict | None = None
        self._prev_collection_state_str: str | None = None
        self._phase_summary: dict[str, dict] = {}
        self._cycle_duration_refresh_requested = False
        self._cycle_command: str | None = None

    def update_app_metric(self, metric: dict[str, Any]) -> bool:
        """Record one scored frame, returning True when it is a new sample."""
        with self._lock:
            group = self._group
            clean = self._normalize_app_metric(metric)
            if clean.get("event_type") == "latency_probe":
                return self._update_latency_probe(group, clean)

            group.latest_metric = clean

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

            cs = clean.get("collection_state")
            if cs is not None:
                curr = cs.get("state")
                if self._prev_collection_state_str == "complete" and curr == "armed":
                    self._phase_summary = {}
                self._prev_collection_state_str = curr
                self._collection_state = cs

            self._update_phase_summary_locked(clean)
            return True

    def update_frame(self, metric: dict[str, Any], image: bytes) -> None:
        """Store a decoded JPEG and its scored metric record."""
        self.update_app_metric(metric)
        with self._lock:
            group = self._group
            group.frame_image = image
            group.frame_sequence += 1
            group.frame_updated_at = time.time()

    def update_preview(self, metric: dict[str, Any], image: bytes) -> None:
        """Store the latest real-time preview without affecting score totals."""
        with self._lock:
            group = self._group
            group.preview_metric = deepcopy(metric)
            group.preview_image = image
            group.preview_sequence += 1
            group.preview_updated_at = time.time()

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

    def update_placement_control(
        self,
        *,
        requested_mode: str | None = None,
        status: str | None = None,
        detail: str | None = None,
    ) -> None:
        with self._lock:
            if requested_mode is not None:
                self._placement_control["requested_mode"] = requested_mode
            if status is not None:
                self._placement_control["status"] = status
            if detail is not None:
                self._placement_control["detail"] = detail
            self._placement_control["updated_at"] = time.time()

    def get_collection_state(self) -> dict | None:
        with self._lock:
            return deepcopy(self._collection_state) if self._collection_state else None

    def set_collection_state(self, state: dict) -> None:
        with self._lock:
            self._collection_state = state

    def reset_phase_summary(self) -> None:
        with self._lock:
            self._phase_summary = {}

    def request_cycle_duration_refresh(self) -> None:
        """Request that the client reload its scenario-derived cycle duration."""
        with self._lock:
            self._cycle_duration_refresh_requested = True

    def consume_cycle_duration_refresh_request(self) -> bool:
        """Return and clear the pending client cycle-duration refresh request."""
        with self._lock:
            requested = self._cycle_duration_refresh_requested
            self._cycle_duration_refresh_requested = False
            return requested

    def request_cycle_command(self, action: str) -> None:
        """Queue a cycle command for the client to receive with its next metric."""
        with self._lock:
            self._cycle_command = action

    def consume_cycle_command(self) -> str | None:
        """Return and clear the pending client cycle command."""
        with self._lock:
            command = self._cycle_command
            self._cycle_command = None
            return command

    def _update_phase_summary_locked(self, metric: dict[str, Any]) -> None:
        """Accumulate per-phase totals. Caller must hold _lock."""
        phase = metric.get("experiment_phase")
        if not phase or phase == "unknown":
            return
        if phase not in self._phase_summary:
            self._phase_summary[phase] = {
                "frames": 0,
                "total_displacement": 0.0,
                "total_latency_ms": 0.0,
                "latency_frames": 0,
                "total_jitter_ms": 0.0,
                "jitter_frames": 0,
                "deadline_misses": 0,
            }
        s = self._phase_summary[phase]
        disp = metric.get("displacement_px")
        lat = metric.get("latency_ms")
        jit = metric.get("jitter_ms")
        miss = metric.get("deadline_miss")
        if disp is not None:
            s["frames"] += 1
            s["total_displacement"] += float(disp)
        if lat is not None:
            s["total_latency_ms"] += float(lat)
            s["latency_frames"] += 1
        if jit is not None:
            s["total_jitter_ms"] += float(jit)
            s["jitter_frames"] += 1
        if miss:
            s["deadline_misses"] += 1

    def snapshot(self, include_history: bool = True) -> dict[str, Any]:
        """Return a JSON-ready immutable dashboard view."""
        with self._lock:
            group = self._group
            latency_values = _numbers(group.history, "latency_ms")
            displacement_values = _numbers(group.history, "displacement_px")
            display_updated_at = group.preview_updated_at or group.frame_updated_at
            dashboard_connected = (
                display_updated_at is not None
                and time.time() - display_updated_at < 5.0
            )
            display_metric = group.preview_metric or group.latest_metric
            display_sequence = (
                group.preview_sequence
                if group.preview_image is not None
                else group.frame_sequence
            )
            has_display_frame = (
                group.preview_image is not None or group.frame_image is not None
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
                    "sequence": display_sequence,
                    "frame_number": display_metric.get("frame_number"),
                    "url": (
                        "/api/video-stream"
                        if has_display_frame
                        else None
                    ),
                    "updated_at": display_updated_at,
                    "true_x": display_metric.get("true_x"),
                    "true_y": display_metric.get("true_y"),
                    "predicted_x": display_metric.get("predicted_x"),
                    "predicted_y": display_metric.get("predicted_y"),
                    "prediction_frame_number": display_metric.get(
                        "prediction_frame_number"
                    ),
                },
                "latency": {
                    "rolling_average_ms": _average(latency_values[-20:]),
                    "min_ms": min(latency_values) if latency_values else None,
                    "max_ms": max(latency_values) if latency_values else None,
                    "p95_ms": _percentile(latency_values, 0.95),
                    "local": _latency_summary_for_mode(group, "local"),
                    "remote": _latency_summary_for_mode(group, "remote"),
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
                        "last_frame_at": display_updated_at,
                    },
                },
                "summary": self._summary(group),
                "placement_control": deepcopy(self._placement_control),
                "collection_state": deepcopy(self._collection_state) if self._collection_state else None,
                "phase_summary": deepcopy(self._phase_summary),
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

    def frame_snapshot(self) -> tuple[int, bytes | None]:
        """Return the current frame sequence and immutable JPEG bytes."""
        with self._lock:
            if self._group.preview_image is not None:
                return self._group.preview_sequence, self._group.preview_image
            return self._group.frame_sequence, self._group.frame_image

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
            fresh.preview_metric = old.preview_metric
            fresh.preview_image = old.preview_image
            fresh.preview_sequence = old.preview_sequence
            fresh.preview_updated_at = old.preview_updated_at
            self._group = fresh

    def health(self) -> dict[str, Any]:
        """Return backend health and live source status."""
        with self._lock:
            return {
                "status": "ok",
                "group_id": self.group_id,
                "kafka": deepcopy(self._kafka_status),
                "placement_control": deepcopy(self._placement_control),
                "max_history": self.max_history,
            }

    def _new_group(self) -> GroupState:
        group = GroupState(
            history=deque(maxlen=self.max_history),
            seen_samples=deque(maxlen=self.max_history * 4),
        )
        group.latency_probes = {
            "local": deque(maxlen=self.max_history),
            "remote": deque(maxlen=self.max_history),
        }
        return group

    def _update_latency_probe(self, group: GroupState, metric: dict[str, Any]) -> bool:
        mode = str(
            metric.get("probe_mode") or metric.get("processing_mode", "")
        ).strip().lower()
        if mode not in ("local", "remote"):
            return False
        if metric.get("latency_ms") is None:
            return False

        metric["probe_mode"] = mode
        metric["processing_mode"] = mode
        group.latency_probes.setdefault(
            mode,
            deque(maxlen=self.max_history),
        ).append(metric)
        return True

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
        """Flatten GPU metrics into a consistent flat dict for the dashboard.

        Handles two formats on the same Kafka topic:
          - Nested (Eldiyar's publisher): has "server", "totals", "models" keys.
          - Flat (gpu_metrics_publisher.py): has "gpu_utilization_pct", etc.
        """
        if "server" in message or "models" in message:
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
                "yolo_success_rps": yolo.get("success_rps", 0.0),
                "yolo_inference_rps": yolo.get("inference_rps", 0.0),
                "yolo_queue_ms": yolo.get("avg_queue_time_ms", 0.0),
                "yolo_input_ms": yolo.get("avg_compute_input_ms", 0.0),
                "yolo_infer_ms": yolo.get("avg_compute_infer_ms", 0.0),
                "yolo_output_ms": yolo.get("avg_compute_output_ms", 0.0),
                "timestamp": message.get("timestamp", 0.0),
            }

        rps = message.get("triton_requests_per_sec", 0.0)
        return {
            "gpu_util_pct": message.get("gpu_utilization_pct", 0.0),
            "gpu_freq_mhz": 0.0,
            "gpu_temp_c": message.get("gpu_temperature_c", 0.0),
            "gpu_mem_used_mb": message.get("gpu_memory_used_mb", 0.0),
            "gpu_mem_total_mb": message.get("gpu_memory_total_mb", 0.0),
            "cpu_util_pct": 0.0,
            "mem_util_pct": 0.0,
            "power_w": message.get("gpu_power_draw_w", 0.0),
            "total_rps": rps,
            "total_success_rps": rps,
            "total_failure_rps": 0.0,
            "total_pending": 0,
            "yolo_pending": 0,
            "yolo_success_rps": rps,
            "yolo_inference_rps": rps,
            "yolo_queue_ms": message.get("triton_queue_duration_ms", 0.0),
            "yolo_input_ms": 0.0,
            "yolo_infer_ms": message.get("triton_inference_duration_ms", 0.0),
            "yolo_output_ms": 0.0,
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
