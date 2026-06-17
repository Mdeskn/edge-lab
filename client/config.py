"""
All configuration loaded from environment variables.
Import Config from here everywhere. Never read os.environ directly elsewhere.
"""
import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

TargetClassFilter = int | tuple[int, ...] | None


@dataclass
class Config:
    """Holds all application configuration loaded from environment variables."""

    group_id: str
    video_path: str
    ground_truth_path: str
    model_path: str
    triton_url: str
    triton_model_name: str
    remote_inference_url: str
    remote_jpeg_quality: int
    remote_inference_timeout: float
    remote_failure_cooldown_sec: float
    kafka_brokers: str
    kafka_app_topic: str        # APP_METRICS_TOPIC env var; fallback: "dnn_partition.client_metrics"
    kafka_gpu_topic: str
    kafka_net_topic: str
    kafka_phase_topic: str
    otlp_endpoint: str
    initial_processing_mode: str
    frame_interval_ms: int
    input_width: int
    input_height: int
    results_log_path: str
    results_by_phase_path: str
    queue_max_size: int
    conf_threshold: float
    target_class_id: TargetClassFilter
    target_conf_threshold: float
    sp_agent_interval_ms: int
    log_level: str
    auto_stop: bool
    phase_timeout_sec: float
    miss_penalty_px: float
    dashboard_enabled: bool
    dashboard_url: str
    dashboard_fps: float
    dashboard_jpeg_quality: int
    dashboard_frame_width: int
    sp_agent_debug_metrics: bool


def load_config() -> Config:
    """Load and return a Config instance from environment variables."""
    group_id = os.environ.get("GROUP_ID", "1")
    target_class_value = os.environ.get("TARGET_CLASS_ID", "").strip()
    conf_threshold = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.3"))
    results_log_path = os.environ.get("RESULTS_LOG_PATH", "results.csv")
    results_by_phase_path = os.environ.get("RESULTS_BY_PHASE_PATH", "").strip()
    if not results_by_phase_path:
        base, ext = os.path.splitext(results_log_path)
        results_by_phase_path = f"{base}_by_phase{ext or '.csv'}"

    return Config(
        group_id=group_id,
        video_path=os.environ["VIDEO_PATH"],
        ground_truth_path=os.environ["GROUND_TRUTH_PATH"],
        model_path=os.environ["MODEL_PATH"],
        triton_url=os.environ.get("TRITON_URL", ""),
        triton_model_name=os.environ.get("TRITON_MODEL_NAME", "yolov10n"),
        remote_inference_url=os.environ.get("REMOTE_INFERENCE_URL", "").strip(),
        remote_jpeg_quality=int(os.environ.get("REMOTE_JPEG_QUALITY", "80")),
        remote_inference_timeout=float(os.environ.get("REMOTE_INFERENCE_TIMEOUT_SEC", "1.5")),
        remote_failure_cooldown_sec=float(os.environ.get("REMOTE_FAILURE_COOLDOWN_SEC", "3.0")),
        kafka_brokers=os.environ.get("KAFKA_BROKERS", ""),
        kafka_app_topic=os.environ.get("APP_METRICS_TOPIC", "").strip()
            or "dnn_partition.client_metrics",
        kafka_gpu_topic=os.environ.get("KAFKA_GPU_TOPIC", "dnn_partition.server_metrics"),
        kafka_net_topic=os.environ.get("KAFKA_NET_TOPIC", "edgelab.network.metrics"),
        kafka_phase_topic=os.environ.get("KAFKA_PHASE_TOPIC", "edgelab.phase"),
        otlp_endpoint=os.environ.get("OTLP_ENDPOINT", ""),
        initial_processing_mode=os.environ.get("INITIAL_PROCESSING_MODE", "local"),
        frame_interval_ms=int(os.environ.get("FRAME_INTERVAL_MS", "100")),
        input_width=int(os.environ.get("MODEL_INPUT_WIDTH", "640")),
        input_height=int(os.environ.get("MODEL_INPUT_HEIGHT", "640")),
        results_log_path=results_log_path,
        results_by_phase_path=results_by_phase_path,
        queue_max_size=int(os.environ.get("QUEUE_MAX_SIZE", "10")),
        conf_threshold=conf_threshold,
        target_class_id=_parse_target_class_filter(target_class_value),
        target_conf_threshold=float(
            os.environ.get("TARGET_CONFIDENCE_THRESHOLD", str(conf_threshold))
        ),
        sp_agent_interval_ms=int(os.environ.get("SP_AGENT_INTERVAL_MS", "500")),
        log_level=os.environ.get("LOG_LEVEL", "INFO"),
        auto_stop=os.environ.get("AUTO_STOP", "true").lower() == "true",
        phase_timeout_sec=float(os.environ.get("PHASE_TIMEOUT_SEC", "300")),
        miss_penalty_px=float(os.environ.get("MISS_PENALTY_PX", "100.0")),
        dashboard_enabled=os.environ.get("DASHBOARD_ENABLED", "false").lower() == "true",
        dashboard_url=os.environ.get("DASHBOARD_URL", "http://localhost:8080"),
        dashboard_fps=float(os.environ.get("DASHBOARD_FPS", "5")),
        dashboard_jpeg_quality=int(os.environ.get("DASHBOARD_JPEG_QUALITY", "70")),
        dashboard_frame_width=int(os.environ.get("DASHBOARD_FRAME_WIDTH", "960")),
        sp_agent_debug_metrics=os.environ.get("SP_AGENT_DEBUG_METRICS", "false").lower() == "true",
    )


def _parse_target_class_filter(value: str) -> TargetClassFilter:
    """Parse TARGET_CLASS_ID as one class id or a comma-separated class-id list."""
    if not value:
        return None
    parts = [part.strip() for part in value.split(",") if part.strip()]
    ids = tuple(int(part) for part in parts)
    if len(ids) == 1:
        return ids[0]
    return ids
