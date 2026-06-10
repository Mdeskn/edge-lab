"""
Entry point. Wires all components together and starts all threads.
"""
import json
import logging
import os
import queue
import threading

from config import load_config, Config
from shared_state import SharedState
from inference.local_server import LocalServer
from inference.remote_client import RemoteClient
from metrics.dashboard_publisher import DashboardPublisher
from metrics.kafka_publisher import AppMetricsPublisher
from metrics.telemetry import setup_telemetry
from threads.frame_reader import FrameReader
from threads.dispatcher import Dispatcher
from threads.scorer import Scorer
from student.sp_agent import SPAgent

logger = logging.getLogger(__name__)


def wait_for_first_phase(config: Config) -> tuple:
    """
    Block until a phase message arrives on the Kafka phase topic.
    Returns (starting_phase_name, kafka_consumer).
    The consumer is returned so the monitor loop can reuse it.
    """
    from confluent_kafka import Consumer

    consumer = Consumer({
        "bootstrap.servers": config.kafka_brokers,
        "group.id": f"phase-monitor-group{config.group_id}",
        "auto.offset.reset": "latest",
    })
    consumer.subscribe([config.kafka_phase_topic])
    logger.info("Waiting for experiment phase to start...")

    while True:
        msg = consumer.poll(timeout=2.0)
        if msg is None:
            continue
        if msg.error():
            logger.warning("Phase consumer error: %s", msg.error())
            continue
        try:
            data = json.loads(msg.value().decode("utf-8"))
            phase = data.get("phase")
            if phase:
                logger.info("Experiment started. First phase: %s", phase)
                return phase, consumer
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue


def monitor_phases(consumer, starting_phase: str, shared_state: SharedState):
    """
    Monitor phase transitions until one full cycle completes.
    A full cycle means all four phases have been observed and the
    current phase transitions back to the starting phase.

    Calls shared_state.request_shutdown() when complete.
    """
    ALL_PHASES = {"baseline", "gpu_load", "bandwidth_50", "mixed"}
    phases_seen = {starting_phase}
    current_phase = starting_phase

    logger.info(
        "Monitoring phases. Will stop after one full cycle (started on: %s).",
        starting_phase,
    )

    while not shared_state.is_shutdown_requested():
        msg = consumer.poll(timeout=1.0)
        if msg is None:
            continue
        if msg.error():
            continue

        try:
            data = json.loads(msg.value().decode("utf-8"))
            new_phase = data.get("phase")
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue

        if new_phase and new_phase != current_phase:
            logger.info("Phase transition: %s -> %s", current_phase, new_phase)
            current_phase = new_phase

            # Check if we completed a full cycle:
            # All 4 phases seen AND we returned to the starting phase
            if current_phase == starting_phase and phases_seen >= ALL_PHASES:
                logger.info(
                    "Full cycle complete (all phases observed, returned to %s). "
                    "Stopping experiment.",
                    starting_phase,
                )
                shared_state.request_shutdown()
                break

            phases_seen.add(current_phase)

    try:
        consumer.close()
    except Exception:
        pass


def main() -> None:
    """
    Start-up sequence:

    1.  Load config
    2.  Configure logging
    3.  Print startup banner
    4.  Set up OpenTelemetry
    5.  Initialise SharedState
    6.  Initialise inference backends
    7.  Initialise Kafka publisher
    8.  Open results CSV
    9.  Build queues
    10. Construct all four thread objects
    11. Phase-aware auto-stop (wait for first phase if enabled)
    12. Start threads
    13. Join threads (KeyboardInterrupt: graceful shutdown)
    14. Print per-phase and overall summary
    """
    # 1. Config
    config = load_config()

    # 2. Logging
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    # 3. Banner
    logger.info("=" * 60)
    logger.info("Edge Computing Lab: Group %s", config.group_id)
    logger.info("  video_path           : %s", config.video_path)
    logger.info("  ground_truth_path    : %s", config.ground_truth_path)
    logger.info("  model_path           : %s", config.model_path)
    logger.info("  remote_inference_url : %s", config.remote_inference_url or "(not set)")
    logger.info("  remote_jpeg_quality  : %d", config.remote_jpeg_quality)
    logger.info("  triton_url           : %s", config.triton_url or "(not set, legacy only)")
    logger.info("  kafka_brokers        : %s", config.kafka_brokers or "(not set)")
    logger.info("  otlp_endpoint        : %s", config.otlp_endpoint or "(not set)")
    logger.info("  initial_mode         : %s", config.initial_processing_mode)
    logger.info("  frame_interval_ms    : %d", config.frame_interval_ms)
    logger.info("  display_output       : %s", config.display_output)
    logger.info("  results_log_path     : %s", config.results_log_path)
    logger.info("  conf_threshold       : %.2f", config.conf_threshold)
    logger.info("  target_class_id      : %s", config.target_class_id)
    logger.info("  target_conf_threshold: %.2f", config.target_conf_threshold)
    logger.info("  sp_agent_interval_ms : %d", config.sp_agent_interval_ms)
    logger.info("  auto_stop            : %s", config.auto_stop)
    logger.info("  miss_penalty_px      : %.1f", config.miss_penalty_px)
    logger.info("  dashboard_enabled    : %s", config.dashboard_enabled)
    logger.info("  dashboard_url        : %s", config.dashboard_url)
    logger.info("=" * 60)

    # 4. OpenTelemetry
    tracer = setup_telemetry(
        config.otlp_endpoint,
        f"edge-lab-client-group{config.group_id}",
    )

    # 5. SharedState
    shared_state = SharedState(initial_mode=config.initial_processing_mode)

    # 6a. LocalServer: fails fast if model is missing
    local_server = LocalServer(
        config.model_path,
        config.conf_threshold,
        config.target_class_id,
        config.target_conf_threshold,
    )

    # 6b. RemoteClient: optional, None when both remote endpoints are empty
    remote_client: RemoteClient | None = None
    if config.remote_inference_url or config.triton_url:
        remote_client = RemoteClient(
            config.triton_url,
            config.triton_model_name,
            config.conf_threshold,
            config.target_class_id,
            config.target_conf_threshold,
            timeout=config.remote_inference_timeout,
            remote_inference_url=config.remote_inference_url,
            jpeg_quality=config.remote_jpeg_quality,
        )
    else:
        logger.warning("REMOTE_INFERENCE_URL/TRITON_URL not set: running in local-only mode")

    # 7. Kafka publisher
    kafka_publisher = AppMetricsPublisher(
        config.kafka_brokers, config.kafka_app_topic, config.group_id
    )
    dashboard_publisher = DashboardPublisher(config)

    # 8. Results CSV
    results_dir = os.path.dirname(config.results_log_path)
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
    results_file = open(config.results_log_path, "a", newline="")

    # 9. Queues
    reader_queue: queue.Queue = queue.Queue(maxsize=config.queue_max_size)
    scorer_queue: queue.Queue = queue.Queue(maxsize=config.queue_max_size)

    # 10. Thread objects
    frame_reader = FrameReader(config, shared_state, reader_queue)
    dispatcher = Dispatcher(
        config, shared_state, local_server, remote_client,
        reader_queue, scorer_queue, tracer,
    )
    scorer = Scorer(
        config,
        shared_state,
        scorer_queue,
        kafka_publisher,
        dashboard_publisher,
        results_file,
    )
    sp_agent = SPAgent(config, shared_state)

    # 11. Phase-aware auto-stop
    phase_consumer = None
    starting_phase = None

    if config.auto_stop and config.kafka_brokers:
        starting_phase, phase_consumer = wait_for_first_phase(config)
        shared_state.update_experiment_phase(starting_phase)
    elif config.auto_stop and not config.kafka_brokers:
        logger.warning(
            "AUTO_STOP is enabled but KAFKA_BROKERS is empty. "
            "Cannot monitor phases. Running until Ctrl+C instead."
        )

    # 12. Start threads
    threads = [
        threading.Thread(target=frame_reader.run, name="FrameReader", daemon=False),
        threading.Thread(target=dispatcher.run, name="Dispatcher", daemon=False),
        threading.Thread(target=scorer.run, name="Scorer", daemon=False),
        threading.Thread(target=sp_agent.run, name="SPAgent", daemon=False),
    ]
    for t in threads:
        t.start()

    # 13. Wait for completion
    try:
        if phase_consumer and starting_phase:
            monitor_phases(phase_consumer, starting_phase, shared_state)
        else:
            for t in threads:
                t.join()
    except KeyboardInterrupt:
        logger.info("Shutdown requested (KeyboardInterrupt)")
        shared_state.request_shutdown()

    for t in threads:
        t.join(timeout=5.0)

    kafka_publisher.flush()
    dashboard_publisher.close()
    results_file.close()

    # 14. Per-phase and overall summary
    phase_summary = shared_state.get_phase_summary()
    score_summary = shared_state.get_score_summary()

    logger.info("=" * 60)
    logger.info("Experiment complete. Results by phase:")
    logger.info("-" * 60)
    for phase_name in ["baseline", "gpu_load", "bandwidth_50", "mixed"]:
        ps = phase_summary.get(phase_name, {"total_displacement": 0, "frames": 0})
        frames = ps["frames"]
        total = ps["total_displacement"]
        avg = total / frames if frames > 0 else 0.0
        logger.info(
            "  %-16s  avg displacement: %7.1f px  (%d frames)",
            phase_name, avg, frames,
        )
    logger.info("-" * 60)
    logger.info(
        "  %-16s  avg displacement: %7.2f px  (%d frames)",
        "overall",
        score_summary["average_displacement"],
        score_summary["frames_processed"],
    )
    logger.info(
        "  %-16s  %7.1f px  (your score, lower is better)",
        "cumulative",
        score_summary["cumulative_displacement"],
    )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
