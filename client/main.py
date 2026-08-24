"""
Entry point. Wires all components together and starts all threads.
"""
import csv
import logging
import os
import queue
import threading
import time

from common.phases import (
    CYCLE_END_PHASE,
    CYCLE_START_PHASE,
    EXPERIMENT_PHASES,
)
from config import Config, load_config
from inference.local_server import LocalServer
from inference.remote_client import RemoteClient
from metrics.dashboard_publisher import DashboardPublisher
from metrics.kafka_publisher import AppMetricsPublisher
from metrics.telemetry import setup_telemetry
from scenario_parser import load_cycle_duration_sec
from shared_state import CollectionState, SharedState
from threads.dispatcher import Dispatcher
from threads.frame_reader import FrameReader
from threads.latency_probe import LatencyProbe
from threads.scorer import Scorer

logger = logging.getLogger(__name__)


def run_cycle_monitor(config: Config, shared_state: SharedState) -> None:
    """
    Daemon thread that drives the collection state machine.

    Polls shared_state.get_experiment_phase() to detect cycle boundaries
    (transitions into cycle_start) and drives DISCONNECTED → ARMED → COLLECTING
    → COMPLETE state transitions.
    """
    logger.info("CycleMonitor started (sync_mode=%s)", config.sync_mode)

    effective_mode = config.sync_mode

    if effective_mode == "wait_for_cycle":
        shared_state.transition_to_armed()
        shared_state.request_start_on_next_cycle()
    elif effective_mode == "manual":
        shared_state.transition_to_armed()

    if effective_mode == "off":
        shared_state.transition_to_armed()
        shared_state.request_start_on_next_cycle()
        shared_state.transition_to_collecting(time.time())
        logger.info("SYNC_MODE=off: collecting indefinitely, no cycle detection")
        return

    poll_interval = 0.5
    cycle_timeout_sec = config.phase_timeout_sec if config.phase_timeout_sec > 0 else None

    while not shared_state.is_shutdown_requested():
        time.sleep(poll_interval)

        current_phase = shared_state.get_experiment_phase()
        previous_phase = shared_state.get_previous_phase()
        collection_state = shared_state.get_collection_state()

        if current_phase != previous_phase:
            shared_state.set_previous_phase(current_phase)
            logger.info(
                "Phase transition: %s -> %s (state=%s)",
                previous_phase, current_phase, collection_state.value,
            )

        # ARMED → COLLECTING on cycle boundary (entering cycle_start from another phase)
        if collection_state == CollectionState.ARMED:
            if (
                current_phase == CYCLE_START_PHASE
                and previous_phase is not None
                and previous_phase != CYCLE_START_PHASE
            ):
                if shared_state.transition_to_collecting(time.time()):
                    logger.info("cycle_start detected, COLLECTING started")
            continue

        # COLLECTING: track phases, detect cycle end, handle timeout
        if collection_state == CollectionState.COLLECTING:
            shared_state.add_phase_seen(current_phase)

            phase_based_done = (current_phase == CYCLE_END_PHASE)
            if phase_based_done:
                score_summary = shared_state.get_score_summary()
                final_score = float(score_summary.get("cumulative_displacement", 0.0))
                shared_state.transition_to_complete(time.time(), final_score)
                logger.info(
                    "Cycle complete. Final cumulative displacement: %.2f px", final_score
                )
                write_phase_summary_csv(config, shared_state, EXPERIMENT_PHASES)

                if effective_mode == "wait_for_cycle":
                    logger.info("wait_for_cycle mode: requesting shutdown")
                    shared_state.request_shutdown()
                    return
                continue

            if cycle_timeout_sec is not None:
                snapshot = shared_state.get_collection_snapshot()
                cycle_started = snapshot.get("cycle_started_at")
                if cycle_started and (time.time() - cycle_started) > cycle_timeout_sec:
                    logger.warning("Cycle timeout (%ds) exceeded, finalizing", cycle_timeout_sec)
                    score_summary = shared_state.get_score_summary()
                    final_score = float(score_summary.get("cumulative_displacement", 0.0))
                    shared_state.transition_to_complete(time.time(), final_score)
                    write_phase_summary_csv(config, shared_state, EXPERIMENT_PHASES)
                    if effective_mode == "wait_for_cycle":
                        shared_state.request_shutdown()
                        return

    logger.info("CycleMonitor stopped")


def write_phase_summary_csv(
    config: Config,
    shared_state: SharedState,
    phase_names: tuple[str, ...],
) -> None:
    """Write per-phase aggregate results to a separate CSV file."""
    phase_summary = shared_state.get_phase_summary()
    ordered_phases = list(phase_names)
    ordered_phases.extend(
        phase for phase in sorted(phase_summary)
        if phase not in set(ordered_phases)
    )

    results_dir = os.path.dirname(config.results_by_phase_path)
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)

    with open(config.results_by_phase_path, "w", newline="") as phase_file:
        writer = csv.writer(phase_file)
        writer.writerow(
            [
                "phase",
                "frames_scored",
                "mean_latency_ms",
                "mean_jitter_ms",
                "deadline_misses",
                "deadline_miss_pct",
                "mean_displacement_px",
                "cumulative_displacement_px",
                "local_frames",
                "remote_frames",
                "local_fallback_frames",
                "local_total_frames",
                "remote_pct",
            ]
        )
        for phase in ordered_phases:
            stats = phase_summary.get(phase, {})
            frames = int(stats.get("frames", 0))
            latency_frames = int(stats.get("latency_frames", 0))
            jitter_frames = int(stats.get("jitter_frames", 0))
            total_displacement = float(stats.get("total_displacement", 0.0))
            total_latency = float(stats.get("total_latency_ms", 0.0))
            total_jitter = float(stats.get("total_jitter_ms", 0.0))
            deadline_misses = int(stats.get("deadline_misses", 0))
            local_frames = int(stats.get("local_frames", 0))
            remote_frames = int(stats.get("remote_frames", 0))
            local_fallback_frames = int(stats.get("local_fallback_frames", 0))
            local_total_frames = local_frames + local_fallback_frames
            mean_latency = total_latency / latency_frames if latency_frames else 0.0
            mean_jitter = total_jitter / jitter_frames if jitter_frames else 0.0
            mean_displacement = total_displacement / frames if frames else 0.0
            remote_pct = (remote_frames / frames * 100.0) if frames else 0.0
            deadline_miss_pct = (deadline_misses / frames * 100.0) if frames else 0.0
            writer.writerow(
                [
                    phase,
                    frames,
                    round(mean_latency, 2),
                    round(mean_jitter, 2),
                    deadline_misses,
                    round(deadline_miss_pct, 2),
                    round(mean_displacement, 2),
                    round(total_displacement, 2),
                    local_frames,
                    remote_frames,
                    local_fallback_frames,
                    local_total_frames,
                    round(remote_pct, 2),
                ]
            )

    logger.info("Wrote per-phase summary CSV: %s", config.results_by_phase_path)


def _load_sp_agent_class(class_name: str):
    """Import and return the SPAgent class indicated by config.sp_agent_class.

    class_name="student"  loads SPAgent from student.sp_agent
    class_name="example"  loads ExampleSPAgent from student.sp_agent_example
    (the example module also exports SPAgent as an alias)
    """
    import importlib
    if class_name == "student":
        module = importlib.import_module("student.sp_agent")
    elif class_name == "example":
        module = importlib.import_module("student.sp_agent_example")
    else:
        raise ValueError(f"Unknown sp_agent_class: {class_name!r}")
    return module.SPAgent


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
    logger.info("  remote_fallback_local: %s", config.remote_fallback_to_local)
    logger.info("  triton_url           : %s", config.triton_url or "(not set, legacy only)")
    logger.info("  kafka_brokers        : %s", config.kafka_brokers or "(not set)")
    logger.info("  otlp_endpoint        : %s", config.otlp_endpoint or "(not set)")
    logger.info("  initial_mode         : %s", config.initial_processing_mode)
    logger.info("  frame_interval_ms    : %d", config.frame_interval_ms)
    logger.info("  results_log_path     : %s", config.results_log_path)
    logger.info("  results_by_phase_path: %s", config.results_by_phase_path)
    logger.info("  conf_threshold       : %.2f", config.conf_threshold)
    logger.info("  target_class_id      : %s", config.target_class_id)
    logger.info("  target_conf_threshold: %.4g", config.target_conf_threshold)
    logger.info("  sp_agent_interval_ms : %d", config.sp_agent_interval_ms)
    logger.info("  manual_placement    : %s", config.manual_placement_control)
    logger.info("  sp_agent_class      : %s", config.sp_agent_class)
    logger.info("  control_topic       : %s", config.kafka_control_topic)
    logger.info("  latency_probes      : %s", config.latency_probes_enabled)
    logger.info("  probe_interval_sec  : %.1f", config.latency_probe_interval_sec)
    logger.info("  sync_mode            : %s", config.sync_mode)
    logger.info("  phase_timeout_sec    : %.0f", config.phase_timeout_sec)
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
    cycle_duration = load_cycle_duration_sec(config.scenario_path)
    shared_state.set_cycle_duration_sec(cycle_duration)
    logger.info("Cycle duration set to %.1fs", cycle_duration)

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
            failure_cooldown=config.remote_failure_cooldown_sec,
        )
    else:
        logger.warning("REMOTE_INFERENCE_URL/TRITON_URL not set: running in local-only mode")

    # 7. Kafka publisher
    kafka_publisher = AppMetricsPublisher(
        config.kafka_brokers, config.kafka_app_topic, config.group_id
    )
    dashboard_publisher = DashboardPublisher(config, shared_state)

    # 8. Results CSV
    results_dir = os.path.dirname(config.results_log_path)
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
    results_file = open(config.results_log_path, "a", newline="")

    # 9. Queues
    reader_queue: queue.Queue = queue.Queue(maxsize=config.queue_max_size)
    scorer_queue: queue.Queue = queue.Queue(maxsize=config.queue_max_size)

    # 10. Thread objects
    frame_reader = FrameReader(
        config,
        shared_state,
        reader_queue,
        dashboard_publisher,
    )
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
    SPAgentClass = _load_sp_agent_class(config.sp_agent_class)
    sp_agent = SPAgentClass(config, shared_state)
    logger.info(
        "Loaded SP-Agent: %s (from SP_AGENT_CLASS=%s)",
        SPAgentClass.__name__,
        config.sp_agent_class,
    )
    latency_probe = None
    if config.latency_probes_enabled:
        if config.kafka_brokers:
            latency_probe = LatencyProbe(
                config,
                shared_state,
                local_server,
                remote_client,
                kafka_publisher,
            )
        else:
            logger.warning("LATENCY_PROBES_ENABLED=true but KAFKA_BROKERS is empty")

    # 11. Start threads
    threads = [
        threading.Thread(target=frame_reader.run, name="FrameReader", daemon=False),
        threading.Thread(target=dispatcher.run, name="Dispatcher", daemon=False),
        threading.Thread(target=scorer.run, name="Scorer", daemon=False),
        threading.Thread(target=sp_agent.run, name="SPAgent", daemon=False),
    ]
    if latency_probe is not None:
        threads.append(
            threading.Thread(target=latency_probe.run, name="LatencyProbe", daemon=False)
        )
    for t in threads:
        t.start()

    # 12. Cycle monitor (daemon thread drives state machine)
    cycle_monitor_thread = threading.Thread(
        target=run_cycle_monitor,
        args=(config, shared_state),
        daemon=True,
        name="CycleMonitor",
    )
    cycle_monitor_thread.start()

    # 13. Wait for completion
    try:
        while not shared_state.is_shutdown_requested():
            time.sleep(0.5)
    except KeyboardInterrupt:
        logger.info("Shutdown requested (KeyboardInterrupt)")
        shared_state.request_shutdown()

    for t in threads:
        t.join(timeout=5.0)

    kafka_publisher.flush()
    dashboard_publisher.close()
    results_file.close()
    write_phase_summary_csv(config, shared_state, EXPERIMENT_PHASES)

    # 14. Per-phase and overall summary
    phase_summary = shared_state.get_phase_summary()
    score_summary = shared_state.get_score_summary()

    logger.info("=" * 60)
    logger.info("Experiment complete. Results by phase:")
    logger.info("-" * 60)
    for phase_name in EXPERIMENT_PHASES:
        ps = phase_summary.get(phase_name, {})
        frames = int(ps.get("frames", 0))
        total_disp = float(ps.get("total_displacement", 0.0))
        total_lat = float(ps.get("total_latency_ms", 0.0))
        lat_frames = int(ps.get("latency_frames", 0))
        total_jitter = float(ps.get("total_jitter_ms", 0.0))
        jitter_frames = int(ps.get("jitter_frames", 0))
        deadline_misses = int(ps.get("deadline_misses", 0))
        avg_disp = total_disp / frames if frames > 0 else 0.0
        avg_lat = total_lat / lat_frames if lat_frames > 0 else 0.0
        avg_jitter = total_jitter / jitter_frames if jitter_frames > 0 else 0.0
        miss_pct = deadline_misses / frames * 100.0 if frames > 0 else 0.0
        logger.info(
            "  %-16s  disp: %6.1f px  lat: %6.1f ms  jitter: %5.1f ms"
            "  misses: %d (%.1f%%)  frames: %d",
            phase_name, avg_disp, avg_lat, avg_jitter,
            deadline_misses, miss_pct, frames,
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

    # Excluded frames change the score, so they are reported rather than left
    # to be discovered in the CSV.
    excluded = scorer.excluded_counts
    if any(excluded.values()):
        logger.info("-" * 60)
        logger.info(
            "  %-16s  %d warm-up spike(s), %d video-wrap frame(s)",
            "excluded",
            excluded["warmup_spike"],
            excluded["video_wrap"],
        )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
