"""
Thread 3: Receives inference results, calculates displacement vs ground truth,
logs to CSV, and publishes to Kafka.
"""
import csv
import io
import logging
import math
import queue

from config import Config
from metrics.dashboard_publisher import DashboardPublisher
from metrics.kafka_publisher import AppMetricsPublisher
from shared_state import SharedState
from spike_filter import WarmupSpikeFilter
from threads.messages import InferenceResult

logger = logging.getLogger(__name__)


class Scorer:
    """
    Consumes inference results, scores them against ground truth, writes CSV
    rows, and publishes metrics to Kafka.
    """

    def __init__(
        self,
        config: Config,
        shared_state: SharedState,
        scorer_queue: queue.Queue,
        kafka_publisher: AppMetricsPublisher,
        dashboard_publisher: DashboardPublisher,
        results_file: io.IOBase,
    ):
        """Store dependencies and CSV writer state."""
        self.config = config
        self.shared_state = shared_state
        self.scorer_queue = scorer_queue
        self.kafka_publisher = kafka_publisher
        self.dashboard_publisher = dashboard_publisher
        self.results_file = results_file
        self._csv_writer = None
        self._last_latency_ms: float | None = None
        self._wrapped_frames = 0
        self._spike_filter = WarmupSpikeFilter(
            enabled=config.warmup_spike_filter_enabled,
            settle_sec=config.warmup_spike_settle_sec,
            multiplier=config.warmup_spike_multiplier,
            floor_ms=config.warmup_spike_floor_ms,
            phases=config.warmup_spike_phases,
        )

    @property
    def excluded_counts(self) -> dict[str, int]:
        """Frames left out of the score, by reason, for end-of-run reporting."""
        return {
            "warmup_spike": self._spike_filter.excluded_count,
            "video_wrap": self._wrapped_frames,
        }

    def run(self) -> None:
        """
        Main thread loop.

        Writes CSV header on startup, then processes each InferenceResult from
        scorer_queue: computes displacement, writes CSV, and publishes Kafka.
        """
        logger.info("Scorer started")

        self._csv_writer = csv.writer(self.results_file)
        self._csv_writer.writerow(
            [
                "timestamp",
                "frame_number",
                "group_id",
                "experiment_phase",
                "processing_mode",
                "latency_ms",
                "jitter_ms",
                "deadline_miss",
                "true_x",
                "true_y",
                "predicted_x",
                "predicted_y",
                "displacement_px",
                "cumulative_displacement_px",
                "excluded_warmup_spike",
                "excluded_video_wrap",
            ]
        )
        self.results_file.flush()

        while not self.shared_state.is_shutdown_requested():
            try:
                result: InferenceResult = self.scorer_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            frame_number = result.frame_number
            frame = result.frame
            pred_x = result.predicted_x
            pred_y = result.predicted_y
            latency_ms = result.latency_ms
            mode = result.mode
            result_time = result.completed_at

            current_phase = self.shared_state.get_experiment_phase()

            # Flags at most one startup-artifact spike per gpu_load phase
            # entry. The real value still reaches Kafka/the dashboard payload
            # further below: only score aggregation and chart history honor
            # this flag, so the SP-Agent's placement decisions are unaffected.
            is_warmup_spike = self._spike_filter.check(current_phase, latency_ms)
            if is_warmup_spike:
                logger.warning(
                    "Excluding warm-up latency spike from score/charts: "
                    "frame=%d latency=%.1fms phase=%s",
                    frame_number, latency_ms, current_phase,
                )

            # Jitter: absolute frame-to-frame latency variation. Excluded
            # frames don't become the reference point either, otherwise the
            # frame right after the spike would inherit its own artifact
            # spike as a jitter echo when latency drops back down.
            if self._last_latency_ms is not None:
                jitter_ms = abs(latency_ms - self._last_latency_ms)
            else:
                jitter_ms = 0.0
            if not is_warmup_spike:
                self._last_latency_ms = latency_ms

            # Deadline miss
            deadline_miss = latency_ms > self.config.latency_deadline_ms

            # Current GT: where the tracked object is RIGHT NOW, when this
            # prediction arrives.
            # Scoring against this (not the bundled capture-time GT) is what makes
            # latency affect displacement: the longer inference takes, the further
            # the tracked object has moved, the higher the penalty.
            (
                _,
                current_gt_x,
                current_gt_y,
                current_video_cycle,
            ) = self.shared_state.get_ground_truth()

            # If the clip looped while this frame was in flight, the current
            # ground truth describes the start of the clip rather than where
            # the tracked object continued to. The resulting displacement would
            # measure the video wrapping, not the placement decision, and it
            # would fall hardest on slow remote frames. Skip those frames.
            wrapped_mid_flight = result.video_cycle != current_video_cycle
            if wrapped_mid_flight:
                self._wrapped_frames += 1
                logger.debug(
                    "Skipping frame %d: video wrapped mid-flight (cycle %d -> %d)",
                    frame_number, result.video_cycle, current_video_cycle,
                )

            # Bundled GT only gates whether this frame is scored. If there was
            # no target at capture time, skip the frame entirely.
            frame_had_ground_truth = result.gt_x is not None and result.gt_y is not None
            has_prediction = result.has_prediction

            if frame_had_ground_truth and not wrapped_mid_flight:
                if has_prediction and current_gt_x is not None:
                    displacement_px = math.sqrt(
                        (current_gt_x - pred_x) ** 2 + (current_gt_y - pred_y) ** 2
                    )
                else:
                    # YOLO missed detection, or current GT unavailable: fixed penalty.
                    displacement_px = self.config.miss_penalty_px
            else:
                displacement_px = None  # not scoreable, skip frame

            # Only count frames toward the cumulative score when actively collecting.
            # In ARMED/COMPLETE/DISCONNECTED the dashboard still shows live values for
            # debugging but they do not influence the saved score.
            is_collecting = self.shared_state.is_collecting()
            excluded_from_score = is_warmup_spike or wrapped_mid_flight

            if displacement_px is not None and is_collecting and not excluded_from_score:
                self.shared_state.add_displacement(displacement_px)
                self.shared_state.add_phase_result(
                    current_phase,
                    displacement_px,
                    latency_ms,
                    jitter_ms,
                    mode,
                    deadline_miss=deadline_miss,
                )
            score_summary = self.shared_state.get_score_summary()

            if is_collecting:
                try:
                    self._csv_writer.writerow(
                        [
                            result_time,
                            frame_number,
                            self.config.group_id,
                            current_phase,
                            mode,
                            round(latency_ms, 2),
                            round(jitter_ms, 2),
                            int(deadline_miss),
                            _round_optional(current_gt_x),
                            _round_optional(current_gt_y),
                            round(pred_x, 2),
                            round(pred_y, 2),
                            _round_optional(displacement_px),
                            round(score_summary["cumulative_displacement"], 2),
                            int(is_warmup_spike),
                            int(wrapped_mid_flight),
                        ]
                    )
                    self.results_file.flush()
                except Exception as exc:
                    logger.error("CSV write error: %s", exc)

            collection_state = self.shared_state.get_collection_snapshot()

            try:
                self.kafka_publisher.publish(
                    frame_number=frame_number,
                    experiment_phase=current_phase,
                    processing_mode=mode,
                    latency_ms=latency_ms,
                    jitter_ms=round(jitter_ms, 2),
                    deadline_miss=bool(deadline_miss),
                    displacement_px=displacement_px,
                    true_x=current_gt_x,
                    true_y=current_gt_y,
                    predicted_x=pred_x,
                    predicted_y=pred_y,
                    cumulative_displacement_px=score_summary["cumulative_displacement"],
                    timestamp=result_time,
                    collection_state=collection_state,
                    excluded=excluded_from_score,
                )
            except Exception as exc:
                logger.error("Kafka publish error in Scorer: %s", exc)

            self.dashboard_publisher.publish_metric(
                frame_number=frame_number,
                timestamp=result_time,
                true_x=current_gt_x,
                true_y=current_gt_y,
                collection_state=collection_state,
                predicted_x=pred_x,
                predicted_y=pred_y,
                processing_mode=mode,
                latency_ms=latency_ms,
                jitter_ms=jitter_ms,
                deadline_miss=deadline_miss,
                displacement_px=displacement_px,
                cumulative_displacement_px=score_summary["cumulative_displacement"],
                experiment_phase=current_phase,
                excluded=excluded_from_score,
            )

            if displacement_px is not None and has_prediction and current_gt_x is not None:
                frame_h, frame_w = frame.shape[:2]
                warn_threshold = math.sqrt(frame_w ** 2 + frame_h ** 2) * 0.05
                if displacement_px > warn_threshold:
                    logger.warning(
                        "Large displacement: %.1fpx (threshold %.1fpx) frame=%d",
                        displacement_px, warn_threshold, frame_number,
                    )

        logger.info("Scorer stopped")

def _round_optional(value: float | None) -> float | None:
    """Round a numeric metric while preserving unavailable values."""
    return round(value, 2) if value is not None else None
