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
from shared_state import SharedState
from metrics.kafka_publisher import AppMetricsPublisher

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

    def run(self) -> None:
        """
        Main thread loop.

        Writes CSV header on startup, then processes each result tuple from
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
            ]
        )
        self.results_file.flush()

        while not self.shared_state.is_shutdown_requested():
            try:
                (
                    frame_number,
                    frame,
                    gt_x,
                    gt_y,
                    pred_x,
                    pred_y,
                    _pred_x1,
                    _pred_y1,
                    _pred_x2,
                    _pred_y2,
                    latency_ms,
                    mode,
                    result_time,
                ) = self.scorer_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            current_phase = self.shared_state.get_experiment_phase()

            # Jitter: absolute frame-to-frame latency variation
            if self._last_latency_ms is not None:
                jitter_ms = abs(latency_ms - self._last_latency_ms)
            else:
                jitter_ms = 0.0
            self._last_latency_ms = latency_ms

            # Deadline miss
            deadline_miss = latency_ms > self.config.latency_deadline_ms

            # Current GT: where the car is RIGHT NOW, when this prediction arrives.
            # Scoring against this (not the bundled capture-time GT) is what makes
            # latency affect displacement: the longer inference takes, the further
            # the car has moved, the higher the penalty.
            _, current_gt_x, current_gt_y = self.shared_state.get_ground_truth()

            # Bundled GT (gt_x, gt_y) only gates whether this frame is scored.
            # If there was no target at capture time, skip the frame entirely.
            frame_had_ground_truth = gt_x is not None and gt_y is not None
            has_prediction = pred_x != 0.0 or pred_y != 0.0

            if frame_had_ground_truth:
                if has_prediction and current_gt_x is not None:
                    displacement_px = math.sqrt(
                        (current_gt_x - pred_x) ** 2 + (current_gt_y - pred_y) ** 2
                    )
                else:
                    # YOLO missed detection, or current GT unavailable: fixed penalty.
                    displacement_px = self.config.miss_penalty_px
            else:
                displacement_px = None  # no target at capture time, skip frame

            # Only count frames toward the cumulative score when actively collecting.
            # In ARMED/COMPLETE/DISCONNECTED the dashboard still shows live values for
            # debugging but they do not influence the saved score.
            is_collecting = self.shared_state.is_collecting()

            if displacement_px is not None and is_collecting:
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
                        ]
                    )
                    self.results_file.flush()
                except Exception as exc:
                    logger.error("CSV write error: %s", exc)

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
                )
            except Exception as exc:
                logger.error("Kafka publish error in Scorer: %s", exc)

            self.dashboard_publisher.publish_metric(
                frame_number=frame_number,
                timestamp=result_time,
                true_x=current_gt_x,
                true_y=current_gt_y,
                collection_state=self.shared_state.get_collection_snapshot(),
                predicted_x=pred_x,
                predicted_y=pred_y,
                processing_mode=mode,
                latency_ms=latency_ms,
                jitter_ms=jitter_ms,
                deadline_miss=deadline_miss,
                displacement_px=displacement_px,
                cumulative_displacement_px=score_summary["cumulative_displacement"],
                experiment_phase=current_phase,
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
