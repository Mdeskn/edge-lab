"""
Thread 3: Receives inference results, calculates displacement vs ground truth,
draws overlay on displayed frame, logs to CSV, publishes to Kafka.
"""
import csv
import io
import logging
import math
import queue
import time
from collections import deque

import cv2
import numpy as np

from config import Config
from shared_state import SharedState
from metrics.dashboard_publisher import DashboardPublisher
from metrics.kafka_publisher import AppMetricsPublisher

logger = logging.getLogger(__name__)

_FONT = cv2.FONT_HERSHEY_SIMPLEX
_WHITE = (255, 255, 255)
_GREEN = (0, 255, 0)
_RED = (0, 0, 255)
_YELLOW = (0, 255, 255)


class Scorer:
    """
    Consumes inference results, scores them against ground truth, draws a HUD
    overlay on the frame, writes CSV rows, and publishes metrics to Kafka.
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
        self._display_latencies: deque = deque(maxlen=5)

    def run(self) -> None:
        """
        Main thread loop.

        Writes CSV header on startup, then processes each result tuple from
        scorer_queue: computes displacement, draws HUD, writes CSV, publishes Kafka.
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
                    latency_ms,
                    mode,
                    result_time,
                ) = self.scorer_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            current_phase = self.shared_state.get_experiment_phase()

            self._display_latencies.append(latency_ms)
            avg_display_latency = sum(self._display_latencies) / len(self._display_latencies)

            has_ground_truth = gt_x is not None and gt_y is not None
            displacement_px = (
                math.sqrt((gt_x - pred_x) ** 2 + (gt_y - pred_y) ** 2)
                if has_ground_truth
                else None
            )
            if displacement_px is not None:
                self.shared_state.add_displacement(displacement_px)
                self.shared_state.add_phase_displacement(current_phase, displacement_px)
            score_summary = self.shared_state.get_score_summary()

            self._draw_overlay(
                frame, gt_x, gt_y, pred_x, pred_y,
                displacement_px, avg_display_latency, mode, current_phase, score_summary,
            )

            if self.config.display_output:
                try:
                    cv2.imshow("Edge Lab: Output with Overlay", frame)
                except Exception as exc:
                    logger.warning("Display (output) error: %s", exc)

            try:
                self._csv_writer.writerow(
                    [
                        result_time,
                        frame_number,
                        self.config.group_id,
                        current_phase,
                        mode,
                        round(latency_ms, 2),
                        _round_optional(gt_x),
                        _round_optional(gt_y),
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
                    displacement_px=displacement_px,
                    true_x=gt_x,
                    true_y=gt_y,
                    predicted_x=pred_x,
                    predicted_y=pred_y,
                    cumulative_displacement_px=score_summary["cumulative_displacement"],
                    timestamp=result_time,
                )
            except Exception as exc:
                logger.error("Kafka publish error in Scorer: %s", exc)

            try:
                self.dashboard_publisher.publish(
                    frame=frame,
                    frame_number=frame_number,
                    timestamp=result_time,
                    true_x=gt_x,
                    true_y=gt_y,
                    predicted_x=pred_x,
                    predicted_y=pred_y,
                    processing_mode=mode,
                    latency_ms=latency_ms,
                    displacement_px=displacement_px,
                    cumulative_displacement_px=score_summary["cumulative_displacement"],
                    experiment_phase=current_phase,
                )
            except Exception as exc:
                logger.warning("Dashboard publish error in Scorer: %s", exc)

            if (
                displacement_px is not None
                and (pred_x > 0.0 or pred_y > 0.0)
                and displacement_px > 50
            ):
                logger.warning(
                    "Large displacement: %.1fpx frame=%d", displacement_px, frame_number
                )

        logger.info("Scorer stopped")

    def _draw_overlay(
        self,
        frame: np.ndarray,
        gt_x: float | None,
        gt_y: float | None,
        pred_x: float,
        pred_y: float,
        displacement_px: float | None,
        avg_display_latency: float,
        mode: str,
        current_phase: str,
        score_summary: dict,
    ) -> None:
        """Draw ground truth, prediction, connecting line, and HUD text onto frame."""
        has_ground_truth = gt_x is not None and gt_y is not None
        has_prediction = pred_x > 0.0 or pred_y > 0.0
        if has_ground_truth:
            cv2.circle(frame, (int(gt_x), int(gt_y)), 8, _GREEN, -1)
            cv2.putText(frame, "GT", (int(gt_x) + 10, int(gt_y) - 8), _FONT, 0.5, _GREEN, 1)
        if has_prediction:
            cv2.circle(frame, (int(pred_x), int(pred_y)), 8, _RED, -1)
            cv2.putText(frame, "PRED", (int(pred_x) + 10, int(pred_y) - 8), _FONT, 0.5, _RED, 1)
        if has_ground_truth and has_prediction:
            cv2.line(
                frame,
                (int(gt_x), int(gt_y)),
                (int(pred_x), int(pred_y)),
                _YELLOW,
                1,
            )

        cv2.putText(frame, f"Mode: {mode}", (10, 30), _FONT, 0.6, _WHITE, 1)
        cv2.putText(frame, f"Phase: {current_phase}", (10, 55), _FONT, 0.6, _WHITE, 1)
        cv2.putText(frame, f"Latency (avg 5): {avg_display_latency:.0f}ms", (10, 80), _FONT, 0.6, _WHITE, 1)
        displacement_text = f"{displacement_px:.1f}px" if displacement_px is not None else "N/A"
        cv2.putText(frame, f"Displacement: {displacement_text}", (10, 105), _FONT, 0.6, _WHITE, 1)
        cv2.putText(
            frame,
            f"Cumulative: {score_summary['cumulative_displacement']:.0f}px",
            (10, 130),
            _FONT,
            0.6,
            _WHITE,
            1,
        )
        cv2.putText(
            frame,
            f"Frames: {score_summary['frames_processed']}",
            (10, 155),
            _FONT,
            0.6,
            _WHITE,
            1,
        )


def _round_optional(value: float | None) -> float | None:
    """Round a numeric metric while preserving unavailable values."""
    return round(value, 2) if value is not None else None
