"""
Thread 1: Reads frames from video at configured frame rate.
Maintains ground truth lookup. Displays raw frame. Pushes to dispatcher queue.
"""
import csv
import logging
import queue
import time
from bisect import bisect_left

import cv2
import numpy as np

from config import Config
from metrics.dashboard_publisher import DashboardPublisher
from shared_state import SharedState
from threads.messages import FrameJob

logger = logging.getLogger(__name__)


class FrameReader:
    """
    Reads video frames at the configured frame rate and pushes them to the
    dispatcher queue alongside the matching ground truth coordinates.
    """

    def __init__(
        self,
        config: Config,
        shared_state: SharedState,
        reader_queue: queue.Queue,
        dashboard_publisher: DashboardPublisher,
    ):
        """Store references and initialise the frame counter."""
        self.config = config
        self.shared_state = shared_state
        self.reader_queue = reader_queue
        self.dashboard_publisher = dashboard_publisher
        self.frame_counter: int = 0
        self._video_cycle: int = 0
        self._ground_truth: dict[int, tuple[float | None, float | None]] = {}
        self._gt_frame_numbers: list[int] = []

    def _load_ground_truth(self) -> dict[int, tuple[float | None, float | None]]:
        """
        Load ground_truth.csv into {frame_number: (center_x, center_y)}.

        Raises FileNotFoundError if the file is missing. Rows whose frame
        number or centre cannot be parsed are skipped. `_lookup_gt` handles
        frames with no exact entry.
        """
        gt: dict[int, tuple[float | None, float | None]] = {}
        path = self.config.ground_truth_path

        try:
            with open(path, newline="") as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    try:
                        fn = int(row["frame_number"])
                        cx = float(row["center_x"])
                        cy = float(row["center_y"])
                        gt[fn] = (
                            None if np.isnan(cx) else cx,
                            None if np.isnan(cy) else cy,
                        )
                    except (ValueError, KeyError):
                        continue
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Ground truth file not found: {path}") from exc

        logger.info("Loaded %d ground truth entries from %s", len(gt), path)
        return gt

    def _lookup_gt(self, frame_number: int) -> tuple[float | None, float | None]:
        """
        Return ground truth coordinates for frame_number.

        Falls back to the nearest available frame number if an exact match
        does not exist, using a binary search over the sorted frame numbers
        built at load time. Logs at DEBUG level for fallback lookups.
        """
        if frame_number in self._ground_truth:
            return self._ground_truth[frame_number]

        if not self._gt_frame_numbers:
            return (0.0, 0.0)

        index = bisect_left(self._gt_frame_numbers, frame_number)
        candidates = []
        if index < len(self._gt_frame_numbers):
            candidates.append(self._gt_frame_numbers[index])
        if index > 0:
            candidates.append(self._gt_frame_numbers[index - 1])
        nearest = min(candidates, key=lambda k: abs(k - frame_number))
        logger.debug("GT fallback: frame %d -> nearest %d", frame_number, nearest)
        return self._ground_truth[nearest]

    def run(self) -> None:
        """
        Main thread loop.

        Opens the video file, reads frames at config.frame_interval_ms,
        updates ground truth in SharedState, and pushes frames to reader_queue.
        Loops back to frame 0 on end-of-video. Exits on shutdown request or 'q' keypress.
        """
        logger.info("FrameReader started")

        self._ground_truth = self._load_ground_truth()
        self._gt_frame_numbers = sorted(self._ground_truth)

        cap = cv2.VideoCapture(self.config.video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {self.config.video_path}")

        try:
            while not self.shared_state.is_shutdown_requested():
                loop_start = time.time()

                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    self.frame_counter = 0
                    self._video_cycle += 1
                    continue

                self.frame_counter += 1

                gt_x, gt_y = self._lookup_gt(self.frame_counter)
                self.shared_state.update_ground_truth(
                    self.frame_counter, gt_x, gt_y, self._video_cycle
                )
                if self.config.latency_probes_enabled:
                    self.shared_state.update_latest_frame(self.frame_counter, frame)

                try:
                    self.reader_queue.put(
                        FrameJob(
                            frame_number=self.frame_counter,
                            video_cycle=self._video_cycle,
                            frame=frame.copy(),
                            gt_x=gt_x,
                            gt_y=gt_y,
                            enqueued_at=time.time(),
                        ),
                        timeout=0.05,
                    )
                except queue.Full:
                    logger.debug("reader_queue full, dropping frame %d", self.frame_counter)

                # The inference queue owns its copy. The preview worker can now
                # annotate the original capture frame without affecting inference.
                latest_prediction = self.shared_state.get_latest_prediction()
                if (
                    latest_prediction is not None
                    and latest_prediction.get("video_cycle") != self._video_cycle
                ):
                    latest_prediction = None
                preview_prediction = latest_prediction
                if preview_prediction is None:
                    preview_prediction = {"processing_mode": "warming_up"}

                self.dashboard_publisher.publish_preview(
                    frame=frame,
                    frame_number=self.frame_counter,
                    timestamp=time.time(),
                    true_x=gt_x,
                    true_y=gt_y,
                    prediction=preview_prediction,
                )

                if self.frame_counter % 100 == 0:
                    logger.debug("FrameReader: frame %d", self.frame_counter)

                elapsed = time.time() - loop_start
                sleep_time = max(0.0, self.config.frame_interval_ms / 1000.0 - elapsed)
                time.sleep(sleep_time)

        finally:
            cap.release()
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
            logger.info("FrameReader stopped")
