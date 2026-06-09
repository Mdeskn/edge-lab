"""
Thread 2: Pulls frames from reader queue, runs inference (local or remote),
pushes results to scorer queue.
"""
import logging
import queue
import time

import numpy as np
import cv2

from config import Config
from shared_state import SharedState
from inference.local_server import LocalServer
from inference.remote_client import RemoteClient

logger = logging.getLogger(__name__)


class Dispatcher:
    """
    Fetches frames from the reader queue, selects inference backend based on
    the requested processing mode, and forwards results to the scorer queue.

    Falls back to local inference automatically when remote inference fails.
    The "local_fallback" result label is reported with the scored frame but is
    never written back as an SP-Agent placement choice.
    """

    def __init__(
        self,
        config: Config,
        shared_state: SharedState,
        local_server: LocalServer,
        remote_client: RemoteClient,
        reader_queue: queue.Queue,
        scorer_queue: queue.Queue,
        tracer,
    ):
        """Store all dependencies; tracer may be a no-op tracer."""
        self.config = config
        self.shared_state = shared_state
        self.local_server = local_server
        self.remote_client = remote_client
        self.reader_queue = reader_queue
        self.scorer_queue = scorer_queue
        self.tracer = tracer

    def run(self) -> None:
        """
        Main thread loop.

        Dequeues (frame_number, frame, gt_x, gt_y, enqueue_time), runs the
        appropriate inference backend, records latency, and enqueues results
        for the Scorer. Remote failures fall back to local inference.
        """
        logger.info("Dispatcher started")

        while not self.shared_state.is_shutdown_requested():
            try:
                frame_number, frame, gt_x, gt_y, enqueue_time = self.reader_queue.get(
                    timeout=1.0
                )
            except queue.Empty:
                continue

            dispatch_start = time.time()
            requested_mode = self.shared_state.get_processing_mode()

            try:
                with self.tracer.start_as_current_span("frame_pipeline") as span:
                    span.set_attribute("frame.number", frame_number)
                    span.set_attribute("processing.mode", requested_mode)
                    span.set_attribute("processing.requested_mode", requested_mode)

                    with self.tracer.start_as_current_span("preprocess") as pre_span:
                        preprocessed = self._preprocess(frame)
                        pre_span.set_attribute("input.shape", str(frame.shape))

                    result_mode = requested_mode
                    remote_ok = (
                        requested_mode == "remote"
                        and self.remote_client is not None
                        and self.remote_client.is_available()
                    )

                    if remote_ok:
                        with self.tracer.start_as_current_span("remote_inference") as ri_span:
                            ri_span.set_attribute(
                                "triton.url", self.config.triton_url
                            )
                            ri_span.set_attribute("model.name", self.config.triton_model_name)
                            try:
                                pred_x, pred_y, pred_x1, pred_y1, pred_x2, pred_y2 = self.remote_client.infer(
                                    preprocessed, frame.shape
                                )
                            except Exception as exc:
                                logger.warning(
                                    "Remote inference failed: %s, falling back to local", exc
                                )
                                with self.tracer.start_as_current_span(
                                    "local_inference_fallback"
                                ) as fb_span:
                                    fb_span.set_attribute("model.name", "yolov10n")
                                    pred_x, pred_y, pred_x1, pred_y1, pred_x2, pred_y2 = self.local_server.infer(
                                        preprocessed, frame.shape
                                    )
                                result_mode = "local_fallback"
                    else:
                        with self.tracer.start_as_current_span("local_inference") as li_span:
                            li_span.set_attribute("model.name", "yolov10n")
                            pred_x, pred_y, pred_x1, pred_y1, pred_x2, pred_y2 = self.local_server.infer(
                                preprocessed, frame.shape
                            )

                    span.set_attribute("processing.result_mode", result_mode)

            except Exception:
                logger.exception("Unhandled error in Dispatcher for frame %d", frame_number)
                continue

            latency_ms = (time.time() - dispatch_start) * 1000.0
            self.shared_state.add_latency(latency_ms)

            try:
                self.scorer_queue.put(
                    (
                        frame_number,
                        frame,
                        gt_x,
                        gt_y,
                        pred_x,
                        pred_y,
                        pred_x1,
                        pred_y1,
                        pred_x2,
                        pred_y2,
                        latency_ms,
                        result_mode,
                        time.time(),
                    ),
                    timeout=0.05,
                )
            except queue.Full:
                logger.debug("scorer_queue full, dropping result for frame %d", frame_number)

        logger.info("Dispatcher stopped")

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        """
        Prepare a BGR video frame for YOLOv10n ONNX inference.

        Steps: resize → BGR→RGB → normalize to [0,1] → HWC→CHW → add batch dim.
        Returns float32 array of shape (1, 3, H, W).
        """
        resized = cv2.resize(frame, (self.config.input_width, self.config.input_height))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        chw = np.transpose(normalized, (2, 0, 1))
        batched = np.expand_dims(chw, axis=0)
        return batched
