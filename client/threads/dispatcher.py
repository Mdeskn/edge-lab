"""
Thread 2: Pulls frames from reader queue, runs inference (local or remote),
pushes results to scorer queue.
"""
import logging
import queue
import time

import numpy as np

from config import Config
from inference.local_server import LocalServer
from inference.preprocess import preprocess_frame
from inference.remote_client import RemoteClient
from shared_state import SharedState
from threads.messages import FrameJob, InferenceResult

logger = logging.getLogger(__name__)

#: What a backend returns when it produced no usable box.
_NO_PREDICTION = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


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

        Dequeues a frame with its video-cycle identity and ground truth, runs
        the appropriate inference backend, records latency, and enqueues the
        result for the Scorer. Remote failures may fall back to local inference.
        """
        logger.info("Dispatcher started")

        while not self.shared_state.is_shutdown_requested():
            try:
                job: FrameJob = self.reader_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            # Drain stale frames that piled up while inference was running.
            # Always process the most recent frame so displacement reflects real movement.
            drained = 0
            while True:
                try:
                    job = self.reader_queue.get_nowait()
                    drained += 1
                except queue.Empty:
                    break
            if drained > 0:
                logger.debug(
                    "Dispatcher: drained %d stale frame(s), processing frame %d",
                    drained,
                    job.frame_number,
                )

            frame_number = job.frame_number
            frame = job.frame

            dispatch_start = time.time()
            requested_mode = self.shared_state.get_processing_mode()
            manual_mode_locked = self.shared_state.is_processing_mode_locked()

            try:
                prediction, result_mode = self._infer_frame(
                    frame, frame_number, requested_mode, manual_mode_locked
                )
            except Exception:
                logger.exception("Unhandled error in Dispatcher for frame %d", frame_number)
                continue

            pred_x, pred_y, pred_x1, pred_y1, pred_x2, pred_y2 = prediction

            latency_ms = (time.time() - dispatch_start) * 1000.0
            self.shared_state.update_latest_prediction(
                frame_number,
                job.video_cycle,
                pred_x,
                pred_y,
                result_mode,
                latency_ms,
            )

            try:
                self.scorer_queue.put(
                    InferenceResult(
                        frame_number=frame_number,
                        video_cycle=job.video_cycle,
                        frame=frame,
                        gt_x=job.gt_x,
                        gt_y=job.gt_y,
                        predicted_x=pred_x,
                        predicted_y=pred_y,
                        predicted_x1=pred_x1,
                        predicted_y1=pred_y1,
                        predicted_x2=pred_x2,
                        predicted_y2=pred_y2,
                        latency_ms=latency_ms,
                        mode=result_mode,
                        completed_at=time.time(),
                    ),
                    timeout=0.05,
                )
            except queue.Full:
                logger.debug("scorer_queue full, dropping result for frame %d", frame_number)

        logger.info("Dispatcher stopped")

    def _infer_frame(
        self,
        frame: np.ndarray,
        frame_number: int,
        requested_mode: str,
        manual_mode_locked: bool,
    ) -> tuple[tuple, str]:
        """
        Run one frame through the selected backend.

        Returns ((cx, cy, x1, y1, x2, y2), result_mode) where result_mode is
        the label that travels with the scored frame: it may differ from
        `requested_mode` when a remote call fell back to local.
        """
        with self.tracer.start_as_current_span("frame_pipeline") as span:
            span.set_attribute("frame.number", frame_number)
            span.set_attribute("processing.mode", requested_mode)
            span.set_attribute("processing.requested_mode", requested_mode)
            span.set_attribute("processing.manual_locked", manual_mode_locked)

            # Placement is handled identically whether it came from the
            # SP-Agent or from the dashboard's manual lock. It used to differ:
            # a manual "remote" both disabled local fallback and bypassed the
            # failure cooldown, so the always-remote baseline students are told
            # to beat took a full miss penalty where a student's agent choosing
            # remote fell back to local and got a real prediction. Part of every
            # group's improvement over that baseline was therefore an artifact
            # of which control set the mode. REMOTE_FALLBACK_TO_LOCAL now
            # governs both paths, so "pure remote, no safety net" is still
            # available, just as a deliberate setting rather than a side effect.
            allow_remote_fallback = self.config.remote_fallback_to_local
            remote_ok = (
                requested_mode == "remote"
                and self.remote_client is not None
                and self.remote_client.is_available()
            )

            tensors = _TensorCache(frame, self.config, self.tracer)

            if remote_ok:
                prediction, result_mode = self._infer_remote(
                    frame, tensors, allow_remote_fallback
                )
            elif requested_mode == "remote":
                # The remote endpoint is in its failure cooldown, so skip
                # straight past the per-call timeout.
                if allow_remote_fallback:
                    prediction = self._infer_local(
                        tensors, frame, "remote_unavailable"
                    )
                    result_mode = "local_fallback"
                else:
                    prediction = _NO_PREDICTION
                    result_mode = "remote_unavailable"
            else:
                prediction = self._infer_local(tensors, frame, None)
                result_mode = "local"

            span.set_attribute("processing.result_mode", result_mode)
            return prediction, result_mode

    def _infer_remote(
        self,
        frame: np.ndarray,
        tensors: "_TensorCache",
        allow_remote_fallback: bool,
    ) -> tuple[tuple, str]:
        """Attempt remote inference, falling back to local when permitted."""
        with self.tracer.start_as_current_span("remote_inference") as ri_span:
            ri_span.set_attribute(
                "remote.url",
                self.config.remote_inference_url or self.config.triton_url,
            )
            ri_span.set_attribute("model.name", self.config.triton_model_name)
            try:
                if self.remote_client.sends_raw_frames():
                    ri_span.set_attribute("payload.kind", "jpeg")
                    return self.remote_client.infer(frame), "remote"
                ri_span.set_attribute("payload.kind", "fp32_tensor")
                tensor = tensors.get()
                return self.remote_client.infer(tensor, frame.shape), "remote"
            except Exception as exc:
                if not allow_remote_fallback:
                    logger.warning(
                        "Remote inference failed: %s, keeping forced remote mode", exc
                    )
                    return _NO_PREDICTION, "remote_unavailable"

                logger.warning(
                    "Remote inference failed: %s, falling back to local", exc
                )
                return self._infer_local(tensors, frame, None), "local_fallback"

    def _infer_local(
        self,
        tensors: "_TensorCache",
        frame: np.ndarray,
        reason: str | None,
    ) -> tuple:
        """
        Run local ONNX inference under the shared CPU lock.

        The lock keeps the latency probe's own full-model inference from
        overlapping this one; without it a probe could add its compute to the
        latency a student is scored on.
        """
        span_name = "local_inference" if reason is None else "local_inference_fallback"
        tensor = tensors.get(
            "preprocess" if reason is None else "preprocess_fallback"
        )
        with self.tracer.start_as_current_span(span_name) as li_span:
            li_span.set_attribute("model.name", "yolov10n")
            if reason is not None:
                li_span.set_attribute("reason", reason)
            with self.shared_state.local_inference_lock:
                return self.local_server.infer(tensor, frame.shape)


class _TensorCache:
    """Preprocess a frame at most once, however many backends ask for it."""

    def __init__(self, frame: np.ndarray, config: Config, tracer):
        self._frame = frame
        self._config = config
        self._tracer = tracer
        self._tensor: np.ndarray | None = None

    def get(self, span_name: str = "preprocess") -> np.ndarray:
        if self._tensor is None:
            with self._tracer.start_as_current_span(span_name) as pre_span:
                self._tensor = preprocess_frame(
                    self._frame,
                    self._config.input_width,
                    self._config.input_height,
                )
                pre_span.set_attribute("input.shape", str(self._frame.shape))
        return self._tensor
