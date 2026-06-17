"""
Optional low-rate latency probes for the inactive inference backend.

Probe samples are published to Kafka as event_type="latency_probe". They keep
the dashboard latency cards live for both placements, but are ignored by
scoring, CSV output, and SP-Agent decisions.
"""
import logging
import time

import cv2
import numpy as np

from config import Config
from inference.local_server import LocalServer
from inference.remote_client import RemoteClient
from metrics.kafka_publisher import AppMetricsPublisher
from shared_state import SharedState

logger = logging.getLogger(__name__)


class LatencyProbe:
    """Periodically measures the inactive backend latency on the latest frame."""

    def __init__(
        self,
        config: Config,
        shared_state: SharedState,
        local_server: LocalServer,
        remote_client: RemoteClient | None,
        kafka_publisher: AppMetricsPublisher,
    ):
        self.config = config
        self.shared_state = shared_state
        self.local_server = local_server
        self.remote_client = remote_client
        self.kafka_publisher = kafka_publisher
        self.interval_s = max(0.5, config.latency_probe_interval_sec)

    def run(self) -> None:
        """Main probe loop. Exits when SharedState requests shutdown."""
        logger.info("LatencyProbe started (interval %.1fs)", self.interval_s)

        while not self.shared_state.is_shutdown_requested():
            cycle_start = time.monotonic()
            frame_number, frame = self.shared_state.get_latest_frame()

            if frame is not None and frame_number is not None:
                requested_mode = self.shared_state.get_processing_mode()
                if requested_mode == "local":
                    self._probe_remote(frame_number, frame)
                elif requested_mode == "remote":
                    self._probe_local(frame_number, frame)
                else:
                    self._probe_local(frame_number, frame)
                    self._probe_remote(frame_number, frame)

            elapsed = time.monotonic() - cycle_start
            time.sleep(max(0.0, self.interval_s - elapsed))

        logger.info("LatencyProbe stopped")

    def _probe_local(self, frame_number: int, frame: np.ndarray) -> None:
        started = time.perf_counter()
        try:
            tensor = self._preprocess(frame)
            self.local_server.infer(tensor, frame.shape)
            latency_ms = (time.perf_counter() - started) * 1000.0
            self.kafka_publisher.publish_latency_probe(
                frame_number=frame_number,
                experiment_phase=self.shared_state.get_experiment_phase(),
                probe_mode="local",
                latency_ms=latency_ms,
            )
        except Exception as exc:
            logger.warning("Local latency probe failed: %s", exc)
            self.kafka_publisher.publish_latency_probe(
                frame_number=frame_number,
                experiment_phase=self.shared_state.get_experiment_phase(),
                probe_mode="local",
                latency_ms=None,
                status="failed",
                error=str(exc),
            )

    def _probe_remote(self, frame_number: int, frame: np.ndarray) -> None:
        if self.remote_client is None:
            return
        if not self.remote_client.is_available():
            self.kafka_publisher.publish_latency_probe(
                frame_number=frame_number,
                experiment_phase=self.shared_state.get_experiment_phase(),
                probe_mode="remote",
                latency_ms=None,
                status="unavailable",
            )
            return

        started = time.perf_counter()
        try:
            if self.remote_client.sends_raw_frames():
                self.remote_client.infer(frame)
            else:
                tensor = self._preprocess(frame)
                self.remote_client.infer(tensor, frame.shape)
            latency_ms = (time.perf_counter() - started) * 1000.0
            self.kafka_publisher.publish_latency_probe(
                frame_number=frame_number,
                experiment_phase=self.shared_state.get_experiment_phase(),
                probe_mode="remote",
                latency_ms=latency_ms,
            )
        except Exception as exc:
            logger.info("Remote latency probe failed: %s", exc)
            self.kafka_publisher.publish_latency_probe(
                frame_number=frame_number,
                experiment_phase=self.shared_state.get_experiment_phase(),
                probe_mode="remote",
                latency_ms=None,
                status="failed",
                error=str(exc),
            )

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        resized = cv2.resize(frame, (self.config.input_width, self.config.input_height))
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        normalized = rgb.astype(np.float32) / 255.0
        chw = np.transpose(normalized, (2, 0, 1))
        return np.expand_dims(chw, axis=0)
