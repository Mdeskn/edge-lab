"""
Sends frames to Triton Inference Server over gRPC.
Uses tritonclient.grpc for synchronous inference.
"""
import logging

import numpy as np
import tritonclient.grpc as grpcclient

from inference.yolo_postprocess import best_detection_box
from inference.yolo_postprocess import TargetClassFilter

logger = logging.getLogger(__name__)


class RemoteClient:
    """gRPC client for NVIDIA Triton Inference Server."""

    def __init__(
        self,
        triton_url: str,
        model_name: str = "yolov10n",
        conf_threshold: float = 0.3,
        target_class_id: TargetClassFilter = None,
        target_conf_threshold: float | None = None,
        timeout: float = 5.0,
    ):
        """
        Initialize Triton gRPC client and perform a health check.

        Sets self._available=False (and logs an error) if the health check fails.
        Never raises. The caller determines what to do with is_available().
        """
        self.model_name = model_name
        self.conf_threshold = conf_threshold
        self.target_class_id = target_class_id
        self.target_conf_threshold = (
            target_conf_threshold
            if target_conf_threshold is not None
            else conf_threshold
        )
        self._timeout = timeout
        self._available = False

        try:
            self._client = grpcclient.InferenceServerClient(url=triton_url)
            alive = self._client.is_server_live()
            if alive:
                self._available = True
                logger.info("RemoteClient connected to Triton at %s", triton_url)
            else:
                logger.error(
                    "Triton health check returned not-live for %s", triton_url
                )
        except Exception as exc:
            logger.error("RemoteClient failed to connect to Triton at %s: %s", triton_url, exc)

    def is_available(self) -> bool:
        """Return True if the Triton server was reachable at startup."""
        return self._available

    def infer(self, preprocessed_frame: np.ndarray, original_shape: tuple) -> tuple:
        """
        Send a preprocessed frame to Triton and return (cx, cy, x1, y1, x2, y2).

        Builds an InferInput named 'images' using the preprocessed frame shape, FP32.
        Requests output named 'output0'.
        Scales result coordinates back to original frame dimensions.

        Raises on network timeout or connection error; the Dispatcher handles fallback.
        """
        _, _, input_h, input_w = preprocessed_frame.shape
        inp = grpcclient.InferInput(
            "images",
            list(preprocessed_frame.shape),
            "FP32",
        )
        inp.set_data_from_numpy(preprocessed_frame)

        out = grpcclient.InferRequestedOutput("output0")

        result = self._client.infer(
            self.model_name,
            inputs=[inp],
            outputs=[out],
            client_timeout=self._timeout,
        )

        output_data = result.as_numpy("output0")  # (1, num_boxes, 6)
        orig_h, orig_w = original_shape[:2]
        return self._postprocess(output_data, orig_h, orig_w, input_h, input_w)

    def _postprocess(
        self,
        output: np.ndarray,
        orig_h: int,
        orig_w: int,
        input_h: int,
        input_w: int,
    ) -> tuple:
        """
        Parse YOLOv10 output and return (cx, cy, x1, y1, x2, y2) for the best detection.

        YOLOv10 output shape: (1, num_boxes, 6).
        Each box: [x1, y1, x2, y2, confidence, class_id].
        Coordinates are in model input space; scaled back to original frame coordinates.

        Returns (0.0, 0.0, 0.0, 0.0, 0.0, 0.0) when no box passes the confidence threshold.
        """
        result = best_detection_box(
            output=output,
            orig_h=orig_h,
            orig_w=orig_w,
            input_h=input_h,
            input_w=input_w,
            conf_threshold=self.conf_threshold,
            target_class_id=self.target_class_id,
            target_conf_threshold=self.target_conf_threshold,
        )
        if result[0] == 0.0 and result[1] == 0.0:
            logger.debug(
                "Triton: no target detection for class=%s above confidence threshold %.2f",
                self.target_class_id if self.target_class_id is not None else "any",
                (
                    self.target_conf_threshold
                    if self.target_class_id is not None
                    else self.conf_threshold
                ),
            )
        return result
