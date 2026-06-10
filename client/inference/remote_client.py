"""Remote inference clients used by the Pi."""
import logging

import cv2
import numpy as np
import requests
import tritonclient.grpc as grpcclient

from inference.yolo_postprocess import best_detection_box
from inference.yolo_postprocess import TargetClassFilter

logger = logging.getLogger(__name__)


class RemoteClient:
    """
    Remote inference client.

    Preferred mode sends a compressed JPEG to the VM2 remote inference API. If
    REMOTE_INFERENCE_URL is not configured, this falls back to the legacy direct
    Triton gRPC path, which sends a preprocessed FP32 tensor over the network.
    """

    def __init__(
        self,
        triton_url: str,
        model_name: str = "yolov10n",
        conf_threshold: float = 0.3,
        target_class_id: TargetClassFilter = None,
        target_conf_threshold: float | None = None,
        timeout: float = 5.0,
        remote_inference_url: str = "",
        jpeg_quality: int = 80,
    ):
        """
        Initialize the selected remote client and perform a health check.

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
        self._remote_inference_url = remote_inference_url.rstrip("/")
        self._jpeg_quality = min(max(jpeg_quality, 1), 100)
        self._mode = "http_jpeg" if self._remote_inference_url else "triton_grpc"
        self._available = False
        self._client = None

        if self._mode == "http_jpeg":
            self._connect_http_gateway()
        else:
            self._connect_triton_grpc(triton_url)

    def is_available(self) -> bool:
        """Return True if the configured remote endpoint was reachable at startup."""
        return self._available

    def sends_raw_frames(self) -> bool:
        """Return True when the client should pass raw BGR frames instead of tensors."""
        return self._mode == "http_jpeg"

    def infer(self, frame_or_tensor: np.ndarray, original_shape: tuple | None = None) -> tuple:
        """Run inference with the selected remote transport."""
        if self._mode == "http_jpeg":
            return self._infer_http_jpeg(frame_or_tensor)

        if original_shape is None:
            raise ValueError("original_shape is required for legacy Triton gRPC inference")
        return self._infer_triton_grpc(frame_or_tensor, original_shape)

    def _connect_http_gateway(self) -> None:
        """Connect to the VM2 JPEG inference API."""
        try:
            resp = requests.get(
                f"{self._remote_inference_url}/health",
                timeout=self._timeout,
            )
            resp.raise_for_status()
            self._available = True
            logger.info(
                "RemoteClient connected to JPEG inference API at %s",
                self._remote_inference_url,
            )
        except Exception as exc:
            logger.error(
                "RemoteClient failed to connect to JPEG inference API at %s: %s",
                self._remote_inference_url,
                exc,
            )

    def _connect_triton_grpc(self, triton_url: str) -> None:
        """Connect to Triton directly using the legacy FP32 tensor transport."""
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

    def _infer_http_jpeg(self, frame: np.ndarray) -> tuple:
        """
        JPEG-compress a raw BGR frame and send it to the VM2 inference API.

        Returns (cx, cy, x1, y1, x2, y2) in original frame pixel coordinates.
        Raises on encode, HTTP, timeout, or response errors; the Dispatcher handles
        local fallback.
        """
        ok, jpeg = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, self._jpeg_quality],
        )
        if not ok:
            raise RuntimeError("Could not encode frame as JPEG")

        resp = requests.post(
            f"{self._remote_inference_url}/infer",
            data=jpeg.tobytes(),
            headers={"Content-Type": "image/jpeg"},
            timeout=self._timeout,
        )
        resp.raise_for_status()
        payload = resp.json()
        prediction = payload.get("prediction", payload)
        return (
            float(prediction.get("cx", 0.0)),
            float(prediction.get("cy", 0.0)),
            float(prediction.get("x1", 0.0)),
            float(prediction.get("y1", 0.0)),
            float(prediction.get("x2", 0.0)),
            float(prediction.get("y2", 0.0)),
        )

    def _infer_triton_grpc(self, preprocessed_frame: np.ndarray, original_shape: tuple) -> tuple:
        """
        Send a preprocessed frame to Triton and return (cx, cy, x1, y1, x2, y2).

        Builds an InferInput named 'images' using the preprocessed frame shape, FP32.
        Requests output named 'output0'.
        Scales result coordinates back to original frame dimensions.

        Raises on network timeout or connection error; the Dispatcher handles fallback.
        """
        if self._client is None:
            raise RuntimeError("Triton gRPC client is not initialized")

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
