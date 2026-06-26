"""Remote inference clients used by the Pi."""
import logging
import time
from http import HTTPStatus

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
        failure_cooldown: float = 3.0,
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
        self._triton_url = triton_url
        self._timeout = timeout
        self._remote_inference_url = remote_inference_url.rstrip("/")
        self._jpeg_quality = min(max(jpeg_quality, 1), 100)
        self._mode = "http_jpeg" if self._remote_inference_url else "triton_grpc"
        self._available = False
        self._client = None
        self._failure_cooldown = failure_cooldown
        self._last_failure_time: float | None = None
        self._session = requests.Session()

        if self._mode == "http_jpeg":
            self._connect_http_gateway()
        else:
            self._connect_triton_grpc(triton_url)

    def is_available(self, ignore_cooldown: bool = False) -> bool:
        """
        Return True if the remote endpoint is reachable and has not failed
        recently. If the startup check failed, retry after the cooldown window.

        A single timed-out call would otherwise block the Dispatcher's single
        processing thread for the full per-call timeout on every subsequent
        frame still routed to "remote", repeating until the SP-Agent reacts.
        Treating a recent failure as "unavailable" for a cooldown window lets
        the Dispatcher skip straight to local without waiting on that timeout
        again, and the client retries remote on its own once the cooldown
        elapses.

        Forced-manual remote mode passes ignore_cooldown=True so a slow frame
        does not turn into a multi-second "remote unavailable" gap.
        """
        if not self._available:
            now = time.time()
            if (
                ignore_cooldown
                or self._failure_cooldown <= 0
                or self._last_failure_time is None
                or now - self._last_failure_time >= self._failure_cooldown
            ):
                if self._mode == "http_jpeg":
                    self._connect_http_gateway()
                else:
                    self._connect_triton_grpc(self._triton_url)
            return self._available
        if self._last_failure_time is not None:
            if ignore_cooldown or self._failure_cooldown <= 0:
                return True
            if time.time() - self._last_failure_time < self._failure_cooldown:
                return False
        return True

    def sends_raw_frames(self) -> bool:
        """Return True when the client should pass raw BGR frames instead of tensors."""
        return self._mode == "http_jpeg"

    def infer(self, frame_or_tensor: np.ndarray, original_shape: tuple | None = None) -> tuple:
        """Run inference with the selected remote transport.

        Records the failure time on any exception so is_available() enters
        its cooldown window; the Dispatcher's caller handles local fallback.
        """
        try:
            if self._mode == "http_jpeg":
                return self._infer_http_jpeg(frame_or_tensor)

            if original_shape is None:
                raise ValueError("original_shape is required for legacy Triton gRPC inference")
            return self._infer_triton_grpc(frame_or_tensor, original_shape)
        except Exception:
            self._last_failure_time = time.time()
            raise

    def _connect_http_gateway(self) -> None:
        """Connect to the VM2 JPEG inference API."""
        try:
            resp = self._session.get(
                f"{self._remote_inference_url}/health",
                timeout=self._timeout,
            )
            resp.raise_for_status()
            self._available = True
            self._last_failure_time = None
            logger.info(
                "RemoteClient connected to JPEG inference API at %s",
                self._remote_inference_url,
            )
        except Exception as exc:
            self._available = False
            self._last_failure_time = time.time()
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
                self._last_failure_time = None
                logger.info("RemoteClient connected to Triton at %s", triton_url)
            else:
                self._available = False
                self._last_failure_time = time.time()
                logger.error(
                    "Triton health check returned not-live for %s", triton_url
                )
        except Exception as exc:
            self._available = False
            self._last_failure_time = time.time()
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

        resp = self._session.post(
            f"{self._remote_inference_url}/infer",
            params=self._target_query_params(),
            data=jpeg.tobytes(),
            headers={"Content-Type": "image/jpeg", "Connection": "close"},
            timeout=self._timeout,
        )
        if not resp.ok:
            detail = resp.text.strip().replace("\n", " ")[:500]
            status_name = HTTPStatus(resp.status_code).phrase
            raise requests.HTTPError(
                f"{resp.status_code} {status_name} from JPEG inference API: {detail}",
                response=resp,
            )
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

    def _target_query_params(self) -> dict[str, str]:
        """Return per-request target filtering settings for the JPEG gateway."""
        params = {
            "conf_threshold": str(self.conf_threshold),
            "target_conf_threshold": str(self.target_conf_threshold),
        }
        if self.target_class_id is not None:
            if isinstance(self.target_class_id, int):
                params["target_class_id"] = str(self.target_class_id)
            else:
                params["target_class_id"] = ",".join(
                    str(class_id) for class_id in self.target_class_id
                )
        return params

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
                "Triton: no target detection for class=%s above confidence threshold %.4g",
                self.target_class_id if self.target_class_id is not None else "any",
                (
                    self.target_conf_threshold
                    if self.target_class_id is not None
                    else self.conf_threshold
                ),
            )
        return result
