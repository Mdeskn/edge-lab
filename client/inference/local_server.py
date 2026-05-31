"""
Wraps YOLOv10n ONNX for CPU inference via onnxruntime.
Loaded once at startup. Thread-safe (onnxruntime sessions are thread-safe).
"""
import logging
import os

import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)


class LocalServer:
    """YOLOv10n ONNX inference server using onnxruntime on CPU."""

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.3,
        target_class_id: int | None = None,
        target_conf_threshold: float | None = None,
    ):
        """
        Load ONNX model from model_path.

        Sets intra_op_num_threads=4 to use all Pi cores.
        Raises FileNotFoundError if model_path does not exist.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"ONNX model not found: {model_path}")

        self.conf_threshold = conf_threshold
        self.target_class_id = target_class_id
        self.target_conf_threshold = (
            target_conf_threshold
            if target_conf_threshold is not None
            else conf_threshold
        )

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4

        self._session = ort.InferenceSession(
            model_path,
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )

        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name

        logger.info(
            "LocalServer loaded model=%s input=%s output=%s target_class=%s",
            model_path,
            self._input_name,
            self._output_name,
            self.target_class_id if self.target_class_id is not None else "any",
        )

    def infer(self, preprocessed_frame: np.ndarray, original_shape: tuple) -> tuple:
        """
        Run inference on a preprocessed frame (shape: 1, 3, H, W, float32).

        Returns (center_x, center_y) in original frame pixel coordinates.
        Returns (0.0, 0.0) and logs a WARNING if no detection is above threshold.
        """
        outputs = self._session.run(
            [self._output_name],
            {self._input_name: preprocessed_frame},
        )
        orig_h, orig_w = original_shape[:2]
        return self._postprocess(outputs[0], orig_h, orig_w)

    def _postprocess(self, output: np.ndarray, orig_h: int, orig_w: int) -> tuple:
        """
        Parse YOLOv10 output and return the center of the best target detection.

        YOLOv10 output shape: (1, num_boxes, 6).
        Each box: [x1, y1, x2, y2, confidence, class_id].
        Coordinates are in model input space (640×640).

        Scales the result back to original frame coordinates.
        Returns (0.0, 0.0) when no box passes the confidence threshold.
        """
        boxes = output.squeeze(0)  # (num_boxes, 6)

        threshold = (
            self.target_conf_threshold
            if self.target_class_id is not None
            else self.conf_threshold
        )
        mask = boxes[:, 4] >= threshold
        if self.target_class_id is not None:
            mask &= boxes[:, 5].astype(int) == self.target_class_id
        filtered = boxes[mask]

        if len(filtered) == 0:
            logger.debug(
                "No target detection for class=%s above confidence threshold %.2f",
                self.target_class_id if self.target_class_id is not None else "any",
                threshold,
            )
            return (0.0, 0.0)

        best = filtered[filtered[:, 4].argmax()]
        x1, y1, x2, y2 = best[:4]

        center_x = (x1 + x2) / 2.0 * orig_w / 640.0
        center_y = (y1 + y2) / 2.0 * orig_h / 640.0

        return (float(center_x), float(center_y))
