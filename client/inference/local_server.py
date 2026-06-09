"""
Wraps YOLOv10n ONNX for CPU inference via onnxruntime.
Loaded once at startup. Thread-safe (onnxruntime sessions are thread-safe).
"""
import logging
import os

import numpy as np
import onnxruntime as ort

from inference.yolo_postprocess import best_detection_box
from inference.yolo_postprocess import TargetClassFilter

logger = logging.getLogger(__name__)


class LocalServer:
    """YOLOv10n ONNX inference server using onnxruntime on CPU."""

    def __init__(
        self,
        model_path: str,
        conf_threshold: float = 0.3,
        target_class_id: TargetClassFilter = None,
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

        Returns (cx, cy, x1, y1, x2, y2) in original frame pixel coordinates.
        Returns (0.0, 0.0, 0.0, 0.0, 0.0, 0.0) when no detection is above threshold.
        """
        outputs = self._session.run(
            [self._output_name],
            {self._input_name: preprocessed_frame},
        )
        orig_h, orig_w = original_shape[:2]
        _, _, input_h, input_w = preprocessed_frame.shape
        return self._postprocess(outputs[0], orig_h, orig_w, input_h, input_w)

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
                "No target detection for class=%s above confidence threshold %.2f",
                self.target_class_id if self.target_class_id is not None else "any",
                (
                    self.target_conf_threshold
                    if self.target_class_id is not None
                    else self.conf_threshold
                ),
            )
        return result
