"""Frame preprocessing shared by the Dispatcher and the latency probe.

Both paths must feed the model byte-identical tensors: if they diverge, the
probe stops being a valid estimate of what the active backend would have cost,
and the SP-Agent's recovery decisions are made on a number that measures
something slightly different.
"""
import cv2
import numpy as np


def preprocess_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    Prepare a BGR video frame for YOLOv10n inference.

    Steps: resize -> BGR to RGB -> normalize to [0,1] -> HWC to CHW -> batch.
    Returns a float32 array of shape (1, 3, height, width).
    """
    resized = cv2.resize(frame, (width, height))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normalized = rgb.astype(np.float32) / 255.0
    chw = np.transpose(normalized, (2, 0, 1))
    return np.expand_dims(chw, axis=0)
