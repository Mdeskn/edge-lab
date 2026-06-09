"""Shared YOLOv10 post-processing helpers for local and remote inference."""
import numpy as np

TargetClassFilter = int | tuple[int, ...] | None

_NO_DETECTION = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def best_detection_box(
    output: np.ndarray,
    orig_h: int,
    orig_w: int,
    input_h: int,
    input_w: int,
    conf_threshold: float,
    target_class_id: TargetClassFilter = None,
    target_conf_threshold: float | None = None,
) -> tuple[float, float, float, float, float, float]:
    """
    Return the best target detection as (cx, cy, x1, y1, x2, y2) in original-frame
    pixel coordinates.

    Expected YOLOv10 output rows are:
        [x1, y1, x2, y2, confidence, class_id]

    Returns (0.0, 0.0, 0.0, 0.0, 0.0, 0.0) when no detection passes the threshold.
    """
    boxes = np.asarray(output)
    if boxes.ndim == 3 and boxes.shape[0] == 1:
        boxes = boxes[0]
    else:
        boxes = np.squeeze(boxes)

    if boxes.ndim == 1:
        boxes = boxes.reshape(1, -1)
    if boxes.ndim != 2 or boxes.shape[1] < 6:
        return _NO_DETECTION

    threshold = (
        target_conf_threshold
        if target_class_id is not None and target_conf_threshold is not None
        else conf_threshold
    )
    mask = boxes[:, 4] >= threshold
    if target_class_id is not None:
        class_ids = boxes[:, 5].astype(int)
        if isinstance(target_class_id, int):
            mask &= class_ids == target_class_id
        else:
            mask &= np.isin(class_ids, target_class_id)

    filtered = boxes[mask]
    if len(filtered) == 0:
        return _NO_DETECTION

    best = filtered[filtered[:, 4].argmax()]
    x1, y1, x2, y2 = best[:4]

    scale_x = orig_w / input_w
    scale_y = orig_h / input_h

    rx1 = float(x1 * scale_x)
    ry1 = float(y1 * scale_y)
    rx2 = float(x2 * scale_x)
    ry2 = float(y2 * scale_y)
    cx = (rx1 + rx2) / 2.0
    cy = (ry1 + ry2) / 2.0

    return (cx, cy, rx1, ry1, rx2, ry2)


# Keep the old name as an alias so any external callers still work.
def best_detection_center(
    output: np.ndarray,
    orig_h: int,
    orig_w: int,
    input_h: int,
    input_w: int,
    conf_threshold: float,
    target_class_id: TargetClassFilter = None,
    target_conf_threshold: float | None = None,
) -> tuple[float, float]:
    result = best_detection_box(
        output, orig_h, orig_w, input_h, input_w,
        conf_threshold, target_class_id, target_conf_threshold,
    )
    return (result[0], result[1])
