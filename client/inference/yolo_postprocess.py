"""Shared YOLOv10 post-processing helpers for local and remote inference."""
import numpy as np

TargetClassFilter = int | tuple[int, ...] | None


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
    """
    Return the best target detection center in original-frame pixel coordinates.

    Expected YOLOv10 output rows are:
        [x1, y1, x2, y2, confidence, class_id]

    Returns (0.0, 0.0) when no detection passes the configured threshold.
    """
    boxes = np.asarray(output)
    if boxes.ndim == 3 and boxes.shape[0] == 1:
        boxes = boxes[0]
    else:
        boxes = np.squeeze(boxes)

    if boxes.ndim == 1:
        boxes = boxes.reshape(1, -1)
    if boxes.ndim != 2 or boxes.shape[1] < 6:
        return (0.0, 0.0)

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
        return (0.0, 0.0)

    best = filtered[filtered[:, 4].argmax()]
    x1, y1, x2, y2 = best[:4]

    center_x = (x1 + x2) / 2.0 * orig_w / input_w
    center_y = (y1 + y2) / 2.0 * orig_h / input_h

    return (float(center_x), float(center_y))
