"""YOLOv10 output parsing shared by the local and remote inference paths."""
import numpy as np
import pytest

from inference.yolo_postprocess import best_detection_box

NO_DETECTION = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def boxes(*rows) -> np.ndarray:
    """Build a (1, N, 6) YOLOv10 output: [x1, y1, x2, y2, conf, class_id]."""
    return np.array([list(rows)], dtype=np.float32)


def test_returns_the_highest_confidence_box_in_original_coordinates() -> None:
    # Model space is 640x640; the original frame is 1280x640, so x doubles.
    output = boxes([100.0, 100.0, 200.0, 200.0, 0.9, 2.0])
    cx, cy, x1, y1, x2, y2 = best_detection_box(
        output, orig_h=640, orig_w=1280, input_h=640, input_w=640,
        conf_threshold=0.25,
    )
    assert (x1, y1, x2, y2) == (200.0, 100.0, 400.0, 200.0)
    assert (cx, cy) == (300.0, 150.0)


def test_below_threshold_is_a_miss() -> None:
    output = boxes([10.0, 10.0, 20.0, 20.0, 0.1, 2.0])
    result = best_detection_box(
        output, orig_h=640, orig_w=640, input_h=640, input_w=640,
        conf_threshold=0.25,
    )
    assert result == NO_DETECTION


def test_picks_the_most_confident_of_several() -> None:
    output = boxes(
        [0.0, 0.0, 10.0, 10.0, 0.4, 2.0],
        [100.0, 100.0, 110.0, 110.0, 0.8, 2.0],
        [200.0, 200.0, 210.0, 210.0, 0.6, 2.0],
    )
    cx, cy, *_ = best_detection_box(
        output, orig_h=640, orig_w=640, input_h=640, input_w=640,
        conf_threshold=0.25,
    )
    assert (cx, cy) == (105.0, 105.0)


def test_class_filter_excludes_other_classes() -> None:
    output = boxes(
        [0.0, 0.0, 10.0, 10.0, 0.9, 0.0],    # person, more confident
        [100.0, 100.0, 110.0, 110.0, 0.5, 2.0],  # car
    )
    cx, cy, *_ = best_detection_box(
        output, orig_h=640, orig_w=640, input_h=640, input_w=640,
        conf_threshold=0.25, target_class_id=2, target_conf_threshold=0.25,
    )
    assert (cx, cy) == (105.0, 105.0)


def test_class_filter_accepts_a_tuple_of_classes() -> None:
    """The lab tracks a red car YOLO sometimes labels as a truck (class 7)."""
    output = boxes([100.0, 100.0, 110.0, 110.0, 0.5, 7.0])
    cx, cy, *_ = best_detection_box(
        output, orig_h=640, orig_w=640, input_h=640, input_w=640,
        conf_threshold=0.25, target_class_id=(2, 7), target_conf_threshold=0.25,
    )
    assert (cx, cy) == (105.0, 105.0)


def test_target_threshold_overrides_the_general_threshold() -> None:
    output = boxes([100.0, 100.0, 110.0, 110.0, 0.15, 2.0])
    filtered = best_detection_box(
        output, orig_h=640, orig_w=640, input_h=640, input_w=640,
        conf_threshold=0.25, target_class_id=2, target_conf_threshold=0.1,
    )
    assert filtered[0] == pytest.approx(105.0)


def test_no_boxes_at_all_is_a_miss() -> None:
    output = np.zeros((1, 0, 6), dtype=np.float32)
    assert best_detection_box(
        output, orig_h=640, orig_w=640, input_h=640, input_w=640,
        conf_threshold=0.25,
    ) == NO_DETECTION


def test_malformed_output_is_a_miss_rather_than_a_crash() -> None:
    """A backend returning an unexpected shape must not kill the pipeline."""
    output = np.zeros((1, 3, 4), dtype=np.float32)
    assert best_detection_box(
        output, orig_h=640, orig_w=640, input_h=640, input_w=640,
        conf_threshold=0.25,
    ) == NO_DETECTION
