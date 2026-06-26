"""
Generate ground truth red-car coordinates from a video using OpenCV only.

Uses HSV colour segmentation to locate the red car in each frame.
Works regardless of camera motion because it is not a background model — it
looks for the target colour directly.

Ground truth is kept independent of YOLO so that YOLO misses remain measurable.

Usage:
    python generate_ground_truth.py \
        --video data/test_video.mp4 \
        --output data/ground_truth.csv

Output:
    frame_number,center_x,center_y,confidence,class_id,class_name
"""

import argparse
import csv
import sys

import cv2
import numpy as np


TARGET_CLASS = 2
TARGET_NAME = "car"

# HSV ranges for red. Red wraps around hue 0 in OpenCV (0-179), so we need
# two bands and OR them together.
_RED_LO1 = np.array([  0,  80,  50], dtype=np.uint8)
_RED_HI1 = np.array([ 10, 255, 255], dtype=np.uint8)
_RED_LO2 = np.array([160,  80,  50], dtype=np.uint8)
_RED_HI2 = np.array([179, 255, 255], dtype=np.uint8)

# Morphological kernels
_OPEN_K  = 7    # removes small noise speckles
_CLOSE_K = 25   # fills holes inside the target blob

# Target size relative to frame area
_MIN_AREA_FRAC = 0.001    # at least 0.1 % of frame
_MAX_AREA_FRAC = 0.60     # at most 60 % of frame


def _detect_target(frame: np.ndarray):
    """
    Return (cx, cy, 1.0, TARGET_CLASS) for the largest red blob in the
    frame, or (None, None, 0.0, -1) when no plausible target-sized red
    region is found.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.bitwise_or(
        cv2.inRange(hsv, _RED_LO1, _RED_HI1),
        cv2.inRange(hsv, _RED_LO2, _RED_HI2),
    )

    open_k  = np.ones((_OPEN_K,  _OPEN_K),  dtype=np.uint8)
    close_k = np.ones((_CLOSE_K, _CLOSE_K), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  open_k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_k)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, 0.0, -1

    frame_area = frame.shape[0] * frame.shape[1]
    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        frac = area / frame_area
        if _MIN_AREA_FRAC <= frac <= _MAX_AREA_FRAC:
            x, y, w, h = cv2.boundingRect(c)
            candidates.append((area, x, y, w, h))

    if not candidates:
        return None, None, 0.0, -1

    # Largest red blob = the target vehicle.
    _, x, y, w, h = max(candidates)
    return x + w / 2.0, y + h / 2.0, 1.0, TARGET_CLASS


def _interpolate_gaps(rows: list) -> list:
    """
    Linearly interpolate NaN rows between known detections.

    Leading NaN rows are filled with the first known position.
    Trailing NaN rows are filled with the last known position.
    """
    n = len(rows)
    detected = [i for i, r in enumerate(rows) if r[1] is not None]
    if not detected:
        return rows

    first, last = detected[0], detected[-1]

    # Fill leading gap
    for i in range(first):
        rows[i][1:6] = [
            rows[first][1], rows[first][2], 1.0,
            TARGET_CLASS, TARGET_NAME,
        ]

    # Fill trailing gap
    for i in range(last + 1, n):
        rows[i][1:6] = [
            rows[last][1], rows[last][2], 1.0,
            TARGET_CLASS, TARGET_NAME,
        ]

    # Fill interior gaps
    i = 0
    while i < n:
        if rows[i][1] is None:
            j = i + 1
            while j < n and rows[j][1] is None:
                j += 1
            if j < n:
                x0, y0 = rows[i - 1][1], rows[i - 1][2]
                x1, y1 = rows[j][1], rows[j][2]
                gap = j - (i - 1)
                for k in range(i, j):
                    t = (k - (i - 1)) / gap
                    rows[k][1] = round(x0 + t * (x1 - x0), 2)
                    rows[k][2] = round(y0 + t * (y1 - y0), 2)
                    rows[k][3], rows[k][4], rows[k][5] = (
                        1.0, TARGET_CLASS, TARGET_NAME
                    )
            i = j
        else:
            i += 1

    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Generate red-car GT from a video via HSV colour detection."
    )
    parser.add_argument("--video",  required=True, help="Path to the video file.")
    parser.add_argument("--output", default="ground_truth.csv", help="Output CSV path.")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: cannot open video: {args.video}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video  : {args.video} ({total_frames} frames)")
    print("Tracker: HSV red-car colour segmentation")

    frame_num = 0
    detected  = 0
    rows      = []   # accumulate before interpolation

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1

        cx, cy, conf, cls_id = _detect_target(frame)

        if cx is None:
            rows.append([frame_num, None, None, 0.0, -1, "none"])
        else:
            detected += 1
            rows.append([
                frame_num, round(cx, 2), round(cy, 2),
                round(conf, 4), cls_id, TARGET_NAME,
            ])

        if frame_num % 100 == 0:
            print(f"  Progress: {frame_num}/{total_frames}")

    cap.release()

    print(f"Detection rate before interpolation: {100*detected/frame_num:.1f}% ({detected}/{frame_num})")

    rows = _interpolate_gaps(rows)

    # Write CSV
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame_number", "center_x", "center_y",
                         "confidence", "class_id", "class_name"])
        for row in rows:
            writer.writerow(row)

    filled = sum(1 for r in rows if r[1] is not None)
    print(f"Detection rate after  interpolation: {100*filled/len(rows):.1f}% ({filled}/{len(rows)})")
    print(f"Output : {args.output}")


if __name__ == "__main__":
    main()
