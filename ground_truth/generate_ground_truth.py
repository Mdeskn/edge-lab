"""
Generate ground truth car coordinates from a drone-view video using OpenCV only.

MOG2 background subtraction detects the car as a moving foreground blob.
A two-pass approach is used so that the background model is already warm from
frame 1 and no frames are skipped.

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


CAR_CLASS = 2

# MOG2 tuning
_MOG2_HISTORY = 200       # frames used to model the background
_MOG2_VAR_THRESHOLD = 40  # lower = more sensitive to motion
_PREPASS_FRAMES = 30      # frames used in the pre-pass to warm up the model
_MIN_AREA_FRAC = 0.0005   # blob must be >= 0.05 % of frame (filters pixel noise)
_MAX_AREA_FRAC = 0.20     # blob must be <= 20 % of frame (filters full-frame clutter)
_MORPH_OPEN_K = 5         # kernel size for opening  (removes small speckles)
_MORPH_CLOSE_K = 25       # kernel size for closing  (fills holes inside the car blob)


def _detect_car(frame, bg_subtractor):
    """
    Return (cx, cy, 1.0, CAR_CLASS) for the largest moving foreground blob,
    or (None, None, 0.0, -1) when no car-sized blob is found.
    """
    fg_mask = bg_subtractor.apply(frame, learningRate=0)

    # Keep only definite foreground (255); discard shadows (127)
    _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)

    open_k = np.ones((_MORPH_OPEN_K, _MORPH_OPEN_K), dtype=np.uint8)
    close_k = np.ones((_MORPH_CLOSE_K, _MORPH_CLOSE_K), dtype=np.uint8)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, open_k)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, close_k)

    contours, _ = cv2.findContours(
        fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return None, None, 0.0, -1

    frame_area = frame.shape[0] * frame.shape[1]
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        frac = area / frame_area
        if _MIN_AREA_FRAC <= frac <= _MAX_AREA_FRAC:
            x, y, w, h = cv2.boundingRect(contour)
            candidates.append((area, x, y, w, h))

    if not candidates:
        return None, None, 0.0, -1

    _, x, y, w, h = max(candidates)
    return x + w / 2.0, y + h / 2.0, 1.0, CAR_CLASS


def _interpolate_gaps(csv_path):
    """
    Read the CSV and linearly interpolate any NaN rows between known detections.

    Leading NaN rows (before the first detection) are filled with the first
    known position. Trailing NaN rows (after the last detection) are filled
    with the last known position.
    Returns a list of [frame_number, cx, cy, conf, cls_id, class_name] rows.
    """
    import csv as _csv

    rows = []
    with open(csv_path, newline="") as f:
        reader = _csv.reader(f)
        next(reader)  # skip header
        for row in reader:
            rows.append(row)

    # Convert to mutable lists; parse numeric positions
    data = []
    for row in rows:
        fn = int(row[0])
        if row[1] == "NaN":
            data.append([fn, None, None, 0.0, -1, "none"])
        else:
            data.append([fn, float(row[1]), float(row[2]), float(row[3]), int(row[4]), row[5]])

    n = len(data)
    # Find first and last detected frames
    detected_indices = [i for i, d in enumerate(data) if d[1] is not None]
    if not detected_indices:
        return data  # nothing to interpolate

    first_det = detected_indices[0]
    last_det = detected_indices[-1]

    # Fill leading NaNs with first known position
    for i in range(first_det):
        fx, fy = data[first_det][1], data[first_det][2]
        data[i][1], data[i][2] = round(fx, 2), round(fy, 2)
        data[i][3], data[i][4], data[i][5] = 1.0, CAR_CLASS, "car"

    # Fill trailing NaNs with last known position
    for i in range(last_det + 1, n):
        fx, fy = data[last_det][1], data[last_det][2]
        data[i][1], data[i][2] = round(fx, 2), round(fy, 2)
        data[i][3], data[i][4], data[i][5] = 1.0, CAR_CLASS, "car"

    # Linearly interpolate interior gaps
    i = 0
    while i < n:
        if data[i][1] is None:
            # Find the next detected frame
            j = i + 1
            while j < n and data[j][1] is None:
                j += 1
            if j < n:
                x0, y0 = data[i - 1][1], data[i - 1][2]
                x1, y1 = data[j][1], data[j][2]
                gap = j - (i - 1)
                for k in range(i, j):
                    t = (k - (i - 1)) / gap
                    data[k][1] = round(x0 + t * (x1 - x0), 2)
                    data[k][2] = round(y0 + t * (y1 - y0), 2)
                    data[k][3], data[k][4], data[k][5] = 1.0, CAR_CLASS, "car"
            i = j
        else:
            i += 1

    return data


def _write_csv(csv_path, rows):
    """Overwrite csv_path with the interpolated rows."""
    import csv as _csv

    with open(csv_path, "w", newline="") as f:
        writer = _csv.writer(f)
        writer.writerow([
            "frame_number", "center_x", "center_y",
            "confidence", "class_id", "class_name",
        ])
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(
        description="Generate car ground truth from a drone-view video (OpenCV MOG2)."
    )
    parser.add_argument("--video", required=True, help="Path to the video file.")
    parser.add_argument("--output", default="ground_truth.csv", help="Output CSV path.")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR: cannot open video: {args.video}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video : {args.video} ({total_frames} frames)")

    # --- Pre-pass: warm up the background model on the first N frames ---
    print(f"Pre-pass: building background model from first {_PREPASS_FRAMES} frames ...")
    bg_subtractor = cv2.createBackgroundSubtractorMOG2(
        history=_MOG2_HISTORY,
        varThreshold=_MOG2_VAR_THRESHOLD,
        detectShadows=True,
    )
    for _ in range(_PREPASS_FRAMES):
        ret, frame = cap.read()
        if not ret:
            break
        bg_subtractor.apply(frame)  # default learningRate; updates the model

    # Rewind to frame 0 for the tracking pass
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # --- Tracking pass: model is frozen (learningRate=0) from here on ---
    print("Tracking pass: detecting car in all frames ...")
    frame_num = 0
    detected = 0

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame_number", "center_x", "center_y",
            "confidence", "class_id", "class_name",
        ])

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_num += 1
            cx, cy, conf, cls_id = _detect_car(frame, bg_subtractor)

            if cx is None:
                writer.writerow([frame_num, "NaN", "NaN", 0.0, -1, "none"])
            else:
                detected += 1
                writer.writerow([
                    frame_num,
                    round(cx, 2),
                    round(cy, 2),
                    round(conf, 4),
                    cls_id,
                    "car",
                ])

            if frame_num % 100 == 0:
                print(f"  Progress: {frame_num}/{total_frames}")

    cap.release()

    print(f"Detection rate before interpolation: {100 * detected / frame_num:.1f}% ({detected}/{frame_num} frames)")

    rows = _interpolate_gaps(args.output)
    _write_csv(args.output, rows)

    filled = sum(1 for r in rows if r[1] != "NaN")
    print(f"\nDone")
    print(f"Detection rate after interpolation:  {100 * filled / len(rows):.1f}% ({filled}/{len(rows)} frames)")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
