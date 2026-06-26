#!/usr/bin/env python3
"""Generate ground-truth CSV rows from the best YOLO detection per frame."""
from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import cv2
import onnxruntime as ort

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from test_yolo_image import class_name, detections, parse_class_filter, preprocess  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate ground_truth.csv from YOLO detections."
    )
    parser.add_argument("--video", required=True, help="Path to the video file.")
    parser.add_argument("--model", default="data/yolov10n.onnx", help="ONNX model path.")
    parser.add_argument("--output", default="data/ground_truth.csv", help="Output CSV path.")
    parser.add_argument("--conf", type=float, default=0.1, help="YOLO confidence threshold.")
    parser.add_argument(
        "--target-class-id",
        default="2,7",
        help="Comma-separated COCO class ids; 2=car, 7=truck.",
    )
    parser.add_argument("--width", type=int, default=640, help="Model input width.")
    parser.add_argument("--height", type=int, default=640, help="Model input height.")
    args = parser.parse_args()

    video_path = Path(args.video)
    model_path = Path(args.model)
    output_path = Path(args.output)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    class_filter = parse_class_filter(args.target_class_id)

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    session = ort.InferenceSession(
        str(model_path),
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    rows = []
    counts: Counter[str] = Counter()
    missing = 0
    frame_number = 0

    print(f"Video  : {video_path} ({total_frames} frames)")
    print(f"Model  : {model_path}")
    print(f"Tracker: YOLO best detection, conf={args.conf}, target={class_filter}")

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_number += 1

        output = session.run(
            [output_name],
            {input_name: preprocess(frame, args.width, args.height)},
        )[0]
        found = detections(output, args.conf, class_filter)

        if len(found) == 0:
            rows.append([frame_number, "", "", 0.0, -1, "none"])
            missing += 1
        else:
            x1, y1, x2, y2, confidence, class_id_float = found[0][:6]
            class_id = int(class_id_float)
            orig_h, orig_w = frame.shape[:2]
            scale_x = orig_w / args.width
            scale_y = orig_h / args.height
            center_x = ((float(x1) + float(x2)) / 2.0) * scale_x
            center_y = ((float(y1) + float(y2)) / 2.0) * scale_y
            name = class_name(class_id)
            counts[name] += 1
            rows.append([
                frame_number,
                round(center_x, 2),
                round(center_y, 2),
                round(float(confidence), 6),
                class_id,
                name,
            ])

        if frame_number % 100 == 0:
            print(f"  Progress: {frame_number}/{total_frames}")

    cap.release()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "frame_number",
            "center_x",
            "center_y",
            "confidence",
            "class_id",
            "class_name",
        ])
        writer.writerows(rows)

    detected = frame_number - missing
    print(f"Detection rate: {100 * detected / frame_number:.1f}% ({detected}/{frame_number})")
    if counts:
        labels = ", ".join(f"{name}:{count}" for name, count in sorted(counts.items()))
        print(f"Classes: {labels}")
    print(f"Output : {output_path}")


if __name__ == "__main__":
    main()
