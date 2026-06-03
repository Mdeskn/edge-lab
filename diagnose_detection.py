#!/usr/bin/env python3
"""
Detection diagnostic for the Edge Lab.

It takes ONE frame from your video, runs the YOLO model on it using
known-correct preprocessing, and prints EVERY detection the model produced
(no class filter, no confidence threshold). This tells us whether the model
sees the ball at all, what label it gives it, and how confident it is.

It also saves two images so you can look with your own eyes:
  diag_frame_raw.jpg        - the plain frame (confirm the ball is visible)
  diag_frame_annotated.jpg  - the same frame with the top detections drawn on

Run it with:
  python3 diagnose_detection.py --model <path> --video <path> --frame 100

If you leave the paths out, it uses the defaults below. Change them if your
files live somewhere else (see the run instructions).
"""
import argparse
import os
import sys

import cv2
import numpy as np
import onnxruntime as ort

# The 80 things YOLO was trained to recognize, in order. Index 32 is "sports ball".
COCO = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]
INPUT_SIZE = 640


def parse_args():
    p = argparse.ArgumentParser(description="YOLO detection diagnostic.")
    p.add_argument("--model", default="/home/mae/edge-lab/data/yolov10n.onnx")
    p.add_argument("--video", default="/home/mae/edge-lab/data/test_video.mp4")
    p.add_argument("--frame", type=int, default=100,
                   help="Which frame to test. Pick one where the ball is clearly visible.")
    p.add_argument("--ground-truth", default="/home/mae/edge-lab/data/ground_truth.csv",
                   help="Optional. If found, prints where the answer key says the ball is.")
    return p.parse_args()


def load_frame(video_path, frame_number):
    if not os.path.exists(video_path):
        print(f"ERROR: video not found at {video_path}")
        print("Find your video with:  find /home -name '*.mp4'")
        sys.exit(1)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"ERROR: could not open video at {video_path}")
        sys.exit(1)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video opened. It has {total} frames. Grabbing frame {frame_number}.")
    if frame_number >= total:
        print(f"WARNING: frame {frame_number} is past the end. Using frame 0 instead.")
        frame_number = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print(f"ERROR: could not read frame {frame_number}.")
        sys.exit(1)
    return frame


def preprocess(frame):
    """The correct, standard YOLO preprocessing. If the app does this differently,
    that difference is the bug."""
    img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))     # 1. resize to 640x640
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)            # 2. BGR -> RGB
    img = img.astype(np.float32) / 255.0                 # 3. scale 0..255 -> 0..1
    img = np.transpose(img, (2, 0, 1))                   # 4. HWC -> CHW
    img = np.expand_dims(img, axis=0)                    # 5. add batch dimension
    return img


def read_ground_truth(path, frame_number):
    """Best-effort: print where the OpenCV answer key says the ball is for this frame."""
    if not os.path.exists(path):
        return None
    try:
        import csv
        with open(path, newline="") as f:
            rows = list(csv.reader(f))
        if not rows:
            return None
        header = [h.strip().lower() for h in rows[0]]
        # find a column for frame, x, y (handles common names)
        def col(*names):
            for n in names:
                if n in header:
                    return header.index(n)
            return None
        fi = col("frame", "frame_id", "frame_number", "idx")
        xi = col("x", "cx", "center_x", "x_center", "ball_x")
        yi = col("y", "cy", "center_y", "y_center", "ball_y")
        if xi is None or yi is None:
            return None
        for r in rows[1:]:
            if fi is not None:
                if str(r[fi]).strip() != str(frame_number):
                    continue
            else:
                # no frame column: assume row order matches frame order
                pass
            return (float(r[xi]), float(r[yi]))
    except Exception as exc:
        print(f"(could not read ground truth: {exc})")
    return None


def main():
    args = parse_args()

    frame = load_frame(args.video, args.frame)
    orig_h, orig_w = frame.shape[:2]
    print(f"Frame size: {orig_w} x {orig_h} pixels")
    cv2.imwrite("diag_frame_raw.jpg", frame)
    print("Saved diag_frame_raw.jpg  (open it and confirm you can see the ball)")

    if not os.path.exists(args.model):
        print(f"ERROR: model not found at {args.model}")
        print("Find your model with:  find /home -name '*.onnx'")
        sys.exit(1)

    img = preprocess(frame)

    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name
    print(f"Model loaded. input='{inp_name}'  output='{out_name}'")

    outputs = sess.run([out_name], {inp_name: img})
    out = outputs[0]
    print(f"Raw model output shape: {out.shape}")

    dets = np.squeeze(out, axis=0) if out.ndim == 3 else out
    if dets.ndim != 2 or dets.shape[1] < 6:
        print("\nThe output is not the expected [N, 6] table. First few rows of raw output:")
        print(dets[:5])
        print("\nTell Claude this output shape so we can adjust the parser.")
        return

    # sort detections by confidence, highest first
    dets = dets[dets[:, 4].argsort()[::-1]]

    print("\n================ TOP 15 DETECTIONS (no filter at all) ================")
    print(f"{'conf':>7}  {'class':>5}  {'label':<14}  center (x, y) in the real frame")
    print("-" * 68)
    annotated = frame.copy()
    for i, d in enumerate(dets[:15]):
        x1, y1, x2, y2, conf, cls = d[0], d[1], d[2], d[3], d[4], int(d[5])
        cx = (x1 + x2) / 2 * orig_w / INPUT_SIZE
        cy = (y1 + y2) / 2 * orig_h / INPUT_SIZE
        label = COCO[cls] if 0 <= cls < len(COCO) else f"class{cls}"
        print(f"{conf:7.3f}  {cls:5d}  {label:<14}  ({cx:6.0f}, {cy:6.0f})")
        # draw the top 3 on the annotated image
        if i < 3:
            ax1 = int(x1 * orig_w / INPUT_SIZE)
            ay1 = int(y1 * orig_h / INPUT_SIZE)
            ax2 = int(x2 * orig_w / INPUT_SIZE)
            ay2 = int(y2 * orig_h / INPUT_SIZE)
            cv2.rectangle(annotated, (ax1, ay1), (ax2, ay2), (0, 0, 255), 2)
            cv2.putText(annotated, f"{label} {conf:.2f}", (ax1, max(ay1 - 5, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    # mark where the answer key says the ball is
    gt = read_ground_truth(args.ground_truth, args.frame)
    if gt is not None:
        cv2.circle(annotated, (int(gt[0]), int(gt[1])), 8, (0, 255, 0), 2)
        print(f"\nGround truth (OpenCV answer key) ball position for this frame: "
              f"({gt[0]:.0f}, {gt[1]:.0f})  -> drawn as a GREEN circle")

    cv2.imwrite("diag_frame_annotated.jpg", annotated)
    print("Saved diag_frame_annotated.jpg  (red boxes = what YOLO found, green circle = truth)")

    # quick automatic read of the situation
    sports_ball = dets[dets[:, 5].astype(int) == 32]
    print("\n======================= QUICK INTERPRETATION =======================")
    if len(dets) == 0 or dets[0, 4] < 0.001:
        print("YOLO found essentially NOTHING. This strongly points to a")
        print("preprocessing bug in the app (BGR/RGB, /255, or layout).")
    elif len(sports_ball) > 0 and sports_ball[0, 4] >= 0.10:
        bx = (sports_ball[0, 0] + sports_ball[0, 2]) / 2 * orig_w / INPUT_SIZE
        by = (sports_ball[0, 1] + sports_ball[0, 3]) / 2 * orig_h / INPUT_SIZE
        print(f"YOLO DID find a sports ball (confidence {sports_ball[0,4]:.3f}) at "
              f"({bx:.0f}, {by:.0f}).")
        print("So the model and this preprocessing work. If the APP misses the ball,")
        print("the app's preprocessing or its config differs from this script.")
    elif len(sports_ball) > 0:
        print(f"YOLO found a sports ball but with LOW confidence ({sports_ball[0,4]:.3f}).")
        print("The nano model may be too weak for this ball. Consider a bigger model.")
    else:
        print("YOLO found things, but NONE were labelled 'sports ball' (class 32).")
        print("Look at the labels above near the ball's true position. The model is")
        print("probably calling the ball something else, so the class-32 filter drops it.")
    print("====================================================================")


if __name__ == "__main__":
    main()
