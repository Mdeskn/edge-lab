"""
Generate ground truth coordinates from a video.

Usage:
    python generate_ground_truth.py \
        --video test_video.mp4 \
        --output ground_truth.csv \
        --tracker tennis-ball-color

Optional YOLO mode:
    python generate_ground_truth.py \
        --video test_video.mp4 \
        --model yolov10n.onnx \
        --output ground_truth.csv \
        --tracker yolo \
        --target-class-id 32

Output:
frame_number,center_x,center_y,confidence,class_id,class_name
"""

import argparse
import csv
import os
import sys

import cv2
import numpy as np
import onnxruntime as ort


COCO_NAMES = [
    "person","bicycle","car","motorcycle","airplane","bus","train",
    "truck","boat","traffic light","fire hydrant","stop sign",
    "parking meter","bench","bird","cat","dog","horse","sheep","cow",
    "elephant","bear","zebra","giraffe","backpack","umbrella",
    "handbag","tie","suitcase","frisbee","skis","snowboard",
    "sports ball","kite","baseball bat","baseball glove",
    "skateboard","surfboard","tennis racket","bottle","wine glass",
    "cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza",
    "donut","cake","chair","couch","potted plant","bed",
    "dining table","toilet","tv","laptop","mouse","remote",
    "keyboard","cell phone","microwave","oven","toaster",
    "sink","refrigerator","book","clock","vase","scissors",
    "teddy bear","hair drier","toothbrush"
]

SPORTS_BALL_CLASS = 32


def load_session(model_path):
    if not os.path.exists(model_path):
        print(f"ERROR: model not found: {model_path}")
        sys.exit(1)

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4

    return ort.InferenceSession(
        model_path,
        sess_options=opts,
        providers=["CPUExecutionProvider"]
    )


def preprocess(frame, size=640):

    resized = cv2.resize(frame, (size, size))

    rgb = cv2.cvtColor(
        resized,
        cv2.COLOR_BGR2RGB
    ).astype(np.float32)

    rgb = rgb / 255.0

    chw = np.transpose(rgb, (2,0,1))

    return np.expand_dims(chw, axis=0)


def postprocess_yolo(
        output,
        orig_h,
        orig_w,
        conf_threshold,
        target_class_id
):

    boxes = output.squeeze()

    # confidence filtering
    boxes = boxes[boxes[:,4] >= conf_threshold]

    if len(boxes) == 0:
        return None,None,0.0,-1

    if target_class_id is not None:
        boxes = boxes[boxes[:,5].astype(int) == target_class_id]
        if len(boxes) == 0:
            return None,None,0.0,-1

    best = boxes[
        boxes[:,4].argmax()
    ]

    x1,y1,x2,y2,conf,cls_id = best

    scale_x = orig_w / 640
    scale_y = orig_h / 640

    cx = ((x1+x2)/2) * scale_x
    cy = ((y1+y2)/2) * scale_y

    return (
        float(cx),
        float(cy),
        float(conf),
        int(cls_id)
    )


def track_tennis_ball(frame):
    """
    Return the centre of the largest plausible yellow-green tennis-ball region.

    This mode is intended for producing independent ground truth for the lab's
    tennis-ball video. It does not run in the measured inference pipeline.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        np.array([25, 80, 80], dtype=np.uint8),
        np.array([50, 255, 255], dtype=np.uint8),
    )
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        np.ones((5, 5), dtype=np.uint8),
    )
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        np.ones((11, 11), dtype=np.uint8),
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 500:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        aspect_ratio = width / height if height else 0.0
        if 0.65 <= aspect_ratio <= 1.45:
            candidates.append((area, x, y, width, height))

    if not candidates:
        return None,None,0.0,-1

    _, x, y, width, height = max(candidates)
    return (
        x + width / 2.0,
        y + height / 2.0,
        1.0,
        SPORTS_BALL_CLASS,
    )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--video",
        required=True
    )

    parser.add_argument(
        "--model",
        help="Path to the YOLO ONNX model. Required for --tracker yolo."
    )

    parser.add_argument(
        "--output",
        default="ground_truth.csv"
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.3
    )

    parser.add_argument(
        "--tracker",
        choices=["yolo", "tennis-ball-color"],
        default="yolo",
        help="Ground-truth source. Use tennis-ball-color for the lab tennis-ball video."
    )

    parser.add_argument(
        "--target-class-id",
        type=int,
        default=None,
        help="Optional COCO class filter for YOLO mode. Sports ball is class 32."
    )

    args = parser.parse_args()

    session = None
    input_name = None
    output_name = None
    if args.tracker == "yolo":
        if not args.model:
            parser.error("--model is required for --tracker yolo")
        session = load_session(args.model)
        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name

    cap = cv2.VideoCapture(args.video)

    total_frames = int(
        cap.get(cv2.CAP_PROP_FRAME_COUNT)
    )

    print(
        f"Video: {args.video} ({total_frames} frames)"
    )

    frame_num = 0
    detected = 0

    with open(
        args.output,
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "frame_number",
            "center_x",
            "center_y",
            "confidence",
            "class_id",
            "class_name"
        ])

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            frame_num += 1

            if args.tracker == "tennis-ball-color":
                cx,cy,conf,cls_id = track_tennis_ball(frame)
            else:
                h,w = frame.shape[:2]
                inp = preprocess(frame)
                outputs = session.run(
                    [output_name],
                    {input_name: inp}
                )
                cx,cy,conf,cls_id = postprocess_yolo(
                    outputs[0],
                    h,
                    w,
                    args.conf,
                    args.target_class_id
                )

            if cx is None:

                writer.writerow([
                    frame_num,
                    "NaN",
                    "NaN",
                    0.0,
                    -1,
                    "none"
                ])

            else:

                detected += 1

                writer.writerow([
                    frame_num,
                    round(cx,2),
                    round(cy,2),
                    round(conf,4),
                    cls_id,
                    COCO_NAMES[cls_id]
                ])

            if frame_num % 100 == 0:
                print(
                    f"Progress: {frame_num}/{total_frames}"
                )

    cap.release()

    print("\nDone")
    print(
        f"Detection rate: {100*detected/frame_num:.1f}%"
    )
    print(
        f"Output: {args.output}"
    )


if __name__ == "__main__":
    main()
