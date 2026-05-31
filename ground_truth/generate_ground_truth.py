"""
Generate ground truth coordinates from a video using YOLOv10 ONNX.

Usage:
    python generate_ground_truth.py \
        --video test_video.mp4 \
        --model yolov10n.onnx \
        --output ground_truth.csv \
        --conf 0.3

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


def postprocess(
        output,
        orig_h,
        orig_w,
        conf_threshold
):

    boxes = output.squeeze()

    # confidence filtering
    boxes = boxes[boxes[:,4] >= conf_threshold]

    if len(boxes) == 0:
        return None,None,0.0,-1

    # ---------- Prefer sports balls ----------
    sports_ball_boxes = boxes[
        boxes[:,5].astype(int) == SPORTS_BALL_CLASS
    ]

    if len(sports_ball_boxes) > 0:

        best = sports_ball_boxes[
            sports_ball_boxes[:,4].argmax()
        ]

    else:
        # fallback
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


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--video",
        required=True
    )

    parser.add_argument(
        "--model",
        required=True
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

    args = parser.parse_args()

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

            h,w = frame.shape[:2]

            inp = preprocess(frame)

            outputs = session.run(
                [output_name],
                {input_name: inp}
            )

            cx,cy,conf,cls_id = postprocess(
                outputs[0],
                h,
                w,
                args.conf
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