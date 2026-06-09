"""Debug YOLO ONNX detections for a video inside the client container."""
import argparse
from collections import Counter

import cv2
import numpy as np
import onnxruntime as ort


COCO_NAMES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass",
    "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
    "donut", "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster",
    "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
]


def parse_class_filter(value: str | None) -> tuple[int, ...] | None:
    if not value:
        return None
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def preprocess(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    resized = cv2.resize(frame, (width, height))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normalized = rgb.astype(np.float32) / 255.0
    chw = np.transpose(normalized, (2, 0, 1))
    return np.expand_dims(chw, axis=0)


def best_box(
    output: np.ndarray,
    conf_threshold: float,
    class_filter: tuple[int, ...] | None = None,
) -> np.ndarray | None:
    boxes = output[0] if output.ndim == 3 and output.shape[0] == 1 else np.squeeze(output)
    if boxes.ndim != 2 or boxes.shape[1] < 6:
        return None

    mask = boxes[:, 4] >= conf_threshold
    if class_filter is not None:
        mask &= np.isin(boxes[:, 5].astype(int), class_filter)

    filtered = boxes[mask]
    if len(filtered) == 0:
        return None
    return filtered[filtered[:, 4].argmax()]


def class_name(class_id: int) -> str:
    if 0 <= class_id < len(COCO_NAMES):
        return COCO_NAMES[class_id]
    return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print raw and target-filtered YOLO detections for a video."
    )
    parser.add_argument("--video", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--conf", type=float, default=0.1)
    parser.add_argument("--target-class-id", default="")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--samples", type=int, default=12)
    args = parser.parse_args()

    class_filter = parse_class_filter(args.target_class_id)

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    session = ort.InferenceSession(
        args.model,
        sess_options=opts,
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {args.video}")

    raw_counts: Counter[int] = Counter()
    target_counts: Counter[int] = Counter()
    raw_misses = 0
    target_misses = 0
    frame_count = 0
    sample_lines = []

    while True:
        if args.max_frames and frame_count >= args.max_frames:
            break

        ok, frame = cap.read()
        if not ok:
            break

        frame_count += 1
        inp = preprocess(frame, args.width, args.height)
        output = session.run([output_name], {input_name: inp})[0]

        raw = best_box(output, args.conf)
        target = best_box(output, args.conf, class_filter)

        if raw is None:
            raw_misses += 1
        else:
            raw_counts[int(raw[5])] += 1

        if target is None:
            target_misses += 1
        else:
            target_counts[int(target[5])] += 1

        if len(sample_lines) < args.samples:
            raw_text = "none"
            target_text = "none"
            if raw is not None:
                raw_cls = int(raw[5])
                raw_text = f"{raw_cls}:{class_name(raw_cls)} conf={raw[4]:.3f}"
            if target is not None:
                target_cls = int(target[5])
                target_text = f"{target_cls}:{class_name(target_cls)} conf={target[4]:.3f}"
            sample_lines.append(
                f"frame={frame_count:04d} raw={raw_text} target={target_text}"
            )

    cap.release()

    print(f"model={args.model}")
    print(f"video={args.video}")
    print(f"input={input_name} output={output_name}")
    print(f"frames={frame_count} conf={args.conf} target_filter={class_filter or 'any'}")
    print("\nSamples:")
    for line in sample_lines:
        print(f"  {line}")

    print("\nRaw best-detection classes:")
    for class_id, count in raw_counts.most_common():
        print(f"  {count:5d}  {class_id:2d}  {class_name(class_id)}")
    print(f"  {raw_misses:5d}  none")

    print("\nTarget-filtered classes:")
    for class_id, count in target_counts.most_common():
        print(f"  {count:5d}  {class_id:2d}  {class_name(class_id)}")
    print(f"  {target_misses:5d}  none")


if __name__ == "__main__":
    main()
