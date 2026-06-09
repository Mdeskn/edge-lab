# Ground Truth Generator

Generates the reference CSV that the Pi client uses to score inference accuracy.
Uses OpenCV MOG2 background subtraction — no YOLO model required.
Keeping ground truth independent of YOLO means YOLO misses remain measurable.

## Requirements

```
opencv-python==4.10.0.84
numpy==1.26.4
```

## Usage

```bash
python generate_ground_truth.py \
    --video  /path/to/test_video.mp4 \
    --output ground_truth.csv
```

The script runs a two-pass approach:
1. Pre-pass (first 30 frames): builds the MOG2 background model
2. Tracking pass (all frames, frozen model): detects the car as the largest moving blob
3. Linear interpolation fills any remaining gaps

Result: 100% frame coverage with no YOLO dependency.

## Output format

| Column | Description |
|--------|-------------|
| `frame_number` | 1-indexed frame counter |
| `center_x` | X pixel coordinate of the car centre |
| `center_y` | Y pixel coordinate |
| `confidence` | Always `1.0` (MOG2 does not produce confidence scores) |
| `class_id` | Always `2` (COCO car class) |
| `class_name` | Always `car` |

## Obtaining the ONNX model (for inference only, not ground truth)

```bash
pip install ultralytics
yolo export model=yolov10n.pt format=onnx
```
