# Ground Truth Generator

Generates the reference CSV that the Pi client uses to score inference accuracy.
Uses HSV colour segmentation to locate the red car directly — no YOLO model
required and no background model needed, so it works with moving cameras (drones).

Keeping ground truth independent of YOLO means YOLO misses remain measurable.

## Requirements

```
opencv-python==4.10.0.84
numpy==1.26.4
```

## Usage

```bash
python generate_ground_truth.py \
    --video  data/test_video.mp4 \
    --output data/ground_truth.csv
```

The script:
1. Converts each frame to HSV and masks both red hue bands (hue 0–10 and 160–179).
2. Cleans the mask with morphological open and close operations.
3. Selects the largest red blob within car-sized area bounds.
4. Writes its bounding-box centre to the CSV.
5. Linearly interpolates any remaining NaN rows between known detections.

Result: 100% frame coverage, no warmup period, robust to camera motion.

## Output format

| Column | Description |
|--------|-------------|
| `frame_number` | 1-indexed frame counter |
| `center_x` | X pixel coordinate of the car centre |
| `center_y` | Y pixel coordinate |
| `confidence` | Always `1.0` (colour detection does not produce a confidence score) |
| `class_id` | Always `2` (COCO car class) |
| `class_name` | Always `car` |

## Obtaining the ONNX model (for inference only, not ground truth)

```bash
pip install ultralytics
yolo export model=yolov10n.pt format=onnx
```
