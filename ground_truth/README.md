# Ground Truth Generator

Generates the reference CSV that the Pi client uses to score inference accuracy.

## Requirements

```
ultralytics==8.2.0
opencv-python==4.10.0.84
numpy==1.26.4
onnxruntime  (any recent version)
```

## Usage

```bash
python generate_ground_truth.py \
    --video  /path/to/video.mp4 \
    --model  /path/to/yolov10n.onnx \
    --output ground_truth.csv \
    --conf   0.3
```

For the cup video, generate cup-only ground truth with YOLO:

```bash
python generate_ground_truth.py \
    --video  /path/to/test_video.mp4 \
    --model  /path/to/yolov10n.onnx \
    --output ground_truth.csv \
    --tracker yolo \
    --target-class-id 41,75 \
    --conf 0.1
```

COCO class `41` is `cup` and class `75` is `vase`. The current cup video is
mostly labelled as `vase` by YOLO, so the measured local and remote inference
paths use the same combined filter through `TARGET_CLASS_ID=41,75`.

For the original tennis-ball video, the color tracker is still available:

```bash
python generate_ground_truth.py \
    --video  /path/to/test_video.mp4 \
    --output ground_truth.csv \
    --tracker tennis-ball-color
```

## Output format

| Column | Description |
|--------|-------------|
| `frame_number` | 1-indexed frame counter |
| `center_x` | X pixel coordinate of the selected target centre |
| `center_y` | Y pixel coordinate |
| `confidence` | YOLO confidence score |
| `class_id` | COCO class index |
| `class_name` | Human-readable class name |

Frames with no detection above the threshold get `NaN` coordinates and
`class_id = -1`.

## Obtaining the ONNX model

```bash
pip install ultralytics
yolo export model=yolov10n.pt format=onnx
```
