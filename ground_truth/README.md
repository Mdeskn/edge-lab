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

For the drone-view video, generate ground truth filtered to person and car:

```bash
python generate_ground_truth.py \
    --video  /path/to/test_video.mp4 \
    --model  /path/to/yolov10n.onnx \
    --output ground_truth.csv \
    --tracker yolo \
    --target-class-id 0,2 \
    --conf 0.3
```

COCO class `0` is `person` and class `2` is `car`. These are the primary targets
in the drone-view video. Note: trees are not a COCO class and cannot be detected
with the standard YOLOv10n model.

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
