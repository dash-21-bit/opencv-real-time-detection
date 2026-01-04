# Project 7 (Upgraded) — YOLO (ONNX) + ByteTrack Tracking + CLI + Eye Detection

**Author:** Adarsh Ravi  
**Tech:** Python, OpenCV, Ultralytics YOLO, Supervision (ByteTrack)  
**What it does:** Real-time multi-class object detection + tracking IDs + FPS overlay + saved outputs

---

## ✅ Overview

This project is a portfolio-grade real-time Computer Vision pipeline:

- **YOLO (ONNX)** inference for object detection
- **ByteTrack** for multi-object tracking (stable IDs across frames)
- **Multi-class** filtering (track only selected COCO classes, e.g., `person,car`)
- **Optional eye detection** using Haar cascades inside detected person boxes
- **CLI interface** (source, thresholds, outputs, headless mode)
- Saves **annotated MP4 video** and **screenshots**

This demonstrates practical ML/CV engineering: detection + tracking + clean tooling.

---

## 📌 Demo outputs (GitHub preview)

- `assets/example_screenshot.jpg`  
- `assets/example_output.mp4` (optional)

---

## 📂 Repository Structure

opencv-real-time-detection/
├── data/
│   ├── models/
│   │   ├── yolo/
│   │   │   ├── yolov8n.onnx
│   │   │   └── coco.names
│   │   └── haar/
│   │       └── haarcascade_eye.xml
│   └── samples/
│       ├── vtest.avi
│       └── lena.jpg
├── src/rtcv/
│   ├── cli.py
│   ├── pipeline.py
│   ├── yolo_onnx.py
│   └── eyes.py
├── scripts/
│   └── run.py
├── outputs/        # generated at runtime (ignored)
└── assets/         # example outputs for GitHub preview

code 

---
---

## ⚙️ Setup (terminal only)

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install opencv-python numpy ultralytics supervision
pip freeze > requirements.txt

```
## Export YOLO to ONNX
```
python - <<'EOF'
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
model.export(format="onnx", opset=12, simplify=True)
print("Export complete.")
EOF

cp runs/detect/export*/weights/yolov8n.onnx data/models/yolo/yolov8n.onnx
```

## Run(sample_video)

```bash
python scripts/run.py --source data/samples/vtest.avi --classes person,car --save-video outputs/videos/vtest_tracked.mp4
```
## Run  (Webcam)
```bash

python scripts/run.py --source 0 --classes person --save-video outputs/videos/webcam_tracked.mp4
```


## CLI Help
```
python scripts/run.py --help
```
Key options:
	•	--source webcam index or video path
	•	--classes filter classes or all
	•	--conf detection confidence threshold
	•	--save-video output mp4 path
	•	--no-display headless mode
	•	--eyes eye detection inside person boxes
	•	ByteTrack params: --track-thresh, --track-buffer, --match-thresh

## 9) Technical Notes

DNN Face Detector

Uses OpenCV’s DNN Caffe SSD face detector:
	•	config: deploy.prototxt
	•	weights: res10_300x300_ssd_iter_140000_fp16.caffemodel

Multi-class (Face + Eyes)

Faces are detected via DNN.
Eyes are detected via Haar cascade inside the face bounding box region.

Tracking (SORT-style)

A lightweight IoU tracker assigns stable IDs to faces across frames.
This is inspired by SORT matching logic and is intentionally simple for learning.




