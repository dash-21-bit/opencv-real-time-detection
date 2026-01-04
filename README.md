# OpenCV Real-Time Detection — Bounding Boxes + FPS + Saved Outputs

**Repo:** opencv-real-time-detection  
**Author:** Adarsh Ravi  
**Tech:** Python, OpenCV, NumPy  
**Outputs:** Real-time detection window, screenshots (JPG), recorded video (MP4)

---

## 1. Overview

This project demonstrates a practical Computer Vision workflow using OpenCV:

- Real-time detection (webcam or video)
- Bounding box drawing
- FPS measurement (performance awareness)
- Saving output screenshots and recorded detection video

It is designed to be portfolio-ready and reproducible, including:
- a pre-trained Haar cascade model file (`.xml`)
- a sample test image (`lena.jpg`) inside the repository

---

## 2. Features

- **Face detection** using Haar Cascade classifier
- **Bounding boxes** drawn around detected faces
- **FPS overlay** for real-time performance monitoring
- **Screenshot saving** by pressing `s`
- **Video output recording** to MP4
- **Image-only test mode** to verify functionality without a webcam

---

## 3. Repository Structure

opencv-real-time-detection/
├── src/rtcv/
│   ├── app.py        # realtime pipeline
│   ├── detector.py   # face detector
│   ├── fps.py        # fps tracker
│   └── utils.py      # drawing helpers
├── scripts/
│   ├── run_image.py  # runs detection on sample image
│   └── run_webcam.py # realtime webcam detection + save video
├── data/
│   ├── models/
│   │   └── haarcascade_frontalface_default.xml
│   └── samples/
│       └── lena.jpg
└── outputs/
├── images/
└── videos/

code 

---

## 4. Setup (Terminal Only)

### Create and activate venv
```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```
## Install deps

```bash

pip install -r requirements.txt
pip install -e .
```

## Run

```bash

python scripts/run_image.py
```
## Run  (real-time)
```bash

python scripts/run_webcam.py
```

## Controls:
	•	Press s to save screenshot → outputs/images/screenshot_*.jpg
	•	Press q to quit

## Output files:
	•	outputs/videos/webcam_output.mp4
	•	outputs/images/screenshot_*.jpg


## 7. Technical Notes

## 7.1 Why grayscale?

Haar cascades operate on intensity patterns, so frames are converted from BGR to grayscale before detection.

7.2 FPS tracking

FPS is computed using a rolling window to reduce noise and avoid recalculating every single frame.

7.3 Saving outputs

Screenshots and video output prove that the program is generating results, which is useful for portfolio review and demonstrations.
