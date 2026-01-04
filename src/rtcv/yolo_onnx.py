from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO


def load_class_names(path: str) -> list[str]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing class names file: {p}")
    return [line.strip() for line in p.read_text().splitlines() if line.strip()]


class YOLOOnnxDetector:
    """
    Runs a YOLO ONNX model using Ultralytics and returns Supervision Detections.
    """

    def __init__(self, model_path: str, class_names_path: str) -> None:
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Missing ONNX model: {model_path}")

        self.model = YOLO(model_path, task="detect")  # Ultralytics can load .onnx models
        self.class_names = load_class_names(class_names_path)

    def infer(self, frame_bgr: np.ndarray, conf: float, iou: float) -> tuple[sv.Detections, dict[int, str]]:
        # Ultralytics expects BGR ok (OpenCV image). It handles preprocessing.
        results = self.model.predict(frame_bgr, conf=conf, iou=iou, verbose=False)[0]

        detections = sv.Detections.from_ultralytics(results)

        # Build id->name mapping
        id_to_name = {i: name for i, name in enumerate(self.class_names)}
        return detections, id_to_name
