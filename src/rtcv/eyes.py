from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np


class EyeDetector:
    def __init__(self, cascade_path: str) -> None:
        p = Path(cascade_path)
        if not p.exists():
            raise FileNotFoundError(f"Missing eye cascade: {p}")
        self.cascade = cv2.CascadeClassifier(str(p))
        if self.cascade.empty():
            raise ValueError("Failed to load eye cascade.")

    def detect(self, frame_bgr: np.ndarray, box_xyxy: tuple[int, int, int, int]) -> list[tuple[int, int, int, int]]:
        x1, y1, x2, y2 = box_xyxy
        roi = frame_bgr[y1:y2, x1:x2]
        if roi.size == 0:
            return []

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        eyes = self.cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(20, 20))

        # Convert ROI coords -> full frame coords
        out = []
        for (ex, ey, ew, eh) in eyes:
            out.append((x1 + int(ex), y1 + int(ey), x1 + int(ex + ew), y1 + int(ey + eh)))
        return out
