from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np


class EyeDetector:
    """
    Detect eyes using Haar cascade within a face ROI.
    """

    def __init__(self, cascade_path: Path) -> None:
        if not cascade_path.exists():
            raise FileNotFoundError(f"Missing eye cascade: {cascade_path}")

        self.cascade = cv2.CascadeClassifier(str(cascade_path))
        if self.cascade.empty():
            raise ValueError("Failed to load eye cascade.")

    def detect_eyes(self, frame_bgr: np.ndarray, face_xywh: tuple[int, int, int, int]) -> list[tuple[int, int, int, int]]:
        x, y, w, h = face_xywh

        # Crop face region
        roi = frame_bgr[y:y + h, x:x + w]
        if roi.size == 0:
            return []

        # Haar works better on grayscale
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        eyes = self.cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(20, 20),
        )

        # Convert ROI coords -> full-frame coords
        return [(x + int(ex), y + int(ey), int(ew), int(eh)) for (ex, ey, ew, eh) in eyes]
