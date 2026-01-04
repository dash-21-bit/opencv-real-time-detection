from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import cv2
import numpy as np


@dataclass(frozen=True)
class Detection:
    x: int
    y: int
    w: int
    h: int


class FaceDetector:
    """
    Face detector using OpenCV Haar cascades.
    """

    def __init__(self, cascade_path: Path) -> None:
        if not cascade_path.exists():
            raise FileNotFoundError(f"Cascade not found: {cascade_path}")

        self.cascade = cv2.CascadeClassifier(str(cascade_path))
        if self.cascade.empty():
            raise ValueError("Failed to load cascade. File may be corrupted or invalid.")

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        """
        Detect faces in a BGR frame and return bounding boxes.
        """
        # Convert BGR -> grayscale (Haar needs grayscale)
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # Detect faces: returns list of rectangles (x,y,w,h)
        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(40, 40),
        )

        return [Detection(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]
