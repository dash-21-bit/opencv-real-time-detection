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
    conf: float
    label: str


class DNNFaceDetector:
    """
    DNN-based face detector using OpenCV's Caffe ResNet-10 SSD model.
    """

    def __init__(self, prototxt: Path, model: Path, conf_threshold: float = 0.5) -> None:
        # Store confidence threshold for filtering weak detections
        self.conf_threshold = conf_threshold

        # Check file existence early to avoid confusing runtime errors
        if not prototxt.exists():
            raise FileNotFoundError(f"Missing prototxt: {prototxt}")
        if not model.exists():
            raise FileNotFoundError(f"Missing caffemodel: {model}")

        # Load DNN model (Caffe)
        self.net = cv2.dnn.readNetFromCaffe(str(prototxt), str(model))
        if self.net.empty():
            raise ValueError("Failed to load DNN model.")

    def detect(self, frame_bgr: np.ndarray) -> list[Detection]:
        """
        Returns face detections as bounding boxes in pixel coordinates.
        """
        (h, w) = frame_bgr.shape[:2]

        # Convert frame into a blob (300x300) with mean subtraction (BGR)
        blob = cv2.dnn.blobFromImage(
            frame_bgr,
            scalefactor=1.0,
            size=(300, 300),
            mean=(104.0, 177.0, 123.0),
            swapRB=False,
            crop=False,
        )

        # Set blob as network input
        self.net.setInput(blob)

        # Forward pass to get detections
        detections = self.net.forward()

        results: list[Detection] = []

        # detections shape: [1, 1, N, 7]
        # Each row: [batch_id, class_id, confidence, x1, y1, x2, y2]
        for i in range(detections.shape[2]):
            conf = float(detections[0, 0, i, 2])

            # Filter low confidence predictions
            if conf < self.conf_threshold:
                continue

            x1 = int(detections[0, 0, i, 3] * w)
            y1 = int(detections[0, 0, i, 4] * h)
            x2 = int(detections[0, 0, i, 5] * w)
            y2 = int(detections[0, 0, i, 6] * h)

            # Clamp to image boundaries
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w - 1))
            y2 = max(0, min(y2, h - 1))

            results.append(
                Detection(
                    x=x1,
                    y=y1,
                    w=max(0, x2 - x1),
                    h=max(0, y2 - y1),
                    conf=conf,
                    label="face",
                )
            )

        return results
