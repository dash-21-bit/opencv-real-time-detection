from __future__ import annotations

from pathlib import Path
import cv2
import numpy as np
from rtcv.detector import Detection


def draw_detections(frame_bgr: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """
    Draw bounding boxes on a frame and return the modified frame.
    """
    out = frame_bgr.copy()

    for det in detections:
        x, y, w, h = det.x, det.y, det.w, det.h
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return out


def draw_fps(frame_bgr: np.ndarray, fps: float) -> np.ndarray:
    """
    Draw FPS text on the frame.
    """
    out = frame_bgr.copy()
    text = f"FPS: {fps:.1f}"
    cv2.putText(out, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return out


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
