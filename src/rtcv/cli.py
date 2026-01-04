from __future__ import annotations
import argparse

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="YOLO(ONNX) + ByteTrack real-time tracking with optional eye detection."
    )

    p.add_argument("--source", default="0",
                   help="Webcam index (0) or path to video file (e.g. data/samples/vtest.avi).")
    p.add_argument("--model", default="data/models/yolo/yolov8n.onnx",
                   help="Path to YOLO ONNX model.")
    p.add_argument("--classes", default="person",
                   help="Comma-separated class names to keep (e.g. person,car,bus). Use 'all' for all classes.")
    p.add_argument("--conf", type=float, default=0.35, help="YOLO confidence threshold.")
    p.add_argument("--iou", type=float, default=0.5, help="YOLO IoU threshold for NMS.")
    p.add_argument("--save-video", default="outputs/videos/output.mp4",
                   help="Where to save the annotated output video.")
    p.add_argument("--out-dir", default="outputs", help="Base directory for screenshots.")
    p.add_argument("--no-display", action="store_true", help="Run without opening a window.")
    p.add_argument("--eyes", action="store_true", help="Detect eyes inside 'person' boxes (Haar).")
    p.add_argument("--track-thresh", type=float, default=0.25, help="ByteTrack track activation threshold.")
    p.add_argument("--track-buffer", type=int, default=30, help="ByteTrack buffer frames for lost tracks.")
    p.add_argument("--match-thresh", type=float, default=0.8, help="ByteTrack matching threshold.")
    p.add_argument("--screenshot-key", default="s", help="Key to save screenshot.")
    p.add_argument("--quit-key", default="q", help="Key to quit.")

    return p.parse_args()
