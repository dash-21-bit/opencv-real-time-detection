from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Real-time CV detection (DNN/Haar) with tracking + FPS.")

    p.add_argument("--source", default="0", help="Webcam index (e.g. 0) or path to video file.")
    p.add_argument("--detector", choices=["dnn", "haar"], default="dnn", help="Detection backend.")
    p.add_argument("--conf", type=float, default=0.5, help="DNN confidence threshold (for dnn).")
    p.add_argument("--iou", type=float, default=0.3, help="IoU threshold for tracker.")
    p.add_argument("--max-misses", type=int, default=10, help="Max missed frames before track is removed.")
    p.add_argument("--eyes", action="store_true", help="Enable eye detection inside face boxes (Haar).")
    p.add_argument("--save-video", default="", help="Path to save MP4 output. Example: outputs/videos/out.mp4")
    p.add_argument("--out-dir", default="outputs", help="Base output directory for screenshots.")
    p.add_argument("--no-display", action="store_true", help="Run without showing window.")
    p.add_argument("--screenshot-key", default="s", help="Key to save screenshot (default: s).")
    p.add_argument("--quit-key", default="q", help="Key to quit (default: q).")

    return p


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()
