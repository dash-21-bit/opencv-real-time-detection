from __future__ import annotations

from pathlib import Path
import cv2

from rtcv.detector import FaceDetector
from rtcv.fps import FPSTracker
from rtcv.utils import draw_detections, draw_fps, ensure_parent


def run_realtime(
    source: int | str,
    cascade_path: Path,
    window_name: str = "Real-time Detection",
    save_video_path: Path | None = None,
) -> None:
    """
    Run real-time detection from webcam (source=int) or video file (source=str).
    Press:
      - 'q' to quit
      - 's' to save a screenshot to outputs/images/
    """
    detector = FaceDetector(cascade_path)
    fps_tracker = FPSTracker(update_every=10)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    writer = None
    if save_video_path is not None:
        ensure_parent(save_video_path)

        # Grab width/height/fps from capture
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if fps and fps > 0 else 30.0  # fallback

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(save_video_path), fourcc, fps, (width, height))

    screenshot_count = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        detections = detector.detect(frame)
        out = draw_detections(frame, detections)

        fps_val = fps_tracker.tick()
        out = draw_fps(out, fps_val)

        if writer is not None:
            writer.write(out)

        cv2.imshow(window_name, out)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == ord("s"):
            screenshot_count += 1
            screenshot_path = Path(f"outputs/images/screenshot_{screenshot_count}.jpg")
            ensure_parent(screenshot_path)
            cv2.imwrite(str(screenshot_path), out)
            print(f"Saved screenshot: {screenshot_path}")

    cap.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
