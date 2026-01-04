from __future__ import annotations

from pathlib import Path
import cv2

from rtcv.fps import FPSTracker
from rtcv.utils import ensure_parent
from rtcv.tracker import IOUTracker
from rtcv.eye_detector import EyeDetector
from rtcv.dnn_detector import DNNFaceDetector


def draw_box(frame, x, y, w, h, text=""):
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    if text:
        cv2.putText(frame, text, (x, max(20, y - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


def draw_fps(frame, fps: float):
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)


def run(
    source: str,
    conf: float,
    iou_threshold: float,
    max_misses: int,
    eyes_enabled: bool,
    save_video_path: str,
    out_dir: str,
    no_display: bool,
    screenshot_key: str,
    quit_key: str,
) -> None:
    # Paths for models
    prototxt = Path("data/models/dnn/deploy.prototxt")
    model = Path("data/models/dnn/res10_300x300_ssd_iter_140000_fp16.caffemodel")
    eye_cascade = Path("data/models/haar/haarcascade_eye.xml")

    # Detector + optional eye detector
    face_detector = DNNFaceDetector(prototxt, model, conf_threshold=conf)
    eye_detector = EyeDetector(eye_cascade) if eyes_enabled else None

    # Tracker and FPS
    tracker = IOUTracker(iou_threshold=iou_threshold, max_misses=max_misses)
    fps_tracker = FPSTracker(update_every=10)

    # Source parsing: if digits -> webcam index
    src = int(source) if source.isdigit() else source

    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source}")

    # Video writer if requested
    writer = None
    if save_video_path:
        out_path = Path(save_video_path)
        ensure_parent(out_path)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fps = fps if fps and fps > 0 else 30.0

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

        print(f"[INFO] Saving video to: {out_path}")

    screenshot_count = 0
    print(f"[INFO] Source: {source}")
    print(f"[INFO] Detector: dnn | conf={conf}")
    print(f"[INFO] Tracker: iou={iou_threshold} max_misses={max_misses}")
    print(f"[INFO] Eyes: {eyes_enabled}")
    print(f"[INFO] Display window: {not no_display}")
    print(f"[INFO] Press '{screenshot_key}' to save screenshot, '{quit_key}' to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[INFO] End of stream or cannot read frame.")
            break

        # 1) Detect faces (DNN)
        dets = face_detector.detect(frame)
        face_boxes = [(d.x, d.y, d.w, d.h) for d in dets]

        # 2) Track faces (stable IDs)
        tracks = tracker.update(face_boxes)

        # 3) Draw tracked faces
        for trk in tracks:
            x, y, w, h = trk.bbox
            draw_box(frame, x, y, w, h, text=f"face #{trk.track_id}")

            # 4) Optional eyes detection inside face ROI
            if eye_detector is not None:
                eyes = eye_detector.detect_eyes(frame, (x, y, w, h))
                for (ex, ey, ew, eh) in eyes:
                    draw_box(frame, ex, ey, ew, eh, text="eye")

        # 5) FPS overlay
        fps_val = fps_tracker.tick()
        draw_fps(frame, fps_val)

        # 6) Save video frame
        if writer is not None:
            writer.write(frame)

        # 7) Display (optional)
        if not no_display:
            cv2.imshow("DNN Detection + Tracking", frame)

        # 8) Keyboard
        key = cv2.waitKey(1) & 0xFF if not no_display else 255

        if key == ord(quit_key):
            print("[INFO] Quit key pressed.")
            break

        if key == ord(screenshot_key):
            screenshot_count += 1
            screenshot_path = Path(out_dir) / "images" / f"screenshot_{screenshot_count}.jpg"
            ensure_parent(screenshot_path)
            cv2.imwrite(str(screenshot_path), frame)
            print(f"[OUTPUT] Saved screenshot: {screenshot_path}")

    cap.release()
    if writer is not None:
        writer.release()
    if not no_display:
        cv2.destroyAllWindows()

    print("[DONE] Finished.")
