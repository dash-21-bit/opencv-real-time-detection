from __future__ import annotations

from pathlib import Path
import time
import cv2
import numpy as np
import supervision as sv

from rtcv.yolo_onnx import YOLOOnnxDetector
from rtcv.eyes import EyeDetector


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def parse_source(source: str):
    return int(source) if source.isdigit() else source


def filter_by_class_names(detections: sv.Detections, id_to_name: dict[int, str], wanted: list[str]) -> sv.Detections:
    if wanted == ["all"]:
        return detections

    keep = []
    for i, cid in enumerate(detections.class_id):
        name = id_to_name.get(int(cid), "")
        keep.append(name in wanted)

    keep = np.array(keep, dtype=bool)
    return detections[keep]


class FPSTracker:
    def __init__(self, every_n: int = 10) -> None:
        self.every_n = every_n
        self.count = 0
        self.t0 = time.time()
        self.fps = 0.0

    def tick(self) -> float:
        self.count += 1
        if self.count % self.every_n == 0:
            t1 = time.time()
            dt = t1 - self.t0
            if dt > 0:
                self.fps = self.every_n / dt
            self.t0 = t1
        return self.fps


def run(
    source: str,
    model_path: str,
    class_names_path: str,
    classes: str,
    conf: float,
    iou: float,
    save_video: str,
    out_dir: str,
    no_display: bool,
    eyes: bool,
    track_thresh: float,
    track_buffer: int,
    match_thresh: float,
    screenshot_key: str,
    quit_key: str,
) -> None:
    detector = YOLOOnnxDetector(model_path=model_path, class_names_path=class_names_path)

    # ByteTrack from Supervision
    tracker = sv.ByteTrack()

    # Annotators (nice looking output)
    box_annotator = sv.BoxAnnotator()
    label_annotator = sv.LabelAnnotator()
    trace_annotator = sv.TraceAnnotator()

    eye_detector = EyeDetector("data/models/haar/haarcascade_eye.xml") if eyes else None

    wanted = [c.strip() for c in classes.split(",")] if classes.strip() else ["person"]
    wanted = ["all"] if wanted == ["all"] else wanted

    cap = cv2.VideoCapture(parse_source(source))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {source}")

    # Video writer
    save_path = Path(save_video)
    ensure_parent(save_path)

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_in = cap.get(cv2.CAP_PROP_FPS)
    fps_in = fps_in if fps_in and fps_in > 0 else 30.0

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(save_path), fourcc, fps_in, (width, height))

    fps_tracker = FPSTracker(every_n=10)

    screenshot_count = 0

    print(f"[INFO] source={source}")
    print(f"[INFO] model={model_path}")
    print(f"[INFO] classes={wanted}")
    print(f"[INFO] conf={conf} iou={iou}")
    print(f"[INFO] ByteTrack track_thresh={track_thresh} track_buffer={track_buffer} match_thresh={match_thresh}")
    print(f"[INFO] save_video={save_path}")
    print(f"[INFO] eyes={eyes}")
    print(f"[INFO] keys: '{screenshot_key}' screenshot | '{quit_key}' quit")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("[INFO] End of stream.")
            break

        # YOLO inference -> detections
        detections, id_to_name = detector.infer(frame, conf=conf, iou=iou)

        # Filter classes
        detections = filter_by_class_names(detections, id_to_name, wanted)

        # ByteTrack assigns tracker_id
        detections = tracker.update_with_detections(detections)

        # Labels: "id class conf"
        labels = []
        for i in range(len(detections)):
            cid = int(detections.class_id[i]) if detections.class_id is not None else -1
            name = id_to_name.get(cid, "obj")
            confv = float(detections.confidence[i]) if detections.confidence is not None else 0.0
            tid = int(detections.tracker_id[i]) if detections.tracker_id is not None else -1
            labels.append(f"#{tid} {name} {confv:.2f}")

        annotated = frame.copy()
        annotated = trace_annotator.annotate(annotated, detections=detections)
        annotated = box_annotator.annotate(annotated, detections=detections)
        annotated = label_annotator.annotate(annotated, detections=detections, labels=labels)

        # Optional eyes inside PERSON boxes only
        if eye_detector is not None and detections.class_id is not None:
            for i in range(len(detections)):
                cid = int(detections.class_id[i])
                if id_to_name.get(cid, "") != "person":
                    continue
                x1, y1, x2, y2 = detections.xyxy[i].astype(int)
                eyes_xyxy = eye_detector.detect(annotated, (x1, y1, x2, y2))
                for (ex1, ey1, ex2, ey2) in eyes_xyxy:
                    cv2.rectangle(annotated, (ex1, ey1), (ex2, ey2), (255, 255, 255), 2)
                    cv2.putText(annotated, "eye", (ex1, max(20, ey1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # FPS overlay
        fps_val = fps_tracker.tick()
        cv2.putText(annotated, f"FPS: {fps_val:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        # Save video frame
        writer.write(annotated)

        # Show window
        if not no_display:
            cv2.imshow("YOLO(ONNX) + ByteTrack", annotated)
            key = cv2.waitKey(1) & 0xFF
        else:
            key = 255

        # Quit
        if key == ord(quit_key):
            print("[INFO] Quit pressed.")
            break

        # Screenshot
        if key == ord(screenshot_key):
            screenshot_count += 1
            sp = Path(out_dir) / "images" / f"screenshot_{screenshot_count}.jpg"
            ensure_parent(sp)
            cv2.imwrite(str(sp), annotated)
            print(f"[OUTPUT] Saved screenshot: {sp}")

    cap.release()
    writer.release()
    if not no_display:
        cv2.destroyAllWindows()

    print("[DONE] Output video:", save_path)
