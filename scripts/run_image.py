from __future__ import annotations

from pathlib import Path
import cv2

from rtcv.detector import FaceDetector
from rtcv.utils import draw_detections, ensure_parent

def main() -> None:
    cascade = Path("data/models/haarcascade_frontalface_default.xml")
    img_path = Path("data/samples/lena.jpg")

    if not img_path.exists():
        raise SystemExit(f"Sample image missing: {img_path}")

    detector = FaceDetector(cascade)

    img = cv2.imread(str(img_path))
    detections = detector.detect(img)
    out = draw_detections(img, detections)

    out_path = Path("outputs/images/lena_detected.jpg")
    ensure_parent(out_path)
    cv2.imwrite(str(out_path), out)

    print(f"Detections found: {len(detections)}")
    print(f"Saved output image: {out_path}")

if __name__ == "__main__":
    main()
