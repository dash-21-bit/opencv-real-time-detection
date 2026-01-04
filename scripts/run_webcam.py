from __future__ import annotations

from pathlib import Path
from rtcv.app import run_realtime

def main() -> None:
    cascade = Path("data/models/haarcascade_frontalface_default.xml")

    # source=0 means default webcam
    run_realtime(
        source=0,
        cascade_path=cascade,
        save_video_path=Path("outputs/videos/webcam_output.mp4"),
    )

if __name__ == "__main__":
    main()

