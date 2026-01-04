from __future__ import annotations

from rtcv.cli import parse_args
from rtcv.app import run


def main() -> None:
    args = parse_args()

    run(
        source=args.source,
        conf=args.conf,
        iou_threshold=args.iou,
        max_misses=args.max_misses,
        eyes_enabled=args.eyes,
        save_video_path=args.save_video,
        out_dir=args.out_dir,
        no_display=args.no_display,
        screenshot_key=args.screenshot_key,
        quit_key=args.quit_key,
    )


if __name__ == "__main__":
    main()
