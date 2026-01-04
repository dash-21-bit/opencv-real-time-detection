from __future__ import annotations
from rtcv.cli import parse_args
from rtcv.pipeline import run

def main() -> None:
    args = parse_args()
    run(
        source=args.source,
        model_path=args.model,
        class_names_path="data/models/yolo/coco.names",
        classes=args.classes,
        conf=args.conf,
        iou=args.iou,
        save_video=args.save_video,
        out_dir=args.out_dir,
        no_display=args.no_display,
        eyes=args.eyes,
        track_thresh=args.track_thresh,
        track_buffer=args.track_buffer,
        match_thresh=args.match_thresh,
        screenshot_key=args.screenshot_key,
        quit_key=args.quit_key,
    )

if __name__ == "__main__":
    main()

