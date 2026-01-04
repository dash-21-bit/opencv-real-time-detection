from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Track:
    track_id: int
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    misses: int = 0


def iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b

    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh

    inter_x1 = max(ax, bx)
    inter_y1 = max(ay, by)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    a_area = aw * ah
    b_area = bw * bh

    union = a_area + b_area - inter_area
    if union == 0:
        return 0.0

    return inter_area / union


class IOUTracker:
    """
    SORT-like tracker using IoU matching (simple + beginner friendly).
    Maintains stable IDs for detections across frames.
    """

    def __init__(self, iou_threshold: float = 0.3, max_misses: int = 10) -> None:
        self.iou_threshold = iou_threshold
        self.max_misses = max_misses
        self.tracks: List[Track] = []
        self._next_id = 1

    def update(self, detections: List[Tuple[int, int, int, int]]) -> List[Track]:
        """
        Match detections to existing tracks by IoU.
        Create new tracks for unmatched detections.
        Remove tracks that have been missing too long.
        """
        matched_track_ids = set()

        # For each detection, find best matching track
        for det in detections:
            best_iou = 0.0
            best_idx = -1

            for idx, trk in enumerate(self.tracks):
                score = iou(trk.bbox, det)
                if score > best_iou:
                    best_iou = score
                    best_idx = idx

            # If match is good enough, update that track
            if best_idx != -1 and best_iou >= self.iou_threshold:
                self.tracks[best_idx].bbox = det
                self.tracks[best_idx].misses = 0
                matched_track_ids.add(self.tracks[best_idx].track_id)
            else:
                # Otherwise create a new track
                self.tracks.append(Track(track_id=self._next_id, bbox=det, misses=0))
                matched_track_ids.add(self._next_id)
                self._next_id += 1

        # Increase misses for tracks not matched this frame
        for trk in self.tracks:
            if trk.track_id not in matched_track_ids:
                trk.misses += 1

        # Remove dead tracks
        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]

        return self.tracks
