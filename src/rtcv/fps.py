from __future__ import annotations

import time


class FPSTracker:
    """
    Tracks FPS using a rolling window approach.
    """

    def __init__(self, update_every: int = 10) -> None:
        # update_every: how often we recompute fps
        self.update_every = update_every
        self._count = 0
        self._start = time.time()
        self.fps = 0.0  # last computed fps

    def tick(self) -> float:
        """
        Call once per frame. Returns latest fps estimate.
        """
        self._count += 1

        # only recompute every N frames to reduce noise
        if self._count % self.update_every == 0:
            now = time.time()
            elapsed = now - self._start

            # avoid division by zero if elapsed is extremely small
            if elapsed > 0:
                self.fps = self.update_every / elapsed

            # reset timer window
            self._start = now

        return self.fps
