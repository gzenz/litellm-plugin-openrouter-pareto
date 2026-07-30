from __future__ import annotations

import threading
import time
from collections.abc import Callable

RATE_LIMIT_WINDOW_S = 300.0
RATE_LIMIT_THRESHOLD = 3


class RateLimitCooldown:
    def __init__(
        self,
        *,
        window_s: float = RATE_LIMIT_WINDOW_S,
        threshold: int = RATE_LIMIT_THRESHOLD,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._window_s = window_s
        self._threshold = threshold
        self._hits: dict[str, list[float]] = {}
        self._clock = clock if clock is not None else time.monotonic
        self._lock = threading.Lock()

    def record(self, slug: str, now: float | None = None) -> None:
        if not slug:
            return
        ts = now if now is not None else self._clock()
        cutoff = ts - self._window_s
        with self._lock:
            self._hits[slug] = [t for t in self._hits.get(slug, ()) if t >= cutoff] + [ts]

    def is_hot(self, slug: str, now: float | None = None) -> bool:
        if not slug:
            return False
        ts = now if now is not None else self._clock()
        cutoff = ts - self._window_s
        with self._lock:
            recent = [t for t in self._hits.get(slug, ()) if t >= cutoff]
            self._hits[slug] = recent
            return len(recent) >= self._threshold
