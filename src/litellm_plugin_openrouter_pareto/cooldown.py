from __future__ import annotations

import re
import threading
import time
from collections.abc import Callable

RATE_LIMIT_WINDOW_S = 300.0
RATE_LIMIT_THRESHOLD = 3
INPUT_CAP_TTL_S = 3600.0

_INPUT_CAP_RE = re.compile(
    r"exceeds the maximum (?:allowed )?(?:input )?length|maximum context length",
    re.IGNORECASE,
)


def is_input_cap_error(status: object, message: str) -> bool:
    if status != 400:
        return False
    return bool(_INPUT_CAP_RE.search(message))


class RateLimitCooldown:
    def __init__(
        self,
        *,
        window_s: float = RATE_LIMIT_WINDOW_S,
        threshold: int = RATE_LIMIT_THRESHOLD,
        input_cap_ttl_s: float = INPUT_CAP_TTL_S,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._window_s = window_s
        self._threshold = threshold
        self._input_cap_ttl_s = input_cap_ttl_s
        self._hits: dict[str, list[float]] = {}
        self._clock = clock if clock is not None else time.monotonic
        self._lock = threading.Lock()
        self._input_capped: dict[str, float] = {}

    def record(self, model: str, slug: str, now: float | None = None) -> None:
        if not slug:
            return
        key = f"{model}\x00{slug}"
        ts = now if now is not None else self._clock()
        cutoff = ts - self._window_s
        with self._lock:
            kept = [t for t in self._hits.get(key, ()) if t >= cutoff]
            kept.append(ts)
            self._hits[key] = kept
            self._hits = {k: v for k, v in self._hits.items() if v and v[-1] >= cutoff}

    def is_hot(self, model: str, slug: str, now: float | None = None) -> bool:
        """Pure predicate: reads the recorded hits without rewriting them. Eviction of
        aged entries happens in `record`, so calling this in a loop cannot mutate shared
        state out from under a concurrent caller."""
        if not slug:
            return False
        key = f"{model}\x00{slug}"
        ts = now if now is not None else self._clock()
        cutoff = ts - self._window_s
        with self._lock:
            recent = sum(1 for t in self._hits.get(key, ()) if t >= cutoff)
            return recent >= self._threshold

    def record_input_cap(self, model: str, slug: str, now: float | None = None) -> None:
        if slug:
            key = f"{model}\x00{slug}"
            ts = now if now is not None else self._clock()
            with self._lock:
                self._input_capped[key] = ts

    def is_input_capped(self, model: str, slug: str, now: float | None = None) -> bool:
        """A provider that raised an input-cap error is skipped only for a bounded TTL: a
        one-off oversized prompt does not make it permanently incapable, and an
        unexpired-forever set is an unbounded leak keyed by model x slug."""
        if not slug:
            return False
        key = f"{model}\x00{slug}"
        ts = now if now is not None else self._clock()
        cutoff = ts - self._input_cap_ttl_s
        with self._lock:
            recorded = self._input_capped.get(key)
            if recorded is None:
                return False
            if recorded < cutoff:
                del self._input_capped[key]
                return False
            return True

    def is_skipped(self, model: str, slug: str, now: float | None = None) -> bool:
        return self.is_hot(model, slug, now) or self.is_input_capped(model, slug, now)
