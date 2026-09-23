from __future__ import annotations

import re
import threading
import time
from collections.abc import Callable
from functools import lru_cache

RATE_LIMIT_WINDOW_S = 300.0
RATE_LIMIT_THRESHOLD = 3
INPUT_CAP_TTL_S = 3600.0
BROKEN_PROVIDER_TTL_S = 3600.0

# The status every body-matching check here is scoped to. A 429 is identified by status
# alone; a 400 needs the body, since a provider can 400 for reasons that say nothing
# about the provider's health (see `matches_400`).
ERROR_STATUS = 400

# The built-in 400-body signatures for "the prompt is larger than this provider will
# accept". A rule can replace this set with `input_cap_patterns`, or switch the check
# off with an empty list.
DEFAULT_INPUT_CAP_PATTERNS: tuple[str, ...] = (
    r"exceeds the maximum (?:allowed )?(?:input )?length",
    r"maximum context length",
)


def _compile_one(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, re.IGNORECASE)


@lru_cache(maxsize=128)
def _compiled(patterns: tuple[str, ...]) -> tuple[re.Pattern[str], ...]:
    """Compile once per distinct pattern tuple. Keyed on the tuple rather than on the
    rule, so rules sharing a pattern set share the compiled regexes and an arbitrarily
    large rule set cannot grow the cache past its 128 entries."""
    return tuple(_compile_one(p) for p in patterns)


def validate_patterns(patterns: tuple[str, ...], *, field: str) -> None:
    """Reject a pattern set that cannot do what the operator meant: an invalid regex, or
    an empty pattern. The latter is not pedantry - `re.compile("").search(body)` matches
    every body, so one stray empty string in `broken_provider_patterns` would mark every
    provider of the model broken. Called at rule construction, so the error surfaces at
    first use rather than on a customer request."""
    for pattern in patterns:
        if not pattern.strip():
            raise ValueError(
                f"{field} must not contain an empty pattern (it matches every error body)"
            )
        try:
            _compile_one(pattern)
        except re.error as exc:
            raise ValueError(f"{field} contains an invalid regex {pattern!r}: {exc}") from exc


def matches_400(patterns: tuple[str, ...], status: object, message: str) -> bool:
    """Whether a failure is a 400 whose body matches one of `patterns`.

    Scoped to 400 deliberately. The two callers classify different conditions from the
    same status code - an oversized prompt (a condition the request caused) and a
    misconfigured provider (a condition the provider owns) - and neither is a statement
    about a 429, a 500, or a 404, so a body carrying a matching phrase under any other
    status is not evidence for either.

    An empty `patterns` matches nothing, which is how a rule switches a check off."""
    if status != ERROR_STATUS or not patterns:
        return False
    return any(p.search(message) for p in _compiled(patterns))


class RateLimitCooldown:
    """The per-(model, slug) memory of providers a failure hook has found unusable.

    Three states live here, all consulted by `is_skipped`:

    - 429 hits, which are windowed: `threshold` hits inside `window_s` make a provider
      hot. This is the only state the class name describes.
    - input caps, a soft skip for a bounded TTL. Soft because an input cap is the
      request's fault, so the caller prefers a non-capped provider but will still try a
      capped one rather than fail a request that might fit.
    - broken providers, a hard skip for a TTL the rule chooses. Hard because the rule
      matched an operator-authored signature for a provider-side fault, so there is
      nothing to gain by trying it.

    The window and the input-cap TTL are process-wide, but the broken TTL is per rule and
    is therefore passed to `record_broken` rather than held here. Each bounded state
    stores the instant it expires, not the instant it was recorded, so the read side
    (`is_input_capped`, `is_broken`) compares timestamps and needs no rule at all - which
    matters because `is_skipped` is called from selection paths that hold no rule."""

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
        self._broken: dict[str, float] = {}

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
                self._input_capped[key] = ts + self._input_cap_ttl_s

    def is_input_capped(self, model: str, slug: str, now: float | None = None) -> bool:
        """A provider that raised an input-cap error is skipped only for a bounded TTL: a
        one-off oversized prompt does not make it permanently incapable, and an
        unexpired-forever set is an unbounded leak keyed by model x slug."""
        return self._is_live(self._input_capped, model, slug, now)

    def record_broken(
        self,
        model: str,
        slug: str,
        ttl_s: float = BROKEN_PROVIDER_TTL_S,
        now: float | None = None,
    ) -> None:
        """Mark a provider as known-broken: a 400 body matched one of the rule's
        `broken_provider_patterns`, so the operator has told us this provider is
        misconfigured rather than merely slow or rate-limited.

        The TTL is a parameter because it is a per-rule setting. It must still be
        bounded: a provider may be fixed at any time, and an unbounded entry is also an
        unbounded leak keyed by model x slug."""
        if slug:
            key = f"{model}\x00{slug}"
            ts = now if now is not None else self._clock()
            with self._lock:
                self._broken[key] = ts + ttl_s

    def is_broken(self, model: str, slug: str, now: float | None = None) -> bool:
        """Whether this provider is currently marked broken, reaping the entry if its
        TTL has passed. A hard skip for its duration: no selection path tries it."""
        return self._is_live(self._broken, model, slug, now)

    def _is_live(
        self,
        store: dict[str, float],
        model: str,
        slug: str,
        now: float | None,
    ) -> bool:
        """Shared read for the two TTL-bounded states, which differ only in which dict
        they live in. Both store an expiry instant, so one comparison serves both."""
        if not slug:
            return False
        key = f"{model}\x00{slug}"
        ts = now if now is not None else self._clock()
        with self._lock:
            expires_at = store.get(key)
            if expires_at is None:
                return False
            if expires_at <= ts:
                del store[key]
                return False
            return True

    def is_skipped(self, model: str, slug: str, now: float | None = None) -> bool:
        return (
            self.is_hot(model, slug, now)
            or self.is_input_capped(model, slug, now)
            or self.is_broken(model, slug, now)
        )
