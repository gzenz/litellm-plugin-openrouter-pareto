from __future__ import annotations

import threading

from litellm_plugin_openrouter_pareto.cooldown import RATE_LIMIT_WINDOW_S, RateLimitCooldown


def test_three_hits_within_window_becomes_hot() -> None:
    cd = RateLimitCooldown(threshold=3)
    cd.record("baseten", now=0.0)
    cd.record("baseten", now=1.0)
    assert cd.is_hot("baseten", now=2.0) is False
    cd.record("baseten", now=2.0)
    assert cd.is_hot("baseten", now=2.0) is True


def test_hits_expire_after_window() -> None:
    cd = RateLimitCooldown(threshold=3)
    cd.record("baseten", now=0.0)
    cd.record("baseten", now=1.0)
    cd.record("baseten", now=2.0)
    assert cd.is_hot("baseten", now=RATE_LIMIT_WINDOW_S + 3.0) is False


def test_exact_window_boundary_is_still_hot() -> None:
    cd = RateLimitCooldown(threshold=3)
    cd.record("baseten", now=0.0)
    cd.record("baseten", now=0.0)
    cd.record("baseten", now=0.0)
    assert cd.is_hot("baseten", now=RATE_LIMIT_WINDOW_S) is True
    assert cd.is_hot("baseten", now=RATE_LIMIT_WINDOW_S + 0.001) is False


def test_record_prunes_expired_hits() -> None:
    cd = RateLimitCooldown(threshold=3)
    cd.record("baseten", now=0.0)
    cd.record("baseten", now=0.0)
    cd.record("baseten", now=RATE_LIMIT_WINDOW_S + 1.0)
    assert cd.is_hot("baseten", now=RATE_LIMIT_WINDOW_S + 1.0) is False


def test_concurrent_records_do_not_lose_hits() -> None:
    cd = RateLimitCooldown(threshold=1000)
    n = 200

    def _hammer() -> None:
        for _ in range(n):
            cd.record("baseten")

    threads = [threading.Thread(target=_hammer) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert cd.is_hot("baseten") is True


def test_empty_slug_is_noop() -> None:
    cd = RateLimitCooldown(threshold=1)
    cd.record("", now=0.0)
    assert cd.is_hot("", now=0.0) is False


def test_is_hot_prunes_and_unhots_after_expiry() -> None:
    cd = RateLimitCooldown(threshold=1)
    cd.record("baseten", now=0.0)
    assert cd.is_hot("baseten", now=1.0) is True
    assert cd.is_hot("baseten", now=RATE_LIMIT_WINDOW_S + 1.0) is False
