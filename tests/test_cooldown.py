from __future__ import annotations

import threading

from litellm_plugin_openrouter_pareto.cooldown import RATE_LIMIT_WINDOW_S, RateLimitCooldown


def test_three_hits_within_window_becomes_hot() -> None:
    cd = RateLimitCooldown(threshold=3)
    cd.record("z-ai/glm-5.2", "baseten", now=0.0)
    cd.record("z-ai/glm-5.2", "baseten", now=1.0)
    assert cd.is_hot("z-ai/glm-5.2", "baseten", now=2.0) is False
    cd.record("z-ai/glm-5.2", "baseten", now=2.0)
    assert cd.is_hot("z-ai/glm-5.2", "baseten", now=2.0) is True


def test_hits_expire_after_window() -> None:
    cd = RateLimitCooldown(threshold=3)
    cd.record("z-ai/glm-5.2", "baseten", now=0.0)
    cd.record("z-ai/glm-5.2", "baseten", now=1.0)
    cd.record("z-ai/glm-5.2", "baseten", now=2.0)
    assert cd.is_hot("z-ai/glm-5.2", "baseten", now=RATE_LIMIT_WINDOW_S + 3.0) is False


def test_exact_window_boundary_is_still_hot() -> None:
    cd = RateLimitCooldown(threshold=3)
    cd.record("z-ai/glm-5.2", "baseten", now=0.0)
    cd.record("z-ai/glm-5.2", "baseten", now=0.0)
    cd.record("z-ai/glm-5.2", "baseten", now=0.0)
    assert cd.is_hot("z-ai/glm-5.2", "baseten", now=RATE_LIMIT_WINDOW_S) is True
    assert cd.is_hot("z-ai/glm-5.2", "baseten", now=RATE_LIMIT_WINDOW_S + 0.001) is False


def test_record_prunes_expired_hits() -> None:
    cd = RateLimitCooldown(threshold=3)
    cd.record("z-ai/glm-5.2", "baseten", now=0.0)
    cd.record("z-ai/glm-5.2", "baseten", now=0.0)
    cd.record("z-ai/glm-5.2", "baseten", now=RATE_LIMIT_WINDOW_S + 1.0)
    assert cd.is_hot("z-ai/glm-5.2", "baseten", now=RATE_LIMIT_WINDOW_S + 1.0) is False


def test_concurrent_records_do_not_lose_hits() -> None:
    cd = RateLimitCooldown(threshold=1000)
    n = 200

    def _hammer() -> None:
        for _ in range(n):
            cd.record("z-ai/glm-5.2", "baseten")

    threads = [threading.Thread(target=_hammer) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert cd.is_hot("z-ai/glm-5.2", "baseten") is True


def test_empty_slug_is_noop() -> None:
    cd = RateLimitCooldown(threshold=1)
    cd.record("", "", now=0.0)
    assert cd.is_hot("", "", now=0.0) is False


def test_is_hot_prunes_and_unhots_after_expiry() -> None:
    cd = RateLimitCooldown(threshold=1)
    cd.record("z-ai/glm-5.2", "baseten", now=0.0)
    assert cd.is_hot("z-ai/glm-5.2", "baseten", now=1.0) is True
    assert cd.is_hot("z-ai/glm-5.2", "baseten", now=RATE_LIMIT_WINDOW_S + 1.0) is False


def test_record_input_cap_makes_skipped() -> None:
    cd = RateLimitCooldown(threshold=3)
    cd.record_input_cap("z-ai/glm-5.2", "baseten/fp8")
    assert cd.is_input_capped("z-ai/glm-5.2", "baseten/fp8") is True
    assert cd.is_skipped("z-ai/glm-5.2", "baseten/fp8") is True
    assert cd.is_skipped("z-ai/glm-5.2", "novita/fp8") is False


def test_input_cap_is_one_shot_not_windowed() -> None:
    cd = RateLimitCooldown(threshold=3, window_s=1.0)
    cd.record_input_cap("z-ai/glm-5.2", "baseten/fp8")
    import time

    time.sleep(1.1)
    assert cd.is_skipped("z-ai/glm-5.2", "baseten/fp8") is True


def test_is_input_cap_error_matches_base_ten_message() -> None:
    from litellm_plugin_openrouter_pareto.cooldown import is_input_cap_error

    msg = "Input length 562514 exceeds the maximum allowed input length of 524256 tokens"
    assert is_input_cap_error(400, msg) is True
    assert is_input_cap_error(400, "This model's maximum context length is 200000 tokens") is True
    assert is_input_cap_error(400, "some other bad request") is False
    assert is_input_cap_error(429, "rate limit") is False


def test_model_scoped_isolation_429() -> None:
    cd = RateLimitCooldown(threshold=3)
    cd.record("model-a", "shared/fp8")
    cd.record("model-a", "shared/fp8")
    cd.record("model-a", "shared/fp8")
    assert cd.is_hot("model-a", "shared/fp8") is True
    assert cd.is_hot("model-b", "shared/fp8") is False


def test_model_scoped_isolation_input_cap() -> None:
    cd = RateLimitCooldown(threshold=3)
    cd.record_input_cap("model-a", "shared/fp8")
    assert cd.is_input_capped("model-a", "shared/fp8") is True
    assert cd.is_input_capped("model-b", "shared/fp8") is False


def test_model_scoped_isolation_skipped() -> None:
    cd = RateLimitCooldown(threshold=1)
    cd.record("model-a", "shared/fp8")
    assert cd.is_skipped("model-a", "shared/fp8") is True
    assert cd.is_skipped("model-b", "shared/fp8") is False


def test_is_hot_does_not_mutate_shared_state() -> None:
    """is_hot is a predicate; calling it must not rewrite the hit history. A mutating
    is_* method is a landmine for any caller that uses it in a comprehension."""
    clock = {"t": 1000.0}
    cd = RateLimitCooldown(threshold=3, clock=lambda: clock["t"])
    cd.record("m", "s")
    cd.record("m", "s")
    before = dict(cd._hits)
    for _ in range(5):
        cd.is_hot("m", "s")
    assert cd._hits == before, "is_hot rewrote _hits"


def test_record_evicts_fully_aged_entries() -> None:
    """Aged-out slugs must not accumulate forever in _hits."""
    clock = {"t": 0.0}
    cd = RateLimitCooldown(window_s=300.0, threshold=3, clock=lambda: clock["t"])
    cd.record("m", "old/fp8")
    assert "m\x00old/fp8" in cd._hits
    clock["t"] = 1000.0
    cd.record("m", "fresh/fp8")
    assert "m\x00old/fp8" not in cd._hits, "aged slug was never evicted"
    assert "m\x00fresh/fp8" in cd._hits


def test_input_cap_expires_after_ttl() -> None:
    """A one-off oversized prompt must not mark a provider input-capped forever."""
    clock = {"t": 0.0}
    cd = RateLimitCooldown(input_cap_ttl_s=3600.0, clock=lambda: clock["t"])
    cd.record_input_cap("m", "s")
    assert cd.is_input_capped("m", "s") is True
    clock["t"] = 3600.1
    assert cd.is_input_capped("m", "s") is False
    assert "m\x00s" not in cd._input_capped, "expired input-cap key was not reaped"


def test_input_cap_within_ttl_still_skips() -> None:
    clock = {"t": 0.0}
    cd = RateLimitCooldown(input_cap_ttl_s=3600.0, clock=lambda: clock["t"])
    cd.record_input_cap("m", "s")
    clock["t"] = 3599.0
    assert cd.is_input_capped("m", "s") is True
