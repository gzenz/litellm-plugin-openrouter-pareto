from __future__ import annotations

import threading

import pytest

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
    """The 429 window does not govern the input-cap skip: it outlives the window it was
    recorded in. Driven by the injected clock, so the test costs no wall time."""
    clock = {"t": 0.0}
    cd = RateLimitCooldown(threshold=3, window_s=1.0, clock=lambda: clock["t"])
    cd.record_input_cap("z-ai/glm-5.2", "baseten/fp8")
    clock["t"] = 1.1
    assert cd.is_skipped("z-ai/glm-5.2", "baseten/fp8") is True


def test_matches_400_matches_base_ten_message() -> None:
    from litellm_plugin_openrouter_pareto.cooldown import (
        DEFAULT_INPUT_CAP_PATTERNS,
        matches_400,
    )

    msg = "Input length 562514 exceeds the maximum allowed input length of 524256 tokens"
    assert matches_400(DEFAULT_INPUT_CAP_PATTERNS, 400, msg) is True
    assert (
        matches_400(DEFAULT_INPUT_CAP_PATTERNS, 400, "This model's maximum context length is 200000 tokens")
        is True
    )
    assert matches_400(DEFAULT_INPUT_CAP_PATTERNS, 400, "some other bad request") is False
    assert matches_400(DEFAULT_INPUT_CAP_PATTERNS, 429, "rate limit") is False


def test_matches_400_is_scoped_to_400() -> None:
    """A 500 whose body happens to carry a matched phrase is not evidence for either
    classification: the two callers ask about a request-shaped condition and a
    provider-shaped one, and neither is a statement about a server error."""
    from litellm_plugin_openrouter_pareto.cooldown import matches_400

    patterns = (r"provider returned error",)
    assert matches_400(patterns, 500, "provider returned error") is False
    assert matches_400(patterns, 404, "provider returned error") is False
    assert matches_400(patterns, 400, "provider returned error") is True


def test_matches_400_empty_patterns_matches_nothing() -> None:
    """An empty pattern list is how a rule switches a check off, so it must not fall
    back to the built-ins or to a match-everything regex."""
    from litellm_plugin_openrouter_pareto.cooldown import matches_400

    assert matches_400((), 400, "maximum context length exceeded") is False


def test_validate_patterns_rejects_empty_pattern() -> None:
    """An empty pattern matches every body, so it would mark every provider of the
    model broken. Rejected at rule construction instead."""
    from litellm_plugin_openrouter_pareto.cooldown import validate_patterns

    with pytest.raises(ValueError, match="empty pattern"):
        validate_patterns(("ok", "   "), field="broken_provider_patterns")


def test_validate_patterns_rejects_invalid_regex() -> None:
    from litellm_plugin_openrouter_pareto.cooldown import validate_patterns

    with pytest.raises(ValueError, match="invalid regex"):
        validate_patterns(("unclosed (group",), field="input_cap_patterns")


def test_record_broken_makes_skipped_and_expires() -> None:
    clock = {"t": 0.0}
    cd = RateLimitCooldown(clock=lambda: clock["t"])
    cd.record_broken("m", "bad/fp8", ttl_s=600.0)
    assert cd.is_broken("m", "bad/fp8") is True
    assert cd.is_skipped("m", "bad/fp8") is True
    clock["t"] = 600.1
    assert cd.is_broken("m", "bad/fp8") is False
    assert "m\x00bad/fp8" not in cd._broken, "expired broken key was not reaped"


def test_broken_ttl_is_per_call() -> None:
    """The TTL is a per-rule setting, so it is resolved when the hit is recorded rather
    than held on the store: two rules sharing one store keep their own windows."""
    clock = {"t": 0.0}
    cd = RateLimitCooldown(clock=lambda: clock["t"])
    cd.record_broken("m", "short/fp8", ttl_s=60.0)
    cd.record_broken("m", "long/fp8", ttl_s=600.0)
    clock["t"] = 100.0
    assert cd.is_broken("m", "short/fp8") is False
    assert cd.is_broken("m", "long/fp8") is True


def test_broken_is_model_scoped() -> None:
    cd = RateLimitCooldown()
    cd.record_broken("model-a", "shared/fp8", ttl_s=600.0)
    assert cd.is_broken("model-a", "shared/fp8") is True
    assert cd.is_broken("model-b", "shared/fp8") is False


def test_empty_slug_is_noop_for_broken() -> None:
    cd = RateLimitCooldown()
    cd.record_broken("m", "", ttl_s=600.0)
    assert cd.is_broken("m", "") is False



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
