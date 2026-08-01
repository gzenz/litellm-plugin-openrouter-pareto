from __future__ import annotations

from typing import Any

import pytest

from litellm_plugin_openrouter_pareto.config import UnverifiedRegionPolicy, rule
from litellm_plugin_openrouter_pareto.cooldown import RateLimitCooldown
from litellm_plugin_openrouter_pareto.plugin import (
    OpenRouterParetoCallback,
    StrictProviderConflict,
    TelemetrySource,
)
from litellm_plugin_openrouter_pareto.telemetry import CacheEntry


def _entry(winner: str | None, safe_set: tuple[str, ...]) -> CacheEntry:
    return CacheEntry(
        winner=winner,
        candidate_winner=winner,
        candidate_streak=1,
        safe_set=safe_set,
        canonical_slug="z-ai/glm-5.2",
        canonical_slug_fetched_at=0.0,
        fetched_at=0.0,
        stale=False,
    )


class _StubTelemetry:
    def __init__(self, entry: CacheEntry | None) -> None:
        self.entry = entry
        self.calls: list[str] = []

    async def get(self, model: str) -> CacheEntry | None:
        self.calls.append(model)
        return self.entry


def _dep(dep_id: str) -> dict[str, Any]:
    return {
        "model_name": "z-ai/glm-5.2",
        "litellm_params": {"model": "openrouter/z-ai/glm-5.2"},
        "model_info": {"id": dep_id},
    }


def _id_of(d: dict[str, object]) -> str:
    mi = d.get("model_info")
    if isinstance(mi, dict):
        raw = mi.get("id")
        if isinstance(raw, str):
            return raw
    return ""


def _callback(
    entry: CacheEntry | None,
    *,
    cooldown: RateLimitCooldown | None = None,
    wildcard: bool = False,
    cold_start_fallback: tuple[str, ...] | None = None,
) -> tuple[OpenRouterParetoCallback, _StubTelemetry]:
    rules = {
        "z-ai/glm-5.2": rule(
            precision="fp8",
            min_context=1_000_000,
            min_stats_requests=100,
            wildcard=wildcard,
            cold_start_fallback=cold_start_fallback if cold_start_fallback is not None else (),
        )
    }
    stub = _StubTelemetry(entry)
    cb = OpenRouterParetoCallback(rules=rules, telemetry=stub, cooldown=cooldown)
    return cb, stub


def _wildcard_dep() -> dict[str, Any]:
    return {
        "model_name": "z-ai/glm-5.2",
        "litellm_params": {"model": "openrouter/z-ai/glm-5.2", "extra_body": {}},
        "model_info": {"id": "or-wildcard"},
    }


def _provider_only(d: dict[str, Any]) -> str | None:
    lp = d.get("litellm_params")
    if isinstance(lp, dict):
        eb = lp.get("extra_body")
        if isinstance(eb, dict):
            p = eb.get("provider")
            if isinstance(p, dict):
                only = p.get("only")
                if isinstance(only, list) and only and isinstance(only[0], str):
                    return only[0]
    return None


def _failure_kwargs(slug: str, exc: object, model: str = "z-ai/glm-5.2") -> dict[str, Any]:
    """Construct realistic async_log_failure_event kwargs matching litellm's
    model_call_details: extra_body at top level + under optional_params,
    model_info.id under litellm_params (a hash, not the or- slug)."""
    return {
        "model": model,
        "exception": exc,
        "extra_body": {"provider": {"only": [slug]}},
        "optional_params": {"extra_body": {"provider": {"only": [slug]}}},
        "litellm_params": {"model_info": {"id": "generated-hash-not-or-slug"}},
    }


def _deployments() -> list[dict[str, object]]:
    return [_dep("or-baseten"), _dep("or-novita"), _dep("or-siliconflow")]


async def test_narrows_to_winner_deployment() -> None:
    cb, _ = _callback(_entry("baseten", ("baseten", "novita", "siliconflow")))
    result = await cb.async_filter_deployments("z-ai/glm-5.2", _deployments(), None)
    assert [_id_of(d) for d in result] == ["or-baseten"]


async def test_winner_not_healthy_falls_back_to_next_preferred_slug() -> None:
    cb, _ = _callback(_entry("baseten", ("baseten", "novita", "siliconflow")))
    healthy = [_dep("or-novita"), _dep("or-siliconflow")]
    result = await cb.async_filter_deployments("z-ai/glm-5.2", healthy, None)
    assert [_id_of(d) for d in result] == ["or-novita"]


async def test_hot_winner_is_skipped_for_new_request() -> None:
    cd = RateLimitCooldown(threshold=1)
    cb, _ = _callback(_entry("baseten", ("baseten", "novita", "siliconflow")), cooldown=cd)
    cd.record("z-ai/glm-5.2", "baseten")
    healthy = _deployments()
    result = await cb.async_filter_deployments("z-ai/glm-5.2", healthy, None)
    assert [_id_of(d) for d in result] == ["or-novita"]


async def test_all_preferred_hot_falls_back_to_constrained_winner() -> None:
    cd = RateLimitCooldown(threshold=1)
    cb, _ = _callback(_entry("baseten", ("baseten", "novita")), cooldown=cd)
    cd.record("z-ai/glm-5.2", "baseten")
    cd.record("z-ai/glm-5.2", "novita")
    healthy = _deployments()
    result = await cb.async_filter_deployments("z-ai/glm-5.2", healthy, None)
    assert [_id_of(d) for d in result] == ["or-baseten"]


async def test_failure_event_records_429_until_hot() -> None:
    cd = RateLimitCooldown(threshold=3)
    cb, _ = _callback(_entry("baseten", ("baseten", "novita")), cooldown=cd)

    class _Exc:
        status_code = 429

    kwargs = _failure_kwargs("baseten/fp8", _Exc())
    await cb.async_log_failure_event(kwargs, None, 0.0, 0.0)
    await cb.async_log_failure_event(kwargs, None, 0.0, 0.0)
    assert cd.is_hot("z-ai/glm-5.2", "baseten/fp8") is False
    await cb.async_log_failure_event(kwargs, None, 0.0, 0.0)
    assert cd.is_hot("z-ai/glm-5.2", "baseten/fp8") is True


async def test_failure_event_ignores_non_429() -> None:
    cd = RateLimitCooldown(threshold=1)
    cb, _ = _callback(_entry("baseten", ("baseten",)), cooldown=cd)

    class _Exc:
        status_code = 500

    kwargs = _failure_kwargs("baseten/fp8", _Exc())
    await cb.async_log_failure_event(kwargs, None, 0.0, 0.0)
    assert cd.is_hot("z-ai/glm-5.2", "baseten/fp8") is False


async def test_failure_event_ignores_unmanaged_model() -> None:
    cd = RateLimitCooldown(threshold=1)
    cb, _ = _callback(_entry("baseten", ("baseten",)), cooldown=cd)

    class _Exc:
        status_code = 429

    kwargs = _failure_kwargs("baseten/fp8", _Exc(), model="openai/gpt-4o")
    await cb.async_log_failure_event(kwargs, None, 0.0, 0.0)
    assert cd.is_hot("z-ai/glm-5.2", "baseten") is False


async def test_failure_event_uses_litellm_metadata() -> None:
    cd = RateLimitCooldown(threshold=1)
    cb, _ = _callback(_entry("baseten", ("baseten",)), cooldown=cd)

    class _Exc:
        status_code = 429

    kwargs = _failure_kwargs("novita/fp8", _Exc())
    await cb.async_log_failure_event(kwargs, None, 0.0, 0.0)
    assert cd.is_hot("z-ai/glm-5.2", "novita/fp8") is True


async def test_failure_event_missing_exception_is_noop() -> None:
    cd = RateLimitCooldown(threshold=1)
    cb, _ = _callback(_entry("baseten", ("baseten",)), cooldown=cd)
    kwargs = {"model": "z-ai/glm-5.2", "extra_body": {"provider": {"only": ["baseten/fp8"]}}}
    await cb.async_log_failure_event(kwargs, None, 0.0, 0.0)
    assert cd.is_hot("z-ai/glm-5.2", "baseten") is False


async def test_failure_event_string_status_is_ignored() -> None:
    cd = RateLimitCooldown(threshold=1)
    cb, _ = _callback(_entry("baseten", ("baseten",)), cooldown=cd)

    class _Exc:
        status_code = "429"

    kwargs = _failure_kwargs("baseten/fp8", _Exc())
    await cb.async_log_failure_event(kwargs, None, 0.0, 0.0)
    assert cd.is_hot("z-ai/glm-5.2", "baseten/fp8") is False


async def test_empty_model_info_id_falls_back_to_extra_body() -> None:
    cd = RateLimitCooldown(threshold=1)
    cb, _ = _callback(_entry("baseten", ("baseten",)), cooldown=cd)

    class _Exc:
        status_code = 429

    kwargs = {
        "model": "z-ai/glm-5.2",
        "exception": _Exc(),
        "extra_body": {},
        "optional_params": {"extra_body": {"provider": {"only": ["novita/fp8"]}}},
        "litellm_params": {"model_info": {"id": ""}},
    }
    await cb.async_log_failure_event(kwargs, None, 0.0, 0.0)
    assert cd.is_hot("z-ai/glm-5.2", "novita/fp8") is True
    assert cd.is_hot("z-ai/glm-5.2", "baseten/fp8") is False


async def test_or_dash_only_id_falls_back_to_extra_body() -> None:
    cd = RateLimitCooldown(threshold=1)
    cb, _ = _callback(_entry("baseten", ("baseten",)), cooldown=cd)

    class _Exc:
        status_code = 429

    kwargs = {
        "model": "z-ai/glm-5.2",
        "exception": _Exc(),
        "extra_body": {},
        "optional_params": {"extra_body": {"provider": {"only": ["novita/fp8"]}}},
        "litellm_params": {"model_info": {"id": "or-"}},
    }
    await cb.async_log_failure_event(kwargs, None, 0.0, 0.0)
    assert cd.is_hot("z-ai/glm-5.2", "novita/fp8") is True


async def test_nonempty_unmatched_healthy_returns_full_list() -> None:
    cb, _ = _callback(_entry("baseten", ("baseten", "novita")))
    healthy = [_dep("or-venice"), _dep("or-zai")]
    result = await cb.async_filter_deployments("z-ai/glm-5.2", healthy, None)
    assert result == healthy
    assert len(result) == 2


async def test_no_safe_member_healthy_returns_full_list() -> None:
    cb, _ = _callback(_entry("baseten", ("baseten", "novita", "siliconflow")))
    healthy = [_dep("or-venice")]
    result = await cb.async_filter_deployments("z-ai/glm-5.2", healthy, None)
    assert result == healthy


async def test_stale_or_no_winner_returns_unchanged() -> None:
    cb, _ = _callback(None)
    healthy = _deployments()
    result = await cb.async_filter_deployments("z-ai/glm-5.2", healthy, None)
    assert result == healthy


async def test_entry_with_no_winner_returns_unchanged() -> None:
    cb, _ = _callback(_entry(None, ("baseten", "novita")))
    healthy = _deployments()
    result = await cb.async_filter_deployments("z-ai/glm-5.2", healthy, None)
    assert result == healthy


async def test_stale_entry_with_winner_returns_unchanged() -> None:
    stale = CacheEntry(
        winner="baseten",
        candidate_winner="baseten",
        candidate_streak=3,
        safe_set=("baseten", "novita", "siliconflow"),
        canonical_slug="z-ai/glm-5.2",
        canonical_slug_fetched_at=0.0,
        fetched_at=0.0,
        stale=True,
    )
    cb, _ = _callback(stale)
    healthy = _deployments()
    result = await cb.async_filter_deployments("z-ai/glm-5.2", healthy, None)
    assert result == healthy


async def test_never_narrows_below_input() -> None:
    cb, _ = _callback(_entry("baseten", ("baseten",)))
    result = await cb.async_filter_deployments("z-ai/glm-5.2", [], None)
    assert result == []


async def test_unmanaged_model_returns_unchanged_and_skips_telemetry() -> None:
    cb, stub = _callback(_entry("baseten", ("baseten",)))
    healthy = _deployments()
    result = await cb.async_filter_deployments("openai/gpt-4o", healthy, None)
    assert result == healthy
    assert stub.calls == []


async def test_dict_input_normalized_to_list_and_filtered() -> None:
    cb, _ = _callback(_entry("baseten", ("baseten",)))
    single = _dep("or-baseten")
    result = await cb.async_filter_deployments("z-ai/glm-5.2", single, None)
    assert result == [single]


async def test_telemetry_exception_returns_unchanged() -> None:
    class _BoomTelemetry:
        async def get(self, model: str) -> CacheEntry | None:
            raise RuntimeError("boom")

    rules = {"z-ai/glm-5.2": rule(precision="fp8", min_context=1_000_000, min_stats_requests=100)}
    cb = OpenRouterParetoCallback(rules=rules, telemetry=_BoomTelemetry())
    healthy = _deployments()
    result = await cb.async_filter_deployments("z-ai/glm-5.2", healthy, None)
    assert result == healthy


def test_slug_of_strips_or_prefix_from_model_info_id() -> None:
    cb, _ = _callback(None)
    assert cb.slug_of(_dep("or-baseten")) == "baseten"
    assert cb.slug_of(_dep("baseten")) == "baseten"


def test_slug_of_prefers_provider_only_over_model_info_id() -> None:
    cb, _ = _callback(None)
    dep: dict[str, Any] = {
        "model_name": "z-ai/glm-5.2",
        "litellm_params": {"model": "openrouter/z-ai/glm-5.2", "extra_body": {"provider": {"only": ["novita/fp8"]}}},
        "model_info": {"id": "or-some-other"},
    }
    assert cb.slug_of(dep) == "novita/fp8"


def test_slug_of_falls_back_to_model_info_id_when_no_provider_only() -> None:
    cb, _ = _callback(None)
    dep: dict[str, Any] = {
        "model_name": "z-ai/glm-5.2",
        "litellm_params": {"model": "openrouter/z-ai/glm-5.2"},
        "model_info": {"id": "or-novita/fp8"},
    }
    assert cb.slug_of(dep) == "novita/fp8"


def test_telemetry_is_a_telemetry_source() -> None:
    from litellm_plugin_openrouter_pareto.telemetry import Telemetry

    rules = {"z-ai/glm-5.2": rule(precision="fp8", min_context=1_000_000, min_stats_requests=100)}
    t = Telemetry(rules, cache_dir="/tmp/or-pareto-test-src")
    assert isinstance(t, TelemetrySource)


async def test_wildcard_injects_winner_provider_only() -> None:
    cb, _ = _callback(_entry("baseten", ("baseten", "novita", "siliconflow")), wildcard=True)
    deps = [_wildcard_dep()]
    result = await cb.async_filter_deployments("z-ai/glm-5.2", deps, None)
    assert _provider_only(result[0]) == "baseten"
    assert _provider_only(deps[0]) is None


async def test_wildcard_does_not_mutate_router_state() -> None:
    cb, _ = _callback(_entry("baseten", ("baseten", "novita", "siliconflow")), wildcard=True)
    deps = [_wildcard_dep()]
    original_extra_body = {"provider": {"only": ["preexisting"]}}
    deps[0]["litellm_params"]["extra_body"] = {"provider": {"only": ["preexisting"]}}
    await cb.async_filter_deployments("z-ai/glm-5.2", deps, None)
    assert deps[0]["litellm_params"]["extra_body"]["provider"]["only"] == ["preexisting"]
    _ = original_extra_body


async def test_wildcard_winner_A_then_B_uses_copies_not_shared_state() -> None:
    cb, _ = _callback(_entry("baseten", ("baseten", "novita")), wildcard=True)
    deps = [_wildcard_dep()]
    r1 = await cb.async_filter_deployments("z-ai/glm-5.2", deps, None)
    assert _provider_only(r1[0]) == "baseten"
    r2 = await cb.async_filter_deployments("z-ai/glm-5.2", deps, None)
    assert _provider_only(r2[0]) == "baseten"
    assert _provider_only(r1[0]) == "baseten"
    assert _provider_only(deps[0]) is None


async def test_wildcard_skips_hot_winner_injects_next() -> None:
    cd = RateLimitCooldown(threshold=1)
    cb, _ = _callback(_entry("baseten", ("baseten", "novita", "siliconflow")), cooldown=cd, wildcard=True)
    cd.record("z-ai/glm-5.2", "baseten")
    deps = [_wildcard_dep()]
    result = await cb.async_filter_deployments("z-ai/glm-5.2", deps, None)
    assert _provider_only(result[0]) == "novita"


async def test_wildcard_all_hot_falls_back_to_winner() -> None:
    cd = RateLimitCooldown(threshold=1)
    cb, _ = _callback(_entry("baseten", ("baseten", "novita")), cooldown=cd, wildcard=True)
    cd.record("z-ai/glm-5.2", "baseten")
    cd.record("z-ai/glm-5.2", "novita")
    deps = [_wildcard_dep()]
    result = await cb.async_filter_deployments("z-ai/glm-5.2", deps, None)
    assert _provider_only(result[0]) == "baseten"


async def test_wildcard_stale_uses_cold_start_fallback() -> None:
    cb, _ = _callback(None, wildcard=True, cold_start_fallback=("novita",))
    deps = [_wildcard_dep()]
    result = await cb.async_filter_deployments("z-ai/glm-5.2", deps, None)
    assert _provider_only(result[0]) == "novita"


async def test_wildcard_stale_no_fallback_returns_clean_no_stale_pin() -> None:
    cb, _ = _callback(None, wildcard=True)
    deps = [_wildcard_dep()]
    deps[0]["litellm_params"]["extra_body"] = {"provider": {"only": ["stale_pin"]}}
    result = await cb.async_filter_deployments("z-ai/glm-5.2", deps, None)
    assert _provider_only(result[0]) is None
    assert _provider_only(deps[0]) == "stale_pin"


async def test_wildcard_exception_uses_cold_start_fallback() -> None:
    class _Boom:
        async def get(self, model: str) -> CacheEntry | None:
            raise RuntimeError("boom")

    rules = {
        "z-ai/glm-5.2": rule(
            precision="fp8",
            min_context=1_000_000,
            min_stats_requests=100,
            wildcard=True,
            cold_start_fallback=["novita"],
        )
    }
    cb = OpenRouterParetoCallback(rules=rules, telemetry=_Boom())
    deps = [_wildcard_dep()]
    result = await cb.async_filter_deployments("z-ai/glm-5.2", deps, None)
    assert _provider_only(result[0]) == "novita"


async def test_non_wildcard_does_not_inject() -> None:
    cb, _ = _callback(_entry("baseten", ("baseten", "novita")))
    deps = [_dep("or-baseten"), _dep("or-novita")]
    result = await cb.async_filter_deployments("z-ai/glm-5.2", deps, None)
    assert [_id_of(d) for d in result] == ["or-baseten"]
    for d in deps:
        assert _provider_only(d) is None


def test_load_rules_from_settings_reads_litellm_attr() -> None:
    import litellm

    from litellm_plugin_openrouter_pareto.config import load_rules_from_settings

    old = getattr(litellm, "openrouter_pareto_rules", None)
    setattr(  # noqa: B010  # dynamic attr the proxy sets at startup; pyright can't see it
        litellm,
        "openrouter_pareto_rules",
        {
            "z-ai/glm-5.2": {
                "precision": ["fp8"],
                "min_context": 500000,
                "min_stats_requests": 50,
                "promotion_polls": 3,
                "wildcard": True,
            }
        },
    )
    try:
        rules = load_rules_from_settings()
    finally:
        if old is None:
            delattr(litellm, "openrouter_pareto_rules")
        else:
            setattr(litellm, "openrouter_pareto_rules", old)  # noqa: B010  # restore
    assert rules is not None
    r = rules["z-ai/glm-5.2"]
    assert r.min_context == 500000
    assert r.promotion_polls == 3
    assert r.wildcard is True


def test_load_rules_from_settings_returns_none_when_unset() -> None:
    import litellm

    from litellm_plugin_openrouter_pareto.config import load_rules_from_settings

    old = getattr(litellm, "openrouter_pareto_rules", None)
    if hasattr(litellm, "openrouter_pareto_rules"):
        delattr(litellm, "openrouter_pareto_rules")
    try:
        assert load_rules_from_settings() is None
    finally:
        if old is not None:
            setattr(litellm, "openrouter_pareto_rules", old)  # noqa: B010  # restore


def test_load_rules_raises_on_unknown_field() -> None:
    import litellm
    import pytest

    from litellm_plugin_openrouter_pareto.config import RuleConfigError, load_rules_from_settings

    old = getattr(litellm, "openrouter_pareto_rules", None)
    setattr(  # noqa: B010  # typo: wildard instead of wildcard; extra="forbid" must reject
        litellm,
        "openrouter_pareto_rules",
        {"z-ai/glm-5.2": {"precision": ["fp8"], "min_context": 1000000, "wildard": True}},
    )
    try:
        with pytest.raises(RuleConfigError):
            load_rules_from_settings()
    finally:
        if old is None:
            delattr(litellm, "openrouter_pareto_rules")
        else:
            setattr(litellm, "openrouter_pareto_rules", old)  # noqa: B010  # restore


def test_load_rules_raises_on_negative_min_context() -> None:
    import litellm
    import pytest

    from litellm_plugin_openrouter_pareto.config import RuleConfigError, load_rules_from_settings

    old = getattr(litellm, "openrouter_pareto_rules", None)
    setattr(  # noqa: B010
        litellm,
        "openrouter_pareto_rules",
        {"z-ai/glm-5.2": {"precision": ["fp8"], "min_context": -1}},
    )
    try:
        with pytest.raises(RuleConfigError):
            load_rules_from_settings()
    finally:
        if old is None:
            delattr(litellm, "openrouter_pareto_rules")
        else:
            setattr(litellm, "openrouter_pareto_rules", old)  # noqa: B010  # restore


def test_load_rules_raises_on_empty_mapping() -> None:
    import litellm
    import pytest

    from litellm_plugin_openrouter_pareto.config import RuleConfigError, load_rules_from_settings

    old = getattr(litellm, "openrouter_pareto_rules", None)
    setattr(litellm, "openrouter_pareto_rules", {})  # noqa: B010
    try:
        with pytest.raises(RuleConfigError):
            load_rules_from_settings()
    finally:
        if old is None:
            delattr(litellm, "openrouter_pareto_rules")
        else:
            setattr(litellm, "openrouter_pareto_rules", old)  # noqa: B010  # restore


def test_load_rules_raises_on_non_dict() -> None:
    import litellm
    import pytest

    from litellm_plugin_openrouter_pareto.config import RuleConfigError, load_rules_from_settings

    old = getattr(litellm, "openrouter_pareto_rules", None)
    setattr(litellm, "openrouter_pareto_rules", "not a dict")  # noqa: B010
    try:
        with pytest.raises(RuleConfigError):
            load_rules_from_settings()
    finally:
        if old is None:
            delattr(litellm, "openrouter_pareto_rules")
        else:
            setattr(litellm, "openrouter_pareto_rules", old)  # noqa: B010  # restore


def test_rule_direct_construction_keeps_wildcard_default_false() -> None:
    from litellm_plugin_openrouter_pareto.config import Rule

    r = Rule(
        precision=("fp8",),
        min_context=1_000_000,
        min_stats_requests=100,
        promotion_polls=2,
        value_regression_tolerance=0.0,
        cold_start_fallback=("novita",),
    )
    assert r.wildcard is False


async def test_explicit_telemetry_not_overwritten_by_yaml_rules() -> None:
    import litellm

    cb, stub = _callback(_entry("baseten", ("baseten",)))
    old = getattr(litellm, "openrouter_pareto_rules", None)
    setattr(  # noqa: B010
        litellm,
        "openrouter_pareto_rules",
        {"z-ai/glm-5.2": {"precision": ["fp8"], "min_context": 1000000, "wildcard": False}},
    )
    try:
        await cb.async_filter_deployments("z-ai/glm-5.2", _deployments(), None)
    finally:
        if old is None:
            delattr(litellm, "openrouter_pareto_rules")
        else:
            setattr(litellm, "openrouter_pareto_rules", old)  # noqa: B010  # restore
    assert stub.calls == ["z-ai/glm-5.2"]



async def test_failure_event_records_input_cap_on_400_too_large() -> None:
    cd = RateLimitCooldown(threshold=10)
    cb, _ = _callback(_entry("baseten/fp8", ("baseten/fp8",)), cooldown=cd)

    class _Exc:
        status_code = 400

        def __str__(self) -> str:
            return "Input length 562514 exceeds the maximum allowed input length of 524256 tokens"

    kwargs = _failure_kwargs("baseten/fp8", _Exc())
    await cb.async_log_failure_event(kwargs, None, 0.0, 0.0)
    assert cd.is_input_capped("z-ai/glm-5.2", "baseten/fp8") is True
    assert cd.is_hot("z-ai/glm-5.2", "baseten/fp8") is False


async def test_input_capped_winner_is_skipped_for_new_request() -> None:
    cd = RateLimitCooldown(threshold=10)
    cb, _ = _callback(_entry("baseten/fp8", ("baseten/fp8", "novita/fp8")), cooldown=cd, wildcard=True)
    cd.record_input_cap("z-ai/glm-5.2", "baseten/fp8")
    deps = [_wildcard_dep()]
    result = await cb.async_filter_deployments("z-ai/glm-5.2", deps, None)
    assert _provider_only(result[0]) == "novita/fp8"


async def test_error_log_only_when_rule_log_errors_true(tmp_path, monkeypatch) -> None:
    import os

    import litellm_plugin_openrouter_pareto.error_log as el

    log_path = tmp_path / "errors.log"
    monkeypatch.setenv("OPENROUTER_PARETO_ERROR_LOG", str(log_path))

    rules_on = {
        "z-ai/glm-5.2": rule(precision="fp8", min_context=1_000_000, min_stats_requests=100, log_errors=True)
    }
    rules_off = {
        "z-ai/glm-5.2": rule(precision="fp8", min_context=1_000_000, min_stats_requests=100, log_errors=False)
    }

    class _Exc:
        status_code = 500

        def __str__(self) -> str:
            return "internal error"

    kwargs = _failure_kwargs("baseten/fp8", _Exc())

    cb_on = OpenRouterParetoCallback(rules=rules_on, telemetry=_StubTelemetry(None))
    await cb_on.async_log_failure_event(kwargs, None, 0.0, 0.0)
    assert log_path.exists()
    assert "status=500" in log_path.read_text()

    log_path.unlink()
    cb_off = OpenRouterParetoCallback(rules=rules_off, telemetry=_StubTelemetry(None))
    await cb_off.async_log_failure_event(kwargs, None, 0.0, 0.0)
    assert not log_path.exists()
    _ = os, el


async def test_wildcard_failure_attribution_skips_failed_provider_next_request() -> None:
    """End-to-end: filter injects winner, that provider 429s, the failure hook
    records the injected slug (from the real model_call_details shape), and the
    next filter call walks to a different provider. Proves the cooldown is
    attributed to the provider that actually failed, not the synthetic id."""
    cd = RateLimitCooldown(threshold=1)
    cb, _ = _callback(
        _entry("baseten/fp8", ("baseten/fp8", "novita/fp8")),
        cooldown=cd,
        wildcard=True,
    )
    deps = [_wildcard_dep()]

    result = await cb.async_filter_deployments("z-ai/glm-5.2", deps, None)
    assert _provider_only(result[0]) == "baseten/fp8"

    class _Exc429:
        status_code = 429

    await cb.async_log_failure_event(_failure_kwargs("baseten/fp8", _Exc429()), None, 0.0, 0.0)
    assert cd.is_hot("z-ai/glm-5.2", "baseten/fp8") is True

    result2 = await cb.async_filter_deployments("z-ai/glm-5.2", deps, None)
    assert _provider_only(result2[0]) == "novita/fp8"


async def test_wildcard_all_input_capped_falls_to_winner_not_cold_start_fallback() -> None:
    """When all preferred are input-capped, the winner is used (preference skip,
    not hard exclusion). cold_start_fallback is only for stale/None telemetry."""
    cd = RateLimitCooldown(threshold=10)
    cb, _ = _callback(
        _entry("baseten/fp8", ("baseten/fp8", "novita/fp8")),
        cooldown=cd,
        wildcard=True,
        cold_start_fallback=("siliconflow/fp8",),
    )
    cd.record_input_cap("z-ai/glm-5.2", "baseten/fp8")
    cd.record_input_cap("z-ai/glm-5.2", "novita/fp8")
    deps = [_wildcard_dep()]
    result = await cb.async_filter_deployments("z-ai/glm-5.2", deps, None)
    assert _provider_only(result[0]) == "baseten/fp8"


async def test_wildcard_all_input_capped_falls_back_to_winner() -> None:
    """Input-cap is a preference (skip for winner pick), not a hard exclusion.
    When all preferred are input-capped, the winner is still pinned (matching
    the wrapper: better to try than to fail). A small request may succeed
    even under the cap."""
    cd = RateLimitCooldown(threshold=10)
    cb, _ = _callback(
        _entry("baseten/fp8", ("baseten/fp8",)),
        cooldown=cd,
        wildcard=True,
        cold_start_fallback=("baseten/fp8",),
    )
    cd.record_input_cap("z-ai/glm-5.2", "baseten/fp8")
    deps = [_wildcard_dep()]
    result = await cb.async_filter_deployments("z-ai/glm-5.2", deps, None)
    assert _provider_only(result[0]) == "baseten/fp8"


async def test_pinned_all_input_capped_falls_back_to_winner() -> None:
    cd = RateLimitCooldown(threshold=10)
    cb, _ = _callback(
        _entry("baseten/fp8", ("baseten/fp8", "novita/fp8")),
        cooldown=cd,
    )
    cd.record_input_cap("z-ai/glm-5.2", "baseten/fp8")
    cd.record_input_cap("z-ai/glm-5.2", "novita/fp8")
    deps = [
        {
            "model_name": "m",
            "litellm_params": {
                "model": "openrouter/z-ai/glm-5.2",
                "extra_body": {"provider": {"only": ["baseten/fp8"]}},
            },
            "model_info": {"id": "or-baseten/fp8"},
        },
        {
            "model_name": "m",
            "litellm_params": {
                "model": "openrouter/z-ai/glm-5.2",
                "extra_body": {"provider": {"only": ["novita/fp8"]}},
            },
            "model_info": {"id": "or-novita/fp8"},
        },
    ]
    result = await cb.async_filter_deployments("z-ai/glm-5.2", deps, None)
    assert [_id_of(d) for d in result] == ["or-baseten/fp8"]


def test_error_log_unwritable_path_does_not_leak_body(tmp_path) -> None:
    import os

    from litellm_plugin_openrouter_pareto.error_log import or_error_log

    log_path = tmp_path / "sub" / "errors.log"
    os.environ["OPENROUTER_PARETO_ERROR_LOG"] = str(log_path)
    log_path.parent.mkdir(parents=True)
    log_path.parent.chmod(0o555)
    try:
        or_error_log("model=m slug=baseten/fp8 status=400 body=SECRET_PROMPT_DATA")
        assert not log_path.exists()
    finally:
        log_path.parent.chmod(0o755)
        del os.environ["OPENROUTER_PARETO_ERROR_LOG"]


async def test_wildcard_fallback_skips_hot_first_fallback() -> None:
    """First cold fallback is hot, second is healthy -> second is chosen."""
    cd = RateLimitCooldown(threshold=1)
    cb, _ = _callback(None, wildcard=True, cold_start_fallback=("a/fp8", "b/fp8"), cooldown=cd)
    cd.record("z-ai/glm-5.2", "a/fp8")
    deps = [_wildcard_dep()]
    result = await cb.async_filter_deployments("z-ai/glm-5.2", deps, None)
    assert _provider_only(result[0]) == "b/fp8"


async def test_wildcard_fallback_all_hot_returns_stripped_constrained() -> None:
    """All fallbacks hot (429-hot) -> _stripped_copy (constrained, no provider)."""
    cd = RateLimitCooldown(threshold=1)
    cb, _ = _callback(None, wildcard=True, cold_start_fallback=("a/fp8", "b/fp8"), cooldown=cd)
    cd.record("z-ai/glm-5.2", "a/fp8")
    cd.record("z-ai/glm-5.2", "b/fp8")
    deps = [_wildcard_dep()]
    result = await cb.async_filter_deployments("z-ai/glm-5.2", deps, None)
    lp = result[0]["litellm_params"]
    assert isinstance(lp, dict)
    eb = lp.get("extra_body")
    assert isinstance(eb, dict)
    provider = eb.get("provider")
    assert isinstance(provider, dict)
    assert "only" not in provider
    assert provider["zdr"] is True
    assert provider["allow_fallbacks"] is False


async def test_wildcard_fallback_first_capped_second_hot_skips_both() -> None:
    """First capped, second hot: both are skipped by is_skipped -> _stripped_copy."""
    cd = RateLimitCooldown(threshold=1)
    cb, _ = _callback(None, wildcard=True, cold_start_fallback=("a/fp8", "b/fp8"), cooldown=cd)
    cd.record_input_cap("z-ai/glm-5.2", "a/fp8")
    cd.record("z-ai/glm-5.2", "b/fp8")
    deps = [_wildcard_dep()]
    result = await cb.async_filter_deployments("z-ai/glm-5.2", deps, None)
    lp = result[0]["litellm_params"]
    assert isinstance(lp, dict)
    eb = lp.get("extra_body")
    assert isinstance(eb, dict)
    provider = eb.get("provider")
    assert isinstance(provider, dict)
    assert "only" not in provider
    assert provider["zdr"] is True
    assert provider["allow_fallbacks"] is False


async def test_wildcard_fallback_all_capped_preserves_zdr() -> None:
    """All fallbacks input-capped -> strips only but keeps zdr and allow_fallbacks."""
    cd = RateLimitCooldown(threshold=1)
    cb, _ = _callback(None, wildcard=True, cold_start_fallback=("a/fp8",), cooldown=cd)
    cd.record_input_cap("z-ai/glm-5.2", "a/fp8")
    deps = [_wildcard_dep()]
    result = await cb.async_filter_deployments("z-ai/glm-5.2", deps, None)
    lp = result[0]["litellm_params"]
    assert isinstance(lp, dict)
    eb = lp.get("extra_body")
    assert isinstance(eb, dict)
    provider = eb.get("provider")
    assert isinstance(provider, dict)
    assert "only" not in provider
    assert provider["zdr"] is True
    assert provider["allow_fallbacks"] is False


def test_error_log_fchmod_failure_does_not_write_body(tmp_path, monkeypatch) -> None:
    import os

    from litellm_plugin_openrouter_pareto.error_log import or_error_log

    log_path = tmp_path / "errors.log"
    monkeypatch.setenv("OPENROUTER_PARETO_ERROR_LOG", str(log_path))

    original_fchmod = os.fchmod
    calls = []

    def _failing_fchmod(fd, mode):
        calls.append((fd, mode))
        raise PermissionError("forced")

    monkeypatch.setattr(os, "fchmod", _failing_fchmod)
    or_error_log("model=m slug=baseten/fp8 status=400 body=SECRET")
    monkeypatch.setattr(os, "fchmod", original_fchmod)
    assert not log_path.exists() or "SECRET" not in log_path.read_text()
    assert len(calls) == 1


async def test_stripped_copy_overrides_unsafe_zdr() -> None:
    """A deployment with zdr=False / allow_fallbacks=True gets overridden."""
    cd = RateLimitCooldown(threshold=1)
    cb, _ = _callback(None, wildcard=True, cold_start_fallback=("a/fp8",), cooldown=cd)
    cd.record_input_cap("z-ai/glm-5.2", "a/fp8")
    dep: dict[str, Any] = {
        "model_name": "m",
        "litellm_params": {
            "model": "openrouter/z-ai/glm-5.2",
            "extra_body": {"provider": {"only": ["stale"], "zdr": False, "allow_fallbacks": True}},
        },
        "model_info": {"id": "or-w"},
    }
    result = await cb.async_filter_deployments("z-ai/glm-5.2", [dep], None)
    lp = result[0]["litellm_params"]
    assert isinstance(lp, dict)
    eb = lp.get("extra_body")
    assert isinstance(eb, dict)
    provider = eb.get("provider")
    assert isinstance(provider, dict)
    assert provider["zdr"] is True
    assert provider["allow_fallbacks"] is False


def test_load_rules_from_settings_parses_exclude_regions_per_model() -> None:
    """exclude_regions is per-model YAML config, uppercased on the way in."""
    import litellm

    from litellm_plugin_openrouter_pareto.config import load_rules_from_settings

    old = getattr(litellm, "openrouter_pareto_rules", None)
    setattr(  # noqa: B010  # dynamic attr the proxy sets at startup; pyright can't see it
        litellm,
        "openrouter_pareto_rules",
        {
            "z-ai/glm-5.2": {
                "precision": ["fp8"],
                "min_context": 1000000,
                "min_stats_requests": 100,
                "exclude_regions": ["us", "SG"],
            },
            "other/model": {
                "precision": ["fp8"],
                "min_context": 1000000,
                "min_stats_requests": 100,
            },
        },
    )
    try:
        rules = load_rules_from_settings()
    finally:
        if old is None:
            delattr(litellm, "openrouter_pareto_rules")
        else:
            setattr(litellm, "openrouter_pareto_rules", old)  # noqa: B010  # restore
    assert rules is not None
    assert rules["z-ai/glm-5.2"].exclude_regions == ("US", "SG")
    assert rules["other/model"].exclude_regions == ()


def _wildcard_region_callback(
    entry: CacheEntry | None,
    *,
    exclude_regions: tuple[str, ...] = ("US",),
    cold_start_fallback: tuple[str, ...] = ("baseten/fp8", "novita/fp8"),
    unverified_region_policy: UnverifiedRegionPolicy = "no_route",
) -> tuple[OpenRouterParetoCallback, _StubTelemetry]:
    rules = {
        "z-ai/glm-5.2": rule(
            precision="fp8",
            min_context=1_000_000,
            min_stats_requests=100,
            wildcard=True,
            cold_start_fallback=list(cold_start_fallback),
            exclude_regions=list(exclude_regions),
            unverified_region_policy=unverified_region_policy,
        )
    }
    stub = _StubTelemetry(entry)
    return OpenRouterParetoCallback(rules=rules, telemetry=stub), stub


def _entry_with_excluded(
    excluded: tuple[str, ...], *, stale: bool, allowed: tuple[str, ...] = ("novita", "baseten")
) -> CacheEntry:
    return CacheEntry(
        winner=None if stale else "novita/fp8",
        candidate_winner="novita/fp8",
        candidate_streak=1,
        safe_set=("novita/fp8",),
        canonical_slug="z-ai/glm-5.2",
        canonical_slug_fetched_at=0.0,
        fetched_at=0.0,
        stale=stale,
        excluded_bases=excluded,
        allowed_bases=tuple(a for a in allowed if a not in excluded),
    )


async def test_cold_start_fallback_skips_region_excluded_org() -> None:
    """Stale telemetry still carries the excluded org bases, so the cold-start
    fallback must skip baseten and take novita."""
    entry = _entry_with_excluded(("baseten",), stale=True)
    cb, _ = _wildcard_region_callback(entry)
    result = await cb.async_filter_deployments("z-ai/glm-5.2", [_wildcard_dep()], None)
    assert _provider_only(result[0]) == "novita/fp8"


async def test_cold_start_no_route_policy_returns_no_deployment() -> None:
    """Default no_route: with no telemetry the region policy is unverifiable, so we
    decline to route rather than send an unpinned request OR may serve from anywhere."""
    cb, _ = _wildcard_region_callback(None)
    result = await cb.async_filter_deployments("z-ai/glm-5.2", [_wildcard_dep()], None)
    assert result == []


async def test_cold_start_unpinned_policy_sends_constrained_unpinned() -> None:
    """unverified_region_policy=unpinned keeps the model available at the cost of
    letting OR choose the provider."""
    cb, _ = _wildcard_region_callback(None, unverified_region_policy="unpinned")
    result = await cb.async_filter_deployments("z-ai/glm-5.2", [_wildcard_dep()], None)
    assert _provider_only(result[0]) is None
    lp = result[0]["litellm_params"]
    assert isinstance(lp, dict)
    eb = lp.get("extra_body")
    assert isinstance(eb, dict)
    provider = eb.get("provider")
    assert isinstance(provider, dict)
    assert provider["zdr"] is True
    assert provider["allow_fallbacks"] is False


async def test_cold_start_trust_fallback_policy_pins_configured_fallback() -> None:
    """unverified_region_policy=trust_fallback takes the operator's word that the
    configured cold_start_fallback slugs are region-compliant."""
    cb, _ = _wildcard_region_callback(None, unverified_region_policy="trust_fallback")
    result = await cb.async_filter_deployments("z-ai/glm-5.2", [_wildcard_dep()], None)
    assert _provider_only(result[0]) == "baseten/fp8"


async def test_cold_start_fallback_unaffected_when_no_exclude_regions() -> None:
    """Without exclude_regions the cold-start path keeps its old behavior."""
    cb, _ = _wildcard_region_callback(None, exclude_regions=())
    result = await cb.async_filter_deployments("z-ai/glm-5.2", [_wildcard_dep()], None)
    assert _provider_only(result[0]) == "baseten/fp8"


async def test_all_fallbacks_region_excluded_declines_to_route_under_no_route() -> None:
    entry = _entry_with_excluded(("baseten", "novita"), stale=True)
    cb, _ = _wildcard_region_callback(entry)
    result = await cb.async_filter_deployments("z-ai/glm-5.2", [_wildcard_dep()], None)
    assert result == []


async def test_all_fallbacks_region_excluded_strips_pin_under_unpinned() -> None:
    entry = _entry_with_excluded(("baseten", "novita"), stale=True)
    cb, _ = _wildcard_region_callback(entry, unverified_region_policy="unpinned")
    result = await cb.async_filter_deployments("z-ai/glm-5.2", [_wildcard_dep()], None)
    assert _provider_only(result[0]) is None


def test_exclude_regions_normalizes_whitespace_case_and_duplicates() -> None:
    from litellm_plugin_openrouter_pareto.config import rule as _rule

    r = _rule(
        precision="fp8",
        min_context=1_000_000,
        min_stats_requests=100,
        exclude_regions=[" us ", "US", "sg"],
    )
    assert r.exclude_regions == ("US", "SG")


def test_unverified_region_policy_rejects_unknown_value() -> None:
    import litellm

    from litellm_plugin_openrouter_pareto.config import RuleConfigError, load_rules_from_settings

    old = getattr(litellm, "openrouter_pareto_rules", None)
    setattr(  # noqa: B010  # dynamic attr the proxy sets at startup
        litellm,
        "openrouter_pareto_rules",
        {
            "z-ai/glm-5.2": {
                "precision": ["fp8"],
                "exclude_regions": ["US"],
                "unverified_region_policy": "yolo",
            }
        },
    )
    try:
        with pytest.raises(RuleConfigError):
            load_rules_from_settings()
    finally:
        if old is None:
            delattr(litellm, "openrouter_pareto_rules")
        else:
            setattr(litellm, "openrouter_pareto_rules", old)  # noqa: B010  # restore


def test_region_policy_fields_load_from_yaml() -> None:
    import litellm

    from litellm_plugin_openrouter_pareto.config import load_rules_from_settings

    old = getattr(litellm, "openrouter_pareto_rules", None)
    setattr(  # noqa: B010  # dynamic attr the proxy sets at startup
        litellm,
        "openrouter_pareto_rules",
        {
            "z-ai/glm-5.2": {
                "precision": ["fp8"],
                "exclude_regions": ["US"],
                "allow_unknown_region": True,
                "unverified_region_policy": "trust_fallback",
            }
        },
    )
    try:
        rules = load_rules_from_settings()
    finally:
        if old is None:
            delattr(litellm, "openrouter_pareto_rules")
        else:
            setattr(litellm, "openrouter_pareto_rules", old)  # noqa: B010  # restore
    assert rules is not None
    r = rules["z-ai/glm-5.2"]
    assert r.allow_unknown_region is True
    assert r.unverified_region_policy == "trust_fallback"
    assert r.exclude_regions == ("US",)


def _pinned_dep(slug: str) -> dict[str, Any]:
    return {
        "model_name": "z-ai/glm-5.2",
        "litellm_params": {
            "model": "openrouter/z-ai/glm-5.2",
            "extra_body": {"provider": {"only": [slug]}},
        },
        "model_info": {"id": f"or-{slug}"},
    }


def _pinned_region_callback(
    entry: CacheEntry | None,
    *,
    unverified_region_policy: UnverifiedRegionPolicy = "no_route",
) -> OpenRouterParetoCallback:
    """Pinned (non-wildcard) mode with a region policy - the DEFAULT deployment mode."""
    rules = {
        "z-ai/glm-5.2": rule(
            precision="fp8",
            min_context=1_000_000,
            min_stats_requests=100,
            wildcard=False,
            exclude_regions=["SG"],
            unverified_region_policy=unverified_region_policy,
        )
    }
    return OpenRouterParetoCallback(rules=rules, telemetry=_StubTelemetry(entry))


def _region_entry(
    excluded: tuple[str, ...],
    winner: str = "allowed/fp8",
    *,
    stale: bool = False,
    allowed: tuple[str, ...] = ("allowed",),
) -> CacheEntry:
    return CacheEntry(
        winner=winner,
        candidate_winner=winner,
        candidate_streak=1,
        safe_set=(winner,),
        canonical_slug="z-ai/glm-5.2",
        canonical_slug_fetched_at=0.0,
        fetched_at=0.0,
        stale=stale,
        excluded_bases=excluded,
        allowed_bases=tuple(a for a in allowed if a not in excluded),
    )


async def test_pinned_cold_start_no_route_refuses_excluded_pin() -> None:
    """A deployment can already be pinned to an excluded org, so the pinned path needs
    the policy too - scorer-side filtering alone leaves the default mode unguarded."""
    cb = _pinned_region_callback(None)
    result = await cb.async_filter_deployments("z-ai/glm-5.2", [_pinned_dep("sg-provider/fp8")], None)
    assert result == []


async def test_pinned_stale_entry_still_drops_excluded_deployment() -> None:
    """A stale entry keeps the last verified excluded_bases, so it remains enforceable."""
    cb = _pinned_region_callback(_region_entry(("sg-provider",), stale=True))
    result = await cb.async_filter_deployments("z-ai/glm-5.2", [_pinned_dep("sg-provider/fp8")], None)
    assert result == []


async def test_pinned_fresh_entry_never_falls_back_to_excluded_deployment() -> None:
    """Winner/safe-set miss must not reintroduce an excluded deployment via the
    'return the full list' fallback."""
    cb = _pinned_region_callback(_region_entry(("blocked",)))
    result = await cb.async_filter_deployments("z-ai/glm-5.2", [_pinned_dep("blocked/fp8")], None)
    assert result == []


async def test_pinned_mixed_list_selects_only_allowed_deployment() -> None:
    cb = _pinned_region_callback(_region_entry(("blocked",)))
    deps = [_pinned_dep("blocked/fp8"), _pinned_dep("allowed/fp8")]
    result = await cb.async_filter_deployments("z-ai/glm-5.2", deps, None)
    assert [_provider_only(d) for d in result] == ["allowed/fp8"]


async def test_pinned_unpinned_policy_strips_excluded_pin() -> None:
    cb = _pinned_region_callback(None, unverified_region_policy="unpinned")
    result = await cb.async_filter_deployments("z-ai/glm-5.2", [_pinned_dep("sg-provider/fp8")], None)
    assert _provider_only(result[0]) is None


async def test_pinned_without_exclude_regions_is_unchanged() -> None:
    """No region policy -> the pinned path keeps its original behavior exactly."""
    cb, _ = _callback(None)
    deps = [_pinned_dep("sg-provider/fp8")]
    result = await cb.async_filter_deployments("z-ai/glm-5.2", deps, None)
    assert [_provider_only(d) for d in result] == ["sg-provider/fp8"]


async def test_pinned_telemetry_exception_applies_region_policy() -> None:
    """A telemetry failure verifies nothing, so the policy must still hold."""

    class _Boom:
        async def get(self, model: str) -> CacheEntry | None:
            raise RuntimeError("telemetry down")

    rules = {
        "z-ai/glm-5.2": rule(
            precision="fp8",
            min_context=1_000_000,
            min_stats_requests=100,
            exclude_regions=["SG"],
        )
    }
    cb = OpenRouterParetoCallback(rules=rules, telemetry=_Boom())
    result = await cb.async_filter_deployments("z-ai/glm-5.2", [_pinned_dep("sg-provider/fp8")], None)
    assert result == []


def test_rule_helper_rejects_invalid_unverified_region_policy() -> None:
    """Validation lives in Rule itself, so YAML and rule() cannot disagree."""
    from litellm_plugin_openrouter_pareto.config import rule as _rule

    for bad in ("typo", "no-route", "NO_ROUTE", ""):
        with pytest.raises(ValueError, match="unverified_region_policy"):
            _rule(
                precision="fp8",
                min_context=1_000_000,
                min_stats_requests=100,
                unverified_region_policy=bad,  # pyright: ignore[reportArgumentType]  # deliberately invalid to test the runtime guard
            )


def test_rule_helper_accepts_every_valid_policy() -> None:
    from litellm_plugin_openrouter_pareto.config import rule as _rule

    for good in ("no_route", "unpinned", "trust_fallback"):
        r = _rule(
            precision="fp8",
            min_context=1_000_000,
            min_stats_requests=100,
            unverified_region_policy=good,
        )
        assert r.unverified_region_policy == good


async def test_provider_absent_from_telemetry_is_not_assumed_allowed() -> None:
    """Closed-world: eligibility needs affirmative evidence. A provider that never
    appeared in telemetry has an unknown location, not a safe one."""
    cb = _pinned_region_callback(_region_entry(("sg-provider",), allowed=("allowed",)))
    result = await cb.async_filter_deployments(
        "z-ai/glm-5.2", [_pinned_dep("never-seen-in-telemetry/fp8")], None
    )
    assert result == []


async def test_verified_allowed_provider_still_routes() -> None:
    """The closed-world check must not reject providers with positive evidence."""
    cb = _pinned_region_callback(_region_entry(("sg-provider",), allowed=("allowed",)))
    result = await cb.async_filter_deployments("z-ai/glm-5.2", [_pinned_dep("allowed/fp8")], None)
    assert [_provider_only(d) for d in result] == ["allowed/fp8"]


async def test_unidentifiable_deployment_is_not_assumed_allowed() -> None:
    """A deployment with no provider.only and no usable model_info.id cannot be
    region-checked, so it must not slip through as an empty base string."""
    cb = _pinned_region_callback(_region_entry(("sg-provider",), allowed=("allowed",)))
    dep: dict[str, Any] = {
        "model_name": "z-ai/glm-5.2",
        "litellm_params": {"model": "openrouter/z-ai/glm-5.2"},
        "model_info": {},
    }
    result = await cb.async_filter_deployments("z-ai/glm-5.2", [dep], None)
    assert result == []


async def test_absent_provider_routes_when_allow_unknown_region() -> None:
    cb = _pinned_region_callback_lenient(_region_entry(("sg-provider",), allowed=("allowed",)))
    result = await cb.async_filter_deployments(
        "z-ai/glm-5.2", [_pinned_dep("never-seen-in-telemetry/fp8")], None
    )
    assert [_provider_only(d) for d in result] == ["never-seen-in-telemetry/fp8"]


def _pinned_region_callback_lenient(entry: CacheEntry | None) -> OpenRouterParetoCallback:
    rules = {
        "z-ai/glm-5.2": rule(
            precision="fp8",
            min_context=1_000_000,
            min_stats_requests=100,
            exclude_regions=["SG"],
            allow_unknown_region=True,
        )
    }
    return OpenRouterParetoCallback(rules=rules, telemetry=_StubTelemetry(entry))


async def test_trust_fallback_with_empty_list_does_not_degrade_to_unpinned() -> None:
    """trust_fallback means 'only the operator-vetted slugs'. Exhausting them must fail
    closed; silently dropping the pin would be the `unpinned` policy instead."""
    cb, _ = _wildcard_region_callback(
        None, cold_start_fallback=(), unverified_region_policy="trust_fallback"
    )
    result = await cb.async_filter_deployments("z-ai/glm-5.2", [_wildcard_dep()], None)
    assert result == []


async def test_trust_fallback_all_cooled_down_does_not_degrade_to_unpinned() -> None:
    cd = RateLimitCooldown(threshold=1)
    rules = {
        "z-ai/glm-5.2": rule(
            precision="fp8",
            min_context=1_000_000,
            min_stats_requests=100,
            wildcard=True,
            cold_start_fallback=["baseten/fp8"],
            exclude_regions=["SG"],
            unverified_region_policy="trust_fallback",
        )
    }
    cb = OpenRouterParetoCallback(rules=rules, telemetry=_StubTelemetry(None), cooldown=cd)
    cd.record("z-ai/glm-5.2", "baseten/fp8")
    result = await cb.async_filter_deployments("z-ai/glm-5.2", [_wildcard_dep()], None)
    assert result == []


async def test_pinned_trust_fallback_only_trusts_configured_slugs() -> None:
    """In pinned mode trust_fallback must vet against cold_start_fallback, not accept
    whatever pin a deployment happens to carry."""
    rules = {
        "z-ai/glm-5.2": rule(
            precision="fp8",
            min_context=1_000_000,
            min_stats_requests=100,
            exclude_regions=["SG"],
            cold_start_fallback=["vetted/fp8"],
            unverified_region_policy="trust_fallback",
        )
    }
    cb = OpenRouterParetoCallback(rules=rules, telemetry=_StubTelemetry(None))
    deps = [_pinned_dep("random-unvetted/fp8"), _pinned_dep("vetted/fp8")]
    result = await cb.async_filter_deployments("z-ai/glm-5.2", deps, None)
    assert [_provider_only(d) for d in result] == ["vetted/fp8"]


def test_exclude_regions_normalizes_and_compares_verbatim() -> None:
    """The plugin does not police which strings are valid country codes; it normalizes
    (strip/upper/dedupe) and compares against whatever OpenRouter reports. Any non-empty
    string the operator and OR agree on works."""
    from litellm_plugin_openrouter_pareto.config import rule as _rule

    r = _rule(
        precision="fp8",
        min_context=1_000_000,
        min_stats_requests=100,
        exclude_regions=[" us ", "US", "sg", ""],
    )
    assert r.exclude_regions == ("US", "SG")


def _multi_pinned_dep(slugs: list[str]) -> dict[str, Any]:
    """A deployment whose provider.only authorizes SEVERAL providers. OpenRouter may
    serve from any entry, so every entry has to clear the region policy."""
    return {
        "model_name": "z-ai/glm-5.2",
        "litellm_params": {
            "model": "openrouter/z-ai/glm-5.2",
            "extra_body": {"provider": {"only": slugs}},
        },
        "model_info": {"id": "or-allowed/fp8"},
    }


async def test_pinned_multi_slug_rejects_when_any_entry_is_excluded() -> None:
    """provider.only is an allowlist, not a scalar pin: reading only only[0] would
    approve [allowed, excluded] while still authorizing the excluded provider."""
    cb = _pinned_region_callback(_region_entry(("blocked",)))
    dep = _multi_pinned_dep(["allowed/fp8", "blocked/fp8"])
    result = await cb.async_filter_deployments("z-ai/glm-5.2", [dep], None)
    assert result == []


async def test_pinned_multi_slug_routes_when_every_entry_is_allowed() -> None:
    cb = _pinned_region_callback(
        _region_entry(("blocked",), allowed=("allowed", "second"))
    )
    dep = _multi_pinned_dep(["allowed/fp8", "second/fp8"])
    result = await cb.async_filter_deployments("z-ai/glm-5.2", [dep], None)
    assert len(result) == 1


async def test_malformed_provider_only_is_unknown_not_rescued_by_model_info() -> None:
    """A present-but-malformed allowlist must not be overridden by the friendlier
    model_info.id identity; that would approve a deployment on the wrong evidence."""
    cb = _pinned_region_callback(_region_entry((), allowed=("allowed",)))
    dep: dict[str, Any] = {
        "model_name": "z-ai/glm-5.2",
        "litellm_params": {
            "model": "openrouter/z-ai/glm-5.2",
            "extra_body": {"provider": {"only": ["allowed/fp8", ""]}},
        },
        "model_info": {"id": "or-allowed/fp8"},
    }
    result = await cb.async_filter_deployments("z-ai/glm-5.2", [dep], None)
    assert result == []


async def test_region_evidence_routes_an_allowed_provider() -> None:
    """A verified-allowed provider routes; the excluded org is filtered out."""
    cb = _pinned_region_callback(_region_entry(("blocked",)))
    result = await cb.async_filter_deployments("z-ai/glm-5.2", [_pinned_dep("allowed/fp8")], None)
    assert [_provider_only(d) for d in result] == ["allowed/fp8"]


async def test_stale_region_evidence_keeps_serving_last_known_verdicts() -> None:
    """Best-effort: a stale entry keeps its last-known region verdicts rather than
    expiring, so an allowed provider still routes during a telemetry outage."""
    stale = _region_entry(("blocked",), stale=True)
    cb = _pinned_region_callback(stale)
    result = await cb.async_filter_deployments("z-ai/glm-5.2", [_pinned_dep("allowed/fp8")], None)
    assert [_provider_only(d) for d in result] == ["allowed/fp8"]


async def test_multi_slug_deployment_narrows_to_the_winner() -> None:
    """A deployment authorizing several providers must not defeat pareto narrowing:
    matching on the allowlist has to be paired with a rewrite to a singleton pin, or
    OpenRouter can still serve from a sibling that lost the value-walk."""
    entry = _entry("second/fp8", ("second/fp8",))
    rules = {
        "z-ai/glm-5.2": rule(
            precision="fp8", min_context=1_000_000, min_stats_requests=100
        )
    }
    cb = OpenRouterParetoCallback(rules=rules, telemetry=_StubTelemetry(entry))
    multi: dict[str, Any] = {
        "model_name": "z-ai/glm-5.2",
        "litellm_params": {
            "extra_body": {"provider": {"only": ["first/fp8", "second/fp8"]}}
        },
        "model_info": {"id": "or-multi"},
    }
    other: dict[str, Any] = {
        "model_name": "z-ai/glm-5.2",
        "litellm_params": {"extra_body": {"provider": {"only": ["third/fp8"]}}},
        "model_info": {"id": "or-third"},
    }
    result = await cb.async_filter_deployments("z-ai/glm-5.2", [multi, other], None)
    ids = [d["model_info"]["id"] for d in result if isinstance(d["model_info"], dict)]
    assert ids == ["or-multi"]
    assert _provider_only(result[0]) == "second/fp8"


def _region_cb(
    entry: CacheEntry | None,
    *,
    policy: UnverifiedRegionPolicy = "no_route",
    fallback: list[str] | None = None,
    cooldown: RateLimitCooldown | None = None,
) -> OpenRouterParetoCallback:
    rules = {
        "z-ai/glm-5.2": rule(
            precision="fp8",
            min_context=1_000_000,
            min_stats_requests=100,
            exclude_regions=["SG"],
            unverified_region_policy=policy,
            cold_start_fallback=fallback or [],
        )
    }
    return OpenRouterParetoCallback(
        rules=rules, telemetry=_StubTelemetry(entry), cooldown=cooldown
    )


def _verified_entry(*, stale: bool = False, winner: str | None = "allowed/fp8") -> CacheEntry:
    return CacheEntry(
        winner=winner,
        candidate_winner=winner,
        candidate_streak=1,
        safe_set=("allowed/fp8",) if winner else (),
        canonical_slug="z-ai/glm-5.2",
        canonical_slug_fetched_at=0.0,
        fetched_at=0.0,
        stale=stale,
        excluded_bases=("blocked",),
        allowed_bases=("allowed",),
    )


async def test_pinned_trust_fallback_skips_a_cooled_down_provider() -> None:
    """Pinned mode must apply the same cooldown test as wildcard mode; otherwise
    switching routing modes silently drops the 429 mitigation."""
    cd = RateLimitCooldown(threshold=1)
    cd.record("z-ai/glm-5.2", "allowed/fp8")
    cb = _region_cb(
        None, policy="trust_fallback", fallback=["allowed/fp8"], cooldown=cd
    )
    result = await cb.async_filter_deployments(
        "z-ai/glm-5.2", [_pinned_dep("allowed/fp8")], None, {}
    )
    assert result == []


async def test_pinned_trust_fallback_routes_when_provider_is_not_cooled() -> None:
    cb = _region_cb(
        None,
        policy="trust_fallback",
        fallback=["allowed/fp8"],
        cooldown=RateLimitCooldown(threshold=1),
    )
    result = await cb.async_filter_deployments(
        "z-ai/glm-5.2", [_pinned_dep("allowed/fp8")], None, {}
    )
    assert [_provider_only(d) for d in result] == ["allowed/fp8"]


async def test_verified_winner_routes_under_region_policy() -> None:
    """A verified-allowed winner routes; the excluded org never reaches the candidate set."""
    cb = _region_cb(_verified_entry())
    ok = await cb.async_filter_deployments("z-ai/glm-5.2", [_pinned_dep("allowed/fp8")], None, {})
    assert [_provider_only(d) for d in ok] == ["allowed/fp8"]



def _strict_callback(entry: CacheEntry | None) -> OpenRouterParetoCallback:
    rules = {
        "z-ai/glm-5.2": rule(
            precision="fp8",
            min_context=1_000_000,
            min_stats_requests=100,
            strict_provider=True,
        )
    }
    return OpenRouterParetoCallback(rules=rules, telemetry=_StubTelemetry(entry))


async def test_default_mode_does_not_mutate_request_kwargs() -> None:
    """Permissive default: the plugin filters/rewrites only the deployments it returns
    and never touches the shared request_kwargs, so a caller's own extra_body survives
    and cannot leak across a fallback."""
    cb, _ = _callback(_entry("baseten", ("baseten",)))
    request_kwargs: dict[str, Any] = {"extra_body": {"provider": {"only": ["deepinfra"]}}}
    before = {"provider": {"only": ["deepinfra"]}}
    await cb.async_filter_deployments("z-ai/glm-5.2", _deployments(), None, request_kwargs)
    assert request_kwargs["extra_body"] == before, "plugin mutated the caller's request kwargs"


async def test_default_mode_client_pin_reaches_wire_unchanged() -> None:
    """Documented freedom: in default mode a client's extra_body.provider wins litellm's
    merge, so the plugin does not fight it. Here we assert the plugin left it alone."""
    cb, _ = _callback(_entry("baseten", ("baseten",)))
    request_kwargs: dict[str, Any] = {
        "extra_body": {"provider": {"only": ["deepinfra"], "zdr": False}}
    }
    await cb.async_filter_deployments("z-ai/glm-5.2", _deployments(), None, request_kwargs)
    provider = request_kwargs["extra_body"]["provider"]
    assert provider == {"only": ["deepinfra"], "zdr": False}


async def test_strict_mode_raises_on_nested_extra_body_provider_only() -> None:
    """Direct router.acompletion(..., extra_body={"provider": ...}) carries the pin
    nested under extra_body."""
    cb = _strict_callback(_entry("baseten", ("baseten",)))
    request_kwargs: dict[str, Any] = {"extra_body": {"provider": {"only": ["deepinfra"]}}}
    with pytest.raises(StrictProviderConflict, match="strict_provider"):
        await cb.async_filter_deployments("z-ai/glm-5.2", _deployments(), None, request_kwargs)


async def test_strict_mode_raises_on_top_level_provider_only() -> None:
    """The primary path: through the proxy or the OpenAI SDK, OpenRouter's `provider`
    block is a TOP-LEVEL body field, splatted into the router as request_kwargs["provider"]
    (litellm only folds it under extra_body later, after deployment selection). Reading
    only the nested form let a normal proxy client bypass the strict gate entirely."""
    cb = _strict_callback(_entry("baseten", ("baseten",)))
    request_kwargs: dict[str, Any] = {"provider": {"only": ["deepinfra"]}}
    with pytest.raises(StrictProviderConflict, match="strict_provider"):
        await cb.async_filter_deployments("z-ai/glm-5.2", _deployments(), None, request_kwargs)


async def test_strict_mode_allows_request_without_client_provider_only() -> None:
    """Strict mode only rejects a client-supplied provider.only; an ordinary request is
    routed normally."""
    cb = _strict_callback(_entry("baseten", ("baseten", "novita")))
    result = await cb.async_filter_deployments("z-ai/glm-5.2", _deployments(), None, {})
    assert [_id_of(d) for d in result] == ["or-baseten"]


async def test_strict_mode_ignores_provider_without_only() -> None:
    """A client provider block that sets, say, quantizations but no `only` does not pin
    a provider, so strict mode lets it through."""
    cb = _strict_callback(_entry("baseten", ("baseten",)))
    request_kwargs: dict[str, Any] = {"extra_body": {"provider": {"quantizations": ["fp8"]}}}
    result = await cb.async_filter_deployments(
        "z-ai/glm-5.2", _deployments(), None, request_kwargs
    )
    assert [_id_of(d) for d in result] == ["or-baseten"]


async def test_strict_mode_does_not_affect_unmanaged_models() -> None:
    """Strict is per-rule; an unmanaged model is passed through untouched even with a
    client provider.only."""
    cb = _strict_callback(_entry("baseten", ("baseten",)))
    dep: dict[str, Any] = {"model_name": "gpt-4o", "litellm_params": {}, "model_info": {"id": "x"}}
    request_kwargs: dict[str, Any] = {"extra_body": {"provider": {"only": ["deepinfra"]}}}
    result = await cb.async_filter_deployments("gpt-4o", [dep], None, request_kwargs)
    assert result == [dep]
