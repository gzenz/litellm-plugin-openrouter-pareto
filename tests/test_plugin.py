from __future__ import annotations

from litellm_plugin_openrouter_pareto.config import rule
from litellm_plugin_openrouter_pareto.cooldown import RateLimitCooldown
from litellm_plugin_openrouter_pareto.plugin import OpenRouterParetoCallback, TelemetrySource
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


def _dep(dep_id: str) -> dict[str, object]:
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
) -> tuple[OpenRouterParetoCallback, _StubTelemetry]:
    rules = {"z-ai/glm-5.2": rule(precision="fp8", min_context=1_000_000, min_stats_requests=100)}
    stub = _StubTelemetry(entry)
    cb = OpenRouterParetoCallback(rules=rules, telemetry=stub, cooldown=cooldown)
    return cb, stub


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
    cd.record("baseten")
    healthy = _deployments()
    result = await cb.async_filter_deployments("z-ai/glm-5.2", healthy, None)
    assert [_id_of(d) for d in result] == ["or-novita"]


async def test_all_preferred_hot_falls_back_to_constrained_winner() -> None:
    cd = RateLimitCooldown(threshold=1)
    cb, _ = _callback(_entry("baseten", ("baseten", "novita")), cooldown=cd)
    cd.record("baseten")
    cd.record("novita")
    healthy = _deployments()
    result = await cb.async_filter_deployments("z-ai/glm-5.2", healthy, None)
    assert [_id_of(d) for d in result] == ["or-baseten"]


async def test_failure_event_records_429_until_hot() -> None:
    cd = RateLimitCooldown(threshold=3)
    cb, _ = _callback(_entry("baseten", ("baseten", "novita")), cooldown=cd)

    class _Exc:
        status_code = 429

    kwargs = {"model": "z-ai/glm-5.2", "exception": _Exc(), "metadata": {"model_info": {"id": "or-baseten"}}}
    await cb.async_log_failure_event(kwargs, None, 0.0, 0.0)
    await cb.async_log_failure_event(kwargs, None, 0.0, 0.0)
    assert cd.is_hot("baseten") is False
    await cb.async_log_failure_event(kwargs, None, 0.0, 0.0)
    assert cd.is_hot("baseten") is True


async def test_failure_event_ignores_non_429() -> None:
    cd = RateLimitCooldown(threshold=1)
    cb, _ = _callback(_entry("baseten", ("baseten",)), cooldown=cd)

    class _Exc:
        status_code = 500

    kwargs = {"model": "z-ai/glm-5.2", "exception": _Exc(), "metadata": {"model_info": {"id": "or-baseten"}}}
    await cb.async_log_failure_event(kwargs, None, 0.0, 0.0)
    assert cd.is_hot("baseten") is False


async def test_failure_event_ignores_unmanaged_model() -> None:
    cd = RateLimitCooldown(threshold=1)
    cb, _ = _callback(_entry("baseten", ("baseten",)), cooldown=cd)

    class _Exc:
        status_code = 429

    kwargs = {"model": "openai/gpt-4o", "exception": _Exc(), "metadata": {"model_info": {"id": "or-baseten"}}}
    await cb.async_log_failure_event(kwargs, None, 0.0, 0.0)
    assert cd.is_hot("baseten") is False


async def test_failure_event_uses_litellm_metadata() -> None:
    cd = RateLimitCooldown(threshold=1)
    cb, _ = _callback(_entry("baseten", ("baseten",)), cooldown=cd)

    class _Exc:
        status_code = 429

    kwargs = {"model": "z-ai/glm-5.2", "exception": _Exc(), "litellm_metadata": {"model_info": {"id": "or-novita"}}}
    await cb.async_log_failure_event(kwargs, None, 0.0, 0.0)
    assert cd.is_hot("novita") is True


async def test_failure_event_missing_exception_is_noop() -> None:
    cd = RateLimitCooldown(threshold=1)
    cb, _ = _callback(_entry("baseten", ("baseten",)), cooldown=cd)
    kwargs = {"model": "z-ai/glm-5.2", "metadata": {"model_info": {"id": "or-baseten"}}}
    await cb.async_log_failure_event(kwargs, None, 0.0, 0.0)
    assert cd.is_hot("baseten") is False


async def test_failure_event_string_status_is_ignored() -> None:
    cd = RateLimitCooldown(threshold=1)
    cb, _ = _callback(_entry("baseten", ("baseten",)), cooldown=cd)

    class _Exc:
        status_code = "429"

    kwargs = {"model": "z-ai/glm-5.2", "exception": _Exc(), "metadata": {"model_info": {"id": "or-baseten"}}}
    await cb.async_log_failure_event(kwargs, None, 0.0, 0.0)
    assert cd.is_hot("baseten") is False


async def test_empty_model_info_id_falls_back_to_extra_body() -> None:
    cd = RateLimitCooldown(threshold=1)
    cb, _ = _callback(_entry("baseten", ("baseten",)), cooldown=cd)

    class _Exc:
        status_code = 429

    kwargs = {
        "model": "z-ai/glm-5.2",
        "exception": _Exc(),
        "metadata": {"model_info": {"id": ""}},
        "litellm_params": {"extra_body": {"provider": {"only": ["novita"]}}},
    }
    await cb.async_log_failure_event(kwargs, None, 0.0, 0.0)
    assert cd.is_hot("novita") is True
    assert cd.is_hot("baseten") is False


async def test_or_dash_only_id_falls_back_to_extra_body() -> None:
    cd = RateLimitCooldown(threshold=1)
    cb, _ = _callback(_entry("baseten", ("baseten",)), cooldown=cd)

    class _Exc:
        status_code = 429

    kwargs = {
        "model": "z-ai/glm-5.2",
        "exception": _Exc(),
        "metadata": {"model_info": {"id": "or-"}},
        "litellm_params": {"extra_body": {"provider": {"only": ["novita"]}}},
    }
    await cb.async_log_failure_event(kwargs, None, 0.0, 0.0)
    assert cd.is_hot("novita") is True


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


def test_slug_of_prefers_model_info_id_over_extra_body() -> None:
    cb, _ = _callback(None)
    dep: dict[str, object] = {
        "model_name": "z-ai/glm-5.2",
        "litellm_params": {"model": "openrouter/z-ai/glm-5.2", "extra_body": {"provider": {"only": ["novita"]}}},
        "model_info": {"id": "some-other-id"},
    }
    assert cb.slug_of(dep) == "some-other-id"


def test_slug_of_falls_back_to_extra_body_when_no_model_info_id() -> None:
    cb, _ = _callback(None)
    dep: dict[str, object] = {
        "model_name": "z-ai/glm-5.2",
        "litellm_params": {"model": "openrouter/z-ai/glm-5.2", "extra_body": {"provider": {"only": ["novita"]}}},
        "model_info": {},
    }
    assert cb.slug_of(dep) == "novita"


def test_telemetry_is_a_telemetry_source() -> None:
    from litellm_plugin_openrouter_pareto.telemetry import Telemetry

    rules = {"z-ai/glm-5.2": rule(precision="fp8", min_context=1_000_000, min_stats_requests=100)}
    t = Telemetry(rules, cache_dir="/tmp/or-pareto-test-src")
    assert isinstance(t, TelemetrySource)
