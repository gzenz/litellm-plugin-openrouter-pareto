from __future__ import annotations

from typing import Any

import pytest

from litellm_plugin_openrouter_pareto.config import UnverifiedRegionPolicy, rule
from litellm_plugin_openrouter_pareto.cooldown import RateLimitCooldown
from litellm_plugin_openrouter_pareto.plugin import (
    AllProvidersOnCooldown,
    OpenRouterParetoCallback,
    StrictProviderConflict,
    TelemetrySource,
)
from litellm_plugin_openrouter_pareto.scorer import Point
from litellm_plugin_openrouter_pareto.telemetry import CacheEntry


def _entry(
    winner: str | None,
    safe_set: tuple[str, ...],
    points: tuple[Point, ...] | None = None,
) -> CacheEntry:
    """A warm entry. `points` defaults to a set where the winner dominates (cheapest
    AND fastest), so the re-run walk ranks it first and the rest cheapest-first -
    the legacy behavior; tests that exercise the re-walk pass real points."""
    return CacheEntry(
        winner=winner,
        candidate_winner=winner,
        candidate_streak=1,
        safe_set=safe_set,
        points=points
        if points is not None
        else tuple(
            Point(
                slug=s,
                input_price_m=1.0 if s == winner else float(i + 2),
                tps=100.0 if s == winner else 1.0,
            )
            for i, s in enumerate(safe_set)
        ),
        canonical_slug="z-ai/glm-5.2",
        canonical_slug_fetched_at=0.0,
        fetched_at=0.0,
        stale=False,
    )


class _StubTelemetry:
    def __init__(self, entry: CacheEntry | None) -> None:
        self.entry = entry
        self.calls: list[str] = []
        self._allowed_providers: frozenset[str] | None = None

    async def get(self, model: str) -> CacheEntry | None:
        self.calls.append(model)
        return self.entry

    def get_allowed_providers(self) -> frozenset[str] | None:
        return self._allowed_providers

    def set_allowed_providers(self, slugs: frozenset[str]) -> None:
        self._allowed_providers = slugs


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
    allowed_providers: frozenset[str] | None = None,
    tolerance: float = 0.0,
    exclude_providers: tuple[str, ...] = (),
    input_cap_patterns: tuple[str, ...] | None = None,
    broken_provider_patterns: tuple[str, ...] = (),
    broken_provider_ttl_s: float | None = None,
) -> tuple[OpenRouterParetoCallback, _StubTelemetry]:
    rule_kwargs: dict[str, Any] = {
        "precision": "fp8",
        "min_context": 1_000_000,
        "min_stats_requests": 100,
        "wildcard": wildcard,
        "cold_start_fallback": cold_start_fallback if cold_start_fallback is not None else (),
        "value_regression_tolerance": tolerance,
        "exclude_providers": exclude_providers,
        "input_cap_patterns": input_cap_patterns,
        "broken_provider_patterns": broken_provider_patterns,
    }
    if broken_provider_ttl_s is not None:
        rule_kwargs["broken_provider_ttl_s"] = broken_provider_ttl_s
    rules = {"z-ai/glm-5.2": rule(**rule_kwargs)}
    stub = _StubTelemetry(entry)
    if allowed_providers is not None:
        stub._allowed_providers = allowed_providers
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


def _post_call_request_data(
    slug: str | None,
    model: str = "z-ai/glm-5.2",
    call_type: str = "pass_through_endpoint",
    *,
    slug_site: str = "provider",
    only: list[str] | None = None,
    response_body: dict[str, Any] | str | None = None,
) -> dict[str, Any]:
    """request_data handed to async_post_call_failure_hook: the merged request body +
    litellm kwargs, with call_type=pass_through_endpoint. slug_site picks where the
    provider pin lands so the extractor is exercised across landing sites (top-level
    OpenRouter `provider` body field, top-level `extra_body`, or
    `optional_params.extra_body` -- the operator-reported site). `only` overrides the
    top-level provider.only list (to exercise ambiguous/empty allowlists)."""
    body: dict[str, Any] = {"model": model, "call_type": call_type}
    if response_body is not None:
        body["response_body"] = response_body
    if slug is None and only is None:
        return body
    if slug_site == "provider":
        body["provider"] = {"only": only if only is not None else [slug]}
    elif slug_site == "extra_body":
        body["extra_body"] = {"provider": {"only": only if only is not None else [slug]}}
    elif slug_site == "optional_params":
        body["optional_params"] = {"extra_body": {"provider": {"only": only if only is not None else [slug]}}}
    return body


class _SyntheticUpstreamExc:
    """Mimics litellm's passthrough upstream-failure HTTPException: a generic
    detail that carries only the status, NOT the upstream error text (which lives
    in request_data["response_body"])."""

    def __init__(self, status_code: int) -> None:
        self.status_code = status_code

    def __str__(self) -> str:
        return f"{self.status_code}: Upstream passthrough request failed with status {self.status_code}"


def _deployments() -> list[dict[str, object]]:
    return [_dep("or-baseten"), _dep("or-novita"), _dep("or-siliconflow")]


async def test_narrows_to_winner_deployment() -> None:
    cb, _ = _callback(_entry("baseten", ("baseten", "novita", "siliconflow")))
    result = await cb.async_filter_deployments("z-ai/glm-5.2", _deployments(), None)
    assert [_id_of(d) for d in result] == ["or-baseten"]


@pytest.mark.parametrize(
    "sent",
    [
        "openrouter/z-ai/glm-5.2",
        "z-ai/glm-5.2[1m]",
        "openrouter/z-ai/glm-5.2[1m]",
    ],
)
async def test_prefixed_or_tagged_model_matches_bare_rule(sent: str) -> None:
    cb, stub = _callback(_entry("baseten", ("baseten", "novita", "siliconflow")))
    result = await cb.async_filter_deployments(sent, _deployments(), None)
    assert [_id_of(d) for d in result] == ["or-baseten"]
    assert stub.calls == ["z-ai/glm-5.2"]  # telemetry keyed on the bare id


async def test_failure_event_records_cooldown_under_bare_key() -> None:
    # A failure arriving as the openrouter/-prefixed [1m] form must record the
    # cooldown under the bare key the routing path checks, or the hot winner is
    # never skipped.
    cd = RateLimitCooldown(threshold=1)
    cb, _ = _callback(_entry("baseten", ("baseten",)), cooldown=cd)

    class _Exc:
        status_code = 429

    kwargs = _failure_kwargs("baseten/fp8", _Exc(), model="openrouter/z-ai/glm-5.2[1m]")
    await cb.async_log_failure_event(kwargs, None, 0.0, 0.0)
    assert cd.is_hot("z-ai/glm-5.2", "baseten/fp8") is True


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


async def test_post_call_failure_records_429_on_passthrough() -> None:
    # The only failure signal litellm fires for a raw passthrough request is
    # async_post_call_failure_hook; without it, the 429 never reaches the cooldown.
    # Real litellm shape: a synthetic HTTPException (generic detail) + the parsed
    # upstream body in response_body. 429 detection is by status_code.
    cd = RateLimitCooldown(threshold=1)
    cb, _ = _callback(_entry("baseten", ("baseten",)), cooldown=cd)

    request_data = _post_call_request_data(
        "baseten/fp8",
        response_body={"error": {"message": "Rate limit exceeded"}},
    )
    await cb.async_post_call_failure_hook(request_data, _SyntheticUpstreamExc(429), None, None)
    assert cd.is_hot("z-ai/glm-5.2", "baseten/fp8") is True


async def test_post_call_failure_records_input_cap_from_response_body() -> None:
    # The passthrough upstream-failure path supplies only a generic HTTPException
    # detail; the input-length text lives in request_data["response_body"]. The
    # hook must read it there or the input-cap is never recorded.
    cd = RateLimitCooldown(threshold=10)
    cb, _ = _callback(_entry("baseten/fp8", ("baseten/fp8",)), cooldown=cd)

    request_data = _post_call_request_data(
        "baseten/fp8",
        response_body={
            "error": {"message": "Input length 562514 exceeds the maximum allowed input length of 524256 tokens"}
        },
    )
    await cb.async_post_call_failure_hook(request_data, _SyntheticUpstreamExc(400), None, None)
    assert cd.is_input_capped("z-ai/glm-5.2", "baseten/fp8") is True
    assert cd.is_hot("z-ai/glm-5.2", "baseten/fp8") is False


async def test_post_call_failure_input_cap_missing_body_not_recorded() -> None:
    # No response_body and a generic synthetic exception -> the input-cap regex
    # has nothing to match, so nothing is recorded (no false positive from the
    # bare "failed with status 400" detail).
    cd = RateLimitCooldown(threshold=10)
    cb, _ = _callback(_entry("baseten/fp8", ("baseten/fp8",)), cooldown=cd)

    request_data = _post_call_request_data("baseten/fp8")
    await cb.async_post_call_failure_hook(request_data, _SyntheticUpstreamExc(400), None, None)
    assert cd.is_input_capped("z-ai/glm-5.2", "baseten/fp8") is False


@pytest.mark.parametrize("call_type", ["acompletion", "anthropic_messages", ""])
async def test_post_call_failure_ignores_non_passthrough(call_type: str) -> None:
    # On non-passthrough paths async_log_failure_event already records; recording
    # here too would double-count and trip the 3-hit cooldown early. The call_type
    # guard keeps this hook a strict no-op outside the passthrough path.
    cd = RateLimitCooldown(threshold=1)
    cb, _ = _callback(_entry("baseten", ("baseten",)), cooldown=cd)

    class _Exc:
        status_code = 429

    request_data = _post_call_request_data("baseten/fp8", call_type=call_type)
    await cb.async_post_call_failure_hook(request_data, _Exc(), None, None)
    assert cd.is_hot("z-ai/glm-5.2", "baseten/fp8") is False


async def test_post_call_failure_ignores_unmanaged_model() -> None:
    cd = RateLimitCooldown(threshold=1)
    cb, _ = _callback(_entry("baseten", ("baseten",)), cooldown=cd)

    class _Exc:
        status_code = 429

    request_data = _post_call_request_data("baseten/fp8", model="openai/gpt-4o")
    await cb.async_post_call_failure_hook(request_data, _Exc(), None, None)
    assert cd.is_hot("openai/gpt-4o", "baseten/fp8") is False


async def test_post_call_failure_ignores_non_429() -> None:
    cd = RateLimitCooldown(threshold=1)
    cb, _ = _callback(_entry("baseten", ("baseten",)), cooldown=cd)

    class _Exc:
        status_code = 500

    request_data = _post_call_request_data("baseten/fp8")
    await cb.async_post_call_failure_hook(request_data, _Exc(), None, None)
    assert cd.is_hot("z-ai/glm-5.2", "baseten/fp8") is False


async def test_post_call_failure_ignores_missing_model() -> None:
    cd = RateLimitCooldown(threshold=1)
    cb, _ = _callback(_entry("baseten", ("baseten",)), cooldown=cd)

    class _Exc:
        status_code = 429

    request_data: dict[str, Any] = {"call_type": "pass_through_endpoint"}
    await cb.async_post_call_failure_hook(request_data, _Exc(), None, None)
    assert cd.is_hot("z-ai/glm-5.2", "baseten/fp8") is False


@pytest.mark.parametrize("slug_site", ["provider", "extra_body", "optional_params"])
async def test_post_call_failure_slug_resolves_across_landing_sites(slug_site: str) -> None:
    # The passthrough request_data is the merged body + kwargs, not
    # model_call_details; the pin can land at several sites depending on how the
    # request arrived. The extractor must find it at any of them.
    cd = RateLimitCooldown(threshold=1)
    cb, _ = _callback(_entry("baseten", ("baseten",)), cooldown=cd)

    class _Exc:
        status_code = 429

    request_data = _post_call_request_data("baseten/fp8", slug_site=slug_site)
    await cb.async_post_call_failure_hook(request_data, _Exc(), None, None)
    assert cd.is_hot("z-ai/glm-5.2", "baseten/fp8") is True


async def test_post_call_failure_normalizes_prefixed_tagged_model() -> None:
    # Claude Code sends the openrouter/-prefixed [1m] form; the cooldown must be
    # recorded under the bare rule key the routing path checks.
    cd = RateLimitCooldown(threshold=1)
    cb, _ = _callback(_entry("baseten", ("baseten",)), cooldown=cd)

    class _Exc:
        status_code = 429

    request_data = _post_call_request_data("baseten/fp8", model="openrouter/z-ai/glm-5.2[1m]")
    await cb.async_post_call_failure_hook(request_data, _Exc(), None, None)
    assert cd.is_hot("z-ai/glm-5.2", "baseten/fp8") is True


async def test_post_call_failure_no_slug_is_safe_noop(capsys: pytest.CaptureFixture[str]) -> None:
    # A managed passthrough 429 with no resolvable slug must not raise into the
    # request path and must not record under a wrong key; it warns once instead.
    cd = RateLimitCooldown(threshold=1)
    cb, _ = _callback(_entry("baseten", ("baseten",)), cooldown=cd)

    class _Exc:
        status_code = 429

    request_data = _post_call_request_data(None)
    await cb.async_post_call_failure_hook(request_data, _Exc(), None, None)
    assert cd.is_hot("z-ai/glm-5.2", "") is False
    assert "no resolvable provider slug" in capsys.readouterr().err


async def test_post_call_failure_no_double_record_with_log_failure_event() -> None:
    # Documents the guard's purpose: when both hooks would see the same acompletion
    # failure, only async_log_failure_event records. Simulate that by firing the
    # post-call hook on a non-passthrough call_type and asserting zero hits,
    # while the log-failure path (same exc/slug) still records once.
    cd = RateLimitCooldown(threshold=2)
    cb, _ = _callback(_entry("baseten", ("baseten",)), cooldown=cd)

    class _Exc:
        status_code = 429

    log_kwargs = _failure_kwargs("baseten/fp8", _Exc())
    post_data = _post_call_request_data("baseten/fp8", call_type="acompletion")

    await cb.async_post_call_failure_hook(post_data, _Exc(), None, None)  # guarded -> no record
    assert cd.is_hot("z-ai/glm-5.2", "baseten/fp8") is False
    await cb.async_log_failure_event(log_kwargs, None, 0.0, 0.0)  # the path that owns recording
    assert cd.is_hot("z-ai/glm-5.2", "baseten/fp8") is False
    await cb.async_log_failure_event(log_kwargs, None, 0.0, 0.0)
    assert cd.is_hot("z-ai/glm-5.2", "baseten/fp8") is True


@pytest.mark.parametrize(
    "only_value",
    [
        ["baseten/fp8", "novita/fp8"],  # multiple providers
        [],  # empty allowlist
        "baseten/fp8",  # not a list at all
        [""],  # empty slug string
        None,  # present but null -- must NOT fall through (key present, not absent)
    ],
    ids=["multi", "empty", "non-list", "empty-slug", "null"],
)
async def test_post_call_failure_ambiguous_top_level_provider_only_returns_none(
    only_value: list[str] | str | None,
) -> None:
    # A present-but-ambiguous top-level provider.only cannot be attributed to one
    # provider. It must NOT fall through to a secondary copy (here a singleton under
    # optional_params) -- that would cool down the wrong provider. A present null is
    # treated as malformed (the key exists), not as absent. Empty slugs are rejected
    # too, consistently with the deployment slug extractor.
    cd = RateLimitCooldown(threshold=1)
    cb, _ = _callback(_entry("baseten", ("baseten",)), cooldown=cd)

    class _Exc:
        status_code = 429

    request_data: dict[str, Any] = {
        "model": "z-ai/glm-5.2",
        "call_type": "pass_through_endpoint",
        "provider": {"only": only_value},
        "optional_params": {"extra_body": {"provider": {"only": ["baseten/fp8"]}}},
    }
    await cb.async_post_call_failure_hook(request_data, _Exc(), None, None)
    assert cd.is_hot("z-ai/glm-5.2", "baseten/fp8") is False
    assert cd.is_hot("z-ai/glm-5.2", "novita/fp8") is False


async def test_post_call_failure_absent_only_falls_through_to_other_sites() -> None:
    # A top-level provider block without `only` (e.g. only quantizations) did not
    # pin, so the extractor may still resolve the pin from a secondary site.
    cd = RateLimitCooldown(threshold=1)
    cb, _ = _callback(_entry("baseten", ("baseten",)), cooldown=cd)

    class _Exc:
        status_code = 429

    request_data: dict[str, Any] = {
        "model": "z-ai/glm-5.2",
        "call_type": "pass_through_endpoint",
        "provider": {"quantizations": ["fp8"]},
        "optional_params": {"extra_body": {"provider": {"only": ["baseten/fp8"]}}},
    }
    await cb.async_post_call_failure_hook(request_data, _Exc(), None, None)
    assert cd.is_hot("z-ai/glm-5.2", "baseten/fp8") is True


async def test_post_call_failure_passthrough_shape_records_under_pinned_slug() -> None:
    # Shape test (not a live-contract test): build request_data the way litellm
    # does for a non-streaming passthrough upstream 429 -- the parsed request body
    # + the passthrough kwargs (call_type=pass_through_endpoint) + response_body
    # (parsed upstream JSON) -- and a synthetic HTTPException whose detail carries
    # only the status. Proves the hook records under the pinned slug from the
    # merged shape. Does NOT prove litellm still emits this shape; the streaming
    # path omits response_body (see test_post_call_failure_streaming_input_cap_undetected).
    cd = RateLimitCooldown(threshold=1)
    cb, _ = _callback(_entry("baseten", ("baseten",)), cooldown=cd)

    parsed_body: dict[str, Any] = {
        "model": "openrouter/z-ai/glm-5.2[1m]",
        "messages": [{"role": "user", "content": "hi"}],
        "provider": {"only": ["baseten/fp8"]},
    }
    passthrough_kwargs: dict[str, Any] = {
        "call_type": "pass_through_endpoint",
        "litellm_call_id": "test-call-id",
        "litellm_params": {"metadata": {}},
    }
    request_data: dict[str, Any] = {**parsed_body, **passthrough_kwargs}
    request_data["response_body"] = {"error": {"message": "Too many requests"}}

    await cb.async_post_call_failure_hook(request_data, _SyntheticUpstreamExc(429), None, None)
    assert cd.is_hot("z-ai/glm-5.2", "baseten/fp8") is True


async def test_post_call_failure_streaming_429_still_recorded() -> None:
    # Streaming passthrough omits response_body, but 429 detection is by status
    # code, so the cooldown still records for the primary (rate-limit) use case.
    cd = RateLimitCooldown(threshold=1)
    cb, _ = _callback(_entry("baseten", ("baseten",)), cooldown=cd)

    request_data = _post_call_request_data("baseten/fp8")  # no response_body, as on the streaming path
    await cb.async_post_call_failure_hook(request_data, _SyntheticUpstreamExc(429), None, None)
    assert cd.is_hot("z-ai/glm-5.2", "baseten/fp8") is True


async def test_post_call_failure_streaming_input_cap_undetected() -> None:
    # KNOWN LIMITATION (documented on async_post_call_failure_hook): litellm's
    # streaming passthrough failure path dispatches this hook WITHOUT
    # response_body (reading the stream body would consume it and break relay).
    # The input-cap text therefore cannot be read, so a 400 input-cap on a
    # streaming passthrough is NOT recorded. 429 (above) still works. This test
    # pins the limitation so a regression here is a deliberate decision, not an
    # accident.
    cd = RateLimitCooldown(threshold=10)
    cb, _ = _callback(_entry("baseten/fp8", ("baseten/fp8",)), cooldown=cd)

    request_data = _post_call_request_data("baseten/fp8")  # no response_body
    await cb.async_post_call_failure_hook(request_data, _SyntheticUpstreamExc(400), None, None)
    assert cd.is_input_capped("z-ai/glm-5.2", "baseten/fp8") is False


@pytest.mark.parametrize(
    "response_body",
    [
        {"error": {"message": "exceeds the maximum allowed input length"}},
        {"error": "exceeds the maximum allowed input length"},
        {"message": "maximum context length exceeded"},
        "exceeds the maximum allowed input length",
    ],
    ids=["nested-error", "string-error", "top-message", "string-body"],
)
async def test_upstream_message_extracts_input_cap_across_shapes(
    response_body: dict[str, Any] | str,
) -> None:
    # _upstream_message must surface the provider error text from each shape
    # OpenRouter/litellm might produce, so the built-in input-cap patterns match it.
    cd = RateLimitCooldown(threshold=10)
    cb, _ = _callback(_entry("baseten/fp8", ("baseten/fp8",)), cooldown=cd)

    request_data = _post_call_request_data("baseten/fp8", response_body=response_body)
    await cb.async_post_call_failure_hook(request_data, _SyntheticUpstreamExc(400), None, None)
    assert cd.is_input_capped("z-ai/glm-5.2", "baseten/fp8") is True


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

        def get_allowed_providers(self) -> frozenset[str] | None:
            return None

        def set_allowed_providers(self, slugs: frozenset[str]) -> None:
            pass

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


async def test_wildcard_all_hot_raises_all_providers_on_cooldown() -> None:
    cd = RateLimitCooldown(threshold=1)
    cb, _ = _callback(_entry("baseten", ("baseten", "novita")), cooldown=cd, wildcard=True)
    cd.record("z-ai/glm-5.2", "baseten")
    cd.record("z-ai/glm-5.2", "novita")
    deps = [_wildcard_dep()]
    with pytest.raises(AllProvidersOnCooldown, match="429 cooldown"):
        await cb.async_filter_deployments("z-ai/glm-5.2", deps, None)


async def test_wildcard_stale_uses_cold_start_fallback() -> None:
    cb, _ = _callback(None, wildcard=True, cold_start_fallback=("novita",))
    deps = [_wildcard_dep()]
    result = await cb.async_filter_deployments("z-ai/glm-5.2", deps, None)
    assert _provider_only(result[0]) == "novita"


async def test_wildcard_stale_no_fallback_declines_no_stale_pin() -> None:
    """Cold start with no fallback configured and no region policy: decline to
    route rather than cede to OpenRouter (which would bypass the precision /
    context filters). No stale pin leaks because nothing is returned."""
    cb, _ = _callback(None, wildcard=True)
    deps = [_wildcard_dep()]
    deps[0]["litellm_params"]["extra_body"] = {"provider": {"only": ["stale_pin"]}}
    result = await cb.async_filter_deployments("z-ai/glm-5.2", deps, None)
    assert result == []
    assert _provider_only(deps[0]) == "stale_pin"


async def test_wildcard_exception_uses_cold_start_fallback() -> None:
    class _Boom:
        async def get(self, model: str) -> CacheEntry | None:
            raise RuntimeError("boom")

        def get_allowed_providers(self) -> frozenset[str] | None:
            return None

        def set_allowed_providers(self, slugs: frozenset[str]) -> None:
            pass

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

    rules_on = {"z-ai/glm-5.2": rule(precision="fp8", min_context=1_000_000, min_stats_requests=100, log_errors=True)}
    rules_off = {"z-ai/glm-5.2": rule(precision="fp8", min_context=1_000_000, min_stats_requests=100, log_errors=False)}

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


async def test_wildcard_fallback_all_hot_raises_all_providers_on_cooldown() -> None:
    """All fallbacks 429-hot -> raise AllProvidersOnCooldown (not strip/cede)."""
    cd = RateLimitCooldown(threshold=1)
    cb, _ = _callback(None, wildcard=True, cold_start_fallback=("a/fp8", "b/fp8"), cooldown=cd)
    cd.record("z-ai/glm-5.2", "a/fp8")
    cd.record("z-ai/glm-5.2", "b/fp8")
    deps = [_wildcard_dep()]
    with pytest.raises(AllProvidersOnCooldown, match="429 cooldown"):
        await cb.async_filter_deployments("z-ai/glm-5.2", deps, None)


async def test_wildcard_fallback_first_capped_second_hot_pins_capped() -> None:
    """First fallback input-capped (not 429-hot), second 429-hot: the capped one
    is non-hot, so it is pinned (better to try than to fail - a small request may
    fit under the cap). Input cap is a soft preference, not a skip."""
    cd = RateLimitCooldown(threshold=1)
    cb, _ = _callback(None, wildcard=True, cold_start_fallback=("a/fp8", "b/fp8"), cooldown=cd)
    cd.record_input_cap("z-ai/glm-5.2", "a/fp8")
    cd.record("z-ai/glm-5.2", "b/fp8")
    deps = [_wildcard_dep()]
    result = await cb.async_filter_deployments("z-ai/glm-5.2", deps, None)
    assert _provider_only(result[0]) == "a/fp8"


async def test_wildcard_fallback_capped_then_clean_pins_clean() -> None:
    """Input cap is a soft preference matched to the warm path: a non-capped
    fallback is preferred over an earlier capped one, so the clean second
    fallback wins (not the capped first)."""
    cd = RateLimitCooldown(threshold=1)
    cb, _ = _callback(None, wildcard=True, cold_start_fallback=("a/fp8", "b/fp8"), cooldown=cd)
    cd.record_input_cap("z-ai/glm-5.2", "a/fp8")
    deps = [_wildcard_dep()]
    result = await cb.async_filter_deployments("z-ai/glm-5.2", deps, None)
    assert _provider_only(result[0]) == "b/fp8"


async def test_wildcard_fallback_all_capped_pins_better_to_try() -> None:
    """All fallbacks input-capped (none 429-hot): pin the first non-hot fallback
    rather than decline - a small request may still fit under the cap. Input cap
    is a soft preference; only all-429-hot is a hard stop. zdr / no-fallback
    posture is still forced by the pin."""
    cd = RateLimitCooldown(threshold=1)
    cb, _ = _callback(None, wildcard=True, cold_start_fallback=("a/fp8",), cooldown=cd)
    cd.record_input_cap("z-ai/glm-5.2", "a/fp8")
    deps = [_wildcard_dep()]
    result = await cb.async_filter_deployments("z-ai/glm-5.2", deps, None)
    assert _provider_only(result[0]) == "a/fp8"
    lp = result[0]["litellm_params"]
    assert isinstance(lp, dict)
    eb = lp.get("extra_body")
    assert isinstance(eb, dict)
    provider = eb.get("provider")
    assert isinstance(provider, dict)
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


def _decision_lines(log_path) -> list[str]:
    assert log_path.exists(), "routing log was not written"
    return [line for line in log_path.read_text().splitlines() if line]


def _decision_callback(
    entry: CacheEntry | None,
    *,
    log_decisions: bool = True,
    cooldown: RateLimitCooldown | None = None,
    wildcard: bool = True,
    cold_start_fallback: tuple[str, ...] = (),
    exclude_regions: tuple[str, ...] = (),
    allow_unknown_region: bool = False,
    unverified_region_policy: UnverifiedRegionPolicy = "no_route",
) -> tuple[OpenRouterParetoCallback, _StubTelemetry]:
    rules = {
        "z-ai/glm-5.2": rule(
            precision="fp8",
            min_context=1_000_000,
            min_stats_requests=100,
            wildcard=wildcard,
            cold_start_fallback=list(cold_start_fallback),
            log_decisions=log_decisions,
            exclude_regions=list(exclude_regions),
            allow_unknown_region=allow_unknown_region,
            unverified_region_policy=unverified_region_policy,
        )
    }
    stub = _StubTelemetry(entry)
    return OpenRouterParetoCallback(rules=rules, telemetry=stub, cooldown=cooldown), stub


async def test_log_decisions_warm_wildcard_records_winner(tmp_path, monkeypatch) -> None:
    log_path = tmp_path / "routing.log"
    monkeypatch.setenv("OPENROUTER_PARETO_ROUTING_LOG", str(log_path))
    cb, _ = _decision_callback(_entry("baseten/fp8", ("baseten/fp8", "novita/fp8")))
    result = await cb.async_filter_deployments("z-ai/glm-5.2", [_wildcard_dep()], None)
    assert _provider_only(result[0]) == "baseten/fp8"
    (line,) = _decision_lines(log_path)
    assert "model=z-ai/glm-5.2" in line
    assert "mode=wildcard" in line
    assert "slug=baseten/fp8" in line
    assert "decision=winner" in line
    assert "region=-" in line


async def test_log_decisions_warm_winner_on_cooldown_records_safe_set(tmp_path, monkeypatch) -> None:
    """A cooldown on the winner means the next provider was walked into: safe_set."""
    log_path = tmp_path / "routing.log"
    monkeypatch.setenv("OPENROUTER_PARETO_ROUTING_LOG", str(log_path))
    cd = RateLimitCooldown(threshold=1)
    cb, _ = _decision_callback(_entry("baseten/fp8", ("baseten/fp8", "novita/fp8")), cooldown=cd)
    cd.record("z-ai/glm-5.2", "baseten/fp8")
    result = await cb.async_filter_deployments("z-ai/glm-5.2", [_wildcard_dep()], None)
    assert _provider_only(result[0]) == "novita/fp8"
    (line,) = _decision_lines(log_path)
    assert "slug=novita/fp8" in line
    assert "decision=safe_set" in line


async def test_log_decisions_all_candidates_input_capped_records_input_cap(tmp_path, monkeypatch) -> None:
    log_path = tmp_path / "routing.log"
    monkeypatch.setenv("OPENROUTER_PARETO_ROUTING_LOG", str(log_path))
    cd = RateLimitCooldown(threshold=10)
    cb, _ = _decision_callback(_entry("baseten/fp8", ("baseten/fp8", "novita/fp8")), cooldown=cd)
    cd.record_input_cap("z-ai/glm-5.2", "baseten/fp8")
    cd.record_input_cap("z-ai/glm-5.2", "novita/fp8")
    result = await cb.async_filter_deployments("z-ai/glm-5.2", [_wildcard_dep()], None)
    assert _provider_only(result[0]) == "baseten/fp8"
    (line,) = _decision_lines(log_path)
    assert "slug=baseten/fp8" in line
    assert "decision=input_cap" in line


async def test_log_decisions_cold_start_records_fallback(tmp_path, monkeypatch) -> None:
    log_path = tmp_path / "routing.log"
    monkeypatch.setenv("OPENROUTER_PARETO_ROUTING_LOG", str(log_path))
    cb, _ = _decision_callback(None, cold_start_fallback=("novita/fp8",))
    result = await cb.async_filter_deployments("z-ai/glm-5.2", [_wildcard_dep()], None)
    assert _provider_only(result[0]) == "novita/fp8"
    (line,) = _decision_lines(log_path)
    assert "slug=novita/fp8" in line
    assert "decision=cold_start" in line


async def test_log_decisions_all_hot_written_before_the_raise(tmp_path, monkeypatch) -> None:
    """No request ever reaches a success hook on this path, so the line has to be
    written before AllProvidersOnCooldown is raised or the decision is invisible."""
    log_path = tmp_path / "routing.log"
    monkeypatch.setenv("OPENROUTER_PARETO_ROUTING_LOG", str(log_path))
    cd = RateLimitCooldown(threshold=1)
    cb, _ = _decision_callback(_entry("baseten/fp8", ("baseten/fp8",)), cooldown=cd)
    cd.record("z-ai/glm-5.2", "baseten/fp8")
    with pytest.raises(AllProvidersOnCooldown):
        await cb.async_filter_deployments("z-ai/glm-5.2", [_wildcard_dep()], None)
    (line,) = _decision_lines(log_path)
    assert "decision=all_hot" in line
    assert "slug=-" in line


async def test_log_decisions_wildcard_fallback_all_hot_records_all_hot(tmp_path, monkeypatch) -> None:
    log_path = tmp_path / "routing.log"
    monkeypatch.setenv("OPENROUTER_PARETO_ROUTING_LOG", str(log_path))
    cd = RateLimitCooldown(threshold=1)
    cb, _ = _decision_callback(None, cold_start_fallback=("a/fp8",), cooldown=cd)
    cd.record("z-ai/glm-5.2", "a/fp8")
    with pytest.raises(AllProvidersOnCooldown):
        await cb.async_filter_deployments("z-ai/glm-5.2", [_wildcard_dep()], None)
    (line,) = _decision_lines(log_path)
    assert "decision=all_hot" in line


async def test_log_decisions_region_excluded_records_declined_with_verdict(tmp_path, monkeypatch) -> None:
    """A region policy that refuses to route says so, and says the region was why."""
    log_path = tmp_path / "routing.log"
    monkeypatch.setenv("OPENROUTER_PARETO_ROUTING_LOG", str(log_path))
    entry = _entry_with_excluded(("baseten", "novita"), stale=True)
    cb, _ = _decision_callback(
        entry,
        cold_start_fallback=("baseten/fp8", "novita/fp8"),
        exclude_regions=("US",),
    )
    result = await cb.async_filter_deployments("z-ai/glm-5.2", [_wildcard_dep()], None)
    assert result == []
    (line,) = _decision_lines(log_path)
    assert "decision=declined" in line
    assert "region=excluded" in line


async def test_log_decisions_unknown_region_records_verdict(tmp_path, monkeypatch) -> None:
    """No telemetry and no trust_fallback: the fallback is unverifiable, so the
    verdict is `unknown` rather than a claim the slug passed the policy."""
    log_path = tmp_path / "routing.log"
    monkeypatch.setenv("OPENROUTER_PARETO_ROUTING_LOG", str(log_path))
    cb, _ = _decision_callback(None, cold_start_fallback=("novita/fp8",), exclude_regions=("US",))
    result = await cb.async_filter_deployments("z-ai/glm-5.2", [_wildcard_dep()], None)
    assert result == []
    (line,) = _decision_lines(log_path)
    assert "decision=declined" in line
    assert "region=unknown" in line


async def test_log_decisions_region_allowed_records_verdict(tmp_path, monkeypatch) -> None:
    """Telemetry has no region for this slug but allow_unknown_region admits it: a
    real verdict was reached, so it is recorded rather than left as `-`."""
    log_path = tmp_path / "routing.log"
    monkeypatch.setenv("OPENROUTER_PARETO_ROUTING_LOG", str(log_path))
    entry = _entry_with_excluded(("baseten",), stale=True, allowed=("novita",))
    cb, _ = _decision_callback(
        entry,
        cold_start_fallback=("novita/fp8",),
        exclude_regions=("US",),
        allow_unknown_region=True,
    )
    result = await cb.async_filter_deployments("z-ai/glm-5.2", [_wildcard_dep()], None)
    assert _provider_only(result[0]) == "novita/fp8"
    (line,) = _decision_lines(log_path)
    assert "decision=cold_start" in line
    assert "region=allowed" in line


async def test_log_decisions_region_policy_warn_once_still_fires(tmp_path, monkeypatch) -> None:
    """Recording the verdict must not consume the warn-once key for that slug, or the
    operator's stderr warning would disappear on a logging-enabled rule."""
    log_path = tmp_path / "routing.log"
    monkeypatch.setenv("OPENROUTER_PARETO_ROUTING_LOG", str(log_path))
    cb, _ = _decision_callback(None, cold_start_fallback=("novita/fp8",), exclude_regions=("US",))
    await cb.async_filter_deployments("z-ai/glm-5.2", [_wildcard_dep()], None)
    assert "region-unverified:z-ai/glm-5.2:novita/fp8" in cb._warned


async def test_log_decisions_unpinned_policy_records_unpinned(tmp_path, monkeypatch) -> None:
    log_path = tmp_path / "routing.log"
    monkeypatch.setenv("OPENROUTER_PARETO_ROUTING_LOG", str(log_path))
    cb, _ = _decision_callback(
        None,
        cold_start_fallback=("novita/fp8",),
        exclude_regions=("US",),
        unverified_region_policy="unpinned",
    )
    result = await cb.async_filter_deployments("z-ai/glm-5.2", [_wildcard_dep()], None)
    assert _provider_only(result[0]) is None
    (line,) = _decision_lines(log_path)
    assert "decision=unpinned" in line
    assert "slug=-" in line


async def test_log_decisions_pinned_warm_winner_records_pinned_mode(tmp_path, monkeypatch) -> None:
    log_path = tmp_path / "routing.log"
    monkeypatch.setenv("OPENROUTER_PARETO_ROUTING_LOG", str(log_path))
    cb, _ = _decision_callback(_entry("baseten/fp8", ("baseten/fp8",)), wildcard=False)
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
    (line,) = _decision_lines(log_path)
    assert "mode=pinned" in line
    assert "slug=baseten/fp8" in line
    assert "decision=winner" in line


async def test_log_decisions_pinned_no_winner_records_pass_through(tmp_path, monkeypatch) -> None:
    """No winner to narrow to: the deployment is handed back as-is, still carrying the
    operator's own pin. That is not a plugin routing decision, and the log says so."""
    log_path = tmp_path / "routing.log"
    monkeypatch.setenv("OPENROUTER_PARETO_ROUTING_LOG", str(log_path))
    cb, _ = _decision_callback(None, wildcard=False)
    deps = [
        {
            "model_name": "m",
            "litellm_params": {
                "model": "openrouter/z-ai/glm-5.2",
                "extra_body": {"provider": {"only": ["baseten/fp8"]}},
            },
            "model_info": {"id": "or-baseten/fp8"},
        }
    ]
    result = await cb.async_filter_deployments("z-ai/glm-5.2", deps, None)
    assert [_id_of(d) for d in result] == ["or-baseten/fp8"]
    (line,) = _decision_lines(log_path)
    assert "mode=pinned" in line
    assert "slug=-" in line
    assert "decision=pass_through" in line


async def test_log_decisions_strict_conflict_records_declined(tmp_path, monkeypatch) -> None:
    log_path = tmp_path / "routing.log"
    monkeypatch.setenv("OPENROUTER_PARETO_ROUTING_LOG", str(log_path))
    rules = {
        "z-ai/glm-5.2": rule(
            precision="fp8",
            min_context=1_000_000,
            min_stats_requests=100,
            wildcard=True,
            strict_provider=True,
            log_decisions=True,
        )
    }
    cb = OpenRouterParetoCallback(rules=rules, telemetry=_StubTelemetry(None))
    with pytest.raises(StrictProviderConflict):
        await cb.async_filter_deployments(
            "z-ai/glm-5.2", [_wildcard_dep()], None, {"provider": {"only": ["baseten/fp8"]}}
        )
    (line,) = _decision_lines(log_path)
    assert "decision=declined" in line
    assert "mode=wildcard" in line


async def test_log_decisions_off_creates_no_file(tmp_path, monkeypatch) -> None:
    log_path = tmp_path / "routing.log"
    monkeypatch.setenv("OPENROUTER_PARETO_ROUTING_LOG", str(log_path))
    cb, _ = _decision_callback(_entry("baseten/fp8", ("baseten/fp8",)), log_decisions=False)
    result = await cb.async_filter_deployments("z-ai/glm-5.2", [_wildcard_dep()], None)
    assert _provider_only(result[0]) == "baseten/fp8"
    assert not log_path.exists()


async def test_log_decisions_unmanaged_model_writes_nothing(tmp_path, monkeypatch) -> None:
    log_path = tmp_path / "routing.log"
    monkeypatch.setenv("OPENROUTER_PARETO_ROUTING_LOG", str(log_path))
    cb, _ = _decision_callback(_entry("baseten/fp8", ("baseten/fp8",)))
    result = await cb.async_filter_deployments("unmanaged/model", [_wildcard_dep()], None)
    assert len(result) == 1
    assert not log_path.exists()


async def test_decision_log_unwritable_path_still_routes(tmp_path) -> None:
    """A logging failure must never fail a routed request."""
    import os

    log_path = tmp_path / "sub" / "routing.log"
    os.environ["OPENROUTER_PARETO_ROUTING_LOG"] = str(log_path)
    log_path.parent.mkdir(parents=True)
    log_path.parent.chmod(0o555)
    try:
        cb, _ = _decision_callback(_entry("baseten/fp8", ("baseten/fp8",)))
        result = await cb.async_filter_deployments("z-ai/glm-5.2", [_wildcard_dep()], None)
        assert _provider_only(result[0]) == "baseten/fp8"
        assert not log_path.exists()
    finally:
        log_path.parent.chmod(0o755)
        del os.environ["OPENROUTER_PARETO_ROUTING_LOG"]


async def test_log_decisions_normalizes_model_name_to_rule_key(tmp_path, monkeypatch) -> None:
    """The logged model is the rule key, not the client's spelling of it, so a line
    greps the same whether Claude Code or a bare-id client sent it."""
    log_path = tmp_path / "routing.log"
    monkeypatch.setenv("OPENROUTER_PARETO_ROUTING_LOG", str(log_path))
    cb, _ = _decision_callback(_entry("baseten/fp8", ("baseten/fp8",)))
    await cb.async_filter_deployments("openrouter/z-ai/glm-5.2[1m]", [_wildcard_dep()], None)
    (line,) = _decision_lines(log_path)
    assert "model=z-ai/glm-5.2 " in line


def test_format_decision_is_single_line_with_fixed_fields() -> None:
    from datetime import datetime, timezone

    from litellm_plugin_openrouter_pareto.decision_log import Decision, format_decision

    now = datetime(2026, 9, 21, 14, 2, 11, tzinfo=timezone.utc)
    line = format_decision("z-ai/glm-5.2", "wildcard", Decision(None, "declined", None), now=now)
    assert line == ("2026-09-21T14:02:11+00:00 model=z-ai/glm-5.2 mode=wildcard slug=- decision=declined region=-")


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


async def test_cold_start_all_eligible_hot_raises_even_if_ineligible_nonhot() -> None:
    """An eligible-but-hot fallback plus a region-ineligible non-hot fallback must
    raise AllProvidersOnCooldown: the ineligible provider is unusable, so it must
    not mask every eligible provider being 429-hot (regression guard: an earlier
    version cleared `all_hot` for any non-hot fallback before the region check)."""
    cd = RateLimitCooldown(threshold=1)
    entry = _entry_with_excluded(("excluded",), stale=True, allowed=("allowed", "excluded"))
    rules = {
        "z-ai/glm-5.2": rule(
            precision="fp8",
            min_context=1_000_000,
            min_stats_requests=100,
            wildcard=True,
            cold_start_fallback=["allowed/fp8", "excluded/fp8"],
            exclude_regions=["US"],
            unverified_region_policy="no_route",
        )
    }
    cb = OpenRouterParetoCallback(rules=rules, telemetry=_StubTelemetry(entry), cooldown=cd)
    cd.record("z-ai/glm-5.2", "allowed/fp8")  # eligible, hot
    # excluded/fp8 is non-hot but region-ineligible (base "excluded")
    with pytest.raises(AllProvidersOnCooldown, match="429 cooldown"):
        await cb.async_filter_deployments("z-ai/glm-5.2", [_wildcard_dep()], None)


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

        def get_allowed_providers(self) -> frozenset[str] | None:
            return None

        def set_allowed_providers(self, slugs: frozenset[str]) -> None:
            pass

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
    result = await cb.async_filter_deployments("z-ai/glm-5.2", [_pinned_dep("never-seen-in-telemetry/fp8")], None)
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
    result = await cb.async_filter_deployments("z-ai/glm-5.2", [_pinned_dep("never-seen-in-telemetry/fp8")], None)
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
    cb, _ = _wildcard_region_callback(None, cold_start_fallback=(), unverified_region_policy="trust_fallback")
    result = await cb.async_filter_deployments("z-ai/glm-5.2", [_wildcard_dep()], None)
    assert result == []


async def test_trust_fallback_all_cooled_down_raises_not_unpinned() -> None:
    """trust_fallback means 'only the operator-vetted slugs'. With every vetted
    slug 429-hot, raise AllProvidersOnCooldown rather than degrade to the
    `unpinned` policy (silently ceding to OpenRouter)."""
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
    with pytest.raises(AllProvidersOnCooldown, match="429 cooldown"):
        await cb.async_filter_deployments("z-ai/glm-5.2", [_wildcard_dep()], None)


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
    cb = _pinned_region_callback(_region_entry(("blocked",), allowed=("allowed", "second")))
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
    rules = {"z-ai/glm-5.2": rule(precision="fp8", min_context=1_000_000, min_stats_requests=100)}
    cb = OpenRouterParetoCallback(rules=rules, telemetry=_StubTelemetry(entry))
    multi: dict[str, Any] = {
        "model_name": "z-ai/glm-5.2",
        "litellm_params": {"extra_body": {"provider": {"only": ["first/fp8", "second/fp8"]}}},
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
    return OpenRouterParetoCallback(rules=rules, telemetry=_StubTelemetry(entry), cooldown=cooldown)


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
    cb = _region_cb(None, policy="trust_fallback", fallback=["allowed/fp8"], cooldown=cd)
    result = await cb.async_filter_deployments("z-ai/glm-5.2", [_pinned_dep("allowed/fp8")], None, {})
    assert result == []


async def test_pinned_trust_fallback_routes_when_provider_is_not_cooled() -> None:
    cb = _region_cb(
        None,
        policy="trust_fallback",
        fallback=["allowed/fp8"],
        cooldown=RateLimitCooldown(threshold=1),
    )
    result = await cb.async_filter_deployments("z-ai/glm-5.2", [_pinned_dep("allowed/fp8")], None, {})
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
    request_kwargs: dict[str, Any] = {"extra_body": {"provider": {"only": ["deepinfra"], "zdr": False}}}
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
    result = await cb.async_filter_deployments("z-ai/glm-5.2", _deployments(), None, request_kwargs)
    assert [_id_of(d) for d in result] == ["or-baseten"]


async def test_strict_mode_does_not_affect_unmanaged_models() -> None:
    """Strict is per-rule; an unmanaged model is passed through untouched even with a
    client provider.only."""
    cb = _strict_callback(_entry("baseten", ("baseten",)))
    dep: dict[str, Any] = {"model_name": "gpt-4o", "litellm_params": {}, "model_info": {"id": "x"}}
    request_kwargs: dict[str, Any] = {"extra_body": {"provider": {"only": ["deepinfra"]}}}
    result = await cb.async_filter_deployments("gpt-4o", [dep], None, request_kwargs)
    assert result == [dep]


def _set_litellm_attr(name: str, value: object) -> object | None:
    """Set a dynamic litellm attr and return the previous value (or None)."""
    import litellm

    old = getattr(litellm, name, None)
    setattr(litellm, name, value)
    return old


def _restore_litellm_attr(name: str, old: object | None) -> None:
    import litellm

    if old is None:
        if hasattr(litellm, name):
            delattr(litellm, name)
    else:
        setattr(litellm, name, old)


def test_load_telemetry_config_reads_litellm_attr() -> None:
    from litellm_plugin_openrouter_pareto.config import load_telemetry_config

    old = _set_litellm_attr("openrouter_pareto_telemetry", {"ssl_verify": False})
    try:
        cfg = load_telemetry_config()
    finally:
        _restore_litellm_attr("openrouter_pareto_telemetry", old)
    assert cfg is not None
    assert cfg.ssl_verify is False
    assert cfg.ssl_ca_cert is None


def test_load_telemetry_config_returns_none_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    import litellm

    from litellm_plugin_openrouter_pareto.config import load_telemetry_config

    old = getattr(litellm, "openrouter_pareto_telemetry", None)
    if hasattr(litellm, "openrouter_pareto_telemetry"):
        delattr(litellm, "openrouter_pareto_telemetry")
    monkeypatch.delenv("OPENROUTER_PARETO_TELEMETRY", raising=False)
    try:
        assert load_telemetry_config() is None
    finally:
        if old is not None:
            setattr(  # noqa: B010  # restore
                litellm, "openrouter_pareto_telemetry", old
            )


def test_load_telemetry_config_reads_env_json(monkeypatch: pytest.MonkeyPatch) -> None:
    import litellm

    from litellm_plugin_openrouter_pareto.config import load_telemetry_config

    old = getattr(litellm, "openrouter_pareto_telemetry", None)
    if hasattr(litellm, "openrouter_pareto_telemetry"):
        delattr(litellm, "openrouter_pareto_telemetry")
    monkeypatch.setenv("OPENROUTER_PARETO_TELEMETRY", '{"ssl_ca_cert": "/ca.pem"}')
    try:
        cfg = load_telemetry_config()
    finally:
        if old is not None:
            setattr(  # noqa: B010  # restore
                litellm, "openrouter_pareto_telemetry", old
            )
    assert cfg is not None
    assert cfg.ssl_ca_cert == "/ca.pem"
    assert cfg.ssl_verify is True


def test_load_telemetry_config_raises_on_unknown_field() -> None:
    from litellm_plugin_openrouter_pareto.config import RuleConfigError, load_telemetry_config

    old = _set_litellm_attr("openrouter_pareto_telemetry", {"ssl_verfy": False})
    try:
        with pytest.raises(RuleConfigError):
            load_telemetry_config()
    finally:
        _restore_litellm_attr("openrouter_pareto_telemetry", old)


def test_load_telemetry_config_raises_on_non_dict() -> None:
    from litellm_plugin_openrouter_pareto.config import RuleConfigError, load_telemetry_config

    old = _set_litellm_attr("openrouter_pareto_telemetry", "not a dict")
    try:
        with pytest.raises(RuleConfigError):
            load_telemetry_config()
    finally:
        _restore_litellm_attr("openrouter_pareto_telemetry", old)


def test_load_telemetry_config_raises_on_ca_with_verify_false() -> None:
    from litellm_plugin_openrouter_pareto.config import RuleConfigError, load_telemetry_config

    old = _set_litellm_attr("openrouter_pareto_telemetry", {"ssl_verify": False, "ssl_ca_cert": "/ca.pem"})
    try:
        with pytest.raises(RuleConfigError):
            load_telemetry_config()
    finally:
        _restore_litellm_attr("openrouter_pareto_telemetry", old)


def test_telemetry_config_rejects_ca_with_verify_false_directly() -> None:
    from litellm_plugin_openrouter_pareto.config import TelemetryConfig

    with pytest.raises(ValueError, match="ssl_ca_cert"):
        TelemetryConfig(ssl_verify=False, ssl_ca_cert="/ca.pem")


async def test_plugin_threads_telemetry_verify_from_config() -> None:
    """A telemetry SSL config (with no explicit telemetry injected) rebuilds the
    telemetry client with the configured verify, so the operator's TLS choice
    reaches the OpenRouter stats fetch."""
    from litellm_plugin_openrouter_pareto.telemetry import Telemetry

    old = _set_litellm_attr("openrouter_pareto_telemetry", {"ssl_verify": False})
    try:
        cb = OpenRouterParetoCallback()
        cb._resolve_rules()
    finally:
        _restore_litellm_attr("openrouter_pareto_telemetry", old)
    tel = cb._telemetry
    assert isinstance(tel, Telemetry)
    assert tel._verify is False


async def test_plugin_threads_custom_ca_from_config() -> None:
    from litellm_plugin_openrouter_pareto.telemetry import Telemetry

    old = _set_litellm_attr("openrouter_pareto_telemetry", {"ssl_ca_cert": "/ca.pem"})
    try:
        cb = OpenRouterParetoCallback()
        cb._resolve_rules()
    finally:
        _restore_litellm_attr("openrouter_pareto_telemetry", old)
    tel = cb._telemetry
    assert isinstance(tel, Telemetry)
    assert tel._verify == "/ca.pem"


async def test_plugin_keeps_default_verify_without_telemetry_config() -> None:
    """No telemetry config and no YAML rules -> the __init__ default client (verify
    = True) is kept, not rebuilt. Backward-compatible baseline."""
    import litellm

    from litellm_plugin_openrouter_pareto.telemetry import Telemetry

    old_rules = getattr(litellm, "openrouter_pareto_rules", None)
    old_tel = getattr(litellm, "openrouter_pareto_telemetry", None)
    if hasattr(litellm, "openrouter_pareto_rules"):
        delattr(litellm, "openrouter_pareto_rules")
    if hasattr(litellm, "openrouter_pareto_telemetry"):
        delattr(litellm, "openrouter_pareto_telemetry")
    try:
        cb = OpenRouterParetoCallback()
        before = cb._telemetry
        cb._resolve_rules()
    finally:
        if old_rules is not None:
            setattr(  # noqa: B010  # restore
                litellm, "openrouter_pareto_rules", old_rules
            )
        if old_tel is not None:
            setattr(  # noqa: B010  # restore
                litellm, "openrouter_pareto_telemetry", old_tel
            )
    assert cb._telemetry is before, "default client was needlessly rebuilt"
    assert isinstance(before, Telemetry)
    assert before._verify is True


async def test_explicit_telemetry_survives_telemetry_config() -> None:
    """An explicitly injected telemetry source is never replaced, even when a
    telemetry SSL config is present (lazy resolution preserves injected deps)."""
    stub = _StubTelemetry(None)
    old = _set_litellm_attr("openrouter_pareto_telemetry", {"ssl_verify": False})
    try:
        cb = OpenRouterParetoCallback(
            rules={"z-ai/glm-5.2": rule(precision="fp8", min_context=1_000_000, min_stats_requests=100)},
            telemetry=stub,
        )
        cb._resolve_rules()
    finally:
        _restore_litellm_attr("openrouter_pareto_telemetry", old)
    assert cb._telemetry is stub


async def test_explicit_rules_still_apply_telemetry_verify_from_config() -> None:
    """Global telemetry SSL config applies even when rules are supplied directly to
    the constructor - the two config domains are independent (regression: an early
    return on explicit rules used to skip load_telemetry_config entirely)."""
    from litellm_plugin_openrouter_pareto.telemetry import Telemetry

    old = _set_litellm_attr("openrouter_pareto_telemetry", {"ssl_verify": False})
    try:
        cb = OpenRouterParetoCallback(
            rules={"z-ai/glm-5.2": rule(precision="fp8", min_context=1_000_000, min_stats_requests=100)},
        )
        cb._resolve_rules()
    finally:
        _restore_litellm_attr("openrouter_pareto_telemetry", old)
    tel = cb._telemetry
    assert isinstance(tel, Telemetry)
    assert tel._verify is False


async def test_explicit_rules_still_apply_custom_ca_from_config() -> None:
    from litellm_plugin_openrouter_pareto.telemetry import Telemetry

    old = _set_litellm_attr("openrouter_pareto_telemetry", {"ssl_ca_cert": "/ca.pem"})
    try:
        cb = OpenRouterParetoCallback(
            rules={"z-ai/glm-5.2": rule(precision="fp8", min_context=1_000_000, min_stats_requests=100)},
        )
        cb._resolve_rules()
    finally:
        _restore_litellm_attr("openrouter_pareto_telemetry", old)
    tel = cb._telemetry
    assert isinstance(tel, Telemetry)
    assert tel._verify == "/ca.pem"


async def test_explicit_rules_raise_on_malformed_telemetry_config() -> None:
    """Bad telemetry config surfaces loudly even when rules are explicit - it must
    not be silently ignored because the rule source was injected."""
    from litellm_plugin_openrouter_pareto.config import RuleConfigError

    old = _set_litellm_attr("openrouter_pareto_telemetry", {"ssl_verfy": False})
    try:
        cb = OpenRouterParetoCallback(
            rules={"z-ai/glm-5.2": rule(precision="fp8", min_context=1_000_000, min_stats_requests=100)},
        )
        with pytest.raises(RuleConfigError):
            cb._resolve_rules()
    finally:
        _restore_litellm_attr("openrouter_pareto_telemetry", old)


# --- AllProvidersOnCooldown: raised (not []) only on an all-429-hot list ---


async def test_all_hot_is_value_error_subclass() -> None:
    assert issubclass(AllProvidersOnCooldown, ValueError)


async def test_warm_all_hot_message_names_model_winner_and_safe_set() -> None:
    """The exception carries the plugin's own reason (not litellm's generic
    RouterRateLimitError with an empty cooldown_list)."""
    cd = RateLimitCooldown(threshold=1)
    cb, _ = _callback(_entry("baseten", ("baseten", "novita", "siliconflow")), cooldown=cd, wildcard=True)
    for slug in ("baseten", "novita", "siliconflow"):
        cd.record("z-ai/glm-5.2", slug)
    with pytest.raises(AllProvidersOnCooldown) as exc_info:
        await cb.async_filter_deployments("z-ai/glm-5.2", [_wildcard_dep()], None)
    msg = str(exc_info.value)
    assert "z-ai/glm-5.2" in msg
    assert "baseten" in msg
    assert "novita" in msg


async def test_warm_single_non_hot_in_safe_set_is_pinned_not_raised() -> None:
    """One non-hot provider in the list is enough: it is pinned instead of raising."""
    cd = RateLimitCooldown(threshold=1)
    cb, _ = _callback(_entry("baseten", ("baseten", "novita")), cooldown=cd, wildcard=True)
    cd.record("z-ai/glm-5.2", "baseten")  # winner hot
    result = await cb.async_filter_deployments("z-ai/glm-5.2", [_wildcard_dep()], None)
    assert _provider_only(result[0]) == "novita"


async def test_warm_all_input_capped_does_not_raise() -> None:
    """Input cap is a soft preference: all-capped falls back to the winner, not
    the all-429-hot raise."""
    cd = RateLimitCooldown(threshold=10)
    cb, _ = _callback(_entry("baseten/fp8", ("baseten/fp8", "novita/fp8")), cooldown=cd, wildcard=True)
    cd.record_input_cap("z-ai/glm-5.2", "baseten/fp8")
    cd.record_input_cap("z-ai/glm-5.2", "novita/fp8")
    result = await cb.async_filter_deployments("z-ai/glm-5.2", [_wildcard_dep()], None)
    assert _provider_only(result[0]) == "baseten/fp8"


async def test_cold_start_all_fallbacks_hot_raises() -> None:
    cd = RateLimitCooldown(threshold=1)
    cb, _ = _callback(None, wildcard=True, cold_start_fallback=("a/fp8", "b/fp8"), cooldown=cd)
    cd.record("z-ai/glm-5.2", "a/fp8")
    cd.record("z-ai/glm-5.2", "b/fp8")
    with pytest.raises(AllProvidersOnCooldown, match="429 cooldown"):
        await cb.async_filter_deployments("z-ai/glm-5.2", [_wildcard_dep()], None)


async def test_cold_start_no_fallback_does_not_raise_declines() -> None:
    """No fallback configured is not an all-429-hot state: decline ([]) instead
    of raising."""
    cb, _ = _callback(None, wildcard=True)
    result = await cb.async_filter_deployments("z-ai/glm-5.2", [_wildcard_dep()], None)
    assert result == []


# --- CooldownConfig loading + plugin resolution (mirrors telemetry config) ---


def test_load_cooldown_config_reads_litellm_attr() -> None:
    from litellm_plugin_openrouter_pareto.config import load_cooldown_config

    old = _set_litellm_attr(
        "openrouter_pareto_cooldown",
        {"rate_limit_window_s": 120.0, "rate_limit_threshold": 5, "input_cap_ttl_s": 7200.0},
    )
    try:
        cfg = load_cooldown_config()
    finally:
        _restore_litellm_attr("openrouter_pareto_cooldown", old)
    assert cfg is not None
    assert cfg.rate_limit_window_s == 120.0
    assert cfg.rate_limit_threshold == 5
    assert cfg.input_cap_ttl_s == 7200.0


def test_load_cooldown_config_returns_none_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    import litellm

    from litellm_plugin_openrouter_pareto.config import load_cooldown_config

    old = getattr(litellm, "openrouter_pareto_cooldown", None)
    if hasattr(litellm, "openrouter_pareto_cooldown"):
        delattr(litellm, "openrouter_pareto_cooldown")
    monkeypatch.delenv("OPENROUTER_PARETO_COOLDOWN", raising=False)
    try:
        assert load_cooldown_config() is None
    finally:
        if old is not None:
            setattr(  # noqa: B010  # restore
                litellm, "openrouter_pareto_cooldown", old
            )


def test_load_cooldown_config_reads_env_json(monkeypatch: pytest.MonkeyPatch) -> None:
    import litellm

    from litellm_plugin_openrouter_pareto.config import load_cooldown_config

    old = getattr(litellm, "openrouter_pareto_cooldown", None)
    if hasattr(litellm, "openrouter_pareto_cooldown"):
        delattr(litellm, "openrouter_pareto_cooldown")
    monkeypatch.setenv("OPENROUTER_PARETO_COOLDOWN", '{"rate_limit_threshold": 7}')
    try:
        cfg = load_cooldown_config()
    finally:
        if old is not None:
            setattr(  # noqa: B010  # restore
                litellm, "openrouter_pareto_cooldown", old
            )
    assert cfg is not None
    assert cfg.rate_limit_threshold == 7
    assert cfg.rate_limit_window_s == 300.0  # default fills the unset fields
    assert cfg.input_cap_ttl_s == 3600.0


def test_load_cooldown_config_raises_on_unknown_field() -> None:
    from litellm_plugin_openrouter_pareto.config import RuleConfigError, load_cooldown_config

    old = _set_litellm_attr("openrouter_pareto_cooldown", {"window_s": 120.0})
    try:
        with pytest.raises(RuleConfigError):
            load_cooldown_config()
    finally:
        _restore_litellm_attr("openrouter_pareto_cooldown", old)


def test_load_cooldown_config_raises_on_non_dict() -> None:
    from litellm_plugin_openrouter_pareto.config import RuleConfigError, load_cooldown_config

    old = _set_litellm_attr("openrouter_pareto_cooldown", "not a dict")
    try:
        with pytest.raises(RuleConfigError):
            load_cooldown_config()
    finally:
        _restore_litellm_attr("openrouter_pareto_cooldown", old)


def test_load_cooldown_config_raises_on_out_of_range(monkeypatch: pytest.MonkeyPatch) -> None:
    import litellm

    from litellm_plugin_openrouter_pareto.config import RuleConfigError, load_cooldown_config

    old = getattr(litellm, "openrouter_pareto_cooldown", None)
    if hasattr(litellm, "openrouter_pareto_cooldown"):
        delattr(litellm, "openrouter_pareto_cooldown")
    monkeypatch.setenv("OPENROUTER_PARETO_COOLDOWN", '{"rate_limit_threshold": 0}')
    try:
        with pytest.raises(RuleConfigError):
            load_cooldown_config()
    finally:
        if old is not None:
            setattr(  # noqa: B010  # restore
                litellm, "openrouter_pareto_cooldown", old
            )


def test_load_cooldown_config_rejects_nonfinite(monkeypatch: pytest.MonkeyPatch) -> None:
    """An infinite window makes recorded 429s effectively permanent and an
    infinite input-cap TTL makes caps permanent; NaN / inf must surface as bad
    config, not be silently accepted as a valid timer."""
    import litellm

    from litellm_plugin_openrouter_pareto.config import RuleConfigError, load_cooldown_config

    for bad in ('{"rate_limit_window_s": Infinity}', '{"input_cap_ttl_s": NaN}'):
        old = getattr(litellm, "openrouter_pareto_cooldown", None)
        if hasattr(litellm, "openrouter_pareto_cooldown"):
            delattr(litellm, "openrouter_pareto_cooldown")
        monkeypatch.setenv("OPENROUTER_PARETO_COOLDOWN", bad)
        try:
            with pytest.raises(RuleConfigError):
                load_cooldown_config()
        finally:
            if old is not None:
                setattr(  # noqa: B010  # restore
                    litellm, "openrouter_pareto_cooldown", old
                )


def test_load_cooldown_config_rejects_bool_and_string(monkeypatch: pytest.MonkeyPatch) -> None:
    """`rate_limit_window_s: true` would become 1.0 and `rate_limit_threshold:
    true` would become 1, shrinking the window / threshold from a typo. Reject
    bool and string masquerading as numbers on all cooldown fields."""
    import litellm

    from litellm_plugin_openrouter_pareto.config import RuleConfigError, load_cooldown_config

    bads = [
        '{"rate_limit_window_s": true}',
        '{"rate_limit_threshold": false}',
        '{"input_cap_ttl_s": "3600"}',
        '{"rate_limit_threshold": 3.0}',
    ]
    for bad in bads:
        old = getattr(litellm, "openrouter_pareto_cooldown", None)
        if hasattr(litellm, "openrouter_pareto_cooldown"):
            delattr(litellm, "openrouter_pareto_cooldown")
        monkeypatch.setenv("OPENROUTER_PARETO_COOLDOWN", bad)
        try:
            with pytest.raises(RuleConfigError):
                load_cooldown_config()
        finally:
            if old is not None:
                setattr(  # noqa: B010  # restore
                    litellm, "openrouter_pareto_cooldown", old
                )


async def test_plugin_threads_cooldown_config_into_rate_limit_cooldown() -> None:
    """A cooldown config (with no explicit cooldown injected) rebuilds the
    cooldown store with the configured window/threshold/ttl."""
    old = _set_litellm_attr(
        "openrouter_pareto_cooldown",
        {"rate_limit_window_s": 90.0, "rate_limit_threshold": 2, "input_cap_ttl_s": 1800.0},
    )
    try:
        cb = OpenRouterParetoCallback()
        cb._resolve_rules()
    finally:
        _restore_litellm_attr("openrouter_pareto_cooldown", old)
    cd = cb._cooldown
    assert isinstance(cd, RateLimitCooldown)
    assert cd._window_s == 90.0
    assert cd._threshold == 2
    assert cd._input_cap_ttl_s == 1800.0


async def test_plugin_keeps_default_cooldown_without_config() -> None:
    """No cooldown config -> the __init__ default cooldown is kept, not rebuilt."""
    import litellm

    old = getattr(litellm, "openrouter_pareto_cooldown", None)
    if hasattr(litellm, "openrouter_pareto_cooldown"):
        delattr(litellm, "openrouter_pareto_cooldown")
    try:
        cb = OpenRouterParetoCallback()
        before = cb._cooldown
        cb._resolve_rules()
    finally:
        if old is not None:
            setattr(  # noqa: B010  # restore
                litellm, "openrouter_pareto_cooldown", old
            )
    assert cb._cooldown is before, "default cooldown was needlessly rebuilt"
    assert isinstance(before, RateLimitCooldown)
    assert before._window_s == 300.0
    assert before._threshold == 3


async def test_explicit_cooldown_survives_cooldown_config() -> None:
    """An explicitly injected cooldown is never replaced by global config
    (lazy resolution preserves injected deps - same contract as telemetry)."""
    injected = RateLimitCooldown(window_s=42.0, threshold=9, input_cap_ttl_s=99.0)
    old = _set_litellm_attr("openrouter_pareto_cooldown", {"rate_limit_window_s": 90.0})
    try:
        cb = OpenRouterParetoCallback(
            rules={"z-ai/glm-5.2": rule(precision="fp8", min_context=1_000_000, min_stats_requests=100)},
            cooldown=injected,
        )
        cb._resolve_rules()
    finally:
        _restore_litellm_attr("openrouter_pareto_cooldown", old)
    assert cb._cooldown is injected
    assert cb._cooldown._window_s == 42.0


# ---------------------------------------------------------------------------
# Allowed-providers allowlist (error-driven discovery + filtering)
# ---------------------------------------------------------------------------

_404_NO_ALLOWED_BODY = (
    "No allowed providers are available for the selected model. "
    "Providers serving z-ai/glm-5.2-20260616: baidu, streamlake, sail-research, "
    "novita, digitalocean, gmicloud, deepinfra, inceptron, coreweave, decart, "
    "akashml, alibaba, ambient, morph, phala, siliconflow, wafer, atlas-cloud, "
    "z-ai, fireworks, cloudflare, friendli, parasail, venice, together, crusoe, "
    "baseten, modelrun, but your account's allowed-providers setting permits "
    "only: z-ai, azure, google-vertex, venice, mistral, together, deepinfra, "
    "perplexity, moonshotai, digitalocean, amazon-bedrock. To change your "
    "allowed providers, visit: https://openrouter.ai/settings/privacy. "
    "Additionally, your request's provider.only preference permits only: "
    "baseten/fast."
)


class _Exc404:
    status_code = 404

    def __str__(self) -> str:
        return _404_NO_ALLOWED_BODY


async def test_allowlist_filters_disallowed_winner_wildcard() -> None:
    """In wildcard mode, the winner is skipped when its base is not in the
    allowlist, and the next allowed safe-set slug is chosen."""
    allowed = frozenset({"z-ai", "venice"})
    entry = _entry("baseten/fast", ("sail-research/fp8", "venice/fp8", "z-ai/fp8", "baseten/fast"))
    cb, _ = _callback(entry, wildcard=True, allowed_providers=allowed)
    result = await cb.async_filter_deployments("z-ai/glm-5.2", [_wildcard_dep()], None)
    assert _provider_only(result[0]) == "venice/fp8"


async def test_allowlist_none_means_no_filtering() -> None:
    """When the allowlist is unknown (None), no filtering is applied — the
    pareto-optimal winner is chosen as before (backward compat)."""
    entry = _entry("baseten/fast", ("venice/fp8", "baseten/fast"))
    cb, _ = _callback(entry, wildcard=True, allowed_providers=None)
    result = await cb.async_filter_deployments("z-ai/glm-5.2", [_wildcard_dep()], None)
    assert _provider_only(result[0]) == "baseten/fast"


async def test_allowlist_filters_cold_start_fallback() -> None:
    """Cold-start fallbacks whose base is not in the allowlist are skipped."""
    allowed = frozenset({"venice"})
    cb, _ = _callback(
        None,
        wildcard=True,
        cold_start_fallback=("novita/fp8", "venice/fp8"),
        allowed_providers=allowed,
    )
    result = await cb.async_filter_deployments("z-ai/glm-5.2", [_wildcard_dep()], None)
    assert _provider_only(result[0]) == "venice/fp8"


async def test_allowlist_all_fallbacks_disallowed_returns_empty() -> None:
    """When every cold-start fallback is disallowed and no region policy cedes,
    the plugin declines to route (returns [])."""
    allowed = frozenset({"z-ai"})
    cb, _ = _callback(
        None,
        wildcard=True,
        cold_start_fallback=("novita/fp8", "baseten/fp8"),
        allowed_providers=allowed,
    )
    result = await cb.async_filter_deployments("z-ai/glm-5.2", [_wildcard_dep()], None)
    assert result == []


async def test_allowlist_filters_narrow_mode() -> None:
    """In non-wildcard (pinned) mode, _preferred_slugs filters by allowlist."""
    allowed = frozenset({"venice"})
    entry = _entry("baseten", ("baseten", "venice", "novita"))
    cb, _ = _callback(entry, wildcard=False, allowed_providers=allowed)
    deps = [
        _dep("or-baseten"),
        _dep("or-venice"),
        _dep("or-novita"),
    ]
    result = await cb.async_filter_deployments("z-ai/glm-5.2", deps, None)
    assert _id_of(result[0]) == "or-venice"


async def test_404_discovers_allowlist_from_error() -> None:
    """A 404 'No allowed providers' error parses the allowlist from the error
    body and stores it on the telemetry stub."""
    cb, stub = _callback(_entry("baseten/fast", ("baseten/fast",)))
    assert stub._allowed_providers is None
    kwargs = _failure_kwargs("baseten/fast", _Exc404())
    await cb.async_log_failure_event(kwargs, None, 0.0, 0.0)
    assert stub._allowed_providers is not None
    assert "z-ai" in stub._allowed_providers
    assert "venice" in stub._allowed_providers
    assert "baseten" not in stub._allowed_providers


async def test_404_passthrough_discovers_allowlist() -> None:
    """The passthrough failure hook also discovers the allowlist from a 404."""
    cb, stub = _callback(_entry("baseten/fast", ("baseten/fast",)))
    assert stub._allowed_providers is None
    request_data = _post_call_request_data(
        "baseten/fast",
        response_body={"error": {"message": _404_NO_ALLOWED_BODY}},
    )
    await cb.async_post_call_failure_hook(request_data, _SyntheticUpstreamExc(404), None)
    assert stub._allowed_providers is not None
    assert "z-ai" in stub._allowed_providers


async def test_non_404_error_does_not_discover_allowlist() -> None:
    """A 500 error does not trigger allowlist discovery."""
    cb, stub = _callback(_entry("baseten/fast", ("baseten/fast",)))

    class _Exc500:
        status_code = 500

    kwargs = _failure_kwargs("baseten/fast", _Exc500())
    await cb.async_log_failure_event(kwargs, None, 0.0, 0.0)
    assert stub._allowed_providers is None


async def test_404_without_allowed_providers_pattern_does_not_discover() -> None:
    """A 404 that doesn't contain the 'No allowed providers' pattern does not
    trigger allowlist discovery."""
    cb, stub = _callback(_entry("baseten/fast", ("baseten/fast",)))

    class _Exc404Other:
        status_code = 404

        def __str__(self) -> str:
            return "Not Found"

    kwargs = _failure_kwargs("baseten/fast", _Exc404Other())
    await cb.async_log_failure_event(kwargs, None, 0.0, 0.0)
    assert stub._allowed_providers is None


async def test_allowlist_filters_then_404_re_discovers() -> None:
    """When the allowlist is set, a disallowed winner is filtered. After a 404
    re-discovers the allowlist, the next request filters against the updated
    list."""
    entry = _entry("baseten/fast", ("baseten/fast", "venice/fp8", "z-ai/fp8"))
    cb, stub = _callback(entry, wildcard=True)
    # No allowlist yet: baseten/fast is chosen
    result = await cb.async_filter_deployments("z-ai/glm-5.2", [_wildcard_dep()], None)
    assert _provider_only(result[0]) == "baseten/fast"
    # 404 discovers the allowlist
    kwargs = _failure_kwargs("baseten/fast", _Exc404())
    await cb.async_log_failure_event(kwargs, None, 0.0, 0.0)
    assert stub._allowed_providers is not None
    # Now baseten/fast is filtered, venice/fp8 is chosen
    result = await cb.async_filter_deployments("z-ai/glm-5.2", [_wildcard_dep()], None)
    assert _provider_only(result[0]) == "venice/fp8"


async def test_wildcard_winner_hot_walks_frontier_before_dominated() -> None:
    """On the winner's 429 the fallback re-runs the value walk over the remaining
    points: novita (+79% throughput for +8% price over relace) is picked over the
    cheaper-but-slower providers, and the dominated cheapest (z-ai) is last."""
    cd = RateLimitCooldown(threshold=1)
    cb, _ = _callback(
        _entry(
            "baseten/fp8",
            ("baseten/fp8", "morph/fp8", "relace/fp4", "novita/fp8"),
            points=(
                Point(slug="morph/fp8", input_price_m=0.12, tps=6.0),
                Point(slug="relace/fp4", input_price_m=0.13, tps=24.0),
                Point(slug="novita/fp8", input_price_m=0.14, tps=43.0),
                Point(slug="baseten/fp8", input_price_m=0.20, tps=193.0),
            ),
        ),
        cooldown=cd,
        wildcard=True,
    )
    cd.record("z-ai/glm-5.2", "baseten/fp8")  # winner hot
    result = await cb.async_filter_deployments("z-ai/glm-5.2", [_wildcard_dep()], None)
    assert _provider_only(result[0]) == "novita/fp8"


async def test_wildcard_winner_hot_and_next_hot_lands_on_re_walked_second() -> None:
    """With the winner and the re-walked first both hot, the fallback takes the
    re-walked second (relace, +79% over morph) - not the cheapest (morph)."""
    cd = RateLimitCooldown(threshold=1)
    cb, _ = _callback(
        _entry(
            "baseten/fp8",
            ("baseten/fp8", "morph/fp8", "relace/fp4", "novita/fp8"),
            points=(
                Point(slug="morph/fp8", input_price_m=0.12, tps=6.0),
                Point(slug="relace/fp4", input_price_m=0.13, tps=24.0),
                Point(slug="novita/fp8", input_price_m=0.14, tps=43.0),
                Point(slug="baseten/fp8", input_price_m=0.20, tps=193.0),
            ),
        ),
        cooldown=cd,
        wildcard=True,
    )
    cd.record("z-ai/glm-5.2", "baseten/fp8")
    cd.record("z-ai/glm-5.2", "novita/fp8")
    result = await cb.async_filter_deployments("z-ai/glm-5.2", [_wildcard_dep()], None)
    assert _provider_only(result[0]) == "relace/fp4"


async def test_wildcard_frontier_all_hot_with_dominated_absent_raises() -> None:
    """A two-point cache: the all-hot raise keys off the re-run walk like any other."""
    cd = RateLimitCooldown(threshold=1)
    cb, _ = _callback(
        _entry(
            "baseten/fp8",
            ("baseten/fp8", "novita/fp8"),
            points=(
                Point(slug="novita/fp8", input_price_m=0.14, tps=43.0),
                Point(slug="baseten/fp8", input_price_m=0.20, tps=193.0),
            ),
        ),
        cooldown=cd,
        wildcard=True,
    )
    cd.record("z-ai/glm-5.2", "baseten/fp8")
    cd.record("z-ai/glm-5.2", "novita/fp8")
    with pytest.raises(AllProvidersOnCooldown):
        await cb.async_filter_deployments("z-ai/glm-5.2", [_wildcard_dep()], None)


async def test_pinned_winner_hot_re_walks_over_real_points() -> None:
    """Pinned mode must reorder under real geometry too, not just wildcard. With the
    same point set as the wildcard re-walk test, a hot winner must narrow to novita -
    the re-walked first pick (+79% throughput for +8% price over relace) - and never to
    morph, which the old cheapest-first ordering would have reached."""
    cd = RateLimitCooldown(threshold=1)
    cb, _ = _callback(
        _entry(
            "baseten",
            ("baseten", "morph", "relace", "novita"),
            points=(
                Point(slug="morph", input_price_m=0.12, tps=6.0),
                Point(slug="relace", input_price_m=0.13, tps=24.0),
                Point(slug="novita", input_price_m=0.14, tps=43.0),
                Point(slug="baseten", input_price_m=0.20, tps=193.0),
            ),
        ),
        cooldown=cd,
    )
    cd.record("z-ai/glm-5.2", "baseten")  # winner hot
    deployments = [_dep("or-baseten"), _dep("or-morph"), _dep("or-relace"), _dep("or-novita")]
    result = await cb.async_filter_deployments("z-ai/glm-5.2", deployments, None)
    assert [_id_of(d) for d in result] == ["or-novita"]


async def test_pinned_winner_hot_and_first_pick_hot_takes_re_walked_second() -> None:
    """Pinned counterpart of the sequential-elimination step: with the winner and the
    re-walked first both hot, pinned mode narrows to the re-walked second (relace)."""
    cd = RateLimitCooldown(threshold=1)
    cb, _ = _callback(
        _entry(
            "baseten",
            ("baseten", "morph", "relace", "novita"),
            points=(
                Point(slug="morph", input_price_m=0.12, tps=6.0),
                Point(slug="relace", input_price_m=0.13, tps=24.0),
                Point(slug="novita", input_price_m=0.14, tps=43.0),
                Point(slug="baseten", input_price_m=0.20, tps=193.0),
            ),
        ),
        cooldown=cd,
    )
    cd.record("z-ai/glm-5.2", "baseten")
    cd.record("z-ai/glm-5.2", "novita")
    deployments = [_dep("or-baseten"), _dep("or-morph"), _dep("or-relace"), _dep("or-novita")]
    result = await cb.async_filter_deployments("z-ai/glm-5.2", deployments, None)
    assert [_id_of(d) for d in result] == ["or-relace"]


async def test_wildcard_excludes_hot_before_walking_not_after() -> None:
    """A provider justified only by a stepping stone must not survive on that stone's
    order position once the stone is hot. `c` is reachable only via `e`, so with `e`
    hot the walk over what remains picks `a`; filtering a full-set order would have
    left `c` first."""
    cd = RateLimitCooldown(threshold=1)
    cb, _ = _callback(
        _entry(
            "c",
            ("a", "e", "c"),
            points=(
                Point(slug="a", input_price_m=1.0, tps=10.0),
                Point(slug="e", input_price_m=1.4, tps=13.1),
                Point(slug="c", input_price_m=1.96, tps=17.16),
            ),
        ),
        cooldown=cd,
        tolerance=0.1,
        wildcard=True,
    )
    cd.record("z-ai/glm-5.2", "e")  # the stepping stone is hot
    result = await cb.async_filter_deployments("z-ai/glm-5.2", [_wildcard_dep()], None)
    assert _provider_only(result[0]) == "a"


async def test_pinned_excludes_hot_before_walking_not_after() -> None:
    """Pinned counterpart of the stepping-stone exclusion."""
    cd = RateLimitCooldown(threshold=1)
    cb, _ = _callback(
        _entry(
            "c",
            ("a", "e", "c"),
            points=(
                Point(slug="a", input_price_m=1.0, tps=10.0),
                Point(slug="e", input_price_m=1.4, tps=13.1),
                Point(slug="c", input_price_m=1.96, tps=17.16),
            ),
        ),
        cooldown=cd,
        tolerance=0.1,
    )
    cd.record("z-ai/glm-5.2", "e")
    deployments = [_dep("or-a"), _dep("or-e"), _dep("or-c")]
    result = await cb.async_filter_deployments("z-ai/glm-5.2", deployments, None)
    assert [_id_of(d) for d in result] == ["or-a"]


async def test_no_points_entry_still_honors_exclude_for_all_hot_raise() -> None:
    """Regression guard: a points-less entry must still honor `exclude`, or the all-hot
    raise never fires and a hot provider is pinned - worse than the behavior before the
    exclude fix. The entry here is hand-built (no points), the supported path for
    entries a caller constructs directly."""
    cd = RateLimitCooldown(threshold=1)
    cb, _ = _callback(
        _entry("a", ("a", "e", "c"), points=()),
        cooldown=cd,
        wildcard=True,
    )
    for slug in ("a", "e", "c"):
        cd.record("z-ai/glm-5.2", slug)  # every candidate is hot
    with pytest.raises(AllProvidersOnCooldown):
        await cb.async_filter_deployments("z-ai/glm-5.2", [_wildcard_dep()], None)


async def test_no_points_entry_excludes_hot_and_pins_a_cold_one() -> None:
    """The same branch, non-degenerate case: the hot provider is dropped from the
    returned list rather than pinned."""
    cd = RateLimitCooldown(threshold=1)
    cb, _ = _callback(
        _entry("a", ("a", "e"), points=()),
        cooldown=cd,
        wildcard=True,
    )
    cd.record("z-ai/glm-5.2", "a")  # winner hot
    result = await cb.async_filter_deployments("z-ai/glm-5.2", [_wildcard_dep()], None)
    assert _provider_only(result[0]) == "e"


async def test_all_disallowed_raise_does_not_blame_rate_limits() -> None:
    """An all-allowlist-disallowed state raises the same exception, so the message
    must not report it as a 429: an operator reading "on a 429 cooldown" would chase
    rate limits that never happened. Verified with zero cooldown hits recorded."""
    cb, _ = _callback(
        _entry("baseten", ("baseten", "novita")),
        wildcard=True,
        allowed_providers=frozenset({"someoneelse"}),
    )
    with pytest.raises(AllProvidersOnCooldown) as exc:
        await cb.async_filter_deployments("z-ai/glm-5.2", [_wildcard_dep()], None)
    message = str(exc.value)
    assert "allowed-providers" in message
    assert "no 429 is involved" in message
    assert "on a 429 cooldown" not in message


async def test_all_hot_raise_still_blames_rate_limits() -> None:
    """The converse: a genuine all-hot state must still say so."""
    cd = RateLimitCooldown(threshold=1)
    cb, _ = _callback(_entry("baseten", ("baseten", "novita")), cooldown=cd, wildcard=True)
    for slug in ("baseten", "novita"):
        cd.record("z-ai/glm-5.2", slug)
    with pytest.raises(AllProvidersOnCooldown) as exc:
        await cb.async_filter_deployments("z-ai/glm-5.2", [_wildcard_dep()], None)
    assert "429 cooldown" in str(exc.value)
    assert "allowed-providers" not in str(exc.value)


# --- exclude_providers: the static blocklist ---


async def test_exclude_providers_skips_winner_wildcard() -> None:
    """A blocklisted winner is dropped from the walk and the next usable provider is
    pinned, without the blocklist ever having to be re-discovered from an error."""
    entry = _entry("baseten/fp8", ("baseten/fp8", "novita/fp8"))
    cb, _ = _callback(entry, wildcard=True, exclude_providers=("baseten",))
    result = await cb.async_filter_deployments("z-ai/glm-5.2", [_wildcard_dep()], None)
    assert _provider_only(result[0]) == "novita/fp8"


async def test_exclude_providers_org_base_skips_every_endpoint_wildcard() -> None:
    """An org-level entry removes every endpoint of that org, not just the winner's."""
    entry = _entry("novita/fp8", ("novita/fp8", "novita/fp16", "baseten/fp8"))
    cb, _ = _callback(entry, wildcard=True, exclude_providers=("novita",))
    result = await cb.async_filter_deployments("z-ai/glm-5.2", [_wildcard_dep()], None)
    assert _provider_only(result[0]) == "baseten/fp8"


async def test_exclude_providers_full_slug_leaves_siblings_eligible() -> None:
    """A full-slug entry is endpoint-precise: the same org's other variant still routes."""
    entry = _entry("novita/fp8", ("novita/fp8", "novita/fp16"))
    cb, _ = _callback(entry, wildcard=True, exclude_providers=("novita/fp8",))
    result = await cb.async_filter_deployments("z-ai/glm-5.2", [_wildcard_dep()], None)
    assert _provider_only(result[0]) == "novita/fp16"


async def test_exclude_providers_filters_narrow_mode() -> None:
    """Pinned mode narrows to an allowed deployment rather than the excluded winner's."""
    entry = _entry("baseten", ("baseten", "venice", "novita"))
    cb, _ = _callback(entry, wildcard=False, exclude_providers=("baseten",))
    deps = [_dep("or-baseten"), _dep("or-venice"), _dep("or-novita")]
    result = await cb.async_filter_deployments("z-ai/glm-5.2", deps, None)
    assert _id_of(result[0]) == "or-venice"


async def test_exclude_providers_filters_cold_start_fallback() -> None:
    """The operator's own fallback list is subject to the operator's own blocklist."""
    cb, _ = _callback(
        None,
        wildcard=True,
        cold_start_fallback=("novita/fp8", "venice/fp8"),
        exclude_providers=("novita",),
    )
    result = await cb.async_filter_deployments("z-ai/glm-5.2", [_wildcard_dep()], None)
    assert _provider_only(result[0]) == "venice/fp8"


async def test_exclude_providers_all_excluded_raise_does_not_blame_rate_limits() -> None:
    """A blocklist that empties the candidate set raises the same exception as an
    all-hot state, so the message must name the blocklist: an operator reading
    "on a 429 cooldown" would chase rate limits that never happened."""
    cb, _ = _callback(
        _entry("baseten", ("baseten", "novita")),
        wildcard=True,
        exclude_providers=("baseten", "novita"),
    )
    with pytest.raises(AllProvidersOnCooldown) as exc:
        await cb.async_filter_deployments("z-ai/glm-5.2", [_wildcard_dep()], None)
    message = str(exc.value)
    assert "exclude_providers" in message
    assert "on a 429 cooldown" not in message


async def test_exclude_providers_vets_trust_fallback_slugs() -> None:
    """`trust_fallback` trusts the configured slugs' geography, not their usability: a
    blocklisted vetted slug is still unusable, and the policy fails closed rather than
    degrading to the `unpinned` cede."""
    rules = {
        "z-ai/glm-5.2": rule(
            precision="fp8",
            min_context=1_000_000,
            min_stats_requests=100,
            wildcard=True,
            cold_start_fallback=["novita/fp8"],
            exclude_regions=["SG"],
            unverified_region_policy="trust_fallback",
            exclude_providers=["novita"],
        )
    }
    cb = OpenRouterParetoCallback(rules=rules, telemetry=_StubTelemetry(None))
    result = await cb.async_filter_deployments("z-ai/glm-5.2", [_wildcard_dep()], None)
    assert result == []


async def test_exclude_providers_default_is_no_filtering() -> None:
    """Every pre-existing rule has an empty blocklist, so the winner still routes."""
    cb, _ = _callback(_entry("baseten/fp8", ("baseten/fp8", "novita/fp8")), wildcard=True)
    result = await cb.async_filter_deployments("z-ai/glm-5.2", [_wildcard_dep()], None)
    assert _provider_only(result[0]) == "baseten/fp8"


# --- broken_provider_patterns / input_cap_patterns: the 400 body checks ---


async def test_broken_provider_recorded_from_configured_pattern() -> None:
    cd = RateLimitCooldown(threshold=10)
    cb, _ = _callback(
        _entry("baseten/fp8", ("baseten/fp8",)),
        cooldown=cd,
        broken_provider_patterns=("provider returned error",),
    )

    class _Exc:
        status_code = 400

        def __str__(self) -> str:
            return "Provider returned error"

    kwargs = _failure_kwargs("baseten/fp8", _Exc())
    await cb.async_log_failure_event(kwargs, None, 0.0, 0.0)
    assert cd.is_broken("z-ai/glm-5.2", "baseten/fp8") is True
    assert cd.is_hot("z-ai/glm-5.2", "baseten/fp8") is False
    assert cd.is_input_capped("z-ai/glm-5.2", "baseten/fp8") is False


async def test_broken_provider_detection_is_off_by_default() -> None:
    """The one behaviour change that would be unsafe to ship on by default: an
    unconfigured rule must not blacklist a provider from a 400 it never claimed to
    understand. Most 400s are the caller's bad request, and the plugin pins one
    provider, so a retry loop would blacklist a healthy one."""
    cd = RateLimitCooldown(threshold=10)
    cb, _ = _callback(_entry("baseten/fp8", ("baseten/fp8",)), cooldown=cd)

    class _Exc:
        status_code = 400

        def __str__(self) -> str:
            return "Provider returned error"

    kwargs = _failure_kwargs("baseten/fp8", _Exc())
    await cb.async_log_failure_event(kwargs, None, 0.0, 0.0)
    assert cd.is_broken("z-ai/glm-5.2", "baseten/fp8") is False
    assert cd.is_skipped("z-ai/glm-5.2", "baseten/fp8") is False


async def test_input_cap_patterns_replace_the_builtin_set() -> None:
    """The field is the whole list, not an extension: narrowing it to one signature
    stops the other built-in signature from classifying a 400."""
    cd = RateLimitCooldown(threshold=10)
    cb, _ = _callback(
        _entry("baseten/fp8", ("baseten/fp8",)),
        cooldown=cd,
        input_cap_patterns=("maximum context length",),
    )

    class _Exc:
        status_code = 400

        def __str__(self) -> str:
            return "Input length 562514 exceeds the maximum allowed input length of 524256 tokens"

    kwargs = _failure_kwargs("baseten/fp8", _Exc())
    await cb.async_log_failure_event(kwargs, None, 0.0, 0.0)
    assert cd.is_input_capped("z-ai/glm-5.2", "baseten/fp8") is False


async def test_input_cap_patterns_empty_disables_the_check() -> None:
    cd = RateLimitCooldown(threshold=10)
    cb, _ = _callback(
        _entry("baseten/fp8", ("baseten/fp8",)),
        cooldown=cd,
        input_cap_patterns=(),
    )

    class _Exc:
        status_code = 400

        def __str__(self) -> str:
            return "This model's maximum context length is 200000 tokens"

    kwargs = _failure_kwargs("baseten/fp8", _Exc())
    await cb.async_log_failure_event(kwargs, None, 0.0, 0.0)
    assert cd.is_input_capped("z-ai/glm-5.2", "baseten/fp8") is False


async def test_input_cap_wins_over_broken_when_both_match() -> None:
    """Precedence matters: input cap is a bounded soft skip the request caused, so a body
    matching both must not be promoted to a hard broken-provider exclusion."""
    cd = RateLimitCooldown(threshold=10)
    cb, _ = _callback(
        _entry("baseten/fp8", ("baseten/fp8",)),
        cooldown=cd,
        input_cap_patterns=("maximum context length",),
        broken_provider_patterns=("maximum context length",),
    )

    class _Exc:
        status_code = 400

        def __str__(self) -> str:
            return "This model's maximum context length is 200000 tokens"

    kwargs = _failure_kwargs("baseten/fp8", _Exc())
    await cb.async_log_failure_event(kwargs, None, 0.0, 0.0)
    assert cd.is_input_capped("z-ai/glm-5.2", "baseten/fp8") is True
    assert cd.is_broken("z-ai/glm-5.2", "baseten/fp8") is False


async def test_broken_winner_is_skipped_for_new_request() -> None:
    cd = RateLimitCooldown(threshold=10)
    cb, _ = _callback(
        _entry("baseten/fp8", ("baseten/fp8", "novita/fp8")),
        cooldown=cd,
        wildcard=True,
        broken_provider_patterns=("provider returned error",),
    )
    cd.record_broken("z-ai/glm-5.2", "baseten/fp8", ttl_s=600.0)
    result = await cb.async_filter_deployments("z-ai/glm-5.2", [_wildcard_dep()], None)
    assert _provider_only(result[0]) == "novita/fp8"


async def test_broken_provider_is_a_hard_skip_unlike_input_cap() -> None:
    """An input-capped provider is still tried when nothing else is left; a broken one is
    not. A provider the operator has declared misconfigured will not serve the request,
    so routing to it wastes the call."""
    cd = RateLimitCooldown(threshold=10)
    cb, _ = _callback(
        _entry("baseten/fp8", ("baseten/fp8",)),
        cooldown=cd,
        wildcard=True,
        broken_provider_patterns=("provider returned error",),
    )
    cd.record_broken("z-ai/glm-5.2", "baseten/fp8", ttl_s=600.0)
    with pytest.raises(AllProvidersOnCooldown) as exc:
        await cb.async_filter_deployments("z-ai/glm-5.2", [_wildcard_dep()], None)
    message = str(exc.value)
    assert "marked broken by a 400 pattern" in message
    assert "429 cooldown" not in message


async def test_broken_provider_ttl_from_rule_expires() -> None:
    """The TTL is read off the rule at record time, so a short one lets the provider back
    in without a restart."""
    clock = {"t": 0.0}
    cd = RateLimitCooldown(threshold=10, clock=lambda: clock["t"])
    cb, _ = _callback(
        _entry("baseten/fp8", ("baseten/fp8",)),
        cooldown=cd,
        broken_provider_patterns=("provider returned error",),
        broken_provider_ttl_s=60.0,
    )

    class _Exc:
        status_code = 400

        def __str__(self) -> str:
            return "Provider returned error"

    kwargs = _failure_kwargs("baseten/fp8", _Exc())
    await cb.async_log_failure_event(kwargs, None, 0.0, 0.0)
    assert cd.is_broken("z-ai/glm-5.2", "baseten/fp8") is True
    clock["t"] = 60.1
    assert cd.is_broken("z-ai/glm-5.2", "baseten/fp8") is False


async def test_broken_provider_detected_on_passthrough_failure() -> None:
    """The passthrough path supplies a synthetic exception whose text says nothing, so
    the upstream body in request_data["response_body"] is what the pattern must see."""
    cd = RateLimitCooldown(threshold=10)
    cb, _ = _callback(
        _entry("baseten/fp8", ("baseten/fp8",)),
        cooldown=cd,
        broken_provider_patterns=("invalid provider config",),
    )
    request_data = _post_call_request_data(
        "baseten/fp8",
        response_body={"error": {"message": "Invalid provider config for this model"}},
    )
    await cb.async_post_call_failure_hook(
        request_data, _SyntheticUpstreamExc(400), None
    )
    assert cd.is_broken("z-ai/glm-5.2", "baseten/fp8") is True


async def test_broken_cold_start_fallback_raises_and_names_the_pattern() -> None:
    cd = RateLimitCooldown(threshold=10)
    cb, _ = _callback(
        None,
        wildcard=True,
        cold_start_fallback=("novita/fp8",),
        cooldown=cd,
        broken_provider_patterns=("provider returned error",),
    )
    cd.record_broken("z-ai/glm-5.2", "novita/fp8", ttl_s=600.0)
    with pytest.raises(AllProvidersOnCooldown) as exc:
        await cb.async_filter_deployments("z-ai/glm-5.2", [_wildcard_dep()], None)
    assert "marked broken by a 400 pattern" in str(exc.value)


async def test_broken_winner_records_provider_broken_decision(tmp_path, monkeypatch) -> None:
    """The routing log names the cause an operator must act on, rather than reporting the
    skip as a generic safe_set move."""
    log = tmp_path / "routing.log"
    monkeypatch.setenv("OPENROUTER_PARETO_ROUTING_LOG", str(log))
    rules = {
        "z-ai/glm-5.2": rule(
            precision="fp8",
            min_context=1_000_000,
            min_stats_requests=100,
            wildcard=True,
            log_decisions=True,
            broken_provider_patterns=["provider returned error"],
        )
    }
    cd = RateLimitCooldown(threshold=10)
    cb = OpenRouterParetoCallback(
        rules=rules, telemetry=_StubTelemetry(_entry("baseten/fp8", ("baseten/fp8", "novita/fp8"))), cooldown=cd
    )
    cd.record_broken("z-ai/glm-5.2", "baseten/fp8", ttl_s=600.0)
    await cb.async_filter_deployments("z-ai/glm-5.2", [_wildcard_dep()], None)
    text = log.read_text()
    assert "decision=provider_broken" in text
    assert "slug=novita/fp8" in text
