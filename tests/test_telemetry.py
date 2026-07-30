from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import httpx
import pytest

from litellm_plugin_openrouter_pareto.config import Rule, rule
from litellm_plugin_openrouter_pareto.models import (
    EndpointEntry,
    StatsDataPolicy,
    StatsEndpoint,
    StatsPricing,
    StatsSample,
)
from litellm_plugin_openrouter_pareto.scorer import Candidate, Selection
from litellm_plugin_openrouter_pareto.telemetry import (
    OR_CACHE_VERSION,
    Telemetry,
)


def _glm_rules() -> dict[str, Rule]:
    return {
        "z-ai/glm-5.2": rule(
            precision="fp8",
            min_context=1_000_000,
            min_stats_requests=100,
            promotion_polls=2,
        )
    }


def _glm_rule() -> Rule:
    return _glm_rules()["z-ai/glm-5.2"]


def _candidate(slug: str, tps: float = 50.0, price: float = 1.4) -> Candidate:
    return Candidate(
        slug=slug,
        tag=f"{slug}/fp8",
        input_price_m=price,
        output_price_m=5.6,
        tps=tps,
        requests=500,
        context=1_000_000,
        uptime=99.7,
    )


def _selection(winner_slug: str, safe: tuple[str, ...]) -> Selection:
    return Selection(
        winner=_candidate(winner_slug),
        safe_set=safe,
        candidates=tuple(_candidate(s) for s in safe),
        frontier=safe,
    )


def _telemetry(cache_dir: Path, rules: dict[str, Rule] | None = None) -> Telemetry:
    return Telemetry(rules or _glm_rules(), cache_dir=cache_dir)


def _stat(slug: str, tps: float, prompt: float) -> StatsEndpoint:
    return StatsEndpoint(
        provider_slug=slug,
        quantization="fp8",
        context_length=1_000_000,
        data_policy=StatsDataPolicy(retainsPrompts=False),
        stats=StatsSample(p50_throughput=tps, request_count=500),
        pricing=StatsPricing(prompt=prompt, completion=0.0000056),
    )


class _FakeTelemetry(Telemetry):
    def __init__(
        self,
        cache_dir: Path,
        models: dict[str, str] | None = None,
        endpoints: tuple[EndpointEntry, ...] = (),
        stats: tuple[StatsEndpoint, ...] = (),
        fail: bool = False,
        delay: float = 0.0,
    ) -> None:
        super().__init__(_glm_rules(), cache_dir=cache_dir)
        self._fake_models = models or {"z-ai/glm-5.2": "z-ai/glm-5.2-dated"}
        self._fake_endpoints = endpoints
        self._fake_stats = stats
        self._fail = fail
        self._delay = delay
        self.endpoint_calls = 0

    async def _fetch_models(self) -> dict[str, str]:
        return self._fake_models

    async def _fetch_endpoints(self, or_id: str) -> tuple[EndpointEntry, ...]:
        self.endpoint_calls += 1
        if self._delay:
            await asyncio.sleep(self._delay)
        if self._fail:
            raise httpx.ConnectError("boom")
        return self._fake_endpoints

    async def _fetch_stats(self, permaslug: str) -> tuple[StatsEndpoint, ...]:
        return self._fake_stats


def test_apply_promotion_first_poll_promotes(tmp_path: Path) -> None:
    t = _telemetry(tmp_path)
    sel = _selection("baseten", ("baseten", "novita"))
    entry = t._apply_promotion(None, sel, _glm_rule(), "slug", 1.0, 2.0)
    assert entry is not None
    assert entry.winner == "baseten"
    assert entry.candidate_streak == 1
    assert entry.stale is False


def test_apply_promotion_evicts_immediately_when_winner_not_in_safe_set(tmp_path: Path) -> None:
    t = _telemetry(tmp_path)
    previous = t._apply_promotion(None, _selection("baseten", ("baseten", "novita")), _glm_rule(), "slug", 1.0, 1.0)
    assert previous is not None
    sel = _selection("novita", ("novita", "siliconflow"))
    entry = t._apply_promotion(previous, sel, _glm_rule(), "slug", 1.0, 2.0)
    assert entry is not None
    assert entry.winner == "novita"
    assert entry.candidate_streak == 1


def test_apply_promotion_debounces_value_change_until_streak_threshold(tmp_path: Path) -> None:
    t = _telemetry(tmp_path)
    rule_obj = _glm_rules()["z-ai/glm-5.2"]
    previous = t._apply_promotion(None, _selection("novita", ("novita", "baseten")), rule_obj, "slug", 1.0, 1.0)
    assert previous is not None
    new_sel = _selection("baseten", ("novita", "baseten"))
    first = t._apply_promotion(previous, new_sel, rule_obj, "slug", 1.0, 2.0)
    assert first is not None
    assert first.winner == "novita"
    assert first.candidate_streak == 1
    second = t._apply_promotion(first, new_sel, rule_obj, "slug", 1.0, 3.0)
    assert second is not None
    assert second.winner == "baseten"
    assert second.candidate_streak == 2


def test_apply_promotion_keeps_winner_when_same(tmp_path: Path) -> None:
    t = _telemetry(tmp_path)
    rule_obj = _glm_rules()["z-ai/glm-5.2"]
    previous = t._apply_promotion(None, _selection("novita", ("novita", "baseten")), rule_obj, "slug", 1.0, 1.0)
    assert previous is not None
    same = t._apply_promotion(previous, _selection("novita", ("novita", "baseten")), rule_obj, "slug", 1.0, 2.0)
    assert same is not None
    assert same.winner == "novita"
    assert same.candidate_streak == 2


def test_apply_promotion_empty_selection_returns_none(tmp_path: Path) -> None:
    t = _telemetry(tmp_path)
    empty = Selection(winner=None, safe_set=(), candidates=(), frontier=())
    assert t._apply_promotion(None, empty, _glm_rule(), "slug", 1.0, 2.0) is None


async def test_get_cold_start_fetches_and_returns_winner(tmp_path: Path) -> None:
    t = _FakeTelemetry(
        tmp_path,
        endpoints=(EndpointEntry(tag="novita/fp8", uptime_last_5m=99.7),),
        stats=(_stat("novita/fp8", 50.0, 0.000000623),),
    )
    try:
        entry = await t.get("z-ai/glm-5.2")
    finally:
        await t.aclose()
    assert entry is not None
    assert entry.winner == "novita"
    assert entry.stale is False


async def test_get_dedupes_concurrent_refreshes_to_one_fetch(tmp_path: Path) -> None:
    t = _FakeTelemetry(
        tmp_path,
        endpoints=(EndpointEntry(tag="novita/fp8", uptime_last_5m=99.7),),
        stats=(_stat("novita/fp8", 50.0, 0.000000623),),
        delay=0.1,
    )
    try:
        results = await asyncio.gather(t.get("z-ai/glm-5.2"), t.get("z-ai/glm-5.2"))
    finally:
        await t.aclose()
    assert t.endpoint_calls == 1
    assert all(r is not None and r.winner == "novita" for r in results)


async def test_refresh_failure_marks_existing_stale(tmp_path: Path) -> None:
    t = _FakeTelemetry(
        tmp_path,
        endpoints=(EndpointEntry(tag="novita/fp8", uptime_last_5m=99.7),),
        stats=(_stat("novita/fp8", 50.0, 0.000000623),),
    )
    try:
        good = await t.get("z-ai/glm-5.2")
        assert good is not None and good.winner == "novita"
        from dataclasses import replace

        t._memory["z-ai/glm-5.2"] = replace(good, fetched_at=0.0)
        t._fake_stats = ()
        t._fail = True
        stale = await t.get("z-ai/glm-5.2")
    finally:
        await t.aclose()
    assert stale is not None
    assert stale.stale is True
    assert stale.winner == "novita"


async def test_cold_start_timeout_returns_none(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import litellm_plugin_openrouter_pareto.telemetry as telmod

    monkeypatch.setattr(telmod, "OR_COLD_START_TIMEOUT_S", 0.05)
    t = _FakeTelemetry(
        tmp_path,
        endpoints=(EndpointEntry(tag="novita/fp8", uptime_last_5m=99.7),),
        stats=(_stat("novita/fp8", 50.0, 0.000000623),),
        delay=0.5,
    )
    start = time.monotonic()
    try:
        entry = await t.get("z-ai/glm-5.2")
    finally:
        await t.aclose()
    elapsed = time.monotonic() - start
    assert entry is None
    assert elapsed < 0.3


def test_version_mismatch_discards_disk_cache(tmp_path: Path) -> None:
    (tmp_path / "cache.json").write_text(
        json.dumps(
            {
                "version": OR_CACHE_VERSION - 1,
                "entries": {
                    "z-ai/glm-5.2": {
                        "winner": "novita",
                        "candidate_winner": "novita",
                        "candidate_streak": 1,
                        "safe_set": ["novita"],
                        "canonical_slug": "slug",
                        "canonical_slug_fetched_at": 0.0,
                        "fetched_at": 0.0,
                        "stale": False,
                    }
                },
            }
        )
    )
    t = _telemetry(tmp_path)
    assert "z-ai/glm-5.2" not in t._memory


def test_persist_then_reload_round_trip(tmp_path: Path) -> None:
    t = _telemetry(tmp_path)
    sel = _selection("baseten", ("baseten", "novita"))
    entry = t._apply_promotion(None, sel, _glm_rule(), "slug", 1.0, 2.0)
    assert entry is not None
    t._memory["z-ai/glm-5.2"] = entry
    t._persist()
    reloaded = _telemetry(tmp_path)
    entry = reloaded._memory["z-ai/glm-5.2"]
    assert entry.winner == "baseten"
    assert entry.safe_set == ("baseten", "novita")
    assert entry.stale is False


async def test_canonical_slug_reused_within_ttl_without_fetch(tmp_path: Path) -> None:
    t = _FakeTelemetry(tmp_path)
    now = time.time()
    from litellm_plugin_openrouter_pareto.telemetry import CacheEntry

    previous = CacheEntry(
        winner="novita",
        candidate_winner="novita",
        candidate_streak=1,
        safe_set=("novita",),
        canonical_slug="cached-slug",
        canonical_slug_fetched_at=now,
        fetched_at=now,
        stale=False,
    )
    t._memory["z-ai/glm-5.2"] = previous
    t._fake_models = {}
    slug, fetched_at = await t._resolve_canonical_slug("z-ai/glm-5.2", previous, now)
    await t.aclose()
    assert slug == "cached-slug"
    assert fetched_at == now
    assert t._models_map is None
