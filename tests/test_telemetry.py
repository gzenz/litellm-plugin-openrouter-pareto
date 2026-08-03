from __future__ import annotations

import asyncio
import json
import ssl
import time
from pathlib import Path

import httpx
import pytest

from litellm_plugin_openrouter_pareto.config import Rule, rule, telemetry_verify
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
    sel = _selection("baseten/fp8", ("baseten/fp8", "novita/fp8"))
    entry = t._apply_promotion(None, sel, _glm_rule(), "slug", 1.0, 2.0)
    assert entry is not None
    assert entry.winner == "baseten/fp8"
    assert entry.candidate_streak == 1
    assert entry.stale is False


def test_apply_promotion_evicts_immediately_when_winner_not_in_safe_set(tmp_path: Path) -> None:
    t = _telemetry(tmp_path)
    previous = t._apply_promotion(
        None, _selection("baseten/fp8", ("baseten/fp8", "novita/fp8")), _glm_rule(), "slug", 1.0, 1.0
    )
    assert previous is not None
    sel = _selection("novita/fp8", ("novita/fp8", "siliconflow/fp8"))
    entry = t._apply_promotion(previous, sel, _glm_rule(), "slug", 1.0, 2.0)
    assert entry is not None
    assert entry.winner == "novita/fp8"
    assert entry.candidate_streak == 1


def test_apply_promotion_debounces_value_change_until_streak_threshold(tmp_path: Path) -> None:
    t = _telemetry(tmp_path)
    rule_obj = _glm_rules()["z-ai/glm-5.2"]
    previous = t._apply_promotion(
        None, _selection("novita/fp8", ("novita/fp8", "baseten/fp8")), rule_obj, "slug", 1.0, 1.0
    )
    assert previous is not None
    new_sel = _selection("baseten/fp8", ("novita/fp8", "baseten/fp8"))
    first = t._apply_promotion(previous, new_sel, rule_obj, "slug", 1.0, 2.0)
    assert first is not None
    assert first.winner == "novita/fp8"
    assert first.candidate_streak == 1
    second = t._apply_promotion(first, new_sel, rule_obj, "slug", 1.0, 3.0)
    assert second is not None
    assert second.winner == "baseten/fp8"
    assert second.candidate_streak == 2


def test_apply_promotion_keeps_winner_when_same(tmp_path: Path) -> None:
    t = _telemetry(tmp_path)
    rule_obj = _glm_rules()["z-ai/glm-5.2"]
    previous = t._apply_promotion(
        None, _selection("novita/fp8", ("novita/fp8", "baseten/fp8")), rule_obj, "slug", 1.0, 1.0
    )
    assert previous is not None
    same = t._apply_promotion(
        previous, _selection("novita/fp8", ("novita/fp8", "baseten/fp8")), rule_obj, "slug", 1.0, 2.0
    )
    assert same is not None
    assert same.winner == "novita/fp8"
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
    assert entry.winner == "novita/fp8"
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
    assert all(r is not None and r.winner == "novita/fp8" for r in results)


async def test_refresh_failure_marks_existing_stale(tmp_path: Path) -> None:
    t = _FakeTelemetry(
        tmp_path,
        endpoints=(EndpointEntry(tag="novita/fp8", uptime_last_5m=99.7),),
        stats=(_stat("novita/fp8", 50.0, 0.000000623),),
    )
    try:
        good = await t.get("z-ai/glm-5.2")
        assert good is not None and good.winner == "novita/fp8"
        from dataclasses import replace

        t._memory["z-ai/glm-5.2"] = replace(good, fetched_at=0.0)
        t._fake_stats = ()
        t._fail = True
        stale = await t.get("z-ai/glm-5.2")
    finally:
        await t.aclose()
    assert stale is not None
    assert stale.stale is True
    assert stale.winner == "novita/fp8"


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
    t._ensure_disk()
    assert "z-ai/glm-5.2" not in t._memory


def test_persist_then_reload_round_trip(tmp_path: Path) -> None:
    t = _telemetry(tmp_path)
    sel = _selection("baseten/fp8", ("baseten/fp8", "novita/fp8"))
    entry = t._apply_promotion(None, sel, _glm_rule(), "slug", 1.0, 2.0)
    assert entry is not None
    t._memory["z-ai/glm-5.2"] = entry
    t._persist(tuple(t._memory.items()), now=0.0)
    reloaded = _telemetry(tmp_path)
    reloaded._ensure_disk()
    entry = reloaded._memory["z-ai/glm-5.2"]
    assert entry.winner == "baseten/fp8"
    assert entry.safe_set == ("baseten/fp8", "novita/fp8")
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


def test_merge_preserves_other_instance_entries(tmp_path: Path) -> None:
    shared = {"model-a": _glm_rule(), "model-b": _glm_rule()}
    a = _telemetry(tmp_path, shared)
    b = _telemetry(tmp_path, shared)
    sel_a = _selection("baseten/fp8", ("baseten/fp8",))
    sel_b = _selection("novita/fp8", ("novita/fp8",))
    entry_a = a._apply_promotion(None, sel_a, _glm_rule(), "slug", 1.0, 1.0)
    entry_b = b._apply_promotion(None, sel_b, _glm_rule(), "slug", 1.0, 2.0)
    assert entry_a is not None and entry_b is not None
    a._memory["model-a"] = entry_a
    b._memory["model-b"] = entry_b
    a._persist(tuple(a._memory.items()), now=0.0)
    b._persist(tuple(b._memory.items()), now=0.0)
    c = _telemetry(tmp_path, shared)
    c._ensure_disk()
    assert "model-a" in c._memory
    assert "model-b" in c._memory


def test_merge_keeps_newer_entry_per_model(tmp_path: Path) -> None:
    shared = {"model-x": _glm_rule()}
    a = _telemetry(tmp_path, shared)
    b = _telemetry(tmp_path, shared)
    older = a._apply_promotion(None, _selection("novita/fp8", ("novita/fp8",)), _glm_rule(), "slug", 1.0, 1.0)
    newer = b._apply_promotion(None, _selection("baseten/fp8", ("baseten/fp8",)), _glm_rule(), "slug", 1.0, 5.0)
    assert older is not None and newer is not None
    a._memory["model-x"] = older
    a._persist(tuple(a._memory.items()), now=0.0)
    b._memory["model-x"] = newer
    b._persist(tuple(b._memory.items()), now=0.0)
    c = _telemetry(tmp_path, shared)
    c._ensure_disk()
    assert c._memory["model-x"].winner == "baseten/fp8"
    assert c._memory["model-x"].fetched_at == 5.0


def _persist_in_child(cache_dir: str, model: str, winner: str, fetched_at: float, barrier_path: str) -> None:
    """Runs in a separate PROCESS: wait for the barrier file, then persist one model."""
    import time as _time

    from litellm_plugin_openrouter_pareto.config import rule as _rule
    from litellm_plugin_openrouter_pareto.telemetry import Telemetry as _Telemetry

    r = _rule(precision="fp8", min_context=1_000_000, min_stats_requests=100, promotion_polls=2)
    t = _Telemetry({model: r}, cache_dir=Path(cache_dir))
    entry = t._apply_promotion(None, _selection(winner, (winner,)), r, "slug", 1.0, fetched_at)
    assert entry is not None
    t._memory[model] = entry
    for _ in range(500):
        if Path(barrier_path).exists():
            break
        _time.sleep(0.01)
    t._persist(tuple(t._memory.items()), now=0.0)


def test_two_processes_persisting_concurrently_preserve_both_entries(tmp_path: Path) -> None:
    """Real cross-process test: two OS processes race to persist different models.
    Without the file lock + read-merge-write, one worker's entry is lost."""
    import multiprocessing as mp

    barrier = tmp_path / "go"
    ctx = mp.get_context("spawn")
    procs = [
        ctx.Process(target=_persist_in_child, args=(str(tmp_path), "model-a", "baseten/fp8", 1.0, str(barrier))),
        ctx.Process(target=_persist_in_child, args=(str(tmp_path), "model-b", "novita/fp8", 2.0, str(barrier))),
    ]
    for pr in procs:
        pr.start()
    barrier.write_text("go")
    for pr in procs:
        pr.join(timeout=60)
    assert all(pr.exitcode == 0 for pr in procs), [pr.exitcode for pr in procs]

    reader = _telemetry(tmp_path, {"model-a": _glm_rule(), "model-b": _glm_rule()})
    reader._ensure_disk()
    assert "model-a" in reader._memory, "lost the first worker's entry"
    assert "model-b" in reader._memory, "lost the second worker's entry"


def test_persist_survives_lock_held_by_another_process(tmp_path: Path) -> None:
    """A lock timeout must degrade without corrupting the existing cache."""
    from filelock import FileLock

    shared = {"model-seed": _glm_rule(), "model-blocked": _glm_rule()}
    seed = _telemetry(tmp_path, shared)
    entry = seed._apply_promotion(
        None, _selection("baseten/fp8", ("baseten/fp8",)), _glm_rule(), "slug", 1.0, 1.0
    )
    assert entry is not None
    seed._memory["model-seed"] = entry
    seed._persist(tuple(seed._memory.items()), now=0.0)

    blocked = _telemetry(tmp_path, shared)
    other = blocked._apply_promotion(
        None, _selection("novita/fp8", ("novita/fp8",)), _glm_rule(), "slug", 1.0, 9.0
    )
    assert other is not None
    blocked._memory["model-blocked"] = other
    held = FileLock(str(tmp_path / "cache.lock"), timeout=1)
    held.acquire()
    try:
        blocked._persist(tuple(blocked._memory.items()), now=0.0)
    finally:
        held.release()

    reader = _telemetry(tmp_path, shared)
    reader._ensure_disk()
    assert reader._memory["model-seed"].winner == "baseten/fp8", "pre-existing cache was corrupted"


def _rule_excluding(regions: list[str]) -> Rule:
    return rule(
        precision="fp8",
        min_context=1_000_000,
        min_stats_requests=100,
        promotion_polls=2,
        exclude_regions=regions,
    )


def test_stricter_rule_rejects_cached_winner_from_permissive_process(tmp_path: Path) -> None:
    """The shared cache must not let a permissive worker's winner leak into a
    worker configured with a stricter exclude_regions."""
    permissive = _rule_excluding([])
    strict = _rule_excluding(["US"])

    writer = Telemetry({"z-ai/glm-5.2": permissive}, cache_dir=tmp_path)
    entry = writer._apply_promotion(
        None, _selection("baseten/fp8", ("baseten/fp8",)), permissive, "slug", 1.0, 1.0
    )
    assert entry is not None
    writer._memory["z-ai/glm-5.2"] = entry
    writer._persist(tuple(writer._memory.items()), now=0.0)

    same = Telemetry({"z-ai/glm-5.2": permissive}, cache_dir=tmp_path)
    same._ensure_disk()
    assert "z-ai/glm-5.2" in same._memory

    stricter = Telemetry({"z-ai/glm-5.2": strict}, cache_dir=tmp_path)
    stricter._ensure_disk()
    assert "z-ai/glm-5.2" not in stricter._memory, "stricter rule reused a permissive winner"


def test_persist_preserves_same_model_entry_written_under_a_different_rule(tmp_path: Path) -> None:
    """Rule-filtering happens on read into memory, but persisting must merge against
    the RAW disk state. Otherwise a strict worker silently deletes the permissive
    worker's entry for the same model, and the permissive worker loses its cache."""
    permissive = _rule_excluding([])
    strict = _rule_excluding(["US"])
    model = "z-ai/glm-5.2"

    writer = Telemetry({model: permissive}, cache_dir=tmp_path)
    e1 = writer._apply_promotion(
        None, _selection("baseten/fp8", ("baseten/fp8",)), permissive, "slug", 1.0, 1.0
    )
    assert e1 is not None
    writer._memory[model] = e1
    writer._persist(tuple(writer._memory.items()), now=0.0)
    permissive_fp = e1.rule_fp

    strict_worker = Telemetry({model: strict, "other/model": strict}, cache_dir=tmp_path)
    strict_worker._ensure_disk()
    assert model not in strict_worker._memory, "strict worker reused a permissive winner"
    e2 = strict_worker._apply_promotion(
        None, _selection("novita/fp8", ("novita/fp8",)), strict, "slug", 1.0, 2.0
    )
    assert e2 is not None
    strict_worker._memory["other/model"] = e2
    strict_worker._persist(tuple(strict_worker._memory.items()), now=0.0)

    raw = {k.split("\x00")[0]: v for k, v in Telemetry({}, cache_dir=tmp_path)._load_disk()}
    assert "other/model" in raw
    assert model in raw, "strict worker clobbered the permissive worker's entry"
    assert raw[model].rule_fp == permissive_fp, "permissive entry was overwritten"


def test_rule_fingerprint_ignores_non_scoring_fields() -> None:
    """cold_start_fallback / wildcard / log_errors do not change selection, so they
    must not invalidate a cached winner."""
    from litellm_plugin_openrouter_pareto.config import rule_fingerprint

    base = rule(precision="fp8", min_context=1_000_000, min_stats_requests=100)
    same = rule(
        precision="fp8",
        min_context=1_000_000,
        min_stats_requests=100,
        cold_start_fallback=["x/fp8"],
        wildcard=True,
        log_errors=True,
    )
    differs = rule(
        precision="fp8", min_context=1_000_000, min_stats_requests=100, exclude_regions=["US"]
    )
    assert rule_fingerprint(base) == rule_fingerprint(same)
    assert rule_fingerprint(base) != rule_fingerprint(differs)


def test_same_model_under_two_rules_keeps_both_cache_entries(tmp_path: Path) -> None:
    """The collision that matters: two workers cache the SAME model under different
    policies. Keyed by model alone they evict each other every refresh; keyed by
    (model, rule_fp) both survive and each worker reloads its own."""
    permissive = _rule_excluding([])
    strict = _rule_excluding(["US"])
    model = "z-ai/glm-5.2"

    worker_a = Telemetry({model: permissive}, cache_dir=tmp_path)
    ea = worker_a._apply_promotion(
        None, _selection("baseten/fp8", ("baseten/fp8",)), permissive, "slug", 1.0, 1.0
    )
    assert ea is not None
    worker_a._memory[model] = ea
    worker_a._persist(tuple(worker_a._memory.items()), now=0.0)

    worker_b = Telemetry({model: strict}, cache_dir=tmp_path)
    eb = worker_b._apply_promotion(
        None, _selection("novita/fp8", ("novita/fp8",)), strict, "slug", 1.0, 2.0
    )
    assert eb is not None
    worker_b._memory[model] = eb
    worker_b._persist(tuple(worker_b._memory.items()), now=0.0)

    reload_a = Telemetry({model: permissive}, cache_dir=tmp_path)
    reload_a._ensure_disk()
    assert reload_a._memory[model].winner == "baseten/fp8", "permissive worker lost its cache"

    reload_b = Telemetry({model: strict}, cache_dir=tmp_path)
    reload_b._ensure_disk()
    assert reload_b._memory[model].winner == "novita/fp8", "strict worker lost its cache"


def test_all_excluded_selection_is_still_cached_as_evidence(tmp_path: Path) -> None:
    """A successful fetch that excludes every provider is real evidence. Caching it
    winner-less keeps 'telemetry proved everything is excluded' distinguishable from
    'telemetry never answered', which is what the routing policy branches on."""
    from litellm_plugin_openrouter_pareto.scorer import Selection

    r = _rule_excluding(["US"])
    t = _telemetry(tmp_path, {"z-ai/glm-5.2": r})
    empty_but_informative = Selection(
        winner=None,
        safe_set=(),
        candidates=(),
        frontier=(),
        excluded_bases=("baseten", "novita"),
        allowed_bases=(),
    )
    entry = t._apply_promotion(None, empty_but_informative, r, "slug", 1.0, 5.0)
    assert entry is not None, "all-excluded result was discarded instead of cached"
    assert entry.winner is None
    assert entry.excluded_bases == ("baseten", "novita")
    assert entry.stale is False


def test_empty_selection_without_region_policy_is_not_cached(tmp_path: Path) -> None:
    """Without a region policy an empty selection carries no evidence worth caching."""
    from litellm_plugin_openrouter_pareto.scorer import Selection

    r = _glm_rule()
    t = _telemetry(tmp_path)
    empty = Selection(winner=None, safe_set=(), candidates=(), frontier=())
    assert t._apply_promotion(None, empty, r, "slug", 1.0, 5.0) is None


async def test_stale_fallback_returns_without_waiting_for_the_cache_lock(tmp_path: Path) -> None:
    """The disk cache is an optimization, so a contended cross-process lock must not
    sit in the request path. Awaiting a persist thread does not remove that wait from
    request latency; it only changes which thread blocks. FileLock's timeout is 5s, so
    anything close to that means the caller paid for the write."""
    from filelock import FileLock

    rules = {"model-a": _glm_rule()}
    telemetry = _telemetry(tmp_path, rules)
    entry = telemetry._apply_promotion(
        None, _selection("baseten/fp8", ("baseten/fp8",)), _glm_rule(), "slug", 1.0, 1.0
    )
    assert entry is not None
    telemetry._memory["model-a"] = entry

    held = FileLock(str(tmp_path / "cache.lock"), timeout=10)
    held.acquire()
    try:
        started = time.monotonic()
        result = await telemetry._stale_fallback("model-a", entry)
        elapsed = time.monotonic() - started
    finally:
        held.release()

    assert result is not None and result.stale is True
    assert elapsed < 1.0, f"stale fallback blocked {elapsed:.2f}s on the cache lock"


async def test_stale_fallback_still_persists_once_the_lock_frees(tmp_path: Path) -> None:
    """Backgrounding the write must not mean losing it."""
    rules = {"model-a": _glm_rule()}
    telemetry = _telemetry(tmp_path, rules)
    stamp = time.time()
    entry = telemetry._apply_promotion(
        None, _selection("baseten/fp8", ("baseten/fp8",)), _glm_rule(), "slug", stamp, stamp
    )
    assert entry is not None
    telemetry._memory["model-a"] = entry

    await telemetry._stale_fallback("model-a", entry)
    for task in tuple(telemetry._persist_tasks):
        await task

    reader = _telemetry(tmp_path, rules)
    reader._ensure_disk()
    assert reader._memory["model-a"].stale is True


def test_late_fresh_write_cannot_resurrect_a_stale_entry(tmp_path: Path) -> None:
    """Background writes can land out of order, and marking an entry stale does not
    move fetched_at. Merging on fetch time alone lets an older in-flight fresh snapshot
    undo a failed refresh, so the disk would claim a refresh succeeded when it did not."""
    rules = {"model-a": _glm_rule()}
    telemetry = _telemetry(tmp_path, rules)
    fresh = telemetry._apply_promotion(
        None, _selection("baseten/fp8", ("baseten/fp8",)), _glm_rule(), "slug", 1.0, 100.0
    )
    assert fresh is not None
    stale = telemetry._mark_stale(fresh)

    telemetry._persist((("model-a", stale),), now=0.0)
    telemetry._persist((("model-a", fresh),), now=0.0)

    reader = _telemetry(tmp_path, rules)
    reader._ensure_disk()
    assert reader._memory["model-a"].stale is True


def test_a_failing_worker_cannot_outrank_another_workers_newer_fetch(tmp_path: Path) -> None:
    """The cache file is shared across worker processes. Ordering by any process-local
    counter lets a worker that merely failed repeatedly outrank a different worker's
    genuinely newer telemetry, so the pool would keep serving the older winner."""
    rules = {"model-a": _glm_rule()}
    baseline = _telemetry(tmp_path, rules)
    original = baseline._apply_promotion(
        None, _selection("a/fp8", ("a/fp8",)), _glm_rule(), "slug", 1.0, 100.0
    )
    assert original is not None

    failing = _telemetry(tmp_path, rules)
    state = original
    for _ in range(5):
        state = failing._mark_stale(state)
        failing._persist((("model-a", state),), now=0.0)

    succeeding = _telemetry(tmp_path, rules)
    fresher = succeeding._apply_promotion(
        original, _selection("b/fp8", ("b/fp8",)), _glm_rule(), "slug", 1.0, 200.0
    )
    assert fresher is not None
    succeeding._persist((("model-a", fresher),), now=0.0)

    reader = _telemetry(tmp_path, rules)
    reader._ensure_disk()
    assert reader._memory["model-a"].winner == "b/fp8"
    assert reader._memory["model-a"].stale is False


def test_a_real_refresh_still_supersedes_a_stale_entry(tmp_path: Path) -> None:
    """The ordering fix must not freeze a model in the stale state: a genuinely newer
    refresh has a higher revision and has to win."""
    rules = {"model-a": _glm_rule()}
    telemetry = _telemetry(tmp_path, rules)
    first = telemetry._apply_promotion(
        None, _selection("baseten/fp8", ("baseten/fp8",)), _glm_rule(), "slug", 1.0, 100.0
    )
    assert first is not None
    stale = telemetry._mark_stale(first)
    telemetry._persist((("model-a", stale),), now=0.0)

    recovered = telemetry._apply_promotion(
        stale, _selection("novita/fp8", ("novita/fp8",)), _glm_rule(), "slug", 1.0, 200.0
    )
    assert recovered is not None
    telemetry._persist((("model-a", recovered),), now=0.0)

    reader = _telemetry(tmp_path, rules)
    reader._ensure_disk()
    assert reader._memory["model-a"].stale is False
    assert reader._memory["model-a"].winner == "novita/fp8"


async def test_aclose_drains_pending_persistence(tmp_path: Path) -> None:
    """A queued write carries the winner, stale marker, or region evidence. Shutdown
    must not drop it; production never awaits the private task set the way tests can."""
    rules = {"model-a": _glm_rule()}
    telemetry = _telemetry(tmp_path, rules)
    stamp = time.time()
    entry = telemetry._apply_promotion(
        None, _selection("baseten/fp8", ("baseten/fp8",)), _glm_rule(), "slug", stamp, stamp
    )
    assert entry is not None
    telemetry._memory["model-a"] = telemetry._mark_stale(entry)
    telemetry._persist_soon()

    await telemetry.aclose()

    reader = _telemetry(tmp_path, rules)
    reader._ensure_disk()
    assert "model-a" in reader._memory, "shutdown dropped a queued write"
    assert reader._memory["model-a"].stale is True


async def test_aclose_does_not_hang_on_an_in_flight_refresh(tmp_path: Path) -> None:
    """Draining must be bounded: a hung upstream must not hold shutdown open, since a
    refresh result is only an optimization."""
    rules = {"z-ai/glm-5.2": _glm_rule()}
    telemetry = _FakeTelemetry(
        tmp_path,
        endpoints=(EndpointEntry(tag="novita/fp8", uptime_last_5m=99.7),),
        stats=(_stat("novita/fp8", 50.0, 0.000000623),),
        delay=5.0,
    )
    telemetry._rules = rules
    telemetry._spawn_refresh("z-ai/glm-5.2")
    await asyncio.sleep(0)

    started = time.monotonic()
    await telemetry.aclose()
    elapsed = time.monotonic() - started
    assert elapsed < 1.0, f"aclose blocked {elapsed:.2f}s on an in-flight refresh"


async def test_get_serves_warm_disk_entry_without_spawning_a_refresh(tmp_path: Path) -> None:
    """A restarted worker must serve the last good on-disk selection immediately. If
    get() reads _memory before loading disk, the fast path can never hit and every
    first request per model eats the cold-start path - defeating the persisted cache."""
    writer = _FakeTelemetry(
        tmp_path,
        endpoints=(EndpointEntry(tag="novita/fp8", uptime_last_5m=99.7),),
        stats=(_stat("novita/fp8", 50.0, 0.000000623),),
    )
    warmed = await writer.get("z-ai/glm-5.2")
    assert warmed is not None and warmed.winner is not None
    for task in tuple(writer._persist_tasks):
        await task

    fresh_process = _FakeTelemetry(
        tmp_path,
        endpoints=(EndpointEntry(tag="novita/fp8", uptime_last_5m=99.7),),
        stats=(_stat("novita/fp8", 50.0, 0.000000623),),
    )
    spawned: list[str] = []
    original_spawn = fresh_process._spawn_refresh
    fresh_process._spawn_refresh = lambda m: (spawned.append(m), original_spawn(m))[1]  # pyright: ignore[reportAttributeAccessIssue]  # test spy on private method
    result = await fresh_process.get("z-ai/glm-5.2")
    assert result is not None and result.winner == warmed.winner
    assert fresh_process.endpoint_calls == 0, "warm disk entry should not trigger a network refresh"
    assert spawned == [], "warm disk entry took the cold-start path instead of the fast path"


def test_persist_prunes_disk_entries_abandoned_by_rule_edits(tmp_path: Path) -> None:
    """Editing a rule orphans its old (model, fingerprint) key. Without pruning the
    disk file grows without bound across rule churn and is re-parsed whole on every
    persist. An entry no worker has refreshed within the retention window is dropped."""
    import litellm_plugin_openrouter_pareto.telemetry as telmod

    model = "z-ai/glm-5.2"
    old_rule = rule(precision="fp8", min_context=1_000_000, min_stats_requests=100)
    fresh_stamp = time.time()
    stale_stamp = fresh_stamp - telmod.OR_DISK_RETENTION_S - 1.0

    writer = Telemetry({model: old_rule}, cache_dir=tmp_path)
    orphan = writer._apply_promotion(
        None, _selection("baseten/fp8", ("baseten/fp8",)), old_rule, "slug", stale_stamp, stale_stamp
    )
    assert orphan is not None
    writer._memory[model] = orphan
    writer._persist(tuple(writer._memory.items()), now=stale_stamp)

    new_rule = rule(precision="fp8", min_context=2_000_000, min_stats_requests=100)
    assert new_rule != old_rule
    editor = Telemetry({model: new_rule}, cache_dir=tmp_path)
    current = editor._apply_promotion(
        None, _selection("novita/fp8", ("novita/fp8",)), new_rule, "slug", fresh_stamp, fresh_stamp
    )
    assert current is not None
    editor._memory[model] = current
    editor._persist(tuple(editor._memory.items()), now=fresh_stamp)

    raw = Telemetry({}, cache_dir=tmp_path)._load_disk()
    assert len(raw) == 1, f"stale fingerprint was not pruned: {[k for k, _ in raw]}"


def test_persist_keeps_a_live_entry_from_another_worker(tmp_path: Path) -> None:
    """Pruning is by age, not fingerprint ownership: a different worker's entry for the
    same model under a different rule, refreshed recently, must survive."""
    model = "z-ai/glm-5.2"
    permissive = rule(precision="fp8", min_context=1_000_000, min_stats_requests=100)
    strict = rule(precision="fp8", min_context=1_000_000, min_stats_requests=100, exclude_regions=["US"])
    assert permissive != strict
    stamp = time.time()

    worker_a = Telemetry({model: permissive}, cache_dir=tmp_path)
    ea = worker_a._apply_promotion(
        None, _selection("baseten/fp8", ("baseten/fp8",)), permissive, "slug", stamp, stamp
    )
    assert ea is not None
    worker_a._memory[model] = ea
    worker_a._persist(tuple(worker_a._memory.items()), now=stamp)

    worker_b = Telemetry({model: strict}, cache_dir=tmp_path)
    eb = worker_b._apply_promotion(
        None, _selection("novita/fp8", ("novita/fp8",)), strict, "slug", stamp, stamp
    )
    assert eb is not None
    worker_b._memory[model] = eb
    worker_b._persist(tuple(worker_b._memory.items()), now=stamp + 60.0)

    raw = Telemetry({}, cache_dir=tmp_path)._load_disk()
    assert len(raw) == 2, "a live cross-worker entry was pruned"


class _SpyClient:
    """Stand-in for httpx.AsyncClient that records the kwargs it was built with.
    Avoids opening a real connection pool so a bad CA path cannot fail the build."""

    is_closed = False

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs

    async def aclose(self) -> None:
        self.is_closed = True


def _spy_asyncclient(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
    """Record every httpx.AsyncClient(...) construction as a kwargs dict in order."""
    builds: list[dict[str, object]] = []

    def _build(**kwargs: object) -> _SpyClient:
        builds.append(kwargs)
        return _SpyClient(**kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", _build)
    return builds


def test_client_defaults_to_verifying_tls(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    builds = _spy_asyncclient(monkeypatch)
    t = _telemetry(tmp_path)
    assert t._client_or_create() is not None
    assert builds[-1]["verify"] is True


def test_client_passes_verify_false_when_disabled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    builds = _spy_asyncclient(monkeypatch)
    t = Telemetry(_glm_rules(), cache_dir=tmp_path, verify=False)
    assert t._client_or_create() is not None
    assert builds[-1]["verify"] is False


def test_client_passes_custom_ca_as_ssl_context(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A CA-bundle path is converted to an SSLContext (not the deprecated verify=<str>
    form) before reaching httpx, so a bad path raises inside the refresh path rather
    than at client construction."""
    builds = _spy_asyncclient(monkeypatch)
    sentinel = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)

    def _ctx(**_kwargs: object) -> ssl.SSLContext:
        return sentinel

    monkeypatch.setattr(ssl, "create_default_context", _ctx)
    t = Telemetry(_glm_rules(), cache_dir=tmp_path, verify="/etc/ssl/corp-ca.pem")
    assert t._client_or_create() is not None
    assert builds[-1]["verify"] is sentinel


def test_ssl_verify_returns_bool_unchanged(tmp_path: Path) -> None:
    t_false = Telemetry(_glm_rules(), cache_dir=tmp_path, verify=False)
    assert t_false._ssl_verify() is False
    t_true = Telemetry(_glm_rules(), cache_dir=tmp_path, verify=True)
    assert t_true._ssl_verify() is True


def test_ssl_verify_converts_ca_path_to_context(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def _fake_ctx(**kwargs: object) -> ssl.SSLContext:
        captured.update(kwargs)
        return ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)

    monkeypatch.setattr(ssl, "create_default_context", _fake_ctx)
    t = Telemetry(_glm_rules(), cache_dir=tmp_path, verify="/some/ca.pem")
    ctx = t._ssl_verify()
    assert isinstance(ctx, ssl.SSLContext)
    assert captured == {"cafile": "/some/ca.pem"}


def test_ssl_verify_bad_path_raises(tmp_path: Path) -> None:
    """A missing CA file raises here (inside _client_or_create, inside the refresh
    try/except), so it degrades to a telemetry warning instead of reaching the wire."""
    t = Telemetry(_glm_rules(), cache_dir=tmp_path, verify="/no/such/ca.pem")
    with pytest.raises(OSError):
        t._ssl_verify()


def test_client_rebuilds_after_close_uses_same_verify(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    builds = _spy_asyncclient(monkeypatch)
    t = Telemetry(_glm_rules(), cache_dir=tmp_path, verify=False)
    t._client_or_create()
    # A closed client is replaced on the next _client_or_create; the new build must
    # still carry the configured verify, not silently revert to the default.
    t._client = None
    t._client_or_create()
    assert len(builds) == 2
    assert builds[1]["verify"] is False


def test_telemetry_verify_resolution() -> None:
    from litellm_plugin_openrouter_pareto.config import TelemetryConfig

    assert telemetry_verify(None) is True
    assert telemetry_verify(TelemetryConfig()) is True
    assert telemetry_verify(TelemetryConfig(ssl_verify=False)) is False
    assert telemetry_verify(TelemetryConfig(ssl_ca_cert="/x.pem")) == "/x.pem"
    # A custom CA bundle takes precedence over the boolean default.
    assert telemetry_verify(TelemetryConfig(ssl_verify=True, ssl_ca_cert="/x.pem")) == "/x.pem"


def test_telemetry_config_rejects_empty_ca_cert() -> None:
    from litellm_plugin_openrouter_pareto.config import TelemetryConfig

    with pytest.raises(ValueError, match="ssl_ca_cert"):
        TelemetryConfig(ssl_ca_cert="")
    with pytest.raises(ValueError, match="ssl_ca_cert"):
        TelemetryConfig(ssl_ca_cert="   ")


def test_telemetry_config_strips_ca_cert_whitespace() -> None:
    from litellm_plugin_openrouter_pareto.config import TelemetryConfig

    cfg = TelemetryConfig(ssl_ca_cert="  /ca.pem  ")
    assert cfg.ssl_ca_cert == "/ca.pem"
