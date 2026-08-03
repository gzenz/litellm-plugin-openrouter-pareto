from __future__ import annotations

import asyncio
import json
import os
import ssl
import sys
import time
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

import httpx
import platformdirs
from filelock import FileLock, Timeout
from pydantic import BaseModel, Field

from .config import Rule, rule_fingerprint
from .models import (
    EndpointEntry,
    EndpointsResponse,
    ModelsResponse,
    StatsEndpoint,
    StatsResponse,
)
from .scorer import Selection, select_candidates

OR_POLL_INTERVAL_S = 300.0
OR_CANONICAL_TTL_S = 86400.0
OR_DISK_RETENTION_S = 604800.0
OR_COLD_START_TIMEOUT_S = 16.0
OR_CACHE_VERSION = 4
OR_BASE_URL = "https://openrouter.ai/api"
OR_USER_AGENT = "litellm-plugin-openrouter-pareto"
OR_TIMEOUT_S = 15.0


@dataclass(frozen=True, slots=True)
class CacheEntry:
    winner: str | None
    candidate_winner: str | None
    candidate_streak: int
    safe_set: tuple[str, ...]
    canonical_slug: str | None
    canonical_slug_fetched_at: float
    fetched_at: float
    stale: bool
    rule_fp: str = ""
    excluded_bases: tuple[str, ...] = ()
    allowed_bases: tuple[str, ...] = ()


class _StoredEntry(BaseModel):
    model_config = {"extra": "ignore"}
    winner: str | None = None
    candidate_winner: str | None = None
    candidate_streak: int = 0
    safe_set: list[str] = Field(default_factory=list[str])
    canonical_slug: str | None = None
    canonical_slug_fetched_at: float = 0.0
    fetched_at: float = 0.0
    stale: bool = False
    rule_fp: str = ""
    excluded_bases: list[str] = Field(default_factory=list[str])
    allowed_bases: list[str] = Field(default_factory=list[str])


class _StoredCache(BaseModel):
    model_config = {"extra": "ignore"}
    version: int
    entries: dict[str, _StoredEntry] = Field(default_factory=dict[str, _StoredEntry])


def _entry_from_stored(s: _StoredEntry) -> CacheEntry:
    return CacheEntry(
        winner=s.winner,
        candidate_winner=s.candidate_winner,
        candidate_streak=s.candidate_streak,
        safe_set=tuple(s.safe_set),
        canonical_slug=s.canonical_slug,
        canonical_slug_fetched_at=s.canonical_slug_fetched_at,
        fetched_at=s.fetched_at,
        stale=s.stale,
        rule_fp=s.rule_fp,
        excluded_bases=tuple(s.excluded_bases),
        allowed_bases=tuple(s.allowed_bases),
    )


def _entry_to_stored(e: CacheEntry) -> _StoredEntry:
    return _StoredEntry(
        winner=e.winner,
        candidate_winner=e.candidate_winner,
        candidate_streak=e.candidate_streak,
        safe_set=list(e.safe_set),
        canonical_slug=e.canonical_slug,
        canonical_slug_fetched_at=e.canonical_slug_fetched_at,
        fetched_at=e.fetched_at,
        stale=e.stale,
        rule_fp=e.rule_fp,
        excluded_bases=list(e.excluded_bases),
        allowed_bases=list(e.allowed_bases),
    )


class Telemetry:
    def __init__(
        self,
        rules: Mapping[str, Rule],
        *,
        cache_dir: str | Path | None = None,
        base_url: str = OR_BASE_URL,
        verify: bool | str = True,
    ) -> None:
        self._rules = rules
        self._base_url = base_url.rstrip("/")
        self._verify = verify
        dir_path = (
            Path(cache_dir)
            if cache_dir is not None
            else platformdirs.user_cache_path("litellm-plugin-openrouter-pareto")
        )
        self._cache_path = dir_path / "cache.json"
        self._lock_path = dir_path / "cache.lock"
        self._memory: dict[str, CacheEntry] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._refresh_tasks: set[asyncio.Task[None]] = set()
        self._persist_tasks: set[asyncio.Task[None]] = set()
        self._models_map: dict[str, str] | None = None
        self._models_fetched_at: float = 0.0
        self._client: httpx.AsyncClient | None = None
        self._disk_loaded = False
        self._warned: set[str] = set()

    def _ensure_disk(self) -> None:
        if self._disk_loaded:
            return
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            self._warn("disk-init", self._cache_path.name, exc)
            return
        self._memory.update(dict(self._load_disk_for_rules()))
        self._disk_loaded = True

    def _load_disk(self) -> tuple[tuple[str, CacheEntry], ...]:
        try:
            raw = json.loads(self._cache_path.read_text())
        except (OSError, ValueError):
            return ()
        try:
            stored = _StoredCache.model_validate(raw)
        except ValueError:
            return ()
        if stored.version != OR_CACHE_VERSION:
            return ()
        return tuple((k, _entry_from_stored(v)) for k, v in stored.entries.items())

    def _disk_key(self, model: str) -> str:
        """Disk identity for a model under THIS process's rule. Including the rule
        fingerprint lets workers with different policies share one cache file without
        evicting each other's entries."""
        r = self._rules.get(model)
        fp = rule_fingerprint(r) if r is not None else ""
        return f"{model}\x00{fp}"

    def _load_disk_for_rules(self) -> tuple[tuple[str, CacheEntry], ...]:
        """Disk entries usable by THIS process, keyed back to the plain model name for
        in-memory use. The composite disk key already encodes the rule fingerprint, so
        an entry written under a different policy simply does not match."""
        wanted = {self._disk_key(m): m for m in self._rules}
        return tuple((wanted[k], e) for k, e in self._load_disk() if k in wanted)

    def _supersedes(self, incoming: CacheEntry, stored: CacheEntry) -> bool:
        """Whether `incoming` represents a later state than `stored`.

        Ordering is by `fetched_at`, which is the only globally comparable fact here:
        the cache file is shared across worker processes, so any process-local counter
        would let a worker that merely failed a lot outrank another worker's genuinely
        newer telemetry.

        `stale` breaks a tie within the SAME fetch. Marking an entry stale does not move
        `fetched_at`, so without this an older in-flight fresh snapshot could land after
        the stale write and make disk claim a refresh succeeded when it failed."""
        if incoming.fetched_at != stored.fetched_at:
            return incoming.fetched_at > stored.fetched_at
        return incoming.stale and not stored.stale

    def _prune_stale(
        self, merged: dict[str, CacheEntry], now: float
    ) -> dict[str, CacheEntry]:
        """Drop disk entries no one has refreshed within the retention window. A rule
        edit orphans the old `(model, fingerprint)` key: nothing writes it again, so its
        `fetched_at` ages out while a live worker's own entries keep being refreshed.
        Time is the only signal that separates 'abandoned by a rule change' from 'owned
        by another worker running a different rule for the same model', so pruning is by
        age, not by fingerprint ownership (which cannot tell the two apart)."""
        cutoff = now - OR_DISK_RETENTION_S
        return {k: v for k, v in merged.items() if v.fetched_at >= cutoff}

    def _persist(self, snapshot: tuple[tuple[str, CacheEntry], ...], now: float) -> None:
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            lock = FileLock(str(self._lock_path), timeout=5)
            with lock:
                merged = dict(self._load_disk())
                for model, entry in snapshot:
                    key = self._disk_key(model)
                    prev = merged.get(key)
                    if prev is None or self._supersedes(entry, prev):
                        merged[key] = entry
                pruned = self._prune_stale(merged, now)
                stored = _StoredCache(
                    version=OR_CACHE_VERSION,
                    entries={k: _entry_to_stored(v) for k, v in pruned.items()},
                )
                tmp = self._cache_path.with_name(f".{self._cache_path.name}.{os.getpid()}.tmp")
                tmp.write_text(stored.model_dump_json())
                os.replace(tmp, self._cache_path)
        except Timeout as exc:
            self._warn("persist-lock-timeout", self._cache_path.name, exc)
        except OSError as exc:
            self._warn("persist-io", self._cache_path.name, exc)

    def _ssl_verify(self) -> bool | ssl.SSLContext:
        """Resolve the configured verify value into what httpx accepts without the
        deprecated `verify=<str>` form. A CA-bundle path becomes an SSLContext built
        here (not at config-load time) so a missing or unreadable file raises inside
        the refresh path, degrades to a warning, and never reaches the request path.
        A directory is treated as a capath, matching httpx's own string handling."""
        v = self._verify
        if isinstance(v, str):
            if os.path.isdir(v):
                return ssl.create_default_context(capath=v)
            return ssl.create_default_context(cafile=v)
        return v

    def _client_or_create(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                headers={"User-Agent": OR_USER_AGENT},
                timeout=OR_TIMEOUT_S,
                verify=self._ssl_verify(),
            )
        return self._client

    async def _fetch_models(self) -> dict[str, str]:
        client = self._client_or_create()
        resp = await client.get(f"{self._base_url}/v1/models")
        resp.raise_for_status()
        parsed = ModelsResponse.model_validate(resp.json())
        return {m.id: m.canonical_slug for m in parsed.data if m.canonical_slug}

    async def _fetch_endpoints(self, or_id: str) -> tuple[EndpointEntry, ...]:
        client = self._client_or_create()
        resp = await client.get(f"{self._base_url}/v1/models/{or_id}/endpoints")
        resp.raise_for_status()
        body = EndpointsResponse.model_validate(resp.json())
        endpoints = body.data.endpoints if body.data is not None else ()
        return tuple(endpoints)

    async def _fetch_stats(self, permaslug: str) -> tuple[StatsEndpoint, ...]:
        client = self._client_or_create()
        resp = await client.get(
            f"{self._base_url}/frontend/v1/stats/endpoint",
            params={"permaslug": permaslug, "variant": "standard"},
        )
        resp.raise_for_status()
        parsed = StatsResponse.model_validate(resp.json())
        return tuple(parsed.data)

    async def _resolve_canonical_slug(
        self,
        or_id: str,
        previous: CacheEntry | None,
        now: float,
    ) -> tuple[str | None, float]:
        if (
            previous is not None
            and previous.canonical_slug is not None
            and (now - previous.canonical_slug_fetched_at) < OR_CANONICAL_TTL_S
        ):
            return previous.canonical_slug, previous.canonical_slug_fetched_at
        if self._models_map is None or (now - self._models_fetched_at) >= OR_CANONICAL_TTL_S:
            try:
                self._models_map = await self._fetch_models()
                self._models_fetched_at = now
            except Exception as exc:
                self._warn("canonical-slug", or_id, exc)
        slug = self._models_map.get(or_id) if self._models_map is not None else None
        if slug is not None:
            return slug, now
        if previous is not None and previous.canonical_slug is not None:
            return previous.canonical_slug, previous.canonical_slug_fetched_at
        return or_id, now

    def _apply_promotion(
        self,
        previous: CacheEntry | None,
        selection: Selection,
        rule: Rule,
        canonical_slug: str | None,
        slug_fetched_at: float,
        now: float,
    ) -> CacheEntry | None:
        if selection.winner is None or not selection.safe_set:
            if rule.exclude_regions and (selection.excluded_bases or selection.allowed_bases):
                return CacheEntry(
                    winner=None,
                    candidate_winner=None,
                    candidate_streak=0,
                    safe_set=(),
                    canonical_slug=canonical_slug,
                    canonical_slug_fetched_at=slug_fetched_at,
                    fetched_at=now,
                    stale=False,
                    rule_fp=rule_fingerprint(rule),
                    excluded_bases=selection.excluded_bases,
                    allowed_bases=selection.allowed_bases,
                )
            return None
        selected_slug = selection.winner.slug
        prev_winner = previous.winner if previous is not None else None
        prev_candidate_winner = previous.candidate_winner if previous is not None else None
        prev_streak = previous.candidate_streak if previous is not None else 0
        same_candidate = prev_candidate_winner == selected_slug
        candidate_streak = prev_streak + 1 if same_candidate else 1
        previous_eligible = prev_winner is not None and prev_winner in selection.safe_set
        promote = (
            (not previous_eligible)
            or prev_winner == selected_slug
            or candidate_streak >= rule.promotion_polls
        )
        winner = selected_slug if promote else prev_winner
        return CacheEntry(
            winner=winner,
            candidate_winner=selected_slug,
            candidate_streak=candidate_streak,
            safe_set=selection.safe_set,
            canonical_slug=canonical_slug,
            canonical_slug_fetched_at=slug_fetched_at,
            fetched_at=now,
            stale=False,
            rule_fp=rule_fingerprint(rule),
            excluded_bases=selection.excluded_bases,
            allowed_bases=selection.allowed_bases,
        )

    async def _refresh(self, or_id: str) -> CacheEntry | None:
        rule = self._rules[or_id]
        previous = self._memory.get(or_id)
        now = time.time()
        canonical_slug, slug_fetched_at = await self._resolve_canonical_slug(or_id, previous, now)
        endpoints, stats = await asyncio.gather(
            self._fetch_endpoints(or_id),
            self._fetch_stats(canonical_slug or or_id),
        )
        selection = select_candidates(stats, endpoints, rule)
        return self._apply_promotion(previous, selection, rule, canonical_slug, slug_fetched_at, now)

    def _lock_for(self, or_id: str) -> asyncio.Lock:
        lock = self._locks.get(or_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[or_id] = lock
        return lock

    def _persist_soon(self) -> None:
        """Write the cache to disk without holding up the caller. The disk copy is an
        optimization, so a busy cross-process lock must cost a background task rather
        than up to `FileLock`'s timeout on a customer request; awaiting a thread does
        not take the wait out of request latency, it only moves which thread blocks."""
        snapshot = tuple(self._memory.items())
        now = time.time()
        try:
            task = asyncio.create_task(asyncio.to_thread(self._persist, snapshot, now))
        except RuntimeError:
            self._persist(snapshot, now)
            return
        self._persist_tasks.add(task)
        task.add_done_callback(self._persist_tasks.discard)

    def _mark_stale(self, existing: CacheEntry) -> CacheEntry:
        return replace(existing, stale=True)

    async def _stale_fallback(self, or_id: str, existing: CacheEntry | None) -> CacheEntry | None:
        if existing is not None:
            self._memory[or_id] = self._mark_stale(existing)
            self._persist_soon()
        return self._memory.get(or_id)

    def _warn(self, kind: str, or_id: str, exc: BaseException | None) -> None:
        key = f"{kind}:{or_id}"
        if key in self._warned:
            return
        self._warned.add(key)
        detail = type(exc).__name__ if exc is not None else "empty-selection"
        sys.stderr.write(
            f"openrouter-pareto: telemetry {kind} for {or_id} degraded ({detail}); "
            f"continuing with fallback\n"
        )

    async def _ensure_fresh(self, or_id: str) -> CacheEntry | None:
        self._ensure_disk()
        existing = self._memory.get(or_id)
        if existing is not None and not existing.stale and (time.time() - existing.fetched_at) < OR_POLL_INTERVAL_S:
            return existing
        lock = self._lock_for(or_id)
        async with lock:
            existing = self._memory.get(or_id)
            if existing is not None and not existing.stale and (time.time() - existing.fetched_at) < OR_POLL_INTERVAL_S:
                return existing
            try:
                entry = await self._refresh(or_id)
            except Exception as exc:
                self._warn("refresh", or_id, exc)
                return await self._stale_fallback(or_id, existing)
            if entry is None:
                self._warn("refresh-empty", or_id, None)
                return await self._stale_fallback(or_id, existing)
            self._memory[or_id] = entry
            self._persist_soon()
            return entry

    async def get(self, model: str) -> CacheEntry | None:
        if model not in self._rules:
            return None
        self._ensure_disk()
        existing = self._memory.get(model)
        if existing is not None and not existing.stale and (time.time() - existing.fetched_at) < OR_POLL_INTERVAL_S:
            return existing
        if existing is None:
            # Cold start: kick off the refresh and bound THIS request's wait, but
            # do NOT cancel the refresh on timeout - let it complete in the
            # background so the cache warms and later requests find it. The
            # per-id asyncio.Lock in _ensure_fresh dedupes concurrent triggers.
            self._spawn_refresh(model)
            try:
                existing = await asyncio.wait_for(
                    self._wait_for_cache(model), timeout=OR_COLD_START_TIMEOUT_S
                )
            except (asyncio.TimeoutError, Exception):
                return None
            return existing
        try:
            return await self._ensure_fresh(model)
        except Exception:
            return existing

    async def _wait_for_cache(self, model: str) -> CacheEntry | None:
        deadline = time.monotonic() + OR_COLD_START_TIMEOUT_S
        while time.monotonic() < deadline:
            entry = self._memory.get(model)
            if entry is not None and not entry.stale:
                return entry
            await asyncio.sleep(0.25)
        return None

    def _spawn_refresh(self, model: str) -> None:
        async def _run() -> None:
            try:
                await self._ensure_fresh(model)
            except Exception:
                pass
            finally:
                self._refresh_tasks.discard(task)

        task = asyncio.create_task(_run())
        self._refresh_tasks.add(task)

    async def aclose(self) -> None:
        """Leave no owned background work behind. Pending disk writes are drained
        before the HTTP client closes: a queued write carries the winner, stale marker,
        or region evidence, and dropping it on shutdown loses state the process already
        decided.

        In-flight refreshes are cancelled rather than awaited. Their result is only an
        optimization and a hung upstream must not hold shutdown open; the writes they
        would have produced are exactly what the drain below no longer has to wait for."""
        for task in tuple(self._refresh_tasks):
            task.cancel()
        if self._refresh_tasks:
            await asyncio.gather(*tuple(self._refresh_tasks), return_exceptions=True)
        if self._persist_tasks:
            await asyncio.gather(*tuple(self._persist_tasks), return_exceptions=True)
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
