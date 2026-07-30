from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

import httpx
import platformdirs
from pydantic import BaseModel, Field

from .config import Rule
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
OR_COLD_START_TIMEOUT_S = 16.0
OR_CACHE_VERSION = 2
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
    )


class Telemetry:
    def __init__(
        self,
        rules: Mapping[str, Rule],
        *,
        cache_dir: str | Path | None = None,
        base_url: str = OR_BASE_URL,
    ) -> None:
        self._rules = rules
        self._base_url = base_url.rstrip("/")
        dir_path = (
            Path(cache_dir)
            if cache_dir is not None
            else platformdirs.user_cache_path("litellm-plugin-openrouter-pareto")
        )
        dir_path.mkdir(parents=True, exist_ok=True)
        self._cache_path = dir_path / "cache.json"
        self._memory: dict[str, CacheEntry] = dict(self._load_disk())
        self._locks: dict[str, asyncio.Lock] = {}
        self._models_map: dict[str, str] | None = None
        self._models_fetched_at: float = 0.0
        self._client: httpx.AsyncClient | None = None

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

    def _persist(self) -> None:
        stored = _StoredCache(
            version=OR_CACHE_VERSION,
            entries={k: _entry_to_stored(v) for k, v in self._memory.items()},
        )
        tmp = self._cache_path.with_name(f".{self._cache_path.name}.{os.getpid()}.tmp")
        tmp.write_text(stored.model_dump_json())
        os.replace(tmp, self._cache_path)

    def _client_or_create(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                headers={"User-Agent": OR_USER_AGENT},
                timeout=OR_TIMEOUT_S,
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
            except Exception:
                pass
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

    def _mark_stale(self, existing: CacheEntry) -> CacheEntry:
        return replace(existing, stale=True)

    def _stale_fallback(self, or_id: str, existing: CacheEntry | None) -> CacheEntry | None:
        if existing is not None:
            self._memory[or_id] = self._mark_stale(existing)
            self._persist()
        return self._memory.get(or_id)

    async def _ensure_fresh(self, or_id: str) -> CacheEntry | None:
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
            except Exception:
                return self._stale_fallback(or_id, existing)
            if entry is None:
                return self._stale_fallback(or_id, existing)
            self._memory[or_id] = entry
            self._persist()
            return entry

    async def get(self, model: str) -> CacheEntry | None:
        if model not in self._rules:
            return None
        existing = self._memory.get(model)
        if existing is not None and not existing.stale and (time.time() - existing.fetched_at) < OR_POLL_INTERVAL_S:
            return existing
        try:
            refresh = self._ensure_fresh(model)
            if existing is None:
                entry = await asyncio.wait_for(refresh, timeout=OR_COLD_START_TIMEOUT_S)
            else:
                entry = await refresh
            return entry
        except Exception:
            return existing

    async def aclose(self) -> None:
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
