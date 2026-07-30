from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable

from litellm.integrations.custom_logger import CustomLogger
from litellm.types.utils import AllMessageValues

from .config import DEFAULT_RULES, Rule
from .cooldown import RateLimitCooldown
from .telemetry import CacheEntry, Telemetry

_OR_PREFIX = "or-"


@runtime_checkable
class TelemetrySource(Protocol):
    async def get(self, model: str) -> CacheEntry | None: ...


class OpenRouterParetoCallback(CustomLogger):
    def __init__(
        self,
        rules: Mapping[str, Rule] | None = None,
        telemetry: TelemetrySource | None = None,
        cooldown: RateLimitCooldown | None = None,
    ) -> None:
        self._rules: Mapping[str, Rule] = rules if rules is not None else DEFAULT_RULES
        self._telemetry: TelemetrySource = telemetry if telemetry is not None else Telemetry(self._rules)
        self._cooldown = cooldown if cooldown is not None else RateLimitCooldown()

    def _is_managed(self, model: str) -> bool:
        return model in self._rules

    def _slug_from_id(self, dep_id: object) -> str | None:
        if isinstance(dep_id, str):
            slug = dep_id[len(_OR_PREFIX):] if dep_id.startswith(_OR_PREFIX) else dep_id
            return slug or None
        return None

    def slug_of(self, deployment: Mapping[str, object]) -> str | None:
        model_info = deployment.get("model_info")
        if isinstance(model_info, Mapping):
            slug = self._slug_from_id(model_info.get("id"))
            if slug is not None:
                return slug
        litellm_params = deployment.get("litellm_params")
        if isinstance(litellm_params, Mapping):
            extra_body = litellm_params.get("extra_body")
            if isinstance(extra_body, Mapping):
                provider = extra_body.get("provider")
                if isinstance(provider, Mapping):
                    only = provider.get("only")
                    if isinstance(only, list) and only and isinstance(only[0], str):
                        return only[0]
        return None

    def _slug_from_kwargs(self, kwargs: Mapping[str, object]) -> str | None:
        for meta_key in ("metadata", "litellm_metadata"):
            meta = kwargs.get(meta_key)
            if isinstance(meta, Mapping):
                model_info = meta.get("model_info")
                if isinstance(model_info, Mapping):
                    slug = self._slug_from_id(model_info.get("id"))
                    if slug is not None:
                        return slug
        litellm_params = kwargs.get("litellm_params")
        if isinstance(litellm_params, Mapping):
            extra_body = litellm_params.get("extra_body")
            if isinstance(extra_body, Mapping):
                provider = extra_body.get("provider")
                if isinstance(provider, Mapping):
                    only = provider.get("only")
                    if isinstance(only, list) and only and isinstance(only[0], str):
                        return only[0]
        return None

    def _preferred_slugs(self, entry: CacheEntry) -> tuple[str, ...]:
        candidates = (entry.winner, *entry.safe_set) if entry.winner is not None else entry.safe_set
        unique = dict.fromkeys(candidates)
        return tuple(slug for slug in unique if not self._cooldown.is_hot(slug))

    async def async_filter_deployments(
        self,
        model: str,
        healthy_deployments: list[dict[str, object]] | dict[str, object],
        messages: list[AllMessageValues] | None,
        request_kwargs: dict[str, object] | None = None,
        parent_otel_span: object | None = None,
    ) -> list[dict[str, object]]:
        deployments = [healthy_deployments] if isinstance(healthy_deployments, dict) else healthy_deployments
        if not self._is_managed(model):
            return deployments
        try:
            entry = await self._telemetry.get(model)
        except Exception:
            return deployments
        return self._narrow(entry, deployments)

    def _narrow(
        self,
        entry: CacheEntry | None,
        deployments: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        if entry is None or entry.stale or entry.winner is None:
            return deployments
        for slug in self._preferred_slugs(entry):
            match = [d for d in deployments if self.slug_of(d) == slug]
            if match:
                return match
        winner_match = [d for d in deployments if self.slug_of(d) == entry.winner]
        return winner_match or deployments

    async def async_log_failure_event(
        self,
        kwargs: Mapping[str, object],
        response_obj: object,
        start_time: float,
        end_time: float,
    ) -> None:
        model = kwargs.get("model")
        if not isinstance(model, str) or not self._is_managed(model):
            return
        exc = kwargs.get("exception")
        status = getattr(exc, "status_code", None)
        if status != 429:
            return
        slug = self._slug_from_kwargs(kwargs)
        if slug is not None:
            self._cooldown.record(slug)


openrouter_pareto_callback = OpenRouterParetoCallback()
