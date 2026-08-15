from __future__ import annotations

import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from litellm.integrations.custom_logger import CustomLogger
from litellm.types.utils import AllMessageValues

from .config import (
    DEFAULT_RULES,
    Rule,
    load_allowlist_config,
    load_cooldown_config,
    load_rules_from_settings,
    load_telemetry_config,
    telemetry_verify,
)
from .cooldown import RateLimitCooldown, is_input_cap_error
from .error_log import or_error_log
from .telemetry import DEFAULT_ALLOWLIST_TTL_S, CacheEntry, Telemetry

_OR_PREFIX = "or-"

_NO_ALLOWED_PROVIDERS_RE = re.compile(
    r"allowed-providers setting permits only:\s*(.+?)\.\s*To change",
    re.IGNORECASE,
)


def _parse_allowed_providers(message: str) -> frozenset[str] | None:
    """Extract the account-level allowed-providers list from an OpenRouter 404
    error body. The error message is the only place OpenRouter exposes this
    account setting; there is no API endpoint for it. Returns None when the
    pattern does not match (not a 'no allowed providers' 404)."""
    m = _NO_ALLOWED_PROVIDERS_RE.search(message)
    if m is None:
        return None
    raw = m.group(1)
    slugs = frozenset(s.strip() for s in raw.split(",") if s.strip())
    return slugs if slugs else None


def _is_no_allowed_providers_404(status: object, message: str) -> bool:
    """Whether a failure is an OpenRouter 404 caused by the account's
    allowed-providers restriction blocking the selected provider.only slug."""
    return status == 404 and "No allowed providers" in message


def _normalize_model_group(model: str) -> str:
    """The OpenRouter id a rule is keyed on, extracted from the model group litellm
    hands us. Callers reach a managed model by different names: EverCore sends the
    bare id (`deepseek/deepseek-v4-flash-0731`), while Claude Code sends the
    `openrouter/`-prefixed form with a context tag (`openrouter/deepseek/
    deepseek-v4-flash-0731[1m]`). Rules are authored on the bare id (the key
    OpenRouter's /v1/models uses), so strip a leading `openrouter/` and a trailing
    `[...]` tag to recover it. A name that has neither is returned unchanged."""
    stripped = model.removeprefix("openrouter/")
    bracket = stripped.find("[")
    if bracket != -1 and stripped.endswith("]"):
        stripped = stripped[:bracket]
    return stripped


class StrictProviderConflict(ValueError):
    """A client sent its own `extra_body.provider.only` on a managed model whose rule
    set `strict_provider: true`. The router re-raises this to the caller, so the request
    fails loudly instead of the plugin silently overriding the client's choice."""


class AllProvidersOnCooldown(ValueError):
    """Every provider in the safe set (or cold-start fallback list) for a managed
    model is on a 429 cooldown. Raised from the deployment filter instead of
    returning []: an empty list makes the router raise a generic
    RouterRateLimitError whose cooldown_list is read from litellm's own
    cooldown cache, which is empty here (the plugin's RateLimitCooldown is a
    separate in-process store), so the surfaced error carries no reason. The
    router re-raises filter exceptions verbatim (same path as
    StrictProviderConflict), so this surfaces the plugin's own message. Input
    caps and region policy never raise this - only an all-429-hot state does."""


@dataclass(frozen=True, slots=True)
class _Routable:
    """The region policy permits these deployments; the winner search still runs."""

    deployments: list[dict[str, object]]


@dataclass(frozen=True, slots=True)
class _Unpinned:
    """A terminal `unpinned` decision: the winner is deliberately not pinned, so no
    winner may be re-derived. Distinct from `_Routable` so the stale path cannot be
    mistaken for a still-pinnable list."""

    deployments: list[dict[str, object]]


@dataclass(frozen=True, slots=True)
class _Decline:
    """The policy refuses to route at all; the router surfaces a routing error."""


_RegionResult = _Routable | _Unpinned | _Decline


@runtime_checkable
class TelemetrySource(Protocol):
    async def get(self, model: str) -> CacheEntry | None: ...
    def get_allowed_providers(self) -> frozenset[str] | None: ...
    def set_allowed_providers(self, slugs: frozenset[str]) -> None: ...


class OpenRouterParetoCallback(CustomLogger):
    def __init__(
        self,
        rules: Mapping[str, Rule] | None = None,
        telemetry: TelemetrySource | None = None,
        cooldown: RateLimitCooldown | None = None,
    ) -> None:
        self._explicit_rules = rules
        self._explicit_telemetry = telemetry
        self._rules: Mapping[str, Rule] = rules if rules is not None else DEFAULT_RULES
        self._telemetry: TelemetrySource = telemetry if telemetry is not None else Telemetry(self._rules)
        self._cooldown = cooldown if cooldown is not None else RateLimitCooldown()
        self._rules_resolved = rules is not None
        self._telemetry_resolved = telemetry is not None
        self._cooldown_resolved = cooldown is not None
        self._warned: set[str] = set()

    def _resolve_rules(self) -> Mapping[str, Rule]:
        # Rules and telemetry-client config are independent domains and are resolved
        # independently. An explicit `rules=` must NOT suppress global telemetry SSL
        # config (or its validation), and an explicit `telemetry=` must NOT be
        # replaced by it. Bad telemetry config raises loudly here, the same place
        # bad rules config already raises, before any request is routed.
        rules_changed = False
        if not self._rules_resolved:
            if self._explicit_rules is None:
                configured = load_rules_from_settings()
                if configured is not None:
                    self._rules = configured
                    rules_changed = True
            self._rules_resolved = True
        if not self._telemetry_resolved:
            tel_config = load_telemetry_config()
            allowlist_config = load_allowlist_config()
            # Rebuild the telemetry client only when something actually changed: a
            # newly loaded rule set, or an explicit telemetry/allowlist config. With
            # neither, the default client built in __init__ (verify=True) is correct.
            if rules_changed or tel_config is not None or allowlist_config is not None:
                self._telemetry = Telemetry(
                    self._rules,
                    verify=telemetry_verify(tel_config),
                    allowlist_ttl_s=(
                        allowlist_config.ttl_s
                        if allowlist_config is not None
                        else DEFAULT_ALLOWLIST_TTL_S
                    ),
                )
            self._telemetry_resolved = True
        if not self._cooldown_resolved:
            cd_config = load_cooldown_config()
            # Rebuild the cooldown only when an explicit cooldown config is present.
            # An explicit `cooldown=` passed to __init__ (tests, direct callers) is
            # honored as-is and never replaced by global config.
            if cd_config is not None:
                self._cooldown = RateLimitCooldown(
                    window_s=cd_config.rate_limit_window_s,
                    threshold=cd_config.rate_limit_threshold,
                    input_cap_ttl_s=cd_config.input_cap_ttl_s,
                )
            self._cooldown_resolved = True
        return self._rules

    def _is_managed(self, model: str) -> bool:
        return model in self._resolve_rules()

    def _slug_from_id(self, dep_id: object) -> str | None:
        if isinstance(dep_id, str):
            slug = dep_id[len(_OR_PREFIX):] if dep_id.startswith(_OR_PREFIX) else dep_id
            return slug or None
        return None

    def _slug_from_params(self, litellm_params: object, model_info: object) -> str | None:
        slugs = self._slugs_from_params(litellm_params, model_info)
        return slugs[0] if len(slugs) == 1 else None

    def _slugs_from_params(
        self, litellm_params: object, model_info: object
    ) -> tuple[str, ...]:
        """Every provider slug a deployment authorizes. `provider.only` is an allowlist,
        not a scalar pin, so a region check that reads only the first element would
        approve `[allowed, excluded]` and still send both on the wire. An entry that is
        not a non-empty string makes the whole list unidentifiable: returning () means
        unknown, which the region policy rejects rather than guesses about.

        `model_info.id` is consulted only when `provider.only` is absent entirely; a
        present-but-malformed allowlist must not be overridden by a friendlier identity."""
        if isinstance(litellm_params, Mapping):
            extra_body: object = litellm_params.get("extra_body")
            if isinstance(extra_body, Mapping):
                provider: object = extra_body.get("provider")
                if isinstance(provider, Mapping):
                    only: object = provider.get("only")
                    if isinstance(only, list):
                        entries: list[object] = list(only)
                        strs = tuple(e for e in entries if isinstance(e, str) and e)
                        if entries and len(strs) == len(entries):
                            return strs
                        return ()
                    if only is not None:
                        return ()
        if isinstance(model_info, Mapping):
            slug = self._slug_from_id(model_info.get("id"))
            if slug is not None:
                return (slug,)
        return ()

    def slugs_of(self, deployment: Mapping[str, object]) -> tuple[str, ...]:
        return self._slugs_from_params(
            deployment.get("litellm_params"), deployment.get("model_info")
        )

    def slug_of(self, deployment: Mapping[str, object]) -> str | None:
        return self._slug_from_params(
            deployment.get("litellm_params"), deployment.get("model_info")
        )

    def _slug_from_kwargs(self, kwargs: Mapping[str, object]) -> str | None:
        # The async_log_failure_event kwargs is litellm's model_call_details. The
        # provider pin the plugin wrote onto the chosen deployment surfaces at the top
        # level as extra_body.provider.only (spread from optional_params) and mirrored
        # under optional_params.extra_body.provider.only. model_info.id under
        # litellm_params is a regenerated hash, not the or- slug.
        litellm_params = kwargs.get("litellm_params")
        top_eb = kwargs.get("extra_body")
        if isinstance(top_eb, Mapping):
            provider = top_eb.get("provider")
            if isinstance(provider, Mapping):
                only = provider.get("only")
                if isinstance(only, list) and len(only) == 1 and isinstance(only[0], str):
                    return only[0]
        opt = kwargs.get("optional_params")
        if isinstance(opt, Mapping):
            slug = self._slug_from_params(opt, None)
            if slug is not None:
                return slug
        model_info = None
        if isinstance(litellm_params, Mapping):
            mi = litellm_params.get("model_info")
            if isinstance(mi, Mapping):
                model_info = mi
        return self._slug_from_params(litellm_params, model_info)

    def _slug_from_request_data(self, request_data: Mapping[str, object]) -> str | None:
        # The passthrough failure hook's request_data is the merged request body
        # + litellm kwargs, NOT model_call_details. The top-level OpenRouter
        # `provider.only` body field is what OpenRouter actually used, so treat it
        # as authoritative: a singleton non-empty slug is the attribution; a
        # present-but-ambiguous `only` (multiple providers, empty, malformed, or
        # null) cannot be attributed to one slug, so return None rather than
        # reinterpreting it via a secondary copy in optional_params / extra_body
        # (that would cool down the wrong provider). Only an absent `only` key
        # falls through to the other sites, which mirror _slug_from_kwargs.
        provider = request_data.get("provider")
        if isinstance(provider, Mapping) and "only" in provider:
            only = provider["only"]
            if isinstance(only, list) and len(only) == 1:
                sole = only[0]
                if isinstance(sole, str) and sole:
                    return sole
            # Present but ambiguous (multiple, empty, malformed, or null): the
            # top-level field is authoritative, so do not reinterpret it via a
            # secondary copy in optional_params / extra_body.
            return None
        return self._slug_from_kwargs(request_data)

    def _allowlist(self) -> frozenset[str] | None:
        """The account-level allowed-providers set from telemetry, or None when
        unknown/stale. When None, no filtering is applied — the pareto-optimal
        provider is tried, and a 404 re-discovers the allowlist."""
        return self._telemetry.get_allowed_providers()

    def _allowlist_allows(self, slug: str, allowlist: frozenset[str] | None) -> bool:
        """Whether a slug's base is permitted by the allowlist. None means the
        allowlist is unknown/stale, so everything is permitted (filtering off)."""
        if allowlist is None:
            return True
        return slug.split("/", 1)[0] in allowlist

    def _ordered_candidates(self, entry: CacheEntry) -> tuple[str, ...]:
        """The winner then the safe set, deduped preserving order. This is the
        ranked list the value walk produced; selection walks it to find the
        first provider that is usable right now."""
        candidates = (entry.winner, *entry.safe_set) if entry.winner is not None else entry.safe_set
        return tuple(dict.fromkeys(candidates))

    def _preferred_slugs(self, model: str, entry: CacheEntry) -> tuple[str, ...]:
        allowlist = self._allowlist()
        return tuple(
            slug
            for slug in self._ordered_candidates(entry)
            if not self._cooldown.is_skipped(model, slug)
            and self._allowlist_allows(slug, allowlist)
        )

    def _client_provider_only(self, request_kwargs: dict[str, object] | None) -> bool:
        """Whether the caller sent its own `provider.only`. Read-only; the plugin never
        writes to `request_kwargs`. This only decides whether strict mode rejects the
        request, not what to route.

        Checks BOTH landing sites, because at deployment-filter time the pin is in
        different places depending on how the request arrived. Through the proxy or the
        OpenAI SDK, OpenRouter's `provider` block is a TOP-LEVEL body field, splatted into
        the router as `request_kwargs["provider"]`; litellm only folds it under
        `extra_body` later, inside `completion()`, after deployment selection. A direct
        `router.acompletion(..., extra_body={"provider": ...})` call carries it nested.
        Reading only the nested form let a normal proxy client walk straight through the
        strict gate."""
        if request_kwargs is None:
            return False
        containers = (request_kwargs, request_kwargs.get("extra_body"))
        for container in containers:
            if isinstance(container, Mapping):
                provider = container.get("provider")
                if isinstance(provider, Mapping) and provider.get("only") is not None:
                    return True
        return False

    async def async_filter_deployments(
        self,
        model: str,
        healthy_deployments: list[dict[str, object]] | dict[str, object],
        messages: list[AllMessageValues] | None,
        request_kwargs: dict[str, object] | None = None,
        parent_otel_span: object | None = None,
    ) -> list[dict[str, object]]:
        deployments = [healthy_deployments] if isinstance(healthy_deployments, dict) else healthy_deployments
        model = _normalize_model_group(model)
        rules = self._resolve_rules()
        rule = rules.get(model)
        if rule is None:
            return deployments
        if rule.strict_provider and self._client_provider_only(request_kwargs):
            raise StrictProviderConflict(
                f"openrouter-pareto: {model} is configured strict_provider, but the "
                f"request carries its own extra_body.provider.only; refusing to route "
                f"rather than let it override the plugin's provider choice"
            )
        try:
            entry = await self._telemetry.get(model)
        except Exception:
            if rule.wildcard:
                return self._wildcard_fallback(model, rule, deployments)
            return self._narrow(model, rule, None, deployments)
        if rule.wildcard:
            return self._inject_winner(model, rule, entry, deployments)
        return self._narrow(model, rule, entry, deployments)

    def _inject_winner(
        self,
        model: str,
        rule: Rule,
        entry: CacheEntry | None,
        deployments: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        if entry is None or entry.stale or entry.winner is None:
            return self._wildcard_fallback(model, rule, deployments, entry)
        # On 429: keep trying our own list (winner then safe set) until a
        # provider is not on cooldown. Input cap is a soft preference, not a
        # skip: prefer a non-capped provider, but if every non-hot provider is
        # capped fall back to the first non-hot (the winner if it is not hot)
        # - a small request may still succeed under the cap, better to try than
        # to fail. Only an all-429-hot list is a hard stop.
        allowlist = self._allowlist()
        not_hot = tuple(
            slug
            for slug in self._ordered_candidates(entry)
            if not self._cooldown.is_hot(model, slug)
            and self._allowlist_allows(slug, allowlist)
        )
        if not not_hot:
            raise AllProvidersOnCooldown(
                f"openrouter-pareto: every provider for {model} is on a 429 "
                f"cooldown; declining to route until one clears. "
                f"winner={entry.winner} safe_set={entry.safe_set}"
            )
        not_capped = tuple(
            slug for slug in not_hot if not self._cooldown.is_input_capped(model, slug)
        )
        chosen = not_capped[0] if not_capped else not_hot[0]
        return self._with_provider_only(deployments, chosen)

    def _region_allows(self, model: str, rule: Rule, slug: str, entry: CacheEntry | None) -> bool:
        """Whether a cold-start fallback slug may be pinned under the region policy.
        Geography lives only in live telemetry, so the last selection persists the
        excluded org bases. With no telemetry at all the slug is unverifiable and
        `unverified_region_policy` decides: `trust_fallback` takes the operator's word
        for configured fallbacks, anything else declines to pin it."""
        if not rule.exclude_regions:
            return True
        if entry is None:
            if rule.unverified_region_policy == "trust_fallback":
                return True
            self._warn_once(
                f"region-unverified:{model}:{slug}",
                f"openrouter-pareto: cannot verify region policy for {slug} on {model} "
                f"(no telemetry); not pinning it under "
                f"unverified_region_policy={rule.unverified_region_policy}\n",
            )
            return False
        base = slug.split("/", 1)[0]
        if base in entry.excluded_bases:
            return False
        if base in entry.allowed_bases:
            return True
        if rule.allow_unknown_region:
            return True
        self._warn_once(
            f"region-unverified:{model}:{slug}",
            f"openrouter-pareto: {slug} on {model} has no verified region "
            f"(absent from telemetry); skipping it to honor exclude_regions\n",
        )
        return False

    def _warn_once(self, key: str, message: str) -> None:
        if key in self._warned:
            return
        self._warned.add(key)
        sys.stderr.write(message)

    def _wildcard_fallback(
        self,
        model: str,
        rule: Rule,
        deployments: list[dict[str, object]],
        entry: CacheEntry | None = None,
    ) -> list[dict[str, object]]:
        # Cold start (no winner): walk the operator-vetted fallback list. On 429
        # keep trying until a fallback is not on cooldown. Input cap is a soft
        # preference, matched to the warm path: prefer a non-capped fallback, and
        # only fall back to a capped one (better to try than to fail - a small
        # request may fit under the cap) when every eligible fallback is capped.
        # Region eligibility is checked first: a region-ineligible fallback is
        # unusable, so it does not count toward the all-hot diagnosis (an
        # ineligible non-hot provider must not mask every eligible provider being
        # 429-hot). Only an all-429-hot ELIGIBLE fallback list is a hard stop.
        allowlist = self._allowlist()
        any_allowlist_disallowed = False
        eligible: list[str] = []
        any_eligible = False
        for slug in rule.cold_start_fallback:
            if not self._allowlist_allows(slug, allowlist):
                any_allowlist_disallowed = True
                continue
            if not self._region_allows(model, rule, slug, entry):
                continue
            any_eligible = True
            if self._cooldown.is_hot(model, slug):
                continue
            eligible.append(slug)
        if eligible:
            not_capped = [
                s for s in eligible if not self._cooldown.is_input_capped(model, s)
            ]
            chosen = not_capped[0] if not_capped else eligible[0]
            return self._with_provider_only(deployments, chosen)
        if any_eligible:
            # At least one fallback was region-eligible and every one of them was
            # 429-hot (none made it into `eligible`).
            raise AllProvidersOnCooldown(
                f"openrouter-pareto: every region-eligible cold-start fallback for "
                f"{model} is on a 429 cooldown; declining to route until one "
                f"clears. fallback={rule.cold_start_fallback}"
            )
        # No fallback was region-eligible (or none is configured). Under
        # `unpinned` the operator opted into ceding to OpenRouter (matching
        # pinned mode); anything else declines to route rather than cede, since
        # OpenRouter's auto-selection would bypass max_price, exclude_regions,
        # and the precision/context filters.
        if rule.exclude_regions and rule.unverified_region_policy == "unpinned":
            return self._unpinned_result(deployments)
        reason = "region-ineligible or none configured"
        if any_allowlist_disallowed:
            reason = "allowlist-disallowed, region-ineligible, or none configured"
        self._warn_once(
            f"all-hot:{model}",
            f"openrouter-pareto: no usable cold-start fallback for {model} "
            f"({reason}); declining to route. "
            f"fallback={rule.cold_start_fallback}\n",
        )
        return []

    def _unpinned_result(
        self,
        deployments: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        """A terminal, deliberately unpinned routing decision: return deployment copies
        with `provider.only` stripped (OpenRouter may pick the provider) but the zdr /
        no-fallback posture kept. No winner is re-derived from `model_info.id`."""
        return [self._stripped_copy(d) for d in deployments]

    def _stripped_copy(self, deployment: dict[str, object]) -> dict[str, object]:
        copy = self._clean_copy(deployment)
        litellm_params = copy.get("litellm_params")
        if isinstance(litellm_params, dict):
            extra_body = litellm_params.get("extra_body")
            if not isinstance(extra_body, dict):
                extra_body = {}
                litellm_params["extra_body"] = extra_body
            provider = extra_body.get("provider")
            if not isinstance(provider, dict):
                provider = {}
                extra_body["provider"] = provider
            provider.pop("only", None)
            provider["zdr"] = True
            provider["allow_fallbacks"] = False
        return copy

    def _with_provider_only(
        self,
        deployments: list[dict[str, object]],
        slug: str,
    ) -> list[dict[str, object]]:
        result: list[dict[str, object]] = []
        for d in deployments:
            copy = self._clean_copy(d)
            litellm_params = copy.get("litellm_params")
            if not isinstance(litellm_params, dict):
                result.append(copy)
                continue
            extra_body = litellm_params.get("extra_body")
            if not isinstance(extra_body, dict):
                extra_body = {}
                litellm_params["extra_body"] = extra_body
            provider = extra_body.get("provider")
            if not isinstance(provider, dict):
                provider = {}
                extra_body["provider"] = provider
            provider["only"] = [slug]
            provider["zdr"] = True
            provider["allow_fallbacks"] = False
            result.append(copy)
        return result

    def _clean_copy(self, deployment: dict[str, object]) -> dict[str, object]:
        copy = dict(deployment)
        litellm_params = deployment.get("litellm_params")
        if isinstance(litellm_params, dict):
            lp_copy = dict(litellm_params)
            extra_body = litellm_params.get("extra_body")
            if isinstance(extra_body, dict):
                eb_copy = dict(extra_body)
                provider = extra_body.get("provider")
                if isinstance(provider, dict):
                    eb_copy["provider"] = dict(provider)
                lp_copy["extra_body"] = eb_copy
            copy["litellm_params"] = lp_copy
        return copy

    def _region_eligible(
        self,
        model: str,
        rule: Rule,
        entry: CacheEntry | None,
        deployments: list[dict[str, object]],
    ) -> _RegionResult:
        """Classify what this rule's region policy permits: a still-pinnable list, a
        terminal unpinned list, or a decline. Applies to the pinned path too: a
        deployment can already carry `provider.only` for an excluded org, so scorer-side
        filtering alone would leave the default (non-wildcard) mode unguarded.

        This is a best-effort filter: a stale entry keeps serving its last-known region
        verdicts rather than expiring on a clock, so the filter degrades to last-known
        geography during a telemetry outage instead of flipping the model to no-route."""
        if not rule.exclude_regions:
            return _Routable(deployments)
        if entry is None:
            if rule.unverified_region_policy == "trust_fallback":
                trusted = frozenset(rule.cold_start_fallback)
                vetted = [
                    d
                    for d in deployments
                    if self.slugs_of(d)
                    and set(self.slugs_of(d)) <= trusted
                    and self._usable_slugs(model, self.slugs_of(d))
                ]
                if vetted:
                    return _Routable(vetted)
                self._warn_once(
                    f"region-untrusted:{model}",
                    f"openrouter-pareto: no deployment for {model} matches "
                    f"cold_start_fallback and is off cooldown, so none is usable; "
                    f"declining to route (unverified_region_policy=trust_fallback)\n",
                )
                return _Decline()
            if rule.unverified_region_policy == "unpinned":
                return _Unpinned(deployments)
            self._warn_once(
                f"region-no-route:{model}",
                f"openrouter-pareto: no region-verified provider for {model}; "
                f"declining to route (unverified_region_policy=no_route)\n",
            )
            return _Decline()
        eligible = [d for d in deployments if self._deployment_region_ok(model, rule, entry, d)]
        if eligible:
            return _Routable(eligible)
        if rule.unverified_region_policy == "unpinned":
            return _Unpinned(deployments)
        self._warn_once(
            f"region-all-excluded:{model}",
            f"openrouter-pareto: every healthy deployment for {model} is in an excluded "
            f"region; declining to route (unverified_region_policy="
            f"{rule.unverified_region_policy})\n",
        )
        return _Decline()

    def _deployment_region_ok(
        self,
        model: str,
        rule: Rule,
        entry: CacheEntry,
        deployment: dict[str, object],
    ) -> bool:
        """Affirmative eligibility for a pinned deployment. A deployment we cannot even
        identify is unknown, never allowed. Every slug it authorizes must pass, since
        `provider.only` lets OpenRouter serve from any entry in the list."""
        slugs = self.slugs_of(deployment)
        if not slugs:
            self._warn_once(
                f"region-unidentified:{model}",
                f"openrouter-pareto: a deployment for {model} has no resolvable provider "
                f"slug, so its region cannot be verified; skipping it\n",
            )
            return rule.allow_unknown_region
        return all(self._region_allows(model, rule, slug, entry) for slug in slugs)

    def _usable_slugs(self, model: str, slugs: tuple[str, ...]) -> bool:
        """Whether every slug a deployment authorizes is off cooldown. Pinned mode has
        to apply the same usability test as wildcard mode, or switching routing modes
        silently drops the 429 and input-cap mitigations."""
        return bool(slugs) and all(not self._cooldown.is_skipped(model, s) for s in slugs)

    def _narrow(
        self,
        model: str,
        rule: Rule,
        entry: CacheEntry | None,
        deployments: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        result = self._region_eligible(model, rule, entry, deployments)
        match result:
            case _Decline():
                return []
            case _Unpinned(deployments=unpinned):
                return self._unpinned_result(unpinned)
            case _Routable(deployments=allowed):
                pass
        if entry is None or entry.stale or entry.winner is None:
            # No winner selected: hand back the region-eligible deployments as they are.
            # Each already carries the operator's own pin; the plugin does not rewrite
            # the request, so the caller's own extra_body (default mode) or the strict
            # gate (strict mode) governs the provider, not this plugin.
            return allowed
        for slug in self._preferred_slugs(model, entry):
            match = [d for d in allowed if slug in self.slugs_of(d)]
            if match:
                return self._pin_winner(match, slug)
        allowlist = self._allowlist()
        if self._allowlist_allows(entry.winner, allowlist):
            winner_match = [d for d in allowed if entry.winner in self.slugs_of(d)]
            if winner_match:
                return self._pin_winner(winner_match, entry.winner)
        return allowed

    def _pin_winner(
        self,
        match: list[dict[str, object]],
        slug: str,
    ) -> list[dict[str, object]]:
        """Narrow the chosen deployments to a singleton pin on `slug` in the returned
        deployment COPIES. A deployment can authorize several providers, so matching one
        of them is not enough: without rewriting the copy, OpenRouter could still serve
        from a sibling entry that lost the value-walk. Deployments already pinned to
        exactly this slug are returned untouched (the router's own objects)."""
        if all(self.slugs_of(d) == (slug,) for d in match):
            return match
        return self._with_provider_only(match, slug)

    def _apply_rate_limit_signal(
        self,
        model: str,
        status: object,
        message: str,
        slug: str | None,
        rule: Rule | None,
    ) -> None:
        """Log the failure (if the rule wants it) and record the cooldown hit.

        Shared by the two failure hooks so they cannot drift on what counts as
        a 429 vs an input cap, or on the error-line shape. Returns without
        recording when the provider slug is unresolvable."""
        if isinstance(status, int) and status >= 400 and rule is not None and rule.log_errors:
            trimmed = message[:1000]
            or_error_log(f"model={model} slug={slug} status={status} body={trimmed}")
        if _is_no_allowed_providers_404(status, message):
            discovered = _parse_allowed_providers(message)
            if discovered is not None:
                self._telemetry.set_allowed_providers(discovered)
        if slug is None:
            return
        if status == 429:
            self._cooldown.record(model, slug)
        elif is_input_cap_error(status, message):
            self._cooldown.record_input_cap(model, slug)

    async def async_log_failure_event(
        self,
        kwargs: Mapping[str, object],
        response_obj: object,
        start_time: float,
        end_time: float,
    ) -> None:
        model = kwargs.get("model")
        if not isinstance(model, str):
            return
        model = _normalize_model_group(model)
        if not self._is_managed(model):
            return
        exc = kwargs.get("exception")
        if exc is None:
            return
        status = getattr(exc, "status_code", None)
        message = str(exc)
        slug = self._slug_from_kwargs(kwargs)
        rule = self._resolve_rules().get(model)
        self._apply_rate_limit_signal(model, status, message, slug, rule)

    def _upstream_message(
        self, request_data: Mapping[str, object], original_exception: object
    ) -> str:
        """The error text used for input-cap detection and the error log.

        The passthrough upstream-failure path supplies a synthetic HTTPException
        whose detail is only "Upstream passthrough request failed with status N";
        the real provider error (e.g. an input-length message) lives in
        ``request_data["response_body"]`` (the parsed upstream JSON litellm sets
        at pass_through_endpoints.py). Fall back to the exception string when no
        upstream body is available (e.g. an outer LiteLLM exception)."""
        rb = request_data.get("response_body")
        if isinstance(rb, Mapping):
            err = rb.get("error")
            if isinstance(err, Mapping):
                msg = err.get("message")
                if isinstance(msg, str) and msg:
                    return msg
            elif isinstance(err, str) and err:
                return err
            msg = rb.get("message")
            if isinstance(msg, str) and msg:
                return msg
        elif isinstance(rb, str) and rb:
            return rb
        return str(original_exception)

    async def async_post_call_failure_hook(
        self,
        request_data: Mapping[str, object],
        original_exception: object,
        user_api_key_dict: object,
        traceback_str: str | None = None,
    ) -> None:
        """The only failure signal litellm fires for a raw passthrough request.

        A passthrough forward is raw httpx; there is no litellm ``completion()``
        call, so ``Logging.async_failure_handler`` (which dispatches
        ``async_log_failure_event``) never runs. The only hook litellm fires on
        that path is ``async_post_call_failure_hook`` (proxy/utils.py post-call
        failure loop), so the 429 would never reach ``RateLimitCooldown.record``
        without this hook.

        Guarded to the passthrough path: on every other path (acompletion,
        anthropic_messages) ``async_log_failure_event`` already records, so
        recording here too would double-count and trip the 3-hit cooldown early.
        The passthrough failure ``request_data`` carries
        ``call_type="pass_through_endpoint"``.

        Limitation: 429s are detected by status code on both streaming and
        non-streaming passthrough. Input-cap (400) detection needs the upstream
        error text, which litellm attaches as ``request_data["response_body"]``
        only on the NON-streaming passthrough failure path; the streaming path
        dispatches this hook without it (reading the stream body would consume it
        and break the relay to the client). So input-cap is not detected for
        streaming passthrough - a known limitation. See ``_upstream_message``."""
        call_type = request_data.get("call_type")
        if call_type != "pass_through_endpoint":
            return None
        model = request_data.get("model")
        if not isinstance(model, str):
            return None
        model = _normalize_model_group(model)
        if not self._is_managed(model):
            return None
        status = getattr(original_exception, "status_code", None)
        message = self._upstream_message(request_data, original_exception)
        slug = self._slug_from_request_data(request_data)
        rule = self._resolve_rules().get(model)
        self._apply_rate_limit_signal(model, status, message, slug, rule)
        if slug is None and isinstance(status, int) and status >= 400:
            self._warn_once(
                f"postfail-no-slug:{model}",
                f"openrouter-pareto: passthrough failure on managed {model} "
                f"(status={status}) had no resolvable provider slug in request_data "
                f"(keys={sorted(request_data.keys())}); cooldown not recorded for "
                f"this request\n",
            )
        return None


openrouter_pareto_callback = OpenRouterParetoCallback()
