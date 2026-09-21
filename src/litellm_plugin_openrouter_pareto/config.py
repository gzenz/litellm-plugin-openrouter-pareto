from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field, TypeAdapter, field_validator, model_validator


def _coerce_str_tuple(value: str | tuple[str, ...] | list[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    return tuple(value)


UnverifiedRegionPolicy = Literal["no_route", "unpinned", "trust_fallback"]
_UNVERIFIED_REGION_POLICIES = frozenset({"no_route", "unpinned", "trust_fallback"})


def _normalize_regions(value: str | tuple[str, ...] | list[str]) -> tuple[str, ...]:
    """Strip, uppercase, and dedupe preserving order. Values are compared verbatim
    against whatever OpenRouter reports in provider_info (headquarters / datacenters),
    so an operator's `exclude_regions` and OR's field just have to agree on spelling;
    it is not this plugin's place to police which strings are valid country codes."""
    codes = tuple(r.strip().upper() for r in _coerce_str_tuple(value) if r.strip())
    return tuple(dict.fromkeys(codes))


def _reject_bool(v: object, *, integer: bool = False) -> object:
    """Reject YAML/JSON bool and non-numeric coercion before pydantic lax mode
    turns it into a real value: `max_price: true` would become 1.0 and install a
    $1/M ceiling from a typo. `bool` is an `int` subclass, so reject it
    explicitly. `integer=True` additionally rejects floats (e.g. a `3.0`
    threshold); otherwise ints are accepted for float fields (3 -> 3.0)."""
    if v is None:
        return v
    if integer:
        if isinstance(v, bool) or not isinstance(v, int):
            raise ValueError(f"must be an integer, not {type(v).__name__}")
        return v
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ValueError(f"must be a number, not {type(v).__name__}")
    return v


@dataclass(frozen=True, slots=True)
class Rule:
    precision: tuple[str, ...]
    min_context: int
    min_stats_requests: int
    promotion_polls: int
    value_regression_tolerance: float
    cold_start_fallback: tuple[str, ...]
    wildcard: bool = False
    log_errors: bool = False
    log_decisions: bool = False
    exclude_regions: tuple[str, ...] = ()
    allow_unknown_region: bool = False
    unverified_region_policy: UnverifiedRegionPolicy = "no_route"
    strict_provider: bool = False
    max_price: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "precision", _coerce_str_tuple(self.precision))
        object.__setattr__(self, "promotion_polls", max(1, self.promotion_polls))
        object.__setattr__(
            self,
            "cold_start_fallback",
            _coerce_str_tuple(self.cold_start_fallback),
        )
        object.__setattr__(self, "exclude_regions", _normalize_regions(self.exclude_regions))
        if self.unverified_region_policy not in _UNVERIFIED_REGION_POLICIES:
            raise ValueError(
                f"unverified_region_policy must be one of "
                f"{sorted(_UNVERIFIED_REGION_POLICIES)} (got {self.unverified_region_policy!r})"
            )
        if self.max_price is not None and isinstance(self.max_price, bool):
            raise ValueError(f"max_price must be a number, not bool (got {self.max_price!r})")
        if self.max_price is not None and (
            not math.isfinite(self.max_price) or self.max_price <= 0
        ):
            raise ValueError(f"max_price must be a finite > 0 (got {self.max_price!r})")


def rule(
    *,
    precision: str | tuple[str, ...] | list[str],
    min_context: int,
    min_stats_requests: int,
    promotion_polls: int = 1,
    value_regression_tolerance: float = 0.0,
    cold_start_fallback: str | tuple[str, ...] | list[str] = (),
    wildcard: bool = False,
    log_errors: bool = False,
    log_decisions: bool = False,
    exclude_regions: str | tuple[str, ...] | list[str] = (),
    allow_unknown_region: bool = False,
    unverified_region_policy: UnverifiedRegionPolicy = "no_route",
    strict_provider: bool = False,
    max_price: float | None = None,
) -> Rule:
    return Rule(
        precision=_coerce_str_tuple(precision),
        min_context=min_context,
        min_stats_requests=min_stats_requests,
        promotion_polls=promotion_polls,
        value_regression_tolerance=value_regression_tolerance,
        cold_start_fallback=_coerce_str_tuple(cold_start_fallback),
        wildcard=wildcard,
        log_errors=log_errors,
        log_decisions=log_decisions,
        exclude_regions=_coerce_str_tuple(exclude_regions),
        allow_unknown_region=allow_unknown_region,
        unverified_region_policy=unverified_region_policy,
        strict_provider=strict_provider,
        max_price=max_price,
    )


class RuleSpec(BaseModel):
    model_config = {"extra": "forbid"}
    precision: str | list[str] = "fp8"
    min_context: int = Field(default=1_000_000, ge=1)
    min_stats_requests: int = Field(default=100, ge=1)
    promotion_polls: int = Field(default=1, ge=1)
    value_regression_tolerance: float = Field(default=0.0, ge=0.0)
    cold_start_fallback: list[str] = Field(default_factory=list[str])
    wildcard: bool = False
    log_errors: bool = False
    log_decisions: bool = False
    exclude_regions: list[str] = Field(default_factory=list[str])
    allow_unknown_region: bool = False
    unverified_region_policy: UnverifiedRegionPolicy = "no_route"
    strict_provider: bool = False
    max_price: float | None = Field(default=None, gt=0, allow_inf_nan=False)

    @field_validator("precision")
    @classmethod
    def _precision_nonempty(cls, v: str | list[str]) -> str | list[str]:
        if isinstance(v, str):
            if not v:
                raise ValueError("precision must not be empty")
        elif not v:
            raise ValueError("precision must not be empty")
        return v

    @field_validator("max_price", mode="before")
    @classmethod
    def _max_price_no_bool(cls, v: object) -> object:
        return _reject_bool(v)

    def to_rule(self) -> Rule:
        return rule(
            precision=self.precision,
            min_context=self.min_context,
            min_stats_requests=self.min_stats_requests,
            promotion_polls=self.promotion_polls,
            value_regression_tolerance=self.value_regression_tolerance,
            cold_start_fallback=self.cold_start_fallback,
            wildcard=self.wildcard,
            log_errors=self.log_errors,
            log_decisions=self.log_decisions,
            exclude_regions=self.exclude_regions,
            allow_unknown_region=self.allow_unknown_region,
            unverified_region_policy=self.unverified_region_policy,
            strict_provider=self.strict_provider,
            max_price=self.max_price,
        )


def rule_fingerprint(r: Rule) -> str:
    """Stable digest of every field that changes candidate eligibility or winner
    selection. A cached selection is only reusable by a process whose rule has the
    same fingerprint, so one worker's permissive winner cannot leak into a stricter
    worker via the shared cache file."""
    payload = json.dumps(
        {
            "precision": list(r.precision),
            "min_context": r.min_context,
            "min_stats_requests": r.min_stats_requests,
            "value_regression_tolerance": r.value_regression_tolerance,
            "promotion_polls": r.promotion_polls,
            "exclude_regions": list(r.exclude_regions),
            "allow_unknown_region": r.allow_unknown_region,
            "unverified_region_policy": r.unverified_region_policy,
            "max_price": r.max_price,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class RuleConfigError(ValueError):
    pass


class TelemetryConfig(BaseModel):
    """Global (not per-model) telemetry-client settings. Lives outside the rules
    mapping because there is one HTTP client for all models, so SSL verification is
    a process-wide property of the telemetry fetch, not a selection criterion."""

    model_config = {"extra": "forbid"}
    ssl_verify: bool = True
    ssl_ca_cert: str | None = None

    @field_validator("ssl_ca_cert")
    @classmethod
    def _ssl_ca_nonempty(cls, v: str | None) -> str | None:
        if v is None:
            return v
        stripped = v.strip()
        if not stripped:
            raise ValueError("ssl_ca_cert must be a non-empty path to a CA bundle")
        return stripped

    @model_validator(mode="after")
    def _ssl_ca_requires_verification(self) -> TelemetryConfig:
        if self.ssl_ca_cert is not None and not self.ssl_verify:
            raise ValueError(
                "ssl_ca_cert requires ssl_verify=true; a custom CA bundle is only "
                "meaningful with verification enabled"
            )
        return self


def telemetry_verify(cfg: TelemetryConfig | None) -> bool | str:
    """Resolve a TelemetryConfig into the value httpx accepts as `verify`: True for
    default verification, False to disable, or a CA-bundle path to use that bundle.
    A custom CA bundle takes precedence over the boolean, since pinning a bundle is
    a strictly stronger statement than 'verify with the system roots'."""
    if cfg is None:
        return True
    if cfg.ssl_ca_cert is not None:
        return cfg.ssl_ca_cert
    return cfg.ssl_verify


class CooldownConfig(BaseModel):
    """Global (not per-model) cooldown tuning. Like telemetry, one cooldown store
    serves all models, so the window / threshold / input-cap TTL are process-wide
    properties of rate-limit handling, not per-rule selection criteria."""

    model_config = {"extra": "forbid"}
    rate_limit_window_s: float = Field(default=300.0, gt=0, allow_inf_nan=False)
    rate_limit_threshold: int = Field(default=3, ge=1)
    input_cap_ttl_s: float = Field(default=3600.0, gt=0, allow_inf_nan=False)

    @field_validator("rate_limit_window_s", "input_cap_ttl_s", mode="before")
    @classmethod
    def _float_no_bool(cls, v: object) -> object:
        return _reject_bool(v)

    @field_validator("rate_limit_threshold", mode="before")
    @classmethod
    def _int_no_bool(cls, v: object) -> object:
        return _reject_bool(v, integer=True)


class AllowlistConfig(BaseModel):
    """Global (not per-model) allowlist tuning. The allowlist is discovered from
    OpenRouter 404 error bodies and cached; the TTL controls how long the cached
    list is considered fresh before a new 404 probe re-discovers it."""

    model_config = {"extra": "forbid"}
    ttl_s: float = Field(default=86400.0, gt=0, allow_inf_nan=False)

    @field_validator("ttl_s", mode="before")
    @classmethod
    def _ttl_no_bool(cls, v: object) -> object:
        return _reject_bool(v)


_RULES_ADAPTER: TypeAdapter[dict[str, RuleSpec]] = TypeAdapter(dict[str, RuleSpec])
_TELEMETRY_ADAPTER: TypeAdapter[TelemetryConfig] = TypeAdapter(TelemetryConfig)
_COOLDOWN_ADAPTER: TypeAdapter[CooldownConfig] = TypeAdapter(CooldownConfig)
_ALLOWLIST_ADAPTER: TypeAdapter[AllowlistConfig] = TypeAdapter(AllowlistConfig)


def load_rules_from_settings() -> dict[str, Rule] | None:
    import litellm

    raw: object = getattr(litellm, "openrouter_pareto_rules", None)
    source = "litellm_settings.openrouter_pareto_rules"
    if raw is None:
        env = os.environ.get("OPENROUTER_PARETO_RULES")
        if env:
            import json

            try:
                raw = json.loads(env)
            except ValueError as exc:
                raise RuleConfigError(
                    f"OPENROUTER_PARETO_RULES is not valid JSON: {exc}"
                ) from exc
            source = "OPENROUTER_PARETO_RULES"
    if raw is None:
        return None
    if not isinstance(raw, dict) or not raw:
        raise RuleConfigError(
            f"{source} must be a non-empty mapping of model -> rule fields"
        )
    try:
        specs = _RULES_ADAPTER.validate_python(raw)
    except Exception as exc:
        raise RuleConfigError(f"invalid {source}: {exc}") from exc
    return {model: spec.to_rule() for model, spec in specs.items()}


def load_telemetry_config() -> TelemetryConfig | None:
    import litellm

    raw: object = getattr(litellm, "openrouter_pareto_telemetry", None)
    source = "litellm_settings.openrouter_pareto_telemetry"
    if raw is None:
        env = os.environ.get("OPENROUTER_PARETO_TELEMETRY")
        if env:
            import json

            try:
                raw = json.loads(env)
            except ValueError as exc:
                raise RuleConfigError(
                    f"OPENROUTER_PARETO_TELEMETRY is not valid JSON: {exc}"
                ) from exc
            source = "OPENROUTER_PARETO_TELEMETRY"
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise RuleConfigError(f"{source} must be a mapping of telemetry fields")
    try:
        return _TELEMETRY_ADAPTER.validate_python(raw)
    except Exception as exc:
        raise RuleConfigError(f"invalid {source}: {exc}") from exc


def load_cooldown_config() -> CooldownConfig | None:
    import litellm

    raw: object = getattr(litellm, "openrouter_pareto_cooldown", None)
    source = "litellm_settings.openrouter_pareto_cooldown"
    if raw is None:
        env = os.environ.get("OPENROUTER_PARETO_COOLDOWN")
        if env:
            import json

            try:
                raw = json.loads(env)
            except ValueError as exc:
                raise RuleConfigError(
                    f"OPENROUTER_PARETO_COOLDOWN is not valid JSON: {exc}"
                ) from exc
            source = "OPENROUTER_PARETO_COOLDOWN"
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise RuleConfigError(f"{source} must be a mapping of cooldown fields")
    try:
        return _COOLDOWN_ADAPTER.validate_python(raw)
    except Exception as exc:
        raise RuleConfigError(f"invalid {source}: {exc}") from exc


def load_allowlist_config() -> AllowlistConfig | None:
    import litellm

    raw: object = getattr(litellm, "openrouter_pareto_allowlist", None)
    source = "litellm_settings.openrouter_pareto_allowlist"
    if raw is None:
        env = os.environ.get("OPENROUTER_PARETO_ALLOWLIST")
        if env:
            import json

            try:
                raw = json.loads(env)
            except ValueError as exc:
                raise RuleConfigError(
                    f"OPENROUTER_PARETO_ALLOWLIST is not valid JSON: {exc}"
                ) from exc
            source = "OPENROUTER_PARETO_ALLOWLIST"
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise RuleConfigError(f"{source} must be a mapping of allowlist fields")
    try:
        return _ALLOWLIST_ADAPTER.validate_python(raw)
    except Exception as exc:
        raise RuleConfigError(f"invalid {source}: {exc}") from exc


DEFAULT_RULES: dict[str, Rule] = {
    "z-ai/glm-5.2": rule(
        precision="fp8",
        min_context=1_000_000,
        min_stats_requests=100,
        promotion_polls=2,
        value_regression_tolerance=0.0,
        cold_start_fallback=["novita/fp8"],
    ),
}
