from __future__ import annotations

from dataclasses import dataclass


def _coerce_str_tuple(value: str | tuple[str, ...] | list[str]) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,)
    return tuple(value)


@dataclass(frozen=True, slots=True)
class Rule:
    precision: tuple[str, ...]
    min_context: int
    min_stats_requests: int
    promotion_polls: int
    value_regression_tolerance: float
    cold_start_fallback: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "precision", _coerce_str_tuple(self.precision))
        object.__setattr__(self, "promotion_polls", max(1, self.promotion_polls))
        object.__setattr__(
            self,
            "cold_start_fallback",
            _coerce_str_tuple(self.cold_start_fallback),
        )


def rule(
    *,
    precision: str | tuple[str, ...] | list[str],
    min_context: int,
    min_stats_requests: int,
    promotion_polls: int = 1,
    value_regression_tolerance: float = 0.0,
    cold_start_fallback: str | tuple[str, ...] | list[str] = (),
) -> Rule:
    return Rule(
        precision=_coerce_str_tuple(precision),
        min_context=min_context,
        min_stats_requests=min_stats_requests,
        promotion_polls=promotion_polls,
        value_regression_tolerance=value_regression_tolerance,
        cold_start_fallback=_coerce_str_tuple(cold_start_fallback),
    )


DEFAULT_RULES: dict[str, Rule] = {
    "z-ai/glm-5.2": rule(
        precision="fp8",
        min_context=1_000_000,
        min_stats_requests=100,
        promotion_polls=2,
        value_regression_tolerance=0.0,
        cold_start_fallback=["novita"],
    ),
}
