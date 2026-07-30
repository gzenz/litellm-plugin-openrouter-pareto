from __future__ import annotations

import pytest

from litellm_plugin_openrouter_pareto.config import Rule, rule
from litellm_plugin_openrouter_pareto.models import (
    EndpointEntry,
    StatsDataPolicy,
    StatsEndpoint,
    StatsPricing,
    StatsSample,
)
from litellm_plugin_openrouter_pareto.scorer import select_candidates


def _stat(
    slug: str,
    tps: float,
    prompt: float,
    *,
    quant: str = "fp8",
    context: int = 1_000_000,
    retain: bool = False,
    requests: int = 500,
    completion: float = 0.0000056,
) -> StatsEndpoint:
    return StatsEndpoint(
        provider_slug=slug,
        quantization=quant,
        context_length=context,
        data_policy=StatsDataPolicy(retainsPrompts=retain),
        stats=StatsSample(p50_throughput=tps, request_count=requests),
        pricing=StatsPricing(prompt=prompt, completion=completion),
    )


def _up(tag: str, uptime: float = 99.7) -> EndpointEntry:
    return EndpointEntry(tag=tag, uptime_last_5m=uptime)


def _live_fixtures() -> tuple[tuple[StatsEndpoint, ...], tuple[EndpointEntry, ...]]:
    stats = (
        _stat("novita/fp8", 23.0, 0.000000623),
        _stat("siliconflow/fp8", 29.0, 0.00000119),
        _stat("baseten/fp8", 57.0, 0.0000014, requests=1200),
        _stat("venice/fp8", 40.0, 0.0000014),
        _stat("z-ai/fp8", 30.0, 0.0000014),
    )
    ups = (
        _up("novita/fp8"),
        _up("siliconflow/fp8", 99.0),
        _up("baseten/fp8"),
        _up("venice/fp8"),
        _up("z-ai/fp8"),
    )
    return stats, ups


def _glm_rule() -> Rule:
    return rule(
        precision="fp8",
        min_context=1_000_000,
        min_stats_requests=100,
        promotion_polls=2,
        cold_start_fallback=["novita"],
    )


def test_pareto_frontier_is_all_three_strictly_faster_than_cheaper() -> None:
    stats, ups = _live_fixtures()
    sel = select_candidates(stats, ups, _glm_rule())
    assert sel.frontier == ("novita", "siliconflow", "baseten")


def test_value_walk_picks_baseten_not_cheapest_or_fastest_blindly() -> None:
    stats, ups = _live_fixtures()
    sel = select_candidates(stats, ups, _glm_rule())
    assert sel.winner is not None
    assert sel.winner.slug == "baseten"
    cheapest = min(sel.candidates, key=lambda c: c.input_price_m)
    assert cheapest.slug == "novita"
    assert sel.winner.slug != cheapest.slug


def test_dominated_same_price_slower_providers_are_off_frontier() -> None:
    stats, ups = _live_fixtures()
    sel = select_candidates(stats, ups, _glm_rule())
    assert "venice" not in sel.frontier
    assert "z-ai" not in sel.frontier
    assert set(sel.safe_set) == {"novita", "siliconflow", "baseten", "venice", "z-ai"}


def test_siliconflow_rejected_on_marginal_speed_for_huge_price_premium() -> None:
    stats = (
        _stat("novita/fp8", 23.0, 0.000000623),
        _stat("siliconflow/fp8", 29.0, 0.00000119),
    )
    ups = (_up("novita/fp8"), _up("siliconflow/fp8", 99.0))
    sel = select_candidates(stats, ups, _glm_rule())
    assert sel.winner is not None
    assert sel.winner.slug == "novita"


def test_zero_candidates_when_all_fail_hard_filters() -> None:
    stats = (_stat("novita/fp8", 23.0, 0.000000623, retain=True),)
    ups = (_up("novita/fp8"),)
    sel = select_candidates(stats, ups, _glm_rule())
    assert sel.winner is None
    assert sel.safe_set == ()
    assert sel.candidates == ()
    assert sel.frontier == ()


def test_zdr_violator_dropped() -> None:
    stats = (
        _stat("novita/fp8", 23.0, 0.000000623, retain=False),
        _stat("baseten/fp8", 57.0, 0.0000014, retain=True),
    )
    ups = (_up("novita/fp8"), _up("baseten/fp8"))
    sel = select_candidates(stats, ups, _glm_rule())
    assert sel.winner is not None
    assert sel.winner.slug == "novita"
    assert "baseten" not in sel.safe_set


def test_low_uptime_dropped() -> None:
    stats = (
        _stat("novita/fp8", 23.0, 0.000000623),
        _stat("baseten/fp8", 57.0, 0.0000014),
    )
    ups = (_up("novita/fp8"), _up("baseten/fp8", 80.0))
    sel = select_candidates(stats, ups, _glm_rule())
    assert sel.winner is not None
    assert sel.winner.slug == "novita"
    assert "baseten" not in sel.safe_set


def test_under_sampled_dropped() -> None:
    stats = (
        _stat("novita/fp8", 23.0, 0.000000623, requests=10),
        _stat("baseten/fp8", 57.0, 0.0000014),
    )
    ups = (_up("novita/fp8"), _up("baseten/fp8"))
    sel = select_candidates(stats, ups, _glm_rule())
    assert sel.winner is not None
    assert sel.winner.slug == "baseten"
    assert "novita" not in sel.safe_set


def test_wrong_quantization_dropped() -> None:
    stats = (
        _stat("novita/fp8", 23.0, 0.000000623),
        _stat("baseten/gptq", 57.0, 0.0000014, quant="gptq"),
    )
    ups = (_up("novita/fp8"), _up("baseten/gptq"))
    sel = select_candidates(stats, ups, _glm_rule())
    assert sel.winner is not None
    assert sel.winner.slug == "novita"
    assert "baseten" not in sel.safe_set


def test_dedupe_keeps_fastest_per_base_slug_for_frontier_and_winner() -> None:
    stats = (
        _stat("baseten/fp8", 57.0, 0.0000014),
        _stat("baseten/gptq", 40.0, 0.0000014, quant="fp8"),
    )
    ups = (_up("baseten/fp8"), _up("baseten/gptq"))
    sel = select_candidates(stats, ups, _glm_rule())
    assert len(sel.candidates) == 2
    assert sel.winner is not None
    assert sel.winner.tps == 57.0
    assert sel.frontier == ("baseten",)
    assert sel.safe_set == ("baseten",)


def test_value_regression_tolerance_promotes_marginal_speed() -> None:
    stats = (
        _stat("novita/fp8", 23.0, 0.000000623),
        _stat("baseten/fp8", 24.0, 0.000000700),
    )
    ups = (_up("novita/fp8"), _up("baseten/fp8"))
    strict = rule(precision="fp8", min_context=1_000_000, min_stats_requests=100, promotion_polls=1)
    tolerant = rule(
        precision="fp8",
        min_context=1_000_000,
        min_stats_requests=100,
        promotion_polls=1,
        value_regression_tolerance=0.10,
    )
    strict_sel = select_candidates(stats, ups, strict)
    tol_sel = select_candidates(stats, ups, tolerant)
    assert strict_sel.winner is not None
    assert strict_sel.winner.slug == "novita"
    assert tol_sel.winner is not None
    assert tol_sel.winner.slug == "baseten"


def test_output_price_captured_but_does_not_affect_winner() -> None:
    stats = (
        _stat("novita/fp8", 23.0, 0.000000623, completion=0.5),
        _stat("baseten/fp8", 57.0, 0.0000014, completion=0.000001),
    )
    ups = (_up("novita/fp8"), _up("baseten/fp8"))
    sel = select_candidates(stats, ups, _glm_rule())
    assert sel.winner is not None
    assert sel.winner.slug == "baseten"
    novita = next(c for c in sel.candidates if c.slug == "novita")
    assert novita.output_price_m == pytest.approx(0.5 * 1e6)


def test_zero_throughput_cheapest_dropped_no_division_by_zero() -> None:
    stats = (
        _stat("cheap/fp8", 0.0, 0.000000623),
        _stat("pricier/fp8", 50.0, 0.0000014),
    )
    ups = (_up("cheap/fp8"), _up("pricier/fp8"))
    sel = select_candidates(stats, ups, _glm_rule())
    assert "cheap" not in sel.safe_set
    assert sel.winner is not None
    assert sel.winner.slug == "pricier"


def test_nan_throughput_dropped() -> None:
    stats = (_stat("solo/fp8", float("nan"), 0.000000623),)
    ups = (_up("solo/fp8"),)
    sel = select_candidates(stats, ups, _glm_rule())
    assert sel.winner is None
    assert sel.safe_set == ()
    assert sel.frontier == ()


def test_nan_uptime_dropped() -> None:
    stats = (_stat("solo/fp8", 50.0, 0.000000623),)
    ups = (_up("solo/fp8", float("nan")),)
    sel = select_candidates(stats, ups, _glm_rule())
    assert sel.winner is None
    assert sel.safe_set == ()
