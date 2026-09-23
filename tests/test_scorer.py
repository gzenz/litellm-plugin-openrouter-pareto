from __future__ import annotations

import pytest

from litellm_plugin_openrouter_pareto.config import Rule, rule
from litellm_plugin_openrouter_pareto.models import (
    EndpointEntry,
    StatsDataPolicy,
    StatsEndpoint,
    StatsPricing,
    StatsProviderInfo,
    StatsSample,
)
from litellm_plugin_openrouter_pareto.scorer import (
    Point,
    _sanitized,
    ranked_order,
    select_candidates,
)


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
    hq: str | None = None,
    datacenters: list[str] | None = None,
) -> StatsEndpoint:
    return StatsEndpoint(
        provider_slug=slug,
        quantization=quant,
        context_length=context,
        data_policy=StatsDataPolicy(retainsPrompts=retain),
        stats=StatsSample(p50_throughput=tps, request_count=requests),
        pricing=StatsPricing(prompt=prompt, completion=completion),
        provider_info=(
            StatsProviderInfo(headquarters=hq, datacenters=datacenters)
            if (hq is not None or datacenters is not None)
            else None
        ),
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
        cold_start_fallback=["novita/fp8"],
    )


def test_pareto_frontier_is_all_three_strictly_faster_than_cheaper() -> None:
    stats, ups = _live_fixtures()
    sel = select_candidates(stats, ups, _glm_rule())
    assert sel.frontier == ("novita/fp8", "siliconflow/fp8", "baseten/fp8")


def test_value_walk_picks_baseten_not_cheapest_or_fastest_blindly() -> None:
    stats, ups = _live_fixtures()
    sel = select_candidates(stats, ups, _glm_rule())
    assert sel.winner is not None
    assert sel.winner.slug == "baseten/fp8"
    cheapest = min(sel.candidates, key=lambda c: c.input_price_m)
    assert cheapest.slug == "novita/fp8"
    assert sel.winner.slug != cheapest.slug


def test_dominated_same_price_slower_providers_are_off_frontier() -> None:
    stats, ups = _live_fixtures()
    sel = select_candidates(stats, ups, _glm_rule())
    assert "venice/fp8" not in sel.frontier
    assert "z-ai/fp8" not in sel.frontier
    assert set(sel.safe_set) == {"novita/fp8", "siliconflow/fp8", "baseten/fp8", "venice/fp8", "z-ai/fp8"}


def test_siliconflow_rejected_on_marginal_speed_for_huge_price_premium() -> None:
    stats = (
        _stat("novita/fp8", 23.0, 0.000000623),
        _stat("siliconflow/fp8", 29.0, 0.00000119),
    )
    ups = (_up("novita/fp8"), _up("siliconflow/fp8", 99.0))
    sel = select_candidates(stats, ups, _glm_rule())
    assert sel.winner is not None
    assert sel.winner.slug == "novita/fp8"


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
    assert sel.winner.slug == "novita/fp8"
    assert "baseten/fp8" not in sel.safe_set


def test_low_uptime_dropped() -> None:
    stats = (
        _stat("novita/fp8", 23.0, 0.000000623),
        _stat("baseten/fp8", 57.0, 0.0000014),
    )
    ups = (_up("novita/fp8"), _up("baseten/fp8", 80.0))
    sel = select_candidates(stats, ups, _glm_rule())
    assert sel.winner is not None
    assert sel.winner.slug == "novita/fp8"
    assert "baseten/fp8" not in sel.safe_set


def test_under_sampled_dropped() -> None:
    stats = (
        _stat("novita/fp8", 23.0, 0.000000623, requests=10),
        _stat("baseten/fp8", 57.0, 0.0000014),
    )
    ups = (_up("novita/fp8"), _up("baseten/fp8"))
    sel = select_candidates(stats, ups, _glm_rule())
    assert sel.winner is not None
    assert sel.winner.slug == "baseten/fp8"
    assert "novita/fp8" not in sel.safe_set


def test_wrong_quantization_dropped() -> None:
    stats = (
        _stat("novita/fp8", 23.0, 0.000000623),
        _stat("baseten/gptq", 57.0, 0.0000014, quant="gptq"),
    )
    ups = (_up("novita/fp8"), _up("baseten/gptq"))
    sel = select_candidates(stats, ups, _glm_rule())
    assert sel.winner is not None
    assert sel.winner.slug == "novita/fp8"
    assert "baseten/fp8" not in sel.safe_set


def test_max_price_drops_provider_over_ceiling() -> None:
    """A provider whose input price per million exceeds max_price is dropped
    before the value walk. novita (0.623/m) is the only one under 1.0, so it
    wins outright; the pricier providers appear in neither frontier nor safe set."""
    stats, ups = _live_fixtures()
    r = rule(
        precision="fp8",
        min_context=1_000_000,
        min_stats_requests=100,
        max_price=1.0,
    )
    sel = select_candidates(stats, ups, r)
    assert sel.winner is not None
    assert sel.winner.slug == "novita/fp8"
    assert set(sel.safe_set) == {"novita/fp8"}
    assert sel.frontier == ("novita/fp8",)


def test_max_price_at_ceiling_passes() -> None:
    """The drop is strictly-greater-than; a provider priced exactly at the
    ceiling survives. baseten is 1.4/m, so max_price=1.4 keeps it."""
    stats, ups = _live_fixtures()
    r = rule(
        precision="fp8",
        min_context=1_000_000,
        min_stats_requests=100,
        max_price=1.4,
    )
    sel = select_candidates(stats, ups, r)
    assert "baseten/fp8" in sel.safe_set
    assert sel.winner is not None
    assert sel.winner.slug == "baseten/fp8"


def test_max_price_none_is_no_ceiling() -> None:
    """max_price=None (the default) keeps every provider regardless of price."""
    stats, ups = _live_fixtures()
    sel = select_candidates(stats, ups, _glm_rule())
    assert set(sel.safe_set) == {
        "novita/fp8",
        "siliconflow/fp8",
        "baseten/fp8",
        "venice/fp8",
        "z-ai/fp8",
    }


def test_max_price_rejects_nonpositive() -> None:
    from pydantic import ValidationError

    from litellm_plugin_openrouter_pareto.config import RuleSpec

    with pytest.raises(ValueError, match="max_price"):
        rule(precision="fp8", min_context=1_000_000, min_stats_requests=100, max_price=0)
    with pytest.raises(ValueError, match="max_price"):
        rule(precision="fp8", min_context=1_000_000, min_stats_requests=100, max_price=-1.5)
    # RuleSpec (the YAML/env adapter) bounds it too.
    with pytest.raises(ValidationError):
        RuleSpec(max_price=0)


def test_max_price_rejects_nonfinite() -> None:
    """A NaN silently disables the ceiling (input_price_m > NaN is always false)
    and infinity makes it ineffective; both are malformed operator values and
    must be rejected on every construction path, not silently accepted."""
    from pydantic import ValidationError

    from litellm_plugin_openrouter_pareto.config import RuleSpec

    for bad in (float("inf"), float("-inf"), float("nan")):
        with pytest.raises(ValueError, match="max_price"):
            rule(precision="fp8", min_context=1_000_000, min_stats_requests=100, max_price=bad)
    for bad in (float("inf"), float("nan")):
        with pytest.raises(ValidationError):
            RuleSpec(max_price=bad)


def test_max_price_rejects_bool_and_string() -> None:
    """Pydantic lax mode would turn `max_price: true` into 1.0 and install a $1/M
    ceiling from a typo, silently dropping every pricier provider. Reject bool and
    string on every construction path; accept a real int/float."""
    from pydantic import ValidationError

    from litellm_plugin_openrouter_pareto.config import RuleSpec

    # Direct construction: bool is a numeric subclass the type system permits, so
    # Rule.__post_init__ must reject it at runtime.
    for bad in (True, False):
        with pytest.raises(ValueError, match="max_price"):
            rule(precision="fp8", min_context=1_000_000, min_stats_requests=100, max_price=bad)
    # YAML / env path (dict, untyped): bool AND string must be rejected by the
    # before-validator, not coerced to a number.
    for bad in (True, False, "1.5", "60"):
        with pytest.raises(ValidationError):
            RuleSpec.model_validate({"max_price": bad})
    # real numeric values still pass.
    assert rule(precision="fp8", min_context=1_000_000, min_stats_requests=100, max_price=2).max_price == 2
    assert RuleSpec.model_validate({"max_price": 2}).max_price == 2.0


def test_exclude_providers_drops_full_slug() -> None:
    """An entry naming one endpoint removes exactly that endpoint: same org, other
    variants still eligible."""
    stats, ups = _live_fixtures()
    r = rule(
        precision="fp8",
        min_context=1_000_000,
        min_stats_requests=100,
        exclude_providers=["novita/fp8"],
    )
    sel = select_candidates(stats, ups, r)
    slugs = {c.slug for c in sel.candidates} | set(sel.safe_set)
    assert "novita/fp8" not in slugs
    assert "siliconflow/fp8" in slugs


def test_exclude_providers_org_base_drops_every_endpoint() -> None:
    """An entry naming an org removes every endpoint that org offers, without needing a
    wildcard syntax to say so."""
    stats, ups = _live_fixtures()
    r = rule(
        precision="fp8",
        min_context=1_000_000,
        min_stats_requests=100,
        exclude_providers=["novita"],
    )
    sel = select_candidates(stats, ups, r)
    slugs = {c.slug for c in sel.candidates} | set(sel.safe_set)
    assert not any(s.startswith("novita/") for s in slugs)
    assert "siliconflow/fp8" in slugs


def test_exclude_providers_empty_is_no_filtering() -> None:
    """The default is an empty blocklist, so behaviour is unchanged for every existing
    rule."""
    stats, ups = _live_fixtures()
    base = select_candidates(stats, ups, _glm_rule())
    excluded = select_candidates(
        stats,
        ups,
        rule(precision="fp8", min_context=1_000_000, min_stats_requests=100, exclude_providers=[]),
    )
    assert {c.slug for c in base.candidates} == {c.slug for c in excluded.candidates}


def test_exclude_providers_is_case_and_whitespace_insensitive() -> None:
    """OR slugs are lowercase by construction, so folding case cannot create a false
    match - it only rescues a mis-cased entry, which would otherwise silently fail to
    exclude the provider the operator meant."""
    stats, ups = _live_fixtures()
    for entry in ("Novita", "  NOVITA  "):
        r = rule(
            precision="fp8",
            min_context=1_000_000,
            min_stats_requests=100,
            exclude_providers=entry,
        )
        slugs = {c.slug for c in select_candidates(stats, ups, r).candidates}
        assert not any(s.startswith("novita/") for s in slugs), entry


def test_exclude_providers_dedupes_preserving_order() -> None:
    r = rule(
        precision="fp8",
        min_context=1_000_000,
        min_stats_requests=100,
        exclude_providers=[" Novita ", "novita", "baseten/fp8"],
    )
    assert r.exclude_providers == ("novita", "baseten/fp8")


def test_rule_fingerprint_includes_exclude_providers() -> None:
    """The blocklist changes candidate eligibility, so a cached winner chosen without it
    must not be served to a rule that has it."""
    from litellm_plugin_openrouter_pareto.config import rule_fingerprint

    base = rule(precision="fp8", min_context=1_000_000, min_stats_requests=100)
    differs = rule(
        precision="fp8",
        min_context=1_000_000,
        min_stats_requests=100,
        exclude_providers=["novita/fp8"],
    )
    assert rule_fingerprint(base) != rule_fingerprint(differs)


def test_rule_fingerprint_ignores_failure_classification_fields() -> None:
    """The 400 patterns and their TTL change what is recorded at request time, never
    which candidates a selection contains. Hashing them would orphan every cached entry
    for the model, and re-fetch the whole telemetry snapshot, each time an operator
    broadened a regex."""
    from litellm_plugin_openrouter_pareto.config import rule_fingerprint

    base = rule(precision="fp8", min_context=1_000_000, min_stats_requests=100)
    same = rule(
        precision="fp8",
        min_context=1_000_000,
        min_stats_requests=100,
        input_cap_patterns=["exceeds the maximum"],
        broken_provider_patterns=["provider returned error"],
        broken_provider_ttl_s=60.0,
    )
    assert rule_fingerprint(base) == rule_fingerprint(same)


def test_rule_fingerprint_includes_max_price() -> None:
    """max_price changes candidate eligibility, so it must invalidate a cached
    winner: a tighter ceiling has a different fingerprint than a looser one."""
    from litellm_plugin_openrouter_pareto.config import rule_fingerprint

    base = rule(precision="fp8", min_context=1_000_000, min_stats_requests=100)
    same = rule(
        precision="fp8", min_context=1_000_000, min_stats_requests=100, max_price=2.0
    )
    differs = rule(
        precision="fp8", min_context=1_000_000, min_stats_requests=100, max_price=1.0
    )
    assert rule_fingerprint(base) != rule_fingerprint(same)
    assert rule_fingerprint(same) != rule_fingerprint(differs)
    assert rule_fingerprint(base) == rule_fingerprint(
        rule(precision="fp8", min_context=1_000_000, min_stats_requests=100)
    )


def test_same_org_distinct_endpoints_both_survive_no_per_org_dedupe() -> None:
    stats = (
        _stat("baseten/fp8", 57.0, 0.0000014),
        _stat("baseten/gptq", 40.0, 0.0000014, quant="fp8"),
    )
    ups = (_up("baseten/fp8"), _up("baseten/gptq"))
    sel = select_candidates(stats, ups, _glm_rule())
    assert len(sel.candidates) == 2
    assert set(sel.safe_set) == {"baseten/fp8", "baseten/gptq"}
    assert sel.winner is not None
    assert sel.winner.tps == 57.0


def test_duplicate_full_slug_dedupes_keeps_fastest() -> None:
    stats = (
        _stat("baseten/fp8", 57.0, 0.0000014),
        _stat("baseten/fp8", 40.0, 0.0000014),
    )
    ups = (_up("baseten/fp8"),)
    sel = select_candidates(stats, ups, _glm_rule())
    assert sel.safe_set == ("baseten/fp8",)
    assert sel.winner is not None
    assert sel.winner.tps == 57.0


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
    assert strict_sel.winner.slug == "novita/fp8"
    assert tol_sel.winner is not None
    assert tol_sel.winner.slug == "baseten/fp8"


def test_output_price_captured_but_does_not_affect_winner() -> None:
    stats = (
        _stat("novita/fp8", 23.0, 0.000000623, completion=0.5),
        _stat("baseten/fp8", 57.0, 0.0000014, completion=0.000001),
    )
    ups = (_up("novita/fp8"), _up("baseten/fp8"))
    sel = select_candidates(stats, ups, _glm_rule())
    assert sel.winner is not None
    assert sel.winner.slug == "baseten/fp8"
    novita = next(c for c in sel.candidates if c.slug == "novita/fp8")
    assert novita.output_price_m == pytest.approx(0.5 * 1e6)


def test_zero_throughput_cheapest_dropped_no_division_by_zero() -> None:
    stats = (
        _stat("cheap/fp8", 0.0, 0.000000623),
        _stat("pricier/fp8", 50.0, 0.0000014),
    )
    ups = (_up("cheap/fp8"), _up("pricier/fp8"))
    sel = select_candidates(stats, ups, _glm_rule())
    assert "cheap/fp8" not in sel.safe_set
    assert sel.winner is not None
    assert sel.winner.slug == "pricier/fp8"


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


def _excl_us_rule() -> Rule:
    return rule(
        precision="fp8",
        min_context=1_000_000,
        min_stats_requests=100,
        promotion_polls=2,
        exclude_regions=["US"],
    )


def test_exclude_regions_drops_provider_headquartered_in_region() -> None:
    """An excluded headquarters rejects on its own, even with no datacenter data:
    partial evidence is enough to reject, never enough to approve."""
    stats = (
        _stat("novita/fp8", 23.0, 0.000000623, hq="SG", datacenters=["SG"]),
        _stat("baseten/fp8", 90.0, 0.0000001, hq="US", datacenters=[]),
    )
    ups = (_up("novita/fp8"), _up("baseten/fp8"))
    sel = select_candidates(stats, ups, _excl_us_rule())
    slugs = {c.slug for c in sel.candidates}
    assert "baseten/fp8" not in slugs
    assert "novita/fp8" in slugs
    assert sel.winner is not None and sel.winner.slug == "novita/fp8"


def test_exclude_regions_drops_provider_with_datacenter_in_region() -> None:
    """Real shape: headquarters outside the excluded region, datacenters inside it."""
    stats = (
        _stat("novita/fp8", 23.0, 0.000000623, hq="SG", datacenters=["SG"]),
        _stat("coreweave/fp8", 90.0, 0.0000001, hq="SG", datacenters=["SG", "US"]),
    )
    ups = (_up("novita/fp8"), _up("coreweave/fp8"))
    sel = select_candidates(stats, ups, _excl_us_rule())
    slugs = {c.slug for c in sel.candidates}
    assert "coreweave/fp8" not in slugs
    assert sel.winner is not None and sel.winner.slug == "novita/fp8"


def test_exclude_regions_drops_whole_org_base_not_just_matching_endpoint() -> None:
    """An excluded datacenter on one endpoint drops every endpoint of that org base."""
    stats = (
        _stat("novita/fp8", 23.0, 0.000000623, hq="SG", datacenters=["SG"]),
        _stat("coreweave/fp8", 90.0, 0.0000001, hq="SG", datacenters=["SG", "US"]),
        _stat("coreweave/fast", 95.0, 0.0000001, hq="SG", datacenters=["SG"]),
    )
    ups = (_up("novita/fp8"), _up("coreweave/fp8"), _up("coreweave/fast"))
    sel = select_candidates(stats, ups, _excl_us_rule())
    slugs = {c.slug for c in sel.candidates}
    assert "coreweave/fp8" not in slugs
    assert "coreweave/fast" not in slugs
    assert slugs == {"novita/fp8"}


def test_exclude_regions_is_case_insensitive() -> None:
    stats = (
        _stat("novita/fp8", 23.0, 0.000000623, hq="SG", datacenters=["SG"]),
        _stat("baseten/fp8", 90.0, 0.0000001, hq="us", datacenters=["us"]),
    )
    ups = (_up("novita/fp8"), _up("baseten/fp8"))
    lowercase_rule = rule(
        precision="fp8",
        min_context=1_000_000,
        min_stats_requests=100,
        exclude_regions=["us"],
    )
    sel = select_candidates(stats, ups, lowercase_rule)
    assert {c.slug for c in sel.candidates} == {"novita/fp8"}


def test_no_exclude_regions_keeps_all_providers() -> None:
    """Default (empty exclude_regions) must not filter on geography at all."""
    stats = (
        _stat("novita/fp8", 23.0, 0.000000623, hq="SG"),
        _stat("baseten/fp8", 90.0, 0.0000001, hq="US"),
    )
    ups = (_up("novita/fp8"), _up("baseten/fp8"))
    sel = select_candidates(stats, ups, _glm_rule())
    slugs = {c.slug for c in sel.candidates}
    assert slugs == {"novita/fp8", "baseten/fp8"}


def test_unknown_geography_excluded_by_default() -> None:
    """provider_info absent -> geography unknown. Absence of evidence is not evidence
    of compliance, so it is ineligible unless allow_unknown_region opts in."""
    stats = (
        _stat("novita/fp8", 23.0, 0.000000623, hq="SG", datacenters=["SG"]),
        _stat("mystery/fp8", 90.0, 0.0000001),
    )
    ups = (_up("novita/fp8"), _up("mystery/fp8"))
    sel = select_candidates(stats, ups, _excl_us_rule())
    assert {c.slug for c in sel.candidates} == {"novita/fp8"}


def test_unknown_geography_kept_when_allow_unknown_region() -> None:
    stats = (
        _stat("novita/fp8", 23.0, 0.000000623, hq="SG", datacenters=["SG"]),
        _stat("mystery/fp8", 90.0, 0.0000001),
    )
    ups = (_up("novita/fp8"), _up("mystery/fp8"))
    lenient = rule(
        precision="fp8",
        min_context=1_000_000,
        min_stats_requests=100,
        exclude_regions=["US"],
        allow_unknown_region=True,
    )
    sel = select_candidates(stats, ups, lenient)
    assert {c.slug for c in sel.candidates} == {"novita/fp8", "mystery/fp8"}


def test_unknown_geography_ignored_without_exclude_regions() -> None:
    """No region policy -> geography is irrelevant, unknown providers stay."""
    stats = (
        _stat("novita/fp8", 23.0, 0.000000623),
        _stat("mystery/fp8", 90.0, 0.0000001),
    )
    ups = (_up("novita/fp8"), _up("mystery/fp8"))
    sel = select_candidates(stats, ups, _glm_rule())
    assert {c.slug for c in sel.candidates} == {"novita/fp8", "mystery/fp8"}


def test_empty_geography_fields_count_as_unknown() -> None:
    """hq='' with datacenters=[] carries no location, so it is unknown, not allowed."""
    stats = (
        _stat("novita/fp8", 23.0, 0.000000623, hq="SG", datacenters=["SG"]),
        _stat("blank/fp8", 90.0, 0.0000001, hq="", datacenters=[]),
    )
    ups = (_up("novita/fp8"), _up("blank/fp8"))
    sel = select_candidates(stats, ups, _excl_us_rule())
    assert {c.slug for c in sel.candidates} == {"novita/fp8"}


def test_exclude_regions_also_removes_from_safe_set() -> None:
    """A hard filter: excluded providers leave the safe_set too, not just the winner."""
    stats = (
        _stat("novita/fp8", 23.0, 0.000000623, hq="SG", datacenters=["SG"]),
        _stat("baseten/fp8", 90.0, 0.0000001, hq="US", datacenters=["US"]),
        _stat("siliconflow/fp8", 57.0, 0.0000014, hq="SG", datacenters=["SG"], requests=1200),
    )
    ups = (_up("novita/fp8"), _up("baseten/fp8"), _up("siliconflow/fp8"))
    sel = select_candidates(stats, ups, _excl_us_rule())
    assert "baseten/fp8" not in sel.safe_set
    assert "novita/fp8" in sel.safe_set


def test_headquarters_known_but_datacenters_missing_is_unknown() -> None:
    """The dominant real shape: 19 of 33 live OR endpoints for glm-5.2 report
    headquarters with datacenters null or empty. A known HQ says nothing about where
    the request is actually served from, so it is unknown, not allowed."""
    stats = (
        _stat("novita/fp8", 23.0, 0.000000623, hq="SG", datacenters=["SG"]),
        _stat("deepinfra/fp8", 90.0, 0.0000001, hq="DE", datacenters=None),
    )
    ups = (_up("novita/fp8"), _up("deepinfra/fp8"))
    sel = select_candidates(stats, ups, _excl_us_rule())
    assert {c.slug for c in sel.candidates} == {"novita/fp8"}
    assert "deepinfra" not in sel.allowed_bases


def test_datacenters_known_but_headquarters_missing_is_unknown() -> None:
    stats = (
        _stat("novita/fp8", 23.0, 0.000000623, hq="SG", datacenters=["SG"]),
        _stat("ambient/fp8", 90.0, 0.0000001, hq=None, datacenters=["DE"]),
    )
    ups = (_up("novita/fp8"), _up("ambient/fp8"))
    sel = select_candidates(stats, ups, _excl_us_rule())
    assert {c.slug for c in sel.candidates} == {"novita/fp8"}
    assert "ambient" not in sel.allowed_bases


def test_excluded_datacenter_rejects_even_when_list_is_partly_malformed() -> None:
    """Rejection needs only one excluded location; malformed siblings do not rescue it."""
    stats = (
        _stat("novita/fp8", 23.0, 0.000000623, hq="SG", datacenters=["SG"]),
        _stat("mixed/fp8", 90.0, 0.0000001, hq="DE", datacenters=["US", "Germany"]),
    )
    ups = (_up("novita/fp8"), _up("mixed/fp8"))
    sel = select_candidates(stats, ups, _excl_us_rule())
    assert {c.slug for c in sel.candidates} == {"novita/fp8"}
    assert "mixed" in sel.excluded_bases


def test_partial_geography_is_routable_when_allow_unknown_region() -> None:
    """Operators who opt into unknown geography still get the partial-metadata rows."""
    lenient = rule(
        precision="fp8",
        min_context=1_000_000,
        min_stats_requests=100,
        exclude_regions=["US"],
        allow_unknown_region=True,
    )
    stats = (
        _stat("novita/fp8", 23.0, 0.000000623, hq="SG", datacenters=["SG"]),
        _stat("deepinfra/fp8", 90.0, 0.0000001, hq="DE", datacenters=None),
    )
    ups = (_up("novita/fp8"), _up("deepinfra/fp8"))
    sel = select_candidates(stats, ups, lenient)
    assert {c.slug for c in sel.candidates} == {"novita/fp8", "deepinfra/fp8"}


def test_ranked_order_live_fixture_walk_order() -> None:
    """The live fixture set: baseten wins, then novita (52% cheaper than siliconflow
    for 21% less throughput - the re-run walk declines the move), then venice (once
    novita is gone, venice's +38% throughput over siliconflow pays for +18% price),
    with the cheapest-and-slowest dominated provider (z-ai) last."""
    stats, ups = _live_fixtures()
    sel = select_candidates(stats, ups, _glm_rule())
    assert sel.winner is not None
    order = ranked_order(sel.points, _glm_rule().value_regression_tolerance)
    assert order == ("baseten/fp8", "novita/fp8", "venice/fp8", "siliconflow/fp8", "z-ai/fp8")


def test_ranked_order_walk_terminal_cheapest_is_first() -> None:
    """When the value-walk terminal IS the cheapest frontier point (the walk declined
    the move: +26% throughput does not pay +91% price), it leads the order."""
    stats = (
        _stat("novita/fp8", 23.0, 0.000000623),
        _stat("siliconflow/fp8", 29.0, 0.00000119),
    )
    ups = (_up("novita/fp8"), _up("siliconflow/fp8", 99.0))
    sel = select_candidates(stats, ups, _glm_rule())
    assert sel.winner is not None
    assert sel.winner.slug == "novita/fp8"
    order = ranked_order(sel.points, _glm_rule().value_regression_tolerance)
    assert order == ("novita/fp8", "siliconflow/fp8")


def test_ranked_order_dominated_by_excluded_winner_takes_over() -> None:
    """The option-b behavior: fireworks is dominated by coreweave, so the FULL-set
    walk's first pick is coreweave, but elimination removes coreweave from the set
    before the second pick - and the re-run walk then promotes fireworks first
    (+57% price for +230% throughput over deepinfra), ahead of every cheaper
    provider. This is exactly what a 429 on the winner should fall back to."""
    points = tuple(
        Point(slug=s, input_price_m=p, tps=t)
        for s, p, t in [
            ("morph", 0.120, 6.0),
            ("relace/fp4", 0.130, 24.0),
            ("deepinfra/fp8", 0.140, 43.0),
            ("coreweave/fp8", 0.200, 193.0),
            ("fireworks", 0.220, 142.0),
        ]
    )
    full = ranked_order(points, 0.0)
    assert full[0] == "coreweave/fp8"
    assert full[1] == "fireworks"
    without_winner = ranked_order(points, 0.0, exclude=frozenset({"coreweave/fp8"}))
    assert without_winner[0] == "fireworks"
    assert without_winner[1] == "deepinfra/fp8"


def test_ranked_order_empty_points() -> None:
    assert ranked_order((), 0.0) == ()
    assert ranked_order((), 0.5, exclude=frozenset({"a"})) == ()


def test_ranked_order_exclude_all_yields_empty() -> None:
    points = (Point(slug="a/fp8", input_price_m=1.0, tps=10.0),)
    assert ranked_order(points, 0.0, exclude=frozenset({"a/fp8"})) == ()


def test_ranked_order_drops_non_finite_points() -> None:
    """A point the walk cannot order is dropped rather than left to make the sort
    non-transitive: a non-finite coordinate has no defined position, so keeping it
    would make the result depend on input order. Selection and the cache both screen
    these upstream; this is the guard for a hand-built tuple."""
    nan = float("nan")
    points = (
        Point(slug="morph", input_price_m=0.12, tps=6.0),
        Point(slug="nanprice", input_price_m=nan, tps=500.0),
        Point(slug="baseten", input_price_m=0.20, tps=193.0),
        Point(slug="infprice", input_price_m=float("inf"), tps=900.0),
        Point(slug="nantps", input_price_m=0.13, tps=nan),
    )
    order = ranked_order(points, 0.0)
    assert "nanprice" not in order
    assert "infprice" not in order
    assert "nantps" not in order
    assert order == ("baseten", "morph")


def test_ranked_order_non_finite_dropped_regardless_of_input_order() -> None:
    """The point of screening: the answer no longer depends on where the bad point
    sat in the tuple. Before screening, a NaN price at max tps collapsed the frontier
    to itself and a NaN price at low tps was selected last - same input, different
    answer, decided by tuple position."""
    nan = float("nan")
    base = [Point(slug="a", input_price_m=0.10, tps=10.0),
            Point(slug="b", input_price_m=0.30, tps=50.0)]
    bad = Point(slug="nan", input_price_m=nan, tps=500.0)
    assert ranked_order(tuple([bad, *base]), 0.0) == ranked_order(tuple([*base, bad]), 0.0)
    assert ranked_order(tuple([bad, *base]), 0.0) == ranked_order(tuple(base), 0.0)


def test_ranked_order_collapses_duplicate_slugs_to_the_fastest() -> None:
    """A duplicate slug must not silently drop the other point; the faster one wins,
    matching `_dedupe_by_slug` at selection time."""
    order = ranked_order(
        (
            Point(slug="dup", input_price_m=0.1, tps=10.0),
            Point(slug="dup", input_price_m=0.2, tps=20.0),
        ),
        0.0,
    )
    assert order == ("dup",)


def test_ranked_order_exclude_re_walks_instead_of_filtering_the_order() -> None:
    """The fallback must re-run the walk over available points, not filter a full-set
    order: `c` is only justified by the `e` stepping stone, so with `e` unavailable the
    walk over what remains picks `a`, while filtering the full order would have left
    `c` first - a provider the available-set walk rejects."""
    points = (
        Point(slug="a", input_price_m=1.0, tps=10.0),
        Point(slug="e", input_price_m=1.4, tps=13.1),
        Point(slug="c", input_price_m=1.96, tps=17.16),
    )
    assert ranked_order(points, 0.1) == ("c", "e", "a")
    assert ranked_order(points, 0.1, exclude=frozenset({"e"})) == ("a", "c")


def test_ranked_order_drops_non_positive_tps() -> None:
    """A zero throughput is a division denominator in `_walk_points`, so it must be
    screened like a zero price: keeping it raises ZeroDivisionError on the request
    path when the point becomes the walk incumbent."""
    points = (
        Point(slug="zero", input_price_m=0.1, tps=0.0),
        Point(slug="neg", input_price_m=0.2, tps=-5.0),
        Point(slug="b", input_price_m=0.3, tps=50.0),
    )
    assert ranked_order(points, 0.0) == ("b",)


def test_ranked_order_duplicate_tie_breaks_on_cheaper_price() -> None:
    """At equal throughput the cheaper point survives, matching `_dedupe_by_slug` at
    selection time - otherwise the collapsed choice would depend on input order, which
    is the position-dependence this screening exists to remove."""
    cheap = Point(slug="dup", input_price_m=0.15, tps=20.0)
    dear = Point(slug="dup", input_price_m=0.20, tps=20.0)
    assert _sanitized((dear, cheap)) == (cheap,)
    assert _sanitized((cheap, dear)) == (cheap,)
