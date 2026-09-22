from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from functools import reduce
from itertools import groupby

from .config import Rule
from .models import EndpointEntry, StatsEndpoint

OR_UPTIME_FLOOR = 95.0


@dataclass(frozen=True, slots=True)
class Candidate:
    slug: str
    tag: str
    input_price_m: float
    output_price_m: float
    tps: float
    requests: int
    context: int
    uptime: float


@dataclass(frozen=True, slots=True)
class Point:
    """A scored candidate reduced to what the price/throughput frontier and walk need.
    This is what gets cached, so the frontier can be recomputed per request over the
    points still available rather than frozen at selection time - see `ranked_order`."""

    slug: str
    input_price_m: float
    tps: float


@dataclass(frozen=True, slots=True)
class Selection:
    winner: Candidate | None
    safe_set: tuple[str, ...]
    candidates: tuple[Candidate, ...]
    frontier: tuple[str, ...]
    points: tuple[Point, ...] = ()
    excluded_bases: tuple[str, ...] = ()
    allowed_bases: tuple[str, ...] = ()


_EMPTY = Selection(winner=None, safe_set=(), candidates=(), frontier=())


@dataclass(frozen=True, slots=True)
class _Tier:
    price: float
    best: Candidate


def _uptime_by_tag(endpoints: tuple[EndpointEntry, ...]) -> Mapping[str, float]:
    return {
        e.tag: e.uptime_last_5m
        for e in endpoints
        if e.uptime_last_5m is not None
    }


def _to_candidate(
    e: StatsEndpoint,
    rule: Rule,
    uptime_by_tag: Mapping[str, float],
) -> Candidate | None:
    if e.quantization is None or e.quantization not in rule.precision:
        return None
    if e.context_length is None or e.context_length < rule.min_context:
        return None
    if e.data_policy is None or e.data_policy.retains_prompts is not False:
        return None
    up = uptime_by_tag.get(e.provider_slug)
    if up is None or not math.isfinite(up) or up < OR_UPTIME_FLOOR:
        return None
    sample = e.stats
    if sample is None or sample.p50_throughput is None or sample.request_count is None:
        return None
    tps = float(sample.p50_throughput)
    if not math.isfinite(tps) or tps <= 0:
        return None
    if sample.request_count < rule.min_stats_requests:
        return None
    pricing = e.pricing
    if pricing is None or pricing.prompt is None or pricing.completion is None:
        return None
    input_price_m = float(pricing.prompt) * 1e6
    output_price_m = float(pricing.completion) * 1e6
    if not (math.isfinite(input_price_m) and math.isfinite(output_price_m)):
        return None
    if input_price_m <= 0:
        return None
    if rule.max_price is not None and input_price_m > rule.max_price:
        return None
    return Candidate(
        slug=e.provider_slug,
        tag=e.provider_slug,
        input_price_m=input_price_m,
        output_price_m=output_price_m,
        tps=tps,
        requests=int(sample.request_count),
        context=int(e.context_length),
        uptime=float(up),
    )


def _base_slug(slug: str) -> str:
    return slug.split("/", 1)[0]


def _norm_loc(value: str | None) -> str | None:
    """A provider_info location normalized for comparison, or None when absent. Compared
    verbatim against the operator's normalized `exclude_regions`; the plugin does not
    judge whether the string is a real country code, only whether the two sides match."""
    code = (value or "").strip().upper()
    return code or None


def _region_verdict(e: StatsEndpoint, excluded: frozenset[str]) -> str:
    """`excluded`, `allowed`, or `unknown`. Geography is only on the stats source
    (provider_info); the /endpoints catalogue carries neither headquarters nor
    datacenters, so absent metadata is genuinely unknown rather than compliant.

    `allowed` demands BOTH dimensions: where the provider is based and where it serves
    from. Roughly half of OpenRouter's live rows report `headquarters` while
    `datacenters` is null or empty, and a known headquarters says nothing about which
    datacenters serve the request, so partial metadata is unknown rather than allowed.
    An excluded location still wins outright: partial evidence is enough to reject,
    never enough to approve."""
    info = e.provider_info
    if info is None:
        return "unknown"
    hq = _norm_loc(info.headquarters)
    raw_dcs = info.datacenters
    dcs = tuple(_norm_loc(dc) for dc in (raw_dcs or ()))
    reported = tuple(loc for loc in (hq, *dcs) if loc is not None)
    if any(loc in excluded for loc in reported):
        return "excluded"
    if hq is not None and dcs and all(dc is not None for dc in dcs):
        return "allowed"
    return "unknown"


def _allowed_bases(stats: tuple[StatsEndpoint, ...], rule: Rule) -> frozenset[str]:
    """Org bases with affirmative evidence of eligibility: every endpoint of theirs
    that reports geography reports a non-excluded location, and (unless
    allow_unknown_region) at least one endpoint actually reported geography. Routing
    requires membership here, so a provider missing from telemetry is never assumed
    allowed."""
    if not rule.exclude_regions:
        return frozenset()
    excluded = frozenset(rule.exclude_regions)
    verdicts: dict[str, set[str]] = {}
    for e in stats:
        verdicts.setdefault(_base_slug(e.provider_slug), set()).add(_region_verdict(e, excluded))
    ok = {"allowed", "unknown"} if rule.allow_unknown_region else {"allowed"}
    return frozenset(
        base
        for base, seen in verdicts.items()
        if "excluded" not in seen and seen <= ok and seen
    )


def _excluded_bases(stats: tuple[StatsEndpoint, ...], rule: Rule) -> frozenset[str]:
    """Org bases to drop entirely: any org with an endpoint in an excluded region,
    plus (unless allow_unknown_region) any org whose geography cannot be determined.
    Absence of evidence is not evidence of compliance, so unknown is ineligible by
    default and operators opt in to the looser behavior."""
    if not rule.exclude_regions:
        return frozenset()
    excluded = frozenset(rule.exclude_regions)
    drop = {"excluded"} if rule.allow_unknown_region else {"excluded", "unknown"}
    return frozenset(
        _base_slug(e.provider_slug)
        for e in stats
        if _region_verdict(e, excluded) in drop
    )


def _stage1(
    stats: tuple[StatsEndpoint, ...],
    rule: Rule,
    uptime_by_tag: Mapping[str, float],
    allowed_providers: frozenset[str] | None = None,
) -> tuple[Candidate, ...]:
    excluded_bases = _excluded_bases(stats, rule)
    return tuple(
        c
        for c in (
            _to_candidate(e, rule, uptime_by_tag)
            for e in stats
            if _base_slug(e.provider_slug) not in excluded_bases
            and (
                allowed_providers is None
                or _base_slug(e.provider_slug) in allowed_providers
            )
        )
        if c is not None
    )


def _dedupe_by_slug(stage1: tuple[Candidate, ...]) -> tuple[Candidate, ...]:
    ordered = sorted(stage1, key=lambda c: (c.slug, -c.tps, c.input_price_m))
    return tuple(next(g) for _, g in groupby(ordered, key=lambda c: c.slug))


def _add_tier(tiers: tuple[_Tier, ...], p: Candidate) -> tuple[_Tier, ...]:
    if not tiers:
        return (_Tier(price=p.input_price_m, best=p),)
    current = tiers[-1]
    if abs(p.input_price_m - current.price) / current.price > 0.01:
        return (*tiers, _Tier(price=p.input_price_m, best=p))
    if p.tps > current.best.tps:
        return (*tiers[:-1], _Tier(price=current.price, best=p))
    return tiers


def _push_frontier(
    acc: tuple[tuple[Candidate, ...], float],
    tier_best: Candidate,
) -> tuple[tuple[Candidate, ...], float]:
    frontier, fastest = acc
    if tier_best.tps > fastest:
        return (*frontier, tier_best), tier_best.tps
    return acc


def _tier_and_frontier(
    candidates: tuple[Candidate, ...],
) -> tuple[tuple[Candidate, ...], tuple[str, ...]]:
    if not candidates:
        return (), ()
    sorted_pts = sorted(candidates, key=lambda c: (c.input_price_m, -c.tps))
    tiers = reduce(_add_tier, sorted_pts, ())
    frontier, _ = reduce(_push_frontier, (t.best for t in tiers), ((), -math.inf))
    return frontier, tuple(c.slug for c in frontier)


def _value_walk(
    frontier: tuple[Candidate, ...],
    tolerance: float,
) -> Candidate | None:
    if not frontier:
        return None

    def advance(winner: Candidate, candidate: Candidate) -> Candidate:
        price_increase = (candidate.input_price_m - winner.input_price_m) / winner.input_price_m
        speed_gain = (candidate.tps - winner.tps) / winner.tps
        return candidate if speed_gain + tolerance >= price_increase else winner

    return reduce(advance, frontier[1:], frontier[0])


def _frontier_of(points: tuple[Point, ...]) -> tuple[Point, ...]:
    """The pareto frontier over an arbitrary point set: apply the same tiering and
    monotonic-throughput reduction used at selection time, so a re-run over a subset
    yields exactly the frontier that subset would have produced."""
    if not points:
        return ()
    as_candidates = tuple(
        Candidate(
            slug=p.slug,
            tag=p.slug,
            input_price_m=p.input_price_m,
            output_price_m=0.0,
            tps=p.tps,
            requests=0,
            context=0,
            uptime=0.0,
        )
        for p in points
    )
    frontier, _ = _tier_and_frontier(as_candidates)
    return tuple(
        Point(slug=c.slug, input_price_m=c.input_price_m, tps=c.tps) for c in frontier
    )


def _walk_points(frontier: tuple[Point, ...], tolerance: float) -> Point | None:
    """`_value_walk` over Points. The acceptance rule is identical: move only when the
    percentage throughput gain plus the operator's tolerance covers the percentage
    price increase."""
    if not frontier:
        return None

    def advance(winner: Point, candidate: Point) -> Point:
        price_increase = (candidate.input_price_m - winner.input_price_m) / winner.input_price_m
        speed_gain = (candidate.tps - winner.tps) / winner.tps
        return candidate if speed_gain + tolerance >= price_increase else winner

    return reduce(advance, frontier[1:], frontier[0])


def _sanitized(points: tuple[Point, ...]) -> tuple[Point, ...]:
    """Drop points the walk cannot order (non-finite or non-positive coordinates) and
    collapse duplicate slugs to the fastest, tie-breaking on the cheaper price exactly
    as `_dedupe_by_slug` does. Both non-positive price and non-positive throughput are
    screened because both are division denominators downstream (`_add_tier`,
    `_walk_points`); a zero in either raises on the request path."""
    by_slug: dict[str, Point] = {}
    for p in points:
        if not (math.isfinite(p.input_price_m) and math.isfinite(p.tps)):
            continue
        if p.input_price_m <= 0 or p.tps <= 0:
            continue
        current = by_slug.get(p.slug)
        if current is None or (p.tps, -p.input_price_m) > (current.tps, -current.input_price_m):
            by_slug[p.slug] = p
    return tuple(by_slug.values())


def ranked_order(
    points: tuple[Point, ...],
    tolerance: float,
    *,
    exclude: frozenset[str] = frozenset(),
) -> tuple[str, ...]:
    """The order to try providers in: repeatedly take the value walk's terminal from
    the points still available, then remove it and re-run. Each entry is what the walk
    would have chosen had every earlier one been unavailable, which is what makes the
    operator's `value_regression_tolerance` apply to the fallback too.

    `exclude` drops providers that are already known-unusable (on cooldown, not
    allowlisted). Recomputing the frontier over the remaining points rather than
    freezing it at selection time is the point: a provider dominated by the current
    winner can be the right choice once that winner is gone, and a frontier computed
    with the winner present would never surface it.

    Ordering by the frontier's own price-ascending order would be wrong instead: the
    frontier ascends in price, and the cheap end of a pareto frontier is the slow end
    by construction, so a cheapest-first walk heads for the worst provider rather than
    the next-best one.

    Inputs are screened here rather than trusted: non-finite or non-positive
    coordinates are dropped and duplicate slugs collapse to the fastest (cheaper
    price breaking a tie), because the ordering below cannot order what it cannot
    compare. A NaN price makes the sort non-transitive and leaves the result
    implementation-defined; a NaN throughput empties the tier it lands in; a zero in
    either coordinate is a division denominator in `_add_tier` and `_walk_points`.
    Selection guarantees all of this upstream, and the cache re-checks finiteness on
    load, but neither covers a hand-built or hand-edited tuple - a defined answer
    beats a documented coin flip."""
    sanitized = _sanitized(points)
    available = tuple(p for p in sanitized if p.slug not in exclude)
    order: list[str] = []
    while available:
        winner = _walk_points(_frontier_of(available), tolerance)
        if winner is None:
            break
        order.append(winner.slug)
        available = tuple(p for p in available if p.slug != winner.slug)
    return tuple(order)


def select_candidates(
    stats_endpoints: tuple[StatsEndpoint, ...],
    uptime_endpoints: tuple[EndpointEntry, ...],
    rule: Rule,
    allowed_providers: frozenset[str] | None = None,
) -> Selection:
    uptime_by_tag = _uptime_by_tag(uptime_endpoints)
    stage1 = _stage1(stats_endpoints, rule, uptime_by_tag, allowed_providers)
    if not stage1:
        return replace(
            _EMPTY,
            excluded_bases=tuple(sorted(_excluded_bases(stats_endpoints, rule))),
            allowed_bases=tuple(sorted(_allowed_bases(stats_endpoints, rule))),
        )
    deduped = _dedupe_by_slug(stage1)
    frontier_candidates, frontier = _tier_and_frontier(deduped)
    winner = _value_walk(frontier_candidates, rule.value_regression_tolerance)
    cheap_first = sorted(deduped, key=lambda c: (c.input_price_m, -c.tps))
    safe_set = tuple(c.slug for c in cheap_first)
    return Selection(
        winner=winner,
        safe_set=safe_set,
        candidates=stage1,
        frontier=frontier,
        points=tuple(
            Point(slug=c.slug, input_price_m=c.input_price_m, tps=c.tps) for c in cheap_first
        ),
        excluded_bases=tuple(sorted(_excluded_bases(stats_endpoints, rule))),
        allowed_bases=tuple(sorted(_allowed_bases(stats_endpoints, rule))),
    )
