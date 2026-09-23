"""The routing decision record and its one-line rendering.

A decision is recorded per managed request, not its outcome: litellm already logs
outcomes, and correlating the two would need a request id plus a bounded in-process
map for no added insight into the plugin's own behavior. What this answers is the
question the error log cannot - "which branch chose this provider, and why" - for a
model that never left cold start, a request that landed on a pricier provider, or a
region policy that dropped everything.

Nothing written here is request content: the model is a model id, the slug comes
from config or telemetry, and the region is derived geography. Same redaction
posture as the error log.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

# The closed set of reasons. Every terminal exit of a routing decision names one of
# these, so a branch cannot be added without naming how it chose.
WINNER = "winner"
SAFE_SET = "safe_set"
INPUT_CAP = "input_cap"
# The cached winner is currently marked broken (a 400 body matched one of the rule's
# `broken_provider_patterns`), so the walk moved past it. Unlike most reasons this
# describes the provider that was REJECTED rather than the one chosen, and it states that
# provider's recorded state rather than a causal claim - a winner that is also 429-hot is
# reported here too. It is called out separately from `safe_set` because it is the one
# skip cause an operator has to act on: a cooldown expires and a region verdict is a
# policy, whereas a broken provider stays broken.
PROVIDER_BROKEN = "provider_broken"
COLD_START = "cold_start"
ALL_HOT = "all_hot"
UNPINNED = "unpinned"
DECLINED = "declined"
PASS_THROUGH = "pass_through"

# The closed set of region verdicts, from `_region_verdict`.
REGION_ALLOWED = "allowed"
REGION_EXCLUDED = "excluded"
REGION_UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class Decision:
    """What the plugin decided for one managed request: the provider slug it routed
    to (None when it declined, or deliberately pinned nothing), the reason the branch
    chose it, and a region verdict.

    `slug` and `region` answer different questions and are None when a different one
    was asked. `slug` is None for the decisions that pin nothing: a decline, a
    deliberate cede to OpenRouter under the `unpinned` policy, and the `pass_through`
    that hands a deployment back still carrying the operator's own pin. `region` is
    the verdict on the provider this decision turned on - the pinned slug, or the
    providers a decline refused - and is None when the rule has no `exclude_regions`
    (there is no region policy to have a verdict about) or when the decline came from
    cooldowns rather than from region."""

    slug: str | None
    reason: str
    region: str | None


def format_decision(
    model: str,
    mode: str,
    decision: Decision,
    *,
    now: datetime | None = None,
) -> str:
    """One line per decision: when, which model, which routing mode, what was chosen,
    why, and the region verdict behind it. `-` stands in for an absent value so the
    field count is fixed and the line stays greppable."""
    stamp = (now if now is not None else datetime.now(timezone.utc)).isoformat(
        timespec="seconds"
    )
    return (
        f"{stamp} model={model} mode={mode} "
        f"slug={decision.slug if decision.slug else '-'} "
        f"decision={decision.reason} "
        f"region={decision.region if decision.region else '-'}"
    )
