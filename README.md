# litellm-plugin-openrouter-pareto

A LiteLLM callback that picks the best-value OpenRouter provider per request when a
model is served by several upstream providers. For each request it applies hard
filters (quantization, minimum context, zero-data-retention, 5-minute uptime,
minimum sample size), builds a pareto frontier over input price per million tokens
vs P50 throughput, and walks it greedily to a single winner, with flap damping so the
choice does not thrash.

## What it does

- **Route to the best-value provider.** Per request, narrow a model group to the one
  OpenRouter provider that wins the price-vs-throughput value walk.
- **Reroute around rate-limited providers.** A provider that keeps returning 429 is
  skipped for new requests until its rate limits clear.
- **Skip providers that reject oversized input.** A provider that 400s on a too-large
  prompt is remembered and skipped, so later requests route around it.
- **Exclude providers by region.** Keep a model off providers based in or serving from
  regions you name (best-effort, from OpenRouter's reported geography).
- **Make routing authoritative.** Optionally reject requests that try to pin their own
  provider, so the plugin's choice cannot be overridden per request.

When the plugin rewrites a route (wildcard injection, the unpinned cold-start
fallback, or narrowing a multi-provider deployment to the winner) it also sets a
zero-data-retention, no-provider-fallback posture (`zdr: true`,
`allow_fallbacks: false`). It does not add these to a pinned deployment it hands back
unchanged, so set `zdr: true, allow_fallbacks: false` in each pin (as the pinned-mode
example does) if you want that posture there.

Each use case below is opt-in through a per-model rule; the sections show the config
for each. Start with install and the quick start, then add the rules you need.

## Install

```
pip install litellm-plugin-openrouter-pareto
```

Register the module-level callback instance in `litellm_settings.callbacks` using a
dotted path (the tail must be an instance, not the class):

```yaml
litellm_settings:
  callbacks: ["litellm_plugin_openrouter_pareto.plugin.openrouter_pareto_callback"]
```

The proxy must use an async-native routing strategy (the default `simple-shuffle`
works) so the deployment filter runs. Sync `Router.completion()` and non-async-native
strategies silently skip the callback.

## Quick start

The simplest setup is wildcard mode: one deployment per model, and the plugin injects
the winning provider per request. New OpenRouter providers become usable the moment
they appear, with no per-provider config to maintain.

```yaml
litellm_settings:
  callbacks: ["litellm_plugin_openrouter_pareto.plugin.openrouter_pareto_callback"]
  openrouter_pareto_rules:
    "z-ai/glm-5.2":
      precision: ["fp8"]
      min_context: 1000000
      min_stats_requests: 100
      wildcard: true
model_list:
  - model_name: z-ai/glm-5.2
    litellm_params:
      model: openrouter/z-ai/glm-5.2
    model_info:
      id: or-wildcard
```

That is enough to get best-value routing plus the 429 and input-cap fallbacks, which
are always on for a managed model. The sections below cover each use case in turn.

## Use case: route to the best-value provider

There are two ways to set this up. Both give the same selection; they differ in how
you manage the provider list.

### Wildcard mode (one deployment)

Set `wildcard: true` (shown in the quick start above). The plugin injects the
value-walk winner's `provider.only` on a single deployment per request. There is no
per-provider deployment to maintain, and providers OpenRouter adds later are eligible
immediately.

### Pinned mode (one deployment per provider)

Set `wildcard: false` (the default) and add one deployment per provider endpoint you
want to allow, each pinned via `extra_body.provider.only`. The plugin narrows the
group to the winning deployment.

```yaml
litellm_settings:
  callbacks: ["litellm_plugin_openrouter_pareto.plugin.openrouter_pareto_callback"]
  openrouter_pareto_rules:
    "z-ai/glm-5.2":
      precision: ["fp8"]
      min_context: 1000000
      min_stats_requests: 100
model_list:
  - model_name: z-ai/glm-5.2
    litellm_params:
      model: openrouter/z-ai/glm-5.2
      extra_body:
        provider: {only: [baseten/fp8], zdr: true, allow_fallbacks: false, quantizations: [fp8]}
    model_info:
      id: or-baseten/fp8
  - model_name: z-ai/glm-5.2
    litellm_params:
      model: openrouter/z-ai/glm-5.2
      extra_body:
        provider: {only: [novita/fp8], zdr: true, allow_fallbacks: false, quantizations: [fp8]}
    model_info:
      id: or-novita/fp8
```

The pin is the **full endpoint slug** (e.g. `baseten/fp8`, not just `baseten`) because
a provider can run two same-quant endpoints with different context caps; the full slug
keeps the min-context guarantee on the wire. The plugin matches a deployment to a
winner by reading `extra_body.provider.only` first (the authoritative pin), then
`model_info.id` (stripping an `or-` prefix). Every entry in `provider.only` is
evaluated, not just the first, and a deployment that authorizes several providers is
rewritten to a singleton pin on the winner; `model_info.id` is consulted only when
`provider.only` is absent.

Pinned mode gives you an explicit allowlist of providers; wildcard mode trades that
for zero maintenance and instant availability of new providers. Use pinned when you
want to curate exactly which providers a model may touch.

## Use case: reroute around rate-limited providers (429)

Always on for a managed model. A provider returning HTTP 429 is treated as a
transitory upstream rate limit, handled in two layers:

- **Per-request walk.** LiteLLM puts the 429'ing deployment in cooldown immediately
  and its retry re-runs deployment selection, so the next attempt lands on a different
  provider. Bound this with `num_retries` on the model group.
- **Cross-request cooldown.** The plugin tracks 429s per provider. A provider that
  429s three or more times within five minutes is "hot" and is skipped as the winner
  for new requests until its 429s age out of the window. This is the memory LiteLLM's
  short per-429 cooldown lacks; it stops a persistently rate-limited winner from being
  retried first on every new request.

```yaml
litellm_settings:
  num_retries: 3   # bounds the per-request walk across providers
```

The cross-request cooldown is in-process and not persisted; it resets on restart.
When every provider is hot, the plugin keeps the constrained winner rather than
emitting unconstrained routing.

## Use case: skip providers that reject oversized input (400)

Always on for a managed model. Some providers advertise a context they cannot actually
accept as input (e.g. BaseTen's `baseten/fp8` lists 1M context but caps input at ~524K
tokens, 400'ing any larger request regardless of `max_tokens`). On a 400 whose body
matches "exceeds the maximum ... length" or "maximum context length", the provider
slug is marked input-capped and skipped for the winner pick on later requests. One
observation is enough to start skipping it.

The skip lasts one hour (`INPUT_CAP_TTL_S`), then the provider is retried. A workload
that keeps sending oversized prompts will re-cap it each hour, costing one wasted 400
per provider per hour; a workload whose prompts have shrunk lets it back in.

Unlike a 429, LiteLLM does **not** retry a 400 by default, so the same-request reroute
is not automatic. To get the per-request walk on a too-large 400, add a retry policy
that retries `BadRequestError`:

```yaml
litellm_settings:
  num_retries: 3
  retry_policy:
    BadRequestErrorRetries: 3
```

Without this, the first oversized request to a capped provider fails (one wasted call),
and subsequent requests route around it via the cooldown. The input-cap state is
per-process and also resets on restart; with multiple workers each worker re-learns
independently.

## Use case: exclude providers by region

Set `exclude_regions` to keep a model off providers based in or serving from regions
you name. For example, to keep a model off providers in Singapore:

```yaml
litellm_settings:
  openrouter_pareto_rules:
    "z-ai/glm-5.2":
      precision: ["fp8"]
      min_context: 1000000
      min_stats_requests: 100
      exclude_regions: ["SG"]
      allow_unknown_region: false          # default; true keeps unknown-geography providers
      unverified_region_policy: "no_route"  # default; see the cold-start table below
```

Entries are normalized (trimmed, upper-cased, de-duplicated) and compared verbatim
against the location strings OpenRouter reports in `provider_info`, so your config and
OpenRouter's data only have to agree on spelling. The plugin does not judge which
strings are valid country codes; use whatever OpenRouter uses (for these rows, ISO
3166-1 alpha-2, e.g. `SG`, `US`).

Geography comes from OpenRouter's frontend stats source (`provider_info`), which
carries `headquarters` and `datacenters`. A provider matches when its headquarters is
an excluded region **or** any of its datacenters is. Matching is per organization, not
per endpoint: if one endpoint of an org serves from an excluded region, every endpoint
of that org is dropped, so `alibaba/fp8` and `alibaba/fast` go together. Excluded
providers are removed before scoring, so they appear in neither the winner, the pareto
frontier, nor the safe set.

**This is a best-effort filter, not a compliance guarantee.** It acts on whatever
geography OpenRouter reports, which is third-party data that is often incomplete and
can be stale or wrong. Do not rely on it as your only control where a hard
data-residency guarantee is actually required.

Within that limit the filter is conservative. Approving a provider needs BOTH a known
headquarters and a non-empty datacenter list; a known headquarters says nothing about
where a request is actually served from, so partial metadata counts as unknown rather
than allowed. Rejection is asymmetric: one excluded location is enough to drop a
provider even when the rest of its metadata is missing.

Two under-specified cases each get a knob, both defaulting to the conservative reading:

**Providers with unknown geography** (`provider_info` absent, geography fields empty,
or the provider missing from telemetry entirely) are ineligible by default; absence of
evidence is not treated as evidence of eligibility. `allow_unknown_region: true` keeps
them and widens the candidate pool. Weigh this: on live OpenRouter data a large share
of rows report a headquarters with `datacenters` null or empty (for `z-ai/glm-5.2` at
time of writing, roughly 19 of 33 endpoints), so with the strict default a model can
legitimately end up with zero eligible providers and return no deployment. The strict
default is often a no-route default; the settings that keep a model broadly available
are the ones that filter less. Choose deliberately.

**Genuine cold start** (no cached telemetry yet, so no provider can be checked) is
governed by `unverified_region_policy`:

| Value | Behavior |
|---|---|
| `no_route` (default) | Return no deployment; the router surfaces a routing error. Nothing is sent that could land in an excluded region, but the model is unavailable until telemetry warms up. |
| `unpinned` | Send the request without `provider.only`. Keeps the model available, but OpenRouter may serve it from any provider, including an excluded region. Not a guarantee. |
| `trust_fallback` | Route only to `cold_start_fallback` slugs, treating that list as operator-vetted. Stays available and stays pinned; if no vetted slug is usable it declines to route rather than dropping the pin. Correctness depends on that list being accurate. |

The policy applies to both wildcard and pinned mode. Only `unpinned` ever removes
`provider.only`; `no_route` and `trust_fallback` never silently widen routing.
Declining to route logs a one-time warning to stderr naming the model.

While telemetry is merely stale (rather than absent), the filter keeps acting on the
last-known region verdicts rather than flipping the model to no-route during an outage;
that is the best-effort tradeoff, since old geography can be wrong.

## Use case: make routing authoritative (block client overrides)

By default the plugin rewrites only the deployment objects it hands back to the router;
it does not touch the caller's request. LiteLLM merges a request's `extra_body` over
the chosen deployment's, so a client that sends its own `provider` block wins: its
`provider.only`, `zdr`, and `allow_fallbacks` reach OpenRouter as sent. This is
intentional freedom, so a caller can override the routing decision for a one-off
request. Whether clients may send `extra_body` at all is your call at the proxy's
request-validation or key-permission layer, not this plugin's.

To make the plugin's choice authoritative for a model, set `strict_provider: true`:

```yaml
litellm_settings:
  openrouter_pareto_rules:
    "z-ai/glm-5.2":
      precision: ["fp8"]
      min_context: 1000000
      min_stats_requests: 100
      strict_provider: true
```

A request that carries its own `provider.only` for that managed model is then rejected
with a `StrictProviderConflict` (the router surfaces it as a failed request) instead of
being allowed to override the plugin. Both delivery shapes are caught: a top-level
`provider` field (how the proxy and the OpenAI SDK send OpenRouter's provider block)
and a nested `extra_body.provider` (a direct SDK call). A request that sends no
`provider.only` is routed normally.

## Configuration reference

Rules come from `litellm_settings.openrouter_pareto_rules` or the
`OPENROUTER_PARETO_RULES` JSON env var. Unknown fields and out-of-range values raise at
first use rather than silently falling back to defaults. Without any
`openrouter_pareto_rules`, the built-in default rule for `z-ai/glm-5.2` applies.

| Field | Default | Purpose |
|---|---|---|
| `precision` | `["fp8"]` | Allowed quantizations; a provider endpoint must match one. |
| `min_context` | `1000000` | Minimum advertised context length. |
| `min_stats_requests` | `100` | Minimum recent sample size before a provider is scored. |
| `promotion_polls` | `1` | Consecutive polls a new value-winner must win before it is promoted (flap damping). |
| `value_regression_tolerance` | `0.0` | Slack that lets a pricier, faster provider win over the cheaper incumbent on the frontier. At `0.0` a move happens only when the percentage throughput gain at least matches the percentage price increase; higher values tolerate paying disproportionately more for speed. |
| `wildcard` | `false` | `true` injects the winner per request on one deployment; `false` narrows among pinned deployments. |
| `log_errors` | `false` | `true` appends routed `>=400` errors to `OPENROUTER_PARETO_ERROR_LOG`. |
| `exclude_regions` | `[]` | Regions to exclude by provider geography. |
| `allow_unknown_region` | `false` | `true` keeps providers whose geography is unknown. |
| `unverified_region_policy` | `"no_route"` | Cold-start behavior under a region policy: `no_route`, `unpinned`, or `trust_fallback`. |
| `cold_start_fallback` | `[]` | Provider slugs trusted under `trust_fallback`. |
| `strict_provider` | `false` | `true` rejects a client-supplied `provider.only` on this model. |

`log_errors: true` writes to `OPENROUTER_PARETO_ERROR_LOG` (default a platformdirs user
log path: `errors.log` under the OS log directory), created owner-only (`0o600`, no
symlink following). Error bodies may contain prompt fragments; set an explicit writable
path in production and define retention.

## Telemetry SSL

The telemetry client that fetches OpenRouter provider stats verifies TLS by default.
Two global (not per-model) settings control verification, under
`litellm_settings.openrouter_pareto_telemetry` or the `OPENROUTER_PARETO_TELEMETRY`
JSON env var:

```yaml
litellm_settings:
  openrouter_pareto_telemetry:
    ssl_verify: false              # disable TLS verification (dev / self-signed proxies)
    # ssl_ca_cert: /etc/ssl/corp-ca.pem   # trust a custom CA bundle instead
```

| Field | Default | Purpose |
|---|---|---|
| `ssl_verify` | `true` | `false` disables TLS verification entirely. |
| `ssl_ca_cert` | unset | Path to a custom CA bundle; takes precedence over `ssl_verify` and requires it to stay `true`. |

These only affect the telemetry fetches to `openrouter.ai`; they do not change how
LiteLLM connects to OpenRouter for model traffic. A bad `ssl_ca_cert` path surfaces as
a telemetry degradation warning (stale or no winner) and the plugin falls back to the
healthy deployment set unchanged - it never raises into the request path.

## How selection works

- Telemetry is cached and refreshed on a 5-minute poll. A new value-winner must win
  `promotion_polls` consecutive polls before it is promoted. A winner that stops
  passing the hard filters is evicted immediately.
- On stale or empty telemetry, or no winner, the full healthy deployment set is
  returned unchanged. Absent a region policy the callback never narrows to an empty
  list from telemetry states; an all-input-capped state falls back to the winner (a
  preference skip, not a hard exclusion, since a small request may still succeed under
  the cap).
- If the winner is in cooldown or not healthy, it falls back to the next preferred
  provider in the safe set (cheapest-first), then to the full list, restricted (when a
  region policy is set) to region-eligible deployments.
- The telemetry cache is stored per `(model, rule)` fingerprint, so a worker configured
  with `exclude_regions` never inherits a winner computed by a worker without it, and
  two workers with different policies for the same model do not evict each other's
  cache. It is persisted off the request path and pruned by a retention window.
