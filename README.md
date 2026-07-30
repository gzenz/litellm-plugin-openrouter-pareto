# litellm-plugin-openrouter-pareto

A LiteLLM callback that picks the best-value OpenRouter provider per request for
a model served by multiple upstream providers. It applies hard filters
(quantization, min context, zero-data-retention, 5-minute uptime, min sample
size), builds a pareto frontier over input price per million tokens vs P50
throughput, then runs a greedy value-walk to one winner, with flap damping.

## Install

```
pip install litellm-plugin-openrouter-pareto
```

## Wire it into a LiteLLM proxy

Register the module-level callback instance in `litellm_settings.callbacks`
using a dotted path (the tail must be an instance, not the class):

```yaml
litellm_settings:
  callbacks: ["litellm_plugin_openrouter_pareto.plugin.openrouter_pareto_callback"]
model_list:
  - model_name: z-ai/glm-5.2
    litellm_params:
      model: openrouter/z-ai/glm-5.2
      extra_body:
        provider: {only: [baseten], zdr: true, allow_fallbacks: false, quantizations: [fp8]}
    model_info:
      id: or-baseten
  - model_name: z-ai/glm-5.2
    litellm_params:
      model: openrouter/z-ai/glm-5.2
      extra_body:
        provider: {only: [novita], zdr: true, allow_fallbacks: false, quantizations: [fp8]}
    model_info:
      id: or-novita
```

Each deployment is one OpenRouter provider pinned via `extra_body.provider.only`.
The callback matches a deployment to a winner provider slug by reading
`model_info.id` and stripping an `or-` prefix (so `or-baseten` becomes `baseten`),
falling back to `extra_body.provider.only[0]`.

The proxy must use an async-native routing strategy (the default `simple-shuffle`
works) so the callback deployment filter runs. Use the latest model in a family
unless you have a reason not to.

## Behavior

- On stale or empty telemetry, or no winner, the full healthy deployment set is
  returned unchanged. The callback never narrows to an empty list.
- If the winner is in cooldown / not healthy, it falls back to the next
  preferred provider in the safe set (cheapest-first), then to the full list.
- Telemetry is cached and refreshed on a 5-minute poll; a new value-winner must
  win `promotion_polls` consecutive polls before it is promoted. A winner that
  stops passing the hard filters is evicted immediately.

## 429 rate-limit fallback

A provider that returns HTTP 429 is treated as a transitory upstream rate limit.
Two layers cooperate:

- Per-request walk: LiteLLM puts the 429'ing deployment in cooldown immediately
  and its tenacity retry re-runs deployment selection, so the next attempt lands
  on a different provider. Bound this with `num_retries` on the model group.
- Cross-request cooldown: the callback tracks 429s per provider. A provider that
  429s three or more times within five minutes is "hot" and is skipped as the
  winner for new requests until its 429s age out of the window. This is the
  memory LiteLLM's short per-429 cooldown lacks; it stops a persistently
  rate-limited winner from being retried first on every new request. It is
  in-process and not persisted; it resets on restart.

The 429 is recorded in the `async_log_failure_event` hook (status from
`kwargs["exception"].status_code`); the "hot" skip is applied in
`async_filter_deployments`. When every provider is hot, the constrained winner is
kept rather than emitting unconstrained routing.
