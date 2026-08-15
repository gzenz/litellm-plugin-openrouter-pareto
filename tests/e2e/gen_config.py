"""Generate the OpenRouter pareto e2e proxy config (real providers only).

Writes or_pareto_config.yaml with the model group deepseek/deepseek-v4-flash-0731
across multiple real OpenRouter provider deployments, each pinned via
extra_body.provider.only and model_info.id=or-<slug>, the package callback
registered, and num_retries=3.

No mocks or local fixtures: the e2e proves only what real OpenRouter can
deterministically show (winner selection, stale-telemetry passthrough). The 429
cooldown is unit-tested in test_cooldown.py / test_plugin.py, not here.

The rule is emitted explicitly (deepseek/deepseek-v4-flash-0731 is not in the
package DEFAULT_RULES, which only ships z-ai/glm-5.2), so the config is
self-contained and does not depend on the shipped defaults.

Usage: python3 tests/e2e/gen_config.py [--out PATH] [--providers a,b,c]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

MODEL_NAME = "deepseek/deepseek-v4-flash-0731"
OR_MODEL = "openrouter/deepseek/deepseek-v4-flash-0731"
CALLBACK = "litellm_plugin_openrouter_pareto.plugin.openrouter_pareto_callback"
# fp8 + >=1M context providers on OpenRouter with enough stats to score.
# deepinfra/fp8 is in the account's allowed-providers list; the rest are not,
# so the allowlist filter (discovered from the first 404) narrows the winner
# to deepinfra/fp8. baseten/fp8 is cheapest + fastest and would be the value-walk
# winner without the allowlist; the rest form the safe set.
DEFAULT_PROVIDERS = [
    "deepinfra/fp8",
    "novita/fp8",
    "siliconflow/fp8",
    "baseten/fp8",
    "parasail/fp8",
    "gmicloud/fp8",
    "mancer/fp8",
]
COLD_START_FALLBACK = "deepinfra/fp8"


def _deployment(slug: str) -> str:
    return (
        f"  - model_name: {MODEL_NAME}\n"
        "    litellm_params:\n"
        f"      model: {OR_MODEL}\n"
        "      api_key: os.environ/OPENROUTER_API_KEY\n"
        "      extra_body:\n"
        f"        provider: {{only: [{slug}], zdr: true, allow_fallbacks: false, quantizations: [fp8]}}\n"
        "    model_info:\n"
        f"      id: or-{slug}"
    )


def _rules_block(exclude_regions: list[str]) -> str:
    lines = [
        "  openrouter_pareto_rules:",
        f'    "{MODEL_NAME}":',
        '      precision: ["fp8"]',
        "      min_context: 1000000",
        "      min_stats_requests: 100",
        f'      cold_start_fallback: ["{COLD_START_FALLBACK}"]',
    ]
    if exclude_regions:
        regions = ", ".join(f'"{r}"' for r in exclude_regions)
        lines += [
            f"      exclude_regions: [{regions}]",
            "      allow_unknown_region: false",
            '      unverified_region_policy: "no_route"',
        ]
    return "\n".join(lines) + "\n"


def render(providers: list[str], exclude_regions: list[str] | None = None) -> str:
    deployments = "\n".join(_deployment(s) for s in providers)
    return (
        "model_list:\n"
        + deployments
        + "\n\n"
        + "litellm_settings:\n"
        + f'  callbacks: ["{CALLBACK}"]\n'
        + "  num_retries: 3\n"
        + "  routing_strategy: simple-shuffle\n"
        + _rules_block(exclude_regions or [])
        + "\n"
        + "general_settings:\n"
        + "  master_key: sk-1234\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="tests/e2e/or_pareto_config.yaml")
    parser.add_argument("--providers", default=",".join(DEFAULT_PROVIDERS))
    parser.add_argument("--exclude-regions", default="")
    args = parser.parse_args()
    providers = [p.strip() for p in args.providers.split(",") if p.strip()]
    regions = [r.strip() for r in args.exclude_regions.split(",") if r.strip()]
    path = Path(args.out)
    path.write_text(render(providers, regions))
    print(f"wrote {path}", file=sys.stderr)
    print(path.read_text())


if __name__ == "__main__":
    main()
