"""Generate the OpenRouter pareto e2e proxy config (real providers only).

Writes or_pareto_config.yaml with the model group z-ai/glm-5.2 across multiple
real OpenRouter provider deployments, each pinned via extra_body.provider.only
and model_info.id=or-<slug>, the package callback registered, and num_retries=3.

No mocks or local fixtures: the e2e proves only what real OpenRouter can
deterministically show (winner selection, stale-telemetry passthrough). The 429
cooldown is unit-tested in test_cooldown.py / test_plugin.py, not here.

Usage: python3 tests/e2e/gen_config.py [--out PATH] [--providers a,b,c]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DEFAULT_PROVIDERS = [
    "novita/fp8",
    "siliconflow/fp8",
    "baseten/fp8",
    "crusoe/fp8",
    "sail-research/fp8",
    "z-ai/fp8",
    "venice/fp8",
]
OR_MODEL = "openrouter/z-ai/glm-5.2"
CALLBACK = "litellm_plugin_openrouter_pareto.plugin.openrouter_pareto_callback"


def _deployment(slug: str) -> str:
    return (
        "  - model_name: z-ai/glm-5.2\n"
        "    litellm_params:\n"
        "      model: " + OR_MODEL + "\n"
        "      api_key: os.environ/OPENROUTER_API_KEY\n"
        "      extra_body:\n"
        "        provider: {only: [" + slug + "], zdr: true, allow_fallbacks: false, quantizations: [fp8]}\n"
        "    model_info:\n"
        "      id: or-" + slug
    )


def _rules_block(exclude_regions: list[str]) -> str:
    if not exclude_regions:
        return ""
    regions = ", ".join(f'"{r}"' for r in exclude_regions)
    return (
        "  openrouter_pareto_rules:\n"
        + '    "z-ai/glm-5.2":\n'
        + "      precision: [\"fp8\"]\n"
        + "      min_context: 1000000\n"
        + "      min_stats_requests: 100\n"
        + "      exclude_regions: [" + regions + "]\n"
        + "      allow_unknown_region: false\n"
        + "      unverified_region_policy: \"no_route\"\n"
    )


def render(providers: list[str], exclude_regions: list[str] | None = None) -> str:
    deployments = "\n".join(_deployment(s) for s in providers)
    return (
        "model_list:\n"
        + deployments
        + "\n\n"
        + "litellm_settings:\n"
        + "  callbacks: [\"" + CALLBACK + "\"]\n"
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
