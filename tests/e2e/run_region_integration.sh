#!/usr/bin/env bash
# exclude_regions integration test for litellm-plugin-openrouter-pareto.
#
# Proves the region filter end-to-end through a live litellm proxy: a managed model
# with exclude_regions set narrows the healthy deployment set to only the providers
# telemetry says are outside the excluded region, and never routes to an excluded one.
#
# Real OR proves the region LOGIC (see the notes in run_e2e.sh), but the live
# cold-start refresh writes the telemetry cache asynchronously after the first request
# returns, which races an inline assertion. So this test seeds the region verdicts
# deterministically - winner + allowed_bases + excluded_bases - exactly as a warmed
# cache would hold them, then observes routing. Both deployments are local always-200
# fixtures (one per provider slug) with per-fixture hit counters, so "the excluded
# deployment was never reached" is a hard, countable fact, not a timing guess.
#
# Scope: proves the routing/region-filter wiring only, with no OpenRouter cost.
#
# Prereqs: make bootstrap in litellm, pip install -e . (no OpenRouter key needed).
#
# Usage: bash tests/e2e/run_region_integration.sh
set -uo pipefail

# Ports drawn together so they cannot collide; the ownership check still guards a race.
read -r ALLOC_PROXY ALLOC_ALLOWED ALLOC_EXCLUDED <<EOF
$(python3 -c "
import socket
socks = [socket.socket() for _ in range(3)]
for s in socks:
    s.bind(('127.0.0.1', 0))
print(' '.join(str(s.getsockname()[1]) for s in socks))
for s in socks:
    s.close()")
EOF

PROXY_PORT="${PROXY_PORT:-$ALLOC_PROXY}"
ALLOWED_PORT="${ALLOWED_PORT:-$ALLOC_ALLOWED}"
EXCLUDED_PORT="${EXCLUDED_PORT:-$ALLOC_EXCLUDED}"
PROXY_URL="http://127.0.0.1:${PROXY_PORT}"
MASTER_KEY="sk-1234"
MODEL="z-ai/glm-5.2"
HERE="$(cd "$(dirname "$0")" && pwd)"
CONFIG="$HERE/or_region_integration_config.yaml"
PROXY_LOG="$HERE/proxy_region.log"
LITELLM_DIR="${LITELLM_DIR:-$HOME/PycharmProjects/litellm}"

# novita is the allowed provider (routed to), baseten is the excluded one (must never
# be reached). Slugs are arbitrary labels here; what matters is the seeded verdicts.
ALLOWED_SLUG="novita/fp8"
EXCLUDED_SLUG="baseten/fp8"

TEST_ROOT="$(mktemp -d)"
export XDG_CACHE_HOME="$TEST_ROOT/cache"

# The proxy must run under the interpreter the litellm checkout is installed in,
# because that is where the checkout (and this package, `pip install -e .` into that
# env) is importable. Bare `python3` resolves whatever litellm is in site-packages -
# a different release from the checkout, whose proxy/db may not even carry the
# modules proxy_cli.py imports - and the failure surfaces as a confusing
# ModuleNotFoundError at proxy startup. Prefer the checkout's own .venv, then
# $VIRTUAL_ENV, then whatever python3 is on PATH. The seeding step below imports
# this package too, so it uses the same interpreter.
LITELLM_PY="${LITELLM_PYTHON:-}"
if [ -z "$LITELLM_PY" ]; then
  if [ -x "$LITELLM_DIR/.venv/bin/python" ]; then
    LITELLM_PY="$LITELLM_DIR/.venv/bin/python"
  elif [ -n "${VIRTUAL_ENV:-}" ] && [ -x "$VIRTUAL_ENV/bin/python" ]; then
    LITELLM_PY="$VIRTUAL_ENV/bin/python"
  elif [ -x "$LITELLM_DIR/venv/bin/python" ]; then
    LITELLM_PY="$LITELLM_DIR/venv/bin/python"
  else
    LITELLM_PY="$(command -v python3)"
  fi
fi
[ -x "$LITELLM_PY" ] || { echo "no usable python interpreter for the litellm checkout: $LITELLM_PY" >&2; exit 1; }
if ! "$LITELLM_PY" -c "
import litellm, litellm_plugin_openrouter_pareto
" 2>/dev/null; then
  echo "interpreter $LITELLM_PY cannot import both litellm and litellm_plugin_openrouter_pareto;" >&2
  echo "install this package into that env (pip install -e .) or set LITELLM_PYTHON" >&2
  exit 1
fi
CACHE_DIR="$("$LITELLM_PY" -c "import platformdirs; print(platformdirs.user_cache_path('litellm-plugin-openrouter-pareto'))")"
CACHE_JSON="$CACHE_DIR/cache.json"

pass=0; fail=0
ok()   { echo "  PASS: $1"; pass=$((pass+1)); }
bad()  { echo "  FAIL: $1"; fail=$((fail+1)); }
section() { echo; echo "=== $1 ==="; }

cleanup() {
  [ -n "${PROXY_PID:-}" ] && kill "$PROXY_PID" 2>/dev/null
  [ -n "${ALLOWED_PID:-}" ] && kill "$ALLOWED_PID" 2>/dev/null
  [ -n "${EXCLUDED_PID:-}" ] && kill "$EXCLUDED_PID" 2>/dev/null
  [ -n "${TEST_ROOT:-}" ] && rm -rf "$TEST_ROOT"
}
trap cleanup EXIT

require() { command -v "$1" >/dev/null 2>&1 || { echo "missing: $1"; exit 1; }; }
require curl
require python3

# The rule the proxy runs must match the fingerprint the cache is seeded under, or the
# entry is invisible to this process. Seed winner=novita/fp8 with baseten's ORG base
# excluded and novita's allowed, exactly as a warmed region cache would hold it.
mkdir -p "$CACHE_DIR"
rm -f "$CACHE_JSON"
"$LITELLM_PY" - "$CACHE_JSON" "$ALLOWED_SLUG" "$EXCLUDED_SLUG" <<'PY'
import json, sys, time, pathlib
from litellm_plugin_openrouter_pareto.config import rule, rule_fingerprint
from litellm_plugin_openrouter_pareto.telemetry import OR_CACHE_VERSION

cache_path, allowed_slug, excluded_slug = sys.argv[1], sys.argv[2], sys.argv[3]
allowed_base = allowed_slug.split("/", 1)[0]
excluded_base = excluded_slug.split("/", 1)[0]

# Must equal the proxy's rule (exclude_regions=["US"]) so the fingerprint matches.
r = rule(
    precision="fp8",
    min_context=1_000_000,
    min_stats_requests=100,
    promotion_polls=2,
    exclude_regions=["US"],
)
fp = rule_fingerprint(r)
now = time.time()
entry = {
    "version": OR_CACHE_VERSION,
    "entries": {
        f"z-ai/glm-5.2\x00{fp}": {
            "winner": allowed_slug,
            "candidate_winner": allowed_slug,
            "candidate_streak": 2,
            "safe_set": [allowed_slug],
            "canonical_slug": "z-ai/glm-5.2",
            "canonical_slug_fetched_at": now,
            "fetched_at": now,
            "stale": False,
            "rule_fp": fp,
            "excluded_bases": [excluded_base],
            "allowed_bases": [allowed_base],
        }
    },
}
pathlib.Path(cache_path).write_text(json.dumps(entry))
print(f"  seeded rule_fp={fp}: winner={allowed_slug}, allowed=[{allowed_base}], excluded=[{excluded_base}]", file=sys.stderr)
PY
echo "  pre-populated region cache (winner=$ALLOWED_SLUG, excluded org=${EXCLUDED_SLUG%%/*})"

# Both deployments point at their own always-200 fixture so a hit is countable per slug.
# The region filter should drop the excluded deployment BEFORE the router picks, so the
# excluded fixture must record zero hits.
cat > "$CONFIG" <<EOF
model_list:
  - model_name: z-ai/glm-5.2
    litellm_params:
      model: openrouter/z-ai/glm-5.2
      api_base: http://127.0.0.1:${ALLOWED_PORT}/v1
      api_key: sk-dummy
      extra_body:
        provider: {only: [${ALLOWED_SLUG}], zdr: true, allow_fallbacks: false, quantizations: [fp8]}
    model_info:
      id: or-${ALLOWED_SLUG}
  - model_name: z-ai/glm-5.2
    litellm_params:
      model: openrouter/z-ai/glm-5.2
      api_base: http://127.0.0.1:${EXCLUDED_PORT}/v1
      api_key: sk-dummy
      extra_body:
        provider: {only: [${EXCLUDED_SLUG}], zdr: true, allow_fallbacks: false, quantizations: [fp8]}
    model_info:
      id: or-${EXCLUDED_SLUG}

litellm_settings:
  callbacks: ["litellm_plugin_openrouter_pareto.plugin.openrouter_pareto_callback"]
  num_retries: 0
  disable_cooldowns: true
  routing_strategy: simple-shuffle
  openrouter_pareto_rules:
    "z-ai/glm-5.2":
      precision: ["fp8"]
      min_context: 1000000
      min_stats_requests: 100
      promotion_polls: 2
      exclude_regions: ["US"]

general_settings:
  master_key: sk-1234
EOF

echo "region integration test for litellm-plugin-openrouter-pareto"
echo "proxy:    $PROXY_URL   allowed fixture: 127.0.0.1:$ALLOWED_PORT   excluded fixture: 127.0.0.1:$EXCLUDED_PORT"

section "1. start the allowed and excluded provider fixtures"
python3 "$HERE/healthy_fixture.py" "$ALLOWED_PORT" &
ALLOWED_PID=$!
python3 "$HERE/healthy_fixture.py" "$EXCLUDED_PORT" &
EXCLUDED_PID=$!
sleep 1
for pid in "$ALLOWED_PID" "$EXCLUDED_PID"; do
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "a fixture process exited during startup (pid $pid); a port may be occupied" >&2
    exit 1
  fi
done
for port in "$ALLOWED_PORT" "$EXCLUDED_PORT"; do
  code=$(curl -s -o /dev/null -w "%{http_code}" -X POST "http://127.0.0.1:$port/v1/chat/completions")
  [ "$code" = "200" ] || { echo "fixture on $port returned $code, expected 200" >&2; exit 1; }
done
echo "  both fixtures return 200"

section "2. start the proxy"
rm -f "$PROXY_LOG"
if curl -fs -m 2 "$PROXY_URL/health/liveliness" >/dev/null 2>&1; then
  echo "port $PROXY_PORT is already serving; refusing to run against a proxy we did not start" >&2
  exit 1
fi
( cd "$LITELLM_DIR" && "$LITELLM_PY" litellm/proxy/proxy_cli.py \
    --config "$CONFIG" --detailed_debug --port "$PROXY_PORT" ) >"$PROXY_LOG" 2>&1 &
PROXY_PID=$!
for _ in $(seq 1 60); do curl -fs "$PROXY_URL/health/liveliness" >/dev/null 2>&1 && break; sleep 1; done
if ! kill -0 "$PROXY_PID" 2>/dev/null; then
  echo "the proxy we started exited (pid $PROXY_PID); port $PROXY_PORT may be occupied" >&2
  tail -30 "$PROXY_LOG"
  exit 1
fi
LISTENER_PIDS="$(python3 - "$PROXY_PORT" <<'PY'
import subprocess, sys
port = sys.argv[1]
try:
    import psutil
except ImportError:
    psutil = None
if psutil is not None:
    pids = {
        c.pid
        for c in psutil.net_connections(kind="tcp")
        if c.status == psutil.CONN_LISTEN and c.laddr and c.laddr.port == int(port) and c.pid
    }
    print(" ".join(str(p) for p in sorted(pids)))
else:
    out = subprocess.run(
        ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
        capture_output=True, text=True,
    ).stdout
    print(" ".join(out.split()))
PY
)"
OWNED=0
for lp in $LISTENER_PIDS; do
  probe="$lp"
  for _ in 1 2 3 4 5; do
    [ "$probe" = "$PROXY_PID" ] && { OWNED=1; break; }
    probe="$(ps -o ppid= -p "$probe" 2>/dev/null | tr -d ' ')"
    [ -z "$probe" ] || [ "$probe" = "1" ] || [ "$probe" = "0" ] && break
  done
  [ "$OWNED" = "1" ] && break
done
if [ "$OWNED" != "1" ]; then
  echo "port $PROXY_PORT is served by pid(s) [$LISTENER_PIDS], not by our proxy ($PROXY_PID)" >&2
  tail -30 "$PROXY_LOG"
  exit 1
fi
echo "proxy live (pid $PROXY_PID owns port $PROXY_PORT)"

chat() {
  local body
  body=$(python3 -c "import json; print(json.dumps({'model':'$MODEL','messages':[{'role':'user','content':'hi'}],'max_tokens':4}))")
  curl -sS -X POST "$PROXY_URL/v1/chat/completions" \
    -H "Authorization: Bearer $MASTER_KEY" -H "Content-Type: application/json" -d "$body"
}
fixture_count() {
  curl -s "http://127.0.0.1:$1/count" | python3 -c "import json,sys; print(json.load(sys.stdin)['count'])"
}

section "3. drive requests; every one must land on the allowed provider, never the excluded"
allowed_before=$(fixture_count "$ALLOWED_PORT")
excluded_before=$(fixture_count "$EXCLUDED_PORT")
for i in 1 2 3; do
  out=$(chat)
  choices=$(echo "$out" | python3 -c "import json,sys; d=json.load(sys.stdin); print(len(d.get('choices',[])))" 2>/dev/null || echo "0")
  echo "  request $i -> choices: $choices"
done
allowed_delta=$(( $(fixture_count "$ALLOWED_PORT") - allowed_before ))
excluded_delta=$(( $(fixture_count "$EXCLUDED_PORT") - excluded_before ))
echo "  allowed fixture ($ALLOWED_SLUG) hits: $allowed_delta | excluded fixture ($EXCLUDED_SLUG) hits: $excluded_delta"

if [ "$excluded_delta" -ne 0 ]; then
  bad "the excluded provider was reached $excluded_delta time(s); the region filter did not drop it"
elif [ "$allowed_delta" -ne 3 ]; then
  bad "expected all 3 requests on the allowed provider, saw $allowed_delta"
else
  ok "region filter held: all 3 requests routed to the allowed provider, the excluded one was never reached"
fi

section "summary"
echo "  PASS=$pass FAIL=$fail"
echo "  proxy log: $PROXY_LOG"
[ "$fail" -eq 0 ] && { echo "ALL PASS"; exit 0; } || { echo "SOME FAILURES"; exit 1; }
