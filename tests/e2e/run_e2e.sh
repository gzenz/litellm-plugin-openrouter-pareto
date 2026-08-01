#!/usr/bin/env bash
# OpenRouter pareto plugin end-to-end proof-of-fix (real OpenRouter only).
#
# Drives the litellm-plugin-openrouter-pareto callback the way production does:
# a live litellm proxy on localhost:4000 with the callback registered, multiple
# real OpenRouter deployments of z-ai/glm-5.2, real API calls that cost real
# money. Proves: (1) winner selection matches the value-walk, (2) a warm cache
# keeps serving. The exclude_regions filter is proven separately with seeded
# telemetry (run_region_integration.sh) because its live cold-start refresh writes
# the cache asynchronously and races an inline assertion.
#
# The 429 cooldown path CANNOT be tested against real OR (providers don't 429
# on demand); it is covered by the local-429 integration test
# (tests/e2e/run_429_integration.sh) and by unit tests (test_cooldown.py,
# test_plugin.py).
#
# Per repo CLAUDE.md the proof-of-fix is curl commands + proxy run logs, not
# pytest output. This script prints the curl commands it runs and the grepped
# log lines that prove behavior.
#
# Prereqs (run once):
#   In the litellm repo: `make bootstrap` (provision deps incl. `expression`).
#   pip install -e .                       # this package, into that litellm env
#   export OPENROUTER_API_KEY=...          # (or OPENROUTER_KEY) from your env
#
# Usage: bash tests/e2e/run_e2e.sh
set -uo pipefail

PROXY_PORT=4000
PROXY_URL="http://localhost:${PROXY_PORT}"
MASTER_KEY="sk-1234"
MODEL="z-ai/glm-5.2"
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
CONFIG="$HERE/or_pareto_config.yaml"
PROXY_LOG="$HERE/proxy.log"
CACHE_JSON="$(python3 -c "import platformdirs; print(platformdirs.user_cache_path('litellm-plugin-openrouter-pareto') / 'cache.json')")"
LITELLM_DIR="${LITELLM_DIR:-$HOME/PycharmProjects/litellm}"

pass=0; fail=0
ok()   { echo "  PASS: $1"; pass=$((pass+1)); }
bad()  { echo "  FAIL: $1"; fail=$((fail+1)); }
section() { echo; echo "=== $1 ==="; }

cleanup() { [ -n "${PROXY_PID:-}" ] && kill "$PROXY_PID" 2>/dev/null; }
trap cleanup EXIT

require() { command -v "$1" >/dev/null 2>&1 || { echo "missing: $1"; exit 1; }; }
require curl
require python3

if [ -z "${OPENROUTER_API_KEY:-}" ] && [ -z "${OPENROUTER_KEY:-}" ]; then
  echo "OPENROUTER_API_KEY (or OPENROUTER_KEY) is not set; export it from your env" >&2
  exit 1
fi
export OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-$OPENROUTER_KEY}"

echo "e2e for litellm-plugin-openrouter-pareto (real OpenRouter)"
echo "proxy:    $PROXY_URL  (master key $MASTER_KEY)"
echo "cache:     $CACHE_JSON"
echo "litellm:   $LITELLM_DIR"

section "1. generate config"
python3 "$HERE/gen_config.py" --out "$CONFIG" >/dev/null
echo "config: $CONFIG"

section "2. start the proxy with --detailed_debug"
rm -f "$CACHE_JSON" "$PROXY_LOG"
( cd "$LITELLM_DIR" && python3 litellm/proxy/proxy_cli.py \
    --config "$CONFIG" --detailed_debug --port "$PROXY_PORT" ) >"$PROXY_LOG" 2>&1 &
PROXY_PID=$!
echo "proxy PID $PROXY_PID; waiting for liveness..."
for _ in $(seq 1 60); do
  curl -fs "$PROXY_URL/health/liveliness" >/dev/null 2>&1 && break
  sleep 1
done
curl -fs "$PROXY_URL/health/liveliness" >/dev/null 2>&1 \
  && echo "proxy live" || { echo "proxy did not become live"; tail -40 "$PROXY_LOG"; exit 1; }
echo "  callback loaded:"
grep -iE "Initialized Callbacks" "$PROXY_LOG" | tail -1 | grep -oE "OpenRouterParetoCallback object at 0x[0-9a-f]+" || echo "  (not found)"

chat() {
  local body
  body=$(python3 -c "import json,sys; print(json.dumps({'model':'$MODEL','messages':[{'role':'user','content':sys.argv[1]}],'max_tokens':256}))" "$1")
  echo "  curl -sS -X POST $PROXY_URL/v1/chat/completions -H 'Authorization: Bearer $MASTER_KEY' -d '<body>'" >&2
  curl -sS -X POST "$PROXY_URL/v1/chat/completions" \
    -H "Authorization: Bearer $MASTER_KEY" -H "Content-Type: application/json" \
    -d "$body"
}

# A response is "served" if it parses with >=1 choice. glm-5.2 is a reasoning
# model: with low max_tokens the assistant content can be null (tokens consumed
# by reasoning_content), so we do NOT assert on content; the served deployment
# slug (from the proxy log) is the real signal.
served_ok() {
  python3 -c "import json,sys; d=json.load(sys.stdin); sys.exit(0 if d.get('choices') else 1)" 2>/dev/null
}

section "3. case 1 - winner selection (real OR)"
echo "  warming telemetry cache (cold start; the callback blocks the first"
echo "  managed request up to ~16s for a fresh fetch, then narrows to the winner)"
out=$(chat "warmup")
echo "$out" | python3 -c "import json,sys; d=json.load(sys.stdin); print('  warmup -> provider:', d.get('provider'), '| choices:', len(d.get('choices',[])))" 2>/dev/null || echo "  warmup -> (unparseable)"
echo "  waiting for cache.json to be written by the telemetry refresher..."
for _ in $(seq 1 30); do
  [ -f "$CACHE_JSON" ] && break
  sleep 1
done
if [ ! -f "$CACHE_JSON" ]; then
  echo "  cache.json not written after 30s; the telemetry fetch may have failed or timed out"
fi
echo "  cache.json winner:"
python3 -c "
import json
d = json.load(open('$CACHE_JSON'))
e = next(v for k, v in d['entries'].items() if k.split(chr(0))[0] == '$MODEL')
print('   winner:', e['winner'], '| safe_set:', e['safe_set'], '| stale:', e['stale'])" 2>/dev/null \
  || echo "   (cache unreadable)"
WINNER_SLUG=$(python3 -c "
import json
d = json.load(open('$CACHE_JSON'))
e = next(v for k, v in d['entries'].items() if k.split(chr(0))[0] == '$MODEL')
print(e['winner'])" 2>/dev/null || echo "")
echo "  value-walk winner slug: ${WINNER_SLUG:-(none)}"

echo "  sending 3 requests; EACH must succeed and be narrowed to the winner deployment"
# The winner org and safe-set orgs are needed per request, so resolve them first.
# OR's `provider` field is the org name only ("BaseTen") while the winner slug is the
# full endpoint id ("baseten/fp8"), so comparison is against the org part.
WINNER_ORG="${WINNER_SLUG%%/*}"
SAFE_ORGS=$(python3 -c "
import json, sys, pathlib
p = pathlib.Path(sys.argv[1])
d = json.loads(p.read_text())
e = next((v for k, v in d['entries'].items() if k.split(chr(0))[0] == 'z-ai/glm-5.2'), {})
print(' '.join(sorted({s.split('/')[0] for s in e.get('safe_set', [])})))
" "$CACHE_JSON" 2>/dev/null || echo "")

if [ -z "$WINNER_SLUG" ]; then
  bad "no winner in cache.json; telemetry fetch did not complete"
else
  # Each request is judged on its own: it must return a completion, name a provider,
  # and that provider must be the winner org - or a safe_set org when the winner
  # actually 429'd in the log written by THIS request. Folding all three into one
  # grep would let two broken responses ride along on one good one.
  REQ_FAILURES=0
  SERVED_PROVIDERS=""
  for i in 1 2 3; do
    echo "  --- request $i ---"
    LOG_OFFSET=$(wc -c < "$PROXY_LOG" | tr -d ' ')
    out=$(chat "say hi briefly")
    prov=$(echo "$out" | python3 -c "import json,sys; d=json.load(sys.stdin); print((d.get('provider') or '').lower())" 2>/dev/null || echo "")
    choices=$(echo "$out" | python3 -c "import json,sys; d=json.load(sys.stdin); print(len(d.get('choices',[])))" 2>/dev/null || echo "0")
    echo "  -> provider: ${prov:-(none)} | choices: $choices"
    SERVED_PROVIDERS="$SERVED_PROVIDERS ${prov:-<none>}"
    # 429 evidence scoped to just this request's log slice, and it must name the winner.
    req_429=$(tail -c "+$((LOG_OFFSET + 1))" "$PROXY_LOG" 2>/dev/null \
      | grep -i "429" | grep -ci "$WINNER_ORG" || true)
    if [ "$choices" = "0" ] || [ "$choices" = "?" ]; then
      bad "request $i returned no completion: $(echo "$out" | head -c 200)"
      REQ_FAILURES=$((REQ_FAILURES + 1))
    elif [ -z "$prov" ]; then
      bad "request $i named no provider, so the served org cannot be verified"
      REQ_FAILURES=$((REQ_FAILURES + 1))
    elif echo "$prov" | grep -qi "^${WINNER_ORG}$"; then
      echo "     request $i served by the winner org ($WINNER_ORG)"
    elif [ "${req_429:-0}" -gt 0 ] && [ -n "$SAFE_ORGS" ] && \
         echo "$SAFE_ORGS" | tr ' ' '\n' | grep -qix "$prov"; then
      echo "     request $i: winner 429'd (${req_429} lines); correctly walked to safe_set org $prov"
    else
      bad "request $i served by '$prov', which is neither the winner org ($WINNER_ORG) nor a safe_set org ($SAFE_ORGS) after a winner 429"
      REQ_FAILURES=$((REQ_FAILURES + 1))
    fi
  done
  echo "  served providers across 3 requests:$SERVED_PROVIDERS"
  if [ "$REQ_FAILURES" = "0" ]; then
    ok "all 3 requests were narrowed to the value-walk winner org ($WINNER_ORG from $WINNER_SLUG) or a legitimate 429 fallback"
  fi
fi

section "4. case 2 - repeat request after cache edit (NOT a stale-telemetry test)"
# Deliberately NOT asserted as stale-fallback coverage. Telemetry loads cache.json once
# per process (`_disk_loaded`), so editing the file under a running proxy does not reach
# the in-memory entry and no refresh is forced. Even with a restart, real OR telemetry
# would simply succeed, so this path still would not exercise stale fallback.
#
# Genuine stale-fallback coverage needs an aged cache seeded BEFORE startup plus a
# telemetry endpoint that deterministically fails, which is fixture territory: see
# tests/test_telemetry.py (unit) and run_429_integration.sh (local fixtures).
python3 - "$CACHE_JSON" "$MODEL" <<'PY' 2>/dev/null || true
import json, pathlib, sys
p = pathlib.Path(sys.argv[1]); model = sys.argv[2]
d = json.loads(p.read_text())
key = next(k for k in d["entries"] if k.split(chr(0))[0] == model)
d["entries"][key]["fetched_at"] = 0.0
p.write_text(json.dumps(d))
print("  aged fetched_at -> 0 on disk (in-memory entry is unaffected)")
PY
out=$(chat "say ok")
echo "$out" | python3 -c "import json,sys; d=json.load(sys.stdin); print('  -> provider:', d.get('provider'), '| choices:', len(d.get('choices',[])))" 2>/dev/null || echo "  -> (unparseable)"
if echo "$out" | served_ok; then
  ok "a second request on the same warm cache still routes to a healthy provider"
else
  bad "second request on the warm cache narrowed to zero or errored"
fi

section "summary"
echo "  PASS=$pass FAIL=$fail"
echo "  proxy log: $PROXY_LOG"
echo "  cache:     $CACHE_JSON"
echo "  429 wiring:   see tests/e2e/run_429_integration.sh (local fixture)"
echo "  region filter: see tests/e2e/run_region_integration.sh (local fixture; the"
echo "                 live cold-start refresh writes the cache asynchronously, which"
echo "                 races an inline live assertion, so region routing is proven"
echo "                 deterministically with seeded telemetry instead)"
[ "$fail" -eq 0 ] && { echo "ALL PASS"; exit 0; } || { echo "SOME FAILURES"; exit 1; }
