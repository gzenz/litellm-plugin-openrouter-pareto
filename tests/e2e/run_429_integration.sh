#!/usr/bin/env bash
# 429 cooldown integration test for litellm-plugin-openrouter-pareto.
#
# Proves the litellm -> async_log_failure_event -> RateLimitCooldown -> reroute
# wiring end-to-end: a managed model group's value-walk winner is the local
# always-429 fixture; 429s fire; the cooldown crosses the threshold; the next
# request to the SAME managed model group routes to the healthy alternative.
#
# Real OR can't 429 on demand, so this pre-populates the package's telemetry
# cache with a fake winner (the fixture's slug) so the callback deterministically
# narrows to the fixture. Both deployments are local fixtures (always-429 and
# always-200), clearly-labeled test servers, not real LLMs.
#
# Scope: this proves the local routing/cooldown wiring only. It does NOT test real
# OpenRouter behavior; run_e2e.sh carries the real-provider proof.
#
# Prereqs: make bootstrap in litellm, pip install -e . (no OpenRouter key needed).
#
# Usage: bash tests/e2e/run_429_integration.sh
set -uo pipefail

# Ports are allocated per run rather than hardcoded. A fixed port makes the suite
# depend on ambient machine state: if anything else already listens there, the
# readiness probe is answered by that unrelated process and the run then measures the
# wrong proxy. All three are drawn in ONE process so they cannot collide with each
# other; the ownership check below still catches a race with an outside process.
read -r ALLOC_PROXY ALLOC_FOUR29 ALLOC_HEALTHY <<EOF
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
FOUR29_PORT="${FOUR29_PORT:-$ALLOC_FOUR29}"
HEALTHY_PORT="${HEALTHY_PORT:-$ALLOC_HEALTHY}"
PROXY_URL="http://127.0.0.1:${PROXY_PORT}"
MASTER_KEY="sk-1234"
MODEL="z-ai/glm-5.2"
HERE="$(cd "$(dirname "$0")" && pwd)"
CONFIG="$HERE/or_429_integration_config.yaml"
PROXY_LOG="$HERE/proxy_429.log"
LITELLM_DIR="${LITELLM_DIR:-$HOME/PycharmProjects/litellm}"

TEST_ROOT="$(mktemp -d)"
export XDG_CACHE_HOME="$TEST_ROOT/cache"

# The proxy must run under the interpreter the litellm checkout is installed in,
# because that is where the checkout (and this package, `pip install -e .` into that
# env) is importable. Bare `python3` resolves whatever litellm is in site-packages -
# a different release from the checkout, whose proxy/db may not even carry the
# modules proxy_cli.py imports - and the failure surfaces as a confusing
# ModuleNotFoundError at proxy startup. Prefer the checkout's own .venv, then
# $VIRTUAL_ENV, then whatever python3 is on PATH. The cache-seeding step below
# imports this package too, so it uses the same interpreter.
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
  [ -n "${FOUR29_PID:-}" ] && kill "$FOUR29_PID" 2>/dev/null
  [ -n "${HEALTHY_PID:-}" ] && kill "$HEALTHY_PID" 2>/dev/null
  [ -n "${TEST_ROOT:-}" ] && rm -rf "$TEST_ROOT"
}
trap cleanup EXIT

require() { command -v "$1" >/dev/null 2>&1 || { echo "missing: $1"; exit 1; }; }
require curl
require python3

# This test uses only local fixtures (429 + healthy); no real OpenRouter call is
# required. It proves the litellm -> async_log_failure_event -> RateLimitCooldown
# -> reroute wiring, not real OR behavior.

# Pre-populate the telemetry cache with a fake winner (novita/fp8 = the fixture
# slug) so the callback narrows to the fixture deployment. Winner is fresh
# (not stale), safe_set has novita/fp8 and baseten/fp8.
mkdir -p "$CACHE_DIR"
rm -f "$CACHE_JSON"
"$LITELLM_PY" - "$CACHE_JSON" <<'PY'
import json, sys, time, pathlib
from litellm_plugin_openrouter_pareto.config import DEFAULT_RULES, rule_fingerprint
from litellm_plugin_openrouter_pareto.telemetry import OR_CACHE_VERSION

# Entries are stored under a composite "<model>\x00<rule_fp>" key so workers with
# different policies can share one cache file, so seed that exact identity for the
# rule the proxy will actually use.
cache_path = pathlib.Path(sys.argv[1])
fp = rule_fingerprint(DEFAULT_RULES["z-ai/glm-5.2"])
entry = {
    "version": OR_CACHE_VERSION,
    "entries": {
        f"z-ai/glm-5.2\x00{fp}": {
            "winner": "novita/fp8",
            "candidate_winner": "novita/fp8",
            "candidate_streak": 2,
            "safe_set": ["novita/fp8", "baseten/fp8"],
            "canonical_slug": "z-ai/glm-5.2",
            "canonical_slug_fetched_at": time.time(),
            "fetched_at": time.time(),
            "stale": False,
            "rule_fp": fp,
            "excluded_bases": [],
            "allowed_bases": [],
        }
    },
}
cache_path.parent.mkdir(parents=True, exist_ok=True)
cache_path.write_text(json.dumps(entry))
print(f"  seeded rule_fp={fp}", file=sys.stderr)
PY
echo "  pre-populated cache: winner=novita/fp8 (fixture), safe_set=[novita/fp8, baseten/fp8]"

# Config: fixture deployment pinned to novita/fp8 (the winner), healthy sibling
# pinned to baseten/fp8 (the local healthy fixture). The callback narrows to
# novita/fp8 (the value-walk winner from the cache), the 429 fixture returns 429,
# the cooldown records it, and the next request walks to the healthy fixture.
cat > "$CONFIG" <<EOF
model_list:
  - model_name: z-ai/glm-5.2
    litellm_params:
      model: openrouter/z-ai/glm-5.2
      api_base: http://127.0.0.1:${FOUR29_PORT}/v1
      api_key: sk-dummy
      extra_body:
        provider: {only: [novita/fp8], zdr: true, allow_fallbacks: false, quantizations: [fp8]}
    model_info:
      id: or-novita/fp8
  - model_name: z-ai/glm-5.2
    litellm_params:
      model: openrouter/z-ai/glm-5.2
      api_base: http://127.0.0.1:${HEALTHY_PORT}/v1
      api_key: sk-dummy
      extra_body:
        provider: {only: [baseten/fp8], zdr: true, allow_fallbacks: false, quantizations: [fp8]}
    model_info:
      id: or-baseten/fp8

litellm_settings:
  callbacks: ["litellm_plugin_openrouter_pareto.plugin.openrouter_pareto_callback"]
  num_retries: 0
  disable_cooldowns: true
  routing_strategy: simple-shuffle

general_settings:
  master_key: sk-1234
EOF

echo "429 integration test for litellm-plugin-openrouter-pareto"
echo "proxy:    $PROXY_URL   429 fixture: 127.0.0.1:$FOUR29_PORT   healthy fixture: 127.0.0.1:$HEALTHY_PORT"

section "1. start the 429 and healthy fixtures"
python3 "$HERE/four29.py" "$FOUR29_PORT" &
FOUR29_PID=$!
python3 "$HERE/healthy_fixture.py" "$HEALTHY_PORT" &
HEALTHY_PID=$!
sleep 1
for pid in "$FOUR29_PID" "$HEALTHY_PID"; do
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "a fixture process exited during startup (pid $pid); a port may be occupied" >&2
    exit 1
  fi
done
code=$(curl -s -o /dev/null -w "%{http_code}" -X POST "http://127.0.0.1:$FOUR29_PORT/v1/chat/completions")
if [ "$code" = "429" ]; then
  echo "  429 fixture returns 429"
else
  echo "429 fixture returned $code, expected 429"
  exit 1
fi
hcode=$(curl -s -o /dev/null -w "%{http_code}" -X POST "http://127.0.0.1:$HEALTHY_PORT/v1/chat/completions")
if [ "$hcode" = "200" ]; then
  echo "  healthy fixture returns 200"
else
  echo "healthy fixture returned $hcode, expected 200"
  exit 1
fi

section "2. start the proxy with --detailed_debug"
rm -f "$PROXY_LOG"
# Refuse to start if anything already holds the port. Otherwise the readiness probe
# below can be satisfied by a stranger's proxy and every later assertion measures it.
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
# Prove the responder IS our child: the port must be held by PROXY_PID or one of its
# descendants (the proxy forks workers), not merely reachable.
LISTENER_PIDS="$(python3 - "$PROXY_PORT" <<'PY'
import subprocess, sys
port = sys.argv[1]
# psutil is a litellm dependency, so this needs no extra tooling and works wherever
# python3 does. lsof is not POSIX and is missing on minimal runners.
try:
    import psutil
except ImportError:
    psutil = None
pids = None
if psutil is not None:
    try:
        pids = {
            c.pid
            for c in psutil.net_connections(kind="tcp")
            if c.status == psutil.CONN_LISTEN and c.laddr and c.laddr.port == int(port) and c.pid
        }
    except psutil.AccessDenied:
        # psutil can be importable yet denied at runtime (restricted process
        # environments, no privilege to inspect the FDs of other processes).
        # Fall through to the lsof path, which enumerates sockets owned by the
        # current user without that privilege.
        pids = None
if pids is None:
    out = subprocess.run(
        ["lsof", "-nP", f"-iTCP:{port}", "-sTCP:LISTEN", "-t"],
        capture_output=True, text=True,
    ).stdout
    print(" ".join(out.split()))
else:
    print(" ".join(str(p) for p in sorted(pids)))
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
if curl -fs "$PROXY_URL/health/liveliness" >/dev/null 2>&1; then
  echo "proxy live (pid $PROXY_PID owns port $PROXY_PORT)"
else
  echo "proxy not live"
  tail -30 "$PROXY_LOG"
  exit 1
fi

chat() {
  local body
  body=$(python3 -c "import json,sys; print(json.dumps({'model':'$MODEL','messages':[{'role':'user','content':'hi'}],'max_tokens':4}))")
  curl -sS -X POST "$PROXY_URL/v1/chat/completions" \
    -H "Authorization: Bearer $MASTER_KEY" -H "Content-Type: application/json" -d "$body"
}


fixture_count() {
  curl -s "http://127.0.0.1:$FOUR29_PORT/count" | python3 -c "import json,sys; print(json.load(sys.stdin)['count'])"
}

healthy_count() {
  curl -s "http://127.0.0.1:$HEALTHY_PORT/count" | python3 -c "import json,sys; print(json.load(sys.stdin)['count'])"
}

section "3. drive exactly 3 requests (the cooldown threshold) to the fixture"
before=$(fixture_count)
healthy_before=$(healthy_count)
echo "  fixture count before storm: $before (healthy baseline: $healthy_before)"
echo "  winner is novita/fp8 (fixture). Each call 429s at the fixture; the failure"
echo "  hook records novita/fp8. After 3, novita/fp8 is hot. Sleep 6s between"
echo "  requests so litellm's built-in 429 cooldown (5s) expires between them, letting"
echo "  the package's 3-hit cooldown accumulate."
errors=0
for i in 1 2 3; do
  echo "  --- request $i ---"
  out=$(chat)
  if echo "$out" | python3 -c "import json,sys; sys.exit(0 if json.load(sys.stdin).get('error') else 1)" 2>/dev/null; then
    errors=$((errors+1))
  fi
  echo "  -> $(echo "$out" | python3 -c "import json,sys; d=json.load(sys.stdin); print('error' if d.get('error') else 'choices='+str(len(d.get('choices',[]))))" 2>/dev/null || echo '(unparseable)')"
  sleep 6
done
after=$(fixture_count)
storm_hits=$((after - before))
echo "  fixture count after storm: $after (errors seen: $errors, fixture hits: $storm_hits)"
if [ "$errors" -lt 3 ]; then
  bad "expected 3 fixture-429 errors, got $errors"
fi
if [ "$storm_hits" -ne 3 ]; then
  bad "expected exactly 3 fixture hits during storm, got $storm_hits (before=$before after=$after)"
fi

section "4. send a final request; the cooldown must skip the hot winner"
echo "  after 3 429s, novita/fp8 is hot; this request should route to baseten/fp8"
out=$(chat)
echo "  -> $(echo "$out" | python3 -c "import json,sys; d=json.load(sys.stdin); print('error' if d.get('error') else 'provider='+str(d.get('provider'))+' choices='+str(len(d.get('choices',[]))))" 2>/dev/null || echo '(unparseable)')"
final_ok=$(echo "$out" | python3 -c "
import json, sys
d = json.load(sys.stdin)
sys.exit(0 if (not d.get('error') and d.get('choices')) else 1)
" 2>/dev/null && echo yes || echo no)
final_provider=$(echo "$out" | python3 -c "import json,sys; print(json.load(sys.stdin).get('provider') or '')" 2>/dev/null || echo "")
final_count=$(fixture_count)
healthy_after=$(healthy_count)
healthy_delta=$((healthy_after - healthy_before))
echo "  final response: success=$final_ok provider=$final_provider"
echo "  429-fixture count after final request: $final_count (was $after)"
echo "  healthy-fixture count: $healthy_after (baseline $healthy_before, delta $healthy_delta)"
if [ "$final_ok" != "yes" ]; then
  bad "final request did not return a successful completion"
elif [ "$final_count" -ne "$after" ]; then
  bad "final request touched the 429 fixture (count went $after -> $final_count); cooldown did not skip it"
elif [ "$healthy_delta" -ne 1 ]; then
  bad "expected exactly 1 new healthy-fixture hit for the final request, got $healthy_delta (baseline $healthy_before -> $healthy_after)"
else
  ok "cooldown skipped hot novita/fp8, routed to the healthy deployment (baseten/fp8); 429 fixture not re-hit ($after -> $final_count), healthy fixture hit exactly once ($healthy_before -> $healthy_after)"
fi

section "summary"
echo "  PASS=$pass FAIL=$fail"
echo "  proxy log: $PROXY_LOG"
echo "  note: winner was pre-populated in the cache so the callback deterministically"
echo "        narrows to the fixture; the cooldown wiring is what's under test."
if [ "$fail" -eq 0 ]; then
  echo "ALL PASS"
  exit 0
else
  echo "SOME FAILURES"
  exit 1
fi
