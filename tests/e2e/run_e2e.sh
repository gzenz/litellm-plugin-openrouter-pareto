#!/usr/bin/env bash
# OpenRouter pareto plugin end-to-end proof-of-fix (real OpenRouter only).
#
# Drives the litellm-plugin-openrouter-pareto callback the way production does:
# a live litellm proxy with the callback registered, multiple real OpenRouter
# deployments of z-ai/glm-5.2, real API calls that cost real money. Proves the
# real-OR wiring end-to-end: (1) the value-walk winner's provider.only reaches the
# wire (best-effort narrowing check - the DETERMINISTIC narrowing proof, that the
# callback returns exactly one deployment, lives in
# tests/test_plugin.py::test_narrows_to_winner_deployment; litellm's router does
# not log the post-filter deployment count, so a live test cannot observe the
# narrowed candidate set directly), (2) a warm cache keeps serving. The
# exclude_regions filter is proven separately with seeded telemetry
# (run_region_integration.sh) because its live cold-start refresh writes the
# cache asynchronously and races an inline assertion.
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

# Allocate a random port per run. A fixed port (4000) made the suite depend on
# ambient state: a stale proxy from a previous run that held the port would
# answer the readiness probe, and every later assertion measured that orphan
# (often running different code) instead of the proxy we just started. uvicorn
# worker forks can also survive the master's cleanup and keep holding the port,
# so a fixed port eventually collides with an orphan. A drawn port cannot collide
# with a previous run's orphan.
read -r ALLOC_PROXY <<EOF
$(python3 -c "
import socket
s = socket.socket(); s.bind(('127.0.0.1', 0))
print(s.getsockname()[1]); s.close()")
EOF
PROXY_PORT="${PROXY_PORT:-$ALLOC_PROXY}"
# 127.0.0.1 (not localhost) so curl resolves IPv4 only; a foreign IPv6 ::1
# listener on the same numeric port can then never satisfy the readiness probe
# or serve test requests, removing one TOCTOU impersonation vector.
PROXY_URL="http://127.0.0.1:${PROXY_PORT}"
MASTER_KEY="sk-1234"
MODEL="z-ai/glm-5.2"
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$HERE/../.." && pwd)"
CONFIG="$HERE/or_pareto_config.yaml"
PROXY_LOG="$HERE/proxy.log"
CACHE_JSON="$(python3 -c "import platformdirs; print(platformdirs.user_cache_path('litellm-plugin-openrouter-pareto') / 'cache.json')")"
LITELLM_DIR="${LITELLM_DIR:-$HOME/PycharmProjects/litellm}"

pass=0; fail=0
# PROXY_LAUNCHED is set to 1 as soon as PROXY_PID is assigned (after the setsid
# launch). Cleanup ALWAYS reaps this launch-scoped process group - it is ours, so
# killing it can never touch a foreign proxy, and gating on it means an abort
# after launch (port-ownership verification failure) still reaps the proxy we
# started instead of orphaning it. (Port-ownership verification gates whether the
# TEST proceeds via its own exit-on-failure; it does not gate cleanup.)
PROXY_LAUNCHED=0
ok()   { echo "  PASS: $1"; pass=$((pass+1)); }
bad()  { echo "  FAIL: $1"; fail=$((fail+1)); }
section() { echo; echo "=== $1 ==="; }

cleanup() {
  # Reap only the process GROUP we launched (setsid makes PROXY_PID a session
  # leader, so -PROXY_PID is the whole group: master + uvicorn workers). This is
  # launch-scoped, not port-scoped: it can never kill a foreign proxy that grabbed
  # the port after ours exited, because it targets our recorded group, not the
  # port. Gated on PROXY_LAUNCHED so a verification-failure abort still cleans up
  # the proxy we did start.
  if [ "${PROXY_LAUNCHED:-0}" = "1" ] && [ -n "${PROXY_PID:-}" ]; then
    kill -- -"$PROXY_PID" 2>/dev/null
    # a worker can still be mid-shutdown when the trap fires; give it a moment and
    # then force-reap the group so no orphan lingers.
    sleep 1
    kill -9 -- -"$PROXY_PID" 2>/dev/null
  fi
}
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
# Refuse to start if anything already holds the port. Otherwise the readiness
# probe below can be satisfied by a stranger's proxy and every later assertion
# measures it instead of the proxy we start here.
if curl -fs -m 2 "$PROXY_URL/health/liveliness" >/dev/null 2>&1; then
  echo "port $PROXY_PORT is already serving; refusing to run against a proxy we did not start" >&2
  exit 1
fi
# Launch the proxy in its own session/process group via os.setsid (macOS has no
# setsid(1), so a tiny python wrapper does it). PROXY_PID becomes the session
# leader, so its PGID == its PID and cleanup can kill -- -PROXY_PID to reap the
# master + uvicorn workers as a launch-scoped group rather than matching a
# reusable port string that a foreign proxy could later grab.
export OR_E2E_CONFIG="$CONFIG" OR_E2E_PORT="$PROXY_PORT"
( cd "$LITELLM_DIR" && exec python3 -c '
import os, sys
os.setsid()
os.execvp(sys.executable, [
    sys.executable, "litellm/proxy/proxy_cli.py",
    "--config", os.environ["OR_E2E_CONFIG"],
    "--detailed_debug", "--port", os.environ["OR_E2E_PORT"],
])
' ) >"$PROXY_LOG" 2>&1 &
PROXY_PID=$!
PROXY_LAUNCHED=1
echo "proxy PID $PROXY_PID (process-group leader); waiting for liveness..."
for _ in $(seq 1 60); do
  curl -fs "$PROXY_URL/health/liveliness" >/dev/null 2>&1 && break
  sleep 1
done
curl -fs "$PROXY_URL/health/liveliness" >/dev/null 2>&1 \
  && echo "proxy live" || { echo "proxy did not become live"; tail -40 "$PROXY_LOG"; exit 1; }
# Prove the responder IS our child: resolve EVERY PID listening on the port and
# confirm each is PROXY_PID or a descendant (the proxy forks uvicorn workers).
# Requiring ALL listeners to be ours closes the mixed-listener TOCTOU where a
# foreign process binds the same numeric port (e.g. on another address family)
# and one owned listener would otherwise let the run proceed against a stranger.
# 127.0.0.1 in PROXY_URL already pins curl to IPv4, so a foreign IPv6-only
# listener cannot serve requests, but this check still rejects it. Aborts WITHOUT
# killing the listener if any is not ours (PROXY_LAUNCHED still reaps only our
# own group; the foreign listener is left alone).
LISTENER_PIDS="$(python3 - "$PROXY_PORT" <<'PY'
import subprocess, sys
port = sys.argv[1]
# psutil is a litellm dependency, so this needs no extra tooling. lsof is the
# fallback when psutil is importable yet denied at runtime (restricted envs).
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
if [ -z "$LISTENER_PIDS" ]; then
  echo "no listener resolved on port $PROXY_PORT; cannot prove ownership" >&2
  tail -30 "$PROXY_LOG"
  exit 1
fi
# Every listener on the port must be ours. A single foreign listener (e.g. a
# TOCTOU bind on another address family) fails the whole check so we abort
# without killing anything we did not start.
ALL_OWNED=1
for lp in $LISTENER_PIDS; do
  this_owned=0
  probe="$lp"
  for _ in 1 2 3 4 5; do
    [ "$probe" = "$PROXY_PID" ] && { this_owned=1; break; }
    probe="$(ps -o ppid= -p "$probe" 2>/dev/null | tr -d ' ')"
    [ -z "$probe" ] || [ "$probe" = "1" ] || [ "$probe" = "0" ] && break
  done
  if [ "$this_owned" != "1" ]; then
    ALL_OWNED=0
  fi
done
if [ "$ALL_OWNED" != "1" ]; then
  echo "port $PROXY_PORT has listener(s) [$LISTENER_PIDS] not all in our proxy family ($PROXY_PID); aborting without killing" >&2
  tail -30 "$PROXY_LOG"
  exit 1
fi
echo "  verified: every listener on port $PROXY_PORT is in our proxy family (leader pid $PROXY_PID)"
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

echo "  sending 3 requests; EACH must succeed and carry an accepted provider.only wire pin that OpenRouter honored"
# The winner org and safe-set orgs are needed per request, so resolve them first.
# OR's `provider` field is the org name only ("BaseTen") while the winner slug is the
# full endpoint id ("baseten/fp8"), so the response-provider comparison uses the org.
# The wire `provider.only` pin, however, IS the full slug, so it is compared exactly
# (grep -Fqx) against SAFE_SLUGS / CONFIGURED_SLUGS - org-only comparison would let a
# broken pin like baseten/other-endpoint pass because its org is in both sets.
WINNER_ORG="${WINNER_SLUG%%/*}"
SAFE_ORGS=$(python3 -c "
import json, sys, pathlib
p = pathlib.Path(sys.argv[1])
d = json.loads(p.read_text())
e = next((v for k, v in d['entries'].items() if k.split(chr(0))[0] == 'z-ai/glm-5.2'), {})
print(' '.join(sorted({s.split('/')[0] for s in e.get('safe_set', [])})))
" "$CACHE_JSON" 2>/dev/null || echo "")
SAFE_SLUGS=$(python3 -c "
import json, sys, pathlib
p = pathlib.Path(sys.argv[1])
d = json.loads(p.read_text())
e = next((v for k, v in d['entries'].items() if k.split(chr(0))[0] == 'z-ai/glm-5.2'), {})
print('\n'.join(sorted(set(e.get('safe_set', [])))))
" "$CACHE_JSON" 2>/dev/null || echo "")

# Slugs the proxy actually has a deployment for (parsed from the generated config).
# The value-walk winner is live-OR-dependent and may land on a provider this config
# has no deployment for; in that case the plugin correctly narrows to a configured
# safe-set member, and case 1 must accept that as a telemetry-driven fallback.
CONFIGURED_ORGS=$(python3 -c "
import re, sys, pathlib
text = pathlib.Path(sys.argv[1]).read_text()
print(' '.join(sorted({m.split('/')[0] for m in re.findall(r'only: \[([^\]]+)\]', text)})))
" "$CONFIG" 2>/dev/null || echo "")
CONFIGURED_SLUGS=$(python3 -c "
import re, sys, pathlib
text = pathlib.Path(sys.argv[1]).read_text()
print('\n'.join(sorted(set(re.findall(r'only: \[([^\]]+)\]', text)))))
" "$CONFIG" 2>/dev/null || echo "")
WINNER_DEPLOYABLE=0
# Deployability is an exact-deployment-slug property, not an org property: a
# winner of `baseten/other-endpoint` is NOT deployable just because `baseten/fp8`
# is configured. Compare the full winner slug against the configured slug set.
if [ -n "$WINNER_SLUG" ] && echo "$CONFIGURED_SLUGS" | grep -Fqx "$WINNER_SLUG"; then
  WINNER_DEPLOYABLE=1
fi
if [ "$WINNER_DEPLOYABLE" = "0" ]; then
  echo "  winner org $WINNER_ORG has no deployment in this config ($CONFIGURED_ORGS);"
  echo "  case 1 asserts the plugin narrows to a configured safe-set member instead"
fi

if [ -z "$WINNER_SLUG" ]; then
  bad "no winner in cache.json; telemetry fetch did not complete"
else
  # case 1 is a BEST-EFFORT live-OR narrowing check, NOT a deterministic proof.
  # Each request is judged on its wire `provider.only` pin (the plugin's direct
  # output: it sets `only: [<slug>]` on the chosen deployment copy). The wire pin
  # is stronger than the response `provider` field (OpenRouter's report of who
  # served), but it still only proves WHICH deployment was selected, not that the
  # candidate set was narrowed to one before simple-shuffle selected it: an
  # unrestricted router could, by chance, shuffle to the same/winner deployment
  # across all requests (false-pass ~1/343 strict, ~1/49 fallback). Litellm's
  # router does not log the post-filter deployment count, so a live test cannot
  # observe the narrowed candidate set directly. The DETERMINISTIC narrowing proof
  # lives in tests/test_plugin.py::test_narrows_to_winner_deployment, which
  # asserts the callback returns exactly the one-element deployment list; this
  # e2e proves the real-OR wiring (telemetry -> winner -> wire pin) end-to-end.
  REQ_FAILURES=0
  SERVED_PROVIDERS=""
  WIRE_PINS=""
  for i in 1 2 3; do
    echo "  --- request $i ---"
    LOG_OFFSET=$(wc -c < "$PROXY_LOG" | tr -d ' ')
    out=$(chat "say hi briefly")
    prov=$(echo "$out" | python3 -c "import json,sys; d=json.load(sys.stdin); print((d.get('provider') or '').lower())" 2>/dev/null || echo "")
    choices=$(echo "$out" | python3 -c "import json,sys; d=json.load(sys.stdin); print(len(d.get('choices',[])))" 2>/dev/null || echo "0")
    # The wire pin litellm actually forwarded, from this request's log slice. The
    # "Final returned optional params" line is the canonical post-plugin pin.
    wire_pin=$(tail -c "+$((LOG_OFFSET + 1))" "$PROXY_LOG" 2>/dev/null \
      | grep "Final returned optional params" \
      | grep -o "'only': \['[^']*'\]" | head -1 | sed -E "s/.*\['([^']*)'\].*/\1/")
    echo "  -> provider: ${prov:-(none)} | choices: $choices | wire_pin: ${wire_pin:-(none)}"
    SERVED_PROVIDERS="$SERVED_PROVIDERS ${prov:-<none>}"
    WIRE_PINS="$WIRE_PINS ${wire_pin:-<none>}"
    # 429 evidence scoped to just this request's log slice, and it must name the winner.
    req_429=$(tail -c "+$((LOG_OFFSET + 1))" "$PROXY_LOG" 2>/dev/null \
      | grep -i "429" | grep -ci "$WINNER_ORG" || true)
    wire_org="${wire_pin%%/*}"
    req_ok=0
    if [ "$choices" = "0" ] || [ "$choices" = "?" ]; then
      bad "request $i returned no completion: $(echo "$out" | head -c 200)"
      REQ_FAILURES=$((REQ_FAILURES + 1))
    elif [ -z "$wire_pin" ]; then
      bad "request $i left no wire provider.only pin in the log; cannot prove narrowing"
      REQ_FAILURES=$((REQ_FAILURES + 1))
    elif [ "$WINNER_DEPLOYABLE" = "1" ] && [ "$wire_pin" = "$WINNER_SLUG" ]; then
      echo "     request $i pinned the winner ($WINNER_SLUG) on the wire"
      req_ok=1
    elif [ "$WINNER_DEPLOYABLE" = "1" ] && [ "${req_429:-0}" -gt 0 ] && [ -n "$SAFE_SLUGS" ] && \
         echo "$SAFE_SLUGS" | grep -Fqx "$wire_pin"; then
      echo "     request $i: winner 429'd (${req_429} lines); wire pin walked to safe_set slug $wire_pin"
      req_ok=1
    elif [ "$WINNER_DEPLOYABLE" = "0" ] && [ -n "$CONFIGURED_SLUGS" ] && [ -n "$SAFE_SLUGS" ] && \
         echo "$CONFIGURED_SLUGS" | grep -Fqx "$wire_pin" && \
         echo "$SAFE_SLUGS" | grep -Fqx "$wire_pin"; then
      echo "     request $i: winner $WINNER_ORG not deployable; wire pin is configured safe_set slug $wire_pin"
      req_ok=1
    else
      if [ "$WINNER_DEPLOYABLE" = "0" ]; then
        bad "request $i wire pin '$wire_pin' is not a configured safe_set slug (configured: $CONFIGURED_ORGS; safe_set: $SAFE_ORGS)"
      else
        bad "request $i wire pin '$wire_pin' is neither the winner ($WINNER_SLUG) nor a safe_set slug after a winner 429"
      fi
      REQ_FAILURES=$((REQ_FAILURES + 1))
    fi
    # The wire pin proves LiteLLM FORWARDED the pin. The response `provider` (org
    # only, per OR's API) must match that pin's org, proving OpenRouter HONORED it.
    # Without this, a pin that OR ignored (serving a different provider) would pass.
    if [ "$req_ok" = "1" ]; then
      if [ -z "$prov" ]; then
        bad "request $i pinned '$wire_pin' but the response named no provider; OpenRouter honor cannot be verified"
        REQ_FAILURES=$((REQ_FAILURES + 1))
      elif [ -n "$wire_org" ] && ! printf '%s\n' "$prov" | grep -qix "$wire_org"; then
        bad "request $i pinned '$wire_pin' but OpenRouter reported provider '$prov' (org mismatch); the pin was not honored"
        REQ_FAILURES=$((REQ_FAILURES + 1))
      fi
    fi
  done
  echo "  served providers across 3 requests:$SERVED_PROVIDERS"
  echo "  wire pins across 3 requests:$WIRE_PINS"
  # When the winner has no deployment in this config, the plugin narrows to ONE
  # configured safe_set member (it never tries the unconfigured winner, so there
  # are no winner-429 walks to vary the pick). All 3 wire pins being the SAME slug
  # is strong evidence (simple-shuffle over all configured deployments would tend
  # to vary the pin), but not deterministic proof - see the case 1 header. The
  # deterministic narrowing proof is in tests/test_plugin.py.
  if [ "$REQ_FAILURES" = "0" ] && [ "$WINNER_DEPLOYABLE" = "0" ]; then
    unique_pins=$(echo "$WIRE_PINS" | tr ' ' '\n' | grep -v '^$' | sort -u | wc -l | tr -d ' ')
    if [ "$unique_pins" != "1" ]; then
      bad "winner not deployable but the 3 requests had $unique_pins different wire pins ($WIRE_PINS); the plugin did not narrow to one safe_set slug (unrestricted shuffle?)"
      REQ_FAILURES=$((REQ_FAILURES + 1))
    fi
  fi
  if [ "$REQ_FAILURES" = "0" ]; then
    if [ "$WINNER_DEPLOYABLE" = "1" ]; then
      ok "all 3 requests pinned the value-walk winner ($WINNER_SLUG) on the wire, or walked to a safe_set slug on a winner 429 (best-effort; deterministic proof in test_plugin.py)"
    else
      ok "winner $WINNER_ORG had no deployment; all 3 requests pinned the SAME configured safe_set slug on the wire (best-effort telemetry-driven fallback; deterministic proof in test_plugin.py)"
    fi
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
