#!/usr/bin/env bash
# No zombie work: a client that leaves must stop costing CPU at the next step
# boundary, wherever it left — streaming, non-streaming, during the prompt,
# or while queued behind another request — and the next client must be
# served promptly. Plus the connection cap answers 503 + Retry-After.
#
# MODEL-FREE by default: writes tests/fixture_model.c's "slow" fixture (noise
# weights, 4 wide layers, ~25 MB) so a long generation lasts tens of seconds.
# Pass a real checkpoint as $1 to run the same checks against it.
#
#   sh tests/test_server_cancel.sh [model.gguf]        (make test-server-cancel)
#
# Exit 0 pass, 1 fail, 77 skip (no curl / no python3).
set -uo pipefail

SERVER="${SERVER:-./mynah-slm-server}"
PORT="${PORT:-8139}"
command -v curl >/dev/null && command -v python3 >/dev/null || { echo "SKIP need curl and python3"; exit 77; }
[ -x "$SERVER" ] || { echo "SKIP no $SERVER built"; exit 77; }

TMP=$(mktemp -d)
MODEL="${1:-}"
if [ -z "$MODEL" ]; then
    [ -x tests/write_fixture ] || { echo "SKIP no tests/write_fixture built"; exit 77; }
    MODEL="$TMP/slow.gguf"
    tests/write_fixture slow "$MODEL" || { echo "FAIL cannot write the fixture"; exit 1; }
fi

fail=0
ok()  { echo "ok   $1"; }
bad() { echo "FAIL $1  <- ${2:-}"; fail=$((fail+1)); }
now() { python3 -c 'import time; print(time.time())'; }
elapsed() { python3 -c "print(round($(now) - $1, 2))"; }
lt() { python3 -c "import sys; sys.exit(0 if $1 < $2 else 1)"; }
health() { curl -s "localhost:$PORT/health" | python3 -c "import json,sys; print(json.load(sys.stdin)$1)"; }

LOG="$TMP/server.log"
# CAP_ARGS="" runs a server without the flag (the control against a build
# from before this test existed).
"$SERVER" -m "$MODEL" --port "$PORT" ${CAP_ARGS---max-conns 3} >"$LOG" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null; wait $SRV 2>/dev/null; rm -rf "$TMP"' EXIT
for _ in $(seq 120); do
    curl -sf "localhost:$PORT/health" >/dev/null 2>&1 && break
    kill -0 $SRV 2>/dev/null || break
    sleep 0.25
done
curl -sf "localhost:$PORT/health" >/dev/null 2>&1 || { echo "FAIL server never came up"; cat "$LOG"; exit 1; }

# body <max_tokens> <stream> [prompt]
body() {
    python3 -c "import json,sys; print(json.dumps({'messages':[{'role':'user','content':sys.argv[3]}],'max_tokens':int(sys.argv[1]),'temperature':0,'stream':sys.argv[2]=='1'}))" "$1" "$2" "${3:-Ciao! Come stai?}"
}
post() {   # post <curl-max-time> <body>
    curl -s --max-time "$1" -o /dev/null -X POST "localhost:$PORT/v1/chat/completions" \
        -H 'Content-Type: application/json' -d "$2"
}

# A short request, timed: the yardstick for "served promptly".
T=$(now); post 60 "$(body 4 0)"; SHORT=$(elapsed "$T")
echo "     a 4-token request alone: ${SHORT}s"
# Promptly = the short request's own time plus a generous margin for a
# shared machine, and far below the long generation it would otherwise wait
# behind (8000 tokens).
LIMIT=$(python3 -c "print(round($SHORT * 3 + 2.0, 2))")

# How long would the long one take if nobody cancelled it? Measured, so the
# "promptly" claim has a denominator.
T=$(now); post 60 "$(body 200 0)"; PER=$(python3 -c "print(($(now) - $T) / 200)")
echo "     ~$(python3 -c "print(round($PER*1000,1))") ms/token; 8000 tokens would take ~$(python3 -c "print(round($PER*8000))") s"

# ── 1. non-streaming client leaves mid-generation ─────────────────────────────
post 1 "$(body 8000 0)"
T=$(now); post 60 "$(body 4 0)"; D=$(elapsed "$T")
lt "$D" "$LIMIT" && ok "after a non-stream client left, the next request took ${D}s (< ${LIMIT}s)" \
    || bad "after a non-stream client left, the next request is prompt" "${D}s"

# ── 2. streaming client leaves mid-generation ─────────────────────────────────
curl -sN --max-time 1 -o /dev/null -X POST "localhost:$PORT/v1/chat/completions" \
    -H 'Content-Type: application/json' -d "$(body 8000 1)"
T=$(now); post 60 "$(body 4 0)"; D=$(elapsed "$T")
lt "$D" "$LIMIT" && ok "after a streaming client left, the next request took ${D}s" \
    || bad "after a streaming client left, the next request is prompt" "${D}s"

# ── 3. client leaves during a long PROMPT ─────────────────────────────────────
LONG=$(python3 -c "print('la contabilita non quadra ' * 160)")
post 0.3 "$(body 8 0 "$LONG")"
T=$(now); post 60 "$(body 4 0)"; D=$(elapsed "$T")
lt "$D" "$LIMIT" && ok "after a client left during its prompt, the next request took ${D}s" \
    || bad "after a client left during its prompt, the next request is prompt" "${D}s"

# ── 4. client leaves while QUEUED behind another request ──────────────────────
post 3 "$(body 8000 0)" &
A=$!
sleep 0.5
post 1 "$(body 8000 0)"             # queued behind A, gives up after 1 s
wait $A
Q=$(health "['cancelled_queued']")
[ "${Q:-0}" -ge 1 ] && ok "a client that left while queued was never run (cancelled_queued=$Q)" \
    || bad "a client that left while queued was never run" "cancelled_queued=$Q"

# ── 5. the CPU goes idle once everyone has left ───────────────────────────────
if [ -r "/proc/$SRV/stat" ]; then
    post 1 "$(body 8000 0)"
    sleep 0.3
    T0=$(awk '{print $14 + $15}' "/proc/$SRV/stat"); sleep 1
    T1=$(awk '{print $14 + $15}' "/proc/$SRV/stat")
    TICKS=$((T1 - T0))
    HZ=$(getconf CLK_TCK)
    # A decoding server burns ~threads * HZ ticks a second; idle is ~0.
    lt "$TICKS" "$((HZ / 5))" && ok "1.3 s after the last client left the server used ${TICKS} ticks in 1 s (idle)" \
        || bad "the server goes idle after the last client left" "${TICKS} ticks/s (HZ $HZ)"
else
    echo "SKIP CPU idle check: no /proc"
fi

C=$(health "['cancelled']")
[ "${C:-0}" -ge 4 ] && ok "/health counts the cancellations (cancelled=$C)" \
    || bad "/health counts the cancellations" "cancelled=$C"
grep -q "cancelled: client gone during decode" "$LOG" && ok "the log says who was cancelled, where, after how many tokens" \
    || bad "the log records the cancellation" "$(tail -5 "$LOG")"
grep -q "client gone during prefill" "$LOG" && ok "... including during the prompt" \
    || bad "the log records a cancellation during the prompt" "$(grep cancelled "$LOG")"

# ── 6. the connection cap: 503 + Retry-After, never parked ────────────────────
post 4 "$(body 8000 0)" & P1=$!
post 4 "$(body 8000 0)" & P2=$!
post 4 "$(body 8000 0)" & P3=$!
sleep 0.5
H=$(curl -s -i --max-time 2 "localhost:$PORT/health")
echo "$H" | head -1 | grep -q " 503 " && ok "over --max-conns the answer is 503, at once" \
    || bad "over --max-conns the answer is 503" "$(echo "$H" | head -1)"
echo "$H" | grep -qi "^Retry-After: " && ok "... with Retry-After" || bad "503 carries Retry-After" "$H"
wait $P1 $P2 $P3
sleep 0.3
R=$(health "['rejected']")
[ "${R:-0}" -ge 1 ] && ok "/health counts rejected connections (rejected=$R)" || bad "/health counts rejections" "$R"

echo
[ $fail -eq 0 ] && echo "PASS" || { echo "FAILED ($fail)"; echo "---- server log ----"; tail -20 "$LOG"; }
exit $fail
