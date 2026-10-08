#!/usr/bin/env bash
# Continuous batching end to end (--slots N), MODEL-FREE: the "slow" fixture
# (noise weights, 4 wide layers) unless a checkpoint is given as $1.
#
#   1. the serialized server (--slots 1) answers 4 greedy prompts: the reference
#   2. a --slots 4 server answers the same 4 prompts CONCURRENTLY, byte for
#      byte the same (the default decode product is bit-identical to solo),
#      and a streamed one reassembles to the same text
#   3. /health shows the steps really were batched (mean batch > 1) and which
#      decode product ran; usage carries queue_ms and ttft_ms
#   4. a short request admitted beside 3 long streams gets its first token
#      promptly (sliced prefill + admission at the top of every step)
#   5. slots full + queue full = 503 + Retry-After at once
#   6. clients that leave free their slots: the next request is prompt and the
#      CPU goes idle
#
#   bash tests/test_server_slots.sh [model.gguf]     (make test-server-slots)
set -uo pipefail

SERVER="${SERVER:-./mynah-slm-server}"
PORT="${PORT:-8141}"
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
SRV=""
trap '[ -n "$SRV" ] && kill $SRV 2>/dev/null; rm -rf "$TMP"' EXIT
# Wait on the CLIENTS only: a bare `wait` also waits for the server, which
# never exits (the hang test_server.sh documents).
PIDS=""
spawn_wait() { for p in $PIDS; do wait "$p" 2>/dev/null; done; PIDS=""; }

start() {   # start <log> <args...>
    local log=$1; shift
    "$SERVER" -m "$MODEL" --port "$PORT" "$@" >"$log" 2>&1 &
    SRV=$!
    for _ in $(seq 120); do
        curl -sf "localhost:$PORT/health" >/dev/null 2>&1 && return 0
        kill -0 $SRV 2>/dev/null || break
        sleep 0.25
    done
    echo "FAIL server never came up"; cat "$log"; exit 1
}
stop() { kill $SRV 2>/dev/null; wait $SRV 2>/dev/null; SRV=""; }

# Seeded SAMPLING (temperature 0.8, a seed per prompt) rather than greedy, so
# the sampler's RNG path is part of what must survive batching. (The "slow"
# fixture has an untied head: tied, noise weights echo the prompt's last
# token forever and every answer would be the same.) A seeded sampler is just as deterministic,
# and the default decode product is bit-identical to solo, so batching must
# not change a single sampled token.
body() {   # body <max_tokens> <stream 0|1> <prompt> [seed]
    python3 -c "import json,sys; print(json.dumps({'messages':[{'role':'user','content':sys.argv[3]}],'max_tokens':int(sys.argv[1]),'temperature':0.8,'seed':int(sys.argv[4]),'stream':sys.argv[2]=='1'}))" "$1" "$2" "$3" "${4:-7}"
}
content() {   # the answer text of a non-stream response on stdin, bytes kept
    python3 -c "import json,sys; d=json.loads(sys.stdin.buffer.read().decode('utf-8','replace')); print(repr(d['choices'][0]['message']['content']))"
}
chat() { curl -s --max-time 120 -X POST "localhost:$PORT/v1/chat/completions" -H 'Content-Type: application/json' -d "$1"; }
health() { curl -s "localhost:$PORT/health" | python3 -c "import json,sys; print(json.load(sys.stdin)$1)"; }

PROMPTS=("Ciao! Come stai?" "Wie heisst du?" "The ledger for 1887 lists twelve names." "La contabilidad no cuadra.")

# ONLY="7 8" runs just those self-contained sections (1-6 are one chain,
# named "base"); unset runs everything.
want() { [ -z "${ONLY:-}" ] || [[ " $ONLY " == *" $1 "* ]]; }

if want base; then
# ── 1. the reference: the serialized server ───────────────────────────────────
start "$TMP/ref.log"
for i in 0 1 2 3; do chat "$(body 24 0 "${PROMPTS[$i]}" $((100 + i)))" | content > "$TMP/ref$i"; done
stop
[ -s "$TMP/ref0" ] && ok "the serialized server answered the 4 reference prompts" \
    || bad "reference answers" "$(cat "$TMP/ref.log")"
DISTINCT=$(cat "$TMP"/ref? | sort -u | wc -l)
[ "$DISTINCT" -ge 3 ] && ok "... with $DISTINCT distinct answers (the comparison below is not vacuous)" \
    || bad "reference answers differ from each other" "$(cat "$TMP"/ref?)"

# ── 2. the same prompts, concurrently, on 4 slots ─────────────────────────────
start "$TMP/slots.log" --slots 4 --queue 2 --max-conns 16
for i in 0 1 2 3; do ( chat "$(body 24 0 "${PROMPTS[$i]}" $((100 + i)))" | content > "$TMP/got$i" ) & PIDS="$PIDS $!"; done
curl -sN --max-time 120 -X POST "localhost:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "$(body 24 1 "${PROMPTS[0]}" 100)" > "$TMP/stream0" &
PIDS="$PIDS $!"
spawn_wait
same=1
for i in 0 1 2 3; do
    cmp -s "$TMP/ref$i" "$TMP/got$i" || { same=0; echo "     #$i ref=$(cat "$TMP/ref$i") got=$(cat "$TMP/got$i")"; }
done
[ $same -eq 1 ] && ok "4 concurrent requests on 4 slots == the serialized answers, byte for byte" \
    || bad "concurrent slot answers equal the serialized ones"
STREAMED=$(python3 -c "
import json,sys
out=[]
for line in open(sys.argv[1],'rb').read().decode('utf-8','replace').splitlines():
    line=line.strip()
    if not line.startswith('data: ') or line.endswith('[DONE]'): continue
    d=json.loads(line[6:])
    out.append(d['choices'][0].get('delta',{}).get('content',''))
print(repr(''.join(out)))" "$TMP/stream0")
[ "$STREAMED" = "$(cat "$TMP/ref0")" ] && ok "a stream batched with them reassembles to the same text" \
    || bad "streamed == reference" "stream=$STREAMED ref=$(cat "$TMP/ref0")"
grep -q '"finish_reason":"stop"' "$TMP/stream0" && grep -q 'queue_ms' "$TMP/stream0" \
    && ok "the stream's final event carries finish_reason and queue_ms" \
    || bad "final SSE event" "$(tail -c 400 "$TMP/stream0")"

MB=$(health "['mean_batch']"); PROD=$(health "['decode_product']"); ST=$(health "['steps']")
python3 -c "import sys; sys.exit(0 if $MB > 1.0 else 1)" \
    && ok "the decode steps were batched (mean batch $MB over $ST steps, product $PROD)" \
    || bad "steps were batched" "mean_batch=$MB"
U=$(chat "$(body 4 0 "ok")" | python3 -c "import json,sys; print(json.load(sys.stdin)['usage'])")
echo "$U" | grep -q "queue_ms" && echo "$U" | grep -q "ttft_ms" && ok "usage carries queue_ms and ttft_ms" \
    || bad "usage fields" "$U"

# ── 4. a short request beside 3 long streams gets its first token promptly ────
for i in 1 2 3; do
    curl -sN --max-time 8 -o /dev/null -X POST "localhost:$PORT/v1/chat/completions" \
        -H 'Content-Type: application/json' -d "$(body 4000 1 "${PROMPTS[$i]}")" &
    PIDS="$PIDS $!"
done
sleep 1
T=$(now)
curl -sN --max-time 30 -X POST "localhost:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "$(body 8 1 "${PROMPTS[0]}")" | python3 -c "
import sys,time,json
t0=float(sys.argv[1]); first=None; usage=None
for line in sys.stdin:
    if first is None and '\"content\"' in line: first=round(time.time()-t0,3)
    if '\"usage\"' in line: usage=json.loads(line.strip()[6:])['usage']
print(first if first is not None else 99)
if usage: print('     server side: queue_ms %.1f, ttft_ms %.1f (from admission), prompt %d tokens' % (usage['queue_ms'], usage['ttft_ms'], usage['prompt_tokens']), file=sys.stderr)
" "$T" > "$TMP/first" 2> "$TMP/first.usage"
FIRST=$(cat "$TMP/first"); cat "$TMP/first.usage"
python3 -c "import sys; sys.exit(0 if '$FIRST' and float('${FIRST:-99}') < 2.0 else 1)" \
    && ok "beside 3 long streams a new request's first token arrived in ${FIRST}s" \
    || bad "first token beside 3 long streams" "${FIRST:-never}"
B=$(health "['decoding']")
echo "     (3 long streams still decoding: $B)"
spawn_wait

# ── 5. slots full and queue full: 503 + Retry-After, at once ──────────────────
for i in 1 2 3 4 5 6; do
    curl -s --max-time 6 -o /dev/null -X POST "localhost:$PORT/v1/chat/completions" \
        -H 'Content-Type: application/json' -d "$(body 4000 0 "${PROMPTS[$((i % 4))]}")" &
    PIDS="$PIDS $!"
done
sleep 1.5
T=$(now)
H=$(curl -s -i --max-time 5 -X POST "localhost:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "$(body 4 0 "ok")")
D=$(python3 -c "print(round($(now) - $T, 2))")
echo "$H" | head -1 | grep -q " 503 " && ok "4 slots busy + 2 queued: the 7th gets 503 in ${D}s" \
    || bad "503 when the queue is full" "$(echo "$H" | head -1) after ${D}s"
echo "$H" | grep -qi "^Retry-After: " && ok "... with Retry-After" || bad "Retry-After" "$H"
RQ=$(health "['rejected_queue']"); Q=$(health "['queued']")
[ "${RQ:-0}" -ge 1 ] && ok "/health counts it (rejected_queue $RQ, queued $Q)" || bad "rejected_queue" "$RQ"
spawn_wait

# ── 6. everyone left: slots freed, next request prompt, CPU idle ──────────────
sleep 0.5
L=$(health "['live']")
[ "${L:-1}" -eq 0 ] && ok "every slot was freed when its client left (live 0)" || bad "slots freed" "live=$L"
T=$(now); chat "$(body 4 0 "ok")" >/dev/null; D=$(python3 -c "print(round($(now) - $T, 2))")
python3 -c "import sys; sys.exit(0 if $D < 3.0 else 1)" && ok "the next request took ${D}s" \
    || bad "next request after the leavers" "${D}s"
if [ -r "/proc/$SRV/stat" ]; then
    sleep 0.5
    T0=$(awk '{print $14 + $15}' "/proc/$SRV/stat"); sleep 1
    TICKS=$(( $(awk '{print $14 + $15}' "/proc/$SRV/stat") - T0 ))
    [ "$TICKS" -lt "$(( $(getconf CLK_TCK) / 5 ))" ] && ok "idle server: ${TICKS} ticks in 1 s" \
        || bad "the server goes idle" "${TICKS} ticks/s"
fi
C=$(health "['slot_cancelled']")
[ "${C:-0}" -ge 6 ] && ok "/health counts the slot cancellations ($C)" || bad "slot_cancelled" "$C"
stop
fi

if want 7; then
# ── 7. shutdown with requests still being READ ───────────────────────────────
# Connection threads are detached: one still reading its request when SIGTERM
# arrives reaches the engine after the accept loop has returned. It must get
# a 503 — not a push into a freed queue, not /health stats read from a freed
# scheduler (review B1: a use-after-free under ASan) — and the server must
# still exit cleanly. Run it with an ASan server (SERVER=...) to see the UAF.
start "$TMP/down.log" --slots 2
python3 - "$PORT" "$SRV" > "$TMP/down.out" 2>&1 <<'PY'
import json, os, signal, socket, sys, time
port, pid = int(sys.argv[1]), int(sys.argv[2])
b = json.dumps({'messages': [{'role': 'user', 'content': 'hi'}], 'max_tokens': 4}).encode()
chat = b'POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n' % len(b) + b
health = b'GET /health HTTP/1.1\r\nHost: x\r\n\r\n'
socks = []
for data in (chat, health):
    s = socket.create_connection(('127.0.0.1', port)); s.sendall(data[:10]); socks.append((s, data))
time.sleep(0.3)
os.kill(pid, signal.SIGTERM)
time.sleep(2.5)            # the accept loop (1 s timeout) is gone by now
for s, data in socks:
    try:
        s.sendall(data[10:]); s.settimeout(10)
        r = b''
        while True:
            c = s.recv(4096)
            if not c: break
            r += c
        print(r.split(b'\r\n')[0].decode('latin-1') or 'EMPTY')
    except Exception as e:
        print('ERROR', e)
PY
for _ in $(seq 150); do kill -0 $SRV 2>/dev/null || break; sleep 0.1; done
if kill -0 $SRV 2>/dev/null; then RC=timeout; kill -9 $SRV; wait $SRV 2>/dev/null; else wait $SRV; RC=$?; fi
SRV=""
R1=$(sed -n 1p "$TMP/down.out"); R2=$(sed -n 2p "$TMP/down.out")
echo "$R1" | grep -q " 503 " && ok "a request read after SIGTERM gets 503, not the freed engine ($R1)" \
    || bad "a request read after SIGTERM gets 503" "$R1 | $(cat "$TMP/down.out")"
echo "$R2" | grep -q " 503 " && ok "/health read after SIGTERM gets 503 ($R2)" \
    || bad "/health read after SIGTERM gets 503" "$R2"
[ "$RC" = 0 ] && ! grep -q "AddressSanitizer\|ThreadSanitizer" "$TMP/down.log" \
    && ok "the server exited cleanly (rc $RC)" \
    || bad "the server exits cleanly after a late request" "rc $RC; $(grep -A3 Sanitizer "$TMP/down.log" | head -8)"
fi

if want 8; then
# ── 8. SIGTERM is a bounded shutdown, in both modes ──────────────────────────
# Review R1: the scheduler used to drain every admitted AND queued job to
# max_tokens before exiting (and the serialized server ran the current one
# to the end). Now queued requests get 503 at once and running ones stop at
# their next step with a final error event, so the process exits within
# about a second (the accept loop's poll) instead of minutes.
for MODE in 1 2; do
    start "$TMP/term$MODE.log" --slots $MODE
    python3 - "$PORT" "$SRV" > "$TMP/term$MODE.out" 2>&1 <<'PY'
import json, os, signal, socket, sys, time
port, pid = int(sys.argv[1]), int(sys.argv[2])
b = json.dumps({'messages': [{'role': 'user', 'content': 'hi'}], 'max_tokens': 2500,
                'stream': True, 'temperature': 0}).encode()
req = b'POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n' % len(b) + b
socks = []
for _ in range(3):
    s = socket.create_connection(('127.0.0.1', port)); s.sendall(req); socks.append(s)
time.sleep(1.0)
t = time.time()
os.kill(pid, signal.SIGTERM)
for _ in range(300):                      # 30 s at most
    try:
        os.kill(pid, 0)
    except OSError:
        break
    if open('/proc/%d/stat' % pid).read().split()[2] == 'Z':
        break
    time.sleep(0.1)
print('exit_s %.1f' % (time.time() - t))
for s in socks:
    s.settimeout(2); data = b''
    try:
        while True:
            c = s.recv(1 << 20)
            if not c: break
            data += c
    except Exception:
        pass
    end = data[-200:]
    ok = (b' 503 ' in data[:40]) or (b'server_shutdown' in end and end.endswith(b'data: [DONE]\n\n'))
    print('client', 'final' if ok else 'NO-FINAL', len(data), repr(end[-90:]))
PY
    for _ in $(seq 100); do kill -0 $SRV 2>/dev/null || break; sleep 0.1; done
    if kill -0 $SRV 2>/dev/null; then kill -9 $SRV; fi
    wait $SRV 2>/dev/null; SRV=""
    EX=$(awk '/^exit_s/ {print $2}' "$TMP/term$MODE.out")
    python3 -c "import sys; sys.exit(0 if float('${EX:-99}') < 5.0 else 1)" \
        && ok "--slots $MODE: SIGTERM with 3 long streams (running and queued) exits in ${EX}s" \
        || bad "--slots $MODE: SIGTERM is a bounded shutdown" "exit after ${EX:-never}s"
    N=$(grep -c "^client final" "$TMP/term$MODE.out")
    [ "$N" -eq 3 ] && ok "--slots $MODE: every client got a final answer (error event + [DONE], or 503)" \
        || bad "--slots $MODE: every client got a final answer" "$(cat "$TMP/term$MODE.out")"
    grep -q "Sanitizer" "$TMP/term$MODE.log" && bad "--slots $MODE: no sanitizer report" \
        "$(grep -A3 Sanitizer "$TMP/term$MODE.log" | head -8)"
done
fi

echo
[ $fail -eq 0 ] && echo "PASS" || { echo "FAILED ($fail)"; tail -n 20 "$TMP"/*.log; }
exit $fail
