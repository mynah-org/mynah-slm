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
grep -q '"finish_reason":"length"' "$TMP/stream0" && grep -q 'queue_ms' "$TMP/stream0" \
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

if want 9; then
# ── 9. clients that leave while QUEUED give their place back ──────────────────
# Review B2: nobody probed a queued request, so two clients that gave up while
# queued behind 2 busy slots kept the queue full and a live client got 503.
start "$TMP/ghost.log" --slots 2 --queue 2
python3 - "$PORT" > "$TMP/ghost.out" 2>&1 <<'PY'
import json, socket, sys, time
port = int(sys.argv[1])
def req(n, stream):
    b = json.dumps({'messages': [{'role': 'user', 'content': 'hi'}], 'max_tokens': n,
                    'stream': stream, 'temperature': 0}).encode()
    return b'POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n' % len(b) + b
def health():
    s = socket.create_connection(('127.0.0.1', port)); s.sendall(b'GET /health HTTP/1.1\r\n\r\n')
    r = b''
    while True:
        c = s.recv(65536)
        if not c: break
        r += c
    return json.loads(r.split(b'\r\n\r\n', 1)[1])
live = []
for _ in range(2):
    s = socket.create_connection(('127.0.0.1', port)); s.sendall(req(4000, True)); live.append(s)
time.sleep(0.5)
for _ in range(2):            # ghosts: queued, then they leave
    g = socket.create_connection(('127.0.0.1', port)); g.sendall(req(8, False)); time.sleep(0.1); g.close()
time.sleep(1.0)
h = health()
print('queued %d live %d cancelled %d' % (h['queued'], h['live'], h['slot_cancelled']))
c = socket.create_connection(('127.0.0.1', port)); c.sendall(req(4, False)); c.settimeout(2.0)
try:
    first = c.recv(4096).split(b'\r\n')[0].decode()
except socket.timeout:
    first = 'waiting'          # queued behind the 2 live streams: right
print('fifth', first)
for s in live: s.close()
c.settimeout(60)
r = b''
try:
    while True:
        k = c.recv(65536)
        if not k: break
        r += k
except Exception:
    pass
print('fifth_final', r.split(b'\r\n')[0].decode() if r else first)
PY
stop
Q=$(awk '/^queued/ {print $2}' "$TMP/ghost.out"); GC=$(awk '/^queued/ {print $6}' "$TMP/ghost.out")
[ "${Q:-9}" -eq 0 ] && [ "${GC:-0}" -ge 2 ] && ok "2 clients that left while queued were taken out of the queue (queued $Q, cancelled $GC)" \
    || bad "queued clients that left are taken out of the queue" "$(cat "$TMP/ghost.out")"
grep -q "^fifth waiting" "$TMP/ghost.out" && ok "a live client after them is queued, not refused" \
    || bad "a live client after the ghosts is not refused" "$(grep fifth "$TMP/ghost.out")"
grep -q "^fifth_final HTTP/1.1 200" "$TMP/ghost.out" && ok "... and served once a slot frees" \
    || bad "the live client is served" "$(grep fifth "$TMP/ghost.out")"
fi

if want 10; then
# ── 10. the context ends an answer as "length", in both modes ────────────────
# Review B3: in --slots mode a request whose max_tokens ran past the context
# failed its step at the full cache and got a 500 with its text lost; neither
# mode ever said finish_reason "length" (not for max_tokens either).
for MODE in 1 2; do
    start "$TMP/ctx$MODE.log" --slots $MODE --ctx 96
    python3 - "$PORT" > "$TMP/ctx$MODE.out" 2>&1 <<'PY'
import json, socket, sys
port = int(sys.argv[1])
def post(n, stream):
    b = json.dumps({'messages': [{'role': 'user', 'content': 'Hello there'}], 'max_tokens': n,
                    'stream': stream, 'temperature': 0}).encode()
    s = socket.create_connection(('127.0.0.1', port)); s.settimeout(120)
    s.sendall(b'POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n' % len(b) + b)
    r = b''
    while True:
        c = s.recv(65536)
        if not c: break
        r += c
    return r
r = post(400, False)
st = r.split(b'\r\n')[0].decode()
try:
    d = json.loads(r.split(b'\r\n\r\n', 1)[1].decode('utf-8', 'replace'))
    u = d['usage']
    print('json', st, d['choices'][0]['finish_reason'], u['prompt_tokens'] + u['completion_tokens'],
          len(d['choices'][0]['message']['content']) > 0)
except Exception as e:
    print('json', st, 'unparsable', r[-160:])
r = post(400, True)
fr = [json.loads(l[6:])['choices'][0]['finish_reason'] for l in r.decode('utf-8', 'replace').splitlines()
      if l.startswith('data: {') and '"choices"' in l]
print('stream', 'error' if b'"error"' in r else 'noerror', fr[-1] if fr else None, r.endswith(b'data: [DONE]\n\n'))
d = json.loads(post(3, False).split(b'\r\n\r\n', 1)[1].decode('utf-8', 'replace'))
print('short', d['choices'][0]['finish_reason'], d['usage']['completion_tokens'])
PY
    stop
    J=$(grep '^json' "$TMP/ctx$MODE.out")
    [ "$J" = "json HTTP/1.1 200 OK length 97 True" ] \
        && ok "--slots $MODE: max_tokens past a 96-token context: 200, finish_reason length, every position used" \
        || bad "--slots $MODE: a full context is a length stop" "$J (want: 200 length 97 True)"
    S2=$(grep '^stream' "$TMP/ctx$MODE.out")
    [ "$S2" = "stream noerror length True" ] && ok "--slots $MODE: ... streamed too (final frame says length)" \
        || bad "--slots $MODE: a full context ends a stream with length" "$S2"
    SH=$(grep '^short' "$TMP/ctx$MODE.out")
    [ "$SH" = "short length 3" ] && ok "--slots $MODE: max_tokens 3 says length" \
        || bad "--slots $MODE: max_tokens ends as length" "$SH"
done
fi

if want 11; then
# ── 11. a client that half-closes after its request still gets its answer ─────
# Review R2: POLLRDHUP / a read EOF counted as "gone", so a client that sends
# its request and then shutdown(SHUT_WR)s (nc -N, socat, some proxies) got
# NOTHING, in both modes. EOF alone is not gone now; a write resolves it.
for MODE in 1 2; do
    start "$TMP/half$MODE.log" --slots $MODE
    python3 - "$PORT" > "$TMP/half$MODE.out" 2>&1 <<'PY'
import json, socket, sys
port = int(sys.argv[1])
for stream in (False, True):
    b = json.dumps({'messages': [{'role': 'user', 'content': 'Hello there'}], 'max_tokens': 24,
                    'stream': stream, 'temperature': 0}).encode()
    s = socket.create_connection(('127.0.0.1', port)); s.settimeout(60)
    s.sendall(b'POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n' % len(b) + b)
    s.shutdown(socket.SHUT_WR)
    r = b''
    while True:
        c = s.recv(65536)
        if not c: break
        r += c
    st = r.split(b'\r\n')[0].decode('latin-1') or 'EMPTY'
    body = r.split(b'\r\n\r\n', 1)[1] if b'\r\n\r\n' in r else b''
    if stream:
        good = body.endswith(b'data: [DONE]\n\n') and b'"finish_reason":"length"' in body
    else:
        try:
            d = json.loads(body.decode('utf-8', 'replace'))
            good = d['usage']['completion_tokens'] == 24
        except Exception:
            good = False
    print('stream' if stream else 'json', st, 'complete' if good else 'INCOMPLETE', len(r))
PY
    stop
    for K in json stream; do
        L=$(grep "^$K " "$TMP/half$MODE.out")
        echo "$L" | grep -q "200 OK complete" && ok "--slots $MODE: a half-closed $K client gets the whole answer" \
            || bad "--slots $MODE: a half-closed $K client gets the whole answer" "$L"
    done
done
fi

if want 12; then
# ── 12. slowloris: a request has an absolute read deadline ───────────────────
# Review R3: SO_RCVTIMEO bounds one recv(), not a request. --max-conns
# clients dripping a byte every second held every connection forever and
# every legitimate client got 503. Now the headers have --header-timeout-ms
# from accept (here 1500 ms) and the dripper gets 408.
start "$TMP/loris.log" --slots 2 --max-conns 4 --header-timeout-ms 1500
python3 - "$PORT" > "$TMP/loris.out" 2>&1 <<'PY'
import json, socket, sys, threading, time
port = int(sys.argv[1])
stop = False
got = []
def drip():
    s = socket.create_connection(('127.0.0.1', port))
    hdr = b'POST /v1/chat/completions HTTP/1.1\r\nX-Pad: '
    i = 0
    try:
        while not stop:
            s.sendall(hdr[i:i + 1] if i < len(hdr) else b'a'); i += 1; time.sleep(0.5)
            s.setblocking(False)
            try:
                d = s.recv(4096)
                if d: got.append(d.split(b'\r\n')[0].decode()); return
            except BlockingIOError:
                pass
            s.setblocking(True)
    except OSError:
        pass
ts = [threading.Thread(target=drip, daemon=True) for _ in range(4)]
[t.start() for t in ts]
b = json.dumps({'messages': [{'role': 'user', 'content': 'hi'}], 'max_tokens': 4}).encode()
req = b'POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n' % len(b) + b
def legit():
    s = socket.create_connection(('127.0.0.1', port)); s.settimeout(30); s.sendall(req)
    r = b''
    while True:
        c = s.recv(65536)
        if not c: break
        r += c
    return r.split(b'\r\n')[0].decode()
time.sleep(0.5)
print('early', legit())          # 4 drippers hold every connection: 503 is right
time.sleep(3.0)
print('later', legit())
stop = True
time.sleep(0.6)
print('drippers', ','.join(sorted(set(got))) or 'none')
PY
stop
L=$(grep '^later' "$TMP/loris.out")
echo "$L" | grep -q " 200 " && ok "4 slowloris connections on --max-conns 4 lock clients out for the header deadline only ($L)" \
    || bad "slowloris does not lock legitimate clients out" "$(cat "$TMP/loris.out")"
grep -q "^drippers.*408" "$TMP/loris.out" && ok "... and the drippers got 408" \
    || bad "a request past its header deadline gets 408" "$(grep drippers "$TMP/loris.out")"
fi

if want 13; then
# ── 13. a stream client that stops reading is cancelled, in bounded time ─────
# Review R4: a client that never reads (receive buffer 4 KiB) was still
# being generated for after 80 s: the autotuned send buffer absorbed the
# stream, so neither the 1 MiB pending cap nor the send timeout ever fired.
# Now: the acknowledged byte count must move within --send-timeout-ms
# (here 2000) while bytes are pending, and a stream's SO_SNDBUF is capped.
# Bound: fill the client's window (well under a second here) + 2 s.
for MODE in 1 2; do
    start "$TMP/slowr$MODE.log" --slots $MODE --send-timeout-ms 2000
    python3 - "$PORT" > "$TMP/slowr$MODE.out" 2>&1 <<'PY'
import json, socket, sys, time
port = int(sys.argv[1])
def health():
    s = socket.create_connection(('127.0.0.1', port)); s.sendall(b'GET /health HTTP/1.1\r\n\r\n')
    r = b''
    while True:
        c = s.recv(65536)
        if not c: break
        r += c
    return json.loads(r.split(b'\r\n\r\n', 1)[1])
b = json.dumps({'messages': [{'role': 'user', 'content': 'hi'}], 'max_tokens': 6000,
                'stream': True, 'temperature': 0}).encode()
s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4096)
s.connect(('127.0.0.1', port))
s.sendall(b'POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n' % len(b) + b)
t0 = time.time(); took = None
while time.time() - t0 < 20:
    time.sleep(0.25)
    h = health()
    if h.get('slot_cancelled', h['cancelled_running']) >= 1:
        took = time.time() - t0; break
print('cancelled_after', '%.1f' % took if took is not None else 'never')
s.close()
PY
    stop
    T=$(awk '/^cancelled_after/ {print $2}' "$TMP/slowr$MODE.out")
    [ "$T" != never ] && python3 -c "import sys; sys.exit(0 if float('$T') < 10 else 1)" \
        && ok "--slots $MODE: a stream client that stopped reading was cancelled after ${T}s" \
        || bad "--slots $MODE: a stream client that stopped reading is cancelled within 10 s" "$(cat "$TMP/slowr$MODE.out")"
done
fi

if want 14; then
# ── 14. one KV budget for every slot ──────────────────────────────────────────
# Review R6: each slot's cache was sized per request and never shrank, with
# no bound on the sum: N slots could each keep their largest request's cache
# for good. The fixture's cache is 8 KiB a position (4 layers x 4 KV heads x
# 128 x 2 x bf16), so --kv-budget-mb 2 is 256 positions:
#   - a request needing 433 positions (25 + 400 + 8) can never fit: 503
#   - two needing 183 each (25 + 150 + 8, 1.43 MiB) do not fit together:
#     the second waits first in line and runs when the first is done
#   - a 1.43 MiB cache is over the 1 MiB fair share (budget / 2 slots) and
#     is given back when its request ends
start "$TMP/kv.log" --slots 2 --kv-budget-mb 2
python3 - "$PORT" > "$TMP/kv.out" 2>&1 <<'PY'
import json, socket, sys, threading, time
port = int(sys.argv[1])
def health():
    s = socket.create_connection(('127.0.0.1', port)); s.sendall(b'GET /health HTTP/1.1\r\n\r\n')
    r = b''
    while True:
        c = s.recv(65536)
        if not c: break
        r += c
    return json.loads(r.split(b'\r\n\r\n', 1)[1])
def post(n, out, key):
    b = json.dumps({'messages': [{'role': 'user', 'content': 'hi'}], 'max_tokens': n, 'temperature': 0}).encode()
    s = socket.create_connection(('127.0.0.1', port)); s.settimeout(120)
    s.sendall(b'POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: %d\r\n\r\n' % len(b) + b)
    r = b''
    while True:
        c = s.recv(65536)
        if not c: break
        r += c
    out[key] = r.split(b'\r\n')[0].decode()
out = {}
post(400, out, 'huge')
print('huge', out['huge'])
ta = threading.Thread(target=post, args=(150, out, 'a')); tb = threading.Thread(target=post, args=(150, out, 'b'))
ta.start(); time.sleep(0.3); tb.start()
max_live = 0; max_kv = 0; budget = None; held = 0
while ta.is_alive() or tb.is_alive():
    h = health()
    budget = h.get('kv_budget_bytes'); max_kv = max(max_kv, h.get('kv_bytes', 0))
    max_live = max(max_live, h['live'])
    if h['live'] == 1 and h['queued'] == 1: held = 1
    time.sleep(0.05)
ta.join(); tb.join()
time.sleep(0.2)
h = health()
print('pair', out['a'], '|', out['b'], 'max_live', max_live, 'waited', held)
print('kv max', max_kv, 'budget', budget, 'after', h.get('kv_bytes'))
PY
stop
grep -q "^huge HTTP/1.1 503" "$TMP/kv.out" && ok "a request whose KV could never fit the budget gets 503" \
    || bad "a request past the KV budget is refused" "$(grep huge "$TMP/kv.out")"
grep -q "^pair HTTP/1.1 200 OK | HTTP/1.1 200 OK max_live 1 waited 1" "$TMP/kv.out" \
    && ok "two that do not fit together run one after the other, both 200" \
    || bad "the KV budget serializes requests that do not fit together" "$(grep pair "$TMP/kv.out")"
python3 - "$TMP/kv.out" <<'PY' && ok "the slots never held more KV than the budget, and gave the peak back ($(grep '^kv' "$TMP/kv.out"))" \
    || bad "KV held stays within the budget and shrinks after" "$(grep '^kv' "$TMP/kv.out")"
import sys
f = [l.split() for l in open(sys.argv[1]) if l.startswith('kv')][0]
mx, budget, after = int(f[2]), f[4], f[6]
sys.exit(0 if budget != 'None' and 0 < mx <= int(budget) and after != 'None' and int(after) <= int(budget) // 2 else 1)
PY
fi

if want 15; then
# ── 15. request validation, the same in both modes (review NITs) ─────────────
for MODE in 1 2; do
    start "$TMP/val$MODE.log" --slots $MODE --ctx 96
    python3 - "$PORT" > "$TMP/val$MODE.out" 2>&1 <<'PY'
import json, socket, sys, time
port = int(sys.argv[1])
def raw(data, half=False, timeout=8):
    s = socket.create_connection(('127.0.0.1', port)); s.settimeout(timeout); s.sendall(data)
    if half: s.shutdown(socket.SHUT_WR)
    r = b''
    try:
        while True:
            c = s.recv(65536)
            if not c: break
            r += c
    except socket.timeout:
        return 'TIMEOUT'
    return r.split(b'\r\n')[0].decode() + ' ' + (r.split(b'\r\n\r\n', 1)[1][:160].decode('utf-8', 'replace') if b'\r\n\r\n' in r else '')
def req(body, extra=b''):
    return (b'POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\n' + extra +
            b'Content-Type: application/json\r\nContent-Length: %d\r\n\r\n' % len(body) + body)
def chat(**kw):
    d = {'messages': [{'role': 'user', 'content': kw.pop('content', 'hi')}], 'temperature': 0}
    d.update(kw)
    return json.dumps(d).encode()
for name, v in (('negative', -5), ('huge', 1e30), ('fractional', 2.5), ('string', 'many')):
    body = json.loads(chat()); body['max_tokens'] = v
    print('max_tokens', name, raw(req(json.dumps(body).encode())))
print('long_prompt', raw(req(chat(content='word ' * 60, max_tokens=4))))
# "X-Content-Length" must not be read as Content-Length
print('xcl', raw(req(chat(max_tokens=2), b'X-Content-Length: 999\r\n')))
# a body cut short (Content-Length 40 bytes too long, then half-close):
# the prefix that did arrive is a complete JSON document, and must not be served
b = chat(max_tokens=2)
print('truncated', raw(b'POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Length: %d\r\n\r\n' % (len(b) + 40) + b, half=True))
PY
    stop
    for V in negative huge fractional string; do
        L=$(grep "^max_tokens $V " "$TMP/val$MODE.out")
        echo "$L" | grep -q "400 Bad Request" && ok "--slots $MODE: max_tokens $V -> 400" \
            || bad "--slots $MODE: max_tokens $V is refused with 400" "$L"
    done
    L=$(grep "^long_prompt" "$TMP/val$MODE.out")
    echo "$L" | grep -q "400 Bad Request.*does not fit" && ok "--slots $MODE: a prompt longer than the context -> 400" \
        || bad "--slots $MODE: a prompt longer than the context is 400" "$L"
    L=$(grep "^xcl" "$TMP/val$MODE.out")
    echo "$L" | grep -q "200 OK" && ok "--slots $MODE: X-Content-Length is not Content-Length" \
        || bad "--slots $MODE: Content-Length is parsed as a header name" "$L"
    L=$(grep "^truncated" "$TMP/val$MODE.out")
    echo "$L" | grep -q "400 Bad Request" && ok "--slots $MODE: a truncated body -> 400, not processed" \
        || bad "--slots $MODE: a truncated body is 400" "$L"
done
# A 5xx is the server's problem: its type is not invalid_request_error.
start "$TMP/val503.log" --slots 2 --kv-budget-mb 1
R=$(curl -s -X POST "localhost:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
    -d '{"messages":[{"role":"user","content":"hi"}],"max_tokens":400}' -w ' %{http_code}')
stop
echo "$R" | grep -q ' 503$' && ! echo "$R" | grep -q invalid_request_error \
    && ok "a 503 says service_unavailable, not invalid_request_error" \
    || bad "5xx error types" "$R"
fi

if want 16; then
# ── 16. /health numbers that mean what they say; no allocation per token ─────
# Review NITs: aggregate_decode_tok_s divided the tokens of the last 10 s by
# "now - the oldest step", so an idle server reported a fraction of what it
# had just done; and the per-request output buffers grew by realloc inside
# the token loop (AGENTS.md: zero allocation there).
for MODE in 1 2; do
    start "$TMP/agg$MODE.log" --slots $MODE
    python3 - "$PORT" > "$TMP/agg$MODE.out" 2>&1 <<'PY'
import json, socket, sys, time
port = int(sys.argv[1])
def get(data):
    s = socket.create_connection(('127.0.0.1', port)); s.settimeout(120); s.sendall(data)
    r = b''
    while True:
        c = s.recv(65536)
        if not c: break
        r += c
    return r
def health():
    return json.loads(get(b'GET /health HTTP/1.1\r\n\r\n').split(b'\r\n\r\n', 1)[1])
def chat(n, stream):
    b = json.dumps({'messages': [{'role': 'user', 'content': 'hi'}], 'max_tokens': n,
                    'stream': stream, 'temperature': 0}).encode()
    return get(b'POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Length: %d\r\n\r\n' % len(b) + b)
r = chat(120, False)
u = json.loads(r.split(b'\r\n\r\n', 1)[1].decode('utf-8', 'replace'))['usage']
chat(120, True)
time.sleep(2.0)                     # idle: the rate of what was done must not decay
h = health()
print('rate', u['decode_tok_s'], h.get('aggregate_decode_tok_s', -1))
print('allocs', h.get('token_loop_allocs'))
PY
    stop
    if [ "$MODE" = 2 ]; then
        R=$(grep '^rate' "$TMP/agg$MODE.out")
        python3 -c "import sys; d, a = map(float, '$R'.split()[1:]); sys.exit(0 if a >= 0.5 * d else 1)" \
            && ok "--slots 2: aggregate decode t/s after 2 s idle still reflects the steps ($R)" \
            || bad "--slots 2: aggregate_decode_tok_s is tokens over step time" "$R (stream decode_tok_s, aggregate)"
    fi
    A=$(awk '/^allocs/ {print $2}' "$TMP/agg$MODE.out")
    [ "$A" = 0 ] && ok "--slots $MODE: no output buffer grew inside the token loop (240 tokens, stream and not)" \
        || bad "--slots $MODE: no allocation in the token loop" "token_loop_allocs=$A"
done
fi

if grep -l "Sanitizer" "$TMP"/*.log >/dev/null 2>&1; then
    bad "no sanitizer report in any server log" "$(grep -h -A3 Sanitizer "$TMP"/*.log | head -12)"
fi
echo
[ $fail -eq 0 ] && echo "PASS" || { echo "FAILED ($fail)"; tail -n 20 "$TMP"/*.log; }
exit $fail
