#!/bin/bash
# test_device.sh — `mynah-slm run --device`, end to end, model-free.
#
#   tests/test_device.sh [path/to/mynah-slm] [path/to/write_fixture]
#
# 1. --device cpu-backend (the CPU kernels driven through the backend
#    forward, src/forward_backend.c) prints byte-for-byte what the default
#    path prints, greedy and sampled, on the untied Q4_K_M-mix "slow"
#    fixture, and its speed line names the device; the default line does not.
# 2. --device cuda is REFUSED, exit 1, nothing on stdout, with the reason:
#    "not compiled" in a CPU build, "no CUDA device" in a `make cuda` build on
#    a machine without one. Never a quiet CPU run (engineering-method: an
#    explicit request is not a capability). Skipped on a machine that HAS a
#    device — there it is the device's own parity gate's job.
# 3. An unknown device is a usage error (exit 2).
#
# SPDX-License-Identifier: MIT
set -u
BIN=${1:-./mynah-slm}
WF=${2:-tests/write_fixture}
TMP=$(mktemp -d "${TMPDIR:-/tmp}/mynah_slm_device_XXXXXX") || exit 1
trap 'rm -rf "$TMP"' EXIT
fails=0
ok()   { echo "ok   $*"; }
bad()  { echo "FAIL $*"; fails=$((fails + 1)); }

PROMPT="The ledger for 1887 lists twelve names."
# The "slow" fixture: untied head, so the answer depends on the prompt (the
# tied tiny one echoes its last token forever and would compare nothing).
"$WF" slow "$TMP/slow.gguf" || { echo "FAIL write_fixture"; exit 1; }
for mode in greedy sampled; do
    if [ $mode = greedy ]; then samp="--temp 0"; else samp="--temp 0.8 --seed 3"; fi
    for dev in cpu cpu-backend; do
        # shellcheck disable=SC2086
        "$BIN" run -m "$TMP/slow.gguf" -p "$PROMPT" --raw $samp -n 48 --device "$dev" \
            > "$TMP/$mode.$dev.out" 2> "$TMP/$mode.$dev.err"
        rc=$?
        [ $rc -eq 0 ] || bad "$mode --device $dev exits $rc: $(cat "$TMP/$mode.$dev.err")"
    done
    if cmp -s "$TMP/$mode.cpu.out" "$TMP/$mode.cpu-backend.out" &&
       [ "$(wc -c < "$TMP/$mode.cpu.out")" -gt 40 ]; then
        ok "$mode: --device cpu-backend stdout == default path ($(wc -c < "$TMP/$mode.cpu.out") bytes)"
    else
        bad "$mode: --device cpu-backend stdout differs from the default path"
    fi
    if grep -q "device cpu-backend\]" "$TMP/$mode.cpu-backend.err" &&
       ! grep -q "device" "$TMP/$mode.cpu.err"; then
        ok "$mode: the speed line names the device (and only when one was asked for)"
    else
        bad "$mode: speed line: $(tail -1 "$TMP/$mode.cpu-backend.err") / $(tail -1 "$TMP/$mode.cpu.err")"
    fi
done

"$BIN" run -m "$TMP/slow.gguf" -p "$PROMPT" --raw -n 4 --device cuda > "$TMP/cuda.out" 2> "$TMP/cuda.err"
rc=$?
msg=$(cat "$TMP/cuda.err")
if [ $rc -eq 0 ]; then
    echo "SKIP --device cuda ran: this machine has a device (gate it with make cuda-test)"
elif [ $rc -eq 1 ] && [ ! -s "$TMP/cuda.out" ] &&
     echo "$msg" | grep -qE "not compiled|no CUDA device"; then
    ok "--device cuda refused cleanly: $msg"
else
    bad "--device cuda: exit $rc, stdout $(wc -c < "$TMP/cuda.out") bytes, stderr: $msg"
fi

"$BIN" run -m "$TMP/slow.gguf" -p "$PROMPT" --device tpu > /dev/null 2>&1
[ $? -eq 2 ] && ok "an unknown device is a usage error" || bad "--device tpu was not exit 2"

[ $fails -eq 0 ] && echo "PASS (device)" || echo "FAILED (device: $fails)"
exit $((fails > 0))
