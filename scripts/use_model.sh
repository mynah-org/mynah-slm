#!/usr/bin/env bash
# Stage ONE checkpoint from the NAS onto local disk, for benchmarking.
#
# The NAS (models/) is the library: every checkpoint lives there permanently,
# because the internal disk is small. But it is the wrong place to MEASURE
# from, and not only because loading is slower:
#
#   full weight read + dequant, Qwen3-0.6B-Q4_K_M, warm page cache
#     NAS    17.0 s   (~22 MiB/s)
#     local   7.5 s   (~50 MiB/s)
#
#   Dequantization is the same CPU work on both sides, so the I/O portion of
#   that gap is roughly 0.5 s against 10 s — about 20x.
#
# It matters beyond load time. This engine multiplies straight off the mmap'd
# quantized bytes (that is the point of ingot), so the weights are walked again
# on every decode step. Warm, they come from the page cache and the NAS costs
# nothing; but the moment anything evicts those pages — a second model, the ASR
# and TTS stages running beside us, a large prompt — the re-faults come back
# over SMB, at random, mid-measurement. A tok/s number you cannot reproduce is
# not a measurement.
#
# So: benchmark and quality-run from local, keep the library on the NAS, and
# hold exactly one model here at a time.
#
# Usage:
#   scripts/use_model.sh --status
#   scripts/use_model.sh Qwen3-0.6B-Q4_K_M.gguf
#   scripts/use_model.sh --evict
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
NAS="$ROOT/models"
LOCAL="$ROOT/models-local"
MIN_FREE_GB=5

die() { printf 'use_model.sh: %s\n' "$1" >&2; exit 1; }

free_gb() { df -g "$ROOT" | awk 'NR==2 {print $4}'; }

status() {
    printf 'local  %s\n' "$LOCAL"
    if [ -d "$LOCAL" ] && [ -n "$(ls -A "$LOCAL" 2>/dev/null)" ]; then
        ls -lh "$LOCAL" | awk 'NR>1 {printf "         %-44s %s\n", $9, $5}'
    else
        printf '         (empty)\n'
    fi
    printf '         %s GB free on the internal disk\n\n' "$(free_gb)"

    printf 'NAS    %s\n' "$NAS"
    if ls "$NAS"/*.gguf >/dev/null 2>&1; then
        ls -lh "$NAS"/*.gguf | awk '{printf "         %-44s %s\n", $9, $5}' | sed "s|$NAS/||"
    else
        printf '         (not mounted, or empty — see CLAUDE.md)\n'
    fi
}

case "${1:---status}" in
  --status|-s) status; exit 0 ;;
  --evict|-e)  rm -rf "$LOCAL"; echo "evicted the local cache"; exit 0 ;;
  -h|--help)   sed -n '2,30p' "$0"; exit 0 ;;
esac

NAME="$(basename "$1")"
SRC="$NAS/$NAME"

[ -d "$NAS" ] || die "$NAS is not readable — is the NAS mounted? (see CLAUDE.md)"
[ -f "$SRC" ] || die "no such checkpoint on the NAS: $NAME (try --status)"

if [ -f "$LOCAL/$NAME" ]; then
    echo "already local: $LOCAL/$NAME"
    exit 0
fi

# One at a time. The internal disk has ~19 GB free and a Gemma 4 E2B is 1.5 GB;
# the limit is not really space, it is that two staged models make it ambiguous
# which one a benchmark actually used.
rm -rf "$LOCAL"
mkdir -p "$LOCAL"

need_gb=$(( ($(wc -c < "$SRC") / 1073741824) + 1 ))
have_gb=$(free_gb)
[ "$have_gb" -gt $((need_gb + MIN_FREE_GB)) ] || \
    die "only ${have_gb} GB free, need ~${need_gb} GB plus a ${MIN_FREE_GB} GB margin"

echo "staging $NAME from the NAS ..."
time cp "$SRC" "$LOCAL/$NAME"

echo
echo "-> $LOCAL/$NAME"
echo "   make bench MODEL=$LOCAL/$NAME"
echo "   run 'scripts/use_model.sh --evict' when done"
