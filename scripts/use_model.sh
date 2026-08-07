#!/usr/bin/env bash
# Stage ONE checkpoint from models/ onto local disk, for benchmarking.
#
# Skip this if models/ is already a plain local directory — then it is doing
# nothing for you. It exists because models/ is often not local: a network
# share, an external drive, a symlink to somewhere with room for a checkpoint
# library. That is a fine place to KEEP weights and the wrong place to MEASURE
# from, and not only because loading is slower:
#
#   full weight read + dequant, Qwen3-0.6B-Q4_K_M, warm page cache
#     over SMB   17.0 s   (~22 MiB/s)
#     local       7.5 s   (~50 MiB/s)
#
#   Dequantization is the same CPU work on both sides, so the I/O portion of
#   that gap is roughly 0.5 s against 10 s — about 20x.
#
# It matters beyond load time. This engine multiplies straight off the mmap'd
# quantized bytes (that is the point of ingot), so the weights are walked again
# on every decode step. Warm, they come from the page cache and remote storage
# costs nothing; but the moment anything evicts those pages — a second model,
# another process, a large prompt — the re-faults go back over the network, at
# random, mid-measurement. A tok/s number you cannot reproduce is not a
# measurement.
#
# So: benchmark from local, keep the library wherever it fits, and hold exactly
# one model here at a time.
#
# Usage:
#   scripts/use_model.sh --status
#   scripts/use_model.sh Qwen3-0.6B-Q4_K_M.gguf
#   scripts/use_model.sh --evict
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LIB="$ROOT/models"
LOCAL="$ROOT/models-local"
MIN_FREE_GB=5

die() { printf 'use_model.sh: %s\n' "$1" >&2; exit 1; }

free_gb() { df -g "$ROOT" | awk 'NR==2 {print $4}'; }

status() {
    printf 'staged  %s\n' "$LOCAL"
    if [ -d "$LOCAL" ] && [ -n "$(ls -A "$LOCAL" 2>/dev/null)" ]; then
        ls -lh "$LOCAL" | awk 'NR>1 {printf "          %-44s %s\n", $9, $5}'
    else
        printf '          (empty)\n'
    fi
    printf '          %s GB free on the local disk\n\n' "$(free_gb)"

    printf 'library %s\n' "$LIB"
    if ls "$LIB"/*.gguf >/dev/null 2>&1; then
        ls -lh "$LIB"/*.gguf | awk '{printf "          %-44s %s\n", $9, $5}' | sed "s|$LIB/||"
    else
        printf '          (empty, or its storage is not mounted)\n'
    fi
}

case "${1:---status}" in
  --status|-s) status; exit 0 ;;
  --evict|-e)  rm -rf "$LOCAL"; echo "evicted the staged model"; exit 0 ;;
  -h|--help)   sed -n '2,31p' "$0"; exit 0 ;;
esac

NAME="$(basename "$1")"
SRC="$LIB/$NAME"

[ -d "$LIB" ] || die "$LIB is not readable — is its storage mounted?"
[ -f "$SRC" ] || die "no such checkpoint in $LIB: $NAME (try --status)"

if [ -f "$LOCAL/$NAME" ]; then
    echo "already staged: $LOCAL/$NAME"
    exit 0
fi

# One at a time, and the reason is not disk space: two staged models make it
# ambiguous which one a benchmark actually read.
rm -rf "$LOCAL"
mkdir -p "$LOCAL"

need_gb=$(( ($(wc -c < "$SRC") / 1073741824) + 1 ))
have_gb=$(free_gb)
[ "$have_gb" -gt $((need_gb + MIN_FREE_GB)) ] || \
    die "only ${have_gb} GB free, need ~${need_gb} GB plus a ${MIN_FREE_GB} GB margin"

echo "staging $NAME ..."
time cp "$SRC" "$LOCAL/$NAME"

echo
echo "-> $LOCAL/$NAME"
echo "   make bench MODEL=$LOCAL/$NAME"
echo "   run 'scripts/use_model.sh --evict' when done"
