#!/usr/bin/env bash
# Download a supported checkpoint from HuggingFace into models/.
#
# Public links only: no account, no token, nothing to log into. The download
# resumes if it is interrupted, so re-running it after a dropped connection
# picks up where it stopped.
#
# Non-interactive by design: no prompts, no menu, safe from CI and from an agent.
#
# Usage:
#   scripts/download_model.sh --list
#   scripts/download_model.sh --model qwen3-0.6b-q4
#   scripts/download_model.sh --model qwen3-0.6b-q8 --dir /somewhere/else
set -euo pipefail

# key | HF repo | file | ~size | description
MODELS=(
  "qwen3-0.6b-q4|unsloth/Qwen3-0.6B-GGUF|Qwen3-0.6B-Q4_K_M.gguf|378 MB|v0.1 baseline, the default we ship"
  "qwen3-0.6b-q8|Qwen/Qwen3-0.6B-GGUF|Qwen3-0.6B-Q8_0.gguf|610 MB|v0.1 baseline, OFFICIAL Qwen build — least quant noise, use for oracle parity"
  "qwen3-1.7b-q4|unsloth/Qwen3-1.7B-GGUF|Qwen3-1.7B-Q4_K_M.gguf|1.1 GB|same architecture, scaled up"
  "qwen3-4b-q4|unsloth/Qwen3-4B-GGUF|Qwen3-4B-Q4_K_M.gguf|2.5 GB|same architecture, the quality option"
  # NB the odd filename: the repo is ...-it-qat-q4_0-gguf but the file inside is
  # gemma-4-E2B_q4_0-it.gguf. Verified against the HF API, do not "fix" it.
  # The repo also holds gemma-4-E2B-it-mmproj.gguf — the multimodal projector,
  # which we deliberately do not download: mynah-slm is the text tower only.
  "gemma4-e2b-qat|google/gemma-4-E2B-it-qat-q4_0-gguf|gemma-4-E2B_q4_0-it.gguf|1.5 GB|v0.2 PRODUCTION target — Q4_0 because that is what QAT ships"
)

DEST_DEFAULT="$(cd "$(dirname "$0")/.." && pwd)/models"
CHOICE=""
DEST=""

die() { printf 'download_model.sh: %s\n' "$1" >&2; exit 1; }

list() {
  printf '%-16s %-10s %s\n' KEY SIZE DESCRIPTION
  for m in "${MODELS[@]}"; do
    IFS='|' read -r key _repo _file size desc <<<"$m"
    printf '%-16s %-10s %s\n' "$key" "$size" "$desc"
  done
}

while [ $# -gt 0 ]; do
  case "$1" in
    --list)  list; exit 0 ;;
    --model) CHOICE="${2:-}"; shift 2 ;;
    --dir)   DEST="${2:-}";   shift 2 ;;
    -h|--help) sed -n '2,10p' "$0"; exit 0 ;;
    *) die "unknown argument '$1' (try --list)" ;;
  esac
done

[ -n "$CHOICE" ] || die "nothing to do: pass --model KEY (see --list)"
DEST="${DEST:-$DEST_DEFAULT}"

for m in "${MODELS[@]}"; do
  IFS='|' read -r key repo file size desc <<<"$m"
  [ "$key" = "$CHOICE" ] || continue

  # Two different failures wear the same "not a directory" face here, and only
  # one of them is a problem:
  #   - the destination is a SYMLINK that does not resolve. Somebody points
  #     models/ at external storage — a network share, another drive — and it
  #     is not mounted right now. Say so, instead of quietly writing hundreds
  #     of MB to wherever the link used to go.
  #   - it simply does not exist, which is just a fresh clone. Create it.
  if [ ! -d "$DEST" ]; then
    if [ -L "$DEST" ]; then
      die "$DEST is a symlink that does not resolve — is its target mounted?"
    fi
    mkdir -p "$DEST" || die "cannot create $DEST"
  fi

  out="$DEST/$file"
  if [ -f "$out" ]; then
    printf 'already there: %s\n' "$out"
    exit 0
  fi

  printf 'downloading %s (%s) from %s\n' "$file" "$size" "$repo"
  # -C - resumes a partial file; --fail turns an HTML error page into exit != 0
  # rather than a corrupt .gguf that only fails much later at load time.
  curl -fL -C - --progress-bar \
    -o "$out.part" \
    "https://huggingface.co/$repo/resolve/main/$file"
  mv "$out.part" "$out"

  printf '\n-> %s\n' "$out"
  printf 'check it with: mynah-slm inspect %s\n' "$out"
  exit 0
done

die "unknown model '$CHOICE' (try --list)"
