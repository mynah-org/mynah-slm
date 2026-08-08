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
  "granite-350m-q4|ibm-granite/granite-4.0-350m-GGUF|granite-4.0-350m-Q4_K_M.gguf|226 MB|Granite 4.0 350m, the small/fast rung"
  "granite-350m-q8|ibm-granite/granite-4.0-350m-GGUF|granite-4.0-350m-Q8_0.gguf|361 MB|Granite 4.0 350m — the rung to USE: 27/30 tool calls, and the fastest of the ladder"
  # The middle of the quantization ladder measured in docs/models.md. These four
  # were fetched by hand and lived only in models-local/, which use_model.sh
  # wipes on every stage — so one staging deleted the lot. The measurements
  # survived in the docs; the files did not. Anything worth measuring twice
  # belongs in this list.
  "granite-350m-q2|ibm-granite/granite-4.0-350m-GGUF|granite-4.0-350m-Q2_K.gguf|172 MB|the DESTROYED rung: ppl 26400, word salad. Kept so the result stays reproducible"
  "granite-350m-q3|ibm-granite/granite-4.0-350m-GGUF|granite-4.0-350m-Q3_K_M.gguf|199 MB|survives, but costs 8% of bits/byte to save 27 MB — a poor trade at this size"
  "granite-350m-q5|ibm-granite/granite-4.0-350m-GGUF|granite-4.0-350m-Q5_K_M.gguf|252 MB|ladder rung"
  "granite-350m-q6|ibm-granite/granite-4.0-350m-GGUF|granite-4.0-350m-Q6_K.gguf|279 MB|ladder rung"
  # LFM2.5: TRIED AND REJECTED, 2026-08-08 (docs/models.md). It loses 12 of 16
  # languages to Qwen3-0.6B at 4.4x the file, Italian by 18.5%. The engine
  # supports it and these stay downloadable so the result can be re-checked —
  # that is not the same as recommending it.
  # It is also NOT Apache/MIT: LFM Open License v1.0, commercial use only under
  # $10M annual revenue, so it could never have been a bundled default even had
  # it won. See the LICENSE file in the same HF repo.
  "lfm2.5-2.6b-q4|LiquidAI/LFM2.5-2.6B-GGUF|LFM2.5-2.6B-Q4_K_M.gguf|1.67 GB|REJECTED candidate, non-free license — 22 short-conv + 8 GQA layers; wins only on en/ja"
  "lfm2.5-2.6b-q8|LiquidAI/LFM2.5-2.6B-GGUF|LFM2.5-2.6B-Q8_0.gguf|2.87 GB|the same at Q8_0. Never downloaded: the Q4 answer closed the trial"
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
