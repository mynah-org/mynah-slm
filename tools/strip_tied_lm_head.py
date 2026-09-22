#!/usr/bin/env python3
"""Drop a redundant, byte-identical `lm_head.weight` from a safetensors checkpoint.

WHY THIS EXISTS. Qwen3-0.6B declares `tie_word_embeddings: true` and then ships
`lm_head.weight` AND `model.embed_tokens.weight` as two byte-identical copies of
the same 296.8 MiB tensor -- 1503.3 MB on disk for a 596M-parameter model. That
is upstream's choice, but llama.cpp's Qwen converter does not drop the tied copy
the way its deepseek/hunyuan/pangu converters do, so it becomes a second GGUF
tensor and every quant we build carries it:

    Qwen3-0.6B-Q4_K_M.gguf (community, tied)   378.3 MiB   token_embd Q6_K
    q-Q4_K_M.gguf          (ours,  untied)     461.8 MiB   token_embd Q4_K + output Q6_K
    q-Q3_K_M.gguf          (ours,  untied)     394.8 MiB   token_embd Q3_K + output Q6_K

Removing it is lossless ONLY if the two tensors really are identical, so that is
checked rather than assumed: this tool hashes both and REFUSES if they differ.
A tied model is not merely smaller -- llama-quantize then sees one tensor doing
both jobs and bumps it to Q6_K, so the embedding lookup gets *better* than the
Q4_K/Q3_K copy it had before.

SPDX-License-Identifier: MIT
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
import sys
from pathlib import Path


class Refusal(SystemExit):
    def __init__(self, msg: str) -> None:
        print(f"REFUSED: {msg}", file=sys.stderr)
        super().__init__(2)


def read_header(fp) -> tuple[dict, int]:
    (n,) = struct.unpack("<Q", fp.read(8))
    return json.loads(fp.read(n)), 8 + n


def digest(fp, base: int, entry: dict) -> tuple[str, int]:
    start, end = entry["data_offsets"]
    fp.seek(base + start)
    h = hashlib.sha256()
    left = end - start
    while left:
        chunk = fp.read(min(1 << 22, left))
        if not chunk:
            raise Refusal("short read: the file is truncated")
        h.update(chunk)
        left -= len(chunk)
    return h.hexdigest(), end - start


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path, help="HF checkpoint directory")
    ap.add_argument("dst", type=Path, help="output directory (created)")
    ap.add_argument("--drop", default="lm_head.weight")
    ap.add_argument("--keep", default="model.embed_tokens.weight")
    args = ap.parse_args()

    cfg_path = args.src / "config.json"
    if not cfg_path.is_file():
        raise Refusal(f"no config.json under {args.src}")
    cfg = json.loads(cfg_path.read_text())
    if not cfg.get("tie_word_embeddings"):
        raise Refusal("config.json does not say tie_word_embeddings: true — "
                      "the two tensors are then NOT interchangeable")

    shards = sorted(args.src.glob("*.safetensors"))
    if not shards:
        raise Refusal(f"no .safetensors under {args.src}")
    if len(shards) > 1:
        raise Refusal("sharded checkpoints are not handled; this tool is "
                      "deliberately small and only covers the single-file case")
    src = shards[0]

    with open(src, "rb") as f:
        hdr, base = read_header(f)
        if args.drop not in hdr:
            raise Refusal(f"{args.drop} is not in the checkpoint — nothing to do")
        if args.keep not in hdr:
            raise Refusal(f"{args.keep} is not in the checkpoint")
        if hdr[args.drop]["shape"] != hdr[args.keep]["shape"]:
            raise Refusal("the two tensors have different shapes")
        d_drop, n_drop = digest(f, base, hdr[args.drop])
        d_keep, _ = digest(f, base, hdr[args.keep])
        print(f"  {args.drop:30s} sha256 {d_drop[:24]}")
        print(f"  {args.keep:30s} sha256 {d_keep[:24]}")
        if d_drop != d_keep:
            raise Refusal("the two tensors are NOT byte-identical — dropping one "
                          "would change the model, so this tool will not do it")
        print(f"  identical: dropping {n_drop / 1048576:.1f} MiB is lossless")

        # Rewrite without the dropped tensor. Offsets are recomputed because
        # safetensors requires the data block to be contiguous from zero.
        args.dst.mkdir(parents=True, exist_ok=True)
        order = sorted((k for k in hdr if k != "__metadata__" and k != args.drop),
                       key=lambda k: hdr[k]["data_offsets"][0])
        new_hdr: dict = {}
        if "__metadata__" in hdr:
            new_hdr["__metadata__"] = hdr["__metadata__"]
        cursor = 0
        for k in order:
            s, e = hdr[k]["data_offsets"]
            new_hdr[k] = {"dtype": hdr[k]["dtype"], "shape": hdr[k]["shape"],
                          "data_offsets": [cursor, cursor + (e - s)]}
            cursor += e - s
        blob = json.dumps(new_hdr, separators=(",", ":")).encode("utf-8")
        pad = (-len(blob)) % 8            # the data block must stay 8-aligned
        blob += b" " * pad

        out = args.dst / src.name
        with open(out, "wb") as g:
            g.write(struct.pack("<Q", len(blob)))
            g.write(blob)
            for k in order:
                s, e = hdr[k]["data_offsets"]
                f.seek(base + s)
                left = e - s
                while left:
                    chunk = f.read(min(1 << 22, left))
                    g.write(chunk)
                    left -= len(chunk)

    for extra in args.src.iterdir():
        if extra.is_file() and extra.suffix != ".safetensors":
            shutil.copy2(extra, args.dst / extra.name)

    a, b = src.stat().st_size, out.stat().st_size
    print(f"  {src}  {a / 1e6:.1f} MB")
    print(f"  {out}  {b / 1e6:.1f} MB   (-{(a - b) / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
