"""CLI for the oracle: prompt in, text out, optionally activation dumps.

    uv run python -m oracle.generate ../models/Qwen3-0.6B-Q4_K_M.gguf \
        --prompt "Ciao, come stai?" -n 32
    uv run python -m oracle.generate <model.gguf> --prompt "..." \
        --dump-dir ../tests/golden/it_hello

The dumps are what tools/eval/compare.py checks the C engine against, stage by
stage, at per-stage tolerances.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
from tokenizers import Tokenizer

from .model import Qwen3Model
from .weights import Qwen3Weights

REFERENCE = os.path.join(os.path.dirname(__file__), "..", "..", "reference", "qwen3-0.6b")


def load_tokenizer(ref_dir: str) -> Tokenizer:
    return Tokenizer.from_file(os.path.join(ref_dir, "tokenizer.json"))


def eos_ids(ref_dir: str, cfg) -> set[int]:
    """The GGUF carries ONE terminator; generation_config.json carries two.
    Stopping only on the GGUF one runs past the end of a turn."""
    ids = set(cfg.eos_ids)
    path = os.path.join(ref_dir, "generation_config.json")
    if os.path.exists(path):
        gen = json.load(open(path))
        e = gen.get("eos_token_id", [])
        ids.update(e if isinstance(e, list) else [e])
    return ids


def chat_prompt(text: str, think: bool) -> str:
    """The Qwen3 ChatML turn, written out rather than rendered from the Jinja
    template in the GGUF. The C engine must produce this byte for byte."""
    s = f"<|im_start|>user\n{text}<|im_end|>\n<|im_start|>assistant\n"
    if not think:
        s += "<think>\n\n</think>\n\n"
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("-n", "--max-new", type=int, default=32)
    ap.add_argument("--raw", action="store_true", help="skip the chat template")
    ap.add_argument("--think", action="store_true", help="allow a thinking block")
    ap.add_argument("--dump-dir")
    ap.add_argument("--dump-layers", default="0,13,27")
    ap.add_argument("--reference", default=REFERENCE)
    args = ap.parse_args()

    t0 = time.time()
    weights = Qwen3Weights(args.model)
    model = Qwen3Model(weights)
    load_s = time.time() - t0
    print(f"# {weights.config}", file=sys.stderr)
    print(f"# load {load_s:.1f}s", file=sys.stderr)

    tok = load_tokenizer(args.reference)
    stop = eos_ids(args.reference, weights.config)

    text = args.prompt if args.raw else chat_prompt(args.prompt, args.think)
    ids = tok.encode(text, add_special_tokens=False).ids
    print(f"# prompt {len(ids)} tokens, stop on {sorted(stop)}", file=sys.stderr)

    if args.dump_dir:
        os.makedirs(args.dump_dir, exist_ok=True)
        dump = {"_layers": {int(x) for x in args.dump_layers.split(",") if x != ""}}
        model.forward(ids, dump=dump)
        for name, arr in dump.items():
            if name.startswith("_"):
                continue
            np.save(os.path.join(args.dump_dir, f"{name}.npy"), arr)
        json.dump(
            {"prompt": args.prompt, "raw": args.raw, "tokens": ids,
             "model": os.path.basename(args.model)},
            open(os.path.join(args.dump_dir, "meta.json"), "w"), indent=2,
        )
        # Plain one-id-per-line, so the C parity test can read the exact same
        # token sequence without linking a JSON parser to do it.
        with open(os.path.join(args.dump_dir, "tokens.txt"), "w") as f:
            f.write("".join(f"{i}\n" for i in ids))
        print(f"# dumped {len(dump) - 1} arrays -> {args.dump_dir}", file=sys.stderr)

    t0 = time.time()
    out = model.generate(ids, max_new=args.max_new, eos_ids=stop)
    gen_s = time.time() - t0

    new = out[len(ids):]
    print(tok.decode(new, skip_special_tokens=False))
    print(
        f"# gen {len(new)} tok in {gen_s:.1f}s ({len(new)/gen_s:.2f} tok/s, "
        f"no KV cache — the oracle is slow on purpose)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
