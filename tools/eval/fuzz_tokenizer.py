"""Fuzz the C tokenizer against HF `tokenizers` on generated text.

The committed fixture covers cases somebody thought of. This covers the ones
nobody did: random mixtures of scripts, whitespace, punctuation, digits and
emoji, which is precisely where a hand-written pre-tokenizer diverges from a
regex engine.

    uv run python -m eval.fuzz_tokenizer ../models-local/<model>.gguf -n 2000

Deterministic: a seed is passed in, so a failure is reproducible from the
command line alone.
"""

from __future__ import annotations

import argparse
import os
import random
import subprocess
import sys

from tokenizers import Tokenizer

REF = os.path.join(os.path.dirname(__file__), "..", "..", "reference", "qwen3-0.6b", "tokenizer.json")
CLI = os.path.join(os.path.dirname(__file__), "..", "..", "mynah-slm")

# Deliberately biased towards the boundaries: whitespace runs, punctuation
# clusters and script changes are where the pre-tokenizer earns its keep.
ALPHABETS = [
    "abcdefghijklmnopqrstuvwxyz",
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "àèéìòùçñüößåæø",
    "0123456789",
    " \t\n\r",
    ".,;:!?'\"()[]{}<>/\\|-_=+*&^%$#@~`",
    "абвгдежзийклмн",
    "αβγδεζηθικλμν",
    "日本語中文한국어",
    "العربيةעברית",
    "अनुवादथาไทย",
    "🍕🎉👨‍👩‍👧‍👦🇮🇹😀🚀",
    "𝕌𝕟𝕚𝕔𝕠𝕕𝕖",
    " ​ ﻿",     # exotic spaces and a BOM
]


def random_text(rng: random.Random) -> str:
    parts = []
    for _ in range(rng.randint(1, 12)):
        alpha = rng.choice(ALPHABETS)
        parts.append("".join(rng.choice(alpha) for _ in range(rng.randint(1, 10))))
    return "".join(parts)


def c_tokenize(model: str, text: str, special: bool) -> list[int]:
    cmd = [CLI, "tokenize", "-m", model] + (["--special"] if special else [])
    p = subprocess.run(cmd, input=text.encode(), capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.decode().strip())
    return [int(x) for x in p.stdout.split()]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("-n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    if not os.path.exists(args.model):
        print(f"skip: no model at {args.model}", file=sys.stderr)
        return 77
    if not os.path.exists(CLI):
        print("skip: mynah-slm not built", file=sys.stderr)
        return 77

    tok = Tokenizer.from_file(REF)
    rng = random.Random(args.seed)

    # One process per case would spend all its time loading a 378 MB model, so
    # cases are batched behind a separator that survives tokenization: a run of
    # newlines is its own pre-token, and comparing whole batches is just as
    # strict as comparing them one by one.
    BATCH = 40
    mismatches = 0
    checked = 0

    texts = [random_text(rng) for _ in range(args.n)]
    for i in range(0, len(texts), BATCH):
        batch = texts[i:i + BATCH]
        joined = "\n\n".join(batch)

        want = tok.encode(joined, add_special_tokens=False).ids
        got = c_tokenize(args.model, joined, special=True)
        checked += len(batch)

        if want != got:
            mismatches += 1
            # Narrow it down to the individual case that differs.
            for t in batch:
                w = tok.encode(t, add_special_tokens=False).ids
                g = c_tokenize(args.model, t, special=True)
                if w != g:
                    print(f"MISMATCH on {t!r}\n  want {w}\n  got  {g}", file=sys.stderr)
                    break
            if mismatches >= 3:
                break

    print(f"{'FAILED' if mismatches else 'PASS'}: {checked} random cases, "
          f"{mismatches} mismatching batches (seed {args.seed})")
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
