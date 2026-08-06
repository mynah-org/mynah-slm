"""Compare the C engine's activation dumps against the oracle's, stage by stage.

    uv run python -m eval.compare ../tests/golden/it_hello ../build/dump/it_hello

Exit 0 = every stage within tolerance, 1 = a stage failed, 77 = nothing to
compare (skip, not failure).

Why not a checksum: the two implementations will never be bit-identical — the C
side sums in a different order, uses SIMD reductions, and may accumulate in a
different width. A hash comparison would fail on a correct engine and teach us
to ignore it. What we assert instead is that each stage is within a tolerance
chosen for that stage.

Why per-stage: the residual stream grows by three orders of magnitude between
the embedding and layer 13 (Qwen3 has the usual massive activations), so one
absolute tolerance is either meaningless early or impossible late. Each stage
is judged on RELATIVE error and on cosine similarity, which are scale-free.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

# (stage, max relative error, min cosine similarity)
#
# Tighter early, looser late: error compounds through 28 layers, and the
# logits inherit everything upstream. The cosine floor is the one that matters
# for logits — an argmax only needs the direction, and the greedy trajectory is
# what we actually ship.
TOLERANCES = [
    ("embed",       1e-6, 1.0 - 1e-9),   # a table lookup: should be exact
    ("layer0",      1e-4, 1.0 - 1e-7),
    ("layer13",     1e-3, 1.0 - 1e-6),
    ("layer27",     2e-3, 1.0 - 1e-6),
    ("final_norm",  2e-3, 1.0 - 1e-6),
    ("logits",      5e-3, 1.0 - 1e-5),
]


def rel_error(a: np.ndarray, b: np.ndarray) -> float:
    """||a - b|| / ||a||, scale-free and insensitive to a single outlier in a
    way that max|a-b| is not."""
    denom = np.linalg.norm(a.ravel())
    if denom == 0.0:
        return float(np.linalg.norm((a - b).ravel()))
    return float(np.linalg.norm((a - b).ravel()) / denom)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    x, y = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    n = np.linalg.norm(x) * np.linalg.norm(y)
    return 1.0 if n == 0.0 else float(np.dot(x, y) / n)


def compare_stage(name, golden_dir, actual_dir, max_rel, min_cos):
    gp = os.path.join(golden_dir, f"{name}.npy")
    ap = os.path.join(actual_dir, f"{name}.npy")
    if not os.path.exists(gp):
        return None, f"{name:12s} SKIP  no golden dump"
    if not os.path.exists(ap):
        return None, f"{name:12s} SKIP  engine did not dump this stage"

    g, a = np.load(gp), np.load(ap)
    if g.shape != a.shape:
        return False, f"{name:12s} FAIL  shape {a.shape} != golden {g.shape}"

    rel, cos = rel_error(g, a), cosine(g, a)
    absmax = float(np.abs(g - a).max())
    ok = rel <= max_rel and cos >= min_cos
    return ok, (f"{name:12s} {'ok  ' if ok else 'FAIL'}  rel={rel:.2e} (<= {max_rel:.0e})  "
                f"cos={cos:.9f} (>= {min_cos:.9f})  absmax={absmax:.3e}")


def compare_argmax(golden_dir, actual_dir):
    """The one that decides whether the engine is usable: does greedy decoding
    pick the same token at every position? Logits can drift a little; the
    trajectory must not fork."""
    gp = os.path.join(golden_dir, "logits.npy")
    ap = os.path.join(actual_dir, "logits.npy")
    if not (os.path.exists(gp) and os.path.exists(ap)):
        return None, "argmax       SKIP  no logits on one side"

    g, a = np.load(gp), np.load(ap)
    if g.shape != a.shape:
        return False, "argmax       FAIL  logits shapes differ"

    gi, ai = g.argmax(-1), a.argmax(-1)
    agree = int((gi == ai).sum())
    total = int(gi.size)
    ok = agree == total
    detail = "" if ok else f"  first divergence at position {int(np.argmax(gi != ai))}"
    return ok, f"{'argmax':12s} {'ok  ' if ok else 'FAIL'}  {agree}/{total} positions agree{detail}"


def main() -> int:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("golden", help="oracle dump dir (make golden-dump)")
    ap_.add_argument("actual", help="C engine dump dir")
    args = ap_.parse_args()

    if not os.path.isdir(args.golden):
        print(f"skip: no golden dumps in {args.golden} (make golden-dump)", file=sys.stderr)
        return 77
    if not os.path.isdir(args.actual):
        print(f"skip: no engine dumps in {args.actual}", file=sys.stderr)
        return 77

    results = []
    for name, max_rel, min_cos in TOLERANCES:
        ok, line = compare_stage(name, args.golden, args.actual, max_rel, min_cos)
        print(line)
        results.append(ok)

    ok, line = compare_argmax(args.golden, args.actual)
    print(line)
    results.append(ok)

    compared = [r for r in results if r is not None]
    if not compared:
        print("skip: nothing was actually compared", file=sys.stderr)
        return 77

    failed = compared.count(False)
    print(f"\n{'FAILED' if failed else 'PASS'} ({failed}/{len(compared)} stages failed)")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
