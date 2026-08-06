"""Prove the parity gate actually catches the bugs it exists to catch.

A green test that would stay green on a broken engine is worse than no test:
it authorises writing "PASS" in a commit. So before trusting compare.py, feed
it deliberately wrong implementations and check it says FAIL.

The perturbations are the real candidate bugs from docs/qwen3-arch.md, not
random noise:

  interleaved-rope   the wrong RoPE variant — both run, only one is Qwen3's
  no-qk-norm         forgetting the per-head QK-RMSNorm
  derived-head-dim   head_dim = d_model/n_heads (64) instead of key_length (128)
  fp16-residual      accumulating the residual stream in fp16, which the
                     massive activations at layer 13 (absmax ~8e3) overflow

    uv run python -m eval.test_gate ../models/Qwen3-0.6B-Q4_K_M.gguf
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from eval.compare import TOLERANCES, cosine, rel_error
from oracle import model as M
from oracle.model import Qwen3Model
from oracle.weights import Qwen3Weights

PROMPT_TOKENS = [151644, 872, 198, 34, 8630, 0, 15053, 61587, 30, 151645]


def rope_interleaved(x, positions, theta):
    """The wrong variant: pairs adjacent elements instead of i and i+half."""
    seq, heads, head_dim = x.shape
    half = head_dim // 2
    inv = 1.0 / (theta ** (np.arange(0, half, dtype=np.float32) * 2.0 / head_dim))
    ang = positions.astype(np.float32)[:, None] * inv[None, :]
    cos, sin = np.cos(ang)[:, None, :], np.sin(ang)[:, None, :]
    xe, xo = x[..., 0::2], x[..., 1::2]
    out = np.empty_like(x)
    out[..., 0::2] = xe * cos - xo * sin
    out[..., 1::2] = xo * cos + xe * sin
    return out


def run(weights, patch=None):
    original = M.rope_neox, M.rms_norm
    if patch:
        patch()
    try:
        dump = {"_layers": {0, 13, 27}}
        Qwen3Model(weights).forward(PROMPT_TOKENS, dump=dump)
        return {k: v for k, v in dump.items() if not k.startswith("_")}
    finally:
        M.rope_neox, M.rms_norm = original


def judge(name, good, bad) -> bool:
    """Return True when the gate REJECTS `bad` — that is the passing case."""
    worst = None
    for stage, max_rel, min_cos in TOLERANCES:
        if stage not in good or stage not in bad:
            continue
        rel, cos = rel_error(good[stage], bad[stage]), cosine(good[stage], bad[stage])
        if rel > max_rel or cos < min_cos:
            worst = (stage, rel, cos)
            break

    g_arg, b_arg = good["logits"].argmax(-1), bad["logits"].argmax(-1)
    agree = int((g_arg == b_arg).sum())

    if worst:
        stage, rel, cos = worst
        print(f"ok   {name:18s} REJECTED at {stage} (rel={rel:.2e}, cos={cos:.6f}), "
              f"argmax {agree}/{g_arg.size}")
        return True
    print(f"FAIL {name:18s} slipped through every tolerance — argmax {agree}/{g_arg.size}")
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    args = ap.parse_args()

    weights = Qwen3Weights(args.model)
    print(f"# {weights.config}", file=sys.stderr)

    good = run(weights)
    print("-- the gate must REJECT each of these --")

    ok = True

    def use_interleaved():
        M.rope_neox = rope_interleaved
    ok &= judge("interleaved-rope", good, run(weights, use_interleaved))

    def skip_qk_norm():
        real = M.rms_norm
        # Only the per-head call has a weight sized head_dim (128); the others
        # are d_model. That is enough to disable exactly the QK norm.
        M.rms_norm = lambda x, w, eps: x if w.shape[-1] == 128 else real(x, w, eps)
    ok &= judge("no-qk-norm", good, run(weights, skip_qk_norm))

    def fp16_residual():
        real = M.rms_norm
        M.rms_norm = lambda x, w, eps: real(x.astype(np.float16).astype(np.float32), w, eps)
    ok &= judge("fp16-residual", good, run(weights, fp16_residual))

    print(f"\n{'PASS' if ok else 'FAILED'} — the gate {'catches' if ok else 'MISSES'} "
          f"the bugs it exists for")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
