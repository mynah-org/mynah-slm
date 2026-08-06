"""Unit tests for the oracle's own kernels. No checkpoint, no network.

The oracle is the reference the C engine is measured against, so "the oracle is
right" cannot rest on "it produced fluent text". These check the properties
that a wrong-but-plausible implementation would violate — above all the RoPE
variant, where the interleaved and split-half forms both run, both look
reasonable, and only one is Qwen3's.

    uv run python -m oracle.test_kernels
"""

from __future__ import annotations

import numpy as np

from .model import gqa_attention, rms_norm, rope_neox, silu, softmax

FAILURES = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'ok  ' if ok else 'FAIL'} {name}{'' if ok else '  <- ' + detail}")
    if not ok:
        FAILURES.append(name)


def test_rms_norm() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4, 64)).astype(np.float32)
    w = np.ones(64, dtype=np.float32)

    y = rms_norm(x, w, 1e-6)
    # With unit weights the result has RMS 1 along the last axis.
    rms = np.sqrt(np.mean(y**2, axis=-1))
    check("rms_norm normalizes to unit RMS", np.allclose(rms, 1.0, atol=1e-4),
          f"got {rms}")

    # Scaling the input must not change the output: that is the whole point of
    # RMS (as opposed to Layer) norm, which has no mean subtraction.
    check("rms_norm is scale invariant",
          np.allclose(rms_norm(x * 7.0, w, 1e-6), y, atol=1e-4))

    # The weight is a plain per-channel gain. Gemma's variant uses (1 + weight)
    # instead; if these two ever get unified, this test fails first.
    w2 = rng.standard_normal(64).astype(np.float32)
    check("rms_norm weight is a plain gain",
          np.allclose(rms_norm(x, w2, 1e-6), y * w2, atol=1e-4))


def test_rope_is_split_half() -> None:
    head_dim, theta = 8, 10000.0
    half = head_dim // 2
    x = np.zeros((1, 1, head_dim), dtype=np.float32)

    # Put a 1 at index 0 and a 1 at index `half`. Under SPLIT-HALF rotation
    # these two are partners and mix with each other. Under INTERLEAVED
    # rotation, index 0's partner is index 1, so index `half` would be
    # untouched by index 0's angle. The two are distinguishable in one shot.
    x[0, 0, 0] = 1.0
    x[0, 0, half] = 1.0
    pos = np.array([1], dtype=np.int64)
    y = rope_neox(x, pos, theta)

    inv0 = 1.0 / (theta ** 0.0)          # frequency of pair 0 is theta^0 == 1
    c, s = np.cos(1.0 * inv0), np.sin(1.0 * inv0)
    # split-half: out[0] = x[0]*c - x[half]*s ; out[half] = x[half]*c + x[0]*s
    check("rope is split-half NeoX, not interleaved",
          np.allclose(y[0, 0, 0], c - s, atol=1e-6)
          and np.allclose(y[0, 0, half], c + s, atol=1e-6),
          f"out[0]={y[0,0,0]:.6f} expected {c - s:.6f}")

    # Position 0 must be the identity: no rotation at all.
    rng = np.random.default_rng(1)
    z = rng.standard_normal((3, 2, head_dim)).astype(np.float32)
    check("rope at position 0 is identity",
          np.allclose(rope_neox(z, np.array([0, 0, 0]), theta), z, atol=1e-6))

    # Rotation preserves the norm of each (i, i+half) pair, hence of the head.
    r = rope_neox(z, np.array([5, 9, 13]), theta)
    check("rope preserves head norms",
          np.allclose(np.linalg.norm(r, axis=-1), np.linalg.norm(z, axis=-1), atol=1e-5))

    # Relative-position property: <RoPE(q,m), RoPE(k,n)> depends only on m - n.
    q = rng.standard_normal((1, 1, head_dim)).astype(np.float32)
    k = rng.standard_normal((1, 1, head_dim)).astype(np.float32)
    def rotated_dot(m: int, n: int) -> float:
        a = rope_neox(q, np.array([m]), theta)[0, 0]
        b = rope_neox(k, np.array([n]), theta)[0, 0]
        return float(np.dot(a, b))

    d1, d2 = rotated_dot(7, 4), rotated_dot(9, 6)   # both are distance 3
    check("rope dot product depends only on relative distance",
          abs(d1 - d2) < 1e-4, f"{d1:.6f} vs {d2:.6f}")


def test_softmax_and_silu() -> None:
    # The overflow case. A naive exp() here gives inf, and inf/inf = nan.
    x = np.array([[1000.0, 1001.0, 999.0]], dtype=np.float32)
    y = softmax(x)
    check("softmax survives large positive inputs",
          np.isfinite(y).all() and abs(y.sum() - 1.0) < 1e-6, f"{y}")

    check("silu(0) == 0", abs(float(silu(np.array([0.0]))[0])) < 1e-9)
    big = float(silu(np.array([20.0], dtype=np.float32))[0])
    check("silu saturates to identity", abs(big - 20.0) < 1e-4, f"{big}")


def test_gqa() -> None:
    rng = np.random.default_rng(2)
    seq, n_heads, n_kv, hd = 5, 4, 2, 8
    q = rng.standard_normal((seq, n_heads, hd)).astype(np.float32)
    k = rng.standard_normal((seq, n_kv, hd)).astype(np.float32)
    v = rng.standard_normal((seq, n_kv, hd)).astype(np.float32)

    out = gqa_attention(q, k, v, n_heads, n_kv, hd)
    check("gqa output shape", out.shape == (seq, n_heads, hd), str(out.shape))

    # Position 0 attends only to itself, so it must equal v[0] of its KV head
    # exactly — a one-element softmax is 1.0.
    check("gqa is causal at position 0",
          np.allclose(out[0, 0, :], v[0, 0, :], atol=1e-5))

    # Truncating the sequence must not change earlier outputs: that is what
    # causality means, and it is what makes a KV cache valid in the C engine.
    short = gqa_attention(q[:3], k[:3], v[:3], n_heads, n_kv, hd)
    check("gqa earlier positions ignore later tokens",
          np.allclose(short, out[:3], atol=1e-5))

    # Heads 0 and 1 share KV head 0; heads 2 and 3 share KV head 1. Feeding
    # identical queries to a shared group must give identical outputs.
    q2 = q.copy()
    q2[:, 1, :] = q2[:, 0, :]
    o2 = gqa_attention(q2, k, v, n_heads, n_kv, hd)
    check("gqa groups share their KV head",
          np.allclose(o2[:, 0, :], o2[:, 1, :], atol=1e-5))


def main() -> int:
    for fn in (test_rms_norm, test_rope_is_split_half, test_softmax_and_silu, test_gqa):
        print(f"\n-- {fn.__name__} --")
        fn()
    print(f"\n{'FAILED' if FAILURES else 'PASS'} ({len(FAILURES)} failures)")
    return 1 if FAILURES else 0


if __name__ == "__main__":
    raise SystemExit(main())
