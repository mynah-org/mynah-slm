"""Pure-numpy Qwen3 forward pass. No torch, no transformers.

This is the reference src/arch_qwen3.c must match, stage by stage, at a
numeric tolerance. It is written for legibility over speed: every step matches
a line in docs/qwen3-arch.md, and nothing is fused.

Weight convention: GGUF stores linears as [out_features, in_features] in
row-major, so a projection is `x @ W.T`.
"""

from __future__ import annotations

import numpy as np

from .weights import Qwen3Weights


def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float) -> np.ndarray:
    """x / sqrt(mean(x^2) + eps) * weight, accumulated in f32.

    Note this is the plain form. Gemma's variant scales by (1 + weight) instead
    and is a different function — do not unify them later.
    """
    var = np.mean(x.astype(np.float32) ** 2, axis=-1, keepdims=True)
    return (x / np.sqrt(var + eps)) * weight


def rope_neox(x: np.ndarray, positions: np.ndarray, theta: float) -> np.ndarray:
    """NeoX split-half RoPE: x[i] rotates against x[i + head_dim/2].

    NOT the interleaved variant, which pairs adjacent elements. Both exist in
    the wild and in qwen-tts; picking the wrong one gives a model that runs and
    produces fluent nonsense.

    x is [seq, heads, head_dim].
    """
    seq, _, head_dim = x.shape
    half = head_dim // 2

    inv_freq = 1.0 / (theta ** (np.arange(0, half, dtype=np.float32) * 2.0 / head_dim))
    angles = positions.astype(np.float32)[:, None] * inv_freq[None, :]  # [seq, half]
    cos = np.cos(angles)[:, None, :]                                    # [seq, 1, half]
    sin = np.sin(angles)[:, None, :]

    x1, x2 = x[..., :half], x[..., half:]
    return np.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Max-subtracted. Never exp() a large positive number: the C side has the
    same rule for the same reason (mynah-asr, 2026-07-18)."""
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=axis, keepdims=True)


def silu(x: np.ndarray) -> np.ndarray:
    """x * sigmoid(x), through a STABLE sigmoid.

    The naive x / (1 + exp(-x)) overflows for large negative x. That is the
    same rule the C side follows (mynah_asr_sigmoid, after the 2026-07-18 CI
    incident where an inf under -ffast-math became a NaN on linux/x86 only):
    exp() is only ever called on a non-positive argument, where it cannot
    overflow.
    """
    x = x.astype(np.float32)
    pos = x >= 0
    out = np.empty_like(x)
    out[pos] = x[pos] / (1.0 + np.exp(-x[pos]))
    e = np.exp(x[~pos])
    out[~pos] = x[~pos] * e / (1.0 + e)
    return out


def gqa_attention(q, k, v, n_heads, n_kv_heads, head_dim):
    """Causal grouped-query attention.

    q is [seq, n_heads, head_dim]; k and v are [seq, n_kv_heads, head_dim].
    Each KV head serves n_heads // n_kv_heads query heads.
    """
    seq = q.shape[0]
    group = n_heads // n_kv_heads
    scale = 1.0 / np.sqrt(head_dim)

    out = np.empty((seq, n_heads, head_dim), dtype=np.float32)
    causal = np.triu(np.full((seq, seq), -np.inf, dtype=np.float32), k=1)

    for h in range(n_heads):
        kh = h // group
        scores = (q[:, h, :] @ k[:, kh, :].T) * scale   # [seq, seq]
        out[:, h, :] = softmax(scores + causal) @ v[:, kh, :]
    return out


class Qwen3Model:
    def __init__(self, weights: Qwen3Weights):
        self.w = weights
        self.cfg = weights.config

    def forward(self, tokens, dump: dict | None = None) -> np.ndarray:
        """Full-sequence forward. Returns logits for every position,
        [seq, vocab].

        `dump`, when given, collects named intermediates for the parity
        harness. Prefill-only by design: the oracle has no KV cache, because
        its job is to be obviously correct, not fast. The C side must match it
        with and without a cache — that equivalence is its own test.
        """
        cfg = self.cfg
        tokens = np.asarray(tokens, dtype=np.int64)
        positions = np.arange(len(tokens), dtype=np.int64)

        x = self.w.embed[tokens].astype(np.float32)   # no sqrt(d_model) scaling
        if dump is not None:
            dump["embed"] = x.copy()

        for i, layer in enumerate(self.w.layers):
            h = rms_norm(x, layer["attn_norm"], cfg.rms_eps)

            q = (h @ layer["wq"].T).reshape(-1, cfg.n_heads, cfg.head_dim)
            k = (h @ layer["wk"].T).reshape(-1, cfg.n_kv_heads, cfg.head_dim)
            v = (h @ layer["wv"].T).reshape(-1, cfg.n_kv_heads, cfg.head_dim)

            # QK-norm: RMSNorm over head_dim, per head. The weights are [128],
            # i.e. head_dim, which is how you can tell from the file alone.
            q = rms_norm(q, layer["q_norm"], cfg.rms_eps)
            k = rms_norm(k, layer["k_norm"], cfg.rms_eps)

            q = rope_neox(q, positions, cfg.rope_theta)
            k = rope_neox(k, positions, cfg.rope_theta)

            a = gqa_attention(q, k, v, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim)
            x = x + a.reshape(-1, cfg.q_dim) @ layer["wo"].T

            h = rms_norm(x, layer["ffn_norm"], cfg.rms_eps)
            x = x + (silu(h @ layer["gate"].T) * (h @ layer["up"].T)) @ layer["down"].T

            if dump is not None and i in dump.get("_layers", ()):
                dump[f"layer{i}"] = x.copy()

        x = rms_norm(x, self.w.out_norm, cfg.rms_eps)
        if dump is not None:
            dump["final_norm"] = x.copy()

        # Tied embeddings: the LM head IS the embedding matrix. There is no
        # separate output.weight in this checkpoint.
        logits = x @ self.w.embed.T
        if dump is not None:
            dump["logits"] = logits.copy()
        return logits

    def generate(self, tokens, max_new=32, eos_ids=(), on_token=None):
        """Greedy. Recomputes the whole prefix each step — O(n^2) and slow on
        purpose: a cache in the reference would be a second thing that can be
        wrong."""
        out = list(tokens)
        for _ in range(max_new):
            nxt = int(np.argmax(self.forward(out)[-1]))
            out.append(nxt)
            if on_token:
                on_token(nxt)
            if nxt in eos_ids:
                break
        return out
