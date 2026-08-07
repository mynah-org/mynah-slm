/* kernels.h — the numeric primitives of the decoder.
 *
 * Everything here is f32 in, f32 out, single-threaded and unfused. Threading
 * and fusion come later, behind these same signatures, once the forward pass
 * matches the oracle: a fused kernel that is wrong is much harder to find than
 * a slow one that is wrong.
 *
 * Products against QUANTIZED WEIGHTS are not here — they live in qmat.c, which
 * owns the decode-vs-prefill dispatch. What is here is the arithmetic on
 * activations, where both operands are already f32: attention included, since
 * its matrices are the KV cache and not a weight tensor.
 *
 * SPDX-License-Identifier: MIT */
#ifndef MYNAH_SLM_KERNELS_H
#define MYNAH_SLM_KERNELS_H

#include <stddef.h>
#include <stdint.h>

/* ── allocation ───────────────────────────────────────────────────────────── */

/* 64-byte aligned, because every SIMD and BLAS buffer must be: a cache-line
 * split costs more than the allocation. Returns NULL on failure. */
void *mynah_slm_aligned_alloc(size_t bytes);
void  mynah_slm_aligned_free(void *p);

/* ── norms ────────────────────────────────────────────────────────────────── */

/* out[dim] = x[dim] / sqrt(mean(x^2) + eps) * weight[dim]
 *
 * The plain form, where the stored weight is a gain. Gemma's variant scales by
 * (1 + weight) instead and is a DIFFERENT function — when that lands it gets
 * its own entry point, not a flag here. */
void mynah_slm_rms_norm(float *out, const float *x, const float *weight,
                        uint32_t dim, float eps);

/* RMSNorm applied independently to each head of x[n_heads * head_dim], with a
 * single weight[head_dim] shared across heads. In-place.
 *
 * This is Qwen3's QK-norm. The giveaway in the checkpoint is that
 * attn_q_norm.weight is head_dim long (128) rather than d_model. */
void mynah_slm_rms_norm_per_head(float *x, const float *weight,
                                 uint32_t n_heads, uint32_t head_dim, float eps);

/* ── RoPE ─────────────────────────────────────────────────────────────────── */

/* Precomputed cos/sin, indexed [pos * (head_dim/2) + i]. Built once per model
 * and shared by every layer. */
typedef struct {
    float   *cos;
    float   *sin;
    uint32_t head_dim;
    uint32_t max_pos;
} mynah_slm_rope;

int  mynah_slm_rope_init(mynah_slm_rope *r, uint32_t head_dim, uint32_t max_pos,
                         float theta);
void mynah_slm_rope_free(mynah_slm_rope *r);

/* Rotate x[n_heads * head_dim] in place, at absolute position `pos`.
 *
 * SPLIT-HALF (NeoX): element i pairs with element i + head_dim/2. NOT the
 * interleaved form that pairs 2i with 2i+1. Both variants run and produce
 * fluent output; only one is Qwen3's, and the parity gate rejects the other at
 * layer 0 while its greedy argmax still agrees on half the positions. */
void mynah_slm_rope_apply(const mynah_slm_rope *r, float *x,
                          uint32_t n_heads, uint32_t pos);

/* ── activations ──────────────────────────────────────────────────────────── */

/* x * sigmoid(x), through a stable sigmoid: exp() is only ever called on a
 * non-positive argument. The naive x/(1+exp(-x)) overflows for large negative
 * x, and under -ffast-math the resulting inf is UB that gcc/x86 turns into a
 * NaN (mynah-asr CI, 2026-07-18). We do not build with -ffast-math, and we
 * still do not rely on that. */
void mynah_slm_silu(float *x, size_t n);

/* gate[n] = silu(gate[n]) * up[n] — the SwiGLU second half, in place on gate. */
void mynah_slm_swiglu(float *gate, const float *up, size_t n);

/* In place, max-subtracted. */
void mynah_slm_softmax(float *x, size_t n);

void mynah_slm_add(float *y, const float *x, size_t n);   /* y += x */

/* ── attention ────────────────────────────────────────────────────────────── */

/* Causal grouped-query attention for ONE query position against a KV history.
 *
 *   q       [n_heads * head_dim]      the current token's queries, post-RoPE
 *   k, v    [n_kv * n_kv_heads * head_dim]   history, oldest first
 *   n_kv    how many positions are in the cache, INCLUDING this one
 *   out     [n_heads * head_dim]
 *
 * Causality is structural rather than a mask: the history simply stops at the
 * current token. scratch must hold at least n_kv floats. */
void mynah_slm_attention(float *out, const float *q, const float *k, const float *v,
                         uint32_t n_kv, uint32_t n_heads, uint32_t n_kv_heads,
                         uint32_t head_dim, float *scratch);

/* The same, one head per task. Heads are fully independent — separate scores,
 * separate output slice — so this is bit-identical to the serial version, not
 * merely close. `scratch` must hold n_heads * n_kv floats: a shared scores
 * buffer would be the one piece of state that makes them dependent. */
void mynah_slm_attention_mt(float *out, const float *q, const float *k, const float *v,
                            uint32_t n_kv, uint32_t n_heads, uint32_t n_kv_heads,
                            uint32_t head_dim, float *scratch);

/* Causal GQA for a BATCH of consecutive query positions — prefill.
 *
 *   q, out   [n_q][q_stride], row t is the query at absolute position pos0 + t
 *   k, v     the layer's caches, [n_kv][kv_dim], oldest first
 *   scores   [n_q * (pos0 + n_q)] scratch
 *
 * The one-query-at-a-time form re-reads the whole KV history for every query,
 * which over a prompt is O(n^2) memory traffic on top of O(n^2) arithmetic.
 * Here each head is two sgemms — scores, then the weighted sum of values — so
 * the history is read once per head instead of once per query.
 *
 * CAUSALITY IS A MASK HERE, not a truncation: the batch is a triangle, query t
 * may read exactly pos0 + t + 1 keys. The tail of each row is zeroed after the
 * softmax so the second product cannot see the future.
 *
 * Not bit-identical to the per-query form — sgemm sums in its own order — but
 * a reorder of the same arithmetic, which tests/test_batch.c pins. */
void mynah_slm_attention_batch(float *out, const float *q,
                               const float *k, const float *v,
                               uint32_t pos0, uint32_t n_q, uint32_t n_heads,
                               uint32_t n_kv_heads, uint32_t head_dim,
                               uint32_t q_stride, float *scores);

#endif /* MYNAH_SLM_KERNELS_H */
