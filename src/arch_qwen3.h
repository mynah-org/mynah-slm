/* arch_qwen3.h — inference state for the Qwen3 decoder.
 *
 * The state owns the KV cache and every scratch buffer, all allocated once at
 * init. Nothing in the token loop allocates: that is a hard rule, not a
 * preference, because a malloc per token shows up as jitter in tok/s long
 * before it shows up as a bug.
 *
 * SPDX-License-Identifier: MIT */
#ifndef MYNAH_SLM_ARCH_QWEN3_H
#define MYNAH_SLM_ARCH_QWEN3_H

#include "kernels.h"
#include "kvcache.h"
#include "model.h"

/* Optional per-layer taps, used by the parity harness to dump the residual
 * stream. NULL in production, and the only cost is a null check per layer. */
typedef void (*mynah_slm_layer_cb)(void *ctx, uint32_t layer, const float *x, uint32_t dim);
typedef void (*mynah_slm_final_cb)(void *ctx, const float *x, uint32_t dim);

typedef struct {
    const mynah_slm_model_t *model;
    mynah_slm_rope           rope;

    uint32_t n_ctx;     /* allocated positions */
    uint32_t n_past;    /* positions already in the cache */

    /* The precision K and V are KEPT at, SEPARATELY — they are not equally
     * sensitive. A key goes through the softmax exponent, where an error is
     * amplified before it is normalized; a value is averaged with weights that
     * sum to one, where errors partly cancel. Measured, that difference is
     * worth several bits (docs/perf.md).
     *
     * f32 is the reference the parity gate was set on; anything else trades
     * quality for bytes and has to be measured, never assumed. */
    mynah_slm_kv_type kv_k, kv_v;   /* requested; the cache is built from these */

    /* [n_layers][n_ctx][kv_dim], K and V possibly at different precisions.
     * A position is contiguous, which is the order attention walks. */
    mynah_slm_kv kv;

    /* scratch, all allocated once */
    float *x;          /* residual stream          [d_model]  */
    float *h;          /* normalized input         [d_model]  */
    float *q;          /*                          [q_dim]    */
    float *attn;       /* attention output         [q_dim]    */
    float *proj;       /* projection output        [d_model]  */
    float *gate, *up;  /*                          [d_ff]     */
    float *scores;     /* attention scores         [n_ctx]    */
    /* Per-head scores, [n_heads][n_ctx]. Heads run concurrently and a shared
     * scores buffer would be the single piece of state that makes them
     * dependent — the classic way a threaded attention goes subtly wrong. */
    float *scores_mt;
    float *logits;     /*                          [vocab]    */
    float *embed_row;  /* decoded embedding row    [d_model]  */

    /* ── prefill batch ──────────────────────────────────────────────────────
     * The same buffers as above, one row per token in the batch, plus the
     * dequantization strip the batched product needs. Allocated once at init
     * like everything else: prefill runs inside the token loop too. */
    uint32_t batch_max;               /* rows these hold */
    float *bx, *bh, *bq, *battn, *bproj, *bgate, *bup;
    float *strip;                     /* [STRIP_ROWS * max_cols] */
    float *bscores;                   /* [batch_max * n_ctx] attention scores */
    /* K and V are computed in f32 and then ENCODED into the cache, so they
     * need a landing place first. At f32 the encode is a memcpy; paying it
     * keeps one code path instead of two. */
    float *bk, *bv;                   /* [batch_max * kv_dim] */
    float *kgather, *vgather;         /* [n_ctx * head_dim], batched attention */

    mynah_slm_final_cb on_embed;   /* the residual stream before layer 0 */
    mynah_slm_layer_cb on_layer;
    mynah_slm_final_cb on_final;
    void              *on_layer_ctx;
} mynah_slm_state;

/* One projection: out[rows] = W · in[cols], threaded by output rows and
 * bit-identical to the serial call. Exposed for tests/bench_matvec.c, which is
 * how a kernel regression gets caught as a number rather than as "the engine
 * feels slower". Not part of the public API. */
int  mynah_slm_project(const mynah_slm_model_t *m, const ingot_tensor *w,
                       const float *in, float *out);

/* The KV precision has to be chosen HERE, not assigned to the state
 * afterwards: the cache is allocated and its layout fixed during init, and a
 * field set later would be read by nothing. (It was, once, and every format
 * silently measured as f32.) */
int  mynah_slm_state_init_kv(mynah_slm_state *s, const mynah_slm_model_t *m,
                             uint32_t n_ctx, mynah_slm_kv_type kv_k,
                             mynah_slm_kv_type kv_v, char *err, size_t errsz);

/* f32 unless MYNAH_SLM_KV / _K / _V say otherwise — the form for callers that
 * have no opinion, and the one a sweep can drive without a rebuild. */
int  mynah_slm_state_init(mynah_slm_state *s, const mynah_slm_model_t *m,
                          uint32_t n_ctx, char *err, size_t errsz);
void mynah_slm_state_free(mynah_slm_state *s);

/* Drop the history without reallocating. */
void mynah_slm_state_reset(mynah_slm_state *s);

/* Advance one token. Appends to the KV cache at position n_past and increments
 * it. `logits_out` may be NULL when only the cache update matters (prefill of
 * everything but the last token). Returns 0, or -1 with the context full. */
int  mynah_slm_forward(mynah_slm_state *s, uint32_t token, float *logits_out);

/* Advance `n` tokens at once — prefill.
 *
 * Same model, same cache, same result to within a summation reorder: the only
 * difference is that a weight is read once for the whole batch instead of once
 * per token, which is what turns prefill from memory-bound into compute-bound.
 * `logits_out` receives the LAST token's logits, the only ones anybody wants
 * from a prompt.
 *
 * n must be <= mynah_slm_batch_max(). n == 1 is forwarded to the one-token
 * path verbatim, so decode never changes shape. */
int  mynah_slm_forward_batch(mynah_slm_state *s, const uint32_t *tokens,
                             uint32_t n, float *logits_out);

/* Rows the batch scratch was sized for. `MYNAH_SLM_BATCH` in the environment
 * overrides the default — it is how the width gets measured rather than
 * assumed. */
uint32_t mynah_slm_batch_max(const mynah_slm_state *s);

#endif /* MYNAH_SLM_ARCH_QWEN3_H */
