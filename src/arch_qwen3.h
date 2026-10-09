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

/* ── one sequence ──────────────────────────────────────────────────────────
 * Everything that is a function of ONE token history, and nothing else: the
 * KV cache, how far into it we are, and the short-conv history of a hybrid
 * family. A request owns one; a scheduler holds many and runs them all on one
 * workspace (below), because everything else a forward pass touches is a
 * function of the model and the capacity, not of the sequence.
 *
 * Sized per request (prompt + max_tokens, capped), never to the model's
 * context: a 40960-position cache per request would be most of the machine.
 * mynah_slm_seq_reserve() reuses the allocation when it is big enough, so a
 * slot that serves request after request allocates once. */
typedef struct mynah_slm_seq {
    const mynah_slm_model_t *model;
    uint32_t n_ctx;     /* positions the cache holds */
    uint32_t n_past;    /* positions already in it */

    /* [n_attn_layers][n_ctx][kv_dim], K and V possibly at different
     * precisions (see kv_k/kv_v below). A position is contiguous, which is
     * the order attention walks. Indexed by cfg.op_slot[layer], NOT by layer:
     * on a hybrid model only 8 of 30 layers have a cache at all. */
    mynah_slm_kv kv;

    /* ── short-conv state, for hybrid families ──────────────────────────────
     * [n_conv_layers][taps-1][d_model]: the previous inputs each depthwise FIR
     * still needs, and the whole of what a conv layer carries between tokens.
     *
     * It is CONSTANT in the context length — 22 layers x 2 x 2048 floats is
     * 352 KB for LFM2 whatever the prompt — which is the structural reason the
     * architecture is interesting on a CPU: only 8 of 30 layers hold anything
     * that grows per token. NULL when the model has no conv layers. */
    float   *conv_hist;
} mynah_slm_seq;

/* Allocates a cache of n_ctx positions (clamped to the model's context).
 * Returns 0, or -1 with nothing allocated and the reason in err. */
int  mynah_slm_seq_init(mynah_slm_seq *q, const mynah_slm_model_t *m, uint32_t n_ctx,
                        mynah_slm_kv_type kv_k, mynah_slm_kv_type kv_v,
                        char *err, size_t errsz);
void mynah_slm_seq_free(mynah_slm_seq *q);

/* Empty the history; every allocation kept. */
void mynah_slm_seq_reset(mynah_slm_seq *q);

/* Make `q` an EMPTY sequence of at least n_ctx positions at these
 * precisions, reusing what it already holds when that is big enough and of
 * the same types — the admission-time call of a pooled slot. Allocates only
 * when it must, and never from inside the token loop. `q` may be zeroed
 * (never used). Returns 0, or -1 leaving `q` freed (zeroed). */
int  mynah_slm_seq_reserve(mynah_slm_seq *q, const mynah_slm_model_t *m, uint32_t n_ctx,
                           mynah_slm_kv_type kv_k, mynah_slm_kv_type kv_v,
                           char *err, size_t errsz);

/* ── the workspace ─────────────────────────────────────────────────────────
 * Model-shared state: scratch, batch scratch, the RoPE table. One forward pass
 * runs at a time on it, for whichever sequence it is handed. `own` is the
 * sequence the single-sequence API (mynah_slm_forward & co.) runs on — empty
 * (n_ctx 0) in a workspace a scheduler holds. */
typedef struct {
    const mynah_slm_model_t *model;
    mynah_slm_rope           rope;

    /* Positions the scratch and the RoPE table are sized for: the most any
     * sequence run on this workspace may hold. */
    uint32_t ctx_cap;

    /* The precision K and V are KEPT at, SEPARATELY — they are not equally
     * sensitive. A key goes through the softmax exponent, where an error is
     * amplified before it is normalized; a value is averaged with weights that
     * sum to one, where errors partly cancel. Measured, that difference is
     * worth several bits (docs/perf.md).
     *
     * f32 is the reference the parity gate was set on; anything else trades
     * quality for bytes and has to be measured, never assumed. These are what
     * `own` was built with. */
    mynah_slm_kv_type kv_k, kv_v;

    mynah_slm_seq own;

    /* Per-batch working room for the short-conv layers, allocated at full
     * batch width like every other scratch so prefill never allocates. NULL
     * when the model has no conv layers. */
    float   *conv_bcx;    /* [batch_max][3 * d_model] — B, C and x, in that order */
    float   *conv_bx;     /* [batch_max][d_model]     — the gated FIR input      */
    float   *conv_y;      /* [batch_max][d_model]     — the FIR output           */

    /* scratch, all allocated once */
    float *x;          /* residual stream          [d_model]  */
    float *h;          /* normalized input         [d_model]  */
    float *q;          /*                          [q_dim]    */
    float *attn;       /* attention output         [q_dim]    */
    float *proj;       /* projection output        [d_model]  */
    float *gate, *up;  /*                          [d_ff]     */
    float *scores;     /* attention scores         [ctx_cap]  */
    /* Per-head scores, [n_heads][ctx_cap]. Heads run concurrently and a shared
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
    float *bscores;                   /* [batch_max * ctx_cap] attention scores */
    /* K and V are computed in f32 and then ENCODED into the cache, so they
     * need a landing place first. At f32 the encode is a memcpy; paying it
     * keeps one code path instead of two. */
    float *bk, *bv;                   /* [batch_max * kv_dim] */
    float *kgather, *vgather;         /* [ctx_cap * head_dim], batched attention */

    /* ── multi-sequence decode (mynah_slm_forward_multi) ────────────────────
     * Rows of the batch scratch above are reused, one per sequence; these are
     * what a decode step over B DIFFERENT histories needs on top. NULL and
     * dec_max 0 until mynah_slm_state_init_decode(). */
    uint32_t dec_max;                 /* sequences one step may carry */
    float *mscores;                   /* [dec_max][n_heads][ctx_cap] */
    float *mlogits;                   /* [dec_max][vocab] */
    mynah_slm_attn_seq *mattn;        /* [dec_max] */
    /* [dec_max] prepared activations (qmat.h), one per sequence, for the
     * weight-stationary product (MYNAH_SLM_DECODE_PRODUCT=ws, K7) */
    struct mynah_slm_matvec_in *mprep;

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

/* The precisions mynah_slm_state_init() would pick: f32, or what
 * MYNAH_SLM_KV / _K / _V say. Always returns 0. */
int  mynah_slm_kv_types_from_env(mynah_slm_kv_type *kv_k, mynah_slm_kv_type *kv_v);

/* A workspace with no sequence of its own, for a caller that brings its own
 * (a scheduler). ctx_cap bounds every sequence later run on it. */
int  mynah_slm_state_init_workspace(mynah_slm_state *s, const mynah_slm_model_t *m,
                                    uint32_t ctx_cap, char *err, size_t errsz);

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

/* The same two calls on a sequence the caller owns, run on `ws`'s scratch.
 * mynah_slm_forward(s, ...) is exactly mynah_slm_seq_forward(s, &s->own, ...).
 * -1 when the sequence is full, belongs to another model, or holds more
 * positions than the workspace was sized for. Logits also land in ws->logits. */
int  mynah_slm_seq_forward(mynah_slm_state *ws, mynah_slm_seq *q, uint32_t token,
                           float *logits_out);
int  mynah_slm_seq_forward_batch(mynah_slm_state *ws, mynah_slm_seq *q,
                                 const uint32_t *tokens, uint32_t n, float *logits_out);

/* ── one decode step for several sequences ─────────────────────────────────
 * Allocates what mynah_slm_forward_multi needs for up to dec_max sequences.
 * Refused (-1, reason in err) on a hybrid family — its short-conv layers have
 * no batched form yet — and when dec_max exceeds the batch scratch rows
 * (MYNAH_SLM_BATCH). Call once, at setup. */
int  mynah_slm_state_init_decode(mynah_slm_state *s, uint32_t dec_max,
                                 char *err, size_t errsz);

/* Advance n DIFFERENT sequences by one token each: tokens[b] is fed to
 * seqs[b] at its own position. Every projection is ONE product over the
 * [n x d] rows (weights read once for all n — the reason continuous batching
 * pays on a memory-bound decode); attention runs per sequence over its own
 * history, in one pool region. logits[b] is set to row b of the workspace's
 * logits, valid until the next call on `s`.
 *
 * n == 1 is the single-token path verbatim (bit-identical to
 * mynah_slm_seq_forward). For n > 1 each sequence's logits equal its solo
 * decode to a summation reorder (tests/test_synth), not bit for bit.
 *
 * Step isolation: every sequence is validated before any is touched, and no
 * n_past moves unless the whole step succeeded. A failure can leave K/V
 * written at a sequence's current position — the slot its next attempt
 * overwrites — so re-stepping one sequence alone is exactly its solo step.
 * Returns 0, or -1. */
int  mynah_slm_forward_multi(mynah_slm_state *s, mynah_slm_seq *const *seqs,
                             const uint32_t *tokens, uint32_t n, float **logits);

/* Which product forward_multi uses per weight, from MYNAH_SLM_DECODE_PRODUCT:
 *   "matvec"            n threaded matvecs, our fused kernels — weights read
 *                       n times, but bit-identical to each sequence's solo
 *                       step, and one attention region per layer for all n.
 *   "matmat"            one mynah_slm_qmatmat over the n rows — weights read
 *                       once, but each strip is dequantized to f32 first.
 *                       MEASURED 3-5x SLOWER than n solo steps at n = 2..8 on
 *                       the 0.6B geometry (bench_decode, S1-c). Opt-in.
 *   "ws" (default)      one weight-stationary pass per weight (K7,
 *                       mynah_slm_matvec_ws): weights read once for all n,
 *                       each weight unit decoded once, and STILL bit-identical
 *                       to each sequence's solo step. Where no ws kernel
 *                       takes the tensor (ingot's types) the rows are walked
 *                       in 16-row tiles, each tile multiplied by every
 *                       sequence by the solo call, so the weight is still
 *                       read from memory once. Measured 1.17-1.46x faster
 *                       than "matvec" at n = 2..8 (.work/batched-decode-
 *                       kernel.md); the default since K7.
 * The setter exists for an interleaved in-process A/B (a MYNAH_SLM_DECODE_*
 * value, or -1 back to the environment's choice); call it between steps. */
#define MYNAH_SLM_DECODE_MATVEC 0
#define MYNAH_SLM_DECODE_MATMAT 1
#define MYNAH_SLM_DECODE_WS     2
const char *mynah_slm_decode_product_name(void);
void        mynah_slm_decode_product_set(int product);

/* Rows the batch scratch was sized for. `MYNAH_SLM_BATCH` in the environment
 * overrides the default — it is how the width gets measured rather than
 * assumed. */
uint32_t mynah_slm_batch_max(const mynah_slm_state *s);

#endif /* MYNAH_SLM_ARCH_QWEN3_H */
