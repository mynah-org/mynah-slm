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

    /* [n_layers][n_ctx][kv_dim] — a position is contiguous, which is the order
     * the attention kernel walks. */
    float *k_cache, *v_cache;

    /* scratch, all allocated once */
    float *x;          /* residual stream          [d_model]  */
    float *h;          /* normalized input         [d_model]  */
    float *q;          /*                          [q_dim]    */
    float *attn;       /* attention output         [q_dim]    */
    float *proj;       /* projection output        [d_model]  */
    float *gate, *up;  /*                          [d_ff]     */
    float *scores;     /* attention scores         [n_ctx]    */
    float *logits;     /*                          [vocab]    */
    float *embed_row;  /* decoded embedding row    [d_model]  */

    mynah_slm_final_cb on_embed;   /* the residual stream before layer 0 */
    mynah_slm_layer_cb on_layer;
    mynah_slm_final_cb on_final;
    void              *on_layer_ctx;
} mynah_slm_state;

int  mynah_slm_state_init(mynah_slm_state *s, const mynah_slm_model_t *m,
                          uint32_t n_ctx, char *err, size_t errsz);
void mynah_slm_state_free(mynah_slm_state *s);

/* Drop the history without reallocating. */
void mynah_slm_state_reset(mynah_slm_state *s);

/* Advance one token. Appends to the KV cache at position n_past and increments
 * it. `logits_out` may be NULL when only the cache update matters (prefill of
 * everything but the last token). Returns 0, or -1 with the context full. */
int  mynah_slm_forward(mynah_slm_state *s, uint32_t token, float *logits_out);

#endif /* MYNAH_SLM_ARCH_QWEN3_H */
