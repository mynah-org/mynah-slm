/* arch_qwen3.c — the Qwen3 decoder.
 *
 * One token at a time against a KV cache. There is no separate prefill path:
 * prefill is this function in a loop. That is the qwen-asr lesson: divergent
 * prefill and decode paths breed bugs that only appear
 * on one of them, and the batched prefill is a speed optimization that has to
 * prove itself equal to this, not a second definition of what the model is.
 *
 * Every step matches a line in docs/qwen3-arch.md.
 *
 * SPDX-License-Identifier: MIT */
#include "arch_qwen3.h"

#include "ingot/quant.h"
#include "threads.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ── threaded projection ────────────────────────────────────────────────────
 * ingot's matvec is single-threaded by design, so the parallelism has to come
 * from splitting the OUTPUT ROWS here. Each chunk is an independent matvec
 * over a contiguous slice of the weight matrix writing to a disjoint slice of
 * the output, so the result is bit-identical to the serial call — no reduction
 * crosses a thread, and no summation order changes. The parity gate still
 * holds at 1e-4 with threading on, which is the point.
 *
 * A row is a whole number of blocks (cols is a multiple of the block size for
 * every type we ship), so a chunk boundary never falls inside one. */
typedef struct {
    const uint8_t *base;
    float         *out;
    const float   *in;
    size_t         cols, row_bytes, rows_per_chunk, rows;
    int            type, rc;
} matvec_job;

static void matvec_chunk(void *ctx, int i) {
    matvec_job *j = ctx;
    const size_t first = (size_t)i * j->rows_per_chunk;
    if (first >= j->rows) return;
    size_t n = j->rows_per_chunk;
    if (first + n > j->rows) n = j->rows - first;

    if (ingot_matvec(j->type, j->base + first * j->row_bytes, n, j->cols,
                     j->in, j->out + first) != 0)
        j->rc = -1;              /* benign race: any failure sets the same -1 */
}

/* Weights are [out, in] row-major, so a projection is out = W · x. */
static int project(const mynah_slm_model_t *m, const ingot_tensor *w,
                   const float *in, float *out) {
    const int nth = mynah_slm_threads_count();

    /* ne[0] is the INPUT width in ggml order; rank 1 is a single row. */
    const size_t cols = (size_t)w->ne[0];
    const size_t rows = (w->rank >= 2) ? (size_t)w->ne[1] : 1;

    uint64_t block_elems = 0, block_bytes = 0;
    if (nth <= 1 || rows < 64 ||
        ingot_type_geometry(w->type, &block_elems, &block_bytes) != 0 ||
        block_elems == 0 || cols % block_elems != 0)
        return ingot_gguf_matvec(m->gguf, w, in, out);

    const uint8_t *base = ingot_gguf_data(m->gguf, w);
    if (!base) return -1;

    /* Four chunks per thread: the cores are not equally fast (and on a Mac the
     * pool is pinned to the performance cores for exactly that reason), so a
     * few extra chunks let the counter balance the tail. */
    int chunks = nth * 4;
    if ((size_t)chunks > rows) chunks = (int)rows;

    matvec_job j = {
        .base = base, .out = out, .in = in,
        .cols = cols, .row_bytes = (cols / block_elems) * block_bytes,
        .rows_per_chunk = (rows + (size_t)chunks - 1) / (size_t)chunks,
        .rows = rows, .type = w->type, .rc = 0,
    };
    mynah_slm_parallel_for(chunks, matvec_chunk, &j);
    return j.rc;
}

static float *alloc_f32(size_t n) {
    return mynah_slm_aligned_alloc(n * sizeof(float));
}

int mynah_slm_state_init(mynah_slm_state *s, const mynah_slm_model_t *m,
                         uint32_t n_ctx, char *err, size_t errsz) {
    memset(s, 0, sizeof *s);
    const mynah_slm_config *c = &m->cfg;

    if (n_ctx == 0 || n_ctx > c->n_ctx) n_ctx = c->n_ctx;
    s->model = m;
    s->n_ctx = n_ctx;

    if (mynah_slm_rope_init(&s->rope, c->head_dim, n_ctx, c->rope_theta) != 0) {
        snprintf(err, errsz, "cannot build the RoPE table for %u positions", n_ctx);
        return -1;
    }

    /* One contiguous cache per layer for K and for V, laid out
     * [pos][n_kv_heads * head_dim] so a position is contiguous — that is the
     * order the attention kernel walks. Allocated once, never grown inside the
     * token loop. */
    const size_t per_layer = (size_t)n_ctx * c->kv_dim;
    s->k_cache = alloc_f32(per_layer * c->n_layers);
    s->v_cache = alloc_f32(per_layer * c->n_layers);

    s->x       = alloc_f32(c->d_model);
    s->h       = alloc_f32(c->d_model);
    s->q       = alloc_f32(c->q_dim);
    s->attn    = alloc_f32(c->q_dim);
    s->proj    = alloc_f32(c->d_model);
    s->gate    = alloc_f32(c->d_ff);
    s->up      = alloc_f32(c->d_ff);
    s->scores    = alloc_f32(n_ctx);
    s->scores_mt = alloc_f32((size_t)n_ctx * c->n_heads);
    s->logits  = alloc_f32(c->vocab_size);
    s->embed_row = alloc_f32(c->d_model);

    if (!s->k_cache || !s->v_cache || !s->x || !s->h || !s->q || !s->attn ||
        !s->proj || !s->gate || !s->up || !s->scores || !s->scores_mt ||
        !s->logits || !s->embed_row) {
        snprintf(err, errsz, "out of memory for a %u-position context", n_ctx);
        mynah_slm_state_free(s);
        return -1;
    }
    return 0;
}

void mynah_slm_state_free(mynah_slm_state *s) {
    if (!s) return;
    mynah_slm_rope_free(&s->rope);
    mynah_slm_aligned_free(s->k_cache);
    mynah_slm_aligned_free(s->v_cache);
    mynah_slm_aligned_free(s->x);
    mynah_slm_aligned_free(s->h);
    mynah_slm_aligned_free(s->q);
    mynah_slm_aligned_free(s->attn);
    mynah_slm_aligned_free(s->proj);
    mynah_slm_aligned_free(s->gate);
    mynah_slm_aligned_free(s->up);
    mynah_slm_aligned_free(s->scores);
    mynah_slm_aligned_free(s->scores_mt);
    mynah_slm_aligned_free(s->logits);
    mynah_slm_aligned_free(s->embed_row);
    memset(s, 0, sizeof *s);
}

void mynah_slm_state_reset(mynah_slm_state *s) { s->n_past = 0; }

/* Read one row of the embedding matrix into f32.
 *
 * token_embd is quantized (Q6_K in the Q4_K_M build), so a row is decoded, not
 * copied. ingot has no "dequantize row n" call, so this goes through a matvec
 * against a one-hot vector... which would be 151936 wasted rows. Instead we
 * decode the single block range the row occupies. */
static int embed_row(const mynah_slm_model_t *m, uint32_t token, float *out) {
    const ingot_tensor *e = m->embed;
    const uint32_t d = m->cfg.d_model;

    uint64_t block_elems = 0, block_bytes = 0;
    if (ingot_type_geometry(e->type, &block_elems, &block_bytes) != 0) return -1;
    if (d % block_elems != 0) return -1;      /* a row must be whole blocks */

    const uint8_t *base = ingot_gguf_data(m->gguf, e);
    if (!base) return -1;

    const size_t blocks_per_row = d / block_elems;
    const size_t row_off = (size_t)token * blocks_per_row * block_bytes;

    /* One row = one "matrix" of a single row, decoded straight into out. */
    return ingot_dequant_matrix(e->type, base + row_off, 1, d, out);
}

int mynah_slm_forward(mynah_slm_state *s, uint32_t token, float *logits_out) {
    const mynah_slm_model_t *m = s->model;
    const mynah_slm_config  *c = &m->cfg;

    if (s->n_past >= s->n_ctx) return -1;          /* context exhausted */
    const uint32_t pos  = s->n_past;
    const uint32_t n_kv = pos + 1;

    if (embed_row(m, token, s->x) != 0) return -1; /* no sqrt(d_model) scaling */
    if (s->on_embed) s->on_embed(s->on_layer_ctx, s->x, c->d_model);

    const size_t per_layer = (size_t)s->n_ctx * c->kv_dim;

    for (uint32_t l = 0; l < c->n_layers; l++) {
        const mynah_slm_layer *w = &m->layers[l];

        /* Write K and V for this position straight into the cache: the
         * projection output IS the cache slot, so there is no copy. */
        float *k_layer = s->k_cache + (size_t)l * per_layer;
        float *v_layer = s->v_cache + (size_t)l * per_layer;
        float *k_slot  = k_layer + (size_t)pos * c->kv_dim;
        float *v_slot  = v_layer + (size_t)pos * c->kv_dim;

        mynah_slm_rms_norm(s->h, s->x, (const float *)ingot_gguf_data(m->gguf, w->attn_norm),
                           c->d_model, c->rms_eps);

        if (project(m, w->wq, s->h, s->q)     != 0) return -1;
        if (project(m, w->wk, s->h, k_slot)   != 0) return -1;
        if (project(m, w->wv, s->h, v_slot)   != 0) return -1;

        /* QK-norm before RoPE, per head, weights are head_dim long. */
        if (w->q_norm)
            mynah_slm_rms_norm_per_head(s->q, (const float *)ingot_gguf_data(m->gguf, w->q_norm),
                                        c->n_heads, c->head_dim, c->rms_eps);
        if (w->k_norm)
            mynah_slm_rms_norm_per_head(k_slot, (const float *)ingot_gguf_data(m->gguf, w->k_norm),
                                        c->n_kv_heads, c->head_dim, c->rms_eps);

        /* V is never rotated — only Q and K carry position. */
        mynah_slm_rope_apply(&s->rope, s->q,    c->n_heads,    pos);
        mynah_slm_rope_apply(&s->rope, k_slot,  c->n_kv_heads, pos);

        mynah_slm_attention_mt(s->attn, s->q, k_layer, v_layer, n_kv,
                               c->n_heads, c->n_kv_heads, c->head_dim, s->scores_mt);

        if (project(m, w->wo, s->attn, s->proj) != 0) return -1;
        mynah_slm_add(s->x, s->proj, c->d_model);

        mynah_slm_rms_norm(s->h, s->x, (const float *)ingot_gguf_data(m->gguf, w->ffn_norm),
                           c->d_model, c->rms_eps);

        if (project(m, w->gate, s->h, s->gate) != 0) return -1;
        if (project(m, w->up,   s->h, s->up)   != 0) return -1;
        mynah_slm_swiglu(s->gate, s->up, c->d_ff);

        if (project(m, w->down, s->gate, s->proj) != 0) return -1;
        mynah_slm_add(s->x, s->proj, c->d_model);

        if (s->on_layer) s->on_layer(s->on_layer_ctx, l, s->x, c->d_model);
    }

    mynah_slm_rms_norm(s->h, s->x, (const float *)ingot_gguf_data(m->gguf, m->out_norm),
                       c->d_model, c->rms_eps);
    if (s->on_final) s->on_final(s->on_layer_ctx, s->h, c->d_model);

    /* Tied embeddings: the LM head is the embedding matrix. This single matvec
     * is 151936 x 1024 — the hottest thing in the loop by a wide margin, and
     * the reason token_embd is not "just a lookup table". */
    const ingot_tensor *head = m->lm_head ? m->lm_head : m->embed;
    if (project(m, head, s->h, s->logits) != 0) return -1;

    s->n_past++;
    if (logits_out) memcpy(logits_out, s->logits, c->vocab_size * sizeof(float));
    return 0;
}
