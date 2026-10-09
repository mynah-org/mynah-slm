/* backend_cpu.c — the reference backend: the existing kernels, behind the vtable.
 *
 * NO NEW NUMERICS. Every entry calls the function arch_qwen3.c calls for the
 * same step, with the same arguments, so a forward pass routed through here is
 * the forward pass that the parity gate was set on. tests/test_backend.c holds
 * each op to that bit for bit.
 *
 * The one piece of logic that is not a straight call is matvec's row split,
 * which is arch_qwen3.c's mynah_slm_project() restated for a weight handle
 * rather than a model tensor: same prepare-once hoist, same chunking, same
 * qmat-then-ingot choice per chunk. Splitting by output rows reduces nothing
 * across threads, so it is bit-identical to the serial call by construction.
 *
 * SPDX-License-Identifier: MIT */
#include "backend_ops.h"

#include "kernels.h"
#include "kvcache.h"
#include "qmat.h"
#include "threads.h"

#include "ingot/dtype.h"
#include "ingot/quant.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    /* The dequantization strip mynah_slm_qmatmat needs: STRIP_ROWS x the
     * widest weight uploaded so far. Grown at upload (model load), never in
     * the token loop. */
    float *strip;
    size_t strip_cols;
} cpu_state;

/* Per-sequence scratch, sized at kv_create so attention never allocates. */
typedef struct {
    mynah_slm_kv kv;
    float *scores_mt;   /* [n_heads][n_ctx]   — one-query attention */
    float *bscores;     /* [max_batch][n_ctx] — batched attention   */
    float *kgather;     /* [n_ctx][head_dim]  — packed-KV batch path */
    float *vgather;
} cpu_kv;

static void set_err(char *err, size_t errsz, const char *msg) {
    if (err && errsz > 0) snprintf(err, errsz, "%s", msg);
}

static void cpu_close(void *st) {
    cpu_state *s = st;
    if (!s) return;
    mynah_slm_aligned_free(s->strip);
    free(s);
}

/* ── memory: host memory, so the copies are memcpy and sync is a no-op ──── */

static float *cpu_alloc(void *st, size_t n, char *err, size_t errsz) {
    (void)st;
    float *p = mynah_slm_aligned_alloc(n * sizeof(float));
    if (!p) set_err(err, errsz, "out of memory");
    return p;
}

static void cpu_free(void *st, float *p) { (void)st; mynah_slm_aligned_free(p); }

static int cpu_copy(void *st, float *dst, const float *src, size_t n,
                    char *err, size_t errsz) {
    (void)st; (void)err; (void)errsz;
    if (dst != src) memmove(dst, src, n * sizeof(float));
    return 0;
}

/* ── weights: zero-copy, the mmap'd bytes ARE the weight ───────────────── */

static int cpu_weight_upload(void *st, mynah_slm_bweight *w, char *err, size_t errsz) {
    cpu_state *s = st;
    if (!ingot_type_can_dequant(w->type)) return 1;
    if (w->cols > s->strip_cols) {
        float *grown = mynah_slm_aligned_alloc((size_t)MYNAH_SLM_STRIP_ROWS * w->cols * sizeof(float));
        if (!grown) { set_err(err, errsz, "out of memory for the dequantization strip"); return -1; }
        mynah_slm_aligned_free(s->strip);
        s->strip = grown;
        s->strip_cols = w->cols;
    }
    w->data = (void *)(uintptr_t)w->host;
    return 0;
}

/* ── products ───────────────────────────────────────────────────────────── */

typedef struct {
    const uint8_t *base;
    float         *out;
    const float   *in;
    const mynah_slm_matvec_in *prep;
    size_t         cols, row_bytes, rows_per_chunk, rows;
    int            type, rc;
} matvec_job;

static void matvec_chunk(void *ctx, int i) {
    matvec_job *j = ctx;
    const size_t first = (size_t)i * j->rows_per_chunk;
    if (first >= j->rows) return;
    size_t n = j->rows_per_chunk;
    if (first + n > j->rows) n = j->rows - first;

    const uint8_t *rows = j->base + first * j->row_bytes;
    if (j->prep &&
        mynah_slm_matvec(j->type, rows, n, j->cols, j->in, j->prep, j->out + first) == 0)
        return;
    if (ingot_matvec(j->type, rows, n, j->cols, j->in, j->out + first) != 0)
        j->rc = -1;
}

static int cpu_matvec(void *st, const mynah_slm_bweight *w, const float *x, float *y,
                      char *err, size_t errsz) {
    (void)st;
    const size_t rows = w->rows, cols = w->cols;
    const int nth = mynah_slm_threads_count();

    /* On the stack, as in mynah_slm_project: cols/32 floats of sums plus the
     * int8 copy, and a heap buffer in the token loop would cost more. */
    mynah_slm_matvec_in prep;
    const mynah_slm_matvec_in *prepared = NULL;
    if (mynah_slm_matvec_have(w->type) && cols % 256 == 0 &&
        cols / 32 <= MYNAH_SLM_XSUM_MAX) {
        mynah_slm_matvec_prepare(x, cols, &prep);
        prepared = &prep;
    }

    if (nth <= 1 || rows < 64) {
        if (prepared && mynah_slm_matvec(w->type, w->data, rows, cols, x, prepared, y) == 0)
            return 0;
        if (ingot_matvec(w->type, w->data, rows, cols, x, y) != 0) {
            set_err(err, errsz, "cpu matvec failed");
            return -1;
        }
        return 0;
    }

    int chunks = nth * 4;
    if ((size_t)chunks > rows) chunks = (int)rows;
    matvec_job j = {
        .base = w->data, .out = y, .in = x, .prep = prepared,
        .cols = cols, .row_bytes = w->row_bytes,
        .rows_per_chunk = (rows + (size_t)chunks - 1) / (size_t)chunks,
        .rows = rows, .type = w->type, .rc = 0,
    };
    mynah_slm_parallel_for(chunks, matvec_chunk, &j);
    if (j.rc != 0) set_err(err, errsz, "cpu matvec failed");
    return j.rc;
}

static int cpu_matmat(void *st, const mynah_slm_bweight *w, const float *x, float *y,
                      size_t tokens, char *err, size_t errsz) {
    cpu_state *s = st;
    if (mynah_slm_qmatmat(w->type, w->data, w->rows, w->cols, x, y, tokens, s->strip) != 0) {
        set_err(err, errsz, "cpu matmat failed");
        return -1;
    }
    return 0;
}

/* One stored row decoded straight into the output, as arch_qwen3.c's
 * embed_row() does: a matvec against a one-hot would walk 151936 rows. */
static int cpu_embed(void *st, const mynah_slm_bweight *w, const uint32_t *tokens,
                     size_t n, float *out, char *err, size_t errsz) {
    (void)st;
    const uint8_t *base = w->data;
    for (size_t i = 0; i < n; i++) {
        if (ingot_dequant_matrix(w->type, base + (size_t)tokens[i] * w->row_bytes, 1,
                                 w->cols, out + i * w->cols) != 0) {
            set_err(err, errsz, "cpu embed: row decode failed");
            return -1;
        }
    }
    return 0;
}

/* ── norms and elementwise ──────────────────────────────────────────────── */

static int cpu_rms_norm(void *st, float *out, const float *x, const mynah_slm_bweight *w,
                        size_t rows, uint32_t dim, float eps, char *err, size_t errsz) {
    (void)st; (void)err; (void)errsz;
    for (size_t r = 0; r < rows; r++)
        mynah_slm_rms_norm(out + r * dim, x + r * dim, w->data, dim, eps);
    return 0;
}

static int cpu_rms_norm_heads(void *st, float *x, const mynah_slm_bweight *w,
                              size_t n_heads, uint32_t head_dim, float eps,
                              char *err, size_t errsz) {
    (void)st; (void)err; (void)errsz;
    /* Heads are independent, so one call over all of them is the same as one
     * call per token — the shape arch_qwen3.c uses. */
    if (n_heads > UINT32_MAX) return 1;
    mynah_slm_rms_norm_per_head(x, w->data, (uint32_t)n_heads, head_dim, eps);
    return 0;
}

static int cpu_swiglu(void *st, float *gate, const float *up, size_t n,
                      char *err, size_t errsz) {
    (void)st; (void)err; (void)errsz;
    mynah_slm_swiglu(gate, up, n);
    return 0;
}

static int cpu_add(void *st, float *y, const float *x, size_t n, char *err, size_t errsz) {
    (void)st; (void)err; (void)errsz;
    mynah_slm_add(y, x, n);
    return 0;
}

static int cpu_add_scaled(void *st, float *y, const float *x, float w, size_t n,
                          char *err, size_t errsz) {
    (void)st; (void)err; (void)errsz;
    mynah_slm_add_scaled(y, x, w, n);
    return 0;
}

/* ── RoPE ───────────────────────────────────────────────────────────────── */

static int cpu_rope_create(void *st, mynah_slm_brope *r, char *err, size_t errsz) {
    (void)st;
    mynah_slm_rope *t = calloc(1, sizeof *t);
    if (!t || mynah_slm_rope_init(t, r->head_dim, r->max_pos, r->theta, r->interleaved) != 0) {
        free(t);
        set_err(err, errsz, "cannot build the rope table");
        return -1;
    }
    r->impl = t;
    return 0;
}

static void cpu_rope_free(void *st, mynah_slm_brope *r) {
    (void)st;
    mynah_slm_rope_free(r->impl);
    free(r->impl);
    r->impl = NULL;
}

static int cpu_rope(void *st, const mynah_slm_brope *r, float *x, size_t n_tokens,
                    uint32_t n_heads, uint32_t pos0, char *err, size_t errsz) {
    (void)st; (void)err; (void)errsz;
    const size_t width = (size_t)n_heads * r->head_dim;
    for (size_t t = 0; t < n_tokens; t++)
        mynah_slm_rope_apply(r->impl, x + t * width, n_heads, pos0 + (uint32_t)t);
    return 0;
}

/* ── KV cache and attention ─────────────────────────────────────────────── */

static void cpu_kv_release(cpu_kv *c) {
    mynah_slm_kv_free(&c->kv);
    mynah_slm_aligned_free(c->scores_mt);
    mynah_slm_aligned_free(c->bscores);
    mynah_slm_aligned_free(c->kgather);
    mynah_slm_aligned_free(c->vgather);
    free(c);
}

static int cpu_kv_create(void *st, mynah_slm_bkv *kv, char *err, size_t errsz) {
    (void)st;
    const mynah_slm_bkv_desc *d = &kv->desc;
    cpu_kv *c = calloc(1, sizeof *c);
    if (!c) { set_err(err, errsz, "out of memory for the KV cache"); return -1; }
    const size_t ctx = d->n_ctx;
    c->scores_mt = mynah_slm_aligned_alloc((size_t)d->n_heads * ctx * sizeof(float));
    c->bscores   = mynah_slm_aligned_alloc((size_t)d->max_batch * ctx * sizeof(float));
    c->kgather   = mynah_slm_aligned_alloc(ctx * d->head_dim * sizeof(float));
    c->vgather   = mynah_slm_aligned_alloc(ctx * d->head_dim * sizeof(float));
    if (!c->scores_mt || !c->bscores || !c->kgather || !c->vgather ||
        mynah_slm_kv_init(&c->kv, d->type_k, d->type_v, d->n_layers, d->n_ctx,
                          d->n_kv_heads, d->head_dim) != 0) {
        cpu_kv_release(c);
        set_err(err, errsz, "out of memory for the KV cache (or kv_dim not a multiple of 32)");
        return -1;
    }
    kv->impl = c;
    return 0;
}

static void cpu_kv_free(void *st, mynah_slm_bkv *kv) {
    (void)st;
    if (kv->impl) cpu_kv_release(kv->impl);
    kv->impl = NULL;
}

static int cpu_kv_append(void *st, mynah_slm_bkv *kv, uint32_t layer, uint32_t pos0,
                         size_t n, const float *k, const float *v, char *err, size_t errsz) {
    (void)st; (void)err; (void)errsz;
    cpu_kv *c = kv->impl;
    for (size_t t = 0; t < n; t++) {
        mynah_slm_kv_put_k(&c->kv, layer, pos0 + (uint32_t)t, k + t * kv->kv_dim);
        mynah_slm_kv_put_v(&c->kv, layer, pos0 + (uint32_t)t, v + t * kv->kv_dim);
    }
    return 0;
}

/* The same four calls arch_qwen3.c makes, chosen the same way: f32 caches go
 * through the reference kernels on raw arrays, packed ones through the fused
 * kvcache readers; one query takes the per-head path, a batch the sgemm one. */
static int cpu_attention(void *st, const mynah_slm_bkv *kv, uint32_t layer,
                         const float *q, float *out, uint32_t pos0, size_t n_q,
                         float scale, char *err, size_t errsz) {
    (void)st; (void)err; (void)errsz;
    const mynah_slm_bkv_desc *d = &kv->desc;
    cpu_kv *c = kv->impl;
    const int f32 = c->kv.type_k == MYNAH_SLM_KV_F32 && c->kv.type_v == MYNAH_SLM_KV_F32;
    const size_t per_layer = (size_t)d->n_ctx * kv->kv_dim;
    const float *kl = (const float *)c->kv.k + (size_t)layer * per_layer;
    const float *vl = (const float *)c->kv.v + (size_t)layer * per_layer;
    const uint32_t q_dim = d->n_heads * d->head_dim;

    if (n_q == 1) {
        if (f32)
            mynah_slm_attention_mt(out, q, kl, vl, pos0 + 1, d->n_heads, d->n_kv_heads,
                                   d->head_dim, scale, c->scores_mt);
        else
            mynah_slm_attention_kv_mt(out, q, &c->kv, layer, pos0 + 1, d->n_heads,
                                      d->n_kv_heads, d->head_dim, scale, c->scores_mt);
        return 0;
    }
    if (f32)
        mynah_slm_attention_batch(out, q, kl, vl, pos0, (uint32_t)n_q, d->n_heads,
                                  d->n_kv_heads, d->head_dim, q_dim, scale, c->bscores);
    else
        mynah_slm_attention_kv_batch(out, q, &c->kv, layer, pos0, (uint32_t)n_q,
                                     d->n_heads, d->n_kv_heads, d->head_dim, q_dim,
                                     scale, c->bscores, c->kgather, c->vgather);
    return 0;
}

static int cpu_argmax(void *st, const float *x, size_t n, uint32_t *idx,
                      char *err, size_t errsz) {
    (void)st; (void)err; (void)errsz;
    size_t best = 0;
    for (size_t i = 1; i < n; i++) if (x[i] > x[best]) best = i;
    *idx = (uint32_t)best;
    return 0;
}

int mynah_slm_backend_cpu_open(mynah_slm_backend_ops *ops, void **state,
                               char *err, size_t errsz) {
    cpu_state *s = calloc(1, sizeof *s);
    if (!s) { set_err(err, errsz, "out of memory creating the cpu backend"); return -1; }
    memset(ops, 0, sizeof *ops);
    ops->name           = "cpu";
    ops->close          = cpu_close;
    ops->alloc          = cpu_alloc;
    ops->free           = cpu_free;
    ops->h2d            = cpu_copy;
    ops->d2h            = cpu_copy;
    ops->sync           = NULL;          /* synchronous: nothing to wait for */
    ops->weight_upload  = cpu_weight_upload;
    ops->weight_release = NULL;          /* zero-copy: nothing to release */
    ops->matvec         = cpu_matvec;
    ops->matmat         = cpu_matmat;
    ops->embed          = cpu_embed;
    ops->rms_norm       = cpu_rms_norm;
    ops->rms_norm_heads = cpu_rms_norm_heads;
    ops->swiglu         = cpu_swiglu;
    ops->add            = cpu_add;
    ops->add_scaled     = cpu_add_scaled;
    ops->rope_create    = cpu_rope_create;
    ops->rope_free      = cpu_rope_free;
    ops->rope           = cpu_rope;
    ops->kv_create      = cpu_kv_create;
    ops->kv_free        = cpu_kv_free;
    ops->kv_append      = cpu_kv_append;
    ops->attention      = cpu_attention;
    ops->argmax         = cpu_argmax;
    *state = s;
    return 0;
}
