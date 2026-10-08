/* backend.c — open/close, the weight cache, and the checked wrappers.
 *
 * Everything that is the same for every backend lives here, once: argument
 * validation, the NULL-means-unsupported rule, and the upload-once weight
 * cache. The backends themselves (backend_cpu.c, gpu/cuda/backend_cuda.cu)
 * only do arithmetic on arguments that have already been checked.
 *
 * SPDX-License-Identifier: MIT */
#include "backend_ops.h"

#include "ingot/dtype.h"
#include "ingot/quant.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct mynah_slm_backend {
    mynah_slm_device      device;
    mynah_slm_backend_ops ops;
    void                 *state;

    /* Every weight uploaded so far, searched linearly by host pointer. Only
     * ever touched at model load (a Qwen3 has ~300 tensors), never per token. */
    mynah_slm_bweight **weights;
    size_t              n_weights, cap_weights;
};

static void set_err(char *err, size_t errsz, const char *msg) {
    if (err && errsz > 0) snprintf(err, errsz, "%s", msg);
}

const char *mynah_slm_device_name(mynah_slm_device d) {
    switch (d) {
        case MYNAH_SLM_DEVICE_CPU:  return "cpu";
        case MYNAH_SLM_DEVICE_CUDA: return "cuda";
    }
    return "unknown";
}

const char *mynah_slm_bop_name(mynah_slm_bop op) {
    static const char *const names[MYNAH_SLM_BOP_COUNT] = {
        "matvec", "matmat", "embed", "rms_norm", "rms_norm_heads", "rope",
        "swiglu", "add", "add_scaled", "kv_append", "attention", "argmax",
    };
    return ((unsigned)op < MYNAH_SLM_BOP_COUNT) ? names[op] : "unknown";
}

int mynah_slm_backend_open(mynah_slm_device device, mynah_slm_backend **out,
                           char *err, size_t errsz) {
    if (!out) { set_err(err, errsz, "invalid backend output pointer"); return -1; }
    *out = NULL;

    mynah_slm_backend *b = calloc(1, sizeof *b);
    if (!b) { set_err(err, errsz, "out of memory creating the backend"); return -1; }
    b->device = device;

    int rc;
    switch (device) {
        case MYNAH_SLM_DEVICE_CPU:
            rc = mynah_slm_backend_cpu_open(&b->ops, &b->state, err, errsz);
            break;
        case MYNAH_SLM_DEVICE_CUDA:
#if defined(MYNAH_SLM_ENABLE_CUDA)
            rc = mynah_slm_backend_cuda_open(&b->ops, &b->state, err, errsz);
#else
            set_err(err, errsz, "CUDA backend is not compiled; build with `make cuda`");
            rc = -1;
#endif
            break;
        default:
            set_err(err, errsz, "unknown backend device");
            rc = -1;
            break;
    }
    if (rc != 0) { free(b); return -1; }
    if (err && errsz > 0) err[0] = '\0';
    *out = b;
    return 0;
}

void mynah_slm_backend_close(mynah_slm_backend *b) {
    if (!b) return;
    for (size_t i = 0; i < b->n_weights; i++) {
        if (b->ops.weight_release) b->ops.weight_release(b->state, b->weights[i]);
        free(b->weights[i]);
    }
    free(b->weights);
    if (b->ops.close) b->ops.close(b->state);
    free(b);
}

const char *mynah_slm_backend_name(const mynah_slm_backend *b) {
    return (b && b->ops.name) ? b->ops.name : "unknown";
}

mynah_slm_device mynah_slm_backend_device(const mynah_slm_backend *b) {
    return b ? b->device : MYNAH_SLM_DEVICE_CPU;
}

const mynah_slm_backend_ops *mynah_slm_backend_ops_of(const mynah_slm_backend *b,
                                                      void **state) {
    if (!b) return NULL;
    if (state) *state = b->state;
    return &b->ops;
}

int mynah_slm_backend_recover(mynah_slm_backend *b, char *err, size_t errsz) {
    if (!b) { set_err(err, errsz, "invalid backend"); return -1; }
    if (!b->ops.recover) return 0;
    return b->ops.recover(b->state, err, errsz);
}

int mynah_slm_backend_has(const mynah_slm_backend *b, mynah_slm_bop op) {
    if (!b) return 0;
    const mynah_slm_backend_ops *o = &b->ops;
    switch (op) {
        case MYNAH_SLM_BOP_MATVEC:         return o->matvec != NULL;
        case MYNAH_SLM_BOP_MATMAT:         return o->matmat != NULL;
        case MYNAH_SLM_BOP_EMBED:          return o->embed != NULL;
        case MYNAH_SLM_BOP_RMS_NORM:       return o->rms_norm != NULL;
        case MYNAH_SLM_BOP_RMS_NORM_HEADS: return o->rms_norm_heads != NULL;
        case MYNAH_SLM_BOP_ROPE:           return o->rope != NULL && o->rope_create != NULL;
        case MYNAH_SLM_BOP_SWIGLU:         return o->swiglu != NULL;
        case MYNAH_SLM_BOP_ADD:            return o->add != NULL;
        case MYNAH_SLM_BOP_ADD_SCALED:     return o->add_scaled != NULL;
        case MYNAH_SLM_BOP_KV_APPEND:      return o->kv_append != NULL && o->kv_create != NULL;
        case MYNAH_SLM_BOP_ATTENTION:      return o->attention != NULL && o->kv_create != NULL;
        case MYNAH_SLM_BOP_ARGMAX:         return o->argmax != NULL;
        case MYNAH_SLM_BOP_COUNT:          break;
    }
    return 0;
}

/* ── memory ─────────────────────────────────────────────────────────────── */

float *mynah_slm_backend_alloc(mynah_slm_backend *b, size_t n, char *err, size_t errsz) {
    if (!b || n == 0 || n > SIZE_MAX / sizeof(float)) {
        set_err(err, errsz, "invalid backend allocation");
        return NULL;
    }
    if (!b->ops.alloc) { set_err(err, errsz, "backend cannot allocate"); return NULL; }
    return b->ops.alloc(b->state, n, err, errsz);
}

void mynah_slm_backend_free(mynah_slm_backend *b, float *p) {
    if (b && p && b->ops.free) b->ops.free(b->state, p);
}

int mynah_slm_backend_h2d(mynah_slm_backend *b, float *dst, const float *src,
                          size_t n, char *err, size_t errsz) {
    if (!b || !dst || !src) { set_err(err, errsz, "invalid h2d"); return -1; }
    if (!b->ops.h2d) return 1;
    return n ? b->ops.h2d(b->state, dst, src, n, err, errsz) : 0;
}

int mynah_slm_backend_d2h(mynah_slm_backend *b, float *dst, const float *src,
                          size_t n, char *err, size_t errsz) {
    if (!b || !dst || !src) { set_err(err, errsz, "invalid d2h"); return -1; }
    if (!b->ops.d2h) return 1;
    return n ? b->ops.d2h(b->state, dst, src, n, err, errsz) : 0;
}

int mynah_slm_backend_sync(mynah_slm_backend *b, char *err, size_t errsz) {
    if (!b) { set_err(err, errsz, "invalid backend"); return -1; }
    if (!b->ops.sync) return 0;       /* a synchronous backend has nothing to wait for */
    return b->ops.sync(b->state, err, errsz);
}

/* ── weights ────────────────────────────────────────────────────────────── */

int mynah_slm_backend_weight(mynah_slm_backend *b, int type, const void *data,
                             size_t rows, size_t cols,
                             const mynah_slm_bweight **out, char *err, size_t errsz) {
    if (out) *out = NULL;
    if (!b || !data || !out || rows == 0 || cols == 0) {
        set_err(err, errsz, "invalid weight upload");
        return -1;
    }
    if (!b->ops.weight_upload) return 1;

    for (size_t i = 0; i < b->n_weights; i++) {
        const mynah_slm_bweight *w = b->weights[i];
        if (w->host == data && w->type == type && w->rows == rows && w->cols == cols) {
            *out = w;
            return 0;
        }
    }

    uint64_t block_elems = 0, block_bytes = 0;
    if (ingot_type_geometry(type, &block_elems, &block_bytes) != 0 ||
        block_elems == 0 || cols % block_elems != 0)
        return 1;                     /* not a block format we can index by row */

    if (b->n_weights == b->cap_weights) {
        const size_t cap = b->cap_weights ? b->cap_weights * 2 : 64;
        mynah_slm_bweight **grown = realloc(b->weights, cap * sizeof *grown);
        if (!grown) { set_err(err, errsz, "out of memory in the weight cache"); return -1; }
        b->weights = grown;
        b->cap_weights = cap;
    }
    mynah_slm_bweight *w = calloc(1, sizeof *w);
    if (!w) { set_err(err, errsz, "out of memory in the weight cache"); return -1; }
    w->type      = type;
    w->rows      = rows;
    w->cols      = cols;
    w->row_bytes = (cols / (size_t)block_elems) * (size_t)block_bytes;
    w->host      = data;

    const int rc = b->ops.weight_upload(b->state, w, err, errsz);
    if (rc != 0) { free(w); return rc; }
    b->weights[b->n_weights++] = w;
    *out = w;
    return 0;
}

int    mynah_slm_bweight_type(const mynah_slm_bweight *w) { return w ? w->type : -1; }
size_t mynah_slm_bweight_rows(const mynah_slm_bweight *w) { return w ? w->rows : 0; }
size_t mynah_slm_bweight_cols(const mynah_slm_bweight *w) { return w ? w->cols : 0; }

/* ── products ───────────────────────────────────────────────────────────── */

int mynah_slm_backend_matvec(mynah_slm_backend *b, const mynah_slm_bweight *w,
                             const float *x, float *y, char *err, size_t errsz) {
    if (!b || !w || !x || !y) { set_err(err, errsz, "invalid matvec"); return -1; }
    if (!b->ops.matvec) return 1;
    return b->ops.matvec(b->state, w, x, y, err, errsz);
}

int mynah_slm_backend_matmat(mynah_slm_backend *b, const mynah_slm_bweight *w,
                             const float *x, float *y, size_t tokens,
                             char *err, size_t errsz) {
    if (!b || !w || !x || !y || tokens == 0) { set_err(err, errsz, "invalid matmat"); return -1; }
    if (!b->ops.matmat) return 1;
    return b->ops.matmat(b->state, w, x, y, tokens, err, errsz);
}

int mynah_slm_backend_embed(mynah_slm_backend *b, const mynah_slm_bweight *w,
                            const uint32_t *tokens, size_t n, float *out,
                            char *err, size_t errsz) {
    if (!b || !w || !tokens || !out || n == 0) { set_err(err, errsz, "invalid embed"); return -1; }
    for (size_t i = 0; i < n; i++)
        if (tokens[i] >= w->rows) { set_err(err, errsz, "embed: token id out of range"); return -1; }
    if (!b->ops.embed) return 1;
    return b->ops.embed(b->state, w, tokens, n, out, err, errsz);
}

/* ── norms and elementwise ──────────────────────────────────────────────── */

/* A norm gain is an F32 vector: rows = 1, cols = the width it scales. */
static int is_gain(const mynah_slm_bweight *w, uint32_t width) {
    return w && w->type == 0 /* F32 */ && w->rows == 1 && w->cols == width;
}

int mynah_slm_backend_rms_norm(mynah_slm_backend *b, float *out, const float *x,
                               const mynah_slm_bweight *w, size_t rows,
                               uint32_t dim, float eps, char *err, size_t errsz) {
    if (!b || !out || !x || rows == 0 || dim == 0 || !is_gain(w, dim)) {
        set_err(err, errsz, "invalid rms_norm (the gain must be F32, 1 x dim)");
        return -1;
    }
    if (!b->ops.rms_norm) return 1;
    return b->ops.rms_norm(b->state, out, x, w, rows, dim, eps, err, errsz);
}

int mynah_slm_backend_rms_norm_heads(mynah_slm_backend *b, float *x,
                                     const mynah_slm_bweight *w, size_t n_heads,
                                     uint32_t head_dim, float eps,
                                     char *err, size_t errsz) {
    if (!b || !x || n_heads == 0 || head_dim == 0 || !is_gain(w, head_dim)) {
        set_err(err, errsz, "invalid rms_norm_heads (the gain must be F32, 1 x head_dim)");
        return -1;
    }
    if (!b->ops.rms_norm_heads) return 1;
    return b->ops.rms_norm_heads(b->state, x, w, n_heads, head_dim, eps, err, errsz);
}

int mynah_slm_backend_swiglu(mynah_slm_backend *b, float *gate, const float *up,
                             size_t n, char *err, size_t errsz) {
    if (!b || !gate || !up) { set_err(err, errsz, "invalid swiglu"); return -1; }
    if (!b->ops.swiglu) return 1;
    return n ? b->ops.swiglu(b->state, gate, up, n, err, errsz) : 0;
}

int mynah_slm_backend_add(mynah_slm_backend *b, float *y, const float *x,
                          size_t n, char *err, size_t errsz) {
    if (!b || !y || !x) { set_err(err, errsz, "invalid add"); return -1; }
    if (!b->ops.add) return 1;
    return n ? b->ops.add(b->state, y, x, n, err, errsz) : 0;
}

int mynah_slm_backend_add_scaled(mynah_slm_backend *b, float *y, const float *x,
                                 float w, size_t n, char *err, size_t errsz) {
    if (!b || !y || !x) { set_err(err, errsz, "invalid add_scaled"); return -1; }
    if (!b->ops.add_scaled) return 1;
    return n ? b->ops.add_scaled(b->state, y, x, w, n, err, errsz) : 0;
}

/* ── RoPE ───────────────────────────────────────────────────────────────── */

int mynah_slm_backend_rope_create(mynah_slm_backend *b, uint32_t head_dim,
                                  uint32_t max_pos, float theta, int interleaved,
                                  mynah_slm_brope **out, char *err, size_t errsz) {
    if (out) *out = NULL;
    if (!b || !out || head_dim == 0 || head_dim % 2 != 0 || max_pos == 0 ||
        !(theta > 0.0f)) {
        set_err(err, errsz, "invalid rope table (head_dim must be even, theta > 0)");
        return -1;
    }
    if (!b->ops.rope_create || !b->ops.rope) return 1;
    mynah_slm_brope *r = calloc(1, sizeof *r);
    if (!r) { set_err(err, errsz, "out of memory for the rope table"); return -1; }
    r->head_dim = head_dim;
    r->max_pos = max_pos;
    r->theta = theta;
    r->interleaved = interleaved ? 1 : 0;
    const int rc = b->ops.rope_create(b->state, r, err, errsz);
    if (rc != 0) { free(r); return rc; }
    *out = r;
    return 0;
}

void mynah_slm_backend_rope_free(mynah_slm_backend *b, mynah_slm_brope *r) {
    if (!b || !r) return;
    if (b->ops.rope_free) b->ops.rope_free(b->state, r);
    free(r);
}

int mynah_slm_backend_rope(mynah_slm_backend *b, const mynah_slm_brope *r,
                           float *x, size_t n_tokens, uint32_t n_heads,
                           uint32_t pos0, char *err, size_t errsz) {
    if (!b || !r || !x || n_heads == 0 ||
        (uint64_t)pos0 + n_tokens > (uint64_t)r->max_pos) {
        set_err(err, errsz, "invalid rope (position beyond the table)");
        return -1;
    }
    if (!b->ops.rope) return 1;
    return n_tokens ? b->ops.rope(b->state, r, x, n_tokens, n_heads, pos0, err, errsz) : 0;
}

/* ── KV cache and attention ─────────────────────────────────────────────── */

int mynah_slm_backend_kv_create(mynah_slm_backend *b, const mynah_slm_bkv_desc *d,
                                mynah_slm_bkv **out, char *err, size_t errsz) {
    if (out) *out = NULL;
    if (!b || !d || !out || d->n_layers == 0 || d->n_ctx == 0 || d->n_heads == 0 ||
        d->n_kv_heads == 0 || d->n_heads % d->n_kv_heads != 0 || d->head_dim == 0 ||
        d->max_batch == 0 || d->max_batch > d->n_ctx) {
        set_err(err, errsz, "invalid KV description (n_heads must be a multiple of n_kv_heads)");
        return -1;
    }
    if (!b->ops.kv_create) return 1;
    mynah_slm_bkv *kv = calloc(1, sizeof *kv);
    if (!kv) { set_err(err, errsz, "out of memory for the KV handle"); return -1; }
    kv->desc = *d;
    kv->kv_dim = d->n_kv_heads * d->head_dim;
    const int rc = b->ops.kv_create(b->state, kv, err, errsz);
    if (rc != 0) { free(kv); return rc; }
    *out = kv;
    return 0;
}

void mynah_slm_backend_kv_free(mynah_slm_backend *b, mynah_slm_bkv *kv) {
    if (!b || !kv) return;
    if (b->ops.kv_free) b->ops.kv_free(b->state, kv);
    free(kv);
}

int mynah_slm_backend_kv_append(mynah_slm_backend *b, mynah_slm_bkv *kv,
                                uint32_t layer, uint32_t pos0, size_t n,
                                const float *k, const float *v,
                                char *err, size_t errsz) {
    if (!b || !kv || !k || !v || layer >= kv->desc.n_layers ||
        (uint64_t)pos0 + n > (uint64_t)kv->desc.n_ctx) {
        set_err(err, errsz, "invalid kv_append (layer or position out of range)");
        return -1;
    }
    if (!b->ops.kv_append) return 1;
    return n ? b->ops.kv_append(b->state, kv, layer, pos0, n, k, v, err, errsz) : 0;
}

int mynah_slm_backend_attention(mynah_slm_backend *b, const mynah_slm_bkv *kv,
                                uint32_t layer, const float *q, float *out,
                                uint32_t pos0, size_t n_q, float scale,
                                char *err, size_t errsz) {
    if (!b || !kv || !q || !out || layer >= kv->desc.n_layers || n_q == 0 ||
        n_q > kv->desc.max_batch || (uint64_t)pos0 + n_q > (uint64_t)kv->desc.n_ctx) {
        set_err(err, errsz, "invalid attention (layer, batch or position out of range)");
        return -1;
    }
    if (!b->ops.attention) return 1;
    return b->ops.attention(b->state, kv, layer, q, out, pos0, n_q, scale, err, errsz);
}

/* ── sampling hand-off ──────────────────────────────────────────────────── */

int mynah_slm_backend_argmax(mynah_slm_backend *b, const float *x, size_t n,
                             uint32_t *idx, char *err, size_t errsz) {
    if (!b || !x || !idx || n == 0 || n > UINT32_MAX) {
        set_err(err, errsz, "invalid argmax");
        return -1;
    }
    if (!b->ops.argmax) return 1;
    return b->ops.argmax(b->state, x, n, idx, err, errsz);
}
