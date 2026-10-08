/* forward_backend.c — see forward_backend.h.
 *
 * One core, `run_rows`, carries every shape: a single decode step (one row,
 * matvec), a prefill batch (n rows of ONE sequence at consecutive positions,
 * matmat + one causal attention call). The op order inside it is
 * arch_qwen3.c's, line for line, so the CPU backend reproduces
 * mynah_slm_seq_forward and mynah_slm_seq_forward_batch to the bit; the
 * comments name the reference line each block mirrors.
 *
 * SPDX-License-Identifier: MIT */
#include "forward_backend.h"

#include "backend_ops.h"   /* kv->desc */

#include "kernels.h"       /* mynah_slm_aligned_alloc */

#include "ingot/dtype.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const mynah_slm_bweight *attn_norm, *wq, *wk, *wv, *wo, *q_norm, *k_norm;
    const mynah_slm_bweight *ffn_norm, *gate, *up, *down;
} bf_layer;

struct mynah_slm_bfwd {
    mynah_slm_backend       *b;
    const mynah_slm_model_t *m;
    const mynah_slm_config  *c;

    bf_layer                *L;
    const mynah_slm_bweight *embed, *out_norm, *head;
    mynah_slm_brope         *rope;

    uint32_t ctx_cap, batch_max, dec_max, rows, vocab;
    mynah_slm_kv_type kv_k, kv_v;

    /* Backend buffers, [rows][width] each — device memory on a device. */
    float *x, *h, *q, *k, *v, *attn, *proj, *gate, *up;
    float *logits;     /* [max(1, dec_max)][vocab] */
    float *hlogits;    /* the same, in HOST memory: what a driver hands out */
};

static void set_err(char *err, size_t errsz, const char *msg) {
    if (err && errsz > 0) snprintf(err, errsz, "%s", msg);
}

/* An op's return folded to 0 / -1: an op the backend refuses MID-STEP (1)
 * is a failure of the step, never a quiet host detour. create() has already
 * asked for every op it will issue, so this only fires on a shape refusal. */
static int op(mynah_slm_bfwd *f, int rc, const char *what, char *err, size_t errsz) {
    if (rc == 0) return 0;
    if (rc == 1 && err && errsz > 0)
        snprintf(err, errsz, "the %s backend refused %s", mynah_slm_backend_name(f->b), what);
    return -1;
}

/* ── create ──────────────────────────────────────────────────────────────── */

static int upload(mynah_slm_bfwd *f, const ingot_tensor *t, int must_be_gain,
                  const mynah_slm_bweight **out, char *err, size_t errsz) {
    const size_t cols = (size_t)t->ne[0];
    const size_t rows = t->rank >= 2 ? (size_t)t->ne[1] : 1;
    if (t->rank > 2) {
        snprintf(err, errsz, "tensor %s has rank %u; the backend forward takes matrices",
                 t->name, t->rank);
        return 1;
    }
    if (must_be_gain && t->type != INGOT_TYPE_F32) {
        snprintf(err, errsz, "norm %s is %s; the backend forward needs F32 norms",
                 t->name, ingot_type_name(t->type));
        return 1;
    }
    const void *data = mynah_slm_tensor_data(f->m, t);
    if (!data) { snprintf(err, errsz, "tensor %s has no data", t->name); return -1; }
    const int rc = mynah_slm_backend_weight(f->b, t->type, data, rows, cols, out, err, errsz);
    if (rc == 1)
        snprintf(err, errsz, "the %s backend has no kernel for %s (%s, %zu x %zu)",
                 mynah_slm_backend_name(f->b), ingot_type_name(t->type), t->name, rows, cols);
    return rc;
}

static float *buf(mynah_slm_bfwd *f, size_t n, int *ok, char *err, size_t errsz) {
    if (!*ok) return NULL;
    float *p = mynah_slm_backend_alloc(f->b, n, err, errsz);
    if (!p) *ok = 0;
    return p;
}

int mynah_slm_bfwd_create(mynah_slm_backend *b, const mynah_slm_model_t *m,
                          const mynah_slm_bfwd_desc *d, mynah_slm_bfwd **out,
                          char *err, size_t errsz) {
    if (out) *out = NULL;
    if (!b || !m || !d || !out) { set_err(err, errsz, "invalid backend forward"); return -1; }
    const mynah_slm_config *c = &m->cfg;

    /* Scope. Each of these would be computed WRONG by the op list below
     * rather than slowly, so they are refused by name. */
    if (c->n_attn_layers != c->n_layers) {
        snprintf(err, errsz, "%s has %u short-conv layers; the backend forward runs "
                 "attention-only families", c->arch, c->n_layers - c->n_attn_layers);
        return 1;
    }
    if (c->embed_scale != 1.0f || c->logit_scale != 1.0f) {
        snprintf(err, errsz, "%s scales its embeddings or logits; the backend forward has "
                 "no op for that yet", c->arch);
        return 1;
    }
    for (int i = 0; i < MYNAH_SLM_BOP_COUNT; i++) {
        const mynah_slm_bop o = (mynah_slm_bop)i;
        if (o == MYNAH_SLM_BOP_ADD_SCALED && c->residual_scale == 1.0f) continue;
        if (!mynah_slm_backend_has(b, o)) {
            snprintf(err, errsz, "the %s backend has no %s op", mynah_slm_backend_name(b),
                     mynah_slm_bop_name(o));
            return 1;
        }
    }

    mynah_slm_bfwd *f = calloc(1, sizeof *f);
    if (!f) { set_err(err, errsz, "out of memory for the backend forward"); return -1; }
    f->b = b;
    f->m = m;
    f->c = c;
    f->ctx_cap = (d->ctx_cap == 0 || d->ctx_cap > c->n_ctx) ? c->n_ctx : d->ctx_cap;
    f->batch_max = d->batch_max ? d->batch_max : 256;
    if (f->batch_max > f->ctx_cap) f->batch_max = f->ctx_cap;
    f->dec_max = d->dec_max ? d->dec_max : 1;
    f->rows = f->batch_max > f->dec_max ? f->batch_max : f->dec_max;
    f->kv_k = d->kv_k;
    f->kv_v = d->kv_v;

    int rc = 0;
    f->L = calloc(c->n_layers, sizeof *f->L);
    if (!f->L) { set_err(err, errsz, "out of memory for the layer table"); rc = -1; goto fail; }

    /* Weights, once. The tied head is the embedding's handle: same bytes,
     * same handle (backend.c dedups by pointer), one device copy. */
    if ((rc = upload(f, m->embed, 0, &f->embed, err, errsz)) != 0) goto fail;
    if ((rc = upload(f, m->out_norm, 1, &f->out_norm, err, errsz)) != 0) goto fail;
    if ((rc = upload(f, m->lm_head ? m->lm_head : m->embed, 0, &f->head, err, errsz)) != 0)
        goto fail;
    f->vocab = (uint32_t)mynah_slm_bweight_rows(f->head);
    if (f->vocab != c->vocab_size || mynah_slm_bweight_cols(f->head) != c->d_model ||
        mynah_slm_bweight_cols(f->embed) != c->d_model) {
        snprintf(err, errsz, "embedding / head shape disagrees with the config "
                 "(%u rows vs vocab %u)", f->vocab, c->vocab_size);
        rc = -1;
        goto fail;
    }
    for (uint32_t l = 0; l < c->n_layers; l++) {
        const mynah_slm_layer *w = &m->layers[l];
        bf_layer *o = &f->L[l];
        if (!w->attn_norm || !w->wq || !w->wk || !w->wv || !w->wo || !w->ffn_norm ||
            !w->gate || !w->up || !w->down) {
            snprintf(err, errsz, "layer %u is missing a tensor the backend forward needs", l);
            rc = -1;
            goto fail;
        }
        if ((rc = upload(f, w->attn_norm, 1, &o->attn_norm, err, errsz)) != 0 ||
            (rc = upload(f, w->wq, 0, &o->wq, err, errsz)) != 0 ||
            (rc = upload(f, w->wk, 0, &o->wk, err, errsz)) != 0 ||
            (rc = upload(f, w->wv, 0, &o->wv, err, errsz)) != 0 ||
            (rc = upload(f, w->wo, 0, &o->wo, err, errsz)) != 0 ||
            (w->q_norm && (rc = upload(f, w->q_norm, 1, &o->q_norm, err, errsz)) != 0) ||
            (w->k_norm && (rc = upload(f, w->k_norm, 1, &o->k_norm, err, errsz)) != 0) ||
            (rc = upload(f, w->ffn_norm, 1, &o->ffn_norm, err, errsz)) != 0 ||
            (rc = upload(f, w->gate, 0, &o->gate, err, errsz)) != 0 ||
            (rc = upload(f, w->up, 0, &o->up, err, errsz)) != 0 ||
            (rc = upload(f, w->down, 0, &o->down, err, errsz)) != 0)
            goto fail;
    }

    /* The table the CPU reference builds (workspace_alloc): head_dim, theta
     * and the pairing from the config, positions up to ctx_cap. */
    rc = mynah_slm_backend_rope_create(b, c->head_dim, f->ctx_cap, c->rope_theta,
                                       c->rope_interleaved, &f->rope, err, errsz);
    if (rc == 1)
        snprintf(err, errsz, "the %s backend has no %s RoPE", mynah_slm_backend_name(b),
                 c->rope_interleaved ? "interleaved" : "NeoX");
    if (rc != 0) goto fail;

    /* Probe the KV precision now: a refusal belongs at load, not at the
     * first request. */
    {
        mynah_slm_bkv_desc kd;
        mynah_slm_bkv *probe = NULL;
        mynah_slm_bfwd_kv_desc(f, f->batch_max, &kd);
        rc = mynah_slm_backend_kv_create(b, &kd, &probe, err, errsz);
        if (rc == 1)
            snprintf(err, errsz, "the %s backend refuses a %s/%s KV cache at head_dim %u",
                     mynah_slm_backend_name(b), mynah_slm_kv_type_name(f->kv_k),
                     mynah_slm_kv_type_name(f->kv_v), c->head_dim);
        mynah_slm_backend_kv_free(b, probe);
        if (rc != 0) goto fail;
    }

    {
        int ok = 1;
        const size_t R = f->rows;
        f->x     = buf(f, R * c->d_model, &ok, err, errsz);
        f->h     = buf(f, R * c->d_model, &ok, err, errsz);
        f->q     = buf(f, R * c->q_dim, &ok, err, errsz);
        f->k     = buf(f, R * c->kv_dim, &ok, err, errsz);
        f->v     = buf(f, R * c->kv_dim, &ok, err, errsz);
        f->attn  = buf(f, R * c->q_dim, &ok, err, errsz);
        f->proj  = buf(f, R * c->d_model, &ok, err, errsz);
        f->gate  = buf(f, R * c->d_ff, &ok, err, errsz);
        f->up    = buf(f, R * c->d_ff, &ok, err, errsz);
        f->logits = buf(f, (size_t)f->dec_max * f->vocab, &ok, err, errsz);
        f->hlogits = mynah_slm_aligned_alloc((size_t)f->dec_max * f->vocab * sizeof(float));
        if (ok && !f->hlogits) { set_err(err, errsz, "out of memory for host logits"); ok = 0; }
        if (!ok) { rc = -1; goto fail; }
    }
    *out = f;
    return 0;

fail:
    mynah_slm_bfwd_free(f);
    return rc;
}

void mynah_slm_bfwd_free(mynah_slm_bfwd *f) {
    if (!f) return;
    /* Shutdown: wait for anything queued before releasing what it reads. */
    mynah_slm_backend_sync(f->b, NULL, 0);
    float *bufs[] = { f->x, f->h, f->q, f->k, f->v, f->attn, f->proj, f->gate, f->up,
                      f->logits };
    for (size_t i = 0; i < sizeof bufs / sizeof *bufs; i++) mynah_slm_backend_free(f->b, bufs[i]);
    mynah_slm_backend_rope_free(f->b, f->rope);
    mynah_slm_aligned_free(f->hlogits);
    /* Weight handles belong to the backend (freed at its close or flush). */
    free(f->L);
    free(f);
}

mynah_slm_backend *mynah_slm_bfwd_backend(const mynah_slm_bfwd *f) { return f ? f->b : NULL; }
uint32_t mynah_slm_bfwd_batch_max(const mynah_slm_bfwd *f) { return f ? f->batch_max : 0; }
uint32_t mynah_slm_bfwd_dec_max(const mynah_slm_bfwd *f) { return f ? f->dec_max : 0; }
uint32_t mynah_slm_bfwd_ctx_cap(const mynah_slm_bfwd *f) { return f ? f->ctx_cap : 0; }
uint32_t mynah_slm_bfwd_vocab(const mynah_slm_bfwd *f) { return f ? f->vocab : 0; }

void mynah_slm_bfwd_kv_desc(const mynah_slm_bfwd *f, uint32_t n_ctx, mynah_slm_bkv_desc *o) {
    const mynah_slm_config *c = f->c;
    if (n_ctx == 0 || n_ctx > f->ctx_cap) n_ctx = f->ctx_cap;
    memset(o, 0, sizeof *o);
    o->type_k = f->kv_k;
    o->type_v = f->kv_v;
    o->n_layers = c->n_attn_layers;
    o->n_ctx = n_ctx;
    o->n_heads = c->n_heads;
    o->n_kv_heads = c->n_kv_heads;
    o->head_dim = c->head_dim;
    o->max_batch = f->batch_max < n_ctx ? f->batch_max : n_ctx;
}

/* ── sequences ───────────────────────────────────────────────────────────── */

int mynah_slm_bseq_init(mynah_slm_bfwd *f, mynah_slm_bseq *q, uint32_t n_ctx,
                        char *err, size_t errsz) {
    if (!f || !q) { set_err(err, errsz, "invalid sequence"); return -1; }
    memset(q, 0, sizeof *q);
    mynah_slm_bkv_desc d;
    mynah_slm_bfwd_kv_desc(f, n_ctx, &d);
    const int rc = mynah_slm_backend_kv_create(f->b, &d, &q->kv, err, errsz);
    if (rc != 0) {
        if (rc == 1) snprintf(err, errsz, "the %s backend refuses this KV cache",
                              mynah_slm_backend_name(f->b));
        return -1;
    }
    q->n_ctx = d.n_ctx;
    q->owned = 1;
    return 0;
}

int mynah_slm_bseq_bind(mynah_slm_bfwd *f, mynah_slm_bseq *q, mynah_slm_bkv *kv,
                        char *err, size_t errsz) {
    if (!f || !q || !kv) { set_err(err, errsz, "invalid sequence bind"); return -1; }
    mynah_slm_bkv_desc want;
    mynah_slm_bfwd_kv_desc(f, f->ctx_cap, &want);
    /* Not memcmp of the descs: n_ctx and max_batch may be smaller than
     * ctx_cap, everything else must be this model's. */
    const mynah_slm_bkv_desc *h = &kv->desc;
    if (h->type_k != want.type_k || h->type_v != want.type_v || h->n_layers != want.n_layers ||
        h->n_heads != want.n_heads || h->n_kv_heads != want.n_kv_heads ||
        h->head_dim != want.head_dim || h->n_ctx > f->ctx_cap) {
        set_err(err, errsz, "this KV cache was not made for this model's backend forward");
        return -1;
    }
    memset(q, 0, sizeof *q);
    q->kv = kv;
    q->n_ctx = h->n_ctx;
    return 0;
}

void mynah_slm_bseq_reset(mynah_slm_bseq *q) { if (q) q->n_past = 0; }

void mynah_slm_bseq_free(mynah_slm_bfwd *f, mynah_slm_bseq *q) {
    if (!q) return;
    if (q->owned && f) mynah_slm_backend_kv_free(f->b, q->kv);
    memset(q, 0, sizeof *q);
}

/* ── the forward pass ────────────────────────────────────────────────────── */

/* n == 1: matvec (mynah_slm_project). n > 1: matmat (project_batch). */
static int project(mynah_slm_bfwd *f, const mynah_slm_bweight *w, const float *in,
                   float *out, uint32_t n, char *err, size_t errsz) {
    const int rc = n == 1 ? mynah_slm_backend_matvec(f->b, w, in, out, err, errsz)
                          : mynah_slm_backend_matmat(f->b, w, in, out, n, err, errsz);
    return op(f, rc, n == 1 ? "matvec" : "matmat", err, errsz);
}

static int residual(mynah_slm_bfwd *f, size_t n, char *err, size_t errsz) {
    const mynah_slm_config *c = f->c;
    const int rc = c->residual_scale == 1.0f
        ? mynah_slm_backend_add(f->b, f->x, f->proj, n * c->d_model, err, errsz)
        : mynah_slm_backend_add_scaled(f->b, f->x, f->proj, c->residual_scale,
                                       n * c->d_model, err, errsz);
    return op(f, rc, "the residual add", err, errsz);
}

/* n tokens of ONE sequence at positions pos0 .. pos0+n-1: a decode step at
 * n == 1, a prefill batch above. Leaves the final-normed rows in f->h and,
 * when `head` is set, the LAST row's logits in f->logits. arch_qwen3.c:
 * mynah_slm_seq_forward (n == 1) and mynah_slm_seq_forward_batch. */
static int run_rows(mynah_slm_bfwd *f, mynah_slm_bseq *s, const uint32_t *tokens,
                    uint32_t n, int head, char *err, size_t errsz) {
    mynah_slm_backend *b = f->b;
    const mynah_slm_config *c = f->c;
    const uint32_t pos0 = s->n_past;
    const float eps = c->rms_eps;

    if (op(f, mynah_slm_backend_embed(b, f->embed, tokens, n, f->x, err, errsz),
           "embed", err, errsz)) return -1;

    for (uint32_t l = 0; l < c->n_layers; l++) {
        const bf_layer *w = &f->L[l];
        const uint32_t kvl = c->op_slot[l];

        if (op(f, mynah_slm_backend_rms_norm(b, f->h, f->x, w->attn_norm, n, c->d_model, eps,
                                             err, errsz), "rms_norm", err, errsz) ||
            project(f, w->wq, f->h, f->q, n, err, errsz) ||
            project(f, w->wk, f->h, f->k, n, err, errsz) ||
            project(f, w->wv, f->h, f->v, n, err, errsz))
            return -1;

        /* QK-norm before RoPE, per head; heads are independent, so one call
         * over n * heads is the per-token loop of the reference. */
        if (w->q_norm &&
            op(f, mynah_slm_backend_rms_norm_heads(b, f->q, w->q_norm, (size_t)n * c->n_heads,
                                                   c->head_dim, eps, err, errsz),
               "rms_norm_heads", err, errsz)) return -1;
        if (w->k_norm &&
            op(f, mynah_slm_backend_rms_norm_heads(b, f->k, w->k_norm, (size_t)n * c->n_kv_heads,
                                                   c->head_dim, eps, err, errsz),
               "rms_norm_heads", err, errsz)) return -1;

        /* V is never rotated. K is stored AFTER RoPE. */
        if (op(f, mynah_slm_backend_rope(b, f->rope, f->q, n, c->n_heads, pos0, err, errsz),
               "rope", err, errsz) ||
            op(f, mynah_slm_backend_rope(b, f->rope, f->k, n, c->n_kv_heads, pos0, err, errsz),
               "rope", err, errsz) ||
            op(f, mynah_slm_backend_kv_append(b, s->kv, kvl, pos0, n, f->k, f->v, err, errsz),
               "kv_append", err, errsz) ||
            op(f, mynah_slm_backend_attention(b, s->kv, kvl, f->q, f->attn, pos0, n,
                                              c->attn_scale, err, errsz),
               "attention", err, errsz))
            return -1;

        if (project(f, w->wo, f->attn, f->proj, n, err, errsz) ||
            residual(f, n, err, errsz) ||
            op(f, mynah_slm_backend_rms_norm(b, f->h, f->x, w->ffn_norm, n, c->d_model, eps,
                                             err, errsz), "rms_norm", err, errsz) ||
            project(f, w->gate, f->h, f->gate, n, err, errsz) ||
            project(f, w->up, f->h, f->up, n, err, errsz) ||
            op(f, mynah_slm_backend_swiglu(b, f->gate, f->up, (size_t)n * c->d_ff, err, errsz),
               "swiglu", err, errsz) ||
            project(f, w->down, f->gate, f->proj, n, err, errsz) ||
            residual(f, n, err, errsz))
            return -1;
    }

    /* The final norm on every row (a dump hook sees them in the reference);
     * the LM head on the last row only, as a matvec in both shapes. */
    if (op(f, mynah_slm_backend_rms_norm(b, f->h, f->x, f->out_norm, n, c->d_model, eps,
                                         err, errsz), "rms_norm", err, errsz))
        return -1;
    if (head &&
        op(f, mynah_slm_backend_matvec(b, f->head, f->h + (size_t)(n - 1) * c->d_model,
                                       f->logits, err, errsz), "the LM head", err, errsz))
        return -1;
    return 0;
}

/* Everything checked before anything is queued: a refused call touches
 * neither the cache nor n_past. */
static int check_seq(mynah_slm_bfwd *f, const mynah_slm_bseq *q, const uint32_t *tokens,
                     uint32_t n, char *err, size_t errsz) {
    if (!f || !q || !q->kv || !tokens || n == 0) {
        set_err(err, errsz, "invalid backend forward call");
        return -1;
    }
    if (q->n_ctx > f->ctx_cap || (uint64_t)q->n_past + n > q->n_ctx) {
        set_err(err, errsz, "the sequence is full (context exhausted)");
        return -1;
    }
    if (n > q->kv->desc.max_batch || n > f->batch_max) {
        set_err(err, errsz, "prefill batch wider than the forward was sized for");
        return -1;
    }
    for (uint32_t i = 0; i < n; i++)
        if (tokens[i] >= f->vocab) { set_err(err, errsz, "token id out of range"); return -1; }
    return 0;
}

int mynah_slm_bfwd_step(mynah_slm_bfwd *f, mynah_slm_bseq *q, uint32_t token,
                        float *logits, char *err, size_t errsz) {
    if (check_seq(f, q, &token, 1, err, errsz) != 0) return -1;
    if (run_rows(f, q, &token, 1, logits != NULL, err, errsz) != 0) return -1;
    /* The step's one wait: the row comes back to the host. */
    if (logits &&
        op(f, mynah_slm_backend_d2h(f->b, logits, f->logits, f->vocab, err, errsz),
           "d2h", err, errsz)) return -1;
    q->n_past++;
    return 0;
}

int mynah_slm_bfwd_step_argmax(mynah_slm_bfwd *f, mynah_slm_bseq *q, uint32_t token,
                               uint32_t *next, char *err, size_t errsz) {
    if (!next) { set_err(err, errsz, "invalid argmax output"); return -1; }
    if (check_seq(f, q, &token, 1, err, errsz) != 0) return -1;
    if (run_rows(f, q, &token, 1, 1, err, errsz) != 0 ||
        op(f, mynah_slm_backend_argmax(f->b, f->logits, f->vocab, next, err, errsz),
           "argmax", err, errsz))
        return -1;
    q->n_past++;
    return 0;
}

int mynah_slm_bfwd_prefill(mynah_slm_bfwd *f, mynah_slm_bseq *q, const uint32_t *tokens,
                           uint32_t n, float *logits, char *err, size_t errsz) {
    if (n == 1 && tokens) return mynah_slm_bfwd_step(f, q, tokens[0], logits, err, errsz);
    if (check_seq(f, q, tokens, n, err, errsz) != 0) return -1;
    if (run_rows(f, q, tokens, n, logits != NULL, err, errsz) != 0) return -1;
    if (logits &&
        op(f, mynah_slm_backend_d2h(f->b, logits, f->logits, f->vocab, err, errsz),
           "d2h", err, errsz)) return -1;
    q->n_past += n;
    return 0;
}

/* ── as a generation driver ──────────────────────────────────────────────── */

static int drv_prefill(void *ctx, const uint32_t *tokens, uint32_t n) {
    mynah_slm_bfwd_run *r = ctx;
    return mynah_slm_bfwd_prefill(r->f, r->q, tokens, n, NULL, r->err, sizeof r->err);
}

static float *drv_step(void *ctx, uint32_t token) {
    mynah_slm_bfwd_run *r = ctx;
    return mynah_slm_bfwd_step(r->f, r->q, token, r->f->hlogits, r->err, sizeof r->err) == 0
               ? r->f->hlogits : NULL;
}

void mynah_slm_bfwd_driver(mynah_slm_bfwd_run *r, mynah_slm_gen_driver *out) {
    r->err[0] = '\0';
    out->prefill = drv_prefill;
    out->step = drv_step;
    out->batch_max = r->f ? r->f->batch_max : 1;
    out->ctx = r;
}
