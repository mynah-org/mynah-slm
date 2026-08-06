/* model.c — open a checkpoint, extract the config, bind every tensor.
 *
 * Two jobs, both boring on purpose:
 *
 *   1. read the architecture config out of GGUF metadata, with the key prefix
 *      taken from general.architecture so this file does not know the string
 *      "qwen3";
 *   2. resolve the tensor names into descriptors, once, so the forward pass
 *      never does a lookup or a strcmp.
 *
 * A missing required tensor is an error naming the tensor. A missing metadata
 * key is an error naming the key. Nothing here guesses a default, because a
 * guessed head_dim produces a model that runs and is wrong.
 *
 * SPDX-License-Identifier: MIT */
#include "model.h"

#include "mynah_slm.h"

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Errors are returned as text in the caller's buffer; nothing here writes to
 * stderr. Always returns -1 so call sites can `return fail(...)`. */
static int fail(char *err, size_t errsz, const char *fmt, ...) {
    if (err && errsz) {
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(err, errsz, fmt, ap);
        va_end(ap);
    }
    return -1;
}

/* ── metadata helpers ──────────────────────────────────────────────────────
 * Each returns 0 on success. They take the arch prefix rather than a full key
 * so the call sites read like the GGUF spec. */

static int kv_u32(const ingot_gguf *g, const char *arch, const char *suffix,
                  uint32_t *out, char *err, size_t errsz) {
    char key[128];
    snprintf(key, sizeof key, "%s.%s", arch, suffix);
    const ingot_kv *kv = ingot_gguf_kv_find(g, key);
    if (!kv) return fail(err, errsz, "missing metadata key: %s", key);

    uint64_t v;
    if (ingot_kv_u64(kv, &v) != 0) return fail(err, errsz, "%s is not an integer", key);
    *out = (uint32_t)v;
    return 0;
}

static int kv_f32(const ingot_gguf *g, const char *arch, const char *suffix,
                  float *out, char *err, size_t errsz) {
    char key[128];
    snprintf(key, sizeof key, "%s.%s", arch, suffix);
    const ingot_kv *kv = ingot_gguf_kv_find(g, key);
    if (!kv) return fail(err, errsz, "missing metadata key: %s", key);

    /* Deliberately _f64 and only _f64. ingot's integer accessors will happily
     * convert a FLOAT32 KV, so reading rms_norm_eps through _u64 returns 0
     * without an error — a plausible value, silently wrong. Same trap that hit
     * inspect --meta. */
    double v;
    if (ingot_kv_f64(kv, &v) != 0) return fail(err, errsz, "%s is not a float", key);
    *out = (float)v;
    return 0;
}

static int load_config(mynah_slm_config *c, const ingot_gguf *g,
                       char *err, size_t errsz) {
    memset(c, 0, sizeof *c);

    const char *arch = ingot_gguf_arch(g);
    if (!arch || !*arch)
        return fail(err, errsz, "no general.architecture in this file");
    if (strlen(arch) >= sizeof c->arch)
        return fail(err, errsz, "architecture name too long: %s", arch);
    strcpy(c->arch, arch);

    if (kv_u32(g, arch, "block_count",           &c->n_layers,   err, errsz) ||
        kv_u32(g, arch, "embedding_length",      &c->d_model,    err, errsz) ||
        kv_u32(g, arch, "feed_forward_length",   &c->d_ff,       err, errsz) ||
        kv_u32(g, arch, "attention.head_count",  &c->n_heads,    err, errsz) ||
        kv_u32(g, arch, "attention.head_count_kv", &c->n_kv_heads, err, errsz) ||
        kv_u32(g, arch, "attention.key_length",  &c->head_dim,   err, errsz) ||
        kv_u32(g, arch, "context_length",        &c->n_ctx,      err, errsz) ||
        kv_f32(g, arch, "attention.layer_norm_rms_epsilon", &c->rms_eps, err, errsz) ||
        kv_f32(g, arch, "rope.freq_base",        &c->rope_theta, err, errsz))
        return -1;

    if (c->n_heads == 0 || c->n_kv_heads == 0)
        return fail(err, errsz, "head counts must be non-zero");
    if (c->n_heads % c->n_kv_heads != 0)
        return fail(err, errsz, "n_heads (%u) is not a multiple of n_kv_heads (%u): "
                                "GQA groups would not be whole",
                    c->n_heads, c->n_kv_heads);

    /* value_length is allowed to differ from key_length in principle. Nothing
     * we support does that, so refuse rather than silently use one for both. */
    uint32_t v_len = 0;
    if (kv_u32(g, arch, "attention.value_length", &v_len, err, errsz) == 0 &&
        v_len != c->head_dim)
        return fail(err, errsz, "value_length (%u) != key_length (%u), unsupported",
                    v_len, c->head_dim);

    const ingot_kv *toks = ingot_gguf_kv_find(g, "tokenizer.ggml.tokens");
    if (toks) {
        uint64_t n = 0;
        ingot_kv_arr_len(toks, &n);
        c->vocab_size = (uint32_t)n;
    }

    const ingot_kv *eos = ingot_gguf_kv_find(g, "tokenizer.ggml.eos_token_id");
    if (eos) {
        uint64_t v;
        if (ingot_kv_u64(eos, &v) == 0) c->eos[c->n_eos++] = (uint32_t)v;
    }

    c->q_dim  = c->n_heads    * c->head_dim;
    c->kv_dim = c->n_kv_heads * c->head_dim;
    return 0;
}

/* ── tensor binding ──────────────────────────────────────────────────────── */

/* Required: absence is an error. */
static const ingot_tensor *bind_req(const ingot_gguf *g, char *err, size_t errsz,
                                    int *rc, const char *name) {
    if (*rc) return NULL;                     /* already failed, stay quiet */
    const ingot_tensor *t = ingot_gguf_find(g, name);
    if (!t) { *rc = fail(err, errsz, "missing tensor: %s", name); return NULL; }
    return t;
}

/* Optional: absence is a legitimate answer (no QK-norm in some families). */
static const ingot_tensor *bind_opt(const ingot_gguf *g, const char *name) {
    return ingot_gguf_find(g, name);
}

static int bind_layer(mynah_slm_layer *l, const ingot_gguf *g, uint32_t i,
                      char *err, size_t errsz) {
    char n[128];
    int rc = 0;

#define REQ(field, suffix) \
    snprintf(n, sizeof n, "blk.%u." suffix, i); \
    l->field = bind_req(g, err, errsz, &rc, n)
#define OPT(field, suffix) \
    snprintf(n, sizeof n, "blk.%u." suffix, i); \
    l->field = bind_opt(g, n)

    REQ(attn_norm, "attn_norm.weight");
    REQ(wq,        "attn_q.weight");
    REQ(wk,        "attn_k.weight");
    REQ(wv,        "attn_v.weight");
    REQ(wo,        "attn_output.weight");
    REQ(ffn_norm,  "ffn_norm.weight");
    REQ(gate,      "ffn_gate.weight");
    REQ(up,        "ffn_up.weight");
    REQ(down,      "ffn_down.weight");
    OPT(q_norm,    "attn_q_norm.weight");
    OPT(k_norm,    "attn_k_norm.weight");

#undef REQ
#undef OPT
    return rc;
}

/* ── public ──────────────────────────────────────────────────────────────── */

const void *mynah_slm_tensor_data(const mynah_slm_model_t *m, const ingot_tensor *t) {
    return ingot_gguf_data(m->gguf, t);
}

mynah_slm_model_t *mynah_slm_load(const char *path, char *err, size_t errsz) {
    if (!path) { fail(err, errsz, "null path"); return NULL; }

    mynah_slm_model_t *m = calloc(1, sizeof *m);
    if (!m) { fail(err, errsz, "out of memory"); return NULL; }

    if (ingot_gguf_open_split(&m->gguf, path, err, errsz) != 0) {
        free(m);
        return NULL;
    }
    if (load_config(&m->cfg, m->gguf, err, errsz) != 0) {
        mynah_slm_free(m);
        return NULL;
    }

    int rc = 0;
    m->embed    = bind_req(m->gguf, err, errsz, &rc, "token_embd.weight");
    m->out_norm = bind_req(m->gguf, err, errsz, &rc, "output_norm.weight");
    /* Absent means tied embeddings: token_embd is also the output projection.
     * Qwen3 GGUFs are like this. The safetensors of the same model ships a
     * redundant lm_head.weight instead — see docs/qwen3-arch.md. */
    m->lm_head  = bind_opt(m->gguf, "output.weight");
    if (rc) { mynah_slm_free(m); return NULL; }

    m->layers = calloc(m->cfg.n_layers, sizeof *m->layers);
    if (!m->layers) {
        fail(err, errsz, "out of memory for %u layers", m->cfg.n_layers);
        mynah_slm_free(m);
        return NULL;
    }
    for (uint32_t i = 0; i < m->cfg.n_layers; i++) {
        if (bind_layer(&m->layers[i], m->gguf, i, err, errsz) != 0) {
            mynah_slm_free(m);
            return NULL;
        }
    }

    /* The embedding's row count is the real vocabulary, and it can exceed the
     * tokenizer array (Qwen3 pads 151669 used entries out to 151936). Trust
     * the tensor: it is what the LM head actually produces. */
    if (m->embed->rank >= 2) {
        uint64_t rows = m->embed->ne[1];
        if (m->cfg.vocab_size && rows != m->cfg.vocab_size) {
            /* Not an error — just say which one won, so a surprise later is
             * traceable. */
            m->cfg.vocab_size = (uint32_t)rows;
        } else if (!m->cfg.vocab_size) {
            m->cfg.vocab_size = (uint32_t)rows;
        }
    }
    return m;
}

void mynah_slm_free(mynah_slm_model_t *m) {
    if (!m) return;
    free(m->layers);
    if (m->gguf) ingot_gguf_close(m->gguf);
    free(m);
}

const mynah_slm_config *mynah_slm_model_config(const mynah_slm_model_t *m) {
    return m ? &m->cfg : NULL;
}

const char *mynah_slm_arch(const mynah_slm_model_t *m)      { return m ? m->cfg.arch : ""; }
uint32_t mynah_slm_n_layers(const mynah_slm_model_t *m)     { return m ? m->cfg.n_layers : 0; }
uint32_t mynah_slm_n_ctx(const mynah_slm_model_t *m)        { return m ? m->cfg.n_ctx : 0; }
uint32_t mynah_slm_vocab_size(const mynah_slm_model_t *m)   { return m ? m->cfg.vocab_size : 0; }
