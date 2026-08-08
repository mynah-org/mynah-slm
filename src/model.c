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
#include <math.h>
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

/* head_count_kv, which a family may declare as a scalar or PER LAYER, and
 * which is also where a hybrid family declares its layer map.
 *
 * Three shapes exist in the wild:
 *   scalar             every layer attends, with that many KV heads
 *   array, all equal   the same thing written per layer (Granite)
 *   array with ZEROS   a HYBRID: 0 means "this layer is not an attention layer"
 *
 * The third is LFM2 — 22 zeros and 8 eights on an irregular pattern
 * (docs/lfm2-arch.md). This used to be a uniform-only reader that refused any
 * array whose entries differed, and that was the right refusal for the code as
 * it stood: taking element 0 would have built a model with ZERO KV heads,
 * which is not a smaller model but a broken one.
 *
 * So this reads the array as the layer map it is, and keeps the old refusal
 * for the case it was really guarding: non-zero entries that DISAGREE, which
 * would be a different architecture wearing the same metadata. */
static int load_layer_map(mynah_slm_config *c, const ingot_gguf *g,
                          const char *arch, char *err, size_t errsz) {
    char key[128];
    snprintf(key, sizeof key, "%s.attention.head_count_kv", arch);
    const ingot_kv *kv = ingot_gguf_kv_find(g, key);
    if (!kv) return fail(err, errsz, "missing metadata key: %s", key);

    c->layer_op = calloc(c->n_layers, sizeof *c->layer_op);
    c->op_slot  = calloc(c->n_layers, sizeof *c->op_slot);
    if (!c->layer_op || !c->op_slot)
        return fail(err, errsz, "out of memory for the %u-layer map", c->n_layers);

    uint64_t scalar;
    uint64_t n = 0;
    if (ingot_kv_u64(kv, &scalar) == 0) {
        c->n_kv_heads = (uint32_t)scalar;          /* homogeneous, the usual case */
    } else if (ingot_kv_arr_len(kv, &n) == 0 && n > 0) {
        if (n != c->n_layers)
            return fail(err, errsz, "%s has %llu entries for %u layers",
                        key, (unsigned long long)n, c->n_layers);
        for (uint64_t i = 0; i < n; i++) {
            int64_t e;
            if (ingot_kv_arr_i64(kv, i, &e) != 0)
                return fail(err, errsz, "%s[%llu] is not an integer",
                            key, (unsigned long long)i);
            if (e == 0) { c->layer_op[i] = MYNAH_SLM_OP_SHORTCONV; continue; }
            if (c->n_kv_heads == 0) c->n_kv_heads = (uint32_t)e;
            else if ((uint32_t)e != c->n_kv_heads)
                return fail(err, errsz,
                            "%s disagrees between attention layers (%u vs %u at "
                            "%llu) - unsupported",
                            key, c->n_kv_heads, (uint32_t)e,
                            (unsigned long long)i);
        }
        if (c->n_kv_heads == 0)
            return fail(err, errsz, "%s is all zeros: no attention layer anywhere", key);
    } else {
        return fail(err, errsz, "%s is neither an integer nor an array", key);
    }

    /* Number each kind separately: 8 KV caches and 22 conv states for LFM2's
     * 30 layers, rather than 30 of each with two thirds unused. */
    uint32_t n_conv = 0;
    for (uint32_t i = 0; i < c->n_layers; i++)
        c->op_slot[i] = (c->layer_op[i] == MYNAH_SLM_OP_ATTN)
                      ? c->n_attn_layers++ : n_conv++;
    return 0;
}

/* Optional float with a default: absent is not an error, it is "this family
 * does not use it". */
static void kv_f32_opt(const ingot_gguf *g, const char *arch, const char *suffix,
                       float *out, float dflt) {
    char key[128];
    snprintf(key, sizeof key, "%s.%s", arch, suffix);
    const ingot_kv *kv = ingot_gguf_kv_find(g, key);
    double v;
    *out = (kv && ingot_kv_f64(kv, &v) == 0) ? (float)v : dflt;
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
        kv_u32(g, arch, "context_length",        &c->n_ctx,      err, errsz) ||
        kv_f32(g, arch, "attention.layer_norm_rms_epsilon", &c->rms_eps, err, errsz) ||
        kv_f32(g, arch, "rope.freq_base",        &c->rope_theta, err, errsz))
        return -1;

    if (load_layer_map(c, g, arch, err, errsz) != 0)
        return -1;

    /* The FIR length for the short-conv layers. Optional: absent means this
     * family has none, and load_layer_map will have marked every layer as
     * attention anyway. */
    kv_u32(g, arch, "shortconv.l_cache", &c->conv_taps, NULL, 0);

    /* head_dim has its OWN key and is never d_model / n_heads — for Qwen3-0.6B
     * those give 128 and 64, and deriving it is the classic way to get that
     * family wrong. Two families spell the key differently, so try both and
     * fail rather than fall back to the derivation:
     *   attention.key_length     Qwen3, Gemma
     *   rope.dimension_count     Granite (whose rotary covers the whole head) */
    if (kv_u32(g, arch, "attention.key_length", &c->head_dim, NULL, 0) != 0 &&
        kv_u32(g, arch, "rope.dimension_count", &c->head_dim, NULL, 0) != 0) {
        /* LFM2 publishes NEITHER key. Deriving is safe there and only there,
         * because its projections are square (attn_q is [2048, 2048]) — which
         * is precisely the assumption Qwen3 breaks. Keep the fallback per
         * architecture: making it global would turn the Qwen3 trap back into a
         * silently wrong model instead of a clear error.
         *
         * Two independent confirmations that 64 is right for LFM2.5-2.6B:
         * attn_q_norm is [64] (a per-head norm, so its length IS head_dim) and
         * attn_k is [512, 2048] over 8 KV heads. */
        static const char *const DERIVE_HEAD_DIM[] = { "lfm2", NULL };
        int may_derive = 0;
        for (size_t i = 0; DERIVE_HEAD_DIM[i]; i++)
            if (strcmp(arch, DERIVE_HEAD_DIM[i]) == 0) may_derive = 1;

        if (!may_derive || c->n_heads == 0 || c->d_model % c->n_heads != 0)
            return fail(err, errsz,
                        "no head dimension: neither %s.attention.key_length nor "
                        "%s.rope.dimension_count is present", arch, arch);
        c->head_dim = c->d_model / c->n_heads;
    }

    /* Which RoPE pairing the stored weights expect. There is no metadata key
     * for it — llama.cpp carries the same knowledge as a per-architecture
     * table, because it is a property of how the converter wrote the file.
     * Ours is explicit and small, and the default is NeoX. */
    static const char *const INTERLEAVED[] = { "granite", "llama", NULL };
    for (size_t i = 0; INTERLEAVED[i]; i++)
        if (strcmp(arch, INTERLEAVED[i]) == 0) c->rope_interleaved = 1;

    /* muP scalars. Absent means "this family does not use them", so the
     * defaults are the neutral values and one forward pass serves both. */
    kv_f32_opt(g, arch, "attention.scale",  &c->attn_scale,     0.0f);
    kv_f32_opt(g, arch, "embedding_scale",  &c->embed_scale,    1.0f);
    kv_f32_opt(g, arch, "residual_scale",   &c->residual_scale, 1.0f);
    kv_f32_opt(g, arch, "logit_scale",      &c->logit_scale,    1.0f);
    if (c->attn_scale <= 0.0f)
        c->attn_scale = 1.0f / sqrtf((float)c->head_dim);
    if (c->logit_scale == 0.0f) c->logit_scale = 1.0f;

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
                      mynah_slm_op op, char *err, size_t errsz) {
    char n[128];
    int rc = 0;

#define REQ(field, suffix) \
    snprintf(n, sizeof n, "blk.%u." suffix, i); \
    l->field = bind_req(g, err, errsz, &rc, n)
#define OPT(field, suffix) \
    snprintf(n, sizeof n, "blk.%u." suffix, i); \
    l->field = bind_opt(g, n)

    /* Named attn_norm on every layer, conv ones included — see model.h. */
    REQ(attn_norm, "attn_norm.weight");
    REQ(ffn_norm,  "ffn_norm.weight");
    REQ(gate,      "ffn_gate.weight");
    REQ(up,        "ffn_up.weight");
    REQ(down,      "ffn_down.weight");

    if (op == MYNAH_SLM_OP_SHORTCONV) {
        REQ(conv_in,  "shortconv.in_proj.weight");
        REQ(conv_w,   "shortconv.conv.weight");
        REQ(conv_out, "shortconv.out_proj.weight");
    } else {
        REQ(wq,        "attn_q.weight");
        REQ(wk,        "attn_k.weight");
        REQ(wv,        "attn_v.weight");
        REQ(wo,        "attn_output.weight");
        OPT(q_norm,    "attn_q_norm.weight");
        OPT(k_norm,    "attn_k_norm.weight");
    }

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
    /* The final norm before the LM head. LFM2 spells it `token_embd_norm`,
     * which reads like a norm on the way IN and is not one: upstream it is
     * `model.embedding_norm`, applied once to the last hidden state
     * (docs/lfm2-arch.md). Same slot, different name. */
    m->out_norm = bind_opt(m->gguf, "output_norm.weight");
    if (!m->out_norm) m->out_norm = bind_opt(m->gguf, "token_embd_norm.weight");
    if (!m->out_norm && !rc)
        rc = fail(err, errsz, "missing tensor: neither output_norm.weight nor "
                              "token_embd_norm.weight is present");
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
        if (bind_layer(&m->layers[i], m->gguf, i,
                       (mynah_slm_op)m->cfg.layer_op[i], err, errsz) != 0) {
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
    free(m->cfg.layer_op);
    free(m->cfg.op_slot);
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
