/* inspect.c — report what is inside a checkpoint.
 *
 * The per-type census is the thing worth having early: "does mynah-slm run
 * this file" is really "is every block type in it one we can decode and
 * multiply", and that question is answered by counting, not by guessing from
 * the filename. `model-Q4_K_M.gguf` contains no tensor of type "Q4_K_M" —
 * that suffix names a recipe (Q4_K for most weights, Q6_K for attn_v and
 * ffn_down, F32 for norms), and only the census shows the real mix.
 *
 * SPDX-License-Identifier: MIT */
#include "mynah_slm.h"

#include "ingot/dtype.h"
#include "ingot/gguf.h"

#include <stdio.h>   /* snprintf */
#include <stdlib.h>
#include <string.h>

const char *mynah_slm_version(void) { return MYNAH_SLM_BUILD; }

static int fail(char *err, size_t errsz, const char *msg) {
    if (err && errsz) { strncpy(err, msg, errsz - 1); err[errsz - 1] = '\0'; }
    return -1;
}

/* Find the census row for `type`, appending one if this is its first tensor.
 * Linear: a checkpoint uses a handful of types, never enough to index. */
static mynah_slm_type_use *row_for(mynah_slm_report *r, int type) {
    for (size_t i = 0; i < r->n_types; i++)
        if (r->types[i].type == type) return &r->types[i];

    mynah_slm_type_use *grown =
        realloc(r->types, (r->n_types + 1) * sizeof *grown);
    if (!grown) return NULL;
    r->types = grown;

    mynah_slm_type_use *row = &r->types[r->n_types++];
    memset(row, 0, sizeof *row);
    row->type        = type;
    row->type_name   = ingot_type_name(type);
    row->can_dequant = ingot_type_can_dequant(type);
    return row;
}

/* INGOT_KV_* is the metadata value enum, distinct from INGOT_TYPE_* (the ggml
 * tensor types), so ingot_type_name() is the wrong function here. */
static const char *kv_type_name(int t) {
    switch (t) {
        case INGOT_KV_UINT8:   return "u8";
        case INGOT_KV_INT8:    return "i8";
        case INGOT_KV_UINT16:  return "u16";
        case INGOT_KV_INT16:   return "i16";
        case INGOT_KV_UINT32:  return "u32";
        case INGOT_KV_INT32:   return "i32";
        case INGOT_KV_FLOAT32: return "f32";
        case INGOT_KV_BOOL:    return "bool";
        case INGOT_KV_STRING:  return "str";
        case INGOT_KV_ARRAY:   return "array";
        case INGOT_KV_UINT64:  return "u64";
        case INGOT_KV_INT64:   return "i64";
        case INGOT_KV_FLOAT64: return "f64";
        default:               return "?";
    }
}

/* Render one KV into `dst`. Arrays are summarized, never dumped: the tokenizer
 * vocabulary is a 151936-entry array and printing it helps nobody. */
static void render_kv(const ingot_kv *kv, mynah_slm_meta_kv *out) {
    strncpy(out->name, ingot_kv_key(kv), sizeof out->name - 1);

    if (ingot_kv_type(kv) == INGOT_KV_ARRAY) {
        uint64_t len = 0;
        ingot_kv_arr_len(kv, &len);
        out->is_array = 1;
        snprintf(out->value, sizeof out->value, "[%s x %llu]",
                 kv_type_name(ingot_kv_arr_type(kv)),
                 (unsigned long long)len);
        return;
    }

    /* Switch on the DECLARED type, do not try accessors in turn. ingot's
     * integer accessors happily convert a float, so a first-match chain that
     * tries _u64 before _f64 renders rms_norm_eps = 1e-6 as "0" — which is a
     * config value we depend on, and a plausible-looking lie. */
    const char *s;
    uint64_t u;
    int64_t  i;
    double   f;
    int      b;
    switch (ingot_kv_type(kv)) {
        case INGOT_KV_STRING:
            snprintf(out->value, sizeof out->value, "%s",
                     ingot_kv_str(kv, &s) == 0 ? s : "(unreadable)");
            break;
        case INGOT_KV_BOOL:
            snprintf(out->value, sizeof out->value, "%s",
                     ingot_kv_bool(kv, &b) == 0 ? (b ? "true" : "false") : "(unreadable)");
            break;
        case INGOT_KV_FLOAT32:
        case INGOT_KV_FLOAT64:
            if (ingot_kv_f64(kv, &f) == 0) snprintf(out->value, sizeof out->value, "%g", f);
            else                           snprintf(out->value, sizeof out->value, "(unreadable)");
            break;
        case INGOT_KV_INT8:  case INGOT_KV_INT16:
        case INGOT_KV_INT32: case INGOT_KV_INT64:
            if (ingot_kv_i64(kv, &i) == 0) snprintf(out->value, sizeof out->value, "%lld", (long long)i);
            else                           snprintf(out->value, sizeof out->value, "(unreadable)");
            break;
        default:
            if (ingot_kv_u64(kv, &u) == 0) snprintf(out->value, sizeof out->value, "%llu", (unsigned long long)u);
            else                           snprintf(out->value, sizeof out->value, "(unreadable)");
            break;
    }
}

static int by_bytes_desc(const void *a, const void *b) {
    const mynah_slm_type_use *x = a, *y = b;
    if (x->bytes  != y->bytes)  return x->bytes  < y->bytes  ? 1 : -1;
    return 0;
}

int mynah_slm_inspect_gguf(mynah_slm_report *out, const char *path,
                           char *err, size_t errsz) {
    if (!out || !path) return fail(err, errsz, "null argument");
    memset(out, 0, sizeof *out);

    ingot_gguf *g = NULL;
    if (ingot_gguf_open_split(&g, path, err, errsz) != 0) return -1;

    strncpy(out->path, path, sizeof out->path - 1);
    /* ingot_gguf_arch returns "" rather than NULL when the key is absent, but
     * the storage belongs to the handle — copy before we close it. */
    strncpy(out->arch, ingot_gguf_arch(g), sizeof out->arch - 1);
    out->gguf_version = ingot_gguf_version(g);
    out->shards       = ingot_gguf_shard_count(g);
    out->n_tensors    = ingot_gguf_count(g);
    out->n_kv         = ingot_gguf_kv_count(g);
    out->runnable     = 1;

    out->tensors = calloc(out->n_tensors ? out->n_tensors : 1, sizeof *out->tensors);
    out->meta    = calloc(out->n_kv ? out->n_kv : 1, sizeof *out->meta);
    if (!out->tensors || !out->meta) {
        ingot_gguf_close(g);
        mynah_slm_report_free(out);
        return fail(err, errsz, "out of memory listing the checkpoint");
    }

    for (size_t i = 0; i < out->n_kv; i++)
        render_kv(ingot_gguf_kv_at(g, i), &out->meta[out->n_meta++]);

    uint64_t total_elems = 0;
    for (size_t i = 0; i < out->n_tensors; i++) {
        const ingot_tensor *t = ingot_gguf_at(g, i);
        mynah_slm_type_use *row = row_for(out, t->type);
        if (!row) {
            ingot_gguf_close(g);
            mynah_slm_report_free(out);
            return fail(err, errsz, "out of memory building the type census");
        }

        mynah_slm_tensor_info *info = &out->tensors[out->n_tensor_info++];
        strncpy(info->name, t->name, sizeof info->name - 1);
        info->type      = t->type;
        info->type_name = ingot_type_name(t->type);
        info->rank      = t->rank < MYNAH_SLM_MAX_RANK ? t->rank : MYNAH_SLM_MAX_RANK;
        info->nelem     = t->nelem;
        info->nbytes    = t->nbytes;
        /* ne is ggml order (fastest first); reverse it so callers read the
         * shape the way the model card writes it. */
        for (uint32_t d = 0; d < info->rank; d++)
            info->shape[d] = t->ne[info->rank - 1 - d];

        row->tensors++;
        row->elements += t->nelem;
        row->bytes    += t->nbytes;

        out->total_bytes += t->nbytes;
        total_elems      += t->nelem;
        if (!row->can_dequant) out->runnable = 0;
    }

    /* Bits per weight over the WHOLE file, not per tensor: this is the number
     * that predicts RAM, and it is always worse than the headline quant name
     * because norms and embeddings stay wide. */
    out->bits_per_weight =
        total_elems ? (double)out->total_bytes * 8.0 / (double)total_elems : 0.0;

    qsort(out->types, out->n_types, sizeof *out->types, by_bytes_desc);
    ingot_gguf_close(g);
    return 0;
}

void mynah_slm_report_free(mynah_slm_report *r) {
    if (!r) return;
    free(r->types);
    free(r->tensors);
    free(r->meta);
    r->types         = NULL;
    r->n_types       = 0;
    r->tensors       = NULL;
    r->n_tensor_info = 0;
    r->meta          = NULL;
    r->n_meta        = 0;
}
