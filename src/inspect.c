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

    uint64_t total_elems = 0;
    for (size_t i = 0; i < out->n_tensors; i++) {
        const ingot_tensor *t = ingot_gguf_at(g, i);
        mynah_slm_type_use *row = row_for(out, t->type);
        if (!row) {
            ingot_gguf_close(g);
            mynah_slm_report_free(out);
            return fail(err, errsz, "out of memory building the type census");
        }
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
    r->types   = NULL;
    r->n_types = 0;
}
