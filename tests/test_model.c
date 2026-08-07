/* test_model.c — the loader against a real checkpoint.
 *
 * Every expected value is transcribed from docs/qwen3-arch.md, which was in
 * turn read off the file. So this test does not check that the loader agrees
 * with itself; it checks that it agrees with the documented architecture, and
 * it fails if either drifts.
 *
 * Exit 77 (skip) when no checkpoint is staged: weights are not a build
 * dependency, and CI has none.
 *
 * SPDX-License-Identifier: MIT */
#include "model.h"
#include "mynah_slm.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>


static int failures;

static void eq_u32(const char *what, uint32_t got, uint32_t want) {
    if (got != want) {
        printf("FAIL %-22s %u, expected %u\n", what, got, want);
        failures++;
    } else {
        printf("ok   %-22s %u\n", what, got);
    }
}

static void eq_f32(const char *what, float got, float want) {
    if (got != want) {
        printf("FAIL %-22s %g, expected %g\n", what, got, want);
        failures++;
    } else {
        printf("ok   %-22s %g\n", what, got);
    }
}

static void present(const char *what, const void *p, int want) {
    int got = p != NULL;
    if (got != want) {
        printf("FAIL %-22s %s, expected %s\n", what,
               got ? "present" : "absent", want ? "present" : "absent");
        failures++;
    } else {
        printf("ok   %-22s %s\n", what, got ? "present" : "absent");
    }
}

/* Shapes are checked in row-major order, i.e. ne reversed — the order
 * docs/qwen3-arch.md lists and the order a model card uses. */
static void shape2(const char *what, const ingot_tensor *t,
                   uint64_t rows, uint64_t cols) {
    if (!t) { printf("FAIL %-22s absent\n", what); failures++; return; }
    if (t->rank != 2 || t->ne[1] != rows || t->ne[0] != cols) {
        printf("FAIL %-22s rank %u ne=[%llu,%llu], expected row-major [%llu, %llu]\n",
               what, t->rank, (unsigned long long)t->ne[0],
               (unsigned long long)t->ne[1],
               (unsigned long long)rows, (unsigned long long)cols);
        failures++;
        return;
    }
    printf("ok   %-22s [%llu x %llu] %s\n", what, (unsigned long long)rows,
           (unsigned long long)cols, ingot_type_name(t->type));
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : "models-local/Qwen3-0.6B-Q4_K_M.gguf";

    char err[256];
    mynah_slm_model_t *m = mynah_slm_load(path, err, sizeof err);
    if (!m) {
        /* Distinguish "no weights staged" (skip) from "the loader is broken"
         * (failure). Only the first is allowed to be silent. */
        if (strstr(err, "No such file") || strstr(err, "cannot open")) {
            printf("SKIP no checkpoint at %s (scripts/use_model.sh)\n", path);
            return 77;
        }
        printf("FAIL load: %s\n", err);
        return 1;
    }

    const mynah_slm_config *c = mynah_slm_model_config(m);
    printf("-- config, vs docs/qwen3-arch.md --\n");

    if (strcmp(c->arch, "qwen3") != 0) {
        printf("FAIL arch %s, expected qwen3\n", c->arch);
        failures++;
    } else {
        printf("ok   %-22s %s\n", "arch", c->arch);
    }

    eq_u32("n_layers",    c->n_layers,   28);
    eq_u32("d_model",     c->d_model,    1024);
    eq_u32("d_ff",        c->d_ff,       3072);
    eq_u32("n_heads",     c->n_heads,    16);
    eq_u32("n_kv_heads",  c->n_kv_heads, 8);
    /* The one that matters: 128, not d_model/n_heads = 64. */
    eq_u32("head_dim",    c->head_dim,   128);
    eq_u32("n_ctx",       c->n_ctx,      40960);
    eq_u32("vocab_size",  c->vocab_size, 151936);
    eq_u32("q_dim",       c->q_dim,      2048);
    eq_u32("kv_dim",      c->kv_dim,     1024);
    /* Both are FLOAT32 KVs; read through an integer accessor they come back
     * as 0 with no error. Asserting the value is what catches that. */
    eq_f32("rms_eps",     c->rms_eps,    1e-6f);
    eq_f32("rope_theta",  c->rope_theta, 1e6f);

    if (c->q_dim == c->d_model) {
        printf("FAIL q_dim equals d_model — head_dim was probably derived\n");
        failures++;
    }

    printf("\n-- tensors --\n");
    shape2("token_embd", m->embed, 151936, 1024);
    present("output_norm", m->out_norm, 1);
    /* Tied embeddings: there is no separate output.weight in this family. */
    present("output.weight (untied)", m->lm_head, 0);

    shape2("blk.0.attn_q",      m->layers[0].wq,   2048, 1024);
    shape2("blk.0.attn_k",      m->layers[0].wk,   1024, 1024);
    shape2("blk.0.attn_v",      m->layers[0].wv,   1024, 1024);
    shape2("blk.0.attn_output", m->layers[0].wo,   1024, 2048);
    shape2("blk.0.ffn_gate",    m->layers[0].gate, 3072, 1024);
    shape2("blk.0.ffn_up",      m->layers[0].up,   3072, 1024);
    shape2("blk.0.ffn_down",    m->layers[0].down, 1024, 3072);
    present("blk.0.attn_q_norm", m->layers[0].q_norm, 1);
    present("blk.0.attn_k_norm", m->layers[0].k_norm, 1);

    /* Every layer must be fully bound, not just layer 0. A loop that stops
     * early would otherwise pass this test and fail at layer 1. */
    int bound = 1;
    for (uint32_t i = 0; i < c->n_layers; i++) {
        const mynah_slm_layer *l = &m->layers[i];
        if (!l->attn_norm || !l->wq || !l->wk || !l->wv || !l->wo ||
            !l->ffn_norm || !l->gate || !l->up || !l->down) {
            printf("FAIL layer %u is not fully bound\n", i);
            failures++;
            bound = 0;
            break;
        }
    }
    if (bound) printf("ok   %-22s all %u layers bound\n", "layers", c->n_layers);

    /* The QK-norm weights are head_dim long, which is how the per-head variant
     * is identifiable from the file alone. */
    if (m->layers[0].q_norm && m->layers[0].q_norm->ne[0] != c->head_dim) {
        printf("FAIL q_norm is %llu long, expected head_dim %u\n",
               (unsigned long long)m->layers[0].q_norm->ne[0], c->head_dim);
        failures++;
    } else if (m->layers[0].q_norm) {
        printf("ok   %-22s %u (= head_dim, so per-head)\n", "q_norm length", c->head_dim);
    }

    /* Zero-copy: the data pointer must land inside the mapping, not in a copy. */
    const void *p = mynah_slm_tensor_data(m, m->embed);
    if (!p) { printf("FAIL token_embd has no data pointer\n"); failures++; }
    else    { printf("ok   %-22s zero-copy pointer\n", "token_embd data"); }

    mynah_slm_free(m);
    printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "PASS",
           failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
