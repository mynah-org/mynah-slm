/* test_parity.c — run the C forward pass over the oracle's exact tokens and
 * dump the same stages, for tools/eval/compare.py to judge.
 *
 *   tests/test_parity <model.gguf> <golden-dir> <out-dir>
 *
 * This program asserts nothing itself. It produces evidence; compare.py holds
 * the tolerances, because tolerances belong in one place and that place is the
 * one that also knows what the oracle did.
 *
 * Exit 77 (skip) when the model or the golden tokens are missing.
 *
 * SPDX-License-Identifier: MIT */
#include "arch_qwen3.h"
#include "model.h"
#include "mynah_slm.h"
#include "npy.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Which layers the oracle dumps. Kept in sync with generate.py's default. */
#define DUMP_L0  0
#define DUMP_L1  13
#define DUMP_L2  27

typedef struct {
    uint32_t d_model, vocab, seq, t;        /* t = current position */
    float *embed, *l0, *l13, *l27, *fin, *logits;
} capture;

/* The residual stream after each layer, exactly what the oracle stores as
 * dump[f"layer{i}"]. */
static void on_layer(void *ctx, uint32_t layer, const float *x, uint32_t dim) {
    capture *c = ctx;
    float *dst = NULL;
    if      (layer == DUMP_L0) dst = c->l0;
    else if (layer == DUMP_L1) dst = c->l13;
    else if (layer == DUMP_L2) dst = c->l27;
    if (dst) memcpy(dst + (size_t)c->t * dim, x, dim * sizeof(float));
}

static void on_final(void *ctx, const float *x, uint32_t dim) {
    capture *c = ctx;
    memcpy(c->fin + (size_t)c->t * dim, x, dim * sizeof(float));
}

static void on_embed(void *ctx, const float *x, uint32_t dim) {
    capture *c = ctx;
    memcpy(c->embed + (size_t)c->t * dim, x, dim * sizeof(float));
}

static uint32_t *read_tokens(const char *path, uint32_t *n_out) {
    FILE *f = fopen(path, "r");
    if (!f) return NULL;

    size_t cap = 64, n = 0;
    uint32_t *ids = malloc(cap * sizeof *ids);
    if (!ids) { fclose(f); return NULL; }

    unsigned v;
    while (fscanf(f, "%u", &v) == 1) {
        if (n == cap) {
            cap *= 2;
            uint32_t *grown = realloc(ids, cap * sizeof *ids);
            if (!grown) { free(ids); fclose(f); return NULL; }
            ids = grown;
        }
        ids[n++] = v;
    }
    fclose(f);
    *n_out = (uint32_t)n;
    return ids;
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <model.gguf> <golden-dir> <out-dir>\n", argv[0]);
        return 2;
    }
    const char *model_path = argv[1], *golden = argv[2], *out = argv[3];

    char tok_path[512];
    snprintf(tok_path, sizeof tok_path, "%s/tokens.txt", golden);

    uint32_t seq = 0;
    uint32_t *tokens = read_tokens(tok_path, &seq);
    if (!tokens || seq == 0) {
        printf("SKIP no golden tokens at %s (make golden-dump)\n", tok_path);
        free(tokens);
        return 77;
    }

    char err[256];
    mynah_slm_model_t *m = mynah_slm_load(model_path, err, sizeof err);
    if (!m) {
        if (strstr(err, "No such file") || strstr(err, "cannot open")) {
            printf("SKIP no checkpoint at %s (scripts/use_model.sh)\n", model_path);
            free(tokens);
            return 77;
        }
        printf("FAIL load: %s\n", err);
        free(tokens);
        return 1;
    }

    const mynah_slm_config *c = mynah_slm_model_config(m);
    mynah_slm_state st;
    if (mynah_slm_state_init(&st, m, seq + 1, err, sizeof err) != 0) {
        printf("FAIL state: %s\n", err);
        mynah_slm_free(m);
        free(tokens);
        return 1;
    }

    capture cap = { .d_model = c->d_model, .vocab = c->vocab_size, .seq = seq };
    cap.embed  = calloc((size_t)seq * c->d_model, sizeof(float));
    cap.l0     = calloc((size_t)seq * c->d_model, sizeof(float));
    cap.l13    = calloc((size_t)seq * c->d_model, sizeof(float));
    cap.l27    = calloc((size_t)seq * c->d_model, sizeof(float));
    cap.fin    = calloc((size_t)seq * c->d_model, sizeof(float));
    cap.logits = calloc((size_t)seq * c->vocab_size, sizeof(float));
    if (!cap.embed || !cap.l0 || !cap.l13 || !cap.l27 || !cap.fin || !cap.logits) {
        printf("FAIL out of memory for the capture buffers\n");
        return 1;
    }

    st.on_embed     = on_embed;
    st.on_layer     = on_layer;
    st.on_final     = on_final;
    st.on_layer_ctx = &cap;

    printf("running %u tokens through %s\n", seq, model_path);
    for (uint32_t t = 0; t < seq; t++) {
        cap.t = t;
        if (mynah_slm_forward(&st, tokens[t], cap.logits + (size_t)t * c->vocab_size) != 0) {
            printf("FAIL forward failed at token %u\n", t);
            return 1;
        }
    }

    /* Stage names must match the oracle's dump keys exactly — compare.py looks
     * them up by name. */
    char p[512];
    int rc = 0;
#define DUMP(name, buf, cols) \
    snprintf(p, sizeof p, "%s/%s.npy", out, name); \
    if (npy_write_f32(p, buf, seq, cols) != 0) { \
        printf("FAIL cannot write %s\n", p); rc = 1; }

    DUMP("embed",      cap.embed,  c->d_model)
    DUMP("layer0",     cap.l0,     c->d_model)
    DUMP("layer13",    cap.l13,    c->d_model)
    DUMP("layer27",    cap.l27,    c->d_model)
    DUMP("final_norm", cap.fin,    c->d_model)
    DUMP("logits",     cap.logits, c->vocab_size)
#undef DUMP

    if (rc == 0) printf("dumped 6 stages -> %s\n", out);

    free(cap.embed); free(cap.l0); free(cap.l13); free(cap.l27);
    free(cap.fin); free(cap.logits);
    mynah_slm_state_free(&st);
    mynah_slm_free(m);
    free(tokens);
    return rc;
}
