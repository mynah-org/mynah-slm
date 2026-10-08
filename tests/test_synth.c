/* test_synth.c — the forward pass, on a checkpoint written by the test itself.
 *
 * test_batch holds batched prefill to the one-token path on a real checkpoint,
 * and SKIPS wherever there is none — which is CI, and this cloud branch. This
 * runs the same gate on tests/fixture_model.c's synthetic Qwen3: noise for
 * weights, real architecture, real tensor names, real tokenizer machinery.
 * What it can prove is that the engine agrees with itself; it cannot say
 * anything about text quality.
 *
 * Two fixtures: every matrix F32 (the batched product is then a plain sgemm
 * against the matvec), and the Q4_K_M type mix (Q4_K / Q8_0 / Q6_K), which is
 * the path a real checkpoint takes — our own Q4_K matvec on one side, the
 * dequantized strip + sgemm on the other.
 *
 * SPDX-License-Identifier: MIT */
#include "arch_qwen3.h"
#include "fixture_model.h"
#include "model.h"
#include "mynah_slm.h"
#include "threads.h"
#include "tokenizer.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int failures;

static void check(const char *what, int ok, const char *detail) {
    if (ok) printf("ok   %s\n", what);
    else { printf("FAIL %s\n     <- %s\n", what, detail ? detail : ""); failures++; }
}

static double rel_diff(const float *a, const float *b, size_t n) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < n; i++) {
        const double d = fabs((double)a[i] - (double)b[i]);
        if (d > num) num = d;
        const double m = fabs((double)a[i]);
        if (m > den) den = m;
    }
    return den > 0.0 ? num / den : num;
}

/* Reset AND poison the cache. state_reset only rewinds n_past: the bytes stay,
 * and the previous pass wrote the very same K/V at the very same positions, so
 * a batched path that skipped or misplaced a position would read back correct
 * stale values and pass. Found by mutation (K written one slot late): it
 * survived until this existed. 0xFF bytes are NaN in f32 and in bf16. */
static void reset_poisoned(mynah_slm_state *st) {
    mynah_slm_state_reset(st);
    const mynah_slm_kv *kv = &st->kv;
    const size_t positions = (size_t)kv->n_layers * kv->n_ctx;
    memset(kv->k, 0xff, positions * kv->pos_bytes_k);
    memset(kv->v, 0xff, positions * kv->pos_bytes_v);
}

static size_t argmax(const float *v, size_t n) {
    size_t best = 0;
    for (size_t i = 1; i < n; i++) if (v[i] > v[best]) best = i;
    return best;
}

/* The same text test_batch uses: mixed scripts, so nothing depends on
 * one-byte tokens, and long enough to cross several batch boundaries. */
static const char *TEXT =
    "Il carbone che alimentava le fornaci arrivava dal nord. Die Fabrik lief "
    "Tag und Nacht. The ledger for 1887 lists twelve names. \xe6\xb8\xaf\xe3\x81\xab"
    "\xe3\x81\xaf\xe8\x88\xb9\xe3\x81\x8c\xe4\xb8\xa6\xe3\x82\x93\xe3\x81\xa7.";

static void run(const char *label, int quant) {
    printf("\n-- %s fixture --\n", label);
    fixture_spec spec;
    fixture_spec_tiny(&spec, quant);

    char path[256], err[256];
    if (fixture_write_temp(path, sizeof path, &spec, err, sizeof err) != 0) {
        check("fixture written", 0, err);
        return;
    }
    mynah_slm_model_t *m = mynah_slm_load(path, err, sizeof err);
    check("the loader accepts the fixture", m != NULL, err);
    if (!m) { unlink(path); return; }
    mynah_slm_tokenizer *tok = mynah_slm_tokenizer_load(m->gguf, err, sizeof err);
    check("the tokenizer loads from it", tok != NULL, err);
    if (!tok) { mynah_slm_free(m); unlink(path); return; }

    const long n_tok = mynah_slm_tokenize(tok, TEXT, 0, NULL, 0);
    uint32_t *ids = malloc((size_t)(n_tok > 0 ? n_tok : 1) * sizeof *ids);
    mynah_slm_tokenize(tok, TEXT, 0, ids, (size_t)n_tok);

    /* The vocabulary round-trips: decode every id back and compare bytes. */
    {
        mynah_slm_detok d;
        mynah_slm_detok_init(&d, tok);
        char out[4096], piece[64];
        size_t o = 0;
        for (long i = 0; i < n_tok; i++) {
            const long w = mynah_slm_detok_feed(&d, ids[i], piece, sizeof piece);
            if (w > 0 && o + (size_t)w < sizeof out) { memcpy(out + o, piece, (size_t)w); o += (size_t)w; }
        }
        out[o] = '\0';
        char detail[128];
        snprintf(detail, sizeof detail, "%ld ids, %zu bytes back of %zu", n_tok, o, strlen(TEXT));
        check("tokenize then detokenize is the identity", strcmp(out, TEXT) == 0, detail);
        printf("     %s\n", detail);
    }

    const uint32_t vocab = mynah_slm_vocab_size(m);
    float *ref = malloc(vocab * sizeof *ref);
    float *got = malloc(vocab * sizeof *got);

    mynah_slm_state st;
    if (mynah_slm_state_init_kv(&st, m, (uint32_t)n_tok + 64, MYNAH_SLM_KV_F32,
                                MYNAH_SLM_KV_F32, err, sizeof err) != 0) {
        check("state", 0, err);
        goto out;
    }

    for (long i = 0; i + 1 < n_tok; i++) mynah_slm_forward(&st, ids[i], NULL);
    mynah_slm_forward(&st, ids[n_tok - 1], ref);
    const uint32_t past_ref = st.n_past;
    const size_t pick_ref = argmax(ref, vocab);

    const uint32_t widths[] = { (uint32_t)n_tok, 7, 13 };
    double worst = 0.0;
    for (size_t k = 0; k < sizeof widths / sizeof *widths; k++) {
        uint32_t width = widths[k];
        if (width > mynah_slm_batch_max(&st)) width = mynah_slm_batch_max(&st);
        reset_poisoned(&st);
        long i = 0;
        int bad = 0;
        while (i + 1 < n_tok) {
            long take = (n_tok - 1) - i;
            if (take > (long)width) take = width;
            if (mynah_slm_forward_batch(&st, ids + i, (uint32_t)take, NULL) != 0) { bad = 1; break; }
            i += take;
        }
        if (bad || mynah_slm_forward(&st, ids[n_tok - 1], got) != 0) {
            check("forward_batch runs", 0, "failed");
            continue;
        }
        const double rel = rel_diff(ref, got, vocab);
        if (rel > worst) worst = rel;
        char what[96], detail[160];
        snprintf(what, sizeof what, "batched prefill at width %u == one token at a time", width);
        snprintf(detail, sizeof detail, "rel=%.2e, argmax %zu vs %zu, n_past %u vs %u",
                 rel, argmax(got, vocab), pick_ref, st.n_past, past_ref);
        /* The same 1e-4 test_batch holds a real checkpoint to; a reorder lands
         * orders of magnitude under it. */
        check(what, rel < 1e-4 && argmax(got, vocab) == pick_ref && st.n_past == past_ref, detail);
        printf("     %s\n", detail);
    }

    /* Greedy continuation after the prompt: wrong K/V positions would still
     * score the prompt and then read a history that was never there. */
    uint32_t gen[2][12];
    for (int pass = 0; pass < 2; pass++) {
        reset_poisoned(&st);
        long i = 0;
        while (i + 1 < n_tok) {
            long take = (n_tok - 1) - i;
            const uint32_t w = pass == 0 ? 1u : 16u;
            if (take > (long)w) take = w;
            mynah_slm_forward_batch(&st, ids + i, (uint32_t)take, NULL);
            i += take;
        }
        uint32_t next = ids[n_tok - 1];
        for (int g = 0; g < 12; g++) {
            if (mynah_slm_forward(&st, next, got) != 0) { gen[pass][g] = (uint32_t)-1; continue; }
            next = (uint32_t)argmax(got, vocab);
            gen[pass][g] = next;
        }
    }
    check("greedy continuation is identical after a batched prefill",
          memcmp(gen[0], gen[1], sizeof gen[0]) == 0, "token ids differ");
    printf("     worst rel over the widths: %.2e\n", worst);

    mynah_slm_state_free(&st);
out:
    free(ref); free(got); free(ids);
    mynah_slm_tokenizer_free(tok);
    mynah_slm_free(m);
    unlink(path);
}

int main(void) {
    mynah_slm_threads_init(0);
    run("F32", 0);
    run("Q4_K_M-mix", 1);
    mynah_slm_threads_shutdown();
    printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "PASS",
           failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
