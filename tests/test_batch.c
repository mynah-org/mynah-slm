/* test_batch.c — batched prefill must BE the one-token path, only faster.
 *
 * arch_qwen3.c says it out loud: prefill is a speed optimization that has to
 * prove itself equal to the single-token forward, not a second definition of
 * what the model is. This is where it proves it.
 *
 * The two are not bit-identical and are not supposed to be — sgemm sums in its
 * own order, so the difference is a reorder of the same arithmetic. What must
 * hold is that it stays a reorder: a few ulp, and the same argmax. A batched
 * path that quantized activations to int8 (which is what ingot's own batched
 * Q4_K does by default) would land at ~2.4e-3 and fail here, which is exactly
 * why this test exists rather than a benchmark alone.
 *
 * SPDX-License-Identifier: MIT */
#include "arch_qwen3.h"
#include "model.h"
#include "mynah_slm.h"
#include "tokenizer.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures;

static void check(const char *what, int ok, const char *detail) {
    if (ok) printf("ok   %s\n", what);
    else { printf("FAIL %s\n     <- %s\n", what, detail ? detail : ""); failures++; }
}

/* Relative difference on the largest component, which is the one that decides
 * the token. An absolute epsilon would pass trivially on small logits. */
static double rel_diff(const float *a, const float *b, size_t n, size_t *worst) {
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < n; i++) {
        const double d = fabs((double)a[i] - (double)b[i]);
        if (d > num) { num = d; if (worst) *worst = i; }
        const double m = fabs((double)a[i]);
        if (m > den) den = m;
    }
    return den > 0.0 ? num / den : num;
}

static size_t argmax(const float *v, size_t n) {
    size_t best = 0;
    for (size_t i = 1; i < n; i++) if (v[i] > v[best]) best = i;
    return best;
}

int main(int argc, char **argv) {
    const char *model_path = argc > 1 ? argv[1] : "models-local/Qwen3-0.6B-Q4_K_M.gguf";

    char err[256];
    mynah_slm_model_t *m = mynah_slm_load(model_path, err, sizeof err);
    if (!m) {
        if (strstr(err, "No such file") || strstr(err, "cannot open")) {
            printf("SKIP no checkpoint at %s (scripts/use_model.sh)\n", model_path);
            return 77;
        }
        printf("FAIL load: %s\n", err);
        return 1;
    }

    mynah_slm_tokenizer *tok = mynah_slm_tokenizer_load(m->gguf, err, sizeof err);
    if (!tok) { printf("FAIL tokenizer: %s\n", err); return 1; }

    /* Long enough to cross a batch boundary at the default width, and mixed
     * scripts so nothing depends on one-byte tokens. */
    const char *text =
        "Il carbone che alimentava le fornaci arrivava dal nord, su chiatte "
        "che risalivano il fiume anche d'inverno. Die Fabrik lief Tag und "
        "Nacht. The ledger for 1887 lists twelve names, four of them women, "
        "and a column of figures nobody has explained since. 港には船が並んで "
        "いた。La contabilidad no cuadra, y el error se repite cada marzo.";

    const long n_tok = mynah_slm_tokenize(tok, text, 0, NULL, 0);
    uint32_t *ids = malloc((size_t)n_tok * sizeof *ids);
    mynah_slm_tokenize(tok, text, 0, ids, (size_t)n_tok);
    printf("     %ld tokens\n\n", n_tok);

    const uint32_t vocab = mynah_slm_vocab_size(m);
    float *ref = malloc(vocab * sizeof *ref);
    float *got = malloc(vocab * sizeof *got);
    if (!ref || !got) return 1;

    mynah_slm_state st;
    if (mynah_slm_state_init(&st, m, (uint32_t)n_tok + 64, err, sizeof err) != 0) {
        printf("FAIL state: %s\n", err);
        return 1;
    }
    printf("     batch width %u\n\n", mynah_slm_batch_max(&st));

    /* ── the reference: one token at a time, the definition ───────────────── */
    for (long i = 0; i + 1 < n_tok; i++)
        if (mynah_slm_forward(&st, ids[i], NULL) != 0) { printf("FAIL forward\n"); return 1; }
    if (mynah_slm_forward(&st, ids[n_tok - 1], ref) != 0) { printf("FAIL forward\n"); return 1; }
    const uint32_t past_ref = st.n_past;
    const size_t   pick_ref = argmax(ref, vocab);

    /* ── one batch for the whole prompt ───────────────────────────────────── */
    struct { const char *name; uint32_t width; } cases[] = {
        { "one batch for the whole prompt", (uint32_t)n_tok },
        { "several batches back to back",   7 },
        { "a width that does not divide the prompt", 13 },
    };

    for (size_t k = 0; k < sizeof cases / sizeof *cases; k++) {
        uint32_t width = cases[k].width;
        if (width > mynah_slm_batch_max(&st)) width = mynah_slm_batch_max(&st);

        mynah_slm_state_reset(&st);
        long i = 0;
        int bad = 0;
        while (i + 1 < n_tok) {
            long take = (n_tok - 1) - i;
            if (take > (long)width) take = width;
            if (mynah_slm_forward_batch(&st, ids + i, (uint32_t)take, NULL) != 0) { bad = 1; break; }
            i += take;
        }
        if (bad || mynah_slm_forward(&st, ids[n_tok - 1], got) != 0) {
            check(cases[k].name, 0, "forward_batch failed");
            continue;
        }

        size_t worst = 0;
        const double rel = rel_diff(ref, got, vocab, &worst);
        const size_t pick = argmax(got, vocab);

        char detail[256];
        snprintf(detail, sizeof detail,
                 "width %u: rel=%.2e at %zu, argmax %zu vs %zu, n_past %u vs %u",
                 width, rel, worst, pick, pick_ref, st.n_past, past_ref);

        /* 1e-4 is the same tolerance the parity gate holds layer 0 to. A
         * reorder lands two orders of magnitude below it; anything that does
         * not is a different computation wearing the same name. */
        check(cases[k].name, rel < 1e-4 && pick == pick_ref && st.n_past == past_ref, detail);
        printf("     %s\n", detail);
    }

    /* The cache has to be right, not just the last logits: if K or V landed at
     * the wrong position, the prompt still scores but every token generated
     * after it reads history that was never there. Checked by generating past
     * the prompt from both paths and requiring the same ids. */
    uint32_t gen_ref[12], gen_batch[12];
    for (int pass = 0; pass < 2; pass++) {
        mynah_slm_state_reset(&st);
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
            if (mynah_slm_forward(&st, next, got) != 0) break;
            next = (uint32_t)argmax(got, vocab);
            (pass == 0 ? gen_ref : gen_batch)[g] = next;
        }
    }
    char detail[256] = "";
    int same = 1;
    for (int g = 0; g < 12; g++)
        if (gen_ref[g] != gen_batch[g]) {
            same = 0;
            snprintf(detail, sizeof detail, "token %d: %u vs %u", g, gen_ref[g], gen_batch[g]);
            break;
        }
    check("greedy continuation is identical after a batched prefill", same, detail);

    free(ref); free(got); free(ids);
    mynah_slm_state_free(&st);
    mynah_slm_tokenizer_free(tok);
    mynah_slm_free(m);

    printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "PASS",
           failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
