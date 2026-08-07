/* bench_matvec.c — where a decode step actually goes, per tensor.
 *
 * The engine reports its own tok/s on every run, which says WHETHER something
 * got faster and never WHERE. This says where. It is the instrument that found
 * the Q6_K kernel gap (2.8x slower per element than Q4_K, invisible end to end
 * because `ingot_has_kernel()` answered 1 for both) and the one that has to be
 * re-run before any claim about a kernel, because the ranking has already
 * inverted once: after that fix Q6_K became the FASTER of the two per element,
 * which quietly killed a "quantize the embedding to Q4_K" plan that was written
 * down while the old numbers were true.
 *
 * Reports per-element throughput, not milliseconds alone: the tensors have
 * different shapes, and ms/tensor cannot be compared across them.
 *
 *   make bench
 *   tests/bench_matvec models-local/Qwen3-0.6B-Q4_K_M.gguf [reps]
 *
 * SPDX-License-Identifier: MIT */
#include "arch_qwen3.h"
#include "model.h"
#include "mynah_slm.h"
#include "qmat.h"
#include "threads.h"
#include "timing.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const char         *label;
    const ingot_tensor *w;
} entry;

/* Best of `rounds`, ALTERNATING ours and ingot's inside one process.
 *
 * Two separate runs on a warm laptop disagreed by 80% on tensors that neither
 * change touches — that is thermal drift, not a kernel. Interleaving and
 * taking the best of several rounds removes the drift from the comparison,
 * because both sides see the same machine.
 *
 * Best rather than mean on purpose: the fastest observed time is the one least
 * contaminated by whatever else the OS was doing. */
static void bench_pair(const mynah_slm_model_t *m, const ingot_tensor *w,
                       float *in, float *out, int reps, int rounds,
                       double *own, double *ingot) {
    *own = *ingot = 1e30;

    /* Q6_K defaults to ingot's on ARM, so without this the Q6_K rows would
     * compare ingot against ingot and print a meaningless 1.00x. The A/B has to
     * stay runnable on the machine whose default is the other side — that is
     * how the tie was measured in the first place. */
    mynah_slm_matvec_set_q6k(1);
    mynah_slm_matvec_set_enabled(1);
    mynah_slm_project(m, w, in, out);            /* warm the mapping */

    for (int r = 0; r < rounds; r++) {
        for (int side = 0; side < 2; side++) {
            mynah_slm_matvec_set_enabled(side == 0);
            const double t0 = mynah_slm_now();
            for (int i = 0; i < reps; i++) mynah_slm_project(m, w, in, out);
            const double dt = (mynah_slm_now() - t0) / reps;
            double *slot = side == 0 ? own : ingot;
            if (dt < *slot) *slot = dt;
        }
    }
}

int main(int argc, char **argv) {
    const char *model_path = argc > 1 ? argv[1] : "models-local/Qwen3-0.6B-Q4_K_M.gguf";
    const int   reps       = argc > 2 ? atoi(argv[2]) : 20;

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

    const int nth = mynah_slm_threads_init(0);
    const mynah_slm_config *c = &m->cfg;
    const mynah_slm_layer  *l = &m->layers[0];

    const entry rows[] = {
        { "attn_q",     l->wq   },
        { "attn_k",     l->wk   },
        { "attn_v",     l->wv   },
        { "attn_out",   l->wo   },
        { "ffn_gate",   l->gate },
        { "ffn_up",     l->up   },
        { "ffn_down",   l->down },
        { "lm_head",    m->lm_head ? m->lm_head : m->embed },
    };
    const size_t n_rows = sizeof rows / sizeof *rows;

    /* One input wide enough for any of them, one output tall enough. */
    size_t max_in = 0, max_out = 0;
    for (size_t i = 0; i < n_rows; i++) {
        const size_t cols = (size_t)rows[i].w->ne[0];
        const size_t outr = rows[i].w->rank >= 2 ? (size_t)rows[i].w->ne[1] : 1;
        if (cols > max_in)  max_in  = cols;
        if (outr > max_out) max_out = outr;
    }
    float *in  = mynah_slm_aligned_alloc(max_in  * sizeof *in);
    float *out = mynah_slm_aligned_alloc(max_out * sizeof *out);
    if (!in || !out) { printf("FAIL out of memory\n"); return 1; }
    for (size_t i = 0; i < max_in; i++) in[i] = (float)((i % 17) - 8) * 0.05f;

    const int rounds = 3;
    printf("%s\n", model_path);
    printf("%d threads, %d reps x %d interleaved rounds, best of each\n\n",
           nth, reps, rounds);
    printf("  %-10s %-6s %-16s %9s %9s %8s %10s\n",
           "tensor", "type", "shape", "ours ms", "ingot ms", "ratio", "G elem/s");

    double step_own = 0.0, step_ingot = 0.0;
    for (size_t i = 0; i < n_rows; i++) {
        const ingot_tensor *w = rows[i].w;
        const size_t cols = (size_t)w->ne[0];
        const size_t outr = w->rank >= 2 ? (size_t)w->ne[1] : 1;
        double own = 0.0, ing = 0.0;
        bench_pair(m, w, in, out, reps, rounds, &own, &ing);
        const double elem = (double)cols * (double)outr;

        char shape[32];
        snprintf(shape, sizeof shape, "%zu x %zu", outr, cols);
        printf("  %-10s %-6s %-16s %9.3f %9.3f %7.2fx %10.2f\n",
               rows[i].label, ingot_type_name(w->type), shape,
               own * 1e3, ing * 1e3, ing / own, elem / own / 1e9);

        /* A decode step is every layer tensor n_layers times, plus one head. */
        step_own   += (i + 1 < n_rows) ? own * c->n_layers : own;
        step_ingot += (i + 1 < n_rows) ? ing * c->n_layers : ing;
    }

    printf("\n  a decode step from these numbers: %.1f ms -> %.1f tok/s"
           "   (ingot only: %.1f ms -> %.1f tok/s)\n",
           step_own * 1e3, 1.0 / step_own, step_ingot * 1e3, 1.0 / step_ingot);
    printf("  (matvec only — attention and the norms are on top, and they grow\n"
           "   with the context; see docs/perf.md)\n");

    mynah_slm_aligned_free(in);
    mynah_slm_aligned_free(out);
    mynah_slm_free(m);
    mynah_slm_threads_shutdown();
    return 0;
}
