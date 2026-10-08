/* bench_decode.c — what one decode step costs for B streams, three ways.
 *
 *   solo     B separate single-token steps: what the serialized server does,
 *            every weight read B times
 *   matmat   ONE mynah_slm_forward_multi step, one qmatmat per weight over
 *            the B rows (dequantize a strip, then sgemm) — weights read once
 *   matvec   ONE forward_multi step, B threaded matvecs per weight — the
 *            same kernels as solo, but one attention region per layer
 *
 * Interleaved in ONE process, rotating the arm order every round, because two
 * separate runs on a shared machine disagree by more than the effect
 * (engineering-method.md §2, and tests/bench_matvec.c's 80%).
 *
 * Usage: bench_decode [model.gguf] [position] [rounds]
 * With no model it WRITES one: the 0.6B geometry (28 layers, 1024/3072,
 * 16/8 heads of 128, 151936-row tied Q6_K head, Q4_K_M type mix) with noise
 * for weights, ~400 MB in $TMPDIR, deleted afterwards. Speed does not depend
 * on the values; quality claims cannot come from it.
 *
 * Not part of `make test`: it measures. `make bench-decode`.
 *
 * SPDX-License-Identifier: MIT */
#include "arch_qwen3.h"
#include "fixture_model.h"
#include "model.h"
#include "mynah_slm.h"
#include "sgemm.h"
#include "threads.h"
#include "timing.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define BMAX 8

static int cmp_d(const void *a, const void *b) {
    const double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

int main(int argc, char **argv) {
    const char *path = argc > 1 ? argv[1] : NULL;
    const uint32_t pos = argc > 2 ? (uint32_t)atoi(argv[2]) : 128;
    const int rounds = argc > 3 ? atoi(argv[3]) : 7;
    char tmp[256] = "", err[256];

    const int nth = mynah_slm_threads_init(0);
    if (!path) {
        fixture_spec spec;
        fixture_spec_06b_shape(&spec);
        fprintf(stderr, "writing a 0.6B-geometry fixture (noise weights) ... ");
        const double t0 = mynah_slm_now();
        if (fixture_write_temp(tmp, sizeof tmp, &spec, err, sizeof err) != 0) {
            fprintf(stderr, "failed: %s\n", err);
            return 1;
        }
        fprintf(stderr, "%.1f s, %s\n", mynah_slm_now() - t0, tmp);
        path = tmp;
    }

    mynah_slm_model_t *m = mynah_slm_load(path, err, sizeof err);
    if (!m) { fprintf(stderr, "load: %s\n", err); if (*tmp) unlink(tmp); return 1; }
    const mynah_slm_config *c = mynah_slm_model_config(m);

    mynah_slm_state ws;
    mynah_slm_seq seq[BMAX];
    memset(seq, 0, sizeof seq);
    const uint32_t cap = pos + 64;
    if (mynah_slm_state_init_workspace(&ws, m, cap, err, sizeof err) != 0 ||
        mynah_slm_state_init_decode(&ws, BMAX, err, sizeof err) != 0) {
        fprintf(stderr, "workspace: %s\n", err);
        return 1;
    }
    uint32_t *prompt = malloc(pos * sizeof *prompt);
    uint64_t r = 12345;
    for (uint32_t i = 0; i < pos; i++) {
        r = r * 6364136223846793005ULL + 1442695040888963407ULL;
        prompt[i] = (uint32_t)(r >> 33) % 256u;           /* byte tokens */
    }
    for (int b = 0; b < BMAX; b++) {
        if (mynah_slm_seq_reserve(&seq[b], m, cap, MYNAH_SLM_KV_BF16, MYNAH_SLM_KV_BF16,
                                  err, sizeof err) != 0) {
            fprintf(stderr, "seq: %s\n", err);
            return 1;
        }
        for (uint32_t i = 0; i < pos; i += 64) {
            const uint32_t take = pos - i < 64 ? pos - i : 64;
            mynah_slm_seq_forward_batch(&ws, &seq[b], prompt + i, take, NULL);
        }
    }

    printf("bench_decode | %s | %u layers, d %u, ff %u, vocab %u | position %u | "
           "%d threads | sgemm %s | KV bf16 | %d rounds x 3 steps per arm\n",
           *tmp ? "synthetic 0.6B geometry (noise), local tmp" : path,
           c->n_layers, c->d_model, c->d_ff, c->vocab_size, pos, nth,
           mynah_slm_sgemm_backend(), rounds);
    printf("%-3s %10s %10s %10s   %9s %9s %9s   %s\n", "B", "solo ms", "matmat ms",
           "matvec ms", "solo t/s", "mm t/s", "mv t/s", "matmat/solo");

    const uint32_t widths[] = { 1, 2, 4, 8 };
    for (size_t wi = 0; wi < sizeof widths / sizeof *widths; wi++) {
        const uint32_t B = widths[wi];
        double *t[3];
        for (int a = 0; a < 3; a++) t[a] = calloc((size_t)rounds, sizeof(double));
        mynah_slm_seq *bp[BMAX];
        uint32_t tok[BMAX];
        for (uint32_t b = 0; b < B; b++) { bp[b] = &seq[b]; tok[b] = 'a' + b; }

        /* one warm-up step per arm */
        for (int round = -1; round < rounds; round++) {
            for (int k = 0; k < 3; k++) {
                const int arm = (k + (round < 0 ? 0 : round)) % 3;
                double best = 1e30;
                for (int rep = 0; rep < 3; rep++) {
                    for (uint32_t b = 0; b < B; b++) seq[b].n_past = pos;
                    float *lg[BMAX];
                    const double t0 = mynah_slm_now();
                    if (arm == 0) {
                        for (uint32_t b = 0; b < B; b++)
                            mynah_slm_seq_forward(&ws, &seq[b], tok[b], NULL);
                    } else {
                        mynah_slm_decode_product_set(arm == 1);
                        mynah_slm_forward_multi(&ws, bp, tok, B, lg);
                    }
                    const double dt = mynah_slm_now() - t0;
                    if (dt < best) best = dt;
                }
                if (round >= 0) t[arm][round] = best;
            }
        }
        double med[3];
        for (int a = 0; a < 3; a++) {
            qsort(t[a], (size_t)rounds, sizeof(double), cmp_d);
            med[a] = t[a][rounds / 2];
            free(t[a]);
        }
        printf("%-3u %10.2f %10.2f %10.2f   %9.1f %9.1f %9.1f   %.2fx\n", B,
               med[0] * 1e3, med[1] * 1e3, med[2] * 1e3,
               B / med[0], B / med[1], B / med[2], med[0] / med[1]);
        fflush(stdout);
    }
    printf("(t/s = aggregate decode tokens per second over the B streams; "
           "median of %d rounds, each the best of 3 steps)\n", rounds);

    for (int b = 0; b < BMAX; b++) mynah_slm_seq_free(&seq[b]);
    mynah_slm_state_free(&ws);
    free(prompt);
    mynah_slm_free(m);
    if (*tmp) unlink(tmp);
    mynah_slm_threads_shutdown();
    return 0;
}
