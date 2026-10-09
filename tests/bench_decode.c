/* bench_decode.c — what one decode step costs for B streams, several ways.
 *
 *   solo     B separate single-token steps: what the serialized server does,
 *            every weight read B times
 *   matvec   ONE mynah_slm_forward_multi step, B threaded matvecs per weight —
 *            the same kernels as solo, but one attention region per layer
 *   ws       ONE forward_multi step, one weight-stationary pass per weight
 *            (K7): weights read once for the B rows, still bit-identical
 *   matmat   ONE forward_multi step, one qmatmat per weight (dequantize a
 *            strip, then sgemm). 3-5x slower in S1-c; only with "all"
 *
 * Interleaved in ONE process, rotating the arm order every round, because two
 * separate runs on a shared machine disagree by more than the effect
 * (engineering-method.md §2, and tests/bench_matvec.c's 80%). Each arm's
 * sample in a round is the best of 3 steps; reported: median, min..max over
 * the rounds, and in how many rounds ws beat matvec IN THE SAME ROUND.
 *
 * Usage: bench_decode [model.gguf|-] [position] [rounds] [threads,...] [all]
 * With no model (or "-") it WRITES one: the 0.6B geometry (28 layers,
 * 1024/3072, 16/8 heads of 128, 151936-row tied Q6_K head, Q4_K_M type mix)
 * with noise for weights, ~400 MB in $TMPDIR, deleted afterwards. Speed does
 * not depend on the values; quality claims cannot come from it.
 * MYNAH_SLM_INT8=1 runs every arm with int8 activations (--fast).
 *
 * Not part of `make test`: it measures. `make bench-decode`.
 *
 * SPDX-License-Identifier: MIT */
#include "arch_qwen3.h"
#include "fixture_model.h"
#include "model.h"
#include "mynah_slm.h"
#include "qmat.h"
#include "sgemm.h"
#include "threads.h"
#include "timing.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define BMAX 8

enum { ARM_SOLO, ARM_MATVEC, ARM_WS, ARM_MATMAT, N_ARMS };
static const char *const arm_name[N_ARMS] = { "solo", "matvec", "ws", "matmat" };

static int cmp_d(const void *a, const void *b) {
    const double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

int main(int argc, char **argv) {
    const char *path = (argc > 1 && strcmp(argv[1], "-") != 0) ? argv[1] : NULL;
    const uint32_t pos = argc > 2 ? (uint32_t)atoi(argv[2]) : 128;
    const int rounds = argc > 3 ? atoi(argv[3]) : 7;
    const char *thr_list = argc > 4 ? argv[4] : "0";
    const int n_arms = (argc > 5 && strcmp(argv[5], "all") == 0) ? N_ARMS : ARM_MATMAT;
    char tmp[256] = "", err[256];

    mynah_slm_threads_init(0);
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

    for (const char *tp = thr_list; *tp;) {
        const int want = atoi(tp);
        const int nth = mynah_slm_threads_init(want);
        printf("\nbench_decode | %s | %u layers, d %u, ff %u, vocab %u | position %u | "
               "%d threads | sgemm %s | int8 %s (%s) | KV bf16 | %d rounds, best of 3 steps "
               "per arm per round\n",
               *tmp ? "synthetic 0.6B geometry (noise), local tmp" : path,
               c->n_layers, c->d_model, c->d_ff, c->vocab_size, pos, nth,
               mynah_slm_sgemm_backend(), mynah_slm_matvec_int8_enabled() ? "ON" : "off",
               mynah_slm_matvec_int8_isa(), rounds);
        printf("%-3s", "B");
        for (int a = 0; a < n_arms; a++) printf(" %23s", arm_name[a]);
        printf("   %8s %8s %6s %9s\n", "mv/ws", "solo/ws", "ws win", "ws calls");

        const uint32_t widths[] = { 1, 2, 4, 8 };
        for (size_t wi = 0; wi < sizeof widths / sizeof *widths; wi++) {
            const uint32_t B = widths[wi];
            double *t[N_ARMS];
            for (int a = 0; a < N_ARMS; a++) t[a] = calloc((size_t)rounds, sizeof(double));
            mynah_slm_seq *bp[BMAX];
            uint32_t tok[BMAX];
            for (uint32_t b = 0; b < B; b++) { bp[b] = &seq[b]; tok[b] = 'a' + b; }
            int wins = 0;
            unsigned long ws_calls = 0;

            /* one warm-up round, not recorded */
            for (int round = -1; round < rounds; round++) {
                for (int k = 0; k < n_arms; k++) {
                    const int arm = (k + (round < 0 ? 0 : round)) % n_arms;
                    double best = 1e30;
                    const unsigned long calls0 = mynah_slm_matvec_ws_count();
                    for (int rep = 0; rep < 3; rep++) {
                        for (uint32_t b = 0; b < B; b++) seq[b].n_past = pos;
                        float *lg[BMAX];
                        const double t0 = mynah_slm_now();
                        if (arm == ARM_SOLO) {
                            for (uint32_t b = 0; b < B; b++)
                                mynah_slm_seq_forward(&ws, &seq[b], tok[b], NULL);
                        } else {
                            mynah_slm_decode_product_set(
                                arm == ARM_MATVEC ? MYNAH_SLM_DECODE_MATVEC :
                                arm == ARM_WS     ? MYNAH_SLM_DECODE_WS : MYNAH_SLM_DECODE_MATMAT);
                            mynah_slm_forward_multi(&ws, bp, tok, B, lg);
                        }
                        const double dt = mynah_slm_now() - t0;
                        if (dt < best) best = dt;
                    }
                    if (arm == ARM_WS) ws_calls = mynah_slm_matvec_ws_count() - calls0;
                    if (round >= 0) t[arm][round] = best;
                }
                if (round >= 0 && t[ARM_WS][round] < t[ARM_MATVEC][round]) wins++;
            }
            double med[N_ARMS], lo[N_ARMS], hi[N_ARMS];
            for (int a = 0; a < n_arms; a++) {
                qsort(t[a], (size_t)rounds, sizeof(double), cmp_d);
                med[a] = t[a][rounds / 2];
                lo[a] = t[a][0];
                hi[a] = t[a][rounds - 1];
            }
            printf("%-3u", B);
            for (int a = 0; a < n_arms; a++)
                printf("  %7.2f (%6.2f..%6.2f)", med[a] * 1e3, lo[a] * 1e3, hi[a] * 1e3);
            printf("   %7.2fx %7.2fx %3d/%-2d %9lu\n", med[ARM_MATVEC] / med[ARM_WS],
                   med[ARM_SOLO] / med[ARM_WS], wins, rounds, ws_calls / 3);
            for (int a = 0; a < N_ARMS; a++) free(t[a]);
            fflush(stdout);
        }
        printf("(ms per decode step for all B streams: median (min..max) over the rounds; "
               "mv/ws and solo/ws are ratios of medians; 'ws win' = rounds where ws beat "
               "matvec in the same round; 'ws calls' = weight-stationary kernel calls per "
               "ws step, the proof that path ran — B = 1 is the solo path by design)\n");
        while (*tp && *tp != ',') tp++;
        if (*tp == ',') tp++;
    }

    mynah_slm_decode_product_set(-1);
    for (int b = 0; b < BMAX; b++) mynah_slm_seq_free(&seq[b]);
    mynah_slm_state_free(&ws);
    free(prompt);
    mynah_slm_free(m);
    if (*tmp) unlink(tmp);
    mynah_slm_threads_shutdown();
    return 0;
}
