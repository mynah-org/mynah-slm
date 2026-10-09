/* test_sgemm.c — our f32 GEMM against its definition, and against itself.
 *
 * Three properties, each one a way a plausible GEMM is wrong:
 *
 *   1. AGREES WITH THE REFERENCE. The double-accumulating triple loop, to a
 *      relative bound, over the shapes the three call sites actually produce
 *      plus the edges the tiling can get wrong (k not a multiple of the vector
 *      width, m or n not a multiple of the tile, strided lda/ldb/ldc).
 *   2. BIT-IDENTICAL ACROSS THREAD COUNTS. memcmp, not a tolerance: sgemm.h
 *      claims every element is computed by the same operation sequence
 *      whatever tile or thread holds it, and this is where that claim is
 *      either true or not.
 *   3. beta == 0 NEVER READS C. A NaN-filled C must come out clean.
 *
 * Plus a refusal check: a forced own path that silently ran nothing would
 * make every comparison above vacuous, so the counters must show the
 * families ran.
 *
 * `tests/test_sgemm bench` times the call-site shapes, own vs vendor when the
 * build linked one, INTERLEAVED in one process (.work/engineering-method.md).
 *
 * No model, no network.
 *
 * SPDX-License-Identifier: MIT */
#include "sgemm.h"
#include "threads.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int failures;

static void check(const char *what, int ok, const char *detail) {
    if (ok) printf("ok   %s\n", what);
    else  { printf("FAIL %s  <- %s\n", what, detail ? detail : ""); failures++; }
}

static uint32_t rng = 0x9E3779B9u;
static float frand(void) {
    rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
    return (float)((int32_t)rng) / 2147483648.0f;
}

typedef struct { int tb; size_t m, n, k, lda, ldb, ldc; const char *name; } shape;

/* The call sites (sgemm.h) at Qwen3-0.6B sizes, then the edges. */
static const shape SHAPES[] = {
    /* qmatmat: out[T][128] = in[T][cols] . strip[128][cols]^T, ldc = rows */
    { 1,   1, 128, 1024, 1024, 1024, 3072, "qmatmat T=1"   },
    { 1,   7, 128, 1024, 1024, 1024, 3072, "qmatmat T=7"   },
    { 1,  64, 128, 1024, 1024, 1024, 1024, "qmatmat T=64"  },
    { 1,  33, 128, 3072, 3072, 3072, 1024, "qmatmat T=33 ffn_down" },
    /* attention QK^T: strided q (q_stride = 2048) and k (kv_dim = 1024) */
    { 1,  19,  19,  128, 2048, 1024,   19, "QK^T n_q=19"   },
    { 1,  64, 300,  128, 2048, 1024,  300, "QK^T n_q=64 n_kv=300" },
    /* attention S.V: v strided by kv_dim, out strided by q_stride */
    { 0,  19, 128,   19,   19, 1024, 2048, "SV n_q=19"     },
    { 0,  64, 128,  300,  300, 1024, 2048, "SV n_q=64 n_kv=300" },
    /* edges */
    { 1,   5,   3,   13,   13,   13,    3, "NT ragged k=13" },
    { 1,   9,  11,   37,   40,   41,   12, "NT ragged strided" },
    { 0,   9,  11,   37,   40,   12,   12, "NN ragged n=11" },
    { 0,   1,  50,    7,    7,   50,   50, "NN m=1"         },
    { 0,  70,   3,    5,    5,    3,    3, "NN n=3 < lanes" },
    { 1,   1,   1,    1,    1,    1,    1, "1x1x1"          },
};
#define N_SHAPES (sizeof SHAPES / sizeof SHAPES[0])

static float *alloc_fill(size_t n) {
    float *p = malloc(n * sizeof *p);
    for (size_t i = 0; i < n; i++) p[i] = frand();
    return p;
}

static void run_case(const shape *s, float alpha, float beta) {
    const size_t a_n = s->m * s->lda;
    const size_t b_n = s->tb ? s->n * s->ldb : s->k * s->ldb;
    const size_t c_n = s->m * s->ldc;
    float *a = alloc_fill(a_n), *b = alloc_fill(b_n), *c0 = alloc_fill(c_n);
    float *ref = malloc(c_n * sizeof *ref), *own = malloc(c_n * sizeof *own);
    float *own1 = malloc(c_n * sizeof *own1);

    memcpy(ref, c0, c_n * sizeof *ref);
    mynah_slm_sgemm_reference(s->tb, s->m, s->n, s->k, alpha, a, s->lda, b, s->ldb, beta, ref, s->ldc);

    /* beta == 0: C goes in poisoned and must come out clean. */
    if (beta == 0.0f) for (size_t i = 0; i < c_n; i++) c0[i] = NAN;

    char what[160], detail[160];

    mynah_slm_threads_init(4);
    memcpy(own, c0, c_n * sizeof *own);
    const int rc = mynah_slm_sgemm_own(s->tb, s->m, s->n, s->k, alpha, a, s->lda, b, s->ldb, beta, own, s->ldc);

    /* Relative to the magnitude a k-long dot of [-1,1] values can reach. */
    double worst = 0.0;
    int nan_seen = 0;
    for (size_t i = 0; i < s->m; i++)
        for (size_t j = 0; j < s->n; j++) {
            const float g = own[i * s->ldc + j], w = ref[i * s->ldc + j];
            if (isnan(g)) nan_seen = 1;
            const double e = fabs((double)g - (double)w) / (sqrt((double)s->k) + fabs((double)w));
            if (e > worst) worst = e;
        }
    snprintf(what, sizeof what, "%-26s alpha=%.1f beta=%.1f vs reference", s->name, alpha, beta);
    snprintf(detail, sizeof detail, "rc=%d nan=%d rel=%.3g", rc, nan_seen, worst);
    check(what, rc == 0 && !nan_seen && worst < 1e-5, detail);

    /* Elements outside the m x n window (ldc padding) must be untouched. */
    int pad_ok = 1;
    for (size_t i = 0; i < s->m; i++)
        for (size_t j = s->n; j < s->ldc; j++) {
            const float g = own[i * s->ldc + j], w = c0[i * s->ldc + j];
            if (memcmp(&g, &w, sizeof g) != 0) pad_ok = 0;
        }
    snprintf(what, sizeof what, "%-26s ldc padding untouched", s->name);
    check(what, pad_ok, "wrote outside the n columns");

    /* Thread count must not move a single bit. */
    mynah_slm_threads_init(1);
    memcpy(own1, c0, c_n * sizeof *own1);
    mynah_slm_sgemm_own(s->tb, s->m, s->n, s->k, alpha, a, s->lda, b, s->ldb, beta, own1, s->ldc);
    int same = 1;
    for (size_t i = 0; i < s->m && same; i++)
        if (memcmp(own + i * s->ldc, own1 + i * s->ldc, s->n * sizeof *own) != 0) same = 0;
    snprintf(what, sizeof what, "%-26s 1 thread == 4 threads, memcmp", s->name);
    check(what, same, "thread count changed the result");

    free(a); free(b); free(c0); free(ref); free(own); free(own1);
}

/* An element must not depend on where in the matrix it sits: compute a
 * 1x1 problem on its own and compare it with the same row/column inside a
 * large, multi-tile, multi-thread problem. */
static void tile_invariance(int tb, size_t m, size_t n, size_t k, float beta) {
    /* alpha and beta both non-trivial: with beta != 0 the store reads C and
     * its rounding is part of what must not depend on the tile. */
    const float alpha = 0.3f;
    float *a = alloc_fill(m * k), *b = alloc_fill(n * k), *c0 = alloc_fill(m * n);
    float *c = malloc(m * n * sizeof *c);
    memcpy(c, c0, m * n * sizeof *c);
    mynah_slm_threads_init(4);
    mynah_slm_sgemm_own(tb, m, n, k, alpha, a, k, b, tb ? k : n, beta, c, n);
    int same = 1;
    for (size_t i = 0; i < m; i++)
        for (size_t j = 0; j < n; j++) {
            float one = c0[i * n + j];
            if (tb) mynah_slm_sgemm_own(1, 1, 1, k, alpha, a + i * k, k, b + j * k, k, beta, &one, 1);
            else {
                /* op(B) column j as a 1-wide B: stride n picks it out. */
                mynah_slm_sgemm_own(0, 1, 1, k, alpha, a + i * k, k, b + j, n, beta, &one, 1);
            }
            if (memcmp(&one, &c[i * n + j], sizeof one) != 0) same = 0;
        }
    char what[128];
    snprintf(what, sizeof what, "%s %zux%zux%zu beta=%.1f: element == same element computed alone",
             tb ? "NT" : "NN", m, n, k, beta);
    check(what, same, "an element's value depends on its tile");
    free(a); free(b); free(c); free(c0);
}

/* The NN k-block parks a raw f32 partial in C. With beta == 0 the blocked
 * path runs; with beta == 1 over a zeroed C the unblocked one does, and
 * alpha*acc + 1*0 is alpha*acc exactly. memcmp, not a tolerance. */
static void kblock_exact(void) {
    const size_t m = 13, n = 45, k = 1031;
    float *a = alloc_fill(m * k), *b = alloc_fill(k * n);
    float *c0 = malloc(m * n * sizeof *c0), *c1 = calloc(m * n, sizeof *c1);
    mynah_slm_threads_init(4);
    mynah_slm_sgemm_own(0, m, n, k, 0.75f, a, k, b, n, 0.0f, c0, n);
    mynah_slm_sgemm_own(0, m, n, k, 0.75f, a, k, b, n, 1.0f, c1, n);
    check("NN k-blocked (beta 0) == unblocked (beta 1, C = 0), memcmp",
          memcmp(c0, c1, m * n * sizeof *c0) == 0, "parking the accumulator in C changed bits");
    free(a); free(b); free(c0); free(c1);
}

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

/* The bench shapes: what Qwen3-0.6B (d_model 1024, d_ff 3072, 16 q heads
 * and 8 kv heads of 128) actually issues during prefill, at the default
 * batch width of 256 and at a short prompt tail. */
static const shape BENCH[] = {
    { 1, 256, 128, 1024, 1024, 1024, 2048, "qmatmat q_proj T=256"   },
    { 1, 256, 128, 3072, 3072, 3072, 1024, "qmatmat ffn_down T=256" },
    { 1,  64, 128, 1024, 1024, 1024, 3072, "qmatmat ffn_up T=64"    },
    { 1,  16, 128, 1024, 1024, 1024, 3072, "qmatmat ffn_up T=16"    },
    { 1,   7, 128, 1024, 1024, 1024, 3072, "qmatmat ffn_up T=7"     },
    { 1, 256, 256,  128, 2048, 1024,  256, "QK^T n_q=256 n_kv=256"  },
    { 1, 256, 2048, 128, 2048, 1024, 2048, "QK^T n_q=256 n_kv=2048" },
    { 0, 256, 128,  256,  256, 1024, 2048, "SV n_q=256 n_kv=256"    },
    { 0, 256, 128, 2048, 2048, 1024, 2048, "SV n_q=256 n_kv=2048"   },
};

static int cmp_d(const void *x, const void *y) {
    const double a = *(const double *)x, b = *(const double *)y;
    return (a > b) - (a < b);
}

static double median(double *v, int n) {
    qsort(v, (size_t)n, sizeof *v, cmp_d);
    return (n & 1) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

/* ROUNDS independent rounds; inside each, own and vendor are timed back to
 * back with the order alternating every round, so a drift in machine state
 * (a noisy neighbour on a cloud VM, a frequency step) lands on both sides.
 * Reported: median GFLOP/s per side, the spread (min..max) of own, the
 * coefficient of variation of each side, the median of the PER-ROUND ratio,
 * and in how many rounds own won. A row is called STABLE only if both CVs
 * are under 5% and one side won at least 90% of the rounds. */
#define ROUNDS 15

static int bench(int threads) {
    mynah_slm_threads_init(threads);
    const int have_vendor = strcmp(mynah_slm_sgemm_backend(), "own") != 0;
    printf("# sgemm bench: isa=%s threads=%d vendor=%s rounds=%d, f32, GFLOP/s\n",
           mynah_slm_sgemm_isa(), mynah_slm_threads_count(),
           have_vendor ? mynah_slm_sgemm_backend() : "none", ROUNDS);
    printf("| shape | M | N | K | op(B) | thr | own med | own min..max | own CV | vendor med | vendor CV | ratio med | own wins | stable |\n");
    printf("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n");
    for (size_t si = 0; si < sizeof BENCH / sizeof BENCH[0]; si++) {
        const shape *s = &BENCH[si];
        const size_t b_n = s->tb ? s->n * s->ldb : s->k * s->ldb;
        float *a = alloc_fill(s->m * s->lda), *b = alloc_fill(b_n), *c = alloc_fill(s->m * s->ldc);
        const double flop = 2.0 * (double)s->m * (double)s->n * (double)s->k;
        int reps = (int)(3e8 / flop); if (reps < 3) reps = 3; if (reps > 20000) reps = 20000;

        double own[ROUNDS], ven[ROUNDS], ratio[ROUNDS];
        int wins = 0;
        for (int r = 0; r < 3; r++)   /* warm-up: pages, pool, vendor init */
            mynah_slm_sgemm_own(s->tb, s->m, s->n, s->k, 1.0f, a, s->lda, b, s->ldb, 0.0f, c, s->ldc);
        for (int round = 0; round < ROUNDS; round++) {
            for (int side = 0; side < 2; side++) {
                const int vendor = have_vendor && (side ^ (round & 1));
                if (!have_vendor && side == 1) break;
                const double t0 = now_s();
                for (int r = 0; r < reps; r++) {
                    if (vendor) mynah_slm_sgemm(s->tb, s->m, s->n, s->k, 1.0f, a, s->lda, b, s->ldb, 0.0f, c, s->ldc);
                    else        mynah_slm_sgemm_own(s->tb, s->m, s->n, s->k, 1.0f, a, s->lda, b, s->ldb, 0.0f, c, s->ldc);
                }
                const double gf = flop * reps / (now_s() - t0) * 1e-9;
                if (vendor) ven[round] = gf; else own[round] = gf;
            }
            if (have_vendor) { ratio[round] = own[round] / ven[round]; wins += own[round] > ven[round]; }
        }
        double mn = own[0], mx = own[0], so = 0, so2 = 0, sv = 0, sv2 = 0;
        for (int r = 0; r < ROUNDS; r++) {
            if (own[r] < mn) mn = own[r];
            if (own[r] > mx) mx = own[r];
            so += own[r]; so2 += own[r] * own[r];
            if (have_vendor) { sv += ven[r]; sv2 += ven[r] * ven[r]; }
        }
        const double cv_o = sqrt(fmax(0.0, so2 / ROUNDS - (so / ROUNDS) * (so / ROUNDS))) / (so / ROUNDS);
        const double cv_v = have_vendor ? sqrt(fmax(0.0, sv2 / ROUNDS - (sv / ROUNDS) * (sv / ROUNDS))) / (sv / ROUNDS) : 0.0;
        const double own_med = median(own, ROUNDS);
        if (have_vendor) {
            const double v_med = median(ven, ROUNDS), r_med = median(ratio, ROUNDS);
            const int stable = cv_o < 0.05 && cv_v < 0.05 && (wins * 10 >= ROUNDS * 9 || wins * 10 <= ROUNDS);
            printf("| %s | %zu | %zu | %zu | %s | %d | %.1f | %.1f..%.1f | %.1f%% | %.1f | %.1f%% | %.2fx | %d/%d | %s |\n",
                   s->name, s->m, s->n, s->k, s->tb ? "B^T" : "B", mynah_slm_threads_count(),
                   own_med, mn, mx, 100 * cv_o, v_med, 100 * cv_v, r_med, wins, ROUNDS,
                   stable ? "yes" : "NO");
        } else {
            printf("| %s | %zu | %zu | %zu | %s | %d | %.1f | %.1f..%.1f | %.1f%% | - | - | - | - | %s |\n",
                   s->name, s->m, s->n, s->k, s->tb ? "B^T" : "B", mynah_slm_threads_count(),
                   own_med, mn, mx, 100 * cv_o, cv_o < 0.05 ? "yes" : "NO");
        }
        fflush(stdout);
        free(a); free(b); free(c);
    }
    mynah_slm_sgemm_stats st;
    mynah_slm_sgemm_stats_get(&st);
    printf("# ran: calls=%llu own_nt=%llu own_nn=%llu vendor=%llu reference=%llu\n",
           st.calls, st.own_nt, st.own_nn, st.vendor, st.reference);
    return 0;
}

int main(int argc, char **argv) {
    if (argc > 1 && strcmp(argv[1], "bench") == 0)
        return bench(argc > 2 ? atoi(argv[2]) : 0);

    printf("isa=%s backend=%s\n", mynah_slm_sgemm_isa(), mynah_slm_sgemm_backend());
    mynah_slm_sgemm_stats_reset();

    for (size_t i = 0; i < N_SHAPES; i++) {
        run_case(&SHAPES[i], 1.0f, 0.0f);
        run_case(&SHAPES[i], 0.5f, 0.75f);
    }
    tile_invariance(1, 23, 37, 131, 0.0f);
    tile_invariance(0, 23, 37, 131, 0.0f);
    tile_invariance(1, 37, 75, 131, 0.7f);   /* beta != 0: the store's rounding */
    tile_invariance(0, 37, 75, 131, 0.7f);
    tile_invariance(1, 50, 9, 3072, 0.0f);   /* several NT row chunks */
    tile_invariance(0, 9, 70, 1000, 0.0f);   /* several NN k-blocks, ragged last */
    kblock_exact();

    float dummy = 0.0f;
    check("bad ldc is refused, not computed",
          mynah_slm_sgemm_own(1, 2, 4, 4, 1.0f, &dummy, 4, &dummy, 4, 0.0f, &dummy, 3) == -1, NULL);

    mynah_slm_sgemm_stats st;
    mynah_slm_sgemm_stats_get(&st);
    char d[96];
    snprintf(d, sizeof d, "nt=%llu nn=%llu", st.own_nt, st.own_nn);
    check("both own families actually ran", st.own_nt > 0 && st.own_nn > 0, d);

    mynah_slm_threads_shutdown();
    printf("%s (%d failure%s)\n", failures ? "FAILED" : "PASSED", failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
