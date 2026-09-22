/* main.c — ternary GEMV microbenchmark at real Qwen3-0.6B shapes.
 *
 * Deliberately isolated from libmynah_slm's inference path: it LINKS the
 * engine only to call `mynah_slm_matvec`, so the Q4_K/Q6_K/Q8_0 baselines are
 * the production kernels and not a strawman.
 *
 * No benchmark line is printed for a kernel that has not passed correctness
 * against the T0 oracle first.
 *
 * SPDX-License-Identifier: MIT */
#include "formats.h"
#include "qmat.h"
#include "timing.h"
#include <ingot/quant.h>
#include <ingot/dtype.h>

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>

typedef struct {
    int8_t  *q;
    float   *scale;
    int32_t *sum;
    size_t   cols, groups;
} act_t;
void act_prepare(const float *x, size_t cols, act_t *a);
void tgemv_scalar(ternary_fmt f, const uint8_t *w, size_t rows, size_t cols,
                  const act_t *a, float *y);
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
void tgemv_neon(ternary_fmt f, const uint8_t *w, size_t rows, size_t cols,
                const act_t *a, const int8_t *xs2, const int8_t *xs4, float *y);
#endif

/* The dominant decode GEMV shapes, from reports/ternary/traffic_shapes.csv.
 * Real model dimensions, not square toys. */
typedef struct { const char *name; size_t rows, cols; int instances; } shape_t;
static const shape_t SHAPES[] = {
    {"gate/up_proj", 3072, 1024, 56},
    {"down_proj",    1024, 3072, 28},
    {"q_proj",       2048, 1024, 28},
    {"o_proj",       1024, 2048, 28},
    {"k/v_proj",     1024, 1024, 56},
    {"lm_head",    151936, 1024,  1},
};
#define NSHAPES (sizeof(SHAPES) / sizeof(SHAPES[0]))

static uint64_t rng = 0x2545F4914F6CDD1DULL;
static inline float frand(void)
{
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    return (float)((double)(rng >> 11) / 9007199254740992.0) * 2.0f - 1.0f;
}

/* Deinterleave activations so sdot lanes line up with the packed code order.
 * Done ONCE per GEMV, amortised over every row -- which is why a kernel that
 * owns its activation layout beats one that takes whatever it is handed. */
static void shuffle_stride(const int8_t *in, int8_t *out, size_t cols, int blk, int stride)
{
    for (size_t b = 0; b < cols; b += (size_t)blk)
        for (int s = 0; s < stride; s++)
            for (int i = 0; i < blk / stride; i++)
                out[b + (size_t)s * (blk / stride) + i] = in[b + (size_t)i * stride + s];
}

typedef struct {
    int kind, reps; const void *W; size_t rb, r0, r1, cols;
    const act_t *a; const int8_t *xs2, *xs4; float *y;
    const float *xf; int type; const mynah_slm_matvec_in *prep;
} job_t;

/* The repetition loop lives INSIDE the worker. An earlier version spawned and
 * joined the pool once per iteration, and pthread_create costs tens of
 * microseconds against a 40-80 us GEMV -- the "thread scaling" it produced was
 * mostly thread-creation overhead, and at 4 threads it made the small shapes
 * look SLOWER than at 1. */
static void *worker(void *v)
{
    job_t *j = v;
    for (int it = 0; it < j->reps; it++) {
        if (j->kind == 0)
            tgemv_neon(FMT_T3_FOLD9, (const uint8_t *)j->W + j->r0 * j->rb,
                       j->r1 - j->r0, j->cols, j->a, j->xs2, j->xs4, j->y + j->r0);
        else
            mynah_slm_matvec(j->type, (const uint8_t *)j->W + j->r0 * j->rb,
                             j->r1 - j->r0, j->cols, j->xf, j->prep, j->y + j->r0);
    }
    return NULL;
}

static double bench_threads(int nt, int reps, int kind, const void *W, size_t rb,
                            size_t R, size_t C, const act_t *a, const int8_t *xs2,
                            const int8_t *xs4, float *y, const float *xf, int type,
                            const mynah_slm_matvec_in *prep)
{
    pthread_t th[8];
    job_t     jb[8];
    double t0 = mynah_slm_now();
    for (int t = 0; t < nt; t++) {
        jb[t] = (job_t){kind, reps, W, rb, R * t / nt, R * (t + 1) / nt, C,
                        a, xs2, xs4, y, xf, type, prep};
        pthread_create(&th[t], NULL, worker, &jb[t]);
    }
    for (int t = 0; t < nt; t++) pthread_join(th[t], NULL);
    return (mynah_slm_now() - t0) * 1e9 / reps;
}

int main(int argc, char **argv)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    int reps = argc > 1 ? atoi(argv[1]) : 40;
    printf("# ternary GEMV microbench -- Apple M1, sdot yes / i8mm no, %d reps\n", reps);
    printf("# baselines are mynah_slm_matvec, the production kernels\n\n");

    printf("## representation\n");
    printf("%-14s %9s %9s %9s %7s\n", "format", "payload", "scales", "TOTAL", "passes");
    for (int f = 0; f < FMT__COUNT; f++)
        printf("%-14s %9.3f %9.3f %9.3f %7d\n", TERNARY_FORMATS[f].name,
               TERNARY_FORMATS[f].payload_bpw, TERNARY_FORMATS[f].scale_bpw,
               TERNARY_FORMATS[f].total_bpw, TERNARY_FORMATS[f].planes);
    printf("\n(bits/weight. Entropy of one trit is 1.585 and is NOT a storage rate.)\n\n");

    FILE *csv = fopen("reports/ternary-kernel/mac_bench.csv", "w");
    FILE *cc  = fopen("reports/ternary-kernel/correctness.csv", "w");
    if (csv) fprintf(csv, "shape,rows,cols,kernel,bpw,ns_per_gemv,weight_MiB,GB_s\n");
    if (cc)  fprintf(cc, "shape,format,max_abs,mean_abs,rel_l2\n");

    /* How much of a GEMV is activation preparation? The ternary kernels are
     * timed consuming int8 activations that were quantized and lane-shuffled
     * outside the loop. That work is real; in an engine it is done once per
     * layer input and amortised over the 3-7 projections that share it, but a
     * benchmark that never states its cost is hiding something. */
    {
        size_t C = 1024;
        float *x = malloc(C * sizeof(float));
        for (size_t i = 0; i < C; i++) x[i] = frand();
        act_t a = {malloc(C), malloc(C / TG * sizeof(float)),
                   malloc(C / TG * sizeof(int32_t)), C, C / TG};
        int8_t *s2 = malloc(C), *s4 = malloc(C);
        double t0 = mynah_slm_now();
        for (int it = 0; it < reps * 50; it++) {
            act_prepare(x, C, &a);
            shuffle_stride(a.q, s2, C, 64, 4);
            shuffle_stride(a.q, s4, C, 32, 2);
        }
        double ns = (mynah_slm_now() - t0) * 1e9 / (reps * 50);
        printf("## activation prep + lane shuffle, cols=1024: %.0f ns "
               "(once per layer input, not per projection)\n\n", ns);
        free(x); free(a.q); free(a.scale); free(a.sum); free(s2); free(s4);
    }

    for (size_t s = 0; s < NSHAPES; s++) {
        size_t R = SHAPES[s].rows, C = SHAPES[s].cols;
        printf("## %s  [%zu x %zu]\n", SHAPES[s].name, R, C);

        float *x = malloc(C * sizeof(float));
        float *y = malloc(R * sizeof(float));
        float *yref = malloc(R * sizeof(float));
        for (size_t i = 0; i < C; i++) x[i] = frand();

        act_t a = {malloc(C), malloc(C / TG * sizeof(float)),
                   malloc(C / TG * sizeof(int32_t)), C, C / TG};
        act_prepare(x, C, &a);
        int8_t *xs2 = malloc(C), *xs4 = malloc(C);
        shuffle_stride(a.q, xs2, C, 64, 4);
        shuffle_stride(a.q, xs4, C, 32, 2);

        /* one set of trits, encoded into every format, so the comparison is
         * between REPRESENTATIONS and not between random draws */
        int8_t *t1 = malloc(R * C), *tk2 = malloc(R * C), *tk3 = malloc(R * C);
        float  *ws = malloc(R * (C / TG) * sizeof(float));
        for (size_t i = 0; i < R * C; i++) {
            float v = frand();
            t1[i]  = (int8_t)(v > 0.33f ? 1 : (v < -0.33f ? -1 : 0));
            tk2[i] = (int8_t)(lrintf(v * 4.0f));            /* [-4,4]   */
            tk3[i] = (int8_t)(lrintf(v * 13.0f));           /* [-13,13] */
        }
        for (size_t i = 0; i < R * (C / TG); i++) ws[i] = 0.01f + 0.001f * frand();

        struct { ternary_fmt f; const int8_t *t; } cfg[] = {
            {FMT_T0_INT8, t1}, {FMT_T1_2BIT, t1}, {FMT_T2_BASE3, t1},
            {FMT_T3_FOLD9, tk2}, {FMT_T3_K3, tk3}, {FMT_MASKSIGN, t1},
        };

        for (size_t k = 0; k < sizeof(cfg) / sizeof(cfg[0]); k++) {
            ternary_fmt f = cfg[k].f;
            size_t rb = ternary_row_bytes(f, C);
            uint8_t *W = malloc(R * rb);
            for (size_t r = 0; r < R; r++)
                ternary_encode_row(f, cfg[k].t + r * C, ws + r * (C / TG), C, W + r * rb);

            /* ORACLE. Comparing the NEON kernel against a scalar that shares
             * its decode only proves the two agree; it cannot catch a format
             * whose encode/decode round-trip is wrong. So every format is
             * checked against T0 -- which stores the values verbatim as int8 --
             * built from the SAME value array. That is a real round-trip test.
             * T2 and mask+sign carry only {-1,0,1}, so they are checked against
             * the T0 encoding of t1; T3 formats against T0 of their own array. */
            {
                size_t orb = ternary_row_bytes(FMT_T0_INT8, C);
                uint8_t *O = malloc(R * orb);
                for (size_t r = 0; r < R; r++)
                    ternary_encode_row(FMT_T0_INT8, cfg[k].t + r * C,
                                       ws + r * (C / TG), C, O + r * orb);
                tgemv_scalar(FMT_T0_INT8, O, R, C, &a, y);
                tgemv_scalar(f, W, R, C, &a, yref);
                double mx = 0, num = 0, den = 0;
                for (size_t r = 0; r < R; r++) {
                    double d = fabs((double)yref[r] - y[r]);
                    mx = d > mx ? d : mx;
                    num += d * d;
                    den += (double)y[r] * y[r];
                }
                double rel = den > 0 ? sqrt(num / den) : 0;
                printf("  %-14s round-trip vs T0 oracle: max %.3e  rel_l2 %.3e %s\n",
                       TERNARY_FORMATS[f].name, mx, rel,
                       rel < 1e-6 ? "OK" : "*** ENCODE/DECODE MISMATCH ***");
                if (cc) fprintf(cc, "%s,%s-roundtrip,%.6e,%.6e,%.6e\n",
                                SHAPES[s].name, TERNARY_FORMATS[f].name, mx, 0.0, rel);
                free(O);
                if (rel >= 1e-6) { free(W); continue; }
            }
            tgemv_scalar(f, W, R, C, &a, yref);

            double t0 = mynah_slm_now();
            for (int it = 0; it < reps; it++) tgemv_scalar(f, W, R, C, &a, yref);
            double sc_ns = (mynah_slm_now() - t0) * 1e9 / reps;

            double ne_ns = 0.0;
            int have_neon = 0;
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
            if (f != FMT_T2_BASE3 && f != FMT_MASKSIGN) {
                have_neon = 1;
                tgemv_neon(f, W, R, C, &a, xs2, xs4, y);
                double mx = 0, sum = 0, num = 0, den = 0;
                for (size_t r = 0; r < R; r++) {
                    double d = fabs((double)y[r] - yref[r]);
                    mx = d > mx ? d : mx; sum += d;
                    num += d * d; den += (double)yref[r] * yref[r];
                }
                double rel = den > 0 ? sqrt(num / den) : 0;
                printf("  %-14s neon vs scalar: max %.3e  mean %.3e  rel_l2 %.3e %s\n",
                       TERNARY_FORMATS[f].name, mx, sum / R, rel,
                       rel < 1e-5 ? "OK" : "*** MISMATCH ***");
                if (cc) fprintf(cc, "%s,%s,%.6e,%.6e,%.6e\n", SHAPES[s].name,
                                TERNARY_FORMATS[f].name, mx, sum / R, rel);
                if (rel >= 1e-5) have_neon = 0;   /* refuse to time a wrong kernel */
                else {
                    t0 = mynah_slm_now();
                    for (int it = 0; it < reps; it++)
                        tgemv_neon(f, W, R, C, &a, xs2, xs4, y);
                    ne_ns = (mynah_slm_now() - t0) * 1e9 / reps;
                }
            }
#endif
            double mib = (double)(R * rb) / 1048576.0;
            printf("    %-12s %6.3f bpw  scalar %9.0f ns", TERNARY_FORMATS[f].name,
                   TERNARY_FORMATS[f].total_bpw, sc_ns);
            if (have_neon)
                printf("   NEON %9.0f ns  %6.1f MiB  %6.1f GB/s",
                       ne_ns, mib, (double)(R * rb) / ne_ns);
            printf("\n");
            if (csv) {
                fprintf(csv, "%s,%zu,%zu,%s-scalar,%.4f,%.0f,%.2f,%.2f\n", SHAPES[s].name,
                        R, C, TERNARY_FORMATS[f].name, TERNARY_FORMATS[f].total_bpw,
                        sc_ns, mib, (double)(R * rb) / sc_ns);
                if (have_neon)
                    fprintf(csv, "%s,%zu,%zu,%s-neon,%.4f,%.0f,%.2f,%.2f\n", SHAPES[s].name,
                            R, C, TERNARY_FORMATS[f].name, TERNARY_FORMATS[f].total_bpw,
                            ne_ns, mib, (double)(R * rb) / ne_ns);
            }
            free(W);
        }

        /* Production baselines: the real kernels, same f32 in / f32 out.
         *
         * The row stride MUST be ingot's exact row size. An earlier version
         * padded each row by 64 bytes and the engine, which assumes contiguous
         * ggml rows, read straight through the padding -- Q8_0 and Q6_K came
         * back in 0 ns, which is not a fast kernel, it is a kernel that did not
         * run. Every baseline now checks its return value and refuses to be
         * timed if the engine declined it. */
        int types[] = {INGOT_TYPE_Q8_0, INGOT_TYPE_Q6_K, INGOT_TYPE_Q4_K, INGOT_TYPE_Q3_K};
        const char *tn[] = {"Q8_0", "Q6_K", "Q4_K", "Q3_K"};
        float *wf = malloc(C * sizeof(float));
        for (size_t i = 0; i < C; i++) wf[i] = 0.02f * frand();
        for (size_t ti = 0; ti < 4; ti++) {
            uint64_t rb64 = 0;
            if (!ingot_can_quantize(types[ti]) ||
                ingot_type_nbytes(types[ti], (uint64_t)C, &rb64) != 0 || rb64 == 0) {
                printf("    %-12s REFUSED: no encoder or unknown geometry\n", tn[ti]);
                continue;
            }
            size_t rb = (size_t)rb64;
            double bpw = (double)rb * 8.0 / (double)C;
            uint8_t *W = calloc(R, rb);
            int enc_ok = 1;
            for (size_t r = 0; r < R && enc_ok; r++)
                if (ingot_quantize(types[ti], wf, C, W + r * rb) != 0) enc_ok = 0;
            if (!enc_ok) { printf("    %-12s REFUSED: encode failed\n", tn[ti]); free(W); continue; }

            /* mynah_slm_matvec returns 0 when OUR kernel took it and non-zero
             * when the caller must fall back to ingot's -- that is the contract,
             * not a refusal. Which path ran is printed, because a benchmark that
             * cannot name its own kernel is not evidence. */
            /* TWO ARMS. Our ternary kernels consume int8 activations, so
             * timing them against the engine's DEFAULT f32 activation path
             * would be comparing an int8 kernel with a float one and calling
             * the difference a format win. The int8 arm is what --fast turns
             * on, and it is the like-for-like comparison. */
            mynah_slm_matvec_in prep;
            for (int i8 = 0; i8 < 2; i8++) {
            mynah_slm_matvec_set_int8(i8);
            if (i8 && !mynah_slm_matvec_int8_enabled()) continue;
            mynah_slm_matvec_prepare(x, C, &prep);
            int ours = (mynah_slm_matvec(types[ti], W, R, C, x, &prep, y) == 0);
            const char *who = ours ? "mynah" : (ingot_has_kernel(types[ti]) ? "ingot-simd"
                                                                            : "ingot-generic");
            double t0 = mynah_slm_now();
            if (ours)
                for (int it = 0; it < reps; it++)
                    mynah_slm_matvec(types[ti], W, R, C, x, &prep, y);
            else
                for (int it = 0; it < reps; it++)
                    ingot_matvec(types[ti], W, R, C, x, y);
            double ns = (mynah_slm_now() - t0) * 1e9 / reps;
            double mib = (double)(R * rb) / 1048576.0;
            printf("    %-12s %6.3f bpw  BASELINE [%-13s act=%s] %9.0f ns  %6.1f MiB  %6.1f GB/s\n",
                   tn[ti], bpw, who, i8 ? "int8" : "f32 ", ns, mib,
                   (double)(R * rb) / ns);
            if (csv) fprintf(csv, "%s,%zu,%zu,%s-%s-act%s,%.4f,%.0f,%.2f,%.2f\n",
                             SHAPES[s].name, R, C, tn[ti], who, i8 ? "int8" : "f32",
                             bpw, ns, mib, (double)(R * rb) / ns);
            }
            mynah_slm_matvec_set_int8(0);
            free(W);
        }
        free(wf);

        /* Thread scaling on the headline pair: T3 fold9 (ours) against Q4_K
         * int8 (the engine's). Rows split evenly; each worker owns a disjoint
         * output slice, so nothing reduces across threads. */
        {
            size_t frb = ternary_row_bytes(FMT_T3_FOLD9, C);
            uint8_t *FW = malloc(R * frb);
            for (size_t r = 0; r < R; r++)
                ternary_encode_row(FMT_T3_FOLD9, tk2 + r * C, ws + r * (C / TG), C,
                                   FW + r * frb);
            uint64_t qrb64 = 0;
            ingot_type_nbytes(INGOT_TYPE_Q4_K, (uint64_t)C, &qrb64);
            uint8_t *QW = calloc(R, (size_t)qrb64);
            float *wq = malloc(C * sizeof(float));
            for (size_t i = 0; i < C; i++) wq[i] = 0.02f * frand();
            for (size_t r = 0; r < R; r++)
                ingot_quantize(INGOT_TYPE_Q4_K, wq, C, QW + r * (size_t)qrb64);
            mynah_slm_matvec_set_int8(1);
            mynah_slm_matvec_in prep;
            mynah_slm_matvec_prepare(x, C, &prep);

            for (int nt = 1; nt <= 4; nt *= 2) {
                double tern = bench_threads(nt, reps, 0, FW, frb, R, C, &a, xs2, xs4, y,
                                            NULL, 0, NULL);
                double q4 = bench_threads(nt, reps, 1, QW, (size_t)qrb64, R, C, &a,
                                          xs2, xs4, y, x, INGOT_TYPE_Q4_K, &prep);
                printf("    threads=%d   T3 fold9 %9.0f ns   Q4_K(int8) %9.0f ns   "
                       "ternary is %.2fx\n", nt, tern, q4, q4 / tern);
                if (csv) fprintf(csv, "%s,%zu,%zu,threads%d-fold9,%.4f,%.0f,0,0\n",
                                 SHAPES[s].name, R, C, nt, 4.125, tern);
                if (csv) fprintf(csv, "%s,%zu,%zu,threads%d-Q4_K,%.4f,%.0f,0,0\n",
                                 SHAPES[s].name, R, C, nt, 4.5, q4);
            }
            mynah_slm_matvec_set_int8(0);
            free(FW); free(QW); free(wq);
        }
        printf("\n");
        free(x); free(y); free(yref); free(a.q); free(a.scale); free(a.sum);
        free(xs2); free(xs4); free(t1); free(tk2); free(tk3); free(ws);
    }
    if (csv) fclose(csv);
    if (cc) fclose(cc);
    return 0;
}
