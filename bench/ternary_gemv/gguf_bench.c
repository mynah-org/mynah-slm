/* gguf_bench.c -- benchmark the REAL production path on REAL GGUF tensors.
 *
 * Why this file exists: ingot has no Q3_K encoder, so the synthetic harness in
 * main.c can only REFUSE to benchmark Q3_K. Q3_K_M is the format that wins our
 * quality gate, which makes "we could not measure it" the wrong answer. This
 * mode therefore skips encoding entirely and multiplies straight off the bytes
 * of a real quantized checkpoint, through `mynah_slm_matvec` with its
 * documented fallback to `ingot_matvec` -- the same call the engine makes.
 *
 * It also settles something a synthetic bench structurally cannot see: a
 * "Q3_K_M" file is a RECIPE, and the type each tensor actually got is read
 * from the container rather than assumed.
 *
 * SPDX-License-Identifier: MIT */
#include "formats.h"
#include "qmat.h"
#include "timing.h"
#include <ingot/gguf.h>
#include <ingot/quant.h>
#include <ingot/dtype.h>

#include <math.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int8_t  *q;
    float   *scale;
    int32_t *sum;
    size_t   cols, groups;
} act_t;
void act_prepare(const float *x, size_t cols, act_t *a);
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
void tgemv_neon(ternary_fmt f, const uint8_t *w, size_t rows, size_t cols,
                const act_t *a, const int8_t *xs2, const int8_t *xs4, float *y);
#endif
void shuffle_stride(const int8_t *in, int8_t *out, size_t cols, int blk, int stride);
float bench_frand(void);

/* One representative tensor per decode GEMV shape. Layer 5 rather than layer 0
 * because llama.cpp bumps the type of early layers, and quoting a bumped layer
 * as "the" type would misdescribe the other 27. */
static const struct { const char *label, *tensor; } PICKS[] = {
    {"gate/up_proj", "blk.5.ffn_gate.weight"},
    {"down_proj",    "blk.5.ffn_down.weight"},
    {"q_proj",       "blk.5.attn_q.weight"},
    {"o_proj",       "blk.5.attn_output.weight"},
    {"k_proj",       "blk.5.attn_k.weight"},
    {"v_proj",       "blk.5.attn_v.weight"},
    {"lm_head",      "output.weight"},
};
#define NPICKS (sizeof(PICKS) / sizeof(PICKS[0]))

typedef struct {
    int kind, reps;
    const void *W; size_t rb, r0, r1, cols;
    const act_t *a; const int8_t *xs2, *xs4; float *y;
    const float *xf; int type; const mynah_slm_matvec_in *prep; int ours;
} gjob_t;

static void *gworker(void *v)
{
    gjob_t *j = v;
    for (int it = 0; it < j->reps; it++) {
        if (j->kind == 0)
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
            tgemv_neon(FMT_T3_FOLD9, (const uint8_t *)j->W + j->r0 * j->rb,
                       j->r1 - j->r0, j->cols, j->a, j->xs2, j->xs4, j->y + j->r0);
#else
            (void)0;
#endif
        else if (j->kind == 1)
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
            tgemv_neon(FMT_T1_2BIT, (const uint8_t *)j->W + j->r0 * j->rb,
                       j->r1 - j->r0, j->cols, j->a, j->xs2, j->xs4, j->y + j->r0);
#else
            (void)0;
#endif
        else if (j->ours)
            mynah_slm_matvec(j->type, (const uint8_t *)j->W + j->r0 * j->rb,
                             j->r1 - j->r0, j->cols, j->xf, j->prep, j->y + j->r0);
        else
            ingot_matvec(j->type, (const uint8_t *)j->W + j->r0 * j->rb,
                         j->r1 - j->r0, j->cols, j->xf, j->y + j->r0);
    }
    return NULL;
}

static double gbench(int nt, gjob_t proto, size_t R)
{
    pthread_t th[8];
    gjob_t    jb[8];
    double t0 = mynah_slm_now();
    for (int t = 0; t < nt; t++) {
        jb[t] = proto;
        jb[t].r0 = R * t / nt;
        jb[t].r1 = R * (t + 1) / nt;
        pthread_create(&th[t], NULL, gworker, &jb[t]);
    }
    for (int t = 0; t < nt; t++) pthread_join(th[t], NULL);
    return (mynah_slm_now() - t0) * 1e9 / proto.reps;
}

int gguf_bench(const char *path, int reps, FILE *csv);

int gguf_bench(const char *path, int reps, FILE *csv)
{
    char err[512] = {0};
    ingot_gguf *g = NULL;
    if (ingot_gguf_open(&g, path, err, sizeof err) != 0 || !g) {
        fprintf(stderr, "REFUSED: cannot open %s: %s\n", path, err);
        return 2;
    }
    printf("\n# ===== REAL GGUF: %s =====\n", path);
    printf("# arch=%s  gguf v%u  %zu tensors\n",
           ingot_gguf_arch(g), ingot_gguf_version(g), ingot_gguf_count(g));

    /* A "_K_M" name is a recipe, not a type. Print the census so no row in the
     * table below can be attributed to a type the file does not contain. */
    {
        uint64_t bytes[64] = {0}, elems[64] = {0};
        size_t   cnt[64] = {0};
        for (size_t i = 0; i < ingot_gguf_count(g); i++) {
            const ingot_tensor *t = ingot_gguf_at(g, i);
            if (t->type < 64) { bytes[t->type] += t->nbytes; elems[t->type] += t->nelem; cnt[t->type]++; }
        }
        printf("# type census: ");
        for (int ty = 0; ty < 64; ty++)
            if (cnt[ty])
                printf("%s x%zu (%.1f MiB, %.3f bpw)  ", ingot_type_name(ty), cnt[ty],
                       (double)bytes[ty] / 1048576.0, (double)bytes[ty] * 8.0 / (double)elems[ty]);
        printf("\n\n");
    }

    for (size_t p = 0; p < NPICKS; p++) {
        const ingot_tensor *t = ingot_gguf_find(g, PICKS[p].tensor);
        if (!t) { printf("## %-12s REFUSED: %s absent\n", PICKS[p].label, PICKS[p].tensor); continue; }
        const void *W = ingot_gguf_data(g, t);
        if (!W) { printf("## %-12s REFUSED: no data pointer\n", PICKS[p].label); continue; }

        size_t C = (size_t)t->ne[0];          /* ggml ne[0] is the contraction dim */
        size_t R = (size_t)t->ne[1];
        size_t rb = (size_t)(t->nbytes / R);
        double bpw = (double)t->nbytes * 8.0 / (double)t->nelem;
        printf("## %-12s %-24s [%zu x %zu]  %-6s %.4f bpw  %.1f MiB\n",
               PICKS[p].label, PICKS[p].tensor, R, C, ingot_type_name(t->type), bpw,
               (double)t->nbytes / 1048576.0);

        if (C % TG) { printf("    REFUSED: cols %zu not a multiple of %d\n", C, TG); continue; }

        float *x = malloc(C * sizeof(float));
        float *y = malloc(R * sizeof(float));
        for (size_t i = 0; i < C; i++) x[i] = bench_frand();

        act_t a = {malloc(C), malloc(C / TG * sizeof(float)),
                   malloc(C / TG * sizeof(int32_t)), C, C / TG};
        act_prepare(x, C, &a);
        int8_t *xs2 = malloc(C), *xs4 = malloc(C);
        shuffle_stride(a.q, xs2, C, 64, 4);
        shuffle_stride(a.q, xs4, C, 32, 2);

        /* The real tensor, both activation arms. A GEMV whose output is all
         * zeros has not run; the checksum is printed so that cannot hide. */
        mynah_slm_matvec_in prep;
        int    ours_flag = 0;
        double real_ns[2] = {0, 0};
        for (int i8 = 0; i8 < 2; i8++) {
            mynah_slm_matvec_set_int8(i8);
            if (i8 && !mynah_slm_matvec_int8_enabled()) continue;
            mynah_slm_matvec_prepare(x, C, &prep);
            int ours = (mynah_slm_matvec(t->type, W, R, C, x, &prep, y) == 0);
            if (!ours) ingot_matvec(t->type, W, R, C, x, y);
            ours_flag = ours;
            double chk = 0;
            for (size_t r = 0; r < R; r++) chk += fabs((double)y[r]);
            gjob_t j = {2, reps, W, rb, 0, R, C, &a, xs2, xs4, y, x, t->type, &prep, ours};
            double ns = gbench(1, j, R);
            real_ns[i8] = ns;
            printf("    %-10s [%-13s act=%-4s] %10.0f ns  %6.1f GB/s  sum|y|=%.4g\n",
                   ingot_type_name(t->type),
                   ours ? "mynah" : (ingot_has_kernel(t->type) ? "ingot-simd" : "ingot-generic"),
                   i8 ? "int8" : "f32", ns, (double)t->nbytes / ns, chk / (double)R);
            if (csv) fprintf(csv, "%s,%zu,%zu,%s-real-act%s,%.4f,%.0f,%.2f,%.2f\n",
                             PICKS[p].label, R, C, ingot_type_name(t->type),
                             i8 ? "int8" : "f32", bpw, ns,
                             (double)t->nbytes / 1048576.0, (double)t->nbytes / ns);
        }
        mynah_slm_matvec_set_int8(1);
        mynah_slm_matvec_prepare(x, C, &prep);

        /* The ternary candidates at the SAME shape. Synthetic trits -- this is
         * an execution comparison, not a quality one, and the note says so. */
        double tern_ns[2] = {0, 0};
        size_t trb[2] = {0, 0};
        uint8_t *TW[2] = {NULL, NULL};
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
        {
            int8_t *tv = malloc(R * C);
            float  *ws = malloc(R * (C / TG) * sizeof(float));
            for (size_t i = 0; i < R * C; i++) {
                float v = bench_frand();
                tv[i] = (int8_t)lrintf(v * 4.0f);
            }
            for (size_t i = 0; i < R * (C / TG); i++) ws[i] = 0.01f + 0.001f * bench_frand();
            ternary_fmt fl[2] = {FMT_T3_FOLD9, FMT_T1_2BIT};
            for (int k = 0; k < 2; k++) {
                trb[k] = ternary_row_bytes(fl[k], C);
                TW[k] = malloc(R * trb[k]);
                for (size_t r = 0; r < R; r++) {
                    int8_t tmp[4096];
                    const int8_t *src = tv + r * C;
                    if (fl[k] == FMT_T1_2BIT) {   /* T1 carries trits only */
                        for (size_t i = 0; i < C && i < 4096; i++)
                            tmp[i] = (int8_t)(src[i] > 1 ? 1 : (src[i] < -1 ? -1 : src[i]));
                        src = tmp;
                    }
                    ternary_encode_row(fl[k], src, ws + r * (C / TG), C, TW[k] + r * trb[k]);
                }
                gjob_t j = {k == 0 ? 0 : 1, reps, TW[k], trb[k], 0, R, C, &a, xs2, xs4, y,
                            NULL, 0, NULL, 0};
                tern_ns[k] = gbench(1, j, R);
                printf("    %-10s %6.3f bpw            %10.0f ns  %6.1f GB/s  "
                       "%.2fx the real %s(int8)\n",
                       TERNARY_FORMATS[fl[k]].name, TERNARY_FORMATS[fl[k]].total_bpw,
                       tern_ns[k], (double)(R * trb[k]) / tern_ns[k],
                       real_ns[1] / tern_ns[k], ingot_type_name(t->type));
                if (csv) fprintf(csv, "%s,%zu,%zu,%s-vs-real,%.4f,%.0f,%.2f,%.2f\n",
                                 PICKS[p].label, R, C, TERNARY_FORMATS[fl[k]].name,
                                 TERNARY_FORMATS[fl[k]].total_bpw, tern_ns[k],
                                 (double)(R * trb[k]) / 1048576.0,
                                 (double)(R * trb[k]) / tern_ns[k]);
            }
            free(tv); free(ws);
        }
#endif
        /* Thread scaling, real tensor against T3 fold9. */
        for (int nt = 1; nt <= 4; nt *= 2) {
            gjob_t jr = {2, reps, W, rb, 0, R, C, &a, xs2, xs4, y, x, t->type, &prep, ours_flag};
            double rn = gbench(nt, jr, R);
            double tn = 0;
            if (TW[0]) {
                gjob_t jt = {0, reps, TW[0], trb[0], 0, R, C, &a, xs2, xs4, y, NULL, 0, NULL, 0};
                tn = gbench(nt, jt, R);
            }
            printf("    threads=%d  %-6s %10.0f ns (%5.1f GB/s)   T3 fold9 %10.0f ns "
                   "(%5.1f GB/s)   ternary %.2fx\n", nt, ingot_type_name(t->type), rn,
                   (double)t->nbytes / rn, tn, tn ? (double)(R * trb[0]) / tn : 0,
                   tn ? rn / tn : 0);
            if (csv) {
                fprintf(csv, "%s,%zu,%zu,threads%d-%s-real,%.4f,%.0f,0,%.2f\n",
                        PICKS[p].label, R, C, nt, ingot_type_name(t->type), bpw, rn,
                        (double)t->nbytes / rn);
                if (tn) fprintf(csv, "%s,%zu,%zu,threads%d-fold9,%.4f,%.0f,0,%.2f\n",
                                PICKS[p].label, R, C, nt, 4.125, tn,
                                (double)(R * trb[0]) / tn);
            }
        }
        printf("\n");
        for (int k = 0; k < 2; k++) free(TW[k]);
        free(x); free(y); free(a.q); free(a.scale); free(a.sum); free(xs2); free(xs4);
    }
    mynah_slm_matvec_set_int8(0);
    ingot_gguf_close(g);
    return 0;
}
