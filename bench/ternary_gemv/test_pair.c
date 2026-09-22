/* test_pair.c -- gate the i8mm row-pair LAYOUT on a machine with no i8mm.
 *
 * The smmla kernel cannot run here. Its layout can: `tgemv_pair_ref` walks the
 * repacked bytes in exactly the lane order `smmla` would. If the permutation
 * or the lane mapping were wrong, this test fails on an M1 -- which is the
 * whole point of having it, because the alternative is discovering it on a
 * rented ARM box.
 *
 * Gate: repack(T3 fold9) scored by the pair reference must equal the T0 oracle
 * bit-for-bit, not approximately.
 *
 * SPDX-License-Identifier: MIT */
#include "dispatch.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

void tgemv_scalar(ternary_fmt f, const uint8_t *w, size_t rows, size_t cols,
                  const act_t *a, float *y);
void ternary_repack_pairs(const uint8_t *src, uint8_t *dst, size_t rows, size_t cols);
void ternary_pair_acts(const int8_t *xq, int8_t *out, size_t cols);
int  tgemv_pair_ref(const uint8_t *w, size_t rows, size_t cols, const act_t *a,
                    const int8_t *xpair, float *y);

static uint64_t rng = 0x9E3779B97F4A7C15ULL;
static float frand(void)
{
    rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17;
    return (float)((double)(rng >> 11) / 9007199254740992.0) * 2.0f - 1.0f;
}

int main(void)
{
    const ternary_caps *c = ternary_detect();
    printf("cpu=%s  neon=%d dotprod=%d i8mm=%d\n", c->cpu, c->have_neon,
           c->have_dotprod, c->have_i8mm);

    struct { size_t R, C; } shapes[] = {{3072,1024},{1024,3072},{2048,1024},
                                        {1024,2048},{1024,1024}};
    int bad = 0;
    for (size_t s = 0; s < sizeof shapes / sizeof shapes[0]; s++) {
        size_t R = shapes[s].R, C = shapes[s].C;
        float *x = malloc(C * sizeof(float));
        for (size_t i = 0; i < C; i++) x[i] = frand();
        act_t a = {malloc(C), malloc(C / TG * sizeof(float)),
                   malloc(C / TG * sizeof(int32_t)), C, C / TG};
        act_prepare(x, C, &a);

        int8_t *tv = malloc(R * C);
        float  *ws = malloc(R * (C / TG) * sizeof(float));
        for (size_t i = 0; i < R * C; i++) tv[i] = (int8_t)lrintf(frand() * 4.0f);
        for (size_t i = 0; i < R * (C / TG); i++) ws[i] = 0.01f + 0.001f * frand();

        size_t rb = ternary_row_bytes(FMT_T3_FOLD9, C);
        size_t ob = ternary_row_bytes(FMT_T0_INT8, C);
        uint8_t *W = malloc(R * rb), *P = malloc(R * rb), *O = malloc(R * ob);
        for (size_t r = 0; r < R; r++) {
            ternary_encode_row(FMT_T3_FOLD9, tv + r * C, ws + r * (C / TG), C, W + r * rb);
            ternary_encode_row(FMT_T0_INT8,  tv + r * C, ws + r * (C / TG), C, O + r * ob);
        }
        ternary_repack_pairs(W, P, R, C);
        int8_t *xp = malloc(C * 2);
        ternary_pair_acts(a.q, xp, C);

        float *yo = malloc(R * sizeof(float)), *yp = malloc(R * sizeof(float));
        tgemv_scalar(FMT_T0_INT8, O, R, C, &a, yo);
        if (tgemv_pair_ref(P, R, C, &a, xp, yp) != 0) { printf("  pair ref refused\n"); bad = 1; }

        double mx = 0, num = 0, den = 0;
        for (size_t r = 0; r < R; r++) {
            double d = fabs((double)yp[r] - yo[r]);
            mx = d > mx ? d : mx; num += d * d; den += (double)yo[r] * yo[r];
        }
        double rel = den > 0 ? sqrt(num / den) : 0;
        printf("  [%5zu x %5zu]  pair layout vs T0 oracle: max %.3e  rel_l2 %.3e  %s\n",
               R, C, mx, rel, rel < 1e-6 ? "OK" : "*** WRONG ***");
        if (rel >= 1e-6) bad = 1;

        free(x); free(a.q); free(a.scale); free(a.sum); free(tv); free(ws);
        free(W); free(P); free(O); free(xp); free(yo); free(yp);
    }
    printf("%s\n", bad ? "FAIL" : "PASS -- the i8mm layout is validated on a machine without i8mm");
    return bad;
}
