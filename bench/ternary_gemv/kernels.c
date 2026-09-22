/* kernels.c — ternary GEMV, scalar reference and Apple-Silicon NEON.
 *
 * Machine this targets, detected not assumed: Apple M1, 4P+4E,
 * FEAT_DotProd = 1 (sdot), FEAT_I8MM = 0 (NO smmla), FEAT_BF16 = 0.
 * fucina/docs/PTQTP.md names i8mm/smmla as the fix for ARM's per-instruction
 * density gap, so this machine is close to the WORST ARM case for the format.
 *
 * Two structural choices, both taken from what already works in src/qmat.c:
 *
 *  1. BIASED CODES. Trits are stored biased into unsigned ranges (t+1, c+4)
 *     so the unpack is shifts and masks with no sign extension. The bias is
 *     removed once per group with the identity
 *         sum(t*x) = sum((t+bias)*x) - bias*sum(x)
 *     which turns a per-element subtract into one multiply-add per group.
 *
 *  2. THE ACTIVATION SUM IS HOISTED ACROSS ROWS. sum(x) over a group does not
 *     depend on the row, so it is computed once for the whole matrix. This is
 *     the same cross-row hoist that makes our Q4_K beat ingot's, and it is
 *     exactly what a per-call kernel API cannot do.
 *
 * SPDX-License-Identifier: MIT */
#include "formats.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

/* ---- activation preparation, shared by every ternary kernel -------------- */

typedef struct {
    int8_t  *q;      /* cols int8, symmetric absmax per group       */
    float   *scale;  /* groups                                      */
    int32_t *sum;    /* groups: sum of q, for the bias correction   */
    size_t   cols, groups;
} act_t;

void act_prepare(const float *x, size_t cols, act_t *a)
{
    a->cols = cols;
    a->groups = cols / TG;
    for (size_t g = 0; g < a->groups; g++) {
        const float *src = x + g * TG;
        float amax = 0.0f;
        for (size_t i = 0; i < TG; i++) {
            float v = fabsf(src[i]);
            if (v > amax) amax = v;
        }
        float s = amax / 127.0f;
        float inv = s != 0.0f ? 1.0f / s : 0.0f;
        int32_t acc = 0;
        for (size_t i = 0; i < TG; i++) {
            int v = (int)lrintf(src[i] * inv);
            if (v > 127) v = 127;
            if (v < -127) v = -127;
            a->q[g * TG + i] = (int8_t)v;
            acc += v;
        }
        a->scale[g] = s;
        a->sum[g] = acc;
    }
}

/* ---- scalar references -------------------------------------------------- */

static inline const float *row_scales(const uint8_t *row, ternary_fmt f, size_t cols)
{
    return (const float *)(row + ternary_row_bytes(f, cols) - (cols / TG) * sizeof(float));
}

void tgemv_scalar(ternary_fmt f, const uint8_t *w, size_t rows, size_t cols,
                  const act_t *a, float *y)
{
    size_t rb = ternary_row_bytes(f, cols);
    for (size_t r = 0; r < rows; r++) {
        const uint8_t *row = w + r * rb;
        const float   *ws = row_scales(row, f, cols);
        float          acc = 0.0f;
        for (size_t g = 0; g < a->groups; g++) {
            const int8_t *xq = a->q + g * TG;
            int32_t       d = 0;
            int           bias = 0;
            switch (f) {
            case FMT_T0_INT8:
                for (size_t i = 0; i < TG; i++)
                    d += (int)(int8_t)row[g * TG + i] * xq[i];
                break;
            case FMT_T1_2BIT:
                bias = 1;
                for (size_t i = 0; i < TG; i++) {
                    unsigned b = row[(g * TG + i) / 4];
                    d += (int)((b >> (2 * (i & 3))) & 3) * xq[i];
                }
                break;
            case FMT_T2_BASE3: {
                bias = 1;
                const uint8_t *gp = row + g * ((TG + 4) / 5);
                for (size_t i = 0; i < TG; i += 5) {
                    unsigned v = gp[i / 5];
                    for (int k = 0; k < 5 && i + k < TG; k++) {
                        d += (int)(v % 3) * xq[i + k];
                        v /= 3;
                    }
                }
                break;
            }
            case FMT_T3_FOLD9:
                bias = 4;
                for (size_t i = 0; i < TG; i++) {
                    unsigned b = row[(g * TG + i) / 2];
                    d += (int)((i & 1) ? (b >> 4) : (b & 15)) * xq[i];
                }
                break;
            case FMT_T3_K3: {
                const uint8_t *lo = row + cols / 4;
                int32_t dh = 0, dl = 0;
                for (size_t i = 0; i < TG; i++) {
                    unsigned bh = row[(g * TG + i) / 4];
                    dh += (int)((bh >> (2 * (i & 3))) & 3) * xq[i];
                    unsigned bl = lo[(g * TG + i) / 2];
                    dl += (int)((i & 1) ? (bl >> 4) : (bl & 15)) * xq[i];
                }
                /* c = 9*(h-1) + (l-4) */
                d = 9 * dh + dl;
                bias = 9 * 1 + 4;
                break;
            }
            case FMT_MASKSIGN: {
                const uint8_t *sgn = row + cols / 8;
                for (size_t i = 0; i < TG; i++) {
                    size_t bit = g * TG + i;
                    if (row[bit / 8] & (1u << (bit & 7)))
                        d += (sgn[bit / 8] & (1u << (bit & 7))) ? -xq[i] : xq[i];
                }
                break;
            }
            default: break;
            }
            acc += ws[g] * a->scale[g] * (float)(d - bias * a->sum[g]);
        }
        y[r] = acc;
    }
}

/* ---- NEON ---------------------------------------------------------------- */
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)

/* 64 weights per iteration from 16 packed bytes: 4 shifts + 4 ands + 4 sdots.
 * fucina reports LLVM emitting a "fully-folded 10-ops-per-64-weights sequence"
 * for its TQ2_0 kernel at ~86% of the NEON ALU roofline; this is the same
 * shape, written out. */
static inline int32_t dot_2bit(const uint8_t *p, const int8_t *x, size_t n)
{
    int32x4_t acc = vdupq_n_s32(0);
    const uint8x16_t m3 = vdupq_n_u8(3);
    for (size_t i = 0; i < n; i += 64, p += 16, x += 64) {
        uint8x16_t b = vld1q_u8(p);
        int8x16_t c0 = vreinterpretq_s8_u8(vandq_u8(b, m3));
        int8x16_t c1 = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(b, 2), m3));
        int8x16_t c2 = vreinterpretq_s8_u8(vandq_u8(vshrq_n_u8(b, 4), m3));
        int8x16_t c3 = vreinterpretq_s8_u8(vshrq_n_u8(b, 6));
        /* codes are interleaved 4-per-byte, so the activation lanes must
         * match: x is pre-shuffled by the caller into the same order. */
        acc = vdotq_s32(acc, c0, vld1q_s8(x));
        acc = vdotq_s32(acc, c1, vld1q_s8(x + 16));
        acc = vdotq_s32(acc, c2, vld1q_s8(x + 32));
        acc = vdotq_s32(acc, c3, vld1q_s8(x + 48));
    }
    return vaddvq_s32(acc);
}

/* 32 weights per iteration from 16 packed bytes: 1 shift + 2 ands + 2 sdots. */
static inline int32_t dot_nibble(const uint8_t *p, const int8_t *x, size_t n)
{
    int32x4_t acc = vdupq_n_s32(0);
    const uint8x16_t m15 = vdupq_n_u8(15);
    for (size_t i = 0; i < n; i += 32, p += 16, x += 32) {
        uint8x16_t b = vld1q_u8(p);
        acc = vdotq_s32(acc, vreinterpretq_s8_u8(vandq_u8(b, m15)), vld1q_s8(x));
        acc = vdotq_s32(acc, vreinterpretq_s8_u8(vshrq_n_u8(b, 4)), vld1q_s8(x + 16));
    }
    return vaddvq_s32(acc);
}

static inline int32_t dot_int8(const int8_t *p, const int8_t *x, size_t n)
{
    int32x4_t acc = vdupq_n_s32(0);
    for (size_t i = 0; i < n; i += 16, p += 16, x += 16)
        acc = vdotq_s32(acc, vld1q_s8(p), vld1q_s8(x));
    return vaddvq_s32(acc);
}

void tgemv_neon(ternary_fmt f, const uint8_t *w, size_t rows, size_t cols,
                const act_t *a, const int8_t *xs2, const int8_t *xs4, float *y)
{
    size_t rb = ternary_row_bytes(f, cols);
    for (size_t r = 0; r < rows; r++) {
        const uint8_t *row = w + r * rb;
        const float   *ws = row_scales(row, f, cols);
        float          acc = 0.0f;
        for (size_t g = 0; g < a->groups; g++) {
            int32_t d;
            int     bias;
            switch (f) {
            case FMT_T0_INT8:
                d = dot_int8((const int8_t *)row + g * TG, a->q + g * TG, TG);
                bias = 0;
                break;
            case FMT_T1_2BIT:
                d = dot_2bit(row + g * TG / 4, xs2 + g * TG, TG);
                bias = 1;
                break;
            case FMT_T3_FOLD9:
                d = dot_nibble(row + g * TG / 2, xs4 + g * TG, TG);
                bias = 4;
                break;
            case FMT_T3_K3:
                d = 9 * dot_2bit(row + g * TG / 4, xs2 + g * TG, TG) +
                    dot_nibble(row + cols / 4 + g * TG / 2, xs4 + g * TG, TG);
                bias = 13;
                break;
            default:
                d = 0;
                bias = 0;
                break;
            }
            acc += ws[g] * a->scale[g] * (float)(d - bias * a->sum[g]);
        }
        y[r] = acc;
    }
}
#endif
