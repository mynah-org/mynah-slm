/* kernels_i8mm.c -- the ARMv8.6 smmla arm of T3 fold9, for Neoverse V2.
 *
 * WRITTEN BLIND. This M1 reports FEAT_I8MM = 0, so this translation unit is
 * compiled and its op counts are derivable, but IT HAS NEVER EXECUTED. Nothing
 * in this file may be quoted as a measurement until Axion runs it.
 *
 * ---------------------------------------------------------------------------
 * Why a naive smmla GEMV is a LOSS, and what fixes it
 *
 * `smmla` multiplies a 2x8 int8 matrix by a 2x8 int8 matrix transposed into a
 * 2x2 int32 accumulator: 32 MACs per instruction against `sdot`'s 16. That 2x
 * is real only when there are TWO activation columns. A decode GEMV has one,
 * so the second output column duplicates the first and half the lanes are
 * wasted -- 16 useful MACs per instruction, exactly `sdot`'s rate.
 *
 * Worse, on the layout the sdot kernel uses, getting 8 k-contiguous weights of
 * row r into lanes 0-7 and 8 of row r+1 into lanes 8-15 costs FOUR vcombines
 * per 64 weights on top of the unpack:
 *
 *     sdot   1 load + 1 and + 1 shr + 2 sdot   = 5 ops /  32 weights (1 row)
 *     smmla  2 load + 2 and + 2 shr + 4 vcombine + 4 smmla
 *                                              = 14 ops / 64 weights (2 rows)
 *
 * i.e. 10 ops vs 14 ops for the same work: [DERIVED] naive smmla is ~1.4x
 * WORSE, not better.
 *
 * The fix is not in the kernel, it is in the layout. Store the nibbles of a
 * ROW PAIR interleaved so that one 16-byte load already carries row r in the
 * low half and row r+1 in the high half:
 *
 *     byte i     (i < 8) = row_r  k[i] | row_r  k[i+8] << 4
 *     byte 8 + i (i < 8) = row_r1 k[i] | row_r1 k[i+8] << 4
 *
 * then `and 0xF` yields [row_r k0..7 | row_r1 k0..7] -- an smmla A operand with
 * no shuffling at all, and `shr 4` yields the k+8 block:
 *
 *     smmla  1 load + 1 and + 1 shr + 2 smmla  = 5 ops / 32 weights (2 rows)
 *
 * [DERIVED] 2x fewer ops per weight than sdot. THE SAME CODES, THE SAME 4.125
 * bpw, THE SAME VALUES -- only a within-group permutation and a row-pair
 * blocking. It is not a new representation; `ternary_repack_pairs` below is a
 * pure permutation and the round-trip test proves it.
 *
 * The B operands are the activations duplicated into both halves. They do not
 * depend on the row, so they are built ONCE per GEMV -- the same cross-row
 * hoist that the sdot arm already uses for sum(x).
 *
 * SPDX-License-Identifier: MIT */
#include "dispatch.h"

#include <stdlib.h>
#include <string.h>

/* Permute a T3-fold9 matrix into the row-pair-interleaved layout. Codes are
 * untouched; only their position changes. `rows` must be even. */
void ternary_repack_pairs(const uint8_t *src, uint8_t *dst, size_t rows, size_t cols);
void ternary_repack_pairs(const uint8_t *src, uint8_t *dst, size_t rows, size_t cols)
{
    size_t rb = ternary_row_bytes(FMT_T3_FOLD9, cols);
    size_t payload = cols / 2;                 /* 2 codes per byte */
    size_t nsc = cols / TG;
    for (size_t r = 0; r + 1 < rows; r += 2) {
        const uint8_t *a = src + r * rb, *b = src + (r + 1) * rb;
        uint8_t       *o = dst + r * rb;
        /* 16 output bytes carry 16 codes of row a and 16 of row b */
        for (size_t blk = 0; blk < payload; blk += 8) {
            for (int i = 0; i < 8; i++) {
                /* source nibble j of a row lives at byte j/2, half j&1 */
                size_t k0 = blk * 2 + (size_t)i, k1 = k0 + 8;
                uint8_t a0 = (a[k0 / 2] >> ((k0 & 1) * 4)) & 15;
                uint8_t a1 = (a[k1 / 2] >> ((k1 & 1) * 4)) & 15;
                uint8_t b0 = (b[k0 / 2] >> ((k0 & 1) * 4)) & 15;
                uint8_t b1 = (b[k1 / 2] >> ((k1 & 1) * 4)) & 15;
                o[blk * 2 + (size_t)i]     = (uint8_t)(a0 | (a1 << 4));
                o[blk * 2 + 8 + (size_t)i] = (uint8_t)(b0 | (b1 << 4));
            }
        }
        /* A pair block is [2*payload codes][row r scales][row r+1 scales] and
         * is exactly 2*rb bytes, so the pair layout is the same total size as
         * the row layout -- 4.125 bpw either way. */
        memcpy(o + 2 * payload, a + payload, nsc * sizeof(float));
        memcpy(o + 2 * payload + nsc * sizeof(float), b + payload, nsc * sizeof(float));
    }
}

/* Activation B operands: x[k..k+7] duplicated into both halves of a 16-byte
 * register, in the order the pair layout reads them. Built once per GEMV. */
void ternary_pair_acts(const int8_t *xq, int8_t *out, size_t cols);
void ternary_pair_acts(const int8_t *xq, int8_t *out, size_t cols)
{
    size_t o = 0;
    for (size_t blk = 0; blk < cols; blk += 16) {
        for (int h = 0; h < 2; h++) {           /* h=0: k..k+7, h=1: k+8..k+15 */
            for (int d = 0; d < 2; d++)         /* duplicated into both halves */
                for (int i = 0; i < 8; i++)
                    out[o + (size_t)d * 8 + (size_t)i] = xq[blk + (size_t)h * 8 + (size_t)i];
            o += 16;
        }
    }
}

#if defined(TERNARY_BUILD_I8MM)
#include <arm_neon.h>

/* 32 weights of a ROW PAIR from 16 packed bytes: 1 load + 1 and + 1 shr +
 * 2 smmla. acc lanes: [r.x, r.x, s.x, s.x] -- lanes 1 and 3 are the duplicated
 * activation column and are discarded. */
static inline void dot_pair_nibble(const uint8_t *p, const int8_t *xd, size_t n,
                                   int32_t *dr, int32_t *ds)
{
    int32x4_t acc = vdupq_n_s32(0);
    const uint8x16_t m15 = vdupq_n_u8(15);
    for (size_t i = 0; i < n; i += 16, p += 16, xd += 32) {
        uint8x16_t b = vld1q_u8(p);
        acc = vmmlaq_s32(acc, vreinterpretq_s8_u8(vandq_u8(b, m15)), vld1q_s8(xd));
        acc = vmmlaq_s32(acc, vreinterpretq_s8_u8(vshrq_n_u8(b, 4)), vld1q_s8(xd + 16));
    }
    *dr += vgetq_lane_s32(acc, 0);
    *ds += vgetq_lane_s32(acc, 2);
}

int tgemv_i8mm_fold9(const uint8_t *w, size_t rows, size_t cols, const act_t *a,
                     const int8_t *xpair, float *y);
int tgemv_i8mm_fold9(const uint8_t *w, size_t rows, size_t cols, const act_t *a,
                     const int8_t *xpair, float *y)
{
    if (rows % 2) return -1;                    /* pair layout needs even rows */
    size_t rb = ternary_row_bytes(FMT_T3_FOLD9, cols);
    size_t payload = cols / 2, nsc = cols / TG;
    for (size_t r = 0; r + 1 < rows; r += 2) {
        const uint8_t *pr = w + r * rb;
        const float   *wa = (const float *)(pr + 2 * payload);
        const float   *wb = wa + nsc;
        float          accr = 0.0f, accs = 0.0f;
        for (size_t g = 0; g < nsc; g++) {
            int32_t dr = 0, ds = 0;
            dot_pair_nibble(pr + g * TG, xpair + g * TG * 2, TG, &dr, &ds);
            float s = a->scale[g];
            accr += wa[g] * s * (float)(dr - 4 * a->sum[g]);
            accs += wb[g] * s * (float)(ds - 4 * a->sum[g]);
        }
        y[r]     = accr;
        y[r + 1] = accs;
    }
    return 0;
}
#endif /* TERNARY_BUILD_I8MM */

/* Portable reference for the SAME pair layout, so the layout can be validated
 * against the T0 oracle on a machine that cannot execute smmla. This is how
 * the i8mm arm gets a correctness gate before Axion ever sees it. */
int tgemv_pair_ref(const uint8_t *w, size_t rows, size_t cols, const act_t *a,
                   const int8_t *xpair, float *y);
int tgemv_pair_ref(const uint8_t *w, size_t rows, size_t cols, const act_t *a,
                   const int8_t *xpair, float *y)
{
    if (rows % 2) return -1;
    size_t rb = ternary_row_bytes(FMT_T3_FOLD9, cols);
    size_t payload = cols / 2, nsc = cols / TG;
    for (size_t r = 0; r + 1 < rows; r += 2) {
        const uint8_t *pr = w + r * rb;
        const float   *wa = (const float *)(pr + 2 * payload);
        const float   *wb = wa + nsc;
        float          accr = 0.0f, accs = 0.0f;
        for (size_t g = 0; g < nsc; g++) {
            int32_t dr = 0, ds = 0;
            const uint8_t *p = pr + g * TG;
            const int8_t  *xd = xpair + g * TG * 2;
            for (size_t i = 0; i < TG; i += 16, p += 16, xd += 32) {
                for (int lane = 0; lane < 16; lane++) {
                    int lo = p[lane] & 15, hi = p[lane] >> 4;
                    int32_t *t = lane < 8 ? &dr : &ds;
                    *t += lo * xd[lane] + hi * xd[16 + lane];
                }
            }
            float s = a->scale[g];
            accr += wa[g] * s * (float)(dr - 4 * a->sum[g]);
            accs += wb[g] * s * (float)(ds - 4 * a->sum[g]);
        }
        y[r]     = accr;
        y[r + 1] = accs;
    }
    return 0;
}
