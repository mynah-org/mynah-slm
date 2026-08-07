/* qmat.c — see qmat.h.
 * SPDX-License-Identifier: MIT */
#include "qmat.h"

#include "threads.h"

#include "ingot/dtype.h"
#include "ingot/quant.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(MYNAH_SLM_BLAS_ACCELERATE)
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif

/* ── our Q4_K matvec ────────────────────────────────────────────────────────
 * Q4_K stores 256 weights in 144 bytes: two f16 (d, dmin), eight 6-bit
 * scale/min pairs packed into 12 bytes, and 128 bytes of nibbles. A weight is
 *
 *     w = d * scale[s] * q  -  dmin * min[s]
 *
 * and the obvious kernel — the one ingot ships, and the one we shipped before
 * this — materializes that value per element: unpack the nibble, widen it,
 * multiply by d*scale, subtract dmin*min, then FMA against the input. Three
 * vector ops per four weights before the multiply-add that actually matters.
 *
 * Distribute the sum instead:
 *
 *     SUM_j w_j x_j  =  d*scale * SUM_j (q_j x_j)  -  dmin*min * SUM_j x_j
 *
 * Two consequences, and the second is the one worth having:
 *
 *   - the per-element multiply and subtract disappear. The inner loop is
 *     unpack + widen + FMA, and the scales are applied once per 32-weight
 *     sub-block instead of 32 times.
 *   - SUM_j x_j does not depend on the row. It is a property of the INPUT, so
 *     it is computed once per matvec and reused across every row — 2048 of
 *     them for attn_q. That hoist is why this lives in the engine and not in
 *     a container library: it needs a per-call preamble the row kernel then
 *     reads, which is a different API shape, not a faster loop.
 *
 * Measured against ingot's kernel, same tensors, `make bench`: see docs/perf.md.
 *
 * The arithmetic is the same, the rounding is not — fewer roundings, in fact,
 * since the min term is now one subtraction per sub-block rather than one per
 * weight. The parity gate judges that, at 1e-4 on layer 0. */

/* The stored scales are IEEE half. Read as bytes, not through a native _Float16
 * load: the block is not guaranteed to be 2-byte aligned inside the mapping,
 * and on a strict-alignment target that is a fault rather than a slow path. */
static float mynah_slm_f16_to_f32(const unsigned char *p) {
    const uint16_t h = (uint16_t)(p[0] | ((uint16_t)p[1] << 8));
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    const uint32_t exp  = (h >> 10) & 0x1fu;
    const uint32_t mant = h & 0x3ffu;
    uint32_t bits;

    if (exp == 0) {
        if (mant == 0) { bits = sign; }
        else {
            /* Subnormal half: renormalize into a normal float. */
            uint32_t e = 0, m = mant;
            while ((m & 0x400u) == 0) { m <<= 1; e++; }
            m &= 0x3ffu;
            bits = sign | ((127u - 15u - e + 1u) << 23) | (m << 13);
        }
    } else if (exp == 31u) {
        bits = sign | 0x7f800000u | (mant << 13);
    } else {
        bits = sign | ((exp + 127u - 15u) << 23) | (mant << 13);
    }

    float out;
    memcpy(&out, &bits, sizeof out);
    return out;
}

static int g_own = -1;      /* -1 = not resolved yet */

static int use_own_kernels(void) {
    /* Read once. getenv in a matvec called ~200 times per token would be its
     * own measurement problem. */
    if (g_own < 0) {
        const char *e = getenv("MYNAH_SLM_KERNELS");
        g_own = (e && strcmp(e, "ingot") == 0) ? 0 : 1;
    }
    return g_own;
}

void mynah_slm_matvec_set_enabled(int on) { g_own = on ? 1 : 0; }

/* Q6_K is OURS on x86 and INGOT'S on ARM, and that split is a measurement
 * rather than a preference.
 *
 * ingot ships a fused NEON Q6_K matvec and no AVX2 one at all — on x86 the
 * type falls through to the generic path, which dequantizes each block into a
 * 256-float scratch array and then dots it. So the same kernel measures two
 * completely different things depending on what it is standing next to:
 *
 *   ARM   0.99x on the LM head, 0.87-1.07x elsewhere — a TIE, and this file's
 *         rule is that a tie does not get to replace a validated kernel.
 *   x86   4.65x on the LM head, 4.0-4.2x elsewhere. Not a better vector
 *         kernel: a vector kernel where there was none.
 *
 * MYNAH_SLM_Q6K=own or =ingot forces either side, which is how the A/B stays
 * runnable on a machine whose default is the other one. */
static int g_own_q6k = -1;

static int own_q6_k(void) {
    if (g_own_q6k < 0) {
        const char *e = getenv("MYNAH_SLM_Q6K");
        if (e) g_own_q6k = strcmp(e, "own") == 0;
#if defined(__ARM_NEON)
        else g_own_q6k = 0;
#else
        else g_own_q6k = 1;
#endif
    }
    return g_own_q6k;
}

void mynah_slm_matvec_set_q6k(int own) { g_own_q6k = own ? 1 : 0; }

/* Every kernel below exists in three forms: NEON, AVX2, and a scalar
 * reference. The scalar one is not a fallback nobody runs — it is the
 * definition the other two have to agree with, and it is what a machine
 * without either instruction set actually gets.
 *
 * x86 cannot be executed natively here (this is an M1), so it is verified two
 * ways: `make check-x86` cross-compiles it, and `make test-x86-rosetta` builds
 * the suite as x86_64 and RUNS it under Rosetta, which translates AVX2. */
#if defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)
#define MYNAH_SLM_HAVE_SDOT 1
#elif defined(__AVX2__)
#define MYNAH_SLM_HAVE_SDOT 1     /* maddubs + madd, no VNNI required */
#endif

static int g_int8 = -1;

int mynah_slm_matvec_int8_enabled(void) {
#if defined(MYNAH_SLM_HAVE_SDOT)
    if (g_int8 < 0) {
        /* Off unless asked for. It trades accuracy for speed, and a default
         * that quietly does that is how a quantization claim stops meaning
         * anything. `mynah-slm ppl` is what decides, not this. */
        const char *e = getenv("MYNAH_SLM_INT8");
        g_int8 = (e && strcmp(e, "0") != 0) ? 1 : 0;
    }
    return g_int8;
#else
    return 0;
#endif
}

void mynah_slm_matvec_set_int8(int on) { g_int8 = on ? 1 : 0; }

int mynah_slm_matvec_have(int type) {
    if (!use_own_kernels()) return 0;
    if (type == INGOT_TYPE_Q4_K) return 1;
    if (type == INGOT_TYPE_Q6_K) return own_q6_k();
    return 0;
}

void mynah_slm_matvec_prepare(const float *input, size_t cols,
                              mynah_slm_matvec_in *prep) {
    if (!input || !prep) return;
    prep->have_int8 = mynah_slm_matvec_int8_enabled() && cols <= MYNAH_SLM_XQ_MAX;

    for (size_t s = 0; s < cols / 32; s++) {
        const float *x = input + s * 32;
        float acc = 0.0f, amax = 0.0f;
        for (int i = 0; i < 32; i++) {
            acc += x[i];
            const float a = fabsf(x[i]);
            if (a > amax) amax = a;
        }
        prep->xsum[s] = acc;

        if (prep->have_int8) {
            const float scale = amax / 127.0f;
            prep->xscale[s] = scale;
            const float inv = scale > 0.0f ? 1.0f / scale : 0.0f;
            int8_t *q = prep->xq + s * 32;
            for (int i = 0; i < 32; i++) {
                float v = nearbyintf(x[i] * inv);
                if (v >  127.0f) v =  127.0f;
                if (v < -128.0f) v = -128.0f;
                q[i] = (int8_t)v;
            }
        }
    }
}

/* The 6-bit scale/min pair for sub-block `index`, unpacked from the 12 bytes.
 * Byte-for-byte ggml's layout — this is a format detail, so it is copied and
 * not improvised. */
static void q4_k_scale_min(const unsigned char *scales, int index,
                           unsigned char *scale, unsigned char *minimum) {
    if (index < 4) {
        *scale   = scales[index] & 63u;
        *minimum = scales[index + 4] & 63u;
    } else {
        *scale   = (unsigned char)((scales[index + 4] & 0x0fu) |
                                   ((scales[index - 4] >> 6) << 4));
        *minimum = (unsigned char)((scales[index + 4] >> 4) |
                                   ((scales[index] >> 6) << 4));
    }
}

#if defined(MYNAH_SLM_HAVE_SDOT)
#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

/* ── the int8 path ─────────────────────────────────────────────────────────
 * SDOT does four int8 multiply-accumulates per lane in one instruction, where
 * the f32 form needs a widen, a convert and an FMA per four values. The price
 * is that the ACTIVATIONS are quantized to int8 per 32 values.
 *
 * Both halves of the identity survive it cleanly:
 *
 *     SUM_j w_j x_j = d*scale * xs * SUM_j (q_j * xq_j)  -  dmin*min * SUM_j x_j
 *                                    ^^^^^^^^^^^^^^^^^        ^^^^^^^^^^^^^^^^
 *                                    integer, one SDOT        still exact f32
 *
 * so the min term keeps full precision and only the product term is
 * approximated. Whether that is acceptable is a question for `mynah-slm ppl`,
 * which is why this is off by default. */
#if defined(__AVX2__)
#include <immintrin.h>

/* No VNNI required: maddubs multiplies u8 by i8 into i16 pairs and madd folds
 * those into i32. The nibbles are 0..15 and the activations fit int8, so the
 * largest partial is 15*127*2 = 3810 — far inside i16, and maddubs saturates
 * rather than wraps, which would otherwise be the trap here. */
static inline int hsum256i(__m256i v) {
    __m128i a = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    a = _mm_add_epi32(a, _mm_shuffle_epi32(a, 0x4e));
    a = _mm_add_epi32(a, _mm_shuffle_epi32(a, 0xb1));
    return _mm_cvtsi128_si32(a);
}
#endif

static float q4_k_row_int8(const unsigned char *row, size_t blocks,
                           const int8_t *xq, const float *xscale,
                           const float *xsum) {
    float total = 0.0f, mins = 0.0f;

    for (size_t b = 0; b < blocks; b++) {
        const unsigned char *block = row + b * 144;
        const float d    = mynah_slm_f16_to_f32(block);
        const float dmin = mynah_slm_f16_to_f32(block + 2);
        const unsigned char *scales = block + 4;
        const unsigned char *q      = block + 16;
        const size_t sub0 = b * 8;

        for (int base = 0, si = 0; base < 256; base += 64, si += 2) {
            unsigned char sc0, mn0, sc1, mn1;
            q4_k_scale_min(scales, si,     &sc0, &mn0);
            q4_k_scale_min(scales, si + 1, &sc1, &mn1);

            const int8_t *xlo = xq + b * 256 + base;
            const int8_t *xhi = xlo + 32;
            int sum_lo, sum_hi;
#if defined(__ARM_NEON)
            int32x4_t alo = vdupq_n_s32(0), ahi = vdupq_n_s32(0);
            for (int i = 0; i < 32; i += 16) {
                const uint8x16_t p = vld1q_u8(q + i);
                alo = vdotq_s32(alo, vreinterpretq_s8_u8(vandq_u8(p, vdupq_n_u8(0x0f))),
                                vld1q_s8(xlo + i));
                ahi = vdotq_s32(ahi, vreinterpretq_s8_u8(vshrq_n_u8(p, 4)),
                                vld1q_s8(xhi + i));
            }
            sum_lo = vaddvq_s32(alo);
            sum_hi = vaddvq_s32(ahi);
#else
            const __m256i ones = _mm256_set1_epi16(1);
            const __m256i p = _mm256_loadu_si256((const __m256i *)(const void *)q);
            const __m256i nl = _mm256_and_si256(p, _mm256_set1_epi8(0x0f));
            const __m256i nh = _mm256_and_si256(_mm256_srli_epi16(p, 4),
                                                _mm256_set1_epi8(0x0f));
            const __m256i vl = _mm256_loadu_si256((const __m256i *)(const void *)xlo);
            const __m256i vh = _mm256_loadu_si256((const __m256i *)(const void *)xhi);
            sum_lo = hsum256i(_mm256_madd_epi16(_mm256_maddubs_epi16(nl, vl), ones));
            sum_hi = hsum256i(_mm256_madd_epi16(_mm256_maddubs_epi16(nh, vh), ones));
#endif

            const size_t s0 = sub0 + (size_t)base / 32;
            total += d * (float)sc0 * xscale[s0]     * (float)sum_lo;
            total += d * (float)sc1 * xscale[s0 + 1] * (float)sum_hi;
            mins  += dmin * ((float)mn0 * xsum[s0] + (float)mn1 * xsum[s0 + 1]);
            q += 32;
        }
    }
    return total - mins;
}
#endif


#if defined(__ARM_NEON)
static inline float32x4_t nib_to_f32(uint8x8_t v, int high) {
    const uint16x8_t w = vmovl_u8(v);
    return vcvtq_f32_u32(vmovl_u16(high ? vget_high_u16(w) : vget_low_u16(w)));
}

static float q4_k_row(const unsigned char *row, size_t blocks,
                      const float *x, const float *xsum) {
    float32x4_t total = vdupq_n_f32(0.0f);
    float       mins  = 0.0f;

    for (size_t b = 0; b < blocks; b++) {
        const unsigned char *block = row + b * 144;
        const float d    = mynah_slm_f16_to_f32(block);
        const float dmin = mynah_slm_f16_to_f32(block + 2);
        const unsigned char *scales = block + 4;
        const unsigned char *q      = block + 16;
        const float         *bx     = x    + b * 256;
        const float         *bs     = xsum + b * 8;

        for (int base = 0, si = 0; base < 256; base += 64, si += 2) {
            unsigned char sc0, mn0, sc1, mn1;
            q4_k_scale_min(scales, si,     &sc0, &mn0);
            q4_k_scale_min(scales, si + 1, &sc1, &mn1);

            float32x4_t lo = vdupq_n_f32(0.0f), hi = vdupq_n_f32(0.0f);
            for (int i = 0; i < 32; i += 8) {
                const uint8x8_t packed = vld1_u8(q + i);
                const uint8x8_t nl = vand_u8(packed, vdup_n_u8(0x0f));
                const uint8x8_t nh = vshr_n_u8(packed, 4);
                lo = vmlaq_f32(lo, nib_to_f32(nl, 0), vld1q_f32(bx + base + i));
                lo = vmlaq_f32(lo, nib_to_f32(nl, 1), vld1q_f32(bx + base + i + 4));
                hi = vmlaq_f32(hi, nib_to_f32(nh, 0), vld1q_f32(bx + base + i + 32));
                hi = vmlaq_f32(hi, nib_to_f32(nh, 1), vld1q_f32(bx + base + i + 36));
            }
            /* One scale application per 32 weights instead of 32. */
            total = vmlaq_n_f32(total, lo, d * (float)sc0);
            total = vmlaq_n_f32(total, hi, d * (float)sc1);
            mins += dmin * ((float)mn0 * bs[base / 32] +
                            (float)mn1 * bs[base / 32 + 1]);
            q += 32;
        }
    }
    return vaddvq_f32(total) - mins;
}

#elif defined(__AVX2__)
#include <immintrin.h>

static inline float hsum256(__m256 v) {
    __m128 a = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    a = _mm_add_ps(a, _mm_movehl_ps(a, a));
    a = _mm_add_ss(a, _mm_shuffle_ps(a, a, 0x55));
    return _mm_cvtss_f32(a);
}

static float q4_k_row(const unsigned char *row, size_t blocks,
                      const float *x, const float *xsum) {
    __m256 total = _mm256_setzero_ps();
    float  mins  = 0.0f;

    for (size_t b = 0; b < blocks; b++) {
        const unsigned char *block = row + b * 144;
        const float d    = mynah_slm_f16_to_f32(block);
        const float dmin = mynah_slm_f16_to_f32(block + 2);
        const unsigned char *scales = block + 4;
        const unsigned char *q      = block + 16;
        const float         *bx     = x    + b * 256;
        const float         *bs     = xsum + b * 8;

        for (int base = 0, si = 0; base < 256; base += 64, si += 2) {
            unsigned char sc0, mn0, sc1, mn1;
            q4_k_scale_min(scales, si,     &sc0, &mn0);
            q4_k_scale_min(scales, si + 1, &sc1, &mn1);

            __m256 lo = _mm256_setzero_ps(), hi = _mm256_setzero_ps();
            for (int i = 0; i < 32; i += 8) {
                const __m128i packed = _mm_loadl_epi64((const __m128i *)(const void *)(q + i));
                const __m128i nl = _mm_and_si128(packed, _mm_set1_epi8(0x0f));
                const __m128i nh = _mm_and_si128(_mm_srli_epi16(packed, 4),
                                                 _mm_set1_epi8(0x0f));
                lo = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(nl)),
                                     _mm256_loadu_ps(bx + base + i), lo);
                hi = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(nh)),
                                     _mm256_loadu_ps(bx + base + i + 32), hi);
            }
            total = _mm256_fmadd_ps(lo, _mm256_set1_ps(d * (float)sc0), total);
            total = _mm256_fmadd_ps(hi, _mm256_set1_ps(d * (float)sc1), total);
            mins += dmin * ((float)mn0 * bs[base / 32] +
                            (float)mn1 * bs[base / 32 + 1]);
            q += 32;
        }
    }
    return hsum256(total) - mins;
}

#else   /* the scalar twin, and the reference for what the other two mean */

static float q4_k_row(const unsigned char *row, size_t blocks,
                      const float *x, const float *xsum) {
    float total = 0.0f, mins = 0.0f;

    for (size_t b = 0; b < blocks; b++) {
        const unsigned char *block = row + b * 144;
        const float d    = mynah_slm_f16_to_f32(block);
        const float dmin = mynah_slm_f16_to_f32(block + 2);
        const unsigned char *scales = block + 4;
        const unsigned char *q      = block + 16;
        const float         *bx     = x    + b * 256;
        const float         *bs     = xsum + b * 8;

        for (int base = 0, si = 0; base < 256; base += 64, si += 2) {
            unsigned char sc0, mn0, sc1, mn1;
            q4_k_scale_min(scales, si,     &sc0, &mn0);
            q4_k_scale_min(scales, si + 1, &sc1, &mn1);

            float lo = 0.0f, hi = 0.0f;
            for (int i = 0; i < 32; i++) {
                lo += (float)(q[i] & 0x0f) * bx[base + i];
                hi += (float)(q[i] >> 4)   * bx[base + i + 32];
            }
            total += d * ((float)sc0 * lo + (float)sc1 * hi);
            mins  += dmin * ((float)mn0 * bs[base / 32] +
                             (float)mn1 * bs[base / 32 + 1]);
            q += 32;
        }
    }
    return total - mins;
}

#endif

/* ── Q6_K ──────────────────────────────────────────────────────────────────
 * 210 bytes per 256 weights:  ql[128] | qh[64] | int8 sc[16] | f16 d
 * and `d` sits at the END of the block, at +208, not the start.
 *
 *     w_j = d * sc[j/16] * (q_j - 32),   q_j = 6 bits, in 0..63
 *
 * A group of 16 weights shares one scale and maps to 16 CONTIGUOUS inputs, so
 * group g covers input[16g .. 16g+15] and takes sc[g]. The quants themselves
 * are NOT contiguous: two halves of 128, four quarters per half read at four
 * bit offsets of the same qh byte.
 *
 * WHAT THIS DOES THAT INGOT'S DOES NOT, and it is the only reason to have it:
 * ingot reduces each group of 16 to a scalar with vaddvq_f32 and then applies
 * the scale in scalar float. That is 16 horizontal reductions plus 16 scalar
 * multiply-adds per super-block, and a horizontal reduce is the one NEON
 * operation with no throughput to give. Here the group accumulator is folded
 * into a running VECTOR accumulator with vmlaq_n_f32 — the scale still applied
 * once per 16 weights, but the reduction happens ONCE per row.
 *
 * Deliberately NOT done here: the distribute-the-sum identity that Q4_K uses.
 * SUM q_j x_j - 32 SUM x_j would hoist the bias out, but q is centred on 32,
 * so the two terms are the same size as each other and much larger than their
 * difference — it trades one cheap subtract per 8 weights for a catastrophic
 * cancellation. Q4_K's min term is a separate STORED value and subtracts
 * nothing it just added; this one would. Different identity, different answer.
 *
 * The bias also happens to be free in the int8 path below: q - 32 lands in
 * [-32, 31], which IS an int8, so SDOT takes it directly with no fix-up. */
#if defined(__ARM_NEON)

#define Q6K_ACC(acc, q, off)                                                   \
    do {                                                                       \
        const int16x8_t w_ = vmovl_s8(q);                                      \
        acc = vmlaq_f32(acc, vcvtq_f32_s32(vmovl_s16(vget_low_s16(w_))),       \
                        vld1q_f32(in + (off) + i));                            \
        acc = vmlaq_f32(acc, vcvtq_f32_s32(vmovl_s16(vget_high_s16(w_))),      \
                        vld1q_f32(in + (off) + i + 4));                        \
    } while (0)

static float q6_k_row(const unsigned char *row, size_t blocks, const float *x) {
    float32x4_t total = vdupq_n_f32(0.0f);

    const uint8x8_t nib  = vdup_n_u8(0x0f);
    const uint8x8_t two  = vdup_n_u8(0x03);
    const int8x8_t  bias = vdup_n_s8(32);

    for (size_t b = 0; b < blocks; b++) {
        const unsigned char *block = row + b * 210;
        const unsigned char *ql    = block;
        const unsigned char *qh    = block + 128;
        const signed char   *sc    = (const signed char *)(block + 192);
        const float          d     = mynah_slm_f16_to_f32(block + 208);
        const float         *bx    = x + b * 256;

        for (int half = 0; half < 2; half++) {
            const float       *in = bx + half * 128;
            const signed char *s  = sc + half * 8;

            for (int is = 0; is < 2; is++) {
                float32x4_t a1 = vdupq_n_f32(0.0f), a2 = vdupq_n_f32(0.0f);
                float32x4_t a3 = vdupq_n_f32(0.0f), a4 = vdupq_n_f32(0.0f);

                for (int k = 0; k < 16; k += 8) {
                    const int i = is * 16 + k;
                    const uint8x8_t la = vld1_u8(ql + i);
                    const uint8x8_t lb = vld1_u8(ql + i + 32);
                    const uint8x8_t h  = vld1_u8(qh + i);

                    const int8x8_t q1 = vsub_s8(vreinterpret_s8_u8(
                        vorr_u8(vand_u8(la, nib), vshl_n_u8(vand_u8(h, two), 4))), bias);
                    const int8x8_t q2 = vsub_s8(vreinterpret_s8_u8(
                        vorr_u8(vand_u8(lb, nib),
                                vshl_n_u8(vand_u8(vshr_n_u8(h, 2), two), 4))), bias);
                    const int8x8_t q3 = vsub_s8(vreinterpret_s8_u8(
                        vorr_u8(vshr_n_u8(la, 4),
                                vshl_n_u8(vand_u8(vshr_n_u8(h, 4), two), 4))), bias);
                    /* h >> 6 is already the top two bits — no mask needed. */
                    const int8x8_t q4 = vsub_s8(vreinterpret_s8_u8(
                        vorr_u8(vshr_n_u8(lb, 4), vshl_n_u8(vshr_n_u8(h, 6), 4))), bias);

                    Q6K_ACC(a1, q1, 0);
                    Q6K_ACC(a2, q2, 32);
                    Q6K_ACC(a3, q3, 64);
                    Q6K_ACC(a4, q4, 96);
                }

                /* The whole point: fold into a vector, do not reduce here. */
                total = vmlaq_n_f32(total, a1, d * (float)s[is]);
                total = vmlaq_n_f32(total, a2, d * (float)s[is + 2]);
                total = vmlaq_n_f32(total, a3, d * (float)s[is + 4]);
                total = vmlaq_n_f32(total, a4, d * (float)s[is + 6]);
            }
            ql += 64;
            qh += 32;
        }
    }
    return vaddvq_f32(total);
}
#undef Q6K_ACC

#elif defined(__AVX2__)

#define Q6K_ACC(acc, q, off)                                                   \
    do {                                                                       \
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(q)),     \
                              _mm256_loadu_ps(in + (off) + i), acc);           \
    } while (0)

static float q6_k_row(const unsigned char *row, size_t blocks, const float *x) {
    __m256 total = _mm256_setzero_ps();

    const __m128i nib  = _mm_set1_epi8(0x0f);
    const __m128i two  = _mm_set1_epi8(0x03);
    const __m128i bias = _mm_set1_epi8(32);

    for (size_t b = 0; b < blocks; b++) {
        const unsigned char *block = row + b * 210;
        const unsigned char *ql    = block;
        const unsigned char *qh    = block + 128;
        const signed char   *sc    = (const signed char *)(block + 192);
        const float          d     = mynah_slm_f16_to_f32(block + 208);
        const float         *bx    = x + b * 256;

        for (int half = 0; half < 2; half++) {
            const float       *in = bx + half * 128;
            const signed char *s  = sc + half * 8;

            for (int is = 0; is < 2; is++) {
                __m256 a1 = _mm256_setzero_ps(), a2 = _mm256_setzero_ps();
                __m256 a3 = _mm256_setzero_ps(), a4 = _mm256_setzero_ps();

                for (int k = 0; k < 16; k += 8) {
                    const int i = is * 16 + k;
                    const __m128i la = _mm_loadl_epi64((const __m128i *)(const void *)(ql + i));
                    const __m128i lb = _mm_loadl_epi64((const __m128i *)(const void *)(ql + i + 32));
                    const __m128i h  = _mm_loadl_epi64((const __m128i *)(const void *)(qh + i));

                    /* srli_epi16 shifts 16-bit lanes, so every shift of a byte
                     * vector has to be masked afterwards. The one exception is
                     * h >> 6 followed by & 3 — kept explicit here rather than
                     * relying on that, because the NEON twin can skip it and
                     * this one cannot. */
                    const __m128i h2 = _mm_and_si128(_mm_srli_epi16(h, 2), two);
                    const __m128i h4 = _mm_and_si128(_mm_srli_epi16(h, 4), two);
                    const __m128i h6 = _mm_and_si128(_mm_srli_epi16(h, 6), two);

                    const __m128i q1 = _mm_sub_epi8(
                        _mm_or_si128(_mm_and_si128(la, nib),
                                     _mm_slli_epi16(_mm_and_si128(h, two), 4)), bias);
                    const __m128i q2 = _mm_sub_epi8(
                        _mm_or_si128(_mm_and_si128(lb, nib),
                                     _mm_slli_epi16(h2, 4)), bias);
                    const __m128i q3 = _mm_sub_epi8(
                        _mm_or_si128(_mm_and_si128(_mm_srli_epi16(la, 4), nib),
                                     _mm_slli_epi16(h4, 4)), bias);
                    const __m128i q4 = _mm_sub_epi8(
                        _mm_or_si128(_mm_and_si128(_mm_srli_epi16(lb, 4), nib),
                                     _mm_slli_epi16(h6, 4)), bias);

                    Q6K_ACC(a1, q1, 0);
                    Q6K_ACC(a2, q2, 32);
                    Q6K_ACC(a3, q3, 64);
                    Q6K_ACC(a4, q4, 96);
                }

                total = _mm256_fmadd_ps(a1, _mm256_set1_ps(d * (float)s[is]),     total);
                total = _mm256_fmadd_ps(a2, _mm256_set1_ps(d * (float)s[is + 2]), total);
                total = _mm256_fmadd_ps(a3, _mm256_set1_ps(d * (float)s[is + 4]), total);
                total = _mm256_fmadd_ps(a4, _mm256_set1_ps(d * (float)s[is + 6]), total);
            }
            ql += 64;
            qh += 32;
        }
    }
    return hsum256(total);
}
#undef Q6K_ACC

#else   /* the scalar twin, and the reference for what the other two mean */

static float q6_k_row(const unsigned char *row, size_t blocks, const float *x) {
    float total = 0.0f;

    for (size_t b = 0; b < blocks; b++) {
        const unsigned char *block = row + b * 210;
        const unsigned char *ql    = block;
        const unsigned char *qh    = block + 128;
        const signed char   *sc    = (const signed char *)(block + 192);
        const float          d     = mynah_slm_f16_to_f32(block + 208);
        const float         *bx    = x + b * 256;

        for (int half = 0; half < 2; half++) {
            const float       *in = bx + half * 128;
            const signed char *s  = sc + half * 8;

            for (int is = 0; is < 2; is++) {
                float a1 = 0.0f, a2 = 0.0f, a3 = 0.0f, a4 = 0.0f;
                for (int k = 0; k < 16; k++) {
                    const int i = is * 16 + k;
                    const int h = qh[i];
                    a1 += (float)(((ql[i]      & 0x0f) | ((h & 3) << 4)) - 32) * in[i];
                    a2 += (float)(((ql[i + 32] & 0x0f) | (((h >> 2) & 3) << 4)) - 32) * in[32 + i];
                    a3 += (float)(((ql[i]      >>   4) | (((h >> 4) & 3) << 4)) - 32) * in[64 + i];
                    a4 += (float)(((ql[i + 32] >>   4) | (((h >> 6) & 3) << 4)) - 32) * in[96 + i];
                }
                total += d * ((float)s[is]     * a1 + (float)s[is + 2] * a2 +
                              (float)s[is + 4] * a3 + (float)s[is + 6] * a4);
            }
            ql += 64;
            qh += 32;
        }
    }
    return total;
}

#endif

int mynah_slm_matvec(int type, const void *weights, size_t rows, size_t cols,
                     const float *input, const mynah_slm_matvec_in *prep,
                     float *output) {
    if (!prep || !use_own_kernels()) return -1;
    if (cols % 256 != 0) return -1;         /* K-quant super-blocks, by definition */

    if (type == INGOT_TYPE_Q6_K) {
        if (!own_q6_k()) return -1;
        const size_t blocks = cols / 256;
        const unsigned char *base = (const unsigned char *)weights;
        /* No int8 twin here on purpose. It was written and measured: SDOT
         * collapses the widen-convert-FMA chain into a quarter of an
         * instruction per weight and changed the LM head by 2% — a tie, on the
         * same plateau the f32 kernel hits. Cutting the op count did nothing,
         * which is the evidence that this kernel is not issue-bound, and a
         * quality trade-off that buys nothing is not a trade. */
        for (size_t r = 0; r < rows; r++)
            output[r] = q6_k_row(base + r * blocks * 210, blocks, input);
        return 0;
    }
    if (type != INGOT_TYPE_Q4_K) return -1;

    const size_t blocks = cols / 256;
    const unsigned char *base = (const unsigned char *)weights;

#if defined(MYNAH_SLM_HAVE_SDOT)
    if (prep->have_int8) {
        for (size_t r = 0; r < rows; r++)
            output[r] = q4_k_row_int8(base + r * blocks * 144, blocks,
                                      prep->xq, prep->xscale, prep->xsum);
        return 0;
    }
#endif
    for (size_t r = 0; r < rows; r++)
        output[r] = q4_k_row(base + r * blocks * 144, blocks, input, prep->xsum);
    return 0;
}

/* One strip's dequantization, split across the pool. Each chunk decodes a
 * disjoint run of rows into a disjoint slice of the scratch, so the result
 * does not depend on how many threads ran. */
typedef struct {
    const unsigned char *base;      /* first byte of the strip's first row */
    float               *scratch;
    size_t               cols, row_bytes, rows, rows_per_chunk;
    int                  type, rc;
} dequant_job;

static void dequant_chunk(void *ctx, int i) {
    dequant_job *j = ctx;
    const size_t first = (size_t)i * j->rows_per_chunk;
    if (first >= j->rows) return;
    size_t n = j->rows_per_chunk;
    if (first + n > j->rows) n = j->rows - first;

    if (ingot_dequant_matrix(j->type, j->base + first * j->row_bytes, n, j->cols,
                             j->scratch + first * j->cols) != 0)
        j->rc = -1;                 /* benign race: every failure writes -1 */
}

int mynah_slm_qmatmat(int type, const void *weights, size_t rows, size_t cols,
                      const float *in, float *out, size_t tokens, float *scratch) {
    if (!weights || !in || !out || !scratch || rows == 0 || cols == 0 || tokens == 0)
        return -1;

    uint64_t block_elems = 0, block_bytes = 0;
    if (ingot_type_geometry(type, &block_elems, &block_bytes) != 0 ||
        block_elems == 0 || cols % block_elems != 0)
        return -1;

    const size_t row_bytes = (cols / (size_t)block_elems) * (size_t)block_bytes;
    const int    nth       = mynah_slm_threads_count();

    for (size_t row0 = 0; row0 < rows; row0 += MYNAH_SLM_STRIP_ROWS) {
        size_t n_rows = rows - row0;
        if (n_rows > MYNAH_SLM_STRIP_ROWS) n_rows = MYNAH_SLM_STRIP_ROWS;

        dequant_job j = {
            .base = (const unsigned char *)weights + row0 * row_bytes,
            .scratch = scratch, .cols = cols, .row_bytes = row_bytes,
            .rows = n_rows, .type = type, .rc = 0,
        };

        /* Two chunks per thread here, not four: a strip is small and the
         * dispatch is not free. The rows are equal-cost, unlike a matvec's
         * ragged tail, so there is less imbalance to absorb. */
        int chunks = nth * 2;
        if ((size_t)chunks > n_rows) chunks = (int)n_rows;
        j.rows_per_chunk = (n_rows + (size_t)chunks - 1) / (size_t)chunks;

        mynah_slm_parallel_for(chunks, dequant_chunk, &j);
        if (j.rc != 0) return -1;

        /* BLAS is called from ONE thread with the pool idle, never from inside
         * a parallel region: it brings its own threads, and two pools over the
         * same cores is the throughput collapse mynah-asr measured. */
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    (int)tokens, (int)n_rows, (int)cols,
                    1.0f, in, (int)cols, scratch, (int)cols,
                    0.0f, out + row0, (int)rows);
    }
    return 0;
}
