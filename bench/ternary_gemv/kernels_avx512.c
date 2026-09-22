/* kernels_avx512.c -- the x86 AVX-512 VNNI arm.
 *
 * This is the ISA the literature says is the format's best target, and until
 * now we had no machine to test that on. `[PAPER]` fucina measures its TQ2_0
 * kernel at "~2.1x Q4_K on ARM and ~4.8x on x86-VNNI", and attributes the gap
 * to instruction-set density: `sdot` consumes 16 weights per instruction,
 * `vpdpbusd` 32 on AVX2/VNNI -- and 64 on AVX-512.
 *
 * ---------------------------------------------------------------------------
 * WHY THE BIASED CODES PAY OFF TWICE HERE
 *
 * `vpdpbusd` is asymmetric: its first operand is UNSIGNED bytes, its second
 * SIGNED. On ARM `sdot` both sides are signed, so the bias trick was bought
 * purely to avoid sign-extension in the unpack. On x86 it is bought a second
 * time and for a different reason: the codes are already stored biased into
 * [0,2] (T1) and [0,8] (T3), which is EXACTLY the unsigned operand the
 * instruction wants. A signed ternary encoding would need a correction pass
 * that this one does not.
 *
 * The bias comes back out once per group, unchanged:
 *     sum(t*x) = sum((t+b)*x) - b*sum(x)
 * and sum(x) is hoisted across rows exactly as on ARM.
 *
 * ---------------------------------------------------------------------------
 * OP COUNTS  [DERIVED], to be checked against the measurement
 *
 *   arm            per iteration                          weights  ops/weight
 *   NEON sdot      1 ld + 1 and + 1 shr + 2 sdot   = 5        32      0.156
 *   AVX512 T3      1 ld + 2 and + 1 shr + 2 dpbusd = 6       128      0.047
 *   AVX512 T1      1 ld + 3 and + 3 shr + 4 dpbusd = 11      256      0.043
 *
 * [DERIVED] ~3.3x fewer ops per weight than the NEON arm for T3. That is the
 * same order as the 4.8x/2.1x = 2.3x the paper implies, and the excess is the
 * 512-bit register doing four times sdot's work rather than twice.
 *
 * ---------------------------------------------------------------------------
 * LANE ORDER IS NOT THE SAME AS ON ARM
 *
 * The unpack emits codes in the order the load produced them, so the caller's
 * activation shuffle has to match the REGISTER WIDTH, not just the format:
 *
 *   T3 fold9  NEON   16B load -> 32 codes   -> shuffle_stride(x, .., 32, 2)
 *   T3 fold9  AVX512 64B load -> 128 codes  -> shuffle_stride(x, .., 128, 2)
 *   T1 2-bit  NEON   16B load -> 64 codes   -> shuffle_stride(x, .., 64, 4)
 *   T1 2-bit  AVX512 64B load -> 256 codes  -> shuffle_stride(x, .., 256, 4)
 *
 * Getting this wrong does not crash, it silently returns a plausible wrong
 * number -- which is why every arm is gated against the T0 oracle before it is
 * allowed to be timed.
 *
 * SPDX-License-Identifier: MIT */
#include "dispatch.h"

#if defined(TERNARY_BUILD_AVX512)
#include <immintrin.h>

static inline const float *row_scales_x(const uint8_t *row, ternary_fmt f, size_t cols)
{
    return (const float *)(row + ternary_row_bytes(f, cols) - (cols / TG) * sizeof(float));
}

static inline int32_t hsum512(__m512i v) { return _mm512_reduce_add_epi32(v); }

/* 128 codes per iteration from 64 packed bytes: 1 load + 2 and + 1 shift +
 * 2 vpdpbusd. The nibbles are already the unsigned operand vpdpbusd wants. */
static inline int32_t dot_nibble_512(const uint8_t *p, const int8_t *x, size_t n)
{
    __m512i acc = _mm512_setzero_si512();
    const __m512i m15 = _mm512_set1_epi8(0x0F);
    for (size_t i = 0; i < n; i += 128, p += 64, x += 128) {
        __m512i b  = _mm512_loadu_si512((const void *)p);
        __m512i lo = _mm512_and_si512(b, m15);
        __m512i hi = _mm512_and_si512(_mm512_srli_epi16(b, 4), m15);
        acc = _mm512_dpbusd_epi32(acc, lo, _mm512_loadu_si512((const void *)x));
        acc = _mm512_dpbusd_epi32(acc, hi, _mm512_loadu_si512((const void *)(x + 64)));
    }
    return hsum512(acc);
}

/* 256 codes per iteration from 64 packed bytes: 4 unpacks, 4 vpdpbusd. */
static inline int32_t dot_2bit_512(const uint8_t *p, const int8_t *x, size_t n)
{
    __m512i acc = _mm512_setzero_si512();
    const __m512i m3 = _mm512_set1_epi8(0x03);
    for (size_t i = 0; i < n; i += 256, p += 64, x += 256) {
        __m512i b = _mm512_loadu_si512((const void *)p);
        __m512i c0 = _mm512_and_si512(b, m3);
        __m512i c1 = _mm512_and_si512(_mm512_srli_epi16(b, 2), m3);
        __m512i c2 = _mm512_and_si512(_mm512_srli_epi16(b, 4), m3);
        __m512i c3 = _mm512_and_si512(_mm512_srli_epi16(b, 6), m3);
        acc = _mm512_dpbusd_epi32(acc, c0, _mm512_loadu_si512((const void *)x));
        acc = _mm512_dpbusd_epi32(acc, c1, _mm512_loadu_si512((const void *)(x + 64)));
        acc = _mm512_dpbusd_epi32(acc, c2, _mm512_loadu_si512((const void *)(x + 128)));
        acc = _mm512_dpbusd_epi32(acc, c3, _mm512_loadu_si512((const void *)(x + 192)));
    }
    return hsum512(acc);
}

static inline int32_t dot_int8_512(const int8_t *p, const int8_t *x, size_t n)
{
    __m512i acc = _mm512_setzero_si512();
    for (size_t i = 0; i < n; i += 64, p += 64, x += 64) {
        /* vpdpbusd needs an unsigned first operand, and an int8 weight is not.
         * Split w = (w + 128) - 128: the shifted form is unsigned, and the
         * constant folds into the per-group bias exactly like the code bias. */
        __m512i w = _mm512_loadu_si512((const void *)p);
        __m512i u = _mm512_add_epi8(w, _mm512_set1_epi8((char)0x80));
        acc = _mm512_dpbusd_epi32(acc, u, _mm512_loadu_si512((const void *)x));
    }
    return hsum512(acc);
}

/* ---------------------------------------------------------------------------
 * THE WIDTH CONTROL
 *
 * Our production x86 kernels contain ZERO `_mm512` intrinsics -- every one of
 * them, in src/qmat.c and in ingot, is AVX2 at 256 bits, and Q4_K multiplies
 * with the pre-VNNI `maddubs_epi16 + madd_epi16` pair rather than `vpdpbusd`.
 * So a 512-bit VNNI ternary kernel measured against them is not a comparison
 * of REPRESENTATIONS: it is 4x the register width times 2x the instruction
 * count, and up to 8x of structural advantage that a ternary format did not
 * earn.
 *
 * This arm exists to take the width back out. It is the identical algorithm at
 * 256 bits (EVEX-encoded `vpdpbusd`, which Zen 4 has via AVX512VL+AVX512VNNI),
 * so ternary-at-256 against Q4_K-at-256 isolates the representation and the
 * VNNI-vs-maddubs instruction choice, leaving the register width out of it.
 * ------------------------------------------------------------------------- */
static inline int32_t hsum256i_x(__m256i v)
{
    __m128i a = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    a = _mm_hadd_epi32(a, a);
    a = _mm_hadd_epi32(a, a);
    return _mm_cvtsi128_si32(a);
}

/* 64 codes per iteration from 32 packed bytes. */
static inline int32_t dot_nibble_256(const uint8_t *p, const int8_t *x, size_t n)
{
    __m256i acc = _mm256_setzero_si256();
    const __m256i m15 = _mm256_set1_epi8(0x0F);
    for (size_t i = 0; i < n; i += 64, p += 32, x += 64) {
        __m256i b  = _mm256_loadu_si256((const __m256i *)(const void *)p);
        __m256i lo = _mm256_and_si256(b, m15);
        __m256i hi = _mm256_and_si256(_mm256_srli_epi16(b, 4), m15);
        acc = _mm256_dpbusd_epi32(acc, lo, _mm256_loadu_si256((const __m256i *)(const void *)x));
        acc = _mm256_dpbusd_epi32(acc, hi, _mm256_loadu_si256((const __m256i *)(const void *)(x + 32)));
    }
    return hsum256i_x(acc);
}

/* 128 codes per iteration from 32 packed bytes. */
static inline int32_t dot_2bit_256(const uint8_t *p, const int8_t *x, size_t n)
{
    __m256i acc = _mm256_setzero_si256();
    const __m256i m3 = _mm256_set1_epi8(0x03);
    for (size_t i = 0; i < n; i += 128, p += 32, x += 128) {
        __m256i b = _mm256_loadu_si256((const __m256i *)(const void *)p);
        acc = _mm256_dpbusd_epi32(acc, _mm256_and_si256(b, m3),
                  _mm256_loadu_si256((const __m256i *)(const void *)x));
        acc = _mm256_dpbusd_epi32(acc, _mm256_and_si256(_mm256_srli_epi16(b, 2), m3),
                  _mm256_loadu_si256((const __m256i *)(const void *)(x + 32)));
        acc = _mm256_dpbusd_epi32(acc, _mm256_and_si256(_mm256_srli_epi16(b, 4), m3),
                  _mm256_loadu_si256((const __m256i *)(const void *)(x + 64)));
        acc = _mm256_dpbusd_epi32(acc, _mm256_and_si256(_mm256_srli_epi16(b, 6), m3),
                  _mm256_loadu_si256((const __m256i *)(const void *)(x + 96)));
    }
    return hsum256i_x(acc);
}

static inline int32_t dot_int8_256(const int8_t *p, const int8_t *x, size_t n)
{
    __m256i acc = _mm256_setzero_si256();
    for (size_t i = 0; i < n; i += 32, p += 32, x += 32) {
        __m256i w = _mm256_loadu_si256((const __m256i *)(const void *)p);
        __m256i u = _mm256_add_epi8(w, _mm256_set1_epi8((char)0x80));
        acc = _mm256_dpbusd_epi32(acc, u, _mm256_loadu_si256((const __m256i *)(const void *)x));
    }
    return hsum256i_x(acc);
}

int tgemv_vnni256(ternary_fmt f, const uint8_t *w, size_t rows, size_t cols,
                  const act_t *a, const int8_t *xs2, const int8_t *xs4, float *y);
int tgemv_vnni256(ternary_fmt f, const uint8_t *w, size_t rows, size_t cols,
                  const act_t *a, const int8_t *xs2, const int8_t *xs4, float *y)
{
    if (f != FMT_T0_INT8 && f != FMT_T1_2BIT && f != FMT_T3_FOLD9 && f != FMT_T3_K3)
        return -1;
    size_t rb = ternary_row_bytes(f, cols);
    for (size_t r = 0; r < rows; r++) {
        const uint8_t *row = w + r * rb;
        const float   *ws = row_scales_x(row, f, cols);
        float          acc = 0.0f;
        for (size_t g = 0; g < a->groups; g++) {
            int32_t d;
            int     bias;
            switch (f) {
            case FMT_T0_INT8:
                d = dot_int8_256((const int8_t *)row + g * TG, a->q + g * TG, TG);
                bias = 128;
                break;
            case FMT_T1_2BIT:
                d = dot_2bit_256(row + g * TG / 4, xs2 + g * TG, TG);
                bias = 1;
                break;
            case FMT_T3_FOLD9:
                d = dot_nibble_256(row + g * TG / 2, xs4 + g * TG, TG);
                bias = 4;
                break;
            case FMT_T3_K3:
                d = 9 * dot_2bit_256(row + g * TG / 4, xs2 + g * TG, TG) +
                    dot_nibble_256(row + cols / 4 + g * TG / 2, xs4 + g * TG, TG);
                bias = 13;
                break;
            default:
                return -1;
            }
            acc += ws[g] * a->scale[g] * (float)(d - bias * a->sum[g]);
        }
        y[r] = acc;
    }
    return 0;
}

int tgemv_avx512(ternary_fmt f, const uint8_t *w, size_t rows, size_t cols,
                 const act_t *a, const int8_t *xs2, const int8_t *xs4, float *y);
int tgemv_avx512(ternary_fmt f, const uint8_t *w, size_t rows, size_t cols,
                 const act_t *a, const int8_t *xs2, const int8_t *xs4, float *y)
{
    if (f != FMT_T0_INT8 && f != FMT_T1_2BIT && f != FMT_T3_FOLD9 && f != FMT_T3_K3)
        return -1;
    size_t rb = ternary_row_bytes(f, cols);
    for (size_t r = 0; r < rows; r++) {
        const uint8_t *row = w + r * rb;
        const float   *ws = row_scales_x(row, f, cols);
        float          acc = 0.0f;
        for (size_t g = 0; g < a->groups; g++) {
            int32_t d;
            int     bias;
            switch (f) {
            case FMT_T0_INT8:
                d = dot_int8_512((const int8_t *)row + g * TG, a->q + g * TG, TG);
                bias = 128;          /* the +128 unsigned shift, not a code bias */
                break;
            case FMT_T1_2BIT:
                d = dot_2bit_512(row + g * TG / 4, xs2 + g * TG, TG);
                bias = 1;
                break;
            case FMT_T3_FOLD9:
                d = dot_nibble_512(row + g * TG / 2, xs4 + g * TG, TG);
                bias = 4;
                break;
            case FMT_T3_K3:
                d = 9 * dot_2bit_512(row + g * TG / 4, xs2 + g * TG, TG) +
                    dot_nibble_512(row + cols / 4 + g * TG / 2, xs4 + g * TG, TG);
                bias = 13;
                break;
            default:
                return -1;
            }
            acc += ws[g] * a->scale[g] * (float)(d - bias * a->sum[g]);
        }
        y[r] = acc;
    }
    return 0;
}
#endif /* TERNARY_BUILD_AVX512 */
