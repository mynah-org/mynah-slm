/* qmat.c — see qmat.h.
 * SPDX-License-Identifier: MIT */
#include "qmat.h"

#include "sgemm.h"
#include "threads.h"

#include "ingot/dtype.h"
#include "ingot/quant.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif
#if defined(__AVX2__)
#include <immintrin.h>
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

/* Which types the int8 switch applies to: MYNAH_SLM_INT8_TYPES, a comma list
 * of q4_k / q8_0 / q6_k, default all three. It can only NARROW the switch,
 * never turn int8 on by itself. It exists so the perplexity gate can price
 * each type on ONE binary — the Q6_K head in front of the softmax is a
 * different quality question from the Q4_K layers (.work/int8-q8_0-q6_k.md). */
enum { INT8_Q4_K = 1, INT8_Q8_0 = 2, INT8_Q6_K = 4, INT8_ALL = 7 };
static int g_int8_types = -1;

static int int8_type_bit(int type) {
    return type == INGOT_TYPE_Q4_K ? INT8_Q4_K :
           type == INGOT_TYPE_Q8_0 ? INT8_Q8_0 :
           type == INGOT_TYPE_Q6_K ? INT8_Q6_K : 0;
}

static int int8_type_on(int type) {
    if (g_int8_types < 0) {
        const char *e = getenv("MYNAH_SLM_INT8_TYPES");
        int mask = INT8_ALL;
        if (e && *e) {
            mask = 0;
            if (strstr(e, "q4_k") || strstr(e, "Q4_K")) mask |= INT8_Q4_K;
            if (strstr(e, "q8_0") || strstr(e, "Q8_0")) mask |= INT8_Q8_0;
            if (strstr(e, "q6_k") || strstr(e, "Q6_K")) mask |= INT8_Q6_K;
        }
        g_int8_types = mask;
    }
    return (g_int8_types & int8_type_bit(type)) != 0;
}

void mynah_slm_matvec_set_int8_types(int q4_k, int q8_0, int q6_k) {
    g_int8_types = (q4_k ? INT8_Q4_K : 0) | (q8_0 ? INT8_Q8_0 : 0) | (q6_k ? INT8_Q6_K : 0);
}

int mynah_slm_matvec_have(int type) {
    if (!use_own_kernels()) return 0;
    if (type == INGOT_TYPE_Q4_K) return 1;
    /* Q8_0 and Q6_K: ours only as int8 (K4). Their f32 kernels are ingot's,
     * and docs/perf.md is why — ours tied or lost and went upstream. */
    return (type == INGOT_TYPE_Q8_0 || type == INGOT_TYPE_Q6_K) &&
           mynah_slm_matvec_int8_enabled() && int8_type_on(type);
}

static void matvec_prepare(const float *input, size_t cols,
                           mynah_slm_matvec_in *prep, int want_int8) {
    if (!input || !prep) return;
    prep->have_int8 = want_int8 && cols <= MYNAH_SLM_XQ_MAX;

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

void mynah_slm_matvec_prepare(const float *input, size_t cols,
                              mynah_slm_matvec_in *prep) {
    matvec_prepare(input, cols, prep, mynah_slm_matvec_int8_enabled());
}

void mynah_slm_matvec_prepare_int8(const float *input, size_t cols,
                                   mynah_slm_matvec_in *prep) {
    matvec_prepare(input, cols, prep, 1);
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

/* ── the int8 path ─────────────────────────────────────────────────────────
 * SDOT / VPDPBUSD / maddubs do four int8 multiply-accumulates per lane in one
 * instruction, where the f32 form needs a widen, a convert and an FMA per four
 * values. The price is that the ACTIVATIONS are quantized to int8 per 32
 * values.
 *
 * Both halves of the identity survive it cleanly:
 *
 *     SUM_j w_j x_j = d*scale * xs * SUM_j (q_j * xq_j)  -  dmin*min * SUM_j x_j
 *                                    ^^^^^^^^^^^^^^^^^        ^^^^^^^^^^^^^^^^
 *                                    integer, one SDOT        still exact f32
 *
 * so the min term keeps full precision and only the product term is
 * approximated. Whether that is acceptable is a question for `mynah-slm ppl`,
 * which is why this is off by default.
 *
 * THE SHAPE OF THE KERNEL (.work/q4k-int8-4row.md). Once the dot product is one
 * instruction per 64 weights, the epilogue is the kernel: the first version
 * reduced every 32-weight sub-block to a scalar (a full horizontal int32 sum,
 * five ops on x86) and then did the float scaling in scalar code, plus a
 * branchy 6-bit scale unpack per sub-block. Now:
 *
 *   - the int32 lane partials are converted to f32 and FMA'd with a per-lane
 *     scale into a per-row f32 ACCUMULATOR; the one horizontal reduction
 *     happens at the end of the row;
 *   - all eight scale/min pairs of a block are unpacked at once (ggml's
 *     three-mask form) and the scales built as one 8-wide vector;
 *   - the activation vectors are loaded once and used against FOUR rows.
 *
 * THE CONTRACT that makes this testable rather than "close": every ISA and the
 * scalar twin perform the same float operations in the same order, so a row's
 * result is bit-identical across ISAs, across row grouping (a row in a group of
 * four or in the tail) and across thread counts. Per row:
 *
 *   for each 256-weight block:
 *     S[s] = (d * sc[s]) * xscale[s]                          s = 0..7
 *     M[s] = fma(dmin, mn[s] * xsum[s], M[s])
 *     for each sub-block pair p (s0 = 2p, s1 = 2p+1), lane j = 0..7:
 *       P0[j] = SUM_{i=4j..4j+3} lo[i] * xq[i]       (exact int32)
 *       P1[j] = SUM_{i=4j..4j+3} hi[i] * xq[32 + i]  (exact int32)
 *       E[j]  = fma((float)P0[j], S[s0], E[j])
 *       O[j]  = fma((float)P1[j], S[s1], O[j])
 *   result = tree8(E + O) - tree8(M)
 *
 * The lane definition (four consecutive bytes per int32 lane) is what each
 * target produces natively — maddubs+madd on 32 bytes, vpdpbusd on [lo|hi]
 * 64 bytes (E and O are its two halves), two sdot on 16-byte halves — so no
 * ISA pays a shuffle for the contract. tree8 is the reduction the x86 code
 * already did; NEON spells it out because vaddvq_f32 is a different tree. */

/* All eight 6-bit scale/min pairs of a block at once, ggml's form: bytes 0-7
 * of `out` are the scales, bytes 8-15 the mins. Same values as
 * q4_k_scale_min() for every index (tests/test_kernels.c checks the int8
 * kernels against ingot's dequantizer, which is where a mistake here lands). */
static inline void q4_k_unpack8(const unsigned char *scales, uint8_t out[16]) {
    const uint32_t m1 = 0x3f3f3f3fu, m2 = 0x0f0f0f0fu, m3 = 0x03030303u;
    uint32_t u[4];
    memcpy(u, scales, 12);
    u[3] = ((u[2] >> 4) & m2) | (((u[1] >> 6) & m3) << 4);
    const uint32_t mins_lo = u[1] & m1;
    u[1] = (u[2] & m2) | (((u[0] >> 6) & m3) << 4);
    u[2] = mins_lo;
    u[0] &= m1;
    memcpy(out, u, 16);
}

static inline float q4k_tree8(const float v[8]) {
    const float a0 = v[0] + v[4], a1 = v[1] + v[5];
    const float a2 = v[2] + v[6], a3 = v[3] + v[7];
    return (a0 + a2) + (a1 + a3);
}

/* The scalar twin. Not a fallback anybody ships — no build enables the int8
 * path without a vector kernel — but the executable definition of the
 * contract above, which every vector kernel must match bit for bit. fmaf()
 * rather than a*b+c, so that neither -ffp-contract nor a missing FMA unit can
 * change what it means. */
static float q4k_i8_row_scalar(const unsigned char *row, size_t blocks,
                               const int8_t *xq, const float *xscale,
                               const float *xsum) {
    float e[8] = {0}, o[8] = {0}, m[8] = {0};

    for (size_t b = 0; b < blocks; b++) {
        const unsigned char *block = row + b * 144;
        const float d    = mynah_slm_f16_to_f32(block);
        const float dmin = mynah_slm_f16_to_f32(block + 2);
        uint8_t u[16];
        q4_k_unpack8(block + 4, u);

        float s[8];
        for (int k = 0; k < 8; k++) {
            s[k] = (d * (float)u[k]) * xscale[b * 8 + k];
            const float t = (float)u[8 + k] * xsum[b * 8 + k];
            m[k] = fmaf(dmin, t, m[k]);
        }

        const unsigned char *q = block + 16;
        const int8_t *x = xq + b * 256;
        for (int p = 0; p < 4; p++, q += 32, x += 64) {
            for (int j = 0; j < 8; j++) {
                int32_t p0 = 0, p1 = 0;
                for (int i = 4 * j; i < 4 * j + 4; i++) {
                    p0 += (int32_t)(q[i] & 0x0f) * (int32_t)x[i];
                    p1 += (int32_t)(q[i] >> 4)   * (int32_t)x[32 + i];
                }
                e[j] = fmaf((float)p0, s[2 * p],     e[j]);
                o[j] = fmaf((float)p1, s[2 * p + 1], o[j]);
            }
        }
    }

    float v[8];
    for (int j = 0; j < 8; j++) v[j] = e[j] + o[j];
    return q4k_tree8(v) - q4k_tree8(m);
}

static void q4k_i8_rows_scalar(const unsigned char *base, size_t rows,
                               size_t blocks, const mynah_slm_matvec_in *p,
                               float *out) {
    for (size_t r = 0; r < rows; r++)
        out[r] = q4k_i8_row_scalar(base + r * blocks * 144, blocks,
                                   p->xq, p->xscale, p->xsum);
}

#if defined(MYNAH_SLM_HAVE_SDOT)
#if defined(__GNUC__) || defined(__clang__)
#define Q4K_INLINE static inline __attribute__((always_inline))
#else
#define Q4K_INLINE static inline
#endif

#if defined(__AVX2__)
/* tree8 over a ymm: lo128 + hi128, then movehl, then the last pair. */
static inline float q4k_tree256(__m256 v) {
    __m128 a = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    a = _mm_add_ps(a, _mm_movehl_ps(a, a));
    a = _mm_add_ss(a, _mm_shuffle_ps(a, a, 0x55));
    return _mm_cvtss_f32(a);
}

static inline __m256 q4k_u8x8_f32(const uint8_t *b8) {
    return _mm256_cvtepi32_ps(_mm256_cvtepu8_epi32(
        _mm_loadl_epi64((const __m128i *)(const void *)b8)));
}

/* AVX-512 VNNI, ported from qwen-tts's int8 kernel stack (docs/prior-art.md).
 *
 * WHY THIS EXISTS. The measurement that forced it: on an AMD EPYC 9254 (Zen 4)
 * this Q4_K kernel ran at 7.8 GB/s while the SAME kernel on an Apple M1 ran at
 * 13.1 -- a server was slower than a laptop, because every x86 path we owned
 * was AVX2 at 256 bits with the pre-VNNI `maddubs_epi16 + madd_epi16` pair, on
 * a CPU that has had a single-instruction u8xi8 dot product since 2022.
 *
 * THE LAYOUT TRICK. A Q4_K sub-block pair shares its 32 packed bytes: the low
 * nibbles are elements [base, base+32) and the high nibbles [base+32, base+64).
 * Concatenating the two unpacked halves into one 64-byte vector makes the
 * matching activations EXACTLY the contiguous 64 bytes at xq+base, so one
 * `vpdpbusd` covers both sub-blocks with a single load and no gather:
 *
 *     W = [ lo_nibbles(32B) | hi_nibbles(32B) ]     <- one insert
 *     X = xq[base .. base+64)                       <- one contiguous load
 *     acc = vpdpbusd(0, W, X)   lanes 0-7 -> sub-block s0, lanes 8-15 -> s1
 *
 * vpdpbusd wants an UNSIGNED first operand and the nibbles are 0..15, so no
 * correction is needed. It also removes the one real hazard in the AVX2 path:
 * `maddubs` accumulates into int16 and saturates, which is safe there only
 * because 15*128*2 = 3840 happens to fit. VNNI accumulates in int32.
 *
 * The scale for the pair is one permute of the block's 8-wide scale vector:
 * lanes 0-7 get S[2p], lanes 8-15 get S[2p+1], matching the int32 layout. */
#if defined(__AVX512VNNI__) && defined(__AVX512BW__) && defined(__AVX512F__)
#define MYNAH_SLM_HAVE_AVX512VNNI 1

Q4K_INLINE void q4k_i8_avx512(const unsigned char *row0, size_t row_bytes,
                              const int nr, size_t blocks,
                              const int8_t *xq, const float *xscale,
                              const float *xsum, float *out) {
    __m512 acc[4];
    __m256 mins[4];
    for (int r = 0; r < nr; r++) {
        acc[r]  = _mm512_setzero_ps();
        mins[r] = _mm256_setzero_ps();
    }
    const __m256i nib = _mm256_set1_epi8(0x0f);
    const __m512i sel[4] = {
        _mm512_set_epi32(1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0),
        _mm512_set_epi32(3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2),
        _mm512_set_epi32(5, 5, 5, 5, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4),
        _mm512_set_epi32(7, 7, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6),
    };

    for (size_t b = 0; b < blocks; b++) {
        const __m256 xs = _mm256_loadu_ps(xscale + b * 8);
        const __m256 xm = _mm256_loadu_ps(xsum + b * 8);
        __m512 s[4];
        for (int r = 0; r < nr; r++) {
            const unsigned char *block = row0 + (size_t)r * row_bytes + b * 144;
            uint8_t u[16];
            q4_k_unpack8(block + 4, u);
            const __m256 d = _mm256_set1_ps(mynah_slm_f16_to_f32(block));
            const __m256 dmin = _mm256_set1_ps(mynah_slm_f16_to_f32(block + 2));
            s[r] = _mm512_castps256_ps512(
                _mm256_mul_ps(_mm256_mul_ps(d, q4k_u8x8_f32(u)), xs));
            mins[r] = _mm256_fmadd_ps(dmin, _mm256_mul_ps(q4k_u8x8_f32(u + 8), xm),
                                      mins[r]);
        }
        for (int p = 0; p < 4; p++) {
            const __m512i x = _mm512_loadu_si512((const void *)(xq + b * 256 + (size_t)p * 64));
            for (int r = 0; r < nr; r++) {
                const unsigned char *q =
                    row0 + (size_t)r * row_bytes + b * 144 + 16 + (size_t)p * 32;
                const __m256i pk = _mm256_loadu_si256((const __m256i *)(const void *)q);
                const __m256i nl = _mm256_and_si256(pk, nib);
                const __m256i nh = _mm256_and_si256(_mm256_srli_epi16(pk, 4), nib);
                const __m512i w  = _mm512_inserti64x4(_mm512_castsi256_si512(nl), nh, 1);
                const __m512i i32 = _mm512_dpbusd_epi32(_mm512_setzero_si512(), w, x);
                acc[r] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(i32),
                                         _mm512_permutexvar_ps(sel[p], s[r]), acc[r]);
            }
        }
    }
    for (int r = 0; r < nr; r++) {
        const __m256 lo = _mm512_castps512_ps256(acc[r]);
        const __m256 hi = _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(acc[r]), 1));
        out[r] = q4k_tree256(_mm256_add_ps(lo, hi)) - q4k_tree256(mins[r]);
    }
}
#endif /* AVX-512 VNNI */

#if !defined(MYNAH_SLM_HAVE_AVX512VNNI)
/* AVX2: no VNNI required. maddubs multiplies u8 by i8 into i16 pairs and madd
 * folds those into i32 lanes of four consecutive bytes — the contract's lanes.
 * The nibbles are 0..15 and the activations int8, so the largest i16 partial
 * is 15*128*2 = 3840: far inside i16, and maddubs saturates rather than wraps,
 * which would otherwise be the trap here.
 *
 * Sixteen ymm registers is the whole file: four rows x (E, O) is eight, the two
 * activation halves, the mask and the ones are four more. So the per-block
 * scales live in a small stack array and are broadcast from memory (a load-port
 * op, no register), and the min accumulators are spilled explicitly — they
 * change once per 256 weights, not once per 32. */
Q4K_INLINE void q4k_i8_avx2(const unsigned char *row0, size_t row_bytes,
                            const int nr, size_t blocks,
                            const int8_t *xq, const float *xscale,
                            const float *xsum, float *out) {
    __m256 e[4], o[4];
    _Alignas(32) float mins[4][8];
    _Alignas(32) float s[4][8];
    for (int r = 0; r < nr; r++) {
        e[r] = _mm256_setzero_ps();
        o[r] = _mm256_setzero_ps();
        _mm256_store_ps(mins[r], _mm256_setzero_ps());
    }
    const __m256i nib  = _mm256_set1_epi8(0x0f);
    const __m256i ones = _mm256_set1_epi16(1);

    for (size_t b = 0; b < blocks; b++) {
        const __m256 xs = _mm256_loadu_ps(xscale + b * 8);
        const __m256 xm = _mm256_loadu_ps(xsum + b * 8);
        for (int r = 0; r < nr; r++) {
            const unsigned char *block = row0 + (size_t)r * row_bytes + b * 144;
            uint8_t u[16];
            q4_k_unpack8(block + 4, u);
            const __m256 d = _mm256_set1_ps(mynah_slm_f16_to_f32(block));
            const __m256 dmin = _mm256_set1_ps(mynah_slm_f16_to_f32(block + 2));
            _mm256_store_ps(s[r], _mm256_mul_ps(_mm256_mul_ps(d, q4k_u8x8_f32(u)), xs));
            _mm256_store_ps(mins[r],
                            _mm256_fmadd_ps(dmin, _mm256_mul_ps(q4k_u8x8_f32(u + 8), xm),
                                            _mm256_load_ps(mins[r])));
        }
        for (int p = 0; p < 4; p++) {
            const int8_t *xb = xq + b * 256 + (size_t)p * 64;
            const __m256i xl = _mm256_loadu_si256((const __m256i *)(const void *)xb);
            const __m256i xh = _mm256_loadu_si256((const __m256i *)(const void *)(xb + 32));
            for (int r = 0; r < nr; r++) {
                const unsigned char *q =
                    row0 + (size_t)r * row_bytes + b * 144 + 16 + (size_t)p * 32;
                const __m256i pk = _mm256_loadu_si256((const __m256i *)(const void *)q);
                const __m256i nl = _mm256_and_si256(pk, nib);
                const __m256i nh = _mm256_and_si256(_mm256_srli_epi16(pk, 4), nib);
                const __m256i i0 = _mm256_madd_epi16(_mm256_maddubs_epi16(nl, xl), ones);
                const __m256i i1 = _mm256_madd_epi16(_mm256_maddubs_epi16(nh, xh), ones);
                e[r] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(i0),
                                       _mm256_broadcast_ss(&s[r][2 * p]), e[r]);
                o[r] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(i1),
                                       _mm256_broadcast_ss(&s[r][2 * p + 1]), o[r]);
            }
        }
    }
    for (int r = 0; r < nr; r++)
        out[r] = q4k_tree256(_mm256_add_ps(e[r], o[r])) -
                 q4k_tree256(_mm256_load_ps(mins[r]));
}
#endif /* !AVX-512 VNNI */
#endif /* __AVX2__ */

#if defined(__ARM_NEON)
/* NEON SDOT. vdotq_s32 sums four consecutive bytes per lane of a 16-byte
 * vector, so two of them per 32-weight sub-block give the contract's lanes
 * 0-3 and 4-7. Four rows x (E, O) x two halves is 16 of the 32 q registers;
 * the scales are read as scalars from a stack array by the by-element FMA, so
 * they cost no register either. */
static inline float q4k_tree_neon(float32x4_t lo, float32x4_t hi) {
    const float32x4_t a = vaddq_f32(lo, hi);                  /* v_j + v_{j+4} */
    const float32x2_t b = vadd_f32(vget_low_f32(a), vget_high_f32(a));
    return vget_lane_f32(b, 0) + vget_lane_f32(b, 1);
}

static inline float32x4_t q4k_u8x4_f32(uint8x8_t v, int high) {
    const uint16x8_t w = vmovl_u8(v);
    return vcvtq_f32_u32(vmovl_u16(high ? vget_high_u16(w) : vget_low_u16(w)));
}

Q4K_INLINE void q4k_i8_neon(const unsigned char *row0, size_t row_bytes,
                            const int nr, size_t blocks,
                            const int8_t *xq, const float *xscale,
                            const float *xsum, float *out) {
    float32x4_t e0[4], e1[4], o0[4], o1[4], m0[4], m1[4];
    float s[4][8];
    for (int r = 0; r < nr; r++) {
        e0[r] = e1[r] = o0[r] = o1[r] = m0[r] = m1[r] = vdupq_n_f32(0.0f);
    }
    const uint8x16_t nib = vdupq_n_u8(0x0f);
    const int32x4_t  z   = vdupq_n_s32(0);

    for (size_t b = 0; b < blocks; b++) {
        const float32x4_t xs0 = vld1q_f32(xscale + b * 8), xs1 = vld1q_f32(xscale + b * 8 + 4);
        const float32x4_t xm0 = vld1q_f32(xsum + b * 8),   xm1 = vld1q_f32(xsum + b * 8 + 4);
        for (int r = 0; r < nr; r++) {
            const unsigned char *block = row0 + (size_t)r * row_bytes + b * 144;
            uint8_t u[16];
            q4_k_unpack8(block + 4, u);
            const float d    = mynah_slm_f16_to_f32(block);
            const float dmin = mynah_slm_f16_to_f32(block + 2);
            const uint8x8_t sc = vld1_u8(u), mn = vld1_u8(u + 8);
            /* (d * sc) * xs: vmulq_n_f32 is sc*d, the same rounding. */
            vst1q_f32(s[r],     vmulq_f32(vmulq_n_f32(q4k_u8x4_f32(sc, 0), d), xs0));
            vst1q_f32(s[r] + 4, vmulq_f32(vmulq_n_f32(q4k_u8x4_f32(sc, 1), d), xs1));
            m0[r] = vfmaq_n_f32(m0[r], vmulq_f32(q4k_u8x4_f32(mn, 0), xm0), dmin);
            m1[r] = vfmaq_n_f32(m1[r], vmulq_f32(q4k_u8x4_f32(mn, 1), xm1), dmin);
        }
        for (int p = 0; p < 4; p++) {
            const int8_t *xb = xq + b * 256 + (size_t)p * 64;
            const int8x16_t x0 = vld1q_s8(xb),      x1 = vld1q_s8(xb + 16);
            const int8x16_t x2 = vld1q_s8(xb + 32), x3 = vld1q_s8(xb + 48);
            for (int r = 0; r < nr; r++) {
                const unsigned char *q =
                    row0 + (size_t)r * row_bytes + b * 144 + 16 + (size_t)p * 32;
                const uint8x16_t q0 = vld1q_u8(q), q1 = vld1q_u8(q + 16);
                const int8x16_t l0 = vreinterpretq_s8_u8(vandq_u8(q0, nib));
                const int8x16_t l1 = vreinterpretq_s8_u8(vandq_u8(q1, nib));
                const int8x16_t h0 = vreinterpretq_s8_u8(vshrq_n_u8(q0, 4));
                const int8x16_t h1 = vreinterpretq_s8_u8(vshrq_n_u8(q1, 4));
                const float se = s[r][2 * p], so = s[r][2 * p + 1];
                e0[r] = vfmaq_n_f32(e0[r], vcvtq_f32_s32(vdotq_s32(z, l0, x0)), se);
                e1[r] = vfmaq_n_f32(e1[r], vcvtq_f32_s32(vdotq_s32(z, l1, x1)), se);
                o0[r] = vfmaq_n_f32(o0[r], vcvtq_f32_s32(vdotq_s32(z, h0, x2)), so);
                o1[r] = vfmaq_n_f32(o1[r], vcvtq_f32_s32(vdotq_s32(z, h1, x3)), so);
            }
        }
    }
    for (int r = 0; r < nr; r++)
        out[r] = q4k_tree_neon(vaddq_f32(e0[r], o0[r]), vaddq_f32(e1[r], o1[r])) -
                 q4k_tree_neon(m0[r], m1[r]);
}
#endif /* __ARM_NEON */

#if defined(__ARM_NEON)
#define Q4K_I8_KERNEL q4k_i8_neon
#define MYNAH_SLM_INT8_ISA "neon-dotprod"
#elif defined(MYNAH_SLM_HAVE_AVX512VNNI)
#define Q4K_I8_KERNEL q4k_i8_avx512
#define MYNAH_SLM_INT8_ISA "avx512-vnni"
#else
#define Q4K_I8_KERNEL q4k_i8_avx2
#define MYNAH_SLM_INT8_ISA "avx2"
#endif

/* Four rows per activation load, then the tail one row at a time. The tail
 * runs the same per-row arithmetic, so where a row lands does not change it. */
static void q4k_i8_rows(const unsigned char *base, size_t rows, size_t blocks,
                        const mynah_slm_matvec_in *p, float *out) {
    const size_t rb = blocks * 144;
    size_t r = 0;
    for (; r + 4 <= rows; r += 4)
        Q4K_I8_KERNEL(base + r * rb, rb, 4, blocks, p->xq, p->xscale, p->xsum, out + r);
    for (; r < rows; r++)
        Q4K_I8_KERNEL(base + r * rb, rb, 1, blocks, p->xq, p->xscale, p->xsum, out + r);
}
#endif /* MYNAH_SLM_HAVE_SDOT */

const char *mynah_slm_matvec_int8_isa(void) {
#if defined(MYNAH_SLM_HAVE_SDOT)
    return MYNAH_SLM_INT8_ISA;
#else
    return "none";
#endif
}

int mynah_slm_q4k_int8_ref(const void *weights, size_t rows, size_t cols,
                           const mynah_slm_matvec_in *prep, float *output) {
    if (!weights || !prep || !output || !prep->have_int8 || cols % 256 != 0)
        return -1;
    q4k_i8_rows_scalar((const unsigned char *)weights, rows, cols / 256, prep, output);
    return 0;
}

/* ── int8 activations for Q8_0 and Q6_K (.work/int8-q8_0-q6_k.md) ─────────
 * Same switch as the Q4_K int8 path, same prepared activations (xq int8 per
 * 32 values, xscale), same contract: int32 lanes of four consecutive bytes,
 * f32 accumulators split by the parity of the 32-value chunk (E even, O odd),
 * one tree8 reduction per row, a scalar twin in the same order.
 *
 * Q8_0 (34 B: f16 d, 32 int8). One block is one 32-value chunk with one
 * scale S = d * xscale[b]:
 *     L[j] = SUM_{i=4j..4j+3} w[i] * xq[i]        (exact int32)
 *     E|O[j] = fma((float)L[j], S, E|O[j])        block parity picks E or O
 *
 * Q6_K (210 B: ql[128], qh[64], int8 sc[16], f16 d at the END). Value
 * q = (lo4 | hi2 << 4) - 32 in [-32, 31], one scale per 16 values. A block is
 * eight 32-value chunks c = 0..7; lanes 0-3 of a chunk are group 2c, lanes 4-7
 * group 2c+1:
 *     S[g] = (d * sc[g]) * xscale[8b + g/2]
 *     L[j] = SUM_{i=4j..4j+3} q[32c + i] * xq[32c + i]
 *     E|O[j] = fma((float)L[j], S[2c + j/4], E|O[j])   chunk parity picks
 *
 * Both formats are signed and VNNI multiplies u8 by s8. Q8_0 makes the
 * ACTIVATION unsigned (xq ^ 0x80 = xq + 128, once per load, shared by four
 * rows) and removes 128 * SUM w with a second vpdpbusd — mynah-asr's
 * correction, exact in int32. Q6_K needs no re-encoding at all: the stored
 * u = q + 32 is already unsigned, and 32 * SUM xq depends only on the
 * activation, so it is one vpdpbusd per activation load for four rows. */

static void q80_i8_rows_scalar(const unsigned char *base, size_t rows, size_t nb,
                               const int8_t *xq, const float *xscale, float *out) {
    for (size_t r = 0; r < rows; r++) {
        const unsigned char *row = base + r * nb * 34;
        float e[8] = {0}, o[8] = {0};
        for (size_t b = 0; b < nb; b++) {
            const unsigned char *blk = row + b * 34;
            const float s = mynah_slm_f16_to_f32(blk) * xscale[b];
            const int8_t *w = (const int8_t *)(blk + 2);
            const int8_t *x = xq + b * 32;
            float *acc = (b & 1) ? o : e;
            for (int j = 0; j < 8; j++) {
                int32_t l = 0;
                for (int i = 4 * j; i < 4 * j + 4; i++) l += (int32_t)w[i] * (int32_t)x[i];
                acc[j] = fmaf((float)l, s, acc[j]);
            }
        }
        float v[8];
        for (int j = 0; j < 8; j++) v[j] = e[j] + o[j];
        out[r] = q4k_tree8(v);
    }
}

/* The 6-bit value u = q + 32 of element i (0..31) of chunk k (0..3) of half h. */
static inline int q6k_u(const unsigned char *blk, int h, int k, int i) {
    const unsigned char *ql = blk + 64 * h, *qh = blk + 128 + 32 * h;
    switch (k) {
    case 0:  return (ql[i] & 15)      | (((qh[i] >> 0) & 3) << 4);
    case 1:  return (ql[32 + i] & 15) | (((qh[i] >> 2) & 3) << 4);
    case 2:  return (ql[i] >> 4)      | (((qh[i] >> 4) & 3) << 4);
    default: return (ql[32 + i] >> 4) | (((qh[i] >> 6) & 3) << 4);
    }
}

static void q6k_i8_rows_scalar(const unsigned char *base, size_t rows, size_t nb,
                               const int8_t *xq, const float *xscale, float *out) {
    for (size_t r = 0; r < rows; r++) {
        const unsigned char *row = base + r * nb * 210;
        float e[8] = {0}, o[8] = {0};
        for (size_t b = 0; b < nb; b++) {
            const unsigned char *blk = row + b * 210;
            const float d = mynah_slm_f16_to_f32(blk + 208);
            const int8_t *sc = (const int8_t *)(blk + 192);
            float s[16];
            for (int g = 0; g < 16; g++) s[g] = (d * (float)sc[g]) * xscale[b * 8 + g / 2];
            for (int c = 0; c < 8; c++) {
                const int8_t *x = xq + b * 256 + c * 32;
                float *acc = (c & 1) ? o : e;
                for (int j = 0; j < 8; j++) {
                    int32_t l = 0;
                    for (int i = 4 * j; i < 4 * j + 4; i++)
                        l += (q6k_u(blk, c / 4, c % 4, i) - 32) * (int32_t)x[i];
                    acc[j] = fmaf((float)l, s[2 * c + j / 4], acc[j]);
                }
            }
        }
        float v[8];
        for (int j = 0; j < 8; j++) v[j] = e[j] + o[j];
        out[r] = q4k_tree8(v);
    }
}

#if defined(MYNAH_SLM_HAVE_SDOT)

/* f16 scale read. VCVTPH2PS where the build has it — exact, like the scalar
 * conversion, so the twin and the kernel agree bit for bit either way. */
static inline float q_f16(const unsigned char *p) {
#if defined(__F16C__)
    return _cvtsh_ss((unsigned short)(p[0] | (p[1] << 8)));
#else
    return mynah_slm_f16_to_f32(p);
#endif
}

#if defined(MYNAH_SLM_HAVE_AVX512VNNI)
/* [a x8 | b x8] with AVX-512F only (no DQ). */
static inline __m512 q_halves(float a, float b) {
    return _mm512_castpd_ps(_mm512_insertf64x4(
        _mm512_castps_pd(_mm512_set1_ps(a)), _mm256_castps_pd(_mm256_set1_ps(b)), 1));
}

Q4K_INLINE void q80_i8_avx512(const unsigned char *row0, size_t row_bytes,
                              const int nr, size_t nb, const int8_t *xq,
                              const float *xscale, float *out) {
    __m512 acc[4];
    for (int r = 0; r < nr; r++) acc[r] = _mm512_setzero_ps();
    const __m512i bias = _mm512_set1_epi8((char)0x80);
    const __m512i zero = _mm512_setzero_si512();
    size_t b = 0;
    for (; b + 2 <= nb; b += 2) {
        const __m512i xu = _mm512_xor_si512(
            _mm512_loadu_si512((const void *)(xq + b * 32)), bias);     /* xq + 128 */
        const __m512 xs = q_halves(xscale[b], xscale[b + 1]);
        for (int r = 0; r < nr; r++) {
            const unsigned char *blk = row0 + (size_t)r * row_bytes + b * 34;
            const __m512i w = _mm512_inserti64x4(
                _mm512_castsi256_si512(_mm256_loadu_si256((const __m256i *)(const void *)(blk + 2))),
                _mm256_loadu_si256((const __m256i *)(const void *)(blk + 36)), 1);
            const __m512i l = _mm512_sub_epi32(_mm512_dpbusd_epi32(zero, xu, w),
                                               _mm512_dpbusd_epi32(zero, bias, w));
            const __m512 s = _mm512_mul_ps(q_halves(q_f16(blk), q_f16(blk + 34)), xs);
            acc[r] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(l), s, acc[r]);
        }
    }
    if (b < nb) {                       /* an odd block count: the last is even, so E */
        const __m256i bias8 = _mm256_set1_epi8((char)0x80);
        const __m256i xu = _mm256_xor_si256(
            _mm256_loadu_si256((const __m256i *)(const void *)(xq + b * 32)), bias8);
        for (int r = 0; r < nr; r++) {
            const unsigned char *blk = row0 + (size_t)r * row_bytes + b * 34;
            const __m256i w = _mm256_loadu_si256((const __m256i *)(const void *)(blk + 2));
            const __m256i l = _mm256_sub_epi32(
                _mm256_dpbusd_epi32(_mm256_setzero_si256(), xu, w),
                _mm256_dpbusd_epi32(_mm256_setzero_si256(), bias8, w));
            const __m256 s = _mm256_set1_ps(q_f16(blk) * xscale[b]);
            /* masked: lanes 8-15 (O) keep their value exactly, -0 included */
            acc[r] = _mm512_mask3_fmadd_ps(_mm512_castps256_ps512(_mm256_cvtepi32_ps(l)),
                                           _mm512_castps256_ps512(s), acc[r], 0x00ff);
        }
    }
    for (int r = 0; r < nr; r++) {
        const __m256 lo = _mm512_castps512_ps256(acc[r]);
        const __m256 hi = _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(acc[r]), 1));
        out[r] = q4k_tree256(_mm256_add_ps(lo, hi));
    }
}

/* Q6_K, one half-block (128 values) per step: the 64 bytes of ql are one
 * zmm, and qh (32 bytes) feeds both 32-value chunks of a zmm at shifts 0/2
 * and 4/6. Two vpdpbusd per half per row; the 32 * SUM xq bias and the
 * activation loads are shared by the four rows. */
Q4K_INLINE void q6k_i8_avx512(const unsigned char *row0, size_t row_bytes,
                              const int nr, size_t nb, const int8_t *xq,
                              const float *xscale, float *out) {
    __m512 acc[4];
    for (int r = 0; r < nr; r++) acc[r] = _mm512_setzero_ps();
    const __m512i zero = _mm512_setzero_si512();
    const __m512i m0f  = _mm512_set1_epi8(0x0f);
    const __m512i m30  = _mm512_set1_epi8(0x30);
    const __m512i c32  = _mm512_set1_epi8(32);
    const __m512i dup  = _mm512_set_epi32(7, 7, 6, 6, 5, 5, 4, 4, 3, 3, 2, 2, 1, 1, 0, 0);
    /* lanes 4k..4k+3 <- group base+k */
    const __m512i sel[4] = {
        _mm512_set_epi32(3, 3, 3, 3, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0, 0, 0),
        _mm512_set_epi32(7, 7, 7, 7, 6, 6, 6, 6, 5, 5, 5, 5, 4, 4, 4, 4),
        _mm512_set_epi32(11, 11, 11, 11, 10, 10, 10, 10, 9, 9, 9, 9, 8, 8, 8, 8),
        _mm512_set_epi32(15, 15, 15, 15, 14, 14, 14, 14, 13, 13, 13, 13, 12, 12, 12, 12),
    };

    for (size_t b = 0; b < nb; b++) {
        const __m512 xs = _mm512_permutexvar_ps(
            dup, _mm512_castps256_ps512(_mm256_loadu_ps(xscale + b * 8)));
        __m512 s[4];
        for (int r = 0; r < nr; r++) {
            const unsigned char *blk = row0 + (size_t)r * row_bytes + b * 210;
            const __m512 sc = _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(
                _mm_loadu_si128((const __m128i *)(const void *)(blk + 192))));
            s[r] = _mm512_mul_ps(_mm512_mul_ps(_mm512_set1_ps(q_f16(blk + 208)), sc), xs);
        }
        for (int h = 0; h < 2; h++) {
            const int8_t *xb = xq + b * 256 + (size_t)h * 128;
            const __m512i xa = _mm512_loadu_si512((const void *)xb);
            const __m512i xc = _mm512_loadu_si512((const void *)(xb + 64));
            const __m512i ca = _mm512_dpbusd_epi32(zero, c32, xa);       /* 32*SUM xq */
            const __m512i cc = _mm512_dpbusd_epi32(zero, c32, xc);
            for (int r = 0; r < nr; r++) {
                const unsigned char *blk = row0 + (size_t)r * row_bytes + b * 210;
                const __m512i ql = _mm512_loadu_si512((const void *)(blk + 64 * h));
                const __m256i qh = _mm256_loadu_si256(
                    (const __m256i *)(const void *)(blk + 128 + 32 * h));
                const __m512i h0 = _mm512_inserti64x4(_mm512_castsi256_si512(qh),
                                                      _mm256_srli_epi16(qh, 2), 1);
                const __m512i ua = _mm512_or_si512(
                    _mm512_and_si512(ql, m0f),
                    _mm512_and_si512(_mm512_slli_epi16(h0, 4), m30));
                const __m512i uc = _mm512_or_si512(
                    _mm512_and_si512(_mm512_srli_epi16(ql, 4), m0f),
                    _mm512_and_si512(h0, m30));
                const __m512i la = _mm512_sub_epi32(_mm512_dpbusd_epi32(zero, ua, xa), ca);
                const __m512i lc = _mm512_sub_epi32(_mm512_dpbusd_epi32(zero, uc, xc), cc);
                acc[r] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(la),
                                         _mm512_permutexvar_ps(sel[2 * h], s[r]), acc[r]);
                acc[r] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(lc),
                                         _mm512_permutexvar_ps(sel[2 * h + 1], s[r]), acc[r]);
            }
        }
    }
    for (int r = 0; r < nr; r++) {
        const __m256 lo = _mm512_castps512_ps256(acc[r]);
        const __m256 hi = _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(acc[r]), 1));
        out[r] = q4k_tree256(_mm256_add_ps(lo, hi));
    }
}
#define Q80_I8_KERNEL q80_i8_avx512
#define Q6K_I8_KERNEL q6k_i8_avx512

#elif defined(__AVX2__)
/* AVX2, no VNNI: llama.cpp's sign trick for a signed x signed int8 dot on
 * maddubs (u8 x s8): |w| is the unsigned operand and sign(xq, w) the signed
 * one. |xq| <= 127 (the quantizer never emits -128), so sign() cannot
 * overflow, and a pair is at most 128*127*2 = 32512: inside int16. */
Q4K_INLINE void q80_i8_avx2(const unsigned char *row0, size_t row_bytes,
                            const int nr, size_t nb, const int8_t *xq,
                            const float *xscale, float *out) {
    __m256 e[4], o[4];
    for (int r = 0; r < nr; r++) e[r] = o[r] = _mm256_setzero_ps();
    const __m256i ones = _mm256_set1_epi16(1);
    for (size_t b = 0; b < nb; b++) {
        const __m256i x = _mm256_loadu_si256((const __m256i *)(const void *)(xq + b * 32));
        const float xs = xscale[b];
        for (int r = 0; r < nr; r++) {
            const unsigned char *blk = row0 + (size_t)r * row_bytes + b * 34;
            const __m256i w = _mm256_loadu_si256((const __m256i *)(const void *)(blk + 2));
            const __m256i l = _mm256_madd_epi16(
                _mm256_maddubs_epi16(_mm256_abs_epi8(w), _mm256_sign_epi8(x, w)), ones);
            const __m256 s = _mm256_set1_ps(q_f16(blk) * xs);
            if (b & 1) o[r] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(l), s, o[r]);
            else       e[r] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(l), s, e[r]);
        }
    }
    for (int r = 0; r < nr; r++) out[r] = q4k_tree256(_mm256_add_ps(e[r], o[r]));
}

Q4K_INLINE void q6k_i8_avx2(const unsigned char *row0, size_t row_bytes,
                            const int nr, size_t nb, const int8_t *xq,
                            const float *xscale, float *out) {
    __m256 e[4], o[4];
    _Alignas(32) float s[4][16];
    for (int r = 0; r < nr; r++) e[r] = o[r] = _mm256_setzero_ps();
    const __m256i ones = _mm256_set1_epi16(1);
    const __m256i m0f  = _mm256_set1_epi8(0x0f);
    const __m256i m30  = _mm256_set1_epi8(0x30);
    const __m256i c32  = _mm256_set1_epi8(32);
    const __m256i dup  = _mm256_set_epi32(3, 3, 2, 2, 1, 1, 0, 0);
    const __m256i sel[4] = {           /* lanes 0-3 <- 2k, 4-7 <- 2k+1 */
        _mm256_set_epi32(1, 1, 1, 1, 0, 0, 0, 0), _mm256_set_epi32(3, 3, 3, 3, 2, 2, 2, 2),
        _mm256_set_epi32(5, 5, 5, 5, 4, 4, 4, 4), _mm256_set_epi32(7, 7, 7, 7, 6, 6, 6, 6),
    };

    for (size_t b = 0; b < nb; b++) {
        const __m256 xs8 = _mm256_loadu_ps(xscale + b * 8);
        const __m256 xs_lo = _mm256_permutevar8x32_ps(xs8, dup);                 /* groups 0-7 */
        const __m256 xs_hi = _mm256_permutevar8x32_ps(
            xs8, _mm256_add_epi32(dup, _mm256_set1_epi32(4)));                     /* groups 8-15 */
        for (int r = 0; r < nr; r++) {
            const unsigned char *blk = row0 + (size_t)r * row_bytes + b * 210;
            const __m256 d = _mm256_set1_ps(q_f16(blk + 208));
            const __m128i sc = _mm_loadu_si128((const __m128i *)(const void *)(blk + 192));
            _mm256_store_ps(s[r], _mm256_mul_ps(_mm256_mul_ps(
                d, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(sc))), xs_lo));
            _mm256_store_ps(s[r] + 8, _mm256_mul_ps(_mm256_mul_ps(
                d, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(sc, 8)))), xs_hi));
        }
        for (int h = 0; h < 2; h++) {
            for (int k = 0; k < 4; k++) {
                const int c = 4 * h + k;
                const __m256i x = _mm256_loadu_si256(
                    (const __m256i *)(const void *)(xq + b * 256 + (size_t)c * 32));
                const __m256i cx = _mm256_madd_epi16(_mm256_maddubs_epi16(c32, x), ones);
                for (int r = 0; r < nr; r++) {
                    const unsigned char *blk = row0 + (size_t)r * row_bytes + b * 210;
                    const __m256i ql = _mm256_loadu_si256(
                        (const __m256i *)(const void *)(blk + 64 * h + 32 * (k & 1)));
                    const __m256i qh = _mm256_loadu_si256(
                        (const __m256i *)(const void *)(blk + 128 + 32 * h));
                    __m256i lo, hi;
                    switch (k) {
                    case 0:  lo = ql;                        hi = _mm256_slli_epi16(qh, 4); break;
                    case 1:  lo = ql;                        hi = _mm256_slli_epi16(qh, 2); break;
                    case 2:  lo = _mm256_srli_epi16(ql, 4);  hi = qh;                       break;
                    default: lo = _mm256_srli_epi16(ql, 4);  hi = _mm256_srli_epi16(qh, 2); break;
                    }
                    const __m256i u = _mm256_or_si256(_mm256_and_si256(lo, m0f),
                                                      _mm256_and_si256(hi, m30));
                    const __m256i l = _mm256_sub_epi32(
                        _mm256_madd_epi16(_mm256_maddubs_epi16(u, x), ones), cx);
                    const __m256 sv = _mm256_permutevar8x32_ps(
                        _mm256_load_ps(s[r] + (c >= 4 ? 8 : 0)), sel[c & 3]);
                    if (c & 1) o[r] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(l), sv, o[r]);
                    else       e[r] = _mm256_fmadd_ps(_mm256_cvtepi32_ps(l), sv, e[r]);
                }
            }
        }
    }
    for (int r = 0; r < nr; r++) out[r] = q4k_tree256(_mm256_add_ps(e[r], o[r]));
}
#define Q80_I8_KERNEL q80_i8_avx2
#define Q6K_I8_KERNEL q6k_i8_avx2

#elif defined(__ARM_NEON)
Q4K_INLINE void q80_i8_neon(const unsigned char *row0, size_t row_bytes,
                            const int nr, size_t nb, const int8_t *xq,
                            const float *xscale, float *out) {
    float32x4_t e0[4], e1[4], o0[4], o1[4];
    for (int r = 0; r < nr; r++) e0[r] = e1[r] = o0[r] = o1[r] = vdupq_n_f32(0.0f);
    const int32x4_t z = vdupq_n_s32(0);
    for (size_t b = 0; b < nb; b++) {
        const int8x16_t x0 = vld1q_s8(xq + b * 32), x1 = vld1q_s8(xq + b * 32 + 16);
        const float xs = xscale[b];
        for (int r = 0; r < nr; r++) {
            const unsigned char *blk = row0 + (size_t)r * row_bytes + b * 34;
            const int8_t *w = (const int8_t *)(blk + 2);
            const float s = q_f16(blk) * xs;
            const float32x4_t l0 = vcvtq_f32_s32(vdotq_s32(z, vld1q_s8(w), x0));
            const float32x4_t l1 = vcvtq_f32_s32(vdotq_s32(z, vld1q_s8(w + 16), x1));
            if (b & 1) { o0[r] = vfmaq_n_f32(o0[r], l0, s); o1[r] = vfmaq_n_f32(o1[r], l1, s); }
            else       { e0[r] = vfmaq_n_f32(e0[r], l0, s); e1[r] = vfmaq_n_f32(e1[r], l1, s); }
        }
    }
    for (int r = 0; r < nr; r++)
        out[r] = q4k_tree_neon(vaddq_f32(e0[r], o0[r]), vaddq_f32(e1[r], o1[r]));
}

Q4K_INLINE void q6k_i8_neon(const unsigned char *row0, size_t row_bytes,
                            const int nr, size_t nb, const int8_t *xq,
                            const float *xscale, float *out) {
    float32x4_t e0[4], e1[4], o0[4], o1[4];
    float s[4][16];
    for (int r = 0; r < nr; r++) e0[r] = e1[r] = o0[r] = o1[r] = vdupq_n_f32(0.0f);
    const int32x4_t z = vdupq_n_s32(0);
    const uint8x16_t m0f = vdupq_n_u8(0x0f), m03 = vdupq_n_u8(0x03);
    const int8x16_t  c32 = vdupq_n_s8(32);

    for (size_t b = 0; b < nb; b++) {
        for (int r = 0; r < nr; r++) {
            const unsigned char *blk = row0 + (size_t)r * row_bytes + b * 210;
            const float d = q_f16(blk + 208);
            const int8_t *sc = (const int8_t *)(blk + 192);
            for (int g = 0; g < 16; g++) s[r][g] = (d * (float)sc[g]) * xscale[b * 8 + g / 2];
        }
        for (int c = 0; c < 8; c++) {
            const int h = c / 4, k = c % 4;
            const int8x16_t x0 = vld1q_s8(xq + b * 256 + c * 32);
            const int8x16_t x1 = vld1q_s8(xq + b * 256 + c * 32 + 16);
            for (int r = 0; r < nr; r++) {
                const unsigned char *blk = row0 + (size_t)r * row_bytes + b * 210;
                const unsigned char *ql = blk + 64 * h + 32 * (k & 1);
                const unsigned char *qh = blk + 128 + 32 * h;
                const uint8x16_t l0 = vld1q_u8(ql), l1 = vld1q_u8(ql + 16);
                const uint8x16_t h0 = vld1q_u8(qh), h1 = vld1q_u8(qh + 16);
                uint8x16_t u0, u1;
                switch (k) {
                case 0:
                    u0 = vorrq_u8(vandq_u8(l0, m0f), vshlq_n_u8(vandq_u8(h0, m03), 4));
                    u1 = vorrq_u8(vandq_u8(l1, m0f), vshlq_n_u8(vandq_u8(h1, m03), 4));
                    break;
                case 1:
                    u0 = vorrq_u8(vandq_u8(l0, m0f), vshlq_n_u8(vandq_u8(vshrq_n_u8(h0, 2), m03), 4));
                    u1 = vorrq_u8(vandq_u8(l1, m0f), vshlq_n_u8(vandq_u8(vshrq_n_u8(h1, 2), m03), 4));
                    break;
                case 2:
                    u0 = vorrq_u8(vshrq_n_u8(l0, 4), vshlq_n_u8(vandq_u8(vshrq_n_u8(h0, 4), m03), 4));
                    u1 = vorrq_u8(vshrq_n_u8(l1, 4), vshlq_n_u8(vandq_u8(vshrq_n_u8(h1, 4), m03), 4));
                    break;
                default:
                    u0 = vorrq_u8(vshrq_n_u8(l0, 4), vshlq_n_u8(vshrq_n_u8(h0, 6), 4));
                    u1 = vorrq_u8(vshrq_n_u8(l1, 4), vshlq_n_u8(vshrq_n_u8(h1, 6), 4));
                    break;
                }
                const int8x16_t q0 = vsubq_s8(vreinterpretq_s8_u8(u0), c32);
                const int8x16_t q1 = vsubq_s8(vreinterpretq_s8_u8(u1), c32);
                const float32x4_t a = vcvtq_f32_s32(vdotq_s32(z, q0, x0));
                const float32x4_t bq = vcvtq_f32_s32(vdotq_s32(z, q1, x1));
                const float sa = s[r][2 * c], sb = s[r][2 * c + 1];
                if (c & 1) { o0[r] = vfmaq_n_f32(o0[r], a, sa); o1[r] = vfmaq_n_f32(o1[r], bq, sb); }
                else       { e0[r] = vfmaq_n_f32(e0[r], a, sa); e1[r] = vfmaq_n_f32(e1[r], bq, sb); }
            }
        }
    }
    for (int r = 0; r < nr; r++)
        out[r] = q4k_tree_neon(vaddq_f32(e0[r], o0[r]), vaddq_f32(e1[r], o1[r]));
}
#define Q80_I8_KERNEL q80_i8_neon
#define Q6K_I8_KERNEL q6k_i8_neon
#endif

/* Four rows per activation load, then the tail; see q4k_i8_rows. */
static void q80_i8_rows(const unsigned char *base, size_t rows, size_t nb,
                        const mynah_slm_matvec_in *p, float *out) {
    const size_t rb = nb * 34;
    size_t r = 0;
    for (; r + 4 <= rows; r += 4) Q80_I8_KERNEL(base + r * rb, rb, 4, nb, p->xq, p->xscale, out + r);
    for (; r < rows; r++)         Q80_I8_KERNEL(base + r * rb, rb, 1, nb, p->xq, p->xscale, out + r);
}

static void q6k_i8_rows(const unsigned char *base, size_t rows, size_t nb,
                        const mynah_slm_matvec_in *p, float *out) {
    const size_t rb = nb * 210;
    size_t r = 0;
    for (; r + 4 <= rows; r += 4) Q6K_I8_KERNEL(base + r * rb, rb, 4, nb, p->xq, p->xscale, out + r);
    for (; r < rows; r++)         Q6K_I8_KERNEL(base + r * rb, rb, 1, nb, p->xq, p->xscale, out + r);
}
#endif /* MYNAH_SLM_HAVE_SDOT */

int mynah_slm_q80_int8_ref(const void *weights, size_t rows, size_t cols,
                           const mynah_slm_matvec_in *prep, float *output) {
    if (!weights || !prep || !output || !prep->have_int8 || cols % 32 != 0)
        return -1;
    q80_i8_rows_scalar((const unsigned char *)weights, rows, cols / 32, prep->xq,
                       prep->xscale, output);
    return 0;
}

int mynah_slm_q6k_int8_ref(const void *weights, size_t rows, size_t cols,
                           const mynah_slm_matvec_in *prep, float *output) {
    if (!weights || !prep || !output || !prep->have_int8 || cols % 256 != 0)
        return -1;
    q6k_i8_rows_scalar((const unsigned char *)weights, rows, cols / 256, prep->xq,
                       prep->xscale, output);
    return 0;
}


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

int mynah_slm_matvec(int type, const void *weights, size_t rows, size_t cols,
                     const float *input, const mynah_slm_matvec_in *prep,
                     float *output) {
    if (!prep || !use_own_kernels()) return -1;
    const int int8 = prep->have_int8 && int8_type_on(type);

#if defined(MYNAH_SLM_HAVE_SDOT)
    /* int8 only: with f32 activations these two types are ingot's. */
    if (type == INGOT_TYPE_Q8_0) {
        if (!int8 || cols % 32 != 0) return -1;
        q80_i8_rows((const unsigned char *)weights, rows, cols / 32, prep, output);
        return 0;
    }
    if (type == INGOT_TYPE_Q6_K) {
        if (!int8 || cols % 256 != 0) return -1;
        q6k_i8_rows((const unsigned char *)weights, rows, cols / 256, prep, output);
        return 0;
    }
#endif
    if (cols % 256 != 0) return -1;         /* K-quant super-blocks, by definition */

    if (type != INGOT_TYPE_Q4_K) return -1;

    const size_t blocks = cols / 256;
    const unsigned char *base = (const unsigned char *)weights;

#if defined(MYNAH_SLM_HAVE_SDOT)
    if (int8) {
        q4k_i8_rows(base, rows, blocks, prep, output);
        return 0;
    }
#else
    (void)int8;
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

        /* Called from ONE thread with the pool idle, never from inside a
         * parallel region: ours runs its own region on the pool, and a vendor
         * BLAS brings its own threads — two pools over the same cores is the
         * throughput collapse mynah-asr measured. NT: both operands are
         * contiguous along cols, so nothing is transposed or packed. */
        if (mynah_slm_sgemm(1, tokens, n_rows, cols, 1.0f, in, cols,
                            scratch, cols, 0.0f, out + row0, rows) != 0)
            return -1;
    }
    return 0;
}
