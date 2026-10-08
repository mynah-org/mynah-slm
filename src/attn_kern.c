/* attn_kern.c — the attention inner loops, compiled once per ISA (kern.h).
 *
 * Moved verbatim from kernels.c (dot/axpy over f32 K/V and the per-head loop)
 * and kvcache.c (the fused bf16/q8/q4 KV accessors: the inner loop of the
 * DEFAULT bf16 cache). Dispatch happens once per head, so the per-position
 * calls below stay direct, inlinable calls inside one TU — before, every
 * position was an out-of-line call from kernels.c into kvcache.c.
 *
 * SPDX-License-Identifier: MIT */
#include "kern.h"
#include "kernels.h"
#include "kvcache.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if !defined(MYNAH_SLM_KERN_ID)
#error "attn_kern.c is a kernel TU: build it through the Makefile's kernel rules"
#endif

#if defined(MYNAH_SLM_K_NEON)
#include <arm_neon.h>
#endif
#if defined(MYNAH_SLM_K_AVX2)
#include <immintrin.h>
#endif

/* Packed KV layout: kvcache.c owns it, these must agree with it. */
#define KV_BLOCK   32
#define KV_Q8_BYTES 34   /* f16 scale + 32 int8  */
#define KV_Q4_BYTES 18   /* f16 scale + 32 nibbles */

static const uint8_t *slot(const uint8_t *base, size_t pos_bytes, uint32_t n_ctx,
                           uint32_t layer, uint32_t pos) {
    return base + ((size_t)layer * n_ctx + pos) * pos_bytes;
}

/* ── half precision, for the block scales ──────────────────────────────────
 * Read and written as bytes: a block sits at an arbitrary offset inside the
 * cache, so a native 2-byte load is not guaranteed to be aligned. */
static float f16_load(const uint8_t *p) {
    const uint16_t h = (uint16_t)(p[0] | ((uint16_t)p[1] << 8));
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    const uint32_t exp  = (h >> 10) & 0x1fu;
    const uint32_t mant = h & 0x3ffu;
    uint32_t bits;

    if (exp == 0) {
        if (mant == 0) bits = sign;
        else {
            uint32_t e = 0, m = mant;
            while ((m & 0x400u) == 0) { m <<= 1; e++; }
            bits = sign | ((127u - 15u - e + 1u) << 23) | ((m & 0x3ffu) << 13);
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

/* dot and scaled-accumulate over head_dim, the two inner loops of attention.
 *
 * Worth vectorizing and nothing else is, which took a measurement to learn:
 * at n_kv = 32 the whole non-matvec half of a decode step is 1.6 ms against
 * ~120 ms of matvec, i.e. 1.4%, and RMSNorm/RoPE/SwiGLU are a rounding error
 * inside that. Attention is the only one whose cost grows with the context,
 * and it grows fast — per token, across 28 layers:
 *
 *   n_kv    32 ->   1.4 ms      n_kv   512 ->  20 ms
 *   n_kv   128 ->   5.1 ms      n_kv  2048 -> 106 ms
 *
 * A 37 ms decode step is 4% attention at n_kv 32 and 287% at n_kv 2048. Every
 * benchmark in docs/perf.md so far used a 19-token prompt, which is precisely
 * where this does not show. Summarizing a meeting transcript is not. */
static inline float dot_f32(const float *a, const float *b, uint32_t n) {
#if defined(MYNAH_SLM_K_NEON)
    float32x4_t s0 = vdupq_n_f32(0.0f), s1 = vdupq_n_f32(0.0f);
    float32x4_t s2 = vdupq_n_f32(0.0f), s3 = vdupq_n_f32(0.0f);
    uint32_t i = 0;
    for (; i + 16 <= n; i += 16) {
        s0 = vfmaq_f32(s0, vld1q_f32(a + i),      vld1q_f32(b + i));
        s1 = vfmaq_f32(s1, vld1q_f32(a + i + 4),  vld1q_f32(b + i + 4));
        s2 = vfmaq_f32(s2, vld1q_f32(a + i + 8),  vld1q_f32(b + i + 8));
        s3 = vfmaq_f32(s3, vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
    }
    float sum = vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)));
    for (; i < n; i++) sum += a[i] * b[i];
    return sum;
#elif defined(MYNAH_SLM_K_AVX2)
    __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
    uint32_t i = 0;
    for (; i + 16 <= n; i += 16) {
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0);
        s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), s1);
    }
    const __m256 t = _mm256_add_ps(s0, s1);
    __m128 v = _mm_add_ps(_mm256_castps256_ps128(t), _mm256_extractf128_ps(t, 1));
    v = _mm_hadd_ps(v, v);
    v = _mm_hadd_ps(v, v);
    float sum = _mm_cvtss_f32(v);
    for (; i < n; i++) sum += a[i] * b[i];
    return sum;
#else
    /* The scalar reference keeps a double accumulator; the vector paths above
     * use four (NEON) or two (AVX2) f32 lanes, which is pairwise summation and
     * so no worse in practice — the parity gate agrees, and it is the gate
     * that decides, not the argument. */
    double sum = 0.0;
    for (uint32_t i = 0; i < n; i++) sum += (double)a[i] * (double)b[i];
    return (float)sum;
#endif
}

/* y[i] += w * x[i] */
static inline void axpy_f32(float *y, const float *x, float w, uint32_t n) {
#if defined(MYNAH_SLM_K_NEON)
    const float32x4_t vw = vdupq_n_f32(w);
    uint32_t i = 0;
    for (; i + 16 <= n; i += 16) {
        vst1q_f32(y + i,      vfmaq_f32(vld1q_f32(y + i),      vld1q_f32(x + i),      vw));
        vst1q_f32(y + i + 4,  vfmaq_f32(vld1q_f32(y + i + 4),  vld1q_f32(x + i + 4),  vw));
        vst1q_f32(y + i + 8,  vfmaq_f32(vld1q_f32(y + i + 8),  vld1q_f32(x + i + 8),  vw));
        vst1q_f32(y + i + 12, vfmaq_f32(vld1q_f32(y + i + 12), vld1q_f32(x + i + 12), vw));
    }
    for (; i < n; i++) y[i] += w * x[i];
#elif defined(MYNAH_SLM_K_AVX2)
    const __m256 vw = _mm256_set1_ps(w);
    uint32_t i = 0;
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(y + i, _mm256_fmadd_ps(_mm256_loadu_ps(x + i), vw,
                                                _mm256_loadu_ps(y + i)));
    for (; i < n; i++) y[i] += w * x[i];
#else
    for (uint32_t i = 0; i < n; i++) y[i] += w * x[i];
#endif
}

static void f32_head(float *out, const float *q, const float *k, const float *v,
                           uint32_t h, uint32_t n_kv, uint32_t n_heads,
                           uint32_t n_kv_heads, uint32_t head_dim, float scale,
                           float *scratch) {
    const uint32_t group  = n_heads / n_kv_heads;
    const uint32_t kv_dim = n_kv_heads * head_dim;

    const float   *qh  = q + (size_t)h * head_dim;
    const uint32_t kvh = h / group;

    for (uint32_t t = 0; t < n_kv; t++)
        scratch[t] = dot_f32(qh, k + (size_t)t * kv_dim + (size_t)kvh * head_dim,
                             head_dim) * scale;
    mynah_slm_softmax(scratch, n_kv);

    float *oh = out + (size_t)h * head_dim;
    memset(oh, 0, head_dim * sizeof *oh);
    for (uint32_t t = 0; t < n_kv; t++)
        axpy_f32(oh, v + (size_t)t * kv_dim + (size_t)kvh * head_dim,
                 scratch[t], head_dim);
}

/* ── the fused decode accessors ────────────────────────────────────────────
 * One block at a time, vectorized. Scalar versions of these MEASURED SLOWER
 * than the f32 cache they were meant to beat — 11.5 tok/s against 14.5 — even
 * though they read a quarter of the bytes: the f32 path dots with 16-wide NEON
 * and the saving in traffic does not pay for giving that up. Compression only
 * becomes speed once the decode is vectorized too. */

#if defined(MYNAH_SLM_K_NEON)

/* bf16 widens to f32 by moving its 16 bits into the high half — no table, no
 * rounding, no scale. Fused for the same reason as the block formats: the
 * decode-into-scratch fallback measured 10.3 tok/s against f32's 19.1, which
 * is the cost of the scratch and not a property of bf16. */
static inline float32x4_t bf16_widen(uint16x4_t v) {
    return vreinterpretq_f32_u32(vshll_n_u16(v, 16));
}

static inline float dot_bf16(const uint8_t *p, const float *x, uint32_t n) {
    const uint16_t *b = (const uint16_t *)(const void *)p;
    float32x4_t a0 = vdupq_n_f32(0.0f), a1 = vdupq_n_f32(0.0f);
    uint32_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const uint16x8_t w = vld1q_u16(b + i);
        a0 = vfmaq_f32(a0, bf16_widen(vget_low_u16(w)),  vld1q_f32(x + i));
        a1 = vfmaq_f32(a1, bf16_widen(vget_high_u16(w)), vld1q_f32(x + i + 4));
    }
    float sum = vaddvq_f32(vaddq_f32(a0, a1));
    for (; i < n; i++) {
        const uint32_t bits = (uint32_t)b[i] << 16;
        float f; memcpy(&f, &bits, sizeof f);
        sum += f * x[i];
    }
    return sum;
}

static inline void axpy_bf16(float *o, const uint8_t *p, float w, uint32_t n) {
    const uint16_t *b = (const uint16_t *)(const void *)p;
    uint32_t i = 0;
    for (; i + 8 <= n; i += 8) {
        const uint16x8_t v = vld1q_u16(b + i);
        vst1q_f32(o + i,     vfmaq_n_f32(vld1q_f32(o + i),     bf16_widen(vget_low_u16(v)),  w));
        vst1q_f32(o + i + 4, vfmaq_n_f32(vld1q_f32(o + i + 4), bf16_widen(vget_high_u16(v)), w));
    }
    for (; i < n; i++) {
        const uint32_t bits = (uint32_t)b[i] << 16;
        float f; memcpy(&f, &bits, sizeof f);
        o[i] += w * f;
    }
}

/* 32 int8 against 32 floats, without materializing the dequantized values. */
static inline float dot_q8_block(const int8_t *q, const float *x) {
    const int8x16_t lo = vld1q_s8(q), hi = vld1q_s8(q + 16);
    const int16x8_t l0 = vmovl_s8(vget_low_s8(lo)),  l1 = vmovl_s8(vget_high_s8(lo));
    const int16x8_t h0 = vmovl_s8(vget_low_s8(hi)),  h1 = vmovl_s8(vget_high_s8(hi));

    float32x4_t a = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(l0))),  vld1q_f32(x));
    a = vfmaq_f32(a, vcvtq_f32_s32(vmovl_s16(vget_high_s16(l0))), vld1q_f32(x + 4));
    a = vfmaq_f32(a, vcvtq_f32_s32(vmovl_s16(vget_low_s16(l1))),  vld1q_f32(x + 8));
    a = vfmaq_f32(a, vcvtq_f32_s32(vmovl_s16(vget_high_s16(l1))), vld1q_f32(x + 12));
    a = vfmaq_f32(a, vcvtq_f32_s32(vmovl_s16(vget_low_s16(h0))),  vld1q_f32(x + 16));
    a = vfmaq_f32(a, vcvtq_f32_s32(vmovl_s16(vget_high_s16(h0))), vld1q_f32(x + 20));
    a = vfmaq_f32(a, vcvtq_f32_s32(vmovl_s16(vget_low_s16(h1))),  vld1q_f32(x + 24));
    a = vfmaq_f32(a, vcvtq_f32_s32(vmovl_s16(vget_high_s16(h1))), vld1q_f32(x + 28));
    return vaddvq_f32(a);
}

static inline void axpy_q8_block(float *o, const int8_t *q, float w) {
    const int8x16_t lo = vld1q_s8(q), hi = vld1q_s8(q + 16);
    const int16x8_t v[4] = { vmovl_s8(vget_low_s8(lo)), vmovl_s8(vget_high_s8(lo)),
                             vmovl_s8(vget_low_s8(hi)), vmovl_s8(vget_high_s8(hi)) };
    for (int i = 0; i < 4; i++) {
        vst1q_f32(o + i * 8,
                  vfmaq_n_f32(vld1q_f32(o + i * 8),
                              vcvtq_f32_s32(vmovl_s16(vget_low_s16(v[i]))), w));
        vst1q_f32(o + i * 8 + 4,
                  vfmaq_n_f32(vld1q_f32(o + i * 8 + 4),
                              vcvtq_f32_s32(vmovl_s16(vget_high_s16(v[i]))), w));
    }
}

/* 16 bytes of nibbles: the low half is elements 0..15, the high half 16..31 —
 * llama.cpp's q4_0 layout, biased by 8. */
static inline void q4_split(const uint8_t *q, int16x8_t *out) {
    const uint8x16_t packed = vld1q_u8(q);
    const int8x16_t  bias   = vdupq_n_s8(8);
    const int8x16_t  lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(packed, vdupq_n_u8(0x0f))), bias);
    const int8x16_t  hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(packed, 4)), bias);
    out[0] = vmovl_s8(vget_low_s8(lo));  out[1] = vmovl_s8(vget_high_s8(lo));
    out[2] = vmovl_s8(vget_low_s8(hi));  out[3] = vmovl_s8(vget_high_s8(hi));
}

static inline float dot_q4_block(const uint8_t *q, const float *x) {
    int16x8_t v[4];
    q4_split(q, v);
    float32x4_t a = vdupq_n_f32(0.0f);
    for (int i = 0; i < 4; i++) {
        a = vfmaq_f32(a, vcvtq_f32_s32(vmovl_s16(vget_low_s16(v[i]))),  vld1q_f32(x + i * 8));
        a = vfmaq_f32(a, vcvtq_f32_s32(vmovl_s16(vget_high_s16(v[i]))), vld1q_f32(x + i * 8 + 4));
    }
    return vaddvq_f32(a);
}

static inline void axpy_q4_block(float *o, const uint8_t *q, float w) {
    int16x8_t v[4];
    q4_split(q, v);
    for (int i = 0; i < 4; i++) {
        vst1q_f32(o + i * 8,
                  vfmaq_n_f32(vld1q_f32(o + i * 8),
                              vcvtq_f32_s32(vmovl_s16(vget_low_s16(v[i]))), w));
        vst1q_f32(o + i * 8 + 4,
                  vfmaq_n_f32(vld1q_f32(o + i * 8 + 4),
                              vcvtq_f32_s32(vmovl_s16(vget_high_s16(v[i]))), w));
    }
}

#elif defined(MYNAH_SLM_K_AVX2)

static inline float hsum_avx(__m256 v) {
    __m128 a = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    a = _mm_add_ps(a, _mm_movehl_ps(a, a));
    a = _mm_add_ss(a, _mm_shuffle_ps(a, a, 0x55));
    return _mm_cvtss_f32(a);
}

/* bf16 widens by moving its 16 bits into the high half of an f32 — one shift,
 * which is why it stays the fastest format here. */
static inline __m256 bf16_widen8(const uint16_t *b) {
    return _mm256_castsi256_ps(
        _mm256_slli_epi32(_mm256_cvtepu16_epi32(
            _mm_loadu_si128((const __m128i *)(const void *)b)), 16));
}

static inline float dot_bf16(const uint8_t *p, const float *x, uint32_t n) {
    const uint16_t *b = (const uint16_t *)(const void *)p;
    __m256 acc = _mm256_setzero_ps();
    uint32_t i = 0;
    for (; i + 8 <= n; i += 8)
        acc = _mm256_fmadd_ps(bf16_widen8(b + i), _mm256_loadu_ps(x + i), acc);
    float sum = hsum_avx(acc);
    for (; i < n; i++) {
        const uint32_t bits = (uint32_t)b[i] << 16;
        float f; memcpy(&f, &bits, sizeof f);
        sum += f * x[i];
    }
    return sum;
}

static inline void axpy_bf16(float *o, const uint8_t *p, float w, uint32_t n) {
    const uint16_t *b = (const uint16_t *)(const void *)p;
    const __m256 vw = _mm256_set1_ps(w);
    uint32_t i = 0;
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(o + i, _mm256_fmadd_ps(bf16_widen8(b + i), vw,
                                                _mm256_loadu_ps(o + i)));
    for (; i < n; i++) {
        const uint32_t bits = (uint32_t)b[i] << 16;
        float f; memcpy(&f, &bits, sizeof f);
        o[i] += w * f;
    }
}

static inline float dot_q8_block(const int8_t *q, const float *x) {
    __m256 acc = _mm256_setzero_ps();
    for (int i = 0; i < KV_BLOCK; i += 8) {
        const __m256i v = _mm256_cvtepi8_epi32(
            _mm_loadl_epi64((const __m128i *)(const void *)(q + i)));
        acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(v), _mm256_loadu_ps(x + i), acc);
    }
    return hsum_avx(acc);
}

static inline void axpy_q8_block(float *o, const int8_t *q, float w) {
    const __m256 vw = _mm256_set1_ps(w);
    for (int i = 0; i < KV_BLOCK; i += 8) {
        const __m256i v = _mm256_cvtepi8_epi32(
            _mm_loadl_epi64((const __m128i *)(const void *)(q + i)));
        _mm256_storeu_ps(o + i, _mm256_fmadd_ps(_mm256_cvtepi32_ps(v), vw,
                                                _mm256_loadu_ps(o + i)));
    }
}

/* The low nibbles are elements 0..15 and the high ones 16..31, biased by 8 —
 * llama.cpp's q4_0 layout. */
static inline void q4_expand(const uint8_t *q, float *out) {
    const __m128i packed = _mm_loadu_si128((const __m128i *)(const void *)q);
    const __m128i lo = _mm_and_si128(packed, _mm_set1_epi8(0x0f));
    const __m128i hi = _mm_and_si128(_mm_srli_epi16(packed, 4), _mm_set1_epi8(0x0f));
    const __m256i bias = _mm256_set1_epi32(8);

    /* Unrolled rather than looped: the byte shift is an immediate, so the
     * shift amount has to be a compile-time constant. */
#define KV_Q4_EXPAND(dst, src, sh)                                            \
    _mm256_storeu_ps((dst), _mm256_cvtepi32_ps(_mm256_sub_epi32(              \
        _mm256_cvtepu8_epi32(_mm_srli_si128((src), (sh))), bias)))

    KV_Q4_EXPAND(out + 0,  lo, 0);
    KV_Q4_EXPAND(out + 8,  lo, 8);
    KV_Q4_EXPAND(out + 16, hi, 0);
    KV_Q4_EXPAND(out + 24, hi, 8);
#undef KV_Q4_EXPAND
}

static inline float dot_q4_block(const uint8_t *q, const float *x) {
    float v[KV_BLOCK];
    q4_expand(q, v);
    __m256 acc = _mm256_setzero_ps();
    for (int i = 0; i < KV_BLOCK; i += 8)
        acc = _mm256_fmadd_ps(_mm256_loadu_ps(v + i), _mm256_loadu_ps(x + i), acc);
    return hsum_avx(acc);
}

static inline void axpy_q4_block(float *o, const uint8_t *q, float w) {
    float v[KV_BLOCK];
    q4_expand(q, v);
    const __m256 vw = _mm256_set1_ps(w);
    for (int i = 0; i < KV_BLOCK; i += 8)
        _mm256_storeu_ps(o + i, _mm256_fmadd_ps(_mm256_loadu_ps(v + i), vw,
                                                _mm256_loadu_ps(o + i)));
}

#else   /* the scalar twins, and the definition of what the other two compute */

static inline float dot_bf16(const uint8_t *p, const float *x, uint32_t n) {
    float sum = 0.0f;
    for (uint32_t i = 0; i < n; i++) {
        const uint32_t bits = ((uint32_t)p[2 * i + 1] << 24) | ((uint32_t)p[2 * i] << 16);
        float f; memcpy(&f, &bits, sizeof f);
        sum += f * x[i];
    }
    return sum;
}
static inline void axpy_bf16(float *o, const uint8_t *p, float w, uint32_t n) {
    for (uint32_t i = 0; i < n; i++) {
        const uint32_t bits = ((uint32_t)p[2 * i + 1] << 24) | ((uint32_t)p[2 * i] << 16);
        float f; memcpy(&f, &bits, sizeof f);
        o[i] += w * f;
    }
}

static inline float dot_q8_block(const int8_t *q, const float *x) {
    float a = 0.0f;
    for (int i = 0; i < KV_BLOCK; i++) a += (float)q[i] * x[i];
    return a;
}
static inline void axpy_q8_block(float *o, const int8_t *q, float w) {
    for (int i = 0; i < KV_BLOCK; i++) o[i] += w * (float)q[i];
}
static inline float dot_q4_block(const uint8_t *q, const float *x) {
    float a = 0.0f;
    for (int i = 0; i < KV_BLOCK / 2; i++) {
        a += (float)((int)(q[i] & 0x0fu) - 8) * x[i];
        a += (float)((int)(q[i] >> 4) - 8)    * x[i + KV_BLOCK / 2];
    }
    return a;
}
static inline void axpy_q4_block(float *o, const uint8_t *q, float w) {
    for (int i = 0; i < KV_BLOCK / 2; i++) {
        o[i]                += w * (float)((int)(q[i] & 0x0fu) - 8);
        o[i + KV_BLOCK / 2] += w * (float)((int)(q[i] >> 4) - 8);
    }
}

#endif

static float kv_dot(const mynah_slm_kv *c, uint32_t layer, uint32_t pos,
                         uint32_t head, const float *q) {
    const uint8_t *row = slot(c->k, c->pos_bytes_k, c->n_ctx, layer, pos);
    const uint32_t hd = c->head_dim;

    if (c->type_k == MYNAH_SLM_KV_F32) {
        const float *k = (const float *)row + (size_t)head * hd;
        float acc = 0.0f;
        for (uint32_t i = 0; i < hd; i++) acc += q[i] * k[i];
        return acc;
    }
    if (c->type_k == MYNAH_SLM_KV_BF16)
        return dot_bf16(row + (size_t)head * hd * 2, q, hd);
    if (c->type_k == MYNAH_SLM_KV_Q8) {
        const uint8_t *p = row + (size_t)head * (hd / KV_BLOCK) * KV_Q8_BYTES;
        float acc = 0.0f;
        for (uint32_t b = 0; b < hd / KV_BLOCK; b++) {
            const uint8_t *blk = p + b * KV_Q8_BYTES;
            /* The scale is applied ONCE per block, not per element — the same
             * distribute-the-scale move as the Q4_K weight kernel. */
            acc += dot_q8_block((const int8_t *)(blk + 2), q + b * KV_BLOCK) *
                   f16_load(blk);
        }
        return acc;
    }
    if (c->type_k == MYNAH_SLM_KV_Q4) {
        const uint8_t *p = row + (size_t)head * (hd / KV_BLOCK) * KV_Q4_BYTES;
        float acc = 0.0f;
        for (uint32_t b = 0; b < hd / KV_BLOCK; b++) {
            const uint8_t *blk = p + b * KV_Q4_BYTES;
            acc += dot_q4_block(blk + 2, q + b * KV_BLOCK) * f16_load(blk);
        }
        return acc;
    }

    /* fp8 only: decode a slice and dot it. It is dominated by q8 on every
     * axis measured — smaller, more accurate, faster — so it stays on the slow
     * path as the thing q8 is compared against, not as a candidate. */
    float tmp[512];
    if (hd <= 512) {
        mynah_slm_kv_row_slice(row, c->type_k, head, hd, tmp);
        float acc = 0.0f;
        for (uint32_t i = 0; i < hd; i++) acc += q[i] * tmp[i];
        return acc;
    }
    return 0.0f;
}

static void kv_axpy(const mynah_slm_kv *c, uint32_t layer, uint32_t pos,
                         uint32_t head, float w, float *out) {
    const uint8_t *row = slot(c->v, c->pos_bytes_v, c->n_ctx, layer, pos);
    const uint32_t hd = c->head_dim;

    if (c->type_v == MYNAH_SLM_KV_F32) {
        const float *v = (const float *)row + (size_t)head * hd;
        for (uint32_t i = 0; i < hd; i++) out[i] += w * v[i];
        return;
    }
    if (c->type_v == MYNAH_SLM_KV_BF16) {
        axpy_bf16(out, row + (size_t)head * hd * 2, w, hd);
        return;
    }
    if (c->type_v == MYNAH_SLM_KV_Q8) {
        const uint8_t *p = row + (size_t)head * (hd / KV_BLOCK) * KV_Q8_BYTES;
        for (uint32_t b = 0; b < hd / KV_BLOCK; b++) {
            const uint8_t *blk = p + b * KV_Q8_BYTES;
            axpy_q8_block(out + b * KV_BLOCK, (const int8_t *)(blk + 2),
                          w * f16_load(blk));
        }
        return;
    }
    if (c->type_v == MYNAH_SLM_KV_Q4) {
        const uint8_t *p = row + (size_t)head * (hd / KV_BLOCK) * KV_Q4_BYTES;
        for (uint32_t b = 0; b < hd / KV_BLOCK; b++) {
            const uint8_t *blk = p + b * KV_Q4_BYTES;
            axpy_q4_block(out + b * KV_BLOCK, blk + 2, w * f16_load(blk));
        }
        return;
    }

    float tmp[512];
    if (hd <= 512) {
        mynah_slm_kv_row_slice(row, c->type_v, head, hd, tmp);
        for (uint32_t i = 0; i < hd; i++) out[i] += w * tmp[i];
    }
}


/* One head over a packed cache — the body of the old attn_kv_task. */
static void kv_head(float *oh, const float *qh, const mynah_slm_kv *c,
                    uint32_t layer, uint32_t kvh, uint32_t n_kv, float scale,
                    float *scores) {
    for (uint32_t t = 0; t < n_kv; t++)
        scores[t] = kv_dot(c, layer, t, kvh, qh) * scale;
    mynah_slm_softmax(scores, n_kv);

    memset(oh, 0, c->head_dim * sizeof *oh);
    for (uint32_t t = 0; t < n_kv; t++)
        kv_axpy(c, layer, t, kvh, scores[t], oh);
}

const mynah_slm_attn_kern MYNAH_SLM_KERN_SYM(mynah_slm_attn_kern) = {
    MYNAH_SLM_KERN_NAME, MYNAH_SLM_KERN_ID, f32_head, kv_head, kv_dot, kv_axpy,
};
