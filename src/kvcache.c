/* kvcache.c — see kvcache.h.
 * SPDX-License-Identifier: MIT */
#include "kvcache.h"

#include "kernels.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif

#define KV_BLOCK   32
#define KV_Q8_BYTES 34   /* f16 scale + 32 int8  */
#define KV_Q4_BYTES 18   /* f16 scale + 32 nibbles */

int mynah_slm_kv_type_parse(const char *name, mynah_slm_kv_type *out) {
    if (!name || !out) return -1;
    if (!strcmp(name, "f32"))  { *out = MYNAH_SLM_KV_F32;  return 0; }
    if (!strcmp(name, "bf16")) { *out = MYNAH_SLM_KV_BF16; return 0; }
    if (!strcmp(name, "fp8"))  { *out = MYNAH_SLM_KV_FP8;  return 0; }
    if (!strcmp(name, "q8"))   { *out = MYNAH_SLM_KV_Q8;   return 0; }
    if (!strcmp(name, "q4"))   { *out = MYNAH_SLM_KV_Q4;   return 0; }
    return -1;
}

const char *mynah_slm_kv_type_name(mynah_slm_kv_type t) {
    switch (t) {
        case MYNAH_SLM_KV_F32:  return "f32";
        case MYNAH_SLM_KV_BF16: return "bf16";
        case MYNAH_SLM_KV_FP8:  return "fp8";
        case MYNAH_SLM_KV_Q8:   return "q8";
        case MYNAH_SLM_KV_Q4:   return "q4";
    }
    return "f32";
}

double mynah_slm_kv_bits(mynah_slm_kv_type t) {
    switch (t) {
        case MYNAH_SLM_KV_F32:  return 32.0;
        case MYNAH_SLM_KV_BF16: return 16.0;
        case MYNAH_SLM_KV_FP8:  return 8.0;
        /* One f32 scale per 32 values is another whole bit each. */
        case MYNAH_SLM_KV_Q8:   return 8.0 + 32.0 / KV_BLOCK;
        case MYNAH_SLM_KV_Q4:   return 4.0 + 32.0 / KV_BLOCK;
    }
    return 32.0;
}

/* ── bf16 ───────────────────────────────────────────────────────────────────
 * Truncating would bias every value toward zero, and a bias that survives 28
 * layers is not the same thing as noise. Round to nearest even instead. */
static float bf16_round(float v) {
    uint32_t bits;
    memcpy(&bits, &v, sizeof bits);
    const uint32_t lsb = (bits >> 16) & 1u;
    bits += 0x7fffu + lsb;
    bits &= 0xffff0000u;
    memcpy(&v, &bits, sizeof v);
    return v;
}

/* ── fp8 e4m3 ───────────────────────────────────────────────────────────────
 * The format Hopper and the ML tooling settled on: sign, 4 exponent bits with
 * bias 7, 3 mantissa bits. Max finite 448, smallest normal 2^-6, subnormals to
 * 2^-9. No per-block scale, which is exactly what makes it interesting to
 * measure against q8: same byte count, and all of the adaptivity given up.
 *
 * Encode-then-decode in one pass, since the encoded byte is never stored. */
static float fp8_e4m3_round(float v) {
    if (!isfinite(v)) return v > 0 ? 448.0f : (v < 0 ? -448.0f : v);

    const float sign = v < 0 ? -1.0f : 1.0f;
    float a = fabsf(v);

    if (a >= 448.0f) return sign * 448.0f;      /* saturate, e4m3 has no inf */
    if (a < 0.0009765625f / 2.0f) return sign * 0.0f;   /* below half the last subnormal */

    int e;
    const float m = frexpf(a, &e);              /* a = m * 2^e, m in [0.5, 1) */

    /* frexp's exponent is one above the IEEE one. */
    int exp = e - 1;
    float step;
    if (exp < -6) step = ldexpf(1.0f, -9);      /* subnormal: fixed step 2^-9 */
    else          step = ldexpf(1.0f, exp - 3); /* 3 mantissa bits */
    (void)m;

    float q = nearbyintf(a / step) * step;
    if (q > 448.0f) q = 448.0f;
    return sign * q;
}

/* ── block-scaled integers ─────────────────────────────────────────────────
 * Symmetric, one scale per 32 values, chosen from the block's own maximum.
 * Asymmetric (scale + zero point) would fit the data better, but K and V are
 * roughly centred and the extra field costs as much again per block at these
 * widths — measure the simple one first.
 *
 * The scale is rounded THROUGH f16 here, because that is how the packed cache
 * stores it. Otherwise this function and the packed writer would be two
 * definitions of one format, differing by a rounding — which is exactly what
 * tests/test_kernels.c caught. */
static float f16_load(const uint8_t *p);
static void  f16_store(uint8_t *p, float v);

static float as_f16(float v) {
    uint8_t tmp[2];
    f16_store(tmp, v);
    return f16_load(tmp);
}

static void quantize_block(float *x, size_t n, float levels) {
    float amax = 0.0f;
    for (size_t i = 0; i < n; i++) {
        const float a = fabsf(x[i]);
        if (a > amax) amax = a;
    }
    if (amax == 0.0f) return;

    const float scale = as_f16(amax / levels);
    if (scale == 0.0f) return;
    const float inv   = 1.0f / scale;
    for (size_t i = 0; i < n; i++) {
        float q = nearbyintf(x[i] * inv);
        if (q >  levels)       q =  levels;
        if (q < -levels - 1.0f) q = -levels - 1.0f;   /* two's complement floor */
        x[i] = q * scale;
    }
}

void mynah_slm_kv_roundtrip(mynah_slm_kv_type t, float *x, size_t n) {
    if (t == MYNAH_SLM_KV_F32 || !x || n == 0) return;

    switch (t) {
        case MYNAH_SLM_KV_BF16:
            for (size_t i = 0; i < n; i++) x[i] = bf16_round(x[i]);
            return;
        case MYNAH_SLM_KV_FP8:
            for (size_t i = 0; i < n; i++) x[i] = fp8_e4m3_round(x[i]);
            return;
        case MYNAH_SLM_KV_Q8:
        case MYNAH_SLM_KV_Q4: {
            const float levels = (t == MYNAH_SLM_KV_Q8) ? 127.0f : 7.0f;
            for (size_t off = 0; off < n; off += KV_BLOCK) {
                size_t take = n - off;
                if (take > KV_BLOCK) take = KV_BLOCK;
                quantize_block(x + off, take, levels);
            }
            return;
        }
        case MYNAH_SLM_KV_F32:
            return;
    }
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

static void f16_store(uint8_t *p, float v) {
    uint32_t bits;
    memcpy(&bits, &v, sizeof bits);
    const uint32_t sign = (bits >> 16) & 0x8000u;
    int32_t  exp  = (int32_t)((bits >> 23) & 0xffu) - 127 + 15;
    uint32_t mant = bits & 0x7fffffu;
    uint16_t h;

    if (exp >= 31) {                       /* overflow -> the largest finite */
        h = (uint16_t)(sign | 0x7bffu);
    } else if (exp <= 0) {                 /* subnormal or zero */
        if (exp < -10) h = (uint16_t)sign;
        else {
            mant |= 0x800000u;
            const uint32_t shift = (uint32_t)(14 - exp);
            const uint32_t rounded = (mant + (1u << (shift - 1))) >> shift;
            h = (uint16_t)(sign | rounded);
        }
    } else {
        /* Round to nearest even on the 13 bits being dropped. */
        const uint32_t lsb = (mant >> 13) & 1u;
        mant += 0xfffu + lsb;
        if (mant & 0x800000u) { exp++; mant = 0; }
        h = (uint16_t)(sign | ((uint32_t)exp << 10) | (mant >> 13));
        if (exp >= 31) h = (uint16_t)(sign | 0x7bffu);
    }
    p[0] = (uint8_t)(h & 0xffu);
    p[1] = (uint8_t)(h >> 8);
}

/* ── encoding one block ────────────────────────────────────────────────────*/

static void pack_q8(uint8_t *out, const float *x, size_t n) {
    float amax = 0.0f;
    for (size_t i = 0; i < n; i++) { const float a = fabsf(x[i]); if (a > amax) amax = a; }
    const float scale = amax / 127.0f;
    f16_store(out, scale);

    const float inv = scale > 0.0f ? 1.0f / f16_load(out) : 0.0f;
    int8_t *q = (int8_t *)(out + 2);
    for (size_t i = 0; i < n; i++) {
        float v = nearbyintf(x[i] * inv);
        if (v >  127.0f) v =  127.0f;
        if (v < -128.0f) v = -128.0f;
        q[i] = (int8_t)v;
    }
    for (size_t i = n; i < KV_BLOCK; i++) q[i] = 0;
}

static void pack_q4(uint8_t *out, const float *x, size_t n) {
    float amax = 0.0f;
    for (size_t i = 0; i < n; i++) { const float a = fabsf(x[i]); if (a > amax) amax = a; }
    const float scale = amax / 7.0f;
    f16_store(out, scale);

    const float inv = scale > 0.0f ? 1.0f / f16_load(out) : 0.0f;
    uint8_t *q = out + 2;
    memset(q, 0, KV_BLOCK / 2);
    for (size_t i = 0; i < n; i++) {
        float v = nearbyintf(x[i] * inv);
        if (v >  7.0f) v =  7.0f;
        if (v < -8.0f) v = -8.0f;
        /* Biased by 8 into a nibble, low half first — llama.cpp's q4_0. */
        const uint8_t nib = (uint8_t)((int)v + 8) & 0x0fu;
        if (i < KV_BLOCK / 2) q[i] |= nib;
        else                  q[i - KV_BLOCK / 2] |= (uint8_t)(nib << 4);
    }
}

static void unpack_q8(const uint8_t *in, float *x) {
    const float scale = f16_load(in);
    const int8_t *q = (const int8_t *)(in + 2);
    for (int i = 0; i < KV_BLOCK; i++) x[i] = (float)q[i] * scale;
}

static void unpack_q4(const uint8_t *in, float *x) {
    const float scale = f16_load(in);
    const uint8_t *q = in + 2;
    for (int i = 0; i < KV_BLOCK / 2; i++) {
        x[i]                = (float)((int)(q[i] & 0x0fu) - 8) * scale;
        x[i + KV_BLOCK / 2] = (float)((int)(q[i] >> 4) - 8) * scale;
    }
}

/* ── geometry ──────────────────────────────────────────────────────────────*/

static size_t bytes_per_row(mynah_slm_kv_type t, uint32_t n) {
    switch (t) {
        case MYNAH_SLM_KV_F32:  return (size_t)n * 4;
        case MYNAH_SLM_KV_BF16: return (size_t)n * 2;
        case MYNAH_SLM_KV_FP8:  return (size_t)n;
        case MYNAH_SLM_KV_Q8:   return (size_t)(n / KV_BLOCK) * KV_Q8_BYTES;
        case MYNAH_SLM_KV_Q4:   return (size_t)(n / KV_BLOCK) * KV_Q4_BYTES;
    }
    return (size_t)n * 4;
}

int mynah_slm_kv_init(mynah_slm_kv *c, mynah_slm_kv_type type_k,
                      mynah_slm_kv_type type_v, uint32_t n_layers,
                      uint32_t n_ctx, uint32_t n_kv_heads, uint32_t head_dim) {
    if (!c) return -1;
    memset(c, 0, sizeof *c);

    const uint32_t kv_dim = n_kv_heads * head_dim;
    /* A head must be a whole number of blocks, or a (position, head) slice
     * would start mid-block and every accessor would need a shift. */
    if (kv_dim == 0 || head_dim % KV_BLOCK != 0) {
        if (type_k != MYNAH_SLM_KV_F32 || type_v != MYNAH_SLM_KV_F32) return -1;
    }

    c->type_k = type_k; c->type_v = type_v;
    c->n_layers = n_layers; c->n_ctx = n_ctx;
    c->kv_dim = kv_dim; c->head_dim = head_dim; c->n_kv_heads = n_kv_heads;
    c->pos_bytes_k = bytes_per_row(type_k, kv_dim);
    c->pos_bytes_v = bytes_per_row(type_v, kv_dim);

    c->k = mynah_slm_aligned_alloc(c->pos_bytes_k * n_ctx * n_layers);
    c->v = mynah_slm_aligned_alloc(c->pos_bytes_v * n_ctx * n_layers);
    if (!c->k || !c->v) { mynah_slm_kv_free(c); return -1; }
    return 0;
}

void mynah_slm_kv_free(mynah_slm_kv *c) {
    if (!c) return;
    mynah_slm_aligned_free(c->k);
    mynah_slm_aligned_free(c->v);
    memset(c, 0, sizeof *c);
}

size_t mynah_slm_kv_bytes(const mynah_slm_kv *c) {
    if (!c) return 0;
    return (c->pos_bytes_k + c->pos_bytes_v) * c->n_ctx * c->n_layers;
}

static uint8_t *slot(uint8_t *base, size_t pos_bytes, uint32_t n_ctx,
                     uint32_t layer, uint32_t pos) {
    return base + ((size_t)layer * n_ctx + pos) * pos_bytes;
}

static void put_row(uint8_t *dst, mynah_slm_kv_type t, const float *row, uint32_t n) {
    switch (t) {
        case MYNAH_SLM_KV_F32:
            memcpy(dst, row, (size_t)n * sizeof *row);
            return;
        case MYNAH_SLM_KV_BF16: {
            /* Round to nearest even, then keep the top half. Truncating would
             * bias every value toward zero, and a bias survives 28 layers in a
             * way noise does not. */
            for (uint32_t i = 0; i < n; i++) {
                uint32_t bits;
                memcpy(&bits, &row[i], sizeof bits);
                bits += 0x7fffu + ((bits >> 16) & 1u);
                dst[2 * i]     = (uint8_t)((bits >> 16) & 0xffu);
                dst[2 * i + 1] = (uint8_t)((bits >> 24) & 0xffu);
            }
            return;
        }
        case MYNAH_SLM_KV_FP8: {
            float tmp[KV_BLOCK];
            for (uint32_t off = 0; off < n; off += KV_BLOCK) {
                uint32_t take = n - off; if (take > KV_BLOCK) take = KV_BLOCK;
                memcpy(tmp, row + off, take * sizeof *tmp);
                mynah_slm_kv_roundtrip(MYNAH_SLM_KV_FP8, tmp, take);
                /* Stored as the rounded f32's top byte would not round-trip;
                 * keep the encoded value in a byte via its own quantizer. */
                for (uint32_t i = 0; i < take; i++) {
                    /* e4m3 as sign|exp|mant, rebuilt from the rounded float. */
                    uint32_t bits; memcpy(&bits, &tmp[i], sizeof bits);
                    const uint32_t sgn = (bits >> 24) & 0x80u;
                    const int32_t  e   = (int32_t)((bits >> 23) & 0xffu) - 127 + 7;
                    uint32_t m = (bits >> 20) & 0x7u;
                    uint8_t  b;
                    if (tmp[i] == 0.0f)      b = (uint8_t)sgn;
                    else if (e <= 0) {       /* subnormal: value = m * 2^-9 */
                        const float a = fabsf(tmp[i]) / 0.001953125f;  /* 2^-9 */
                        uint32_t q = (uint32_t)(a + 0.5f);
                        if (q > 7u) q = 7u;
                        b = (uint8_t)(sgn | q);
                    } else if (e >= 15) b = (uint8_t)(sgn | 0x7eu);
                    else b = (uint8_t)(sgn | ((uint32_t)e << 3) | m);
                    dst[off + i] = b;
                }
            }
            return;
        }
        case MYNAH_SLM_KV_Q8:
            for (uint32_t off = 0; off < n; off += KV_BLOCK)
                pack_q8(dst + (off / KV_BLOCK) * KV_Q8_BYTES, row + off, KV_BLOCK);
            return;
        case MYNAH_SLM_KV_Q4:
            for (uint32_t off = 0; off < n; off += KV_BLOCK)
                pack_q4(dst + (off / KV_BLOCK) * KV_Q4_BYTES, row + off, KV_BLOCK);
            return;
    }
}

void mynah_slm_kv_put_k(mynah_slm_kv *c, uint32_t layer, uint32_t pos, const float *row) {
    put_row(slot(c->k, c->pos_bytes_k, c->n_ctx, layer, pos), c->type_k, row, c->kv_dim);
}

void mynah_slm_kv_put_v(mynah_slm_kv *c, uint32_t layer, uint32_t pos, const float *row) {
    put_row(slot(c->v, c->pos_bytes_v, c->n_ctx, layer, pos), c->type_v, row, c->kv_dim);
}

/* One (position, head) slice, decoded to f32. */
static void get_slice(const uint8_t *row, mynah_slm_kv_type t, uint32_t head,
                      uint32_t head_dim, float *out) {
    switch (t) {
        case MYNAH_SLM_KV_F32:
            memcpy(out, (const float *)row + (size_t)head * head_dim,
                   head_dim * sizeof *out);
            return;
        case MYNAH_SLM_KV_BF16: {
            const uint8_t *p = row + (size_t)head * head_dim * 2;
            for (uint32_t i = 0; i < head_dim; i++) {
                const uint32_t bits = ((uint32_t)p[2 * i + 1] << 24) |
                                      ((uint32_t)p[2 * i] << 16);
                memcpy(&out[i], &bits, sizeof out[i]);
            }
            return;
        }
        case MYNAH_SLM_KV_FP8: {
            const uint8_t *p = row + (size_t)head * head_dim;
            for (uint32_t i = 0; i < head_dim; i++) {
                const uint8_t b = p[i];
                const float sgn = (b & 0x80u) ? -1.0f : 1.0f;
                const uint32_t e = (b >> 3) & 0x0fu;
                const uint32_t m = b & 0x7u;
                out[i] = e == 0 ? sgn * (float)m * 0.001953125f
                                : sgn * ldexpf(1.0f + (float)m / 8.0f, (int)e - 7);
            }
            return;
        }
        case MYNAH_SLM_KV_Q8: {
            const uint8_t *p = row + (size_t)head * (head_dim / KV_BLOCK) * KV_Q8_BYTES;
            for (uint32_t b = 0; b < head_dim / KV_BLOCK; b++)
                unpack_q8(p + b * KV_Q8_BYTES, out + b * KV_BLOCK);
            return;
        }
        case MYNAH_SLM_KV_Q4: {
            const uint8_t *p = row + (size_t)head * (head_dim / KV_BLOCK) * KV_Q4_BYTES;
            for (uint32_t b = 0; b < head_dim / KV_BLOCK; b++)
                unpack_q4(p + b * KV_Q4_BYTES, out + b * KV_BLOCK);
            return;
        }
    }
}

void mynah_slm_kv_gather_k(const mynah_slm_kv *c, uint32_t layer, uint32_t head,
                           uint32_t n_kv, float *out) {
    for (uint32_t t = 0; t < n_kv; t++)
        get_slice(slot(c->k, c->pos_bytes_k, c->n_ctx, layer, t), c->type_k,
                  head, c->head_dim, out + (size_t)t * c->head_dim);
}

void mynah_slm_kv_gather_v(const mynah_slm_kv *c, uint32_t layer, uint32_t head,
                           uint32_t n_kv, float *out) {
    for (uint32_t t = 0; t < n_kv; t++)
        get_slice(slot(c->v, c->pos_bytes_v, c->n_ctx, layer, t), c->type_v,
                  head, c->head_dim, out + (size_t)t * c->head_dim);
}

/* ── the fused decode accessors ────────────────────────────────────────────
 * One block at a time, vectorized. Scalar versions of these MEASURED SLOWER
 * than the f32 cache they were meant to beat — 11.5 tok/s against 14.5 — even
 * though they read a quarter of the bytes: the f32 path dots with 16-wide NEON
 * and the saving in traffic does not pay for giving that up. Compression only
 * becomes speed once the decode is vectorized too. */

#if defined(__ARM_NEON)

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

#elif defined(__AVX2__)
#include <immintrin.h>

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


float mynah_slm_kv_dot_k(const mynah_slm_kv *c, uint32_t layer, uint32_t pos,
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
        get_slice(row, c->type_k, head, hd, tmp);
        float acc = 0.0f;
        for (uint32_t i = 0; i < hd; i++) acc += q[i] * tmp[i];
        return acc;
    }
    return 0.0f;
}

void mynah_slm_kv_axpy_v(const mynah_slm_kv *c, uint32_t layer, uint32_t pos,
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
        get_slice(row, c->type_v, head, hd, tmp);
        for (uint32_t i = 0; i < hd; i++) out[i] += w * tmp[i];
    }
}
