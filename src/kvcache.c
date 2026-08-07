/* kvcache.c — see kvcache.h.
 * SPDX-License-Identifier: MIT */
#include "kvcache.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

#define KV_BLOCK 32

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
 * widths — measure the simple one first. */
static void quantize_block(float *x, size_t n, float levels) {
    float amax = 0.0f;
    for (size_t i = 0; i < n; i++) {
        const float a = fabsf(x[i]);
        if (a > amax) amax = a;
    }
    if (amax == 0.0f) return;

    const float scale = amax / levels;
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
