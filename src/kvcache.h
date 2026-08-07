/* kvcache.h — what precision the KV cache is kept at.
 *
 * At a 2275-token context the caches are ~520 MB, MORE THAN THE MODEL, and
 * attention re-reads them on every generated token. So their precision is both
 * the biggest memory item and a real share of decode time — the one knob that
 * moves "lighter" and "faster" together.
 *
 * The risk is entirely on the quality side, and quality is the thing you
 * cannot derive: the memory saving is arithmetic, known before writing a line.
 * So this module starts as a ROUND TRIP — values are quantized and immediately
 * dequantized back into the f32 cache — which measures exactly the numerical
 * damage of each format while changing nothing else. Packed storage (the part
 * that actually saves the bytes) is worth writing only for formats that
 * survive that measurement.
 *
 * Block size is 32 values, the same as ggml's q8_0/q4_0, so the comparison
 * against what llama.cpp offers for its KV cache is like for like.
 *
 * SPDX-License-Identifier: MIT */
#ifndef MYNAH_SLM_KVCACHE_H
#define MYNAH_SLM_KVCACHE_H

#include <stddef.h>

typedef enum {
    MYNAH_SLM_KV_F32 = 0,   /* the reference: what the parity gate was set on */
    MYNAH_SLM_KV_BF16,      /* top 16 bits of the f32, round to nearest even */
    MYNAH_SLM_KV_FP8,       /* e4m3: 4 exponent bits, 3 mantissa, no scale */
    MYNAH_SLM_KV_Q8,        /* int8 + one f32 scale per 32 values */
    MYNAH_SLM_KV_Q4,        /* int4 + one f32 scale per 32 values */
} mynah_slm_kv_type;

/* "f32" / "bf16" / "fp8" / "q8" / "q4". Returns 0 on success. */
int         mynah_slm_kv_type_parse(const char *name, mynah_slm_kv_type *out);
const char *mynah_slm_kv_type_name(mynah_slm_kv_type t);

/* Bits per stored value, scales included — what the memory table is built on. */
double mynah_slm_kv_bits(mynah_slm_kv_type t);

/* Quantize and immediately dequantize `n` values in place: the exact numerical
 * effect of storing them in this format, with none of the plumbing. F32 is a
 * no-op, so the default path costs one branch per cached position. */
void mynah_slm_kv_roundtrip(mynah_slm_kv_type t, float *x, size_t n);

#endif /* MYNAH_SLM_KVCACHE_H */
