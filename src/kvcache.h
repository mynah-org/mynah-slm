/* kvcache.h — the KV cache, and what precision it is kept at.
 *
 * At a 2275-token context the caches are ~520 MB at f32, MORE THAN THE MODEL,
 * and attention re-reads them on every generated token. So their precision is
 * both the biggest memory item and a real share of decode time — the one knob
 * that moves "lighter" and "faster" together.
 *
 * K AND V ARE SEPARATE, because they are not equally sensitive. A key goes
 * through the softmax exponent, where its error is amplified before anything
 * normalizes it; a value is averaged with weights that sum to one, where
 * errors partly cancel. Measured (docs/perf.md): the same 9.4% quantization
 * noise takes perplexity from 2.80 to 27.6 on keys and to 2.84 on values.
 * Four-bit keys are not a trade-off, they are a broken model.
 *
 * Block layout for the scaled formats is ggml's, 32 values to a block with an
 * f16 scale — 34 bytes for q8, 18 for q4 — so the comparison against what
 * llama.cpp offers for its KV cache is like for like.
 *
 * SPDX-License-Identifier: MIT */
#ifndef MYNAH_SLM_KVCACHE_H
#define MYNAH_SLM_KVCACHE_H

#include <stddef.h>
#include <stdint.h>

typedef enum {
    MYNAH_SLM_KV_F32 = 0,   /* the reference: what the parity gate was set on */
    MYNAH_SLM_KV_BF16,      /* top 16 bits of the f32, round to nearest even */
    MYNAH_SLM_KV_FP8,       /* e4m3: 4 exponent bits, 3 mantissa, no scale */
    MYNAH_SLM_KV_Q8,        /* int8 + one f16 scale per 32 values */
    MYNAH_SLM_KV_Q4,        /* int4 + one f16 scale per 32 values */
} mynah_slm_kv_type;

/* "f32" / "bf16" / "fp8" / "q8" / "q4". Returns 0 on success. */
int         mynah_slm_kv_type_parse(const char *name, mynah_slm_kv_type *out);
const char *mynah_slm_kv_type_name(mynah_slm_kv_type t);

/* Bits per stored value, scales included — what the memory table is built on. */
double mynah_slm_kv_bits(mynah_slm_kv_type t);

/* Quantize and immediately dequantize `n` values in place: the numerical
 * effect of the format with none of the plumbing. This is the DEFINITION each
 * packed path has to agree with, and tests/test_kernels.c gates it. */
void mynah_slm_kv_roundtrip(mynah_slm_kv_type t, float *x, size_t n);

/* ── the cache itself ──────────────────────────────────────────────────────
 * [n_layers][n_ctx][kv_dim], K and V held separately and possibly at different
 * precisions. A position is contiguous, which is the order attention walks. */
typedef struct mynah_slm_kv {
    mynah_slm_kv_type type_k, type_v;
    uint32_t n_layers, n_ctx, kv_dim, head_dim, n_kv_heads;
    size_t   pos_bytes_k, pos_bytes_v;   /* one position, one layer */
    uint8_t *k, *v;
} mynah_slm_kv;

/* kv_dim must be a multiple of 32 for the scaled formats. Returns 0, or -1
 * with nothing allocated. */
int    mynah_slm_kv_init(mynah_slm_kv *c, mynah_slm_kv_type type_k,
                         mynah_slm_kv_type type_v, uint32_t n_layers,
                         uint32_t n_ctx, uint32_t n_kv_heads, uint32_t head_dim);
void   mynah_slm_kv_free(mynah_slm_kv *c);
size_t mynah_slm_kv_bytes(const mynah_slm_kv *c);

/* Store one position's kv_dim f32 values, encoding as the type requires. */
void mynah_slm_kv_put_k(mynah_slm_kv *c, uint32_t layer, uint32_t pos, const float *row);
void mynah_slm_kv_put_v(mynah_slm_kv *c, uint32_t layer, uint32_t pos, const float *row);

/* [n_kv][head_dim] f32, contiguous — what the batched (sgemm) path needs. */
void mynah_slm_kv_gather_k(const mynah_slm_kv *c, uint32_t layer, uint32_t head,
                           uint32_t n_kv, float *out);
void mynah_slm_kv_gather_v(const mynah_slm_kv *c, uint32_t layer, uint32_t head,
                           uint32_t n_kv, float *out);

/* The decode path, fused: the packed bytes are read once and never
 * materialized. Dequantizing the history into scratch instead would move MORE
 * memory than the compression saves, which is the whole reason these exist.
 *
 *   dot_k    scores[t] = q . K[t][head]
 *   axpy_v   out += w * V[t][head]
 */
float mynah_slm_kv_dot_k(const mynah_slm_kv *c, uint32_t layer, uint32_t pos,
                         uint32_t head, const float *q);
void  mynah_slm_kv_axpy_v(const mynah_slm_kv *c, uint32_t layer, uint32_t pos,
                          uint32_t head, float w, float *out);

#endif /* MYNAH_SLM_KVCACHE_H */
