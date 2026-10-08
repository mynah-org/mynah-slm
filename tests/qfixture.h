/* qfixture.h — synthetic quantized matrices for the kernel tests and benches.
 *
 * Random bytes are a valid block of every ggml type we test here: any nibble,
 * any 6-bit scale, any int8. Only the f16 scale fields need care — a random
 * half can be inf, nan or 6e4, which tests nothing — so they are patched to a
 * small positive range. No checkpoint, no quantizer: a 151936-row LM head
 * fixture is generated in well under a second, which is what lets the bench
 * run the real shapes.
 *
 * Header-only on purpose: test_kernels.c and bench_qmat.c both need exactly
 * this and nothing else does.
 *
 * SPDX-License-Identifier: MIT */
#ifndef MYNAH_SLM_QFIXTURE_H
#define MYNAH_SLM_QFIXTURE_H

#include "ingot/dtype.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

typedef struct { uint64_t s; } qfx_rng;

static inline uint32_t qfx_next(qfx_rng *r) {
    r->s = r->s * 6364136223846793005ull + 1442695040888963407ull;
    return (uint32_t)(r->s >> 33);
}

/* uniform in [lo, hi) */
static inline float qfx_uniform(qfx_rng *r, float lo, float hi) {
    return lo + (hi - lo) * (float)(qfx_next(r) & 0xffffffu) / 16777216.0f;
}

static inline void qfx_put_f16(unsigned char *p, float v) {
    const uint16_t h = ingot_f32_to_f16(v);
    p[0] = (unsigned char)(h & 0xff);
    p[1] = (unsigned char)(h >> 8);
}

/* Bytes per block and values per block for the types the fixtures cover. */
static inline int qfx_geometry(int type, size_t *elems, size_t *bytes) {
    switch (type) {
    case INGOT_TYPE_Q4_K: *elems = 256; *bytes = 144; return 0;
    case INGOT_TYPE_Q6_K: *elems = 256; *bytes = 210; return 0;
    case INGOT_TYPE_Q8_0: *elems = 32;  *bytes = 34;  return 0;
    default: return -1;
    }
}

/* Fill rows x cols of `type` into `out` (rows * cols/elems * bytes bytes). */
static inline int qfx_fill(int type, unsigned char *out, size_t rows, size_t cols,
                           uint64_t seed) {
    size_t elems = 0, bytes = 0;
    if (qfx_geometry(type, &elems, &bytes) != 0 || cols % elems != 0) return -1;
    const size_t nblocks = rows * (cols / elems);
    qfx_rng r = { seed * 2654435761ull + 1u };

    for (size_t i = 0; i < nblocks * bytes; i += 4) {
        const uint32_t v = qfx_next(&r);
        const size_t n = nblocks * bytes - i < 4 ? nblocks * bytes - i : 4;
        memcpy(out + i, &v, n);
    }
    for (size_t b = 0; b < nblocks; b++) {
        unsigned char *blk = out + b * bytes;
        switch (type) {
        case INGOT_TYPE_Q4_K:
            qfx_put_f16(blk,     qfx_uniform(&r, 1e-3f, 2e-2f));   /* d    */
            qfx_put_f16(blk + 2, qfx_uniform(&r, 0.0f,  1e-2f));   /* dmin */
            break;
        case INGOT_TYPE_Q6_K:
            qfx_put_f16(blk + 208, qfx_uniform(&r, 1e-4f, 2e-3f)); /* d at the END */
            break;
        case INGOT_TYPE_Q8_0:
            qfx_put_f16(blk, qfx_uniform(&r, 1e-4f, 2e-3f));
            break;
        }
    }
    return 0;
}

/* Activations shaped like a residual stream after RMSNorm: mostly O(1), with
 * a few channels an order of magnitude larger. Uniform noise would flatter a
 * per-32 int8 scale; the outliers are what make it work for its living. */
static inline void qfx_activations(float *x, size_t n, uint64_t seed) {
    qfx_rng r = { seed * 40503ull + 7u };
    for (size_t i = 0; i < n; i++) {
        float a = 0.0f;
        for (int k = 0; k < 4; k++) a += qfx_uniform(&r, -1.0f, 1.0f);
        if (qfx_next(&r) % 97u == 0) a *= 12.0f;
        x[i] = a;
    }
}

#endif /* MYNAH_SLM_QFIXTURE_H */
