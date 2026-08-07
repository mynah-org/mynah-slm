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

int mynah_slm_matvec_have(int type) {
    return type == INGOT_TYPE_Q4_K && use_own_kernels();
}

void mynah_slm_matvec_prepare(const float *input, size_t cols, float *xsum) {
    if (!input || !xsum) return;
    for (size_t s = 0; s < cols / 32; s++) {
        const float *x = input + s * 32;
        float acc = 0.0f;
        for (int i = 0; i < 32; i++) acc += x[i];
        xsum[s] = acc;
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

#if defined(__ARM_NEON)
#include <arm_neon.h>

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

#else   /* the scalar twin, and the reference for what the NEON one means */

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
                     const float *input, const float *xsum, float *output) {
    if (!xsum || type != INGOT_TYPE_Q4_K || !use_own_kernels()) return -1;
    if (cols % 256 != 0) return -1;         /* Q4_K super-blocks, by definition */

    const size_t blocks = cols / 256;
    const unsigned char *base = (const unsigned char *)weights;
    for (size_t r = 0; r < rows; r++)
        output[r] = q4_k_row(base + r * blocks * 144, blocks, input, xsum);
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
