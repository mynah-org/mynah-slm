/* qmat.c — see qmat.h.
 * SPDX-License-Identifier: MIT */
#include "qmat.h"

#include "kern.h"
#include "sgemm.h"
#include "threads.h"

#include "ingot/dtype.h"
#include "ingot/quant.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* The kernels themselves live in qmat_kern.c, compiled once per ISA; this
 * file is the API, the activation prep and the dispatch through the table
 * src/isa.c resolved (kern.h). */

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

/* Every kernel exists in a vector form per ISA and a scalar reference. The
 * scalar one is not a fallback nobody runs — it is the definition the others
 * have to agree with, and it is what a machine without the instructions gets.
 *
 * x86 cannot be executed natively on the M1, so it is verified two ways there:
 * `make check-x86` cross-compiles every x86 kernel TU, and `make
 * test-x86-rosetta` builds the suite as x86_64 and RUNS it under Rosetta.
 * `mynah-slm --dispatch` says which tables actually resolved. */

static int g_int8 = -1;

/* Requested (MYNAH_SLM_INT8 / --fast) AND the resolved ISA has vector int8
 * kernels. On a CPU without them the request is not honoured, and
 * `--dispatch` says so: int8 through a scalar twin would be slower than the
 * f32 path it replaces. */
int mynah_slm_matvec_int8_enabled(void) {
    if (g_int8 < 0) {
        /* Off unless asked for. It trades accuracy for speed, and a default
         * that quietly does that is how a quantization claim stops meaning
         * anything. `mynah-slm ppl` is what decides, not this. */
        const char *e = getenv("MYNAH_SLM_INT8");
        g_int8 = (e && strcmp(e, "0") != 0) ? 1 : 0;
    }
    return g_int8 && mynah_slm_kern_qmat()->int8;
}

int mynah_slm_matvec_int8_requested(void) {
    (void)mynah_slm_matvec_int8_enabled();
    return g_int8 > 0;
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

const char *mynah_slm_matvec_int8_isa(void) {
    const mynah_slm_qmat_kern *k = mynah_slm_kern_qmat();
    return k->int8 ? k->int8_name : "none";
}

/* The scalar twins, from the scalar table whatever resolved. */
int mynah_slm_q4k_int8_ref(const void *weights, size_t rows, size_t cols,
                           const mynah_slm_matvec_in *prep, float *output) {
    if (!weights || !prep || !output || !prep->have_int8 || cols % 256 != 0)
        return -1;
    mynah_slm_qmat_kern_scalar.q4k_i8((const unsigned char *)weights, rows, cols / 256,
                                      prep->xq, prep->xscale, prep->xsum, output);
    return 0;
}

int mynah_slm_q80_int8_ref(const void *weights, size_t rows, size_t cols,
                           const mynah_slm_matvec_in *prep, float *output) {
    if (!weights || !prep || !output || !prep->have_int8 || cols % 32 != 0)
        return -1;
    mynah_slm_qmat_kern_scalar.q80_i8((const unsigned char *)weights, rows, cols / 32,
                                      prep->xq, prep->xscale, output);
    return 0;
}

int mynah_slm_q6k_int8_ref(const void *weights, size_t rows, size_t cols,
                           const mynah_slm_matvec_in *prep, float *output) {
    if (!weights || !prep || !output || !prep->have_int8 || cols % 256 != 0)
        return -1;
    mynah_slm_qmat_kern_scalar.q6k_i8((const unsigned char *)weights, rows, cols / 256,
                                      prep->xq, prep->xscale, output);
    return 0;
}

int mynah_slm_matvec(int type, const void *weights, size_t rows, size_t cols,
                     const float *input, const mynah_slm_matvec_in *prep,
                     float *output) {
    if (!prep || !use_own_kernels()) return -1;
    const mynah_slm_qmat_kern *k = mynah_slm_kern_qmat();
    const unsigned char *w = (const unsigned char *)weights;
    const int int8 = prep->have_int8 && k->int8 && int8_type_on(type);

    /* int8 only: with f32 activations these two types are ingot's. */
    if (type == INGOT_TYPE_Q8_0) {
        if (!int8 || cols % 32 != 0) return -1;
        k->q80_i8(w, rows, cols / 32, prep->xq, prep->xscale, output);
        return 0;
    }
    if (type == INGOT_TYPE_Q6_K) {
        if (!int8 || cols % 256 != 0) return -1;
        k->q6k_i8(w, rows, cols / 256, prep->xq, prep->xscale, output);
        return 0;
    }
    if (type != INGOT_TYPE_Q4_K || cols % 256 != 0) return -1;

    if (int8) k->q4k_i8(w, rows, cols / 256, prep->xq, prep->xscale, prep->xsum, output);
    else      k->q4k_f32(w, rows, cols / 256, input, prep->xsum, output);
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
