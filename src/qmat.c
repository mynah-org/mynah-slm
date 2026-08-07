/* qmat.c — see qmat.h.
 * SPDX-License-Identifier: MIT */
#include "qmat.h"

#include "threads.h"

#include "ingot/dtype.h"
#include "ingot/quant.h"

#if defined(MYNAH_SLM_BLAS_ACCELERATE)
#include <Accelerate/Accelerate.h>
#else
#include <cblas.h>
#endif

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
