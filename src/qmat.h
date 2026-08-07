/* qmat.h — quantized matrix products, ours.
 *
 * THE BOUNDARY, stated because it is a decision and not an accident:
 *
 *   ingot reads containers.  It maps a GGUF, knows every block geometry, and
 *   decodes a block to f32 bit-for-bit against llama.cpp. That is a complete
 *   job and it is the job it should keep.
 *
 *   THIS file computes.  A product is where an engine's own shapes, thread
 *   pool, layouts and fusions live, and those are ours — nobody else consuming
 *   a weight container wants our prefill batch width or our KV layout. Kernels
 *   written here can specialize; a kernel written to be generic cannot.
 *
 * So we call ingot for what it is: `ingot_dequant_matrix` to turn a strip of
 * stored blocks into f32, plus the geometry. What happens to those floats is
 * this file's business.
 *
 * The policy itself is ported, not invented — mynah-asr's `qmat.h` reached it
 * first, on the same hardware:
 *
 *   decode (one token)   quantized matvec straight off the stored bytes. The
 *                        work is memory-bound; nothing is amortized, so
 *                        touching each weight once is the whole game.
 *   prefill (T tokens)   dequantize a row STRIP once, then sgemm it against
 *                        all T columns. The weight read is amortized T-fold
 *                        and the problem stops being memory-bound.
 *
 * A strip rather than the whole matrix because the whole matrix in f32 is
 * 12 MB for one FFN tensor and 600 MB for the LM head. The strip is sized to
 * stay in cache, so the dequantized floats are consumed while still warm.
 *
 * SPDX-License-Identifier: MIT */
#ifndef MYNAH_SLM_QMAT_H
#define MYNAH_SLM_QMAT_H

#include <stddef.h>

/* Rows dequantized per pass. Sized so one strip of f32 stays in L2 next to the
 * activations it is about to multiply. */
#define MYNAH_SLM_STRIP_ROWS 128

/* out[tokens][rows] = in[tokens][cols] * weights^T
 *
 * `weights` is the row-major [rows, cols] block matrix exactly as stored, so
 * `cols` must be a whole number of blocks. `scratch` holds
 * MYNAH_SLM_STRIP_ROWS * cols floats and is owned by the caller — nothing here
 * allocates, because this runs inside the token loop.
 *
 * NOT bit-identical to `tokens` separate matvecs: sgemm sums in its own order.
 * The difference is a reorder, not an approximation (~1e-6 relative), and
 * tests/test_batch.c holds it to that against the one-token path rather than
 * assuming it. Returns 0, or -1. */
int mynah_slm_qmatmat(int type, const void *weights, size_t rows, size_t cols,
                      const float *in, float *out, size_t tokens, float *scratch);

#endif /* MYNAH_SLM_QMAT_H */
