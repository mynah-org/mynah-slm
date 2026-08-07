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
#include <stdint.h>

/* The per-32-element input sums our Q4_K kernel reads. Sized for a stack
 * buffer: cols/32, so 16384 columns. Everything we ship is far under it. */
#define MYNAH_SLM_XSUM_MAX 512
#define MYNAH_SLM_XQ_MAX  16384

/* The activations, prepared once per matvec and read by every row. Three
 * fields because the kernel needs three things and computing any of them per
 * row would undo the point:
 *   xsum    exact f32 sums per 32 values — the min term stays exact
 *   xq/xs   the same activations as int8 with a scale per 32, for the SDOT
 *           path. Only filled when that path is on. */
typedef struct {
    float  xsum[MYNAH_SLM_XSUM_MAX];
    float  xscale[MYNAH_SLM_XSUM_MAX];
    int8_t xq[MYNAH_SLM_XQ_MAX];
    int    have_int8;
} mynah_slm_matvec_in;

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

/* ── decode: one token, one row range ──────────────────────────────────────
 * output[rows] = weights * input[cols], for a slice of the rows.
 *
 * Returns 0 when this build has a kernel of its own for `type` and used it,
 * and non-zero when the caller should fall back to ingot's. That is the whole
 * contract: we only take over where we have measured a win, everything else
 * keeps the validated generic path.
 *
 * `xsum` holds cols/32 floats and is filled by mynah_slm_matvec_prepare() ONCE
 * per input vector, not per row range — hoisting it out of the row loop is
 * most of the point (see qmat.c). Pass NULL to skip our kernels entirely.
 *
 * MYNAH_SLM_KERNELS=ingot in the environment forces the fallback, which is how
 * the A/B in `make bench` is run. */
int mynah_slm_matvec(int type, const void *weights, size_t rows, size_t cols,
                     const float *input, const mynah_slm_matvec_in *prep,
                     float *output);

/* Fill `prep` from the input. Does the int8 half only when that path is on. */
void mynah_slm_matvec_prepare(const float *input, size_t cols,
                              mynah_slm_matvec_in *prep);

/* Is the int8-activation path compiled in AND enabled? It is a QUALITY
 * trade-off, not a free win: activations quantized to int8 per 32 values, so
 * it must be gated by `mynah-slm ppl` and never by a benchmark alone. */
int  mynah_slm_matvec_int8_enabled(void);
void mynah_slm_matvec_set_int8(int on);

/* Does this build have a kernel of its own for `type`? */
int mynah_slm_matvec_have(int type);

/* Force our kernels on or off, overriding MYNAH_SLM_KERNELS. Exists so
 * tests/bench_matvec.c can A/B ours against ingot's INTERLEAVED in one
 * process: two separate runs on a warm laptop disagree by 80% on tensors
 * neither change touches, which is how a real regression gets called noise
 * and a real win gets called a regression. */
void mynah_slm_matvec_set_enabled(int on);

/* Q6_K is ours on x86 and ingot's on ARM by default, because that is what the
 * A/B measured on each (see qmat.c). This forces the choice either way, which
 * is what lets the gate in tests/test_kernels.c exercise OUR kernel on a Mac
 * and `make bench` re-run the comparison there. MYNAH_SLM_Q6K=own|ingot does
 * the same from the environment. */
void mynah_slm_matvec_set_q6k(int own);

#endif /* MYNAH_SLM_QMAT_H */
