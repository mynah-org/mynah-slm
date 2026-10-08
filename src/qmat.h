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
#include <stdio.h>

/* The per-32-element input sums our Q4_K kernel reads. Sized for a stack
 * buffer: cols/32, so 16384 columns. Everything we ship is far under it. */
#define MYNAH_SLM_XSUM_MAX 512
#define MYNAH_SLM_XQ_MAX  16384

/* The activations, prepared once per matvec and read by every row. Three
 * fields because the kernel needs three things and computing any of them per
 * row would undo the point:
 *   xsum    exact f32 sums per 32 values — the min term stays exact
 *   xq/xs   the same activations as int8 with a scale per 32, for the SDOT
 *           path. Only filled when that path is on — and have_int8 stays 0
 *           for a vector holding a NaN or inf, so it takes the f32 path,
 *           which propagates it (int8 would return a finite number). */
typedef struct {
    float  xsum[MYNAH_SLM_XSUM_MAX];
    float  xscale[MYNAH_SLM_XSUM_MAX];
    int8_t xq[MYNAH_SLM_XQ_MAX];
    int    have_int8;
    size_t cols;       /* the width prepared; 0 = nothing (too wide, or no input) */
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
 * and non-zero when the caller should fall back to ingot's. Q4_K is ours in
 * both forms; Q8_0 and Q6_K are ours only with int8 activations
 * (prep->have_int8), and ingot's otherwise. That is the whole
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

/* ── decode: several tokens, one weight read (K7) ──────────────────────────
 * out[t * ldo + r] = weights[r] . in[t * ldx], for t < ntok and a slice of
 * the rows — the weight-stationary product batched decode runs. Each weight
 * unit is decoded once and applied to every token. `prep[t]` is token t's
 * mynah_slm_matvec_prepare() of in + t * ldx (`in` may be NULL for the int8
 * types, which read only `prep`).
 *
 * The contract that makes it usable for decode: for every token the result
 * is BIT-IDENTICAL to mynah_slm_matvec on the same rows with prep[t]. So it
 * only runs when the single-token call would have taken ONE of our kernels
 * for every token alike (_ws_ok); otherwise it returns -1 and the caller does
 * the tokens one by one. Rows may be split across threads freely: a row's
 * result does not depend on which call computed it. Returns 0, or -1. */
int mynah_slm_matvec_ws(int type, const void *weights, size_t rows, size_t cols,
                        size_t ntok, const float *in, size_t ldx,
                        const mynah_slm_matvec_in *prep, float *out, size_t ldo);
int mynah_slm_matvec_ws_ok(int type, size_t cols, size_t ntok,
                           const mynah_slm_matvec_in *prep);

/* Fill `prep` from the input. Does the int8 half only when that path is on. */
void mynah_slm_matvec_prepare(const float *input, size_t cols,
                              mynah_slm_matvec_in *prep);

/* The same, with the int8 half filled whatever the switch says. For tests and
 * benches that exercise the int8 kernels and their scalar twins directly. */
void mynah_slm_matvec_prepare_int8(const float *input, size_t cols,
                                   mynah_slm_matvec_in *prep);

/* The scalar twin of the int8 Q4_K kernel: the executable definition of its
 * accumulation order (qmat.c), which every vector kernel must match BIT FOR
 * BIT. `prep` must hold int8 activations. Returns 0, or -1. */
int mynah_slm_q4k_int8_ref(const void *weights, size_t rows, size_t cols,
                           const mynah_slm_matvec_in *prep, float *output);

/* The scalar twins of the Q8_0 and Q6_K int8 kernels (K4), same contract. */
int mynah_slm_q80_int8_ref(const void *weights, size_t rows, size_t cols,
                           const mynah_slm_matvec_in *prep, float *output);
int mynah_slm_q6k_int8_ref(const void *weights, size_t rows, size_t cols,
                           const mynah_slm_matvec_in *prep, float *output);

/* Which int8 kernel this build runs ("avx512-vnni", "avx2", "neon-dotprod",
 * or "none"). A benchmark of the int8 path is invalid until this is printed. */
const char *mynah_slm_matvec_int8_isa(void);

/* Is the int8-activation path requested AND available on the resolved ISA
 * (src/isa.c)? It is a QUALITY
 * trade-off, not a free win: activations quantized to int8 per 32 values, so
 * it must be gated by `mynah-slm ppl` and never by a benchmark alone. */
int  mynah_slm_matvec_int8_enabled(void);
void mynah_slm_matvec_set_int8(int on);

/* Was int8 asked for, whether or not the resolved ISA can honour it? */
int  mynah_slm_matvec_int8_requested(void);

/* Narrow the int8 switch to some types (MYNAH_SLM_INT8_TYPES=q4_k,q8_0,q6_k,
 * default all). Never enables int8 on its own. */
void mynah_slm_matvec_set_int8_types(int q4_k, int q8_0, int q6_k);

/* The resolved type mask (MYNAH_SLM_INT8_* bits), for --dispatch. */
#define MYNAH_SLM_INT8_Q4_K 1
#define MYNAH_SLM_INT8_Q8_0 2
#define MYNAH_SLM_INT8_Q6_K 4
int mynah_slm_matvec_int8_types(void);

/* The parser behind MYNAH_SLM_INT8_TYPES: comma-separated exact tokens,
 * case-insensitive; NULL or "" = all; unknown tokens are reported on `warn`
 * (when not NULL) and ignored. Returns the mask. */
int mynah_slm_matvec_int8_types_parse(const char *s, FILE *warn);

/* Does this build have a kernel of its own for `type`? */
int mynah_slm_matvec_have(int type);

/* Force our kernels on or off, overriding MYNAH_SLM_KERNELS. Exists so
 * tests/bench_matvec.c can A/B ours against ingot's INTERLEAVED in one
 * process: two separate runs on a warm laptop disagree by 80% on tensors
 * neither change touches, which is how a real regression gets called noise
 * and a real win gets called a regression. */
void mynah_slm_matvec_set_enabled(int on);

#endif /* MYNAH_SLM_QMAT_H */
