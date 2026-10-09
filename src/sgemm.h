/* sgemm.h — our own f32 GEMM, so no external BLAS has to be in the process.
 *
 * WHY THIS EXISTS. Ownership first, speed second — the same decision
 * mynah-tts took (its .work/no-blas.md, "OpenBLAS leaves the process for
 * good"). A vendor BLAS brings a second thread pool into our address space,
 * with its own spin policy and its own idea of how many cores it may use, and
 * every measurement then has to carry environment variables that must be
 * ABSENT for it to be valid. mynah-asr measured the failure mode outright:
 * several inferences sharing one OpenBLAS pool collapsed aggregate throughput
 * from four concurrent requests up. A server that batches requests (PLAN.md,
 * S-items) cannot carry that.
 *
 * It is also a dependency: on Linux the build refuses to start without
 * `libopenblas-dev`, for three call sites. `make BLAS=none` removes it.
 *
 * OPT-IN, NOT YET THE DEFAULT. Measured on a 4-vCPU Cascade Lake VM
 * (.work/no-blas.md), this file is 0.5-0.65x OpenBLAS on the T=256 prefill
 * shapes and 0.3x on QK^T at head_dim 128, and ahead only on short batches.
 * The default flips when it reaches parity on the target hosts, not before:
 * closing the gap needs packed panels and 2-D register blocking, which is a
 * separate, measured item.
 *
 * THE SURFACE IS THREE CALL SITES, all prefill or batched attention — decode
 * never reaches a GEMM:
 *
 *   src/qmat.c     out[T][rows] = in[T][cols] . strip[rows][cols]^T   (NT)
 *   src/kernels.c  S[n_q][n_kv] = Q[n_q][hd]  . K[n_kv][hd]^T          (NT)
 *                  O[n_q][hd]   = S[n_q][n_kv] . V[n_kv][hd]           (NN)
 *
 * so two families, and both are written for what these shapes are rather than
 * for a general panel GEMM:
 *
 *   NT  op(B) = B^T: both operands are contiguous along k. One output element
 *       is a dot product; a tile of MR x NR of them is held in registers and
 *       the k axis is vectorised. Nothing is packed and nothing is copied.
 *   NN  op(B) = B:   op(B)'s columns are contiguous. The mynah-tts micro-kernel
 *       shape: broadcast one value of A, FMA it against NV vectors of B.
 *
 * DETERMINISM IS STRONGER THAN IN mynah-tts, and it is free. Every output
 * element is computed by the SAME sequence of operations whichever tile,
 * edge kernel or thread computes it:
 *
 *   NT  per-lane fma chain over p, one fixed horizontal-sum tree, then the
 *       scalar k-tail folded in with fmaf in order;
 *   NN  one fma chain over p in order, and the scalar column tail uses fmaf so
 *       it rounds exactly like a vector lane.
 *
 * So the result is bit-identical across thread counts AND across tilings, not
 * merely across thread counts, and tests/test_sgemm.c asserts it with memcmp.
 * The reduction over k is never split into partial sums that get added: a
 * k-block (NN, beta == 0) parks the raw f32 accumulator in C and resumes the
 * same fma chain from it, which is exact.
 *
 * Against a vendor BLAS the result is NOT bit-identical (they reassociate);
 * against the double-precision reference it agrees to a stated relative bound.
 *
 * SPDX-License-Identifier: MIT */
#ifndef MYNAH_SLM_SGEMM_H
#define MYNAH_SLM_SGEMM_H

#include <stddef.h>

/* C[m][n] = alpha * A[m][k] . op(B) + beta * C, row-major.
 *
 *   trans_b == 0   B is [k][n], row stride ldb >= n
 *   trans_b != 0   B is [n][k], row stride ldb >= k   (op(B) = B^T)
 *
 * A is never transposed: no call site needs it. lda >= k, ldc >= n.
 *
 * beta == 0 means C is WRITTEN, never read — an uninitialised or NaN-carrying
 * C must not poison the result. That is cblas' contract and ours.
 *
 * Runs on the mynah thread pool. MUST NOT be called from inside a
 * mynah_slm_parallel_for task: the pool has one region at a time.
 *
 * Which implementation runs: ours, unless this build linked a vendor BLAS
 * (BLAS=openblas / BLAS=accelerate) — then the vendor's, unless
 * MYNAH_SLM_SGEMM=own says otherwise. Read once, at the first call.
 * Returns 0, or -1 on a bad argument (nothing computed). */
int mynah_slm_sgemm(int trans_b, size_t m, size_t n, size_t k, float alpha,
                    const float *a, size_t lda, const float *b, size_t ldb,
                    float beta, float *c, size_t ldc);

/* Ours, always, whatever the build linked. For the A/B and the tests. */
int mynah_slm_sgemm_own(int trans_b, size_t m, size_t n, size_t k, float alpha,
                        const float *a, size_t lda, const float *b, size_t ldb,
                        float beta, float *c, size_t ldc);

/* The definition of correctness: an i, j, p triple loop accumulating in
 * double. Never vectorised, never threaded. */
void mynah_slm_sgemm_reference(int trans_b, size_t m, size_t n, size_t k,
                               float alpha, const float *a, size_t lda,
                               const float *b, size_t ldb, float beta,
                               float *c, size_t ldc);

/* "neon", "avx512", "avx2" or "scalar" — what this translation unit compiled.
 * A fact about the build, answered by the file that owns it. */
const char *mynah_slm_sgemm_isa(void);

/* "own", "openblas" or "accelerate" — what mynah_slm_sgemm() resolves to in
 * THIS process, environment included. Resolves on first use, like the call. */
const char *mynah_slm_sgemm_backend(void);

/* What RAN, not what could run. Relaxed counters, one bump per call. A
 * benchmark of the own path that shows `own == 0` measured something else
 * (.work/engineering-method.md: a benchmark is invalid until dispatch is
 * proven). */
typedef struct {
    unsigned long long calls, own_nt, own_nn, reference, vendor, refused;
} mynah_slm_sgemm_stats;

void mynah_slm_sgemm_stats_get(mynah_slm_sgemm_stats *out);
void mynah_slm_sgemm_stats_reset(void);

#endif /* MYNAH_SLM_SGEMM_H */
