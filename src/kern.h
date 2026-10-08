/* kern.h — per-ISA kernel translation units and their dispatch tables.
 *
 * Every SIMD kernel we own used to be chosen at COMPILE time, so a binary
 * built for a baseline ISA (the portable release tarball) ran the scalar
 * kernels on every machine. Now each kernel source is compiled once PER ISA,
 * with that ISA's flags, into its own object (the Makefile's "kernel TUs"),
 * and each object exports one table of function pointers. src/isa.c picks a
 * table per family at run time — CPUID / getauxval / sysctl, narrowed by
 * MYNAH_SLM_ISA, verified against the scalar table on first use.
 *
 * Why separate compiles and not __attribute__((target)): the kernels already
 * exist as `#if __AVX2__ / __ARM_NEON / else` code with dozens of inline
 * helpers. Compiling the same source with different flags keeps every one of
 * them exactly as written and has the compiler check each variant; target
 * attributes would have meant re-annotating every helper and rewriting every
 * guard (.work/isa-runtime-dispatch.md).
 *
 * Inside a kernel TU the Makefile defines:
 *   MYNAH_SLM_KERN_TU   the suffix (scalar, avx2, avx512, avx512vnni, neon,
 *                       neon_dotprod) pasted onto exported symbols
 *   MYNAH_SLM_KERN_ID   the same as a number, for #if
 *
 * SPDX-License-Identifier: MIT */
#ifndef MYNAH_SLM_KERN_H
#define MYNAH_SLM_KERN_H

#include <stddef.h>
#include <stdint.h>

/* Kernel TU ids, which are also the ISA levels of src/isa.h. On x86 they form
 * a ladder scalar < avx2 < avx512 < avx512vnni; on arm64 scalar < neon <
 * neon_dotprod. AVX-VNNI (VEX) is deliberately NOT a rung: it is a separate
 * feature (Ice Lake server has EVEX VNNI and no VEX VNNI), and we have no
 * kernel for it. */
#define MYNAH_SLM_KERN_ID_SCALAR        0
#define MYNAH_SLM_KERN_ID_NEON          1
#define MYNAH_SLM_KERN_ID_NEON_DOTPROD  2
#define MYNAH_SLM_KERN_ID_AVX2          3
#define MYNAH_SLM_KERN_ID_AVX512        4
#define MYNAH_SLM_KERN_ID_AVX512VNNI    5

#if defined(MYNAH_SLM_KERN_ID)
#define MYNAH_SLM_KERN_PASTE2(a, b) a##_##b
#define MYNAH_SLM_KERN_PASTE(a, b)  MYNAH_SLM_KERN_PASTE2(a, b)
#define MYNAH_SLM_KERN_SYM(base)    MYNAH_SLM_KERN_PASTE(base, MYNAH_SLM_KERN_TU)
#define MYNAH_SLM_KERN_STR2(x)      #x
#define MYNAH_SLM_KERN_STR(x)       MYNAH_SLM_KERN_STR2(x)
#define MYNAH_SLM_KERN_NAME         MYNAH_SLM_KERN_STR(MYNAH_SLM_KERN_TU)

/* What this TU may use. A TU that was not given the flags its id needs is a
 * build error, not a silently scalar "avx2" table. */
#if MYNAH_SLM_KERN_ID == MYNAH_SLM_KERN_ID_SCALAR
/* nothing: the scalar twins */
#elif MYNAH_SLM_KERN_ID == MYNAH_SLM_KERN_ID_AVX2
#if !defined(__AVX2__) || !defined(__FMA__) || !defined(__F16C__)
#error "the avx2 kernel TU needs -mavx2 -mfma -mf16c"
#endif
#define MYNAH_SLM_K_AVX2 1
#elif MYNAH_SLM_KERN_ID == MYNAH_SLM_KERN_ID_AVX512 || MYNAH_SLM_KERN_ID == MYNAH_SLM_KERN_ID_AVX512VNNI
#if !defined(__AVX2__) || !defined(__FMA__) || !defined(__F16C__) || !defined(__AVX512F__) || \
    !defined(__AVX512BW__) || !defined(__AVX512VL__) || !defined(__AVX512DQ__)
#error "the avx512 kernel TUs need -mavx2 -mfma -mf16c -mavx512f -mavx512bw -mavx512vl -mavx512dq"
#endif
#define MYNAH_SLM_K_AVX2 1
#define MYNAH_SLM_K_AVX512 1
#if MYNAH_SLM_KERN_ID == MYNAH_SLM_KERN_ID_AVX512VNNI
#if !defined(__AVX512VNNI__)
#error "the avx512vnni kernel TU needs -mavx512vnni"
#endif
#define MYNAH_SLM_K_VNNI 1
#endif
#elif MYNAH_SLM_KERN_ID == MYNAH_SLM_KERN_ID_NEON || MYNAH_SLM_KERN_ID == MYNAH_SLM_KERN_ID_NEON_DOTPROD
#if !defined(__ARM_NEON)
#error "the neon kernel TUs need an aarch64 target"
#endif
#define MYNAH_SLM_K_NEON 1
#if MYNAH_SLM_KERN_ID == MYNAH_SLM_KERN_ID_NEON_DOTPROD
#if !defined(__ARM_FEATURE_DOTPROD)
#error "the neon_dotprod kernel TU needs +dotprod"
#endif
#define MYNAH_SLM_K_DOTPROD 1
#endif
#else
#error "unknown MYNAH_SLM_KERN_ID"
#endif
#endif /* MYNAH_SLM_KERN_ID */

/* ── qmat: matvec straight off quantized weights (src/qmat_kern.c) ────────
 * Row kernels: `rows` consecutive rows of `blocks` super-blocks (Q4_K, Q6_K)
 * or 32-value blocks (Q8_0). The int8 entries take the activations prepared
 * once per matvec (qmat.h); in the scalar table they are the TWINS, the
 * executable definition of the int8 contract, and `int8` is 0 because no
 * build ships int8 without a vector kernel. */
typedef struct {
    const char *name;
    int         id;
    int         int8;            /* vector int8 kernels present */
    const char *int8_name;       /* "avx512-vnni", "avx2", "neon-dotprod" */
    void (*q4k_f32)(const unsigned char *w, size_t rows, size_t blocks,
                    const float *x, const float *xsum, float *out);
    void (*q4k_i8)(const unsigned char *w, size_t rows, size_t blocks,
                   const int8_t *xq, const float *xscale, const float *xsum,
                   float *out);
    void (*q80_i8)(const unsigned char *w, size_t rows, size_t nb,
                   const int8_t *xq, const float *xscale, float *out);
    void (*q6k_i8)(const unsigned char *w, size_t rows, size_t nb,
                   const int8_t *xq, const float *xscale, float *out);

    /* Weight-stationary twins of the four above (K7, .work/batched-decode-
     * kernel.md): `nt` tokens (1..MYNAH_SLM_WS_MAX) against the same rows,
     * each weight unit decoded ONCE and applied to every token. Token t reads
     * x[t] / xq[t], xscale[t], xsum[t] and writes out[t][0..rows). Per token
     * the result is BIT-IDENTICAL to the single-token entry of the same
     * table: same operations in the same order. The int8 ones are NULL where
     * the int8 ones above are. */
    void (*q4k_f32_ws)(const unsigned char *w, size_t rows, size_t blocks, size_t nt,
                       const float *const *x, const float *const *xsum,
                       float *const *out);
    void (*q4k_i8_ws)(const unsigned char *w, size_t rows, size_t blocks, size_t nt,
                      const int8_t *const *xq, const float *const *xscale,
                      const float *const *xsum, float *const *out);
    void (*q80_i8_ws)(const unsigned char *w, size_t rows, size_t nb, size_t nt,
                      const int8_t *const *xq, const float *const *xscale,
                      float *const *out);
    void (*q6k_i8_ws)(const unsigned char *w, size_t rows, size_t nb, size_t nt,
                      const int8_t *const *xq, const float *const *xscale,
                      float *const *out);
} mynah_slm_qmat_kern;

/* Tokens per weight-stationary call, and per register group inside one. */
#define MYNAH_SLM_WS_MAX   16
#define MYNAH_SLM_WS_GROUP 4

/* ── attn: the attention inner loops (src/attn_kern.c) ─────────────────────
 * Dispatched once per HEAD, not once per position: an indirect call per
 * position would cost a few percent of a 128-wide dot. */
struct mynah_slm_kv;
typedef struct {
    const char *name;
    int         id;
    /* one head over an f32 K/V history: exactly the old attention_head() */
    void  (*f32_head)(float *out, const float *q, const float *k, const float *v,
                      uint32_t h, uint32_t n_kv, uint32_t n_heads,
                      uint32_t n_kv_heads, uint32_t head_dim, float scale,
                      float *scratch);
    /* one head over a packed cache: scores, softmax, weighted V */
    void  (*kv_head)(float *out_h, const float *q_h, const struct mynah_slm_kv *c,
                     uint32_t layer, uint32_t kvh, uint32_t n_kv, float scale,
                     float *scores);
    /* single positions, for mynah_slm_kv_dot_k / _axpy_v */
    float (*kv_dot)(const struct mynah_slm_kv *c, uint32_t layer, uint32_t pos,
                    uint32_t head, const float *q);
    void  (*kv_axpy)(const struct mynah_slm_kv *c, uint32_t layer, uint32_t pos,
                     uint32_t head, float w, float *out);
} mynah_slm_attn_kern;

/* ── sgemm: our own f32 GEMM's planner + micro-kernels (src/sgemm.c) ─────── */
typedef struct {
    const char *name;
    int         id;
    int (*run)(int trans_b, size_t m, size_t n, size_t k, float alpha,
               const float *a, size_t lda, const float *b, size_t ldb,
               float beta, float *c, size_t ldc);
} mynah_slm_sgemm_kern;

/* The tables each TU exports. Which ones exist depends on the target
 * architecture; src/isa.c references only those the Makefile builds. */
extern const mynah_slm_qmat_kern  mynah_slm_qmat_kern_scalar;
extern const mynah_slm_attn_kern  mynah_slm_attn_kern_scalar;
extern const mynah_slm_sgemm_kern mynah_slm_sgemm_kern_scalar;
#if defined(__x86_64__) || defined(_M_X64)
extern const mynah_slm_qmat_kern  mynah_slm_qmat_kern_avx2;
extern const mynah_slm_qmat_kern  mynah_slm_qmat_kern_avx512vnni;
extern const mynah_slm_attn_kern  mynah_slm_attn_kern_avx2;
extern const mynah_slm_sgemm_kern mynah_slm_sgemm_kern_avx2;
extern const mynah_slm_sgemm_kern mynah_slm_sgemm_kern_avx512;
#elif defined(__aarch64__)
extern const mynah_slm_qmat_kern  mynah_slm_qmat_kern_neon;
extern const mynah_slm_qmat_kern  mynah_slm_qmat_kern_neon_dotprod;
extern const mynah_slm_attn_kern  mynah_slm_attn_kern_neon;
extern const mynah_slm_sgemm_kern mynah_slm_sgemm_kern_neon;
#endif

/* The resolved tables (src/isa.c). Never NULL: the scalar table is always
 * there to fall back to. */
const mynah_slm_qmat_kern  *mynah_slm_kern_qmat(void);
const mynah_slm_attn_kern  *mynah_slm_kern_attn(void);
const mynah_slm_sgemm_kern *mynah_slm_kern_sgemm(void);

#endif /* MYNAH_SLM_KERN_H */
