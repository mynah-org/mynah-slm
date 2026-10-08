/* sgemm.c — see sgemm.h for why this exists and what the call sites are.
 *
 * Shape of the file, ported from mynah-tts src/sgemm.c and cut down to what an
 * SLM needs: one vector abstraction over NEON / AVX-512 / AVX2 / scalar, a
 * micro-kernel per family stamped out by a macro, a planner that cuts the
 * output into disjoint tiles, and the pool running the tiles. No packing: on
 * these shapes both operands are already contiguous along the axis the kernel
 * walks.
 *
 * SPDX-License-Identifier: MIT */
#include "sgemm.h"

#include "threads.h"

#include <math.h>
#include <stdatomic.h>
#include <stdlib.h>
#include <string.h>

#if defined(MYNAH_SLM_BLAS_ACCELERATE)
#include <Accelerate/Accelerate.h>
#define SG_VENDOR "accelerate"
#elif defined(MYNAH_SLM_BLAS_OPENBLAS)
#include <cblas.h>
#define SG_VENDOR "openblas"
#endif

/* ── ISA abstraction ───────────────────────────────────────────────────────
 * The scalar build is not a second algorithm: SG_L == 1 makes a "vector" one
 * float and the same loop nest runs. fmaf, not a*b+c, so a scalar lane rounds
 * exactly like a vector lane — that is what makes the edge kernels agree with
 * the full ones bit for bit. */
#if defined(__ARM_NEON) || defined(__aarch64__)
#include <arm_neon.h>
#define SG_ISA "neon"
#define SG_L 4
typedef float32x4_t sg_v;
#define sg_zero()        vdupq_n_f32(0.0f)
#define sg_load(p)       vld1q_f32(p)
#define sg_store(p, v)   vst1q_f32((p), (v))
#define sg_dup(x)        vdupq_n_f32(x)
#define sg_fma(acc, a, b) vfmaq_f32((acc), (a), (b))
/* 32 v-registers: a 4x4 NT tile is 16 accumulators + 4 A + 1 B. */
#define SG_NT_MR 4
#define SG_NT_NR 4
#define SG_NN_NV 4
static inline float sg_hsum(sg_v v) {
    /* Fixed tree, written out rather than vaddvq_f32 so the order is ours. */
    const float32x2_t s = vadd_f32(vget_low_f32(v), vget_high_f32(v));
    return vget_lane_f32(s, 0) + vget_lane_f32(s, 1);
}

#elif defined(__AVX512F__)
#include <immintrin.h>
#define SG_ISA "avx512"
#define SG_L 16
typedef __m512 sg_v;
#define sg_zero()        _mm512_setzero_ps()
#define sg_load(p)       _mm512_loadu_ps(p)
#define sg_store(p, v)   _mm512_storeu_ps((p), (v))
#define sg_dup(x)        _mm512_set1_ps(x)
#define sg_fma(acc, a, b) _mm512_fmadd_ps((a), (b), (acc))
/* 32 zmm registers, the same budget as NEON. */
#define SG_NT_MR 4
#define SG_NT_NR 4
#define SG_NN_NV 4
static inline float sg_hsum(sg_v v) {
    const __m256 h = _mm256_add_ps(_mm512_castps512_ps256(v),
                                   _mm512_extractf32x8_ps(v, 1));
    const __m128 q = _mm_add_ps(_mm256_castps256_ps128(h), _mm256_extractf128_ps(h, 1));
    const __m128 d = _mm_add_ps(q, _mm_movehl_ps(q, q));
    return _mm_cvtss_f32(_mm_add_ss(d, _mm_movehdup_ps(d)));
}

#elif defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define SG_ISA "avx2"
#define SG_L 8
typedef __m256 sg_v;
#define sg_zero()        _mm256_setzero_ps()
#define sg_load(p)       _mm256_loadu_ps(p)
#define sg_store(p, v)   _mm256_storeu_ps((p), (v))
#define sg_dup(x)        _mm256_set1_ps(x)
#define sg_fma(acc, a, b) _mm256_fmadd_ps((a), (b), (acc))
/* 16 ymm registers: 4x2 = 8 accumulators + 4 A + 1 B, no spill. */
#define SG_NT_MR 4
#define SG_NT_NR 2
#define SG_NN_NV 2
static inline float sg_hsum(sg_v v) {
    const __m128 q = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    const __m128 d = _mm_add_ps(q, _mm_movehl_ps(q, q));
    return _mm_cvtss_f32(_mm_add_ss(d, _mm_movehdup_ps(d)));
}

#else
#define SG_ISA "scalar"
#define SG_L 1
typedef float sg_v;
#define sg_zero()        0.0f
#define sg_load(p)       (*(p))
#define sg_store(p, v)   (*(p) = (v))
#define sg_dup(x)        (x)
#define sg_fma(acc, a, b) fmaf((a), (b), (acc))
#define SG_NT_MR 4
#define SG_NT_NR 2
#define SG_NN_NV 4
static inline float sg_hsum(sg_v v) { return v; }
#endif

#define SG_NN_MR 4

/* Below this many MACs the pool dispatch costs more than the arithmetic:
 * mynah-tts measured ~20 us of wake-up per region, and 2^17 MACs is a few
 * tens of microseconds on one core. Cost-model estimate, not a sweep. */
#define SG_PARALLEL_MIN_WORK 131072u

/* ── counters ──────────────────────────────────────────────────────────────*/
static atomic_ullong g_calls, g_nt, g_nn, g_ref, g_vendor, g_refused;

static void bump(atomic_ullong *c) { atomic_fetch_add_explicit(c, 1ull, memory_order_relaxed); }

void mynah_slm_sgemm_stats_get(mynah_slm_sgemm_stats *o) {
    if (!o) return;
    o->calls     = atomic_load_explicit(&g_calls,   memory_order_relaxed);
    o->own_nt    = atomic_load_explicit(&g_nt,      memory_order_relaxed);
    o->own_nn    = atomic_load_explicit(&g_nn,      memory_order_relaxed);
    o->reference = atomic_load_explicit(&g_ref,     memory_order_relaxed);
    o->vendor    = atomic_load_explicit(&g_vendor,  memory_order_relaxed);
    o->refused   = atomic_load_explicit(&g_refused, memory_order_relaxed);
}

void mynah_slm_sgemm_stats_reset(void) {
    atomic_store(&g_calls, 0); atomic_store(&g_nt, 0); atomic_store(&g_nn, 0);
    atomic_store(&g_ref, 0);   atomic_store(&g_vendor, 0); atomic_store(&g_refused, 0);
}

const char *mynah_slm_sgemm_isa(void) { return SG_ISA; }

/* ── reference ─────────────────────────────────────────────────────────────*/
void mynah_slm_sgemm_reference(int trans_b, size_t m, size_t n, size_t k,
                               float alpha, const float *a, size_t lda,
                               const float *b, size_t ldb, float beta,
                               float *c, size_t ldc) {
    for (size_t i = 0; i < m; i++)
        for (size_t j = 0; j < n; j++) {
            double s = 0.0;
            for (size_t p = 0; p < k; p++)
                s += (double)a[i * lda + p] *
                     (double)(trans_b ? b[j * ldb + p] : b[p * ldb + j]);
            float *cp = c + i * ldc + j;
            *cp = (beta == 0.0f) ? (float)(alpha * s)
                                 : (float)(alpha * s) + beta * *cp;
        }
}

/* ── store, shared by both families ────────────────────────────────────────
 * beta == 0 never reads C. */
static inline void sg_put(float *cp, float v, float alpha, float beta) {
    *cp = (beta == 0.0f) ? alpha * v : alpha * v + beta * *cp;
}

/* ── NT: C[i][j] = A[i][:] . B[j][:] ───────────────────────────────────────
 * MR rows of A against NR rows of B, k vectorised. Per element: lane-wise fma
 * chain, sg_hsum, then the k % SG_L tail with fmaf. Same sequence in every
 * instantiation, hence bit-identical edges. */
#define SG_DEFINE_NT(NAME, MR, NR)                                              \
    static void NAME(size_t k, const float *a, size_t lda, const float *b,      \
                     size_t ldb, float alpha, float beta, float *c,             \
                     size_t ldc) {                                              \
        sg_v acc[MR][NR];                                                       \
        for (int r = 0; r < (MR); r++)                                          \
            for (int q = 0; q < (NR); q++) acc[r][q] = sg_zero();               \
        size_t p = 0;                                                           \
        for (; p + SG_L <= k; p += SG_L) {                                      \
            sg_v av[MR];                                                        \
            for (int r = 0; r < (MR); r++) av[r] = sg_load(a + r * lda + p);    \
            for (int q = 0; q < (NR); q++) {                                    \
                const sg_v bv = sg_load(b + q * ldb + p);                       \
                for (int r = 0; r < (MR); r++)                                  \
                    acc[r][q] = sg_fma(acc[r][q], av[r], bv);                   \
            }                                                                   \
        }                                                                       \
        for (int r = 0; r < (MR); r++)                                          \
            for (int q = 0; q < (NR); q++) {                                    \
                float s = sg_hsum(acc[r][q]);                                   \
                float t = 0.0f;                                                 \
                for (size_t pp = p; pp < k; pp++)                               \
                    t = fmaf(a[r * lda + pp], b[q * ldb + pp], t);              \
                sg_put(c + r * ldc + q, s + t, alpha, beta);                    \
            }                                                                   \
    }

SG_DEFINE_NT(sg_nt_full, SG_NT_MR, SG_NT_NR)
SG_DEFINE_NT(sg_nt_row,  1,        SG_NT_NR)
SG_DEFINE_NT(sg_nt_col,  SG_NT_MR, 1)
SG_DEFINE_NT(sg_nt_one,  1,        1)

/* ── NN: C[i][:] = sum_p A[i][p] * B[p][:] ─────────────────────────────────
 * The mynah-tts micro-kernel: broadcast A, FMA against NV vectors of B. One
 * fma chain per output element over p in order; the column tail does the
 * same chain with fmaf. */
#define SG_DEFINE_NN(NAME, MR, NV)                                              \
    static void NAME(size_t k, const float *a, size_t lda, const float *b,      \
                     size_t ldb, float alpha, float beta, float *c,             \
                     size_t ldc, int resume, int finish) {                      \
        sg_v acc[MR][NV];                                                       \
        for (int r = 0; r < (MR); r++)                                          \
            for (int v = 0; v < (NV); v++)                                      \
                acc[r][v] = resume ? sg_load(c + r * ldc + v * SG_L) : sg_zero(); \
        for (size_t p = 0; p < k; p++) {                                        \
            sg_v bv[NV];                                                        \
            for (int v = 0; v < (NV); v++) bv[v] = sg_load(b + p * ldb + v * SG_L); \
            for (int r = 0; r < (MR); r++) {                                    \
                const sg_v av = sg_dup(a[r * lda + p]);                         \
                for (int v = 0; v < (NV); v++)                                  \
                    acc[r][v] = sg_fma(acc[r][v], av, bv[v]);                   \
            }                                                                   \
        }                                                                       \
        for (int r = 0; r < (MR); r++)                                          \
            for (int v = 0; v < (NV); v++) {                                    \
                if (!finish) { sg_store(c + r * ldc + v * SG_L, acc[r][v]); continue; } \
                float tmp[SG_L];                                                \
                sg_store(tmp, acc[r][v]);                                       \
                for (int l = 0; l < SG_L; l++)                                  \
                    sg_put(c + r * ldc + v * SG_L + l, tmp[l], alpha, beta);    \
            }                                                                   \
    }

SG_DEFINE_NN(sg_nn_full, SG_NN_MR, SG_NN_NV)
SG_DEFINE_NN(sg_nn_row,  1,        SG_NN_NV)
SG_DEFINE_NN(sg_nn_v1,   SG_NN_MR, 1)
SG_DEFINE_NN(sg_nn_v1r,  1,        1)

static void sg_nn_tail(size_t rows, size_t cols, size_t k, const float *a,
                       size_t lda, const float *b, size_t ldb, float alpha,
                       float beta, float *c, size_t ldc, int resume, int finish) {
    for (size_t r = 0; r < rows; r++)
        for (size_t j = 0; j < cols; j++) {
            float s = resume ? c[r * ldc + j] : 0.0f;
            for (size_t p = 0; p < k; p++) s = fmaf(a[r * lda + p], b[p * ldb + j], s);
            if (finish) sg_put(c + r * ldc + j, s, alpha, beta);
            else        c[r * ldc + j] = s;
        }
}

/* ── tiles and the job ─────────────────────────────────────────────────────*/
typedef struct {
    int trans_b;
    size_t m, n, k, lda, ldb, ldc;
    float alpha, beta;
    const float *a, *b;
    float *c;
    size_t rb, cb;          /* rows / cols per task                         */
    size_t grid_n;          /* column blocks                                */
} sg_job;

/* Cache budgets, in bytes. COST-MODEL ESTIMATES, not sweeps: half of the
 * smallest private L2 among the cores we target (1 MiB on Cascade Lake and
 * Neoverse V2; Apple's shared L2 is larger). Re-derive on the target hosts.
 *
 * Neither changes a single bit of the result: a row chunk only reorders
 * which tile runs first, and a k-block parks the raw f32 accumulator in C
 * and reloads it, which is exact — the same fma chain continues on the same
 * float. tests/test_sgemm.c asserts both with memcmp. */
#define SG_NT_A_BYTES (256u * 1024u)   /* rows of A kept hot across the column tiles */
#define SG_NN_KC      256u             /* k per block: a 256 x (NV*L) B panel */

static void sg_nt_rows(const sg_job *j, size_t i0, size_t i1, size_t j0, size_t j1) {
    size_t jj = j0;
    for (; jj + SG_NT_NR <= j1; jj += SG_NT_NR) {
        const float *bp = j->b + jj * j->ldb;
        size_t ii = i0;
        for (; ii + SG_NT_MR <= i1; ii += SG_NT_MR)
            sg_nt_full(j->k, j->a + ii * j->lda, j->lda, bp, j->ldb,
                       j->alpha, j->beta, j->c + ii * j->ldc + jj, j->ldc);
        for (; ii < i1; ii++)
            sg_nt_row(j->k, j->a + ii * j->lda, j->lda, bp, j->ldb,
                      j->alpha, j->beta, j->c + ii * j->ldc + jj, j->ldc);
    }
    for (; jj < j1; jj++) {
        const float *bp = j->b + jj * j->ldb;
        size_t ii = i0;
        for (; ii + SG_NT_MR <= i1; ii += SG_NT_MR)
            sg_nt_col(j->k, j->a + ii * j->lda, j->lda, bp, j->ldb,
                      j->alpha, j->beta, j->c + ii * j->ldc + jj, j->ldc);
        for (; ii < i1; ii++)
            sg_nt_one(j->k, j->a + ii * j->lda, j->lda, bp, j->ldb,
                      j->alpha, j->beta, j->c + ii * j->ldc + jj, j->ldc);
    }
}

static void sg_block_nt(const sg_job *j, size_t i0, size_t i1, size_t j0, size_t j1) {
    /* Column tiles inside a ROW CHUNK: the chunk of A stays in L2 while every
     * column tile of B sweeps past it. Without the chunk, a 256-token batch
     * at K = 1024 (1 MiB of A) is re-streamed from L3 once per column tile —
     * 32 times for a 128-row weight strip. */
    size_t chunk = SG_NT_A_BYTES / (j->k * sizeof(float));
    chunk -= chunk % SG_NT_MR;
    if (chunk < SG_NT_MR) chunk = SG_NT_MR;
    for (size_t ic = i0; ic < i1; ic += chunk)
        sg_nt_rows(j, ic, (ic + chunk < i1) ? ic + chunk : i1, j0, j1);
}

static void sg_nn_pass(const sg_job *j, size_t i0, size_t i1, size_t j0, size_t j1,
                       size_t k0, size_t kc, int resume, int finish) {
    const size_t wide = (size_t)SG_NN_NV * SG_L;
    const float *b = j->b + k0 * j->ldb;
    size_t ii = i0;
    while (ii < i1) {
        const size_t rows = (ii + SG_NN_MR <= i1) ? SG_NN_MR : 1;
        const float *ap = j->a + ii * j->lda + k0;
        float *cp = j->c + ii * j->ldc;
        size_t jj = j0;
        for (; jj + wide <= j1; jj += wide) {
            if (rows == SG_NN_MR) sg_nn_full(kc, ap, j->lda, b + jj, j->ldb, j->alpha, j->beta, cp + jj, j->ldc, resume, finish);
            else                  sg_nn_row (kc, ap, j->lda, b + jj, j->ldb, j->alpha, j->beta, cp + jj, j->ldc, resume, finish);
        }
        for (; jj + SG_L <= j1; jj += SG_L) {
            if (rows == SG_NN_MR) sg_nn_v1 (kc, ap, j->lda, b + jj, j->ldb, j->alpha, j->beta, cp + jj, j->ldc, resume, finish);
            else                  sg_nn_v1r(kc, ap, j->lda, b + jj, j->ldb, j->alpha, j->beta, cp + jj, j->ldc, resume, finish);
        }
        if (jj < j1)
            sg_nn_tail(rows, j1 - jj, kc, ap, j->lda, b + jj, j->ldb,
                       j->alpha, j->beta, cp + jj, j->ldc, resume, finish);
        ii += rows;
    }
}

static void sg_block_nn(const sg_job *j, size_t i0, size_t i1, size_t j0, size_t j1) {
    /* k-blocked when C may hold the raw partial sum, i.e. beta == 0 (every
     * call site). S.V at n_kv = 2048 otherwise streams a 2048-row V panel
     * from L3 once per 4 query rows. With beta != 0, C's old value is still
     * needed at the end, so the unblocked pass runs. */
    if (j->beta != 0.0f || j->k <= SG_NN_KC) {
        sg_nn_pass(j, i0, i1, j0, j1, 0, j->k, 0, 1);
        return;
    }
    for (size_t k0 = 0; k0 < j->k; k0 += SG_NN_KC) {
        const size_t kc = (k0 + SG_NN_KC < j->k) ? SG_NN_KC : j->k - k0;
        sg_nn_pass(j, i0, i1, j0, j1, k0, kc, k0 != 0, k0 + kc == j->k);
    }
}

static void sg_task(void *ctx, int t) {
    const sg_job *j = ctx;
    const size_t bi = (size_t)t / j->grid_n, bj = (size_t)t % j->grid_n;
    const size_t i0 = bi * j->rb, j0 = bj * j->cb;
    if (i0 >= j->m || j0 >= j->n) return;
    const size_t i1 = (i0 + j->rb < j->m) ? i0 + j->rb : j->m;
    const size_t j1 = (j0 + j->cb < j->n) ? j0 + j->cb : j->n;
    if (j->trans_b) sg_block_nt(j, i0, i1, j0, j1);
    else            sg_block_nn(j, i0, i1, j0, j1);
}

static size_t round_up(size_t x, size_t q) { return (x + q - 1) / q * q; }

int mynah_slm_sgemm_own(int trans_b, size_t m, size_t n, size_t k, float alpha,
                        const float *a, size_t lda, const float *b, size_t ldb,
                        float beta, float *c, size_t ldc) {
    bump(&g_calls);
    if (m == 0 || n == 0) return 0;
    if (!c || ldc < n || (k && (!a || !b || lda < k || ldb < (trans_b ? k : n)))) {
        bump(&g_refused);
        return -1;
    }
    if (k == 0) {
        bump(&g_ref);
        mynah_slm_sgemm_reference(trans_b, m, n, 0, alpha, a, lda, b, ldb, beta, c, ldc);
        return 0;
    }
    bump(trans_b ? &g_nt : &g_nn);

    sg_job j = { trans_b, m, n, k, lda, ldb, ldc, alpha, beta, a, b, c, m, n, 1 };

    /* The plan. Because every element is computed identically whatever tile
     * holds it, the plan is free to depend on the thread count — unlike
     * mynah-tts, where only the row axis may. Split the longer axis first:
     * qmatmat has n = 128 weight rows and m = T tokens, QK^T has n = n_kv. */
    const int nth = mynah_slm_threads_count();
    const double work = (double)m * (double)n * (double)k;
    size_t tasks = 1;
    if (nth > 1 && work >= SG_PARALLEL_MIN_WORK) tasks = (size_t)nth * 2;

    const size_t qm = trans_b ? SG_NT_MR : SG_NN_MR;
    const size_t qn = trans_b ? SG_NT_NR : (size_t)SG_NN_NV * SG_L;
    size_t gm = 1, gn = 1;
    while (gm * gn < tasks) {
        const size_t can_m = (m + qm - 1) / qm, can_n = (n + qn - 1) / qn;
        const int grow_n = (n / gn >= m / gm && gn < can_n) || gm >= can_m;
        if (grow_n) { if (gn >= can_n) break; gn++; }
        else        { gm++; }
    }
    j.rb = round_up((m + gm - 1) / gm, qm);
    j.cb = round_up((n + gn - 1) / gn, qn);
    j.grid_n = (n + j.cb - 1) / j.cb;
    const size_t grid_m = (m + j.rb - 1) / j.rb;
    const size_t n_tasks = grid_m * j.grid_n;

    if (n_tasks <= 1) sg_task(&j, 0);
    else mynah_slm_parallel_for((int)n_tasks, sg_task, &j);
    return 0;
}

/* ── the entry point ───────────────────────────────────────────────────────*/
static int g_use_own = -1;

static int use_own(void) {
    if (g_use_own < 0) {
#if defined(SG_VENDOR)
        const char *e = getenv("MYNAH_SLM_SGEMM");
        g_use_own = (e && strcmp(e, "own") == 0) ? 1 : 0;
#else
        g_use_own = 1;
#endif
    }
    return g_use_own;
}

const char *mynah_slm_sgemm_backend(void) {
#if defined(SG_VENDOR)
    return use_own() ? "own" : SG_VENDOR;
#else
    return "own";
#endif
}

int mynah_slm_sgemm(int trans_b, size_t m, size_t n, size_t k, float alpha,
                    const float *a, size_t lda, const float *b, size_t ldb,
                    float beta, float *c, size_t ldc) {
    if (use_own())
        return mynah_slm_sgemm_own(trans_b, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
#if defined(SG_VENDOR)
    bump(&g_calls);
    bump(&g_vendor);
    /* Called from ONE thread with our pool idle, never from inside a parallel
     * region: the vendor brings its own threads, and two pools over the same
     * cores is the collapse mynah-asr measured. */
    cblas_sgemm(CblasRowMajor, CblasNoTrans, trans_b ? CblasTrans : CblasNoTrans,
                (int)m, (int)n, (int)k, alpha, a, (int)lda, b, (int)ldb,
                beta, c, (int)ldc);
    return 0;
#else
    return -1;
#endif
}
