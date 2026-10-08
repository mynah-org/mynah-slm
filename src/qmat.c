/* qmat.c — see qmat.h.
 * SPDX-License-Identifier: MIT */
#include "qmat.h"

#include "kern.h"
#include "sgemm.h"
#include "threads.h"

#include "ingot/dtype.h"
#include "ingot/quant.h"

#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

/* The kernels themselves live in qmat_kern.c, compiled once per ISA; this
 * file is the API, the activation prep and the dispatch through the table
 * src/isa.c resolved (kern.h). */

/* The three switches below come from the environment, read ONCE: getenv in a
 * matvec called ~200 times per token would be its own measurement problem.
 *
 * Once means pthread_once, not "if (g < 0) g = ...": the first matvec of a
 * process can run inside the pool's workers, and several of them resolving
 * the same plain int at once is a data race (ThreadSanitizer reported it,
 * review R2) — benign in practice, undefined in C. After the once, the values
 * are relaxed atomics: the setters below exist for tests and benches that
 * flip a switch between calls, with the pool idle, and a relaxed load is a
 * plain load on every target we build. */
enum { INT8_Q4_K = MYNAH_SLM_INT8_Q4_K, INT8_Q8_0 = MYNAH_SLM_INT8_Q8_0,
       INT8_Q6_K = MYNAH_SLM_INT8_Q6_K, INT8_ALL = 7 };

static pthread_once_t g_env_once = PTHREAD_ONCE_INIT;
static _Atomic int g_own = 1;           /* our kernels (0: MYNAH_SLM_KERNELS=ingot) */
static _Atomic int g_int8 = 0;          /* int8 requested (MYNAH_SLM_INT8 / --fast) */
static _Atomic int g_int8_types = INT8_ALL;

static int parse_int8_types(const char *e);

static void env_init(void) {
    const char *e = getenv("MYNAH_SLM_KERNELS");
    atomic_store_explicit(&g_own, (e && strcmp(e, "ingot") == 0) ? 0 : 1,
                          memory_order_relaxed);
    /* Off unless asked for. It trades accuracy for speed, and a default that
     * quietly does that is how a quantization claim stops meaning anything.
     * `mynah-slm ppl` is what decides, not this. */
    e = getenv("MYNAH_SLM_INT8");
    atomic_store_explicit(&g_int8, (e && strcmp(e, "0") != 0) ? 1 : 0,
                          memory_order_relaxed);
    atomic_store_explicit(&g_int8_types, parse_int8_types(getenv("MYNAH_SLM_INT8_TYPES")),
                          memory_order_relaxed);
}

static void env_resolve(void) { pthread_once(&g_env_once, env_init); }

static int env_get(_Atomic int *v) {
    env_resolve();
    return atomic_load_explicit(v, memory_order_relaxed);
}

/* A setter resolves first, so a later lazy init cannot overwrite it. */
static void env_set(_Atomic int *v, int value) {
    env_resolve();
    atomic_store_explicit(v, value, memory_order_relaxed);
}

static int use_own_kernels(void) { return env_get(&g_own); }

void mynah_slm_matvec_set_enabled(int on) { env_set(&g_own, on ? 1 : 0); }

/* Every kernel exists in a vector form per ISA and a scalar reference. The
 * scalar one is not a fallback nobody runs — it is the definition the others
 * have to agree with, and it is what a machine without the instructions gets.
 *
 * x86 cannot be executed natively on the M1, so it is verified two ways there:
 * `make check-x86` cross-compiles every x86 kernel TU, and `make
 * test-x86-rosetta` builds the suite as x86_64 and RUNS it under Rosetta.
 * `mynah-slm --dispatch` says which tables actually resolved. */

/* Requested (MYNAH_SLM_INT8 / --fast) AND the resolved ISA has vector int8
 * kernels. On a CPU without them the request is not honoured, and
 * `--dispatch` says so: int8 through a scalar twin would be slower than the
 * f32 path it replaces. */
int mynah_slm_matvec_int8_enabled(void) {
    return env_get(&g_int8) && mynah_slm_kern_qmat()->int8;
}

int mynah_slm_matvec_int8_requested(void) { return env_get(&g_int8) > 0; }

void mynah_slm_matvec_set_int8(int on) { env_set(&g_int8, on ? 1 : 0); }

/* Which types the int8 switch applies to: MYNAH_SLM_INT8_TYPES, a comma list
 * of q4_k / q8_0 / q6_k, default all three. It can only NARROW the switch,
 * never turn int8 on by itself. It exists so the perplexity gate can price
 * each type on ONE binary — the Q6_K head in front of the softmax is a
 * different quality question from the Q4_K layers (.work/int8-q8_0-q6_k.md). */
/*
 * Exact tokens, not substrings: "q4_k_m" or "noq6_k" used to match, and a
 * typo silently selected nothing. Case-insensitive, blanks around a token
 * ignored, an unknown token reported (once: the environment is read once).
 * An empty or unset value means all three; a value with no known token means
 * none — narrow is the safe direction. */
int mynah_slm_matvec_int8_types_parse(const char *e, FILE *warn) {
    if (!e || !*e) return INT8_ALL;
    static const struct { const char *name; int bit; } known[] = {
        { "q4_k", INT8_Q4_K }, { "q8_0", INT8_Q8_0 }, { "q6_k", INT8_Q6_K },
    };
    int mask = 0;
    const char *p = e;
    for (;;) {
        const char *end = strchr(p, ',');
        if (!end) end = p + strlen(p);
        const char *a = p, *b = end;
        while (a < b && (*a == ' ' || *a == '\t')) a++;
        while (b > a && (b[-1] == ' ' || b[-1] == '\t')) b--;
        const size_t n = (size_t)(b - a);
        int hit = 0;
        for (size_t k = 0; k < sizeof known / sizeof *known; k++)
            if (n == strlen(known[k].name) && strncasecmp(a, known[k].name, n) == 0) {
                mask |= known[k].bit;
                hit = 1;
            }
        if (!hit && n > 0 && warn)
            fprintf(warn, "mynah-slm: MYNAH_SLM_INT8_TYPES: unknown type \"%.*s\" ignored "
                          "(known: q4_k, q8_0, q6_k)\n", (int)n, a);
        if (!*end) break;
        p = end + 1;
    }
    return mask;
}

static int parse_int8_types(const char *e) {
    return mynah_slm_matvec_int8_types_parse(e, stderr);
}

int mynah_slm_matvec_int8_types(void) { return env_get(&g_int8_types); }

static int int8_type_bit(int type) {
    return type == INGOT_TYPE_Q4_K ? INT8_Q4_K :
           type == INGOT_TYPE_Q8_0 ? INT8_Q8_0 :
           type == INGOT_TYPE_Q6_K ? INT8_Q6_K : 0;
}

static int int8_type_on(int type) {
    return (env_get(&g_int8_types) & int8_type_bit(type)) != 0;
}

void mynah_slm_matvec_set_int8_types(int q4_k, int q8_0, int q6_k) {
    env_set(&g_int8_types,
            (q4_k ? INT8_Q4_K : 0) | (q8_0 ? INT8_Q8_0 : 0) | (q6_k ? INT8_Q6_K : 0));
}

int mynah_slm_matvec_have(int type) {
    if (!use_own_kernels()) return 0;
    if (type == INGOT_TYPE_Q4_K) return 1;
    /* Q8_0 and Q6_K: ours only as int8 (K4). Their f32 kernels are ingot's,
     * and docs/perf.md is why — ours tied or lost and went upstream. */
    return (type == INGOT_TYPE_Q8_0 || type == INGOT_TYPE_Q6_K) &&
           mynah_slm_matvec_int8_enabled() && int8_type_on(type);
}

/* The smallest block maximum the int8 quantizer scales: 127 / 2^-120 is
 * ~1.7e38, still finite. FLT_MIN would not do — 127 / FLT_MIN overflows. */
#define MYNAH_SLM_XQ_AMAX_MIN 0x1p-120f

static void matvec_prepare(const float *input, size_t cols,
                           mynah_slm_matvec_in *prep, int want_int8) {
    if (!prep) return;
    /* The arrays are fixed-size: a wider vector is refused, not overrun, and
     * `cols` says what was prepared so a product can check it (review N7). */
    prep->cols = 0;
    prep->have_int8 = 0;
    if (!input || cols / 32 > MYNAH_SLM_XSUM_MAX) return;
    prep->cols = cols;
    prep->have_int8 = want_int8 && cols <= MYNAH_SLM_XQ_MAX;
    int finite = 1;

    for (size_t s = 0; s < cols / 32; s++) {
        const float *x = input + s * 32;
        float acc = 0.0f, amax = 0.0f;
        for (int i = 0; i < 32; i++) {
            acc += x[i];
            const float a = fabsf(x[i]);
            if (a > amax) amax = a;
        }
        prep->xsum[s] = acc;
        if (!isfinite(acc)) finite = 0;     /* a NaN or inf in the block */

        if (prep->have_int8) {
            int8_t *q = prep->xq + s * 32;
            /* A block that is all zeros, too small for 127/amax to be
             * finite, or not finite at all quantizes to zero with a zero
             * scale. The old form took inv = 1/(amax/127): for amax under
             * ~3.7e-37 the scale went subnormal and inv +inf, every nonzero
             * value saturated to -128/127 and 0*inf cast a NaN to int8 (UB).
             * -128 then broke AVX2's sign(xq, w), which assumes |xq| <= 127.
             * Such a block contributes < 1e-35 per unit weight: zero is the
             * honest answer, and it is the same answer on every ISA. */
            if (!(amax >= MYNAH_SLM_XQ_AMAX_MIN) || !isfinite(acc)) {
                prep->xscale[s] = 0.0f;
                memset(q, 0, 32);
                continue;
            }
            prep->xscale[s] = amax / 127.0f;
            const float inv = 127.0f / amax;      /* finite: amax >= 2^-120 */
            for (int i = 0; i < 32; i++) {
                float v = nearbyintf(x[i] * inv);
                /* |x * inv| <= 127 up to one rounding; clamp SYMMETRICALLY
                 * so -128 can never be emitted (the AVX2 Q8_0 kernel and
                 * the 128*127*2 int16 bound both rely on it). */
                if (v >  127.0f) v =  127.0f;
                if (v < -127.0f) v = -127.0f;
                q[i] = (int8_t)v;
            }
        }
    }

    /* A non-finite activation must stay visible. int8 cannot carry it: the
     * block above became zeros, and Q8_0/Q6_K (which never read xsum) would
     * return a FINITE dot from a NaN input — a broken layer that samples
     * normally. So the whole vector takes the f32 path, where ingot's and our
     * f32 kernels propagate it. Costs nothing on finite input: the check is
     * on the sums already computed. */
    if (!finite) prep->have_int8 = 0;
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
    if (!weights || !prep || !output || !prep->have_int8 || prep->cols != cols || cols % 256 != 0)
        return -1;
    mynah_slm_qmat_kern_scalar.q4k_i8((const unsigned char *)weights, rows, cols / 256,
                                      prep->xq, prep->xscale, prep->xsum, output);
    return 0;
}

int mynah_slm_q80_int8_ref(const void *weights, size_t rows, size_t cols,
                           const mynah_slm_matvec_in *prep, float *output) {
    if (!weights || !prep || !output || !prep->have_int8 || prep->cols != cols || cols % 32 != 0)
        return -1;
    mynah_slm_qmat_kern_scalar.q80_i8((const unsigned char *)weights, rows, cols / 32,
                                      prep->xq, prep->xscale, output);
    return 0;
}

int mynah_slm_q6k_int8_ref(const void *weights, size_t rows, size_t cols,
                           const mynah_slm_matvec_in *prep, float *output) {
    if (!weights || !prep || !output || !prep->have_int8 || prep->cols != cols || cols % 256 != 0)
        return -1;
    mynah_slm_qmat_kern_scalar.q6k_i8((const unsigned char *)weights, rows, cols / 256,
                                      prep->xq, prep->xscale, output);
    return 0;
}

int mynah_slm_matvec(int type, const void *weights, size_t rows, size_t cols,
                     const float *input, const mynah_slm_matvec_in *prep,
                     float *output) {
    if (!prep || prep->cols != cols || !use_own_kernels()) return -1;
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
