/* bench_qmat.c — kernel A/B on synthetic matrices of the real shapes.
 *
 * tests/bench_matvec.c needs a checkpoint. This does not: it builds the
 * Qwen3-0.6B shapes from tests/qfixture.h, so a kernel change can be measured
 * on a machine with no weights at all — and then has to be measured again on
 * one with weights, because a synthetic matrix says nothing about a decode
 * step. The output says so on every line.
 *
 * Method (.work/engineering-method.md §5): A and B INTERLEAVED in one process,
 * the order alternating every round, same data, same thread pool, same row
 * split as mynah_slm_project (four chunks per thread). Each side is timed over
 * a block of calls sized to ~30 ms; the per-call time of each round is one
 * sample. Reported: median, min..max, coefficient of variation, and how many
 * rounds B beat A in the SAME round. A control row runs one kernel against
 * itself: if it does not land on 1.00x, the machine is not quiet enough for
 * the other rows to mean anything.
 *
 *   make bench-qmat                         (all suites, 1 and 4 threads)
 *   tests/bench_qmat k3|k4|k5|k6 [rounds] [threads]  e.g. tests/bench_qmat k3 21 4
 *
 * SPDX-License-Identifier: MIT */
#include "kern.h"
#include "kernels.h"
#include "qfixture.h"
#include "qmat.h"
#include "threads.h"
#include "timing.h"

#include "ingot/dtype.h"
#include "ingot/quant.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#endif
#if defined(__AVX2__)
#include <immintrin.h>
#endif

/* ── the BEFORE side of K3, verbatim from src/qmat.c at 0860c56 ─────────────
 * Kept here, not in src/, because it is a measurement baseline and nothing
 * else: the engine must not carry a second kernel for the sake of a bench.
 * Delete it with the K3 note's evidence when nobody needs to re-run that A/B. */
#if (defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD)) || defined(__AVX2__)
#define BENCH_HAVE_OLD_INT8 1
static float old_f16(const unsigned char *p) {
    const uint16_t h = (uint16_t)(p[0] | ((uint16_t)p[1] << 8));
    return ingot_f16_to_f32(h);
}

static void old_scale_min(const unsigned char *scales, int index,
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

#if defined(__AVX2__)
static inline int old_hsum256i(__m256i v) {
    __m128i a = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
    a = _mm_add_epi32(a, _mm_shuffle_epi32(a, 0x4e));
    a = _mm_add_epi32(a, _mm_shuffle_epi32(a, 0xb1));
    return _mm_cvtsi128_si32(a);
}
#if defined(__AVX512VNNI__) && defined(__AVX512BW__) && defined(__AVX512F__)
#define OLD_VNNI 1
static inline void old_pair_vnni(const unsigned char *q, const int8_t *xbase,
                                 int *sum_lo, int *sum_hi) {
    const __m256i p  = _mm256_loadu_si256((const __m256i *)(const void *)q);
    const __m256i m  = _mm256_set1_epi8(0x0f);
    const __m256i nl = _mm256_and_si256(p, m);
    const __m256i nh = _mm256_and_si256(_mm256_srli_epi16(p, 4), m);
    const __m512i w  = _mm512_inserti64x4(_mm512_castsi256_si512(nl), nh, 1);
    const __m512i x  = _mm512_loadu_si512((const void *)xbase);
    const __m512i acc = _mm512_dpbusd_epi32(_mm512_setzero_si512(), w, x);
    *sum_lo = old_hsum256i(_mm512_extracti32x8_epi32(acc, 0));
    *sum_hi = old_hsum256i(_mm512_extracti32x8_epi32(acc, 1));
}
#endif
#endif

static float old_q4_k_row_int8(const unsigned char *row, size_t blocks,
                               const int8_t *xq, const float *xscale,
                               const float *xsum) {
    float total = 0.0f, mins = 0.0f;

    for (size_t b = 0; b < blocks; b++) {
        const unsigned char *block = row + b * 144;
        const float d    = old_f16(block);
        const float dmin = old_f16(block + 2);
        const unsigned char *scales = block + 4;
        const unsigned char *q      = block + 16;
        const size_t sub0 = b * 8;

        for (int base = 0, si = 0; base < 256; base += 64, si += 2) {
            unsigned char sc0, mn0, sc1, mn1;
            old_scale_min(scales, si,     &sc0, &mn0);
            old_scale_min(scales, si + 1, &sc1, &mn1);

            const int8_t *xlo = xq + b * 256 + base;
            const int8_t *xhi = xlo + 32;
            int sum_lo, sum_hi;
#if defined(__ARM_NEON)
            int32x4_t alo = vdupq_n_s32(0), ahi = vdupq_n_s32(0);
            for (int i = 0; i < 32; i += 16) {
                const uint8x16_t p = vld1q_u8(q + i);
                alo = vdotq_s32(alo, vreinterpretq_s8_u8(vandq_u8(p, vdupq_n_u8(0x0f))),
                                vld1q_s8(xlo + i));
                ahi = vdotq_s32(ahi, vreinterpretq_s8_u8(vshrq_n_u8(p, 4)),
                                vld1q_s8(xhi + i));
            }
            sum_lo = vaddvq_s32(alo);
            sum_hi = vaddvq_s32(ahi);
#elif defined(OLD_VNNI)
            (void)xhi;
            old_pair_vnni(q, xlo, &sum_lo, &sum_hi);
#else
            const __m256i ones = _mm256_set1_epi16(1);
            const __m256i p = _mm256_loadu_si256((const __m256i *)(const void *)q);
            const __m256i nl = _mm256_and_si256(p, _mm256_set1_epi8(0x0f));
            const __m256i nh = _mm256_and_si256(_mm256_srli_epi16(p, 4),
                                                _mm256_set1_epi8(0x0f));
            const __m256i vl = _mm256_loadu_si256((const __m256i *)(const void *)xlo);
            const __m256i vh = _mm256_loadu_si256((const __m256i *)(const void *)xhi);
            sum_lo = old_hsum256i(_mm256_madd_epi16(_mm256_maddubs_epi16(nl, vl), ones));
            sum_hi = old_hsum256i(_mm256_madd_epi16(_mm256_maddubs_epi16(nh, vh), ones));
#endif

            const size_t s0 = sub0 + (size_t)base / 32;
            total += d * (float)sc0 * xscale[s0]     * (float)sum_lo;
            total += d * (float)sc1 * xscale[s0 + 1] * (float)sum_hi;
            mins  += dmin * ((float)mn0 * xsum[s0] + (float)mn1 * xsum[s0 + 1]);
            q += 32;
        }
    }
    return total - mins;
}
#endif

/* ── the sides ───────────────────────────────────────────────────────────── */

typedef struct {
    int type;
    size_t cols, row_bytes;
    const float *x;
    const mynah_slm_matvec_in *prep;
} bctx;

typedef int (*rows_fn)(const bctx *c, const unsigned char *w, size_t n, float *out);

#if defined(BENCH_HAVE_OLD_INT8)
static int side_old_q4k_int8(const bctx *c, const unsigned char *w, size_t n, float *out) {
    for (size_t r = 0; r < n; r++)
        out[r] = old_q4_k_row_int8(w + r * c->row_bytes, c->cols / 256,
                                   c->prep->xq, c->prep->xscale, c->prep->xsum);
    return 0;
}
#endif

/* ours, whatever path `prep` selects (int8 when prep->have_int8) */
static int side_ours(const bctx *c, const unsigned char *w, size_t n, float *out) {
    return mynah_slm_matvec(c->type, w, n, c->cols, c->x, c->prep, out);
}

typedef struct {
    rows_fn fn;
    const bctx *c;
    const unsigned char *w;
    size_t rows, per;
    float *out;
    int rc;
} drv;

static void drv_chunk(void *ctx, int i) {
    drv *d = ctx;
    const size_t first = (size_t)i * d->per;
    if (first >= d->rows) return;
    size_t n = d->per;
    if (first + n > d->rows) n = d->rows - first;
    if (d->fn(d->c, d->w + first * d->c->row_bytes, n, d->out + first) != 0) d->rc = -1;
}

/* The row split of mynah_slm_project: four chunks per thread. */
static int run(rows_fn fn, const bctx *c, const unsigned char *w, size_t rows, float *out) {
    const int nth = mynah_slm_threads_count();
    if (nth <= 1) return fn(c, w, rows, out);
    int chunks = nth * 4;
    if ((size_t)chunks > rows) chunks = (int)rows;
    drv d = { fn, c, w, rows, (rows + (size_t)chunks - 1) / (size_t)chunks, out, 0 };
    mynah_slm_parallel_for(chunks, drv_chunk, &d);
    return d.rc;
}

/* ── statistics ──────────────────────────────────────────────────────────── */

static int cmp_d(const void *a, const void *b) {
    const double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y;
}

typedef struct { double med, min, max, cv; } stats;

static stats stats_of(const double *v, int n) {
    double s[256];
    memcpy(s, v, (size_t)n * sizeof *v);
    qsort(s, (size_t)n, sizeof *s, cmp_d);
    double mean = 0.0, var = 0.0;
    for (int i = 0; i < n; i++) mean += s[i];
    mean /= n;
    for (int i = 0; i < n; i++) var += (s[i] - mean) * (s[i] - mean);
    stats st = { n % 2 ? s[n / 2] : 0.5 * (s[n / 2 - 1] + s[n / 2]), s[0], s[n - 1],
                 mean > 0.0 ? sqrt(var / n) / mean : 0.0 };
    return st;
}

/* One A/B row. Returns 0, or -1 when a side failed or the outputs disagree
 * beyond `tol` (a bench of a wrong kernel is not a measurement). */
static int ab(const char *label, const char *a_name, rows_fn a, const char *b_name,
              rows_fn b, const bctx *c, const unsigned char *w, size_t rows,
              int rounds, double tol) {
    float *oa = mynah_slm_aligned_alloc(rows * sizeof *oa);
    float *ob = mynah_slm_aligned_alloc(rows * sizeof *ob);
    double ta[256], tb[256];
    if (!oa || !ob || rounds > 256) { printf("FAIL alloc\n"); return -1; }

    for (int k = 0; k < 2; k++)
        if (run(a, c, w, rows, oa) != 0 || run(b, c, w, rows, ob) != 0) {
            printf("FAIL %s: a side declined\n", label);
            return -1;
        }
    double worst = 0.0, scale = 0.0;
    for (size_t r = 0; r < rows; r++) {
        const double d = fabs((double)oa[r] - (double)ob[r]);
        if (d > worst) worst = d;
        if (fabs((double)oa[r]) > scale) scale = fabs((double)oa[r]);
    }
    const double rel = scale > 0.0 ? worst / scale : worst;
    if (rel > tol) {
        printf("FAIL %s: outputs disagree, rel %.2e > %.0e\n", label, rel, tol);
        return -1;
    }

    /* reps per timed block: ~30 ms */
    double t0 = mynah_slm_now();
    run(b, c, w, rows, ob);
    const double one = mynah_slm_now() - t0;
    int reps = one > 0.0 ? (int)(0.03 / one) : 100;
    if (reps < 1) reps = 1;

    int wins = 0;
    for (int r = 0; r < rounds; r++) {
        for (int s = 0; s < 2; s++) {
            const int side_b = (r % 2 == 0) ? s : !s;     /* alternate the order */
            rows_fn fn = side_b ? b : a;
            float *o = side_b ? ob : oa;
            t0 = mynah_slm_now();
            for (int i = 0; i < reps; i++) run(fn, c, w, rows, o);
            const double dt = (mynah_slm_now() - t0) / reps;
            (side_b ? tb : ta)[r] = dt;
        }
        if (tb[r] < ta[r]) wins++;
    }
    const stats sa = stats_of(ta, rounds), sb = stats_of(tb, rounds);
    const double ratio = sa.med / sb.med;
    const int stable_dir = wins * 10 >= rounds * 9 || wins * 10 <= rounds;
    const int stable_mag = sa.cv < 0.05 && sb.cv < 0.05;
    printf("| %-22s | %d | %s %.3f (%.3f..%.3f, cv %2.0f%%) | %s %.3f (%.3f..%.3f, cv %2.0f%%) "
           "| %.2fx | %d/%d | %s | %s | %.1e |\n",
           label, mynah_slm_threads_count(),
           a_name, sa.med * 1e3, sa.min * 1e3, sa.max * 1e3, sa.cv * 100.0,
           b_name, sb.med * 1e3, sb.min * 1e3, sb.max * 1e3, sb.cv * 100.0,
           ratio, wins, rounds, stable_dir ? "stable" : "NOT stable",
           stable_mag ? "yes" : "no", rel);
    fflush(stdout);
    mynah_slm_aligned_free(oa);
    mynah_slm_aligned_free(ob);
    return 0;
}

typedef struct { const char *name; size_t rows, cols; } shape;

static const shape k_shapes[] = {
    { "attn_k/v 1024x1024", 1024, 1024 },
    { "ffn_gate/up 3072x1024", 3072, 1024 },
    { "ffn_down 1024x3072", 1024, 3072 },
    { "lm_head 151936x1024", 151936, 1024 },
};

static void header(void) {
    printf("| shape | thr | A ms (min..max, cv) | B ms (min..max, cv) | A/B | B wins "
           "| direction | magnitude | rel diff |\n");
    printf("|---|---|---|---|---|---|---|---|---|\n");
}

/* K3: the int8 Q4_K kernel, before vs after. */
static int suite_k3(int rounds) {
#if defined(BENCH_HAVE_OLD_INT8)
    printf("\nK3 — Q4_K int8 matvec: A = old kernel (0860c56), B = new (%s)\n",
           mynah_slm_matvec_int8_isa());
    header();
    int rc = 0;
    for (size_t s = 0; s < sizeof k_shapes / sizeof *k_shapes; s++) {
        const shape *sh = &k_shapes[s];
        const size_t row_bytes = sh->cols / 256 * 144;
        unsigned char *w = mynah_slm_aligned_alloc(sh->rows * row_bytes);
        float *x = mynah_slm_aligned_alloc(sh->cols * sizeof *x);
        mynah_slm_matvec_in *prep = mynah_slm_aligned_alloc(sizeof *prep);
        if (!w || !x || !prep) { printf("FAIL alloc\n"); return -1; }
        qfx_fill(INGOT_TYPE_Q4_K, w, sh->rows, sh->cols, 11 + s);
        qfx_activations(x, sh->cols, 5 + s);
        mynah_slm_matvec_prepare_int8(x, sh->cols, prep);
        const bctx c = { INGOT_TYPE_Q4_K, sh->cols, row_bytes, x, prep };
        /* the old and new kernels round differently; 1e-5 catches a broken
         * one and passes a reorder */
        rc |= ab(sh->name, "old", side_old_q4k_int8, "new", side_ours, &c, w, sh->rows,
                 rounds, 1e-5);
        if (s == 0)
            rc |= ab("CONTROL new vs new", "new", side_ours, "new", side_ours, &c, w,
                     sh->rows, rounds, 0.0);
        mynah_slm_aligned_free(w);
        mynah_slm_aligned_free(x);
        mynah_slm_aligned_free(prep);
    }
    return rc;
#else
    (void)rounds;
    printf("\nK3: SKIP — no int8 kernel in this build\n");
    return 0;
#endif
}

static int side_ingot(const bctx *c, const unsigned char *w, size_t n, float *out) {
    return ingot_matvec(c->type, w, n, c->cols, c->x, out);
}

/* K4: Q8_0 and Q6_K. A = ingot's f32 kernel (what runs today, with or without
 * --fast), B = our int8 kernel. Different numerics on purpose: this is the
 * decision "does turning int8 on for this type pay", so the outputs only have
 * to agree to the int8 budget. */
static int suite_k4(int rounds) {
    if (strcmp(mynah_slm_matvec_int8_isa(), "none") == 0) {
        printf("\nK4: SKIP — no int8 kernel in this build\n");
        return 0;
    }
    int rc = 0;
    const int types[] = { INGOT_TYPE_Q8_0, INGOT_TYPE_Q6_K };
    for (size_t t = 0; t < sizeof types / sizeof *types; t++) {
        const int type = types[t];
        size_t elems = 0, bytes = 0;
        qfx_geometry(type, &elems, &bytes);
        printf("\nK4 — %s matvec: A = ingot f32 activations, B = ours int8 (%s)\n",
               ingot_type_name(type), mynah_slm_matvec_int8_isa());
        header();
        for (size_t s = 0; s < sizeof k_shapes / sizeof *k_shapes; s++) {
            const shape *sh = &k_shapes[s];
            const size_t row_bytes = sh->cols / elems * bytes;
            unsigned char *w = mynah_slm_aligned_alloc(sh->rows * row_bytes);
            float *x = mynah_slm_aligned_alloc(sh->cols * sizeof *x);
            mynah_slm_matvec_in *prep = mynah_slm_aligned_alloc(sizeof *prep);
            if (!w || !x || !prep) { printf("FAIL alloc\n"); return -1; }
            qfx_fill(type, w, sh->rows, sh->cols, 31 + s);
            qfx_activations(x, sh->cols, 17 + s);
            mynah_slm_matvec_prepare_int8(x, sh->cols, prep);
            const bctx c = { type, sh->cols, row_bytes, x, prep };
            rc |= ab(sh->name, "ingot", side_ingot, "int8", side_ours, &c, w, sh->rows,
                     rounds, 5e-2);
            if (s == 0)
                rc |= ab("CONTROL ingot vs ingot", "ingot", side_ingot, "ingot", side_ingot,
                         &c, w, sh->rows, rounds, 0.0);
            mynah_slm_aligned_free(w);
            mynah_slm_aligned_free(x);
            mynah_slm_aligned_free(prep);
        }
    }
    return rc;
}

/* K6: is a dense 2-byte matvec already at the memory roof? A = a pure
 * streaming read of the same bytes with the same row split (the roof this
 * process can reach), B = ingot's BF16 / F16 matvec. If B is within a few
 * percent of A, no dot-product instruction — BFDOT, VDPBF16PS — can make
 * decode faster, and K6 is not worth writing (.work/bf16-native-matvec.md). */
static int side_roof(const bctx *c, const unsigned char *w, size_t n, float *out) {
    for (size_t r = 0; r < n; r++) {
        const unsigned char *p = w + r * c->row_bytes;
        uint64_t a0 = 0, a1 = 0, a2 = 0, a3 = 0;
        size_t i = 0;
        for (; i + 32 <= c->row_bytes; i += 32) {
            uint64_t v[4];
            memcpy(v, p + i, sizeof v);
            a0 += v[0]; a1 += v[1]; a2 += v[2]; a3 += v[3];
        }
        out[r] = (float)((a0 ^ a1 ^ a2 ^ a3) & 0xffu);
    }
    return 0;
}

static int suite_k6(int rounds) {
    int rc = 0;
    const int types[] = { INGOT_TYPE_BF16, INGOT_TYPE_F16 };
    for (size_t t = 0; t < sizeof types / sizeof *types; t++) {
        const int type = types[t];
        printf("\nK6 — %s matvec vs the read roof: A = streaming read of the same bytes, "
               "B = ingot %s matvec (A/B near 1.00x = at the roof)\n",
               ingot_type_name(type), ingot_type_name(type));
        header();
        for (size_t s = 0; s < sizeof k_shapes / sizeof *k_shapes; s++) {
            const shape *sh = &k_shapes[s];
            if (sh->cols != 1024) continue;                 /* one width is enough */
            const size_t row_bytes = sh->cols * 2;
            unsigned char *w = mynah_slm_aligned_alloc(sh->rows * row_bytes);
            float *x = mynah_slm_aligned_alloc(sh->cols * sizeof *x);
            if (!w || !x) { printf("FAIL alloc\n"); return -1; }
            qfx_rng r = { 99 + s };
            for (size_t i = 0; i < sh->rows * sh->cols; i++) {
                const float v = qfx_uniform(&r, -0.05f, 0.05f);
                const uint16_t h = type == INGOT_TYPE_BF16 ? ingot_f32_to_bf16(v)
                                                           : ingot_f32_to_f16(v);
                memcpy(w + 2 * i, &h, 2);
            }
            qfx_activations(x, sh->cols, 3 + s);
            const bctx c = { type, sh->cols, row_bytes, x, NULL };
            rc |= ab(sh->name, "read", side_roof, "ingot", side_ingot, &c, w, sh->rows,
                     rounds, 1e300);
            const double mb = (double)sh->rows * (double)row_bytes / 1e6;
            printf("|   (%.1f MB per call) | | | | | | | | |\n", mb);
            mynah_slm_aligned_free(w);
            mynah_slm_aligned_free(x);
        }
    }
    return rc;
}

/* K5: what runtime dispatch costs. A = the resolved table's row kernel called
 * directly, B = the same kernel reached through mynah_slm_matvec (table
 * lookup, pthread_once fast path, type switch). Same kernel, same numbers, so
 * the ratio is the overhead and nothing else. Smallest real shape, where a
 * fixed per-call cost shows most. */
static int side_direct_q4k(const bctx *c, const unsigned char *w, size_t n, float *out) {
    const mynah_slm_qmat_kern *k = mynah_slm_kern_qmat();
    if (c->prep->have_int8 && k->int8)
        k->q4k_i8(w, n, c->cols / 256, c->prep->xq, c->prep->xscale, c->prep->xsum, out);
    else
        k->q4k_f32(w, n, c->cols / 256, c->x, c->prep->xsum, out);
    return 0;
}

static int suite_k5(int rounds) {
    printf("\nK5 — dispatch overhead: A = %s table called directly, B = mynah_slm_matvec\n",
           mynah_slm_kern_qmat()->name);
    header();
    int rc = 0;
    const shape *sh = &k_shapes[0];
    const size_t row_bytes = sh->cols / 256 * 144;
    unsigned char *w = mynah_slm_aligned_alloc(sh->rows * row_bytes);
    float *x = mynah_slm_aligned_alloc(sh->cols * sizeof *x);
    mynah_slm_matvec_in *prep = mynah_slm_aligned_alloc(sizeof *prep);
    if (!w || !x || !prep) { printf("FAIL alloc\n"); return -1; }
    qfx_fill(INGOT_TYPE_Q4_K, w, sh->rows, sh->cols, 51);
    qfx_activations(x, sh->cols, 52);
    for (int int8 = 0; int8 < 2; int8++) {
        if (int8) mynah_slm_matvec_prepare_int8(x, sh->cols, prep);
        else { mynah_slm_matvec_set_int8(0); mynah_slm_matvec_prepare(x, sh->cols, prep); }
        const bctx c = { INGOT_TYPE_Q4_K, sh->cols, row_bytes, x, prep };
        rc |= ab(int8 ? "1024x1024 Q4_K int8" : "1024x1024 Q4_K f32", "direct",
                 side_direct_q4k, "dispatch", side_ours, &c, w, sh->rows, rounds, 0.0);
    }
    mynah_slm_aligned_free(w);
    mynah_slm_aligned_free(x);
    mynah_slm_aligned_free(prep);
    return rc;
}

int main(int argc, char **argv) {
    const char *suite = argc > 1 ? argv[1] : "all";
    const int rounds = argc > 2 ? atoi(argv[2]) : 15;
    const char *thr = argc > 3 ? argv[3] : "1,4";
    if (rounds < 3 || rounds > 256) { fprintf(stderr, "rounds: 3..256\n"); return 2; }

    printf("bench_qmat — SYNTHETIC matrices, no checkpoint. CLOUD/LOCAL-SPECIFIC:\n"
           "revalidate on the target host with `make bench` on a real model.\n"
           "int8 kernel in this build: %s; %d interleaved rounds, order alternating\n",
           mynah_slm_matvec_int8_isa(), rounds);

    int rc = 0;
    for (const char *p = thr; *p;) {
        const int n = atoi(p);
        mynah_slm_threads_init(n);
        printf("\n== %d thread(s) ==\n", mynah_slm_threads_count());
        if (!strcmp(suite, "all") || !strcmp(suite, "k3")) rc |= suite_k3(rounds);
        if (!strcmp(suite, "all") || !strcmp(suite, "k4")) rc |= suite_k4(rounds);
        if (!strcmp(suite, "all") || !strcmp(suite, "k6")) rc |= suite_k6(rounds);
        if (!strcmp(suite, "all") || !strcmp(suite, "k5")) rc |= suite_k5(rounds);
        const char *comma = strchr(p, ',');
        if (!comma) break;
        p = comma + 1;
    }
    mynah_slm_threads_shutdown();
    return rc ? 1 : 0;
}
