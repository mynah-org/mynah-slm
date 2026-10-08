/* self_test.c — every CUDA kernel against the CPU backend, with stated gates.
 *
 * Both sides are driven through src/backend.h, so this also proves the CUDA
 * vtable is wired the way the CPU one is (an op landing on the wrong entry,
 * a wrong layer offset in the KV, a swapped K/V would all fail here).
 *
 * The CPU backend is the reference: it IS the engine's arithmetic
 * (tests/test_backend.c pins that bit for bit), pinned here to f32
 * activations — MYNAH_SLM_INT8 would quantize the CPU side's activations to
 * int8 and turn every product gate into a measure of that instead.
 * Tolerances, one per kind of difference, also written in
 * .work/cuda-backend.md:
 *
 *   GEMV / batched GEMV   |gpu - cpu| <= 1e-6 * sum_i |w_i x_i|, per row.
 *                         A reorder bound: the CPU sums per block in its own
 *                         order, the device per lane then a shuffle tree.
 *                         Both are f32 sums of the same f32 products; a
 *                         host emulation of this kernel (reviewer, warp by
 *                         warp) differed from the CPU by at most ~5.5e-8 of
 *                         sum|wx| over Q4_K / Q6_K / Q8_0 / F32 rows, so
 *                         1e-6 leaves ~18x headroom and still catches a
 *                         wrong scale, a dropped block or a lost lane, which
 *                         cost orders of magnitude more.
 *   block decode, RoPE    BITWISE (tol 0). Same products in the same order
 *                         (__fmul_rn/__fsub_rn on the device, no contraction
 *                         on the gcc -std=c11 CPU side), and the host check
 *                         already runs these helpers to err 0. A CPU build
 *                         that contracts a*c - b*s into an FMA (clang's
 *                         default -ffp-contract=on with -march=native) breaks
 *                         the premise and fails here, as it should.
 *   bf16 KV storage       bitwise on the HOST only (host check); the device
 *                         planes are not read back, so on the device the
 *                         stored bits are gated only through attention.
 *   add                   bitwise.
 *   add_scaled            2^-22 * max|cpu| (same FMA caveat).
 *   RMSNorm, QK-norm      2e-6 * max|cpu|: the CPU sums squares in double,
 *                         the device in f32 through a tree.
 *   SwiGLU                1e-6 * max|cpu|: device expf vs libm expf.
 *   attention (bf16 KV)   1e-4 * max|cpu|: same stored bf16 bits on both
 *                         sides; online softmax vs two-pass softmax.
 *   argmax                the exact index the CPU gives: ties to the first,
 *                         NaN never taken, x[0] NaN gives 0.
 *
 * SPDX-License-Identifier: MIT */
#include "cuda_self_test.h"

#include "backend.h"
#include "kernels.h"
#include "qmat.h"

#include "ingot/dtype.h"
#include "ingot/quant.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    FILE *log;
    int   failures, checks;
    mynah_slm_backend *cpu, *gpu;
    char  err[256];
} ctx;

static void report(ctx *c, const char *what, int ok, double got, double tol) {
    c->checks++;
    if (!ok) c->failures++;
    fprintf(c->log, "%s %-58s err %.3g  tol %.3g\n", ok ? "ok  " : "FAIL", what, got, tol);
}

static void fail(ctx *c, const char *what) {
    c->checks++;
    c->failures++;
    fprintf(c->log, "FAIL %-58s %s\n", what, c->err);
}

static uint32_t rng = 0x1234567u;
static float frand(void) {
    rng = rng * 1664525u + 1013904223u;
    return (float)((int32_t)(rng >> 8) % 20001 - 10000) / 10000.0f;
}
static void fill(float *x, size_t n, float s) { for (size_t i = 0; i < n; i++) x[i] = frand() * s; }

static double max_abs(const float *x, size_t n) {
    double m = 0.0;
    for (size_t i = 0; i < n; i++) if (fabs((double)x[i]) > m) m = fabs((double)x[i]);
    return m;
}
static double max_diff(const float *a, const float *b, size_t n) {
    double m = 0.0;
    for (size_t i = 0; i < n; i++) {
        const double d = fabs((double)a[i] - (double)b[i]);
        if (d > m || d != d) m = d;
    }
    return m;
}

/* A device copy of host data; NULL (with c->err) on failure. */
static float *up(ctx *c, const float *h, size_t n) {
    float *d = mynah_slm_backend_alloc(c->gpu, n, c->err, sizeof c->err);
    if (d && mynah_slm_backend_h2d(c->gpu, d, h, n, c->err, sizeof c->err) != 0) {
        mynah_slm_backend_free(c->gpu, d);
        return NULL;
    }
    return d;
}
static int down(ctx *c, float *h, const float *d, size_t n) {
    return mynah_slm_backend_d2h(c->gpu, h, d, n, c->err, sizeof c->err);
}

/* Upload the same bytes to both backends. */
static int both_weights(ctx *c, int type, const void *q, size_t rows, size_t cols,
                        const mynah_slm_bweight **wc, const mynah_slm_bweight **wg) {
    if (mynah_slm_backend_weight(c->cpu, type, q, rows, cols, wc, c->err, sizeof c->err) != 0)
        return -1;
    const int rc = mynah_slm_backend_weight(c->gpu, type, q, rows, cols, wg, c->err, sizeof c->err);
    if (rc == 1) snprintf(c->err, sizeof c->err, "type %d unsupported on cuda", type);
    return rc == 0 ? 0 : -1;
}

static void *quantize(int type, const float *f, size_t n) {
    uint64_t bytes = 0;
    if (ingot_type_nbytes(type, n, &bytes) != 0) return NULL;
    void *q = malloc((size_t)bytes);
    if (!q) return NULL;
    if (type == INGOT_TYPE_F32) memcpy(q, f, n * sizeof(float));
    else if (ingot_quantize(type, f, n, q) != 0) { free(q); return NULL; }
    return q;
}

/* ── products and block decode ──────────────────────────────────────────── */

static void check_products(ctx *c, int type, const char *tname, size_t rows, size_t cols,
                           void **keep) {
    char what[96];
    enum { TOK = 4 };
    float *wf = malloc(rows * cols * sizeof(float));
    float *wd = malloc(rows * cols * sizeof(float));      /* what the file decodes to */
    float *x = malloc(TOK * cols * sizeof(float));
    float *yc = malloc(TOK * rows * sizeof(float));
    float *yg = malloc(TOK * rows * sizeof(float));
    fill(wf, rows * cols, 0.5f);
    fill(x, TOK * cols, 1.0f);
    void *q = quantize(type, wf, rows * cols);
    *keep = q;
    float *dx = NULL, *dy = NULL;
    const mynah_slm_bweight *wc = NULL, *wg = NULL;

    snprintf(what, sizeof what, "matvec %s %zux%zu", tname, rows, cols);
    if (!q || !wf || !wd || !x || !yc || !yg ||
        ingot_dequant_matrix(type, q, rows, cols, wd) != 0 ||
        both_weights(c, type, q, rows, cols, &wc, &wg) != 0 ||
        !(dx = up(c, x, TOK * cols)) ||
        !(dy = mynah_slm_backend_alloc(c->gpu, TOK * rows, c->err, sizeof c->err))) {
        fail(c, what);
        goto out;
    }

    for (int pass = 0; pass < 2; pass++) {
        const size_t tok = pass ? TOK : 1;
        const int rc_c = pass ? mynah_slm_backend_matmat(c->cpu, wc, x, yc, tok, c->err, sizeof c->err)
                              : mynah_slm_backend_matvec(c->cpu, wc, x, yc, c->err, sizeof c->err);
        const int rc_g = pass ? mynah_slm_backend_matmat(c->gpu, wg, dx, dy, tok, c->err, sizeof c->err)
                              : mynah_slm_backend_matvec(c->gpu, wg, dx, dy, c->err, sizeof c->err);
        snprintf(what, sizeof what, "%s %s %zux%zu%s", pass ? "matmat" : "matvec", tname, rows, cols,
                 pass ? " x4 tokens" : "");
        if (rc_c != 0 || rc_g != 0 || down(c, yg, dy, tok * rows) != 0) { fail(c, what); continue; }
        /* Worst ratio of the error to its own row's bound. */
        double worst = 0.0;
        for (size_t t = 0; t < tok; t++)
            for (size_t r = 0; r < rows; r++) {
                double mag = 0.0;
                for (size_t k = 0; k < cols; k++)
                    mag += fabs((double)wd[r * cols + k] * (double)x[t * cols + k]);
                /* 1e-6: see the table at the top. The 1e-30 only keeps an
                 * all-zero row (mag 0, both sides exactly 0) from 0/0. */
                const double bound = 1e-6 * mag + 1e-30;
                const double e = fabs((double)yg[t * rows + r] - (double)yc[t * rows + r]);
                const double ratio = (e != e) ? 1e30 : e / bound;
                if (ratio > worst) worst = ratio;
            }
        report(c, what, worst <= 1.0, worst, 1.0);   /* reported as error / bound */
    }

    /* Embedding rows: the decode alone, no product. */
    {
        const uint32_t ids[4] = { 0, (uint32_t)rows - 1, 7, (uint32_t)rows / 2 };
        float *ec = malloc(4 * cols * sizeof(float)), *eg = malloc(4 * cols * sizeof(float));
        float *de = mynah_slm_backend_alloc(c->gpu, 4 * cols, c->err, sizeof c->err);
        snprintf(what, sizeof what, "embed (block decode) %s %zux%zu, 4 rows, bitwise", tname, rows, cols);
        if (!ec || !eg || !de ||
            mynah_slm_backend_embed(c->cpu, wc, ids, 4, ec, c->err, sizeof c->err) != 0 ||
            mynah_slm_backend_embed(c->gpu, wg, ids, 4, de, c->err, sizeof c->err) != 0 ||
            down(c, eg, de, 4 * cols) != 0) {
            fail(c, what);
        } else {
            const double tol = 0.0;                 /* bitwise: see the table */
            const double e = max_diff(eg, ec, 4 * cols);
            report(c, what, e <= tol, e, tol);
        }
        free(ec); free(eg);
        mynah_slm_backend_free(c->gpu, de);
    }

out:
    mynah_slm_backend_free(c->gpu, dx);
    mynah_slm_backend_free(c->gpu, dy);
    free(wf); free(wd); free(x); free(yc); free(yg);
}

/* ── norms, RoPE, elementwise ───────────────────────────────────────────── */

static void check_elementwise(ctx *c) {
    enum { D = 1024, HD = 128, NH = 16, T = 4, N = T * NH * HD };   /* Qwen3-0.6B */
    static float x[N], u[N], rc[N], rg[N], g[D], gh[HD];
    /* Residual-stream scale: Qwen3 reaches |x| ~ 8e3 by layer 13. */
    fill(x, N, 4000.0f);
    fill(u, N, 4.0f);
    for (int i = 0; i < D; i++) g[i] = 1.0f + frand() * 0.3f;
    for (int i = 0; i < HD; i++) gh[i] = 1.0f + frand() * 0.3f;

    const mynah_slm_bweight *gc = NULL, *gg = NULL, *hc = NULL, *hg = NULL;
    float *dx = up(c, x, N), *du = up(c, u, N);
    float *dy = mynah_slm_backend_alloc(c->gpu, N, c->err, sizeof c->err);
    if (!dx || !du || !dy || both_weights(c, INGOT_TYPE_F32, g, 1, D, &gc, &gg) != 0 ||
        both_weights(c, INGOT_TYPE_F32, gh, 1, HD, &hc, &hg) != 0) {
        fail(c, "elementwise setup");
        goto out;
    }

    /* RMSNorm, T rows of d_model. */
    if (mynah_slm_backend_rms_norm(c->cpu, rc, x, gc, T, D, 1e-6f, c->err, sizeof c->err) != 0 ||
        mynah_slm_backend_rms_norm(c->gpu, dy, dx, gg, T, D, 1e-6f, c->err, sizeof c->err) != 0 ||
        down(c, rg, dy, (size_t)T * D) != 0) {
        fail(c, "rms_norm 4x1024");
    } else {
        const double tol = 2e-6 * max_abs(rc, (size_t)T * D), e = max_diff(rg, rc, (size_t)T * D);
        report(c, "rms_norm 4x1024, |x| up to 4e3", e <= tol, e, tol);
    }

    /* QK-norm in place: T tokens x 16 heads x 128. */
    memcpy(rc, x, sizeof x);
    if (mynah_slm_backend_h2d(c->gpu, dy, x, N, c->err, sizeof c->err) != 0 ||
        mynah_slm_backend_rms_norm_heads(c->cpu, rc, hc, (size_t)T * NH, HD, 1e-6f, c->err, sizeof c->err) != 0 ||
        mynah_slm_backend_rms_norm_heads(c->gpu, dy, hg, (size_t)T * NH, HD, 1e-6f, c->err, sizeof c->err) != 0 ||
        down(c, rg, dy, N) != 0) {
        fail(c, "rms_norm_heads");
    } else {
        const double tol = 2e-6 * max_abs(rc, N), e = max_diff(rg, rc, N);
        report(c, "rms_norm_heads (QK-norm) 4 tok x 16 heads x 128", e <= tol, e, tol);
    }

    /* RoPE, both pairings, theta 1e6 (Qwen3's), positions 1000..1003. */
    for (int il = 0; il <= 1; il++) {
        const char *what = il ? "rope interleaved, pos 1000..1003, bitwise"
                              : "rope NeoX split-half, pos 1000..1003, bitwise";
        mynah_slm_brope *pc = NULL, *pg = NULL;
        float *xn = rc;
        for (size_t i = 0; i < N; i++) xn[i] = x[i] / 4000.0f;
        if (mynah_slm_backend_h2d(c->gpu, dy, xn, N, c->err, sizeof c->err) != 0 ||
            mynah_slm_backend_rope_create(c->cpu, HD, 2048, 1e6f, il, &pc, c->err, sizeof c->err) != 0 ||
            mynah_slm_backend_rope_create(c->gpu, HD, 2048, 1e6f, il, &pg, c->err, sizeof c->err) != 0 ||
            mynah_slm_backend_rope(c->cpu, pc, rc, T, NH, 1000, c->err, sizeof c->err) != 0 ||
            mynah_slm_backend_rope(c->gpu, pg, dy, T, NH, 1000, c->err, sizeof c->err) != 0 ||
            down(c, rg, dy, N) != 0) {
            fail(c, what);
        } else {
            const double tol = 0.0, e = max_diff(rg, rc, N);     /* bitwise */
            report(c, what, e <= tol, e, tol);
        }
        mynah_slm_backend_rope_free(c->cpu, pc);
        mynah_slm_backend_rope_free(c->gpu, pg);
    }

    /* SwiGLU, add, add_scaled. */
    for (int op = 0; op < 3; op++) {
        static const char *names[3] = { "swiglu", "add (bitwise)", "add_scaled 0.263" };
        memcpy(rc, u, sizeof u);
        int ok = mynah_slm_backend_h2d(c->gpu, dy, u, N, c->err, sizeof c->err) == 0;
        if (ok && op == 0)
            ok = mynah_slm_backend_swiglu(c->cpu, rc, x, N, c->err, sizeof c->err) == 0 &&
                 mynah_slm_backend_swiglu(c->gpu, dy, dx, N, c->err, sizeof c->err) == 0;
        if (ok && op == 1)
            ok = mynah_slm_backend_add(c->cpu, rc, x, N, c->err, sizeof c->err) == 0 &&
                 mynah_slm_backend_add(c->gpu, dy, dx, N, c->err, sizeof c->err) == 0;
        if (ok && op == 2)
            ok = mynah_slm_backend_add_scaled(c->cpu, rc, x, 0.263f, N, c->err, sizeof c->err) == 0 &&
                 mynah_slm_backend_add_scaled(c->gpu, dy, dx, 0.263f, N, c->err, sizeof c->err) == 0;
        if (!ok || down(c, rg, dy, N) != 0) { fail(c, names[op]); continue; }
        const double m = max_abs(rc, N);
        const double tol = op == 0 ? 1e-6 * m : op == 1 ? 0.0 : ldexp(1.0, -22) * m;
        const double e = max_diff(rg, rc, N);
        report(c, names[op], e <= tol, e, tol);
    }

out:
    mynah_slm_backend_free(c->gpu, dx);
    mynah_slm_backend_free(c->gpu, du);
    mynah_slm_backend_free(c->gpu, dy);
}

/* ── KV append + GQA attention, bf16 on both sides ──────────────────────── */

static void check_attention(ctx *c, uint32_t nh, uint32_t nkv, uint32_t hd) {
    enum { L = 2, CTX = 256, MB = 8, FILL = 200 };
    const uint32_t kv_dim = nkv * hd, q_dim = nh * hd;
    const float scale = 1.0f / sqrtf((float)hd);
    char what[96];

    mynah_slm_bkv_desc d = { MYNAH_SLM_KV_BF16, MYNAH_SLM_KV_BF16, L, CTX, nh, nkv, hd, MB };
    mynah_slm_bkv *kc = NULL, *kg = NULL;
    float *k = malloc((size_t)FILL * kv_dim * sizeof(float));
    float *v = malloc((size_t)FILL * kv_dim * sizeof(float));
    float *q = malloc((size_t)MB * q_dim * sizeof(float));
    float *oc = malloc((size_t)MB * q_dim * sizeof(float));
    float *og = malloc((size_t)MB * q_dim * sizeof(float));
    float *dk = NULL, *dv = NULL, *dq = NULL, *dout = NULL;
    if (!k || !v || !q || !oc || !og) { snprintf(c->err, sizeof c->err, "oom"); fail(c, "attention setup"); goto out; }
    fill(k, (size_t)FILL * kv_dim, 2.0f);
    fill(v, (size_t)FILL * kv_dim, 2.0f);
    fill(q, (size_t)MB * q_dim, 2.0f);

    if (mynah_slm_backend_kv_create(c->cpu, &d, &kc, c->err, sizeof c->err) != 0 ||
        mynah_slm_backend_kv_create(c->gpu, &d, &kg, c->err, sizeof c->err) != 0 ||
        !(dk = up(c, k, (size_t)FILL * kv_dim)) || !(dv = up(c, v, (size_t)FILL * kv_dim)) ||
        !(dq = up(c, q, (size_t)MB * q_dim)) ||
        !(dout = mynah_slm_backend_alloc(c->gpu, (size_t)MB * q_dim, c->err, sizeof c->err))) {
        fail(c, "attention setup");
        goto out;
    }

    /* Layer 1, a 150-row prefill append then one row at a time, so both the
     * layer offset and the position offset are exercised. */
    int ok = 1;
    ok = ok && mynah_slm_backend_kv_append(c->cpu, kc, 1, 0, 150, k, v, c->err, sizeof c->err) == 0;
    ok = ok && mynah_slm_backend_kv_append(c->gpu, kg, 1, 0, 150, dk, dv, c->err, sizeof c->err) == 0;
    for (uint32_t p = 150; ok && p < FILL; p++) {
        ok = mynah_slm_backend_kv_append(c->cpu, kc, 1, p, 1, k + (size_t)p * kv_dim,
                                         v + (size_t)p * kv_dim, c->err, sizeof c->err) == 0 &&
             mynah_slm_backend_kv_append(c->gpu, kg, 1, p, 1, dk + (size_t)p * kv_dim,
                                         dv + (size_t)p * kv_dim, c->err, sizeof c->err) == 0;
    }
    if (!ok) { fail(c, "kv_append"); goto out; }

    /* Decode at the last position, then a causal batch of 8 ending there;
     * then the short histories where some of the ATT_WARPS (4) warps see no
     * position at all and must drop out of the merge instead of adding a
     * NaN: decode at pos0 = 0, 1, 2, and a causal batch of 8 from pos0 = 0
     * (its rows 0..2 are such rows). */
    static const struct { uint32_t pos0, nq; const char *tag; } passes[] = {
        { FILL - 1,  1,  "decode, 200 positions" },
        { FILL - MB, MB, "batch of 8 (causal)" },
        { 0,         1,  "decode at pos 0" },
        { 1,         1,  "decode at pos 1" },
        { 2,         1,  "decode at pos 2" },
        { 0,         MB, "batch of 8 from pos 0" },
    };
    for (size_t pass = 0; pass < sizeof passes / sizeof passes[0]; pass++) {
        const uint32_t pos0 = passes[pass].pos0;
        const size_t nq = passes[pass].nq;
        snprintf(what, sizeof what, "attention bf16 %s, %u/%u heads x %u",
                 passes[pass].tag, nh, nkv, hd);
        if (mynah_slm_backend_attention(c->cpu, kc, 1, q, oc, pos0, nq, scale, c->err, sizeof c->err) != 0 ||
            mynah_slm_backend_attention(c->gpu, kg, 1, dq, dout, pos0, nq, scale, c->err, sizeof c->err) != 0 ||
            down(c, og, dout, nq * q_dim) != 0) {
            fail(c, what);
            continue;
        }
        const double tol = 1e-4 * max_abs(oc, nq * q_dim), e = max_diff(og, oc, nq * q_dim);
        report(c, what, e <= tol, e, tol);
    }

out:
    mynah_slm_backend_kv_free(c->cpu, kc);
    mynah_slm_backend_kv_free(c->gpu, kg);
    mynah_slm_backend_free(c->gpu, dk);
    mynah_slm_backend_free(c->gpu, dv);
    mynah_slm_backend_free(c->gpu, dq);
    mynah_slm_backend_free(c->gpu, dout);
    free(k); free(v); free(q); free(oc); free(og);
}

static void check_argmax(ctx *c) {
    enum { V = 151936 };                         /* Qwen3's vocabulary */
    float *x = malloc(V * sizeof(float));
    if (!x) { snprintf(c->err, sizeof c->err, "oom"); fail(c, "argmax"); return; }
    float *dx = mynah_slm_backend_alloc(c->gpu, V, c->err, sizeof c->err);
    /* A planted tie (the first wins), then NaN the way a parallel argmax can
     * get it wrong: first in a thread's stride hiding that stride's max
     * (1024 threads: index 5 and 5 + 1024 share thread 5), and at x[0],
     * where the CPU answers 0. Expected index from the CPU, never assumed. */
    static const char *const what[3] = {
        "argmax 151936, a tie (exact index)",
        "argmax 151936, NaN first in a stride (exact index)",
        "argmax 151936, NaN at x[0] (exact index)",
    };
    static const uint32_t expect[3] = { 98765, 5 + 1024, 0 };
    for (int k = 0; k < 3; k++) {
        fill(x, V, 10.0f);
        if (k == 0) { x[123456] = 50.0f; x[98765] = 50.0f; }
        if (k == 1) { x[5 + 1024] = 100.0f; x[5] = NAN; }
        if (k == 2) { x[0] = NAN; x[777] = 100.0f; }
        uint32_t ic = 0, ig = 0;
        if (!dx || mynah_slm_backend_h2d(c->gpu, dx, x, V, c->err, sizeof c->err) != 0 ||
            mynah_slm_backend_argmax(c->cpu, x, V, &ic, c->err, sizeof c->err) != 0 ||
            mynah_slm_backend_argmax(c->gpu, dx, V, &ig, c->err, sizeof c->err) != 0) {
            fail(c, what[k]);
            continue;
        }
        report(c, what[k], ig == ic && ic == expect[k], (double)ig, (double)ic);
    }
    mynah_slm_backend_free(c->gpu, dx);
    free(x);
}

/* The NULL/1 contract on the device: refusals, not failures. */
static void check_refusals(ctx *c) {
    mynah_slm_bkv_desc d = { MYNAH_SLM_KV_F32, MYNAH_SLM_KV_F32, 1, 16, 16, 8, 128, 1 };
    mynah_slm_bkv *kv = NULL;
    const int r1 = mynah_slm_backend_kv_create(c->gpu, &d, &kv, c->err, sizeof c->err);
    report(c, "cuda refuses an f32 KV cache (returns 1)", r1 == 1 && !kv, r1, 1);
    d.type_k = d.type_v = MYNAH_SLM_KV_BF16;
    d.head_dim = 96;
    const int r2 = mynah_slm_backend_kv_create(c->gpu, &d, &kv, c->err, sizeof c->err);
    report(c, "cuda refuses head_dim 96 (returns 1)", r2 == 1 && !kv, r2, 1);
    static uint8_t q5k[176];
    const mynah_slm_bweight *w = NULL;
    const int r3 = mynah_slm_backend_weight(c->gpu, INGOT_TYPE_Q5_K, q5k, 1, 256, &w, c->err, sizeof c->err);
    report(c, "cuda refuses a Q5_K weight (returns 1)", r3 == 1 && !w, r3, 1);
    /* A refusal is passed through by the slot pool: 1, not -1, and no pool. */
    mynah_slm_bslots_desc sd = { { MYNAH_SLM_KV_F32, MYNAH_SLM_KV_F32, 1, 16, 16, 8, 128, 1 }, 0, 2 };
    mynah_slm_bslots *sp = NULL;
    const int r4 = mynah_slm_backend_slots_create(c->gpu, &sd, &sp, c->err, sizeof c->err);
    report(c, "cuda slot pool over an f32 KV is refused (returns 1)", r4 == 1 && !sp, r4, 1);
    mynah_slm_backend_slots_destroy(c->gpu, sp);
}

/* Cancellation on the device: a slot released while its step may still be
 * running is parked, never freed and never waited on; it comes back once its
 * fence passes. And an error that belongs to one request (here: an
 * allocation the device cannot satisfy) leaves the backend serving. */
static void check_slots(ctx *c) {
    mynah_slm_bslots_desc d = {
        { MYNAH_SLM_KV_BF16, MYNAH_SLM_KV_BF16, 1, 512, 16, 8, 128, 1 }, 4096, 2 };
    mynah_slm_bslots *p = NULL;
    uint32_t s0 = 9, s1 = 9, s2 = 9;
    uint64_t g0 = 0, g1 = 0, g2 = 0;
    if (mynah_slm_backend_slots_create(c->gpu, &d, &p, c->err, sizeof c->err) != 0 ||
        mynah_slm_backend_slot_acquire(c->gpu, p, &s0, &g0, c->err, sizeof c->err) != 0 ||
        mynah_slm_backend_slot_acquire(c->gpu, p, &s1, &g1, c->err, sizeof c->err) != 0) {
        fail(c, "slot pool on the device");
        mynah_slm_backend_slots_destroy(c->gpu, p);
        return;
    }
    /* Queue real work on slot 0, then cancel it at once: no sync. */
    float *scr = mynah_slm_backend_slot_scratch(p, s0);
    mynah_slm_bkv *kv = mynah_slm_backend_slot_kv(p, s0);
    int ok = mynah_slm_backend_kv_append(c->gpu, kv, 0, 0, 1, scr, scr + 1024, c->err, sizeof c->err) == 0 &&
             mynah_slm_backend_attention(c->gpu, kv, 0, scr, scr + 2048, 0, 1, 0.088f, c->err, sizeof c->err) == 0 &&
             mynah_slm_backend_slot_release(c->gpu, p, s0, c->err, sizeof c->err) == 0;
    /* Right after release the fence may or may not have passed: either the
     * slot comes back (0) or the pool is busy (1). Never an error, never a wait. */
    const int early = ok ? mynah_slm_backend_slot_acquire(c->gpu, p, &s2, &g2, c->err, sizeof c->err) : -1;
    if (early == 0) ok = ok && mynah_slm_backend_slot_release(c->gpu, p, s2, c->err, sizeof c->err) == 0;
    ok = ok && (early == 0 || early == 1) &&
         mynah_slm_backend_sync(c->gpu, c->err, sizeof c->err) == 0 &&
         mynah_slm_backend_slot_acquire(c->gpu, p, &s2, &g2, c->err, sizeof c->err) == 0;
    report(c, "cancelled slot parks behind a fence, comes back after it",
           ok && s2 == s0 && g2 > g0, (double)early, 1);

    /* A request whose allocation cannot be met: reported, cleared, and the
     * next op of another request still runs. */
    float *huge = mynah_slm_backend_alloc(c->gpu, (size_t)1 << 44, c->err, sizeof c->err);
    const int rec = mynah_slm_backend_recover(c->gpu, c->err, sizeof c->err);
    mynah_slm_bkv *kv1 = mynah_slm_backend_slot_kv(p, s1);
    float *scr1 = mynah_slm_backend_slot_scratch(p, s1);
    const int after = mynah_slm_backend_kv_append(c->gpu, kv1, 0, 0, 1, scr1, scr1 + 1024, c->err, sizeof c->err) == 0 &&
                      mynah_slm_backend_sync(c->gpu, c->err, sizeof c->err) == 0;
    report(c, "a failed allocation retires one request, not the backend",
           huge == NULL && rec == 0 && after, rec, 0);
    mynah_slm_backend_free(c->gpu, huge);
    mynah_slm_backend_slots_destroy(c->gpu, p);
}

int mynah_slm_cuda_self_test(FILE *log) {
    ctx c;
    memset(&c, 0, sizeof c);
    c.log = log ? log : stdout;

    if (mynah_slm_cuda_device_count(c.err, sizeof c.err) <= 0) {
        fprintf(c.log, "SKIP cuda self-test: %s\n", c.err);
        return 77;
    }
    if (mynah_slm_backend_open(MYNAH_SLM_DEVICE_CPU, &c.cpu, c.err, sizeof c.err) != 0 ||
        mynah_slm_backend_open(MYNAH_SLM_DEVICE_CUDA, &c.gpu, c.err, sizeof c.err) != 0) {
        fprintf(c.log, "FAIL cannot open the backends: %s\n", c.err);
        mynah_slm_backend_close(c.cpu);
        return 1;
    }

    /* The f32-activation CPU reference, whatever MYNAH_SLM_INT8 says. */
    const int int8_was = mynah_slm_matvec_int8_enabled();
    mynah_slm_matvec_set_int8(0);

    /* Qwen3-0.6B shapes: q/o/k/v-like 1024-wide, the FFN 3072 both ways, and
     * an LM-head-like Q6_K slab (the real one is 151936 rows). Then 1026 rows
     * of every type: not a multiple of the 4 rows per block, so the last
     * block runs with two warps past the end. */
    void *keep[12] = { 0 };
    check_products(&c, INGOT_TYPE_F32,  "F32",  512,  1024, &keep[0]);
    check_products(&c, INGOT_TYPE_Q8_0, "Q8_0", 1024, 1024, &keep[1]);
    check_products(&c, INGOT_TYPE_Q4_K, "Q4_K", 3072, 1024, &keep[2]);
    check_products(&c, INGOT_TYPE_Q4_K, "Q4_K", 1024, 3072, &keep[3]);
    check_products(&c, INGOT_TYPE_Q6_K, "Q6_K", 4096, 1024, &keep[4]);
    check_products(&c, INGOT_TYPE_F32,  "F32",  1026, 1024, &keep[5]);
    check_products(&c, INGOT_TYPE_Q8_0, "Q8_0", 1026, 1024, &keep[6]);
    check_products(&c, INGOT_TYPE_Q4_K, "Q4_K", 1026, 1024, &keep[7]);
    check_products(&c, INGOT_TYPE_Q6_K, "Q6_K", 1026, 1024, &keep[8]);
    check_elementwise(&c);
    check_attention(&c, 16, 8, 128);   /* Qwen3-0.6B: GQA group 2, head_dim 128 */
    check_attention(&c, 8, 1, 64);     /* MQA at 64: the template, not a special case */
    check_attention(&c, 8, 2, 256);
    check_argmax(&c);
    check_refusals(&c);
    check_slots(&c);

    mynah_slm_backend_close(c.gpu);
    mynah_slm_backend_close(c.cpu);
    for (int i = 0; i < 12; i++) free(keep[i]);
    mynah_slm_matvec_set_int8(int8_was);

    fprintf(c.log, "%d/%d cuda checks passed\n", c.checks - c.failures, c.checks);
    return c.failures ? 1 : 0;
}
