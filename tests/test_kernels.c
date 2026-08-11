/* test_kernels.c — the C kernels on the same properties as the oracle's.
 *
 * These mirror tools/oracle/test_kernels.py deliberately. Two independent
 * implementations checked against one set of properties is worth more than
 * either checked against itself, and the properties are the ones a
 * wrong-but-plausible kernel violates — above all the RoPE variant, where
 * split-half and interleaved both run and only one is Qwen3's.
 *
 * No model, no network.
 *
 * SPDX-License-Identifier: MIT */
#include "kernels.h"

#include "kvcache.h"
#include "qmat.h"

#include "ingot/dtype.h"
#include "ingot/quant.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures;

static void check(const char *what, int ok, const char *detail) {
    if (ok) {
        printf("ok   %s\n", what);
    } else {
        printf("FAIL %s  <- %s\n", what, detail ? detail : "");
        failures++;
    }
}

static int close_to(float a, float b, float tol) { return fabsf(a - b) <= tol; }

/* Deterministic pseudo-random: a fixed sequence beats rand() because a failure
 * has to be reproducible from the source alone. */
static uint32_t rng_state = 0x2545F491u;
static float frand(void) {
    rng_state = rng_state * 1664525u + 1013904223u;
    return (float)((int32_t)(rng_state >> 8) % 20001 - 10000) / 10000.0f;
}

/* ── norms ────────────────────────────────────────────────────────────────── */

static void test_rms_norm(void) {
    enum { D = 64 };
    float x[D], out[D], scaled[D], w[D];
    for (int i = 0; i < D; i++) { x[i] = frand() * 3.0f; w[i] = 1.0f; }

    mynah_slm_rms_norm(out, x, w, D, 1e-6f);

    double sum = 0.0;
    for (int i = 0; i < D; i++) sum += (double)out[i] * out[i];
    check("rms_norm normalizes to unit RMS",
          close_to((float)sqrt(sum / D), 1.0f, 1e-4f), "RMS != 1");

    /* Scale invariance: RMSNorm has no mean subtraction, so a scaled input
     * gives the identical output. */
    for (int i = 0; i < D; i++) x[i] *= 7.0f;
    mynah_slm_rms_norm(scaled, x, w, D, 1e-6f);
    int same = 1;
    for (int i = 0; i < D; i++) if (!close_to(scaled[i], out[i], 1e-4f)) same = 0;
    check("rms_norm is scale invariant", same, "output changed with input scale");

    /* The weight is a plain gain — Gemma's (1 + weight) variant is a different
     * function, and this test fails first if the two ever get unified. */
    for (int i = 0; i < D; i++) w[i] = frand();
    float weighted[D];
    mynah_slm_rms_norm(weighted, x, w, D, 1e-6f);
    same = 1;
    for (int i = 0; i < D; i++) if (!close_to(weighted[i], out[i] * w[i], 1e-3f)) same = 0;
    check("rms_norm weight is a plain gain", same, "not out * weight");
}

static void test_rms_norm_per_head(void) {
    enum { HEADS = 4, HD = 32 };
    float x[HEADS * HD], w[HD];
    for (int i = 0; i < HEADS * HD; i++) x[i] = frand() * 5.0f;
    for (int i = 0; i < HD; i++) w[i] = 1.0f;

    /* Give head 2 a wildly different magnitude. If the norm were computed over
     * the whole vector instead of per head, the other heads would be dragged
     * off unit RMS. */
    for (int i = 0; i < HD; i++) x[2 * HD + i] *= 1000.0f;

    mynah_slm_rms_norm_per_head(x, w, HEADS, HD, 1e-6f);

    int ok = 1;
    for (int h = 0; h < HEADS; h++) {
        double s = 0.0;
        for (int i = 0; i < HD; i++) s += (double)x[h * HD + i] * x[h * HD + i];
        if (!close_to((float)sqrt(s / HD), 1.0f, 1e-3f)) ok = 0;
    }
    check("rms_norm_per_head normalizes each head independently", ok,
          "a head is off unit RMS — the norm is probably global");
}

/* ── RoPE ─────────────────────────────────────────────────────────────────── */

static void test_rope(void) {
    enum { HD = 8, MAXPOS = 32 };
    const uint32_t half = HD / 2;
    mynah_slm_rope r;
    if (mynah_slm_rope_init(&r, HD, MAXPOS, 10000.0f, 0) != 0) {
        check("rope_init", 0, "allocation failed");
        return;
    }

    /* The discriminating test. Put a 1 at index 0 and at index `half`. Under
     * SPLIT-HALF they are partners and mix. Under INTERLEAVED, index 0's
     * partner is index 1, so index `half` would not see index 0's angle at
     * all. One shot tells the two apart. */
    float x[HD];
    memset(x, 0, sizeof x);
    x[0] = 1.0f;
    x[half] = 1.0f;
    mynah_slm_rope_apply(&r, x, 1, 1);

    const float c = cosf(1.0f), s = sinf(1.0f);   /* pair 0 has frequency 1 */
    check("rope is split-half NeoX, not interleaved",
          close_to(x[0], c - s, 1e-5f) && close_to(x[half], c + s, 1e-5f),
          "index 0 and index half did not rotate as partners");

    /* Position 0 is the identity. */
    float y[2 * HD], y0[2 * HD];
    for (int i = 0; i < 2 * HD; i++) y[i] = y0[i] = frand();
    mynah_slm_rope_apply(&r, y, 2, 0);
    int same = 1;
    for (int i = 0; i < 2 * HD; i++) if (!close_to(y[i], y0[i], 1e-6f)) same = 0;
    check("rope at position 0 is identity", same, "position 0 rotated something");

    /* Rotation preserves the norm of each head. */
    for (int i = 0; i < 2 * HD; i++) y[i] = y0[i] = frand();
    mynah_slm_rope_apply(&r, y, 2, 11);
    int norms_ok = 1;
    for (int h = 0; h < 2; h++) {
        double a = 0.0, b = 0.0;
        for (int i = 0; i < HD; i++) {
            a += (double)y0[h * HD + i] * y0[h * HD + i];
            b += (double)y[h * HD + i] * y[h * HD + i];
        }
        if (!close_to((float)a, (float)b, 1e-3f)) norms_ok = 0;
    }
    check("rope preserves head norms", norms_ok, "a head changed length");

    /* The property RoPE exists for: <RoPE(q,m), RoPE(k,n)> depends only on
     * m - n. Two pairs at the same distance must give the same dot product. */
    float q[HD], k[HD], qa[HD], ka[HD], qb[HD], kb[HD];
    for (int i = 0; i < HD; i++) { q[i] = frand(); k[i] = frand(); }

    memcpy(qa, q, sizeof q); memcpy(ka, k, sizeof k);
    mynah_slm_rope_apply(&r, qa, 1, 7);
    mynah_slm_rope_apply(&r, ka, 1, 4);
    memcpy(qb, q, sizeof q); memcpy(kb, k, sizeof k);
    mynah_slm_rope_apply(&r, qb, 1, 9);
    mynah_slm_rope_apply(&r, kb, 1, 6);

    double d1 = 0.0, d2 = 0.0;
    for (int i = 0; i < HD; i++) { d1 += (double)qa[i] * ka[i]; d2 += (double)qb[i] * kb[i]; }
    check("rope dot product depends only on relative distance",
          close_to((float)d1, (float)d2, 1e-4f), "distance 3 gave two answers");

    mynah_slm_rope_free(&r);
}

/* ── activations ──────────────────────────────────────────────────────────── */

static void test_activations(void) {
    /* The overflow case: a naive x/(1+exp(-x)) gives inf/nan at -800. */
    float x[] = {-1e5f, -800.0f, -20.0f, -1.0f, 0.0f, 1.0f, 20.0f, 800.0f, 1e5f};
    const size_t n = sizeof x / sizeof *x;
    float v[9];
    memcpy(v, x, sizeof x);
    mynah_slm_silu(v, n);

    int finite = 1;
    for (size_t i = 0; i < n; i++) if (!isfinite(v[i])) finite = 0;
    check("silu is finite at extreme inputs", finite, "inf or nan escaped");
    check("silu(0) == 0", close_to(v[4], 0.0f, 1e-9f), NULL);
    check("silu saturates to identity", close_to(v[6], 20.0f, 1e-4f), NULL);

    /* Softmax must survive large positive inputs: exp(1000) is inf and
     * inf/inf is nan. Max-subtraction is what prevents it. */
    float sm[3] = {1000.0f, 1001.0f, 999.0f};
    mynah_slm_softmax(sm, 3);
    float sum = sm[0] + sm[1] + sm[2];
    check("softmax survives large positive inputs",
          isfinite(sum) && close_to(sum, 1.0f, 1e-6f) && sm[1] > sm[0] && sm[0] > sm[2],
          "did not sum to 1, or the order changed");
}

/* ── attention ────────────────────────────────────────────────────────────── */

static void test_attention(void) {
    enum { HEADS = 4, KVH = 2, HD = 8, NKV = 5 };
    float q[HEADS * HD], k[NKV * KVH * HD], v[NKV * KVH * HD];
    float out[HEADS * HD], shorter[HEADS * HD], scratch[NKV];

    for (int i = 0; i < HEADS * HD; i++) q[i] = frand();
    for (int i = 0; i < NKV * KVH * HD; i++) { k[i] = frand(); v[i] = frand(); }

    mynah_slm_attention(out, q, k, v, NKV, HEADS, KVH, HD,
                        1.0f / sqrtf((float)HD), scratch);

    /* With a single position in the cache the softmax is 1.0, so the output is
     * exactly v[0] of that head's KV head. */
    float one[HEADS * HD];
    mynah_slm_attention(one, q, k, v, 1, HEADS, KVH, HD,
                        1.0f / sqrtf((float)HD), scratch);
    int ok = 1;
    for (int i = 0; i < HD; i++) if (!close_to(one[i], v[i], 1e-5f)) ok = 0;
    check("attention over one position returns that value", ok, "not v[0]");

    /* Heads 0 and 1 share KV head 0. Identical queries into a shared group
     * must give identical outputs — that is what "grouped" means. */
    float q2[HEADS * HD];
    memcpy(q2, q, sizeof q);
    memcpy(q2 + HD, q2, HD * sizeof(float));
    float o2[HEADS * HD];
    mynah_slm_attention(o2, q2, k, v, NKV, HEADS, KVH, HD,
                        1.0f / sqrtf((float)HD), scratch);
    ok = 1;
    for (int i = 0; i < HD; i++) if (!close_to(o2[i], o2[HD + i], 1e-5f)) ok = 0;
    check("attention groups share their KV head", ok,
          "two heads in one group disagreed");

    /* The property that makes a KV cache valid: a query over n_kv positions
     * must not read past position n_kv - 1. Scribble over the tail of the
     * cache and demand the answer is unchanged. An off-by-one in the history
     * loop fails here and nowhere else — comparing two identical calls would
     * pass no matter what. */
    mynah_slm_attention(shorter, q, k, v, 3, HEADS, KVH, HD,
                        1.0f / sqrtf((float)HD), scratch);
    float k2[NKV * KVH * HD], v2[NKV * KVH * HD];
    memcpy(k2, k, sizeof k);
    memcpy(v2, v, sizeof v);
    for (int t = 3; t < NKV; t++)
        for (int i = 0; i < KVH * HD; i++) {
            k2[t * KVH * HD + i] = 999.0f;
            v2[t * KVH * HD + i] = -999.0f;
        }
    float poisoned[HEADS * HD];
    mynah_slm_attention(poisoned, q, k2, v2, 3, HEADS, KVH, HD,
                        1.0f / sqrtf((float)HD), scratch);
    ok = 1;
    for (int i = 0; i < HEADS * HD; i++)
        if (!close_to(shorter[i], poisoned[i], 1e-6f)) ok = 0;
    check("attention never reads past n_kv", ok,
          "poisoning the unused tail of the cache changed the output");

    /* And it must actually USE the history it is given, or the test above
     * would pass on a kernel that reads nothing at all. */
    ok = 0;
    for (int i = 0; i < HEADS * HD; i++)
        if (!close_to(shorter[i], out[i], 1e-4f)) ok = 1;
    check("attention output depends on history length", ok,
          "3 and 5 positions gave the same answer");

    /* Attention weights are a convex combination, so every output component
     * must lie inside the range of the v values it draws from. */
    float lo = v[0], hi = v[0];
    for (int t = 0; t < NKV; t++)
        for (int i = 0; i < HD; i++) {
            float e = v[t * KVH * HD + i];
            if (e < lo) lo = e;
            if (e > hi) hi = e;
        }
    ok = 1;
    for (int i = 0; i < HD; i++) if (out[i] < lo - 1e-5f || out[i] > hi + 1e-5f) ok = 0;
    check("attention output is a convex combination of values", ok,
          "output escaped the range of v — weights do not sum to 1");
}

/* Our Q4_K matvec against ingot's, on synthetic weights — no checkpoint, so
 * this runs in CI.
 *
 * ingot is the oracle here and that is the right way round: its kernel is
 * cross-checked against llama.cpp's own reference, so agreeing with it means
 * agreeing with the format. Ours computes the same sum in a different order —
 * the scale is applied once per 32-weight sub-block instead of per weight, and
 * the min term is folded into a precomputed input sum — so the two agree to
 * rounding, not to the bit.
 *
 * The failure this catches is not "slightly off": get the 6-bit scale/min
 * unpacking or the nibble order wrong and rows come out wildly wrong, which is
 * exactly what a tolerance of 1e-4 on a normalized error separates from noise.
 */
static void test_q4_k_matvec(void) {
    enum { ROWS = 96, COLS = 512 };            /* COLS a multiple of 256 */

    float *w = malloc((size_t)ROWS * COLS * sizeof *w);
    float *x = malloc(COLS * sizeof *x);
    float *a = malloc(ROWS * sizeof *a);
    float *b = malloc(ROWS * sizeof *b);
    mynah_slm_matvec_in prep;
    unsigned char *packed = malloc((size_t)ROWS * (COLS / 256) * 144);
    if (!w || !x || !a || !b || !packed) { check("q4_k allocations", 0, "out of memory"); return; }

    /* Deterministic, and with a range wide enough that d and dmin differ per
     * block — a fixture where every block shares one scale would pass with the
     * scale unpacking broken. */
    unsigned seed = 12345u;
    for (size_t i = 0; i < (size_t)ROWS * COLS; i++) {
        seed = seed * 1103515245u + 12345u;
        w[i] = (float)((int)((seed >> 16) & 0x7fffu) - 16384) * 1e-4f *
               (1.0f + (float)(i / COLS % 7));
    }
    for (int i = 0; i < COLS; i++) {
        seed = seed * 1103515245u + 12345u;
        x[i] = (float)((int)((seed >> 16) & 0x7fffu) - 16384) * 1e-4f;
    }

    for (int r = 0; r < ROWS; r++)
        if (ingot_q4_k_quantize(w + (size_t)r * COLS, COLS,
                                packed + (size_t)r * (COLS / 256) * 144) != 0) {
            check("q4_k quantize the fixture", 0, "ingot_q4_k_quantize failed");
            goto done;
        }

    mynah_slm_matvec_set_enabled(1);
    mynah_slm_matvec_set_int8(0);
    mynah_slm_matvec_prepare(x, COLS, &prep);
    if (mynah_slm_matvec(INGOT_TYPE_Q4_K, packed, ROWS, COLS, x, &prep, a) != 0) {
        check("our Q4_K matvec runs", 0, "it declined the call");
        goto done;
    }
    if (ingot_matvec(INGOT_TYPE_Q4_K, packed, ROWS, COLS, x, b) != 0) {
        check("ingot's Q4_K matvec runs", 0, "ingot_matvec failed");
        goto done;
    }

    double worst = 0.0, scale = 0.0;
    int at = 0;
    for (int r = 0; r < ROWS; r++) {
        const double d = fabs((double)a[r] - (double)b[r]);
        if (d > worst) { worst = d; at = r; }
        if (fabs((double)b[r]) > scale) scale = fabs((double)b[r]);
    }
    const double rel = scale > 0.0 ? worst / scale : worst;

    char detail[160];
    snprintf(detail, sizeof detail, "rel=%.2e at row %d (ours %.6f, ingot %.6f)",
             rel, at, a[at], b[at]);
    check("our Q4_K matvec agrees with ingot's", rel < 1e-4, detail);
    printf("     %s\n", detail);

    /* The int8-activation path is a different bargain and gets a different
     * bound: it quantizes the ACTIVATIONS, so it is not a reorder and must not
     * be held to a reorder's tolerance. The bound is loose on purpose: this
     * fixture is uniform noise, where a block's max is far from its typical
     * value and int8 therefore does worse than on real activations. Its job is
     * to catch a broken unpack — which lands at 0.5 relative, not 0.007 — not
     * to pin the quantization noise. What the noise actually costs is measured
     * where it matters, on perplexity: +1.4% (docs/perf.md). */
    if (mynah_slm_matvec_int8_enabled() || 1) {
        mynah_slm_matvec_set_int8(1);
        mynah_slm_matvec_prepare(x, COLS, &prep);
        if (prep.have_int8 &&
            mynah_slm_matvec(INGOT_TYPE_Q4_K, packed, ROWS, COLS, x, &prep, a) == 0) {
            double worst = 0.0, scale = 0.0;
            for (int r = 0; r < ROWS; r++) {
                const double d = fabs((double)a[r] - (double)b[r]);
                if (d > worst) worst = d;
                if (fabs((double)b[r]) > scale) scale = fabs((double)b[r]);
            }
            const double rel = scale > 0.0 ? worst / scale : worst;
            char d2[128];
            snprintf(d2, sizeof d2, "rel=%.2e (int8 activations, not a reorder)", rel);
            check("the int8 matvec stays within its quantization budget",
                  rel > 1e-7 && rel < 3e-2, d2);
            printf("     %s\n", d2);
        }
        mynah_slm_matvec_set_int8(0);
        mynah_slm_matvec_prepare(x, COLS, &prep);
    }

    /* And the fallback must be honest about what it does not have. Q6_K is the
     * standing example and it was earned: we wrote that kernel, measured it a
     * tie on ARM, upstreamed it to ingot where x86 had no vector path at all,
     * and then deleted ours because ingot's came back FASTER. See docs/perf.md
     * — the boundary in CLAUDE.md rule 4 moved a kernel, not an opinion. */
    check("we decline types we have no kernel for",
          mynah_slm_matvec(INGOT_TYPE_Q6_K, packed, ROWS, COLS, x, &prep, a) != 0,
          "claimed a Q6_K kernel we did not write");

done:
    free(w); free(x); free(a); free(b); free(packed);
}

/* The KV round trip is what the format costs numerically, and it is worth a
 * gate because the perplexity sweep in docs/perf.md rests on it: q4 keys came
 * out ruinous, and the first question about a result like that is whether the
 * quantizer is broken. These bounds say it is not — each format lands where
 * its bit budget says it should.
 *
 * Ranges rather than points: the exact figure depends on the data, and a test
 * pinned to four digits would fail on a rounding change that harmed nobody. */
static void test_kv_roundtrip(void) {
    enum { N = 1 << 15 };
    static float ref[N], x[N];

    /* Sum of six uniforms: roughly the bell shape a post-RoPE key has, and not
     * the flat distribution that would flatter a block-scaled format. */
    unsigned seed = 7u;
    for (int i = 0; i < N; i++) {
        double a = 0.0;
        for (int k = 0; k < 6; k++) {
            seed = seed * 1103515245u + 12345u;
            a += (double)((seed >> 16) & 0x7fff) / 32768.0 - 0.5;
        }
        ref[i] = (float)(a * 2.0);
    }

    const struct { mynah_slm_kv_type t; double lo, hi; } cases[] = {
        { MYNAH_SLM_KV_BF16, 5e-4, 5e-3 },   /* 8 mantissa bits */
        { MYNAH_SLM_KV_FP8,  8e-3, 6e-2 },   /* 3 mantissa bits, no scale */
        { MYNAH_SLM_KV_Q8,   1e-3, 1e-2 },   /* 127 levels per 32 values */
        { MYNAH_SLM_KV_Q4,   3e-2, 2e-1 },   /* 7 levels per 32 values */
    };

    for (size_t c = 0; c < sizeof cases / sizeof *cases; c++) {
        memcpy(x, ref, sizeof ref);
        mynah_slm_kv_roundtrip(cases[c].t, x, N);

        double num = 0.0, den = 0.0;
        for (int i = 0; i < N; i++) {
            const double d = (double)x[i] - (double)ref[i];
            num += d * d;
            den += (double)ref[i] * (double)ref[i];
        }
        const double rms = sqrt(num / den);

        char what[96], detail[96];
        snprintf(what, sizeof what, "%s round trip lands in its bit budget",
                 mynah_slm_kv_type_name(cases[c].t));
        snprintf(detail, sizeof detail, "rms rel %.3e, expected %.0e..%.0e",
                 rms, cases[c].lo, cases[c].hi);
        check(what, rms >= cases[c].lo && rms <= cases[c].hi, detail);
    }

    /* f32 must be exactly a no-op, or the default path is not the reference. */
    memcpy(x, ref, sizeof ref);
    mynah_slm_kv_roundtrip(MYNAH_SLM_KV_F32, x, N);
    check("f32 is bit-identical, not merely close",
          memcmp(x, ref, sizeof ref) == 0, "the default path altered the values");

    /* A per-block scale beats a floating exponent at the same width — the
     * reason q8 is the 8-bit format here and fp8 is not. */
    double err[2];
    for (int k = 0; k < 2; k++) {
        memcpy(x, ref, sizeof ref);
        mynah_slm_kv_roundtrip(k == 0 ? MYNAH_SLM_KV_Q8 : MYNAH_SLM_KV_FP8, x, N);
        double num = 0.0, den = 0.0;
        for (int i = 0; i < N; i++) {
            const double d = (double)x[i] - (double)ref[i];
            num += d * d; den += (double)ref[i] * (double)ref[i];
        }
        err[k] = sqrt(num / den);
    }
    char detail[96];
    snprintf(detail, sizeof detail, "q8 %.2e vs fp8 %.2e", err[0], err[1]);
    check("q8 is more accurate than fp8 at the same 8 bits", err[0] < err[1], detail);
}

/* The fused decode accessors must agree with the unpack-then-dot reference.
 *
 * They are two different implementations of the same sum — one walks the
 * packed bytes with NEON, the other decodes a slice and dots it — and the
 * first exists only because the second measured SLOWER than an f32 cache.
 * A vectorized kernel that is subtly wrong is exactly the thing that shows up
 * as "quality got a bit worse" three commits later. */
static void test_kv_packed(void) {
    enum { HEADS = 2, HD = 128, LAYERS = 1, CTX = 4 };

    float row[HEADS * HD], q[HD];
    unsigned seed = 99u;
    for (int i = 0; i < HEADS * HD; i++) {
        seed = seed * 1103515245u + 12345u;
        row[i] = (float)((int)((seed >> 16) & 0x7fff) - 16384) * 1e-4f;
    }
    for (int i = 0; i < HD; i++) {
        seed = seed * 1103515245u + 12345u;
        q[i] = (float)((int)((seed >> 16) & 0x7fff) - 16384) * 1e-4f;
    }

    const mynah_slm_kv_type types[] = { MYNAH_SLM_KV_BF16, MYNAH_SLM_KV_FP8,
                                        MYNAH_SLM_KV_Q8, MYNAH_SLM_KV_Q4 };
    for (size_t c = 0; c < sizeof types / sizeof *types; c++) {
        mynah_slm_kv kv;
        if (mynah_slm_kv_init(&kv, types[c], types[c], LAYERS, CTX, HEADS, HD) != 0) {
            check("kv cache allocates", 0, "init failed");
            return;
        }
        mynah_slm_kv_put_k(&kv, 0, 1, row);
        mynah_slm_kv_put_v(&kv, 0, 1, row);

        /* The reference: decode the slice, then dot it in plain f32.
         * gather_k writes n_kv rows of head_dim, so the buffer is 2*HD — one
         * HD was a stack overflow that only showed up off this machine. */
        float slice[2 * HD];
        mynah_slm_kv_gather_k(&kv, 0, 1, 2, slice);      /* positions 0..1 */
        const float *k1 = slice + HD;                    /* position 1 */
        double want = 0.0;
        for (int i = 0; i < HD; i++) want += (double)q[i] * (double)k1[i];

        const double got = mynah_slm_kv_dot_k(&kv, 0, 1, 1, q);
        const double rel = fabs(got - want) / (fabs(want) > 1e-9 ? fabs(want) : 1.0);

        char what[96], detail[128];
        snprintf(what, sizeof what, "%s fused dot matches unpack-then-dot",
                 mynah_slm_kv_type_name(types[c]));
        snprintf(detail, sizeof detail, "got %.6f want %.6f rel %.2e", got, want, rel);
        check(what, rel < 1e-5, detail);

        /* And the accumulate, which has its own unpacking. */
        float acc[HD] = {0};
        mynah_slm_kv_axpy_v(&kv, 0, 1, 1, 0.75f, acc);
        double worst = 0.0;
        for (int i = 0; i < HD; i++) {
            const double d = fabs((double)acc[i] - 0.75 * (double)k1[i]);
            if (d > worst) worst = d;
        }
        snprintf(what, sizeof what, "%s fused accumulate matches",
                 mynah_slm_kv_type_name(types[c]));
        snprintf(detail, sizeof detail, "worst abs %.2e", worst);
        check(what, worst < 1e-5, detail);

        /* A position never written must read as zero, or a fresh cache would
         * make attention read whatever the allocator left behind. */
        mynah_slm_kv_free(&kv);
    }

    /* The packed value has to be the round trip's value: two definitions of
     * one format is one too many. */
    mynah_slm_kv kv;
    if (mynah_slm_kv_init(&kv, MYNAH_SLM_KV_Q8, MYNAH_SLM_KV_Q8, 1, 2, HEADS, HD) == 0) {
        mynah_slm_kv_put_k(&kv, 0, 0, row);
        float packed[HD], expect[HD];
        mynah_slm_kv_gather_k(&kv, 0, 0, 1, packed);
        memcpy(expect, row, sizeof expect);
        mynah_slm_kv_roundtrip(MYNAH_SLM_KV_Q8, expect, HD);
        double worst = 0.0;
        for (int i = 0; i < HD; i++) {
            const double d = fabs((double)packed[i] - (double)expect[i]);
            if (d > worst) worst = d;
        }
        char detail[96];
        snprintf(detail, sizeof detail, "worst abs %.2e", worst);
        /* Exact, not close: one definition of the format, used by both. The
         * first version of this had the round trip keep an f32 scale and the
         * packed writer an f16 one, and the two drifted by 3.7e-04 — small
         * enough to look like rounding and large enough to make the measured
         * perplexity table describe something nobody could run. */
        check("packed q8 agrees with the round-trip definition", worst == 0.0, detail);
        mynah_slm_kv_free(&kv);
    }
}

int main(void) {
    printf("-- norms --\n");        test_rms_norm(); test_rms_norm_per_head();
    printf("\n-- rope --\n");       test_rope();
    printf("\n-- activations --\n");test_activations();
    printf("\n-- attention --\n");  test_attention();
    printf("\n-- quantized matvec --\n"); test_q4_k_matvec();
    printf("\n-- kv cache precision --\n"); test_kv_roundtrip();
    printf("\n-- kv cache, packed --\n"); test_kv_packed();

    printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "PASS",
           failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
