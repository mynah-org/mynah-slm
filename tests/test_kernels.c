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
    if (mynah_slm_rope_init(&r, HD, MAXPOS, 10000.0f) != 0) {
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

    mynah_slm_attention(out, q, k, v, NKV, HEADS, KVH, HD, scratch);

    /* With a single position in the cache the softmax is 1.0, so the output is
     * exactly v[0] of that head's KV head. */
    float one[HEADS * HD];
    mynah_slm_attention(one, q, k, v, 1, HEADS, KVH, HD, scratch);
    int ok = 1;
    for (int i = 0; i < HD; i++) if (!close_to(one[i], v[i], 1e-5f)) ok = 0;
    check("attention over one position returns that value", ok, "not v[0]");

    /* Heads 0 and 1 share KV head 0. Identical queries into a shared group
     * must give identical outputs — that is what "grouped" means. */
    float q2[HEADS * HD];
    memcpy(q2, q, sizeof q);
    memcpy(q2 + HD, q2, HD * sizeof(float));
    float o2[HEADS * HD];
    mynah_slm_attention(o2, q2, k, v, NKV, HEADS, KVH, HD, scratch);
    ok = 1;
    for (int i = 0; i < HD; i++) if (!close_to(o2[i], o2[HD + i], 1e-5f)) ok = 0;
    check("attention groups share their KV head", ok,
          "two heads in one group disagreed");

    /* The property that makes a KV cache valid: a query over n_kv positions
     * must not read past position n_kv - 1. Scribble over the tail of the
     * cache and demand the answer is unchanged. An off-by-one in the history
     * loop fails here and nowhere else — comparing two identical calls would
     * pass no matter what. */
    mynah_slm_attention(shorter, q, k, v, 3, HEADS, KVH, HD, scratch);
    float k2[NKV * KVH * HD], v2[NKV * KVH * HD];
    memcpy(k2, k, sizeof k);
    memcpy(v2, v, sizeof v);
    for (int t = 3; t < NKV; t++)
        for (int i = 0; i < KVH * HD; i++) {
            k2[t * KVH * HD + i] = 999.0f;
            v2[t * KVH * HD + i] = -999.0f;
        }
    float poisoned[HEADS * HD];
    mynah_slm_attention(poisoned, q, k2, v2, 3, HEADS, KVH, HD, scratch);
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

int main(void) {
    printf("-- norms --\n");        test_rms_norm(); test_rms_norm_per_head();
    printf("\n-- rope --\n");       test_rope();
    printf("\n-- activations --\n");test_activations();
    printf("\n-- attention --\n");  test_attention();

    printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "PASS",
           failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
