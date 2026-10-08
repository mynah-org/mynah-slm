/* test_backend.c — the CPU backend IS the existing kernels, bit for bit.
 *
 * The backend boundary is only safe to route a forward pass through if the
 * CPU side of it changes nothing. Every op here is run through the vtable and
 * through the function arch_qwen3.c calls directly, on the same random data,
 * and the two outputs are compared with memcmp — not a tolerance, because it
 * is the same code and anything but identical bits is a wiring bug (a wrong
 * stride, a wrong layer offset, a swapped K and V).
 *
 * Also pinned: the return contract (1 = unsupported, -1 = invalid) and that a
 * build without `make cuda` refuses the CUDA backend instead of quietly
 * handing back a CPU one.
 *
 * No model, no network, no GPU.
 *
 * SPDX-License-Identifier: MIT */
#include "backend.h"

#include "kernels.h"
#include "kvcache.h"
#include "qmat.h"
#include "threads.h"

#include "ingot/dtype.h"
#include "ingot/quant.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures;

static void check(const char *what, int ok, const char *detail) {
    if (ok) printf("ok   %s\n", what);
    else { printf("FAIL %s  <- %s\n", what, detail ? detail : ""); failures++; }
}

static uint32_t rng_state = 0x9E3779B9u;
static float frand(void) {
    rng_state = rng_state * 1664525u + 1013904223u;
    return (float)((int32_t)(rng_state >> 8) % 20001 - 10000) / 10000.0f;
}
static void fill(float *x, size_t n, float scale) {
    for (size_t i = 0; i < n; i++) x[i] = frand() * scale;
}

static int same(const float *a, const float *b, size_t n) {
    return memcmp(a, b, n * sizeof(float)) == 0;
}

static char err[256];

/* Weight bytes handed to the backend must outlive it (the CPU backend keeps
 * the pointer, like a model keeps its mmap), so they are freed after close. */
static void *keep[32];
static int   n_keep;
static void *kept(void *p) { if (p && n_keep < 32) keep[n_keep++] = p; return p; }

static size_t row_bytes(int type, size_t cols) {
    uint64_t e = 0, by = 0;
    ingot_type_geometry(type, &e, &by);
    return (size_t)(cols / e * by);
}

/* ── lifecycle and contract ─────────────────────────────────────────────── */

static void test_open(mynah_slm_backend *b) {
    check("cpu backend is named cpu", strcmp(mynah_slm_backend_name(b), "cpu") == 0, NULL);
    check("cpu backend reports the cpu device",
          mynah_slm_backend_device(b) == MYNAH_SLM_DEVICE_CPU, NULL);
    int all = 1;
    for (int op = 0; op < MYNAH_SLM_BOP_COUNT; op++)
        if (!mynah_slm_backend_has(b, (mynah_slm_bop)op)) {
            printf("     missing op %s\n", mynah_slm_bop_name((mynah_slm_bop)op));
            all = 0;
        }
    check("cpu backend implements every op", all, "an entry is NULL");

#if !defined(MYNAH_SLM_ENABLE_CUDA)
    mynah_slm_backend *cuda = NULL;
    const int rc = mynah_slm_backend_open(MYNAH_SLM_DEVICE_CUDA, &cuda, err, sizeof err);
    check("a CPU-only build refuses the CUDA backend, loudly",
          rc == -1 && cuda == NULL && strstr(err, "make cuda") != NULL, err);
#endif
}

/* Quantize a random [rows][cols] matrix into `type`; F32 is stored as-is. */
static void *make_weight(int type, size_t rows, size_t cols, float **f32_out) {
    float *f = malloc(rows * cols * sizeof(float));
    fill(f, rows * cols, 0.5f);
    uint64_t bytes = 0;
    if (ingot_type_nbytes(type, (uint64_t)(rows * cols), &bytes) != 0) { free(f); return NULL; }
    void *q = malloc((size_t)bytes);
    if (type == INGOT_TYPE_F32) memcpy(q, f, rows * cols * sizeof(float));
    else if (ingot_quantize(type, f, rows * cols, q) != 0) { free(f); free(q); return NULL; }
    if (f32_out) *f32_out = f; else free(f);
    return kept(q);
}

static void test_weights(mynah_slm_backend *b) {
    float *dummy = NULL;
    void *q = make_weight(INGOT_TYPE_Q8_0, 4, 64, &dummy);
    const mynah_slm_bweight *w1 = NULL, *w2 = NULL;
    const int r1 = mynah_slm_backend_weight(b, INGOT_TYPE_Q8_0, q, 4, 64, &w1, err, sizeof err);
    const int r2 = mynah_slm_backend_weight(b, INGOT_TYPE_Q8_0, q, 4, 64, &w2, err, sizeof err);
    check("the same bytes uploaded twice give ONE handle (tied embedding = one copy)",
          r1 == 0 && r2 == 0 && w1 == w2, err);
    check("a handle reports its geometry",
          mynah_slm_bweight_rows(w1) == 4 && mynah_slm_bweight_cols(w1) == 64 &&
          mynah_slm_bweight_type(w1) == INGOT_TYPE_Q8_0, NULL);

    /* Q1_0 (id 41) is the one ggml type nothing can decode: unsupported, not
     * an error, and no handle. */
    const mynah_slm_bweight *w3 = NULL;
    const int r3 = mynah_slm_backend_weight(b, 41, q, 4, 64, &w3, err, sizeof err);
    check("an undecodable type is UNSUPPORTED (1), not a failure", r3 == 1 && w3 == NULL, NULL);

    /* weights_flush empties the host-pointer cache (every handle so far is
     * invalid) and leaves the backend serving: the same bytes upload again
     * and multiply to the same bits. */
    static float x[64], y1[4], y2[4];
    fill(x, 64, 1.0f);
    const int m1 = mynah_slm_backend_matvec(b, w1, x, y1, err, sizeof err);
    mynah_slm_backend_weights_flush(b);
    const mynah_slm_bweight *w4 = NULL;
    const int r4 = mynah_slm_backend_weight(b, INGOT_TYPE_Q8_0, q, 4, 64, &w4, err, sizeof err);
    const int m2 = r4 == 0 ? mynah_slm_backend_matvec(b, w4, x, y2, err, sizeof err) : -1;
    check("after weights_flush the same bytes re-upload and give the same product",
          m1 == 0 && r4 == 0 && m2 == 0 && same(y1, y2, 4), err);
    free(dummy);
}

/* ── products ───────────────────────────────────────────────────────────── */

static void test_products(mynah_slm_backend *b) {
    static const int types[] = { INGOT_TYPE_F32, INGOT_TYPE_Q8_0, INGOT_TYPE_Q4_K, INGOT_TYPE_Q6_K };
    static const char *names[] = { "F32", "Q8_0", "Q4_K", "Q6_K" };
    enum { ROWS = 300, COLS = 1024, TOK = 5 };

    float *x  = mynah_slm_aligned_alloc((size_t)TOK * COLS * sizeof(float));
    /* Large enough for a matmat output (TOK x ROWS) and three embed rows. */
    float *y  = mynah_slm_aligned_alloc((size_t)TOK * COLS * sizeof(float));
    float *yr = mynah_slm_aligned_alloc((size_t)TOK * COLS * sizeof(float));
    float *strip = mynah_slm_aligned_alloc((size_t)MYNAH_SLM_STRIP_ROWS * COLS * sizeof(float));
    fill(x, (size_t)TOK * COLS, 1.0f);

    for (size_t t = 0; t < sizeof types / sizeof types[0]; t++) {
        char what[96];
        void *q = make_weight(types[t], ROWS, COLS, NULL);
        const mynah_slm_bweight *w = NULL;
        if (!q || mynah_slm_backend_weight(b, types[t], q, ROWS, COLS, &w, err, sizeof err) != 0) {
            snprintf(what, sizeof what, "%s weight uploads", names[t]);
            check(what, 0, err);
                    continue;
        }

        /* Reference: one serial call over the whole matrix, the path
         * mynah_slm_project takes when it does not split. */
        mynah_slm_matvec_in prep;
        const mynah_slm_matvec_in *p = NULL;
        if (mynah_slm_matvec_have(types[t])) { mynah_slm_matvec_prepare(x, COLS, &prep); p = &prep; }
        if (!(p && mynah_slm_matvec(types[t], q, ROWS, COLS, x, p, yr) == 0))
            ingot_matvec(types[t], q, ROWS, COLS, x, yr);
        const int rc = mynah_slm_backend_matvec(b, w, x, y, err, sizeof err);
        snprintf(what, sizeof what, "matvec %s 300x1024 (threaded row split) == serial kernel, bitwise", names[t]);
        check(what, rc == 0 && same(y, yr, ROWS), rc ? err : "bits differ");

        mynah_slm_qmatmat(types[t], q, ROWS, COLS, x, yr, TOK, strip);
        const int rm = mynah_slm_backend_matmat(b, w, x, y, TOK, err, sizeof err);
        snprintf(what, sizeof what, "matmat %s x5 tokens == mynah_slm_qmatmat, bitwise", names[t]);
        check(what, rm == 0 && same(y, yr, (size_t)TOK * ROWS), rm ? err : "bits differ");

        const uint32_t ids[3] = { 0, 299, 17 };
        for (int i = 0; i < 3; i++)
            ingot_dequant_matrix(types[t], (const uint8_t *)q + ids[i] * row_bytes(types[t], COLS),
                                 1, COLS, yr + (size_t)i * COLS);
        const int re = mynah_slm_backend_embed(b, w, ids, 3, y, err, sizeof err);
        snprintf(what, sizeof what, "embed %s rows {0,299,17} == ingot row decode, bitwise", names[t]);
        check(what, re == 0 && same(y, yr, 3 * COLS), re ? err : "bits differ");
    }

    const mynah_slm_bweight *w = NULL;
    void *q = make_weight(INGOT_TYPE_F32, 4, 32, NULL);
    mynah_slm_backend_weight(b, INGOT_TYPE_F32, q, 4, 32, &w, err, sizeof err);
    const uint32_t bad = 4;
    check("embed refuses a token id past the vocabulary",
          mynah_slm_backend_embed(b, w, &bad, 1, y, err, sizeof err) == -1, NULL);

    mynah_slm_aligned_free(x);
    mynah_slm_aligned_free(y);
    mynah_slm_aligned_free(yr);
    mynah_slm_aligned_free(strip);
}

/* ── norms, rope, elementwise ───────────────────────────────────────────── */

static void test_elementwise(mynah_slm_backend *b) {
    enum { D = 1024, HD = 128, NH = 16, T = 3, N = T * NH * HD };
    static float x[N], y[N], r[N], g[D], gh[HD], u[N];
    fill(x, N, 40.0f);              /* residual-stream magnitudes, not unit noise */
    for (int i = 0; i < D; i++) g[i] = 1.0f + frand() * 0.3f;
    for (int i = 0; i < HD; i++) gh[i] = 1.0f + frand() * 0.3f;

    const mynah_slm_bweight *wg = NULL, *wh = NULL;
    mynah_slm_backend_weight(b, INGOT_TYPE_F32, g, 1, D, &wg, err, sizeof err);
    mynah_slm_backend_weight(b, INGOT_TYPE_F32, gh, 1, HD, &wh, err, sizeof err);

    for (int t = 0; t < T; t++) mynah_slm_rms_norm(r + t * D, x + t * D, g, D, 1e-6f);
    int rc = mynah_slm_backend_rms_norm(b, y, x, wg, T, D, 1e-6f, err, sizeof err);
    check("rms_norm 3 rows == mynah_slm_rms_norm, bitwise", rc == 0 && same(y, r, (size_t)T * D), rc ? err : "bits differ");
    check("rms_norm refuses a gain of the wrong width",
          mynah_slm_backend_rms_norm(b, y, x, wh, T, D, 1e-6f, err, sizeof err) == -1, NULL);

    memcpy(r, x, sizeof x); memcpy(y, x, sizeof x);
    mynah_slm_rms_norm_per_head(r, gh, T * NH, HD, 1e-6f);
    rc = mynah_slm_backend_rms_norm_heads(b, y, wh, (size_t)T * NH, HD, 1e-6f, err, sizeof err);
    check("rms_norm_heads (QK-norm, 3 tokens x 16 heads x 128) == per_head, bitwise",
          rc == 0 && same(y, r, N), rc ? err : "bits differ");

    for (int il = 0; il <= 1; il++) {
        mynah_slm_rope tab;
        mynah_slm_rope_init(&tab, HD, 64, 1e6f, il);
        mynah_slm_brope *br = NULL;
        rc = mynah_slm_backend_rope_create(b, HD, 64, 1e6f, il, &br, err, sizeof err);
        memcpy(r, x, sizeof x); memcpy(y, x, sizeof x);
        for (int t = 0; t < T; t++) mynah_slm_rope_apply(&tab, r + t * NH * HD, NH, 40 + (uint32_t)t);
        if (rc == 0) rc = mynah_slm_backend_rope(b, br, y, T, NH, 40, err, sizeof err);
        check(il ? "rope interleaved, positions 40..42 == rope_apply, bitwise"
                 : "rope NeoX split-half, positions 40..42 == rope_apply, bitwise",
              rc == 0 && same(y, r, N), rc ? err : "bits differ");
        check("rope refuses positions past its table",
              mynah_slm_backend_rope(b, br, y, T, NH, 62, err, sizeof err) == -1, NULL);
        mynah_slm_backend_rope_free(b, br);
        mynah_slm_rope_free(&tab);
    }

    fill(u, N, 3.0f);
    memcpy(r, x, sizeof x); memcpy(y, x, sizeof x);
    mynah_slm_swiglu(r, u, N);
    rc = mynah_slm_backend_swiglu(b, y, u, N, err, sizeof err);
    check("swiglu == mynah_slm_swiglu, bitwise", rc == 0 && same(y, r, N), rc ? err : "bits differ");

    memcpy(r, x, sizeof x); memcpy(y, x, sizeof x);
    mynah_slm_add(r, u, N);
    rc = mynah_slm_backend_add(b, y, u, N, err, sizeof err);
    check("add == mynah_slm_add, bitwise", rc == 0 && same(y, r, N), rc ? err : "bits differ");

    memcpy(r, x, sizeof x); memcpy(y, x, sizeof x);
    mynah_slm_add_scaled(r, u, 0.263f, N);
    rc = mynah_slm_backend_add_scaled(b, y, u, 0.263f, N, err, sizeof err);
    check("add_scaled == mynah_slm_add_scaled, bitwise", rc == 0 && same(y, r, N), rc ? err : "bits differ");

    /* Ties go to the first index — what greedy decoding on the CPU does. */
    float v[8] = { 1, 5, 2, 5, -1, 0, 5, 3 };
    uint32_t idx = 99;
    rc = mynah_slm_backend_argmax(b, v, 8, &idx, err, sizeof err);
    check("argmax returns the FIRST maximum", rc == 0 && idx == 1, rc ? err : "bits differ");
}

/* ── KV cache and attention, at three precisions ────────────────────────── */

static void test_attention(mynah_slm_backend *b, mynah_slm_kv_type kt) {
    /* Qwen3-0.6B's attention shape: 16 query heads over 8 KV heads of 128. */
    enum { L = 2, CTX = 64, NH = 16, NKV = 8, HD = 128, MB = 8, FILL = 20 };
    const uint32_t kv_dim = NKV * HD, q_dim = NH * HD;
    const float scale = 1.0f / sqrtf((float)HD);

    mynah_slm_bkv_desc d = { kt, kt, L, CTX, NH, NKV, HD, MB };
    mynah_slm_bkv *kv = NULL;
    int rc = mynah_slm_backend_kv_create(b, &d, &kv, err, sizeof err);
    mynah_slm_kv ref;
    mynah_slm_kv_init(&ref, kt, kt, L, CTX, NKV, HD);

    static float k[FILL * NKV * HD], v[FILL * NKV * HD], q[MB * NH * HD];
    static float out[MB * NH * HD], outr[MB * NH * HD];
    float *scores  = mynah_slm_aligned_alloc((size_t)NH * CTX * MB * sizeof(float));
    float *kg = mynah_slm_aligned_alloc((size_t)CTX * HD * sizeof(float));
    float *vg = mynah_slm_aligned_alloc((size_t)CTX * HD * sizeof(float));
    fill(k, sizeof k / sizeof k[0], 2.0f);
    fill(v, sizeof v / sizeof v[0], 2.0f);
    fill(q, sizeof q / sizeof q[0], 2.0f);

    /* Layer 1, so a missing per-layer offset cannot pass by reading layer 0. */
    if (rc == 0) rc = mynah_slm_backend_kv_append(b, kv, 1, 0, FILL, k, v, err, sizeof err);
    for (uint32_t t = 0; t < FILL; t++) {
        mynah_slm_kv_put_k(&ref, 1, t, k + (size_t)t * kv_dim);
        mynah_slm_kv_put_v(&ref, 1, t, v + (size_t)t * kv_dim);
    }

    const int f32 = kt == MYNAH_SLM_KV_F32;
    const size_t per_layer = (size_t)CTX * kv_dim;
    const float *kl = (const float *)ref.k + per_layer;
    const float *vl = (const float *)ref.v + per_layer;
    char what[112];

    /* Decode: one query at position FILL-1, reading all FILL positions. */
    if (f32) mynah_slm_attention_mt(outr, q, kl, vl, FILL, NH, NKV, HD, scale, scores);
    else     mynah_slm_attention_kv_mt(outr, q, &ref, 1, FILL, NH, NKV, HD, scale, scores);
    if (rc == 0) rc = mynah_slm_backend_attention(b, kv, 1, q, out, FILL - 1, 1, scale, err, sizeof err);
    snprintf(what, sizeof what, "attention %s KV, decode (GQA 16/8, head_dim 128) == reference, bitwise",
             mynah_slm_kv_type_name(kt));
    check(what, rc == 0 && same(out, outr, q_dim), rc ? err : "bits differ");

    /* Prefill: 4 queries at positions 16..19, causal triangle. */
    if (f32) mynah_slm_attention_batch(outr, q, kl, vl, 16, 4, NH, NKV, HD, q_dim, scale, scores);
    else     mynah_slm_attention_kv_batch(outr, q, &ref, 1, 16, 4, NH, NKV, HD, q_dim, scale,
                                          scores, kg, vg);
    if (rc == 0) rc = mynah_slm_backend_attention(b, kv, 1, q, out, 16, 4, scale, err, sizeof err);
    snprintf(what, sizeof what, "attention %s KV, batch of 4 (causal) == reference, bitwise",
             mynah_slm_kv_type_name(kt));
    check(what, rc == 0 && same(out, outr, (size_t)4 * q_dim), rc ? err : "bits differ");

    if (kt == MYNAH_SLM_KV_F32) {
        check("attention refuses a batch wider than max_batch",
              mynah_slm_backend_attention(b, kv, 1, q, out, 0, MB + 1, scale, err, sizeof err) == -1, NULL);
        check("kv_append refuses positions past n_ctx",
              mynah_slm_backend_kv_append(b, kv, 0, CTX - 1, 2, k, v, err, sizeof err) == -1, NULL);
        check("kv_append refuses a layer past n_layers",
              mynah_slm_backend_kv_append(b, kv, L, 0, 1, k, v, err, sizeof err) == -1, NULL);
        mynah_slm_bkv_desc bad = d;
        bad.n_kv_heads = 5;     /* 16 heads cannot be grouped over 5 */
        mynah_slm_bkv *kv2 = NULL;
        check("kv_create refuses n_heads not a multiple of n_kv_heads",
              mynah_slm_backend_kv_create(b, &bad, &kv2, err, sizeof err) == -1 && !kv2, NULL);
    }

    mynah_slm_backend_kv_free(b, kv);
    mynah_slm_kv_free(&ref);
    mynah_slm_aligned_free(scores);
    mynah_slm_aligned_free(kg);
    mynah_slm_aligned_free(vg);
}

/* ── the slot pool: release and reuse, the cancellation path ───────────── */

static void test_slots(mynah_slm_backend *b) {
    enum { CTX = 32, NH = 16, NKV = 8, HD = 128, SCR = 256 };
    const uint32_t kv_dim = NKV * HD, q_dim = NH * HD;
    const float scale = 1.0f / sqrtf((float)HD);
    mynah_slm_bslots_desc d = {
        { MYNAH_SLM_KV_BF16, MYNAH_SLM_KV_BF16, 1, CTX, NH, NKV, HD, 4 }, SCR, 3 };
    mynah_slm_bslots *p = NULL;
    int rc = mynah_slm_backend_slots_create(b, &d, &p, err, sizeof err);
    check("slot pool of 3 is created up front", rc == 0 && p != NULL, err);
    if (rc != 0) return;

    uint32_t s[4] = { 9, 9, 9, 9 };
    uint64_t g[4] = { 0 };
    int ok = 1;
    for (int i = 0; i < 3; i++)
        ok = ok && mynah_slm_backend_slot_acquire(b, p, &s[i], &g[i], err, sizeof err) == 0;
    check("three acquires give slots 0, 1, 2", ok && s[0] == 0 && s[1] == 1 && s[2] == 2, err);
    check("a fourth acquire is BUSY (1), not an error and not an allocation",
          mynah_slm_backend_slot_acquire(b, p, &s[3], &g[3], err, sizeof err) == 1, NULL);
    check("held count is 3", mynah_slm_backend_slots_held(p) == 3, NULL);

    /* Slot 1's request writes some history, then is cancelled mid-sequence. */
    mynah_slm_bkv *kv_before = mynah_slm_backend_slot_kv(p, 1);
    float *scr_before = mynah_slm_backend_slot_scratch(p, 1);
    static float k[5 * NKV * HD], v[5 * NKV * HD], q[NH * HD], out[NH * HD], ref[NH * HD];
    fill(k, sizeof k / sizeof k[0], 2.0f);
    fill(v, sizeof v / sizeof v[0], 2.0f);
    fill(q, sizeof q / sizeof q[0], 2.0f);
    mynah_slm_backend_kv_append(b, kv_before, 0, 0, 5, k, v, err, sizeof err);

    rc = mynah_slm_backend_slot_release(b, p, 1, err, sizeof err);
    check("cancelling a row releases its slot at the step boundary", rc == 0, err);
    check("the generation moves on release (stale outputs are recognisable)",
          mynah_slm_backend_slot_generation(p, 1) == g[1] + 1, NULL);
    check("a double release is refused (-1)",
          mynah_slm_backend_slot_release(b, p, 1, err, sizeof err) == -1, NULL);

    uint32_t again = 9;
    uint64_t gen = 0;
    rc = mynah_slm_backend_slot_acquire(b, p, &again, &gen, err, sizeof err);
    check("the released slot is handed out again (lowest free index)",
          rc == 0 && again == 1 && gen == g[1] + 1, err);
    check("reuse makes no allocation: same KV handle, same scratch",
          mynah_slm_backend_slot_kv(p, 1) == kv_before &&
          mynah_slm_backend_slot_scratch(p, 1) == scr_before, NULL);

    /* The new request writes position 0 and attends to it. The previous
     * occupant's positions 1..4 are still in the cache and must be invisible:
     * the answer equals a fresh cache that only ever saw the new row. */
    const float *k_new = k + 2 * kv_dim, *v_new = v + 2 * kv_dim;
    mynah_slm_backend_kv_append(b, kv_before, 0, 0, 1, k_new, v_new, err, sizeof err);
    rc = mynah_slm_backend_attention(b, kv_before, 0, q, out, 0, 1, scale, err, sizeof err);
    mynah_slm_kv fresh;
    mynah_slm_kv_init(&fresh, MYNAH_SLM_KV_BF16, MYNAH_SLM_KV_BF16, 1, CTX, NKV, HD);
    mynah_slm_kv_put_k(&fresh, 0, 0, k_new);
    mynah_slm_kv_put_v(&fresh, 0, 0, v_new);
    float *scores = mynah_slm_aligned_alloc((size_t)NH * CTX * sizeof(float));
    mynah_slm_attention_kv_mt(ref, q, &fresh, 0, 1, NH, NKV, HD, scale, scores);
    check("a reused slot sees none of the cancelled request's history, bitwise",
          rc == 0 && same(out, ref, q_dim), rc ? err : "bits differ");
    mynah_slm_aligned_free(scores);
    mynah_slm_kv_free(&fresh);

    check("a synchronous backend always recovers (nothing pending)",
          mynah_slm_backend_recover(b, err, sizeof err) == 0, err);

    for (int i = 0; i < 3; i++) mynah_slm_backend_slot_release(b, p, (uint32_t)i, err, sizeof err);
    check("all slots released", mynah_slm_backend_slots_held(p) == 0, NULL);
    mynah_slm_backend_slots_destroy(b, p);

    /* A KV precision the backend refuses makes the pool refuse too. */
    d.kv.n_heads = 15;
    p = NULL;
    check("a pool over an invalid KV description is refused",
          mynah_slm_backend_slots_create(b, &d, &p, err, sizeof err) != 0 && p == NULL, NULL);
}

int main(void) {
    /* Threaded: the matvec row split and the per-head attention tasks must
     * stay bit-identical with the pool actually running. */
    mynah_slm_threads_init(4);

    mynah_slm_backend *b = NULL;
    if (mynah_slm_backend_open(MYNAH_SLM_DEVICE_CPU, &b, err, sizeof err) != 0) {
        printf("FAIL cannot open the cpu backend: %s\n", err);
        return 1;
    }
    test_open(b);
    test_weights(b);
    test_products(b);
    test_elementwise(b);
    test_attention(b, MYNAH_SLM_KV_F32);
    test_attention(b, MYNAH_SLM_KV_BF16);
    test_attention(b, MYNAH_SLM_KV_Q8);
    test_slots(b);
    mynah_slm_backend_close(b);
    for (int i = 0; i < n_keep; i++) free(keep[i]);
    mynah_slm_threads_shutdown();

    if (failures) { printf("%d backend check(s) FAILED\n", failures); return 1; }
    printf("all backend checks passed\n");
    return 0;
}
