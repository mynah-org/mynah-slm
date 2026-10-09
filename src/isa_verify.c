/* isa_verify.c — verify-on-first-use, before a vector table is trusted.
 *
 * mynah-tts's rule (src/qmat.c, "THE PROVE-ON-FIRST-USE GATE"): a kernel
 * nobody here could execute must not resolve on the strength of a CPUID bit
 * and a careful reading. Each candidate table is run once per process against
 * the scalar table on small, deliberately awkward fixtures — row counts that
 * are not a multiple of four, an odd number of Q8_0 blocks, lengths that leave
 * vector tails — and a disagreement drops that table, visibly
 * (`mynah-slm --dispatch`), instead of shipping a wrong answer fast.
 *
 * The int8 kernels are held to their CONTRACT: bit-identical to the scalar
 * twin (memcmp). The f32 kernels are held to a tolerance, because their scalar
 * twins round in a different order by design; the tolerance catches a wrong
 * shuffle or a wrong nibble order (errors of order 1), not a reordered sum.
 *
 * Small on purpose: this runs at model load (or at first use), single
 * threaded, and must cost well under a millisecond.
 *
 * SPDX-License-Identifier: MIT */
#include "isa.h"

#include "kern.h"
#include "kvcache.h"
#include "qmat.h"

#include "ingot/dtype.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static uint32_t vrand(uint32_t *s) {
    *s = *s * 1664525u + 1013904223u;
    return *s >> 8;
}

static float vfrand(uint32_t *s) {
    return (float)((int)(vrand(s) % 20001u) - 10000) / 10000.0f;
}

static void put_f16(unsigned char *p, float v) {
    const uint16_t h = ingot_f32_to_f16(v);
    p[0] = (unsigned char)(h & 0xffu);
    p[1] = (unsigned char)(h >> 8);
}

static void fill_bytes(unsigned char *p, size_t n, uint32_t seed) {
    for (size_t i = 0; i < n; i++) p[i] = (unsigned char)vrand(&seed);
}

/* max |a - b| / max |b|; a NaN anywhere fails */
static double rel_diff(const float *a, const float *b, size_t n) {
    double worst = 0.0, scale = 0.0;
    for (size_t i = 0; i < n; i++) {
        if (!isfinite(a[i]) || !isfinite(b[i])) return INFINITY;
        const double d = fabs((double)a[i] - (double)b[i]);
        if (d > worst) worst = d;
        if (fabs((double)b[i]) > scale) scale = fabs((double)b[i]);
    }
    return scale > 0.0 ? worst / scale : worst;
}

/* ── qmat ──────────────────────────────────────────────────────────────────*/

enum { VR = 6, VC = 512, VC8 = 96, VNT = 5 };   /* 5 tokens: a group of 4 + a tail */

/* The weight-stationary entries (K7) against the SAME table's single-token
 * ones, token by token, memcmp: that equality is their whole contract (the
 * f32 kernel is per-ISA, so the reference is k's own, not the scalar one).
 * Two distinct activation vectors alternate over the five tokens, so a token
 * reading its neighbour's activations shows up. */
static int verify_ws(const mynah_slm_qmat_kern *k, const unsigned char *w4,
                     const unsigned char *w6, const unsigned char *w8,
                     const float *x, const mynah_slm_matvec_in *p,
                     const float *x8, const mynah_slm_matvec_in *p8, char *why, size_t n) {
    static float xr[VC], xr8[VC8];
    static mynah_slm_matvec_in q, q8;
    for (int i = 0; i < VC; i++)  xr[i]  = x[VC - 1 - i];
    for (int i = 0; i < VC8; i++) xr8[i] = x8[VC8 - 1 - i];
    mynah_slm_matvec_prepare_int8(xr, VC, &q);
    mynah_slm_matvec_prepare_int8(xr8, VC8, &q8);

    float got[VNT][VR], want[VNT][VR];
    const float  *xs[VNT], *sm[VNT], *sc[VNT], *sc8[VNT];
    const int8_t *xq[VNT], *xq8[VNT];
    float        *o[VNT];
    for (int t = 0; t < VNT; t++) {
        const mynah_slm_matvec_in *pt = (t & 1) ? &q : p, *pt8 = (t & 1) ? &q8 : p8;
        xs[t] = (t & 1) ? xr : x;
        sm[t] = pt->xsum;  sc[t] = pt->xscale;  xq[t] = pt->xq;
        sc8[t] = pt8->xscale; xq8[t] = pt8->xq;
        o[t] = got[t];
    }

    if (!k->q4k_f32_ws) { snprintf(why, n, "no q4_k f32 ws kernel"); return -1; }
    k->q4k_f32_ws(w4, VR, VC / 256, VNT, xs, sm, o);
    for (int t = 0; t < VNT; t++) k->q4k_f32(w4, VR, VC / 256, xs[t], sm[t], want[t]);
    if (memcmp(got, want, sizeof got) != 0) { snprintf(why, n, "q4_k f32 ws != single"); return -1; }

    if (!k->int8) return 0;
    k->q4k_i8_ws(w4, VR, VC / 256, VNT, xq, sc, sm, o);
    for (int t = 0; t < VNT; t++) k->q4k_i8(w4, VR, VC / 256, xq[t], sc[t], sm[t], want[t]);
    if (memcmp(got, want, sizeof got) != 0) { snprintf(why, n, "q4_k int8 ws != single"); return -1; }
    k->q80_i8_ws(w8, VR, VC8 / 32, VNT, xq8, sc8, o);
    for (int t = 0; t < VNT; t++) k->q80_i8(w8, VR, VC8 / 32, xq8[t], sc8[t], want[t]);
    if (memcmp(got, want, sizeof got) != 0) { snprintf(why, n, "q8_0 int8 ws != single"); return -1; }
    k->q6k_i8_ws(w6, VR, VC / 256, VNT, xq, sc, o);
    for (int t = 0; t < VNT; t++) k->q6k_i8(w6, VR, VC / 256, xq[t], sc[t], want[t]);
    if (memcmp(got, want, sizeof got) != 0) { snprintf(why, n, "q6_k int8 ws != single"); return -1; }
    return 0;
}

int mynah_slm_isa_verify_qmat(const mynah_slm_qmat_kern *k, char *why, size_t n) {
    const mynah_slm_qmat_kern *ref = &mynah_slm_qmat_kern_scalar;
    enum { R = VR, C = VC, C8 = VC8 };           /* 4 + 2 rows; 3 Q8_0 blocks */
    static unsigned char w4[R * (C / 256) * 144], w6[R * (C / 256) * 210],
                         w8[R * (C8 / 32) * 34];
    static mynah_slm_matvec_in p, p8;            /* ~22 KB each: not on a stack */
    float x[C], x8[C8], a[R], b[R];

    fill_bytes(w4, sizeof w4, 11u);
    fill_bytes(w6, sizeof w6, 12u);
    fill_bytes(w8, sizeof w8, 13u);
    uint32_t s = 14u;
    for (size_t i = 0; i < sizeof w4 / 144; i++) {
        put_f16(w4 + i * 144,     0.01f + 0.01f * (float)(vrand(&s) % 7u));
        put_f16(w4 + i * 144 + 2, 0.005f * (float)(vrand(&s) % 5u));
    }
    for (size_t i = 0; i < sizeof w6 / 210; i++) put_f16(w6 + i * 210 + 208, 0.001f);
    for (size_t i = 0; i < sizeof w8 / 34; i++)  put_f16(w8 + i * 34, 0.002f);
    for (int i = 0; i < C; i++)  x[i]  = vfrand(&s) * ((i % 37 == 0) ? 9.0f : 1.0f);
    for (int i = 0; i < C8; i++) x8[i] = vfrand(&s);
    /* The quantizer's two edges in both fixtures: a block whose amax is too
     * small to scale (it must quantize to zero, never saturate to -128), and
     * a block of exact +-max. Either one broke AVX2's sign(xq, w) once. */
    for (int i = 32; i < 64; i++) {
        x[i] = x8[i] = (float)(i % 5 - 2) * 1e-37f;
        x[i + 32] = x8[i + 32] = (i & 1) ? -3.5f : 3.5f;
    }
    mynah_slm_matvec_prepare_int8(x, C, &p);
    mynah_slm_matvec_prepare_int8(x8, C8, &p8);

    k->q4k_f32(w4, R, C / 256, x, p.xsum, a);
    ref->q4k_f32(w4, R, C / 256, x, p.xsum, b);
    const double rel = rel_diff(a, b, R);
    if (!(rel < 1e-4)) { snprintf(why, n, "q4_k f32 rel %.1e", rel); return -1; }

    if (!k->int8) return verify_ws(k, w4, w6, w8, x, &p, x8, &p8, why, n);
    k->q4k_i8(w4, R, C / 256, p.xq, p.xscale, p.xsum, a);
    ref->q4k_i8(w4, R, C / 256, p.xq, p.xscale, p.xsum, b);
    if (memcmp(a, b, sizeof a) != 0) { snprintf(why, n, "q4_k int8 != twin"); return -1; }
    k->q80_i8(w8, R, C8 / 32, p8.xq, p8.xscale, a);
    ref->q80_i8(w8, R, C8 / 32, p8.xq, p8.xscale, b);
    if (memcmp(a, b, sizeof a) != 0) { snprintf(why, n, "q8_0 int8 != twin"); return -1; }
    k->q6k_i8(w6, R, C / 256, p.xq, p.xscale, a);
    ref->q6k_i8(w6, R, C / 256, p.xq, p.xscale, b);
    if (memcmp(a, b, sizeof a) != 0) { snprintf(why, n, "q6_k int8 != twin"); return -1; }
    return verify_ws(k, w4, w6, w8, x, &p, x8, &p8, why, n);
}

/* ── attn ──────────────────────────────────────────────────────────────────*/

int mynah_slm_isa_verify_attn(const mynah_slm_attn_kern *k, char *why, size_t n) {
    const mynah_slm_attn_kern *ref = &mynah_slm_attn_kern_scalar;
    enum { H = 4, KVH = 2, HD = 64, NKV = 7 };
    static float q[H * HD], kk[NKV * KVH * HD], vv[NKV * KVH * HD];
    float oa[H * HD], ob[H * HD], sa[NKV], sb[NKV];
    const float scale = 0.125f;
    uint32_t s = 21u;
    for (int i = 0; i < H * HD; i++) q[i] = vfrand(&s);
    for (int i = 0; i < NKV * KVH * HD; i++) { kk[i] = vfrand(&s); vv[i] = vfrand(&s); }

    for (uint32_t h = 0; h < H; h++) {
        k->f32_head(oa, q, kk, vv, h, NKV, H, KVH, HD, scale, sa);
        ref->f32_head(ob, q, kk, vv, h, NKV, H, KVH, HD, scale, sb);
    }
    double rel = rel_diff(oa, ob, H * HD);
    if (!(rel < 1e-4)) { snprintf(why, n, "f32 head rel %.1e", rel); return -1; }

    const mynah_slm_kv_type types[] = { MYNAH_SLM_KV_BF16, MYNAH_SLM_KV_Q8,
                                        MYNAH_SLM_KV_Q4, MYNAH_SLM_KV_F32 };
    for (size_t t = 0; t < sizeof types / sizeof *types; t++) {
        mynah_slm_kv c;
        if (mynah_slm_kv_init(&c, types[t], types[t], 1, NKV, KVH, HD) != 0) {
            snprintf(why, n, "kv init failed");
            return -1;
        }
        for (uint32_t pos = 0; pos < NKV; pos++) {
            mynah_slm_kv_put_k(&c, 0, pos, kk + (size_t)pos * KVH * HD);
            mynah_slm_kv_put_v(&c, 0, pos, vv + (size_t)pos * KVH * HD);
        }
        k->kv_head(oa, q, &c, 0, 1, NKV, scale, sa);
        ref->kv_head(ob, q, &c, 0, 1, NKV, scale, sb);
        mynah_slm_kv_free(&c);
        rel = rel_diff(oa, ob, HD);
        if (!(rel < 1e-4)) {
            snprintf(why, n, "kv %s head rel %.1e", mynah_slm_kv_type_name(types[t]), rel);
            return -1;
        }
    }
    return 0;
}

/* ── sgemm ─────────────────────────────────────────────────────────────────
 * Shapes far below the planner's parallel threshold, so this never touches
 * the thread pool — it may run inside a pool worker on first use. */
int mynah_slm_isa_verify_sgemm(const mynah_slm_sgemm_kern *k, char *why, size_t n) {
    const mynah_slm_sgemm_kern *ref = &mynah_slm_sgemm_kern_scalar;
    static float a[7 * 37], bt[5 * 37], bn[11 * 19], ca[7 * 19], cb[7 * 19];
    uint32_t s = 31u;
    for (size_t i = 0; i < sizeof a / sizeof *a; i++)   a[i]  = vfrand(&s);
    for (size_t i = 0; i < sizeof bt / sizeof *bt; i++) bt[i] = vfrand(&s);
    for (size_t i = 0; i < sizeof bn / sizeof *bn; i++) bn[i] = vfrand(&s);

    /* NT 7x5x37: k leaves a tail on every width; rows and columns leave edges */
    k->run(1, 7, 5, 37, 1.0f, a, 37, bt, 37, 0.0f, ca, 5);
    ref->run(1, 7, 5, 37, 1.0f, a, 37, bt, 37, 0.0f, cb, 5);
    double rel = rel_diff(ca, cb, 7 * 5);
    if (!(rel < 1e-5)) { snprintf(why, n, "NT rel %.1e", rel); return -1; }

    /* NN 6x19x11 with beta != 0: C is read */
    for (int i = 0; i < 6 * 19; i++) ca[i] = cb[i] = vfrand(&s);
    k->run(0, 6, 19, 11, 0.75f, a, 37, bn, 19, 0.5f, ca, 19);
    ref->run(0, 6, 19, 11, 0.75f, a, 37, bn, 19, 0.5f, cb, 19);
    rel = rel_diff(ca, cb, 6 * 19);
    if (!(rel < 1e-5)) { snprintf(why, n, "NN rel %.1e", rel); return -1; }
    return 0;
}
