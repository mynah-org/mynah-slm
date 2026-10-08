/* test_isa.c — runtime ISA dispatch (src/isa.c, .work/isa-runtime-dispatch.md).
 *
 * What a dispatch layer can get wrong, each checked here without a model:
 *
 *   - the probe disagrees with itself (a level resolved that the CPU flags
 *     do not support);
 *   - MYNAH_SLM_ISA WIDENS instead of narrowing, or an unknown / other-arch
 *     name changes anything;
 *   - a family resolves to a table above the ceiling, or skips a better one
 *     that was compiled and allowed;
 *   - verify-on-first-use passes a broken table (checked with two deliberately
 *     broken tables: a wrong f32 kernel, and an int8 kernel off by ONE ulp);
 *   - a vector table disagrees with the scalar one beyond the f32 tolerance,
 *     or a threaded run differs from a 1-thread run, at ANY level the CPU can
 *     run — not only at the level this machine picks by default.
 *
 * SPDX-License-Identifier: MIT */
#include "isa.h"

#include "kern.h"
#include "kernels.h"
#include "kvcache.h"
#include "qfixture.h"
#include "qmat.h"
#include "sgemm.h"
#include "threads.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures;

static void check(const char *what, int ok, const char *detail) {
    if (ok) printf("ok   %s\n", what);
    else { printf("FAIL %s  <- %s\n", what, detail ? detail : ""); failures++; }
}

/* the ladder of this architecture, best first */
static const char *const all_levels[] = { "avx512vnni", "avx512", "avx2", "dotprod", "neon", "scalar" };
#define N_LEVELS (sizeof all_levels / sizeof *all_levels)

static int rank_of(int id) {
    switch (id) {
    case MYNAH_SLM_KERN_ID_AVX2: case MYNAH_SLM_KERN_ID_NEON: return 1;
    case MYNAH_SLM_KERN_ID_AVX512: case MYNAH_SLM_KERN_ID_NEON_DOTPROD: return 2;
    case MYNAH_SLM_KERN_ID_AVX512VNNI: return 3;
    default: return 0;
    }
}

static void test_probe(void) {
    const mynah_slm_cpu *c = mynah_slm_cpu_caps();
    const int det = mynah_slm_isa_detected();
    char d[160];
    snprintf(d, sizeof d, "detected %s", mynah_slm_isa_level_name(det));
    printf("     %s\n", d);
    int ok = 1;
    if (det == MYNAH_SLM_KERN_ID_AVX2 || det == MYNAH_SLM_KERN_ID_AVX512 ||
        det == MYNAH_SLM_KERN_ID_AVX512VNNI)
        ok &= c->x86 && c->avx2 && c->fma && c->f16c;
    if (det == MYNAH_SLM_KERN_ID_AVX512 || det == MYNAH_SLM_KERN_ID_AVX512VNNI)
        ok &= c->avx512f && c->avx512bw && c->avx512vl && c->avx512dq;
    if (det == MYNAH_SLM_KERN_ID_AVX512VNNI) ok &= c->avx512vnni;
    if (det == MYNAH_SLM_KERN_ID_NEON_DOTPROD) ok &= c->arm64 && c->dotprod;
    check("the detected level is backed by the probed CPU flags", ok, d);
#if defined(__x86_64__)
    check("x86 never detects an arm level", c->x86 && !c->arm64 &&
          det != MYNAH_SLM_KERN_ID_NEON && det != MYNAH_SLM_KERN_ID_NEON_DOTPROD, d);
#elif defined(__aarch64__)
    check("arm64 detects neon at least", c->arm64 && c->neon && rank_of(det) >= 1, d);
#endif
}

static void test_narrowing(void) {
    const int det = mynah_slm_isa_detected();
    for (size_t i = 0; i < N_LEVELS; i++) {
        const int id = mynah_slm_isa_parse(all_levels[i]);
        if (id < 0) continue;                       /* not this architecture */
        const int got = mynah_slm_isa_narrow(all_levels[i]);
        const int want = rank_of(id) <= rank_of(det) ? id : det;
        char what[128], d[160];
        snprintf(what, sizeof what, "MYNAH_SLM_ISA=%s -> ceiling %s", all_levels[i],
                 mynah_slm_isa_level_name(want));
        snprintf(d, sizeof d, "got %s", mynah_slm_isa_level_name(got));
        check(what, got == want, d);

        const int q = mynah_slm_kern_qmat()->id, a = mynah_slm_kern_attn()->id,
                  s = mynah_slm_kern_sgemm()->id;
        snprintf(what, sizeof what, "  under %s: qmat %s attn %s sgemm %s, none above the ceiling",
                 mynah_slm_isa_level_name(got), mynah_slm_kern_qmat()->name,
                 mynah_slm_kern_attn()->name, mynah_slm_kern_sgemm()->name);
        check(what, rank_of(q) <= rank_of(got) && rank_of(a) <= rank_of(got) &&
              rank_of(s) <= rank_of(got), "a family exceeded the ceiling");
        /* and none BELOW what was compiled and allowed: the best is taken */
#if defined(__x86_64__)
        const int best_q = got == MYNAH_SLM_KERN_ID_AVX512 ? MYNAH_SLM_KERN_ID_AVX2 : got;
        const int best_a = rank_of(got) >= 1 ? MYNAH_SLM_KERN_ID_AVX2 : 0;
        const int best_s = got == MYNAH_SLM_KERN_ID_AVX512VNNI ? MYNAH_SLM_KERN_ID_AVX512 : got;
#elif defined(__aarch64__)
        const int best_q = got;
        const int best_a = rank_of(got) >= 1 ? MYNAH_SLM_KERN_ID_NEON : 0;
        const int best_s = rank_of(got) >= 1 ? MYNAH_SLM_KERN_ID_NEON : 0;
#else
        const int best_q = 0, best_a = 0, best_s = 0;
#endif
        check("  ...and each family takes the best table it is allowed",
              q == best_q && a == best_a && s == best_s, "a better compiled table was skipped");
    }
    check("an unknown name is refused and changes nothing",
          mynah_slm_isa_narrow("avx1024") == -1, NULL);
#if defined(__x86_64__)
    check("an arm level on x86 is refused", mynah_slm_isa_parse("dotprod") == -1, NULL);
#elif defined(__aarch64__)
    check("an x86 level on arm64 is refused", mynah_slm_isa_parse("avx2") == -1, NULL);
#endif
    mynah_slm_isa_narrow(NULL);
    check("narrow(NULL) restores the default ceiling",
          mynah_slm_isa_ceiling() == det || getenv("MYNAH_SLM_ISA") != NULL, NULL);
}

/* ── the gate must have teeth ─────────────────────────────────────────────*/

static const mynah_slm_qmat_kern *g_wrapped;

static void bad_q4k_f32(const unsigned char *w, size_t rows, size_t blocks,
                        const float *x, const float *xsum, float *out) {
    g_wrapped->q4k_f32(w, rows, blocks, x, xsum, out);
    out[rows - 1] = -out[rows - 1] + 1.0f;          /* a "wrong shuffle" */
}

static void ulp_q4k_i8(const unsigned char *w, size_t rows, size_t blocks,
                       const int8_t *xq, const float *xscale, const float *xsum,
                       float *out) {
    g_wrapped->q4k_i8(w, rows, blocks, xq, xscale, xsum, out);
    out[0] = nextafterf(out[0], INFINITY);          /* one ulp: a reordered sum */
}

static void test_verify_teeth(void) {
    char why[128];
    for (size_t i = 0; i < N_LEVELS; i++) {
        const int id = mynah_slm_isa_parse(all_levels[i]);
        if (id < 0 || mynah_slm_isa_narrow(all_levels[i]) != id) continue;
        char what[128];
        snprintf(what, sizeof what, "verify passes the resolved tables at %s", all_levels[i]);
        why[0] = '\0';
        int ok = mynah_slm_isa_verify_qmat(mynah_slm_kern_qmat(), why, sizeof why) == 0 &&
                 mynah_slm_isa_verify_attn(mynah_slm_kern_attn(), why, sizeof why) == 0 &&
                 mynah_slm_isa_verify_sgemm(mynah_slm_kern_sgemm(), why, sizeof why) == 0;
        check(what, ok, why);
    }
    mynah_slm_isa_narrow(NULL);

    mynah_slm_qmat_kern bad = *mynah_slm_kern_qmat();
    g_wrapped = mynah_slm_kern_qmat();
    bad.q4k_f32 = bad_q4k_f32;
    why[0] = '\0';
    check("verify REJECTS a wrong f32 kernel",
          mynah_slm_isa_verify_qmat(&bad, why, sizeof why) != 0, "it passed");
    printf("     (%s)\n", why);
    if (g_wrapped->int8) {
        bad = *g_wrapped;
        bad.q4k_i8 = ulp_q4k_i8;
        why[0] = '\0';
        check("verify REJECTS an int8 kernel one ulp off its twin",
              mynah_slm_isa_verify_qmat(&bad, why, sizeof why) != 0, "it passed");
        printf("     (%s)\n", why);
    }
}

/* ── every level against scalar, and threads, on the real shapes ─────────*/

static double rel_diff(const float *a, const float *b, size_t n) {
    double w = 0.0, s = 0.0;
    for (size_t i = 0; i < n; i++) {
        const double d = fabs((double)a[i] - (double)b[i]);
        if (d > w) w = d;
        if (fabs((double)b[i]) > s) s = fabs((double)b[i]);
    }
    return s > 0.0 ? w / s : w;
}

static void test_levels_agree(void) {
    enum { M = 64, N = 128, K = 1024, H = 16, KVH = 8, HD = 128, NKV = 77 };
    float *a = malloc(M * K * 4), *b = malloc(N * K * 4), *c1 = malloc(M * N * 4),
          *c4 = malloc(M * N * 4), *cref = malloc(M * N * 4);
    float *q = malloc(H * HD * 4), *kk = malloc(NKV * KVH * HD * 4), *vv = malloc(NKV * KVH * HD * 4);
    float *o1 = malloc(H * HD * 4), *o4 = malloc(H * HD * 4), *oref = malloc(H * HD * 4),
          *scr = malloc(H * NKV * 4);
    if (!a || !b || !c1 || !c4 || !cref || !q || !kk || !vv || !o1 || !o4 || !oref || !scr) {
        check("allocations", 0, "oom");
        return;
    }
    qfx_activations(a, M * K, 1); qfx_activations(b, N * K, 2);
    qfx_activations(q, H * HD, 3); qfx_activations(kk, NKV * KVH * HD, 4);
    qfx_activations(vv, NKV * KVH * HD, 5);
    mynah_slm_sgemm_reference(1, M, N, K, 1.0f, a, K, b, K, 0.0f, cref, N);

    mynah_slm_kv cache;
    if (mynah_slm_kv_init(&cache, MYNAH_SLM_KV_BF16, MYNAH_SLM_KV_BF16, 1, NKV, KVH, HD) != 0) {
        check("kv init", 0, NULL);
        return;
    }
    for (int p = 0; p < NKV; p++) {
        mynah_slm_kv_put_k(&cache, 0, (uint32_t)p, kk + (size_t)p * KVH * HD);
        mynah_slm_kv_put_v(&cache, 0, (uint32_t)p, vv + (size_t)p * KVH * HD);
    }
    mynah_slm_isa_narrow("scalar");
    mynah_slm_threads_init(1);
    mynah_slm_attention_kv_mt(oref, q, &cache, 0, NKV, H, KVH, HD, 0.088f, scr);

    for (size_t i = 0; i < N_LEVELS; i++) {
        const int id = mynah_slm_isa_parse(all_levels[i]);
        if (id < 0 || mynah_slm_isa_narrow(all_levels[i]) != id) continue;
        char what[160], d[96];

        mynah_slm_threads_init(1);
        mynah_slm_sgemm_own(1, M, N, K, 1.0f, a, K, b, K, 0.0f, c1, N);
        mynah_slm_attention_kv_mt(o1, q, &cache, 0, NKV, H, KVH, HD, 0.088f, scr);
        mynah_slm_threads_init(4);
        mynah_slm_sgemm_own(1, M, N, K, 1.0f, a, K, b, K, 0.0f, c4, N);
        mynah_slm_attention_kv_mt(o4, q, &cache, 0, NKV, H, KVH, HD, 0.088f, scr);

        double r = rel_diff(c1, cref, M * N);
        snprintf(what, sizeof what, "%s: sgemm (%s) agrees with the reference",
                 all_levels[i], mynah_slm_kern_sgemm()->name);
        snprintf(d, sizeof d, "rel %.1e", r);
        check(what, r < 1e-5, d);
        snprintf(what, sizeof what, "%s: sgemm 4 threads == 1 thread, bit for bit", all_levels[i]);
        check(what, memcmp(c1, c4, M * N * 4) == 0, NULL);

        r = rel_diff(o1, oref, H * HD);
        snprintf(what, sizeof what, "%s: bf16-KV attention (%s) agrees with scalar",
                 all_levels[i], mynah_slm_kern_attn()->name);
        snprintf(d, sizeof d, "rel %.1e", r);
        check(what, r < 1e-5, d);
        snprintf(what, sizeof what, "%s: attention 4 threads == 1 thread, bit for bit", all_levels[i]);
        check(what, memcmp(o1, o4, H * HD * 4) == 0, NULL);
    }
    mynah_slm_isa_narrow(NULL);
    mynah_slm_kv_free(&cache);
    free(a); free(b); free(c1); free(c4); free(cref);
    free(q); free(kk); free(vv); free(o1); free(o4); free(oref); free(scr);
}

static void test_report(void) {
    FILE *f = tmpfile();
    if (!f) { check("tmpfile", 0, NULL); return; }
    const int rc = mynah_slm_isa_report(f);
    rewind(f);
    char buf[4096];
    const size_t n = fread(buf, 1, sizeof buf - 1, f);
    buf[n] = '\0';
    fclose(f);
    check("the report names every family",
          strstr(buf, "qmat") && strstr(buf, "attn") && strstr(buf, "sgemm") &&
          strstr(buf, "ingot"), NULL);
    check("the report is OK when nothing was asked for",
          rc == 0 || getenv("MYNAH_SLM_ISA") || getenv("MYNAH_SLM_INT8"), buf);
}

int main(void) {
    printf("-- probe --\n");      test_probe();
    printf("\n-- narrowing --\n"); test_narrowing();
    printf("\n-- verify --\n");   test_verify_teeth();
    printf("\n-- every level --\n"); test_levels_agree();
    printf("\n-- report --\n");   test_report();
    mynah_slm_threads_shutdown();
    printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "PASS", failures,
           failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
