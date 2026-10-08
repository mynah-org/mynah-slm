/* isa.c — see isa.h.
 *
 * Ported in shape from mynah-tts src/qmat.c (qmat_x86_probe, the "clamped
 * down, never up" level resolution, the verify-on-first-use gate) and ingot's
 * src/cpu.c probe. Two lessons from there are load-bearing here:
 *
 *   - a CPUID bit is not a usable unit until XGETBV says the OS saves its
 *     registers (YMM for AVX2, ZMM+opmask for AVX-512);
 *   - EVEX and VEX VNNI are independent features, not a ladder: Ice Lake
 *     server has AVX512-VNNI and no AVX-VNNI. mynah-tts once granted a VEX
 *     kernel on such a host from an enum comparison and died with SIGILL.
 *     We have no VEX-VNNI kernel; AVX-VNNI is probed and reported only.
 *
 * SPDX-License-Identifier: MIT */
#include "isa.h"

#include "kern.h"
#include "qmat.h"
#include "sgemm.h"

#include "ingot/quant.h"

#include <pthread.h>
#include <stdlib.h>
#include <string.h>

#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
#include <cpuid.h>
#define ISA_X86 1
#endif
#if defined(__aarch64__)
#define ISA_ARM64 1
#if defined(__APPLE__)
#include <sys/sysctl.h>
#elif defined(__linux__)
#include <sys/auxv.h>
#endif
#endif

/* ── the probe ─────────────────────────────────────────────────────────────*/

static mynah_slm_cpu g_cpu;

#if defined(ISA_X86)
static unsigned long long xgetbv0(void) {
    unsigned int eax, edx;
    __asm__ __volatile__(".byte 0x0f, 0x01, 0xd0" : "=a"(eax), "=d"(edx) : "c"(0));
    return ((unsigned long long)edx << 32) | eax;
}
#endif

#if defined(ISA_ARM64) && defined(__APPLE__)
static int sysctl_flag(const char *name) {
    int value = 0;
    size_t size = sizeof value;
    return sysctlbyname(name, &value, &size, NULL, 0) == 0 && value != 0;
}
#endif

static void probe(void) {
    memset(&g_cpu, 0, sizeof g_cpu);
#if defined(ISA_X86)
    g_cpu.x86 = 1;
    unsigned int a = 0, b = 0, c = 0, d = 0;
    int os_ymm = 0, os_zmm = 0, os_amx = 0;
    if (__get_cpuid(1, &a, &b, &c, &d)) {
        g_cpu.fma  = (c >> 12) & 1u;
        g_cpu.f16c = (c >> 29) & 1u;
        if ((c >> 27) & 1u) {                                     /* OSXSAVE */
            const unsigned long long xcr0 = xgetbv0();
            os_ymm = (xcr0 & 0x6ull) == 0x6ull;                   /* XMM|YMM */
            os_zmm = os_ymm && (xcr0 & 0xe0ull) == 0xe0ull;       /* +opmask, ZMM */
            os_amx = (xcr0 & 0x60000ull) == 0x60000ull;           /* XTILECFG|XTILEDATA */
        }
    }
    g_cpu.fma  = g_cpu.fma && os_ymm;
    g_cpu.f16c = g_cpu.f16c && os_ymm;
    if (__get_cpuid_count(7, 0, &a, &b, &c, &d)) {
        g_cpu.avx2       = os_ymm && ((b >> 5) & 1u);
        g_cpu.avx512f    = os_zmm && ((b >> 16) & 1u);
        g_cpu.avx512dq   = os_zmm && ((b >> 17) & 1u);
        g_cpu.avx512bw   = os_zmm && ((b >> 30) & 1u);
        g_cpu.avx512vl   = os_zmm && ((b >> 31) & 1u);
        g_cpu.avx512vnni = os_zmm && ((c >> 11) & 1u);
        g_cpu.amx        = os_amx && ((d >> 24) & 1u);           /* AMX-TILE */
        /* leaf 7 SUBLEAF 1, EAX: a different subleaf, the usual trap */
        if (__get_cpuid_count(7, 1, &a, &b, &c, &d)) {
            g_cpu.avxvnni    = os_ymm && ((a >> 4) & 1u);
            g_cpu.avx512bf16 = os_zmm && ((a >> 5) & 1u);
        }
    }
#elif defined(ISA_ARM64)
    g_cpu.arm64 = 1;
    g_cpu.neon  = 1;                         /* baseline on aarch64 */
#if defined(__APPLE__)
    g_cpu.dotprod = sysctl_flag("hw.optional.arm.FEAT_DotProd");
    g_cpu.i8mm    = sysctl_flag("hw.optional.arm.FEAT_I8MM");
    g_cpu.bf16    = sysctl_flag("hw.optional.arm.FEAT_BF16");
#elif defined(__linux__)
    const unsigned long hw = getauxval(AT_HWCAP), hw2 = getauxval(AT_HWCAP2);
#if defined(HWCAP_ASIMDDP)
    g_cpu.dotprod = (hw & HWCAP_ASIMDDP) != 0;
#endif
#if defined(HWCAP_SVE)
    g_cpu.sve = (hw & HWCAP_SVE) != 0;
#endif
#if defined(HWCAP2_I8MM)
    g_cpu.i8mm = (hw2 & HWCAP2_I8MM) != 0;
#endif
#if defined(HWCAP2_BF16)
    g_cpu.bf16 = (hw2 & HWCAP2_BF16) != 0;
#endif
    (void)hw; (void)hw2;
#endif
#endif
}

/* ── levels ────────────────────────────────────────────────────────────────*/

static int detect_level(void) {
#if defined(ISA_X86)
    const mynah_slm_cpu *c = &g_cpu;
    if (!(c->avx2 && c->fma && c->f16c)) return MYNAH_SLM_KERN_ID_SCALAR;
    if (!(c->avx512f && c->avx512bw && c->avx512vl && c->avx512dq)) return MYNAH_SLM_KERN_ID_AVX2;
    if (!c->avx512vnni) return MYNAH_SLM_KERN_ID_AVX512;
    return MYNAH_SLM_KERN_ID_AVX512VNNI;
#elif defined(ISA_ARM64)
    return g_cpu.dotprod ? MYNAH_SLM_KERN_ID_NEON_DOTPROD : MYNAH_SLM_KERN_ID_NEON;
#else
    return MYNAH_SLM_KERN_ID_SCALAR;
#endif
}

/* Position on this architecture's ladder; -1 = not a level here. */
static int rank(int id) {
    switch (id) {
    case MYNAH_SLM_KERN_ID_SCALAR:       return 0;
#if defined(ISA_X86)
    case MYNAH_SLM_KERN_ID_AVX2:         return 1;
    case MYNAH_SLM_KERN_ID_AVX512:       return 2;
    case MYNAH_SLM_KERN_ID_AVX512VNNI:   return 3;
#elif defined(ISA_ARM64)
    case MYNAH_SLM_KERN_ID_NEON:         return 1;
    case MYNAH_SLM_KERN_ID_NEON_DOTPROD: return 2;
#endif
    default:                             return -1;
    }
}

const char *mynah_slm_isa_level_name(int id) {
    switch (id) {
    case MYNAH_SLM_KERN_ID_SCALAR:       return "scalar";
    case MYNAH_SLM_KERN_ID_NEON:         return "neon";
    case MYNAH_SLM_KERN_ID_NEON_DOTPROD: return "dotprod";
    case MYNAH_SLM_KERN_ID_AVX2:         return "avx2";
    case MYNAH_SLM_KERN_ID_AVX512:       return "avx512";
    case MYNAH_SLM_KERN_ID_AVX512VNNI:   return "avx512vnni";
    default:                             return "?";
    }
}

static int g_detected = MYNAH_SLM_KERN_ID_SCALAR;

int mynah_slm_isa_parse(const char *name) {
    if (!name) return -1;
    if (!strcmp(name, "native") || !strcmp(name, "auto")) return g_detected;
    static const struct { const char *n; int id; } names[] = {
        { "scalar", MYNAH_SLM_KERN_ID_SCALAR },
        { "neon", MYNAH_SLM_KERN_ID_NEON },
        { "dotprod", MYNAH_SLM_KERN_ID_NEON_DOTPROD },
        { "neon_dotprod", MYNAH_SLM_KERN_ID_NEON_DOTPROD },
        { "neon-dotprod", MYNAH_SLM_KERN_ID_NEON_DOTPROD },
        { "avx2", MYNAH_SLM_KERN_ID_AVX2 },
        { "avx512", MYNAH_SLM_KERN_ID_AVX512 },
        { "avx512vnni", MYNAH_SLM_KERN_ID_AVX512VNNI },
        { "avx512-vnni", MYNAH_SLM_KERN_ID_AVX512VNNI },
    };
    for (size_t i = 0; i < sizeof names / sizeof *names; i++)
        if (!strcmp(name, names[i].n)) return rank(names[i].id) >= 0 ? names[i].id : -1;
    return -1;
}

/* ── per-family resolution ─────────────────────────────────────────────────*/

typedef struct {
    const char *family;
    int  chosen_id;
    char note[192];        /* fallbacks and failed verifies, for the report */
    int  verify_failed;
} fam_state;

static fam_state g_fq = { "qmat", 0, "", 0 };
static fam_state g_fa = { "attn", 0, "", 0 };
static fam_state g_fs = { "sgemm", 0, "", 0 };

static const mynah_slm_qmat_kern  *g_qmat  = &mynah_slm_qmat_kern_scalar;
static const mynah_slm_attn_kern  *g_attn  = &mynah_slm_attn_kern_scalar;
static const mynah_slm_sgemm_kern *g_sgemm = &mynah_slm_sgemm_kern_scalar;

/* The compiled tables, best first. Exactly the set the Makefile builds for
 * this architecture (KERN_*_x86 / _arm64 / _generic). */
static const mynah_slm_qmat_kern *const qmat_tables[] = {
#if defined(ISA_X86)
    &mynah_slm_qmat_kern_avx512vnni, &mynah_slm_qmat_kern_avx2,
#elif defined(ISA_ARM64)
    &mynah_slm_qmat_kern_neon_dotprod, &mynah_slm_qmat_kern_neon,
#endif
    &mynah_slm_qmat_kern_scalar,
};
static const mynah_slm_attn_kern *const attn_tables[] = {
#if defined(ISA_X86)
    &mynah_slm_attn_kern_avx2,
#elif defined(ISA_ARM64)
    &mynah_slm_attn_kern_neon,
#endif
    &mynah_slm_attn_kern_scalar,
};
static const mynah_slm_sgemm_kern *const sgemm_tables[] = {
#if defined(ISA_X86)
    &mynah_slm_sgemm_kern_avx512, &mynah_slm_sgemm_kern_avx2,
#elif defined(ISA_ARM64)
    &mynah_slm_sgemm_kern_neon,
#endif
    &mynah_slm_sgemm_kern_scalar,
};

#define N_OF(a) (sizeof (a) / sizeof *(a))

static int g_ceiling = MYNAH_SLM_KERN_ID_SCALAR;
static int g_env_bad = 0, g_env_clamped = 0;
static char g_env[48] = "";

/* Pick the best table at or under the ceiling that agrees with the scalar
 * one. A table that disagrees is not used, and the report says so: a
 * fallback nobody asked for must be visible. */
#define RESOLVE(state, tables, verify, out)                                       \
    do {                                                                          \
        (state).note[0] = '\0';                                                   \
        (state).verify_failed = 0;                                                \
        for (size_t i_ = 0; i_ < N_OF(tables); i_++) {                            \
            const int id_ = (tables)[i_]->id;                                     \
            if (rank(id_) > rank(g_ceiling)) continue;                            \
            char why_[128] = "";                                                  \
            if (id_ != MYNAH_SLM_KERN_ID_SCALAR &&                                \
                verify((tables)[i_], why_, sizeof why_) != 0) {                   \
                (state).verify_failed = 1;                                        \
                size_t l_ = strlen((state).note);                                 \
                snprintf((state).note + l_, sizeof (state).note - l_,            \
                         "%s VERIFY FAILED (%s); ", (tables)[i_]->name, why_);    \
                continue;                                                         \
            }                                                                     \
            (out) = (tables)[i_];                                                 \
            (state).chosen_id = id_;                                              \
            break;                                                                \
        }                                                                         \
    } while (0)

static void resolve_families(void) {
    RESOLVE(g_fq, qmat_tables, mynah_slm_isa_verify_qmat, g_qmat);
    RESOLVE(g_fa, attn_tables, mynah_slm_isa_verify_attn, g_attn);
    RESOLVE(g_fs, sgemm_tables, mynah_slm_isa_verify_sgemm, g_sgemm);
}

/* MYNAH_SLM_ISA narrows; it is clamped to the CPU and never raises the
 * ceiling. An unknown value is ignored and flagged rather than fatal: an
 * environment variable must not be able to stop the engine. */
static int ceiling_for(const char *req) {
    g_env_bad = g_env_clamped = 0;
    if (!req || !*req) return g_detected;
    const int id = mynah_slm_isa_parse(req);
    if (id < 0) { g_env_bad = 1; return g_detected; }
    if (rank(id) > rank(g_detected)) { g_env_clamped = 1; return g_detected; }
    return id;
}

static pthread_once_t g_once = PTHREAD_ONCE_INIT;

static void init_once(void) {
    probe();
    g_detected = detect_level();
    const char *e = getenv("MYNAH_SLM_ISA");
    if (e) snprintf(g_env, sizeof g_env, "%s", e);
    g_ceiling = ceiling_for(e);
    resolve_families();
}

void mynah_slm_isa_init(void) { pthread_once(&g_once, init_once); }

const mynah_slm_cpu *mynah_slm_cpu_caps(void) { mynah_slm_isa_init(); return &g_cpu; }
int mynah_slm_isa_detected(void) { mynah_slm_isa_init(); return g_detected; }
int mynah_slm_isa_ceiling(void)  { mynah_slm_isa_init(); return g_ceiling; }

const mynah_slm_qmat_kern  *mynah_slm_kern_qmat(void)  { mynah_slm_isa_init(); return g_qmat; }
const mynah_slm_attn_kern  *mynah_slm_kern_attn(void)  { mynah_slm_isa_init(); return g_attn; }
const mynah_slm_sgemm_kern *mynah_slm_kern_sgemm(void) { mynah_slm_isa_init(); return g_sgemm; }

int mynah_slm_isa_narrow(const char *name) {
    mynah_slm_isa_init();
    if (name && mynah_slm_isa_parse(name) < 0) return -1;
    if (name) {
        snprintf(g_env, sizeof g_env, "%s", name);
    } else {
        const char *e = getenv("MYNAH_SLM_ISA");
        snprintf(g_env, sizeof g_env, "%s", e ? e : "");
    }
    g_ceiling = ceiling_for(g_env);
    resolve_families();
    return g_ceiling;
}

/* ── the report ────────────────────────────────────────────────────────────*/

static void fam_line(FILE *f, const fam_state *s, const char *name,
                     const char *const *compiled, size_t n, const char *extra) {
    fprintf(f, "  %-6s -> %-11s compiled {", s->family, name);
    for (size_t i = 0; i < n; i++) fprintf(f, "%s%s", i ? "," : "", compiled[i]);
    fprintf(f, "}  verify %s%s%s%s\n",
            s->verify_failed ? "FAILED: " : (s->chosen_id ? "PASS" : "n/a (scalar is the reference)"),
            s->note, extra && *extra ? "  " : "", extra ? extra : "");
}

int mynah_slm_isa_report(FILE *f) {
    mynah_slm_isa_init();
    const mynah_slm_cpu *c = &g_cpu;
    int bad = 0;

    fprintf(f, "mynah-slm dispatch report\n");
#if defined(ISA_X86)
    fprintf(f, "  cpu    x86_64: avx2=%d fma=%d f16c=%d avx512f=%d bw=%d vl=%d dq=%d "
               "vnni=%d | avx512_bf16=%d avx_vnni=%d amx=%d\n",
            c->avx2, c->fma, c->f16c, c->avx512f, c->avx512bw, c->avx512vl,
            c->avx512dq, c->avx512vnni, c->avx512bf16, c->avxvnni, c->amx);
#elif defined(ISA_ARM64)
    fprintf(f, "  cpu    arm64: neon=%d dotprod=%d | i8mm=%d bf16=%d sve=%d\n",
            c->neon, c->dotprod, c->i8mm, c->bf16, c->sve);
#else
    fprintf(f, "  cpu    (no probe for this architecture: scalar kernels only)\n");
    (void)c;
#endif
    fprintf(f, "  build  baseline ISA of this binary: %s\n",
#if defined(__AVX512F__)
            "avx512"
#elif defined(__AVX2__)
            "avx2"
#elif defined(__ARM_FEATURE_DOTPROD)
            "neon+dotprod"
#elif defined(__ARM_NEON)
            "neon"
#elif defined(__SSE4_2__)
            "x86-64-v2 (sse4.2)"
#else
            "baseline"
#endif
    );
    fprintf(f, "  level  detected=%s  MYNAH_SLM_ISA=%s  ceiling=%s",
            mynah_slm_isa_level_name(g_detected), g_env[0] ? g_env : "(unset)",
            mynah_slm_isa_level_name(g_ceiling));
    if (g_env_bad)     { fprintf(f, "  <- UNKNOWN on this architecture, ignored"); bad = 1; }
    if (g_env_clamped) { fprintf(f, "  <- REQUEST ABOVE THE CPU, clamped down"); bad = 1; }
    fprintf(f, "\n");

    const char *qn[N_OF(qmat_tables)], *an[N_OF(attn_tables)], *sn[N_OF(sgemm_tables)];
    for (size_t i = 0; i < N_OF(qmat_tables); i++)  qn[i] = qmat_tables[i]->name;
    for (size_t i = 0; i < N_OF(attn_tables); i++)  an[i] = attn_tables[i]->name;
    for (size_t i = 0; i < N_OF(sgemm_tables); i++) sn[i] = sgemm_tables[i]->name;

    char q_extra[128];
    const int want8 = mynah_slm_matvec_int8_requested();
    snprintf(q_extra, sizeof q_extra, "int8 kernels: %s, switch %s%s",
             g_qmat->int8 ? g_qmat->int8_name : "none",
             want8 ? "ON" : "off",
             want8 && !g_qmat->int8 ? " <- REQUESTED BUT UNAVAILABLE at this level" : "");
    if (want8 && !g_qmat->int8) bad = 1;
    char s_extra[96];
    snprintf(s_extra, sizeof s_extra, "backend %s", mynah_slm_sgemm_backend());

    fam_line(f, &g_fq, g_qmat->name, qn, N_OF(qmat_tables), q_extra);
    fam_line(f, &g_fa, g_attn->name, an, N_OF(attn_tables), NULL);
    fam_line(f, &g_fs, g_sgemm->name, sn, N_OF(sgemm_tables), s_extra);
    bad |= g_fq.verify_failed | g_fa.verify_failed | g_fs.verify_failed;

    /* ingot dispatches its own kernels (the f32-activation Q6_K / Q8_0 / Q5_K
     * paths, every type we have no kernel for) on CPUID AND on what it was
     * compiled with. It is not ours to change; it is ours to report. */
    const ingot_cpu_caps ic = ingot_cpu();
#if defined(ISA_X86)
    fprintf(f, "  ingot  its own dispatch: avx2=%d avx512=%d avx512_vnni=%d f16c=%d%s\n",
            ic.avx2, ic.avx512, ic.avx512_vnni, ic.f16c,
            (c->avx2 && !ic.avx2) ? "  <- this CPU has AVX2 but ingot's build does not: "
                                    "its kernels run SCALAR (rebuild with ARCH_FLAGS >= x86-64-v3)" : "");
#elif defined(ISA_ARM64)
    fprintf(f, "  ingot  its own dispatch: neon=%d dotprod=%d i8mm=%d bf16=%d%s\n",
            ic.neon, ic.dotprod, ic.i8mm, ic.bf16,
            (c->dotprod && !ic.dotprod) ? "  <- this CPU has dotprod but ingot's build does not "
                                          "(its SDOT paths stay off)" : "");
#else
    (void)ic;
#endif

    /* Units this CPU has and no kernel of ours uses. Not a fault: a list of
     * what a future measurement could look at (K6 for bf16). */
    fprintf(f, "  idle   ");
    int any = 0;
#if defined(ISA_X86)
    if (c->avx512bf16) { fprintf(f, "avx512_bf16 "); any = 1; }
    if (c->avxvnni)    { fprintf(f, "avx_vnni(VEX) "); any = 1; }
    if (c->amx)        { fprintf(f, "amx "); any = 1; }
#elif defined(ISA_ARM64)
    if (c->i8mm)       { fprintf(f, "i8mm "); any = 1; }
    if (c->bf16)       { fprintf(f, "bf16 "); any = 1; }
    if (c->sve)        { fprintf(f, "sve "); any = 1; }
#endif
    fprintf(f, "%s\n", any ? "(present, no kernel of ours uses them)" : "none");
    fprintf(f, "  status %s\n", bad ? "NOT AS REQUESTED (see the flagged lines)" : "OK");
    return bad;
}
