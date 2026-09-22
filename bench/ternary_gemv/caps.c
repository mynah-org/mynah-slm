/* caps.c -- CPU feature detection, from the OS and not from the compiler.
 *
 * The distinction matters twice over. A binary built with -march=armv8.6-a+i8mm
 * running on an M1 does not fall back, it takes SIGILL. And ingot's own x86
 * history in this repo is the other half: Rosetta EXECUTES AVX2 while not
 * advertising it, so a whole test suite passed without running one AVX2 kernel.
 * Ask the OS, then say out loud which arm was taken.
 *
 * SPDX-License-Identifier: MIT */
#include "dispatch.h"

#include <stdio.h>
#include <string.h>

#if defined(__APPLE__)
#include <sys/sysctl.h>
static int sysctl_flag(const char *name)
{
    int    v = 0;
    size_t sz = sizeof v;
    return sysctlbyname(name, &v, &sz, NULL, 0) == 0 ? v : 0;
}
#elif defined(__linux__) && defined(__aarch64__)
#include <sys/auxv.h>
#ifndef HWCAP_ASIMDDP
#define HWCAP_ASIMDDP (1 << 20)
#endif
#ifndef HWCAP2_I8MM
#define HWCAP2_I8MM (1 << 13)
#endif
#ifndef HWCAP_ASIMD
#define HWCAP_ASIMD (1 << 1)
#endif
#endif

static ternary_caps g_caps;
static int          g_done;
static char         g_cpu[128];

#if defined(__linux__)
static void cpu_from_proc(const char *key)
{
    FILE *f = fopen("/proc/cpuinfo", "r");
    char  line[512];
    size_t klen = strlen(key);
    if (!f) return;
    while (fgets(line, sizeof line, f))
        if (strncmp(line, key, klen) == 0) {
            char *c = strchr(line, ':');
            if (c) {
                while (*++c == ' ') {}
                snprintf(g_cpu, sizeof g_cpu, "%s", c);
            }
            break;
        }
    fclose(f);
    for (char *c = g_cpu; *c; c++) if (*c == '\n') *c = 0;
}
#else
__attribute__((unused)) static void cpu_from_proc(const char *key) { (void)key; }
#endif

const ternary_caps *ternary_detect(void)
{
    if (g_done) return &g_caps;
    g_done = 1;
    strcpy(g_cpu, "unknown");
    g_caps.cpu = g_cpu;
#if defined(__APPLE__) && defined(__aarch64__)
    size_t sz = sizeof g_cpu;
    if (sysctlbyname("machdep.cpu.brand_string", g_cpu, &sz, NULL, 0) != 0)
        strcpy(g_cpu, "apple-arm64");
    g_caps.have_neon    = 1;
    g_caps.have_dotprod = sysctl_flag("hw.optional.arm.FEAT_DotProd");
    g_caps.have_i8mm    = sysctl_flag("hw.optional.arm.FEAT_I8MM");
#elif defined(__linux__) && defined(__aarch64__)
    unsigned long h1 = getauxval(AT_HWCAP), h2 = getauxval(AT_HWCAP2);
    g_caps.have_neon    = (h1 & HWCAP_ASIMD) != 0;
    g_caps.have_dotprod = (h1 & HWCAP_ASIMDDP) != 0;
    g_caps.have_i8mm    = (h2 & HWCAP2_I8MM) != 0;
    cpu_from_proc("CPU part");
#elif defined(__x86_64__) || defined(__i386__)
    /* __builtin_cpu_supports consults CPUID at run time. The Rosetta incident
     * in this repo is the reminder that a build-time #ifdef is not evidence:
     * it executed AVX2 while not advertising it, and a whole suite passed
     * without running one AVX2 kernel. Here the failure mode is the reverse
     * and worse -- an unsupported AVX-512 instruction is SIGILL. */
    __builtin_cpu_init();
    g_caps.have_avx2 = !!__builtin_cpu_supports("avx2");
    g_caps.have_avx512vnni = !!(__builtin_cpu_supports("avx512f") &&
                             __builtin_cpu_supports("avx512bw") &&
                             __builtin_cpu_supports("avx512vnni"));
    cpu_from_proc("model name");
#endif
    return &g_caps;
}

const char *ternary_arm_name(tgemv_arm a)
{
    switch (a) {
    case ARM_REF:     return "ref";
    case ARM_DOTPROD: return "sdot";
    case ARM_I8MM:    return "smmla";
    case ARM_AVX512:  return "avx512-vnni";
    case ARM_VNNI256: return "vnni256";
    default:          return "?";
    }
}

int ternary_arm_available(tgemv_arm a)
{
    const ternary_caps *c = ternary_detect();
    (void)c;   /* unused when neither ISA arm was compiled in */
    switch (a) {
    case ARM_REF: return 1;
#if defined(TERNARY_BUILD_DOTPROD)
    case ARM_DOTPROD: return c->have_dotprod;
#endif
#if defined(TERNARY_BUILD_I8MM)
    case ARM_I8MM: return c->have_i8mm;
#endif
#if defined(TERNARY_BUILD_AVX512)
    case ARM_AVX512:  return c->have_avx512vnni;
    case ARM_VNNI256: return c->have_avx512vnni;   /* EVEX 256-bit form */
#endif
    default: return 0;
    }
}

tgemv_arm ternary_best_arm(void)
{
    if (ternary_arm_available(ARM_AVX512))  return ARM_AVX512;
    if (ternary_arm_available(ARM_DOTPROD)) return ARM_DOTPROD;
    return ARM_REF;
}

/* Deinterleave activations so the dot-product lanes line up with the packed
 * code order. Done ONCE per GEMV and amortised over every row -- which is why a
 * kernel that owns its activation layout beats one that takes what it is
 * handed. It lives here, beside the arm selection, because the block width is a
 * property of the ARM and not of the caller. */
void shuffle_stride(const int8_t *in, int8_t *out, size_t cols, int blk, int stride);
void shuffle_stride(const int8_t *in, int8_t *out, size_t cols, int blk, int stride)
{
    for (size_t b = 0; b < cols; b += (size_t)blk)
        for (int s = 0; s < stride; s++)
            for (int i = 0; i < blk / stride; i++)
                out[b + (size_t)s * (blk / stride) + i] = in[b + (size_t)i * stride + s];
}

void ternary_shuffle_acts(tgemv_arm arm, const int8_t *q, int8_t *xs2,
                          int8_t *xs4, size_t cols)
{
    if (arm == ARM_AVX512) {          /* 64-byte loads */
        shuffle_stride(q, xs2, cols, 256, 4);
        shuffle_stride(q, xs4, cols, 128, 2);
    } else if (arm == ARM_VNNI256) {  /* 32-byte loads */
        shuffle_stride(q, xs2, cols, 128, 4);
        shuffle_stride(q, xs4, cols, 64, 2);
    } else {                          /* 16-byte loads */
        shuffle_stride(q, xs2, cols, 64, 4);
        shuffle_stride(q, xs4, cols, 32, 2);
    }
}

#if defined(TERNARY_BUILD_DOTPROD) && defined(__ARM_NEON)
void tgemv_neon(ternary_fmt f, const uint8_t *w, size_t rows, size_t cols,
                const act_t *a, const int8_t *xs2, const int8_t *xs4, float *y);
#endif
#if defined(TERNARY_BUILD_AVX512)
int tgemv_avx512(ternary_fmt f, const uint8_t *w, size_t rows, size_t cols,
                 const act_t *a, const int8_t *xs2, const int8_t *xs4, float *y);
int tgemv_vnni256(ternary_fmt f, const uint8_t *w, size_t rows, size_t cols,
                  const act_t *a, const int8_t *xs2, const int8_t *xs4, float *y);
#endif
void tgemv_scalar(ternary_fmt f, const uint8_t *w, size_t rows, size_t cols,
                  const act_t *a, float *y);

int tgemv(tgemv_arm arm, ternary_fmt f, const uint8_t *w, size_t rows, size_t cols,
          const act_t *a, const int8_t *xs2, const int8_t *xs4, float *y)
{
    if (!ternary_arm_available(arm)) return -1;
    switch (arm) {
    case ARM_REF:
        tgemv_scalar(f, w, rows, cols, a, y);
        return 0;
#if defined(TERNARY_BUILD_DOTPROD) && defined(__ARM_NEON)
    case ARM_DOTPROD:
        if (f == FMT_T2_BASE3 || f == FMT_MASKSIGN) return -1;
        tgemv_neon(f, w, rows, cols, a, xs2, xs4, y);
        return 0;
#endif
#if defined(TERNARY_BUILD_AVX512)
    case ARM_AVX512:
        return tgemv_avx512(f, w, rows, cols, a, xs2, xs4, y);
    case ARM_VNNI256:
        return tgemv_vnni256(f, w, rows, cols, a, xs2, xs4, y);
#endif
    default:
        return -1;
    }
}
