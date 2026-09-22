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
#elif defined(__linux__)
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
    {   /* the model name, so a report can never be attributed to the wrong box */
        FILE *f = fopen("/proc/cpuinfo", "r");
        char  line[256];
        if (f) {
            while (fgets(line, sizeof line, f))
                if (strncmp(line, "CPU part", 8) == 0) {
                    char *c = strchr(line, ':');
                    if (c) snprintf(g_cpu, sizeof g_cpu, "aarch64 CPU part%s", c + 1);
                    break;
                }
            fclose(f);
        }
        for (char *c = g_cpu; *c; c++) if (*c == '\n') *c = 0;
    }
#endif
    return &g_caps;
}

const char *ternary_arm_name(tgemv_arm a)
{
    switch (a) {
    case ARM_REF:     return "ref";
    case ARM_DOTPROD: return "sdot";
    case ARM_I8MM:    return "smmla";
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
    default: return 0;
    }
}
