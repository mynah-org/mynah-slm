/* isa.h — which of our kernels this process runs, and the proof.
 *
 * A benchmark is invalid until dispatch is proven (.work/engineering-method.md).
 * This module is that proof for our own kernels: it probes the CPU (CPUID +
 * XGETBV on x86, getauxval on Linux arm64, sysctl on macOS), takes the
 * MYNAH_SLM_ISA override — which can only NARROW, never grant an instruction
 * the CPU lacks — picks one kernel table per family (kern.h), runs each
 * picked table once against the scalar table before trusting it
 * (verify-on-first-use), and prints all of it on `mynah-slm --dispatch`.
 *
 * Levels: x86 scalar < avx2 < avx512 < avx512vnni; arm64 scalar < neon <
 * dotprod. Resolution happens once per process (pthread_once) on first use,
 * or at model load.
 *
 * SPDX-License-Identifier: MIT */
#ifndef MYNAH_SLM_ISA_H
#define MYNAH_SLM_ISA_H

#include "kern.h"

#include <stdio.h>

typedef struct {
    int x86, arm64;
    /* x86, each one only if the OS also saves the register state */
    int avx2, fma, f16c, avx512f, avx512bw, avx512vl, avx512dq, avx512vnni;
    int avx512bf16, avxvnni, amx;
    /* arm64 */
    int neon, dotprod, i8mm, bf16, sve;
} mynah_slm_cpu;

/* What the CPU has (probed once). */
const mynah_slm_cpu *mynah_slm_cpu_caps(void);

/* Level ids are kern.h's MYNAH_SLM_KERN_ID_*. */
int         mynah_slm_isa_detected(void);   /* the highest level the CPU runs */
int         mynah_slm_isa_ceiling(void);    /* after MYNAH_SLM_ISA */
const char *mynah_slm_isa_level_name(int id);

/* "scalar", "neon", "dotprod", "avx2", "avx512", "avx512vnni" -> id, or -1
 * for an unknown name or one that does not exist on this architecture.
 * "native" and "auto" -> the detected level. */
int mynah_slm_isa_parse(const char *name);

/* Re-resolve every family under `name` as the ceiling (NULL = back to what
 * MYNAH_SLM_ISA / the CPU says). Clamped to the CPU, like the environment.
 * Returns the ceiling granted, or -1 for a name `mynah_slm_isa_parse` refuses.
 * For tests and benches; call it with the thread pool idle. */
int mynah_slm_isa_narrow(const char *name);

/* Force resolution now (model load does), so the verify cost is not paid
 * inside the first token. */
void mynah_slm_isa_init(void);

/* The dispatch report. Returns 0, or 1 when something is not what was asked
 * for: an unknown MYNAH_SLM_ISA, a request above the CPU (clamped), a
 * verify that failed and fell back, or int8 requested but unavailable. */
int mynah_slm_isa_report(FILE *f);

/* Verify-on-first-use (isa_verify.c): run `k` against the scalar table on
 * fixed, awkward fixtures. 0 = agrees; else -1 with the reason in `why`. */
int mynah_slm_isa_verify_qmat(const mynah_slm_qmat_kern *k, char *why, size_t n);
int mynah_slm_isa_verify_attn(const mynah_slm_attn_kern *k, char *why, size_t n);
int mynah_slm_isa_verify_sgemm(const mynah_slm_sgemm_kern *k, char *why, size_t n);

#endif /* MYNAH_SLM_ISA_H */
