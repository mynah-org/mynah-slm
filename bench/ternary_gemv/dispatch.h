/* dispatch.h -- one interface, several ISA arms, chosen at run time.
 *
 * The Mac can only ever exercise two of these. Neoverse V2 (GCP Axion) has
 * FEAT_DotProd AND FEAT_I8MM, and the i8mm arm exists so that the prediction
 * written into .work/ can be tested there rather than argued about here.
 *
 *   tgemv_ref      portable C, the oracle. Always built, always correct.
 *   tgemv_dotprod  ARM sdot          (M1: yes, Neoverse V2: yes)
 *   tgemv_i8mm     ARM smmla         (M1: NO,  Neoverse V2: yes)
 *   tgemv_avx512   x86 vpdpbusd, 512-bit  (Zen 4 / Ice Lake+: yes)
 *   tgemv_vnni256  x86 vpdpbusd, 256-bit  -- the WIDTH CONTROL, see below
 *
 * An arm that the running CPU does not implement is not selectable: the caps
 * are read from the OS, never from the compiler. A kernel compiled for an
 * extension the CPU lacks does not run slowly, it raises SIGILL.
 *
 * SPDX-License-Identifier: MIT */
#ifndef TERNARY_DISPATCH_H
#define TERNARY_DISPATCH_H

#include "formats.h"
#include <stddef.h>
#include <stdint.h>

typedef struct {
    int8_t  *q;
    float   *scale;
    int32_t *sum;
    size_t   cols, groups;
} act_t;

typedef enum { ARM_REF = 0, ARM_DOTPROD, ARM_I8MM, ARM_AVX512, ARM_VNNI256,
               ARM__COUNT } tgemv_arm;

typedef struct {
    int have_dotprod;      /* FEAT_DotProd / HWCAP_ASIMDDP     */
    int have_i8mm;         /* FEAT_I8MM    / HWCAP2_I8MM       */
    int have_neon;
    int have_avx2;
    int have_avx512vnni;   /* AVX512F + AVX512BW + AVX512VNNI  */
    const char *cpu;
} ternary_caps;

/* Read from sysctl (macOS) or getauxval (Linux). Never from #ifdef. */
const ternary_caps *ternary_detect(void);
const char         *ternary_arm_name(tgemv_arm a);
int                 ternary_arm_available(tgemv_arm a);

void act_prepare(const float *x, size_t cols, act_t *a);

/* The best vector arm this CPU can actually run, or ARM_REF if none. */
tgemv_arm ternary_best_arm(void);

/* Deinterleave the quantized activations so the lanes line up with the packed
 * code order. The block width depends on the REGISTER WIDTH, not just the
 * format -- a 16-byte NEON load emits 32 nibbles, a 64-byte AVX-512 load emits
 * 128 -- so the caller must say which arm it is about to time. Getting this
 * wrong returns a plausible wrong number rather than crashing, which is why
 * every arm is gated against the T0 oracle before it is timed. */
void ternary_shuffle_acts(tgemv_arm arm, const int8_t *q, int8_t *xs2,
                          int8_t *xs4, size_t cols);

/* y[rows]. `xs2`/`xs4` are the lane-shuffled activations for the 2-bit and
 * nibble planes; the ref arm ignores them. Returns 0, or -1 when the arm
 * cannot run this format on this CPU -- in which case NOTHING is written to y,
 * so a caller that ignores the code times an empty loop and finds out. */
int tgemv(tgemv_arm arm, ternary_fmt f, const uint8_t *w, size_t rows, size_t cols,
          const act_t *a, const int8_t *xs2, const int8_t *xs4, float *y);

#endif
