/* formats.h — candidate PHYSICAL ternary representations, and their real cost.
 *
 * Isolated from the engine on purpose: nothing here is wired into
 * libmynah_slm, and nothing here may be until a measurement justifies it.
 *
 * The bit rates below are what the bytes actually cost, including scales and
 * padding. "1.58 bits" is the entropy of one trit and is never a storage rate;
 * see .work/ternary-feasibility.md Phase H.
 *
 * SPDX-License-Identifier: MIT */
#ifndef TERNARY_FORMATS_H
#define TERNARY_FORMATS_H

#include <stddef.h>
#include <stdint.h>

/* Contract-dimension group. 256 matches ggml's K-quant super-block and the
 * `tq2_0_fx4` block of fucina/docs/PTQTP.md, so scale overheads are comparable. */
#define TG 256

typedef enum {
    FMT_T0_INT8 = 0, /* oracle: one int8 per trit, no packing           8.125 bpw */
    FMT_T1_2BIT,     /* one plane, 4 trits/byte                         2.125 bpw */
    FMT_T2_BASE3,    /* one plane, 5 trits/byte (3^5=243<=256)          1.725 bpw */
    FMT_T3_FOLD9,    /* K=2 tied: c = 3t1+t2 in [-4,4], 2 codes/byte    4.125 bpw */
    FMT_T3_K3,       /* K=3 tied: c = 9t1+r, r in [-4,4]; 2-bit + nibble 6.125 bpw */
    FMT_MASKSIGN,    /* nonzero bitplane + sign bitplane                2.125 bpw */
    FMT__COUNT
} ternary_fmt;

typedef struct {
    const char *name;
    double      payload_bpw;  /* the codes themselves                  */
    double      scale_bpw;    /* f32 per group of TG, per plane-set    */
    double      total_bpw;    /* what a file would cost                */
    int         planes;       /* passes a GEMV must make over the row  */
} fmt_info;

extern const fmt_info TERNARY_FORMATS[FMT__COUNT];

/* Bytes one row of `cols` weights occupies in `f`. */
size_t ternary_row_bytes(ternary_fmt f, size_t cols);

/* Encode one row of trits (values in {-1,0,+1}, or for K>1 the already-fitted
 * composite in {-4..4} / {-13..13}) plus its per-group scales. */
void ternary_encode_row(ternary_fmt f, const int8_t *trits, const float *scales,
                        size_t cols, void *out);

#endif
