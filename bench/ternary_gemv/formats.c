/* formats.c — encoders and the honest bit accounting.
 *
 * One f32 scale per group of TG weights, per plane-set. A K=2 or K=3 tied
 * composite needs only ONE scale, because tying fixes the ratio between the
 * planes (fucina/docs/PTQTP.md: alpha = (3s, s) collapses to a uniform grid).
 * That is the whole reason the tie is worth its reconstruction penalty.
 *
 * SPDX-License-Identifier: MIT */
#include "formats.h"

#include <string.h>

#define SCALE_BPW (32.0 / (double)TG)

const fmt_info TERNARY_FORMATS[FMT__COUNT] = {
    {"T0 int8",      8.0,       SCALE_BPW, 8.0 + SCALE_BPW,       1},
    {"T1 2-bit",     2.0,       SCALE_BPW, 2.0 + SCALE_BPW,       1},
    /* 256/5 = 51.2, so a group pads to 52 bytes: 1.625 payload, not 1.600.
     * Base-3 aligns with no power-of-two group size; this is that cost. */
    {"T2 base-3",    52.0 * 8 / TG, SCALE_BPW, 52.0 * 8 / TG + SCALE_BPW, 1},
    {"T3 fold9 K=2", 4.0,       SCALE_BPW, 4.0 + SCALE_BPW,       1},
    {"T3 K=3",       6.0,       SCALE_BPW, 6.0 + SCALE_BPW,       2},
    {"mask+sign",    2.0,       SCALE_BPW, 2.0 + SCALE_BPW,       1},
};

size_t ternary_row_bytes(ternary_fmt f, size_t cols)
{
    size_t groups = cols / TG;
    size_t payload;
    switch (f) {
    case FMT_T0_INT8:  payload = cols;                       break;
    case FMT_T1_2BIT:  payload = cols / 4;                   break;
    case FMT_T2_BASE3: payload = groups * ((TG + 4) / 5);    break;
    case FMT_T3_FOLD9: payload = cols / 2;                   break;
    /* K=3: a 2-bit high plane plus a nibble low plane. */
    case FMT_T3_K3:    payload = cols / 4 + cols / 2;        break;
    case FMT_MASKSIGN: payload = cols / 8 * 2;               break;
    default:           payload = cols;                       break;
    }
    return payload + groups * sizeof(float);
}

/* Layout, for every format: [payload bytes][groups * f32 scales].
 * Scales trail the codes so a kernel can stream the payload contiguously and
 * touch the scale array once per group. */
void ternary_encode_row(ternary_fmt f, const int8_t *t, const float *scales,
                        size_t cols, void *out)
{
    uint8_t *p = (uint8_t *)out;
    size_t   groups = cols / TG;

    switch (f) {
    case FMT_T0_INT8:
        for (size_t i = 0; i < cols; i++) p[i] = (uint8_t)(int8_t)t[i];
        p += cols;
        break;

    case FMT_T1_2BIT: /* 2 bits, biased by +1 so codes are 0,1,2 */
        for (size_t i = 0; i < cols; i += 4) {
            p[i / 4] = (uint8_t)(((t[i] + 1) & 3) | (((t[i + 1] + 1) & 3) << 2) |
                                 (((t[i + 2] + 1) & 3) << 4) |
                                 (((t[i + 3] + 1) & 3) << 6));
        }
        p += cols / 4;
        break;

    case FMT_T2_BASE3: /* 5 base-3 digits per byte, 3^5 = 243 <= 256, per GROUP */
        for (size_t g = 0; g < groups; g++) {
            uint8_t *gp = p + g * ((TG + 4) / 5);
            for (size_t i = 0; i < TG; i += 5) {
                unsigned v = 0, m = 1;
                for (int k = 0; k < 5 && i + k < TG; k++) {
                    v += (unsigned)(t[g * TG + i + k] + 1) * m;
                    m *= 3;
                }
                gp[i / 5] = (uint8_t)v;
            }
        }
        p += groups * ((TG + 4) / 5);
        break;

    case FMT_T3_FOLD9: /* composite in [-4,4], biased by +4 -> nibble 0..8 */
        for (size_t i = 0; i < cols; i += 2)
            p[i / 2] = (uint8_t)(((t[i] + 4) & 15) | (((t[i + 1] + 4) & 15) << 4));
        p += cols / 2;
        break;

    case FMT_T3_K3: {
        /* composite c in [-13,13] split as c = 9*hi + lo, lo in [-4,4].
         * hi goes in a 2-bit plane (biased +1), lo in a nibble (biased +4). */
        uint8_t *lo = p + cols / 4;
        for (size_t i = 0; i < cols; i += 4) {
            uint8_t b = 0;
            for (int k = 0; k < 4; k++) {
                int c = t[i + k];
                int h = (c + (c >= 0 ? 4 : -4)) / 9;      /* round-to-nearest ninth */
                b |= (uint8_t)(((h + 1) & 3) << (2 * k));
            }
            p[i / 4] = b;
        }
        for (size_t i = 0; i < cols; i += 2) {
            int c0 = t[i], c1 = t[i + 1];
            int h0 = (c0 + (c0 >= 0 ? 4 : -4)) / 9, h1 = (c1 + (c1 >= 0 ? 4 : -4)) / 9;
            lo[i / 2] = (uint8_t)((((c0 - 9 * h0) + 4) & 15) |
                                  ((((c1 - 9 * h1) + 4) & 15) << 4));
        }
        p += cols / 4 + cols / 2;
        break;
    }

    case FMT_MASKSIGN: { /* bitplane 1: nonzero; bitplane 2: sign (1 = negative) */
        uint8_t *sgn = p + cols / 8;
        memset(p, 0, cols / 8 * 2);
        for (size_t i = 0; i < cols; i++) {
            if (t[i] != 0) {
                p[i / 8] |= (uint8_t)(1u << (i & 7));
                if (t[i] < 0) sgn[i / 8] |= (uint8_t)(1u << (i & 7));
            }
        }
        p += cols / 8 * 2;
        break;
    }
    default: break;
    }
    memcpy(p, scales, groups * sizeof(float));
}
