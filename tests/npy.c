/* npy.c — write a 2-D f32 array in NumPy .npy v1.0 format.
 *
 * Test-side only: the engine never writes .npy. This exists so the parity
 * harness can hand the C engine's activations to tools/eval/compare.py, which
 * is the thing that actually judges them.
 *
 * The format is a 6-byte magic, a version, a 2-byte little-endian header
 * length, then an ASCII dict padded with spaces to a 64-byte boundary and
 * terminated by a newline, then the raw data.
 *
 * SPDX-License-Identifier: MIT */
#include "npy.h"

#include <stdio.h>
#include <string.h>

int npy_write_f32(const char *path, const float *data, size_t rows, size_t cols) {
    char header[128];
    const int n = snprintf(header, sizeof header,
                           "{'descr': '<f4', 'fortran_order': False, 'shape': (%zu, %zu), }",
                           rows, cols);
    if (n < 0 || (size_t)n >= sizeof header) return -1;

    /* magic(6) + version(2) + len(2) + header + '\n' must be a multiple of 64 */
    size_t unpadded = 10 + (size_t)n + 1;
    size_t padded   = (unpadded + 63) / 64 * 64;
    size_t hlen     = padded - 10;

    FILE *f = fopen(path, "wb");
    if (!f) return -1;

    int rc = 0;
    const unsigned char preamble[8] = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0};
    const unsigned char len16[2] = {(unsigned char)(hlen & 0xFF),
                                    (unsigned char)((hlen >> 8) & 0xFF)};
    if (fwrite(preamble, 1, 8, f) != 8) rc = -1;
    if (fwrite(len16, 1, 2, f) != 2) rc = -1;
    if (fwrite(header, 1, (size_t)n, f) != (size_t)n) rc = -1;
    for (size_t i = unpadded; i < padded; i++)
        if (fputc(' ', f) == EOF) rc = -1;
    if (fputc('\n', f) == EOF) rc = -1;

    const size_t nelem = rows * cols;
    if (fwrite(data, sizeof(float), nelem, f) != nelem) rc = -1;

    if (fclose(f) != 0) rc = -1;
    return rc;
}
