/* npy.h — minimal .npy writer for the parity harness. Test-side only.
 * SPDX-License-Identifier: MIT */
#ifndef MYNAH_SLM_TEST_NPY_H
#define MYNAH_SLM_TEST_NPY_H

#include <stddef.h>

/* Row-major f32, shape (rows, cols). Returns 0 on success. */
int npy_write_f32(const char *path, const float *data, size_t rows, size_t cols);

#endif
