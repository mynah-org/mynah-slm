/* cuda_self_test.h — the CUDA backend checked against the CPU backend.
 *
 * C, so the test driver and (later) the CLI can call it without nvcc. Only
 * linked into a `make cuda` / `make cuda-test` build.
 *
 * SPDX-License-Identifier: MIT */
#ifndef MYNAH_SLM_CUDA_SELF_TEST_H
#define MYNAH_SLM_CUDA_SELF_TEST_H

#include <stddef.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Usable devices; 0 with `err` explaining why when there are none (no
 * device, no driver). Never leaves a CUDA error pending. */
int mynah_slm_cuda_device_count(char *err, size_t errsz);

/* The __host__ __device__ halves of the kernels (block decode, f16 decode,
 * bf16 rounding, RoPE pair mapping) run ON THE HOST against ingot and the
 * CPU kernels. Needs no device. 0 pass, 1 fail. */
int mynah_slm_cuda_host_check(FILE *log);

/* Runs every CUDA kernel against the CPU backend on random data at Qwen3
 * shapes, printing one ok/FAIL line per check to `log` with the measured
 * error and the tolerance it was held to. Returns:
 *    0  every check passed
 *    1  at least one failed (or the backend failed to run an op)
 *   77  no CUDA device — reported, not a crash */
int mynah_slm_cuda_self_test(FILE *log);

#ifdef __cplusplus
}
#endif

#endif /* MYNAH_SLM_CUDA_SELF_TEST_H */
