/* test_cuda.c — `make cuda-test`: the CUDA self-test as a program.
 *
 * Two halves. The host check always runs: it executes the kernels'
 * __host__ __device__ helpers on the CPU, so even a machine with no GPU
 * exercises the block layouts and the RoPE mapping. The device self-test then
 * runs every kernel against the CPU backend, or reports "no CUDA device".
 *
 * Exit 0 pass, 1 fail (either half), 77 host half passed and there is no
 * device (the repo's skip code). It never crashes for want of a GPU, which is
 * itself the first thing the compile-only CI job checks.
 *
 * SPDX-License-Identifier: MIT */
#include "cuda_self_test.h"

#include <stdio.h>

int main(void) {
    const int host = mynah_slm_cuda_host_check(stdout);
    const int rc = mynah_slm_cuda_self_test(stdout);
    if (rc == 77) printf("no CUDA device: device self-test skipped\n");
    if (host != 0) { printf("host check FAILED\n"); return 1; }
    return rc;
}
