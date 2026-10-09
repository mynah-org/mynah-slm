/* test_qmat_init.c — the FIRST use of the qmat dispatch happens inside the
 * thread pool, from every worker at once.
 *
 * That is what a server does on its first decode step: mynah_slm_matvec and
 * its environment switches (MYNAH_SLM_KERNELS, MYNAH_SLM_INT8,
 * MYNAH_SLM_INT8_TYPES) and the ISA tables are resolved lazily, and nothing
 * guarantees the caller touched them first. They used to be plain ints
 * written on first read — a data race ThreadSanitizer reported (review R2).
 * This test is only meaningful as the first thing a process does, which is
 * why it is its own binary; `make tsan` runs it instrumented.
 *
 * SPDX-License-Identifier: MIT */
#include "qfixture.h"
#include "qmat.h"
#include "threads.h"

#include "ingot/dtype.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { ROWS = 256, COLS = 1024, PER = 16 };

static unsigned char W[ROWS * (COLS / 256) * 144];
static float X[COLS], Y[ROWS], Z[ROWS];
static mynah_slm_matvec_in P;
static int g_rc;

/* Each worker asks the same questions a projection does — is this type ours,
 * is int8 on, prepare, multiply — with nothing resolved beforehand. */
static void chunk(void *ctx, int i) {
    (void)ctx;
    static mynah_slm_matvec_in local[ROWS / PER];
    if (!mynah_slm_matvec_have(INGOT_TYPE_Q4_K)) { g_rc = -1; return; }
    (void)mynah_slm_matvec_int8_enabled();
    mynah_slm_matvec_prepare(X, COLS, &local[i]);
    if (mynah_slm_matvec(INGOT_TYPE_Q4_K, W + (size_t)i * PER * (COLS / 256) * 144, PER, COLS,
                         X, &P, Y + (size_t)i * PER) != 0)
        g_rc = -1;
}

int main(void) {
    qfx_fill(INGOT_TYPE_Q4_K, W, ROWS, COLS, 7);
    qfx_activations(X, COLS, 8);
    setenv("MYNAH_SLM_INT8", "1", 1);
    setenv("MYNAH_SLM_INT8_TYPES", "q4_k,q6_k", 1);
    mynah_slm_threads_init(4);
    /* P is prepared by hand-rolled code that touches no switch, so the
     * workers' calls are the first reads of every lazily resolved value. */
    memset(&P, 0, sizeof P);
    for (size_t s = 0; s < COLS / 32; s++) {
        float acc = 0.0f;
        for (int k = 0; k < 32; k++) acc += X[s * 32 + k];
        P.xsum[s] = acc;
    }
    P.have_int8 = 0;                      /* the f32 path: no int8 fields needed */
    P.cols = COLS;
    mynah_slm_parallel_for(ROWS / PER, chunk, NULL);

    /* Same answer serially, now that everything is resolved. */
    int ok = g_rc == 0 &&
             mynah_slm_matvec(INGOT_TYPE_Q4_K, W, ROWS, COLS, X, &P, Z) == 0 &&
             memcmp(Y, Z, sizeof Y) == 0;
    printf("%s  first qmat use from %d pool workers at once == the serial result\n",
           ok ? "ok  " : "FAIL", mynah_slm_threads_count());
    mynah_slm_threads_shutdown();
    printf("\n%s (%d failure%s)\n", ok ? "PASS" : "FAILED", !ok, ok ? "s" : "");
    return ok ? 0 : 1;
}
