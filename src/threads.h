/* threads.h — a persistent pool and a parallel-for over disjoint output ranges.
 *
 * The guarantee that matters: results are BIT-IDENTICAL to the serial loop by
 * construction. Every task runs the same code over a disjoint slice of the
 * output, so nothing is reduced across threads and no summation order changes.
 * That is what lets the parity gate keep a 1e-4 tolerance at layer 0 with
 * threading on — the alternative, a cross-thread reduction, would move the
 * numbers and we would have to loosen a gate to accommodate our own scheduler.
 *
 * The pool is persistent. A decode step dispatches ~200 parallel regions (7
 * projections x 28 layers, plus attention), so spawning threads per region
 * would cost more than the work.
 *
 * SPDX-License-Identifier: MIT */
#ifndef MYNAH_SLM_THREADS_H
#define MYNAH_SLM_THREADS_H

#include <stddef.h>

/* Online cores, minus nothing: this is a single-inference CLI. A server that
 * runs several at once must lower it, the way mynah-asr does. */
int mynah_slm_num_cpus(void);

/* n <= 0 means "use every core". Safe to call more than once; the second call
 * with the same count is free. Returns the effective count. */
int mynah_slm_threads_init(int n);
void mynah_slm_threads_shutdown(void);
int mynah_slm_threads_count(void);

/* Runs fn(ctx, i) for i in [0, n). The caller takes part rather than idling,
 * so a 4-way split uses 4 threads and not 5. With n <= 1 or a single thread it
 * runs inline with no synchronization at all. */
void mynah_slm_parallel_for(int n, void (*fn)(void *ctx, int i), void *ctx);

#endif /* MYNAH_SLM_THREADS_H */
