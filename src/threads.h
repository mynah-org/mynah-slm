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
 * would cost more than the work — and for the same reason idle workers spin
 * for a bounded time before parking: a condvar wake per region is tens of
 * microseconds, against regions that are often not much longer
 * (.work/sibling-port-map.md row 1, bench/pool_ab/).
 *
 * SPDX-License-Identifier: MIT */
#ifndef MYNAH_SLM_THREADS_H
#define MYNAH_SLM_THREADS_H

#include <stddef.h>

/* Cores this process may run on: the affinity mask on Linux (taskset, a
 * cgroup cpuset, a pinned prefork worker), the online count elsewhere. */
int mynah_slm_num_cpus(void);

/* n <= 0 means "use every core". Safe to call more than once; the second call
 * with the same count is free. Returns the effective count. */
int mynah_slm_threads_init(int n);
void mynah_slm_threads_shutdown(void);
int mynah_slm_threads_count(void);

/* Runs fn(ctx, i) for i in [0, n). The caller takes part rather than idling,
 * so a 4-way split uses 4 threads and not 5. With n <= 1 or a single thread it
 * runs inline with no synchronization at all.
 *
 * ONE REGION AT A TIME. Called from inside a task (nested), or from a second
 * thread while another thread's region is live, it runs fn inline on the
 * calling thread: correct, but serial. Performance-sensitive callers must not
 * rely on either; the scheduler thread should be the only one that computes. */
void mynah_slm_parallel_for(int n, void (*fn)(void *ctx, int i), void *ctx);

/* How long an idle worker spins before it parks, and how long the caller
 * spins waiting for the last task, in microseconds. MYNAH_SLM_POOL_SPIN_US
 * at init, 0 = park at once (the pre-2026-10 behaviour). -1 before init.
 * The setter exists for the interleaved A/B in tests/test_threads.c; call it
 * between regions, never from inside one. */
long mynah_slm_threads_spin_us(void);
void mynah_slm_threads_set_spin_us(long us);

#endif /* MYNAH_SLM_THREADS_H */
