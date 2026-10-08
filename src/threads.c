/* threads.c — see threads.h.
 *
 * Ported in design from mynah-tts src/threads.c, cut down to the one region
 * at a time this engine needs. Three changes against the pool it replaces,
 * each with the reason it exists:
 *
 *   1. CLAIMING IS ONE ATOMIC, not a mutex round-trip per task. A decode step
 *      dispatches ~225 regions (7 projections x 28 layers, attention, the
 *      head), each of nth*4 tasks: the old pool took and released the global
 *      mutex twice per task, with every worker contending for it.
 *   2. WORKERS SPIN BEFORE THEY PARK. A condvar wake costs 20-30 us
 *      (measured in mynah-tts: 22-29 us per cold helper wake). Inside a decode
 *      step the gap between two regions is an RMSNorm or a RoPE — a few
 *      microseconds — so a worker that parks between them pays the wake on
 *      every region. The spin budget is a TIME, not an iteration count,
 *      because `pause` costs ~100x more on some cores than on others
 *      (mynah-tts measured exactly that between x86 generations). Past the
 *      budget a worker parks, so an idle server burns nothing.
 *   3. THE WIDTH COMES FROM THE AFFINITY MASK, not sysconf. A process pinned
 *      to 8 cpus on a 64-core host (a container, a prefork worker, taskset)
 *      must not start 64 threads; mynah-tts found 32 threads in an 8-cpu
 *      worker exactly that way.
 *
 * What did NOT change, because it is this repo's own lesson: the caller waits
 * on TASKS FINISHED, never on workers present (see the comment in
 * mynah_slm_parallel_for), and every task writes a disjoint output slice, so
 * the result is bit-identical to the serial loop whatever the thread count.
 *
 * SPDX-License-Identifier: MIT */
#if defined(__linux__) && !defined(_GNU_SOURCE)
#define _GNU_SOURCE   /* sched_getaffinity, CPU_COUNT */
#endif
#include "threads.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#if defined(__linux__)
#include <sched.h>
#endif

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define cpu_relax() _mm_pause()
#elif defined(__aarch64__)
#define cpu_relax() __asm__ __volatile__("yield" ::: "memory")
#else
#define cpu_relax() ((void)0)
#endif

/* Default spin budget. mynah-tts settled on 35 us after a sweep on Neoverse
 * V2 (spin=0 cost +5.8% at 8 threads; a budget far above the region gap only
 * burns cores another stage could use). Ours starts from theirs and is
 * overridable — MYNAH_SLM_POOL_SPIN_US=0 restores park-immediately, which is
 * how the A/B against the old behaviour runs. */
#define DEFAULT_SPIN_US 50

/* ── the region ────────────────────────────────────────────────────────────
 * `claim` packs the generation, the task count AND the next index into ONE
 * word, so the bound a worker checks and the index it takes come from the
 * same atomic read. That is not a style choice, it is the bug the first
 * version of this pool had: with n in a separate field, a worker that woke
 * late could read the NEXT region's (larger) n while the claim word still
 * said "previous region, exhausted", pass the bound check, and win the
 * compare-exchange — running a task that did not exist and counting it into
 * the next region's g_done. tests/test_threads.c caught it as a hang
 * (done = 96 for a 95-task region).
 *
 *   bits 63..40  generation   (24 bits, wraps; only equality is ever tested)
 *   bits 39..20  task count n (20 bits)
 *   bits 19..0   next index   (20 bits)
 *
 * fn/ctx are read only AFTER a successful claim, at which point the region
 * cannot have completed (the claimed task is unfinished), so the caller
 * cannot have overwritten them for the next one. */
#define CLAIM_BITS 20
#define CLAIM_MASK ((1u << CLAIM_BITS) - 1u)
#define REGION_MAX ((int)CLAIM_MASK)   /* tasks per published region */

typedef struct {
    void (*fn)(void *ctx, int i);
    void  *ctx;
} region;

static region            g_r;
static _Atomic uint64_t  g_claim;      /* gen | n | next index */
static atomic_int        g_done;       /* tasks of the current region finished */
static atomic_int        g_stop;

static pthread_mutex_t   g_mu = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t    g_work = PTHREAD_COND_INITIALIZER;   /* workers park here */
static pthread_cond_t    g_fin  = PTHREAD_COND_INITIALIZER;   /* the caller parks here */
static int               g_parked;          /* workers in g_work, under g_mu */
static int               g_caller_parked;   /* under g_mu */

static pthread_t        *g_workers;
static int               g_count = 1;      /* 1 = inline, no pool */
static atomic_long       g_spin_ns = -1;   /* resolved at init; relaxed loads */

static uint32_t claim_gen(uint64_t c) { return (uint32_t)(c >> (2 * CLAIM_BITS)); }
static uint32_t claim_n(uint64_t c)   { return (uint32_t)(c >> CLAIM_BITS) & CLAIM_MASK; }
static uint32_t claim_idx(uint64_t c) { return (uint32_t)c & CLAIM_MASK; }
static uint64_t claim_make(uint32_t gen, uint32_t n) {
    return ((uint64_t)(gen & 0xffffffu) << (2 * CLAIM_BITS)) | ((uint64_t)n << CLAIM_BITS);
}

static uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}

static long resolve_spin_ns(void) {
    const char *e = getenv("MYNAH_SLM_POOL_SPIN_US");
    long us = DEFAULT_SPIN_US;
    if (e && *e) {
        char *end;
        const long v = strtol(e, &end, 10);
        if (end != e && v >= 0) us = v;
    }
    return us * 1000;
}

int mynah_slm_num_cpus(void) {
    /* Every core this process may run on, efficiency cores included.
     *
     * The tempting rule on an asymmetric Mac is "performance cores only",
     * since a static split would hand the slow cores an equal share and make
     * the whole region wait on them. Measured, that rule is wrong here:
     * 4 threads gave 12.9 tok/s and 8 gave 14.6 (+13%). The counter-based
     * claiming in parallel_for is what saves it — a slow core simply claims
     * fewer chunks — which is the reason chunks outnumber threads 4:1.
     *
     * On Linux the AFFINITY MASK decides, not the online count: under
     * taskset, a cgroup cpuset or a prefork worker pinned to a slice, sysconf
     * reports the whole machine and the pool would oversubscribe the slice. */
#if defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    if (sched_getaffinity(0, sizeof set, &set) == 0) {
        const int n = CPU_COUNT(&set);
        if (n > 0) return n;
    }
#endif
    const long online = sysconf(_SC_NPROCESSORS_ONLN);
    return online > 0 ? (int)online : 1;
}

/* Claim and run tasks of generation `gen` until it is exhausted. */
static void run_tasks(uint32_t gen) {
    uint64_t c = atomic_load_explicit(&g_claim, memory_order_acquire);
    for (;;) {
        if (claim_gen(c) != gen) return;
        const uint32_t i = claim_idx(c);
        const uint32_t n = claim_n(c);      /* same read as i: see above */
        if (i >= n) return;
        if (!atomic_compare_exchange_weak_explicit(&g_claim, &c, c + 1,
                                                   memory_order_acq_rel,
                                                   memory_order_acquire))
            continue;   /* c was reloaded */

        /* Claimed: the region is alive until this task is counted, so fn
         * and ctx are stable. */
        g_r.fn(g_r.ctx, (int)i);

        const int fin = atomic_fetch_add_explicit(&g_done, 1, memory_order_acq_rel) + 1;
        if (fin == (int)n) {
            /* Last task of the region. The caller may be parked: wake it
             * under the mutex, the same protocol as the worker side. */
            pthread_mutex_lock(&g_mu);
            if (g_caller_parked) pthread_cond_signal(&g_fin);
            pthread_mutex_unlock(&g_mu);
            return;
        }
        c = atomic_load_explicit(&g_claim, memory_order_acquire);
    }
}

/* Spin on the generation for the budget, then park. Correctness never
 * depends on the spin: the publisher bumps the generation BEFORE taking the
 * mutex, and a parking worker re-checks it UNDER the mutex, so either the
 * worker sees the new generation or it is already waiting when the broadcast
 * comes. This is the lost-wakeup trap mynah-tts hit on x86 ("trap 8") and
 * the reason the check is not an atomic load outside the lock. */
static uint32_t wait_for_work(uint32_t seen) {
    uint32_t g = claim_gen(atomic_load_explicit(&g_claim, memory_order_acquire));
    if (g != seen) return g;

    const long spin_ns = atomic_load_explicit(&g_spin_ns, memory_order_relaxed);
    if (spin_ns > 0) {
        const uint64_t deadline = now_ns() + (uint64_t)spin_ns;
        for (unsigned it = 0;; it++) {
            cpu_relax();
            g = claim_gen(atomic_load_explicit(&g_claim, memory_order_acquire));
            if (g != seen || atomic_load_explicit(&g_stop, memory_order_relaxed)) return g;
            if ((it & 63u) == 63u && now_ns() >= deadline) break;
        }
    }

    pthread_mutex_lock(&g_mu);
    g_parked++;
    for (;;) {
        g = claim_gen(atomic_load_explicit(&g_claim, memory_order_acquire));
        if (g != seen || atomic_load_explicit(&g_stop, memory_order_relaxed)) break;
        pthread_cond_wait(&g_work, &g_mu);
    }
    g_parked--;
    pthread_mutex_unlock(&g_mu);
    return g;
}

static void *worker(void *arg) {
    (void)arg;
    uint32_t seen = claim_gen(atomic_load_explicit(&g_claim, memory_order_acquire));
    for (;;) {
        const uint32_t g = wait_for_work(seen);
        if (atomic_load_explicit(&g_stop, memory_order_relaxed)) return NULL;
        seen = g;
        run_tasks(g);
    }
}

int mynah_slm_threads_init(int n) {
    if (n <= 0) n = mynah_slm_num_cpus();
    if (n == g_count && atomic_load(&g_spin_ns) >= 0) return g_count;

    mynah_slm_threads_shutdown();
    atomic_store(&g_spin_ns, resolve_spin_ns());
    if (n <= 1) { g_count = 1; return 1; }

    g_workers = calloc((size_t)(n - 1), sizeof *g_workers);
    if (!g_workers) { g_count = 1; return 1; }

    atomic_store(&g_stop, 0);
    int spawned = 0;
    for (int i = 0; i < n - 1; i++)
        if (pthread_create(&g_workers[spawned], NULL, worker, NULL) == 0) spawned++;

    g_count = spawned + 1;
    return g_count;
}

void mynah_slm_threads_shutdown(void) {
    if (g_count <= 1) { free(g_workers); g_workers = NULL; return; }

    pthread_mutex_lock(&g_mu);
    atomic_store(&g_stop, 1);
    pthread_cond_broadcast(&g_work);
    pthread_mutex_unlock(&g_mu);

    for (int i = 0; i < g_count - 1; i++) pthread_join(g_workers[i], NULL);
    free(g_workers);
    g_workers = NULL;
    g_count   = 1;
}

int mynah_slm_threads_count(void) { return g_count; }

long mynah_slm_threads_spin_us(void) {
    const long ns = atomic_load(&g_spin_ns);
    return ns < 0 ? -1 : ns / 1000;
}

void mynah_slm_threads_set_spin_us(long us) { atomic_store(&g_spin_ns, us < 0 ? 0 : us * 1000); }

/* A region wider than the claim word can count is split into several, each
 * running fn at an offset. No call site comes near 2^20 tasks; this exists so
 * the limit is a performance detail and never a correctness one. */
typedef struct { void (*fn)(void *, int); void *ctx; int base; } offset_job;
static void offset_task(void *c, int i) {
    const offset_job *o = c;
    o->fn(o->ctx, o->base + i);
}

void mynah_slm_parallel_for(int n, void (*fn)(void *ctx, int i), void *ctx) {
    if (n <= 0) return;
    if (n == 1 || g_count <= 1) {
        for (int i = 0; i < n; i++) fn(ctx, i);
        return;
    }
    if (n > REGION_MAX) {
        for (int base = 0; base < n; base += REGION_MAX) {
            offset_job o = { fn, ctx, base };
            mynah_slm_parallel_for(n - base < REGION_MAX ? n - base : REGION_MAX,
                                   offset_task, &o);
        }
        return;
    }

    /* Publish. The previous region is complete (we waited for it), so no
     * worker can be reading g_r: a late worker holds at most a stale claim
     * word, and its compare-exchange fails on the generation. */
    g_r.fn  = fn;
    g_r.ctx = ctx;
    atomic_store_explicit(&g_done, 0, memory_order_relaxed);
    const uint32_t gen = (claim_gen(atomic_load_explicit(&g_claim, memory_order_relaxed)) + 1) & 0xffffffu;
    atomic_store_explicit(&g_claim, claim_make(gen, (uint32_t)n), memory_order_release);

    pthread_mutex_lock(&g_mu);
    if (g_parked) pthread_cond_broadcast(&g_work);
    pthread_mutex_unlock(&g_mu);

    run_tasks(gen);            /* the caller works too rather than idling */

    /* Wait on TASKS FINISHED, never on workers being present.
     *
     * The first version counted participating workers instead, and it was
     * wrong in a way that only showed up as wrong answers. A worker that had
     * not yet woken would increment the counter AFTER the caller had already
     * decremented it to zero and returned; the next region then reset the
     * counter under that straggler, and its caller could see zero — or
     * negative — and return BEFORE its own workers had written their output
     * slices. The next layer read half-written buffers, so greedy decoding
     * produced a different answer run to run. Caught by
     * tests/test_server.sh, which asks the same question five times.
     *
     * The acquire on g_done is what makes every task's writes visible here. */
    if (atomic_load_explicit(&g_done, memory_order_acquire) == n) return;
    const long spin_ns = atomic_load_explicit(&g_spin_ns, memory_order_relaxed);
    if (spin_ns > 0) {
        const uint64_t deadline = now_ns() + (uint64_t)spin_ns;
        for (unsigned it = 0;; it++) {
            cpu_relax();
            if (atomic_load_explicit(&g_done, memory_order_acquire) == n) return;
            if ((it & 63u) == 63u && now_ns() >= deadline) break;
        }
    }
    pthread_mutex_lock(&g_mu);
    g_caller_parked = 1;
    while (atomic_load_explicit(&g_done, memory_order_acquire) != n)
        pthread_cond_wait(&g_fin, &g_mu);
    g_caller_parked = 0;
    pthread_mutex_unlock(&g_mu);
}
