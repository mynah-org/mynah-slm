/* threads.c — see threads.h.
 * SPDX-License-Identifier: MIT */
#include "threads.h"

#include <pthread.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef struct {
    void  (*fn)(void *ctx, int i);
    void   *ctx;
    int     n;
    int     next;        /* next task index to claim */
    int     active;      /* workers still inside the region */
    unsigned generation; /* bumped per region, so a worker cannot re-run one */
} region;

static pthread_t       *g_workers;
static int              g_count = 1;      /* 1 = inline, no pool */
static int              g_stop;
static pthread_mutex_t  g_mu  = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t   g_work = PTHREAD_COND_INITIALIZER;
static pthread_cond_t   g_done = PTHREAD_COND_INITIALIZER;
static region           g_r;
static unsigned         g_gen;

int mynah_slm_num_cpus(void) {
    /* Every online core, efficiency cores included.
     *
     * The tempting rule on an asymmetric Mac is "performance cores only",
     * since a static split would hand the slow cores an equal share and make
     * the whole region wait on them. Measured, that rule is wrong here:
     * 4 threads gave 12.9 tok/s and 8 gave 14.6 (+13%). The counter-based
     * claiming in parallel_for is what saves it — a slow core simply claims
     * fewer chunks — which is the reason chunks outnumber threads 4:1.
     *
     * Kept as a comment rather than a #if because the next machine may
     * disagree, and the way to find out is `-t N`, not a guess. */
    const long online = sysconf(_SC_NPROCESSORS_ONLN);
    return online > 0 ? (int)online : 1;
}

/* Claim task indices until the region is exhausted. Work-stealing by counter
 * rather than a fixed per-thread range: chunks are not equal in cost (the last
 * row block of a matvec can be short) and an idle core is worse than a lock. */
static void run_tasks(void) {
    for (;;) {
        pthread_mutex_lock(&g_mu);
        const int i = (g_r.next < g_r.n) ? g_r.next++ : -1;
        pthread_mutex_unlock(&g_mu);
        if (i < 0) return;
        g_r.fn(g_r.ctx, i);
    }
}

static void *worker(void *arg) {
    (void)arg;
    unsigned seen = 0;
    for (;;) {
        pthread_mutex_lock(&g_mu);
        while (!g_stop && g_gen == seen) pthread_cond_wait(&g_work, &g_mu);
        if (g_stop) { pthread_mutex_unlock(&g_mu); return NULL; }
        seen = g_gen;
        g_r.active++;
        pthread_mutex_unlock(&g_mu);

        run_tasks();

        pthread_mutex_lock(&g_mu);
        if (--g_r.active == 0) pthread_cond_signal(&g_done);
        pthread_mutex_unlock(&g_mu);
    }
}

int mynah_slm_threads_init(int n) {
    if (n <= 0) n = mynah_slm_num_cpus();
    if (n == g_count) return g_count;

    mynah_slm_threads_shutdown();
    if (n <= 1) { g_count = 1; return 1; }

    g_workers = calloc((size_t)(n - 1), sizeof *g_workers);
    if (!g_workers) { g_count = 1; return 1; }

    g_stop = 0;
    g_gen  = 0;
    int spawned = 0;
    for (int i = 0; i < n - 1; i++)
        if (pthread_create(&g_workers[i], NULL, worker, NULL) == 0) spawned++;

    g_count = spawned + 1;
    return g_count;
}

void mynah_slm_threads_shutdown(void) {
    if (g_count <= 1) { free(g_workers); g_workers = NULL; return; }

    pthread_mutex_lock(&g_mu);
    g_stop = 1;
    pthread_cond_broadcast(&g_work);
    pthread_mutex_unlock(&g_mu);

    for (int i = 0; i < g_count - 1; i++) pthread_join(g_workers[i], NULL);
    free(g_workers);
    g_workers = NULL;
    g_count   = 1;
}

int mynah_slm_threads_count(void) { return g_count; }

void mynah_slm_parallel_for(int n, void (*fn)(void *ctx, int i), void *ctx) {
    if (n <= 0) return;
    if (n == 1 || g_count <= 1) {
        for (int i = 0; i < n; i++) fn(ctx, i);
        return;
    }

    pthread_mutex_lock(&g_mu);
    g_r.fn     = fn;
    g_r.ctx    = ctx;
    g_r.n      = n;
    g_r.next   = 0;
    g_r.active = 1;            /* the caller counts as a participant */
    g_gen++;
    pthread_cond_broadcast(&g_work);
    pthread_mutex_unlock(&g_mu);

    run_tasks();               /* the caller works too rather than idling */

    pthread_mutex_lock(&g_mu);
    if (--g_r.active > 0)
        while (g_r.active > 0) pthread_cond_wait(&g_done, &g_mu);
    pthread_mutex_unlock(&g_mu);
}
