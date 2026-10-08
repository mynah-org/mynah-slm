/* test_threads.c — the pool's contract, under the conditions that broke the
 * previous one.
 *
 *   1. EVERY TASK RUNS EXACTLY ONCE, and its writes are visible to the caller
 *      when parallel_for returns. Hammered over tens of thousands of regions
 *      of random width, back to back — the pattern of a decode step — with
 *      the spin budget at 0 (every worker parks between regions, the wake
 *      path) and at its default (the spin path). The bug threads.c documents
 *      (a caller returning before stragglers had written) shows up here as a
 *      task count that is off by one, long before it shows up as an answer.
 *   2. A REGION FOLLOWED BY ANOTHER OF A DIFFERENT FUNCTION never runs a task
 *      of the first with the function of the second: the claim word carries
 *      the generation, and this checks it by giving every region its own tag.
 *   3. RESIZE AND SHUTDOWN are clean, repeatedly.
 *   4. RE-ENTRY: a parallel_for from inside a task, and two threads calling
 *      at once, both complete with every task run exactly once (inline).
 *
 * `tests/test_threads bench` measures the cost of one region — what a decode
 * step pays ~225 times — interleaved spin vs park in one process.
 *
 * No model, no network.
 *
 * SPDX-License-Identifier: MIT */
#include "threads.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int failures;

static void check(const char *what, int ok, const char *detail) {
    if (ok) printf("ok   %s\n", what);
    else  { printf("FAIL %s  <- %s\n", what, detail ? detail : ""); failures++; }
}

#define MAX_TASKS 256

typedef struct {
    int      tag;
    int      hits[MAX_TASKS];   /* plain ints: the pool must make them visible */
    atomic_int wrong_tag;
} job;

static void task_a(void *ctx, int i) {
    job *j = ctx;
    if (j->tag % 2 != 0) atomic_fetch_add(&j->wrong_tag, 1);
    j->hits[i]++;
}

static void task_b(void *ctx, int i) {
    job *j = ctx;
    if (j->tag % 2 != 1) atomic_fetch_add(&j->wrong_tag, 1);
    /* A little work, so tasks overlap and workers really race to claim. */
    volatile unsigned x = (unsigned)i;
    for (int k = 0; k < 50; k++) x = x * 1664525u + 1013904223u;
    j->hits[i] += 1 + (int)(x & 0u);
}

static uint32_t rng = 12345u;
static uint32_t next_rand(void) { rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng; }

static void hammer(int threads, long spin_us, int regions) {
    mynah_slm_threads_init(threads);
    mynah_slm_threads_set_spin_us(spin_us);

    static job j;
    long bad_counts = 0;
    int bad_tags = 0;
    for (int r = 0; r < regions; r++) {
        const int n = 1 + (int)(next_rand() % MAX_TASKS);
        memset(j.hits, 0, sizeof j.hits);
        j.tag = r;
        atomic_store(&j.wrong_tag, 0);
        mynah_slm_parallel_for(n, (r & 1) ? task_b : task_a, &j);
        for (int i = 0; i < MAX_TASKS; i++)
            if (j.hits[i] != (i < n ? 1 : 0)) bad_counts++;
        bad_tags += atomic_load(&j.wrong_tag);
    }

    char what[128], detail[96];
    snprintf(what, sizeof what, "%d threads, spin %ld us: %d regions, each task exactly once",
             mynah_slm_threads_count(), spin_us, regions);
    snprintf(detail, sizeof detail, "%ld task slots wrong", bad_counts);
    check(what, bad_counts == 0, detail);
    snprintf(what, sizeof what, "%d threads, spin %ld us: no task ran under another region's function",
             mynah_slm_threads_count(), spin_us);
    snprintf(detail, sizeof detail, "%d mismatches", bad_tags);
    check(what, bad_tags == 0, detail);
}

/* ── re-entry: nested and concurrent callers ─────────────────────────────── */

typedef struct { atomic_int inner_hits; int outer_hits[16]; } nest_job;

static void inner_task(void *ctx, int i) {
    (void)i;
    atomic_fetch_add(&((nest_job *)ctx)->inner_hits, 1);
}

static void outer_task(void *ctx, int i) {
    nest_job *j = ctx;
    mynah_slm_parallel_for(8, inner_task, j);   /* nested: must run inline */
    j->outer_hits[i]++;
}

static void nested_case(void) {
    mynah_slm_threads_init(4);
    static nest_job j;
    int bad = 0;
    for (int r = 0; r < 2000; r++) {
        memset(j.outer_hits, 0, sizeof j.outer_hits);
        atomic_store(&j.inner_hits, 0);
        mynah_slm_parallel_for(16, outer_task, &j);
        for (int i = 0; i < 16; i++) bad += j.outer_hits[i] != 1;
        bad += atomic_load(&j.inner_hits) != 16 * 8;
    }
    char d[64];
    snprintf(d, sizeof d, "%d bad regions", bad);
    check("nested parallel_for runs inline: every outer and inner task exactly once", bad == 0, d);
}

typedef struct { int hits[MAX_TASKS]; long bad; } caller_job;

static void count_task(void *ctx, int i) { ((caller_job *)ctx)->hits[i]++; }

static void *caller_thread(void *arg) {
    caller_job *c = arg;
    for (int r = 0; r < 20000; r++) {
        const int n = 2 + r % 64;
        memset(c->hits, 0, sizeof c->hits);
        mynah_slm_parallel_for(n, count_task, c);
        for (int i = 0; i < MAX_TASKS; i++) c->bad += c->hits[i] != (i < n);
    }
    return NULL;
}

static void concurrent_case(void) {
    mynah_slm_threads_init(4);
    static caller_job c[2];
    pthread_t t[2];
    for (int k = 0; k < 2; k++) { c[k].bad = 0; pthread_create(&t[k], NULL, caller_thread, &c[k]); }
    for (int k = 0; k < 2; k++) pthread_join(t[k], NULL);
    char d[64];
    snprintf(d, sizeof d, "%ld + %ld task slots wrong", c[0].bad, c[1].bad);
    check("two threads calling parallel_for at once: both correct (one runs inline)",
          c[0].bad == 0 && c[1].bad == 0, d);
}

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

typedef struct { float *out; const float *in; int cols, rows_per; } work_job;

/* A region shaped like one decode projection slice: nth*4 chunks of
 * rows_per x cols multiply-adds. `cols` = 0 makes it an empty region (pure
 * dispatch). Deliberately not vectorisable-friendly: it stands in for work,
 * it does not measure a kernel. */
static void work_task(void *ctx, int i) {
    work_job *w = ctx;
    for (int r = 0; r < w->rows_per; r++) {
        float s = 0.0f;
        for (int c = 0; c < w->cols; c++) s += w->in[c] * (float)(r + 1);
        w->out[i * w->rows_per + r] = s;
    }
}

static int bench(int threads) {
    mynah_slm_threads_init(threads);
    const int nth = mynah_slm_threads_count();
    const long spin_default = mynah_slm_threads_spin_us();
    printf("pool bench: threads=%d default spin=%ld us\n", nth, spin_default);
    printf("%-34s %12s %12s %8s\n", "region", "park us/reg", "spin us/reg", "ratio");

    static float in[4096], out[1 << 16];
    for (int i = 0; i < 4096; i++) in[i] = (float)i * 1e-3f;

    const struct { const char *name; int cols, rows_per; } cases[] = {
        /* Work per task in multiply-adds; time it serially on your machine
         * before reading the ratios (bench/pool_ab/run.sh prints the serial cost). */
        { "empty (pure dispatch)",        0,    1 },
        { "tiny   (256 MAC/task)",       64,    4 },
        { "small  (4 K MAC/task)",      256,   16 },
        { "medium (32 K MAC/task)",    1024,   32 },
    };
    for (size_t c = 0; c < sizeof cases / sizeof cases[0]; c++) {
        work_job w = { out, in, cases[c].cols, cases[c].rows_per };
        const int regions = 2000;
        double best[2] = { 1e30, 1e30 };
        /* Interleaved: park, spin, park, spin. */
        for (int round = 0; round < 6; round++)
            for (int mode = 0; mode < 2; mode++) {
                mynah_slm_threads_set_spin_us(mode ? (spin_default > 0 ? spin_default : 50) : 0);
                /* Let workers settle into the mode before timing. */
                for (int r = 0; r < 50; r++) mynah_slm_parallel_for(nth * 4, work_task, &w);
                const double t0 = now_s();
                for (int r = 0; r < regions; r++) mynah_slm_parallel_for(nth * 4, work_task, &w);
                const double t = (now_s() - t0) / regions * 1e6;
                if (t < best[mode]) best[mode] = t;
            }
        printf("%-34s %12.2f %12.2f %7.2fx\n", cases[c].name, best[0], best[1], best[0] / best[1]);
    }
    mynah_slm_threads_set_spin_us(spin_default);
    mynah_slm_threads_shutdown();
    return 0;
}

int main(int argc, char **argv) {
    if (argc > 1 && strcmp(argv[1], "bench") == 0)
        return bench(argc > 2 ? atoi(argv[2]) : 0);

    char d[64];
    snprintf(d, sizeof d, "%d", mynah_slm_num_cpus());
    check("num_cpus is positive (affinity mask on Linux)", mynah_slm_num_cpus() > 0, d);

    hammer(4, 0, 20000);
    hammer(4, 50, 20000);
    hammer(3, 5, 20000);     /* a budget near the region gap: both paths mixed */
    hammer(8, 50, 5000);     /* oversubscribed on a small runner */

    nested_case();
    concurrent_case();

    for (int k = 0; k < 20; k++) { mynah_slm_threads_init(1 + k % 5); mynah_slm_threads_shutdown(); }
    hammer(1, 50, 1000);     /* single thread: inline, no pool */
    check("repeated init/shutdown at varying widths", 1, NULL);

    mynah_slm_threads_shutdown();
    printf("%s (%d failure%s)\n", failures ? "FAILED" : "PASSED", failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
