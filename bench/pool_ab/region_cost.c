/* region_cost.c — what one parallel region costs, against a serial loop.
 *
 * The SAME source is linked against two pools (Makefile): the one before
 * 2026-10-08 (extracted from git, never edited) and the current src/threads.c.
 * Separate binaries, so run.sh alternates them process by process (ABAB) —
 * the honest substitute for an in-process A/B when the thing under test is
 * process-global.
 *
 * Output: one "<case> <microseconds per region>" line per case, then the
 * same work run serially, so a pool that is SLOWER than no pool at all shows
 * up as exactly that.
 *
 * SPDX-License-Identifier: MIT */
#include "threads.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

typedef struct { float *out; const float *in; int cols, rows_per; } work_job;

static void work_task(void *ctx, int i) {
    work_job *w = ctx;
    for (int r = 0; r < w->rows_per; r++) {
        float s = 0.0f;
        for (int c = 0; c < w->cols; c++) s += w->in[c] * (float)(r + 1);
        w->out[i * w->rows_per + r] = s;
    }
}

static double now_s(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

int main(int argc, char **argv) {
    const int thr = argc > 1 ? atoi(argv[1]) : 4;
    mynah_slm_threads_init(thr);
    const int nth = mynah_slm_threads_count();
    static float in[4096], out[1 << 16];
    for (int i = 0; i < 4096; i++) in[i] = (float)i * 1e-3f;

    /* Work per task in multiply-adds; nth*4 tasks per region, like a
     * decode projection. */
    const struct { const char *name; int cols, rows; } cs[] = {
        { "empty", 0, 1 }, { "tiny", 64, 4 }, { "small", 256, 16 }, { "medium", 1024, 32 },
    };
    const int R = 3000;
    for (int c = 0; c < 4; c++) {
        work_job w = { out, in, cs[c].cols, cs[c].rows };
        for (int r = 0; r < 200; r++) mynah_slm_parallel_for(nth * 4, work_task, &w);
        const double t0 = now_s();
        for (int r = 0; r < R; r++) mynah_slm_parallel_for(nth * 4, work_task, &w);
        printf("%s %.3f\n", cs[c].name, (now_s() - t0) / R * 1e6);
    }
    for (int c = 1; c < 4; c++) {
        work_job w = { out, in, cs[c].cols, cs[c].rows };
        const double t0 = now_s();
        for (int r = 0; r < R; r++)
            for (int i = 0; i < nth * 4; i++) work_task(&w, i);
        printf("serial_%s %.3f\n", cs[c].name, (now_s() - t0) / R * 1e6);
    }
    return 0;
}
