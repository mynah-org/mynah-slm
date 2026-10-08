/* burst_cpu.c — wall AND cpu time of a decode-shaped burst.
 *
 * 225 back-to-back regions (one Qwen3-0.6B decode step's worth), then an
 * idle gap (sampling, a callback, a client), 100 times. A spinning pool can
 * win wall time by burning cores the ASR and TTS stages next to us need, so
 * the CPU column is the half of the answer a tok/s number hides.
 *
 * SPDX-License-Identifier: MIT */
#include "threads.h"

#include <stdio.h>
#include <stdlib.h>
#include <sys/resource.h>
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

static double cpu_s(void) {
    struct rusage u;
    getrusage(RUSAGE_SELF, &u);
    return (double)(u.ru_utime.tv_sec + u.ru_stime.tv_sec) +
           1e-6 * (double)(u.ru_utime.tv_usec + u.ru_stime.tv_usec);
}

int main(int argc, char **argv) {
    const int thr = argc > 1 ? atoi(argv[1]) : 4;
    const int gap_us = argc > 2 ? atoi(argv[2]) : 1000;
    mynah_slm_threads_init(thr);
    const int nth = mynah_slm_threads_count();
    static float in[4096], out[1 << 16];
    work_job w = { out, in, 256, 16 };
    const double w0 = now_s(), c0 = cpu_s();
    for (int step = 0; step < 100; step++) {
        for (int r = 0; r < 225; r++) mynah_slm_parallel_for(nth * 4, work_task, &w);
        struct timespec ts = { 0, gap_us * 1000L };
        nanosleep(&ts, NULL);
    }
    const double wall = now_s() - w0, cpu = cpu_s() - c0;
    printf("wall_ms_per_step %.3f\ncpu_ms_per_step %.3f\n", wall * 10, cpu * 10);
    return 0;
}
