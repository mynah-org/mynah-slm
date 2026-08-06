/* timing.c — see timing.h.
 * SPDX-License-Identifier: MIT */
#include "timing.h"

#include <stdio.h>
#include <string.h>
#include <time.h>

double mynah_slm_now(void) {
    struct timespec ts;
    /* MONOTONIC, not REALTIME: an NTP step mid-generation must not be able to
     * produce a negative decode time or a 40-year one. */
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

void mynah_slm_timing_reset(mynah_slm_timing *t) { memset(t, 0, sizeof *t); }

void mynah_slm_timing_start(mynah_slm_timing *t) { t->t0_ = mynah_slm_now(); }

void mynah_slm_timing_end_load(mynah_slm_timing *t) {
    const double now = mynah_slm_now();
    t->load_s = now - t->t0_;
    t->t0_    = now;
}

void mynah_slm_timing_end_prefill(mynah_slm_timing *t, uint32_t n_prompt) {
    const double now = mynah_slm_now();
    t->prefill_s = now - t->t0_;
    t->n_prompt  = n_prompt;
    t->t0_       = now;
}

void mynah_slm_timing_token(mynah_slm_timing *t) {
    if (!t->ttft_done_) {
        /* TTFT spans prefill plus the first decode step: it is what a caller
         * waits through, not what the decoder alone costs. */
        t->ttft_s     = t->prefill_s + (mynah_slm_now() - t->t0_);
        t->ttft_done_ = 1;
    }
    t->n_gen++;
}

void mynah_slm_timing_end_decode(mynah_slm_timing *t) {
    t->decode_s = mynah_slm_now() - t->t0_;
}

double mynah_slm_prefill_tok_s(const mynah_slm_timing *t) {
    return t->prefill_s > 0.0 ? (double)t->n_prompt / t->prefill_s : 0.0;
}

double mynah_slm_decode_tok_s(const mynah_slm_timing *t) {
    return t->decode_s > 0.0 ? (double)t->n_gen / t->decode_s : 0.0;
}

int mynah_slm_timing_format(const mynah_slm_timing *t, char *buf, size_t n) {
    return snprintf(buf, n,
        "[load %.2fs | prompt %u tok, prefill %.1f tok/s | "
        "gen %u tok, decode %.1f tok/s | TTFT %.0f ms | %d threads]",
        t->load_s, t->n_prompt, mynah_slm_prefill_tok_s(t),
        t->n_gen, mynah_slm_decode_tok_s(t), t->ttft_s * 1000.0, t->n_threads);
}
