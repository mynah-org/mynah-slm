/* timing.h — one timing struct, filled by the engine, read by CLI and server.
 *
 * Every delivery mode reports its own speed; see PLAN.md "Speed Is Always
 * Reported". One implementation and three consumers, so a tok/s printed by the
 * CLI and a decode_tok_s returned by the server can never mean two different
 * things.
 *
 * SPDX-License-Identifier: MIT */
#ifndef MYNAH_SLM_TIMING_H
#define MYNAH_SLM_TIMING_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    double load_s;        /* open + mmap + bind */
    double prefill_s;     /* prompt tokens through the model */
    double decode_s;      /* generated tokens, EXCLUDING prefill */
    double ttft_s;        /* stamped at the FIRST token, never reconstructed */

    uint32_t n_prompt;
    uint32_t n_gen;
    int      n_threads;

    /* internal marks */
    double t0_;
    int    ttft_done_;
} mynah_slm_timing;

/* Monotonic seconds. Never wall-clock: a clock adjustment mid-run must not be
 * able to produce a negative decode time. */
double mynah_slm_now(void);

void mynah_slm_timing_reset(mynah_slm_timing *t);

/* Mark the start of a phase, then close it. */
void mynah_slm_timing_start(mynah_slm_timing *t);
void mynah_slm_timing_end_load(mynah_slm_timing *t);
void mynah_slm_timing_end_prefill(mynah_slm_timing *t, uint32_t n_prompt);

/* Call for every generated token. The first call stamps TTFT — measured, not
 * derived from a total afterwards, because those two numbers differ and the
 * difference is exactly what a streaming pipeline cares about. */
void mynah_slm_timing_token(mynah_slm_timing *t);
void mynah_slm_timing_end_decode(mynah_slm_timing *t);

double mynah_slm_prefill_tok_s(const mynah_slm_timing *t);
double mynah_slm_decode_tok_s(const mynah_slm_timing *t);

/* The one-line summary. Callers send it to STDERR so stdout stays pipeable
 * into mynah-tts. Returns the number of bytes written. */
int mynah_slm_timing_format(const mynah_slm_timing *t, char *buf, size_t n);

#endif /* MYNAH_SLM_TIMING_H */
