/* sched.h — the continuous-batching scheduler, model-free.
 *
 * ONE thread runs this loop and is the only thread that touches the model.
 * Everything here is policy over opaque jobs; what a job IS (a KV cache, a
 * sampler, a socket) lives behind the engine callbacks, so the policy can be
 * unit-tested with a fake engine (tests/test_sched.c) and the model-facing
 * adapter stays small.
 *
 * One iteration, in this order (mynah-tts src/inference.c serve(), ported):
 *
 *   1. ADMIT   fill free slots from the pending queue, oldest first. Block
 *              for a job ONLY when no slot is live — otherwise an empty queue
 *              must not delay the step the live streams are waiting for.
 *   2. REAP    poll every live AND queued job's `cancelled` hook (client
 *              gone, write failed, deadline) and retire those at once: a disconnected
 *              client never gets another slice or step, and its slot and KV
 *              are free for the next admission in this same iteration's
 *              future. No zombie work.
 *   3. PREFILL slices of at most `prefill_slice` prompt tokens, FIFO TO
 *              COMPLETION (the oldest admitted prefill takes every slice
 *              until it is done), within `prefill_budget_s` of wall time per
 *              iteration while anyone is decoding — at least one slice
 *              always runs, so a prefill cannot starve. With nobody
 *              decoding there is nobody to stall, and prefill runs uncapped
 *              — until a prompt completes: from then on someone decodes, so
 *              the pass stops there and that job is stepped at once rather
 *              than after every other queued prompt.
 *              A job whose prefill completes is stepped in the SAME
 *              iteration.
 *   4. STEP    one decode step for every decoding job, as ONE engine call
 *              (one [B x d] product per weight). If the batched call fails,
 *              each job is re-stepped ALONE and only the ones that fail alone
 *              are retired: one request's failure retires one request.
 *   5. RETIRE  jobs that finished (EOS, max_tokens, a callback stop) or
 *              failed leave their slot.
 *
 * Deliberately NOT here — each was built and measured in mynah-tts and lost
 * (.work/serving-continuous-batching.md): a low-priority prefill helper
 * thread, a second submitter on the engine's pool, cross-worker batching,
 * utilization-aware admission, emission quanta (text has no playback clock:
 * every step emits one token per stream).
 *
 * SPDX-License-Identifier: MIT */
#ifndef MYNAH_SLM_SCHED_H
#define MYNAH_SLM_SCHED_H

#include <stddef.h>
#include <stdint.h>

#include "jobq.h"

#define MYNAH_SLM_SCHED_NO_SLOT UINT32_MAX

/* How a job left its slot. */
typedef enum {
    MYNAH_SLM_JOB_DONE = 0,     /* finished normally (EOS, max_tokens, ...) */
    MYNAH_SLM_JOB_FAILED,       /* the engine failed on it */
    MYNAH_SLM_JOB_CANCELLED,    /* `cancelled` said so: never stepped again */
    MYNAH_SLM_JOB_REFUSED,      /* `admit` refused it: it never held a slot's state */
} mynah_slm_job_outcome;

/* Every callback runs on the scheduler thread. `slot` is in [0, slots). */
typedef struct {
    void *ud;
    /* Build the job's per-request state in slot `slot` (reserve its KV, start
     * its clock). 0, or -1 to refuse it (it is then retired REFUSED). */
    int  (*admit)(void *ud, void *job, uint32_t slot);
    /* At most `budget` more prompt tokens (0 = all). 1 = prompt done,
     * 0 = more remains, -1 = failed. */
    int  (*prefill)(void *ud, void *job, uint32_t slot, uint32_t budget);
    /* ONE decode step for n jobs at once. status[i]: 1 = wants another step,
     * 0 = finished, -1 = failed alone. Returns 0, or -1 when the call failed
     * as a whole — then it must have advanced NO job, so each can be
     * re-stepped alone (isolation). */
    int  (*step)(void *ud, void *const *jobs, const uint32_t *slots, uint32_t n,
                 int *status);
    /* The job leaves its slot. Called exactly once per job taken from the
     * queue, whatever happened to it. `slot` is MYNAH_SLM_SCHED_NO_SLOT for
     * a job cancelled while still queued (it never held one). */
    void (*retire)(void *ud, void *job, uint32_t slot, mynah_slm_job_outcome how);
    /* Non-zero when nobody wants the job any more. Polled once per
     * iteration for every live job AND every queued one (a client that left
     * while queued must not keep its place, or the capacity it is counted
     * in, until a slot frees), and again at admission. Runs under the
     * queue's lock for queued jobs: cheap, and never touching the queue.
     * Optional (NULL = never). */
    int  (*cancelled)(void *ud, void *job);
    /* Monotonic seconds; a fake clock in tests. Optional (NULL = real). */
    double (*now)(void *ud);
} mynah_slm_sched_engine;

typedef struct {
    uint32_t slots;            /* jobs live at once: the decode width */
    uint32_t prefill_slice;    /* prompt tokens per slice; 0 = whole prompt */
    double   prefill_budget_s; /* prefill wall time per iteration while anyone
                                  decodes; 0 = uncapped */
} mynah_slm_sched_cfg;

/* Defaults, overridable by MYNAH_SLM_PREFILL_SLICE / MYNAH_SLM_PREFILL_STEP_MS:
 * 32 tokens and 40 ms, mynah-tts's qualified values — NOT measured for this
 * engine yet (an LLM prompt token costs a different amount). */
void mynah_slm_sched_cfg_defaults(mynah_slm_sched_cfg *c, uint32_t slots);

typedef struct {
    uint32_t live, preparing, decoding;  /* right now */
    uint64_t iterations, steps;          /* steps = engine step calls */
    uint64_t step_rows;                  /* sum of batch widths: mean width = rows / steps */
    uint64_t prefill_slices, isolations;
    uint64_t admitted, done, failed, cancelled, refused;
} mynah_slm_sched_stats;

typedef struct mynah_slm_sched mynah_slm_sched;

mynah_slm_sched *mynah_slm_sched_new(const mynah_slm_sched_cfg *cfg,
                                     const mynah_slm_sched_engine *engine,
                                     mynah_slm_jobq *queue);
void mynah_slm_sched_free(mynah_slm_sched *s);

/* One iteration. With `block_when_idle`, waits for a job when nothing is live.
 * Returns 1 when the queue is closed and nothing is live (drained), else 0. */
int  mynah_slm_sched_iterate(mynah_slm_sched *s, int block_when_idle);

/* Iterate until drained: close the queue to stop it. Jobs already admitted
 * finish; jobs still queued are admitted and finish too. */
void mynah_slm_sched_run(mynah_slm_sched *s);

/* A consistent copy, safe from any thread. */
void mynah_slm_sched_get_stats(mynah_slm_sched *s, mynah_slm_sched_stats *out);

#endif /* MYNAH_SLM_SCHED_H */
