/* slots.h — continuous batching for the server: the model-facing engine the
 * scheduler (src/sched.h) drives, and the per-request writer.
 *
 * OWNERSHIP, which is the whole design (mynah-tts serving-design.md §2):
 *
 *   scheduler thread   the model, the pool, the workspace, every slot's
 *                      sequence (KV), every request's sampler and generation.
 *                      Never reads or writes a socket.
 *   connection thread  its own socket: parses, enqueues (or answers 503),
 *                      then becomes that request's WRITER — sends the SSE
 *                      header when the request is admitted, then whatever
 *                      the scheduler appended, and probes the peer while it
 *                      waits. Never touches the model.
 *
 * Between them, per request, one mutex over an append buffer. The scheduler
 * only ever appends and signals, so a slow client can never block a step:
 * if its unsent bytes pass a cap, or its send times out, it is CANCELLED at
 * the next iteration (backpressure is cancellation, never a blocking write).
 *
 * SPDX-License-Identifier: MIT */
#ifndef MYNAH_SLM_SERVER_SLOTS_H
#define MYNAH_SLM_SERVER_SLOTS_H

#include <stddef.h>
#include <stdint.h>

#include "http.h"
#include "model.h"
#include "sampler.h"
#include "sched.h"
#include "timing.h"
#include "tokenizer.h"

/* What one chat request asks for. Pointers are borrowed for the duration of
 * slots_run. */
typedef struct {
    const uint32_t *ids;
    size_t          n_prompt;
    uint32_t        max_new;
    mynah_slm_sampler_params sp;
    const uint32_t *eos;
    size_t          n_eos;
    long            think_open, think_close;    /* -1 = no split */
    long            tool_open, tool_close;      /* -1 = no tool channel */
    int             stream;
    const char     *req_id;
    const char     *model_json;                 /* already a JSON string */
} slots_params;

typedef struct {
    mynah_slm_job_outcome outcome;
    char  *content;         /* non-stream answer text, malloc'd (caller frees) */
    size_t content_used;
    char  *tool;            /* the tool channel, malloc'd (caller frees) */
    size_t tool_used;
    mynah_slm_timing tm;    /* clock started at ADMISSION */
    double queue_ms;        /* arrival to admission */
    int    header_sent;     /* the SSE header went out (at admission) */
    int    client_gone;
    int    shutdown;        /* cancelled because the server is stopping */
    int    stop;            /* mynah_slm_stop: why the generation ended */
    int    error_status;    /* HTTP status for a refusal/failure (0 = default) */
    char   error[192];      /* why it was refused or failed */
} slots_result;

/* Loads nothing: takes the loaded model and tokenizer, builds the workspace
 * for `ctx_cap` positions and `n_slots` sequences, and starts the scheduler
 * thread. queue_cap bounds the requests waiting for a slot. 0 or -1.
 *
 * kv_budget_mb bounds the KV cache ALL slots hold together (allocated, not
 * used: a pooled slot keeps its cache between requests). A request whose
 * cache would pass it waits first in line while others run, and is refused
 * (503) if it could not fit even alone. A slot whose cache grew past its
 * fair share (budget / slots) gives it back when its request ends.
 * 0 = default: slots x ctx positions, capped at a quarter of physical
 * memory, never below one full-context request. Per position the cache is
 * 2 (K, V) x layers x kv_heads x head_dim x 2 bytes (bf16): 112 KiB for a
 * 0.6B Qwen3 (28 x 8 x 128), so --slots 4 --ctx 8192 is 3.5 GiB at most. */
int  slots_start(mynah_slm_model_t *m, const mynah_slm_tokenizer *tok,
                 uint32_t n_slots, uint32_t ctx_cap, uint32_t queue_cap,
                 uint32_t kv_budget_mb, char *err, size_t errsz);

/* Shutdown, in two calls, because connection threads are detached and may
 * still be inside slots_run / slots_health when the accept loop returns:
 *
 *   slots_shutdown  marks the engine STOPPING (every later slots_run and
 *                   slots_health refuses, under a lock, without touching the
 *                   queue or the scheduler), closes the queue and joins the
 *                   scheduler thread. Nothing is freed. The join is
 *                   BOUNDED: every request still queued is cancelled at
 *                   once (result.shutdown = 1, never admitted), and every
 *                   live one at its next step once `grace_ms` has passed —
 *                   one step or one prefill slice, not max_tokens.
 *   slots_free      waits, at most `wait_ms`, for the last thread inside
 *                   slots_run / slots_health to leave, then frees the queue,
 *                   the scheduler, the slots and the workspace. Returns 0, or
 *                   -1 when a thread is still inside: then NOTHING is freed
 *                   (the process is exiting; a leak is safe, a free under a
 *                   running thread is not). */
void slots_shutdown(int grace_ms);
int  slots_free(int wait_ms);

/* Runs one request on the calling connection thread: enqueue, then write
 * what the scheduler produces until it retires the request. Returns 0 with
 * `out` filled; -1 when the queue is full (answer 503, nothing was
 * written); -2 when the server is shutting down (answer 503, nothing was
 * written). */
#define SLOTS_BUSY     (-1)
#define SLOTS_STOPPING (-2)
int  slots_run(http_conn *conn, const slots_params *p, slots_result *out);

/* JSON object members for /health (no braces): slots, live, queue depth,
 * steps, mean batch width, aggregate and per-stream recent decode t/s,
 * cancellations, the decode product. -1 (nothing written) once the server
 * is shutting down. */
int  slots_health(char *buf, size_t n);

/* Times a per-request output buffer grew from inside the token loop, in
 * either serving mode (expected: 0). The serialized path records its own
 * through slots_note_loop_alloc. */
void          slots_note_loop_alloc(void);
unsigned long slots_loop_allocs(void);

#endif /* MYNAH_SLM_SERVER_SLOTS_H */
