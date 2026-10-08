/* slots.c — see slots.h.
 * SPDX-License-Identifier: MIT */
#include "slots.h"

#include "arch_qwen3.h"
#include "generate.h"
#include "jobq.h"
#include "json.h"
#include "mynah_slm.h"
#include "threads.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* Unsent stream bytes past which a client counts as too slow and is
 * cancelled (MYNAH_SLM_STREAM_MAX_BYTES). The writer drains into the socket
 * as fast as the client reads, so this only fills when the client stops. */
#define DEFAULT_MAX_PENDING (1u << 20)

typedef struct slots_req {
    slots_params p;
    http_conn   *conn;
    double       created;

    /* ── the scheduler's alone ─────────────────────────────────────────── */
    mynah_slm_sampler   *sam;
    mynah_slm_gen        gen;
    mynah_slm_gen_params gp;
    mynah_slm_timing     tm;
    int                  started;      /* gen_start ran */
    double               admitted_at;

    /* ── shared, under mu ──────────────────────────────────────────────── */
    pthread_mutex_t mu;
    pthread_cond_t  cv;
    char  *out;  size_t out_used, out_cap;          /* stream frames to send */
    char  *text; size_t text_used, text_cap;        /* non-stream answer */
    char  *tool; size_t tool_used, tool_cap;        /* tool channel */
    int    admitted, done, oom;
    mynah_slm_job_outcome outcome;
    char   error[192];

    /* Set by the writer (send failed / peer gone) or by an append that
     * found the client too slow; read by the scheduler's cancel poll. */
    atomic_int gone;
    /* Set by the scheduler when it cancelled this request because the
     * server is stopping (not because the client left). */
    atomic_int shutdown;
} slots_req;

/* ── the engine ───────────────────────────────────────────────────────────── */

#define AGG_RING 128

static struct {
    mynah_slm_model_t         *model;
    const mynah_slm_tokenizer *tok;
    uint32_t        n_slots, ctx_cap;
    size_t          max_pending;
    mynah_slm_state ws;
    mynah_slm_seq  *seqs;           /* one per slot, pooled across requests */
    mynah_slm_jobq *q;
    mynah_slm_sched *sched;
    /* Requests in the system — queued or holding a slot — counted at
     * ARRIVAL. The refusal is `in_flight >= slots + queue` (tts's rule:
     * running + queued >= slots + queue_cap), not "the queue is full": a
     * burst arrives faster than the scheduler's next admission pass, and
     * judging by the queue alone refused requests while slots stood free. */
    atomic_uint in_flight;
    uint32_t    capacity;
    pthread_t       thread;
    int             running;

    /* Lifetime. Connection threads are detached, so at shutdown some may
     * still be inside slots_run / slots_health — reading a request that
     * arrived late, or asking for /health. `users` counts the threads
     * inside; `stopping` makes every later entry refuse without touching
     * the queue or the scheduler. Both under life_mu, which is static and
     * never destroyed, so a thread arriving after slots_free still finds a
     * valid lock and a "no". Nothing behind it is freed while users > 0. */
    pthread_mutex_t life_mu;
    pthread_cond_t  life_cv;
    int             stopping;
    unsigned        users;

    /* Shutdown deadline: once `halting` is set, queued requests are
     * cancelled at once and live ones when the clock passes halt_at.
     * halt_at is written before the release store and read after the
     * acquire load. */
    atomic_int      halting;
    double          halt_at;
    const char     *product;        /* resolved before the thread starts */

    /* step scratch, scheduler thread only */
    mynah_slm_seq **step_seqs;
    uint32_t       *step_tok;
    float         **step_logits;
    uint32_t       *step_pick;

    /* /health: recent per-stream decode t/s, and tokens per step for the
     * aggregate rate. Written by the scheduler, read by HTTP threads. */
    pthread_mutex_t stat_mu;
    double   stream_tok_s[16];
    unsigned n_stream, stream_head;
    double   agg_t[AGG_RING];
    unsigned agg_n[AGG_RING];
    unsigned agg_head, agg_count;
} S;

/* Enter / leave the engine from a connection thread. -1 = stopping. */
static int life_enter(void) {
    pthread_mutex_lock(&S.life_mu);
    const int ok = !S.stopping;
    if (ok) S.users++;
    pthread_mutex_unlock(&S.life_mu);
    return ok ? 0 : -1;
}

static void life_leave(void) {
    pthread_mutex_lock(&S.life_mu);
    if (--S.users == 0) pthread_cond_broadcast(&S.life_cv);
    pthread_mutex_unlock(&S.life_mu);
}

static void stat_step(unsigned tokens) {
    pthread_mutex_lock(&S.stat_mu);
    S.agg_t[S.agg_head] = mynah_slm_now();
    S.agg_n[S.agg_head] = tokens;
    S.agg_head = (S.agg_head + 1) % AGG_RING;
    if (S.agg_count < AGG_RING) S.agg_count++;
    pthread_mutex_unlock(&S.stat_mu);
}

static void stat_stream(double tok_s) {
    pthread_mutex_lock(&S.stat_mu);
    S.stream_tok_s[S.stream_head] = tok_s;
    S.stream_head = (S.stream_head + 1) % 16;
    if (S.n_stream < 16) S.n_stream++;
    pthread_mutex_unlock(&S.stat_mu);
}

/* Append under the request's lock. Growth reallocs: the buffers are per
 * request and amortized, and the alternative — a fixed ring — would turn a
 * long non-stream answer into a truncation. */
static int append(char **buf, size_t *used, size_t *cap, const char *s, size_t n) {
    if (*used + n + 1 > *cap) {
        size_t nc = (*used + n + 1) * 2;
        if (nc < 256) nc = 256;
        char *g = realloc(*buf, nc);
        if (!g) return -1;
        *buf = g;
        *cap = nc;
    }
    memcpy(*buf + *used, s, n);
    *used += n;
    (*buf)[*used] = '\0';
    return 0;
}

/* The answer channel. Stream: one SSE frame per token, appended for the
 * writer. Non-stream: the text, accumulated. Never a socket write here. */
static int answer_cb(void *ctx, uint32_t id, const char *text, size_t len) {
    (void)id;
    slots_req *r = ctx;
    if (len == 0) return 0;
    int rc = 0;
    pthread_mutex_lock(&r->mu);
    if (r->p.stream) {
        char esc[2048], frame[3072];
        json_escape(text, len, esc, sizeof esc);
        const int n = snprintf(frame, sizeof frame,
            "data: {\"id\":\"%s\",\"object\":\"chat.completion.chunk\","
            "\"model\":%s,\"choices\":[{\"index\":0,\"delta\":{\"content\":%s},"
            "\"finish_reason\":null}]}\n\n",
            r->p.req_id, r->p.model_json, esc);
        if (append(&r->out, &r->out_used, &r->out_cap, frame, (size_t)n) != 0) r->oom = rc = 1;
        else if (r->out_used > S.max_pending) atomic_store(&r->gone, 1);   /* too slow */
    } else if (append(&r->text, &r->text_used, &r->text_cap, text, len) != 0) {
        r->oom = rc = 1;
    }
    pthread_cond_signal(&r->cv);
    pthread_mutex_unlock(&r->mu);
    return rc;
}

static int tool_cb(void *ctx, uint32_t id, const char *text, size_t len) {
    (void)id;
    slots_req *r = ctx;
    if (len == 0) return 0;
    pthread_mutex_lock(&r->mu);
    const int rc = append(&r->tool, &r->tool_used, &r->tool_cap, text, len) != 0;
    if (rc) r->oom = 1;
    pthread_mutex_unlock(&r->mu);
    return rc;
}

static int e_admit(void *ud, void *job, uint32_t slot) {
    (void)ud;
    slots_req *r = job;
    r->admitted_at = mynah_slm_now();
    mynah_slm_timing_reset(&r->tm);
    r->tm.n_threads = mynah_slm_threads_count();
    mynah_slm_timing_start(&r->tm);

    if (r->p.n_prompt + 1 > S.ctx_cap) {
        snprintf(r->error, sizeof r->error,
                 "the prompt (%zu tokens) does not fit the %u-token context",
                 r->p.n_prompt, S.ctx_cap);
        return -1;
    }
    /* KV sized to what this request can use, from the slot's pooled
     * allocation when it is big enough: no allocation in steady state. */
    uint64_t need = (uint64_t)r->p.n_prompt + r->p.max_new + 8;
    if (need > S.ctx_cap) need = S.ctx_cap;
    if (mynah_slm_seq_reserve(&S.seqs[slot], S.model, (uint32_t)need, MYNAH_SLM_KV_BF16,
                              MYNAH_SLM_KV_BF16, r->error, sizeof r->error) != 0)
        return -1;
    mynah_slm_timing_end_load(&r->tm);

    r->sam = mynah_slm_sampler_new(&r->p.sp, mynah_slm_vocab_size(S.model));
    if (!r->sam) { snprintf(r->error, sizeof r->error, "out of memory"); return -1; }

    mynah_slm_gen_params_init(&r->gp);
    r->gp.prompt = r->p.ids;
    r->gp.n_prompt = r->p.n_prompt;
    r->gp.max_new = r->p.max_new;
    /* The slot's cache may be smaller than prompt + max_new (capped by the
     * context): reaching its end is a LENGTH stop, never a failed step. */
    r->gp.n_ctx = S.seqs[slot].n_ctx;
    r->gp.eos = r->p.eos;
    r->gp.n_eos = r->p.n_eos;
    r->gp.cb = answer_cb;
    r->gp.cb_ctx = r;
    r->gp.think_open = r->p.think_open;
    r->gp.think_close = r->p.think_close;
    if (r->p.tool_open >= 0 && r->p.tool_close >= 0) {
        r->gp.tool_open = r->p.tool_open;
        r->gp.tool_close = r->p.tool_close;
        r->gp.cb_tool = tool_cb;
        r->gp.cb_tool_ctx = r;
    }
    /* No gp.cancel: the scheduler polls the same probe once per iteration
     * and between prefill slices, which is the same boundary. */
    if (mynah_slm_gen_start(&r->gen, S.tok, r->sam, &r->gp, &r->tm) != 0) {
        snprintf(r->error, sizeof r->error, "empty prompt");
        return -1;
    }
    r->started = 1;

    /* Admission is when the client hears back: its writer sends the SSE
     * header now, so TTFB means "you have a slot" (tts serving-design §7). */
    pthread_mutex_lock(&r->mu);
    r->admitted = 1;
    pthread_cond_signal(&r->cv);
    pthread_mutex_unlock(&r->mu);
    return 0;
}

static int e_prefill(void *ud, void *job, uint32_t slot, uint32_t budget) {
    (void)ud;
    slots_req *r = job;
    return mynah_slm_gen_prefill(&r->gen, &S.ws, &S.seqs[slot], budget);
}

static int e_step(void *ud, void *const *jobs, const uint32_t *slots, uint32_t n,
                  int *status) {
    (void)ud;
    uint32_t m = 0;
    for (uint32_t i = 0; i < n; i++) {
        slots_req *r = jobs[i];
        status[i] = 0;
        if (!mynah_slm_gen_wants_step(&r->gen)) continue;   /* max_tokens 0 */
        S.step_pick[m] = i;
        S.step_seqs[m] = &S.seqs[slots[i]];
        S.step_tok[m] = mynah_slm_gen_next_token(&r->gen);
        m++;
    }
    if (m == 0) return 0;
    /* All or nothing: on failure no sequence moved, and the scheduler
     * re-steps each alone (isolation). */
    if (mynah_slm_forward_multi(&S.ws, S.step_seqs, S.step_tok, m, S.step_logits) != 0)
        return -1;
    for (uint32_t k = 0; k < m; k++) {
        slots_req *r = jobs[S.step_pick[k]];
        mynah_slm_gen_accept_logits(&r->gen, S.step_logits[k]);
        status[S.step_pick[k]] = r->gen.stop == MYNAH_SLM_STOP_ERROR ? -1
                               : mynah_slm_gen_wants_step(&r->gen) ? 1 : 0;
    }
    stat_step(m);
    return 0;
}

static void e_retire(void *ud, void *job, uint32_t slot, mynah_slm_job_outcome how) {
    (void)ud;
    (void)slot;
    slots_req *r = job;
    if (r->started) {
        mynah_slm_gen_finish(&r->gen);          /* flushes held bytes, ends the clock */
        if (how == MYNAH_SLM_JOB_CANCELLED) r->tm.cancelled = 1;
        if (how == MYNAH_SLM_JOB_DONE && r->tm.n_gen) stat_stream(mynah_slm_decode_tok_s(&r->tm));
    }
    mynah_slm_sampler_free(r->sam);
    r->sam = NULL;
    /* Last touch: after the unlock the connection thread may free r. */
    pthread_mutex_lock(&r->mu);
    r->outcome = how;
    r->done = 1;
    pthread_cond_signal(&r->cv);
    pthread_mutex_unlock(&r->mu);
}

static int e_cancelled(void *ud, void *job) {
    (void)ud;
    slots_req *r = job;
    if (atomic_load(&r->gone)) return 1;
    if (http_peer_gone(r->conn)) { atomic_store(&r->gone, 1); return 1; }
    /* Stopping: nobody new is admitted, and a live request runs on only
     * for the grace period. Bounded shutdown, not a drain to max_tokens. */
    if (atomic_load_explicit(&S.halting, memory_order_acquire) &&
        (!r->started || mynah_slm_now() >= S.halt_at)) {
        atomic_store(&r->shutdown, 1);
        return 1;
    }
    return 0;
}

static void *sched_main(void *arg) {
    (void)arg;
    mynah_slm_sched_run(S.sched);
    return NULL;
}

int slots_start(mynah_slm_model_t *m, const mynah_slm_tokenizer *tok,
                uint32_t n_slots, uint32_t ctx_cap, uint32_t queue_cap,
                char *err, size_t errsz) {
    memset(&S, 0, sizeof S);
    pthread_mutex_init(&S.life_mu, NULL);
    pthread_cond_init(&S.life_cv, NULL);
    S.model = m;
    S.tok = tok;
    S.n_slots = n_slots ? n_slots : 1;
    pthread_mutex_init(&S.stat_mu, NULL);
    const char *e = getenv("MYNAH_SLM_STREAM_MAX_BYTES");
    S.max_pending = e ? (size_t)strtoull(e, NULL, 10) : DEFAULT_MAX_PENDING;
    if (S.max_pending < 4096) S.max_pending = 4096;

    if (mynah_slm_state_init_workspace(&S.ws, m, ctx_cap, err, errsz) != 0 ||
        mynah_slm_state_init_decode(&S.ws, S.n_slots, err, errsz) != 0)
        return -1;
    S.ctx_cap = S.ws.ctx_cap;
    S.seqs = calloc(S.n_slots, sizeof *S.seqs);
    S.step_seqs = calloc(S.n_slots, sizeof *S.step_seqs);
    S.step_tok = calloc(S.n_slots, sizeof *S.step_tok);
    S.step_logits = calloc(S.n_slots, sizeof *S.step_logits);
    S.step_pick = calloc(S.n_slots, sizeof *S.step_pick);
    const uint32_t queue = queue_cap ? queue_cap : S.n_slots * 2;
    S.capacity = S.n_slots + queue;
    atomic_init(&S.in_flight, 0);
    atomic_init(&S.halting, 0);
    /* The ring holds everything that may be in flight, so a push the
     * counter allowed can never fail on it. */
    S.q = mynah_slm_jobq_new(S.capacity);
    if (!S.seqs || !S.step_seqs || !S.step_tok || !S.step_logits || !S.step_pick || !S.q) {
        snprintf(err, errsz, "out of memory for %u slots", S.n_slots);
        return -1;
    }
    mynah_slm_sched_cfg cfg;
    mynah_slm_sched_cfg_defaults(&cfg, S.n_slots);
    mynah_slm_sched_engine eng = { NULL, e_admit, e_prefill, e_step, e_retire,
                                   e_cancelled, NULL };
    S.sched = mynah_slm_sched_new(&cfg, &eng, S.q);
    if (!S.sched) { snprintf(err, errsz, "cannot build the scheduler"); return -1; }
    /* Resolved here, on this thread, before the scheduler exists: the name
     * is what /health reports, so the dispatch that RAN is on record. */
    S.product = mynah_slm_decode_product_name();
    fprintf(stderr, "slots: %u | capacity %zu (slots + queue) | ctx %u | decode product %s | prefill slice %u "
                    "tokens, %.0f ms per step\n",
            S.n_slots, mynah_slm_jobq_cap(S.q), S.ctx_cap, S.product,
            cfg.prefill_slice, cfg.prefill_budget_s * 1000.0);
    if (pthread_create(&S.thread, NULL, sched_main, NULL) != 0) {
        snprintf(err, errsz, "cannot start the scheduler thread");
        return -1;
    }
    S.running = 1;
    return 0;
}

void slots_shutdown(int grace_ms) {
    pthread_mutex_lock(&S.life_mu);
    S.stopping = 1;
    pthread_mutex_unlock(&S.life_mu);
    S.halt_at = mynah_slm_now() + (grace_ms > 0 ? grace_ms : 0) / 1000.0;
    atomic_store_explicit(&S.halting, 1, memory_order_release);
    if (!S.running) return;
    /* The queue and the scheduler stay valid: a thread already inside
     * slots_run may still push (refused: closed) or wait for its retire. */
    mynah_slm_jobq_close(S.q);
    pthread_join(S.thread, NULL);
    S.running = 0;
}

int slots_free(int wait_ms) {
    struct timespec dl;
    clock_gettime(CLOCK_REALTIME, &dl);
    dl.tv_sec += wait_ms / 1000;
    dl.tv_nsec += (long)(wait_ms % 1000) * 1000000L;
    if (dl.tv_nsec >= 1000000000L) { dl.tv_nsec -= 1000000000L; dl.tv_sec++; }
    pthread_mutex_lock(&S.life_mu);
    S.stopping = 1;
    while (S.users > 0)
        if (pthread_cond_timedwait(&S.life_cv, &S.life_mu, &dl) != 0) break;
    const unsigned users = S.users;
    pthread_mutex_unlock(&S.life_mu);
    if (users > 0 || S.running) return -1;
    mynah_slm_sched_free(S.sched);
    mynah_slm_jobq_free(S.q);
    for (uint32_t i = 0; i < S.n_slots; i++) mynah_slm_seq_free(&S.seqs[i]);
    free(S.seqs); free(S.step_seqs); free(S.step_tok); free(S.step_logits); free(S.step_pick);
    mynah_slm_state_free(&S.ws);
    S.sched = NULL;
    S.q = NULL;
    S.seqs = NULL;
    return 0;
}

/* ── the writer: the connection thread ────────────────────────────────────── */

static int run_entered(http_conn *conn, const slots_params *p, slots_result *out);
static int health_entered(char *buf, size_t n);

int slots_run(http_conn *conn, const slots_params *p, slots_result *out) {
    memset(out, 0, sizeof *out);
    if (life_enter() != 0) return SLOTS_STOPPING;
    const int rc = run_entered(conn, p, out);
    life_leave();
    return rc;
}

static int run_entered(http_conn *conn, const slots_params *p, slots_result *out) {
    slots_req *r = calloc(1, sizeof *r);
    if (!r) { snprintf(out->error, sizeof out->error, "out of memory"); out->outcome = MYNAH_SLM_JOB_FAILED; return 0; }
    r->p = *p;
    r->conn = conn;
    r->created = mynah_slm_now();
    pthread_mutex_init(&r->mu, NULL);
    pthread_cond_init(&r->cv, NULL);
    atomic_init(&r->gone, 0);
    atomic_init(&r->shutdown, 0);

    if (atomic_fetch_add(&S.in_flight, 1) >= S.capacity ||
        mynah_slm_jobq_push(S.q, r) != 0) {
        atomic_fetch_sub(&S.in_flight, 1);
        pthread_cond_destroy(&r->cv);
        pthread_mutex_destroy(&r->mu);
        free(r);
        /* A push refused because shutdown closed the queue is not "busy". */
        pthread_mutex_lock(&S.life_mu);
        const int stopping = S.stopping;
        pthread_mutex_unlock(&S.life_mu);
        return stopping ? SLOTS_STOPPING : SLOTS_BUSY;
    }

    char  *wbuf = NULL;
    size_t wcap = 0;
    pthread_mutex_lock(&r->mu);
    for (;;) {
        if (r->admitted && p->stream && !out->header_sent && !atomic_load(&r->gone)) {
            pthread_mutex_unlock(&r->mu);
            http_begin_sse(conn);
            out->header_sent = 1;
            pthread_mutex_lock(&r->mu);
            continue;
        }
        if (r->out_used && out->header_sent) {
            /* Take the bytes, write them unlocked: the scheduler keeps
             * appending while this thread sits in send(). */
            if (r->out_used > wcap) {
                char *g = realloc(wbuf, r->out_used);
                if (!g) { atomic_store(&r->gone, 1); r->out_used = 0; continue; }
                wbuf = g;
                wcap = r->out_used;
            }
            const size_t n = r->out_used;
            memcpy(wbuf, r->out, n);
            r->out_used = 0;
            pthread_mutex_unlock(&r->mu);
            if (!atomic_load(&r->gone) && http_write(conn, wbuf, n) != 0)
                atomic_store(&r->gone, 1);            /* reaped at the next step */
            pthread_mutex_lock(&r->mu);
            continue;
        }
        if (r->done) break;
        /* A client whose read side reached EOF may have half-closed or
         * closed: write to it (header, then a probe every interval) so a
         * closed one answers with a reset the scheduler's probe sees. */
        if (!atomic_load(&r->gone)) {
            pthread_mutex_unlock(&r->mu);
            http_keepalive(conn, p->stream);
            /* Stopped reading: the acknowledged byte count has not moved
             * for the send timeout. Reaped at the next iteration. */
            if (http_send_stalled(conn)) atomic_store(&r->gone, 1);
            pthread_mutex_lock(&r->mu);
            if (http_head_sent(conn) && p->stream) out->header_sent = 1;
            /* Frames may have arrived while unlocked: their signal is gone. */
            if ((r->out_used && out->header_sent) || r->done) continue;
        }
        /* Still queued: nobody else looks at this client until it is
         * admitted, so probe it here. A client that left while queued is
         * marked gone, and the scheduler takes it out of the queue at its
         * next iteration — its place and its share of the capacity are
         * given back then, not when a slot frees. */
        if (!r->admitted && !atomic_load(&r->gone) && http_peer_gone(conn))
            atomic_store(&r->gone, 1);
        struct timespec dl;
        clock_gettime(CLOCK_REALTIME, &dl);
        dl.tv_nsec += 50 * 1000000L;
        if (dl.tv_nsec >= 1000000000L) { dl.tv_nsec -= 1000000000L; dl.tv_sec++; }
        pthread_cond_timedwait(&r->cv, &r->mu, &dl);
    }
    /* Retired: the scheduler is done with r. */
    out->outcome = r->outcome;
    out->content = r->text;          r->text = NULL;
    out->content_used = r->text_used;
    out->tool = r->tool;             r->tool = NULL;
    out->tool_used = r->tool_used;
    out->tm = r->tm;
    out->queue_ms = r->admitted_at > 0.0 ? (r->admitted_at - r->created) * 1000.0 : 0.0;
    out->client_gone = atomic_load(&r->gone);
    out->shutdown = atomic_load(&r->shutdown);
    out->stop = r->started ? (int)r->gen.stop : MYNAH_SLM_STOP_NONE;
    if (r->oom && out->outcome == MYNAH_SLM_JOB_DONE) out->outcome = MYNAH_SLM_JOB_FAILED;
    snprintf(out->error, sizeof out->error, "%s", r->error);
    pthread_mutex_unlock(&r->mu);

    atomic_fetch_sub(&S.in_flight, 1);
    free(wbuf);
    free(r->out);
    pthread_cond_destroy(&r->cv);
    pthread_mutex_destroy(&r->mu);
    free(r);
    return 0;
}

int slots_health(char *buf, size_t n) {
    if (life_enter() != 0) return -1;
    const int rc = health_entered(buf, n);
    life_leave();
    return rc;
}

static int health_entered(char *buf, size_t n) {
    mynah_slm_sched_stats st;
    mynah_slm_sched_get_stats(S.sched, &st);
    pthread_mutex_lock(&S.stat_mu);
    double per = 0.0;
    for (unsigned i = 0; i < S.n_stream; i++) per += S.stream_tok_s[i];
    if (S.n_stream) per /= S.n_stream;
    /* Aggregate: tokens over the steps of the last 10 s. */
    const double now = mynah_slm_now();
    double t_old = now;
    unsigned tokens = 0;
    for (unsigned k = 0; k < S.agg_count; k++) {
        const unsigned i = (S.agg_head + AGG_RING - 1 - k) % AGG_RING;
        if (now - S.agg_t[i] > 10.0) break;
        tokens += S.agg_n[i];
        t_old = S.agg_t[i];
    }
    pthread_mutex_unlock(&S.stat_mu);
    const double agg = (now - t_old) > 0.0 ? tokens / (now - t_old) : 0.0;
    return snprintf(buf, n,
        "\"slots\":%u,\"live\":%u,\"preparing\":%u,\"decoding\":%u,\"queued\":%zu,"
        "\"capacity\":%u,\"steps\":%llu,\"mean_batch\":%.2f,"
        "\"aggregate_decode_tok_s\":%.2f,\"recent_stream_decode_tok_s\":%.2f,"
        "\"slot_cancelled\":%llu,\"slot_failed\":%llu,\"slot_done\":%llu,"
        "\"decode_product\":\"%s\"",
        S.n_slots, st.live, st.preparing, st.decoding, mynah_slm_jobq_depth(S.q),
        S.capacity, (unsigned long long)st.steps,
        st.steps ? (double)st.step_rows / (double)st.steps : 0.0, agg, per,
        (unsigned long long)st.cancelled, (unsigned long long)st.failed,
        (unsigned long long)st.done, S.product);
}
