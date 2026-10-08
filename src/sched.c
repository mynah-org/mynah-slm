/* sched.c — see sched.h.
 * SPDX-License-Identifier: MIT */
#include "sched.h"

#include "timing.h"

#include <pthread.h>
#include <stdlib.h>
#include <string.h>

typedef enum { SLOT_FREE = 0, SLOT_PREPARING, SLOT_DECODING } slot_state;

typedef struct {
    slot_state state;
    void      *job;
    uint64_t   prep_seq;     /* admission order: FIFO prefill picks the lowest */
} slot;

struct mynah_slm_sched {
    mynah_slm_sched_cfg    cfg;
    mynah_slm_sched_engine eng;
    mynah_slm_jobq        *q;
    slot                  *slots;
    uint64_t               next_seq;

    /* step scratch, allocated once */
    void     **jobs;
    uint32_t  *idx;
    int       *status;

    pthread_mutex_t stat_mu;      /* guards `st` for readers on other threads */
    mynah_slm_sched_stats st;
};

static double env_double(const char *name, double dflt, double lo, double hi) {
    const char *e = getenv(name);
    if (!e || !*e) return dflt;
    char *end = NULL;
    const double v = strtod(e, &end);
    return (end != e && v >= lo && v <= hi) ? v : dflt;
}

void mynah_slm_sched_cfg_defaults(mynah_slm_sched_cfg *c, uint32_t slots) {
    memset(c, 0, sizeof *c);
    c->slots = slots ? slots : 1;
    c->prefill_slice = (uint32_t)env_double("MYNAH_SLM_PREFILL_SLICE", 32, 0, 1e6);
    c->prefill_budget_s = env_double("MYNAH_SLM_PREFILL_STEP_MS", 40, 0, 1e6) / 1000.0;
}

static double now_s(const mynah_slm_sched *s) {
    return s->eng.now ? s->eng.now(s->eng.ud) : mynah_slm_now();
}

mynah_slm_sched *mynah_slm_sched_new(const mynah_slm_sched_cfg *cfg,
                                     const mynah_slm_sched_engine *engine,
                                     mynah_slm_jobq *queue) {
    if (!cfg || cfg->slots == 0 || !engine || !engine->admit || !engine->prefill ||
        !engine->step || !engine->retire || !queue)
        return NULL;
    mynah_slm_sched *s = calloc(1, sizeof *s);
    if (!s) return NULL;
    s->cfg = *cfg;
    s->eng = *engine;
    s->q = queue;
    s->slots  = calloc(cfg->slots, sizeof *s->slots);
    s->jobs   = calloc(cfg->slots, sizeof *s->jobs);
    s->idx    = calloc(cfg->slots, sizeof *s->idx);
    s->status = calloc(cfg->slots, sizeof *s->status);
    if (!s->slots || !s->jobs || !s->idx || !s->status) {
        mynah_slm_sched_free(s);
        return NULL;
    }
    pthread_mutex_init(&s->stat_mu, NULL);
    return s;
}

void mynah_slm_sched_free(mynah_slm_sched *s) {
    if (!s) return;
    if (s->slots) pthread_mutex_destroy(&s->stat_mu);
    free(s->slots);
    free(s->jobs);
    free(s->idx);
    free(s->status);
    free(s);
}

/* Counters are written here and read by /health from another thread. */
static void count(mynah_slm_sched *s, uint64_t *field, uint64_t by) {
    pthread_mutex_lock(&s->stat_mu);
    *field += by;
    pthread_mutex_unlock(&s->stat_mu);
}

static void retire(mynah_slm_sched *s, uint32_t i, mynah_slm_job_outcome how) {
    slot *sl = &s->slots[i];
    void *job = sl->job;
    sl->state = SLOT_FREE;
    sl->job = NULL;
    s->eng.retire(s->eng.ud, job, i, how);
    pthread_mutex_lock(&s->stat_mu);
    switch (how) {
    case MYNAH_SLM_JOB_DONE:      s->st.done++; break;
    case MYNAH_SLM_JOB_FAILED:    s->st.failed++; break;
    case MYNAH_SLM_JOB_CANCELLED: s->st.cancelled++; break;
    case MYNAH_SLM_JOB_REFUSED:   s->st.refused++; break;
    }
    pthread_mutex_unlock(&s->stat_mu);
}

static uint32_t count_state(const mynah_slm_sched *s, slot_state st) {
    uint32_t n = 0;
    for (uint32_t i = 0; i < s->cfg.slots; i++) n += s->slots[i].state == st;
    return n;
}

static int is_cancelled(const mynah_slm_sched *s, void *job) {
    return s->eng.cancelled && s->eng.cancelled(s->eng.ud, job);
}

/* Returns 1 when the queue reported closed-and-empty while nothing was live. */
static int admit_pass(mynah_slm_sched *s, int block_when_idle) {
    for (;;) {
        uint32_t free_i = s->cfg.slots;
        for (uint32_t i = 0; i < s->cfg.slots; i++)
            if (s->slots[i].state == SLOT_FREE) { free_i = i; break; }
        if (free_i == s->cfg.slots) return 0;

        const uint32_t live = s->cfg.slots - count_state(s, SLOT_FREE);
        const int block = block_when_idle && live == 0;
        void *job = mynah_slm_jobq_pop(s->q, block);
        if (!job) return block ? 1 : 0;      /* blocking pop returns NULL only when closed */

        slot *sl = &s->slots[free_i];
        sl->job = job;
        /* Someone who left while queued costs nothing: not a byte of state
         * is built for them. */
        if (is_cancelled(s, job)) { retire(s, free_i, MYNAH_SLM_JOB_CANCELLED); continue; }
        if (s->eng.admit(s->eng.ud, job, free_i) != 0) {
            retire(s, free_i, MYNAH_SLM_JOB_REFUSED);
            continue;
        }
        sl->state = SLOT_PREPARING;
        sl->prep_seq = s->next_seq++;
        count(s, &s->st.admitted, 1);
    }
}

static void reap_pass(mynah_slm_sched *s) {
    if (!s->eng.cancelled) return;
    for (uint32_t i = 0; i < s->cfg.slots; i++)
        if (s->slots[i].state != SLOT_FREE && is_cancelled(s, s->slots[i].job))
            retire(s, i, MYNAH_SLM_JOB_CANCELLED);
}

static void prefill_pass(mynah_slm_sched *s) {
    const int capped = s->cfg.prefill_budget_s > 0.0 && count_state(s, SLOT_DECODING) > 0;
    const double t0 = capped ? now_s(s) : 0.0;
    uint64_t served = 0;
    for (;;) {
        /* FIFO to completion: always the OLDEST preparing job. Round-robin
         * is processor sharing, which maximises the prefills in flight; tts
         * measured FIFO at -29% TTFA p95 with every request class improving. */
        uint32_t pick = s->cfg.slots;
        for (uint32_t i = 0; i < s->cfg.slots; i++)
            if (s->slots[i].state == SLOT_PREPARING &&
                (pick == s->cfg.slots || s->slots[i].prep_seq < s->slots[pick].prep_seq))
                pick = i;
        if (pick == s->cfg.slots) break;
        /* At least one slice always runs: a cap that can starve a prefill is
         * a deadlock, not a bound. */
        if (capped && served > 0 && now_s(s) - t0 >= s->cfg.prefill_budget_s) break;

        /* Between slices too: a long prompt from a client that just left
         * stops here, not after the whole prompt. */
        if (served > 0 && is_cancelled(s, s->slots[pick].job)) {
            retire(s, pick, MYNAH_SLM_JOB_CANCELLED);
            continue;
        }
        const int rc = s->eng.prefill(s->eng.ud, s->slots[pick].job, pick,
                                      s->cfg.prefill_slice);
        served++;
        if (rc < 0) retire(s, pick, MYNAH_SLM_JOB_FAILED);
        else if (rc == 1) {
            s->slots[pick].state = SLOT_DECODING;
            /* Someone decodes now. An uncapped pass was uncapped only
             * because nobody did: carrying on would make this new stream
             * wait for every other queued prompt before its first step.
             * Stop here, step it in this iteration, and let the next pass
             * run under the budget. */
            if (!capped && s->cfg.prefill_budget_s > 0.0) break;
        }
    }
    if (served) count(s, &s->st.prefill_slices, served);
}

static void apply_status(mynah_slm_sched *s, uint32_t i, int status) {
    if (status == 0) retire(s, i, MYNAH_SLM_JOB_DONE);
    else if (status < 0) retire(s, i, MYNAH_SLM_JOB_FAILED);
}

static void step_pass(mynah_slm_sched *s) {
    uint32_t n = 0;
    for (uint32_t i = 0; i < s->cfg.slots; i++)
        if (s->slots[i].state == SLOT_DECODING) {
            s->jobs[n] = s->slots[i].job;
            s->idx[n] = i;
            n++;
        }
    if (n == 0) return;

    const int rc = s->eng.step(s->eng.ud, s->jobs, s->idx, n, s->status);
    count(s, &s->st.steps, 1);
    count(s, &s->st.step_rows, n);
    if (rc == 0) {
        for (uint32_t k = 0; k < n; k++) apply_status(s, s->idx[k], s->status[k]);
        return;
    }
    if (n == 1) { retire(s, s->idx[0], MYNAH_SLM_JOB_FAILED); return; }

    /* Isolation: the batch failed as a whole and advanced nobody, so each
     * job re-stepped alone takes exactly the step it would have taken alone.
     * Only the ones that fail alone are retired. */
    count(s, &s->st.isolations, 1);
    for (uint32_t k = 0; k < n; k++) {
        void *job = s->jobs[k];
        uint32_t i = s->idx[k];
        int st = -1;
        if (s->eng.step(s->eng.ud, &job, &i, 1, &st) != 0) st = -1;
        count(s, &s->st.steps, 1);
        count(s, &s->st.step_rows, 1);
        apply_status(s, i, st);
    }
}

int mynah_slm_sched_iterate(mynah_slm_sched *s, int block_when_idle) {
    const int drained = admit_pass(s, block_when_idle);
    reap_pass(s);
    prefill_pass(s);
    step_pass(s);

    pthread_mutex_lock(&s->stat_mu);
    s->st.iterations++;
    s->st.preparing = count_state(s, SLOT_PREPARING);
    s->st.decoding  = count_state(s, SLOT_DECODING);
    s->st.live      = s->st.preparing + s->st.decoding;
    const uint32_t live = s->st.live;
    pthread_mutex_unlock(&s->stat_mu);
    return drained && live == 0;
}

void mynah_slm_sched_run(mynah_slm_sched *s) {
    while (!mynah_slm_sched_iterate(s, 1)) {}
}

void mynah_slm_sched_get_stats(mynah_slm_sched *s, mynah_slm_sched_stats *out) {
    pthread_mutex_lock(&s->stat_mu);
    *out = s->st;
    pthread_mutex_unlock(&s->stat_mu);
}
