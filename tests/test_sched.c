/* test_sched.c — the scheduler's policy, with a fake engine and a fake clock.
 *
 * src/sched.c is policy over opaque jobs, so every rule it promises can be
 * pinned without a model: admission order, blocking only when idle, the slice
 * and the per-iteration budget, FIFO-to-completion prefill, decode never
 * starved by a long prompt, retirement, cancellation freeing the slot at the
 * next iteration, step isolation, and the bounded queue's 503. The fake
 * engine logs every call; the fake clock advances by a fixed cost per prefill
 * token and per step, so time budgets are exact and the test is deterministic.
 *
 * The last part is threaded (producers + the scheduler thread, the shape the
 * server runs) and is what `make tsan-sched` runs under ThreadSanitizer.
 *
 * SPDX-License-Identifier: MIT */
#include "jobq.h"
#include "sched.h"

#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int failures;

static void check(const char *what, int ok, const char *detail) {
    if (ok) printf("ok   %s\n", what);
    else { printf("FAIL %s\n     <- %s\n", what, detail ? detail : ""); failures++; }
}

/* ── the fake engine ──────────────────────────────────────────────────────── */

typedef struct {
    int id;
    int prompt;          /* tokens to prefill */
    int max_steps;       /* finishes (status 0) after this many steps */
    int poison;          /* fails every step, alone or not */
    int refuse;          /* admit() refuses it */
    atomic_int cancel;   /* the client left */

    /* filled by the engine */
    int prefilled, steps, admitted_slot, retired;
    mynah_slm_job_outcome outcome;
    long admit_iter, first_step_iter, retire_iter;
} job;

#define LOG_MAX 4096

typedef struct {
    double clock;                  /* seconds */
    double cost_token, cost_step;  /* fake costs */
    long   iter;                   /* set by the test before each iterate */
    /* event log */
    char   kind[LOG_MAX];          /* 'A' admit, 'P' prefill, 'S' step, 'R' retire */
    int    who[LOG_MAX], arg[LOG_MAX];
    long   when[LOG_MAX];
    int    n;
    int    max_width;
    int    live, max_live;
} fake;

static void logev(fake *f, char k, int who, int arg) {
    if (f->n < LOG_MAX) {
        f->kind[f->n] = k; f->who[f->n] = who; f->arg[f->n] = arg; f->when[f->n] = f->iter;
        f->n++;
    }
}

static int f_admit(void *ud, void *j_, uint32_t slot) {
    fake *f = ud; job *j = j_;
    if (j->refuse) return -1;
    j->admitted_slot = (int)slot;
    j->admit_iter = f->iter;
    j->first_step_iter = -1;
    logev(f, 'A', j->id, (int)slot);
    if (++f->live > f->max_live) f->max_live = f->live;
    return 0;
}

static int f_prefill(void *ud, void *j_, uint32_t slot, uint32_t budget) {
    fake *f = ud; job *j = j_;
    (void)slot;
    int take = j->prompt - j->prefilled;
    if (budget && take > (int)budget) take = (int)budget;
    j->prefilled += take;
    f->clock += take * f->cost_token;
    logev(f, 'P', j->id, take);
    return j->prefilled == j->prompt ? 1 : 0;
}

static int f_step(void *ud, void *const *jobs, const uint32_t *slots, uint32_t n, int *status) {
    fake *f = ud;
    (void)slots;
    if ((int)n > f->max_width) f->max_width = (int)n;
    /* A poisoned job fails the batch as a whole and advances nobody — the
     * contract forward_multi keeps — and fails alone too. */
    for (uint32_t i = 0; i < n; i++)
        if (((job *)jobs[i])->poison) { f->clock += f->cost_step; return -1; }
    for (uint32_t i = 0; i < n; i++) {
        job *j = jobs[i];
        if (j->first_step_iter < 0) j->first_step_iter = f->iter;
        j->steps++;
        logev(f, 'S', j->id, (int)n);
        status[i] = j->steps >= j->max_steps ? 0 : 1;
    }
    f->clock += f->cost_step;
    return 0;
}

static void f_retire(void *ud, void *j_, uint32_t slot, mynah_slm_job_outcome how) {
    fake *f = ud; job *j = j_;
    (void)slot;
    j->retired++;
    j->outcome = how;
    j->retire_iter = f->iter;
    logev(f, 'R', j->id, (int)how);
    if (how != MYNAH_SLM_JOB_REFUSED && !(how == MYNAH_SLM_JOB_CANCELLED && j->admitted_slot < 0))
        f->live--;
}

static int f_cancelled(void *ud, void *j_) {
    (void)ud;
    return atomic_load(&((job *)j_)->cancel);
}

static double f_now(void *ud) { return ((fake *)ud)->clock; }

static mynah_slm_sched_engine engine_for(fake *f) {
    mynah_slm_sched_engine e = { f, f_admit, f_prefill, f_step, f_retire, f_cancelled, f_now };
    return e;
}

static void job_init(job *j, int id, int prompt, int max_steps) {
    memset(j, 0, sizeof *j);
    j->id = id;
    j->prompt = prompt;
    j->max_steps = max_steps;
    j->admitted_slot = -1;
    atomic_init(&j->cancel, 0);
}

/* Run until every job in `jobs` retired, at most `limit` iterations. */
static int run_until_done(mynah_slm_sched *s, fake *f, job *jobs, int n, int limit) {
    for (int it = 0; it < limit; it++) {
        f->iter++;
        mynah_slm_sched_iterate(s, 0);
        int all = 1;
        for (int k = 0; k < n; k++) if (!jobs[k].retired) all = 0;
        if (all) return it + 1;
    }
    return -1;
}

static int index_of(const fake *f, char kind, int who) {
    for (int i = 0; i < f->n; i++) if (f->kind[i] == kind && f->who[i] == who) return i;
    return -1;
}

/* ── the cases ────────────────────────────────────────────────────────────── */

static void t_admission_order(void) {
    printf("\n-- admission --\n");
    fake f = { .cost_token = 0.001, .cost_step = 0.01 };
    mynah_slm_jobq *q = mynah_slm_jobq_new(16);
    mynah_slm_sched_engine e = engine_for(&f);
    mynah_slm_sched_cfg cfg = { 2, 0, 0 };
    mynah_slm_sched *s = mynah_slm_sched_new(&cfg, &e, q);
    job jobs[5];
    for (int k = 0; k < 5; k++) { job_init(&jobs[k], k, 3, 2); mynah_slm_jobq_push(q, &jobs[k]); }
    const int its = run_until_done(s, &f, jobs, 5, 100);

    int order_ok = 1, prev = -1;
    for (int i = 0; i < f.n; i++)
        if (f.kind[i] == 'A') { if (f.who[i] != prev + 1) order_ok = 0; prev = f.who[i]; }
    char d[128];
    snprintf(d, sizeof d, "%d iterations, max live %d, max width %d", its, f.max_live, f.max_width);
    check("every job admitted, in the order it was queued", its > 0 && order_ok && prev == 4, d);
    check("never more live than slots, and the step batches all of them",
          f.max_live == 2 && f.max_width == 2, d);
    int once = 1;
    for (int k = 0; k < 5; k++) if (jobs[k].retired != 1 || jobs[k].outcome != MYNAH_SLM_JOB_DONE) once = 0;
    check("each job retired exactly once, DONE", once, NULL);
    check("a freed slot is reused by the next job",
          jobs[2].admitted_slot >= 0 && jobs[2].admit_iter > jobs[0].admit_iter, NULL);

    /* Block only when nothing is live: with a live job and an empty queue,
     * iterate(block=1) must return at once (a hang here is the failure). */
    job late;
    job_init(&late, 9, 1, 50);
    mynah_slm_jobq_push(q, &late);
    f.iter++;
    mynah_slm_sched_iterate(s, 1);           /* admits it */
    f.iter++;
    mynah_slm_sched_iterate(s, 1);           /* live job, empty queue: must not block */
    check("with a live slot an empty queue does not block the step", late.steps == 2, NULL);
    atomic_store(&late.cancel, 1);
    mynah_slm_jobq_close(q);
    mynah_slm_sched_run(s);                  /* drains: returns because closed and idle */
    check("closing the queue drains the scheduler", late.retired == 1, NULL);

    mynah_slm_sched_free(s);
    mynah_slm_jobq_free(q);
}

static void t_prefill_policy(void) {
    printf("\n-- prefill: slices, budget, FIFO, no starvation --\n");
    /* 10 ms per prompt token, 32-token slices = 320 ms a slice; budget 500 ms
     * while someone decodes: the first slice always runs, the second starts
     * at 320 < 500, the third would start at 640 >= 500 and waits. */
    fake f = { .cost_token = 0.010, .cost_step = 0.030 };
    mynah_slm_jobq *q = mynah_slm_jobq_new(16);
    mynah_slm_sched_engine e = engine_for(&f);
    mynah_slm_sched_cfg cfg = { 4, 32, 0.5 };
    mynah_slm_sched *s = mynah_slm_sched_new(&cfg, &e, q);

    job dec, big1, big2;
    job_init(&dec, 0, 1, 1000);              /* a stream already decoding */
    job_init(&big1, 1, 200, 3);              /* a long prompt */
    job_init(&big2, 2, 100, 3);              /* another, admitted after it */
    mynah_slm_jobq_push(q, &dec);
    f.iter++; mynah_slm_sched_iterate(s, 0);
    check("a 1-token prompt goes from admission to its first step in one iteration",
          dec.first_step_iter == dec.admit_iter && dec.steps == 1, NULL);

    mynah_slm_jobq_push(q, &big1);
    mynah_slm_jobq_push(q, &big2);
    const int steps_before = dec.steps;
    int budget_ok = 1, iters = 0;
    double worst_ms = 0.0;
    while (!(big1.steps && big2.steps) && iters < 100) {
        const int n0 = f.n;
        f.iter++; iters++;
        mynah_slm_sched_iterate(s, 0);
        int tokens = 0;
        for (int i = n0; i < f.n; i++) if (f.kind[i] == 'P') {
            tokens += f.arg[i];
            if (f.arg[i] > 32) budget_ok = 0;
        }
        if (tokens * 10.0 > worst_ms) worst_ms = tokens * 10.0;
    }
    char d[160];
    snprintf(d, sizeof d, "%d iterations for 300 prompt tokens, worst iteration %.0f ms of prefill",
             iters, worst_ms);
    check("no slice is longer than prefill_slice", budget_ok, d);
    /* The bound is the budget plus the one slice already running when it ran
     * out: 500 + 320 ms. Uncapped, one iteration would take all 3000 ms. */
    check("prefill per iteration is bounded by budget + one slice (<= 820 ms)",
          worst_ms <= 820.0 && iters >= 4, d);
    check("the decoding stream stepped every iteration during both prefills",
          dec.steps - steps_before == iters, d);

    /* FIFO to completion: every slice of big1 before any slice of big2. */
    int last_big1 = -1, first_big2 = -1;
    for (int i = 0; i < f.n; i++) {
        if (f.kind[i] == 'P' && f.who[i] == 1) last_big1 = i;
        if (f.kind[i] == 'P' && f.who[i] == 2 && first_big2 < 0) first_big2 = i;
    }
    check("prefill is FIFO to completion: the older prompt finishes first",
          last_big1 >= 0 && first_big2 > last_big1, NULL);
    check("a prefill that completes is stepped in the same iteration",
          big1.first_step_iter >= 0 && f.when[last_big1] == big1.first_step_iter, NULL);

    /* With nobody decoding there is nobody to stall: uncapped. */
    atomic_store(&dec.cancel, 1);
    for (int k = 0; k < 10 && !(big1.retired && big2.retired); k++) { f.iter++; mynah_slm_sched_iterate(s, 0); }
    job lone;
    job_init(&lone, 3, 200, 1);
    mynah_slm_jobq_push(q, &lone);
    const int n0 = f.n;
    f.iter++; mynah_slm_sched_iterate(s, 0);
    int slices = 0;
    for (int i = n0; i < f.n; i++) if (f.kind[i] == 'P') slices++;
    snprintf(d, sizeof d, "%d slices in one iteration, prefilled %d of 200", slices, lone.prefilled);
    check("alone, a prompt is prefilled in one iteration (no budget applies)",
          lone.prefilled == 200 && lone.steps == 1, d);

    mynah_slm_sched_free(s);
    mynah_slm_jobq_free(q);
}

/* A prompt that completes during an UNCAPPED pass (nobody was decoding when
 * it began) turns its job into a decoder: the pass must not go on prefilling
 * every other queued prompt before that job's first step. The review's
 * reproducer (three 3-slice prompts, 0.3 s a slice, a 40 ms budget): A's
 * first step came 1.8 s of B's and C's prefill after its prompt was done. */
static void t_prefill_turns_decoder(void) {
    printf("\n-- prefill: a job that starts decoding mid-pass is stepped at once --\n");
    fake f = { .cost_token = 0.3 / 32, .cost_step = 0.02 };
    mynah_slm_jobq *q = mynah_slm_jobq_new(8);
    mynah_slm_sched_engine e = engine_for(&f);
    mynah_slm_sched_cfg cfg = { 4, 32, 0.040 };
    mynah_slm_sched *s = mynah_slm_sched_new(&cfg, &e, q);
    job a, b, c;
    job_init(&a, 0, 96, 50);
    job_init(&b, 1, 96, 50);
    job_init(&c, 2, 96, 50);
    mynah_slm_jobq_push(q, &a);
    mynah_slm_jobq_push(q, &b);
    mynah_slm_jobq_push(q, &c);
    f.iter++;
    mynah_slm_sched_iterate(s, 0);
    const int a_step = index_of(&f, 'S', 0), b_pre = index_of(&f, 'P', 1);
    double ms = 0.0;     /* prefill run between A's last slice and its first step */
    int after_a = 0;
    for (int i = 0; i < f.n && (a_step < 0 || i < a_step); i++) {
        if (f.kind[i] == 'P' && f.who[i] == 0) { after_a = 1; ms = 0.0; continue; }
        if (after_a && f.kind[i] == 'P') ms += f.arg[i] * f.cost_token * 1000.0;
    }
    char d[160];
    snprintf(d, sizeof d, "A stepped at event %d, B's first slice at event %d; %.0f ms of "
             "other prefill between A's prompt and its first step", a_step, b_pre, ms);
    check("a prompt finished in an uncapped pass is stepped before any other prompt is prefilled",
          a_step >= 0 && a.first_step_iter == 1 && (b_pre < 0 || a_step < b_pre), d);
    /* From then on the budget applies: B and C advance by at most budget +
     * one slice per iteration while A decodes every iteration. */
    int worst = 0, a_steps0 = a.steps, iters = 0;
    while (!(b.steps && c.steps) && iters < 50) {
        const int n0 = f.n;
        f.iter++; iters++;
        mynah_slm_sched_iterate(s, 0);
        int slices = 0;
        for (int i = n0; i < f.n; i++) slices += f.kind[i] == 'P';
        if (slices > worst) worst = slices;
    }
    snprintf(d, sizeof d, "%d iterations, at most %d slices in one, A stepped %d times",
             iters, worst, a.steps - a_steps0);
    check("... and the other prompts then run under the budget (one 300 ms slice per iteration)",
          worst == 1 && a.steps - a_steps0 == iters, d);
    atomic_store(&a.cancel, 1);
    atomic_store(&b.cancel, 1);
    atomic_store(&c.cancel, 1);
    mynah_slm_jobq_close(q);
    mynah_slm_sched_run(s);
    mynah_slm_sched_free(s);
    mynah_slm_jobq_free(q);
}

static void t_cancel_and_isolation(void) {
    printf("\n-- cancellation and isolation --\n");
    fake f = { .cost_token = 0.001, .cost_step = 0.01 };
    mynah_slm_jobq *q = mynah_slm_jobq_new(16);
    mynah_slm_sched_engine e = engine_for(&f);
    mynah_slm_sched_cfg cfg = { 2, 16, 0.02 };
    mynah_slm_sched *s = mynah_slm_sched_new(&cfg, &e, q);

    /* A decoding job whose client leaves: retired at the NEXT iteration,
     * never stepped again, and its slot taken by the waiting job. */
    job a, b, c;
    job_init(&a, 0, 1, 1000);
    job_init(&b, 1, 1, 1000);
    job_init(&c, 2, 1, 2);
    mynah_slm_jobq_push(q, &a);
    mynah_slm_jobq_push(q, &b);
    mynah_slm_jobq_push(q, &c);              /* waits: both slots are taken */
    for (int k = 0; k < 3; k++) { f.iter++; mynah_slm_sched_iterate(s, 0); }
    const int a_steps = a.steps;
    atomic_store(&a.cancel, 1);
    f.iter++;
    const long cancel_iter = f.iter;
    mynah_slm_sched_iterate(s, 0);
    char d[160];
    snprintf(d, sizeof d, "a: %d steps before, %d after, outcome %d at iter %ld; c admitted at %ld",
             a_steps, a.steps, (int)a.outcome, a.retire_iter, c.admit_iter);
    check("a cancelled job retires at the next iteration and is not stepped again",
          a.retired == 1 && a.outcome == MYNAH_SLM_JOB_CANCELLED && a.steps == a_steps &&
          a.retire_iter == cancel_iter, d);
    f.iter++;
    mynah_slm_sched_iterate(s, 0);
    check("its slot goes to the waiting job",
          c.admitted_slot == a.admitted_slot && c.steps >= 1, d);

    /* Cancelled while queued: never admitted, no engine state built. */
    job gone;
    job_init(&gone, 3, 50, 5);
    atomic_store(&gone.cancel, 1);
    mynah_slm_jobq_push(q, &gone);
    for (int k = 0; k < 4; k++) { f.iter++; mynah_slm_sched_iterate(s, 0); }
    check("a job cancelled while queued is retired without admission or prefill",
          gone.retired == 1 && gone.outcome == MYNAH_SLM_JOB_CANCELLED &&
          index_of(&f, 'A', 3) < 0 && index_of(&f, 'P', 3) < 0, NULL);

    /* Cancelled between prefill slices: the rest of the prompt never runs. */
    atomic_store(&b.cancel, 1);
    for (int k = 0; k < 3 && !c.retired; k++) { f.iter++; mynah_slm_sched_iterate(s, 0); }
    job longp;
    job_init(&longp, 4, 1000, 5);
    job other;
    job_init(&other, 5, 1, 1000);           /* decoding, so the budget applies */
    mynah_slm_jobq_push(q, &other);
    f.iter++; mynah_slm_sched_iterate(s, 0);
    mynah_slm_jobq_push(q, &longp);
    f.iter++; mynah_slm_sched_iterate(s, 0);
    const int done_before = longp.prefilled;
    atomic_store(&longp.cancel, 1);
    f.iter++; mynah_slm_sched_iterate(s, 0);
    snprintf(d, sizeof d, "prefilled %d of 1000 before the cancel, %d after", done_before, longp.prefilled);
    check("a prompt whose client left is not prefilled any further",
          longp.retired == 1 && longp.prefilled == done_before && done_before < 1000, d);

    /* Isolation: a poisoned job in a batch of two. The batch fails, each is
     * re-stepped alone, the poisoned one retires FAILED and the other takes
     * exactly ONE step for that iteration (not zero, not two). */
    job bad;
    job_init(&bad, 6, 1, 1000);
    bad.poison = 1;
    mynah_slm_jobq_push(q, &bad);
    f.iter++; mynah_slm_sched_iterate(s, 0);
    const int other_before = other.steps;
    mynah_slm_sched_stats st;
    mynah_slm_sched_get_stats(s, &st);
    snprintf(d, sizeof d, "bad outcome %d, other stepped %d -> %d, isolations %llu",
             (int)bad.outcome, other_before, other.steps, (unsigned long long)st.isolations);
    check("a failing job retires alone; its batch-mate steps exactly once",
          bad.retired == 1 && bad.outcome == MYNAH_SLM_JOB_FAILED && st.isolations == 1, d);
    f.iter++; mynah_slm_sched_iterate(s, 0);
    check("... and keeps stepping afterwards", other.steps == other_before + 1 && !other.retired, d);

    /* Refused at admission: retired REFUSED, never prefilled. */
    job ref;
    job_init(&ref, 7, 5, 5);
    ref.refuse = 1;
    mynah_slm_jobq_push(q, &ref);
    f.iter++; mynah_slm_sched_iterate(s, 0);
    check("a job admit() refuses is retired REFUSED and never prefilled",
          ref.retired == 1 && ref.outcome == MYNAH_SLM_JOB_REFUSED && index_of(&f, 'P', 7) < 0, NULL);

    atomic_store(&other.cancel, 1);
    mynah_slm_jobq_close(q);
    mynah_slm_sched_run(s);
    mynah_slm_sched_free(s);
    mynah_slm_jobq_free(q);
}

static void t_queue_bound(void) {
    printf("\n-- the pending queue --\n");
    mynah_slm_jobq *q = mynah_slm_jobq_new(2);
    int x, y, z;
    const int a = mynah_slm_jobq_push(q, &x), b = mynah_slm_jobq_push(q, &y);
    const int c = mynah_slm_jobq_push(q, &z);
    check("a full queue refuses at once (the caller answers 503)", a == 0 && b == 0 && c == -1, NULL);
    check("FIFO", mynah_slm_jobq_pop(q, 0) == &x && mynah_slm_jobq_pop(q, 0) == &y &&
                  mynah_slm_jobq_pop(q, 0) == NULL, NULL);
    mynah_slm_jobq_close(q);
    check("a closed queue refuses pushes and a blocking pop returns NULL",
          mynah_slm_jobq_push(q, &x) == -1 && mynah_slm_jobq_pop(q, 1) == NULL, NULL);
    mynah_slm_jobq_free(q);
}

/* ── threaded: producers + one scheduler thread, the server's shape ───────── */

#define PRODUCERS 4
#define PER_PROD  60

typedef struct { mynah_slm_jobq *q; job *jobs; atomic_int *rejected; int base; } prod_arg;

static void *producer(void *arg) {
    prod_arg *p = arg;
    for (int k = 0; k < PER_PROD; k++) {
        job *j = &p->jobs[p->base + k];
        if (mynah_slm_jobq_push(p->q, j) != 0) {
            atomic_fetch_add(p->rejected, 1);    /* a 503 */
            j->retired = -1;                     /* never entered */
        }
        if (k % 7 == 3) atomic_store(&j->cancel, 1);   /* some clients leave */
        if (k % 5 == 0) usleep(200);
    }
    return NULL;
}

typedef struct { mynah_slm_sched *s; } sched_arg;
static void *sched_thread(void *arg) { mynah_slm_sched_run(((sched_arg *)arg)->s); return NULL; }

static void t_threaded(void) {
    printf("\n-- threaded: %d producers x %d jobs, one scheduler --\n", PRODUCERS, PER_PROD);
    fake f = { .cost_token = 0, .cost_step = 0 };
    mynah_slm_jobq *q = mynah_slm_jobq_new(8);
    mynah_slm_sched_engine e = engine_for(&f);
    e.now = NULL;                           /* the real clock */
    mynah_slm_sched_cfg cfg = { 3, 4, 0.001 };
    mynah_slm_sched *s = mynah_slm_sched_new(&cfg, &e, q);

    static job jobs[PRODUCERS * PER_PROD];
    for (int k = 0; k < PRODUCERS * PER_PROD; k++) job_init(&jobs[k], k, 1 + k % 9, 1 + k % 5);
    atomic_int rejected;
    atomic_init(&rejected, 0);

    pthread_t st, pt[PRODUCERS];
    sched_arg sa = { s };
    prod_arg pa[PRODUCERS];
    pthread_create(&st, NULL, sched_thread, &sa);
    for (int p = 0; p < PRODUCERS; p++) {
        pa[p] = (prod_arg){ q, jobs, &rejected, p * PER_PROD };
        pthread_create(&pt[p], NULL, producer, &pa[p]);
    }
    for (int p = 0; p < PRODUCERS; p++) pthread_join(pt[p], NULL);
    mynah_slm_jobq_close(q);
    pthread_join(st, NULL);

    int once = 1, never = 0, entered = 0;
    for (int k = 0; k < PRODUCERS * PER_PROD; k++) {
        if (jobs[k].retired == -1) { never++; continue; }
        entered++;
        if (jobs[k].retired != 1) once = 0;
    }
    mynah_slm_sched_stats sst;
    mynah_slm_sched_get_stats(s, &sst);
    char d[200];
    snprintf(d, sizeof d, "%d entered, %d refused 503 (counted %d); done %llu cancelled %llu",
             entered, never, atomic_load(&rejected), (unsigned long long)sst.done,
             (unsigned long long)sst.cancelled);
    check("every queued job retired exactly once, every refusal counted", once &&
          never == atomic_load(&rejected) &&
          sst.done + sst.cancelled + sst.failed + sst.refused == (uint64_t)entered, d);
    printf("     %s\n", d);
    mynah_slm_sched_free(s);
    mynah_slm_jobq_free(q);
}

int main(void) {
    t_admission_order();
    t_prefill_policy();
    t_prefill_turns_decoder();
    t_cancel_and_isolation();
    t_queue_bound();
    t_threaded();
    printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "PASS",
           failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
