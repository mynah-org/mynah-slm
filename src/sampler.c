/* sampler.c — see sampler.h.
 *
 * Filter order is fixed and matters: penalties, then temperature, then top-k,
 * then top-p, then min-p. Applying min-p before temperature, for instance,
 * measures a probability the caller never asked for.
 *
 * SPDX-License-Identifier: MIT */
#include "sampler.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

typedef struct { float p; uint32_t id; } cand;

struct mynah_slm_sampler {
    mynah_slm_sampler_params par;
    uint32_t vocab;

    cand    *buf;          /* vocab-sized working set, allocated once */
    uint32_t *recent;      /* ring of accepted tokens, for the penalties */
    uint32_t  n_recent, recent_head;

    uint64_t rng;
};

void mynah_slm_sampler_defaults(mynah_slm_sampler_params *p) {
    memset(p, 0, sizeof *p);
    p->temp           = 0.6f;      /* Qwen3 generation_config.json */
    p->top_k          = 20;
    p->top_p          = 0.95f;
    p->min_p          = 0.0f;
    p->repeat_penalty = 1.0f;
    p->penalty_window = 64;
    p->seed           = 0;
}

/* splitmix64: small, fast, and reproducible across platforms — which rand()
 * emphatically is not. */
static double rng_next(uint64_t *state) {
    uint64_t z = (*state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    z =  z ^ (z >> 31);
    return (double)(z >> 11) * (1.0 / 9007199254740992.0);   /* [0,1) */
}

mynah_slm_sampler *mynah_slm_sampler_new(const mynah_slm_sampler_params *p,
                                         uint32_t vocab_size) {
    mynah_slm_sampler *s = calloc(1, sizeof *s);
    if (!s) return NULL;

    s->par   = *p;
    s->vocab = vocab_size;
    s->rng   = p->seed ? p->seed : 0x853C49E6748FEA9Bull;

    s->buf = malloc((size_t)vocab_size * sizeof *s->buf);
    if (p->penalty_window) s->recent = calloc(p->penalty_window, sizeof *s->recent);
    if (!s->buf || (p->penalty_window && !s->recent)) {
        mynah_slm_sampler_free(s);
        return NULL;
    }
    return s;
}

void mynah_slm_sampler_free(mynah_slm_sampler *s) {
    if (!s) return;
    free(s->buf);
    free(s->recent);
    free(s);
}

void mynah_slm_sampler_accept(mynah_slm_sampler *s, uint32_t token) {
    if (!s->recent || s->par.penalty_window == 0) return;
    s->recent[s->recent_head] = token;
    s->recent_head = (s->recent_head + 1) % s->par.penalty_window;
    if (s->n_recent < s->par.penalty_window) s->n_recent++;
}

static void apply_penalties(mynah_slm_sampler *s, float *logits) {
    const mynah_slm_sampler_params *p = &s->par;
    if (!s->recent || s->n_recent == 0) return;
    if (p->repeat_penalty == 1.0f && p->freq_penalty == 0.0f && p->presence_penalty == 0.0f)
        return;

    for (uint32_t i = 0; i < s->n_recent; i++) {
        const uint32_t tok = s->recent[i];
        if (tok >= s->vocab) continue;

        /* Count occurrences for the frequency term. O(window^2) with a window
         * of 64 is nothing; a histogram over 151936 entries per token would be
         * the expensive choice here. */
        uint32_t count = 0;
        for (uint32_t k = 0; k < s->n_recent; k++) if (s->recent[k] == tok) count++;

        float l = logits[tok];
        if (p->repeat_penalty != 1.0f) {
            /* Divide when positive, multiply when negative: scaling a negative
             * logit down would REWARD the token, which is the classic bug. */
            l = (l > 0.0f) ? l / p->repeat_penalty : l * p->repeat_penalty;
        }
        l -= p->freq_penalty * (float)count;
        l -= p->presence_penalty;
        logits[tok] = l;
    }
}

static int by_prob_desc(const void *a, const void *b) {
    const cand *x = a, *y = b;
    if (x->p > y->p) return -1;
    if (x->p < y->p) return  1;
    /* Ties broken by id, so the result does not depend on qsort's internals. */
    return (x->id < y->id) ? -1 : (x->id > y->id);
}

uint32_t mynah_slm_sample(mynah_slm_sampler *s, float *logits) {
    const mynah_slm_sampler_params *p = &s->par;
    const uint32_t V = s->vocab;

    apply_penalties(s, logits);

    /* Greedy: no sort, no softmax, no RNG. Also the path the parity harness
     * uses, so it has to be exactly argmax and nothing else. */
    if (p->temp <= 0.0f) {
        uint32_t best = 0;
        float bl = logits[0];
        for (uint32_t i = 1; i < V; i++) if (logits[i] > bl) { bl = logits[i]; best = i; }
        return best;
    }

    /* Softmax at temperature, max-subtracted. */
    float mx = logits[0];
    for (uint32_t i = 1; i < V; i++) if (logits[i] > mx) mx = logits[i];

    double sum = 0.0;
    for (uint32_t i = 0; i < V; i++) {
        const float e = expf((logits[i] - mx) / p->temp);
        s->buf[i].p  = e;
        s->buf[i].id = i;
        sum += (double)e;
    }
    const float inv = (float)(1.0 / sum);
    for (uint32_t i = 0; i < V; i++) s->buf[i].p *= inv;

    /* top-k first: it bounds the sort, and everything after is cheap. */
    uint32_t n = V;
    if (p->top_k > 0 && p->top_k < n) {
        qsort(s->buf, n, sizeof *s->buf, by_prob_desc);
        n = p->top_k;
    } else {
        qsort(s->buf, n, sizeof *s->buf, by_prob_desc);
    }

    /* min-p: relative to the best candidate, so it adapts to how confident the
     * distribution is instead of imposing a fixed floor. */
    if (p->min_p > 0.0f && n > 0) {
        const float floor_p = p->min_p * s->buf[0].p;
        uint32_t keep = 1;
        while (keep < n && s->buf[keep].p >= floor_p) keep++;
        n = keep;
    }

    /* top-p: smallest prefix whose mass reaches p. The token that crosses the
     * threshold is kept, or a top_p below the best token's own probability
     * would leave nothing to sample from. */
    if (p->top_p < 1.0f && n > 0) {
        double acc = 0.0;
        uint32_t keep = 0;
        while (keep < n) {
            acc += (double)s->buf[keep].p;
            keep++;
            if (acc >= (double)p->top_p) break;
        }
        n = keep;
    }
    if (n == 0) n = 1;

    double total = 0.0;
    for (uint32_t i = 0; i < n; i++) total += (double)s->buf[i].p;

    double r = rng_next(&s->rng) * total;
    for (uint32_t i = 0; i < n; i++) {
        r -= (double)s->buf[i].p;
        if (r <= 0.0) return s->buf[i].id;
    }
    return s->buf[n - 1].id;      /* only reachable through rounding */
}
