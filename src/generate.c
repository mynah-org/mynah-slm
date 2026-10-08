/* generate.c — see generate.h.
 * SPDX-License-Identifier: MIT */
#include "generate.h"

#include <stdlib.h>
#include <string.h>

void mynah_slm_gen_params_init(mynah_slm_gen_params *p) {
    if (!p) return;
    memset(p, 0, sizeof *p);
    p->think_open = p->think_close = -1;
    p->tool_open  = p->tool_close  = -1;
}

static int is_eos(const mynah_slm_gen_params *p, uint32_t id) {
    for (size_t i = 0; i < p->n_eos; i++) if (p->eos[i] == id) return 1;
    return 0;
}

enum { CH_ANSWER = 0, CH_THINK, CH_TOOL, CH_N };

int mynah_slm_gen_start(mynah_slm_gen *g, const mynah_slm_tokenizer *tok,
                        mynah_slm_sampler *sam, const mynah_slm_gen_params *p,
                        mynah_slm_timing *t) {
    memset(g, 0, sizeof *g);
    if (!p || p->n_prompt == 0 || !p->prompt) { g->stop = MYNAH_SLM_STOP_ERROR; return -1; }
    g->p   = *p;
    g->tok = tok;
    g->sam = sam;
    g->t   = t;
    g->chan = CH_ANSWER;
    /* The markers are whole tokens, so no real UTF-8 sequence ever straddles
     * them and one detokenizer per channel is exact. */
    for (int i = 0; i < CH_N; i++) mynah_slm_detok_init(&g->d[i], tok);
    return 0;
}

size_t mynah_slm_gen_prefill_left(const mynah_slm_gen *g) {
    return g->decoding ? 0 : (g->p.n_prompt - 1) - g->prefilled;
}

/* The reference forward pass (arch_qwen3.c) as a driver. */
typedef struct { mynah_slm_state *ws; mynah_slm_seq *q; } ref_driver;

static int ref_prefill(void *ctx, const uint32_t *tokens, uint32_t n) {
    ref_driver *r = ctx;
    return mynah_slm_seq_forward_batch(r->ws, r->q, tokens, n, NULL);
}

static float *ref_step(void *ctx, uint32_t token) {
    ref_driver *r = ctx;
    return mynah_slm_seq_forward(r->ws, r->q, token, r->ws->logits) == 0 ? r->ws->logits : NULL;
}

int mynah_slm_gen_prefill(mynah_slm_gen *g, mynah_slm_state *ws, mynah_slm_seq *q,
                          uint32_t budget) {
    ref_driver r = { ws, q };
    const mynah_slm_gen_driver d = { ref_prefill, ref_step, mynah_slm_batch_max(ws), &r };
    return mynah_slm_gen_prefill_driver(g, &d, budget);
}

int mynah_slm_gen_prefill_driver(mynah_slm_gen *g, const mynah_slm_gen_driver *d,
                                 uint32_t budget) {
    if (g->stop == MYNAH_SLM_STOP_ERROR || g->stop == MYNAH_SLM_STOP_CANCELLED) return -1;
    if (g->decoding) return 1;

    /* Every prompt token but the last only needs to populate the KV cache, so
     * its logits are never materialized. On a 151936-row tied head that skips
     * one 151936x1024 matvec per prompt token, which is most of the prefill
     * cost — and it is exactly the kind of saving that has to be free of any
     * behaviour change, since the last token's logits are computed the same
     * way either way. */
    const size_t n_pre = g->p.n_prompt - 1;
    const uint32_t bmax = d->batch_max ? d->batch_max : 1;
    size_t left = budget ? budget : n_pre;
    while (g->prefilled < n_pre && left > 0) {
        /* Before every batch: a prompt of thousands of tokens is seconds of
         * work, and the client may already have left. */
        if (mynah_slm_gen_check_cancel(g)) return -1;
        size_t take = n_pre - g->prefilled;
        if (take > bmax) take = bmax;
        if (take > left) take = left;

        /* One call per batch instead of one per token: the weights are read
         * once for the whole group, which is the difference between prefill
         * being memory-bound and being compute-bound. A batch of one falls
         * back to the single-token path inside forward_batch. */
        if (d->prefill(d->ctx, g->p.prompt + g->prefilled, (uint32_t)take) != 0) {
            g->stop = MYNAH_SLM_STOP_ERROR;
            return -1;
        }
        if (g->sam)
            for (size_t k = 0; k < take; k++)
                mynah_slm_sampler_accept(g->sam, g->p.prompt[g->prefilled + k]);
        g->prefilled += take;
        left -= take;
    }
    if (g->prefilled < n_pre) return 0;

    if (g->t) mynah_slm_timing_end_prefill(g->t, (uint32_t)g->p.n_prompt);
    g->next = g->p.prompt[g->p.n_prompt - 1];
    if (g->sam) mynah_slm_sampler_accept(g->sam, g->next);
    g->decoding = 1;
    return 1;
}

int mynah_slm_gen_wants_step(const mynah_slm_gen *g) {
    return g->decoding && g->stop == MYNAH_SLM_STOP_NONE && g->step < g->p.max_new;
}

uint32_t mynah_slm_gen_next_token(const mynah_slm_gen *g) { return g->next; }

int mynah_slm_gen_check_cancel(mynah_slm_gen *g) {
    if (g->stop == MYNAH_SLM_STOP_CANCELLED) return 1;
    if (!g->p.cancel || g->stop != MYNAH_SLM_STOP_NONE) return 0;
    if (!g->p.cancel(g->p.cancel_ctx)) return 0;
    g->stop = MYNAH_SLM_STOP_CANCELLED;
    if (g->t) g->t->cancelled = 1;
    return 1;
}

void mynah_slm_gen_fail(mynah_slm_gen *g) {
    if (g->stop == MYNAH_SLM_STOP_NONE) g->stop = MYNAH_SLM_STOP_ERROR;
}

int mynah_slm_gen_accept_logits(mynah_slm_gen *g, float *logits) {
    if (!mynah_slm_gen_wants_step(g)) return 0;
    g->step++;

    const uint32_t id = mynah_slm_sample(g->sam, logits);
    mynah_slm_sampler_accept(g->sam, id);

    /* Stop BEFORE emitting: a terminator is a control token, and printing
     * "<|im_end|>" at the end of every answer is a bug users see. */
    if (is_eos(&g->p, id)) {
        g->produced++;
        if (g->t) mynah_slm_timing_token(g->t);
        g->stop = MYNAH_SLM_STOP_EOS;
        return 0;
    }

    if (g->t) mynah_slm_timing_token(g->t);
    g->produced++;

    /* The markers are structure, not content: they open and close a channel
     * and are emitted on neither. */
    const mynah_slm_gen_params *p = &g->p;
    const int split_think = (p->think_open >= 0 && p->think_close >= 0);
    const int split_tool  = (p->tool_open  >= 0 && p->tool_close  >= 0);
    int marker = 1;
    if      (split_think && id == (uint32_t)p->think_open)  g->chan = CH_THINK;
    else if (split_think && id == (uint32_t)p->think_close) g->chan = CH_ANSWER;
    else if (split_tool  && id == (uint32_t)p->tool_open)   g->chan = CH_TOOL;
    else if (split_tool  && id == (uint32_t)p->tool_close)  g->chan = CH_ANSWER;
    else marker = 0;
    if (marker) {
        g->next = id;
        goto out;
    }

    char piece[512];
    const long w = mynah_slm_detok_feed(&g->d[g->chan], id, piece, sizeof piece - 1);
    if (w < 0) { g->stop = MYNAH_SLM_STOP_ERROR; return 0; }
    piece[w] = '\0';

    /* No callback for this channel means the caller wants it discarded —
     * which is the right default for thinking in a speech pipeline. */
    const mynah_slm_token_cb cb[CH_N] = { p->cb, p->cb_think, p->cb_tool };
    void *const cb_ctx[CH_N] = { p->cb_ctx, p->cb_think_ctx, p->cb_tool_ctx };
    if (cb[g->chan] && cb[g->chan](cb_ctx[g->chan], id, piece, (size_t)w) != 0) {
        g->stop = MYNAH_SLM_STOP_CALLBACK;
        return 0;
    }
    g->next = id;

out:
    if (g->step >= p->max_new) { g->stop = MYNAH_SLM_STOP_LENGTH; return 0; }
    return 1;
}

void mynah_slm_gen_finish(mynah_slm_gen *g) {
    if (g->finished) return;
    g->finished = 1;
    if (g->t) mynah_slm_timing_end_decode(g->t);

    const mynah_slm_token_cb cb[CH_N] = { g->p.cb, g->p.cb_think, g->p.cb_tool };
    void *const cb_ctx[CH_N] = { g->p.cb_ctx, g->p.cb_think_ctx, g->p.cb_tool_ctx };
    char piece[512];
    for (int i = 0; i < CH_N; i++) {
        const long w = mynah_slm_detok_finish(&g->d[i], piece, sizeof piece - 1);
        if (w > 0 && cb[i]) { piece[w] = '\0'; cb[i](cb_ctx[i], 0, piece, (size_t)w); }
    }
}

long mynah_slm_generate(mynah_slm_state *st, const mynah_slm_tokenizer *tok,
                        mynah_slm_sampler *sam, const mynah_slm_gen_params *p,
                        mynah_slm_timing *t) {
    if (!st) return -1;
    ref_driver r = { st, &st->own };
    const mynah_slm_gen_driver d = { ref_prefill, ref_step, mynah_slm_batch_max(st), &r };
    return mynah_slm_generate_driver(&d, tok, sam, p, t);
}

long mynah_slm_generate_driver(const mynah_slm_gen_driver *d, const mynah_slm_tokenizer *tok,
                               mynah_slm_sampler *sam, const mynah_slm_gen_params *p,
                               mynah_slm_timing *t) {
    if (!d || !p || p->n_prompt == 0) return -1;

    mynah_slm_gen g;
    if (mynah_slm_gen_start(&g, tok, sam, p, t) != 0) return -1;
    if (mynah_slm_gen_prefill_driver(&g, d, 0) != 1) {
        if (g.stop != MYNAH_SLM_STOP_CANCELLED) return -1;
        mynah_slm_gen_finish(&g);
        return 0;                    /* cancelled during the prompt: nothing generated */
    }

    while (mynah_slm_gen_wants_step(&g)) {
        /* Before every step, whatever channel is open: thinking and tool
         * tokens are never written, so a write failure cannot be the only
         * way to notice the client left. */
        if (mynah_slm_gen_check_cancel(&g)) break;
        float *logits = d->step(d->ctx, mynah_slm_gen_next_token(&g));
        if (!logits) {
            mynah_slm_gen_fail(&g);
            break;
        }
        mynah_slm_gen_accept_logits(&g, logits);
    }
    mynah_slm_gen_finish(&g);
    return g.produced;
}
