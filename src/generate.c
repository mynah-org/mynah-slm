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

long mynah_slm_generate(mynah_slm_state *st, const mynah_slm_tokenizer *tok,
                        mynah_slm_sampler *sam, const mynah_slm_gen_params *p,
                        mynah_slm_timing *t) {
    if (!st || !p || p->n_prompt == 0) return -1;

    /* ── prefill ──────────────────────────────────────────────────────────
     * Every prompt token but the last only needs to populate the KV cache, so
     * its logits are never materialized. On a 151936-row tied head that skips
     * one 151936x1024 matvec per prompt token, which is most of the prefill
     * cost — and it is exactly the kind of saving that has to be free of any
     * behaviour change, since the last token's logits are computed the same
     * way either way. */
    const size_t n_pre = p->n_prompt - 1;
    const uint32_t bmax = mynah_slm_batch_max(st);
    for (size_t i = 0; i < n_pre; ) {
        size_t take = n_pre - i;
        if (take > bmax) take = bmax;

        /* One call per batch instead of one per token: the weights are read
         * once for the whole group, which is the difference between prefill
         * being memory-bound and being compute-bound. A batch of one falls
         * back to the single-token path inside forward_batch. */
        if (mynah_slm_forward_batch(st, p->prompt + i, (uint32_t)take, NULL) != 0)
            return -1;
        for (size_t k = 0; k < take; k++) mynah_slm_sampler_accept(sam, p->prompt[i + k]);
        i += take;
    }
    mynah_slm_timing_end_prefill(t, (uint32_t)p->n_prompt);

    uint32_t next = p->prompt[p->n_prompt - 1];
    mynah_slm_sampler_accept(sam, next);

    /* One detokenizer PER CHANNEL. A shared one would carry a half-finished
     * UTF-8 sequence across a marker and complete it on the wrong side. The
     * markers are whole tokens, so no real sequence ever straddles them. */
    enum { CH_ANSWER = 0, CH_THINK, CH_TOOL, CH_N };
    mynah_slm_detok d[CH_N];
    for (int i = 0; i < CH_N; i++) mynah_slm_detok_init(&d[i], tok);

    const mynah_slm_token_cb cb[CH_N] = { p->cb, p->cb_think, p->cb_tool };
    void *const cb_ctx[CH_N] = { p->cb_ctx, p->cb_think_ctx, p->cb_tool_ctx };

    const int split_think = (p->think_open >= 0 && p->think_close >= 0);
    const int split_tool  = (p->tool_open  >= 0 && p->tool_close  >= 0);
    int chan = CH_ANSWER;

    char piece[512];
    long produced = 0;

    for (uint32_t step = 0; step < p->max_new; step++) {
        if (mynah_slm_forward(st, next, st->logits) != 0) break;

        const uint32_t id = mynah_slm_sample(sam, st->logits);
        mynah_slm_sampler_accept(sam, id);

        /* Stop BEFORE emitting: a terminator is a control token, and printing
         * "<|im_end|>" at the end of every answer is a bug users see. */
        if (is_eos(p, id)) { produced++; mynah_slm_timing_token(t); break; }

        mynah_slm_timing_token(t);
        produced++;

        /* The markers are structure, not content: they open and close a
         * channel and are emitted on neither. */
        if (split_think && id == (uint32_t)p->think_open)  { chan = CH_THINK;  next = id; continue; }
        if (split_think && id == (uint32_t)p->think_close) { chan = CH_ANSWER; next = id; continue; }
        if (split_tool  && id == (uint32_t)p->tool_open)   { chan = CH_TOOL;   next = id; continue; }
        if (split_tool  && id == (uint32_t)p->tool_close)  { chan = CH_ANSWER; next = id; continue; }

        const long w = mynah_slm_detok_feed(&d[chan], id, piece, sizeof piece - 1);
        if (w < 0) break;
        piece[w] = '\0';

        /* No callback for this channel means the caller wants it discarded —
         * which is the right default for thinking in a speech pipeline. */
        if (cb[chan] && cb[chan](cb_ctx[chan], id, piece, (size_t)w) != 0) break;
        next = id;
    }

    mynah_slm_timing_end_decode(t);

    for (int i = 0; i < CH_N; i++) {
        const long w = mynah_slm_detok_finish(&d[i], piece, sizeof piece - 1);
        if (w > 0 && cb[i]) { piece[w] = '\0'; cb[i](cb_ctx[i], 0, piece, (size_t)w); }
    }

    return produced;
}
