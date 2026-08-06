/* generate.c — see generate.h.
 * SPDX-License-Identifier: MIT */
#include "generate.h"

#include <stdlib.h>
#include <string.h>

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
    for (size_t i = 0; i + 1 < p->n_prompt; i++) {
        if (mynah_slm_forward(st, p->prompt[i], NULL) != 0) return -1;
        mynah_slm_sampler_accept(sam, p->prompt[i]);
    }
    mynah_slm_timing_end_prefill(t, (uint32_t)p->n_prompt);

    uint32_t next = p->prompt[p->n_prompt - 1];
    mynah_slm_sampler_accept(sam, next);

    mynah_slm_detok d;
    mynah_slm_detok_init(&d, tok);

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

        const long w = mynah_slm_detok_feed(&d, id, piece, sizeof piece - 1);
        if (w < 0) break;
        piece[w] = '\0';

        if (p->cb && p->cb(p->cb_ctx, id, piece, (size_t)w) != 0) break;
        next = id;
    }

    mynah_slm_timing_end_decode(t);

    const long w = mynah_slm_detok_finish(&d, piece, sizeof piece - 1);
    if (w > 0 && p->cb) { piece[w] = '\0'; p->cb(p->cb_ctx, 0, piece, (size_t)w); }

    return produced;
}
