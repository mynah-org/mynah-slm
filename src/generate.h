/* generate.h — prefill, then decode, with the token callback that makes
 * streaming the primary path rather than an add-on.
 *
 * SPDX-License-Identifier: MIT */
#ifndef MYNAH_SLM_GENERATE_H
#define MYNAH_SLM_GENERATE_H

#include "arch_qwen3.h"
#include "sampler.h"
#include "timing.h"
#include "tokenizer.h"

/* Called once per generated token, with the decoded text for it. `text` may be
 * empty when the token completed no codepoint — that is normal mid-emoji, not
 * an error. Return non-zero to stop generation. */
typedef int (*mynah_slm_token_cb)(void *ctx, uint32_t id, const char *text, size_t len);

typedef struct {
    const uint32_t *prompt;
    size_t          n_prompt;

    uint32_t        max_new;
    const uint32_t *eos;          /* stop on ANY of these */
    size_t          n_eos;

    mynah_slm_token_cb cb;
    void              *cb_ctx;
} mynah_slm_gen_params;

/* Runs to completion. Returns the number of tokens generated, or -1.
 * `t` is filled in as it goes and is safe to print afterwards. */
long mynah_slm_generate(mynah_slm_state *st, const mynah_slm_tokenizer *tok,
                        mynah_slm_sampler *sam, const mynah_slm_gen_params *p,
                        mynah_slm_timing *t);

#endif /* MYNAH_SLM_GENERATE_H */
