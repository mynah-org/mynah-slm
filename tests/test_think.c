/* test_think.c — reasoning must never reach the answer channel.
 *
 * This is a safety property, not a formatting preference: the answer channel
 * is what gets piped into mynah-tts, and a TTS stage reading the model's
 * internal monologue out loud is the feature failing, not a cosmetic defect.
 *
 * Kept deliberately short (a handful of tokens) so it stays in `make test`
 * while the engine is still single-threaded and slow.
 *
 * SPDX-License-Identifier: MIT */
#include "arch_qwen3.h"
#include "generate.h"
#include "model.h"
#include "mynah_slm.h"
#include "sampler.h"
#include "template.h"
#include "timing.h"
#include "tokenizer.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures;

typedef struct { char buf[8192]; size_t used; } sink;

static int sink_cb(void *ctx, uint32_t id, const char *text, size_t len) {
    (void)id;
    sink *s = ctx;
    if (s->used + len < sizeof s->buf) {
        memcpy(s->buf + s->used, text, len);
        s->used += len;
        s->buf[s->used] = '\0';
    }
    return 0;
}

static void check(const char *what, int ok, const char *detail) {
    if (ok) printf("ok   %s\n", what);
    else { printf("FAIL %s  <- %s\n", what, detail ? detail : ""); failures++; }
}

int main(int argc, char **argv) {
    const char *model_path = argc > 1 ? argv[1] : "models-local/Qwen3-0.6B-Q4_K_M.gguf";

    char err[256];
    mynah_slm_model_t *m = mynah_slm_load(model_path, err, sizeof err);
    if (!m) {
        if (strstr(err, "No such file") || strstr(err, "cannot open")) {
            printf("SKIP no checkpoint at %s (scripts/use_model.sh)\n", model_path);
            return 77;
        }
        printf("FAIL load: %s\n", err);
        return 1;
    }

    mynah_slm_tokenizer *tok = mynah_slm_tokenizer_load(m->gguf, err, sizeof err);
    if (!tok) { printf("FAIL tokenizer: %s\n", err); return 1; }

    const long open_id  = mynah_slm_token_find(tok, "<think>");
    const long close_id = mynah_slm_token_find(tok, "</think>");
    check("<think> resolves by name", open_id >= 0, "not in this vocabulary");
    check("</think> resolves by name", close_id >= 0, "not in this vocabulary");
    if (open_id < 0 || close_id < 0) return 1;

    /* A prompt that reliably makes a reasoning model think first. */
    mynah_slm_message msg = { MYNAH_SLM_ROLE_USER, "Quanto fa 17+25?" };
    const long need = mynah_slm_render_chat(&msg, 1, MYNAH_SLM_THINK_ON, NULL, 0);
    char *text = malloc((size_t)need + 1);
    mynah_slm_render_chat(&msg, 1, MYNAH_SLM_THINK_ON, text, (size_t)need + 1);

    /* With thinking ON the template must NOT pre-fill an empty think block —
     * that is what suppresses reasoning, and it would make this test vacuous. */
    check("think=on leaves the block open", strstr(text, "</think>") == NULL,
          "the template pre-filled a closed think block");

    const long n_prompt = mynah_slm_tokenize(tok, text, 1, NULL, 0);
    uint32_t *ids = malloc((size_t)n_prompt * sizeof *ids);
    mynah_slm_tokenize(tok, text, 1, ids, (size_t)n_prompt);

    mynah_slm_state st;
    if (mynah_slm_state_init(&st, m, (uint32_t)n_prompt + 40, err, sizeof err) != 0) {
        printf("FAIL state: %s\n", err);
        return 1;
    }

    mynah_slm_sampler_params sp;
    mynah_slm_sampler_defaults(&sp);
    sp.temp = 0.0f;                        /* greedy: reproducible */
    mynah_slm_sampler *sam = mynah_slm_sampler_new(&sp, mynah_slm_vocab_size(m));

    const uint32_t eos[] = { mynah_slm_tokenizer_eos(tok), 151643u };
    sink answer = {0}, think = {0};

    mynah_slm_timing tm;
    mynah_slm_timing_reset(&tm);
    mynah_slm_timing_start(&tm);
    mynah_slm_timing_end_load(&tm);

    mynah_slm_gen_params gp = {
        .prompt = ids, .n_prompt = (size_t)n_prompt,
        .max_new = 24,                     /* enough to be well inside the block */
        .eos = eos, .n_eos = 2,
        .cb = sink_cb, .cb_ctx = &answer,
        .cb_think = sink_cb, .cb_think_ctx = &think,
        .think_open = open_id, .think_close = close_id,
    };
    const long n = mynah_slm_generate(&st, tok, sam, &gp, &tm);
    check("generation produced tokens", n > 0, "nothing came out");

    printf("     think channel:  %zu bytes\n", think.used);
    printf("     answer channel: %zu bytes\n", answer.used);

    /* The point of the whole thing. */
    check("reasoning reached the think channel", think.used > 0,
          "nothing was routed to thinking — the split may not be active");
    check("answer channel stayed empty while thinking", answer.used == 0,
          answer.buf);

    /* The markers are structure and must appear on neither channel, or a
     * client would render a literal "<think>" in the output. */
    check("no marker leaked into the answer",
          strstr(answer.buf, "<think>") == NULL && strstr(answer.buf, "</think>") == NULL,
          answer.buf);
    check("no marker leaked into the thinking",
          strstr(think.buf, "<think>") == NULL && strstr(think.buf, "</think>") == NULL,
          think.buf);

    /* And with no thinking callback the text must be DISCARDED, not spilled
     * onto the answer channel. That is the default a speech pipeline gets. */
    mynah_slm_state_reset(&st);
    mynah_slm_sampler_free(sam);
    sam = mynah_slm_sampler_new(&sp, mynah_slm_vocab_size(m));
    sink answer2 = {0};
    gp.cb_ctx = &answer2;
    gp.cb_think = NULL;
    gp.cb_think_ctx = NULL;
    mynah_slm_generate(&st, tok, sam, &gp, &tm);
    check("thinking is discarded when no channel is attached", answer2.used == 0,
          answer2.buf);

    free(text); free(ids);
    mynah_slm_sampler_free(sam);
    mynah_slm_state_free(&st);
    mynah_slm_tokenizer_free(tok);
    mynah_slm_free(m);

    printf("\n%s (%d failure%s)\n", failures ? "FAILED" : "PASS",
           failures, failures == 1 ? "" : "s");
    return failures ? 1 : 0;
}
