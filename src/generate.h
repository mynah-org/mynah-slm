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

    /* The ANSWER channel. Everything a downstream TTS stage should speak, and
     * nothing else. */
    mynah_slm_token_cb cb;
    void              *cb_ctx;

    /* The THINKING channel, separate on purpose.
     *
     * Reasoning must never reach a TTS stage — being read the model's internal
     * monologue out loud is not a cosmetic defect, it is the feature failing.
     * So the split is structural rather than a filter the caller might forget
     * to apply: text between the two markers below goes here, or nowhere when
     * this is NULL. The markers are resolved by NAME at setup
     * (mynah_slm_token_find), never hardcoded, and they are single tokens in
     * Qwen3 — so the boundary is exact and cannot fall mid-string.
     *
     * Set think_open/think_close to -1 to disable the split entirely. */
    mynah_slm_token_cb cb_think;
    void              *cb_think_ctx;
    long               think_open;
    long               think_close;

    /* The TOOL-CALL channel, split for the same reason and by the same
     * mechanism.
     *
     * `<tool_call>` and `</tool_call>` are single tokens in Qwen3, so the JSON
     * a function call is made of never reaches the answer channel and never
     * gets spoken. What arrives here is the bare JSON, markers already
     * consumed — feed it to mynah_slm_tool_calls_parse().
     *
     * Set tool_open/tool_close to -1 to disable the split, which is what a
     * caller that passed no tools wants: then a model that emits the markers
     * anyway is simply talking. */
    mynah_slm_token_cb cb_tool;
    void              *cb_tool_ctx;
    long               tool_open;
    long               tool_close;

    /* "Does anybody still want this?" Polled before EVERY decode step and
     * before every prefill batch; non-zero stops the generation there
     * (stop = CANCELLED) without computing another token.
     *
     * The channel callbacks cannot do this job: a disconnect is only noticed
     * when a write fails, and nothing is written during a prefill, on the
     * thinking channel (discarded), while a tool call accumulates, or at all
     * in a non-streaming request. Without this a client that left keeps the
     * CPU busy to max_tokens — zombie work, holding the model while everyone
     * else waits. NULL = never cancelled (the CLI). */
    int  (*cancel)(void *ctx);
    void  *cancel_ctx;
} mynah_slm_gen_params;

/* Zero the params and DISABLE both channel splits.
 *
 * Use this instead of brace-initializing the struct. A marker field left at 0
 * is not "unset": 0 is a real token id, and the day a model emits it the
 * answer channel would silently switch to another channel and the user would
 * lose text. So the disabled value is -1 and it has to be written by someone. */
void mynah_slm_gen_params_init(mynah_slm_gen_params *p);

/* Runs to completion, or until the cancel hook fires. Returns the number of
 * tokens generated (also when cancelled), or -1 on failure.
 * `t` is filled in as it goes and is safe to print afterwards.
 *
 * A thin driver over mynah_slm_gen below, on the state's own sequence. */
long mynah_slm_generate(mynah_slm_state *st, const mynah_slm_tokenizer *tok,
                        mynah_slm_sampler *sam, const mynah_slm_gen_params *p,
                        mynah_slm_timing *t);

/* ── one generation, one step at a time ────────────────────────────────────
 * Everything generate() keeps between tokens, as a value: the prompt cursor,
 * the token to feed next, the step count, which channel is open, the three
 * detokenizers. With it a caller other than generate() — a scheduler running
 * many of these on one model — can take ONE step of a generation, or one
 * slice of its prompt, and come back later.
 *
 * It is THE token loop: generate() is written on top of it, so the CLI, the
 * serialized server and the scheduler cannot drift apart (rule 2).
 *
 *     gen_start
 *     while (gen_prefill(g, ws, seq, slice) == 0) ;      prompt, in slices
 *     while (gen_wants_step(g)) {
 *         forward(seq, gen_next_token(g)) -> logits      alone or batched
 *         gen_accept_logits(g, logits);                  sample, stop, emit
 *     }
 *     gen_finish                                          flush, end timing
 *
 * The params, tokenizer, sampler and timing are BORROWED and must outlive it;
 * so must the prompt array the params point at. */

typedef enum {
    MYNAH_SLM_STOP_NONE = 0,   /* still running */
    MYNAH_SLM_STOP_EOS,        /* a terminator was sampled */
    MYNAH_SLM_STOP_LENGTH,     /* max_new reached */
    MYNAH_SLM_STOP_CALLBACK,   /* a channel callback asked to stop */
    MYNAH_SLM_STOP_ERROR,      /* the forward pass or the detokenizer failed */
    MYNAH_SLM_STOP_CANCELLED,  /* the cancel hook said nobody wants it */
} mynah_slm_stop;

typedef struct {
    mynah_slm_gen_params p;
    const mynah_slm_tokenizer *tok;
    mynah_slm_sampler *sam;
    mynah_slm_timing  *t;

    size_t   prefilled;    /* prompt tokens already in the cache */
    int      decoding;     /* prefill finished */
    uint32_t next;         /* the token the next decode step feeds */
    uint32_t step;         /* decode steps taken */
    long     produced;     /* generated tokens, a terminator included */
    int      chan;
    mynah_slm_stop stop;
    int      finished;     /* gen_finish ran */

    /* One detokenizer PER CHANNEL. A shared one would carry a half-finished
     * UTF-8 sequence across a marker and complete it on the wrong side. */
    mynah_slm_detok d[3];
} mynah_slm_gen;

/* Arms a generation. Nothing is computed. Returns 0, or -1 on an empty
 * prompt. The caller starts `t` (mynah_slm_timing_start) when it wants the
 * clock to start — at admission, for a server. */
int  mynah_slm_gen_start(mynah_slm_gen *g, const mynah_slm_tokenizer *tok,
                         mynah_slm_sampler *sam, const mynah_slm_gen_params *p,
                         mynah_slm_timing *t);

/* Feeds up to `budget` more prompt tokens (0 = all that remain) into `q`, in
 * batches no wider than the workspace allows. Every prompt token but the last
 * only populates the cache; the last is the first decode step's input.
 * Returns 1 when the prompt is done (and from then on), 0 when more remains,
 * -1 on failure (stop = ERROR) or cancellation (stop = CANCELLED). */
int  mynah_slm_gen_prefill(mynah_slm_gen *g, mynah_slm_state *ws, mynah_slm_seq *q,
                           uint32_t budget);

/* ── who runs the forward pass ─────────────────────────────────────────────
 * The token loop above does not care WHAT computes the logits, only that
 * something fills the cache with a slice of prompt and returns one logits
 * row per decode step. gen_prefill and generate() drive arch_qwen3.c through
 * this; src/forward_backend.c supplies one that drives the backend vtable
 * (`--device`). One loop, several forward passes — never two loops (rule 2).
 *
 *   prefill  feed n <= batch_max prompt tokens into the cache, no logits;
 *            0 or -1
 *   step     feed one token, return its logits row (HOST memory, valid until
 *            the next call; the sampler modifies it in place), NULL on
 *            failure */
typedef struct {
    int    (*prefill)(void *ctx, const uint32_t *tokens, uint32_t n);
    float *(*step)(void *ctx, uint32_t token);
    uint32_t batch_max;
    void    *ctx;
} mynah_slm_gen_driver;

/* gen_prefill and generate() with the forward pass supplied by `d`.
 * mynah_slm_gen_prefill(g, ws, q, b) is this with arch_qwen3.c on (ws, q). */
int  mynah_slm_gen_prefill_driver(mynah_slm_gen *g, const mynah_slm_gen_driver *d,
                                  uint32_t budget);
long mynah_slm_generate_driver(const mynah_slm_gen_driver *d, const mynah_slm_tokenizer *tok,
                               mynah_slm_sampler *sam, const mynah_slm_gen_params *p,
                               mynah_slm_timing *t);

/* Prompt tokens still to prefill. */
size_t mynah_slm_gen_prefill_left(const mynah_slm_gen *g);

/* Does this generation want a decode step: prefill done, not stopped, and
 * fewer than max_new steps taken. */
int  mynah_slm_gen_wants_step(const mynah_slm_gen *g);

/* The token the next decode step must feed to the forward pass. */
uint32_t mynah_slm_gen_next_token(const mynah_slm_gen *g);

/* Consumes the logits of that step (modified in place by the sampler):
 * sample, stop test, channel split, detokenize, callback. Returns 1 when the
 * generation wants another step, 0 when it has stopped. */
int  mynah_slm_gen_accept_logits(mynah_slm_gen *g, float *logits);

/* Polls the cancel hook (if any). Non-zero = cancelled (stop = CANCELLED),
 * and every later call agrees. Drivers call it before each forward pass;
 * gen_prefill calls it before each batch on its own. */
int  mynah_slm_gen_check_cancel(mynah_slm_gen *g);

/* Records a failed forward pass for this generation (stop = ERROR). */
void mynah_slm_gen_fail(mynah_slm_gen *g);

/* Closes the decode clock and flushes every channel's held bytes through its
 * callback. Idempotent. */
void mynah_slm_gen_finish(mynah_slm_gen *g);

#endif /* MYNAH_SLM_GENERATE_H */
