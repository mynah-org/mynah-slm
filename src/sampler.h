/* sampler.h — turning logits into a token.
 *
 * Deterministic given a seed, and that is a contract rather than a nicety: an
 * A/B test where the two sides cannot be made to agree is not an A/B test.
 *
 * SPDX-License-Identifier: MIT */
#ifndef MYNAH_SLM_SAMPLER_H
#define MYNAH_SLM_SAMPLER_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    float    temp;          /* <= 0 means greedy, and skips every other filter */
    uint32_t top_k;         /* 0 = off */
    float    top_p;         /* 1.0 = off */
    float    min_p;         /* 0.0 = off */

    /* Penalties applied over the last `penalty_window` tokens. */
    float    repeat_penalty;    /* 1.0 = off */
    float    freq_penalty;
    float    presence_penalty;
    uint32_t penalty_window;

    uint64_t seed;
} mynah_slm_sampler_params;

/* Qwen3's own generation_config.json: temp 0.6, top_k 20, top_p 0.95. Defaults
 * come from the model, not from our taste. */
void mynah_slm_sampler_defaults(mynah_slm_sampler_params *p);

typedef struct mynah_slm_sampler mynah_slm_sampler;

mynah_slm_sampler *mynah_slm_sampler_new(const mynah_slm_sampler_params *p,
                                         uint32_t vocab_size);
void mynah_slm_sampler_free(mynah_slm_sampler *s);

/* Record a token so the penalties can see it. Prompt tokens count too — a
 * model that repeats the prompt back is exactly what they exist to discourage. */
void mynah_slm_sampler_accept(mynah_slm_sampler *s, uint32_t token);

/* Pick one. `logits` is modified in place. */
uint32_t mynah_slm_sample(mynah_slm_sampler *s, float *logits);

#endif /* MYNAH_SLM_SAMPLER_H */
