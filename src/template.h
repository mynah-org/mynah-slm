/* template.h — chat turns, rendered to text the tokenizer can eat.
 *
 * The template is written out here rather than interpreted from the Jinja
 * string in the GGUF. Shipping a Jinja engine to render six lines of ChatML
 * would be the largest dependency in the project, for the least of its
 * features. What we do instead is pin the output byte for byte against HF's
 * apply_chat_template in a test.
 *
 * SPDX-License-Identifier: MIT */
#ifndef MYNAH_SLM_TEMPLATE_H
#define MYNAH_SLM_TEMPLATE_H

#include <stddef.h>

typedef enum {
    MYNAH_SLM_ROLE_SYSTEM = 0,
    MYNAH_SLM_ROLE_USER,
    MYNAH_SLM_ROLE_ASSISTANT,
    MYNAH_SLM_ROLE_TOOL,
} mynah_slm_role;

/* off  — thinking suppressed. Qwen3 does this with an empty <think> block,
 *        which is what its own template emits for enable_thinking=false.
 * low  — allowed but budget-capped by the caller, tokens hidden.
 * on   — emitted and surfaced on a separate channel.
 *
 * The channel separation is not cosmetic: a TTS stage downstream must never
 * speak the model's reasoning out loud. */
typedef enum {
    MYNAH_SLM_THINK_OFF = 0,
    MYNAH_SLM_THINK_LOW,
    MYNAH_SLM_THINK_ON,
} mynah_slm_think;

typedef struct {
    mynah_slm_role  role;
    const char     *content;
} mynah_slm_message;

/* Render `n` messages plus the assistant's opening. Returns the number of
 * bytes the full rendering needs (excluding the NUL), like snprintf, so a
 * caller can size a buffer with out=NULL. Negative on error. */
long mynah_slm_render_chat(const mynah_slm_message *msgs, size_t n,
                           mynah_slm_think think, char *out, size_t max);

#endif /* MYNAH_SLM_TEMPLATE_H */
