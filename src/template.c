/* template.c — Qwen3 ChatML. See template.h.
 * SPDX-License-Identifier: MIT */
#include "template.h"

#include <stdio.h>
#include <string.h>

static const char *role_name(mynah_slm_role r) {
    switch (r) {
        case MYNAH_SLM_ROLE_SYSTEM:    return "system";
        case MYNAH_SLM_ROLE_USER:      return "user";
        case MYNAH_SLM_ROLE_ASSISTANT: return "assistant";
        case MYNAH_SLM_ROLE_TOOL:      return "tool";
    }
    return "user";
}

/* Append to a snprintf-style cursor that keeps counting past the end, so one
 * pass with out=NULL gives the required size. */
typedef struct { char *buf; size_t max, used; } cursor;

static void put(cursor *c, const char *s) {
    const size_t n = strlen(s);
    if (c->buf && c->used < c->max) {
        const size_t room = c->max - c->used - 1;
        memcpy(c->buf + c->used, s, n < room ? n : room);
    }
    c->used += n;
}

long mynah_slm_render_chat(const mynah_slm_message *msgs, size_t n,
                           mynah_slm_think think, char *out, size_t max) {
    if (!msgs && n) return -1;

    cursor c = { .buf = out, .max = max, .used = 0 };

    for (size_t i = 0; i < n; i++) {
        if (!msgs[i].content) return -1;
        put(&c, "<|im_start|>");
        put(&c, role_name(msgs[i].role));
        put(&c, "\n");
        put(&c, msgs[i].content);
        put(&c, "<|im_end|>\n");
    }

    put(&c, "<|im_start|>assistant\n");

    /* Qwen3 suppresses reasoning by PRE-FILLING an empty think block rather
     * than by instructing the model not to think — that is what its own
     * template emits for enable_thinking=false, and an instruction would be
     * something the model could decline. */
    if (think == MYNAH_SLM_THINK_OFF) put(&c, "<think>\n\n</think>\n\n");

    if (c.buf && c.max) c.buf[c.used < c.max ? c.used : c.max - 1] = '\0';
    return (long)c.used;
}
