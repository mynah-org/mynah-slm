/* template.c — Qwen3 ChatML. See template.h.
 * SPDX-License-Identifier: MIT */
#include "template.h"

#include <stdio.h>
#include <string.h>

/* The tool preamble, word for word from the model's own chat template. It is
 * not advice we are giving the model: it is the exact string it was trained
 * against, so every character of it is load-bearing. */
static const char TOOLS_HEAD[] =
    "# Tools\n\nYou may call one or more functions to assist with the user "
    "query.\n\nYou are provided with function signatures within "
    "<tools></tools> XML tags:\n<tools>";
static const char TOOLS_TAIL[] =
    "\n</tools>\n\nFor each function call, return a json object with function "
    "name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n"
    "{\"name\": <function-name>, \"arguments\": <args-json-object>}\n"
    "</tool_call>";

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

/* One tool, serialized the way the template's `tool | tojson` does it —
 * OpenAI's nesting, a space after every colon and comma, `parameters` passed
 * through untouched. The spacing is not cosmetic: it is what the model saw
 * during training, and tests/test_tools.c pins it against HF byte for byte. */
static void put_tool(cursor *c, const mynah_slm_tool *t) {
    put(c, "{\"type\": \"function\", \"function\": {\"name\": \"");
    put(c, t->name);
    put(c, "\"");
    if (t->description) {
        put(c, ", \"description\": \"");
        put(c, t->description);
        put(c, "\"");
    }
    if (t->parameters) {
        put(c, ", \"parameters\": ");
        put(c, t->parameters);
    }
    put(c, "}}");
}

static void put_turn(cursor *c, mynah_slm_role role, const char *content) {
    put(c, "<|im_start|>");
    put(c, role_name(role));
    put(c, "\n");
    put(c, content);
    put(c, "<|im_end|>\n");
}

long mynah_slm_render_chat_tools(const mynah_slm_message *msgs, size_t n,
                                 const mynah_slm_tool *tools, size_t n_tools,
                                 mynah_slm_think think, char *out, size_t max) {
    if (!msgs && n) return -1;
    if (n_tools && !tools) return -1;

    cursor c = { .buf = out, .max = max, .used = 0 };

    /* A leading system message is consumed by the head, whether or not tools
     * are present — the template emits it there and skips it in the loop. */
    const int head_system = (n > 0 && msgs[0].role == MYNAH_SLM_ROLE_SYSTEM);
    if (head_system && !msgs[0].content) return -1;

    if (n_tools) {
        put(&c, "<|im_start|>system\n");
        if (head_system) { put(&c, msgs[0].content); put(&c, "\n\n"); }
        put(&c, TOOLS_HEAD);
        for (size_t i = 0; i < n_tools; i++) {
            if (!tools[i].name) return -1;
            put(&c, "\n");
            put_tool(&c, &tools[i]);
        }
        put(&c, TOOLS_TAIL);
        put(&c, "<|im_end|>\n");
    } else if (head_system) {
        put_turn(&c, MYNAH_SLM_ROLE_SYSTEM, msgs[0].content);
    }

    for (size_t i = head_system ? 1 : 0; i < n; i++) {
        const mynah_slm_message *m = &msgs[i];
        const char *content = m->content;
        if (!content && !(m->role == MYNAH_SLM_ROLE_ASSISTANT && m->n_tool_calls))
            return -1;
        if (!content) content = "";

        switch (m->role) {
            case MYNAH_SLM_ROLE_SYSTEM:
            case MYNAH_SLM_ROLE_USER:
                put_turn(&c, m->role, content);
                break;

            case MYNAH_SLM_ROLE_ASSISTANT:
                put(&c, "<|im_start|>assistant\n");
                put(&c, content);
                for (size_t k = 0; k < m->n_tool_calls; k++) {
                    if (!m->tool_calls) return -1;
                    /* A newline separates a call from whatever precedes it —
                     * text, or the previous call. Not before the first one
                     * when the turn had no text, or the turn would open with
                     * a blank line the model never saw in training. */
                    if (k > 0 || content[0]) put(&c, "\n");
                    put(&c, "<tool_call>\n{\"name\": \"");
                    put(&c, m->tool_calls[k].name);
                    put(&c, "\", \"arguments\": ");
                    put(&c, m->tool_calls[k].arguments ? m->tool_calls[k].arguments : "{}");
                    put(&c, "}\n</tool_call>");
                }
                put(&c, "<|im_end|>\n");
                break;

            case MYNAH_SLM_ROLE_TOOL: {
                /* Tool results are not a role of their own on the wire: they
                 * are <tool_response> blocks inside ONE user turn, however
                 * many of them there are. Splitting them into a turn each is
                 * the classic way to make parallel tool calls come back wrong.
                 */
                const int opens = (i == 0) || (msgs[i - 1].role != MYNAH_SLM_ROLE_TOOL);
                const int closes = (i + 1 == n) || (msgs[i + 1].role != MYNAH_SLM_ROLE_TOOL);
                if (opens) put(&c, "<|im_start|>user");
                put(&c, "\n<tool_response>\n");
                put(&c, content);
                put(&c, "\n</tool_response>");
                if (closes) put(&c, "<|im_end|>\n");
                break;
            }
        }
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

long mynah_slm_render_chat(const mynah_slm_message *msgs, size_t n,
                           mynah_slm_think think, char *out, size_t max) {
    return mynah_slm_render_chat_tools(msgs, n, NULL, 0, think, out, max);
}
