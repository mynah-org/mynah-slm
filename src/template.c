/* template.c — Qwen3 ChatML. See template.h.
 * SPDX-License-Identifier: MIT */
#include "template.h"

#include "json.h"

#include <stdio.h>
#include <string.h>

/* The preambles, word for word from each model's own chat template. They are
 * not advice we are giving the model: they are the exact strings it was
 * trained against, so every character is load-bearing. */
static const mynah_slm_chat_family CHATML = {
    .name       = "chatml",
    .role_open  = "<|im_start|>",
    .role_close = "\n",
    .turn_end   = "<|im_end|>\n",
    .tools_prefix =
        "# Tools\n\nYou may call one or more functions to assist with the user "
        "query.\n\nYou are provided with function signatures within "
        "<tools></tools> XML tags:\n<tools>",
    .tools_suffix =
        "\n</tools>\n\nFor each function call, return a json object with function "
        "name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n"
        "{\"name\": <function-name>, \"arguments\": <args-json-object>}\n"
        "</tool_call>",
    .default_system = NULL,
    .think_prefill  = 1,
    .call_open      = "<tool_call>",
    .call_close     = "</tool_call>",
};

static const mynah_slm_chat_family GRANITE = {
    .name       = "granite",
    .role_open  = "<|start_of_role|>",
    .role_close = "<|end_of_role|>",
    .turn_end   = "<|end_of_text|>\n",
    .tools_prefix =
        "You are a helpful assistant with access to the following tools. You may "
        "call one or more tools to assist with the user query.\n\nYou are "
        "provided with function signatures within <tools></tools> XML tags:\n"
        "<tools>",
    .tools_suffix =
        "\n</tools>\n\nFor each tool call, return a json object with function "
        "name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n"
        "{\"name\": <function-name>, \"arguments\": <args-json-object>}\n"
        "</tool_call>. If a tool does not exist in the provided list of tools, "
        "notify the user that you do not have the ability to fulfill the request.",
    /* Granite always opens with a system turn, cannned when the caller gave
     * neither one nor tools. Leaving it out is not "no system prompt", it is a
     * prompt shape the model never saw. */
    .default_system =
        "You are a helpful assistant. Please ensure responses are professional, "
        "accurate, and safe.",
    .think_prefill = 0,
    .call_open     = "<tool_call>",
    .call_close    = "</tool_call>",
};

/* LFM2 shares ChatML's markers to the byte and disagrees about everything
 * around them: a BOS at the top, the schemas inside the system turn, Pythonic
 * calls on the way out, and reasoning opened by the template rather than
 * suppressed by it. See docs/lfm2-arch.md and reference/lfm2.5-2.6b/. */
static const mynah_slm_chat_family LFM2 = {
    .name       = "lfm2",
    .bos        = "<|startoftext|>",
    .role_open  = "<|im_start|>",
    .role_close = "\n",
    .turn_end   = "<|im_end|>\n",
    /* Not an XML block and not a paragraph of instructions: the template
     * appends exactly this to the system prompt, and the model was trained
     * against that literal shape. */
    .tools_prefix = "List of tools: [",
    .tools_suffix = "]",
    .default_system = NULL,
    .think_prefill  = 0,
    .think_open     = "<think>",
    .tool_style     = MYNAH_SLM_TOOLS_PYTHONIC,
    .call_open      = "<|tool_call_start|>",
    .call_close     = "<|tool_call_end|>",
};

const mynah_slm_chat_family *mynah_slm_chat_family_for(const char *arch) {
    if (arch && strcmp(arch, "granite") == 0) return &GRANITE;
    if (arch && strcmp(arch, "lfm2")    == 0) return &LFM2;
    return &CHATML;
}

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

/* One JSON value as a PYTHON literal, for LFM2's call expressions.
 *
 * Not a JSON re-emit with different brackets: strings take single quotes with
 * Python's escapes, and booleans are True/False, capitalised. Objects and
 * lists are the one place the template does fall back to JSON — `arg_value |
 * tojson` — so they pass through verbatim, and `true` inside a nested object
 * legitimately stays lowercase. Copying that inconsistency is the point; it is
 * what the model was trained against. */
static void put_py_value(cursor *c, const json_val *v) {
    switch (v->kind) {
        case JSON_STRING: {
            put(c, "'");
            /* The raw span minus its quotes. JSON escapes that Python shares
             * (\n, \r, \\) are already in the right form; a single quote is
             * the one character JSON leaves bare and Python cannot. */
            for (const char *p = v->start + 1; p + 1 < v->end; p++) {
                if (*p == '\'') put(c, "\\'");
                else { const char one[2] = { *p, 0 }; put(c, one); }
            }
            put(c, "'");
            break;
        }
        case JSON_BOOL:
            put(c, v->boolean ? "True" : "False");
            break;
        case JSON_NULL:
            put(c, "None");
            break;
        default: {
            /* Numbers, objects and arrays: the source span, unchanged. */
            char tmp[512];
            const size_t n = (size_t)(v->end - v->start);
            if (n < sizeof tmp) {
                memcpy(tmp, v->start, n);
                tmp[n] = '\0';
                put(c, tmp);
            }
            break;
        }
    }
}

/* `name(arg=value, arg2=value2)` from a name and a JSON arguments object. An
 * unparseable or non-object arguments string yields a bare `name()` rather
 * than a broken expression: the model can recover from a call with no
 * arguments, and cannot from a syntax error. */
static void put_py_call(cursor *c, const mynah_slm_tool_call *tc) {
    put(c, tc->name);
    put(c, "(");

    json_val args;
    if (tc->arguments &&
        json_parse(tc->arguments, tc->arguments + strlen(tc->arguments), &args) == 0 &&
        args.kind == JSON_OBJECT) {
        for (size_t i = 0;; i++) {
            json_val k, v;
            if (json_object_at(&args, i, &k, &v) != 0) break;
            if (i) put(c, ", ");
            char key[128];
            const long kn = json_string_copy(&k, key, sizeof key);
            if (kn < 0) break;
            put(c, key);
            put(c, "=");
            put_py_value(c, &v);
        }
    }
    put(c, ")");
}

static void put_open(cursor *c, const mynah_slm_chat_family *f, const char *role) {
    put(c, f->role_open);
    put(c, role);
    put(c, f->role_close);
}

static void put_turn(cursor *c, const mynah_slm_chat_family *f,
                     mynah_slm_role role, const char *content) {
    put_open(c, f, role_name(role));
    put(c, content);
    put(c, f->turn_end);
}

long mynah_slm_render_chat_tools(const mynah_slm_chat_family *f,
                                 const mynah_slm_message *msgs, size_t n,
                                 const mynah_slm_tool *tools, size_t n_tools,
                                 mynah_slm_think think, char *out, size_t max) {
    if (!msgs && n) return -1;
    if (n_tools && !tools) return -1;
    if (!f) f = &CHATML;

    cursor c = { .buf = out, .max = max, .used = 0 };
    const int pythonic = (f->tool_style == MYNAH_SLM_TOOLS_PYTHONIC);

    if (f->bos) put(&c, f->bos);

    /* A leading system message is consumed by the head, whether or not tools
     * are present — the template emits it there and skips it in the loop. */
    const int head_system = (n > 0 && msgs[0].role == MYNAH_SLM_ROLE_SYSTEM);
    if (head_system && !msgs[0].content) return -1;

    if (n_tools) {
        put_open(&c, f, "system");
        /* One newline for LFM2, a blank line for the XML families: each is
         * what its own template puts between the system text and the tools. */
        if (head_system) { put(&c, msgs[0].content); put(&c, pythonic ? "\n" : "\n\n"); }
        put(&c, f->tools_prefix);
        for (size_t i = 0; i < n_tools; i++) {
            if (!tools[i].name) return -1;
            put(&c, pythonic ? (i ? ", " : "") : "\n");
            put_tool(&c, &tools[i]);
        }
        put(&c, f->tools_suffix);
        put(&c, f->turn_end);
    } else if (head_system) {
        put_turn(&c, f, MYNAH_SLM_ROLE_SYSTEM, msgs[0].content);
    } else if (f->default_system) {
        put_turn(&c, f, MYNAH_SLM_ROLE_SYSTEM, f->default_system);
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
                put_turn(&c, f, m->role, content);
                break;

            case MYNAH_SLM_ROLE_ASSISTANT:
                put_open(&c, f, "assistant");
                put(&c, content);
                if (pythonic && m->n_tool_calls) {
                    /* All the calls live inside ONE bracketed list, which is
                     * how the template renders parallel calls and how the
                     * model emits them. One block per call would be a shape it
                     * never saw. */
                    if (!m->tool_calls) return -1;
                    put(&c, "<|tool_call_start|>[");
                    for (size_t k = 0; k < m->n_tool_calls; k++) {
                        if (k) put(&c, ", ");
                        put_py_call(&c, &m->tool_calls[k]);
                    }
                    put(&c, "]<|tool_call_end|>");
                    put(&c, f->turn_end);
                    break;
                }
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
                put(&c, f->turn_end);
                break;

            case MYNAH_SLM_ROLE_TOOL: {
                /* LFM2 does have a `tool` role on the wire: its template gives
                 * every non-assistant message the same `<|im_start|>role\n...`
                 * treatment, so a result is its own turn and there is no
                 * <tool_response> wrapper to speak of. */
                if (pythonic) { put_turn(&c, f, MYNAH_SLM_ROLE_TOOL, content); break; }

                /* Everywhere else tool results are not a role of their own:
                 * they are <tool_response> blocks inside ONE user turn, however
                 * many of them there are. Splitting them into a turn each is
                 * the classic way to make parallel tool calls come back wrong.
                 */
                const int opens = (i == 0) || (msgs[i - 1].role != MYNAH_SLM_ROLE_TOOL);
                const int closes = (i + 1 == n) || (msgs[i + 1].role != MYNAH_SLM_ROLE_TOOL);
                /* ChatML opens the turn WITHOUT its trailing newline here — the
                 * one before <tool_response> is the template's. Granite's
                 * marker carries its own terminator, so both are just
                 * role_open + "user" + role_close minus ChatML's newline. */
                if (opens) {
                    put(&c, f->role_open);
                    put(&c, "user");
                    if (f->role_close[0] != '\n') put(&c, f->role_close);
                }
                put(&c, "\n<tool_response>\n");
                put(&c, content);
                put(&c, "\n</tool_response>");
                if (closes) put(&c, f->turn_end);
                break;
            }
        }
    }

    put_open(&c, f, "assistant");

    /* Qwen3 suppresses reasoning by PRE-FILLING an empty think block rather
     * than by instructing the model not to think — that is what its own
     * template emits for enable_thinking=false, and an instruction would be
     * something the model could decline. Families without the mechanism get
     * nothing rather than an invented equivalent. */
    if (f->think_prefill && think == MYNAH_SLM_THINK_OFF)
        put(&c, "<think>\n\n</think>\n\n");

    /* LFM2 is the mirror image: its own template ends with
     * `<|im_start|>assistant\n<think>`, unconditionally — reasoning is the
     * default and the opening tag is part of the PROMPT, not something the
     * model decides to emit. So "off" here means withholding that tag, which
     * is the only lever the template offers. There is no enable_thinking flag
     * to set (docs/lfm2-arch.md). */
    if (f->think_open && think != MYNAH_SLM_THINK_OFF)
        put(&c, f->think_open);

    if (c.buf && c.max) c.buf[c.used < c.max ? c.used : c.max - 1] = '\0';
    return (long)c.used;
}

long mynah_slm_render_chat(const mynah_slm_chat_family *f,
                           const mynah_slm_message *msgs, size_t n,
                           mynah_slm_think think, char *out, size_t max) {
    return mynah_slm_render_chat_tools(f, msgs, n, NULL, 0, think, out, max);
}
