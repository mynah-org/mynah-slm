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

#include "tools.h"

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

    /* An assistant turn that called functions. Replayed verbatim into the next
     * prompt, because a model that cannot see its own call cannot make sense
     * of the tool result that follows it. May be NULL; `content` may be NULL
     * only when this is set. */
    const mynah_slm_tool_call *tool_calls;
    size_t                     n_tool_calls;
} mynah_slm_message;

/* ── families ──────────────────────────────────────────────────────────────
 * Three so far. Qwen3 opens a turn with `<|im_start|>role\n` and closes it
 * with `<|im_end|>`, Granite with `<|start_of_role|>role<|end_of_role|>` and
 * `<|end_of_text|>`, LFM2 with Qwen3's markers exactly.
 *
 * For the first two the tool-call and tool-response BODIES were byte-identical,
 * which is why this began as a table of strings and not a second renderer.
 * LFM2 ends that: it emits a PYTHON CALL EXPRESSION rather than a JSON object
 * (`<|tool_call_start|>[get_weather(city='Verona')]<|tool_call_end|>`) and it
 * lists the schemas inside the system turn instead of an XML block. That is a
 * genuinely different body, so `tool_style` branches the renderer — pretending
 * two shapes are one is exactly what the old comment warned against.
 *
 * The fields are what the templates actually disagree on; anything they agree
 * on is in the renderer, once. */
typedef enum {
    /* <tools>{json}</tools> in, <tool_call>{json}</tool_call> out. */
    MYNAH_SLM_TOOLS_JSON_XML = 0,
    /* `List of tools: [{json}, ...]` in the system turn, and a bracketed list
     * of Python call expressions out. LFM2. */
    MYNAH_SLM_TOOLS_PYTHONIC,
} mynah_slm_tool_style;

typedef struct {
    const char *name;
    /* Emitted once, before anything else. LFM2's template starts with
     * `{{- bos_token -}}`; Qwen3's and Granite's do not. NULL means none —
     * and it must stay NULL for them, because a BOS the model never saw at
     * training time is a prompt shape, not a harmless prefix. */
    const char *bos;
    const char *role_open;       /* before the role name */
    const char *role_close;      /* after it */
    const char *turn_end;        /* closes a turn, newline included */
    const char *tools_prefix;    /* the preamble before the tool list */
    const char *tools_suffix;    /* the instruction after it */
    /* Granite emits a canned system turn when the caller supplied neither a
     * system message nor tools. Qwen3 emits nothing. NULL means nothing. */
    const char *default_system;
    /* Qwen3 suppresses reasoning by PRE-FILLING an empty think block. Granite
     * has no such mechanism, so `think` simply does not apply to it. */
    int         think_prefill;
    /* LFM2 goes the other way: its template ALWAYS opens `<think>` after the
     * assistant marker, so reasoning is the default and the only lever is
     * whether that tag gets emitted. NULL for families without it. */
    const char *think_open;
    mynah_slm_tool_style tool_style;
    /* The markers that delimit a call in GENERATED text. The tool channel is
     * split by token id, so these are looked up in the vocabulary rather than
     * matched as strings, and a family that spells them differently would
     * otherwise have its calls land in the visible answer as prose. */
    const char *call_open;
    const char *call_close;
} mynah_slm_chat_family;

/* By `general.architecture`. Unknown families get ChatML, which is what most
 * of the small-model world emits — stated as a default rather than pretended
 * to be detection. */
const mynah_slm_chat_family *mynah_slm_chat_family_for(const char *arch);

/* Render `n` messages plus the assistant's opening. Returns the number of
 * bytes the full rendering needs (excluding the NUL), like snprintf, so a
 * caller can size a buffer with out=NULL. Negative on error. */
long mynah_slm_render_chat(const mynah_slm_chat_family *f,
                           const mynah_slm_message *msgs, size_t n,
                           mynah_slm_think think, char *out, size_t max);

/* The same, with the function schemas the model may call. `tools` goes into
 * the system turn, merged with msgs[0] when that is a system message — which
 * is where Qwen3's own template puts it, not in a turn of its own.
 *
 * Passing n_tools = 0 is exactly mynah_slm_render_chat: tool calling is not a
 * mode the engine is in, it is a block in the prompt.
 *
 * A ROLE_ASSISTANT turn is rendered WITHOUT a reasoning block. HF's template
 * only replays reasoning for the final assistant turn, and we always append
 * the generation prompt instead of ending on one, so the two agree. */
long mynah_slm_render_chat_tools(const mynah_slm_chat_family *f,
                                 const mynah_slm_message *msgs, size_t n,
                                 const mynah_slm_tool *tools, size_t n_tools,
                                 mynah_slm_think think, char *out, size_t max);

#endif /* MYNAH_SLM_TEMPLATE_H */
