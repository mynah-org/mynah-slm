/* tools.h — function calling: the schemas that go into the prompt, and the
 * calls that come back out of the token stream.
 *
 * Two directions, deliberately kept apart:
 *
 *   IN   a tool set (name, description, JSON-Schema parameters) is rendered
 *        into the system turn by template.c, because the wording of that block
 *        is part of the chat template and nothing else.
 *   OUT  the model answers with `<tool_call>{"name":..,"arguments":{..}}`
 *        between two SPECIAL TOKENS. Those are whole tokens in Qwen3, so
 *        generate.c can route them to their own channel the same way it routes
 *        reasoning — the answer channel never sees the JSON, and a TTS stage
 *        downstream never reads a function call out loud.
 *
 * `parameters` is emitted VERBATIM. A JSON Schema is the caller's text and
 * re-serializing it would only add a way to differ from what HF's template
 * produces; the byte-parity test in tests/test_tools.c depends on this.
 *
 * SPDX-License-Identifier: MIT */
#ifndef MYNAH_SLM_TOOLS_H
#define MYNAH_SLM_TOOLS_H

#include <stddef.h>

/* One function the model may call. Strings are borrowed, not owned. */
typedef struct {
    const char *name;
    const char *description;   /* may be NULL */
    const char *parameters;    /* JSON Schema object, emitted verbatim */
} mynah_slm_tool;

/* One call the model asked for. `arguments` is the JSON object as text, always
 * a valid object — a block that did not parse never becomes one of these. */
typedef struct {
    char  name[128];
    char *arguments;           /* owned */
} mynah_slm_tool_call;

/* ── out: calls parsed from generated text ─────────────────────────────────
 * Accepts both shapes it can arrive in: the bare JSON of the tool channel
 * (markers already stripped by generate.c) and text that still carries the
 * literal <tool_call> wrappers, which is what a caller who never split the
 * channels has in hand.
 *
 * Returns how many WELL-FORMED calls were found — which can exceed `max`,
 * snprintf-style, so a caller can size an array — or -1 on a bad argument.
 * `n_blocks`, when not NULL, receives how many calls were ATTEMPTED. The two
 * numbers differing is the model emitting malformed JSON, and that is a thing
 * worth reporting rather than silently dropping.
 *
 * Only the first `max` entries are written; free exactly those. */
long mynah_slm_tool_calls_parse(const char *text, size_t len,
                                mynah_slm_tool_call *out, size_t max,
                                size_t *n_blocks);

void mynah_slm_tool_calls_free(mynah_slm_tool_call *calls, size_t n);

/* Render calls as OpenAI's `tool_calls` array. `arguments` becomes a JSON
 * STRING there, not an object — that is their wire format, not our choice.
 * Each entry carries an `index`, which streaming clients need to reassemble
 * deltas and non-streaming ones ignore.
 * snprintf-style: returns the length the full text needs. */
size_t mynah_slm_tool_calls_to_json(const mynah_slm_tool_call *calls, size_t n,
                                    char *buf, size_t max);

/* The inverse: read back an OpenAI `tool_calls` array, so a client can replay
 * the assistant turn that asked for the tool alongside the result it got.
 * Same return contract as _parse. */
long mynah_slm_tool_calls_from_json(const char *text, size_t len,
                                    mynah_slm_tool_call *out, size_t max);

/* ── in: a tool set parsed from OpenAI-shaped JSON ─────────────────────────
 * `[{"type":"function","function":{"name":..,"description":..,"parameters":{..}}}]`
 * — the array a client sends and the array a CLI user keeps in a file, so both
 * front ends share one path. A bare `{"name":..}` without the `function`
 * wrapper is accepted too; several clients send that.
 *
 * The set owns its strings, so the source buffer may go away afterwards. */
typedef struct mynah_slm_tool_set mynah_slm_tool_set;

mynah_slm_tool_set *mynah_slm_tools_parse(const char *text, size_t len,
                                          char *err, size_t errsz);
void  mynah_slm_tools_free(mynah_slm_tool_set *s);

const mynah_slm_tool *mynah_slm_tools_items(const mynah_slm_tool_set *s);
size_t                mynah_slm_tools_count(const mynah_slm_tool_set *s);

#endif /* MYNAH_SLM_TOOLS_H */
