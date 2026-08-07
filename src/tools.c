/* tools.c — see tools.h.
 * SPDX-License-Identifier: MIT */
#include "tools.h"

#include "json.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char TOOL_OPEN[]  = "<tool_call>";
static const char TOOL_CLOSE[] = "</tool_call>";

/* ── out: parsing calls back out of generated text ─────────────────────────*/

static const char *find_sub(const char *s, const char *end, const char *needle) {
    const size_t n = strlen(needle);
    if (n == 0 || (size_t)(end - s) < n) return NULL;
    for (const char *p = s; p + n <= end; p++)
        if (memcmp(p, needle, n) == 0) return p;
    return NULL;
}

static char *dup_span(const char *s, size_t n) {
    char *d = malloc(n + 1);
    if (!d) return NULL;
    memcpy(d, s, n);
    d[n] = '\0';
    return d;
}

/* Fill one call from a parsed `{"name":..,"arguments":..}`. Returns 0 when the
 * result is usable — a name and an arguments OBJECT. Anything else is the
 * model having produced something we must not hand to a caller as if it were
 * a function call. */
static int call_from_object(const json_val *o, mynah_slm_tool_call *c) {
    json_val name, args;

    if (json_object_get(o, "name", &name) != 0 || name.kind != JSON_STRING) return -1;
    if (json_string_copy(&name, c->name, sizeof c->name) <= 0) return -1;

    /* A no-argument function is legal and some models omit the key entirely. */
    if (json_object_get(o, "arguments", &args) != 0 || args.kind == JSON_NULL) {
        c->arguments = dup_span("{}", 2);
        return c->arguments ? 0 : -1;
    }

    if (args.kind == JSON_OBJECT) {
        c->arguments = dup_span(args.start, (size_t)(args.end - args.start));
        return c->arguments ? 0 : -1;
    }

    /* Stringified arguments: `"arguments": "{\"city\": \"Verona\"}"`. Several
     * fine-tunes emit this, and OpenAI's own wire format uses it, so accept it
     * — but only if what it contains really is an object. */
    if (args.kind == JSON_STRING) {
        char *inner = json_string_dup(&args);
        if (!inner) return -1;
        json_val chk;
        const size_t n = strlen(inner);
        if (json_parse(inner, inner + n, &chk) != 0 || chk.kind != JSON_OBJECT) {
            free(inner);
            return -1;
        }
        c->arguments = inner;
        return 0;
    }

    return -1;
}

/* Every JSON object in [s,end), in order. `found` is the running count of
 * well-formed calls across regions, which is also the next output slot. */
static long parse_region(const char *s, const char *end,
                         mynah_slm_tool_call *out, size_t max, long found,
                         size_t *blocks) {
    while (s < end) {
        while (s < end && *s != '{') s++;
        if (s >= end) break;

        json_val v;
        if (json_parse(s, end, &v) != 0 || v.kind != JSON_OBJECT) {
            /* Truncated or malformed: it was an attempt, and stopping here is
             * the only safe move — resynchronizing mid-JSON invents calls. */
            (*blocks)++;
            break;
        }
        (*blocks)++;

        mynah_slm_tool_call c;
        memset(&c, 0, sizeof c);
        if (call_from_object(&v, &c) == 0) {
            if (out && (size_t)found < max) out[found] = c;
            else free(c.arguments);
            found++;
        }
        s = v.end;
    }
    return found;
}

long mynah_slm_tool_calls_parse(const char *text, size_t len,
                                mynah_slm_tool_call *out, size_t max,
                                size_t *n_blocks) {
    if (!text) return -1;

    size_t blocks = 0;
    long found = 0;
    const char *const end = text + len;
    const char *p = text;

    if (find_sub(text, end, TOOL_OPEN)) {
        /* Wrapped: only what sits between the markers is a call. Text outside
         * them is the model talking, and a stray `{` in prose must not become
         * a function call. */
        for (;;) {
            const char *o = find_sub(p, end, TOOL_OPEN);
            if (!o) break;
            const char *js = o + sizeof TOOL_OPEN - 1;
            const char *c  = find_sub(js, end, TOOL_CLOSE);
            /* No closing marker means generation was cut short. The JSON in
             * front of it is still worth parsing — it usually is complete. */
            const char *je = c ? c : end;
            found = parse_region(js, je, out, max, found, &blocks);
            p = c ? c + sizeof TOOL_CLOSE - 1 : end;
        }
    } else {
        /* Bare: this is the tool channel, whose markers generate.c already
         * consumed. Everything here is meant to be a call. */
        found = parse_region(p, end, out, max, found, &blocks);
    }

    if (n_blocks) *n_blocks = blocks;
    return found;
}

void mynah_slm_tool_calls_free(mynah_slm_tool_call *calls, size_t n) {
    if (!calls) return;
    for (size_t i = 0; i < n; i++) {
        free(calls[i].arguments);
        calls[i].arguments = NULL;
    }
}

/* ── a snprintf-style cursor, same shape as template.c's ───────────────────*/

typedef struct { char *buf; size_t max, used; } cursor;

static void put(cursor *c, const char *s) {
    const size_t n = strlen(s);
    if (c->buf && c->used + n < c->max) memcpy(c->buf + c->used, s, n);
    c->used += n;
}

/* Writes only when the whole escaped string fits; otherwise it just counts,
 * which is what the sizing pass wants. */
static void put_escaped(cursor *c, const char *s) {
    const size_t n = strlen(s);
    const size_t need = json_escape(s, n, NULL, 0);
    if (c->buf && c->used + need + 1 <= c->max)
        json_escape(s, n, c->buf + c->used, c->max - c->used);
    c->used += need;
}

size_t mynah_slm_tool_calls_to_json(const mynah_slm_tool_call *calls, size_t n,
                                    char *buf, size_t max) {
    cursor c = { .buf = buf, .max = max, .used = 0 };

    put(&c, "[");
    for (size_t i = 0; i < n; i++) {
        char id[96];
        snprintf(id, sizeof id, "%s{\"index\":%zu,\"id\":\"call_%zu\","
                                "\"type\":\"function\",\"function\":{\"name\":",
                 i ? "," : "", i, i);
        put(&c, id);
        put_escaped(&c, calls[i].name);
        put(&c, ",\"arguments\":");
        put_escaped(&c, calls[i].arguments ? calls[i].arguments : "{}");
        put(&c, "}}");
    }
    put(&c, "]");

    if (c.buf && c.max) c.buf[c.used < c.max ? c.used : c.max - 1] = '\0';
    return c.used;
}

long mynah_slm_tool_calls_from_json(const char *text, size_t len,
                                    mynah_slm_tool_call *out, size_t max) {
    if (!text) return -1;

    json_val arr;
    if (json_parse(text, text + len, &arr) != 0 || arr.kind != JSON_ARRAY) return -1;

    long found = 0;
    for (size_t i = 0;; i++) {
        json_val elem, fn;
        if (json_array_at(&arr, i, &elem) != 0) break;
        if (elem.kind != JSON_OBJECT) continue;
        if (json_object_get(&elem, "function", &fn) != 0 || fn.kind != JSON_OBJECT)
            fn = elem;

        mynah_slm_tool_call c;
        memset(&c, 0, sizeof c);
        if (call_from_object(&fn, &c) != 0) continue;
        if (out && (size_t)found < max) out[found] = c;
        else free(c.arguments);
        found++;
    }
    return found;
}

/* ── in: the tool set ──────────────────────────────────────────────────────*/

struct mynah_slm_tool_set {
    mynah_slm_tool *items;
    size_t          n;
};

static int fail(char *err, size_t errsz, const char *msg) {
    if (err && errsz) snprintf(err, errsz, "%s", msg);
    return -1;
}

static char *dup_json_string(const json_val *v) { return json_string_dup(v); }

mynah_slm_tool_set *mynah_slm_tools_parse(const char *text, size_t len,
                                          char *err, size_t errsz) {
    if (err && errsz) err[0] = '\0';
    if (!text) { fail(err, errsz, "no tool definitions"); return NULL; }

    json_val root;
    if (json_parse(text, text + len, &root) != 0) {
        fail(err, errsz, "tools: malformed JSON");
        return NULL;
    }
    if (root.kind != JSON_ARRAY && root.kind != JSON_OBJECT) {
        fail(err, errsz, "tools: expected a JSON array of tool definitions");
        return NULL;
    }

    mynah_slm_tool_set *s = calloc(1, sizeof *s);
    if (!s) { fail(err, errsz, "out of memory"); return NULL; }

    for (size_t i = 0;; i++) {
        json_val elem;
        if (root.kind == JSON_OBJECT) {
            if (i > 0) break;
            elem = root;                        /* a single tool, unwrapped */
        } else if (json_array_at(&root, i, &elem) != 0) {
            break;
        }
        if (elem.kind != JSON_OBJECT) {
            fail(err, errsz, "tools: every entry must be an object");
            goto bad;
        }

        /* OpenAI nests the real definition under "function"; several clients
         * send it flat. Both are unambiguous, so take both. */
        json_val fn;
        if (json_object_get(&elem, "function", &fn) != 0 || fn.kind != JSON_OBJECT)
            fn = elem;

        json_val name, desc, params;
        if (json_object_get(&fn, "name", &name) != 0 || name.kind != JSON_STRING) {
            fail(err, errsz, "tools: a tool needs a string \"name\"");
            goto bad;
        }

        mynah_slm_tool *grown = realloc(s->items, (s->n + 1) * sizeof *grown);
        if (!grown) { fail(err, errsz, "out of memory"); goto bad; }
        s->items = grown;

        mynah_slm_tool *t = &s->items[s->n];
        memset(t, 0, sizeof *t);
        s->n++;

        t->name = dup_json_string(&name);
        if (!t->name) { fail(err, errsz, "out of memory"); goto bad; }

        if (json_object_get(&fn, "description", &desc) == 0 && desc.kind == JSON_STRING) {
            t->description = dup_json_string(&desc);
            if (!t->description) { fail(err, errsz, "out of memory"); goto bad; }
        }

        /* Kept as the caller wrote it: a JSON Schema is text we pass through,
         * and re-serializing it would only invent differences from what the
         * model was trained on. */
        if (json_object_get(&fn, "parameters", &params) == 0 && params.kind == JSON_OBJECT) {
            t->parameters = dup_span(params.start, (size_t)(params.end - params.start));
            if (!t->parameters) { fail(err, errsz, "out of memory"); goto bad; }
        }
    }

    if (s->n == 0) {
        fail(err, errsz, "tools: the array is empty");
        goto bad;
    }
    return s;

bad:
    mynah_slm_tools_free(s);
    return NULL;
}

void mynah_slm_tools_free(mynah_slm_tool_set *s) {
    if (!s) return;
    for (size_t i = 0; i < s->n; i++) {
        free((void *)s->items[i].name);
        free((void *)s->items[i].description);
        free((void *)s->items[i].parameters);
    }
    free(s->items);
    free(s);
}

const mynah_slm_tool *mynah_slm_tools_items(const mynah_slm_tool_set *s) {
    return s ? s->items : NULL;
}

size_t mynah_slm_tools_count(const mynah_slm_tool_set *s) { return s ? s->n : 0; }
