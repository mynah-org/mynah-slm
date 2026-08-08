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

/* ── LFM2's Pythonic calls ─────────────────────────────────────────────────
 *
 *   <|tool_call_start|>[get_weather(city='Verona'), set_timer(minutes=5)]<|tool_call_end|>
 *
 * A list of Python call expressions, not JSON, and the exact inverse of what
 * template.c renders. Everything here is scanning rather than parsing: the
 * only structure that matters is where an argument ends, and a comma inside a
 * quoted string or a nested object is not that place. Getting THAT wrong is
 * how a parser silently truncates `city='Bad, Homburg'` into a call with a
 * broken argument and no error anywhere.
 *
 * Values are converted back to JSON, so a caller downstream of this never
 * learns which dialect the model spoke. */

static const char PY_OPEN[]  = "<|tool_call_start|>";
static const char PY_CLOSE[] = "<|tool_call_end|>";

/* One argument's worth of scanning: stop at a top-level `,` or `)`. */
static const char *py_scan_value(const char *s, const char *end) {
    int depth = 0;
    while (s < end) {
        const char ch = *s;
        if (ch == '\'' || ch == '"') {
            const char q = ch;
            for (s++; s < end && *s != q; s++)
                if (*s == '\\' && s + 1 < end) s++;      /* skip the escaped one */
            if (s < end) s++;
            continue;
        }
        if (ch == '(' || ch == '[' || ch == '{') depth++;
        else if (ch == ')' || ch == ']' || ch == '}') {
            if (depth == 0) return s;                    /* the call's own ')' */
            depth--;
        } else if (ch == ',' && depth == 0) return s;
        s++;
    }
    return end;
}

/* Everything between `[` and `]`, as calls. Defined below the cursor it
 * builds each arguments object with. */
static long py_parse_list(const char *s, const char *end,
                          mynah_slm_tool_call *out, size_t max, long found,
                          size_t *blocks);

long mynah_slm_tool_calls_parse(const char *text, size_t len,
                                mynah_slm_tool_call *out, size_t max,
                                size_t *n_blocks) {
    if (!text) return -1;

    size_t blocks = 0;
    long found = 0;
    const char *const end = text + len;
    const char *p = text;

    if (find_sub(text, end, PY_OPEN)) {
        /* LFM2. Only what sits between the markers is a call; the model's
         * prose around them is prose. */
        for (;;) {
            const char *o = find_sub(p, end, PY_OPEN);
            if (!o) break;
            const char *ls = o + sizeof PY_OPEN - 1;
            const char *cl = find_sub(ls, end, PY_CLOSE);
            const char *le = cl ? cl : end;
            found = py_parse_list(ls, le, out, max, found, &blocks);
            p = cl ? cl + sizeof PY_CLOSE - 1 : end;
        }
        if (n_blocks) *n_blocks = blocks;
        return found;
    }

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
         * consumed. Everything here is meant to be a call — but in which
         * dialect is no longer obvious once the markers are gone, so decide on
         * the first character that carries meaning. A Pythonic payload is the
         * bracketed list `[f(a=1)]`; a JSON one is an object. */
        const char *q = p;
        while (q < end && (*q == ' ' || *q == '\n' || *q == '\t' || *q == '\r')) q++;
        found = (q < end && *q == '[')
              ? py_parse_list(q, end, out, max, found, &blocks)
              : parse_region(p, end, out, max, found, &blocks);
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

/* ── the Pythonic side, now that there is a cursor to build with ───────────*/

static void put_char(cursor *c, char ch) {
    const char one[2] = { ch, '\0' };
    put(c, one);
}

/* One Python literal, appended as its JSON equivalent. */
static void py_value_to_json(cursor *c, const char *s, const char *end) {
    while (s < end && (*s == ' ' || *s == '\t' || *s == '\n')) s++;
    while (end > s && (end[-1] == ' ' || end[-1] == '\t' || end[-1] == '\n')) end--;
    if (s >= end) { put(c, "null"); return; }

    const size_t n = (size_t)(end - s);
    /* Capitalised, which is exactly why this cannot be a JSON passthrough. */
    if (n == 4 && memcmp(s, "True", 4) == 0)  { put(c, "true");  return; }
    if (n == 5 && memcmp(s, "False", 5) == 0) { put(c, "false"); return; }
    if (n == 4 && memcmp(s, "None", 4) == 0)  { put(c, "null");  return; }

    if (*s == '\'' || *s == '"') {
        const char q = *s;
        put(c, "\"");
        for (const char *p = s + 1; p + 1 < end; p++) {
            if (*p == '\\' && p + 2 < end) {
                const char nx = p[1];
                /* \' is Python's; JSON has no such escape and needs the bare
                 * quote. \\ \n \r \t are spelled identically in both. */
                if (nx == '\'')      put(c, "'");
                else if (nx == '"')  put(c, "\\\"");
                else { put_char(c, '\\'); put_char(c, nx); }
                p++;
                continue;
            }
            if (*p == '"') { put(c, "\\\""); continue; }
            put_char(c, *p);
        }
        put(c, "\"");
        (void)q;
        return;
    }

    /* Numbers, and the objects and lists the template emitted as JSON in the
     * first place — both are already valid JSON, so they go through as they
     * are rather than being re-serialized into a second chance to be wrong. */
    for (const char *p = s; p < end; p++) put_char(c, *p);
}

static long py_parse_list(const char *s, const char *end,
                          mynah_slm_tool_call *out, size_t max, long found,
                          size_t *blocks) {
    while (s < end && *s != '[') s++;
    if (s < end) s++;                                     /* past the '[' */

    while (s < end) {
        while (s < end && (*s == ' ' || *s == ',' || *s == '\n' || *s == '\t')) s++;
        if (s >= end || *s == ']') break;

        const char *nstart = s;
        while (s < end && *s != '(' && *s != ']') s++;
        if (s >= end || *s != '(') break;                 /* not a call at all */
        const char *nend = s;
        while (nend > nstart && (nend[-1] == ' ' || nend[-1] == '\n')) nend--;
        if (nend == nstart) break;
        (*blocks)++;
        s++;                                              /* past the '(' */

        char args[4096];
        cursor ac = { .buf = args, .max = sizeof args, .used = 0 };
        put(&ac, "{");
        int first = 1;

        for (;;) {
            while (s < end && (*s == ' ' || *s == '\n' || *s == '\t')) s++;
            if (s >= end || *s == ')') break;

            const char *ks = s;
            while (s < end && *s != '=' && *s != ')' && *s != ',') s++;
            if (s >= end) break;
            if (*s != '=') {
                /* A positional argument. It has no name to key on, and
                 * inventing one would be worse than dropping it: skip to the
                 * next argument and let the call stand without it. */
                s = py_scan_value(ks, end);
                if (s < end && *s == ',') s++;
                continue;
            }
            const char *ke = s;
            while (ke > ks && (ke[-1] == ' ')) ke--;
            s++;                                          /* past the '=' */

            const char *vs = s;
            const char *ve = py_scan_value(vs, end);

            if (!first) put(&ac, ", ");
            first = 0;
            put(&ac, "\"");
            for (const char *p = ks; p < ke; p++) put_char(&ac, *p);
            put(&ac, "\": ");
            py_value_to_json(&ac, vs, ve);

            s = ve;
            if (s < end && *s == ',') s++;
        }
        put(&ac, "}");
        if (s < end && *s == ')') s++;

        /* Overflowed the argument buffer, or overlong name: count the block as
         * attempted and do not pretend it parsed. */
        const size_t nlen = (size_t)(nend - nstart);
        if (ac.used >= sizeof args || nlen >= sizeof out[0].name) continue;

        if ((size_t)found < max && out) {
            memcpy(out[found].name, nstart, nlen);
            out[found].name[nlen] = '\0';
            out[found].arguments = dup_span(args, ac.used);
            if (!out[found].arguments) continue;
        }
        found++;
    }
    return found;
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
