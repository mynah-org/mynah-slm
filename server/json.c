/* json.c — see json.h.
 * SPDX-License-Identifier: MIT */
#include "json.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char *skip_ws(const char *s, const char *end) {
    while (s < end && (*s == ' ' || *s == '\t' || *s == '\n' || *s == '\r')) s++;
    return s;
}

/* Walk one value and return the byte after it, or NULL on malformed input.
 * Strings are scanned with escape awareness so a \" cannot end them early —
 * getting that wrong is how a parser reads past a value and into the next. */
static const char *scan(const char *s, const char *end);

static const char *scan_string(const char *s, const char *end) {
    if (s >= end || *s != '"') return NULL;
    s++;
    while (s < end) {
        if (*s == '\\') { s += 2; continue; }
        if (*s == '"')  return s + 1;
        s++;
    }
    return NULL;
}

static const char *scan_container(const char *s, const char *end, char open, char close) {
    if (s >= end || *s != open) return NULL;
    s++;
    for (;;) {
        s = skip_ws(s, end);
        if (s >= end) return NULL;
        if (*s == close) return s + 1;
        s = scan(s, end);
        if (!s) return NULL;
        s = skip_ws(s, end);
        if (s < end && (*s == ',' || *s == ':')) s++;
    }
}

static const char *scan(const char *s, const char *end) {
    s = skip_ws(s, end);
    if (s >= end) return NULL;
    switch (*s) {
        case '"': return scan_string(s, end);
        case '{': return scan_container(s, end, '{', '}');
        case '[': return scan_container(s, end, '[', ']');
        case 't': return (end - s >= 4) ? s + 4 : NULL;
        case 'f': return (end - s >= 5) ? s + 5 : NULL;
        case 'n': return (end - s >= 4) ? s + 4 : NULL;
        default: {
            const char *p = s;
            while (p < end && (*p == '-' || *p == '+' || *p == '.' ||
                               (*p >= '0' && *p <= '9') || *p == 'e' || *p == 'E')) p++;
            return p > s ? p : NULL;
        }
    }
}

int json_parse(const char *s, const char *end, json_val *out) {
    s = skip_ws(s, end);
    if (s >= end) return -1;
    const char *stop = scan(s, end);
    if (!stop) return -1;

    memset(out, 0, sizeof *out);
    out->start = s;
    out->end   = stop;
    switch (*s) {
        case '"': out->kind = JSON_STRING; break;
        case '{': out->kind = JSON_OBJECT; break;
        case '[': out->kind = JSON_ARRAY;  break;
        case 't': out->kind = JSON_BOOL; out->boolean = 1; break;
        case 'f': out->kind = JSON_BOOL; out->boolean = 0; break;
        case 'n': out->kind = JSON_NULL; break;
        default:  out->kind = JSON_NUMBER; out->num = strtod(s, NULL); break;
    }
    return 0;
}

int json_object_get(const json_val *obj, const char *key, json_val *out) {
    if (!obj || obj->kind != JSON_OBJECT) return -1;
    const char *s = obj->start + 1, *end = obj->end;
    const size_t klen = strlen(key);

    for (;;) {
        s = skip_ws(s, end);
        if (s >= end || *s == '}') return -1;

        const char *kstart = s;
        const char *kstop  = scan_string(s, end);
        if (!kstop) return -1;

        s = skip_ws(kstop, end);
        if (s >= end || *s != ':') return -1;
        s++;

        json_val v;
        if (json_parse(s, end, &v) != 0) return -1;

        /* Compare the raw span, minus the quotes. Escapes in a KEY are legal
         * but nothing we accept uses them, so a literal compare is honest. */
        if ((size_t)(kstop - kstart - 2) == klen &&
            memcmp(kstart + 1, key, klen) == 0) {
            *out = v;
            return 0;
        }

        s = skip_ws(v.end, end);
        if (s < end && *s == ',') s++;
    }
}

int json_array_at(const json_val *arr, size_t idx, json_val *out) {
    if (!arr || arr->kind != JSON_ARRAY) return -1;
    const char *s = arr->start + 1, *end = arr->end;

    for (size_t i = 0;; i++) {
        s = skip_ws(s, end);
        if (s >= end || *s == ']') return -1;

        json_val v;
        if (json_parse(s, end, &v) != 0) return -1;
        if (i == idx) { *out = v; return 0; }

        s = skip_ws(v.end, end);
        if (s < end && *s == ',') s++;
    }
}

static int hex4(const char *s) {
    int v = 0;
    for (int i = 0; i < 4; i++) {
        const char c = s[i];
        v <<= 4;
        if      (c >= '0' && c <= '9') v |= c - '0';
        else if (c >= 'a' && c <= 'f') v |= c - 'a' + 10;
        else if (c >= 'A' && c <= 'F') v |= c - 'A' + 10;
        else return -1;
    }
    return v;
}

static int utf8_put(unsigned cp, char *out) {
    if (cp < 0x80)    { out[0] = (char)cp; return 1; }
    if (cp < 0x800)   { out[0] = (char)(0xC0 | (cp >> 6)); out[1] = (char)(0x80 | (cp & 0x3F)); return 2; }
    if (cp < 0x10000) { out[0] = (char)(0xE0 | (cp >> 12)); out[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
                        out[2] = (char)(0x80 | (cp & 0x3F)); return 3; }
    out[0] = (char)(0xF0 | (cp >> 18));         out[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
    out[2] = (char)(0x80 | ((cp >> 6) & 0x3F)); out[3] = (char)(0x80 | (cp & 0x3F));
    return 4;
}

long json_string_copy(const json_val *v, char *buf, size_t max) {
    if (!v || v->kind != JSON_STRING) return -1;
    const char *s = v->start + 1, *end = v->end - 1;
    size_t o = 0;

    while (s < end) {
        if (o + 4 >= max) return -1;
        if (*s != '\\') { buf[o++] = *s++; continue; }
        s++;
        if (s >= end) break;
        switch (*s) {
            case 'n': buf[o++] = '\n'; s++; break;
            case 't': buf[o++] = '\t'; s++; break;
            case 'r': buf[o++] = '\r'; s++; break;
            case 'b': buf[o++] = '\b'; s++; break;
            case 'f': buf[o++] = '\f'; s++; break;
            case '/': buf[o++] = '/';  s++; break;
            case '"': buf[o++] = '"';  s++; break;
            case '\\': buf[o++] = '\\'; s++; break;
            case 'u': {
                if (end - s < 5) return -1;
                int cp = hex4(s + 1);
                if (cp < 0) return -1;
                s += 5;
                /* A surrogate pair is two \u escapes; joining them is what
                 * makes emoji survive a JSON round trip. */
                if (cp >= 0xD800 && cp <= 0xDBFF && end - s >= 6 &&
                    s[0] == '\\' && s[1] == 'u') {
                    const int lo = hex4(s + 2);
                    if (lo >= 0xDC00 && lo <= 0xDFFF) {
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                        s += 6;
                    }
                }
                o += (size_t)utf8_put((unsigned)cp, buf + o);
                break;
            }
            default: buf[o++] = *s++; break;
        }
    }
    buf[o] = '\0';
    return (long)o;
}

char *json_string_dup(const json_val *v) {
    if (!v || v->kind != JSON_STRING) return NULL;
    const size_t max = (size_t)(v->end - v->start) + 4;
    char *buf = malloc(max);
    if (!buf) return NULL;
    if (json_string_copy(v, buf, max) < 0) { free(buf); return NULL; }
    return buf;
}

double json_get_number(const json_val *obj, const char *key, double dflt) {
    json_val v;
    if (json_object_get(obj, key, &v) != 0 || v.kind != JSON_NUMBER) return dflt;
    return v.num;
}

int json_get_bool(const json_val *obj, const char *key, int dflt) {
    json_val v;
    if (json_object_get(obj, key, &v) != 0 || v.kind != JSON_BOOL) return dflt;
    return v.boolean;
}

size_t json_escape(const char *s, size_t n, char *buf, size_t max) {
    size_t o = 0;
#define PUT(c) do { if (o + 1 < max) buf[o] = (c); o++; } while (0)
    PUT('"');
    for (size_t i = 0; i < n; i++) {
        const unsigned char c = (unsigned char)s[i];
        switch (c) {
            case '"':  PUT('\\'); PUT('"');  break;
            case '\\': PUT('\\'); PUT('\\'); break;
            case '\n': PUT('\\'); PUT('n');  break;
            case '\r': PUT('\\'); PUT('r');  break;
            case '\t': PUT('\\'); PUT('t');  break;
            case '\b': PUT('\\'); PUT('b');  break;
            case '\f': PUT('\\'); PUT('f');  break;
            default:
                if (c < 0x20) {
                    /* Control characters must be escaped or the JSON is
                     * invalid. Everything >= 0x20 passes through as UTF-8. */
                    char tmp[7];
                    snprintf(tmp, sizeof tmp, "\\u%04x", c);
                    for (int k = 0; tmp[k]; k++) PUT(tmp[k]);
                } else {
                    PUT((char)c);
                }
        }
    }
    PUT('"');
#undef PUT
    if (max) buf[o < max ? o : max - 1] = '\0';
    return o;
}
