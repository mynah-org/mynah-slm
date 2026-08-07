/* json.h — just enough JSON for an OpenAI-shaped request body.
 *
 * Not a general parser and not trying to be. It reads the handful of shapes
 * this server accepts and writes correctly escaped strings; anything richer is
 * a dependency, and the whole point of the project is not having one.
 *
 * The reader is destructive-free: it borrows spans out of the caller's buffer
 * rather than copying, so the body must outlive every value taken from it.
 *
 * SPDX-License-Identifier: MIT */
#ifndef MYNAH_SLM_JSON_H
#define MYNAH_SLM_JSON_H

#include <stddef.h>

typedef enum {
    JSON_NULL = 0, JSON_BOOL, JSON_NUMBER, JSON_STRING, JSON_ARRAY, JSON_OBJECT,
} json_kind;

typedef struct json_val json_val;

struct json_val {
    json_kind   kind;
    const char *start;    /* first byte of this value in the source */
    const char *end;      /* one past the last */
    double      num;
    int         boolean;
};

/* Find `key` in the object at `obj` and fill `out`. Returns 0 when found.
 * Only scans the object's own members, not nested ones. */
int json_object_get(const json_val *obj, const char *key, json_val *out);

/* Parse one value at `s` (skipping leading space). Returns 0 on success. */
int json_parse(const char *s, const char *end, json_val *out);

/* Iterate an array: call with idx 0,1,... until it returns non-zero. */
int json_array_at(const json_val *arr, size_t idx, json_val *out);

/* Copy a JSON string value into `buf`, resolving escapes. Returns the length
 * written, or -1 if it does not fit. `v` must be JSON_STRING. */
long json_string_copy(const json_val *v, char *buf, size_t max);

/* Allocating twin, for message contents of unknown size. Caller frees. */
char *json_string_dup(const json_val *v);

/* Convenience readers with defaults, for optional request fields. */
double json_get_number(const json_val *obj, const char *key, double dflt);
int    json_get_bool(const json_val *obj, const char *key, int dflt);

/* Append `s` to `buf` as an escaped JSON string INCLUDING the quotes. Returns
 * the number of bytes that would be written, snprintf-style. */
size_t json_escape(const char *s, size_t n, char *buf, size_t max);

#endif /* MYNAH_SLM_JSON_H */
