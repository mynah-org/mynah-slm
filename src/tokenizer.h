/* tokenizer.h — byte-level BPE, loaded straight from GGUF metadata.
 *
 * No tokenizer.json, no vocab.json, no merges.txt at runtime: everything comes
 * from `tokenizer.ggml.*` in the checkpoint the user already downloaded. A
 * second file to keep in sync with the weights is a second thing that can be
 * the wrong version.
 *
 * SPDX-License-Identifier: MIT */
#ifndef MYNAH_SLM_TOKENIZER_H
#define MYNAH_SLM_TOKENIZER_H

#include <stddef.h>
#include <stdint.h>

#include "ingot/gguf.h"

typedef struct mynah_slm_tokenizer mynah_slm_tokenizer;

/* Reads tokens, merges, token_type and the special ids. Returns NULL with the
 * reason in `err`. */
mynah_slm_tokenizer *mynah_slm_tokenizer_load(const ingot_gguf *g,
                                              char *err, size_t errsz);
void mynah_slm_tokenizer_free(mynah_slm_tokenizer *t);

size_t   mynah_slm_tokenizer_vocab_size(const mynah_slm_tokenizer *t);
uint32_t mynah_slm_tokenizer_eos(const mynah_slm_tokenizer *t);

/* Encode UTF-8 text. Writes at most `max` ids and returns how many were
 * produced, or -1 on error. Pass out=NULL to ask for the count first.
 *
 * `parse_special` controls whether literal "<|im_start|>" in the input becomes
 * the control token or gets encoded as text. Off by default everywhere user
 * content is involved: a chat template inserts control tokens by id, and text
 * that merely looks like one must never become one. */
long mynah_slm_tokenize(const mynah_slm_tokenizer *t, const char *text,
                        int parse_special, uint32_t *out, size_t max);

/* The literal piece for one id, as stored (byte-level encoded). Mostly useful
 * for diagnostics — for output, use the streaming decoder below. */
const char *mynah_slm_token_text(const mynah_slm_tokenizer *t, uint32_t id);

/* Is this a control token (a chat delimiter rather than text)? */
int mynah_slm_token_is_control(const mynah_slm_tokenizer *t, uint32_t id);

/* The id whose stored text is exactly `text`, or -1. Used to resolve markers
 * like "<think>" by name instead of hardcoding an id that is only right for
 * one vocabulary. */
long mynah_slm_token_find(const mynah_slm_tokenizer *t, const char *text);

/* ── streaming detokenizer ──────────────────────────────────────────────────
 * A token's bytes can end mid-UTF-8-sequence: emoji and CJK routinely split
 * across two tokens. Emitting a partial codepoint to a terminal prints a
 * replacement character that never repairs itself, and to an HTTP client
 * produces invalid JSON. So held bytes stay held until they complete.
 *
 * Not a nicety: it is the difference between a streaming API that works in
 * every language and one that works in English. */
typedef struct {
    const mynah_slm_tokenizer *tok;
    char   pending[8];      /* an incomplete UTF-8 sequence, at most 4 bytes */
    size_t n_pending;
} mynah_slm_detok;

void mynah_slm_detok_init(mynah_slm_detok *d, const mynah_slm_tokenizer *t);

/* Append one token's text. Writes only COMPLETE codepoints into `out` and
 * returns how many bytes were written; anything trailing is carried to the
 * next call. Returns -1 if `out` is too small. */
long mynah_slm_detok_feed(mynah_slm_detok *d, uint32_t id, char *out, size_t max);

/* Flush whatever is held at end of stream. Incomplete bytes are dropped: they
 * were never a character. */
long mynah_slm_detok_finish(mynah_slm_detok *d, char *out, size_t max);

#endif /* MYNAH_SLM_TOKENIZER_H */
