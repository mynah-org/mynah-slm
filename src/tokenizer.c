/* tokenizer.c — GPT-2 style byte-level BPE over the GGUF vocabulary.
 *
 * Three stages, in this order, and the order is the whole thing:
 *
 *   1. pre-tokenize: split the text on the Qwen2 regex. Merges may never cross
 *      one of these boundaries, which is what stops "dog." and "dog" being
 *      different words and what keeps digits apart.
 *   2. byte-level encode: map each raw byte to a printable codepoint, so the
 *      vocabulary is pure text and any byte sequence is representable. This is
 *      why "hello" appears in the vocab as "Ġhello" with a leading U+0120.
 *   3. BPE: repeatedly merge the adjacent pair with the lowest rank.
 *
 * SPDX-License-Identifier: MIT */
#include "tokenizer.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "unicode_tables.inc"

/* ── small helpers ────────────────────────────────────────────────────────── */

static int fail(char *err, size_t errsz, const char *msg) {
    if (err && errsz) { snprintf(err, errsz, "%s", msg); }
    return -1;
}

static int in_ranges(uint32_t cp, const uint32_t (*r)[2], int n) {
    int lo = 0, hi = n - 1;
    while (lo <= hi) {
        const int mid = (lo + hi) / 2;
        if (cp < r[mid][0])      hi = mid - 1;
        else if (cp > r[mid][1]) lo = mid + 1;
        else                     return 1;
    }
    return 0;
}

static int uc_is_letter(uint32_t cp) { return in_ranges(cp, uc_letter_ranges, UC_LETTER_RANGES_COUNT); }
static int uc_is_number(uint32_t cp) { return in_ranges(cp, uc_number_ranges, UC_NUMBER_RANGES_COUNT); }
static int uc_is_space(uint32_t cp)  { return in_ranges(cp, uc_space_ranges,  UC_SPACE_RANGES_COUNT); }

/* Decode one UTF-8 codepoint. Returns its byte length, or 0 on malformed
 * input (the caller then advances one byte and treats it as U+FFFD-ish). */
static int utf8_next(const unsigned char *s, size_t n, uint32_t *cp) {
    if (n == 0) return 0;
    const unsigned char c = s[0];
    if (c < 0x80) { *cp = c; return 1; }
    if ((c & 0xE0) == 0xC0 && n >= 2 && (s[1] & 0xC0) == 0x80) {
        *cp = ((uint32_t)(c & 0x1F) << 6) | (s[1] & 0x3F);
        return *cp >= 0x80 ? 2 : 0;
    }
    if ((c & 0xF0) == 0xE0 && n >= 3 && (s[1] & 0xC0) == 0x80 && (s[2] & 0xC0) == 0x80) {
        *cp = ((uint32_t)(c & 0x0F) << 12) | ((uint32_t)(s[1] & 0x3F) << 6) | (s[2] & 0x3F);
        return *cp >= 0x800 ? 3 : 0;
    }
    if ((c & 0xF8) == 0xF0 && n >= 4 && (s[1] & 0xC0) == 0x80 && (s[2] & 0xC0) == 0x80 &&
        (s[3] & 0xC0) == 0x80) {
        *cp = ((uint32_t)(c & 0x07) << 18) | ((uint32_t)(s[1] & 0x3F) << 12) |
              ((uint32_t)(s[2] & 0x3F) << 6) | (s[3] & 0x3F);
        return *cp >= 0x10000 && *cp <= 0x10FFFF ? 4 : 0;
    }
    return 0;
}

static int utf8_encode(uint32_t cp, char *out) {
    if (cp < 0x80)    { out[0] = (char)cp; return 1; }
    if (cp < 0x800)   { out[0] = (char)(0xC0 | (cp >> 6));  out[1] = (char)(0x80 | (cp & 0x3F)); return 2; }
    if (cp < 0x10000) { out[0] = (char)(0xE0 | (cp >> 12)); out[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
                        out[2] = (char)(0x80 | (cp & 0x3F)); return 3; }
    out[0] = (char)(0xF0 | (cp >> 18));         out[1] = (char)(0x80 | ((cp >> 12) & 0x3F));
    out[2] = (char)(0x80 | ((cp >> 6) & 0x3F)); out[3] = (char)(0x80 | (cp & 0x3F));
    return 4;
}

/* How many bytes a UTF-8 sequence starting with this byte should have.
 * 0 for a continuation byte or an invalid lead. */
static size_t utf8_len(unsigned char c) {
    if (c < 0x80) return 1;
    if ((c & 0xE0) == 0xC0) return 2;
    if ((c & 0xF0) == 0xE0) return 3;
    if ((c & 0xF8) == 0xF0) return 4;
    return 0;
}

/* ── the byte-level alphabet ──────────────────────────────────────────────── */

/* GPT-2's mapping: printable ASCII and Latin-1 stay themselves; everything
 * else moves up to U+0100+. Space becomes U+0120 'Ġ', newline U+010A. */
static uint32_t byte_to_cp[256];
static int      cp_to_byte_init;
static int      cp_to_byte[0x200];      /* only 0x00..0x1FF are ever produced */

static void byte_alphabet_init(void) {
    if (cp_to_byte_init) return;
    for (int i = 0; i < 0x200; i++) cp_to_byte[i] = -1;

    int n = 0;
    for (int b = 0; b < 256; b++) {
        const int printable = (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) ||
                              (b >= 0xAE && b <= 0xFF);
        byte_to_cp[b] = printable ? (uint32_t)b : (uint32_t)(256 + n++);
    }
    for (int b = 0; b < 256; b++) cp_to_byte[byte_to_cp[b]] = b;
    cp_to_byte_init = 1;
}

/* ── the tokenizer ────────────────────────────────────────────────────────── */

typedef struct {
    const char *key;     /* "left right", not NUL-terminated */
    size_t      len;
    int32_t     rank;
} merge_entry;

typedef struct {
    const char *key;
    size_t      len;
    int32_t     id;
} vocab_entry;

struct mynah_slm_tokenizer {
    /* Token strings, pointing into the GGUF mapping — not copied. A 151936
     * entry vocabulary is 2 MB of text and there is no reason to duplicate it. */
    const char **tok;
    size_t      *tok_len;
    int32_t     *tok_type;
    size_t       n_tok;

    /* Open-addressed hash tables. Power of two, linear probing: the load
     * factor is fixed at load time so there is no rehash to get wrong. */
    vocab_entry *vmap;
    size_t       vmask;
    merge_entry *mmap_;
    size_t       mmask;

    uint32_t eos;

    /* Special tokens sorted longest-first, so "<|im_start|>" wins over a
     * shorter prefix when scanning. */
    uint32_t *special;
    size_t    n_special;
};

static uint64_t hash_bytes(const char *s, size_t n) {
    uint64_t h = 1469598103934665603ull;          /* FNV-1a */
    for (size_t i = 0; i < n; i++) { h ^= (unsigned char)s[i]; h *= 1099511628211ull; }
    return h;
}

static int32_t vocab_lookup(const mynah_slm_tokenizer *t, const char *s, size_t n) {
    size_t i = hash_bytes(s, n) & t->vmask;
    while (t->vmap[i].key) {
        if (t->vmap[i].len == n && memcmp(t->vmap[i].key, s, n) == 0) return t->vmap[i].id;
        i = (i + 1) & t->vmask;
    }
    return -1;
}

static void vocab_insert(mynah_slm_tokenizer *t, const char *s, size_t n, int32_t id) {
    size_t i = hash_bytes(s, n) & t->vmask;
    while (t->vmap[i].key) {
        /* Duplicate strings exist in padded vocabularies; the first id wins,
         * which matches what the reference tokenizer does. */
        if (t->vmap[i].len == n && memcmp(t->vmap[i].key, s, n) == 0) return;
        i = (i + 1) & t->vmask;
    }
    t->vmap[i].key = s;
    t->vmap[i].len = n;
    t->vmap[i].id  = id;
}

static int32_t merge_rank(const mynah_slm_tokenizer *t, const char *a, size_t na,
                          const char *b, size_t nb) {
    /* The key is "a b" without building it: hash and compare in two pieces. */
    char buf[512];
    if (na + 1 + nb > sizeof buf) return -1;
    memcpy(buf, a, na);
    buf[na] = ' ';
    memcpy(buf + na + 1, b, nb);
    const size_t n = na + 1 + nb;

    size_t i = hash_bytes(buf, n) & t->mmask;
    while (t->mmap_[i].key) {
        if (t->mmap_[i].len == n && memcmp(t->mmap_[i].key, buf, n) == 0)
            return t->mmap_[i].rank;
        i = (i + 1) & t->mmask;
    }
    return -1;
}

static size_t round_pow2(size_t n) {
    size_t p = 1;
    while (p < n) p <<= 1;
    return p;
}

mynah_slm_tokenizer *mynah_slm_tokenizer_load(const ingot_gguf *g, char *err, size_t errsz) {
    byte_alphabet_init();

    const ingot_kv *kv_tok = ingot_gguf_kv_find(g, "tokenizer.ggml.tokens");
    const ingot_kv *kv_mrg = ingot_gguf_kv_find(g, "tokenizer.ggml.merges");
    if (!kv_tok) { fail(err, errsz, "no tokenizer.ggml.tokens in this checkpoint"); return NULL; }
    if (!kv_mrg) { fail(err, errsz, "no tokenizer.ggml.merges in this checkpoint"); return NULL; }

    uint64_t n_tok = 0, n_mrg = 0;
    ingot_kv_arr_len(kv_tok, &n_tok);
    ingot_kv_arr_len(kv_mrg, &n_mrg);
    if (n_tok == 0) { fail(err, errsz, "empty vocabulary"); return NULL; }

    mynah_slm_tokenizer *t = calloc(1, sizeof *t);
    if (!t) { fail(err, errsz, "out of memory"); return NULL; }

    t->n_tok    = (size_t)n_tok;
    t->tok      = calloc(t->n_tok, sizeof *t->tok);
    t->tok_len  = calloc(t->n_tok, sizeof *t->tok_len);
    t->tok_type = calloc(t->n_tok, sizeof *t->tok_type);
    t->vmask    = round_pow2(t->n_tok * 2) - 1;
    t->vmap     = calloc(t->vmask + 1, sizeof *t->vmap);
    t->mmask    = round_pow2((n_mrg ? (size_t)n_mrg : 1) * 2) - 1;
    t->mmap_    = calloc(t->mmask + 1, sizeof *t->mmap_);
    if (!t->tok || !t->tok_len || !t->tok_type || !t->vmap || !t->mmap_) {
        fail(err, errsz, "out of memory building the vocabulary");
        mynah_slm_tokenizer_free(t);
        return NULL;
    }

    const ingot_kv *kv_type = ingot_gguf_kv_find(g, "tokenizer.ggml.token_type");
    for (uint64_t i = 0; i < n_tok; i++) {
        const char *s = NULL;
        size_t len = 0;
        if (ingot_kv_arr_str(kv_tok, i, &s, &len) != 0) {
            fail(err, errsz, "malformed tokenizer.ggml.tokens");
            mynah_slm_tokenizer_free(t);
            return NULL;
        }
        t->tok[i]     = s;
        t->tok_len[i] = len;
        if (kv_type) {
            int64_t ty = 1;
            ingot_kv_arr_i64(kv_type, i, &ty);
            t->tok_type[i] = (int32_t)ty;
        } else {
            t->tok_type[i] = 1;    /* NORMAL */
        }
        vocab_insert(t, s, len, (int32_t)i);
    }

    for (uint64_t i = 0; i < n_mrg; i++) {
        const char *s = NULL;
        size_t len = 0;
        if (ingot_kv_arr_str(kv_mrg, i, &s, &len) != 0) continue;
        size_t j = hash_bytes(s, len) & t->mmask;
        while (t->mmap_[j].key) j = (j + 1) & t->mmask;
        t->mmap_[j].key  = s;
        t->mmap_[j].len  = len;
        t->mmap_[j].rank = (int32_t)i;      /* earlier line = higher priority */
    }

    /* Control tokens, for the special-token scan. GGUF token_type 3 = CONTROL,
     * 4 = USER_DEFINED; both are matched literally when parse_special is on. */
    for (size_t i = 0; i < t->n_tok; i++)
        if (t->tok_type[i] == 3 || t->tok_type[i] == 4) t->n_special++;
    if (t->n_special) {
        t->special = calloc(t->n_special, sizeof *t->special);
        size_t k = 0;
        for (size_t i = 0; i < t->n_tok && t->special; i++)
            if (t->tok_type[i] == 3 || t->tok_type[i] == 4) t->special[k++] = (uint32_t)i;
    }

    const ingot_kv *kv_eos = ingot_gguf_kv_find(g, "tokenizer.ggml.eos_token_id");
    if (kv_eos) { uint64_t v = 0; ingot_kv_u64(kv_eos, &v); t->eos = (uint32_t)v; }

    return t;
}

void mynah_slm_tokenizer_free(mynah_slm_tokenizer *t) {
    if (!t) return;
    free(t->tok); free(t->tok_len); free(t->tok_type);
    free(t->vmap); free(t->mmap_); free(t->special);
    free(t);
}

size_t   mynah_slm_tokenizer_vocab_size(const mynah_slm_tokenizer *t) { return t->n_tok; }
uint32_t mynah_slm_tokenizer_eos(const mynah_slm_tokenizer *t)        { return t->eos; }

const char *mynah_slm_token_text(const mynah_slm_tokenizer *t, uint32_t id) {
    return id < t->n_tok ? t->tok[id] : NULL;
}

int mynah_slm_token_is_control(const mynah_slm_tokenizer *t, uint32_t id) {
    return id < t->n_tok && t->tok_type[id] == 3;
}

/* ── pre-tokenizer ────────────────────────────────────────────────────────── */

/* The Qwen2 split, implemented directly rather than through a regex engine:
 *
 *   (?i:'s|'t|'re|'ve|'m|'ll|'d)      contractions
 * | [^\r\n\p{L}\p{N}]?\p{L}+          a word, with one leading symbol
 * | \p{N}                             one digit at a time
 * |  ?[^\s\p{L}\p{N}]+[\r\n]*         punctuation run, optional leading space
 * | \s*[\r\n]+                        newlines
 * | \s+(?!\S)                         trailing whitespace
 * | \s+                               any whitespace
 *
 * Alternation order matters: the first branch that matches wins, exactly as
 * the regex crate would resolve it.
 *
 * Returns the byte length of the piece starting at s[0].
 */
static size_t pretoken_len(const unsigned char *s, size_t n) {
    uint32_t cp;
    int len = utf8_next(s, n, &cp);
    if (len == 0) return 1;                      /* invalid byte: take one */

    /* contractions */
    if (cp == '\'' && n >= 2) {
        const char *c1 = "stmd";
        for (const char *p = c1; *p; p++)
            if ((s[1] | 32) == (unsigned char)*p) return 2;
        if (n >= 3) {
            const char *two[] = {"re", "ve", "ll"};
            for (int i = 0; i < 3; i++)
                if ((s[1] | 32) == (unsigned char)two[i][0] &&
                    (s[2] | 32) == (unsigned char)two[i][1]) return 3;
        }
    }

    /* [^\r\n\p{L}\p{N}]? \p{L}+ — one optional non-letter, then letters. The
     * optional part only counts if a letter actually follows. */
    if (!uc_is_letter(cp) && !uc_is_number(cp) && cp != '\r' && cp != '\n') {
        uint32_t next;
        const int l2 = utf8_next(s + len, n - len, &next);
        if (l2 > 0 && uc_is_letter(next)) {
            size_t off = (size_t)len;
            while (off < n) {
                uint32_t c;
                const int l = utf8_next(s + off, n - off, &c);
                if (l == 0 || !uc_is_letter(c)) break;
                off += (size_t)l;
            }
            return off;
        }
    }
    if (uc_is_letter(cp)) {
        size_t off = 0;
        while (off < n) {
            uint32_t c;
            const int l = utf8_next(s + off, n - off, &c);
            if (l == 0 || !uc_is_letter(c)) break;
            off += (size_t)l;
        }
        return off;
    }

    /* \p{N} — a single digit, never a run */
    if (uc_is_number(cp)) return (size_t)len;

    /*  ?[^\s\p{L}\p{N}]+[\r\n]* */
    {
        size_t off = 0;
        if (cp == ' ') {
            uint32_t next;
            const int l2 = utf8_next(s + 1, n - 1, &next);
            if (l2 > 0 && !uc_is_space(next) && !uc_is_letter(next) && !uc_is_number(next))
                off = 1;
        }
        if (off > 0 || (!uc_is_space(cp) && !uc_is_letter(cp) && !uc_is_number(cp))) {
            size_t start = off;
            while (off < n) {
                uint32_t c;
                const int l = utf8_next(s + off, n - off, &c);
                if (l == 0 || uc_is_space(c) || uc_is_letter(c) || uc_is_number(c)) break;
                off += (size_t)l;
            }
            if (off > start) {
                while (off < n && (s[off] == '\r' || s[off] == '\n')) off++;
                return off;
            }
        }
    }

    /* \s*[\r\n]+
     *
     * `\s*` is greedy and backtracks, so the match runs to the LAST \r or \n
     * inside the whitespace run — not to the first one, and not to the end of
     * the run. Verified against HF's own pre-tokenizer:
     *
     *   "\n  "              -> "\n" + "  "        (stops at the last newline)
     *   " \n \n "           -> " \n \n" + " "
     *   "\t\t\n\r\t\t\n\t\n"-> the whole run       (it ends on a newline)
     *
     * Getting this wrong splits one whitespace piece into several, and BPE
     * cannot merge across a pre-token boundary — so the tokens differ while
     * the text round-trips perfectly, which is the hardest kind of wrong to
     * notice. It cost a fuzz run to find. */
    if (uc_is_space(cp)) {
        size_t off = 0, last_nl = 0;
        int    seen_nl = 0;
        while (off < n) {
            uint32_t c;
            const int l = utf8_next(s + off, n - off, &c);
            if (l == 0 || !uc_is_space(c)) break;
            if (s[off] == '\r' || s[off] == '\n') { seen_nl = 1; last_nl = off; }
            off += (size_t)l;
        }
        if (seen_nl) return last_nl + 1;
    }

    /* \s+(?!\S) then \s+ — a whitespace run; if more text follows, the last
     * space belongs to it (that is what the negative lookahead does). */
    if (uc_is_space(cp)) {
        size_t off = 0, last = 0;
        while (off < n) {
            uint32_t c;
            const int l = utf8_next(s + off, n - off, &c);
            if (l == 0 || !uc_is_space(c)) break;
            last = off;
            off += (size_t)l;
        }
        if (off < n && off > 0) return last > 0 ? last : off;   /* leave one space */
        return off;
    }

    return (size_t)len;
}

/* ── BPE ──────────────────────────────────────────────────────────────────── */

typedef struct { size_t start, len; int prev, next; } symbol;

static long bpe_piece(const mynah_slm_tokenizer *t, const char *enc, size_t n,
                      uint32_t *out, size_t max, size_t written) {
    /* Symbols start as single codepoints of the byte-encoded string. */
    symbol *sym = malloc((n + 1) * sizeof *sym);
    if (!sym) return -1;

    int count = 0;
    for (size_t i = 0; i < n; ) {
        uint32_t cp;
        int l = utf8_next((const unsigned char *)enc + i, n - i, &cp);
        if (l == 0) l = 1;
        sym[count].start = i;
        sym[count].len   = (size_t)l;
        sym[count].prev  = count - 1;
        sym[count].next  = (i + (size_t)l < n) ? count + 1 : -1;
        i += (size_t)l;
        count++;
    }

    /* Repeatedly merge the best-ranked adjacent pair. O(n^2) in the worst
     * case; pre-tokenized pieces are short, and a heap here would be a second
     * thing that can be wrong for no measurable gain. */
    for (;;) {
        int32_t best_rank = -1;
        int best = -1;
        for (int i = 0; i >= 0 && i < count; i = sym[i].next) {
            const int j = sym[i].next;
            if (j < 0) break;
            const int32_t r = merge_rank(t, enc + sym[i].start, sym[i].len,
                                         enc + sym[j].start, sym[j].len);
            if (r >= 0 && (best_rank < 0 || r < best_rank)) { best_rank = r; best = i; }
        }
        if (best < 0) break;

        const int j = sym[best].next;
        sym[best].len += sym[j].len;
        sym[best].next = sym[j].next;
        if (sym[j].next >= 0) sym[sym[j].next].prev = best;
    }

    long produced = 0;
    for (int i = 0; i >= 0; i = sym[i].next) {
        const int32_t id = vocab_lookup(t, enc + sym[i].start, sym[i].len);
        if (id < 0) {
            /* Cannot happen with a complete byte-level vocabulary: every
             * single byte has a token. If it does, the vocabulary is wrong and
             * silence would be worse than a hard failure. */
            free(sym);
            return -1;
        }
        if (out && written + (size_t)produced < max) out[written + produced] = (uint32_t)id;
        produced++;
        if (sym[i].next < 0) break;
    }
    free(sym);
    return produced;
}

/* Byte-level encode a raw span into `enc`, returning its length. */
static size_t byte_encode(const char *s, size_t n, char *enc) {
    size_t o = 0;
    for (size_t i = 0; i < n; i++) o += (size_t)utf8_encode(byte_to_cp[(unsigned char)s[i]], enc + o);
    return o;
}

long mynah_slm_tokenize(const mynah_slm_tokenizer *t, const char *text,
                        int parse_special, uint32_t *out, size_t max) {
    if (!t || !text) return -1;
    const size_t len = strlen(text);
    size_t written = 0;
    long total = 0;

    for (size_t pos = 0; pos < len; ) {
        /* Special tokens are matched on the RAW text, before pre-tokenization,
         * because "<|im_start|>" contains characters the splitter would break
         * apart. Longest match wins. */
        if (parse_special) {
            int hit = -1;
            size_t hit_len = 0;
            for (size_t k = 0; k < t->n_special; k++) {
                const uint32_t id = t->special[k];
                const size_t sl = t->tok_len[id];
                if (sl > hit_len && sl <= len - pos && memcmp(text + pos, t->tok[id], sl) == 0) {
                    hit = (int)id;
                    hit_len = sl;
                }
            }
            if (hit >= 0) {
                if (out && written < max) out[written] = (uint32_t)hit;
                written++; total++;
                pos += hit_len;
                continue;
            }
        }

        size_t piece = pretoken_len((const unsigned char *)text + pos, len - pos);
        if (piece == 0) piece = 1;
        /* A special token further along must not be swallowed by this piece. */
        if (parse_special) {
            for (size_t k = 0; k < t->n_special; k++) {
                const uint32_t id = t->special[k];
                const char *found = memchr(text + pos, t->tok[id][0], piece);
                while (found && (size_t)(found - (text + pos)) < piece) {
                    const size_t off = (size_t)(found - (text + pos));
                    if (t->tok_len[id] <= len - pos - off &&
                        memcmp(found, t->tok[id], t->tok_len[id]) == 0 && off > 0) {
                        piece = off;
                        break;
                    }
                    found = memchr(found + 1, t->tok[id][0], piece - off - 1);
                }
            }
        }

        char stack_enc[1024];
        char *enc = stack_enc;
        if (piece * 2 + 4 > sizeof stack_enc) {
            enc = malloc(piece * 2 + 4);
            if (!enc) return -1;
        }
        const size_t enc_len = byte_encode(text + pos, piece, enc);

        const long got = bpe_piece(t, enc, enc_len, out, max, written);
        if (enc != stack_enc) free(enc);
        if (got < 0) return -1;

        written += (size_t)got;
        total   += got;
        pos     += piece;
    }
    return total;
}

/* ── streaming detokenizer ────────────────────────────────────────────────── */

void mynah_slm_detok_init(mynah_slm_detok *d, const mynah_slm_tokenizer *t) {
    memset(d, 0, sizeof *d);
    d->tok = t;
}

/* Undo the byte-level alphabet: each codepoint of the stored text maps back to
 * exactly one raw byte. */
static size_t token_raw_bytes(const mynah_slm_tokenizer *t, uint32_t id,
                              char *out, size_t max) {
    if (id >= t->n_tok) return 0;
    const char *s = t->tok[id];
    const size_t n = t->tok_len[id];
    size_t o = 0;
    for (size_t i = 0; i < n && o < max; ) {
        uint32_t cp;
        int l = utf8_next((const unsigned char *)s + i, n - i, &cp);
        if (l == 0) { i++; continue; }
        i += (size_t)l;
        const int b = (cp < 0x200) ? cp_to_byte[cp] : -1;
        out[o++] = (b >= 0) ? (char)b : (char)cp;
    }
    return o;
}

long mynah_slm_detok_feed(mynah_slm_detok *d, uint32_t id, char *out, size_t max) {
    char raw[1024];
    size_t n = token_raw_bytes(d->tok, id, raw, sizeof raw);

    /* Prepend whatever was carried over from last time. */
    char buf[1032];
    size_t total = 0;
    if (d->n_pending) { memcpy(buf, d->pending, d->n_pending); total = d->n_pending; }
    if (total + n > sizeof buf) n = sizeof buf - total;
    memcpy(buf + total, raw, n);
    total += n;

    /* Find the last byte position that ends a complete codepoint. Anything
     * after it is an unfinished sequence and must be held back. */
    size_t complete = total;
    for (size_t back = 1; back <= 4 && back <= total; back++) {
        const size_t i = total - back;
        const size_t need = utf8_len((unsigned char)buf[i]);
        if (need == 0) continue;                 /* continuation byte, keep looking */
        if (i + need > total) complete = i;      /* truncated: hold from here */
        break;
    }

    if (complete > max) return -1;
    memcpy(out, buf, complete);

    d->n_pending = total - complete;
    if (d->n_pending > sizeof d->pending) d->n_pending = sizeof d->pending;
    memcpy(d->pending, buf + complete, d->n_pending);
    return (long)complete;
}

long mynah_slm_detok_finish(mynah_slm_detok *d, char *out, size_t max) {
    (void)out; (void)max;
    /* Held bytes at end of stream never completed a character. Emitting them
     * would produce invalid UTF-8; dropping them loses nothing real. */
    d->n_pending = 0;
    return 0;
}
