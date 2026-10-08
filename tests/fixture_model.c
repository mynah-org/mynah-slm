/* fixture_model.c — see fixture_model.h.
 * SPDX-License-Identifier: MIT */
#include "fixture_model.h"

#include "ingot/dtype.h"
#include "ingot/write.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

void fixture_spec_tiny(fixture_spec *s, int quant) {
    memset(s, 0, sizeof *s);
    s->n_layers = 2;
    s->d_model = 256;
    s->d_ff = 512;
    s->n_heads = 8;
    s->n_kv_heads = 2;     /* GQA groups of 4 */
    /* q_dim 512 != d_model 256: the projections are not square, like Qwen3.
     * Every row width is whole 256-blocks — see the Q8_0 note in pick_type. */
    s->head_dim = 64;
    s->n_ctx = 512;
    s->quant = quant;
    s->seed = 0x5eed5eedULL;
}

void fixture_spec_slow(fixture_spec *s) {
    memset(s, 0, sizeof *s);
    s->n_layers = 4;
    s->d_model = 1024;
    s->d_ff = 3072;
    s->n_heads = 8;
    s->n_kv_heads = 4;
    s->head_dim = 128;
    s->n_ctx = 8192;
    s->quant = 1;
    s->seed = 0x510e510eULL;
}

void fixture_spec_06b_shape(fixture_spec *s) {
    memset(s, 0, sizeof *s);
    s->n_layers = 28;
    s->d_model = 1024;
    s->d_ff = 3072;
    s->n_heads = 16;
    s->n_kv_heads = 8;
    s->head_dim = 128;
    s->n_ctx = 4096;
    s->quant = 1;
    s->seed = 0x06b06bULL;
    /* the tokenizer below has ~300 entries; pad the rows to the real head */
    s->vocab_pad = 151936;
}

/* ── deterministic noise ──────────────────────────────────────────────────── */

static uint64_t rng_next(uint64_t *s) {         /* xorshift64* */
    uint64_t x = *s;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    *s = x;
    return x * 0x2545F4914F6CDD1DULL;
}

/* uniform in [-1, 1) */
static float rng_unit(uint64_t *s) {
    return (float)((double)(rng_next(s) >> 11) * (1.0 / 9007199254740992.0)) * 2.0f - 1.0f;
}

/* ── the vocabulary ───────────────────────────────────────────────────────── */

/* GPT-2's byte alphabet, the same mapping src/tokenizer.c decodes: printable
 * bytes stand for themselves, the rest are shifted to U+0100 and up. */
static void byte_symbol(int b, char out[4]) {
    static int map[256], init;
    if (!init) {
        int n = 0;
        for (int i = 0; i < 256; i++) {
            const int printable = (i >= '!' && i <= '~') || (i >= 0xA1 && i <= 0xAC) ||
                                  (i >= 0xAE && i <= 0xFF);
            map[i] = printable ? i : 256 + n++;
        }
        init = 1;
    }
    const int cp = map[b];
    if (cp < 0x80) { out[0] = (char)cp; out[1] = 0; }
    else { out[0] = (char)(0xC0 | (cp >> 6)); out[1] = (char)(0x80 | (cp & 0x3F)); out[2] = 0; }
}

/* A few merges so the BPE loop is exercised, in byte-symbol spelling: "Ġ" is
 * the space. Each result is appended to the vocabulary in this order. */
static const char *const MERGES[][2] = {
    { "\xc4\xa0", "t" }, { "h", "e" }, { "\xc4\xa0t", "he" }, { "i", "n" },
    { "e", "r" }, { "a", "n" }, { "o", "n" }, { "\xc4\xa0", "a" },
    { "r", "e" }, { "\xc4\xa0", "s" }, { "e", "n" }, { "a", "t" },
    { "o", "r" }, { "\xc4\xa0", "c" }, { "i", "a" }, { "\xc4\xa0", "o" },
};

/* Qwen3's control and channel tokens, by name — the template and generate.c
 * resolve them that way, never by id. Type 3 = CONTROL, 4 = USER_DEFINED. */
static const struct { const char *text; int type; } SPECIALS[] = {
    { "<|endoftext|>", 3 }, { "<|im_start|>", 3 }, { "<|im_end|>", 3 },
    { "<think>", 4 }, { "</think>", 4 }, { "<tool_call>", 4 }, { "</tool_call>", 4 },
};

#define N_MERGES   (sizeof MERGES / sizeof *MERGES)
#define N_SPECIALS (sizeof SPECIALS / sizeof *SPECIALS)
#define N_TOKENS   (256 + N_MERGES + N_SPECIALS)

static int add_tokenizer(ingot_gguf_writer *w, uint32_t *eos_out) {
    static char sym[256][4];
    static char merged[N_MERGES][32];
    static char merge_line[N_MERGES][40];
    const char *tokens[N_TOKENS];
    const char *merges[N_MERGES];
    int32_t types[N_TOKENS];
    size_t n = 0;

    for (int b = 0; b < 256; b++) {
        byte_symbol(b, sym[b]);
        types[n] = 1;
        tokens[n++] = sym[b];
    }
    for (size_t i = 0; i < N_MERGES; i++) {
        snprintf(merged[i], sizeof merged[i], "%s%s", MERGES[i][0], MERGES[i][1]);
        snprintf(merge_line[i], sizeof merge_line[i], "%s %s", MERGES[i][0], MERGES[i][1]);
        merges[i] = merge_line[i];
        types[n] = 1;
        tokens[n++] = merged[i];
    }
    uint32_t eos = 0;
    for (size_t i = 0; i < N_SPECIALS; i++) {
        if (!strcmp(SPECIALS[i].text, "<|im_end|>")) eos = (uint32_t)n;
        types[n] = SPECIALS[i].type;
        tokens[n++] = SPECIALS[i].text;
    }
    *eos_out = eos;

    int rc = 0;
    rc |= ingot_gguf_kv_string(w, "tokenizer.ggml.model", "gpt2");
    rc |= ingot_gguf_kv_string(w, "tokenizer.ggml.pre", "qwen2");
    rc |= ingot_gguf_kv_array_string(w, "tokenizer.ggml.tokens", tokens, n);
    rc |= ingot_gguf_kv_array_i32(w, "tokenizer.ggml.token_type", types, n);
    rc |= ingot_gguf_kv_array_string(w, "tokenizer.ggml.merges", merges, N_MERGES);
    rc |= ingot_gguf_kv_u32(w, "tokenizer.ggml.eos_token_id", eos);
    return rc;
}

/* ── tensors ──────────────────────────────────────────────────────────────── */

/* The type a matrix with `cols` inputs gets under the spec.
 *
 * Anything that is not whole 256-blocks falls back to F32, NOT to Q8_0, even
 * though Q8_0's own block is 32 wide: ingot's ingot_q8_0_dequant (and its
 * matmat) decode through the k-quant super-block core, 8 Q8_0 blocks at a
 * time, and refuse a row that is not a multiple of 256. ingot_dequant_matrix
 * therefore fails on a 128-wide Q8_0 row while ingot_matvec accepts it, and
 * the batched prefill — which dequantizes a strip — fails where decode works.
 * Found by this fixture (2026-10-08); a reader fix, so it belongs upstream in
 * ingot (rule 4). Latent for Qwen3, whose widths are all multiples of 256. */
static int pick_type(const fixture_spec *s, uint64_t cols, int want) {
    if (!s->quant) return INGOT_TYPE_F32;
    return cols % 256 == 0 ? want : INGOT_TYPE_F32;
}

/* rows x cols, ggml order ne = {cols, rows}. `gain` scales a uniform draw;
 * `bias` is added (1.0 for a norm weight, 0 elsewhere). The buffer is freed as
 * soon as the writer has it: add_f32 quantizes into storage of its own. */
static int add_matrix(ingot_gguf_writer *w, const char *name, int type,
                      uint64_t rows, uint64_t cols, float gain, float bias,
                      uint64_t *rng) {
    const size_t n = (size_t)(rows * cols);
    float *v = malloc(n * sizeof *v);
    if (!v) return -1;
    for (size_t i = 0; i < n; i++) v[i] = bias + gain * rng_unit(rng);
    const uint64_t ne2[2] = { cols, rows };
    const uint64_t ne1[1] = { cols };
    const int rc = rows == 1 ? ingot_gguf_add_f32(w, name, type, 1, ne1, v)
                             : ingot_gguf_add_f32(w, name, type, 2, ne2, v);
    free(v);
    return rc;
}

/* The tied embedding, with every CONTROL/channel token's row zeroed. The head
 * is tied, so those tokens' logits are exactly 0 while the ~280 noise logits
 * spread around it: greedy never picks one. Noise weights would otherwise
 * sample <|im_end|> about once per vocabulary-size tokens and end a
 * "long" generation by themselves, which a disconnect test cannot tell from
 * a cancellation. As an INPUT a zero row is harmless: the template's markers
 * go through RMSNorm as zero vectors, no NaN (eps). */
static int add_embedding(ingot_gguf_writer *w, int type, uint64_t rows, uint64_t cols,
                         uint64_t *rng) {
    const size_t n = (size_t)(rows * cols);
    float *v = malloc(n * sizeof *v);
    if (!v) return -1;
    for (size_t i = 0; i < n; i++) v[i] = rng_unit(rng);
    for (size_t t = 256 + N_MERGES; t < N_TOKENS; t++)
        memset(v + t * (size_t)cols, 0, (size_t)cols * sizeof *v);
    const uint64_t ne[2] = { cols, rows };
    const int rc = ingot_gguf_add_f32(w, "token_embd.weight", type, 2, ne, v);
    free(v);
    return rc;
}

int fixture_write(const char *path, const fixture_spec *s, char *err, size_t errsz) {
    ingot_gguf_writer *w = ingot_gguf_writer_new();
    if (!w) { snprintf(err, errsz, "writer_new failed"); return -1; }

    uint64_t rng = s->seed ? s->seed : 1;
    const uint32_t q_dim = s->n_heads * s->head_dim;
    const uint32_t kv_dim = s->n_kv_heads * s->head_dim;
    int rc = 0;

    rc |= ingot_gguf_kv_string(w, "general.architecture", "qwen3");
    rc |= ingot_gguf_kv_string(w, "general.name", "mynah-slm synthetic fixture");
    rc |= ingot_gguf_kv_u32(w, "qwen3.block_count", s->n_layers);
    rc |= ingot_gguf_kv_u32(w, "qwen3.embedding_length", s->d_model);
    rc |= ingot_gguf_kv_u32(w, "qwen3.feed_forward_length", s->d_ff);
    rc |= ingot_gguf_kv_u32(w, "qwen3.attention.head_count", s->n_heads);
    rc |= ingot_gguf_kv_u32(w, "qwen3.attention.head_count_kv", s->n_kv_heads);
    rc |= ingot_gguf_kv_u32(w, "qwen3.attention.key_length", s->head_dim);
    rc |= ingot_gguf_kv_u32(w, "qwen3.attention.value_length", s->head_dim);
    rc |= ingot_gguf_kv_u32(w, "qwen3.context_length", s->n_ctx);
    rc |= ingot_gguf_kv_f32(w, "qwen3.attention.layer_norm_rms_epsilon", 1e-6f);
    rc |= ingot_gguf_kv_f32(w, "qwen3.rope.freq_base", 1e6f);

    uint32_t eos = 0;
    rc |= add_tokenizer(w, &eos);
    const uint64_t vocab = s->vocab_pad > N_TOKENS ? s->vocab_pad : N_TOKENS;

    /* Gains keep the residual stream O(1) through the layers: 1/sqrt(fan-in)
     * for the projections, a bit more for the embedding so the tied head
     * spreads its logits. The norms sit near 1, not at it, so a norm applied
     * to the wrong vector does not silently cancel. */
    const float g_in  = 1.0f / sqrtf((float)s->d_model);
    const float g_q   = 1.0f / sqrtf((float)q_dim);
    const float g_ff  = 1.0f / sqrtf((float)s->d_ff);

    rc |= add_embedding(w, pick_type(s, s->d_model, INGOT_TYPE_Q6_K), vocab,
                        s->d_model, &rng);
    rc |= add_matrix(w, "output_norm.weight", INGOT_TYPE_F32, 1, s->d_model, 0.1f, 1.0f, &rng);

    for (uint32_t l = 0; l < s->n_layers && rc == 0; l++) {
        char nm[96];
#define T(suffix) (snprintf(nm, sizeof nm, "blk.%u." suffix, l), nm)
        rc |= add_matrix(w, T("attn_norm.weight"), INGOT_TYPE_F32, 1, s->d_model, 0.1f, 1.0f, &rng);
        rc |= add_matrix(w, T("attn_q.weight"), pick_type(s, s->d_model, INGOT_TYPE_Q4_K),
                         q_dim, s->d_model, g_in, 0.0f, &rng);
        rc |= add_matrix(w, T("attn_k.weight"), pick_type(s, s->d_model, INGOT_TYPE_Q4_K),
                         kv_dim, s->d_model, g_in, 0.0f, &rng);
        /* Q8_0 here so the fixture carries every type a Q4_K_M-era file does:
         * Q4_K on our kernel, Q6_K and Q8_0 on ingot's. */
        rc |= add_matrix(w, T("attn_v.weight"), pick_type(s, s->d_model, INGOT_TYPE_Q8_0),
                         kv_dim, s->d_model, g_in, 0.0f, &rng);
        rc |= add_matrix(w, T("attn_output.weight"), pick_type(s, q_dim, INGOT_TYPE_Q4_K),
                         s->d_model, q_dim, g_q, 0.0f, &rng);
        rc |= add_matrix(w, T("attn_q_norm.weight"), INGOT_TYPE_F32, 1, s->head_dim, 0.1f, 1.0f, &rng);
        rc |= add_matrix(w, T("attn_k_norm.weight"), INGOT_TYPE_F32, 1, s->head_dim, 0.1f, 1.0f, &rng);
        rc |= add_matrix(w, T("ffn_norm.weight"), INGOT_TYPE_F32, 1, s->d_model, 0.1f, 1.0f, &rng);
        rc |= add_matrix(w, T("ffn_gate.weight"), pick_type(s, s->d_model, INGOT_TYPE_Q4_K),
                         s->d_ff, s->d_model, g_in, 0.0f, &rng);
        rc |= add_matrix(w, T("ffn_up.weight"), pick_type(s, s->d_model, INGOT_TYPE_Q4_K),
                         s->d_ff, s->d_model, g_in, 0.0f, &rng);
        rc |= add_matrix(w, T("ffn_down.weight"), pick_type(s, s->d_ff, INGOT_TYPE_Q6_K),
                         s->d_model, s->d_ff, g_ff, 0.0f, &rng);
#undef T
    }

    if (rc != 0) {
        snprintf(err, errsz, "building the fixture failed (writer refused a key or a tensor)");
        ingot_gguf_writer_free(w);
        return -1;
    }
    rc = ingot_gguf_writer_save(w, path, err, errsz);
    ingot_gguf_writer_free(w);
    return rc;
}

int fixture_write_temp(char *path, size_t pathsz, const fixture_spec *s,
                       char *err, size_t errsz) {
    const char *dir = getenv("TMPDIR");
    if (!dir || !*dir) dir = "/tmp";
    if ((size_t)snprintf(path, pathsz, "%s/mynah_slm_synth_XXXXXX", dir) >= pathsz) {
        snprintf(err, errsz, "temp path too long");
        return -1;
    }
    const int fd = mkstemp(path);
    if (fd < 0) { snprintf(err, errsz, "mkstemp %s failed", path); return -1; }
    close(fd);
    if (fixture_write(path, s, err, errsz) != 0) { unlink(path); return -1; }
    return 0;
}
