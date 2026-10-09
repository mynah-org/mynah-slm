/* fixture_model.h — a synthetic Qwen3 checkpoint, written in-tree.
 *
 * No checkpoint is reachable from CI, so every test that needs a forward pass
 * used to SKIP there. This writes one: the real architecture keys, the real
 * tensor names, deterministic pseudo-random weights, and a byte-level BPE
 * vocabulary small enough to read — the 256 byte symbols, a handful of merges
 * and the Qwen3 control tokens — so the loader, the tokenizer, the chat
 * template and the detokenizer all run unchanged on it.
 *
 * It proves the ENGINE agrees with itself (batched vs one token, a sequence in
 * a batch vs the same sequence alone). It says nothing about text quality:
 * the weights are noise, and so is the text.
 *
 * SPDX-License-Identifier: MIT */
#ifndef MYNAH_SLM_FIXTURE_MODEL_H
#define MYNAH_SLM_FIXTURE_MODEL_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    uint32_t n_layers, d_model, d_ff, n_heads, n_kv_heads, head_dim, n_ctx;
    /* Embedding rows beyond the tokenizer, like Qwen3's 151669 -> 151936
     * padding. 0 = exactly the tokenizer's vocabulary. */
    uint32_t vocab_pad;
    /* 0: every matrix F32. 1: the Q4_K_M recipe's type mix — Q4_K for
     * q/k/o/gate/up, Q8_0 for v, Q6_K for ffn_down and the tied embedding,
     * F32 norms (and F32 for any row that is not whole 256-blocks). */
    int      quant;
    uint64_t seed;
    /* Write a separate output.weight instead of tying the head to the
     * embedding. With a tied head and noise layers the residual stream stays
     * close to the input token's own embedding, so its self-logit (~|h||E|,
     * hundreds) beats every other (~tens) and the model echoes its input
     * forever — every answer is the prompt's last token repeated. */
    int      untied;
} fixture_spec;

/* d_model 256, 2 layers, 8 query heads / 2 KV heads of 64, d_ff 512, n_ctx 512.
 * 256 and not 64 because a Q4_K block is 256 wide: a fixture that cannot hold
 * a Q4_K row cannot run our own Q4_K matvec. */
void fixture_spec_tiny(fixture_spec *s, int quant);

/* Slow enough to cancel: 0.6B-wide layers (1024/3072, 8 query / 4 KV heads of
 * 128) but only 4 of them, the ~280-token vocabulary, 8192 positions. A few
 * ms per token on a laptop, so an 8000-token generation is tens of seconds —
 * long enough that a request that keeps running after its client left is
 * unmistakable — and ~25 MB, written in seconds. tests/test_server_cancel.sh. */
void fixture_spec_slow(fixture_spec *s);

/* The 0.6B GEOMETRY (28 layers, 1024/3072, 16/8 heads of 128, 151936 rows)
 * with noise for weights. ~400 MB at quant=1 and slow to write: for a bench
 * that needs real tensor shapes, never for `make test`. */
void fixture_spec_06b_shape(fixture_spec *s);

/* Writes the GGUF. Returns 0, or -1 with the reason in err. */
int fixture_write(const char *path, const fixture_spec *s, char *err, size_t errsz);

/* mkstemp + fixture_write. `path` receives the file name (>= 64 bytes); the
 * caller unlinks it. Returns 0 or -1. */
int fixture_write_temp(char *path, size_t pathsz, const fixture_spec *s,
                       char *err, size_t errsz);

#endif /* MYNAH_SLM_FIXTURE_MODEL_H */
