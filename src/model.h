/* model.h — the loaded checkpoint: config plus bound tensors.
 *
 * Everything here is architecture-agnostic. The per-architecture forward pass
 * lives behind the vtable at the bottom, so arch_qwen3.c and arch_gemma4.c
 * never #ifdef each other and adding a family touches no shared code.
 *
 * Tensors are kept as ingot descriptors, not as dequantized buffers: the whole
 * point of reading GGUF through ingot is multiplying straight off the stored
 * quantized bytes. Dequantization, where it happens at all, is a decision the
 * kernel makes per call.
 *
 * SPDX-License-Identifier: MIT */
#ifndef MYNAH_SLM_MODEL_H
#define MYNAH_SLM_MODEL_H

#include <stddef.h>
#include <stdint.h>

#include "ingot/gguf.h"

#define MYNAH_SLM_MAX_EOS 8

/* What operator a layer runs where attention would normally be.
 *
 * Every family shipped before LFM2 is homogeneous — 30 attention layers, or
 * 28, and the question never came up. LFM2 is 22 short-conv layers and 8
 * attention layers interleaved on an irregular pattern, so "which operator"
 * becomes per-layer data read from the file, exactly like every other
 * architecture constant (rule 1). See docs/lfm2-arch.md. */
typedef enum {
    MYNAH_SLM_OP_ATTN = 0,
    MYNAH_SLM_OP_SHORTCONV,
} mynah_slm_op;

/* Architecture config, read from GGUF metadata. No value here is a #define:
 * every field comes from the file, and the metadata key prefix comes from
 * general.architecture rather than being hardcoded per family. */
typedef struct {
    char     arch[32];
    uint32_t n_layers;
    uint32_t d_model;
    uint32_t d_ff;
    uint32_t n_heads;
    uint32_t n_kv_heads;
    /* Its own key (attention.key_length), NOT d_model / n_heads. For
     * Qwen3-0.6B those give 128 and 64 respectively, and deriving it is the
     * classic way to get this family wrong. */
    uint32_t head_dim;
    uint32_t n_ctx;
    uint32_t vocab_size;
    float    rms_eps;
    float    rope_theta;

    /* muP-style scalars. Granite carries all four in its GGUF; families that
     * do not use them get neutral values, so there is one forward pass rather
     * than one per family.
     *
     * They are not decoration: attn_scale is 0.015625 for Granite-350m where
     * 1/sqrt(head_dim) would be 0.125 — eight times larger. A model run with
     * the wrong one of those produces fluent text and wrong tokens. */
    float    attn_scale;      /* the softmax scale; default 1/sqrt(head_dim) */
    float    embed_scale;     /* embeddings are multiplied by this */
    float    residual_scale;  /* each residual branch is multiplied by this */
    float    logit_scale;     /* logits are DIVIDED by this */

    /* Which RoPE pairing the STORED weights expect.
     *
     * 0 = NeoX split-half: element i pairs with i + head_dim/2 (Qwen3, Gemma).
     * 1 = interleaved: 2i pairs with 2i+1 (Granite, Llama).
     *
     * This is not a property of the architecture, it is a property of the
     * FILE. llama.cpp's converter permutes q and k for Llama-family models so
     * that the interleaved form applies, and Granite inherits that path. Get
     * it wrong and the model still produces fluent English — perplexity 412
     * against 65 on the same text, which is the whole difference between
     * "a weak 350M" and "our bug". */
    int      rope_interleaved;

    /* A GGUF carries one terminator; generation_config.json can carry more.
     * Plural from the start so the second one is not an afterthought. */
    uint32_t eos[MYNAH_SLM_MAX_EOS];
    size_t   n_eos;

    /* ── per-layer operator, for hybrid families ────────────────────────────
     * `layer_op[i]` is what layer i runs; `kv_slot[i]` is which KV cache it
     * uses, or MYNAH_SLM_NO_KV when it holds none. Both are n_layers long and
     * owned by the config.
     *
     * A short-conv layer carries no KV cache at all, so allocating one per
     * layer would waste 22 of 30 for LFM2. The slot indirection is what lets
     * the cache be n_attn_layers deep while the forward pass still indexes by
     * layer number. For a homogeneous family layer_op is all-attention and
     * kv_slot is the identity, which is the shape every existing caller
     * already assumes. */
    uint8_t  *layer_op;        /* mynah_slm_op per layer */
    uint32_t *kv_slot;
    uint32_t  n_attn_layers;   /* how many layers actually hold a cache */

    /* Depthwise causal FIR length for the short-conv layers (lfm2's
     * `shortconv.l_cache`). 0 when the family has no conv layers. */
    uint32_t  conv_taps;

    /* Derived, cached because every layer needs them. */
    uint32_t q_dim;    /* n_heads    * head_dim */
    uint32_t kv_dim;   /* n_kv_heads * head_dim */
} mynah_slm_config;

#define MYNAH_SLM_NO_KV 0xFFFFFFFFu

/* One decoder block. NULL is legal for optional tensors (q_norm/k_norm exist
 * in Qwen3, not in every family), so the loader distinguishes "absent" from
 * "missing" per architecture rather than globally. */
typedef struct {
    /* The pre-operator norm. GGUF calls it attn_norm on EVERY layer, including
     * the 22 LFM2 layers that have no attention — upstream it is honestly
     * `operator_norm`. Never infer the layer kind from this tensor's presence
     * (docs/lfm2-arch.md); cfg.layer_op is the only authority. */
    const ingot_tensor *attn_norm;
    const ingot_tensor *wq, *wk, *wv, *wo;
    const ingot_tensor *q_norm, *k_norm;
    const ingot_tensor *ffn_norm;
    const ingot_tensor *gate, *up, *down;

    /* Short-conv layers only, NULL otherwise. in_proj is [3*d_model, d_model]
     * producing (B, C, x); conv_w is [d_model, conv_taps], one filter per
     * channel; out_proj is [d_model, d_model]. */
    const ingot_tensor *conv_in, *conv_w, *conv_out;
} mynah_slm_layer;

struct mynah_slm_model {
    ingot_gguf        *gguf;
    mynah_slm_config   cfg;

    const ingot_tensor *embed;      /* token_embd.weight */
    const ingot_tensor *out_norm;   /* output_norm.weight */
    /* Separate LM head when the checkpoint has one. NULL means tied, and the
     * embedding matrix doubles as the output projection — which is the Qwen3
     * case, and the reason token_embd is a hot GEMV and not just a lookup. */
    const ingot_tensor *lm_head;

    mynah_slm_layer   *layers;
};

typedef struct mynah_slm_model mynah_slm_model_t;

/* Zero-copy pointer to a tensor's stored bytes. */
const void *mynah_slm_tensor_data(const mynah_slm_model_t *m, const ingot_tensor *t);

/* The full config. Internal: the public header exposes accessors instead, so
 * this struct can change without breaking an installed mynah_slm.h. */
const mynah_slm_config *mynah_slm_model_config(const mynah_slm_model_t *m);

#endif /* MYNAH_SLM_MODEL_H */
