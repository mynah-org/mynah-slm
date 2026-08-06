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

    /* A GGUF carries one terminator; generation_config.json can carry more.
     * Plural from the start so the second one is not an afterthought. */
    uint32_t eos[MYNAH_SLM_MAX_EOS];
    size_t   n_eos;

    /* Derived, cached because every layer needs them. */
    uint32_t q_dim;    /* n_heads    * head_dim */
    uint32_t kv_dim;   /* n_kv_heads * head_dim */
} mynah_slm_config;

/* One decoder block. NULL is legal for optional tensors (q_norm/k_norm exist
 * in Qwen3, not in every family), so the loader distinguishes "absent" from
 * "missing" per architecture rather than globally. */
typedef struct {
    const ingot_tensor *attn_norm;
    const ingot_tensor *wq, *wk, *wv, *wo;
    const ingot_tensor *q_norm, *k_norm;
    const ingot_tensor *ffn_norm;
    const ingot_tensor *gate, *up, *down;
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

#endif /* MYNAH_SLM_MODEL_H */
