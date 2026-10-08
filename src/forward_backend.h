/* forward_backend.h — the Qwen3 forward pass driven through the backend vtable.
 *
 * WHY A SECOND DRIVER. arch_qwen3.c is the reference and stays the default
 * path; it calls the kernels directly on host memory. This module issues the
 * SAME op sequence through src/backend.h, so the buffers can live on a device:
 * weights uploaded once at create, the KV cache in backend memory (a sequence
 * of its own or a slot of a mynah_slm_bslots pool), every activation in
 * backend buffers, and exactly one host wait per token — the logits copy-back,
 * or the 4-byte argmax for greedy.
 *
 * With the CPU backend it IS arch_qwen3.c's arithmetic: every op of the CPU
 * backend is the function arch_qwen3.c calls, with the same arguments, in the
 * same order, so the logits are memcmp-identical to mynah_slm_seq_forward /
 * _forward_batch / mynah_slm_forward_multi (tests/test_forward_backend.c).
 * That is what makes the CPU backend a usable oracle for a device backend.
 *
 * SCOPE, refused at create rather than computed wrong: a hybrid family
 * (short-conv layers), a non-neutral embedding or logit scale (Granite), a
 * norm tensor that is not F32, a tensor type the backend has no kernel for, a
 * KV precision or head_dim the backend refuses. Everything numeric comes from
 * the model config — layers, widths, heads, head_dim, eps, theta, the RoPE
 * pairing, attention and residual scales.
 *
 * Not thread-safe: one forward at a time per mynah_slm_bfwd (its scratch is
 * shared), like a mynah_slm_state workspace.
 *
 * SPDX-License-Identifier: MIT */
#ifndef MYNAH_SLM_FORWARD_BACKEND_H
#define MYNAH_SLM_FORWARD_BACKEND_H

#include "backend.h"
#include "generate.h"
#include "model.h"

typedef struct mynah_slm_bfwd mynah_slm_bfwd;

typedef struct {
    /* Most positions any sequence run on it may hold: sizes the RoPE table.
     * 0 or more than the model's context = the model's context. */
    uint32_t ctx_cap;
    /* Widest prefill call (tokens). 0 = 256, clamped to ctx_cap. */
    uint32_t batch_max;
    /* Most sequences one mynah_slm_bfwd_multi step may carry. 0 = 1. */
    uint32_t dec_max;
    /* The precision every KV cache made for this forward is kept at. */
    mynah_slm_kv_type kv_k, kv_v;
} mynah_slm_bfwd_desc;

/* Uploads every weight the model uses (the tied embedding once), builds the
 * RoPE table, allocates the scratch. Returns 0; 1 when the model or the
 * backend is outside the scope above (`err` says what, so a caller can refuse
 * loudly); -1 on failure. The backend and the model must outlive it. */
int  mynah_slm_bfwd_create(mynah_slm_backend *b, const mynah_slm_model_t *m,
                           const mynah_slm_bfwd_desc *d, mynah_slm_bfwd **out,
                           char *err, size_t errsz);
void mynah_slm_bfwd_free(mynah_slm_bfwd *f);

mynah_slm_backend *mynah_slm_bfwd_backend(const mynah_slm_bfwd *f);
uint32_t mynah_slm_bfwd_batch_max(const mynah_slm_bfwd *f);
uint32_t mynah_slm_bfwd_dec_max(const mynah_slm_bfwd *f);
uint32_t mynah_slm_bfwd_ctx_cap(const mynah_slm_bfwd *f);
uint32_t mynah_slm_bfwd_vocab(const mynah_slm_bfwd *f);

/* The KV description for a cache of n_ctx positions (clamped to ctx_cap) on
 * this forward — what a mynah_slm_bslots pool must be created with. */
void mynah_slm_bfwd_kv_desc(const mynah_slm_bfwd *f, uint32_t n_ctx,
                            mynah_slm_bkv_desc *out);

/* ── one sequence ──────────────────────────────────────────────────────────
 * Its KV cache, and how far into it we are. Either OWNED (bseq_init allocates
 * one, bseq_free releases it) or BORROWED from a slot pool (bseq_bind; the
 * pool keeps it — bseq_free then only forgets it). */
typedef struct {
    mynah_slm_bkv *kv;
    uint32_t n_ctx;     /* positions the cache holds */
    uint32_t n_past;    /* positions already in it */
    int      owned;
} mynah_slm_bseq;

int  mynah_slm_bseq_init(mynah_slm_bfwd *f, mynah_slm_bseq *q, uint32_t n_ctx,
                         char *err, size_t errsz);
/* An empty sequence over `kv` (a slot's cache, made with bfwd_kv_desc). */
int  mynah_slm_bseq_bind(mynah_slm_bfwd *f, mynah_slm_bseq *q, mynah_slm_bkv *kv,
                         char *err, size_t errsz);
void mynah_slm_bseq_reset(mynah_slm_bseq *q);
void mynah_slm_bseq_free(mynah_slm_bfwd *f, mynah_slm_bseq *q);

/* ── the forward pass ──────────────────────────────────────────────────────
 * step      one token at position n_past. `logits` (HOST, vocab floats) gets
 *           the row — the one wait of the step — or NULL to only fill the
 *           cache, which then does not wait at all.
 * argmax    the same step, but only the index of the first maximum comes
 *           back: 4 bytes instead of a vocab-sized row (greedy decode).
 * prefill   n tokens at once (n <= batch_max; n == 1 is `step` verbatim), the
 *           batched product per weight; `logits` = the LAST token's, or NULL.
 * All three: -1 with `err` set, and n_past unchanged, on failure. */
int  mynah_slm_bfwd_step(mynah_slm_bfwd *f, mynah_slm_bseq *q, uint32_t token,
                         float *logits, char *err, size_t errsz);
int  mynah_slm_bfwd_step_argmax(mynah_slm_bfwd *f, mynah_slm_bseq *q, uint32_t token,
                                uint32_t *next, char *err, size_t errsz);
int  mynah_slm_bfwd_prefill(mynah_slm_bfwd *f, mynah_slm_bseq *q, const uint32_t *tokens,
                            uint32_t n, float *logits, char *err, size_t errsz);

/* ── as a generation driver (generate.h) ───────────────────────────────────
 * Lets generate.c's one token loop run on this forward: prefill in slices of
 * batch_max, then one step per token whose logits row lands in a host buffer
 * the forward owns. A failure's text is kept in `err` (the driver interface
 * only carries a status). `r` must outlive the driver. */
typedef struct {
    mynah_slm_bfwd *f;
    mynah_slm_bseq *q;
    char            err[256];
} mynah_slm_bfwd_run;

void mynah_slm_bfwd_driver(mynah_slm_bfwd_run *r, mynah_slm_gen_driver *out);

#endif /* MYNAH_SLM_FORWARD_BACKEND_H */
