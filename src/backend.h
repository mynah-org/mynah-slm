/* backend.h — the compute boundary a decode step can run behind.
 *
 * WHY THIS EXISTS. The CPU path is the reference, and it stays one: every op
 * here has a CPU implementation that calls the kernels arch_qwen3.c already
 * calls, so the CPU backend is the same arithmetic, not a second definition of
 * the model. What the boundary buys is that a GPU can implement the SAME op
 * list on device-resident buffers, behind `make cuda`, without a CUDA type
 * ever appearing in this header and without the default build needing nvcc.
 * The structure is mynah-tts's (src/backend.{c,h} there); the kernels behind
 * it are not — see .work/cuda-backend.md.
 *
 * THE RETURN CONTRACT, the same for every op:
 *     0   done
 *     1   UNSUPPORTED — this backend has no implementation (its entry is
 *         NULL), or refuses this type / shape. Nothing was touched, no error
 *         text is set, and the caller decides what a fallback means.
 *    -1   failed; `err` says why.
 * NULL means unsupported, never "quietly do it on the CPU". A silent host
 * detour inside a device-resident step is two copies and a sync per op, which
 * is how a GPU ends up idle 40% of the time (mynah-tts pocket-l40s-plateau).
 *
 * BUFFERS. Activations are `float *` returned by mynah_slm_backend_alloc and
 * only ever dereferenced by this backend's own ops: host memory on the CPU,
 * device memory on CUDA. Data crosses with h2d / d2h. Ops are stream-ordered;
 * on a device they return before the work is done, and only d2h, sync and
 * argmax wait for it.
 *
 * WEIGHTS are uploaded ONCE, by their stored GGUF bytes, and referred to by an
 * opaque handle afterwards. The backend owns every handle and frees it at
 * close. Uploading the same bytes twice returns the same handle — which is
 * what keeps a tied embedding (input lookup AND LM head) to one device copy.
 *
 * Single submitter: a backend is driven by one thread at a time.
 *
 * SPDX-License-Identifier: MIT */
#ifndef MYNAH_SLM_BACKEND_H
#define MYNAH_SLM_BACKEND_H

#include <stddef.h>
#include <stdint.h>

#include "kvcache.h"

typedef enum {
    MYNAH_SLM_DEVICE_CPU  = 0,
    MYNAH_SLM_DEVICE_CUDA = 1,
} mynah_slm_device;

typedef struct mynah_slm_backend mynah_slm_backend;
typedef struct mynah_slm_bweight mynah_slm_bweight;  /* an uploaded weight   */
typedef struct mynah_slm_brope   mynah_slm_brope;    /* a RoPE cos/sin table */
typedef struct mynah_slm_bkv     mynah_slm_bkv;      /* one sequence's KV    */

/* Every op, for mynah_slm_backend_has(): a dispatch report can then say which
 * ones this backend lacks instead of discovering it mid-step. */
typedef enum {
    MYNAH_SLM_BOP_MATVEC = 0,
    MYNAH_SLM_BOP_MATMAT,
    MYNAH_SLM_BOP_EMBED,
    MYNAH_SLM_BOP_RMS_NORM,
    MYNAH_SLM_BOP_RMS_NORM_HEADS,
    MYNAH_SLM_BOP_ROPE,
    MYNAH_SLM_BOP_SWIGLU,
    MYNAH_SLM_BOP_ADD,
    MYNAH_SLM_BOP_ADD_SCALED,
    MYNAH_SLM_BOP_KV_APPEND,
    MYNAH_SLM_BOP_ATTENTION,
    MYNAH_SLM_BOP_ARGMAX,
    MYNAH_SLM_BOP_COUNT
} mynah_slm_bop;

const char *mynah_slm_device_name(mynah_slm_device d);
const char *mynah_slm_bop_name(mynah_slm_bop op);

/* ── lifecycle ──────────────────────────────────────────────────────────────
 * CUDA in a build without `make cuda` fails with a message that says so; it
 * never falls back to the CPU behind the caller's back. */
int  mynah_slm_backend_open(mynah_slm_device device, mynah_slm_backend **out,
                            char *err, size_t errsz);
void mynah_slm_backend_close(mynah_slm_backend *b);
const char       *mynah_slm_backend_name(const mynah_slm_backend *b);
mynah_slm_device  mynah_slm_backend_device(const mynah_slm_backend *b);
int               mynah_slm_backend_has(const mynah_slm_backend *b, mynah_slm_bop op);

/* ── memory ─────────────────────────────────────────────────────────────────
 * Counts are in floats. alloc returns NULL with `err` set on failure. Never
 * call alloc or free inside the token loop: on a device, a free synchronises
 * the whole GPU. */
float *mynah_slm_backend_alloc(mynah_slm_backend *b, size_t n, char *err, size_t errsz);
void   mynah_slm_backend_free(mynah_slm_backend *b, float *p);
int    mynah_slm_backend_h2d(mynah_slm_backend *b, float *dst, const float *src,
                             size_t n, char *err, size_t errsz);
int    mynah_slm_backend_d2h(mynah_slm_backend *b, float *dst, const float *src,
                             size_t n, char *err, size_t errsz);
int    mynah_slm_backend_sync(mynah_slm_backend *b, char *err, size_t errsz);

/* ── weights ────────────────────────────────────────────────────────────────
 * `data` is the row-major [rows][cols] block matrix exactly as stored in the
 * GGUF (`type` is its ggml type id), and must stay valid for the backend's
 * life: the CPU backend keeps the pointer rather than a copy. A norm weight is
 * an F32 tensor with rows = 1. Returns 1 when this backend has no kernel for
 * `type`. */
int mynah_slm_backend_weight(mynah_slm_backend *b, int type, const void *data,
                             size_t rows, size_t cols,
                             const mynah_slm_bweight **out, char *err, size_t errsz);
int    mynah_slm_bweight_type(const mynah_slm_bweight *w);
size_t mynah_slm_bweight_rows(const mynah_slm_bweight *w);
size_t mynah_slm_bweight_cols(const mynah_slm_bweight *w);

/* ── products ───────────────────────────────────────────────────────────────
 *   matvec   y[rows]          = W · x[cols]                  one token
 *   matmat   y[tokens][rows]  = x[tokens][cols] · W^T        a prefill batch
 *   embed    out[n][cols]     = W[tokens[i]]                 rows decoded to f32
 * `tokens` is a HOST array. */
int mynah_slm_backend_matvec(mynah_slm_backend *b, const mynah_slm_bweight *w,
                             const float *x, float *y, char *err, size_t errsz);
int mynah_slm_backend_matmat(mynah_slm_backend *b, const mynah_slm_bweight *w,
                             const float *x, float *y, size_t tokens,
                             char *err, size_t errsz);
int mynah_slm_backend_embed(mynah_slm_backend *b, const mynah_slm_bweight *w,
                            const uint32_t *tokens, size_t n, float *out,
                            char *err, size_t errsz);

/* ── norms and elementwise ──────────────────────────────────────────────────
 *   rms_norm        out[r] = x[r] / rms(x[r]) * w, for `rows` rows of `dim`
 *   rms_norm_heads  the same per head, IN PLACE, on `n_heads` consecutive
 *                   head_dim-wide slices sharing one head_dim-long weight —
 *                   Qwen3's QK-norm. Pass tokens * heads for a batch.
 *   swiglu          gate = silu(gate) * up
 *   add             y += x
 *   add_scaled      y += w * x */
int mynah_slm_backend_rms_norm(mynah_slm_backend *b, float *out, const float *x,
                               const mynah_slm_bweight *w, size_t rows,
                               uint32_t dim, float eps, char *err, size_t errsz);
int mynah_slm_backend_rms_norm_heads(mynah_slm_backend *b, float *x,
                                     const mynah_slm_bweight *w, size_t n_heads,
                                     uint32_t head_dim, float eps,
                                     char *err, size_t errsz);
int mynah_slm_backend_swiglu(mynah_slm_backend *b, float *gate, const float *up,
                             size_t n, char *err, size_t errsz);
int mynah_slm_backend_add(mynah_slm_backend *b, float *y, const float *x,
                          size_t n, char *err, size_t errsz);
int mynah_slm_backend_add_scaled(mynah_slm_backend *b, float *y, const float *x,
                                 float w, size_t n, char *err, size_t errsz);

/* ── RoPE ───────────────────────────────────────────────────────────────────
 * A table is built once per model (theta from the config, never a constant).
 * rope() rotates x[n_tokens][n_heads * head_dim] in place, row t at absolute
 * position pos0 + t. `interleaved` = 0 is NeoX split-half (Qwen3), 1 pairs
 * 2i with 2i+1 (Granite) — see kernels.h for why both exist and why the wrong
 * one still produces fluent text. A backend may refuse one form (returns 1). */
int  mynah_slm_backend_rope_create(mynah_slm_backend *b, uint32_t head_dim,
                                   uint32_t max_pos, float theta, int interleaved,
                                   mynah_slm_brope **out, char *err, size_t errsz);
void mynah_slm_backend_rope_free(mynah_slm_backend *b, mynah_slm_brope *r);
int  mynah_slm_backend_rope(mynah_slm_backend *b, const mynah_slm_brope *r,
                            float *x, size_t n_tokens, uint32_t n_heads,
                            uint32_t pos0, char *err, size_t errsz);

/* ── KV cache and attention ─────────────────────────────────────────────────
 * One cache per sequence, [n_layers][n_ctx][n_kv_heads * head_dim] for K and
 * for V at the precisions requested (the CPU takes every mynah_slm_kv_type; a
 * device backend may refuse some with 1). `max_batch` sizes the scratch for
 * the widest attention() call, so nothing is allocated in the token loop.
 *
 *   kv_append   store n rows of K and V ([n][kv_dim] buffers, post-RoPE) at
 *               positions pos0 .. pos0+n-1 of `layer`
 *   attention   causal GQA: query row t (absolute position pos0 + t) reads
 *               positions 0 .. pos0+t. q and out are [n_q][n_heads*head_dim];
 *               kv head = q head / (n_heads / n_kv_heads). `scale` is a
 *               parameter, never 1/sqrt(head_dim) computed in here. */
typedef struct {
    mynah_slm_kv_type type_k, type_v;
    uint32_t n_layers, n_ctx;
    uint32_t n_heads, n_kv_heads, head_dim;
    uint32_t max_batch;
} mynah_slm_bkv_desc;

int  mynah_slm_backend_kv_create(mynah_slm_backend *b, const mynah_slm_bkv_desc *d,
                                 mynah_slm_bkv **out, char *err, size_t errsz);
void mynah_slm_backend_kv_free(mynah_slm_backend *b, mynah_slm_bkv *kv);
int  mynah_slm_backend_kv_append(mynah_slm_backend *b, mynah_slm_bkv *kv,
                                 uint32_t layer, uint32_t pos0, size_t n,
                                 const float *k, const float *v,
                                 char *err, size_t errsz);
int  mynah_slm_backend_attention(mynah_slm_backend *b, const mynah_slm_bkv *kv,
                                 uint32_t layer, const float *q, float *out,
                                 uint32_t pos0, size_t n_q, float scale,
                                 char *err, size_t errsz);

/* ── sampling hand-off ──────────────────────────────────────────────────────
 * Index of the first maximum of x[n], delivered to the HOST. On a device this
 * is the op that waits: greedy decode is then one 4-byte copy per token. For
 * anything but greedy, d2h the logits row instead. */
int mynah_slm_backend_argmax(mynah_slm_backend *b, const float *x, size_t n,
                             uint32_t *idx, char *err, size_t errsz);

/* ── per-request state: a slot pool ─────────────────────────────────────────
 * Every request's device state — its KV cache and its scratch — lives in a
 * SLOT, and every slot is allocated when the pool is created (model load),
 * never on the request path. Admission takes a slot; retirement — finished,
 * failed, or CANCELLED because the client went away — gives it back.
 *
 * WHY: cudaFree and cudaFreeHost synchronize the whole device. Freeing a
 * request's memory at retirement stalls every other live request for that
 * long, and allocating at admission costs 11-30 ms (mynah-tts
 * pocket-cuda-slot-pool.md: 11-30 ms -> 0.05 ms with a pool).
 *
 * RELEASE NEVER WAITS. The device may still be running the step that was in
 * flight when the scheduler decided to drop the row. Release records a fence
 * on the stream behind that work and returns at once; the slot is PARKED,
 * and acquire hands it out again only once its fence has passed (polled,
 * never waited on). Until then acquire picks another slot, or returns 1 —
 * the admission ladder's "busy", not an error.
 *
 * The contract for a cancelled row, which the scheduler (not the backend)
 * enforces at step boundaries:
 *   - at most the step already submitted finishes for it; its outputs are
 *     dropped, recognised by the slot GENERATION, which release bumps — a
 *     result tagged with an older generation belongs to a retired request;
 *   - the next step's batch simply does not include it: per-row tables
 *     (slot -> KV base, position) are rebuilt per step, so nothing about
 *     the row survives in them;
 *   - inside a captured graph the row becomes an inert pad row, or the
 *     batch drops to a smaller width bucket — never a re-capture on the hot
 *     path (.work/cuda-backend.md).
 * A KV cache is not cleared on reuse: attention only reads positions the new
 * request wrote (kv_append before attention, always). */
typedef struct mynah_slm_bslots mynah_slm_bslots;

typedef struct {
    mynah_slm_bkv_desc kv;        /* every slot's cache */
    size_t   scratch;             /* floats of per-slot scratch, may be 0 */
    uint32_t n_slots;
} mynah_slm_bslots_desc;

/* Allocates every slot. The ONLY allocation the pool ever makes. */
int  mynah_slm_backend_slots_create(mynah_slm_backend *b, const mynah_slm_bslots_desc *d,
                                    mynah_slm_bslots **out, char *err, size_t errsz);
/* Frees every slot; may synchronize. Shutdown only. */
void mynah_slm_backend_slots_destroy(mynah_slm_backend *b, mynah_slm_bslots *p);

/* 0 and a free, quiescent slot; 1 when none is (all busy, or parked behind a
 * fence still in flight) — retry at a later step boundary; -1 on error.
 * Never allocates, never waits. */
int  mynah_slm_backend_slot_acquire(mynah_slm_backend *b, mynah_slm_bslots *p,
                                    uint32_t *slot, uint64_t *generation,
                                    char *err, size_t errsz);
/* Park the slot behind a fence on the stream. Never frees, never waits.
 * -1 for a slot that is not held (a double release is a scheduler bug). */
int  mynah_slm_backend_slot_release(mynah_slm_backend *b, mynah_slm_bslots *p,
                                    uint32_t slot, char *err, size_t errsz);

mynah_slm_bkv *mynah_slm_backend_slot_kv(mynah_slm_bslots *p, uint32_t slot);
float         *mynah_slm_backend_slot_scratch(mynah_slm_bslots *p, uint32_t slot);
uint64_t       mynah_slm_backend_slot_generation(const mynah_slm_bslots *p, uint32_t slot);
uint32_t       mynah_slm_backend_slots_held(const mynah_slm_bslots *p);

/* ── errors that belong to one request ──────────────────────────────────────
 * After an op on one request fails, ask whether the BACKEND survived:
 *    0   recoverable — the error record is cleared (e.g. an allocation
 *        failure, an invalid launch configuration); retire that request's
 *        slot and keep serving the others;
 *   -1   the device context is poisoned (a sticky error: illegal address,
 *        launch failure). No read can clear it; every request on this
 *        backend fails and the backend must be closed and reopened.
 * A synchronous backend has nothing pending and always returns 0. */
int mynah_slm_backend_recover(mynah_slm_backend *b, char *err, size_t errsz);

#endif /* MYNAH_SLM_BACKEND_H */
