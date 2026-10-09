/* backend_cuda.cu — the CUDA backend: device lifecycle, memory, weights, and
 * the vtable entries that launch gpu/cuda/kernels_cuda.cu.
 *
 * Built only by `make cuda` / `make cuda-test`. The default build never sees
 * this file and never needs nvcc; src/backend.c reaches it through
 * mynah_slm_backend_cuda_open() only when compiled with MYNAH_SLM_ENABLE_CUDA.
 *
 * The STRUCTURE is mynah-tts's gpu/cuda/backend_cuda.cu, cut down to what a
 * Qwen3 decode step needs:
 *   - ONE stream. Every op AND every copy is stream-ordered on it (no copy
 *     runs on the legacy default stream); only d2h, sync, argmax and the
 *     load-time uploads wait. One submitter (mynah-tts dropped a second stream after MPS showed
 *     no concurrency headroom, pocket-l40s-plateau L14-L16).
 *   - every runtime call goes through ce(), which reports the error AND
 *     clears a pending cudaErrorMemoryAllocation (see below);
 *   - weights uploaded once per distinct host pointer (the dedup itself is in
 *     src/backend.c, for every backend), freed at close;
 *   - no allocation and no free on the per-token path: cudaFree synchronises
 *     the device (mynah-tts pocket-cuda-slot-pool).
 * The kernels are not theirs — see kernels_cuda.h.
 *
 * SPDX-License-Identifier: MIT */
#include "kernels_cuda.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <new>

extern "C" {
#include "backend_ops.h"
#include "kernels.h"
}

#include "cuda_self_test.h"

namespace {

void set_err(char *e, size_t c, const char *m) {
    if (e && c > 0) std::snprintf(e, c, "%s", m);
}

/* A failed allocation returns cudaErrorMemoryAllocation AND records it as
 * the calling thread's last error. That record is not sticky — the context is
 * fine and the next allocation may succeed — but left unread it is what the
 * next launch check (cudaGetLastError) reports, failing an unrelated kernel
 * with "out of memory". So once the failure has been reported through its own
 * return code, that one record is cleared. Nothing else is: a sticky error
 * (illegal address, launch failure) cannot be cleared by a read anyway and
 * must stay visible to the next check. Same rule as mynah-tts ce(). */
void clear_alloc_error() {
    if (cudaPeekAtLastError() == cudaErrorMemoryAllocation) (void)cudaGetLastError();
}

int ce(cudaError_t r, char *e, size_t c, const char *what) {
    if (r == cudaSuccess) return 0;
    if (r == cudaErrorMemoryAllocation) clear_alloc_error();
    if (e && c > 0) std::snprintf(e, c, "CUDA %s: %s", what, cudaGetErrorString(r));
    return -1;
}

struct cuda_state {
    int          device;
    cudaStream_t stream;
    uint32_t    *d_tokens;      /* embed's token ids, on the device */
    size_t       tokens_cap;
    uint32_t    *d_argmax;      /* one index */
    uint32_t    *h_argmax;      /* pinned, so the 4-byte copy back is a real async DMA */
};

struct rope_table {
    float *cos_t, *sin_t;
};

struct kv_planes {
    uint16_t *k, *v;            /* [n_layers][n_ctx][kv_dim] bf16 */
};

/* The current device is per HOST THREAD, and cudaSetDevice ran only on the
 * thread that opened the backend. A server hands a request to whichever
 * worker is free, so every op selects the device first; otherwise an op from
 * another thread allocates, copies and launches on that thread's current
 * device (0 by default) — the wrong GPU, or a stream from another device.
 * cudaGetDevice is a thread-local read, so the common case costs no driver
 * call; cudaSetDevice only runs when the thread is pointed elsewhere. */
int on_device(const cuda_state *s, char *e, size_t c) {
    int cur = -1;
    if (cudaGetDevice(&cur) == cudaSuccess && cur == s->device) return 0;
    return ce(cudaSetDevice(s->device), e, c, "set device");
}

/* ── lifecycle ──────────────────────────────────────────────────────────── */

void cuda_close(void *st) {
    auto *s = static_cast<cuda_state *>(st);
    if (!s) return;
    (void)on_device(s, nullptr, 0);
    if (s->stream) cudaStreamSynchronize(s->stream);
    if (s->d_tokens) cudaFree(s->d_tokens);
    if (s->d_argmax) cudaFree(s->d_argmax);
    if (s->h_argmax) cudaFreeHost(s->h_argmax);
    if (s->stream) cudaStreamDestroy(s->stream);
    delete s;
}

/* ── memory ─────────────────────────────────────────────────────────────── */

float *cuda_alloc(void *st, size_t n, char *e, size_t c) {
    auto *s = static_cast<cuda_state *>(st);
    if (on_device(s, e, c)) return nullptr;
    void *p = nullptr;
    if (ce(cudaMalloc(&p, n * sizeof(float)), e, c, "alloc")) return nullptr;
    return static_cast<float *>(p);
}

/* Every release below drains the stream first. cudaFree happens to
 * synchronize the device today, but a kernel still queued that reads the
 * buffer is a use-after-free if it ever stops doing so (stream-ordered
 * allocators do not), and an explicit drain says what the code relies on.
 * None of these runs on the token path. */
void drain(cuda_state *s) {
    if (s->stream) (void)cudaStreamSynchronize(s->stream);
}

void cuda_free(void *st, float *p) {
    auto *s = static_cast<cuda_state *>(st);
    (void)on_device(s, nullptr, 0);
    drain(s);
    cudaFree(p);
}

/* Stream-ordered on s->stream, so the next kernel on the stream sees the data.
 * From pageable memory (every caller today) the call returns once the source
 * has been staged, so `src` may be reused at once; from PINNED memory it
 * returns before the DMA, and the source must stay untouched until a sync. */
int cuda_h2d(void *st, float *dst, const float *src, size_t n, char *e, size_t c) {
    auto *s = static_cast<cuda_state *>(st);
    if (on_device(s, e, c)) return -1;
    return ce(cudaMemcpyAsync(dst, src, n * sizeof(float), cudaMemcpyHostToDevice, s->stream),
              e, c, "h2d");
}

int cuda_d2h(void *st, float *dst, const float *src, size_t n, char *e, size_t c) {
    auto *s = static_cast<cuda_state *>(st);
    if (on_device(s, e, c)) return -1;
    if (ce(cudaMemcpyAsync(dst, src, n * sizeof(float), cudaMemcpyDeviceToHost, s->stream),
           e, c, "d2h"))
        return -1;
    /* The host reads dst as soon as this returns, so it has to have landed. */
    return ce(cudaStreamSynchronize(s->stream), e, c, "d2h sync");
}

int cuda_sync(void *st, char *e, size_t c) {
    auto *s = static_cast<cuda_state *>(st);
    if (on_device(s, e, c)) return -1;
    return ce(cudaStreamSynchronize(s->stream), e, c, "sync");
}

/* A load-time copy (weights, the RoPE table) that has fully landed when this
 * returns. NOT plain cudaMemcpy: that runs on the legacy default stream, which
 * does not order against our cudaStreamNonBlocking stream, and from pageable
 * memory it returns once the source is STAGED, before the DMA has written the
 * device buffer — so a kernel on s->stream could read it half-written. Here
 * the copy is on s->stream (ordered before every later kernel) and the stream
 * is drained before returning (load time: one sync per tensor is fine, and it
 * makes a failed DMA fail the upload rather than some later launch). */
int upload(cuda_state *s, void *dst, const void *src, size_t bytes, char *e, size_t c,
           const char *what) {
    if (ce(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, s->stream), e, c, what))
        return -1;
    return ce(cudaStreamSynchronize(s->stream), e, c, what);
}

/* ── weights ────────────────────────────────────────────────────────────── */

int cuda_weight_upload(void *st, mynah_slm_bweight *w, char *e, size_t c) {
    auto *s = static_cast<cuda_state *>(st);
    if (on_device(s, e, c)) return -1;
    if (!mynah_cuda::type_supported(w->type)) return 1;
    const size_t bytes = w->rows * w->row_bytes;
    void *d = nullptr;
    if (ce(cudaMalloc(&d, bytes), e, c, "weight alloc")) return -1;
    /* Model load, from pageable mmap: see upload() for why not cudaMemcpy. */
    if (upload(s, d, w->host, bytes, e, c, "weight upload")) {
        cudaFree(d);
        return -1;
    }
    w->data = d;
    return 0;
}

void cuda_weight_release(void *st, mynah_slm_bweight *w) {
    auto *s = static_cast<cuda_state *>(st);
    (void)on_device(s, nullptr, 0);
    drain(s);
    if (w->data) cudaFree(w->data);
    w->data = nullptr;
}

/* ── products ───────────────────────────────────────────────────────────── */

int cuda_matvec(void *st, const mynah_slm_bweight *w, const float *x, float *y,
                char *e, size_t c) {
    auto *s = static_cast<cuda_state *>(st);
    if (on_device(s, e, c)) return -1;
    return ce(mynah_cuda::launch_matvec(w->type, w->data, w->row_bytes, w->rows, w->cols,
                                        x, y, 1, s->stream), e, c, "matvec");
}

/* Correct, not fast: the GEMV kernel with tokens on grid.y re-reads the
 * weight once per token. A prefill-grade product (dequantize a strip, then a
 * tensor-core GEMM — the qmat.h policy on the device) is future work. */
int cuda_matmat(void *st, const mynah_slm_bweight *w, const float *x, float *y,
                size_t tokens, char *e, size_t c) {
    auto *s = static_cast<cuda_state *>(st);
    if (on_device(s, e, c)) return -1;
    if (tokens > 65535u) return 1;
    return ce(mynah_cuda::launch_matvec(w->type, w->data, w->row_bytes, w->rows, w->cols,
                                        x, y, tokens, s->stream), e, c, "matmat");
}

int cuda_embed(void *st, const mynah_slm_bweight *w, const uint32_t *tokens, size_t n,
               float *out, char *e, size_t c) {
    auto *s = static_cast<cuda_state *>(st);
    if (on_device(s, e, c)) return -1;
    /* The id buffer is sized at open for a full prefill batch; a wider call
     * would need an allocation on the token path, so it is refused instead. */
    if (n > s->tokens_cap) return 1;
    if (ce(cudaMemcpyAsync(s->d_tokens, tokens, n * sizeof(uint32_t),
                           cudaMemcpyHostToDevice, s->stream), e, c, "embed ids"))
        return -1;
    return ce(mynah_cuda::launch_embed(w->type, w->data, w->row_bytes, w->cols,
                                       s->d_tokens, n, out, s->stream), e, c, "embed");
}

/* ── norms and elementwise ──────────────────────────────────────────────── */

int cuda_rms_norm(void *st, float *out, const float *x, const mynah_slm_bweight *w,
                  size_t rows, uint32_t dim, float eps, char *e, size_t c) {
    auto *s = static_cast<cuda_state *>(st);
    if (on_device(s, e, c)) return -1;
    return ce(mynah_cuda::launch_rms_norm(out, x, static_cast<const float *>(w->data),
                                          rows, dim, eps, s->stream), e, c, "rms_norm");
}

int cuda_rms_norm_heads(void *st, float *x, const mynah_slm_bweight *w, size_t n_heads,
                        uint32_t head_dim, float eps, char *e, size_t c) {
    auto *s = static_cast<cuda_state *>(st);
    if (on_device(s, e, c)) return -1;
    return ce(mynah_cuda::launch_rms_norm_heads(x, static_cast<const float *>(w->data),
                                                n_heads, head_dim, eps, s->stream),
              e, c, "rms_norm_heads");
}

int cuda_swiglu(void *st, float *g, const float *u, size_t n, char *e, size_t c) {
    auto *s = static_cast<cuda_state *>(st);
    if (on_device(s, e, c)) return -1;
    return ce(mynah_cuda::launch_swiglu(g, u, n, s->stream), e, c, "swiglu");
}

int cuda_add(void *st, float *y, const float *x, size_t n, char *e, size_t c) {
    auto *s = static_cast<cuda_state *>(st);
    if (on_device(s, e, c)) return -1;
    return ce(mynah_cuda::launch_add(y, x, n, s->stream), e, c, "add");
}

int cuda_add_scaled(void *st, float *y, const float *x, float w, size_t n, char *e, size_t c) {
    auto *s = static_cast<cuda_state *>(st);
    if (on_device(s, e, c)) return -1;
    return ce(mynah_cuda::launch_add_scaled(y, x, w, n, s->stream), e, c, "add_scaled");
}

/* ── RoPE: the CPU builds the table, the device keeps a copy ────────────── */

int cuda_rope_create(void *st, mynah_slm_brope *r, char *e, size_t c) {
    auto *s = static_cast<cuda_state *>(st);
    if (on_device(s, e, c)) return -1;
    mynah_slm_rope host;
    if (mynah_slm_rope_init(&host, r->head_dim, r->max_pos, r->theta, r->interleaved) != 0) {
        set_err(e, c, "cannot build the rope table");
        return -1;
    }
    auto *t = new (std::nothrow) rope_table{nullptr, nullptr};
    const size_t bytes = (size_t)r->max_pos * (r->head_dim / 2) * sizeof(float);
    int rc = t ? 0 : -1;
    if (!t) set_err(e, c, "out of memory for the rope handle");
    if (rc == 0 && (ce(cudaMalloc(&t->cos_t, bytes), e, c, "rope alloc") ||
                    ce(cudaMalloc(&t->sin_t, bytes), e, c, "rope alloc") ||
                    upload(s, t->cos_t, host.cos, bytes, e, c, "rope upload") ||
                    upload(s, t->sin_t, host.sin, bytes, e, c, "rope upload")))
        rc = -1;
    mynah_slm_rope_free(&host);
    if (rc != 0) {
        if (t) { cudaFree(t->cos_t); cudaFree(t->sin_t); delete t; }
        return -1;
    }
    r->impl = t;
    return 0;
}

void cuda_rope_free(void *st, mynah_slm_brope *r) {
    auto *s = static_cast<cuda_state *>(st);
    (void)on_device(s, nullptr, 0);
    auto *t = static_cast<rope_table *>(r->impl);
    if (!t) return;
    drain(s);
    cudaFree(t->cos_t);
    cudaFree(t->sin_t);
    delete t;
    r->impl = nullptr;
}

int cuda_rope(void *st, const mynah_slm_brope *r, float *x, size_t n_tokens,
              uint32_t n_heads, uint32_t pos0, char *e, size_t c) {
    auto *s = static_cast<cuda_state *>(st);
    if (on_device(s, e, c)) return -1;
    auto *t = static_cast<const rope_table *>(r->impl);
    return ce(mynah_cuda::launch_rope(x, t->cos_t, t->sin_t, n_tokens, n_heads, r->head_dim,
                                      pos0, r->interleaved, s->stream), e, c, "rope");
}

/* ── KV: bf16 only, for now ─────────────────────────────────────────────── */

int cuda_kv_create(void *st, mynah_slm_bkv *kv, char *e, size_t c) {
    auto *s = static_cast<cuda_state *>(st);
    if (on_device(s, e, c)) return -1;
    const mynah_slm_bkv_desc &d = kv->desc;
    /* bf16 is what AGENTS.md asks of a KV cache; q8/fp8 on the device are a
     * quality question with their own gate, not a default to slip in here. */
    if (d.type_k != MYNAH_SLM_KV_BF16 || d.type_v != MYNAH_SLM_KV_BF16) return 1;
    if (!mynah_cuda::head_dim_supported(d.head_dim)) return 1;
    auto *k = new (std::nothrow) kv_planes{nullptr, nullptr};
    if (!k) { set_err(e, c, "out of memory for the KV handle"); return -1; }
    const size_t bytes = (size_t)d.n_layers * d.n_ctx * kv->kv_dim * sizeof(uint16_t);
    if (ce(cudaMalloc(&k->k, bytes), e, c, "kv alloc") ||
        ce(cudaMalloc(&k->v, bytes), e, c, "kv alloc")) {
        cudaFree(k->k);
        delete k;
        return -1;
    }
    kv->impl = k;
    return 0;
}

void cuda_kv_free(void *st, mynah_slm_bkv *kv) {
    auto *s = static_cast<cuda_state *>(st);
    (void)on_device(s, nullptr, 0);
    auto *k = static_cast<kv_planes *>(kv->impl);
    if (!k) return;
    drain(s);
    cudaFree(k->k);
    cudaFree(k->v);
    delete k;
    kv->impl = nullptr;
}

size_t kv_offset(const mynah_slm_bkv *kv, uint32_t layer, uint32_t pos) {
    return ((size_t)layer * kv->desc.n_ctx + pos) * kv->kv_dim;
}

int cuda_kv_append(void *st, mynah_slm_bkv *kv, uint32_t layer, uint32_t pos0, size_t n,
                   const float *k, const float *v, char *e, size_t c) {
    auto *s = static_cast<cuda_state *>(st);
    if (on_device(s, e, c)) return -1;
    auto *ck = static_cast<kv_planes *>(kv->impl);
    const size_t off = kv_offset(kv, layer, pos0);
    return ce(mynah_cuda::launch_kv_append_bf16(ck->k + off, ck->v + off, k, v, n, kv->kv_dim,
                                                s->stream), e, c, "kv_append");
}

int cuda_attention(void *st, const mynah_slm_bkv *kv, uint32_t layer, const float *q,
                   float *out, uint32_t pos0, size_t n_q, float scale, char *e, size_t c) {
    auto *s = static_cast<cuda_state *>(st);
    if (on_device(s, e, c)) return -1;
    auto *ck = static_cast<const kv_planes *>(kv->impl);
    const size_t off = kv_offset(kv, layer, 0);
    return ce(mynah_cuda::launch_attention_bf16(out, q, ck->k + off, ck->v + off, pos0, n_q,
                                                kv->desc.n_heads, kv->desc.n_kv_heads,
                                                kv->desc.head_dim, scale, s->stream),
              e, c, "attention");
}

/* The one op that waits by design: greedy decode is a 4-byte copy and one
 * sync per token, and that sync is the only one the step should have. */
int cuda_argmax(void *st, const float *x, size_t n, uint32_t *idx, char *e, size_t c) {
    auto *s = static_cast<cuda_state *>(st);
    if (on_device(s, e, c)) return -1;
    if (ce(mynah_cuda::launch_argmax(x, n, s->d_argmax, s->stream), e, c, "argmax") ||
        ce(cudaMemcpyAsync(s->h_argmax, s->d_argmax, sizeof(uint32_t), cudaMemcpyDeviceToHost,
                           s->stream), e, c, "argmax copy") ||
        ce(cudaStreamSynchronize(s->stream), e, c, "argmax sync"))
        return -1;
    *idx = *s->h_argmax;
    return 0;
}

/* ── slot-pool fences and per-request error recovery ───────────────────── */

int cuda_fence_create(void *st, void **fence, char *e, size_t c) {
    auto *s = static_cast<cuda_state *>(st);
    if (on_device(s, e, c)) return -1;
    cudaEvent_t ev = nullptr;
    /* Created once per slot at pool creation; timing off, it is only a marker. */
    if (ce(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming), e, c, "fence create")) return -1;
    *fence = ev;
    return 0;
}

void cuda_fence_destroy(void *st, void *fence) {
    auto *s = static_cast<cuda_state *>(st);
    (void)on_device(s, nullptr, 0);
    if (fence) cudaEventDestroy(static_cast<cudaEvent_t>(fence));
}

/* Behind everything queued so far: the step that was in flight for the row. */
int cuda_fence_record(void *st, void *fence, char *e, size_t c) {
    auto *s = static_cast<cuda_state *>(st);
    if (on_device(s, e, c)) return -1;
    return ce(cudaEventRecord(static_cast<cudaEvent_t>(fence), s->stream), e, c, "fence record");
}

/* Polled, never waited on. */
int cuda_fence_query(void *st, void *fence, char *e, size_t c) {
    auto *s = static_cast<cuda_state *>(st);
    if (on_device(s, e, c)) return -1;
    const cudaError_t r = cudaEventQuery(static_cast<cudaEvent_t>(fence));
    if (r == cudaSuccess) return 1;
    if (r == cudaErrorNotReady) {
        /* "Not yet" is an answer, not an error; never leave it as the
         * thread's last error for an unrelated launch check to report. */
        if (cudaPeekAtLastError() == cudaErrorNotReady) (void)cudaGetLastError();
        return 0;
    }
    return ce(r, e, c, "fence query");
}

/* Called after one request's op failed. cudaGetLastError clears a
 * non-sticky record (an allocation failure, a bad launch configuration);
 * a sticky one (illegal address, launch failure) survives the read and also
 * surfaces from the stream query — then the context is gone for everyone. */
int cuda_recover(void *st, char *e, size_t c) {
    auto *s = static_cast<cuda_state *>(st);
    if (on_device(s, e, c)) return -1;
    (void)cudaGetLastError();
    cudaError_t r = cudaPeekAtLastError();
    if (r == cudaSuccess) {
        r = cudaStreamQuery(s->stream);
        if (r == cudaErrorNotReady) {
            if (cudaPeekAtLastError() == cudaErrorNotReady) (void)cudaGetLastError();
            r = cudaSuccess;
        }
    }
    if (r == cudaSuccess) return 0;
    if (e && c > 0)
        std::snprintf(e, c, "CUDA context lost (sticky error, reopen the backend): %s",
                      cudaGetErrorString(r));
    return -1;
}

/* Widest embed() call served without an allocation: a prefill batch. */
constexpr size_t TOKENS_CAP = 4096;

int device_from_env(int count) {
    const char *v = std::getenv("MYNAH_SLM_CUDA_DEVICE");
    if (!v || !*v) return 0;
    char *end = nullptr;
    const long d = std::strtol(v, &end, 10);
    return (end && *end == '\0' && d >= 0 && d < count) ? (int)d : -1;
}

}  // namespace

extern "C" int mynah_slm_cuda_device_count(char *err, size_t errsz) {
    int n = 0;
    const cudaError_t r = cudaGetDeviceCount(&n);
    if (r != cudaSuccess) {
        /* "no CUDA-capable device" and "driver not installed" both land here;
         * neither leaves a sticky error behind, so clear the record. */
        (void)cudaGetLastError();
        if (err && errsz > 0) std::snprintf(err, errsz, "no CUDA device: %s", cudaGetErrorString(r));
        return 0;
    }
    if (n == 0 && err && errsz > 0) std::snprintf(err, errsz, "no CUDA device");
    return n;
}

extern "C" int mynah_slm_backend_cuda_open(mynah_slm_backend_ops *ops, void **state,
                                           char *err, size_t errsz) {
    const int count = mynah_slm_cuda_device_count(err, errsz);
    if (count <= 0) return -1;
    const int dev = device_from_env(count);
    if (dev < 0) {
        set_err(err, errsz, "MYNAH_SLM_CUDA_DEVICE is not a valid device index");
        return -1;
    }

    auto *s = new (std::nothrow) cuda_state{};
    if (!s) { set_err(err, errsz, "out of memory creating the cuda backend"); return -1; }
    s->device = dev;
    s->tokens_cap = TOKENS_CAP;
    void *dt = nullptr, *da = nullptr, *ha = nullptr;
    if (ce(cudaSetDevice(dev), err, errsz, "set device") ||
        ce(cudaStreamCreateWithFlags(&s->stream, cudaStreamNonBlocking), err, errsz, "stream") ||
        ce(cudaMalloc(&dt, TOKENS_CAP * sizeof(uint32_t)), err, errsz, "alloc") ||
        ce(cudaMalloc(&da, sizeof(uint32_t)), err, errsz, "alloc") ||
        ce(cudaHostAlloc(&ha, sizeof(uint32_t), cudaHostAllocDefault), err, errsz, "pinned alloc")) {
        s->d_tokens = static_cast<uint32_t *>(dt);
        s->d_argmax = static_cast<uint32_t *>(da);
        s->h_argmax = static_cast<uint32_t *>(ha);
        cuda_close(s);
        return -1;
    }
    s->d_tokens = static_cast<uint32_t *>(dt);
    s->d_argmax = static_cast<uint32_t *>(da);
    s->h_argmax = static_cast<uint32_t *>(ha);

    std::memset(ops, 0, sizeof *ops);
    ops->name           = "cuda";
    ops->close          = cuda_close;
    ops->alloc          = cuda_alloc;
    ops->free           = cuda_free;
    ops->h2d            = cuda_h2d;
    ops->d2h            = cuda_d2h;
    ops->sync           = cuda_sync;
    ops->weight_upload  = cuda_weight_upload;
    ops->weight_release = cuda_weight_release;
    ops->matvec         = cuda_matvec;
    ops->matmat         = cuda_matmat;
    ops->embed          = cuda_embed;
    ops->rms_norm       = cuda_rms_norm;
    ops->rms_norm_heads = cuda_rms_norm_heads;
    ops->swiglu         = cuda_swiglu;
    ops->add            = cuda_add;
    ops->add_scaled     = cuda_add_scaled;
    ops->rope_create    = cuda_rope_create;
    ops->rope_free      = cuda_rope_free;
    ops->rope           = cuda_rope;
    ops->kv_create      = cuda_kv_create;
    ops->kv_free        = cuda_kv_free;
    ops->kv_append      = cuda_kv_append;
    ops->attention      = cuda_attention;
    ops->argmax         = cuda_argmax;
    ops->fence_create   = cuda_fence_create;
    ops->fence_destroy  = cuda_fence_destroy;
    ops->fence_record   = cuda_fence_record;
    ops->fence_query    = cuda_fence_query;
    ops->recover        = cuda_recover;
    *state = s;
    return 0;
}
