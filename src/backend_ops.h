/* backend_ops.h — what a backend implements. Internal: callers use backend.h.
 *
 * A table of function pointers plus one opaque state pointer, filled by the
 * backend's open function. Any entry may be NULL, and NULL means UNSUPPORTED:
 * the wrapper in backend.c returns 1 for it without calling anything. That is
 * the whole fallback policy, and it belongs to the caller, not to the backend.
 *
 * C and C++ both include this (gpu/cuda/backend_cuda.cu fills the same table),
 * so it holds no CUDA type and nothing C++ cannot parse.
 *
 * SPDX-License-Identifier: MIT */
#ifndef MYNAH_SLM_BACKEND_OPS_H
#define MYNAH_SLM_BACKEND_OPS_H

#include "backend.h"

#ifdef __cplusplus
extern "C" {
#endif

/* The handles' common part. The backend fills `data` / `impl`; the wrapper
 * owns the rest, so validation lives in one place for every backend. */
struct mynah_slm_bweight {
    int         type;        /* ggml type id */
    size_t      rows, cols;
    size_t      row_bytes;   /* one stored row */
    const void *host;        /* the bytes it was uploaded from — the cache key */
    void       *data;        /* what the kernels read: == host on the CPU */
};

struct mynah_slm_brope {
    uint32_t head_dim, max_pos;
    float    theta;
    int      interleaved;
    void    *impl;
};

struct mynah_slm_bkv {
    mynah_slm_bkv_desc desc;
    uint32_t kv_dim;         /* n_kv_heads * head_dim */
    void    *impl;
};

typedef struct mynah_slm_backend_ops {
    const char *name;
    void (*close)(void *st);

    float *(*alloc)(void *st, size_t n, char *err, size_t errsz);
    void   (*free)(void *st, float *p);
    int    (*h2d)(void *st, float *dst, const float *src, size_t n, char *err, size_t errsz);
    int    (*d2h)(void *st, float *dst, const float *src, size_t n, char *err, size_t errsz);
    int    (*sync)(void *st, char *err, size_t errsz);

    /* Fill w->data from w->host (rows/cols/type/row_bytes already set and
     * validated). Return 1 to refuse the type. Called once per distinct
     * weight; the wrapper does the dedup. */
    int  (*weight_upload)(void *st, mynah_slm_bweight *w, char *err, size_t errsz);
    void (*weight_release)(void *st, mynah_slm_bweight *w);

    int (*matvec)(void *st, const mynah_slm_bweight *w, const float *x, float *y,
                  char *err, size_t errsz);
    int (*matmat)(void *st, const mynah_slm_bweight *w, const float *x, float *y,
                  size_t tokens, char *err, size_t errsz);
    int (*embed)(void *st, const mynah_slm_bweight *w, const uint32_t *tokens,
                 size_t n, float *out, char *err, size_t errsz);

    int (*rms_norm)(void *st, float *out, const float *x, const mynah_slm_bweight *w,
                    size_t rows, uint32_t dim, float eps, char *err, size_t errsz);
    int (*rms_norm_heads)(void *st, float *x, const mynah_slm_bweight *w,
                          size_t n_heads, uint32_t head_dim, float eps,
                          char *err, size_t errsz);
    int (*swiglu)(void *st, float *gate, const float *up, size_t n,
                  char *err, size_t errsz);
    int (*add)(void *st, float *y, const float *x, size_t n, char *err, size_t errsz);
    int (*add_scaled)(void *st, float *y, const float *x, float w, size_t n,
                      char *err, size_t errsz);

    /* Fill r->impl. Return 1 to refuse (e.g. a pairing it does not implement). */
    int  (*rope_create)(void *st, mynah_slm_brope *r, char *err, size_t errsz);
    void (*rope_free)(void *st, mynah_slm_brope *r);
    int  (*rope)(void *st, const mynah_slm_brope *r, float *x, size_t n_tokens,
                 uint32_t n_heads, uint32_t pos0, char *err, size_t errsz);

    /* Fill kv->impl. Return 1 to refuse a precision or a head_dim. */
    int  (*kv_create)(void *st, mynah_slm_bkv *kv, char *err, size_t errsz);
    void (*kv_free)(void *st, mynah_slm_bkv *kv);
    int  (*kv_append)(void *st, mynah_slm_bkv *kv, uint32_t layer, uint32_t pos0,
                      size_t n, const float *k, const float *v, char *err, size_t errsz);
    int  (*attention)(void *st, const mynah_slm_bkv *kv, uint32_t layer,
                      const float *q, float *out, uint32_t pos0, size_t n_q,
                      float scale, char *err, size_t errsz);

    int (*argmax)(void *st, const float *x, size_t n, uint32_t *idx,
                  char *err, size_t errsz);
} mynah_slm_backend_ops;

/* Each backend's constructor: fill `ops` (pre-zeroed) and `*state`. */
int mynah_slm_backend_cpu_open(mynah_slm_backend_ops *ops, void **state,
                               char *err, size_t errsz);
/* Defined only in a `make cuda` build (gpu/cuda/backend_cuda.cu). */
int mynah_slm_backend_cuda_open(mynah_slm_backend_ops *ops, void **state,
                                char *err, size_t errsz);

#ifdef __cplusplus
}
#endif

#endif /* MYNAH_SLM_BACKEND_OPS_H */
