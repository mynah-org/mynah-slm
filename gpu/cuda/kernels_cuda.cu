/* kernels_cuda.cu — the Qwen3 device kernels.
 *
 * Every kernel here has a CPU twin in src/ and a parity gate against it in
 * gpu/cuda/self_test.c; the tolerances live there and in
 * .work/cuda-backend.md. Where the CPU computes a product and then a sum,
 * the device uses __fmul_rn / __fsub_rn so nvcc cannot contract the pair into
 * an FMA behind our back: that is what lets the block decode and the RoPE
 * rotation match the CPU to the bit instead of to "a few ulp".
 *
 * Correct-first, deliberately. The GEMV is a warp per row decoding one
 * element at a time — it re-reads a block's scales for every element and
 * leaves bandwidth on the table. The structure that wins (lanes per block,
 * scales hoisted, x in shared memory, several rows per warp) is the next
 * step, taken only with a GPU to measure it against this one.
 *
 * SPDX-License-Identifier: MIT */
#include "kernels_cuda.h"

#include "cuda_self_test.h"

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

extern "C" {
#include "kernels.h"
#include "kvcache.h"
}
#include "ingot/dtype.h"
#include "ingot/quant.h"

/* Helpers that run on BOTH sides. The block decode, the bf16 rounding and the
 * RoPE rotation are written once, as __host__ __device__, and the kernels call
 * them; mynah_slm_cuda_host_check() calls the very same functions on the host
 * against ingot and src/kvcache.c. That half of the evidence needs no GPU. */
#define MYNAH_HD __host__ __device__ __forceinline__

namespace mynah_cuda {

/* Round-to-nearest products and sums that nvcc may not contract into an FMA.
 * On the host the C++ compiler runs in ISO mode, where contraction is off. */
MYNAH_HD float mul_rn(float a, float b) {
#ifdef __CUDA_ARCH__
    return __fmul_rn(a, b);
#else
    return a * b;
#endif
}
MYNAH_HD float sub_rn(float a, float b) {
#ifdef __CUDA_ARCH__
    return __fsub_rn(a, b);
#else
    return a - b;
#endif
}
MYNAH_HD float add_rn(float a, float b) {
#ifdef __CUDA_ARCH__
    return __fadd_rn(a, b);
#else
    return a + b;
#endif
}
MYNAH_HD float u2f(unsigned u) {
#ifdef __CUDA_ARCH__
    return __uint_as_float(u);
#else
    float f;
    std::memcpy(&f, &u, sizeof f);
    return f;
#endif
}
MYNAH_HD unsigned f2u(float f) {
#ifdef __CUDA_ARCH__
    return __float_as_uint(f);
#else
    unsigned u;
    std::memcpy(&u, &f, sizeof u);
    return u;
#endif
}

bool type_supported(int type) {
    return type == TYPE_F32 || type == TYPE_Q8_0 || type == TYPE_Q4_K || type == TYPE_Q6_K;
}

bool head_dim_supported(uint32_t head_dim) {
    return head_dim == 64 || head_dim == 128 || head_dim == 256;
}

/* ── block decode, one element at a time ──────────────────────────────────
 * Layouts are ggml's, as ingot decodes them (third_party/ingot/src/dequant.c):
 *   Q8_0  34 B / 32:  f16 d, 32 x int8                      x = q * d
 *   Q4_K 144 B / 256: f16 d, f16 dmin, 12 B packed 6-bit scales/mins,
 *                     128 B of nibbles                      x = q * (d*sc) - dmin*m
 *   Q6_K 210 B / 256: 128 B low nibbles, 64 B high 2-bit pairs,
 *                     16 x int8 scales, f16 d               x = (q - 32) * (d*sc)
 * Bytes are read one at a time: a 34-byte block puts its f16 at offsets that
 * are only 2-aligned, and correctness comes before load width here. */

/* IEEE half -> float, exact, subnormals included. Written out rather than
 * __half2float so the host check exercises the same code. */
MYNAH_HD float f16_to_f32(unsigned h) {
    const unsigned sign = (h & 0x8000u) << 16;
    unsigned exp = (h >> 10) & 0x1fu, man = h & 0x3ffu;
    if (exp == 0) {
        if (man == 0) return u2f(sign);
        int e = -1;                                   /* normalize the subnormal */
        do { e++; man <<= 1; } while (!(man & 0x400u));
        return u2f(sign | ((unsigned)(112 - e) << 23) | ((man & 0x3ffu) << 13));
    }
    if (exp == 31) return u2f(sign | 0x7f800000u | (man << 13));
    return u2f(sign | ((exp + 112u) << 23) | (man << 13));
}

MYNAH_HD float ld_f16(const uint8_t *p) {
    return f16_to_f32((unsigned)p[0] | ((unsigned)p[1] << 8));
}

/* ggml's get_scale_min_k4: eight 6-bit scales and mins in 12 bytes. */
MYNAH_HD void k4_scale_min(const uint8_t *s, int idx,
                                             unsigned *sc, unsigned *mn) {
    if (idx < 4) {
        *sc = s[idx] & 63u;
        *mn = s[idx + 4] & 63u;
    } else {
        *sc = (s[idx + 4] & 0x0fu) | ((unsigned)(s[idx - 4] >> 6) << 4);
        *mn = (s[idx + 4] >> 4)    | ((unsigned)(s[idx] >> 6) << 4);
    }
}

template <int TYPE>
MYNAH_HD float dq(const uint8_t *row, size_t i);

template <>
MYNAH_HD float dq<TYPE_F32>(const uint8_t *row, size_t i) {
    const uint8_t *p = row + i * 4u;
    return u2f((unsigned)p[0] | ((unsigned)p[1] << 8) | ((unsigned)p[2] << 16) | ((unsigned)p[3] << 24));
}

template <>
MYNAH_HD float dq<TYPE_Q8_0>(const uint8_t *row, size_t i) {
    const uint8_t *b = row + (i >> 5) * 34u;
    const int q = (int)(int8_t)b[2 + (i & 31u)];
    return mul_rn((float)q, ld_f16(b));
}

template <>
MYNAH_HD float dq<TYPE_Q4_K>(const uint8_t *row, size_t i) {
    const uint8_t *b = row + (i >> 8) * 144u;
    const unsigned j = (unsigned)(i & 255u);
    const unsigned g = j >> 6, w = j & 63u;           /* 64-element group, offset */
    const uint8_t *qs = b + 16 + g * 32u;
    unsigned sc, mn;
    k4_scale_min(b + 4, (int)(2u * g + (w >= 32u ? 1u : 0u)), &sc, &mn);
    const unsigned q = (w < 32u) ? (qs[w] & 0x0fu) : (qs[w - 32u] >> 4);
    const float d0 = mul_rn(ld_f16(b), (float)sc);
    const float m0 = mul_rn(ld_f16(b + 2), (float)mn);
    return sub_rn(mul_rn((float)q, d0), m0);
}

template <>
MYNAH_HD float dq<TYPE_Q6_K>(const uint8_t *row, size_t i) {
    const uint8_t *b = row + (i >> 8) * 210u;
    const unsigned j = (unsigned)(i & 255u);
    const unsigned half = j >> 7, jj = j & 127u;
    const unsigned l = jj & 31u, quad = jj >> 5;      /* 4 quads of 32 per half */
    const uint8_t *ql = b + half * 64u;
    const uint8_t *qh = b + 128u + half * 32u;
    const int8_t  *sc = reinterpret_cast<const int8_t *>(b + 192u + half * 8u);
    const unsigned is = l >> 4;
    unsigned lo, hi;
    switch (quad) {
        case 0:  lo = ql[l] & 0x0fu;        hi = (qh[l] >> 0) & 3u; break;
        case 1:  lo = ql[l + 32u] & 0x0fu;  hi = (qh[l] >> 2) & 3u; break;
        case 2:  lo = ql[l] >> 4;           hi = (qh[l] >> 4) & 3u; break;
        default: lo = ql[l + 32u] >> 4;     hi = (qh[l] >> 6) & 3u; break;
    }
    const int q = (int)(lo | (hi << 4)) - 32;
    const float f = mul_rn(ld_f16(b + 208u), (float)sc[is + 2u * quad]);
    return mul_rn((float)q, f);
}

/* ── GEMV / batched GEMV: a warp per output row ───────────────────────── */

constexpr int MV_WARPS = 4;   /* rows per block */

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

template <int TYPE>
__global__ void __launch_bounds__(MV_WARPS * 32)
k_matvec(const uint8_t *w, size_t row_bytes, size_t rows, size_t cols,
         const float *x, float *y) {
    const size_t row = (size_t)blockIdx.x * MV_WARPS + (threadIdx.x >> 5);
    const unsigned lane = threadIdx.x & 31u;
    const size_t t = blockIdx.y;                      /* token */
    if (row >= rows) return;
    const uint8_t *r = w + row * row_bytes;
    const float *xt = x + t * cols;
    float acc = 0.0f;
    for (size_t c = lane; c < cols; c += 32u) acc = fmaf(dq<TYPE>(r, c), xt[c], acc);
    acc = warp_sum(acc);
    if (lane == 0) y[t * rows + row] = acc;
}

template <int TYPE>
__global__ void k_embed(const uint8_t *w, size_t row_bytes, size_t cols,
                        const uint32_t *tokens, float *out) {
    const size_t i = blockIdx.x;
    const uint8_t *r = w + (size_t)tokens[i] * row_bytes;
    for (size_t c = threadIdx.x; c < cols; c += blockDim.x) out[i * cols + c] = dq<TYPE>(r, c);
}

cudaError_t launch_matvec(int type, const void *w, size_t row_bytes, size_t rows,
                          size_t cols, const float *x, float *y, size_t tokens,
                          cudaStream_t s) {
    if (tokens == 0 || tokens > 65535u || rows == 0) return cudaErrorInvalidValue;
    const dim3 grid((unsigned)((rows + MV_WARPS - 1) / MV_WARPS), (unsigned)tokens);
    const dim3 block(MV_WARPS * 32);
    const uint8_t *wb = static_cast<const uint8_t *>(w);
    switch (type) {
        case TYPE_F32:  k_matvec<TYPE_F32><<<grid, block, 0, s>>>(wb, row_bytes, rows, cols, x, y); break;
        case TYPE_Q8_0: k_matvec<TYPE_Q8_0><<<grid, block, 0, s>>>(wb, row_bytes, rows, cols, x, y); break;
        case TYPE_Q4_K: k_matvec<TYPE_Q4_K><<<grid, block, 0, s>>>(wb, row_bytes, rows, cols, x, y); break;
        case TYPE_Q6_K: k_matvec<TYPE_Q6_K><<<grid, block, 0, s>>>(wb, row_bytes, rows, cols, x, y); break;
        default: return cudaErrorInvalidValue;
    }
    return cudaGetLastError();
}

cudaError_t launch_embed(int type, const void *w, size_t row_bytes, size_t cols,
                         const uint32_t *tokens, size_t n, float *out, cudaStream_t s) {
    if (n == 0 || n > 0x7fffffffu) return cudaErrorInvalidValue;
    const dim3 grid((unsigned)n), block(256);
    const uint8_t *wb = static_cast<const uint8_t *>(w);
    switch (type) {
        case TYPE_F32:  k_embed<TYPE_F32><<<grid, block, 0, s>>>(wb, row_bytes, cols, tokens, out); break;
        case TYPE_Q8_0: k_embed<TYPE_Q8_0><<<grid, block, 0, s>>>(wb, row_bytes, cols, tokens, out); break;
        case TYPE_Q4_K: k_embed<TYPE_Q4_K><<<grid, block, 0, s>>>(wb, row_bytes, cols, tokens, out); break;
        case TYPE_Q6_K: k_embed<TYPE_Q6_K><<<grid, block, 0, s>>>(wb, row_bytes, cols, tokens, out); break;
        default: return cudaErrorInvalidValue;
    }
    return cudaGetLastError();
}

/* ── RMSNorm ───────────────────────────────────────────────────────────────
 * f32 throughout. The CPU sums in double because the residual stream reaches
 * |x| ~ 8e3; a tree reduction in f32 loses ~log2(dim) ulp instead of the
 * ~dim ulp of a sequential f32 sum, which is what the gate measures. */

constexpr int NORM_THREADS = 256;

__global__ void __launch_bounds__(NORM_THREADS)
k_rms_norm(float *out, const float *x, const float *w, uint32_t dim, float eps) {
    __shared__ float part[NORM_THREADS / 32];
    const float *xr = x + (size_t)blockIdx.x * dim;
    float *orow = out + (size_t)blockIdx.x * dim;
    float ss = 0.0f;
    for (uint32_t i = threadIdx.x; i < dim; i += NORM_THREADS) ss = fmaf(xr[i], xr[i], ss);
    ss = warp_sum(ss);
    if ((threadIdx.x & 31u) == 0) part[threadIdx.x >> 5] = ss;
    __syncthreads();
    if (threadIdx.x < 32) {
        float v = threadIdx.x < NORM_THREADS / 32 ? part[threadIdx.x] : 0.0f;
        v = warp_sum(v);
        if (threadIdx.x == 0) part[0] = v;
    }
    __syncthreads();
    const float scale = 1.0f / sqrtf(part[0] / (float)dim + eps);
    for (uint32_t i = threadIdx.x; i < dim; i += NORM_THREADS)
        orow[i] = __fmul_rn(__fmul_rn(xr[i], scale), w[i]);
}

/* QK-norm: a warp per head, in place. */
__global__ void k_rms_norm_heads(float *x, const float *w, size_t n_heads,
                                 uint32_t head_dim, float eps) {
    const size_t h = (size_t)blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const unsigned lane = threadIdx.x & 31u;
    if (h >= n_heads) return;
    float *xh = x + h * head_dim;
    float ss = 0.0f;
    for (uint32_t i = lane; i < head_dim; i += 32u) ss = fmaf(xh[i], xh[i], ss);
    ss = warp_sum(ss);
    const float scale = 1.0f / sqrtf(ss / (float)head_dim + eps);
    /* The CPU computes xh[i] *= s * weight[i]: the gain first, then x. */
    for (uint32_t i = lane; i < head_dim; i += 32u) xh[i] = __fmul_rn(xh[i], __fmul_rn(scale, w[i]));
}

cudaError_t launch_rms_norm(float *out, const float *x, const float *w, size_t rows,
                            uint32_t dim, float eps, cudaStream_t s) {
    if (rows == 0 || rows > 0x7fffffffu) return cudaErrorInvalidValue;
    k_rms_norm<<<(unsigned)rows, NORM_THREADS, 0, s>>>(out, x, w, dim, eps);
    return cudaGetLastError();
}

cudaError_t launch_rms_norm_heads(float *x, const float *w, size_t n_heads,
                                  uint32_t head_dim, float eps, cudaStream_t s) {
    const unsigned heads_per_block = 4;
    const size_t blocks = (n_heads + heads_per_block - 1) / heads_per_block;
    if (blocks == 0 || blocks > 0x7fffffffu) return cudaErrorInvalidValue;
    k_rms_norm_heads<<<(unsigned)blocks, heads_per_block * 32, 0, s>>>(x, w, n_heads, head_dim, eps);
    return cudaGetLastError();
}

/* ── RoPE ──────────────────────────────────────────────────────────────────
 * One thread per rotated pair. NeoX split-half pairs i with i + half (Qwen3);
 * interleaved pairs 2i with 2i+1. Same table, same products as the CPU. */
MYNAH_HD void rope_pair(float *x, const float *cos_t, const float *sin_t, size_t p,
                       uint32_t n_heads, uint32_t head_dim, uint32_t pos0, int interleaved) {
    const uint32_t half = head_dim / 2;
    const uint32_t i = (uint32_t)(p % half);
    const size_t head = p / half;                      /* token * n_heads + h */
    const uint32_t pos = pos0 + (uint32_t)(head / n_heads);
    float *xh = x + head * head_dim;
    const float c = cos_t[(size_t)pos * half + i], sn = sin_t[(size_t)pos * half + i];
    const uint32_t ia = interleaved ? 2u * i : i;
    const uint32_t ib = interleaved ? 2u * i + 1u : i + half;
    const float a = xh[ia], b = xh[ib];
    xh[ia] = sub_rn(mul_rn(a, c), mul_rn(b, sn));
    xh[ib] = add_rn(mul_rn(b, c), mul_rn(a, sn));
}

__global__ void k_rope(float *x, const float *cos_t, const float *sin_t, size_t n_pairs,
                       uint32_t n_heads, uint32_t head_dim, uint32_t pos0, int interleaved) {
    const size_t p = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (p < n_pairs) rope_pair(x, cos_t, sin_t, p, n_heads, head_dim, pos0, interleaved);
}

cudaError_t launch_rope(float *x, const float *cos_t, const float *sin_t,
                        size_t n_tokens, uint32_t n_heads, uint32_t head_dim,
                        uint32_t pos0, int interleaved, cudaStream_t s) {
    const size_t n_pairs = n_tokens * n_heads * (head_dim / 2);
    const size_t blocks = (n_pairs + 255) / 256;
    if (blocks == 0 || blocks > 0x7fffffffu) return cudaErrorInvalidValue;
    k_rope<<<(unsigned)blocks, 256, 0, s>>>(x, cos_t, sin_t, n_pairs, n_heads, head_dim,
                                            pos0, interleaved);
    return cudaGetLastError();
}

/* ── elementwise ─────────────────────────────────────────────────────────── */

/* expf only ever sees a non-positive argument, as on the CPU. */
__device__ __forceinline__ float stable_sigmoid(float v) {
    if (v >= 0.0f) return 1.0f / (1.0f + expf(-v));
    const float e = expf(v);
    return e / (1.0f + e);
}

__global__ void k_swiglu(float *g, const float *u, size_t n) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) g[i] = __fmul_rn(__fmul_rn(g[i], stable_sigmoid(g[i])), u[i]);
}

__global__ void k_add(float *y, const float *x, size_t n) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __fadd_rn(y[i], x[i]);
}

__global__ void k_add_scaled(float *y, const float *x, float w, size_t n) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __fadd_rn(y[i], __fmul_rn(w, x[i]));
}

static unsigned blocks_for(size_t n) { return (unsigned)((n + 255) / 256); }

cudaError_t launch_swiglu(float *gate, const float *up, size_t n, cudaStream_t s) {
    if (n == 0 || (n + 255) / 256 > 0x7fffffffu) return cudaErrorInvalidValue;
    k_swiglu<<<blocks_for(n), 256, 0, s>>>(gate, up, n);
    return cudaGetLastError();
}

cudaError_t launch_add(float *y, const float *x, size_t n, cudaStream_t s) {
    if (n == 0 || (n + 255) / 256 > 0x7fffffffu) return cudaErrorInvalidValue;
    k_add<<<blocks_for(n), 256, 0, s>>>(y, x, n);
    return cudaGetLastError();
}

cudaError_t launch_add_scaled(float *y, const float *x, float w, size_t n, cudaStream_t s) {
    if (n == 0 || (n + 255) / 256 > 0x7fffffffu) return cudaErrorInvalidValue;
    k_add_scaled<<<blocks_for(n), 256, 0, s>>>(y, x, w, n);
    return cudaGetLastError();
}

/* ── bf16 KV ───────────────────────────────────────────────────────────────
 * The exact integer arithmetic of src/kvcache.c put_row: add 0x7fff plus the
 * lsb of the kept half, keep the top 16 bits. Same stored bits, so a device
 * cache and a CPU bf16 cache hold the same history. */
MYNAH_HD uint16_t bf16_rne(float v) {
    unsigned bits = f2u(v);
    bits += 0x7fffu + ((bits >> 16) & 1u);
    return (uint16_t)(bits >> 16);
}

MYNAH_HD float bf16_to_f32(uint16_t b) {
    return u2f((unsigned)b << 16);
}

__global__ void k_kv_append_bf16(uint16_t *kc, uint16_t *vc, const float *k,
                                 const float *v, size_t n) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    kc[i] = bf16_rne(k[i]);
    vc[i] = bf16_rne(v[i]);
}

cudaError_t launch_kv_append_bf16(uint16_t *kc, uint16_t *vc, const float *k,
                                  const float *v, size_t n, uint32_t kv_dim,
                                  cudaStream_t s) {
    const size_t total = n * kv_dim;
    if (total == 0 || (total + 255) / 256 > 0x7fffffffu) return cudaErrorInvalidValue;
    k_kv_append_bf16<<<blocks_for(total), 256, 0, s>>>(kc, vc, k, v, total);
    return cudaGetLastError();
}

/* ── GQA attention over a bf16 cache ─────────────────────────────────────
 * Block = (q head, query row). ATT_WARPS warps split the history round-robin
 * (warp w takes positions w, w + ATT_WARPS, ...), each keeping an online
 * softmax (running max m, running sum l, unnormalized accumulator). Lane L
 * owns elements L, L+32, ... of the head, so a position's K and V are read
 * as contiguous 64-byte runs. The warps then merge in a FIXED order, so the
 * result does not depend on scheduling: deterministic, run to run.
 *
 * kv head = q head / (n_heads / n_kv_heads) — Qwen3-0.6B: 16 q heads over 8. */
constexpr int ATT_WARPS = 4;

template <int HD>
__global__ void __launch_bounds__(ATT_WARPS * 32)
k_attention_bf16(float *out, const float *q, const uint16_t *kc, const uint16_t *vc,
                 uint32_t pos0, uint32_t n_heads, uint32_t n_kv_heads, float scale) {
    constexpr int PER = HD / 32;
    const uint32_t h = blockIdx.x;
    const uint32_t t = blockIdx.y;
    const unsigned warp = threadIdx.x >> 5, lane = threadIdx.x & 31u;
    const uint32_t kvh = h / (n_heads / n_kv_heads);
    const size_t kv_dim = (size_t)n_kv_heads * HD;
    const size_t q_dim = (size_t)n_heads * HD;
    const uint32_t n = pos0 + t + 1;                   /* causal: positions 0..pos0+t */

    const float *qh = q + (size_t)t * q_dim + (size_t)h * HD;
    float qr[PER], acc[PER];
#pragma unroll
    for (int j = 0; j < PER; j++) { qr[j] = qh[lane + 32 * j]; acc[j] = 0.0f; }

    float m = -INFINITY, l = 0.0f;
    for (uint32_t s = warp; s < n; s += ATT_WARPS) {
        const uint16_t *kr = kc + (size_t)s * kv_dim + (size_t)kvh * HD;
        const uint16_t *vr = vc + (size_t)s * kv_dim + (size_t)kvh * HD;
        float dot = 0.0f;
#pragma unroll
        for (int j = 0; j < PER; j++) dot = fmaf(qr[j], bf16_to_f32(kr[lane + 32 * j]), dot);
        const float score = warp_sum(dot) * scale;
        const float m_new = fmaxf(m, score);
        const float corr = expf(m - m_new);            /* m = -inf the first time: 0 */
        const float p = expf(score - m_new);
        l = fmaf(l, corr, p);
#pragma unroll
        for (int j = 0; j < PER; j++)
            acc[j] = fmaf(acc[j], corr, p * bf16_to_f32(vr[lane + 32 * j]));
        m = m_new;
    }

    __shared__ float sm[ATT_WARPS], sl[ATT_WARPS];
    __shared__ float sacc[ATT_WARPS][HD];
    if (lane == 0) { sm[warp] = m; sl[warp] = l; }
#pragma unroll
    for (int j = 0; j < PER; j++) sacc[warp][lane + 32 * j] = acc[j];
    __syncthreads();

    float mx = -INFINITY;
#pragma unroll
    for (int w = 0; w < ATT_WARPS; w++) mx = fmaxf(mx, sm[w]);
    float wsc[ATT_WARPS], L = 0.0f;
#pragma unroll
    for (int w = 0; w < ATT_WARPS; w++) {
        /* A warp that saw no position (n < ATT_WARPS) has l = 0 and adds
         * nothing; its m = -inf must not reach expf(-inf - -inf) = NaN. */
        wsc[w] = sl[w] > 0.0f ? expf(sm[w] - mx) : 0.0f;
        L = fmaf(sl[w], wsc[w], L);
    }
    const float inv = 1.0f / L;
    float *oh = out + (size_t)t * q_dim + (size_t)h * HD;
    for (uint32_t e = threadIdx.x; e < (uint32_t)HD; e += ATT_WARPS * 32) {
        float o = 0.0f;
#pragma unroll
        for (int w = 0; w < ATT_WARPS; w++) o = fmaf(sacc[w][e], wsc[w], o);
        oh[e] = o * inv;
    }
}

cudaError_t launch_attention_bf16(float *out, const float *q, const uint16_t *kc,
                                  const uint16_t *vc, uint32_t pos0, size_t n_q,
                                  uint32_t n_heads, uint32_t n_kv_heads,
                                  uint32_t head_dim, float scale, cudaStream_t s) {
    if (n_q == 0 || n_q > 65535u || n_heads == 0 || n_kv_heads == 0 ||
        n_heads % n_kv_heads != 0)
        return cudaErrorInvalidValue;
    const dim3 grid(n_heads, (unsigned)n_q), block(ATT_WARPS * 32);
    switch (head_dim) {
        case 64:  k_attention_bf16<64><<<grid, block, 0, s>>>(out, q, kc, vc, pos0, n_heads, n_kv_heads, scale); break;
        case 128: k_attention_bf16<128><<<grid, block, 0, s>>>(out, q, kc, vc, pos0, n_heads, n_kv_heads, scale); break;
        case 256: k_attention_bf16<256><<<grid, block, 0, s>>>(out, q, kc, vc, pos0, n_heads, n_kv_heads, scale); break;
        default: return cudaErrorInvalidValue;
    }
    return cudaGetLastError();
}

/* ── argmax: one block, first index of the maximum ──────────────────────── */

constexpr int ARGMAX_THREADS = 1024;

__global__ void __launch_bounds__(ARGMAX_THREADS)
k_argmax(const float *x, size_t n, uint32_t *idx) {
    __shared__ float sv[ARGMAX_THREADS];
    __shared__ uint32_t si[ARGMAX_THREADS];
    /* Each thread scans a stride; ties keep the lower index, as the CPU's
     * strict `>` over an ascending scan does. */
    float bv = -INFINITY;
    uint32_t bi = 0xffffffffu;
    for (size_t i = threadIdx.x; i < n; i += ARGMAX_THREADS) {
        const float v = x[i];
        if (bi == 0xffffffffu || v > bv) { bv = v; bi = (uint32_t)i; }
    }
    sv[threadIdx.x] = bv;
    si[threadIdx.x] = bi;
    __syncthreads();
    for (int stride = ARGMAX_THREADS / 2; stride > 0; stride >>= 1) {
        if ((int)threadIdx.x < stride) {
            const float ov = sv[threadIdx.x + stride];
            const uint32_t oi = si[threadIdx.x + stride];
            const bool take = oi != 0xffffffffu &&
                (si[threadIdx.x] == 0xffffffffu || ov > sv[threadIdx.x] ||
                 (ov == sv[threadIdx.x] && oi < si[threadIdx.x]));
            if (take) { sv[threadIdx.x] = ov; si[threadIdx.x] = oi; }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) *idx = si[0];
}

cudaError_t launch_argmax(const float *x, size_t n, uint32_t *d_idx, cudaStream_t s) {
    if (n == 0 || n > 0xfffffffeu) return cudaErrorInvalidValue;
    k_argmax<<<1, ARGMAX_THREADS, 0, s>>>(x, n, d_idx);
    return cudaGetLastError();
}

}  // namespace mynah_cuda

/* ── the host half of the evidence ───────────────────────────────────────
 * The __host__ __device__ helpers above, run on the CPU against the engine's
 * own definitions. Needs no GPU, so it runs in every `make cuda-test`: on a
 * machine without a device this is the part of the CUDA code that is
 * actually executed. What it proves: the block layouts (the indexing of every
 * nibble, high-bit pair and packed scale), the f16 decode, the bf16 rounding
 * and the RoPE pair mapping. What it cannot prove: anything warp-level —
 * reductions, the online softmax, the launch geometry. */
namespace {

uint32_t hc_rng = 0xC0FFEE11u;
float hc_rand() {
    hc_rng = hc_rng * 1664525u + 1013904223u;
    return (float)((int32_t)(hc_rng >> 8) % 20001 - 10000) / 10000.0f;
}

int hc_report(FILE *log, const char *what, bool ok, double err, double tol) {
    std::fprintf(log, "%s %-58s err %.3g  tol %.3g\n", ok ? "ok  " : "FAIL", what, err, tol);
    return ok ? 0 : 1;
}

template <int TYPE>
int hc_decode(FILE *log, const char *name) {
    const size_t rows = 4, cols = 1024, n = rows * cols;
    float *f = static_cast<float *>(std::malloc(n * sizeof(float)));
    float *ref = static_cast<float *>(std::malloc(n * sizeof(float)));
    uint64_t bytes = 0, be = 0, bb = 0;
    if (!f || !ref || ingot_type_nbytes(TYPE, n, &bytes) != 0 ||
        ingot_type_geometry(TYPE, &be, &bb) != 0) {
        std::free(f); std::free(ref);
        return hc_report(log, name, false, -1, 0);
    }
    uint8_t *q = static_cast<uint8_t *>(std::malloc((size_t)bytes));
    for (size_t i = 0; i < n; i++) f[i] = hc_rand() * 3.0f;
    int bad = 0;
    double worst = 0.0, mx = 0.0;
    if (!q) bad = 1;
    else if (TYPE == mynah_cuda::TYPE_F32) std::memcpy(q, f, n * sizeof(float));
    else if (ingot_quantize(TYPE, f, n, q) != 0) bad = 1;
    if (!bad && ingot_dequant_matrix(TYPE, q, rows, cols, ref) != 0) bad = 1;
    const size_t row_bytes = (size_t)(cols / be * bb);
    for (size_t r = 0; !bad && r < rows; r++)
        for (size_t c = 0; c < cols; c++) {
            const double got = mynah_cuda::dq<TYPE>(q + r * row_bytes, c);
            const double want = ref[r * cols + c];
            const double d = std::fabs(got - want);
            if (d > worst || d != d) worst = d;
            if (std::fabs(want) > mx) mx = std::fabs(want);
        }
    std::free(f); std::free(ref); std::free(q);
    char what[96];
    std::snprintf(what, sizeof what, "host: dq<%s> == ingot decode, 4x1024", name);
    const double tol = std::ldexp(1.0, -22) * mx;
    return hc_report(log, what, !bad && worst <= tol, worst, tol);
}

}  // namespace

extern "C" int mynah_slm_cuda_host_check(FILE *log) {
    using namespace mynah_cuda;
    if (!log) log = stdout;
    int fails = 0;

    /* Every one of the 65536 halves, NaNs compared as NaNs. */
    {
        unsigned wrong = 0;
        for (unsigned h = 0; h < 65536u; h++) {
            const float got = f16_to_f32(h), want = ingot_f16_to_f32((uint16_t)h);
            if (want != want ? got == got : f2u(got) != f2u(want)) wrong++;
        }
        fails += hc_report(log, "host: f16_to_f32, all 65536 halves, bitwise", wrong == 0, wrong, 0);
    }

    fails += hc_decode<TYPE_F32>(log, "F32");
    fails += hc_decode<TYPE_Q8_0>(log, "Q8_0");
    fails += hc_decode<TYPE_Q4_K>(log, "Q4_K");
    fails += hc_decode<TYPE_Q6_K>(log, "Q6_K");

    /* bf16 rounding vs the CPU KV cache's own definition. */
    {
        enum { N = 4096 };
        static float v[N], r[N];
        for (int i = 0; i < N; i++) v[i] = hc_rand() * std::ldexp(1.0f, (i % 40) - 20);
        v[0] = 1.00390625f;            /* exactly halfway: ties to even (down) */
        v[1] = 1.01171875f;            /* exactly halfway: ties to even (up) */
        std::memcpy(r, v, sizeof v);
        mynah_slm_kv_roundtrip(MYNAH_SLM_KV_BF16, r, N);
        unsigned wrong = 0;
        for (int i = 0; i < N; i++) if (f2u(bf16_to_f32(bf16_rne(v[i]))) != f2u(r[i])) wrong++;
        fails += hc_report(log, "host: bf16 round-to-nearest-even == kvcache.c, bitwise", wrong == 0, wrong, 0);
    }

    /* RoPE: every pair index mapped and rotated as the CPU does, both forms. */
    for (int il = 0; il <= 1; il++) {
        enum { HD = 128, NH = 16, T = 3, N = T * NH * HD };
        static float x[N], ref[N];
        for (int i = 0; i < N; i++) x[i] = hc_rand() * 2.0f;
        std::memcpy(ref, x, sizeof x);
        mynah_slm_rope tab;
        if (mynah_slm_rope_init(&tab, HD, 1100, 1e6f, il) != 0) { fails++; continue; }
        for (int t = 0; t < T; t++) mynah_slm_rope_apply(&tab, ref + t * NH * HD, NH, 1000 + t);
        for (size_t p = 0; p < (size_t)T * NH * (HD / 2); p++)
            rope_pair(x, tab.cos, tab.sin, p, NH, HD, 1000, il);
        double worst = 0.0, mx = 0.0;
        for (int i = 0; i < N; i++) {
            const double d = std::fabs((double)x[i] - (double)ref[i]);
            if (d > worst || d != d) worst = d;
            if (std::fabs((double)ref[i]) > mx) mx = std::fabs((double)ref[i]);
        }
        mynah_slm_rope_free(&tab);
        const double tol = std::ldexp(1.0, -22) * mx;
        fails += hc_report(log, il ? "host: rope_pair interleaved == rope_apply, pos 1000..1002"
                                   : "host: rope_pair NeoX == rope_apply, pos 1000..1002",
                           worst <= tol, worst, tol);
    }
    return fails ? 1 : 0;
}
