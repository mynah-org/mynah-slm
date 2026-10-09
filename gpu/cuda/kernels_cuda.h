/* kernels_cuda.h — host-side launchers for the Qwen3 device kernels.
 *
 * C++ (nvcc) only, and private to gpu/cuda/: the engine never sees a CUDA
 * type, it sees src/backend.h. Every launcher enqueues on the given stream,
 * does not synchronize, and returns the launch status (cudaGetLastError), so
 * a bad configuration is reported at the call that caused it.
 *
 * Written FOR Qwen3, not ported from mynah-tts: GQA (kv head = q head /
 * group), head_dim from the config (64 / 128 / 256 instantiated, anything else
 * refused), NeoX split-half RoPE, RMSNorm and per-head QK-RMSNorm, GGUF block
 * formats read straight from the stored bytes. Correct-first: one warp per
 * GEMV row and per-element block decode. Fast comes after a measured A/B on a
 * real GPU, not before (.work/cuda-backend.md).
 *
 * SPDX-License-Identifier: MIT */
#ifndef MYNAH_SLM_KERNELS_CUDA_H
#define MYNAH_SLM_KERNELS_CUDA_H

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace mynah_cuda {

/* ggml type ids the device kernels decode. Everything else is refused. */
constexpr int TYPE_F32  = 0;
constexpr int TYPE_Q8_0 = 8;
constexpr int TYPE_Q4_K = 12;
constexpr int TYPE_Q6_K = 14;

bool type_supported(int type);
bool head_dim_supported(uint32_t head_dim);

/* y[t][rows] = W[rows][cols] . x[t][cols] for t < tokens (tokens = 1 is the
 * decode GEMV). `w` is the stored block matrix, `row_bytes` one stored row. */
cudaError_t launch_matvec(int type, const void *w, size_t row_bytes, size_t rows,
                          size_t cols, const float *x, float *y, size_t tokens,
                          cudaStream_t s);

/* out[i][cols] = decoded row tokens[i] of W. `tokens` is a DEVICE array. */
cudaError_t launch_embed(int type, const void *w, size_t row_bytes, size_t cols,
                         const uint32_t *tokens, size_t n, float *out, cudaStream_t s);

cudaError_t launch_rms_norm(float *out, const float *x, const float *w, size_t rows,
                            uint32_t dim, float eps, cudaStream_t s);
cudaError_t launch_rms_norm_heads(float *x, const float *w, size_t n_heads,
                                  uint32_t head_dim, float eps, cudaStream_t s);

/* cos/sin: the [max_pos][head_dim/2] table, built on the host by the CPU's
 * own mynah_slm_rope_init and uploaded, so both sides rotate by the same
 * floats. */
cudaError_t launch_rope(float *x, const float *cos_t, const float *sin_t,
                        size_t n_tokens, uint32_t n_heads, uint32_t head_dim,
                        uint32_t pos0, int interleaved, cudaStream_t s);

cudaError_t launch_swiglu(float *gate, const float *up, size_t n, cudaStream_t s);
cudaError_t launch_add(float *y, const float *x, size_t n, cudaStream_t s);
cudaError_t launch_add_scaled(float *y, const float *x, float w, size_t n, cudaStream_t s);

/* Store n rows of kv_dim floats as bf16 (round to nearest even, the exact bit
 * arithmetic of src/kvcache.c). `kc`/`vc` point at the first destination row. */
cudaError_t launch_kv_append_bf16(uint16_t *kc, uint16_t *vc, const float *k,
                                  const float *v, size_t n, uint32_t kv_dim,
                                  cudaStream_t s);

/* Causal GQA over a bf16 cache. `kc`/`vc` are the LAYER's base,
 * [n_ctx][n_kv_heads * head_dim]. Query row t (position pos0 + t) reads
 * positions 0 .. pos0 + t. One block per (q head, query row). */
cudaError_t launch_attention_bf16(float *out, const float *q, const uint16_t *kc,
                                  const uint16_t *vc, uint32_t pos0, size_t n_q,
                                  uint32_t n_heads, uint32_t n_kv_heads,
                                  uint32_t head_dim, float scale, cudaStream_t s);

/* *d_idx = index of the first maximum of x[n], NaN treated exactly as the
 * CPU argmax treats it (x[0] NaN gives 0; NaN elsewhere is never taken).
 * One block. */
cudaError_t launch_argmax(const float *x, size_t n, uint32_t *d_idx, cudaStream_t s);

}  // namespace mynah_cuda

#endif /* MYNAH_SLM_KERNELS_CUDA_H */
