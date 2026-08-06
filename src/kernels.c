/* kernels.c — f32 reference implementations.
 *
 * Ported in shape from qwen-tts (docs/prior-art.md), stripped of the TTS half
 * and of the SIMD for now. Correctness first: these are what the parity gate
 * judges, and a vectorized kernel that is wrong costs far more to find than a
 * scalar one that is slow. The SIMD paths land behind these same signatures.
 *
 * SPDX-License-Identifier: MIT */
#include "kernels.h"

#include "threads.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

#if defined(__ARM_NEON)
#include <arm_neon.h>
#define MYNAH_SLM_NEON 1
#elif defined(__AVX2__)
#include <immintrin.h>
#define MYNAH_SLM_AVX2 1
#endif

/* dot and scaled-accumulate over head_dim, the two inner loops of attention.
 *
 * Worth vectorizing and nothing else is, which took a measurement to learn:
 * at n_kv = 32 the whole non-matvec half of a decode step is 1.6 ms against
 * ~120 ms of matvec, i.e. 1.4%, and RMSNorm/RoPE/SwiGLU are a rounding error
 * inside that. Attention is the only one whose cost grows with the context,
 * and it grows fast — per token, across 28 layers:
 *
 *   n_kv    32 ->   1.4 ms      n_kv   512 ->  20 ms
 *   n_kv   128 ->   5.1 ms      n_kv  2048 -> 106 ms
 *
 * A 37 ms decode step is 4% attention at n_kv 32 and 287% at n_kv 2048. Every
 * benchmark in docs/perf.md so far used a 19-token prompt, which is precisely
 * where this does not show. Summarizing a meeting transcript is not. */
static inline float dot_f32(const float *a, const float *b, uint32_t n) {
#if defined(MYNAH_SLM_NEON)
    float32x4_t s0 = vdupq_n_f32(0.0f), s1 = vdupq_n_f32(0.0f);
    float32x4_t s2 = vdupq_n_f32(0.0f), s3 = vdupq_n_f32(0.0f);
    uint32_t i = 0;
    for (; i + 16 <= n; i += 16) {
        s0 = vfmaq_f32(s0, vld1q_f32(a + i),      vld1q_f32(b + i));
        s1 = vfmaq_f32(s1, vld1q_f32(a + i + 4),  vld1q_f32(b + i + 4));
        s2 = vfmaq_f32(s2, vld1q_f32(a + i + 8),  vld1q_f32(b + i + 8));
        s3 = vfmaq_f32(s3, vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
    }
    float sum = vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)));
    for (; i < n; i++) sum += a[i] * b[i];
    return sum;
#elif defined(MYNAH_SLM_AVX2)
    __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
    uint32_t i = 0;
    for (; i + 16 <= n; i += 16) {
        s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0);
        s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), s1);
    }
    const __m256 t = _mm256_add_ps(s0, s1);
    __m128 v = _mm_add_ps(_mm256_castps256_ps128(t), _mm256_extractf128_ps(t, 1));
    v = _mm_hadd_ps(v, v);
    v = _mm_hadd_ps(v, v);
    float sum = _mm_cvtss_f32(v);
    for (; i < n; i++) sum += a[i] * b[i];
    return sum;
#else
    /* The scalar reference keeps a double accumulator; the vector paths above
     * use four (NEON) or two (AVX2) f32 lanes, which is pairwise summation and
     * so no worse in practice — the parity gate agrees, and it is the gate
     * that decides, not the argument. */
    double sum = 0.0;
    for (uint32_t i = 0; i < n; i++) sum += (double)a[i] * (double)b[i];
    return (float)sum;
#endif
}

/* y[i] += w * x[i] */
static inline void axpy_f32(float *y, const float *x, float w, uint32_t n) {
#if defined(MYNAH_SLM_NEON)
    const float32x4_t vw = vdupq_n_f32(w);
    uint32_t i = 0;
    for (; i + 16 <= n; i += 16) {
        vst1q_f32(y + i,      vfmaq_f32(vld1q_f32(y + i),      vld1q_f32(x + i),      vw));
        vst1q_f32(y + i + 4,  vfmaq_f32(vld1q_f32(y + i + 4),  vld1q_f32(x + i + 4),  vw));
        vst1q_f32(y + i + 8,  vfmaq_f32(vld1q_f32(y + i + 8),  vld1q_f32(x + i + 8),  vw));
        vst1q_f32(y + i + 12, vfmaq_f32(vld1q_f32(y + i + 12), vld1q_f32(x + i + 12), vw));
    }
    for (; i < n; i++) y[i] += w * x[i];
#elif defined(MYNAH_SLM_AVX2)
    const __m256 vw = _mm256_set1_ps(w);
    uint32_t i = 0;
    for (; i + 8 <= n; i += 8)
        _mm256_storeu_ps(y + i, _mm256_fmadd_ps(_mm256_loadu_ps(x + i), vw,
                                                _mm256_loadu_ps(y + i)));
    for (; i < n; i++) y[i] += w * x[i];
#else
    for (uint32_t i = 0; i < n; i++) y[i] += w * x[i];
#endif
}

/* ── allocation ───────────────────────────────────────────────────────────── */

void *mynah_slm_aligned_alloc(size_t bytes) {
    void *p = NULL;
    if (bytes == 0) bytes = 1;
    if (posix_memalign(&p, 64, bytes) != 0) return NULL;
    return p;
}

void mynah_slm_aligned_free(void *p) { free(p); }

/* ── norms ────────────────────────────────────────────────────────────────── */

/* The sum accumulates in double. The residual stream reaches absmax ~8e3 by
 * layer 13 in Qwen3-0.6B (the usual massive activations), and squaring that
 * over 1024 channels is where an f32 accumulator starts shedding bits. */
static float rms_scale(const float *x, uint32_t dim, float eps) {
    double sum = 0.0;
    for (uint32_t i = 0; i < dim; i++) sum += (double)x[i] * (double)x[i];
    return (float)(1.0 / sqrt(sum / (double)dim + (double)eps));
}

void mynah_slm_rms_norm(float *out, const float *x, const float *weight,
                        uint32_t dim, float eps) {
    const float s = rms_scale(x, dim, eps);
    for (uint32_t i = 0; i < dim; i++) out[i] = x[i] * s * weight[i];
}

void mynah_slm_rms_norm_per_head(float *x, const float *weight,
                                 uint32_t n_heads, uint32_t head_dim, float eps) {
    for (uint32_t h = 0; h < n_heads; h++) {
        float *xh = x + (size_t)h * head_dim;
        const float s = rms_scale(xh, head_dim, eps);
        for (uint32_t i = 0; i < head_dim; i++) xh[i] *= s * weight[i];
    }
}

/* ── RoPE ─────────────────────────────────────────────────────────────────── */

int mynah_slm_rope_init(mynah_slm_rope *r, uint32_t head_dim, uint32_t max_pos,
                        float theta) {
    memset(r, 0, sizeof *r);
    if (head_dim == 0 || head_dim % 2 != 0 || max_pos == 0) return -1;

    const uint32_t half = head_dim / 2;
    r->cos = mynah_slm_aligned_alloc((size_t)max_pos * half * sizeof(float));
    r->sin = mynah_slm_aligned_alloc((size_t)max_pos * half * sizeof(float));
    if (!r->cos || !r->sin) { mynah_slm_rope_free(r); return -1; }

    r->head_dim = head_dim;
    r->max_pos  = max_pos;

    for (uint32_t i = 0; i < half; i++) {
        /* exponent 2i/head_dim, matching the oracle exactly */
        const double inv = 1.0 / pow((double)theta, (2.0 * (double)i) / (double)head_dim);
        for (uint32_t p = 0; p < max_pos; p++) {
            const double a = (double)p * inv;
            r->cos[(size_t)p * half + i] = (float)cos(a);
            r->sin[(size_t)p * half + i] = (float)sin(a);
        }
    }
    return 0;
}

void mynah_slm_rope_free(mynah_slm_rope *r) {
    if (!r) return;
    mynah_slm_aligned_free(r->cos);
    mynah_slm_aligned_free(r->sin);
    memset(r, 0, sizeof *r);
}

void mynah_slm_rope_apply(const mynah_slm_rope *r, float *x,
                          uint32_t n_heads, uint32_t pos) {
    const uint32_t half = r->head_dim / 2;
    const float *cos_p = r->cos + (size_t)pos * half;
    const float *sin_p = r->sin + (size_t)pos * half;

    for (uint32_t h = 0; h < n_heads; h++) {
        float *xh = x + (size_t)h * r->head_dim;
        for (uint32_t i = 0; i < half; i++) {
            /* Split-half: i pairs with i + half. */
            const float a = xh[i], b = xh[i + half];
            const float c = cos_p[i], s = sin_p[i];
            xh[i]        = a * c - b * s;
            xh[i + half] = b * c + a * s;
        }
    }
}

/* ── activations ──────────────────────────────────────────────────────────── */

static inline float stable_sigmoid(float x) {
    if (x >= 0.0f) return 1.0f / (1.0f + expf(-x));
    const float e = expf(x);          /* x < 0, so this cannot overflow */
    return e / (1.0f + e);
}

void mynah_slm_silu(float *x, size_t n) {
    for (size_t i = 0; i < n; i++) x[i] = x[i] * stable_sigmoid(x[i]);
}

void mynah_slm_swiglu(float *gate, const float *up, size_t n) {
    for (size_t i = 0; i < n; i++) gate[i] = gate[i] * stable_sigmoid(gate[i]) * up[i];
}

void mynah_slm_softmax(float *x, size_t n) {
    if (n == 0) return;

    float m = x[0];
    for (size_t i = 1; i < n; i++) if (x[i] > m) m = x[i];

    double sum = 0.0;
    for (size_t i = 0; i < n; i++) {
        x[i] = expf(x[i] - m);        /* argument <= 0, cannot overflow */
        sum += (double)x[i];
    }
    const float inv = (float)(1.0 / sum);
    for (size_t i = 0; i < n; i++) x[i] *= inv;
}

void mynah_slm_add(float *y, const float *x, size_t n) {
    for (size_t i = 0; i < n; i++) y[i] += x[i];
}

/* ── attention ────────────────────────────────────────────────────────────── */

void mynah_slm_attention(float *out, const float *q, const float *k, const float *v,
                         uint32_t n_kv, uint32_t n_heads, uint32_t n_kv_heads,
                         uint32_t head_dim, float *scratch) {
    const uint32_t group   = n_heads / n_kv_heads;
    const uint32_t kv_dim  = n_kv_heads * head_dim;
    const float    scale   = 1.0f / sqrtf((float)head_dim);

    for (uint32_t h = 0; h < n_heads; h++) {
        const float   *qh  = q + (size_t)h * head_dim;
        const uint32_t kvh = h / group;

        for (uint32_t t = 0; t < n_kv; t++)
            scratch[t] = dot_f32(qh, k + (size_t)t * kv_dim + (size_t)kvh * head_dim,
                                 head_dim) * scale;
        mynah_slm_softmax(scratch, n_kv);

        float *oh = out + (size_t)h * head_dim;
        memset(oh, 0, head_dim * sizeof *oh);
        for (uint32_t t = 0; t < n_kv; t++)
            axpy_f32(oh, v + (size_t)t * kv_dim + (size_t)kvh * head_dim,
                     scratch[t], head_dim);
    }
}

/* One head of the loop above, factored out so it can be a task. */
static void attention_head(float *out, const float *q, const float *k, const float *v,
                           uint32_t h, uint32_t n_kv, uint32_t n_heads,
                           uint32_t n_kv_heads, uint32_t head_dim, float *scratch) {
    const uint32_t group  = n_heads / n_kv_heads;
    const uint32_t kv_dim = n_kv_heads * head_dim;
    const float    scale  = 1.0f / sqrtf((float)head_dim);

    const float   *qh  = q + (size_t)h * head_dim;
    const uint32_t kvh = h / group;

    for (uint32_t t = 0; t < n_kv; t++)
        scratch[t] = dot_f32(qh, k + (size_t)t * kv_dim + (size_t)kvh * head_dim,
                             head_dim) * scale;
    mynah_slm_softmax(scratch, n_kv);

    float *oh = out + (size_t)h * head_dim;
    memset(oh, 0, head_dim * sizeof *oh);
    for (uint32_t t = 0; t < n_kv; t++)
        axpy_f32(oh, v + (size_t)t * kv_dim + (size_t)kvh * head_dim,
                 scratch[t], head_dim);
}

typedef struct {
    float *out, *scratch;
    const float *q, *k, *v;
    uint32_t n_kv, n_heads, n_kv_heads, head_dim;
} attn_job;

static void attn_task(void *ctx, int i) {
    attn_job *j = ctx;
    attention_head(j->out, j->q, j->k, j->v, (uint32_t)i, j->n_kv,
                   j->n_heads, j->n_kv_heads, j->head_dim,
                   j->scratch + (size_t)i * j->n_kv);
}

void mynah_slm_attention_mt(float *out, const float *q, const float *k, const float *v,
                            uint32_t n_kv, uint32_t n_heads, uint32_t n_kv_heads,
                            uint32_t head_dim, float *scratch) {
    attn_job j = { out, scratch, q, k, v, n_kv, n_heads, n_kv_heads, head_dim };
    mynah_slm_parallel_for((int)n_heads, attn_task, &j);
}
