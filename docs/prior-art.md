# Prior art — what we reuse, and from where

Survey date: 2026-08-06. Every source below is **ours and MIT**, CPU-first, and
already shipping. Nothing here is a dependency: we lift code and lessons, we do
not link against sibling projects (PLAN.md: "No dependency on mynah-asr or
mynah-tts").

Rule of thumb: **port, don't invent.** If a kernel already exists in a sibling
and is validated, porting it costs an afternoon and inherits its bug fixes. If
it's generic enough for a weight-container consumer, it belongs **upstream in
`ingot`**, not copied a fourth time.

---

## 1. `../qwen-tts` — the jackpot

`qwen-tts` runs **Qwen3-TTS 0.6B/1.7B**, whose "Talker" is a **Qwen3 decoder**.
That is literally the v0.1 target architecture, already implemented, already
numerically validated against a Python reference, already loading through
`ingot`.

### `qwen_tts_talker.c` (1462 lines) — the Qwen3 forward pass

Its own header comment is our spec:

> - GQA (Grouped Query Attention) with 2:1 ratio
> - Per-head Q/K RMSNorm
> - **NeoX split-half RoPE (NOT interleaved)**
> - SwiGLU MLP

That RoPE line is the single most expensive trap in a Qwen3 port and it is
already resolved here: `apply_rope_neox_inplace()` rotates
`x[i]` against `x[i + head_dim/2]`, with a precomputed `cos`/`sin` table over
`half_dim`. The interleaved variant in `qwen_tts_kernels.h`
(`qwen_apply_rope_interleaved`) is for a **different** part of that model — do
not grab the wrong one.

Also directly reusable:
- `kv_cache_grow()` — geometric growth of a **bf16** KV cache (halves KV memory
  vs f32 at no measurable quality cost).
- The per-layer weight loading via `ingot_st_find` with `LOAD_F32(q_norm,
  "…layers.%d.self_attn.q_norm.weight", i)`-style macros.
- RoPE table precomputation (`rope_inv_freq`, `rope_cos`, `rope_sin`).
- The `QWEN_ACT_MAP` activation-capture hook — exactly the shape of the
  per-layer dump we need for oracle parity in M2.5.

**Caveat:** it loads safetensors with model-specific tensor names
(`talker.model.layers.N…`). We need a name-mapping table per container (GGUF
uses `blk.N.attn_q.weight`). That mapping is *our* new code.

### `qwen_tts_kernels.{h,c}` (5129 lines) + NEON/AVX/generic splits

The single richest file in the whole survey. What lands in `mynah-slm` almost
verbatim:

| Kernel | Note |
|---|---|
| `qwen_rms_norm` | plain RMSNorm |
| `qwen_rms_norm_residual` | **fused** residual-add + RMSNorm, one pass over `x` |
| `qwen_rms_norm_per_head` | this *is* Qwen3's QK-norm |
| `qwen_matvec_bf16` / `_int8` / `_q4_0` / `_q2_0` | threaded, NEON + AVX2 + AVX-512-VNNI |
| `qwen_matmat_bf16` / `_int8` / `_q4_0` | **batched** (B≤64): weights read from DRAM once, reused across B columns — the prefill and speculative-verify primitive |
| `qwen_matvec_*_qkv` | fused Q/K/V in one dispatch: 1 barrier instead of 3 |
| `qwen_causal_attention` | causal GQA, f32 KV |
| `qwen_causal_attention_windowed` | **sliding window** — Gemma 4 needs exactly this |
| `qwen_causal_attention_bf16kv` | bf16 KV cache attention |
| `qwen_swiglu_inplace` | fused SwiGLU on interleaved gate/up, `vvexpf` on macOS |
| `qwen_argmax_matvec_{bf16,int8,q4_0}` | argmax **without materializing logits** — with a 262144-vocab Gemma head this is a real win for greedy decoding |
| `qwen_bf16_to_f32_vec`, `qwen_bf16_accum_f32` | vectorized converts |
| `qwen_f16_to_f32` / `qwen_f32_to_f16` | native `__fp16` on aarch64, bit-exact portable fallback (handles subnormals) |
| `q4_0_block_t` | 32 weights in 18 bytes, fp16 scale — same layout as llama.cpp q4_0 |

Plus the infrastructure that is pure profit and easy to forget to build:
- `qwen_ftz_on()` — flush-to-zero / denormals-are-zero per thread (FPCR on ARM,
  MXCSR on x86). **Every** compute thread must call it; ~1–2 cycles, and it
  removes the denormal cliff.
- `qwen_check_runtime_isa()` — abort with a clear message when an `-mavx2`
  binary lands on a CPU without AVX2, instead of SIGILL.
- `qwen_caps_report()` — prints the *actually compiled* SIMD capabilities,
  derived from the same `#ifdef`s the kernels use, so a "we thought AVX existed"
  gap cannot hide behind documentation.
- `qwen_kernel_selftest()` — dispatched matvecs vs an f32 reference on
  deterministic random data. **Cross-ISA correctness proof that does not depend
  on a full-pipeline golden**, so it survives a greedy-trajectory fork.
- `qwen_matmat_bench()` — batched-vs-looped-matvec microbench with no model loaded.
- `aligned_malloc`/`aligned_calloc` — 64 B cache-line alignment for every
  SIMD/BLAS buffer.

### Other qwen-tts modules

- `qwen_tts_tokenizer.{h,c}` (986 lines) — **GPT-2 style byte-level BPE over the
  Qwen vocab**, `vocab.json` + `merges.txt`. Encode, decode, special tokens.
  We need to add: GGUF `tokenizer.ggml.*` as an input source, and UTF-8-safe
  *incremental* detokenization for streaming.
- `qwen_tts_thread.{h,c}` (385 lines) — thread pool.
- `qwen_tts_server.c` (1265 lines) — HTTP server, no framework.
- `qwen_tts_sampling.c` (202 lines) — sampling.
- `qwen_tts_backend.c` — the `request → resolve() → note → graceful CPU
  fallback` backend pattern, later copied into `mynah-asr`.
- `Makefile` — the ingot subtree wiring and the cross-ISA compile check.

---

## 2. `../keyra` — the Gemma trap list, for free

Keyra is a diffusion engine, but its **text conditioning encoders are exactly
our architectures**: `src/core/gemma3.c` (581 lines) and `src/core/qwen3vl.c`
(669 lines), both loading through `ingot`.

`gemma3.c`'s header comment is a pre-paid debugging session for M7 — these are
the things that are *silently* wrong if you get them wrong:

> - four norms per layer (pre/post around BOTH attention and FFN), not two
> - **RMSNorm scales by `(1 + weight)`**, and normalises in f32
> - **embeddings are multiplied by `sqrt(hidden_size)`**
> - attention alternates sliding-window and global layers, **each with its own
>   RoPE base** (1e4 local / 1e6 global)
> - the MLP is **GeGLU with tanh-approximated GELU**, not SwiGLU
> - `head_dim` 256 while hidden is 3840, so the projections are not square

And in the body, the one that cost real time:

> `config.json rope_scaling {linear, factor 8}`: it applies to the **GLOBAL
> layers only** — the sliding layers keep unscaled positions on the local base.
> Missing this leaves layers 0–4 bit-exact and breaks everything from layer 5 on.

Every one of these is still live in Gemma 4 E2B (our verified `text_config`
shows the same dual-theta RoPE, `gelu_pytorch_tanh`, `head_dim` 256 vs hidden
1536, and a `rope_type: proportional` + `partial_rotary_factor: 0.25` twist on
top). The layer pattern differs — Gemma 3 is 5 sliding : 1 global, Gemma 4 E2B
is **4 sliding : 1 full** per its `layer_types` array — so read the config, do
not port the constant.

Also worth lifting from keyra:
- `src/core/linenoise.c` (1768 lines) — MIT line editing. `mynah-slm chat` gets
  history, arrow keys and Ctrl-R for free instead of `fgets`.
- `src/core/gguf_quant.c` (1940 lines) — quantization paths.
- `src/core/cpu_caps.c` — runtime CPU feature detection.
- `src/core/allocator.c` — arena allocator; useful for "zero allocation in the
  token loop".
- `src/core/qwen3vl_cache.c` — text-encoder cache; the same idea as our prompt
  cache in v0.3.

---

## 3. `../mynah-asr` — the repo skeleton and the server

Not much *math* overlap (it's an encoder, we're a decoder), but it is the
canonical shape of a Mynah repo and the closest sibling in age:

- `Makefile` — BLAS auto-detect, `make lib/shared/test/bench/debug/ubsan/asan/leaks`,
  the `third_party/ingot` subtree wiring and `make update-ingot`.
- `src/threads.{h,c}` — `parallel_for` with a documented guarantee: results are
  **bit-identical to the serial loop by construction**. Plus
  `mynah_asr_blas_set_concurrency()`, which solves a bug we *will* hit: on a
  server running several inferences at once, each call spawns `nth` BLAS threads
  and they fight over the OpenBLAS lock — measured throughput collapse from 4
  concurrent requests up. Declare the concurrency, not the thread count.
- `src/qmat.{h,c}` — INT8 per-row / INT4 per-group-of-32 quantized matrices with
  a **product dispatch policy** worth copying wholesale: f32 → `cblas_sgemm`;
  quantized + small T (decode) → direct dot kernel (bandwidth bound);
  quantized + large T (prefill) → dequant into scratch + `sgemm`. And the policy
  of only quantizing the large linears — norms, biases and embeddings stay f32.
- `src/backend.{h,c}` — CPU/Metal/CUDA dispatch behind one interface with
  graceful fallback.
- `server/` (743 + 80 lines) — OpenAI-compatible HTTP with no framework and no
  libcurl. Our SSE endpoint starts here.
- `mynah_asr_sigmoid()` — a **stable** sigmoid, with the incident attached:
  under `-ffast-math` an `inf` is UB, gcc on x86 vectorizes `expf` through
  libmvec and the inf becomes NaN. CI caught it on linux/x86 only on 2026-07-18;
  clang/ARM survived by luck. Never `expf(large positive)`.
- `.gitattributes` `linguist-vendored`, the CI workflow set, and the
  PLAN/TODO/CLAUDE-are-gitignored convention.

---

## 4. `../mynah-tts` — layout confirmation

Small and clean (`src/`: audio, backend, graph, kernels, qmat, threads,
tokenizer, weights + `cli/main.c` + `server/`). Confirms the module naming we
adopt, and shows the `graph.{c,h}` idea — an explicit compute graph — which we
deliberately do **not** take: a decoder loop is simpler written straight.

---

## 5. `adriancable/qwen3.c` — external, MIT

~1000 lines, zero dependencies, Q8_0, OpenMP, `-r 1` thinking toggle. Read as a
**shape reference** for a minimal correct Qwen3 forward pass and as a
cross-check for the numerical checklist. We do not take its weight path
(Python-export-only) — we read GGUF/safetensors directly through `ingot` — and
we do not take its single-file structure.

---

## What this changes in the plan

M2 was scoped as "write the engine". After this survey it is closer to **"port
`qwen_tts_talker.c` + `qwen_tts_kernels.c`, rename, generalize the weight
mapping to GGUF, and delete the TTS-specific half"**. The genuinely new code in
v0.1 is:

1. GGUF ⇄ safetensors tensor-name mapping and config extraction (no sibling
   reads GGUF *model metadata* for a decoder yet)
2. tokenizer from GGUF `tokenizer.ggml.*` + UTF-8-safe **incremental** decode
3. chat templates, thinking channel, tool-call parsing
4. grammar-constrained decoding
5. session/KV reuse across turns
6. the SSE streaming layer on top of the ported server

Everything else has a validated ancestor in this list.
