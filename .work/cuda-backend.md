# G1 — CUDA backend foundation for Qwen3: a backend vtable, then Qwen3 kernels behind `make cuda`

Status: **IN PROGRESS** (opened 2026-10-08)

Item: `PLAN.md` §0 G1. Written before the code, extended per step (G1-a, G1-b).

Task · Question · Known facts · Unknowns · Files inspected · Evidence ·
Conclusion · Next action — in that order.

---

## Task

Port the **structure** of mynah-tts's CUDA backend — not its kernels — so that a
GPU can later run a Qwen3 decode step while the CPU path stays the reference
and the default build never needs `nvcc`. Three steps, one commit each:

- **G1-a** `src/backend.{c,h}`: a C vtable for the ops one Qwen3 decode step
  needs, `NULL` = unsupported, no CUDA type in the header, weights referenced by
  opaque handles uploaded once. A CPU implementation that only calls the
  existing `src/kernels.c` / `src/qmat.c` / `src/kvcache.c` / ingot functions —
  no new numerics. `tests/test_backend.c`, model-free, bitwise where it is the
  same code. Compiled always; changes no existing code path.
- **G1-b** `gpu/cuda/`: the CUDA implementation behind `make cuda` only, with
  kernels written for Qwen3, each paired with a self-test entry against the CPU
  backend (`mynah_slm_cuda_self_test()`, binary built by `make cuda-test`).
- **G1-c** this note.

Out of scope here (other agents own them right now): `src/arch_qwen3.c`,
`src/qmat.c`, `src/threads.c`, `src/sgemm.c`, `server/`. The forward pass is
therefore **not** wired to the backend in G1; the wiring is written below as a
plan.

## Question

1. Can a backend boundary be added that lets a GPU run the whole Qwen3 decode
   step device-resident, without changing a single bit of the CPU path?
2. Which Qwen3 kernels does that need, and what does each one's parity gate
   against the CPU oracle look like (tolerance stated, not "close")?
3. Which of mynah-tts's hard-won CUDA lessons transfer to a text decoder, and
   which do not?

## Known facts

- **The model** (`docs/qwen3-arch.md`, `src/model.h`): Qwen3-0.6B, 28 layers,
  `d_model` 1024, 16 query heads, 8 KV heads (GQA group 2), `head_dim` **128**
  (its own GGUF key; `n_heads * head_dim = 2048 != d_model`), `d_ff` 3072, vocab
  151936, RMSNorm eps 1e-6, per-head QK-RMSNorm (weights `head_dim` long),
  **NeoX split-half RoPE** with theta from the config, SwiGLU, tied
  embeddings: `token_embd` is the input lookup **and** the LM head.
- **The file**: the Q4_K_M build is `Q4_K` + `Q6_K` + `F32` inside; the tied
  151936 x 1024 LM head is stored **Q6_K** and is the hottest GEMV of the
  step. Norm weights are F32.
- **The CPU decode step today** (`src/arch_qwen3.c:mynah_slm_forward`), per
  layer: `rms_norm` → `project` q/k/v (threaded row split, qmat or ingot) →
  `rms_norm_per_head` q and k → `rope_apply` q and k → `kv_put_k/v` →
  `attention_mt` (f32 KV) or `attention_kv_mt` (packed KV) → `project` o →
  `add` → `rms_norm` → `project` gate, up → `swiglu` → `project` down → `add`.
  Then final `rms_norm`, LM-head `project`, logit scale. One `embed_row` at the
  start (`ingot_dequant_matrix` on one row).
- **mynah-tts backend structure** (`src/backend.c:31-153`): one struct of
  function pointers plus an opaque `void *state`; `mynah_backend_open`
  (`:731`) fills the CPU entries, then overwrites them with the CUDA ones only
  `#if defined(MYNAH_ENABLE_CUDA)`, else returns "CUDA backend is not compiled;
  use make cuda". Wrappers (`:1833` onward) return -1 when an entry is NULL —
  **NULL means unsupported**, never a silent CPU detour for a device op.
- **mynah-tts CUDA host side** (`gpu/cuda/backend_cuda.cu`): weights cached by
  host pointer (`cached_weight`, `:1795`) and uploaded once; **one stream**
  (`:3813`); every runtime call through `ce()` (`:1619-1644`), which clears a
  pending `cudaErrorMemoryAllocation` after reporting it — that record is
  per-thread and not sticky, and left unread it fails the next unrelated launch
  — while leaving genuinely sticky errors (illegal address, launch failure)
  for the next check; graphs captured per (key, identity) and replayed
  (`graph_begin/end/launch`, `:12170`).
- **mynah-tts build boundary** (`Makefile:836-922`): `nvcc` compiles only the
  `.cu`; the C sources are rebuilt with `-DMYNAH_ENABLE_CUDA` into a separate
  `build/cuda/` tree; `CUDA_ARCH` and `ROW_CAP` are stamp files so changing
  them rebuilds; nvcc links (so no `-lcuda` driver library, only the runtime);
  CI compiles in `nvidia/cuda:*-devel` containers with no GPU
  (`.github/workflows/build.yml:386`) and found a link error on its first run.

### What transfers from mynah-tts, and what does not

| Lesson (source) | Transfers? | Why |
|---|---|---|
| Vtable + NULL = unsupported + no CUDA type in the header (`backend.c`) | **Yes** | Keeps CUDA optional and the CPU path the reference. Taken as-is in G1-a |
| Weights uploaded once, keyed by host pointer (`cached_weight`) | **Yes** | The GGUF is mmap'd and its tensor pointers are stable for the model's life. Here the key also dedupes the tied embedding: input lookup and LM head get ONE device copy |
| One stream (`:3813`) | **Yes** | One submitter. Their L15/L16 (second stream, two row groups) were dropped after MPS showed no concurrency headroom (`pocket-l40s-plateau.md` L14) |
| `ce()` clears only the allocation error record | **Yes**, verbatim in spirit | An OOM handled by a fallback must not fail the next launch; a sticky error must not be hidden |
| Host-side overhead dominates once kernels are resident (`pocket-l40s-plateau.md` F1, F4-F6: GPU idle ~40% on L40S, host 55% of the loop, ~32 ms/iteration) | **Yes, and worse for us** | A Qwen3-0.6B decode step is ~2 ms of GPU work at batch 1 on an L4-class part; a single extra sync or a 0.2 ms host stall is a 10% loss. Design for **one sync per token** from the start |
| `cudaFree` / `cudaFreeHost` synchronise the device (`pocket-cuda-slot-pool.md`) | **Yes** | Never free on the request path; pool per-request device state (slot pool) |
| Slot pool: admission 11-30 ms → 0.05 ms (`pocket-cuda-slot-pool.md`) | **Yes** (with S1) | Per-request KV and scratch parked and re-taken, not allocated |
| Graphs keyed by width bucket, inert pad rows (`pocket-l40s-plateau.md` L25) | **Yes** | Exact-width keys re-capture on every membership change (their F5) |
| int8 KV records, per head and position (`pocket-cuda-kv-int8.md`) | **Later** | Halved KV bytes for them; here keys are the sensitive side (`docs/perf.md`: 4-bit keys break the model). bf16 first, int8 only behind a perplexity gate |
| cuBLASLt heuristics / BF16 tensor-core GEMMs (`pocket-tts-cuda-streaming-parity.md` CUDA-06) | **Prefill only** | Decode at batch 1 is GEMV, memory-bound, and our weights are GGUF blocks, not f32. Lt matters when S1 batches decode rows or for prefill on dequantized strips |
| Their kernels (`k_self_attention_bf16_batch_split` etc.) | **No** | Head width hardcoded 64, no GQA, interleaved RoPE, LayerNorm, no GGUF blocks. The **layout idea** (flash-decoding split, online softmax, fixed merge order = deterministic) transfers; the code does not |
| `k_q8_quantize_rows` int8 activations (`:435`) | **No, not by default** | In an LLM every projection is inside the AR loop; int8 activations stay opt-in and ppl-gated like `--fast` (`sibling-port-map.md`) |
| Prefork / lane split / emit quantum | **No** | See `sibling-port-map.md` "does NOT transfer" |

## Unknowns

- Whether any of this runs: **there is no GPU in the environment this was
  written in.** Compile-only at best (see Evidence for whether `nvcc` was
  obtainable).
- Real decode tok/s on L4 / L40S; the share of host overhead per token for a
  0.6B model.
- Whether f32 tree-reduced RMSNorm on the GPU stays inside the parity gate at
  the residual magnitudes Qwen3 reaches (absmax ~8e3 by layer 13; the CPU sums
  in double for that reason, `src/kernels.c:rms_scale`).
- Whether bf16 KV on the device matches the CPU bf16 KV path's perplexity
  (it should: same rounding, same stored bits — to be measured).

## Files inspected

mynah-slm: `AGENTS.md`, `.work/README.md`, `.work/engineering-method.md`,
`.work/sibling-port-map.md` (row 9, "does NOT transfer" CUDA row),
`src/arch_qwen3.{c,h}`, `src/kernels.{c,h}`, `src/kvcache.{c,h}`, `src/qmat.h`,
`src/model.h`, `src/threads.h`, `Makefile`, `third_party/ingot/include/ingot/quant.h`,
`third_party/ingot/src/dequant.c` (Q8_0, Q4_K, Q6_K formulas).
mynah-tts (`66bd533`): `src/backend.h`, `src/backend.c` (`:31-153`, `:731`,
`:799-916`, `:1833`), `gpu/cuda/backend_cuda.cu` (`:435`, `:1619-1644`,
`:1795`, `:3813`, `:10799`, `:12170`), `Makefile:836-922`,
`.github/workflows/build.yml:370-430`, `.work/pocket-tts-cuda-streaming-parity.md`,
`pocket-cuda-slot-pool.md`, `pocket-cuda-kv-int8.md`, `pocket-l40s-plateau.md`,
`pocket-cuda-runtime-gaps.md`.

---

## Plan — the vtable (G1-a)

`src/backend.h`, C only. Opaque types: `mynah_slm_backend`, `mynah_slm_bweight`
(an uploaded weight), `mynah_slm_brope` (a RoPE table), `mynah_slm_bkv` (a KV
cache). Activations are **backend buffers**: `float *` that only the backend's
own ops may dereference (host memory for CPU, device memory for CUDA), moved
with explicit `h2d` / `d2h`. Every op returns **0** done, **1** unsupported
(the entry is NULL, or the backend refuses this type/shape — nothing was
touched), **-1** failed with the error text filled in.

| Op | CPU implementation (existing code, no new numerics) |
|---|---|
| `weight` (upload once, by type + rows + cols) | zero-copy: keeps the mmap pointer |
| `matvec` one token | the row split of `arch_qwen3.c:mynah_slm_project`, calling `mynah_slm_matvec` (ours) or `ingot_matvec` |
| `matmat` T tokens | `mynah_slm_qmatmat` with a strip sized at upload |
| `rms_norm` rows x dim | `mynah_slm_rms_norm` |
| `rms_norm_heads` (QK-norm) | `mynah_slm_rms_norm_per_head` |
| `rope_create` / `rope` | `mynah_slm_rope_init` / `mynah_slm_rope_apply` |
| `swiglu`, `add`, `add_scaled` | the kernels of the same name |
| `embed` (token ids → rows) | `ingot_dequant_matrix` on one row, as `embed_row` |
| `kv_create` / `kv_append` / `attention` | `mynah_slm_kv_*`, `mynah_slm_attention_mt` / `_kv_mt` (one query) and `_batch` / `_kv_batch` (a prefill batch) |
| `argmax` | first index of the maximum, host result |

## Plan — the CUDA kernels (G1-b) and their parity gates

Every kernel is compared against the **CPU backend** on random data by
`mynah_slm_cuda_self_test()`; tolerances below are the gates.

| Kernel | Shape / layout | Gate vs CPU backend |
|---|---|---|
| Dequant (embedding rows) Q8_0 / Q4_K / Q6_K / F32 | block per token | **bitwise** (same float ops, `__fmul_rn`/`__fsub_rn` forbid FMA contraction) |
| GEMV Q8_0, Q4_K, Q6_K, F32 | a warp per output row, f32 accumulate, shuffle reduce | `abs(gpu - cpu) <= 1e-4 * sum_i abs(w_i x_i) + 1e-6` per row (a reorder bound, not a quality claim) |
| RMSNorm | block per row, f32 tree reduction | `<= 2e-6 * max|out| + 1e-7` |
| QK-RMSNorm | warp per head | same as RMSNorm |
| NeoX RoPE | thread per pair, table from host (double, the CPU's own table) | **bitwise** (same table, `__fmul_rn`) |
| SwiGLU | elementwise, stable sigmoid (`expf` of a non-positive argument only) | `<= 1e-6` relative (device `expf` vs libm) |
| Residual add | elementwise | **bitwise** |
| KV append, bf16 RNE | thread per value | stored bits identical to `src/kvcache.c:put_row` |
| GQA attention, bf16 KV | block per (q head, query row), warps split positions, online softmax, fixed merge order | `<= 1e-4 * max|out|` vs `mynah_slm_attention_kv_mt` on the same bf16 cache |
| Argmax | one block, first index of the max | **exact** index |

`head_dim` comes from the config: the attention kernel is templated on 64 / 128
/ 256 and **refuses** (returns 1) anything else rather than guessing.

## Plan — integration into the forward pass (later, not in G1)

Written as the plan, because `arch_qwen3.c` is owned by another agent right now.

1. **Model open** (`--device cuda`): `mynah_slm_backend_open(CUDA)`; upload
   every layer tensor once with `mynah_slm_backend_weight` (keyed by the mmap
   pointer, so the tied embedding is one copy used twice); build the RoPE table
   once at `n_ctx`. A type the backend refuses (returns 1) makes the whole
   device path refuse at open, loudly — a per-tensor silent CPU fallback in a
   device-resident step would be a sync and two copies per layer, i.e. the
   L40S plateau by construction.
2. **State**: device buffers for x, h, q, k, v, attn, proj, gate, up, logits;
   one `mynah_slm_bkv` per sequence (bf16).
3. **Decode step, device-resident**: token id → `embed`, then per layer exactly
   the CPU op sequence above on device buffers, final norm, LM head GEMV,
   `argmax` on device for greedy, **one** `d2h` (4 bytes greedy, or the logits
   row when sampling on the host) and **one sync per token**. No per-layer
   sync, no per-op error check that reads back state.
4. **CUDA graphs**: capture the whole step once per **batch-width bucket**
   (1, 2, 4, 8, ... up to the slot cap), with the position and the KV length
   read from a device scalar rather than baked into the launch, so one graph
   replays at every position. Pad rows are inert (mynah-tts L25). Key by
   bucket, never by exact membership (their F5: re-capture on every change).
5. **Slot pool** (with S1): per-request KV and scratch parked on retire and
   re-taken at admission; no `cudaFree` on the request path.
6. **KV growth**: allocate per request in chunks (e.g. 256 positions) up to
   `n_ctx`, not `n_ctx` up front (mynah-tts `kv-window-allocation.md`: 196.6 →
   4.1 MB per context); growth is a device-to-device copy outside the graph,
   rare, and forces a re-capture of that bucket only.
7. **Prefill**: `matmat` on the device = dequantize a strip to bf16/f32 then a
   cuBLAS(Lt) GEMM — the `qmat.h` policy, on the GPU. Not in G1.
8. **Gate before any speed claim**: `tests/test_parity` stage dumps with
   `--device cuda` against the oracle at the existing per-stage tolerances,
   then `mynah-slm ppl` at bf16 KV against the CPU at bf16 KV, then decode
   tok/s with the dispatch proven (`nsys` shows our kernels, no host fallback).

---

## Evidence

(Filled per step below.)

## Conclusion

(Open.)

## Next action

G1-a: write `src/backend.{c,h}` and `tests/test_backend.c`.
