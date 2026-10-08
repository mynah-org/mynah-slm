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
| Dequant (embedding rows) Q8_0 / Q4_K / Q6_K / F32 | block per token | **bitwise**, tol 0 on host and device (same float ops, `__fmul_rn`/`__fsub_rn` forbid FMA contraction; host half built with `-ffp-contract=off`) |
| GEMV Q8_0, Q4_K, Q6_K, F32 | a warp per output row, f32 accumulate, shuffle reduce | `abs(gpu - cpu) <= 1e-6 * sum_i abs(w_i x_i)` per row, CPU pinned to f32 activations (a reorder bound, not a quality claim; ~18x over the ~5.5e-8 a warp-by-warp host emulation shows) |
| RMSNorm | block per row, f32 tree reduction | `<= 2e-6 * max|out| + 1e-7` |
| QK-RMSNorm | warp per head | same as RMSNorm |
| NeoX RoPE | thread per pair, table from host (double, the CPU's own table) | **bitwise**, tol 0 on host and device (same table, `__fmul_rn`) |
| SwiGLU | elementwise, stable sigmoid (`expf` of a non-positive argument only) | `<= 1e-6` relative (device `expf` vs libm) |
| Residual add | elementwise | **bitwise** |
| KV append, bf16 RNE | thread per value | stored bits identical to `src/kvcache.c:put_row` — gated **on the host only**; the device planes are not read back, so there it is gated through attention |
| GQA attention, bf16 KV | block per (q head, query row), warps split positions, online softmax, fixed merge order | `<= 1e-4 * max|out|` vs `mynah_slm_attention_kv_mt` on the same bf16 cache |
| Argmax | one block, first index of the max | **exact** index, NaN as the CPU treats it (x[0] NaN → 0, NaN never taken) |

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

## Plan — cancellation: a client that leaves must not keep the GPU busy

Requirement (user, 2026-10-08): a disconnected client's request must not run
to completion on the device. **The scheduler decides, at step boundaries**
(the serving side polls a peer-gone probe every decode step and between
prefill slices); the backend's job is to make acting on that decision cheap
and safe. API in `src/backend.h` ("per-request state: a slot pool" and
"errors that belong to one request"), implementation `src/backend_slots.c`.

1. **Per-request device state lives in a slot pool**, allocated whole at
   load (`slots_create`: KV + scratch + one fence per slot). Retiring a row —
   finished, failed or cancelled — is `slot_release`: **no `cudaFree`, no
   `cudaFreeHost`** (both synchronize the device, mynah-tts
   `pocket-cuda-slot-pool.md`), no wait. Release records a fence (a CUDA
   event) behind the work already queued and PARKS the slot; `slot_acquire`
   polls parked fences (`cudaEventQuery`, never a wait) and hands a slot out
   only once its fence has passed, else returns 1 = busy — the admission
   ladder's fail-fast, not an error. The KV is not cleared on reuse:
   attention reads only positions the new request appended.
2. **At most the in-flight step finishes for a cancelled row**, because a
   step already submitted to the stream cannot be recalled without a device
   sync, and a sync per cancellation would stall every other row. Its outputs
   are dropped: release bumps the slot's **generation**, and the scheduler
   compares the generation a result was tagged with. The next step's batch
   excludes the row; the per-row tables (slot → KV base, position, generation)
   are host-pinned and **rebuilt every step** from the live set, as in
   mynah-tts, so nothing of the row survives in them.
3. **Inside a captured graph** (integration step 4): a cancelled row becomes
   an **inert pad row** (its table entry points at a dummy slot whose output
   nobody reads), or the batch is re-selected into a **smaller width bucket**
   whose graph already exists. Never a re-capture on the hot path (mynah-tts
   F5: re-recording on every membership change). Lowest-index-first slot
   reuse keeps live rows dense, which is what lets a smaller bucket take them.
4. **A device error on one request retires that request only.**
   `mynah_slm_backend_recover()` clears a non-sticky record — allocation
   failure, invalid launch configuration — the way mynah-tts `ce()` does, and
   returns 0: release that request's slot, keep serving. A **sticky** error
   (illegal address, launch failure) survives the read and poisons the whole
   context; recover returns -1 with "reopen the backend". No API can do
   better than that on CUDA, and it is said rather than hidden.

---

## Evidence

### G1-a — the vtable and the CPU backend (2026-10-08, cloud VM, x86, no model)

Files: `src/backend.h` (public, C only, no CUDA type), `src/backend_ops.h`
(the table a backend fills; C and C++ both include it), `src/backend.c`
(open/close, the upload-once weight cache, every argument check, NULL → 1),
`src/backend_cpu.c` (the reference backend), `tests/test_backend.c`.

What ran, `make test` with 4 pool threads, OpenBLAS and `BLAS=none`:

- matvec F32 / Q8_0 / Q4_K / Q6_K, 300 x 1024 (threaded row split) ==
  one serial `mynah_slm_matvec`-or-`ingot_matvec` call: **bitwise**.
- matmat, 5 tokens, == `mynah_slm_qmatmat`: **bitwise**. embed rows ==
  `ingot_dequant_matrix` row decode: **bitwise**.
- rms_norm, QK-norm (3 tokens x 16 heads x 128), NeoX and interleaved RoPE,
  swiglu, add, add_scaled == the kernels.c calls: **bitwise**.
- attention at f32 / bf16 / q8 KV, Qwen3's 16/8 GQA at head_dim 128, layer 1
  (so a missing per-layer offset cannot pass by reading layer 0), decode and a
  causal batch of 4 == `attention_mt` / `_kv_mt` / `_batch` / `_kv_batch` on a
  reference `mynah_slm_kv` filled with the same rows: **bitwise**.
- Contract: same bytes uploaded twice → one handle; Q1_0 → 1 (unsupported);
  out-of-range token, layer, position, batch, a 16/5 head grouping, a gain of
  the wrong width → -1; a CPU-only build refuses `MYNAH_SLM_DEVICE_CUDA` with
  "build with `make cuda`".
- **Mutation check:** dropping the layer offset in `cpu_attention` makes both
  f32 attention checks fail (observed on a rebuilt object, then reverted).
- ASan+UBSan build of the test: clean. `-Wall -Wextra -Wpedantic -Werror`
  on the three new files with gcc 13 and clang: clean.

No existing code path changed: nothing outside the new files calls the
backend yet.

### G1-b — the CUDA backend behind `make cuda` (2026-10-08, cloud VM, NO GPU)

Toolchain: Ubuntu 24.04 `nvidia-cuda-toolkit` **12.0.140** (apt, ~4 GB),
host compiler gcc 12 via the packaged nvcc profile. No device, no driver.

Files, all under `gpu/cuda/`, none compiled by the default build:

- `kernels_cuda.{h,cu}` — the Qwen3 kernels listed in the Plan table: GEMV
  (warp per row, tokens on grid.y for a correct-but-slow matmat) and block
  decode for F32 / Q8_0 / Q4_K / Q6_K, RMSNorm (block per row, f32 tree),
  QK-RMSNorm (warp per head), RoPE (NeoX and interleaved, table built by the
  CPU's `mynah_slm_rope_init` and uploaded), SwiGLU (stable sigmoid), add,
  add_scaled, bf16 KV append (the integer RNE of `kvcache.c`), GQA attention
  over bf16 KV templated on head_dim 64/128/256 (block per (q head, query
  row), 4 warps split positions, online softmax, fixed-order merge), argmax
  (one block, first index). The block decode, f16 decode, bf16 rounding and
  RoPE pair rotation are `__host__ __device__` so the host can run them.
- `backend_cuda.cu` — lifecycle (device count, `MYNAH_SLM_CUDA_DEVICE`, one
  non-blocking stream, pinned 4-byte argmax slot, a 4096-id token buffer
  sized at open), `ce()` with the allocation-record clear, weight upload by
  handle (`cudaMalloc` + one synchronous copy at load), the vtable. KV is bf16
  only; f32/q8/fp8 and head_dim outside 64/128/256 return **1**.
- `self_test.c` (C, through `src/backend.h` on BOTH sides) —
  `mynah_slm_cuda_self_test()`: every op vs the CPU backend at Qwen3 shapes
  (1024 / 3072 wide Q4_K both ways, a 4096 x 1024 Q6_K LM-head slab, 16/8 x 128
  GQA over 200 positions decode + causal batch of 8, MQA 8/1 x 64, 8/2 x 256,
  argmax over 151936 with a planted tie, the three refusals). Tolerances as in
  the Plan table. Returns 77 with "no CUDA device" when there is none.
- `test_cuda.c` — `make cuda-test`: host check, then device self-test.

What ran here:

- `make cuda CUDA_ARCH=sm_80`, `sm_89`, `sm_90`, with OpenBLAS and with
  `BLAS=none`: **compile and link**. The arch stamp rebuilds the `.cu`
  objects on a switch. Both `.cu` files also compile with
  `-Werror all-warnings -Xcompiler -Wall,-Wextra,-Werror`. The only link
  output is nvlink's harmless "Skipping incompatible libpthread.a" (it scans
  `-l` libraries for device code).
- **Host check** (`mynah_slm_cuda_host_check`, executes the device helpers on
  the CPU): f16 decode of all 65536 halves == `ingot_f16_to_f32`; `dq<F32,
  Q8_0, Q4_K, Q6_K>` over 4 x 1024 == `ingot_dequant_matrix`; bf16 RNE ==
  `mynah_slm_kv_roundtrip` on 4096 values incl. two exact ties; NeoX and
  interleaved `rope_pair` over every pair == `mynah_slm_rope_apply` at
  positions 1000..1002 — **all bitwise (err 0)**.
- **Mutation check:** shifting Q6_K quad 1's high bits by 4 instead of 2, and
  reading Q4_K's packed min from the wrong nibble, fail the host check (err
  4.5 and 0.406); reverted and re-passed.
- `build/cuda/test_cuda` without a device: prints the host results, then
  "SKIP cuda self-test: no CUDA device: no CUDA-capable device is detected",
  exit 77; `make cuda-test` reports SKIP and exits 0. No crash.

CI: `.github/workflows/build.yml` job `cuda-compile`, a matrix over
sm_80 / sm_89 / sm_90 in `nvidia/cuda:12.6.2-devel-ubuntu24.04` (mynah-tts's
pattern): `make cuda BLAS=none`, then `build/cuda/test_cuda` must print no
`FAIL`, say "no CUDA device" and exit 77. The YAML parses; the job itself has
not run yet (it runs on the PR). Locally the same steps pass with CUDA 12.0.

What did NOT run: **every device kernel**. The warp reductions, the online
softmax and its merge, the launch geometry, the stream ordering and the
`ce()` paths are compiled, not executed. The device self-test is written and
linked and is the first thing to run on a GPU.

### G1-d — slot pool, non-synchronizing release, per-request recovery

- CPU (`tests/test_backend.c`, model-free): a pool of 3 at Qwen3's 16/8 x 128
  bf16 shape — three acquires give slots 0,1,2; a fourth is BUSY (1); release
  of a row with 5 positions written moves its generation; a double release is
  -1; re-acquire returns the same slot with the new generation and the SAME
  KV handle and scratch pointer (no allocation); the new request's attention
  after one append equals a fresh cache that only saw that row, **bitwise** —
  the cancelled request's history is invisible; `recover` is 0 on a
  synchronous backend; an invalid KV description refuses the pool.
- CUDA (`self_test.c` `check_slots`, compiled, **not run**): release right
  after queuing an append + attention on the slot, re-acquire without a sync
  must be 0 or 1 (never -1, never a wait), after a sync the slot returns with
  a newer generation; a 2^44-float allocation fails, `recover` returns 0,
  and another slot's next op still runs.
- `make cuda` sm_89 builds; `backend_cuda.cu` compiles under nvcc -Werror.

### Review findings and fixes (2026-10-08, cloud VM, NO GPU)

A review of G1-a/b/d with host reproducers (a warp-by-warp host emulator
that `#include`s `kernels_cuda.cu`, an argmax transcription, a contraction
probe). One commit per finding; every one passed `make clean && make all &&
make test`, `make warnings` (gcc, clang), `check_plan`, `make cuda` sm_89
(`BLAS=none`) and sm_80, `test_cuda` = 77 with no `FAIL`.

| Finding | Fix | Commit |
|---|---|---|
| **B1** weight and RoPE uploads were plain `cudaMemcpy`: legacy default stream, unordered with the non-blocking kernel stream, and from pageable memory it returns once STAGED, before the DMA lands | `upload()`: `cudaMemcpyAsync` on `s->stream` + `cudaStreamSynchronize`; every other copy audited (already on `s->stream`); `h2d` documents the pinned-source caveat | `643b0b8` |
| **R1** `cudaSetDevice` only in open; the current device is per host thread | `on_device()` at the start of every op (a `cudaGetDevice` read, a set only when it differs), `recover` and `close` included; `backend.h`: any thread may drive it, `recover()` on the thread that saw the error | `028763a` |
| **R3/R4/R5** weights released before the stream was drained; frees relied on `cudaFree`'s implicit sync; cache keyed by host pointer with no lifetime rule; `recover()` 0 overstated | `mynah_slm_backend_weights_flush()` (sync, release all, empty the cache), used by close before anything is freed; explicit drain in every CUDA free; `backend.h` states "must not outlive the model" and the late-async-fault caveat; test: flush then re-upload gives the same product bits | `399beb5` |
| **N2** the `.cu` host half was compiled with g++'s default `-ffp-contract=fast` (GCC turns it off only for ISO C), so on an FMA host the host check would compare a fused rounding | `-Xcompiler -ffp-contract=off` in the nvcc rule; comment corrected. With `-mfma`: 1 host `vfm*` before, 0 after | `965f02f` |
| **N1** a NaN as a thread's first element hid that thread's stride: device 159 vs CPU 1029 on the reviewer's case | argmax over `__host__ __device__` scan/merge/final helpers with the CPU's NaN rule (x[0] NaN → 0, NaN never taken); host check runs the kernel's exact order on 6 cases vs the CPU backend; 2 NaN cases on the device; mutations fail (159, 318, 777); reviewer emulator re-run on the new file: 0 failures | `aa65407` |
| **R2/N3/N4** no attention case with n < 4 warps, no ragged GEMV block, `MYNAH_SLM_INT8` could change the CPU reference, "bitwise" gated at 2^-22, GEMV gate 1e-4 | attention at pos0 0/1/2 and a batch from 0; 1026-row products for all 4 types; CPU pinned to f32 activations; decode and RoPE tol 0 on device and host (bf16 storage: bitwise on the host only, said so); GEMV gate 1e-6 · Σ\|wx\| (emulator worst ~5.5e-8 · Σ\|wx\|, ~18x headroom) | `d10fee8` |
| **N5** the slot-reuse check attended at pos 0 after one row, so it could not see old history; the pool-refusal comment described a refusal but tested an invalid description | old request 8 rows, new request 3 then 4, attended at pos 2 and 3 vs a fresh cache; a control reads old row 4 bit for bit; mutation (2 rows) fails; refusal tested on the device (f32 KV pool → 1), CPU tests the invalid description (-1) | `118a7c7` |
| **N7** CUDA CI job: apt could prompt; root over a runner-owned checkout breaks `git describe` | `DEBIAN_FRONTEND=noninteractive`; `safe.directory "$GITHUB_WORKSPACE"` | `c5f9a66` |

Still unvalidated, all of it for want of a GPU: every device kernel, the
stream ordering, the per-thread device selection, the drains. The tol-0
device gates assume the CPU side is not FMA-contracted (gcc `-std=c11` is
not; clang's default `-ffp-contract=on` with `-march=native` may be — then
those checks fail, which is the honest outcome).

## Conclusion

G1-a: **KEEP** — the boundary exists, the CPU side of it is the engine's own
arithmetic to the bit, and nothing routes through it yet.
G1-b: **INCONCLUSIVE** until `make cuda-test` runs on a GPU. Compile/link on
three architectures and the bitwise host half are real evidence for the
layouts; none of it is evidence that a kernel launches or that a warp
reduction is right.

## Next action

1. On an L4 or L40S: `make cuda-test CUDA_ARCH=sm_89` (expect every line `ok`);
   then `compute-sanitizer --tool memcheck build/cuda/test_cuda` and
   `--tool racecheck` (the attention merge and the norm reduction use shared
   memory).
2. Only then the integration in the Plan above, behind `--device cuda`, with
   `tests/test_parity` stage dumps as its gate.
3. Speed work (GEMV with lanes per block and hoisted scales; a dequant-strip +
   cuBLASLt prefill product; graphs per width bucket) each as its own A/B
   against this correct-first version, with nsys proving the kernel ran.
