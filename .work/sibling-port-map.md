# P1 — what mynah-tts and mynah-asr have that an SLM engine should take, and why

Status: **REFERENCE** (written 2026-10-08; re-read before picking up any K, S or G item)

Item: `PLAN.md` §0 P1. Sources read at `mynah-tts` main (shallow clone, 2026-10-08)
and `mynah-asr` main plus its draft PR #3 (`refs/pull/3/head`). File:line
references below are to those trees.

Task · Question · Known facts · Unknowns · Files inspected · Evidence ·
Conclusion · Next action — in that order.

---

## Task

mynah-slm is the middle stage of ASR → SLM → TTS. The two sibling engines are
months ahead in kernels, serving and GPU. Decide what transfers to
autoregressive text decoding, in what order, and what does not.

## Question

For each mature technique in the siblings: does the reason it won there exist
in an SLM? A technique moves only if that reason moves with it.

## Known facts

- **mynah-slm today** (`cde1599`): Qwen3 dense decoder, GGUF via ingot,
  own Q4_K matvec (distribute-the-sum, xsum hoisted per matvec), optional int8
  activations (`--fast`), batched prefill through a dequantized strip + sgemm,
  bf16/fp8/q8/q4 KV. Pool: mutex + condvar, one region at a time. Server:
  **one request at a time** behind `infer_mu` (`server/main.c:3-15`). No GPU.
- **mynah-tts serving doctrine** (its `.work/serving-doctrine.md`), from the
  author of the reference implementation: (1) server design + pinned core split
  + batching, (2) dataflow, (3) kernels — in that order. A 36x kernel win left
  them at two or three real-time streams because the limit was never the kernel.
- **The general law** they extracted (`.work/serving-design.md`): changes that
  removed work from the serial critical path won; changes that added a second
  submitter, a gate or a parking policy on the same pool lost. Fourteen
  approaches are falsified there with numbers — not to be re-run here.

## Unknowns

- Decode tok/s effect of anything here on this engine: no model reachable in
  the cloud environment this was written in. Every end-to-end claim below is a
  **prediction**, labelled as such.
- Whether the siblings' per-region costs (20-30 us cold wake, Neoverse V2)
  match Apple M-series or the user's Linux hosts.

## Files inspected

mynah-tts: `src/threads.c`, `src/sgemm.c`, `src/sgemm_rt.c`, `src/qmat.c`,
`src/dispatch.c`, `src/costmap.c`, `src/inference.c`, `server/main.c`,
`server/prefork.c`, `server/stream_out.c`, `src/backend.c`,
`gpu/cuda/backend_cuda.cu`, `Makefile`, and the `.work/` notes named inline.
mynah-asr: `src/qmat.c`, `server/sched.c`, `server/slot.c`, PR #3 diff stat.
mynah-slm: all of `src/`, `server/`, `Makefile`, `docs/perf.md`.

---

## Evidence — the transfer table

Classification per `.work/engineering-method.md`: **A** common runtime, **B**
common design with backend-specific code, **C** backend/ISA-specific.

### Transfers, and why it fits an SLM

| # | Technique (source) | Why it won there | Does that reason exist here? | Class | Item |
|---|---|---|---|---|---|
| 1 | Spin-then-park pool, atomic claim, affinity width (tts `src/threads.c:145-290, 694-705, 55-90`) | 50-120 pool regions per decoder call, each needing ~200 us of work to pay a 20-30 us wake | **Yes, more so.** A Qwen3-0.6B decode step is ~225 regions at ~120 us average (36.5 tok/s on M1, 8 threads). Small models are dispatch-bound before they are bandwidth-bound | A | **K2 — landed** |
| 2 | Own f32 GEMM, `BLAS=none` (tts `src/sgemm.c`, `.work/no-blas.md`) | Ownership: OpenBLAS's second pool made every profile carry env vars; mynah-asr saw throughput collapse with concurrent inferences on one OpenBLAS pool | **Yes for ownership and dependencies, NOT yet for speed.** Our hot GEMM shape (T=256 tokens × 128-row strip) is not theirs (n=16 frame batch), so their kernel families do not transfer as-is — see K1 | B | **K1 — landed opt-in** |
| 3 | Continuous batching: scheduler thread owns the model, slots, admit at the top of each step, retire at stop (tts `src/inference.c:2021-2930`) | Each step reads the weights once for every live stream | **Yes, and it is the biggest lever.** Decode is memory-bound: B streams' projections become one [B×d]·W product that reads W once. The serialized server leaves all of that on the table | B | S1 (design) |
| 4 | Sliced, FIFO-to-completion prefill interleaved with decode (tts `prefill_slice_budget`, `prefill_fifo`) | Inline prefill stalled every live stream; FIFO beat round-robin (TTFA p95 445 → 318 ms) | **Yes.** A 2275-token transcript prefill is ~10 s here (docs/perf.md); unsliced it freezes every other stream for that long | A | S1 |
| 5 | Admission ladder with fail-fast 503, listener always polled (tts `server/prefork.c:1595-1646`) | Clients held in the kernel backlog gave an invisible 4.6 s TTFA p95 | Yes, unchanged | A | S1 |
| 6 | Weight-stationary batched quantized matvec (tts `qmat_batch_rows`, asr `QMAT_WS_X4`) | 1.2-2.6x per-row at T≥4 (asr), B4 step 29.2 → 25.0 ms (tts) | Yes, it is what makes #3 pay: B tokens against one Q4_K row without dequantizing to f32 | C | S1 follow-up |
| 7 | Dispatch report + shape census, "a benchmark is invalid until dispatch is proven" (tts `src/dispatch.c`) | Silent fallbacks (Rosetta CPUID, an unlinked kernel) were reported as measurements | Yes — already bit us twice (AGENTS.md). K1's counters are the first piece | A | later |
| 8 | Runtime ISA dispatch with verify-on-first-use (tts `qmat.c:243-311, 1197`, `sgemm_rt.c`) | A portable binary got scalar kernels | **Yes for `make dist`**: our own kernels are compile-time only (`-march=native`) | C | needs a note |
| 9 | Backend vtable, "NULL = unsupported", no CUDA type across the header (tts `src/backend.c:31-153, 731-916`) | Kept CUDA optional and the CPU path the reference | Yes, the shape transfers; the kernels do not (see below) | B | G1 (design) |
| 10 | Slot pool for per-request state; KV sized to the request, not to `max_ctx` (tts `.work/kv-window-allocation.md`, `pocket-cuda-slot-pool.md`) | Admission went from 11-30 ms to 0.05 ms (GPU); KV 196.6 → 4.1 MB per context | Yes: our KV is allocated at n_ctx per state; with slots it must be per request or lazily grown | B | S1 |

### Does NOT transfer, or not as-is

| Technique | Why not here |
|---|---|
| Prefork W workers with fd passing (tts `server/prefork.c`) | Their workers each hold a separate model graph per language; one SLM model is shared read-only via mmap anyway. A single process with one scheduler is the first step; prefork is a scaling step after S1 is measured, not before |
| Per-slot emit quantum ramp 1,2,2,4,4 (tts `slot_quantum`) | Exists because audio frames must arrive faster than playback. Text tokens are emitted one per step; there is no playback clock to stay ahead of |
| Decoder lane / core split (tts `mynah_lane_split_prepare`) | Off by default even there; the batched gang shipped instead. An SLM has one stage per step, not AR + codec |
| Their sgemm families as-is | Tuned for m=512 n=16 k=512 frame batches. Our shapes are T×128×1024 (NT) and attention with k = head_dim 128. Measured: ported naively the NT path is 0.3-0.65x OpenBLAS on our prefill shapes (K1) |
| CUDA kernels (tts `gpu/cuda/backend_cuda.cu`) | No GQA, head width hardcoded to 64 (Qwen3 is 128), interleaved RoPE (Qwen3 is NeoX), LayerNorm not RMSNorm/QK-norm, no GGUF block formats. The **structure** transfers (vtable, graphs keyed by width bucket, pinned per-row tables, one stream, slot pool, int8 KV); the kernels must be written for this model family |
| int8 activations as a default | tts made it default only where the error does not re-enter the AR loop. In an LLM every projection is inside the loop; ours stays `--fast`, gated by perplexity (docs/perf.md: +1.4% ppl) |
| AMX | Neither sibling has a kernel; nothing to port |

### What mynah-asr PR #3 adds on top

The draft PR (77 files, +81k) is mostly serving infrastructure — `server/sched.c`,
`server/slot.c`, a KV layout module with its own test (`tests/test_kv_layout.c`),
fault probes, a CUDA stream test, and a large load-generation toolkit
(`tools/bench/stream_load.py`, `streaming_metrics.py`, `v2_verdict.py`).
**For S1 the relevant parts are `slot.{c,h}` and `kvcache.{c,h}` (per-slot KV
ownership) and the WAVE-then-SOAK qualification tooling.** It is a draft and will
change; re-read it before S1 starts rather than porting from this snapshot.

---

## Conclusion

The order, from the cost model rather than from what is easiest to write:

1. **K2 — pool dispatch.** Cheapest, model-free to validate, and on a 0.6B
   model it sits on the critical path ~225 times per token. **Landed.**
2. **S1 — continuous batching with batched decode projections.** The largest
   aggregate-throughput lever and the reason the siblings serve tens of streams
   on CPU. Needs a model to validate, so it ships as a design note now.
3. **K1 — own GEMM.** Landed **opt-in** for the dependency and ownership
   reasons; it does not become the default until it measures at parity.
4. **G1 — CUDA**, after S1, because the GPU path is the same scheduler with a
   different `step_batch`. Design note only.

## Next action

Downstream validation of K2 on the Linux hosts with a real model (decode tok/s
before/after, `MYNAH_SLM_POOL_SPIN_US=0` as the in-binary control,
`bench/pool_ab/run.sh` for the synthetic half); then write the S1 note before
starting it — S1 and G1 have no note yet, by the rule that a note is written
before the work, not as a placeholder.
