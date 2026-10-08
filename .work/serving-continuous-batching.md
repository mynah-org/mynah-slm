# S1 — continuous batching: one scheduler, per-request state, batched decode

Status: **IN PROGRESS** (opened 2026-10-08; built step by step on the cloud
branch, no checkpoint reachable — every end-to-end number is UNMEASURED)

Item: `PLAN.md` §0 S1. Map: [`sibling-port-map.md`](sibling-port-map.md) rows 3,
4, 5, 6 and 10. Sources the design is ported FROM (read-only):
mynah-tts `src/inference.c` (`serve` :2021, `admit_pass` :1526,
`slots_prefill_slice` :472, `prefill_slice_budget` :268, `prefill_fifo` :455,
`step_live` :1221, `step_isolate` :1199, `retire_pass` :1748),
`server/main.c` (`scheduler_main` :880, `sink_next_job` :654,
`sink_cancelled` :844, `job_enqueue` :491), `server/stream_out.c`, and its
`.work/serving-design.md`, `serving-doctrine.md`, `streaming-cadence.md`,
`kv-window-allocation.md`; mynah-asr draft PR #3 `server/slot.{c,h}`,
`server/sched.{c,h}`, `src/kvcache.{c,h}`, `tests/test_kv_layout.c`.

Task · Question · Known facts · Unknowns · Files inspected · Evidence ·
Conclusion · Next action — in that order. One section per step (S1-a … S1-e)
under Evidence, each written before its code.

---

## Task

Turn the serialized server (`server/main.c`: one request at a time behind
`infer_mu`) into the serving foundation the siblings proved: one scheduler
thread that owns the model, slots, admission at the top of every step, sliced
FIFO prefill interleaved with a batched decode step, retirement at
EOS / max_tokens / client gone, and step isolation. Behind a flag: the default
stays the serialized path until a real checkpoint has measured the new one.

## Question

Does a decode step that advances B independent sequences with ONE [B x d]
product per weight give each sequence the logits it would have had alone (to
the reorder tolerance the batched prefill is already held to), and can the
serving machinery around it be proven model-free so that only the throughput
question is left for a host with weights?

## Known facts

- **Decode is memory-bound and reads every weight once per token.** A Qwen3-0.6B
  Q4_K_M step walks ~400 MB of quantized weights for ONE token; B streams
  stepped separately walk it B times. One [B x d] product reads it once for all
  B — that, and only that, is why continuous batching pays on an LLM. The
  sibling cost law (`serving-design.md` §4) has the same shape:
  `T_step(B) ≈ a + b·B`, the constant is the weight stream.
- **Batched prefill already exists and is held to a reorder**
  (`docs/perf.md`: `rel=1.47e-06`, same argmax, 12 identical greedy tokens).
  `mynah_slm_qmatmat` is the product it uses: dequantize a 128-row strip, then
  sgemm against all T rows. Its activations stay f32 (ingot's batched Q4_K
  quantizes them to int8, 2.4e-3, and fails that gate).
- **The server already sizes KV per request** (`server/main.c`: prompt +
  max_tokens + 8, bf16) but builds and frees a whole `mynah_slm_state` per
  request — scratch, RoPE table and all — under the serialization lock.
- The sibling falsified list (`serving-design.md` §9) applies here unchanged:
  no low-priority prefill helper thread, no global cross-worker batching, no
  second submitter on the engine pool, no utilization-aware admission, no
  sub-quantum emission games. Text tokens are emitted one per step; there is no
  playback clock, so tts's emit-quantum ramp (`streaming-cadence.md` §3) has no
  reason to exist here.

## Unknowns

1. Decode tok/s per stream and aggregate at C = 1, 2, 4, 8 on a real checkpoint
   (weights LOCAL), and whether `qmatmat`'s dequantize-a-strip form beats B
   separate fused matvecs at decode widths (B = 2..8) — see S1-c's cost model.
2. Whether batch-vs-solo determinism holds at the TOKEN level on a real model
   (logits agree to ~1e-6 by construction; a near-tie can still flip a greedy
   argmax — see S1-c).
3. TTFT p95 under load with the prefill slice and step budget at their
   defaults, and the right defaults for this engine (tts's 32 tokens / 40 ms
   were measured for a different per-token cost).

## Files inspected

mynah-slm: `src/arch_qwen3.{c,h}`, `src/generate.{c,h}`, `src/kvcache.{c,h}`,
`src/sampler.*`, `src/tokenizer.h`, `src/qmat.{c,h}`, `src/model.*`,
`src/kernels.{c,h}`, `src/timing.*`, `server/main.c`, `server/http.*`,
`tests/test_batch.c`, `tests/test_server.sh`, `tests/test_inspect.c`,
`include/mynah_slm.h`, `Makefile`, `docs/perf.md`. Siblings: as listed above.

---

## Evidence

### S1-a — a synthetic checkpoint, so the forward pass runs in CI

**Why first.** Every later step has a numeric gate (batched decode == solo
decode) and a server test, and none of them can run here: there is no
checkpoint in this environment and HF is blocked. `test_batch`, `test_parity`,
`test_think` and `test_server.sh` all SKIP. A gate that cannot run is not a
gate.

**What it is.** `tests/fixture_model.{c,h}`: writes a Qwen3-architecture GGUF
with deterministic pseudo-random weights through ingot's own writer (the same
writer `test_inspect` uses), plus a byte-level BPE vocabulary — the 256 byte
symbols, a handful of merges and the Qwen3 control tokens — so the real
tokenizer, chat template and detokenizer load from it unchanged. Two shapes:

| spec | dims | tensors |
|---|---|---|
| `tiny`, quant | d_model 256, 2 layers, 4 q heads / 2 KV heads of 32, d_ff 512, vocab = tokenizer (~300) | Q4_K where `cols % 256 == 0` (q, k, v, gate, up), Q8_0 for `attn_output` (cols 128), Q6_K for `ffn_down` and the tied embedding, F32 norms — the Q4_K_M recipe's type mix |
| `tiny`, f32 | same | every matrix F32 |

d_model is 256 and not 64 on purpose: Q4_K blocks are 256 wide, and a fixture
that cannot hold a Q4_K row cannot exercise our own Q4_K matvec — the kernel
the whole decode path runs on. The spec struct also takes the 0.6B geometry
(used by S1-c's bench, never by `make test`: it is ~400 MB).

**What it does NOT cover**: real weights (no claim about text quality, argmax
margins or perplexity can come from it); the hybrid short-conv path (no
`shortconv` tensors are written); `Q5_K`/IQ types; the 151936-entry vocabulary.

**Gate**: `tests/test_synth` — `forward_batch` at widths {all, 7, 13} vs N x
`forward` on both fixtures, f32 KV, `rel < 1e-4` (the gate `test_batch` holds a
real checkpoint to), same argmax, same `n_past`, and 12 identical greedy tokens
after a batched prefill; plus the fixture tokenizer round-trips a mixed-script
string.

**Result (2026-10-08, cloud x86 AVX-512, gcc 13; qemu-aarch64 `BLAS=none`)**

| fixture | widths 139 / 7 / 13: rel | argmax, n_past, 12 greedy tokens |
|---|---|---|
| F32, x86 OpenBLAS | 4.4e-07 / 3.3e-07 / 1.1e-07 | equal |
| Q4_K_M mix, x86 OpenBLAS | 1.1e-07 x3 | equal |
| F32, qemu NEON, own sgemm | 6.9e-08 x3 | equal |
| Q4_K_M mix, qemu NEON, own sgemm | 1.1e-07 x3 | equal |

Two things the fixture found on its first run, both kept as evidence:

1. **ingot cannot dequantize a Q8_0 row that is not whole 256-blocks.**
   `ingot_q8_0_dequant` and its matmat run through the k-quant super-block
   core (8 Q8_0 blocks per call), while Q8_0's geometry says 32. So
   `ingot_dequant_matrix(Q8_0, cols=128)` returns -1 where `ingot_matvec`
   succeeds — decode works and the batched prefill (`qmatmat` dequantizes a
   strip) fails on the same tensor. The first fixture (head_dim 32, a 128-wide
   `attn_output` in Q8_0) failed all three widths. **Latent for Qwen3** (every
   width is a multiple of 256); real for any family with a Q8_0 tensor of
   another width. A reader bug, so the fix belongs upstream in ingot (rule 4),
   not here; the fixture now keeps every row whole 256-blocks and falls back
   to F32 otherwise, with the reason in `pick_type`.
2. **The batched-prefill gate could pass with K written to the wrong slot.**
   Mutation: in `forward_batch`, write row 3's K one position late. It
   PASSED — `state_reset` only rewinds `n_past`, and the reference pass had
   already written the identical K at the skipped position, so the stale
   bytes were the right ones. `test_synth` and `test_batch` now poison the
   cache (0xFF = NaN in f32 and bf16) on every reset; the same mutation then
   fails all four checks. This blind spot was in `test_batch` too, which is
   the gate `docs/perf.md` quotes for the batched prefill.

### S1-b — per-request state, split from what the model shares

**Problem.** `mynah_slm_state` is one sequence's KV cache *and* every scratch
buffer *and* the RoPE table, built per request. N sequences on one model would
mean N copies of ~19 MB of batch scratch (`docs/perf.md`, width 256) and N RoPE
tables for nothing, and no way for one step to see two sequences. And
`mynah_slm_generate` is a closed loop — prefill, then decode to the end — so a
scheduler cannot take one step of it.

**Split** (ownership, not copying — the sibling rule is "copy the ownership"):

| | lives in | why |
|---|---|---|
| KV cache, `n_past`, `n_ctx`, short-conv history | `mynah_slm_seq` — per request | the only state that is a function of the sequence |
| sampler (penalty history, RNG), the three detokenizers, the channel (answer / think / tool), the prompt cursor, the step count, timings | `mynah_slm_gen` — per request | ditto, at the token level |
| scratch, batch scratch, the dequantization strip, the RoPE table | `mynah_slm_state` — per scheduler | a pure function of (model, capacity); one step runs at a time, so one copy serves every sequence. tts measured the same thing for its RoPE table (`kv-window-allocation.md`: 6.1 MB per context, now shared) |

`mynah_slm_state` keeps a `mynah_slm_seq own`, and the single-sequence API
(`state_init[_kv]`, `forward`, `forward_batch`, `state_reset`, `generate`) is
unchanged in signature and behaviour: it is the per-sequence API applied to
`own`. A workspace with no `own` (`mynah_slm_state_init_workspace`) is what a
scheduler holds.

`mynah_slm_generate` becomes a driver over `mynah_slm_gen`: `gen_prefill`
(up to N prompt tokens per call — the slice), `gen_next_token`,
`gen_accept_logits` (sample, stop test, channel split, detokenize, callback),
`gen_finish`. ONE implementation of the token loop, used by the CLI, the
serialized server and the scheduler — rule 2, and the qwen-asr lesson it
comes from.

**KV per request, at admission.** `mynah_slm_seq_reserve` sizes the cache to
what the request can use (prompt + max_tokens, capped by the server's
context), REUSES the slot's existing allocation when it is big enough and of
the same precision, and reallocates only otherwise — so a steady-state
admission allocates nothing, and nothing ever allocates in the token loop.
Ported from asr PR #3's slot (`stream` "pooled: opened lazily, reset per
session") and tts's per-request KV sizing; NOT ported: asr's ring/slide KV
layouts (an encoder's sliding window — a decoder's history only grows) and
tts's windowed compaction (no Qwen3 layer is windowed).

**Gate** (`tests/test_synth`): (1) two sequences interleaved token by token on
ONE workspace produce logits bit-identical (memcmp) to each run alone on its
own state — the split shares nothing a sequence can see; (2) `generate()` on
the fixture's tokenizer emits the same ids and text as before the refactor
(greedy, recorded from the pre-refactor binary), and a hand-driven
`gen_prefill` in slices of 5 + `gen_accept_logits` loop emits the same ids as
`generate()`; (3) `seq_reserve` reuses a big-enough allocation (same
pointer) and reallocates a too-small one.

(2) was written as "`generate()` == a hand-written greedy loop" (forward one
token at a time, argmax, stop on EOS) rather than against recorded ids, so it
does not pin the fixture's weights; it was run and PASSED on the
pre-refactor `generate()` first, then on the new one.

**Result (2026-10-08, cloud x86, both fixtures)**: all PASS — two bf16-KV
sequences interleaved on one workspace equal each alone by memcmp (logits and
6 greedy ids); a 400-position sequence on a 256-position workspace is refused;
`generate()` == greedy reference (24 ids); hand-driven `gen` with 6 prefill
slices == `generate()` (24 ids, same text, stop = LENGTH); `seq_reserve`
reuses / regrows / retypes as specified.

**Not covered**: the hybrid short-conv path on a sequence (no LFM2 fixture);
the CLI and the server still call the unchanged single-sequence API, so they
exercise `own` only.

## Conclusion

(open)

## Next action

S1-b, then S1-c (batched decode across sequences).
