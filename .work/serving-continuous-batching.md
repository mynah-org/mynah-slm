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

### S1-c — one decode step for B sequences

**What.** `mynah_slm_forward_multi(ws, seqs[B], tokens[B], logits[B])`
advances B independent sequences by one token each. Per layer: the B rows'
norms, then ONE product per weight over the [B x d] activations
(`mynah_slm_qmatmat`, the batched-prefill product — or B matvecs, see the
switch below), then per row QK-norm,
RoPE at that sequence's own position and the K/V write into that sequence's
own cache, then attention for all B sequences in ONE pool region over
(sequence, head) pairs, each over its own history. The LM head is one
[B x d] product too. B = 1 calls the single-token path verbatim.

**Cost model, written before the code** (`engineering-method.md` §3):

| field | |
|---|---|
| current cost | B streams = B full decode steps. 0.6B Q4_K_M on M1, 8 threads: ~27 ms/step (36.5 tok/s, `docs/perf.md`), of which the weight walk is the bulk; attention is 4% at n_kv 32 |
| suspected cause | each step walks every weight for ONE token: ~400 MB of quantized bytes per token per stream |
| transformation | one product per weight over B rows: the bytes are read once for B tokens |
| max plausible saving | weight traffic /B. At B = 4: up to ~3x aggregate tok/s IF the step stays memory-bound and the product's per-weight work does not grow with B |
| new work | (a) `qmatmat` DEQUANTIZES each 128-row strip to f32 before the sgemm: per-weight decode work is paid once per step (same as a matvec) but in a separate pass that writes and re-reads 4 B/weight through L2; (b) sgemm at M = B = 2..8 is a skinny GEMM, where vendor BLAS is weakest; (c) 2 pool regions per strip (dequant + sgemm) instead of 1 per matvec — the LM head alone is 1187 strips |
| risk | (a)-(c) can make one [B x d] qmatmat SLOWER than B fused matvecs at small B: the prefill product was tuned for T = 128-256, never for T = 2..8. Then batching would cost throughput, not buy it |
| smallest experiment that kills it | per decode step on the real 0.6B geometry (the fixture's `fixture_spec_06b_shape`: real shapes, noise weights — speed does not depend on values): B x single-token step vs one `forward_multi` step, interleaved in one process, B = 1, 2, 4, 8 |

Because of that risk the product is a **seam with a switch**, not a
hard-wired call: `MYNAH_SLM_DECODE_PRODUCT=matmat` (one `qmatmat` per
weight, weights read once) or `=matvec` (B threaded matvecs per weight —
weights read B times). The design intent was matmat by default; the
measurement below reversed that before commit, so the shipped default is
matvec. The resolved choice is
readable (`mynah_slm_decode_product_name`) so a benchmark can prove which ran.
The follow-up that would remove the risk is sibling-port-map row 6 — a
weight-stationary batched quantized matvec (B rows against one Q4_K row,
no f32 strip) — and it is NOT in this step: a kernel change is separate work
and `src/qmat.c` belongs to another agent.

**Why attention is one region.** Each (sequence, head) task runs the exact
per-head code the single-token path runs (`attention_head` /
the packed-KV head), so a sequence's attention output is bit-identical to its
solo decode; one region per layer instead of B keeps the dispatch count of a
step at the single-token step's (~225 regions on 0.6B), not B times it. Load
balance across sequences of different lengths comes from the pool's atomic
claiming.

**Step isolation.** `forward_multi` validates every sequence (fits the
workspace, has room, token in range, no sequence twice) BEFORE touching any
of them, and advances no `n_past` unless the whole step succeeded. A failure
mid-step can leave K/V written at each sequence's position `n_past` — exactly
the slot the next attempt overwrites — so re-stepping a sequence alone is the
step it would have taken alone. That is the property tts's `step_isolate`
relies on, and the scheduler (S1-d) uses it the same way.

**Gate** (`tests/test_synth`, both fixtures): B = 1..8 sequences at
DIFFERENT positions (different prompt lengths, f32 and bf16 KV): each
sequence's logits in the batch vs its logits decoded alone, `rel < 1e-4` with
the same argmax, and continued greedily for several steps in the batch vs
alone; B = 1 is memcmp-identical to `seq_forward`; a batch containing an
invalid sequence fails WITHOUT advancing any `n_past`.

**Gate result (2026-10-08, x86, both fixtures, KV f32 and bf16 mixed in one
batch, 6 greedy steps per B)**: PASS for B = 1..8 in both product modes.
`matvec`: memcmp-identical to solo at every B (the same kernels in the same
order — which also proves the attention region, the per-row RoPE positions and
the per-sequence KV writes are exact). `matmat`: worst rel 2.3e-06 (F32
fixture) / 2.3e-07 (Q4_K_M mix), same argmax every step; B = 1 memcmp. A batch
holding a full sequence, or one sequence twice, returns -1 and moves nobody.
Mutation: K written one slot late for batch row 1 → 29 checks fail (after
the cache poisoning of S1-a was applied to this test too).

**The smallest experiment — it killed the default.** `tests/bench_decode`
(`make bench-decode`), 0.6B geometry with noise weights (Q4_K/Q8_0/Q6_K mix,
151936-row tied Q6_K head), position 128, bf16 KV, 4 threads, cloud 4-vCPU
x86, weights on local tmp, arms interleaved in one process. **CONTENDED**:
other agents were running (load average 3-6), so magnitudes are soft; the
direction held at every B in both runs.

| B | solo ms | matmat ms | matvec ms | matmat vs solo | matvec vs solo |
|---|---|---|---|---|---|
| OpenBLAS, 7 rounds: 1 | 65.5 | 69.2 | 63.8 | 0.95x | 1.03x |
| 2 | 166.4 | 779.2 | 152.1 | **0.21x** | 1.09x |
| 4 | 240.7 | 920.4 | 211.4 | **0.26x** | 1.14x |
| 8 | 505.8 | 1478.7 | 485.1 | **0.34x** | 1.04x |
| own sgemm (`MYNAH_SLM_SGEMM=own`), 5 rounds: 2 | 178.9 | 581.3 | 149.4 | 0.31x | 1.20x |
| 4 | 340.3 | 772.1 | 331.0 | 0.44x | 1.03x |
| 8 | 524.0 | 663.8 | 483.7 | 0.79x | 1.08x |

Reading it, with the control that would embarrass the explanation:

- **One `qmatmat` per weight is 3-5x SLOWER than B separate steps** at
  B = 2..8. The cost model's risk row, measured. The own-sgemm control keeps
  the loss (0.31x at B = 2), so it is not OpenBLAS's second thread pool
  fighting ours: it is the product's shape — every weight is dequantized to
  f32 into a strip (4 bytes written and re-read per weight) before a skinny
  sgemm, a cost amortized at prefill's T = 128-256 and not at T = 2..8.
- **`matvec` multi-step is 1.03-1.20x faster than solo**, from what it does
  share: one attention region per layer for all B and one step call; the
  weights are still read B times. Not the continuous-batching win — that
  needs the weights read once.
- **Decision: the default product is `matvec`** (bit-identical to solo, never
  slower here); `matmat` stays behind `MYNAH_SLM_DECODE_PRODUCT=matmat` as
  the A/B arm. The weight-read-once win needs sibling-port-map row 6 — a
  weight-stationary kernel that multiplies B rows against each Q4_K block
  while it is in registers, no f32 strip — which is kernel work in
  `src/qmat.c` (another agent's file) and a separate, measured item. The seam
  it plugs into is `project_rows` in `arch_qwen3.c`.
- Unexplained, left as a question rather than a story: solo B = 2 costs 2.6x
  a solo B = 1 in the first run (166 vs 65 ms) instead of 2x. Contention is
  the cheap hypothesis; the per-arm minimum-of-3 should have absorbed most of
  it. Re-measure on an idle host before reading anything into it.

### S1-z — no zombie work: a client that leaves stops costing CPU

Added to the queue by the user mid-task, done before the scheduler so that
S1-d is built on it. Port of mynah-tts `sink_cancelled` (polled once per step,
`stream_out_peer_gone`), its 5 s send timeout, and "backpressure is
cancellation, never a blocking write".

**Audit of the serialized server before this change** (from the coordinator,
confirmed by reading `server/main.c` / `server/http.c` at `2f9977b`):

| # | defect | consequence |
|---|---|---|
| 1 | a disconnect was noticed only when a token was WRITTEN (`answer_cb` → `http_write` fails) | never during a prefill (~10 s at 2275 tokens), never on the thinking channel (discarded, never written), never while a tool call accumulates, never in non-streaming mode: generation ran to max_tokens holding the model while every other client waited |
| 2 | a client that left while QUEUED on `infer_mu` still got its whole generation | pure waste, in front of live clients |
| 3 | no `SO_SNDTIMEO` | a client that stops reading without closing fills the TCP window; `send()` blocks forever inside the lock: the whole server stalls |
| 4 | an unbounded detached thread per connection, no refusal | overload parks clients invisibly |

**Fix.**

- `http_peer_gone` / `http_fd_peer_gone`: one `poll(2)` with zero timeout
  (`POLLIN | POLLRDHUP` on Linux), and a one-byte `MSG_PEEK | MSG_DONTWAIT`
  when readable — EOF is gone, a pipelined byte is not. Sticky, and set too by
  a failed or timed-out send. ~1 µs; asked once per step, so free against a
  10-60 ms step.
- `mynah_slm_gen_params.cancel`: polled before every prefill batch and every
  decode step, on every channel; stop = `CANCELLED`, `timing.cancelled = 1`
  (the CLI summary line says `CANCELLED`). generate() returns the tokens it
  made. Granularity: one decode step, or one prefill batch (≤ `batch_max`,
  256 tokens by default).
- The serialization point is a gate (flag + condvar), not a bare mutex: a
  queued request wakes every 20 ms, probes its client and leaves without
  running; probed again once inside. The model is released at the step
  boundary, the state freed, a log line written
  (`[chatcmpl-N cancelled: client gone during decode|prefill, K tokens
  generated of max_tokens M, prompt P]`), nothing sent.
- Every accepted socket: `TCP_NODELAY` (was already set), `SO_SNDTIMEO` 5 s
  (`--send-timeout-ms`), `SO_RCVTIMEO` 30 s while the request is read,
  `MSG_NOSIGNAL`.
- `--max-conns N` (default 64): one more connection gets
  `503 + Retry-After: 1` at accept — request bytes drained without waiting,
  then `shutdown(SHUT_WR)`, so the client reads the 503 rather than a reset
  (the hole tts names in its own fail-fast path).
- `/health`: `running`, `waiting`, `connections`, `rejected`, `cancelled`
  (= `cancelled_queued` + `cancelled_running`).

**Evidence (cloud x86, contended; model-free)**

| check | result |
|---|---|
| `tests/test_http` (loopback TCP): idle alive; pipelined byte alive and not consumed; FIN gone after 1 probe; RST gone; fd -1 gone | PASS |
| `tests/test_synth`: cancel hook fired on its first call (the prompt's only batch) → 0 tokens, `n_past` 0; fired before decode step 5 → exactly 4 tokens, `n_past` = prompt-1+4, `timing.cancelled` 1 | PASS, both fixtures |
| `tests/test_server_cancel.sh` (`make test-server-cancel`) on the "slow" fixture (4 wide layers, noise, 8.5 ms/token here, so 8000 tokens ≈ 68 s): next request after a non-stream / stream / mid-prompt disconnect | 0.24 / 0.28 / 0.46 s (a 4-token request alone: 0.23 s) |
| ... a client that left while queued | never run (`cancelled_queued` 1) |
| ... CPU 1.3 s after the last client left (`/proc/pid/stat`) | 0 ticks in 1 s |
| ... `/health` `cancelled`, log lines for decode and prefill | 6; present |
| ... 4th connection with `--max-conns 3` | `503`, `Retry-After`, `rejected` 1 |
| CONTROL: the same script against the pre-fix server (`2f9977b`, `CAP_ARGS=""`) | **all 11 checks FAIL**: the next request after each disconnect waited past curl's 60 s cap (the abandoned 8000-token generation runs ~115 s); the server burned 362 ticks/s (3.6 of 4 cores) with every client gone; no 503 at the cap |

**Downstream** (validation matrix): with a real checkpoint, LOCAL weights,
`make test-server` (its new case: a 2000-token request abandoned after 1 s,
stream and non-stream, then an 8-token request within 3x its solo time + 2 s,
`/health` `cancelled` ≥ 2) and `make test-server-cancel MODEL=...`
(`bash tests/test_server_cancel.sh <model>`). By hand:
`curl --max-time 1 ... max_tokens 4000` against a long generation, then
`pidstat -p $(pgrep mynah-slm-server) 1` — CPU at idle within one step — and
a second request's `ttft_ms` equal to its value on an idle server.

**Not covered**: the hybrid short-conv path (same hook, no fixture); a client
that half-closes its write side but keeps reading (POLLRDHUP calls it gone —
HTTP/1.1 clients do not do that, and tts made the same call); prefill
cancellation granularity is one batch, so a 256-token batch on a large model
can still run ~1 s after the client left.

### S1-d — the scheduler, model-free

`src/sched.{c,h}` (policy over opaque jobs, engine behind callbacks) and
`src/jobq.{c,h}` (the bounded pending queue). One thread runs
`mynah_slm_sched_run` and is the only thread that enters the model. One
iteration: **admit → reap → prefill → step**, with retirement inline.

| rule | ported from | why it fits an LLM | here |
|---|---|---|---|
| one scheduler thread owns the model and the pool | tts `scheduler_main`, asr `sched.c` | a second submitter on the pool serialized behind the regions and lost its cohort (tts TTFA 167 → 1200 ms); a decode step is ~225 pool regions on 0.6B, all ours | `sched_run` on one thread; HTTP threads only push to the queue |
| admit at the top of every iteration; block only when nothing is live | tts `admit_pass` | a live stream's next token must not wait for an empty queue | `admit_pass(block = live == 0)` |
| reap cancelled jobs every iteration, and before admission, and between prefill slices | tts `sink_cancelled`, polled per step | S1-z: no zombie work; the slot and its KV are free for the next admission | `reap_pass`, the pre-admit check, the between-slices check |
| prefill in slices, FIFO to completion, a wall-time budget per iteration, ≥ 1 slice always, uncapped when nobody decodes | tts `prefill_slice_budget`, `prefill_fifo`, `prefill_step_budget_s` | a 2275-token prompt is ~10 s here; inline it would freeze every live stream for that long. FIFO beat round-robin by 29% TTFA p95 in tts with every request class improving | `prefill_slice` 32 tokens, `prefill_budget_s` 40 ms (tts's qualified values, **not measured here**; `MYNAH_SLM_PREFILL_SLICE`, `MYNAH_SLM_PREFILL_STEP_MS`) |
| a prefill that completes steps in the same iteration | tts `slots_prefill_slice` | TTFT pays the slicing, not a loop round trip | prefill pass runs before the step pass |
| one step call for all decoding jobs; on failure re-step each alone, retire only who fails alone | tts `step_live` / `step_isolate` | one request's failure retires one request; `forward_multi`'s all-or-nothing contract makes the re-step exact | `step_pass` |
| bounded queue, fail-fast refusal | tts `job_enqueue` (503) | a wait no metric sees is worse than a refusal | `jobq_push` returns -1, never waits |

**Not ported, on purpose** (tts measured each and it lost, or the reason does
not exist here): a low-priority prefill helper thread (TTFA p95 435 → 2379 ms);
global cross-worker batching (arrivals coincide 1.6% of the time); a second
submitter on the pool; utilization-aware admission (stall@250 → 50%); emit
quanta and the quantum ramp (audio has a playback clock to stay ahead of; a
text stream emits one token per step); prefork + fd passing (a scaling step
after S1 is measured, `sibling-port-map.md`); async admission helpers
(admission here is a KV reserve that `seq_reserve` makes allocation-free in
steady state).

**Gate**: `tests/test_sched` (fake engine logging every call, fake clock with
a fixed cost per prefill token and per step, so budgets are exact) — admission
FIFO and never more live than slots; a freed slot reused; a live slot plus an
empty queue does not block; closing the queue drains; no slice longer than
`prefill_slice`; exactly 2 slices of 320 ms inside a 500 ms budget; the
decoding stream steps every iteration through two long prefills; FIFO to
completion; same-iteration first step; uncapped when alone; a cancelled
decoding job retires at the next iteration, is never stepped again, and its
slot goes to the waiting job; cancelled while queued → never admitted;
cancelled mid-prompt → not prefilled further; a poisoned job retires FAILED
and its batch-mate steps exactly once; a refused admission is never
prefilled; the queue refuses when full or closed; and a threaded run (4
producers x 60 jobs, one scheduler thread, some clients leaving) where every
queued job retires exactly once and every refusal is counted — also under
ThreadSanitizer.

### S1-e — the server, behind `--slots N`

`server/slots.{c,h}` is the model-facing engine for the scheduler, and
`server/main.c` picks it with `--slots N` (N > 1). **`--slots 1` is the
default and is the serialized path of S1-z unchanged**: the default does not
move until a real checkpoint has measured the new one.

| piece | how | ported from |
|---|---|---|
| ownership | the scheduler thread owns the model, the pool, the workspace, every slot's sequence, every request's sampler and generation; the connection thread owns its socket and becomes that request's WRITER | tts `scheduler_main` + `stream_out`; asr PR #3 slot ownership |
| slot pool | one `mynah_slm_seq` per slot, `seq_reserve`d at admission to prompt + max_tokens + 8 (capped by `--ctx`), reused across requests: no allocation in steady state, none ever in the token loop; bf16 KV as before | tts `kv-window-allocation.md`, asr `slot.c` "pooled, reset per session" |
| bounded admission, fail-fast 503 | requests in the system (queued + holding a slot) are counted at ARRIVAL; `live + queued >= slots + --queue` (default queue 2 x slots) → `503` + `Retry-After: 1` from the connection thread, before anything is written; `/health.rejected_queue`, `capacity` | tts's ladder: `running + queued >= slots + queue_cap` → 503 |
| SSE header at admission | the writer sends it when the scheduler marks the request admitted, so a client's first byte means "you have a slot" and TTFB is not TTFT | tts `sink_next_job`, serving-design §7 |
| per-stream writes that never block the step | the scheduler appends SSE frames to the request's buffer and signals; the writer drains it into the socket unlocked. Unsent bytes > 1 MiB (`MYNAH_SLM_STREAM_MAX_BYTES`), a failed send, a 5 s send timeout or the peer-gone probe mark the request gone; the scheduler reaps it at the next iteration | tts `stream_callback` ("backpressure is cancellation, never a blocking write"), `stream_out.c` |
| TCP_NODELAY, timeouts, connection cap | from S1-z, unchanged | — |
| timings | per request, clock started at ADMISSION (like the lock in the serialized path), TTFT stamped at the first token; `usage` adds `queue_ms` (arrival → admission) and `slots`, so queueing is reported, never folded in or hidden | repo rule "speed is always reported" |
| /health | `slots`, `live`, `preparing`, `decoding`, `queued`, `queue_cap`, `steps`, `mean_batch`, `aggregate_decode_tok_s` (tokens over the last 10 s of steps), `recent_stream_decode_tok_s` (mean of the last 16 finished streams), `slot_cancelled/failed/done`, `decode_product` (the dispatch that ran, resolved before the scheduler starts) | AGENTS.md "/health exposes recent decode t/s" |

**Evidence (cloud x86, contended, model-free — `tests/test_server_slots.sh`,
`make test-server-slots`, on the "slow" fixture: 4 wide layers, untied
noise head, ~8.6 ms/token here)**

| check | result |
|---|---|
| 4 reference answers from `--slots 1` (seeded sampling, temperature 0.8) | 4 distinct answers |
| the same 4 prompts concurrently on `--slots 4 --queue 2` | **byte-identical** to the serialized server's, and a fifth, streamed, reassembles to the same text |
| were the steps batched? (`/health`) | mean batch 2.45 over 49 steps, `decode_product` matvec |
| first token of a 37-token request admitted beside 3 long streams | 0.41 s at the client; server side `queue_ms` 9.7, `ttft_ms` 348 from admission |
| 4 slots busy + 2 queued, a 7th request | `503` + `Retry-After` in 0.06 s; `rejected_queue` 1 |
| all clients leave | `live` 0, next request 0.2 s, 0 CPU ticks in 1 s, `slot_cancelled` 9 |
| ThreadSanitizer server (`BLAS=none`, `-fsanitize=thread`) driven by this test and by `test_server_cancel.sh` | **0 reports** (timing checks fail under TSan's ~10x slowdown, as expected) |
| mutation: swap two sequences' input tokens inside the batched step | the concurrent-equality and stream checks FAIL |

Found and fixed while making this pass:

1. **The 503 was judged on the queue alone**: a burst of 5 requests on 4
   free slots got 503s, because they arrived faster than the scheduler's next
   admission pass and the 2-deep queue overflowed with slots standing free.
   The refusal is now tts's rule — `live + queued >= slots + queue`, counted
   at arrival (`/health.capacity`).
2. **Noise weights with a tied head echo their input**: the residual stream
   stays near the input token's embedding, whose self-logit (~|h||E|, ~600)
   beats every other (~55), so every answer was the prompt's last token ("\n")
   repeated and "same answer" comparisons were vacuous. The "slow" fixture now
   has an untied head (control rows zeroed in both matrices); the test asserts
   its reference answers are distinct before comparing them.
3. **Shutdown freed the model under detached connection threads** (TSan,
   serialized path, pre-existing): `main` now waits (bounded, 30 s) for the
   live-connection counter to reach zero — the counter's decrement is each
   thread's last act, so the wait is also the ordering — and exits without
   freeing if threads remain.

Noted, not acted on: `ttft_ms` 348 for a 37-token prompt means the two
prefill slices (32 + 4 tokens) cost far more than two decode steps — the same
f32-strip cost S1-c measured for `qmatmat` at small T, since prefill slices
run through it. A 32-token slice may be the wrong default for this engine
(tts's value, for a different per-token cost); measure slice 32 / 64 / 128
against TTFT and inter-token gaps on a real checkpoint before changing it.

**Response format is shared**: the tail of a completion (tool-call parse,
finish reason, final SSE frames or the JSON body) is one function,
`send_completion`, used by both modes; the slot path's usage carries the same
members plus `queue_ms` and `slots`.

## Conclusion

The serving foundation exists and is proven model-free: per-request state,
a multi-sequence decode step whose default product is bit-identical to solo
decode, a scheduler whose every policy rule has a unit test, no zombie work
in either serving mode, and `--slots N` end to end on a synthetic checkpoint.

What is NOT shown, and is the question S1 exists to answer: that batching
makes the server FASTER. On this VM the only batched product that reads the
weights once (`matmat`) is 3-5x slower than B solo steps, so the default
reads them B times and gains only 3-20% from sharing the attention region and
the step. Continuous batching here buys fairness, admission, isolation and
cancellation today; it buys throughput only once a weight-stationary batched
quantized kernel exists. Verdict for the code: **KEEP** (behind a flag, the
default unchanged); for the throughput claim: **INCONCLUSIVE** until measured.

## Next action

1. Downstream, on a host with a checkpoint (weights LOCAL): the rows of the
   validation matrix in the final report — `make test-server`,
   `make test-server-cancel MODEL=...`, `make test-server-slots` with the
   model as `$1`, `make bench-decode`, and a concurrent-client load test at
   C = 1, 2, 4, 8 with `--slots 1` and `--slots 8`.
2. The weight-stationary batched kernel (sibling-port-map row 6) as its own
   item: B rows against each Q4_K block while it is in registers. It plugs in
   at `project_rows`; `bench_decode` is its A/B harness.
3. Upstream ingot: Q8_0 dequant for rows that are not whole 256-blocks.
4. Re-measure the prefill slice / step budget defaults for this engine.
