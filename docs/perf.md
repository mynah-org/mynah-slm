# Performance — measured, not estimated

All numbers from an M-series Mac, weights **staged locally** (`scripts/use_model.sh`),
`Qwen3-0.6B-Q4_K_M.gguf`, `-O3 -march=native`, no `-ffast-math`. Thread count is
stated per measurement — it is never implied.

Reproduce with `mynah-slm run ...`; every run prints its own line to stderr.

---

## Where it stands

```
$ mynah-slm run -m models-local/Qwen3-0.6B-Q4_K_M.gguf -p "Ciao! Come stai?" -n 24 --temp 0
Ciao! Sto bene! Cosa ne hai?
[load 0.02s | prompt 19 tok, prefill 28.9 tok/s | gen 12 tok, decode 27.1 tok/s | TTFT 697 ms | 8 threads]
```

**27.1 tok/s decode, from 4.1 where this started** — 6.6x, in steps that were
each found by measuring rather than by guessing:

| | decode tok/s | TTFT |
|---|---|---|
| single-threaded, stock ingot | 4.1 | 5680 ms |
| + threading (8) | 14.6 | 1326 ms |
| + fused Q6_K kernel (ingot) | 24.0 | 802 ms |
| + fused Q4_K kernel (ingot) | **27.1** | **697 ms** |

The last two were the same bug in two places, and the second only became
visible once the first was fixed: Q6_K went from 2.8x slower per element than
Q4_K to nearly 2x faster, which is what made Q4_K worth looking at at all.

### The pattern, twice

Both kernels decoded a super-block into a 256-float scratch array and then
dotted it — 1 KB written and re-read per block. Q6_K did it for every row
(it had no kernel of its own at all and fell through to the shared
decode-then-dot helper). Q4_K's "dual" kernel did it for half of them: two rows
per iteration, but only the second went straight into accumulators, and the
fused path it needed already existed in the same file, used only for the odd
tail row.

Neither showed up as a missing kernel, because `ingot_has_kernel()` answered 1
for Q4_K and Q6_K alike. What exposed it was a per-tensor measurement showing
Q6_K moving 1.46x the bytes of Q4_K in 2.84x the time.

### Threading scales as measured, not as hoped

M-series, 4 performance + 4 efficiency cores:

| threads | prefill tok/s | decode tok/s | TTFT | vs 1 thread |
|---|---|---|---|---|
| 1 | 3.5 | 4.1 | 5680 ms | 1.00x |
| 2 | 8.1 | 7.6 | 2475 ms | 1.85x |
| 4 | 13.8 | 12.9 | 1456 ms | 3.15x |
| 6 | 14.8 | 14.3 | 1349 ms | 3.49x |
| 8 | 15.1 | 14.6 | 1326 ms | **3.56x** |

The plausible rule — "performance cores only, or the region waits on the slow
ones" — turns out to be wrong on this machine: 4 threads give 12.9 and 8 give
14.6, a further 13%. Counter-based claiming in `parallel_for` is what saves it
(a slow core simply claims fewer chunks), which is also why chunks outnumber
threads 4:1. The default is therefore every online core, and `-t N` is how you
find out whether that holds on yours.

### The threading is bit-identical, and that is checked

Every parallel region writes to a disjoint output slice: matvec splits by row,
attention splits by head with per-head scratch. Nothing reduces across threads,
so no summation order changes.

That is verified rather than asserted — the same parity dump, serial and
threaded, compared against each other:

```
embed/layer0/layer13/layer27/final_norm/logits   rel=0.00e+00  absmax=0.00e+00
argmax                                           19/19 positions agree
```

Exactly zero, not "within tolerance". `MYNAH_SLM_THREADS=1` forces the serial
path so the A/B stays reproducible, and `tests/test_parity` runs threaded by
default — a gate that only ever sees one thread would prove nothing about the
other. Clean under ThreadSanitizer.

## Where the remaining time goes

Per-matvec, averaged over repeated calls, weights warm, **single-threaded** —
these are the ratios threading then divides, and they still hold:

| tensor | type | shape | before | after |
|---|---|---|---|---|
| `attn_q` | Q4_K | 2048 x 1024 | 0.55 ms | **0.46 ms** |
| `ffn_up` | Q4_K | 3072 x 1024 | 0.84 ms | **0.70 ms** |
| `ffn_down` | Q6_K | 1024 x 3072 | 2.47 ms | **0.54 ms** |
| **`lm_head`** (= `token_embd`, tied) | Q6_K | **151936 x 1024** | 126.75 ms | **42.55 ms** |

Adding it up: ~6.2 ms per layer x 28 layers = ~173 ms, plus ~127 ms for the LM
head, gives ~300 ms per token — the 4.1 tok/s that was observed on one thread,
so nothing significant was hiding elsewhere.

Two findings, both actionable:

### 1. The LM head is 42% of a decode step, on its own

Threading divides it but does not change its share, so this stays the largest
single item.

One matvec, 151936 x 1024, every single token. This is the concrete cost of the
tied embedding noted in `docs/models.md`: it is not a lookup table, it is the
hottest GEMV in the loop.

It also makes a quantization question urgent rather than academic. `token_embd`
is held at Q6_K by the `Q4_K_M` recipe. At Q4_K's measured throughput the same
matvec would take ~48 ms instead of ~127 ms — a 26% cut in total decode time
from one tensor. Whether that costs output quality is exactly what M5b has to
measure; it sits in front of the softmax that picks the token, so the answer is
not obvious in either direction.

### 2. ingot's Q6_K matvec was 2.8x slower per element than Q4_K — FIXED upstream

`ffn_up` (Q4_K) and `ffn_down` (Q6_K) are the same shape, yet Q6_K moved 1.46x
the bytes and took 2.84x the time. Not bandwidth, then. `ingot_has_kernel()`
reported 1 for both, which is what hid it: Q6_K had no kernel of its own at all
and fell through to dequantize-into-a-256-float-scratch-then-dot, with a scalar
decode loop. Q4_K and Q5_K both had hand-written NEON paths; Q6_K did not.

Fixed in ingot (`Q6_K: fused NEON matvec`), decoding straight into the
accumulator and hoisting the scale multiply out of the element loop — each run
of 16 outputs shares one scale:

| | before | after | |
|---|---|---|---|
| `ffn_down` `[1024 x 3072]` | 2.47 ms | **0.54 ms** | 4.6x |
| `lm_head` `[151936 x 1024]` | 126.75 ms | **43.96 ms** | 2.9x |

Q6_K now runs at 5.87 G elem/s against Q4_K's ~2.9, i.e. faster per element
than the type it was 2.8x behind. Correctness rests on ingot's own suite, which
already checks every format's matvec against its own dequant; cross-checked
separately against `ingot_dequant_matrix` at 6.7e-08 relative error, and the
engine's 1e-6 parity gate is unchanged.

**A build bug hid the win for one measurement.** `$(INGOT_LIB)` was only an
order-only prerequisite, so the rebuilt archive did not relink the binaries:
end-to-end tok/s did not move at all while the microbenchmark showed 2.9x,
which reads as "the kernel does not matter" rather than "the kernel is not
there". It is a real prerequisite now.

## Batched prefill: 26 -> 292 tok/s, and TTFT 7.6 s -> 0.85 s

Prefill used to be the decode path in a loop: one token, every weight read,
repeat. Reading 400 MB of weights to advance one position is the definition of
memory-bound, and a prompt pays it per token.

`mynah_slm_forward_batch()` runs N tokens at once. A weight strip is
dequantized once and multiplied against all N activations with `sgemm`
(`src/qmat.c`), so the weight read is amortized N-fold and the problem stops
being memory-bound.

Measured, `Qwen3-0.6B-Q4_K_M`, 8 threads, local weights. Two prompt lengths,
because they are bounded by different things:

**198 tokens** — a tool-calling turn, the interactive case:

| batch | prefill tok/s | TTFT |
|---|---|---|
| 1 (the old path) | 26.3 | 7568 ms |
| 16 | 80.6 | 2501 ms |
| 32 | 134.1 | 1517 ms |
| 64 | 184.1 | 1122 ms |
| **128** (default) | **244.2** | **852 ms** |
| 256 | 292.2 | 721 ms |

**2275 tokens** — a transcript to summarize, which is what the ASR→SLM→TTS
pipeline actually does:

| batch | prefill tok/s | TTFT |
|---|---|---|
| 1 (the old path) | 21.5 | 105841 ms |
| 32 | 79.7 | 28596 ms |
| 64 | 92.4 | 24675 ms |
| **128** (default) | **101.7** | **22431 ms** |
| 256 | 107.0 | 21330 ms |
| 512 | 100.5 | 22691 ms |

**9.3x on a short prompt, 4.7x on a long one — and the gap between those two
numbers is the next bottleneck talking.** If prefill were still bound by the
projections, tok/s would not care how long the prompt is. It cares a lot: 292
at 198 tokens against 107 at 2275. That is attention, which is O(n²) over a
prompt and is still computed one query at a time. Batching it is the next
change, and it is the one that matters for summarization.

Decode is unchanged, as it must be: a batch of one goes down the single-token
path verbatim.

### It is a reorder, and that is checked

`sgemm` sums in its own order, so batched prefill is *not* bit-identical to N
matvecs. `tests/test_batch.c` holds it to being a reorder and nothing more:

```
width 128: rel=1.47e-06 at 99734, argmax 576 vs 576, n_past 95 vs 95
```

1.5e-06 against the 1e-4 the parity gate holds layer 0 to, the same argmax, and
— separately checked, because wrong K/V positions would still score the prompt
fine — twelve greedy tokens generated after the prompt, identical either way.

This is also why the product is ours and not `ingot_matmat`: from two tokens up
ingot's batched Q4_K quantizes the ACTIVATIONS to int8, ~2.4e-3 relative. That
is 20x this gate. Fine for a container library's general-purpose kernel,
not fine for a path that has to agree with decode.

## What has not been done yet

- **Batched attention.** The projections are batched; attention is not. Query t
  may read exactly `pos+t+1` keys, so a batch is a triangle rather than a
  rectangle, and sharing the KV read across it is a separate piece of work.
  The two prefill tables above are the measurement that says it is now the
  thing worth doing.
- **SIMD in our own kernels: measured, and mostly not worth it.** See below.
- **Q4_0 has a kernel now** (NEON + AVX2, 7-10x over the generic path), which
  changes nothing for Qwen3 — it is Q4_K/Q6_K — and everything for Gemma 4,
  whose QAT checkpoints ship as Q4_0. Untested end to end until that model is
  downloaded.
- **x86 is compile-verified only.** `make check-x86` in ingot passes at
  AVX2+F16C and AVX-512, but this machine is aarch64: the AVX2 paths have never
  executed. That is a limitation, not a claim.

## The scalar kernels: one of them mattered, four did not

The note here used to say our own kernels "will matter once the matvecs get
faster". Measured, at a 19-token prompt, per token, single-threaded:

| kernel | calls/token | per token |
|---|---|---|
| attention | 28 | 1.306 ms |
| SwiGLU | 28 | 0.193 ms |
| RMSNorm (plain + per-head) | 113 | 0.122 ms |
| RoPE | 56 | 0.008 ms |
| residual add | 56 | 0.004 ms |
| **total non-matvec** | | **1.632 ms** |

Against ~120 ms of matvec on one thread, that is **1.4%**. Vectorizing RMSNorm,
RoPE or SwiGLU would have been effort spent on a rounding error.

**Attention is the exception, and only because it grows with the context.**
Per token across 28 layers, single-threaded:

| n_kv | scalar | SIMD | share of a 37 ms step (scalar) |
|---|---|---|---|
| 32 | 1.42 ms | 0.47 ms | 4% |
| 128 | 5.06 ms | 2.31 ms | 14% |
| 512 | 20.17 ms | 5.97 ms | 55% |
| 2048 | 106.18 ms | 41.43 ms | 287% |
| 8192 | 564.27 ms | 299.07 ms | 1525% |

Every benchmark above this section used a 19-token prompt — precisely where
attention does not show. Summarizing a meeting transcript, which is what the
ASR→SLM→TTS pipeline exists for, is a 2000-token prompt, where scalar attention
costs nearly three times everything else combined.

End to end, interleaved A/B, 60 generated tokens:

| | prefill | decode |
|---|---|---|
| n_kv ~19, SIMD vs scalar | — | 27.1 vs 27.1 tok/s |
| n_kv ~1189, SIMD | 22.7 / 22.5 | **20.5 / 19.7** tok/s |
| n_kv ~1189, scalar | 20.9 / 20.7 | 17.1 / 16.2 tok/s |

About +9% prefill and +20% decode at long context, nothing at short — which is
what the profile predicted.

**A caution about how that was nearly got wrong.** The first end-to-end A/B was
a single non-interleaved run of 30 tokens, and it said SIMD was 11% SLOWER in
decode. That was noise on a machine warm from an hour of benchmarking. Two
interleaved runs of 60 tokens reversed it consistently. Measure differences,
interleaved, with a sample big enough to survive the thermal state — a single
number here would have led to reverting a real improvement.

## Method notes

- Weights are staged on the local disk for every measurement
  (`scripts/use_model.sh`). Read over a network share the same full weight read
  takes 17.0 s against 7.5 s, and page-cache pressure re-faults over SMB
  mid-run, which makes a number unreproducible rather than merely slow.
- TTFT is stamped at the first token, never reconstructed from a total.
- Prefill tok/s and decode tok/s are different numbers and are never quoted for
  one another.
