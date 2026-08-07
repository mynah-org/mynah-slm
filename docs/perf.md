# Performance — measured, not estimated

All numbers from an M-series Mac, weights **staged locally** (`scripts/use_model.sh`),
`Qwen3-0.6B-Q4_K_M.gguf`, `-O3 -march=native`, no `-ffast-math`. Thread count is
stated per measurement — it is never implied.

Reproduce with `mynah-slm run ...`; every run prints its own line to stderr.

---

## Where it stands

```
$ mynah-slm run -m models-local/Qwen3-0.6B-Q4_K_M.gguf -p "Racconta una storia breve su un faro." -n 60 --temp 0
[load 0.01s | prompt 24 tok, prefill 127.5 tok/s | gen 60 tok, decode 36.5 tok/s | TTFT 218 ms | 8 threads]
```

**36.5 tok/s decode, from 4.1 where this started** — 8.9x, in steps that were
each found by measuring rather than by guessing:

| | decode tok/s | TTFT (19-tok prompt) |
|---|---|---|
| single-threaded, stock ingot | 4.1 | 5680 ms |
| + threading (8) | 14.6 | 1326 ms |
| + fused Q6_K kernel (ingot) | 24.0 | 802 ms |
| + fused Q4_K kernel (ingot) | 27.1 | 697 ms |
| + **our Q4_K matvec** (src/qmat.c) | **36.5** | — |
| + **batched prefill** | 36.5 | **318 ms** |

Two of those steps were the same bug in two places, and the second only became
visible once the first was fixed: Q6_K went from 2.8x slower per element than
Q4_K to nearly 2x faster, which is what made Q4_K worth looking at at all — and
then made ITS kernel the slow one, which is the step after that.

Prefill is a separate number and moved further: **26 -> 370 tok/s** on a
198-token prompt once it stopped being one token at a time. Never quote one for
the other.

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

**This paragraph used to recommend requantizing it to Q4_K, and that was
wrong** — kept here rather than deleted, because the way it went wrong is the
point. It was written when Q6_K had no kernel and ran at a third of Q4_K's
speed, so moving fewer bytes obviously won. Two kernel fixes later the ranking
had inverted, and `make bench` says the head runs at **29.0 G elem/s at Q6_K**
while the comparable Q4_K tensors reach 25-26. Requantizing it would move 32%
fewer bytes on a slower kernel and come out behind.

So `token_embd` at Q4_K is a **footprint** decision (121.7 MiB -> ~83 MiB, on a
396 MB file), not a speed one, and it still has to clear M5b's quality gate
because it sits in front of the softmax that picks the token. A number that was
true stops being true when the thing under it changes; that is what the bench
is for.

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

## Our own Q4_K matvec: decode 28 -> 36.5 tok/s

`make bench` measures each projection, and it said something specific: Q4_K —
five of the seven tensors in a layer — ran at 11-14 G elem/s while Q6_K reached
18-25. Q4_K was the slow kernel, and it is 65% of a decode step.

A Q4_K weight is `d * scale[s] * q - dmin * min[s]`. The obvious kernel
materializes that per element: unpack the nibble, widen it, multiply, subtract,
then FMA against the input — three vector ops before the one that counts.
Distribute the sum instead:

```
SUM_j w_j x_j  =  d*scale * SUM_j (q_j x_j)  -  dmin*min * SUM_j x_j
```

The per-element multiply and subtract disappear; the scales apply once per
32-weight sub-block. And `SUM_j x_j` depends only on the INPUT, so it is
computed once per matvec and reused across every row — 2048 of them for
`attn_q`. **That hoist is why the kernel lives in the engine and not in a
container library**: it needs a per-call preamble the row loop then reads,
which is a different API shape, not a faster loop.

Interleaved A/B in one process, ours against ingot's, best of three rounds:

| tensor | type | ours | ingot | |
|---|---|---|---|---|
| `attn_q` | Q4_K | 0.125 ms | 0.197 ms | **1.58x** |
| `attn_k` | Q4_K | 0.065 | 0.100 | **1.53x** |
| `attn_out` | Q4_K | 0.089 | 0.140 | **1.57x** |
| `ffn_gate` | Q4_K | 0.123 | 0.188 | **1.53x** |
| `ffn_up` | Q4_K | 0.121 | 0.187 | **1.54x** |
| `attn_v` | Q6_K | 0.072 | 0.073 | 1.01x |
| `ffn_down` | Q6_K | 0.154 | 0.151 | 0.98x |
| `lm_head` | Q6_K | 5.361 | 5.321 | 0.99x |

The Q6_K rows are the control: we do not touch that type, and they land on
1.00x. A ratio that moved there would mean the harness was measuring the
machine rather than the kernel — which is exactly what two *separate* runs did,
disagreeing by 80% on those same untouched tensors. Interleaved, in one
process, best-of-N.

End to end, interleaved, 60 generated tokens, four runs across two sessions:

| | decode |
|---|---|
| ours | 36.5 / 36.5 / 35.2 / 34.9 tok/s |
| ingot | 28.0 / 27.8 / 26.6 / 26.3 tok/s |

**+30 to +32%, and the greedy output is byte-identical.** The ratio is what is
stable; the absolute number drifts about 5% with the thermal state of the
machine, which is exactly why the comparison is interleaved and why a single
run of each would be worth nothing. (The lower pair was measured after killing
two idle server processes left over from a test run — they were using no CPU,
but they held ~800 MB of mapped weights on a 16 GB machine. Check what else is
running before quoting an absolute.) Correctness is gated in two
places rather than assumed: `tests/test_kernels.c` checks our kernel against
ingot's on synthetic Q4_K weights with no checkpoint (`rel=3.5e-07`, and ingot
is the right oracle there because it is cross-checked against llama.cpp), and
the parity gate still holds at `1.50e-06` on layer 0 with 19/19 argmax
agreement.

## KV cache precision: 4-bit keys are ruinous, 4-bit values are nearly free

At a 2275-token context the KV caches are **~520 MB — larger than the model**,
and attention re-reads them on every generated token. So their precision is the
biggest single memory item and a real share of decode time.

The memory saving is arithmetic and known in advance. The QUALITY is not, and
that is the only part worth measuring, so `mynah-slm ppl` exists: mean negative
log-likelihood of a fixed text, teacher-forced, same text and same threads with
only the format changing. Greedy output cannot answer this — text diverges as
readily from a 1e-6 difference as from a ruinous one, because all it takes is
one argmax flipping early.

917 tokens across it/en/de/fr/es/ja/zh, `Qwen3-0.6B-Q4_K_M`:

| K | V | ppl | vs f32 | bits/value |
|---|---|---|---|---|
| f32 | f32 | 2.802 | — | 32 / 32 |
| bf16 | bf16 | 2.801 | -0.0% | 16 / 16 |
| **q8** | **q8** | **2.803** | **+0.0%** | 8.5 / 8.5 |
| q8 | f32 | 2.799 | -0.1% | 8.5 / 32 |
| fp8 | fp8 | 2.817 | +0.5% | 8 / 8 |
| bf16 | q4 | 2.817 | +0.5% | 16 / 4.5 |
| f32 | q4 | 2.835 | +1.2% | 32 / 4.5 |
| **q8** | **q4** | **2.871** | **+2.5%** | 8.5 / 4.5 |
| q4 | q8 | **26.776** | **+855%** | 4.5 / 8.5 |
| q4 | q4 | 27.583 | +884% | 4.5 / 4.5 |

**K and V are not equally sensitive, and the gap is enormous.** A key goes
through the softmax exponent, where its error is amplified before anything
normalizes it; a value is averaged with weights that sum to one, where errors
partly cancel. Four-bit keys destroy the model — 10x the perplexity, and the
generated text switches language mid-sentence. Four-bit VALUES, with the same
9.4% quantization noise, cost 0.5-1.2%.

That asymmetry is the practical result: **quantize V harder than K.**

**q8 beats fp8 at the same 8 bits** (2.799 vs 2.817). One scale per 32 values
adapts to the block; e4m3 spends 4 of its 8 bits on an exponent that a
well-scaled block does not need. Worth knowing before reaching for fp8 because
the hardware has a name for it.

The quantizer itself is gated rather than trusted — `tests/test_kernels.c`
checks each format's round-trip error against its bit budget (bf16 1.6e-3,
q8 5.2e-3, fp8 2.6e-2, q4 9.4e-2 RMS relative), because the first question
about a 10x perplexity is whether the tool is broken. It is not: q4 really is
9.4% noise, and keys do not survive it.

### The packed cache: bf16 is 2x smaller AND 26% faster, and is now the default

The formats above were first measured by ROUND TRIP — quantized and dequantized
straight back into an f32 cache — which isolates the quality question from the
plumbing. `src/kvcache.c` now stores them packed for real. Measured at a
2400-position context, 28 layers, decode with 2275 tokens of history, mean of
four runs:

| K / V | KV cache | vs f32 | decode | ppl |
|---|---|---|---|---|
| f32 / f32 | 550 MB | — | 18.8 tok/s | 2.802 |
| **bf16 / bf16** (default) | **275 MB** | **2.0x** | **23.7 tok/s (+26%)** | **2.802** |
| q8 / q8 | 146 MB | 3.8x | 19.7 tok/s | 2.797 |
| q8 / q4 | 112 MB | 4.9x | 18.9 tok/s | 2.834 |

**bf16 wins on every axis at once** — half the memory, a quarter more decode,
and a perplexity identical to f32 to three decimals — so it is what the CLI and
the server now use. `--kv f32` gets the reference back; `--kv-k q8 --kv-v q4`
is for when memory is the binding constraint and 4.9x is worth 1.1% of
perplexity.

That ordering is not what the byte counts predict, and getting to it took two
corrections worth keeping:

**The first packed version was SLOWER than f32** — 11.5 tok/s against 14.5 —
while reading a quarter of the bytes. Its dot and accumulate were scalar loops,
competing against an f32 path that dots with 16-wide NEON. Compression only
becomes speed once the decode is vectorized too.

**Then bf16 measured at 10.3 tok/s, half of f32.** It had no fused path at all:
it decoded a slice into scratch and dotted that, per position, per head. That
number was measuring the missing kernel, not the format — and publishing it
would have buried the best option in the table. With a fused widen-and-FMA it
went from 10.3 to 23.7.

Which also explains why q8 gains so little despite reading four times less: an
int8 lane must be widened twice and converted to float before it can enter an
FMA, so a memory-bound loop turns into a compute-bound one. bf16 needs a single
shift, and keeps the saving.

Timing note: single runs disagreed by 25% (16.1 and 20.7 tok/s for the same
configuration). Every figure here is the mean of four, taken after killing two
idle server processes that were holding ~800 MB of mapped weights; at that
point best and mean agree to 1%.

## Batched prefill: 26 -> 370 tok/s, and TTFT 7.6 s -> 0.58 s

Prefill used to be the decode path in a loop: one token, every weight read,
repeat. Reading 400 MB of weights to advance one position is the definition of
memory-bound, and a prompt pays it per token.

`mynah_slm_forward_batch()` runs N tokens at once. A weight strip is
dequantized once and multiplied against all N activations with `sgemm`
(`src/qmat.c`), so the weight read is amortized N-fold and the problem stops
being memory-bound.

Measured, `Qwen3-0.6B-Q4_K_M`, 8 threads, local weights. Two prompt lengths,
because they are bounded by different things:

Attention is batched too, and it had to be — see the two rounds below.

**198 tokens** — a tool-calling turn, the interactive case:

| batch | projections batched | + attention batched |
|---|---|---|
| 1 (the old path) | 26.3 tok/s, TTFT 7568 ms | — |
| 32 | 134.1 | — |
| 64 | 184.1 | — |
| 128 | 244.2, TTFT 852 ms | 307.6, TTFT 686 ms |
| **256** (default) | 292.2 | **370.5, TTFT 577 ms** |

**2275 tokens** — a transcript to summarize, which is what the ASR→SLM→TTS
pipeline actually does:

| batch | projections batched | + attention batched |
|---|---|---|
| 1 (the old path) | 21.5 tok/s, TTFT 105841 ms | — |
| 64 | 92.4 | 153.1 |
| 128 | 101.7, TTFT 22431 ms | 201.6, TTFT 11348 ms |
| **256** (default) | 107.0 | **231.8, TTFT 9878 ms** |
| 384 | — | 245.8 |
| 512 | 100.5 | 243.8 |

**14x on a short prompt, 10.8x on a long one.**

The two rounds are worth keeping separate, because the first one's *shape* is
what pointed at the second. With only the projections batched, prefill ran at
292 tok/s on a 198-token prompt and 107 on a 2275-token one. A projection-bound
prefill does not care how long the prompt is; this cared enormously, which is
O(n²) attention talking. Batching it lifted the long prompt from 101.7 to 201.6
at the same width — and it barely moved the short one, exactly as the profile
said it would.

Past 256 the long prompt is flat (245.8 at 384, 246.4 at 1024) while the
scratch keeps growing, so 256 is the default. `MYNAH_SLM_BATCH` overrides it.

**Decode is unchanged**, which is the part to check rather than assume: a batch
of one goes down the single-token path verbatim. A first reading suggested
decode dropped from 17.8 to 15.5 tok/s at very wide batches; interleaved
repeats put b=64 at 17.6/17.1 and b=256 at 16.8/17.4, i.e. noise. The same
caution as the SIMD A/B further down — one number here would have invented a
regression that is not there.

### What the batch costs in memory

At width 256 with a 3072-position context: ~19 MB of scratch (activations 12 MB,
attention scores 3 MB, the dequantization strip 1.5 MB). Allocated once at
session init, never in the token loop. It scales with the batch width, which is
the other reason not to take 384 for 6% more prefill.

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

- **Decode is 36.5 tok/s and is NOT yet at the memory roof.** The fastest
  tensor moves 22 GB/s and the machine has far more than that, so decode is
  still limited by the kernel and not by the bus. The Q6_K matvec is now the
  biggest single item (`lm_head`, 5.4 ms of a 26 ms step) and has had no
  equivalent treatment — its scales are int8 per 16 weights rather than a
  packed 6-bit pair, so the same distribute-the-sum trick does not transfer
  unchanged, but the per-element scale multiply is there to remove.
- **Packed KV storage.** The precision question below is answered; what is not
  yet written is the storage that turns the answer into bytes saved. Today the
  formats are measured by ROUND TRIP — quantized and dequantized straight back
  into the f32 cache — which costs the quality without paying the saving.
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
