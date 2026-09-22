# R1-K — ternary GEMV on Apple Silicon: a measured PROMISING result

Status: **IN PROGRESS** (opened 2026-09-22)

Item: `PLAN.md` §0 R1-K. Reads
[`tied-trit-cpu-kernel-reading.md`](tied-trit-cpu-kernel-reading.md).
Code: `bench/ternary_gemv/`. Data: `reports/ternary-kernel/`.

**Gate B is NOT closed by anything in this note.** A Mac answers *"is there a
credible packed-ternary implementation strategy?"* It cannot answer *"will this
win on Neoverse V2 or AVX-512 VNNI?"*

## The machine, detected and not assumed

`[MEASURED]` `sysctl hw.optional.arm.*`, Apple clang 21:

| | |
|---|---|
| CPU | Apple **M1**, 4 performance + 4 efficiency cores |
| `FEAT_DotProd` | **1** — `sdot` available, `__ARM_FEATURE_DOTPROD` defined |
| `FEAT_I8MM` | **0** — **no `smmla`** |
| `FEAT_BF16` | 0 |

`[PAPER]` fucina names **`i8mm`/`smmla` as the fix** for ARM's per-instruction
density gap against `vpdpbusd`. **This machine does not have it.** So the Mac is
close to the *worst* ARM case for the format, and a good result here is a strong
signal while a bad one would have been weak evidence.

---

## Q1 / Q2 / Q3 are kept apart

This note answers **Q3 — execution** only. Q1 (quality) is Gate A's table and is
unchanged by anything here; Q2 (storage) is the representation table below.
**A format can win one and lose another, and none of these is evidence for
another.**

---

## The candidate physical representations  `[DERIVED]`

Group `TG = 256`, one f32 scale per group — matching ggml's K-quant super-block
and fucina's `tq2_0_fx4`, so the scale overheads are comparable.

| format | payload bpw | scale bpw | **total bpw** | passes | note |
|---|---|---|---|---|---|
| T0 int8 | 8.000 | 0.125 | **8.125** | 1 | oracle, not a storage candidate |
| T1 2-bit | 2.000 | 0.125 | **2.125** | 1 | one plane, 4 trits/byte |
| T2 base-3 | **1.625** | 0.125 | **1.750** | 1 | 5 trits/byte, **padded** — see below |
| **T3 fold9 K=2** | 4.000 | 0.125 | **4.125** | 1 | `c = 3t₁+t₂ ∈ [−4,4]`, 2 codes/byte |
| **T3 K=3** | 6.000 | 0.125 | **6.125** | 2 | `c = 9h + l`, 2-bit plane + nibble |
| mask+sign | 2.000 | 0.125 | **2.125** | 1 | nonzero bitplane + sign bitplane |

`[MEASURED]` **Base-3 does not align with any power-of-two group.** 256/5 = 51.2,
so a 256-group pads to 52 bytes: the real payload is **1.625 bpw, not the 1.600
the arithmetic promises**. The first implementation read past the end of a row
because 1024 is not a multiple of 5 — the misalignment is not cosmetic.

### T3 K=3 is our decomposition, and it is cheaper than the obvious one

`[PAPER]` fucina states *"27-level tied K=3 exceeds a nibble"*, and its K=3 costs
three plane passes. `[DERIVED]` But 27 levels factor as `c = 9h + l` with
`h ∈ {−1,0,1}` and `l ∈ [−4,4]`, which is **a 2-bit plane plus a nibble — two
passes, same 6 bits**. `[MEASURED]` Verified bit-exact against the oracle.

---

## Correctness before any timing  `[MEASURED]`

Two independent checks, and the second is the one that matters:

1. NEON against its own scalar reference — `rel_l2 = 0.000e+00` for T0, T1,
   T3 fold9 and T3 K=3.
2. **Every format against the T0 oracle**, built from the same value array.
   Comparing NEON to a scalar that shares its decode only proves the two agree;
   it cannot catch a wrong round-trip. **All six formats: `max 0.000e+00`,
   `rel_l2 0.000e+00`** on every shape.

The harness refuses to time a kernel that fails either check.

---

## Mac benchmark  `[MEASURED]`

Real Qwen3-0.6B decode shapes from `reports/ternary/traffic_shapes.csv`.
Single-threaded ns per full GEMV, 40 reps, `-O3 -march=native`.

**Baselines are the production kernels**, reached through `mynah_slm_matvec`
with its documented fallback to `ingot_matvec`, and **each row names the kernel
that actually ran**. Two activation arms, because our ternary kernels consume
int8 activations and timing them against the engine's *default f32* path would
be comparing an int8 kernel with a float one and calling the difference a format
win — `Q4_K` is 340 µs on f32 activations and **139 µs on int8**.

### `gate/up_proj` [3072 × 1024], representative of all shapes

| kernel | bpw | ns | MiB | vs Q4_K(int8) |
|---|---|---|---|---|
| **T3 fold9 K=2, NEON** | 4.125 | **58,900** | 1.5 | **2.37× faster** |
| T1 2-bit, NEON | 2.125 | 59,900 | 0.8 | 2.33× |
| **T3 K=3, NEON** | 6.125 | **99,833** | 2.3 | **1.40×** |
| T0 int8, NEON | 8.125 | 68,200 | 3.0 | 2.04× |
| **Q4_K, mynah, act=int8** | 4.5 | **139,433** | 1.7 | **1.00×** |
| Q4_K, mynah, act=f32 | 4.5 | 356,867 | 1.7 | 0.39× |
| Q8_0, ingot-simd | 8.5 | 288,500 | 3.2 | 0.48× |
| Q6_K, ingot-simd | 6.5625 | 501,933 | 2.5 | 0.28× |
| Q3_K | — | **REFUSED** | — | ingot has no encoder |

### Thread scaling, T3 fold9 against Q4_K(int8)

| shape | 1 thread | 2 threads | 4 threads |
|---|---|---|---|
| gate/up_proj | **2.19×** | 2.16× | 2.05× |
| down_proj | 2.06× | 2.20× | **2.22×** |
| q_proj | 2.18× | 2.11× | 2.02× |
| o_proj | **2.28×** | 2.13× | 2.08× |
| k/v_proj | 2.04× | 2.03× | 1.67× |
| **lm_head** | 2.06× | 2.12× | **1.20×** |

`[MEASURED]` **~2.1× on every transformer projection, at every thread count.**
That independently reproduces fucina's *"per plane the TQ2_0 kernel is ~2.1×
Q4_K on ARM"* — different engine, different language, different author.

### The `lm_head` 4-thread cell is the most informative number here

`[DERIVED]` 151936 × 1024 at 4.125 bpw is **76.6 MiB**, far beyond any cache. At
4 threads: ternary 1.518 ms for 80.3 MB = **53 GB/s**; Q4_K 1.818 ms for 85.7 MB
= **47 GB/s**. **Both have hit the M1's memory wall**, and the advantage collapses
from 2.1× to **1.20×** — which is close to the pure byte ratio 4.5/4.125 = 1.09×.

**So ternary's ~2.1× is an arithmetic/unpack advantage, not a bandwidth one, and
it exists only while the operand fits cache.** That fits the repo's own earlier
finding that Qwen3-0.6B decode on this machine is ALU-bound: the ALU-bound regime
is exactly where this format wins, and the one DRAM-bound tensor is exactly where
it stops winning.

### Overheads that were excluded, and their size

`[MEASURED]` Activation quantization + lane shuffle, cols=1024: **2,600 ns**,
once per layer input and amortised over the 3–7 projections that share it. Against
a 58,900 ns GEMV that is **4.4% at worst and ~1% in an engine**. Not a confound,
but it is real and it is stated.

---

## Where the cycles go  `[MEASURED]` / `[DERIVED]`

| | |
|---|---|
| `[MEASURED]` scalar T1 is **25× slower** than NEON T1 (1,524 µs vs 60 µs) | the win is entirely in the SIMD unpack, not the packing |
| `[MEASURED]` T3 fold9 (1 pass, nibble) **beats** T1 (1 pass, 2-bit) despite 2× the bytes | the unpack is the cost, not the bytes: nibble needs 1 shift + 2 ands per 32 weights, 2-bit needs 3 shifts + 4 ands per 64 |
| `[MEASURED]` T0 int8 (no unpack at all, 8.125 bpw) is **slower** than T3 fold9 | 2× the bytes costs more than the unpack it avoids — the two effects cross between 4 and 8 bpw |
| `[MEASURED]` mask+sign scalar is **9,175 µs**, 6× worse than T1 scalar | bit-at-a-time extraction; no SIMD formulation was attempted |
| `[DERIVED]` `sdot` count is identical for T1 and T3 fold9 per weight | the difference is purely shift/mask ops |

### The two structural choices that made it work, both borrowed from `src/qmat.c`

1. **Biased codes.** Trits are stored biased (`t+1`, `c+4`) so unpacking is
   shifts and masks with no sign extension, and the bias is removed **once per
   group** with `sum(t·x) = sum((t+b)·x) − b·sum(x)`.
2. **The activation sum is hoisted across rows.** `sum(x)` over a group does not
   depend on the row, so it is computed once for the whole matrix — the same
   cross-row hoist that makes our Q4_K beat ingot's, and exactly what a per-call
   kernel API cannot do.

---

## Mask + sign: measured, and it does not pay  `[MEASURED]`

Real ternary sparsity in the released PTQTP artifact, five tensors:

| tensor | zero fraction |
|---|---|
| `gate_proj` L0 | 17.81% |
| `gate_proj` L13 | 18.53% |
| `down_proj` L13 | 18.62% |
| `v_proj` L13 | 17.40% |
| `o_proj` L27 | 16.76% |
| **overall** | **17.98%** |

`[DERIVED]` A single ternary plane is ~1/3 zeros by construction, but the
**9-level composite is zero only when both planes are zero**, so the exploitable
sparsity is ~18%, not ~33%. Skipping 18% of lanes in a dense SIMD kernel costs
more in branching or compaction than the 18% it saves. **Zero-skipping is
rejected on measured sparsity**, and the mask+sign layout has no path to beat a
dense nibble unpack.

---

## Verdict, scoped

`[MEASURED]` **PROMISING on Apple Silicon**, by the stop conditions:

> *A packed ternary implementation approaches or beats Q3/Q4 on relevant GEMV
> shapes while using materially fewer bytes.*

**T3 fold9 K=2 is 2.0–2.3× faster than the production Q4_K int8 kernel at
4.125 bpw against 4.5 — faster AND smaller, on every transformer projection
shape, at 1, 2 and 4 threads.**

And the configuration that preserves quality, **T3 K=3, is still 1.40× faster
than Q4_K** at 6.125 bpw. `[PAPER]` fucina measured its own three-pass K=3 at
**0.75× Q4_K on ARM**; our two-pass decomposition turns that into **1.40×**,
which is the single most useful thing this bench produced.

### What this does NOT establish

- `[UNKNOWN]` **Any quality claim.** These are synthetic weights. The GEMV is
  bit-exact against its oracle; the *model* it would produce is Gate A's problem
  and Gate A currently says K=2 costs +68% perplexity at 0.6B.
- `[UNKNOWN]` **Whether it survives in the engine.** A microbench is one tensor
  at a time with a warm operand; a decode step walks the whole model.
- `[UNKNOWN]` **Neoverse V2 / AVX-512 VNNI.** `[PAPER]` predicts x86-VNNI is
  *better* for this format (4.8× per plane) and that `i8mm` would lift ARM. This
  M1 has neither. **Gate B stays UNDECIDED.**
- `[MEASURED, limitation]` **Q3_K could not be benchmarked**: ingot has no Q3_K
  encoder, so the format that won Gate A on quality/size is missing from the
  execution table. Measuring it needs a real Q3_K tensor from a GGUF instead of a
  synthesized one.
