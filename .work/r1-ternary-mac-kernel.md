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
density gap against `vpdpbusd`.

Stated at the strength the evidence supports, and no further:

- `[MEASURED]` This M1 **has `FEAT_DotProd` and lacks `FEAT_I8MM`**.
- `[HYPOTHESIS]` A Neoverse-V2 implementation using `i8mm`/`smmla` **may** improve
  the ternary result relative to this M1 path. **Axion is the test of that
  hypothesis, and nothing here is evidence for or against it.**

The earlier wording — *"the Mac is close to the worst ARM case"* — is **withdrawn**.
It ranked unmeasured machines from a single measured one. A missing feature bit is
a fact; a position in an ordering of implementations we have never run is not.

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
| Q3_K | — | **REFUSED** | — | ingot has no encoder — measured later off a real GGUF: **462,750 ns**, 3.4x slower than Q4_K |

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

### The `lm_head` 4-thread cell — read the correction below before using it

> **Scope warning.** Everything in this subsection compares ternary against a
> **synthetic `Q4_K` lm_head**. `[MEASURED]` **No shipped GGUF has one**: both
> `q-Q3_K_M.gguf` and `q-Q4_K_M.gguf` store `output.weight` as **`Q6_K`**. See
> *"Thread scaling and the `lm_head` correction"* below.

`[DERIVED]` 151936 × 1024 at 4.125 bpw is **76.6 MiB**, far beyond any cache. At
4 threads: ternary 1.518 ms for 80.3 MB = **53 GB/s**; Q4_K 1.818 ms for 85.7 MB
= **47 GB/s**. **Both have hit the M1's memory wall**, and the advantage collapses
from 2.1× to **1.20×** — which is close to the pure byte ratio 4.5/4.125 = 1.09×.

**Stated at the strength of the evidence:** on the measured **cache-resident
projection GEMVs**, the observed advantage is dominated by
**representation/unpack/kernel efficiency rather than raw weight bandwidth**. On
the large `lm_head` both paths approach the **memory-bandwidth regime** and the
advantage **falls substantially**, towards the byte ratio.

`[HYPOTHESIS]` that this is a general cache-residency threshold rather than a
property of these two particular kernels at these two particular sizes. One shape
crossing one boundary on one machine does not locate that boundary.

It is at least *consistent* with the repo's earlier finding that Qwen3-0.6B decode
on this machine is ALU-bound — but consistency is not confirmation, and no third
measurement was made to separate the two explanations.

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
- ~~`[MEASURED, limitation]` **Q3_K could not be benchmarked**~~ — **closed**, see
  the next section. A `--gguf` mode now multiplies straight off a real
  checkpoint, and Q3_K is measured.

---

# The real-GGUF benchmark: Q3_K measured, and two earlier claims corrected

`bench/ternary_gemv/ternary_gemv --gguf <file>` skips encoding entirely and
multiplies off the bytes of a real quantized checkpoint through the same
`mynah_slm_matvec` call the engine makes. That is the only way to reach Q3_K,
which ingot **reads** but cannot **write**.

Files: `models-local/q-Q3_K_M.gguf` and `models-local/q-Q4_K_M.gguf`, both
produced here from the same BF16 Qwen3-0.6B with the same imatrix
(`quantize.imatrix.chunks_count = 200`, WikiText-2 train). Machine: Apple M1,
`FEAT_DotProd=1`, `FEAT_I8MM=0`. Raw output in
`reports/ternary-kernel/gguf_q3km.txt`, `gguf_q4km.txt`,
`gguf_q4km_repeat.txt`; rows in `gguf_bench.csv`.

## First: a `_K_M` name is a recipe, and the recipe is not what we assumed

`[MEASURED]` type census straight from the containers:

| file | F32 | Q3_K | Q4_K | Q5_K | Q6_K |
|---|---|---|---|---|---|
| `q-Q3_K_M.gguf` | ×113, 0.2 MiB | **×113, 172.0 MiB** | ×81, 91.7 MiB | ×3, 3.4 MiB | **×1, 121.7 MiB** |
| `q-Q4_K_M.gguf` | ×113, 0.2 MiB | — | ×169, 288.2 MiB | — | **×29, 167.7 MiB** |

`[MEASURED]` per-tensor types at `blk.5`:

| tensor | in Q3_K_M | in Q4_K_M |
|---|---|---|
| `ffn_gate` / `ffn_up` | Q3_K | Q4_K |
| `attn_q` / `attn_k` | Q3_K | Q4_K |
| `attn_output` | **Q4_K** | Q4_K |
| `attn_v` | **Q4_K** | **Q6_K** |
| `ffn_down` | **Q4_K** | **Q6_K** |
| `token_embd` | Q3_K | Q4_K |
| **`output` (lm_head)** | **Q6_K** | **Q6_K** |

**Two consequences, and both invalidate a row of the synthetic table.**

1. `[MEASURED]` **The `lm_head` is `Q6_K` in both files.** The synthetic bench
   compared ternary against a *`Q4_K` lm_head*, which **no shipped file
   contains**. Every `lm_head` conclusion drawn from that row describes a
   configuration that does not exist.
2. `[MEASURED]` **The tied embedding is stored twice, at two different rates.**
   `token_embd` + `output` are **185.5 MiB of the 394.8 MiB Q3_K_M file (47.0%)**
   and **205.1 MiB of the 461.8 MiB Q4_K_M file (44.4%)**. This is the GGUF
   counterpart of the duplicate-tied-embedding finding already recorded for the
   safetensors checkpoint — the same tensor, the same waste, a different
   container.

## Q3_K measured, and it is the slowest path we ship  `[MEASURED]`

Single thread, real tensors, `ns` per full GEMV, and **effective GB/s over the
tensor's own bytes** (the fair per-format bandwidth figure):

| tensor | type | bpw | kernel | ns | **GB/s** |
|---|---|---|---|---|---|
| `ffn_gate` [3072×1024] | **Q3_K** | 3.4375 | ingot-simd | **462,750** | **2.9** |
| `attn_q` [2048×1024] | **Q3_K** | 3.4375 | ingot-simd | **294,750** | **3.1** |
| `attn_k` [1024×1024] | **Q3_K** | 3.4375 | ingot-simd | **147,025** | **3.1** |
| `ffn_gate` [3072×1024] | Q4_K | 4.5 | mynah, int8 | 135,375 | 13.1 |
| `attn_q` [2048×1024] | Q4_K | 4.5 | mynah, int8 | 87,975 | 13.4 |
| `ffn_down` [1024×3072] | Q6_K | 6.5625 | ingot-simd | 506,000 | 5.1 |
| `output` [151936×1024] | Q6_K | 6.5625 | ingot-simd | 24,302,550 | 5.3 |
| — T3 fold9 K=2, same shapes | — | 4.125 | ours, NEON | — | **25–27** |
| — T1 2-bit, same shapes | — | 2.125 | ours, NEON | — | **13–14** |

`[MEASURED]` **Q3_K's production GEMV runs at 2.9–3.1 GB/s against Q4_K's
13.1–13.4 — it is 3.4× slower per byte and, at 3.4375 bpw against 4.5,
still ~3.4× slower per GEMV.** So on this machine the shape of the trade is
**not** "Q3 = better quality, somewhat slower". It is *much* slower.

### Why, and the part of it that is not the format's fault  `[MEASURED]`

`[MEASURED]` `mynah_slm_matvec_have()` returns true for **`Q4_K` only**
(`src/qmat.c:130`). Every other type falls through to `ingot_matvec`, which takes
**f32 activations** — which is exactly what the two arms show: Q4_K goes
356,400 → 135,375 ns when the int8 arm is switched on, while Q3_K moves
513,675 → 462,750 ns and Q6_K does not move at all (506,000 → 516,375, inside
the noise). **There is no int8-activation Q3_K kernel anywhere in our stack.**

### The gap decomposes, and most of it is not the bit width

`[MEASURED]` Reading `third_party/ingot/src/kernels.c`: both
`q3_k_dot_block_neon` and `q6_k_dot_block_neon` are **f32-domain** kernels. They
widen the quants `int8 → int16 → int32 → float` and accumulate with
`vmlaq_f32` against f32 activations — four FMAs plus eight widening ops per 16
weights. Our `Q4_K` int8 kernel accumulates with `sdot`: **16 int8 MACs in one
instruction, no widening at all.**

That is measurable independently of the format, because our own `Q4_K` kernel has
**both arms on the same bytes**:

| | GB/s | |
|---|---|---|
| `Q4_K`, ours, **int8** arm (`sdot`) | **13.1–13.4** | |
| `Q4_K`, ours, **f32** arm | **5.0–5.2** | ÷2.6 |
| `Q6_K`, ingot, f32-domain | 5.1–5.3 | same rate as the f32 arm |
| `Q3_K`, ingot, f32-domain | **2.9–3.1** | ÷1.7 again |

`[DERIVED]` So the 4.5× **per byte** gap between our `Q4_K` and ingot's `Q3_K`
factors cleanly:

```
  2.6x   arithmetic domain: int8 sdot vs f32 fmla + widening
x 1.7x   Q3_K's decode complexity vs Q6_K's, both in f32
= 4.5x   per byte
x 0.76   Q3_K's byte credit (3.4375 / 4.5 bpw)
= 3.4x   per GEMV                             <- the measured 462,750 / 135,375
```

**`[MEASURED]` 462,750 / 135,375 = 3.42×, and the decomposition predicts 3.44×.**

The honest reading:

- `[MEASURED]` **The `Q3_K` path we ship today is 3.4× slower than the `Q4_K`
  path.** That is a fact about the engine as it exists and it is what a user
  would feel. It is the number that belongs in any Q3-vs-ternary trade table
  *today*.
- `[MEASURED]` **Most of it is the arithmetic domain, not the bit width.** 2.6×
  of the 4.5× is `f32` versus `sdot`, measured on our own kernel's two arms
  against identical bytes. **`Q3_K` has no int8-activation kernel anywhere in our
  stack**, and that is a gap in our engine, not a property of 3.4375 bpw.
- `[UNKNOWN]` What a `Q3_K` kernel written to the same standard would reach.
  `[DERIVED]` The residual 1.7× says it would still land below `Q4_K` — the
  format really is harder to decode (2-bit payload plus a high-bit mask plus
  6-bit packed scales) — but "3.4× slower" would not survive it. **"Q3_K is
  intrinsically slow" is not a claim this bench supports.**
- `[DERIVED]` The same argument cuts the other way for ternary: T3 fold9's
  25–27 GB/s is **also** an `sdot` number. Against a hypothetical int8 `Q3_K`
  kernel the honest expectation is ~2× or less, not 7.8×. **The 7.8× figure
  compares our best kernel with ingot's worst one**, and it is quoted here only
  as *the gap between the paths that exist today*.

## Ternary against the types the files actually use  `[MEASURED]`

T3 fold9 K=2, NEON, single thread, against the **real** tensor at the same shape.
**Read the decomposition above first**: every row against `Q3_K` or `Q6_K`
compares an `sdot` kernel with an `f32` one, and ~2.6× of each of those ratios is
the arithmetic domain rather than the representation.

| tensor | shape | real type | ternary is |
|---|---|---|---|
| `ffn_gate` | 3072×1024 | Q3_K | **7.79×** |
| `attn_q` | 2048×1024 | Q3_K | **7.30×** |
| `attn_k` | 1024×1024 | Q3_K | **7.26×** |
| `ffn_gate` | 3072×1024 | Q4_K | **2.10×** |
| `ffn_down` | 1024×3072 | Q4_K | 2.21× |
| `attn_output` | 1024×2048 | Q4_K | 2.20× |
| `ffn_down` | 1024×3072 | Q6_K | **7.97×** |
| `attn_v` | 1024×1024 | Q6_K | 7.88× |
| **`output`** | **151936×1024** | **Q6_K** | **7.63×** |

## Thread scaling and the `lm_head` correction  `[MEASURED]`

Cleanest run (`gguf_q4km_repeat.txt`, repeat 1, 120 reps). **Effective GB/s, the
number that says whether a path is bandwidth-bound:**

| tensor | type | 1 thread | 2 threads | 4 threads |
|---|---|---|---|---|
| `ffn_gate` | Q4_K | 13.5 | 26.1 | 50.1 |
| `ffn_gate` | T3 fold9 | 27.4 | 52.6 | **99.7** |
| `ffn_down` | Q6_K | 5.1 | 10.3 | 18.5 |
| `ffn_down` | T3 fold9 | 27.8 | 54.2 | **102.0** |
| **`output`** | **Q6_K** | 5.2 | 9.8 | **16.6** |
| **`output`** | **T3 fold9** | 25.8 | 50.5 | **53.2** |

`[MEASURED]` **T3 fold9 saturates and Q6_K does not.** On the cache-resident
tensors ternary scales to ~100 GB/s; on the 76.6 MiB `lm_head` it stops at
**53.2 GB/s**, which is the M1's DRAM wall and is where its scaling curve breaks
(25.8 → 50.5 → 53.2). `Q6_K` climbs to only **16.6 GB/s** on the same tensor —
**it never reaches the wall at all**, because it is still compute-bound there.

**Correction to the synthetic table.** It reported *"`lm_head` at 4 threads
collapses to 1.20×, both paths are DRAM-bound"*. That is true **of ternary
against a synthetic `Q4_K` lm_head**, and only of that. Against the `Q6_K`
lm_head the files actually ship, ternary at 4 threads is **5.10×**
(7,689,608 ns vs 1,508,883 ns), because only one of the two paths is
bandwidth-bound. Scope both statements or neither:

| `lm_head`, 4 threads | ratio | why |
|---|---|---|
| T3 fold9 vs **synthetic Q4_K** | **1.20×** | both saturate DRAM; ratio → byte ratio |
| T3 fold9 vs **real Q6_K** | **5.10×** | ternary saturates, Q6_K is still compute-bound |

### Measurement noise, stated  `[MEASURED]`

Three runs of the same file. The **ratio** T3/real is stable — 2.0–2.3× against
Q4_K and 7.7–9.0× against Q6_K on every shape at every thread count. The
**absolute** 4-thread numbers are not: `ffn_gate` Q4_K came back as 35,342 /
59,300 / 60,975 ns across the three. The M1 is **4 performance + 4 efficiency
cores** and these threads carry no affinity, so at `nt=4` some land on E-cores.
**The 1- and 2-thread columns are the reliable ones**; 4-thread absolutes are
quoted only where the ratio is what matters.

---

# x86: two of my own hypotheses, falsified by measurement

Machine: **AMD EPYC 9254 (Zen 4)**, 24 cores, 377 GB, `avx512_vnni` + `avx512_bf16`
+ `avx512vbmi`. Raw output in `reports/ternary-kernel/box_*.txt`. **This is not
Axion** — it is x86, so it tests the *other* and stronger prediction in the
literature (`[PAPER]` "~4.8x on x86-VNNI") and leaves `i8mm` still untested.

## The naive reading, and why it is wrong

`[MEASURED]` `gate/up_proj [3072 × 1024]`, 1 thread, 120 reps, idle box:

| kernel | bpw | ns | GB/s |
|---|---|---|---|
| T1 2-bit, avx512-vnni | 2.125 | 42,906 | 19.5 |
| **T3 fold9, avx512-vnni** | 4.125 | **46,787** | **34.7** |
| T0 int8 (oracle, **no ternary**) | 8.125 | 48,801 | 65.5 |
| **Q4_K, ours, int8** | 4.5 | **232,010** | **7.6** |
| Q8_0, ingot | 8.5 | 221,561 | 15.1 |
| Q6_K, ingot | 6.5625 | 390,995 | 6.6 |

Read naively that is **4.74× the production `Q4_K` path**, against 2.2× on the
M1 — apparent confirmation of the paper. It is not. Three controls take it
apart, and **all three contradicted a hypothesis I had already written down.**

## Control 1 — the ISA. HYPOTHESIS FALSIFIED  `[MEASURED]`

My claim was: *our `Q4_K` is slow on x86 because every production kernel we own
is AVX2 with the pre-VNNI `maddubs + madd` pair.* `grep -c _mm512` returned **0**
in `src/qmat.c` and **0** in `third_party/ingot/src/kernels.c`, so the claim was
at least well-founded. I ported the AVX-512 VNNI path from qwen-tts and A/B'd it
on one machine, changing only the library's flags:

| | `objdump \| grep -c vpdpbusd` | `Q4_K` int8 |
|---|---|---|
| `-march=native` | **1** | 228,185 ns |
| `-mno-avx512f -mno-avx512bw -mno-avx512vnni -mno-avx512vl` | **0** | 229,124 ns |

**0.4%, inside the noise**, with `objdump` proving the kernel really changed.

`[DERIVED]` Why, counted afterwards: per 64 elements the kernel does **one**
`dpbusd` against **two `hsum256i`** (≈6 ops each) and **six float scale
operations**. The multiply-accumulate is about **1/20 of the work**. Replacing
4 instructions with 1 could never have bought more than ~12%, and bought 0.4%.
**The MAC instruction was never the bottleneck.**

## Control 2 — the register width. HYPOTHESIS FALSIFIED  `[MEASURED]`

My second claim was that 512-bit ternary against 256-bit `Q4_K` was 4× of unearned
width. The `vnni256` arm runs the identical algorithm at 256 bits:

| format | 512-bit | 256-bit | |
|---|---|---|---|
| T0 int8 | 51,430 | **48,801** | 256-bit is **faster** |
| T3 fold9 | 46,950 | 49,724 | 512-bit +5.9% |
| T3 K=3 | 85,369 | 89,370 | +4.7% |
| T1 2-bit | 42,906 | 51,180 | +19% |

**Register width is worth single-digit percent here, not 4×.**

## Control 3 — the scale granularity. THIS is the cause  `[MEASURED]`

What was left: ggml's K-quants carry an f32 scale **every 32 weights**, so a
256-weight superblock pays **8 horizontal reductions and 8 float epilogues**.
Our ternary block carries **one scale per 256** — and therefore **one**. So
`TG` was made settable from the build and swept, changing nothing else:

**T0 int8 — plain int8 weights, not a trit in sight:**

| TG | bpw | ns | GB/s |
|---|---|---|---|
| **32** (ggml's granularity) | 9.000 | **150,615** | 23.5 |
| 64 | 8.500 | 88,018 | 38.0 |
| 128 | 8.250 | 60,696 | 53.4 |
| **256** | 8.125 | **48,801** | 65.5 |

**`[MEASURED]` 3.09× from scale granularity alone.** Same bit width, same
kernel, same weights, same machine. T3 fold9 shows the same shape: 102,794 ns at
TG=64 → 46,787 at TG=256, **2.20×**.

### The bpw-matched comparison, which is the one that counts

At TG=64 the ternary format costs **exactly 4.5 bpw — the same as `Q4_K`**:

| | bpw | ns | |
|---|---|---|---|
| T3 fold9, TG=64 | **4.500** | 102,794 | |
| `Q4_K`, ours, int8 | **4.500** | 232,307 | **2.26×** |

**So of the headline 4.74×, roughly half was scale granularity and not the
representation at all.** And the bpw-matched x86 figure, **2.26×**, lands on top
of the M1's **2.2×**, where both sides were already our own `sdot` kernels.

`[MEASURED]` **The honest, ISA-independent, bpw-matched ternary advantage is
~2.2× on both architectures measured so far.** The 4.7× was an artifact of
comparing formats at different scale granularities.

### What this hands the engine, independently of ternary

`[DERIVED]` A coarser scale group is worth up to **3.09×** on an int8 GEMV and
costs bits: a scale every 32 weights is **1.000 bpw**, every 256 is **0.125**.
That trade is available to *any* format, including the ones we already ship, and
it is a quality question — coarser scales mean a wider dynamic range per group —
which Gate A has never measured. `[UNKNOWN]` where that curve turns.

### Limitations, stated

- `[MEASURED, limitation]` **T3 fold9 has no vector kernel at TG=32**: the AVX-512
  unpack consumes 128 codes per iteration and the 2-bit one 256, both larger than
  the group. Only the scalar reference ran there, so the T3 row at ggml's own
  granularity is missing and the bpw-matched comparison uses TG=64.
- `[MEASURED, limitation]` `Q8_0`'s baseline moved between runs (221,561 /
  243,715 / 313,545 ns) although `TG` cannot affect it. `Q4_K` was stable within
  0.3% across every run, so `Q4_K` is the baseline quoted; the `Q8_0` spread is
  recorded rather than averaged away.
- `[UNKNOWN]` **`i8mm` on Neoverse V2 is still untested.** Nothing here touches it.
