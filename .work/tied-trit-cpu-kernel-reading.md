# Tied Trit-Planes, read as a kernel paper

Status: **DONE 2026-09-22**

Item: `PLAN.md` §0 R1-K. Feeds [`r1-ternary-mac-kernel.md`](r1-ternary-mac-kernel.md).

Sources, both read directly:

- `[PAPER]` arXiv **2608.08910**, Grella, Aug 2026 — PDF fetched and extracted
  locally with `pdftotext -layout`.
- `[PAPER]` **`fucina/docs/PTQTP.md`** @ `main` — the format specification the
  paper points to for the block layout, fetched raw. This is the implementation
  document and it contains **measurements on Qwen3-0.6B and Qwen3-1.7B** that
  the arXiv paper does not.

Engine: `matteo-grella/fucina`, MIT, Zig, commit `d30ce52` in the paper.
*"fucina is an independent Zig engine whose hot matmul/GEMV kernels are its own;
its block-format encoders and numeric codebooks are documented
operation-for-operation ports from ggml/llama.cpp."*

---

## §1 `[PAPER]` What the authors actually implement

### The identity that makes it a 4-bit format

PTQTP fits `W ≈ α₁T₁ + α₂T₂`. Generic scale ratios give up to nine distinct
composite values; **the composite is uniformly spaced exactly when
|α₁/α₂| = 3**. Constraining `α = (3s, s)` and re-fitting collapses the pair into
a single uniform nine-level quantizer:

```
Ŵ = s · c ,   c = 3t₁ + t₂ ∈ {−4, −3, …, 3, 4}
```

`[PAPER]` §2.1. The constrained family is a subset of the free one, so at the
respective optima the tied fit's reconstruction error is **worse or equal** — and
they measure it worse (≈0.18 per-expert relative Frobenius, which is the same
0.178 our own Method C produces).

### The persistent format

`[PAPER]` `fucina/docs/PTQTP.md`, "Native folded expert format":

| | |
|---|---|
| GGUF type | `tq2_0_fx4`, fucina enum **43** (not stock-GGUF parseable) |
| block | `BlockTQ2_0Foldedx4` — **520 bytes per 4 columns × 256 elements** |
| payload | 512 code bytes, **two 4-bit codes per byte** |
| scales | **four f16 column scales** = 8 bytes |
| rate | **4.0625 bpw** |
| constraints | contract dim % 256, out dim % 4, **K=2 tied only** (27-level K=3 exceeds a nibble) |

`[DERIVED]` `520 × 8 / (4 × 256) = 4.0625`. The scale share is
`4 × 16 / 1024 = 0.0625` bpw.

**"The fold is the file"**: disk bytes, cache slab bytes and kernel input bytes
are identical, with no transcode. Compare the untied sibling format at
**4.125 bpw** (two separate trit planes), and the non-folded resident path at
**~12.3 bpw** (planes + repacks + fold held simultaneously).

### The kernel

`[PAPER]` §2.2: *"codes decode to {−4,…,4} by **arithmetic** (no tables
required), dot against block-quantized activations via integer SIMD (**sdot** on
NEON; **vpshufb/vpsignb/vpdpbusd** on AVX2/AVX-VNNI), **one float multiply-add
per block**."*

So: **no LUT**, int8 lanes, activations block-quantized, scales applied once per
block. The two ISA arms are bitwise-identical by construction — *"exact lane-sum
equivalence between sdot and lane-folded vpdpbusd formulations; shared float
schedule"* — and pinned by tests.

`[PAPER]` `PTQTP.md`: *"the ARM kernel is verified at **~86% of the NEON ALU
roofline** (LLVM emits the fully-folded **10-ops-per-64-weights** sequence; hand
asm has ≤14% headroom), and the ARM↔x86 per-instruction gap is **instruction-set
density** (`sdot` 16 weights/instr vs `vpdpbusd` 32; **i8mm `smmla` would close
it** on ARMv8.6+ targets)."*

---

## §2 `[PAPER]` The measurements that matter to us — they are on Qwen3

`fucina/docs/PTQTP.md`, "Measured (M1 Max, ReleaseFast)". **Their NLL protocol is
teacher-forced over 512 held-out tokens and is NOT our 146×2048 WikiText-2
harness — absolutes are not comparable to ours. Ratios within their table are.**

### Quality, and the finding neither we nor PTQTP's own paper produced

| model / source | baseline | **K=2** | K=2 +down3+o3 | **K=3** |
|---|---|---|---|---|
| 0.6B f16 | 24.96 | 80.47 | 51.69 | **27.36** |
| 1.7B Q4_K_M | 19.41 | 330.96 | — | 21.71 |
| 1.7B BF16 | 18.57 | 184.45 | 45.70 | **18.43** |

`[DERIVED]` ratios: 0.6B K=2 **3.22×**, K=3 **1.10×**; 1.7B K=2 **9.93×**,
K=3 **0.992×** — *below* the baseline.

**Three planes is the result.** `[PAPER]` *"K=3 from the bf16 original matches
the full-precision baseline at 1.7B (statistical parity; the greedy completion is
flawless) and lands within 10% at 0.6B, with weight-space rel err 0.067 on the
27-level bound (~1/3 of dual's 0.179)."*

Their K=2 on 1.7B (9.93×) is **far worse than our measured artifact result
(2.894×)**, and their K=2 on 0.6B (3.22×) is worse than ours (1.683×). Different
solver, data-free vs whatever the artifact used, different protocol. `[UNKNOWN]`
which difference dominates. What survives across all three sources is the
*ordering*: **1.7B degrades more than 0.6B under K=2**, which is now attested by
our measurement, by PTQTP's own table, and by theirs.

### Speed, 1.7B, their engine, M1 Max

| config | t/s | vs bf16 | ppl |
|---|---|---|---|
| bf16 baseline | 20.2–21.0 | 1× | 18.57 |
| K=2 | 47.1–48.3 | 2.3× | 184 |
| **K=3** | **39.5–39.8** | **1.9×** | **18.43** |
| K=3 + K=3 head | 33.9 | 1.65× | 18.40 |

`[PAPER]` *"Against a **Q4_K_M baseline (~44–48 t/s at 1.7B)** K=2 is
**speed-parity** and K=3 **~0.75×**; at 0.6B K=2 is ~1.9× the f16 source and
**parity with Q4_K_M**."*

### The number this whole exercise was trying to find

`[PAPER]` *"Kernel truth is `zig build bench-ternary`: **per plane the TQ2_0
kernel is ~2.1× Q4_K on ARM and ~4.8× on x86-VNNI**."*

---

## §3 `[DERIVED]` What that arithmetic implies for us

A per-plane kernel at 2.1× Q4_K on ARM, with K planes costing K passes:

| K | plane passes | ARM vs Q4_K | x86-VNNI vs Q4_K | quality (1.7B) |
|---|---|---|---|---|
| 1 | 1 | **2.1×** | **4.8×** | collapses (our naive: 32,648×) |
| 2 | 2 | 1.05× | 2.4× | 9.93× theirs / 2.894× the artifact |
| **3** | 3 | **0.70×** | **1.6×** | **0.992× — parity** |

`[DERIVED]` **This is the crux, and it is ISA-dependent exactly as suspected.**
The configuration that preserves quality (K=3) is **slower than Q4_K on ARM and
faster on x86-VNNI**, because the gap is instruction-set density: `sdot` consumes
16 weights per instruction, `vpdpbusd` consumes 32.

`[PAPER]` And the stated fix for ARM is named: **`i8mm`/`smmla` on ARMv8.6+**,
which is exactly what Neoverse V2 has and what the Axion box would test.

### Storage, honestly

`[PAPER]` *"1.7B linears 3.2 GiB bf16 → 693 MiB (K=2) / 1040 MiB (K=3)."*

`[DERIVED]` 1,409,286,144 linear params → K=3 at 1040 MiB is **6.19 bpw**; K=2 at
693 MiB is **4.13 bpw**. So the quality-preserving configuration costs **more
than `Q4_K` (4.5) and about as much as `Q6_K` (6.5625)**.

**Put together: on ARM today, K=3 is bigger than Q4_K_M, slower than Q4_K_M, and
slightly better in quality.** That is a losing trade on this machine and a
possibly winning one on VNNI — which is precisely why Gate B cannot be closed
here.

---

## §4 `[HYPOTHESIS]` What might transfer to dense Qwen3, and what might not

| | |
|---|---|
| `[HYPOTHESIS]` | the 2.1×/4.8× per-plane kernel ratio transfers — it is a property of the *format and ISA*, measured on a GEMV, not of the MoE serving stack |
| `[HYPOTHESIS]` | K=3 quality parity transfers to 0.6B only partially — their own table shows 1.10× at 0.6B against 0.992× at 1.7B |
| `[UNKNOWN]` | whether their K=2 quality (much worse than the released artifact's) reflects the data-free solver or the tie constraint |
| **NOT transferable** | every speed number in the arXiv paper's headline. It is a 284B MoE streamed from SSD where reads dominate; a dense 0.6B decoding from page cache is a different regime, and they say so |
| **NOT transferable** | the −10.6% byte-read win. That is an expert-streaming property |

---

## §5 What this changes about our plan

1. **The execution question has a published answer to beat or refute**: a single
   packed-ternary plane GEMV at **2.1× Q4_K on ARM**. Our microbench's first job
   is to see whether we can reproduce that ratio on Apple Silicon at Qwen3
   shapes. If we cannot get near it, the bottleneck is ours, not the format's.
2. **K=3 must be in the design**, not just K=2. The quality result is the whole
   reason the format is interesting, and a K=2-only bench would measure the
   configuration that does not work.
3. **The folded 9-level code is a strictly better container than two planes** and
   costs 4.0625 bpw. Our T3 candidate should be the fold, not two separate
   streams — with the caveat that **K=3's 27 levels do not fit a nibble**, so K=3
   needs 5 bits or a different packing. That tension is the interesting design
   space.
4. **`i8mm`/`smmla` is the named ARM fix.** Apple Silicon M1 does **not** have
   i8mm (ARMv8.4); M4 does. So this Mac is close to the worst ARM case for the
   format, which makes a negative Mac result even less transferable than usual.
