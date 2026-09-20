# R1-P — PTQTP, read properly: what "2 × 1.58-bit" physically is

Status: **DONE 2026-09-20**

Item: `PLAN.md` §0 R1-P. Feeds [`ternary-feasibility.md`](ternary-feasibility.md).

Sources actually read, not summarised:

- **Paper**: arXiv 2509.16989, **both versions**. `v1` (21 Sep 2025) and `v2`
  (28 Oct 2025) PDFs downloaded and extracted locally with `pdftotext -layout`.
  Every citation below is to a section, equation or table in that text. The
  storage appendix and the Qwen3 result rows are **identical in v1 and v2**.
- **Code**: <https://github.com/HeXiao-55/PTQTP> — *"The codes will be released
  as soon as possible after the review."* **No implementation is public.**
- **Artifacts**: the authors' released checkpoints are under a co-author's
  account, `yang31210999/*-PTQTP-1.58b` (Llama-2-7b, Qwen3-0.6B/1.7B/4B/8B/14B,
  Llama-3-8B-Instruct). `yang31210999/Qwen3-0.6B-PTQTP-1.58b`, sha
  `6dfb5a8fa4055ef2a67a4712342ab1562a2bb969`, was inspected directly.

> **A methodological note that earned its place.** The first pass at this paper
> was a URL handed to a summariser, and it returned *"achieves 4.63× faster than
> the FP16 baseline model"* as a quote from the abstract. **That number does not
> exist in either version of the paper.** The largest inference speedup PTQTP
> claims anywhere is **1.16×** (Table 6). The 17.73-28.79× figure in Fig. 1(b) is
> *quantization* runtime against ARB-LLM, not inference. Nothing derived from a
> paraphrase is load-bearing in this note.

---

## Question

The brief required resolving, before any estimate is used:

1. why the method is described as 2 × 1.58-bit trit planes;
2. why Table 1 reports PTQTP as **1.58** W-bit;
3. what the actual physical/storage representation is;
4. what scale/metadata overhead exists;
5. what operations inference actually requires;
6. which tensors are quantized and which are protected.

All six are answered below, and 3, 4 and 6 come out **different from what this
repo assumed**.

---

## 1-2. Why "2 × 1.58", and why the tables say 1.58

**§3.1.** The decomposition is

> W ≈ Ŵ = Σ_{k=1}^{2} diag(α^(k)) T^(k),  T^(k) ∈ {−1, 0, 1}^{n×d},  α^(k) ∈ ℝ^n

**Both planes are full n × d matrices.** Every weight therefore carries **two**
ternary digits, not one. A trit's information content is log₂3 = **1.585 bits**,
so "2 × 1.58-bit" is a literal and accurate description of the *alphabet*: two
trits per weight, 3² = 9 reachable values per group.

**Table 1's `# Bits` column lists PTQTP at `1.58`.** That column places methods in
categories — FP16 16.00, AWQ/GPTQ 3.00 and 2.00, BiLLM and ARB-LLM_RC **1.06**,
PTQTP **1.58** — so PTQTP sits beside BitNet-b1.58 (a QAT baseline it compares
against in §4) rather than beside the binary methods. **It is the per-plane
alphabet, not the storage rate, and not the sum over the two planes.** The paper
never claims 1.58 bytes-on-disk anywhere; its own storage appendix says something
else entirely.

§3.1 also notes the two planes are *"akin to the residual/error compensation
bit-plane"* of BiLLM/ARB-LLM, *"though PTQTP treats the two planes unanimously"*.
So it is structurally a two-level residual scheme whose levels happen to be
ternary.

## 3-4. The physical representation, and what it really costs

**Appendix A.3, verbatim:**

> A matrix W ∈ ℝ^{n×d} in FP16 format requires 2nd bytes. However, each ternary
> element c_m ∈ {−1,0,1} **can be encoded with 2 bits** (since 3 ≤ 2²), thus two
> trit-planes stored by **nd/2 bytes**. Moreover, two vectors α^(1), α^(2) ∈ ℝ^n
> in FP16 require 4n bytes. Therefore, the total storage complexity can be
> presented as O(nd), achieving a **compression ratio of 4×** for the trit-plane
> relative to FP16.

So the authors' representation is **2 bits per trit × 2 planes = 4.000
bits/weight**, plus scales. That checks: 2 planes × nd elements × 2 bits = 4nd
bits = nd/2 bytes, and 2nd ÷ (nd/2) = 4. **The paper's own stated ratio is 4×,
i.e. 4 bits/weight — not 1.58.**

**Scale overhead, Eq. 9** (`M = n·d·m + ⌈d/k⌉ · n · 16`, group size k): §3.2 sets
**G = 128**, and there are two α vectors, so the grouped overhead is
`2 × 16 / 128 = 0.250` bits/weight. **Grouped PTQTP = 4.250 bits/weight** over
the tensors it quantizes. Row-wise (ungrouped) it is 4.000 + 32/d ≈ 4.03.

§3.2 also records something useful: PTQTP is *"bias-free and mask-free"* — there
is no μ shift and no salient-weight mask, unlike PT²-LLM's (α, μ) and unlike
BiLLM's salient-column bitmaps. So there is no metadata beyond the two scale
vectors.

### The worked example in A.3 is wrong by 4×, in both versions

Immediately after stating 4×, the same paragraph says:

> for a typical LLM layer with n = 1024 and d = 4096, PTQTP reduces storage from
> 8MB to 1.004MB (i.e., **0.5 MB for trit-planes** and **0.504 MB for scaling
> coefficients**), illustrating a **7.96× compression ratio**.

By the paper's own formula on the preceding line: trit-planes are `nd/2` =
1024·4096/2 = **2 MB**, not 0.5 MB; the scales are `4n` = **4 KB**, not 0.504 MB.
Total **2.004 MB → 4.0×**, which is exactly the ratio the paper states two
sentences earlier. **The "7.96×" contradicts its own derivation.**

**Table 4 settles it empirically, with the authors' own measurements:**

| | LLaMA-7B | LLaMA-13B |
|---|---|---|
| FP16 | 13.48 GB | 26.03 GB |
| PTQTP, ungrouped | 3.51 GB | 6.53 GB |
| PTQTP, grouped | 3.69 GB | 6.89 GB |

13.48 / 3.69 = **3.65×**; 13.48 / 3.51 = **3.84×**. Consistent with 4×, nowhere
near 7.96×. Appendix G adds a third slip in the same direction — *"8 ternary
elements are stored in a single byte"* is impossible (3⁸ = 6561 > 256; the
maximum is **5**, which is exactly what ggml's `TQ1_0` does).

Appendix G is also an admission that matters: bit-packing is listed under
**Limitations and Future Works**, i.e. **the representation as implemented is
not packed.**

### What this does to our estimate

The repo's Phase 7a used **3.375 bits/weight** for PTQTP — two `TQ1_0`-packed
planes at 1.6875 including the block scale. That is the **packed-optimal** figure
and it is *better than anything the authors implemented*. Both should be carried,
labelled:

| PTQTP variant | bits/weight on quantized linears | source |
|---|---|---|
| **as implemented by the authors**, grouped G=128 | **4.250** | App. A.3 + Eq. 9 |
| as implemented, ungrouped | 4.03 | App. A.3 |
| **packed-optimal** (base-3, 5 trits/byte, `TQ1_0`-style) | **3.375** | ours; ggml `TQ1_0` geometry |
| the `# Bits` column of Table 1 | 1.58 | a category label, not a size |

**Our 299.2 MiB/token figure was too generous, not too harsh.** Recomputed on
the Qwen3-0.6B census, with the embedding at Q6_K:

| PTQTP variant | linears | + Q6_K embedding | total | vs shipped Q4_K_M (372.7 MiB) |
|---|---|---|---|---|
| as implemented, 4.250 bpw | 223.1 MiB | 121.7 MiB | **345.1 MiB** | **1.08× smaller** |
| packed-optimal, 3.375 bpw | 177.2 MiB | 121.7 MiB | 299.2 MiB | 1.25× smaller |

With a BF16 embedding, as the papers actually run it, the as-implemented variant
is **520.1 MiB — 1.40× LARGER than the file we ship today.**

## 5. What inference requires

**Appendix A.1.** `Ŵ = diag(α^(1))T^(1) + diag(α^(2))T^(2)`, so a GEMV is

    y = α₁ ⊙ (T₁ x) + α₂ ⊙ (T₂ x)

with each `α·c_m` for `c_m ∈ {−1,0,1}` *"implemented as an addition, eliminating
floating-point multipliers and replacing them with sign flips, identity mappings,
or zeroing operations"*, and *"each plane can be processed independently on
parallelized architectures"*. Two full passes over the weight matrix, one per
plane, each an accumulate-with-sign.

The paper's own efficiency argument is explicitly a bandwidth argument:

> This **bandwidth reduction** is critical for latency-bound applications, as
> **memory access often dominates inference time** in modern hardware
> architectures, though there are two parallel trit-planes to represent the
> original weights.

That is the same premise Gate B has to test per ISA, and it is stated as an
assumption in the paper rather than measured.

### The authors' own kernel is slower than 4-bit GPTQ

**Table 5**, `gate_proj` latency in ms on an RTX 4090 — their kernel, their
measurement:

| LLaMA2 | seq | FP16 | GPTQ 4-bit | AWQ 4-bit | AQLM 2×2-bit | **PTQTP** |
|---|---|---|---|---|---|---|
| 7B | 1 | 0.122 | **0.085** | 0.092 | 0.049 | 0.120 |
| 7B | 128 | 0.164 | 0.159 | 0.325 | 5.656 | 0.208 |
| 7B | 2048 | 1.153 | 1.693 | 4.890 | 85.425 | 2.094 |
| 70B | 1 | 0.519 | 0.274 | 0.380 | 0.217 | 0.454 |

At batch 1 — **the case this project cares about** — PTQTP is **1.41× slower than
GPTQ 4-bit** and only 1.02× faster than FP16 on 7B. At seq 2048 it is **1.82×
slower than FP16**. Table 6 (attention, decode) reports 1.141× / 1.087× / 1.163×
over FP16 for 7B/13B/70B. The paper's own text: *"PTQTP delivers inference speed
second only to GPTQ 4-bit."*

**This is the single most decision-relevant fact in the paper for us.** Our
question is whether ternary beats an optimised Q4. On the authors' own hardware,
with the authors' own kernel, **it does not** — on a GPU, which is the platform
they targeted. It says nothing directly about NEON I8MM or AVX-512 VNNI, and it
is not a Gate B verdict. It does mean the burden of proof sits on the ternary
side.

## 6. Which tensors are quantized — the paper does not say, so we measured

**§4.1** says only: *"All linear layers were quantized with tolerance ε = 10⁻⁴
and maximum iterations T_max = 50."* Fig. 2 draws Q, K, V, W_o and the two FC
layers. **Neither the embedding nor the LM head is mentioned anywhere in the
paper.** In HF Qwen3, `lm_head` *is* an `nn.Linear`, so "all linear layers" is
genuinely ambiguous.

So it was resolved against the released artifact instead — and the answer is
**not** "all linear layers".

---

## The released artifact, inspected directly

`yang31210999/Qwen3-0.6B-PTQTP-1.58b` @ `6dfb5a8f`. Inspected by HTTP **range
reads** against the live file — the header first, then individual tensor rows —
so the findings below cost a few hundred KB, not 1.14 GB.

### It is a dense FP16 checkpoint, not a compressed one

| | |
|---|---|
| `model.safetensors` | **1,192,134,784 B = 1136.9 MiB** |
| tensors | **310** (no `lm_head.weight`; the tied weight is stored once) |
| dtypes | **F16 for all 596,049,920 parameters** |

**The artifact ships at 16 bits/weight.** It is a *simulated* (fake-quantized)
checkpoint: the ternary values are materialised back into FP16. There is no
packing, no trit-plane storage, and no scale tensors in the file. This is normal
for a PTQ research release — it is how you evaluate quality without writing a
kernel — but it means **the artifact cannot be used to verify the storage claims
of Appendix A.3**, and anyone reading "1.58b" in the repo name off a 1.14 GB file
should not be surprised.

### The quantized tensors are unmistakably two trit-planes

Reading one row and counting distinct values per group of 128 is a complete test,
because `α₁T₁ + α₂T₂` with `T ∈ {−1,0,1}` can take at most **9** values per group.

`model.layers.13.mlp.gate_proj.weight`, row 0, first 128 elements — exactly 9
distinct values:

```
{ 0, ±0.04310, ±0.09045, ±0.13354, ±0.22400 }
```

which factorises exactly as `α₁ = 0.13354`, `α₂ = 0.09045`:

| combination | value | observed |
|---|---|---|
| α₁ + α₂ | 0.22399 | 0.22400 |
| α₁ | 0.13354 | 0.13354 |
| α₂ | 0.09045 | 0.09045 |
| α₁ − α₂ | 0.04309 | 0.04310 |
| 0 | 0 | 0 |

**§3.1 and §3.2 confirmed empirically**: two planes, row-wise α, group size 128,
no bias term (the set is symmetric about zero, so there is no μ).

### Coverage is 80% of the linears, not 100% — `q_proj` and `k_proj` are untouched

Every projection in all 28 blocks, two rows each, distinct values per 128-group:

| family | blocks with ≤9 distinct (ternary) | verdict |
|---|---|---|
| `q_proj` | **0 / 28** | **not ternarized** |
| `k_proj` | **0 / 28** | **not ternarized** |
| `v_proj` | 28 / 28 | ternarized |
| `o_proj` | 28 / 28 | ternarized |
| `gate_proj` | 28 / 28 | ternarized |
| `up_proj` | 28 / 28 | ternarized |
| `down_proj` | 28 / 28 | ternarized |
| `embed_tokens` | — | **protected**, bit-identical to the original |
| `model.norm` | — | **protected**, bit-identical to the original |

`q_proj` and `k_proj` land at 121-128 distinct values in every 128-element group,
i.e. dense. **This contradicts §4.1's "All linear layers were quantized"**, and
the paper gives no rule that would predict it. Qwen3 is the family with per-head
QK-RMSNorm, so `q_proj`/`k_proj` are the two projections feeding `q_norm`/`k_norm`
— a plausible reason to protect them, but the paper does not say so and we should
not invent it on their behalf.

### The real coverage and the real bit budget

| | params | share of model |
|---|---|---|
| ternarized (v, o, gate, up, down × 28) | 352,321,536 | **59.1%** |
| protected `q_proj` + `k_proj` | 88,080,384 | 14.8% |
| protected embedding (tied, = LM head) | 155,582,464 | 26.1% |
| norms | 65,536 | 0.01% |

**PTQTP covers 80.0% of the transformer linears and 59.1% of the model.** The
40.9% left at high precision is the number that decides the bit budget, and it is
larger than the brief's Qwen3-4B reference point (81.62% coverage).

Effective bits over the **complete** Qwen3-0.6B, at the authors' own 4.250
bits/weight, with the protected tensors where the artifact actually leaves them:

| configuration | MiB | whole-model bits/weight | vs shipped Q4_K_M (372.7 MiB) |
|---|---|---|---|
| the artifact as published (all FP16) | 1136.9 | 16.00 | 3.05× **larger** |
| packed at 4.250 bpw, protected tensors FP16 | **643.3** | **9.05** | **1.73× larger** |
| packed at 4.250 bpw, protected tensors Q6_K | 434.2 | 6.11 | 1.16× larger |
| packed-optimal 3.375 bpw, protected tensors Q6_K | **332.4** | **4.68** | 1.12× smaller |

**Even the most generous reading — a packing the authors did not implement, plus
Q6_K on tensors they left at FP16 — is 1.12× smaller than the Q4_K_M file we ship
today**, and that is before any quality comparison. With the authors' actual
representation and actual protected set, it is **1.73× larger**.

### ~~An unexplained anomaly~~ — RESOLVED: a channel-scale reparameterization the paper does not describe

> **This section previously read "the artifact may be broken". That was wrong
> and is corrected here.** The reconstruction figures below are real, but the
> conclusion drawn from them was not: the artifact scores **35.256** perplexity
> (better than the paper's own 38.02), so it plainly works. Chasing that
> contradiction produced the most useful finding of this note.

Relative reconstruction error `‖Ŵ − W‖/‖W‖` and norm ratio `‖Ŵ‖/‖W‖` against the
original `Qwen/Qwen3-0.6B`, whole tensors:

| block | v_proj | o_proj | **gate_proj** | **up_proj** | down_proj |
|---|---|---|---|---|---|
| 0 | 0.93 | 0.99 | **1.74** | **2.11** | 0.81 |
| 5 | 0.78 | 0.99 | **2.99** | **3.92** | 0.74 |
| 13 | 0.71 | 0.99 | **4.06** | **4.95** | 0.82 |
| 21 | 0.72 | 0.99 | **2.09** | **2.40** | 0.86 |
| 27 | 1.00 | 0.99 | 0.81 | 1.27 | 0.68 |

`gate_proj` L13 is 4.06× too large **on all 3072 rows** (median per-row ratio
4.046, every row above 2), so it is not a sampling artifact either.

**The layernorms absorb it exactly.** Norm ratios of the *norm vectors* against
the original, beside the projections they feed:

| L | input_ln | q | k | v | post_attn_ln | gate | up | **post_ln × gate** |
|---|---|---|---|---|---|---|---|---|
| 0 | 1.032 | 0.949 | 0.950 | 0.935 | 0.574 | 1.741 | 2.108 | **0.999** |
| 5 | 1.253 | 0.790 | 0.794 | 0.778 | 0.334 | 2.988 | 3.924 | **0.999** |
| 13 | 1.314 | 0.729 | 0.729 | 0.706 | 0.244 | 4.061 | 4.951 | **0.990** |
| 21 | 1.277 | 0.725 | 0.738 | 0.718 | 0.480 | 2.090 | 2.399 | **1.004** |
| 27 | 1.000 | 1.000 | 1.000 | 0.986 | 0.860 | 0.813 | 1.265 | **0.699** |

And exactly, per input channel rather than in aggregate. For L13 `q_proj`, with
`s_j = ln_orig[j] / ln_artifact[j]`:

```
|| W_artifact − W_orig · s ||  /  || W_artifact ||  =  2.92e-04     <- fp16 rounding
|| W_artifact − W_orig     ||  /  || W_artifact ||  =  4.12e-01     <- the raw diff
```

**The artifact divides each RMSNorm's per-channel weight by `s_j` and multiplies
column `j` of every projection that consumes it by `s_j`.** For `RMSNorm → Linear`
this is exactly function-preserving, which is why the model runs and why nothing
had to be "compensated".

### Why it matters, and why it is a finding rather than a footnote

This is **per-channel scale migration** — the same idea as AWQ's and
SmoothQuant's activation/weight scale absorption — used here to flatten the
per-channel dynamic range of the weight matrix *before* fitting a ternary grid.
A flatter matrix is dramatically easier for 9 reachable values to cover.

Three consequences:

1. **The paper does not describe it.** §3.1-3.2 and Algorithm 1 contain no
   channel-scaling step; §3.2 advertises the method as *"bias-free and
   mask-free"*. Anyone reimplementing PTQTP from the paper alone — which is what
   we did — produces a **different and weaker** algorithm.
2. **It is free at inference.** The norms already exist and are already
   multiplied in; only their stored values change. That is a strictly better
   deal than TWLA's KOTMS, which buys a similar kind of conditioning but leaves
   a permanent runtime rotation (see `ternary-feasibility.md`, T4).
3. **It must be added to Method C** before our numbers are read as "PTQTP".
   Until then, ours is *PTQTP-as-published*, and theirs is *PTQTP-as-implemented*.

### What the scale actually is, as far as it can be recovered

The scale is recoverable exactly: `s_j = ln_orig[j] / ln_artifact[j]`. What it is
*made of* is less clear, and the honest answer is that we could not derive it.

**It is genuinely per-channel, not a per-layer constant.** Fraction of the 1024
channels whose `s_j` sits within 1% / 5% of that layer's median:

| layer | norm | median s | within 1% | within 5% |
|---|---|---|---|---|
| 5 | post_attn | 2.931 | 27.8% | 77.1% |
| 13 | post_attn | 4.021 | 20.6% | 70.9% |
| 21 | post_attn | 1.964 | 13.3% | 63.5% |
| 13 | input | 0.719 | 17.4% | 62.7% |

So there is a large per-layer component with real per-channel variation on top.
(A per-layer *scalar* would have been a no-op for quality — a ternary fit is
scale-equivariant, since α absorbs any global factor. The per-channel part is
where the benefit must come from.)

**It is not a simple weight-column statistic.** Log-correlation of `s` against
the obvious candidates over the columns of `[gate; up]`, α ∈ {0.5, 1}:

| layer | col max\|W\| | col rms | col mean\|W\| |
|---|---|---|---|
| 5 | +0.464 | +0.516 | +0.424 |
| 13 | +0.199 | +0.492 | +0.470 |
| 21 | +0.043 | −0.096 | −0.147 |

Correlations that weak — and that change sign by layer — rule out a weight-only
rule of that family.

**Which leaves an uncomfortable possibility, stated as a hypothesis and not as a
finding.** The natural remaining candidate is an *activation*-derived scale, the
AWQ/SmoothQuant form `s_j ∝ (mean|x_j|)^α` over calibration data. If that is what
it is, it contradicts §4.1's *"No task-specific calibration, tuning, or
fine-tuning was applied in any experiment."* We cannot settle this: the code is
unreleased, and a scale can be inverted but not attributed. What can be said with
the evidence in hand:

1. the artifact carries a per-channel scale migration;
2. it is function-preserving and free at inference;
3. it is absent from the paper;
4. it is not derivable from the weights by any of the usual statistics.

Anything further needs their code.

### What this corrects

| earlier claim | status |
|---|---|
| "the released artifact may be broken" | **WRONG.** It is a function-preserving reparameterization, and the artifact scores 35.256 |
| "our implementation disagrees with the artifact by 19× on gate_proj" | **True but explained** — we do not do scale absorption; the comparison was never like-for-like |
| "`q_proj`/`k_proj` are protected" | **Still true for ternarization** — 127 distinct values per 128-group, they are not ternary — but they *are* modified, by the channel rescaling |
| "coverage is 80.0% of linears / 59.1% of the model" | **Unchanged** — ternarization coverage is what the bit budget depends on |

The methodological lesson is the one the repo already has a rule for: *when a
result is absurd, suspect the setup before the subject.* A 4× norm ratio in a
working model was never plausible as damage, and one `ppl` run said so.

---

## Cross-validation: our Algorithm 1 against the released artifact

Method C was implemented from the paper (`tools/qwen_ternary_feasibility.py`,
`ptqtp_fit`) rather than adapted from the artifact. First 64 rows of three real
Qwen3-0.6B tensors, G=128, on CPU:

| tensor | iters=1 | iters=5 | iters=50 | distinct/group | naive W1.58 |
|---|---|---|---|---|---|
| `gate_proj` L13 | 0.5151 | 0.2279 | **0.1834** (ratio 0.965) | 9 | 0.4559 |
| `v_proj` L0 | 0.5028 | 0.2190 | **0.1779** (ratio 0.968) | 9 | 0.4444 |
| `down_proj` L27 | 0.5548 | 0.2792 | **0.2106** (ratio 0.953) | 9 | 0.4993 |

Everything the paper claims about the algorithm reproduces: the error decreases
monotonically (Alg. 1's guarantee), it converges to **exactly 9 distinct values
per 128-group**, the norm ratio settles just below 1 as a least-squares fit
should, and it beats naive single-plane ternary by **2.4-2.5×**.

**Now compare against the released artifact, tensor for tensor:**

| tensor | ours, rel err | artifact, rel err | artifact norm ratio |
|---|---|---|---|
| `v_proj` L0 | 0.178 | **0.18** ✓ | 0.93 |
| `down_proj` L27 | 0.211 | 0.25-0.29 ≈ | 0.82-0.91 |
| **`gate_proj` L13** | **0.183** | **3.42** ✗ | **4.34** |

**Our implementation agrees with the artifact wherever the artifact does not
rescale, and disagrees by ~19× exactly where it does.** Read together with the
scale-absorption finding above, that says:

1. the algorithm as published works, and our implementation of Algorithm 1 is
   faithful to the paper;
2. the artifact is **not** defective — the gap is a channel-scale
   reparameterization the paper omits, applied before fitting;
3. `v_proj` is the control that proves both: 0.178 ours against 0.18 theirs, on a
   tensor whose input channels were barely rescaled.

Consequence: **our current Method C is PTQTP-as-published, not
PTQTP-as-implemented**, and it should be expected to score worse than 38.02
until channel-scale absorption is added. Both are worth measuring, and the gap
between them is precisely the value of the undocumented step.
