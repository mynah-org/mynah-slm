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

### An unexplained anomaly: `gate_proj` and `up_proj` do not reconstruct

Relative reconstruction error `‖Ŵ − W‖/‖W‖` and norm ratio `‖Ŵ‖/‖W‖` against the
original `Qwen/Qwen3-0.6B`, first 8 rows per tensor:

| block | v_proj | o_proj | **gate_proj** | **up_proj** | down_proj |
|---|---|---|---|---|---|
| | ratio / err | ratio / err | ratio / err | ratio / err | ratio / err |
| 0 | 0.93 / 0.18 | 0.98 / 0.19 | **1.77 / 0.81** | **1.81 / 0.87** | 0.82 / 0.25 |
| 5 | 0.80 / 0.26 | 0.98 / 0.17 | **3.21 / 2.27** | **3.27 / 2.34** | 0.75 / 0.31 |
| 13 | 0.70 / 0.34 | 0.99 / 0.17 | **4.34 / 3.42** | **3.56 / 2.65** | 0.82 / 0.29 |
| 21 | 0.70 / 0.33 | 0.98 / 0.17 | **2.34 / 1.45** | 1.32 / 0.45 | 0.90 / 0.26 |
| 27 | 0.99 / 0.17 | 0.98 / 0.17 | 0.82 / 0.27 | 1.10 / 0.54 | 0.91 / 0.57 |

`v_proj`, `o_proj` and `down_proj` look like sane ternary fits: norm ratio near 1,
relative error 0.17-0.57, cosine 0.86-0.99.

**`gate_proj` and `up_proj` do not.** Their norm is up to **4.34× too large** and
their relative error exceeds **1.0** in five of the nine blocks sampled — an error
larger than the tensor itself, i.e. *worse than replacing the tensor with zeros*.
The cosine is still 0.94, so the trit pattern is directionally right and only the
scale is wrong. In a SwiGLU MLP, `gate` and `up` are multiplied together, so a 4×
on each is 16× into `down_proj`.

A least-squares ridge fit (Eq. 1, Eq. 4) cannot produce a 4× overshoot; the
paper's Algorithm 1 line 2 initialises `α ← [1,1]` and iterates to *decrease*
`‖W − Ŵ‖²_F` monotonically. So this is a property of the upload, not of the
method as described.

**Provenance does not resolve it either.** The artifact's `q_proj` and
`input_layernorm` match **neither** `Qwen/Qwen3-0.6B` nor `Qwen/Qwen3-0.6B-Base`
— they sit ~5% away from both (cosine 1.000 against the instruct model, so the
direction is identical and only the magnitudes are perturbed) — while
`embed_tokens`, `model.norm` and some `k_proj` blocks **are bit-identical to the
instruct model**. That mixture is not explained by either base checkpoint.

**Conclusion for the reproduction gate: this artifact cannot be used to reproduce
Table 1's 38.02 until it is shown to run.** The decisive test is to load it and
measure perplexity, which is Gate A infrastructure we need anyway; it is running
(`tools/qwen_ternary_feasibility.py ppl`). Whatever it returns:

- if it lands near 38, the reconstruction analysis above is missing something and
  this note gets corrected;
- if it does not, the artifact is broken, and reproducing PTQTP means
  **implementing Algorithm 1 ourselves** — which §4.1 makes cheap, because the
  method needs **no calibration data at all**.

Either way the coverage finding (80% of linears, `q_proj`/`k_proj` protected) and
the storage finding (dense FP16, 4.25 bpw by the paper's own accounting) stand
independently of it: both are read off the file and the paper, not inferred from
behaviour.

## Calibration: there is none, and that simplifies Phase 1

**§4.1:** *"No task-specific calibration, tuning, or fine-tuning was applied in
any experiment."* PTQTP is a closed-form weight-only fit: Eq. 1 builds
`A_i = S_iᵀS_i + λ_i I₂` and `b_i = S_iᵀW_iᵀ` **from the weights alone**, and Eq. 5
searches the 9 ternary pairs against the weight, not against an activation.

**Consequence for R1: Method C needs no calibration corpus at all.** Phase 1's
calibration set is required only by the activation-aware methods (PT²-AGA, GPTQ
compensation). Method C can be implemented and evaluated before Phase 1 exists.
That reorders the plan in our favour.

## Published Qwen3-0.6B results, for the reproduction gate

- **Table 1** (WikiText2, group size 128): Qwen3-0.6B FP16 **20.9** → PTQTP
  **38.02**. Same row: 1.7B 16.70 → 32.46; 4B 13.64 → 18.25; 8B 9.71 → 11.8;
  32B 8.64 → 10.06.
- **Table 10** (MMLU, Qwen3 0.6B-32B): PTQTP-b1.58 **33.64** at 0.6B, then 43.82,
  63.65, 68.23, 76.20, 80.56.
- Baselines on Qwen3-0.6B for scale: AWQ-3bit 2.20E2, GPTQ-3bit 3.14E4,
  BiLLM 5.87E4, ARB-LLM_RC 8.43E2. **PTQTP at 38.02 is by far the best sub-4-bit
  result on this model** — the comparison it loses is against 4-bit, which the
  paper does not run at 0.6B.

## Answers, in one table

| # | question | answer |
|---|---|---|
| 1 | why "2 × 1.58"? | two full-size ternary planes per weight; log₂3 = 1.585 per trit (§3.1) |
| 2 | why 1.58 in Table 1? | a category label for the `# Bits` column, placing it beside 1.58-bit QAT and against 1.06-bit binary PTQ. Not a storage rate |
| 3 | physical representation? | **2 bits per trit, unpacked → 4.000 bits/weight** (App. A.3). Packing is listed under future work (App. G) |
| 4 | scale/metadata overhead? | two fp16 α vectors, G=128 → **+0.250 bits/weight**; total **4.250**. No bias, no mask (§3.2) |
| 5 | inference operations? | `y = α₁⊙(T₁x) + α₂⊙(T₂x)`, two sign-accumulate passes (App. A.1). Their own kernel is 1.41× slower than GPTQ-4bit at batch 1 (Table 5) |
| 6 | quantized vs protected? | paper says only "all linear layers"; resolved empirically against the released checkpoint |

## Next action

Fold the corrected bit accounting into `ternary-feasibility.md` (Phase 0 budget
and Phase 7a traffic get a second PTQTP row, "as implemented"), then run Gate A.

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

**Our implementation agrees with the artifact on the tensors that look sane and
disagrees by a factor of ~19 exactly where the artifact's norm ratio is 4.34.**
That is the cleanest available evidence that:

1. the method as published works and our implementation of it is faithful;
2. the released `Qwen3-0.6B-PTQTP-1.58b` upload is **defective on `gate_proj`
   and `up_proj`**, not representative of the paper's own experiments.

Consequence: **the reproduction gate runs against our implementation**, and the
artifact is used only as a spot check on `v_proj`/`o_proj`/`down_proj`. Whatever
perplexity the artifact returns is a fact about the upload, not about PTQTP.
