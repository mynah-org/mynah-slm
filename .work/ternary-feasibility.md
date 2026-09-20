# R1 — post-training ternarization of Qwen3-0.6B: is it worth a CPU backend?

Status: **IN PROGRESS** (opened 2026-09-20). Phase 0 done, Phases 1-7 planned.

Item: `PLAN.md` §0 R1. Tool: `tools/qwen_ternary_feasibility.py`.
Report: `docs/qwen3-0.6b-ternary-feasibility.md`.
Checkpoint: `Qwen/Qwen3-0.6B`, revision `c1899de289a04d12100db370d81485cdf75e47ca`,
staged at `models-local/qwen3-0.6b-bf16` (1433.7 MiB on disk). Apache 2.0.

**This is a feasibility study, not a kernel task.** No C, no SIMD, no new ggml
type. The deliverable is DATA about whether a ternary CPU backend would ever pay
for itself.

---

## Task

Determine whether a normal pretrained Qwen3-0.6B safetensors checkpoint can be
converted post-training into a ternary or multi-trit representation that retains
useful language-model quality, and whether the result justifies building a
CPU-native ternary backend in `src/`.

## Question

Stated the way this repo has to answer it, because "small is the point" is the
project's entire claim:

> **Does any ternary scheme beat the `Q4_K_M` file we ship today — or the `IQ2`
> and `TQ1_0` formats ingot already decodes — on the axis that matters, at a
> quality cost this model can absorb?**

Not "does ternarization work". It demonstrably works on 7B-class models. The
question is whether it works *here*, on a 0.6B model with a 151936-entry
vocabulary whose job is multilingual chat and tool calling.

---

## Known facts

### F1 — the parameter census, measured (Phase 0, `census.json`)

`python3 tools/qwen_ternary_feasibility.py census`, from the safetensors header
of the pinned revision. No weights loaded.

| family | tensors | shape | params | % of model |
|---|---|---|---|---|
| embedding | 1 | `[151936, 1024]` | 155,582,464 | **26.10%** |
| down_proj | 28 | `[1024, 3072]` | 88,080,384 | 14.78% |
| gate_proj | 28 | `[3072, 1024]` | 88,080,384 | 14.78% |
| up_proj | 28 | `[3072, 1024]` | 88,080,384 | 14.78% |
| o_proj | 28 | `[1024, 2048]` | 58,720,256 | 9.85% |
| q_proj | 28 | `[2048, 1024]` | 58,720,256 | 9.85% |
| k_proj | 28 | `[1024, 1024]` | 29,360,128 | 4.93% |
| v_proj | 28 | `[1024, 1024]` | 29,360,128 | 4.93% |
| norm | 113 | `[1024]`, `[128]` | 65,536 | 0.01% |

- distinct parameters **596,049,920**; transformer linears **440,401,920
  (73.89%)** — attention 29.55%, MLP 44.33%, i.e. **MLP is 60% of all
  ternarizable weight**;
- **the embedding is 26.10% of the model and is not a linear layer we can
  ternarize** without answering a separate question.

### F2 — the checkpoint stores the tied weight twice, and they are byte-identical

`config.json` says `tie_word_embeddings: true`, and the file nevertheless ships
both `lm_head.weight` and `model.embed_tokens.weight`: 311,164,928 bytes each,
**verified byte-identical by sha256**. 751,632,384 stored parameters against
596,049,920 distinct. The duplicate is 296.8 MiB of a 1433.7 MiB file.

Consequence for this study: **every "% of model" and every bits/weight figure
must be taken over the 596M distinct parameters.** Counting the stored 751M
would inflate the apparent compression of any scheme by 26% for free, and it is
exactly the kind of number that ends up in a table nobody re-derives. The tool
deduplicates and prints the check.

### F3 — the tied embedding is the LM head, so it is not a lookup table

Already established in `docs/models.md` and `PLAN.md`: `token_embd.weight` is
the input lookup **and** the output projection. It is simultaneously the largest
tensor, the hottest GEMV in the decode loop, and the thing directly in front of
the softmax. llama.cpp holds it at Q6_K while pushing everything else to Q4_K,
which is why the shipped file is 5.24 bits/weight and not 4.5.

**Every ternary paper in the brief quantizes "all linear layers" and leaves the
embedding alone.** For a 7B model that footnote costs a few percent. Here it
costs 26% of the model, and it is the single fact that decides this study.

### F4 — the storage budget, validated against a file we have weighed

`python3 tools/qwen_ternary_feasibility.py budget`. Scale overhead computed from
the real per-tensor row/column geometry, not assumed. The model reproduces the
measured `Q4_K_M` file to **0.01%** (372.7 MiB modelled vs 372.7 MiB measured,
5.245 vs 5.24 bits/weight) — the tool refuses to print the rest of the table if
that check fails.

| scheme | linear bpw | embedding | MiB | whole-model bpw |
|---|---|---|---|---|
| BF16 as published | 16.000 | bf16 | 1137.0 | 16.002 |
| Q8_0 everywhere | 8.500 | q8_0 | 604.1 | 8.503 |
| **Q4_K_M, exact recipe (shipped today)** | 4.500 | q6_k | **372.7** | **5.245** |
| Q3_K_M | 3.438 | q6_k | 302.4 | 4.256 |
| **IQ2_XXS linears** | 2.062 | q6_k | **230.2** | **3.240** |
| **IQ1_S linears** | 1.562 | q6_k | **204.0** | **2.871** |
| W1.58 row-wise + bf16 embedding | 1.688 | bf16 | 385.6 | 5.427 |
| W1.58 row-wise + q6_k embedding | 1.688 | q6_k | 210.6 | 2.963 |
| W1.58 G=128 (α,μ) + q6_k embedding | 1.938 | q6_k | 223.7 | 3.148 |
| **PTQTP 2×1.58 row-wise + bf16 embedding** | 3.375 | bf16 | **474.2** | **6.674** |
| PTQTP 2×1.58 row-wise + q6_k embedding | 3.375 | q6_k | 299.2 | 4.210 |
| PTQTP 2×1.58 G=128 + q6_k embedding | 3.625 | q6_k | 312.3 | 4.395 |

Ternary bits are the real `TQ1_0` packed rate (1.6875 bpw: five base-3 digits
per byte plus the block's own f16 scale, `third_party/ingot/docs/QUANTS.md`),
not the nominal 1.58.

### F5 — what the papers report for *this exact model*

PTQTP (arXiv 2509.16989) evaluates the Qwen3 family down to 0.6B. Its own
numbers, which are the strongest published evidence available and were produced
by the method's authors on their own implementation:

| model | WikiText2 ppl FP16 → PTQTP | MMLU FP16 → PTQTP |
|---|---|---|
| **Qwen3-0.6B** | **20.90 → 38.02  (+82%)** | **47.1% → 33.6%** |
| Qwen3-1.7B | 16.70 → 32.46  (+94%) | 60.0% → 43.8% |
| Qwen3-4B | 13.64 → 18.25  (+34%) | 69.7% → 63.7% |
| Qwen3-32B | 8.64 → 10.06  (+16%) | — |

**The degradation is monotone in model size and 0.6B is the worst case in the
paper.** That is the opposite of a favourable setting.

Independent corroboration on Qwen3-4B (arXiv 2609.01962, TWLA-derived: KOTMS
rotation + E2M-ATQ + GPTQ error compensation, W-only, A16): WikiText2
13.639 → 18.748, PTB 24.700 → 31.992, C4 19.831 → 28.966; ten scored
capabilities 64.5% → 54.7%, with ARC-Challenge collapsing to 43.8%. Effective
**1.641 bits/weight over 81.62% of parameters**, and the packed model goes
8.29 GiB → 3.96 GiB — a **2.09×** whole-file compression, not the 9.7× the
"1.58-bit" label suggests. Two unrelated methods landing on 18.25 and 18.748 for
Qwen3-4B is convergent evidence, not coincidence.

That study also states plainly: *"we therefore do not claim that compression
alone yields faster inference"* — their Triton ternary GEMV measured **4.6×
slower than FP16 cuBLAS**.

PT²-LLM (arXiv 2510.03267) is single-plane W1.58 with ITF + AGA + SSR, 128
calibration samples of length 2048 from WikiText2, block size 128, row-wise
(α, μ). Published models are LLaMA-7B/13B/65B, LLaMA-2, LLaMA-3-8B and
Qwen3-14B-Base — **nothing at 0.6B scale**, and its LLaMA-7B result (5.68 →
11.39) is a 2× perplexity cost on a model twelve times larger than ours.

### F6 — the container question is already answered, for one scheme only

ingot decodes `TQ1_0` (id 34, 1.69 bpw) and `TQ2_0` (id 35, 2.06 bpw) today,
generic kernel, no SIMD. So:

- **single-plane W1.58 has a container already** — `TQ1_0` is an exact fit, and
  a converter could emit a normal GGUF that our engine opens unchanged;
- **PTQTP's dual plane does not.** Two trit-planes plus two scale sets is not
  any ggml type. It would need either a new type upstream in ingot or a
  side-channel, and rule 4 says a container change is ingot's business while a
  kernel is ours. That is a real cost on the PTQTP side of the ledger that the
  paper does not carry.

### F7 — this repo's own sub-4-bit bar, already measured

`Q2_K` on granite-350m: perplexity **26400**, bits/byte 3.38 against Q8_0's
1.59, word-salad output — recorded as DESTROYED, not a candidate. `Q3_K_M`
survives but costs 8% of bits/byte to save 27 MB and was shelved as a poor
trade. Qwen3-0.6B at Q4_K_M scores **25/30** on the tool-call eval and needs
`--think on` to get there (21/30 without).

A model at MMLU 33.6 does not call tools. The tool-call eval, not perplexity, is
the gate this project actually ships against.

---

## The cost model, written before any code

Per `.work/engineering-method.md` §3: if the maximum plausible saving is too
small for the product goal, do not optimise.

| field | |
|---|---|
| current cost | 372.7 MiB, 5.245 bits/weight, 36.5 decode tok/s (M1, local, Q4_K_M) |
| suspected cause | the 26.1% embedding sets a floor no linear-weight scheme can cross |
| proposed transformation | ternarize the 73.89% that is linear |
| **maximum plausible saving, storage** | see below — **it is negative against IQ1_S** |
| **maximum plausible saving, bandwidth** | 372.7 → 210.6 MiB = 1.77× fewer weight bytes per token, but only if the embedding also drops to Q6_K, which is orthogonal to ternary |
| new work introduced | a PTQ pipeline, a converter, a new ggml type for dual-plane, a NEON+AVX2 kernel family, a quality gate per language |
| risk | published evidence says this exact model loses 82% perplexity and 13 MMLU points |
| **smallest experiment capable of killing it** | Phase 3a below — one tensor family, naive W1.58, layer-output error |

### The saving is negative against formats we already read

Comparing like with like (all rows keep the embedding at Q6_K, because that
decision is independent of ternary):

| | MiB | vs Q4_K_M | vs best ternary |
|---|---|---|---|
| Q4_K_M (shipped) | 372.7 | — | |
| **PTQTP 2×1.58** | 299.2 | 1.25× smaller | |
| W1.58 single plane | 210.6 | 1.77× smaller | — |
| **IQ2_XXS — ingot reads it today** | 230.2 | 1.62× smaller | ternary wins by **9%** |
| **IQ1_S — ingot reads it today** | 204.0 | 1.83× smaller | ternary **loses by 3%** |

So the mandatory method, PTQTP, is **30% larger than IQ2_XXS and 47% larger than
IQ1_S**, both of which need zero research and zero new kernels. And the best
ternary scheme ties with `IQ1_S`.

**On bytes, ternary buys nothing here.** Whatever case exists must rest on
arithmetic — a multiplication-free GEMV — which is a kernel question, explicitly
out of scope for this task, and which the one published attempt measured at 4.6×
*slower* than the dense baseline.

### Why the arithmetic case is not yet dead

The bandwidth-bound argument cuts both ways and we have not measured our side of
it. Qwen3-0.6B Q4_K_M decodes at 36.5 tok/s on this M1 — 372.7 MiB per token is
**~13.6 GB/s against the M1's ~68 GB/s**, i.e. **20% of peak**. Decode is *not*
saturating memory bandwidth here; a meaningful share of the step is dequant and
arithmetic. That is the one honest opening for a multiplication-free kernel, and
it is measurable without implementing one (Phase 7).

It also matches the sibling-repo finding in `engineering-method.md` §8: the
smaller the model, the more of the time is preparation rather than arithmetic.

---

## Unknowns

1. **U1** Does our reproduction of PTQTP land near the paper's 38.02 on
   Qwen3-0.6B? If not, is the gap our implementation or their unstated
   calibration? (The paper says "no task-specific calibration" but PTQTP is a
   weight-only closed-form fit, so there may genuinely be none.)
2. **U2** Is 0.6B the variable, or is *any* sub-2-bit format destroyed at this
   size? **This is the control that would make the preferred story look stupid.**
   If `IQ2_XXS` and `IQ1_S` are also word salad here, the finding is about the
   model, not about ternary, and question 12 (repeat at 4B/7B) becomes the real
   question.
3. **U3** Which projection families and blocks are sensitive. The brief lists
   plausible patterns and explicitly says not to assume them.
4. **U4** How much activation-aware reconstruction (PT²/TWLA) buys over naive
   weight-MSE ternarization at *this* size, where calibration statistics are
   noisier relative to the weight count.
5. **U5** What the embedding can tolerate. Everything above holds it at Q6_K by
   assumption; the whole-model budget moves more when that assumption moves than
   when the linear scheme does.
6. **U6** Whether the decode step is dequant-bound or arithmetic-bound at 20% of
   peak bandwidth — the only number that could rescue the arithmetic case.

## Files inspected

- `models-local/qwen3-0.6b-bf16/{config.json,model.safetensors}` (header + the
  two embedding blobs only)
- `reference/qwen3-0.6b/safetensors_header.json` — **superseded**: it is the
  751M-parameter stored view and double-counts the tied embedding
- `docs/models.md` §Qwen3-0.6B — the measured quant ladder and tensor census
- `third_party/ingot/docs/QUANTS.md` — block geometry for every candidate type
- arXiv 2509.16989, 2510.03267, 2606.13054, 2609.01962

---

## Plan

Each phase has a gate. A phase that does not pass its gate stops the study and
the verdict is written from what is known — a NO-GO is a successful result here.

### Phase 0 — census and budget — **DONE 2026-09-20**

`census` and `budget` subcommands. Gate: the budget model reproduces the
measured `Q4_K_M` file within 1%. **Passed at 0.01%.**

### Phase 1 — calibration corpus

128 sequences × 2048 tokens from WikiText-2-raw-v1 train, seed 0 — the
configuration PT²-LLM and the GPTQ lineage both use, so our numbers are
comparable to theirs. Held-out: WikiText-2 test, plus a small multilingual
held-out set because a WikiText-only verdict would not cover what this engine
ships for.

Cache tokenized calibration *and* per-layer inputs to `reports/ternary/cache/`.
Gate: BF16 baseline perplexity reproduces the published FP16 figure for this
checkpoint within a few percent — **if we cannot reproduce 20.9 we cannot
interpret 38.0.**

### Phase 2 — methods

- **A, control**: naive single-plane `W → αT`, threshold/scale search, per-tensor
  and per-row/group variants. Weight MSE only.
- **B**: PT²/TWLA-style single-plane — ITF then AGA against cached layer inputs,
  minimising `‖WX − ŴX‖²_F` rather than weight error, block 128, row-wise
  (α, μ). Adapted, not reproduced: every deviation from the authors' code gets a
  line in the report. SSR reordering and KOTMS rotation are separate flags so
  their contribution is separable.
- **C, mandatory**: PTQTP dual plane. `W ≈ diag(α₁)T₁ + diag(α₂)T₂`, initialise
  `T = sign(W)`, alternate a closed-form 2×2 ridge solve for (α₁, α₂) with an
  element-wise exhaustive search over the 9 ternary pairs, adaptive λ on the
  condition number, stop at `max‖Δα‖_F < 1e-4` or 50 iterations.

Compute estimate, so this is not discovered halfway: the PTQTP inner loop is
9 candidate pairs × 440M weights per iteration. At ~10 effective iterations that
is ~40 G element-operations — fine on MPS in torch, hours in numpy. Budget it as
a torch-MPS job, single process, and cache per-tensor results so a rerun of one
family does not redo the model.

Gate: each method reproduces its own paper's number on *some* published
configuration before we trust it on ours. For C that is the Qwen3-0.6B row
itself (38.02); for B, PT²-LLM publishes nothing at this size, so the gate is
LLaMA-7B if it is affordable, otherwise B is explicitly labelled **adapted,
unvalidated**.

### Phase 3 — layer sensitivity

Quantize one tensor at a time, everything else BF16. Per tensor record: weight
NMSE, cosine, relative Frobenius, {-1,0,+1} occupancy, scale distribution,
outlier stats; and on cached calibration activations the layer-output NMSE,
cosine, relative and max error. Then block × family heatmaps.

**Phase 3a is the kill switch and runs first**: naive W1.58 on *all*
`down_proj`, and separately on all `gate_proj`+`up_proj`, nothing else, then
measure perplexity. 60% of ternarizable weight is MLP; if the MLP alone cannot
survive the control method, no amount of method sophistication changes the
verdict at this size. Two runs, no per-tensor sweep needed to learn it.

Gate to continue to Phase 4: at least one method × family combination keeps
held-out perplexity within **2×** of the BF16 baseline. Below that the model is
not a language model any more and the ranking is noise.

### Phase 4 — progressive coverage

Rank by Phase 3 sensitivity, then sweep 10/25/50/70/80/90/100% of eligible
linear weights independently for A, B and C — three degradation curves, not one
checkpoint. Plus the logical groupings: attention only, MLP only, both, all
transformer linears, all eligible tensors.

### Phase 5 — mixed precision

Search a practical layout from the Phase 4 ranking. Report effective bits **both
ways**: over quantized linears, and over the complete model including scales,
metadata and every unquantized tensor. The second number is the only one that
describes a file.

### Phase 6 — the conventional controls (**do not skip, this is U2**)

BF16, INT8, Q4_K_M, Q3_K_M, **IQ2_XXS and IQ1_S** on the identical eval, using
llama.cpp to produce the GGUFs and our own `mynah-slm` + `tools/eval/` to score
them — same prompts, same seed, same threads. Compare model bytes, perplexity
delta, layer reconstruction and theoretical weight traffic per token.

Do **not** benchmark a fake dequantized PyTorch ternary path and conclude ternary
is slow: that measures the wrong runtime. This phase is about quality per byte.

### Phase 7 — CPU-backend feasibility, on paper only

Gated on a GO from Phases 4-6. Matrix-shape census for autoregressive decode,
weight traffic per token per layout, and the roofline question U6: at 20% of
peak bandwidth, how much of a decode step is dequant?

Then reason — without implementing — about whether
`y = α₁(T₁x) + α₂(T₂x)` maps to AVX2 / AVX-512BW / VNNI / AMX and to
NEON / DOTPROD / I8MM / SVE2, and whether the planes should be packed
separately, interleaved, unpacked to INT8 tiles, decoded into cache, or consumed
by LUT/bitwise kernels. Separate storage bandwidth from arithmetic cost. Note
that `TQ1_0`'s base-3 packing is decode-unfriendly (a multiply-high and a
fixed-point divide per five weights) — a ternary format optimised for a CPU
kernel is not necessarily the one optimised for bytes, and that tension belongs
in the report.

---

## Pre-registered prediction

Written before Phase 1 so it can be wrong in public:

> Qwen3-0.6B will **not** tolerate post-training ternarization at useful
> coverage. Single-plane W1.58 will be unusable past roughly 40-50% coverage;
> PTQTP will land near the paper's +82% perplexity and will fail the tool-call
> gate; and even where quality holds, the whole-model storage win against
> `IQ2_XXS`/`IQ1_S` — which ingot already decodes — is between −3% and +30%,
> i.e. **not a reason to write a kernel**.
>
> The interesting result will be the *scaling*: ternary's cost falls monotonically
> with model size (82% → 94% → 34% → 16% across 0.6B/1.7B/4B/32B), and the
> embedding's share of the model falls with it (26.1% at 0.6B against ~9.7% at
> 4B). The question worth keeping open is **Qwen3-4B**, not Qwen3-0.6B.

If that prediction survives Phases 3-6, the verdict is **REJECT for v0.1/v0.2**
with a documented re-open condition, and the study has done its job.

## Acceptance gate for the item as a whole

R1 is DONE when `docs/qwen3-0.6b-ternary-feasibility.md` answers all twelve
questions in the brief with measured numbers or an explicit "not measured, and
here is why", and carries a completion block:

`WHAT CHANGED · WHAT PATH ACTUALLY RAN · WHAT WAS MEASURED · WHAT REMAINS
UNKNOWN · VERDICT: PROMOTE / KEEP / INCONCLUSIVE / REJECT`.

No C is written under this item under any verdict. A GO produces a *new* item
with its own note and its own gates.

---

## Evidence log

Append-only. FACT · HYPOTHESIS · TEST · RESULT · DECISION · CLAIM SCOPE.

### 2026-09-20 — Phase 0

- **FACT** Checkpoint `Qwen/Qwen3-0.6B` @ `c1899de2` staged locally, 1433.7 MiB.
  311 tensors, 751,632,384 stored / 596,049,920 distinct parameters.
- **FACT** `lm_head.weight` and `model.embed_tokens.weight` are byte-identical
  (sha256). The published file wastes 296.8 MiB on the duplicate.
- **FACT** Linears 73.89% of distinct parameters (MLP 44.33%, attention 29.55%);
  embedding 26.10%; norms 0.01%.
- **TEST** Budget model against the one file we have weighed.
- **RESULT** Modelled `Q4_K_M` 372.7 MiB / 5.245 bpw vs measured 372.7 MiB /
  5.24 bpw — 0.01%. The tool refuses to print the table if this exceeds 1%.
- **RESULT** PTQTP 2×1.58 with a BF16 embedding is **474.2 MiB, 27% larger than
  the file we ship today**. With a Q6_K embedding, 299.2 MiB — still 30% larger
  than `IQ2_XXS` and 47% larger than `IQ1_S`, both already decodable by ingot.
- **DECISION** Phase 6 is promoted from "if easy using existing libraries" to
  **mandatory and early**: `IQ2_XXS`/`IQ1_S` are the real controls, not Q4.
- **DECISION** Phase 3a (MLP-only naive W1.58, two runs) runs before the
  per-tensor sweep, as the cheapest experiment that can end the study.
- **CLAIM SCOPE** Storage only. Nothing here is a statement about quality or
  about decode speed; both are unmeasured.

### Rejected before being tried

- **Quoting "1.58 bits" or "3.16 bits" as the model's bit rate.** The packed
  `TQ1_0` rate is 1.6875 and the whole-model rate with a Q6_K embedding is
  2.963. The 9.7× compression the label implies does not exist at this size; the
  Qwen3-4B study measured 2.09× on a model where the embedding is a *smaller*
  share than ours.
- **Benchmarking a dequantized PyTorch ternary path.** It measures the wrong
  runtime. Explicitly forbidden by the brief and by
  `.work/engineering-method.md` ("never benchmark a contradiction").
- **Using `reference/qwen3-0.6b/safetensors_header.json` for the census.** It is
  the stored view and double-counts the tied embedding: every percentage taken
  from it is wrong by 26%.
