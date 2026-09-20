# R1 — post-training ternarization of Qwen3-0.6B: is it worth a CPU backend?

Status: **IN PROGRESS** (opened 2026-09-20).

**Two gates, and they are independent. Do not let one answer the other.**

| gate | question | status |
|---|---|---|
| **A — MODEL** | does Qwen3-0.6B retain useful quality under PTQTP / ternary PTQ? | **UNMEASURED** — Phases 1-6 not run. This is the blocking question |
| **B — CPU BACKEND** | can a target ISA execute the representation efficiently? | **UNDECIDED** — awaiting Gate A *and* target-ISA evidence |

**Correction, 2026-09-20.** Addendum 2 and Phase 7b below issued a
`VERDICT: REJECT` on Gate B from Mac Q4/Q8 runtime measurements. **That verdict
is withdrawn.** The measurements stand and are retained — thread scaling, Q4 vs
Q8, the activation path, the LM-head cost and the decode byte census are all
real and all useful — but they describe *the current Q4 implementation on one
M1*. They cannot close a backend question for Neoverse V2 I8MM/SMMLA, AVX-512
VNNI, AVX-512 BW or AMX, and no ternary kernel has been benchmarked on any
machine. Gate B stays open until Gate A is measured and at least one
representative target-server microbenchmark or analytical kernel study exists.

Order of work is now fixed: **read PTQTP → inspect/reproduce the authors'
Qwen3-0.6B artifact → run the quality and sensitivity phases.** No further
effort goes into arguing the backend from the present Mac Q4 runtime.

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

> **CORRECTION 2026-09-20 (R1-A claim audit).** This paragraph read *"the
> degradation is monotone in model size and 0.6B is the worst case in the
> paper."* **Both halves are false on the table directly above them.** The
> ratios are 0.6B **1.819**, 1.7B **1.944**, 4B 1.338, 8B 1.215, 32B 1.164:
> not monotone, and **1.7B is the worst case, not 0.6B**. The narrative was
> written over arithmetic that contradicts it. What the table supports is only
> that the two smallest models degrade far more than the three largest, and
> even that is `[PAPER]`, not ours. Phase F measures it on our harness.

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

---

# Addendum, 2026-09-20 — what the PocketTTS study transfers, and what does not

`mynah-tts` ran the same question against PocketTTS and closed it
(`../mynah-tts/docs/ternary-feasibility.md`, `PLAN.md` E11, commit `1de7cad`).
Its verdict was **no-go on the backbone, a conditional small yes on the codec,
and int8 is the actual win**. Read for evidence, not for its conclusion: a TTS
model with a non-autoregressive codec beside an AR backbone has different
economics from a dense decoder-only LM, and the whole point of this section is
to find out where they differ.

## Transfers — these are CPU and algebra facts, not PocketTTS facts

**T1. No target ISA has a sub-byte multiply-accumulate.** NEON `SDOT`/`SMMLA`,
AVX-512 VNNI `VPDPBUSD` and AMX are all int8-lane. Our own `src/qmat.c` already
shows the shape: the Q4 path widens nibbles to int8 *before* the vector dot.
A ternary weight would do exactly the same. **Ternary's MACs per instruction are
identical to int8's, so its entire CPU case is weight traffic** — there is no
arithmetic prize at the end of this, only a bandwidth one. This reframes the
whole study and it is not in dispute.

**T2. PT²'s AGA and TWLA's E2M-ATQ stage 2 are the same estimator.** Both freeze
`T` and solve the identical per-row 2×2 system under the calibration metric
`S = XᵀX`; the papers differ in notation and in how they reach `T`, not in the
relocation. `mynah-tts` implemented **both** and measured them agreeing to
**within 0.3% on every layer group**. That answers the brief's question 7 with
someone else's compute: **implement AGA only**, and keep the warm-start
difference (E2M's μ-initialisation and residual-mean correction) as a flag on the
same code path, so what is measured is the warm start and nothing else.

**T3. GPTQ-style error compensation dominates everything else.** On PocketTTS:
activation-aware grid fitting bought 1.1-8.9×; GPTQ-style compensation bought
**1.3-108×, median ≈13×**. Their sentence: *"ternary without error compensation
is not competitive with anything."* Consequence here: **naive W1.58 is a control,
not a candidate**, and GPTQ compensation is promoted from an implementation
detail of Method B to a first-class experimental condition that every method is
run with and without.

**T4. KOTMS is not worth the runtime it costs.** It beat AGA on all eight layers
measured (1.2-3.7×) and still lost to GPTQ on every one (2-6×) — and it is the
only method that leaves a permanent inference cost, because `R = R₁ ⊗ R₂` must be
applied to the activation on every call forever, while GPTQ is a build-time
transform that costs inference nothing. In a model where activation preparation
already dominates at small sizes (`engineering-method.md` §8), a persistent
activation rotation is the worst possible place to spend. **Demoted to optional;
if run at all, its quality benefit is reported separately from its inference
cost.**

**T5. Sub-byte unpacking has already lost twice in this family.** `mynah-tts`
E10's rejected list: batched int4 GEMM measured **0.80-0.97×** on three x86
boxes, and int8 ConvTranspose was slower than f32 sgemm. Whatever the traffic
model says, the unpacking is not free.

**T6. The nominal bit-width never appears in the honest total.** PocketTTS's
"1.58-bit" codec came out at **2.05 bits/weight** once the plane, the per-row
scale and the per-row shift were counted. Our equivalent is 1.6875 packed
(`TQ1_0` carries its own block scale) and **2.963 bits/weight over the complete
model** — see F4.

## Does NOT transfer — and this is the genuine difference

**D1. PocketTTS failed on a MAC/byte mismatch that Qwen3 structurally cannot
have.** There, the Mimi decoder held **75.4% of the MACs but 11.0% of the ternary
bytes**, while the backbone held 22.0% of MACs and **79.4% of the bytes** — and
the backbone was the part that failed quality. Ternary's only lever is bytes, the
bytes were all in the region that failed, and the region that passed was
cache-resident and already int8. That is why it was a no-go there.

**In a dense decoder at batch 1, every weight is read exactly once per token and
feeds exactly one MAC.** `%MAC` and `%weights` are therefore the same number for
every region — confirmed in the shape census below, where the two columns track
each other exactly. **There is no cache-resident region doing the arithmetic off
few bytes, and no byte-heavy region sitting outside the hot path.**

So the brief's question 3 gets a clean answer, and it is *not* the PocketTTS
answer: **yes, unlike Pocket, ternary here attacks the region that actually owns
the memory traffic.** Everything the method touches is in the AR loop.

**D2. The failure mode to watch for is different.** PocketTTS died of AR/EOS
instability that compounded with utterance length — 3/7 long utterances failed to
terminate at temperature 0. Qwen3's published failure is capability loss
(+82% perplexity, MMLU 47.1 → 33.6), not non-termination. Both are AR feedback;
do not assume ours shows up as the same symptom. **The analogue of Pocket's
"duration ratio" gate is our tool-call eval**, which is a behaviour gate and not
a perplexity delta, and it is the one that decides.

**D3. The premise "weight bandwidth dominates decode" is established for Pocket
and NOT established here.** `../mynah-tts/.work/backbone-bandwidth.md` measured
the PocketTTS backbone at 151 MB per AR step, **32-38 GB/s, and eight cores
buying only 1.15×** — a textbook memory wall on the same class of machine.

Qwen3-0.6B at Q4_K_M moves **372.7 MiB per token at a measured 36.5 tok/s =
13.6 GB/s**, which is **less than half the bandwidth that same Mac was shown to
sustain**. Either our decode is bound by something other than weight bytes, or
the comparison is unsound. **This is now the pivotal unknown (U6) and it is
measurable without writing a kernel** — Phase 7b below.

---

## Phase 7a — decode traffic and GEMV shapes  `[MEASURED]`

`python3 tools/qwen_ternary_feasibility.py traffic`. Arithmetic from the census;
KV cache in bf16 per the repo's quantization policy.

### Weight bytes per decoded token

Because every weight is read once, **weight bytes per token is the model file.**

| scheme | MiB/token | q/k/v/o + gate/up/down | tied lm_head | vs Q4_K_M | AI (FLOP/B) |
|---|---|---|---|---|---|
| BF16 | 1137.0 | 73.9% | 26.1% | 0.33× | 1.00 |
| INT8 / Q8_0 | 604.1 | 73.9% | 26.1% | 0.62× | 1.88 |
| **Q4_K_M — shipped** | **372.7** | **67.3%** | **32.7%** | **1.00×** | **3.05** |
| Q3_K_M | 302.4 | 59.7% | 40.2% | 1.23× | 3.76 |
| **PTQTP 2×1.58** | 299.2 | 59.2% | 40.7% | **1.25×** | 3.80 |
| IQ2_XXS | 230.2 | 47.0% | 52.9% | 1.62× | 4.94 |
| **W1.58 single plane** | 210.6 | 42.1% | 57.8% | **1.77×** | 5.40 |
| IQ1_S | 204.0 | 40.2% | 59.7% | 1.83× | 5.57 |
| *linears at zero bits (bound)* | *122.0* | *0%* | *99.8%* | *3.06×* | *9.32* |

**The projections' share of traffic falls as you ternarize them.** They are 67.3%
of the bytes at Q4 and 42.1% at W1.58, because the tied `lm_head` — which no
method in the brief quantizes — does not move. That is Amdahl's law on bytes, and
it caps the whole exercise at **3.06×** even with free linear weights.

### The KV cache eats most of what is left

`2 × n_kv_heads(8) × head_dim(128) × n_layers(28) × 2 B` = **114,688 bytes
(112 KiB) per context position**, read in full every decode step, and **no
weight-side scheme touches it.**

End-to-end decode traffic speedup against the shipped Q4_K_M:

| scheme | L=128 | L=512 | L=1024 | L=2048 | L=4096 |
|---|---|---|---|---|---|
| INT8 / Q8_0 | 0.63× | 0.65× | 0.68× | 0.72× | 0.78× |
| Q3_K_M | 1.22× | 1.20× | 1.17× | 1.13× | 1.09× |
| **PTQTP 2×1.58** | **1.23×** | 1.21× | 1.18× | 1.14× | **1.10×** |
| IQ2_XXS | 1.58× | 1.50× | 1.42× | 1.31× | 1.21× |
| **W1.58 single plane** | **1.72×** | 1.61× | 1.50× | 1.37× | **1.25×** |
| **IQ1_S** | **1.77×** | 1.65× | 1.53× | 1.39× | **1.26×** |
| *linears at zero bits* | *2.84×* | *2.41×* | *2.07×* | *1.72×* | *1.44×* |

**PTQTP — the method the brief makes mandatory — buys between 10% and 23% of
decode traffic against the file we ship today**, and it costs +82% perplexity on
this model according to its own authors. `IQ1_S`, which ingot already decodes,
beats every ternary scheme at every context length.

### Dominant GEMV shapes at batch 1 (M=1)

| N × K | instances | params | % MACs | % bytes @Q4 | % bytes @W1.58 | family |
|---|---|---|---|---|---|---|
| **3072 × 1024** | 56 | 176,160,768 | **29.56** | 26.91 | 16.83 | gate_proj, up_proj |
| **151936 × 1024** | 1 | 155,582,464 | **26.11** | **32.66** | **57.81** | tied lm_head |
| 1024 × 3072 | 28 | 88,080,384 | 14.78 | 13.45 | 8.42 | down_proj |
| 1024 × 1024 | 56 | 58,720,256 | 9.85 | 8.97 | 5.61 | k_proj, v_proj |
| 1024 × 2048 | 28 | 58,720,256 | 9.85 | 8.97 | 5.61 | o_proj |
| 2048 × 1024 | 28 | 58,720,256 | 9.85 | 8.97 | 5.61 | q_proj |

Two observations a kernel project would need:

- **There are only five distinct linear shapes, all with K ∈ {1024, 2048, 3072}
  and all a multiple of 256.** That is unusually friendly: a ternary GEMV kernel
  would need five specialisations plus the head, not a general GEMM. `% MACs` and
  `% bytes` track each other, which is D1 restated as a table.
- **The tied `lm_head` becomes the single dominant tensor the moment the linears
  shrink**: 32.7% of bytes at Q4, **57.8% at W1.58**. Any ternary backend that
  did not also solve the 151936 × 1024 head would spend most of its traffic
  budget on the one matrix it cannot touch.

## Phase 7b — is decode actually traffic-bound here?  `[PLANNED]`

The cheapest experiment that decides whether any of the above converts into time.
Method copied from `../mynah-tts/.work/backbone-bandwidth.md`, because the
comparison is only meaningful if the measurement is the same one:

- `mynah-slm` on `Qwen3-0.6B-Q4_K_M`, **staged local**, five runs per thread
  count at 1 / 2 / 4 / 8 threads, median decode tok/s, identical prompt, seed,
  `--think off`, fixed generated-token count, quiet machine, one process at a
  time.
- Derive achieved GB/s = decode tok/s × 372.7 MiB, and compare against the
  32-38 GB/s the sibling measurement demonstrated on this same machine.

Reading, decided **before** the run so the result cannot be rationalised after:

| observation | conclusion |
|---|---|
| tok/s flat from 2 threads up, ≥30 GB/s | traffic-bound. A 1.77× traffic cut is worth up to 1.77×, and the ternary case is live |
| tok/s scales to 8 threads, ≪30 GB/s | **not** traffic-bound. Cutting weight bytes cannot pay, and the bottleneck must be found before any format work |
| in between | the honest answer is a roofline with both terms, and the study reports the fraction attributable to traffic rather than a speedup |

Note that AI **rises** from 3.05 to 5.40 FLOP/byte going from Q4 to W1.58: if the
machine balance sits in that window, ternarizing moves the step from
traffic-bound to compute-bound and the saved bytes stop turning into time
somewhere inside the transition. A traffic model alone cannot see that.

## The brief's sharpened questions, as far as Phase 0/7a can answer them

| # | question | answer |
|---|---|---|
| 1 | % of decode weight bytes in q/k/v/o + gate/up/down | **67.3% at Q4_K_M**, falling to 59.2% (PTQTP) and 42.1% (W1.58) as they shrink `[MEASURED]` |
| 2 | are the largest matrices also the dominant traffic? | **yes, necessarily** — at batch 1 bytes ∝ params. Largest single tensor is the tied head (32.7% of Q4 bytes) `[MEASURED]` |
| 3 | unlike Pocket, does ternary attack the real bottleneck? | **it attacks the right *region*** (no MAC/byte mismatch exists here) — but whether that region is the *bottleneck* is unproven: 13.6 GB/s against a demonstrated 32-38 `[MEASURED / OPEN]` |
| 4 | bytes/token for BF16 / INT8 / Q4 / W1.58 / PTQTP | 1137.0 / 604.1 / 372.7 / 210.6 / 299.2 MiB `[MEASURED]` |
| 5 | prefill vs decode | decode AI 3.05 (Q4); prefill over B tokens amortises weights, AI ≈ 3.05·B, so at B=32 it is ~97 FLOP/byte — **prefill is compute-bound and the weight format is nearly irrelevant to it.** The case is decode-only `[MEASURED]` |
| 6 | test GPTQ compensation explicitly | **adopted** — promoted to a first-class condition on the strength of T3 |
| 7 | are PT²-AGA and TWLA E2M equivalent? | **yes, on the part that matters** — same per-row 2×2 system under `S = XᵀX`; sibling measured ≤0.3% difference. Implement AGA only `[PUBLISHED + SIBLING-MEASURED]` |
| 8 | KOTMS skepticism | **justified** — loses to GPTQ 2-6× and is the only method with a permanent runtime activation rotation. Optional, cost reported separately |
| 9 | reproduce PTQTP's Qwen3-small result first | unchanged, it is the Phase 2 gate: land near **38.02** on Qwen3-0.6B before inventing anything |
| 10 | quality × bytes, not quantization error | table below; the bytes column is measured, the quality column is **published-only** until Phase 3 |

### Quality × bytes, with what is known today

`[MEASURED]` bytes, `[PUBLISHED]` quality (PTQTP's own Qwen3-0.6B row), `[NOT
MEASURED]` everything else. This is the table the final report must fill in
completely.

| configuration | effective model bits | MiB/token decode | WikiText2 ppl | relative quality loss |
|---|---|---|---|---|
| BF16 | 16.00 | 1137.0 | 20.90 `[PUBLISHED]` | baseline |
| INT8 / Q8_0 | 8.50 | 604.1 | `[NOT MEASURED]` | expected ≈ baseline |
| Q4_K_M (shipped) | 5.24 | 372.7 | `[NOT MEASURED]` | ships today; 25/30 tool calls |
| PTQTP 2×1.58 + Q6_K head | 4.21 | 299.2 | **38.02 `[PUBLISHED]`** | **+82%; MMLU 47.1 → 33.6** |
| IQ2_XXS + Q6_K head | 3.24 | 230.2 | `[NOT MEASURED]` | **the control that decides U2** |
| W1.58 + Q6_K head | 2.96 | 210.6 | `[NOT MEASURED]` | expected worse than PTQTP |
| IQ1_S + Q6_K head | 2.87 | 204.0 | `[NOT MEASURED]` | **the control that decides U2** |

## Changes to the plan, from this addendum

1. **Method B collapses.** PT²-AGA and TWLA-E2M are one estimator (T2). Implement
   AGA; E2M's warm start becomes a flag. Saves roughly half of Phase 2.
2. **GPTQ compensation becomes a first-class axis** (T3), run against every
   method rather than inside one. Naive W1.58 is demoted to a control.
3. **KOTMS is optional and reported with its runtime cost attached** (T4).
4. **Phase 7b is promoted ahead of Phases 3-5.** D3 says the premise of the whole
   exercise — that weight bandwidth dominates batch-1 decode — is unproven on
   this model, and it costs one afternoon of benchmarking to settle. If decode is
   not traffic-bound, no quality result can rescue the kernel case, and the
   remaining phases become a much smaller quality study rather than a backend
   investigation.
5. **The tied `lm_head` gets its own experiment.** At W1.58 it is 57.8% of decode
   traffic. Any GO verdict is conditional on a scheme for the 151936 × 1024 head,
   and no paper in the brief supplies one.

## Evidence log — 2026-09-20, addendum

- **FACT (T1)** No target ISA has a sub-byte MAC; ternary's MACs/instruction
  equal int8's. Ternary's only lever on CPU is weight traffic.
- **FACT (T2)** PT²-AGA ≡ TWLA-E2M stage 2, sibling-measured ≤0.3% apart on every
  layer group. **DECISION** implement one.
- **FACT (T3)** GPTQ compensation 1.3-108× (median ≈13×) vs AGA's 1.1-8.9× on
  PocketTTS. **DECISION** first-class condition; naive is a control.
- **FACT (T4)** KOTMS loses to GPTQ 2-6× and carries a permanent runtime
  activation rotation. **DECISION** optional, cost reported separately.
- **TEST** Batch-1 decode traffic and GEMV shape census from the measured census.
- **RESULT (D1)** `%MAC == %weights` for every region, so the MAC/byte mismatch
  that sank PocketTTS cannot occur here. **Ternary attacks the right region.**
- **RESULT** Ceiling with linears at *zero* bits is **3.06×** weight traffic, and
  **1.44×** end-to-end at L=4096. PTQTP delivers **1.10-1.23×** end-to-end.
- **RESULT** `IQ1_S`, already decodable by ingot, beats every ternary scheme at
  every context length.
- **RESULT** The tied head is 32.7% of Q4 traffic and **57.8%** at W1.58.
- **CONTRADICTION, priority evidence (D3)** Our measured decode is 13.6 GB/s
  where the sibling demonstrated 32-38 GB/s on the same machine. **The premise
  that batch-1 decode is weight-bandwidth-bound is not established for this
  model.** **DECISION** Phase 7b promoted ahead of Phases 3-5.
- **CLAIM SCOPE** Traffic and shapes only. No quality of ours is measured, and no
  decode time has yet been attributed to any term.

---

# Addendum 2, 2026-09-20 — the premise was already falsified in our own `docs/perf.md`

Before running Phase 7b, the repo was searched for an existing answer, per the
method's rule that rejected ideas and past measurements are evidence. **It was
already there, measured, and nobody had connected it to this question.**

## E1 — a format that reads 1.6× MORE bytes decodes 30% FASTER

`docs/perf.md`, the Q8_0 fused-matvec section, on granite-350m, ARM, 8 threads:

> **Q8_0 was the best-quality rung of the Granite ladder and the slowest; it is
> now the best-quality rung *and the fastest*, ahead of Q4_K_M (54.8 against
> 42.2).** […] If this decode were bandwidth-bound, Q8_0 could not win — it reads
> **1.6× the bytes** of Q4_K_M. **It is ALU-bound**, so the format that costs one
> multiply per weight beats the ones that cost a bit-plane reassembly, and the
> extra bytes are affordable. **The formats were never the ranking; the kernels
> were.**

That is a direct, already-measured refutation of this entire study's premise. The
proposition "fewer weight bytes ⇒ faster decode" is **false for this engine at
this model size**, and it was falsified by an experiment run for an unrelated
reason.

It is also the single most important paragraph in the repo for R1, and it came
from the same interleaved round in which *"every one of the other three
checkpoints drifted DOWN 5-10% on a warming machine while Q8_0 went up 61%"* —
i.e. it has its own control.

## E2 — the tied LM head is 42% of a decode step, and no method in the brief touches it

`docs/perf.md`, per-matvec, single-threaded, weights warm:

| tensor | type | shape | time |
|---|---|---|---|
| `attn_q` | Q4_K | 2048 × 1024 | 0.46 ms |
| `ffn_up` | Q4_K | 3072 × 1024 | 0.70 ms |
| `ffn_down` | Q6_K | 1024 × 3072 | 0.54 ms |
| **`lm_head`** (tied `token_embd`) | Q6_K | **151936 × 1024** | **42.55 ms** |

> **The LM head is 42% of a decode step, on its own.** Threading divides it but
> does not change its share, so this stays the largest single item.

Put beside Phase 7a: the head is **32.7% of decode bytes at Q4, 57.8% under
W1.58, and 42% of decode *time*.** A ternary scheme that ternarizes 100% of the
projections is optimising **58% of the bytes and 58% of the time**, and the
better it does, the larger the share it cannot touch. This is Amdahl's law
arriving twice, from two independent measurements, at the same tensor.

## E3 — threading does not behave like a memory wall

`docs/perf.md`'s thread table (taken earlier in the optimisation history, so the
absolute numbers are stale; the *shape* is what matters):

| threads | decode tok/s | vs 1 thread |
|---|---|---|
| 1 | 4.1 | 1.00× |
| 4 | 12.9 | 3.15× |
| 8 | 14.6 | **3.56×** |

Against PocketTTS's backbone, which is the reference bandwidth wall on this same
class of machine: **1.15× from eight cores.** Ours gets 3.56×. A region that
scales 3.6× with cores is not waiting on DRAM.

## E4 — `--fast` buys 25% without changing a single weight byte

The shipped CLI flag: *"`--fast`: int8 activations in the matvec. ~25% more
decode tok/s for ~1.4% perplexity — measured, not free."*

Quantizing **activations** changes zero weight bytes. If a 25% decode gain is
available on the activation side, at least that much of the step is not weight
streaming.

## What this does to the study

**The brief's sharpened question — "can Qwen3 preserve enough capability at
~1.6-3.2 effective bits that the reduction in DRAM traffic makes a dedicated CPU
ternary decode kernel worthwhile" — has a premise that our own measurements
reject.** The reduction in DRAM traffic is not the thing that makes decode fast
here. It is ALU-bound, the kernels are the ranking, and the largest single cost
is a matrix no ternary method quantizes.

Combined with Phase 7a's ceiling (PTQTP buys 1.10-1.23× of *traffic*, and
traffic is not the bottleneck) and F5's published quality (+82% perplexity on
this exact model), the ledger is:

| | |
|---|---|
| best case traffic saving, PTQTP | 1.10-1.23× end-to-end |
| best case traffic saving, W1.58 | 1.25-1.72× end-to-end |
| fraction of that convertible into time | **unknown, and demonstrably < 1** |
| share of decode time the method cannot touch | **42%** (the tied head) |
| published quality cost on this model | **+82% ppl, MMLU 47.1 → 33.6** |
| storage vs a format ingot already decodes | PTQTP **+30%**, W1.58 **+3%** |
| new work | PTQ pipeline, converter, new ggml type, NEON+AVX2 kernels, per-language quality gate |

**Phase 7b is still run** — with our own current numbers on Qwen3-0.6B rather
than granite-350m, because a conclusion inherited from another checkpoint is a
hypothesis — but it is now a confirmation, not a discovery.

## Evidence log — addendum 2

- **FACT (E1)** `docs/perf.md`: Q8_0 decodes 54.8 tok/s against Q4_K_M's 42.2 on
  granite-350m while reading **1.6× the bytes**. Recorded conclusion: *"It is
  ALU-bound […] the formats were never the ranking; the kernels were."*
- **FACT (E2)** The tied LM head is **42% of a decode step** and 57.8% of bytes
  under W1.58. No method in the brief quantizes it.
- **FACT (E3)** Decode scales **3.56×** on eight cores; the reference bandwidth
  wall scales 1.15×.
- **FACT (E4)** `--fast` buys ~25% decode by quantizing activations, changing no
  weight bytes.
- **DECISION** The premise "weight traffic dominates batch-1 decode" is rejected
  for this engine at this model size, on four independent pieces of our own prior
  evidence. Phase 7b is demoted from discovery to confirmation on Qwen3-0.6B.
- **CLAIM SCOPE** This says nothing about a 4B or 8B model, where the head's
  share falls and the weight/ALU balance shifts. Question 12 of the brief is
  untouched by it.

---

# Phase 7b — **DONE 2026-09-20**: decode is ALU-bound, measured on Qwen3-0.6B

Run against the pre-registered reading in the Phase 7b plan above, so the verdict
could not be chosen after seeing the numbers.

**Setup.** `mynah-slm v0.0.1-3-g2e52761`, built from a **clean committed tree**
(`make clean && make`), `Qwen3-0.6B-Q4_K_M.gguf` **staged local** in
`models-local/`, `-p "Write a short paragraph about the sea." -n 128 --think off
--no-stream --seed 1 --temp 0`. Deterministic: every run generates the identical
68 tokens, so the work is byte-for-byte the same at every thread count. Two
discarded warm-up runs, then five measured runs per point, one process at a time.
Median reported. M1, 4 performance + 4 efficiency cores. A development signal,
not a product claim.

| threads | decode tok/s | achieved GB/s | vs 1 thread | spread | `--fast` tok/s | `--fast` GB/s | gain |
|---|---|---|---|---|---|---|---|
| 1 | 12.8 | 5.00 | 1.00× | 0.8% | 18.9 | 7.39 | **+48%** |
| 2 | 22.2 | 8.68 | 1.73× | 3.2% | 31.6 | 12.35 | +42% |
| **4** | **37.8** | **14.77** | **2.95×** | 3.7% | **43.5** | **17.00** | +15% |
| 8 | 37.1 | 14.50 | 2.90× | 1.9% | *37.7* | *14.73* | *+2%* |

The 8-thread `--fast` row scattered from 22.6 to 45.4 tok/s — **60.5% spread**.
Per the execution discipline it is **not a measurement** and nothing is concluded
from it. Every other point held within 4%.

## Verdict against the pre-registered reading

The plan said: *flat from 2 threads up and ≥30 GB/s → traffic-bound, ternary is
live; scales with cores and ≪30 GB/s → not traffic-bound, cutting weight bytes
cannot pay.*

**It scales 2.95× on four cores and tops out at 14.8 GB/s.** The plateau from 4
to 8 threads is the P-core count, not a DRAM ceiling: the same machine was
demonstrated to sustain **32-38 GB/s** on the PocketTTS backbone
(`../mynah-tts/.work/backbone-bandwidth.md`), and that region scaled only 1.15×
on eight cores. Ours reaches **39-46% of the bandwidth this machine is known to
deliver** while scaling almost linearly with performance cores. Those are the two
signatures of a compute-bound region, and they are the opposite of the two
signatures of a memory wall.

**The decisive control is `--fast`.** It quantizes **activations** to int8 and
changes **not one weight byte**. At one thread it buys **+48%**. A step whose
time were set by streaming weights from DRAM cannot gain 48% from arithmetic on
the other operand. That single A/B, on our binary and our model, settles it more
cleanly than the roofline does.

## What this means for R1

The brief's sharpened question assumes *"the reduction in DRAM traffic makes a
dedicated CPU ternary decode kernel worthwhile"*. **On this engine, at this model
size, DRAM traffic is not what decode is waiting for.** So the chain

> ternary → 1.77× fewer weight bytes → 1.77× faster decode

is broken at the second arrow, and it is broken by our own measurement, not by an
argument. Phase 7a already showed the first arrow was worth only 1.10-1.23×
end-to-end for PTQTP once the KV cache and the untouchable LM head are counted.

Five independent lines now agree, three of them measured here and two inherited:

| evidence | source |
|---|---|
| decode scales 2.95× on 4 cores, 14.8 GB/s of a demonstrated 32-38 | **this run** |
| `--fast` buys +48% at 1 thread with zero weight-byte change | **this run** |
| Q8_0 beats Q4_K_M 54.8 vs 42.2 tok/s reading **1.6× the bytes** | `docs/perf.md` |
| the tied LM head is **42%** of a decode step and no method quantizes it | `docs/perf.md` |
| PTQTP buys 1.10-1.23× of *traffic* end-to-end vs the shipped file | Phase 7a |

## Consequences for the plan

1. **~~The kernel question is closed for v0.1/v0.2~~ — WITHDRAWN.** What holds:
   on *this* machine, with *our current Q4 kernels*, a traffic saving has little
   to convert into, so a ternary kernel would have to win on arithmetic or on a
   different ISA rather than on bytes. That is a constraint on the design, not a
   verdict. No C is written under R1 regardless — that was the rule from the
   start.
2. **Phases 1-5 are re-scoped from "is a backend justified" to "what is the
   quality floor of this model".** That is still worth having — it feeds the
   `IQ2`/`IQ3` decision the quantization policy already wants, and it answers U2
   — but it is a smaller study with a different deliverable.
3. **Phase 6's controls become the main event.** `IQ2_XXS` and `IQ1_S` are
   already decodable, already smaller than PTQTP, and their quality on this
   checkpoint is unmeasured. That is the cheapest unanswered question in the
   whole brief.
4. **The real optimisation target is now named and is not ternary**: the tied
   `lm_head`, 151936 × 1024, 42% of a decode step, and the activation path that
   `--fast` proves has 48% in it. Both are ordinary engineering in `src/`, and
   both belong on the board as their own items rather than inside R1.
5. **Question 12 is untouched.** Every number above is about a 0.6B model on an
   M1. At 4B the head's share falls to ~9.7% of parameters and the arithmetic per
   byte changes; on a many-core server with lower per-core bandwidth the roofline
   moves. **Nothing here forecloses ternary at 4B on a server**, and the brief's
   last question remains the one worth keeping open.

## Evidence log — Phase 7b

- **TEST** Thread sweep 1/2/4/8, five runs each, identical deterministic work,
  clean committed tree, local weights, `--fast` as the activation-side control.
- **RESULT** 12.8 / 22.2 / 37.8 / 37.1 tok/s; **2.95× on four cores**; peak
  **14.8 GB/s** against 32-38 demonstrated on this machine.
- **RESULT** `--fast` **+48%** at one thread, changing no weight bytes.
- **REFUSED** The 8-thread `--fast` point: 60.5% spread. Not reported as a
  measurement.
- **DECISION — WITHDRAWN 2026-09-20.** This line read *"VERDICT for the backend
  question: REJECT for v0.1/v0.2"*. It over-reached: the evidence shows what the
  **current Q4 implementation on this M1** is bound by, which is not the same
  question as whether a ternary representation can be executed efficiently on a
  target server ISA. **Gate B is UNDECIDED.** What the evidence does support,
  and what is kept: *batch-1 decode of Qwen3-0.6B at Q4_K_M on this Mac is
  ALU-bound, not weight-bandwidth-bound.* That is a fact about our current
  kernels, and it raises the bar a ternary kernel would have to clear here — it
  does not decide the backend.
- **DECISION** R1 continues as a *quality* study (U2 and the `IQ` controls), not
  as a backend investigation. Two new board items fall out of it: the tied LM
  head, and the activation path.
- **CLAIM SCOPE** One M1, one 0.6B checkpoint, one engine, Q4_K_M weights. Says
  nothing about 4B/8B, about servers, or about any other engine's kernels.

---

# Addendum 3, 2026-09-20 — PTQTP's real accounting, and what its artifact contains

Full detail in [`ptqtp-paper-reading.md`](ptqtp-paper-reading.md). What changes
here:

## The bit accounting was wrong, and in our favour

Phase 0 and Phase 7a used **3.375 bits/weight** for PTQTP — two `TQ1_0`-packed
base-3 planes. **The authors do not pack.** Appendix A.3 stores *"each ternary
element … with 2 bits (since 3 ≤ 2²)"*, i.e. **4.000 bits/weight**, plus 0.250
for grouped fp16 scales = **4.250**. Bit-packing is listed under *Limitations and
Future Works*.

And the released Qwen3-0.6B artifact **protects `q_proj` and `k_proj` in all 28
blocks** — verified by counting distinct values per 128-group across every
projection — so real coverage is **80.0% of the linears, 59.1% of the model**,
not §4.1's "all linear layers".

Re-run with both axes (`tools/qwen_ternary_feasibility.py traffic`):

| PTQTP variant | coverage | MiB/token | vs shipped Q4_K_M |
|---|---|---|---|
| **as implemented, the artifact's own coverage** | **59.1%** | **369.4** | **1.01×** |
| as implemented, all linears | 73.9% | 345.1 | 1.08× |
| packed-optimal, the artifact's coverage | 59.1% | 332.6 | 1.12× |
| packed-optimal, all linears (our old figure) | 73.9% | 299.2 | 1.25× |

**PTQTP as the authors actually built it is the same size as the `Q4_K_M` file we
ship today — 1.01×.** The 1.25× in Addendum 1 was a best case that requires a
packing nobody implemented *and* a coverage the artifact does not have. Both
rows stay in the tool, labelled, because the optimistic one is the right bound
for "could a future implementation do better" and the pessimistic one is the
right number for "what did the paper do".

## Three further facts that bear on the gates

1. **The paper's own GPU kernel loses to 4-bit GPTQ** (Table 5, RTX 4090,
   `gate_proj`, batch 1, 7B): FP16 0.122 ms, **GPTQ-4bit 0.085**, PTQTP 0.120.
   At seq 2048 PTQTP is 1.82× slower than FP16. Max speedup claimed anywhere in
   the paper is **1.16×** (Table 6, attention). This is evidence *for Gate B's
   difficulty*, on a GPU; it does not decide Gate B on a CPU ISA, and it is not
   used to.
2. **PTQTP needs no calibration data** (§4.1: *"No task-specific calibration,
   tuning, or fine-tuning was applied in any experiment"*). Method C is a
   closed-form weight-only fit. **Phase 1 is therefore not a prerequisite for
   Method C** — only for the activation-aware methods. Method C moves ahead of
   Phase 1 in the plan.
3. **The artifact applies a channel-scale step the paper never mentions.**
   `gate_proj`/`up_proj` carry norm ratios up to **4.95×** against the original —
   and the layernorms that feed them are divided by exactly the reciprocal:
   `post_attention_layernorm × gate_proj = 0.99-1.02` on every layer, and per
   input channel `‖W_artifact − W_orig·s‖/‖W_artifact‖ = 2.9e-04`. This is
   **per-channel scale migration**, the AWQ/SmoothQuant trick, used to flatten
   the weight distribution before fitting a ternary grid. It is exactly
   function-preserving and **free at inference** — the norms are already there.
   A reimplementation from the paper alone (ours) does not have it. Detail in
   [`ptqtp-paper-reading.md`](ptqtp-paper-reading.md).

## Plan changes

- **Gate A starts now**, with `tools/qwen_ternary_feasibility.py ppl` — WikiText-2,
  non-overlapping 2048-token windows, the protocol the GPTQ lineage uses, so our
  numbers are comparable to the papers'. First gate: our BF16 baseline must
  reproduce the published **20.9** before any quantized number is interpretable.
- **Method C is implemented from Algorithm 1**, not adapted from the artifact,
  and cross-checked against the artifact only where the artifact looks sane.
- Phase 1 (calibration) is demoted behind Method C.

## Evidence log — addendum 3

- **FACT** PTQTP's own representation is **4.250 bits/weight** (App. A.3 + Eq. 9),
  not 3.375 and not 1.58. Its Appendix A.3 worked example claims 7.96×
  compression where its own formula and its own Table 4 give **4×** / 3.65×.
- **FACT** The released `Qwen3-0.6B-PTQTP-1.58b` is **dense FP16, 1136.9 MiB**,
  310 tensors, no packing and no scale tensors.
- **FACT** Coverage is **5 of 7 projection families**; `q_proj` and `k_proj` are
  dense in **0/28** blocks ternarized. `embed_tokens` and `model.norm` are
  bit-identical to the original.
- **FACT** A quantized group decomposes exactly: `gate_proj` L13 row 0 group 0 is
  `{0, ±0.04310, ±0.09045, ±0.13354, ±0.22400}` = `α₁=0.13354, α₂=0.09045`.
- **RESULT** **PTQTP as implemented, at the artifact's coverage, is 1.01× the
  size of the file we already ship.**
- **CONTRADICTION, RESOLVED** `gate_proj`/`up_proj` norm ratios of up to 4.95×
  looked like a broken upload. The artifact scores **35.256** ppl, so it is not.
  The layernorms absorb the scale exactly (`post_ln × gate = 0.99-1.02`;
  per-channel residual 2.9e-04): an **undocumented channel-scale
  reparameterization**. The premature "may be broken" reading is corrected in
  `ptqtp-paper-reading.md`. *When a result is absurd, suspect the setup before
  the subject* — one `ppl` run settled it.
- **CLAIM SCOPE** Storage, coverage and reconstruction only. No perplexity of
  ours has been measured yet; Gate A is unanswered.

---

# GATE A — 2026-09-20

## A0 — the evaluation protocol reproduces the paper's FP16 number  `[MEASURED]`

Before any quantized number can be read, the harness has to land on a figure
someone else published. `tools/qwen_ternary_feasibility.py ppl`: WikiText-2-raw
test, joined with `\n\n`, tokenized once, cut into **non-overlapping 2048-token
windows** — the GPTQ-lineage protocol that PT²-LLM, TWLA and the Qwen3-4B study
all inherit — averaging token NLL over all 146 full windows (298,862 tokens),
fp16 on MPS.

| | WikiText-2 ppl |
|---|---|
| **ours, `Qwen/Qwen3-0.6B` @ `c1899de2`** | **20.954** |
| PTQTP Table 1, Qwen3 0.6B FP16 | **20.9** |
| difference | **+0.26%** |

**The gate passes.** Our numbers are directly comparable to the paper's, so a
quantized result of ours can be read against their 38.02 without an asterisk.
Worth stating why that matters: had the baseline come out at, say, 13 or 30, no
quantized measurement afterwards could have been interpreted at all.

Recorded: `reports/ternary/ppl_qwen3-0.6b-bf16.json`.

## A1 — the four conditions  `[MEASURED]`

Identical harness, identical protocol, identical 146 windows. Every checkpoint is
a dense fake-quantized FP16 safetensors, so the *only* variable is the weight
values. `reports/ternary/ppl_*.json`.

| # | condition | coverage | WikiText-2 ppl | vs BF16 |
|---|---|---|---|---|
| 0 | **BF16 baseline** (ours) | — | **20.954** | — |
| | *PTQTP Table 1, Qwen3-0.6B FP16* | | *20.9* | *+0.26% vs ours* |
| 1 | **authors' released artifact** | 59.1% | **35.256** | **+68%** |
| | *PTQTP Table 1, Qwen3-0.6B PTQTP* | *59.1%* | *38.02* | *+82%* |
| 2 | **Method C, `q`/`k` protected** (artifact's coverage) | 59.1% | **43.331** | **+107%** |
| 3 | **Method C, all seven families** (§4.1's claim) | 73.9% | **59.539** | **+184%** |
| 4 | **naive single-plane W1.58**, all families | 73.9% | **684,110** | **destroyed** |

Reconstruction error over the 196 quantized tensors, for reference: Method C
mean **0.1784** (min 0.1738, max 0.1903); naive mean **0.4446**.

### What each row settles

**Row 4 answers the brief's question 1 outright.** *"Can Qwen3-0.6B tolerate
single-plane post-training W1.58 at useful coverage?"* — **No.** Not at 100%
coverage, not by a factor of thirty thousand. This is naive fitting without GPTQ
compensation, so it is the control the brief asked for and not the best a
single-plane method can do; but it establishes that the two trit planes are
load-bearing, not a refinement.

**Row 3 vs row 2 prices the coverage decision at 16 perplexity points.**
Ternarizing `q_proj` and `k_proj` takes the model from 43.3 to 59.5. Whatever the
authors' reason for protecting them, it is worth a third of the model's remaining
quality, and §4.1's "all linear layers were quantized" would have cost us that
had the artifact not been inspected.

**Row 2 vs row 1 prices the undocumented step at 8 perplexity points.** Same
coverage, same algorithm, same tolerance and iteration cap; the only difference
is the per-channel scale absorption the paper does not describe.
**PTQTP-as-published scores 43.3; PTQTP-as-implemented scores 35.3.** That is the
value of the missing paragraph, and it is large.

**Row 1 beats the paper's own published number** (35.256 against 38.02), which is
a useful sanity check in both directions: our harness is not being kind to the
artifact, and the artifact is a real PTQTP conversion rather than a broken upload.

### Gate A verdict, on the evidence so far

**Qwen3-0.6B does not tolerate post-training ternarization well.** The best result
anyone has produced on this checkpoint — the authors' own, with a technique they
did not publish — is **+68% perplexity**, and the paper's companion MMLU number
for the same model is **47.1 → 33.64**, which is close to the floor for a
four-choice benchmark. Our faithful reimplementation is +107%.

Put beside Phase 0/7a: **PTQTP as implemented is 369.4 MiB against the 372.7 MiB
`Q4_K_M` file we ship today — 1.01×.** So on this model the method costs +68%
perplexity and buys **1% of size**.

**This is not yet a verdict on ternary in general, and explicitly not on Gate B.**
Two things are still missing and both are cheap:

1. **The `IQ2_XXS` / `IQ1_S` controls (U2).** This is the measurement that decides
   whether the finding is about *ternary* or about *0.6B models*. If `IQ2_XXS` at
   230.2 MiB also collapses, then nothing survives below ~3 bits at this size and
   ternary was never the variable. If it holds up, ternary is dominated on both
   axes at once and that is the end of it for v0.1/v0.2.
2. **Channel-scale absorption added to Method C.** Row 1 proves it is worth 8
   points, it is function-preserving, and it costs nothing at inference. Our
   Method C is incomplete without it, and a single-plane method *with* it has not
   been tried by anyone.

## Evidence log — Gate A

- **TEST** Four conditions, one harness, 146 non-overlapping 2048-token windows.
- **RESULT** BF16 **20.954** vs published 20.9 — protocol validated (+0.26%).
- **RESULT** Naive single-plane W1.58 at full coverage: **684,110**. Question 1
  of the brief is answered: **no**.
- **RESULT** Method C all-linears **59.539**; `q`/`k` protected **43.331**.
  Coverage is worth **16 points**.
- **RESULT** Authors' artifact **35.256**, better than their own published 38.02.
  The undocumented channel rescale is worth **8 points**.
- **DECISION** Gate A leans **negative for Qwen3-0.6B**: +68% perplexity at best,
  for 1.01× the size of the shipped file. **Not final** — the `IQ` controls and
  scale-absorbed Method C are outstanding.
- **CLAIM SCOPE** One checkpoint, WikiText-2 only, no tool-call or multilingual
  eval yet, weights-only, A16. Says nothing about 4B, and nothing about Gate B.

## A2 — channel-scale absorption, implemented and tested

The artifact's undocumented step is worth 8 perplexity points (A1). It was
implemented here in two forms, because which one it is decides whether §4.1's
"no calibration" claim survives.

### It is function-preserving, proven rather than asserted  `[MEASURED]`

Absorb only, quantize nothing, compare logits on a 256-token sequence against
the untouched model:

| | groups rescaled | rel_l2(logits) | argmax agreement |
|---|---|---|---|
| weight-only, α=0.5 | 56 | **3.10e-06** | **100.00%** |
| weight-only, α=1.0 | 56 | **2.60e-06** | **100.00%** |

56 = 28 blocks × 2 absorbable norms. The residual is fp32 rounding. The control
— reloading the original state dict and re-running — reproduces the base logits
**exactly**, so the harness itself is not the thing being measured.

### The weight-only form buys nothing  `[MEASURED]`

`s_j = (col_rms of the stacked consumers)^(-α)`, geomean-normalised — the AWQ
shape without AWQ's activation term, which is what §4.1's "no calibration"
permits. Reconstruction error of Method C over nine representative tensors
(blocks 0/13/27 × `v_proj`/`gate_proj`/`down_proj`):

| α | 0.00 | 0.25 | 0.50 | 0.75 | 1.00 |
|---|---|---|---|---|---|
| mean rel err | 0.1789 | 0.1788 | 0.1787 | 0.1787 | 0.1788 |

**Flat to 0.1%.** In hindsight this is predictable and worth writing down:
PTQTP's α is already fitted **per group of 128 consecutive input channels**
(§3.2 reshapes `n × d` to `nd/G × G`). A smooth per-channel rescale inside a
group is largely redundant with the group's own scale, so flattening by a weight
statistic moves almost nothing.

**That is itself evidence about the artifact.** Their scale buys 8 perplexity
points; the weight-only form of the same transformation buys ~0. Combined with
the log-correlations of 0.04-0.52 against column max/rms/mean, the remaining
candidate is an **activation**-derived scale — which §4.1 says was not used.

Caveat kept explicit: reconstruction error is measured in *weight* space, and an
activation-aware scale deliberately trades weight error for output error. A flat
weight curve does not prove the weight-only form is useless for perplexity; it
proves it does not reduce weight error. The perplexity run is the test.

### The activation-aware form

`s_j = (mean|x_j|)^α` over the norm's output, geomean-normalised, 32 calibration
sequences of 2048 tokens from **WikiText-2 train** — never the test split the
perplexity is measured on. Running: α ∈ {0.5, 1.0} with `q`/`k` protected, plus
an **absorb-only control that must return 20.954** and which is the end-to-end
proof that nothing else changed.

### A2 results — the absorption we could implement makes it WORSE  `[MEASURED]`

| condition | ppl | vs the same thing without absorption |
|---|---|---|
| **control: absorption only, nothing quantized** | **20.939** | baseline is 20.954 → **−0.07%** |
| Method C + `q`/`k` protected, no absorption | 43.331 | — |
| **Method C + `q`/`k` protected + activation absorption α=+0.5** | **62.316** | **+44% worse** |

**The control is the important row.** 20.939 against a 20.954 baseline, with 311
tensors untouched and 56 norm groups rescaled: absorption is function-preserving
end to end, and the 0.07% is fp16 storage rounding of the rescaled values. So the
62.316 is a real effect of the scale, not a bug in the plumbing — the plumbing was
proven first, separately, on logits (3.1e-06) and now on perplexity.

**And the effect is negative.** AWQ's scale direction — `s_j = (mean|x_j|)^α`,
amplifying the weight columns of high-activation channels so their *relative*
quantization error shrinks — costs 19 perplexity points here.

The likely reason, stated as a hypothesis before testing it: **AWQ's direction is
tuned for an asymmetric integer grid whose step is set by `max|W|` within a
group.** Amplifying a few salient columns there is affordable. A trit-plane pair
has **9 reachable values per group of 128**, and widening the group's dynamic
range is exactly what a 9-level grid cannot absorb — it spends its levels on the
amplified columns and coarsens everything else. For ternary the useful direction
may be the opposite: *flatten* the range rather than protect the salient channels.

That is testable with one sign flip (`--absorb-alpha -0.5`), and it is queued.

**Either way, one thing is already settled**: the 8-point gain in the authors'
artifact is **not** reproduced by the AWQ-shaped rule, in either the weight-only
form (which moved nothing, §A2) or this activation form (which moved 19 points in
the wrong direction). Whatever their undocumented step is, it is not this.

## A3 — the conventional sub-4-bit controls  `[MEASURED]` — this answers U2

The question U2 asked: **is the failure about ternary, or about 0.6B models?**
The control that decides it is whether ordinary codebook quants at the same bit
budget survive where ternary does not.

`llama.cpp` (`2115b73`), Metal, `-c 2048 --chunks 60`, importance matrix built
from WikiText-2 **train** (200 chunks) and used for every IQ quantization. All
five rows use the identical harness and the identical 60 chunks, so they are
mutually comparable.

**They are NOT comparable to the HF numbers above**: `llama-perplexity` scores
only the second half of each window, which is why its F16 baseline is 17.15
where our BF16 baseline is 20.95. Ratios within a harness are the comparable
quantity, and both columns are given.

| format | bytes | ppl | ratio to that harness's baseline |
|---|---|---|---|
| **F16** | 1,509,347,584 | **17.151** | 1.000 |
| **Q4_K_M** | 484,220,512 | **17.618** | **1.027** |
| **IQ3_XXS** | 345,867,872 | **31.886** | **1.859** |
| **IQ2_XXS** | 280,496,736 | **1732.32** | **101.0** |
| **IQ1_S** | 259,066,464 | **6211.39** | **362.2** |

Caveat on the byte column, because it would otherwise flatter the comparison in
the wrong direction: our `convert_hf_to_gguf.py` kept `output.weight` as a
separate tensor despite `tie_word_embeddings: true`, so **every file above
carries the 155.6M-parameter embedding twice.** The shipped `Q4_K_M` build is
397 MB where ours is 484 MB, for exactly that reason. Quality is unaffected;
size comparisons must use the deduplicated figures from Phase 0.

### U2 is answered: the failure is about the model, not about ternary

**Nothing conventional survives below ~3 bits on Qwen3-0.6B.** `IQ2_XXS` at
2.06 bpw is 101× the baseline and `IQ1_S` at 1.56 bpw is 362× — word salad, the
same class of failure as the `Q2_K` result this repo already recorded on
granite-350m (ppl 26400). `IQ3_XXS` at 3.06 bpw survives but costs +86%.

So the earlier framing in Phase 0 — *"IQ1_S is smaller than every ternary scheme
and needs no research"* — **was wrong in the way that matters.** `IQ1_S` is
smaller and it is also unusable. It was never the cheap alternative; it is not an
alternative at all.

### Which makes ternary look considerably better than Phase 0 implied

Put the two harnesses side by side as ratios to their own baselines:

| scheme | effective bits on linears | ppl ratio | harness |
|---|---|---|---|
| Q4_K_M | 4.5 (5.24 whole-model) | **1.027** | llama.cpp |
| **PTQTP, authors' artifact** | **4.25 on 59.1% coverage** | **1.683** | HF |
| IQ3_XXS | 3.06 | 1.859 | llama.cpp |
| Method C (ours, no absorption) | 4.25 on 59.1% | 2.068 | HF |
| IQ2_XXS | 2.06 | **101.0** | llama.cpp |
| IQ1_S | 1.56 | **362.2** | llama.cpp |
| naive W1.58 | 1.69 | **32,648** | HF |

**The two trit planes are doing real work.** At a comparable bit budget, PTQTP
lands at 1.68× baseline where `IQ2_XXS` — a mature, imatrix-calibrated codebook
quant at 2.06 bpw — is at 101×, and naive single-plane ternary at 1.69 bpw is at
32,648×. Ternary is not a worse idea than the conventional sub-3-bit options; on
this model it is **decisively better than all of them**.

### And it still does not clear the bar that matters

The comparison that decides shipping is not against `IQ2_XXS`. It is against
`Q4_K_M`, which costs **+2.7%** perplexity for 397 MB — and PTQTP costs **+68%**
for a file of the same size (369.4 MiB, Addendum 3). The honest one-line summary
of Gate A:

> **Sub-4-bit is not viable on Qwen3-0.6B by any method measured, and PTQTP is
> the best sub-4-bit method measured.** Both halves are true and neither is
> interesting on its own.

## Evidence log — A3

- **TEST** Five formats, one llama.cpp harness, 60 identical chunks, imatrix from
  wikitext train.
- **RESULT** F16 17.151 · Q4_K_M 17.618 (**1.027×**) · IQ3_XXS 31.886 (1.859×) ·
  IQ2_XXS 1732.3 (**101×**) · IQ1_S 6211.4 (**362×**).
- **DECISION, U2 ANSWERED** The collapse is a property of **0.6B at low bit
  depth**, not of ternarization. Phase 0's "IQ1_S beats every ternary scheme on
  size and needs no research" is **retracted**: it is smaller and unusable.
- **RESULT** Ternary is the *best* sub-4-bit representation measured on this
  model — 1.68× against IQ2_XXS's 101× at a comparable budget.
- **CLAIM SCOPE** WikiText-2 only, 60 chunks, one checkpoint. No tool-call or
  multilingual eval. Two harnesses, compared as ratios and never as absolutes.

### A2 continued — the sign hypothesis is falsified  `[MEASURED]`

The hypothesis recorded above was that ternary wants the *opposite* of AWQ's
direction: flatten the per-channel range rather than protect the salient
channels. One sign flip tests it.

| α | what it does | ppl |
|---|---|---|
| **−0.5** | flatten: shrink high-activation channels' weight columns | **781.96** |
| **0** | no absorption | **43.33** |
| **+0.5** | AWQ: amplify high-activation channels' weight columns | **62.32** |

**Wrong. Flattening is 12× worse than AWQ's direction and 18× worse than doing
nothing.** The hypothesis is dead: AWQ's sign is the *right* sign, it is just
harmful at this magnitude. The curve's minimum is at or very near α=0, and it is
strongly asymmetric — the penalty for going the flattening way is an order of
magnitude larger.

The reading that survives: **Qwen3-0.6B's channels are already close to
well-conditioned for a group-wise trit fit**, and any per-channel scale we can
derive — from weights or from activations, in either direction — costs more than
it buys. That is a complete negative result across **four** conditions
(weight-only α∈[0,1]: flat; activation α=+0.5: +19 points; activation α=−0.5:
+739 points), and it closes the question we opened:

> **The authors' undocumented 8-point step is not channel-scale absorption of the
> AWQ family.** We can invert their scale exactly and reproduce the transform,
> but not derive it, and every derivable rule in that family makes things worse.
> What it is remains open, and settling it needs their code.

---

# PHASE A — the locked table, one harness, 2026-09-20

Everything below is `[MEASURED]` on **one** harness with **one** protocol. The
earlier split between our HF runs and llama.cpp's is gone: GGUF files are now
loaded through `transformers` (which dequantizes them), so every format is
scored identically and the absolutes are comparable across the whole table.

## Reproducibility record

| | |
|---|---|
| model | `Qwen/Qwen3-0.6B` @ `c1899de289a04d12100db370d81485cdf75e47ca` |
| artifact | `yang31210999/Qwen3-0.6B-PTQTP-1.58b` @ `6dfb5a8fa4055ef2a67a4712342ab1562a2bb969` |
| tokenizer | the model's own, same revision |
| dataset | `Salesforce/wikitext`, config `wikitext-2-raw-v1`, split `test` |
| join | `"\n\n".join(rows)`, tokenized once |
| protocol | **non-overlapping** windows, score positions 1..L−1 of each |
| seq len | 2048 |
| windows | **146**, **298,862 tokens scored** |
| dtype / device | fp16 / mps (fp32 where noted) |
| loss | `lm_head` + cross-entropy in 256-position chunks, summed then averaged |
| GGUF quants | `llama.cpp` `2115b73`, imatrix from WikiText-2 **train**, 200 chunks |
| versions | recorded per run in `reports/ternary/ppl_*.json` |

**Harness validation**: the rewritten harness reproduces the numbers taken
before it to four decimals — BF16 **20.9541** against 20.954, PTQTP artifact
**35.2560** against 35.256.

## The table

| condition | bits (linears) | ppl | ratio to BF16 |
|---|---|---|---|
| **BF16** | 16 | **20.9541** | 1.000 |
| **Q8_0** | 8.5 | **20.9080** | **0.998** |
| **Q4_K_M** | 4.5 | **21.6455** | **1.033** |
| **Q3_K_M** | 3.4375 | **24.6408** | **1.176** |
| **PTQTP, authors' artifact** | ~4.25 on 59.1% coverage | **35.2560** | **1.683** |
| **IQ3_XXS** | 3.0625 | **39.8882** | **1.904** |
| **Method C, q/k protected** | 4.25 on 59.1% | **43.3312** | **2.068** |
| **Method C, all linears** | 4.25 on 73.9% | **59.5391** | **2.841** |
| **IQ2_XXS** | 2.0625 | **1265.43** | **60.4** |
| **IQ1_S** | 1.5625 | **NaN in fp16** | — |
| **naive W1.58** | 1.6875 | **684,110.5** | **32,648** |

`[MEASURED]` `IQ1_S` returns NaN under fp16: the damaged model leaves the fp16
activation range. That is a property of the quantization, not of the metric, and
it is reported rather than hidden. An fp32 re-run is in flight.

## The result that changes the picture

**`Q3_K_M` beats PTQTP on both axes at once.**

| | bits/weight | whole-model MiB (deduplicated, Phase 0 model) | ppl ratio |
|---|---|---|---|
| **Q3_K_M** | 3.4375 | **302.4** | **1.176** |
| PTQTP as implemented | 4.25 on 59.1% | 369.4 | 1.683 |

`[DERIVED]` **18% smaller and 43% less perplexity damage**, in a format ingot
already decodes and `src/qmat.c` already has a fused kernel for.

This had been invisible because Q3_K_M's quality on *this* model had never been
measured — `docs/models.md` carries it only for granite-350m, where it was
recorded as "survives but a poor trade". On Qwen3-0.6B it is not a poor trade at
all; it is the best sub-4-bit point measured, by a wide margin.

## Two more corrections the table forces

1. **`IQ3_XXS` is not the sub-3-bit survivor it looked like.** At 1.904 it is
   *worse* than the PTQTP artifact, and far worse than Q3_K_M at a higher bit
   rate. The K-quants beat the IQ codebook quants badly on this model — the
   opposite of the ordering the bit rates suggest.
2. **The cliff is between 3.06 and 2.06 bpw, and it is a cliff, not a slope**:
   1.904 → 60.4 → NaN. C3's restated form survives, with the interval now
   bounded from both sides by measurements on the same harness.

## What the table does NOT say

`[UNKNOWN]` Nothing here is a behavioural result. Per C11, a perplexity ratio of
1.68 may or may not correspond to a usable model; the Tied paper documents a case
where reconstruction error and perplexity both worsened while task fidelity did
not. The tool-call eval is the missing measurement and it is cheap.
