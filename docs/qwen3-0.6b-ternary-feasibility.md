# Post-training ternarization of Qwen3-0.6B — feasibility

**Status: Phase 0 complete. Phases 1-7 not measured.** Everything below is
labelled `[MEASURED]`, `[PUBLISHED]` (someone else's number, cited) or
`[NOT MEASURED]`. Nothing is estimated and presented as a result.

Work item and plan: [`.work/ternary-feasibility.md`](../.work/ternary-feasibility.md).
Tool: `tools/qwen_ternary_feasibility.py`. Raw outputs: `reports/ternary/`.

Checkpoint: `Qwen/Qwen3-0.6B`, revision `c1899de289a04d12100db370d81485cdf75e47ca`,
Apache 2.0, BF16 safetensors, 1433.7 MiB. Machine: Apple M1, 16 GB, weights
staged **local** (never the NAS).

---

## 1. What is being asked

Whether a normal pretrained Qwen3-0.6B can be converted post-training into a
ternary or multi-trit representation that keeps useful quality, and whether that
would justify a CPU-native ternary backend.

The question has to be asked against what this repo already ships, not against
BF16: `Qwen3-0.6B-Q4_K_M` at **372.7 MiB / 5.24 bits/weight**, and the `IQ1`-`IQ4`
and `TQ1_0`/`TQ2_0` formats that `third_party/ingot` already decodes.

## 2. Phase 0 — the census  `[MEASURED]`

`python3 tools/qwen_ternary_feasibility.py census` — read from the safetensors
header, no weights loaded.

**596,049,920 distinct parameters**, 751,632,384 stored.

| family | tensors | shape | params | % of model |
|---|---|---|---|---|
| embedding (tied) | 1 | `[151936, 1024]` | 155,582,464 | **26.10%** |
| down_proj | 28 | `[1024, 3072]` | 88,080,384 | 14.78% |
| gate_proj | 28 | `[3072, 1024]` | 88,080,384 | 14.78% |
| up_proj | 28 | `[3072, 1024]` | 88,080,384 | 14.78% |
| o_proj | 28 | `[1024, 2048]` | 58,720,256 | 9.85% |
| q_proj | 28 | `[2048, 1024]` | 58,720,256 | 9.85% |
| k_proj | 28 | `[1024, 1024]` | 29,360,128 | 4.93% |
| v_proj | 28 | `[1024, 1024]` | 29,360,128 | 4.93% |
| norms | 113 | `[1024]`, `[128]` | 65,536 | 0.01% |

| group | params | share |
|---|---|---|
| transformer linears (ternarizable) | 440,401,920 | **73.89%** |
| — attention | 176,160,768 | 29.55% (40% of linears) |
| — MLP | 264,241,152 | 44.33% (**60% of linears**) |
| embedding | 155,582,464 | **26.10%** |
| norms | 65,536 | 0.01% |

### The checkpoint stores the tied weight twice  `[MEASURED]`

`config.json` declares `tie_word_embeddings: true` and the file ships both
`lm_head.weight` and `model.embed_tokens.weight` — 311,164,928 bytes each,
**byte-identical by sha256**. The duplicate is 296.8 MiB of a 1433.7 MiB file.

Any percentage taken over the stored 751M parameters instead of the distinct
596M overstates a scheme's coverage and compression by 26%. The tool
deduplicates and prints the check; `reference/qwen3-0.6b/safetensors_header.json`
is the stored view and must not be used for this.

### The embedding is the LM head

`token_embd.weight` is the input lookup *and* the output projection: the largest
tensor, the hottest GEMV in the decode loop, and the layer directly in front of
the softmax. llama.cpp holds it at Q6_K while pushing the rest to Q4_K — which
is why the shipped file is 5.24 bits/weight, not 4.5 (`docs/models.md`).

Every ternarization method in the literature quantizes "all linear layers" and
leaves this tensor alone. At 7B that footnote is worth a few percent. **Here it
is 26.10% of the model**, and it sets a floor no weight-only ternary scheme can
cross.

## 3. Phase 0 — the storage budget  `[MEASURED]`

`python3 tools/qwen_ternary_feasibility.py budget`. Scale overhead derived from
the real per-tensor row/column geometry. Ternary rows use the packed `TQ1_0`
rate of **1.6875 bpw**, not the nominal 1.58.

**Model validation:** the exact `Q4_K_M` recipe (embedding + `attn_v`/`ffn_down`
on 14 of 28 layers at Q6_K) models to **372.7 MiB / 5.245 bpw** against the
**372.7 MiB / 5.24 bpw measured** on the real file — 0.01%. The tool refuses to
print the table if that check exceeds 1%.

| scheme | linear bpw | embedding | MiB | whole-model bpw |
|---|---|---|---|---|
| BF16 as published | 16.000 | bf16 | 1137.0 | 16.002 |
| Q8_0 everywhere | 8.500 | q8_0 | 604.1 | 8.503 |
| **Q4_K_M — shipped today** | 4.500 | q6_k | **372.7** | **5.245** |
| Q3_K_M | 3.438 | q6_k | 302.4 | 4.256 |
| **IQ2_XXS linears** — ingot decodes it | 2.062 | q6_k | **230.2** | **3.240** |
| **IQ1_S linears** — ingot decodes it | 1.562 | q6_k | **204.0** | **2.871** |
| W1.58 single plane, row-wise α | 1.688 | bf16 | 385.6 | 5.427 |
| W1.58 single plane, row-wise α | 1.688 | q6_k | 210.6 | 2.963 |
| W1.58 single plane, (α,μ) per 128 | 1.938 | q6_k | 223.7 | 3.148 |
| **PTQTP 2×1.58, row-wise α** | 3.375 | bf16 | **474.2** | **6.674** |
| PTQTP 2×1.58, row-wise α | 3.375 | q6_k | 299.2 | 4.210 |
| PTQTP 2×1.58, grouped α (G=128) | 3.625 | q6_k | 312.3 | 4.395 |

### Two results that follow immediately

1. **PTQTP as the papers run it — all linears ternary, embedding untouched — is
   474.2 MiB: 27% LARGER than the Q4_K_M file we ship today.** The "1.58-bit"
   label describes 73.89% of the parameters at 3.375 bits, next to 26.10% still
   at 16.
2. Once the embedding is allowed to drop to Q6_K (a decision independent of
   ternary), the comparison that matters is against formats ingot already reads:

| | MiB | vs Q4_K_M | vs best ternary |
|---|---|---|---|
| PTQTP 2×1.58 | 299.2 | 1.25× smaller | 30% **larger** than IQ2_XXS |
| W1.58 single plane | 210.6 | 1.77× smaller | — |
| IQ2_XXS | 230.2 | 1.62× smaller | ternary wins by 9% |
| IQ1_S | 204.0 | 1.83× smaller | ternary **loses by 3%** |

**On bytes, ternary buys nothing on this model** that an existing ggml codebook
format does not already buy with no research and no new kernel. Any remaining
case rests on arithmetic — a multiplication-free GEMV — which is a kernel
question and out of scope here.

## 4. What the literature reports for this exact model  `[PUBLISHED]`

PTQTP (arXiv [2509.16989](https://arxiv.org/abs/2509.16989)) evaluates the Qwen3
family down to 0.6B — the authors' own numbers, their own implementation:

| model | WikiText2 ppl FP16 → PTQTP | MMLU FP16 → PTQTP |
|---|---|---|
| **Qwen3-0.6B** | **20.90 → 38.02 (+82%)** | **47.1% → 33.6%** |
| Qwen3-1.7B | 16.70 → 32.46 (+94%) | 60.0% → 43.8% |
| Qwen3-4B | 13.64 → 18.25 (+34%) | 69.7% → 63.7% |
| Qwen3-32B | 8.64 → 10.06 (+16%) | — |

**Degradation is monotone in model size, and 0.6B is the worst case in the
paper.**

A practical Qwen3-4B conversion (arXiv
[2609.01962](https://arxiv.org/abs/2609.01962), KOTMS + E2M-ATQ + GPTQ error
compensation, weight-only, A16) corroborates independently: WikiText2
13.639 → 18.748, PTB 24.700 → 31.992, C4 19.831 → 28.966; ten capabilities
64.5% → 54.7%. **1.641 effective bits/weight over 81.62% of parameters**, packed
8.29 GiB → 3.96 GiB — a **2.09×** whole-file compression, not the 9.7× the label
suggests. Its own conclusion on speed: *"we therefore do not claim that
compression alone yields faster inference"* — their Triton ternary GEMV measured
**4.6× slower than FP16 cuBLAS**.

PT²-LLM (arXiv [2510.03267](https://arxiv.org/abs/2510.03267), ITF + AGA + SSR,
128×2048 WikiText2 calibration, block 128) publishes LLaMA-7B/13B/65B, LLaMA-2,
LLaMA-3-8B and Qwen3-14B-Base — **nothing near 0.6B**. Its LLaMA-7B result is
5.68 → 11.39, a 2× perplexity cost on a model twelve times larger than this one.

TWLA (arXiv [2606.13054](https://arxiv.org/abs/2606.13054)) contributes E2M-ATQ,
KOTMS and ILA-AMP; its headline configuration pairs W1.58 with **A4**, which this
study deliberately does not attempt — weights first.

## 5. Container support  `[MEASURED]`

`third_party/ingot` decodes `TQ1_0` (id 34, 1.69 bpw) and `TQ2_0` (id 35,
2.06 bpw) today, generic kernel, no SIMD on either architecture.

- **Single-plane W1.58 has a container already.** `TQ1_0` is an exact fit; a
  converter could emit an ordinary GGUF our engine opens unchanged.
- **PTQTP's dual plane does not.** Two trit-planes plus two scale sets is not any
  ggml type. It needs a new type upstream in ingot or a side channel — a real
  cost on the PTQTP side of the ledger that the paper does not carry.

`TQ1_0` packs five base-3 digits per byte (a multiply-high plus a fixed-point
divide per five weights). A ternary format optimised for bytes is not
automatically one optimised for a CPU kernel; that tension is Phase 7 work.

## 6. The twelve questions

| # | question | status |
|---|---|---|
| 1 | Can Qwen3-0.6B tolerate single-plane W1.58 at useful coverage? | `[NOT MEASURED]` — Phase 3a |
| 2 | How much does PT²/TWLA improve over naive? | `[NOT MEASURED]` — Phase 2/4 |
| 3 | How much better is PTQTP 2×1.58? | `[PUBLISHED]` +82% ppl on this model; ours `[NOT MEASURED]` |
| 4 | Which families/blocks are most sensitive? | `[NOT MEASURED]` — Phase 3 |
| 5 | Is MLP more attractive than attention? | MLP is 60% of ternarizable weight `[MEASURED]`; sensitivity `[NOT MEASURED]` |
| 6 | What fraction can be ternarized safely? | `[NOT MEASURED]` — Phase 4 |
| 7 | Best mixed-precision configuration? | `[NOT MEASURED]` — Phase 5 |
| 8 | Real complete-model effective bit budget? | **`[MEASURED]`** — §3; 2.963-6.674 bpw depending on scheme, never 1.58 |
| 9 | How does it compare with ordinary Q4? | **`[MEASURED]` on storage** — §3; on quality `[NOT MEASURED]` |
| 10 | Worth a dedicated CPU ternary backend? | **Storage says no** `[MEASURED]`; arithmetic `[NOT MEASURED]` |
| 11 | Which shapes and ISA paths first? | `[NOT MEASURED]` — Phase 7, gated on a GO |
| 12 | Worth repeating at 4B/7B? | **This is the live question** — see below |

## 7. Where this is heading

A pre-registered prediction, recorded before Phase 1 so it can be wrong in
public, is in the work note. In short: this model is expected to fail, the
storage win against `IQ2_XXS`/`IQ1_S` is between −3% and +30%, and the result
worth having is the *scaling* — ternary's perplexity cost falls monotonically
with model size (82% → 94% → 34% → 16% across 0.6B/1.7B/4B/32B) while the
embedding's share falls with it (26.10% at 0.6B against ~9.7% at 4B).

**Qwen3-0.6B is close to the worst possible case for ternarization**, and the
brief's question 12 may turn out to be the only one with a GO answer.

The study continues to Phase 6 regardless, because the control that would make
that story look stupid has not been run: if `IQ2_XXS` and `IQ1_S` are *also*
destroyed on this checkpoint, the finding is about 0.6B models and not about
ternary at all.

---

## Reproducing

```bash
hf download Qwen/Qwen3-0.6B --local-dir models-local/qwen3-0.6b-bf16
python3 tools/qwen_ternary_feasibility.py \
    --model models-local/qwen3-0.6b-bf16 \
    --revision c1899de289a04d12100db370d81485cdf75e47ca \
    --output-dir reports/ternary census
python3 tools/qwen_ternary_feasibility.py ... budget
```

Subcommands for phases that have not been written exit 2 with `REFUSED:` rather
than print a placeholder.
