# Post-training ternarization of Qwen3-0.6B — feasibility

**Status: Phases 0, 7a and 7b complete. Quality phases (1-6) not measured.**
Everything below is labelled `[MEASURED]`, `[PUBLISHED]` (someone else's number,
cited) or `[NOT MEASURED]`. Nothing is estimated and presented as a result.

> **Verdict on the backend question: REJECT for v0.1/v0.2.** Not because
> ternarization fails — that is still unmeasured here — but because its premise
> does not hold on this engine. Batch-1 decode of Qwen3-0.6B is **ALU-bound, not
> weight-bandwidth-bound**: it scales 2.95x on four cores at 14.8 GB/s where this
> machine is known to sustain 32-38, and `--fast` buys **+48%** by quantizing
> activations without changing one weight byte. Cutting weight bytes is not what
> makes this decode faster. The study continues as a *quality* investigation.

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

## 6. Decode traffic and GEMV shapes  `[MEASURED]`

`python3 tools/qwen_ternary_feasibility.py traffic`. KV cache in bf16 per the
repo's quantization policy.

In a dense decoder at batch 1, **every weight is read exactly once per token and
feeds exactly one MAC**. So weight bytes per token *is* the model file, and
`% of MACs` equals `% of weights` for every region — there is no cache-resident
region doing the arithmetic off few bytes.

### Weight bytes per decoded token

| scheme | MiB/token | q/k/v/o + gate/up/down | tied lm_head | vs Q4_K_M | AI (FLOP/B) |
|---|---|---|---|---|---|
| BF16 | 1137.0 | 73.9% | 26.1% | 0.33x | 1.00 |
| INT8 / Q8_0 | 604.1 | 73.9% | 26.1% | 0.62x | 1.88 |
| **Q4_K_M — shipped** | **372.7** | **67.3%** | **32.7%** | **1.00x** | **3.05** |
| Q3_K_M | 302.4 | 59.7% | 40.2% | 1.23x | 3.76 |
| **PTQTP 2x1.58** | 299.2 | 59.2% | 40.7% | **1.25x** | 3.80 |
| IQ2_XXS | 230.2 | 47.0% | 52.9% | 1.62x | 4.94 |
| **W1.58 single plane** | 210.6 | 42.1% | 57.8% | **1.77x** | 5.40 |
| IQ1_S | 204.0 | 40.2% | 59.7% | 1.83x | 5.57 |
| *linears at zero bits (bound)* | *122.0* | *0%* | *99.8%* | *3.06x* | *9.32* |

**The projections' share of traffic falls as you ternarize them** — 67.3% at Q4,
42.1% at W1.58 — because the tied `lm_head` does not move. Amdahl's law on bytes
caps the whole exercise at **3.06x** even with free linear weights.

### The KV cache takes most of what is left

`2 x 8 kv-heads x 128 head-dim x 28 layers x 2 B` = **114,688 B (112 KiB) per
context position**, read in full every decode step. No weight-side scheme touches
it. End-to-end decode traffic against the shipped `Q4_K_M`:

| scheme | L=128 | L=512 | L=1024 | L=2048 | L=4096 |
|---|---|---|---|---|---|
| INT8 / Q8_0 | 0.63x | 0.65x | 0.68x | 0.72x | 0.78x |
| **PTQTP 2x1.58** | **1.23x** | 1.21x | 1.18x | 1.14x | **1.10x** |
| IQ2_XXS | 1.58x | 1.50x | 1.42x | 1.31x | 1.21x |
| **W1.58 single plane** | **1.72x** | 1.61x | 1.50x | 1.37x | **1.25x** |
| **IQ1_S** | **1.77x** | 1.65x | 1.53x | 1.39x | **1.26x** |
| *linears at zero bits* | *2.84x* | *2.41x* | *2.07x* | *1.72x* | *1.44x* |

**PTQTP — the mandatory method — buys 10-23% of decode traffic against the file
we ship, for a published +82% perplexity on this model.** `IQ1_S`, which ingot
already decodes, beats every ternary scheme at every context length.

### Dominant GEMV shapes at batch 1 (M=1)

| N x K | instances | params | % MACs | % bytes @Q4 | % bytes @W1.58 | family |
|---|---|---|---|---|---|---|
| **3072 x 1024** | 56 | 176,160,768 | **29.56** | 26.91 | 16.83 | gate_proj, up_proj |
| **151936 x 1024** | 1 | 155,582,464 | **26.11** | **32.66** | **57.81** | tied lm_head |
| 1024 x 3072 | 28 | 88,080,384 | 14.78 | 13.45 | 8.42 | down_proj |
| 1024 x 1024 | 56 | 58,720,256 | 9.85 | 8.97 | 5.61 | k_proj, v_proj |
| 1024 x 2048 | 28 | 58,720,256 | 9.85 | 8.97 | 5.61 | o_proj |
| 2048 x 1024 | 28 | 58,720,256 | 9.85 | 8.97 | 5.61 | q_proj |

- **Only five distinct linear shapes**, all with K in {1024, 2048, 3072} and all
  a multiple of 256. A ternary GEMV backend would need five specialisations plus
  the head, not a general GEMM — unusually friendly for a kernel project.
- **The tied head becomes the dominant tensor the moment the linears shrink**:
  32.7% of bytes at Q4, **57.8% at W1.58**. A ternary backend that did not also
  solve the 151936 x 1024 head would spend most of its traffic budget on the one
  matrix no method in the brief quantizes.

### Prefill is a different problem

Prefill over B tokens amortises the weights, so arithmetic intensity is
approximately `3.05 x B` FLOP/byte at Q4 — about 97 at B=32. **Prefill is
compute-bound and the weight format is nearly irrelevant to it.** The entire
ternary case is a batch-1 decode case.

## 7. What the PocketTTS study transfers  `[SIBLING-MEASURED]`

`mynah-tts` closed the same question against PocketTTS
(`../mynah-tts/docs/ternary-feasibility.md`). Four findings transfer because they
are CPU or algebra facts rather than PocketTTS facts:

- **No target ISA has a sub-byte multiply-accumulate.** NEON `SDOT`/`SMMLA`,
  AVX-512 VNNI `VPDPBUSD` and AMX are all int8-lane, and our own `src/qmat.c`
  already widens Q4 nibbles to int8 before the dot. **Ternary's MACs per
  instruction equal int8's, so its entire CPU case is weight traffic.**
- **PT²'s AGA and TWLA's E2M-ATQ stage 2 are the same estimator** — both freeze
  `T` and solve the same per-row 2x2 system under `S = XᵀX`. The sibling
  implemented both and measured them **within 0.3% on every layer group**. We
  implement one.
- **GPTQ-style error compensation dominates**: 1.3-108x against naive (median
  ~13x), where activation-aware grid fitting bought 1.1-8.9x. *"Ternary without
  error compensation is not competitive with anything."* Naive W1.58 is therefore
  a control, not a candidate.
- **KOTMS beats AGA and still loses to GPTQ by 2-6x**, and is the only method
  leaving a permanent runtime activation rotation (`R = R₁ ⊗ R₂` on every call).
  Optional, and its cost is reported apart from its quality.

**Where Qwen3 genuinely differs.** PocketTTS failed on a MAC/byte mismatch: 75.4%
of MACs in the Mimi codec against 79.4% of the ternary bytes in the AR backbone,
and the backbone was the part that failed quality. A dense decoder at batch 1
cannot have that mismatch, so **unlike PocketTTS, ternary here does attack the
region that owns the traffic.** It simply does not attack it by much.

## 8. Phase 7b — is decode actually traffic-bound?  `[MEASURED]`

The pre-registered reading was fixed before the run: *flat from 2 threads and
>=30 GB/s -> traffic-bound, ternary is live; scales with cores and <<30 GB/s ->
not traffic-bound, cutting weight bytes cannot pay.*

`mynah-slm v0.0.1-3-g2e52761` from a **clean committed tree**, `Q4_K_M` staged
local, `-n 128 --think off --no-stream --seed 1 --temp 0` — deterministic, so
every run does byte-for-byte identical work. Two warm-ups discarded, five
measured runs per point, one process at a time. M1, 4 P + 4 E cores.

| threads | decode tok/s | achieved GB/s | vs 1 thread | spread | `--fast` tok/s | gain |
|---|---|---|---|---|---|---|
| 1 | 12.8 | 5.00 | 1.00x | 0.8% | 18.9 | **+48%** |
| 2 | 22.2 | 8.68 | 1.73x | 3.2% | 31.6 | +42% |
| **4** | **37.8** | **14.77** | **2.95x** | 3.7% | **43.5** | +15% |
| 8 | 37.1 | 14.50 | 2.90x | 1.9% | *not measured* | — |

The 8-thread `--fast` point scattered 22.6-45.4 tok/s (**60.5% spread**) and is
therefore **not reported as a measurement**. Every other point held within 4%.

**It scales 2.95x on four cores and tops out at 14.8 GB/s.** The 4-to-8 plateau
is the performance-core count, not a DRAM ceiling: the same machine was
demonstrated to sustain **32-38 GB/s** on the PocketTTS backbone, a region that
scaled only **1.15x** on eight cores. Near-linear scaling with P-cores at 39-46%
of demonstrated achievable bandwidth are the two signatures of a compute-bound
region, and the exact opposite of a memory wall.

**The decisive control is `--fast`**, which quantizes activations to int8 and
changes no weight byte at all: **+48% at one thread**. A step whose time were set
by streaming weights from DRAM cannot gain 48% from arithmetic on the other
operand.

### It agrees with two measurements already in `docs/perf.md`

- **Q8_0 decodes 54.8 tok/s against Q4_K_M's 42.2 while reading 1.6x the bytes.**
  The recorded conclusion there: *"It is ALU-bound […] the formats were never the
  ranking; the kernels were."*
- **The tied LM head is 42% of a decode step on its own** (42.55 ms of a
  single-threaded step, `151936 x 1024`). It is also 32.7% of decode bytes at Q4
  and **57.8%** under W1.58 — and no method in the brief quantizes it.

So the chain `ternary -> 1.77x fewer weight bytes -> 1.77x faster decode` breaks
at the second arrow, and Phase 7a already showed the first arrow was worth only
1.10-1.23x end-to-end for PTQTP.

## 9. The twelve questions

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
| 10 | Worth a dedicated CPU ternary backend? | **No, for v0.1/v0.2** `[MEASURED]` — storage says no, and decode is ALU-bound so the traffic saving has nothing to convert into |
| 11 | Which shapes and ISA paths first? | Shapes `[MEASURED]` — five distinct linear shapes plus the head, all K in {1024, 2048, 3072}. ISA selection moot under the current verdict |
| 12 | Worth repeating at 4B/7B? | **This is the live question** — see below |

## 10. Where this leaves the study

**The backend question is closed for v0.1/v0.2, and it closed on the premise
rather than on the quality.** Five independent lines agree, three measured here
and two inherited from measurements taken for other reasons:

| evidence | source |
|---|---|
| decode scales 2.95x on 4 cores at 14.8 GB/s, of a demonstrated 32-38 | Phase 7b |
| `--fast` buys +48% at 1 thread with zero weight-byte change | Phase 7b |
| Q8_0 beats Q4_K_M 54.8 vs 42.2 tok/s **reading 1.6x the bytes** | `docs/perf.md` |
| the tied LM head is **42%** of a decode step, and no method quantizes it | `docs/perf.md` |
| PTQTP buys **1.10-1.23x** of traffic end-to-end against the shipped file | Phase 7a |

Even a ternary scheme with *zero* quality loss would be optimising a term that is
not the bottleneck, for at most 1.25-1.72x of traffic, on 58% of the bytes and
58% of the time. No CPU kernel is justified by this data.

### What the study becomes instead

1. **A quality study.** The pre-registered prediction — Qwen3-0.6B fails
   ternarization at useful coverage — is still unmeasured and still worth
   measuring, because it feeds the `IQ2`/`IQ3` decision the quantization policy
   already wants.
2. **With the controls as the main event.** `IQ2_XXS` (230.2 MiB) and `IQ1_S`
   (204.0 MiB) are already decodable by ingot, already smaller than PTQTP, and
   their quality on this checkpoint is unknown. If they are *also* destroyed
   here, the finding is about 0.6B models and not about ternary at all — and that
   is the control that would make the ternary story look stupid if it were wrong.
3. **Having named two real optimisation targets, neither of them ternary**: the
   tied `lm_head` at `151936 x 1024` and 42% of a decode step, and the activation
   path that `--fast` proves has 48% in it at one thread. Both are ordinary
   engineering in `src/`.

### What is NOT concluded

Every number here is one 0.6B checkpoint, one engine, one M1, `Q4_K_M` weights.
At 4B the embedding falls from 26.10% of parameters to ~9.7% and PTQTP's
published perplexity cost falls from +82% to +34%; on a many-core server with
lower per-core bandwidth the roofline moves the other way. **Nothing here
forecloses ternary at 4B on a server**, and the brief's question 12 remains the
one worth keeping open.

---

## Reproducing

```bash
hf download Qwen/Qwen3-0.6B --local-dir models-local/qwen3-0.6b-bf16
REV=c1899de289a04d12100db370d81485cdf75e47ca
T="python3 tools/qwen_ternary_feasibility.py --model models-local/qwen3-0.6b-bf16 --revision $REV"
$T census      # Phase 0 — tensor census, tie check
$T budget      # Phase 0 — storage under every representation
$T traffic     # Phase 7a — bytes/token, KV cache, GEMV shapes
```

Phase 7b, from a clean committed tree, weights staged local, one process at a
time, five runs per point:

```bash
make clean && make
for t in 1 2 4 8; do
  for i in 1 2 3 4 5; do
    ./mynah-slm run -m models-local/Qwen3-0.6B-Q4_K_M.gguf \
        -p "Write a short paragraph about the sea." \
        -n 128 --think off --no-stream --seed 1 --temp 0 -t $t 2>&1 >/dev/null
  done
done
```

Subcommands for phases that have not been written exit 2 with `REFUSED:` rather
than print a placeholder.
