# mynah-slm Development Plan

> **Project Goal**
>
> `mynah-slm` is a lightweight, CPU-first, standalone Small Language Model (SLM)
> inference engine written in C11, following the same philosophy as
> [`mynah-asr`](https://github.com/mynah-org/mynah-asr) and
> [`mynah-tts`](https://github.com/mynah-org/mynah-tts).
>
> It is **not** an orchestration framework and **not** a general LLM server.
> Its purpose is to provide fast, efficient, local text inference for:
>
> - text generation
> - summarization
> - text cleanup
> - multilingual conversations
> - structured output
> - function/tool calling
> - lightweight agent workflows
>
> It should work independently, while being composable with:
>
> ```
> mynah-asr        audio → text
>      ↓
> mynah-slm        text → text / tool calls / summary
>      ↓
> mynah-tts        text → audio
> ```
>
> without requiring any integration between the three projects.

---

## How this plan is organised

`PLAN.md` is a **board**: one line per work item, with a link to the note under
[`.work/`](.work/) that carries the detail. Detail does not belong here — when a
task is picked up, read its note. Session logs, in-flight measurements and
post-mortems go in `.work/`; durable measured results go in `docs/models.md` and
`docs/perf.md`.

Same convention as `mynah-tts`, so the sibling repos are read the same way. See
[`.work/README.md`](.work/README.md). `PLAN.md` and `.work/` are **tracked**;
`CLAUDE.md` is a symlink to `AGENTS.md` and is the only planning document kept
out of git.

The sections after the board are the durable contract: identity, language
policy, design principles, repository shape, weight loading, quantization
policy, model strategy, prior art, features, API, roadmap. They change only when
a decision changes.

---

## 0. Work board

Legend: `[ ]` open · `[~]` in progress · `[x]` done · `[-]` dropped

### Reference — read before starting anything

- [x] The method: cost model before code, every tool declares a refusal, a
      benchmark is invalid until dispatch is proven, the completion rule →
      [`.work/engineering-method.md`](.work/engineering-method.md)
- [x] Repo contracts, NAS policy, quantization policy, the Qwen3 and Gemma 4
      trap lists → [`AGENTS.md`](AGENTS.md)
- [x] The M0-M6 task breakdown, moved verbatim from the old `TASKS.md`; the
      `[ ]` lines there are still the honest backlog for those milestones →
      [`.work/archive-2026-08-tasks.md`](.work/archive-2026-08-tasks.md)
- [x] Measured quantization ladder, per-tensor census, tool-call and
      multilingual scores → `docs/models.md`; kernel A/Bs → `docs/perf.md`

### v0.1 — Qwen3-0.6B, in progress

Generation, streaming, chat, tools and thinking run end to end. What remains is
tracked in the archive note, per milestone: M2 engine gaps, M2c timing
instrumentation, M3 CLI polish, M5b quantization & footprint, M6 release.

- [~] M0-M6 → [`.work/archive-2026-08-tasks.md`](.work/archive-2026-08-tasks.md)

### R — research items

- [~] **R1** — can a pretrained Qwen3-0.6B be post-training ternarized and still
      be worth running? Two independent gates. **GATE A (model quality):
      UNMEASURED.** **GATE B (CPU backend): UNDECIDED** — awaiting Gate A and
      target-ISA evidence; this Mac cannot close it
      → [`.work/ternary-feasibility.md`](.work/ternary-feasibility.md)
- [~] **R1-A** — claim audit: every load-bearing conclusion in R1, with its
      evidence class and what would falsify it. Five claims changed status
      → [`.work/r1-claim-audit.md`](.work/r1-claim-audit.md)
- [x] **R1-P** — reading PTQTP properly: what "2 x 1.58-bit" physically is, and
      what the authors' released Qwen3-0.6B artifact actually contains
      → [`.work/ptqtp-paper-reading.md`](.work/ptqtp-paper-reading.md)
- [ ] **R2** — the tied `lm_head`, `151936 x 1024`, is **42% of a decode step**
      and 32.7% of decode bytes. Fell out of R1; needs its own note before it is
      picked up
- [ ] **R3** — the activation path: `--fast` buys **+48%** at one thread by
      quantizing activations, changing no weight byte. Fell out of R1; needs its
      own note before it is picked up

---

# Project Identity

| | |
|---|---|
| **Org** | [github.com/mynah-org](https://github.com/mynah-org) |
| **Repo** | `mynah-org/mynah-slm` (new repository) |
| **License** | **MIT** — same as the rest of the Mynah ecosystem |
| **Language** | C11 (public header also parses as C++) |
| **Build** | Plain `Makefile` (no CMake, no configure, no package manager) |
| **Runtime deps** | `libc`, `-lpthread`, `-lm`. Optional BLAS. Nothing else. |
| **Vendored** | `third_party/ingot` (git subtree) — GGUF + safetensors reader |
| **Siblings** | `mynah-asr`, `mynah-tts`, `ingot` — same org, same rules, no cross-dependency |

---

# Language Policy — the repo is English, all of it

**Everything written into this repository, or published from it, is in English.
No exceptions.** The conversation that produces the work can be in Italian; what
lands in git cannot. Concretely, this covers:

- **Code comments** — every `/* */` and `//` in `src/`, `cli/`, `server/`,
  `tests/`, `include/`, down to a one-word note next to a magic constant.
- **Identifiers and strings** — function, variable, struct and file names;
  log lines, error messages, CLI `--help` text, server JSON error bodies.
- **Markdown** — `README.md`, everything under `docs/`, `AGENTS.md`, `PLAN.md`
  and every note under `.work/`. All of those are tracked. `CLAUDE.md` is a
  symlink to `AGENTS.md`, so it is the same English file under a second name.
- **Build and tooling** — `Makefile` comments, `scripts/*.sh`, `tools/**` Python
  (docstrings, comments, `argparse` help), CI workflow files.
- **Git and GitHub** — commit subjects and bodies, tag annotations, branch
  descriptions, PR titles and bodies, issue text, release notes.

Two deliberate exceptions, both narrow. The comment *explaining* an exception is
still English — only the payload is not.

1. **Multilingual test and eval data**, where the non-English text *is* the input
   under test, and where changing it would silently invalidate a committed
   golden: `tools/eval/lang_ppl.py`, `tools/eval/tool_calls.py`, the
   mixed-language prompt in `tests/test_batch.c`, the Italian prompts in
   `tests/test_server.sh` / `test_think.c` / `test_tools.c`, and the whole
   parity chain — `tests/fixtures/tokenizer.txt` (the token ids ARE the
   fixture), `tests/fixtures/chat_tools_granite.txt`, `tests/golden/it_hello/`,
   the `PROMPT ?=` default in the Makefile that regenerates it, and the
   `gen_*_fixture.py` generators that must keep matching it.
2. **Verbatim transcripts of measured runs** in `README.md` and `docs/perf.md`.
   A pasted `[load … | decode … tok/s]` line is a record of a command that was
   actually executed: rewriting its prompt would make the `prompt N tok` count
   next to it false. These become English by **re-measuring** with an English
   prompt on a quiet machine, never by editing the text in place.

Why it is a hard rule and not a preference: this is a public MIT repo in an org
(`mynah-org`) whose other projects are English, the audience for an inference
engine is international, and a half-translated codebase is worse than either
extreme — the reader never knows which half to trust. Grep for a term and get
one of two vocabularies back and the search stops being reliable.

---

# Design Principles

- **CPU-first.** The CPU path is the reference implementation and is always
  correct and always fast enough to ship. GPU backends, if they ever land, are
  accelerators — never the only way to run.
- **Custom C engine.** No ggml, no llama.cpp, no ONNX Runtime, no PyTorch. The
  tensor code is ours, written for this class of models, readable end to end.
- **Zero / few dependencies.** `ingot` for weight containers, an optional BLAS,
  nothing at runtime. Python exists only as offline tooling (conversion,
  oracle, eval) and is never required to run inference.
- **One weight loader for both worlds.** GGUF (community quants, llama.cpp
  ecosystem) and safetensors (upstream HF checkpoints) via `ingot`, converging
  on a single in-memory representation right after load.
- **Streaming-first.** Token-by-token output is the primary API, not an add-on.
- **Small memory footprint**, small binary, fast startup — optimized for edge
  devices and for being one stage of a three-stage speech pipeline.
- **Architecture-independent public API.** No model-specific symbols exposed.
- **Config-driven.** No architecture constants in `#define`s. Layer counts,
  head dims, sliding-window sizes, RoPE thetas, vocab sizes all come from the
  model's metadata (GGUF KV or `config.json`).
- **Portable.** macOS (Apple Silicon + Intel) and Linux (x86-64 + aarch64) are
  first-class. NEON / AVX2 / AVX-512 kernels selected at runtime, never at
  configure time.

---

# Repository Structure

```
mynah-slm/

include/mynah_slm.h        public C API (the only header users include)
src/                       engine: one module = one cohesive .c + .h
    model.c/.h             architecture-agnostic model container
    arch_qwen3.c           Qwen3 dense decoder
    arch_gemma4.c          Gemma 4 text decoder
    weights.c/.h           ingot → engine tensor mapping
    tokenizer.c/.h         BPE / SentencePiece over GGUF or tokenizer.json
    template.c/.h          chat templates + thinking control
    tools.c/.h             tool-call schema injection + parsing
    sampler.c/.h           greedy / top-k / top-p / min-p / repetition
    grammar.c/.h           GBNF-lite constrained decoding (JSON mode)
    kvcache.c/.h           paged KV cache, sliding-window aware
    kernels.c/.h           quantized + dense matmul, RMSNorm, RoPE, softmax
    threads.c/.h           thread pool
    session.c/.h           conversation state machine
cli/main.c                 mynah-slm
server/                    mynah-slm-server (OpenAI-compatible)
tools/                     Python (uv) — converter, oracle, eval. Offline only.
reference/<model>/         configs/tokenizers extracted from checkpoints
models/                    downloaded weights (GITIGNORED)
tests/
docs/
third_party/ingot/         git subtree
Makefile
README.md
CLAUDE.md
LICENSE                    MIT
```

---

# Weight Loading — ingot

`ingot` is the Mynah weight-container library and is used here exactly as in
`mynah-asr`: vendored as a **git subtree** under `third_party/ingot`, built by
its own Makefile, linked as a static archive.

```make
INGOT_DIR := third_party/ingot
INGOT_LIB := $(INGOT_DIR)/libingot.a
CFLAGS    += -I$(INGOT_DIR)/include
LDFLAGS   += $(INGOT_LIB)
```

What we get for free, and therefore must **not** reimplement:

- GGUF v2/v3 with typed metadata, split files, zero-copy `mmap`
- safetensors: single file, sharded dir, `index.json`
- all 33 ggml quantized block types decoding, verified against llama.cpp
- SIMD matvec/matmat for `Q2_K Q3_K Q4_K Q5_K Q6_K Q8_0` plus dense BF16/F16,
  with NEON / AVX2 / SDOT / SMMLA / AVX-512-VNNI paths chosen at runtime
- BF16/F16→F32 bulk conversion, page-cache control, O(1) tensor lookup

`mynah-slm` therefore owns: attention, KV cache, RoPE, norms, sampling,
tokenizer, templates, tool calls, sessions, server. Not dequantization, not
container parsing, and not the hot GEMV for the K-quants.

Where a kernel is genuinely missing from `ingot` and is hot for us, the fix
goes **upstream into ingot**, not into a private copy here.

## Coverage check (2026-08-06)

**Reading is a solved problem.** Every ggml block type has a geometry in ingot
and all but one decode bit-for-bit against llama.cpp. Note that `Q4_K_M` is a
*recipe*, not a type — such a file is `Q4_K` + `Q6_K` + `F32` inside, all
supported. Same for `UD-Q4_K_XL` and the other community variants.

The Gemma 4 QAT checkpoint we want as the default is plain **`Q4_0` (id 2):
decode ✅, encode ✅**.

**The one real gap is speed, not support.** `Q4_0`'s kernel in ingot is
*generic* — decode a row, then multiply. The SIMD kernels exist only for
`Q8_0`, `Q2_K`, `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K`. Since our default production
model *is* `Q4_0`, it would run on the slow path.

**Fix, and it is nearly free:** `qwen-tts` already has a NEON-optimized
`qwen_matvec_q4_0` and a batched `qwen_matmat_q4_0` using the *identical*
18-byte block layout (fp16 scale + 16 nibble pairs — its own comment says "same
layout as llama.cpp q4_0"). Port it **upstream into ingot**. That is the
canonical example of the rule above.

The only type ingot cannot decode is `Q1_0` (id 41) — llama.cpp's own reference
package has no decoder either, so nothing could validate one. It still opens
and names the type. Not a practical problem.

---

# Quantization Policy — small is the point

This is a **mini engine**. The default must be tiny, not maximal quality.

- **Default: GGUF Q4.** `Q4_K_M` where a good community quant exists; **`Q4_0`
  for Gemma 4**, because that is what Google's QAT checkpoint ships and
  QAT-at-Q4_0 beats post-training Q4_K at the same file size.
- **Go below 4 bits wherever quality holds.** ingot already decodes `Q3_K`
  (3.44 bpw), `Q2_K` (2.63 — and it has a **SIMD** kernel), the full `IQ1`–`IQ4`
  codebook family (`IQ3_XXS` 3.06, `IQ2_S` 2.56, `IQ2_XXS` 2.06, `IQ1_S` 1.56)
  and ternary `TQ1_0`/`TQ2_0` (1.69 / 2.06). Sub-4-bit is a **measured**
  decision per model, gated on a multilingual + tool-call regression suite —
  never on vibes, and never silently.
- **Mixed precision by role**, not one global type: only the large linears are
  quantized; norms, biases and embeddings stay f32 (the `mynah-asr` `qmat.h`
  policy). Gemma's per-layer embedding tables are lookups, not GEMM operands —
  they follow the embedding rule.
- **KV cache in bf16**, not f32 (qwen-tts precedent: halves KV memory at no
  measurable quality cost). This matters more than weight bits on long
  conversations.
- Every quant claim in the README must carry the measured RAM, tokens/s and
  regression score that backs it.

**Q4 is the baseline everywhere**, not just on the small model: Qwen3-0.6B /
1.7B / 4B at `Q4_K_M`, Gemma 4 E2B-it at `Q4_0` (QAT), and Qwen3.5 at Q4 too
whenever v0.4 gets there. Bigger model, same bit budget — that is the whole
point of a mini engine. Sub-4-bit is then a per-model question on top.

Order of magnitude we are aiming at: **Qwen3-0.6B around 400 MB at Q4_K_M,
Gemma 4 E2B-it QAT around 1.5 GB at Q4_0** — small enough that the model, the
engine and the KV cache all fit in the page cache of a laptop that is also
running an ASR and a TTS engine.

## The embedding tables are the real size budget

Measured on the actual checkpoint (`docs/models.md`): **Qwen3-0.6B at Q4_K_M is
5.24 bits/weight, not 4.5.** Twenty-nine tensors held at Q6_K are 45% of the
file. A "4-bit" build of a small model with a 151936-entry vocabulary is
nowhere near 4 bits.

For Gemma 4 E2B this gets more extreme, and it follows from the config before
we measure anything. With `vocab_size_per_layer_input: 262144`,
`hidden_size_per_layer_input: 256` and 35 layers, the Per-Layer Embedding
tables alone are

```
262144 × 256 × 35  ≈  2.35 B parameters
```

on top of a `262144 × 1536` ≈ 0.4 B main embedding. That is ~2.75 B of the
5.1 B total — and it is exactly the gap between "5.1 B total" and "2.3 B
effective" on the model card. **The majority of that checkpoint is lookup
tables, not GEMM operands.**

**But "big table → quantize it hard" is only half right, and the half that is
wrong is expensive.** The Qwen3 tensor list (`docs/models.md`) settles it: the
29 Q6_K tensors are `token_embd.weight` at 121.7 MiB — a third of the whole
file — plus `attn_v` and `ffn_down` on 14 of the 28 layers. And there is **no
`output.weight`**: `tie_word_embeddings` is true, so that one tensor is the
input lookup *and* the LM head.

So the two families behave in opposite ways:

1. **Gemma's PLE tables are looked up, never multiplied.** No kernel, no
   accumulation error — only bytes and a dequant on read. Cheapest possible
   place to be aggressive, and the largest. First thing to measure in M5b.
2. **Qwen3's tied `token_embd` is the LM head.** It is multiplied against all
   151936 rows on every decode step, it is the hottest GEMV in the loop, and it
   sits directly in front of the softmax that picks the token. That is exactly
   why llama.cpp holds it at Q6_K while pushing everything else to Q4_K.
   Quantizing it like a lookup table would trade a third of the file against
   output quality at the worst possible place.

The inherited `mynah-asr` rule ("embeddings stay f32") is wrong for both, but
for different reasons, and the replacement is not one rule: it is
**per-tensor by role — indexed, multiplied, or both** — decided per model and
backed by a measurement.

Still to check against the real file: whether the shipped Gemma QAT GGUF
already quantizes its PLE tables or leaves them wide, and whether Gemma's
embedding is tied the way Qwen3's is (`text_config` says
`tie_word_embeddings: true`, so probably — in which case Gemma has *both*
problems at once).

---

# Model Strategy

Selection criteria, in order: **multilingual** → **tool calling** →
**thinking on/off** → **small enough for edge CPU** → **permissive license
(Apache 2.0 / MIT)** → implementation cost in a hand-written C engine.

## Tier 1 — engine validation baseline

**Qwen3-0.6B / Qwen3-1.7B** (Apache 2.0)

The smallest genuinely useful multilingual instruct models with both hybrid
thinking and tool calling. Plain dense decoder — the cheapest possible way to
get the whole pipeline (loader → tokenizer → attention → KV → sampler →
template → tools → SSE) correct end to end.

Verified `Qwen3-0.6B` config:

| | |
|---|---|
| layers | 28 |
| hidden_size | 1024 |
| heads / kv_heads | 16 / 8 (GQA) |
| head_dim | 128 (note: `head_dim * heads != hidden_size`) |
| intermediate_size | 3072 |
| vocab_size | 151936, `tie_word_embeddings: true` |
| rope_theta | 1e6, no scaling, no sliding window |
| norm | RMSNorm eps 1e-6, **QK-RMSNorm per head** |
| act | SiLU (SwiGLU MLP) |
| ctx | 32768 (40960 max_position_embeddings) |
| langs | 100+ |
| thinking | `/think` and `/no_think` soft switches + `enable_thinking` |
| tools | yes (Qwen-Agent style, Hermes-ish `<tool_call>` JSON) |

`Qwen3-4B` is the same architecture scaled up and is the "good quality, still
small" option for people with RAM to spare — supported for free once 0.6B works.

## Tier 2 — primary production model

**Gemma 4 E2B-it**, default variant **`gemma-4-E2B-it-qat-q4_0-gguf`**
(Apache 2.0, released 2026-07-02)

2.3B effective / 5.1B total parameters thanks to Per-Layer Embeddings, 35+
languages out of the box (140+ pre-trained), native tool calling, built-in
reasoning mode, 128K context. Google ships an official QAT Q4_0 GGUF, so the
recommended deployment path needs no conversion step at all.

We implement the **text tower only**. The vision and audio towers in the
checkpoint are ignored — `mynah-asr` is the ASR stage of this pipeline, and
staying text-only keeps the engine small and the scope honest.

Verified `text_config` (this is the real complexity budget of v0.2):

| | |
|---|---|
| layers | 35 |
| hidden_size | 1536 |
| heads / kv_heads | 8 / **1 (MQA)** |
| head_dim | 256; `global_head_dim` 512 |
| intermediate_size | 6144, `use_double_wide_mlp: true` |
| activation | `gelu_pytorch_tanh` |
| vocab_size | 262144, `tie_word_embeddings: true` |
| layer_types | repeating **4× sliding + 1× full**, `sliding_window: 512` |
| `num_kv_shared_layers` | **20** — KV cache shared across the last 20 layers |
| PLE | `hidden_size_per_layer_input: 256`, `vocab_size_per_layer_input: 262144` |
| RoPE | sliding: theta 1e4 default. full: theta 1e6, **`rope_type: proportional`, `partial_rotary_factor: 0.25`** |
| logits | `final_logit_softcapping: 30.0` |
| ctx | 131072 |
| eos | `[1, 106]` (two terminators) |
| thinking | `<|think|>` at the start of the system prompt |

**Known traps** (to be written into `docs/gemma4-arch.md` before any code):
partial rotary on the global layers only; two different RoPE thetas in the same
model; per-layer embedding lookup added into each decoder layer's input; KV
sharing meaning fewer physical caches than layers; final logit softcapping;
MQA with a single KV head; double-wide MLP; the 4:1 sliding/global pattern
driving both mask construction and cache sizing.

## Tier 3 — future / optional

- **Qwen3.5-0.8B / 2B / 4B** (Apache 2.0, 201 languages, 262K ctx). Strictly
  better models, but a **hybrid Gated DeltaNet (linear attention) + gated
  full attention + sparse MoE** architecture. That is a new kernel family
  (recurrent delta-rule state) plus expert routing — a whole milestone of its
  own, not a variant of Qwen3. Deliberately deferred to v0.4.
- **LFM2.5-2.6B** — excellent on-device agentic model (GQA + short conv
  blocks), but the LFM Open License is not MIT/Apache; an opt-in third-party
  target, never a default. **Promoted to a trial candidate on 2026-08-08** —
  full survey below, tasks in TASKS.md M6b.
- **SmolLM3-3B** (Apache 2.0, dual reasoning mode, 128K) — plain architecture,
  cheap to add, but only 6 European languages.

### IBM Granite 4.0 (surveyed 2026-08-07)

Apache 2.0, 12 languages **including Italian**, and — the part that matters
here — tool calling in the **exact format we already implement**: an OpenAI
function schema in, `<tool_call>{"name": .., "arguments": {..}}</tool_call>`
out. `src/tools.c` would parse a Granite call today, unchanged. The family
splits in two, and the split is the whole decision:

| | params | layers | verdict |
|---|---|---|---|
| `granite-4.0-micro` | 3B **dense** | 40 attention | **portable now**, but not small |
| `granite-4.0-h-350m` | 340M **hybrid** | 4 attention + 28 **Mamba-2** | tiny and interesting, new kernel family |
| `granite-4.0-h-tiny` | 7B total / 1B active, hybrid **MoE** | 4 attention + 36 Mamba-2, 64 experts (6 active) | both new families at once |

**`granite-4.0-micro` is the only one this engine could run soon.** GQA, RoPE,
SwiGLU, RMSNorm, tied embeddings — the Qwen3 forward pass minus QK-norm, so
`arch_qwen3.c` generalizes rather than gets rewritten. 128K context, 2560
hidden, 40 heads / 8 KV heads. The cost is size, not complexity: at Q4 it is
~1.9 GB against Qwen3-0.6B's 0.4 GB, and decode is memory-bandwidth bound, so
expect roughly a fifth of the 27 tok/s we measure today. That is a *quality*
option for a laptop, not the small-and-fast default this project is about.

**The `h-` models are where the appeal is, and they are a v0.4 question.**
340M with 32K context, and Mamba-2 decode carries a fixed-size state instead of
a KV cache that grows with the conversation — on a CPU that is the right shape.
But 28 of its 32 layers are Mamba-2: selective scan, causal conv1d, per-head
state. That is a new kernel family, the same size of job as the Gated DeltaNet
study above — and the two overlap enough that doing one makes the other
cheaper. llama.cpp only got `GraniteMoeHybrid` conversion recently, and ships a
"use dense Micro instead" note for environments where Mamba-2 is not covered;
we would be in exactly that position. ingot reads the GGUF either way — the gap
is the forward pass, never the container.

### MEASURED, 2026-08-07: `granite-4.0-350m` beats Qwen3-0.6B at tool calling

The dense 350m is not hypothetical any more — the engine runs it, and it is
better at the thing this project exists for.

| | Qwen3-0.6B Q4_K_M | granite-4.0-350m Q8_0 | granite Q4_K_M |
|---|---|---|---|
| **tool-call decisions** | 25/30 (needs `--think on`) | **27/30, no thinking** | 25/30 |
| JSON validity | 30/30 | 30/30 | 30/30 |
| perplexity, 55-token English | 100.5 | 121.7 | 161.2 |
| file | 378 MB | 361 MB | **226 MB** |
| decode | 24.7 tok/s | 27.9 | **37.1** |
| prefill | 229 tok/s | 292 | **306** |
| KV per position | 1024 f32 | **256** | **256** |
| vocabulary (= LM head rows) | 151936 | **100352** | **100352** |
| languages | 100+ | 12 (Italian included) | 12 |

**Q8_0 is the one to use.** At Q4_K_M the tool-call score drops from 27 to 25
and perplexity from 122 to 161 — a 350M model has far less room for
quantization noise than a 600M one, which is the answer to the question the
first measurement raised. `Q4_K_M` is still the fastest thing here by a
distance if footprint matters more than the last two cases.

The remaining three misses are all the same shape and the OPPOSITE of Qwen3's:
Granite calls a tool for a chat turn (eager) where Qwen3 answers in words
instead of calling (reluctant). Eager is the easier one to fix from the prompt.

**Getting here took correcting a wrong verdict, and the correction is the
lesson.** The first measurement scored Granite 9/30 and concluded "not good
enough". That was OUR BUG: llama.cpp's converter permutes q and k for
Llama-family checkpoints, so a Granite GGUF wants INTERLEAVED RoPE where Qwen3
wants NeoX split-half. Both variants run and produce fluent English — the same
trap this repo's own trap list warns about for Qwen3, in the opposite
direction. Perplexity on one 55-token passage: **412 wrong, 122 right**, with
coherent text either way.

What found it was refusing to accept a bad number: an oracle sweep of the four
muP scalars showed all four were right (removing any one cost orders of
magnitude), which left RoPE as the only untested assumption. Note also that
the oracle agreeing with the C proved they matched EACH OTHER, not that either
was right — both were mine, and both were wrong the same way. An oracle is
only independent of the implementation, never of the author's misreading.

**Verdict: a serious candidate for the v0.1 default**, and the strongest reason
is not the score but the shape: 12 languages including Italian, tool calling in
the format we already parse, Apache 2.0, half the KV cache and a third less LM
head. What it gives up against Qwen3 is breadth of languages (12 against 100+)
and the thinking mode.

#### Superseded: the first Granite measurement (kept for the lesson)

This is the run that scored 9/30, before the interleaved-RoPE fix above. Read it
as the record of a wrong verdict, not as a result.

| | Qwen3-0.6B Q4_K_M | granite-4.0-350m Q4_K_M |
|---|---|---|
| tool-call decisions | **25/30** | **9/30** |
| JSON validity | 30/30 | 30/30 |
| file at Q4_K_M | 378 MB | 226 MB |
| prefill | 202 tok/s | 353 tok/s |
| KV cache per position | 1024 f32 | **256** |
| vocabulary (= LM head rows) | 151936 | 100352 |

**The engine numbers are all better and the model is much worse.** 21 of the 30
misses are the same failure: it answers in words instead of calling. Through
its chat template it degenerates badly — with the canned system turn it emits
the single token "assistant" and stops — while raw completions are fine ("The
capital of Italy is" -> " Rome").

That was not taken on faith. The tokenizer produces ids identical to HF on the
exact prompt (36 for 36), the template is byte-identical to
`apply_chat_template`, and the numpy ORACLE — extended to Granite for this —
produces the SAME degenerate output token for token. Two independent
implementations agreeing on a bad answer means the answer belongs to the model.

**Verdict: not a replacement for Qwen3-0.6B at v0.1.** Worth one more cheap
test before closing it — Q8_0 is 375 MB and a 350M model has much less room for
quantization noise than a 600M one, so `Q4_K_M` may be doing real damage here.
The support stays in either way: it cost four config fields and a string table,
and it is what makes the engine multi-family rather than a Qwen3 runner.

### LiquidAI LFM2.5-2.6B (surveyed 2026-08-08, nothing measured yet)

Granite closed on 2026-08-07 — measured, ranked, written up — so the candidate
list gets its next entry. This one is the interesting kind: the model is built
for exactly what this project is (on-device, agentic, CPU), and the reason it
was parked before was licensing, not architecture.

**Verified from `config.json`, not the card:**

| | |
|---|---|
| params | 2.69B, `Lfm2ForCausalLM` |
| layers | **30 — 22 `conv` + 8 `full_attention`**, from `layer_types` (read the array) |
| hidden / intermediate | 2048 / 10752, SwiGLU (`block_use_swiglu`) |
| heads / kv_heads | 32 / **8 (GQA)**, on the 8 attention layers only |
| conv | `conv_dim` 2048, **`conv_L_cache: 3`**, `conv_bias: false` |
| vocab | 128000, `tie_word_embeddings: true` |
| RoPE | theta **1e7**, `rope_type: default`, `use_pos_enc: true` |
| ctx | 131072 |
| norm | RMSNorm, eps 1e-5 |
| languages | 16 **declared**, Italian included — see below |
| GGUF | official `LiquidAI/LFM2.5-2.6B-GGUF`: Q4_0 1.59 GB, **Q4_K_M 1.67 GB**, Q5_K_M 1.94, Q6_K 2.22, **Q8_0 2.87** |

**The short conv block is the whole engineering question, and it is small.**
22 of 30 layers are "double-gated short convolution": a projection to (B, C, x),
a **depthwise causal FIR of length 3** over B·x, and an output projection gated
by C. That is a fixed 3-tap filter with a 3-slot per-channel state — no
selective scan, no data-dependent state transition, none of Mamba-2's
machinery. Decode carries a **constant 2048×3 state per conv layer instead of a
growing KV cache**, and only 8 of 30 layers keep a KV cache at all. On a CPU
that is the right shape, and it is a far cheaper way into the recurrent-state
kernel family than the Mamba-2 / Gated DeltaNet study parked in M10.

**Three things stand between it and a verdict, in order of cost:**

1. **Tool calls are Pythonic, ours are JSON.** The model emits
   `<|tool_call_start|>[func(arg=1)]<|tool_call_end|>` — a Python call
   expression, not `<tool_call>{"name":..}</tool_call>`. `src/tools.c` parses
   nothing of that today. The card says a system-prompt instruction switches it
   to JSON; **measure both** — the native format is what the model was trained
   on, and the JSON detour may cost exactly the accuracy we are shopping for.
2. **Size.** 1.67 GB at Q4_K_M is 4.4× the Qwen3-0.6B default we ship. Our
   decode is ALU-bound rather than bandwidth-bound at this scale
   (docs/perf.md), so expect single-digit tok/s on the M1 before any work —
   against 24.7 for Qwen3-0.6B. Liquid's own numbers (220 tok/s on an M5 Max,
   30 on a phone) are a different machine class and a different engine; they
   are not a prediction for ours.
3. **License: LFM Open License v1.0.** Commercial use only under **$10M annual
   revenue**, notices must be preserved, modified files must say so. That is
   not MIT/Apache and it never becomes the shipped default or a bundled
   download — opt-in target, license stated at the point of download.

**Which languages, and which generation.** Read out of the GGUF itself
(`general.languages`, 2026-08-08), LFM2.5-2.6B declares **16**: `ar zh en fr de
hi id it ja ko pl pt ru es th vi`. The widely-quoted **8** — en, ar, zh, fr, de,
ja, ko, es — belong to the **previous** generation, LFM2-2.6B, and are an exact
subset of these 16; LFM2.5 adds hi, id, **it**, pl, **pt**, ru, th, vi. Liquid
also says the family is tuned hardest for English and Japanese. Do not mix the
two lists up, and do not mistake either for evidence: these are **declared**
languages. `tools/eval/lang_ppl.py` is what turns a claim into a number, and it
needs the C engine to run at all.

**What would make it worth the kernel:** Italian on the declared list (the
axis where Granite lost to Qwen3), ToolSandbox 77.83 / BFCLv4 56.88 on the
card, a `<think>` mode, and 128K context. What would kill it: single-digit
decode, or a multilingual bits/byte that does not beat Qwen3-0.6B by enough to
pay for 4.4× the file. Both are cheap to find out — `inspect` and
`tools/eval/lang_ppl.py` run before a single line of forward pass is written.

---

# Prior Art — port, don't invent

Full survey in **[docs/prior-art.md](docs/prior-art.md)** (2026-08-06). Summary:
most of the v0.1 engine already exists, MIT, in our own projects.

**`../qwen-tts` is the jackpot.** Qwen3-TTS's "Talker" *is* a Qwen3 decoder,
already implemented, already validated, already loading through `ingot`:

- `qwen_tts_talker.c` (1462 lines) — GQA, **per-head Q/K RMSNorm**, **NeoX
  split-half RoPE (not interleaved)**, SwiGLU, bf16 KV cache with geometric
  growth, per-layer activation dump hook.
- `qwen_tts_kernels.c` (5129 lines) — RMSNorm (plain / fused-residual /
  per-head), matvec **and batched matmat** for bf16 / int8 / q4_0 / q2_0 with
  NEON + AVX2 + AVX-512-VNNI, fused QKV matvec, causal GQA attention **including
  a sliding-window variant** (Gemma 4 needs exactly that) and a bf16-KV variant,
  fused SwiGLU, **fused argmax-matvec that never materializes the logits** (a
  real win against Gemma's 262144-row head), f16↔f32 with a bit-exact portable
  fallback, plus `ftz_on`, runtime-ISA check, capability report, **kernel
  self-test** and a matmat microbench.
- `qwen_tts_tokenizer.c` (986 lines) — byte-level BPE over the Qwen vocab.
- `qwen_tts_thread.c`, `qwen_tts_server.c`, `qwen_tts_sampling.c`,
  `qwen_tts_backend.c`, and the ingot-subtree Makefile wiring.

**`../keyra`** ships `src/core/gemma3.c` — a Gemma text encoder whose header
comment is a pre-paid debugging session for v0.2: RMSNorm scales by
`(1 + weight)`; embeddings × `sqrt(hidden)`; four norms per layer; two RoPE
bases in one model with the scaling applied to the **global layers only**;
GeGLU-tanh not SwiGLU; non-square projections. Also `linenoise.c` (MIT line
editing → `mynah-slm chat` gets history and Ctrl-R for free), `cpu_caps.c`,
`allocator.c`, `gguf_quant.c`.

**`../mynah-asr`** gives the repo skeleton: Makefile with BLAS auto-detect and
the `make lib/test/bench/debug/ubsan/leaks` target set, `threads.c` with a
bit-identical-to-serial `parallel_for` **and a BLAS concurrency budget** (a
server running 4+ concurrent inferences otherwise collapses on the OpenBLAS
lock), `qmat.c`'s quantized-product dispatch policy (f32 → sgemm; quantized +
small T → direct dot; quantized + large T → dequant + sgemm), `backend.c`'s
CPU/Metal/CUDA fallback pattern, the OpenAI-compatible `server/`, and the CI
workflow set.

**`../mynah-tts`** confirms the module naming. Its explicit `graph.c` we
deliberately skip — a decoder loop is simpler written straight.

External, read but not vendored:

- [`adriancable/qwen3.c`](https://github.com/adriancable/qwen3.c) — **MIT**,
  ~1000 lines, no dependencies, Q8_0, OpenMP, `-r 1` thinking toggle. The
  reference for "how small can a correct Qwen3 forward pass be". We take the
  shape of the loop and the numerical checklist; not its Python-export-only
  weight path, not its single-file structure.
- `llama.cpp` — GGUF metadata key conventions and chat-template quirks only.

**Rule:** if a kernel is generic enough for any weight-container consumer, it
goes **upstream into `ingot`**, not into a fourth private copy.

---

# Main Features

## CLI

```bash
# one-shot
mynah-slm -m models/qwen3-0.6b-q4_k_m.gguf -p "Summarize this text: ..."

# stdin, pipeline-friendly (this is how ASR feeds it)
mynah-asr transcribe -m ... -i meeting.wav | mynah-slm summarize -m ...

# interactive chat, streaming by default
mynah-slm chat -m models/gemma-4-E2B-it-qat-q4_0.gguf

# thinking control
mynah-slm chat -m ... --think off|low|on

# tool calling: schemas in, tool calls out. Execution is the caller's job.
mynah-slm -m ... --tools tools.json -p "What is the weather in Verona?"

# structured output
mynah-slm -m ... --json-schema out.schema.json -p "extract the name and the date"
```

## HTTP server

OpenAI-compatible, same server skeleton as `mynah-asr`:

```
GET  /health
GET  /v1/models
POST /v1/chat/completions      (+ stream, + tools, + response_format)
POST /v1/completions
POST /v1/responses
POST /v1/tokenize
```

Streaming through **SSE** (primary), optional NDJSON, optional WebSocket.
`stream: true` must emit the first token before the full response exists —
that is the whole point of the ASR→SLM→TTS pipeline: TTS starts speaking the
first sentence while the SLM is still writing the second.

---

# Speed Is Always Reported

Three delivery modes, and **every one of them reports its own speed**. Not
behind a `--verbose`, not only in `make bench`: a run that does not tell you
how fast it was is a run you cannot compare, and this engine's whole premise is
that it is fast enough on a CPU.

The numbers, and why each one:

| metric | why it is the one that matters |
|---|---|
| **load** (s) | dominated by SMB vs local; must be stated or benchmarks lie |
| **TTFT** (ms) | time to first token — what the TTS stage downstream actually waits on |
| **prefill** (tok/s) | throughput over the prompt; scales with batch, unlike decode |
| **decode** (tok/s) | **the headline number** — memory-bandwidth bound, and what a user feels |
| tokens in / out | so a t/s figure can be checked rather than trusted |

`mynah-asr` sets the precedent with its one-line summary
(`[5.2s audio | load 0.04s | inference 0.48s | RTF 0.092]`). Ours:

```
$ mynah-slm -m models/Qwen3-0.6B-Q4_K_M.gguf -p "..."
...answer...
[load 0.21s | prompt 42 tok, prefill 310 tok/s | gen 128 tok, decode 47.3 tok/s | TTFT 138 ms | 8 threads]
```

Rules:

- **CLI**: the summary goes to **stderr**, always, so `mynah-slm ... | mynah-tts`
  still gets clean text on stdout. `--quiet` suppresses it; nothing else does.
- **Streaming CLI**: the same line after the last token, with TTFT measured at
  the *first* token rather than reconstructed at the end.
- **Server**: timings ride in the response — `usage` extended with
  `prefill_tok_s`, `decode_tok_s`, `ttft_ms`, `load_ms`. On an SSE stream they
  arrive in the final event, so a client never has to time the socket itself.
  `GET /health` reports the last N requests' decode t/s, which is how a
  regression gets noticed in production instead of in a benchmark.
- Every number is measured on the **warm** path unless labelled cold, and
  benchmarks state quant, thread count, machine, and whether the weights were on
  the NAS or local.

---

# Public API

```c
mynah_slm_model   *mynah_slm_load(const char *path, const mynah_slm_params *p, char *err, size_t errsz);
mynah_slm_session *mynah_slm_session_new(mynah_slm_model *m, const mynah_slm_session_params *sp);

int  mynah_slm_add_message(mynah_slm_session *s, mynah_slm_role role, const char *content);
int  mynah_slm_set_tools(mynah_slm_session *s, const char *tools_json);
int  mynah_slm_set_thinking(mynah_slm_session *s, mynah_slm_think mode);   /* off | low | on */

/* streaming: callback returns 0 to continue, non-zero to stop */
int  mynah_slm_generate(mynah_slm_session *s, mynah_slm_token_cb cb, void *user);

const mynah_slm_tool_call *mynah_slm_pending_tool_calls(mynah_slm_session *s, size_t *n);
int  mynah_slm_add_tool_result(mynah_slm_session *s, const char *id, const char *result_json);

void mynah_slm_session_free(mynah_slm_session *s);
void mynah_slm_free(mynah_slm_model *m);
```

No model-specific symbols. Errors are returned as strings, never printed.
Nothing writes to `stderr` from the library.

---

# Tool Calling Philosophy

The engine **generates** tool calls. It **never executes** them.
Execution always belongs to the application.

Modes:

```
off       no tool tokens injected, no parsing
emit      schemas injected into the prompt, calls parsed and returned (default)
strict    as emit, but decoding is grammar-constrained to the tool schema
```

Per-family formats (Qwen `<tool_call>{json}</tool_call>` vs Gemma's native
format) are handled in `template.c` and normalized to one struct before
crossing the public API.

**Built 2026-08-07 for Qwen3, minus `strict`.** Two things the implementation
settled that the plan had not:

- The call travels on its **own channel**, split by token id in `generate.c`
  exactly like reasoning — not filtered out of the answer afterwards. Same
  argument as thinking: a client that ignores `tool_calls` must not find the
  JSON in `content`, because the next stage speaks it.
- **`strict` is worth less than we assumed.** Measured over 60 turns on
  Qwen3-0.6B, every single one produced parseable JSON and no invented
  function name. The model's failure is the *decision* — answering from its own
  parameters instead of calling — and a grammar cannot fix that. `--think on`
  can, and does: 21/30 → 26/30. Grammar stays on the list for
  `response_format`/JSON mode, where a caller needs a guarantee and not a rate.
  Numbers in [docs/tools.md](docs/tools.md).

---

# Thinking Mode

```
off     thinking suppressed (Qwen: /no_think, Gemma: no <|think|>)
low     thinking allowed but budget-capped, tokens hidden from the callback
on      thinking emitted and surfaced separately from the answer
```

Thinking tokens are **always** delivered on a distinct channel so a TTS stage
downstream never speaks the model's reasoning aloud.

---

# Runtime Priorities

1. Low latency (time to first token)
2. Small RAM usage
3. Small binary size
4. Fast startup
5. Good CPU throughput

Not maximum throughput. Not batch serving.

---

# Development Roadmap

## v0.1 — Engine validation

**Target: Qwen3-0.6B, then Qwen3-1.7B / 4B for free.**

GGUF + safetensors load via ingot · BPE tokenizer · dense decoder (RMSNorm,
QK-norm, GQA, RoPE, SwiGLU) · KV cache · sampling · chat template · CLI ·
HTTP server · SSE streaming · session management · tool-call emit/parse ·
JSON-schema constrained decoding · Python oracle parity per stage.

Exit criterion: byte-identical greedy output vs the numpy oracle on a fixed
prompt set, in 5+ languages, with and without thinking.

## v0.2 — Primary production model

**Target: Gemma 4 E2B-it (QAT Q4_0 GGUF as the default download).**

Gemma 4 text architecture (PLE, MQA, 4:1 sliding/global, dual RoPE, partial
rotary, KV sharing, logit softcapping) · SentencePiece 262K vocab · Gemma chat
template · native tool calling · thinking off/low/on · summarization presets.

This is the model the README recommends.

## v0.3 — Performance

Prompt cache + shared prefix cache · sliding-window KV compaction ·
speculative decoding (Qwen3-0.6B drafting for a larger Qwen3; Gemma draft
model when available) · faster startup · memory reuse · ARM/AVX kernel tuning ·
optional Metal, optional CUDA · optional continuous batching.

## v0.4 — Next-gen architectures

Recurrent-state models, which is one kernel family and two candidates:
Qwen3.5's Gated DeltaNet + sparse MoE, and **Granite 4.0 `h-*`'s Mamba-2**
(340M at 32K context, tool calling in the format we already parse). A scan
kernel written for one is most of the other; a fixed-size decode state instead
of a growing KV cache is the reason either is worth the milestone on a CPU.
Evaluated on their own merits at that point — and gated by
`tools/eval/tool_calls.py` on the real checkpoint, not by a model card.

---

# Ecosystem Role

`mynah-slm` is the middle stage of a fully MIT-licensed, fully local,
fully offline voice stack:

```
mic → mynah-asr → mynah-slm → mynah-tts → speaker
        (C)         (C)         (C)
```

Each stage: standalone binary, standalone C API, standalone HTTP server, no
Python at runtime, no cloud, no telemetry. Composed by the application
(e.g. `mynah-app`), never by the engines themselves.

Concretely, the SLM stage is what turns a transcript into: a cleaned-up text,
a summary, an intent, a JSON payload, a tool call, or a spoken reply — and it
must stream, because the TTS stage downstream is waiting on its first sentence.

---

# Explicit Non Goals

Not a generic inference framework. Not a llama.cpp replacement. Not an
orchestration or agent framework. Not a workflow engine. Not a vector
database. Not an embedding server. Not multimodal (that is what the sibling
projects are for).

The scope stays intentionally narrow.

---

# Long-Term Vision

A family of specialized, standalone, MIT inference engines:

```
mynah-asr    audio → text
mynah-slm    text  → text / tool calls / structured output
mynah-tts    text  → audio
ingot        weights → memory (shared foundation)
```

Each usable independently, each with a CLI, an HTTP server and a C API, each
sharing design principles and nothing else.
