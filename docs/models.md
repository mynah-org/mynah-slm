# Models

Every number here is measured with `mynah-slm inspect`, not copied from a model
card.

```sh
scripts/download_model.sh --list
scripts/download_model.sh --model qwen3-0.6b-q4
./mynah-slm inspect models/Qwen3-0.6B-Q4_K_M.gguf
```

---

## Qwen3-0.6B — v0.1 baseline

Apache 2.0. Dense decoder, GQA + per-head QK-RMSNorm, NeoX split-half RoPE,
SwiGLU. 100+ languages, `/think` `/no_think`, tool calling. Verified config:
28 layers, hidden 1024, 16 Q heads / 8 KV heads, head_dim 128 (so
`heads × head_dim ≠ hidden` — the projections are not square), intermediate
3072, vocab 151936 with tied embeddings, rope_theta 1e6, no sliding window.

Both builds carry **310 tensors** and report `general.architecture = qwen3`,
GGUF v3.

### `Qwen3-0.6B-Q4_K_M.gguf` — the default we ship

unsloth build, 378 MB on disk.

| type | tensors | size |
|---|---|---|
| Q4_K | 168 | 204.8 MiB |
| Q6_K | 29 | 167.7 MiB |
| F32 | 113 | 256.0 KiB |
| **total** | **310** | **372.7 MiB — 5.24 bits/weight** |

**The headline number is 5.24 bits/weight, not 4.5.** This is the single most
useful thing the census tells us, and no filename would have: 29 tensors held
at Q6_K account for **45% of the file**. A "4-bit" checkpoint of this model is
nowhere near 4 bits, because a 0.6B model with a 151936-entry vocabulary is
dominated by things that are not 4-bit.

**Which 29 tensors** (`inspect --tensors Q6_K`, answered 2026-08-06 — the
"28 layers + 1" guess was wrong):

| what | count | size |
|---|---|---|
| `token_embd.weight` `[151936 x 1024]` | 1 | **121.7 MiB — 32.7% of the file** |
| `blk.N.ffn_down.weight` `[1024 x 3072]` | 14 | 2.5 MiB each |
| `blk.N.attn_v.weight` `[1024 x 1024]` | 14 | 840 KiB each |

The layer indices are 0, 1, 2, 5, 8, 11, 14, 17, 20, 23, 24, 25, 26, 27 — the
llama.cpp `Q4_K_M` heuristic, which bumps `attn_v` and `ffn_down` on the first
and last few layers plus every third one in between. Fourteen of twenty-eight.

Everything else is `Q4_K`; the 113 F32 tensors are the norms, including
`attn_q_norm` / `attn_k_norm` at `[128]` = head_dim, which is the per-head
QK-RMSNorm confirmed on the real file rather than read off a config.

### The embedding is the LM head — do not quantize it like a lookup table

There is **no `output.weight` tensor** in this checkpoint. `tie_word_embeddings`
is true, so `token_embd.weight` is used twice: as the input lookup *and* as the
output projection.

That matters, and it corrects an assumption worth stating plainly because the
opposite is true for Gemma. An embedding table that is only ever *indexed* costs
no kernel and no accumulation error, so it is cheap to quantize hard. A **tied**
embedding is not that: it is multiplied against all 151936 rows on every single
decode step. It is simultaneously

- the largest single tensor (a third of the file), and
- the hottest GEMV in the decode loop, and
- directly in front of the softmax that picks the token.

Which is exactly why llama.cpp leaves it at Q6_K while pushing the rest to Q4_K,
and why a naive "quantize the big tables harder" pass would trade a third of the
file size against output quality at the worst possible place. Measure this one
per model; do not generalize it from Gemma's PLE tables, which really are
lookup-only.

### `Qwen3-0.6B-Q8_0.gguf` — the parity build

Official Qwen build, 610 MB on disk. Use this one for oracle parity work in
M1/M2: least quantization noise, so a numeric mismatch is our bug and not the
quantizer's.

| type | tensors | size |
|---|---|---|
| Q8_0 | 197 | 603.9 MiB |
| F32 | 113 | 256.0 KiB |
| **total** | **310** | **604.1 MiB — 8.50 bits/weight** |

Flat: one quantized type, exactly 8.50 bits/weight, which is Q8_0's block rate
(34 bytes per 32 weights). The 113 F32 tensors are the norms — 256 KiB total,
i.e. free. That the F32 count is *identical* across both builds is a good sign:
the two quantizers disagree about weights, not about what stays wide.

Note the metadata key count differs (28 vs 32): the unsloth build carries extra
keys. Do not assume a fixed key set when extracting config.

---

## Granite 4.0 350m — the light alternative, measured against Qwen3

Apache 2.0. Dense decoder despite the `granitemoehybrid` config class
(`layer_types` is 28× "attention", `num_local_experts` 0): GQA, **interleaved**
RoPE, SwiGLU, RMSNorm, tied embeddings, **no QK-norm**, and four muP scalars
that are the difference between fluent text and garbage. 28 layers, hidden
1024, 16 Q heads / **4 KV heads**, head_dim 64, intermediate 2048, vocab
**100352**, rope_theta 1e7, 32K context. 12 languages.

Everything below is one machine, one afternoon, best of three, weights staged
locally. Ratios travel; absolutes do not (see docs/perf.md).

### Speed and footprint

| | Qwen3-0.6B Q4_K_M | granite-350m **Q8_0** | granite-350m Q4_K_M |
|---|---|---|---|
| file | 378 MB | 361 MB | **226 MB** |
| decode, short context | 32.2 tok/s | **55.8** | 39.5 |
| decode at 2.3K context | 22.3 tok/s | **39.7** | 36.0 |
| prefill at 2.3K context | 223 tok/s | **301** | 298 |
| TTFT, tool-calling turn (~200 tok) | 532 ms | 397 ms | **392 ms** |
| KV cache per position | 1024 floats | **256** | **256** |
| vocabulary (= LM head rows) | 151936 | **100352** | **100352** |

Re-measured after the Q8_0 kernel landed in ingot (see the ladder below and
docs/perf.md). **Granite at Q8_0 stopped being the slow-but-good option and
became the fast one**, which is the rare case of a kernel fix changing a model
recommendation rather than a benchmark table.

The KV row is the one that matters on a laptop: four times smaller per
position, so a 32K-token conversation costs a quarter of the memory before any
quantization of the cache is applied on top.

### Tool calling — Granite wins, and it wins cheaply

30 prompts across 7 languages, scored on the DECISION (call / don't call), the
function name, and the arguments (`tools/eval/tool_calls.py`):

| | score | how |
|---|---|---|
| Qwen3-0.6B Q4_K_M | 25/30 | **needs `--think on`** — 21/30 without it |
| **granite-350m Q8_0** | **27/30** | no thinking mode at all |
| granite-350m Q4_K_M | 25/30 | no thinking mode at all |

JSON validity 30/30 for all three. Granite's advantage is bigger than the score
suggests: Qwen3 needs to emit a reasoning block first, which is ~100 extra
tokens of latency per turn, and Granite does not have one.

The failure modes are OPPOSITE, which is worth knowing before choosing:

- **Qwen3 is reluctant** — it answers in words instead of calling (9 of 9
  misses without thinking, 3 of 5 with).
- **Granite is eager** — all 3 misses are calling a tool for a chat turn.

Eager is the easier one to correct from a system prompt.

Format-wise Granite needs nothing from us: `<tool_call>` and `</tool_call>` are
single special tokens in its vocabulary (100270/100271) exactly as in Qwen3
(151657/151658), so the channel split that keeps call JSON out of the answer
works unchanged — markers are resolved by NAME, never by id. `tests/test_server.sh`
passes end to end against Granite: `finish_reason`, arguments, the second turn
with the tool result, SSE, and "the call JSON stays out of content".

### Languages — Qwen3 is better, including Italian

Same paragraph in 16 languages, scored in **bits per byte**
(`tools/eval/lang_ppl.py`). Perplexity would not be comparable here: it is per
TOKEN and the two vocabularies disagree about what a token is.

**Read across a row, never down a column** — Cyrillic and CJK spend more bytes
per character, which moves the denominator for reasons unrelated to quality.

| | granite Q8_0 | Qwen3 Q4_K_M | | | granite Q8_0 | Qwen3 Q4_K_M |
|---|---|---|---|---|---|---|
| en | **1.416** | 1.424 | | ko | 1.877 | **1.617** |
| de | 1.711 | **1.676** | | nl | 1.973 | **1.769** |
| es | 1.565 | **1.318** | | zh | 2.020 | **1.672** |
| fr | 1.577 | **1.410** | | ru \* | 1.214 | **0.996** |
| ja | 1.633 | **1.575** | | pl \* | 2.194 | **1.973** |
| pt | 1.714 | **1.411** | | tr \* | 2.343 | **2.205** |
| ar | 1.383 | **1.319** | | el \* | 1.552 | **1.312** |
| cs | **2.214** | 2.221 | | **mean** | 1.780 | **1.597** |
| **it** | 2.094 | **1.653** | | | | |

\* not on Granite's claimed list. Note it is not catastrophic there either —
which says the 12-language claim is about what IBM validated, not a hard edge.

**Qwen3 is better on 14 of 16, and the gap on Italian is the largest of all
(2.094 against 1.653, 27%).** That is the fact that decides this for a pipeline
whose first language is Italian. Note also that this compares Granite at Q8_0
against Qwen3 at Q4_K_M — Qwen3 at Q8_0 would widen the gap further.

### The quantization ladder — is there a middle ground?

The Q4-versus-Q8 gap above invites an obvious question: does Q5_K_M or Q6_K buy
the quality without the size? All four are the SAME build
(`ibm-granite/granite-4.0-350m-GGUF`), which is the precondition for the
comparison meaning anything. Interleaved, **mean of three** rounds, staged
locally, M1, 8 threads. (The speed table further up reports best-of-three, so
its Q4_K_M reads 49.3 where this one reads 47.4 — same runs, different
statistic. Means are used here because the question is a ranking across four
close candidates, where one lucky round would decide it.)

| | Q4_K_M | Q5_K_M | Q6_K | Q8_0 |
|---|---|---|---|---|
| file | **226 MB** | 252 MB | 279 MB | 361 MB |
| bits/weight | 5.30 | 5.91 | 6.57 | 8.50 |
| decode, short | 39.5 tok/s | 38.4 | 39.5 | **55.8** |
| decode at 2.3K | 36.0 tok/s | 31.6 | 33.6 | **39.7** |
| prefill at 2.3K | 298.0 tok/s | 296.1 | 287.0 | **301.1** |
| bits/byte, mean of 16 langs | 1.8928 | 1.8094 | 1.7936 | **1.7800** |
| tool calls | 25/30 | 23/30 | 22/30 | **27/30** |

**Two of these speed rows moved after kernel work in ingot, and the ladder
changed shape twice.** Q8_0 read 34.0 here and was the SLOWEST of the four:
ingot had no vector matvec for it on either architecture and was round-tripping
every block through a scratch array. Q5_K_M read 34.4 and was 25% behind its
neighbours: its NEON kernel materialized each weight instead of distributing
the sum, and x86 had no kernel at all. Both fixed upstream (docs/perf.md).

Absolutes drift — Q4_K_M reads 39.5 in this round against 47.4 in the first,
same binary, warmer machine — so only compare WITHIN a round. Within this one:

- **Q8_0 +64%**, from last place to first by a wide margin.
- **Q5_K_M +12%**, from 25% behind its neighbours to level with them.

So the ladder no longer has a speed-versus-quality trade at all. **Q8_0 is best
on every measured axis** — lowest bits/byte, best tool score, fastest decode —
and costs only 135 MB of file. That follows from what the kernel work
established rather than contradicting it: this decode is ALU-bound, not
bandwidth-bound, so the format that decodes with ONE multiply beats the ones
that reassemble weights from bit planes, even reading 1.6x the bytes.

**Prefill barely moves** — 286 to 304 tok/s across a ladder that spans 60% more
bits. That is the expected shape, not an anomaly: prefill dequantizes a row
strip and hands it to `sgemm`, so it is compute-bound and the weight format
only changes the strip-filling cost. Decode reads the whole weight set per
token and is bandwidth-bound, which is why the same ladder spreads it 34 to 47.

**Two things here disagree, and the disagreement is the finding.**

Perplexity is monotone and behaves exactly as theory says: every extra bit
helps, with sharply diminishing returns. Q4→Q5 closes 74% of the whole Q4→Q8
quality gap; Q5→Q6 another 14%; Q6→Q8 the last 12%. By that instrument **Q6_K
is within 0.8% of Q8_0** and is a genuine middle ground.

The tool-call score is NOT monotone: 25, 23, 22, 27. A score cannot legitimately
fall and then rise as precision increases, so at least part of that ordering is
noise. It is: 30 binary decisions give a 95% interval of roughly **±4.5 cases**,
which makes Q4, Q5 and Q6 statistically indistinguishable from one another.
Only the Q8_0 lead (5 cases over Q6_K) reaches the edge of what this suite can
resolve, and "at the edge" is not "established". **The honest reading is that
the eval has 30 cases and cannot rank the middle of this ladder** — reported
here rather than smoothed over, because picking Q6_K over Q4_K_M on a 22-vs-25
that is within noise would be picking on nothing.

What the failure MODES show is more useful than the totals: Q4_K_M errs by
staying silent (4 answered instead of calling), while Q5_K_M and Q6_K err by
over-calling (6 spurious calls each, on 15 negative cases). More bits made the
model more eager, not more accurate — and eagerness is the correctable one.

**Q5_K_M used to be the one to skip, and that is no longer true** — kept here
because it is the clearest case on this page of a *kernel* masquerading as a
*format*. It read 34.4 against Q6_K's 42.2 while being 27 MB SMALLER, and
`make bench` localized it: on the identical `ffn_gate` shape Q5_K moved 13.9
G elem/s against Q6_K's 16.3. Not a bits effect — Q5_K's NEON kernel was the
only K-quant still building each weight before using it (multiply by d*scale,
subtract dmin*min, then the multiply-add) instead of distributing the sum the
way Q4_K does. Rewritten upstream, it runs at **17.5 G elem/s** and the model
went 34.4 -> 38.4 tok/s, level with its neighbours. A format that looked
dominated was a kernel nobody had measured.

The same measurement explains why Q4_K_M is so far ahead of everything —
**and it is not only the bits.** Q4_K is the one type we have written a matvec
for, and it beats ingot's by 1.3–1.7x on every layer tensor. Q5_K, Q6_K and
Q8_0 have no kernel of ours to call, so they land within noise of ingot
(0.8–1.2x, scattered either side of 1.0 with no consistent win). Q4_K_M's speed
is a kernel we own as much as a format we chose.

That locates where the remaining time sits, with a number on it: in a Q4_K_M
decode step, **46% of the matvec time is spent on Q6_K tensors** (`attn_v`,
`ffn_down`, and the tied `lm_head`) running ingot's kernel rather than ours.

**That has since been measured, and it ended somewhere unexpected.** We wrote
the Q6_K matvec twice (restructured f32, then int8 SDOT); both tie ingot's NEON
kernel to within 2%, and cutting the instruction count by ~1.5x moved nothing.
On x86 the same kernel was worth **4.65x**, because ingot had no AVX2 Q6_K at
all — so it went **upstream into ingot**, which is where a kernel with no
per-call specialization belongs. ingot's version then measured *faster* than
ours, and our copy was deleted. One implementation, validated against
llama.cpp by ingot's own suite. Full arc in docs/perf.md.

### The verdict, and it is a split one

| use it for | model |
|---|---|
| tool calling and agent loops | **granite-350m Q8_0** — better decisions, no thinking tax, smaller KV, and since the kernel fix the fastest of the four to decode |
| multilingual chat and summarization | **Qwen3-0.6B** — better on 14 of 16 languages, and by 27% on Italian |
| the smallest thing that still works | **granite-350m Q4_K_M** — 226 MB, 47 tok/s, 25/30 on tools |
| the smallest file that is still good | **granite-350m Q6_K** — within 0.8% of Q8_0 on perplexity for 82 MB less. No longer the *fast* middle ground: Q8_0 now decodes faster than it. Q5_K_M is a legitimate point again too, 27 MB smaller at the same speed |

**Qwen3-0.6B stays the v0.1 default**, because the ASR→SLM→TTS pipeline this
engine exists for is multilingual first and Italian in particular, and that is
exactly where Granite is weakest. Granite is documented as the light/agentic
alternative rather than the default — and if the priority ever inverts toward
footprint, the measurement to re-run is this page.

Granite's cost sits in the quantization choice: the 27/30 was measured at Q8_0,
361 MB, which is no longer light. At Q4_K_M it is 226 MB and 25/30 — the same
score as Qwen3, at 60% of the size and 1.5x the speed, but with the language
gap intact.

**An earlier version of this page called that a "cliff", then said the choice
was 135 MB against 28% of the decode rate. Both are now wrong**, and they are
kept here because of how they went wrong. The cliff was an artifact of reading
a 30-case tool eval as though it could resolve 2 cases — perplexity across
Q4/Q5/Q6/Q8 is a smooth curve. The decode-rate half was true when written and
was killed by a kernel: Q8_0 had no vector matvec anywhere in ingot, and once
it did, Q8_0 became the fastest of the four rather than the slowest.

What survives is stronger than either: **Q8_0 wins on every measured axis
except file size.** The remaining question is only whether 135 MB matters for a
given deployment, and that is a question a benchmark cannot answer.

### Traps, for whoever ports the next family

- **`granite-4.0-h-350m` is a DIFFERENT MODEL, not a variant name.** The `-h-`
  repo is the Mamba-2 hybrid: arch `granitehybrid`, 32 blocks, SSM layers,
  `head_count_kv` genuinely varying per layer (0 on the SSM ones). The dense
  350m we run is `ibm-granite/granite-4.0-350m-GGUF`, arch `granite`, 28
  blocks. Both ship files named `granite-4.0-...` and only the repo tells them
  apart. We load the hybrid far enough to reject it by name — the
  "head_count_kv varies per layer" refusal in `src/model.c` is what catches it,
  which is the argument for refusing rather than reading element 0.
- **Interleaved RoPE, not NeoX split-half.** llama.cpp's converter permutes q
  and k for Llama-family checkpoints; Granite inherits that path. Both variants
  produce fluent English: perplexity 412 wrong against 122 right on one
  55-token passage. There is no metadata key for it — it is a per-architecture
  table in llama.cpp and in `src/model.c`.
- **Four muP scalars**, all in the GGUF: `attention.scale` 0.015625 (where
  1/sqrt(64) would be 0.125), `embedding_scale` 12, `residual_scale` 0.263,
  `logit_scale` 4 (logits are DIVIDED). Removing any one costs orders of
  magnitude of perplexity.
- **`attention.head_count_kv` is an ARRAY** of 28 entries, from the hybrid
  config class. Uniform here; an array whose entries differ must be refused,
  not read as its first element.
- **No `attention.key_length`.** head_dim comes from `rope.dimension_count`.
  Never from d_model / n_heads.
- **Its own chat template**: `<|start_of_role|>role<|end_of_role|>` …
  `<|end_of_text|>`, and a canned system turn when the caller supplies neither
  one nor tools. The `<tool_call>` and `<tool_response>` bodies are byte-
  identical to Qwen's.

---

## Gemma 4 E2B-it — v0.2 production target

Apache 2.0, released 2026-07-02. 2.3B effective / 5.1B total via Per-Layer
Embeddings. 35+ languages out of the box (140+ pre-trained), native tool
calling, built-in reasoning, 128K context.

Default download: `google/gemma-4-E2B-it-qat-q4_0-gguf`, ~1.5 GB.

Two things about that repo, both verified against the HF API:

- The file inside is **`gemma-4-E2B_q4_0-it.gguf`** — not the repo-name pattern
  you would guess. `scripts/download_model.sh` has the real name.
- The repo also ships `gemma-4-E2B-it-mmproj.gguf`, the multimodal projector.
  We deliberately do not download it: mynah-slm is the **text tower only**, and
  ASR is `mynah-asr`'s job.

**Not yet inspected** — not downloaded. Its architecture traps will be written
up in `docs/gemma4-arch.md` and verified against a real checkpoint in M7.1.

The QAT checkpoint is plain **`Q4_0`**, which ingot decodes but only through its
*generic* kernel. See TASKS.md 0.4: the fast NEON kernel already exists in
qwen-tts and needs porting upstream into ingot before this model's performance
means anything.

---

### The size budget is in the lookup tables

Derived from `text_config`, to be confirmed against the file. With
`vocab_size_per_layer_input: 262144`, `hidden_size_per_layer_input: 256` and 35
layers, the Per-Layer Embedding tables are

```
262144 × 256 × 35  ≈  2.35 B parameters
```

plus a `262144 × 1536` ≈ 0.4 B main embedding. That is ~2.75 B of the 5.1 B
total, and it lands exactly on the "5.1 B total / 2.3 B effective" split the
model card advertises — which is a good sign the derivation is right.

So the majority of this checkpoint is **lookup tables, not GEMM operands**.
They are read, never multiplied: no kernel, no accumulation error, just bytes.
That makes them both the largest and the cheapest thing to quantize hard, and
it inverts the inherited `mynah-asr` rule of keeping embeddings f32 — a rule
that was correct for a 0.6B encoder with a small vocabulary.

First thing to check once the file is on disk: does the shipped QAT GGUF
already quantize its PLE tables, or leave them wide?

---

## Why `Q4_K_M` is not a type

Worth stating once, because it decides how to read every table above. A file
called `model-Q4_K_M.gguf` contains no tensor of type `Q4_K_M` — there is no
such type. The suffix names a **recipe**: which real block types the quantizer
assigned to which tensors. The census above is the recipe, made visible.

This is also why `inspect` reports bits/weight over the whole file: it is the
number that predicts RAM, and it is always worse than the headline.
