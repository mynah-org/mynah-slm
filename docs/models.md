# Models

Every number here is measured with `mynah-slm inspect`, not copied from a model
card. Weights live on the NAS (`models/` is a symlink); see CLAUDE.md.

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

**Not yet inspected** — not downloaded. Architecture traps are in CLAUDE.md and
will be verified against a real checkpoint in M7.1.

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
