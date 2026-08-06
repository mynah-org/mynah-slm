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

> **Open question for M1.** Which 29 tensors are they? 28 layers + 1 suggests
> "one per layer plus the embedding", but the census counts types, not names,
> so this is arithmetic and not evidence. Resolve it while building the
> GGUF⇄safetensors name map — and if the embedding really is 167 MiB of a
> 372 MiB file, then embedding quantization, not weight quantization, is where
> the size budget actually lives.

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

## Why `Q4_K_M` is not a type

Worth stating once, because it decides how to read every table above. A file
called `model-Q4_K_M.gguf` contains no tensor of type `Q4_K_M` — there is no
such type. The suffix names a **recipe**: which real block types the quantizer
assigned to which tensors. The census above is the recipe, made visible.

This is also why `inspect` reports bits/weight over the whole file: it is the
number that predicts RAM, and it is always worse than the headline.
