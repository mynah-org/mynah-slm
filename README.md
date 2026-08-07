# Mynah SLM

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Build & Test](https://github.com/mynah-org/mynah-slm/actions/workflows/build.yml/badge.svg)](https://github.com/mynah-org/mynah-slm/actions/workflows/build.yml)
[![Memory Safety](https://github.com/mynah-org/mynah-slm/actions/workflows/safety.yml/badge.svg)](https://github.com/mynah-org/mynah-slm/actions/workflows/safety.yml)
[![Code Quality](https://github.com/mynah-org/mynah-slm/actions/workflows/codeql.yml/badge.svg)](https://github.com/mynah-org/mynah-slm/actions/workflows/codeql.yml)

**A fast native C inference engine for small language models** — CPU-first,
streaming, multilingual, with tool calling and a thinking on/off switch.
No Python at runtime, no cloud, no telemetry.

> **Status: v0.1 in progress — Qwen3-0.6B runs end to end.** Generation,
> streaming, thinking on/off and the HTTP server all work today, at
> **27.1 tok/s decode** on an M-series Mac (`Q4_K_M`, 8 threads, weights staged
> locally). Tool calling and Gemma 4 are not written yet — see
> [what works today](#what-works-today) for the honest line between the two.

```
mic → mynah-asr → mynah-slm → mynah-tts → speaker
        (C)         (C)          (C)
```

`mynah-slm` is the text stage of the [Mynah](https://github.com/mynah-org)
voice stack: it turns a transcript into a cleaned-up text, a summary, an
intent, a JSON payload, a tool call, or a spoken reply — and it streams,
because the TTS stage downstream is waiting on its first sentence.

Each stage is a standalone binary with a standalone C API and a standalone
HTTP server. They are composed by the application, never by each other.

## What it is

- **Pure C11**, zero runtime dependencies beyond `libc`/`pthread`/`libm`
  (optional BLAS)
- **CPU-first** — the CPU path is the reference implementation, always correct
  and always fast enough to ship. GPU backends are accelerators, never the only
  way to run
- **Custom engine** — no ggml, no llama.cpp, no ONNX Runtime, no PyTorch
- **Both weight containers** — GGUF and safetensors through
  [`ingot`](https://github.com/mynah-org/ingot), converging on one in-memory
  representation right after load
- **Streaming-first** — token-by-token is the primary API, not an add-on
- **Tool calling** *(planned)* — the engine will *emit* tool calls and never
  execute them. Execution belongs to the application

## What works today

Qwen3-0.6B is implemented and validated; Gemma 4 is not started.

| | state |
|---|---|
| `inspect` — per-type census, tensors, metadata KV | ✅ |
| Byte-level BPE tokenizer straight from GGUF | ✅ id-identical to HF on 45 fixture cases |
| Qwen3 forward pass (GQA, per-head QK-RMSNorm, NeoX RoPE, SwiGLU) | ✅ matches the numpy oracle to 1e-6 |
| Generation: chat template, sampler (`--temp/--top-k/--top-p/--min-p/--seed`) | ✅ |
| Streaming, with TTFT measured at the first token | ✅ |
| Thinking `off` / `low` / `on`, reasoning kept off stdout | ✅ |
| Multi-threading | ✅ 3.56x on 8 threads, bit-identical to the serial path |
| HTTP server: `/health`, `/v1/models`, `/v1/chat/completions` (+SSE), `/v1/tokenize` | ✅ |
| Tool calling | ❌ not written |
| Gemma 4 E2B (v0.2) | ❌ not started |

Speed, on an M-series Mac with `Qwen3-0.6B-Q4_K_M` staged locally — 4.1 → 27.1
tok/s decode in four measured steps, each one found by measuring rather than
guessing. The full method, the per-tensor breakdown and the long-context
numbers are in [docs/perf.md](docs/perf.md).

```
$ mynah-slm run -m models-local/Qwen3-0.6B-Q4_K_M.gguf -p "Ciao! Come stai?" -n 24 --temp 0
Ciao! Sto bene! Cosa ne hai?
[load 0.02s | prompt 19 tok, prefill 28.9 tok/s | gen 12 tok, decode 27.1 tok/s | TTFT 697 ms | 8 threads]
```

Every run reports its own speed on **stderr**, so stdout stays clean enough to
pipe into `mynah-tts`. `--quiet` is the only thing that hides it.

## Target models

| | Model | Why |
|---|---|---|
| Baseline | **Qwen3-0.6B / 1.7B / 4B** (Apache 2.0) | dense GQA + QK-norm, 100+ languages, `/think` `/no_think`, tool calling |
| Production | **Gemma 4 E2B-it QAT Q4_0** (Apache 2.0) | 2.3B effective, 35+ languages, native tool calling, built-in reasoning, 128K context |

Text tower only — the vision and audio towers in the Gemma checkpoint are
ignored on purpose. ASR is [`mynah-asr`](https://github.com/mynah-org/mynah-asr)'s job.

## Building

Needs a C11 compiler and a BLAS: Accelerate on macOS (nothing to install),
`libopenblas-dev` on Linux. `ingot` is vendored in-tree as a subtree, so a
plain clone builds — there is no submodule to init.

```sh
make            # mynah-slm + mynah-slm-server
make lib        # libmynah_slm.a       (make shared for the .so/.dylib)
make test       # unit + parity; cases that need weights report SKIP
make help       # every target
```

`make test` passes without a checkpoint: the tests that need one exit 77 and
say so. That is also why CI never downloads a model.

## Using it

```sh
# one-shot generation, streaming, thinking off
mynah-slm run -m model.gguf -p "Riassumi in una riga: ..." --think off

# what is actually inside a checkpoint
mynah-slm inspect model.gguf --tensors Q6_K

# OpenAI-shaped HTTP server, SSE included
mynah-slm-server -m model.gguf --port 8080
```

Weights are **not** in this repo and never will be. Point `-m` at any GGUF
Qwen3 checkpoint.

## License

MIT — like every other project in the Mynah ecosystem.
