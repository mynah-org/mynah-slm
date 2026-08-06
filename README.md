# Mynah SLM

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**A fast native C inference engine for small language models** — CPU-first,
streaming, multilingual, with tool calling and a thinking on/off switch.
No Python at runtime, no cloud, no telemetry.

> **Status: scaffolding.** Nothing runs yet. See [docs/prior-art.md](docs/prior-art.md)
> for what this is being built from.

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
- **Tool calling** — the engine *emits* tool calls and never executes them.
  Execution belongs to the application

## Target models

| | Model | Why |
|---|---|---|
| Baseline | **Qwen3-0.6B / 1.7B / 4B** (Apache 2.0) | dense GQA + QK-norm, 100+ languages, `/think` `/no_think`, tool calling |
| Production | **Gemma 4 E2B-it QAT Q4_0** (Apache 2.0) | 2.3B effective, 35+ languages, native tool calling, built-in reasoning, 128K context |

Text tower only — the vision and audio towers in the Gemma checkpoint are
ignored on purpose. ASR is [`mynah-asr`](https://github.com/mynah-org/mynah-asr)'s job.

## Building

```sh
make            # mynah-slm CLI
make server     # mynah-slm-server
make lib        # libmynah_slm.a
make test
```

## License

MIT — like every other project in the Mynah ecosystem.
