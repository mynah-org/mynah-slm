# Mynah SLM

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Build & Test](https://github.com/mynah-org/mynah-slm/actions/workflows/build.yml/badge.svg)](https://github.com/mynah-org/mynah-slm/actions/workflows/build.yml)
[![Memory Safety](https://github.com/mynah-org/mynah-slm/actions/workflows/safety.yml/badge.svg)](https://github.com/mynah-org/mynah-slm/actions/workflows/safety.yml)
[![Code Quality](https://github.com/mynah-org/mynah-slm/actions/workflows/codeql.yml/badge.svg)](https://github.com/mynah-org/mynah-slm/actions/workflows/codeql.yml)

**A fast native C inference engine for small language models** — CPU-first,
streaming, multilingual, with tool calling and a thinking on/off switch.
No Python at runtime, no cloud, no telemetry.

> **Status: v0.1 in progress — Qwen3-0.6B runs end to end.** Generation,
> streaming, thinking on/off, tool calling and the HTTP server all work today,
> at **~36 tok/s decode and ~370 tok/s prefill** on an M-series Mac (`Q4_K_M`,
> 8 threads, weights staged locally). Constrained decoding and Gemma 4 are not
> written yet — see [what works today](#what-works-today) for the honest line
> between the two.

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
| Our own Q4_K matvec (`src/qmat.c`) | ✅ **1.55x** over ingot's per tensor, decode 28 → **36.5 tok/s** |
| KV cache precision (`--kv f32\|bf16\|fp8\|q8\|q4`) | ✅ **bf16 by default**: half the memory, +26% decode at 2K context, same perplexity |
| `--fast` — int8 activations (SDOT) | ✅ opt-in: **+25% decode** for +1.4% perplexity. Off by default, because it is the one change that is not free |
| Batched prefill (projections **and** attention) | ✅ **14x** — 26 → 370 tok/s on a 198-token prompt, TTFT 7.6 s → 0.58 s |
| HTTP server: `/health`, `/v1/models`, `/v1/chat/completions` (+SSE), `/v1/tokenize` | ✅ |
| Tool calling: schemas in, calls out, second turn, SSE | ✅ prompt byte-identical to HF; **26/30** with `--think on`, 21/30 without ([docs/tools.md](docs/tools.md)) |
| Constrained decoding (`strict` mode, JSON schema) | ❌ not written |
| Second model family: **Granite 4.0 350m** (dense) | ✅ runs — muP scalars, interleaved RoPE, its own chat template, all config-driven |
| Gemma 4 E2B (v0.2) | ❌ not started |

Speed, on an M-series Mac with `Qwen3-0.6B-Q4_K_M` staged locally — decode
4.1 → 36.5 tok/s in five measured steps, and prefill 26 → 370 tok/s once it
stopped being one token at a time. Each step was found by measuring rather than
guessing; the method, the per-tensor breakdown and the long-context numbers are
in [docs/perf.md](docs/perf.md).

```
$ mynah-slm run -m models-local/Qwen3-0.6B-Q4_K_M.gguf -p "Racconta una storia breve su un faro." -n 60 --temp 0
[load 0.01s | prompt 24 tok, prefill 127.5 tok/s | gen 60 tok, decode 36.5 tok/s | TTFT 218 ms | 8 threads]
```

Every run reports its own speed on **stderr**, so stdout stays clean enough to
pipe into `mynah-tts`. `--quiet` is the only thing that hides it.

## Target models

| | Model | Why |
|---|---|---|
| Baseline | **Qwen3-0.6B / 1.7B / 4B** (Apache 2.0) | dense GQA + QK-norm, 100+ languages, `/think` `/no_think`, tool calling |
| Light / agentic | **Granite 4.0 350m** (Apache 2.0) | **runs today.** Better at tool calling (27/30 at Q8_0 vs 25/30, and no thinking tax), a quarter of the KV cache, 226 MB and 47 tok/s at Q4. Weaker multilingual — 27% behind on Italian. [The full comparison](docs/models.md#granite-40-350m--the-light-alternative-measured-against-qwen3), including [the Q4/Q5/Q6/Q8 ladder](docs/models.md#the-quantization-ladder--is-there-a-middle-ground) |
| Production | **Gemma 4 E2B-it QAT Q4_0** (Apache 2.0) | 2.3B effective, 35+ languages, native tool calling, built-in reasoning, 128K context |

Text tower only — the vision and audio towers in the Gemma checkpoint are
ignored on purpose. ASR is [`mynah-asr`](https://github.com/mynah-org/mynah-asr)'s job.

## Quickstart

**1 — Build.** A C11 compiler and a BLAS is the whole list: Accelerate on macOS
(nothing to install), OpenBLAS on Linux. `ingot` is vendored in-tree as a
subtree, so a plain clone builds — there is no submodule to init.

```sh
git clone https://github.com/mynah-org/mynah-slm.git && cd mynah-slm
make          # macOS: Accelerate, zero deps
              # Linux: sudo apt install libopenblas-dev  (Fedora: openblas-devel)
```

**2 — Get a model.** `scripts/download_model.sh` pulls a GGUF checkpoint
straight from HuggingFace — no account, no token, resumable:

```sh
scripts/download_model.sh --list                  # every supported checkpoint
scripts/download_model.sh --model qwen3-0.6b-q4   # 378 MB, the v0.1 default
```

| if you want… | alias | size |
|---|---|---|
| the default — the smallest thing that works well | `qwen3-0.6b-q4` | 378 MB |
| the least quantization noise (official Qwen build) | `qwen3-0.6b-q8` | 610 MB |
| the same architecture, scaled up | `qwen3-1.7b-q4` | 1.1 GB |
| the quality option | `qwen3-4b-q4` | 2.5 GB |
| the v0.2 production target — **not implemented yet** | `gemma4-e2b-qat` | 1.5 GB |

**3 — Run it.**

```sh
./mynah-slm run -m models/Qwen3-0.6B-Q4_K_M.gguf -p "Ciao! Come stai?"
```

```
Ciao! Sto bene, grazie! Come posso aiutarti oggi?
[load 0.03s | prompt 19 tok, prefill 69.9 tok/s | gen 12 tok, decode 25.2 tok/s | TTFT 318 ms | 8 threads]
```

The answer is on stdout and the timing line on stderr, so the pipe into the
next stage stays clean:

```sh
# think first, say only the answer, straight into TTS
./mynah-slm run -m models/Qwen3-0.6B-Q4_K_M.gguf -p "Summarize in one line: ..." \
  --think on | mynah-tts speak

# tool calling: schemas in, one JSON line out. The engine never executes them.
# --think on is worth it here: 26/30 against 21/30 on the decision suite.
./mynah-slm run -m models/Qwen3-0.6B-Q4_K_M.gguf --think on \
  --tools docs/examples/weather.json -p "What is the weather in Verona?"
#> {"tool_calls":[{"index":0,"id":"call_0","type":"function",
#>   "function":{"name":"get_weather","arguments":"{\"city\": \"Verona\"}"}}]}

# what is actually inside a checkpoint
./mynah-slm inspect models/Qwen3-0.6B-Q4_K_M.gguf --tensors Q6_K

# OpenAI-shaped HTTP server, SSE included
./mynah-slm-server -m models/Qwen3-0.6B-Q4_K_M.gguf --port 8080
```

Weights are **not** in this repo and never will be; `models/` is gitignored.
Any GGUF Qwen3 checkpoint works, whether this script fetched it or not.

## Build & test

```sh
make            # mynah-slm + mynah-slm-server
make lib        # libmynah_slm.a       (make shared for the .so/.dylib)
make test       # unit + parity; cases that need weights report SKIP
make help       # every target
```

`make test` passes without a checkpoint: the tests that need one exit 77 and
say so. That is also why CI never downloads a model — and why the forward pass,
the tokenizer round-trip and generation are covered locally but not by CI.

## License

MIT — like every other project in the Mynah ecosystem.
