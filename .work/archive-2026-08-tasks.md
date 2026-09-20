# Archive — the M0-M6 task breakdown (moved from `TASKS.md`, 2026-09-20)

Status: **ARCHIVE**. This is the old `TASKS.md`, moved here verbatim when
`PLAN.md` became a board and `.work/` became the place for detail (the
`mynah-tts` convention). Nothing below has been edited: it is the record of what
was planned and ticked off through M0-M6, and the `[ ]` lines in it are still
the honest backlog for those milestones.

New work does not get appended here. It gets a line on the `PLAN.md` board and
its own note in `.work/`.

---

# mynah-slm — Task Breakdown

Reference: [PLAN.md](PLAN.md). One milestone per section, tasks in dependency
order. A task is done only when it is committed **and** covered by a test or a
documented manual check.

Conventions:
- "validated vs oracle" = numeric comparison at a per-stage tolerance against
  `tools/oracle/` — never an md5, never "looks right".
- Every task that touches a numeric stage must be de-risked on paper (docs/)
  before code is written.
- Small, frequent commits. **Everything in the repo is in English** — code
  comments, identifiers, log and error strings, every `.md`, the Makefile,
  `scripts/`, `tools/` Python, CI, and every commit message and PR body. See
  PLAN.md "Language Policy". The one exception is multilingual test/eval data,
  where the non-English text is the input under test.

---

## M0 — Repository foundation

### 0.1 Repo scaffolding
- [x] `git init` (branch `main`) — 2026-08-06
- [x] `LICENSE` — **MIT**, same text as `mynah-asr`/`ingot`/`keyra`
- [x] `.gitignore` — sibling style: **PLAN.md / TASKS.md / CLAUDE.md / AGENTS.md
      gitignored** (local notes, never public), plus `*.o`, `build/`, binaries,
      `models/`, `.venv`, `*.gguf`, `*.safetensors`
- [x] `.gitattributes` — `reference/**`, `third_party/**`, `tools/**`
      `linguist-vendored` (keeps GitHub from mislabelling the repo)
- [x] `README.md` skeleton: mission, ASR→SLM→TTS diagram, target models, build
- [x] `CLAUDE.md` — repo contracts (map, NAS policy, quant policy, rules, build
      & test, known traps), `AGENTS.md` symlinked to it
- [x] `docs/prior-art.md` — the reuse survey across qwen-tts / keyra / mynah-asr
      / mynah-tts / qwen3.c
- [x] NAS: `/Volumes/shared/mynah-slm/models/` created, `./models` symlinked to it
- [x] `mynah-org/mynah-slm` created on GitHub, `origin` added
      (`git@github.com:mynah-org/mynah-slm.git`) — 2026-08-06
- [x] First commit + `git push -u origin main` — 2026-08-07, 26 commits at once.
      ingot pushed the same day (`f6888a7..4147d14`, the three kernel commits)
- [ ] Copy the Mynah logo into `assets/`

### 0.2 Build system
- [x] `Makefile` modelled on `mynah-asr/Makefile`: C11, `-O3 -march=native`,
      Accelerate/OpenBLAS auto-detect, no CMake — 2026-08-06
- [x] **Deliberate deviation: no `-ffast-math`.** mynah-asr uses it; a decoder
      runs `expf` over logits and softmax over scores, and under `-ffast-math`
      an inf is UB that gcc/x86 turns into NaN through libmvec. That is the
      2026-07-18 mynah-asr CI incident. Documented at the top of the Makefile
- [x] Targets: `all` (CLI), `lib`, `shared`, `test`, `clean`, `debug`, `ubsan`,
      `asan` (Linux CI only), `leaks`, `install`, `update-ingot`, `help`
- [x] Sanitizer variants go through `EXTRA_CFLAGS`, never `CFLAGS=` — overriding
      drops the computed include paths and the quoted `-DMYNAH_SLM_BUILD`
      (caught by `make ubsan` failing to compile on the first try)
- [x] Build is clean with `-Wall -Wextra`, **zero warnings**
- [x] `make leaks` clean (0 leaks), `make ubsan` clean
- [x] `compile_flags.txt` for clangd
- [ ] `server` target (needs `server/`)
- [x] `bench` target — `tests/bench_matvec.c`, per-tensor matvec throughput
      with an INTERLEAVED A/B of our kernels against ingot's in one process.
      Two separate runs disagreed by 80% on tensors neither side touches; those
      same tensors landing on 1.00x is now the control — 2026-08-07

### 0.3 Vendor ingot
- [x] `git subtree add --prefix third_party/ingot …` — 2026-08-06
- [x] Makefile wiring: `$(INGOT_LIB)` built by its own Makefile,
      `-Ithird_party/ingot/include`, and `clean` recurses into it (otherwise a
      stale `libingot.a` survives a subtree update and links silently)
- [x] `make update-ingot` target (subtree pull + clean)
- [x] `tests/test_ingot.c` — pins the container contract with **no model**:
      block geometry for Q4_0/Q8_0/Q4_K/Q6_K/Q3_K/Q2_K against docs/QUANTS.md,
      dequant coverage for every quant we might ship, and `Q1_0` asserted as
      the known gap so the day it changes we hear it from a test
- [x] `src/inspect.c` + `mynah-slm inspect` — per-type census of a checkpoint,
      because "does it run" is "is every block type in it decodable", and a
      `Q4_K_M` filename tells you nothing (that file is Q4_K + Q6_K + F32)
- [x] `tests/test_inspect.c` — synthesizes a Q4_K_M-shaped GGUF via ingot's
      writer and asserts the census end-to-end, no download needed.
      **Found a real use-after-free on the first run**: `arch` borrowed a string
      owned by the ingot handle, which is closed before returning
- [x] Run `inspect` on real checkpoints and record the census in
      `docs/models.md` — Qwen3-0.6B Q4_K_M and Q8_0 done 2026-08-06.
      **Finding: the Q4_K_M build is 5.24 bits/weight, not 4.5** — 29 tensors
      at Q6_K are 45% of the file
- [x] `inspect --tensors Q6_K` — the 29 are `token_embd` (121.7 MiB, **32.7% of
      the file on its own**) + `attn_v`/`ffn_down` on 14 of 28 layers. No
      `output.weight`: embeddings are tied, so `token_embd` is ALSO the LM head
      — a hot GEMV in front of the softmax, not a lookup table. Do not quantize
      it like one
- [ ] Same for Gemma 4 QAT once downloaded: are the PLE tables quantized in the
      shipped GGUF, and is its embedding tied too?

### 0.4 Upstream: Q4_0 SIMD kernel into ingot ⚠️ blocks v0.2 performance
> ingot decodes `Q4_0` fine but its kernel is **generic** (row-decode then
> multiply). Gemma 4 QAT — our default production model — *is* `Q4_0`, so it
> would run on the slow path. qwen-tts already has the fast kernel with the
> identical 18-byte block layout.
- [ ] Port `qwen_matvec_q4_0` (NEON, threaded) from `../qwen-tts/qwen_tts_kernels.c`
      into `ingot/src/kernels.c`
- [ ] Port the batched `qwen_matmat_q4_0` twin (nibble unpack amortized over B columns)
- [ ] Add the AVX2 / AVX-512-VNNI paths to match the K-quant kernels' coverage
- [ ] Wire into `ingot_matvec` / `ingot_matmat` type dispatch
- [ ] Validate bit-exactness vs ingot's own generic path, add to ingot's `make test`
- [ ] Benchmark: `Q4_0` matvec before/after, on ARM and x86
- [ ] Upstream PR to `mynah-org/ingot`, then `make update-ingot` here

### 0.5 CI
- [x] `.github/workflows/build.yml` — 4 rows: linux x86_64 native, linux x86_64
      **without AVX2** (the scalar twin of the attention kernel, never compiled
      under `-march=native` on a hosted runner), linux aarch64, macos arm64.
      Build + `make test` + lib/shared + staged `make install` — 2026-08-07
- [x] `.github/workflows/safety.yml` — UBSan and ASan on linux x86_64 + aarch64,
      `make leaks` on macOS, `-Werror` with gcc/clang — 2026-08-07
- [x] `.github/workflows/codeql.yml` — CodeQL (`third_party` excluded: ingot has
      its own upstream) + clang-tidy with ingot's check set — 2026-08-07
- [x] **No model is ever downloaded in CI.** The tests that need weights exit 77
      and `make test` returns 0 — a model-free run is a supported mode.
      Consequence to keep in mind: the forward pass, the BPE round-trip and
      generation are NOT covered by CI, only locally.
- [ ] Synthetic tiny GGUF built in-tree with ingot's writer, so the forward pass
      and generation get a CI gate too (a few MB, no download)
- [ ] `.github/workflows/release.yml` — tagged binaries

### 0.6 Offline tooling skeleton
- [x] `scripts/download_model.sh` — non-interactive, resumable (`curl -C -`),
      `--fail` so an HTML error page never lands as a corrupt `.gguf`,
      **writes to `models/` = the NAS**, and refuses to run if the NAS is not
      mounted rather than filling the internal disk
- [x] Qwen3-0.6B Q4_K_M + Q8_0 downloaded to the NAS
- [ ] `tools/` as a `uv` project (numpy, safetensors, tokenizers, huggingface-hub;
      `torch` only as an `oracle` extra)
- [ ] Confirm Python is never needed at inference time (documented in README)

---

## M1 — Qwen3 de-risking (before any runtime code)

### 1.1 Architecture study
- [x] GGUF Q4_K_M + Q8_0 on the NAS. **safetensors NOT downloaded** — the
      header is JSON at byte 8, so a range request read all 311 tensor
      descriptors for 35 KB instead of 1.2 GB
- [x] `inspect --meta` — GGUF metadata dumped. **Fixed a rendering bug it
      exposed**: `rms_norm_eps`/`rope.freq_base` are FLOAT32 KVs and ingot's
      integer accessors convert them silently, so a first-match accessor chain
      showed `rms_norm_eps = 0`. Regression test added
- [x] Full GGUF⇄safetensors name mapping table → `docs/qwen3-arch.md`.
      310 vs 311 tensors: safetensors carries a redundant `lm_head.weight`
      duplicating the tied embedding at its own offsets, **verified
      byte-identical with a 200-byte range read** rather than assumed
- [x] `docs/qwen3-arch.md`: shapes, forward pass, metadata keys, trap list
- [x] `reference/qwen3-0.6b/` committed: config, generation_config,
      safetensors header
- [ ] Read [`adriancable/qwen3.c`](https://github.com/adriancable/qwen3.c) (MIT)
      in detail and extend `docs/prior-art.md` with its numerical checklist

### 1.2 Python oracle
- [x] `tools/` as a uv project; `tools/oracle/` — pure numpy Qwen3 forward pass,
      no torch. **Reads the same GGUF the engine reads**, not bf16 safetensors:
      identical weights mean a mismatch is our bug, not quantization
- [x] Works end to end — coherent, correctly terminated answers in it/en/de
- [x] `--dump-dir`: embed, layers 0/13/27, final_norm, logits → `.npy`
- [x] `oracle/test_kernels.py` — kernel properties, no checkpoint needed.
      Distinguishes split-half from interleaved RoPE in one shot; asserts that
      truncating the sequence leaves earlier attention outputs unchanged (the
      property that makes the C KV cache valid at all)
- [x] `tools/eval/compare.py` — per-stage relative error + cosine, never a hash
      (the two implementations will never be bit-identical, and a checksum would
      fail on a correct engine and teach us to ignore it)
- [x] `tools/eval/test_gate.py` — **proves the gate rejects the bugs it exists
      for**. fp16-residual is caught at layer 0 on relative error while its
      greedy argmax still agrees 10/10; interleaved-RoPE agrees 5/10. A
      "same text?" check would have passed both
- [x] Fixed `silu` to use a stable sigmoid — CLAUDE.md carried the rule and the
      oracle was breaking it
- [ ] Validate the ORACLE itself against HF `transformers` (needs the `check`
      extra: torch + the 1.2 GB safetensors). The three checks above are
      internal consistency; this is the external one

---

## M2 — Core engine (v0.1)

> **Read [docs/prior-art.md](docs/prior-art.md) before writing a line here.**
> `../qwen-tts` already contains a validated Qwen3 decoder (`qwen_tts_talker.c`)
> and the whole SIMD kernel set (`qwen_tts_kernels.c`). Most of M2 is *porting,
> renaming and generalizing*, not inventing. The genuinely new code is: the
> GGUF⇄safetensors name mapping, the GGUF tokenizer path, incremental UTF-8
> detokenization, templates/thinking/tools, grammar decoding, and session KV
> reuse.

### 2.1 Model container
- [x] `src/model.{c,h}` — config + bound tensors, architecture-agnostic.
      Key prefix from `general.architecture`, so the file never contains the
      string "qwen3"
- [x] `head_dim` from `attention.key_length`, **never derived** (128 vs the
      64 that `d_model/n_heads` would give). Test asserts `q_dim != d_model`
      to catch a regression to the derived form
- [x] Floats read through `_f64` only — ingot's integer accessors silently
      convert a FLOAT32 KV, turning `rms_eps` into 0
- [x] Tensors kept as ingot descriptors (zero-copy, quantized as stored);
      `q_norm`/`k_norm` may be NULL so a family without QK-norm is "absent",
      not "missing"
- [x] Refuses to run on `n_heads % n_kv_heads != 0` and on
      `value_length != key_length` rather than guessing
- [x] `tests/test_model.c` — checked against **docs/qwen3-arch.md**, not against
      itself; walks all 28 layers; passes on Q4_K_M and Q8_0; missing file
      skips (77) while a non-GGUF fails. Zero leaks, UBSan clean
- [ ] safetensors loading path (the name map is written; the code is not)
- [ ] `arch` vtable (`load`/`forward`/`free`) — lands with arch_qwen3.c, since
      a vtable with one implementation and no forward pass would be scaffolding
      that lies

### 2.2 Tokenizer
- [ ] `src/tokenizer.c` — BPE from GGUF `tokenizer.ggml.*` arrays
- [ ] `tokenizer.json` (HF) fallback path for safetensors checkpoints
- [ ] Byte-level pre-tokenization, byte fallback, special/added tokens
- [ ] UTF-8 safe incremental detokenization (never emit a partial codepoint
      to the stream callback)
- [ ] Test: round-trip 10k lines across 10 languages (it/en/de/fr/es/zh/ja/ar/ru/pt),
      token-id-identical to HF `tokenizers`

### 2.3 Kernels — port from qwen-tts
- [ ] Port `qwen_rms_norm`, `qwen_rms_norm_residual` (fused), `qwen_rms_norm_per_head`
      (= Qwen3's QK-norm) → `src/kernels.c`
- [ ] Port `qwen_swiglu_inplace`, `qwen_silu`, `qwen_bf16_to_f32_vec`,
      `qwen_bf16_accum_f32`, `qwen_f16_to_f32`/`qwen_f32_to_f16`
- [ ] Port `qwen_causal_attention`, `_windowed` (Gemma 4 needs it) and `_bf16kv`
- [ ] Port `qwen_argmax_matvec_*` — argmax without materializing the logits;
      with Gemma's 262144-row head this is a real decode win
- [ ] Port the fused QKV matvec (1 dispatch instead of 3 barriers)
- [ ] Port `aligned_malloc`/`aligned_calloc` (64 B) and use them for **every**
      SIMD/BLAS buffer
- [ ] Port the infra: `ftz_on()` (call on every pool worker), `check_runtime_isa()`,
      `caps_report()`, `kernel_selftest()`, `matmat_bench()`
- [ ] `src/threads.c` — port qwen-tts's pool + mynah-asr's `blas_set_concurrency()`
      (a server with 4+ concurrent inferences collapses on the OpenBLAS lock without it)
- [x] Matmul routing — `src/qmat.c`, mynah-asr's policy ported: quantized +
      one token → direct dot; quantized + many tokens (prefill) → dequantize a
      row strip + `sgemm`. **Not delegated to `ingot_matmat`**: from two tokens
      up it quantizes activations to int8 (2.4e-3), 20x our parity gate
- [x] **Our own Q4_K matvec** (2026-08-07): distribute the sum so the scale
      applies once per 32-weight sub-block, and hoist `SUM x_j` — which depends
      only on the input — out of the row loop. **1.53-1.58x** over ingot's per
      tensor, decode **28 → 36.5 tok/s** end to end, greedy output identical.
      Gated against ingot on synthetic weights (rel 3.5e-07, no checkpoint)
- [x] Batched prefill: projections AND attention. 26 → 370 tok/s on a
      198-token prompt (TTFT 7.6 s → 0.58 s), 21 → 232 on a 2275-token one
- [x] BLAS path for the prefill GEMM and for batched attention (Accelerate / OpenBLAS)
- [ ] Test: each kernel vs numpy, tolerance 1e-5 (f32) / 1e-2 (q4), **plus**
      `kernel_selftest` in CI on both ARM and x86

### 2.4 KV cache
- [ ] **bf16 KV cache** — the next big one, and the only change that makes the
      engine lighter AND faster at once: at a 2275-token context the caches are
      ~520 MB, more than the model, and attention re-reads them every token.
      Needs its own quality measurement (it moves what the parity gate watches)
- [ ] `src/kvcache.c` — per-layer ring buffer, GQA-aware, contiguous per head
- [ ] Preallocated at session creation from `n_ctx`; zero allocation in the
      token loop
- [ ] Sliding-window support in the data structure from day one (Gemma 4 needs it)
- [ ] Test: memory ceiling assertion, and prefill+decode == full-sequence forward

### 2.5 Qwen3 forward pass — port from `qwen_tts_talker.c`
- [ ] `src/arch_qwen3.c` — embed → 28×(RMSNorm, QKV, **QK-RMSNorm**, RoPE, GQA
      attention, RMSNorm, SwiGLU) → final norm → tied LM head
- [ ] Take `apply_rope_neox_inplace` (**split-half NeoX, NOT interleaved** —
      `qwen_tts_kernels.h` ships both, grabbing the wrong one is the single most
      expensive trap in a Qwen3 port) + the `rope_inv_freq`/`cos`/`sin` precompute
- [ ] Take the bf16 KV cache with geometric growth (`kv_cache_grow`)
- [ ] Take the per-layer activation-dump hook (qwen-tts's `QWEN_ACT_MAP`) —
      it is exactly the shape the oracle parity check needs
- [ ] Strip everything TTS-specific (codec heads, speech tokens, voice cloning)
- [ ] Prefill (batched over prompt) and decode (single token) share **one code
      path** — divergent paths breed bugs (qwen-asr lesson)
- [ ] Validate layer by layer vs the oracle dumps
- [ ] Validate final logits, then greedy generation: 64 tokens identical to the
      oracle on 20 prompts in 5 languages

### 2.6 Sampling
- [ ] `src/sampler.c` — greedy, temperature, top-k, top-p, min-p,
      repetition/frequency/presence penalties, seeded RNG
- [ ] Deterministic with a fixed seed, documented and tested
- [ ] Logit bias, EOS list support (Gemma 4 has two terminators)

### 2.7 Session + chat template
- [ ] `src/session.c` — message list, prompt assembly, incremental KV reuse
      across turns, context-window eviction policy
- [ ] `src/template.c` — Qwen3 ChatML template, `<think>` handling,
      `/think` `/no_think` switches
- [ ] Thinking channel separated from the answer channel in the callback
- [ ] Test: rendered prompt string byte-identical to HF `apply_chat_template`

### 2.8 Public API
- [ ] `include/mynah_slm.h` — the API from PLAN.md, C++-parsable, no internal types
- [ ] Errors returned as strings; nothing written to `stderr`; no hidden allocations
- [ ] `examples/minimal.c` — load, chat, stream, in under 60 lines
- [ ] `make leaks` + `make ubsan` clean on the example

---

## M2c — Timing instrumentation (cuts across M3/M4, build it once)

> Every mode reports its own speed. Not behind `--verbose`, not only in
> `make bench`. See PLAN.md "Speed Is Always Reported".
- [ ] `src/timing.{c,h}` — monotonic clock, one struct carrying load / prefill /
      decode / TTFT / token counts, filled by the engine and read by all three
      front ends. One implementation, three consumers
- [ ] **TTFT stamped at the first token callback**, never reconstructed from a
      total at the end (they differ, and the difference is the thing we care about)
- [ ] Separate prefill t/s from decode t/s — never quote one for the other
- [ ] CLI summary line to **stderr** so stdout stays pipeable into `mynah-tts`;
      `--quiet` suppresses it, nothing else does
- [ ] Server: extend `usage` with `prefill_tok_s`, `decode_tok_s`, `ttft_ms`,
      `load_ms`; on SSE they ride in the final event
- [ ] `GET /health` reports recent decode t/s (regressions show up in
      production, not only in a benchmark)
- [ ] Record whether weights were on the NAS or local in every benchmark line —
      SMB dominates cold load
- [ ] Test: assert the reported token counts match what was actually generated
      (a t/s number nobody can check is a t/s number nobody should trust)

---

## M3 — CLI (v0.1)

- [ ] `cli/main.c` — subcommands `run` (default), `chat`, `summarize`, `tokenize`, `bench`
- [ ] Flags: `-m/--model`, `-p/--prompt`, `-f/--file`, `--stdin`, `-n`, `--temp`,
      `--top-p`, `--top-k`, `--min-p`, `--seed`, `--threads`, `--ctx`, `--system`
- [ ] Streaming to stdout by default; `--no-stream` buffers
- [ ] `--think off|low|on`
- [ ] Pipe-friendly: clean stdout (text only), diagnostics to stderr, correct exit codes
- [ ] End-to-end check: `mynah-asr transcribe ... | mynah-slm summarize ...`
- [ ] Load/inference timings line, `mynah-asr`-style

---

## M4 — HTTP server (v0.1)

- [x] `server/` — sockets only, no framework, no libcurl, no TLS (a reverse
      proxy does TLS better than we would)
- [x] `server/json.{c,h}` — just enough JSON for the request shapes, with
      surrogate-pair joining so emoji survive the round trip
- [x] `GET /health` — model, threads, and **rolling decode tok/s**, so a
      regression shows up in production and not only in a benchmark
- [x] `GET /v1/models`
- [x] `POST /v1/chat/completions`, non-streaming and `stream: true` → **SSE**,
      one frame per token, `TCP_NODELAY` + `X-Accel-Buffering: no` (Nagle and
      proxy buffering each defeat streaming on their own)
- [x] `POST /v1/tokenize`, with `parse_special` OFF — caller content that looks
      like a control token stays text
- [x] Timings ride in `usage`: `load_ms`, `ttft_ms`, `prefill_tok_s`,
      `decode_tok_s`, `threads`; on SSE they arrive in the final event
- [x] Reasoning is discarded, never placed in `content`
- [x] **Inference serialized** (one at a time, all threads) rather than N
      concurrent with 1/N threads. The work is bandwidth-bound, so splitting
      cores does not raise aggregate throughput — it just makes every request
      slower and multiplies KV footprint. Queueing is latency, never an error
- [x] `tests/test_server.sh` — shape, determinism (sequential AND concurrent),
      SSE framing, streamed-equals-non-streamed, control-token forging
- [ ] `POST /v1/completions`, `POST /v1/responses`
- [ ] Optional NDJSON stream mode; WebSocket left as a stretch
- [ ] Test: `openai` Python client talks to it unmodified

---

## M5 — Tool calling & structured output (v0.1)

- [x] `src/tools.c` — schemas in, calls out. The JSON reader is the server's,
      **moved to `src/json.c`** rather than copied: two parsers is one more
      than the number of parsers that can be right — 2026-08-07
- [x] Schema injection into the prompt (Qwen `# Tools` + `<tool_call>`), in
      `src/template.c` because the wording IS the chat template
- [x] Streaming-safe detection: `<tool_call>`/`</tool_call>` are single tokens,
      so `generate.c` routes them to their own channel exactly like `<think>`.
      The JSON never touches the answer channel, so a TTS stage cannot speak it
- [x] `mynah_slm_tool_call` + `mynah_slm_tool_set`, and the OpenAI wire format
      in both directions (`_to_json` / `_from_json`)
- [x] Second-turn continuation: assistant `tool_calls` and `tool` results are
      replayed into the prompt; consecutive tool results become ONE user turn
      with several `<tool_response>` blocks, per the template
- [x] `mynah_slm_gen_params_init()` — the marker fields default to **-1**, not
      0. 0 is a real token id, and the failure would be silent text loss
- [x] Server: `tools` and `tool_choice` wired through; `finish_reason:
      "tool_calls"`, `content: null`, one whole call per SSE delta.
      `"required"` is accepted but NOT enforced — that needs the grammar, and
      claiming it without one is a lie a client finds out in production
- [x] CLI `--tools tools.json`; a call is one JSON line on stdout, malformed
      blocks counted on stderr
- [x] `tests/test_tools.c` — 47 checks, **no checkpoint**: ten renderings byte-
      identical to HF `apply_chat_template` (fixture from
      `tools/gen_chat_fixture.py`), plus the parser against truncated JSON, a
      missing name, scalar arguments, braces inside strings and prose
- [x] `tests/test_server.sh` — the wire format end to end
- [x] `tools/eval/tool_calls.py` — 30 prompts × 7 languages, scoring the
      DECISION (call / don't call), the function name and the arguments
      separately. Half the suite must NOT produce a call
- [x] **Measured on Qwen3-0.6B-Q4_K_M (2026-08-07): 21/30 with thinking off,
      26/30 with it on.** Thinking is not a preference here, it is the feature
      working: with it off, 8 of the 9 misses were the model inventing the
      weather while holding a `get_weather` function. Syntax was never the
      problem — **60/60 turns produced parseable JSON** and no invented
      function names, which is also why `strict` mode is worth less than it
      looks. Full write-up in docs/tools.md
- [ ] Modes `off` / `emit` / `strict` (`off` and `emit` exist as
      "were tools passed"; `strict` needs the grammar below)
- [ ] `src/grammar.c` — GBNF-lite constrained decoding; JSON mode and
      JSON-Schema mode built on it. **Deprioritized by the measurement above**:
      it fixes malformed output, and this model's output was never malformed.
      Still wanted for `response_format`/JSON mode, where the caller needs a
      guarantee rather than a rate
- [ ] Server: `response_format` wired through (needs the grammar)
- [ ] `docs/tools.md` results table refreshed whenever the model or the
      template changes

---

## M5b — Quantization & footprint (small is the point)

- [ ] Multilingual + tool-call **regression suite** that gates every quant claim:
      N prompts × 10 languages × (thinking on/off) × (tools on/off), scored
      automatically. This must exist *before* any sub-4-bit experiment
- [ ] Baseline the suite at F16 and Q8_0 — the reference to degrade from
- [ ] Measure `Q4_K_M` (Qwen3) and `Q4_0` (Gemma 4 QAT): score, RAM, tokens/s
- [ ] Push below 4 bits and record where it breaks: `Q3_K` (3.44 bpw),
      **`Q2_K` (2.63, has a SIMD kernel)**, `IQ3_XXS` (3.06), `IQ2_S` (2.56),
      `IQ2_XXS` (2.06), `IQ1_S` (1.56), `TQ1_0`/`TQ2_0` (ternary)
- [ ] Confirm QAT-at-Q4_0 really beats post-training Q4_K at equal size for
      Gemma 4 — that is the whole reason `Q4_0` is our default there
- [ ] Mixed precision by role: quantize only the large linears; norms, biases,
      embeddings and Gemma's per-layer embedding tables stay f32
- [ ] bf16 KV cache vs f32: measure the quality delta on long conversations
      (expected: none) and the memory saving (expected: half)
- [ ] Binary size budget: report stripped size of CLI + server, and check what
      `-DINGOT_NO_KERNELS` would cost us (probably not usable — we need them)
- [ ] `docs/quantization.md` — the measured table, with the default per model
      and the honest statement of where each one degrades

---

## M6 — v0.1 release

- [ ] `docs/models.md` — Qwen3 0.6B/1.7B/4B, verified configs, quant sizes, RAM
- [ ] `docs/api.md`, `docs/server.md`, `docs/tools.md`
- [ ] Benchmarks: tokens/s prefill + decode on M-series and on x86, per quant
- [ ] Multilingual quality spot-check in 10 languages, committed as fixtures
- [ ] README with real terminal output
- [ ] Tag `v0.1.0`, release notes, binaries from CI

---

## M6b — Candidate trial: LFM2.5-2.6B (opt-in, never a default)

Granite 350m is **closed** (measured 2026-08-07: better at tools, worse at
Italian, Q8_0 is the rung). Next candidate, surveyed in PLAN.md 2026-08-08:
LiquidAI **LFM2.5-2.6B** — 22 short-conv + 8 GQA layers, 16 languages incl.
Italian, agentic-first, **LFM Open License v1.0** (commercial only under $10M
revenue) so it is an opt-in target and never a bundled default.

**CLOSED 2026-08-08: NO-GO.** Measured, written up in docs/models.md, and the
weights are not a recommendation. The engine keeps `lfm2` support — the hybrid
layer map, the depthwise FIR, the per-kind state slots and `json_object_at()`
are what any second hybrid family needs, and M10's two candidates are both
hybrids. The one number never obtained is the **tool-call score**: the eval was
stopped mid-run, so it is unknown rather than bad. That is the first thing to
measure if this is ever reopened, and it is cheap now.

**Everything cheap comes first — no forward-pass code until the model earns it.**

### 6b.1 Zero-code screening (nothing but `inspect` and the eval tools)
- [x] Download `LiquidAI/LFM2.5-2.6B-GGUF` **to the NAS** (`lfm2.5-2.6b-q4`) —
      2026-08-08, 1.67 GB. Q8_0 not pulled yet; not needed before a GO.
- [x] `mynah-slm inspect` the Q4_K_M — 2026-08-08. **ingot reads it with zero
      changes**: `arch lfm2`, 266 tensors, 4.94 bits/weight, Q4_K 148 / Q6_K 19
      / F32 99. Nothing to fix upstream. Conv tensors are
      `blk.N.shortconv.{in_proj,conv,out_proj}`, the FIR itself F32 `[2048 x 3]`.
- [x] Confirm the 22:8 split is in GGUF metadata KV — **it is**, as
      `lfm2.attention.head_count_kv = [i32 x 30]`, 0 on conv and 8 on attention.
      Cross-checked against `layer_types` in config.json and against which
      tensors each block carries: three sources, same answer. The map is
      **irregular** (attention at 2,5,9,13,17,21,24,27), so there was never a
      ratio to hardcode. All of it written up in **docs/lfm2-arch.md**.
- [x] Establish where the engine stops today — one line, and the right one:
      `head_count_kv varies per layer (0 at 0, 8 at 2) - unsupported`.
      `head_dim` is the second wall: LFM2 publishes neither `key_length` nor
      `rope.dimension_count`, so it has to fall back to `d_model / n_heads`.
- [ ] ~~`tools/eval/lang_ppl.py` vs Qwen3-0.6B and granite~~ — **cannot run
      yet, and the task as written was wrong.** `lang_ppl.py` and
      `tool_calls.py` both drive `mynah-slm` (the C engine) by subprocess, and
      the numpy oracle is Qwen3-only (`tools/oracle/model.py`). There is no
      zero-code path to a quality number for LFM2: the gate needs EITHER a
      transformers/torch screening script (new, ~40 lines, `check` extra
      already declares the deps) OR the C engine to run LFM2 first, which is
      6b.2. Decide before spending either.
- [x] GO/NO-GO gate — **NO-GO, 2026-08-08.** Italian measures **1.9589** against
      Qwen3-0.6B's 1.6530: 18.5% WORSE, at 4.4x the file. It loses 12 of 16
      languages to a model a quarter its size, and wins only where Liquid says
      it is tuned (en, ja, plus el and a hair of de). Full table and the
      reasoning in **docs/models.md**. 6b.2 and 6b.3 below are therefore closed
      as not-to-be-done, EXCEPT that they were already done in passing — see
      the note under M6b.

### 6b.2 The short conv kernel (only if 6b.1 says GO)
- [ ] `docs/lfm2-arch.md` first — shapes, forward pass, trap list, exactly as
      `docs/qwen3-arch.md` did for Qwen3
- [ ] Extend the numpy oracle to LFM2; validate vs `transformers` before any C
- [ ] `src/kernels.c`: depthwise causal FIR, L=3, per-channel state.
      **Prefill and decode through one code path** (rule 2) — the conv is a
      sliding window at decode and a full causal conv at prefill
- [ ] Hybrid state: a 2048×3 conv state per conv layer + KV cache on the 8
      attention layers only. Allocate 8 caches, not 30
- [ ] Per-stage oracle parity, then greedy parity on 20 prompts
- [ ] RoPE theta 1e7 on the attention layers; tied embeddings = LM head

### 6b.3 Behaviour and the verdict
- [ ] Chat template: `<|im_start|>` / `<|im_end|>` / `<|startoftext|>`, `<think>`
- [ ] **Pythonic tool calls**: `<|tool_call_start|>[f(a=1)]<|tool_call_end|>`.
      Score BOTH the native Pythonic format and the JSON-via-system-prompt
      override with `tools/eval/tool_calls.py` — if JSON mode costs accuracy,
      `src/tools.c` gets a second parser rather than the model getting a detour
- [ ] Speed on the M1, 8 threads, staged locally, stated per CLAUDE.md
- [ ] Write the verdict into docs/models.md **with the license terms next to
      the numbers**, and keep the download opt-in

---

## M7 — Gemma 4 (v0.2)

### 7.1 De-risking
- [ ] Download `google/gemma-4-E2B-it-qat-q4_0-gguf` and the `-it` safetensors
- [ ] `docs/gemma4-arch.md` — full verified architecture and the trap list:
      PLE (`hidden_size_per_layer_input: 256`), MQA (1 KV head), the 4:1
      sliding/global `layer_types` pattern, `sliding_window: 512`,
      `num_kv_shared_layers: 20`, dual RoPE (1e4 sliding / 1e6 global),
      `rope_type: proportional` + `partial_rotary_factor: 0.25` on global layers,
      `use_double_wide_mlp`, `final_logit_softcapping: 30.0`,
      `gelu_pytorch_tanh`, tied embeddings, eos `[1, 106]`
- [ ] Confirm which tensors belong to the vision/audio towers and are skipped
- [ ] Extend the numpy oracle to Gemma 4 text; validate vs `transformers`

### 7.2 Implementation
- [ ] SentencePiece-style tokenizer for the 262144 vocab (from GGUF)
- [ ] Per-Layer Embeddings: load, lookup, and injection into each layer's input
- [ ] Hybrid attention: per-layer mask + per-layer RoPE config from `layer_types`
- [ ] Sliding-window KV cache (512) alongside full-attention caches
- [ ] KV sharing across the last 20 layers — allocate physical caches, not 35
- [ ] Partial rotary (25%) + proportional scaling on global layers
- [ ] Double-wide MLP, `gelu_pytorch_tanh`, final logit softcapping
- [ ] Layer-by-layer oracle validation, then greedy parity on 20 prompts

### 7.3 Behaviour
- [ ] Gemma 4 chat template + `<|think|>` system-prompt trigger
- [ ] Native tool-call format → normalized to the same public struct
- [ ] `--think off|low|on` mapped onto Gemma semantics; thinking-token budget for `low`
- [ ] Summarization presets (transcript cleanup, meeting summary, bullets, title)
- [ ] Multilingual regression suite, 15 languages
- [ ] Make Gemma 4 E2B-it QAT the README's recommended default

### 7.4 v0.2 release
- [ ] RAM/latency table: E2B QAT Q4_0 vs Qwen3-1.7B Q4_K_M on the same machine
- [ ] Tag `v0.2.0`

---

## M8 — Performance (v0.3)

- [ ] Prompt cache: persist and reload the KV of a fixed system prompt
- [ ] Shared prefix cache across sessions (the server's biggest win)
- [ ] Sliding-window KV compaction / eviction
- [ ] Speculative decoding: Qwen3-0.6B drafting for Qwen3-4B; verify identical output
- [ ] Startup time: lazy dequant, `mmap` prefault tuning via ingot
- [ ] Kernel tuning: NEON / AVX2 / AVX-512-VNNI, upstream anything reusable to ingot
- [ ] Optional Metal backend
- [ ] Optional CUDA backend
- [ ] Optional continuous batching in the server
- [ ] Tag `v0.3.0`

---

## M9 — Ecosystem

- [ ] `docs/pipeline.md` — the full ASR→SLM→TTS recipe, CLI and HTTP variants
- [ ] End-to-end demo script: mic → `mynah-asr` → `mynah-slm` → `mynah-tts` → speaker,
      measuring end-to-end latency with streaming on at every stage
- [ ] Python (ctypes) and Node (koffi) bindings over `libmynah_slm`, matching
      the `mynah-asr` binding style
- [ ] Integration hooks for `mynah-app`
- [ ] Homebrew formula / release artifacts

---

## M10 — Next-gen architectures (v0.4, evaluated later)

- [ ] Feasibility study: Qwen3.5 Gated DeltaNet (linear attention, recurrent
      delta-rule state) + sparse MoE routing in a hand-written C engine
- [ ] Same study for **Granite 4.0 `h-350m`** (Apache 2.0, 340M, 4 attention +
      28 **Mamba-2** layers, 32K ctx, 12 languages incl. Italian, `<tool_call>`
      format identical to Qwen's — `src/tools.c` parses it unchanged).
      Selective scan + causal conv1d + per-head state is the same *kind* of
      kernel as DeltaNet: cost one, get most of the other
- [ ] Before either: score the real checkpoint with `tools/eval/tool_calls.py`.
      A 340M model that cannot pick the right function in Italian does not
      justify a new kernel family, whatever the model card says
- [ ] Decide GO/NO-GO against the alternative of just tracking Gemma/Qwen dense releases
- [ ] If GO: state kernels → MoE router → Qwen3.5-0.8B parity → 2B / 4B

### Not v0.4: `granite-4.0-micro` (3B dense)
- [ ] Optional, portable **now** — GQA + RoPE + SwiGLU + RMSNorm + tied
      embeddings is `arch_qwen3.c` minus QK-norm. ~1.9 GB at Q4 and decode is
      bandwidth-bound, so expect ~1/5 of today's tok/s. A quality option for a
      laptop, never the small-and-fast default. Revisit if a user asks for
      quality over footprint
