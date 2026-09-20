# mynah-slm — repo contracts

## What this project is
Pure-C11 SLM inference engine (llama.cpp style, but ours) for small multilingual
instruct models. CPU-first, streaming, tool calling, thinking on/off.
v0.1 target: **Qwen3-0.6B**. v0.2 target: **Gemma 4 E2B-it QAT Q4_0**.
Plan in PLAN.md, task detail in `.work/`, reuse map in docs/prior-art.md.
**When docs and code diverge, trust the code.**

## Map
- `src/` — libmynah_slm (C11). One module = one small, cohesive .c+.h.
  NO monolithic files, NO empty placeholder files with names that lie.
- `cli/main.c` — `mynah-slm` CLI. `server/` — `mynah-slm-server`.
- `tools/` — Python managed with **uv** (`uv run`, `uv sync`). Offline only:
  converter, oracle, eval. **Never required at runtime.**
- `reference/<model>/` — configs/tokenizers extracted from checkpoints (committed).
- `models/` — **symlink to the NAS** (see below). GITIGNORED, never commit.
- `models-local/` — one-at-a-time local staging for measurement. GITIGNORED.
- `third_party/ingot/` — git subtree. GGUF + safetensors. `make update-ingot`.
- `PLAN.md` — the **board**: one line per work item, linking to `.work/`.
- `.work/` — one detail note per board item. See `.work/README.md`.
- `docs/qwen3-arch.md`, `docs/gemma4-arch.md` — THE reference for each
  implementation; every numeric value comes from there or from the model config.
- `docs/prior-art.md` — what we port and from which sibling project.
- `docs/models.md`, `docs/perf.md` — durable measured results.

## How the plan is organised

`PLAN.md` is a **board, not a log**: one line per work item, with a link to the
note under `.work/` that carries the detail. Write the note *before* starting —
problem, evidence, plan, acceptance gate — and read it when picking the item up.
Session checkpoints, in-flight measurements, post-mortems and rejected ideas live
in `.work/`; durable measured results live in `docs/models.md` and
`docs/perf.md`. Update `PLAN.md` itself only when a decision or a milestone
genuinely changes.

Read [`.work/engineering-method.md`](.work/engineering-method.md) before starting
anything. It is the short version of how this project avoids fooling itself:
cost model before code, every tool declares a refusal, a benchmark is invalid
until dispatch is proven, and the completion rule.

`tools/check_plan.py` enforces the link contract between `PLAN.md` and `.work/`.

## Weights live on the NAS, never on the laptop

The M1's internal disk is chronically full (**~19 GB free**, 228 GB total). LLM
checkpoints are not allowed to live there.

```
/Volumes/shared/<project>/models/     ← the NAS  (SMB, ~555 GB free)
./models -> /Volumes/shared/mynah-slm/models
```

This repo's `models/` is a **symlink to `/Volumes/shared/mynah-slm/models`**.
Same convention already in use by `mynah-asr`, `mynah-tts`, `qwen-tts`,
`keyra-models` (see qwen-tts/.gitignore: "no trailing slash — these may be
symlinks to the NAS copy, not real dirs").

Rules:
- Download and keep every `.gguf` / `.safetensors` on the NAS. It is the
  library. `scripts/download_model.sh` writes there by default.
- **Never measure from the NAS.** Stage one model locally first:
  `scripts/use_model.sh <name.gguf>` → `models-local/`, one at a time,
  `--evict` when done. Measured on Qwen3-0.6B-Q4_K_M, full weight read +
  dequant with a warm page cache: **NAS 17.0 s vs local 7.5 s**. Dequant is the
  same CPU work on both sides, so the I/O part of that gap is ~0.5 s against
  ~10 s — roughly **20×**.
  It is not only load time: this engine multiplies straight off the mmap'd
  quantized bytes, so weights are walked again every decode step. Warm they
  come from the page cache; the moment anything evicts them — a second model,
  the ASR and TTS stages beside us, a long prompt — the re-faults come back
  over SMB mid-measurement, at random. On 16 GB of RAM that is not
  hypothetical. **A tok/s number you cannot reproduce is not a measurement.**
- Every benchmark line states quant, threads, machine and **NAS or local**.
- **Gotcha:** `mkdir -p` fails on the smbfs mount root with
  `Authentication error`. Use plain `mkdir` one level at a time.
- **The SMB mount drops.** It has gone into `Authentication error` (reads fail
  while `mount` still lists it) three times in one session. Tools that read
  `models/` must fail with a clear "is the NAS mounted?" message, never a raw
  `FileNotFoundError`. Remounting needs Finder / the user's credentials.

## Speed is always reported — not optional, not behind a flag

Every delivery mode reports its own speed. A run that does not say how fast it
was cannot be compared, and "fast enough on a CPU" is this project's entire
claim.

- **CLI**: one summary line to **stderr** after every run (stdout stays clean so
  `mynah-slm ... | mynah-tts` works). `--quiet` is the only thing that hides it.
  `[load 0.21s | prompt 42 tok, prefill 310 tok/s | gen 128 tok, decode 47.3 tok/s | TTFT 138 ms | 8 threads]`
- **Streaming**: same line after the last token, with **TTFT measured at the
  first token**, never reconstructed at the end.
- **Server**: timings ride in the response (`usage` + `prefill_tok_s`,
  `decode_tok_s`, `ttft_ms`, `load_ms`); on SSE they arrive in the final event.
  `/health` exposes recent decode t/s so a regression shows up in production.
- **decode tok/s is the headline.** It is memory-bandwidth bound and it is what
  a user feels. Prefill t/s is a different number and must never be quoted in
  its place.
- Benchmarks state quant, threads, machine, and **whether the weights were on
  the NAS or local** — SMB dominates cold load and will otherwise flatter or
  ruin a measurement at random.

## Quantization policy — small is the point

This is a **mini engine**: the default must be tiny, not maximal quality.

- Default distribution format: **GGUF Q4** (`Q4_K_M` where a community quant
  exists; **`Q4_0`** for Gemma 4, because that is what Google's QAT checkpoint
  ships and QAT-at-Q4_0 beats post-training Q4_K at the same size).
- **Go below 4 bits when quality holds.** ingot decodes `Q3_K` (3.44 bpw),
  `Q2_K` (2.63, **SIMD kernel**), the whole `IQ1`–`IQ4` codebook family
  (`IQ2_XXS` 2.06, `IQ1_S` 1.56) and ternary `TQ1_0`/`TQ2_0`. Every sub-4-bit
  candidate must be gated by a **measured** multilingual + tool-call
  regression, never by vibes.
- **Quote bits/weight over the whole file, never the type's block rate.** A
  `Q4_K_M` build of Qwen3-0.6B is **5.24** bits/weight, a "2-bit" one is 4.02
  and a "3-bit" one is 4.65, because a 0.6B model with a 151936-entry vocabulary
  is dominated by things that are not 4-bit. A compression claim that ignores
  the embedding is not a claim about the file on disk.
- Only the large linears get quantized. Norms, biases and embeddings stay f32
  (mynah-asr `qmat.h` policy).
- KV cache in **bf16**, not f32 (qwen-tts precedent: halves KV memory, no
  measurable quality cost).

## Rules
1. **Config-driven**: no architecture constant in a `#define`. Layers, head dims,
   sliding windows, RoPE thetas, vocab sizes all come from GGUF metadata KV or
   `config.json`. (keyra's `gemma3.c` hardcodes them — do NOT copy that part.)
2. **A single code path** for prefill and decode wherever possible
   (qwen-asr lesson: divergent paths breed bugs).
3. Every new numeric stage must be **validated against the Python oracle**
   (`tools/oracle/`) before building on top of it. Per-stage tolerances, not hashes.
4. **ingot READS, we COMPUTE.** `../ingot` is the same team's repo
   (`mynah-org/ingot`, MIT) and its job is the container: mapping a GGUF,
   block geometry, and decoding a block to f32 bit-for-bit against llama.cpp.
   A reader bug, a missing format, a wrong geometry — fix it THERE. Workflow:
   edit `../ingot`, run its own `make test` (the geometry and llama.cpp
   cross-checks are the safety net), commit upstream, then `make update-ingot`
   here. `third_party/ingot` is a subtree, so never edit that copy directly —
   the next `update-ingot` would silently discard it.

   **New kernels go in `src/`, not upstream.** A matrix product is not a
   container concern: it depends on our shapes, our thread pool, our prefill
   batch width, our KV layout and our fusions. A kernel written to be generic
   cannot specialize, and specializing is the entire reason to have our own
   engine rather than linking one. `src/qmat.c` owns the product dispatch,
   `src/kernels.c` the rest. ingot's kernels stay as the fallback for types we
   have not hand-written, and anything written here has to beat them in a
   measured A/B before it replaces them.

   The concrete case that settled it: ingot's batched Q4_K quantizes
   ACTIVATIONS to int8 from two tokens up (2.4e-3 relative) — a perfectly good
   default for a general-purpose library, and 20x our parity gate. Prefill has
   to agree with decode, so the batched product is ours (docs/perf.md).
5. **Port, don't invent.** Check docs/prior-art.md first: qwen-tts already has a
   validated Qwen3 decoder and the whole SIMD kernel set; keyra has the Gemma
   trap list; mynah-asr has the server, threads and the qmat dispatch policy
   (f32 → sgemm; quantized + one token → direct dot; quantized + many tokens →
   dequantize a strip + sgemm) that `src/qmat.c` now implements.
6. No build artifacts in the repo (`.o`, binaries, weights).
7. Small, frequent commits. **Everything written into the repo or published from
   it is in English, always**: commit messages, tag annotations, code comments,
   CI workflows, docs, GitHub release notes. Chat with the user can be in
   Italian; the repo cannot.
8. `PLAN.md` is a board and `.work/` holds the detail — see "How the plan is
   organised" above. **Both are tracked.** `CLAUDE.md` is a symlink to this file
   and is the only planning document that stays out of git.
9. For A/B tests always use identical explicit parameters (model, quant, seed,
   threads, prompt).
10. **A task is complete only when the report says**: WHAT CHANGED · WHAT PATH
    ACTUALLY RAN · WHAT WAS MEASURED · WHAT REMAINS UNKNOWN · VERDICT
    (PROMOTE / KEEP / INCONCLUSIVE / REJECT).

## Build & test
- `make` — CLI. `make server`, `make lib`, `make clean`.
- `make test` — per-stage parity vs oracle + e2e (exit 77 = skip without model).
- **Memory/UB on macOS: `make leaks` (native, fast) + `make ubsan` (low
  overhead). NEVER ASan on Mac** (extremely slow, hangs with large models);
  `make asan` exists only for the Linux CI. Same pattern as mynah-asr/qwen-tts.
- `make bench` — tokens/s prefill + decode.
- Any gate that decides something gets a `make clean` first. Apple's GNU Make
  3.81 compares file times in **whole seconds**, so a same-second edit is
  silently skipped and the binary keeps the old code with the new source on
  disk. It can make a gate fail that should pass, and — worse, because it is
  silent — pass one that should fail.

## Known implementation traps
### Qwen3 (v0.1)
- **NeoX split-half RoPE, NOT interleaved.** `qwen_tts_kernels.h` ships both —
  grab `apply_rope_neox_inplace` from `qwen_tts_talker.c`, not
  `qwen_apply_rope_interleaved` (that one belongs to a different sub-model).
- Per-head **QK-RMSNorm** (= `qwen_rms_norm_per_head`).
- `head_dim` 128 with `hidden_size` 1024 → `n_heads * head_dim != hidden_size`.
  The projections are not square. Never derive head_dim from hidden/heads.
- `tie_word_embeddings: true` — one tensor, two uses. `token_embd.weight` is the
  input lookup **and** the output projection: the largest tensor in the file
  (32.7% of the Q4_K_M build), the hottest GEMV in the decode loop, and directly
  in front of the softmax. It is not a lookup table and must not be quantized
  like one.

### Gemma 4 (v0.2) — from keyra/src/core/gemma3.c, still live
- **RMSNorm scales by `(1 + weight)`** and normalises in f32.
- **Embeddings multiplied by `sqrt(hidden_size)`.**
- Four norms per layer (pre/post around BOTH attention and FFN), not two.
- **Two RoPE bases in one model**: 1e4 on sliding layers, 1e6 on global. RoPE
  scaling applies to the **global layers only** — get this wrong and layers 0–4
  stay bit-exact while everything from layer 5 on is garbage.
- Gemma 4 adds `rope_type: proportional` + `partial_rotary_factor: 0.25` on the
  global layers only.
- MLP is **GeGLU with tanh-approximated GELU**, not SwiGLU. `use_double_wide_mlp`.
- Layer pattern is in `layer_types`: E2B is **4 sliding : 1 full** (Gemma 3 was
  5:1). Read the array, don't hardcode the ratio.
- **MQA: 1 KV head.** `num_kv_shared_layers: 20` → fewer physical KV caches than
  the 35 layers.
- Per-Layer Embeddings: `hidden_size_per_layer_input: 256`, added into each
  decoder layer's input.
- `final_logit_softcapping: 30.0`. Two EOS ids: `[1, 106]`.

### ingot coverage (checked 2026-08-06 against docs/QUANTS.md)
- **Reading is a solved problem.** Every ggml block type has a geometry and all
  but one decode bit-for-bit vs llama.cpp. `Q4_K_M` is a *recipe*, not a type —
  a `Q4_K_M` file is `Q4_K` + `Q6_K` + `F32` inside, all supported. The Gemma 4
  QAT checkpoint is plain **`Q4_0` (id 2): decode ✅, encode ✅**.
- **The gap is speed, not support.** `Q4_0`'s kernel in ingot is *generic*
  (decode a row, then multiply) — the SIMD kernels exist for `Q8_0`, `Q2_K`,
  `Q3_K`, `Q4_K`, `Q5_K`, `Q6_K` only. Since Gemma 4 QAT *is* `Q4_0`, our
  default production model would run on the slow path.
  **Fix: port qwen-tts's NEON `qwen_matvec_q4_0` / `qwen_matmat_q4_0` upstream
  into ingot** — it already uses the identical 18-byte block layout
  (fp16 scale + 16 nibble-pairs, "same layout as llama.cpp q4_0").
- **x86 coverage was far thinner, and we closed most of it** (2026-08-07).
  Added upstream: `Q6_K` AVX2 (`fee653e`, 4.65x on an LM head), `Q8_0` fused
  NEON+AVX2 (`5379a13`, it had NO vector path anywhere — 2.84x on the head,
  +61% end-to-end), `Q5_K` distribute-the-sum NEON + first-ever AVX2
  (+26 to +51% per tensor, +12% end-to-end).
  Then `Q4_K` distribute-the-sum both forms, and `Q2_K`/`Q3_K` fused matvecs
  both forms — those two had NO kernel on EITHER architecture (1.8x per tensor,
  ~1.3x end-to-end). **Every ggml K-quant in ingot now has a fused matvec on
  both architectures**; the scratch round-trip is gone from the library.
- **Sub-4-bit gate, answered 2026-08-07 on granite-350m**: `Q2_K` is DESTROYED
  (ppl 26400, bits/byte 3.38 vs Q8_0's 1.59, word-salad output) — not a
  candidate. `Q3_K_M` survives but costs 8% of bits/byte to save 27 MB, a poor
  trade at this size. Note "2-bit" is really **4.02** bits/weight here and
  "3-bit" 4.65 — same vocabulary-dominated effect as Qwen3's 5.24.
- **ingot's own Q4_K NEON/AVX2 kernels still use the naive per-element form**
  (multiply by d*scale, subtract dmin*min, per element) — which is exactly why
  `src/qmat.c`'s Q4_K beats them 1.3-1.7x. Distributing the sum there the way
  Q5_K now does would close most of that gap for every other ingot consumer.
  We keep ours because of the cross-row `SUM x` hoist, which ingot's per-call
  API cannot do; the per-element part is fair game upstream.
- **Rosetta lies about CPUID**: it executes AVX2 but does not advertise it, so
  ingot's runtime dispatch picked scalar and its whole x86 suite passed without
  running a single AVX2 kernel. `INGOT_CAPS_ASSUME=avx2` (added upstream) trusts
  the build instead; `make test-x86-rosetta` sets it. Before believing any x86
  measurement here, check that the path was actually taken.
- **Our Q6_K kernel was written, measured, upstreamed, then deleted** — a tie on
  ARM, and once ingot had the AVX2 twin its version measured *faster* than ours.
  A kernel with no per-call specialization belongs upstream; that is rule 4
  working, not an exception to it. See docs/perf.md.
- The IQ / TQ / MXFP4 families all decode but are all generic-kernel too. Fine
  for an experiment, measure before shipping one as a default.
- The single unsupported type is **`Q1_0` (id 41)** — llama.cpp's own reference
  package has no decoder for it either, so nothing can validate one. It still
  opens and reports the type by name. Not a practical problem.

### General
- Never `expf(large positive)`: under `-ffast-math` the `inf` is UB, gcc/x86
  vectorizes through libmvec and it becomes NaN. Use the stable form.
  (mynah-asr CI, 2026-07-18: encoder NaNs on linux/x86 only.)
- Call `ftz_on()` on **every** compute thread, pool workers included.
- 64-byte-align every SIMD/BLAS buffer.
- Zero allocation inside the token loop.
