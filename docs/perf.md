# Performance — measured, not estimated

All numbers from an M-series Mac, weights **staged locally** (`scripts/use_model.sh`),
`Qwen3-0.6B-Q4_K_M.gguf`, single-threaded, `-O3 -march=native`, no `-ffast-math`.

Reproduce with `mynah-slm run ...`; every run prints its own line to stderr.

---

## Where it stands: v0.1, correct and slow

```
$ mynah-slm run -m models-local/Qwen3-0.6B-Q4_K_M.gguf -p "Ciao! Come stai?" -n 40 --temp 0
Ciao! Sto bene! Cosa ne hai?
[load 0.04s | prompt 19 tok, prefill 3.5 tok/s | gen 12 tok, decode 4.1 tok/s | TTFT 5663 ms | 1 threads]
```

**4.1 tok/s decode.** That is roughly 30x off what this model should do on this
machine, and it is stated here rather than buried because the engine's whole
claim is that CPU is enough. Correctness came first on purpose — the forward
pass matches the numpy oracle at 1e-6 — but this number is not shippable.

The cause is not mysterious, and it was measured rather than guessed.

## Where the time goes

Per-matvec, averaged over repeated calls, weights warm:

| tensor | type | shape | per call | throughput |
|---|---|---|---|---|
| `attn_q` | Q4_K | 2048 x 1024 | 0.65 ms | 3.20 G elem/s |
| `ffn_up` | Q4_K | 3072 x 1024 | 0.87 ms | 3.60 G elem/s |
| `ffn_down` | Q6_K | 1024 x 3072 | 2.47 ms | 1.27 G elem/s |
| **`lm_head`** (= `token_embd`, tied) | Q6_K | **151936 x 1024** | **126.75 ms** | 1.23 G elem/s |

Adding it up: ~6.2 ms per layer x 28 layers = ~173 ms, plus ~127 ms for the LM
head, gives ~300 ms per token — which is the 4.1 tok/s observed, so nothing
significant is hiding elsewhere.

Two findings, both actionable:

### 1. The LM head is 42% of a decode step, on its own

One matvec, 151936 x 1024, every single token. This is the concrete cost of the
tied embedding noted in `docs/models.md`: it is not a lookup table, it is the
hottest GEMV in the loop.

It also makes a quantization question urgent rather than academic. `token_embd`
is held at Q6_K by the `Q4_K_M` recipe. At Q4_K's measured throughput the same
matvec would take ~48 ms instead of ~127 ms — a 26% cut in total decode time
from one tensor. Whether that costs output quality is exactly what M5b has to
measure; it sits in front of the softmax that picks the token, so the answer is
not obvious in either direction.

### 2. ingot's Q6_K matvec is 2.8x slower per element than Q4_K

`ffn_up` (Q4_K) and `ffn_down` (Q6_K) are the same shape. Q6_K moves 1.46x the
bytes and takes 2.84x the time, so it is not bandwidth — the kernel is leaving
something on the table. `ingot_has_kernel()` reports 1 for both, so this is not
the generic fallback path either.

Since Q6_K carries **45% of a Q4_K_M checkpoint** (`docs/models.md`), that gap
is worth chasing, and it belongs **upstream in ingot** rather than here — the
same rule that sends the missing Q4_0 NEON kernel there (TASKS 0.4).

## What has not been done yet

- **Threading.** Everything above is one core. `ingot_matvec` is single-threaded
  by design; `ingot_matmat` parallelizes over rows. The thread pool is the
  single largest win available and has not been written.
- **Batched prefill.** Prefill currently runs the same one-token path as decode
  (deliberately — see `arch_qwen3.c`), so it gets no batching benefit. Moving it
  to `ingot_matmat` should help substantially, with the caveat in ingot's
  precision contract: from two tokens up, Q4_K/Q5_K batched matmat quantizes
  activations to int8 by default (rel error ~2.4e-3). That would blow the
  parity gate's 1e-4 tolerance at layer 0, so the parity path must keep using
  the exact twins or `INGOT_SDOT=0`. **Not yet wired** — noting it before it
  becomes a confusing test failure.
- **SIMD in our own kernels.** RMSNorm, RoPE, SwiGLU and attention are scalar.
  They are not the bottleneck at these sizes, but they will matter once the
  matvecs get faster.

## Method notes

- Weights are staged locally for every measurement. Off the NAS the same full
  weight read takes 17.0 s against 7.5 s, and page-cache pressure re-faults over
  SMB mid-run — see CLAUDE.md.
- TTFT is stamped at the first token, never reconstructed from a total.
- Prefill tok/s and decode tok/s are different numbers and are never quoted for
  one another.
