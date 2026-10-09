# K6 — native BF16 / F16 weight matvec (BFDOT, VDPBF16PS): not written, and why

Status: **REJECTED 2026-10-08** (no kernel code; a roof measurement in `tests/bench_qmat k6` instead)

Item: `PLAN.md` §0 K6. Lineage: mynah-tts `.work/bf16-native-weights.md`
(E4-8: "the checkpoint is bf16, the CPU has bf16, and we use neither"), its
`src/qmat.c` bf16 kernels, and ingot's own note above `dense_at()` in
`third_party/ingot/src/kernels.c` ("No BFDOT/BFMMLA here yet: both need the
ACTIVATIONS truncated to bf16 too, which is a precision decision the consumer
should opt into").

## Task

Decide whether this engine should own a native bf16 (and f16) weight matvec
on the hosts that have the instructions — ARMv8.6 BFDOT/BFMMLA (Neoverse V2,
Apple M2+), x86 AVX512_BF16 `VDPBF16PS` (Zen 4, Sapphire Rapids) — and write
it only if the numbers justify it.

## Question

Is there decode (or prefill) time on the table for 2-byte weights that the
existing widening path cannot reach and a native bf16 dot can?

## Known facts

- **What ingot already does** (`dense_rows`, `third_party/ingot/src/kernels.c`):
  BF16 and F16 matvec/matmat with SIMD on both architectures — NEON widens
  bf16 with `vshll_n_u16(h, 16)` and f16 with `vcvt_f32_f16`; AVX2 widens bf16
  with `cvtepu16_epi32` + `slli 16` and f16 with F16C `cvtph_ps`; f32 FMA; four
  tokens per weight load in the batched form. One FMA per weight, f32
  activations, f32 accumulation. Nothing scalar is left on the hot path.
- **What we ship**: Qwen3-0.6B Q4_K_M (Q4_K + Q6_K + F32 norms), Q8_0, and the
  v0.2 target Gemma 4 QAT is Q4_0 (AGENTS.md). `docs/models.md` lists no
  BF16/F16 checkpoint; the Python oracle and eval harness use bf16 through
  PyTorch, not through this engine. A BF16 GGUF here is the quality-reference
  rung a user might load, not a product format — 4x the bytes of Q4_K_M,
  against this project's "small is the point" rule.
- **What the hosts have**: this VM (Cascade Lake) has no AVX512_BF16. Apple M1
  has no FEAT_BF16 (M2 and later do). Neoverse V2 has BFDOT/BFMMLA; Zen 4 and
  Sapphire Rapids have AVX512_BF16. So a kernel written here could be checked
  for correctness under `qemu-aarch64 -cpu max` and compiled for
  AVX512_BF16, and measured for speed on nothing available to this session.
- **The precision cost is not optional**: BFDOT and VDPBF16PS multiply bf16 by
  bf16. The activations would have to be rounded to bf16 (8 significant bits)
  per matvec — a numerical change of the same class as `--fast`, on the one
  format whose reason to exist is being the high-precision reference.

## Cost model (before any code)

| field | |
|---|---|
| current cost | ingot's widening kernel: one load + two widen + two FMA per 8 weights (NEON); decode reads 2 bytes/weight |
| suspected cause | none measured — the question is whether there IS a cost above the roof |
| proposed transformation | BFDOT (2 MACs/lane/instr) or VDPBF16PS with activations rounded to bf16 once per matvec |
| maximum plausible saving | bounded by how far the current kernel sits below the read roof: **6-13% on the head, measured below** |
| new work introduced | activation rounding pass; two ISA kernels + scalar twin + runtime checks |
| risk | precision change on the reference format; kernels nobody here can time |
| smallest experiment that kills it | stream the same bytes with no arithmetic at all, same row split, interleaved with ingot's matvec: if ingot is within ~15% of that on the bandwidth-bound shape, decode cannot gain more than that from any dot instruction |

## Evidence — measured here, CLOUD-SPECIFIC, revalidate on target

`tests/bench_qmat k6 11 1,4`: A = a streaming read of the same weight bytes
(four u64 accumulators, nothing else — the roof this process can reach with
this row split), B = `ingot_matvec` BF16 / F16, interleaved, order
alternating, 11 rounds. 4 vCPU Cascade Lake VM, gcc 13 -O3 `-march=native`,
load average 3.1-3.9 from other agents. Synthetic weights in RAM. A/B = read
time / matvec time: **1.00x means the matvec is AT the roof.**

| type | shape | MB/call | thr | read ms median (min..max) | ingot ms median (min..max) | read/ingot | direction |
|---|---|---|---|---|---|---|---|
| BF16 | 1024x1024 | 2.1 | 1 | 0.103 (0.084..0.114) | 0.120 (0.113..0.156) | 0.86x | stable (0/11) |
| BF16 | 3072x1024 | 6.3 | 1 | 0.315 (0.278..0.400) | 0.378 (0.365..0.419) | 0.83x | stable (1/11) |
| BF16 | **151936x1024** | **311.2** | 1 | 47.53 (43.89..63.83) | 50.50 (46.02..66.98) | **0.94x** | NOT stable (3/11) |
| F16 | 1024x1024 | 2.1 | 1 | 0.099 (0.098..0.104) | 0.208 (0.204..0.212) | 0.48x | stable |
| F16 | 3072x1024 | 6.3 | 1 | 0.302 (0.286..0.414) | 0.406 (0.372..0.744) | 0.74x | stable |
| F16 | **151936x1024** | **311.2** | 1 | 47.49 (42.43..65.12) | 51.02 (45.44..56.27) | **0.93x** | NOT stable (3/11) |
| BF16 | 1024x1024 | 2.1 | 4 | 0.127 | 0.166 | 0.76x | stable |
| BF16 | 3072x1024 | 6.3 | 4 | 0.306 | 0.397 | 0.77x | NOT stable |
| BF16 | **151936x1024** | **311.2** | 4 | 18.04 (16.81..20.61) | 20.63 (17.48..22.53) | **0.87x** | stable (1/11) |
| F16 | 1024x1024 | 2.1 | 4 | 0.081 | 0.125 | 0.64x | stable |
| F16 | 3072x1024 | 6.3 | 4 | 0.209 | 0.306 | 0.68x | stable |
| F16 | **151936x1024** | **311.2** | 4 | 14.11 (13.58..15.61) | 16.03 (15.04..18.01) | **0.88x** | stable (0/11) |

What it says:

- **A BF16 decode step is a DRAM stream.** Qwen3-0.6B in BF16 reads ~1.2 GB
  of weights per token; the head alone is 311 MB. On that shape ingot's
  widening kernel already runs at 0.87-0.94x of reading the bytes and doing
  nothing with them. A native dot product still has to read every byte, so
  it can recover at most the remaining 6-13% of the head, and less of the
  step. Not worth two ISA kernels and a precision change.
- The L2-resident layer shapes show more headroom (0.64-0.86x) — but on a
  real BF16 step those tensors are also streamed from DRAM (1.2 GB does not
  fit in any cache), so the head row is the representative one.
- F16 at 1 thread on 1024x1024 (0.48x) is the one row with real headroom, and
  it is ingot's single-token F16C path, not a missing instruction: worth an
  upstream look in ingot (`dense_rows` f16 single-token loop) if an F16
  checkpoint ever ships, not a native-bf16 kernel here.

## Conclusion

**REJECT — no native bf16/f16 kernel in `src/`.** Three independent reasons,
any one of which would be enough:

1. **No format we ship has 2-byte weights.** The default is Q4, the precision
   rung is Q8_0, Gemma 4's QAT is Q4_0.
2. **Decode on 2-byte weights is already at the read roof** here (0.87-0.94x
   on the 311 MB head): the ceiling for any dot instruction is ~6-13% of one
   tensor.
3. **The instruction changes the numerics** (activations rounded to bf16) of
   the format whose only purpose in this engine would be to be the reference.

Prefill, where mynah-tts found BFMMLA worth 2-4x because a 16-row batch is
compute-bound, would be a bf16 GEMM inside `mynah_slm_qmatmat`'s strip path —
K1-class work on a format we do not ship, not a matvec.

WHAT CHANGED: this note, and `tests/bench_qmat k6` (the roof measurement, so
the verdict can be re-checked on a host that has the instructions).
WHAT PATH ACTUALLY RAN: ingot `dense_rows` AVX2 (+F16C for F16), x86.
WHAT WAS MEASURED: matvec vs streaming-read roof, above.
WHAT REMAINS UNKNOWN: the same ratio on Neoverse V2 / Zen 4 / M2, where the
memory system is different and ingot's NEON widening path runs instead.

## Next action — reopen only if

- a BF16/F16 checkpoint enters `docs/models.md` as a shipped rung, AND
- on a host with BFDOT or AVX512_BF16, `tests/bench_qmat k6 21 1,<ncores>`
  shows ingot's head below **0.8x** of the read roof (i.e. >= 25% on the
  table), measured with weights local.

Then: write the NEON BFDOT + AVX512_BF16 kernels behind runtime capability
checks (K5's dispatch), with activations rounded to bf16 as an explicit opt-in
under `--fast`, gated by `mynah-slm ppl`. Until then the downstream check is
just the measurement:

```
tests/bench_qmat k6 21 1,<ncores>     # Neoverse V2, Zen 4 / SPR, M2+: does the head row stay >= 0.85x?
```
