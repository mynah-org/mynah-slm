# K7 — weight-stationary batched quantized matvec for batched decode

Status: **IN PROGRESS** (opened 2026-10-08)

Item: the follow-up S1 named but did not do
([`serving-continuous-batching.md`](serving-continuous-batching.md) S1-c);
sibling-port-map row 6 ([`sibling-port-map.md`](sibling-port-map.md)).
Lineage: mynah-tts `src/qmat.c` `qmat_batch_rows` (~:3946),
`matvec_q8_neon_x4` (~:1605), `matvec_q8_i8mm_np` (~:3794); mynah-asr
`src/qmat.c` `QMAT_WS_X4` and `qgemm` (~:884-1121). Contracts it must keep:
K3/K4's int8 lane contract ([`q4k-int8-4row.md`](q4k-int8-4row.md),
[`int8-q8_0-q6_k.md`](int8-q8_0-q6_k.md)) and K5's per-ISA tables
([`isa-runtime-dispatch.md`](isa-runtime-dispatch.md)).

## Task

Give `mynah_slm_forward_multi` a product that reads each weight ONCE per
decode step for all B sequences, without changing a single bit of any
sequence's result: for B tokens against one Q4_K (f32 activations AND int8),
Q6_K and Q8_0 (int8) matrix, decode each weight block once into registers and
apply it to every token's activations. Scalar twin, threads by disjoint row
ranges, runtime-dispatched through the K5 tables, wired as
`MYNAH_SLM_DECODE_PRODUCT=ws`. Default for `forward_multi` only if an
interleaved A/B says so.

## Question

At decode widths B = 2..8 on the 0.6B geometry, is one weight-stationary
pass per weight faster than B fused single-token matvecs (today's default
`matvec` product), with every sequence's logits still memcmp-identical to
its solo decode?

## Known facts

- S1-c measured on this VM class: `matmat` (dequantize a 128-row strip to
  f32, then sgemm) is 3-5x SLOWER than B solo steps at B = 2..8; `matvec`
  (B threaded fused matvecs per weight, weights read B times) is 1.03-1.20x
  faster than solo and memcmp-identical to it. The default is `matvec`.
- A decode matvec is memory-bound on the M1 (docs/perf.md); on this 4-vCPU
  VM the Q4_K f32 kernel is partly compute-bound (unpack + widen + FMA per
  weight), so the WS win has two sources: weight bytes read once instead of
  B times, and the nibble unpack / int8 decode done once instead of B times.
- mynah-tts groups up to 4 (NEON x4) / 8 (i8mm) activations per pass over a
  weight block and falls back to pairs and singles for the tail; mynah-asr's
  `QMAT_WS_X4` is the same idea on x86. Both report 1.2-2.6x per row at
  B >= 4 and a B=4 step 29.2 -> 25.0 ms.
- The single-token kernels' accumulation order per row is FIXED and, for the
  int8 kernels, identical across ISAs by contract (memcmp vs the scalar
  twin). The f32 Q4_K kernel is NOT identical across ISAs (AVX2, NEON and
  scalar each have their own order); bit-identity can only be asked of the
  same ISA's single-token kernel, which is also what batched == solo needs.

## Cost model (written before the code)

| field | |
|---|---|
| current cost | `matvec` product: B threaded matvecs per weight. Weight bytes B times per step; Q4_K unpack (and, with `--fast`, int8 decode) B times |
| suspected cause | every weight is walked once per token |
| proposed transformation | per row, per 32/64-weight unit: decode once into registers, then B (in groups of <= 4) FMA chains / int8 dots against each token's activations, each token keeping its own accumulators in the single-token order |
| maximum plausible saving | weight traffic and unpack work /B. If decode were purely memory-bound: up to ~B x on the weight walk at small B, minus attention/norms (B-proportional, unchanged). On this VM's 4 cores the walk is the bulk; expect 1.3-2x per step at B = 4 if it holds at all |
| new work introduced | per-token accumulators no longer fit registers beyond ~4 tokens (groups of 4, row re-read from L1 per group); Q6_K/Q8_0 int8 need a weight-side correction instead of the activation-side one so it can be shared by the tokens; f32 activations of B tokens compete for L1 with the row |
| risk | (a) the f32 Q4_K kernel is at 8+ ymm of decoded weights per 64 weights — with 4 token accumulators AVX2's 16 registers spill; (b) bit-identity: clang contracts a*b+c by default, so the per-token epilogue must be the same expression as the single kernel's or the two diverge; (c) on this VM 4 threads x B tokens may still be compute-bound, so the win could be small |
| smallest experiment that kills it | `tests/bench_decode` on the 0.6B-geometry fixture, `ws` vs `matvec` interleaved, B = 1, 2, 4, 8, threads 1 and 4: if `ws` is not faster than `matvec` at B >= 2 in most rounds, it does not become the default |

## Design

- **Bit-identity by construction where possible.** The f32 Q4_K single-token
  kernel of each ISA becomes the `nt = 1` instance of a token-group template
  (`always_inline`, `nt` a compile-time constant at each call site); the WS
  kernel is the `nt = 2..4` instance. Same expressions, so the same
  contraction decisions under any compiler. The int8 WS kernels follow the
  K3/K4 lane contract and are held to the scalar twin by memcmp like every
  other int8 kernel.
- **Q6_K / Q8_0 int8 on VNNI/AVX2 swap the bias side.** `vpdpbusd` /
  `maddubs` are u8 x s8. The single-token Q6_K kernel subtracts
  `32 * SUM xq` (activation-side, shared by four ROWS); WS shares across
  TOKENS, so it makes the activation unsigned (`xq ^ 0x80`) and subtracts
  `128 * SUM q` (weight-side, computed once per block and shared by the
  tokens). Exact in int32 either way, so the lanes — and the floats — are
  identical.
- **Threads**: disjoint row ranges, exactly `mynah_slm_project`'s split
  (four chunks per thread); per row, tokens in groups of <= 4.
- **Fallback**: any token whose activations would not take our kernel on the
  solo path (int8 off for Q6_K/Q8_0, a non-finite vector, mixed int8/f32
  preparations, a width over the prep arrays) sends the WHOLE projection back
  to B solo `mynah_slm_project` calls — never a different kernel than solo.

## Unknowns

- Speed on M1 / Neoverse V2 / Zen 4 (only NEON correctness runs here, under
  qemu).
- Whether `ws` beats `matvec` at B = 2 (the gain is smallest and the
  per-token setup largest there).
- Real-checkpoint end-to-end aggregate tok/s under the server's `--slots`.

## Files inspected

`src/qmat.{c,h}`, `src/qmat_kern.c`, `src/kern.h`, `src/isa.c`,
`src/isa_verify.c`, `src/arch_qwen3.{c,h}` (`mynah_slm_project`,
`project_rows`, `forward_multi`, the decode-product switch),
`tests/bench_decode.c`, `tests/bench_qmat.c`, `tests/test_kernels.c`,
`tests/test_isa.c`, `tests/test_synth.c`; mynah-tts `src/qmat.c`
(`qmat_batch_rows`, `matvec_q8_neon_x4`, `matvec_q8_i8mm_np`); mynah-asr
`src/qmat.c` (`QMAT_WS_X4`, `qgemm`).

## Evidence

(none yet)

## Conclusion

(open)

## Next action

Write the kernels and their tests, then run the decode A/B.
