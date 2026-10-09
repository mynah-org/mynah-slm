# K7 — weight-stationary batched quantized matvec for batched decode

Status: **DONE 2026-10-08** (PROMOTED: `ws` is the default batched-decode product; speed on target hardware unmeasured)

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

All on the 4-vCPU x86 cloud VM (AVX-512 VNNI, no bf16/AMX), gcc 13 and
clang 18, plus qemu-aarch64 for NEON correctness. **NOISY**: other agents'
builds and tests ran throughout (load average 3.6-7.3 during the A/B).

### Correctness (proven here)

| check | result |
|---|---|
| `tests/test_kernels`, every ISA level (avx512vnni, avx512->avx2 table, avx2, scalar; qemu: dotprod, neon, scalar): ws == the same level's single-token kernel, memcmp per token, B = 1..18 (past the 16-token split), Q4_K f32 37x512 / 6x3072, Q4_K int8, Q8_0 int8 37x96 (odd block count) / 19x1024, Q6_K int8 37x256 / 6x3072, hostile quantizer blocks in some tokens | PASS (gcc, clang, qemu `-cpu max`, `-cpu cortex-a53`) |
| same, rows split raggedly over 4 threads (chunks of 1/3/7/64 rows) at B = 7 and 16 | PASS |
| ws declines where the single kernel declines (Q8_0/Q6_K with f32 activations, int8 at the scalar/neon levels) and declines a batch mixing int8 and f32 (NaN) tokens | PASS |
| single-token outputs unchanged by turning the f32 Q4_K kernel into the nt = 1 template: hashes of every single kernel at every level before/after | IDENTICAL (gcc, clang, aarch64 qemu) |
| verify-on-first-use checks every ws entry against its own table's single entry (5 tokens = a group of 4 + a tail); `tests/test_isa` teeth: a ws kernel one ulp off for one token is rejected (f32 and int8) | PASS |
| mutations: VNNI Q6_K ws scale selectors swapped; AVX2 Q8_0 ws reading token 0's scale | verify rejects the table, `test_isa` fails |
| mutation: a group of 3 tokens run as 2 | 56 `test_kernels` failures |
| `tests/test_synth` `[ws]`, both fixtures, B = 1..8 sequences at different positions, f32 + bf16 KV, 6 greedy steps: each sequence memcmp-identical to its solo decode, f32 activations and int8 (avx512-vnni; qemu: neon-dotprod) | PASS; 6720 ws kernel calls f32 / 10080 int8 on the Q4_K_M mix, 0 on the F32 fixture (all tiled solo calls) |

The Q6_K f32 path (and Q8_0 f32) is ingot's kernel on the solo path. A ws
kernel bit-identical to it would have to reproduce ingot's internal order,
which is not ours to pin. Instead the ws product walks those tensors in
16-row tiles and runs the solo call per token on each tile: weight bytes from
memory once, decode work still B times. Scratch microbench (151936x1024 Q6_K,
B = 4, ingot kernel, 9 interleaved rounds): tiled won 7/9 at 4 threads
(~99 -> ~81 ms) and 8/9 at 1 thread, memcmp-identical outputs.

### Speed — CLOUD-SPECIFIC, revalidate on target

`tests/bench_decode - 128 7 1,4` (and `MYNAH_SLM_INT8=1`): 0.6B geometry
(28 layers, 1024/3072, 151936-row tied Q6_K head, Q4_K_M type mix), noise
weights written to local tmp, position 128, bf16 KV, OpenBLAS, arms
interleaved in one process with the order rotated per round, each sample the
best of 3 steps, 7 rounds. ms per decode step for all B streams, median
(min..max). "ws calls" per step proves the path ran (B = 1 is the solo path
by design, so its row is a CONTROL: matvec and ws run the same code there).

f32 activations (the default), 1 thread:

| B | solo | matvec | ws | matvec/ws | solo/ws | ws wins |
|---|---|---|---|---|---|---|
| 1 (control) | 175.1 (155.1..210.5) | 182.9 (160.3..226.3) | 169.4 (156.3..193.9) | 1.08x | 1.03x | 4/7 |
| 2 | 332.4 (311.0..397.5) | 325.9 (313.6..394.6) | 278.3 (244.3..346.2) | **1.17x** | 1.19x | 7/7 |
| 4 | 708.9 (635.6..779.2) | 626.0 (609.6..710.5) | 484.2 (457.8..523.0) | **1.29x** | 1.46x | 7/7 |
| 8 | 1351.3 (1305.5..1525.5) | 1347.1 (1258.0..1491.9) | 978.2 (919.8..1186.8) | **1.38x** | 1.38x | 7/7 |

f32 activations, 4 threads:

| B | solo | matvec | ws | matvec/ws | solo/ws | ws wins |
|---|---|---|---|---|---|---|
| 1 (control) | 162.2 (82.6..216.4) | 139.8 (126.5..200.3) | 133.4 (109.7..198.2) | 1.05x | 1.22x | 3/7 |
| 2 | 336.0 (309.0..442.3) | 276.7 (226.0..353.7) | 226.3 (208.4..301.0) | **1.22x** | 1.48x | 6/7 |
| 4 | 367.3 (310.9..495.9) | 294.7 (274.9..320.2) | 206.8 (190.1..322.0) | **1.43x** | 1.78x | 6/7 |
| 8 | 891.1 (662.5..1085.0) | 820.9 (555.8..1165.0) | 563.7 (457.8..790.6) | **1.46x** | 1.58x | 6/7 |

int8 activations (`MYNAH_SLM_INT8=1`, avx512-vnni: every tensor has a ws
kernel, the Q6_K head included):

| threads | B | matvec | ws | matvec/ws | solo/ws | ws wins |
|---|---|---|---|---|---|---|
| 1 | 1 (control) | 99.9 (95.7..137.3) | 111.5 (98.8..134.0) | 0.90x | 1.06x | 2/7 |
| 1 | 2 | 221.2 (180.9..263.4) | 111.7 (98.4..123.7) | **1.98x** | 2.18x | 7/7 |
| 1 | 4 | 395.1 (359.0..452.2) | 164.7 (149.0..175.3) | **2.40x** | 3.07x | 7/7 |
| 1 | 8 | 702.4 (601.7..823.9) | 307.2 (291.4..324.0) | **2.29x** | 2.94x | 7/7 |
| 4 | 1 (control) | 50.4 (41.0..111.9) | 48.1 (42.4..96.7) | 1.05x | 1.02x | 4/7 |
| 4 | 2 | 126.9 (104.3..247.9) | 73.3 (54.3..123.1) | **1.73x** | 1.86x | 7/7 |
| 4 | 4 | 269.2 (182.7..387.5) | 143.9 (80.8..181.4) | **1.87x** | 2.86x | 7/7 |
| 4 | 8 | 697.0 (346.1..952.9) | 275.1 (186.5..343.0) | **2.53x** | 3.11x | 7/7 |

Reading it:

- The B = 1 controls land at 0.90-1.08x with a split win count: the noise
  floor of this machine is ~10%. Every B >= 2 row is outside it and ws won
  the same-round comparison in 6/7 or 7/7 rounds in all four
  configurations.
- f32: 1.17-1.46x over matvec, growing with B, consistent with half the
  step's weight bytes (the Q4_K tensors) being decoded once and the other
  half (ingot's Q6_K/Q8_0) at least being read once.
- int8: 1.7-2.5x. Every tensor has a ws kernel, and at 1 thread B = 2 costs
  the same as B = 1 (111.7 vs 111.5 ms): the second stream is nearly free,
  i.e. the int8 step is bound by the weight walk, not by the dots.

## Conclusion

**PROMOTE `ws` to the default product of `forward_multi`.** It is exact
(memcmp-identical per sequence to solo decode, the same gate S1 set for
`matvec`) and faster than `matvec` at every B >= 2, both thread counts, both
activation paths, outside the B = 1 noise floor. `matvec` and `matmat`
remain behind `MYNAH_SLM_DECODE_PRODUCT` for A/B.

WHAT CHANGED: per-ISA weight-stationary kernels (`kern.h` table entries,
`mynah_slm_matvec_ws`), the f32 Q4_K kernel as a token-group template whose
nt = 1 instance is the single-token kernel, verify-on-first-use for the new
entries, the `ws` product with a tiled fallback for ingot's types, and
`bench_decode`'s ws arm.
WHAT PATH ACTUALLY RAN: x86 qmat avx512vnni (f32 Q4_K kernel: the AVX2
template compiled in the VNNI TU); int8 avx512-vnni; ws kernel calls counted
per step (2240 f32 / 3152 int8 at 4 threads). NEON ran only under qemu, for
correctness.
WHAT WAS MEASURED: decode-step time on synthetic 0.6B-geometry weights, above.
WHAT REMAINS UNKNOWN: M1 / Neoverse / Zen 4 speed; a quiet host; real-model
aggregate tok/s through the server's `--slots`; whether a ws kernel of our
own for Q6_K f32 (instead of the tiled ingot calls) is worth a second
definition of that type's arithmetic.
VERDICT: PROMOTE.

## Next action — downstream (real model, weights LOCAL)

```
make clean && make -j && make test
tests/bench_decode models-local/Qwen3-0.6B-Q4_K_M.gguf 128 9 1,<ncores>
MYNAH_SLM_INT8=1 tests/bench_decode models-local/Qwen3-0.6B-Q4_K_M.gguf 128 9 1,<ncores>
make test-server-slots          # --slots 4 end to end, now on the ws product
```

On M1 (8 threads), a Linux x86 AVX2 host and Neoverse V2. Expect ws faster
than matvec at B >= 2; if the M1 shows the f32 case at < 1.05x, the Q6_K
head (tiled, not stationary) is the first suspect.
