# K3 — the Q4_K int8 kernel: four rows per activation load, one reduction per row

Status: **DONE 2026-10-08** (kernel landed behind the existing opt-in `--fast` / `MYNAH_SLM_INT8=1`; decode tok/s UNMEASURED, no model here)

Item: `PLAN.md` §0 K3. Lineage: mynah-tts `src/qmat.c` `dot4_u8_evex` /
`dot4_u8_vex` (four weight rows per activation load) and its "one horizontal
reduction per row" rule, which ingot's Q8_0 kernel already follows
(docs/perf.md, Q8_0 section). Map: [`sibling-port-map.md`](sibling-port-map.md).
Code: `src/qmat.c` (int8 Q4_K path only), `tests/test_kernels.c`,
`tests/bench_qmat.c`.

## Task

Restructure the opt-in int8 Q4_K matvec (`--fast` / `MYNAH_SLM_INT8=1`) so an
activation block is loaded once and used against four weight rows, and so the
float reduction happens once per row instead of once per 32 weights. AVX2,
AVX-512 VNNI and NEON SDOT, with a scalar twin in the same accumulation order.
The exact (f32-activation) Q4_K path is not touched.

## Question

Is the per-sub-block horizontal reduction a material share of the int8 kernel
on the shapes Qwen3-0.6B issues, and does removing it make the matvec faster
here without changing what the int8 path computes beyond a float reorder?

## Known facts

- Before this item, per 64 weights (one Q4_K sub-block pair) the int8 kernel
  does: unpack (3-4 ops), one `vpdpbusd` (AVX-512) or two `maddubs+madd`
  (AVX2) or four `sdot` (NEON), then **two full horizontal int32 reductions**
  (`hsum256i` = 5 ops each on x86; `vaddvq_s32` on NEON), then four scalar
  float multiplies and two scalar adds per sub-block, plus two scalar 6-bit
  scale/min unpacks per sub-block (`q4_k_scale_min`, branchy).
- So on x86 the reduction and scalar epilogue are ~2-3x the instruction count
  of the dot product itself. The kernel is int8 because the dot became cheap;
  the epilogue did not.
- The activation (`xq`, `xscale`, `xsum`) is prepared once per matvec in
  `mynah_slm_matvec_prepare` and is the same for every row. It is 1-3 KiB:
  L1-resident. Re-loading it per row costs load-port slots, not bandwidth.
- Shapes (Qwen3-0.6B, Q4_K_M): `attn_q` 2048x1024, `attn_k/v` 1024x1024,
  `attn_out` 1024x2048, `ffn_gate/up` 3072x1024, `ffn_down` 1024x3072 (Q6_K in
  Q4_K_M files, Q4_K in others), `lm_head` 151936x1024 (Q6_K in Q4_K_M; Q4_K
  only in a pure-Q4_K requant). Row counts are all multiples of 4, but chunk
  boundaries from the pool need not be.
- The int8 path is a quality trade already gated by perplexity (+1.4% ppl on
  M1, docs/perf.md); a float reorder inside it is not a new quality decision,
  but it must still be stated and bounded.

## Cost model (written before the code)

| field | |
|---|---|
| current cost | unknown here in ms; instruction count per 64 weights ~20-25 (x86 AVX-512), of which ~4 are the dot |
| suspected cause | per-sub-block horizontal reduction + scalar epilogue + scalar scale unpack |
| proposed transformation | int32 lane partials -> f32 lanes, FMA with a per-lane scale into a per-row f32 accumulator; one reduction per row; scale/min unpack for all 8 sub-blocks at once (ggml's 3-mask form) and vectorized; activation vectors loaded once per 4 rows |
| maximum plausible saving | if the epilogue is ~2/3 of the issued ops and the kernel is issue-bound (docs/perf.md says Q4_K decode is not bandwidth-bound on M1), up to ~2x per tensor; ~1.3x is the realistic hope because loads and the unpack remain |
| new work introduced | one cvt + one FMA (+ one permute on AVX-512) per 32 weights; 8-wide scale vector per block |
| risk | register pressure at 4 rows on AVX2 (16 ymm); a reorder of the float sum (stated, bounded) |
| smallest experiment that kills it | interleaved A/B of old vs new int8 kernel on 1024x1024 at 1 thread: if not >= 1.10x, stop |

## Design: the canonical accumulation order

Every ISA and the scalar twin compute, per row, exactly this (so a row's
result does not depend on the ISA's grouping, on whether the row sits in a
group of four or in the tail, or on the thread count):

```
for each 256-weight block b:
    S[s] = (d * sc[s]) * xscale[8b+s]                      s = 0..7, two f32 multiplies
    M[s] = fma(dmin, mn[s] * xsum[8b+s], M[s])
    for each sub-block pair p = 0..3 (s0 = 2p, s1 = 2p+1):
        for lane j = 0..7:
            P0[j] = SUM_{i=4j..4j+3} lo_nibble[i] * xq[64p + i]          (exact int32)
            P1[j] = SUM_{i=4j..4j+3} hi_nibble[i] * xq[64p + 32 + i]     (exact int32)
            E[j]  = fma((float)P0[j], S[s0], E[j])
            O[j]  = fma((float)P1[j], S[s1], O[j])
result = tree8(E + O) - tree8(M),   tree8(v) = ((v0+v4)+(v2+v6)) + ((v1+v5)+(v3+v7))
```

Why this lane definition: it is what every target produces natively, so no
ISA pays a shuffle for the contract. `maddubs+madd` on 32 bytes gives eight
int32 lanes of four consecutive bytes; `vpdpbusd` on the concatenated
[lo|hi] 64-byte vector gives the same eight lanes for each sub-block in its
two 256-bit halves (E and O are the two halves of one zmm accumulator); two
`sdot` on the two 16-byte halves give lanes 0-3 and 4-7. `tree8` is the
reduction `_mm256` code already does (`lo128+hi128`, `movehl`, `shuffle`), and
NEON performs it explicitly rather than with `vaddvq_f32`, whose internal order
is not the same tree.

## Unknowns

- Speed on M1 / Neoverse V2 (NEON path runs here only under qemu: correctness).
- Whether the AVX2 four-row form spills on a real Haswell-class core.
- End-to-end decode tok/s with `--fast` (no model reachable here).

## Files inspected

`src/qmat.c`, `src/qmat.h`, `src/arch_qwen3.c` (`mynah_slm_project`,
`matvec_chunk`), `tests/test_kernels.c`, `tests/bench_matvec.c`,
`docs/perf.md` (Q4_K, `--fast`, Q6_K, Q8_0), mynah-tts `src/qmat.c`
(`dot4_u8_evex`, `dot4_u8_vex`, `dot_q8_i32_avx512bw`, verify-on-first-use),
mynah-asr `src/qmat.c` (VNNI row kernels), ingot `src/kernels.c`
(`q4_k_dot_block_x86`, Q8_0 row-level fold).

## Evidence

### Proven here (no model; cloud VM)

| Check | x86 AVX-512 VNNI (`-march=native`) | x86 AVX2+FMA (`-mavx2 -mfma -mf16c`) | x86 scalar (`-march=x86-64-v2`) | aarch64 SDOT (`armv8.2-a+dotprod`, qemu `-cpu max`) | aarch64 baseline (`armv8-a`, qemu `-cpu cortex-a53`) |
|---|---|---|---|---|---|
| vector kernel == scalar twin, **memcmp**, 103x1024 / 37x256 / 6x3072 | PASS gcc + clang | PASS | n/a (no int8 kernel compiled; twin checked alone) | PASS | n/a (no dotprod: int8 off, as before) |
| one row per call (tail kernel) == groups of four, memcmp | PASS | PASS | n/a | PASS | n/a |
| 4 threads, ragged chunks (1, 3, 7, 64 rows) == 1 thread, memcmp | PASS | PASS | n/a | PASS | n/a |
| twin vs ingot dequant x the int8 activations it was given, double | 1.2e-7 .. 2.2e-7 rel | same | same | same | same |
| twin vs ingot dequant x f32 activations (the int8 approximation itself) | 4.3e-3 .. 2.1e-2 rel (< 3e-2 gate; outlier-heavy synthetic activations) | same | same | same | same |
| full `tests/test_kernels` | PASS | PASS | PASS | PASS | PASS |

The reference for the "dequant x int8" row is built from ingot alone: the
min term is NOT int8 in this kernel (it uses the exact f32 `xsum`), so the
reference dequantizes a copy with every nibble zeroed to obtain `-dmin*min`
per sub-block independently of our unpack code. The first version of the test
omitted that and failed at 3e-4 — a wrong reference, not a wrong kernel.

Mutation check: changing the scalar twin's reduction tree from
`(a0+a2)+(a1+a3)` to `(a0+a1)+(a2+a3)` makes the memcmp gates fail (6
failures), so they are not passing by construction.

`make clean && make -j4 all && make test` PASS (model tests SKIP);
`make warnings` gcc and clang PASS; `python3 tools/check_plan.py` OK.

### Measured here — CLOUD-SPECIFIC, revalidate on target

Machine: cloud VM, 4 vCPU Intel Cascade Lake class (AVX-512F/BW/VL/DQ +
AVX512_VNNI, no AVX512_BF16, no AMX), gcc 13.3 `-O3`. **Other agents were
running on the same VM**: load average 2.4-3.6 during these runs. Synthetic
Q4_K matrices (`tests/qfixture.h`), weights in RAM (no file I/O at all).
`tests/bench_qmat k3 15 1,4`: A = the old kernel copied verbatim from
`0860c56`, B = the new one, interleaved in one process, order alternating per
round, ~30 ms per timed block, 15 rounds; row split = `mynah_slm_project`'s
(four chunks per thread). "direction stable" = B won >= 90% or <= 10% of
rounds; "magnitude stable" = both CVs < 5%.

**AVX-512 VNNI build (`-march=native`, the default local build here):**

| shape (Q4_K, int8 act.) | thr | old ms median (min..max, CV) | new ms median (min..max, CV) | old/new | new wins | direction | magnitude |
|---|---|---|---|---|---|---|---|
| 1024x1024 (attn_k/v) | 1 | 0.194 (0.185..0.305, 16%) | 0.082 (0.079..0.097, 6%) | **2.37x** | 15/15 | stable | no |
| CONTROL new vs new, 1024x1024 | 1 | 0.081 (0.078..0.111, 11%) | 0.081 (0.078..0.127, 14%) | 1.00x | 7/15 | (control) | no |
| 3072x1024 (ffn_gate/up) | 1 | 0.574 (0.559..0.605, 2%) | 0.263 (0.254..0.281, 3%) | **2.18x** | 15/15 | stable | yes |
| 1024x3072 (ffn_down shape) | 1 | 0.581 (0.557..0.709, 6%) | 0.253 (0.245..0.269, 2%) | **2.30x** | 15/15 | stable | no |
| 151936x1024 (lm_head shape) | 1 | 33.18 (30.35..47.44, 13%) | 26.64 (24.69..33.83, 10%) | **1.25x** | 15/15 | stable | no |
| 1024x1024 | 4 | 0.167 (0.114..0.255, 22%) | 0.078 (0.041..0.130, 26%) | 2.14x | 15/15 | stable | no |
| CONTROL new vs new | 4 | 0.081 | 0.078 | 1.05x | 9/15 | (control) | no |
| 3072x1024 | 4 | 0.469 (0.353..0.698, 23%) | 0.188 (0.121..0.356, 31%) | 2.50x | 15/15 | stable | no |
| 1024x3072 | 4 | 0.544 (0.352..0.728, 20%) | 0.228 (0.145..0.395, 30%) | 2.39x | 15/15 | stable | no |
| 151936x1024 | 4 | 24.64 (14.59..29.45, 17%) | 20.76 (16.81..25.99, 12%) | 1.19x | 11/15 | **NOT stable** | no |

**AVX2+FMA build (`-mavx2 -mfma -mf16c`, same VM — what a Haswell..Skylake
client or a Zen 2/3 would run):**

| shape | thr | old ms | new ms | old/new | new wins | direction |
|---|---|---|---|---|---|---|
| 1024x1024 | 1 | 0.169 (0.160..0.288) | 0.083 (0.078..0.127) | **2.04x** | 15/15 | stable |
| CONTROL | 1 | 0.084 | 0.082 | 1.02x | 9/15 | (control) |
| 3072x1024 | 1 | 0.497 (0.483..3.247) | 0.267 (0.248..0.281) | **1.86x** | 15/15 | stable |
| 1024x3072 | 1 | 0.537 (0.498..0.653) | 0.265 (0.256..0.332) | **2.03x** | 15/15 | stable |
| 151936x1024 | 1 | 29.26 (27.90..48.97) | 29.05 (27.88..34.17) | 1.01x | 12/15 | NOT stable |
| 1024x1024 | 4 | 0.064 | 0.033 | 1.94x | 15/15 | stable |
| 3072x1024 | 4 | 0.170 | 0.086 | 1.98x | 15/15 | stable |
| 1024x3072 | 4 | 0.193 | 0.092 | 2.09x | 15/15 | stable |
| 151936x1024 | 4 | 9.10 | 9.02 | 1.01x | 9/15 | NOT stable |

What these support, and nothing more:

- **The epilogue was the kernel.** On every L2-sized layer shape the new form
  is ~2x the old at 1 and 4 threads, both x86 widths, direction 15/15. The
  control rows land on 0.95-1.07x, so the harness is not inventing it.
- **The head does not move the same way, and should not.** 87.5 MB of Q4_K per
  call streams from DRAM; once the compute is halved, the head is bound by
  bandwidth (~3.3 GB/s per core on this VM at 1 thread). It still won 1.25x on
  AVX-512 at 1 thread and tied on AVX2. A Q4_K head only exists in a pure-Q4_K
  requant; in Q4_K_M files it is Q6_K and this kernel never sees it.
- **Absolute 4-thread numbers are not reproducible here**: the same native
  binary measured 0.078 ms and, ten minutes later at lower load, 0.042 ms on
  1024x1024. The RATIO held (2.14x vs 2.09x). Only ratios are quoted.
- Magnitudes are "no" almost everywhere: CV > 5% on a shared VM. Direction is
  the claim; the size is an order of magnitude, not a number to plan on.

## Conclusion

**PROMOTE (within the opt-in).** The restructured int8 Q4_K kernel replaces the
old one under `--fast` / `MYNAH_SLM_INT8=1`; the default (f32-activation) path
is byte-for-byte untouched. What changed numerically: the int8 path's float
sum is reordered (old vs new differ by 1.4e-7 .. 3.9e-7 relative on these
fixtures); the quantization itself is identical (same `xq`, `xscale`, `xsum`).
That is three orders of magnitude below the +1.4% perplexity the int8 path
already spends, but it is a numerical change and the ppl gate below re-runs it.

Defect found on the way: `src/qmat.c` at `0860c56` **did not compile for
aarch64 without dotprod** (`-march=armv8-a`, i.e. any portable arm64 build):
`arm_neon.h` was only included inside the SDOT block while the f32 NEON
kernel below it uses NEON types. Fixed here by hoisting the include; the
`armv8-a` build now passes `test_kernels` under qemu `-cpu cortex-a53`.

WHAT CHANGED: int8 Q4_K kernel (AVX-512 VNNI, AVX2, NEON SDOT) + scalar twin;
`mynah_slm_matvec_prepare_int8`, `mynah_slm_q4k_int8_ref`,
`mynah_slm_matvec_int8_isa` for tests/benches; `tests/qfixture.h`,
`tests/bench_qmat.c`, `make bench-qmat`; the armv8-a include fix.
WHAT PATH ACTUALLY RAN: avx512-vnni (native), avx2 (variant build), printed by
`mynah_slm_matvec_int8_isa()` in both the test and the bench header; NEON
under qemu for correctness only.
WHAT WAS MEASURED: per-tensor kernel time on synthetic matrices, above.
WHAT REMAINS UNKNOWN: decode tok/s with `--fast`, ppl with `--fast` after the
reorder, NEON speed on M1 / Neoverse V2, AVX2 spills on a real Haswell-class
core.

## Next action — downstream (real hardware, real model, weights LOCAL)

On each of M1, x86 Linux (AVX2 and AVX-512 VNNI hosts if both exist),
Neoverse V2:

```
make clean && make -j && make test test-parity
tests/bench_qmat k3 21 1,<ncores>            # kernel A/B, synthetic: expect old/new >= 1.3x on layer shapes
make bench MODEL=models-local/Qwen3-0.6B-Q4_K_M.gguf   # control rows (Q6_K) must stay 1.00x
# end-to-end, interleaved, same binary, same prompt/seed/threads, >= 5 runs each:
./mynah-slm run -m models-local/Qwen3-0.6B-Q4_K_M.gguf -p "Racconta una storia breve su un faro." -n 128 --temp 0 -t <n>
./mynah-slm run -m models-local/Qwen3-0.6B-Q4_K_M.gguf -p "Racconta una storia breve su un faro." -n 128 --temp 0 -t <n> --fast
# quality, before (0860c56) and after, int8 only (ppl has no --fast flag; the env switch is the same switch):
MYNAH_SLM_INT8=1 ./mynah-slm ppl -m models-local/Qwen3-0.6B-Q4_K_M.gguf -f <the 917-token text behind docs/perf.md>   # expect 2.842 +- 0.002
```

Prediction, written before any of it runs: `--fast` decode +5-15% over the
previous `--fast` on x86 (the Q4_K layer tensors are ~55% of a Q4_K_M step,
the Q6_K tensors and the head are untouched); smaller on M1, where the old
NEON epilogue (`vaddvq_s32`) was cheaper than x86's `hsum256i`. If the ppl
moves by more than 0.002, revert this commit: the reorder is not supposed to
be visible.
