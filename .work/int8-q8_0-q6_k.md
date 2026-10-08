# K4 — int8-activation matvec for Q8_0 and Q6_K, under the existing `--fast`

Status: **DONE 2026-10-08** (both kernels KEPT opt-in under `--fast`; perplexity and decode tok/s UNMEASURED, no model here)

Item: `PLAN.md` §0 K4. Lineage: K3's contract
([`q4k-int8-4row.md`](q4k-int8-4row.md)); mynah-asr `src/qmat.c` VNNI rows
with the `+128` / `dpbusd` correction; llama.cpp's `abs/sign` trick for
`maddubs`. History that bears on it: docs/perf.md "The Q6_K kernel that ended
up in ingot instead" (an int8 SDOT Q6_K kernel was written, tied on M1 at
1.02x, and deleted).

## Task

Give `--fast` / `MYNAH_SLM_INT8=1` int8 kernels for the two other types a
Qwen3 checkpoint carries, reusing the activations K3 already quantizes once
per matvec (`mynah_slm_matvec_in`: `xq` int8 per 32, `xscale`, `xsum`).
Decide per type, with numbers, whether it stays.

## Question

Does an int8 dot beat ingot's f32-activation kernel for Q8_0 and for Q6_K on
the shapes that matter, by enough to be worth a second quality trade inside
`--fast`?

## Known facts

- **Q8_0** (34 B / 32 values: f16 `d`, 32 int8). Same block size as our
  activation quantization, so a block is exactly one 32-byte int8 dot and one
  scale `d * xscale[b]`. ingot's kernel (`ingot_q8_0_matvec_avx2/neon`) is f32:
  per 8 weights a widen to int32, a convert and an FMA — four ops per eight
  weights where an int8 dot is one op per 32 (VNNI) or 64.
- **Q6_K** (210 B / 256 values: `ql[128]`, `qh[64]`, int8 `sc[16]`, f16 `d`
  at the END). Value `q = (lo4 | hi2<<4) - 32` in [-32, 31], one scale per
  16 values. In Q4_K_M files it is `attn_v`, `ffn_down` and the tied
  `lm_head` 151936x1024 — the single largest matvec of a decode step (42%
  on M1, docs/perf.md). Our f32 Q6_K kernel was deleted because it tied
  ingot's on ARM and lost on x86. **The int8 SDOT variant also tied on M1
  (1.02x)**, so the justification for trying again is specifically the x86
  arm: on x86, ingot's Q6_K is f32 AVX2 (no VNNI, no 512-bit), and VNNI turns
  the 6-bit reassembly + 64-byte dot into one instruction.
- Both formats are SIGNED, VNNI's `vpdpbusd` is u8 x s8:
  - Q8_0: activation made unsigned (`xq ^ 0x80` = xq+128, once per activation
    load, shared by four rows) and the bias removed with a second `vpdpbusd`
    of the weights against a constant 128 — mynah-asr's on-the-fly
    correction, exact in int32. AVX2: llama.cpp's `maddubs(|w|, sign(xq, w))`;
    with `|xq| <= 127` the products stay in int16 (128*127*2 = 32512).
  - Q6_K: the raw 6-bit `u = q + 32` IS unsigned, so `vpdpbusd(u, xq)` needs
    no re-encoding; the bias `32 * SUM xq` depends only on the activation and
    is one more `vpdpbusd` per activation load, shared by four rows. NEON
    simply subtracts 32 before `sdot`.
- K3's contract carries over unchanged: int32 lanes of four consecutive bytes,
  f32 accumulators split even/odd 32-value chunk, one reduction per row, the
  same `tree8`, a scalar twin in the same order, memcmp across ISAs, row
  grouping and threads.

## Cost model

| field | Q8_0 | Q6_K |
|---|---|---|
| current cost | ingot f32 AVX2/NEON | ingot f32 AVX2/NEON |
| suspected cause | widen+convert+FMA per 8 weights | same plus the 6-bit reassembly per 8 |
| proposed transformation | int8 dot per 32/64, one scale FMA per 8 lanes | decode 64 bytes of `u` per vector op, int8 dot, per-16 scale as a lane vector |
| maximum plausible saving | ~2-3x per tensor if issue-bound (Q8_0 was ALU-bound on M1, docs/perf.md) | head: bandwidth sets the ceiling (it was ~1.25x for Q4_K's head in K3) |
| new work introduced | `+128` correction (one extra dot per row on VNNI) | 32*SUM xq bias per activation load |
| risk | a second int8 quality trade inside `--fast` | the head is in front of the softmax: int8 error on it is the most visible kind |
| smallest experiment that kills it | interleaved A/B vs ingot on 1024x1024 at 1 thread: < 1.10x → do not ship the type | same, on the 151936x1024 head |

## Unknowns

- Perplexity cost of int8 on the head (Q6_K) and on Q8_0 files.
- M1 / Neoverse V2 speed (NEON runs here only under qemu).

## Files inspected

`src/qmat.c` (after K3), `src/arch_qwen3.c` (`mynah_slm_project`),
`third_party/ingot/src/kernels.c` (`ingot_q8_0_matvec_*`, `q6_k_dot_block_*`,
`q6_dequant_block`), docs/perf.md (Q6_K, Q8_0, `--fast`), mynah-asr
`src/qmat.c` (VNNI rows), mynah-tts `src/qmat.c` (`dot4_u8_evex`,
`dot_u8_i32` correction).

## Evidence

### Proven here (no model; cloud VM)

| Check (`tests/test_kernels`, `int8_contract`) | x86 AVX-512 VNNI (gcc + clang) | x86 AVX2+FMA+F16C | x86 scalar (`x86-64-v2`) | aarch64 SDOT (qemu `-cpu max`) | aarch64 `armv8-a` (qemu cortex-a53) |
|---|---|---|---|---|---|
| Q8_0 kernel == scalar twin, memcmp: 103x1024, 37x96 (3 blocks: odd count, masked tail), 6x3072 | PASS | PASS | twin only | PASS | twin only |
| Q6_K kernel == scalar twin, memcmp: 103x1024, 37x256, 6x3072 | PASS | PASS | twin only | PASS | twin only |
| one-row calls == groups of four; 4 threads x ragged chunks (1/3/7/64) == 1 thread, memcmp | PASS both types | PASS | n/a | PASS | n/a |
| twin vs ingot dequant x the int8 activations, double | Q8_0 9.3e-8..2.2e-7, Q6_K 1.0e-7..1.8e-7 rel | same | same | same | same |
| twin vs ingot dequant x f32 activations (the int8 approximation) | Q8_0 5.0e-3..1.9e-2, Q6_K 6.0e-3..9.3e-3 rel (gate 3e-2) | same | same | same | same |
| `xq` never -128 (AVX2's `sign(xq, w)` relies on it) | PASS on the spike fixture only — **the claim was FALSE** for a block with amax < ~3.7e-37 (subnormal scale, inv = +inf, saturation to -128, 0*inf cast to int8): fixed and pinned, see "Correction (review B1)" below | | | | |
| `MYNAH_SLM_INT8_TYPES` narrows per type, never widens | PASS | PASS | n/a | PASS | n/a |
| with f32 activations we still DECLINE Q6_K and Q8_0 (ingot's kernels) | PASS | PASS | PASS | PASS | PASS |

The odd-count Q8_0 shape cannot use ingot's dequantizer (it walks 256-value
strides — its own "stride trap" note), so that one shape is referenced to the
format's definition `d * q`; the other two Q8_0 shapes reference ingot.

Mutation checks: making the Q6_K twin read the wrong group's scale
(`s[2c + 1 - j/4]`) fails 15 checks including the ingot reference; replacing
the Q8_0 twin's `fmaf` with an unfused multiply-add (one rounding more) fails
the 9 memcmp checks and none of the tolerance checks — which is exactly why
the contract is memcmp and not a tolerance. (Swapping the E and O
accumulators is an EQUIVALENT mutation, E+O is per-lane commutative, and
correctly fails nothing.)

### Measured here — CLOUD-SPECIFIC, revalidate on target

Same VM as K3 (4 vCPU Cascade Lake class, AVX-512 VNNI, no AVX512_BF16),
gcc 13.3 -O3, **load average 4.4-6.3 from other agents** during these runs.
`tests/bench_qmat k4 15 1,4`: A = `ingot_matvec` (f32 activations: what runs
for these types today, with or without `--fast`), B = ours int8 (`--fast` after
this item); interleaved, order alternating, 15 rounds, ~30 ms blocks, the row
split of `mynah_slm_project`. The outputs differ by the int8 budget (rel diff
column). Synthetic weights in RAM.

**AVX-512 VNNI build, 1 thread** (two independent sessions, ratio a / ratio b):

| type | shape | ingot ms median (min..max) | int8 ms median (min..max) | ingot/int8 (a / b) | int8 wins | direction |
|---|---|---|---|---|---|---|
| Q8_0 | 1024x1024 | 0.241 (0.231..0.408) | 0.089 (0.078..0.217) | **2.71x / 2.70x** | 15/15, 15/15 | stable |
| Q8_0 | CONTROL ingot vs ingot | 0.238 | 0.238 | 1.00x / 1.02x | 4/15, 8/15 | (control) |
| Q8_0 | 3072x1024 | 0.857 (0.704..0.967) | 0.292 (0.266..0.447) | **2.93x / 2.77x** | 15/15 | stable |
| Q8_0 | 1024x3072 | 0.844 (0.736..1.100) | 0.271 (0.231..0.445) | **3.11x / 2.86x** | 15/15 | stable |
| Q8_0 | 151936x1024 (head) | 42.95 (42.22..48.52) | 38.97 (37.81..44.54) | **1.10x / 1.11x** | 14/15 | stable |
| Q6_K | 1024x1024 (attn_v) | 0.297 (0.292..0.331) | 0.054 (0.053..0.082) | **5.46x / 5.49x** | 15/15 | stable |
| Q6_K | CONTROL ingot vs ingot | 0.310 | 0.303 | 1.02x / 1.03x | 8/15 | (control) |
| Q6_K | 3072x1024 | 0.900 (0.883..1.246) | 0.198 (0.184..0.319) | **4.54x / 4.34x** | 15/15 | stable |
| Q6_K | 1024x3072 (ffn_down) | 0.937 (0.872..1.117) | 0.176 (0.166..0.247) | **5.31x / 4.86x** | 15/15 | stable |
| Q6_K | 151936x1024 (lm_head) | 47.63 (45.76..62.16) | 29.27 (27.30..38.63) | **1.63x / 1.66x** | 15/15 | stable |

**AVX-512 VNNI build, 4 threads** (one session): Q8_0 2.34x / 2.49x / 2.62x on
the three layer shapes, head 1.18x (14/15); Q6_K 4.75x / 4.31x / 4.03x, head
**1.99x** (15/15). Controls 0.97x and 0.90x — at 4 threads on this loaded VM
only the direction is a claim.

**AVX2+FMA build (no VNNI, no 512-bit), same VM:**

| type | shape | 1 thread ingot/int8 (wins) | 4 threads ingot/int8 (wins) |
|---|---|---|---|
| Q8_0 | 1024x1024 | 2.41x (15/15) | 2.14x (15/15) |
| Q8_0 | 3072x1024 | 2.22x (15/15) | 2.09x (14/15) |
| Q8_0 | 1024x3072 | 2.52x (15/15) | 2.53x (15/15) |
| Q8_0 | 151936x1024 | 1.22x (13/15, NOT stable) | 1.01x (7/15, NOT stable) |
| Q6_K | 1024x1024 | 2.36x (15/15) | 2.08x (15/15) |
| Q6_K | 3072x1024 | 2.51x (15/15) | 1.99x (15/15) |
| Q6_K | 1024x3072 | 2.61x (15/15) | 1.78x (14/15) |
| Q6_K | 151936x1024 | 1.26x (15/15) | 1.21x (11/15, NOT stable) |
| controls | ingot vs ingot | 0.98x, 1.07x | 0.97x, **1.21x** |

The 4-thread AVX2 Q6_K control at 1.21x is a contaminated round set (load
6.3); nothing in that column is quoted as a magnitude.

What these support:

- **The int8 arm is the x86 argument, and it holds on both x86 widths.** The
  ARM history (int8 Q6_K tied ingot at 1.02x on M1) was an f32-vs-int8
  comparison on NEON, where ingot's f32 kernel is already tight; on x86,
  ingot's Q6_K and Q8_0 kernels are f32 AVX2 (a widen + convert + FMA per 8
  weights), and an int8 dot per 32-64 weights beats them 2-5x per layer
  tensor, direction 14-15/15 everywhere.
- **The Q6_K head is the number that matters for Q4_K_M files**: 151936x1024,
  42% of a decode step on M1 (docs/perf.md). 1.63-1.66x at 1 thread and 1.99x
  at 4 threads on AVX-512; 1.21-1.26x on AVX2. Q8_0's head gains less (1.1x):
  34 bytes per 32 weights is more bandwidth than Q6_K's 26.25, and the head is
  where bandwidth binds.
- Q6_K int8 is FASTER per element than Q4_K int8 on the same shape here
  (0.055 vs 0.082 ms on 1024x1024): no 6-bit scale unpack, no min term.

### Correction (review B1, 2026-10-08)

The "`xq` never -128" row above was proven on ONE fixture (a spike) and the
general claim was false. The quantizer computed `scale = amax/127` and
`inv = 1/scale`; for a block with `amax` below ~3.7e-37 the scale is
subnormal and `inv` overflows to +inf, so every nonzero value saturated and
the `[-128, 127]` clamp emitted **-128**, and a zero became `0 * inf = NaN`
cast to int8 (UB). With `xq = -128`, AVX2's `sign(xq, w)` wraps and the AVX2
Q8_0 kernel disagreed with its twin (reproduced by the review's corner
fuzzer: twin 1.0e-37 vs avx2 -1.0e-37 on one block).

Fix (`src/qmat.c`): `inv = 127/amax` directly; a block with `amax < 2^-120`
(the smallest amax for which `127/amax` is finite — `FLT_MIN` is not enough,
`127/FLT_MIN` overflows), NaN or a non-finite sum quantizes to zero with a
zero scale; the clamp is symmetric `[-127, 127]`. Pinned in
`tests/test_kernels.c` (tiny-amax, all-+-max and denormal blocks, and the
same two hostile blocks inside every `int8_contract` fixture, so the memcmp
against the twin covers them at every ISA level) and in
`mynah_slm_isa_verify_qmat`'s fixtures. Activations that small do not occur
in a normalized transformer; the fix is about the contract being true, not
about a measured quality change.

## Conclusion

**KEEP both, opt-in, inside `--fast`** — per type:

- **Q8_0: KEEP.** 2.2-3.1x on layer shapes at both x86 widths; head 1.1x.
  Quality cost unmeasured (Q8_0 weights are the most precise we ship, so the
  int8 activation becomes the dominant error term there — measure before
  recommending `--fast` on a Q8_0 file).
- **Q6_K: KEEP, with the head as the open quality question.** 4-5x on
  `attn_v`/`ffn_down` (AVX-512), 1.6-2.0x on the head. The head sits directly
  in front of the softmax: if `--fast` perplexity moves by more than the
  +1.4% Q4_K-only budget, narrow the default with
  `MYNAH_SLM_INT8_TYPES=q4_k,q8_0` rather than reverting the kernel, and record
  the number. On ARM the prior evidence predicts a tie; if M1 measures < 1.05x
  for Q6_K, int8 there buys nothing and costs quality — narrow it on ARM.

Nothing changes for anyone who does not pass `--fast` / `MYNAH_SLM_INT8=1`:
with f32 activations, `mynah_slm_matvec` still declines Q8_0 and Q6_K and
ingot's kernels run, asserted in `tests/test_kernels.c`.

WHAT CHANGED: int8 Q8_0 and Q6_K kernels (AVX-512 VNNI, AVX2, NEON SDOT) +
scalar twins + `mynah_slm_q80_int8_ref` / `mynah_slm_q6k_int8_ref`;
`mynah_slm_matvec_have()` now claims Q8_0/Q6_K only when int8 is on;
`MYNAH_SLM_INT8_TYPES` narrowing; `mynah_slm_project` prepares activations for
any column count that is a multiple of 32 (was 256: Q8_0 rows need not be
256-aligned); `tests/bench_qmat k4`.
WHAT PATH ACTUALLY RAN: avx512-vnni and avx2 (both printed by the bench), NEON
under qemu for correctness only.
WHAT WAS MEASURED: per-tensor kernel time, synthetic weights, above.
WHAT REMAINS UNKNOWN: perplexity with int8 on the head / on Q8_0 files; decode
tok/s; ARM speed.

## Next action — downstream (real model, weights LOCAL)

On x86 Linux (AVX2 and AVX-512 hosts), M1, Neoverse V2:

```
make clean && make -j && make test test-parity
tests/bench_qmat k4 21 1,<ncores>                 # synthetic A/B; expect >= 1.5x on Q6_K layer shapes on x86
# decode tok/s, interleaved, same binary/prompt/seed/threads, >= 5 runs each, Q4_K_M:
for t in q4_k q4_k,q6_k; do
  MYNAH_SLM_INT8_TYPES=$t ./mynah-slm run -m models-local/Qwen3-0.6B-Q4_K_M.gguf \
    -p "Racconta una storia breve su un faro." -n 128 --temp 0 -t <n> --fast
done
# quality, per type, same text as docs/perf.md (917 tokens):
MYNAH_SLM_INT8=1 MYNAH_SLM_INT8_TYPES=q4_k      ./mynah-slm ppl -m models-local/Qwen3-0.6B-Q4_K_M.gguf -f <text>  # the old --fast: 2.842
MYNAH_SLM_INT8=1 MYNAH_SLM_INT8_TYPES=q4_k,q6_k ./mynah-slm ppl -m models-local/Qwen3-0.6B-Q4_K_M.gguf -f <text>  # + head and ffn_down
MYNAH_SLM_INT8=1                                ./mynah-slm ppl -m models-local/Qwen3-0.6B-Q8_0.gguf   -f <text>  # vs the same file without
./mynah-slm ppl -m models-local/Qwen3-0.6B-Q8_0.gguf -f <text>
```

Prediction, written before it runs: on x86, `--fast` decode on Q4_K_M rises a
further 15-30% from Q6_K (the head is ~40% of the step); ppl moves by +0.3 to
+1.0% on top of the Q4_K-only 2.842. On M1, Q6_K within ±5% of ingot.
