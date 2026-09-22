# R1-N — the ternary microbench on Neoverse V2 (GCP Axion)

Status: **OPEN** (prepared 2026-09-22, not yet run on target hardware)

Item: `PLAN.md` §0 R1-N. Follows [`r1-ternary-mac-kernel.md`](r1-ternary-mac-kernel.md)
and [`tied-trit-cpu-kernel-reading.md`](tied-trit-cpu-kernel-reading.md).
Code: `bench/ternary_gemv/` (`make axion`).

**This note exists so the prediction is on the record BEFORE the measurement.**
A prediction written after the run is not a prediction.

## What is ported, and what is deliberately not

`[MEASURED, Mac]` The Mac produced one clear candidate and one weaker one. Only
those cross:

| | ported | why |
|---|---|---|
| **T3 fold9 K=2**, 4.125 bpw | **yes** | 2.0–2.3× the production Q4_K path at fewer bits |
| **T1 2-bit**, 2.125 bpw | **yes** | same speed, half the bytes; the storage end of the trade |
| T3 K=3, 6.125 bpw | yes | the only configuration with a credible quality story |
| T0 int8 | oracle only | not a storage candidate |
| T2 base-3 | **no** | 1.625 bpw payload but no SIMD formulation; scalar-only |
| mask+sign | **no** | rejected on 17.98% measured sparsity |

The scalar reference travels with them and stays the correctness gate.

## The arms

`bench/ternary_gemv/dispatch.h`. Capabilities are read from the OS
(`sysctlbyname` on macOS, `getauxval` on Linux) and **never from `#ifdef`** —
`caps.c` carries the reason: a binary built with `+i8mm` on a CPU without it does
not fall back, it takes `SIGILL`; and this repo already has the mirror-image
incident where **Rosetta executes AVX2 without advertising it** and an entire
x86 suite passed without running one AVX2 kernel.

| arm | instruction | M1 | Neoverse V2 |
|---|---|---|---|
| `ARM_REF` | portable C | yes | yes |
| `ARM_DOTPROD` | `sdot` | **yes** | yes |
| `ARM_I8MM` | `smmla` | **no** (`FEAT_I8MM=0`) | **yes** |

`make axion` builds `-mcpu=neoverse-v2` with both arms — `-mcpu`, not
`-march=native`, so the binary states the ISA it was built for instead of
inheriting whatever the build host happened to be.

---

## THE PREDICTION  `[DERIVED]`, made before the run

`[PAPER]` fucina names `i8mm`/`smmla` as the fix for ARM's per-instruction
density gap against `vpdpbusd`. Worked through for a **decode GEMV**, that is
**not** what the instruction does.

`smmla` multiplies a 2×8 int8 matrix by a 2×8 int8 matrix transposed into a 2×2
int32 accumulator: **32 MACs per instruction against `sdot`'s 16**. That 2× is
only real with **two activation columns**. A decode GEMV has one, so the second
output column duplicates the first and **half the lanes are wasted — 16 useful
MACs per instruction, exactly `sdot`'s rate.**

On the layout the `sdot` kernel already uses it is worse than parity, because
assembling the A operand costs four `vcombine`s per 64 weights:

| | ops | weights | rows | **ops / weight** |
|---|---|---|---|---|
| `sdot` | 1 load + 1 and + 1 shr + 2 sdot = **5** | 32 | 1 | **0.156** |
| `smmla`, row layout | 2 load + 2 and + 2 shr + 4 vcombine + 4 smmla = **14** | 64 | 2 | **0.219** |
| **`smmla`, row-PAIR layout** | 1 load + 1 and + 1 shr + 2 smmla = **5** | 32 | **2** | **0.078** |

**`[DERIVED]` P1 — naive `smmla` on the existing layout is ~1.4× WORSE than
`sdot` for GEMV, not better.**

**`[DERIVED]` P2 — with a row-pair-interleaved layout it is ~2× better**, because
one 16-byte load then already carries row *r* in the low half and row *r+1* in
the high half:

```
byte i     (i < 8) = row_r  k[i] | row_r  k[i+8] << 4
byte 8 + i (i < 8) = row_s  k[i] | row_s  k[i+8] << 4
```

`and 0xF` is an `smmla` A operand with **no shuffling at all**; `shr 4` is the
k+8 block. The B operands are the activations duplicated into both halves, they
do not depend on the row, and they are built **once per GEMV** — the same
cross-row hoist the `sdot` arm already uses for `sum(x)`.

**This is not a new representation.** Same codes, same nine levels, same
**4.125 bpw** — a pair block is `[2·payload codes][row r scales][row s scales]`
and is exactly `2 × row_bytes`. `ternary_repack_pairs()` is a pure permutation.

**`[DERIVED]` P3 — `i8mm` is the wrong tool for decode and the right tool for
prefill.** With ≥2 activation columns both `smmla` lanes carry real work and the
2× is genuine. If P1/P2 hold, the engine's conclusion is *`sdot` for decode,
`smmla` for the batched prefill product* — not one kernel for both.

### The prediction already has a correctness gate, and it passed on a CPU with no i8mm

`[MEASURED]` `bench/ternary_gemv/test_pair.c` scores the repacked bytes with
`tgemv_pair_ref`, which walks them in **exactly the lane order `smmla` would**,
and compares against the T0 oracle:

```
cpu=Apple M1  neon=1 dotprod=1 i8mm=0
  [ 3072 x  1024]  pair layout vs T0 oracle: max 0.000e+00  rel_l2 0.000e+00  OK
  [ 1024 x  3072]  ... OK      [ 2048 x  1024]  ... OK
  [ 1024 x  2048]  ... OK      [ 1024 x  1024]  ... OK
PASS -- the i8mm layout is validated on a machine without i8mm
```

`[UNKNOWN]` **The `smmla` kernel itself has never executed.** It compiles; its op
counts are derivable; nothing in `kernels_i8mm.c` may be quoted as a measurement
until Axion runs it.

---

## What Axion must measure

Same shapes, same thread counts, same `--gguf` mode against the same two real
checkpoints, so the comparison is against **production** kernels and not
synthetic ones:

```
make axion
./ternary_gemv --gguf q-Q3_K_M.gguf 120      # Q3_K, Q4_K, Q6_K + T1/T3/T3-pair
./ternary_gemv --gguf q-Q4_K_M.gguf 120
./ternary_gemv 120                            # synthetic sweep + representation table
./test_pair                                   # layout gate, must PASS first
```

Report per shape: **bpw · ns/GEMV · 1/2/4-thread scaling · effective GB/s**, and
name the kernel that ran on every line.

### The questions this answers, and the one it does not

1. `[UNKNOWN]` Does `sdot` on Neoverse V2 behave like `sdot` on M1? The Mac's
   ~2.1× is against **our** Q4_K kernel; the ratio is a property of both sides.
2. `[UNKNOWN]` **P1/P2/P3.** If the pair layout is not ~2× the `sdot` arm, the
   op-count model above is wrong and the note says why.
3. `[UNKNOWN]` **Does `Q3_K` stay 3.4× slower than `Q4_K` there?** The Mac gap is
   partly ingot's unoptimised f32-activation Q3_K kernel, and that kernel is the
   same code on both machines — so a *different* ratio on Axion would isolate
   how much of the gap is the machine and how much is the kernel.
4. **NOT answered: anything about quality.** Gate A is unchanged by every number
   this produces.

`[HYPOTHESIS]` `[PAPER]` predicts x86-VNNI is the better target still (4.8× per
plane vs ARM's 2.1×). **No x86 arm is written.** If Axion disappoints, that is
the next thing to build, not a reason to close Gate B.
