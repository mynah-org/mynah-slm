# K5 — runtime ISA dispatch for our kernels, so a portable binary is a fast one

Status: **DONE 2026-10-08** (landed in two commits: core dispatch, then the sgemm.c change; real-silicon probes UNVALIDATED)

Item: `PLAN.md` §0 K5. Lineage: mynah-tts `src/qmat.c` (`qmat_x86_probe`
~:243, `qmat_u8_level_uncached` ~:311 — "clamped down, never up", the
verify-on-first-use gate ~:1197), `src/sgemm_rt.c`, its
`.work/linux-build-and-dispatch.md` and `.work/x86-kernel-tiers.md`; ingot's
`src/cpu.c` probe (CPUID+XGETBV, getauxval, sysctl). Map:
[`sibling-port-map.md`](sibling-port-map.md) row 8.

## Task

Every SIMD kernel we own is selected at COMPILE time (`#if defined(__AVX2__)`
and friends). A binary built for a baseline ISA therefore runs the scalar
kernels on every machine, however capable. Make the portable build select
AVX2 / AVX-512 / AVX-512 VNNI (x86) and NEON dotprod (arm64) at run time, with
an environment override that can only narrow, a verify-on-first-use against
the scalar twin, and a printable dispatch report.

## Question

Can one binary built with `-march=x86-64-v2` (or `armv8-a`) run exactly the
kernels a `-march=native` build runs on the same machine — proven in-process,
not inferred from the build flags?

## Known facts

- `.github/workflows/release.yml` ships `-march=x86-64-v2` (portable) and
  `-march=x86-64-v3` builds for Linux x86_64, `armv8.2-a+dotprod` for Linux
  aarch64, `-mcpu=apple-m1` for macOS. On the v2 tarball today: Q4_K f32
  matvec scalar, int8 path absent (`--fast` silently does nothing), attention
  dot/axpy scalar, bf16/q8/q4 KV accessors scalar (docs/perf.md: scalar KV
  accessors measured SLOWER than the f32 cache), own sgemm scalar.
- ISA-specific code we own: `src/qmat.c` (Q4_K f32: NEON/AVX2/scalar; int8
  Q4_K/Q8_0/Q6_K: NEON-SDOT/AVX-512-VNNI/AVX2 + scalar twins), `src/kernels.c`
  (attention `dot_f32`/`axpy_f32`: NEON/AVX2/scalar), `src/kvcache.c` (fused
  bf16/q8/q4 KV dot/axpy: NEON/AVX2/scalar — this is the attention inner loop
  for the DEFAULT bf16 cache, so it is in scope with the kernels.c pair),
  `src/sgemm.c` (NEON/AVX-512/AVX2/scalar micro-kernels).
- **ingot is not ours to change** (read-only here). Its kernels gate on
  `INGOT_COMPILED_*` and are built with the same `ARCH_FLAGS`, so on a v2 build
  ingot's Q6_K / Q8_0 / Q5_K kernels are scalar too. That is out of reach of
  this item and is reported, not fixed (see Conclusion).
- Levels on x86 form a ladder for OUR kernels: scalar < avx2 (AVX2+FMA+F16C)
  < avx512 (F+BW+VL+DQ) < avx512vnni. mynah-tts's warning applies: AVX-VNNI
  (VEX) is NOT on that ladder (Ice Lake server has EVEX VNNI and no VEX VNNI);
  we have no VEX-VNNI kernel, so it is probed and reported, never selected.
  arm64: scalar < neon < neon-dotprod. i8mm / bf16 / SVE are probed and
  reported as idle hardware: no kernel uses them (K6 rejected bf16).

## Design

**Per-ISA translation units, not target attributes.** The kernels already
exist as `#if __AVX2__ / __ARM_NEON / else` code. Compiling the SAME source
once per ISA with that ISA's flags keeps every helper, every intrinsic and
every guard exactly as written and lets the compiler check each variant;
target attributes would have meant re-annotating ~40 static inline helpers
and rewriting every guard, the change most likely to break what already
works.

- `src/kern.h` — the TU contract: `MYNAH_SLM_KERN_ID` (which ISA this TU is),
  derived capability macros, the symbol-suffix macro, and the three
  function-pointer tables (`qmat`, `attn`, `sgemm`).
- `src/qmat_kern.c` — the matvec kernels moved out of `qmat.c` (now API,
  activation prep and dispatch only). TUs: x86 scalar/avx2/avx512vnni,
  arm64 scalar/neon/neon_dotprod.
- `src/attn_kern.c` — the attention head loops with the f32 dot/axpy (from
  `kernels.c`) and the fused KV accessors (from `kvcache.c`), dispatched once
  per head instead of once per position. TUs: x86 scalar/avx2, arm64
  scalar/neon.
- `src/sgemm.c` — compiled twice-over in place (API compile + kernel TUs
  scalar/avx2/avx512 or scalar/neon), the smallest change to a file another
  lane owns: a forced-ISA prefix on its ISA chain, `#if` around the API half,
  and the planner+kernels renamed per TU. Numerics untouched.
- `src/isa.c` — CPUID+XGETBV / getauxval / sysctl probe, `MYNAH_SLM_ISA`
  (scalar, neon, dotprod, avx2, avx512, avx512vnni) clamped DOWN to what the
  CPU has (never up; an ungrantable request is reported as such), one
  `pthread_once` resolution per process, per-family verify-on-first-use
  against the scalar TU (memcmp for the int8 contract, a tolerance for the
  f32 kernels whose scalar twin rounds differently), fall back one level and
  say so on a failure, and `mynah-slm --dispatch`.
- Default local build stays `-march=native`; its numerics do not move (the
  same code paths resolve on the same machine — asserted below).

## Gates (acceptance)

1. `make ARCH_FLAGS=-march=x86-64-v2 test`: `--dispatch` shows qmat
   avx512vnni, attn avx2, sgemm avx512 on this VM, and every family's verify
   PASS; the int8 twin contract passes at every level ≤ detected.
2. `MYNAH_SLM_ISA=avx2` and `=scalar` narrow every family; a request above
   the CPU (none possible here except via a fake) is clamped and flagged.
3. aarch64 cross build (`armv8-a`) under `qemu-aarch64 -cpu max` resolves
   neon-dotprod for qmat; under `-cpu cortex-a53` resolves neon and `--fast`
   reports int8 unavailable. Twin tests pass in both.
4. Native build: `make test` green, results bit-identical to before for the
   resolved paths (test_kernels / test_sgemm memcmp gates unchanged).

## Unknowns

- Real-hardware dispatch on M1 (sysctl path), Neoverse V2, Zen 4 (AVX-512 +
  VNNI + AVX512_BF16 + AVX-VNNI), an AVX2-only host.
- Whether the indirect call per matvec chunk / attention head is measurable
  (cost model: ~200 regions x ~16 chunks per token = ~3k indirect calls, a few
  microseconds per token against ~27 ms).

## Files inspected

`src/qmat.c`, `src/kernels.c`, `src/kvcache.c`, `src/sgemm.c`, `src/threads.c`
(probe precedent), `cli/main.c`, `Makefile`, `.github/workflows/release.yml`,
`.github/workflows/build.yml`, `third_party/ingot/src/cpu.c`, mynah-tts
`src/qmat.c`, `src/sgemm_rt.c`, `.work/linux-build-and-dispatch.md`.

## Evidence

All on the 4-vCPU Cascade Lake-class cloud VM (AVX-512F/BW/VL/DQ + VNNI, no
AVX512_BF16, no AMX), gcc 13.3 and clang, plus `qemu-aarch64` for arm64.

### What resolves, per build and CPU (`mynah-slm --dispatch`)

| build | CPU | qmat | attn | sgemm | int8 | ingot's own dispatch | exit |
|---|---|---|---|---|---|---|---|
| `-march=native` (x86) | Cascade Lake VM | avx512vnni | avx2 | avx512 | avx512-vnni | avx2/avx512/vnni on | 0 |
| `-march=x86-64-v2` (x86, the release's portable flags) | same | **avx512vnni** | **avx2** | **avx512** | avx512-vnni | **all off: flagged "its kernels run SCALAR"** | 0 |
| `make dist ARCH_FLAGS="-march=x86-64-v2 -mtune=generic"`, the unpacked tarball | same | avx512vnni | avx2 | avx512 | avx512-vnni | all off, flagged | 0 |
| v2 binary, `MYNAH_SLM_ISA=avx2` | same | avx2 | avx2 | avx2 | avx2 | — | 0 |
| v2 binary, `MYNAH_SLM_ISA=scalar MYNAH_SLM_INT8=1` | same | scalar | scalar | scalar | none, **"REQUESTED BUT UNAVAILABLE"** | — | 1 |
| any x86 binary, `MYNAH_SLM_ISA=neon` | same | **scalar** (was: unchanged = the widest level — failed OPEN; review R3) | scalar | scalar | none | — | 1, "UNKNOWN on this architecture: fell back to SCALAR" + one stderr warning |
| aarch64 `-march=armv8-a` (cross) | qemu `-cpu max` | **neon_dotprod** | neon | neon | neon-dotprod | dotprod off, flagged | 0 |
| same binary | qemu `-cpu cortex-a53` | **neon** | neon | neon | none | — | 0 |
| same, `MYNAH_SLM_INT8=1` | cortex-a53 | neon | | | "REQUESTED BUT UNAVAILABLE" | — | 1 |
| same, `MYNAH_SLM_ISA=dotprod` | cortex-a53 | neon | | | | — | 1, "REQUEST ABOVE THE CPU, clamped down" |
| same, `MYNAH_SLM_ISA=neon` | `-cpu max` | neon | neon | neon | none | — | 0 |
| aarch64 `-march=armv8.2-a+dotprod` (the release's arm flags UNTIL review R7) | `-cpu max` | neon_dotprod | neon | neon | neon-dotprod | dotprod on | 0 — but `-cpu cortex-a53`: **SIGILL** before dispatch (gcc inlines LSE atomics into threads.o and sgemm.o at armv8.1+); the release now builds `-march=armv8-a` |

### Numerics did not move

`numdump` (scratch tool: hashes the output bytes of every dispatched stage on
fixed data — Q4_K f32, Q4_K/Q6_K/Q8_0 int8 matvec at 1024x1024 on the pool,
f32 attention, packed-KV attention for bf16/q8/q4/f32/fp8 at n_kv = 300 plus a
single `kv_dot_k`, own sgemm NT and NN at 256x128x1024 on 4 threads):

| comparison | result |
|---|---|
| pre-K5 tree (`8bca897`, compile-time ISA, `-march=native`) vs K5 native build | **IDENTICAL, 12/12 stages** |
| K5 native build vs K5 `-march=x86-64-v2` build | **IDENTICAL, 12/12 stages** |

So the default build is byte-for-byte what it was, and the portable build is
byte-for-byte the native build on this machine.

### Gates

| check | result |
|---|---|
| `tests/test_isa` (new): probe vs flags; every `MYNAH_SLM_ISA` level → that ceiling, clamped never raised; each family takes the BEST table it is allowed and none above; unknown / other-arch names refused; verify passes every resolved table at every level; verify **rejects** a wrong f32 kernel (rel 2.0) and an int8 kernel **one ulp** off its twin; at every level, own sgemm vs reference < 1e-5 and bf16-KV attention vs scalar < 1e-5, both 4-thread == 1-thread by memcmp | PASS x86 native, x86 v2, aarch64 qemu max, aarch64 qemu cortex-a53 |
| `tests/test_kernels`: the int8 contract (memcmp vs twin, grouping, 4 threads) now run at EVERY level the CPU allows — x86: avx512vnni, avx512, avx2, scalar | PASS (gcc, clang, v2, qemu max, qemu cortex-a53) |
| `make test` native, `ARCH_FLAGS=-march=x86-64-v2`, `CC=clang` | PASS |
| `make warnings` gcc and clang (lib incl. every kernel TU, `ARCH_FLAGS=` empty, -Werror) | PASS |
| `make check-x86 CC=clang X86_TARGET=x86_64-linux-gnu` (every x86 kernel TU incl. AVX-512, with its own flags) | PASS |
| ThreadSanitizer: `test_isa`, `test_kernels`, `test_sgemm` (`BLAS=none`) | 0 reports |
| `make ubsan` | PASS, no runtime errors |

### Dispatch overhead — CLOUD-SPECIFIC, revalidate on target

`tests/bench_qmat k5 21 1,4`: A = the resolved table's Q4_K row kernel called
directly, B = the same kernel through `mynah_slm_matvec` (pthread_once fast
path + table lookup + type switch), 1024x1024 (the smallest real shape),
interleaved, 21 rounds, load average 2.0-2.6:

| path | thr | direct ms median (min..max) | dispatched ms median (min..max) | direct/dispatched | dispatched wins |
|---|---|---|---|---|---|
| Q4_K f32 | 1 | 0.251 (0.197..0.377) | 0.249 (0.202..0.467) | 1.01x | 9/21 |
| Q4_K int8 | 1 | 0.106 (0.081..0.138) | 0.105 (0.077..0.150) | 1.01x | 12/21 |
| Q4_K f32 | 4 | 0.205 (0.131..0.334) | 0.201 (0.136..0.377) | 1.02x | 11/21 |
| Q4_K int8 | 4 | 0.049 (0.039..0.119) | 0.055 (0.040..0.085) | 0.90x | 7/21 |

No direction is stable in any row: the overhead is below this VM's noise, as
the cost model said (~3k indirect calls per token, nanoseconds each).

## Conclusion

**PROMOTE** — gate 1-4 all met here.

- A baseline build now runs the same kernels as `-march=native` on the same
  CPU, proven in-process by `--dispatch` and byte-for-byte by the stage
  hashes; the default build's numerics are unchanged.
- `MYNAH_SLM_ISA` only narrows; a request above the CPU or for another
  architecture is clamped/ignored AND reported with exit 1, so a benchmark
  script can refuse to run on it (engineering-method: "an explicit request
  is a request, not a capability").
- Every vector table is verified against the scalar table before use; the
  gate demonstrably rejects a one-ulp int8 deviation.
- `--fast` on a CPU without vector int8 is now visible ("REQUESTED BUT
  UNAVAILABLE", exit 1) instead of silently doing nothing.

**Found, not fixed (not ours to fix): ingot's kernels on a portable build are
scalar.** ingot gates its SIMD on `INGOT_COMPILED_*`, i.e. on the same
`ARCH_FLAGS`, so the `x86-64-v2` tarball runs ingot's Q6_K / Q8_0 / Q5_K
f32-activation matvecs — the Q6_K LM head of every Q4_K_M file — without
AVX2 on every machine. `--dispatch` flags it on x86 and arm64. The fix belongs
upstream in ingot (per-ISA compilation of its `kernels.c`, the same pattern as
here); until then the release's `x86-64-v3` tarball is the one to recommend
for AVX2 hosts, and `--fast` covers the Q6_K head with our own dispatched
kernel (K4).

WHAT CHANGED: `src/kern.h`, `src/isa.{c,h}`, `src/isa_verify.c`,
`src/qmat_kern.c` (kernels moved out of `qmat.c`), `src/attn_kern.c` (moved
out of `kernels.c` / `kvcache.c`), the Makefile kernel-TU rules and
`make dispatch`, `mynah-slm --dispatch`, eager resolution at model load,
`tests/test_isa.c`, per-level int8 contracts in `tests/test_kernels.c`,
`tests/bench_qmat k5`, the release workflow prints `--dispatch` for the
packaged binary; then, as a separate commit, the minimal `src/sgemm.c`
dual-mode change.
WHAT PATH ACTUALLY RAN: the table above, per build and CPU.
WHAT WAS MEASURED: dispatch overhead (below noise); numerics (identical).
WHAT REMAINS UNKNOWN: real-silicon probes on M1 (sysctl), Neoverse V2
(getauxval on hardware, not qemu), Zen 4 (AVX-512 + VNNI + AVX512_BF16 +
AVX-VNNI all present), an AVX2-only host, and an AVX-512-without-VNNI host
(Skylake-SP: qmat must resolve avx2, sgemm avx512); decode tok/s of the
portable tarball vs the native build with a model.

## Next action — downstream

On each host (M1, x86 Linux AVX2-only if available, x86 AVX-512, Neoverse V2),
weights LOCAL:

```
make clean && make -j && ./mynah-slm --dispatch && make test test-parity
make clean && make -j ARCH_FLAGS="-march=x86-64-v2 -mtune=generic" && ./mynah-slm --dispatch   # x86: same tables as native?
make clean && make -j ARCH_FLAGS=-march=armv8-a && ./mynah-slm --dispatch                       # arm64: neon_dotprod on V2 / M1?
MYNAH_SLM_ISA=avx2 ./mynah-slm --dispatch; MYNAH_SLM_ISA=scalar ./mynah-slm --dispatch          # narrowing on silicon
# decode tok/s, portable vs native binary, same model/prompt/seed/threads, interleaved, >= 5 runs:
./mynah-slm run -m models-local/Qwen3-0.6B-Q4_K_M.gguf -p "Racconta una storia breve su un faro." -n 128 --temp 0 -t <n>
tests/bench_qmat k5 21 1,<ncores>      # dispatch overhead on the target
```

Prediction: the portable x86 tarball's decode within 5% of native on Q4_K
tensors, and measurably slower on the Q6_K head until ingot is compiled per
ISA upstream (the `--dispatch` ingot line says so).
